"""Per-defect depth from the LED frames: a physics-informed neural field (docs/twin/PHYSICS_PINN.md 2).

estimate_depth(frames, meta, box_mm, material, thickness_mm) fits a height field h(u, v) (mm) over
a crop round one detected dent or pit so that the forward model (trinetra.physics.optics) of
h reproduces the observed ambient-subtracted LED frames:

    h_theta(u, v) = D(x, y) * s_h * MLP([sin 2 pi B x, cos 2 pi B x])     x, y in [-1, 1] over the crop
    D(x, y)       = (1 - x^2)(1 - y^2)                                    hard BC: h = 0 on the crop border
    S_g           = k e^{a_g} [m(u, v) L_g(n(h)) + e^{e_g} Env_g(n(h))] + b_g             (FS)
    m(u, v)       = exp(s_m MLP_2(...))   a smooth material factor shared by all LED groups
    loss          = mean rho((S_g - O_g) / sigma) + lam_c |Hess h|^2 + lam_s relu(h - tau)^2 + priors

- L_g is optics.shade (direct LED light, Principled approximation); a_g a per-group exposure
  correction (LED jitter, exposure); Env_g the hood's one-bounce inter-reflection seen in the
  plate (optics.hood_env / env_radiance, ring only) with a learnable gain e_g; b_g >= 0 a small
  offset (reports/physics_pinn.md 1.2).
- base, roughness (and metallic for mill scale) are learnable per crop with priors from the
  material class (DESIGN_STEEL B); they are fitted first on the flat crop border with h = 0.
- m(u, v) absorbs local material changes (a burnished dent floor, mottle, a dark pit): in the
  dark-field tail GGX gives D ~ alpha^2 / sin^4(theta_h), so roughness and albedo act as one
  per-pixel factor common to every group, while the ratios between groups depend on the normal
  only. With L/R the slope along L-R is then fixed by the ratio and the rest by the BC; with
  3+ sectors the problem is over-determined (reports/physics_pinn.md 3.3).
- rho is the Cauchy loss; sigma^2 = sensor noise variance + (eps_rel O)^2 + eps_abs^2 (model error).
- Optimised with Adam, then optionally L-BFGS, on CPU. n_seeds fits with different Fourier
  features give the ensemble spread; depth_ci_mm combines it with a systematic term calibrated
  on the twin depth coupons (SYS_REL_TABLE, SYS_ABS).

method='grid' replaces the MLP by a free height lattice (same physics, loss and BC): the
'is the network needed?' ablation. Classical baselines are in physics/baselines.py.
torch is imported lazily: the module imports with numpy only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from . import optics as O

# Material class priors (DESIGN_STEEL B table; luma of the base colour, linear). The twin draws
# each plate's values inside these ranges; on hardware they are the starting point of the fit.
MATERIAL_PRIORS: dict[str, dict[str, float]] = {
    "ms_hot_rolled": {"base": 0.13, "rough": 0.62, "metal": 0.5, "aniso": 0.08},
    "ms_cold_rolled": {"base": 0.53, "rough": 0.25, "metal": 1.0, "aniso": 0.3},
    "ss_brushed": {"base": 0.60, "rough": 0.20, "metal": 1.0, "aniso": 0.55},
    "ss_2b": {"base": 0.55, "rough": 0.32, "metal": 1.0, "aniso": 0.12},
    "galvanised": {"base": 0.62, "rough": 0.30, "metal": 1.0, "aniso": 0.10},
    "aluminium": {"base": 0.90, "rough": 0.26, "metal": 1.0, "aniso": 0.30},
}
DEFAULT_PRIOR = {"base": 0.5, "rough": 0.35, "metal": 1.0, "aniso": 0.1}

# Uncertainty: depth_ci_mm = sqrt((Z * ens_sd)^2 + (SYS_REL * depth)^2 + SYS_ABS^2), a ~90 % interval.
# SYS_REL is the 90th percentile of the twin depth-coupon errors (6 materials x 16 dents and pits,
# Blender per-LED renders + sensor noise) per class and number of LED groups (reports/physics_pinn.md
# section 5); it was then tested out of sample on the pilot plates.
Z90 = 1.645
ENV_UNIT = 0.01            # FS: scale of the environment-term parameters (O(1) for Adam)
SYS_REL = 0.46                  # dents, L/R (the device as built)
SYS_ABS = 0.02
SYS_REL_TABLE = {("dent", 2): 0.461, ("pit", 2): 2.152, ("dent", 3): 0.232, ("pit", 3): 1.115,
                 ("dent", 6): 0.103, ("pit", 6): 1.061}


def sys_rel(kind: str, n_groups: int) -> float:
    """Calibrated relative systematic error for a class and LED-group count (the nearest calibrated count below)."""
    keys = sorted(n for (k, n) in SYS_REL_TABLE if k == kind and n <= max(2, int(n_groups)))
    return SYS_REL_TABLE[(kind, keys[-1])] if keys else SYS_REL

DEFAULTS: dict[str, Any] = {
    "max_px": 48,            # working lattice at most this many px across (larger crops are block-averaged)
    "ss": 2,                 # supersampling of the render lattice when the crop is not block-averaged
    "margin_frac": 0.45,     # crop = box grown by this fraction of its size a side ...
    "margin_min_mm": 0.8,    # ... and at least this
    "n_fourier": 48,
    "fourier_sigma": 1.2,    # B ~ N(0, sigma^2 (crop / defect size)) cycles per half-crop
    "hidden": 64,
    "layers": 2,
    "iters_material": 80,
    "iters": 220,
    "lr": 4e-3,
    "lbfgs_iters": 0,
    "n_seeds": 3,
    "eps_rel": 0.06,         # model error, relative (validation: reports/physics_pinn.md 1)
    "eps_abs": 4e-4,         # FS
    "lam_curv": 2e-4,
    "material_field": True,  # learn m(u, v) (shared multiplicative material factor)
    "m_scale": 1.0,          # log m = m_scale * output
    "lam_m": 0.05,           # prior weight on (log m)^2 (sigma ~ 1 / sqrt(2 lam_m) in log units)
    "lam_m_smooth": 2e-3,    # smoothness of log m (mm^2 weighted gradient)
    "lam_sign": 2.0,
    "sign_tau_mm": 0.05,     # dents and pits are mostly below the surface; a rim up to tau is free
    "psf_sigma_mm": O.PSF_SIGMA_MM,
    "valid_reflection": True,   # Blender's normal-map clamp (twin); False for real hardware
    "shadows": False,           # cast shadows (real hardware); the twin's normal maps have none
    "converge_tol": 0.03,       # relative change of the max depth over the last 20 % of iterations
    "resid_max": 3.0,           # robust normalised residual RMS above this: not converged
}


@dataclass
class DepthResult:
    max_depth_mm: float | None
    depth_ci_mm: float | None
    volume_mm3: float | None
    profile: dict
    residual: float
    converged: bool
    runtime_s: float
    method: str = "pinn"
    members: list[float] = field(default_factory=list)
    rel_thickness: float | None = None
    material_fit: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    height_map: np.ndarray | None = None
    crop: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """Record-friendly dict (no arrays)."""
        r = lambda x, n=3: None if x is None else round(float(x), n)
        return {"method": self.method, "max_depth_mm": r(self.max_depth_mm), "depth_ci_mm": r(self.depth_ci_mm),
                "volume_mm3": r(self.volume_mm3, 4), "rel_thickness": r(self.rel_thickness, 4),
                "converged": bool(self.converged), "residual": r(self.residual), "runtime_s": r(self.runtime_s, 2),
                "members": [r(m) for m in self.members], "notes": list(self.notes)}


# ----------------------------------------------------------------------------- crop and lattice

def _torch():
    import torch
    return torch


def grain_angle_deg(img: np.ndarray) -> float:
    """Plate-frame grain direction (deg from +u towards +v) of a plate-aligned image: the
    streaks run along the grain, so it is the structure tensor's minor axis."""
    y = np.asarray(img, float)
    gy, gx = np.gradient(y)
    jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    theta_grad = 0.5 * math.degrees(math.atan2(2 * jxy, jxx - jyy))
    return (theta_grad + 90.0) % 180.0


def plate_grain_deg(img_grid: np.ndarray, H_grid_plate, plate_size_mm=None, hp_sigma_px: float = 4.0) -> float:
    """Grain direction (plate deg) of a whole rectified plate image (e.g. Lc + Rc): structure
    tensor of the high-passed image inside the plate. The app passes it as meta['grain_deg']."""
    import cv2
    y = np.asarray(img_grid, np.float32)
    hp = y - cv2.GaussianBlur(y, (0, 0), hp_sigma_px)
    gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=3)
    if plate_size_mm is not None:
        hh, ww = y.shape
        ys, xs = np.mgrid[0:hh, 0:ww]
        uv = O.grid_to_plate(H_grid_plate, np.stack([xs.ravel(), ys.ravel()], 1)).reshape(hh, ww, 2)
        W, Hh = plate_size_mm
        m = (uv[..., 0] > 2) & (uv[..., 0] < W - 2) & (uv[..., 1] > 2) & (uv[..., 1] < Hh - 2)
        lim = np.percentile(np.abs(hp[m]), 99)
        m &= np.abs(hp) < lim                           # defects and edges out
        gx, gy = gx[m], gy[m]
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    th = 0.5 * math.atan2(2 * jxy, jxx - jyy) + math.pi / 2       # grid angle of the streaks
    A, _ = O.grid_affine(H_grid_plate)
    d = A @ np.array([math.cos(th), math.sin(th)])
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


@dataclass
class Crop:
    obs: dict[str, np.ndarray]          # plate-aligned observed frames (Nv, Nu), FS
    var: dict[str, np.ndarray] | None   # noise variance (FS^2) or None
    valid: np.ndarray                   # (Nv, Nu) bool: inside the plate
    u0: float                           # plate mm of pixel (0, 0) centre
    v0: float
    s: float                            # working spacing mm
    f: int                              # block-average factor applied to the grid
    box_mm: list[float]
    border: np.ndarray                  # (Nv, Nu) bool: flat ring used for the material fit


def make_crop(frames: Mapping[str, np.ndarray], meta: Mapping[str, Any], box_mm: Sequence[float],
              cfg: Mapping[str, Any], groups: Sequence[str], var: Mapping[str, np.ndarray] | None = None) -> Crop:
    """Plate-aligned crop round box_mm (grown by the margin, kept inside the plate), sampled from
    the rectified grid frames at exact grid pixel centres, block-averaged to <= max_px."""
    s0 = float(meta.get("mm_per_px", 0.08))
    H = np.asarray(meta["H_grid_plate"], float)
    u0, v0, u1, v1 = (float(x) for x in box_mm)
    size = max(u1 - u0, v1 - v0, s0)
    m = max(float(cfg["margin_min_mm"]), float(cfg["margin_frac"]) * size)
    W, Hp = (float(x) for x in meta.get("plate_size_mm", (1e9, 1e9)))
    cu0, cv0 = max(0.2, u0 - m), max(0.2, v0 - m)
    cu1, cv1 = min(W - 0.2, u1 + m), min(Hp - 0.2, v1 + m)
    # snap the origin to a grid pixel centre
    g0 = O.plate_to_grid_px(H, [[cu0, cv0]])[0]
    g0 = np.round(g0)
    o = O.grid_to_plate(H, g0[None])[0]
    nu = int(math.floor((cu1 - o[0]) / s0)) + 1
    nv = int(math.floor((cv1 - o[1]) / s0)) + 1
    f = max(1, int(math.ceil(max(nu, nv) / float(cfg["max_px"]))))
    nu, nv = (nu // f) * f, (nv // f) * f
    U = o[0] + s0 * np.arange(nu)
    V = o[1] + s0 * np.arange(nv)
    UU, VV = np.meshgrid(U, V)
    g = O.plate_to_grid_px(H, np.stack([UU.ravel(), VV.ravel()], 1))
    gx = np.rint(g[:, 0]).astype(int).reshape(nv, nu)
    gy = np.rint(g[:, 1]).astype(int).reshape(nv, nu)
    any_f = next(iter(frames.values()))
    hh, ww = any_f.shape[:2]
    inb = (gx >= 0) & (gx < ww) & (gy >= 0) & (gy < hh)
    gxc, gyc = np.clip(gx, 0, ww - 1), np.clip(gy, 0, hh - 1)

    def pool(a):
        if f == 1:
            return a
        return a.reshape(nv // f, f, nu // f, f).mean(axis=(1, 3))

    obs = {gname: pool(np.where(inb, np.asarray(frames[gname], np.float64)[gyc, gxc], 0.0)) for gname in groups}
    vv = None
    if var is not None:
        vv = {gname: pool(np.where(inb, np.asarray(var[gname], np.float64)[gyc, gxc], 0.0)) / (f * f)
              for gname in groups if gname in var}
    valid = pool(inb.astype(float)) > 0.999
    s = s0 * f
    u0c, v0c = o[0] + 0.5 * s0 * (f - 1), o[1] + 0.5 * s0 * (f - 1)
    Nv, Nu = nv // f, nu // f
    Uc = u0c + s * np.arange(Nu)
    Vc = v0c + s * np.arange(Nv)
    inbox = ((Uc[None, :] >= u0 - 0.25 * m) & (Uc[None, :] <= u1 + 0.25 * m)
             & (Vc[:, None] >= v0 - 0.25 * m) & (Vc[:, None] <= v1 + 0.25 * m))
    border = valid & ~inbox
    return Crop(obs, vv, valid, float(u0c), float(v0c), float(s), f, [u0, v0, u1, v1], border)


class Renderer:
    """Differentiable crop renderer: height on the fine lattice -> predicted frames (FS) per group."""

    def __init__(self, crop: Crop, meta: Mapping[str, Any], groups: Sequence[str], cfg: Mapping[str, Any],
                 tangent: np.ndarray, dtype=None):
        torch = _torch()
        self.torch = torch
        self.dtype = dtype or torch.float32
        self.cfg = cfg
        ss = int(cfg["ss"]) if crop.f == 1 else 1
        self.ss = ss
        Nv, Nu = next(iter(crop.obs.values())).shape
        self.shape = (Nv, Nu)
        self.ds = crop.s / ss
        # fine lattice: ss x ss sub-samples centred on each working pixel
        off = (np.arange(ss) + 0.5) / ss - 0.5
        uf = (crop.u0 + crop.s * (np.arange(Nu)[:, None] + off[None, :])).ravel()
        vf = (crop.v0 + crop.s * (np.arange(Nv)[:, None] + off[None, :])).ravel()
        self.uf, self.vf = uf, vf
        groups_ls = O.group_lights(meta["lights"], groups, "luma")
        missing = [g for g in groups if g not in groups_ls]
        if missing:
            raise ValueError(f"no lights for groups {missing}")
        lat = O.make_lattice(uf[0], vf[0], len(uf), len(vf), self.ds, self.ds, groups_ls,
                             np.asarray(meta["camera_centre"], float))
        self.lat = O.lattice_to(lat, dtype=self.dtype)
        self.groups = list(groups)
        self.k = float(meta["k"])
        self.tangent = torch.as_tensor(tangent, dtype=self.dtype)
        # normalised coordinates of the fine lattice (x along u, y along v), in [-1, 1] at the crop border
        Lu = crop.s * Nu / 2.0
        Lv = crop.s * Nv / 2.0
        cu = crop.u0 - crop.s / 2 + Lu
        cv = crop.v0 - crop.s / 2 + Lv
        X, Y = np.meshgrid((uf - cu) / Lu, (vf - cv) / Lv)
        self.xy = torch.as_tensor(np.stack([X, Y], -1), dtype=self.dtype)
        self.D = (1 - self.xy[..., 0] ** 2) * (1 - self.xy[..., 1] ** 2)
        # PSF (on the fine lattice) then ss x ss average pooling to the working lattice
        sig = math.sqrt(max(float(cfg["psf_sigma_mm"]) ** 2 - (crop.s / ss) ** 2 / 12.0 * (ss > 1), 0.0)) / self.ds
        k1 = O.gaussian_kernel1d(sig) if sig > 0.2 else np.array([1.0])
        self.k1 = torch.as_tensor(k1, dtype=self.dtype)
        self.V = self.lat.V
        # the hood's inter-reflection (ring in the hood): one env map per group at the crop centre
        self.env = None
        hood = meta.get("hood")
        if hood and cfg.get("hood_env", True) and meta.get("camera_R") is not None:
            pm = meta.get("env_material") or {}
            cu0 = crop.u0 + crop.s * (Nu - 1) / 2.0
            cv0 = crop.v0 + crop.s * (Nv - 1) / 2.0
            env = O.hood_env(groups_ls, hood, np.asarray(meta["camera_R"], float), np.asarray(meta["camera_t"], float),
                             O.plate_points(np.array([cu0, cv0])), base_luma=float(pm.get("base", 0.5)),
                             metallic=float(pm.get("metal", 1.0)), rough=float(pm.get("rough", 0.3)),
                             plate_size_mm=meta.get("plate_size_mm"), n=int(cfg.get("env_n", 48)),
                             lobe_k=float(cfg.get("env_lobe_k", 2.0)))
            self.env = {g: torch.as_tensor(env.maps[g], dtype=self.dtype) for g in groups}
            self.env_irr = {g: float(env.irradiance[g]) for g in groups}

    def blur_pool(self, img):
        torch = self.torch
        import torch.nn.functional as Fnn
        x = img[None, None]
        r = (len(self.k1) - 1) // 2
        if r > 0:
            x = Fnn.pad(x, (r, r, r, r), mode="replicate")
            x = Fnn.conv2d(x, self.k1.view(1, 1, 1, -1))
            x = Fnn.conv2d(x, self.k1.view(1, 1, -1, 1))
        if self.ss > 1:
            x = Fnn.avg_pool2d(x, self.ss)
        return x[0, 0]

    def normals(self, h):
        n = O.normals_from_height(h, self.ds, self.ds)
        if self.cfg.get("valid_reflection", True):
            n = O.valid_reflection_normals(n, self.V)
        return n

    def render(self, h, mat: O.Material, nuis: Mapping[str, Any], m=None):
        """Predicted frames {g: (Nv, Nu)} in FS for height h (fine lattice, mm) and material factor m."""
        torch = self.torch
        n = self.normals(h)
        shadow = None
        if self.cfg.get("shadows"):
            shadow = {g: O.horizon_shadow(h, self.lat.incident[g][0], self.ds, self.ds) for g in self.groups}
        rad = O.shade(n, self.lat, mat, self.groups, multiscatter=True, valid_reflection=False, shadow=shadow)
        out = {}
        for g in self.groups:
            a = nuis["log_gain"][g]
            direct = rad[g][..., 0] if m is None else rad[g][..., 0] * m
            if self.env is not None:
                envr = O.env_radiance(n, self.V, self.env[g], mat.base[0], mat.metallic, self.env_irr[g])
                direct = direct + torch.exp(nuis["env_gain"][g]) * envr
            img = self.k * torch.exp(a) * direct + nuis["env_b"][g]
            out[g] = self.blur_pool(img)
        return out


# ----------------------------------------------------------------------------- fields

class FourierField:
    """h(x, y) = D(x, y) * scale * MLP_1(gamma(x, y)) (hard zero BC on the crop border) and
    log m(x, y) = m_scale * MLP_2(gamma(x, y)) (the material factor; no BC, prior towards 0)."""

    def __init__(self, xy, D, n_fourier: int, sigma: float, hidden: int, layers: int, scale_mm: float, seed: int,
                 dtype, m_scale: float = 1.0, material_field: bool = True):
        torch = _torch()
        g = torch.Generator().manual_seed(int(seed))
        self.B = torch.randn(2, n_fourier, generator=g, dtype=dtype) * sigma
        feats = 2 * n_fourier
        mods = []
        d_in = feats
        for _ in range(layers):
            lin = torch.nn.Linear(d_in, hidden).to(dtype)
            with torch.no_grad():
                bound = math.sqrt(6.0 / d_in)
                lin.weight.uniform_(-bound, bound, generator=g)
                lin.bias.zero_()
            mods += [lin, torch.nn.SiLU()]
            d_in = hidden
        last = torch.nn.Linear(d_in, 2).to(dtype)
        with torch.no_grad():
            last.weight.uniform_(-1e-3, 1e-3, generator=g)
            last.bias.zero_()
        mods.append(last)
        self.net = torch.nn.Sequential(*mods)
        proj = 2 * math.pi * (xy @ self.B)
        self.feat = torch.cat([torch.sin(proj), torch.cos(proj)], -1)
        self.D = D
        self.scale = float(scale_mm)
        self.m_scale = float(m_scale)
        self.material_field = bool(material_field)

    def parameters(self):
        return list(self.net.parameters())

    def both(self):
        out = self.net(self.feat)
        h = self.D * self.scale * out[..., 0]
        logm = self.m_scale * out[..., 1] if self.material_field else None
        return h, logm

    def __call__(self):
        return self.both()[0]


class GridField:
    """h = D * scale * z and log m = w with free lattices z, w (the no-network ablation)."""

    def __init__(self, D, scale_mm: float, dtype, material_field: bool = True):
        torch = _torch()
        self.z = torch.zeros(D.shape, dtype=dtype, requires_grad=True)
        self.w = torch.zeros(D.shape, dtype=dtype, requires_grad=True)
        self.D = D
        self.scale = float(scale_mm)
        self.material_field = bool(material_field)

    def parameters(self):
        return [self.z, self.w] if self.material_field else [self.z]

    def both(self):
        return self.D * self.scale * self.z, (self.w if self.material_field else None)

    def __call__(self):
        return self.both()[0]


# ----------------------------------------------------------------------------- fitting

def _inv_sigmoid(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class Fit:
    """Material, nuisance and height parameters plus the loss of one crop."""

    def __init__(self, crop: Crop, rend: Renderer, prior: Mapping[str, float], cfg: Mapping[str, Any],
                 aniso: float):
        torch = _torch()
        self.torch = torch
        dt = rend.dtype
        self.crop, self.rend, self.cfg, self.prior = crop, rend, cfg, prior
        self.p_base = torch.tensor(math.log(prior["base"]), dtype=dt, requires_grad=True)
        self.p_rough = torch.tensor(_inv_sigmoid(prior["rough"]), dtype=dt, requires_grad=True)
        self.learn_metal = prior["metal"] < 0.999
        self.p_metal = torch.tensor(_inv_sigmoid(prior["metal"] if self.learn_metal else 0.5), dtype=dt,
                                    requires_grad=self.learn_metal)
        self.aniso = float(aniso)
        self.log_gain = {g: torch.zeros((), dtype=dt, requires_grad=True) for g in rend.groups}
        # environment term in units of ENV_UNIT FS: b = ENV_UNIT softplus(q_b) >= 0, c = ENV_UNIT q_c
        self.q_b = {g: torch.full((), -4.0, dtype=dt, requires_grad=True) for g in rend.groups}
        self.log_env = {g: torch.zeros((), dtype=dt, requires_grad=True) for g in rend.groups}
        self.obs = {g: torch.as_tensor(crop.obs[g], dtype=dt) for g in rend.groups}
        eps_rel, eps_abs = float(cfg["eps_rel"]), float(cfg["eps_abs"])
        self.sigma = {}
        for g in rend.groups:
            v = crop.var[g] if crop.var is not None and g in crop.var else np.zeros_like(crop.obs[g])
            s2 = np.maximum(v, 0) + (eps_rel * np.abs(crop.obs[g])) ** 2 + eps_abs ** 2
            self.sigma[g] = torch.as_tensor(np.sqrt(s2), dtype=dt)
        self.valid = torch.as_tensor(crop.valid, dtype=dt)
        self.border = torch.as_tensor(crop.border, dtype=dt)
        self.zero_h = torch.zeros(rend.D.shape, dtype=dt)

    def material(self):
        torch = self.torch
        metal = torch.sigmoid(self.p_metal) if self.learn_metal else 1.0 + 0.0 * self.p_metal
        return O.Material(torch.exp(self.p_base).reshape(1), torch.sigmoid(self.p_rough), metal, self.aniso,
                          self.rend.tangent)

    @property
    def env_b(self):
        F = self.torch.nn.functional
        return {g: ENV_UNIT * F.softplus(v) for g, v in self.q_b.items()}

    def nuis(self):
        return {"log_gain": self.log_gain, "env_b": self.env_b, "env_gain": self.log_env}

    def material_params(self):
        ps = [self.p_base, self.p_rough] + list(self.log_gain.values()) + list(self.q_b.values()) \
            + (list(self.log_env.values()) if self.rend.env is not None else [])
        return ps + ([self.p_metal] if self.learn_metal else [])

    def photometric(self, pred, mask):
        torch = self.torch
        tot, cnt = 0.0, 0.0
        per = {}
        for g in self.rend.groups:
            r = (pred[g] - self.obs[g]) / self.sigma[g]
            rho = torch.log1p(0.5 * r * r)                    # Cauchy, scale 1 sigma
            tot = tot + (rho * mask).sum()
            cnt = cnt + mask.sum()
            per[g] = r
        return tot / max(float(cnt), 1.0), per

    def priors(self):
        torch = self.torch
        pr = self.prior
        lp = ((self.p_base - math.log(pr["base"])) / 0.35) ** 2
        lp = lp + ((self.p_rough - _inv_sigmoid(pr["rough"])) / 0.6) ** 2
        if self.learn_metal:
            lp = lp + ((self.p_metal - _inv_sigmoid(pr["metal"])) / 0.8) ** 2
        eb = self.env_b
        for g in self.rend.groups:
            lp = lp + (self.log_gain[g] / 0.1) ** 2
            lp = lp + (self.log_env[g] / 0.5) ** 2
            lp = lp + (eb[g] / 0.002) ** 2
        return lp

    def regularisers(self, h):
        torch = self.torch
        ds = self.rend.ds
        huu = (h[:, 2:] - 2 * h[:, 1:-1] + h[:, :-2]) / ds ** 2
        hvv = (h[2:] - 2 * h[1:-1] + h[:-2]) / ds ** 2
        huv = (h[1:, 1:] - h[1:, :-1] - h[:-1, 1:] + h[:-1, :-1]) / ds ** 2
        curv = (huu ** 2).mean() + (hvv ** 2).mean() + 2 * (huv ** 2).mean()
        tau = float(self.cfg["sign_tau_mm"])
        sign = (torch.relu(h - tau) ** 2).mean() / tau ** 2
        return float(self.cfg["lam_curv"]) * curv + float(self.cfg["lam_sign"]) * sign

    def m_regulariser(self, logm):
        if logm is None:
            return 0.0
        ds = self.rend.ds
        gu = (logm[:, 1:] - logm[:, :-1]) / ds
        gv = (logm[1:] - logm[:-1]) / ds
        return float(self.cfg["lam_m"]) * (logm ** 2).mean() + float(self.cfg["lam_m_smooth"]) * (
            (gu ** 2).mean() + (gv ** 2).mean())


def _material_stage(fit: Fit, iters: int, lr: float) -> None:
    torch = fit.torch
    opt = torch.optim.Adam(fit.material_params(), lr=lr)
    # start the environment offset at the border's mean residual of the direct model
    with torch.no_grad():
        pred = fit.rend.render(fit.zero_h, fit.material(), fit.nuis())
        for g in fit.rend.groups:
            m = fit.border > 0
            if m.any():
                b = float(torch.clamp((fit.obs[g] - pred[g])[m].median(), min=1e-5)) / ENV_UNIT
                b = min(b, 0.3)
                fit.q_b[g].fill_(math.log(math.expm1(b)) if b < 20 else b)
    npx = max(float(fit.border.sum()), 1.0)
    for _ in range(iters):
        opt.zero_grad()
        pred = fit.rend.render(fit.zero_h, fit.material(), fit.nuis())
        loss, _ = fit.photometric(pred, fit.border)
        loss = loss + fit.priors() / npx
        loss.backward()
        opt.step()


def _depth_of(h: np.ndarray) -> float:
    return float(max(0.0, -np.min(h)))


def fit_once(crop: Crop, rend: Renderer, prior: Mapping[str, float], cfg: Mapping[str, Any], aniso: float,
             seed: int, method: str, mat_state: dict | None = None, size_mm: float = 1.0) -> dict:
    torch = _torch()
    fit = Fit(crop, rend, prior, cfg, aniso)
    if mat_state is None:
        _material_stage(fit, int(cfg["iters_material"]), 0.03)
    else:
        with torch.no_grad():
            for name in ("p_base", "p_rough", "p_metal"):
                getattr(fit, name).copy_(mat_state[name])
            for name in ("log_gain", "q_b", "log_env"):
                for g, v in mat_state[name].items():
                    getattr(fit, name)[g].copy_(v)
    state = {"p_base": fit.p_base.detach().clone(), "p_rough": fit.p_rough.detach().clone(),
             "p_metal": fit.p_metal.detach().clone(),
             "log_gain": {g: v.detach().clone() for g, v in fit.log_gain.items()},
             "q_b": {g: v.detach().clone() for g, v in fit.q_b.items()},
             "log_env": {g: v.detach().clone() for g, v in fit.log_env.items()}}
    scale = 0.3
    if method == "grid":
        field_ = GridField(rend.D, scale, rend.dtype, bool(cfg["material_field"]))
    else:
        Lmax = max(rend.shape) * crop.s
        sigma = float(cfg["fourier_sigma"]) * Lmax / max(size_mm, 2 * crop.s) / 2.0
        field_ = FourierField(rend.xy, rend.D, int(cfg["n_fourier"]), sigma, int(cfg["hidden"]), int(cfg["layers"]),
                              scale, seed, rend.dtype, float(cfg["m_scale"]), bool(cfg["material_field"]))
    params = field_.parameters()
    opt = torch.optim.Adam([{"params": params, "lr": float(cfg["lr"]) * (5.0 if method == "grid" else 1.0)},
                            {"params": fit.material_params(), "lr": 0.01}])
    iters = int(cfg["iters"])
    npx = max(float(fit.valid.sum()), 1.0)
    hist = []
    for it in range(iters):
        opt.zero_grad()
        h, logm = field_.both()
        pred = rend.render(h, fit.material(), fit.nuis(), None if logm is None else torch.exp(logm))
        loss, _ = fit.photometric(pred, fit.valid)
        loss = loss + fit.regularisers(h) + fit.m_regulariser(logm) + fit.priors() / npx
        loss.backward()
        opt.step()
        if it >= int(0.8 * iters) - 1 or it % 10 == 0:
            hist.append(_depth_of(h.detach().numpy()))
    if int(cfg.get("lbfgs_iters") or 0) > 0:
        lb = torch.optim.LBFGS(params, lr=0.5, max_iter=int(cfg["lbfgs_iters"]), line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            hh, lm = field_.both()
            pr = rend.render(hh, fit.material(), fit.nuis(), None if lm is None else torch.exp(lm))
            ls, _ = fit.photometric(pr, fit.valid)
            ls = ls + fit.regularisers(hh) + fit.m_regulariser(lm) + fit.priors() / npx
            ls.backward()
            return ls

        lb.step(closure)
    with torch.no_grad():
        h, logm = field_.both()
        pred = rend.render(h, fit.material(), fit.nuis(), None if logm is None else torch.exp(logm))
        _, per = fit.photometric(pred, fit.valid)
        r = torch.cat([per[g][fit.valid > 0].ravel() for g in rend.groups])
        resid = float(1.4826 * torch.median(torch.abs(r)))
    hf = h.detach().numpy().astype(np.float64)
    # working-lattice height (mean of the ss x ss sub-samples)
    ss = rend.ss
    Nv, Nu = rend.shape
    hw = hf.reshape(Nv, ss, Nu, ss).mean(axis=(1, 3))
    tail = hist[-max(3, len(hist) // 5):]
    d_end = _depth_of(hf)
    stable = (max(tail) - min(tail)) <= float(cfg["converge_tol"]) * max(d_end, 0.02) if tail else False
    return {"h_fine": hf, "h": hw, "depth": d_end, "resid": resid, "stable": bool(stable),
            "logm": None if logm is None else logm.detach().numpy(),
            "material": {"base": float(torch.exp(fit.p_base.detach())), "rough": float(torch.sigmoid(fit.p_rough.detach())),
                         "metal": float(torch.sigmoid(fit.p_metal.detach())) if fit.learn_metal else 1.0,
                         "gain": {g: float(torch.exp(v.detach())) for g, v in fit.log_gain.items()},
                         "env_b": {g: float(v.detach()) for g, v in fit.env_b.items()},
                         "env_gain": {g: float(torch.exp(v.detach())) for g, v in fit.log_env.items()}},
            "mat_state": state, "pred": {g: pred[g].numpy() for g in rend.groups}}


def estimate_depth(frames: Mapping[str, np.ndarray], meta: Mapping[str, Any], box_mm: Sequence[float],
                   material: str, thickness_mm: float | None = None, *, method: str = "pinn",
                   groups: Sequence[str] | None = None, n_seeds: int | None = None, threads: int | None = None,
                   cfg: Mapping[str, Any] | None = None, return_map: bool = False,
                   noise_var: Mapping[str, np.ndarray] | None = None, kind: str = "dent") -> DepthResult:
    """Depth of one dent or pit from the ambient-subtracted LED frames (docs/twin/PHYSICS_PINN.md 2).

    frames: {group: (H, W) float} rectified plate-grid frames in FS (luma), e.g. {'L': Lc, 'R': Rc}
      (or ring sectors). meta: 'lights' (LED dicts with 'group', world metres), 'camera_centre'
      (world m), 'H_grid_plate', 'mm_per_px', 'k' (FS per radiance unit), 'plate_size_mm'.
    box_mm: [u0, v0, u1, v1] plate mm of the detection. material: plate.material.kind (priors).
    method: 'pinn' (Fourier-feature MLP) or 'grid' (free lattice ablation).
    Returns DepthResult; max_depth_mm is relative to the crop border plane.
    """
    torch = _torch()
    t0 = time.perf_counter()
    c = dict(DEFAULTS)
    c.update(cfg or {})
    old_threads = torch.get_num_threads()
    if threads:
        torch.set_num_threads(int(threads))
    try:
        groups = list(groups or [g for g in ("L", "R") if g in frames] or list(frames))
        crop = make_crop(frames, meta, box_mm, c, groups, noise_var)
        prior = dict(MATERIAL_PRIORS.get(material, DEFAULT_PRIOR))
        img_sum = sum(crop.obs[g] for g in groups)
        grain = meta.get("grain_deg")
        if grain is None:
            ref = float(np.median(img_sum[crop.border])) if crop.border.any() else float(np.median(img_sum))
            grain = grain_angle_deg(np.where(crop.border, img_sum, ref)) if prior["aniso"] > 0.05 else 0.0
        grain = float(grain)
        tangent = O.tangent_from_grain(grain)
        meta = dict(meta)
        meta.setdefault("env_material", prior)
        rend = Renderer(crop, meta, groups, c, tangent)
        size = max(box_mm[2] - box_mm[0], box_mm[3] - box_mm[1])
        n = int(n_seeds if n_seeds is not None else c["n_seeds"])
        runs = []
        mat_state = None
        for s in range(max(1, n)):
            r = fit_once(crop, rend, prior, c, prior["aniso"], 1000 + 17 * s, method, mat_state, size)
            mat_state = r["mat_state"]
            runs.append(r)
            if method == "grid":
                break
        depths = np.array([r["depth"] for r in runs])
        best = runs[int(np.argmin([r["resid"] for r in runs]))]
        hmean = np.mean([r["h"] for r in runs], axis=0)
        d = float(np.median(depths))
        ens_sd = float(np.std(depths, ddof=1)) if len(depths) > 1 else 0.0
        rel, ab = sys_rel(kind, len(groups)), SYS_ABS
        ci = math.sqrt((Z90 * ens_sd) ** 2 + (rel * d) ** 2 + ab ** 2)
        resid = float(np.median([r["resid"] for r in runs]))
        converged = bool(all(r["stable"] for r in runs)) and resid <= float(c["resid_max"])
        notes = []
        if not converged:
            notes.append("fit not converged" if resid <= float(c["resid_max"]) else
                         f"model does not explain the frames (residual {resid:.1f} sigma)")
        if crop.f > 1:
            notes.append(f"crop block-averaged {crop.f}x ({crop.s:.2f} mm/px)")
        if size < 4 * crop.s:
            notes.append(f"defect spans < 4 px ({size:.2f} mm at {crop.s:.2f} mm/px): depth weakly observable")
        # profile through the deepest point (working lattice)
        i, j = np.unravel_index(int(np.argmin(hmean)), hmean.shape)
        Nv, Nu = hmean.shape
        prof = {"axis_u": {"u_mm": (crop.u0 + crop.s * np.arange(Nu)).round(3).tolist(),
                           "h_mm": hmean[i].round(4).tolist(), "v_mm": round(crop.v0 + crop.s * i, 3)},
                "axis_v": {"v_mm": (crop.v0 + crop.s * np.arange(Nv)).round(3).tolist(),
                           "h_mm": hmean[:, j].round(4).tolist(), "u_mm": round(crop.u0 + crop.s * j, 3)},
                "deepest_mm": [round(crop.u0 + crop.s * j, 3), round(crop.v0 + crop.s * i, 3)]}
        vol = float(np.sum(np.maximum(-hmean, 0.0)) * crop.s ** 2)
        res = DepthResult(max_depth_mm=d, depth_ci_mm=ci, volume_mm3=vol, profile=prof, residual=resid,
                          converged=converged, runtime_s=time.perf_counter() - t0, method=method,
                          members=[float(x) for x in depths],
                          rel_thickness=(d / float(thickness_mm)) if thickness_mm else None,
                          material_fit=best["material"], notes=notes,
                          height_map=hmean if return_map else None,
                          crop={"u0": crop.u0, "v0": crop.v0, "s": crop.s, "f": crop.f, "shape": list(hmean.shape),
                                "grain_deg": grain, "groups": groups,
                                "pred": best["pred"] if return_map else None,
                                "obs": crop.obs if return_map else None})
        return res
    finally:
        torch.set_num_threads(old_threads)
