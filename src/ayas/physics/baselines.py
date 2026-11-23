"""Baselines for the depth estimator (docs/twin/PHYSICS_PINN.md 2.4).

(a) classical_depth: per-pixel photometric slopes, then integration with the same boundary
    condition as the PINN (h = 0 on the crop border):
    1. the same crop, renderer (optics: direct LEDs + hood environment) and material fit on the
       flat border as depth_pinn (so the comparison isolates the shape model);
    2. per pixel, independently, the plate slopes (p, q) = (dh/du, dh/dv) minimising the same
       robust photometric residual of every LED group (plus a small Tikhonov term), by Adam on
       all pixels at once (no coupling between pixels). With 3+ groups a per-pixel material
       factor is solved too (classical photometric stereo); with L/R it is held at 1, and q is
       unconstrained by the data where L/R are mirror-symmetric (it stays near 0);
    3. Poisson integration of (p, q) with h = 0 on the crop border (5-point Laplacian, DST-I:
       the Frankot-Chellappa least-squares surface with Dirichlet instead of periodic BC).
(b) no_depth: today's behaviour (rules.depth_not_measured): no number, HOLD_FOR_QA above minor.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import depth_pinn as DP
from . import optics as O


def dst1(x: np.ndarray, axis: int) -> np.ndarray:
    """Unnormalised DST-I along an axis (scipy.fft.dst type 1 convention), numpy only."""
    x = np.moveaxis(np.asarray(x, float), axis, -1)
    n = x.shape[-1]
    ext = np.zeros(x.shape[:-1] + (2 * (n + 1),))
    ext[..., 1:n + 1] = x
    ext[..., n + 2:] = -x[..., ::-1]
    y = -np.fft.fft(ext, axis=-1).imag[..., 1:n + 1]
    return np.moveaxis(y, -1, axis)


def poisson_dirichlet(p: np.ndarray, q: np.ndarray, s: float) -> np.ndarray:
    """Height h (same shape) with h = 0 on the border whose gradient best matches (p, q) (mm/mm),
    spacing s mm: solve lap h = dp/du + dq/dv on the interior by DST-I."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    nv, nu = p.shape
    div = np.gradient(p, s, axis=1) + np.gradient(q, s, axis=0)
    f = div[1:-1, 1:-1]
    m, n = f.shape
    if m < 1 or n < 1:
        return np.zeros_like(p)
    F = dst1(dst1(f, 0), 1)
    j = np.arange(1, n + 1)
    k = np.arange(1, m + 1)
    lam = (2 * np.cos(np.pi * j / (n + 1)) - 2)[None, :] / s ** 2 + (2 * np.cos(np.pi * k / (m + 1)) - 2)[:, None] / s ** 2
    H = F / lam
    h_in = dst1(dst1(H, 0), 1) / (4.0 * (m + 1) * (n + 1))
    h = np.zeros_like(p)
    h[1:-1, 1:-1] = h_in
    return h


def classical_depth(frames: Mapping[str, np.ndarray], meta: Mapping[str, Any], box_mm: Sequence[float],
                    material: str, thickness_mm: float | None = None, *, groups: Sequence[str] | None = None,
                    cfg: Mapping[str, Any] | None = None, threads: int | None = None, iters: int = 250,
                    tikhonov: float = 1e-3, noise_var: Mapping[str, np.ndarray] | None = None,
                    return_map: bool = False) -> DP.DepthResult:
    """Photometric slopes per pixel + Poisson integration with h = 0 on the crop border."""
    import torch
    t0 = time.perf_counter()
    c = dict(DP.DEFAULTS)
    c.update(cfg or {})
    c["ss"] = 1                                  # one normal per observed pixel
    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(int(threads))
    try:
        groups = list(groups or [g for g in ("L", "R") if g in frames] or list(frames))
        crop = DP.make_crop(frames, meta, box_mm, c, groups, noise_var)
        prior = dict(DP.MATERIAL_PRIORS.get(material, DP.DEFAULT_PRIOR))
        grain = meta.get("grain_deg")
        if grain is None:
            img = sum(crop.obs[g] for g in groups)
            ref = float(np.median(img[crop.border])) if crop.border.any() else float(np.median(img))
            grain = DP.grain_angle_deg(np.where(crop.border, img, ref)) if prior["aniso"] > 0.05 else 0.0
        meta = dict(meta)
        meta.setdefault("env_material", prior)
        rend = DP.Renderer(crop, meta, groups, c, O.tangent_from_grain(float(grain)))
        fit = DP.Fit(crop, rend, prior, c, prior["aniso"])
        DP._material_stage(fit, int(c["iters_material"]), 0.03)
        for prm in fit.material_params():
            prm.requires_grad_(False)
        shape = rend.D.shape
        P = torch.zeros(shape, dtype=rend.dtype, requires_grad=True)
        Q = torch.zeros(shape, dtype=rend.dtype, requires_grad=True)
        free_m = len(groups) >= 3
        Mlog = torch.zeros(shape, dtype=rend.dtype, requires_grad=free_m)
        opt = torch.optim.Adam([P, Q] + ([Mlog] if free_m else []), lr=0.02)
        mat = fit.material()
        nuis = fit.nuis()
        for _ in range(iters):
            opt.zero_grad()
            n = O.normals_from_gradients(P, Q)
            if c.get("valid_reflection", True):
                n = O.valid_reflection_normals(n, rend.V)
            rad = O.shade(n, rend.lat, mat, groups, multiscatter=True, valid_reflection=False)
            loss = 0.0
            for g in groups:
                direct = rad[g][..., 0] * torch.exp(Mlog)
                if rend.env is not None:
                    envr = O.env_radiance(n, rend.V, rend.env[g], mat.base[0], mat.metallic, rend.env_irr[g])
                    direct = direct + torch.exp(nuis["env_gain"][g]) * envr
                pred = rend.k * torch.exp(nuis["log_gain"][g]) * direct + nuis["env_b"][g]
                r = (pred - fit.obs[g]) / fit.sigma[g]
                loss = loss + (torch.log1p(0.5 * r * r) * fit.valid).sum()
            loss = loss + tikhonov * ((P ** 2 + Q ** 2).sum() + 0.1 * (Mlog ** 2).sum()) * len(groups)
            loss.backward()
            opt.step()
        p, q = P.detach().numpy().astype(float), Q.detach().numpy().astype(float)
        h = poisson_dirichlet(p, q, crop.s)
        d = float(max(0.0, -h.min()))
        vol = float(np.sum(np.maximum(-h, 0.0)) * crop.s ** 2)
        i, j = np.unravel_index(int(np.argmin(h)), h.shape)
        prof = {"deepest_mm": [round(crop.u0 + crop.s * j, 3), round(crop.v0 + crop.s * i, 3)],
                "axis_u": {"h_mm": h[i].round(4).tolist()}, "axis_v": {"h_mm": h[:, j].round(4).tolist()}}
        return DP.DepthResult(max_depth_mm=d, depth_ci_mm=None, volume_mm3=vol, profile=prof, residual=float("nan"),
                              converged=True, runtime_s=time.perf_counter() - t0,
                              method="classical_ps" if free_m else "classical_2light",
                              rel_thickness=(d / thickness_mm) if thickness_mm else None,
                              height_map=h if return_map else None,
                              crop={"u0": crop.u0, "v0": crop.v0, "s": crop.s, "f": crop.f, "groups": groups})
    finally:
        torch.set_num_threads(old)


def no_depth() -> DP.DepthResult:
    """Today's behaviour: depth is not measured (rules.depth_not_measured -> HOLD_FOR_QA above minor)."""
    return DP.DepthResult(max_depth_mm=None, depth_ci_mm=None, volume_mm3=None, profile={}, residual=float("nan"),
                          converged=False, runtime_s=0.0, method="none", notes=["depth not measured"])
