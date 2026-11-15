"""Forward image-formation model of TriNetra's LED frames (docs/twin/PHYSICS_PINN.md section 1).

Given a height map h(u, v) in mm on a lattice of plate points, per-pixel (or per-material)
albedo, roughness and metallic, the plan's (or device's) LEDs and the camera centre, this
predicts each LED group's ambient-subtracted radiance, and with the fixed exposure k the frame
in full-scale (FS) units, as lightproc's Lc = L - A and Rc = R - A.

Frames and units (trinetra.twin.frames): plate mm, u right and v down; world metres
X = u/1000, Y = -v/1000, Z = w/1000; the plate front face is Z = 0.

    n(u, v)   = normalize(-dh/du, +dh/dv, 1)                     world shading normal (h in mm)
    Phi_i(x)  = P_i / (4 pi d_i^2) * spot_i(cos a_i)             W/m^2, LED i at distance d_i
    E_i(x, n) = Phi_i(x) * max(0, n . l_i)                       = lighting.light_irradiance_w
    L_g(x)    = sum_{i in g} f(l_i, v; n, material) E_i c_i     W m^-2 sr^-1 (Blender "W")
    S_g(x)    = k * L_g(x)                                       FS, k = burst.exposure_k

f is an approximation of Blender's Principled BSDF (v2, Blender >= 4; the pilot used 5.2.3):
- metal (weight metallic): GGX microfacet, alpha = roughness^2, anisotropic alpha_x = alpha /
  sqrt(1 - 0.9 a), alpha_y = alpha * sqrt(1 - 0.9 a) along the plan's tangent, height-correlated
  Smith G2, Schlick Fresnel with F0 = base colour (Specular Tint white makes the F82 term 0),
  and Cycles' multiscatter energy compensation 1 + F_ms (1 - E(mu_v)) / E(mu_v);
- dielectric (weight 1 - metallic): the same GGX lobe with the exact dielectric Fresnel at
  IOR 1.5, over a Lambertian base/pi weighted by 1 - F(mu_v) (albedo scaling).
The light is not occluded (Blender renders the twin's relief as a normal map: no cast shadows,
no displacement, no inter-reflection). shadows=True adds a horizon test on the height field
for real hardware. Cycles' ensure_valid_specular_reflection (normal-map node) is applied:
it caps the shading normal's tilt at about 45 deg towards the camera, so the twin cannot show
steeper walls.

Every function takes numpy arrays, or torch tensors (then it is differentiable); the two give
the same numbers (tests/test_physics_optics.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

try:                                                    # optional: the PINN needs it, the core does not
    import torch as _torch
except Exception:                                       # pragma: no cover - torch missing on the Pi app venv
    _torch = None

LUMA = np.array([0.2126, 0.7152, 0.0722])
IOR_DIELECTRIC = 1.5                                     # Principled v2 default IOR
ANISO_K = 0.9                                            # Cycles: aspect = sqrt(1 - 0.9 * anisotropic)
GRAIN_TO_TANGENT_DEG = 90.0                              # render_scene.py: tangent = grain + 90 deg
MIN_ALPHA = 1e-4
EPS = 1e-7


# ----------------------------------------------------------------------------- array backend

class _NumpyNS:
    is_torch = False

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def clip(x, lo=None, hi=None):
        return np.clip(x, lo, hi)

    @staticmethod
    def where(c, a, b):
        return np.where(c, a, b)

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)

    @staticmethod
    def minimum(a, b):
        return np.minimum(a, b)

    @staticmethod
    def sum(x, axis):
        return np.sum(x, axis=axis)

    @staticmethod
    def stack(xs, axis):
        return np.stack(xs, axis=axis)

    @staticmethod
    def cross(a, b):
        return np.cross(a, b)

    @staticmethod
    def const(x, like):
        return np.asarray(x, dtype=np.asarray(like).dtype)

    @staticmethod
    def floor(x):
        return np.floor(x)

    @staticmethod
    def take(table, idx):
        return np.asarray(table)[idx]


class _TorchNS:
    is_torch = True

    @staticmethod
    def sqrt(x):
        return _torch.sqrt(x)

    @staticmethod
    def exp(x):
        return _torch.exp(x)

    @staticmethod
    def abs(x):
        return _torch.abs(x)

    @staticmethod
    def clip(x, lo=None, hi=None):
        return _torch.clamp(x, lo, hi)

    @staticmethod
    def where(c, a, b):
        return _torch.where(c, a, b)

    @staticmethod
    def maximum(a, b):
        if not isinstance(b, _torch.Tensor):
            return _torch.clamp(a, min=b)
        if not isinstance(a, _torch.Tensor):
            return _torch.clamp(b, min=a)
        return _torch.maximum(a, b)

    @staticmethod
    def minimum(a, b):
        if not isinstance(b, _torch.Tensor):
            return _torch.clamp(a, max=b)
        if not isinstance(a, _torch.Tensor):
            return _torch.clamp(b, max=a)
        return _torch.minimum(a, b)

    @staticmethod
    def sum(x, axis):
        return _torch.sum(x, dim=axis)

    @staticmethod
    def stack(xs, axis):
        return _torch.stack(xs, dim=axis)

    @staticmethod
    def cross(a, b):
        a, b = _torch.broadcast_tensors(a, b)
        return _torch.cross(a, b, dim=-1)

    @staticmethod
    def const(x, like):
        return _torch.as_tensor(np.asarray(x), dtype=like.dtype, device=like.device)

    @staticmethod
    def floor(x):
        return _torch.floor(x)

    @staticmethod
    def take(table, idx):
        return table[idx]


def is_torch(x) -> bool:
    return _torch is not None and isinstance(x, _torch.Tensor)


def ns(*xs):
    """The array namespace (numpy or torch) of the first torch tensor among xs, else numpy."""
    return _TorchNS if any(is_torch(x) for x in xs) else _NumpyNS


def _dot(a, b, xp):
    return xp.sum(a * b, -1)


def _normalize(v, xp):
    return v / xp.sqrt(xp.maximum(_dot(v, v, xp), 1e-20))[..., None]


# ----------------------------------------------------------------------------- lights

@dataclass
class LightSet:
    """One LED group: point/spot lights in world metres (plan 'lights' entries)."""

    pos: np.ndarray                     # (N, 3) m
    axis: np.ndarray                    # (N, 3) unit emission direction (plan 'dir')
    power: np.ndarray                   # (N,) Blender W
    cos_half: np.ndarray                # (N,) cos(spot_deg / 2); -1 for a point light
    blend: np.ndarray                   # (N,)
    color: np.ndarray                   # (N, C) per-channel colour weights (C = 3 RGB, or 1 luma)
    names: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.power))


def light_set(lights: Sequence[Mapping[str, Any]], channels: str = "luma") -> LightSet:
    """Plan light dicts (kind spot | point) -> LightSet. channels 'luma' (C = 1) or 'rgb' (C = 3)."""
    pos, axis, pw, ch, bl, col, names = [], [], [], [], [], [], []
    for L in lights:
        kind = str(L.get("kind", "spot")).lower()
        if kind not in ("spot", "point"):
            raise ValueError(f"light {L.get('name')}: optics models point/spot LEDs, not {kind!r}")
        pos.append(np.asarray(L["pos_m"], float))
        a = np.asarray(L.get("dir", (0.0, 0.0, -1.0)), float)
        axis.append(a / np.linalg.norm(a))
        pw.append(float(L.get("power_w", 0.0)))
        if kind == "spot":
            ch.append(math.cos(math.radians(min(float(L.get("spot_deg", 180.0)), 180.0)) / 2.0))
            bl.append(float(L.get("blend", 0.0)))
        else:
            ch.append(-1.0)
            bl.append(0.0)
        c = np.asarray(L.get("color", (1.0, 1.0, 1.0)), float)[:3]
        col.append([float(LUMA @ c)] if channels == "luma" else list(c))
        names.append(str(L.get("name", "")))
    if not pos:
        raise ValueError("light_set: no lights")
    return LightSet(np.array(pos), np.array(axis), np.array(pw), np.array(ch), np.array(bl), np.array(col), names)


def group_lights(lights: Sequence[Mapping[str, Any]], groups: Mapping[str, Any] | Sequence[str] = ("L", "R"),
                 channels: str = "luma") -> dict[str, LightSet]:
    """{group: LightSet}. groups: names matched against each light's 'group', or a mapping
    {name: predicate(light) -> bool} (e.g. ring sectors, see sector_groups)."""
    out = {}
    if isinstance(groups, Mapping):
        for g, pred in groups.items():
            sel = [L for L in lights if pred(L)]
            if sel:
                out[g] = light_set(sel, channels)
    else:
        for g in groups:
            sel = [L for L in lights if L.get("group") == g]
            if sel:
                out[g] = light_set(sel, channels)
    return out


def ring_azimuth_deg(light: Mapping[str, Any], R: np.ndarray, t: np.ndarray, centre_world: np.ndarray) -> float:
    """Azimuth (deg, camera frame, atan2(y, x)) of a ring LED about the optical-axis plate point."""
    p = np.asarray(R, float) @ np.asarray(light["pos_m"], float) + np.asarray(t, float)
    c = np.asarray(R, float) @ np.asarray(centre_world, float) + np.asarray(t, float)
    return float(math.degrees(math.atan2(p[1] - c[1], p[0] - c[0])))


def sector_groups(lights: Sequence[Mapping[str, Any]], R, t, centre_world, n_sectors: int,
                  offset_deg: float = 0.0, mount: str | None = "ring") -> dict[str, list[dict]]:
    """Split the LEDs (mount 'ring' by default) into n_sectors equal azimuth sectors (camera frame,
    sector k covers [offset + k 360/n, offset + (k+1) 360/n)). Returns {f'S{n}_{k}': [lights]}.

    n_sectors = 2 with offset 90 deg reproduces L (camera x < 0) and R."""
    out: dict[str, list[dict]] = {}
    for L in lights:
        if mount is not None and L.get("mount") != mount:
            continue
        if L.get("group") not in ("L", "R"):
            continue
        az = (ring_azimuth_deg(L, R, t, centre_world) - offset_deg) % 360.0
        k = int(az // (360.0 / n_sectors)) % n_sectors
        out.setdefault(f"S{n_sectors}_{k}", []).append(dict(L))
    return dict(sorted(out.items()))


def spot_mask(cos_a, cos_half, blend, xp=None):
    """Cycles spot attenuation (vectorised lighting.spot_mask): smoothstep inside the cone, 0 outside."""
    xp = xp or ns(cos_a)
    cos_half = xp.const(cos_half, cos_a) if not is_torch(cos_half) else cos_half
    blend = xp.const(blend, cos_a) if not is_torch(blend) else blend
    den = xp.maximum((1.0 - cos_half) * blend, 1e-12)
    f = xp.clip((cos_a - cos_half) / den, 0.0, 1.0)
    s = f * f * (3.0 - 2.0 * f)
    s = xp.where(blend > 0, s, xp.where(cos_a > cos_half, 1.0 + 0.0 * s, 0.0 * s))
    return xp.where(cos_a > cos_half, s, 0.0 * s)


def plate_points(uv_mm, xp=None):
    """(..., 2) plate mm -> (..., 3) world metres on the front face (Z = 0)."""
    xp = xp or ns(uv_mm)
    u, v = uv_mm[..., 0], uv_mm[..., 1]
    return xp.stack([u / 1000.0, -v / 1000.0, 0.0 * u], -1)


def incident(ls: LightSet, P, like=None):
    """Per light: unit direction to the light l (N, ..., 3) and normal-incidence irradiance
    Phi (N, ...) W/m^2 (inverse square and spot; the n . l cosine is applied in shade())."""
    xp = ns(P, like)
    ref = P if like is None else like
    pos = xp.const(ls.pos, ref).reshape((ls.n,) + (1,) * (P.ndim - 1) + (3,))
    axis = xp.const(ls.axis, ref).reshape(pos.shape)
    shp = (ls.n,) + (1,) * (P.ndim - 1)
    pw = xp.const(ls.power, ref).reshape(shp)
    ch = xp.const(ls.cos_half, ref).reshape(shp)
    bl = xp.const(ls.blend, ref).reshape(shp)
    d = pos - P[None]
    r2 = xp.maximum(_dot(d, d, xp), 1e-12)
    r = xp.sqrt(r2)
    l = d / r[..., None]
    cos_a = -_dot(l, axis, xp)
    Phi = pw / (4.0 * math.pi * r2) * spot_mask(cos_a, ch, bl, xp)
    return l, Phi


def view_dirs(P, centre_world, like=None):
    """Unit vectors (..., 3) from plate points to the camera centre (world m)."""
    xp = ns(P, like)
    c = xp.const(np.asarray(centre_world, float), P if like is None else like)
    return _normalize(c - P, xp)


# ----------------------------------------------------------------------------- normals

def height_gradients(h, du_mm: float, dv_mm: float, xp=None):
    """dh/du, dh/dv (dimensionless) of a height lattice h (..., Nv, Nu) in mm, with the
    same stencil as textures.normal_from_height (central inside, one-sided at the borders)."""
    xp = xp or ns(h)
    if xp.is_torch:
        gu = _torch.cat([(h[..., :, 1:2] - h[..., :, 0:1]) / du_mm,
                         (h[..., :, 2:] - h[..., :, :-2]) / (2 * du_mm),
                         (h[..., :, -1:] - h[..., :, -2:-1]) / du_mm], dim=-1)
        gv = _torch.cat([(h[..., 1:2, :] - h[..., 0:1, :]) / dv_mm,
                         (h[..., 2:, :] - h[..., :-2, :]) / (2 * dv_mm),
                         (h[..., -1:, :] - h[..., -2:-1, :]) / dv_mm], dim=-2)
        return gu, gv
    h = np.asarray(h)
    gu = np.concatenate([(h[..., :, 1:2] - h[..., :, 0:1]) / du_mm, (h[..., :, 2:] - h[..., :, :-2]) / (2 * du_mm),
                         (h[..., :, -1:] - h[..., :, -2:-1]) / du_mm], axis=-1)
    gv = np.concatenate([(h[..., 1:2, :] - h[..., 0:1, :]) / dv_mm, (h[..., 2:, :] - h[..., :-2, :]) / (2 * dv_mm),
                         (h[..., -1:, :] - h[..., -2:-1, :]) / dv_mm], axis=-2)
    return gu, gv


def normals_from_gradients(gu, gv, xp=None):
    """World normals (..., 3) from plate-frame slopes: normalize(-dh/du, +dh/dv, 1)."""
    xp = xp or ns(gu)
    inv = 1.0 / xp.sqrt(gu * gu + gv * gv + 1.0)
    return xp.stack([-gu * inv, gv * inv, inv], -1)


def normals_from_height(h, du_mm: float, dv_mm: float | None = None, xp=None):
    xp = xp or ns(h)
    gu, gv = height_gradients(h, du_mm, du_mm if dv_mm is None else dv_mm, xp)
    return normals_from_gradients(gu, gv, xp)


def valid_reflection_normals(n, v, xp=None):
    """Cycles ensure_valid_specular_reflection(Ng = +Z, I = v, N = n): tilt N back towards Ng
    until the view's mirror direction is at least min(0.9 Iz, 0.01) above the surface."""
    xp = xp or ns(n, v)
    nv = _dot(n, v, xp)
    R = 2.0 * nv[..., None] * n - v
    Iz = v[..., 2]
    thr = xp.minimum(0.9 * Iz, 0.01)
    ok = R[..., 2] >= thr
    X = n - n[..., 2:3] * xp.const([0.0, 0.0, 1.0], n)
    X = _normalize(X, xp)
    Ix = _dot(v, X, xp)
    a = Ix * Ix + Iz * Iz
    b = 2.0 * (a + Iz * thr)
    c = (thr + Iz) ** 2
    disc = xp.sqrt(xp.maximum(b * b - 4.0 * a * c, 0.0))
    nz2 = xp.where(Ix < 0, 0.25 * (b + disc) / a, 0.25 * (b - disc) / a)
    nx = xp.sqrt(xp.clip(1.0 - nz2, 0.0, 1.0))
    nz = xp.sqrt(xp.clip(nz2, 0.0, 1.0))
    fixed = nx[..., None] * X + nz[..., None] * xp.const([0.0, 0.0, 1.0], n)
    return xp.where(ok[..., None], n, fixed)


# ----------------------------------------------------------------------------- BRDF

def _lambda_ggx(wx, wy, wz, ax, ay, xp):
    """Smith Lambda of the (anisotropic) GGX distribution for a direction in the shading frame."""
    z2 = xp.maximum(wz * wz, 1e-12)
    return 0.5 * (-1.0 + xp.sqrt(1.0 + (ax * ax * wx * wx + ay * ay * wy * wy) / z2))


def fresnel_dielectric(cos_i, eta: float = IOR_DIELECTRIC, xp=None):
    """Exact unpolarised Fresnel reflectance of a dielectric (air -> eta)."""
    xp = xp or ns(cos_i)
    c = xp.clip(xp.abs(cos_i), 0.0, 1.0)
    g2 = eta * eta - 1.0 + c * c
    g = xp.sqrt(xp.maximum(g2, 0.0))
    A = (g - c) / (g + c)
    B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0)
    return 0.5 * A * A * (1.0 + B * B)


@lru_cache(maxsize=2)
def ggx_albedo_table(n: int = 32, nq: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Directional albedo E(rough, mu) of single-scattering isotropic GGX with F = 1 (n x n,
    rough and mu on linspace(0, 1, n)) and its cosine-weighted average E_avg(rough), by
    stratified importance-sampled quadrature (as Cycles' ggx_E / ggx_Eavg tables)."""
    r = np.linspace(0.0, 1.0, n)
    mu = np.linspace(0.0, 1.0, n)
    u1 = (np.arange(nq) + 0.5) / nq
    u2 = (np.arange(nq) + 0.5) / nq
    U1, U2 = np.meshgrid(u1, u2, indexing="ij")
    E = np.zeros((n, n))
    for i, rr in enumerate(r):
        a = max(rr * rr, 1e-3)
        th = np.arctan(a * np.sqrt(U1 / (1 - U1)))
        ph = 2 * np.pi * U2
        hx, hy, hz = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
        for j, m in enumerate(mu):
            m = max(m, 1e-3)
            vx, vz = math.sqrt(1 - m * m), m
            vh = vx * hx + vz * hz
            lx, ly, lz = 2 * vh * hx - vx, 2 * vh * hy, 2 * vh * hz - vz
            lam_v = 0.5 * (-1 + math.sqrt(1 + a * a * (1 - m * m) / (m * m)))
            lam_l = 0.5 * (-1 + np.sqrt(1 + a * a * (lx * lx + ly * ly) / np.maximum(lz * lz, 1e-12)))
            G2 = 1.0 / (1.0 + lam_v + lam_l)
            w = np.where((lz > 0) & (vh > 0), G2 * vh / (hz * m), 0.0)
            E[i, j] = w.mean()
    E = np.clip(E, 1e-3, 1.0)
    E_avg = 2.0 * np.trapezoid(E * mu[None, :], mu, axis=1) if hasattr(np, "trapezoid") else \
        2.0 * np.trapz(E * mu[None, :], mu, axis=1)
    return E, np.clip(E_avg, 1e-3, 1.0)


def _lerp_table(table, x, xp, n: int):
    """Linear interpolation of a 1-D table on linspace(0, 1, n) at x (array)."""
    t = xp.clip(x, 0.0, 1.0) * (n - 1)
    i0 = xp.clip(xp.floor(t), 0, n - 2)
    w = t - i0
    if xp.is_torch:
        i0 = i0.long()
        tab = _torch.as_tensor(table, dtype=x.dtype, device=x.device)
    else:
        i0 = i0.astype(int)
        tab = np.asarray(table)
    return xp.take(tab, i0) * (1 - w) + xp.take(tab, i0 + 1) * w


def _bilerp_table(table, x, y, xp, n: int):
    """Bilinear interpolation of an (n, n) table indexed [x, y] on linspace(0, 1, n)."""
    tx = xp.clip(x, 0.0, 1.0) * (n - 1)
    ty = xp.clip(y, 0.0, 1.0) * (n - 1)
    i0 = xp.clip(xp.floor(tx), 0, n - 2)
    j0 = xp.clip(xp.floor(ty), 0, n - 2)
    wx, wy = tx - i0, ty - j0
    if xp.is_torch:
        tab = _torch.as_tensor(table.ravel(), dtype=x.dtype, device=x.device)
        i0, j0 = i0.long(), j0.long()
    else:
        tab = np.asarray(table).ravel()
        i0, j0 = i0.astype(int), j0.astype(int)
    k = i0 * n + j0

    def at(off):
        return xp.take(tab, k + off)

    return (at(0) * (1 - wx) * (1 - wy) + at(n) * wx * (1 - wy) + at(1) * (1 - wx) * wy + at(n + 1) * wx * wy)


def multiscatter_factor(rough, mu_v, F_avg, xp=None):
    """Cycles' GGX energy compensation (albedo scaling, Kulla-Conty F_ms): 1 + F_ms (1 - E) / E."""
    xp = xp or ns(rough, mu_v)
    E_tab, Eavg_tab = ggx_albedo_table()
    n = E_tab.shape[0]
    E = _bilerp_table(E_tab, rough + 0.0 * mu_v, mu_v + 0.0 * rough, xp, n)
    Ea = _lerp_table(Eavg_tab, rough + 0.0 * mu_v, xp, n)
    Fms = F_avg * Ea[..., None] / (1.0 - F_avg * (1.0 - Ea[..., None]))
    return 1.0 + Fms * ((1.0 - E) / E)[..., None]


@dataclass
class Material:
    """Per-pixel or uniform surface parameters (arrays broadcast against the lattice (..., Nv, Nu)).

    base: (..., C) linear colour (C = 1 luma or 3); roughness, metallic, anisotropy: (...,) or scalars;
    tangent: world tangent (3,) (the anisotropy direction; see tangent_from_grain)."""

    base: Any
    roughness: Any
    metallic: Any
    anisotropy: Any = 0.0
    tangent: Any = None


def tangent_from_grain(grain_deg: float) -> np.ndarray:
    """World tangent of render_scene.tangent(): plate angle grain + 90 deg, direction (cos a, -sin a, 0)."""
    a = math.radians(float(grain_deg) + GRAIN_TO_TANGENT_DEG)
    return np.array([math.cos(a), -math.sin(a), 0.0])


def material_from_plan(plan: Mapping[str, Any], channels: str = "luma") -> Material:
    """Uniform Material from the plan's plate.material (the sampler's nominal values)."""
    m = (plan.get("plate") or {}).get("material") or {}
    base = np.asarray(m.get("base_color", (0.9, 0.9, 0.9)), float)[:3]
    base = np.array([LUMA @ base]) if channels == "luma" else base
    grain = m.get("grain_dir_deg")
    if grain is None and m.get("aniso_rotation") is not None:
        grain = (-360.0 * float(m["aniso_rotation"])) % 180.0
    return Material(base, float(m.get("roughness", 0.3)), float(m.get("metallic", 1.0)),
                    float(m.get("anisotropy", 0.0)), tangent_from_grain(float(grain or 0.0)))


def brdf_cos(n, l, v, mat: Material, multiscatter: bool = True, xp=None):
    """f(l, v) * max(n . l, 0) of the Principled approximation, for N lights.

    n, v: (..., 3) unit; l: (N, ..., 3) unit. Returns (N, ..., C)."""
    xp = xp or ns(n, l, v, mat.base, mat.roughness)
    like = n
    base = mat.base if is_torch(mat.base) else xp.const(mat.base, like)
    rough = mat.roughness if is_torch(mat.roughness) else xp.const(mat.roughness, like)
    metal = mat.metallic if is_torch(mat.metallic) else xp.const(mat.metallic, like)
    aniso = mat.anisotropy if is_torch(mat.anisotropy) else xp.const(mat.anisotropy, like)
    rough = xp.clip(rough, 0.0, 1.0)
    alpha = xp.maximum(rough * rough, MIN_ALPHA)
    aspect = xp.sqrt(1.0 - ANISO_K * xp.clip(aniso, 0.0, 1.0))
    ax, ay = alpha / aspect, alpha * aspect

    nl = _dot(n[None], l, xp)                                    # (N, ...)
    nv = xp.maximum(_dot(n, v, xp), 1e-4)                        # (...)
    hv = _normalize(l + v[None], xp)
    nh = _dot(n[None], hv, xp)
    vh = xp.maximum(_dot(v[None], hv, xp), 0.0)
    T = mat.tangent if mat.tangent is not None else np.array([1.0, 0.0, 0.0])
    T = T if is_torch(T) else xp.const(T, like)
    t = T - _dot(T, n, xp)[..., None] * n
    t = _normalize(t, xp)
    b = xp.cross(n, t)
    hx, hy = _dot(t[None], hv, xp), _dot(b[None], hv, xp)
    den = hx * hx / (ax * ax) + hy * hy / (ay * ay) + nh * nh
    D = 1.0 / (math.pi * ax * ay * den * den)
    lam_l = _lambda_ggx(_dot(t[None], l, xp), _dot(b[None], l, xp), nl, ax, ay, xp)
    lam_v = _lambda_ggx(_dot(t, v, xp), _dot(b, v, xp), nv, ax, ay, xp)
    G2 = 1.0 / (1.0 + lam_l + lam_v[None])
    pos = (nl > 0) & (nh > 0)
    nlp = xp.maximum(nl, 0.0)
    spec = xp.where(pos, D * G2 / (4.0 * nv[None]), 0.0 * nl)   # f_spec * (n . l), without Fresnel

    # metal: Schlick with F0 = base (per channel), F90 = 1
    s5 = (1.0 - vh) ** 5
    F_metal = base + (1.0 - base) * s5[..., None]                 # (N, ..., C)
    f_metal = spec[..., None] * F_metal
    # dielectric: exact Fresnel (IOR 1.5) specular + Lambert under albedo scaling
    F_d = fresnel_dielectric(vh, IOR_DIELECTRIC, xp)
    f_dspec = (spec * F_d)[..., None]
    w_diff = 1.0 - fresnel_dielectric(nv, IOR_DIELECTRIC, xp)
    f_diff = (nlp * w_diff[None])[..., None] * base / math.pi
    if multiscatter:
        F_avg_m = base + (1.0 - base) / 21.0
        f_metal = f_metal * multiscatter_factor(rough + 0.0 * nv, nv, F_avg_m, xp)[None]
        f_dspec = f_dspec * multiscatter_factor(rough + 0.0 * nv, nv, 0.092 + 0.0 * base, xp)[None]
    m = metal[..., None]
    return m * f_metal + (1.0 - m) * (f_dspec + f_diff)


# ----------------------------------------------------------------------------- shadows (real hardware only)

def horizon_shadow(h, l, du_mm: float, dv_mm: float, max_mm: float = 2.0, step_px: float = 0.75, xp=None):
    """Cast-shadow factor (N, Nv, Nu) in [0, 1] of a height lattice h (Nv, Nu) mm for light
    directions l (N, Nv, Nu, 3): a point is shadowed when the height field rises above the ray
    towards the light within max_mm. Soft (sigmoid of the clearance) so torch can differentiate.
    Not used for the twin: Blender renders relief as a normal map, which casts no shadows."""
    xp = xp or ns(h, l)
    lu = l[..., 0]
    lv = -l[..., 1]
    lw = l[..., 2]
    horiz = xp.sqrt(xp.maximum(lu * lu + lv * lv, 1e-12))
    tan_el = lw / horiz
    du_dir, dv_dir = lu / horiz, lv / horiz
    nv_, nu_ = h.shape[-2], h.shape[-1]
    n_steps = max(1, int(max_mm / (step_px * min(du_mm, dv_mm))))
    clearance = None
    if xp.is_torch:
        import torch.nn.functional as Fnn
        H = h[None, None].expand(l.shape[0], 1, nv_, nu_)
        ys, xs = _torch.meshgrid(_torch.arange(nv_, dtype=h.dtype), _torch.arange(nu_, dtype=h.dtype), indexing="ij")
    else:
        import cv2
        ys, xs = np.meshgrid(np.arange(nv_, dtype=np.float32), np.arange(nu_, dtype=np.float32), indexing="ij")
    for k in range(1, n_steps + 1):
        s_mm = k * step_px * min(du_mm, dv_mm)
        cx = xs[None] + du_dir * s_mm / du_mm
        cy = ys[None] + dv_dir * s_mm / dv_mm
        if xp.is_torch:
            grid = _torch.stack([2 * cx / max(nu_ - 1, 1) - 1, 2 * cy / max(nv_ - 1, 1) - 1], -1)
            hs = Fnn.grid_sample(H, grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
        else:
            hs = np.stack([cv2.remap(np.asarray(h, np.float32), cx[i].astype(np.float32), cy[i].astype(np.float32),
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for i in range(cx.shape[0])])
        c = hs - h[None] - s_mm * tan_el
        clearance = c if clearance is None else xp.maximum(clearance, c)
    beta = 1.0 / max(0.25 * min(du_mm, dv_mm) * 0.05, 1e-6)    # ~ 1/(5 um): a soft edge, nearly binary
    if xp.is_torch:
        return _torch.sigmoid(-beta * clearance)
    return 1.0 / (1.0 + np.exp(np.clip(beta * clearance, -60, 60)))


# ----------------------------------------------------------------------------- rendering

@dataclass
class Lattice:
    """Plate sample points of a crop: uv (Nv, Nu, 2) mm on a regular lattice of spacing (du, dv) mm
    along +u / +v, plus the precomputed per-light terms for each group."""

    uv: np.ndarray
    du: float
    dv: float
    P: np.ndarray
    V: np.ndarray
    groups: dict[str, LightSet]
    incident: dict[str, tuple[np.ndarray, np.ndarray]]

    @property
    def shape(self) -> tuple[int, int]:
        return self.uv.shape[:2]


def make_lattice(u0: float, v0: float, nu: int, nv: int, du: float, dv: float | None,
                 groups: Mapping[str, LightSet], centre_world, dtype=np.float64) -> Lattice:
    """A regular plate lattice with sample (r, c) at (u0 + c du, v0 + r dv) mm."""
    dv = du if dv is None else dv
    u = u0 + du * np.arange(nu)
    v = v0 + dv * np.arange(nv)
    U, V = np.meshgrid(u, v)
    uv = np.stack([U, V], -1).astype(dtype)
    P = plate_points(uv)
    Vd = view_dirs(P, centre_world)
    inc = {g: incident(ls, P) for g, ls in groups.items()}
    return Lattice(uv, float(du), float(dv), P, Vd, dict(groups), inc)


def lattice_to(lat: Lattice, device=None, dtype=None):
    """Torch copy of a lattice's arrays (P, V, incident terms), for the differentiable path."""
    if _torch is None:
        raise ImportError("torch is required for the differentiable optics path")
    dt = dtype or _torch.float32

    def T(a):
        return _torch.as_tensor(np.asarray(a), dtype=dt, device=device)

    inc = {g: (T(l), T(phi)) for g, (l, phi) in lat.incident.items()}
    return Lattice(lat.uv, lat.du, lat.dv, T(lat.P), T(lat.V), lat.groups, inc)


def shade(n, lat: Lattice, mat: Material, groups: Sequence[str] | None = None, multiscatter: bool = True,
          valid_reflection: bool = True, shadow=None):
    """Radiance (Nv, Nu, C) of each LED group for shading normals n (Nv, Nu, 3).

    shadow: optional {group: (N, Nv, Nu) factor} (horizon_shadow) for real hardware."""
    xp = ns(n, lat.V)
    V = lat.V
    if valid_reflection:
        n = valid_reflection_normals(n, V, xp)
    out = {}
    for g in (groups or list(lat.incident)):
        l, Phi = lat.incident[g]
        ls = lat.groups[g]
        col = xp.const(ls.color, n).reshape((ls.n,) + (1,) * (n.ndim - 1) + (ls.color.shape[1],))
        fc = brdf_cos(n, l, V, mat, multiscatter, xp)            # (N, Nv, Nu, C)
        w = Phi if shadow is None or g not in shadow else Phi * shadow[g]
        out[g] = xp.sum(fc * w[..., None] * col, 0)
    return out


def render_height(h, lat: Lattice, mat: Material, groups: Sequence[str] | None = None, **kw):
    """Radiance of each group for a height lattice h (Nv, Nu) mm on lat (normals as textures.py)."""
    n = normals_from_height(h, lat.du, lat.dv)
    return shade(n, lat, mat, groups, **kw)


def luma(img):
    """(..., 3) -> (...) Rec. 709 luma; (..., 1) -> (...)."""
    if img.shape[-1] == 1:
        return img[..., 0]
    xp = ns(img)
    return xp.sum(img * xp.const(LUMA, img), -1)


# ----------------------------------------------------------------------------- grid helpers

def grid_affine(H_grid_plate) -> tuple[np.ndarray, np.ndarray]:
    """(A, b) with plate (u, v) = A @ (x, y) + b for grid pixel centres (x, y): the rectified grid
    is a similarity of the plate (post.grid_geometry maps the plate corners onto its corners)."""
    Hinv = np.linalg.inv(np.asarray(H_grid_plate, float))
    Hinv = Hinv / Hinv[2, 2]
    return Hinv[:2, :2].copy(), Hinv[:2, 2].copy()


def grid_to_plate(H_grid_plate, xy) -> np.ndarray:
    A, b = grid_affine(H_grid_plate)
    return np.asarray(xy, float) @ A.T + b


def plate_to_grid_px(H_grid_plate, uv) -> np.ndarray:
    Hm = np.asarray(H_grid_plate, float)
    p = np.c_[np.atleast_2d(np.asarray(uv, float)), np.ones(len(np.atleast_2d(uv)))] @ Hm.T
    return p[:, :2] / p[:, 2:3]


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    r = int(math.ceil(3 * sigma)) if radius is None else radius
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


# camera + rectification point spread on the 0.04 mm lattice (fit in reports/physics_pinn.md 1.3):
# Cycles' 1.5 px Blackman-Harris pixel filter and post.rectify's bilinear resampling.
PSF_SIGMA_MM = 0.035


# ----------------------------------------------------------------------------- hood inter-reflection

HOOD_LID_Z_CAM_M = -0.001        # render_scene.HOOD_LID_Z_M: the lid just behind the lens
HOOD_TOP_Z_CAM_M = 0.017         # lighting.HOOD top_z_cam_mm


def hood_planes(hood: Mapping[str, Any], R, t) -> tuple[list, list]:
    """Inner faces of the hood as half-spaces (point, outward unit normal), world metres.

    Two convex parts (render_scene.build_hood / lighting.hood_corners): the frustum from the
    top rectangle (camera z = top_z_cam_mm) to the bottom plane Z = gap_mm, and the collar box
    from the top up to the lid (camera z = -1 mm). Returns (frustum planes + its top opening,
    collar planes + lid)."""
    R = np.asarray(R, float)
    t = np.asarray(t, float)
    C = -R.T @ t
    zb = float(hood.get("gap_mm", 0.0)) / 1000.0
    zt = float(hood.get("top_z_cam_mm", HOOD_TOP_Z_CAM_M * 1000.0)) / 1000.0
    a = R[2]
    depth = (zb - C[2]) / a[2]
    wt, ht = (float(x) / 1000.0 for x in hood.get("top_mm", (170.0, 160.0)))
    wb, hb = (float(x) / 1000.0 for x in hood.get("bottom_mm", (200.0, 190.0)))
    top, bot = [], []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        T = C + R.T @ np.array([sx * wt / 2, sy * ht / 2, zt])
        B = C + R.T @ np.array([sx * wb / 2, sy * hb / 2, depth])
        top.append(T)
        bot.append(T + (zb - T[2]) / (B[2] - T[2]) * (B - T))
    top, bot = np.array(top), np.array(bot)
    centre = np.vstack([top, bot]).mean(0)
    frustum = []
    for k in range(4):
        p0, p1, p2 = top[k], top[(k + 1) % 4], bot[(k + 1) % 4]
        n = np.cross(p1 - p0, p2 - p0)
        n /= np.linalg.norm(n)
        if np.dot(centre - p0, n) > 0:
            n = -n
        frustum.append((p0, n, f"wall{k}"))
    frustum.append((top[0], -a / np.linalg.norm(a), "top_opening"))
    collar = []
    for axis, half in ((0, wt / 2), (1, ht / 2)):
        for sgn in (-1, 1):
            p = C + R.T @ np.array([sgn * half if axis == 0 else 0.0, sgn * half if axis == 1 else 0.0, zt])
            n = sgn * R[axis]
            collar.append((p, n / np.linalg.norm(n), f"collar{axis}{'+' if sgn > 0 else '-'}"))
    collar.append((C + R.T @ np.array([0.0, 0.0, HOOD_LID_Z_CAM_M]), -a / np.linalg.norm(a), "lid"))
    return frustum, collar


def _exit(o: np.ndarray, d: np.ndarray, planes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First exit (t, plane index) of rays o + t d (N, 3) through a convex set of half-spaces."""
    tbest = np.full(len(d), np.inf)
    ibest = np.full(len(d), -1)
    for i, (p, n, _) in enumerate(planes):
        dn = d @ n
        tt = np.where(dn > 1e-9, ((p - o) @ n) / np.where(dn > 1e-9, dn, 1.0), np.inf)
        better = tt < tbest
        tbest = np.where(better, tt, tbest)
        ibest = np.where(better, i, ibest)
    return tbest, ibest, o + tbest[:, None] * d


def _schlick(f0, cos):
    return f0 + (1.0 - f0) * (1.0 - np.clip(cos, 0.0, 1.0)) ** 5


def hood_wall_irradiance(w: np.ndarray, n_in: np.ndarray, ls: LightSet, plate_reflect, plate_size_mm=None):
    """Irradiance (W/m^2, white) on hood points w (N, 3) with inward normals n_in (N, 3) from one
    LED group: direct, plus the LEDs' mirror images in the plate (Z = 0), weighted by the plate's
    specular reflectance plate_reflect(cos_incidence) where the reflection point lies on the plate."""
    E = np.zeros(len(w))
    for i in range(ls.n):
        p = ls.pos[i]
        # direct
        d = p[None] - w
        r = np.linalg.norm(d, axis=1)
        cos_w = np.einsum("ij,ij->i", d, n_in) / r
        cos_a = ((w - p[None]) / r[:, None]) @ ls.axis[i]
        sm = spot_mask(cos_a, np.full(len(w), ls.cos_half[i]), np.full(len(w), ls.blend[i]))
        E += ls.power[i] / (4 * math.pi * r * r) * sm * np.maximum(cos_w, 0.0) * float(ls.color[i].mean())
        # mirror image through the plate plane
        pim = np.array([p[0], p[1], -p[2]])
        d2 = pim[None] - w
        r2 = np.linalg.norm(d2, axis=1)
        s = p[2] / np.maximum(p[2] + w[:, 2], 1e-9)
        q = pim[None] + s[:, None] * (w - pim[None])          # reflection point on Z = 0
        ok = w[:, 2] > 0
        if plate_size_mm is not None:
            u, v = q[:, 0] * 1000.0, -q[:, 1] * 1000.0
            ok &= (u > 0) & (u < plate_size_mm[0]) & (v > 0) & (v < plate_size_mm[1])
        inc = q - p[None]
        rin = np.linalg.norm(inc, axis=1)
        cos_a2 = (inc / rin[:, None]) @ ls.axis[i]
        sm2 = spot_mask(cos_a2, np.full(len(w), ls.cos_half[i]), np.full(len(w), ls.blend[i]))
        cos_q = p[2] / rin
        cos_w2 = np.einsum("ij,ij->i", d2, n_in) / r2
        E += np.where(ok, ls.power[i] / (4 * math.pi * r2 * r2) * sm2 * plate_reflect(cos_q)
                      * np.maximum(cos_w2, 0.0) * float(ls.color[i].mean()), 0.0)
    return E


def hood_wall_irradiance_scatter(w: np.ndarray, n_in: np.ndarray, ls: LightSet, mat: Material,
                                 plate_rect_mm, step_mm: float = 3.0, chunk: int = 512) -> np.ndarray:
    """Irradiance (W/m^2) on hood points w (N, 3) (inward normals n_in) from LED light reflected
    once by the flat plate (Z = 0), integrated over the plate with the full BRDF (specular peak,
    GGX tail and diffuse): sum_q f(l_i, w - q) Phi_i(q) cos_q cos_w dA / |w - q|^2."""
    u0, v0, u1, v1 = plate_rect_mm
    uu = np.arange(u0 + step_mm / 2, u1, step_mm)
    vv = np.arange(v0 + step_mm / 2, v1, step_mm)
    U, V = np.meshgrid(uu, vv)
    q = plate_points(np.stack([U.ravel(), V.ravel()], -1))            # (M, 3)
    dA = (step_mm / 1000.0) ** 2
    nz = np.array([0.0, 0.0, 1.0])
    nq = np.repeat(nz[None], len(q), 0)
    l, Phi = incident(ls, q)                                          # (Nl, M, 3), (Nl, M)
    col = ls.color.mean(axis=1)
    E = np.zeros(len(w))
    for a in range(0, len(w), chunk):
        ww, nn = w[a:a + chunk], n_in[a:a + chunk]
        dvec = ww[:, None, :] - q[None, :, :]                          # (n, M, 3) plate -> wall
        r2 = np.maximum(np.einsum("nmk,nmk->nm", dvec, dvec), 1e-12)
        om = dvec / np.sqrt(r2)[..., None]
        cos_q = om[..., 2]
        cos_w = -np.einsum("nmk,nk->nm", om, nn)
        ok = (cos_q > 0) & (cos_w > 0)
        acc = np.zeros(ww.shape[0])
        for i in range(ls.n):
            li = np.broadcast_to(l[i][None], dvec.shape)
            f = brdf_cos(np.broadcast_to(nq[None], dvec.shape), li[None], om, mat, multiscatter=True)[0, ..., 0]
            # brdf_cos includes max(n . l, 0) for the incoming LED direction; f(l, om) is reciprocal
            Lq = f * Phi[i][None] * float(col[i])
            acc += np.sum(np.where(ok, Lq * cos_q * cos_w / r2, 0.0), axis=1) * dA
        E[a:a + chunk] = acc
    return E


@dataclass
class HoodEnv:
    """Hood radiance seen from one plate point, on a grid of mirror directions r = (r_x, r_y)
    (world X, Y components, |r| < 1), per LED group, prefiltered by the reflection lobe."""

    n: int
    maps: dict[str, np.ndarray]            # (n, n) W m^-2 sr^-1, rows r_y, cols r_x on linspace(-1, 1, n)
    irradiance: dict[str, float]           # cosine-weighted hood irradiance on a flat plate (W/m^2)


def hood_env(lights: Mapping[str, LightSet], hood: Mapping[str, Any], R, t, x0_world, *, base_luma: float,
             metallic: float, rough: float, plate_size_mm=None, n: int = 64, albedo: float | None = None,
             lobe_k: float = 2.0, scatter: bool = False, step_mm: float = 3.0) -> HoodEnv:
    """One-bounce hood environment at plate point x0 (docs/twin/PHYSICS_PINN.md 1.2).

    The hood interior (lighting.HOOD, albedo extras.hood.albedo, Lambertian) is lit by each LED
    group directly and by the LED light the plate reflects once (scatter=True: the plate's full
    BRDF integrated over its area, step_mm; scatter=False: only the LEDs' mirror images). Its
    radiance albedo E / pi is tabulated over the mirror directions of the plate point and blurred
    by the reflection lobe (sigma_r = lobe_k alpha). Second hood bounces (3 % of the first) are
    left out."""
    alb = float(hood.get("albedo", 0.03) if albedo is None else albedo)
    frustum, collar = hood_planes(hood, R, t)
    g = np.linspace(-1.0, 1.0, n)
    RX, RY = np.meshgrid(g, g)
    rr = RX ** 2 + RY ** 2
    inside = rr < 0.995
    dz = np.sqrt(np.clip(1.0 - rr, 1e-6, 1.0))
    d = np.stack([RX, RY, dz], -1).reshape(-1, 3)
    o = np.repeat(np.asarray(x0_world, float)[None], len(d), 0) + np.array([0.0, 0.0, 1e-5])
    tf, idf, pf = _exit(o, d, frustum)
    through_top = idf == len(frustum) - 1
    tc, idc, pc = _exit(pf, d, collar)
    hit = np.where(through_top[:, None], pc, pf)
    nrm = np.zeros_like(hit)
    for i, (_, nn, _) in enumerate(frustum):
        nrm[(idf == i) & ~through_top] = -nn
    for i, (_, nn, _) in enumerate(collar):
        nrm[(idc == i) & through_top] = -nn
    F0 = base_luma

    def plate_reflect(cos_q):
        return metallic * _schlick(F0, cos_q) + (1.0 - metallic) * fresnel_dielectric(cos_q)

    pmat = Material(np.array([base_luma]), float(rough), float(metallic), 0.0, None)
    rect = (0.0, 0.0, float(plate_size_mm[0]), float(plate_size_mm[1])) if plate_size_mm is not None else None
    alpha = max(rough * rough, 1e-3)
    sig_px = lobe_k * alpha / (2.0 / (n - 1))
    maps, irr = {}, {}
    dA = (2.0 / (n - 1)) ** 2                                  # projected solid angle per map cell
    sel = inside.ravel()
    for gname, ls in lights.items():
        if scatter and rect is not None:
            E = np.zeros(len(hit))
            E[sel] = hood_wall_irradiance(hit[sel], nrm[sel], LightSet(ls.pos, ls.axis, ls.power, ls.cos_half, ls.blend,
                                                                      ls.color, ls.names),
                                          lambda c: 0.0 * c, None)
            E[sel] += hood_wall_irradiance_scatter(hit[sel], nrm[sel], ls, pmat, rect, step_mm)
        else:
            E = hood_wall_irradiance(hit, nrm, ls, plate_reflect, plate_size_mm)
        L = (alb / math.pi * E).reshape(n, n) * inside
        irr[gname] = float(np.sum(L.ravel() * dA))
        if sig_px > 0.3:
            import cv2
            Lb = cv2.GaussianBlur(L.astype(np.float32), (0, 0), sig_px)
            wb = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), sig_px)
            L = np.where(inside, Lb / np.maximum(wb, 1e-6), 0.0)
        maps[gname] = L.astype(np.float32)
    return HoodEnv(n, maps, irr)


def env_lookup(env_map, r, xp=None):
    """Bilinear lookup of an (n, n) env map at mirror directions r (..., 3) (world X, Y components)."""
    xp = xp or ns(r)
    n = env_map.shape[0]
    if xp.is_torch:
        import torch.nn.functional as Fnn
        m = env_map if is_torch(env_map) else _torch.as_tensor(env_map, dtype=r.dtype)
        grid = r[..., :2].reshape(1, -1, 1, 2)
        out = Fnn.grid_sample(m[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out.reshape(r.shape[:-1])
    import cv2
    x = ((np.asarray(r[..., 0]) + 1.0) * (n - 1) / 2.0).astype(np.float32)
    y = ((np.asarray(r[..., 1]) + 1.0) * (n - 1) / 2.0).astype(np.float32)
    return cv2.remap(np.asarray(env_map, np.float32), x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def env_radiance(n, v, env_map, base, metallic, irradiance: float = 0.0, xp=None):
    """Radiance (...) the plate sends to the camera from the hood: specular part F(mu) x env(r)
    with r the view's mirror direction about n (split-sum approximation), plus the diffuse part
    (1 - metallic) base / pi x hood irradiance (flat-plate value)."""
    xp = xp or ns(n, v)
    nv = xp.maximum(_dot(n, v, xp), 1e-4)
    r = 2.0 * nv[..., None] * n - v
    Le = env_lookup(env_map, r, xp)
    F_m = base + (1.0 - base) * (1.0 - nv) ** 5
    F_d = fresnel_dielectric(nv, IOR_DIELECTRIC, xp)
    spec = metallic * F_m + (1.0 - metallic) * F_d
    return spec * Le + (1.0 - metallic) * (1.0 - F_d) * base / math.pi * irradiance
