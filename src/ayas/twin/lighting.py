"""Physically scaled ambient lighting, the clip-on hood, its LED ring and LED calibration (DESIGN_STEEL C, C2, D).

Device v2 (C2): in hood mode the white LEDs are the hood's low-angle (dark-field) ring,
ring_lights / ring_points (twin.yaml rig.ring), instead of the handheld's near-coaxial pods.

Units follow Blender (DESIGN_STEEL C): light power and strength are radiometric "W" with
683 lm/W, so E lux of white light is E/683 W/m^2. These formulas were checked against Cycles
5.2.3 renders of a Lambertian plane (point, spot, area, disk, sun, world all within 1 %):
- point or spot of power P: intensity P/(4 pi) W/sr. A spot multiplies it by
  smoothstep((cos a - cos(spot/2)) / ((1 - cos(spot/2)) * blend)), with no extra cosine;
- area light of power P and area A: one-sided Lambertian, radiance P/(pi A), emitting along dir;
- sun of strength S: S W/m^2 on a surface facing it;
- world of colour c and strength s: radiance c*s, so pi*s W/m^2 on an unoccluded plate.

Photometric lux weights a light's colour by its Rec.709 luma: lux = 683 * E_W * luma(colour).
Ambient powers are solved from the target lux this way, so the plate centre gets the target
photometric illuminance from the lights plus the world (unoccluded). LED irradiance
(led_irradiance) is power-based W/m^2 without the colour weighting, as burst.py needs it.

Frames (DESIGN 1): world metres, plate front face Z = 0, X = u/1000, Y = -v/1000; OpenCV camera
X_cam = R @ X_world + t. Plan light dicts follow DESIGN_STEEL D; area lights also carry x_dir
(their local +X, along size_m[0], in world) so Blender and this module agree on orientation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import frames as F

LM_PER_W = 683.0
LUMA = np.array([0.2126, 0.7152, 0.0722])
CONDITIONS = ("C0_dark", "C1_office", "C2_warm_bulb", "C3_window", "C4_side_lamp", "C5_reflected_lamp", "C6_sun")
# target lux at the plate centre (log-uniform), CCT (K), burst frame sigma and moving-shadow shift (mm)
SPEC: dict[str, dict[str, Any]] = {
    "C0_dark": {"lux": (0.0, 0.0), "cct": None, "sigma": (0.0, 0.0), "shift": (0.0, 0.0)},
    "C1_office": {"lux": (250.0, 700.0), "cct": (4000.0, 5000.0), "sigma": (0.0, 0.03), "shift": (0.0, 0.0)},
    "C2_warm_bulb": {"lux": (300.0, 3000.0), "cct": (2700.0, 3000.0), "sigma": (0.0, 0.01), "shift": (0.0, 0.0)},
    "C3_window": {"lux": (1500.0, 20000.0), "cct": (5500.0, 7000.0), "sigma": (0.0, 0.01), "shift": (0.0, 0.0)},
    # shift: a hand moving 0-15 mm/s at ~10 frames/s (full mode) moves its shadow edge 0-1.5 mm per
    # frame; the old 0-4 mm (40 mm/s, a waving hand) refused most C4 captures (review_post.md)
    "C4_side_lamp": {"lux": (1000.0, 8000.0), "cct": (3000.0, 6500.0), "sigma": (0.01, 0.01), "shift": (0.0, 1.5)},
    "C5_reflected_lamp": {"lux": (300.0, 1500.0), "cct": (4000.0, 6000.0), "sigma": (0.0, 0.02), "shift": (0.0, 0.0)},
    "C6_sun": {"lux": (30000.0, 100000.0), "cct": (5500.0, 6500.0), "sigma": (0.0, 0.01), "shift": (0.0, 0.0)},
}
WORLD_FRAC = (0.05, 0.15)            # share of the condition lux that comes from the world
DEVICE_BOX_MM = {"x": [-134.0, 134.0], "y": [-54.0, 80.0], "z": [-74.0, 0.0]}   # rig.device_box_mm default
HOOD = {"top_mm": [170.0, 160.0], "bottom_mm": [200.0, 190.0], "wall_mm": 2.0, "albedo": 0.03,
        "top_z_cam_mm": 17.0}
HOOD_BENCH_P = 0.7                   # the hood rests on the bench (when the plate fits inside it)
# device v2 (DESIGN_STEEL C2): the hood's low-angle (dark-field) white LED ring, twin.yaml rig.ring
RING = {"enabled": True, "n_leds": 18, "inset_mm": 6.0, "h_mm": 18.0}
TRIES = 200


# ----------------------------------------------------------------------------- units
def lux_to_w(lux: float) -> float:
    """Illuminance (lux, white light) -> irradiance in Blender W/m^2."""
    return float(lux) / LM_PER_W


def w_to_lux(e_w: float) -> float:
    return float(e_w) * LM_PER_W


def led_power_w(lumens: float) -> float:
    """Blender power of a Lambertian LED of this flux: I = lumens/pi cd on axis, power = 4 pi I / 683."""
    return 4.0 * math.pi * (float(lumens) / math.pi) / LM_PER_W


def luma(rgb) -> float:
    return float(np.dot(LUMA, np.asarray(rgb, float)[:3]))


def cct_rgb(kelvin: float) -> list[float]:
    """Linear RGB (max channel 1) of a black body (Tanner Helland's sRGB fit, as sampler.kelvin_rgb)."""
    t = float(kelvin) / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    srgb = np.clip(np.array([r, g, b]) / 255.0, 0.0, 1.0)
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return [round(float(c), 4) for c in lin / lin.max()]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _r(x: float, nd: int = 5) -> float:
    return round(float(x), nd)


def _sig(x: float, n: int = 6) -> float:
    """x rounded to n significant digits (powers span 1e-3 .. 1e4 W)."""
    x = float(x)
    return 0.0 if x == 0 else round(x, n - 1 - int(math.floor(math.log10(abs(x)))))


def _vec(v, nd: int = 6) -> list[float]:
    return [_r(x, nd) for x in v]


def _log_uniform(rng, lo: float, hi: float) -> float:
    return float(lo) if hi <= lo else float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


# ----------------------------------------------------------------------------- analytic irradiance
def spot_mask(cos_a: float, spot_deg: float, blend: float) -> float:
    """Cycles spot attenuation for a direction cos_a off the spot axis (1 inside, 0 outside the cone)."""
    ch = math.cos(math.radians(min(float(spot_deg), 180.0)) / 2.0)
    if cos_a <= ch:
        return 0.0
    if blend <= 0:
        return 1.0
    f = min(1.0, (cos_a - ch) / ((1.0 - ch) * float(blend)))
    return f * f * (3.0 - 2.0 * f)


def light_axes(light: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World axes (x, y, z) of a light: z = -dir (Blender emits along local -Z), x = x_dir made orthogonal."""
    z = -_unit(light.get("dir", (0.0, 0.0, -1.0)))
    xd = light.get("x_dir")
    x = np.asarray(xd if xd is not None else (1.0, 0.0, 0.0), float)
    x = x - np.dot(x, z) * z
    if np.linalg.norm(x) < 1e-6:                         # x_dir along the axis: any perpendicular
        x = np.cross(z, [0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0])
    x = _unit(x)
    return x, np.cross(z, x), z


def _area_samples(light: dict, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Midpoints (N, 3) and their areas (N,) of an area light, and its total area (m^2)."""
    x, y, _ = light_axes(light)
    c = np.asarray(light["pos_m"], float)
    w, h = (float(s) for s in (list(light.get("size_m") or [0.1]) * 2)[:2])
    if str(light.get("shape", "RECTANGLE")).upper() in ("DISK", "ELLIPSE"):
        if str(light.get("shape")).upper() == "DISK":
            h = w
        rr = (np.arange(k) + 0.5) / k
        ring = math.pi * ((np.arange(k) + 1) ** 2 - np.arange(k) ** 2) / k ** 2 * (w / 2) * (h / 2)
        m = 4 * k
        th = 2 * math.pi * (np.arange(m) + 0.5) / m
        a = (rr[:, None] * np.cos(th)[None] * w / 2).ravel()
        b = (rr[:, None] * np.sin(th)[None] * h / 2).ravel()
        dA = np.repeat(ring / m, m)
        area = math.pi * w * h / 4
    else:
        g = (np.arange(k) + 0.5) / k - 0.5
        a, b = (q.ravel() for q in np.meshgrid(g * w, g * h))
        dA = np.full(a.size, w * h / k ** 2)
        area = w * h
    return c + a[:, None] * x + b[:, None] * y, dA, area


def light_irradiance_w(light: dict, point, normal=(0.0, 0.0, 1.0), k: int = 16) -> float:
    """Unoccluded white-equivalent irradiance (W/m^2) of one plan light at a point with this normal."""
    p = np.asarray(point, float)
    n = _unit(normal)
    kind = str(light.get("kind", "spot")).lower()
    if kind == "sun":
        return float(light.get("strength", 0.0)) * max(0.0, -float(np.dot(_unit(light["dir"]), n)))
    P = float(light.get("power_w", 0.0))
    if kind == "area":
        q, dA, area = _area_samples(light, k)
        v = q - p
        r = np.linalg.norm(v, axis=1)
        cos_s = (v @ n) / r
        cos_l = -(v @ _unit(light["dir"])) / r
        ok = (cos_s > 0) & (cos_l > 0)
        return float(np.sum((P / (math.pi * area)) * cos_s[ok] * cos_l[ok] * dA[ok] / r[ok] ** 2))
    v = np.asarray(light["pos_m"], float) - p
    d = float(np.linalg.norm(v))
    cos_s = float(np.dot(v, n)) / d
    if cos_s <= 0:
        return 0.0
    mask = 1.0
    if kind == "spot":
        mask = spot_mask(float(np.dot(_unit(light["dir"]), -v / d)), light.get("spot_deg", 180.0),
                         light.get("blend", 0.0))
    return P / (4.0 * math.pi * d * d) * cos_s * mask


def lights_lux(lights: list[dict], point, normal=(0.0, 0.0, 1.0)) -> float:
    """Photometric illuminance (lux) of plan lights at a point, unoccluded."""
    return sum(LM_PER_W * light_irradiance_w(L, point, normal) * luma(L.get("color", (1, 1, 1))) for L in lights)


def world_lux(ambient: dict) -> float:
    """Illuminance (lux) of the uniform world on an unoccluded upward plate."""
    c = ambient.get("world_color") or (0.0, 0.0, 0.0)
    return LM_PER_W * math.pi * float(ambient.get("world_strength", 0.0)) * luma(c)


def ambient_lux(ambient: dict, point, normal=(0.0, 0.0, 1.0)) -> float:
    """Ambient illuminance (lux) at a point: lights plus world, ignoring occluders, device and hood."""
    return lights_lux(ambient.get("lights") or [], point, normal) + world_lux(ambient)


# ----------------------------------------------------------------------------- plan geometry
def plate_centre(plan: dict) -> np.ndarray:
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    return F.plate_to_world([[W / 2, H / 2]])[0]


def _pose(plan: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, t, camera centre); a plan without a camera gets a straight-down view from 200 mm."""
    cam = plan.get("camera") or {}
    if cam.get("R") is not None:
        R, t = np.asarray(cam["R"], float), np.asarray(cam["t"], float)
    else:
        W, H = (float(x) for x in plan["plate"]["size_mm"])
        R, t = F.camera_pose([W / 2, H / 2], 200.0)
    return R, t, F.camera_centre(R, t)


def axis_point(plan: dict, z_m: float = 0.0) -> np.ndarray:
    """Where the optical axis meets the plane Z = z_m (the plate front face by default)."""
    R, t, C = _pose(plan)
    a = R[2]
    if a[2] > -1e-9:
        return plate_centre(plan)
    return C + (z_m - C[2]) / a[2] * a


def led_irradiance(plan: dict, groups=("L", "R"), point=None) -> float:
    """Analytic LED irradiance (W/m^2, Blender units, no colour weighting) at the optical-axis plate point."""
    p = axis_point(plan) if point is None else np.asarray(point, float)
    return float(sum(light_irradiance_w(L, p) for L in plan.get("lights") or [] if L.get("group") in groups))


def _device_box(cfg: dict | None) -> tuple[np.ndarray, np.ndarray]:
    box = ((cfg or {}).get("rig") or {}).get("device_box_mm") or DEVICE_BOX_MM
    lo = np.array([min(box[k]) for k in "xyz"], float) / 1000.0
    hi = np.array([max(box[k]) for k in "xyz"], float) / 1000.0
    return lo, hi


def _slab(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray, tmax: float) -> bool:
    """Does the ray o + s d, 0 < s < tmax, cross the axis-aligned box [lo, hi]?"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t0, t1 = (lo - o) * inv, (hi - o) * inv
    near = np.nan_to_num(np.minimum(t0, t1), nan=-np.inf)
    far = np.nan_to_num(np.maximum(t0, t1), nan=np.inf)
    for i in range(3):                                  # ray parallel to a slab and outside it
        if d[i] == 0 and not (lo[i] <= o[i] <= hi[i]):
            return False
    a, b = max(near.max(), 1e-9), min(far.min(), tmax)
    return a < b


def device_blocks(plan: dict, cfg: dict | None, point, direction, tmax: float = np.inf) -> bool:
    """Does the device box (camera frame) block the ray from a world point along a world direction?"""
    R, t, _ = _pose(plan)
    lo, hi = _device_box(cfg)
    return _slab(R @ np.asarray(point, float) + t, R @ _unit(direction), lo, hi, tmax)


def _blocked_to(plan, cfg, point, light: dict) -> bool:
    if light["kind"] == "sun":
        return device_blocks(plan, cfg, point, -np.asarray(light["dir"], float))
    v = np.asarray(light["pos_m"], float) - point
    return device_blocks(plan, cfg, point, v, float(np.linalg.norm(v)))


def euler_xyz_deg(M: np.ndarray) -> list[float]:
    """Blender XYZ Euler (degrees) of a rotation matrix M = Rz @ Ry @ Rx (columns = local axes in world)."""
    M = np.asarray(M, float)
    b = math.asin(max(-1.0, min(1.0, -M[2, 0])))
    a = math.atan2(M[2, 1], M[2, 2])
    g = math.atan2(M[1, 0], M[0, 0])
    return [_r(math.degrees(x), 6) for x in (a, b, g)]


def euler_xyz_matrix(rot_deg) -> np.ndarray:
    a, b, g = (math.radians(float(x)) for x in rot_deg)
    return F._rz(g) @ F._ry(b) @ F._rx(a)


def occluder_corners(occ: dict) -> np.ndarray:
    """(8, 3) world corners of a box occluder (center_m, size_m, rot_deg)."""
    M = euler_xyz_matrix(occ.get("rot_deg", (0.0, 0.0, 0.0)))
    h = np.asarray(occ["size_m"], float) / 2
    loc = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    return np.asarray(occ["center_m"], float) + loc @ M.T


def occluder_blocks(occ: dict, point, direction, tmax: float = np.inf) -> bool:
    M = euler_xyz_matrix(occ.get("rot_deg", (0.0, 0.0, 0.0)))
    h = np.asarray(occ["size_m"], float) / 2
    o = M.T @ (np.asarray(point, float) - np.asarray(occ["center_m"], float))
    return _slab(o, M.T @ _unit(direction), -h, h, tmax)


def shadowed(ambient: dict, point, light: dict) -> bool:
    """Is a plate point in the shadow of an ambient occluder for this light?"""
    if light["kind"] == "sun":
        d, tmax = -np.asarray(light["dir"], float), np.inf
    else:
        v = np.asarray(light["pos_m"], float) - np.asarray(point, float)
        d, tmax = v, float(np.linalg.norm(v))
    return any(occluder_blocks(o, point, d, tmax) for o in ambient.get("occluders") or [])


# ----------------------------------------------------------------------------- conditions
def pick_condition(rng, cfg: dict | None) -> str:
    w = ((cfg or {}).get("sampler") or {}).get("conditions") or {c: 1.0 for c in CONDITIONS}
    keys = [k for k in w if k in CONDITIONS]
    p = np.array([w[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _dir_from(az_deg: float, el_deg: float) -> np.ndarray:
    """World unit vector at azimuth az (from +X toward +Y) and elevation el above the plate."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _tinted(rng, kelvin: float, white_mix: float) -> list[float]:
    c = (1 - white_mix) * np.asarray(cct_rgb(kelvin)) + white_mix
    c = c * (1 + rng.normal(0.0, 0.01, 3))
    return _vec(np.clip(c / c.max(), 0.0, 1.0), 4)


def _world(rng, cond: str, cct: float | None, e_lux: float) -> tuple[list[float], float]:
    """World colour and strength giving e_lux on an unoccluded plate."""
    if cond == "C0_dark" or e_lux <= 0:
        return [0.0, 0.0, 0.0], 0.0
    if cond == "C6_sun":
        color = _tinted(rng, rng.uniform(10000, 16000), 0.0)      # blue sky
    elif cond == "C3_window":
        color = _tinted(rng, rng.uniform(7000, 10000), 0.3)       # sky through the window + room
    else:
        color = _tinted(rng, cct or 4500.0, 0.5)                  # room bounce, a little warm
    return color, _sig(e_lux / (LM_PER_W * math.pi * luma(color)))


def _solve(light: dict, point, e_lux: float) -> dict:
    """Scale a light's power (or sun strength) so that it gives e_lux at the point."""
    key = "strength" if light["kind"] == "sun" else "power_w"
    light[key] = 1.0
    per = LM_PER_W * light_irradiance_w(light, point) * luma(light["color"])
    if per <= 0:
        raise ValueError(f"{light['name']} does not light the plate centre")
    light[key] = _sig(e_lux / per)
    light["lux"] = _r(e_lux, 2)
    return light


def _area(name, pos, direction, x_dir, size, color, shape="RECTANGLE") -> dict:
    return {"name": name, "group": "AMB", "kind": "area", "pos_m": _vec(pos), "dir": _vec(_unit(direction)),
            "x_dir": _vec(_unit(x_dir)), "size_m": [_r(s, 4) for s in size], "shape": shape, "power_w": 0.0,
            "color": color}


def _office(rng, plan: dict, color, e_lux: float, n_range=(2, 6)) -> list[dict]:
    """2-6 identical ceiling panels or tubes 2.2-3 m up on a jittered grid; equal powers giving e_lux."""
    p0 = plate_centre(plan)
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    tube = bool(rng.random() < 0.5)
    size = [1.2, 0.08] if tube else [0.6, 0.6]
    phi = rng.uniform(0, math.pi)
    ex, ey = np.array([math.cos(phi), math.sin(phi), 0.0]), np.array([-math.sin(phi), math.cos(phi), 0.0])
    s = rng.uniform(1.2, 2.4)
    off = rng.uniform(0, s, 2)
    grid = [(i * s + off[0] - 3 * s, j * s + off[1] - 3 * s) for i in range(7) for j in range(7)]
    grid.sort(key=lambda g: math.hypot(*g))
    h = rng.uniform(2.2, 3.0)
    lights = []
    for k, (a, b) in enumerate(grid[:n]):
        pos = p0 + a * ex + b * ey + np.array([0.0, 0.0, h + rng.uniform(-0.02, 0.02)])
        lights.append(_area(f"AMB_{'tube' if tube else 'panel'}_{k}", pos, (0, 0, -1), ex, size, color))
    for L in lights:
        L["power_w"] = 1.0
    per = lights_lux(lights, p0)
    for L in lights:
        L["power_w"] = _sig(e_lux / per)
        L["lux"] = _r(lights_lux([L], p0), 2)
    return lights


def _bulbs(rng, plan, cfg, color, e_lux) -> tuple[list[dict], bool]:
    p0 = plate_centre(plan)
    n = int(rng.integers(1, 3))
    w = float(rng.uniform(0.55, 0.9))
    share = [1.0] if n == 1 else [w, 1.0 - w]
    out, blocked = [], False
    for k in range(n):
        for _ in range(TRIES):
            d, el, az = rng.uniform(0.4, 1.2), rng.uniform(20, 55), rng.uniform(0, 360)
            L = {"name": f"AMB_bulb_{k}", "group": "AMB", "kind": "point",
                 "pos_m": _vec(p0 + d * _dir_from(az, el)), "dir": [0.0, 0.0, -1.0],
                 "radius_m": _r(rng.uniform(0.02, 0.04), 4), "power_w": 0.0, "color": color}
            if not _blocked_to(plan, cfg, p0, L):
                break
        else:
            blocked = True
        out.append(_solve(L, p0, share[k] * e_lux))
    return out, blocked


def _window(rng, plan, cfg, color, e_lux) -> tuple[list[dict], bool]:
    p0 = plate_centre(plan)
    blocked = True
    for _ in range(TRIES):
        d, az = rng.uniform(1.5, 4.0), rng.uniform(0, 360)
        el = rng.uniform(max(15.0, math.degrees(math.asin(min(1.0, 0.7 / d)))), 50.0)
        u = _dir_from(az, el)
        horiz = np.array([u[0], u[1], 0.0])
        L = _area("AMB_window", p0 + d * u, -horiz, (-u[1], u[0], 0.0), [1.5, 1.2], color)
        if not _blocked_to(plan, cfg, p0, L):
            blocked = False
            break
    return [_solve(L, p0, e_lux)], blocked


def _sun(rng, plan, cfg, color, e_lux) -> tuple[list[dict], bool]:
    p0 = plate_centre(plan)
    blocked = True
    for _ in range(TRIES):
        el, az = rng.uniform(20, 70), rng.uniform(0, 360)
        u = _dir_from(az, el)
        L = {"name": "AMB_sun", "group": "AMB", "kind": "sun", "pos_m": _vec(p0 + 10.0 * u), "dir": _vec(-u),
             "strength": 0.0, "angle_deg": 0.53, "color": color}
        if not _blocked_to(plan, cfg, p0, L):
            blocked = False
            break
    return [_solve(L, p0, e_lux)], blocked


def _view_rays(plan: dict, n: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Camera centre and plate points (n x n grid, corners included) the camera must see."""
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    g = np.stack(np.meshgrid(np.linspace(0, W, n), np.linspace(0, H, n)), -1).reshape(-1, 2)
    return _pose(plan)[2], F.plate_to_world(g)


def _occluder_ok(occ: dict, plan: dict, cfg: dict | None) -> bool:
    """Above the plate, outside the camera's view of the plate, off the white LEDs' paths to the
    plate (L/R must be the same render under every condition) and clear of the device box."""
    corners = occluder_corners(occ)
    if corners[:, 2].min() < 0.03:
        return False
    C, pts = _view_rays(plan)
    for p in pts:
        v = p - C
        if occluder_blocks(occ, C, v, float(np.linalg.norm(v))):
            return False
    for L in plan.get("lights") or []:
        if L.get("group") in ("L", "R"):
            for p in pts:
                v = p - np.asarray(L["pos_m"], float)
                if occluder_blocks(occ, L["pos_m"], v, float(np.linalg.norm(v))):
                    return False
    R, t, _ = _pose(plan)
    lo, hi = _device_box(cfg)
    lo, hi = lo - 0.02, hi + 0.02                         # 2 cm clearance
    M = euler_xyz_matrix(occ["rot_deg"])
    h = np.asarray(occ["size_m"], float) / 2
    grid = np.stack(np.meshgrid(*(np.linspace(-1, 1, k) for k in (5, 5, 9)), indexing="ij"), -1).reshape(-1, 3)
    pw = np.asarray(occ["center_m"], float) + (grid * h) @ M.T
    pc = pw @ R.T + t
    if np.any(np.all((pc >= lo) & (pc <= hi), axis=1)):
        return False
    hood = _hood_of(plan)
    if hood is not None:                                  # never through the hood's walls
        c = hood_corners(plan, hood)
        q = (np.vstack([c["outer_top"], c["outer_bottom"]]) - np.asarray(occ["center_m"], float)) @ M
        if np.any(in_hood(plan, pw, hood, margin_m=0.01)) or np.any(np.all(np.abs(q) <= h, axis=1)):
            return False
    box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    q = ((box - t) @ R - np.asarray(occ["center_m"], float)) @ M          # device corners in box frame
    return not np.any(np.all(np.abs(q) <= h, axis=1))


def _side_lamp(rng, plan, cfg, color, e_lux) -> tuple[list[dict], list[dict], bool]:
    """Work lamp (spot 30-60 deg) at 10-30 deg elevation, 0.4-1 m, and an arm whose shadow edge crosses the plate."""
    p0 = plate_centre(plan)
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    best = None                                          # (lamp, arm, lamp unblocked, arm usable)
    for _ in range(TRIES):
        d, el, az = rng.uniform(0.4, 1.0), rng.uniform(10, 30), rng.uniform(0, 360)
        S = p0 + d * _dir_from(az, el)
        L = {"name": "AMB_lamp", "group": "AMB", "kind": "spot", "pos_m": _vec(S), "dir": _vec(_unit(p0 - S)),
             "radius_m": _r(rng.uniform(0.005, 0.015), 4), "spot_deg": _r(rng.uniform(30, 60), 3),
             "blend": _r(rng.uniform(0.15, 0.3), 3), "power_w": 0.0, "color": color}
        occ = _arm(rng, S, W, H)
        lamp_ok = not _blocked_to(plan, cfg, p0, L)
        arm_ok = lamp_ok and _occluder_ok(occ, plan, cfg)
        if best is None or (lamp_ok and not best[2]) or arm_ok:
            best = (L, occ, lamp_ok, arm_ok)
        if arm_ok:
            break
    L, occ, lamp_ok, arm_ok = best
    # an arm that would hide the plate from the camera, shade the LEDs or cut the device or hood is dropped
    return [_solve(L, p0, e_lux)], ([occ] if arm_ok else []), not lamp_ok


def _arm(rng, S: np.ndarray, W: float, H: float) -> dict:
    """A box 'arm' between the lamp S and a plate point Q whose shadow edge runs through Q.

    One face of the box lies in the vertical plane through S and Q, so it projects (from S) to the
    line on the plate through Q along the lamp's horizontal direction: a straight shadow edge
    across the plate, shadow on one side. The box sits 0.25-0.5 m from Q along the ray, kept
    within 0.35-0.6 of the lamp distance (a lamp at 10-30 deg and 0.4-1 m is at most 0.5 m high,
    so the arm cannot also be 0.25-0.5 m above the plate); its 8-12 cm depth across the ray
    makes the edge long. With the 1-3 cm lamp the penumbra is about 0.5-3 cm wide.
    """
    Q = F.plate_to_world([[rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H]])[0]
    ur = _unit(S - Q)
    side = _unit(np.cross(ur, [0.0, 0.0, 1.0])) * (1.0 if rng.random() < 0.5 else -1.0)
    wax = np.cross(ur, side)                                          # right-handed (ur, side, wax)
    size = np.array([rng.uniform(0.18, 0.26), rng.uniform(0.07, 0.10), rng.uniform(0.08, 0.12)])
    dist = float(np.linalg.norm(S - Q))
    s = min(max(rng.uniform(0.25, 0.5), 0.35 * dist), 0.6 * dist)     # distance from Q along the ray
    centre = Q + s * ur + side * size[1] / 2 + wax * rng.uniform(-0.3, 0.3) * size[2]
    M = np.stack([ur, side, wax], axis=1)                             # columns: local x, y, z in world
    return {"kind": "box", "center_m": _vec(centre), "size_m": _vec(size, 4), "rot_deg": euler_xyz_deg(M),
            "albedo": _r(rng.uniform(0.25, 0.5), 3), "edge_uv_mm": _vec(F.world_to_plate([Q])[0][:2], 3),
            "edge_dir_deg": _r(math.degrees(math.atan2(-ur[1], ur[0])), 3)}


def _mirror_point(rng, plan: dict, cfg: dict | None) -> tuple[np.ndarray, np.ndarray, bool]:
    """A plate point whose mirror reflection (for the camera) clears the device box, and the reflected direction."""
    C = _pose(plan)[2]
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    uu, vv = np.meshgrid(np.linspace(0.1 * W, 0.9 * W, 17), np.linspace(0.1 * H, 0.9 * H, 17))
    pts = F.plate_to_world(np.c_[uu.ravel(), vv.ravel()])
    ok = []
    for p in pts:
        r = p - C
        r = _unit([r[0], r[1], -r[2]])
        if not device_blocks(plan, cfg, p, r):
            ok.append((p, r))
    if ok:
        p, r = ok[int(rng.integers(len(ok)))]
        return p, r, False
    p = pts[int(rng.integers(len(pts)))]
    r = p - C
    return p, _unit([r[0], r[1], -r[2]]), True


def _glare_panel(rng, plan, cfg, color, e_max: float) -> tuple[dict, list[float], bool]:
    """A bright panel (5000-40000 cd/m^2: tube to bare lamp) on the mirror path of a plate point, facing it.

    Its luminance is what makes the glare, so it is kept; its size and distance along the
    mirror ray (the image stays on the same plate point) keep its lux at the centre <= e_max.
    """
    p0 = plate_centre(plan)
    Q, r, blocked = _mirror_point(rng, plan, cfg)
    lum = _log_uniform(rng, 5000.0, 40000.0)               # panel luminance, cd/m^2
    disk = rng.random() < 0.3
    k_area = math.pi / 4 if disk else 1.0
    D = rng.uniform(1.0, 2.5)
    side = max(0.1, min(rng.uniform(0.2, 0.5), math.sqrt(e_max * D * D / (lum * k_area))))
    x_dir = np.cross(r, [0.0, 0.0, 1.0]) if abs(r[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    w_lum = lum / (LM_PER_W * luma(color))                  # radiance, W/m^2/sr
    for _ in range(40):
        L = _area("AMB_glare", Q + D * r, -r, x_dir, [side, side], color, shape="DISK" if disk else "RECTANGLE")
        L["power_w"] = _sig(w_lum * math.pi * side * side * k_area)
        lux = lights_lux([L], p0)
        if lux <= e_max:
            break
        D *= 1.15
    L["lux"] = _r(lux, 2)
    L["luminance_cd_m2"] = _r(L["power_w"] / (math.pi * side * side * k_area) * LM_PER_W * luma(color), 1)
    return L, _vec(F.world_to_plate([Q])[0][:2], 3), blocked


def ambient_for(rng, plan: dict, cfg: dict | None, condition: str | None = None) -> dict:
    """plan['ambient'] for one lighting condition (DESIGN_STEEL C, D); drawn from cfg weights if None.

    Lights are placed relative to the plate and the camera; their powers are solved so that the
    plate centre gets the target lux (lights + world, unoccluded). Directional conditions keep
    their lamp, window or sun unblocked by the device box at the plate centre when they can
    (device_shadow says when they could not).
    """
    cond = condition or pick_condition(rng, cfg)
    if cond not in SPEC:
        raise ValueError(f"unknown lighting condition {cond!r}; expected one of {CONDITIONS}")
    sp = SPEC[cond]
    lux = _log_uniform(rng, *sp["lux"]) if sp["lux"][1] > 0 else 0.0
    cct = float(rng.uniform(*sp["cct"])) if sp["cct"] else None
    sigma = float(rng.uniform(*sp["sigma"]))
    shift = float(rng.uniform(*sp["shift"])) if sp["shift"][1] > 0 else 0.0
    out: dict[str, Any] = {"condition": cond, "lux": _r(lux, 2), "cct_k": _r(cct, 1) if cct else None,
                           "world_color": [0.0, 0.0, 0.0], "world_strength": 0.0, "lights": [], "occluders": [],
                           "change": {"frame_sigma": _r(sigma, 4), "shadow_shift_mm": _r(shift, 3)},
                           "device_shadow": False}
    if cond == "C0_dark":
        out["world_lux"] = 0.0
        return out
    frac = float(rng.uniform(*WORLD_FRAC))
    color = _tinted(rng, cct, 0.0)
    out["world_color"], out["world_strength"] = _world(rng, cond, cct, frac * lux)
    e = (1.0 - frac) * lux
    blocked = False
    if cond == "C1_office":
        out["lights"] = _office(rng, plan, color, e)
    elif cond == "C2_warm_bulb":
        out["lights"], blocked = _bulbs(rng, plan, cfg, color, e)
    elif cond == "C3_window":
        out["lights"], blocked = _window(rng, plan, cfg, color, e)
    elif cond == "C4_side_lamp":
        out["lights"], out["occluders"], blocked = _side_lamp(rng, plan, cfg, color, e)
    elif cond == "C5_reflected_lamp":
        glare, uv, mblocked = _glare_panel(rng, plan, cfg, color, 0.6 * e)
        out["lights"] = _office(rng, plan, color, e - glare["lux"], n_range=(2, 4)) + [glare]
        out["mirror_uv_mm"], out["mirror_blocked"] = uv, mblocked
    elif cond == "C6_sun":
        out["lights"], blocked = _sun(rng, plan, cfg, color, e)
    out["device_shadow"] = bool(blocked)
    out["world_lux"] = _r(world_lux(out), 2)
    return out


# ----------------------------------------------------------------------------- hood
def hood_corners(plan: dict, hood: dict | None = None, z_m: float | None = None) -> dict[str, np.ndarray]:
    """World corners (4, 3 each) of the hood frustum: inner/outer at the top and on the plane Z = z_m.

    Built in the camera frame: the top rectangle (top_mm, centred on the optical axis) at
    z_cam = top_z_cam_mm, the edges running to the bottom_mm rectangle at the depth where the axis
    meets the bottom plane, intersected with that plane (default Z = gap_mm / 1000). Outer
    rectangles are wall_mm larger on every side. Corner order: (-x,-y), (+x,-y), (+x,+y), (-x,+y)
    in the camera frame. render_scene.py builds the same mesh.
    """
    h = dict(HOOD, **(hood or (plan.get("extras") or {}).get("hood") or {}))
    R, t, C = _pose(plan)
    zb = float(h.get("gap_mm", 0.0)) / 1000.0 if z_m is None else float(z_m)
    zt = float(h.get("top_z_cam_mm", HOOD["top_z_cam_mm"])) / 1000.0
    a = R[2]
    depth = (zb - C[2]) / a[2]
    wall = float(h["wall_mm"]) / 1000.0
    out = {}
    for name, grow in (("inner", 0.0), ("outer", 2 * wall)):
        wt, ht = (float(x) / 1000.0 + grow for x in h["top_mm"])
        wb, hb = (float(x) / 1000.0 + grow for x in h["bottom_mm"])
        top, bot = [], []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            T = C + R.T @ np.array([sx * wt / 2, sy * ht / 2, zt])
            B = C + R.T @ np.array([sx * wb / 2, sy * hb / 2, depth])
            top.append(T)
            bot.append(T + (zb - T[2]) / (B[2] - T[2]) * (B - T))
        out[f"{name}_top"], out[f"{name}_bottom"] = np.array(top), np.array(bot)
    return out


def _hood_of(plan: dict) -> dict | None:
    """The plan's hood dict when Blender builds one (render_scene.build_hood), else None."""
    ex = plan.get("extras") or {}
    hood = ex.get("hood")
    if not isinstance(hood, dict) or (ex.get("device") or {}).get("mode", "hood") == "open":
        return None
    return hood


def in_hood(plan: dict, pts, hood: dict | None = None, margin_m: float = 0.0) -> np.ndarray:
    """Which world points (N, 3) lie inside the hood's outer frustum (top to bottom plane), grown by margin_m?"""
    c = hood_corners(plan, hood)
    top, bot = c["outer_top"], c["outer_bottom"]
    mid = np.vstack([top, bot]).mean(axis=0)
    p = np.atleast_2d(np.asarray(pts, float))
    inside = np.ones(len(p), bool)
    for f in [top, bot] + [np.array([top[k], top[(k + 1) % 4], bot[(k + 1) % 4]]) for k in range(4)]:
        n = _unit(np.cross(f[1] - f[0], f[2] - f[0]))
        if np.dot(mid - f[0], n) > 0:                     # outward normal
            n = -n
        inside &= (p - f[0]) @ n <= margin_m
    return inside


def _inside(poly: np.ndarray, pts: np.ndarray, margin: float) -> bool:
    """All pts (N, 2) inside the convex polygon poly (4, 2) by at least margin."""
    c = poly.mean(axis=0)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        n = np.array([b[1] - a[1], a[0] - b[0]])
        n = n / np.linalg.norm(n)
        if np.dot(c - a, n) < 0:
            n = -n
        if np.any((pts - a) @ n < margin):
            return False
    return True


def plate_fits_hood(plan: dict, hood: dict | None = None, margin_mm: float = 1.0) -> bool:
    """Does the plate lie inside the hood's inner opening (at its front face and at the bench)?"""
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    t = float(plan["plate"]["thickness_mm"]) / 1000.0
    pts = F.plate_to_world([[0, 0], [W, 0], [W, H], [0, H]])[:, :2]
    return all(_inside(hood_corners(plan, hood, z)["inner_bottom"][:, :2], pts, margin_mm / 1000.0)
               for z in (0.0, -t))


def hood_for(rng, plan: dict, cfg: dict | None, device_mode: str | None) -> dict | None:
    """plan['extras']['hood'] for device_mode 'hood' (None otherwise), DESIGN_STEEL C.

    gap_mm is the height of the hood's bottom rim above the plate front face (bottom plane
    Z = gap_mm / 1000). When the plate fits inside the opening the hood rests on the bench 70 %
    of the time: gap_mm = -thickness (rim on the bench, sealed). Otherwise it rests on the part
    (or is held just above it): gap_mm 0.02-1 mm, and ambient leaks in under the rim around the plate.
    """
    if (device_mode or "open") != "hood":
        return None
    hood = {k: (list(v) if isinstance(v, list) else v) for k, v in HOOD.items()}
    hood.update((cfg or {}).get("hood") or {})
    t = float(plan["plate"]["thickness_mm"])
    fits = plate_fits_hood(plan, dict(hood, gap_mm=-t))
    if fits and rng.random() < HOOD_BENCH_P:
        hood["gap_mm"], hood["rests_on"] = _r(-t, 4), "bench"
    else:
        hood["gap_mm"], hood["rests_on"] = _r(rng.uniform(0.02, 1.0), 4), "part"
    hood["plate_fits"] = bool(fits)
    return hood


# ----------------------------------------------------------------------------- device v2: the LED ring
def ring_config(cfg: dict | None) -> dict:
    """twin.yaml rig.ring over the RING defaults (enabled, n_leds, inset_mm, h_mm)."""
    return {**RING, **(((cfg or {}).get("rig") or {}).get("ring") or {})}


def illumination_for(cfg: dict | None, device_mode: str | None) -> str:
    """'ring' for the hood when rig.ring is enabled (device v2), else 'pods' (the handheld's pods)."""
    return "ring" if (device_mode or "open") == "hood" and bool(ring_config(cfg)["enabled"]) else "pods"


def _inset_quad(quad: np.ndarray, inset: float) -> np.ndarray:
    """A convex quadrilateral (4, 2) with every edge moved inward by inset (same units)."""
    c = quad.mean(axis=0)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        n = np.array([b[1] - a[1], a[0] - b[0]], float)
        n /= np.linalg.norm(n)
        if np.dot(c - a, n) < 0:
            n = -n
        lines.append((a + inset * n, b - a))
    out = []
    for i in range(4):                                    # corner i joins edges i - 1 and i
        (p, u), (q, v) = lines[i - 1], lines[i]
        s = np.linalg.solve(np.array([u, -v]).T, q - p)[0]
        out.append(p + s * u)
    return np.array(out)


def ring_points(plan: dict, cfg: dict | None = None, hood: dict | None = None) -> np.ndarray:
    """World positions (N, 3) of the ring LEDs (DESIGN_STEEL C2).

    The ring lies in the plane h_mm above the plate front face (Z = h_mm / 1000), on the hood's
    inner cross-section there shrunk by inset_mm (the LED bodies sit on the inner walls). The
    n_leds are spread evenly along that perimeter, starting half a spacing from the middle of
    the hood's -y edge (image top) towards +x, so the ring is mirror-symmetric about the
    camera's y-z plane and n_leds / 2 sit on each image half.
    """
    rc = ring_config(cfg)
    h = hood if hood is not None else ((plan.get("extras") or {}).get("hood") or {})
    z = float(rc["h_mm"]) / 1000.0
    quad = hood_corners(plan, h, z)["inner_bottom"][:, :2]
    quad = _inset_quad(quad, float(rc["inset_mm"]) / 1000.0)
    n = int(rc["n_leds"])
    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    lens = np.array([np.linalg.norm(b - a) for a, b in edges])
    per = float(lens.sum())
    start = 0.5 * lens[0]                                  # middle of edge 0: corners (-x,-y) -> (+x,-y)
    pts = []
    for k in range(n):
        s = (start + (k + 0.5) * per / n) % per
        i = 0
        while s > lens[i] and i < 3:
            s -= lens[i]
            i += 1
        a, b = edges[i]
        pts.append([*(a + (b - a) * min(s / lens[i], 1.0)), z])
    return np.array(pts)


def ring_lights(plan: dict, cfg: dict | None = None, white: dict | None = None,
                power_scale: dict[str, float] | None = None, hood: dict | None = None) -> list[dict]:
    """Plan 'lights' entries of the hood's dark-field LED ring (DESIGN_STEEL C2, D).

    Each LED is a white spot like the pods' (rig.white: Lambertian, spot_deg, blend, radius,
    power from capture.yaml lumens) at ring_points, aimed at the plate point on the optical
    axis, so light arrives at about atan(h / distance) = 8-12 deg elevation: flat specular
    metal mirrors it away from the camera (no LED highlights), while scratches, dents, pits,
    rust and stains scatter it into the lens. Group L = LEDs with camera-frame x < 0 (the
    image's left half), R = the others; power_scale 'white_L' / 'white_R' multiply power_w
    as rig.lights_world does for the pods.
    """
    w = dict(white or ((cfg or {}).get("rig") or {}).get("white") or {})
    scale = power_scale or {}
    R, t, _ = _pose(plan)
    target = axis_point(plan)
    out, idx = [], {"L": 0, "R": 0}
    for p in ring_points(plan, cfg, hood):
        g = "L" if float((R @ p + t)[0]) < 0 else "R"
        out.append({"name": f"WL_{g}_{idx[g]}", "group": g, "kind": w.get("kind", "spot"), "mount": "ring",
                    "pos_m": [round(float(v), 7) for v in p],
                    "dir": [round(float(v), 7) for v in _unit(target - p)],
                    "radius_m": round(float(w.get("radius_mm", 2.0)) / 1000.0, 6),
                    "spot_deg": w.get("spot_deg", 150), "blend": w.get("blend", 0.6),
                    "power_w": round(float(w.get("power_w", 0.35)) * scale.get(f"white_{g}", 1.0), 5),
                    "color": [float(c) for c in w.get("color", (1.0, 1.0, 1.0))]})
        idx[g] += 1
    return out


def ring_elevations_deg(plan: dict, lights: list[dict] | None = None) -> np.ndarray:
    """Elevation (deg above the plate) of each ring LED seen from the optical-axis plate point."""
    p0 = axis_point(plan)
    L = [x for x in (lights if lights is not None else plan.get("lights") or []) if x.get("mount") == "ring"]
    v = np.array([np.asarray(x["pos_m"], float) - p0 for x in L]).reshape(-1, 3)
    return np.degrees(np.arctan2(v[:, 2], np.linalg.norm(v[:, :2], axis=1)))
