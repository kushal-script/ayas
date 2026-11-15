"""Capture-burst simulation for the twin (DESIGN_STEEL.md E): light-group passes -> sensor frames.

Blender renders each light group once (linear radiance, W/m^2/sr in Blender units, 683 lm/W):
L (left pod), R (right pod), AMB (every ambient light + world), maybe UV (ignored: off in the
white burst). simulate_burst turns them into the five frames the Pi captures, with the
device's FIXED exposure, per-frame ambient change, a moving shadow, hand shake and sensor
noise in electrons. The frames are linear, black-level-free RGB in full-scale units: the
same contract as lightproc.raw_to_rgb on the Pi, so lightproc.process_burst runs unchanged.

simulate_naive is the baseline arm (the old method): one both-pods frame through auto-
exposure, grey-world AWB, sRGB gamma and a mild unsharp mask.

Exposure: k = ref_full_scale * pi / (ref_albedo * E_ref) converts radiance to FS, so a matte
card of albedo ref_albedo under both pods (no ambient) reads ref_full_scale. E_ref is the
analytic LED irradiance at the plate point on the optical axis (twin/lighting.py), scaled
to burst.ref_distance_mm by inverse square: the exposure is set once, at the calibration
distance, and does not follow the scene (a plate held at 250 mm is 0.64x as bright).
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .. import lightproc as LP

GROUPS = ("L", "R", "AMB")
POISSON_MAX = 20.0            # exact Poisson below this many electrons, Gaussian approximation above
DEFAULT_PX_PER_MM = 1.0 / 0.065
SHADOW_SIGMA_MM = 5.0         # moving shadow: illumination = AMB low-passed at this sigma (capture.yaml burst)
SHADOW_MAX_GAIN = 20.0        # cap of E(x - s) / E(x) (a deep shadow moving off a pixel)
NAIVE_DEFAULTS = {"ae_target": 0.18, "unsharp": 0.5, "unsharp_sigma_px": 1.0}


# ---- exposure and geometry -------------------------------------------------------------

def led_irradiance(plan: Mapping[str, Any]) -> float:
    """LED irradiance (W/m^2, both pods) at the plate point on the optical axis (twin/lighting.py)."""
    lighting = importlib.import_module("trinetra.twin.lighting")     # lazy: written by another module owner
    return float(lighting.led_irradiance(plan, groups=("L", "R")))


def exposure_k(plan: Mapping[str, Any], capture_cfg: Mapping[str, Any], e_led: float | None = None) -> dict[str, float]:
    """{'k', 'E_led', 'E_ref'}: radiance -> FS scale of the device's fixed exposure."""
    E = float(e_led) if e_led is not None else led_irradiance(plan)
    if not (E > 0 and math.isfinite(E)):
        raise ValueError(f"LED irradiance must be > 0 W/m^2, got {E}")
    b = capture_cfg.get("burst") or {}
    d_ref = b.get("ref_distance_mm")
    d = ((plan.get("camera") or {}).get("pose") or {}).get("distance_mm")
    E_ref = E * (float(d) / float(d_ref)) ** 2 if d_ref and d else E
    k = float(b.get("ref_full_scale", 0.85)) * math.pi / (float(b.get("ref_albedo", 0.9)) * E_ref)
    return {"k": k, "E_led": E, "E_ref": E_ref}


def px_per_mm(plan: Mapping[str, Any], shape: Sequence[int]) -> float:
    """Image px per plate mm at the target point (f / distance), for the pass resolution."""
    cam = plan.get("camera") or {}
    K = cam.get("K")
    d = (cam.get("pose") or {}).get("distance_mm")
    if not (K and d):
        return DEFAULT_PX_PER_MM
    return float(K[0][0]) * res_scale(plan, shape) / float(d)


def res_scale(plan: Mapping[str, Any], shape: Sequence[int]) -> float:
    """Pass px per full-resolution sensor px: the render's res_scale (pass width / ROI width)."""
    roi = (plan.get("camera") or {}).get("roi")
    return shape[1] / float(roi[2]) if roi else 1.0


def illumination(plan: Mapping[str, Any]) -> str:
    """The plan's device illumination: 'pods' (handheld, device v1) or 'ring' (hood, v2)."""
    return str(((plan.get("extras") or {}).get("device") or {}).get("illumination") or "pods")


def _groups(groups: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    ref = next((groups[g] for g in GROUPS if g in groups and groups[g] is not None), None)
    if ref is None:
        raise ValueError(f"no light-group pass among {GROUPS}")
    shape = np.asarray(ref).shape[:2] + (3,)
    out = {}
    for g in GROUPS:
        a = groups.get(g)
        out[g] = np.zeros(shape, np.float32) if a is None else np.ascontiguousarray(np.asarray(a)[..., :3], np.float32)
    return out


# ---- random processes ------------------------------------------------------------------------

def shake_walk(rng: np.random.Generator, n: int, step_px: float) -> np.ndarray:
    """(n, 2) hand-shake positions: frame 0 at (0, 0), then a Gaussian random walk whose
    per-frame step has RMS length step_px."""
    pos = np.zeros((n, 2))
    if step_px > 0 and n > 1:
        pos[1:] = np.cumsum(rng.normal(0.0, step_px / math.sqrt(2.0), (n - 1, 2)), axis=0)
    return pos


def _blur_small(x: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur of a wide sigma on a grid reduced by about sigma / 3 (fast; low-pass only)."""
    h, w = x.shape[:2]
    f = max(1, int(sigma_px // 3))
    small = cv2.resize(x, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA) if f > 1 else x
    small = cv2.GaussianBlur(small, (0, 0), max(sigma_px / f, 0.5), borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if f > 1 else small


def illumination_parts(img: np.ndarray, sigma_px: float, mask: np.ndarray | None = None
                       ) -> list[tuple[np.ndarray | None, np.ndarray]]:
    """[(region, low-pass)]: the Gaussian low-pass (sigma_px) of an (H, W, C) pass, i.e. its
    illumination. With a mask (the plate), one normalised convolution per side of the outline,
    each defined (extrapolated) over the whole image, so that moving one never pulls the
    bench's illumination estimate into the plate or the other way round."""
    img = np.asarray(img, np.float32)
    if mask is None:
        return [(None, _blur_small(img, sigma_px))]
    out = []
    for part in (np.asarray(mask, bool), ~np.asarray(mask, bool)):
        if part.any():
            m = part.astype(np.float32)
            den = np.maximum(_blur_small(m, sigma_px), 1e-3)
            out.append((part, _blur_small(img * m[..., None], sigma_px) / den[..., None]))
    return out


def shadow_shift(amb: np.ndarray, shift_px: Sequence[float], sigma_px: float, mask: np.ndarray | None = None,
                 parts: list | None = None, max_gain: float = SHADOW_MAX_GAIN) -> np.ndarray:
    """The AMB pass with only its illumination moved by shift_px (a moving hand shadow).

    AMB = rho(x) E(x); its low-pass ~ rho_mean E(x) holds the illumination (shadow edge, lamp
    falloff), the rest is texture and sharp reflections. The moved pass is AMB x E(x - s) / E(x)
    = rho(x) E(x - s): the shadow edge moves, texture and reflections stay where they are
    (DESIGN_STEEL E; the old model translated the whole pass, so alignment locked onto the
    ambient-lit texture of the A frame). The gain is capped at max_gain. parts: a cached
    illumination_parts(amb, sigma_px, mask).
    """
    if abs(float(shift_px[0])) < 1e-6 and abs(float(shift_px[1])) < 1e-6:
        return amb
    parts = illumination_parts(amb, sigma_px, mask) if parts is None else parts
    out = np.array(amb, np.float32, copy=True)
    for region, lf in parts:
        eps = np.float32(1e-3 * max(float(np.median(lf if region is None else lf[region])), 1e-9))
        moved = LP.translate(lf, float(shift_px[0]), float(shift_px[1]), border=cv2.BORDER_REFLECT)
        gain = np.clip((moved + eps) / (lf + eps), 0.0, max_gain)
        if region is None:
            out *= gain
        else:
            out[region] *= gain[region]
    return out


def ambient_factors(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """Per-frame ambient multipliers a_f = 1 + N(0, sigma), >= 0."""
    return np.maximum(1.0 + rng.normal(0.0, sigma, n), 0.0) if sigma > 0 else np.ones(n)


def expose(S: np.ndarray, rng: np.random.Generator, sensor: Mapping[str, Any]) -> np.ndarray:
    """Signal in FS -> noisy, quantised float32 frame in FS (black-level-free, clipped at 1).

    Electrons: Poisson(S * e_fs) + N(0, read_noise); the Poisson part is exact below
    POISSON_MAX electrons and Gaussian above (indistinguishable there, ~10x faster).
    Quantised to the DN steps between black and white level; values may go below 0 down to
    -black_level (as raw minus black does on the Pi), never above 1.
    A negative S (the negative lobes of the cubic sub-pixel shift next to sparse bright pixels:
    render speckle on a black background, e.g. light leaking under the hood) is kept as a
    noise-free offset rather than clipped to 0: clipping made such a frame brighter on average
    (review, 15 Nov: the ambient frame of a leaking hood +8 %, so M = L - A came out 7 % dark,
    up to -7 DN). The Poisson draw uses max(S, 0).
    """
    e_fs = LP.electrons_per_fs(sensor)
    rn = float(sensor.get("read_noise_e", 0.0))
    levels = LP.raw_levels(sensor)
    S = np.asarray(S, np.float32)
    lam = np.maximum(S, 0.0) * np.float32(e_fs)
    e = rng.standard_normal(lam.shape, dtype=np.float32)
    e *= np.sqrt(lam + np.float32(rn * rn))
    e += lam
    low = lam < POISSON_MAX
    if low.any():
        n_low = int(low.sum())
        e[low] = rng.poisson(lam[low]).astype(np.float32) + np.float32(rn) * rng.standard_normal(n_low, dtype=np.float32)
    neg = S < 0
    if neg.any():
        e[neg] += S[neg] * np.float32(e_fs)                 # mean-preserving: the lobe's deficit stays
    e *= np.float32(levels / e_fs)
    np.rint(e, out=e)
    lo = -float(sensor.get("black_level", 0.0))
    np.clip(e, lo, float(levels), out=e)
    e *= np.float32(1.0 / levels)
    return e


def _roi(roi_mask: np.ndarray | None, shape: Sequence[int]) -> np.ndarray:
    return np.ones(shape[:2], bool) if roi_mask is None else np.asarray(roi_mask) > 0


def _median_luma(img: np.ndarray, roi: np.ndarray) -> float:
    v = LP.luma(img)[::2, ::2][roi[::2, ::2]]
    return float(np.median(v)) if v.size else 0.0


# ---- the two arms ---------------------------------------------------------------------------

def simulate_burst(groups: Mapping[str, np.ndarray], plan: Mapping[str, Any], rng: np.random.Generator,
                   capture_cfg: Mapping[str, Any], roi_mask: np.ndarray | None = None, *,
                   e_led: float | None = None, shifts: Mapping[str, Sequence[float]] | None = None) -> LP.Burst:
    """Light-group passes -> lightproc.Burst of the B1, L, A, R, B2 frames (DESIGN_STEEL E).

    S_f = k (l gL L + r gR R + a_f AMB_f), (l, r) from capture.yaml burst.led_state;
    gL, gR = 1 +- led_jitter per burst (pod imbalance) times a per-frame ripple
    N(0, leds.frame_sigma); a_f = 1 + N(0, ambient.change.frame_sigma); AMB_f is AMB with its
    illumination (low-pass, burst.shadow_sigma_mm) moved by f x shadow_shift_mm along one
    random direction (a moving hand shadow, frame index f; shadow_shift: texture and
    reflections stay put); then the whole frame moves by the hand-shake random walk (B1 stays
    at 0, so labels made from the plan stay exact after aligning to B1), then sensor noise.
    capture.shake_px_per_frame is in full-resolution sensor px and is scaled to the pass by
    res_scale (a 0.25x render shakes 0.25x as many of its px); shifts ({frame: (dx, dy)},
    pass px) overrides the walk as given. e_led overrides the analytic LED irradiance.
    meta: shifts, ambient_factors, shadow_px, led_gains, k, E_led, E_ref, r_true (noise-free
    ambient ratio in the ROI), px_per_mm, px_scale (pass px per sensor px: lightproc reports
    motion in sensor px), shake_px_per_frame (pass px), illumination (plan
    extras.device.illumination: lightproc's combine), sequence, material.
    """
    bcfg = capture_cfg.get("burst") or {}
    seq = tuple(bcfg.get("sequence", LP.SEQUENCE))
    states = bcfg.get("led_state") or {"B1": [1, 1], "L": [1, 0], "A": [0, 0], "R": [0, 1], "B2": [1, 1]}
    sensor = LP.sensor_for(capture_cfg.get("sensor"), illumination(plan))     # per-illumination analogue gain
    g = _groups(groups)
    shape = g["L"].shape
    ex = exposure_k(plan, capture_cfg, e_led)
    k = ex["k"]
    cap = plan.get("capture") or {}
    change = (plan.get("ambient") or {}).get("change") or {}
    scale = res_scale(plan, shape)
    step = float(cap.get("shake_px_per_frame", 0.0) or 0.0) * scale
    jit = float(cap.get("led_jitter", 0.0) or 0.0)
    ripple = float((capture_cfg.get("leds") or {}).get("frame_sigma", 0.0) or 0.0)

    gains = {"L": 1.0 + float(rng.uniform(-jit, jit)), "R": 1.0 + float(rng.uniform(-jit, jit))}
    walk = shake_walk(rng, len(seq), step)
    if shifts is not None:
        walk = np.array([shifts.get(f, (0.0, 0.0)) for f in seq], float)
    a_f = ambient_factors(rng, len(seq), float(change.get("frame_sigma", 0.0) or 0.0))
    ppm = px_per_mm(plan, shape)
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    sh = float(change.get("shadow_shift_mm", 0.0) or 0.0) * ppm
    shadow = {f: [i * sh * math.cos(theta), i * sh * math.sin(theta)] for i, f in enumerate(seq)}
    sigma = float(bcfg.get("shadow_sigma_mm", SHADOW_SIGMA_MM)) * ppm
    parts = illumination_parts(g["AMB"], sigma, None if roi_mask is None else np.asarray(roi_mask) > 0) \
        if sh > 0 else None

    frames: dict[str, np.ndarray] = {}
    for i, f in enumerate(seq):
        l_on, r_on = states[f]
        rip = 1.0 + rng.normal(0.0, ripple, 2) if ripple > 0 else np.ones(2)
        amb = g["AMB"] if sh == 0 or i == 0 else shadow_shift(g["AMB"], shadow[f], sigma, parts=parts)
        S = amb * np.float32(k * a_f[i])
        if l_on:
            S += g["L"] * np.float32(k * l_on * gains["L"] * rip[0])
        if r_on:
            S += g["R"] * np.float32(k * r_on * gains["R"] * rip[1])
        S = LP.translate(S, float(walk[i, 0]), float(walk[i, 1]), interp=cv2.INTER_CUBIC, border=cv2.BORDER_REFLECT)
        frames[f] = expose(S, rng, sensor)

    roi = _roi(roi_mask, shape)
    led = _median_luma(g["L"] * np.float32(gains["L"]) + g["R"] * np.float32(gains["R"]), roi)
    r_true = _median_luma(g["AMB"], roi) / led if led > 0 else float("inf")
    meta = {
        "shifts": {f: [float(walk[i, 0]), float(walk[i, 1])] for i, f in enumerate(seq)},
        "ambient_factors": {f: float(a_f[i]) for i, f in enumerate(seq)},
        "shadow_px": shadow, "led_gains": gains, "k": k, "E_led": ex["E_led"], "E_ref": ex["E_ref"],
        "r_true": float(min(r_true, LP.R_CAP)), "px_per_mm": ppm, "px_scale": scale,
        "shake_px_per_frame": step, "illumination": illumination(plan), "sequence": list(seq),
        "material": ((plan.get("plate") or {}).get("material") or {}).get("kind"),
    }
    return LP.Burst(frames, meta)


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055).astype(np.float32)


def simulate_naive(groups: Mapping[str, np.ndarray], plan: Mapping[str, Any], rng: np.random.Generator,
                   capture_cfg: Mapping[str, Any], roi_mask: np.ndarray | None = None, *,
                   e_led: float | None = None) -> np.ndarray:
    """Baseline arm: one both-pods frame L + R + AMB -> uint8 sRGB, as a phone-style ISP does it.

    Auto-exposure puts the plate (roi_mask, else whole-image) median luma at
    naive.ae_target FS, then the same sensor noise, grey-world AWB on the plate, clip, sRGB
    OETF and a mild unsharp mask (naive.unsharp, naive.unsharp_sigma_px).
    """
    nv = {**NAIVE_DEFAULTS, **(capture_cfg.get("naive") or {})}
    sensor = LP.sensor_for(capture_cfg.get("sensor"))           # the ISP's auto-exposure: the burst's ring gain is not used
    g = _groups(groups)
    k = exposure_k(plan, capture_cfg, e_led)["k"]
    jit = float((plan.get("capture") or {}).get("led_jitter", 0.0) or 0.0)
    gl, gr = 1.0 + float(rng.uniform(-jit, jit)), 1.0 + float(rng.uniform(-jit, jit))
    S = (g["L"] * np.float32(gl) + g["R"] * np.float32(gr) + g["AMB"]) * np.float32(k)
    roi = _roi(roi_mask, S.shape)
    med = _median_luma(S, roi)
    if med > 1e-9:
        S *= np.float32(float(nv["ae_target"]) / med)                       # auto-exposure
    img = expose(S, rng, sensor)
    means = img[::2, ::2][roi[::2, ::2]].mean(axis=0) if roi.any() else img.reshape(-1, 3).mean(axis=0)
    if np.all(means > 1e-6):
        img *= (means[1] / means).astype(np.float32)                         # grey-world AWB
    img = srgb_oetf(np.clip(img, 0.0, 1.0))
    amt = float(nv["unsharp"])
    if amt > 0:
        blur = cv2.GaussianBlur(img, (0, 0), float(nv["unsharp_sigma_px"]))
        img = img + np.float32(amt) * (img - blur)
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


# ---- a toy scene (tests, demos, smoke runs without Blender) ------------------------------------

def _smooth_noise(rng: np.random.Generator, shape: Sequence[int], sigma: float) -> np.ndarray:
    n = cv2.GaussianBlur(rng.standard_normal(tuple(shape[:2])).astype(np.float32), (0, 0), sigma)
    return n / max(float(n.std()), 1e-9)


def toy_scene(h: int = 400, w: int = 400, *, seed: int = 0, r: float = 0.0, albedo: float = 0.35,
              highlight: float = 0.5, e_led: float = 10.0, margin_px: int = 20,
              tint: Sequence[float] = (1.0, 0.85, 0.65), shadow: bool = True, distance_mm: float = 200.0,
              shake_px_per_frame: float = 0.0, led_jitter: float = 0.0, frame_sigma: float = 0.0,
              shadow_shift_mm: float = 0.0) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, float]:
    """Synthetic light groups of a steel plate -> (groups, plan, plate roi_mask, e_led).

    Matte textured plate (grain + mottling, a scratch, a pit, a stain) on a dark bench.
    The pods' irradiance sums to e_led everywhere (left pod brighter on the left); each pod
    adds one mirror highlight reaching `highlight` FS above the plate (left one in L only,
    right one in R only). AMB is a tinted, graded light with a shadow edge whose median
    over the plate is r x the LED irradiance, so r_ambient is about r.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rho = albedo * (1.0 + 0.08 * _smooth_noise(rng, (h, w), 1.2) + 0.10 * _smooth_noise(rng, (h, w), 8.0))
    rho = np.clip(rho, 0.01, 1.0)[..., None] * np.array([1.02, 1.0, 0.97], np.float32)
    # defects: scratch (dark line), pit (dark disc), stain (yellow-brown patch)
    dmask = np.zeros((h, w), np.uint8)
    cv2.line(dmask, (int(0.2 * w), int(0.72 * h)), (int(0.6 * w), int(0.62 * h)), 1, 2)
    rho[dmask > 0] *= 0.35
    pit = (xx - 0.65 * w) ** 2 + (yy - 0.3 * h) ** 2 < (0.012 * w) ** 2
    rho[pit] *= 0.2
    stain = ((xx - 0.38 * w) / (0.06 * w)) ** 2 + ((yy - 0.3 * h) / (0.035 * h)) ** 2 < 1
    rho[stain] *= np.array([0.95, 0.8, 0.55], np.float32)
    plate = np.zeros((h, w), bool)
    plate[margin_px:h - margin_px, margin_px:w - margin_px] = True
    rho[~plate] = 0.05

    e_l = (0.5 * e_led * (1.15 - 0.3 * xx / w))[..., None]
    e_r = (0.5 * e_led * (0.85 + 0.3 * xx / w))[..., None]
    peak = highlight * 0.9 * e_led / (0.85 * math.pi)        # radiance that reads `highlight` FS
    sig = w / 25.0
    hl_l = peak * np.exp(-((xx - 0.3 * w) ** 2 + (yy - 0.45 * h) ** 2) / (2 * sig ** 2))[..., None]
    hl_r = peak * np.exp(-((xx - 0.7 * w) ** 2 + (yy - 0.55 * h) ** 2) / (2 * sig ** 2))[..., None]
    L = rho * e_l / math.pi + hl_l * plate[..., None]
    R = rho * e_r / math.pi + hl_r * plate[..., None]
    e_a = 0.6 + 0.8 * xx / w
    if shadow:
        e_a = e_a * (0.25 + 0.375 * (1.0 - np.tanh((yy - 0.62 * h) / 4.0)))
    med = float(np.median(e_a[plate])) if r > 0 else 1.0
    amb = rho * (r * e_led * e_a / med)[..., None] / math.pi * np.asarray(tint, np.float32)
    groups = {"L": L.astype(np.float32), "R": R.astype(np.float32), "AMB": amb.astype(np.float32)}
    f = distance_mm / 0.065
    plan = {
        "schema": "trinetra.twin.scene/2", "scene_id": f"toy{seed}", "seed": seed,
        "plate": {"size_mm": [(w - 2 * margin_px) * 0.065, (h - 2 * margin_px) * 0.065], "material": {"kind": "ms_cold_rolled"}},
        "camera": {"K": [[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]], "roi": [0, 0, w, h],
                   "pose": {"distance_mm": distance_mm}},
        "ambient": {"condition": "toy", "lux": r * e_led * 683.0,
                    "change": {"frame_sigma": frame_sigma, "shadow_shift_mm": shadow_shift_mm}},
        "capture": {"sensor": "full", "shake_px_per_frame": shake_px_per_frame, "led_jitter": led_jitter},
        "extras": {"device": {"mode": "open"}},
    }
    return groups, plan, plate, e_led
