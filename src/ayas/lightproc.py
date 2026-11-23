"""Lighting-invariant capture processing, shared by the Pi and the twin (DESIGN_STEEL.md E).

A burst is five linear, black-level-free RGB frames in full-scale units (FS, 1.0 = clip):
B1 (both pods), L (left only), A (all off = ambient), R (right only), B2 (both again).
process_burst aligns them to B1, subtracts the ambient frame, combines the left-only and
right-only images, normalises deterministically and runs the quality gate, which refuses
(RETAKE) any capture it cannot handle instead of failing silently. The combination follows
the device's illumination (burst.meta['illumination'], capture.yaml process.combine):
- pods (handheld, device v1): the luma-guided minimum of L and R, which removes the pods'
  mirror highlights;
- ring (hood head, device v2): the mean of L and R. The dark-field ring sends no mirror
  highlight into the lens, and a scratch lit from one side scatters more than from the
  other, so the minimum would throw half of the signal away; the mean is (B - A) / 2 with
  less noise. "L" and "R" are then the ring's image-left and image-right halves.

Raw-only maths: never feed ISP, gamma or JPEG output into the subtraction. On the Pi,
use raw_to_rgb on the unpacked SRGGB10 stream. numpy + cv2 only (runs on the Pi).

Conventions:
- shifts are (dx, dy) image pixels; estimate_shift(a, b) = d means b(x + d) = a(x), i.e. the
  content of b moved by +d; translate(img, dx, dy) moves content by +(dx, dy);
- luma uses linear Rec. 709 weights; FS units throughout; metrics are JSON-safe floats.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEQUENCE = ("B1", "L", "A", "R", "B2")
VERDICTS = ("PASS", "PASS_FLAGGED", "RETAKE")
LUMA_W = np.array([0.2126, 0.7152, 0.0722], np.float32)      # linear Rec. 709 / sRGB primaries
R_CAP = 1.0e3                                                # finite stand-in for r = inf (JSON)
TRIM = 0.01                                                  # residual_sigma drops the top 1 % of z^2
TRIM_C = 0.92479                                             # E[z^2 | |z| < 2.5758] for z ~ N(0, 1)

# process defaults (each can be overridden in capture.yaml process:)
DEFAULTS = {
    "sat_level": 0.95, "min_method": "luma", "percentile": 99.5, "gamma": 2.2,
    "align": "phase_correlation", "min_guide_sigma_px": 1.5, "roi_erode_px": 6,
    "align_crop_px": 768, "align_lowpass": 0.05, "align_whiten": 0.6, "align_min_response": 0.15,
    "align_max_px": 24, "align_a_slack_px": 3.0, "align_transform": "sqrt",
    "stat_stride": 3, "resid_bin": 4, "blur_bin": 2, "percentile_bin": 4, "inpaint_px": 3,
    "ghost_z": 6.0, "ghost_contrast": 0.1, "ghost_hl_factor": 3.0, "ghost_closure": "min", "ghost_coarse_px": 0, "ghost_coarse_closure": "min", "ghost_coarse_min_r": 1.0,
    "local_r_px": 32, "local_r_max": 8.0,
    "combine": {"pods": "min_luma", "ring": "mean"}, "align_braced": {"ring": True},
    "align_refine": True, "align_refine_iters": 8, "align_refine_max_px": 1.5, "align_refine_sigma_px": 1.0,
    "align_refine_z": 3.0, "ghost_amb_factor": 1.0,
    "flicker_bin": 8, "flicker_z": 5.0, "flicker_q": 99.0,
}
COMBINES = ("min_luma", "mean")
ILLUMINATIONS = ("pods", "ring")
SAT_TOO_BRIGHT = 0.5          # non-finite metrics with this share of the ROI saturated: too bright, not invalid


# ---- data structures ------------------------------------------------------------------

@dataclass
class Burst:
    """frames: name -> float32 (H, W, 3) linear FS (B1, L, A, R, B2); meta: free-form, JSON-safe."""
    frames: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    verdict: str                       # PASS | PASS_FLAGGED | RETAKE
    reasons: list[str]                 # keys of gate.messages (plus 'ambient_high', 'invalid')
    message: str                       # operator text: the first matching gate.messages entry
    details: list[str] = field(default_factory=list)   # human-readable "metric value vs limit"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Processed:
    image_u8: np.ndarray               # (H, W, 3) uint8 RGB, the detector input (before rectify)
    M_lin: np.ndarray                  # (H, W, 3) float32 LED-only, glare-free, fixed-WB, FS units
    relief: np.ndarray                 # (H, W) float32 (lum Lc - lum Rc) / (lum Lc + lum Rc + 1e-4)
    sat_mask: np.ndarray               # (H, W) bool, pixels M could not recover (see process_burst)
    shifts: dict[str, list[float]]     # estimated [dx, dy] of each frame relative to B1
    metrics: dict[str, Any]
    gate: GateResult


# ---- config -----------------------------------------------------------------------------

def load_capture_config(path: str | Path = "configs/capture.yaml") -> dict[str, Any]:
    """Read capture.yaml; a relative path is tried from the cwd, then from the repo root."""
    p = Path(path)
    if not p.exists() and not p.is_absolute() and (ROOT / p).exists():
        p = ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def _proc(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("process") or {})
    return out


def combine_for(pc: Mapping[str, Any], illumination: str | None) -> str:
    """The L/R combination for an illumination: process.combine is {illumination: method} or
    one method for all; without it, process.min_method (luma = min_luma). Methods: COMBINES."""
    illum = str(illumination or "pods")
    c = pc.get("combine")
    if isinstance(c, Mapping):
        c = c.get(illum, c.get("pods"))
    if c is None:
        c = "min_luma" if pc.get("min_method", "luma") == "luma" else pc.get("min_method")
    if c not in COMBINES:
        raise ValueError(f"process.combine {c!r} for illumination {illum!r} not supported {COMBINES}")
    return str(c)


def sensor_for(sensor: Mapping[str, Any] | None, illumination: str | None = None) -> dict[str, Any]:
    """capture.yaml sensor settings for one illumination: the base values with
    sensor.by_illumination[illumination] over them (e.g. ring: {analogue_gain: 1.5}: the
    camera has one exposure and the hood's ring gives about 1/1.5 of the pods' irradiance at
    the plate); a value may also be {pods: x, ring: y} (pods is the fallback)."""
    illum = str(illumination or "pods")
    s = dict(sensor or {})
    over = (s.pop("by_illumination", None) or {}).get(illum) or {}
    s.update(over)
    return {k: (v.get(illum, v.get("pods")) if isinstance(v, Mapping) else v) for k, v in s.items()}


def electrons_per_fs(sensor: Mapping[str, Any], illumination: str | None = None) -> float:
    """Electrons at full scale: the ADC clips at full_well / analogue_gain (gain >= 1)."""
    s = sensor_for(sensor, illumination)
    return float(s.get("full_well_e", 9610.0)) / max(float(s.get("analogue_gain", 1.0) or 1.0), 1.0)


def raw_levels(sensor: Mapping[str, Any]) -> int:
    """Number of DN steps between the black level and the white level."""
    s = sensor_for(sensor)
    return (1 << int(s.get("bit_depth", 10))) - 1 - int(s.get("black_level", 0))


# ---- small image helpers ----------------------------------------------------------------

def luma(x: np.ndarray) -> np.ndarray:
    """Linear Rec. 709 luma, float32 (H, W); a 2-D input is returned as float32."""
    x = np.asarray(x)
    if x.ndim == 2:
        return x.astype(np.float32, copy=False)
    x = np.ascontiguousarray(x[..., :3], dtype=np.float32)
    return cv2.transform(x, LUMA_W.reshape(1, 3)).reshape(x.shape[:2])


def _maxc(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(x[..., 0], x[..., 1]), x[..., 2])


def translate(img: np.ndarray, dx: float, dy: float, interp: int = cv2.INTER_LINEAR,
              border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Move the image content by (+dx, +dy) px, sub-pixel: out(x, y) = img(x - dx, y - dy) (cv2.warpAffine)."""
    img = np.asarray(img)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return img
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=border)


def align(img: np.ndarray, shift: Sequence[float]) -> np.ndarray:
    """Undo an estimated shift (estimate_shift convention): the result lines up with the reference."""
    return translate(img, -float(shift[0]), -float(shift[1]))


@lru_cache(maxsize=8)
def _window(h: int, w: int) -> np.ndarray:
    return cv2.createHanningWindow((w, h), cv2.CV_32F)


@lru_cache(maxsize=8)
def _freqs(h: int, w: int, lowpass: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(h).astype(np.float32)[:, None]
    fx = np.fft.fftfreq(w).astype(np.float32)[None, :]
    wt = np.exp(-(fx ** 2 + fy ** 2) / (2.0 * lowpass ** 2)).astype(np.float32)
    return fx, fy, wt


def _spectrum(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, np.float32)
    y = (y - float(y.mean())) * _window(*y.shape)
    return cv2.dft(y, flags=cv2.DFT_COMPLEX_OUTPUT)


def _gauss_peak(cm: float, c0: float, cp: float) -> float:
    """Sub-pixel offset of a peak from three samples (Gaussian fit, parabola as fallback)."""
    if min(cm, c0, cp) > 0:
        lm, l0, lp = math.log(cm), math.log(c0), math.log(cp)
        d = lm - 2.0 * l0 + lp
        if d < 0:
            return float(np.clip(0.5 * (lm - lp) / d, -0.5, 0.5))
    d = cm - 2.0 * c0 + cp
    return float(np.clip(0.5 * (cm - cp) / d, -0.5, 0.5)) if d < 0 else 0.0


def _peak(r: np.ndarray, center: Sequence[float], radius: float | None) -> tuple[int, int, int, int]:
    """(row, col, dy, dx) of the correlation maximum: wrapped indices and unwrapped offsets.

    With a radius, only offsets within +-radius px (a square) of center = (dx, dy) count: a
    frame cannot jump further than the hand moves, and a far peak is another bright feature
    (e.g. the ambient frame's lamp reflection matched to an LED highlight in B1).
    """
    h, w = r.shape
    if radius is None or 2 * math.ceil(radius) + 1 >= min(h, w):
        i, j = np.unravel_index(int(np.argmax(r)), r.shape)
        return int(i), int(j), int(i if i <= h // 2 else i - h), int(j if j <= w // 2 else j - w)
    rad = int(math.ceil(radius))
    oy = np.arange(-rad, rad + 1) + int(round(float(center[1])))
    ox = np.arange(-rad, rad + 1) + int(round(float(center[0])))
    sub = r[np.ix_(oy % h, ox % w)]
    a, b = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return int(oy[a] % h), int(ox[b] % w), int(oy[a]), int(ox[b])


def _correlate(Fa: np.ndarray, Fb: np.ndarray, lowpass: float, whiten: float,
               center: Sequence[float] = (0.0, 0.0), radius: float | None = None
               ) -> tuple[float, float, float, float]:
    """Generalised phase correlation of two spectra -> (dx, dy, response, noise floor).

    response is the peak relative to identical images (1.0); floor is the expected highest
    peak of two unrelated images (random phases) on the same scale. radius (px, optional)
    limits the peak search to around center (see _peak).
    """
    h, w = Fa.shape[:2]
    C = cv2.mulSpectrums(Fb, Fa, 0, conjB=True)                   # Fb * conj(Fa): peak at +d
    mag = cv2.magnitude(C[..., 0], C[..., 1])
    _, _, wt = _freqs(h, w, float(lowpass))
    g = wt / np.power(mag + 1e-30, np.float32(whiten))
    C *= g[..., None]
    a = mag * g                                                    # |weighted cross-power|
    s1 = float(a.sum())
    ideal = s1 / (h * w)                                           # peak of identical images
    floor = math.sqrt(float((a * a).sum())) / s1 * math.sqrt(2.0 * math.log(h * w)) if s1 > 0 else 1.0
    r = cv2.idft(C, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    i, j, y0, x0 = _peak(r, center, radius)
    dx = _gauss_peak(r[i, (j - 1) % w], r[i, j], r[i, (j + 1) % w])
    dy = _gauss_peak(r[(i - 1) % h, j], r[i, j], r[(i + 1) % h, j])
    resp = float(r[i, j]) / ideal if ideal > 0 else 0.0
    return float(x0 + dx), float(y0 + dy), float(np.clip(resp, 0.0, 1.0)), float(floor)


def estimate_shift(a: np.ndarray, b: np.ndarray, lowpass: float = DEFAULTS["align_lowpass"],
                   whiten: float = DEFAULTS["align_whiten"], max_px: float | None = None,
                   center: Sequence[float] = (0.0, 0.0)) -> tuple[float, float, float]:
    """Translation of b relative to a by phase correlation on luma -> (dx, dy, response).

    b(x + d) = a(x). Generalised phase correlation (cross-power spectrum / |.|^whiten, Gaussian
    low-pass `lowpass` cycles/px, Hanning window) with a 3-point Gaussian peak fit: ~0.01-0.05
    px on textured images, robust to the L/R/ambient illumination change. response is the
    peak relative to identical images (1.0); unrelated images stay near the noise floor
    (about 0.05-0.15 for 256-768 px windows). max_px (optional) searches the peak only within
    +-max_px of center = (dx, dy).
    """
    ya, yb = luma(a), luma(b)
    if ya.shape != yb.shape:
        raise ValueError(f"shape mismatch {ya.shape} vs {yb.shape}")
    return _correlate(_spectrum(ya), _spectrum(yb), lowpass, whiten, center, max_px)[:3]


# ---- ROI and robust statistics -------------------------------------------------------------

def _roi(roi_mask: np.ndarray | None, shape: Sequence[int], erode_px: int) -> np.ndarray:
    """Boolean ROI eroded by erode_px (image border excluded the same way)."""
    h, w = shape[:2]
    m = np.ones((h, w), np.uint8) if roi_mask is None else (np.asarray(roi_mask) > 0).astype(np.uint8)
    if m.shape != (h, w):
        raise ValueError(f"roi_mask shape {m.shape} != frame shape {(h, w)}")
    if erode_px > 0:
        m[:erode_px], m[-erode_px:], m[:, :erode_px], m[:, -erode_px:] = 0, 0, 0, 0
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode_px + 1, 2 * erode_px + 1)))
    return m.astype(bool)


def _vals(x: np.ndarray, roi: np.ndarray, stride: int) -> np.ndarray:
    s = max(int(stride), 1)
    return x[::s, ::s][roi[::s, ::s]]


def _median(x: np.ndarray, roi: np.ndarray, stride: int) -> float:
    v = _vals(x, roi, stride)
    return float(np.median(v)) if v.size else float("nan")


def _crop_box(roi: np.ndarray, size: int) -> tuple[slice, slice]:
    """A centred, at most size x size, even-sized window on the ROI bounding box."""
    ys, xs = np.nonzero(roi[::4, ::4])
    h, w = roi.shape
    if ys.size:
        y0, y1, x0, x1 = ys.min() * 4, min(h, ys.max() * 4 + 4), xs.min() * 4, min(w, xs.max() * 4 + 4)
    else:
        y0, y1, x0, x1 = 0, h, 0, w
    ch, cw = min(size, y1 - y0), min(size, x1 - x0)
    ch, cw = ch - ch % 2, cw - cw % 2
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    ya, xa = max(0, min(h - ch, cy - ch // 2)), max(0, min(w - cw, cx - cw // 2))
    return slice(ya, ya + ch), slice(xa, xa + cw)


def _bin(x: np.ndarray, b: int) -> np.ndarray:
    h, w = x.shape[:2]
    hb, wb = h // b, w // b
    return cv2.resize(x[:hb * b, :wb * b], (wb, hb), interpolation=cv2.INTER_AREA)


# ---- alignment ------------------------------------------------------------------------------

def _align_view(y: np.ndarray, transform: str) -> np.ndarray:
    """What the shift estimate correlates: sqrt(luma) by default.

    sqrt is the variance-stabilising transform of shot noise: it keeps the clipped LED
    highlights and lamp reflections (whose shapes differ between frames lit differently)
    from dominating the correlation over the plate texture. On toy plates with clipped
    highlights it halves the L/R error (p90 0.41 -> 0.22 px) and the A error.
    """
    if transform == "sqrt":
        return np.sqrt(np.maximum(y, 0.0))
    if transform == "linear":
        return y
    raise ValueError(f"process.align_transform {transform!r} not supported (sqrt, linear)")


def burst_shifts(y: Mapping[str, np.ndarray], roi: np.ndarray, pc: Mapping[str, Any], braced: bool = False
                 ) -> tuple[dict[str, list[float]], dict[str, float], list[str]]:
    """Shifts of L, A, R, B2 relative to B1 from their lumas -> (shifts, responses, fallbacks).

    Each frame is correlated with B1 (on _align_view of the luma). An estimate counts only if
    its response is at least max(align_min_response, 1.5 x the correlation's noise floor).
    L, R and B2 are searched within +-align_max_px of 0. A (LEDs off) is searched within
    align_a_slack_px + |R - L| of the mean of the L and R shifts (it is exposed between
    them): its only structure may be a lamp reflection, which otherwise locks onto an LED
    highlight in B1 tens of px away. A often has no common structure (dark room): it then
    takes the mean of the L and R shifts. An unreliable L, R or B2 (featureless plate) is
    interpolated in time between B1 and B2, or set to 0 when B2 is unreliable too; fallbacks
    lists those frames.
    braced (process.align_braced for the illumination; the hood resting on the part): L, A
    and R follow B2 linearly in time (B1 -> B2 is the only unbiased estimate: the same light).
    A braced device drifts by a fraction of a px per frame, while L and R, lit from opposite
    sides, correlate with B1 0.3-0.7 px off (relief and directional scatter; twin pilot,
    14 Nov): enough for 'changed' ghosts on a still capture.
    """
    if pc.get("align", "phase_correlation") != "phase_correlation":
        raise ValueError(f"process.align {pc.get('align')!r} not supported (phase_correlation)")
    sy, sx = _crop_box(roi, int(pc["align_crop_px"]))
    tf = str(pc.get("align_transform", "sqrt"))
    F0 = _spectrum(_align_view(y["B1"][sy, sx], tf))
    shifts: dict[str, list[float]] = {"B1": [0.0, 0.0]}
    resp: dict[str, float] = {"B1": 1.0}
    ok = {"B1": True}
    max_px = float(pc["align_max_px"]) if pc.get("align_max_px") else None

    def corr(k: str, center: Sequence[float], radius: float | None) -> None:
        dx, dy, r, fl = _correlate(F0, _spectrum(_align_view(y[k][sy, sx], tf)), pc["align_lowpass"],
                                   pc["align_whiten"], center, radius)
        shifts[k], resp[k] = [dx, dy], r
        ok[k] = r >= max(float(pc["align_min_response"]), 1.5 * fl)

    if braced:
        corr("B2", (0.0, 0.0), max_px)
        end = shifts["B2"] if ok["B2"] else [0.0, 0.0]
        for k in ("L", "A", "R"):
            t = SEQUENCE.index(k) / (len(SEQUENCE) - 1)
            shifts[k], resp[k] = [t * end[0], t * end[1]], resp["B2"]          # they follow the B2 estimate
        if not ok["B2"]:
            shifts["B2"] = [0.0, 0.0]
        return shifts, resp, ([] if ok["B2"] else ["B2"])
    for k in ("L", "R", "B2"):
        corr(k, (0.0, 0.0), max_px)
    fallbacks = []
    end = shifts["B2"] if ok["B2"] else [0.0, 0.0]
    if not ok["B2"]:
        shifts["B2"] = [0.0, 0.0]
        fallbacks.append("B2")
    for k in ("L", "R"):
        if not ok[k]:
            t = SEQUENCE.index(k) / (len(SEQUENCE) - 1)
            shifts[k] = [t * end[0], t * end[1]]
            fallbacks.append(k)
    mid = [(shifts["L"][i] + shifts["R"][i]) / 2.0 for i in range(2)]
    rad_a = float(pc.get("align_a_slack_px", 3.0)) + math.hypot(shifts["R"][0] - shifts["L"][0],
                                                                 shifts["R"][1] - shifts["L"][1])
    rad_a = rad_a if max_px is None else min(rad_a, max_px)
    corr("A", mid, rad_a)
    if not ok["A"]:
        shifts["A"] = mid
        fallbacks.append("A")
    return shifts, resp, fallbacks


def refine_shifts(y: Mapping[str, np.ndarray], shifts: Mapping[str, Sequence[float]], roi: np.ndarray,
                  usable: np.ndarray, pc: Mapping[str, Any], g_both: float = 1.0
                  ) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Sub-pixel refinement of the L, A and R shifts that minimises the burst closure -> (shifts, info).

    For a static scene the frames add up exactly in linear raw: B1 + A = L + R and B2 + A = L + R
    (L and R each hold one half's LEDs plus the ambient), whatever the lighting looks like. The
    coarse shifts cannot use that: L and R correlated with B1 are biased by their opposite-side
    lighting (0.2-0.7 px, free alignment), and braced shifts assume the device moves linearly
    from B1 to B2, while a hand's shake is a random walk (twin full-resolution pilot, 15 Nov: L
    0.16 px off with the hood, at 0.18 px/frame shake). Both leave closure ghosts at sharp
    features of a perfectly still part. Here B1 and B2 keep their (unbiased: same light) shifts
    and the shifts of A, L and R take damped Gauss-Newton steps on
        sum over the usable pixels of ((B1 + B2) / 2 + A - L - R)^2
    (lumas Gaussian-smoothed by align_refine_sigma_px, on the align_crop_px window). A frame
    without structure (a dark ambient frame) stays where it was (the damping). Translations
    cannot explain a real change (a moving reflection or shadow, flicker, a roll), so the
    closure and ghost checks still see those. The coarse shifts are kept when the refinement is
    not trusted: a correction beyond align_refine_max_px or a closure rms that does not drop;
    and a frame keeps its coarse shift when its correction is below align_refine_z standard
    errors (a frame with no structure, e.g. a dark ambient frame: the step would only fit noise).
    usable must exclude clipped pixels (they break B = L + R + A) with a margin for the motion
    and the smoothing. info: per-frame correction (px), iterations, closure rms before/after
    (fraction of full scale), or why it was skipped. LED droop: both channels on may give g x
    the sum of each alone, B - A = g (L + R - 2A). Left in the closure, a droop was fitted by
    moving L and R apart (review, 15 Nov: 5 % droop, L and R 0.58 px off on a still toy plate,
    M 2.5 % wrong). So the closure minimised is c (B1 + B2) / 2 + (2 - c) A - L - R with the
    scale c a free nuisance parameter, started at 1 / g_both (leds.both_on_gain); info
    'led_scale' = 1 / c. Only the shifts are used: the closure statistics keep g_both, so an
    uncalibrated droop is still refused, never absorbed.
    """
    out = {k: [float(v[0]), float(v[1])] for k, v in shifts.items()}
    iters = int(pc.get("align_refine_iters", 8) or 0)
    if not pc.get("align_refine", True) or iters <= 0:
        return out, {"skipped": "off"}
    sy, sx = _crop_box(roi, int(pc["align_crop_px"]))
    m = (roi & usable)[sy, sx]
    if int(m.sum()) < 2000:
        return out, {"skipped": "too few usable pixels"}
    lim = float(pc.get("align_refine_max_px", 1.5))
    big = max(math.hypot(*out[k]) for k in ("L", "A", "R", "B2")) + lim + 3.0
    h, w = roi.shape
    P = int(math.ceil(big))
    ya, yb, xa, xb = max(0, sy.start - P), min(h, sy.stop + P), max(0, sx.start - P), min(w, sx.stop + P)
    inner = (slice(sy.start - ya, sy.stop - ya), slice(sx.start - xa, sx.stop - xa))
    sg = float(pc.get("align_refine_sigma_px", 1.0))
    src = {k: np.ascontiguousarray(y[k][ya:yb, xa:xb], dtype=np.float32) for k in SEQUENCE}
    if sg > 0:
        src = {k: cv2.GaussianBlur(v, (0, 0), sg) for k, v in src.items()}
    bb = 0.5 * (align(src["B1"], out["B1"])[inner] + align(src["B2"], out["B2"])[inner])
    cs = [1.0 / float(g_both) if g_both and g_both > 0 else 1.0]        # the LED scale c (nuisance)
    names = ("A", "L", "R")
    s = {k: np.array(out[k], float) for k in names}
    # the Jacobian over the whole window, rows zeroed off the usable mask (no gathers or copies
    # per iteration: the normal equations are 7 x 7 BLAS products, accumulated in float32)
    wm = m.astype(np.float32).ravel()
    n_m = float(wm.sum())
    Jt = np.empty((7, wm.size), np.float32)

    def residual(warped):
        c = np.float32(cs[0])
        r = (c * bb + (np.float32(2.0) - c) * warped["A"] - warped["L"] - warped["R"]).ravel() * wm
        return r, math.sqrt(float(np.dot(r, r)) / n_m)

    def jacobian(warped):                            # d/ds of F(x + s) = grad F at x + s; d/dc
        for i, (k, sgn) in enumerate(zip(names, (2.0 - cs[0], -1.0, -1.0))):
            gy, gx = np.gradient(warped[k])
            np.multiply(gx.ravel(), np.float32(sgn) * wm, out=Jt[2 * i])
            np.multiply(gy.ravel(), np.float32(sgn) * wm, out=Jt[2 * i + 1])
        np.multiply((bb - warped["A"]).ravel(), wm, out=Jt[6])
        return (Jt @ Jt.T).astype(np.float64)

    rms0 = rms = None
    n_it = 0
    for n_it in range(1, iters + 1):
        warped = {k: align(src[k], s[k])[inner] for k in names}
        r, rms = residual(warped)
        rms0 = rms if rms0 is None else rms0
        H = jacobian(warped)
        tr = float(np.trace(H[:6, :6]))
        if not tr > 1e-20:                           # no structure at all (a blank, noise-free plate)
            return out, {"skipped": "no structure", "iters": n_it}
        damp = np.full(7, 1e-3 * tr / 6.0)           # damped (LM-style); the scale c on its own scale
        damp[6] = 1e-3 * H[6, 6] + 1e-30
        H += np.diag(0.1 * np.diag(H) + damp)
        d = np.linalg.solve(H, -(Jt @ r).astype(np.float64))
        for i, k in enumerate(names):
            s[k] = s[k] + d[2 * i: 2 * i + 2]
        cs[0] += float(d[6])
        if float(np.abs(d[:6]).max()) < 0.005:
            break
    warped = {k: align(src[k], s[k])[inner] for k in names}
    _, rms1 = residual(warped)
    # standard error of each frame's shift: residual variance x (J'J)^-1, inflated by the
    # correlation area of the smoothed noise (4 pi sigma^2 px)
    cov = np.linalg.pinv(jacobian(warped)) * (rms1 * rms1) * max(1.0, 4.0 * math.pi * sg * sg)
    z_min = float(pc.get("align_refine_z", 3.0))
    corr, kept = {}, []
    for i, k in enumerate(names):
        c = s[k] - np.asarray(out[k], float)
        se = math.sqrt(max(float(cov[2 * i, 2 * i] + cov[2 * i + 1, 2 * i + 1]), 0.0))
        corr[k] = [round(float(c[0]), 4), round(float(c[1]), 4)]
        if math.hypot(*c) > z_min * se:              # a real correction, not a fit to the noise
            kept.append(k)
    info = {"correction": corr, "kept": kept, "iters": n_it, "closure_rms_before": rms0, "closure_rms_after": rms1,
            "led_scale": round(1.0 / cs[0], 5) if cs[0] > 0 else None}
    if rms1 > rms0 or max(math.hypot(*c) for c in corr.values()) > lim:
        info["skipped"] = "not trusted (beyond align_refine_max_px, or the closure did not drop)"
        info["kept"] = []
        return out, info
    for k in kept:
        out[k] = [float(s[k][0]), float(s[k][1])]
    return out, info


# ---- processing -----------------------------------------------------------------------------

def _wb(cfg: Mapping[str, Any]) -> np.ndarray:
    c = np.asarray(((cfg or {}).get("leds") or {}).get("colour", [1.0, 1.0, 1.0]), np.float32)
    return (1.0 / np.maximum(c, 1e-6)).astype(np.float32)


def _load_flat(flat, shape: Sequence[int]) -> np.ndarray | None:
    if flat is None:
        return None
    f = np.load(flat) if isinstance(flat, (str, Path)) else np.asarray(flat)
    f = f.astype(np.float32, copy=False)
    if f.shape[:2] != tuple(shape[:2]):
        raise ValueError(f"flat-field shape {f.shape} != image shape {tuple(shape[:2])}")
    return f if f.ndim == 3 else f[..., None]


def make_flat(white: Burst | np.ndarray, roi_mask: np.ndarray | None = None, smooth_px: float = 8.0) -> np.ndarray:
    """Flat-field gain map (H, W) for M from a matte white card: median(roi) / smoothed(card).

    card = what M is for the card: per pixel the darker of lum(L - A) and lum(R - A), each
    smoothed. Each pod falls off towards its far side, so the falloff of M is the min of the
    two, not that of B (= L + R): a flat from B1 - A would leave a tent across M. A burst
    without L and R uses lum(B1 - A); a plain image is used as is (e.g. a card's M_lin).
    """
    if isinstance(white, Burst):
        f = white.frames
        A = luma(f["A"]) if "A" in f else 0.0
        if "L" in f and "R" in f:
            yl = cv2.GaussianBlur(np.maximum(luma(f["L"]) - A, 0.0), (0, 0), smooth_px)
            yr = cv2.GaussianBlur(np.maximum(luma(f["R"]) - A, 0.0), (0, 0), smooth_px)
            y = np.minimum(yl, yr)
        else:
            y = cv2.GaussianBlur(np.asarray(luma(f["B1"]) - A, np.float32), (0, 0), smooth_px)
    else:
        y = cv2.GaussianBlur(luma(white), (0, 0), smooth_px)
    roi = _roi(roi_mask, y.shape, 0)
    med = float(np.median(y[roi])) if roi.any() else float(np.median(y))
    return (med / np.maximum(y, 1e-6 * max(med, 1e-6))).astype(np.float32)


def normalise_u8(M: np.ndarray, mask: np.ndarray | None = None, cfg: Mapping[str, Any] | None = None,
                 norm: float | None = None) -> tuple[np.ndarray, float]:
    """Deterministic display encoding of a linear RGB image (DESIGN_STEEL E 7, after WB/flat).

    norm = the process.percentile of the max channel over `mask` (bool, None = all pixels),
    taken on process.percentile_bin bins so sensor noise cannot move it; then
    uint8(255 * clip(M / norm, 0, 1) ** (1 / gamma)). Pass norm to reuse a scale (e.g. for
    a clean pass). Returns (image_u8, norm). The twin's noise-free reference images can use
    the same function to match the device output (process_burst's image_u8 equals
    normalise_u8(M_lin, eroded roi & ~sat_mask, cfg) except at inpainted pixels).
    """
    pc = _proc(cfg)
    if norm is None:
        nb = max(int(pc["percentile_bin"]), 1)
        m = np.ones(M.shape[:2], bool) if mask is None else np.asarray(mask, bool)
        keep = _bin(m.astype(np.float32), nb) > 0.999 if nb > 1 else m
        pv = _maxc(_bin(M, nb) if nb > 1 else M)[keep]
        norm = float(np.percentile(pv, float(pc["percentile"]))) if pv.size else 0.0
        norm = norm if norm > 1e-6 else 1.0
    return _gamma_u8(np.asarray(M, np.float32), 1.0 / float(norm), float(pc["gamma"])), float(norm)


def _gamma_u8(x: np.ndarray, scale: float, gamma: float) -> np.ndarray:
    """uint8 of clip(x * scale, 0, 1) ** (1 / gamma) through a 16-bit LUT (fast; < 0.1 DN off)."""
    v = x * np.float32(scale * 65535.0)
    np.clip(v, 0.0, 65535.0, out=v)
    return _gamma_lut(float(gamma))[v.astype(np.uint16)]


@lru_cache(maxsize=4)
def _gamma_lut(gamma: float) -> np.ndarray:
    t = np.arange(65536, dtype=np.float64) / 65535.0
    return np.clip(np.rint(255.0 * t ** (1.0 / gamma)), 0, 255).astype(np.uint8)


def _noise_var_bin(y_signal_bin: np.ndarray, n_frames: int, sensor: Mapping[str, Any], b: int,
                   demosaiced: bool) -> np.ndarray:
    """Expected variance of a b x b binned luma sum/difference of n_frames frames whose binned
    lumas add up to y_signal_bin (shot + read + quantisation noise).

    Shot noise is taken per channel as if R = G = B = luma (exact for grey; ~10 % high for
    strongly coloured pixels such as rust, i.e. slightly lenient)."""
    e = electrons_per_fs(sensor)
    rn = float(sensor.get("read_noise_e", 0.0)) / e
    q = 1.0 / max(raw_levels(sensor), 1)
    # Bayer: only 1/4 (R, B) or 1/2 (G) of the pixels in a bin are real samples
    samples = np.array([0.25, 0.5, 0.25] if demosaiced else [1.0, 1.0, 1.0], np.float32)
    wsum = float((LUMA_W ** 2 / samples).sum())
    per_px = np.maximum(y_signal_bin, 0.0) * np.float32(1.0 / e) + np.float32(n_frames * (rn * rn + q * q / 12.0))
    return per_px * np.float32(wsum / (b * b))


def _trimmed_z(d_bin: np.ndarray, var_bin: np.ndarray, keep: np.ndarray) -> float:
    """Trimmed RMS of d / sigma over the kept bins (top TRIM of z^2 dropped; 1.0 = pure noise)."""
    if not keep.any():
        return float("nan")
    z2 = (d_bin[keep].astype(np.float64) ** 2 / np.maximum(var_bin[keep], 1e-12))
    n = max(1, int(round(z2.size * (1.0 - TRIM))))
    z2 = np.partition(z2, n - 1)[:n] if n < z2.size else z2
    return float(math.sqrt(z2.mean() / TRIM_C))


def _change_stats(y: Mapping[str, np.ndarray], roi: np.ndarray, unsat: np.ndarray, sensor: Mapping[str, Any],
                  b: int, demosaiced: bool, y_raw: Mapping[str, np.ndarray] | None = None,
                  pc: Mapping[str, Any] | None = None, motion_px: float = 0.0) -> tuple[float, float, int]:
    """(residual_sigma, closure_sigma, ghost_bins, ghost_bins_coarse) on b x b bins of the aligned lumas y.

    residual: B2 - B1 (motion left over, flicker, moving shadows between the brackets).
    closure: B + A - L - R for B = B1 and B2, which is 0 + noise when the ambient is the same
    in every frame (the LED terms cancel); it sees changes confined to the middle frames.
    Both are trimmed RMS z over the ROI, blind to a local ghost; ghost_bins counts those
    (_ghost_bins, needs y_raw = the unaligned lumas of B1 and A), and ghost_bins_coarse the
    same test on process.ghost_coarse_px blocks (0 when off).
    """
    keep = (_bin(roi.astype(np.float32), b) > 0.999) & (_bin(unsat.astype(np.float32), b) > 0.999)
    yb = {k: _bin(v, b) for k, v in y.items()}
    resid = _trimmed_z(yb["B2"] - yb["B1"], _noise_var_bin(yb["B1"] + yb["B2"], 2, sensor, b, demosaiced), keep)
    mid = yb["A"] + yb["L"] + yb["R"]
    clos, dev, z, ds = [], [], [], []
    for k in ("B1", "B2"):
        d = yb[k] + yb["A"] - yb["L"] - yb["R"]
        var = _noise_var_bin(yb[k] + mid, 4, sensor, b, demosaiced)
        clos.append(_trimmed_z(d, var, keep))
        ds.append(d)
        dev.append(np.abs(d))
        z.append(np.abs(d) / np.sqrt(np.maximum(var, 1e-12)))
    ghost = coarse = 0
    if y_raw is not None:
        pc = pc or DEFAULTS
        led = yb["B1"] - yb["A"]
        kg = _ghost_keep(keep, _bin(y_raw["B1"], b), _bin(y_raw["A"], b), pc, b, motion_px)
        ghost = int(np.count_nonzero(_ghost_hits(dev, z, led, kg, pc, yb["A"])))
        f = int(pc.get("ghost_coarse_px", 0) or 0) // b
        if f > 1:
            # the same test on f x f blocks of bins: a change that is coherent over a region but
            # below the noise of one bin (flicker under a mirrored lamp, a soft moving shadow). Both
            # closures must see it (process.ghost_coarse_closure, min): flicker moves both, while the
            # B1 / B2 alignment residue at the pods' brushed-metal highlights moves one and adds up
            # over a block (dark-room open captures of brushed stainless: 3-12 blocks with max)
            kc = _bin(kg.astype(np.float32), f) > 0.999
            # an ambient-made ghost needs ambient: blocks where A is at least ghost_coarse_min_r x the
            # LED level (in the dark room, near-black brushed stainless at ~7 e- gave noise-level z 6-7)
            kc &= _bin(yb["A"], f) >= float(pc.get("ghost_coarse_min_r", 1.0)) * np.maximum(_bin(led, f), 0.0)
            if kc.any():
                devc, zc = [], []
                for k, d in zip(("B1", "B2"), ds):
                    dc = np.abs(_bin(d, f))
                    varc = _noise_var_bin(_bin(yb[k] + mid, f), 4, sensor, b * f, demosaiced)
                    devc.append(dc)
                    zc.append(dc / np.sqrt(np.maximum(varc, 1e-12)))
                pcc = {**pc, "ghost_closure": pc.get("ghost_coarse_closure", "min")}
                coarse = int(np.count_nonzero(_ghost_hits(devc, zc, _bin(led, f), kc, pcc)))
    return resid, max(clos), ghost, coarse


def _ghost_bins(dev: Sequence[np.ndarray], z: Sequence[np.ndarray], led: np.ndarray, keep: np.ndarray,
                b1_raw: np.ndarray, a_raw: np.ndarray, pc: Mapping[str, Any], b: int, motion_px: float) -> int:
    """Bins where M has a local ghost: a closure |B + A - L - R| (dev, one per bracket frame
    B1, B2, with its noise z) that is significant (z > ghost_z) and large (> ghost_contrast x
    the LED level B1 - A); process.ghost_closure (one value, or {illumination: value}) says
    whether both closures must be (min) or either one (max).

    The closure is the error that a misaligned or changed A (or L, R) leaves in L - A and
    R - A, i.e. in M. min ignores a change in one bracket frame only, which M does not use,
    but the five frames cannot tell that apart from a step between L and A (or A and R),
    which M does use: per-frame ambient offsets e_L = a_L - a_A, e_R = a_R - a_A, b_i =
    a_Bi - a_A give closures b_1 - e_L - e_R and b_2 - e_L - e_R, so a B2-only blip and a
    step after L both read (0, a). max refuses both (full-resolution pilot, 15 Nov: a moving
    hand shadow left M 56 % dark locally, -14 DN, in an accepted open capture with min).
    A lamp reflection in A that moves or flickers, or that alignment misses by a fraction of
    a px at its sharp edge, gives this ghost in a small area that the trimmed residual_sigma
    and closure_sigma cannot see. LED highlights of the raw B1 (> ghost_hl_factor x the
    median, ambient frame dark there) are left out, grown by the motion: on the device they
    stay put while the plate moves.
    """
    k = _ghost_keep(keep, b1_raw, a_raw, pc, b, motion_px)
    return int(np.count_nonzero(_ghost_hits(dev, z, led, k, pc)))


def _ghost_keep(keep: np.ndarray, b1_raw: np.ndarray, a_raw: np.ndarray, pc: Mapping[str, Any], b: int,
                motion_px: float) -> np.ndarray:
    """keep minus the LED highlights of the raw B1, grown by the motion (see _ghost_bins)."""
    k = keep.copy()
    if not k.any():
        return k
    med_b = float(np.median(b1_raw[k]))
    hl = (b1_raw > float(pc.get("ghost_hl_factor", 3.0)) * med_b) & (a_raw < 0.5 * b1_raw)
    if hl.any():
        n = 1 + int(math.ceil(max(float(motion_px), 0.0) / b))
        k &= ~cv2.dilate(hl.view(np.uint8), np.ones((2 * n + 1, 2 * n + 1), np.uint8)).astype(bool)
    return k


def _ghost_hits(dev: Sequence[np.ndarray], z: Sequence[np.ndarray], led: np.ndarray, k: np.ndarray,
                pc: Mapping[str, Any], amb: np.ndarray | None = None) -> np.ndarray:
    """Bool map of the kept bins with a ghost (process.ghost_z, ghost_contrast, ghost_closure).

    ghost_closure: min = both closures flag the bin, max = either does, max_ambient = both, or
    either where the ambient frame there could have made the deviation (amb x ghost_amb_factor >=
    the deviation; amb = the binned A luma). max_ambient is the ring's rule (second review, 15 Nov):
    max refused still hood captures in the dark room (B1/B2 alignment residue at sharp dark-field
    features, LED-made: no ambient there), while min let an ambient step after the L frame through
    under a leaking hood (a hand shadow arriving at the A frame, sun through a lifted rim: accepted,
    +9 DN over 70 mm^2). An ambient-made ghost needs ambient; without amb it is min."""
    if not k.any():
        return k
    med = float(np.median(led[k]))
    den = np.maximum(led, 0.25 * max(med, 1e-6))
    gz, gc = float(pc.get("ghost_z", 6.0)), float(pc.get("ghost_contrast", 0.1))
    flags = [(zi > gz) & (di / den > gc) for di, zi in zip(dev, z)]
    mode = str(pc.get("ghost_closure", "min"))
    if mode not in ("min", "max", "max_ambient"):
        raise ValueError(f"process.ghost_closure must be min, max or max_ambient, got {mode!r}")
    hit = np.logical_and.reduce(flags)
    if mode == "max":
        hit = np.logical_or.reduce(flags)
    elif mode == "max_ambient" and amb is not None:
        fa = np.maximum(amb, 0.0) * np.float32(pc.get("ghost_amb_factor", 1.0))
        hit = hit | np.logical_or.reduce([f & (fa >= di) for f, di in zip(flags, dev)])
    return k & hit


def _led_both_ratio(y: Mapping[str, np.ndarray], keep: np.ndarray, b: int) -> float | None:
    """Median over the well-lit b x b bins of (B1 + B2 - 2A) / (2 (L + R - 2A)): the brightness of
    both LED channels together relative to the sum of each alone (1.0 = no droop; the closure
    model). Diagnostic for leds.both_on_gain (a white card in the dark gives the clean value);
    None without bins whose single-channel sum is >= 0.5 x its median."""
    kb = _bin(keep.astype(np.float32), b) > 0.999
    if not kb.any():
        return None
    yb = {k: _bin(v, b) for k, v in y.items()}
    den = yb["L"] + yb["R"] - 2.0 * yb["A"]
    num = 0.5 * (yb["B1"] + yb["B2"]) - yb["A"]
    med = float(np.median(den[kb]))
    ok = kb & (den >= 0.5 * med) if med > 0 else kb & False
    return round(float(np.median(num[ok] / den[ok])), 5) if ok.any() else None


def _ambient_dominated(y_a: np.ndarray, y_led: np.ndarray, roi: np.ndarray, pc: Mapping[str, Any],
                       floor: float = 1e-6) -> float:
    """Share of the ROI (process.local_r_px blocks inside it) where the ambient frame is more than
    process.local_r_max x the LED level B1 - A: there the ambient's shot noise in L - A and R - A
    outweighs the LED signal, so M is noise, biased bright by the clamp at 0. A lamp mirrored in a
    dark plate does this locally while the median r stays low (pilot 15 Nov: local r 22, M +236 %,
    +21 DN, at r 0.31). The LED level is floored at `floor` (FS, process_burst: 3 x the read
    noise): a block with no LED signal is too dark, not ambient-dominated (review 15 Nov: a
    mirror-polished plate in the DARK room was refused as 'Lamp reflection on the part', A = read
    noise over B1 - A = 0.2 e-). NaN without a whole block inside the ROI."""
    bs = int(pc.get("local_r_px", 32) or 0)
    if bs <= 1:
        return 0.0
    kb = _bin(roi.astype(np.float32), bs) > 0.999
    if not kb.any():
        return float("nan")
    rl = _bin(y_a, bs) / np.maximum(_bin(y_led, bs), max(float(floor), 1e-6))
    return float((rl[kb] > float(pc.get("local_r_max", 8.0))).mean())


def _ambient_flicker(y: Mapping[str, np.ndarray], keep_px: np.ndarray, y_m: np.ndarray, sensor: Mapping[str, Any],
                     pc: Mapping[str, Any], floor: float = 1e-6) -> dict[str, Any] | None:
    """Ambient change between frames that is coherent over the whole plate -> {'kappa', 'se', 'n_bins',
    'a_over_m', 'flicker', 'err'}, or None when the ambient frame is too dark to measure it (dark room,
    sealed hood: then it cannot matter either).

    With per-frame ambient factors the closures B_i + A - L - R read (b_i - e_L - e_R) A and B2 - B1
    reads (b_2 - b_1) A (b_i = a_Bi / a_A - 1, e_L = a_L / a_A - 1, e_R = a_R / a_A - 1), plus LED
    ripple or droop times the LED image. A flicker of a few % moves no 4 x 4 bin beyond its noise and
    no 32 px block beyond ghost_contrast, yet M = L - A (R - A) carries e x A, many times M where the
    ambient outshines the darker pod (second review, 15 Nov: 5 % random flicker at r 0.11 on
    cold-rolled steel was accepted with M +15 DN over 8 % of the plate; ghost_bins 0, closure_sigma
    1.5). Each of the three is regressed on [A, B1 - A] over process.flicker_bin bins (keep_px whole,
    A above 10 x its noise; weighted, one 4-MAD trimmed refit so local changes do not count): kappa =
    the A coefficient, se = 3 x its standard error (spatially correlated residuals). flicker = the
    largest |kappa| above process.flicker_z se (0.0 if none), err = flicker x the process.flicker_q
    percentile of A / M over the kept bins (M = y_m, the combined LED-only luma, floored at `floor`):
    the relative M error such a change makes there. Only b_i - e_L - e_R are observable: a change that
    keeps both closures at 0 (e_L = -e_R with steady brackets) still passes. For random flicker the
    observable combinations are as large as e_L and e_R, so most of it is caught."""
    b = max(int(pc.get("flicker_bin", 8) or 0), 2)
    keep = _bin(keep_px.astype(np.float32), b) > 0.999
    if int(keep.sum()) < 50:
        return None
    yb = {k: _bin(v, b) for k, v in y.items()}
    A = yb["A"]
    e = electrons_per_fs(sensor)
    rn = float(sensor.get("read_noise_e", 0.0)) / e
    good = keep & (A > 10.0 * np.sqrt(np.maximum(A, 0.0) / e + rn * rn) / b)
    good &= cv2.erode(keep.view(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if int(good.sum()) < 50:
        return None
    # regressors from the 8 neighbouring bins: a bin's own noise of A (and B1) is in the closures too,
    # and would bias kappa towards +1 x noise / spatial variance of A (errors in variables)
    nb = lambda x: (cv2.boxFilter(x, -1, (3, 3), normalize=False, borderType=cv2.BORDER_REFLECT) - x) / 8.0  # noqa: E731
    X = np.stack([nb(A)[good], nb(yb["B1"] - A)[good]], 1).astype(np.float64)
    w = 1.0 / ((np.maximum(yb["B1"] + A + yb["L"] + yb["R"], 0.0)[good] / e + 4.0 * rn * rn) / (b * b) + 1e-30)
    w = w.astype(np.float64)
    kappa, se = {}, {}
    for name, c in (("B1", yb["B1"] + A - yb["L"] - yb["R"]), ("B2", yb["B2"] + A - yb["L"] - yb["R"]),
                    ("B2-B1", yb["B2"] - yb["B1"])):
        t = c[good].astype(np.float64)
        sel = np.ones(t.size, bool)
        for _ in range(2):
            sw = np.sqrt(w[sel])
            coef = np.linalg.lstsq(X[sel] * sw[:, None], t[sel] * sw, rcond=None)[0]
            res = t - X @ coef
            new = np.abs(res) <= 4.0 * 1.4826 * float(np.median(np.abs(res[sel]))) + 1e-30
            sel = new if int(new.sum()) >= 10 else sel
        r = res[sel]
        cov = np.linalg.pinv((X[sel] * w[sel][:, None]).T @ X[sel]) * float(np.mean(r * r * w[sel]))
        kappa[name] = round(float(coef[0]), 5)
        se[name] = round(3.0 * math.sqrt(max(float(cov[0, 0]), 0.0)), 6)
    zmin = float(pc.get("flicker_z", 5.0))
    sig = [abs(kappa[k]) for k in kappa if abs(kappa[k]) > zmin * se[k]]
    ratio = A[keep] / np.maximum(_bin(y_m, b)[keep], max(float(floor), 1e-6))
    q = float(np.percentile(ratio, float(pc.get("flicker_q", 99.0))))
    flick = max(sig, default=0.0)
    return {"kappa": kappa, "se": se, "n_bins": int(good.sum()), "a_over_m": round(q, 4),
            "flicker": round(flick, 5), "err": round(flick * q, 5)}


def _blur(y_norm: np.ndarray, roi: np.ndarray, b: int) -> float:
    """Variance of the Laplacian of the normalised luma (b x b binned) inside the ROI."""
    yb = _bin(y_norm, b) if b > 1 else y_norm
    rb = _bin(roi.astype(np.float32), b) > 0.999 if b > 1 else roi
    lap = cv2.Laplacian(yb, cv2.CV_32F)
    rb = cv2.erode(rb.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    v = lap[rb]
    return float(v.var()) if v.size else float("nan")


def _blur_ref(cfg: Mapping[str, Any], meta: Mapping[str, Any]) -> float | None:
    ref = cfg.get("blur_ref", (cfg.get("gate") or {}).get("blur_ref"))
    if isinstance(ref, Mapping):
        ref = ref.get(meta.get("material"), ref.get("default"))
    return float(ref) if ref else None


def process_burst(burst: Burst, roi_mask: np.ndarray | None = None, cfg: Mapping[str, Any] | None = None,
                  *, flat=None) -> Processed:
    """Burst -> glare-free, ambient-free detector image + metrics + gate (DESIGN_STEEL E 1-9).

    sat_mask marks pixels M cannot recover. min_luma (pods): A saturated, or both L and R
    saturated; where only one pod saturates, M takes the other pod (that is what the L/R
    split is for). mean (ring): A, L or R saturated (M there is the mean of the clipped
    values: a lower bound). sat_frac_any reports the share of the ROI where any of B1, L, A,
    R, B2 saturates, sat_frac_b where B1 or B2 does. When the gate does not refuse, image_u8
    has the sat_mask pixels inside roi_mask inpainted (process.inpaint_px; metric
    inpainted_px), except, for the ring, LED-clipped pixels (A unclipped) whose lower bound is
    display white in every channel: image_u8 is exact there (metric clipped_white_px). M_lin
    keeps every pixel as it is.
    burst.meta (optional): 'illumination' pods | ring (process.combine; default pods),
    'px_scale' (frame px per full-resolution sensor px, default 1: motion_px is reported in
    sensor px, the unit of gate.motion_max_px), 'demosaiced' True for Bayer data from
    raw_to_rgb (noise model of residual_sigma / closure_sigma), 'E_led' (W/m^2, twin) for
    lux_est, 'material' for a per-material blur_ref.
    flat (optional): gain map (H, W) or (H, W, 3) or .npy path, e.g. from make_flat;
    cfg['process']['flat'] is used when not given.
    """
    t0 = time.perf_counter()
    cfg = cfg if cfg is not None else load_capture_config()
    pc = _proc(cfg)
    illum = str(burst.meta.get("illumination") or "pods")
    sensor = sensor_for(cfg.get("sensor"), illum)                 # per-illumination gain (noise model)
    missing = [k for k in SEQUENCE if k not in burst.frames]
    if missing:
        raise ValueError(f"burst is missing frames {missing} (need {SEQUENCE})")
    fr = {k: np.ascontiguousarray(burst.frames[k][..., :3], dtype=np.float32) for k in SEQUENCE}
    shape = fr["B1"].shape
    if any(f.shape != shape for f in fr.values()):
        raise ValueError("burst frames differ in shape")
    roi = _roi(roi_mask, shape, int(pc["roi_erode_px"]))
    st, sat = int(pc["stat_stride"]), float(pc["sat_level"])
    comb = combine_for(pc, illum)
    px_scale = float(burst.meta.get("px_scale") or 1.0)

    # 1-2. align L, A, R, B2 to B1 (translation, phase correlation on luma)
    y = {k: luma(v) for k, v in fr.items()}
    braced = pc.get("align_braced") or {}
    braced = bool(braced.get(illum, False) if isinstance(braced, Mapping) else braced)
    # LED droop with both channels on (leds.both_on_gain, measured on a white card: B - A = g (L + R - 2A)).
    # M never uses B1/B2, but the closure B + A - L - R reads -(1 - g) x the LED image: a 5 % sag
    # refused every capture as 'changed', even in the dark, and the closure refinement shifted A, L, R
    # to fit it (review, 15 Nov). The refinement and the closure statistics use B' = A + (B - A) / g;
    # led_both_ratio reports the measured g of this burst (the day-1 calibration).
    g_both = _limit(cfg.get("leds") or {}, "both_on_gain", 1.0, illum)
    shifts, resp, fallbacks = burst_shifts(y, roi, pc, braced)
    refine: dict[str, Any] = {"skipped": "off"}
    if pc.get("align_refine", True):
        # clipped pixels break B = L + R + A: leave them (and a margin for the motion) out
        clip = np.zeros(shape[:2], np.uint8)
        for k in SEQUENCE:
            clip |= (_maxc(fr[k]) >= sat).view(np.uint8)
        grow = 2 + int(math.ceil(max(math.hypot(*shifts[k]) for k in SEQUENCE)
                                 + 3.0 * float(pc.get("align_refine_sigma_px", 1.0))))
        usable = ~cv2.dilate(clip, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)
        shifts, refine = refine_shifts(y, shifts, roi, usable, pc, g_both)
    al = {k: (fr[k] if k == "B1" else align(fr[k], shifts[k])) for k in SEQUENCE}
    ya = {k: (y[k] if k == "B1" else luma(al[k])) for k in SEQUENCE}
    motion = float(math.hypot(*shifts["B2"]))                     # frame px (ghost_bins); gate: / px_scale

    # 3. ambient subtraction (linear raw)
    A = al["A"]
    Lc = al["L"] - A
    np.maximum(Lc, 0.0, out=Lc)
    Rc = al["R"] - A
    np.maximum(Rc, 0.0, out=Rc)
    yA = ya["A"]
    yBA = y["B1"] - yA

    # 4. saturation, thresholded on the raw frames and moved like them (_move_mask)
    a_max = _maxc(A)
    sm = {k: _move_mask(_maxc(fr[k]) >= sat, shifts[k]) for k in SEQUENCE}
    l_sat, r_sat, a_sat, b_sat = sm["L"], sm["R"], sm["A"], sm["B1"]
    sat_mask = a_sat | ((l_sat | r_sat) if comb == "mean" else (l_sat & r_sat))
    bb_sat = b_sat | sm["B2"]
    any_sat = a_sat | l_sat | r_sat | bb_sat
    gcl = pc.get("ghost_closure", "min")
    gcl = str(gcl.get(illum, "min") if isinstance(gcl, Mapping) else gcl)
    yc = dict(ya)
    if abs(g_both - 1.0) > 1e-6:
        for k in ("B1", "B2"):
            yc[k] = ya["A"] + (ya[k] - ya["A"]) * np.float32(1.0 / g_both)
    resid, closure, ghost, ghost_c = _change_stats(yc, roi, ~any_sat, sensor, int(pc["resid_bin"]),
                                          bool(burst.meta.get("demosaiced", False)), y,
                                          {**pc, "ghost_closure": gcl}, motion)
    led_both = _led_both_ratio(ya, roi & ~any_sat, int(pc["resid_bin"]))
    amb_mask = sat_mask & (a_max >= 0.5 * sat)                    # masked mostly by ambient, not LED glare

    # 5. combine L and R (process.combine by illumination)
    yl, yr = luma(Lc), luma(Rc)
    # 6. relief (dents, punch marks)
    relief = ((yl - yr) / (yl + yr + np.float32(1e-4))).astype(np.float32)
    if comb == "mean":
        # ring (dark-field): the mean of the two halves keeps both directional signals. Where a
        # half clips, the mean of the clipped values is a lower bound (see the fill mask in 8.)
        M = cv2.addWeighted(Lc, 0.5, Rc, 0.5, 0.0)
    else:
        # pods: luma-guided min(Lc, Rc), the pod whose (lightly smoothed) luma is darker
        sg = float(pc["min_guide_sigma_px"])
        gl, gr = (cv2.GaussianBlur(yl, (0, 0), sg), cv2.GaussianBlur(yr, (0, 0), sg)) if sg > 0 else (yl, yr)
        pick_l = np.where(l_sat ^ r_sat, r_sat, gl <= gr)        # a saturated pod yields to the other
        M = Rc                                                    # Rc is ours: reuse it as M
        M = cv2.copyTo(Lc, pick_l.view(np.uint8), M)
    y_m = luma(M)                                                 # (raw units, for the ambient-to-M ratio)

    # 7. fixed WB, optional flat-field, ROI percentile, gamma, uint8
    M = cv2.multiply(M, tuple(float(v) for v in _wb(cfg)) + (0.0,), dst=M)
    ff = _load_flat(flat if flat is not None else pc.get("flat"), shape)
    if ff is not None:
        M *= ff
    image_u8, norm = normalise_u8(M, roi & ~sat_mask, cfg)

    # 8. metrics
    med_a, med_ba = _median(yA, roi, st), _median(yBA, roi, st)
    if not (math.isfinite(med_a) and math.isfinite(med_ba)):
        r = float("nan")                                          # no ROI: unknown, never 0 (the gate says invalid)
    else:
        r = med_a / med_ba if med_ba > 1e-6 else (R_CAP if med_a > 1e-6 else 0.0)
        r = float(min(max(r, 0.0), R_CAP))
    e = electrons_per_fs(sensor)
    rn = float(sensor.get("read_noise_e", 0.0) or 0.0)
    amb_dom = _ambient_dominated(yA, yBA, roi, pc, 3.0 * rn / e)
    flick = _ambient_flicker(yc, roi & ~any_sat, y_m, sensor, pc, 3.0 * rn / e)
    med_b1 = _median(y["B1"], roi, st)
    # per-pixel SNR of B1 - A in the ROI median (shot noise of B1 and A plus two reads)
    snr = med_ba * e / math.sqrt(max(med_b1, 0.0) * e + max(med_a, 0.0) * e + 2.0 * rn * rn + 1e-9) if med_ba > 0 else 0.0
    roi_n = max(int(roi[::st, ::st].sum()), 1)
    frac = lambda m: float(m[::st, ::st][roi[::st, ::st]].sum()) / roi_n   # noqa: E731
    blur = _blur(luma(M) / np.float32(norm), roi, int(pc["blur_bin"]))
    ref = _blur_ref(cfg, burst.meta)
    e_led = burst.meta.get("E_led")
    # E_led is power-based (Blender W/m^2, no colour weighting): lux = 683 x E x luma(LED colour)
    led_lux = (float(e_led) * 683.0 * float(LUMA_W @ (1.0 / _wb(cfg))) if e_led
               else float((cfg.get("leds") or {}).get("led_lux_at_plate", 0.0)))
    metrics: dict[str, Any] = {
        "r_ambient": r,
        "sat_frac_a": frac(a_sat),
        "led_level": float(med_ba),
        "blur": blur,
        "blur_rel": (blur / ref) if ref else None,
        "motion_px": motion / px_scale,                           # full-resolution sensor px
        "motion_frame_px": motion,
        "residual_sigma": resid,
        "closure_sigma": closure,
        "ghost_bins": ghost,
        "ghost_bins_coarse": ghost_c,
        "ghost_closure": gcl,
        "masked_frac": frac(sat_mask),
        "masked_ambient_frac": frac(amb_mask),
        "ambient_dominated_frac": amb_dom,
        "ambient_flicker": flick["flicker"] if flick else None,
        "ambient_flicker_err": flick["err"] if flick else None,
        "ambient_flicker_fit": flick,
        "sat_frac_any": frac(any_sat),
        "sat_frac_b": frac(bb_sat),
        "illumination": illum,
        "combine": comb,
        "shifts": shifts,
        "align_response": resp,
        "align_fallback": fallbacks,
        "align_mode": "braced" if braced else "free",
        "align_refine": refine,
        "led_both_ratio": led_both,
        "snr": float(snr),
        "lux_est": float(r * led_lux) if led_lux and r < R_CAP else None,   # None: ambient saturates
        "norm": norm,
        "roi_px": int(roi.sum()),
    }
    # 9. gate
    gate = quality_gate(metrics, cfg)
    # unrecoverable pixels are black (A clipped: L - A = 0) or flat in M; on an accepted
    # capture the detector must not see them as pits or stains: fill them from around.
    # Ring exception: a pixel clipped in L or R only (A not) is light that relief scatters
    # into the lens (no ring LED is in the mirror direction of the plate), and its mean is a
    # lower bound. When that bound is display white in every channel, image_u8 is exact there
    # (255), so it is kept: inpainting erased the strongest dark-field signal of the defects
    # (full-resolution pilot, 15 Nov: a pit cluster 101 -> 3 DN contrast, a scratch 191 -> 16).
    # sat_mask, masked_frac and the gate are unchanged.
    fill = sat_mask
    metrics["clipped_white_px"] = 0
    if comb == "mean" and sat_mask.any():
        white = ~a_sat & (np.minimum(np.minimum(M[..., 0], M[..., 1]), M[..., 2]) >= np.float32(norm))
        fill = sat_mask & ~white
        metrics["clipped_white_px"] = int(np.count_nonzero(sat_mask & white & roi))
    n_inp = 0
    if int(pc.get("inpaint_px") or 0) > 0 and gate.verdict != "RETAKE" and fill.any():
        image_u8, n_inp = _inpaint(image_u8, fill, roi_mask, int(pc["inpaint_px"]))
    metrics["inpainted_px"] = n_inp
    metrics["process_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    return Processed(image_u8, M, relief, sat_mask, shifts, metrics, gate)


def _move_mask(m: np.ndarray, shift: Sequence[float]) -> np.ndarray:
    """A raw frame's bool mask moved by align(., shift): every aligned pixel that took any
    interpolation weight from a masked one is masked.

    Saturation is thresholded on the raw frames and moved this way. Thresholding after
    alignment would miss the edge of a clipped region: interpolation mixes the clipped value
    (an underestimate) with its neighbours and drops it below sat_level.
    """
    if abs(float(shift[0])) < 1e-6 and abs(float(shift[1])) < 1e-6:
        return m
    return align(m.astype(np.float32), shift) > 1e-3


def _inpaint(img: np.ndarray, mask: np.ndarray, roi_mask: np.ndarray | None, radius: int) -> tuple[np.ndarray, int]:
    """img with mask (grown by 1 px, limited to the plate roi_mask) filled by cv2.inpaint
    (Telea) on the masks' bounding box -> (img, pixels filled)."""
    t = cv2.dilate(mask.view(np.uint8), np.ones((3, 3), np.uint8))
    if roi_mask is not None:
        t &= (np.asarray(roi_mask) > 0).view(np.uint8)
    n = int(cv2.countNonZero(t))
    if n == 0:
        return img, 0
    x, y, w, h = cv2.boundingRect(t)
    pad = 2 * radius + 2
    y0, y1, x0, x1 = max(0, y - pad), min(img.shape[0], y + h + pad), max(0, x - pad), min(img.shape[1], x + w + pad)
    img[y0:y1, x0:x1] = cv2.inpaint(np.ascontiguousarray(img[y0:y1, x0:x1]), np.ascontiguousarray(t[y0:y1, x0:x1]),
                                    float(radius), cv2.INPAINT_TELEA)
    return img, n


def naive_process(frame_u8: np.ndarray) -> np.ndarray:
    """Baseline arm: the ISP output is used as is."""
    return np.asarray(frame_u8)


# ---- quality gate ---------------------------------------------------------------------------

def _finite(x) -> bool:
    if x is None or isinstance(x, (bool, str)):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _limit(g: Mapping[str, Any], key: str, default: float, illum: str) -> float:
    """A gate threshold: a number, or {illumination: number} (pods | ring; pods is the fallback)."""
    v = g.get(key, default)
    if isinstance(v, Mapping):
        v = v.get(illum, v.get("pods", default))
    return float(default if v is None else v)


def quality_gate(metrics: Mapping[str, Any], cfg: Mapping[str, Any]) -> GateResult:
    """PASS | PASS_FLAGGED | RETAKE from capture.yaml gate (cfg may be the whole config or its gate).

    RETAKE if r > leak_r_max (hood_leak; ring only in the shipped config: the seated hood sees almost
    no ambient); r > motion_strict_upto (too_bright); sat_frac_a too high: a local lamp reflection
    (sat_frac_a <= glare_frac_max at r <= glare_r_max: glare_ambient, 'tilt, shade or hood')
    or ambient flooding the sensor (too_bright); led_level too low (too_dark); motion_px
    (sensor px) > motion_max_px, or flag < r <= motion_strict_upto with motion_px >
    motion_strict_px (motion); masked_frac too high (saturated: LED glare, tilt; when most
    masked pixels have the ambient frame above half of sat_level, glare_ambient or
    too_bright as for sat_frac_a); masked_ambient_frac (the masked pixels the ambient
    frame clips) above masked_ambient_max (glare_ambient or too_bright); ambient_dominated_frac
    (where the ambient outshines the LEDs locally) above ambient_dominated_max (glare_ambient or
    too_bright); residual_sigma or closure_sigma (when present) above
    residual_sigma_max, or ghost_bins (when present) above ghost_bins_max, or ghost_bins_coarse
    above ghost_coarse_max, or ambient_flicker_err (a plate-wide ambient change between frames times
    the ambient-to-M ratio, _ambient_flicker) above flicker_err_max (changed);
    blur_rel too low (blur, only when a blur reference is configured); or a required metric
    missing or not finite: too_bright when at least SAT_TOO_BRIGHT of the ROI is saturated
    in B1/B2 or any frame (the statistics had no unclipped pixel left), else invalid.
    Otherwise PASS_FLAGGED if pass < r <= motion_strict_upto ('ambient_high'), else PASS.
    Any threshold may be {pods: x, ring: y}, picked by metrics['illumination'] (default
    pods). reasons keep detection order; message follows the order of gate.messages.
    """
    g = cfg.get("gate", cfg) if isinstance(cfg, Mapping) else {}
    illum = str(metrics.get("illumination") or "pods")
    ra = g.get("r_ambient") or {}
    r_pass, r_flag, r_upto = float(ra.get("pass", 0.3)), float(ra.get("flag", 1.0)), float(ra.get("motion_strict_upto", 2.0))
    reasons: list[str] = []
    details: list[str] = []

    def fail(key: str, text: str) -> None:
        if key not in reasons:
            reasons.append(key)
        details.append(text)

    def lim(key: str, default: float = math.inf) -> float:
        return _limit(g, key, default, illum)

    req = ("r_ambient", "sat_frac_a", "led_level", "motion_px", "masked_frac", "residual_sigma")
    bad = [k for k in req if not _finite(metrics.get(k))]
    r = float(metrics.get("r_ambient") or 0.0) if _finite(metrics.get("r_ambient")) else 0.0
    mot = float(metrics["motion_px"]) if _finite(metrics.get("motion_px")) else 0.0
    if bad:
        sat_share = max((float(metrics[k]) for k in ("sat_frac_b", "sat_frac_any") if _finite(metrics.get(k))), default=0.0)
        if sat_share >= SAT_TOO_BRIGHT:              # nothing unclipped left to compute them from
            fail("too_bright", f"metrics not finite ({', '.join(bad)}): {sat_share:.0%} of the plate saturated")
        else:
            fail("invalid", f"metrics not finite: {', '.join(bad)}")

    def ambient_glare(frac: float) -> bool:          # a local lamp reflection, not a bright room
        return r <= lim("glare_r_max", r_flag) and frac <= lim("glare_frac_max", 0.0)

    if r > lim("leak_r_max"):
        # the hood's dark-field ring sees almost no ambient when the hood is seated (twin pilot: r <= 0.021
        # in sun, <= 0.071 with the rim lifted 3 mm): more is light getting under it
        fail("hood_leak", f"r_ambient {r:.3f} > {lim('leak_r_max'):g} (light under the hood)")
    if r > r_upto:
        fail("too_bright", f"r_ambient {r:.2f} > {r_upto}")
    sfa = metrics.get("sat_frac_a")
    if _finite(sfa) and sfa > lim("sat_frac_a_max", 1.0):
        fail("glare_ambient" if ambient_glare(float(sfa)) else "too_bright",
             f"sat_frac_a {float(sfa):.4f} > {lim('sat_frac_a_max', 1.0)} at r_ambient {r:.2f}")
    if _finite(metrics.get("led_level")) and metrics["led_level"] < lim("led_level_min", 0.0):
        fail("too_dark", f"led_level {metrics['led_level']:.4f} < {lim('led_level_min', 0.0)}")
    if mot > lim("motion_max_px"):
        fail("motion", f"motion_px {mot:.2f} > {lim('motion_max_px')}")
    if r_flag < r <= r_upto and mot > lim("motion_strict_px"):
        fail("motion", f"motion_px {mot:.2f} > {lim('motion_strict_px')} at r_ambient {r:.2f}")
    mf = metrics.get("masked_frac")
    if _finite(mf) and mf > lim("masked_frac_max", 1.0):
        amb = float(metrics.get("masked_ambient_frac") or 0.0)
        by_amb = amb > 0.5 * mf                                   # ambient-driven: shade, not tilt
        key = ("glare_ambient" if ambient_glare(amb) else "too_bright") if by_amb else "saturated"
        fail(key, f"masked_frac {mf:.4f} > {lim('masked_frac_max', 1.0)}" + (" (ambient)" if by_amb else ""))
    ma = metrics.get("masked_ambient_frac")
    if _finite(ma) and ma > lim("masked_ambient_max", 1.0):
        # pixels the ambient frame clips are lost in every frame, and a defect that scatters the
        # ambient light (a scratch under a window) clips there itself: inpainting would erase it
        fail("glare_ambient" if ambient_glare(float(ma)) else "too_bright",
             f"masked_ambient_frac {float(ma):.4f} > {lim('masked_ambient_max', 1.0)}")
    ad = metrics.get("ambient_dominated_frac")
    if _finite(ad) and ad > lim("ambient_dominated_max", 1.0):
        # the ambient outshines the LEDs over part of the plate (a mirrored lamp): M is noise there
        fail("glare_ambient" if ambient_glare(float(ad)) else "too_bright",
             f"ambient_dominated_frac {float(ad):.3f} > {lim('ambient_dominated_max', 1.0)} at r_ambient {r:.2f}")
    rmax = lim("residual_sigma_max")
    for key in ("residual_sigma", "closure_sigma"):
        v = metrics.get(key)
        if _finite(v) and v > rmax:
            fail("changed", f"{key} {float(v):.2f} > {rmax}")
    gb = metrics.get("ghost_bins")
    if _finite(gb) and float(gb) > lim("ghost_bins_max"):
        fail("changed", f"ghost_bins {int(gb)} > {lim('ghost_bins_max'):g} (local ghost of a moved or flickering reflection)")
    gcb = metrics.get("ghost_bins_coarse")
    if _finite(gcb) and float(gcb) > lim("ghost_coarse_max"):
        fail("changed", f"ghost_bins_coarse {int(gcb)} > {lim('ghost_coarse_max'):g} (a regional ghost: flicker "
                        "under a mirrored lamp or a moving shadow)")
    fe = metrics.get("ambient_flicker_err")
    if _finite(fe) and float(fe) > lim("flicker_err_max"):
        fail("changed", f"ambient_flicker_err {float(fe):.3f} > {lim('flicker_err_max'):g} (the ambient changed "
                        f"by {float(metrics.get('ambient_flicker') or 0):.1%} between frames over the whole plate)")
    br = metrics.get("blur_rel")
    if _finite(br) and br < lim("blur_rel_min", 0.0):
        fail("blur", f"blur_rel {br:.2f} < {lim('blur_rel_min', 0.0)}")

    if reasons:
        verdict = "RETAKE"
    elif r > r_pass:
        verdict = "PASS_FLAGGED"
        fail("ambient_high", f"r_ambient {r:.2f} > {r_pass}")
    else:
        verdict = "PASS"
    # gate.messages_by_illumination overrides texts per illumination, in gate.messages order (the hood
    # fixes the distance, so its 'too_dark' is a mirror-like finish or a ring fault, not 'move to 200 mm')
    msgs = dict(g.get("messages") or {})
    msgs.update({k: v for k, v in ((g.get("messages_by_illumination") or {}).get(illum) or {}).items() if k in msgs})
    message = next((str(v) for k, v in msgs.items() if k in reasons), "")
    if not message and reasons:
        message = "Capture refused: retake." if verdict == "RETAKE" else ""
    return GateResult(verdict, reasons, message, details)


# ---- raw helpers for the Pi --------------------------------------------------------------------

_BAYER_LAYOUT = {"RGGB": ((0, 0), (1, 1)), "BGGR": ((1, 1), (0, 0)), "GRBG": ((0, 1), (1, 0)), "GBRG": ((1, 0), (0, 1))}
_BAYER_CODES = ("COLOR_BayerRG2RGB", "COLOR_BayerBG2RGB", "COLOR_BayerGR2RGB", "COLOR_BayerGB2RGB")


@lru_cache(maxsize=4)
def bayer_code(pattern: str = "RGGB") -> int:
    """The cv2 demosaic code that turns a `pattern` mosaic into RGB, found by a self-test.

    OpenCV names Bayer codes after a different 2x2 phase than sensor vendors, and the naming
    may differ between versions, so pick the code that puts red where the red samples are.
    """
    if pattern not in _BAYER_LAYOUT:
        raise ValueError(f"unknown Bayer pattern {pattern!r} ({', '.join(_BAYER_LAYOUT)})")
    (ry, rx), (by, bx) = _BAYER_LAYOUT[pattern]
    mos = np.full((8, 8), 500, np.uint16)
    mos[ry::2, rx::2] = 1000
    mos[by::2, bx::2] = 100
    for name in _BAYER_CODES:
        code = getattr(cv2, name, None)
        if code is None:
            continue
        px = cv2.cvtColor(mos, code)[3:5, 3:5].reshape(-1, 3).mean(axis=0)
        if abs(px[0] - 1000) < 1 and abs(px[1] - 500) < 1 and abs(px[2] - 100) < 1:
            return int(code)
    raise RuntimeError(f"no cv2 Bayer code demosaics {pattern} correctly")


def unpack_raw(raw_u16: np.ndarray, bit_depth: int = 10, left_shifted: bool | None = None) -> np.ndarray:
    """Unpacked raw (uint16) -> DN at bit_depth. Pi 5 unpacked streams are left-shifted to 16 bits;
    None detects that from the range."""
    raw = np.asarray(raw_u16)
    if raw.dtype != np.uint16:
        raw = raw.astype(np.uint16)
    if left_shifted is None:
        left_shifted = int(raw.max(initial=0)) > (1 << bit_depth) - 1
    return (raw >> np.uint16(16 - bit_depth)) if left_shifted else raw


def raw_to_rgb(raw_u16: np.ndarray, black_level: float, bit_depth: int = 10, pattern: str = "RGGB",
               left_shifted: bool | None = None) -> np.ndarray:
    """Bayer raw (uint16, e.g. SRGGB10 unpacked) -> float32 (H, W, 3) linear RGB in FS units.

    Demosaic (cv2 bilinear, on uint16, linear), then subtract the black level (DN at bit_depth)
    and divide by (2**bit_depth - 1 - black_level). Values below black stay negative (unbiased
    noise); no clipping, white balance or gamma: this is the input for process_burst.
    """
    dn = unpack_raw(raw_u16, bit_depth, left_shifted)
    rgb = cv2.cvtColor(np.ascontiguousarray(dn), bayer_code(pattern)).astype(np.float32)
    rgb -= np.float32(black_level)
    rgb *= np.float32(1.0 / ((1 << bit_depth) - 1 - float(black_level)))
    return rgb


# ---- burst files ---------------------------------------------------------------------------------

def save_burst(path: str | Path, burst: Burst, **extra: np.ndarray) -> None:
    """Write a burst as .npz: frames as float16 (`frame_<name>`), meta as JSON, extra arrays as is."""
    import json
    arrays = {f"frame_{k}": np.asarray(v, np.float16) for k, v in burst.frames.items()}
    arrays["meta_json"] = np.frombuffer(json.dumps(burst.meta, default=_json_default).encode(), np.uint8)
    arrays.update({k: np.asarray(v) for k, v in extra.items()})
    np.savez_compressed(path, **arrays)


def load_burst(path: str | Path) -> tuple[Burst, dict[str, np.ndarray]]:
    """Read save_burst output -> (Burst with float32 frames, extra arrays)."""
    import json
    with np.load(path) as z:
        frames = {k[6:]: z[k].astype(np.float32) for k in z.files if k.startswith("frame_")}
        meta = json.loads(z["meta_json"].tobytes().decode()) if "meta_json" in z.files else {}
        extra = {k: z[k] for k in z.files if not k.startswith("frame_") and k != "meta_json"}
    return Burst(frames, meta), extra


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")
