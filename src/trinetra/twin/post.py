"""Post-processing of twin renders into the detector dataset (DESIGN_STEEL.md E, DESIGN.md 4 and 5).

render.exr (linear light groups L, R, AMB) -> per variant: an ambient sweep (AMB x a
multiplier inside the condition's lux range) -> the capture arms in the ROI image:
  burst: burst.simulate_burst -> lightproc.process_burst (the new method, with its gate);
  naive: burst.simulate_naive (the old single both-pods frame through AE and an ISP);
-> rectify to the plate grid (0.08 mm/px, corners ordered by image position as on the Pi)
-> footprint masks -> visibility filter -> YOLO tiles + labels (train arm, gate not RETAKE),
full grids per eval arm with per-grid sidecars (test split), burst .npz files for the Pi
demo, and a scene sidecar JSON with the truth in mm and grid px. Real photos and bursts
use the same geometry, mask, lightproc and tile code (realdata.py).

Conventions:
- image and grid coordinates put pixel centres on integers (frames.py, OpenCV);
- mask boxes (`bbox_grid`, YOLO boxes) use pixel edges: pixel i spans [i, i + 1);
- object masks are stored cropped: `mask` (bool) with its top-left grid pixel `origin`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .. import tiles as T
from ..config import Grid
from . import frames as F

SCHEMA = "trinetra.twin.sidecar/2"
GRID_SCHEMA = "trinetra.twin.grid/1"            # per-grid sidecar next to each eval grid image
BURST_SCHEMA = "trinetra.twin.burst/1"          # meta_json of out/bursts/*.npz
GROUPS = ("L", "R", "UV", "AMB")
SIDES = ("L", "R")
FEATURES = ("hole", "open_hole", "rivet", "slot")   # part features: rasterised (ring exclusion), never labelled
ARMS = ("burst", "naive")                        # capture arms (DESIGN_STEEL E); fixed order = fixed RNG streams
RETAKE = "RETAKE"
BURST_FRAMES = ("B1", "L", "A", "R", "B2")
# DESIGN_STEEL C target illuminance at the plate centre (lux); configs/twin.yaml post.amb_sweep.lux overrides.
LUX_RANGES = {"C0_dark": (0.0, 0.0), "C1_office": (250.0, 700.0), "C2_warm_bulb": (300.0, 3000.0),
              "C3_window": (1500.0, 20000.0), "C4_side_lamp": (1000.0, 8000.0),
              "C5_reflected_lamp": (300.0, 1500.0), "C6_sun": (30000.0, 100000.0)}
REPO_ROOT = Path(__file__).resolve().parents[3]
CORNER_NAMES = ("TL", "TR", "BR", "BL")
LUMA_LIN = np.array([0.2126, 0.7152, 0.0722], np.float32)   # linear sRGB primaries (Rec. 709)
LUMA_U8 = np.array([0.299, 0.587, 0.114], np.float32)       # Rec. 601, like cv2 RGB2GRAY on frames
CIRCLE_PTS = 64
DEBUG_SCENES = 50
SHIFT = 4                                                     # fillPoly sub-pixel bits (1/16 px)
CONVENTIONS = {
    "grid_px": "pixel centres at integer coordinates (frames.py); polys_grid, centre_px",
    "bbox_grid": "[x0, y0, x1, y1] pixel edges, x1/y1 exclusive (as YOLO boxes)",
    "plate_mm": "u right, v down from the datum corner",
}


# ---- EXR ---------------------------------------------------------------------------

def _group_regex(groups: Sequence[str]) -> re.Pattern:
    alt = "|".join(re.escape(g) for g in groups)
    return re.compile(rf"(^|\.)Combined_({alt})(\.([RGB]))?$")


def read_light_groups(exr_path: str | Path, groups: Sequence[str] = GROUPS) -> dict[str, np.ndarray]:
    """Light-group passes of a Blender multilayer EXR as float32 (H, W, 3); missing -> zeros."""
    import OpenEXR

    rx = _group_regex(groups)
    found: dict[str, np.ndarray] = {}
    split: dict[str, dict[int, np.ndarray]] = {}
    for part in OpenEXR.File(str(exr_path)).parts:
        for name, ch in part.channels.items():
            m = rx.search(name)
            if not m:
                continue
            g, c, px = m.group(2), m.group(4), np.asarray(ch.pixels)
            if c:
                split.setdefault(g, {})["RGB".index(c)] = px
            elif px.ndim == 3:
                found[g] = px[..., :3]
            else:
                found[g] = np.repeat(px[..., None], 3, axis=2)
    for g, chans in split.items():
        if g not in found and len(chans) == 3:
            found[g] = np.stack([chans[i] for i in range(3)], axis=2)
    if not found:
        raise ValueError(f"{exr_path}: no Combined_<group> passes for groups {list(groups)}")
    shape = next(iter(found.values())).shape[:2]
    out = {}
    for g in groups:
        a = found.get(g)
        out[g] = (np.zeros((*shape, 3), np.float32) if a is None
                  else np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    return out


# ---- compose and camera model ------------------------------------------------------------

def _uniform(rng: np.random.Generator, r) -> float:
    if isinstance(r, (int, float)):
        return float(r)
    lo, hi = sorted((float(r[0]), float(r[1])))
    return float(rng.uniform(lo, hi))


def draw_gains(rng: np.random.Generator, post_cfg: Mapping[str, Any]) -> dict[str, float]:
    return {g: _uniform(rng, r) for g, r in (post_cfg.get("gains") or {}).items()}


def _weighted(groups: Mapping[str, np.ndarray], gains: Mapping[str, float], names: Iterable[str]) -> np.ndarray:
    shape = next(iter(groups.values())).shape
    img = np.zeros(shape, np.float32)
    for g in names:
        if g in groups and gains.get(g, 0.0):
            img += np.float32(gains[g]) * groups[g]
    return img


def side_frames(groups: Mapping[str, np.ndarray], gains: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """The L-only and R-only captures: each pod plus the ambient (and UV) light."""
    amb = _weighted(groups, gains, [g for g in groups if g not in SIDES])
    return amb + _weighted(groups, gains, ["L"]), amb + _weighted(groups, gains, ["R"])


def luma_lin(img: np.ndarray) -> np.ndarray:
    return img @ LUMA_LIN


def compose(groups: Mapping[str, np.ndarray], gains: Mapping[str, float], fusion: str = "both_rgb") -> np.ndarray:
    """Linear image from light groups (DESIGN 5.2): both_rgb = sum of gain x group;
    lr3 = [luma(L + amb), luma(R + amb), relief]."""
    if fusion == "both_rgb":
        return _weighted(groups, gains, list(groups))
    if fusion == "lr3":
        fl, fr = side_frames(groups, gains)
        a, b = luma_lin(fl), luma_lin(fr)
        relief = 0.5 + 0.5 * (a - b) / (a + b + 1e-4)
        return np.dstack([a, b, relief]).astype(np.float32)
    raise ValueError(f"unknown fusion {fusion!r} (both_rgb | lr3)")


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055).astype(np.float32)


def srgb_eotf(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, np.float32)
    return np.where(y <= 0.04045, y / 12.92, np.power((np.maximum(y, 0.04045) + 0.055) / 1.055, 2.4)).astype(np.float32)


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(x, np.float32) * 255.0), 0, 255).astype(np.uint8)


def motion_kernel(length_px: float, angle_deg: float) -> np.ndarray | None:
    """Normalised line kernel (bilinear splats), or None below half a pixel."""
    if length_px < 0.5:
        return None
    half = length_px / 2
    size = 2 * math.ceil(half) + 1
    c = size // 2
    s = np.linspace(-half, half, max(8, math.ceil(length_px * 8)))
    a = math.radians(angle_deg)
    xs, ys = c + s * math.cos(a), c + s * math.sin(a)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    k = np.zeros((size + 1, size + 1), np.float64)
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(k, (y0 + dy, x0 + dx), wgt)
    k = k[:size, :size]
    return (k / k.sum()).astype(np.float32)


def draw_camera(rng: np.random.Generator, post_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Random camera/ISP settings for one variant (JSON-serialisable)."""
    noise = post_cfg.get("noise") or {}
    j = float(post_cfg.get("wb_jitter", 0.0))
    return {
        "exposure_p50": _uniform(rng, post_cfg.get("exposure_p50", 0.18)),
        "wb": [_uniform(rng, [1 - j, 1 + j]), 1.0, _uniform(rng, [1 - j, 1 + j])],
        "full_well_e": float(noise.get("full_well_e", 6000.0)),
        "gain": _uniform(rng, noise.get("gain", 1.0)),
        "read_e": _uniform(rng, noise.get("read_e", 0.0)),
        "unsharp": _uniform(rng, post_cfg.get("unsharp", 0.0)),
        "unsharp_sigma_px": float(post_cfg.get("unsharp_sigma_px", 1.0)),
        "motion_px": _uniform(rng, post_cfg.get("motion_blur_px", 0.0)),
        "motion_deg": _uniform(rng, [0.0, 180.0]),
        "noise_seed": int(rng.integers(0, 2**31 - 1)),
    }


def exposure_for(img_lin: np.ndarray, target: float, plate_mask: np.ndarray | None = None) -> float:
    """Scale that puts the median luma of the plate region at target."""
    y = luma_lin(img_lin)
    vals = y[plate_mask] if plate_mask is not None and plate_mask.any() else y.ravel()
    med = float(np.median(vals))
    return float(target) / med if med > 1e-9 else 1.0


def apply_camera(img_lin: np.ndarray, cam: Mapping[str, Any], plate_mask: np.ndarray | None = None,
                 exposure_scale: float | None = None, noise_stream: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Linear RGB -> uint8 sRGB in the DESIGN 5.3 order; deterministic given cam.

    Noise is heteroscedastic Gaussian in electrons (the Poisson-Gaussian model): the
    standard-normal fields come from cam['noise_seed'], so a clean pass rendered with the
    same cam gets the same noise and |I - I_clean| measures only the defect.
    """
    img = np.clip(np.asarray(img_lin, np.float32), 0.0, None)
    if exposure_scale is None:
        exposure_scale = exposure_for(img, cam["exposure_p50"], plate_mask)
    img = img * np.float32(exposure_scale)                                   # 1. exposure
    img *= np.asarray(cam["wb"], np.float32)                                  # 2. white balance
    e_per_unit = np.float32(cam["full_well_e"] / max(cam["gain"], 1e-6))      # 3. electrons
    zr = np.random.default_rng([int(cam["noise_seed"]), int(noise_stream)])
    e = img * e_per_unit
    e += np.sqrt(e) * zr.standard_normal(e.shape, dtype=np.float32)
    e += np.float32(cam["read_e"]) * zr.standard_normal(e.shape, dtype=np.float32)
    img = np.clip(e / e_per_unit, 0.0, 1.0)                                    # 4. clip
    img = srgb_oetf(img)                                                      # 5. OETF
    if cam.get("unsharp", 0.0) > 0:                                           # 6. unsharp mask
        blur = cv2.GaussianBlur(img, (0, 0), float(cam.get("unsharp_sigma_px", 1.0)))
        img = np.clip(img + np.float32(cam["unsharp"]) * (img - blur), 0.0, 1.0)
    kern = motion_kernel(float(cam.get("motion_px", 0.0)), float(cam.get("motion_deg", 0.0)))
    if kern is not None:                                                      # 7. motion blur
        img = cv2.filter2D(img, -1, kern, borderType=cv2.BORDER_REFLECT)
    return to_u8(img), {"exposure_scale": float(exposure_scale)}              # 8. quantise


def camera_model(img_lin: np.ndarray, rng: np.random.Generator, post_cfg: Mapping[str, Any],
                 plate_mask: np.ndarray | None = None) -> np.ndarray:
    """Linear RGB ROI image -> uint8 RGB with settings drawn from post_cfg."""
    return apply_camera(img_lin, draw_camera(rng, post_cfg), plate_mask)[0]


def fuse_lr3_u8(l_u8: np.ndarray, r_u8: np.ndarray) -> np.ndarray:
    """8-bit L and R frames -> [luma L, luma R, relief] (relief from linearised lumas)."""
    a = l_u8.astype(np.float32) @ LUMA_U8 / 255.0
    b = r_u8.astype(np.float32) @ LUMA_U8 / 255.0
    la, lb = srgb_eotf(a), srgb_eotf(b)
    relief = 0.5 + 0.5 * (la - lb) / (la + lb + 1e-4)
    return to_u8(np.dstack([a, b, relief]))


def render_variant(groups: Mapping[str, np.ndarray], gains: Mapping[str, float], cam: Mapping[str, Any],
                   fusion: str = "both_rgb", plate_mask: np.ndarray | None = None,
                   exposure_scale: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Light groups -> the uint8 ROI image the detector input is made from.

    lr3 is fused after the camera model: the Pi captures the L and R frames separately
    (each with its own noise) at one exposure, so relief is never pushed through the ISP.
    """
    if fusion == "both_rgb":
        return apply_camera(compose(groups, gains, fusion), cam, plate_mask, exposure_scale)
    if fusion == "lr3":
        fl, fr = side_frames(groups, gains)
        if exposure_scale is None:
            exposure_scale = exposure_for(0.5 * (fl + fr), cam["exposure_p50"], plate_mask)
        l8, _ = apply_camera(fl, cam, plate_mask, exposure_scale, noise_stream=0)
        r8, _ = apply_camera(fr, cam, plate_mask, exposure_scale, noise_stream=1)
        return fuse_lr3_u8(l8, r8), {"exposure_scale": float(exposure_scale)}
    raise ValueError(f"unknown fusion {fusion!r} (both_rgb | lr3)")


# ---- geometry ------------------------------------------------------------------------

def plate_corners(w_mm: float, h_mm: float) -> np.ndarray:
    """Plate corners in plate order: datum (0, 0), (W, 0), (W, H), (0, H)."""
    return np.array([[0.0, 0.0], [w_mm, 0.0], [w_mm, h_mm], [0.0, h_mm]])


def _translate(x0: float, y0: float) -> np.ndarray:
    return np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])


def grid_geometry(src, grid_cfg: Mapping[str, Any], plate_size_mm: Sequence[float] | None = None,
                  roi: Sequence[float] | None = None) -> dict[str, Any]:
    """Plate -> image -> grid mappings for a plan (dict) or a full-sensor H_img<-plate.

    The 4 imaged plate corners are ordered TL, TR, BR, BL by image position
    (frames.order_corners, as on the Pi); H_grid<-img maps them to the grid plate corners,
    and the grid is (W, H) or (H, W), whichever plate edge is imaged as TL -> TR.
    """
    cam = None
    if isinstance(src, Mapping):
        c = src["camera"]
        K = np.asarray(F.nominal_K() if c.get("K") is None else c["K"], float)
        R, t = np.asarray(c["R"], float), np.asarray(c["t"], float)
        cam = {"K": K, "R": R, "t": t}
        H_img_plate = F.homography_plate_to_image(K, R, t)
        if plate_size_mm is None:
            plate_size_mm = src["plate"]["size_mm"]
        if roi is None:
            full = c.get("full_res") or [F.IMX500["width"], F.IMX500["height"]]
            roi = c.get("roi") if c.get("roi") is not None else [0, 0, full[0], full[1]]
    else:
        H_img_plate = np.asarray(src, float)
        if plate_size_mm is None:
            raise ValueError("grid_geometry(H_img_plate, ...) needs plate_size_mm")
    H_img_plate = H_img_plate / H_img_plate[2, 2]
    W, H = float(plate_size_mm[0]), float(plate_size_mm[1])
    corners = F.apply_h(H_img_plate, plate_corners(W, H))
    ordered = F.order_corners(corners)
    idx = [int(np.argmin(np.linalg.norm(corners - p, axis=1))) for p in ordered]
    if sorted(idx) != [0, 1, 2, 3] or any((idx[i] - idx[0]) % 4 != i for i in range(4)):
        raise ValueError(f"plate corners are not imaged in plate order (mirrored or degenerate): {idx}")
    gw, gh = (W, H) if idx[0] % 2 == 0 else (H, W)
    grid = Grid(gw, gh, border_mm=float(grid_cfg.get("border_mm", 5.0)), mm_per_px=float(grid_cfg.get("mm_per_px", 0.08)))
    H_grid_img = cv2.getPerspectiveTransform(ordered.astype(np.float32), grid.plate_corners_px()).astype(float)
    x0, y0 = (float(roi[0]), float(roi[1])) if roi is not None else (0.0, 0.0)
    H_grid_plate = H_grid_img @ H_img_plate
    return {
        "H_img_plate": H_img_plate,
        "H_roi_plate": _translate(-x0, -y0) @ H_img_plate,
        "H_grid_img": H_grid_img,
        "H_grid_roi": H_grid_img @ _translate(x0, y0),
        "H_grid_plate": H_grid_plate / H_grid_plate[2, 2],
        "roi": [int(v) for v in roi] if roi is not None else None,
        "corners_img": ordered,
        "corners_roi": ordered - [x0, y0],
        "corner_plate_idx": idx,
        "datum_corner": CORNER_NAMES[idx.index(0)],
        "plate_size_mm": (W, H),
        "grid_plate_mm": (gw, gh),
        "grid": grid,
        "size": grid.size,
        "shape": grid.shape,
        "mm_per_px": grid.mm_per_px,
        "camera": cam,
    }


def _image_to_roi(shape: Sequence[int], roi: Sequence[int] | None) -> np.ndarray:
    """Image px -> ROI px, for EXRs rendered at a different resolution (render --res-scale).

    Like Blender (sensor_fit AUTO, render.scale_K): pixel edges scale about the image centre
    by s = max(ws, hs) / max(w, h), with pixel centres on integers in both images."""
    if roi is None:
        return np.eye(3)
    hs, ws = shape[:2]
    w, h = roi[2], roi[3]
    if (ws, hs) == (w, h):
        return np.eye(3)
    s = max(ws, hs) / max(w, h)
    return np.array([[1 / s, 0.0, (0.5 - ws / 2) / s + w / 2 - 0.5],
                     [0.0, 1 / s, (0.5 - hs / 2) / s + h / 2 - 0.5], [0.0, 0.0, 1.0]])


def rectify(img_roi: np.ndarray, geom: Mapping[str, Any]) -> np.ndarray:
    """ROI image (or full photo when roi = [0, 0, w, h]) -> the rectified plate grid."""
    H = geom["H_grid_roi"] @ _image_to_roi(img_roi.shape, geom["roi"])
    return cv2.warpPerspective(img_roi, H, tuple(geom["size"]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def plate_mask_image(geom: Mapping[str, Any], shape: Sequence[int]) -> np.ndarray:
    """Bool mask of the imaged plate in an ROI-image of the given shape."""
    pts = F.apply_h(np.linalg.inv(_image_to_roi(shape, geom["roi"])), geom["corners_roi"])
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, SHIFT)
    return m.astype(bool)


def plate_to_grid(geom: Mapping[str, Any], uv_mm, w_mm=None) -> np.ndarray:
    """Plate mm (N, 2) -> grid px. Points raised by w_mm (e.g. on a lap strip) are projected
    through the camera when the geometry has one, so labels sit where they are imaged."""
    uv = np.atleast_2d(np.asarray(uv_mm, float))
    if w_mm is None or geom.get("camera") is None or not np.any(np.asarray(w_mm, float)):
        return F.apply_h(geom["H_grid_plate"], uv)
    c = geom["camera"]
    img = F.project(c["K"], c["R"], c["t"], F.plate_to_world(uv, w_mm))
    return F.apply_h(geom["H_grid_img"], img)


# ---- footprints and masks ---------------------------------------------------------------

def ellipse_poly(u: float, v: float, d_major: float, d_minor: float, angle_deg: float = 0.0,
                 n: int = CIRCLE_PTS) -> np.ndarray:
    """Ellipse outline in plate mm; angle from +u towards +v."""
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a = math.radians(angle_deg)
    x, y = 0.5 * d_major * np.cos(s), 0.5 * d_minor * np.sin(s)
    return np.c_[u + x * math.cos(a) - y * math.sin(a), v + x * math.sin(a) + y * math.cos(a)]


def circle_poly(u: float, v: float, d: float, n: int = CIRCLE_PTS) -> np.ndarray:
    return ellipse_poly(u, v, d, d, 0.0, n)


def rect_poly(u: float, v: float, length: float, width: float, angle_deg: float = 0.0) -> np.ndarray:
    """length x width rectangle centred on (u, v), its length along angle_deg (from +u to +v)."""
    a = math.radians(angle_deg)
    d, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    hl, hw = 0.5 * length, 0.5 * width
    c = np.array([u, v])
    return np.array([c - hl * d - hw * n, c + hl * d - hw * n, c + hl * d + hw * n, c - hl * d + hw * n])


def _sheet_thickness(sheets: Mapping[str, Any], sheet_id) -> float:
    s = sheets.get(sheet_id) if sheet_id else None
    return float(s.get("thickness_mm", 0.0)) if s else 0.0


def surface_height(plan: Mapping[str, Any], uv_mm) -> np.ndarray:
    """Front-surface height w (mm) at plate points: the top of a front sheet, else 0.
    Sheet rect_mm is [u0, v0, width, height] like spec.yaml zones."""
    uv = np.atleast_2d(np.asarray(uv_mm, float))
    w = np.zeros(len(uv))
    for s in plan.get("sheets") or []:
        x, y, sw, sh = map(float, s["rect_mm"])
        inside = (uv[:, 0] >= x) & (uv[:, 0] <= x + sw) & (uv[:, 1] >= y) & (uv[:, 1] <= y + sh)
        w = np.where(inside, np.maximum(w, float(s.get("thickness_mm", 0.0))), w)
    return w


def _polys(footprint: Mapping[str, Any] | None) -> list[np.ndarray]:
    polys = (footprint or {}).get("polys_mm") or []
    return [np.asarray(p, float).reshape(-1, 2) for p in polys if len(p) >= 2]


def plan_items(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Part features, defects and penetrant indications of a plan as footprints in plate mm.

    Features (group "feature", never labelled: holes are part features, DESIGN_STEEL A) are
    still rasterised so visibility rings skip them. As geometry.fastener_outline: a filled
    hole with a rivet (v1 plans) is the rivet's head circle (head_d_mm); any other hole is
    an ellipse (major axis d_mm x ovality along angle_deg, minor axis d_mm)."""
    holes = {h["id"]: h for h in plan.get("holes") or []}
    sheets = {s["id"]: s for s in plan.get("sheets") or []}
    items: list[dict[str, Any]] = []
    riveted = set()
    for r in plan.get("rivets") or []:
        h = holes.get(r.get("hole"))
        if h is None or not h.get("filled") or h["id"] in riveted:
            continue
        riveted.add(h["id"])
        d, w = float(r["head_d_mm"]), _sheet_thickness(sheets, h.get("sheet"))
        items.append({"id": r["id"], "cls": "rivet", "kind": "rivet", "group": "feature", "label": False,
                      "polys_mm": [circle_poly(h["u"], h["v"], d)], "w_mm": [w],
                      "measure": {"diameter_mm": d}, "near": None,
                      "feature": {"hole": h["id"], "u_mm": float(h["u"]), "v_mm": float(h["v"]), "d_mm": d,
                                   "hole_d_mm": float(h["d_mm"]), "ovality": 1.0, "angle_deg": 0.0, "w_mm": w}})
    for h in holes.values():
        if h["id"] in riveted:
            continue
        d, ov, ang = float(h["d_mm"]), float(h.get("ovality", 1.0)), float(h.get("angle_deg", 0.0))
        w = _sheet_thickness(sheets, h.get("sheet"))
        items.append({"id": h["id"], "cls": "hole", "kind": h.get("kind") or "hole", "group": "feature", "label": False,
                      "polys_mm": [ellipse_poly(h["u"], h["v"], d * ov, d, ang)], "w_mm": [w],
                      "measure": {"diameter_mm": d, "ovality": ov}, "near": None,
                      "feature": {"hole": h["id"], "u_mm": float(h["u"]), "v_mm": float(h["v"]), "d_mm": d,
                                   "hole_d_mm": d, "ovality": ov, "angle_deg": ang, "w_mm": w}})
    for dft in plan.get("defects") or []:
        polys = _polys(dft.get("footprint"))
        items.append({"id": dft["id"], "cls": dft["cls"], "kind": dft.get("kind"), "group": "defect",
                      "label": bool(dft.get("label", True)) and dft["cls"] in T.CLASSES and bool(polys),
                      "polys_mm": polys, "w_mm": [surface_height(plan, p) for p in polys],
                      "measure": (dft.get("footprint") or {}).get("measure") or {}, "near": dft.get("near")})
    for ind in (plan.get("penetrant") or {}).get("indications") or []:
        polys = _polys(ind.get("footprint"))
        items.append({"id": ind["id"], "cls": ind["cls"], "kind": ind["cls"], "group": "indication",
                      "label": False, "polys_mm": polys, "w_mm": [surface_height(plan, p) for p in polys],
                      "measure": (ind.get("footprint") or {}).get("measure") or {}, "near": None,
                      "extra": {"from_defect": ind.get("from_defect")}})
    return items


def _fill(polys_px: list[np.ndarray], shape: Sequence[int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Rasterise polygons (even-odd, plus their outlines so thin ones stay connected)
    into a crop of the grid; returns (bool mask, (x0, y0))."""
    gh, gw = shape[:2]
    polys_px = [p for p in polys_px if len(p) >= 2 and np.all(np.isfinite(p))]
    if not polys_px:
        return np.zeros((0, 0), bool), (0, 0)
    allp = np.concatenate(polys_px)
    x0, y0 = max(0, int(math.floor(allp[:, 0].min())) - 2), max(0, int(math.floor(allp[:, 1].min())) - 2)
    x1, y1 = min(gw, int(math.ceil(allp[:, 0].max())) + 3), min(gh, int(math.ceil(allp[:, 1].max())) + 3)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), bool), (0, 0)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = [np.round((p - [x0, y0]) * (1 << SHIFT)).astype(np.int32).reshape(-1, 1, 2) for p in polys_px]
    cv2.fillPoly(m, pts, 1, cv2.LINE_8, SHIFT)
    cv2.polylines(m, pts, True, 1, 1, cv2.LINE_8, SHIFT)
    return m.astype(bool), (x0, y0)


def _width_px(measure: Mapping[str, Any] | None, mask: np.ndarray, mm_per_px: float) -> float:
    """Feature width in grid px: width_mm, else diameter_mm, else twice the mask's inradius."""
    measure = measure or {}
    for key in ("width_mm", "diameter_mm"):
        if measure.get(key):
            return float(measure[key]) / mm_per_px
    if mask.size and mask.any():
        dt = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)
        return float(2.0 * dt.max())
    return 1.0


def rasterise_items(items: Sequence[Mapping[str, Any]], geom: Mapping[str, Any],
                    grid_shape: Sequence[int]) -> list[dict[str, Any]]:
    """Footprints (plate mm) -> objects with grid polygons and cropped masks."""
    out = []
    for it in items:
        ws = it.get("w_mm") or [None] * len(it["polys_mm"])
        polys_px = [plate_to_grid(geom, p, w) for p, w in zip(it["polys_mm"], ws)]
        mask, origin = _fill(polys_px, grid_shape)
        o = {k: v for k, v in it.items() if k != "w_mm"}
        o.update(poly_grid=polys_px, mask=mask, origin=origin, n_px=int(mask.sum()),
                 width_px=_width_px(it.get("measure"), mask, geom["mm_per_px"]))
        if it["polys_mm"]:
            allp = np.concatenate(it["polys_mm"])
            o.update(centre_mm=allp.mean(axis=0), bbox_mm=np.r_[allp.min(axis=0), allp.max(axis=0)])
        out.append(o)
    return out


def object_masks(plan: Mapping[str, Any], geom: Mapping[str, Any], grid_shape: Sequence[int]) -> list[dict[str, Any]]:
    """All part features, defects (labelled or not) and penetrant indications of a plan on the grid.

    Each: {id, cls, kind, group (feature|defect|indication), label, mask, origin, poly_grid,
    measure, near, n_px, width_px}."""
    return rasterise_items(plan_items(plan), geom, grid_shape)


def mask_bbox(obj: Mapping[str, Any]) -> list[int] | None:
    """[x0, y0, x1, y1] of the mask pixels in grid px (pixel edges), or None if empty."""
    m = obj["mask"]
    if not m.size or not m.any():
        return None
    ys, xs = np.nonzero(m)
    ox, oy = obj["origin"]
    return [int(xs.min() + ox), int(ys.min() + oy), int(xs.max() + ox + 1), int(ys.max() + oy + 1)]


# ---- visibility ------------------------------------------------------------------------

def _as_u8(img: np.ndarray) -> np.ndarray:
    """uint8 as is; float is linear light in [0, 1] and is sRGB-encoded first."""
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    return to_u8(srgb_oetf(np.clip(img, 0.0, 1.0)))


def _absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = cv2.absdiff(a, b).astype(np.float32)
    return d.mean(axis=2) if d.ndim == 3 else d


def _shrink(img: np.ndarray, masks: Sequence[np.ndarray], f: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Area-average a crop by f; a shrunk mask keeps pixels at least half covered."""
    size = (max(1, img.shape[1] // f), max(1, img.shape[0] // f))
    out = []
    for m in masks:
        a = cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        out.append((a >= 0.5) if (a >= 0.5).any() else (a > 0))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), out


OFF_PLATE = 255


def ring_exclude(objects: Sequence[Mapping[str, Any]], geom: Mapping[str, Any], edge_px: int = 3) -> np.ndarray:
    """Occupancy map for visibility() (uint8, grid shape): OFF_PLATE off the plate and within
    edge_px of its edge, else how many object footprints (dilated 2 px) cover the pixel.
    Background rings use only free (0) pixels; an object's inside drops the pixels another
    object or the plate edge claims (a scratch over a rivet head, a nick on the edge)."""
    grid: Grid = geom["grid"]
    occ = np.zeros(geom["shape"], np.uint16)
    k = np.ones((5, 5), np.uint8)
    for o in objects:
        if o.get("group") == "indication" or not o["n_px"]:
            continue
        m = cv2.dilate(o["mask"].astype(np.uint8), k)
        x0, y0 = o["origin"]
        occ[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]] += m
    occ = np.minimum(occ, OFF_PLATE - 1).astype(np.uint8)
    occ[~grid.plate_mask(edge_px * grid.mm_per_px)] = OFF_PLATE
    return occ


def _ring(msk: np.ndarray, ex: np.ndarray) -> np.ndarray:
    dist = cv2.distanceTransform((~msk).astype(np.uint8), cv2.DIST_L2, 5)
    return (dist >= 3) & (dist <= 8) & ~ex


def _plane_residual(img: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """|I - background| (mean over channels), the background a least-squares plane per
    channel through the ring pixels (follows the L/R illumination gradient)."""
    h, w = img.shape[:2]
    vals = img.reshape(h, w, -1).astype(np.float64)
    ys, xs = np.nonzero(ring)
    A = np.c_[np.ones(len(xs)), xs, ys]
    coef = np.linalg.lstsq(A, vals[ring], rcond=None)[0]                     # (3, C)
    Y, X = np.mgrid[:h, :w]
    bg = np.stack([np.ones((h, w)), X, Y], axis=-1) @ coef
    return np.abs(vals - bg).mean(axis=2).astype(np.float32)


def visibility(grid: np.ndarray, grid_clean: np.ndarray | None, obj: Mapping[str, Any],
               cfg: Mapping[str, Any], exclude: np.ndarray | None = None) -> tuple[bool, float | None]:
    """(visible, contrast) of one object in 8-bit units (DESIGN 5.5).

    With a clean grid: mean |I - I_clean| in the 1-px-dilated mask >= diff_threshold.
    Without: p90 |I - background| in the mask minus p90 in a 3-8 px ring around it
    >= local_threshold. The background is medianBlur(I, k), k odd and >= 3 x width px,
    while that fits under the 31-px cap (width <= 10 px). Wider objects (a 20 mm smooth dent
    survives any 31-px median) are tested on the crop shrunk by f = ceil(3 x width / 31),
    with a plane fitted to the ring as the background. Classes in visibility.always_visible pass.
    exclude: ring_exclude()'s occupancy map (or a bool never-background mask); the ring
    keeps free pixels only and the inside drops other objects and the plate edge band.
    """
    v = cfg.get("visibility", cfg)
    mask = obj["mask"]
    if not mask.size or not mask.any():
        return False, None
    always = obj["cls"] in (v.get("always_visible") or ())
    width = max(float(obj.get("width_px") or 1.0), float(v.get("min_width_px", 1.0)))
    n = math.ceil(3 * width - 1e-9)
    k = max(3, n + 1 - n % 2)
    f = 1 if grid_clean is not None or k <= 31 else math.ceil(n / 31)
    ox, oy = obj["origin"]
    mh, mw = mask.shape
    gh, gw = grid.shape[:2]
    m = (min(k, 31) // 2 + 10) * f
    x0, y0, x1, y1 = max(0, ox - m), max(0, oy - m), min(gw, ox + mw + m), min(gh, oy + mh + m)
    img = _as_u8(grid[y0:y1, x0:x1])
    msk = np.zeros((y1 - y0, x1 - x0), bool)
    msk[oy - y0 : oy - y0 + mh, ox - x0 : ox - x0 + mw] = mask
    if grid_clean is not None:
        d = _absdiff(img, _as_u8(grid_clean[y0:y1, x0:x1]))
        dil = cv2.dilate(msk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        contrast = float(d[dil].mean())
        return bool(always or contrast >= float(v.get("diff_threshold", 3.0))), contrast
    if exclude is None:
        ex = other = np.zeros_like(msk)
    else:
        occ = exclude[y0:y1, x0:x1]
        occ = occ.astype(np.uint8) * OFF_PLATE if occ.dtype == bool else occ
        own = cv2.dilate(msk.astype(np.uint8), np.ones((5, 5), np.uint8))
        ex, other = occ > 0, occ.astype(np.int16) - own > 0
    inside = msk & ~other if (msk & ~other).any() else msk
    if f == 1:
        d = _absdiff(img, cv2.medianBlur(img, k))
        ring = _ring(msk, ex)
    else:
        img, (msk, inside, ex) = _shrink(img, (msk, inside, ex), f)
        ring = _ring(msk, ex)
        if ring.sum() < 6:                                  # hemmed in: any free pixel nearby
            ring = ~msk & ~ex
        if ring.sum() < 6:
            return always, None
        d = _plane_residual(img, ring)
    outside = float(np.percentile(d[ring], 90)) if ring.any() else 0.0
    contrast = float(np.percentile(d[inside], 90)) - outside
    return bool(always or contrast >= float(v.get("local_threshold", 4.0))), contrast


# ---- tiles and labels -------------------------------------------------------------------

def _pad_box(box, min_px: float):
    x0, y0, x1, y1 = map(float, box)
    if x1 - x0 < min_px:
        c = 0.5 * (x0 + x1)
        x0, x1 = c - 0.5 * min_px, c + 0.5 * min_px
    if y1 - y0 < min_px:
        c = 0.5 * (y0 + y1)
        y0, y1 = c - 0.5 * min_px, c + 0.5 * min_px
    return (x0, y0, x1, y1)


def label_boxes(objects: Sequence[Mapping[str, Any]], tiles: Sequence[T.Tile],
                label_cfg: Mapping[str, Any]) -> list[tuple[int, T.Tile, tuple]]:
    """(object index, tile, tile-local box) for every labelled, visible object piece kept
    in a tile: the bbox of its in-tile mask pixels, padded to min_box_px and clipped to the
    tile, kept if it holds >= keep_frac of the pixels or its diagonal is >= keep_min_px."""
    min_box = float(label_cfg.get("min_box_px", 8))
    keep_frac = float(label_cfg.get("keep_frac", 0.3))
    keep_min = float(label_cfg.get("keep_min_px", 24))
    out = []
    for i, o in enumerate(objects):
        if not o.get("label", True) or o.get("visible", True) is False or o["cls"] not in T.CLASSES or not o["n_px"]:
            continue
        m, (ox, oy) = o["mask"], o["origin"]
        for t in tiles:
            cx0, cy0 = max(0, t.x0 - ox), max(0, t.y0 - oy)
            cx1, cy1 = min(m.shape[1], t.x1 - ox), min(m.shape[0], t.y1 - oy)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            ys, xs = np.nonzero(m[cy0:cy1, cx0:cx1])
            if not len(xs):
                continue
            box = (xs.min() + cx0 + ox, ys.min() + cy0 + oy, xs.max() + cx0 + ox + 1, ys.max() + cy0 + oy + 1)
            if len(xs) < keep_frac * o["n_px"] and math.hypot(box[2] - box[0], box[3] - box[1]) < keep_min:
                continue
            local = T.clip_box(_pad_box(box, min_box), t)
            if local is not None:
                out.append((i, t, local))
    return out


def tile_labels(objects: Sequence[Mapping[str, Any]], tiles: Sequence[T.Tile],
                label_cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    """YOLO label lines per tile name (every tile has an entry, maybe empty)."""
    return lines_from_boxes(objects, tiles, label_boxes(objects, tiles, label_cfg))


def lines_from_boxes(objects, tiles, boxes) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {t.name: [] for t in tiles}
    for i, t, box in boxes:
        labels[t.name].append(T.yolo_line(T.CLASSES.index(objects[i]["cls"]), box, t.size))
    return labels


def write_rgb(path: Path, img: np.ndarray, quality: int | None = None) -> None:
    """Write an RGB (or grey) uint8 image with OpenCV (which expects BGR)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if quality is not None else []
    if not cv2.imwrite(str(path), bgr, params):
        raise OSError(f"could not write {path}")


def write_tiles(grid_u8: np.ndarray, tiles: Sequence[T.Tile], labels: Mapping[str, list[str]], out_dir: Path,
                split: str, stem: str, quality: int = 95, keep_empty: Sequence[bool] | None = None) -> list[str]:
    """Tiles + YOLO label files under out/yolo/{images,labels}/<split>/; returns the tile
    names written. Empty tiles are written only where keep_empty is true (None = all)."""
    img_dir, lab_dir = Path(out_dir) / "yolo" / "images" / split, Path(out_dir) / "yolo" / "labels" / split
    lab_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (t, crop) in enumerate(zip(tiles, T.cut(grid_u8, list(tiles)))):
        lines = labels.get(t.name, [])
        if not lines and keep_empty is not None and not keep_empty[i]:
            continue
        name = f"{stem}_{t.name}"
        write_rgb(img_dir / f"{name}.jpg", crop, quality)
        (lab_dir / f"{name}.txt").write_text("".join(line + "\n" for line in lines))
        written.append(t.name)
    return written


# ---- sidecar, debug, data.yaml ------------------------------------------------------------

def json_safe(x):
    """JSON-safe copy (numpy -> lists/scalars, non-finite -> None)."""
    if isinstance(x, Mapping):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def _r(x, nd: int = 3):
    return None if x is None else round(float(x), nd)


def _round(a, nd: int = 3):
    return None if a is None else np.round(np.asarray(a, float), nd)


def near_edge_mm(near: Mapping[str, Any] | None) -> float | None:
    """Closest edge-to-edge distance (mm, to a plate edge or hole) of a plan's `near` dict: its
    near_edge_mm when the sampler gives one, else the min over its *_edge_mm values."""
    near = near or {}
    if isinstance(near.get("near_edge_mm"), (int, float)):
        return round(float(near["near_edge_mm"]), 3)
    vals = [float(v) for k, v in near.items() if str(k).endswith("_edge_mm") and isinstance(v, (int, float))]
    return round(min(vals), 3) if vals else None


def record_fields(o: Mapping[str, Any], geom: Mapping[str, Any]) -> dict[str, Any]:
    """Ground truth of a defect in the trinetra.inspection/3.0 defect shape, so the rules
    engine can judge it (rules.judge_steel): type, box_px (grid px, pixel edges), location_mm
    (grid-aligned plate mm of the box centre, as the Pi measures it: app detection_record,
    Grid.px_to_mm; the footprint's vertex mean only when the mask is empty), size (the plan's
    measure without depth_mm, which the device never measures) and near_edge_mm.

    The box centre, not the vertex mean: for a rust streak or a curved crack they are up to
    8 mm apart, enough to change the rules' pit-in-rust pairing (1 mm) and clusters."""
    size = {k: v for k, v in (o.get("measure") or {}).items() if k != "depth_mm"}
    box = mask_bbox(o)
    loc = None
    if box is not None:
        c = ((box[0] + box[2]) / 2.0 - 0.5, (box[1] + box[3]) / 2.0 - 0.5)    # pixel edges -> centres
    elif o.get("centre_mm") is not None:
        c = plate_to_grid(geom, [o["centre_mm"]])[0]
    if box is not None or o.get("centre_mm") is not None:
        x, y = geom["grid"].px_to_mm(float(c[0]), float(c[1]))
        loc = {"x": round(x, 2), "y": round(y, 2)}
    return {"type": o["cls"], "box_px": box, "location_mm": loc, "size": size,
            "near_edge_mm": near_edge_mm(o.get("near"))}


def object_record(o: Mapping[str, Any], geom: Mapping[str, Any], tiles_of: Mapping[str, list[str]]) -> dict[str, Any]:
    """Sidecar entry for one object (part feature, defect or indication)."""
    rec: dict[str, Any] = {"id": o["id"], "cls": o["cls"]}
    if o["group"] == "feature":
        f = o["feature"]
        rec.update({k: f[k] for k in ("hole", "u_mm", "v_mm", "d_mm", "hole_d_mm", "ovality", "angle_deg")})
        rec["centre_px"] = np.round(plate_to_grid(geom, [[f["u_mm"], f["v_mm"]]], f.get("w_mm"))[0], 2)
        rec["d_px"] = _r(f["d_mm"] / geom["mm_per_px"], 2)
        rec["label"] = False
    else:
        rec.update(kind=o.get("kind"), label=bool(o.get("label")), measure=o.get("measure"), near=o.get("near"),
                   centre_mm=_round(o.get("centre_mm")), bbox_mm=_round(o.get("bbox_mm")),
                   polys_grid=[np.round(p, 2) for p in o["poly_grid"]])
        if o["group"] == "defect":
            rec.update(record_fields(o, geom))
    rec.update(bbox_grid=mask_bbox(o), n_px=o["n_px"], width_px=_r(o["width_px"], 2),
               visible=o.get("visible"), contrast=_r(o.get("contrast"), 2), tiles=tiles_of.get(o["id"], []))
    rec.update(o.get("extra") or {})
    return rec


def sidecar_base(geom: Mapping[str, Any], grid_cfg: Mapping[str, Any], tiles: Sequence[T.Tile]) -> dict[str, Any]:
    g: Grid = geom["grid"]
    return {
        "schema": SCHEMA,
        "conventions": CONVENTIONS,
        "plate_size_mm": list(geom["plate_size_mm"]),
        "grid": {"mm_per_px": g.mm_per_px, "border_mm": g.border_mm, "size_px": list(geom["size"]),
                 "plate_mm": list(geom["grid_plate_mm"]), "datum_corner": geom["datum_corner"],
                 "corner_plate_idx": geom["corner_plate_idx"], "tile": int(grid_cfg.get("tile", T.TILE)),
                 "tiles": {t.name: [t.x0, t.y0] for t in tiles}},
        "roi": geom["roi"],
        "corners_img": np.round(geom["corners_img"], 3),
        "H_img_plate": geom["H_img_plate"],
        "H_grid_img": geom["H_grid_img"],
        "H_grid_plate": geom["H_grid_plate"],
    }


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON atomically (a crash never leaves a half sidecar that marks a scene done)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(data), indent=1))
    os.replace(tmp, path)


_COLOURS = {"feature": (255, 160, 0), "visible": (0, 220, 0), "invisible": (0, 0, 255),
            "unlabelled": (0, 200, 255), "indication": (255, 0, 255), "box": (255, 255, 0)}


def write_debug(path: Path, grid_u8: np.ndarray, objects: Sequence[Mapping[str, Any]],
                boxes: Sequence[tuple], tiles: Sequence[T.Tile]) -> None:
    """Grid with footprints (green visible, red invisible, orange unlabelled, blue part
    features, magenta indications), YOLO boxes (cyan) and tile borders (grey)."""
    img = grid_u8 if grid_u8.ndim == 3 else np.dstack([grid_u8] * 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for t in tiles:
        cv2.rectangle(img, (t.x0, t.y0), (t.x1 - 1, t.y1 - 1), (110, 110, 110), 1)
    for o in objects:
        if o["group"] in ("feature", "indication"):
            col = _COLOURS[o["group"]]
        elif not o.get("label"):
            col = _COLOURS["unlabelled"]
        else:
            col = _COLOURS["visible" if o.get("visible") is not False else "invisible"]
        pts = [np.round(p * (1 << SHIFT)).astype(np.int32).reshape(-1, 1, 2) for p in o["poly_grid"] if len(p) >= 2]
        if pts:
            cv2.polylines(img, pts, True, col, 1, cv2.LINE_AA, SHIFT)
        b = mask_bbox(o)
        if b:
            cv2.putText(img, str(o["id"]), (b[0], max(10, b[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    for _, t, b in boxes:
        cv2.rectangle(img, (int(b[0] + t.x0), int(b[1] + t.y0)), (int(math.ceil(b[2] + t.x0)) - 1, int(math.ceil(b[3] + t.y0)) - 1),
                      _COLOURS["box"], 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _want_debug(out: Path, debug: bool | None) -> bool:
    if debug is not None:
        return bool(debug)
    scenes = {p.stem.rsplit("_v", 1)[0] for p in (out / "debug").glob("*.jpg")}
    return len(scenes) < DEBUG_SCENES


def write_data_yaml(out_dir: str | Path, classes: Sequence[str] = T.CLASSES) -> Path:
    """Ultralytics dataset file out/yolo/data.yaml (path, train, val, [test], names)."""
    if list(classes) != list(T.CLASSES):
        raise ValueError(f"classes {list(classes)} differ from trinetra.tiles.CLASSES {list(T.CLASSES)}")
    root = Path(out_dir).resolve() / "yolo"
    root.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"path": str(root), "train": "images/train", "val": "images/val"}
    for test in ("test", "twin_test"):
        if (root / "images" / test).is_dir():
            data["test"] = f"images/{test}"
            break
    data["names"] = {i: c for i, c in enumerate(classes)}
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# ---- capture arms (DESIGN_STEEL E) -------------------------------------------------------

def capture_modules():
    """(lightproc, burst): the shared capture processing and the twin burst simulator.
    Imported lazily (another build owns them); tests swap in stubs with monkeypatch."""
    from .. import lightproc
    from . import burst

    return lightproc, burst


def capture_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """capture.yaml settings: cfg['capture_config'] is the dict itself or its path (relative
    paths are tried from the cwd, then from the repo root)."""
    c = cfg.get("capture_config", "configs/capture.yaml")
    if isinstance(c, Mapping):
        return dict(c)
    path = Path(c)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    loader = getattr(capture_modules()[0], "load_capture_config", None)
    return dict(loader(str(path)) if loader is not None else yaml.safe_load(path.read_text()))


def scene_info(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Lighting condition, rendered lux, device mode and the part (material, thickness, size) of a plan."""
    amb = plan.get("ambient") or {}
    dev = (plan.get("extras") or {}).get("device") or {}
    p = plan.get("plate") or {}
    mat = p.get("material") or {}
    return {"condition": amb.get("condition"), "lux_rendered": amb.get("lux"), "cct_k": amb.get("cct_k"),
            "device_mode": dev.get("mode", "open"),
            "part": {"material": mat.get("kind") or p.get("finish"), "thickness_mm": p.get("thickness_mm"),
                     "size_mm": list(p.get("size_mm") or [])}}


def lux_range(condition: str | None, post_cfg: Mapping[str, Any]) -> tuple[float, float] | None:
    table = (post_cfg.get("amb_sweep") or {}).get("lux") or LUX_RANGES
    r = table.get(condition) if condition else None
    return None if r is None else (float(min(r)), float(max(r)))


def amb_multiplier(rng: np.random.Generator, ambient: Mapping[str, Any] | None, k: int,
                   post_cfg: Mapping[str, Any]) -> float:
    """Scale of the AMB pass for variant k: a lux drawn log-uniformly in the condition's range,
    over the rendered lux, so one render gives several ambient levels. C0 (range 0) gives 0;
    variant 0 keeps the rendered lux (amb_sweep.keep_first); plans with no condition or lux
    keep 1. One uniform is always drawn so the variant's other draws never shift."""
    sweep = post_cfg.get("amb_sweep") or {}
    u = float(rng.random())
    amb = ambient or {}
    r = lux_range(amb.get("condition"), post_cfg)
    if r is not None and r[1] <= 0:
        return 0.0
    if not sweep.get("enabled", True) or (k == 0 and sweep.get("keep_first", True)):
        return 1.0
    lux0 = amb.get("lux")
    if r is None or not lux0 or float(lux0) <= 0 or r[0] <= 0:
        return 1.0
    lo, hi = math.log(r[0]), math.log(r[1])
    return math.exp(lo + u * (hi - lo)) / float(lux0)


def scale_ambient(groups: Mapping[str, np.ndarray], mult: float) -> dict[str, np.ndarray]:
    out = dict(groups)
    if "AMB" in out and mult != 1.0:
        out["AMB"] = out["AMB"] * np.float32(mult)
    return out


def _led_min(groups: Mapping[str, np.ndarray], capture_cfg: Mapping[str, Any], combine: str = "min_luma") -> np.ndarray:
    """The noise-free LED image as lightproc combines L and R: luma min (pods) or mean (ring)."""
    col = np.asarray((capture_cfg.get("leds") or {}).get("colour") or [1.0, 1.0, 1.0], np.float32)
    L, R = np.clip(groups["L"], 0.0, None), np.clip(groups["R"], 0.0, None)
    if combine == "mean":
        return 0.5 * (L + R) / col
    return np.where((luma_lin(L) <= luma_lin(R))[..., None], L, R) / col


def combine_of(plan: Mapping[str, Any], capture_cfg: Mapping[str, Any]) -> str:
    """lightproc's L/R combination for the plan's illumination (extras.device.illumination)."""
    LP = capture_modules()[0]
    illum = ((plan.get("extras") or {}).get("device") or {}).get("illumination") or "pods"
    fn = getattr(LP, "combine_for", None)
    return fn({**getattr(LP, "DEFAULTS", {}), **(capture_cfg.get("process") or {})}, illum) if fn else "min_luma"


def reference_image(groups: Mapping[str, np.ndarray], roi_mask: np.ndarray | None,
                    capture_cfg: Mapping[str, Any], norm: float | None = None,
                    combine: str = "min_luma") -> tuple[np.ndarray, float]:
    """Noise-free, ambient-free LED capture (uint8 RGB) that judges which defects the device's
    own lighting can show, whatever the condition or arm: per pixel the L and R passes combined
    as lightproc does for the device's illumination (combine: min_luma = the darker by luma,
    pods; mean, ring), divided by the LED colour, normalised to the ROI percentile and
    gamma-encoded like lightproc.process_burst (max-channel ROI percentile). Returns
    (image, norm); pass norm back to encode a clean pass at the same scale."""
    proc = capture_cfg.get("process") or {}
    M = _led_min(groups, capture_cfg, combine)
    if norm is None:
        y = M.max(axis=-1)
        vals = y[roi_mask] if roi_mask is not None and roi_mask.any() else y.ravel()
        norm = float(np.percentile(vals, float(proc.get("percentile", 99.5))))
    x = np.clip(M / norm, 0.0, 1.0) if norm > 1e-12 else np.zeros_like(M)
    return to_u8(np.power(x, 1.0 / float(proc.get("gamma", 2.2)))), float(norm)


def naive_metrics(img_u8: np.ndarray, roi_mask: np.ndarray | None) -> dict[str, Any]:
    """What the old method can report about itself: ROI median luma and clipped share."""
    m = roi_mask if roi_mask is not None and roi_mask.any() else np.ones(img_u8.shape[:2], bool)
    y = img_u8.astype(np.float32) @ LUMA_U8
    return {"median_u8": round(float(np.median(y[m])), 2),
            "sat_pct": round(100.0 * float((img_u8[m] >= 250).any(axis=-1).mean()), 3)}


def run_arm(arm: str, groups: Mapping[str, np.ndarray], plan: Mapping[str, Any], rng: np.random.Generator,
            capture_cfg: Mapping[str, Any], roi_mask: np.ndarray, modules=None) -> tuple[dict[str, Any], Any]:
    """One capture arm on the light-group passes -> ({image (uint8 ROI), gate, reasons, message,
    metrics, gated}, the simulated Burst or None). The naive arm has no gate (gated false): the
    old method never refuses, so its gate is always PASS."""
    LP, B = modules or capture_modules()
    if arm == "burst":
        burst = B.simulate_burst(groups, plan, rng, capture_cfg, roi_mask=roi_mask)
        proc = LP.process_burst(burst, roi_mask, capture_cfg)
        g = proc.gate
        return {"image": np.asarray(proc.image_u8), "gate": g.verdict, "reasons": list(g.reasons or []),
                "message": g.message or "", "details": list(getattr(g, "details", None) or []),
                "metrics": json_safe(proc.metrics), "gated": True}, burst
    if arm == "naive":
        kw = {"roi_mask": roi_mask} if _accepts(B.simulate_naive, "roi_mask") else {}   # AE on the plate median
        img = np.asarray(B.simulate_naive(groups, plan, rng, capture_cfg, **kw))
        img = LP.naive_process(img) if hasattr(LP, "naive_process") else img
        return {"image": np.asarray(img), "gate": "PASS", "reasons": [], "message": "", "details": [],
                "metrics": naive_metrics(img, roi_mask), "gated": False}, None
    raise ValueError(f"unknown arm {arm!r} (one of {ARMS})")


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def judge_objects(grid: np.ndarray, objects: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any],
                  exclude: np.ndarray, ref_vis: Mapping[str, tuple], use_ref: bool,
                  penetrant: bool = False) -> list[dict[str, Any]]:
    """Objects with visible/contrast on one grid: the local-contrast test (DESIGN 5.5), or the
    clean-pass reference result when use_ref. Features and unlabelled objects get None."""
    out = []
    for o in objects:
        vis = con = None
        if o["label"] and o["group"] == "defect" and not penetrant:
            vis, con = ref_vis[o["id"]] if use_ref else visibility(grid, None, o, cfg, exclude)
        out.append({**o, "visible": vis, "contrast": con})
    return out


# ---- outputs: eval grids, bursts --------------------------------------------------------

def write_grid_image(path: Path, grid_u8: np.ndarray, fmt: str = "png", quality: int = 95) -> Path:
    """A full rectified grid as <path>.png (lossless) or <path>.jpg; returns the path written."""
    path = path.with_name(path.name + (".jpg" if fmt in ("jpg", "jpeg") else ".png"))
    write_rgb(path, grid_u8, quality if path.suffix == ".jpg" else None)
    return path


def label_records(objs: Sequence[Mapping[str, Any]], geom: Mapping[str, Any],
                  with_mm: bool = False) -> list[dict[str, Any]]:
    """Ground-truth defects (labelled or not) with grid px, mm and record-3.0 fields; with_mm
    adds the footprint polygons in plate mm (polys_mm, for realdata replays of a burst)."""
    recs = []
    for o in objs:
        if o["group"] == "defect":
            rec = object_record(o, geom, {})
            if with_mm:
                rec["polys_mm"] = [np.round(np.asarray(q, float), 4) for q in o.get("polys_mm") or []]
            recs.append(rec)
    return recs


def _json_bytes(x) -> np.ndarray:
    return np.frombuffer(json.dumps(json_safe(x)).encode(), np.uint8)


def save_burst(path: Path, burst, geom: Mapping[str, Any], labels: Sequence[Mapping[str, Any]],
               meta: Mapping[str, Any]) -> Path:
    """out/bursts/<scene>_v<k>.npz for the Pi demo (TwinSource) and realdata replays, in
    lightproc.save_burst's layout, so lightproc.load_burst reads it back as (Burst, extra).

    - frame_<B1|L|A|R|B2>: float16 (H, W, 3), linear, black-level-free, full-scale units;
    - meta_json (uint8 JSON bytes): the Burst meta (shifts, k, E_led, ..., material) plus
      "twin": scene_id, plate, k (variant), split, condition, lux, lux_rendered, amb_mult,
      device_mode, part and the twin's own gate result for this burst;
    - extra arrays: sequence; corners_px (4, 2) the imaged plate corners TL, TR, BR, BL in
      frame px; corner_plate_idx (plate corner of each: 0 datum (0, 0), 1 (W, 0), 2 (W, H),
      3 (0, H)); plate_size_mm [W, H]; grid_plate_mm (grid plate W x H, image-aligned);
      H_frame_plate; mm_per_px; border_mm; condition, lux, device_mode, material,
      thickness_mm; labels_json (uint8 JSON bytes: the sidecar's ground-truth defects plus
      polys_mm)."""
    seq = [str(n) for n in ((burst.meta or {}).get("sequence") or burst.frames)]
    shape = np.asarray(burst.frames[seq[0]]).shape
    to_frame = np.linalg.inv(_image_to_roi(shape, geom["roi"]))
    H = to_frame @ geom["H_roi_plate"]
    part = meta.get("part") or {}
    lux = meta.get("lux")
    arrays = {f"frame_{n}": np.asarray(burst.frames[n], np.float16) for n in seq}
    arrays.update(
        meta_json=_json_bytes({**(burst.meta or {}), "twin": {"schema": BURST_SCHEMA, **meta}}),
        sequence=np.array(seq), corners_px=F.apply_h(to_frame, geom["corners_roi"]),
        corner_plate_idx=np.asarray(geom["corner_plate_idx"], np.int64),
        plate_size_mm=np.asarray(geom["plate_size_mm"], float), grid_plate_mm=np.asarray(geom["grid_plate_mm"], float),
        H_frame_plate=H / H[2, 2], mm_per_px=np.float64(geom["mm_per_px"]),
        border_mm=np.float64(geom["grid"].border_mm),
        condition=np.array(meta.get("condition") or ""), lux=np.float64(np.nan if lux is None else lux),
        device_mode=np.array(meta.get("device_mode") or ""), material=np.array(part.get("material") or ""),
        thickness_mm=np.float64(np.nan if part.get("thickness_mm") is None else part["thickness_mm"]),
        labels_json=_json_bytes(list(labels)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


# ---- one scene ------------------------------------------------------------------------

def _arms_for(post_cfg: Mapping[str, Any], split: str, eval_arms: Sequence[str] | None,
              save_bursts: bool) -> tuple[str, list[str], list[str]]:
    """(train arm, eval arms, every arm to run) for a scene of this split."""
    train_arm = str(post_cfg.get("train_arm", "burst"))
    if eval_arms is None:
        grid_splits = post_cfg.get("grid_splits", ["test", "twin_test"])
        eval_arms = list(post_cfg.get("eval_arms") or []) if split in grid_splits else []
    eval_arms = list(dict.fromkeys(eval_arms))
    for a in [train_arm, *eval_arms]:
        if a not in ARMS:
            raise ValueError(f"unknown arm {a!r} (one of {ARMS})")
    run = [a for a in ARMS if a in {train_arm, *eval_arms} or (a == "burst" and save_bursts)]
    return train_arm, eval_arms, run


def process_scene(work_dir: str | Path, out_dir: str | Path, cfg: Mapping[str, Any],
                  variant_seed: int | None = None, debug: bool | None = None,
                  eval_arms: Sequence[str] | None = None, save_bursts: bool | None = None) -> dict[str, Any]:
    """plan.json + render.exr (+ render_clean.exr) -> the scene's dataset outputs (DESIGN_STEEL E).

    Per variant k (rng [seed, k]; each arm has its own stream [seed, k, 1 + arm index]):
    the ambient multiplier, the capture arms on the scaled passes (roi_mask = the projected
    plate), rectify, then
    - YOLO tiles + labels from post.train_arm for splits in post.tile_splits, unless its gate
      is RETAKE (recorded, no tiles); visibility is judged on that grid;
    - full grids per eval arm (post.eval_arms for splits in post.grid_splits, or eval_arms)
      at out/grids/<arm>/<split>/<scene>_v<k>.png with a sidecar <same>.json next to it;
    - out/bursts/<scene>_v<k>.npz for the first post.save_bursts_variants variants if save_bursts;
    - penetrant scenes: the train-arm grid at out/penetrant/<scene>_v<k>.png, no tiles.
    The scene sidecar's defect `visible` comes from the noise-free LED reference image (or
    the clean pass), so ground truth is the same for every condition and arm. Returns the sidecar."""
    t_start = time.perf_counter()
    work, out = Path(work_dir), Path(out_dir)
    plan = json.loads((work / "plan.json").read_text())
    post_cfg, grid_cfg, label_cfg = cfg.get("post") or {}, cfg.get("grid") or {}, cfg.get("label") or {}
    names = tuple((cfg.get("render") or {}).get("light_groups") or GROUPS)
    lights = read_light_groups(work / "render.exr", names)
    clean_path = work / "render_clean.exr"
    clean = read_light_groups(clean_path, names) if clean_path.exists() else None
    modules = capture_modules()
    cap_cfg = capture_config(cfg)

    geom = grid_geometry(plan, grid_cfg)
    gw, gh = geom["size"]
    tiles = T.tile_grid(gw, gh, int(grid_cfg.get("tile", T.TILE)), int(grid_cfg.get("min_overlap", T.MIN_OVERLAP)))
    objects = object_masks(plan, geom, geom["shape"])
    exclude = ring_exclude(objects, geom)
    pmask = plate_mask_image(geom, next(iter(lights.values())).shape)

    scene, mode = plan["scene_id"], plan.get("mode", "surface")
    split = plan.get("split", "train")
    penetrant = mode == "penetrant"
    info = scene_info(plan)
    plate_id = (plan.get("matrix") or {}).get("plate") or scene
    save_bursts = bool(post_cfg.get("save_bursts", False)) if save_bursts is None else bool(save_bursts)
    n_save = int(post_cfg.get("save_bursts_variants", 1))
    train_arm, eval_arms, run = _arms_for(post_cfg, split, eval_arms, save_bursts)
    tile_here = not penetrant and split in (post_cfg.get("tile_splits") or ["train", "val"])
    base_seed = int(plan.get("seed", 0) if variant_seed is None else variant_seed)
    keep_p = float(post_cfg.get("empty_tile_keep", 1.0))
    quality = int(post_cfg.get("jpeg_quality", 95))
    fmt = str(post_cfg.get("grid_format", "png")).lower()
    do_debug = _want_debug(out, debug)
    base = sidecar_base(geom, grid_cfg, tiles)

    # ground truth that does not depend on the condition, the arm or the noise
    comb = combine_of(plan, cap_cfg)
    ref_img, norm = reference_image(lights, pmask, cap_cfg, combine=comb)
    ref_grid = rectify(ref_img, geom)
    ref_clean = rectify(reference_image(clean, pmask, cap_cfg, norm, comb)[0], geom) if clean is not None else None
    ref_vis = {o["id"]: visibility(ref_grid, ref_clean, o, cfg, exclude)
               for o in objects if o["label"] and o["group"] == "defect" and not penetrant}
    ref_objs = judge_objects(ref_grid, objects, cfg, exclude, ref_vis, True, penetrant)
    labels_gt = label_records(ref_objs, geom, with_mm=True)

    variants = []
    for k in range(max(1, int(post_cfg.get("variants_per_scene", 1)))):
        rng = np.random.default_rng([base_seed, k])
        mult = amb_multiplier(rng, plan.get("ambient"), k, post_cfg)
        keep_draw = rng.random(len(tiles)) < keep_p
        groups = scale_ambient(lights, mult)
        lux0 = info["lux_rendered"]
        meta = {"scene_id": scene, "plate": plate_id, "k": k, "split": split, "condition": info["condition"],
                "lux": None if lux0 is None else round(float(lux0) * mult, 2), "lux_rendered": lux0,
                "amb_mult": round(mult, 6), "device_mode": info["device_mode"], "part": info["part"]}
        stem = f"{scene}_v{k}"
        res: dict[str, dict[str, Any]] = {}
        burst = None
        for arm in run:
            r, b = run_arm(arm, groups, plan, np.random.default_rng([base_seed, k, 1 + ARMS.index(arm)]),
                           cap_cfg, pmask, modules)
            r["grid"] = rectify(r.pop("image"), geom)
            res[arm] = r
            burst = b if b is not None else burst

        tr = res[train_arm]
        tr_objs = judge_objects(tr["grid"], objects, cfg, exclude, ref_vis, clean is not None, penetrant)
        boxes: list[tuple] = []
        written: list[str] = []
        if penetrant:
            write_rgb(out / "penetrant" / f"{stem}.png", tr["grid"])
        elif tile_here and tr["gate"] != RETAKE:
            boxes = label_boxes(tr_objs, tiles, label_cfg)
            written = write_tiles(tr["grid"], tiles, lines_from_boxes(tr_objs, tiles, boxes), out, split, stem,
                                  quality, keep_draw)
        tiles_of: dict[str, list[str]] = {}
        for i, t, _ in boxes:
            tiles_of.setdefault(tr_objs[i]["id"], []).append(t.name)

        arm_recs = {}
        for arm, r in res.items():
            rec = {k_: r[k_] for k_ in ("gate", "reasons", "message", "details", "metrics", "gated")}
            rec["grid"] = None
            if arm in eval_arms:
                objs = tr_objs if arm == train_arm else judge_objects(r["grid"], objects, cfg, exclude, ref_vis,
                                                                       clean is not None, penetrant)
                img = write_grid_image(out / "grids" / arm / split / stem, r["grid"], fmt, quality)
                rec["grid"] = img.relative_to(out).as_posix()
                rec["visible"] = {o["id"]: o["visible"] for o in objs if o["visible"] is not None}
                defects = label_records(ref_objs, geom)
                for d, o in zip(defects, [o for o in objs if o["group"] == "defect"]):
                    d.update(visible_arm=o["visible"], contrast_arm=_r(o["contrast"], 2))
                write_json(img.with_suffix(".json"), {
                    **base, "schema": GRID_SCHEMA, "stem": stem, "arm": arm, "mode": mode,
                    "source": plan.get("source"), "matrix": plan.get("matrix"), "cct_k": info["cct_k"], **meta,
                    "gate": rec["gate"], "gate_reasons": rec["reasons"], "gate_message": rec["message"],
                    "gate_details": rec["details"], "gated": rec["gated"], "metrics": rec["metrics"], "grid_image": img.name,
                    "defects": defects,
                    "features": [object_record(o, geom, {}) for o in ref_objs if o["group"] == "feature"],
                    "distractors": _distractors(plan)})
            arm_recs[arm] = rec

        npz = None
        if save_bursts and k < n_save and burst is not None:
            twin_gate = {k_: arm_recs["burst"][k_] for k_ in ("gate", "message", "metrics")}
            npz = save_burst(out / "bursts" / f"{stem}.npz", burst, geom, labels_gt,
                             {**meta, "gate": twin_gate}).relative_to(out).as_posix()
        if do_debug:
            write_debug(out / "debug" / f"{stem}.jpg", tr["grid"], tr_objs, boxes, tiles)
        variants.append({
            **{k_: meta[k_] for k_ in ("k", "condition", "lux", "amb_mult", "device_mode")},
            "seed": [base_seed, k], "arm": train_arm, "gate": tr["gate"], "gate_message": tr["message"],
            "metrics": tr["metrics"], "arms": arm_recs, "tiles_written": written, "n_labels": len(boxes),
            "visible": {o["id"]: o["visible"] for o in tr_objs if o["visible"] is not None},
            "contrast": {o["id"]: _r(o["contrast"], 2) for o in tr_objs if o["contrast"] is not None},
            "tiles": tiles_of, "burst_npz": npz,
            "penetrant_png": f"penetrant/{stem}.png" if penetrant else None,
        })

    tiles_of = variants[0]["tiles"]
    sidecar = {
        "schema": SCHEMA, "scene_id": scene, "plate": plate_id, "seed": plan.get("seed"), "split": split,
        "mode": mode, "source": plan.get("source"), "matrix": plan.get("matrix"), "clean_pass": clean is not None,
        "condition": info["condition"], "lux": info["lux_rendered"], "cct_k": info["cct_k"],
        "device_mode": info["device_mode"], "part": info["part"], "train_arm": train_arm, "eval_arms": eval_arms,
        **base,
        "features": [object_record(o, geom, tiles_of) for o in ref_objs if o["group"] == "feature"],
        "defects": [object_record(o, geom, tiles_of) for o in ref_objs if o["group"] == "defect"],
        "indications": [object_record(o, geom, tiles_of) for o in ref_objs if o["group"] == "indication"],
        "distractors": _distractors(plan),
        "variants": variants,
        "post_s": round(time.perf_counter() - t_start, 2),
    }
    write_json(out / "sidecar" / f"{scene}.json", sidecar)
    return json_safe(sidecar)


def _distractors(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"kind": d.get("kind"), "u_mm": (d.get("params") or {}).get("u"), "v_mm": (d.get("params") or {}).get("v")}
            for d in plan.get("distractors") or []]


def _scene_dirs(paths: Sequence[str]) -> list[Path]:
    dirs = []
    for p in map(Path, paths):
        if (p / "plan.json").exists():
            dirs.append(p)
        else:
            dirs.extend(sorted(q.parent for q in p.glob("*/plan.json")))
    return dirs


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="trinetra-twin post", description="Post-process rendered twin scenes into YOLO tiles + sidecars.")
    p.add_argument("--work", nargs="+", required=True, help="scene work dirs, or folders that hold them")
    p.add_argument("--out", required=True, help="dataset root (yolo/, sidecar/, grids/, bursts/, penetrant/, debug/)")
    p.add_argument("--config", default="configs/twin.yaml")
    p.add_argument("--debug", action="store_true", help="debug overlays for every scene (default: the first 50)")
    p.add_argument("--variant-seed", type=int, help="base seed instead of each plan's seed")
    p.add_argument("--eval-arms", nargs="*", choices=ARMS,
                   help="write full grids for these arms whatever the split (default: post.eval_arms for post.grid_splits)")
    p.add_argument("--save-bursts", action="store_true", help="also write out/bursts/<scene>_v<k>.npz")
    a = p.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    dirs = _scene_dirs(a.work)
    if not dirs:
        raise SystemExit(f"no plan.json under {a.work}")
    for d in dirs:
        sc = process_scene(d, a.out, cfg, a.variant_seed, True if a.debug else None, a.eval_arms,
                           True if a.save_bursts else None)
        n = sum(v["n_labels"] for v in sc["variants"])
        gates = ",".join(str(v["gate"]) for v in sc["variants"])
        print(f"{sc['scene_id']}: {sc['mode']} {sc['condition']}/{sc['device_mode']}, {len(sc['variants'])} variant(s) "
              f"[{gates}], {n} labels, {sc['post_s']} s")
    print(f"wrote {write_data_yaml(a.out, cfg.get('classes', T.CLASSES))}")


if __name__ == "__main__":
    main()
