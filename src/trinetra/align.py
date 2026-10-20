"""Warp the camera images into the common grid and line them up with the reference.

After this step every grid pixel holds a signature of the same physical spot on the
plate: its colour under the top light, its brightness from every camera/light pair,
and (added later) its surface slope from the 3D shape map.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import cv2
import numpy as np

from .calibration import Calibration
from .capture import channel, lit
from .config import Config, Grid


def _antialias(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Blur before downsampling so thin defects are averaged, not skipped."""
    h, w = img.shape
    pts = np.array([[[w / 2, h / 2], [w / 2 + 1, h / 2], [w / 2, h / 2 + 1]]], np.float32)
    p = cv2.perspectiveTransform(pts, H)[0]
    sx, sy = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0])  # grid px per image px
    if sx >= 1 and sy >= 1:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=max(0.5 / sx - 0.5, 0.01), sigmaY=max(0.5 / sy - 0.5, 0.01))


def warp_to_grid(img: np.ndarray, H: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    warped = cv2.warpPerspective(_antialias(img, H), H, grid.size, flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpPerspective(np.ones(img.shape, np.uint8), H, grid.size, flags=cv2.INTER_NEAREST, borderValue=0)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0
    return warped, valid


def build_stack(frames: Mapping[str, np.ndarray], cal: Calibration, cfg: Config) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Dark-subtracted image maps (see Config.image_maps), warped into the grid."""
    grid = cfg.grid
    stack, valid = {}, np.ones(grid.shape, bool)
    cache: dict[tuple[str, str], np.ndarray] = {}
    for m in cfg.image_maps():
        if (m.view, m.light) not in cache:
            cache[(m.view, m.light)] = lit(frames, m.view, m.light)
        warped, v = warp_to_grid(channel(cache[(m.view, m.light)], m.channel), cal.homographies[m.view], grid)
        stack[m.key] = warped
        valid &= v
    return stack, valid


def _smooth_half(img: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.GaussianBlur(img, (0, 0), 2.0), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def estimate_motion(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rigid in-plane shift/rotation of the part relative to the reference (ECC on the
    smoothed image, so plate edges drive it, not the surface texture)."""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        _, warp = cv2.findTransformECC(
            _smooth_half(reference), _smooth_half(moving), warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
        )
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)
    warp[:, 2] *= 2.0  # back to full resolution
    return warp


def apply_motion(stack: dict[str, np.ndarray], valid: np.ndarray, warp: np.ndarray, grid: Grid):
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    out = {k: cv2.warpAffine(img, warp, grid.size, flags=flags, borderValue=0) for k, img in stack.items()}
    v = cv2.warpAffine(valid.astype(np.uint8), warp, grid.size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    return out, v


def match_gain(stack: dict[str, np.ndarray], medians: Mapping[str, float], roi: np.ndarray, keys: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Cancel small LED brightness drift: scale each image so its plate median matches the reference."""
    out = dict(stack)
    for k in keys if keys is not None else medians:
        m = float(np.median(stack[k][roi]))
        out[k] = stack[k] * float(np.clip(medians[k] / max(m, 1e-6), 0.8, 1.25))
    return out


def prepare(
    frames: Mapping[str, np.ndarray],
    cal: Calibration,
    cfg: Config,
    reference: np.ndarray | None,
    medians: Mapping[str, float] | None,
    setup,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Frames -> aligned, gain-matched stack with the 3D shape maps added."""
    from .shape import shape_maps

    grid = cfg.grid
    roi = grid.plate_mask(grid.edge_margin_mm)
    stack, valid = build_stack(frames, cal, cfg)
    if reference is not None:
        stack, valid = apply_motion(stack, valid, estimate_motion(stack[cfg.align_key], reference), grid)
    if medians is not None:
        stack = match_gain(stack, medians, roi)
    stack.update(shape_maps(stack, setup, roi & valid))
    return stack, valid
