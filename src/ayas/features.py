"""Turn each anomaly region into numbers that describe it physically.

Evidence comes from three families:
- colour (top camera under its own white light): a stain or rust changes the colour,
  and the change looks the same from every camera;
- view/light geometry: scratches and dust scatter light, so they brighten strongly at
  the grazing camera under the grazing light; relief (dents, bumps) shades one side
  dark and the other bright under an oblique light;
- 3D shape (photometric stereo on the top camera): slope change and, with a full
  setup, depth in mm.
Plus the region's shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CHANNELS, Config, Grid, key
from .detect import Anomalies, Region

FEATURE_NAMES = (
    "c_top_r",
    "c_top_g",
    "c_top_b",
    "c_top_oblique_light",
    "c_top_grazing_light",
    "c_oblique45",
    "c_grazing",
    "a_top",
    "a_oblique45",
    "a_grazing",
    "geometry_ratio",
    "grazing_balance",
    "view_spread",
    "chroma_shift",
    "red_blue_ratio",
    "brightest_view",
    "sign_balance",
    "polarity",
    "slope_dev",
    "depth_mm",
    "area_mm2",
    "length_mm",
    "width_mm",
    "elongation",
    "circularity",
    "solidity",
)


@dataclass
class RegionFeatures:
    region_id: str
    x_mm: float
    y_mm: float
    orientation_deg: float
    values: dict[str, float]
    depth_available: bool

    def vector(self) -> np.ndarray:
        return np.array([self.values[n] for n in FEATURE_NAMES], np.float32)


@dataclass
class MapRoles:
    """Which stack keys play which role, for this rig's light names."""

    top_rgb: tuple[str, str, str]
    top_oblique: str
    top_grazing: str
    oblique45: str
    grazing: str
    oblique_light: str

    @classmethod
    def from_config(cls, cfg: Config) -> "MapRoles":
        t, o, g = cfg.pair_light("top"), cfg.pair_light("oblique45"), cfg.pair_light("grazing")
        return cls(
            top_rgb=tuple(f"{key('top', t)}_{c}" for c in CHANNELS),
            top_oblique=key("top", o),
            top_grazing=key("top", g),
            oblique45=key("oblique45", o),
            grazing=key("grazing", g),
            oblique_light=o,
        )


def _shape(mask: np.ndarray, mpp: float) -> dict[str, float]:
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = np.vstack(contours)  # a region can be several nearby pieces
    area_px = float(mask.sum())
    (_, _), (w, h), angle = cv2.minAreaRect(outline)
    length_px = max(w, h, 1.0)
    width_px = max(min(min(w, h), area_px / length_px), 1.0)  # area/length is better for curved lines
    if w < h:
        angle += 90.0
    perimeter = max(sum(cv2.arcLength(c, True) for c in contours), 1.0)
    hull_area = max(cv2.contourArea(cv2.convexHull(outline)), 1.0)
    return {
        "area_mm2": area_px * mpp * mpp,
        "length_mm": length_px * mpp,
        "width_mm": width_px * mpp,
        "elongation": length_px / width_px,
        "circularity": float(min(4 * np.pi * area_px / perimeter**2, 1.0)),
        "solidity": float(min(area_px / hull_area, 1.0)),
        "orientation_deg": float(angle % 180.0),
    }


def _depth(height: np.ndarray, region: Region, grid: Grid) -> float:
    """Signed depth (mm) of the region against a ring of surface around it (+ = raised)."""
    x0, y0, x1, y1 = region.bbox
    pad = int(round(1.5 / grid.mm_per_px)) + 1
    h, w = height.shape
    X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)
    full = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    full[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0] = region.mask
    ring = (cv2.dilate(full, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0) & (full == 0)
    patch = height[Y0:Y1, X0:X1]
    if ring.sum() < 5:
        return 0.0
    dev = patch[full > 0] - float(np.median(patch[ring]))
    return float(dev[np.argmax(np.abs(dev))])


def region_features(
    region: Region,
    anomalies: Anomalies,
    grid: Grid,
    cfg: Config,
    roles: MapRoles,
    height: np.ndarray | None = None,
) -> RegionFeatures:
    x0, y0, x1, y1 = region.bbox
    mask = region.mask
    eps = 1e-6
    rel = anomalies.rel
    v: dict[str, float] = {}

    def signed(k: str) -> float:
        return float(rel[k][y0:y1, x0:x1][mask].mean())

    def absolute(k: str) -> float:
        return float(np.abs(rel[k][y0:y1, x0:x1][mask]).mean())

    top = np.array([signed(k) for k in roles.top_rgb])
    for c, val in zip(CHANNELS, top):
        v[f"c_top_{c}"] = float(val)
    v["c_top_oblique_light"] = signed(roles.top_oblique)
    v["c_top_grazing_light"] = signed(roles.top_grazing)
    v["c_oblique45"] = signed(roles.oblique45)
    v["c_grazing"] = signed(roles.grazing)

    a_top = float(np.mean([absolute(k) for k in roles.top_rgb]))
    a_obl, a_graz = absolute(roles.oblique45), absolute(roles.grazing)
    v.update(a_top=a_top, a_oblique45=a_obl, a_grazing=a_graz)
    a_vals = np.array([a_top, a_obl, a_graz])
    v["geometry_ratio"] = a_graz / (a_top + 0.01)
    v["grazing_balance"] = float(v["c_grazing"] / (a_graz + eps))  # +1: uniformly brighter (scatter)
    v["view_spread"] = float(a_vals.std() / (a_vals.mean() + eps))
    v["chroma_shift"] = float(np.log(np.clip(1.0 + top, 0.05, None)).std())
    v["red_blue_ratio"] = float((1.0 + top[0]) / max(1.0 + top[2], 0.05))
    # Light trapping (cracks, pits): darker from the top under every light and from the
    # 45-degree camera. The grazing camera is left out: crack edges scatter light into it.
    v["brightest_view"] = max(float(top.mean()), v["c_oblique45"], v["c_top_oblique_light"], v["c_top_grazing_light"])

    # Shading under the oblique light, seen from the top: a dent is dark on the side
    # facing the light and bright on the far side; a bump is the reverse.
    shade = rel[roles.top_oblique][y0:y1, x0:x1][mask]
    v["sign_balance"] = float(abs(shade.mean()) / (np.abs(shade).mean() + eps))
    ys, xs = np.nonzero(mask)
    away = cfg.lights[roles.oblique_light].away_xy()
    u = (xs - xs.mean()) * away[0] + (ys - ys.mean()) * away[1]
    v["polarity"] = float(np.corrcoef(u, shade)[0, 1]) if len(u) > 3 and u.std() > 0 and shade.std() > 0 else 0.0

    if "ps_nx" in rel:
        dn = np.hypot(rel["ps_nx"][y0:y1, x0:x1][mask], rel["ps_ny"][y0:y1, x0:x1][mask])
        v["slope_dev"] = float(dn.mean())
    elif "ps_sa" in rel:
        v["slope_dev"] = float(np.abs(rel["ps_sa"][y0:y1, x0:x1][mask]).mean())
    else:
        v["slope_dev"] = 0.0
    v["depth_mm"] = _depth(height, region, grid) if height is not None else 0.0

    shape = _shape(mask, grid.mm_per_px)
    orientation = shape.pop("orientation_deg")
    v.update(shape)

    cx, cy = grid.px_to_mm(x0 + xs.mean(), y0 + ys.mean())
    return RegionFeatures(region.id, float(cx), float(cy), orientation, v, height is not None)
