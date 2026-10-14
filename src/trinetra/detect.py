"""Find where the part differs from the golden reference in any map (images and 3D slopes)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Detection, Grid
from .golden import Golden


@dataclass
class Region:
    id: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in grid px (x1, y1 exclusive)
    mask: np.ndarray  # bool, bbox-sized
    peak_z: float

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())


@dataclass
class Anomalies:
    regions: list[Region]
    score: np.ndarray  # max |z| over all maps
    rel: dict[str, np.ndarray]  # images: (image - golden) / golden; shape maps: slope - golden slope
    roi: np.ndarray


def find_anomalies(stack: dict[str, np.ndarray], valid: np.ndarray, golden: Golden, grid: Grid, det: Detection) -> Anomalies:
    roi = grid.plate_mask(grid.edge_margin_mm) & valid & golden.valid
    score = np.zeros(grid.shape, np.float32)
    rel = {}
    for k in golden.keys:
        mu, sd = golden.mean[k], golden.std[k]
        d = stack[k] - mu
        is_shape = k.startswith("ps_")
        floor = det.shape_sigma_floor if is_shape else det.sigma_floor
        z = d / np.sqrt(sd * sd + floor**2)
        if det.smooth_px > 0:
            z = cv2.GaussianBlur(z, (0, 0), det.smooth_px)
        np.maximum(score, np.abs(z), out=score)
        rel[k] = d if is_shape else d / np.maximum(mu, 0.02)

    mask = ((score > det.z_threshold) & roi).astype(np.uint8)
    k = max(int(round(det.close_mm / grid.mm_per_px)) | 1, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask &= roi.astype(np.uint8)

    # Pieces closer than group_mm (e.g. the patches of one irregular stain) form one region.
    g = max(int(round(det.group_mm / grid.mm_per_px)) | 1, 3)
    grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    labels[mask == 0] = 0
    min_px = det.min_area_mm2 / grid.mm_per_px**2
    found = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        ys, xs = np.nonzero(labels[y : y + h, x : x + w] == i)
        if len(xs) < min_px:
            continue
        x0, y0, x1, y1 = x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1
        local = labels[y0:y1, x0:x1] == i
        found.append((float(score[y0:y1, x0:x1][local].max()), (int(x0), int(y0), int(x1), int(y1)), local))
    found.sort(key=lambda r: -r[0])
    regions = [Region(f"D{i + 1}", bbox, local, peak) for i, (peak, bbox, local) in enumerate(found)]
    return Anomalies(regions, score, rel, roi)
