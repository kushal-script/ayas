"""Patch anomaly map on the rectified plate grid (JEPA encoder + memory-bank head): inference side.

PROPOSAL, not wired into the app (docs/JEPA.md, "Integration"): the head goes into the pipeline
only if it passes the adoption gate on the twin test plates. numpy + cv2; onnxruntime only when
an ONNX head is loaded (OnnxPatchScorer), so this module imports on the Pi without torch.

The ONNX head (training/jepa/export_onnx.py) takes one normalised 224 x 224 RGB crop and returns
a (1, 14, 14) patch score: the distance of each patch embedding to its nearest neighbour in a
coreset of normal-surface patches (PatchCore-style), z-scored against the bank's own
nearest-neighbour distances. The grid is cut into windows (scale 2 = 448 px windows resized to
224, 2.56 mm patches: 9 windows for a 100 x 100 mm plate), the patch scores are pasted back,
averaged where windows overlap and lightly smoothed. `summarize` turns the map into the small
record block the Llama advisor reads; `unexplained` keeps the regions no detector box covers
(possible anomalies outside the 6 classes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

SCHEMA = "trinetra.anomaly/1"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def window_starts(length: int, win: int, stride: int) -> list[int]:
    """Start offsets of windows of size win covering [0, length); the last one ends at length."""
    if length <= win:
        return [0]
    s = list(range(0, length - win + 1, max(1, stride)))
    if s[-1] != length - win:
        s.append(length - win)
    return s


def windows(rect: Sequence[int], win: int, stride: int | None = None) -> list[tuple[int, int]]:
    """Top-left corners of windows covering the rectangle [x0, y0, x1, y1)."""
    x0, y0, x1, y1 = (int(v) for v in rect)
    stride = int(stride or win)
    return [(x0 + dx, y0 + dy) for dy in window_starts(y1 - y0, win, stride)
            for dx in window_starts(x1 - x0, win, stride)]


def crop(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    a, b, c, d = max(0, x), max(0, y), min(w, x + size), min(h, y + size)
    if c > a and d > b:
        out[b - y:d - y, a - x:c - x] = img[b:d, a:c]
    return out


def preprocess(crop_u8: np.ndarray, out: int = 224) -> np.ndarray:
    """uint8 RGB crop -> (1, 3, out, out) float32, ImageNet-normalised."""
    if crop_u8.shape[0] != out or crop_u8.shape[1] != out:
        crop_u8 = cv2.resize(crop_u8, (out, out), interpolation=cv2.INTER_AREA)
    x = (crop_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].copy()


def assemble(shape: tuple[int, int], items: Iterable[tuple[int, int, np.ndarray]], win: int,
             mask: np.ndarray | None = None, sigma_px: float = 0.0) -> np.ndarray:
    """Paste per-window patch scores (gh x gw, each patch = win / gw px) into a full-size map.

    Overlapping windows are averaged; pixels no window covered are 0; outside `mask` is 0."""
    h, w = shape
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for x, y, s in items:
        up = cv2.resize(np.asarray(s, np.float32), (win, win), interpolation=cv2.INTER_NEAREST)
        a, b, c, d = max(0, x), max(0, y), min(w, x + win), min(h, y + win)
        if c <= a or d <= b:
            continue
        acc[b:d, a:c] += up[b - y:d - y, a - x:c - x]
        cnt[b:d, a:c] += 1
    heat = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype(np.float32)
    if sigma_px > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigma_px)
    if mask is not None:
        heat = np.where(mask, heat, 0).astype(np.float32)
    return heat


def summarize(heat: np.ndarray, mm_per_px: float, threshold: float, origin_px: Sequence[float] = (0.0, 0.0),
              mask: np.ndarray | None = None, top_k: int = 3, min_area_mm2: float = 1.0,
              model: str = "") -> dict[str, Any]:
    """The record block: peak and 99th-percentile score, area above the threshold and the top_k
    regions (plate mm, grid frame: origin_px is the plate corner in grid px)."""
    vals = heat[mask] if mask is not None else heat.ravel()
    vals = vals[np.isfinite(vals)]
    above = (heat > threshold) & (mask if mask is not None else True)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(above.astype(np.uint8), connectivity=8)
    px_area = mm_per_px ** 2
    regions = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area * px_area < min_area_mm2:
            continue
        peak = float(heat[lab == i].max())
        regions.append({"x_mm": round((x - origin_px[0]) * mm_per_px, 1), "y_mm": round((y - origin_px[1]) * mm_per_px, 1),
                        "w_mm": round(w * mm_per_px, 1), "h_mm": round(h * mm_per_px, 1),
                        "area_mm2": round(area * px_area, 1), "peak": round(peak, 2),
                        "box_px": [x, y, x + w, y + h]})
    regions.sort(key=lambda r: -r["peak"])
    return {"schema": SCHEMA, "model": model, "threshold": round(float(threshold), 3),
            "score_max": round(float(vals.max()), 3) if vals.size else None,
            "score_p99": round(float(np.percentile(vals, 99)), 3) if vals.size else None,
            "area_mm2_above": round(float(sum(r["area_mm2"] for r in regions)), 1),
            "n_regions": len(regions), "regions": regions[:top_k]}


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of box a covered by box b ([x0, y0, x1, y1])."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def unexplained(summary: dict[str, Any], det_boxes_px: Sequence[Sequence[float]], cover: float = 0.3) -> list[dict[str, Any]]:
    """Anomaly regions that no detector box covers (>= `cover` of the region's box)."""
    return [r for r in summary.get("regions") or []
            if not any(_overlap(r["box_px"], b) >= cover for b in det_boxes_px)]


class OnnxPatchScorer:
    """An exported JEPA anomaly head (<name>.onnx + <name>.json from training/jepa/export_onnx.py)."""

    def __init__(self, onnx_path: str | Path, threads: int = 4):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        meta_path = Path(onnx_path).with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.size = int(self.meta.get("img_size", 224))
        self.scale = float(self.meta.get("scale", 2))
        self.threshold = float(self.meta.get("threshold", 3.0))

    def patch_scores(self, crop_u8: np.ndarray) -> np.ndarray:
        return np.asarray(self.sess.run(None, {self.input: preprocess(crop_u8, self.size)})[0])[0]

    def score_grid(self, grid_rgb: np.ndarray, rect: Sequence[int], mask: np.ndarray | None = None,
                   stride_frac: float = 1.0, sigma_px: float = 4.0) -> np.ndarray:
        """Heat map over the plate rectangle [x0, y0, x1, y1) of a grid image (RGB uint8)."""
        win = int(round(self.size * self.scale))
        items = [(x, y, self.patch_scores(crop(grid_rgb, x, y, win)))
                 for x, y in windows(rect, win, int(win * stride_frac))]
        return assemble(grid_rgb.shape[:2], items, win, mask, sigma_px)
