"""One inspection, end to end (DESIGN_STEEL.md H): burst -> record -> alert -> log -> advice.

InspectionPipeline.run(burst, meta) -> record (trinetra.inspection/3.0):
1. lightproc.process_burst with the plate ROI (from meta corners_px, else found on B1 - A);
2. RETAKE gate (or no plate found): a RETAKE record with the gate message, nothing detected;
3. else rectify to the plate grid (0.08 mm/px, corners ordered by image position, exactly as
   twin/post.py), cut 640 px tiles, detector (npu or FakeDetector), merge_tiles;
4. grid px -> plate mm (config.Grid.px_to_mm), size measures (see measure_defect), near_edge_mm;
5. rules.judge_steel with spec_steel, report.build_steel_record (per-stage timing_ms);
6. alert (console line, optional buzzer, UI listeners), then the SQLite log (store.py schema)
   and log_dir/inspections.jsonl;
7. advisor (AdvisorService) AFTER the alert, on its own worker thread by default: the verdict and
   the log never wait for Llama; the advice is added to the record, the DB row and
   advice.jsonl when it arrives. A queued explanation is skipped (advice.jsonl
   'explain_skipped') when a newer part was inspected before the advisor got to it.

Conventions: frame and grid pixel centres are integers (frames.py); detector boxes use pixel
edges (pixel i spans [i, i + 1)), so a box edge x maps to plate mm at Grid.px_to_mm(x - 0.5).
Plate mm here are grid-aligned (origin at the imaged top-left plate corner), as on the twin's
records (twin/post.record_fields). No Blender or OpenEXR imports.
"""

from __future__ import annotations

import json
import math
import queue
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .. import lightproc as LP
from .. import npu, rules
from .. import report as RP
from .. import tiles as T
from ..config import Grid
from ..store import SCHEMA as STORE_SCHEMA
from ..store import Store
from ..twin import frames as F
from . import repo_path
from .advisor import QUESTIONS, AdviceResult

NO_PLATE = "Plate not found: put the whole plate in view, on a dark background, and retake."
SINGLE_MSG = "Single image: no capture-quality check (baseline arm)."
DEVICE_MODES = ("open", "hood")
MEASURE = {"pad_px": 4, "k_sigma": 3.0, "min_dn": 8.0, "min_px": 3, "rel": 0.5}   # rel: half maximum (geometric classes)
# RGB colours of the result image, by disposition (and verdict for the banner)
COLOURS = {"PASS": (60, 190, 90), "CLEAN_AND_REINSPECT": (20, 170, 220), "REWORK": (245, 140, 20),
           "HOLD_FOR_QA": (180, 90, 230), "REJECT": (235, 50, 50), "RETAKE": (250, 180, 0)}


# ---- config ---------------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(repo_path(path).read_text()) or {}


def load_app_config(path: str | Path = "configs/app.yaml") -> dict[str, Any]:
    """configs/app.yaml plus the shared configs it points at: capture (capture.yaml), grid (twin.yaml
    `grid:`) and spec (spec_steel.yaml), loaded as dicts under _capture, _grid and _spec."""
    cfg = load_yaml(path)
    cfg["_capture"] = LP.load_capture_config(repo_path(cfg.get("capture_config", "configs/capture.yaml")))
    cfg["_grid"] = dict(load_yaml(cfg.get("grid_config", "configs/twin.yaml")).get("grid") or {})
    cfg["_spec"] = rules.load_spec(repo_path(cfg.get("spec", "configs/spec_steel.yaml")))
    return cfg


def build_detector(spec: str, conf: float = 0.25):
    """'fake' | 'onnx:<path>' | 'hailo:<hef>' | 'pt:<path>' -> (detector, vision backend name)."""
    kind, _, arg = str(spec).partition(":")
    if kind == "fake":
        det = FakeDetector(truth=arg != "empty")
        return det, det.name
    path = repo_path(arg)
    if kind == "onnx":
        return npu.OnnxDetector(path, conf=conf), f"yolov8n onnxruntime CPU ({path.name})"
    if kind == "hailo":
        return npu.HailoDetector(path, conf=conf), f"yolov8n Hailo-10H HailoRT ({path.name})"
    if kind == "pt":
        return npu.UltralyticsDetector(path, conf=conf), f"yolov8n ultralytics ({path.name})"
    raise ValueError(f"unknown detector {spec!r}: fake | onnx:<path> | hailo:<hef> | pt:<path>")


# ---- geometry ---------------------------------------------------------------------------------

@dataclass
class GridMap:
    grid: Grid                     # image-aligned plate grid (plate_w_mm is the imaged TL -> TR edge)
    H: np.ndarray                  # 3 x 3, grid px <- frame px
    corners: np.ndarray            # (4, 2) frame px, TL, TR, BR, BL by image position

    @property
    def size(self) -> tuple[int, int]:
        return self.grid.size

    @property
    def mm_per_px(self) -> float:
        return self.grid.mm_per_px


def grid_map(corners, plate_size_mm: Sequence[float], grid_cfg: Mapping[str, Any] | None = None,
             grid_plate_mm: Sequence[float] | None = None, corner_plate_idx: Sequence[int] | None = None) -> GridMap:
    """Frame -> grid homography from the 4 imaged plate corners (any order).

    The grid is (W, H) or (H, W), whichever plate edge is imaged TL -> TR: given by grid_plate_mm
    (twin bursts), else by corner_plate_idx (plate corner of the imaged TL; 0 datum, 1 (W, 0), ...),
    else by the imaged edge lengths (the longer plate side goes along the longer imaged edge)."""
    g = grid_cfg or {}
    ordered = F.order_corners(corners)
    W, H = float(plate_size_mm[0]), float(plate_size_mm[1])
    if grid_plate_mm is not None:
        gw, gh = float(grid_plate_mm[0]), float(grid_plate_mm[1])
    elif corner_plate_idx is not None:
        gw, gh = (W, H) if int(corner_plate_idx[0]) % 2 == 0 else (H, W)
    else:
        top = float(np.linalg.norm(ordered[1] - ordered[0]))
        left = float(np.linalg.norm(ordered[3] - ordered[0]))
        big, small = max(W, H), min(W, H)
        gw, gh = (big, small) if top >= left else (small, big)
    grid = Grid(gw, gh, border_mm=float(g.get("border_mm", 5.0)), mm_per_px=float(g.get("mm_per_px", 0.08)))
    Hm = cv2.getPerspectiveTransform(ordered.astype(np.float32), grid.plate_corners_px()).astype(float)
    return GridMap(grid, Hm, ordered)


def rectify(img: np.ndarray, gm: GridMap) -> np.ndarray:
    """Frame image -> the plate grid (cv2.warpPerspective, bilinear, black outside), like post.rectify."""
    return cv2.warpPerspective(np.ascontiguousarray(img), gm.H, gm.size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def roi_mask(corners, shape: Sequence[int]) -> np.ndarray:
    """Bool mask of the plate quadrilateral in a frame of this shape (1/16 px corners, like post)."""
    pts = F.order_corners(corners)
    m = np.zeros(tuple(shape[:2]), np.uint8)
    cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, 4)
    return m.astype(bool)


def auto_corners(item, min_area_frac: float = 0.02) -> np.ndarray | None:
    """Plate corners found on the LED-only image (B1 - A) of a burst, or on a single image's grey
    (calibration.find_plate_corners: bright plate on a dark background); None if not found."""
    from ..calibration import CalibrationError, find_plate_corners

    if isinstance(item, LP.Burst):
        y = LP.luma(item.frames["B1"]) - LP.luma(item.frames["A"])
        top = float(np.percentile(y, 99.0))
        gray = np.clip(y / top * 255.0, 0, 255).astype(np.uint8) if top > 1e-6 else np.zeros(y.shape, np.uint8)
    else:
        img = np.asarray(item)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    try:
        c = find_plate_corners(gray)
    except (CalibrationError, cv2.error, ValueError):
        return None
    area = abs(cv2.contourArea(np.asarray(c, np.float32).reshape(-1, 1, 2)))
    return np.asarray(c, float) if area >= min_area_frac * gray.shape[0] * gray.shape[1] else None


# ---- measures from a box ----------------------------------------------------------------------

def defect_mask(grid_u8: np.ndarray, box: Sequence[float], plate: np.ndarray, pad_px: int = 4,
                k_sigma: float = 3.0, min_dn: float = 8.0, min_px: int = 3, rel: float = 0.0):
    """Pixels inside a detector box that differ from the local background -> (mask, x0, y0) or None.

    Background = per-channel median of a pad_px ring around the box (on the plate); a pixel is
    'defect' when its largest channel difference exceeds median + k_sigma x MAD of the ring's
    differences (at least min_dn) and, with rel > 0, rel x the box's peak difference (rel 0.5 =
    full width at half maximum: widths do not depend on the defect's contrast or edge blur).
    Single isolated pixels are dropped. The window origin (x0, y0) is in grid px."""
    h, w = grid_u8.shape[:2]
    c0, r0 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    c1, r1 = min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    if c1 - c0 < 1 or r1 - r0 < 1:
        return None
    wx0, wy0, wx1, wy1 = max(0, c0 - pad_px), max(0, r0 - pad_px), min(w, c1 + pad_px), min(h, r1 + pad_px)
    win = grid_u8[wy0:wy1, wx0:wx1].astype(np.float32)
    win = win if win.ndim == 3 else win[..., None]
    on = plate[wy0:wy1, wx0:wx1]
    inbox = np.zeros(on.shape, bool)
    inbox[r0 - wy0:r1 - wy0, c0 - wx0:c1 - wx0] = True
    ref = ~inbox & on
    if ref.sum() < 12:
        ref = inbox & on
    if ref.sum() < 4:
        return None
    bg = np.median(win[ref], axis=0)
    diff = np.abs(win - bg).max(axis=-1)
    rd = diff[ref]
    med = float(np.median(rd))
    mad = 1.4826 * float(np.median(np.abs(rd - med)))
    thr = max(med + k_sigma * mad, min_dn)
    inside = diff[inbox & on]
    if rel > 0 and inside.size:
        thr = max(thr, rel * float(np.percentile(inside, 99.0)))
    m = (diff > thr) & inbox & on
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 2]
    if keep:
        m = np.isin(lab, keep)
    if int(m.sum()) < min_px:
        return None
    return m, wx0, wy0


def measure_defect(cls: str, box: Sequence[float], grid_u8: np.ndarray, gm: GridMap, plate: np.ndarray,
                   opts: Mapping[str, Any] | None = None) -> tuple[dict[str, float], list[float], str]:
    """(size in mm, extent box in grid px edges, method 'mask' | 'box') for one detection.

    The device never measures depth. Detector boxes are padded to >= 8 px (0.64 mm) in training,
    so sizes that rules compare with sub-mm limits come from the defect's own pixels (defect_mask;
    half-maximum threshold for scratch, crack and pit, every changed pixel for corrosion):
    - scratch, crack: length = the box diagonal (the box is the mark's extent; a straight mark is
      at most this long, and faint or curved parts that the mask misses still count; on the
      rendered twin the mask alone gave 25 mm for a 36 mm curved scriber mark); width = mask area
      / the mask's own long side (+1 px), i.e. the mean visible width; a patch-like scratch box
      (aspect < 3, abrasion) adds area_mm2; box fallback width = the short side (an upper bound);
    - pit: one blob -> diameter (equivalent circle of the blob), count 1; several blobs (cluster)
      -> count, pit_d_mm (largest blob), diameter and area of their convex hull; box fallback:
      diameter = long side, count 1;
    - corrosion: area = mask area (box fallback: the inscribed ellipse), equivalent diameter, and
      length_mm = long side for a streak (aspect >= 3);
    - dent: diameter = long box side (a dent shows as shading over its whole footprint);
    - stain: area = the box's inscribed ellipse (faint rings and smears fill it only in part).
    """
    o = {**MEASURE, **(opts or {})}
    mpp = gm.mm_per_px
    x0, y0, x1, y1 = (float(v) for v in box)
    bw, bh = max(x1 - x0, 1.0) * mpp, max(y1 - y0, 1.0) * mpp
    big, small = max(bw, bh), min(bw, bh)
    blob = None
    if cls in ("scratch", "crack", "pit", "corrosion"):
        rel = float(o["rel"]) if cls in ("scratch", "crack", "pit") else 0.0   # areas keep every changed pixel
        blob = defect_mask(grid_u8, box, plate, int(o["pad_px"]), float(o["k_sigma"]), float(o["min_dn"]),
                           int(o["min_px"]), rel)
    ext = [x0, y0, x1, y1]
    if blob is not None:
        m, ox, oy = blob
        ys, xs = np.nonzero(m)
        pts = np.column_stack([xs + ox, ys + oy]).astype(np.float32)
        n_px = int(xs.size)
        ext = [float(xs.min() + ox), float(ys.min() + oy), float(xs.max() + ox + 1), float(ys.max() + oy + 1)]
    if cls in ("scratch", "crack"):
        if blob is None:
            return {"length_mm": math.hypot(bw, bh), "width_mm": small}, ext, "box"
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        own = max(rw, rh) + 1.0
        width = min(max(n_px / own, 1.0), min(rw, rh) + 1.0)
        size = {"length_mm": max(math.hypot(bw, bh), own * mpp), "width_mm": width * mpp}
        if cls == "scratch" and big < 3.0 * small:
            size["area_mm2"] = n_px * mpp * mpp
        return size, ext, "mask"
    if cls == "pit":
        if blob is None:
            return {"diameter_mm": big, "count": 1}, ext, "box"
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        areas = st[1:, cv2.CC_STAT_AREA].astype(float)
        d_max = math.sqrt(4.0 * float(areas.max()) / math.pi) * mpp
        if len(areas) <= 1:
            return {"diameter_mm": d_max, "count": 1}, ext, "mask"
        hull = max(float(cv2.contourArea(cv2.convexHull(pts))), float(n_px))
        return {"diameter_mm": math.sqrt(4.0 * hull / math.pi) * mpp, "count": int(len(areas)),
                "area_mm2": hull * mpp * mpp, "pit_d_mm": d_max}, ext, "mask"
    if cls == "corrosion":
        area = n_px * mpp * mpp if blob is not None else math.pi / 4.0 * bw * bh
        size = {"area_mm2": area, "diameter_mm": math.sqrt(4.0 * area / math.pi)}
        if big >= 3.0 * small:
            size["length_mm"] = big
        return size, ext, "mask" if blob is not None else "box"
    if cls == "dent":
        return {"diameter_mm": big}, ext, "box"
    if cls == "stain":
        area = math.pi / 4.0 * bw * bh
        return {"area_mm2": area, "diameter_mm": math.sqrt(4.0 * area / math.pi)}, ext, "box"
    return {"length_mm": big}, ext, "box"


def box_to_mm(gm: GridMap, box: Sequence[float]) -> tuple[float, float, float, float]:
    """Pixel-edge box in grid px -> (x0, y0, x1, y1) in grid-aligned plate mm."""
    a = gm.grid.px_to_mm(float(box[0]) - 0.5, float(box[1]) - 0.5)
    b = gm.grid.px_to_mm(float(box[2]) - 0.5, float(box[3]) - 0.5)
    return a[0], a[1], b[0], b[1]


def box_gap_mm(box_mm: Sequence[float], hole: Sequence[float]) -> float:
    """Edge-to-edge gap (mm) from a box (x0, y0, x1, y1) to a hole (x, y, d), 0 when they overlap."""
    x0, y0, x1, y1 = box_mm
    cx, cy, d = (float(v) for v in hole[:3])
    return max(0.0, math.hypot(max(x0 - cx, 0.0, cx - x1), max(y0 - cy, 0.0, cy - y1)) - d / 2.0)


def holes_grid_mm(meta: Mapping[str, Any], gm: GridMap) -> list[list[float]]:
    """Through holes as [x, y, d] in grid-aligned plate mm: meta 'holes_mm' (already grid-aligned,
    e.g. a jig's part.holes_mm), plus meta 'holes_plate_mm' [u, v, d] in plate mm mapped through
    meta 'H_frame_plate' (twin bursts) and the grid map."""
    out = [[float(v) for v in h[:3]] for h in meta.get("holes_mm") or []]
    hp, Hfp = meta.get("holes_plate_mm"), meta.get("H_frame_plate")
    if hp is not None and len(hp) and Hfp is not None:
        hp = np.asarray(hp, float).reshape(-1, 3)
        px = F.apply_h(gm.H @ np.asarray(Hfp, float), hp[:, :2])
        out += [[*gm.grid.px_to_mm(float(x), float(y)), float(d)] for (x, y), d in zip(px, hp[:, 2])]
    return out


def detection_record(d: T.Det, grid_u8: np.ndarray, gm: GridMap, plate: np.ndarray,
                     opts: Mapping[str, Any] | None = None, holes: Sequence[Sequence[float]] = ()) -> dict[str, Any]:
    """One merged detection -> a record-3.0 defect (before the rules): type, confidence, box_px,
    location_mm (box centre), size (mm), near_edge_mm (edge-to-edge distance of the box, the
    defect's extent, to the nearest plate edge or known hole [x, y, d] (grid-aligned mm,
    holes_grid_mm): never larger than the mask's, so the edge rule errs on the safe side. The
    device does not detect holes: without a hole list only the plate edges count)."""
    box = [d.x0, d.y0, d.x1, d.y1]
    size, _, how = measure_defect(d.name, box, grid_u8, gm, plate, opts)
    cx, cy = gm.grid.px_to_mm((d.x0 + d.x1) / 2.0 - 0.5, (d.y0 + d.y1) / 2.0 - 0.5)
    ex0, ey0, ex1, ey1 = box_to_mm(gm, box)
    near = min([ex0, ey0, gm.grid.plate_w_mm - ex1, gm.grid.plate_h_mm - ey1]
               + [box_gap_mm((ex0, ey0, ex1, ey1), h) for h in holes])
    return {"type": d.name, "confidence": float(d.conf), "box_px": box, "location_mm": {"x": cx, "y": cy},
            "size": size, "near_edge_mm": max(0.0, near), "measured_by": how, "tile": d.tile}


# ---- detectors --------------------------------------------------------------------------------

class FakeDetector:
    """Stand-in detector for tests and for the demo without a trained model.

    Boxes (grid px, pixel edges) come from `boxes` [(cls, conf, [x0, y0, x1, y1]), ...] and,
    with truth=True, from the twin burst's ground-truth labels (meta['labels'], visible labelled
    defects): a replay of the truth through the real tile/merge/measure/rules path. The record
    names this backend 'fake (ground-truth replay)': it is never a detection result."""

    names = list(T.CLASSES)

    def __init__(self, boxes: Sequence[tuple[str, float, Sequence[float]]] | None = None, truth: bool = True,
                 conf: float = 0.9):
        self.fixed = [(str(c), float(s), [float(v) for v in b]) for c, s, b in (boxes or [])]
        self.truth, self.conf = bool(truth), float(conf)
        self.name = "fake (ground-truth replay)" if truth else "fake (fixed boxes)"
        self.calls = 0
        self._pending: dict[int, list[T.Det]] = {}

    def prime(self, meta: Mapping[str, Any], tiles: Sequence[T.Tile], size: Sequence[int]) -> None:
        """Put each box, unclipped, in the tile whose centre is nearest to the box centre."""
        boxes = list(self.fixed)
        if self.truth:
            for lab in meta.get("labels") or []:
                cls = lab.get("type") or lab.get("cls")
                if (cls in T.CLASSES and lab.get("box_px") is not None and lab.get("label", True)
                        and lab.get("visible") is not False):
                    boxes.append((cls, float(lab.get("confidence") or self.conf), lab["box_px"]))
        self._pending = {}
        for cls, conf, b in boxes:
            if cls not in T.CLASSES:
                continue
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            i = min(range(len(tiles)), key=lambda k: (tiles[k].x0 + tiles[k].size / 2 - cx) ** 2
                    + (tiles[k].y0 + tiles[k].size / 2 - cy) ** 2)
            t = tiles[i]
            self._pending.setdefault(i, []).append(
                T.Det(T.CLASSES.index(cls), conf, b[0] - t.x0, b[1] - t.y0, b[2] - t.x0, b[3] - t.y0))

    def detect(self, tiles: list[np.ndarray]) -> list[list[T.Det]]:
        self.calls += 1
        out = [list(self._pending.get(i, [])) for i in range(len(tiles))]
        self._pending = {}
        return out


# ---- log ------------------------------------------------------------------------------------

class AppStore(Store):
    """store.Store (same tables) shared by the app's threads: one connection behind a lock."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(STORE_SCHEMA)

    def add_inspection(self, report: dict[str, Any]) -> int:
        with self.lock:
            return super().add_inspection(report)

    def add_chat(self, inspection_id: int, role: str, text: str) -> None:
        with self.lock:
            super().add_chat(inspection_id, role, text)

    def update_inspection(self, inspection_id: int, report: dict[str, Any]) -> None:
        with self.lock:
            self.db.execute("UPDATE inspections SET verdict = ?, defect_count = ?, report_json = ? WHERE id = ?",
                            (report["verdict"], len(report["defects"]), json.dumps(report), int(inspection_id)))
            self.db.commit()

    def recent(self, n: int = 10) -> list[tuple]:
        with self.lock:
            return super().recent(n)

    def verdict_counts(self) -> dict[str, int]:
        with self.lock:
            return super().verdict_counts()

    def records(self, n: int = 50) -> list[dict[str, Any]]:
        """The last n records, oldest first."""
        with self.lock:
            rows = self.db.execute("SELECT report_json FROM inspections ORDER BY id DESC LIMIT ?", (int(n),)).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    def chats(self, inspection_id: int) -> list[tuple]:
        with self.lock:
            return self.db.execute("SELECT ts, role, text FROM chats WHERE inspection_id = ? ORDER BY id",
                                   (int(inspection_id),)).fetchall()

    def close(self) -> None:
        with self.lock:
            self.db.close()


class AppLog:
    """log_dir/trinetra.db (SQLite), inspections.jsonl (one record per line), advice.jsonl
    (advisor answers and operator acknowledgements) and images/<id>_<part>.jpg."""

    def __init__(self, log_dir: str | Path, save_images: bool = True):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = AppStore(self.dir / "trinetra.db")
        self.jsonl = self.dir / "inspections.jsonl"
        self.advice_jsonl = self.dir / "advice.jsonl"
        self.images = self.dir / "images" if save_images else None
        self._lock = threading.Lock()

    def _append(self, path: Path, obj: Mapping[str, Any]) -> None:
        with self._lock, open(path, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def inspection(self, record: dict[str, Any], jpeg: bytes | None = None) -> int:
        i = self.store.add_inspection(record)
        record["inspection_id"] = i
        self._append(self.jsonl, record)
        if self.images is not None and jpeg:
            self.images.mkdir(exist_ok=True)
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(record["part_id"]))[:60]
            (self.images / f"{i:06d}_{safe}.jpg").write_bytes(jpeg)
        return i

    def event(self, obj: Mapping[str, Any]) -> None:
        self._append(self.advice_jsonl, {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **obj})

    def close(self) -> None:
        self.store.close()


class Alerter:
    """The operator alert: one console line per record and an optional GPIO buzzer (lazy gpiozero)."""

    def __init__(self, buzzer_pin: int | None = None, beep_ms: int = 120, beep_on: Sequence[str] = ("REJECT",),
                 echo: bool = True, out: Callable[[str], None] | None = None):
        self.beep_ms, self.beep_on, self.echo = int(beep_ms), tuple(beep_on), bool(echo)
        self.out = out or (lambda s: print(s, flush=True))
        self._buzzer = None
        if buzzer_pin is not None:
            try:
                from gpiozero import DigitalOutputDevice
                self._buzzer = DigitalOutputDevice(int(buzzer_pin), initial_value=False)
            except Exception as e:   # no GPIO (Mac) or pin busy: the screen still alerts
                print(f"buzzer disabled: {e}", file=sys.stderr)

    def __call__(self, record: Mapping[str, Any]) -> None:
        if self.echo:
            self.out(alert_line(record))
        if self._buzzer is not None and record.get("verdict") in self.beep_on:
            self._buzzer.blink(on_time=self.beep_ms / 1e3, off_time=0.1, n=2, background=True)


def alert_line(record: Mapping[str, Any]) -> str:
    cap = record.get("capture") or {}
    if record.get("verdict") == "RETAKE":
        return f"[RETAKE] {record['part_id']}: {cap.get('gate_message') or 'capture refused'}"
    by = ", ".join(f"{k} {v}" for k, v in (record.get("summary") or {}).get("by_type", {}).items()) or "no defects"
    flag = " (capture flagged)" if cap.get("gate") == "PASS_FLAGGED" else ""
    return (f"[{record['verdict']}] {record['part_id']}: {by}{flag}; {record.get('timing_ms', {}).get('total', 0):.0f} ms. "
            f"{record.get('verdict_instruction', '')}")


def overlay_jpeg(img_rgb: np.ndarray, defects: Sequence[Mapping[str, Any]], verdict: str,
                 corners: np.ndarray | None = None, px: int = 480, quality: int = 85) -> bytes:
    """Result image for the UI and the log: boxes coloured by disposition with their ids, or the
    plate outline on the unrectified frame (RETAKE); longest side px."""
    img = np.asarray(img_rgb)
    h, w = img.shape[:2]
    s = min(1.0, float(px) / max(h, w))
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    small = np.ascontiguousarray(small)
    if corners is not None:
        pts = np.round(np.asarray(corners, float) * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(small, [pts], True, COLOURS.get(verdict, (200, 200, 200)), 2, cv2.LINE_AA)
    for d in defects:
        b = d.get("box_px")
        if not b:
            continue
        col = COLOURS.get(d.get("disposition", ""), (255, 255, 255))
        p0 = (int(b[0] * s), int(b[1] * s))
        p1 = (max(int(math.ceil(b[2] * s)), p0[0] + 2), max(int(math.ceil(b[3] * s)), p0[1] + 2))
        cv2.rectangle(small, (p0[0] - 2, p0[1] - 2), (p1[0] + 2, p1[1] + 2), col, 2, cv2.LINE_AA)
        cv2.putText(small, str(d.get("id", "")), (p0[0], max(12, p0[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes() if ok else b""


# ---- the pipeline -----------------------------------------------------------------------------

@dataclass
class Inspection:
    record: dict[str, Any]
    jpeg: bytes
    meta: dict[str, Any]
    key: int                                   # the log id (or a running number without a log)
    advice: list[AdviceResult] = field(default_factory=list)
    surface: Any = None                        # surface3d.SurfaceSource for an on-demand 3D view (bursts only)


class _Lap:
    def __init__(self, tm: dict[str, float]):
        self.tm, self.t = tm, time.perf_counter()

    def __call__(self, key: str) -> None:
        now = time.perf_counter()
        self.tm[key] = self.tm.get(key, 0.0) + (now - self.t) * 1e3
        self.t = now


class InspectionPipeline:
    """camera burst -> record -> alert -> log -> advice (see the module docstring).

    advise: 'thread' (default: the advisor works on its own thread after the alert), 'sync'
    (right after the alert, before run() returns: CLI and tests) or 'off'. listeners get
    (event, inspection, **kw) for 'inspection', 'advice_start', 'advice_text' (text=...) and
    'advice_done' (result=AdviceResult)."""

    def __init__(self, capture_cfg: Mapping[str, Any], grid_cfg: Mapping[str, Any], detector, spec: Mapping[str, Any],
                 part: Mapping[str, Any] | None = None, advisor=None, *, log: AppLog | None = None,
                 alerter: Callable[[Mapping[str, Any]], None] | None = None, advise: str = "thread",
                 vision_name: str | None = None, image_px: int = 480, measure: Mapping[str, Any] | None = None):
        self.capture_cfg, self.grid_cfg = dict(capture_cfg), dict(grid_cfg or {})
        self.detector, self.spec, self.part = detector, dict(spec), dict(part or {})
        self.advisor, self.log, self.advise = advisor, log, advise
        self.alerter = alerter if alerter is not None else Alerter(echo=False)
        self.vision_name = vision_name or getattr(detector, "name", type(detector).__name__)
        self.image_px, self.measure = int(image_px), dict(measure or {})
        self.listeners: list[Callable[..., None]] = []
        self.lock = threading.RLock()
        self.last: Inspection | None = None
        self.count = 0
        self._q: queue.Queue | None = None
        self._worker: threading.Thread | None = None

    # -- one inspection --
    def run(self, item, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Burst (or single uint8 RGB image) + source meta -> the record (alert and log done)."""
        return self.inspect(item, meta).record

    def inspect(self, item, meta: Mapping[str, Any] | None = None) -> Inspection:
        t_start = time.perf_counter()
        meta = dict(meta or {})
        partner = meta.pop("partner", None)          # (burst, meta) of the same inspection's second burst
        tm: dict[str, float] = {"capture": float(meta.get("capture_ms") or 0.0)}
        lap = _Lap(tm)
        part = self._part(meta)
        device_mode = str(meta.get("device_mode") or self.part.get("device_mode") or "open")
        device_mode = device_mode if device_mode in DEVICE_MODES else "open"
        corners = None if meta.get("corners_px") is None else np.asarray(meta["corners_px"], float).reshape(4, 2)
        proc, surface = None, None
        if isinstance(item, LP.Burst):
            if corners is None:
                corners = auto_corners(item)
            shape = next(iter(item.frames.values())).shape
            proc = LP.process_burst(item, None if corners is None else roi_mask(corners, shape), self.capture_cfg)
            image, gate = proc.image_u8, proc.gate
            if corners is None:
                gate = LP.GateResult("RETAKE", [*gate.reasons, "no_plate"], NO_PLATE, [*gate.details, "plate not found"])
            capture = RP.capture_from_metrics(proc.metrics, gate, mode="burst5", device_mode=device_mode,
                                              lux_est=proc.metrics.get("lux_est"))
            capture.update(self._capture_extra(proc.metrics, gate))
        else:
            image = np.ascontiguousarray(item)
            if image.dtype != np.uint8 or image.ndim != 3:
                raise ValueError("a single image must be uint8 (H, W, 3) RGB")
            if corners is None:
                corners = auto_corners(image)
            gate = (LP.GateResult("PASS", [], SINGLE_MSG) if corners is not None
                    else LP.GateResult("RETAKE", ["no_plate"], NO_PLATE))
            capture = RP.capture_block({"mode": "naive", "device_mode": device_mode, "gate": gate.verdict,
                                        "gate_message": gate.message, "gated": False})
        lap("process")

        judged: list[dict[str, Any]] = []
        verdict = "RETAKE"
        view, view_corners = image, corners
        if capture["gate"] != "RETAKE":
            gm = grid_map(corners, meta.get("plate_size_mm") or part["size_mm"], self.grid_cfg,
                          meta.get("grid_plate_mm"), meta.get("corner_plate_idx"))
            grid_u8 = rectify(image, gm)
            lap("rectify")
            if proc is not None:
                from .surface3d import SurfaceSource
                surface = SurfaceSource(item.frames, proc.shifts, gm.H, gm.size, gm.mm_per_px,
                                        gm.grid.plate_corners_px(), meta.get("plate_size_mm") or part["size_mm"],
                                        meta.get("corner_plate_idx"), dict(item.meta or {}),
                                        {k: meta.get(k) for k in ("part_id", "scene_id", "source", "device_mode")})
            dets = self._detect(grid_u8, meta, lap)
            plate = gm.grid.plate_mask()
            jig = self.part.get("holes_mm") if meta.get("source") != "twin" else None   # app.yaml part (a jig)
            holes = holes_grid_mm({**meta, "holes_mm": meta.get("holes_mm") or jig}, gm)
            defects = [detection_record(d, grid_u8, gm, plate, self.measure, holes) for d in dets]
            lap("measure")
            if partner is not None:                                   # P1h: the inspection's second burst
                p_defects, p_gate = self._partner_defects(partner, part, meta)
                if p_defects is None:
                    capture["gate"] = "RETAKE"
                    capture["gate_message"] = f"Second burst refused: {p_gate.message}"
                    capture["gate_reasons"] = [*(capture.get("gate_reasons") or []), "second_burst"]
                else:
                    defects = rules.confirm_defects(defects, p_defects, self.spec)
                    capture["confirm"] = {"bursts": 2,
                                          "confirmed": sum(1 for d in defects if d.get("confirmed") is True),
                                          "unconfirmed": sum(1 for d in defects if d.get("confirmed") is False)}
                lap("confirm")
        if capture["gate"] != "RETAKE":
            verdict, judged = rules.judge_steel(defects, self.spec, part)
            verdict, policy_note = rules.apply_device_policy(verdict, device_mode, self.spec)
            if policy_note:
                capture["policy_note"] = policy_note
            lap("rules")
            view, view_corners = grid_u8, None
        record = RP.build_steel_record(
            meta.get("part_id") or f"part-{self.count + 1:05d}", part, judged, verdict, capture, tm,
            {"vision": self.vision_name, "advisor": None}, spec=self.spec,
            extra={"source": self._source(meta, part), "advice": None})
        if record["verdict"] != "RETAKE" and capture.get("policy_note"):
            record["verdict_instruction"] = f"{capture['policy_note']} {record['verdict_instruction']}"
        lap("report")
        jpeg = overlay_jpeg(view, record["defects"], record["verdict"], view_corners, self.image_px)
        lap("image")
        record["timing_ms"] = {k: round(v, 1) for k, v in tm.items()}
        record["timing_ms"]["total"] = round(tm["capture"] + (time.perf_counter() - t_start) * 1e3, 1)

        self.alerter(record)                                     # the alert: before the log and the model
        with self.lock:
            self.count += 1
            key = self.log.inspection(record, jpeg) if self.log is not None else self.count
            insp = Inspection(record, jpeg, meta, key,
                              surface=surface if record["verdict"] != "RETAKE" else None)
            self.last = insp
        self._notify("inspection", insp)
        if self.advisor is not None and self.advise != "off":
            if self.advise == "sync":
                self.explain(insp)
            else:
                self.submit(self._explain_latest, insp)
        return insp

    def _partner_defects(self, partner, part: Mapping[str, Any], meta: Mapping[str, Any]):
        """The second burst's measured defects, or (None, gate) when its own quality gate refuses it
        (then the whole inspection is a RETAKE: the pair is retaken, never judged on one burst)."""
        item, pmeta = partner
        pmeta = {**meta, **dict(pmeta or {})}
        corners = None if pmeta.get("corners_px") is None else np.asarray(pmeta["corners_px"], float).reshape(4, 2)
        if corners is None:
            corners = auto_corners(item)
        shape = next(iter(item.frames.values())).shape
        proc = LP.process_burst(item, None if corners is None else roi_mask(corners, shape), self.capture_cfg)
        gate = proc.gate
        if corners is None:
            gate = LP.GateResult("RETAKE", [*gate.reasons, "no_plate"], NO_PLATE, [*gate.details, "plate not found"])
        if gate.verdict == "RETAKE":
            return None, gate
        gm = grid_map(corners, pmeta.get("plate_size_mm") or part["size_mm"], self.grid_cfg,
                      pmeta.get("grid_plate_mm"), pmeta.get("corner_plate_idx"))
        grid_u8 = rectify(proc.image_u8, gm)
        dets = self._detect(grid_u8, pmeta, _Lap({}))
        holes = holes_grid_mm(pmeta, gm)
        return [detection_record(d, grid_u8, gm, gm.grid.plate_mask(), self.measure, holes) for d in dets], gate

    def _part(self, meta: Mapping[str, Any]) -> dict[str, Any]:
        p = {**{k: v for k, v in self.part.items() if k in ("material", "thickness_mm", "size_mm")},
             **{k: v for k, v in (meta.get("part") or {}).items() if v is not None}}
        if meta.get("plate_size_mm") is not None:
            p["size_mm"] = list(meta["plate_size_mm"])
        return rules.steel_part(self.spec, p)

    @staticmethod
    def _capture_extra(metrics: Mapping[str, Any], gate) -> dict[str, Any]:
        keep = ("led_level", "residual_sigma", "closure_sigma", "snr", "sat_frac_a", "masked_frac", "align_fallback")
        out = {k: (round(float(metrics[k]), 4) if isinstance(metrics[k], (int, float)) and not isinstance(metrics[k], bool)
                   else metrics[k]) for k in keep if k in metrics and metrics[k] is not None}
        out = {k: v for k, v in out.items() if not (isinstance(v, float) and not math.isfinite(v))}
        if getattr(gate, "details", None):
            out["gate_details"] = list(gate.details)
        return out

    def _detect(self, grid_u8: np.ndarray, meta: Mapping[str, Any], lap: _Lap) -> list[T.Det]:
        h, w = grid_u8.shape[:2]
        tiles = T.tile_grid(w, h, int(self.grid_cfg.get("tile", T.TILE)), int(self.grid_cfg.get("min_overlap", T.MIN_OVERLAP)))
        crops = T.cut(grid_u8, tiles)
        if hasattr(self.detector, "prime"):
            self.detector.prime(meta, tiles, (w, h))
        lap("tile")
        per_tile = self.detector.detect(crops)
        lap("detect")
        if len(per_tile) != len(tiles):
            raise RuntimeError(f"detector returned {len(per_tile)} results for {len(tiles)} tiles")
        names = npu.class_names(self.detector)
        if names != list(T.CLASSES):
            per_tile = [npu.remap_classes(ds, names) for ds in per_tile]
        dets = T.merge_tiles(per_tile, tiles, w, h)
        dets = [replace(d, x0=max(0.0, d.x0), y0=max(0.0, d.y0), x1=min(float(w), d.x1), y1=min(float(h), d.y1))
                for d in dets]
        lap("merge")
        return [d for d in dets if d.x1 > d.x0 and d.y1 > d.y0]

    def _source(self, meta: Mapping[str, Any], part: Mapping[str, Any]) -> dict[str, Any]:
        src = {"kind": meta.get("source"), "arm": meta.get("arm", "burst"),
               "file": Path(meta["path"]).name if meta.get("path") else None}
        for k in ("scene_id", "variant", "condition", "lux", "split"):
            if meta.get(k) is not None:
                src[k] = meta[k]
        if (meta.get("camera") or {}).get("warnings"):
            src["camera_warnings"] = list(meta["camera"]["warnings"])
        labels = [d for d in meta.get("labels") or [] if d.get("label", True) and d.get("visible") is not False
                  and (d.get("type") or d.get("cls")) in T.CLASSES and d.get("location_mm")]
        if meta.get("source") == "twin":
            gt = [{"type": d.get("type") or d.get("cls"), "confidence": 1.0, "box_px": d.get("box_px"),
                   "location_mm": d["location_mm"], "size": d.get("size") or {}, "near_edge_mm": d.get("near_edge_mm")}
                  for d in labels]
            v, _ = rules.judge_steel(gt, self.spec, part)
            src["truth"] = {"verdict": v, "defects": len(gt), "note": "rules on the twin's visible ground truth"}
        return src

    # -- advice (after the alert) --
    def submit(self, fn: Callable, *args: Any) -> None:
        """Queue an advisor job on the single advisor thread (one model run at a time)."""
        with self.lock:
            if self._q is None:
                self._q = queue.Queue()
                self._worker = threading.Thread(target=self._work, name="advisor", daemon=True)
                self._worker.start()
        self._q.put((fn, args))

    def _work(self) -> None:
        while True:
            job = self._q.get()
            try:
                if job is None:
                    return
                fn, args = job
                fn(*args)
            except Exception as e:   # the station keeps running whatever the advisor does
                print(f"advisor job failed: {e}", file=sys.stderr)
            finally:
                self._q.task_done()

    def wait_advice(self, timeout: float | None = None) -> bool:
        """Block until queued advisor jobs are done (True) or the timeout passes (False)."""
        if self._q is None:
            return True
        t_end = None if timeout is None else time.perf_counter() + timeout
        while self._q.unfinished_tasks:
            if t_end is not None and time.perf_counter() > t_end:
                return False
            time.sleep(0.01)
        return True

    def _stream(self, insp: Inspection | None):
        return lambda text: self._notify("advice_text", insp, text=text)

    def _explain_latest(self, insp: Inspection) -> AdviceResult | None:
        """The advisor thread's job: explain insp unless a newer part was inspected while it waited
        (Llama takes longer than one auto cycle on the Pi; the operator only sees the latest part,
        and without this the queue, and the delay of every answer, grows without bound)."""
        with self.lock:
            stale = self.last is not insp
        if stale:
            if self.log is not None:
                self.log.event({"event": "explain_skipped", "inspection_id": insp.key, "part_id": insp.record["part_id"],
                                "reason": "a newer part was inspected before the advisor was free"})
            return None
        return self.explain(insp)

    def explain(self, insp: Inspection) -> AdviceResult:
        self._notify("advice_start", insp, kind="explain", question=None)
        res = self.advisor.explain(insp.record, key=insp.key, on_text=self._stream(insp))
        self.record_advice(insp, res)
        return res

    def ask(self, insp: Inspection, question: str) -> AdviceResult:
        """A follow-up question ('why' and 'fix' are the UI buttons' canned questions)."""
        question = QUESTIONS.get(question, question)
        self._notify("advice_start", insp, kind="ask", question=question)
        res = self.advisor.ask(insp.record, question, key=insp.key, on_text=self._stream(insp))
        self.record_advice(insp, res)
        return res

    def shift(self, last_n: int = 50) -> AdviceResult:
        records = self.log.store.records(last_n) if self.log is not None else ([self.last.record] if self.last else [])
        self._notify("advice_start", None, kind="shift", question="shift summary")
        res = self.advisor.shift(records, last_n, on_text=self._stream(None))
        self.record_advice(None, res)
        return res

    def acknowledge(self, insp: Inspection) -> None:
        if self.log is not None:
            self.log.store.add_chat(insp.key, "operator", "ACK: alert acknowledged")
            self.log.event({"event": "ack", "inspection_id": insp.key, "part_id": insp.record["part_id"],
                            "verdict": insp.record["verdict"]})

    def record_advice(self, insp: Inspection | None, res: AdviceResult) -> None:
        stats = {k: v for k, v in res.stats.items() if k != "summary"}
        if insp is not None:
            with self.lock:
                insp.advice.append(res)
                if res.kind == "explain":
                    r = insp.record
                    r["advice"] = res.text
                    r["backend"]["advisor"] = res.backend
                    r["timing_ms"]["advisor"] = float(stats.get("call_ms") or 0.0)
                    r["advisor_stats"] = {"model": res.model, "problems": res.problems, **stats}
                snapshot = json.loads(json.dumps(insp.record))
        if self.log is not None:
            if insp is not None:
                if res.kind == "explain":
                    self.log.store.update_inspection(insp.key, snapshot)
                elif res.question:
                    self.log.store.add_chat(insp.key, "operator", res.question)
                self.log.store.add_chat(insp.key, "assistant", res.text)
            self.log.event({"event": res.kind, "inspection_id": None if insp is None else insp.key,
                            "part_id": None if insp is None else insp.record["part_id"], "question": res.question,
                            "answer": res.text, "backend": res.backend, "model": res.model, "problems": res.problems,
                            "stats": stats, "error": res.error})
        self._notify("advice_done", insp, result=res)

    def _notify(self, event: str, insp: Inspection | None, **kw: Any) -> None:
        for fn in list(self.listeners):
            try:
                fn(event, insp, **kw)
            except Exception as e:   # a UI problem never stops an inspection
                print(f"listener failed on {event}: {e}", file=sys.stderr)

    def close(self) -> None:
        if self._q is not None:
            self._q.put(None)
            if self._worker is not None:
                self._worker.join(timeout=5.0)
        close = getattr(self.detector, "close", None)
        if close:
            close()
        if self.log is not None:
            self.log.close()
