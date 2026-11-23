"""Tile detectors for the surface-defect YOLOv8n, and detection on a whole rectified grid.

Every detector takes 640 x 640 uint8 RGB tiles cut from the rectified plate grid
(ayas.tiles) and returns tile-local boxes in pixels. Tiles are never resized: the
model must see defects at the grid scale (0.08 mm/px).

- OnnxDetector: onnxruntime on the CPU (Mac, Pi fallback). Ultralytics YOLOv8 ONNX,
  exported with nms=False: input (1, 3, 640, 640) float32 RGB / 255, output
  (1, 4 + nc, N) with boxes as (cx, cy, w, h) in input pixels and sigmoid class scores.
- HailoDetector: a Model Zoo yolov8n HEF on the Hailo-10H with on-chip NMS
  (HailoRT 5.x InferModel API). Verify it on the Pi with the stock HEF first.
- UltralyticsDetector: the .pt checkpoint, for development and export checks.

Nothing here is tied to one class list. Each detector exposes `names` (class names by
id): from the ONNX metadata, the model JSON written by training/export_onnx.py next to
the model (<model>.json, "classes"), the checkpoint, or else ayas.tiles.CLASSES.
detect_grid maps a model whose names differ from tiles.CLASSES onto tiles.CLASSES ids
before the seam merge (which needs to know which classes are elongated).

Heavy runtimes (onnxruntime, hailo_platform, ultralytics/torch) are imported lazily, so
this module imports anywhere.
"""

from __future__ import annotations

import ast
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .tiles import CLASSES, MIN_OVERLAP, TILE, Det, cut, merge_tiles, tile_grid


class Detector(Protocol):
    def detect(self, tiles: list[np.ndarray]) -> list[list[Det]]:
        """uint8 (H, W, 3) RGB tiles -> per tile, detections in tile-local pixels."""
        ...


def check_tiles(tiles: Sequence[np.ndarray], size: int) -> None:
    for i, t in enumerate(tiles):
        if t.dtype != np.uint8 or t.shape != (size, size, 3):
            raise ValueError(f"tile {i}: expected uint8 ({size}, {size}, 3) RGB, got {t.dtype} {t.shape}; "
                             "cut tiles with ayas.tiles, never resize")


# --- NMS and decoding (pure numpy) -------------------------------------------------------

def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS on (N, 4) xyxy boxes; returns the kept indices, best first.

    A box is suppressed when its IoU with a better kept box is > iou (as torchvision).
    """
    order = np.argsort(-scores, kind="stable")
    x0, y0, x1, y1 = boxes.T
    areas = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        ix = np.clip(np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]), 0, None)
        iy = np.clip(np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]), 0, None)
        inter = ix * iy
        union = areas[i] + areas[rest] - inter
        ov = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[ov <= iou]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float) -> np.ndarray:
    """Class-wise NMS: boxes of different classes never suppress each other."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    span = float(boxes.max() - boxes.min()) + 1.0          # shift each class clear of the others
    offset = classes.astype(np.float64)[:, None] * span
    return nms_indices(boxes.astype(np.float64) + offset, scores, iou)


def decode_yolov8(pred: np.ndarray, conf: float = 0.25, iou: float = 0.5, max_det: int = 300,
                  size: int = TILE) -> list[Det]:
    """One image of Ultralytics YOLOv8 raw output (4 + nc, N) -> detections in input pixels.

    Mirrors Ultralytics' default NMS: one label per box (the best class), keep scores
    > conf, class-wise NMS, at most max_det boxes, then clip to the image.
    """
    pred = np.asarray(pred, np.float32)
    if pred.ndim != 2 or pred.shape[0] < 5:
        raise ValueError(f"expected (4 + nc, N) YOLOv8 output, got {pred.shape}")
    scores = pred[4:]
    cls = scores.argmax(axis=0)
    score = scores[cls, np.arange(pred.shape[1])]
    m = score > conf
    cx, cy, w, h = pred[:4, m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = batched_nms(boxes, score[m], cls[m], iou)[:max_det]
    boxes = np.clip(boxes[keep], 0, size)
    return [Det(int(c), float(s), *map(float, b)) for b, s, c in zip(boxes, score[m][keep], cls[m][keep])]


def parse_nms_by_class(raw: Any, n_classes: int, conf: float = 0.25, size: int = TILE) -> list[Det]:
    """Hailo on-chip NMS output (HAILO_NMS_BY_CLASS, float32) of one frame -> detections in pixels.

    Accepts the two forms HailoRT returns:
    - a list of n_classes arrays, each (k, 5) of [y_min, x_min, y_max, x_max, score]
      (what hailo_platform's get_buffer() returns for an NMS output);
    - the raw flat float32 buffer: per class, a box count followed by that many
      5-float boxes, packed class after class (trailing padding is ignored).
    Coordinates are normalised 0..1 to the model input; size scales them to pixels.
    """
    if isinstance(raw, (list, tuple)):
        per_class = [np.asarray(a, np.float32).reshape(-1, 5) for a in raw]
        if len(per_class) != n_classes:
            raise ValueError(f"NMS output has {len(per_class)} classes, expected {n_classes}")
    else:
        flat = np.asarray(raw, np.float32).ravel()
        per_class, i = [], 0
        for c in range(n_classes):
            if i >= flat.size:
                raise ValueError(f"NMS buffer ends before class {c}")
            k = int(flat[i])
            if k < 0 or i + 1 + 5 * k > flat.size:
                raise ValueError(f"bad box count {k} for class {c} at offset {i}")
            per_class.append(flat[i + 1 : i + 1 + 5 * k].reshape(k, 5))
            i += 1 + 5 * k
    dets = []
    for c, boxes in enumerate(per_class):
        for y0, x0, y1, x1, s in boxes:
            if s > conf:
                b = np.clip(np.array([x0, y0, x1, y1], np.float64), 0.0, 1.0) * size
                dets.append(Det(c, float(s), *map(float, b)))
    return sorted(dets, key=lambda d: -d.conf)


# --- Class names -------------------------------------------------------------------------

def model_meta(path: str | Path) -> dict[str, Any] | None:
    """The model JSON next to a model file (<stem>.json from training/export_onnx.py), if any."""
    p = Path(path).with_suffix(".json")
    try:
        meta = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _as_list(names: Any) -> list[str] | None:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    if isinstance(names, (list, tuple)) and names:
        return [str(n) for n in names]
    return None


def class_names(detector: Any) -> list[str]:
    """A detector's class names by id (tiles.CLASSES when it does not say)."""
    return _as_list(getattr(detector, "names", None)) or list(CLASSES)


def remap_classes(dets: list[Det], names: Sequence[str]) -> list[Det]:
    """Detections with model class ids -> tiles.CLASSES ids, matched by name; others dropped."""
    if list(names) == list(CLASSES):
        return dets
    ids = {i: CLASSES.index(n) for i, n in enumerate(names) if n in CLASSES}
    return [replace(d, cls=ids[d.cls]) for d in dets if d.cls in ids]


# --- Detectors ---------------------------------------------------------------------------

def _onnx_names(session) -> list[str] | None:
    """Class names from Ultralytics ONNX metadata ("names": "{0: 'scratch', ...}"), if present."""
    try:
        meta = session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"])
    except (AttributeError, KeyError, ValueError, SyntaxError):
        return None
    return _as_list(names)


class OnnxDetector:
    """Ultralytics YOLOv8 ONNX (nms=False) on onnxruntime; NMS in numpy."""

    def __init__(self, path: str | Path, conf: float = 0.25, iou: float = 0.5, max_det: int = 300,
                 providers: list[str] | None = None):
        import onnxruntime as ort

        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = list(inp.shape)
        self.size = shape[2] if isinstance(shape[2], int) else TILE
        self.batch = shape[0] if isinstance(shape[0], int) else None    # None: dynamic batch
        self.names = _onnx_names(self.session) or _as_list((model_meta(path) or {}).get("classes"))

    @staticmethod
    def preprocess(tiles: Sequence[np.ndarray]) -> np.ndarray:
        """uint8 (H, W, 3) RGB tiles -> (B, 3, H, W) float32 in 0..1."""
        x = np.stack(tiles).transpose(0, 3, 1, 2).astype(np.float32)
        return np.ascontiguousarray(x / 255.0)

    def raw(self, tiles: Sequence[np.ndarray]) -> np.ndarray:
        """Raw model output (B, 4 + nc, N) for the tiles."""
        check_tiles(tiles, self.size)
        step = self.batch or max(1, len(tiles))
        outs = []
        for i in range(0, len(tiles), step):
            chunk = list(tiles[i : i + step])
            n = len(chunk)
            if self.batch and n < self.batch:      # a fixed-batch model needs a full batch
                chunk += [np.zeros_like(chunk[0])] * (self.batch - n)
            y = self.session.run(None, {self.input_name: self.preprocess(chunk)})[0]
            nc = len(self.names) if self.names else None
            if nc and y.shape[1] != 4 + nc and y.shape[2] == 4 + nc:   # (B, N, 4 + nc) layout
                y = y.transpose(0, 2, 1)
            elif nc is None and y.shape[1] > y.shape[2]:               # unnamed: anchors outnumber rows
                y = y.transpose(0, 2, 1)
            outs.append(y[:n])
        return np.concatenate(outs) if outs else np.zeros((0, 4 + len(class_names(self)), 0), np.float32)

    def detect(self, tiles: list[np.ndarray]) -> list[list[Det]]:
        if not tiles:
            return []
        return [decode_yolov8(p, self.conf, self.iou, self.max_det, self.size) for p in self.raw(tiles)]


class UltralyticsDetector:
    """A .pt checkpoint through Ultralytics (training venv only; for development and checks)."""

    def __init__(self, weights: str | Path, conf: float = 0.25, iou: float = 0.5, max_det: int = 300,
                 device: str | None = None, size: int = TILE):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf, self.iou, self.max_det, self.device, self.size = conf, iou, max_det, device, size
        self.names = _as_list(getattr(self.model, "names", None))

    def detect(self, tiles: list[np.ndarray]) -> list[list[Det]]:
        if not tiles:
            return []
        check_tiles(tiles, self.size)
        # Ultralytics reads numpy images as BGR (OpenCV order).
        results = self.model.predict([np.ascontiguousarray(t[..., ::-1]) for t in tiles], imgsz=self.size,
                                     conf=self.conf, iou=self.iou, max_det=self.max_det, device=self.device,
                                     verbose=False)
        out = []
        for r in results:
            b = r.boxes
            xyxy, cls, cf = (b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy())
            out.append([Det(int(c), float(s), *map(float, x)) for x, c, s in zip(xyxy, cls, cf)])
        return out


class HailoDetector:
    """A Model Zoo yolov8n HEF with on-chip NMS on the Hailo-10H (HailoRT 5.x Python API).

    API assumptions (HailoRT 5.1.1 from the apt package hailo-h10-all; the same calls as
    picamera2.devices.Hailo and Hailo's InferModel examples). Check them on the Pi with
    the STOCK yolov8n HEF before trusting a custom one (plan go/no-go 1):
      params = VDevice.create_params(); params.scheduling_algorithm = ROUND_ROBIN
      model = VDevice(params).create_infer_model(hef); model.set_batch_size(b)
      model.input().set_format_type(UINT8); model.output().set_format_type(FLOAT32)
      configured = model.configure().__enter__()
      bindings = configured.create_bindings(); bindings.input().set_buffer(uint8 HxWx3)
      bindings.output().set_buffer(np.empty(model.output().shape, float32))
      configured.run([bindings, ...], timeout_ms); bindings.output().get_buffer()
    The output is HAILO_NMS_BY_CLASS: per class, boxes [y_min, x_min, y_max, x_max, score]
    normalised 0..1 (see parse_nms_by_class). The HEF's own NMS score/IoU thresholds
    (Model Zoo nms config) apply first; conf here only filters further. Short batches
    are padded with black tiles so the scheduler never waits for a full batch.

    Classes: `names` if given, else "classes" of the model JSON next to the HEF
    (<hef stem>.json: copy the export_onnx.py JSON there), else tiles.CLASSES. An explicit
    n_classes (e.g. 80 for the stock COCO HEF) wins; names are then kept only if they match.
    """

    def __init__(self, hef_path: str | Path, conf: float = 0.25, batch_size: int = 9,
                 n_classes: int | None = None, timeout_ms: int = 10000, names: Sequence[str] | None = None):
        try:
            from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
        except ImportError as e:
            raise ImportError("hailo_platform not found. On the Pi install apt hailo-h10-all (HailoRT 5.1.1) "
                              "and create the venv with --system-site-packages") from e
        known = _as_list(names) or _as_list((model_meta(hef_path) or {}).get("classes")) or list(CLASSES)
        if n_classes is None:
            n_classes = len(known)
        self.names = known if len(known) == n_classes else None
        self.conf, self.batch_size, self.n_classes, self.timeout_ms = conf, batch_size, int(n_classes), timeout_ms
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)
        self._model = self._vdevice.create_infer_model(str(hef_path))
        self._model.set_batch_size(batch_size)
        self._model.input().set_format_type(FormatType.UINT8)
        self._model.output().set_format_type(FormatType.FLOAT32)
        self.input_shape = tuple(self._model.input().shape)
        self.output_shape = tuple(self._model.output().shape)
        if len(self.input_shape) != 3 or self.input_shape[0] != self.input_shape[1] or self.input_shape[2] != 3:
            raise ValueError(f"expected a square HxWx3 HEF input, got {self.input_shape}")
        self.size = int(self.input_shape[0])
        self._ctx = self._model.configure()
        self._configured = self._ctx.__enter__()

    def _run_batch(self, tiles: list[np.ndarray]) -> list[Any]:
        bindings = []
        for t in tiles:
            b = self._configured.create_bindings()
            b.input().set_buffer(np.ascontiguousarray(t))
            b.output().set_buffer(np.empty(self.output_shape, dtype=np.float32))
            bindings.append(b)
        self._configured.run(bindings, self.timeout_ms)
        return [b.output().get_buffer() for b in bindings]

    def detect(self, tiles: list[np.ndarray]) -> list[list[Det]]:
        check_tiles(tiles, self.size)
        out: list[list[Det]] = []
        for i in range(0, len(tiles), self.batch_size):
            chunk = list(tiles[i : i + self.batch_size])
            n = len(chunk)
            chunk += [np.zeros_like(chunk[0])] * (self.batch_size - n)
            raws = self._run_batch(chunk)[:n]
            out += [parse_nms_by_class(r, self.n_classes, self.conf, self.size) for r in raws]
        return out

    def close(self) -> None:
        if getattr(self, "_ctx", None) is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        release = getattr(getattr(self, "_vdevice", None), "release", None)
        if release:
            release()
        self._vdevice = None

    def __enter__(self) -> "HailoDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_detector(path: str | Path, conf: float = 0.25, **kw: Any) -> Detector:
    """Pick the backend from the file type: .onnx, .hef or .pt."""
    suffix = Path(path).suffix.lower()
    if suffix == ".onnx":
        return OnnxDetector(path, conf=conf, **kw)
    if suffix == ".hef":
        return HailoDetector(path, conf=conf, **kw)
    if suffix == ".pt":
        return UltralyticsDetector(path, conf=conf, **kw)
    raise ValueError(f"unknown detector file type: {path}")


# --- Whole grid --------------------------------------------------------------------------

def detect_grid(grid_rgb: np.ndarray, detector: Detector, tile: int = TILE, min_overlap: int = MIN_OVERLAP,
                iou: float = 0.5) -> list[Det]:
    """Rectified uint8 RGB grid -> merged detections in grid pixels (seam duplicates joined).

    Class ids of the result are tiles.CLASSES ids (a model with other names is remapped by
    name; classes tiles.CLASSES does not have are dropped with a warning).
    """
    h, w = grid_rgb.shape[:2]
    tiles = tile_grid(w, h, tile, min_overlap)
    per_tile = detector.detect(cut(grid_rgb, tiles))
    if len(per_tile) != len(tiles):
        raise RuntimeError(f"detector returned {len(per_tile)} results for {len(tiles)} tiles")
    names = class_names(detector)
    if names != list(CLASSES):
        extra = sorted(set(names) - set(CLASSES))
        if extra:
            warnings.warn(f"model classes {extra} are not in tiles.CLASSES; their detections are dropped")
        per_tile = [remap_classes(ds, names) for ds in per_tile]
    dets = merge_tiles(per_tile, tiles, w, h, iou)
    # boxes reaching into the padding of a grid smaller than a tile are clipped back
    return [replace(d, x0=max(0.0, d.x0), y0=max(0.0, d.y0), x1=min(float(w), d.x1), y1=min(float(h), d.y1))
            for d in dets]
