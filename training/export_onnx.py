"""Export a trained YOLOv8n to ONNX (for onnxruntime and the Hailo compile) and check it.

YOLO(best).export(format='onnx', imgsz=640, opset=11, simplify=True, dynamic=False,
nms=False) gives input (1, 3, 640, 640) float32 RGB / 255 and output (1, 4 + nc, 8400):
boxes (cx, cy, w, h) in input pixels plus sigmoid class scores.

Checks, on a few tiles (--check-tiles, else synthetic ones):
- raw: onnxruntime output (trinetra.npu.OnnxDetector) against the PyTorch model on the CPU;
- detections: OnnxDetector against UltralyticsDetector (same conf and IoU);
- graph: the six head convolutions the Hailo Model Zoo yolov8n.yaml cuts the graph at
  (/model.22/cv2.i/cv2.i.2/Conv and /model.22/cv3.i/cv3.i.2/Conv).

Writes <out>/<name>.onnx and <out>/<name>.json (classes, imgsz, grid mm/px, versions,
check results). trinetra.npu reads "classes" from that JSON (copy it next to the HEF as
<hef stem>.json). Exit code 1 if the raw check fails; a warning if the model's classes
are not trinetra.tiles.CLASSES (6 steel classes: scratch, dent, pit, corrosion, crack, stain).

Example (training venv):
  .venv-train/bin/python training/export_onnx.py --weights runs/train/real_only/weights/best.pt \\
      --name v8n_real_only --check-tiles data/real/yolo/images/val
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from trinetra.npu import OnnxDetector, UltralyticsDetector, class_names
from trinetra.tiles import CLASSES, TILE, Det

ROOT = Path(__file__).resolve().parents[1]
HAILO_END_NODES = [f"/model.22/cv{k}.{i}/cv{k}.{i}.2/Conv" for i in range(3) for k in (2, 3)]
IMG_EXT = (".png", ".jpg", ".jpeg")


def check_tiles(src: Path | None, n: int, size: int = TILE, seed: int = 0) -> list[np.ndarray]:
    """n uint8 RGB tiles: centre crops of images in src, else synthetic plates with marks."""
    tiles = []
    if src is not None:
        for p in sorted(q for q in Path(src).iterdir() if q.suffix.lower() in IMG_EXT)[:n]:
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            if min(im.shape[:2]) < size:
                continue
            y0, x0 = (im.shape[0] - size) // 2, (im.shape[1] - size) // 2
            tiles.append(np.ascontiguousarray(im[y0 : y0 + size, x0 : x0 + size]))
    rng = np.random.default_rng(seed)
    while len(tiles) < n:
        t = np.full((size, size, 3), 150, np.uint8) + rng.integers(0, 20, (size, size, 3), dtype=np.uint8)
        for _ in range(4):
            c = tuple(int(v) for v in rng.integers(40, size - 40, 2))
            cv2.circle(t, c, int(rng.integers(8, 30)), (60, 60, 60), -1)
        tiles.append(t)
    return tiles


def torch_raw(weights: Path, tiles: list[np.ndarray]) -> np.ndarray:
    """PyTorch model output (B, 4 + nc, N) on the CPU in float32, same preprocessing as ONNX."""
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights)).model.float().eval().cpu()
    x = torch.from_numpy(OnnxDetector.preprocess(tiles))
    with torch.no_grad():
        y = model(x)
    y = y[0] if isinstance(y, (list, tuple)) else y
    return y.numpy()


def det_iou(a: Det, b: Det) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    union = a.area + b.area - ix * iy
    return ix * iy / union if union > 0 else 0.0


def compare_dets(a: list[list[Det]], b: list[list[Det]], iou: float = 0.9) -> dict[str, Any]:
    """Greedy same-class matching of two detectors' outputs, tile by tile."""
    matched = only_a = only_b = 0
    dconf = 0.0
    for da, db in zip(a, b):
        left = list(db)
        for d in da:
            best = max((e for e in left if e.cls == d.cls), key=lambda e: det_iou(d, e), default=None)
            if best is not None and det_iou(d, best) >= iou:
                matched += 1
                dconf = max(dconf, abs(d.conf - best.conf))
                left.remove(best)
            else:
                only_a += 1
        only_b += len(left)
    return {"matched": matched, "only_onnx": only_a, "only_torch": only_b, "max_conf_diff": round(dconf, 5)}


def graph_nodes(onnx_path: Path) -> dict[str, Any]:
    import onnx

    m = onnx.load(str(onnx_path))
    names = {n.name for n in m.graph.node}
    inp = m.graph.input[0]
    dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    return {"opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
            "input": {"name": inp.name, "shape": dims},
            "hailo_end_nodes_missing": [n for n in HAILO_END_NODES if n not in names]}


def grid_config() -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
    return dict(cfg["grid"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(weights: Path, out: Path, name: str, opset: int = 11) -> Path:
    from ultralytics import YOLO

    path = Path(YOLO(str(weights)).export(format="onnx", imgsz=TILE, opset=opset, simplify=True,
                                          dynamic=False, nms=False, batch=1, device="cpu"))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{name}.onnx"
    if path.resolve() != dst.resolve():
        shutil.move(str(path), dst)
    return dst


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path, help="best.pt")
    ap.add_argument("--name", required=True, help="output base name")
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--check-tiles", type=Path, help="directory of real tiles (default: synthetic tiles)")
    ap.add_argument("--n-check", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tol-score", type=float, default=2e-3, help="max |score| difference allowed")
    ap.add_argument("--tol-box-px", type=float, default=0.25, help="max box difference on the 100 best anchors")
    a = ap.parse_args(argv)

    import onnx
    import onnxruntime
    import onnxslim
    import torch
    import ultralytics

    onnx_path = export(a.weights, a.out, a.name, a.opset)
    tiles = check_tiles(a.check_tiles, a.n_check)
    ort_det = OnnxDetector(onnx_path, conf=a.conf, iou=a.iou)
    ro, rt = ort_det.raw(tiles), torch_raw(a.weights, tiles)
    score_diff = float(np.abs(ro[:, 4:] - rt[:, 4:]).max())
    top = np.argsort(-rt[:, 4:].max(axis=1), axis=1)[:, :100][:, None, :]     # 100 strongest anchors per tile
    box_diff = float(np.abs(np.take_along_axis(ro[:, :4], top, 2) - np.take_along_axis(rt[:, :4], top, 2)).max())
    raw_ok = score_diff <= a.tol_score and box_diff <= a.tol_box_px
    dets = compare_dets(ort_det.detect(tiles), UltralyticsDetector(a.weights, conf=a.conf, iou=a.iou,
                                                                   device="cpu").detect(tiles))
    graph = graph_nodes(onnx_path)
    names = class_names(ort_det)
    meta = {
        "name": a.name, "file": onnx_path.name, "sha256": sha256(onnx_path), "source_weights": str(a.weights),
        "classes": names, "n_classes": len(names), "classes_match_tiles": names == list(CLASSES),
        "imgsz": TILE, "grid": grid_config(),
        "input": {"layout": "NCHW", "dtype": "float32", "channels": "RGB", "scale": "1/255",
                  "shape": graph["input"]["shape"]},
        "output": "(1, 4 + nc, N): cx, cy, w, h in input px, then sigmoid class scores; no NMS",
        "export": {"opset": graph["opset"], "simplify": True, "dynamic": False, "nms": False,
                   "ultralytics": ultralytics.__version__, "torch": torch.__version__, "onnx": onnx.__version__,
                   "onnxruntime": onnxruntime.__version__, "onnxslim": onnxslim.__version__},
        "check": {"tiles": len(tiles), "source": str(a.check_tiles) if a.check_tiles else "synthetic",
                  "max_score": float(f"{rt[:, 4:].max():.4g}"), "raw_max_score_diff": float(f"{score_diff:.3g}"),
                  "raw_max_box_diff_px_top100": float(f"{box_diff:.3g}"), "raw_ok": raw_ok, "detections": dets},
        "hailo_end_nodes_missing": graph["hailo_end_nodes_missing"],
    }
    (a.out / f"{a.name}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["check"], indent=1))
    if not meta["classes_match_tiles"]:
        print(f"WARNING: model classes {names} != trinetra.tiles.CLASSES {list(CLASSES)}; "
              "detect_grid remaps by name and drops unknown classes")
    if graph["hailo_end_nodes_missing"]:
        print("WARNING: Hailo yolov8n end nodes missing:", graph["hailo_end_nodes_missing"])
    if not raw_ok:
        print("FAIL: onnxruntime and PyTorch outputs differ beyond tolerance")
        sys.exit(1)
    return meta


if __name__ == "__main__":
    main()
