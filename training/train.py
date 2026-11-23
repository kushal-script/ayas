"""Train YOLOv8n on native-scale 640 px tiles with Ultralytics, using the Ayas settings.

Tiles are already 640 px cut from the 0.08 mm/px grid, so imgsz=640 means no resizing,
and scale=0.1 keeps mm/px nearly fixed. Colour jitter follows the lighting research
(docs/research/2026-11-14_lighting_invariance.md, training recipe): hsv_h=0.01 (keeps
rust and stain hues), hsv_s=0.4, hsv_v=0.4, as a last line of defence after the capture
method. Tiles are plain RGB (the burst's min(L, R) image or the naive ISP frame), so flips
are safe. Fine-tuning (--weights) defaults to SGD, lr0=0.002 and a frozen backbone
(freeze=10); with Ultralytics' optimizer=auto the lr0 would be silently replaced, so the
optimizer is set explicitly. Classes come from the data.yaml (ayas.tiles.CLASSES).

--test runs Ultralytics val on the data.yaml test split (the real test tiles) with
best.pt and stores tile-level mAP50 in <run>/ayas_train.json.

Examples (training venv):
  .venv-train/bin/python training/train.py --data runs/datasets/real_only/data.yaml --name real_only --test
  .venv-train/bin/python training/train.py --data runs/datasets/synth_then_real/stage2/data.yaml \\
      --weights runs/train/synth_then_real_s1/weights/best.pt --name synth_then_real --test
  .venv-train/bin/python training/train.py --data <yaml> --name smoke --model yolov8n.yaml --epochs 1 --no-plots
Runs on Apple MPS, CUDA (Colab, JarvisLabs) or CPU (--device auto picks).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Native-scale augmentation: keep mm/px; mild hue, research colour/brightness jitter; flips are fine.
TRAIN_SETTINGS = dict(imgsz=640, hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, scale=0.1, translate=0.1, degrees=0.0,
                      shear=0.0, perspective=0.0, fliplr=0.5, flipud=0.5, mosaic=1.0, mixup=0.0,
                      close_mosaic=10, deterministic=True)
FINETUNE = dict(optimizer="SGD", lr0=0.002, freeze=10)


def pick_device(device: str = "auto") -> str:
    """auto -> CUDA 0, else Apple mps, else cpu."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def versions() -> dict[str, str]:
    import torch
    import ultralytics

    return {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "python": platform.python_version()}


def train_kwargs(a: argparse.Namespace, device: str) -> dict[str, Any]:
    kw = dict(TRAIN_SETTINGS, data=str(Path(a.data).resolve()), epochs=a.epochs, patience=a.patience,
              batch=a.batch, seed=a.seed, device=device, project=str(Path(a.project).resolve()), name=a.name,
              exist_ok=a.exist_ok, plots=a.plots, workers=a.workers, cache=a.cache or False, fraction=a.fraction)
    if a.weights:
        kw.update(FINETUNE)
    for k in ("optimizer", "lr0", "freeze"):
        v = getattr(a, k)
        if v is not None:
            kw[k] = v
    return kw


def test_metrics(best: Path, data: str, device: str, batch: int, save_dir: Path) -> dict[str, Any] | None:
    """Ultralytics val on the test split (tile-level metrics), or None if data.yaml has no test."""
    from ultralytics import YOLO

    if not yaml.safe_load(Path(data).read_text()).get("test"):
        print("no test split in", data)
        return None
    m = YOLO(str(best)).val(data=data, split="test", imgsz=TRAIN_SETTINGS["imgsz"], batch=batch, device=device,
                            plots=False, verbose=False, project=str(save_dir), name="test", exist_ok=True)
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    box = m.box
    return {"map50": float(box.map50), "map50_95": float(box.map), "precision": float(box.mp),
            "recall": float(box.mr),
            "per_class_ap50": {names[int(c)]: float(box.ap50[i]) for i, c in enumerate(box.ap_class_index)}}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Ultralytics data.yaml (from build_datasets.py)")
    ap.add_argument("--name", required=True, help="run name: <project>/<name>")
    ap.add_argument("--model", default="yolov8n.pt", help="start point: yolov8n.pt (COCO) or yolov8n.yaml (scratch)")
    ap.add_argument("--weights", help="fine-tune from this checkpoint (SGD, lr0=0.002, freeze=10)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | mps | cpu | 0")
    ap.add_argument("--project", default=str(ROOT / "runs/train"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the train list to use")
    ap.add_argument("--cache", choices=["ram"], help="cache images in RAM (never on disk: it is tight)")
    ap.add_argument("--optimizer", help="override (default: Ultralytics auto, or SGD when fine-tuning)")
    ap.add_argument("--lr0", type=float, help="override (only honoured with an explicit optimizer)")
    ap.add_argument("--freeze", type=int, help="override the number of frozen layers")
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.add_argument("--exist-ok", action="store_true", help="reuse <project>/<name> instead of name2, name3...")
    ap.add_argument("--test", action="store_true", help="after training, val best.pt on the test split")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    device = pick_device(a.device)
    kw = train_kwargs(a, device)
    t0 = time.time()
    model = YOLO(a.weights or a.model)
    model.train(**kw)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    summary: dict[str, Any] = {
        "name": a.name, "data": kw["data"], "start": a.weights or a.model, "device": device,
        "settings": {k: v for k, v in kw.items() if k not in ("data", "project", "name")},
        "epochs": a.epochs, "train_seconds": round(time.time() - t0, 1), "save_dir": str(save_dir),
        "best": str(best), "versions": versions(),
    }
    if a.test:
        summary["test"] = test_metrics(best, kw["data"], device, a.batch, save_dir)
    (save_dir / "ayas_train.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("best", "train_seconds", "device")}))
    return summary


if __name__ == "__main__":
    main()
