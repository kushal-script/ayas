"""The 5-row sim-to-real ablation: train -> export best -> evaluate on the real test plates.

Rows (see build_datasets.py): real_only, real_25, synth_only, synth_then_real, mixed.
For each row: train.py (with --test: Ultralytics tile mAP50 on the real test tiles),
export_onnx.py (models/ablation_<row>.onnx), and eval_plates.py on the full test grids
(per-class recall with Wilson 95 % intervals, precision, FP per clean plate). Then it
writes reports/ablation.md. Every step is a subprocess of this Python, so --dry-run
just prints the commands.

Honesty rules: the test plate (P5) is never trained, tuned or calibrated on; epochs,
thresholds and twin settings are fixed before looking at the test numbers; every row
uses the same seed, epochs and test set.

Examples (training venv):
  .venv-train/bin/python training/ablation.py --dry-run
  .venv-train/bin/python training/ablation.py --epochs 150 --test-grids data/real/out/grids/test
  .venv-train/bin/python training/ablation.py --quick --rows real_only synth_only
  .venv-train/bin/python training/ablation.py --report-only
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ayas.tiles import CLASSES

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ROWS = ("real_only", "real_25", "synth_only", "synth_then_real", "mixed")
DATA_DESC = {"real_only": "real", "real_25": "25 % real", "synth_only": "synth",
             "synth_then_real": "synth -> real", "mixed": "synth + real"}
CLASS_COLS = tuple(CLASSES)      # one recall column per detector class


def paths(a: argparse.Namespace) -> dict[str, Path]:
    return {"datasets": a.out / "datasets", "runs": a.out / "train", "models": a.models,
            "reports": a.reports / "ablation"}


def commands(a: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """(step name, argv) for the whole ablation, in order."""
    py, p = sys.executable, paths(a)

    def common(epochs: int) -> list[str]:
        c = ["--epochs", str(epochs), "--batch", str(a.batch), "--seed", str(a.seed), "--device", a.device,
             "--project", str(p["runs"]), "--workers", str(a.workers), "--exist-ok"]
        return c + (["--no-plots", "--patience", "1"] if a.quick else [])

    steps = [("datasets", [py, str(HERE / "build_datasets.py"), "--synth", str(a.synth), "--real", str(a.real),
                           "--out", str(p["datasets"]), "--seed", str(a.seed), "--real-frac", str(a.real_frac),
                           "--real-oversample", str(a.real_oversample), "--rows", *a.rows])]
    for row in a.rows:
        ds = p["datasets"] / row
        if row == "synth_then_real":
            s1 = "synth_then_real_s1"
            steps.append((f"{row}:train1", [py, str(HERE / "train.py"), "--data", str(ds / "stage1/data.yaml"),
                                            "--name", s1, "--model", a.model, *common(a.epochs)]))
            steps.append((f"{row}:train2", [py, str(HERE / "train.py"), "--data", str(ds / "stage2/data.yaml"),
                                            "--name", row, "--weights", str(p["runs"] / s1 / "weights/best.pt"),
                                            *common(a.epochs_ft or a.epochs), "--test"]))
        else:
            steps.append((f"{row}:train", [py, str(HERE / "train.py"), "--data", str(ds / "data.yaml"),
                                           "--name", row, "--model", a.model, *common(a.epochs), "--test"]))
        exp = [py, str(HERE / "export_onnx.py"), "--weights", str(p["runs"] / row / "weights/best.pt"),
               "--name", f"ablation_{row}", "--out", str(p["models"])]
        if a.check_tiles:
            exp += ["--check-tiles", str(a.check_tiles)]
        steps.append((f"{row}:export", exp))
        if a.test_grids:
            ev = [py, str(HERE / "eval_plates.py"), "--model", str(p["models"] / f"ablation_{row}.onnx"),
                  "--grids", str(a.test_grids), "--out", str(p["reports"] / row), "--conf", str(a.conf)]
            if a.test_sidecars:
                ev += ["--sidecars", str(a.test_sidecars)]
            steps.append((f"{row}:eval", ev))
    return steps


def has_data(a: argparse.Namespace, row: str) -> bool:
    ds = paths(a)["datasets"] / row
    if row == "synth_then_real":
        return (ds / "stage1/data.yaml").exists() and (ds / "stage2/data.yaml").exists()
    return (ds / "data.yaml").exists()


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _cell(r: dict[str, Any] | None) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{r['k']}/{r['n']} {r['value']:.2f} [{r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}]"


def report(a: argparse.Namespace) -> str:
    """reports/ablation.md from the saved manifests, train summaries and plate evaluations."""
    p = paths(a)
    head = ["Row", "Train data (tiles)", "Epochs", *[f"{c} recall" for c in CLASS_COLS],
            "Defect precision", "FP / clean plate", "Tile mAP50 (real test)"]
    lines = ["# Sim-to-real ablation (YOLOv8n, 640 px native-scale tiles)", "",
             "Recall is per physical object on the real test plates (found in at least half of its captures), "
             "as k/n value [Wilson 95 % interval]. Precision covers the defect classes. The test plate (P5) "
             "was never used for training, tuning or calibration.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in a.rows:
        man = _load(p["datasets"] / row / "manifest.json") or _load(p["datasets"] / row / "stage2/manifest.json")
        if row == "synth_then_real":
            m1 = _load(p["datasets"] / row / "stage1/manifest.json")
            m2 = _load(p["datasets"] / row / "stage2/manifest.json")
            data = (f"{DATA_DESC[row]} ({m1['counts']['train']} -> {m2['counts']['train']})"
                    if m1 and m2 else DATA_DESC[row])
        else:
            data = f"{DATA_DESC[row]} ({man['counts']['train']})" if man else DATA_DESC[row]
        tr = _load(p["runs"] / row / "ayas_train.json")
        tr1 = _load(p["runs"] / "synth_then_real_s1" / "ayas_train.json") if row == "synth_then_real" else None
        epochs = (f"{tr1['epochs']} + {tr['epochs']}" if tr1 and tr else str(tr["epochs"]) if tr else "n/a")
        ev = _load(p["reports"] / row / "eval.json")
        pc = ev["per_class"] if ev else {}
        recalls = [_cell(pc.get(c, {}).get("recall_physical")) for c in CLASS_COLS]
        prec = _cell(pc.get("all_defects", {}).get("precision"))
        fpc = ev.get("fp_per_clean_plate") if ev else None
        test = (tr or {}).get("test") or {}
        map50 = f"{test['map50']:.3f}" if test.get("map50") is not None else "n/a"
        lines.append("| " + " | ".join([row, data, epochs, *recalls, prec,
                                        "n/a" if fpc is None else f"{fpc:.2f}", map50]) + " |")
    md = "\n".join(lines) + "\n"
    out = a.reports / "ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    return md


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", default=list(ROWS), choices=ROWS)
    ap.add_argument("--synth", type=Path, default=ROOT / "data/twin/yolo")
    ap.add_argument("--real", type=Path, default=ROOT / "data/real/yolo")
    ap.add_argument("--test-grids", type=Path, help="full rectified real test grids (skip plate eval if absent)")
    ap.add_argument("--test-sidecars", type=Path, help="their sidecars (default: next to the grids)")
    ap.add_argument("--check-tiles", type=Path, help="real tiles for the ONNX export check")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--epochs-ft", type=int, help="stage-2 epochs for synth_then_real (default --epochs)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--real-frac", type=float, default=0.25)
    ap.add_argument("--real-oversample", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run: 1 epoch, batch 4, no plots")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--report-only", action="store_true", help="only rebuild reports/ablation.md")
    a = ap.parse_args(argv)
    if a.quick:
        a.epochs, a.epochs_ft, a.batch, a.workers = 1, 1, 4, 0
    if a.test_grids is not None and not a.test_grids.is_dir():
        print(f"no test grids at {a.test_grids}: plate evaluation skipped")
        a.test_grids = None
    if not a.report_only:
        log, skipped = [], set()
        for name, cmd in commands(a):
            row = name.split(":")[0]
            if row in skipped:
                continue
            print(f"\n# {name}\n{shlex.join(cmd)}", flush=True)
            if a.dry_run:
                continue
            if name.endswith((":train", ":train1")) and not has_data(a, row):
                print(f"skip {row}: no dataset (see the datasets step)")
                skipped.add(row)
                continue
            t0 = time.time()
            subprocess.run(cmd, check=True, cwd=ROOT)
            log.append({"step": name, "seconds": round(time.time() - t0, 1)})
        if a.dry_run:
            return
        paths(a)["reports"].mkdir(parents=True, exist_ok=True)
        (paths(a)["reports"] / "steps.json").write_text(json.dumps(log, indent=1))
    print(report(a))


if __name__ == "__main__":
    main()
