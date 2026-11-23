"""Train one YOLOv8n per capture arm for the lighting experiment (DESIGN_STEEL G).

Arms: `naive` = the old method (one auto-exposed both-pods frame through the ISP) and
`burst` = the new method (5-frame burst, ambient subtraction, min(L, R), quality gate).
Both arms use the same training plans, the same renders and the same tile pipeline; only
post's `train_arm` differs. Each model is evaluated on its own arm's grids by
training/eval_lighting.py.

Making the two tile trees from the same renders. Post deletes a scene's work dir after
it succeeds, so render in chunks with --keep-work, post the kept work dirs a second time
with train_arm: naive, then delete them. A kept work dir is about 63 MB (EXR + maps), so a
chunk of 100 needs about 6.3 GB; run stops below 2 GB free, and re-running the same loop
fills the gaps (finished scenes are skipped; only new work dirs get the naive pass):

  .venv/bin/python training/train_arms.py --write-naive-config runs/twin_naive.yaml
  for s in $(seq 0 100 1400); do
    .venv/bin/python -m ayas.twin.cli run --n 100 --start-seed $s --out data/twin/arm_burst --keep-work
    .venv/bin/python -m ayas.twin.cli post --work data/twin/arm_burst/work --out data/twin/arm_naive \\
        --config runs/twin_naive.yaml
    rm -rf data/twin/arm_burst/work
  done

(burst tiles skip variants whose gate says RETAKE; the naive arm has no gate.) Then, in
the training venv:

  .venv-train/bin/python training/train_arms.py --burst data/twin/arm_burst/yolo \\
      --naive data/twin/arm_naive/yolo --matrix data/twin_matrix
  .venv-train/bin/python training/train_arms.py ... --dry-run      # print the commands only

Steps: datasets (runs/datasets/arm_<arm>/: each arm's own train/val tiles of the same
rendered plans; the sidecar scene lists of the two trees are compared and a mismatch is
reported. --match scenes keeps only scenes that have tiles in both trees, which drops from
the naive arm the scenes whose burst variants were all refused, so it is not the default:
it would weaken the baseline), train.py (native scale: imgsz 640 = no resize,
scale 0.1, hsv_h 0.01, hsv_s 0.4, hsv_v 0.4, same seed and epochs), export_onnx.py
(models/v8n_<arm>.onnx + .json), and with --matrix, eval_lighting.py on the matrix
(reports/lighting_matrix.md). Tiles of lighting-matrix plates (seed >= 900000) are refused.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

import build_datasets as BD
from train import TRAIN_SETTINGS
from ayas.twin.matrix import TEST_SEED_MIN

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ARMS = ("naive", "burst")
SCENE = re.compile(r"^(s(\d{6,}))")          # twin tile stems: s<seed>[_<row>]_v<k>_r<i>c<j>
VARIANT = re.compile(r"_v\d+$")


def scene_key(img: Path) -> str:
    """Scene id of a twin tile: the tile stem without _r<i>c<j> and _v<k>."""
    return VARIANT.sub("", BD.capture_key(img))


def check_seeds(images: list[Path]) -> None:
    """Refuse any lighting-matrix plate (seed >= TEST_SEED_MIN) in the training lists."""
    for p in images:
        m = SCENE.match(p.stem)
        if m and int(m.group(2)) >= TEST_SEED_MIN:
            raise SystemExit(f"lighting-matrix test plate in the training tiles: {p}")


def plan_check(roots: dict[str, Path]) -> dict[str, Any]:
    """Compare the scenes post processed for each arm (<tree>/../sidecar/*.json): same plans?"""
    scenes = {arm: {p.stem for p in (Path(r).parent / "sidecar").glob("*.json")} for arm, r in roots.items()}
    known = {a: s for a, s in scenes.items() if s}
    same = len({frozenset(s) for s in known.values()}) <= 1
    if not same:
        allx = set.union(*known.values())
        print("WARNING: the arms were post-processed from different scenes: "
              + ", ".join(f"{a} misses {len(allx - s)}" for a, s in known.items()))
    return {"same_scenes": same, "n_scenes": {a: len(s) for a, s in scenes.items()}}


def arm_lists(roots: dict[str, Path], match: str = "none") -> dict[str, dict[str, list[Path]]]:
    """{arm: {train, val}} image lists; match=scenes keeps only scenes present in every arm."""
    lists = {arm: {s: BD.list_images(root, s) for s in ("train", "val")} for arm, root in roots.items()}
    for arm in lists:
        BD.check_names(roots[arm])
        check_seeds(lists[arm]["train"] + lists[arm]["val"])
    if match == "scenes" and len(lists) > 1:
        for split in ("train", "val"):
            common = set.intersection(*({scene_key(p) for p in v[split]} for v in lists.values()))
            for v in lists.values():
                v[split] = [p for p in v[split] if scene_key(p) in common]
    return lists


def build_arm_datasets(roots: dict[str, Path], out: Path, match: str = "none", seed: int = 0) -> dict[str, Path]:
    """Write runs/datasets/arm_<arm>/{train,val}.txt, data.yaml, manifest.json; returns {arm: data.yaml}."""
    lists = arm_lists(roots, match)
    plans = plan_check(roots)
    written = {}
    for arm, v in lists.items():
        if not v["train"] or not v["val"]:
            raise SystemExit(f"arm {arm}: no train or val tiles under {roots[arm]} (match={match}). For the burst "
                             "arm, check how many variants the gate refused (sidecar variants[].gate)")
        meta = {"arm": arm, "source": str(roots[arm]), "match": match, "seed": seed, "plans": plans,
                "scenes": {s: len({scene_key(p) for p in v[s]}) for s in ("train", "val")},
                "settings": {k: TRAIN_SETTINGS[k] for k in ("imgsz", "hsv_h", "hsv_s", "hsv_v", "scale")}}
        written[arm] = BD.write_split(out / f"arm_{arm}", v["train"], v["val"], [], meta)
        print(f"arm {arm}: {len(v['train'])} train / {len(v['val'])} val tiles "
              f"({meta['scenes']['train']} / {meta['scenes']['val']} scenes)")
    return written


def write_naive_config(path: Path, base: Path = ROOT / "configs/twin.yaml") -> Path:
    """A copy of the twin config with post.train_arm: naive (for the second post pass)."""
    cfg = yaml.safe_load(Path(base).read_text())
    cfg.setdefault("post", {})["train_arm"] = "naive"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def commands(a: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """(step, argv) for training, export and the optional evaluation."""
    py = sys.executable
    steps = []
    for arm in a.arms:
        common = ["--epochs", str(a.epochs), "--batch", str(a.batch), "--seed", str(a.seed), "--device", a.device,
                  "--project", str(a.runs), "--workers", str(a.workers), "--exist-ok"]
        if a.quick:
            common += ["--no-plots", "--patience", "1"]
        steps.append((f"{arm}:train", [py, str(HERE / "train.py"), "--data", str(a.datasets / f"arm_{arm}" / "data.yaml"),
                                       "--name", f"arm_{arm}", "--model", a.model, *common]))
        exp = [py, str(HERE / "export_onnx.py"), "--weights", str(a.runs / f"arm_{arm}" / "weights/best.pt"),
               "--name", f"v8n_{arm}", "--out", str(a.models)]
        if a.check_tiles:
            exp += ["--check-tiles", str(a.check_tiles)]
        steps.append((f"{arm}:export", exp))
    if a.matrix:
        ev = [py, str(HERE / "eval_lighting.py"), "--matrix", str(a.matrix), "--out", str(a.reports),
              "--conf", str(a.conf), "--spec", str(a.spec)]
        for arm in a.arms:
            ev += ["--model", f"{arm}={a.models / f'v8n_{arm}.onnx'}"]
        steps.append(("eval", ev))
    return steps


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--burst", type=Path, default=ROOT / "data/twin/arm_burst/yolo", help="burst-arm YOLO tree")
    ap.add_argument("--naive", type=Path, default=ROOT / "data/twin/arm_naive/yolo", help="naive-arm YOLO tree")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--match", choices=("none", "scenes"), default="none",
                    help="none: each arm's own tiles of the same renders; scenes: only scenes with tiles in both")
    ap.add_argument("--matrix", type=Path, help="lighting matrix dir: evaluate the two models on it at the end")
    ap.add_argument("--spec", type=Path, default=ROOT / "configs/spec_steel.yaml")
    ap.add_argument("--datasets", type=Path, default=ROOT / "runs/datasets")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs/train")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--check-tiles", type=Path, help="tiles for the ONNX export check")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt (COCO start) or yolov8n.yaml (scratch)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--quick", action="store_true", help="smoke run: 1 epoch, batch 4, no plots")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only (writes nothing)")
    ap.add_argument("--write-naive-config", type=Path, metavar="PATH",
                    help="write configs/twin.yaml with post.train_arm: naive to PATH and exit")
    a = ap.parse_args(argv)
    if a.write_naive_config:
        print(f"wrote {write_naive_config(a.write_naive_config)}")
        return {}
    if a.quick:
        a.epochs, a.batch, a.workers = 1, 4, 0
    roots = {arm: getattr(a, arm) for arm in a.arms}
    print("# datasets: " + ", ".join(f"{arm} <- {root}" for arm, root in roots.items()))
    if not a.dry_run:
        build_arm_datasets(roots, a.datasets, a.match, a.seed)
    log = []
    for name, cmd in commands(a):
        print(f"\n# {name}\n{shlex.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=ROOT)
        log.append({"step": name, "seconds": round(time.time() - t0, 1)})
    if not a.dry_run:
        a.runs.mkdir(parents=True, exist_ok=True)
        (a.runs / "train_arms_steps.json").write_text(json.dumps(log, indent=1))
    return {"steps": log}


if __name__ == "__main__":
    main()
