"""Build the Ultralytics image lists for the 5-row sim-to-real ablation.

Inputs are two YOLO trees, each images|labels/{train,val,test}: the twin (data/twin/yolo)
and the real captures (data/real/yolo). For each row this writes
runs/datasets/<row>/{train.txt,val.txt,test.txt,data.yaml,manifest.json}; synth_then_real
gets stage1/ (twin) and stage2/ (real) subdirectories instead.

Rows:
  real_only        real train
  real_25          a seeded 25 % of the real train CAPTURES (all tiles of a picked
                   capture), stratified by coupon name prefix (P1, P3, ...)
  synth_only       twin train
  synth_then_real  stage1 = twin train; stage2 = real train, fine-tuned from stage1
  mixed            twin train + real train (real listed --real-oversample times)

val is the real val split if it has images, else the twin val split; test is always the
real test split (P5: never trained or tuned on). Lists hold absolute paths; Ultralytics
finds each label by replacing /images/ with /labels/ (a missing label = background).

Example (training venv):
  .venv-train/bin/python training/build_datasets.py --synth data/twin/yolo --real data/real/yolo
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

from ayas.tiles import CLASSES

ROOT = Path(__file__).resolve().parents[1]

ROWS = ("real_only", "real_25", "synth_only", "synth_then_real", "mixed")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
TILE_SUFFIX = re.compile(r"_r\d+c\d+$")      # ayas.tiles.Tile.name at the end of a tile stem
COUPON = re.compile(r"^([A-Za-z]+\d+)")


def list_images(root: Path | None, split: str) -> list[Path]:
    if root is None:
        return []
    d = Path(root) / "images" / split
    if not d.is_dir():
        return []
    return sorted(p.resolve() for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def label_path(img: Path) -> Path:
    s = str(img)
    i = s.rfind("/images/")
    return Path(s[:i] + "/labels/" + s[i + 8 :]).with_suffix(".txt")


def capture_key(img: Path) -> str:
    """Tile stem without its r<row>c<col> suffix: all tiles of one capture share it."""
    return TILE_SUFFIX.sub("", img.stem)


def coupon_key(img: Path) -> str:
    m = COUPON.match(img.stem)
    return m.group(1) if m else img.stem.split("_")[0]


def subsample_captures(images: list[Path], frac: float, seed: int) -> list[Path]:
    """Seeded fraction of the captures, at least one per coupon, keeping every tile of a picked capture."""
    by_coupon: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for p in images:
        by_coupon[coupon_key(p)][capture_key(p)].append(p)
    rng = random.Random(seed)
    picked: list[Path] = []
    for coupon in sorted(by_coupon):
        caps = sorted(by_coupon[coupon])
        k = max(1, round(frac * len(caps)))
        for c in sorted(rng.sample(caps, k)):
            picked += by_coupon[coupon][c]
    return sorted(picked)


def check_names(root: Path | None) -> None:
    """If a tree has a data.yaml, its class names must be ayas.tiles.CLASSES."""
    if root is None or not (Path(root) / "data.yaml").exists():
        return
    names = yaml.safe_load((Path(root) / "data.yaml").read_text()).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if names is not None and list(names) != list(CLASSES):
        raise SystemExit(f"{root}/data.yaml names {names} != {list(CLASSES)}")


def write_split(out: Path, train: list[Path], val: list[Path], test: list[Path], meta: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    data = {"path": str(out.resolve()), "names": dict(enumerate(CLASSES))}
    for name, items in (("train", train), ("val", val), ("test", test)):
        if items:
            (out / f"{name}.txt").write_text("".join(f"{p}\n" for p in items))
            data[name] = str((out / f"{name}.txt").resolve())
        elif (out / f"{name}.txt").exists():
            (out / f"{name}.txt").unlink()
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    uniq = set(train)
    meta = dict(meta, counts={"train": len(train), "train_unique": len(uniq), "val": len(val), "test": len(test)},
                missing_labels={"train": sum(not label_path(p).exists() for p in uniq),
                                "val": sum(not label_path(p).exists() for p in val)})
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    return out / "data.yaml"


def build(synth: Path | None, real: Path | None, out: Path, rows=ROWS, frac: float = 0.25,
          oversample: int = 1, seed: int = 0) -> dict[str, list[Path]]:
    """Write the dataset lists; returns {row: [data.yaml, ...]} for the rows that have data."""
    check_names(synth)
    check_names(real)
    s_train, s_val = list_images(synth, "train"), list_images(synth, "val")
    r_train, r_val, r_test = list_images(real, "train"), list_images(real, "val"), list_images(real, "test")
    leak = {p.name for p in r_test} & {p.name for p in r_train + r_val}
    if leak:
        raise SystemExit(f"{len(leak)} real test images also appear in train/val, e.g. {sorted(leak)[:3]}")
    val, val_src = (r_val, "real") if r_val else (s_val, "synth")
    base = {"seed": seed, "val_source": val_src, "test_source": "real" if r_test else None,
            "synth": str(synth) if synth else None, "real": str(real) if real else None}
    trains = {
        "real_only": [r_train],
        "real_25": [subsample_captures(r_train, frac, seed)],
        "synth_only": [s_train],
        "synth_then_real": [s_train, r_train],
        "mixed": [s_train + r_train * max(1, oversample)],
    }
    written: dict[str, list[Path]] = {}
    for row in rows:
        if row not in trains:
            raise SystemExit(f"unknown row {row}; choose from {ROWS}")
        stages = trains[row]
        if any(not t for t in stages) or not val:
            print(f"skip {row}: no training or val images")
            continue
        meta = dict(base, row=row, real_frac=frac if row == "real_25" else None,
                    real_oversample=oversample if row == "mixed" else None)
        if len(stages) == 1:
            written[row] = [write_split(out / row, stages[0], val, r_test, meta)]
        else:
            written[row] = [write_split(out / row / f"stage{i + 1}", t, val, r_test, dict(meta, stage=i + 1))
                            for i, t in enumerate(stages)]
        print(f"{row}: " + ", ".join(f"{len(t)} train" for t in stages) + f", {len(val)} val ({val_src}), "
              f"{len(r_test)} test")
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", type=Path, default=ROOT / "data/twin/yolo", help="twin YOLO tree")
    ap.add_argument("--real", type=Path, default=ROOT / "data/real/yolo", help="real YOLO tree")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/datasets")
    ap.add_argument("--rows", nargs="+", default=list(ROWS), choices=ROWS)
    ap.add_argument("--real-frac", type=float, default=0.25, help="fraction of real captures for real_25")
    ap.add_argument("--real-oversample", type=int, default=1, help="times the real list is repeated in mixed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    synth = a.synth if a.synth.is_dir() else None
    real = a.real if a.real.is_dir() else None
    if synth is None and real is None:
        raise SystemExit(f"neither {a.synth} nor {a.real} exists")
    build(synth, real, a.out, a.rows, a.real_frac, a.real_oversample, a.seed)


if __name__ == "__main__":
    main()
