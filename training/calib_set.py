"""INT8 calibration set for the Hailo compile: N random 640 x 640 crops of REAL training images.

Quantisation statistics must match what the camera sees, so the set comes from real
captures and never from the test split (P5). Twin tiles (names s<seed>_...) are refused
unless --allow-twin is given: Phase 1 (steel, no real captures yet) calibrates on the
burst-arm TRAIN tiles of the twin, which is what the deployed pipeline feeds the model;
replace them with real burst captures as soon as those exist. Lighting-matrix plates
(seed >= 900000, trinetra.twin.matrix) are always refused. Images bigger than a tile
(rectified grids) give random crops; 640 px tiles are used whole. When there are fewer
source images than N, flipped copies (horizontal, vertical, both) fill the set; past
that, images repeat with fresh crops.

Writes <out>/calib_640.npy, (N, 640, 640, 3) uint8 RGB with raw 0..255 values (the
.alls normalisation runs on chip), and <out>/calib_640/*.png, the directory that
`hailomz compile --calib-path` reads. N = 1024 takes about 1.26 GB for the .npy plus
about 0.6-0.9 GB of PNGs: use --no-npy when disk is tight.

Examples:
  .venv-train/bin/python training/calib_set.py --images data/real/yolo/images/train --out runs/calib
  .venv-train/bin/python training/calib_set.py --images data/twin/arm_burst/yolo/images/train \
      --allow-twin --out runs/calib            # Phase 1: twin burst tiles
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from trinetra.twin.matrix import TEST_SEED_MIN     # lighting-matrix test plates start here

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
FLIPS = (None, 1, 0, -1)        # cv2.flip codes: none, horizontal, vertical, both
COUPON = re.compile(r"^([A-Za-z]+\d+)")
TWIN = re.compile(r"^s(\d{6,})(?:_|$)")       # twin scene names: s<seed>, s<seed>_v<k>_r<i>c<j>


def source_images(dirs: list[Path], test_dir: Path | None, forbid: set[str],
                  allow_twin: bool = False) -> list[Path]:
    """Training images; refuses a test split, a forbidden coupon, matrix plates and (unless allowed) twin tiles."""
    files = sorted({p.resolve() for d in dirs for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT})
    if not files:
        raise SystemExit(f"no images in {dirs}")
    test_names = set()
    if test_dir is not None and Path(test_dir).is_dir():
        test_names = {p.name for p in Path(test_dir).rglob("*") if p.suffix.lower() in IMG_EXT}
    for p in files:     # explicit checks, not assert: they must survive python -O
        if "test" in p.parts or "twin_test" in p.parts:
            raise SystemExit(f"test-split image in the calibration sources: {p}")
        if p.name in test_names:
            raise SystemExit(f"image also in the test split: {p}")
        m = COUPON.match(p.stem)
        if m and m.group(1) in forbid:
            raise SystemExit(f"held-out coupon {m.group(1)} image in the calibration sources: {p}")
        t = TWIN.match(p.stem)
        if t and int(t.group(1)) >= TEST_SEED_MIN:
            raise SystemExit(f"lighting-matrix test plate in the calibration sources: {p}")
        if t and not allow_twin:
            raise SystemExit(f"twin render in the calibration sources (pass --allow-twin for Phase 1): {p}")
    return files


def plan(files: list[Path], n: int, seed: int) -> list[tuple[Path, int | None]]:
    """n (image, flip) pairs: every image once, then flipped copies, then repeats."""
    rng = random.Random(seed)
    pool: list[tuple[Path, int | None]] = []
    for flip in FLIPS:                     # a flip tier only fills what the previous tiers left
        tier = [(f, flip) for f in files]
        rng.shuffle(tier)
        pool += tier[: n - len(pool)]
        if len(pool) >= n:
            break
    if len(pool) < n:
        print(f"note: only {len(pool)} unique image/flip pairs for {n} samples; repeating with new crops")
        pool = [pool[i % len(pool)] for i in range(n)]
    rng.shuffle(pool)
    return pool


def crop(img: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size or w < size:
        img = np.pad(img, ((0, max(0, size - h)), (0, max(0, size - w)), (0, 0)))
        h, w = img.shape[:2]
    y0, x0 = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
    return img[y0 : y0 + size, x0 : x0 + size]


def write(files: list[Path], out: Path, n: int = 1024, size: int = 640, seed: int = 0,
          npy: bool = True, png: bool = True) -> dict:
    need = (n * size * size * 3 if npy else 0) + (n * size * size * 2 if png else 0)   # PNG ~2/3 of raw
    free = shutil.disk_usage(out.parent if out.parent.exists() else ROOT).free
    if need + 1e9 > free:
        raise SystemExit(f"need about {need / 1e9:.1f} GB (+1 GB margin), {free / 1e9:.1f} GB free")
    out.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out / f"calib_{size}.npy", "w+", np.uint8, (n, size, size, 3)) if npy else None
    pdir = out / f"calib_{size}"
    if png:
        pdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    cache: dict[Path, np.ndarray] = {}
    used = []
    for i, (f, flip) in enumerate(plan(files, n, seed)):
        if f not in cache:
            if len(cache) > 64:
                cache.clear()
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is None:
                raise SystemExit(f"cannot read {f}")
            cache[f] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = crop(cache[f], size, rng)
        if flip is not None:
            t = cv2.flip(t, flip)
        t = np.ascontiguousarray(t)
        if arr is not None:
            arr[i] = t
        if png:
            cv2.imwrite(str(pdir / f"calib_{i:04d}.png"), cv2.cvtColor(t, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
        used.append({"i": i, "src": f.name, "flip": flip})
    if arr is not None:
        arr.flush()
        del arr
    info = {"n": n, "size": size, "seed": seed, "n_sources": len(files), "dtype": "uint8", "channels": "RGB",
            "range": "0..255 (not normalised)", "samples": used}
    (out / "calib_manifest.json").write_text(json.dumps(info))
    return info


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, nargs="+", default=[ROOT / "data/real/yolo/images/train"],
                    help="REAL training images (tiles or rectified grids)")
    ap.add_argument("--test-dir", type=Path, default=ROOT / "data/real/yolo/images/test",
                    help="real test split: none of its images may be used")
    ap.add_argument("--forbid", default="P5", help="comma-separated held-out coupon prefixes")
    ap.add_argument("--allow-twin", action="store_true",
                    help="accept twin TRAIN tiles (Phase 1, before real burst captures exist)")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/calib")
    ap.add_argument("--no-npy", dest="npy", action="store_false")
    ap.add_argument("--no-png", dest="png", action="store_false")
    a = ap.parse_args(argv)
    forbid = {s.strip() for s in a.forbid.split(",") if s.strip()}
    files = source_images(a.images, a.test_dir, forbid, a.allow_twin)
    info = write(files, a.out, a.n, a.size, a.seed, a.npy, a.png)
    print(f"{info['n']} calibration tiles from {info['n_sources']} images -> {a.out}")
    return info


if __name__ == "__main__":
    main()
