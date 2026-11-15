"""Pack grids into a compact upload bundle: grids/<arm>/<split>/*.jpg + sidecar .json (+ manifests).

PNG grids are re-encoded as JPEG q95 (~0.56 MB instead of ~2.9 MB for a 100 x 100 mm grid) and
the sidecar's `grid_image` is updated. Debug overlays, plans, bursts, work dirs and YOLO tiles
are left out. Only twin-rendered data goes in: nothing personal, no third-party datasets.

  python training/jepa/pack.py --src <pairs dir> [<dir> ...] --out /tmp/jepa_data.tgz [--splits jepa_train]
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Sequence

import cv2


def _anon(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    """No local user or group names in the archive (it goes to a cloud instance)."""
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    return ti


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = _anon(tarfile.TarInfo(name))
    ti.size = len(data)
    ti.mode = 0o644
    ti.mtime = int(time.time())
    tar.addfile(ti, io.BytesIO(data))


def pack(srcs: Sequence[str | Path], out: str | Path, splits: Sequence[str] | None = None,
         arms: Sequence[str] | None = None, quality: int = 95) -> dict:
    n, size = 0, 0
    with tarfile.open(out, "w:gz") as tar:
        for src in srcs:
            src = Path(src)
            root = src.name
            for man in ("pairs_manifest.json", "matrix_manifest.json"):
                if (src / man).exists():
                    tar.add(src / man, f"{root}/{man}", filter=_anon)
            for meta_path in sorted((src / "grids").glob("*/*/*.json")):
                arm, split = meta_path.parent.parent.name, meta_path.parent.name
                if (splits and split not in splits) or (arms and arm not in arms):
                    continue
                meta = json.loads(meta_path.read_text())
                img_path = meta_path.parent / meta.get("grid_image", meta_path.stem + ".png")
                if not img_path.exists():
                    continue
                rel = f"{root}/grids/{arm}/{split}"
                if img_path.suffix.lower() == ".png":
                    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                    data = buf.tobytes()
                    meta["grid_image"] = meta_path.stem + ".jpg"
                    meta["packed_from"] = "png->jpg q%d" % quality
                else:
                    data = img_path.read_bytes()
                _add_bytes(tar, f"{rel}/{meta['grid_image']}", data)
                _add_bytes(tar, f"{rel}/{meta_path.name}", json.dumps(meta).encode())
                n += 1
                size += len(data)
    return {"grids": n, "image_bytes": size, "out": str(out), "tar_bytes": Path(out).stat().st_size}


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--quality", type=int, default=95)
    a = ap.parse_args(argv)
    res = pack(a.src, a.out, a.splits, a.arms, a.quality)
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
