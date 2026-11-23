"""Grid index, canonical plate orientation and masks (numpy + cv2; no torch).

Post writes one full rectified grid per (scene, variant, arm): grids/<arm>/<split>/<scene>_v<k>.{png,jpg}
with a sidecar .json next to it (schema ayas.twin.grid/1). Within one device mode a plate's
rows share the camera pose, so their grids are pixel-aligned (checked on the pilot: hood rows
within 0.02 px of C0h, accepted open rows within ~0.3 px of C0o). Between the open and hood
rows the grid can be rotated by k x 90 deg (the grid follows the image's corner order: the
sidecar's `datum_corner` / `H_grid_plate`). `canonical_affine` maps every grid to one plate
frame (datum top-left: x = u / mm_per_px + border, y = v / mm_per_px + border), so the same
plate region has the same pixel coordinates in every row, open or hood.

H_grid_plate maps plate mm -> grid px (pixel centres at integer coordinates).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

TEST_SEED_MIN = 900000
RETAKE = "RETAKE"
IMG_EXT = (".png", ".jpg", ".jpeg")


@dataclass
class GridRec:
    """One rectified grid image and what the pairing and evaluation need from its sidecar."""

    image: Path
    meta_path: Path
    plate: str
    seed: int
    condition: str
    device_mode: str
    tag: str
    k: int
    arm: str
    split: str
    gate: str
    material: str
    size_mm: tuple[float, float]
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def accepted(self) -> bool:
        return self.gate != RETAKE

    @property
    def group(self) -> tuple[str, str, int, str]:
        """Rows of one pairing group: same plate, device mode, variant and arm."""
        return (self.plate, self.device_mode, self.k, self.arm)

    def meta(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text())


def _image_for(meta_path: Path, meta: dict[str, Any]) -> Path | None:
    name = meta.get("grid_image")
    if name and (meta_path.parent / name).exists():
        return meta_path.parent / name
    for ext in IMG_EXT:
        p = meta_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def rec_from_meta(meta_path: Path, meta: dict[str, Any]) -> GridRec | None:
    img = _image_for(meta_path, meta)
    if img is None:
        return None
    m = meta.get("matrix") or {}
    part = meta.get("part") or {}
    cond = meta.get("condition") or m.get("condition") or ""
    dev = meta.get("device_mode") or m.get("device_mode") or ""
    tag = m.get("row") or (cond.split("_")[0] + (dev[:1] if dev else ""))
    seed = m.get("plate_seed")
    if seed is None:
        try:
            seed = int(str(meta.get("plate", "s0"))[1:])
        except ValueError:
            seed = -1
    return GridRec(image=img, meta_path=meta_path, plate=str(meta.get("plate") or m.get("plate")),
                   seed=int(seed), condition=cond, device_mode=dev, tag=tag, k=int(meta.get("k", 0)),
                   arm=str(meta.get("arm", "")), split=str(meta.get("split", "")),
                   gate=str(meta.get("gate", "")), material=str(part.get("material", "")),
                   size_mm=tuple(float(x) for x in (meta.get("plate_size_mm") or part.get("size_mm") or (0, 0))))


def index_grids(roots: Iterable[str | Path], arms: Sequence[str] | None = None,
                splits: Sequence[str] | None = None) -> list[GridRec]:
    """Every grid with a sidecar under <root>/grids/<arm>/<split>/ (sorted, deterministic)."""
    recs: list[GridRec] = []
    for root in roots:
        base = Path(root) / "grids"
        if not base.is_dir():
            base = Path(root)            # a packed dataset (pack.py) keeps the same tree
            if not (base / "grids").is_dir() and not any(base.glob("*/*/*.json")):
                continue
            base = base / "grids" if (base / "grids").is_dir() else base
        for meta_path in sorted(base.glob("*/*/*.json")):
            arm, split = meta_path.parent.parent.name, meta_path.parent.name
            if arms and arm not in arms:
                continue
            if splits and split not in splits:
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue                  # a file being rewritten by post right now
            if not str(meta.get("schema", "")).startswith("ayas.twin.grid"):
                continue
            rec = rec_from_meta(meta_path, meta)
            if rec is not None:
                recs.append(rec)
    return recs


def refuse_test_plates(recs: Iterable[GridRec]) -> None:
    bad = sorted({r.plate for r in recs if r.seed >= TEST_SEED_MIN or r.split == "test"})
    if bad:
        raise ValueError(f"test plates (seed >= {TEST_SEED_MIN} or split test) must never be trained on: {bad[:5]}")


def groups(recs: Iterable[GridRec]) -> dict[tuple, list[GridRec]]:
    out: dict[tuple, list[GridRec]] = {}
    for r in recs:
        out.setdefault(r.group, []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r.tag)
    return out


# ---- canonical orientation ------------------------------------------------------------------

def canonical_size(meta: dict[str, Any]) -> tuple[int, int]:
    g = meta["grid"]
    mmpp, border = float(g["mm_per_px"]), float(g.get("border_mm", 0.0))
    w_mm, h_mm = (float(x) for x in meta.get("plate_size_mm") or g["plate_mm"])
    return int(round((w_mm + 2 * border) / mmpp)), int(round((h_mm + 2 * border) / mmpp))


def canonical_affine(meta: dict[str, Any]) -> np.ndarray:
    """2x3 affine: grid px -> canonical px (datum top-left, x = u/mmpp + b, y = v/mmpp + b)."""
    g = meta["grid"]
    mmpp, border = float(g["mm_per_px"]), float(g.get("border_mm", 0.0))
    H = np.asarray(meta["H_grid_plate"], float)          # plate mm -> grid px
    A = H / H[2, 2]
    if np.abs(A[2, :2]).max() > 1e-6:
        raise ValueError("H_grid_plate is not affine; canonical_affine expects a rectified grid")
    C = np.array([[1 / mmpp, 0, border / mmpp], [0, 1 / mmpp, border / mmpp], [0, 0, 1]])   # plate -> canonical
    M = C @ np.linalg.inv(A)                              # grid -> canonical
    return M[:2]


def _is_integral(M: np.ndarray, tol: float = 1e-3) -> bool:
    return bool(np.all(np.abs(M - np.round(M)) < tol))


def to_canonical(img: np.ndarray, meta: dict[str, Any], interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Warp a grid image (H x W [x C]) into the canonical plate frame (exact for 90 deg turns)."""
    M = canonical_affine(meta)
    w, h = canonical_size(meta)
    if _is_integral(M):
        M = np.round(M)
        if np.allclose(M, [[1, 0, 0], [0, 1, 0]]) and img.shape[:2] == (h, w):
            return img
        interp = cv2.INTER_NEAREST
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def read_rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_canonical(rec: GridRec, meta: dict[str, Any] | None = None) -> np.ndarray:
    meta = meta or rec.meta()
    return to_canonical(read_rgb(rec.image), meta)


# ---- masks (canonical frame) ----------------------------------------------------------------

def plate_rect(meta: dict[str, Any]) -> tuple[int, int, int, int]:
    """Canonical pixel rectangle [x0, y0, x1, y1) of the plate (pixel centres inside the plate)."""
    g = meta["grid"]
    mmpp, border = float(g["mm_per_px"]), float(g.get("border_mm", 0.0))
    w_mm, h_mm = (float(x) for x in meta.get("plate_size_mm") or g["plate_mm"])
    b = border / mmpp
    x0, y0 = int(np.floor(b)) + 1, int(np.floor(b)) + 1
    return x0, y0, int(np.ceil(b + w_mm / mmpp)), int(np.ceil(b + h_mm / mmpp))


def plate_mask(meta: dict[str, Any]) -> np.ndarray:
    w, h = canonical_size(meta)
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = plate_rect(meta)
    m[y0:y1, x0:x1] = True
    return m


def _fill(objs: Sequence[dict[str, Any]], M: np.ndarray, shape: tuple[int, int], min_px: float = 1.0) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    for o in objs:
        polys = o.get("polys_grid") or []
        drawn = False
        for poly in polys:
            p = warp_points(poly, M)
            if len(p) >= 3:
                cv2.fillPoly(out, [np.round(p).astype(np.int32)], 1)
                cv2.polylines(out, [np.round(p).astype(np.int32)], True, 1, max(1, int(round(min_px))))
                drawn = True
        if not drawn and o.get("bbox_grid"):
            x0, y0, x1, y1 = o["bbox_grid"]
            q = warp_points([[x0, y0], [x1 - 1, y1 - 1]], M)
            a, b = np.floor(q.min(0)).astype(int), np.ceil(q.max(0)).astype(int)
            out[max(0, a[1]):b[1] + 1, max(0, a[0]):b[0] + 1] = 1
    return out.astype(bool)


def dilate(mask: np.ndarray, r_px: float) -> np.ndarray:
    r = int(round(r_px))
    if r <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def defect_masks(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Per labelled defect: {id, cls, kind, visible, mask (canonical bool)}."""
    M = canonical_affine(meta)
    w, h = canonical_size(meta)
    out = []
    for d in meta.get("defects") or []:
        if not d.get("label", True):
            continue
        out.append({"id": d.get("id"), "cls": d.get("cls"), "kind": d.get("kind"),
                    "visible": d.get("visible"), "visible_arm": d.get("visible_arm"),
                    "mask": _fill([d], M, (h, w))})
    return out


def any_defect_mask(meta: dict[str, Any], margin_mm: float = 0.0, labelled_only: bool = False) -> np.ndarray:
    """Union of defect footprints (labelled and unlabelled by default) dilated by margin_mm."""
    M = canonical_affine(meta)
    w, h = canonical_size(meta)
    objs = [d for d in meta.get("defects") or [] if d.get("label", True) or not labelled_only]
    m = _fill(objs, M, (h, w))
    return dilate(m, margin_mm / float(meta["grid"]["mm_per_px"]))


def distractor_points(meta: dict[str, Any]) -> list[tuple[str, float, float]]:
    """(kind, x, y) canonical px of the unlabelled distractors (marker, dust, blend, ...)."""
    g = meta["grid"]
    mmpp, b = float(g["mm_per_px"]), float(g.get("border_mm", 0.0)) / float(g["mm_per_px"])
    return [(d.get("kind", "?"), d["u_mm"] / mmpp + b, d["v_mm"] / mmpp + b)
            for d in meta.get("distractors") or [] if "u_mm" in d and "v_mm" in d]


def feature_mask(meta: dict[str, Any]) -> np.ndarray:
    """Part features (through holes, slots): normal, not defects."""
    M = canonical_affine(meta)
    w, h = canonical_size(meta)
    return _fill(meta.get("features") or [], M, (h, w))
