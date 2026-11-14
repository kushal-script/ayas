"""Native-scale tiles of the rectified plate grid, and merging tile detections back.

The same code cuts training tiles (twin and real) and inference tiles on the Pi, so the
detector always sees defects at the grid scale (0.08 mm/px by default). Never resize a
grid to 640 px: a 0.3 mm scratch would become sub-pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

TILE = 640
MIN_OVERLAP = 128

# YOLO class ids for the steel/metal surface detector (index = id; DESIGN_STEEL.md A).
# Keep in sync with configs/twin.yaml `classes`. Holes and slots are part features, not classes.
CLASSES = ("scratch", "dent", "pit", "corrosion", "crack", "stain")
ELONGATED = ("scratch", "crack")   # these may run across tile seams and get merged back


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    x0: int
    y0: int
    size: int = TILE

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size

    @property
    def name(self) -> str:
        return f"r{self.row}c{self.col}"


@dataclass(frozen=True)
class Det:
    cls: int
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float
    tile: str = ""
    cut: bool = False   # touches an inner tile border, so it may be the part of a longer defect

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    @property
    def name(self) -> str:
        return CLASSES[self.cls] if 0 <= self.cls < len(CLASSES) else str(self.cls)


def tile_starts(length: int, tile: int = TILE, min_overlap: int = MIN_OVERLAP) -> list[int]:
    """Evenly spread tile origins covering [0, length) with at least min_overlap between tiles."""
    if length <= tile:
        return [0]
    n = math.ceil((length - min_overlap) / (tile - min_overlap))
    n = max(n, 2)
    return [int(round(i * (length - tile) / (n - 1))) for i in range(n)]


def tile_grid(width: int, height: int, tile: int = TILE, min_overlap: int = MIN_OVERLAP) -> list[Tile]:
    xs, ys = tile_starts(width, tile, min_overlap), tile_starts(height, tile, min_overlap)
    return [Tile(r, c, x, y, tile) for r, y in enumerate(ys) for c, x in enumerate(xs)]


def cut(image: np.ndarray, tiles: list[Tile], pad_value: int = 0) -> list[np.ndarray]:
    """Crop tiles; grids smaller than a tile are padded at the bottom/right."""
    out = []
    h, w = image.shape[:2]
    for t in tiles:
        crop = image[t.y0 : min(t.y1, h), t.x0 : min(t.x1, w)]
        if crop.shape[0] < t.size or crop.shape[1] < t.size:
            pad = [(0, t.size - crop.shape[0]), (0, t.size - crop.shape[1])] + [(0, 0)] * (image.ndim - 2)
            crop = np.pad(crop, pad, constant_values=pad_value)
        out.append(np.ascontiguousarray(crop))
    return out


def clip_box(box, tile: Tile):
    """Intersect a grid-pixel box (x0, y0, x1, y1) with a tile; returns tile-local coords or None."""
    x0, y0 = max(box[0], tile.x0), max(box[1], tile.y0)
    x1, y1 = min(box[2], tile.x1), min(box[3], tile.y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0 - tile.x0, y0 - tile.y0, x1 - tile.x0, y1 - tile.y0)


def yolo_line(cls: int, box_local, size: int = TILE) -> str:
    x0, y0, x1, y1 = box_local
    return f"{cls} {(x0 + x1) / 2 / size:.6f} {(y0 + y1) / 2 / size:.6f} {(x1 - x0) / size:.6f} {(y1 - y0) / size:.6f}"


def _iou(a: Det, b: Det) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _iom(a: Det, b: Det) -> float:
    """Intersection over the smaller box: catches a truncated copy inside a full box."""
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    m = min(a.area, b.area)
    return ix * iy / m if m > 0 else 0.0


def nms(dets: list[Det], iou: float = 0.5, iom: float = 0.8) -> list[Det]:
    """Class-wise greedy NMS on IoU, plus suppression of boxes mostly inside a better one."""
    keep: list[Det] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(k.cls != d.cls or (_iou(k, d) < iou and _iom(k, d) < iom) for k in keep):
            keep.append(d)
    return keep


def to_grid(dets_local: list[Det], tile: Tile, grid_w: int, grid_h: int, edge_px: float = 2.0) -> list[Det]:
    """Shift tile-local detections into grid pixels and mark ones cut by an inner tile border."""
    out = []
    for d in dets_local:
        g = replace(d, x0=d.x0 + tile.x0, y0=d.y0 + tile.y0, x1=d.x1 + tile.x0, y1=d.y1 + tile.y0, tile=tile.name)
        cut_ = ((d.x0 <= edge_px and tile.x0 > 0) or (d.y0 <= edge_px and tile.y0 > 0)
                or (d.x1 >= tile.size - edge_px and tile.x1 < grid_w)
                or (d.y1 >= tile.size - edge_px and tile.y1 < grid_h))
        out.append(replace(g, cut=cut_))
    return out


def _shared(a: Tile, b: Tile) -> tuple[int, int, int, int] | None:
    """The grid region two tiles both see, or None."""
    x0, y0, x1, y1 = max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _iou_in(a: Det, b: Det, r) -> float:
    """IoU of two boxes after clipping both to the region r (0 if either misses it)."""
    ax0, ay0, ax1, ay1 = max(a.x0, r[0]), max(a.y0, r[1]), min(a.x1, r[2]), min(a.y1, r[3])
    bx0, by0, bx1, by1 = max(b.x0, r[0]), max(b.y0, r[1]), min(b.x1, r[2]), min(b.y1, r[3])
    if ax1 <= ax0 or ay1 <= ay0 or bx1 <= bx0 or by1 <= by0:
        return 0.0
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _still_cut(d: Det, src: list[Tile], grid_w: int, grid_h: int, edge_px: float = 2.0) -> bool:
    """Whether a joined box still reaches an inner border of the area its source tiles cover."""
    x0, y0 = min(t.x0 for t in src), min(t.y0 for t in src)
    x1, y1 = max(t.x1 for t in src), max(t.y1 for t in src)
    return ((d.x0 <= x0 + edge_px and x0 > 0) or (d.y0 <= y0 + edge_px and y0 > 0)
            or (d.x1 >= x1 - edge_px and x1 < grid_w) or (d.y1 >= y1 - edge_px and y1 < grid_h))


def join_seam_pieces(dets: list[Det], tiles: list[Tile], grid_w: int, grid_h: int, iou: float = 0.5) -> list[Det]:
    """Rebuild defects that run across a tile seam from their per-tile pieces (before NMS).

    Overlapping tiles see the same pixels in the region they share, so a defect that leaves
    one tile (a cut box) is seen by the neighbour with the same extent inside that region.
    Same-class boxes from different tiles, at least one of them cut, whose parts inside a
    shared region have IoU >= iou are one defect: they become their union (the more
    confident piece's tile, max confidence). Without this NMS keeps one truncated piece (a
    32 mm scratch over a seam came back 27 mm long) or a big rust patch comes back in two.
    Two separate defects either side of a seam do not share the region, so they stay apart.
    """
    by_name = {t.name: t for t in tiles}
    items = [(d, {d.tile}) for d in dets]
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, sa), (b, sb) = items[i], items[j]
                if a.cls != b.cls or not (a.cut or b.cut) or sa & sb:
                    continue
                regions = (_shared(by_name[p], by_name[q]) for p in sa for q in sb if p in by_name and q in by_name)
                if not any(r is not None and _iou_in(a, b, r) >= iou for r in regions):
                    continue
                best = a if a.conf >= b.conf else b
                u = replace(best, conf=max(a.conf, b.conf), x0=min(a.x0, b.x0), y0=min(a.y0, b.y0),
                            x1=max(a.x1, b.x1), y1=max(a.y1, b.y1))
                src = [by_name[n] for n in sa | sb if n in by_name]
                items[i] = (replace(u, cut=_still_cut(u, src, grid_w, grid_h)), sa | sb)
                del items[j]
                changed = True
                break
            if changed:
                break
    return [d for d, _ in items]


def merge_tiles(per_tile: list[list[Det]], tiles: list[Tile], grid_w: int, grid_h: int,
                iou: float = 0.5, seam_gap_px: float = 6.0) -> list[Det]:
    """Tile detections -> one list in grid pixels.

    1. Shift into the grid; pieces of one defect cut by tile borders are joined across the
       overlap (join_seam_pieces); NMS removes the duplicates from overlapping tiles.
    2. Elongated classes (scratch, crack) cut by a seam are joined with same-class boxes
       they touch or nearly touch, so a 40 mm scratch comes back as one defect.
    """
    dets = [d for t, ds in zip(tiles, per_tile) for d in to_grid(ds, t, grid_w, grid_h)]
    dets = nms(join_seam_pieces(dets, tiles, grid_w, grid_h, iou), iou)
    elong = {CLASSES.index(c) for c in ELONGATED}
    merged = [d for d in dets if d.cls not in elong]
    for c in sorted(elong):
        group = [d for d in dets if d.cls == c]
        changed = True
        while changed:
            changed = False
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    near = (a.x0 - seam_gap_px <= b.x1 and b.x0 - seam_gap_px <= a.x1
                            and a.y0 - seam_gap_px <= b.y1 and b.y0 - seam_gap_px <= a.y1)
                    if near and (a.cut or b.cut):
                        group[i] = Det(c, max(a.conf, b.conf), min(a.x0, b.x0), min(a.y0, b.y0),
                                       max(a.x1, b.x1), max(a.y1, b.y1), a.tile, a.cut and b.cut)
                        del group[j]
                        changed = True
                        break
                if changed:
                    break
        merged.extend(group)
    return sorted(merged, key=lambda d: -d.conf)
