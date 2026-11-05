"""Plate-level evaluation of a detector on full rectified test grids: the numbers we report.

For each test grid (<grids>/*.png|jpg, e.g. out/grids/test from realdata, or twin grids)
and its sidecar JSON, run trinetra.npu.detect_grid (the same tiling and seam merge as the
Pi) and match the detections to the ground-truth objects (defects and fasteners).

Matching rule (same class only). A detection matches a ground-truth object when
  IoU(det box, gt bbox) >= --match-iou (0.25), or
  the det box covers >= --cover (0.5) of the gt mask pixels (the gt bbox if no mask).
The cover test keeps thin scratches and cracks fair: their true box is a sliver, so a
sensible detection box has a low IoU with it. A mask pixel counts as covered when its
centre lies inside the box (±0.5 px conventions do not matter at these thresholds).

Counting:
- Recall: an object is found when at least one same-class detection matches it.
  "instances" counts every capture; "physical" counts each physical object
  (plate, face, id) once, found when it is found in at least half of its captures.
  Captures of one defect are not independent, so quote the physical Wilson interval.
- Precision: a detection is a true positive when it matches any same-class object
  (several boxes on one object are all TPs and are also counted as duplicates).
  Detections on objects with visible: false are ignored (neither TP nor FP).
- Clean plates: grids with no labelled defect. FP per clean plate = unmatched
  defect-class detections (every class in trinetra.tiles.CLASSES except fastener
  classes, if a model has any) per clean grid; the false-reject rate is the share of
  clean grids with at least one such detection.
- Confusion: per object, the class of the best-overlapping detection of any class under
  the same rule, else "missed"; unmatched detections touching no object go in the
  "background" row.

Writes <out>/eval.json and <out>/eval.md.

Example:
  .venv-train/bin/python training/eval_plates.py --model models/v8n_real.onnx \\
      --grids data/real/out/grids/test --sidecars data/real/out/sidecar --out reports/eval_real
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from trinetra.npu import detect_grid, load_detector
from trinetra.tiles import CLASSES, Det

FASTENER_CLASSES = ("rivet", "open_hole")      # only in the old aero class list; holes are not YOLO classes now
FASTENER_IDS = tuple(i for i, c in enumerate(CLASSES) if c in FASTENER_CLASSES)
DEFECT_IDS = tuple(i for i, c in enumerate(CLASSES) if c not in FASTENER_CLASSES)
IMG_EXT = (".png", ".jpg", ".jpeg")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (n = 0 gives (0, 1))."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "value": k / n if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


# --- Ground truth from a sidecar ---------------------------------------------------------

@dataclass
class Gt:
    key: str                     # physical identity "<plate><face>:<id>"
    cls: int
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 grid px (continuous: pixel j spans [j, j+1])
    mask: np.ndarray             # bool crop
    mx: int                      # crop origin (column, row) in the grid
    my: int
    visible: bool = True
    defect: bool = True


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _cls_id(c: Any) -> int | None:
    """YOLO class id from a name or an id; None for classes the detector does not have."""
    if isinstance(c, (int, np.integer)) or str(c).isdigit():
        return int(c)
    return CLASSES.index(str(c)) if str(c) in CLASSES else None


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def _mask_from_polys(polys: list, shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """Rasterise grid-px polygons (pixel centres on integers) into a cropped bool mask."""
    pts = [np.asarray(p, np.float64).reshape(-1, 2) for p in polys if len(p) >= 2]
    allp = np.concatenate(pts)
    x0, y0 = (np.floor(allp.min(0)) - 1).astype(int)
    x1, y1 = (np.ceil(allp.max(0)) + 2).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(shape[1], x1), min(shape[0], y1)
    m = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8)
    shifted = [np.round((p - [x0, y0]) * 16).astype(np.int32) for p in pts]
    cv2.fillPoly(m, [s for s in shifted if len(s) >= 3], 1, lineType=cv2.LINE_8, shift=4)
    if not m.any():     # sub-pixel sliver (or a 2-point line): keep its outline pixels
        cv2.polylines(m, shifted, True, 1, 1, cv2.LINE_8, shift=4)
    return m.astype(bool), x0, y0


def _mask_from_circle(cx: float, cy: float, r: float, shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    x0, y0 = max(0, int(math.floor(cx - r))), max(0, int(math.floor(cy - r)))
    x1, y1 = min(shape[1], int(math.ceil(cx + r)) + 1), min(shape[0], int(math.ceil(cy + r)) + 1)
    yy, xx = np.mgrid[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, x0, y0


def _mask_from_box(b, shape) -> tuple[np.ndarray, int, int]:
    x0, y0 = max(0, int(math.floor(b[0]))), max(0, int(math.floor(b[1])))
    x1, y1 = min(shape[1], int(math.ceil(b[2]))), min(shape[0], int(math.ceil(b[3])))
    return np.ones((max(1, y1 - y0), max(1, x1 - x0)), bool), x0, y0


def _box_of(mask: np.ndarray, mx: int, my: int) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(mx + xs.min()), float(my + ys.min()), float(mx + xs.max() + 1), float(my + ys.max() + 1))


def plate_key(sc: dict, stem: str) -> str:
    """Physical plate face: plate/coupon (+ face), else the first '_' token of the capture name
    (real "P5A_s03" -> "P5A"; a twin scene "s000042" is its own plate)."""
    plate = _first(sc, "plate", "coupon")
    if plate is None:
        plate = str(_first(sc, "scene_id") or stem).split("_")[0]
    face = sc.get("face") or ""
    return f"{plate}{face}" if not str(plate).endswith(face) else str(plate)


def gt_objects(sc: dict, shape: tuple[int, int], stem: str = "",
               visible: dict[str, bool] | None = None) -> list[Gt]:
    """Ground-truth objects of one grid from its sidecar (defects and fasteners).

    Keys read (first found wins). Defects: cls, id, polys_grid | polys_px | polys (grid px)
    or polys_mm (mapped with H_grid_plate), bbox_grid | bbox_px | bbox, visible, label.
    Fasteners: cls, id, centre_px | center_px | centre_grid | center_grid (or centre_mm |
    center_mm with H_grid_plate), d_px | diameter_px (or d_mm | diameter_mm), or polys/bbox.
    Objects of classes the detector does not have are skipped. visible (id -> bool), when
    given, overrides the sidecar's own visible flag (per-variant or reference visibility).
    """
    H = np.asarray(sc["H_grid_plate"], float) if sc.get("H_grid_plate") is not None else None
    scale = math.sqrt(abs(np.linalg.det(H[:2, :2]))) if H is not None else None   # grid px per mm
    pk = plate_key(sc, stem)
    out = []
    for kind, items in (("fastener", sc.get("fasteners") or []), ("defect", sc.get("defects") or [])):
        for i, o in enumerate(items):
            cls = _cls_id(o.get("cls"))
            if o.get("label") is False or cls is None:
                continue
            polys = _first(o, "polys_grid", "polys_px", "polys")
            if polys is None and o.get("polys_mm") is not None and H is not None:
                polys = [_apply_h(H, np.asarray(p, float)) for p in o["polys_mm"]]
            centre = _first(o, "centre_px", "center_px", "centre_grid", "center_grid")
            d = _first(o, "d_px", "diameter_px")
            if centre is None and H is not None and _first(o, "centre_mm", "center_mm") is not None:
                centre = _apply_h(H, np.asarray([_first(o, "centre_mm", "center_mm")], float))[0]
            if d is None and scale is not None and _first(o, "d_mm", "diameter_mm") is not None:
                d = float(_first(o, "d_mm", "diameter_mm")) * scale
            bbox = _first(o, "bbox_grid", "bbox_px", "bbox")
            if polys:
                mask, mx, my = _mask_from_polys(polys, shape)
            elif centre is not None and d is not None:
                mask, mx, my = _mask_from_circle(float(centre[0]), float(centre[1]), float(d) / 2, shape)
            elif bbox is not None:
                mask, mx, my = _mask_from_box(bbox, shape)
            else:
                raise ValueError(f"{stem}: {kind} {o.get('id', i)} has no polys, centre/diameter or bbox")
            box = _box_of(mask, mx, my) or (tuple(map(float, bbox)) if bbox is not None else None)
            if box is None:
                continue
            if not mask.any():
                mask, mx, my = _mask_from_box(box, shape)
            oid = o.get("id", f"{kind}{i}")
            vis = visible[oid] if visible is not None and oid in visible else o.get("visible", True)
            out.append(Gt(f"{pk}:{oid}", cls, box, mask, mx, my, visible=vis is not False,
                          defect=kind == "defect" and cls in DEFECT_IDS))
    return out


# --- Matching ----------------------------------------------------------------------------

def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def coverage(d: Det, g: Gt) -> float:
    """Share of the gt mask pixels whose centres (j + 0.5, i + 0.5) lie inside the det box."""
    h, w = g.mask.shape
    c0 = max(0, math.ceil(d.x0 - 0.5) - g.mx)
    c1 = min(w, math.floor(d.x1 - 0.5) - g.mx + 1)
    r0 = max(0, math.ceil(d.y0 - 0.5) - g.my)
    r1 = min(h, math.floor(d.y1 - 0.5) - g.my + 1)
    total = int(g.mask.sum())
    if total == 0 or c1 <= c0 or r1 <= r0:
        return 0.0
    return float(g.mask[r0:r1, c0:c1].sum()) / total


def overlap(d: Det, g: Gt, match_iou: float, cover: float) -> float:
    """Match strength (> 0 when the rule matches): max of IoU and coverage, else 0."""
    iou = box_iou((d.x0, d.y0, d.x1, d.y1), g.box)
    cov = coverage(d, g)
    return max(iou, cov) if (iou >= match_iou or cov >= cover) else 0.0


def match_grid(dets: list[Det], gts: list[Gt], match_iou: float = 0.25, cover: float = 0.5) -> dict[str, Any]:
    """Match one grid's detections to its ground truth (see the module docstring)."""
    ov = np.array([[overlap(d, g, match_iou, cover) for g in gts] for d in dets]).reshape(len(dets), len(gts))
    objects = []
    claimed = np.zeros(len(gts), int)
    for j, g in enumerate(gts):
        same = [i for i, d in enumerate(dets) if d.cls == g.cls and ov[i, j] > 0]
        anyc = [i for i in range(len(dets)) if ov[i, j] > 0]
        pred = g.cls if same else (dets[max(anyc, key=lambda i: ov[i, j])].cls if anyc else None)
        objects.append({"key": g.key, "cls": g.cls, "defect": g.defect, "visible": g.visible,
                        "found": bool(same), "pred": pred})
    out_dets = []
    for i, d in enumerate(dets):
        same = [j for j, g in enumerate(gts) if g.cls == d.cls and ov[i, j] > 0]
        vis = [j for j in same if gts[j].visible]
        if vis:
            j = max(vis, key=lambda j: ov[i, j])
            status, gt, dup = "tp", gts[j].key, bool(claimed[j])
            claimed[j] += 1
        elif same:
            status, gt, dup = "ignored", gts[same[0]].key, False
        else:
            status, gt, dup = "fp", None, False
        touches = bool(ov[i].any()) if len(gts) else False
        out_dets.append({"cls": d.cls, "conf": round(d.conf, 4), "box": [round(v, 1) for v in (d.x0, d.y0, d.x1, d.y1)],
                         "status": status, "gt": gt, "dup": dup, "background": status == "fp" and not touches})
    clean = not any(g.defect for g in gts)
    return {"objects": objects, "dets": out_dets, "clean": clean}


def summarize(grids: list[dict[str, Any]], n_classes: int = len(CLASSES), majority: float = 0.5) -> dict[str, Any]:
    """Aggregate per-grid match results into per-class recall/precision, FP rates and confusion."""
    names = list(CLASSES[:n_classes])
    per_class: dict[str, Any] = {}
    hits: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for gr in grids:
        for o in gr["objects"]:
            if o["visible"]:
                hits[(o["cls"], o["key"])].append(o["found"])
    groups = {"all_defects": DEFECT_IDS}
    if FASTENER_IDS:
        groups["all_fasteners"] = FASTENER_IDS
    for c in range(n_classes):
        groups[names[c]] = (c,)
    for gname, ids in groups.items():
        inst = [f for (c, _), v in hits.items() if c in ids for f in v]
        phys = [sum(v) / len(v) >= majority for (c, _), v in hits.items() if c in ids]
        dets = [d for gr in grids for d in gr["dets"] if d["cls"] in ids and d["status"] != "ignored"]
        tp = sum(d["status"] == "tp" for d in dets)
        per_class[gname] = {"recall_physical": rate(sum(phys), len(phys)),
                            "recall_instances": rate(sum(inst), len(inst)),
                            "precision": rate(tp, len(dets)),
                            "duplicates": sum(d["dup"] for d in dets)}
    clean = [gr for gr in grids if gr["clean"]]
    fp_clean = [sum(d["status"] == "fp" and d["cls"] in DEFECT_IDS for d in gr["dets"]) for gr in clean]
    labels = names + ["missed"]
    conf = np.zeros((n_classes + 1, n_classes + 1), int)       # rows: true classes + background
    for gr in grids:
        for o in gr["objects"]:
            if o["visible"]:
                conf[o["cls"], o["pred"] if o["pred"] is not None else n_classes] += 1
        for d in gr["dets"]:
            if d["background"]:
                conf[n_classes, d["cls"]] += 1
    return {
        "per_class": per_class,
        "n_grids": len(grids), "n_clean_grids": len(clean),
        "fp_per_clean_plate": (sum(fp_clean) / len(clean)) if clean else None,
        "false_reject_rate": rate(sum(n > 0 for n in fp_clean), len(clean)),
        "confusion": {"rows": names + ["background"], "cols": labels, "matrix": conf.tolist()},
    }


# --- I/O ---------------------------------------------------------------------------------

def find_sidecar(img: Path, sidecars: Path) -> Path | None:
    stem = img.stem
    scene = re.sub(r"_v\d+$", "", stem)            # twin variants <scene>_v<k> share <scene>.json
    for cand in (sidecars / f"{stem}.json", img.with_suffix(".json"), sidecars / f"{scene}.json"):
        if cand.exists():
            return cand
    return None


def variant_index(stem: str) -> int | None:
    """k of a twin grid named <scene>_v<k>, else None."""
    m = re.search(r"_v(\d+)$", stem)
    return int(m.group(1)) if m else None


def variant_entry(sc: dict, k: int | None) -> dict:
    """The sidecar's variants[] entry with that k ({} if none)."""
    if k is None:
        return {}
    for v in sc.get("variants") or []:
        if v.get("k") == k:
            return v
    return {}


def variant_visible(sc: dict, stem: str) -> dict[str, bool] | None:
    """Per-variant visibility (variants[k].visible) for a twin grid, else None (use the top-level flag)."""
    vis = variant_entry(sc, variant_index(stem)).get("visible")
    return {str(i): bool(b) for i, b in vis.items() if b is not None} if isinstance(vis, dict) else None


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def evaluate(detector, grids_dir: Path, sidecars: Path, match_iou: float = 0.25, cover: float = 0.5,
             tile: int = 640, min_overlap: int = 128) -> dict[str, Any]:
    images = sorted(p for p in Path(grids_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"no grid images in {grids_dir}")
    results = []
    for img in images:
        sc_path = find_sidecar(img, Path(sidecars))
        if sc_path is None:
            raise SystemExit(f"no sidecar for {img.name} in {sidecars}")
        rgb = load_rgb(img)
        sc = json.loads(sc_path.read_text())
        gts = gt_objects(sc, rgb.shape[:2], img.stem, variant_visible(sc, img.stem))
        dets = detect_grid(rgb, detector, tile, min_overlap)
        r = match_grid(dets, gts, match_iou, cover)
        r["image"] = img.name
        results.append(r)
    summary = summarize(results)
    summary["rule"] = {"match_iou": match_iou, "cover": cover, "physical_majority": 0.5}
    summary["grids"] = results
    return summary


def fmt_rate(r: dict[str, Any]) -> str:
    if not r["n"]:
        return "n/a"
    return f"{r['k']}/{r['n']} {r['value']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"


def to_markdown(s: dict[str, Any], title: str = "Plate-level evaluation") -> str:
    lines = [f"# {title}", "",
             f"{s['n_grids']} grids ({s['n_clean_grids']} clean). Match: same class and IoU >= "
             f"{s['rule']['match_iou']} or box covers >= {s['rule']['cover']:.0%} of the true mask. "
             "Intervals are Wilson 95 %.", "",
             "| Class | Recall (physical) | Recall (captures) | Precision | Duplicates |", "|---|---|---|---|---|"]
    for name, r in s["per_class"].items():
        lines.append(f"| {name} | {fmt_rate(r['recall_physical'])} | {fmt_rate(r['recall_instances'])} | "
                     f"{fmt_rate(r['precision'])} | {r['duplicates']} |")
    fpc = s["fp_per_clean_plate"]
    lines += ["", f"False positives per clean plate (defect classes): {'n/a' if fpc is None else f'{fpc:.2f}'}; "
              f"false-reject rate: {fmt_rate(s['false_reject_rate'])}.", "",
              "Confusion (rows: truth, columns: best-overlapping detection):", "",
              "| truth \\ pred | " + " | ".join(s["confusion"]["cols"]) + " |",
              "|---" * (len(s["confusion"]["cols"]) + 1) + "|"]
    for name, row in zip(s["confusion"]["rows"], s["confusion"]["matrix"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".onnx (any venv), .pt (training venv) or .hef (Pi)")
    ap.add_argument("--grids", required=True, type=Path, help="directory of rectified test grids")
    ap.add_argument("--sidecars", type=Path, help="sidecar directory (default: next to the grids)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.25)
    ap.add_argument("--cover", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--min-overlap", type=int, default=128)
    a = ap.parse_args(argv)
    det = load_detector(a.model, conf=a.conf)
    s = evaluate(det, a.grids, a.sidecars or a.grids, a.match_iou, a.cover, a.tile, a.min_overlap)
    s.update(model=str(a.model), grids_dir=str(a.grids), conf=a.conf)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "eval.json").write_text(json.dumps(s, indent=1))
    (a.out / "eval.md").write_text(to_markdown(s, f"Plate-level evaluation: {Path(a.model).name}"))
    print(to_markdown(s))
    return s


if __name__ == "__main__":
    main()
