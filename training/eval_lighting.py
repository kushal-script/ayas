"""The lighting experiment: does the inspector work in different lighting? (DESIGN_STEEL G).

Inputs come from ayas.twin.matrix (plans + manifest) rendered and post-processed with
post.eval_arms [burst, naive]:
  <matrix>/matrix_manifest.json                     plate -> rows (condition, device_mode)
  <matrix>/grids/<arm>/<split>/<scene>_v<k>.png     full rectified grids, one per arm and variant
  <matrix>/sidecar/<scene>.json                     labels, condition, device_mode, per-variant gate

For each model x arm x (condition, device_mode) row, on the full grids (ayas.npu
detect_grid, the same tiling and seam merge as the Pi), matched to the sidecar labels with
the eval_plates rule (same class; IoU >= 0.25 or the box covers >= 50 % of the mask):
- precision, recall and F1 of the defects, Wilson 95 % intervals (F1: the Wilson interval
  of TP / (TP + (FP + FN) / 2), an approximation). Retaken captures are left out: the
  device refused them and showed a message, so nothing was inspected;
- false positives per clean plate;
- retake rate: the gate said RETAKE, with the gate reasons counted (retake_reasons) and the
  most common one (main_reason). The naive arm (the old method) has no gate, so it never
  retakes, unless the sidecar stores a gate for that arm;
- verdicts: predicted = rules.judge_steel on the detections, ground truth = judge_steel on
  the ground-truth defects. --measure box (default): both use the same box -> mm measurement,
  with every box padded to label.min_box_px per side as the training labels are (a perfect
  detector then gets the ground-truth verdict), so the comparison isolates what the lighting
  does to detection. A box only bounds a thin defect: a diagonal 0.1 mm scratch measures as
  wide as its box, on both sides. --measure app: detections are measured exactly as the
  on-device app does (ayas.app.pipeline.detection_record: sizes from the defect's own
  pixels in the grid image, so lighting can change them; plate-edge distance only) and the
  ground truth defaults to the analytic measures (--gt-measure true: the sidecar's record
  fields): the verdict the device would really show against the physical truth;
- silent-failure rate: the gate let the capture through (PASS or PASS_FLAGGED) but the
  verdict differs from the ground truth. "unsafe" is the subset where the device's verdict
  is milder and not a hold for a person (e.g. PASS on a part that needs REWORK; HOLD_FOR_QA
  on a REJECT part is wrong but not unsafe). A RETAKE is never a silent failure;
- verdict agreement with the ground truth and with the same plate's C0 (dark room) capture of
  the same device (the accepted C0 capture with the same k, else the plate's most common
  accepted C0 verdict): hood rows against C0 with the hood, open rows against C0 open;
- SSIM of the rectified grid against the C0 grid of the same plate, arm and device (luma,
  Gaussian window, inside the plate, after mapping both through H_grid_plate), over accepted
  captures like P/R/F1 (ssim_c0_all: every capture); mad_c0 is the direct invariance number
  on the same pairs: mean |grid - grid_C0| over the plate's pixels and RGB channels, in 8-bit DN.

Oracle detector (--model oracle, or NAME=oracle): a perfect detector on what each grid shows. Per
capture it returns the label boxes (padded to label.min_box_px) of the labelled defects post judged
visible in that arm's own grid (per-grid sidecar visible_arm), cut per tile like a model's output,
so detect_grid's seam merge, the matching and the rules run unchanged. It bounds what the images
allow (a defect the lighting washes out is a miss); it is not a model result and makes no false
positives.

Ground-truth visibility (--gt-visibility): c0 (default) = a defect counts when post found it
visible in the plate's C0 capture with the same device (the ring and the pods show different
defects), so a defect that glare hides in sun counts as a miss; own = the capture's own
visibility; all = every labelled defect.

PASS criterion (DESIGN_STEEL G, device v2, 14 Nov), per model and arm:
1. HOOD (device v2, the hood head with its LED ring: the product configuration), every
   condition C0-C6: retake rate <= 5 %, F1 >= F1(C0, hood) - 0.05, verdict agreement with
   C0 (hood) >= 95 % and 0 silent failures;
2. OPEN (the handheld without the hood): 0 silent failures in every condition. Retakes are
   allowed; they are reported per condition with the main gate reason (status INFO).
A hood row with no accepted capture fails. Missing hood rows (or C0 open) make the result
INCOMPLETE. Every table keeps the naive arm (the old method) beside the burst arm.

Writes <out>/lighting_matrix.json, <out>/lighting_matrix.md and per-plate montages
<out>/figures/lighting_<plate>.png (columns: rows of the matrix; rows: arms; green = ground
truth, red = detections).

Example (dev or training venv):
  .venv/bin/python training/eval_lighting.py --matrix data/twin_matrix \\
      --model naive=models/v8n_naive.onnx --model burst=models/v8n_burst.onnx \\
      --spec configs/spec_steel.yaml --out reports
  ... --measure app --out reports/device_measures     # the verdicts the device would show
  .venv/bin/python training/eval_lighting.py --matrix data/twin_pilot --model oracle --out reports/oracle
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

import eval_plates as EP
from ayas.npu import detect_grid, load_detector
from ayas.tiles import CLASSES, ELONGATED, Det, clip_box, tile_grid
from ayas.twin import matrix as MX

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ayas.lighting_matrix/1"
ARMS = ("naive", "burst")
REFERENCE = ("C0_dark", "open")
HOOD_REFERENCE = ("C0_dark", "hood")
PRODUCT = "hood"                  # device v2: the hood head (LED ring) must work in every condition
F1_DROP_MAX = 0.05
AGREE_MIN = 0.95
HOOD_RETAKE_MAX = 0.05            # the product may refuse at most 5 % of captures in any condition
# capture.yaml gate.messages order: a refused capture's main reason is the one whose message it showed
REASON_ORDER = ("hood_leak", "too_bright", "glare_ambient", "saturated", "too_dark", "motion", "blur", "changed", "invalid",
                "no_plate")
MIN_BOX_PX = 8.0                  # configs/twin.yaml label.min_box_px: training boxes are padded to this
MEASURES = ("box", "app")         # detections -> mm: padded box (both sides alike) | the on-device app's measures
RETAKE = "RETAKE"
VERDICT_ORDER = ("PASS", "CLEAN_AND_REINSPECT", "REWORK", "HOLD_FOR_QA", "HOLD_FOR_MRB", "REJECT")   # worst last
NAMES = {"C0_dark": "dark room", "C1_office": "office ceiling light", "C2_warm_bulb": "warm bulb close by",
         "C3_window": "window daylight", "C4_side_lamp": "low side lamp with a hand shadow",
         "C5_reflected_lamp": "a lamp reflected in the plate", "C6_sun": "direct sun"}
IN = {"C0_dark": "in the dark room", "C1_office": "under office ceiling light", "C2_warm_bulb": "under a warm bulb close by",
      "C3_window": "in window daylight", "C4_side_lamp": "under a low side lamp with a hand shadow",
      "C5_reflected_lamp": "with a lamp reflected in the plate", "C6_sun": "in direct sun"}
ARM_NAMES = {"naive": "Old method (one auto-exposed frame through the ISP)",
             "burst": "New method (5-frame LED burst, ambient subtraction, L/R combination, quality gate)"}
Judge = Callable[[list, dict, dict], tuple]


def criterion() -> dict[str, Any]:
    return {"reference": {"hood": list(HOOD_REFERENCE), "open": list(REFERENCE)}, "product": PRODUCT,
            "f1_drop_max": F1_DROP_MAX, "agreement_min": AGREE_MIN, "hood_retake_max": HOOD_RETAKE_MAX,
            "silent_max": 0,
            "text": ("HOOD (device v2, the product configuration), every condition C0-C6: retake rate <= 5 %, "
                     "F1 >= F1(C0, hood) - 0.05, verdict agreement with C0 (hood) >= 95 % and 0 silent failures. "
                     "OPEN (handheld without the hood): 0 silent failures in every condition; retakes allowed and "
                     "reported per condition with the main gate reason. SSIM is against C0 of the same device.")}


def reference_of(device_mode: str) -> tuple[str, str]:
    """The dark-room row a row is compared with: C0 with the same device."""
    return ("C0_dark", device_mode)


# --- Sidecar fields ----------------------------------------------------------------------

@dataclass
class Capture:
    arm: str
    scene: str
    k: int | None
    plate: str
    condition: str
    device_mode: str
    image: Path
    sidecar: dict
    gate: str
    gate_message: str
    lux: float | None
    part: dict
    gate_reasons: tuple[str, ...] = ()

    @property
    def row(self) -> tuple[str, str]:
        return (self.condition, self.device_mode)


def tag(condition: str, device_mode: str) -> str:
    return MX.row_tag(condition, device_mode)


def variant(sc: dict, k: int | None, arm: str | None = None) -> dict:
    """variants[] entry k, preferring one marked with this arm, else one with no arm."""
    vs = [v for v in sc.get("variants") or [] if v.get("k") == k] if k is not None else []
    for v in vs:
        if arm is not None and v.get("arm") == arm:
            return v
    return next((v for v in vs if v.get("arm") in (None, arm)), vs[0] if vs else {})


def _pick(*vals: Any) -> Any:
    return next((v for v in vals if v is not None), None)


def row_of(sc: dict, k: int | None, arm: str, info: dict) -> tuple[str | None, str | None]:
    v = variant(sc, k, arm)
    m = sc.get("matrix") or {}
    cond = _pick(sc.get("condition"), v.get("condition"), m.get("condition"),
                 (sc.get("ambient") or {}).get("condition"), info.get("condition"))
    mode = _pick(sc.get("device_mode"), v.get("device_mode"), m.get("device_mode"),
                 (sc.get("device") or {}).get("mode"), info.get("device_mode"))
    return cond, mode


def gate_of(sc: dict, k: int | None, arm: str) -> tuple[str, str]:
    """(verdict, message) of the capture's quality gate.

    Read from, in order: a per-grid sidecar of this arm (post writes grids/<arm>/<split>/<stem>.json
    with top-level gate, gate_message, gated), variants[k].arms[arm], then variants[k].gate for any
    arm but naive. The naive arm (the old method) has no gate unless the sidecar says it was
    gated: PASS. A burst capture without gate info counts as PASS (more silent failures, never fewer).
    """
    if sc.get("arm") == arm and sc.get("gate") is not None:
        rec = sc
    else:
        v = variant(sc, k, arm)
        rec = (v.get("arms") or {}).get(arm) or (v if arm != "naive" else {})
    g = rec.get("gate")
    if g is None or rec.get("gated") is False:
        return "PASS", ""
    if isinstance(g, str):
        return g, str(rec.get("gate_message") or rec.get("message") or "")
    return str(g.get("verdict") or "PASS"), str(g.get("message") or "")


def gate_reasons_of(sc: dict, k: int | None, arm: str) -> tuple[str, ...]:
    """The gate reasons (lightproc keys: too_bright, saturated, motion, ...) of a capture, read
    from the same places as gate_of; () when none are stored."""
    if sc.get("arm") == arm and sc.get("gate") is not None:
        rec = sc
    else:
        v = variant(sc, k, arm)
        rec = (v.get("arms") or {}).get(arm) or (v if arm != "naive" else {})
    if rec.get("gated") is False:
        return ()
    g = rec.get("gate")
    rs = rec.get("gate_reasons", rec.get("reasons"))
    if rs is None and isinstance(g, dict):
        rs = g.get("reasons")
    return tuple(str(x) for x in rs or ())


def plate_of(sc: dict, info: dict) -> str:
    plate = sc.get("plate") if isinstance(sc.get("plate"), str) else None
    return str(_pick(info.get("plate"), (sc.get("matrix") or {}).get("plate"), plate)
               or str(sc.get("scene_id") or "").split("_")[0])


def part_for(sc: dict, info: dict, plans_dir: Path | None) -> dict:
    if info.get("part"):
        return dict(info["part"])
    if isinstance(sc.get("part"), dict):
        return dict(sc["part"])
    if plans_dir is not None:
        p = Path(plans_dir) / f"{sc.get('scene_id')}.json"
        if p.exists():
            return MX.part_of(json.loads(p.read_text()))
    return {"material": None, "thickness_mm": None, "size_mm": sc.get("plate_size_mm")}


def collect(grids_root: Path, sidecars: Path, arms: Sequence[str] = ARMS, index: dict | None = None,
            plans_dir: Path | None = None) -> list[Capture]:
    """Every grid of the given arms with its sidecar, row, gate and part."""
    index = index or {}
    caps, cache = [], {}
    for arm in arms:
        d = Path(grids_root) / arm
        if not d.is_dir():
            print(f"note: no grids for arm {arm} in {d}")
            continue
        for img in sorted(p for p in d.rglob("*") if p.suffix.lower() in EP.IMG_EXT):
            sp = EP.find_sidecar(img, Path(sidecars))
            if sp is None:
                raise SystemExit(f"no sidecar for {img} in {sidecars}")
            if sp not in cache:
                cache[sp] = json.loads(sp.read_text())
            sc = cache[sp]
            k = EP.variant_index(img.stem)
            scene = str(sc.get("scene_id") or re.sub(r"_v\d+$", "", img.stem))
            info = index.get(scene, {})
            cond, mode = row_of(sc, k, arm, info)
            if cond is None or mode is None:
                raise SystemExit(f"{img}: the sidecar has no condition/device_mode and the manifest does not list {scene}")
            gate, msg = gate_of(sc, k, arm)
            v = variant(sc, k, arm)
            lux = _pick(v.get("lux"), sc.get("lux"))
            caps.append(Capture(arm, scene, k, plate_of(sc, info), cond, mode, img, sc, gate, msg,
                                None if lux is None else float(lux), part_for(sc, info, plans_dir),
                                gate_reasons_of(sc, k, arm)))
    return caps


# --- Visibility, measures and verdicts ---------------------------------------------------

def visibility_map(sc: dict, k: int | None) -> dict[str, bool]:
    """Defect id -> visible in this capture: a per-grid sidecar's visible_arm, else variants[k].visible,
    else the defect's own flag."""
    out = {str(d.get("id")): _pick(d.get("visible_arm"), d.get("visible")) is not False
           for d in sc.get("defects") or []}
    vis = variant(sc, k).get("visible")
    if isinstance(vis, dict):
        out.update({str(i): bool(b) for i, b in vis.items() if b is not None})
    return out


def reference_visibility(caps: Sequence[Capture]) -> dict[tuple[str, str], dict[str, bool]]:
    """Per (plate, device mode), the defects post found visible in any C0 capture with that device
    (the ring and the pods light different defects); 'any' holds the union over both, for a
    device mode without a C0 row."""
    ref: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for c in caps:
        if c.condition == REFERENCE[0]:
            for key in ((c.plate, c.device_mode), (c.plate, "any")):
                for i, b in visibility_map(c.sidecar, c.k).items():
                    ref[key][i] = ref[key].get(i, False) or b
    return dict(ref)


def grid_frame(sc: dict) -> tuple[np.ndarray, float, tuple[float, float, float, float]]:
    """(H_grid<-plate, mm per grid px, plate rectangle x0, y0, x1, y1 in grid px) of a sidecar.

    The grid is image-aligned (DESIGN 1), so the plate is an axis-aligned rectangle in it.
    """
    H = np.asarray(sc["H_grid_plate"], float)
    mpp = (sc.get("grid") or {}).get("mm_per_px") or 1.0 / math.sqrt(abs(np.linalg.det(H[:2, :2])))
    size = sc.get("plate_size_mm") or (sc.get("part") or {}).get("size_mm")
    if size is None:
        raise ValueError("sidecar has no plate_size_mm")
    W, Hh = float(size[0]), float(size[1])
    c = EP._apply_h(H, np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], float))
    return H, float(mpp), (float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max()))


def box_size(cls_name: str, w_mm: float, h_mm: float) -> dict[str, float]:
    """mm measures of a box, every key the steel rules may ask for (depth is never measured)."""
    length = math.hypot(w_mm, h_mm) if cls_name in ELONGATED else max(w_mm, h_mm)
    size = {"length_mm": length, "width_mm": min(w_mm, h_mm), "diameter_mm": max(w_mm, h_mm),
            "area_mm2": math.pi / 4 * w_mm * h_mm}
    if cls_name in ("corrosion", "stain"):
        size["diameter_mm"] = math.sqrt(max(0.0, w_mm * h_mm))
    if cls_name == "pit":
        size["count"] = 1
    return {k: round(v, 3) for k, v in size.items()}


def _holes_px(sc: dict, H: np.ndarray, mpp: float) -> list[tuple[float, float, float]]:
    """(cx, cy, r) in grid px of the part's holes listed in the sidecar (features, holes, fasteners)."""
    out = []
    for f in list(sc.get("features") or []) + list(sc.get("holes") or []) + list(sc.get("fasteners") or []):
        d = f.get("hole_d_mm") or f.get("d_mm")
        if f.get("centre_px") is not None:
            cx, cy = map(float, f["centre_px"][:2])
        elif f.get("u_mm", f.get("u")) is not None and f.get("v_mm", f.get("v")) is not None:
            cx, cy = EP._apply_h(H, np.array([[f.get("u_mm", f.get("u")), f.get("v_mm", f.get("v"))]], float))[0]
        else:
            continue
        r = float(d) / 2 / mpp if d is not None else (float(f["d_px"]) / 2 if f.get("d_px") is not None else None)
        if r is not None:
            out.append((float(cx), float(cy), r))
    return out


def pad_box(box: Sequence[float], min_px: float) -> tuple[float, float, float, float]:
    """A box grown about its centre to at least min_px per side (post._pad_box: the label rule)."""
    x0, y0, x1, y1 = map(float, box)
    if x1 - x0 < min_px:
        c = 0.5 * (x0 + x1)
        x0, x1 = c - 0.5 * min_px, c + 0.5 * min_px
    if y1 - y0 < min_px:
        c = 0.5 * (y0 + y1)
        y0, y1 = c - 0.5 * min_px, c + 0.5 * min_px
    return x0, y0, x1, y1


def box_defect(cls_name: str, conf: float, box: Sequence[float], sc: dict, did: str,
               min_box_px: float = MIN_BOX_PX) -> dict[str, Any]:
    """A record 3.0 defect dict from a grid-px box, measured as the Pi does.

    The box is first padded to min_box_px per side, the rule post applies to every training
    label (label.min_box_px), so a detector trained on those labels and the ground truth are
    measured alike: a box side below that size carries no width information. location_mm is
    grid-aligned plate mm (config.Grid.px_to_mm: the image-aligned plate frame, origin at the
    plate corner nearest the grid origin); near_edge_mm is the distance from the box to the
    nearest plate edge or hole edge.
    """
    H, mpp, (gx0, gy0, gx1, gy1) = grid_frame(sc)
    x0, y0, x1, y1 = pad_box(box, min_box_px)
    edge = max(0.0, min(x0 - gx0, gx1 - x1, y0 - gy0, gy1 - y1)) * mpp
    for cx, cy, r in _holes_px(sc, H, mpp):
        gap = (math.hypot(max(x0 - cx, 0.0, cx - x1), max(y0 - cy, 0.0, cy - y1)) - r) * mpp
        edge = min(edge, max(0.0, gap))
    return {"id": did, "type": cls_name, "confidence": round(float(conf), 4),
            "box_px": [round(float(v), 1) for v in box],
            "location_mm": {"x": round(((x0 + x1) / 2 - gx0) * mpp, 2), "y": round(((y0 + y1) / 2 - gy0) * mpp, 2)},
            "size": box_size(cls_name, (x1 - x0) * mpp, (y1 - y0) * mpp),
            "near_edge_mm": round(edge, 2)}


def pred_defects(dets: Sequence[Det], sc: dict, min_box_px: float = MIN_BOX_PX) -> list[dict[str, Any]]:
    return [box_defect(CLASSES[d.cls], d.conf, (d.x0, d.y0, d.x1, d.y1), sc, f"P{i + 1}", min_box_px)
            for i, d in enumerate(dets) if 0 <= d.cls < len(CLASSES)]


def grid_of(sc: dict):
    """config.Grid of a sidecar's image-aligned plate grid (grid.plate_mm / border_mm, else from H_grid_plate)."""
    from ayas.config import Grid

    _, mpp, (x0, y0, x1, y1) = grid_frame(sc)
    g = sc.get("grid") or {}
    pw, ph = g.get("plate_mm") or ((x1 - x0) * mpp, (y1 - y0) * mpp)
    border = g.get("border_mm")
    return Grid(float(pw), float(ph), border_mm=float(x0 * mpp if border is None else border), mm_per_px=mpp)


def app_defects(dets: Sequence[Det], sc: dict, rgb: np.ndarray) -> list[dict[str, Any]]:
    """Record 3.0 defects measured as the on-device app measures them (ayas.app.pipeline.
    detection_record on this grid image: mask-based sizes, near_edge_mm to the plate edge)."""
    from ayas.app import pipeline as AP

    grid = grid_of(sc)
    gm = AP.GridMap(grid, np.eye(3), np.zeros((4, 2)))
    plate = plate_mask(sc, rgb.shape[:2], inset_mm=0.0)
    out = []
    for i, d in enumerate(dets):
        if 0 <= d.cls < len(CLASSES):
            rec = AP.detection_record(d, rgb, gm, plate)
            out.append({k: v for k, v in rec.items() if k != "tile"} | {"id": f"P{i + 1}"})
    return out


def gt_defects(gts: Sequence[EP.Gt], sc: dict, measure: str = "box", min_box_px: float = MIN_BOX_PX,
               part: dict | None = None, plan: dict | None = None) -> list[dict[str, Any]]:
    """Visible labelled ground-truth defects as record 3.0 defects (confidence 1).

    measure="box": measured from the ground-truth mask box exactly like a detection, padded to
    min_box_px like the training labels (isolates what the lighting does to detection: a
    perfect detector gets the ground-truth verdict); "true": the analytic measures, converted
    exactly as the canonical ground-truth verdict does (twin.records.gt_verdict ->
    defects_from_labels: the sidecar's record fields plus the plan's defect params, e.g.
    pitted rust, pit counts); without twin.records, the sidecar's record fields alone.
    """
    by_id = {str(d.get("id")): d for d in sc.get("defects") or []}
    from_label = None
    if measure == "true":
        try:
            from ayas.twin.records import defect_from_label as from_label
        except ImportError:
            pass
    plan_defects = {str(d["id"]): d for d in (plan or {}).get("defects") or [] if "id" in d}
    out = []
    for g in gts:
        if not (g.defect and g.visible):
            continue
        did = g.key.split(":", 1)[1]
        rec = box_defect(CLASSES[g.cls], 1.0, g.box, sc, did, min_box_px)
        src = by_id.get(did) or {}
        if from_label is not None and src:
            lbl = dict(src)
            if all(lbl.get(k) is None for k in ("location_mm", "centre_mm", "polys_mm", "u_mm", "bbox_mm")):
                lbl["location_mm"] = rec["location_mm"]           # never drop a defect for want of a position
            if lbl.get("near_edge_mm") is None and not lbl.get("near"):
                lbl["near_edge_mm"] = rec["near_edge_mm"]
            d = from_label(lbl, 1.0, len(out) + 1, part, (plan_defects.get(did) or {}).get("params"))
            if d is not None:
                out.append(d)
                continue
        if measure == "true":
            size = src.get("size") or src.get("measure") or {}
            if size:
                rec["size"] = {k: v for k, v in size.items() if isinstance(v, (int, float)) and k != "depth_mm"}
            if isinstance(src.get("near_edge_mm"), (int, float)):
                rec["near_edge_mm"] = float(src["near_edge_mm"])
            else:
                near = [v for v in (src.get("near") or {}).values() if isinstance(v, (int, float))]
                if near:
                    rec["near_edge_mm"] = round(max(0.0, min(near)), 2)
            if isinstance(src.get("location_mm"), dict):
                rec["location_mm"] = dict(src["location_mm"])
        out.append(rec)
    return out


def default_judge() -> Judge:
    from ayas import rules

    if not hasattr(rules, "judge_steel"):
        raise SystemExit("ayas.rules.judge_steel is missing (the spec-advisor workstream adds it)")
    return rules.judge_steel


def default_spec(path: Path = ROOT / "configs/spec_steel.yaml") -> dict:
    """configs/spec_steel.yaml (rules.load_spec). An empty spec would make judge_steel hold every
    defect as 'unknown', so evaluate_matrix never runs the real rules without one."""
    from ayas import rules

    return rules.load_spec(path)


def label_min_box(path: Path = ROOT / "configs/twin.yaml") -> float:
    """label.min_box_px of the twin config (the padding of every training box), else MIN_BOX_PX."""
    try:
        import yaml

        return float(((yaml.safe_load(Path(path).read_text()) or {}).get("label") or {}).get("min_box_px", MIN_BOX_PX))
    except (OSError, ValueError, TypeError):
        return MIN_BOX_PX


def verdict_order() -> tuple[str, ...]:
    try:
        from ayas.rules import VERDICTS
    except ImportError:
        return VERDICT_ORDER
    return tuple(VERDICTS)


def less_severe(pred: str, gt: str, order: Sequence[str] = VERDICT_ORDER) -> bool:
    """True when the device's verdict is milder than the truth (unknown verdicts count as milder)."""
    if pred not in order or gt not in order:
        return True
    return order.index(pred) < order.index(gt)


def unsafe(pred: str, gt: str, order: Sequence[str] = VERDICT_ORDER) -> bool:
    """A wrong verdict that lets the part go on: milder than the truth and not a hold for a person."""
    return pred != gt and not str(pred).startswith("HOLD") and less_severe(pred, gt, order)


# --- SSIM --------------------------------------------------------------------------------

def luma_u8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean SSIM (Wang et al. 2004: 11 x 11 Gaussian window, sigma 1.5, 8-bit constants) inside mask."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    if mask is not None and mask.any():
        return float(s[mask].mean())
    return float(s.mean())


def plate_mask(sc: dict, shape: tuple[int, int], inset_mm: float = 0.5) -> np.ndarray:
    """The plate (minus a small inset) in grid pixels."""
    _, mpp, (x0, y0, x1, y1) = grid_frame(sc)
    d = inset_mm / mpp
    m = np.zeros(shape, bool)
    m[max(0, math.ceil(y0 + d)) : max(0, math.floor(y1 - d)), max(0, math.ceil(x0 + d)) : max(0, math.floor(x1 - d))] = True
    return m


def align_to(img: np.ndarray, sc_src: dict, sc_ref: dict, shape: tuple[int, int]) -> np.ndarray:
    """Map a grid into the reference grid's pixels through the plate frame (identity when they agree)."""
    H = np.asarray(sc_ref["H_grid_plate"], float) @ np.linalg.inv(np.asarray(sc_src["H_grid_plate"], float))
    H /= H[2, 2]
    if img.shape[:2] == tuple(shape) and np.allclose(H, np.eye(3), atol=1e-6):
        return img
    return cv2.warpPerspective(img, H, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)


def mean_abs_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean |a - b| in 8-bit DN over the mask's pixels and every channel (the direct invariance number)."""
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    d = d.reshape(*d.shape[:2], -1).mean(axis=2)
    return float(d[mask].mean()) if mask is not None and mask.any() else float(d.mean())


def reference_scores(caps: Sequence[Capture]) -> dict[tuple, dict[str, float] | None]:
    """(arm, scene, k) -> {ssim, mad} of the grid against the same plate's C0 grid with the same
    device (same arm, same k if present): SSIM on luma and mean |grid - grid_C0| (RGB, DN), both
    inside the plate. An accepted C0 capture is preferred as the reference; a refused one is used
    only when the plate has no other; None without a C0 of that device."""
    good: dict[tuple, Capture] = {}
    anyc: dict[tuple, Capture] = {}
    for c in caps:
        if c.condition == REFERENCE[0]:
            for key in ((c.arm, c.device_mode, c.plate, c.k), (c.arm, c.device_mode, c.plate, None)):
                anyc.setdefault(key, c)
                if c.gate != RETAKE:
                    good.setdefault(key, c)
    cache: dict[Path, np.ndarray] = {}

    def rgb(p: Path) -> np.ndarray:
        if p not in cache:
            if len(cache) > 32:
                cache.clear()
            cache[p] = EP.load_rgb(p)
        return cache[p]

    out = {}
    for c in caps:
        m = c.device_mode
        r = (good.get((c.arm, m, c.plate, c.k)) or good.get((c.arm, m, c.plate, None))
             or anyc.get((c.arm, m, c.plate, c.k)) or anyc.get((c.arm, m, c.plate, None)))
        if r is None:
            out[(c.arm, c.scene, c.k)] = None
            continue
        b = rgb(r.image)
        a = align_to(rgb(c.image), c.sidecar, r.sidecar, b.shape[:2])
        mask = plate_mask(r.sidecar, b.shape[:2])
        out[(c.arm, c.scene, c.k)] = {"ssim": round(ssim(luma_u8(a), luma_u8(b), mask), 4),
                                      "mad": round(mean_abs_diff(a, b, mask), 3)}
    return out


def ssim_vs_reference(caps: Sequence[Capture]) -> dict[tuple, float | None]:
    """(arm, scene, k) -> SSIM of the grid against the same plate's C0 grid with the same device
    (reference_scores' ssim)."""
    return {k: (None if v is None else v["ssim"]) for k, v in reference_scores(caps).items()}


# --- Oracle detector ---------------------------------------------------------------------

ORACLE = "oracle"


class OracleDetector:
    """A perfect detector on what each grid shows (see the module docstring).

    inspect() calls prime(sidecar, shape, tile, min_overlap) before each grid: the boxes are the
    labelled defects whose visible_arm (else visible) is true in that per-grid sidecar, padded to
    min_box_px like the training labels; detect(tiles) returns them clipped to each tile.
    """

    names = list(CLASSES)

    def __init__(self, min_box_px: float = MIN_BOX_PX, conf: float = 0.9):
        self.min_box_px, self.conf = float(min_box_px), float(conf)
        self.boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        self.tiles: list = []

    def prime(self, sc: dict, shape: Sequence[int], tile: int = 640, min_overlap: int = 128) -> None:
        self.tiles = tile_grid(int(shape[1]), int(shape[0]), tile, min_overlap)
        self.boxes = []
        for d in sc.get("defects") or []:
            seen = _pick(d.get("visible_arm"), d.get("visible"))
            box = _pick(d.get("bbox_grid"), d.get("bbox_px"), d.get("bbox"))
            if d.get("label", True) and seen and box is not None and d.get("cls") in CLASSES:
                self.boxes.append((CLASSES.index(d["cls"]), pad_box(box, self.min_box_px)))

    def detect(self, tiles: Sequence[np.ndarray]) -> list[list[Det]]:
        if len(tiles) != len(self.tiles):
            raise RuntimeError("OracleDetector.detect: prime() was not called for this grid")
        return [[Det(cls, self.conf, *loc) for cls, box in self.boxes if (loc := clip_box(box, t)) is not None]
                for t in self.tiles]


# --- Evaluation --------------------------------------------------------------------------

def f1_rate(tp: int, fp: int, fn: int) -> dict[str, Any]:
    den = tp + 0.5 * (fp + fn)
    lo, hi = EP.wilson(tp, den) if den > 0 else (0.0, 1.0)
    return {"tp": tp, "fp": fp, "fn": fn, "value": (tp / den) if den > 0 else None,
            "ci95": [round(lo, 4), round(hi, 4)]}


def link_c0(out: list[dict[str, Any]]) -> None:
    """Set each capture's c0_verdict: the verdict of the same plate's accepted C0 capture with the
    same device and k, else that plate and device's most common accepted C0 verdict; RETAKE
    when every such C0 capture was refused (not comparable), None when there is none."""
    same_k: dict[tuple, str] = {}
    per_plate: dict[tuple, list[str]] = defaultdict(list)
    refused: set[tuple] = set()
    for r in out:
        if r["condition"] != REFERENCE[0]:
            continue
        pm = (r["plate"], r.get("device_mode", "open"))
        if r["pred_verdict"] == RETAKE:
            refused.add(pm)
            continue
        same_k.setdefault((*pm, r["k"]), r["pred_verdict"])
        per_plate[pm].append(r["pred_verdict"])
    for r in out:
        pm = (r["plate"], r.get("device_mode", "open"))
        v = same_k.get((*pm, r["k"]))
        if v is None and per_plate.get(pm):
            v = Counter(per_plate[pm]).most_common(1)[0][0]
        r["c0_verdict"] = v if v is not None else (RETAKE if pm in refused else None)


def inspect(detector, caps: Sequence[Capture], spec: dict, judge: Judge, ref_vis: dict[str, dict[str, bool]],
            gt_measure: str = "box", gt_visibility: str = "c0", match_iou: float = 0.25, cover: float = 0.5,
            tile: int = 640, min_overlap: int = 128, min_box_px: float = MIN_BOX_PX,
            measure: str = "box", plans_dir: Path | None = None) -> list[dict[str, Any]]:
    """One model on some captures: detections, matching and verdicts per capture (measure: how the
    detections are turned into mm, box | app; see the module docstring). plans_dir: the matrix
    plans, whose defect params the "true" ground truth uses (pitted rust, pit counts)."""
    out = []
    plans: dict[str, dict | None] = {}
    for c in caps:
        rgb = EP.load_rgb(c.image)
        sc = dict(c.sidecar, plate=c.plate, face="")         # physical identity = the plate, across rows
        if gt_visibility == "all":
            vis = {str(d.get("id")): True for d in sc.get("defects") or []}
        elif gt_visibility == "c0" and ((c.plate, c.device_mode) in ref_vis or (c.plate, "any") in ref_vis):
            vis = ref_vis.get((c.plate, c.device_mode)) or ref_vis[(c.plate, "any")]
        else:
            vis = visibility_map(c.sidecar, c.k)
        gts = EP.gt_objects(sc, rgb.shape[:2], c.image.stem, vis)
        if gt_measure == "true" and c.scene not in plans:
            pp = Path(plans_dir) / f"{c.scene}.json" if plans_dir is not None else None
            plans[c.scene] = json.loads(pp.read_text()) if pp is not None and pp.exists() else None
        gt = gt_defects(gts, sc, gt_measure, min_box_px, c.part, plans.get(c.scene))
        gt_v = str(judge(gt, spec, c.part)[0])
        if c.gate == RETAKE:          # the device refused: nothing is detected or judged
            dets, match, pred_v = [], None, RETAKE
        else:
            prime = getattr(detector, "prime", None)         # the oracle reads what this grid shows
            if prime is not None:
                prime(c.sidecar, rgb.shape[:2], tile, min_overlap)
            dets = detect_grid(rgb, detector, tile, min_overlap)
            match = EP.match_grid(dets, gts, match_iou, cover)
            pred = app_defects(dets, sc, rgb) if measure == "app" else pred_defects(dets, sc, min_box_px)
            pred_v = str(judge(pred, spec, c.part)[0])
        out.append({
            "arm": c.arm, "scene": c.scene, "k": c.k, "plate": c.plate, "condition": c.condition,
            "device_mode": c.device_mode, "tag": tag(c.condition, c.device_mode), "image": str(c.image),
            "gate": c.gate, "gate_message": c.gate_message, "gate_reasons": list(c.gate_reasons),
            "lux": c.lux, "part": c.part,
            "gt_verdict": gt_v, "pred_verdict": pred_v, "match": match,
            "dets": [[d.cls, round(d.conf, 3), *(round(v, 1) for v in (d.x0, d.y0, d.x1, d.y1))] for d in dets],
            "gt": [[g.cls, *(round(v, 1) for v in g.box), g.visible] for g in gts if g.defect],
        })
    link_c0(out)
    return out


def row_metrics(rs: Sequence[dict[str, Any]], order: Sequence[str] = VERDICT_ORDER) -> dict[str, Any]:
    """Aggregate one row (condition, device_mode) of one model x arm."""
    n = len(rs)
    acc = [r for r in rs if r["gate"] != RETAKE]
    summary = EP.summarize([r["match"] for r in acc]) if acc else None
    objs = [o for r in acc for o in r["match"]["objects"] if o["defect"] and o["visible"]]
    tp = sum(o["found"] for o in objs)
    fp = sum(d["status"] == "fp" and d["cls"] in EP.DEFECT_IDS for r in acc for d in r["match"]["dets"])
    wrong = [r for r in acc if r["pred_verdict"] != r["gt_verdict"]]
    bad = [r for r in wrong if unsafe(r["pred_verdict"], r["gt_verdict"], order)]
    comparable = [r for r in acc if r.get("c0_verdict") not in (None, RETAKE)]
    # SSIM over what the device kept (like P/R/F1): a refused capture's image is never used
    ssims = [r["ssim_c0"] for r in acc if r.get("ssim_c0") is not None]
    ssims_all = [r["ssim_c0"] for r in rs if r.get("ssim_c0") is not None]
    mads = [r["mad_c0"] for r in acc if r.get("mad_c0") is not None]
    luxes = [r["lux"] for r in rs if r.get("lux") is not None]
    all_obj = sum(len([g for g in r["gt"] if g[-1]]) for r in rs)
    pc = summary["per_class"] if summary else {}
    reasons = Counter(x for r in rs if r["gate"] == RETAKE for x in (r.get("gate_reasons") or ["(none stored)"]))
    rank = {k: i for i, k in enumerate(REASON_ORDER)}
    shown = Counter(min(r.get("gate_reasons") or ["(none stored)"], key=lambda x: rank.get(x, len(rank)))
                    for r in rs if r["gate"] == RETAKE)
    return {
        "n_captures": n, "n_plates": len({r["plate"] for r in rs}), "n_accepted": len(acc),
        "lux_median": round(statistics.median(luxes), 1) if luxes else None,
        "retake": EP.rate(n - len(acc), n),
        "flagged": sum(r["gate"] == "PASS_FLAGGED" for r in rs),
        "precision": pc.get("all_defects", {}).get("precision", EP.rate(0, 0)),
        "recall": EP.rate(tp, len(objs)),
        "recall_physical": pc.get("all_defects", {}).get("recall_physical", EP.rate(0, 0)),
        "recall_with_retakes_as_misses": EP.rate(tp, all_obj),
        "f1": f1_rate(tp, fp, len(objs) - tp),
        "per_class_recall": {c: pc[c]["recall_instances"] for c in CLASSES if c in pc},
        "fp_per_clean_plate": summary["fp_per_clean_plate"] if summary else None,
        "n_clean_accepted": summary["n_clean_grids"] if summary else 0,
        "verdict_agreement_gt": EP.rate(len(acc) - len(wrong), len(acc)),
        "verdict_agreement_c0": EP.rate(sum(r["pred_verdict"] == r["c0_verdict"] for r in comparable), len(comparable)),
        "silent": EP.rate(len(wrong), n),
        "silent_unsafe": EP.rate(len(bad), n),
        "silent_flagged": sum(r["gate"] == "PASS_FLAGGED" for r in wrong),
        "silent_pairs": dict(Counter(f"{r['pred_verdict']} (truth {r['gt_verdict']})" for r in wrong)),
        "ssim_c0": {"mean": round(float(np.mean(ssims)), 4) if ssims else None,
                    "min": round(float(np.min(ssims)), 4) if ssims else None, "n": len(ssims)},
        "ssim_c0_all": {"mean": round(float(np.mean(ssims_all)), 4) if ssims_all else None, "n": len(ssims_all)},
        "mad_c0": {"mean": round(float(np.mean(mads)), 3) if mads else None,
                   "max": round(float(np.max(mads)), 3) if mads else None, "n": len(mads)},
        "gate_messages": dict(Counter(r["gate_message"] for r in rs if r["gate"] != "PASS" and r["gate_message"])),
        "retake_reasons": dict(reasons.most_common()),           # every reason of every refused capture
        "main_reason": shown.most_common(1)[0][0] if shown else None,   # the most common reason shown first
    }


def _check(name: str, row: str, status: str, detail: str) -> dict[str, str]:
    return {"check": name, "row": row, "status": status, "detail": detail}


def checks_for(rows: dict[tuple[str, str], dict[str, Any]], conditions: Sequence[str]) -> list[dict[str, str]]:
    """The PASS criterion (DESIGN_STEEL G, device v2) as a list of checks, status PASS | FAIL |
    MISSING | INFO. HOOD (the product): every condition present, retake <= 5 %, F1 within 0.05
    of C0 hood, verdict = C0 hood >= 95 %, no silent failure. OPEN: no silent failure in any
    row; its retakes are INFO (rate and main gate reason)."""
    out = []
    must = [(c, PRODUCT) for c in conditions] + [REFERENCE]
    for key in must:
        if key not in rows:
            out.append(_check("row present", tag(*key), "MISSING", "no captures for this row"))
    for key, m in rows.items():
        t = tag(*key)
        s = m["silent"]
        out.append(_check("no silent failure", t, "PASS" if s["k"] == 0 else "FAIL",
                          f"{s['k']} of {s['n']} captures gave a wrong verdict without a retake "
                          f"({m['silent_unsafe']['k']} unsafe: milder than the truth, no hold)"))
        rt = m["retake"]
        why = f" (main reason: {m['main_reason']})" if m.get("main_reason") else ""
        if key[1] != PRODUCT:
            if rt["n"]:
                out.append(_check("retakes reported", t, "INFO", f"the gate refused {rt['k']} of {rt['n']} captures{why}"))
            continue
        out.append(_check("retake rate <= 5 %", t, "PASS" if rt["n"] and rt["value"] <= HOOD_RETAKE_MAX + 1e-12 else "FAIL",
                          f"the gate refused {rt['k']} of {rt['n']} captures{why}"))
        if key == reference_of(PRODUCT):
            continue
        ref = rows.get(reference_of(PRODUCT))
        f1_ref = ref["f1"]["value"] if ref else None
        f1 = m["f1"]["value"]
        if m["n_accepted"] == 0:
            out.append(_check("F1 within 0.05 of C0", t, "FAIL", "no accepted capture"))
        elif f1_ref is None or f1 is None:
            out.append(_check("F1 within 0.05 of C0", t, "MISSING",
                              "no C0 hood reference" if ref is None else "no defects to score"))
        else:
            ok = f1 >= f1_ref - F1_DROP_MAX - 1e-12
            out.append(_check("F1 within 0.05 of C0", t, "PASS" if ok else "FAIL", f"F1 {f1:.3f} vs C0 hood {f1_ref:.3f}"))
        ag = m["verdict_agreement_c0"]
        if not ag["n"]:
            out.append(_check("verdict = C0 >= 95 %", t, "FAIL" if m["n_accepted"] == 0 else "MISSING",
                              "no capture to compare with C0 hood"))
        else:
            out.append(_check("verdict = C0 >= 95 %", t, "PASS" if ag["value"] >= AGREE_MIN else "FAIL",
                              f"{ag['k']} of {ag['n']} verdicts equal the C0 hood verdict"))
    return out


def overall(checks: Sequence[dict[str, str]]) -> str:
    st = {c["status"] for c in checks}
    return "FAIL" if "FAIL" in st else "INCOMPLETE" if "MISSING" in st else "PASS"


def plain_verdict(arm: str, model: str, verdict: str, rows: dict[tuple[str, str], dict[str, Any]],
                  checks: Sequence[dict[str, str]]) -> str:
    """A short plain-language verdict for one model x arm."""
    label = ARM_NAMES.get(arm, arm) + (f" [model {model}]" if model != arm else "")
    ref = rows.get(reference_of(PRODUCT))
    f1_ref = ref["f1"]["value"] if ref else None
    open_rows = [(c, m) for (c, d), m in rows.items() if d != PRODUCT]
    refused = [f"{IN.get(c, 'in ' + c)} {m['retake']['k']}/{m['retake']['n']}"
               + (f" ({m['main_reason']})" if m.get("main_reason") else "") for c, m in open_rows if m["retake"]["k"]]
    open_text = (" Without the hood it never gave a wrong verdict without asking for a retake"
                 + (f"; it refused {', '.join(refused)}." if refused else "."))
    if verdict == "PASS":
        return (f"{label}: PASS. With the hood head (device v2) it worked in every lighting condition: "
                f"at most {HOOD_RETAKE_MAX:.0%} retakes, detection within {F1_DROP_MAX} F1 of the dark room"
                f"{' (F1 ' + format(f1_ref, '.2f') + ')' if f1_ref is not None else ''}, the dark-room verdict in at "
                f"least {AGREE_MIN:.0%} of captures and no silent failure." + open_text)
    bad = [c for c in checks if c["status"] in ("FAIL", "MISSING")]
    said = list(dict.fromkeys(_sentence(c, rows, f1_ref) for c in bad))       # one sentence per problem
    body = "; ".join(said[:6]) + ("; ..." if len(said) > 6 else "")
    word = "FAIL" if verdict == "FAIL" else "INCOMPLETE"
    return f"{label}: {word}. {body[:1].upper()}{body[1:]}."


def _sentence(c: dict[str, str], rows: dict[tuple[str, str], dict[str, Any]], f1_ref: float | None) -> str:
    """One failed or missing check in plain words."""
    key = next((k for k in rows if tag(*k) == c["row"]), None)
    if key is None:
        cond = next((x for x in NAMES if c["row"].startswith(x.split("_")[0])), c["row"])
        where = f"{NAMES.get(cond, cond)} with the {'hood' if c['row'].endswith('h') else 'open device'}"
        where = where[:1].upper() + where[1:]
        return f"{where} was not tested" if c["status"] == "MISSING" else f"{where}: {c['detail']}"
    m = rows[key]
    where = f"{IN.get(key[0], 'in ' + key[0])} with the {'hood' if key[1] == 'hood' else 'open device'}"
    if c["check"] == "no silent failure":
        pairs = m.get("silent_pairs") or {}
        top = f", most often {max(pairs, key=pairs.get)}" if pairs else ""
        return (f"{where}, {m['silent']['k']} of {m['silent']['n']} captures got a wrong verdict with no warning "
                f"({m['silent_unsafe']['k']} unsafe: milder than the truth and not held{top})")
    why = f" (mostly {m['main_reason']})" if m.get("main_reason") else ""
    if m["n_accepted"] == 0:
        return f"{where}, every capture was refused{why}, so nothing was inspected"
    if c["check"].startswith("retake") and c["status"] == "FAIL":
        return f"{where}, the gate refused {m['retake']['k']} of {m['retake']['n']} captures{why}"
    if c["check"].startswith("F1") and c["status"] == "FAIL":
        return f"{where}, detection F1 fell to {m['f1']['value']:.2f} from {f1_ref:.2f} in the dark room"
    if c["check"].startswith("verdict") and c["status"] == "FAIL":
        ag = m["verdict_agreement_c0"]
        return f"{where}, only {ag['k']} of {ag['n']} verdicts matched the dark room"
    return f"{where}: {c['check']} - {c['detail']}"


def row_order(keys, conditions: Sequence[str]) -> list[tuple[str, str]]:
    rank = {c: i for i, c in enumerate(conditions)}
    return sorted(keys, key=lambda k: (k[1] != "open", rank.get(k[0], len(rank)), k[0]))


def pairs_for(models: dict[str, Any], arms: Sequence[str], pairs: str = "own") -> list[tuple[str, str]]:
    """(model, arm) pairs: own = a model named like an arm runs on that arm only (others on every arm)."""
    out = []
    for name in models:
        for arm in arms:
            if pairs == "all" or name not in arms or name == arm:
                out.append((name, arm))
    return out


def evaluate_matrix(models: dict[str, Any], grids_root: Path, sidecars: Path, arms: Sequence[str] = ARMS,
                    manifest: dict | None = None, plans_dir: Path | None = None, spec: dict | None = None,
                    judge: Judge | None = None, pairs: str = "own", gt_measure: str | None = None,
                    gt_visibility: str = "c0", match_iou: float = 0.25, cover: float = 0.5, tile: int = 640,
                    min_overlap: int = 128, min_box_px: float = MIN_BOX_PX, measure: str = "box") -> dict[str, Any]:
    """The whole lighting experiment -> the lighting_matrix report dict. spec None: the default
    configs/spec_steel.yaml (never an empty spec with the real rules). measure: box | app (how
    detections become mm); gt_measure None: box with measure box, true (analytic) with app."""
    if measure not in MEASURES:
        raise ValueError(f"measure {measure!r} not in {MEASURES}")
    gt_measure = gt_measure or ("true" if measure == "app" else "box")
    judge = judge or default_judge()
    spec = spec if spec is not None else default_spec()
    index = MX.scene_index(manifest) if manifest else {}
    caps = collect(grids_root, sidecars, arms, index, plans_dir)
    if not caps:
        raise SystemExit(f"no grids under {grids_root}/<arm>/ for arms {list(arms)}")
    conditions = list((manifest or {}).get("conditions") or MX.ALL)
    order = verdict_order()
    ref_vis = reference_visibility(caps)
    scores = reference_scores(caps)
    results, captures = [], []
    for name, arm in pairs_for(models, arms, pairs):
        cs = [c for c in caps if c.arm == arm]
        if not cs:
            continue
        t0 = time.time()
        rs = inspect(models[name], cs, spec, judge, ref_vis, gt_measure, gt_visibility, match_iou, cover,
                     tile, min_overlap, min_box_px, measure, plans_dir)
        for r in rs:
            r["model"] = name
            sc_ = scores.get((arm, r["scene"], r["k"])) or {}
            r["ssim_c0"], r["mad_c0"] = sc_.get("ssim"), sc_.get("mad")
        by_row = defaultdict(list)
        for r in rs:
            by_row[(r["condition"], r["device_mode"])].append(r)
        rows = {k: row_metrics(by_row[k], order) for k in row_order(by_row, conditions)}
        checks = checks_for(rows, conditions)
        verdict = overall(checks)
        results.append({"model": name, "arm": arm, "verdict": verdict,
                        "summary": plain_verdict(arm, name, verdict, rows, checks), "checks": checks,
                        "rows": [dict(condition=c, device_mode=m, tag=tag(c, m), **v) for (c, m), v in rows.items()],
                        "seconds": round(time.time() - t0, 1)})
        captures += [{k: v for k, v in r.items() if k != "match"} | {"n_found": _found(r), "n_gt": _n_gt(r)}
                     for r in rs]
    return {"schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "criterion": criterion(),
            "rule": {"match_iou": match_iou, "cover": cover, "measure": measure, "gt_measure": gt_measure,
                     "gt_visibility": gt_visibility,
                     "min_box_px": min_box_px, "ssim": "over accepted captures (ssim_c0_all: every capture)",
                     "mad": "mean |grid - grid_C0| over the plate, RGB, 8-bit DN, accepted captures",
                     "silent": "gate not RETAKE and verdict != ground-truth verdict"},
            "arms": list(arms), "conditions": conditions, "classes": list(CLASSES),
            "n_plates": len({c.plate for c in caps}), "results": results,
            "verdict_text": "\n\n".join(r["summary"] for r in results), "captures": captures}


def _found(r: dict[str, Any]) -> int:
    return sum(o["found"] for o in (r["match"] or {}).get("objects", []) if o["defect"] and o["visible"])


def _n_gt(r: dict[str, Any]) -> int:
    return sum(1 for g in r["gt"] if g[-1])


# --- Report ------------------------------------------------------------------------------

def _r(x: dict[str, Any] | None, pct: bool = False) -> str:
    if not x or not x.get("n") and x.get("value") is None:
        return "n/a"
    v = x.get("value")
    if v is None:
        return "n/a"
    s = f"{v:.0%}" if pct else f"{v:.2f}"
    return f"{s} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]"


def _frac(x: dict[str, Any]) -> str:
    return f"{x['k']}/{x['n']}" if x.get("n") else "n/a"


def to_markdown(rep: dict[str, Any]) -> str:
    lines = ["# Lighting experiment: does the inspector work in different lighting?", "",
             f"{rep['n_plates']} test plates (seeds >= {MX.TEST_SEED_MIN}, never trained on); "
             f"arms: {', '.join(rep['arms'])}; created {rep['created']}.", "",
             "## Verdict", ""]
    for r in rep["results"]:
        lines += [f"- **{r['verdict']}** - {r['summary']}"]
    c = rep["criterion"]
    lines += ["", "## Pass criterion", "", c["text"], "",
              f"Matching: same class and IoU >= {rep['rule']['match_iou']} or box covers >= "
              f"{rep['rule']['cover']:.0%} of the true mask. Detections measured: {rep['rule'].get('measure', 'box')} "
              f"({'as the on-device app measures them' if rep['rule'].get('measure') == 'app' else 'padded box, like the ground truth'}). "
              f"Ground-truth verdicts: rules engine on the ground-truth "
              f"defects ({rep['rule']['gt_measure']} measures; visibility: {rep['rule']['gt_visibility']}). "
              "Silent failure = the gate let the capture through but the verdict is wrong; 'unsafe' = milder than "
              "the truth. P/R/F1 are over accepted (not retaken) captures; intervals are Wilson 95 % (F1 approximate; "
              "variants of one plate are not independent).", ""]
    for r in rep["results"]:
        lines += [f"## {r['arm']} arm, model {r['model']}: {r['verdict']}", "",
                  "| Row | Median lux | Captures | Retake | Main retake reason | Precision | Recall | F1 | "
                  "FP / clean plate | Verdict = C0 | Verdict = truth | Silent (unsafe) | SSIM vs C0 | |Δ| vs C0 (DN) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in r["rows"]:
            fpc = m["fp_per_clean_plate"]
            lux = m["lux_median"]
            lines.append(
                f"| {m['tag']} {NAMES.get(m['condition'], m['condition'])}{' + hood' if m['device_mode'] == 'hood' else ''} "
                f"| {'n/a' if lux is None else f'{lux:,.0f}'} | {m['n_captures']} | {_frac(m['retake'])} "
                f"| {m.get('main_reason') or '-'} | {_r(m['precision'])} | {_r(m['recall'])} | {_r(m['f1'])} "
                f"| {'n/a' if fpc is None else f'{fpc:.2f}'} | {_frac(m['verdict_agreement_c0'])} "
                f"| {_frac(m['verdict_agreement_gt'])} | {m['silent']['k']}/{m['silent']['n']} "
                f"({m['silent_unsafe']['k']}) | {'n/a' if m['ssim_c0']['mean'] is None else format(m['ssim_c0']['mean'], '.3f')} "
                f"| {'n/a' if (m.get('mad_c0') or {}).get('mean') is None else format(m['mad_c0']['mean'], '.1f')} |")
        lines += ["", "Hood rows are compared with C0 hood, open rows with C0 open (verdicts and SSIM).",
                  "", "Checks:", ""]
        lines += [f"- {ch['status']}: {ch['row']} {ch['check']} ({ch['detail']})" for ch in r["checks"]]
        msgs = Counter()
        for m in r["rows"]:
            msgs.update(m["gate_messages"])
        if msgs:
            lines += ["", "Gate messages shown to the operator:", ""]
            lines += [f"- {n} x \"{msg}\"" for msg, n in msgs.most_common()]
        lines.append("")
    return "\n".join(lines) + "\n"


# --- Figures -----------------------------------------------------------------------------

def _boxes(img: np.ndarray, boxes, scale: float, colour, names: bool = False) -> None:
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in b[-4:])
        cv2.rectangle(img, (x0, y0), (max(x1, x0 + 1), max(y1, y0 + 1)), colour, 1)
        if names:
            cv2.putText(img, CLASSES[b[0]][:4] if 0 <= b[0] < len(CLASSES) else str(b[0]), (x0, max(9, y0 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)


def cell_image(cap: dict[str, Any], size: int = 280) -> np.ndarray:
    """One capture as a BGR cell: the grid with ground truth (green) and detections (red), plus a caption."""
    rgb = EP.load_rgb(Path(cap["image"]))
    s = size / max(rgb.shape[:2])
    small = cv2.resize(rgb, (max(1, round(rgb.shape[1] * s)), max(1, round(rgb.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    img = np.zeros((size + 34, size, 3), np.uint8)
    img[: small.shape[0], : small.shape[1]] = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    _boxes(img, [[g[1] - 3, g[2] - 3, g[3] + 3, g[4] + 3] for g in cap["gt"] if g[-1]], s, (0, 200, 0))   # just outside
    _boxes(img, [[d[0], *d[2:6]] for d in cap["dets"]], s, (0, 0, 255), names=True)
    lux = cap.get("lux")
    top = f"{cap['tag']}  {'' if lux is None else f'{lux / 1000:.1f} klx'}  gate {cap['gate']}"
    bottom = "RETAKE: refused" if cap["pred_verdict"] == RETAKE else f"{cap['pred_verdict']} (truth {cap['gt_verdict']})"
    ok = cap["pred_verdict"] in (RETAKE, cap["gt_verdict"])
    cv2.putText(img, top, (3, size + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, bottom, (3, size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 220, 80) if ok else (60, 60, 255),
                1, cv2.LINE_AA)
    return img


def plate_montage(caps: Sequence[dict[str, Any]], conditions: Sequence[str], size: int = 280) -> np.ndarray:
    """Columns = matrix rows, rows = arms (first variant of each); a white label column on the left."""
    keys = row_order({(c["condition"], c["device_mode"]) for c in caps}, conditions)
    arms = sorted({(c["arm"], c["model"]) for c in caps}, key=lambda a: (a[0] != "naive", a))
    label_w, cell_h = 90, size + 34
    out = np.full((cell_h * len(arms), label_w + size * len(keys), 3), 32, np.uint8)
    for i, (arm, model) in enumerate(arms):
        cv2.putText(out, arm, (6, i * cell_h + cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if model != arm:
            cv2.putText(out, model[:10], (6, i * cell_h + cell_h // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1, cv2.LINE_AA)
        for j, key in enumerate(keys):
            cs = sorted((c for c in caps if c["arm"] == arm and c["model"] == model
                         and (c["condition"], c["device_mode"]) == key), key=lambda c: c["k"] if c["k"] is not None else -1)
            if cs:
                out[i * cell_h : (i + 1) * cell_h, label_w + j * size : label_w + (j + 1) * size] = cell_image(cs[0], size)
    return out


def write_figures(rep: dict[str, Any], out_dir: Path, n_plates: int = 3, size: int = 280) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_plate = defaultdict(list)
    for c in rep["captures"]:
        by_plate[c["plate"]].append(c)
    # defective plates first: they show the failures
    plates = sorted(by_plate, key=lambda p: (not any(c["n_gt"] for c in by_plate[p]), p))[:n_plates]
    paths = []
    for p in plates:
        path = out_dir / f"lighting_{p}.png"
        cv2.imwrite(str(path), plate_montage(by_plate[p], rep["conditions"], size))
        paths.append(path)
    return paths


def write_report(rep: dict[str, Any], out: Path, figures: int = 3) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "lighting_matrix.json", "md": out / "lighting_matrix.md"}
    figs = write_figures(rep, out / "figures", figures) if figures else []
    rep["figures"] = [str(p) for p in figs]
    paths["json"].write_text(json.dumps(rep, indent=1, default=str))
    md = to_markdown(rep)
    if figs:
        md += "\n## Figures\n\n" + "\n".join(f"![{p.stem}](figures/{p.name})" for p in figs) + "\n"
    paths["md"].write_text(md)
    return paths


def parse_models(specs: Sequence[str], conf: float) -> dict[str, Any]:
    models = {}
    for s in specs:
        name, _, path = s.partition("=")
        if not path:
            name, path = Path(s).stem, s
        models[name] = OracleDetector(label_min_box()) if path == ORACLE else load_detector(path, conf=conf)
    return models


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, required=True, help="matrix dir (matrix_manifest.json, grids/, sidecar/)")
    ap.add_argument("--grids", type=Path, help="default <matrix>/grids (with <arm>/ subdirs)")
    ap.add_argument("--sidecars", type=Path, help="default <matrix>/sidecar")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                    help="repeat: naive=models/v8n_naive.onnx burst=models/v8n_burst.onnx (.onnx/.pt/.hef); "
                         "'oracle' = a perfect detector on each grid's visible labels (runs on every arm)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--pairs", choices=("own", "all"), default="own",
                    help="own: model 'naive' on the naive arm, 'burst' on the burst arm; all: every model on every arm")
    ap.add_argument("--spec", type=Path, default=ROOT / "configs/spec_steel.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    ap.add_argument("--figures", type=int, default=3, help="plates with a montage (0 = none)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.25)
    ap.add_argument("--cover", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--min-overlap", type=int, default=128)
    ap.add_argument("--measure", choices=MEASURES, default="box",
                    help="box: detections and truth measured from padded boxes alike (isolates detection); "
                         "app: detections measured as the on-device app does (ayas.app.pipeline)")
    ap.add_argument("--gt-measure", choices=("box", "true"),
                    help="ground-truth measures (default: box with --measure box, true with --measure app)")
    ap.add_argument("--gt-visibility", choices=("c0", "own", "all"), default="c0")
    ap.add_argument("--min-box-px", type=float, default=label_min_box(),
                    help="boxes are measured padded to this many px per side (default: twin.yaml label.min_box_px)")
    a = ap.parse_args(argv)
    from ayas import rules

    man_path = a.matrix / MX.MANIFEST
    manifest = MX.load_manifest(man_path) if man_path.exists() else None
    rep = evaluate_matrix(parse_models(a.model, a.conf), a.grids or a.matrix / "grids", a.sidecars or a.matrix / "sidecar",
                          a.arms, manifest, a.matrix / "plans", rules.load_spec(a.spec), default_judge(), a.pairs,
                          a.gt_measure, a.gt_visibility, a.match_iou, a.cover, a.tile, a.min_overlap, a.min_box_px,
                          a.measure)
    rep.update(models={m.partition("=")[0]: m.partition("=")[2] or m for m in a.model}, spec=str(a.spec), conf=a.conf)
    paths = write_report(rep, a.out, a.figures)
    print(rep["verdict_text"])
    print(f"wrote {paths['md']} and {paths['json']}")
    return rep


if __name__ == "__main__":
    main()
