"""Twin defect records and the advisor regression harness (`trinetra-twin regress`).

Steel and metal surfaces (configs/spec_steel.yaml, docs/twin/DESIGN_STEEL.md F; the default):
- steel_record_from_plan / records_from_sidecar build the record 3.0 a PERFECT detector would give
  from the plan or sidecar labels (footprint measures, centre, near_edge_mm; never depth), with
  the capture block from the sidecar's quality gate (a RETAKE gate gives a RETAKE record with no
  defects). records_from_plan dispatches on the spec domain.
- gt_verdict(labels_mm, spec, part) is the ground-truth verdict the lighting evaluation compares
  against: the steel rules on the labelled defects at confidence 1.
- run_regression runs many cases through the rules and the advisor (mock, or the ExecuTorch
  backend when a .pte is given) and reports the verdict distribution, guardrail violations (0
  with the mock), failures and ground-truth mismatches.

Aerospace (configs/spec_aero.yaml, the 5 Nov build): records_from_plan with the aero spec turns
a scene plan into the report a perfect detector would give: every labelled defect where the
camera can see it (the part hidden under rivet heads removed, `near` distances from the plan),
every hole as a rivet or open hole, and burr distractors as `unknown`. run_regression with the
aero spec also counts fastener findings and mismatches against the coupon `expect` list or the
sampler's fastener violations. Dent depth is not observable, so records carry no depth unless
with_depth=True.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from ..aero import Fastener
from ..report import build_aero_report, build_steel_record, capture_block, capture_from_metrics, validate_record
from ..rules import STEEL_CLASSES, is_steel, judge_steel, steel_part

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "schemas" / "inspection.schema.json"
PLAN_SCHEMA = "trinetra.twin.scene/1"
PLAN_SCHEMAS = ("trinetra.twin.scene/1", "trinetra.twin.scene/2")
SIDECAR_PREFIX = "trinetra.twin.sidecar/"
STEEL_SPEC = "configs/spec_steel.yaml"
STEEL_QUESTIONS = ("What should I do next?", "Can I ship it anyway?", "How deep is it?", "Explain in Hindi.")
PTE_SEQ_LEN = 2048            # llama_main --seq_len for --backend <.pte> (prompt + answer)
SIZE_KEYS = ("length_mm", "width_mm", "area_mm2", "diameter_mm", "count", "pit_d_mm")
EDGE_KEYS = ("near_edge_mm", "plate_edge_mm", "hole_edge_mm", "fastener_edge_mm", "edge_mm")
PERFECT = "twin perfect detector"
SURFACE = ("dent", "scratch", "crack", "corrosion")
UNKNOWN_KINDS = ("burr",)
RES_MM = 0.05                 # raster step for visibility and centroids
MIN_VISIBLE_CRACK_MM = 1.0    # a partly hidden crack needs this much length outside the heads (as coupons.py)
MIN_VISIBLE_FRAC = 0.05       # other defects: visible fraction of the outline area
QUESTIONS = ("What should I do next?", "Can I use it as is?")
FASTENER_FINDINGS = ("edge_distance", "pitch", "missing_fastener", "elongated_hole")
VIOLATION_TYPES = {"edge_distance": "edge_distance", "pitch": "pitch", "missing": "missing_fastener",
                   "oval": "elongated_hole"}


# ----------------------------------------------------------------------------- geometry helpers

def fasteners_from_plan(plan: dict[str, Any], conf: float = 1.0) -> list[Fastener]:
    """Holes as the camera sees them: a rivet head circle, or the open hole ellipse."""
    rivets = {r["hole"]: r for r in plan.get("rivets", [])}
    D = _plan_D(plan)
    out = []
    for h in plan.get("holes", []):
        r = rivets.get(h["id"]) if h.get("filled", True) else None
        if h.get("filled", True):
            head = float(r["head_d_mm"]) if r else 2.0 * (D or float(h["d_mm"]))
            out.append(Fastener(h["id"], "rivet", float(h["u"]), float(h["v"]), head, 1.0, 0.0, conf))
        else:
            out.append(Fastener(h["id"], "open_hole", float(h["u"]), float(h["v"]), float(h["d_mm"]),
                                float(h.get("ovality", 1.0)), float(h.get("angle_deg", 0.0)), conf))
    return out


def _plan_D(plan: dict[str, Any]) -> float | None:
    D = (plan.get("fasteners") or {}).get("D_mm")
    return float(D) if D else None


def _to_px(poly, u0: float, v0: float, res: float) -> np.ndarray:
    """Plate mm -> raster px (pixel (r, c) centred on u0 + (c + 0.5) res), fixed point with shift 4."""
    p = (np.asarray(poly, float) - [u0, v0]) / res - 0.5
    return np.round(p * 16).astype(np.int32).reshape(-1, 1, 2)


def _rasters(polys, heads, res: float = RES_MM):
    """(defect mask, visible mask, u0, v0) on a window round the outline."""
    pts = np.vstack([np.asarray(p, float) for p in polys])
    u0, v0 = pts.min(axis=0) - 0.5
    u1, v1 = pts.max(axis=0) + 0.5
    nu, nv = int(math.ceil((u1 - u0) / res)), int(math.ceil((v1 - v0) / res))
    mask = np.zeros((nv, nu), np.uint8)
    cv2.fillPoly(mask, [_to_px(p, u0, v0, res) for p in polys], 1, lineType=cv2.LINE_8, shift=4)
    hidden = np.zeros_like(mask)
    for (u, v, r) in heads:
        c = _to_px([[u, v]], u0, v0, res).reshape(2)
        cv2.circle(hidden, (int(c[0]), int(c[1])), int(round(r / res * 16)), 1, -1, cv2.LINE_8, 4)
    return mask.astype(bool), mask.astype(bool) & ~hidden.astype(bool), float(u0), float(v0)


def _inside(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test for many points."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    a, b = poly, np.roll(poly, -1, axis=0)
    cond = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return ((cond & (x < xc)).sum(axis=1) % 2) == 1


def _pts_to_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(axis=1), 1e-18)
    s = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(axis=2) / L2, 0.0, 1.0)
    q = a[None] + s[..., None] * ab[None]
    return float(np.sqrt(((p[:, None, :] - q) ** 2).sum(axis=2)).min())


def _cross(a1, a2, b1, b2) -> bool:
    """Any proper crossing between the edges a1->a2 and b1->b2."""
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    A1, A2, B1, B2 = a1[:, None], a2[:, None], b1[None], b2[None]
    o1, o2 = orient(A1, A2, B1), orient(A1, A2, B2)
    o3, o4 = orient(B1, B2, A1), orient(B1, B2, A2)
    return bool(((o1 * o2 < 0) & (o3 * o4 < 0)).any())


def poly_gap(a, b) -> float:
    """Edge-to-edge distance in mm between two polygons (0 when they touch or overlap)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if _inside(a[:1], b)[0] or _inside(b[:1], a)[0]:
        return 0.0
    a2, b2 = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
    if _cross(a, a2, b, b2):
        return 0.0
    return min(_pts_to_segments(a, b, b2), _pts_to_segments(b, a, a2))


# ----------------------------------------------------------------------------- records

def _shape_stats(vis: np.ndarray, u0: float, v0: float, res: float) -> dict[str, float]:
    rr, cc = np.nonzero(vis)
    u, v = u0 + (cc + 0.5) * res, v0 + (rr + 0.5) * res
    cu, cv = float(u.mean()), float(v.mean())
    if len(u) >= 3:
        w, vec = np.linalg.eigh(np.cov(np.stack([u - cu, v - cv])))
        major = vec[:, 1]
    else:
        major = np.array([1.0, 0.0])
    s = (u - cu) * major[0] + (v - cv) * major[1]
    t = -(u - cu) * major[1] + (v - cv) * major[0]
    ang = math.degrees(math.atan2(major[1], major[0])) % 180.0
    return {"u": cu, "v": cv, "length": float(s.max() - s.min() + res), "width": float(t.max() - t.min() + res),
            "area": float(len(u) * res * res), "angle": ang}


def _defect_record(d: dict[str, Any], plan: dict[str, Any], heads, damage, with_depth: bool,
                   conf: float) -> dict[str, Any] | None:
    fp = d.get("footprint") or {}
    polys = [p for p in fp.get("polys_mm", []) if len(p) >= 3]
    if not polys:
        return None
    mask, vis, u0, v0 = _rasters(polys, heads)
    if not mask.any():
        return None
    frac = float(vis.sum()) / float(mask.sum())
    m = fp.get("measure", {})
    if d["cls"] == "crack":       # a crack partly under a head shows only if enough of it sticks out
        if frac < 0.999 and float(m.get("length_mm", 0.0)) * frac < MIN_VISIBLE_CRACK_MM:
            return None
    elif frac < MIN_VISIBLE_FRAC:
        return None
    st = _shape_stats(vis, u0, v0, RES_MM)
    length = float(m["length_mm"]) * (frac if d["cls"] == "crack" else 1.0) if "length_mm" in m else st["length"]
    width = float(m.get("width_mm", st["width"]))
    area = float(m.get("area_mm2", st["area"]))
    diameter = float(m.get("diameter_mm", math.sqrt(4 * area / math.pi)))
    if d["cls"] == "dent":
        length = max(length, diameter) if "length_mm" in m else diameter
    near = dict(d.get("near") or {})
    if d["cls"] == "dent" and damage:
        gaps = [poly_gap(p, q) for oid, qs in damage if oid != d["id"] for p in polys for q in qs]
        if gaps:
            near["damage_edge_mm"] = round(min(gaps), 4)
    depth = m.get("depth_mm")
    rec = {
        "id": "", "twin_id": d.get("id"), "type": d["cls"], "confidence": conf,
        "location_mm": {"x": round(st["u"], 1), "y": round(st["v"], 1)},
        "size": {"length_mm": round(length, 2), "width_mm": round(width, 2), "area_mm2": round(area, 2),
                 "diameter_mm": round(diameter, 2)},
        "orientation_deg": int(round(st["angle"])) % 180, "relief": "recessed" if d["cls"] != "corrosion" else "flat",
        "depth_mm": -round(abs(float(depth)), 3) if (with_depth and depth is not None) else None,
        "evidence": {"basis": f"twin perfect detector ({d['cls']}.{d.get('kind')}, {frac:.0%} visible)"},
    }
    if near:
        rec["near"] = near
    return rec


def _unknown_record(x: dict[str, Any], plan: dict[str, Any], k: int) -> dict[str, Any] | None:
    p = x.get("params") or {}
    holes = {h["id"]: h for h in plan.get("holes", [])}
    if p.get("hole") in holes:
        h = holes[p["hole"]]
        u, v = float(h["u"]), float(h["v"])
        dd = float(h["d_mm"]) * float(h.get("ovality", 1.0)) + 2 * float(p.get("w_mm", 0.3))
    elif "u" in p and "v" in p:
        u, v = float(p["u"]), float(p["v"])
        dd = float(p.get("d_mm") or 2 * float(p.get("r_mm", 0.5)))
    else:
        return None
    return {"id": "", "twin_id": x.get("id") or f"X{k}", "type": "unknown", "confidence": 0.5,
            "location_mm": {"x": round(u, 1), "y": round(v, 1)},
            "size": {"length_mm": round(dd, 2), "width_mm": round(dd, 2), "area_mm2": round(math.pi * dd * dd / 4, 2),
                     "diameter_mm": round(dd, 2)},
            "orientation_deg": 0, "relief": "unknown", "depth_mm": None,
            "evidence": {"basis": f"anomaly map (twin distractor: {x['kind']})"}}


def records_from_plan(plan: dict[str, Any], spec: dict[str, Any], *, with_depth: bool = False,
                      unknown_kinds: Sequence[str] = UNKNOWN_KINDS, conf: float = 0.95,
                      timestamp: str | None = None) -> dict[str, Any]:
    """The report a perfect detector would give for one scene plan.

    Steel spec: record 3.0 (steel_record_from_plan). Aero spec: the aerospace report (schema 2.1);
    fastener diameter D comes from plan['fasteners']['D_mm'] when the sampler wrote it, else from
    the spec; the plate thickness from the plan. Defects get ids D1.. and keep their plan id as
    twin_id.
    """
    if is_steel(spec):
        return steel_record_from_plan(plan, spec, conf=conf, timestamp=timestamp)
    plate = plan["plate"]
    W, H = (float(x) for x in plate["size_mm"])
    D = _plan_D(plan) or float(spec["part"]["fastener_d_mm"])
    t = float(plate.get("thickness_mm") or spec["part"].get("thickness_mm", 3.0))
    fasteners = fasteners_from_plan(plan)
    heads = [(f.u, f.v, f.d_mm / 2) for f in fasteners if f.cls == "rivet"]
    labelled = [d for d in plan.get("defects", []) if d.get("label", True) and d.get("cls") in SURFACE
                and d.get("visible", True)]
    clear_of = spec["defects"]["dent"].get("clear_of", ["crack"])
    damage = [(d["id"], d["footprint"]["polys_mm"]) for d in labelled if d["cls"] in clear_of and d.get("footprint")]
    defects = [r for d in labelled if (r := _defect_record(d, plan, heads, damage, with_depth, conf))]
    defects += [r for k, x in enumerate(plan.get("distractors", []), 1)
                if x.get("kind") in unknown_kinds and (r := _unknown_record(x, plan, k))]
    for k, r in enumerate(defects, 1):
        r["id"] = f"D{k}"
    source = {k: plan.get(k) for k in ("scene_id", "seed", "source", "split", "mode")}
    return build_aero_report(plan.get("scene_id", "twin"), defects, fasteners, spec, (W, H), {},
                             detector="twin perfect detector", D=D, t=t, timestamp=timestamp,
                             extra={"source": source})


# ----------------------------------------------------------------------------- steel records (3.0)

def steel_part_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """{'material', 'thickness_mm', 'size_mm'} of a plan's plate (a scene/1 plan without a kind is aluminium)."""
    plate = plan.get("plate") or {}
    mat = plate.get("material") if isinstance(plate.get("material"), dict) else {}
    kind = mat.get("kind") or plate.get("material_kind") or ("aluminium" if plate.get("finish") else None)
    return {"material": kind, "thickness_mm": plate.get("thickness_mm"), "size_mm": plate.get("size_mm")}


def sidecar_part(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Part info from a sidecar: its 'part' block (post: material, thickness_mm, size_mm), else a
    'plate' block, else just the plate size."""
    block = next((b for b in (sidecar.get("part"), sidecar.get("plate")) if isinstance(b, dict)), {})
    mat = block.get("material")
    kind = mat.get("kind") if isinstance(mat, dict) else mat
    return {"material": kind or sidecar.get("material"),
            "thickness_mm": block.get("thickness_mm") or sidecar.get("thickness_mm"),
            "size_mm": block.get("size_mm") or sidecar.get("plate_size_mm")}


def near_edge_mm(near: dict[str, Any] | None) -> float | None:
    """The smallest edge-to-edge distance to a plate edge or hole in a `near` dict (None if unknown)."""
    vals = [float(near[k]) for k in EDGE_KEYS if (near or {}).get(k) is not None]
    return round(max(0.0, min(vals)), 3) if vals else None


def _polys_stats(polys: Sequence[Any]) -> tuple[float, float, float] | None:
    """(area mm2, centroid u, centroid v) of outlines (shoelace; vertex mean when degenerate)."""
    arrs = [np.asarray(p, float).reshape(-1, 2) for p in polys if len(p) >= 3]
    if not arrs:
        return None
    a_sum, cu, cv = 0.0, 0.0, 0.0
    for q in arrs:
        x, y = q[:, 0], q[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cross = x * y1 - x1 * y
        a = 0.5 * cross.sum()
        if abs(a) > 1e-12:
            cu += ((x + x1) * cross).sum() / 6.0
            cv += ((y + y1) * cross).sum() / 6.0
            a_sum += a
    allp = np.vstack(arrs)
    if abs(a_sum) < 1e-9:
        return 0.0, float(allp[:, 0].mean()), float(allp[:, 1].mean())
    return abs(a_sum), float(cu / a_sum), float(cv / a_sum)


def _plate_edge_gap(box: Sequence[float], size_mm: Sequence[float]) -> float:
    u0, v0, u1, v1 = (float(x) for x in box)
    W, H = (float(x) for x in size_mm)
    return max(0.0, min(u0, v0, W - u1, H - v1))


def defect_from_label(lbl: dict[str, Any], conf: float = 1.0, k: int = 1, part: dict[str, Any] | None = None,
                      params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """One ground-truth label as an unjudged record-3.0 defect (None if not a steel class).

    Accepts a plan defect (cls, params, footprint {polys_mm, measure}, near), a sidecar defect
    (cls, measure, near, centre_mm, bbox_mm, bbox_grid), a ground_truth.csv-like row (cls, u_mm,
    v_mm and measure columns) or an already record-style dict (type, size, location_mm). The
    confidence is always `conf` (ground truth is not a detection). Depth is never copied: the
    camera cannot see it. box_px is the sidecar's bbox_grid (grid px) when present.
    """
    cls = lbl.get("cls") or lbl.get("type")
    if cls not in STEEL_CLASSES:
        return None
    fp = lbl.get("footprint") or {}
    params = {**(lbl.get("params") or {}), **(params or {})}
    raw = lbl.get("measure") or fp.get("measure") or lbl.get("size") or {k2: lbl[k2] for k2 in SIZE_KEYS if k2 in lbl}
    size = {k2: float(raw[k2]) for k2 in SIZE_KEYS if raw.get(k2) not in (None, "")}
    polys = fp.get("polys_mm") or lbl.get("polys_mm") or []
    stats = _polys_stats(polys)
    if lbl.get("location_mm"):
        u, v = float(lbl["location_mm"]["x"]), float(lbl["location_mm"]["y"])
    elif lbl.get("centre_mm") is not None:
        u, v = (float(x) for x in lbl["centre_mm"][:2])
    elif stats is not None:
        u, v = stats[1], stats[2]
    elif lbl.get("u_mm") is not None:
        u, v = float(lbl["u_mm"]), float(lbl["v_mm"])
    elif lbl.get("bbox_mm") is not None:
        b = [float(x) for x in lbl["bbox_mm"]]
        u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    elif "u" in params:
        u, v = float(params["u"]), float(params["v"])
    else:
        return None
    if not size.get("area_mm2"):
        if stats is not None and stats[0] > 0:
            size["area_mm2"] = stats[0]
        elif size.get("diameter_mm"):
            size["area_mm2"] = math.pi * size["diameter_mm"] ** 2 / 4
    if not size.get("diameter_mm") and size.get("area_mm2"):
        size["diameter_mm"] = math.sqrt(4 * size["area_mm2"] / math.pi)
    if cls in ("scratch", "crack") and "width_mm" not in size and params.get("width_mm") is not None:
        size["width_mm"] = float(params["width_mm"])
    if cls == "pit" and "pit_d_mm" not in size:
        pd = params.get("pit_d_mm")
        if isinstance(pd, (list, tuple)) and pd:
            size["pit_d_mm"] = float(max(pd))
        elif params.get("d_mm") is not None:
            size["pit_d_mm"] = float(params["d_mm"])
    if cls == "pit" and "count" not in size:
        size["count"] = float(params.get("n", 1))
    near = lbl.get("near_edge_mm")
    if near is None:
        near = near_edge_mm(lbl.get("near"))
    if near is None and part and part.get("size_mm"):
        box = lbl.get("bbox_mm")
        if box is None and polys:
            allp = np.vstack([np.asarray(p, float).reshape(-1, 2) for p in polys])
            box = [*allp.min(axis=0), *allp.max(axis=0)]
        if box is not None:
            near = round(_plate_edge_gap(box, part["size_mm"]), 3)
    box_px = lbl.get("box_px") or lbl.get("bbox_grid")
    rec = {"id": f"D{k}", "type": cls, "confidence": float(conf),
           "box_px": [float(x) for x in box_px] if box_px is not None and len(box_px) == 4 else None,
           "location_mm": {"x": round(u, 2), "y": round(v, 2)},
           "size": {k2: (int(round(x)) if k2 == "count" else round(x, 3)) for k2, x in size.items()},
           "near_edge_mm": None if near is None else float(near),
           "kind": lbl.get("kind"), "twin_id": lbl.get("id") or lbl.get("defect_id") or lbl.get("twin_id")}
    if cls == "corrosion" and (params.get("pitted") or lbl.get("pitted")):
        rec["pitted"] = True
    return rec


def defects_from_labels(labels: Iterable[dict[str, Any]], conf: float = 1.0, part: dict[str, Any] | None = None,
                        plan_defects: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Labelled ground-truth defects (label not False) as unjudged record defects D1, D2, ..."""
    out: list[dict[str, Any]] = []
    for lbl in labels:
        if lbl.get("label") is False:
            continue
        params = ((plan_defects or {}).get(lbl.get("id")) or {}).get("params")
        d = defect_from_label(lbl, conf, len(out) + 1, part, params)
        if d is not None:
            out.append(d)
    return out


def labels_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """The plan's labelled steel defects with a footprint (the ground truth in plate mm)."""
    return [d for d in plan.get("defects", []) if d.get("label", True) and d.get("cls") in STEEL_CLASSES
            and (d.get("footprint") or {}).get("polys_mm")]


def gt_verdict(labels_mm: Iterable[dict[str, Any]], spec: dict[str, Any], part: dict[str, Any] | None = None,
               visible: dict[str, Any] | None = None, plan: dict[str, Any] | None = None) -> str:
    """Ground-truth verdict: the steel rules on the labelled defects at confidence 1.

    labels_mm: plan defects, sidecar defects, ground_truth.csv-like rows or record-style dicts
    (see defect_from_label). part = {'material', 'thickness_mm', 'size_mm'} (spec defaults fill the
    rest). visible: optional {defect id: bool} (a sidecar variant's map) to drop labels the camera
    could not see; by default every labelled defect counts. plan: adds the plan's defect params
    (pitted rust, pit sizes) to sidecar labels, as records_from_sidecar does.
    """
    labels = [lb for lb in labels_mm if visible is None or visible.get(lb.get("id")) is not False]
    p = steel_part(spec, part)
    plan_defects = {d["id"]: d for d in (plan or {}).get("defects", []) if "id" in d}
    return judge_steel(defects_from_labels(labels, 1.0, p, plan_defects), spec, p)[0]


def capture_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """A nominal capture block for an unrendered plan: gate PASS, ambient lux and device mode from the plan."""
    dev = ((plan.get("extras") or {}).get("device") or {}).get("mode") or "open"
    return capture_block({"mode": "burst5", "device_mode": dev, "lux_est": (plan.get("ambient") or {}).get("lux"),
                          "gate": "PASS", "gate_message": ""})


def sidecar_views(sidecar: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    """(variant k, arm, info) per rendered variant and arm.

    info merges the sidecar's top-level condition / lux / device_mode with the variant's and, when
    the variant has `arms: {burst: {gate, reasons, message, metrics, visible, ...}, naive: {...}}`
    (post, sidecar/2), that arm's block; gate_message and gate_reasons are normalised from the
    arm's message / reasons. The variant-level visible map (the train arm's) is used when an arm
    has none of its own.
    """
    top = {k: sidecar[k] for k in ("condition", "lux", "device_mode", "gate", "metrics", "arm") if k in sidecar}
    out = []
    for i, var in enumerate(sidecar.get("variants") or [{}]):
        base = {**top, **{k: v for k, v in var.items() if k != "arms"}}
        k = int(base.get("k", i))
        arms = var.get("arms")
        if not (isinstance(arms, dict) and arms):
            out.append((k, str(base.get("arm") or "burst"), base))
            continue
        for arm, a in arms.items():
            info = {**{kk: vv for kk, vv in base.items() if kk not in ("gate", "gate_message", "gate_reasons", "metrics")},
                    **a, "arm": arm}
            info["gate_message"] = a.get("message", a.get("gate_message", ""))
            info["gate_reasons"] = list(a.get("reasons", a.get("gate_reasons")) or [])
            out.append((k, arm, info))
    return out


def capture_from_view(info: dict[str, Any], arm: str = "burst") -> dict[str, Any]:
    """Capture block for one sidecar variant and arm. The naive arm has no gate: it never refuses.

    lux_est is the device's own estimate (lightproc metrics lux_est; None when the ambient
    saturates, and for the naive arm, which has no ambient frame), never the twin's true lux
    (that is in the record's source). Only metrics without a lux_est key fall back to it."""
    gate = info.get("gate")
    if arm == "naive" or gate is None:
        gate = "PASS"
    elif isinstance(gate, str):
        gate = {"verdict": gate, "message": info.get("gate_message") or "", "reasons": info.get("gate_reasons") or []}
    metrics = info.get("metrics") or {}
    if arm == "naive":
        lux = None
    else:
        lux = metrics["lux_est"] if "lux_est" in metrics else info.get("lux")
    return capture_from_metrics(metrics, gate, mode="naive" if arm == "naive" else "burst5",
                                device_mode=info.get("device_mode") or "open", lux_est=lux)


def _source(obj: dict[str, Any], **more: Any) -> dict[str, Any]:
    src = {k: obj.get(k) for k in ("scene_id", "seed", "split", "source") if obj.get(k) is not None}
    src.update({k: v for k, v in more.items() if v is not None})
    return src


def steel_record_from_plan(plan: dict[str, Any], spec: dict[str, Any], *, conf: float = 0.95,
                           capture: dict[str, Any] | None = None, timestamp: str | None = None) -> dict[str, Any]:
    """Record 3.0 a perfect detector would give for a steel scene plan (every labelled defect;
    capture from `capture` or nominal from the plan)."""
    part = steel_part(spec, steel_part_from_plan(plan))
    defects = defects_from_labels(labels_from_plan(plan), conf, part)
    verdict, judged = judge_steel(defects, spec, part)
    amb = plan.get("ambient") or {}
    dev = ((plan.get("extras") or {}).get("device") or {}).get("mode")
    return build_steel_record(plan.get("scene_id", "twin"), part, judged, verdict, capture or capture_from_plan(plan),
                              {}, {"vision": PERFECT, "advisor": None}, spec=spec, timestamp=timestamp,
                              extra={"source": _source(plan, condition=amb.get("condition"), device_mode=dev)})


def records_from_sidecar(sidecar: dict[str, Any], spec: dict[str, Any], *, plan: dict[str, Any] | None = None,
                         part: dict[str, Any] | None = None, conf: float = 0.95, arms: Sequence[str] | None = None,
                         timestamp: str | None = None) -> list[dict[str, Any]]:
    """One perfect-detector record 3.0 per sidecar variant and arm.

    Defects: the sidecar's labelled defects that are visible in that variant (measure, near ->
    near_edge_mm, centre_mm, bbox_grid -> box_px); plan params (pitted rust, pit sizes) when the
    plan is given. Capture: the variant's gate and metrics; a RETAKE gate gives a RETAKE record
    with no defects. Part: `part`, else the plan's plate, else the sidecar's plate block.
    """
    part = steel_part(spec, part or (steel_part_from_plan(plan) if plan else sidecar_part(sidecar)))
    plan_defects = {d["id"]: d for d in (plan or {}).get("defects", []) if "id" in d}
    scene = sidecar.get("scene_id", "twin")
    out = []
    for k, arm, info in sidecar_views(sidecar):
        if arms and arm not in arms:
            continue
        vis = info.get("visible") or {}
        labels = [d for d in sidecar.get("defects", []) if vis.get(d.get("id")) is not False]
        defects = defects_from_labels(labels, conf, part, plan_defects)
        verdict, judged = judge_steel(defects, spec, part)
        cap = capture_from_view(info, arm)
        pid = f"{scene}_v{k}" + ("" if arm == "burst" else f"_{arm}")
        src = _source(sidecar, variant=k, arm=arm, condition=info.get("condition"), lux=info.get("lux"),
                      device_mode=info.get("device_mode"))
        out.append(build_steel_record(pid, part, judged, verdict, cap, {}, {"vision": PERFECT, "advisor": None},
                                      spec=spec, timestamp=timestamp, extra={"source": src}))
    return out


def _load_json_items(src: str | Path | dict | Iterable[Any]) -> list[dict[str, Any]]:
    if isinstance(src, (str, Path, dict)):
        src = [src]
    items: list[dict[str, Any]] = []
    for s in src:
        if isinstance(s, dict):
            items.append(s)
            continue
        p = Path(s)
        if p.is_dir():
            subs = [p / d for d in ("sidecar", "plans") if (p / d).is_dir()] or [p]
            items += [json.loads(f.read_text()) for d in subs for f in sorted(d.glob("*.json"))]
        else:
            items.append(json.loads(p.read_text()))
    return items


def load_steel_cases(src: str | Path | dict | Iterable[Any], n: int | None = None
                     ) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    """(kind, item, paired plan) cases: every sidecar (with its plan when one is loaded) and every
    plan without a sidecar. Directories are read through their sidecar/ and plans/ subdirectories."""
    items = _load_json_items(src)
    sidecars = [x for x in items if str(x.get("schema", "")).startswith(SIDECAR_PREFIX)]
    plans = [x for x in items if x.get("schema", PLAN_SCHEMA) in PLAN_SCHEMAS and "plate" in x]
    by_id = {p.get("scene_id"): p for p in plans}
    done = {s.get("scene_id") for s in sidecars}
    cases = [("sidecar", s, by_id.get(s.get("scene_id"))) for s in sidecars]
    cases += [("plan", p, None) for p in plans if p.get("scene_id") not in done]
    return cases[:n] if n else cases


def run_steel_regression(src: str | Path | Iterable[Any], spec: dict[str, Any], be: Any, n: int | None = None,
                         out: str | Path | None = None, questions: Sequence[str] = STEEL_QUESTIONS) -> dict[str, Any]:
    """Steel rules + advisor over plans and sidecars (see run_regression); summary dict.

    Per record: schema (3.0), verdict = worst disposition (or RETAKE with no defects), the
    ground-truth verdict (gt_verdict on the labels; a sidecar variant may differ only through
    defects the camera could not see, counted as hidden_by_visibility), every advisor reply
    checked by check_reply, and a RETAKE explanation that carries the capture advice.
    """
    from ..llm import Advisor, MockBackend, capture_advice, check_reply
    from ..rules import worst

    cases = load_steel_cases(src, n)
    counters = {k: Counter() for k in ("verdict", "gate", "defect", "backend")}
    by_condition: dict[str, Counter] = {}
    failures: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    gt_mismatch: list[dict[str, Any]] = []
    hidden = replies = records = 0
    fh = None
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fh = open(out, "w")
    try:
        for kind, item, plan in cases:
            if kind == "plan":
                recs = [steel_record_from_plan(item, spec)]
                labels, part_src = labels_from_plan(item), steel_part_from_plan(item)
            else:
                recs = records_from_sidecar(item, spec, plan=plan)
                labels = item.get("defects", [])
                part_src = steel_part_from_plan(plan) if plan else sidecar_part(item)
            truth = gt_verdict(labels, spec, part_src, plan=plan)
            for rec in recs:
                records += 1
                rid = rec["part_id"]
                fail = [f"schema: {e}" for e in validate_record(rec)]
                if rec["verdict"] == "RETAKE":
                    if rec["defects"]:
                        fail.append("RETAKE record carries defects")
                elif rec["verdict"] != worst(d["disposition"] for d in rec["defects"]):
                    fail.append("verdict is not the worst disposition")
                if rec["verdict"] != "RETAKE" and rec["verdict"] != truth:
                    if kind == "sidecar" and len(rec["defects"]) < len(defects_from_labels(labels)):
                        hidden += 1
                    else:
                        gt_mismatch.append({"part_id": rid, "verdict": rec["verdict"], "gt": truth})
                advisor = Advisor(be, spec, rec)
                convo = []
                for q in (None, *questions):
                    text = advisor.explain() if q is None else advisor.ask(q)
                    last = dict(advisor.last)
                    replies += 1
                    counters["backend"][last["backend"]] += 1
                    if last["problems"] and last["rejected"] is not None:
                        violations.append({"part_id": rid, "question": q, "problems": last["problems"]})
                    final = check_reply(text, rec, spec, context=q or "")
                    if final:
                        fail.append(f"reply to {q or 'explain'!r} breaks the guardrail: {final}")
                    if q is None and rec["verdict"] == "RETAKE" and last["backend"] == MockBackend.name:
                        advice = capture_advice(rec, spec, general=False)
                        if not all(a in text for a in advice):
                            fail.append("RETAKE explanation gives no capture advice")
                    convo.append({"q": q, "backend": last["backend"], "problems": last["problems"],
                                  "rejected": last["rejected"], "text": text})
                counters["verdict"][rec["verdict"]] += 1
                counters["gate"][rec["capture"]["gate"]] += 1
                counters["defect"].update(f"{d['type']}:{d['severity']}:{d['disposition']}" for d in rec["defects"])
                cond = (rec.get("source") or {}).get("condition") or "none"
                by_condition.setdefault(cond, Counter())[rec["verdict"]] += 1
                failures += [{"part_id": rid, "problem": p} for p in fail]
                if fh:
                    fh.write(json.dumps({"part_id": rid, "source": rec.get("source"), "verdict": rec["verdict"],
                                         "gt_verdict": truth, "gate": rec["capture"]["gate"],
                                         "defects": [[d["id"], d.get("twin_id"), d["type"], d["severity"], d["disposition"]]
                                                     for d in rec["defects"]],
                                         "replies": convo, "failures": fail}, ensure_ascii=False) + "\n")
    finally:
        if fh:
            fh.close()
    return {"domain": "steel_surface", "cases": len(cases), "records": records,
            "by_verdict": dict(counters["verdict"]), "by_gate": dict(counters["gate"]),
            "by_condition": {c: dict(v) for c, v in sorted(by_condition.items())},
            "by_defect": dict(sorted(counters["defect"].items())), "replies": replies,
            "backends": dict(counters["backend"]), "guardrail_violations": len(violations),
            "violation_examples": violations[:10], "failures": failures, "gt_mismatches": gt_mismatch,
            "hidden_by_visibility": hidden, "out": str(out) if out is not None else None}


# ----------------------------------------------------------------------------- expectations

def expected_findings(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Designed findings: the coupon `expect` list of a replica, else the sampler's fastener violations."""
    if plan.get("expect") is not None:
        return list(plan["expect"])
    out = []
    for v in (plan.get("fasteners") or {}).get("violations", []) or []:
        kind = VIOLATION_TYPES.get(v.get("kind"))
        if kind == "pitch":
            out.append({"type": kind, "holes": list(v["holes"])})
        elif kind:
            out += [{"type": kind, "hole": h} for h in v["holes"]]
    return out


def check_expectations(plan: dict[str, Any], report: dict[str, Any]) -> list[str]:
    """Mismatches: designed findings the rules missed; for coupon replicas also unexpected fastener findings.

    A designed missing fastener also matches an elongated hole (the hole was made oval too) or a
    lone open hole when the same hole was displaced (it has an edge-distance or pitch finding, so
    it no longer sits on the row pitch).
    """
    findings = report.get("findings", [])
    by_twin = {d.get("twin_id"): d for d in report["defects"]}
    exp = expected_findings(plan)
    oval = {h["id"] for h in plan.get("holes", []) if float(h.get("ovality", 1.0)) > 1.0}
    displaced = {h for f in findings if f["type"] in ("edge_distance", "pitch") for h in f["fasteners"]}
    problems, matched = [], set()
    for e in exp:
        t = e["type"]
        if t in FASTENER_FINDINGS:
            want = set(e["holes"]) if t == "pitch" else {e["hole"]}
            ok_types = {t}
            if t == "missing_fastener":
                ok_types |= {"elongated_hole"} if want & oval else set()
                ok_types |= {"lone_open_hole"} if want & displaced else set()
            hits = [f["id"] for f in findings if f["type"] in ok_types and
                    (set(f["fasteners"]) == want if t == "pitch" else want <= set(f["fasteners"]))]
            matched.update(hits)
            if not hits:
                problems.append(f"missed {t} at {sorted(want)}")
            continue
        d = by_twin.get(e.get("defect") or e.get("distractor"))
        if t.startswith("dent_"):
            ok = d is not None and d.get("dent_class") == t[5:]
        elif t == "crack_indication":
            ok = d is not None and d["type"] == "crack" and d["disposition"] == "REJECT"
        elif t == "corrosion":
            ok = d is not None and d["disposition"] == "HOLD_FOR_MRB"
        elif t == "hold_unknown":
            ok = d is not None and d["disposition"] == "HOLD_FOR_QA"
        else:
            continue
        if not ok:
            got = "not reported" if d is None else f"{d['type']} {d.get('dent_class', '')} {d['disposition']}".strip()
            problems.append(f"expected {t} for {e.get('defect') or e.get('distractor')}, got {got}")
    if plan.get("expect") is not None:
        for f in findings:
            if f["type"] in FASTENER_FINDINGS and f["id"] not in matched:
                problems.append(f"unexpected {f['type']} at {f['fasteners']}")
    return problems


# ----------------------------------------------------------------------------- regression

def load_plans(src: str | Path | dict | Iterable[Any], n: int | None = None) -> list[dict[str, Any]]:
    """Scene plans from a directory (or its plans/ subdir), JSON files, dicts, or a list of these."""
    if isinstance(src, (str, Path, dict)):
        src = [src]
    plans: list[dict[str, Any]] = []
    for s in src:
        if isinstance(s, dict):
            plans.append(s)
            continue
        p = Path(s)
        if p.is_dir():
            d = p / "plans" if (p / "plans").is_dir() else p
            plans += [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
        else:
            plans.append(json.loads(p.read_text()))
    plans = [p for p in plans if p.get("schema", PLAN_SCHEMA) in PLAN_SCHEMAS and "plate" in p]
    return plans[:n] if n else plans


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.exists() or p.is_absolute() else ROOT / p


def _backend(backend: Any, tokenizer: str | Path | None = None, runner: str | None = None):
    """mock, a backend object, an llm config dict or YAML file, or a .pte path (ExecuTorch runner
    from `runner`, $TRINETRA_LLAMA_RUNNER or llama_main on PATH; tokenizer.model next to the .pte)."""
    from ..llm import ExecuTorchBackend, MockBackend, build_backend

    if backend in (None, "mock"):
        return MockBackend()
    if isinstance(backend, dict):
        return build_backend(backend, ROOT)
    if isinstance(backend, (str, Path)) and str(backend).endswith(".pte"):
        pte = _resolve(backend)
        tok = _resolve(tokenizer) if tokenizer else pte.with_name("tokenizer.model")
        # the Pi's context (configs/rig.pi.yaml, app.yaml): a worst-case steel prompt is ~1,300 tokens,
        # so ExecuTorchBackend's 1536 default would leave almost no room for the answer
        return ExecuTorchBackend(runner or os.environ.get("TRINETRA_LLAMA_RUNNER", "llama_main"), str(pte), str(tok),
                                 seq_len=PTE_SEQ_LEN)
    if isinstance(backend, (str, Path)):
        import yaml

        cfg = yaml.safe_load(_resolve(backend).read_text())
        return build_backend(cfg.get("llm", cfg), ROOT)
    return backend


def _validator():
    try:
        import jsonschema
    except ImportError:
        return None
    return jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def run_regression(plans: str | Path | Iterable[Any], spec_path: str | Path = STEEL_SPEC,
                   backend: Any = "mock", n: int | None = None, out: str | Path | None = None,
                   questions: Sequence[str] | None = None, with_depth: bool = False, *,
                   tokenizer: str | Path | None = None, runner: str | None = None) -> dict[str, Any]:
    """Rules + advisor over many cases; writes one JSONL line per case and returns a summary.

    The spec's domain picks the harness: steel (run_steel_regression; plans and/or sidecars) or
    aerospace (plans). backend: 'mock' (templates), a .pte path (ExecuTorch; tokenizer and runner
    as in _backend), an llm config dict / YAML path (see llm.build_backend), or a backend object.
    A guardrail violation is a model reply that check_reply rejected (the advisor then answered
    from the template). A failure is a bug in the rules or the harness: a schema error, a verdict
    that is not the worst disposition, a missing NCR (aero), or a final reply that breaks the
    guardrail. An aero mismatch is a designed finding (coupon `expect`, sampler violation) the
    rules did not reproduce; replica jitter can legitimately cause one.
    """
    from ..rules import load_spec

    spec = load_spec(_resolve(spec_path))
    be = _backend(backend, tokenizer, runner)
    if out is None and isinstance(plans, (str, Path)) and Path(plans).is_dir():
        out = Path(plans) / ".." / "regress" / "cases.jsonl"
    if is_steel(spec):
        return run_steel_regression(plans, spec, be, n, out, STEEL_QUESTIONS if questions is None else questions)
    return _run_aero_regression(load_plans(plans, n), spec, be, out, QUESTIONS if questions is None else questions,
                                with_depth)


def _run_aero_regression(items: list[dict[str, Any]], spec: dict[str, Any], be: Any, out: str | Path | None,
                         questions: Sequence[str], with_depth: bool) -> dict[str, Any]:
    from ..llm import Advisor, check_reply
    from ..rules import worst

    validator = _validator()
    by_verdict: Counter = Counter()
    by_finding: Counter = Counter()
    by_defect: Counter = Counter()
    backends: Counter = Counter()
    failures: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    replies = 0
    fh = None
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fh = open(out, "w")
    try:
        for plan in items:
            sid = plan.get("scene_id", "?")
            report = records_from_plan(plan, spec, with_depth=with_depth)
            case_miss = check_expectations(plan, report)
            case_fail = []
            if validator is not None:
                case_fail += [f"schema: {e.message}" for e in validator.iter_errors(json.loads(json.dumps(report)))]
            if report["verdict"] != worst(i["disposition"] for i in [*report["defects"], *report["findings"]]):
                case_fail.append("verdict is not the worst disposition")
            if (report["ncr"] is None) != (report["verdict"] == "PASS"):
                case_fail.append("NCR draft missing or present on a PASS")
            advisor = Advisor(be, spec, report)
            convo = []
            for q in (None, *questions):
                text = advisor.explain() if q is None else advisor.ask(q)
                last = dict(advisor.last)
                replies += 1
                backends[last["backend"]] += 1
                if last["problems"] and last["rejected"] is not None:
                    violations.append({"scene_id": sid, "question": q, "problems": last["problems"]})
                final = check_reply(text, report, spec, context=q or "")
                if final:
                    case_fail.append(f"reply to {q or 'explain'!r} breaks the guardrail: {final}")
                convo.append({"q": q, "backend": last["backend"], "problems": last["problems"],
                              "rejected": last["rejected"], "text": text})
            by_verdict[report["verdict"]] += 1
            by_finding.update(f["type"] for f in report["findings"])
            by_defect.update(f"{d['type']}:{d['disposition']}" for d in report["defects"])
            failures += [{"scene_id": sid, "problem": p} for p in case_fail]
            mismatches += [{"scene_id": sid, "source": plan.get("source"), "problem": p} for p in case_miss]
            if fh:
                fh.write(json.dumps({
                    "scene_id": sid, "source": report.get("source"), "verdict": report["verdict"],
                    "findings": [[f["id"], f["type"], f["fasteners"], f["disposition"]] for f in report["findings"]],
                    "defects": [[d["id"], d.get("twin_id"), d["type"], d.get("dent_class"), d["disposition"]]
                                for d in report["defects"]],
                    "replies": convo, "failures": case_fail, "mismatches": case_miss}) + "\n")
    finally:
        if fh:
            fh.close()
    return {"plans": len(items), "by_verdict": dict(by_verdict), "by_finding": dict(by_finding),
            "by_defect": dict(sorted(by_defect.items())), "replies": replies, "backends": dict(backends),
            "guardrail_violations": len(violations), "violation_examples": violations[:10],
            "failures": failures, "mismatches": mismatches, "cases": str(out) if out is not None else None}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trinetra-twin regress",
                                description="Rules + advisor regression on twin scene plans or sidecars (perfect-detector records).")
    p.add_argument("plans", nargs="+", help="dataset or plan/sidecar directories, or plan/sidecar JSON files")
    p.add_argument("--spec", default=STEEL_SPEC, help="configs/spec_steel.yaml (default) or configs/spec_aero.yaml")
    p.add_argument("--backend", default="mock",
                   help="mock, a Llama .pte (ExecuTorch), or a YAML file with an llm section (backend: executorch ...)")
    p.add_argument("--tokenizer", help="tokenizer for --backend <.pte> (default: tokenizer.model next to it)")
    p.add_argument("--runner", help="llama_main runner for --backend <.pte> (default: $TRINETRA_LLAMA_RUNNER or llama_main)")
    p.add_argument("--n", type=int, help="at most this many plans or sidecars")
    p.add_argument("--out", help="JSONL file for the cases (default: <plans>/../regress/cases.jsonl for one directory)")
    p.add_argument("--with-depth", action="store_true", help="aero only: give dents their plan depth (the camera cannot)")
    p.add_argument("--question", action="append", help="follow-up question to ask (repeatable)")
    a = p.parse_args(argv)
    src: Any = a.plans[0] if len(a.plans) == 1 else a.plans
    summary = run_regression(src, a.spec, a.backend, a.n, a.out, tuple(a.question) if a.question else None,
                             a.with_depth, tokenizer=a.tokenizer, runner=a.runner)
    json.dump(summary, sys.stdout, indent=1)
    print()
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
