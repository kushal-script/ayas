"""Aerospace QC rules: fastener geometry, Piper-style dent classes and Q219P dispositions.

Plate frame (docs/twin/DESIGN.md section 1): mm from the datum corner, u right, v down. A
defect's ``location_mm`` {x, y} is its centre (u, v). Angles are degrees from +u toward +v.
D is the nominal fastener diameter (spec ``part.fastener_d_mm``).

Distance conventions (the same as twin/geometry.py and twin/coupons.py):
- Edge distance and pitch run from hole centres (AC 43.13-1B 4-57c).
- Dent-to-fastener distance is edge to edge between the dent circle (its measured diameter)
  and the fastener's VISIBLE outline: the head circle for a rivet, the hole ellipse for an
  open hole (Piper SL 1095A: at least 2 x the fastener diameter from the nearest fastener).

Dispositions follow Boeing Q219P: the supplier may only rework or scrap, anything else is
HOLD_FOR_MRB. The rules only flag and grade; the qualified inspector and the customer MRB
decide. Thresholds come from configs/spec_aero.yaml (a public-source DEMO table).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .rules import worst, zone_for
from .tiles import CLASSES, ELONGATED

FASTENER_CLASSES = ("rivet", "open_hole")


# ----------------------------------------------------------------------------- fasteners

@dataclass
class Fastener:
    """A rivet or an open hole in plate mm.

    d_mm is the visible outline: the head diameter for a rivet, the minor (drill) diameter for
    an open hole, whose major axis is d_mm * ovality along angle_deg.
    """

    id: str
    cls: str
    u: float
    v: float
    d_mm: float
    ovality: float = 1.0
    angle_deg: float = 0.0
    conf: float = 1.0

    @property
    def major_mm(self) -> float:
        return self.d_mm * self.ovality

    def radius_toward(self, angle_deg: float) -> float:
        """Centre-to-visible-outline distance along angle_deg (ellipse polar radius)."""
        b = self.d_mm / 2
        a = b * self.ovality
        phi = math.radians(angle_deg - self.angle_deg)
        return a * b / math.hypot(b * math.cos(phi), a * math.sin(phi))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("u", "v", "d_mm", "angle_deg"):
            d[k] = round(float(d[k]), 2)
        d["ovality"], d["conf"] = round(float(d["ovality"]), 3), round(float(d["conf"]), 3)
        return d


def as_fastener(f: Fastener | dict[str, Any]) -> Fastener:
    if isinstance(f, Fastener):
        return f
    keys = ("id", "cls", "u", "v", "d_mm", "ovality", "angle_deg", "conf")
    return Fastener(**{k: f[k] for k in keys if k in f})


def _det(d: Any) -> tuple[str, float, float, float, float, float]:
    """(class name, conf, u0, v0, u1, v1) from {cls, conf, box_mm}, {cls, conf, x0..y1} or a tiles.Det in mm."""
    if isinstance(d, dict):
        cls = d.get("cls", d.get("type"))
        conf = d.get("conf", d.get("confidence", 1.0))
        box = d["box_mm"] if "box_mm" in d else (d["x0"], d["y0"], d["x1"], d["y1"])
    else:
        cls, conf, box = d.cls, d.conf, (d.x0, d.y0, d.x1, d.y1)
    if isinstance(cls, (int, np.integer)):
        cls = CLASSES[int(cls)]
    u0, v0, u1, v1 = (float(x) for x in box)
    return str(cls), float(conf), min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)


def fasteners_from_dets(dets_mm: Iterable[Any]) -> list[Fastener]:
    """Rivet and open-hole detections (boxes in plate mm) -> Fasteners, ids H1.. in reading order.

    A box only gives an axis-aligned size: refine open holes with refine_hole first when the
    grid image is at hand.
    """
    out = []
    for d in dets_mm:
        cls, conf, u0, v0, u1, v1 = _det(d)
        if cls not in FASTENER_CLASSES:
            continue
        w, h = u1 - u0, v1 - v0
        if cls == "rivet":
            f = Fastener("", cls, (u0 + u1) / 2, (v0 + v1) / 2, (w + h) / 2, 1.0, 0.0, conf)
        else:
            lo, hi = min(w, h), max(w, h)
            f = Fastener("", cls, (u0 + u1) / 2, (v0 + v1) / 2, lo, hi / lo if lo > 0 else 1.0,
                         0.0 if w >= h else 90.0, conf)
        out.append(f)
    out.sort(key=lambda f: (round(f.v), f.u))
    for k, f in enumerate(out, 1):
        f.id = f"H{k}"
    return out


def defects_from_dets(dets_mm: Iterable[Any], start: int = 1) -> list[dict[str, Any]]:
    """Surface-defect detections (boxes in plate mm) -> unjudged defect dicts D<start>...

    Box sizes only: an elongated class (scratch, crack) gets the box diagonal as its length and
    the short side as its width, which overestimates both for a diagonal line (conservative).
    """
    out = []
    for d in dets_mm:
        cls, conf, u0, v0, u1, v1 = _det(d)
        if cls in FASTENER_CLASSES:
            continue
        w, h = u1 - u0, v1 - v0
        elong = cls in ELONGATED
        length = math.hypot(w, h) if elong else max(w, h)
        area = w * h if elong else math.pi / 4 * w * h
        out.append({
            "id": f"D{start + len(out)}", "type": cls, "confidence": round(conf, 3),
            "location_mm": {"x": round((u0 + u1) / 2, 1), "y": round((v0 + v1) / 2, 1)},
            "size": {"length_mm": round(length, 2), "width_mm": round(min(w, h), 2),
                     "area_mm2": round(area, 2), "diameter_mm": round((w + h) / 2, 2)},
            "orientation_deg": 0 if w >= h else 90, "relief": "unknown", "depth_mm": None,
            "evidence": {"basis": "detector box; sizes from the box"},
        })
    return out


def refine_hole(grid_img: np.ndarray, box_px: Sequence[float], mm_per_px: float,
                pad: float = 0.3) -> tuple[float, float, float, float, float] | None:
    """Fit an ellipse to the dark hole inside a detection box of the rectified grid.

    Returns (cx, cy, major_mm, minor_mm, angle_deg): the centre in grid pixels (pixel centres
    on integers), the full axes in mm and the major-axis angle from +x toward +y in [0, 180).
    None if no dark region of at least 5 contour points is found.
    """
    g = np.asarray(grid_img)
    if g.ndim == 3:
        g = cv2.cvtColor(np.ascontiguousarray(g[..., :3]), cv2.COLOR_RGB2GRAY)
    H, W = g.shape[:2]
    x0, y0, x1, y1 = (float(x) for x in box_px)
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    X0, Y0 = max(0, int(math.floor(x0 - px))), max(0, int(math.floor(y0 - py)))
    X1, Y1 = min(W, int(math.ceil(x1 + px)) + 1), min(H, int(math.ceil(y1 + py)) + 1)
    crop = g[Y0:Y1, X0:X1]
    if crop.size == 0:
        return None
    if crop.dtype != np.uint8:
        lo, hi = float(crop.min()), float(crop.max())
        crop = np.zeros(crop.shape, np.uint8) if hi <= lo else ((crop - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n < 2:
        return None
    r = min(max(int(round((y0 + y1) / 2)) - Y0, 0), crop.shape[0] - 1)
    c = min(max(int(round((x0 + x1) / 2)) - X0, 0), crop.shape[1] - 1)
    lab = int(labels[r, c]) or 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == lab).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    (ex, ey), (a, b), ang = cv2.fitEllipse(cnt)
    # Contour points are the centres of the region's boundary pixels, half a pixel inside its edge.
    a, b = a + 1.0, b + 1.0
    if a < b:
        a, b, ang = b, a, ang + 90.0
    return float(ex + X0), float(ey + Y0), float(a * mm_per_px), float(b * mm_per_px), float(ang % 180.0)


# ----------------------------------------------------------------------------- rows and findings

def _runs(order: np.ndarray, s: np.ndarray, tol: float) -> list[list[int]]:
    """Split points sorted along a line into runs whose spacings agree with the run's first one."""
    runs, run, g = [], [int(order[0])], None
    for k in range(1, len(order)):
        gap = float(s[k] - s[k - 1])
        if gap < 1e-6:
            continue
        if g is None or abs(gap - g) <= tol * g:
            run.append(int(order[k]))
            g = gap if g is None else g
        else:
            runs.append(run)
            run, g = [int(order[k - 1]), int(order[k])], gap
    runs.append(run)
    return runs


def find_rows(fasteners: Sequence[Fastener], tol_mm: float = 1.0, pitch_tol: float = 0.25) -> list[list[int]]:
    """Rows: >= 3 fasteners within tol_mm of a line with consistent spacing (+-pitch_tol).

    Returns lists of indices into fasteners, sorted along each row; runs inside a longer run
    are dropped.
    """
    P = np.array([[f.u, f.v] for f in fasteners], float).reshape(-1, 2)
    found: dict[frozenset, list[int]] = {}
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d = P[j] - P[i]
            L = float(np.hypot(*d))
            if L < 1e-6:
                continue
            t = d / L
            rel = P - P[i]
            idx = np.nonzero(np.abs(rel[:, 0] * t[1] - rel[:, 1] * t[0]) <= tol_mm)[0]
            s = rel[idx] @ t
            k = np.argsort(s)
            for run in _runs(idx[k], s[k], pitch_tol):
                if len(run) >= 3:
                    found.setdefault(frozenset(run), run)
    return [r for key, r in found.items() if not any(key < other for other in found)]


def _finding(spec: dict[str, Any], kind: str, ids: list[str], loc: tuple[float, float],
             measure: dict[str, Any], rule: str) -> dict[str, Any]:
    r = spec["fastener_rules"]["findings"][kind]
    return {"id": "", "type": kind, "fasteners": ids,
            "location_mm": {"x": round(loc[0], 1), "y": round(loc[1], 1)},
            "measure": {k: round(v, 3) if isinstance(v, float) else v for k, v in measure.items()},
            "rule": rule, "reference": list(r.get("reference", [])), "severity": r["severity"],
            "disposition": r["disposition"], "action": r["action"]}


def fastener_findings(fasteners: Sequence[Fastener | dict[str, Any]], plate_size_mm: Sequence[float],
                      spec: dict[str, Any], D: float | None = None) -> list[dict[str, Any]]:
    """Edge distance, pitch, missing fastener, elongated and lone open holes (ids F1..).

    Edge distance: hole centre to the nearest plate edge < edge_distance_min_D * D.
    Pitch: centre to centre < pitch_min_D * D along a row (or for fasteners not in rows), or
    < transverse_min_D * D between two parallel rows.
    Missing fastener: an open round hole in a row (>= 3 collinear at a consistent pitch) that
    holds at least one rivet. Elongated: an open hole with ovality >= ovality_max or major axis
    > D + oversize_mm. Lone open hole: any other open hole (possibly a tooling hole).
    """
    fs = [as_fastener(f) for f in fasteners]
    fr = spec["fastener_rules"]
    D = float(D or spec["part"]["fastener_d_mm"])
    W, H = (float(x) for x in plate_size_mm)
    rows = find_rows(fs, fr.get("row_tol_mm", 1.0), fr.get("pitch_tol", 0.25))
    rows_of = [{k for k, r in enumerate(rows) if i in r} for i in range(len(fs))]
    out: list[dict[str, Any]] = []

    k_edge = fr["edge_distance_min_D"]
    for f in fs:
        e = min(f.u, f.v, W - f.u, H - f.v)
        if e < k_edge * D:
            out.append(_finding(spec, "edge_distance", [f.id], (f.u, f.v),
                                {"edge_mm": e, "edge_D": e / D, "D_mm": D},
                                f"edge distance {e:.1f} mm = {e / D:.2f}D < {k_edge}D = {k_edge * D:.1f} mm "
                                f"(hole centre to plate edge)"))

    k_pitch, k_trans = fr["pitch_min_D"], fr.get("transverse_min_D", fr["pitch_min_D"])
    for i, a in enumerate(fs):
        for j in range(i + 1, len(fs)):
            b = fs[j]
            p = math.hypot(a.u - b.u, a.v - b.v)
            across = bool(rows_of[i] and rows_of[j] and not rows_of[i] & rows_of[j])
            k = k_trans if across else k_pitch
            if p < k * D:
                where = "between rows" if across else "along a row" if rows_of[i] & rows_of[j] else "nearest neighbour"
                out.append(_finding(spec, "pitch", [a.id, b.id], ((a.u + b.u) / 2, (a.v + b.v) / 2),
                                    {"pitch_mm": p, "pitch_D": p / D, "D_mm": D},
                                    f"pitch {p:.1f} mm = {p / D:.2f}D < {k}D = {k * D:.1f} mm ({where})"))

    omax, over = fr["ovality_max"], fr["oversize_mm"]
    for i, f in enumerate(fs):
        if f.cls != "open_hole":
            continue
        if f.ovality >= omax - 1e-9 or f.major_mm > D + over:
            why = (f"ovality {f.ovality:.2f} >= {omax:.2f}" if f.ovality >= omax - 1e-9
                   else f"major axis {f.major_mm:.2f} mm > D + {over} = {D + over:.2f} mm")
            out.append(_finding(spec, "elongated_hole", [f.id], (f.u, f.v),
                                {"major_mm": f.major_mm, "minor_mm": f.d_mm, "ovality": f.ovality, "D_mm": D},
                                f"open hole {f.major_mm:.2f} x {f.d_mm:.2f} mm: {why}"))
            continue
        riveted = [rows[k] for k in sorted(rows_of[i]) if any(fs[m].cls == "rivet" for m in rows[k] if m != i)]
        if riveted:
            row = max(riveted, key=len)
            pos = [(fs[m].u, fs[m].v) for m in row]
            pitch = float(np.median([math.dist(p, q) for p, q in zip(pos, pos[1:])]))
            out.append(_finding(spec, "missing_fastener", [f.id], (f.u, f.v),
                                {"row": [fs[m].id for m in row], "row_pitch_mm": pitch, "D_mm": D},
                                f"open hole in a riveted row of {len(row)} at {pitch:.1f} mm pitch: fastener missing"))
        else:
            out.append(_finding(spec, "lone_open_hole", [f.id], (f.u, f.v),
                                {"d_mm": f.d_mm, "D_mm": D},
                                f"open hole {f.d_mm:.2f} mm not in a riveted row: possible tooling hole"))
    for k, f in enumerate(out, 1):
        f["id"] = f"F{k}"
    return out


# ----------------------------------------------------------------------------- dents

def _unit(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


def _centre(d: dict[str, Any]) -> np.ndarray:
    return np.array([d["location_mm"]["x"], d["location_mm"]["y"]], float)


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    L2 = float(ab @ ab)
    s = 0.0 if L2 == 0 else min(1.0, max(0.0, float((p - a) @ ab) / L2))
    return float(np.hypot(*(p - a - s * ab)))


def damage_gap(dent: dict[str, Any], other: dict[str, Any]) -> float:
    """Edge-to-edge gap between a dent circle and another defect taken as a segment of its
    length along orientation_deg with half its width as radius (a crack or scratch)."""
    c, r = _centre(dent), dent["size"]["diameter_mm"] / 2
    sz = other["size"]
    L, w = float(sz.get("length_mm") or 0.0), float(sz.get("width_mm") or 0.0)
    if L <= 0:
        L = w = float(sz.get("diameter_mm") or 0.0)
    half = max(L - w, 0.0) / 2
    t = _unit(float(other.get("orientation_deg", 0.0)))
    o = _centre(other)
    return max(0.0, _point_segment(c, o - half * t, o + half * t) - r - w / 2)


def fastener_gap(dent: dict[str, Any], f: Fastener) -> float:
    """Edge-to-edge gap between a dent circle and a fastener's visible outline."""
    c = _centre(dent)
    dx, dy = c[0] - f.u, c[1] - f.v
    dist = math.hypot(dx, dy)
    ang = math.degrees(math.atan2(dy, dx)) if dist > 0 else 0.0
    return max(0.0, dist - dent["size"]["diameter_mm"] / 2 - f.radius_toward(ang))


def dent_class(dent: dict[str, Any], fasteners: Sequence[Fastener | dict[str, Any]],
               cracks: Sequence[dict[str, Any]], t: float, spec: dict[str, Any], D: float | None = None,
               plate_size_mm: Sequence[float] | None = None) -> dict[str, Any]:
    """Piper SL 1095A-style class: negligible | minor | major, or edge_nick at a plate edge.

    negligible: diameter <= max_diameter_mm, depth <= t / max_depth_t_div (only when depth is
    known), >= min_fastener_gap_D * D from the nearest visible fastener edge and no crack or
    abrasion (``cracks``) within clear_x_diameter * diameter, edge to edge. minor: the same but
    closer to a fastener. major: beyond. A dent's ``near`` (twin records: exact outline
    distances fastener_edge_mm / fastener / plate_edge_mm / damage_edge_mm) overrides the
    circle approximations.
    """
    lim = spec["defects"]["dent"]["limits"]
    D = float(D or spec["part"]["fastener_d_mm"])
    fs = [as_fastener(f) for f in fasteners]
    near = dent.get("near") or {}
    dd = float(dent["size"]["diameter_mm"])
    x, y = dent["location_mm"]["x"], dent["location_mm"]["y"]
    depth = dent.get("depth_mm")

    if "fastener_edge_mm" in near:
        gap, nearest = near["fastener_edge_mm"], near.get("fastener")
    elif fs:
        gap, nearest = min((fastener_gap(dent, f), f.id) for f in fs)
    else:
        gap, nearest = None, None
    if "damage_edge_mm" in near:
        dmg = near["damage_edge_mm"]
    else:
        dmg = min((damage_gap(dent, c) for c in cracks if c is not dent), default=None)
    if "plate_edge_mm" in near:
        edge = near["plate_edge_mm"]
    elif plate_size_mm is not None:
        W, H = plate_size_mm
        edge = max(0.0, min(x, y, W - x, H - y) - dd / 2)
    else:
        edge = None

    max_depth = t / lim["max_depth_t_div"]
    k_gap, k_clear = lim["min_fastener_gap_D"], lim["clear_x_diameter"]
    parts = [f"dent {dd:.2f} mm across"]
    if edge is not None and edge <= lim["edge_nick_mm"]:
        cls = "edge_nick"
        parts.append(f"{edge:.2f} mm from the plate edge (edge nick)")
    else:
        major = []
        if dd > lim["max_diameter_mm"]:
            major.append(f"diameter > {lim['max_diameter_mm']} mm")
        if depth is not None and abs(depth) > max_depth:
            major.append(f"depth {abs(depth):.2f} mm > t/{lim['max_depth_t_div']} = {max_depth:.2f} mm")
        if dmg is not None and dmg < k_clear * dd:
            major.append(f"crack or abrasion {dmg:.1f} mm away < {k_clear}x diameter")
        if gap is not None:
            parts.append(f"{gap:.1f} mm = {gap / D:.2f}D from fastener {nearest} edge "
                         f"({'<' if gap < k_gap * D else '>='} {k_gap}D)")
        if major:
            cls = "major"
            parts += major
        else:
            cls = "minor" if gap is not None and gap < k_gap * D else "negligible"
        if depth is None:
            parts.append("depth not measured")
    return {"class": cls, "rule": f"{', '.join(parts)}: {cls} (Piper SL 1095A-style)",
            "fastener_gap_mm": None if gap is None else round(float(gap), 2),
            "fastener_gap_D": None if gap is None else round(float(gap) / D, 2),
            "nearest_fastener": nearest, "damage_gap_mm": None if dmg is None else round(float(dmg), 2),
            "depth_checked": depth is not None}


# ----------------------------------------------------------------------------- judging

def judge_aero_defect(defect: dict[str, Any], fasteners: Sequence[Fastener], damage: Sequence[dict[str, Any]],
                      spec: dict[str, Any], plate_size_mm: Sequence[float] | None = None,
                      D: float | None = None, t: float | None = None) -> dict[str, Any]:
    """Add zone, severity, disposition, rule, action and reference to one surface defect."""
    rules = spec["defects"]
    kind, conf = defect["type"], float(defect["confidence"])
    min_conf = spec.get("min_confidence", 0.6)
    out = dict(defect, zone=zone_for(spec, defect["location_mm"]["x"], defect["location_mm"]["y"]))
    t = float(t or spec["part"].get("thickness_mm", 3.0))
    if kind == "unknown" or kind not in rules or kind in FASTENER_CLASSES or conf < min_conf:
        rule = rules["unknown"]
        why = (f"confidence {conf:.2f} below {min_conf}" if kind in rules and kind != "unknown"
               else f"type {kind} not recognised")
        out.update(severity="unclassified", disposition="HOLD_FOR_QA", rule=f"{why} -> hold for QA",
                   action=rule["action"], reference=list(rule.get("reference", [])))
        return out
    rule = rules[kind]
    size = defect["size"]
    if kind == "dent":
        info = dent_class(defect, fasteners, damage, t, spec, D, plate_size_mm)
        c = rule["classes"][info["class"]]
        if c["disposition"] == "PASS":
            action = rule["note"]
        else:
            action = rule["edge_nick_action"] if info["class"] == "edge_nick" else rule["action"]
        out.update(severity=c["severity"], disposition=c["disposition"], rule=info["rule"], action=action,
                   dent_class=info["class"],
                   fastener_gap_mm=info["fastener_gap_mm"], nearest_fastener=info["nearest_fastener"])
    elif kind == "scratch":
        lim = rule["limits"]
        L, w = float(size["length_mm"]), float(size["width_mm"])
        ok = L <= lim["length_mm"] and w <= lim["width_mm"]
        c = rule["within" if ok else "beyond"]
        sign = "within" if ok else "beyond"
        out.update(severity=c["severity"], disposition=c["disposition"],
                   rule=f"scratch {L:.1f} mm long, {w:.2f} mm wide: {sign} the demo blend limits "
                        f"(length <= {lim['length_mm']} mm, width <= {lim['width_mm']} mm)",
                   action=rule["action"] if ok else rule["beyond_action"])
    else:
        m = rule.get("measure", "length_mm")
        val = float(size.get(m) or 0.0)
        name, unit = m.rsplit("_", 1)
        label = "crack-like indication" if kind == "crack" else kind
        out.update(severity=rule["severity"], disposition=rule["disposition"],
                   rule=f"{label}, {name} {val:.1f} {unit}: any {label} on a production part",
                   action=rule["action"])
    out["reference"] = list(rule.get("reference", []))
    return out


def judge_aero(defects: Sequence[dict[str, Any]], fasteners: Sequence[Fastener | dict[str, Any]],
               spec: dict[str, Any], plate_size_mm: Sequence[float], D: float | None = None,
               t: float | None = None) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """(verdict, judged surface defects, fastener findings); the worst disposition wins.

    Defects typed rivet/open_hole are ignored here: pass them as fasteners.
    """
    fs = [as_fastener(f) for f in fasteners]
    findings = fastener_findings(fs, plate_size_mm, spec, D)
    surface = [d for d in defects if d["type"] not in FASTENER_CLASSES]
    min_conf = spec.get("min_confidence", 0.6)
    clear_of = spec["defects"]["dent"].get("clear_of", ["crack"])
    damage = [d for d in surface if d["type"] in clear_of and d["confidence"] >= min_conf]
    judged = [judge_aero_defect(d, fs, damage, spec, plate_size_mm, D, t) for d in surface]
    verdict = worst([d["disposition"] for d in judged] + [f["disposition"] for f in findings])
    return verdict, judged, findings


def references_used(items: Iterable[dict[str, Any]], spec: dict[str, Any],
                    always: Sequence[str] = ()) -> list[dict[str, str]]:
    """[{id, text}] for every reference id cited by items (plus always), in spec order."""
    used = set(always)
    for it in items:
        used.update(it.get("reference") or [])
    refs = spec.get("references", {})
    return [{"id": k, "text": v} for k, v in refs.items() if k in used]
