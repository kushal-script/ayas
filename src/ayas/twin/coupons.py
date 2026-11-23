"""Physical test coupons: load, check, tabulate and export the single source of truth.

configs/coupons.yaml has one entry per imaged face (P1A = plate P1, face A) in the
scene-plan structure of docs/twin/DESIGN.md section 3 (plate, sheets, holes, rivets,
defects, distractors) plus name, role, make (fabrication steps) and expect (the rule
findings the coupon is designed to trigger).

This module fills in the YAML defaults, checks every coupon (inside the plate, unique
ids, rivets in filled holes, DESIGN 3.1 parameter ranges, makeable spacing, and that the
designed findings really hold geometrically), prints fastener tables and writes the
ground-truth CSV template of DESIGN section 6.

Plate frame: mm from the datum corner A, u right, v down. Angles in degrees from +u
towards +v. The rule thresholds below are the public demo heuristics (FAA AC 43.13-1B
4-57c and Piper SL 1095A); the runtime rules live in aero.py and configs/spec_aero.yaml,
and the finding names match theirs (fastener finding types, dent_<class>).

Ground-truth CSV: one row per defect (eval_class = cls, or 'hidden' for a crack that runs
under an installed head and cannot be seen) plus one row per fastener (cls rivet for a
rivet or screw head, open_hole for an open hole), so real photos get the same six YOLO
classes as the twin.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

DEFAULT_PATH = "configs/coupons.yaml"
ROOT = Path(__file__).resolve().parents[3]

ROLES = ("train", "test", "calib")
TEST_PLATES = ("P5",)                 # the held-out plate: role test, never rendered for training
KEY_RE = re.compile(r"[A-Z]+\d+[A-Z]")  # face id = plate id + face letter (P2A); 'P5' alone would dodge TEST_PLATES
RIVET_KINDS = ("blind_rivet", "screw")
KINDS = {
    "dent": ("punch", "ball", "smooth", "nick"),
    "scratch": ("scriber", "gouge", "abrasion"),
    "crack": ("slit", "fatigue"),
    "corrosion": ("pitting", "galvanic", "fretting"),
}
DISTRACTOR_KINDS = ("fingerprint", "oil", "water_spot", "marker", "dust", "blend", "burr", "glove")
EDGES = ("top", "bottom", "left", "right")

# Required params and DESIGN 3.1 ranges (inclusive) per (cls, kind).
PARAMS = {
    ("dent", "punch"): ("u", "v", "d_mm", "depth_mm", "rim_h_mm"),
    ("dent", "ball"): ("u", "v", "d_mm", "depth_mm"),
    ("dent", "smooth"): ("u", "v", "d_mm", "depth_mm", "aspect", "angle_deg"),
    ("dent", "nick"): ("u", "v", "edge", "d_mm", "depth_mm"),
    ("scratch", "scriber"): ("ctrl_mm", "width_mm", "depth_mm", "burr"),
    ("scratch", "gouge"): ("ctrl_mm", "width_mm", "depth_mm", "burr"),
    ("scratch", "abrasion"): ("u", "v", "w_mm", "h_mm", "angle_deg", "n_lines", "depth_mm"),
    ("crack", "slit"): ("u0", "v0", "angle_deg", "length_mm", "width_mm"),
    ("crack", "fatigue"): ("pts_mm", "width_mm"),
    ("corrosion", "pitting"): ("u", "v", "r_mm", "n_pits", "pit_d_mm", "powder"),
    ("corrosion", "galvanic"): ("hole", "ring_w_mm", "powder"),
    ("corrosion", "fretting"): ("rivet", "angle_deg", "length_mm", "width_mm", "darkness"),
}
RANGES = {
    ("dent", "punch"): {"d_mm": (0.5, 2.5), "depth_mm": (0.05, 0.6), "rim_h_mm": (0.02, 0.05)},
    ("dent", "ball"): {"d_mm": (3.0, 12.0), "depth_mm": (0.1, 0.8)},
    ("dent", "smooth"): {"d_mm": (5.0, 40.0), "depth_mm": (0.1, 1.5), "aspect": (1.0, 2.0)},
    ("dent", "nick"): {"d_mm": (1.0, 3.0)},
    ("scratch", "scriber"): {"width_mm": (0.05, 0.5), "depth_mm": (0.005, 0.08), "burr": (0.0, 1.0)},
    ("scratch", "gouge"): {"width_mm": (0.5, 2.0), "depth_mm": (0.1, 0.5), "burr": (0.0, 1.0)},
    ("scratch", "abrasion"): {"w_mm": (5.0, 15.0), "h_mm": (5.0, 15.0), "n_lines": (20, 200)},
    ("crack", "slit"): {"length_mm": (1.0, 10.0), "width_mm": (0.25, 0.6)},
    ("crack", "fatigue"): {"width_mm": (0.01, 0.08)},
    ("corrosion", "pitting"): {"r_mm": (1.5, 8.0), "powder": (0.0, 1.0)},
    ("corrosion", "galvanic"): {"ring_w_mm": (1.0, 4.0), "powder": (0.0, 1.0)},
    ("corrosion", "fretting"): {"length_mm": (3.0, 15.0), "width_mm": (1.0, 4.0), "darkness": (0.3, 0.9)},
}
PIT_D_RANGE = (0.1, 1.0)
NON_SCALAR_PARAMS = ("edge", "hole", "rivet", "ctrl_mm", "pts_mm", "pit_d_mm")

# Demo rule thresholds (public sources only; D = fastener_d_mm). Same values as
# configs/spec_aero.yaml (fastener_rules, defects.dent.limits).
EDGE_MIN_D = 2.0              # AC 43.13-1B 4-57c: edge distance (centre to edge) >= 2D
PITCH_MIN_D = 3.0             # AC 43.13-1B 4-57c: spacing (centre to centre) >= 3D
OVAL_MIN = 1.10               # an open hole with major/minor >= this is elongated
OVERSIZE_MM = 0.3             # ... or with a major axis > D + this (spec_aero oversize_mm)
ROW_TOL_MM = 1.0              # an open round hole in line with a filled one is a missing fastener
DENT_FASTENER_MIN_D = 2.0     # Piper SL 1095A: >= 2x fastener diameter from the nearest fastener
DENT_DEPTH_FRAC = 1 / 1.666   # Piper SL 1095A: depth <= t / 1.666 (thicknesses not in its table)
DENT_MAX_D_MM = 50.8          # Piper SL 1095A: no more than 2.0 in across
DENT_CLEAR_X = 2.0            # Piper SL 1095A: no crack, abrasion or crease within 2x the dent diameter
EDGE_NICK_MM = 0.5            # spec_aero: a dent this close to a plate edge is an edge nick, not a skin dent
MIN_VISIBLE_CRACK_MM = 1.0    # crack length outside every fastener head needed for a visible indication
HIDDEN = "hidden"             # eval_class of a crack the camera cannot see (under a head): no label, no miss
MIN_HALF_WIDTH_MM = 0.05      # thinner CSV outlines are widened to 0.1 mm, as the twin's footprints

# Makeability.
MIN_GAP_MM = 3.0              # between unrelated features
NEAR_DENT_MIN_GAP_MM = 1.5    # a dent placed deliberately near a fastener head
HEAD_MIN_GAP_MM = 1.0         # between fastener heads
MIN_EDGE_MATERIAL_MM = 2.0    # between a hole edge and the plate edge
ATTACH_TOL_MM = 0.05          # a slit start on a hole edge or plate edge
SAMPLE_MM = 0.2               # outline sampling for gap checks

EXPECT_TYPES = {
    "edge_distance": "hole", "pitch": "holes", "missing_fastener": "hole", "elongated_hole": "hole",
    "lone_open_hole": "hole",
    "dent_negligible": "defect", "dent_minor": "defect", "dent_major": "defect", "dent_edge_nick": "defect",
    "crack_indication": "defect", "corrosion": "defect", "hold_unknown": "distractor",
}
HOLD_KINDS = ("burr",)        # distractors the rules must send to HOLD (unknown)

CSV_COLUMNS = ("defect_id", "plate", "face", "cls", "kind", "u_mm", "v_mm", "length_mm", "width_mm",
               "diameter_mm", "angle_deg", "poly_mm", "eval_class", "notes")


# ----------------------------------------------------------------------------- loading

def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists() and not p.is_absolute() and (ROOT / p).exists():
        return ROOT / p
    return p


def _expand(key: str, raw: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """One coupon with the YAML defaults filled in (scene-plan structure)."""
    d = top.get("defaults", {})
    c = copy.deepcopy(raw)
    c["id"] = key
    c["plate_id"], c["face"] = key[:-1], key[-1]
    c["fastener_d_mm"] = float(c.get("fastener_d_mm", top.get("fastener_d_mm", 4.0)))
    c["plate"] = {**d.get("plate", {}), **c.get("plate", {})}
    c["datum"] = {**d.get("datum", {}), **c.get("datum", {})}
    c["holes"] = [{**d.get("hole", {}), **h} for h in c.get("holes", [])]
    rivet_defaults = d.get("rivet", {})
    c["rivets"] = [{**rivet_defaults.get(r.get("kind", "blind_rivet"), {}), **r} for r in c.get("rivets", [])]
    for k in ("sheets", "defects", "distractors", "expect", "make"):
        c[k] = c.get(k) or []
    return c


def load_coupons(path: str | Path = DEFAULT_PATH) -> dict[str, dict[str, Any]]:
    """Coupons keyed by face id (P1A, ...), defaults merged in, in file order."""
    top = yaml.safe_load(_resolve(path).read_text())
    return {k: _expand(k, v, top) for k, v in top["coupons"].items()}


def for_replica(coupons: dict[str, dict[str, Any]], names: Iterable[str]) -> list[dict[str, Any]]:
    """Coupons for replica scenes: names are plate ids (P2 -> every face of P2) or face ids.

    Raises ValueError for a test coupon: P5 must never be rendered for training.
    """
    out = []
    for n in names:
        hits = [c for k, c in coupons.items() if k == n or c["plate_id"] == n]
        if not hits:
            raise ValueError(f"no coupon {n}")
        for c in hits:
            if c["role"] == "test":
                raise ValueError(f"{c['id']} is the held-out test coupon and must not be rendered")
            out.append(c)
    return out


# ----------------------------------------------------------------------------- geometry

def plate_size(c: dict[str, Any]) -> tuple[float, float]:
    w, h = c["plate"]["size_mm"]
    return float(w), float(h)


def holes_by_id(c: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {h["id"]: h for h in c["holes"]}


def rivet_in(c: dict[str, Any], hole_id: str) -> dict[str, Any] | None:
    return next((r for r in c["rivets"] if r["hole"] == hole_id), None)


def hole_axes(h: dict[str, Any]) -> tuple[float, float]:
    """(major, minor) half-axes of a hole in mm; the major axis lies along angle_deg."""
    d = float(h["d_mm"])
    return d * float(h.get("ovality", 1.0)) / 2, d / 2


def head_radius(c: dict[str, Any], hole_id: str) -> float | None:
    r = rivet_in(c, hole_id)
    return float(r["head_d_mm"]) / 2 if r else None


def fastener_radius(c: dict[str, Any], h: dict[str, Any]) -> float:
    """Visible fastener radius: the head for a filled hole, the major half-axis for an open one."""
    rh = head_radius(c, h["id"])
    return rh if rh is not None else hole_axes(h)[0]


def _dir(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


def ellipse(cu: float, cv: float, a: float, b: float, angle_deg: float = 0.0, n: int | None = None) -> np.ndarray:
    n = n or max(24, int(math.ceil(2 * math.pi * max(a, b) / SAMPLE_MM)))
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x, y = a * np.cos(t), b * np.sin(t)
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.stack([cu + ca * x - sa * y, cv + sa * x + ca * y], axis=1)


def rect(cu: float, cv: float, length: float, width: float, angle_deg: float) -> np.ndarray:
    """Corners of a length x width rectangle centred on (cu, cv), length along angle_deg."""
    d = _dir(angle_deg)
    n = np.array([-d[1], d[0]])
    c = np.array([cu, cv])
    hl, hw = length / 2, width / 2
    return np.array([c - hl * d - hw * n, c + hl * d - hw * n, c + hl * d + hw * n, c - hl * d + hw * n])


def bezier(ctrl, n: int = 200) -> np.ndarray:
    p = np.asarray(ctrl, float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3) * p[0] + 3 * ((1 - t) ** 2) * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]


def polyline_length(pts) -> float:
    p = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def densify(pts, closed: bool, step: float = SAMPLE_MM) -> np.ndarray:
    p = np.asarray(pts, float)
    if closed:
        p = np.vstack([p, p[:1]])
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * (np.arange(1, k + 1)[:, None] / k))
    q = np.vstack(out)
    return q[:-1] if closed else q


def slit_segment(params: dict[str, Any]) -> np.ndarray:
    p0 = np.array([float(params["u0"]), float(params["v0"])])
    return np.array([p0, p0 + float(params["length_mm"]) * _dir(float(params["angle_deg"]))])


def crack_path(d: dict[str, Any]) -> np.ndarray:
    """Centreline of a crack defect (slit segment or fatigue polyline)."""
    return slit_segment(d["params"]) if d["kind"] == "slit" else np.asarray(d["params"]["pts_mm"], float)


def fretting_rect(c: dict[str, Any], params: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """(cu, cv, length, width, angle) of a fretting streak running out from the rivet head edge."""
    rv = next(r for r in c["rivets"] if r["id"] == params["rivet"])
    h = holes_by_id(c)[rv["hole"]]
    rh = float(rv["head_d_mm"]) / 2
    L = float(params["length_mm"])
    ctr = np.array([h["u"], h["v"]], float) + (rh + L / 2) * _dir(float(params["angle_deg"]))
    return float(ctr[0]), float(ctr[1]), L, float(params["width_mm"]), float(params["angle_deg"])


def galvanic_radius(c: dict[str, Any], params: dict[str, Any]) -> float:
    """Outer radius of a galvanic ring: fastener head (or hole) edge + ring width."""
    h = holes_by_id(c)[params["hole"]]
    return fastener_radius(c, h) + float(params["ring_w_mm"])


@dataclass
class Shape:
    """Feature outline for gap checks: samples along an outline (closed) or a centreline."""

    id: str
    pts: np.ndarray
    hw: float = 0.0          # half width around the samples
    closed: bool = False     # the polygon's interior belongs to the shape


def _circle(id_: str, u: float, v: float, r: float) -> Shape:
    return Shape(id_, ellipse(u, v, r, r), 0.0, True)


def hole_shapes(c: dict[str, Any]) -> list[Shape]:
    """Each fastener as seen: the head circle if filled, the hole ellipse if open."""
    out = []
    for h in c["holes"]:
        rh = head_radius(c, h["id"])
        if rh is not None:
            out.append(_circle(h["id"], h["u"], h["v"], rh))
        else:
            a, b = hole_axes(h)
            out.append(Shape(h["id"], ellipse(h["u"], h["v"], a, b, h.get("angle_deg", 0.0)), 0.0, True))
    return out


def defect_shape(c: dict[str, Any], d: dict[str, Any]) -> Shape:
    p, kind = d["params"], d["kind"]
    if d["cls"] == "dent":
        if kind == "smooth":
            a = float(p["d_mm"]) / 2
            return Shape(d["id"], ellipse(p["u"], p["v"], a, a / float(p["aspect"]), p["angle_deg"]), 0.0, True)
        return _circle(d["id"], p["u"], p["v"], float(p["d_mm"]) / 2)
    if kind in ("scriber", "gouge"):
        return Shape(d["id"], densify(bezier(p["ctrl_mm"]), False), float(p["width_mm"]) / 2)
    if kind == "abrasion":
        return Shape(d["id"], densify(rect(p["u"], p["v"], p["w_mm"], p["h_mm"], p["angle_deg"]), True), 0.0, True)
    if d["cls"] == "crack":
        return Shape(d["id"], densify(crack_path(d), False), float(p["width_mm"]) / 2)
    if kind == "pitting":
        return _circle(d["id"], p["u"], p["v"], float(p["r_mm"]))
    if kind == "galvanic":
        h = holes_by_id(c)[p["hole"]]
        return _circle(d["id"], h["u"], h["v"], galvanic_radius(c, p))
    if kind == "fretting":
        return Shape(d["id"], densify(rect(*fretting_rect(c, p)), True), 0.0, True)
    raise ValueError(f"unknown defect {d['cls']}.{kind}")


def distractor_shape(c: dict[str, Any], x: dict[str, Any]) -> Shape | None:
    p = x.get("params", {})
    if x["kind"] == "burr" and p.get("hole") in holes_by_id(c):
        h = holes_by_id(c)[p["hole"]]
        a, b = hole_axes(h)
        w = float(p.get("w_mm", 0.4))
        return Shape(x["id"], ellipse(h["u"], h["v"], a + w, b + w, h.get("angle_deg", 0.0)), 0.0, True)
    if "u" in p and "v" in p:
        r = float(p.get("d_mm", 2 * float(p.get("r_mm", 1.0)))) / 2
        return _circle(x["id"], p["u"], p["v"], r)
    return None


def datum_shape(c: dict[str, Any]) -> Shape:
    dm = c["datum"]
    s = float(dm.get("size_mm", 3.0))
    return Shape("datum", densify(rect(dm["u"], dm["v"], s, s, 0.0), True), 0.0, True)


def _inside(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test for many points."""
    x, y = pts[:, :1], pts[:, 1:2]
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    cross = (y0 > y) != (y1 > y)
    dy = np.where(y1 != y0, y1 - y0, 1e-12)
    xi = x0 + (y - y0) * (x1 - x0) / dy
    return ((cross & (x < xi)).sum(axis=1) % 2) == 1


def gap(a: Shape, b: Shape) -> float:
    """Edge-to-edge distance in mm; 0 when the shapes touch or overlap."""
    if (a.closed and _inside(b.pts, a.pts).any()) or (b.closed and _inside(a.pts, b.pts).any()):
        return 0.0
    dmin = np.inf
    for i in range(0, len(a.pts), 512):
        diff = a.pts[i : i + 512, None, :] - b.pts[None, :, :]
        dmin = min(dmin, float(np.sqrt((diff ** 2).sum(axis=2)).min()))
    return max(0.0, dmin - a.hw - b.hw)


def visible_crack_mm(c: dict[str, Any], d: dict[str, Any], step: float = 0.01) -> float:
    """Crack length outside every fastener head (an installed head hides what is under it)."""
    path = crack_path(d)
    heads = [(np.array([h["u"], h["v"]], float), head_radius(c, h["id"])) for h in c["holes"]]
    heads = [(ctr, r) for ctr, r in heads if r is not None]
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        L = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(L / step)))
        mids = a + (b - a) * ((np.arange(n) + 0.5) / n)[:, None]
        hidden = np.zeros(n, bool)
        for ctr, r in heads:
            hidden |= np.linalg.norm(mids - ctr, axis=1) < r
        total += L * float((~hidden).sum()) / n
    return total


# ----------------------------------------------------------------------------- rules

def fastener_table(c: dict[str, Any]) -> list[dict[str, Any]]:
    """Per hole: position, size, nearest-edge distance and nearest-neighbour pitch (mm and D)."""
    W, H = plate_size(c)
    D = c["fastener_d_mm"]
    rows = []
    for h in c["holes"]:
        u, v = float(h["u"]), float(h["v"])
        edge = min(u, v, W - u, H - v)
        others = [(math.hypot(o["u"] - u, o["v"] - v), o["id"]) for o in c["holes"] if o["id"] != h["id"]]
        pitch, nb = min(others) if others else (None, None)
        r = rivet_in(c, h["id"])
        rows.append({
            "hole": h["id"], "u": u, "v": v, "d_mm": float(h["d_mm"]), "ovality": float(h.get("ovality", 1.0)),
            "fastener": r["kind"] if r else "open",
            "edge_mm": round(edge, 3), "edge_D": round(edge / D, 3),
            "pitch_mm": None if pitch is None else round(pitch, 3),
            "pitch_D": None if pitch is None else round(pitch / D, 3), "neighbour": nb,
        })
    return rows


def dent_class(c: dict[str, Any], d: dict[str, Any]) -> str | None:
    """Piper SL 1095A-style class (negligible | minor | major), or edge_nick for a nick or any
    dent within EDGE_NICK_MM of a plate edge (as aero.dent_class); None for non-dents."""
    if d["cls"] != "dent":
        return None
    p = d["params"]
    t = float(c["plate"]["thickness_mm"])
    dd = float(p["d_mm"])
    W, H = plate_size(c)
    u, v = float(p["u"]), float(p["v"])
    if d["kind"] == "nick" or min(u, v, W - u, H - v) - dd / 2 <= EDGE_NICK_MM:
        return "edge_nick"
    if float(p["depth_mm"]) > t * DENT_DEPTH_FRAC or dd > DENT_MAX_D_MM:
        return "major"
    me = defect_shape(c, d)
    for o in c["defects"]:
        if o["id"] != d["id"] and o["cls"] in ("crack", "scratch") and gap(me, defect_shape(c, o)) < DENT_CLEAR_X * dd:
            return "major"
    gaps = [gap(me, s) for s in hole_shapes(c)]
    if gaps and min(gaps) < DENT_FASTENER_MIN_D * c["fastener_d_mm"]:
        return "minor"
    return "negligible"


def findings(c: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule findings that the planned geometry triggers (design-time check of `expect`)."""
    out: list[dict[str, Any]] = []
    D = c["fastener_d_mm"]
    for row in fastener_table(c):
        if row["edge_mm"] < EDGE_MIN_D * D:
            out.append({"type": "edge_distance", "hole": row["hole"]})
    hs = c["holes"]
    for i, a in enumerate(hs):
        for b in hs[i + 1 :]:
            if math.hypot(a["u"] - b["u"], a["v"] - b["v"]) < PITCH_MIN_D * D:
                out.append({"type": "pitch", "holes": sorted([a["id"], b["id"]])})
    for h in hs:
        if rivet_in(c, h["id"]):
            continue
        if float(h.get("ovality", 1.0)) >= OVAL_MIN or 2 * hole_axes(h)[0] > D + OVERSIZE_MM:
            out.append({"type": "elongated_hole", "hole": h["id"]})
        elif any(rivet_in(c, o["id"]) and (abs(o["u"] - h["u"]) <= ROW_TOL_MM or abs(o["v"] - h["v"]) <= ROW_TOL_MM)
                 for o in hs if o["id"] != h["id"]):
            out.append({"type": "missing_fastener", "hole": h["id"]})
        else:                            # aero: an open hole outside a riveted row (maybe a tooling hole)
            out.append({"type": "lone_open_hole", "hole": h["id"]})
    for d in c["defects"]:
        cl = dent_class(c, d)
        if cl:
            out.append({"type": f"dent_{cl}", "defect": d["id"]})
        elif d["cls"] == "crack" and visible_crack_mm(c, d) >= MIN_VISIBLE_CRACK_MM - 1e-9:
            out.append({"type": "crack_indication", "defect": d["id"]})
        elif d["cls"] == "corrosion":
            out.append({"type": "corrosion", "defect": d["id"]})
    for x in c["distractors"]:
        if x["kind"] in HOLD_KINDS:
            out.append({"type": "hold_unknown", "distractor": x["id"]})
    return out


def finding_key(f: dict[str, Any]) -> tuple:
    if "holes" in f:
        return (f["type"], "holes", tuple(sorted(f["holes"])))
    for k in ("hole", "defect", "distractor"):
        if k in f:
            return (f["type"], k, f[k])
    return (f.get("type"),)


# ----------------------------------------------------------------------------- validation

def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_structure(c: dict[str, Any]) -> list[str]:
    """Missing ids/keys and non-numeric values, reported instead of crashing the checks below."""
    errs = []
    size = c["plate"].get("size_mm") if isinstance(c["plate"], dict) else None
    if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(_num(x) for x in size)):
        errs.append("plate.size_mm must be [W, H] in mm")
    if not isinstance(c["plate"], dict) or not _num(c["plate"].get("thickness_mm")):
        errs.append("plate.thickness_mm must be a number")
    need = {"holes": ("id", "u", "v", "d_mm"), "rivets": ("id", "hole", "head_d_mm"),
            "defects": ("id", "cls", "kind", "params"), "distractors": ("id", "kind")}
    for k, keys in need.items():
        for i, x in enumerate(c[k]):
            if not isinstance(x, dict):
                errs.append(f"{k}[{i}] must be a mapping")
                continue
            if missing := [q for q in keys if q not in x]:
                errs.append(f"{k}[{i}] ({x.get('id', '?')}): missing {missing}")
            if k == "holes":
                errs += [f"{x.get('id')}: {q}={x[q]!r} must be a number"
                         for q in ("u", "v", "d_mm", "ovality", "angle_deg") if q in x and not _num(x[q])]
            if k in ("defects", "distractors") and not isinstance(x.get("params", {}), dict):
                errs.append(f"{x.get('id')}: params must be a mapping")
    if not all(isinstance(f, dict) and "type" in f for f in c["expect"]):
        errs.append("expect entries must be mappings with a type")
    return errs


def _check_params(d: dict[str, Any]) -> list[str]:
    key = (d.get("cls"), d.get("kind"))
    p = d.get("params") or {}
    errs = [f"{d['id']}: missing param {k}" for k in PARAMS[key] if k not in p]
    if errs:
        return errs
    errs = [f"{d['id']}: {k}={p[k]!r} must be a number" for k in PARAMS[key]
            if k not in NON_SCALAR_PARAMS and not _num(p[k])]
    for k in set(PARAMS[key]) & {"ctrl_mm", "pts_mm", "pit_d_mm"}:
        try:
            np.asarray(p[k], float)
        except (TypeError, ValueError):
            errs.append(f"{d['id']}: {k} must be numbers")
    if errs:
        return errs
    for k, (lo, hi) in RANGES.get(key, {}).items():
        if not _num(p[k]) or not lo <= p[k] <= hi:
            errs.append(f"{d['id']}: {k}={p[k]} outside DESIGN 3.1 range [{lo}, {hi}]")
    if key == ("corrosion", "pitting"):
        lo, hi = p["pit_d_mm"]
        if not PIT_D_RANGE[0] <= lo <= hi <= PIT_D_RANGE[1]:
            errs.append(f"{d['id']}: pit_d_mm {p['pit_d_mm']} outside [{PIT_D_RANGE[0]}, {PIT_D_RANGE[1]}]")
    if key in (("scratch", "scriber"), ("scratch", "gouge")) and np.asarray(p["ctrl_mm"], float).shape != (4, 2):
        errs.append(f"{d['id']}: ctrl_mm must be 4 (u, v) points")
    if key == ("crack", "fatigue") and len(p["pts_mm"]) < 2:
        errs.append(f"{d['id']}: pts_mm needs at least 2 points")
    if key == ("dent", "nick") and p["edge"] not in EDGES:
        errs.append(f"{d['id']}: edge must be one of {EDGES}")
    return errs


def _attachments(c: dict[str, Any]) -> dict[str, set[str]]:
    """Feature ids each defect/distractor is deliberately attached to (exempt from spacing)."""
    hole_of_rivet = {r["id"]: r["hole"] for r in c["rivets"]}
    hb = holes_by_id(c)
    att: dict[str, set[str]] = {}
    for x in c["defects"] + c["distractors"]:
        p = x.get("params", {})
        s: set[str] = set()
        if p.get("hole") in hb:
            s.add(p["hole"])
        if p.get("rivet") in hole_of_rivet:
            s.add(hole_of_rivet[p["rivet"]])
        if x.get("kind") == "slit" and "u0" in p:
            for h in c["holes"]:
                a, b = hole_axes(h)
                if a == b and abs(math.hypot(p["u0"] - h["u"], p["v0"] - h["v"]) - a) <= ATTACH_TOL_MM:
                    s.add(h["id"])
        att[x["id"]] = s
    for x in c["defects"]:
        if x.get("grows_from"):
            att[x["id"]] |= {x["grows_from"]} | att.get(x["grows_from"], set())
    return att


def _check_inside(c: dict[str, Any]) -> list[str]:
    W, H = plate_size(c)
    tol = 1e-6
    errs = []

    def out_of_plate(pts) -> bool:
        p = np.asarray(pts, float)
        return bool(((p[:, 0] < -tol) | (p[:, 0] > W + tol) | (p[:, 1] < -tol) | (p[:, 1] > H + tol)).any())

    for h in c["holes"]:
        a, _ = hole_axes(h)
        edge = min(h["u"], h["v"], W - h["u"], H - h["v"])
        if edge - a < MIN_EDGE_MATERIAL_MM - tol:
            errs.append(f"{h['id']}: only {edge - a:.2f} mm of material to the plate edge (< {MIN_EDGE_MATERIAL_MM})")
        rh = head_radius(c, h["id"])
        if rh is not None and edge < rh - tol:
            errs.append(f"{h['id']}: fastener head overhangs the plate edge")
    for d in c["defects"]:
        p = d["params"]
        if d["kind"] == "nick":
            on = {"left": abs(p["u"]) <= tol, "right": abs(p["u"] - W) <= tol,
                  "top": abs(p["v"]) <= tol, "bottom": abs(p["v"] - H) <= tol}[p["edge"]]
            along = 0 <= p["u"] <= W and 0 <= p["v"] <= H
            if not (on and along):
                errs.append(f"{d['id']}: nick centre must lie on the {p['edge']} edge")
            continue
        if d["kind"] == "slit":
            s = slit_segment(p)
            on_edge = min(s[0][0], s[0][1], W - s[0][0], H - s[0][1]) <= ATTACH_TOL_MM
            on_hole = bool(_attachments(c)[d["id"]])
            if not (on_edge or on_hole):
                errs.append(f"{d['id']}: slit must start on a round hole edge or a plate edge")
            if out_of_plate(s):
                errs.append(f"{d['id']}: slit leaves the plate")
            continue
        if out_of_plate(defect_shape(c, d).pts):
            errs.append(f"{d['id']}: outside the plate")
    for x in c["distractors"]:
        s = distractor_shape(c, x)
        if s is not None and out_of_plate(s.pts):
            errs.append(f"{x['id']}: outside the plate")
    return errs


def _check_spacing(c: dict[str, Any]) -> list[str]:
    errs = []
    hs = c["holes"]
    for i, a in enumerate(hs):
        for b in hs[i + 1 :]:
            ctr = math.hypot(a["u"] - b["u"], a["v"] - b["v"])
            if ctr - hole_axes(a)[0] - hole_axes(b)[0] < MIN_GAP_MM:
                errs.append(f"{a['id']}-{b['id']}: hole edges closer than {MIN_GAP_MM} mm")
            ra, rb = head_radius(c, a["id"]), head_radius(c, b["id"])
            if ra is not None and rb is not None and ctr - ra - rb < HEAD_MIN_GAP_MM:
                errs.append(f"{a['id']}-{b['id']}: fastener heads closer than {HEAD_MIN_GAP_MM} mm")
    att = _attachments(c)
    fasteners = hole_shapes(c)
    feats = [(d, defect_shape(c, d)) for d in c["defects"]]
    feats += [(x, s) for x in c["distractors"] if (s := distractor_shape(c, x)) is not None]
    datum = datum_shape(c)
    for s in fasteners:
        if gap(s, datum) < MIN_GAP_MM:
            errs.append(f"{s.id}: closer than {MIN_GAP_MM} mm to the datum mark")
    for i, (fa, sa) in enumerate(feats):
        mine = att.get(fa["id"], set())
        if gap(sa, datum) < MIN_GAP_MM:
            errs.append(f"{fa['id']}: closer than {MIN_GAP_MM} mm to the datum mark")
        for s in fasteners:
            if s.id in mine:
                continue
            lim = NEAR_DENT_MIN_GAP_MM if fa.get("cls") == "dent" else MIN_GAP_MM
            if (g := gap(sa, s)) < lim:
                errs.append(f"{fa['id']}: {g:.2f} mm from fastener {s.id} (< {lim})")
        for fb, sb in feats[i + 1 :]:
            if fb["id"] in mine or fa["id"] in att.get(fb["id"], set()):
                continue
            if (g := gap(sa, sb)) < MIN_GAP_MM:
                errs.append(f"{fa['id']}-{fb['id']}: {g:.2f} mm apart (< {MIN_GAP_MM})")
    return errs


def validate(c: dict[str, Any]) -> list[str]:
    """Problems with one coupon; an empty list means it is consistent and makeable."""
    errs: list[str] = []
    for k in ("name", "role", "plate", "holes", "rivets", "defects", "distractors", "expect", "make"):
        if k not in c:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    if not KEY_RE.fullmatch(str(c.get("id"))) or f"{c.get('plate_id')}{c.get('face')}" != c.get("id"):
        return [f"coupon key {c.get('id')!r} must be a plate id plus a face letter, e.g. P2A"]
    errs += _check_structure(c)
    if errs:
        return errs
    if c["role"] not in ROLES:
        errs.append(f"role {c['role']} not in {ROLES}")
    if (c["plate_id"] in TEST_PLATES) != (c["role"] == "test"):
        errs.append(f"{c['id']}: role must be 'test' exactly for the held-out plates {TEST_PLATES}")
    if not c["make"] or not all(isinstance(s, str) and s for s in c["make"]):
        errs.append("make must be a non-empty list of steps")
    if any("footprint" in d or "near" in d for d in c["defects"]):
        errs.append("footprint/near are computed by the sampler; do not add them")
    W, H = plate_size(c)
    if W <= 0 or H <= 0 or float(c["plate"]["thickness_mm"]) <= 0:
        errs.append("plate size and thickness must be positive")

    ids = [x["id"] for k in ("holes", "rivets", "defects", "distractors") for x in c[k]]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errs.append(f"duplicate ids {dup}")
    hb = holes_by_id(c)
    filled = {h["id"] for h in c["holes"] if h.get("filled")}
    used: list[str] = []
    for r in c["rivets"]:
        if r.get("kind") not in RIVET_KINDS:
            errs.append(f"{r['id']}: kind must be one of {RIVET_KINDS}")
        if r.get("hole") not in hb:
            errs.append(f"{r['id']}: hole {r.get('hole')} does not exist")
        elif r["hole"] not in filled:
            errs.append(f"{r['id']}: hole {r['hole']} is not filled")
        used.append(r.get("hole"))
    for h in sorted(filled):
        if used.count(h) != 1:
            errs.append(f"{h}: filled hole needs exactly one rivet (has {used.count(h)})")
    for h in c["holes"]:
        if float(h.get("ovality", 1.0)) < 1.0 or float(h["d_mm"]) <= 0:
            errs.append(f"{h['id']}: needs d_mm > 0 and ovality >= 1")
        if h.get("sheet") is not None and h["sheet"] not in {s["id"] for s in c["sheets"]}:
            errs.append(f"{h['id']}: unknown sheet {h['sheet']}")

    bad_defs = []
    for d in c["defects"]:
        if d.get("cls") not in KINDS or d.get("kind") not in KINDS[d.get("cls")]:
            errs.append(f"{d['id']}: unknown class/kind {d.get('cls')}.{d.get('kind')}")
            bad_defs.append(d["id"])
            continue
        e = _check_params(d)
        errs += e
        if e:
            bad_defs.append(d["id"])
        p = d["params"]
        if d["kind"] == "galvanic" and p.get("hole") not in hb:
            errs.append(f"{d['id']}: galvanic hole {p.get('hole')} does not exist")
            bad_defs.append(d["id"])
        elif d["kind"] == "galvanic" and not rivet_in(c, p["hole"]):
            errs.append(f"{d['id']}: galvanic corrosion needs a fastener (dissimilar metal) in {p['hole']}")
        if d["kind"] == "fretting" and p.get("rivet") not in {r["id"] for r in c["rivets"]}:
            errs.append(f"{d['id']}: fretting rivet {p.get('rivet')} does not exist")
            bad_defs.append(d["id"])
        if d.get("grows_from") and d["grows_from"] not in {x["id"] for x in c["defects"]}:
            errs.append(f"{d['id']}: grows_from {d['grows_from']} does not exist")
    for x in c["distractors"]:
        if x.get("kind") not in DISTRACTOR_KINDS:
            errs.append(f"{x.get('id')}: distractor kind {x.get('kind')} not in {DISTRACTOR_KINDS}")
        if x.get("kind") == "burr" and x.get("params", {}).get("hole") not in hb:
            errs.append(f"{x.get('id')}: burr hole does not exist")
    if bad_defs or dup or any("does not exist" in e for e in errs):
        return errs                      # geometry checks need consistent references

    errs += _check_inside(c)
    errs += _check_spacing(c)

    refs = {"hole": set(hb), "holes": set(hb), "defect": {d["id"] for d in c["defects"]},
            "distractor": {x["id"] for x in c["distractors"]}}
    for f in c["expect"]:
        field = EXPECT_TYPES.get(f.get("type"))
        if field is None:
            errs.append(f"expect {f}: unknown type (use one of {sorted(EXPECT_TYPES)})")
            continue
        vals = f.get(field)
        vals = vals if isinstance(vals, list) else [vals]
        if field == "holes" and len(vals) != 2:
            errs.append(f"expect {f}: pitch needs two holes")
        if not all(v in refs[field] for v in vals):
            errs.append(f"expect {f}: refers to a missing {field}")
    want = {finding_key(f) for f in c["expect"]}
    got = {finding_key(f): f for f in findings(c)}
    for f in c["expect"]:
        if finding_key(f) not in got:
            errs.append(f"expect {f} does not hold geometrically")
    for k, f in got.items():
        if k not in want:
            errs.append(f"unexpected finding {f} (fix the geometry or add it to expect)")
    return errs


def validate_all(coupons: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Problems per coupon, plus a '*' entry for set-level problems."""
    out = {k: validate(c) for k, c in coupons.items()}
    if not any(c["role"] == "test" for c in coupons.values()):
        out["*"] = ["no held-out test coupon (role: test)"]
    return out


# ----------------------------------------------------------------------------- ground truth

def _r(x: float | None, nd: int = 2):
    return "" if x is None else round(float(x), nd)


def _orientation(angle_deg: float) -> float:
    return round(float(angle_deg) % 180.0, 1)


def _buffer(pts: np.ndarray, hw: float) -> np.ndarray:
    """Closed outline of a polyline widened by hw (at least MIN_HALF_WIDTH_MM) on each side."""
    hw = max(float(hw), MIN_HALF_WIDTH_MM)
    p = np.asarray(pts, float)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return np.vstack([p + hw * n, (p - hw * n)[::-1]])


def nick_poly(c: dict[str, Any], p: dict[str, Any], n: int = 33) -> np.ndarray:
    """Half disc of diameter d_mm cut into the plate at the named edge (the twin's nick outline)."""
    W, H = plate_size(c)
    inward = np.array({"top": (0.0, 1.0), "bottom": (0.0, -1.0), "left": (1.0, 0.0), "right": (-1.0, 0.0)}[p["edge"]])
    along = np.array([inward[1], -inward[0]])
    ctr = np.array([0.0 if p["edge"] == "left" else W if p["edge"] == "right" else float(p["u"]),
                    0.0 if p["edge"] == "top" else H if p["edge"] == "bottom" else float(p["v"])])
    a = np.linspace(0.0, math.pi, n)
    return ctr + float(p["d_mm"]) / 2 * (np.outer(np.cos(a), along) + np.outer(np.sin(a), inward))


def ring_poly(cu: float, cv: float, r_in: float, r_out: float, n: int = 64) -> np.ndarray:
    """Annulus as one keyhole polygon (outer circle, bridge, inner circle reversed), as the twin."""
    outer, inner = ellipse(cu, cv, r_out, r_out, 0.0, n), ellipse(cu, cv, r_in, r_in, 0.0, n)
    return np.vstack([outer, outer[:1], inner[:1], inner[:0:-1], inner[:1]])


def _poly_json(pts) -> str:
    return json.dumps(np.round(np.asarray(pts, float), 3).tolist())


def _notes(c: dict[str, Any], d: dict[str, Any]) -> str:
    bits = [f"measured {d['measured']}" if d.get("measured") else "planned"]
    if d.get("tool"):
        bits.append(str(d["tool"]))
    p = d["params"]
    if "depth_mm" in p:
        bits.append(f"depth {p['depth_mm']} mm")
    if d["cls"] == "crack":
        bits.append(f"visible beyond heads {visible_crack_mm(c, d):.2f} mm")
    if d["kind"] == "galvanic":
        bits.append(f"ring outside the fastener head at {p['hole']}")
    if d.get("note"):
        bits.append(str(d["note"]))
    return "; ".join(bits)


def ground_truth_rows(c: dict[str, Any]) -> list[dict[str, Any]]:
    """One DESIGN section 6 CSV row per defect, with the planned geometry.

    eval_class = cls, except HIDDEN for a crack with less than MIN_VISIBLE_CRACK_MM outside
    the installed heads: it exists, but no camera can see it (refer to NDT), so it must be
    neither a YOLO label nor a scored miss. poly_mm is given where a circle or rectangle
    would be wrong: edge nick (half disc), galvanic ring, oval smooth dent, curved scratch,
    fatigue crack.
    """
    rows = []
    for d in c["defects"]:
        p, kind = d["params"], d["kind"]
        u = v = length = width = diam = angle = None
        poly = ""
        if d["cls"] == "dent":
            u, v, diam = p["u"], p["v"], p["d_mm"]
            if kind == "smooth":
                angle = _orientation(p["angle_deg"])
                if float(p["aspect"]) != 1.0:
                    poly = _poly_json(ellipse(u, v, diam / 2, diam / 2 / float(p["aspect"]), p["angle_deg"], 64))
            if kind == "nick":
                poly = _poly_json(nick_poly(c, p))
        elif kind in ("scriber", "gouge"):
            ctrl = np.asarray(p["ctrl_mm"], float)
            chord = ctrl[3] - ctrl[0]
            nrm = np.array([-chord[1], chord[0]]) / max(np.linalg.norm(chord), 1e-9)
            straight = float(np.abs((ctrl - ctrl[0]) @ nrm).max()) <= 0.05
            path = bezier(ctrl)
            u, v = (ctrl[0] + ctrl[3]) / 2
            length, width = polyline_length(path), p["width_mm"]
            angle = _orientation(math.degrees(math.atan2(chord[1], chord[0])))
            if not straight:
                poly = json.dumps(np.round(_buffer(path[::10], width / 2), 3).tolist())
        elif kind == "abrasion":
            u, v, length, width, angle = p["u"], p["v"], p["w_mm"], p["h_mm"], _orientation(p["angle_deg"])
        elif kind == "slit":
            s = slit_segment(p)
            u, v = s.mean(axis=0)
            length, width, angle = p["length_mm"], p["width_mm"], _orientation(p["angle_deg"])
        elif kind == "fatigue":
            pts = np.asarray(p["pts_mm"], float)
            u, v = (pts[0] + pts[-1]) / 2
            ch = pts[-1] - pts[0]
            length, width = polyline_length(pts), p["width_mm"]
            angle = _orientation(math.degrees(math.atan2(ch[1], ch[0])))
            poly = json.dumps(np.round(_buffer(pts, width / 2), 3).tolist())
        elif kind == "pitting":
            u, v, diam = p["u"], p["v"], 2 * float(p["r_mm"])
        elif kind == "galvanic":
            h = holes_by_id(c)[p["hole"]]
            u, v, diam = h["u"], h["v"], 2 * galvanic_radius(c, p)
            poly = _poly_json(ring_poly(u, v, fastener_radius(c, h), diam / 2))
        elif kind == "fretting":
            u, v, length, width, ang = fretting_rect(c, p)
            angle = _orientation(ang)
        hidden = d["cls"] == "crack" and visible_crack_mm(c, d) < MIN_VISIBLE_CRACK_MM - 1e-9
        rows.append({
            "defect_id": d["id"], "plate": c["plate_id"], "face": c["face"], "cls": d["cls"], "kind": kind,
            "u_mm": _r(u), "v_mm": _r(v), "length_mm": _r(length), "width_mm": _r(width, 3),
            "diameter_mm": _r(diam), "angle_deg": "" if angle is None else angle, "poly_mm": poly,
            "eval_class": HIDDEN if hidden else d["cls"], "notes": _notes(c, d),
        })
    return rows


def fastener_rows(c: dict[str, Any]) -> list[dict[str, Any]]:
    """One CSV row per fastener, as the twin labels them (post.plan_items): cls rivet (a rivet
    or screw head circle, defect_id = rivet id) or open_hole (the hole ellipse, defect_id =
    hole id). Without these rows real photos would teach the detector that rivets are background."""
    rows = []
    for h in c["holes"]:
        r = rivet_in(c, h["id"])
        base = {"plate": c["plate_id"], "face": c["face"], "u_mm": _r(h["u"]), "v_mm": _r(h["v"]),
                "length_mm": "", "width_mm": "", "angle_deg": "", "poly_mm": ""}
        bits = [f"measured {h['measured']}" if h.get("measured") else "planned"]
        if r is not None:
            bits.append(f"{r['kind']} head {float(r['head_d_mm']):g} mm in {h['id']} (drill {float(h['d_mm']):g})")
            rows.append({**base, "defect_id": r["id"], "cls": "rivet", "kind": r["kind"],
                         "diameter_mm": _r(r["head_d_mm"]), "eval_class": "rivet", "notes": "; ".join(bits)})
            continue
        a, b = hole_axes(h)
        row = {**base, "defect_id": h["id"], "cls": "open_hole", "kind": "open_hole",
               "diameter_mm": _r(2 * b), "eval_class": "open_hole"}
        if a > b:
            row.update(length_mm=_r(2 * a), width_mm=_r(2 * b, 3), angle_deg=_orientation(h.get("angle_deg", 0.0)),
                       poly_mm=_poly_json(ellipse(h["u"], h["v"], a, b, h.get("angle_deg", 0.0), 64)))
            bits.append(f"oval {2 * a:.2f} x {2 * b:.2f} mm")
        else:
            bits.append(f"open hole {2 * b:g} mm")
        rows.append({**row, "notes": "; ".join(bits)})
    return rows


def write_ground_truth_template(coupons: dict[str, dict[str, Any]] | Iterable[dict[str, Any]],
                                path: str | Path, fasteners: bool = True) -> Path:
    """CSV with the planned value of every fastener (unless fasteners=False) and defect;
    regenerate data/real/ground_truth.csv from the measured YAML with it."""
    items = coupons.values() if isinstance(coupons, dict) else coupons
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for c in items:
            if fasteners:
                w.writerows(fastener_rows(c))
            w.writerows(ground_truth_rows(c))
    return path


# ----------------------------------------------------------------------------- CLI

def _fmt(x) -> str:
    return "-" if x is None else (f"{x:g}" if isinstance(x, float) else str(x))


def print_table(c: dict[str, Any], file=sys.stdout) -> None:
    W, H = plate_size(c)
    print(f"{c['id']}  {c['name']}  ({c['role']}, {W:g} x {H:g} x {c['plate']['thickness_mm']:g} mm, "
          f"D = {c['fastener_d_mm']:g} mm)", file=file)
    rows = fastener_table(c)
    if rows:
        print("  hole      u      v    d  oval  fastener     edge mm (D)   pitch mm (D)  to", file=file)
        for r in rows:
            print(f"  {r['hole']:<4} {r['u']:6.2f} {r['v']:6.2f} {r['d_mm']:4.1f} {r['ovality']:5.2f}  "
                  f"{r['fastener']:<11} {r['edge_mm']:6.2f} ({r['edge_D']:4.2f})  "
                  f"{_fmt(r['pitch_mm']):>6} ({_fmt(r['pitch_D'])})  {_fmt(r['neighbour'])}", file=file)
    for f in findings(c):
        print("  finding:", ", ".join(f"{k}={v}" for k, v in f.items()), file=file)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ayas coupons",
                                 description="Check the physical coupons and write their templates.")
    ap.add_argument("--file", default=DEFAULT_PATH, help="coupons YAML (default %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="check every coupon; exit 1 on problems")
    t = sub.add_parser("table", help="fastener tables and findings")
    t.add_argument("keys", nargs="*", help="coupon ids (default: all)")
    w = sub.add_parser("templates", help="1:1 SVG templates, index.html and the ground-truth CSV template")
    w.add_argument("--out", default="docs/twin/templates")
    w.add_argument("--csv", default=None, help="CSV path (default <out>/ground_truth_template.csv)")
    a = ap.parse_args(argv)

    coupons = load_coupons(a.file)
    problems = {k: v for k, v in validate_all(coupons).items() if v}
    if a.cmd == "table":
        if unknown := [k for k in a.keys if k not in coupons]:
            print(f"unknown coupon(s) {unknown}; known: {', '.join(coupons)}", file=sys.stderr)
            return 2
        for k in a.keys or list(coupons):
            print_table(coupons[k])
            print()
        return 0
    for k, errs in problems.items():
        for e in errs:
            print(f"{k}: {e}")
    if a.cmd == "validate":
        if not problems:
            print(f"{len(coupons)} coupons OK: {', '.join(coupons)}")
        return 1 if problems else 0
    if problems:
        print("not writing templates: fix the problems above first")
        return 1
    from .templates import write_all

    out = Path(a.out)
    paths = write_all(coupons, out)
    csv_path = write_ground_truth_template(coupons, a.csv or out / "ground_truth_template.csv")
    for p in [*paths, csv_path]:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
