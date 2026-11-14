"""Defect footprints, fastener outlines and distances in plate millimetres (pure numpy).

Conventions (DESIGN.md sections 1 and 3, DESIGN_STEEL.md A):
- Plate frame: u right, v down, mm, origin at the datum corner. ``angle_deg`` is measured
  from +u toward +v, so a positive angle turns clockwise as seen on the front face.
- A polygon is an (N, 2) array (or a list of [u, v]) of vertices, not closed (last != first).
  ``footprint`` returns plain lists so plans stay JSON.
- A ring (galvanic corrosion) is one "keyhole" polygon: the outer circle, a zero-width bridge
  and the inner circle in the opposite direction. cv2.fillPoly, the shoelace area and the
  even-odd point test all treat it as an annulus.

Visible fastener outline. Every fastener distance in this module (``near()['fastener_edge_mm']``)
is edge to edge between a defect outline and the outline the camera SEES of the fastener:
- a filled hole shows its rivet or screw head: a circle of head_d_mm round the hole centre;
- an open hole shows the hole itself: an ellipse with minor axis d_mm and major axis
  d_mm * ovality along the hole's angle_deg.
Steel plates have no rivets, so every hole is open and the fastener outline is the hole edge.
``near()['near_edge_mm']`` is the record 3.0 field: the distance from the outline to the
nearest plate edge or hole (fastener) edge, whichever is closer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

CIRCLE_N = 64          # vertices of a full circle: sagitta < 0.03 px at 0.08 mm/px for r = 12 mm
CAP_N = 12             # vertices of a round end cap
MIN_HALF_WIDTH = 0.05  # mm: thinner lines are widened to 0.1 mm so they rasterise
PIT_N = 24             # vertices per pit disc in a pit-cluster hull
STREAK_TIP = 0.35      # a rust streak tapers to this fraction of its start width
FINGERPRINT_ASPECT = 1.25   # default (widest) fingerprint ellipse when params carry no aspect


def crack_lip_mm(width_mm: float) -> float:
    """Raised-lip width on each side of an open crack (inside its footprint, see _shape)."""
    return min(0.1, max(0.04, 0.5 * float(width_mm)))


# ----------------------------------------------------------------------------- primitives
def unit(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


def circle(c, r: float, n: int = CIRCLE_N) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.asarray(c, float) + r * np.c_[np.cos(a), np.sin(a)]


def ellipse(c, a: float, b: float, angle_deg: float = 0.0, n: int = CIRCLE_N) -> np.ndarray:
    """Ellipse with semi-axis a along angle_deg and b across it."""
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    e1, e2 = unit(angle_deg), unit(angle_deg + 90.0)
    return np.asarray(c, float) + np.outer(a * np.cos(t), e1) + np.outer(b * np.sin(t), e2)


def rect(c, w: float, h: float, angle_deg: float = 0.0) -> np.ndarray:
    """Rectangle centred on c: w along angle_deg, h across it."""
    e1, e2 = unit(angle_deg) * w / 2, unit(angle_deg + 90.0) * h / 2
    c = np.asarray(c, float)
    return np.array([c - e1 - e2, c + e1 - e2, c + e1 + e2, c - e1 + e2])


def ring(c, r_in: float, r_out: float, n: int = CIRCLE_N) -> np.ndarray:
    """Annulus as one keyhole polygon (outer circle, bridge, inner circle reversed)."""
    outer, inner = circle(c, r_out, n), circle(c, r_in, n)
    return np.vstack([outer, outer[:1], inner[:1], inner[:0:-1], inner[:1]])


def polygon_area(poly) -> float:
    """Shoelace area (absolute value); a keyhole ring gives outer minus inner."""
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polyline_length(pts) -> float:
    p = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def convex_hull(pts) -> np.ndarray:
    """Convex hull (Andrew's monotone chain), counter-clockwise in (u, v), collinear points dropped."""
    p = np.unique(np.asarray(pts, float).reshape(-1, 2), axis=0)
    if len(p) <= 2:
        return p

    def half(seq):
        out: list = []
        for q in seq:
            while len(out) >= 2 and _cross(np.asarray(out[-2]), np.asarray(out[-1]), q) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1], float)


def equivalent_diameter(area_mm2: float) -> float:
    """Diameter of the circle with the same area."""
    return 2.0 * math.sqrt(max(float(area_mm2), 0.0) / math.pi)


def bezier_points(ctrl, n: int = 65) -> np.ndarray:
    """Points of the cubic Bezier with 4 control points, at n even parameter steps."""
    P = np.asarray(ctrl, float).reshape(4, 2)
    t = np.linspace(0.0, 1.0, n)[:, None]
    s = 1.0 - t
    return s**3 * P[0] + 3 * s**2 * t * P[1] + 3 * s * t**2 * P[2] + t**3 * P[3]


def bezier_length(ctrl, n: int = 2049) -> float:
    """Arc length of a cubic Bezier (dense polyline; exact for a straight, monotone control polygon)."""
    return polyline_length(bezier_points(ctrl, n))


def buffer_polyline(pts, half_width: float, cap_n: int = CAP_N) -> np.ndarray:
    """Outline of a polyline thickened by half_width on each side, with round end caps.

    Offsets each vertex along the averaged segment normal (miter, capped at 2x), so it is
    meant for smooth curves and gently jagged cracks, not for hairpin bends tighter than the
    half-width. half_width is raised to MIN_HALF_WIDTH.
    """
    hw = max(float(half_width), MIN_HALF_WIDTH)
    p = np.asarray(pts, float).reshape(-1, 2)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]
    p = p[keep]
    if len(p) == 1:
        return circle(p[0], hw, 4 * cap_n)
    t = np.diff(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.c_[-t[:, 1], t[:, 0]]                       # segment normals (t turned +90 deg)
    vn = np.vstack([n[:1], n[:-1] + n[1:], n[-1:]])
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    seg_n = np.vstack([n[:1], n])                      # normal of a segment touching each vertex
    scale = 1.0 / np.maximum(np.sum(vn * seg_n, axis=1), 0.5)
    off = vn * (hw * scale)[:, None]
    left, right = p + off, p - off

    def cap(c, ang0):
        a = ang0 + np.linspace(0.0, -math.pi, cap_n + 2)[1:-1]
        return c + hw * np.c_[np.cos(a), np.sin(a)]

    a_end = math.atan2(t[-1, 1], t[-1, 0]) + math.pi / 2     # from +n round the front to -n
    a_start = math.atan2(t[0, 1], t[0, 0]) - math.pi / 2     # from -n round the back to +n
    return np.vstack([left, cap(p[-1], a_end), right[::-1], cap(p[0], a_start)])


# ----------------------------------------------------------------------------- predicates, distances
def point_in_polygon(pts, poly) -> np.ndarray:
    """Even-odd test of (M, 2) points against a polygon; returns (M,) bools."""
    q = np.atleast_2d(np.asarray(pts, float))
    P = np.asarray(poly, float)
    x1, y1 = P[:, 0], P[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    x, y = q[:, :1], q[:, 1:2]
    crosses = (y1 > y) != (y2 > y)
    dy = np.where(y2 == y1, 1.0, y2 - y1)
    xint = x1 + (y - y1) * (x2 - x1) / dy
    return (np.sum(crosses & (x < xint), axis=1) % 2) == 1


def _seg(poly) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(poly, float)
    return P, np.roll(P, -1, axis=0)


def point_segment_distance(pts, a, b) -> np.ndarray:
    """(M, K) distances from M points to K segments a[k]-b[k]."""
    q = np.atleast_2d(np.asarray(pts, float))[:, None, :]
    a, b = np.asarray(a, float)[None], np.asarray(b, float)[None]
    ab = b - a
    L2 = np.maximum(np.sum(ab * ab, axis=-1), 1e-18)
    s = np.clip(np.sum((q - a) * ab, axis=-1) / L2, 0.0, 1.0)
    return np.linalg.norm(q - (a + s[..., None] * ab), axis=-1)


def _cross(o, a, b):
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def edges_intersect(poly_a, poly_b) -> bool:
    """True if any edge of a crosses or touches any edge of b (collinear overlaps excluded)."""
    a0, a1 = _seg(poly_a)
    b0, b1 = _seg(poly_b)
    A0, A1, B0, B1 = a0[:, None], a1[:, None], b0[None], b1[None]
    d1, d2 = _cross(B0, B1, A0), _cross(B0, B1, A1)
    d3, d4 = _cross(A0, A1, B0), _cross(A0, A1, B1)
    hit = (d1 * d2 <= 0) & (d3 * d4 <= 0) & ~((d1 == 0) & (d2 == 0))
    return bool(hit.any())


def poly_distance(poly_a, poly_b) -> float:
    """Edge-to-edge distance between two polygons; 0 if they touch, cross or one holds the other."""
    A, B = np.asarray(poly_a, float), np.asarray(poly_b, float)
    if point_in_polygon(A, B).any() or point_in_polygon(B, A).any() or edges_intersect(A, B):
        return 0.0
    a0, a1 = _seg(A)
    b0, b1 = _seg(B)
    return float(min(point_segment_distance(A, b0, b1).min(), point_segment_distance(B, a0, a1).min()))


def polys_distance(polys_a, polys_b) -> float:
    return min(poly_distance(a, b) for a in polys_a for b in polys_b)


def _isect(a, b, axis: int, bound: float) -> tuple[float, float]:
    t = (bound - a[axis]) / (b[axis] - a[axis])
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def clip_to_rect(poly, u0: float, v0: float, u1: float, v1: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to the rectangle [u0, u1] x [v0, v1] (may be empty)."""
    p = [tuple(map(float, q)) for q in np.asarray(poly, float)]
    for axis, bound, lower in ((0, u0, True), (0, u1, False), (1, v0, True), (1, v1, False)):
        out = []
        for i, cur in enumerate(p):
            prev = p[i - 1]
            cin = cur[axis] >= bound if lower else cur[axis] <= bound
            pin = prev[axis] >= bound if lower else prev[axis] <= bound
            if cin != pin:
                out.append(_isect(prev, cur, axis, bound))
            if cin:
                out.append(cur)
        p = out
        if not p:
            break
    return np.array(p, float).reshape(-1, 2)


def bbox(polys) -> tuple[float, float, float, float]:
    """(u0, v0, u1, v1) of a list of polygons."""
    p = np.vstack([np.asarray(q, float) for q in polys])
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def plate_edge_distance(polys, size_mm) -> float:
    """Distance from the outlines to the nearest plate edge (0 if any vertex is on or outside it)."""
    W, H = size_mm
    p = np.vstack([np.asarray(q, float) for q in polys])
    d = np.minimum.reduce([p[:, 0], W - p[:, 0], p[:, 1], H - p[:, 1]])
    return max(0.0, float(d.min()))


# ----------------------------------------------------------------------------- fasteners
def _by_id(items: list[dict], key: str, value: str) -> dict | None:
    return next((x for x in items if x.get(key) == value), None)


def rivet_for(hole: dict, plan: dict) -> dict | None:
    return _by_id(plan.get("rivets", []), "hole", hole["id"]) if hole.get("filled") else None


def visible_radius(hole: dict, rivet: dict | None = None) -> float:
    """Largest radius of the visible fastener outline (head radius, or the hole's major semi-axis)."""
    if rivet is not None:
        return rivet["head_d_mm"] / 2
    return hole["d_mm"] * max(1.0, hole.get("ovality", 1.0)) / 2


def fastener_outline(hole: dict, rivet: dict | None = None, n: int = CIRCLE_N) -> np.ndarray:
    """VISIBLE outline of a fastener (see the module docstring).

    rivet given (filled hole): head circle of head_d_mm round the hole centre (tilt ignored).
    rivet None (open hole): ellipse, minor axis d_mm, major axis d_mm * ovality along angle_deg.
    """
    c = (hole["u"], hole["v"])
    if rivet is not None:
        return circle(c, rivet["head_d_mm"] / 2, n)
    r = hole["d_mm"] / 2
    return ellipse(c, r * hole.get("ovality", 1.0), r, hole.get("angle_deg", 0.0), n)


def fastener_outlines(plan: dict) -> list[tuple[str, np.ndarray]]:
    """(hole id, visible outline) for every hole of the plan."""
    return [(h["id"], fastener_outline(h, rivet_for(h, plan))) for h in plan.get("holes", [])]


def outline_radius_at(hole: dict, rivet: dict | None, angle_deg: float) -> float:
    """Distance from the hole centre to its visible outline along angle_deg."""
    if rivet is not None:
        return rivet["head_d_mm"] / 2
    b = hole["d_mm"] / 2
    a = b * hole.get("ovality", 1.0)
    phi = math.radians(angle_deg - hole.get("angle_deg", 0.0))
    return a * b / math.hypot(b * math.cos(phi), a * math.sin(phi))


def near(polys_mm, plan: dict) -> dict[str, Any]:
    """Edge-to-edge distances of a defect outline (DESIGN 3: ``near``).

    fastener_edge_mm: to the nearest VISIBLE fastener outline (head circle for a filled hole,
    hole ellipse for an open one), 0 if they overlap or touch, None if the plan has no holes.
    plate_edge_mm: to the nearest plate edge. fastener: id of the nearest hole (extra key).
    near_edge_mm: min(plate_edge_mm, fastener_edge_mm), the record 3.0 edge distance.
    """
    polys = [np.asarray(p, float) for p in polys_mm]
    best, best_id = None, None
    for hid, outline in fastener_outlines(plan):
        d = min(poly_distance(p, outline) for p in polys)
        if best is None or d < best:
            best, best_id = d, hid
    plate = round(plate_edge_distance(polys, plan["plate"]["size_mm"]), 4)
    hole = None if best is None else round(best, 4)
    return {"fastener_edge_mm": hole, "plate_edge_mm": plate, "fastener": best_id,
            "near_edge_mm": plate if hole is None else min(plate, hole)}


# ----------------------------------------------------------------------------- footprints
def _hw(width: float) -> float:
    return max(width / 2, MIN_HALF_WIDTH)


def _nick(p: dict, size_mm) -> np.ndarray:
    """Half disc of diameter d_mm cut into the plate at (u, v) on the named edge."""
    W, H = size_mm
    inward = {"top": (0.0, 1.0), "bottom": (0.0, -1.0), "left": (1.0, 0.0), "right": (-1.0, 0.0)}[p["edge"]]
    nin = np.array(inward)
    tt = np.array([nin[1], -nin[0]])
    c = np.array([p["u"], p["v"]], float)
    c = np.array([0.0 if p["edge"] == "left" else W if p["edge"] == "right" else c[0],
                  0.0 if p["edge"] == "top" else H if p["edge"] == "bottom" else c[1]])
    a = np.linspace(0.0, math.pi, CIRCLE_N // 2 + 1)
    return c + (p["d_mm"] / 2) * (np.outer(np.cos(a), tt) + np.outer(np.sin(a), nin))


def _fretting(p: dict, hole: dict, rivet: dict) -> np.ndarray:
    """Streak from the head edge along angle_deg: width_mm at the head, half that at the
    rounded tail, length_mm from the head edge to the tip; the head itself is excluded."""
    c = np.array([hole["u"], hole["v"]], float)
    R = rivet["head_d_mm"] / 2
    t, n = unit(p["angle_deg"]), unit(p["angle_deg"] + 90.0)
    hw0 = min(p["width_mm"] / 2, 0.95 * R)
    hw1 = max(hw0 / 2, MIN_HALF_WIDTH)
    s0 = math.sqrt(R * R - hw0 * hw0)
    s1 = max(R + p["length_mm"] - hw1, s0 + 1e-3)
    s = np.linspace(s0, s1, 24)
    hw = hw0 + (hw1 - hw0) * (s - s0) / (s1 - s0)
    left = c + np.outer(s, t) + hw[:, None] * n
    right = c + np.outer(s, t) - hw[:, None] * n
    th = math.atan2(t[1], t[0])
    a = th + math.pi / 2 - np.linspace(0.0, math.pi, CAP_N + 2)[1:-1]
    tail = c + s1 * t + hw1 * np.c_[np.cos(a), np.sin(a)]
    al = math.asin(hw0 / R)
    arc_a = np.linspace(th - al, th + al, 16)[1:-1]             # right -> left, facing the streak
    arc = c + R * np.c_[np.cos(arc_a), np.sin(arc_a)]
    return np.vstack([left, tail, right[::-1], arc])


def streak_widths(p: dict) -> tuple[float, float, float]:
    """(start half-width, tip half-width, distance from the start to the tip-cap centre) of a rust streak."""
    hw0 = max(float(p["width_mm"]) / 2, MIN_HALF_WIDTH)
    hw1 = max(STREAK_TIP * hw0, MIN_HALF_WIDTH)
    return hw0, hw1, max(float(p["length_mm"]) - hw1, 1e-3)


def _rust_streak(p: dict) -> np.ndarray:
    """Tapered streak from (u0, v0) along angle_deg: round start cap of width_mm, round tip
    STREAK_TIP times as wide at length_mm, plus the optional head spot (head_r_mm) at the start.
    The pieces are convex, so their hull is the outline."""
    s0 = np.array([p["u0"], p["v0"]], float)
    hw0, hw1, D = streak_widths(p)
    pts = [circle(s0, hw0, 32), circle(s0 + D * unit(p["angle_deg"]), hw1, 16)]
    head = float(p.get("head_r_mm") or 0.0)
    if head > 0:
        pts.append(circle(s0, head, 48))
    return convex_hull(np.vstack(pts))


def pit_discs(p: dict) -> list[tuple[float, float, float]]:
    """(u, v, outline radius) of the pits of a pit.cluster: pit radius plus the rust halo."""
    halo = float(p.get("halo_mm") or 0.0)
    return [(float(q[0]), float(q[1]), float(q[2]) / 2 + halo) for q in p.get("pits") or []]


def _pit_cluster(p: dict) -> np.ndarray:
    discs = pit_discs(p)
    if not discs:
        return circle((p["u"], p["v"]), p["r_mm"])
    return convex_hull(np.vstack([circle((u, v), r, PIT_N) for u, v, r in discs]))


def _steel_shape(defect: dict, plan: dict) -> tuple[list[np.ndarray], dict[str, float]] | None:
    """Footprints of the steel kinds (DESIGN_STEEL A); None for any other kind."""
    cls, kind, p = defect["cls"], defect.get("kind"), defect["params"]
    W, H = plan["plate"]["size_mm"]
    if cls == "pit" and kind == "single":
        poly = circle((p["u"], p["v"]), p["d_mm"] / 2 + float(p.get("halo_mm") or 0.0))
        return [poly], {"diameter_mm": p["d_mm"], "count": 1}
    if cls == "pit" and kind == "cluster":
        poly = _pit_cluster(p)
        area = polygon_area(poly)
        pits = p.get("pits") or []
        big = max((float(q[2]) for q in pits), default=float(max(p.get("pit_d_mm") or [0.0])))
        return [poly], {"diameter_mm": equivalent_diameter(area), "count": len(pits) or p.get("n", 0),
                        "area_mm2": area, "pit_d_mm": big}     # pit_d_mm: the largest pit (spec_steel pit rows)
    if cls == "corrosion" and kind == "rust":
        a = p["r_mm"]
        b = a / p.get("aspect", 1.0)
        area = float(p.get("coverage", 1.0)) * math.pi * a * b
        return [ellipse((p["u"], p["v"]), a, b, p.get("angle_deg", 0.0))], \
            {"area_mm2": area, "diameter_mm": equivalent_diameter(area)}
    if cls == "corrosion" and kind == "rust_streak":
        poly = _rust_streak(p)
        area = polygon_area(clip_to_rect(poly, 0.0, 0.0, W, H))
        return [poly], {"area_mm2": area, "diameter_mm": equivalent_diameter(area), "length_mm": p["length_mm"]}
    if cls == "corrosion" and kind == "white":
        r = p["r_mm"]
        return [circle((p["u"], p["v"]), r)], {"area_mm2": math.pi * r * r, "diameter_mm": 2 * r}
    if cls == "crack" and kind == "open":
        poly = buffer_polyline(p["pts_mm"], _hw(p["width_mm"]) + crack_lip_mm(p["width_mm"]))
        return [poly], {"length_mm": polyline_length(p["pts_mm"]), "width_mm": p["width_mm"]}
    if cls == "stain" and kind in ("oil", "fingerprint"):
        a = p["r_mm"]
        b = a / p.get("aspect", 1.0 if kind == "oil" else FINGERPRINT_ASPECT)
        return [ellipse((p["u"], p["v"]), a, b, p.get("angle_deg", 0.0))], {"area_mm2": math.pi * a * b,
                                                                          "diameter_mm": 2 * a}
    if cls == "stain" and kind == "water":
        r = p["r_mm"]
        return [circle((p["u"], p["v"]), r)], {"area_mm2": math.pi * r * r, "diameter_mm": 2 * r}
    return None


def _shape(defect: dict, plan: dict) -> tuple[list[np.ndarray], dict[str, float]]:
    cls, kind, p = defect["cls"], defect.get("kind"), defect["params"]   # indications have no kind
    steel = _steel_shape(defect, plan)
    if steel is not None:
        return steel
    if cls == "dent":
        if kind == "nick":
            return [_nick(p, plan["plate"]["size_mm"])], {"diameter_mm": p["d_mm"]}
        c = (p["u"], p["v"])
        if kind == "smooth":
            poly = ellipse(c, p["d_mm"] / 2, p["d_mm"] / 2 / p.get("aspect", 1.0), p.get("angle_deg", 0.0))
        else:                                                  # punch, ball: the rim lies inside d_mm
            poly = circle(c, p["d_mm"] / 2)
        return [poly], {"diameter_mm": p["d_mm"], "depth_mm": p["depth_mm"]}
    if cls == "scratch":
        if kind == "abrasion":
            w, h = p["w_mm"], p["h_mm"]
            poly = rect((p["u"], p["v"]), w, h, p.get("angle_deg", 0.0))
            return [poly], {"area_mm2": w * h, "length_mm": max(w, h), "width_mm": min(w, h)}
        # scriber, gouge: the groove plus burr lips of up to width/4 a side
        hw = max(p["width_mm"] * (0.5 + 0.25 * p.get("burr", 0.0)), MIN_HALF_WIDTH)
        poly = buffer_polyline(bezier_points(p["ctrl_mm"]), hw)
        return [poly], {"length_mm": bezier_length(p["ctrl_mm"]), "width_mm": p["width_mm"]}
    if cls == "crack":
        if kind == "slit":
            d = unit(p["angle_deg"])
            c = np.array([p["u0"], p["v0"]]) + d * p["length_mm"] / 2
            poly = rect(c, p["length_mm"], 2 * _hw(p["width_mm"]), p["angle_deg"])
            return [poly], {"length_mm": p["length_mm"], "width_mm": p["width_mm"]}
        poly = buffer_polyline(p["pts_mm"], _hw(p["width_mm"]))         # fatigue
        return [poly], {"length_mm": polyline_length(p["pts_mm"]), "width_mm": p["width_mm"]}
    if cls == "corrosion":
        if kind == "pitting":
            r = p["r_mm"]
            return [circle((p["u"], p["v"]), r)], {"area_mm2": math.pi * r * r, "diameter_mm": 2 * r}
        if kind == "galvanic":
            hole = _by_id(plan["holes"], "id", p["hole"])
            ri = visible_radius(hole, rivet_for(hole, plan))
            ro = ri + p["ring_w_mm"]
            return [ring((hole["u"], hole["v"]), ri, ro)], {"area_mm2": math.pi * (ro * ro - ri * ri),
                                                           "diameter_mm": 2 * ro}
        if kind == "fretting":
            rivet = _by_id(plan["rivets"], "id", p["rivet"])
            hole = _by_id(plan["holes"], "id", rivet["hole"])
            poly = _fretting(p, hole, rivet)
            return [poly], {"area_mm2": polygon_area(poly), "length_mm": p["length_mm"]}
    if cls == "linear_indication":
        hw = _hw(p.get("width_mm", 0.0)) + p["bloom_mm"]
        return [buffer_polyline(p["pts_mm"], hw)], {"length_mm": polyline_length(p["pts_mm"]) + 2 * hw,
                                                    "width_mm": 2 * hw}
    if cls == "rounded_indication":
        return [circle((p["u"], p["v"]), p["d_mm"] / 2)], {"diameter_mm": p["d_mm"]}
    raise ValueError(f"no footprint for {cls}.{kind}")


def _r4(x) -> Any:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r4(y) for y in x]
    return round(float(x), 4)


def footprint(defect: dict, plan: dict) -> dict[str, Any]:
    """Analytic outline(s) and measures of a defect or penetrant indication (DESIGN 3.1, 3.2,
    DESIGN_STEEL A).

    Returns {'polys_mm': [[[u, v], ...], ...], 'measure': {...}}. Outline conventions:
    punch/ball circle of d_mm (rim inside); smooth ellipse (major d_mm); nick half disc on the
    edge; scriber/gouge Bezier buffered by width/2 plus burr lips (width/4 * burr a side);
    abrasion w x h rectangle; slit rectangle from (u0, v0); fatigue buffered polyline;
    pitting circle r_mm; galvanic keyhole ring outside the visible fastener outline; fretting
    tapered streak outside the head; linear indication: line buffered by width/2 + bloom;
    rounded indication: circle d_mm. Lines thinner than 0.1 mm are widened to 0.1 mm.
    Steel kinds: pit.single circle d_mm/2 + halo_mm; pit.cluster convex hull of the pit discs
    (d/2 + halo_mm each); corrosion.rust ellipse r_mm x r_mm/aspect along angle_deg (the ragged
    rust lies inside it); corrosion.rust_streak tapered streak (streak_widths) plus head spot;
    corrosion.white circle r_mm; crack.open polyline buffered by width/2 + crack_lip_mm(width);
    stain.oil and stain.fingerprint ellipse r_mm x r_mm/aspect; stain.water circle r_mm.
    Measures: diameter_mm is the equivalent diameter for pit.cluster and corrosion (area based),
    the major axis for stains; rust area_mm2 = coverage x outline area; pit.cluster also gives
    pit_d_mm, its largest pit's diameter (the rules' pit size; not the pit_d_mm range bound).
    Outlines are clipped to the plate face [0, W] x [0, H]; measures are not changed by it
    (except the rust streak area, taken on the plate).
    """
    polys, measure = _shape(defect, plan)
    W, H = plan["plate"]["size_mm"]
    inside = []
    for p in polys:
        if p[:, 0].min() < 0 or p[:, 1].min() < 0 or p[:, 0].max() > W or p[:, 1].max() > H:
            p = clip_to_rect(p, 0.0, 0.0, W, H)
        if len(p) >= 3:
            inside.append(p)
    measure = {k: int(v) if k == "count" else _r4(v) for k, v in measure.items()}   # counts stay integers
    return {"polys_mm": [_r4(p) for p in inside], "measure": measure}
