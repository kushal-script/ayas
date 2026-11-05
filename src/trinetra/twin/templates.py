"""1:1 drill and marking templates for the physical coupons (SVG, sizes in mm).

Each SVG is plate width + 2 * MARGIN_MM wide, with width/height in mm and a viewBox in
mm, so printing at 100 % gives true size (check the 50 mm bar with a caliper). Tape the
print on face A with the plate outline on the plate edges and the 'A' on the datum
corner, then centre-punch through the paper at the crosshairs.

Drawn in the plate frame (u right, v down = SVG x right, y down): outline, datum 'A',
rulers, hole crosshairs with drill size and id, rivet/screw heads (dashed), defects
(circle = dent, line = scratch, arrow = slit, hatched = corrosion, labelled distractors),
a 50 mm scale bar and the fabrication steps.
"""

from __future__ import annotations

import html
import math
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from . import coupons as C

MARGIN_MM = 20.0          # left and right of the plate (svg width = W + 2 * MARGIN_MM)
TOP_MM = 22.0             # title and ruler band above the plate
FOOT_MM = 16.0            # scale bar band below the plate
LINE_MM = 3.3             # notes line height
NOTE_FONT = 2.3
LABEL_FONT = 1.8
FONT = "Helvetica, Arial, sans-serif"
INK = {"hole": "#000000", "head": "#666666", "dent": "#c0392b", "scratch": "#1f5fbf", "crack": "#8e24aa",
       "corrosion": "#b35a00", "distractor": "#2e7d32", "grid": "#999999"}


def _f(x: float) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def svg_size(c: dict[str, Any]) -> tuple[float, float]:
    """(width, height) of the template in mm."""
    W, H = C.plate_size(c)
    return W + 2 * MARGIN_MM, TOP_MM + H + FOOT_MM + LINE_MM * len(_note_lines(c)) + 4.0


# ----------------------------------------------------------------------------- notes

def _expect_text(f: dict[str, Any]) -> str:
    field = C.EXPECT_TYPES.get(f["type"], "")
    v = f.get(field)
    return f"{f['type']} {'-'.join(v) if isinstance(v, list) else v}"


def _note_lines(c: dict[str, Any]) -> list[str]:
    W, _ = C.plate_size(c)
    chars = int((W + 2 * MARGIN_MM - 16) / (NOTE_FONT * 0.5))
    lines = ["Datum corner A = top-left of face A: punch a small 'A' at the marked spot. "
             "Coordinates in mm from A: u right, v down."]
    lines = [w for s in lines for w in textwrap.wrap(s, chars)]
    lines.append("Make:")
    for i, step in enumerate(c["make"], 1):
        lines += textwrap.wrap(f"{i}. {step}", chars, subsequent_indent="    ")
    lines.append("Designed findings:")
    exp = ", ".join(_expect_text(f) for f in c["expect"]) or "none (reference)"
    lines += textwrap.wrap(exp, chars)
    return lines


# ----------------------------------------------------------------------------- labels

class _Labeler:
    """Label placement: the first nearby candidate box clear of drawn features and other labels."""

    CLEAR_MM = 0.6

    def __init__(self, obstacles: np.ndarray, bounds: tuple[float, float, float, float]):
        self.obs = obstacles
        self.boxes: list[tuple[float, float, float, float]] = []
        self.bounds = bounds

    @staticmethod
    def _box_pts_dist(box, pts: np.ndarray) -> float:
        if len(pts) == 0:
            return 1e9
        dx = np.maximum(np.maximum(box[0] - pts[:, 0], 0), pts[:, 0] - box[2])
        dy = np.maximum(np.maximum(box[1] - pts[:, 1], 0), pts[:, 1] - box[3])
        return float(np.hypot(dx, dy).min())

    @staticmethod
    def _box_box_dist(a, b) -> float:
        dx = max(a[0] - b[2], b[0] - a[2], 0.0)
        dy = max(a[1] - b[3], b[1] - a[3], 0.0)
        return math.hypot(dx, dy)

    def _clearance(self, box) -> float:
        bx0, by0, bx1, by1 = self.bounds
        if box[0] < bx0 or box[2] > bx1 or box[1] < by0 or box[3] > by1:
            return -1.0
        d = self._box_pts_dist(box, self.obs)
        if self.boxes:
            d = min(d, min(self._box_box_dist(box, b) for b in self.boxes))
        return d

    def place(self, anchor: tuple[float, float, float, float], text: str, size: float = LABEL_FONT):
        """(x, y_baseline, box) for a left-aligned label next to the anchor box."""
        w, h = len(text) * size * 0.56, size
        x0, y0, x1, y1 = anchor
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        cands = []
        for g in (0.8, 2.5, 5.0, 8.0):
            cands += [(x1 + g, yc - h / 2), (xc - w / 2, y0 - g - h), (xc - w / 2, y1 + g), (x0 - g - w, yc - h / 2),
                      (x1 + g, y0 - g - h), (x1 + g, y1 + g), (x0 - g - w, y0 - g - h), (x0 - g - w, y1 + g)]
        best, best_d = None, -1e9
        for lx, ly in cands:
            box = (lx, ly, lx + w, ly + h)
            d = self._clearance(box)
            if d >= self.CLEAR_MM:
                best = box
                break
            if d > best_d:
                best, best_d = box, d
        self.boxes.append(best)
        return best[0], best[3] - 0.25 * h, best


def _leader(box, anchor, ink) -> str:
    """Thin line from a label box to its feature when they are not adjacent."""
    ax = min(max((box[0] + box[2]) / 2, anchor[0]), anchor[2])
    ay = min(max((box[1] + box[3]) / 2, anchor[1]), anchor[3])
    bx = min(max(ax, box[0]), box[2])
    by = min(max(ay, box[1]), box[3])
    if math.hypot(ax - bx, ay - by) < 1.5:
        return ""
    return (f'<line x1="{_f(bx)}" y1="{_f(by)}" x2="{_f(ax)}" y2="{_f(ay)}" stroke="{ink}" '
            f'stroke-width="0.08" stroke-dasharray="0.4 0.3"/>')


def _bbox(pts) -> tuple[float, float, float, float]:
    p = np.asarray(pts, float)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def _around(p, r: float = 0.6) -> tuple[float, float, float, float]:
    return (p[0] - r, p[1] - r, p[0] + r, p[1] + r)


# ----------------------------------------------------------------------------- drawing

def _ellipse(u, v, a, b, ang, stroke, sw, extra="") -> str:
    return (f'<ellipse cx="{_f(u)}" cy="{_f(v)}" rx="{_f(a)}" ry="{_f(b)}" '
            f'transform="rotate({_f(ang)} {_f(u)} {_f(v)})" fill="none" stroke="{stroke}" '
            f'stroke-width="{_f(sw)}"{extra}/>')


def _cross(u, v, r, stroke=INK["hole"], sw=0.12) -> str:
    return (f'<path d="M{_f(u - r)} {_f(v)}H{_f(u + r)}M{_f(u)} {_f(v - r)}V{_f(v + r)}" '
            f'stroke="{stroke}" stroke-width="{_f(sw)}" fill="none"/>')


def _poly(pts, stroke, fill, sw=0.15, closed=True, extra="") -> str:
    tag = "polygon" if closed else "polyline"
    s = " ".join(f"{_f(x)},{_f(y)}" for x, y in np.asarray(pts, float))
    return f'<{tag} points="{s}" fill="{fill}" stroke="{stroke}" stroke-width="{_f(sw)}"{extra}/>'


def _text(x, y, s, size=LABEL_FONT, fill="#000", anchor="start", weight="normal") -> str:
    return (f'<text x="{_f(x)}" y="{_f(y)}" font-size="{_f(size)}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{_esc(s)}</text>')


def _defs() -> str:
    def hatch(id_, color, ang):
        return (f'<pattern id="{id_}" patternUnits="userSpaceOnUse" width="0.8" height="0.8" '
                f'patternTransform="rotate({ang})"><path d="M0 0.4H0.8" stroke="{color}" stroke-width="0.15"/></pattern>')
    return ("<defs>" + hatch("hatch-corrosion", INK["corrosion"], 45) + hatch("hatch-scratch", INK["scratch"], 0)
            + f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" '
              f'markerWidth="1.4" markerHeight="1.4" orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK["crack"]}"/></marker>'
            + "</defs>")


def _hole_label(c: dict[str, Any], h: dict[str, Any]) -> str:
    r = C.rivet_in(c, h["id"])
    s = f"{h['id']} Ø{h['d_mm']:g}"
    if float(h.get("ovality", 1.0)) > 1.0:
        a, b = C.hole_axes(h)
        s += f" oval {2 * a:.1f}x{2 * b:.1f}"
    if r is None:
        s += " open"
    elif r["kind"] == "screw":
        s += " M4"
    return s


def _defect_items(c: dict[str, Any], d: dict[str, Any]) -> tuple[list[str], str, tuple]:
    """(svg elements, label, label anchor box) for one defect."""
    p, kind, cls = d["params"], d["kind"], d["cls"]
    ink = INK[cls]
    if cls == "dent":
        r = float(p["d_mm"]) / 2
        els = [_ellipse(p["u"], p["v"], r, r, 0, ink, 0.2), _cross(p["u"], p["v"], r + 1.0, ink, 0.1)]
        if kind == "smooth":
            els[0] = _ellipse(p["u"], p["v"], r, r / float(p["aspect"]), p["angle_deg"], ink, 0.2)
        if kind == "punch":
            label = f"{d['id']} " + str(d.get("tool", "punch")).replace("punch ", "")
        else:
            label = f"{d['id']} {kind}" + (f" Ø{p['d_mm']:g}" if kind != "nick" else "")
        return els, label, _bbox(C.ellipse(p["u"], p["v"], r + 1.0, r + 1.0))
    if kind in ("scriber", "gouge"):
        q = np.asarray(p["ctrl_mm"], float)
        path = "M{} {}C{} {} {} {} {} {}".format(*[_f(x) for x in q.ravel()])
        sw = max(float(p["width_mm"]), 0.25)
        els = [f'<path d="{path}" fill="none" stroke="{ink}" stroke-width="{_f(sw)}" stroke-linecap="round"/>',
               f'<circle cx="{_f(q[0][0])}" cy="{_f(q[0][1])}" r="0.6" fill="{ink}"/>',
               _cross(q[3][0], q[3][1], 1.0, ink, 0.12)]
        L = C.polyline_length(C.bezier(q))
        mid = C.bezier(q, 3)[1]
        return els, f"{d['id']} {kind} {L:.0f}", _around(mid, 0.6 + sw)
    if kind == "abrasion":
        pts = C.rect(p["u"], p["v"], p["w_mm"], p["h_mm"], p["angle_deg"])
        return ([_poly(pts, ink, "url(#hatch-scratch)", 0.2)],
                f"{d['id']} abrasion", _bbox(pts))
    if kind in ("slit", "fatigue"):
        path = C.crack_path(d)
        sw = max(float(p["width_mm"]), 0.12)
        if kind == "slit":
            els = [f'<line x1="{_f(path[0][0])}" y1="{_f(path[0][1])}" x2="{_f(path[1][0])}" y2="{_f(path[1][1])}" '
                   f'stroke="{INK["crack"]}" stroke-width="{_f(sw)}" marker-end="url(#arrow)"/>']
            label = f"{d['id']} slit {p['length_mm']:g}"
        else:
            els = [_poly(path, INK["crack"], "none", 0.25, closed=False, extra=' stroke-dasharray="0.6 0.4"')]
            label = f"{d['id']} crack ~{C.polyline_length(path):.0f}"
        return els, label, _around(path[-1], 0.8)
    if kind == "pitting":
        r = float(p["r_mm"])
        return ([_ellipse(p["u"], p["v"], r, r, 0, ink, 0.2).replace('fill="none"', 'fill="url(#hatch-corrosion)"')],
                f"{d['id']} pitting", _bbox(C.ellipse(p["u"], p["v"], r, r)))
    if kind == "galvanic":
        h = C.holes_by_id(c)[p["hole"]]
        ro, ri = C.galvanic_radius(c, p), C.fastener_radius(c, h)
        u, v = h["u"], h["v"]
        ring = (f"M{_f(u + ro)} {_f(v)}A{_f(ro)} {_f(ro)} 0 1 0 {_f(u - ro)} {_f(v)}A{_f(ro)} {_f(ro)} 0 1 0 {_f(u + ro)} {_f(v)}Z"
                f"M{_f(u + ri)} {_f(v)}A{_f(ri)} {_f(ri)} 0 1 0 {_f(u - ri)} {_f(v)}A{_f(ri)} {_f(ri)} 0 1 0 {_f(u + ri)} {_f(v)}Z")
        return ([f'<path d="{ring}" fill="url(#hatch-corrosion)" fill-rule="evenodd" stroke="{ink}" stroke-width="0.2"/>'],
                f"{d['id']} galvanic", _bbox(C.ellipse(u, v, ro, ro)))
    if kind == "fretting":
        pts = C.rect(*C.fretting_rect(c, p))
        return [_poly(pts, ink, "url(#hatch-corrosion)", 0.2)], f"{d['id']} fretting", _bbox(pts)
    raise ValueError(f"unknown defect {cls}.{kind}")


def _distractor_items(c: dict[str, Any], x: dict[str, Any]) -> tuple[list[str], str, tuple] | None:
    p, ink = x.get("params", {}), INK["distractor"]
    if x["kind"] == "burr":
        h = C.holes_by_id(c)[p["hole"]]
        a, b = C.hole_axes(h)
        w = float(p.get("w_mm", 0.4))
        el = _ellipse(h["u"], h["v"], a + w, b + w, h.get("angle_deg", 0.0), ink, 0.15, ' stroke-dasharray="0.3 0.3"')
        return [el], f"{x['id']} burr (leave)", _bbox(C.ellipse(h["u"], h["v"], a + w, b + w, h.get("angle_deg", 0.0)))
    if "u" in p and "v" in p:
        r = float(p.get("d_mm", 2 * float(p.get("r_mm", 1.0)))) / 2
        els = [_ellipse(p["u"], p["v"], r, r, 0, ink, 0.2, ' stroke-dasharray="1 0.6"'),
               _ellipse(p["u"], p["v"], r / 2, r / 2, 0, ink, 0.1, ' stroke-dasharray="0.5 0.5"')]
        label = f"{x['id']} {x['kind']}" + (" 20:1" if x["kind"] == "blend" else "")
        return els, label, _bbox(C.ellipse(p["u"], p["v"], r, r))
    return None


def _obstacles(c: dict[str, Any]) -> np.ndarray:
    W, H = C.plate_size(c)
    pts = [C.densify(np.array([[0, 0], [W, 0], [W, H], [0, H]], float), True, 1.0), C.datum_shape(c).pts]
    for k in (-1.5, -3.0, -4.5):        # ruler bands left of and above the plate
        pts.append(C.densify(np.array([[k, 0], [k, H]], float), False, 1.0))
        pts.append(C.densify(np.array([[0, k], [W + 8, k]], float), False, 1.0))
    for h in c["holes"]:
        a, b = C.hole_axes(h)
        pts.append(C.ellipse(h["u"], h["v"], a, b, h.get("angle_deg", 0.0)))
        rh = C.head_radius(c, h["id"])
        pts.append(np.array([[h["u"] - a - 2, h["v"]], [h["u"] + a + 2, h["v"]],
                             [h["u"], h["v"] - a - 2], [h["u"], h["v"] + a + 2]]))
        if rh:
            pts.append(C.ellipse(h["u"], h["v"], rh, rh))
    for d in c["defects"]:
        s = C.defect_shape(c, d)
        pts.append(s.pts)
    for x in c["distractors"]:
        s = C.distractor_shape(c, x)
        if s is not None:
            pts.append(s.pts)
    return np.vstack(pts)


def svg(c: dict[str, Any]) -> str:
    """The template for one coupon as an SVG string (1 user unit = 1 mm)."""
    W, H = C.plate_size(c)
    SW, SH = svg_size(c)
    t = float(c["plate"]["thickness_mm"])
    out = [f'<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(SW)}mm" height="{_f(SH)}mm" '
           f'viewBox="0 0 {_f(SW)} {_f(SH)}" font-family="{FONT}">',
           f"<title>{_esc(c['id'])} {_esc(c['name'])} - TriNetra coupon template (1:1)</title>",
           _defs(), f'<rect x="0" y="0" width="{_f(SW)}" height="{_f(SH)}" fill="#ffffff"/>']
    # title band (svg coordinates)
    out.append(_text(MARGIN_MM, 7.5, f"{c['id']}  {c['name']}", 4.2, weight="bold"))
    drills = sorted({float(h["d_mm"]) for h in c["holes"]})
    holes = (f"  |  drill Ø{'/'.join(f'{d:g}' for d in drills)} for Ø{c['fastener_d_mm']:g} fasteners" if drills else "")
    out.append(_text(MARGIN_MM, 12.0, f"role: {c['role']}  |  plate {W:g} x {H:g} x {t:g} mm{holes}  |  "
                     "TriNetra aircraft-representative coupon (commercial alloy)", 2.0))
    if c["role"] == "test":
        out.append(_text(SW - MARGIN_MM, 7.5, "HELD-OUT TEST PLATE", 3.0, fill="#c0392b", anchor="end", weight="bold"))

    g = [f'<g transform="translate({_f(MARGIN_MM)} {_f(TOP_MM)})">']
    # rulers
    for x in range(0, int(W) + 1, 5):
        L = 2.0 if x % 10 == 0 else 1.0
        g.append(f'<line x1="{x}" y1="-0.6" x2="{x}" y2="{_f(-0.6 - L)}" stroke="{INK["grid"]}" stroke-width="0.12"/>')
        if x % 10 == 0:
            g.append(_text(x, -3.6, str(x), 1.6, INK["grid"], "middle"))
    for y in range(0, int(H) + 1, 5):
        L = 2.0 if y % 10 == 0 else 1.0
        g.append(f'<line x1="-0.6" y1="{y}" x2="{_f(-0.6 - L)}" y2="{y}" stroke="{INK["grid"]}" stroke-width="0.12"/>')
        if y % 10 == 0:
            g.append(_text(-3.4, y + 0.55, str(y), 1.6, INK["grid"], "end"))
    g.append(_text(W + 1.5, -3.6, "u (mm)", 1.6, INK["grid"]))
    g.append(_text(-3.4, H + 4.0, "v (mm)", 1.6, INK["grid"], "end"))
    # plate and datum
    g.append(f'<rect x="0" y="0" width="{_f(W)}" height="{_f(H)}" fill="none" stroke="#000" stroke-width="0.35"/>')
    dm = c["datum"]
    s = float(dm.get("size_mm", 3.0))
    g.append(f'<rect x="{_f(dm["u"] - s / 2)}" y="{_f(dm["v"] - s / 2)}" width="{_f(s)}" height="{_f(s)}" '
             f'fill="none" stroke="#000" stroke-width="0.1" stroke-dasharray="0.4 0.3"/>')
    g.append(_text(dm["u"], dm["v"] + 1.1, dm.get("mark", "A"), 3.0, weight="bold", anchor="middle"))

    lab = _Labeler(_obstacles(c), (-MARGIN_MM + 1, -TOP_MM + 13, W + MARGIN_MM - 1, H + 3.5))
    labels: list[str] = []
    # fasteners
    for h in c["holes"]:
        a, b = C.hole_axes(h)
        rh = C.head_radius(c, h["id"])
        g.append(f"<g><title>{_esc(_hole_label(c, h))}</title>")
        g.append(_cross(h["u"], h["v"], a + 2.0))
        g.append(_ellipse(h["u"], h["v"], a, b, h.get("angle_deg", 0.0), INK["hole"], 0.2))
        if rh:
            g.append(_ellipse(h["u"], h["v"], rh, rh, 0, INK["head"], 0.12, ' stroke-dasharray="0.8 0.5"'))
        g.append("</g>")
        r = rh or a
        anchor = (h["u"] - r, h["v"] - r, h["u"] + r, h["v"] + r)
        x, y, box = lab.place(anchor, _hole_label(c, h))
        labels += [_leader(box, anchor, INK["hole"]), _text(x, y, _hole_label(c, h))]
    # defects and distractors
    for d in c["defects"]:
        els, label, anchor = _defect_items(c, d)
        tip = "; ".join(str(v) for v in (d.get("tool"), d.get("note")) if v)
        g.append(f"<g><title>{_esc(d['id'] + ' ' + d['cls'] + '.' + d['kind'] + (': ' + tip if tip else ''))}</title>")
        g += els
        g.append("</g>")
        x, y, box = lab.place(anchor, label)
        labels += [_leader(box, anchor, INK[d["cls"]]), _text(x, y, label, fill=INK[d["cls"]])]
    for xd in c["distractors"]:
        item = _distractor_items(c, xd)
        if item is None:
            continue
        els, label, anchor = item
        g += els
        x, y, box = lab.place(anchor, label)
        labels += [_leader(box, anchor, INK["distractor"]), _text(x, y, label, fill=INK["distractor"])]
    g += [x for x in labels if x]
    # scale bar
    yb = H + 7.0
    g.append(f'<path d="M0 {_f(yb)}H50" stroke="#000" stroke-width="0.4"/>')
    for x in range(0, 51, 10):
        g.append(f'<line x1="{x}" y1="{_f(yb - 1.2)}" x2="{x}" y2="{_f(yb + 1.2)}" stroke="#000" stroke-width="0.2"/>')
        g.append(_text(x, yb + 3.6, str(x), 1.8, anchor="middle"))
    g.append(_text(53, yb + 0.6, "Print at 100 % (actual size, no 'fit to page').", 2.2))
    g.append(_text(53, yb + 3.6, "Check: this bar must measure 50.0 mm.", 2.2, weight="bold"))
    g.append("</g>")
    out += g
    # notes
    y = TOP_MM + H + FOOT_MM + 1.0
    for line in _note_lines(c):
        weight = "bold" if line in ("Make:", "Designed findings:") else "normal"
        out.append(_text(8.0, y, line, NOTE_FONT, weight=weight))
        y += LINE_MM
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write_svg(c: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg(c), encoding="utf-8")
    return path


def _index_html(coupons: dict[str, dict[str, Any]]) -> str:
    rows, sections = [], []
    for k, c in coupons.items():
        W, H = C.plate_size(c)
        rows.append(f"<tr><td><a href='#{k}'>{k}</a></td><td>{_esc(c['name'])}</td><td>{c['role']}</td>"
                    f"<td>{W:g} x {H:g} x {c['plate']['thickness_mm']:g}</td><td>{len(c['holes'])}</td>"
                    f"<td>{len(c['defects'])}</td><td><a href='{k}.svg'>{k}.svg</a></td></tr>")
        steps = "".join(f"<li>{_esc(s)}</li>" for s in c["make"])
        exp = "".join(f"<li>{_esc(_expect_text(f))}</li>" for f in c["expect"]) or "<li>none (reference)</li>"
        sections.append(f"<section id='{k}'><h2>{k} {_esc(c['name'])} <small>({c['role']})</small></h2>"
                        f"<p><a href='{k}.svg'>Open {k}.svg to print</a></p>"
                        f"<img src='{k}.svg' alt='{k} template'><h3>Make</h3><ol>{steps}</ol>"
                        f"<h3>Designed findings</h3><ul>{exp}</ul></section>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>TriNetra coupon templates</title><style>"
            "body{font-family:Helvetica,Arial,sans-serif;max-width:900px;margin:0 auto;padding:16px;"
            "background:#fff;color:#111;line-height:1.45}table{border-collapse:collapse;width:100%}"
            "td,th{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left}"
            "img{max-width:100%;height:auto;border:1px solid #ccc}section{margin-top:32px}"
            ".warn{background:#fff4e5;border-left:4px solid #b35a00;padding:8px 12px}</style></head><body>"
            "<h1>TriNetra coupon templates</h1>"
            "<p class='warn'>Print each SVG on its own at <b>100 % / actual size</b> (no 'fit to page'), then check "
            "the 50 mm bar with a caliper before punching. Generated from configs/coupons.yaml by "
            "<code>python -m trinetra.twin.coupons templates</code>; see docs/twin/coupon_fabrication.md.</p>"
            "<table><tr><th>Face</th><th>Name</th><th>Role</th><th>Plate (mm)</th><th>Holes</th><th>Defects</th>"
            "<th>Template</th></tr>" + "".join(rows) + "</table>" + "".join(sections) + "</body></html>\n")


def write_all(coupons: dict[str, dict[str, Any]], out_dir: str | Path) -> list[Path]:
    """One SVG per coupon plus index.html; returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [write_svg(c, out / f"{k}.svg") for k, c in coupons.items()]
    idx = out / "index.html"
    idx.write_text(_index_html(coupons), encoding="utf-8")
    return [*paths, idx]
