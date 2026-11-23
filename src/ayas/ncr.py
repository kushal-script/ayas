"""Draft nonconformance reports (NCR) from an aerospace inspection report.

The draft has AS9102-style characteristic accountability lines (characteristic, requirement,
actual, result), a nonconformance description, containment steps and the disposition options
of Boeing Q219P: the supplier may rework to drawing or scrap; use-as-is and repair belong to
the customer MRB. It is a DRAFT for the qualified inspector, never an approval, and the
requirements are the illustrative public-source demo limits of configs/spec_aero.yaml.
"""

from __future__ import annotations

import math
from typing import Any

STATUS = "DRAFT for the qualified inspector - not an approval"

RESULT = {
    "PASS": "CONFORMING (logged)",
    "CLEAN_AND_REINSPECT": "RE-INSPECT after cleaning",
    "REWORK": "NONCONFORMING - rework to drawing",
    "HOLD_FOR_QA": "UNDETERMINED - QA review",
    "HOLD_FOR_MRB": "NONCONFORMING - MRB",
    "REJECT": "NONCONFORMING - scrap recommended",
}
RECOMMENDED = {
    "PASS": "none (no nonconformance found)",
    "CLEAN_AND_REINSPECT": "clean and re-inspect",
    "REWORK": "REWORK to drawing, then re-inspect",
    "HOLD_FOR_QA": "QA review of the unidentified indication(s)",
    "HOLD_FOR_MRB": "MRB: supplier may rework to drawing or scrap; otherwise customer MRB",
    "REJECT": "SCRAP",
}
DISPOSITION_OPTIONS = {"supplier": ["REWORK to drawing", "SCRAP"], "customer_mrb_only": ["USE-AS-IS", "REPAIR"]}
ALWAYS_REFS = ("Boeing Q219P", "AS9100D 8.7")


def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def _fastener_lines(report: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    fasteners = report.get("fasteners") or []
    if not fasteners:
        return []
    fr = spec["fastener_rules"]
    part = report["part"]
    D = float(part.get("fastener_d_mm") or spec["part"]["fastener_d_mm"])
    W, H = (float(x) for x in (part.get("size_mm") or spec["part"]["size_mm"]))
    found: dict[str, list[dict[str, Any]]] = {}
    for f in report.get("findings") or []:
        found.setdefault(f["type"], []).append(f)

    def line(name: str, req: str, actual: str, kinds: tuple[str, ...], review: tuple[str, ...] = ()) -> dict[str, Any]:
        bad = [f for k in kinds for f in found.get(k, [])]
        unsure = [f for k in review for f in found.get(k, [])]
        result = "NONCONFORMING" if bad else "UNDETERMINED - QA review" if unsure else "CONFORMING"
        return {"characteristic": name, "requirement": req, "actual": actual, "result": result,
                "refs": [f["id"] for f in bad + unsure]}

    k_e, k_p, k_t = fr["edge_distance_min_D"], fr["pitch_min_D"], fr.get("transverse_min_D", fr["pitch_min_D"])
    e, eid = min((min(f["u"], f["v"], W - f["u"], H - f["v"]), f["id"]) for f in fasteners)
    out = [line(f"Fastener edge distance, {len(fasteners)} fasteners",
                f">= {k_e}D = {_fmt(k_e * D)} mm, hole centre to plate edge (illustrative; AC 43.13-1B 4-57c)",
                f"min {_fmt(e)} mm ({e / D:.2f}D) at {eid}", ("edge_distance",))]
    if len(fasteners) > 1:
        p, pa, pb = min((math.hypot(a["u"] - b["u"], a["v"] - b["v"]), a["id"], b["id"])
                        for i, a in enumerate(fasteners) for b in fasteners[i + 1:])
        out.append(line("Fastener pitch, centre to centre",
                        f">= {k_p}D = {_fmt(k_p * D)} mm along a row, >= {k_t}D = {_fmt(k_t * D)} mm between rows "
                        f"(illustrative; AC 43.13-1B 4-57c)",
                        f"min {_fmt(p)} mm ({p / D:.2f}D) between {pa} and {pb}", ("pitch",)))
    rivets = sum(f["cls"] == "rivet" for f in fasteners)
    holes = [f for f in fasteners if f["cls"] == "open_hole"]
    out.append(line("Fastener presence", "every hole in a riveted row holds its fastener (drawing)",
                    f"{rivets} rivets, {len(holes)} open holes, {len(found.get('missing_fastener', []))} missing",
                    ("missing_fastener",), ("lone_open_hole",)))
    if holes:
        ov = max(f["ovality"] for f in holes)
        dm = max(f["d_mm"] * f["ovality"] for f in holes)
        out.append(line("Open-hole size and ovality",
                        f"ovality < {fr['ovality_max']} and diameter <= D + {fr['oversize_mm']} = "
                        f"{D + fr['oversize_mm']:.2f} mm (illustrative; AC 43.13-1B 4-57b)",
                        f"max ovality {ov:.2f}, max diameter {dm:.2f} mm", ("elongated_hole",)))
    return out


def _requirement(d: dict[str, Any], spec: dict[str, Any], D: float, t: float) -> str:
    rules = spec["defects"]
    kind = d["type"]
    if kind == "dent":
        lim = rules["dent"]["limits"]
        return (f"<= {lim['max_diameter_mm']} mm across, depth <= t/{lim['max_depth_t_div']} = "
                f"{t / lim['max_depth_t_div']:.2f} mm, >= {lim['min_fastener_gap_D']}D = "
                f"{lim['min_fastener_gap_D'] * D:.1f} mm from fastener edges, no crack within "
                f"{lim['clear_x_diameter']}x diameter (illustrative; Piper SL 1095A)")
    if kind == "scratch":
        lim = rules["scratch"]["limits"]
        return (f"within blend limits: length <= {lim['length_mm']} mm, width <= {lim['width_mm']} mm "
                f"(illustrative; AC 43.13-1B 6-115)")
    if kind == "crack":
        return "no crack-like indication (any indication rejects)"
    if kind == "corrosion":
        return "no active corrosion on a production part (AC 43.13-1B 6-113)"
    return "indication identified by the qualified inspector"


def _actual(d: dict[str, Any]) -> str:
    s = d.get("size", {})
    kind = d["type"]
    if kind == "dent":
        gap = d.get("fastener_gap_mm")
        depth = d.get("depth_mm")
        txt = f"{s.get('diameter_mm')} mm across"
        if gap is not None:
            txt += f", {gap} mm from fastener {d.get('nearest_fastener')} edge"
        txt += f", depth {abs(depth):.2f} mm" if depth is not None else ", depth not measured"
        return f"{txt} -> {d.get('dent_class', '?')}"
    if kind in ("scratch", "crack"):
        return f"{s.get('length_mm')} mm long, {s.get('width_mm')} mm wide"
    if kind == "corrosion":
        return f"{s.get('area_mm2')} mm2, {s.get('diameter_mm')} mm across"
    return f"unidentified mark, confidence {d.get('confidence', 0):.2f}"


def ncr_draft(report: dict[str, Any], spec: dict[str, Any], ncr_id: str, timestamp: str | None = None) -> dict[str, Any]:
    """A draft NCR (plain dict, JSON-ready) for one aerospace inspection report."""
    part = report["part"]
    D = float(part.get("fastener_d_mm") or spec["part"]["fastener_d_mm"])
    t = float(part.get("thickness_mm") or spec["part"].get("thickness_mm", 3.0))
    lines = _fastener_lines(report, spec)
    for d in report["defects"]:
        x, y = d["location_mm"]["x"], d["location_mm"]["y"]
        lines.append({"characteristic": f"Surface: {d['type']} {d['id']} at u={x}, v={y} mm",
                      "requirement": _requirement(d, spec, D, t), "actual": _actual(d),
                      "result": RESULT[d["disposition"]], "refs": [d["id"]]})
    for k, ln in enumerate(lines, 1):
        ln["no"] = k

    items = [it for it in [*(report.get("findings") or []), *report["defects"]] if it["disposition"] != "PASS"]
    nonconf = [{"ref": it["id"], "type": it["type"], "location_mm": it["location_mm"], "description": it["rule"],
                "disposition": it["disposition"], "action": it["action"], "reference": list(it.get("reference") or [])}
               for it in items]
    if nonconf:
        description = (f"{len(nonconf)} item(s) on part {report['part_id']} need a disposition against the demo "
                       f"acceptance table: "
                       + "; ".join(f"{n['ref']} {n['type'].replace('_', ' ')}" for n in nonconf) + ".")
    else:
        description = f"No nonconformance found on part {report['part_id']} against the demo acceptance table."

    containment = []
    if nonconf:
        containment += [
            "Segregate the part in the MRB area with a hold tag carrying this NCR number (Boeing Q219P).",
            "Do not ship or move the part on until the disposition is recorded.",
            "Check the other parts from the same lot and operation for the same condition.",
        ]
    if any(n["type"] == "crack" for n in nonconf):
        containment.append("Confirm the crack-like indication by eddy current by a qualified NAS 410 / EN 4179 "
                           "Level 2; never stop-drill a production part.")
    if any(n["disposition"] == "HOLD_FOR_QA" for n in nonconf):
        containment.append("Qualified inspector to identify the unrecognised indication(s).")
    if any(n["type"] in ("edge_distance", "pitch", "elongated_hole") for n in nonconf):
        containment.append("Check the drill template or jig used for this part before drilling more parts.")

    cited = [r for it in items for r in it.get("reference") or []]
    refs = spec.get("references", {})
    used = set(cited) | (set(ALWAYS_REFS) if nonconf else set())
    return {
        "ncr_id": ncr_id,
        "status": STATUS,
        "timestamp": timestamp or report.get("timestamp"),
        "part_id": report["part_id"],
        "part": {k: part[k] for k in ("name", "material") if k in part},
        "verdict": report["verdict"],
        "characteristics": lines,
        "nonconformance": {"description": description, "items": nonconf},
        "containment": containment,
        "disposition": {"recommended": RECOMMENDED[report["verdict"]], **DISPOSITION_OPTIONS,
                        "decision": None, "decided_by": None},
        "references": [{"id": k, "text": v} for k, v in refs.items() if k in used],
        "disclaimer": spec.get("disclaimer", ""),
    }


def _cell(x: Any) -> str:
    return str(x).replace("|", "/").replace("\n", " ")


def to_markdown(ncr: dict[str, Any]) -> str:
    """The NCR draft as Markdown (for the report folder or a printout)."""
    out = [f"# NCR {ncr['ncr_id']} (DRAFT)", "", f"**{ncr['status']}.**", "",
           f"- Part: {ncr['part_id']} - {ncr['part'].get('name', '')} ({ncr['part'].get('material', '')})",
           f"- Date: {ncr.get('timestamp') or '-'}", f"- Rules verdict: {ncr['verdict']}", "",
           "## Characteristic accountability (AS9102-style)", "",
           "| # | Characteristic | Requirement (illustrative) | Actual | Result |", "|---|---|---|---|---|"]
    for ln in ncr["characteristics"]:
        out.append(f"| {ln['no']} | {_cell(ln['characteristic'])} | {_cell(ln['requirement'])} | "
                   f"{_cell(ln['actual'])} | {_cell(ln['result'])} |")
    nc = ncr["nonconformance"]
    out += ["", "## Nonconformance", "", nc["description"], ""]
    for it in nc["items"]:
        refs = f" ({', '.join(it['reference'])})" if it["reference"] else ""
        out.append(f"- **{it['ref']}** {it['type'].replace('_', ' ')}: {it['description']}. "
                   f"Rules: {it['disposition']}. {it['action']}{refs}")
    out += ["", "## Containment", ""]
    out += [f"{k}. {c}" for k, c in enumerate(ncr["containment"], 1)] or ["None needed."]
    disp = ncr["disposition"]
    out += ["", "## Disposition", "", f"- Recommended by the rules: {disp['recommended']}",
            f"- Supplier options (Boeing Q219P): {', '.join(disp['supplier'])}",
            f"- Customer MRB only: {', '.join(disp['customer_mrb_only'])}",
            "- Decision: ____________________  by (qualified inspector / MRB): ____________________", ""]
    if ncr["references"]:
        out += ["## References", ""] + [f"- {r['id']}: {r['text']}" for r in ncr["references"]] + [""]
    out.append(f"_{ncr['disclaimer']}_")
    return "\n".join(out) + "\n"
