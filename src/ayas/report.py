"""The inspection JSON: the contract between the vision side and the LLM."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .classify import Prediction
from .config import VIEWS, Config
from .features import RegionFeatures

SCHEMA_ID = "ayas.inspection/2.0"
# 2.1 = 2.0 plus the additive aerospace fields (domain, fasteners, findings, ncr, references).
SCHEMA_ID_AERO = "ayas.inspection/2.1"
# 3.0 = the steel and metal surface hand-off record (docs/twin/DESIGN_STEEL.md F): one JSON with
# defect, severity, confidence and box, plus the capture-quality block. Not a superset of 2.x.
SCHEMA_ID_STEEL = "ayas.inspection/3.0"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"

RELIEF = {
    "scratch": "recessed",
    "crack": "recessed",
    "pit": "recessed",
    "dent": "recessed",
    "bump": "raised",
    "dust": "raised",
    "stain": "flat",
    "unknown": "unknown",
}

VERDICT_INSTRUCTION = {
    "PASS": "Part accepted. Move it to the OK bin.",
    "CLEAN_AND_REINSPECT": "Clean the part as described, then run the inspection again.",
    "REWORK": "Send the part to rework with the listed actions, then re-inspect.",
    "HOLD_FOR_QA": "Put the part in the HOLD tray and call the QA supervisor.",
    "HOLD_FOR_MRB": "Segregate the part in the MRB area and raise an NCR. Supplier options: rework to drawing or scrap. "
    "Use-as-is or repair only by the customer MRB.",
    "REJECT": "Put the part in the red REJECT bin and tag it with the defect IDs.",
}

# Steel mode (record 3.0). RETAKE: the capture gate refused the image, so nothing was judged.
VERDICT_INSTRUCTION_STEEL = {
    "PASS": "Move the part to the OK bin. Logged imperfections need no repair.",
    "CLEAN_AND_REINSPECT": "Clean the marked areas as listed, dry the part, then inspect it again.",
    "REWORK": "Send the part to rework (blend, grind or clean as listed), then inspect it again.",
    "HOLD_FOR_QA": "Put the part in the HOLD tray. QA checks the depth or confirms the finding with a gauge or dye penetrant.",
    "REJECT": "Put the part in the red REJECT bin, tag it with the defect IDs and inform QA.",
    "RETAKE": "No result: the capture check refused the image. Follow the capture message and inspect the part again.",
}

# Aerospace mode (Boeing Q219P): the assistant never says accept or release.
VERDICT_INSTRUCTION_AERO = {
    **VERDICT_INSTRUCTION,
    "PASS": "No nonconformance against the demo table. Log the result; the qualified inspector confirms against the drawing.",
    "REWORK": "Rework to the drawing with the listed actions (rework restores full conformance), then re-inspect.",
    "HOLD_FOR_QA": "Put the part in the HOLD tray for the qualified inspector.",
    "REJECT": "Scrap recommended. Segregate the part with a red tag and raise an NCR; the qualified inspector confirms.",
}

def defect_entry(f: RegionFeatures, pred: Prediction) -> dict[str, Any]:
    v = f.values
    contrast = {
        "top_rgb": [round(v[f"c_top_{c}"], 3) for c in "rgb"],
        "top_under_45deg_light": round(v["c_top_oblique_light"], 3),
        "top_under_15deg_light": round(v["c_top_grazing_light"], 3),
        "oblique45": round(v["c_oblique45"], 3),
        "grazing": round(v["c_grazing"], 3),
    }
    single = {k: abs(x) for k, x in contrast.items() if k != "top_rgb"}
    single["top"] = float(np.mean(np.abs(contrast["top_rgb"])))
    strongest = max(single, key=single.get)
    return {
        "id": f.region_id,
        "type": pred.label,
        "confidence": pred.confidence,
        "location_mm": {"x": round(f.x_mm, 1), "y": round(f.y_mm, 1)},
        "size": {
            "length_mm": round(v["length_mm"], 2),
            "width_mm": round(v["width_mm"], 2),
            "area_mm2": round(v["area_mm2"], 2),
            "diameter_mm": round(float(np.sqrt(4 * v["area_mm2"] / np.pi)), 2),
        },
        "orientation_deg": int(round(f.orientation_deg)),
        "relief": RELIEF[pred.label],
        "depth_mm": round(v["depth_mm"], 3) if f.depth_available else None,
        "evidence": {
            "contrast": contrast,
            "geometry_ratio": round(v["geometry_ratio"], 2),
            "color_shift": round(v["chroma_shift"], 3),
            "slope_change": round(v["slope_dev"], 3),
            "shading_polarity": round(v["polarity"], 2),
            "strongest": strongest,
            "basis": pred.basis,
        },
    }


def rig_info(cfg: Config, shape: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "views_elevation_deg": {v: cfg.elevation_deg(v) for v in VIEWS},
        "lights": [
            {
                "name": s.name,
                "elevation_deg": s.elevation_deg,
                "azimuth_deg": s.azimuth_deg,
                "pair": s.pair,
                "shape": s.shape,
            }
            for s in cfg.lights.values()
        ],
        "shape": shape or {"mode": "none"},
    }


def build_report(
    part_id: str,
    verdict: str,
    defects: list[dict[str, Any]],
    cfg: Config,
    spec: dict[str, Any],
    classifier_name: str,
    golden_count: int | None,
    timing_ms: dict[str, float],
    capture_dir: str | None = None,
    overlay_path: str | None = None,
    mode: str = "surface",
    shape: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for d in defects:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    # For REJECT / HOLD only the deciding defects matter; rework steps on a scrapped part are noise.
    deciding = (verdict,) if verdict in ("REJECT", "HOLD_FOR_QA") else ("REWORK", "CLEAN_AND_REINSPECT")
    actions = list(dict.fromkeys(d["action"] for d in defects if d["disposition"] in deciding))
    report = {
        "schema": SCHEMA_ID,
        "mode": mode,
        "part_id": part_id,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "part": {k: spec["part"][k] for k in ("name", "material")},
        "rig": rig_info(cfg, shape),
        "verdict": verdict,
        "verdict_instruction": VERDICT_INSTRUCTION[verdict],
        "summary": {"defect_count": len(defects), "by_type": by_type},
        "defects": defects,
        "actions": actions,
        "models": {"classifier": classifier_name, "golden_parts": golden_count},
        "timing_ms": {k: round(v, 1) for k, v in timing_ms.items()},
        "capture_dir": capture_dir,
        "overlay": overlay_path,
    }
    if extra:
        report.update(extra)
    return report


def handheld_rig(detector: str, mm_per_px: float = 0.08) -> dict[str, Any]:
    return {"kind": "handheld", "camera": "Raspberry Pi AI Camera (IMX500)", "detector": detector,
            "grid_mm_per_px": mm_per_px, "shape": {"mode": "none"}}


def _deciding_actions(verdict: str, items: list[dict[str, Any]]) -> list[str]:
    """For a reject or hold only the deciding items matter; rework steps on a scrapped part are noise."""
    if verdict in ("REJECT", "HOLD_FOR_QA", "HOLD_FOR_MRB"):
        deciding: tuple[str, ...] = (verdict,)
    elif verdict == "PASS":
        deciding = ("PASS",)   # logged items still carry a check (e.g. verify dent depth)
    else:
        deciding = ("REWORK", "CLEAN_AND_REINSPECT")
    return list(dict.fromkeys(i["action"] for i in items if i["disposition"] in deciding and i.get("action")))


def build_aero_report(
    part_id: str,
    defects_mm: list[dict[str, Any]],
    fasteners: list[Any],
    spec: dict[str, Any],
    plate_size_mm,
    timing: dict[str, float],
    *,
    detector: str = "yolov8n",
    D: float | None = None,
    t: float | None = None,
    rig: dict[str, Any] | None = None,
    ncr_id: str | None = None,
    timestamp: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aerospace inspection report (schema 2.1) from surface defects and fasteners in plate mm.

    defects_mm: unjudged defect dicts (id, type, confidence, location_mm, size, orientation_deg,
    ...; see aero.defects_from_dets), or raw detections, which are converted. fasteners:
    aero.Fastener or dicts. D and t override the spec's fastener diameter and thickness.
    A draft NCR is attached unless the verdict is PASS.
    """
    from .aero import as_fastener, defects_from_dets, judge_aero, references_used
    from .ncr import ncr_draft

    defects = [d for d in defects_mm if "type" in d]
    raw = [d for d in defects_mm if "type" not in d]
    if raw:
        defects += defects_from_dets(raw, start=len(defects) + 1)
    fs = [as_fastener(f) for f in fasteners]
    D = float(D or spec["part"]["fastener_d_mm"])
    t = float(t or spec["part"].get("thickness_mm", 3.0))
    verdict, judged, findings = judge_aero(defects, fs, spec, plate_size_mm, D, t)
    by_type: dict[str, int] = {}
    for d in judged:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    by_finding: dict[str, int] = {}
    for f in findings:
        by_finding[f["type"]] = by_finding.get(f["type"], 0) + 1
    items = [*findings, *judged]
    nonconforming = any(i["disposition"] != "PASS" for i in items)
    stamp = timestamp or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    report = {
        "schema": SCHEMA_ID_AERO,
        "domain": "aerospace",
        "mode": "surface",
        "part_id": part_id,
        "timestamp": stamp,
        "part": {**{k: spec["part"][k] for k in ("name", "material")},
                 "size_mm": [float(x) for x in plate_size_mm], "thickness_mm": t, "fastener_d_mm": D},
        "rig": rig or handheld_rig(detector),
        "verdict": verdict,
        "verdict_instruction": VERDICT_INSTRUCTION_AERO[verdict],
        "summary": {"defect_count": len(judged), "by_type": by_type,
                    "finding_count": len(findings), "by_finding": by_finding,
                    "fastener_count": {"rivet": sum(f.cls == "rivet" for f in fs),
                                       "open_hole": sum(f.cls == "open_hole" for f in fs)}},
        "defects": judged,
        "fasteners": [f.to_dict() for f in fs],
        "findings": findings,
        "actions": _deciding_actions(verdict, items),
        "references": references_used(items, spec, ("Boeing Q219P", "AS9100D 8.7") if nonconforming else ()),
        "out_of_scope": [f"{o['item']}: {o['method']}" for o in spec.get("out_of_scope", [])],
        "disclaimer": spec.get("disclaimer", ""),
        "models": {"classifier": detector, "golden_parts": None},
        "timing_ms": {k: round(v, 1) for k, v in timing.items()},
        "capture_dir": None,
        "overlay": None,
        "ncr": None,
    }
    if extra:
        report.update(extra)
    if verdict != "PASS":
        report["ncr"] = ncr_draft(report, spec, ncr_id or f"NCR-{part_id}-{stamp[:10]}", stamp)
    return report


# ---- steel record 3.0 -------------------------------------------------------------------------

CAPTURE_KEYS = ("mode", "device_mode", "r_ambient", "lux_est", "motion_px", "sat_pct", "blur", "gate", "gate_message")
GATES = ("PASS", "PASS_FLAGGED", "RETAKE")
_CAPTURE_ROUND = {"r_ambient": 3, "lux_est": 0, "motion_px": 2, "sat_pct": 3, "blur": 3}


def _num(x: Any, nd: int) -> float | None:
    """A finite rounded float, or None (JSON has no inf or NaN)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _jsonable(x: Any) -> Any:
    """numpy scalars and arrays, tuples and nested containers as plain JSON values."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _gate_parts(gate: Any) -> tuple[str, str, list[str]]:
    """(verdict, message, reasons) from a lightproc.GateResult, a dict or a plain string."""
    if gate is None:
        return "PASS", "", []
    if isinstance(gate, str):
        return gate, "", []
    if isinstance(gate, dict):
        return str(gate.get("verdict", "PASS")), str(gate.get("message") or ""), [str(r) for r in gate.get("reasons") or []]
    return (str(getattr(gate, "verdict", "PASS")), str(getattr(gate, "message", "") or ""),
            [str(r) for r in getattr(gate, "reasons", []) or []])


def capture_block(capture: dict[str, Any] | None) -> dict[str, Any]:
    """The record's capture-quality block with every contract key present (None when unknown).

    capture: mode (burst5 | naive), device_mode (open | hood), r_ambient, lux_est, motion_px,
    sat_pct, blur, gate (a verdict string, a GateResult or its dict), gate_message, optional
    gate_reasons; any other keys are kept as they are.
    """
    c = dict(capture or {})
    verdict, message, reasons = _gate_parts(c.get("gate"))
    if verdict not in GATES:
        raise ValueError(f"unknown capture gate {verdict!r}")
    out: dict[str, Any] = {"mode": c.get("mode") or "burst5", "device_mode": c.get("device_mode") or "open"}
    for k, nd in _CAPTURE_ROUND.items():
        out[k] = _num(c.get(k), nd)
    out["gate"] = verdict
    out["gate_message"] = str(c.get("gate_message") or message or "")
    reasons = [str(r) for r in c.get("gate_reasons") or reasons]
    if reasons:
        out["gate_reasons"] = reasons
    out.update({k: _jsonable(v) for k, v in c.items() if k not in out and k not in ("gate_reasons",)})
    return out


def capture_from_metrics(metrics: dict[str, Any], gate: Any, mode: str = "burst5", device_mode: str = "open",
                         lux_est: float | None = None, led_lux: float | None = None) -> dict[str, Any]:
    """Capture block from lightproc metrics and its GateResult. lux_est defaults to the metrics'
    lux_est, else r_ambient x led_lux (the LEDs' own illuminance at the plate) when led_lux is
    given. blur is blur_rel when known (1 = the material reference; the gate retakes below
    capture.yaml blur_rel_min)."""
    m = metrics or {}
    r = _num(m.get("r_ambient"), 6)
    if lux_est is None:
        lux_est = m.get("lux_est")
    if lux_est is None and led_lux is not None and r is not None:
        lux_est = r * float(led_lux)
    sat = m.get("masked_frac", m.get("sat_frac_a"))
    sat_pct = 100.0 * float(sat) if sat is not None else m.get("sat_pct")   # naive arm: sat_pct directly
    blur = m.get("blur_rel") if m.get("blur_rel") is not None else m.get("blur")
    return capture_block({"mode": mode, "device_mode": device_mode, "r_ambient": r, "lux_est": lux_est,
                          "motion_px": m.get("motion_px"), "sat_pct": sat_pct, "blur": blur, "gate": gate})


def _steel_defect_entry(d: dict[str, Any]) -> dict[str, Any]:
    """One judged defect in the record shape. A type the rules do not know (e.g. an old detector's
    'rivet') becomes 'unknown' (the schema's enum), keeping the raw label as detected_type; a
    negative edge distance (a box overlapping the plate edge) is 0."""
    from .rules import STEEL_CLASSES

    e = dict(d)
    if e.get("type") not in (*STEEL_CLASSES, "unknown"):
        e["detected_type"], e["type"] = str(e.get("type")), "unknown"
    box = e.get("box_px")
    e["box_px"] = None if box is None else [round(float(x), 1) for x in box]
    near = _num(e.get("near_edge_mm"), 2)
    e["near_edge_mm"] = None if near is None else max(0.0, near)
    e["confidence"] = round(float(e.get("confidence", 0.0)), 3)
    loc = e.get("location_mm") or {}
    e["location_mm"] = {"x": round(float(loc.get("x", 0.0)), 1), "y": round(float(loc.get("y", 0.0)), 1)}
    e["size"] = {k: (int(v) if k == "count" else round(float(v), 3)) for k, v in (e.get("size") or {}).items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return e


def build_steel_record(part_id: str, part: dict[str, Any], defects_judged: list[dict[str, Any]], verdict: str,
                       capture: dict[str, Any] | None, timing_ms: dict[str, float] | None,
                       backend: dict[str, Any] | None, *, spec: dict[str, Any] | None = None,
                       timestamp: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The steel hand-off record (schema ayas.inspection/3.0, DESIGN_STEEL.md F).

    defects_judged and verdict come from rules.judge_steel. A RETAKE capture gate makes the verdict
    RETAKE and drops every defect: the record says it refused, it never fails silently. spec (the
    steel spec) adds the material name, the reference texts and the disclaimer. backend =
    {'vision': ..., 'advisor': ...}; the app fills 'advisor' once the advisor has answered.
    """
    from .rules import RECORD_VERDICTS, steel_part

    p = steel_part(spec or {}, part)
    cap = capture_block(capture)
    defects = [_steel_defect_entry(d) for d in defects_judged]
    if cap["gate"] == "RETAKE":
        verdict, defects = "RETAKE", []
    elif verdict == "RETAKE":
        raise ValueError("verdict RETAKE needs a RETAKE capture gate")
    if verdict not in RECORD_VERDICTS:
        raise ValueError(f"unknown steel verdict {verdict!r}")
    by_type: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for d in defects:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
        by_sev[d["severity"]] = by_sev.get(d["severity"], 0) + 1
    timing = {k: round(float(v), 1) for k, v in (timing_ms or {}).items()}
    if timing and "total" not in timing:
        timing["total"] = round(sum(timing.values()), 1)
    ref_text = (spec or {}).get("references") or {}
    used = list(dict.fromkeys(r for d in defects for r in split_refs(d.get("reference"))))
    be = dict(backend or {})
    record = {
        "schema": SCHEMA_ID_STEEL,
        "domain": "steel_surface",
        "part_id": str(part_id),
        "timestamp": timestamp or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "material": p["material"],
        "material_name": p["material_name"],
        "thickness_mm": round(p["thickness_mm"], 3),
        "size_mm": [round(x, 3) for x in p["size_mm"]],
        "verdict": verdict,
        "verdict_instruction": VERDICT_INSTRUCTION_STEEL[verdict],
        "capture": cap,
        "defects": defects,
        "summary": {"defect_count": len(defects), "by_type": by_type, "by_severity": by_sev},
        "actions": _deciding_actions(verdict, defects),
        "references": [{"id": r, "text": str(ref_text.get(r, ""))} for r in used],
        "disclaimer": str((spec or {}).get("disclaimer", "")),
        "timing_ms": timing,
        "backend": {**be, "vision": str(be.get("vision") or ""), "advisor": be.get("advisor")},
    }
    if extra:
        record.update(extra)
    return record


def split_refs(reference: str | list[str] | None) -> list[str]:
    """Reference ids from a record field ('; '-joined string or list)."""
    if not reference:
        return []
    items = reference if isinstance(reference, list) else str(reference).split("; ")
    return [r.strip() for r in items if r and r.strip()]


def is_steel_record(report: dict[str, Any]) -> bool:
    return str(report.get("schema", "")).startswith(("ayas.inspection/3", "trinetra.inspection/3")) or report.get("domain") == "steel_surface"  # trinetra.*: records from before the rename


def schema_path(report: dict[str, Any]) -> Path:
    """The JSON schema file for a record: inspection.v3.schema.json for 3.x, else inspection.schema.json
    (which also accepts 3.0)."""
    return SCHEMA_DIR / ("inspection.v3.schema.json" if is_steel_record(report) else "inspection.schema.json")


def validate_record(report: dict[str, Any]) -> list[str]:
    """Schema problems of a record (empty = valid). Needs jsonschema; without it, returns []."""
    try:
        import jsonschema
    except ImportError:
        return []
    path = schema_path(report)
    if not path.exists():
        return [f"schema file not found: {path}"]
    v = jsonschema.Draft202012Validator(json.loads(path.read_text()))
    return [f"{'/'.join(str(x) for x in e.absolute_path) or '<root>'}: {e.message}"
            for e in v.iter_errors(json.loads(json.dumps(report)))]


LLM_MAX_ITEMS = 6   # the Pi's Llama context is 2048 tokens for system prompt, report, notes and answer
STEEL_LLM_MAX_ITEMS = 3   # steel items carry rule, reference and action text: about 200 tokens each


def llm_items(report: dict[str, Any], max_items: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(shown, omitted) items for the LLM: worst disposition first, findings before defects.
    max_items defaults to STEEL_LLM_MAX_ITEMS for a steel record, else LLM_MAX_ITEMS."""
    from .rules import VERDICTS

    if max_items is None:
        max_items = STEEL_LLM_MAX_ITEMS if is_steel_record(report) else LLM_MAX_ITEMS
    items = [*report.get("findings", []), *report["defects"]]
    order = sorted(range(len(items)), key=lambda k: (-VERDICTS.index(items[k]["disposition"]), k))
    shown = sorted(order[:max_items])
    return [items[k] for k in shown], [items[k] for k in order[max_items:]]


def _compact_aero(report: dict[str, Any]) -> str:
    """Aerospace items without sizes (their rule states the measures) or per-item references
    (the SOP notes list them once per type); at most LLM_MAX_ITEMS items, the rest counted."""
    keep = ("id", "type", "confidence", "location_mm", "depth_mm", "severity", "disposition", "rule", "dent_class")
    shown, omitted = llm_items(report)
    defects, findings = [], []
    for it in shown:
        if "fasteners" in it:
            findings.append({k: it[k] for k in ("id", "type", "fasteners", "severity", "disposition", "rule")})
        else:
            defects.append({k: it[k] for k in keep if it.get(k) is not None})   # the rule carries the sizes
    slim = {
        "domain": "aerospace",
        "part_id": report["part_id"],
        "part": {k: report["part"][k] for k in ("name", "thickness_mm", "fastener_d_mm") if k in report["part"]},
        "verdict": report["verdict"],
        "verdict_instruction": report["verdict_instruction"],
        "fasteners": report.get("summary", {}).get("fastener_count", {}),
        "defects": defects,
        "findings": findings,
        "actions": report.get("actions", []),
    }
    if omitted:
        by: dict[str, int] = {}
        for it in omitted:
            by[it["disposition"]] = by.get(it["disposition"], 0) + 1
        slim["not_shown"] = by
    return json.dumps(slim, separators=(",", ":"), ensure_ascii=False)


STEEL_LLM_KEYS = ("id", "type", "confidence", "severity", "tier", "disposition", "size", "location_mm",
                  "near_edge_mm", "rule", "reference", "action", "status", "est_rust_grade")
STEEL_LLM_CAPTURE = ("gate", "gate_message", "r_ambient", "lux_est", "motion_px", "sat_pct", "blur", "device_mode")


def _compact_steel(report: dict[str, Any]) -> str:
    """Record 3.0 for the LLM: at most LLM_MAX_ITEMS defects (worst first, the rest counted), zero
    sizes dropped, a repeated action text replaced by 'same as D<k>', and the capture quality."""
    shown, omitted = llm_items(report)
    seen: dict[str, str] = {}
    defects = []
    for d in shown:
        e: dict[str, Any] = {}
        for k in STEEL_LLM_KEYS:
            v = d.get(k)
            if k == "size" and isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if vv}
            if v in (None, "", [], {}):
                continue
            if k == "rule":
                v = str(v).rsplit(" => ", 1)[0]   # the disposition is its own field
            if k == "action" and d.get("status") and " Rule action: " in v:
                v = v.split(" Rule action: ", 1)[1]   # the status says it is uncertain
            if k == "action":
                if v in seen:
                    v = f"same as {seen[v]}"
                else:
                    seen[v] = d["id"]
            e[k] = v
        defects.append(e)
    cap = report.get("capture") or {}
    slim: dict[str, Any] = {
        "domain": "steel_surface",
        "part_id": report["part_id"],
        "material": report.get("material_name") or report.get("material"),
        "thickness_mm": report.get("thickness_mm"),
        "verdict": report["verdict"],
        "verdict_instruction": report["verdict_instruction"],
        "capture": {k: cap[k] for k in STEEL_LLM_CAPTURE if cap.get(k) not in (None, "")},
        "defects": defects,
    }
    if omitted:
        by: dict[str, int] = {}
        for it in omitted:
            by[it["disposition"]] = by.get(it["disposition"], 0) + 1
        slim["not_shown"] = by
    return json.dumps(slim, separators=(",", ":"), ensure_ascii=False)


def compact_for_llm(report: dict[str, Any]) -> str:
    """Only what the LLM needs, minified: on a CPU every prompt token costs latency."""
    if is_steel_record(report):
        return _compact_steel(report)
    if report.get("domain") == "aerospace":
        return _compact_aero(report)
    keep = (
        "id",
        "type",
        "confidence",
        "location_mm",
        "zone",
        "size",
        "relief",
        "depth_mm",
        "relevant",
        "severity",
        "disposition",
        "rule",
        "action",
    )
    defects = []
    for d in report["defects"]:
        e = {k: d[k] for k in keep if k in d and d[k] is not None}
        e["evidence"] = {k: val for k, val in d.get("evidence", {}).items() if k != "basis"}
        defects.append(e)
    slim = {
        "mode": report.get("mode", "surface"),
        "part_id": report["part_id"],
        "part": report["part"],
        "verdict": report["verdict"],
        "verdict_instruction": report["verdict_instruction"],
        "defects": defects,
    }
    if "penetrant" in report:
        p = report["penetrant"]
        slim["penetrant"] = {k: p[k] for k in ("profile", "relevant_mm", "process", "aligned_groups") if k in p}
    return json.dumps(slim, separators=(",", ":"), ensure_ascii=False)
