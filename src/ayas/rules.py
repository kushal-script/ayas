"""Deterministic severity and verdict from the acceptance spec.

The LLM never decides pass/fail. It only explains what this module decided.
Three domains: the original station spec (spec.yaml), aerospace (spec_aero.yaml, via aero.py)
and steel and metal surfaces (spec_steel.yaml, judge_steel).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

# Worst last. The part's verdict is the worst verdict of its defects.
# HOLD_FOR_MRB (aerospace, Boeing Q219P): segregate and raise an NCR; the supplier may only
# rework to drawing or scrap, use-as-is and repair are decided by the customer MRB.
VERDICTS = ("PASS", "CLEAN_AND_REINSPECT", "REWORK", "HOLD_FOR_QA", "HOLD_FOR_MRB", "REJECT")
SEVERITIES = ("negligible", "minor", "major", "critical")


def load_spec(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def zone_for(spec: dict[str, Any], x_mm: float, y_mm: float) -> str:
    for zone in spec.get("zones", []):
        x, y, w, h = zone["rect_mm"]
        if x <= x_mm <= x + w and y <= y_mm <= y + h:
            return zone["name"]
    return "general"


def severity_for(value: float, limits: dict[str, float]) -> str:
    for level in ("critical", "major", "minor"):
        if value >= limits[level]:
            return level
    return "negligible"


def judge_defect(defect: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Add zone, severity, disposition, rule and action to one defect."""
    rules = spec["defects"]
    kind = defect["type"]
    zone = zone_for(spec, defect["location_mm"]["x"], defect["location_mm"]["y"])
    out = dict(defect, zone=zone)

    if kind == "unknown" or defect["confidence"] < spec.get("min_confidence", 0.6):
        rule = rules["unknown"]
        reason = "type not recognised" if kind == "unknown" else f"confidence {defect['confidence']:.2f} below {spec.get('min_confidence', 0.6)}"
        out.update(severity="unclassified", disposition="HOLD_FOR_QA", rule=f"{reason} -> hold for QA", action=rule["action"])
        return out

    rule = rules[kind]
    measure = rule["measure"]
    limits = rule.get("zone_limits", {}).get(zone, rule["limits"])
    value = float(defect["size"][measure])
    severity = severity_for(value, limits)
    if "disposition" in rule:
        disposition = rule["disposition"]
    elif severity == "critical":
        disposition = "REJECT"
    elif severity == "major":
        disposition = "REWORK" if rule.get("reworkable") else "REJECT"
    else:
        disposition = "PASS"
    threshold = limits[severity] if severity != "negligible" else limits["minor"]
    sign = ">=" if severity != "negligible" else "<"
    out.update(
        severity=severity,
        disposition=disposition,
        rule=f"{kind} {measure} {value:.2f} {sign} {threshold} ({severity}, zone {zone})",
        action=rule["action"] if disposition != "PASS" else "No action needed; logged for trend tracking.",
    )
    return out


def worst(dispositions) -> str:
    """The worst of some dispositions (PASS for none)."""
    return max(dispositions, key=VERDICTS.index, default="PASS")


def is_aero(spec: dict[str, Any]) -> bool:
    return spec.get("domain") == "aerospace"


def judge_part(defects: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if is_steel(spec):  # part material, thickness and size from the spec defaults
        return judge_steel(defects, spec, spec.get("part"))
    if is_aero(spec):  # surface defects only; fastener geometry needs aero.judge_aero
        from .aero import judge_aero

        verdict, judged, _ = judge_aero(defects, [], spec, spec["part"]["size_mm"])
        return verdict, judged
    judged = [judge_defect(d, spec) for d in defects]
    return worst(d["disposition"] for d in judged), judged


# ---- steel and metal surfaces (configs/spec_steel.yaml, docs/twin/DESIGN_STEEL.md F) ------------

STEEL_DOMAIN = "steel_surface"
STEEL_CLASSES = ("scratch", "dent", "pit", "corrosion", "crack", "stain")
# Worst last. RETAKE is not a disposition: the record sets it when the capture gate refuses the image.
STEEL_VERDICTS = ("PASS", "CLEAN_AND_REINSPECT", "REWORK", "HOLD_FOR_QA", "REJECT")
RECORD_VERDICTS = (*STEEL_VERDICTS, "RETAKE")
LEVELS = ("minor", "major", "critical")
FLAG_TEXT = {"touches_edge": "touches an edge or hole", "pitted": "pitted rust", "with_rust": "pits inside rust"}


def is_steel(spec: dict[str, Any]) -> bool:
    return spec.get("domain") == STEEL_DOMAIN


def _fmt(x: float) -> str:
    """A number as the rule strings show it: at most 2 decimals, no trailing zeros."""
    x = float(x)
    s = f"{x:.{3 if 0 < abs(x) < 0.1 else 2}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def steel_part(spec: dict[str, Any], part: dict[str, Any] | None = None) -> dict[str, Any]:
    """The part's material, thickness and size (spec defaults filled in), plus the material name and family."""
    base = dict(spec.get("part") or {})
    base.update({k: v for k, v in (part or {}).items() if v is not None})
    material = str(base.get("material") or "ms_hot_rolled")
    mat = (spec.get("materials") or {}).get(material, {})
    size = base.get("size_mm") or (100.0, 100.0)
    return {"material": material, "material_name": mat.get("name", material), "family": mat.get("family", "unknown"),
            "thickness_mm": float(base.get("thickness_mm") or 3.0), "size_mm": [float(size[0]), float(size[1])]}


def depth_context(spec: dict[str, Any], part: dict[str, Any]) -> dict[str, str]:
    """Placeholders for the action texts and references, from the plate thickness (depth_targets)."""
    dt = spec.get("depth_targets") or {}
    t = float(part["thickness_mm"])
    band = next((b for b in dt.get("bands", []) if b["t_mm"][0] <= t < b["t_mm"][1]), None)
    ctx = {"t": _fmt(t), "third_t": _fmt(t / 3.0)}
    if band is None:
        ref = dt.get("outside_reference", "")
        ctx.update(
            depth_rule=f"This {_fmt(t)} mm part is outside the plate depth tables: use the depth limit agreed with the customer.",
            grind_rule="stay within the thickness tolerance agreed with the customer",
            dent_rule="Compare the gap with the depth limit agreed with the customer.",
            t1=ref, t2=ref, t3=ref)
        return ctx
    lo, hi = (_fmt(x) for x in band["t_mm"])
    imp, unrep, grind = (_fmt(band[k]) for k in ("imperfection_mm", "unrepaired_mm", "grind_mm"))
    grind_rule = f"grind no deeper than {grind} mm below the minimum thickness"
    is2062 = dt.get("is2062") or {}
    if part["material"] in is2062.get("materials", []):
        pct = float(is2062.get("grind_pct", 4))
        grind_rule += f", and lose at most {_fmt(pct)} % of the thickness here ({_fmt(t * pct / 100)} mm of {_fmt(t)} mm)"
    ctx.update(
        depth_rule=(f"On {lo}-{hi} mm plate up to {imp} mm deep may stay; {imp}-{unrep} mm only if the affected "
                    f"area is 5 % or less; deeper must be repaired."),
        grind_rule=grind_rule,
        dent_rule=f"It may stay if the gap is {unrep} mm or less.",
        t1=band.get("t1", ""), t2=band.get("t2", ""), t3=band.get("t3", ""))
    return ctx


def _measures(d: dict[str, Any], part: dict[str, Any]) -> dict[str, float]:
    """Size measures plus derived ones: count (default 1), area_pct of the plate face, pit_d_mm."""
    size = d.get("size") or {}
    m = {k: float(v) for k, v in size.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    m["count"] = float(max(1, int(round(m.get("count", 1) or 1))))
    area = m.get("area_mm2") or 0.0
    if not area and m.get("diameter_mm"):
        area = math.pi * m["diameter_mm"] ** 2 / 4
    W, H = part["size_mm"]
    m["area_pct"] = 100.0 * area / (W * H) if W * H > 0 else 0.0
    if "pit_d_mm" not in m and m["count"] <= 1 and "diameter_mm" in m:
        m["pit_d_mm"] = m["diameter_mm"]
    return m


def _extent(d: dict[str, Any]) -> float:
    s = d.get("size") or {}
    return max(float(s.get("length_mm") or 0.0), float(s.get("diameter_mm") or 0.0))


def _gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Approximate edge-to-edge gap in mm (centre distance minus half extents, at least 0)."""
    la, lb = a.get("location_mm") or {}, b.get("location_mm") or {}
    if not la or not lb:
        return math.inf
    dist = math.hypot(float(la["x"]) - float(lb["x"]), float(la["y"]) - float(lb["y"]))
    return max(0.0, dist - (_extent(a) + _extent(b)) / 2)


def _clusters(idx: list[int], defects: list[dict[str, Any]], within: float) -> list[list[int]]:
    """Single-linkage groups of defects whose gap is below `within`."""
    groups: list[list[int]] = []
    for i in idx:
        joined = [g for g in groups if any(_gap(defects[i], defects[j]) < within for j in g)]
        merged = [i] + [j for g in joined for j in g]
        groups = [g for g in groups if g not in joined] + [sorted(merged)]
    return groups


def steel_context(defects: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Per defect: cluster membership (ISO 7788 40 mm rule) and pits-in-rust pairing."""
    mods = spec.get("modifiers") or {}
    within = float((mods.get("cluster") or {}).get("within_mm", 40.0))
    pairing = float(mods.get("pairing_mm", 1.0))
    ctx: list[dict[str, Any]] = [{} for _ in defects]
    for cls, rule in (spec.get("classes") or {}).items():
        if not rule.get("cluster_min_count"):
            continue
        idx = [i for i, d in enumerate(defects) if d.get("type") == cls]
        for g in _clusters(idx, defects, within):
            if len(g) < 2:
                continue
            n = sum(max(1, int((defects[i].get("size") or {}).get("count") or 1)) for i in g)
            if n >= int(rule["cluster_min_count"]):
                for i in g:
                    ctx[i]["cluster"] = {"count": n, "with": [defects[j].get("id") for j in g if j != i]}
    for i, a in enumerate(defects):
        for j, b in enumerate(defects):
            if a.get("type") == "pit" and b.get("type") == "corrosion" and _gap(a, b) <= pairing:
                ctx[i]["with_rust"] = True
                ctx[j]["pitted"] = True
    return ctx


def _band(cls: str, rule: dict[str, Any], m: dict[str, float], flags: dict[str, bool]) -> tuple[str, str]:
    """(severity, why) from the class bands: the worst band any measure or flag reaches, else minor."""
    if rule.get("always"):
        shown = ", ".join(f"{k} {_fmt(m[k])}" for k in rule.get("measures", []) if k in m)
        return rule["always"], f"any {cls} is a {rule['always']} finding regardless of size" + (f" ({shown})" if shown else "")
    for level in ("critical", "major"):
        hits = []
        for key, thr in (rule.get(level) or {}).items():
            if isinstance(thr, bool):
                if thr and flags.get(key):
                    hits.append(FLAG_TEXT.get(key, key))
            elif m.get(key) is not None and m[key] >= float(thr):
                hits.append(f"{key} {_fmt(m[key])} >= {_fmt(thr)}")
        if hits:
            return level, ", ".join(hits)
    major = rule.get("major") or {}
    below = [f"{k} {_fmt(m[k])} < {_fmt(major[k])}" for k in rule.get("measures", [])
             if k in m and not isinstance(major.get(k, True), bool)]
    return "minor", ", ".join(below) or "below the major band"


def _resolve_refs(refs: list[str], dctx: dict[str, str]) -> list[str]:
    out = [r.format_map(dctx) if "{" in r else r for r in refs]
    return [r for r in dict.fromkeys(out) if r]


def judge_steel_defect(d: dict[str, Any], spec: dict[str, Any], part: dict[str, Any],
                       ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    """Severity, tier, disposition, rule, reference and action for one steel defect (record 3.0 fields)."""
    ctx = ctx or {}
    part = part if part and "family" in part else steel_part(spec, part)
    classes = spec.get("classes") or {}
    mods = spec.get("modifiers") or {}
    tiers = spec.get("tiers") or {}
    kind = d.get("type")
    out = dict(d)
    out.setdefault("box_px", None)
    out.setdefault("near_edge_mm", None)
    if kind not in classes or kind == "unknown":
        rule = classes.get("unknown") or {}
        out.update(severity="unclassified", tier="Unclassified", disposition="HOLD_FOR_QA",
                   rule=f"type {kind!r} not recognised -> HOLD_FOR_QA", rule_id=rule.get("code", "UNK"),
                   reference="", action=(rule.get("action") or {}).get("unclassified", "Put the part in the HOLD tray and call QA."),
                   notes=[])
        return out
    rule = classes[kind]
    dctx = depth_context(spec, part)
    m = _measures(d, part)
    near = d.get("near_edge_mm")
    edge = mods.get("edge") or {}
    flags = {"touches_edge": near is not None and float(near) <= float(edge.get("touch_mm", 0.5)),
             "pitted": bool(d.get("pitted")) or bool(ctx.get("pitted")),
             "with_rust": bool(ctx.get("with_rust"))}
    severity, why = _band(kind, rule, m, flags)
    steps = [f"{kind}: {why} -> {severity}"]
    suffix: list[str] = []
    refs = list(rule.get("reference") or [])

    cluster = ctx.get("cluster")
    if cluster:
        cl = mods.get("cluster") or {}
        floor = cl.get("severity", "major")
        steps.append(f"{cluster['count']} {kind} finding(s) closer than {_fmt(cl.get('within_mm', 40))} mm (cluster)"
                     + (f" -> {floor}" if LEVELS.index(severity) < LEVELS.index(floor) else ""))
        severity = max(severity, floor, key=LEVELS.index)
        suffix.append("CLU")
        if cl.get("reference"):
            refs.append(cl["reference"])
    cap = rule.get("max_severity") or LEVELS[-1]
    if near is not None and float(near) <= float(edge.get("within_mm", 5.0)) and severity != "critical":
        up = LEVELS[min(LEVELS.index(cap), LEVELS.index(severity) + int(edge.get("levels", 1)))]
        where = f"{_fmt(near)} mm from an edge or hole (within {_fmt(edge.get('within_mm', 5.0))} mm: +1)"
        steps.append(f"{where} -> {up}" if up != severity else f"{where}: {kind} stays {severity} (capped)")
        if up != severity:
            suffix.append("EDGE")
        severity = up
    if LEVELS.index(severity) > LEVELS.index(cap):
        severity = cap

    disposition = rule["disposition"][severity]
    family = part.get("family")
    white = family in (rule.get("white_families") or [])
    actions = rule.get("action_white" if white else "action") or {}
    action = str(actions.get(severity, "")).format_map(dctx)
    if white:
        refs = list(rule.get("reference_white") or [])
    if kind == "corrosion" and family == "carbon_steel" and rule.get("rust_grade"):
        rg = rule["rust_grade"]
        grade = rg.get("pitted", "D") if flags["pitted"] else next(
            (g for thr, g in rg.get("bands", []) if m["area_pct"] >= float(thr)), "A")
        out["est_rust_grade"] = grade
        steps.append(f"estimated rust grade {grade} (visual analogue)")

    notes: list[str] = []
    depth = mods.get("depth_not_measured") or {}
    if kind in depth.get("classes", []) and severity != "minor":
        notes.append(depth.get("note", "depth not measured"))
        target = depth.get("disposition", "HOLD_FOR_QA")
        if VERDICTS.index(disposition) < VERDICTS.index(target):
            steps.append(f"depth not measured -> {target}")
            disposition = target
            suffix.append("DEPTH")

    low = mods.get("low_confidence") or {}
    below = float(low.get("below", spec.get("min_confidence", 0.5)))
    conf = float(d.get("confidence", 1.0))
    if conf < below:
        status = low.get("status", "uncertain, operator to confirm")
        out["status"] = status
        suffix.append("LOWCONF")
        confirm = low.get("confirm") or {}
        if kind in low.get("keep_disposition", []) and confirm and d.get("confirmed") is False:
            # P1h: not seen in the second burst -> capped (still held, never dropped)
            target = confirm.get("unconfirmed_disposition", "HOLD_FOR_QA")
            out["status"] = confirm.get("status", status)
            suffix.append("UNCONF")
            if VERDICTS.index(disposition) > VERDICTS.index(target):
                steps.append(f"confidence {_fmt(conf)} < {_fmt(below)}, not in the second burst -> {target}")
                disposition = target
            else:
                steps.append(f"confidence {_fmt(conf)} < {_fmt(below)}, not in the second burst ({kind} keeps {disposition})")
        elif kind in low.get("keep_disposition", []):
            confirmed = " and confirmed by the second burst" if d.get("confirmed") is True else ""
            steps.append(f"confidence {_fmt(conf)} < {_fmt(below)}{confirmed}: {status} ({kind} keeps {disposition})")
        else:
            target = low.get("disposition", "HOLD_FOR_QA")
            steps.append(f"confidence {_fmt(conf)} < {_fmt(below)}: {status} -> {target}")
            disposition = target
            if low.get("action"):
                action = f"{low['action']} Rule action: {action}"

    if "area_pct" in rule.get("measures", []):
        out["area_pct"] = round(m["area_pct"], 3)
    if cluster:
        out["cluster"] = dict(cluster)
    out.update(severity=severity, tier=tiers.get(severity, severity), disposition=disposition,
               rule="; ".join(steps) + f" => {disposition}",
               rule_id=f"{rule.get('code', kind[:3].upper())}-{severity[:3].upper()}" + "".join(f"+{s}" for s in suffix),
               reference="; ".join(_resolve_refs(refs, dctx)), action=action, notes=notes)
    return out


def confirm_defects(defects: list[dict[str, Any]], partner: list[dict[str, Any]],
                    spec: dict[str, Any]) -> list[dict[str, Any]]:
    """P1h: mark each LOW-confidence defect confirmed (True) or not (False) by the second burst of the
    same inspection: a same-type defect within low_confidence.confirm.within_mm (location_mm, plate
    frame). Confident defects get no flag. Returns new dicts; the rules then cap an unconfirmed crack."""
    low = (spec.get("modifiers") or {}).get("low_confidence") or {}
    confirm = low.get("confirm") or {}
    below = float(low.get("below", spec.get("min_confidence", 0.5)))
    within = float(confirm.get("within_mm", 3.0))
    out = []
    for d in defects:
        e = dict(d)
        if confirm and float(d.get("confidence", 1.0)) < below:
            loc = d.get("location_mm") or {}
            x, y = float(loc.get("x", 0.0)), float(loc.get("y", 0.0))
            e["confirmed"] = any(
                p.get("type") == d.get("type")
                and math.hypot(float((p.get("location_mm") or {}).get("x", 1e9)) - x,
                               float((p.get("location_mm") or {}).get("y", 1e9)) - y) <= within
                for p in partner)
        out.append(e)
    return out


def apply_device_policy(verdict: str, device_mode: str | None, spec: dict[str, Any]) -> tuple[str, str | None]:
    """Hood-only certification (adopted 22 Nov 2026): in a device mode listed in spec['device_policy']
    an accepting verdict (PASS, CLEAN_AND_REINSPECT) becomes HOLD_FOR_QA with a note. Returns (verdict, note)."""
    pol = (spec.get("device_policy") or {}).get(str(device_mode or ""))
    if not pol or verdict not in pol.get("no_accept", []):
        return verdict, None
    return pol.get("becomes", "HOLD_FOR_QA"), pol.get("note")


def judge_steel(defects: list[dict[str, Any]], spec: dict[str, Any],
                part: dict[str, Any] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """(verdict, judged defects) for a steel or metal part; verdict = worst disposition (PASS for none).

    part = {'material', 'thickness_mm', 'size_mm'}; missing keys come from spec['part']. Defects use
    the record 3.0 fields (type, confidence, box_px, location_mm, size in mm, near_edge_mm).
    """
    p = steel_part(spec, part)
    defects = [dict(d, id=d.get("id") or f"D{k}") for k, d in enumerate(defects, 1)]
    ctx = steel_context(defects, spec)
    judged = [judge_steel_defect(d, spec, p, c) for d, c in zip(defects, ctx)]
    return worst(d["disposition"] for d in judged), judged
