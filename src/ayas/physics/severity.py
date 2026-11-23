"""Depth-aware severity: the proposed rules change, as code (docs/twin/PHYSICS_PINN.md 5.1).

Today rules.judge_steel sends every dent, pit or scratch above the minor band to HOLD_FOR_QA
with "depth not measured" (spec_steel.yaml modifiers.depth_not_measured). With a measured depth
d and a ~90 % half-width ci (depth_pinn.DepthResult), this module judges the depth against the
standard tables already in configs/spec_steel.yaml depth_targets (EN 10163-2 / ISO 7788, by the
plate thickness t):

    d <= T1 (imperfection_mm)                       minor  (Table 1: may stay, any number)
    T1 < d <= T2 (unrepaired_mm), area <= 5 %       minor  (Table 2: may stay unrepaired)
    T1 < d <= T2, area > 5 %   or   d > T2          major  (repair)
    d > t / 3 (pits, IACS Rec. 47) or d > 0.2 t     critical (IS 2062 cl. 7: weld repair limit 20 %)

and uses it only when the estimate is trustworthy: converged, ci <= ci_max_mm, and [d - ci,
d + ci] inside one band. Otherwise today's behaviour stands (HOLD_FOR_QA, "depth not measured"
or "depth uncertain"). Depth governs dents and pits (EN 10163-2 grades them by depth); for
scratches the worse of the size band (paint/cosmetic) and the depth band is kept.
Edge, cluster and confidence modifiers are applied by rules.judge_steel as today.
This module never edits configs or rules.py; it wraps rules.judge_steel_defect.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

LEVELS = ("minor", "major", "critical")
DEPTH_CLASSES = ("dent", "pit", "scratch")
CI_MAX_MM = 0.08            # narrow enough to separate the 0.2 / 0.3 / 0.4 / 0.5 mm table limits


def depth_band(spec: Mapping[str, Any], thickness_mm: float) -> dict | None:
    """The depth_targets band for this thickness ({t_mm, imperfection_mm, unrepaired_mm, grind_mm, t1..t3}) or None."""
    dt = spec.get("depth_targets") or {}
    t = float(thickness_mm)
    return next((dict(b) for b in dt.get("bands", []) if b["t_mm"][0] <= t < b["t_mm"][1]), None)


def depth_severity(depth_mm: float, thickness_mm: float, area_pct: float, cls: str,
                   spec: Mapping[str, Any]) -> tuple[str, str, str]:
    """(severity, why, reference) of a depth by the EN 10163-2 / ISO 7788 tables (see module doc).

    Outside the plate tables (t < 3 mm or >= 25 mm, IS 1079 / IS 2062 by agreement) the depth
    is still compared with t/3 and 0.2 t; below those it is major (QA applies the agreed limit)."""
    t = float(thickness_mm)
    d = float(depth_mm)
    band = depth_band(spec, t)
    crit = t / 3.0 if cls == "pit" else 0.2 * t
    crit_ref = "IACS Rec. 47 cl. 6.7 (pits < 1/3 t)" if cls == "pit" else "IS 2062 (Part 1):2025 cl. 7 (4 % / 7 % / 20 %)"
    if d > crit:
        return "critical", f"depth {d:.2f} mm > {crit:.2f} mm ({'t/3' if cls == 'pit' else '20 % of t'})", crit_ref
    if band is None:
        ref = (spec.get("depth_targets") or {}).get("outside_reference", "")
        return "major", f"depth {d:.2f} mm on {t:g} mm stock: limit by agreement", ref
    t1, t2 = float(band["imperfection_mm"]), float(band["unrepaired_mm"])
    if d <= t1:
        return "minor", f"depth {d:.2f} mm <= {t1:g} mm (Table 1)", band.get("t1", "")
    if d <= t2 and area_pct <= 5.0:
        return "minor", f"depth {d:.2f} mm <= {t2:g} mm, area {area_pct:.2f} % <= 5 % (Table 2)", band.get("t2", "")
    return "major", f"depth {d:.2f} mm > {t1 if d <= t2 else t2:g} mm (repair)", band.get("t2", "")


def _area_pct(defect: Mapping[str, Any], part: Mapping[str, Any]) -> float:
    from ayas import rules
    return float(rules._measures(dict(defect), dict(part))["area_pct"])


def judge_with_depth(defect: Mapping[str, Any], spec: Mapping[str, Any], part: Mapping[str, Any] | None = None,
                     depth: Mapping[str, Any] | None = None, ctx: Mapping[str, Any] | None = None,
                     ci_max_mm: float = CI_MAX_MM) -> dict:
    """rules.judge_steel_defect with a measured depth ({'max_depth_mm', 'depth_ci_mm', 'converged'}).

    Returns the judged defect plus 'depth' = {'used': bool, 'severity', 'why', 'reason'}."""
    from ayas import rules
    p = rules.steel_part(dict(spec), dict(part or {}))
    kind = defect.get("type")
    base = rules.judge_steel_defect(dict(defect), dict(spec), p, dict(ctx or {}))
    if kind not in DEPTH_CLASSES or depth is None or depth.get("max_depth_mm") is None:
        base["depth"] = {"used": False, "reason": "not measured"}
        return base
    d = float(depth["max_depth_mm"])
    ci = float(depth.get("depth_ci_mm") or 0.0)
    t = float(p["thickness_mm"])
    area = _area_pct(defect, p)
    lo_sev, _, _ = depth_severity(max(0.0, d - ci), t, area, kind, spec)
    hi_sev, _, ref = depth_severity(d + ci, t, area, kind, spec)
    _, why, _ = depth_severity(d, t, area, kind, spec)
    why = why.replace(f"depth {d:.2f} mm", f"depth {d:.2f} +- {ci:.2f} mm", 1)
    usable = bool(depth.get("converged", False)) and ci <= ci_max_mm and lo_sev == hi_sev
    if not usable:
        reason = ("fit not converged" if not depth.get("converged", False) else
                  f"ci {ci:.2f} mm > {ci_max_mm:.2f} mm" if ci > ci_max_mm else
                  f"{d - ci:.2f}-{d + ci:.2f} mm straddles a limit")
        base["depth"] = {"used": False, "reason": reason, "max_depth_mm": round(d, 3), "depth_ci_mm": round(ci, 3)}
        if kind in DEPTH_CLASSES and base["severity"] != "minor":
            base["notes"] = list(base.get("notes") or []) + [f"depth uncertain ({d:.2f} +- {ci:.2f} mm)"]
        return base
    # judge again without the 'depth not measured' cap, with the depth band as the severity
    sp = copy.deepcopy(dict(spec))
    mods = sp.setdefault("modifiers", {})
    dn = dict(mods.get("depth_not_measured") or {})
    dn["classes"] = [c for c in dn.get("classes", []) if c != kind]
    mods["depth_not_measured"] = dn
    rule = sp["classes"][kind]
    sev_size = base["severity"]
    sev_depth = hi_sev
    if kind == "scratch":
        target = max(sev_size, sev_depth, key=LEVELS.index)
    else:
        target = sev_depth
    # force the band: replace the class bands by one that yields `target` before modifiers
    rule = dict(rule)
    rule.pop("major", None)
    rule.pop("critical", None)
    if target != "minor":
        rule["always"] = target
    sp["classes"][kind] = rule
    out = rules.judge_steel_defect(dict(defect), sp, p, dict(ctx or {}))
    shown = f"{why}" + (f" (size band: {sev_size})" if kind == "scratch" else "")
    out["rule"] = (out["rule"].replace(f"any {kind} is a {target} finding regardless of size", shown)
                   .replace("below the major band", shown))
    out["reference"] = "; ".join(x for x in [ref, out.get("reference", "")] if x)
    out["depth"] = {"used": True, "severity": sev_depth, "why": why, "max_depth_mm": round(d, 3),
                    "depth_ci_mm": round(ci, 3), "size_severity": sev_size}
    return out
