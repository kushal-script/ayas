"""Steel and metal surface severity rules (spec_steel.yaml, rules.judge_steel) and record 3.0."""

import json
from pathlib import Path

import pytest

from ayas.report import (SCHEMA_ID_STEEL, VERDICT_INSTRUCTION_STEEL, build_steel_record, capture_block,
                             capture_from_metrics, compact_for_llm, validate_record)
from ayas.rules import (STEEL_VERDICTS, VERDICTS, judge_part, judge_steel, judge_steel_defect, load_spec,
                            steel_part, worst)

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_spec(ROOT / "configs" / "spec_steel.yaml")
PART = {"material": "ms_hot_rolled", "thickness_mm": 3.0, "size_mm": [100.0, 100.0]}
V3 = ROOT / "schemas" / "inspection.v3.schema.json"
COMBINED = ROOT / "schemas" / "inspection.schema.json"


def d(kind, conf=0.9, x=50.0, y=50.0, near=30.0, did="D1", **size):
    return {"id": did, "type": kind, "confidence": conf, "box_px": [10.0, 10.0, 60.0, 40.0],
            "location_mm": {"x": x, "y": y}, "size": size, "near_edge_mm": near}


def one(defect, part=PART):
    verdict, judged = judge_steel([defect], SPEC, part)
    return verdict, judged[0]


def record(defects, gate="PASS", message="", part=PART, **cap):
    verdict, judged = judge_steel(defects, SPEC, part)
    capture = {"mode": "burst5", "device_mode": "open", "r_ambient": 0.12, "lux_est": 400, "motion_px": 0.3,
               "sat_pct": 0.0, "blur": 0.97, "gate": gate, "gate_message": message, **cap}
    return build_steel_record("P1", part, judged, verdict, capture,
                              {"capture": 410.0, "process": 95.0, "detect": 300.0, "rules": 1.0},
                              {"vision": "onnx:best.onnx", "advisor": None}, spec=SPEC,
                              timestamp="2026-11-14T10:00:00+05:30")


# ---- spec ----------------------------------------------------------------------------------------

def test_spec_shape():
    assert SPEC["domain"] == "steel_surface"
    assert set(("scratch", "dent", "pit", "corrosion", "crack", "stain")) <= set(SPEC["classes"])
    assert SPEC["tiers"] == {"minor": "Imperfection", "major": "Defect", "critical": "Critical defect"}
    assert "illustrative" in SPEC["disclaimer"].lower() and "does not measure depth" in SPEC["disclaimer"]
    for cls, rule in SPEC["classes"].items():
        for sev, disp in rule["disposition"].items():
            assert disp in STEEL_VERDICTS, (cls, sev)
            assert rule["action"][sev], (cls, sev)


def test_every_spec_reference_is_whitelisted():
    refs = set(SPEC["references"])
    used = [SPEC["modifiers"]["cluster"]["reference"], SPEC["depth_targets"]["outside_reference"],
            SPEC["depth_targets"]["is2062"]["reference"]]
    for b in SPEC["depth_targets"]["bands"]:
        used += [b["t1"], b["t2"], b["t3"]]
    for rule in SPEC["classes"].values():
        used += [r for r in [*rule.get("reference", []), *rule.get("reference_white", [])] if "{" not in r]
    assert set(used) <= refs


def test_spec_texts_contain_no_banned_phrase():
    from ayas.llm import _negated, _phrase_re

    texts = [SPEC["disclaimer"], *SPEC["references"].values(), *VERDICT_INSTRUCTION_STEEL.values(),
             SPEC["modifiers"]["low_confidence"]["action"], SPEC["capture_advice"]["PASS_FLAGGED"],
             SPEC["capture_advice"]["RETAKE"], *SPEC["capture_advice"]["reasons"].values()]
    for rule in SPEC["classes"].values():
        texts += [*rule["action"].values(), *rule.get("action_white", {}).values(), *rule.get("likely_causes", []),
                  rule.get("prevention", "")]
    for t in texts:
        for phrase in SPEC["banned_phrases"]:
            hits = [m for m in _phrase_re(phrase).finditer(t) if not _negated(t, m.start())]
            assert not hits, (phrase, t)


# ---- class rows ------------------------------------------------------------------------------------

@pytest.mark.parametrize("width, length, near, severity, disposition", [
    (0.1, 8.0, 30.0, "minor", "PASS"),
    (0.3, 8.0, 30.0, "major", "HOLD_FOR_QA"),       # REWORK row, but depth is not measured
    (0.1, 30.0, 30.0, "major", "HOLD_FOR_QA"),
    (0.6, 8.0, 30.0, "critical", "HOLD_FOR_QA"),
    (0.1, 8.0, 0.2, "critical", "HOLD_FOR_QA"),     # reaches an edge or hole
])
def test_scratch_row(width, length, near, severity, disposition):
    _, j = one(d("scratch", width_mm=width, length_mm=length, near=near))
    assert (j["severity"], j["disposition"]) == (severity, disposition)
    assert j["tier"] == SPEC["tiers"][severity]
    if severity != "minor":
        assert "depth not measured, operator check with a gauge" in j["notes"]


def test_scratch_major_row_without_depth_rule_is_rework():
    spec = json.loads(json.dumps(SPEC))
    spec["modifiers"]["depth_not_measured"]["classes"] = []
    _, j = judge_steel([d("scratch", width_mm=0.3, length_mm=8.0)], spec, PART)
    assert (j[0]["severity"], j[0]["disposition"]) == ("major", "REWORK")


@pytest.mark.parametrize("diameter, severity, disposition", [
    (3.0, "minor", "PASS"), (5.0, "major", "HOLD_FOR_QA"), (7.0, "critical", "REJECT")])
def test_dent_row(diameter, severity, disposition):
    _, j = one(d("dent", diameter_mm=diameter))
    assert (j["severity"], j["disposition"]) == (severity, disposition)


@pytest.mark.parametrize("size, severity, disposition", [
    ({"diameter_mm": 0.3, "count": 1}, "minor", "PASS"),
    ({"diameter_mm": 0.8, "count": 1}, "major", "HOLD_FOR_QA"),
    ({"diameter_mm": 4.0, "count": 8, "area_mm2": 12.0, "pit_d_mm": 0.4}, "major", "HOLD_FOR_QA"),
    ({"diameter_mm": 2.5, "count": 1}, "critical", "HOLD_FOR_QA"),
    ({"diameter_mm": 30.0, "count": 4, "area_mm2": 600.0, "pit_d_mm": 0.4}, "critical", "HOLD_FOR_QA"),
])
def test_pit_row(size, severity, disposition):
    _, j = one(d("pit", **size))
    assert (j["severity"], j["disposition"]) == (severity, disposition)


def test_pits_inside_rust_are_critical_and_rust_grade_d():
    pit = d("pit", did="D1", x=50.0, y=50.0, diameter_mm=0.3, count=1)
    rust = d("corrosion", did="D2", x=51.0, y=50.0, area_mm2=30.0, diameter_mm=6.2)
    _, (jp, jr) = judge_steel([pit, rust], SPEC, PART)
    assert jp["severity"] == "critical" and "pits inside rust" in jp["rule"]
    assert jr["severity"] == "critical" and jr["est_rust_grade"] == "D"


@pytest.mark.parametrize("area, severity, disposition, grade", [
    (50.0, "minor", "REWORK", "A"), (300.0, "major", "REWORK", "B"), (1000.0, "critical", "HOLD_FOR_QA", "C")])
def test_corrosion_row(area, severity, disposition, grade):
    _, j = one(d("corrosion", area_mm2=area))
    assert (j["severity"], j["disposition"], j["est_rust_grade"]) == (severity, disposition, grade)
    assert j["area_pct"] == pytest.approx(area / 100.0)
    assert "ISO 8501-1 / IS 9954 (rust grades A-D)" in j["reference"]


def test_pitted_rust_is_grade_d_critical():
    _, j = one(dict(d("corrosion", area_mm2=50.0), pitted=True))
    assert (j["severity"], j["est_rust_grade"]) == ("critical", "D")


@pytest.mark.parametrize("material", ["galvanised", "aluminium"])
def test_white_corrosion_has_no_rust_grade(material):
    _, j = one(d("corrosion", area_mm2=300.0), {**PART, "material": material})
    assert "est_rust_grade" not in j
    assert "white corrosion" in j["action"] and "ISO 8501-1" not in j["reference"]


def test_crack_row():
    _, j = one(d("crack", length_mm=3.0, width_mm=0.1))
    assert (j["severity"], j["tier"], j["disposition"]) == ("critical", "Critical defect", "REJECT")
    assert j["reference"] == "EN 10163-2 / ISO 7788 (cracks, shells, seams)"
    assert "dye penetrant" in j["action"]


@pytest.mark.parametrize("length", [0.05, 0.5, 3.0, 40.0])
@pytest.mark.parametrize("conf", [0.05, 0.3, 0.49, 0.5, 0.99])
@pytest.mark.parametrize("near", [None, 0.0, 3.0, 40.0])
def test_crack_is_never_accepted(length, conf, near):
    verdict, j = one(d("crack", conf=conf, near=near, length_mm=length, width_mm=0.05))
    assert verdict == "REJECT" and j["disposition"] == "REJECT" and j["severity"] == "critical"
    assert ("status" in j) == (conf < 0.5)


@pytest.mark.parametrize("area", [1.0, 50.0, 500.0, 5000.0])
@pytest.mark.parametrize("near", [None, 0.0, 2.0, 30.0])
@pytest.mark.parametrize("conf", [0.2, 0.9])
def test_stain_is_never_rejected(area, near, conf):
    verdict, j = one(d("stain", conf=conf, near=near, area_mm2=area))
    assert j["disposition"] != "REJECT" and verdict != "REJECT"
    assert j["severity"] in ("minor", "major")
    assert j["disposition"] == ("CLEAN_AND_REINSPECT" if conf >= 0.5 else "HOLD_FOR_QA")


# ---- modifiers -----------------------------------------------------------------------------------

@pytest.mark.parametrize("defect", [
    d("scratch", conf=0.3, width_mm=0.1, length_mm=5.0),     # would be PASS
    d("dent", conf=0.3, diameter_mm=8.0),                    # would be REJECT
    d("corrosion", conf=0.3, area_mm2=50.0),                 # would be REWORK
])
def test_low_confidence_goes_to_hold(defect):
    verdict, j = one(defect)
    assert verdict == j["disposition"] == "HOLD_FOR_QA"
    assert j["status"] == "uncertain, operator to confirm"
    assert "+LOWCONF" in j["rule_id"] and "uncertain" in j["rule"]


def test_confidence_does_not_change_severity():
    _, hi = one(d("dent", conf=0.95, diameter_mm=8.0))
    _, lo = one(d("dent", conf=0.3, diameter_mm=8.0))
    assert hi["severity"] == lo["severity"] == "critical"


@pytest.mark.parametrize("kind, size, far, near", [
    ("dent", {"diameter_mm": 3.0}, ("minor", "PASS"), ("major", "HOLD_FOR_QA")),
    ("dent", {"diameter_mm": 5.0}, ("major", "HOLD_FOR_QA"), ("critical", "REJECT")),
    ("scratch", {"width_mm": 0.1, "length_mm": 5.0}, ("minor", "PASS"), ("major", "HOLD_FOR_QA")),
    ("corrosion", {"area_mm2": 300.0}, ("major", "REWORK"), ("critical", "HOLD_FOR_QA")),
    ("stain", {"area_mm2": 20.0}, ("minor", "CLEAN_AND_REINSPECT"), ("major", "CLEAN_AND_REINSPECT")),
])
def test_edge_proximity_raises_severity(kind, size, far, near):
    _, jf = one(d(kind, near=10.0, **size))
    _, jn = one(d(kind, near=4.0, **size))
    assert (jf["severity"], jf["disposition"]) == far
    assert (jn["severity"], jn["disposition"]) == near
    assert "+EDGE" in jn["rule_id"] and "from an edge or hole" in jn["rule"]


def test_unknown_edge_distance_is_not_raised():
    _, j = one(d("dent", near=None, diameter_mm=3.0))
    assert j["severity"] == "minor"


def test_scratch_cluster_of_three_is_major():
    three = [d("scratch", did=f"D{k}", x=20.0 + 10 * k, y=50.0, width_mm=0.1, length_mm=5.0) for k in range(3)]
    verdict, j = judge_steel(three, SPEC, PART)
    assert all(x["severity"] == "major" and x["cluster"]["count"] == 3 for x in j)
    assert all("ISO 7788 cl. 6.1 (40 mm cluster)" in x["reference"] for x in j)
    assert verdict == "HOLD_FOR_QA"
    _, two = judge_steel(three[:2], SPEC, PART)
    assert all(x["severity"] == "minor" for x in two)


def test_scratches_far_apart_are_not_a_cluster():
    wide = {**PART, "size_mm": [300.0, 100.0]}
    far = [d("scratch", did=f"D{k}", x=10.0 + 100 * k, y=50.0, width_mm=0.1, length_mm=5.0) for k in range(3)]
    _, j = judge_steel(far, SPEC, wide)
    assert all(x["severity"] == "minor" and "cluster" not in x for x in j)


def test_pit_detections_close_together_are_counted_together():
    pits = [d("pit", did=f"D{k}", x=30.0 + 5 * k, y=30.0, diameter_mm=0.3, count=3) for k in range(2)]
    _, j = judge_steel(pits, SPEC, PART)
    assert all(x["severity"] == "major" and x["cluster"]["count"] == 6 for x in j)


# ---- verdict, thickness, part ----------------------------------------------------------------------

def test_verdict_order_and_worst():
    assert STEEL_VERDICTS == ("PASS", "CLEAN_AND_REINSPECT", "REWORK", "HOLD_FOR_QA", "REJECT")
    assert all(v in VERDICTS for v in STEEL_VERDICTS)
    idx = [VERDICTS.index(v) for v in STEEL_VERDICTS]
    assert idx == sorted(idx)
    assert worst([]) == "PASS"
    assert judge_steel([], SPEC, PART) == ("PASS", [])
    mixed = [d("stain", did="D1", x=20, area_mm2=20.0), d("corrosion", did="D2", x=80, area_mm2=50.0)]
    assert judge_steel(mixed, SPEC, PART)[0] == "REWORK"
    assert judge_steel([*mixed, d("crack", did="D3", x=50, y=90, length_mm=2.0)], SPEC, PART)[0] == "REJECT"


def test_judge_part_dispatches_to_steel():
    verdict, j = judge_part([d("crack", length_mm=2.0)], SPEC)
    assert verdict == "REJECT" and j[0]["tier"] == "Critical defect"


def test_unknown_type_goes_to_hold():
    verdict, j = one(d("rivet"))
    assert verdict == "HOLD_FOR_QA" and j["severity"] == "unclassified"


def test_record_of_an_unknown_type_and_a_box_over_the_edge_validates():
    # an old detector's class (not in the schema enum) and a box overlapping the plate edge (near < 0)
    rec = record([d("rivet", did="D1", x=20), d("scratch", did="D2", x=80, near=-1.2, width_mm=0.1, length_mm=5.0)])
    assert validate_record(rec) == [] and _errors(rec, COMBINED) == []
    d1, d2 = rec["defects"]
    assert d1["type"] == "unknown" and d1["detected_type"] == "rivet" and d1["disposition"] == "HOLD_FOR_QA"
    assert rec["summary"]["by_type"] == {"unknown": 1, "scratch": 1}
    assert d2["near_edge_mm"] == 0.0 and d2["severity"] == "critical"      # it touches the edge


@pytest.mark.parametrize("t, ref, text", [
    (3.0, "EN 10163-2 / ISO 7788 Table 1 (3-8 mm: 0.2 mm)", "On 3-8 mm plate up to 0.2 mm"),
    (10.0, "EN 10163-2 / ISO 7788 Table 1 (8-25 mm: 0.3 mm)", "On 8-15 mm plate up to 0.3 mm"),
    (1.5, "IS 1079 / IS 2062 (limits by agreement)", "agreed with the customer"),
])
def test_thickness_sets_depth_targets(t, ref, text):
    _, j = one(d("scratch", width_mm=0.3, length_mm=8.0), {**PART, "thickness_mm": t})
    assert ref in j["reference"] and text in j["action"]


def test_is2062_grinding_limit_only_for_hot_rolled_mild_steel():
    _, hot = one(d("scratch", width_mm=0.3, length_mm=8.0))
    _, ss = one(d("scratch", width_mm=0.3, length_mm=8.0), {**PART, "material": "ss_brushed"})
    assert "4 % of the thickness here (0.12 mm of 3 mm)" in hot["action"]
    assert "4 %" not in ss["action"]


def test_steel_part_defaults_from_spec():
    p = steel_part(SPEC, {"material": "ss_2b"})
    assert p["thickness_mm"] == 3.0 and p["size_mm"] == [100.0, 100.0] and p["family"] == "stainless"


def test_judge_steel_defect_without_context():
    j = judge_steel_defect(d("dent", diameter_mm=5.0), SPEC, PART)
    assert j["disposition"] == "HOLD_FOR_QA" and j["rule_id"] == "DNT-MAJ"


# ---- record 3.0 ------------------------------------------------------------------------------------

jsonschema = pytest.importorskip("jsonschema")


def _errors(rec, path):
    v = jsonschema.Draft202012Validator(json.loads(path.read_text()))
    return [e.message for e in v.iter_errors(json.loads(json.dumps(rec)))]


def test_record_validates_against_v3_and_combined_schema():
    rec = record([d("scratch", did="D1", x=20, width_mm=0.3, length_mm=8.0),
                  d("stain", did="D2", x=80, area_mm2=20.0)])
    assert rec["schema"] == SCHEMA_ID_STEEL and rec["verdict"] == "HOLD_FOR_QA"
    assert rec["verdict_instruction"] == VERDICT_INSTRUCTION_STEEL["HOLD_FOR_QA"]
    assert _errors(rec, V3) == [] and _errors(rec, COMBINED) == [] and validate_record(rec) == []
    d1 = rec["defects"][0]
    for key in ("id", "type", "confidence", "box_px", "location_mm", "size", "near_edge_mm", "severity", "tier",
                "disposition", "rule", "reference", "action"):
        assert key in d1
    assert set(rec["capture"]) >= {"mode", "device_mode", "r_ambient", "lux_est", "motion_px", "sat_pct", "blur",
                                   "gate", "gate_message"}
    assert rec["timing_ms"]["total"] == pytest.approx(806.0)
    assert {r["id"] for r in rec["references"]} >= {"EN 10163-2 / ISO 7788 Table 1 (3-8 mm: 0.2 mm)"}
    assert rec["summary"] == {"defect_count": 2, "by_type": {"scratch": 1, "stain": 1},
                              "by_severity": {"major": 1, "minor": 1}}


def test_retake_record_has_no_defects():
    msg = "Too bright: shade the part or fit the hood, then retake."
    rec = record([d("crack", length_mm=3.0)], gate="RETAKE", message=msg, r_ambient=7.5)
    assert rec["verdict"] == "RETAKE" and rec["defects"] == [] and rec["actions"] == []
    assert rec["capture"]["gate_message"] == msg
    assert _errors(rec, V3) == [] and _errors(rec, COMBINED) == []


def test_builder_rejects_inconsistent_verdicts():
    with pytest.raises(ValueError):
        build_steel_record("P", PART, [], "RETAKE", {"gate": "PASS"}, {}, {"vision": "x"}, spec=SPEC)
    with pytest.raises(ValueError):
        build_steel_record("P", PART, [], "HOLD_FOR_MRB", {"gate": "PASS"}, {}, {"vision": "x"}, spec=SPEC)


def test_schema_rejects_inconsistent_retake():
    rec = record([d("crack", length_mm=3.0)])
    bad = dict(rec, verdict="RETAKE")
    assert _errors(bad, V3)
    bad = json.loads(json.dumps(rec))
    bad["capture"]["gate"] = "RETAKE"
    assert _errors(bad, V3)
    bad = json.loads(json.dumps(rec))
    del bad["defects"][0]["box_px"]
    assert _errors(bad, V3)


def test_combined_schema_embeds_the_v3_schema():
    v3 = json.loads(V3.read_text())
    combined = json.loads(COMBINED.read_text())
    for name, schema in v3["$defs"].items():
        assert combined["$defs"][name] == schema, name


def test_combined_schema_keeps_2x_checks():
    report = {"schema": "ayas.inspection/2.0", "mode": "surface", "part_id": "P", "timestamp": "t",
              "part": {"name": "plate", "material": "al"}, "verdict": "PASS", "verdict_instruction": "ok",
              "summary": {"defect_count": 0, "by_type": {}}, "defects": [], "actions": [],
              "models": {"classifier": "x", "golden_parts": None}, "timing_ms": {}}
    assert _errors(report, COMBINED)            # no rig
    report["rig"] = {"kind": "handheld", "camera": "imx500", "detector": "yolo"}
    assert _errors(report, COMBINED) == []
    assert _errors(dict(report, domain="aerospace", schema="ayas.inspection/2.1"), COMBINED)  # aero needs more


class FakeGate:
    verdict, reasons, message = "PASS_FLAGGED", ["r_ambient 0.6 > 0.3"], ""


def test_capture_block_and_metrics():
    cap = capture_from_metrics({"r_ambient": 0.6, "motion_px": 0.21, "masked_frac": 0.002, "blur": 700.0,
                                "blur_rel": 0.93, "sat_frac_a": 0.0}, FakeGate(), device_mode="hood", led_lux=3000.0)
    assert cap["gate"] == "PASS_FLAGGED" and cap["gate_reasons"] == ["r_ambient 0.6 > 0.3"]
    assert cap["lux_est"] == 1800.0 and cap["sat_pct"] == pytest.approx(0.2) and cap["blur"] == 0.93
    assert cap["device_mode"] == "hood" and cap["mode"] == "burst5"
    assert capture_block({"gate": {"verdict": "RETAKE", "message": "Hold still and retake.", "reasons": ["motion"]},
                          "r_ambient": float("inf")})["r_ambient"] is None
    with pytest.raises(ValueError):
        capture_block({"gate": "MAYBE"})


def test_compact_for_llm_steel():
    defects = [d("stain", did=f"D{k + 1}", x=10.0 + 15 * k, y=20.0 + 10 * k, area_mm2=20.0) for k in range(6)]
    defects.append(d("crack", did="D7", x=50.0, y=90.0, length_mm=2.0))
    rec = record(defects, gate="PASS_FLAGGED", message="")
    slim = json.loads(compact_for_llm(rec))
    assert slim["verdict"] == "REJECT" and slim["capture"]["gate"] == "PASS_FLAGGED"
    assert len(slim["defects"]) == 3 and slim["not_shown"] == {"CLEAN_AND_REINSPECT": 4}
    assert [e["id"] for e in slim["defects"]] == ["D1", "D2", "D7"]
    assert slim["defects"][1]["action"] == "same as D1"
    for e in slim["defects"]:
        assert {"type", "severity", "tier", "disposition", "size", "location_mm", "rule", "reference", "action",
                "confidence"} <= set(e)
        assert "=>" not in e["rule"]
    assert len(compact_for_llm(rec)) < 2000


def test_compact_drops_the_low_confidence_prefix():
    rec = record([d("dent", conf=0.3, diameter_mm=5.0)])
    full = rec["defects"][0]["action"]
    slim = json.loads(compact_for_llm(rec))["defects"][0]
    assert full.startswith("1. Uncertain detection") and slim["status"] == "uncertain, operator to confirm"
    assert slim["action"] == full.split(" Rule action: ", 1)[1]
