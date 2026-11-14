"""Twin records: perfect-detector records 3.0 from steel plans and sidecars, the ground-truth
verdict and the advisor regression harness (twin/records.py). Plans and sidecars are small
hand-made fixtures in the DESIGN_STEEL D/E shapes, so these tests do not depend on the sampler."""

import json
import math
import stat
import sys
from pathlib import Path

import pytest

from trinetra.report import validate_record
from trinetra.rules import load_spec
from trinetra.twin import records as R

ROOT = Path(__file__).resolve().parents[1]
STEEL = load_spec(ROOT / "configs" / "spec_steel.yaml")
TOO_BRIGHT = "Too bright: shade the part or fit the hood, then retake."


def circle(u, v, d, n=32):
    return [[u + d / 2 * math.cos(2 * math.pi * k / n), v + d / 2 * math.sin(2 * math.pi * k / n)] for k in range(n)]


def rect(u, v, length, width):
    return [[u - length / 2, v - width / 2], [u + length / 2, v - width / 2], [u + length / 2, v + width / 2],
            [u - length / 2, v + width / 2]]


def defect(did, cls, kind, poly, measure, plate_edge, params=None, label=True):
    return {"id": did, "cls": cls, "kind": kind, "params": params or {}, "label": label,
            "footprint": {"polys_mm": [poly], "measure": measure},
            "near": {"fastener_edge_mm": None, "plate_edge_mm": plate_edge, "fastener": None}}


def steel_plan(seed=900001, material="ms_hot_rolled", t=3.0, defects=None):
    if defects is None:
        defects = [
            defect("D1", "scratch", "scriber", rect(30, 40, 20, 0.3), {"length_mm": 20.0, "width_mm": 0.3}, 30.0),
            defect("D2", "stain", "oil", circle(70, 70, 10), {"area_mm2": 78.5, "diameter_mm": 10.0}, 25.0),
            defect("D3", "corrosion", "rust", circle(85, 15, 12), {"area_mm2": 113.1, "diameter_mm": 12.0}, 9.0,
                   params={"u": 85, "v": 15, "r_mm": 6, "coverage": 0.8, "pitted": True}),
            defect("D4", "pit", "cluster", circle(20, 80, 6), {"diameter_mm": 6.0, "count": 9, "area_mm2": 28.3}, 17.0,
                   params={"u": 20, "v": 80, "r_mm": 3, "n": 9, "pit_d_mm": [0.2, 0.6]}),
            defect("D5", "crack", "fatigue", rect(50, 50, 4, 0.05), {"length_mm": 4.0, "width_mm": 0.05}, 48.0,
                   label=False),
        ]
    return {"schema": "trinetra.twin.scene/2", "scene_id": f"s{seed:06d}", "seed": seed, "split": "test",
            "mode": "surface", "source": "random",
            "plate": {"size_mm": [100.0, 100.0], "thickness_mm": t, "material": {"kind": material}},
            "holes": [], "rivets": [], "defects": defects, "distractors": [{"kind": "marker", "params": {"u": 10, "v": 10}}],
            "ambient": {"condition": "C3_window", "lux": 5000.0, "lights": []}, "extras": {"device": {"mode": "open"}}}


def steel_sidecar(plan, record_fields=True):
    """Sidecar in post's sidecar/2 layout: defects carry the record-3.0 fields (type, box_px,
    location_mm, size, near_edge_mm) next to measure / near / centre_mm / bbox_grid."""
    defs = []
    for d in plan["defects"]:
        pts = d["footprint"]["polys_mm"][0]
        us, vs = [p[0] for p in pts], [p[1] for p in pts]
        box = [(min(us) + 5) / 0.08, (min(vs) + 5) / 0.08, (max(us) + 5) / 0.08, (max(vs) + 5) / 0.08]
        rec = {"id": d["id"], "cls": d["cls"], "kind": d["kind"], "label": d["label"],
               "measure": d["footprint"]["measure"], "near": d["near"],
               "centre_mm": [sum(us) / len(us), sum(vs) / len(vs)], "bbox_mm": [min(us), min(vs), max(us), max(vs)],
               "bbox_grid": box, "visible": True}
        if record_fields:
            rec.update(type=d["cls"], box_px=box, size=dict(d["footprint"]["measure"]),
                       location_mm={"x": round(sum(us) / len(us), 2), "y": round(sum(vs) / len(vs), 2)},
                       near_edge_mm=d["near"]["plate_edge_mm"])
        defs.append(rec)
    metrics = {"r_ambient": 0.2, "sat_frac_a": 0.0, "led_level": 0.4, "blur": 812.0, "blur_rel": 0.95,
               "motion_px": 0.3, "residual_sigma": 1.1, "masked_frac": 0.001}
    every = {d["id"]: True for d in plan["defects"]}
    return {
        "schema": "trinetra.twin.sidecar/2", "scene_id": plan["scene_id"], "plate": "P1", "seed": plan["seed"],
        "split": "test", "plate_size_mm": [100.0, 100.0], "condition": "C3_window", "device_mode": "open",
        "part": {"material": plan["plate"]["material"]["kind"], "thickness_mm": plan["plate"]["thickness_mm"],
                 "size_mm": [100.0, 100.0]},
        "defects": defs,
        "variants": [
            {"k": 0, "lux": 4200.0, "arm": "burst", "gate": {"verdict": "PASS", "reasons": [], "message": ""},
             "metrics": metrics, "visible": every},
            {"k": 1, "lux": 9000.0, "arm": "burst",
             "gate": {"verdict": "PASS_FLAGGED", "reasons": ["r_ambient 0.6 > 0.3"], "message": ""},
             "metrics": {**metrics, "r_ambient": 0.6}, "visible": {**every, "D1": False}},
            {"k": 2, "lux": 60000.0, "condition": "C6_sun", "visible": every, "arm": "burst", "gate": "RETAKE",
             "gate_message": TOO_BRIGHT, "metrics": {**metrics, "r_ambient": 7.5, "sat_frac_a": 0.04},
             "arms": {"burst": {"gate": "RETAKE", "reasons": ["too_bright"], "message": TOO_BRIGHT, "gated": True,
                                "metrics": {**metrics, "r_ambient": 7.5, "sat_frac_a": 0.04}},
                      "naive": {"gate": "PASS", "reasons": [], "message": "", "gated": False,
                                "metrics": {"median_u8": 141.0, "sat_pct": 3.2}}}},
        ],
    }


# ---- records from plans and sidecars ---------------------------------------------------------------

def test_steel_record_from_plan_is_a_perfect_detector():
    plan = steel_plan()
    rec = R.records_from_plan(plan, STEEL, timestamp="2026-11-14T10:00:00+05:30")
    assert rec["schema"] == "trinetra.inspection/3.0" and validate_record(rec) == []
    assert [d["twin_id"] for d in rec["defects"]] == ["D1", "D2", "D3", "D4"]      # D5 is unlabelled
    by = {d["twin_id"]: d for d in rec["defects"]}
    assert by["D3"]["near_edge_mm"] == 9.0 and by["D3"]["est_rust_grade"] == "D" and by["D3"]["severity"] == "critical"
    assert by["D4"]["size"]["count"] == 9 and by["D4"]["size"]["pit_d_mm"] == 0.6
    assert by["D1"]["location_mm"] == {"x": 30.0, "y": 40.0} and by["D1"]["box_px"] is None
    assert all("depth_mm" not in d["size"] for d in rec["defects"])
    assert rec["material"] == "ms_hot_rolled" and rec["thickness_mm"] == 3.0
    assert rec["capture"]["gate"] == "PASS" and rec["capture"]["lux_est"] == 5000.0
    assert rec["source"]["condition"] == "C3_window" and rec["backend"]["vision"] == R.PERFECT
    assert rec["verdict"] == "HOLD_FOR_QA" == R.gt_verdict(R.labels_from_plan(plan), STEEL, R.steel_part_from_plan(plan))


def test_records_from_sidecar_one_per_variant_and_arm():
    plan = steel_plan()
    recs = R.records_from_sidecar(steel_sidecar(plan), STEEL, plan=plan)
    assert [r["part_id"] for r in recs] == ["s900001_v0", "s900001_v1", "s900001_v2", "s900001_v2_naive"]
    assert all(validate_record(r) == [] for r in recs)
    v0, v1, v2, naive = recs
    assert len(v0["defects"]) == 4 and v0["defects"][0]["box_px"] == pytest.approx([312.5, 560.6, 562.5, 564.4], abs=0.1)
    assert v0["capture"]["r_ambient"] == 0.2 and v0["capture"]["sat_pct"] == pytest.approx(0.1)
    assert v0["capture"]["lux_est"] == 4200.0 and v0["capture"]["blur"] == 0.95
    assert [d["twin_id"] for d in v1["defects"]] == ["D2", "D3", "D4"]              # D1 not visible in v1
    assert v1["capture"]["gate"] == "PASS_FLAGGED" and v1["capture"]["gate_reasons"] == ["r_ambient 0.6 > 0.3"]
    assert v2["verdict"] == "RETAKE" and v2["defects"] == [] and v2["capture"]["gate_message"] == TOO_BRIGHT
    assert v2["source"]["condition"] == "C6_sun" and v2["source"]["arm"] == "burst"
    assert v2["capture"]["gate_reasons"] == ["too_bright"] and v2["capture"]["r_ambient"] == 7.5
    assert naive["capture"]["mode"] == "naive" and naive["capture"]["gate"] == "PASS" and len(naive["defects"]) == 4
    assert naive["capture"]["gate_message"] == "" and naive["capture"]["sat_pct"] == 3.2
    assert {d["twin_id"]: d.get("est_rust_grade") for d in v0["defects"]}["D3"] == "D"   # pitted from the plan
    only_naive = R.records_from_sidecar(steel_sidecar(plan), STEEL, plan=plan, arms=("naive",))
    assert [r["part_id"] for r in only_naive] == ["s900001_v2_naive"]


def test_capture_lux_is_the_device_estimate_not_the_twin_truth():
    # post's sidecar/2 burst metrics carry lightproc's lux_est: 0 under the hood, None when the
    # ambient saturates; the true condition lux stays in the record's source only.
    plan = steel_plan()
    sc = steel_sidecar(plan)
    sc["device_mode"] = "hood"
    sc["variants"][0]["metrics"] = {**sc["variants"][0]["metrics"], "r_ambient": 0.0, "lux_est": 0.0}
    sc["variants"][2]["arms"]["burst"]["metrics"]["lux_est"] = None
    v0, _, v2, naive = R.records_from_sidecar(sc, STEEL, plan=plan)
    assert v0["capture"]["lux_est"] == 0.0 and v0["source"]["lux"] == 4200.0
    assert v0["capture"]["device_mode"] == "hood"
    assert v2["capture"]["lux_est"] is None and v2["source"]["lux"] == 60000.0
    assert naive["capture"]["lux_est"] is None                  # no ambient frame in the baseline arm
    assert all(validate_record(r) == [] for r in (v0, v2, naive))


def test_sidecar_without_plan_uses_its_part_block():
    sc = steel_sidecar(steel_plan())
    sc["part"] = {"material": "galvanised", "thickness_mm": 2.0, "size_mm": [100.0, 100.0]}
    rec = R.records_from_sidecar(sc, STEEL)[0]
    assert rec["material"] == "galvanised" and rec["thickness_mm"] == 2.0
    assert "est_rust_grade" not in {d["twin_id"]: d for d in rec["defects"]}["D3"]


def test_sidecar_labels_without_record_fields():
    plan = steel_plan()
    old = R.records_from_sidecar(steel_sidecar(plan, record_fields=False), STEEL, plan=plan)
    new = R.records_from_sidecar(steel_sidecar(plan), STEEL, plan=plan)
    assert [r["verdict"] for r in old] == [r["verdict"] for r in new]
    assert old[0]["defects"][0]["box_px"] == new[0]["defects"][0]["box_px"]


# ---- ground truth --------------------------------------------------------------------------------------

def test_gt_verdict_is_the_same_for_every_label_format():
    plan = steel_plan()
    part = R.steel_part_from_plan(plan)
    sc = steel_sidecar(plan)
    csv_rows = [{"defect_id": d["id"], "cls": d["cls"], "u_mm": d["footprint"]["polys_mm"][0][0][0],
                 "v_mm": d["footprint"]["polys_mm"][0][0][1], **d["footprint"]["measure"]}
                for d in plan["defects"] if d["label"]]
    rec_style = R.records_from_plan(plan, STEEL)["defects"]
    verdicts = {R.gt_verdict(R.labels_from_plan(plan), STEEL, part), R.gt_verdict(sc["defects"], STEEL, part, plan=plan),
                R.gt_verdict(sc["defects"], STEEL, part), R.gt_verdict(csv_rows, STEEL, part),
                R.gt_verdict(rec_style, STEEL, part)}
    assert verdicts == {"HOLD_FOR_QA"}


def test_gt_verdict_cases():
    part = {"material": "ms_hot_rolled", "thickness_mm": 3.0, "size_mm": [100.0, 100.0]}
    plan = steel_plan()
    assert R.gt_verdict([], STEEL, part) == "PASS"
    assert R.gt_verdict(plan["defects"], STEEL, part, visible={"D1": False, "D3": False, "D4": False}) == "CLEAN_AND_REINSPECT"
    crack = defect("C1", "crack", "open", rect(50, 50, 3, 0.1), {"length_mm": 3.0, "width_mm": 0.1}, 40.0)
    assert R.gt_verdict([crack], STEEL, part) == "REJECT"
    stain_at_edge = defect("S1", "stain", "water", circle(3, 50, 5), {"area_mm2": 19.6, "diameter_mm": 5.0}, 0.5)
    assert R.gt_verdict([stain_at_edge], STEEL, part) == "CLEAN_AND_REINSPECT"


def test_defect_from_label_fills_missing_measures():
    lbl = {"id": "X", "cls": "stain", "footprint": {"polys_mm": [rect(50, 50, 10, 4)]}}
    d = R.defect_from_label(lbl, part={"size_mm": [100.0, 100.0]})
    assert d["size"]["area_mm2"] == pytest.approx(40.0) and d["size"]["diameter_mm"] == pytest.approx(7.136, abs=1e-3)
    assert d["near_edge_mm"] == pytest.approx(45.0) and d["location_mm"] == {"x": 50.0, "y": 50.0}
    assert R.defect_from_label({"cls": "rivet"}) is None
    assert R.near_edge_mm({"plate_edge_mm": 12.0, "fastener_edge_mm": 3.5}) == 3.5
    assert R.near_edge_mm({"plate_edge_mm": None}) is None


# ---- regression harness ----------------------------------------------------------------------------

def _dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "ds"
    (ds / "plans").mkdir(parents=True)
    (ds / "sidecar").mkdir()
    plans = [steel_plan(900001), steel_plan(900002, "galvanised", 1.5), steel_plan(900003, "ss_brushed", 6.0),
             steel_plan(900004, defects=[])]
    for p in plans:
        (ds / "plans" / f"{p['scene_id']}.json").write_text(json.dumps(p))
    sc = steel_sidecar(plans[0])
    (ds / "sidecar" / f"{sc['scene_id']}.json").write_text(json.dumps(sc))
    return ds


def test_run_regression_steel_mock(tmp_path):
    ds = _dataset(tmp_path)
    s = R.run_regression(ds)
    assert s["domain"] == "steel_surface" and s["cases"] == 4 and s["records"] == 7
    assert s["guardrail_violations"] == 0 and s["failures"] == [] and s["gt_mismatches"] == []
    assert s["by_verdict"]["RETAKE"] == 1 and s["by_gate"] == {"PASS": 5, "PASS_FLAGGED": 1, "RETAKE": 1}
    assert s["by_verdict"]["PASS"] == 1                                     # the defect-free plate
    assert s["backends"] == {"mock-template": 7 * (1 + len(R.STEEL_QUESTIONS))}
    lines = (tmp_path / "regress" / "cases.jsonl").read_text().splitlines()
    assert len(lines) == 7 and json.loads(lines[0])["replies"][0]["q"] is None


def test_run_regression_counts_model_violations(tmp_path):
    runner = tmp_path / "llama_main"
    runner.write_text(f"#!{sys.executable}\nprint('The plate is certified and compliant with ISO 9001.<|eot_id|>')\n")
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    pte = tmp_path / "llama.pte"
    pte.write_bytes(b"")
    s = R.run_regression([steel_plan()], backend=str(pte), runner=str(runner), questions=("What should I do next?",))
    assert s["guardrail_violations"] == 2 and s["failures"] == []
    assert s["backends"] == {"mock-template": 2}
    assert "banned phrase" in " ".join(s["violation_examples"][0]["problems"])


def test_aero_regression_still_runs():
    plan = {"schema": "trinetra.twin.scene/1", "scene_id": "a000001", "seed": 1, "split": "train", "mode": "surface",
            "source": "random", "plate": {"size_mm": [100.0, 100.0], "thickness_mm": 3.0, "finish": "brushed"},
            "holes": [], "rivets": [], "distractors": [],
            "defects": [defect("D1", "scratch", "scriber", rect(30, 40, 20, 0.3), {"length_mm": 20.0, "width_mm": 0.3}, 30.0)],
            "fasteners": {"pattern": "none", "D_mm": None, "violations": []}}
    s = R.run_regression([plan], "configs/spec_aero.yaml")
    assert s["plans"] == 1 and s["by_verdict"] == {"REWORK": 1} and s["failures"] == []


def test_load_plans_accepts_both_plan_schemas():
    old = {"schema": "trinetra.twin.scene/1", "plate": {}}
    assert len(R.load_plans([old, steel_plan(), {"schema": "trinetra.twin.sidecar/2"}])) == 2


def test_gt_verdict_ignores_detection_confidence():
    part = {"material": "ms_hot_rolled", "thickness_mm": 3.0, "size_mm": [100.0, 100.0]}
    det = {"id": "D1", "type": "dent", "confidence": 0.2, "location_mm": {"x": 50.0, "y": 50.0},
           "size": {"diameter_mm": 8.0}, "near_edge_mm": 30.0}
    assert R.gt_verdict([det], STEEL, part) == "REJECT"     # a low-confidence detection would be HOLD_FOR_QA


def test_pte_backend_uses_the_pi_context(tmp_path):
    pte = tmp_path / "llama.pte"
    pte.write_bytes(b"")
    be = R._backend(str(pte), runner="llama_main")
    assert be.seq_len == R.PTE_SEQ_LEN == 2048 and "--seq_len=2048" in be.command("x")
