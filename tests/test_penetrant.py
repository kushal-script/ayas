import json
from pathlib import Path

import pytest

from ayas.capture import capture_set
from ayas.penetrant import PenetrantInspector
from ayas.sim import SimDefect, SimRig

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "inspection.schema.json"


def run(cfg, indications, profile=None, seed=3, development_min=10):
    rig = SimRig(cfg, seed=seed)
    rig.place_developed(indications)
    frames = capture_set(rig.lights, rig.cameras, (cfg.pair_light("top"), "dark"), settle_s=0)
    return PenetrantInspector(cfg, profile).inspect(frames, "pt", {"development_min": development_min})


def test_clean_developer_passes(sim_cfg):
    report = run(sim_cfg, None)
    assert report["verdict"] == "PASS" and report["defects"] == []
    assert report["mode"] == "penetrant"


def test_crack_is_a_linear_indication_and_rejects(sim_cfg):
    report = run(sim_cfg, [SimDefect("crack", 40, 40, 9.0, angle_deg=20, seed=4)])
    (d,) = report["defects"]
    assert d["type"] == "linear_indication" and d["relevant"]
    assert d["disposition"] == "REJECT" and d["severity"] == "critical"
    assert report["verdict"] == "REJECT"


def test_faint_smear_from_poor_removal_is_ignored(sim_cfg):
    assert run(sim_cfg, [SimDefect("smear", 40, 40, 6.0)])["defects"] == []


def test_pores_are_graded_by_size(sim_cfg):
    report = run(sim_cfg, [SimDefect("pore", 20, 20, 1.0), SimDefect("pore", 60, 20, 2.5), SimDefect("pore", 40, 60, 5.5)])
    by_x = sorted(report["defects"], key=lambda d: (d["location_mm"]["y"], d["location_mm"]["x"]))
    small, medium, large = by_x
    assert all(d["type"] == "rounded_indication" for d in by_x)
    assert not small["relevant"] and small["disposition"] == "PASS"  # <= 1/16 in: logged only
    assert medium["relevant"] and medium["disposition"] == "PASS"  # under the 3/16 in limit
    assert large["disposition"] == "REJECT"  # over 3/16 in
    assert large["size"]["length_mm"] == pytest.approx(5.5, abs=0.6)


def test_four_pores_in_a_line_reject_under_asme(sim_cfg):
    report = run(sim_cfg, [SimDefect("pore_line", 40, 40, 2.0, angle_deg=15)])
    assert len(report["defects"]) == 4
    assert all(d["type"] == "rounded_indication" and d["relevant"] for d in report["defects"])
    assert report["penetrant"]["aligned_groups"] and len(report["penetrant"]["aligned_groups"][0]) == 4
    assert report["verdict"] == "REJECT"


def test_iso_profile_merges_close_pores_into_one_linear_indication(sim_cfg):
    report = run(sim_cfg, [SimDefect("pore_line", 40, 40, 2.0, angle_deg=15)], profile="iso23277_al2")
    (d,) = report["defects"]
    assert d["type"] == "linear_indication" and d["disposition"] == "REJECT"


def test_iso_level_2_accepts_a_short_linear_indication(sim_cfg):
    report = run(sim_cfg, [SimDefect("slit", 40, 40, 3.0, angle_deg=60)], profile="iso23277_al2")
    (d,) = report["defects"]
    assert d["type"] == "linear_indication" and d["size"]["length_mm"] < 4.0
    assert report["verdict"] == "PASS"


def test_development_time_outside_window_warns(sim_cfg):
    report = run(sim_cfg, None, development_min=90)
    assert report["penetrant"]["warnings"]


def test_penetrant_report_matches_schema(sim_cfg):
    jsonschema = pytest.importorskip("jsonschema")
    report = run(sim_cfg, ["crack", "pore"])
    jsonschema.validate(json.loads(json.dumps(report)), json.loads(SCHEMA.read_text()))
