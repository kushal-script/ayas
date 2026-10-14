from pathlib import Path

import pytest

from trinetra.rules import judge_defect, judge_part, load_spec, zone_for

SPEC = load_spec(Path(__file__).resolve().parents[1] / "configs" / "spec.yaml")


def defect(kind="scratch", x=40.0, y=40.0, conf=0.9, **size):
    base = {"length_mm": 0.0, "width_mm": 0.0, "area_mm2": 0.0, "diameter_mm": 0.0}
    base.update(size)
    return {"id": "D1", "type": kind, "confidence": conf, "location_mm": {"x": x, "y": y}, "size": base}


def test_zones():
    assert zone_for(SPEC, 40, 5) == "sealing_face"
    assert zone_for(SPEC, 40, 40) == "general"


@pytest.mark.parametrize(
    "length, severity, disposition",
    [(0.5, "negligible", "PASS"), (3.0, "minor", "PASS"), (7.0, "major", "REWORK"), (20.0, "critical", "REJECT")],
)
def test_scratch_ladder(length, severity, disposition):
    d = judge_defect(defect(length_mm=length), SPEC)
    assert (d["severity"], d["disposition"]) == (severity, disposition)


def test_sealing_face_is_stricter():
    assert judge_defect(defect(y=5, length_mm=3.0), SPEC)["severity"] == "major"
    assert judge_defect(defect(y=40, length_mm=3.0), SPEC)["severity"] == "minor"


def test_any_crack_rejects():
    assert judge_defect(defect("crack", length_mm=0.3), SPEC)["disposition"] == "REJECT"


def test_dust_means_clean_and_reinspect():
    assert judge_defect(defect("dust", diameter_mm=0.4), SPEC)["disposition"] == "CLEAN_AND_REINSPECT"


def test_low_confidence_and_unknown_go_to_qa():
    assert judge_defect(defect(conf=0.4, length_mm=7.0), SPEC)["disposition"] == "HOLD_FOR_QA"
    assert judge_defect(defect("unknown"), SPEC)["disposition"] == "HOLD_FOR_QA"


def test_part_verdict_is_worst_defect():
    verdict, _ = judge_part([defect("dust", diameter_mm=0.3), defect(length_mm=7.0)], SPEC)
    assert verdict == "REWORK"
    verdict, _ = judge_part([], SPEC)
    assert verdict == "PASS"
