"""Policies adopted 22 Nov 2026: P1h two-burst confirmation of low-confidence detections, and
hood-only certification (open mode never outputs PASS or CLEAN_AND_REINSPECT)."""

from pathlib import Path

import pytest

from ayas import rules

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec():
    return rules.load_spec(ROOT / "configs/spec_steel.yaml")


def crack(conf, x=50.0, y=40.0):
    return {"type": "crack", "confidence": conf, "box_px": [0, 0, 60, 16], "location_mm": {"x": x, "y": y},
            "size": {"length_mm": 5.0}, "near_edge_mm": 20.0}


PART = {"material": "ms_hot_rolled", "thickness_mm": 5.0, "size_mm": [100, 100]}


def test_spec_has_the_adopted_policies(spec):
    low = spec["modifiers"]["low_confidence"]
    assert low["confirm"]["within_mm"] == 3.0 and low["confirm"]["unconfirmed_disposition"] == "HOLD_FOR_QA"
    assert "crack" in low["keep_disposition"]
    assert spec["device_policy"]["open"]["no_accept"] == ["PASS", "CLEAN_AND_REINSPECT"]


def test_confirm_flags_only_low_confidence(spec):
    out = rules.confirm_defects([crack(0.3), crack(0.9), crack(0.3, x=80)],
                                [crack(0.6, x=51.5, y=41.0)], spec)
    assert out[0]["confirmed"] is True            # same type within 3 mm (1.8 mm)
    assert "confirmed" not in out[1]              # confident: no flag
    assert out[2]["confirmed"] is False           # 30 mm away
    other_type = rules.confirm_defects([crack(0.3)], [dict(crack(0.6), type="scratch")], spec)
    assert other_type[0]["confirmed"] is False    # another class at the same place does not confirm


def test_unconfirmed_low_conf_crack_is_capped_at_hold(spec):
    [d] = rules.confirm_defects([crack(0.32)], [], spec)
    verdict, judged = rules.judge_steel([d], spec, PART)
    assert verdict == "HOLD_FOR_QA" and "UNCONF" in judged[0]["rule_id"]


def test_confirmed_low_conf_crack_keeps_reject(spec):
    [d] = rules.confirm_defects([crack(0.32)], [crack(0.41, x=50.5)], spec)
    verdict, judged = rules.judge_steel([d], spec, PART)
    assert verdict == "REJECT" and "confirmed by the second burst" in judged[0]["rule"]


def test_single_burst_keeps_the_old_rule(spec):
    verdict, _ = rules.judge_steel([crack(0.32)], spec, PART)   # no confirmation info: crack never downgraded
    assert verdict == "REJECT"


def test_confident_crack_rejects_without_confirmation(spec):
    [d] = rules.confirm_defects([crack(0.8)], [], spec)
    assert rules.judge_steel([d], spec, PART)[0] == "REJECT"


@pytest.mark.parametrize("verdict, mode, want", [
    ("PASS", "open", "HOLD_FOR_QA"), ("CLEAN_AND_REINSPECT", "open", "HOLD_FOR_QA"),
    ("REJECT", "open", "REJECT"), ("HOLD_FOR_QA", "open", "HOLD_FOR_QA"),
    ("PASS", "hood", "PASS"), ("CLEAN_AND_REINSPECT", "hood", "CLEAN_AND_REINSPECT")])
def test_open_mode_never_accepts(spec, verdict, mode, want):
    got, note = rules.apply_device_policy(verdict, mode, spec)
    assert got == want
    assert (note is not None) == (got != verdict) and (note is None or "fit the hood" in note)


def test_twin_pair_is_one_inspection_and_a_refused_second_burst_retakes(tmp_path):
    from tests.test_app import make_pipeline, write_npz
    from ayas.app import sources as SR

    write_npz(tmp_path / "ok" / "s1_v0.npz", seed=0)
    write_npz(tmp_path / "ok" / "s1_v1.npz", seed=1)
    item = SR.TwinSource(tmp_path / "ok", loop=False).next()
    assert "partner" in item[1]
    rec = make_pipeline(tmp_path / "a").run(*item)
    assert rec["verdict"] != "RETAKE" and rec["capture"]["confirm"]["bursts"] == 2

    write_npz(tmp_path / "bad" / "s2_v0.npz", seed=0)
    write_npz(tmp_path / "bad" / "s2_v1.npz", amb=0.9, seed=1)          # second burst too bright
    rec = make_pipeline(tmp_path / "b").run(*SR.TwinSource(tmp_path / "bad", loop=False).next())
    assert rec["verdict"] == "RETAKE" and rec["capture"]["gate_message"].startswith("Second burst refused")
    assert "second_burst" in rec["capture"]["gate_reasons"] and rec["defects"] == []


def test_single_variant_folder_has_no_partner(tmp_path):
    from tests.test_app import write_npz
    from ayas.app import sources as SR

    write_npz(tmp_path / "s3_v0.npz")
    assert "partner" not in SR.TwinSource(tmp_path, loop=False).next()[1]
