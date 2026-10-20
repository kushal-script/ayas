import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import capture

from trinetra.calibration import calibrate
from trinetra.capture import bright_view
from trinetra.config import VIEWS
from trinetra.sim import DEFECT_KINDS, SimDefect, SimRig

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "inspection.schema.json"


def test_calibration_recovers_true_mapping(sim_cfg):
    rig = SimRig(sim_cfg, seed=1)
    rig.jitter_mm = rig.jitter_deg = 0.0
    rig.place_part(None)
    frames = capture(sim_cfg, rig)
    cal = calibrate({v: bright_view(frames, v, sim_cfg.lights) for v in VIEWS}, sim_cfg)
    for view in VIEWS:
        corners = np.array(sim_cfg.cameras[view]["sim"]["plate_corners_px"], np.float32)
        inner = (0.6 * corners + 0.4 * corners.mean(0))[None]  # points well inside the plate
        got = cv2.perspectiveTransform(inner, cal.homographies[view])[0]
        want = cv2.perspectiveTransform(inner, np.linalg.inv(rig.true_h[view]))[0]
        assert np.abs(got - want).max() < 4.0, view  # grid px = 0.4 mm; grazing is coarse along y


def test_capture_covers_every_light_and_dark(sim_cfg, rig):
    rig.place_part(None)
    frames = capture(sim_cfg, rig)
    assert set(frames) == {f"{v}_{s}" for v in VIEWS for s in (*sim_cfg.lights, "dark")}


def test_good_parts_pass(sim_cfg, inspector, rig):
    for i in range(3):
        rig.place_part(None)
        report, _ = inspector.inspect(capture(sim_cfg, rig), f"good-{i}")
        assert report["verdict"] == "PASS"
        assert report["defects"] == []


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_each_defect_type_is_found_and_named(sim_cfg, inspector, rig, kind):
    truth = rig.place_part([kind])[0]
    report, _ = inspector.inspect(capture(sim_cfg, rig), f"{kind}-1")
    assert len(report["defects"]) == 1, report["defects"]
    d = report["defects"][0]
    assert d["type"] == kind
    assert np.hypot(d["location_mm"]["x"] - truth.x_mm, d["location_mm"]["y"] - truth.y_mm) < 1.5


@pytest.mark.parametrize("kind, sign", [("dent", -1), ("bump", +1)])
def test_3d_map_measures_depth(sim_cfg, inspector, rig, kind, sign):
    rig.place_part([SimDefect(kind, 40.0, 40.0, 3.5)])
    report, analysis = inspector.inspect(capture(sim_cfg, rig), f"{kind}-depth")
    (d,) = report["defects"]
    true_depth = sign * 0.18 * 3.5  # simulator: depth = 18 % of radius
    assert d["depth_mm"] == pytest.approx(true_depth, rel=0.15)
    assert report["rig"]["shape"]["mode"] == "full"
    assert analysis.height is not None


def test_scratch_length_is_measured(sim_cfg, inspector, rig):
    rig.place_part([SimDefect("scratch", 40.0, 50.0, 10.0, angle_deg=30.0)])
    report, _ = inspector.inspect(capture(sim_cfg, rig), "scratch-len")
    (d,) = report["defects"]
    assert 9.0 < d["size"]["length_mm"] < 12.0  # mask includes the scatter halo at both ends
    assert abs(((d["orientation_deg"] - 30 + 90) % 180) - 90) < 5


def test_room_light_change_is_removed_by_the_dark_frame(sim_cfg, inspector, rig):
    """Golden was built at up to 15 % room light; inspect under much brighter, tinted room light."""
    rig.ambient_max = 0.35
    rig.flicker = 0.01
    for i in range(3):
        rig.place_part(None)
        report, _ = inspector.inspect(capture(sim_cfg, rig), f"bright-room-{i}")
        assert report["verdict"] == "PASS", report["defects"]
    rig.place_part(["scratch"])
    report, _ = inspector.inspect(capture(sim_cfg, rig), "bright-room-scratch")
    assert [d["type"] for d in report["defects"]] == ["scratch"]


@pytest.mark.parametrize("kind", ["dent", "bump", "scratch", "stain", "crack"])
def test_three_light_arch_still_classifies(arch_only_cfg, kind):
    """Only the 3 arch lights: the 3D map falls back to slope-only, shading tells dent from bump."""
    from trinetra.pipeline import Inspector

    insp = Inspector(arch_only_cfg)
    assert insp.setup.mode == "slope-only"
    rig = SimRig(arch_only_cfg, seed=21)
    rig.place_part([kind])
    report, _ = insp.inspect(capture(arch_only_cfg, rig), f"arch-{kind}")
    assert [d["type"] for d in report["defects"]] == [kind]
    assert report["defects"][0]["depth_mm"] is None
    assert report["rig"]["shape"]["mode"] == "slope-only"


def test_report_matches_schema(sim_cfg, inspector, rig, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    rig.place_part(["scratch", "stain", "crack"])
    report, _ = inspector.inspect(capture(sim_cfg, rig), "schema-1", overlay_path=tmp_path / "o.png")
    jsonschema.validate(json.loads(json.dumps(report)), json.loads(SCHEMA.read_text()))
    assert (tmp_path / "o.png").exists()
    assert report["verdict"] == "REJECT"  # any crack is critical
