"""On-demand 3D of relief defects (app/surface3d.py) and the station's /api/surface."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_app import make_pipeline, wait_for, write_npz
from trinetra.app import sources as SR
from trinetra.app import surface3d as S3
from trinetra.app import ui as UI

ROOT = Path(__file__).resolve().parents[1]
TWIN_BURST = ROOT / "artifacts/jarvislabs/trinetra-render_574738_final/twin_matrix/post_final/bursts/s900100_C0h_v0.npz"


def inspected(tmp_path, **kw):
    path = write_npz(tmp_path / "b" / "s1_v0.npz", **kw)
    return make_pipeline(tmp_path, advisor=None).inspect(*SR.TwinSource(path.parent, loop=False).next())


def test_inspection_keeps_a_surface_source_and_the_grid_mapping_round_trips(tmp_path):
    insp = inspected(tmp_path)
    src = insp.surface
    assert src is not None and set(src.frames) >= {"L", "A", "R"}
    H = S3.H_grid_plate(src)
    corners = np.asarray(src.plate_corners_px, float)
    W, Hh = src.plate_size_mm
    for (u, v), g in zip([(0, 0), (W, 0), (W, Hh), (0, Hh)], corners):      # corner_plate_idx 0,1,2,3
        p = H @ [u, v, 1.0]
        assert np.allclose(p[:2] / p[2], g, atol=1e-3)
    box = S3.box_plate_mm([g for g in (*corners[0], *corners[2])], H)
    assert np.allclose(box, [0, 0, W, Hh], atol=1e-3)


def test_grid_frames_are_light_only_and_scratch_window_finds_the_groove(tmp_path):
    insp = inspected(tmp_path)
    fr = S3.grid_frames(insp.surface)
    assert fr["L"].shape == fr["R"].shape == insp.surface.grid_size[::-1]
    d = insp.record["defects"][0]
    win = S3.scratch_window(fr, d["box_px"], S3.H_grid_plate(insp.surface))
    assert pytest.approx(win[2] - win[0]) == 2 * S3.SCRATCH_HALF_MM
    x0, y0, x1, y1 = S3.box_plate_mm(d["box_px"], S3.H_grid_plate(insp.surface))
    cu, cv = (win[0] + win[2]) / 2, (win[1] + win[3]) / 2
    assert x0 - 0.5 <= cu <= x1 + 0.5 and y0 - 0.5 <= cv <= y1 + 0.5      # inside the detection


def test_without_light_geometry_3d_is_unavailable_with_a_reason(tmp_path):
    insp = inspected(tmp_path)
    res = S3.measure(insp.surface, insp.record["defects"], "ms_cold_rolled", 3.0, plans_dir=tmp_path / "none")
    assert res["available"] is False and "hood calibration" in res["why"] and res["items"] == []


def test_refused_capture_has_no_surface(tmp_path):
    assert inspected(tmp_path, amb=0.9).surface is None


def test_station_surface_endpoint_reports_unavailable(tmp_path):
    pipe = make_pipeline(tmp_path, advisor=None)
    pipe.surface_cfg = {"plans_dir": str(tmp_path / "none")}
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    st = UI.Station(pipe, SR.TwinSource(path.parent, loop=True))
    try:
        assert st.surface_state()["status"] == "idle"
        assert st.request("inspect")
        wait_for(lambda: pipe.last is not None)
        assert st.surface_request()
        wait_for(lambda: st.surface_state()["status"] != "measuring")
        s = st.surface_state()
        assert s["status"] == "unavailable" and s["for"] == pipe.last.key and s["why"]
        assert st.state()["surface"]["status"] == "unavailable"
        assert "/api/surface" in UI.PAGE and 'id="b-3d"' in UI.PAGE and "<script src" not in UI.PAGE
    finally:
        st.close()


@pytest.mark.skipif(not TWIN_BURST.exists(), reason="twin burst not on this machine")
def test_twin_gouge_depth_close_to_truth(tmp_path):
    torch = pytest.importorskip("torch")
    from trinetra import rules
    from trinetra.app import pipeline as PL
    cfg = PL.load_app_config(ROOT / "configs/app.yaml")
    pipe = PL.InspectionPipeline(cfg["_capture"], cfg["_grid"], PL.FakeDetector(), rules.load_spec(ROOT / "configs/spec_steel.yaml"),
                                 cfg["part"], None, log=None, advise="off")
    insp = pipe.inspect(*SR.read_twin_npz(TWIN_BURST, cfg["part"]))
    res = S3.measure(insp.surface, insp.record["defects"], "ms_hot_rolled", 5.463, method="pinn", n_seeds=1, threads=2,
                     plans_dir=ROOT / "data/twin_demo/plans")
    gouge = max(res["items"], key=lambda it: it["depth_mm"] or 0)
    assert gouge["type"] == "scratch" and abs(gouge["depth_mm"] - 0.309) < 0.1          # twin truth 0.309 mm
    assert len(gouge["height_mm"]) <= S3.VIEW_PX and json.dumps(res)                     # small and JSON-ready


@pytest.mark.skipif(not TWIN_BURST.exists(), reason="twin burst not on this machine")
def test_parallel_workers_match_sequential_and_stream(tmp_path):
    pytest.importorskip("torch")
    from trinetra import rules
    from trinetra.app import pipeline as PL
    cfg = PL.load_app_config(ROOT / "configs/app.yaml")
    pipe = PL.InspectionPipeline(cfg["_capture"], cfg["_grid"], PL.FakeDetector(), rules.load_spec(ROOT / "configs/spec_steel.yaml"),
                                 cfg["part"], None, log=None, advise="off")
    insp = pipe.inspect(*SR.read_twin_npz(TWIN_BURST, cfg["part"]))
    args = (insp.surface, insp.record["defects"], "ms_hot_rolled", 5.463)
    seq = S3.measure(*args, threads=1, plans_dir=ROOT / "data/twin_demo/plans")
    seen = []
    par = S3.measure(*args, workers=2, plans_dir=ROOT / "data/twin_demo/plans", on_item=seen.append, first="D3")
    assert [i["id"] for i in par["items"]] == ["D3", "D2"] and len(seen) == 2
    by_id = {i["id"]: i["depth_mm"] for i in seq["items"]}
    assert all(abs(i["depth_mm"] - by_id[i["id"]]) < 1e-6 for i in par["items"])
