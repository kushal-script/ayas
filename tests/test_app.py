"""On-device app (src/trinetra/app, DESIGN_STEEL.md H): sources, pipeline, advisor, UI, bench, CLI.

Synthetic bursts only (numpy frames written in twin/post.save_burst's npz layout); the Pi camera,
GPIO and the executorch TextLLMRunner are fakes injected into sys.modules; the llama_main runner
is a small Python script.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
import time
import types
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from trinetra.llm import CAUSE_LABEL

from trinetra import lightproc as LP
from trinetra import report as RP
from trinetra import rules
from trinetra.app import advisor as AD
from trinetra.app import bench as BE
from trinetra.app import cli as CLI
from trinetra.app import pipeline as PL
from trinetra.app import sources as SR
from trinetra.app import ui as UI

CFG = PL.load_app_config()
SPEC = CFG["_spec"]
N = 480                                            # synthetic frame size (px)
CORNERS = [[80.0, 80.0], [400.0, 80.0], [400.0, 400.0], [80.0, 400.0]]   # a 100 mm plate imaged as 320 px
S = 1250.0 / 320.0                                 # grid px per frame px (0.08 mm/px grid)
SCRATCH = (150, 239, 330, 241)                     # frame pixel edges x0, y0, x1, y1: 56.25 x 0.625 mm


def grid_edge(e: float) -> float:
    """A frame pixel-edge coordinate -> grid pixel-edge coordinate (plate corner 80 -> grid 62.5)."""
    return 62.5 + (e - 0.5 - 80.0) * S + 0.5


def plate_frames(amb: float = 0.01, seed: int = 0, scratch: bool = True, leds: bool = True) -> dict[str, np.ndarray]:
    """B1, L, A, R, B2 of a textured plate on a dark bench: B = 2 lit + amb, L = R = lit + amb, A = amb."""
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.standard_normal((N, N)).astype(np.float32), (0, 0), 1.5)
    lit = np.full((N, N), 0.02, np.float32)
    m = np.zeros((N, N), np.uint8)
    cv2.fillPoly(m, [np.array(CORNERS, np.int32)], 1)
    lit[m > 0] = (0.3 + 0.05 * t / t.std())[m > 0]
    if scratch:
        x0, y0, x1, y1 = SCRATCH
        lit[y0:y1, x0:x1] *= 0.4
    lit = np.repeat(lit[..., None], 3, -1) * (1.0 if leds else 0.0)
    a = np.full_like(lit, amb)
    fr = {"B1": 2 * lit + a, "L": lit + a, "A": a.copy(), "R": lit + a, "B2": 2 * lit + a}
    return {k: np.clip(v + rng.normal(0, 0.002, v.shape).astype(np.float32), -0.05, 1.0) for k, v in fr.items()}


def scratch_label() -> dict:
    x0, y0, x1, y1 = SCRATCH
    box = [grid_edge(x0) - 4, grid_edge(y0) - 4, grid_edge(x1) + 4, grid_edge(y1) + 4]   # padded like a YOLO box
    return {"id": "D1", "cls": "scratch", "type": "scratch", "kind": "scriber", "label": True, "visible": True,
            "box_px": box, "location_mm": {"x": 50.0, "y": 50.0},
            "size": {"length_mm": 56.25, "width_mm": 0.625}, "near_edge_mm": 21.9}


def write_npz(path: Path, amb: float = 0.01, seed: int = 0, scratch: bool = True, corners: bool = True,
              leds: bool = True) -> Path:
    """A burst file in twin/post.save_burst's layout (lightproc.save_burst + the extra arrays)."""
    labels = [scratch_label()] if scratch else []
    b = LP.Burst(plate_frames(amb, seed, scratch, leds),
                 {"material": "ms_cold_rolled", "sequence": list(LP.SEQUENCE),
                  "twin": {"schema": "trinetra.twin.burst/1", "scene_id": path.stem.split("_v")[0], "k": 0,
                           "split": "test", "condition": "C1_office", "device_mode": "open",
                           "part": {"material": "ms_cold_rolled", "thickness_mm": 3.0, "size_mm": [100, 100]}}})
    extra = dict(sequence=np.array(LP.SEQUENCE), plate_size_mm=np.array([100.0, 100.0]),
                 grid_plate_mm=np.array([100.0, 100.0]), corner_plate_idx=np.array([0, 1, 2, 3]),
                 condition=np.array("C1_office"), lux=np.float64(400.0), device_mode=np.array("open"),
                 material=np.array("ms_cold_rolled"), thickness_mm=np.float64(3.0),
                 labels_json=np.frombuffer(json.dumps(labels).encode(), np.uint8))
    if corners:
        extra["corners_px"] = np.array(CORNERS)
    path.parent.mkdir(parents=True, exist_ok=True)
    LP.save_burst(path, b, **extra)
    return path


def make_pipeline(tmp_path: Path, detector=None, advisor="mock", advise="sync", log=True) -> PL.InspectionPipeline:
    adv = advisor if isinstance(advisor, AD.AdvisorService) else (
        AD.AdvisorService.from_config({"backend": advisor}, SPEC) if advisor else None)
    return PL.InspectionPipeline(CFG["_capture"], CFG["_grid"], detector or PL.FakeDetector(), SPEC, CFG["part"], adv,
                                 log=PL.AppLog(tmp_path / "log") if log else None, advise=advise)


def wait_for(cond, timeout: float = 20.0, step: float = 0.05):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        v = cond()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timed out waiting")


# ---- sources ------------------------------------------------------------------------------------

def test_twin_source_reads_the_post_npz_layout(tmp_path):
    d = tmp_path / "bursts"
    write_npz(d / "s1_v0.npz")
    write_npz(d / "s2_v0.npz", seed=3)
    (d / "s3_v0.part.npz").write_bytes(b"partial")          # an interrupted write is ignored
    src = SR.TwinSource(d, loop=False)
    items = list(src)
    assert len(items) == 2 and len(src) == 2
    burst, meta = items[0]
    assert set(burst.frames) == set(LP.SEQUENCE)
    assert all(f.dtype == np.float32 and f.shape == (N, N, 3) for f in burst.frames.values())
    assert burst.meta["material"] == "ms_cold_rolled" and burst.meta["demosaiced"] is False
    assert meta["source"] == "twin" and meta["arm"] == "burst" and meta["part_id"] == "s1_v0"
    assert np.allclose(meta["corners_px"], CORNERS)
    assert meta["plate_size_mm"] == [100.0, 100.0] and meta["grid_plate_mm"] == [100.0, 100.0]
    assert meta["condition"] == "C1_office" and meta["lux"] == 400.0 and meta["device_mode"] == "open"
    assert meta["part"] == {"material": "ms_cold_rolled", "thickness_mm": 3.0, "size_mm": [100.0, 100.0]}
    assert meta["labels"][0]["type"] == "scratch" and meta["capture_ms"] >= 0
    looped = SR.TwinSource(d, loop=True)
    assert [looped.next()[1]["part_id"] for _ in range(3)] == ["s1_v0", "s2_v0", "s1_v0"]


def test_file_source_single_image_arm(tmp_path):
    d = tmp_path / "photos"
    d.mkdir()
    img = np.full((N, N, 3), 12, np.uint8)
    cv2.fillPoly(img, [np.array(CORNERS, np.int32)], (170, 172, 175))
    cv2.imwrite(str(d / "p1.png"), img)
    cv2.imwrite(str(d / "p2.png"), img)
    (d / "corners.json").write_text(json.dumps({"p1.png": {"corners_px": CORNERS, "material": "ss_2b"}}))
    pipe = make_pipeline(tmp_path, PL.FakeDetector(truth=False))
    for name in ("p1", "p2"):                               # p1: corners given; p2: found automatically
        item = SR.FileSource(d).next() if name == "p1" else list(SR.FileSource(d))[1]
        assert item[1]["arm"] == "single" and item[0].dtype == np.uint8
        rec = pipe.run(*item)
        assert RP.validate_record(rec) == []
        assert rec["capture"]["mode"] == "naive" and rec["capture"]["gated"] is False
        # open device: a clean part is never certified (hood-only policy, 22 Nov) -> HOLD_FOR_QA + note
        assert rec["capture"]["gate"] == "PASS" and rec["verdict"] == "HOLD_FOR_QA" and rec["defects"] == []
        assert "fit the hood" in rec["verdict_instruction"] and rec["capture"]["policy_note"]
    assert pipe.log.store.recent(5)[1][1] == "p1"
    first = pipe.log.store.records(5)[0]
    assert first["material"] == "ss_2b"
    pipe.close()


# ---- geometry and measures -----------------------------------------------------------------------

def test_grid_map_orientation_and_mm():
    tall = [[100.0, 100.0], [300.0, 100.0], [300.0, 400.0], [100.0, 400.0]]   # 150 x 100 plate, long side vertical
    gm = PL.grid_map(tall, [150, 100], CFG["_grid"])
    assert (gm.grid.plate_w_mm, gm.grid.plate_h_mm) == (100.0, 150.0)
    assert gm.size == (1375, 2000)
    assert PL.grid_map(tall, [150, 100], CFG["_grid"], corner_plate_idx=[1, 2, 3, 0]).grid.plate_w_mm == 100.0
    assert PL.grid_map(tall, [150, 100], CFG["_grid"], grid_plate_mm=[150, 100]).grid.plate_w_mm == 150.0
    # the imaged TL corner lands on the grid plate corner (62.5 px = 5 mm border), in any input order
    gm = PL.grid_map(CORNERS[::-1], [100, 100], CFG["_grid"])
    p = cv2.perspectiveTransform(np.array([[CORNERS[0]]], np.float32), gm.H)[0, 0]
    assert np.allclose(p, [62.5, 62.5], atol=1e-3)
    assert PL.box_to_mm(gm, [62.5 + 0.5, 62.5 + 0.5, 1312.5 + 0.5, 1312.5 + 0.5]) == pytest.approx((0, 0, 100, 100))


def _grid_image(seed: int = 0) -> tuple[np.ndarray, PL.GridMap]:
    rng = np.random.default_rng(seed)
    gm = PL.grid_map([[62.5, 62.5], [1312.5, 62.5], [1312.5, 1312.5], [62.5, 1312.5]], [100, 100], CFG["_grid"])
    g = np.clip(200 + rng.normal(0, 2, (1375, 1375, 3)), 0, 255).astype(np.uint8)
    return g, gm


def test_measures_come_from_the_defect_pixels():
    g, gm = _grid_image()
    plate = gm.grid.plate_mask()
    g[600:603, 300:550] = 90                               # scratch 3 px x 250 px = 0.24 x 20 mm
    size, ext, how = PL.measure_defect("scratch", [296, 596, 554, 607], g, gm, plate)
    assert how == "mask"
    assert size["length_mm"] == pytest.approx(np.hypot(258, 11) * 0.08)   # length: the box (the mark's extent)
    assert size["width_mm"] == pytest.approx(0.24, abs=0.05)               # width: the mark's own pixels
    assert ext == [300.0, 600.0, 550.0, 603.0]
    cv2.circle(g, (900, 900), 2, (60, 60, 60), -1)         # one 5 px pit (0.4 mm) in an 8 px-padded box
    size, _, how = PL.measure_defect("pit", [895, 895, 906, 906], g, gm, plate)
    assert how == "mask" and size["count"] == 1 and size["diameter_mm"] < 0.5
    for c in ((1000, 300), (1012, 305), (1006, 318), (995, 312)):   # a cluster of 4 pits
        cv2.circle(g, c, 2, (60, 60, 60), -1)
    size, _, _ = PL.measure_defect("pit", [985, 290, 1025, 330], g, gm, plate)
    assert size["count"] == 4 and size["pit_d_mm"] < 0.5 and size["diameter_mm"] > size["pit_d_mm"]
    size, _, how = PL.measure_defect("scratch", [700, 1100, 800, 1110], g, gm, plate)   # nothing there
    assert how == "box" and size["length_mm"] == pytest.approx(np.hypot(8.0, 0.8), abs=1e-6)
    size, _, how = PL.measure_defect("dent", [200, 200, 275, 250], g, gm, plate)
    assert how == "box" and size == {"diameter_mm": pytest.approx(6.0)}
    det = PL.T.Det(0, 0.8, 296, 596, 554, 607)
    d = PL.detection_record(det, g, gm, plate)
    assert d["type"] == "scratch" and d["location_mm"]["x"] == pytest.approx(424.5 * 0.08 - 5.0, abs=1e-6)
    assert d["near_edge_mm"] == pytest.approx((296 - 63) * 0.08)        # from the box edge (box edge 63 = 0 mm)


def test_near_edge_counts_known_holes(tmp_path):
    """The spec's +1 level applies within 5 mm of an edge OR HOLE: known holes count (twin burst in
    plate mm via H_frame_plate, or a jig's holes in grid-aligned mm)."""
    assert PL.box_gap_mm((0, 0, 10, 10), (15, 5, 4)) == pytest.approx(3.0)
    assert PL.box_gap_mm((0, 0, 10, 10), (5, 5, 4)) == 0.0
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["H_frame_plate"] = np.array([[3.2, 0, 80.0], [0, 3.2, 80.0], [0, 0, 1.0]])   # plate mm -> frame px
    arrays["holes_plate_mm"] = np.array([[50.0, 46.0, 4.0]])          # hole edge at y = 48 mm, above the scratch
    np.savez_compressed(path, **arrays)
    burst, meta = SR.TwinSource(path.parent, loop=False).next()
    pipe = make_pipeline(tmp_path, advisor=None)
    (d,) = pipe.run(burst, meta)["defects"]
    box_top_mm = (scratch_label()["box_px"][1] - 63.0) * 0.08
    assert d["near_edge_mm"] == pytest.approx(box_top_mm - 48.0, abs=0.02)
    g, gm = _grid_image()
    det = PL.T.Det(0, 0.8, 296, 596, 554, 607)                      # box x 18.6-39.3 mm, y 42.6-43.5 mm
    near = PL.detection_record(det, g, gm, gm.grid.plate_mask(), holes=[[30.0, 47.0, 2.0]])["near_edge_mm"]
    assert near == pytest.approx((47.0 - 1.0) - (607 - 63) * 0.08, abs=1e-6)
    pipe.close()


# ---- the pipeline ----------------------------------------------------------------------------------

def test_full_run_gives_a_3_0_record_timings_and_a_log_row(tmp_path):
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    pipe = make_pipeline(tmp_path)
    seen = []
    pipe.listeners.append(lambda ev, insp, **kw: seen.append((ev, None if insp is None else insp.record.get("advice"))))
    rec = pipe.run(*SR.TwinSource(path.parent, loop=False).next())

    assert RP.validate_record(rec) == []
    assert rec["schema"] == "trinetra.inspection/3.0" and rec["verdict"] in rules.RECORD_VERDICTS
    assert rec["capture"]["mode"] == "burst5" and rec["capture"]["gate"] in ("PASS", "PASS_FLAGGED")
    (d,) = rec["defects"]
    assert d["type"] == "scratch" and d["id"] == "D1" and d["measured_by"] == "mask"
    assert d["location_mm"] == {"x": pytest.approx(50.0, abs=0.3), "y": pytest.approx(50.0, abs=0.3)}
    assert d["size"]["length_mm"] == pytest.approx(56.25, abs=1.0)
    assert 0.4 < d["size"]["width_mm"] < 0.8
    lab = scratch_label()
    assert d["near_edge_mm"] == pytest.approx((lab["box_px"][0] - 63.0) * 0.08, abs=0.01)   # box edge: the safe side
    assert d["near_edge_mm"] <= lab["near_edge_mm"]
    assert rec["verdict"] == rec["source"]["truth"]["verdict"] != "RETAKE"
    t = rec["timing_ms"]
    for k in ("capture", "process", "rectify", "tile", "detect", "merge", "measure", "rules", "total", "advisor"):
        assert t[k] >= 0, k
    assert t["total"] >= t["process"] + t["capture"]
    assert rec["backend"] == {"vision": "fake (ground-truth replay)", "advisor": "mock-template"}
    assert rec["verdict"] in rec["advice"] and rec["advisor_stats"]["source"] == "template"
    assert seen[0] == ("inspection", None)                   # the alert went out before any advice
    assert [e for e, _ in seen] == ["inspection", "advice_start", "advice_done"]

    db = sqlite3.connect(tmp_path / "log" / "trinetra.db")
    (row,) = db.execute("SELECT part_id, verdict, defect_count, report_json FROM inspections").fetchall()
    assert row[:3] == ("s1_v0", rec["verdict"], 1) and json.loads(row[3])["advice"] == rec["advice"]
    assert db.execute("SELECT role FROM chats").fetchall() == [("assistant",)]
    lines = (tmp_path / "log" / "inspections.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["part_id"] == "s1_v0"
    assert json.loads((tmp_path / "log" / "advice.jsonl").read_text())["event"] == "explain"
    assert len(list((tmp_path / "log" / "images").glob("*.jpg"))) == 1
    assert pipe.last.jpeg[:2] == b"\xff\xd8"
    pipe.close()


def test_retake_burst_gives_retake_and_no_detection(tmp_path):
    path = write_npz(tmp_path / "b" / "sun_v0.npz", amb=0.9, seed=1)
    det = PL.FakeDetector()
    pipe = make_pipeline(tmp_path, det)
    rec = pipe.run(*SR.TwinSource(path.parent, loop=False).next())
    assert RP.validate_record(rec) == []
    assert rec["verdict"] == "RETAKE" and rec["defects"] == [] and det.calls == 0
    assert rec["capture"]["gate"] == "RETAKE" and "too_bright" in rec["capture"]["gate_reasons"]
    assert rec["capture"]["gate_message"] == CFG["_capture"]["gate"]["messages"]["too_bright"]
    assert "detect" not in rec["timing_ms"]
    assert rec["source"]["truth"]["verdict"] != "RETAKE"     # a refusal, never a silent pass
    assert "RETAKE" in rec["advice"] and rec["capture"]["gate_message"] in rec["advice"]
    pipe.close()


def test_no_plate_found_is_a_retake(tmp_path):
    path = write_npz(tmp_path / "b" / "dark_v0.npz", corners=False, leds=False)   # pods never lit: B1 = A
    rec = make_pipeline(tmp_path, advisor=None).run(*SR.TwinSource(path.parent, loop=False).next())
    assert rec["verdict"] == "RETAKE" and "no_plate" in rec["capture"]["gate_reasons"]
    assert rec["capture"]["gate_message"] == PL.NO_PLATE


def test_corners_found_on_the_led_image_when_not_given(tmp_path):
    path = write_npz(tmp_path / "b" / "s1_v0.npz", corners=False)
    burst, meta = SR.TwinSource(path.parent, loop=False).next()
    c = PL.auto_corners(burst)
    assert c is not None and np.allclose(PL.F.order_corners(c), CORNERS, atol=1.5)
    rec = make_pipeline(tmp_path, advisor=None).run(burst, meta)
    assert rec["verdict"] != "RETAKE" and len(rec["defects"]) == 1


class SlowBackend:
    name = "slow test model"

    def __init__(self, delay: float = 0.4, fail: bool = False):
        self.delay, self.fail = delay, fail

    def generate(self, messages, context=None):
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("model crashed")
        return "Follow the instruction on the screen."


def test_the_verdict_never_waits_for_the_advisor(tmp_path):
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    pipe = make_pipeline(tmp_path, advisor=AD.AdvisorService(SlowBackend(0.5), SPEC), advise="thread")
    t0 = time.perf_counter()
    rec = pipe.run(*SR.TwinSource(path.parent, loop=False).next())
    assert time.perf_counter() - t0 < 0.5 and rec["advice"] is None and rec["backend"]["advisor"] is None
    assert pipe.wait_advice(10.0)
    assert rec["advice"] and rec["timing_ms"]["advisor"] >= 400
    assert pipe.log.store.records(1)[0]["advice"] == rec["advice"]          # the DB row got the advice too
    pipe.close()


def test_advisor_failure_falls_back_and_keeps_the_record(tmp_path):
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    pipe = make_pipeline(tmp_path, advisor=AD.AdvisorService(SlowBackend(0.0, fail=True), SPEC))
    rec = pipe.run(*SR.TwinSource(path.parent, loop=False).next())
    assert rec["backend"]["advisor"] == "mock-template" and "unavailable" in rec["advice"]
    assert pipe.log.store.recent(1)[0][3] == rec["verdict"]
    pipe.close()


def test_advisor_error_is_never_credited_to_a_model(tmp_path):
    """When not even the template answers, the record must not name the model (or an earlier answer's
    backend) as the one that answered, and the UI still gets its advice_done."""
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    svc = AD.AdvisorService(SlowBackend(0.0), SPEC)
    pipe = make_pipeline(tmp_path, advisor=svc)
    insp = pipe.inspect(*SR.TwinSource(path.parent, loop=False).next())
    adv = svc._advisor(insp.record, insp.key)
    adv.last = {"backend": "slow test model"}                          # left from the earlier answer

    def boom():
        raise RuntimeError("template crashed")

    adv.ask = lambda q: boom()
    res = pipe.ask(insp, "why")
    assert res.error == "template crashed" and res.backend == AD.NO_ANSWER and "Advisor failed" in res.text
    bad = svc.explain({"schema": "trinetra.inspection/3.0"}, key="bad")  # a record the prompt builder cannot use
    assert bad.error and bad.backend == AD.NO_ANSWER
    pipe.close()


# ---- the Llama runner ---------------------------------------------------------------------------------

RUNNER = textwrap.dedent('''
    import json, sys, time
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if args.get("model_path", "").endswith("broken.pte"):
        sys.stderr.write("failed to load model"); sys.exit(1)
    sys.stdout.write(args["prompt"]); sys.stdout.flush()
    for tok in ["parts ", "sliding ", "on ", "a ", "dirty ", "bench", "."]:
        time.sleep(0.03); sys.stdout.write(tok); sys.stdout.flush()
    sys.stdout.write("<|eot_id|>\\n")
    print("PyTorchObserver " + json.dumps({"prompt_tokens": 300, "generated_tokens": 7, "model_load_start_ms": 1000,
          "model_load_end_ms": 2500, "inference_start_ms": 3000, "prompt_eval_end_ms": 6000, "first_token_ms": 6100,
          "inference_end_ms": 7400, "SCALING_FACTOR_UNITS_PER_SECOND": 1000}))
''')


def runner_service(tmp_path: Path, model: str = "m.pte") -> AD.AdvisorService:
    script = tmp_path / "fake_llama_main.py"
    script.write_text(RUNNER)
    for name in (model, "tok.model"):
        (tmp_path / name).write_bytes(b"")
    cfg = {**CFG["llm"], "backend": "executorch", "runner": f"{sys.executable} {script}", "model_path": model,
           "tokenizer_path": "tok.model", "timeout_s": 30}
    return AD.AdvisorService.from_config(cfg, SPEC, root=tmp_path, fallback=False)


def test_streaming_runner_measures_ttft_and_tokens_per_s(tmp_path):
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    pipe = make_pipeline(tmp_path, advisor=None)
    insp = pipe.inspect(*SR.TwinSource(path.parent, loop=False).next())
    svc = runner_service(tmp_path)
    assert isinstance(svc.backend, AD.StreamingRunner)
    parts: list[str] = []
    res = svc.explain(insp.record, key=insp.key, on_text=parts.append)
    assert res.text.startswith("Verdict: HOLD_FOR_QA. Decided by D1 scratch") and res.backend == svc.name
    assert res.text.endswith(CAUSE_LABEL + "The scratch most likely comes from parts sliding on a dirty bench.") and not res.problems
    st = res.stats
    assert st["source"] == "runner_stats" and st["ttft_ms"] == 3100.0 and st["total_ms"] == 4400.0
    assert st["decode_tokens_per_s"] == 5.0 and st["prefill_tokens_per_s"] == 100.0 and st["load_ms"] == 1500.0
    assert st["wall_ttft_ms"] is not None and st["wall_ms"] >= st["wall_ttft_ms"]
    assert len(parts) >= 3 and parts[-1] == res.text and all("INSPECTION" not in p for p in parts)
    follow = svc.ask(insp.record, "why", key=insp.key)
    assert follow.question == AD.QUESTIONS["why"] and follow.text == "parts sliding on a dirty bench."   # no header on follow-ups
    broken = runner_service(tmp_path, "broken.pte").explain(insp.record)
    assert broken.backend == "mock-template" and "unavailable" in broken.text


def test_observer_stats_units_and_missing_line():
    s = AD.observer_stats({"prompt_tokens": 10, "generated_tokens": 20, "inference_start_ms": 0,
                           "prompt_eval_end_ms": 500_000, "first_token_ms": 600_000, "inference_end_ms": 2_500_000,
                           "SCALING_FACTOR_UNITS_PER_SECOND": 1_000_000})       # microsecond clock
    assert s["ttft_ms"] == 600.0 and s["prefill_tokens_per_s"] == 20.0 and s["decode_tokens_per_s"] == 10.0
    assert AD.parse_observer("no stats here") == {}
    assert AD.answer_part("PROMPTanswer<|eot_id|>junk", "PROMPT") == "answer"
    assert AD.answer_part("PROM", "PROMPT") is None


def test_streaming_runner_keeps_the_prompt_budget(tmp_path):
    """llm.Advisor sizes its prompt from backend.seq_len; the streaming wrapper must pass it on, or
    follow-up questions grow past the runner's context (and every answer falls back)."""
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    insp = make_pipeline(tmp_path, advisor=None).inspect(*SR.TwinSource(path.parent, loop=False).next())
    svc = runner_service(tmp_path)
    assert svc.backend.seq_len == CFG["llm"]["seq_len"]
    adv = svc._advisor(insp.record, insp.key)
    assert adv.max_prompt_bytes == int((CFG["llm"]["seq_len"] - AD.llm.ANSWER_TOKENS) * AD.llm.PROMPT_BYTES_PER_TOKEN)
    small = AD.StreamingRunner(AD.llm.ExecuTorchBackend("x", "m.pte", "t.model", seq_len=AD.llm.ANSWER_TOKENS + 1))
    adv = AD.llm.Advisor(small, SPEC, insp.record, history_turns=3)
    adv.turns = [{"role": r, "content": "x" * 400} for _ in range(3) for r in ("user", "assistant")]
    msgs = adv._messages("why?")
    assert [m["role"] for m in msgs] == ["system", "user", "user"]          # old turns dropped to fit


class FakeConfig:
    def __init__(self):
        self.echo, self.max_new_tokens, self.warming, self.seq_len = True, -1, False, -1
        self.temperature, self.num_bos, self.num_eos = 0.8, 0, 0


class FakeStats:
    def to_json_string(self):
        return json.dumps({"prompt_tokens": 200, "generated_tokens": 4, "model_load_start_ms": 0,
                           "model_load_end_ms": 0, "inference_start_ms": 1000, "prompt_eval_end_ms": 2000,
                           "first_token_ms": 2100, "inference_end_ms": 2500, "SCALING_FACTOR_UNITS_PER_SECOND": 1000})


class FakeTextLLMRunner:
    last = None

    def __init__(self, model_path, tokenizer_path, data_path=None):
        self.paths, self.resets, self.prompts, self.configs = (model_path, tokenizer_path), 0, [], []
        FakeTextLLMRunner.last = self

    def reset(self):
        self.resets += 1

    def generate(self, prompt, config, token_callback, stats_callback):
        self.prompts.append(prompt)
        self.configs.append(dict(vars(config)))
        for tok in ("Hold ", "it ", "for ", "QA."):
            token_callback(tok)
        stats_callback(FakeStats())


def test_in_process_textllmrunner_backend(monkeypatch, tmp_path):
    """executorch_py against the documented TextLLMRunner API (a fake module: executorch is not
    installed on the Mac): reset per answer, own BOS, num_bos 0, streaming, stats, prompt budget."""
    mod = types.SimpleNamespace(TextLLMRunner=FakeTextLLMRunner, GenerationConfig=FakeConfig)
    for name in ("executorch", "executorch.extension", "executorch.extension.llm"):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "executorch.extension.llm.runner", mod)
    svc = AD.AdvisorService.from_config({**CFG["llm"], "backend": "executorch_py"}, SPEC, fallback=False)
    be, fr = svc.backend, FakeTextLLMRunner.last
    assert isinstance(be, AD.PyRunnerBackend) and fr.paths[0].endswith(CFG["llm"]["py"]["model_path"])
    assert be.seq_len == CFG["llm"]["seq_len"] and len(fr.prompts) == 1      # context for the budget; warm-up ran
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    insp = make_pipeline(tmp_path, advisor=None).inspect(*SR.TwinSource(path.parent, loop=False).next())
    parts: list[str] = []
    res = svc.explain(insp.record, key=insp.key, on_text=parts.append)
    assert parts[-1].startswith("Verdict: ") and parts[-1].endswith(CAUSE_LABEL + "The scratch most likely comes from Hold it for QA.") and fr.resets == 2   # warm-up, answer
    assert fr.prompts[-1].startswith("<|begin_of_text|>") and fr.prompts[-1].count("<|begin_of_text|>") == 1
    c = fr.configs[-1]
    assert c["num_bos"] == 0 and c["echo"] is False and c["max_new_tokens"] == CFG["llm"]["py"]["max_new_tokens"]
    st = res.stats
    assert st["ttft_ms"] == 1100.0 and st["decode_tokens_per_s"] == 8.0 and st["prefill_tokens_per_s"] == 200.0
    assert svc._advisor(insp.record, insp.key).max_prompt_bytes is not None


def test_unavailable_backend_degrades_to_the_template(tmp_path):
    svc = AD.AdvisorService.from_config({"backend": "executorch_py", "py": {"model_path": "/nonexistent.pte",
                                                                          "tokenizer_path": "/nonexistent"}}, SPEC)
    assert isinstance(svc.backend, AD.llm.MockBackend) and "executorch_py unavailable" in svc.degraded
    # llama_main runner or model files missing: one warning at start-up, not a fallback on every answer
    svc = AD.AdvisorService.from_config({**CFG["llm"], "backend": "executorch", "runner": str(tmp_path / "llama_main")},
                                        SPEC, root=tmp_path)
    assert isinstance(svc.backend, AD.llm.MockBackend)
    assert "runner not found" in svc.degraded and "model_path not found" in svc.degraded
    with pytest.raises(Exception):
        AD.AdvisorService.from_config({"backend": "nope"}, SPEC, fallback=False)


# ---- the Pi camera (fakes) ------------------------------------------------------------------------------

class FakeLED:
    state: dict[int, bool] = {}
    made: list = []

    def __init__(self, pin, active_high=True, initial_value=False):
        self.pin, self.closed = pin, False
        FakeLED.state[pin] = bool(initial_value)
        FakeLED.made.append(self)

    def on(self):
        FakeLED.state[self.pin] = True

    def off(self):
        FakeLED.state[self.pin] = False

    def close(self):
        self.closed = True


class FakeRequest:
    def __init__(self, raw, md):
        self.raw, self.md, self.released = raw, md, False

    def make_array(self, name):
        assert name == "raw"
        return self.raw

    def get_metadata(self):
        return self.md

    def release(self):
        self.released = True


class FakeCam:
    last = None

    def __init__(self, index):
        self.index, self.flushes, self.leds, self.ts, self.closed = index, [], [], 0, False
        FakeCam.last = self

    def create_still_configuration(self, **kw):
        self.config_kw = kw
        return kw

    def configure(self, conf):
        self.conf = conf

    def camera_configuration(self):
        return {"raw": {"format": "SRGGB10", "size": (64, 48)}}

    def set_controls(self, controls):
        self.controls = controls

    def start(self):
        pass

    def capture_request(self, flush=False):
        lit = (FakeLED.state.get(17, False), FakeLED.state.get(27, False))
        self.flushes.append(flush)
        self.leds.append(tuple(int(x) for x in lit))
        dn = 64 + 200 * lit[0] + 100 * lit[1]                        # 10-bit DN, uniform mosaic
        raw16 = np.full((48, 64 + 16), dn << 6, np.uint16)          # MSB-aligned, with stride padding
        self.ts += 100_000
        gain = float((getattr(self, "controls", None) or {}).get("AnalogueGain", 1.0))   # the gain it was set to
        return FakeRequest(raw16.view(np.uint8), {"ExposureTime": 10000, "AnalogueGain": gain,
                                                  "SensorTimestamp": self.ts, "SensorBlackLevels": [4096] * 4})

    def stop(self):
        pass

    def close(self):
        self.closed = True


def test_camera_source_burst_with_fake_picamera2(monkeypatch):
    FakeLED.state = {}
    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=FakeCam))
    monkeypatch.setitem(sys.modules, "gpiozero", types.SimpleNamespace(DigitalOutputDevice=FakeLED))
    cam = SR.CameraSource(CFG["_capture"], CFG["camera"], CFG["pods"], CFG["part"])
    fc = FakeCam.last
    assert fc.config_kw["raw"]["format"] == "SRGGB10" and cam.pattern == "RGGB"
    c = fc.controls
    sensor = CFG["_capture"]["sensor"]
    assert c["AeEnable"] is False and c["AwbEnable"] is False and c["ExposureTime"] == sensor["exposure_us"]
    assert c["AnalogueGain"] == sensor["analogue_gain"] and "NoiseReductionMode" in c and "ColourGains" in c
    assert c["FrameDurationLimits"][0] == c["FrameDurationLimits"][1] >= sensor["exposure_us"]
    n_warm = len(fc.flushes)
    burst, meta = cam.next()
    states = CFG["_capture"]["burst"]["led_state"]
    seq = [tuple(states[k]) for k in LP.SEQUENCE]
    assert fc.leds[n_warm:] == seq + seq          # pods switched before every frame; two bursts (P1h)
    partner, pmeta = meta["partner"]
    assert set(partner.frames) == set(LP.SEQUENCE) and pmeta["camera"] is not meta["camera"]
    assert all(fc.flushes)                                                # capture_request(flush=True) each time
    assert set(burst.frames) == set(LP.SEQUENCE) and burst.meta["demosaiced"] is True
    assert burst.meta["illumination"] == "pods"                           # open: pods (hood: the ring, v2)
    hood = SR.CameraSource(CFG["_capture"], CFG["camera"], CFG["pods"], {**CFG["part"], "device_mode": "hood"})
    assert hood.illumination == "ring"
    # one exposure for both heads; the ring's lower irradiance is taken up by its own analogue gain
    # (capture.yaml sensor.by_illumination, the same value the twin simulates: review 15 Nov)
    assert FakeCam.last.controls["AnalogueGain"] == LP.sensor_for(sensor, "ring")["analogue_gain"] == 1.5
    assert FakeCam.last.controls["ExposureTime"] == sensor["exposure_us"]
    fs = 1023 - sensor["black_level"]
    for k, (l, r) in states.items():
        f = burst.frames[k]
        assert f.shape == (48, 64, 3) and f.dtype == np.float32
        assert f.mean() == pytest.approx((200 * l + 100 * r) / fs, abs=1e-4), k
    assert not any(FakeLED.state.values())                               # pods off after the burst
    assert meta["source"] == "camera" and meta["part_id"].startswith("cam-") and meta["camera"]["warnings"] == []
    assert meta["camera"]["left_shifted"] is True
    cam.close()
    assert fc.closed and all(p.closed for p in cam.pods.values())


class CompressedCam(FakeCam):
    def camera_configuration(self):
        return {"raw": {"format": "BGGR16_PISP_COMP1", "size": (64, 48)}}   # Pi 5 default raw: compressed


def test_camera_source_refuses_packed_or_compressed_raw(monkeypatch):
    FakeLED.state = {}
    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=CompressedCam))
    monkeypatch.setitem(sys.modules, "gpiozero", types.SimpleNamespace(DigitalOutputDevice=FakeLED))
    FakeLED.made = []
    with pytest.raises(RuntimeError, match="compressed"):        # it would demosaic to garbage, silently
        SR.CameraSource(CFG["_capture"], CFG["camera"], CFG["pods"], CFG["part"])
    assert FakeLED.made and all(p.closed for p in FakeLED.made) and FakeCam.last.closed   # pins and camera freed
    with pytest.raises(ValueError):
        SR.CameraSource(CFG["_capture"], {**CFG["camera"], "raw_format": "SRGGB10_CSI2P"}, CFG["pods"], CFG["part"])
    assert SR.packed_or_compressed("BGGR_PISP_COMP1") and not SR.packed_or_compressed("SRGGB16")
    assert SR.bayer_pattern("SRGGB16") == "RGGB"


def test_bayer_pattern_and_raw_view():
    assert SR.bayer_pattern("SRGGB10") == "RGGB" and SR.bayer_pattern("SBGGR10") == "BGGR"
    assert SR.bayer_pattern("GRBG") == "GRBG"
    with pytest.raises(ValueError):
        SR.bayer_pattern("YUV420")
    raw = np.arange(12, dtype=np.uint16).reshape(2, 6)
    assert np.array_equal(SR.raw_u16(raw.view(np.uint8), 4), raw[:, :4])


# ---- UI ---------------------------------------------------------------------------------------------------

def _http(url: str, data: dict | None = None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST")
    with opener.open(req, timeout=10) as r:
        body = r.read()
        return body if r.headers.get("Content-Type", "").startswith("image/") else (
            json.loads(body) if "json" in r.headers.get("Content-Type", "") else body.decode())


def test_ui_server_serves_the_station(tmp_path):
    path = write_npz(tmp_path / "b" / "s1_v0.npz")
    pipe = make_pipeline(tmp_path, advise="thread")
    station = UI.Station(pipe, SR.TwinSource(path.parent, loop=True))
    srv, _ = UI.serve_in_thread(station, port=0)
    base = UI.url_of(srv)
    try:
        page = _http(base)
        assert "<html" in page and "Inspect" in page and "Shift summary" in page
        st = _http(base + "api/latest")
        assert st["inspection"] is None and "network" in st and st["vision"] == "fake (ground-truth replay)"
        assert _http(base + "api/inspect", {}) == {"queued": True}
        st = wait_for(lambda: (s := _http(base + "api/latest"))["inspection"] and s["advice"]["status"] == "done" and s)
        i = st["inspection"]
        assert i["verdict"] == pipe.last.record["verdict"] and i["colour"].startswith("#")
        assert i["defects"][0]["type"] == "scratch" and "Gate" in i["capture_line"]
        assert st["acked"] is (i["verdict"] not in UI.ALERT_VERDICTS)
        assert _http(base + "api/image.jpg")[:2] == b"\xff\xd8"
        assert _http(base + "api/record")["schema"] == "trinetra.inspection/3.0"
        assert _http(base + "api/ask", {"kind": "why"}) == {"queued": True}
        st = wait_for(lambda: (s := _http(base + "api/latest"))["advice"]["kind"] == "ask"
                      and s["advice"]["status"] == "done" and s)
        assert st["advice"]["question"] == AD.QUESTIONS["why"] and st["advice"]["text"]
        assert _http(base + "api/ask", {"kind": "shift"}) == {"queued": True}
        wait_for(lambda: (s := _http(base + "api/latest"))["advice"]["kind"] == "shift" and s["advice"]["status"] == "done")
        assert _http(base + "api/ack", {}) == {"acknowledged": True} and _http(base + "api/latest")["acked"]
        assert len(_http(base + "api/history")) == 1
    finally:
        srv.shutdown()
        srv.server_close()
        station.close()
        pipe.close()


def test_station_never_shows_one_parts_advice_under_another(tmp_path):
    """An answer still streaming when the next part is inspected must not land on the new part."""
    pipe = make_pipeline(tmp_path, advisor="mock", advise="thread", log=False)
    st = UI.Station(pipe, None)
    try:
        r1 = {"part_id": "p1", "verdict": "REJECT"}
        i1, i2 = PL.Inspection(r1, b"", {}, 1), PL.Inspection({"part_id": "p2", "verdict": "PASS"}, b"", {}, 2)
        done = AD.AdviceResult("explain", "Scrap p1.", "m", "m")
        st._on_event("inspection", i1)
        st._on_event("advice_start", i1, kind="explain")
        st._on_event("advice_text", i1, text="Scrap")
        st._on_event("inspection", i2)                                   # next part while p1's answer streams
        st._on_event("advice_text", i1, text="Scrap p1")
        st._on_event("advice_done", i1, result=done)
        assert st.advice["for"] == 2 and st.advice["text"] == "" and st.advice["status"] == "waiting"
        st._on_event("advice_start", i1, kind="ask")                     # a queued job about p1: not shown
        st._on_event("advice_done", i1, result=done)
        assert st.advice["text"] == ""
        st._on_event("advice_start", i2, kind="explain")
        st._on_event("advice_text", i2, text="Pass")
        st._on_event("advice_done", i2, result=AD.AdviceResult("explain", "Pass p2.", "m", "m"))
        assert st.advice["text"] == "Pass p2." and st.advice["status"] == "done"
        st._on_event("advice_start", None, kind="shift")                 # the shift summary is shown too
        st._on_event("advice_done", None, result=AD.AdviceResult("shift", "Shift.", "m", "m"))
        assert st.advice["text"] == "Shift."
    finally:
        st.close()
        pipe.close()


def test_stale_explanations_are_skipped_not_queued(tmp_path):
    """Llama slower than the inspection rate: explain only the latest part, log the skipped ones."""
    d = tmp_path / "b"
    for i in range(4):
        write_npz(d / f"s{i}_v0.npz", seed=i)
    slow = SlowBackend(0.8)
    calls: list[str] = []
    gen = slow.generate
    slow.generate = lambda m, c=None: (calls.append(c["report"]["part_id"]), gen(m, c))[1]
    pipe = make_pipeline(tmp_path, advisor=AD.AdvisorService(slow, SPEC), advise="thread")
    recs = [pipe.run(*item) for item in SR.TwinSource(d, loop=False)]
    assert pipe.wait_advice(20.0)
    assert calls[0] == "s0_v0" and calls[-1] == "s3_v0" and len(set(calls)) < 4   # a part may get a guarded rewrite
    assert recs[-1]["advice"] and not recs[1]["advice"]
    events = [json.loads(x) for x in (tmp_path / "log" / "advice.jsonl").read_text().splitlines()]
    skipped = [e["part_id"] for e in events if e["event"] == "explain_skipped"]
    assert len(skipped) == 4 - len(set(calls)) and "s3_v0" not in skipped
    pipe.close()


def test_page_is_self_contained_and_offline_check():
    assert "http://" not in UI.PAGE and "https://" not in UI.PAGE and "<script src" not in UI.PAGE
    head = "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\n"
    assert UI._default_route_linux(head + "eth0\t00000000\t0102A8C0\t0003\t0\t0\t0\t00000000\n")
    assert not UI._default_route_linux(head + "eth0\t0002A8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\n")
    assert UI.network_state()["detail"]


def test_app_imports_no_network_client():
    code = ("import sys, json; import trinetra.app.cli, trinetra.app.ui, trinetra.app.bench; "
            "bad = ('requests', 'urllib3', 'httpx', 'aiohttp', 'websockets', 'paramiko', 'boto3', 'ftplib', 'smtplib'); "
            "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in bad or m == 'urllib.request')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == []


# ---- bench and CLI ------------------------------------------------------------------------------------------

def test_bench_writes_a_report(tmp_path):
    d = tmp_path / "b"
    write_npz(d / "s1_v0.npz")
    write_npz(d / "s2_v0.npz", amb=0.9, seed=1)
    pipe = make_pipeline(tmp_path, advise="thread")
    rep = BE.run_bench(pipe, SR.TwinSource(d, loop=True), 3, warmup=1, out_dir=tmp_path / "reports", echo=False,
                       config={"source": "twin", "detector": "fake", "advisor": "mock"})
    j, m = Path(rep["json"]), Path(rep["md"])
    assert j.exists() and m.exists() and j.name.startswith("bench_") and pipe.advise == "thread"
    data = json.loads(j.read_text())
    assert data["n"] == 3 and data["warmup"] == 1 and len(data["runs"]) == 3
    assert data["vision"]["total"]["p50"] > 0 and data["vision"]["process"]["n"] == 3
    assert data["fps"]["vision"] > 0 and data["memory_mb"]["peak_rss"] > 0
    assert data["llm"]["n"] == 3 and sum(data["llm"]["answered_by"].values()) == 3
    assert set(data["llm"]["answered_by"]) <= {"mock-template", "rules-engine"}   # a RETAKE needs no model
    assert set(data["gates"]) <= {"PASS", "PASS_FLAGGED", "RETAKE"} and "RETAKE" in data["verdicts"]
    md = m.read_text()
    assert "| process |" in md and "FPS (vision" in md and "Peak RSS" in md
    assert len(pipe.log.store.recent(10)) == 4                           # warm-up + 3, all logged
    pipe.close()


def test_cli_console_run_and_bench(tmp_path, capsys):
    d = tmp_path / "b"
    write_npz(d / "s1_v0.npz")
    write_npz(d / "s2_v0.npz", amb=0.9, seed=1)
    log = tmp_path / "log"
    assert CLI.main(["run", "--source", f"twin:{d}", "--detector", "fake", "--advisor", "mock",
                     "--log-dir", str(log), "--quiet"]) == 0
    lines = (log / "inspections.jsonl").read_text().splitlines()
    assert [json.loads(x)["part_id"] for x in lines] == ["s1_v0", "s2_v0"]
    assert all(json.loads(x)["schema"] == "trinetra.inspection/3.0" for x in lines)
    assert CLI.main(["run", "--source", f"twin:{d}", "--detector", "fake", "--advisor", "mock", "--log-dir",
                     str(log), "--bench", "2", "--warmup", "0", "--reports", str(tmp_path / "rep"), "--quiet"]) == 0
    assert len(list((tmp_path / "rep").glob("bench_*.json"))) == 1
    assert "wrote" in capsys.readouterr().out


def test_reads_bursts_written_by_post_save_burst(tmp_path):
    """Contract check with the twin's own writer (twin/post.save_burst + twin/burst.simulate_burst)."""
    from trinetra.twin import burst as BU
    from trinetra.twin import post as P

    if not (hasattr(P, "save_burst") and hasattr(BU, "toy_scene")):
        pytest.skip("twin post.save_burst / burst.toy_scene not available")
    h = w = 400
    m = 20
    s = (w - 2 * m) / 100.0
    H = np.array([[s, 0, m - 0.5], [0, s, m - 0.5], [0, 0, 1.0]])       # plate mm -> frame px
    geom = P.grid_geometry(H, CFG["_grid"], [100.0, 100.0], roi=[0, 0, w, h])
    d = tmp_path / "bursts"
    for r in (0.0, 4.0):
        g, plan, roi, e = BU.toy_scene(h, w, r=r, seed=1, margin_px=m)
        b = BU.simulate_burst(g, plan, np.random.default_rng(2), CFG["_capture"], roi, e_led=e)
        P.save_burst(d / f"toy_r{r:g}_v0.npz", b, geom, [],
                     {"scene_id": f"toy_r{r:g}", "k": 0, "condition": "C6_sun" if r else "C0_dark", "lux": 0.0,
                      "device_mode": "open", "part": {"material": "ss_2b", "thickness_mm": 2.0, "size_mm": [100, 100]}})
    pipe = make_pipeline(tmp_path, PL.FakeDetector(truth=False), advisor=None)
    recs = {meta["part_id"]: pipe.run(burst, meta) for burst, meta in SR.TwinSource(d, loop=False)}
    dark, sun = recs["toy_r0_v0"], recs["toy_r4_v0"]
    assert dark["capture"]["gate"] == "PASS" and dark["material"] == "ss_2b"
    assert dark["verdict"] == "HOLD_FOR_QA" and "fit the hood" in dark["verdict_instruction"]   # open: never PASS
    assert dark["thickness_mm"] == 2.0 and dark["source"]["condition"] == "C0_dark"
    assert sun["verdict"] == "RETAKE" and sun["capture"]["gate"] == "RETAKE"
    assert all(RP.validate_record(r) == [] for r in recs.values())
    pipe.close()
