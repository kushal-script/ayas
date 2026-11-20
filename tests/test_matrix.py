import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from trinetra.twin import matrix as MX
from trinetra.twin import sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_demo_video as V  # noqa: E402

CONDS = ("C0_dark", "C1_office", "C4_side_lamp", "C6_sun")


def fake_make_plan(cfg, seed, split, mode=None, source=None, coupons=None, condition=None, device_mode=None):
    """Stands in for sampler.make_plan: the plate and defects depend on the seed only (per-part streams)."""
    rng = np.random.default_rng(seed)
    mode_ = device_mode or "open"
    return {"schema": "trinetra.twin.scene/2", "scene_id": sampler.scene_id(seed), "seed": seed, "split": split,
            "plate": {"size_mm": [100.0, 100.0], "thickness_mm": float(rng.uniform(1.5, 6)),
                      "material": {"kind": "ms_hot_rolled"}},
            "defects": [{"id": "D1", "cls": "pit", "kind": "single", "params": {"u": float(rng.uniform(10, 90))},
                         "label": True}],
            "distractors": [{"kind": "dust", "params": {"u": float(rng.uniform(10, 90))}}],
            "ambient": {"condition": condition or "C1_office", "lux": 400.0},
            "extras": {"device": {"mode": mode_}, "hood": {"gap_mm": 0.5} if mode_ == "hood" else None},
            "camera": {"pose": {"tilt_deg": [0.0, 0.0] if mode_ == "hood" else [5.0, -3.0]}}}


@pytest.fixture()
def fake_sampler(monkeypatch):
    monkeypatch.setattr(MX.sampler, "make_plan", fake_make_plan)


def test_rows_and_tags():
    rows = MX.rows_for(CONDS, ("C4_side_lamp", "C6_sun"))
    assert rows == [(c, "open") for c in CONDS] + [("C4_side_lamp", "hood"), ("C6_sun", "hood")]
    assert [MX.row_tag(c, m) for c, m in rows] == ["C0o", "C1o", "C4o", "C6o", "C4h", "C6h"]
    with pytest.raises(ValueError):
        MX.rows_for(("C0_dark",), ("C6_sun",))              # hood row for a condition not in the matrix
    with pytest.raises(ValueError):
        MX.row_tag("C0_dark", "closed")
    assert MX.ALL[0] == "C0_dark" and "C6_sun" in MX.ALL and len(MX.ALL) == 7


def test_matrix_rows_share_plate_and_defects(tmp_path, fake_sampler):
    man = MX.write_matrix_plans({}, 3, tmp_path, CONDS, ("C4_side_lamp", "C6_sun"))
    assert man["schema"] == MX.SCHEMA and man["start_seed"] == 900000 and man["split"] == "test"
    assert [p["seed"] for p in man["plates"]] == [900000, 900001, 900002]
    assert json.loads((tmp_path / MX.MANIFEST).read_text()) == man
    pl = man["plates"][0]
    assert pl["plate"] == "s900000" and pl["part"] == {"material": "ms_hot_rolled",
                                                        "thickness_mm": pytest.approx(pl["part"]["thickness_mm"]),
                                                        "size_mm": [100.0, 100.0]}
    assert not pl["clean"] and pl["defects"] == [{"id": "D1", "cls": "pit", "kind": "single", "label": True}]
    assert [r["scene_id"] for r in pl["rows"]] == ["s900000_C0o", "s900000_C1o", "s900000_C4o", "s900000_C6o",
                                                   "s900000_C4h", "s900000_C6h"]
    plans = [json.loads((tmp_path / r["plan"]).read_text()) for r in pl["rows"]]
    for p, r in zip(plans, pl["rows"]):
        assert p["scene_id"] == r["scene_id"] and p["split"] == "test" and p["seed"] == 900000
        assert p["ambient"]["condition"] == r["condition"] and p["extras"]["device"]["mode"] == r["device_mode"]
        assert p["matrix"] == {"plate": "s900000", "plate_seed": 900000, "condition": r["condition"],
                               "device_mode": r["device_mode"], "row": r["tag"]}
        for k in ("plate", "defects", "distractors"):
            assert p[k] == plans[0][k]
    assert plans[-1]["camera"] != plans[0]["camera"]               # the hood forces a straight pose
    other = json.loads((tmp_path / man["plates"][1]["rows"][0]["plan"]).read_text())
    assert other["defects"] != plans[0]["defects"]                  # different plates differ
    assert len(list((tmp_path / "plans").glob("*.json"))) == 3 * 6
    idx = MX.scene_index(MX.load_manifest(tmp_path))
    assert idx["s900001_C6h"] == {"plate": "s900001", "seed": 900001, "part": man["plates"][1]["part"],
                                  "condition": "C6_sun", "device_mode": "hood", "tag": "C6h"}
    assert len(MX.plan_paths(tmp_path)) == 18 and all(p.exists() for p in MX.plan_paths(tmp_path))


def test_matrix_never_uses_training_seeds(tmp_path, fake_sampler):
    with pytest.raises(ValueError, match="training"):
        MX.write_matrix_plans({}, 2, tmp_path, CONDS, start_seed=899999)
    with pytest.raises(ValueError):
        MX.matrix_plans({}, 12, [("C0_dark", "open")])
    assert MX.is_matrix_seed(900000) and not MX.is_matrix_seed(899999)


def test_matrix_refuses_a_sampler_that_breaks_the_contract(tmp_path, monkeypatch):
    def ignores(cfg, seed, split, **kw):
        return fake_make_plan(cfg, seed, split)                          # drops condition/device_mode

    monkeypatch.setattr(MX.sampler, "make_plan", ignores)
    with pytest.raises(ValueError, match="ignored the override"):
        MX.write_matrix_plans({}, 1, tmp_path, ("C0_dark", "C6_sun"), ("C6_sun",))

    def reshuffles(cfg, seed, split, condition=None, device_mode=None, **kw):
        p = fake_make_plan(cfg, seed, split, condition=condition, device_mode=device_mode)
        p["defects"][0]["params"]["u"] += len(condition)                 # defects move with the light
        return p

    monkeypatch.setattr(MX.sampler, "make_plan", reshuffles)
    with pytest.raises(ValueError, match="defects"):
        MX.write_matrix_plans({}, 1, tmp_path, ("C0_dark", "C6_sun"), ())


def test_matrix_with_the_real_sampler(tmp_path):
    params = inspect.signature(sampler.make_plan).parameters
    if "condition" not in params or "device_mode" not in params:
        pytest.skip("sampler.make_plan has no condition/device_mode overrides yet (steel-twin workstream)")
    cfg = yaml.safe_load((ROOT / "configs" / "twin.yaml").read_text())
    cfg["root"] = str(ROOT)
    man = MX.write_matrix_plans(cfg, 1, tmp_path, ("C0_dark", "C6_sun"), ("C6_sun",))
    plans = [json.loads((tmp_path / r["plan"]).read_text()) for r in man["plates"][0]["rows"]]
    assert [p["matrix"]["row"] for p in plans] == ["C0o", "C6o", "C6h"]
    assert all(p["defects"] == plans[0]["defects"] and p["plate"] == plans[0]["plate"] for p in plans)


# --- Demo video ----------------------------------------------------------------------------

def tiny_png(path: Path, colour, size=(90, 120)) -> Path:
    img = np.zeros((*size, 3), np.uint8)
    img[:] = colour
    cv2.rectangle(img, (20, 20), (50, 40), (255, 255, 255), -1)
    cv2.imwrite(str(path), img)
    return path


def capture(arm, cond, mode, image, pred, gt, gate="PASS", msg="", n_gt=1):
    return {"arm": arm, "model": arm, "plate": "s900000", "scene": f"s900000_{MX.row_tag(cond, mode)}", "k": 0,
            "condition": cond, "device_mode": mode, "tag": MX.row_tag(cond, mode), "image": str(image),
            "gate": gate, "gate_message": msg, "lux": 52000.0 if cond == "C6_sun" else 300.0,
            "pred_verdict": pred, "gt_verdict": gt, "dets": [[1, 0.9, 20, 20, 50, 40]] if pred != "PASS" else [],
            "gt": [[1, 20, 20, 50, 40, True]], "n_gt": n_gt}


def fake_report(tmp_path):
    a = tiny_png(tmp_path / "a.png", (90, 90, 90))
    b = tiny_png(tmp_path / "b.png", (250, 250, 250))
    caps = [capture("naive", "C0_dark", "open", a, "REWORK", "REWORK"),
            capture("burst", "C0_dark", "open", a, "REWORK", "REWORK"),
            capture("naive", "C6_sun", "open", b, "PASS", "REWORK"),
            capture("burst", "C6_sun", "open", b, "RETAKE", "REWORK", "RETAKE", "Too bright: fit the hood."),
            capture("naive", "C6_sun", "hood", a, "REWORK", "REWORK"),
            capture("burst", "C6_sun", "hood", a, "REWORK", "REWORK")]
    return {"captures": caps, "conditions": ["C0_dark", "C6_sun"], "classes": ["scratch", "dent"],
            "results": [{"arm": "naive", "verdict": "FAIL"}, {"arm": "burst", "verdict": "PASS"}],
            "criterion": {"text": "C0-C5 open: F1 >= F1(C0) - 0.05."}, "verdict_text": "Burst passes."}


def test_demo_cards_from_report(tmp_path):
    ov = tmp_path / "overview"
    ov.mkdir()
    tiny_png(ov / "s900000_C6o_overview.png", (0, 120, 255))
    cards = V.cards_from_report(fake_report(tmp_path), overviews=ov)
    assert len(cards) == 1 + 3 + 1                                      # title, C0o, C6o, C6h, result
    c6 = cards[2]
    assert c6["title"].startswith("Direct sun") and "52,000 lux" in c6["title"]
    assert c6["caption"] == "old method fails / new method refuses safely (retake)"
    # a hood row captures BOTH arms through the hood: the old method must not look hood-less
    assert cards[3]["caption"] == "old method + hood passes / new method + hood passes"
    assert [p["label"] for p in cards[3]["panels"]] == ["Old method (single frame, auto-exposure + hood)",
                                                       "New method (burst + hood)"]
    assert [p["label"] for p in c6["panels"]] == ["Old method (single frame, auto-exposure)", "New method (burst)"]
    assert c6["overview"] == ov / "s900000_C6o_overview.png" and cards[1]["overview"] is None
    assert c6["gate"] == "Too bright: fit the hood."
    old, new = c6["panels"]
    assert old["ok"] is False and "WRONG" in old["status"] and new["ok"] is None
    assert [b[5] for b in cards[1]["panels"][0]["boxes"]] == ["gt", "det"] and cards[1]["panels"][0]["boxes"][1][4] == "dent"
    assert "Old method (one frame): FAIL" in cards[-1]["title"] and "Burst passes." in cards[-1]["text"]
    assert not any("table" in c for c in cards)                          # no rows in the report: no table card
    tiny_png(ov / "s900000_C6h.png", (0, 0, 255))                      # pipeline.run_plans --overview naming
    assert V.find_overview(ov, "s900000_C6h", "C6_sun") == ov / "s900000_C6h.png"
    img = V.render_card(c6)
    assert img.shape == (720, 1280, 3) and img.dtype == np.uint8
    assert V.ascii_text("“ok” ≥ 1") == '"ok" >= 1'


@pytest.mark.skipif(V.find_ffmpeg() is None, reason="ffmpeg not installed")
def test_demo_video_three_cards(tmp_path):
    cards = [{"title": f"Card {i}", "caption": "old method fails / new method + hood passes",
              "overview": str(tiny_png(tmp_path / f"o{i}.png", (40 * i, 80, 160))),
              "panels": [{"label": "Old", "image": str(tiny_png(tmp_path / f"n{i}.png", (200, 200, 200))),
                          "boxes": [[20, 20, 50, 40, "dent", "det"], [18, 18, 52, 42, "", "gt"]],
                          "status": "PASS (truth REWORK) - WRONG", "ok": False},
                         {"label": "New", "image": str(tiny_png(tmp_path / f"b{i}.png", (90, 90, 90))),
                          "boxes": [], "status": "RETAKE", "ok": None}],
              "gate": "Too bright: shade the part or fit the hood, then retake."} for i in range(3)]
    out = V.build_video(cards, tmp_path / "demo.mp4", seconds=0.4, fps=10)
    assert out.exists() and out.stat().st_size > 1000
    probe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    if not Path(probe).exists():
        return
    r = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,codec_name,nb_frames:format=duration", "-of", "json", str(out)],
                       capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    st = info["streams"][0]
    assert (st["width"], st["height"], st["codec_name"]) == (1280, 720, "h264")
    assert float(info["format"]["duration"]) == pytest.approx(1.2, abs=0.25)


def test_run_matrix_renders_the_written_plans(tmp_path, fake_sampler, monkeypatch):
    from trinetra.twin import pipeline

    MX.write_matrix_plans({}, 2, tmp_path, ("C0_dark", "C6_sun"), ("C6_sun",))
    seen = {}

    def run_plans(cfg, paths, out, **kw):
        seen.update(paths=[Path(p).name for p in paths], out=out, kw=kw)
        return {"done": len(paths)}

    monkeypatch.setattr(pipeline, "run_plans", run_plans)
    assert MX.run_matrix({}, tmp_path, overview=True, res_scale=0.25) == {"done": 6}
    assert seen["paths"][:3] == ["s900000_C0o.json", "s900000_C6o.json", "s900000_C6h.json"]
    assert seen["kw"] == {"overview": True, "res_scale": 0.25}
    MX.run_matrix({}, tmp_path, only=["s900001_C6h"])
    assert seen["paths"] == ["s900001_C6h.json"]


def test_demo_summary_table(tmp_path):
    """The results card sums every plate's rows per arm and device mode; the verdict card stays last and
    a long verdict_text is left out (the table carries the numbers)."""
    def row(cond, mode, silent, unsafe, retake, f1, lux):
        return {"condition": cond, "device_mode": mode, "lux_median": lux, "f1": {"value": f1},
                "silent": {"k": silent, "n": 30}, "silent_unsafe": {"k": unsafe, "n": 30}, "retake": {"k": retake, "n": 30}}
    rep = fake_report(tmp_path)
    rep["n_plates"] = 15
    rep["verdict_text"] = "x" * 400
    rep["results"] = [
        {"arm": "naive", "verdict": "FAIL", "rows": [row("C0_dark", "open", 2, 0, 0, 0.85, 0),
                                                     row("C6_sun", "open", 5, 0, 0, 0.91, 44872),
                                                     row("C6_sun", "hood", 2, 0, 0, 0.92, 44872)]},
        {"arm": "burst", "verdict": "FAIL", "rows": [row("C0_dark", "open", 4, 0, 6, 0.82, 0),
                                                     row("C6_sun", "open", 0, 0, 30, None, 44872),
                                                     row("C6_sun", "hood", 1, 0, 0, 0.83, 44872)]},
    ]
    cards = V.cards_from_report(rep, notes="Hood: lighting changed nothing.")
    table, last = cards[-2], cards[-1]
    assert table["title"] == "Results over all 15 test plates" and "44,872 lux" in table["caption"]
    assert table["text"] == "Hood: lighting changed nothing." and table["seconds"] == 12
    rows = table["table"]["rows"]
    assert [r[:2] for r in rows] == [["New method (burst)", "hood (product)"], ["New method (burst)", "open (no hood)"],
                                     ["Old method (one frame)", "hood (product)"], ["Old method (one frame)", "open (no hood)"]]
    assert rows[0][2] == ["1 / 30", "bad"] and rows[1][2] == ["4 / 60", "bad"] and rows[1][4] == ["36 / 60", "dim"]
    assert rows[1][5] == "0.82-0.82"                                     # a row without F1 (all refused) is skipped
    assert rows[3][4] == ["no gate", "dim"] and rows[2][3] == ["0", "good"]
    assert last["title"].startswith("Result: ") and "x" * 400 not in last["text"]
    img = V.render_card(table)
    assert img.shape == (720, 1280, 3)


def test_demo_cards_use_each_arms_own_model(tmp_path):
    """With eval_lighting --pairs all, both models run on both arms: the panels must show the
    naive model on the naive grids and the burst model on the burst grids."""
    rep = fake_report(tmp_path)
    other = tiny_png(tmp_path / "other.png", (0, 0, 255))
    for c in list(rep["captures"]):
        swapped = dict(c, model="burst" if c["arm"] == "naive" else "naive", image=str(other), pred_verdict="REJECT")
        rep["captures"].insert(0, swapped)                       # the other model's captures come first
    cards = V.cards_from_report(rep)
    for card in cards[1:-1]:
        old, new = card["panels"]
        assert old["image"] != str(other) and new["image"] != str(other)
    assert cards[2]["caption"] == "old method fails / new method refuses safely (retake)"


def test_default_hood_rows_follow_the_chosen_conditions(tmp_path, fake_sampler):
    """Device v2: the hood head is the product, so by default every chosen condition also gets a
    hood row (was C4 and C6 only; a subset without C4 once raised ValueError)."""
    man = MX.write_matrix_plans({}, 1, tmp_path, ("C0_dark", "C6_sun"))
    assert [r["tag"] for r in man["rows"]] == ["C0o", "C6o", "C0h", "C6h"]
    assert man["hood_conditions"] == ["C0_dark", "C6_sun"]
    assert MX.rows_for(("C0_dark", "C1_office"), ()) == [("C0_dark", "open"), ("C1_office", "open")]
    full = MX.rows_for(MX.ALL)
    assert [r for r in full if r[1] == "hood"] == [(c, "hood") for c in MX.ALL] and len(full) == 14
    assert MX.HOOD_CONDITIONS == MX.ALL
    with pytest.raises(ValueError):                                     # an explicit list is still checked
        MX.write_matrix_plans({}, 1, tmp_path, ("C0_dark", "C6_sun"), ("C4_side_lamp",))
    cli = ["--config", str(ROOT / "configs/twin.yaml"), "--n", "1", "--conditions", "C0_dark", "C6_sun"]
    man = MX.main(cli + ["--out", str(tmp_path / "cli")])
    assert [r["tag"] for r in man["rows"]] == ["C0o", "C6o", "C0h", "C6h"]
    man = MX.main(cli + ["--out", str(tmp_path / "cli2"), "--hood-conditions"])
    assert [r["tag"] for r in man["rows"]] == ["C0o", "C6o"]


def test_matrix_keeps_some_defect_free_plates(tmp_path, monkeypatch):
    """The research matrix has 2-3 defect-free plates in 8-10 (false rejects of clean parts). With
    consecutive seeds there may be none: later clean seeds then replace the last defective ones."""
    clean_seeds = {900011, 900013, 900020}

    def make_plan(cfg, seed, split, **kw):
        plan = fake_make_plan(cfg, seed, split, **kw)
        if seed in clean_seeds:
            plan["defects"], plan["empty"] = [], True
        return plan

    monkeypatch.setattr(MX.sampler, "make_plan", make_plan)
    man = MX.write_matrix_plans({}, 8, tmp_path, CONDS)                 # min_clean = 8 // 4 = 2
    seeds = [p["seed"] for p in man["plates"]]
    assert seeds == [900000, 900001, 900002, 900003, 900004, 900005, 900011, 900013]
    assert [p["clean"] for p in man["plates"]] == [False] * 6 + [True] * 2 and man["n_clean"] == 2
    assert MX.pick_seeds({}, 4, 900010, 1) == [900010, 900011, 900012, 900013]     # already has one: unchanged
    assert MX.pick_seeds({}, 3, 900000, 0) == [900000, 900001, 900002]
    # not enough clean seeds within max_scan: keep what was found, the plate count stays n
    assert MX.pick_seeds({}, 3, 900014, 3, max_scan=10) == [900014, 900015, 900020]
    man = MX.write_matrix_plans({}, 3, tmp_path / "m3", CONDS, min_clean=0)
    assert [p["seed"] for p in man["plates"]] == [900000, 900001, 900002]
