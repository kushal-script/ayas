import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from trinetra.tiles import CLASSES, Det

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import ablation  # noqa: E402
import build_datasets  # noqa: E402
import calib_set  # noqa: E402
import eval_lighting as EL  # noqa: E402
import eval_plates as E  # noqa: E402
import train  # noqa: E402
import train_arms  # noqa: E402

SCRIPTS = ("build_datasets", "train", "eval_plates", "ablation", "calib_set", "export_onnx", "train_arms",
           "eval_lighting")
# Classes common to the old aero list and the steel list, so these tests hold for either tiles.CLASSES.
DENT, SCRATCH, CRACK, CORROSION = (CLASSES.index(c) for c in ("dent", "scratch", "crack", "corrosion"))


def test_wilson_known_values():
    lo, hi = E.wilson(13, 15)
    assert lo == pytest.approx(0.621, abs=2e-3) and hi == pytest.approx(0.963, abs=2e-3)
    lo, hi = E.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = E.wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    assert E.wilson(0, 0) == (0.0, 1.0)


def sidecar():
    return {
        "schema": "trinetra.twin.sidecar/1", "scene_id": "x", "plate": "P5", "face": "A",
        "H_grid_plate": [[12.5, 0, 62.5], [0, 12.5, 62.5], [0, 0, 1]],
        "fasteners": [{"id": "H9", "cls": "open_hole", "centre_px": [1200, 1200], "d_px": 50}],   # aero class only
        "defects": [
            {"id": "D0", "cls": "dent", "centre_px": [300, 300], "d_px": 100},
            {"id": "D4", "cls": "corrosion", "centre_mm": [60, 20], "d_mm": 4.0},                 # via H_grid_plate
            {"id": "D1", "cls": "scratch", "polys_grid": [[[500, 600], [900, 600], [900, 603], [500, 603]]]},
            {"id": "D2", "cls": "dent", "bbox_grid": [1000, 1000, 1040, 1040]},
            {"id": "D3", "cls": "crack", "polys_grid": [[[200, 1100], [260, 1100], [260, 1102], [200, 1102]]],
             "visible": False},
            {"id": "X1", "cls": "fingerprint", "bbox_grid": [10, 10, 50, 50]},          # not a YOLO class
        ],
    }


def test_classes_follow_tiles():
    assert E.DEFECT_IDS == tuple(i for i, c in enumerate(CLASSES) if c not in ("rivet", "open_hole"))
    assert list(ablation.CLASS_COLS) == list(CLASSES)
    assert train.TRAIN_SETTINGS["hsv_h"] == 0.01 and train.TRAIN_SETTINGS["hsv_s"] == 0.4
    assert train.TRAIN_SETTINGS["hsv_v"] == 0.4 and train.TRAIN_SETTINGS["scale"] == 0.1
    assert train.TRAIN_SETTINGS["imgsz"] == 640


def test_gt_objects_from_sidecar():
    gts = {g.key: g for g in E.gt_objects(sidecar(), (1375, 1375))}
    want = {"P5A:D0", "P5A:D4", "P5A:D1", "P5A:D2", "P5A:D3"} | ({"P5A:H9"} if "open_hole" in CLASSES else set())
    assert set(gts) == want
    d0 = gts["P5A:D0"]
    assert d0.cls == DENT and d0.defect and d0.box == (250, 250, 351, 351)
    assert d0.mask.sum() == pytest.approx(np.pi * 50 ** 2, rel=0.02)
    d4 = gts["P5A:D4"]                                   # (60, 20) mm -> (812.5, 312.5) px, d = 50 px
    assert d4.cls == CORROSION and (d4.box[0] + d4.box[2]) / 2 == pytest.approx(813, abs=1)
    assert d4.box[2] - d4.box[0] == pytest.approx(50, abs=1)
    d1 = gts["P5A:D1"]
    assert d1.cls == SCRATCH and d1.box == (500, 600, 901, 604) and d1.mask.sum() == 401 * 4
    assert not gts["P5A:D3"].visible
    over = {g.key: g.visible for g in E.gt_objects(sidecar(), (1375, 1375), visible={"D3": True, "D1": False})}
    assert over["P5A:D3"] and not over["P5A:D1"] and over["P5A:D0"]


def test_plate_key_groups_captures_of_one_face():
    assert E.plate_key({"plate": "P5", "face": "A"}, "x") == "P5A"
    assert E.plate_key({"coupon": "P5A", "scene_id": "P5A_s03"}, "x") == "P5A"      # realdata.py sidecar
    assert E.plate_key({"coupon": None, "scene_id": "P5A_s07"}, "x") == "P5A"
    assert E.plate_key({"scene_id": "s000042"}, "s000042_v1") == "s000042"           # twin: one scene, one plate
    assert E.plate_key({"scene_id": "s900000_C6h"}, "s900000_C6h_v0") == "s900000"   # matrix rows share a plate


def test_match_rules_and_summary():
    gts = E.gt_objects(sidecar(), (1375, 1375))
    dets = [
        Det(DENT, 0.9, 255, 255, 345, 345),          # dent D0: IoU match
        Det(DENT, 0.6, 260, 260, 340, 340),          # duplicate dent box
        Det(SCRATCH, 0.8, 480, 560, 920, 650),       # scratch: IoU 0.04 but covers the whole mask
        Det(SCRATCH, 0.7, 1000, 1000, 1040, 1040),   # dent D2 called scratch
        Det(CRACK, 0.5, 195, 1095, 265, 1107),       # crack on an invisible crack: ignored
        Det(CORROSION, 0.4, 1200, 100, 1250, 150),   # corrosion on bare plate
    ]
    r = E.match_grid(dets, gts)
    status = [(d["cls"], d["status"], d["dup"], d["background"]) for d in r["dets"]]
    assert status == [(DENT, "tp", False, False), (DENT, "tp", True, False), (SCRATCH, "tp", False, False),
                      (SCRATCH, "fp", False, False), (CRACK, "ignored", False, False),
                      (CORROSION, "fp", False, True)]
    found = {o["key"]: (o["found"], o["pred"]) for o in r["objects"]}
    assert found["P5A:D0"] == (True, DENT) and found["P5A:D4"] == (False, None)
    assert found["P5A:D1"] == (True, SCRATCH) and found["P5A:D2"] == (False, SCRATCH)
    assert not r["clean"]

    clean = E.match_grid([Det(DENT, 0.5, 10, 10, 30, 30)],
                         E.gt_objects({"plate": "P2", "fasteners": sidecar()["fasteners"]}, (1375, 1375)))
    assert clean["clean"]
    s = E.summarize([r, clean])
    pc = s["per_class"]
    assert pc["dent"]["recall_physical"]["k"] == 1 and pc["dent"]["recall_physical"]["n"] == 2   # D0 yes, D2 no
    assert pc["dent"]["precision"]["k"] == 2 and pc["dent"]["precision"]["n"] == 3 and pc["dent"]["duplicates"] == 1
    assert pc["corrosion"]["recall_physical"]["value"] == 0.0 and pc["corrosion"]["precision"]["n"] == 1
    assert pc["scratch"]["precision"]["k"] == 1 and pc["scratch"]["precision"]["n"] == 2
    assert pc["crack"]["recall_physical"]["n"] == 0 and pc["crack"]["precision"]["n"] == 0
    assert pc["all_defects"]["recall_physical"]["k"] == 2 and pc["all_defects"]["recall_physical"]["n"] == 4
    assert pc["all_defects"]["precision"]["k"] == 3 and pc["all_defects"]["precision"]["n"] == 6
    assert ("all_fasteners" in pc) == ("rivet" in CLASSES)
    assert s["n_clean_grids"] == 1 and s["fp_per_clean_plate"] == 1.0
    assert s["false_reject_rate"]["k"] == 1
    m = np.array(s["confusion"]["matrix"])
    rows, cols = s["confusion"]["rows"], s["confusion"]["cols"]
    assert m[rows.index("dent"), cols.index("scratch")] == 1
    assert m[rows.index("corrosion"), cols.index("missed")] == 1
    assert m[rows.index("background"), cols.index("corrosion")] == 1
    assert m[rows.index("background"), cols.index("dent")] == 1      # the clean-plate false alarm


def test_physical_majority_across_captures():
    obj = {"key": "P5A:D1", "cls": SCRATCH, "defect": True, "visible": True, "pred": SCRATCH}
    grids = [{"objects": [dict(obj, found=f)], "dets": [], "clean": False} for f in (True, False, False)]
    s = E.summarize(grids)
    assert s["per_class"]["scratch"]["recall_physical"]["k"] == 0          # found in 1 of 3 captures
    assert s["per_class"]["scratch"]["recall_instances"]["k"] == 1
    s = E.summarize(grids[:2])
    assert s["per_class"]["scratch"]["recall_physical"]["k"] == 1          # 1 of 2: half counts


class BoxDetector:
    """Detects the red square drawn on the grid as a dent."""

    def detect(self, tiles):
        out = []
        for t in tiles:
            ys, xs = np.nonzero(np.all(t == (255, 0, 0), axis=2))
            out.append([Det(DENT, 0.9, float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))]
                       if len(xs) else [])
        return out


def test_evaluate_end_to_end(tmp_path):
    grids, sides = tmp_path / "grids", tmp_path / "sidecar"
    grids.mkdir()
    sides.mkdir()
    g = np.full((1375, 1375, 3), 128, np.uint8)
    g[400:440, 400:440] = (255, 0, 0)
    cv2.imwrite(str(grids / "P5A_s01.png"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
    sc = {"plate": "P5", "face": "A", "defects": [{"id": "D1", "cls": "dent", "bbox_grid": [400, 400, 440, 440]}]}
    (sides / "P5A_s01.json").write_text(json.dumps(sc))
    s = E.evaluate(BoxDetector(), grids, sides)
    assert s["per_class"]["dent"]["recall_physical"]["k"] == 1
    assert s["per_class"]["dent"]["precision"]["value"] == 1.0
    assert "| dent | 1/1" in E.to_markdown(s)


def test_evaluate_uses_per_variant_visibility(tmp_path):
    grids, sides = tmp_path / "grids", tmp_path / "sidecar"
    grids.mkdir()
    sides.mkdir()
    g = np.full((700, 700, 3), 128, np.uint8)
    for k in (0, 1):
        cv2.imwrite(str(grids / f"s000007_v{k}.png"), g)          # nothing detected in either variant
    sc = {"scene_id": "s000007", "defects": [{"id": "D1", "cls": "dent", "bbox_grid": [100, 100, 140, 140],
                                               "visible": True}],
          "variants": [{"k": 0, "visible": {"D1": True}}, {"k": 1, "visible": {"D1": False}}]}
    (sides / "s000007.json").write_text(json.dumps(sc))
    s = E.evaluate(BoxDetector(), grids, sides)
    assert s["per_class"]["dent"]["recall_instances"]["n"] == 1       # v1: invisible, not counted as a miss
    assert E.variant_index("s000007_v1") == 1 and E.variant_index("P5A_s01") is None


def touch(root: Path, split: str, names, label=True):
    for n in names:
        img = root / "images" / split / n
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"")
        if label:
            lab = root / "labels" / split / Path(n).with_suffix(".txt")
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("")


def real_tree(root: Path, with_val=True):
    touch(root, "train", ["P1A_c01_r0c0.jpg", "P1A_c01_r0c1.jpg", "P1A_c02_r0c0.jpg", "P1A_c03_r0c0.jpg",
                          "P1A_c04_r0c0.jpg", "P3A_c01_r0c0.jpg"])
    touch(root, "train", ["P3A_c02_r0c0.jpg"], label=False)
    if with_val:
        touch(root, "val", ["P1A_c05_r0c0.jpg"])
    touch(root, "test", ["P5A_c01_r0c0.jpg"])
    return root


def lines(p: Path) -> list[str]:
    return p.read_text().splitlines()


def test_build_datasets_rows(tmp_path):
    synth, real, out = tmp_path / "synth", real_tree(tmp_path / "real"), tmp_path / "ds"
    touch(synth, "train", ["s000001_v0_r0c0.jpg", "s000002_v0_r0c0.jpg"])
    touch(synth, "val", ["s000003_v0_r0c0.jpg"])
    written = build_datasets.build(synth, real, out, oversample=2, seed=3)
    assert set(written) == set(build_datasets.ROWS)
    ro = out / "real_only"
    assert len(lines(ro / "train.txt")) == 7 and all(Path(x).is_absolute() and "/images/train/" in x
                                                     for x in lines(ro / "train.txt"))
    assert [Path(x).name for x in lines(ro / "val.txt")] == ["P1A_c05_r0c0.jpg"]
    assert [Path(x).name for x in lines(ro / "test.txt")] == ["P5A_c01_r0c0.jpg"]
    data = yaml.safe_load((ro / "data.yaml").read_text())
    assert data["names"] == dict(enumerate(CLASSES)) and Path(data["train"]).is_absolute()
    man = json.loads((ro / "manifest.json").read_text())
    assert man["missing_labels"]["train"] == 1 and man["val_source"] == "real"

    r25 = [Path(x).stem for x in lines(out / "real_25" / "train.txt")]
    caps = {build_datasets.capture_key(Path(x)) for x in r25}
    assert {c[:2] for c in caps} == {"P1", "P3"} and len(caps) == 2           # 1 of 4 P1, 1 of 2 P3 captures
    if "P1A_c01" in caps:
        assert {"P1A_c01_r0c0", "P1A_c01_r0c1"} <= set(r25)                    # whole capture kept
    build_datasets.build(synth, real, tmp_path / "ds2", rows=["real_25"], seed=3)
    assert lines(out / "real_25" / "train.txt") == lines(tmp_path / "ds2" / "real_25" / "train.txt")

    s1, s2 = out / "synth_then_real" / "stage1", out / "synth_then_real" / "stage2"
    assert all("/synth/" in x for x in lines(s1 / "train.txt")) and len(lines(s2 / "train.txt")) == 7
    assert lines(s1 / "val.txt") == lines(ro / "val.txt") == lines(s2 / "val.txt")
    assert len(lines(out / "mixed" / "train.txt")) == 2 + 2 * 7
    assert len(lines(out / "synth_only" / "train.txt")) == 2


def test_build_datasets_val_fallback_and_leak(tmp_path):
    synth, real = tmp_path / "synth", real_tree(tmp_path / "real", with_val=False)
    touch(synth, "train", ["s000001_v0_r0c0.jpg"])
    touch(synth, "val", ["s000003_v0_r0c0.jpg"])
    build_datasets.build(synth, real, tmp_path / "ds", rows=["real_only"])
    assert [Path(x).name for x in lines(tmp_path / "ds" / "real_only" / "val.txt")] == ["s000003_v0_r0c0.jpg"]
    touch(real, "train", ["P5A_c01_r0c0.jpg"])
    with pytest.raises(SystemExit):
        build_datasets.build(synth, real, tmp_path / "ds3")


def test_calib_set_real_train_only(tmp_path):
    src, test = tmp_path / "train", tmp_path / "test"
    src.mkdir()
    test.mkdir()
    red = np.zeros((64, 64, 3), np.uint8)
    red[..., 2] = 255                                      # red in OpenCV BGR
    for i in range(3):
        cv2.imwrite(str(src / f"P1A_c0{i}_r0c0.png"), red)
    cv2.imwrite(str(src / "P3A_grid.png"), np.zeros((90, 120, 3), np.uint8))
    files = calib_set.source_images([src], test, {"P5"})
    info = calib_set.write(files, tmp_path / "out", n=10, size=64, seed=1)
    arr = np.load(tmp_path / "out" / "calib_64.npy")
    assert arr.shape == (10, 64, 64, 3) and arr.dtype == np.uint8 and info["n"] == 10
    assert len(list((tmp_path / "out" / "calib_64").glob("*.png"))) == 10
    reds = [a for a in arr if a.any()]
    assert reds and all((a[..., 0] == 255).all() and (a[..., 2] == 0).all() for a in reds)   # stored as RGB
    flips = [s["flip"] for s in info["samples"]]
    assert flips.count(None) == 4 and flips.count(1) == 4 and flips.count(0) == 2   # all plain, then h, then v
    cv2.imwrite(str(test / "P1A_c00_r0c0.png"), red)
    with pytest.raises(SystemExit):
        calib_set.source_images([src], test, {"P5"})
    cv2.imwrite(str(src / "P5A_c01_r0c0.png"), red)
    with pytest.raises(SystemExit):
        calib_set.source_images([src], None, {"P5"})


def test_ablation_dry_run_and_report(tmp_path, capsys):
    grids = tmp_path / "grids"
    grids.mkdir()
    args = ["--out", str(tmp_path / "runs"), "--models", str(tmp_path / "models"), "--reports",
            str(tmp_path / "reports"), "--test-grids", str(grids), "--epochs", "7", "--epochs-ft", "3"]
    ablation.main(args + ["--dry-run"])
    out = capsys.readouterr().out
    assert out.count("train.py") == 6 and out.count("export_onnx.py") == 5 and out.count("eval_plates.py") == 5
    assert "stage2/data.yaml" in out and "synth_then_real_s1/weights/best.pt" in out
    assert "--epochs 3" in out and "--test" in out
    assert not (tmp_path / "reports" / "ablation.md").exists()

    runs = tmp_path / "runs"
    (runs / "datasets" / "real_only").mkdir(parents=True)
    (runs / "datasets" / "real_only" / "manifest.json").write_text(json.dumps({"counts": {"train": 860}}))
    (runs / "train" / "real_only").mkdir(parents=True)
    (runs / "train" / "real_only" / "trinetra_train.json").write_text(
        json.dumps({"epochs": 7, "test": {"map50": 0.4321}}))
    ev = {"per_class": {"dent": {"recall_physical": {"k": 13, "n": 15, "value": 13 / 15, "ci95": [0.62, 0.96]}},
                        "all_defects": {"precision": {"k": 9, "n": 10, "value": 0.9, "ci95": [0.6, 0.98]}}},
          "fp_per_clean_plate": 0.5}
    (tmp_path / "reports" / "ablation" / "real_only").mkdir(parents=True)
    (tmp_path / "reports" / "ablation" / "real_only" / "eval.json").write_text(json.dumps(ev))
    ablation.main(args + ["--report-only", "--rows", "real_only", "synth_only"])
    md = (tmp_path / "reports" / "ablation.md").read_text()
    row = next(x for x in md.splitlines() if x.startswith("| real_only"))
    assert "real (860)" in row and "13/15 0.87 [0.62-0.96]" in row and "0.432" in row and "0.50" in row
    assert next(x for x in md.splitlines() if x.startswith("| synth_only")).count("n/a") >= 9


def test_compile_notebook_is_valid_nbformat4():
    nb = json.loads((ROOT / "hailo" / "compile_hef_colab.ipynb").read_text())
    assert nb["nbformat"] == 4 and isinstance(nb["nbformat_minor"], int) and isinstance(nb["metadata"], dict)
    kinds = {c["cell_type"] for c in nb["cells"]}
    assert kinds == {"markdown", "code"}
    code = []
    for c in nb["cells"]:
        assert isinstance(c["metadata"], dict) and isinstance(c["source"], (list, str))
        src = "".join(c["source"])
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            # IPython shell/magic lines are not Python; the rest must parse
            ast.parse("\n".join(ln if not ln.lstrip().startswith(("!", "%")) else
                                ln[: len(ln) - len(ln.lstrip())] + "pass" for ln in src.splitlines()))
            code.append(src)
    text = "\n".join(code)
    for needle in ("v5.1.0", "hailo10h", "--classes 6", "calibset_size=1024", "(3, 12)", "SNR"):
        assert needle in text, needle
    assert "--performance" not in text and "password" not in text.lower()


@pytest.mark.parametrize("name", SCRIPTS)
def test_training_scripts_have_help(name):
    r = subprocess.run([sys.executable, str(ROOT / "training" / f"{name}.py"), "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout


# --- The lighting experiment (eval_lighting) ----------------------------------------------

H_GRID = [[12.5, 0, 62.5], [0, 12.5, 62.5], [0, 0, 1]]          # 0.08 mm/px, 5 mm border
PLATE_MM = 40.0                                                   # grid (40 + 10) / 0.08 = 625 px: one tile
RED, GREEN = (255, 0, 0), (0, 255, 0)
DEFECTS = [{"id": "D1", "cls": "dent", "kind": "ball", "bbox_grid": [100, 100, 160, 160], "label": True,
            "measure": {"diameter_mm": 4.8}, "near": {"plate_edge_mm": 3.0}, "centre_mm": [5.0, 5.0]},
           {"id": "D2", "cls": "scratch", "kind": "scriber", "label": True,
            "polys_grid": [[[300, 400], [500, 400], [500, 404], [300, 404]]], "measure": {"length_mm": 16.0}}]
ROWS4 = (("C0_dark", "open"), ("C1_office", "open"), ("C6_sun", "open"), ("C6_sun", "hood"))
ROWS6 = (("C0_dark", "open"), ("C1_office", "open"), ("C6_sun", "open"),       # device v2: hood rows everywhere
         ("C0_dark", "hood"), ("C1_office", "hood"), ("C6_sun", "hood"))
TOO_BRIGHT = "Too bright: shade the part or fit the hood, then retake."


class PaintDetector:
    """Perfect on the painted grids: red block = dent, green block = scratch."""

    def detect(self, tiles):
        out = []
        for t in tiles:
            dets = []
            for cls, rgb in ((DENT, RED), (SCRATCH, GREEN)):
                n, _, stats, _ = cv2.connectedComponentsWithStats(np.all(t == rgb, axis=2).astype(np.uint8))
                dets += [Det(cls, 0.9, float(x), float(y), float(x + w), float(y + h)) for x, y, w, h, _ in stats[1:]]
            out.append(dets)
        return out


def judge_stub(log=None):
    """REJECT for a crack, REWORK for any other defect, PASS for none (stands in for rules.judge_steel)."""
    def judge(defects, spec, part):
        if log is not None:
            log.append((defects, part))
        types = {d["type"] for d in defects}
        return ("REJECT" if "crack" in types else "REWORK" if types else "PASS"), defects
    return judge


def paint(defective: bool, washed: bool) -> np.ndarray:
    g = np.full((625, 625, 3), 128, np.uint8)
    if defective:
        g[100:160, 100:160] = RED
        g[400:404, 300:500] = GREEN
    if washed:
        g[:] = 250                                   # glare: nothing left to see
    return g


def lighting_matrix(root: Path, rows=ROWS6, washed=(("C6_sun", "open"),), retake=(("C6_sun", "open"),),
                    layout="scene"):
    """A tiny matrix: plate s900000 (dent + scratch) and clean s900001, 1 variant, arms naive + burst.

    layout "scene": one sidecar per scene (variants[k].gate); "grid": post's per-grid sidecars next to
    each grid image (top-level arm, gate, gate_message, gated), no scene sidecar.
    """
    plates = []
    (root / "sidecar").mkdir(parents=True, exist_ok=True)
    for seed, defective in ((900000, True), (900001, False)):
        plate, entries = f"s{seed:06d}", []
        part = {"material": "ms_cold_rolled", "thickness_mm": 3.0, "size_mm": [PLATE_MM, PLATE_MM]}
        for cond, mode in rows:
            scene = f"{plate}_{EL.tag(cond, mode)}"
            entries.append({"scene_id": scene, "condition": cond, "device_mode": mode, "tag": EL.tag(cond, mode)})
            lux = 50000.0 if cond == "C6_sun" else 400.0
            base = {"scene_id": scene, "seed": seed, "split": "test", "plate_size_mm": [PLATE_MM, PLATE_MM],
                    "grid": {"mm_per_px": 0.08}, "H_grid_plate": H_GRID, "defects": DEFECTS if defective else []}
            for arm in ("naive", "burst"):
                d = root / "grids" / arm / "test"
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{scene}_v0.png"),
                            cv2.cvtColor(paint(defective, (cond, mode) in washed), cv2.COLOR_RGB2BGR))
                if layout == "grid":
                    refused = arm == "burst" and (cond, mode) in retake
                    (d / f"{scene}_v0.json").write_text(json.dumps(dict(
                        base, schema="trinetra.twin.grid/1", stem=f"{scene}_v0", arm=arm, k=0, plate=plate,
                        condition=cond, device_mode=mode, lux=lux, part=part, gated=arm == "burst",
                        matrix={"plate": plate, "condition": cond, "device_mode": mode},
                        gate="RETAKE" if refused else "PASS", gate_message=TOO_BRIGHT if refused else "",
                        gate_reasons=["too_bright"] if refused else [])))
            gate = {"verdict": "RETAKE", "message": TOO_BRIGHT, "reasons": ["too_bright"]} if (cond, mode) in retake else \
                {"verdict": "PASS", "message": "", "reasons": []}
            sc = dict(base, condition=cond, device_mode=mode,
                      variants=[{"k": 0, "lux": lux, "gate": gate,
                                 "visible": {"D1": True, "D2": True} if defective else {}}])
            if layout == "scene":
                (root / "sidecar" / f"{scene}.json").write_text(json.dumps(sc))
        plates.append({"plate": plate, "seed": seed, "rows": entries,
                       "part": {"material": "ms_cold_rolled", "thickness_mm": 3.0, "size_mm": [PLATE_MM, PLATE_MM]}})
    conds = list(dict.fromkeys(c for c, _ in rows))
    return {"conditions": conds, "hood_conditions": [c for c, m in rows if m == "hood"], "plates": plates}


def run_matrix_eval(root, manifest, judge=None, **kw):
    det = PaintDetector()
    return EL.evaluate_matrix({"naive": det, "burst": det}, root / "grids", root / "sidecar",
                              manifest=manifest, spec={}, judge=judge or judge_stub(), **kw)


def by_arm(rep):
    return {r["arm"]: (r, {m["tag"]: m for m in r["rows"]}) for r in rep["results"]}


def test_lighting_perfect_detector_passes_everywhere(tmp_path):
    log = []
    man = lighting_matrix(tmp_path, washed=(), retake=())
    rep = run_matrix_eval(tmp_path, man, judge_stub(log))
    assert {r["arm"]: r["model"] for r in rep["results"]} == {"naive": "naive", "burst": "burst"}   # own pairs
    for arm, (res, rows) in by_arm(rep).items():
        assert res["verdict"] == "PASS", res["checks"]
        assert set(rows) == {"C0o", "C1o", "C6o", "C0h", "C1h", "C6h"}
        for m in rows.values():
            assert m["f1"]["value"] == 1.0 and m["f1"]["fn"] == 0 and m["f1"]["fp"] == 0
            assert m["silent"]["k"] == 0 and m["retake"]["k"] == 0 and m["n_captures"] == 2
            assert m["verdict_agreement_c0"]["k"] == 2 and m["verdict_agreement_gt"]["value"] == 1.0
            assert m["fp_per_clean_plate"] == 0.0 and m["ssim_c0"]["mean"] == pytest.approx(1.0)
    defects, part = next((d, p) for d, p in log if d)
    assert part["material"] == "ms_cold_rolled" and part["thickness_mm"] == 3.0
    d1 = next(d for d in defects if d["type"] == "dent")
    assert set(d1) >= {"type", "confidence", "box_px", "location_mm", "size", "near_edge_mm"}
    assert d1["location_mm"] == {"x": pytest.approx(5.4), "y": pytest.approx(5.4)}      # box centre (130, 130) px
    assert d1["size"]["diameter_mm"] == pytest.approx(4.8) and d1["near_edge_mm"] == pytest.approx(3.0)
    sc = next(d for d in defects if d["type"] == "scratch")
    # the 201 x 5 px mask box is measured padded to label.min_box_px (8 px), as the training labels are
    assert sc["size"]["length_mm"] == pytest.approx(math.hypot(16.08, 0.64), abs=0.01)
    assert sc["size"]["width_mm"] == pytest.approx(0.64) and sc["box_px"] == [300.0, 400.0, 501.0, 405.0]


def test_lighting_detector_failing_only_in_sun(tmp_path):
    man = lighting_matrix(tmp_path)
    rep = run_matrix_eval(tmp_path, man)
    arms = by_arm(rep)
    burst, brow = arms["burst"]
    naive, nrow = arms["naive"]
    # new method: refuses direct sun without the hood (not a silent failure), passes with the hood
    assert brow["C6o"]["retake"]["k"] == 2 and brow["C6o"]["silent"]["k"] == 0 and brow["C6o"]["n_accepted"] == 0
    assert brow["C6o"]["gate_messages"] == {TOO_BRIGHT: 2}
    assert brow["C6h"]["f1"]["value"] == 1.0 and brow["C6h"]["silent"]["k"] == 0
    assert brow["C6o"]["retake_reasons"] == {"too_bright": 2} and brow["C6o"]["main_reason"] == "too_bright"
    assert burst["verdict"] == "PASS", burst["checks"]
    assert "in direct sun 2/2 (too_bright)" in burst["summary"] and "PASS" in burst["summary"]
    info = [c for c in burst["checks"] if c["status"] == "INFO" and c["row"] == "C6o"]
    assert info and "main reason: too_bright" in info[0]["detail"]               # open refusals: reported, allowed
    # old method: no gate, nothing detected in the glare -> the defective plate passes silently
    c6 = nrow["C6o"]
    assert c6["retake"]["k"] == 0 and c6["recall"]["k"] == 0 and c6["recall"]["n"] == 2
    assert c6["f1"]["value"] == 0.0 and c6["silent"]["k"] == 1 and c6["silent"]["n"] == 2
    assert c6["silent_unsafe"]["k"] == 1                       # PASS on a REWORK part
    assert c6["verdict_agreement_c0"]["k"] == 1 and c6["verdict_agreement_gt"]["k"] == 1
    assert c6["ssim_c0"]["mean"] < 0.9 and nrow["C1o"]["ssim_c0"]["mean"] == pytest.approx(1.0)
    assert naive["verdict"] == "FAIL"
    failed = [c for c in naive["checks"] if c["status"] == "FAIL"]
    assert [(c["check"], c["row"]) for c in failed] == [("no silent failure", "C6o")]
    assert "direct sun" in naive["summary"]

    paths = EL.write_report(rep, tmp_path / "reports", figures=2)
    md = paths["md"].read_text()
    assert "**PASS**" in md and "**FAIL**" in md and "C6o direct sun" in md and TOO_BRIGHT in md
    assert "Main retake reason" in md and "| too_bright |" in md and "C6h direct sun + hood" in md
    js = json.loads(paths["json"].read_text())
    assert js["schema"] == EL.SCHEMA and len(js["captures"]) == 2 * 2 * 6 and "match" not in js["captures"][0]
    figs = sorted((tmp_path / "reports" / "figures").glob("*.png"))
    assert [f.name for f in figs] == ["lighting_s900000.png", "lighting_s900001.png"]
    img = cv2.imread(str(figs[0]))
    assert img.shape == (2 * (280 + 34), 90 + 6 * 280, 3)


def test_lighting_missing_rows_and_refusing_everything(tmp_path):
    rows = (("C0_dark", "open"), ("C1_office", "open"), ("C0_dark", "hood"), ("C1_office", "hood"))
    man = lighting_matrix(tmp_path, rows=rows, retake=(("C1_office", "open"), ("C1_office", "hood")))
    man["conditions"] = ["C0_dark", "C1_office", "C3_window"]          # C3 was planned but not rendered
    rep = run_matrix_eval(tmp_path, man)
    res, rows = by_arm(rep)["burst"]
    checks = {(c["check"], c["row"]): c["status"] for c in res["checks"]}
    assert checks[("row present", "C3h")] == "MISSING" and ("row present", "C3o") not in checks
    assert checks[("retake rate <= 5 %", "C1h")] == "FAIL"             # the product may not refuse a condition
    assert checks[("F1 within 0.05 of C0", "C1h")] == "FAIL"           # a gate that refuses everything fails
    assert checks[("retakes reported", "C1o")] == "INFO"               # open: refusing is allowed
    assert ("F1 within 0.05 of C0", "C1o") not in checks
    assert res["verdict"] == "FAIL" and "every capture was refused (mostly too_bright)" in res["summary"]
    no_hood = lighting_matrix(tmp_path / "old", rows=ROWS4)            # an old matrix (hood only in C6)
    res = by_arm(run_matrix_eval(tmp_path / "old", no_hood))["burst"][0]
    missing = {(c["check"], c["row"]) for c in res["checks"] if c["status"] == "MISSING"}
    assert res["verdict"] == "INCOMPLETE" and missing == {("row present", "C0h"), ("row present", "C1h"),
                                                          ("F1 within 0.05 of C0", "C6h"),   # no C0 hood reference
                                                          ("verdict = C0 >= 95 %", "C6h")}


def test_lighting_reads_post_per_grid_sidecars(tmp_path):
    man = lighting_matrix(tmp_path, layout="grid")
    rep = run_matrix_eval(tmp_path, {"conditions": man["conditions"]})        # everything from the sidecars
    arms = by_arm(rep)
    assert arms["burst"][0]["verdict"] == "PASS" and arms["naive"][0]["verdict"] == "FAIL"
    assert arms["burst"][1]["C6o"]["retake"]["k"] == 2 and arms["naive"][1]["C6o"]["retake"]["k"] == 0
    assert arms["burst"][1]["C6o"]["gate_messages"] == {TOO_BRIGHT: 2}
    assert {c["part"]["material"] for c in rep["captures"]} == {"ms_cold_rolled"}
    assert {c["plate"] for c in rep["captures"]} == {"s900000", "s900001"} and rep["n_plates"] == 2


def test_oracle_detects_what_each_grid_shows_and_mad_is_zero_on_identical_grids(tmp_path):
    man = lighting_matrix(tmp_path, washed=(), retake=(), layout="grid")
    for js in (tmp_path / "grids").rglob("*.json"):              # post's per-grid visible_arm and bbox_grid
        sc = json.loads(js.read_text())
        for d in sc["defects"]:
            d["bbox_grid"] = d.get("bbox_grid") or [300, 400, 501, 405]
            d["visible_arm"] = not (d["id"] == "D2" and (sc["condition"], sc["device_mode"]) == ("C6_sun", "open"))
        js.write_text(json.dumps(sc))
    models = EL.parse_models(["oracle"], 0.25)
    assert isinstance(models["oracle"], EL.OracleDetector)
    rep = EL.evaluate_matrix(models, tmp_path / "grids", tmp_path / "sidecar", manifest=man, spec={},
                             judge=judge_stub())
    assert {(r["model"], r["arm"]) for r in rep["results"]} == {("oracle", "naive"), ("oracle", "burst")}
    for res in rep["results"]:
        rows = {m["tag"]: m for m in res["rows"]}
        assert rows["C0h"]["f1"]["value"] == 1.0 and rows["C1o"]["f1"]["value"] == 1.0
        assert rows["C6o"]["recall"]["k"] == 1 and rows["C6o"]["recall"]["n"] == 2      # D2 washed out in this grid
        assert rows["C6o"]["f1"]["fp"] == 0 and rows["C6o"]["fp_per_clean_plate"] == 0.0
        assert rows["C1h"]["mad_c0"] == {"mean": 0.0, "max": 0.0, "n": 2}               # the same painted grid
    with pytest.raises(RuntimeError):                                                   # never used unprimed
        EL.OracleDetector().detect([np.zeros((640, 640, 3), np.uint8)])
    a, b = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 10, np.uint8)
    b[0, 0] = [40, 40, 40]
    m = np.ones((4, 4), bool)
    m[0, 0] = False
    assert EL.mean_abs_diff(a, b) == pytest.approx((15 * 10 + 40) / 16) and EL.mean_abs_diff(a, b, m) == 10.0


def test_lighting_with_the_real_steel_rules(tmp_path):
    from trinetra import rules

    spec_path = ROOT / "configs" / "spec_steel.yaml"
    if not hasattr(rules, "judge_steel") or not spec_path.exists():
        pytest.skip("rules.judge_steel / spec_steel.yaml not there yet (spec-advisor workstream)")
    man = lighting_matrix(tmp_path)
    det = PaintDetector()
    rep = EL.evaluate_matrix({"naive": det, "burst": det}, tmp_path / "grids", tmp_path / "sidecar", manifest=man,
                             spec=rules.load_spec(spec_path), judge=EL.default_judge())
    caps = {(c["arm"], c["tag"], c["plate"]): (c["pred_verdict"], c["gt_verdict"]) for c in rep["captures"]}
    assert caps[("burst", "C0o", "s900001")] == ("PASS", "PASS")
    assert caps[("burst", "C0o", "s900000")][0] == caps[("burst", "C0o", "s900000")][1] != "PASS"
    assert caps[("naive", "C6o", "s900000")][0] == "PASS"                     # glare: the old method passes it
    assert by_arm(rep)["burst"][0]["verdict"] == "PASS" and by_arm(rep)["naive"][0]["verdict"] == "FAIL"


def test_silent_failure_logic():
    def r(gate, pred, gt, c0=None):
        return {"gate": gate, "pred_verdict": pred, "gt_verdict": gt, "c0_verdict": c0, "plate": "p", "lux": None,
                "gt": [], "gate_message": "", "match": {"objects": [], "dets": [], "clean": True}}
    rs = [r("PASS", "PASS", "REWORK"),               # silent, unsafe
          r("PASS", "REJECT", "REWORK"),             # silent, conservative
          r("PASS_FLAGGED", "PASS", "REJECT"),       # flagged but let through: still silent
          r("RETAKE", "RETAKE", "REJECT"),           # refused: never silent
          r("PASS", "REWORK", "REWORK", "REWORK")]   # right
    m = EL.row_metrics(rs)
    assert m["silent"] == EP_rate(3, 5) and m["silent_unsafe"]["k"] == 2 and m["silent_flagged"] == 1
    assert m["retake"]["k"] == 1 and m["n_accepted"] == 4 and m["verdict_agreement_gt"]["k"] == 1
    assert m["verdict_agreement_c0"] == EP_rate(1, 1)
    assert EL.less_severe("PASS", "REWORK") and not EL.less_severe("REJECT", "HOLD_FOR_QA")
    assert EL.unsafe("PASS", "REWORK") and EL.unsafe("WEIRD", "PASS")       # unknown verdicts count as unsafe
    assert not EL.unsafe("HOLD_FOR_QA", "REJECT") and not EL.unsafe("REJECT", "REWORK")   # a hold goes to a person
    assert EL.row_metrics([r("PASS", "HOLD_FOR_QA", "REJECT")])["silent_unsafe"]["k"] == 0


def EP_rate(k, n):
    return E.rate(k, n)


def test_gate_and_row_fields_from_sidecar():
    sc = {"condition": "C6_sun", "device_mode": "open",
          "variants": [{"k": 0, "gate": {"verdict": "RETAKE", "message": "Too bright"}},
                       {"k": 1, "gate": "PASS_FLAGGED", "gate_message": "flagged"},
                       {"k": 2, "arms": {"naive": {"gate": {"verdict": "PASS"}}, "burst": {"gate": "RETAKE"}}}]}
    assert EL.gate_of(sc, 0, "burst") == ("RETAKE", "Too bright")
    assert EL.gate_of(sc, 0, "naive") == ("PASS", "")                  # the old method has no gate
    assert EL.gate_of(sc, 1, "burst") == ("PASS_FLAGGED", "flagged")
    assert EL.gate_of(sc, 2, "burst") == ("RETAKE", "") and EL.gate_of(sc, 2, "naive") == ("PASS", "")
    assert EL.gate_of(sc, 9, "burst") == ("PASS", "")                  # no info: counted as let through
    assert EL.row_of(sc, 0, "burst", {}) == ("C6_sun", "open")
    assert EL.row_of({}, 0, "burst", {"condition": "C1_office", "device_mode": "hood"}) == ("C1_office", "hood")
    assert EL.row_of({"matrix": {"condition": "C2_warm_bulb", "device_mode": "open"}}, 0, "x", {}) == \
        ("C2_warm_bulb", "open")
    per_arm = {"variants": [{"k": 0, "arm": "naive", "gate": "PASS"}, {"k": 0, "arm": "burst", "gate": "RETAKE"}]}
    assert EL.gate_of(per_arm, 0, "burst")[0] == "RETAKE"
    arms = {"variants": [{"k": 0, "gate": "RETAKE", "gate_message": "train arm",
                          "arms": {"burst": {"gate": "PASS_FLAGGED", "message": "moved", "gated": True},
                                   "naive": {"gate": "PASS", "message": "", "gated": False}}}]}
    assert EL.gate_of(arms, 0, "burst") == ("PASS_FLAGGED", "moved") and EL.gate_of(arms, 0, "naive") == ("PASS", "")
    grid = {"arm": "burst", "gate": "RETAKE", "gate_message": "Hold still and retake.", "gated": True}
    assert EL.gate_of(grid, 0, "burst") == ("RETAKE", "Hold still and retake.")
    assert EL.gate_of(dict(grid, arm="naive", gated=False), 0, "naive") == ("PASS", "")
    assert EL.plate_of({"scene_id": "s900003_C4h"}, {}) == "s900003"


def test_box_defect_measures_and_hole_distance():
    sc = {"H_grid_plate": H_GRID, "plate_size_mm": [100, 100], "grid": {"mm_per_px": 0.08},
          "fasteners": [{"u_mm": 30.0, "v_mm": 10.0, "d_mm": 4.0}]}
    d = EL.box_defect("crack", 0.7, (62.5 + 12.5 * 20, 62.5 + 12.5 * 8, 62.5 + 12.5 * 26, 62.5 + 12.5 * 9), sc, "P1")
    assert d["location_mm"] == {"x": 23.0, "y": 8.5}
    assert d["size"]["length_mm"] == pytest.approx(math.hypot(6, 1), abs=1e-3) and d["size"]["width_mm"] == 1.0
    assert d["near_edge_mm"] == pytest.approx(math.hypot(4, 1) - 2, abs=0.01)   # hole edge (plate edge: 8 mm)
    p = EL.box_size("pit", 1.0, 0.5)
    assert p["count"] == 1 and p["diameter_mm"] == 1.0
    assert EL.box_size("stain", 4.0, 1.0)["diameter_mm"] == pytest.approx(2.0)


def test_ssim_and_alignment():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    assert EL.ssim(a, a) == pytest.approx(1.0)
    noisy = np.clip(a.astype(int) + rng.normal(0, 40, a.shape), 0, 255).astype(np.uint8)
    assert EL.ssim(noisy, a) < 0.9
    ref = {"H_grid_plate": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "plate_size_mm": [160, 120]}
    rot = {"H_grid_plate": [[0, 1, 0], [-1, 0, 159], [0, 0, 1]], "plate_size_mm": [160, 120]}   # grid rotated 90 deg
    turned = np.ascontiguousarray(np.rot90(a))                       # how that grid sees the same plate
    assert turned[0, 0] == a[0, 159] and turned.shape == (160, 120)
    back = EL.align_to(turned, rot, ref, a.shape)
    assert np.array_equal(back[2:-2, 2:-2], a[2:-2, 2:-2])
    assert EL.plate_mask(ref, a.shape).sum() > 0.95 * a.size * (159 / 160) * (119 / 120) * 0.9


def test_train_arms_lists_same_scenes_and_refuses_matrix_plates(tmp_path):
    burst, naive = tmp_path / "arm_burst" / "yolo", tmp_path / "arm_naive" / "yolo"
    touch(burst, "train", ["s000001_v0_r0c0.jpg", "s000001_v1_r0c0.jpg", "s000002_v0_r0c1.jpg"])
    touch(naive, "train", ["s000001_v0_r0c0.jpg", "s000002_v0_r0c1.jpg", "s000003_v0_r0c0.jpg"])
    touch(burst, "val", ["s000010_v0_r0c0.jpg"])
    touch(naive, "val", ["s000010_v0_r0c0.jpg"])
    for tree, scenes in ((burst, ("s000001", "s000002", "s000003", "s000010")), (naive, ("s000001", "s000002"))):
        (tree.parent / "sidecar").mkdir(exist_ok=True)
        for sc in scenes:
            (tree.parent / "sidecar" / f"{sc}.json").write_text("{}")
    written = train_arms.build_arm_datasets({"naive": naive, "burst": burst}, tmp_path / "ds")
    nb = [Path(x).name for x in lines(tmp_path / "ds" / "arm_naive" / "train.txt")]
    assert "s000003_v0_r0c0.jpg" in nb and len(nb) == 3          # default: each arm keeps its own tiles
    train_arms.build_arm_datasets({"naive": naive, "burst": burst}, tmp_path / "dsm", match="scenes")
    nb = [Path(x).name for x in lines(tmp_path / "dsm" / "arm_naive" / "train.txt")]
    bb = [Path(x).name for x in lines(tmp_path / "dsm" / "arm_burst" / "train.txt")]
    assert "s000003_v0_r0c0.jpg" not in nb and len(nb) == 2 and len(bb) == 3      # s000003 only in naive
    man = json.loads((tmp_path / "dsm" / "arm_burst" / "manifest.json").read_text())
    assert man["scenes"] == {"train": 2, "val": 1} and man["settings"]["hsv_v"] == 0.4
    assert man["plans"] == {"same_scenes": False, "n_scenes": {"naive": 2, "burst": 4}}
    assert yaml.safe_load(written["burst"].read_text())["names"] == dict(enumerate(CLASSES))
    assert train_arms.scene_key(Path("s900000_C6h_v1_r2c0.jpg")) == "s900000_C6h"
    touch(burst, "train", ["s900000_C0o_v0_r0c0.jpg"])
    with pytest.raises(SystemExit):
        train_arms.build_arm_datasets({"naive": naive, "burst": burst}, tmp_path / "ds2")


def test_train_arms_dry_run_and_naive_config(tmp_path, capsys):
    args = ["--burst", str(tmp_path / "b"), "--naive", str(tmp_path / "n"), "--matrix", str(tmp_path / "m"),
            "--runs", str(tmp_path / "runs"), "--models", str(tmp_path / "models"), "--epochs", "5", "--dry-run"]
    train_arms.main(args)
    out = capsys.readouterr().out
    assert out.count("train.py") == 2 and out.count("export_onnx.py") == 2 and out.count("eval_lighting.py") == 1
    assert "arm_naive/data.yaml" in out and "v8n_burst" in out and "--epochs 5" in out
    assert "naive=" in out and "burst=" in out
    assert not (tmp_path / "runs").exists()
    cfg = train_arms.write_naive_config(tmp_path / "twin_naive.yaml")
    assert yaml.safe_load(cfg.read_text())["post"]["train_arm"] == "naive"


def test_calib_set_twin_tiles_need_flag_and_never_matrix(tmp_path):
    src = tmp_path / "train"
    src.mkdir()
    cv2.imwrite(str(src / "s000004_v0_r0c0.png"), np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(SystemExit):
        calib_set.source_images([src], None, {"P5"})
    assert len(calib_set.source_images([src], None, {"P5"}, allow_twin=True)) == 1
    cv2.imwrite(str(src / "s900002_C6h_v0_r0c0.png"), np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(SystemExit):
        calib_set.source_images([src], None, {"P5"}, allow_twin=True)


# --- Review fixes (14 Nov): measurement, C0 reference, SSIM, warnings ----------------------

THIN = {"H_grid_plate": H_GRID, "plate_size_mm": [PLATE_MM, PLATE_MM], "grid": {"mm_per_px": 0.08},
        "plate": "s900005", "defects": [
            {"id": "D1", "cls": "scratch", "label": True,                 # 16 mm x 0.16 mm, horizontal
             "polys_grid": [[[200, 300], [400, 300], [400, 301], [200, 301]]]},
            {"id": "D2", "cls": "pit", "label": True, "bbox_grid": [300, 200, 304, 204]}]}   # 0.32 mm pit


def test_thin_defects_measured_like_the_training_labels():
    """A perfect detector outputs the label boxes (padded to label.min_box_px), so measuring the
    ground-truth mask box unpadded gave it a different verdict: a silent failure with no error."""
    from trinetra import rules

    spec_path = ROOT / "configs" / "spec_steel.yaml"
    if not hasattr(rules, "judge_steel") or not spec_path.exists():
        pytest.skip("rules.judge_steel / spec_steel.yaml not there yet")
    spec, part = rules.load_spec(spec_path), {"material": "ms_cold_rolled", "thickness_mm": 3.0, "size_mm": [40, 40]}
    gts = E.gt_objects(THIN, (625, 625), "s900005_C0o_v0")
    labels = [Det(g.cls, 0.9, *EL.pad_box(g.box, 8)) for g in gts]       # what post writes as YOLO boxes
    pred = rules.judge_steel(EL.pred_defects(labels, THIN), spec, part)[0]
    truth = rules.judge_steel(EL.gt_defects(gts, THIN), spec, part)[0]
    assert pred == truth
    raw = rules.judge_steel(EL.gt_defects(gts, THIN, min_box_px=0), spec, part)[0]
    assert raw != pred                                                    # the old, unpadded measure disagreed
    d = {x["type"]: x for x in EL.gt_defects(gts, THIN)}
    assert d["scratch"]["size"]["width_mm"] == pytest.approx(0.64) and d["pit"]["size"]["diameter_mm"] == pytest.approx(0.64)
    assert d["pit"]["box_px"] == [300.0, 200.0, 304.0, 204.0]             # the record keeps the real box
    assert EL.pad_box((10, 10, 30, 12), 8) == (10, 7, 30, 15) and EL.label_min_box() == 8.0


def test_c0_reference_verdict_prefers_an_accepted_capture():
    def r(plate, cond, k, pred):
        return {"plate": plate, "condition": cond, "device_mode": "open", "k": k, "pred_verdict": pred}

    out = [r("a", "C0_dark", 0, "REWORK"), r("a", "C0_dark", 1, "RETAKE"), r("a", "C1_office", 1, "REWORK"),
           r("b", "C0_dark", 0, "RETAKE"), r("b", "C1_office", 0, "PASS"), r("c", "C1_office", 0, "PASS")]
    EL.link_c0(out)
    c0 = {(x["plate"], x["condition"], x["k"]): x["c0_verdict"] for x in out}
    assert c0[("a", "C1_office", 1)] == "REWORK"       # v1 refused in the dark room: compare with the accepted v0
    assert c0[("b", "C1_office", 0)] == "RETAKE"       # every dark-room capture refused: not comparable
    assert c0[("c", "C1_office", 0)] is None           # no dark-room row at all


def test_ssim_is_over_accepted_captures_and_prefers_an_accepted_reference(tmp_path):
    rng = np.random.default_rng(1)
    plate = cv2.GaussianBlur(rng.integers(0, 255, (625, 625, 3), dtype=np.uint8), (0, 0), 2)
    paths = {}
    for name, img in (("good", plate), ("ghost", rng.integers(0, 255, (625, 625, 3), dtype=np.uint8))):
        paths[name] = tmp_path / f"{name}.png"
        cv2.imwrite(str(paths[name]), img)
    sc = {"H_grid_plate": H_GRID, "plate_size_mm": [PLATE_MM, PLATE_MM], "grid": {"mm_per_px": 0.08}}

    def cap(cond, k, gate, image):
        return EL.Capture("burst", f"s900000_{EL.tag(cond, 'open')}", k, "s900000", cond, "open", paths[image], sc,
                          gate, "", None, {})

    caps = [cap("C0_dark", 0, "PASS", "good"), cap("C0_dark", 1, "RETAKE", "ghost"), cap("C1_office", 1, "PASS", "good")]
    ss = EL.ssim_vs_reference(caps)
    assert ss[("burst", "s900000_C1o", 1)] == pytest.approx(1.0)        # against the accepted C0 v0, not the ghost

    def row(gate, s):
        return {"gate": gate, "pred_verdict": "RETAKE" if gate == "RETAKE" else "PASS", "gt_verdict": "PASS",
                "c0_verdict": "PASS", "plate": "p", "lux": None, "gt": [], "gate_message": "", "ssim_c0": s,
                "match": {"objects": [], "dets": [], "clean": True}}
    m = EL.row_metrics([row("PASS", 0.95), row("RETAKE", 0.01)])
    assert m["ssim_c0"] == {"mean": 0.95, "min": 0.95, "n": 1}
    assert m["ssim_c0_all"] == {"mean": pytest.approx(0.48), "n": 2}


def test_dark_room_retakes_fail_the_hood_and_are_reported_open(tmp_path):
    man = lighting_matrix(tmp_path, retake=(("C6_sun", "open"), ("C0_dark", "open"), ("C0_dark", "hood")))
    for arm in ("naive", "burst"):                                      # a second C0 capture per plate, accepted
        for p in ("s900000", "s900001"):
            for t in ("C0o", "C0h"):
                src = tmp_path / "grids" / arm / "test" / f"{p}_{t}_v0.png"
                (tmp_path / "grids" / arm / "test" / f"{p}_{t}_v1.png").write_bytes(src.read_bytes())
    for p in ("s900000", "s900001"):
        for t in ("C0o", "C0h"):
            f = tmp_path / "sidecar" / f"{p}_{t}.json"
            sc = json.loads(f.read_text())
            sc["variants"].append(dict(sc["variants"][0], k=1, gate={"verdict": "PASS", "message": ""}))
            f.write_text(json.dumps(sc))
    rep = run_matrix_eval(tmp_path, man)
    res, by_tag = by_arm(rep)["burst"]
    assert by_tag["C0o"]["retake"]["k"] == 2 and by_tag["C0o"]["n_accepted"] == 2
    info = [c for c in res["checks"] if c["status"] == "INFO" and c["row"] == "C0o"]
    assert info and "2 of 4" in info[0]["detail"] and "too_bright" in info[0]["detail"]   # open: reported only
    fail = [c for c in res["checks"] if c["status"] == "FAIL"]
    assert [(c["check"], c["row"]) for c in fail] == [("retake rate <= 5 %", "C0h")]     # the product: 2 of 4
    assert res["verdict"] == "FAIL"
    # C1 has v0 only, and every C0 v0 was refused: it is compared with the plate's accepted C0 v1 (same device)
    assert by_tag["C1o"]["verdict_agreement_c0"] == E.rate(2, 2)
    assert by_tag["C1h"]["verdict_agreement_c0"] == E.rate(2, 2)


def test_evaluate_matrix_loads_the_steel_spec_by_default(tmp_path):
    from trinetra import rules

    if not (ROOT / "configs" / "spec_steel.yaml").exists():
        pytest.skip("spec_steel.yaml not there yet")
    man = lighting_matrix(tmp_path, washed=(), retake=())
    seen = []

    def judge(defects, spec, part):
        seen.append(spec.get("domain"))
        return "PASS", defects
    EL.evaluate_matrix({"naive": PaintDetector()}, tmp_path / "grids", tmp_path / "sidecar", arms=("naive",),
                       manifest=man, judge=judge)
    assert seen and set(seen) == {getattr(rules, "STEEL_DOMAIN", "steel_surface")}


# --- Review 2 (14 Nov, evening): device-faithful measures ---------------------------------

class FixedBoxDetector:
    """Returns the same tile-local boxes for every tile (a 625 px grid is one tile)."""

    def __init__(self, dets):
        self.dets = dets

    def detect(self, tiles):
        return [list(self.dets) for _ in tiles]


def test_measure_app_uses_the_on_device_measures(tmp_path):
    """--measure box gives a diagonal 0.16 mm scratch the width of its box (24 mm); the app measures
    the defect's own pixels. measure=app must use the app's numbers, and the truth then defaults to
    the analytic (sidecar) measures."""
    pytest.importorskip("trinetra.app.pipeline")
    g = np.full((625, 625, 3), 128, np.uint8)
    cv2.line(g, (150, 150), (450, 450), (30, 30, 30), 2)
    d = tmp_path / "grids" / "burst" / "test"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "s900000_C0o_v0.png"), g)
    truth = {"length_mm": 33.94, "width_mm": 0.16}
    sc = {"scene_id": "s900000_C0o", "plate": "s900000", "condition": "C0_dark", "device_mode": "open",
          "plate_size_mm": [PLATE_MM, PLATE_MM], "H_grid_plate": H_GRID,
          "grid": {"mm_per_px": 0.08, "border_mm": 5.0, "plate_mm": [PLATE_MM, PLATE_MM]},
          "part": {"material": "ms_cold_rolled", "thickness_mm": 3.0, "size_mm": [PLATE_MM, PLATE_MM]},
          "defects": [{"id": "D1", "cls": "scratch", "label": True, "visible": True, "size": truth,
                       "near_edge_mm": 7.0, "polys_grid": [[[149, 151], [151, 149], [451, 449], [449, 451]]]}],
          "variants": [{"k": 0, "gate": "PASS"}]}
    (tmp_path / "sidecar").mkdir()
    (tmp_path / "sidecar" / "s900000_C0o.json").write_text(json.dumps(sc))
    det = FixedBoxDetector([Det(SCRATCH, 0.9, 147.0, 147.0, 454.0, 454.0)])
    man = {"conditions": ["C0_dark"]}
    widths = {}
    for measure in ("box", "app"):
        log = []
        rep = EL.evaluate_matrix({"burst": det}, tmp_path / "grids", tmp_path / "sidecar", arms=("burst",),
                                 manifest=man, spec={}, judge=judge_stub(log), measure=measure)
        pred = next(x for ds, _ in log for x in ds if x["id"] == "P1")
        gt = next(x for ds, _ in log for x in ds if x["id"] == "D1")
        widths[measure] = pred["size"]["width_mm"]
        assert rep["rule"]["measure"] == measure
        if measure == "app":
            assert rep["rule"]["gt_measure"] == "true" and gt["size"] == truth and gt["near_edge_mm"] == 7.0
            assert pred["size"]["length_mm"] == pytest.approx(33.9, abs=1.5)
            assert pred["near_edge_mm"] == pytest.approx(6.7, abs=0.5)       # plate edge, measured extent
        else:
            assert rep["rule"]["gt_measure"] == "box" and gt["size"]["width_mm"] == pytest.approx(24.2, abs=0.2)
    assert widths["box"] > 20 and widths["app"] < 0.5
    with pytest.raises(ValueError):
        EL.evaluate_matrix({"burst": det}, tmp_path / "grids", tmp_path / "sidecar", arms=("burst",),
                           manifest=man, spec={}, judge=judge_stub(), measure="ruler")


def test_true_ground_truth_is_the_canonical_gt_verdict():
    """--gt-measure true converts the sidecar labels exactly as twin.records.gt_verdict does,
    plan params included (pitted rust is critical; the sidecar alone does not say so)."""
    from trinetra import rules

    spec_path = ROOT / "configs" / "spec_steel.yaml"
    records = pytest.importorskip("trinetra.twin.records")
    if not hasattr(records, "gt_verdict") or not spec_path.exists():
        pytest.skip("twin.records.gt_verdict / spec_steel.yaml not there yet")
    spec = rules.load_spec(spec_path)
    part = {"material": "ms_cold_rolled", "thickness_mm": 3.0, "size_mm": [PLATE_MM, PLATE_MM]}
    sc = {"H_grid_plate": H_GRID, "plate_size_mm": [PLATE_MM, PLATE_MM], "grid": {"mm_per_px": 0.08}, "plate": "s900000",
          "defects": [
              {"id": "D1", "cls": "corrosion", "kind": "rust", "label": True, "bbox_grid": [200, 200, 260, 250],
               "size": {"area_mm2": 15.0, "diameter_mm": 4.37}, "location_mm": {"x": 13.4, "y": 13.0},
               "near_edge_mm": 11.0},
              {"id": "D2", "cls": "scratch", "kind": "scriber", "label": True,
               "polys_grid": [[[300, 400], [500, 400], [500, 402], [300, 402]]],
               "size": {"length_mm": 16.0, "width_mm": 0.1}, "location_mm": {"x": 27.0, "y": 27.0}, "near_edge_mm": 7.8}]}
    plan = {"defects": [{"id": "D1", "cls": "corrosion", "kind": "rust", "params": {"pitted": True}},
                        {"id": "D2", "cls": "scratch", "kind": "scriber", "params": {}}]}
    gts = E.gt_objects(sc, (625, 625), "s900000_C0o_v0")
    ds = EL.gt_defects(gts, sc, "true", part=part, plan=plan)
    assert {d["type"] for d in ds} == {"corrosion", "scratch"}
    assert next(d for d in ds if d["type"] == "corrosion").get("pitted") is True
    v = rules.judge_steel(ds, spec, part)[0]
    assert v == records.gt_verdict(sc["defects"], spec, part, plan=plan)
    bare = rules.judge_steel(EL.gt_defects(gts, sc, "true", part=part), spec, part)[0]   # no plan: no pitted flag
    assert rules.VERDICTS.index(bare) < rules.VERDICTS.index(v)
    # a label without any position is kept (placed at its mask box), never silently dropped
    lone = {**sc, "defects": [{k: v_ for k, v_ in sc["defects"][1].items() if k != "location_mm"}]}
    assert [d["type"] for d in EL.gt_defects(E.gt_objects(lone, (625, 625), "x"), lone, "true", part=part)] == ["scratch"]
