"""Real photos and LED bursts go through the twin's own geometry, mask, lightproc and tile code."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from trinetra import tiles as T
from trinetra.twin import frames as F
from trinetra.twin import post as P
from trinetra.twin import realdata as RD
from test_twin_post import (DRAWN, HOLE2, SCRATCH, SIZE, fake_capture, make_scene, read_labels,  # noqa: F401
                             twin_cfg)

pytestmark = pytest.mark.usefixtures("fake_capture")      # the lightproc stand-in of test_twin_post
FIELDS = list(RD.CSV_COLUMNS)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def gt_rows_for_scene(plan: dict, plate="P3", face="A") -> list[dict]:
    """The fixture plan's defects as ground-truth rows (exact polygons), a hole and a distractor."""
    rows = []
    for d in plan["defects"]:
        row = {"defect_id": d["id"], "plate": plate, "face": face, "cls": d["cls"], "kind": d["kind"],
               "poly_mm": json.dumps(d["footprint"]["polys_mm"][0])}
        if d["id"] == "D2":                                   # the scratch as a CSV rectangle instead
            row = {"defect_id": "D2", "plate": plate, "face": face, "cls": "scratch", "kind": "scriber", "u_mm": SCRATCH[0],
                   "v_mm": SCRATCH[1], "length_mm": SCRATCH[2], "width_mm": SCRATCH[3], "angle_deg": SCRATCH[4]}
        rows.append(row)
    return rows + [
        {"defect_id": "H2", "plate": plate, "face": face, "cls": "hole", "u_mm": HOLE2[0], "v_mm": HOLE2[1],
         "diameter_mm": HOLE2[2]},
        {"defect_id": "F1", "plate": plate, "face": face, "cls": "fingerprint", "u_mm": 52, "v_mm": 22,
         "diameter_mm": 8, "eval_class": "ignore", "notes": "distractor"},
    ]


def test_load_ground_truth_and_footprints(tmp_path):
    rows = [{"defect_id": "D1", "plate": "P1", "face": "A", "cls": "dent", "u_mm": 10, "v_mm": 20, "diameter_mm": 2},
            {"defect_id": "S1", "plate": "P1", "face": "A", "cls": "scratch", "u_mm": 30, "v_mm": 40,
             "length_mm": 10, "width_mm": 0.4, "angle_deg": 90},
            {"defect_id": "C1", "plate": "P1", "face": "B", "cls": "crack", "poly_mm": "[[1, 1], [5, 1], [5, 1.3], [1, 1.3]]"},
            {"defect_id": "G1", "plate": "P2", "face": "A", "cls": "corrosion",
             "poly_mm": "[[[0, 0], [4, 0], [4, 4], [0, 4]], [[1, 1], [3, 1], [3, 3], [1, 3]]]"},
            {"defect_id": "#comment"}]
    write_csv(tmp_path / "gt.csv", rows)
    gt = RD.load_ground_truth(tmp_path / "gt.csv")
    assert [r["defect_id"] for r in gt] == ["D1", "S1", "C1", "G1"]
    assert gt[0]["u_mm"] == 10.0 and gt[0]["length_mm"] is None and gt[0]["poly_mm"] is None
    circle = RD.footprint_from_row(gt[0])[0]
    assert cv2.contourArea(circle.astype(np.float32)) == pytest.approx(math.pi, rel=0.01)
    rect = RD.footprint_from_row(gt[1])[0]                       # length along +v at 90 deg
    assert np.ptp(rect[:, 1]) == pytest.approx(10.0) and np.ptp(rect[:, 0]) == pytest.approx(0.4)
    assert np.allclose(rect.mean(axis=0), [30, 40])
    assert len(RD.footprint_from_row(gt[2])) == 1 and len(RD.footprint_from_row(gt[3])) == 2
    assert [r["defect_id"] for r in RD.rows_for(gt, "P1A")] == ["D1", "S1"]
    assert [r["defect_id"] for r in RD.rows_for(gt, "P1")] == ["D1", "S1", "C1"]
    with pytest.raises(ValueError):
        RD.footprint_from_row({"defect_id": "X", "u_mm": 1.0, "v_mm": 2.0, "length_mm": 3.0})
    (tmp_path / "bad.csv").write_text("id,cls\n1,dent\n")
    with pytest.raises(ValueError, match="missing columns"):
        RD.load_ground_truth(tmp_path / "bad.csv")


def test_parse_split_map():
    assert RD.parse_split_map("P1A:train, P4A:val,P5A:test") == {"P1A": "train", "P4A": "val", "P5A": "test"}
    with pytest.raises(ValueError):
        RD.parse_split_map("P1A")


@pytest.mark.parametrize("roll,tilt", [(0, (0, 0)), (90, (8, -6))])
def test_real_path_gives_the_twin_labels(tmp_path, roll, tilt):
    cfg = twin_cfg()
    plan, work, groups = make_scene(tmp_path, roll=roll, tilt=tilt)
    twin = P.process_scene(work, tmp_path / "twin", cfg, debug=False)
    geom = P.grid_geometry(plan, cfg["grid"])
    # the "photo" is the ROI image (no noise), its corners given in plate order
    quiet = {"exposure_p50": 0.2, "wb": [1, 1, 1], "full_well_e": 1e9, "gain": 1, "read_e": 0,
             "unsharp": 0, "motion_px": 0, "noise_seed": 0}
    photo = P.apply_camera(P.compose(groups, {"L": 1, "R": 1, "AMB": 1}), quiet)[0]
    corners = F.apply_h(geom["H_roi_plate"], P.plate_corners(*SIZE))
    write_csv(tmp_path / "gt.csv", gt_rows_for_scene(plan))
    rows = RD.rows_for(RD.load_ground_truth(tmp_path / "gt.csv"), "P3A")
    keep = [r for r in rows if r["defect_id"] != "D2"]            # the twin drops D2 as invisible
    real = RD.process_capture(photo, corners, SIZE, keep, tmp_path / "real", "train", cfg, "photo")

    assert np.allclose(np.array(real["H_grid_plate"]), geom["H_grid_plate"], atol=1e-3)
    assert real["grid"]["size_px"] == twin["grid"]["size_px"] and real["grid"]["datum_corner"] == twin["grid"]["datum_corner"]
    tl, rl = read_labels(tmp_path / "twin", "train", "s000007_v0"), read_labels(tmp_path / "real", "train", "photo")
    assert set(tl) == set(rl)
    for name in tl:
        a, b = sorted(tl[name]), sorted(rl[name])
        assert len(a) == len(b), name
        assert np.allclose(np.array(a), np.array(b), atol=1.5 / 640), name
    assert sum(map(len, rl.values())) > 0
    # the grids match too (same warp of the same image)
    tg = P.rectify(photo, geom)
    rg = cv2.cvtColor(cv2.imread(str(tmp_path / "real" / "grids" / "train" / "photo.png")), cv2.COLOR_BGR2RGB)
    assert np.abs(tg.astype(int) - rg.astype(int)).mean() < 0.5
    side = {d["id"]: d for d in real["defects"]}
    assert set(side) == {*DRAWN, "F1"} and side["F1"]["label"] is False and side["F1"]["tiles"] == []
    assert side["D1"]["contrast"] > 4 and side["D1"]["tiles"] and side["D1"]["type"] == "dent"
    assert [f["id"] for f in real["features"]] == ["H2"] and real["features"][0]["label"] is False
    assert real["arm"] == "single" and real["gate"] is None

    # ground truth is labelled whatever its contrast, unless asked to filter like the twin
    all_rows = RD.process_capture(photo, corners, SIZE, rows, tmp_path / "real2", "train", cfg, "photo")
    assert {d["id"]: d["visible"] for d in all_rows["defects"]}["D2"] is True
    sc_id = T.CLASSES.index("scratch")
    assert any(l[0] == sc_id for ls in read_labels(tmp_path / "real2", "train", "photo").values() for l in ls)
    filt = RD.process_capture(photo, corners, SIZE, rows, tmp_path / "real3", "train", cfg, "photo", filter_visibility=True)
    assert {d["id"]: d["visible"] for d in filt["defects"]}["D2"] is False


def _photo(size_px=(1200, 900)):
    """A grey plate (60 x 45 mm at 15 px/mm, rotated 90 deg: datum at the image's top right)
    on a dark background, with a red patch at plate (45, 10)."""
    W, H = SIZE
    s = 15.0
    H_img_plate = np.array([[0.0, -s, 150 + H * s], [s, 0.0, 100.0], [0.0, 0.0, 1.0]])
    corners = F.apply_h(H_img_plate, P.plate_corners(W, H))
    img = np.full((size_px[1], size_px[0], 3), 20, np.uint8)
    cv2.fillPoly(img, [np.round(corners).astype(np.int32)], (170, 170, 170))
    patch = F.apply_h(H_img_plate, P.rect_poly(45, 10, 4, 4))
    cv2.fillPoly(img, [np.round(patch).astype(np.int32)], (230, 20, 20))      # RGB red
    return img, corners, H_img_plate


def test_cli_real_tiles(tmp_path):
    cfg = twin_cfg()
    cfg_path = tmp_path / "twin.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    img, corners, _ = _photo()
    cap = tmp_path / "captures"
    cap.mkdir()
    cv2.imwrite(str(cap / "p3a_001.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cap / "p3a_002.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cap / "p9a_001.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    (tmp_path / "corners.json").write_text(json.dumps({
        "p3a_001.png": {"coupon": "P3A", "corners_px": corners.tolist(), "plate_size_mm": list(SIZE)},
        "p3a_002.png": {"coupon": "P3A", "datum_image_corner": "TR", "plate_size_mm": list(SIZE)},
        "p9a_001.png": {"coupon": "P9A", "corners_px": corners.tolist()},
    }))
    write_csv(tmp_path / "gt.csv", [
        {"defect_id": "D1", "plate": "P3", "face": "A", "cls": "dent", "u_mm": 20, "v_mm": 30, "diameter_mm": 3},
        {"defect_id": "D9", "plate": "P9", "face": "A", "cls": "dent", "u_mm": 20, "v_mm": 30, "diameter_mm": 3},
    ])
    out = tmp_path / "real"
    RD.main(["--captures", str(cap), "--corners", str(tmp_path / "corners.json"), "--gt", str(tmp_path / "gt.csv"),
             "--out", str(out), "--split-map", "P1A:train,P3A:test", "--config", str(cfg_path)])
    assert not (out / "sidecar" / "p9a_001.json").exists()        # coupon not in the split map
    a = json.loads((out / "sidecar" / "p3a_001.json").read_text())
    b = json.loads((out / "sidecar" / "p3a_002.json").read_text())   # corners auto-detected
    assert a["source"] == "real" and a["coupon"] == "P3A" and a["grid"]["datum_corner"] == "TR"
    assert a["image"].endswith("p3a_001.png")
    probe = np.array([[0, 0], [60, 0], [60, 45], [0, 45], [20, 30]], float)
    assert np.abs(F.apply_h(np.array(a["H_grid_plate"]), probe) - F.apply_h(np.array(b["H_grid_plate"]), probe)).max() < 1.5
    assert [d["id"] for d in a["defects"]] == ["D1"] and a["defects"][0]["tiles"]
    labels = read_labels(out, "test", "p3a_001")
    dent = T.CLASSES.index("dent")
    assert sum(len(v) for v in labels.values()) >= 1 and all(l[0] == dent for v in labels.values() for l in v)
    assert a["arm"] == "single" and a["capture"]["arm"] == "single"
    # colour order: the red patch is still red (no BGR/RGB swap on the way)
    grid = cv2.cvtColor(cv2.imread(str(out / "grids" / "test" / "p3a_001.png")), cv2.COLOR_BGR2RGB)
    gx, gy = F.apply_h(np.array(a["H_grid_plate"]), [[45, 10]])[0]
    r, g, bl = grid[int(round(gy)), int(round(gx))].astype(int)
    assert r > 200 and g < 60 and bl < 60
    data = yaml.safe_load((out / "yolo" / "data.yaml").read_text())
    assert data["test"] == "images/test" and data["names"][1] == "dent"


# ---- LED bursts (arm "burst": lightproc.process_burst, as on the Pi) ----------------------

def test_twin_burst_npz_replays_through_realdata(tmp_path):
    """A twin burst file read back by realdata gives the twin's own labels (same lightproc path)."""
    cfg = twin_cfg(variants_per_scene=1)
    plan, work, _ = make_scene(tmp_path, roll=90, tilt=(8, -6))
    twin = P.process_scene(work, tmp_path / "twin", cfg, debug=False, save_bursts=True)
    npz = tmp_path / "twin" / twin["variants"][0]["burst_npz"]
    frames, extras = RD.load_burst(npz)
    assert set(frames) == set(P.BURST_FRAMES) and frames["B1"].dtype == np.float32
    geom = P.grid_geometry(plan, cfg["grid"])
    assert np.allclose(extras["corners_plate_order_px"], F.apply_h(geom["H_roi_plate"], P.plate_corners(*SIZE)))
    assert extras["plate_size_mm"] == list(SIZE) and extras["meta"]["twin"]["scene_id"] == "s000007"
    rows = [r for r in RD.rows_from_labels(extras["labels"], "T1") if r["defect_id"] != "D2"]   # D2: invisible
    assert {r["defect_id"] for r in rows} == set(DRAWN) and all(r["cls"] in T.CLASSES for r in rows)
    real = RD.process_burst_capture(frames, extras["corners_plate_order_px"], extras["plate_size_mm"], rows,
                                    tmp_path / "real", "train", cfg, "replay")
    assert real["arm"] == "burst" and real["gate"] == "PASS" and real["capture"]["metrics"]["r_ambient"] < 0.3
    assert np.allclose(np.array(real["H_grid_plate"]), geom["H_grid_plate"], atol=1e-3)
    tl, rl = read_labels(tmp_path / "twin", "train", "s000007_v0"), read_labels(tmp_path / "real", "train", "replay")
    assert set(tl) == set(rl) and sum(map(len, rl.values())) > 0
    for name in tl:
        a, b = sorted(tl[name]), sorted(rl[name])
        assert len(a) == len(b), name
        assert np.allclose(np.array(a), np.array(b), atol=1.5 / 640), name
    img = cv2.imread(str(tmp_path / "twin" / "yolo" / "images" / "train" / "s000007_v0_r0c0.jpg")).astype(int)
    rimg = cv2.imread(str(tmp_path / "real" / "yolo" / "images" / "train" / "replay_r0c0.jpg")).astype(int)
    assert np.abs(img - rimg).mean() < 1.0                       # float16 frames: the same image

    # the CLI: no --gt, so the npz's own labels (all labelled defects) are used
    (tmp_path / "corners.json").write_text(json.dumps({npz.name: {"coupon": "T1"}}))
    cfg_path = tmp_path / "twin.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    RD.main(["--captures", str(npz.parent), "--corners", str(tmp_path / "corners.json"), "--out", str(tmp_path / "cli"),
             "--split-map", "T1:val", "--config", str(cfg_path)])
    sc = json.loads((tmp_path / "cli" / "sidecar" / "s000007_v0.json").read_text())
    assert sc["arm"] == "burst" and sc["gate"] == "PASS" and sc["coupon"] == "T1"
    assert {d["id"] for d in sc["defects"]} == {"D2", *DRAWN}
    assert (tmp_path / "cli" / "grids" / "val" / "s000007_v0.png").exists()
    # regression: D2 (never drawn, visible false in the twin's ground truth) is kept unlabelled, so
    # the CLI replay labels exactly what the twin labelled (it used to train on the missing scratch)
    d2 = {d["id"]: d for d in sc["defects"]}["D2"]
    assert d2["label"] is False and d2["tiles"] == [] and d2["notes"] == "twin (invisible)"
    cl = read_labels(tmp_path / "cli", "val", "s000007_v0")
    assert not any(l[0] == T.CLASSES.index("scratch") for ls in cl.values() for l in ls)
    for name in tl:
        assert np.allclose(np.array(sorted(tl[name])), np.array(sorted(cl[name])), atol=1.5 / 640), name


def test_burst_cli_passes_the_capture_meta_to_lightproc(tmp_path, monkeypatch):
    """Regression: real-tiles used to rebuild the Burst meta from the file name and material only,
    dropping the capture's own meta_json, so a Pi burst (demosaiced: True, which sets lightproc's
    noise model for residual_sigma, about 1.5x) was judged as non-demosaiced data."""
    seen = []
    LP = P.capture_modules()[0]
    real_pb = LP.process_burst

    def spy(burst, roi_mask, cfg):
        seen.append(dict(burst.meta))
        return real_pb(burst, roi_mask, cfg)

    monkeypatch.setattr(LP, "process_burst", spy)
    cfg = twin_cfg(variants_per_scene=1)
    cfg_path = tmp_path / "twin.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    plan, _, groups = make_scene(tmp_path)
    corners = F.apply_h(P.grid_geometry(plan, cfg["grid"])["H_roi_plate"], P.plate_corners(*SIZE))
    cap = tmp_path / "captures"
    cap.mkdir()
    f = {n: np.clip(0.5 * (l * groups["L"] + r * groups["R"] + groups["AMB"]), 0, 1).astype(np.float16)
         for n, (l, r) in {"B1": (1, 1), "L": (1, 0), "A": (0, 0), "R": (0, 1), "B2": (1, 1)}.items()}
    pi_meta = {"demosaiced": True, "material": "ss_2b", "exposure_us": 10000}
    np.savez(cap / "pi1.npz", **{f"frame_{n}": a for n, a in f.items()},
             meta_json=np.frombuffer(json.dumps(pi_meta).encode(), np.uint8))
    entries = {"pi1.npz": {"coupon": "P3A", "corners_px": corners.tolist(), "plate_size_mm": list(SIZE)},
               "png1": {"coupon": "P3A", "burst": _write_png_burst(cap, "png1", groups), "corners_px": corners.tolist(),
                        "plate_size_mm": list(SIZE), "meta": {"demosaiced": True}, "material": "galvanised"}}
    (tmp_path / "corners.json").write_text(json.dumps(entries))
    write_csv(tmp_path / "gt.csv", gt_rows_for_scene(plan))
    RD.main(["--captures", str(cap), "--corners", str(tmp_path / "corners.json"), "--gt", str(tmp_path / "gt.csv"),
             "--out", str(tmp_path / "real"), "--split-map", "P3A:train", "--config", str(cfg_path)])
    npz_meta, png_meta = seen
    assert npz_meta["demosaiced"] is True and npz_meta["material"] == "ss_2b" and npz_meta["exposure_us"] == 10000
    assert npz_meta["file"] == "pi1.npz" and npz_meta["source"] == "real"
    assert png_meta["demosaiced"] is True and png_meta["material"] == "galvanised"
    assert RD.burst_meta({"material": "a"}, {"material": "b"}, "x") == {"material": "b", "file": "x"}


def _write_png_burst(folder: Path, stem: str, groups: dict, amb: float = 1.0) -> dict:
    """A burst as 16-bit PNGs (linear, [0, 1] full scale = 0..65535)."""
    state = {"B1": (1, 1), "L": (1, 0), "A": (0, 0), "R": (0, 1), "B2": (1, 1)}
    names = {}
    for n, (l, r) in state.items():
        f = np.clip(0.5 * (l * groups["L"] + r * groups["R"] + amb * groups["AMB"]), 0, 1)
        names[n] = f"{stem}_{n}.png"
        cv2.imwrite(str(folder / names[n]), cv2.cvtColor(np.rint(f * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
    return names


def test_real_png_bursts_and_retake(tmp_path):
    cfg = twin_cfg()
    cfg_path = tmp_path / "twin.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    plan, _, groups = make_scene(tmp_path)
    geom = P.grid_geometry(plan, cfg["grid"])
    corners = F.apply_h(geom["H_roi_plate"], P.plate_corners(*SIZE)).tolist()
    cap = tmp_path / "captures"
    cap.mkdir()
    entries = {"cap1": {"coupon": "P3A", "burst": _write_png_burst(cap, "cap1", groups), "corners_px": corners,
                        "plate_size_mm": list(SIZE)},
               "cap2": {"coupon": "P3A", "burst": _write_png_burst(cap, "cap2", groups, amb=30.0), "corners_px": corners,
                        "plate_size_mm": list(SIZE)}}                       # in the sun: frames saturate
    (tmp_path / "corners.json").write_text(json.dumps(entries))
    write_csv(tmp_path / "gt.csv", gt_rows_for_scene(plan))
    frames, _ = RD.load_burst(entries["cap1"]["burst"], cap)
    assert frames["A"].dtype == np.float32 and np.allclose(frames["A"], np.clip(0.5 * groups["AMB"], 0, 1), atol=1e-4)
    out = tmp_path / "real"
    RD.main(["--captures", str(cap), "--corners", str(tmp_path / "corners.json"), "--gt", str(tmp_path / "gt.csv"),
             "--out", str(out), "--split-map", "P3A:train", "--config", str(cfg_path)])
    ok = json.loads((out / "sidecar" / "cap1.json").read_text())
    assert ok["arm"] == "burst" and ok["gate"] == "PASS" and ok["tiles_written"] and ok["n_labels"] > 0
    assert any(l[0] == T.CLASSES.index("dent") for ls in read_labels(out, "train", "cap1").values() for l in ls)
    sun = json.loads((out / "sidecar" / "cap2.json").read_text())
    assert sun["gate"] == "RETAKE" and sun["gate_message"].startswith("Too bright")
    assert sun["tiles_written"] == [] and sun["n_labels"] == 0 and not read_labels(out, "train", "cap2")
    assert (out / "grids" / "train" / "cap2.png").exists()           # recorded, never silently dropped


def test_load_burst_plain_layout_and_missing_frames(tmp_path):
    f = np.full((4, 5, 3), 0.25, np.float32)
    labels = [{"id": "D1", "cls": "pit", "kind": "single", "polys_mm": [[[1, 1], [2, 1], [2, 2]]], "centre_mm": [1.5, 1.3]},
              {"id": "D9", "cls": "scratch", "label": False, "polys_mm": [[[0, 0], [1, 0], [1, 1]]]}]
    np.savez(tmp_path / "b.npz", **{n: f for n in P.BURST_FRAMES}, corners_plate_order_px=np.zeros((4, 2)),
             labels_json=np.array(json.dumps(labels)))
    frames, extras = RD.load_burst(tmp_path / "b.npz")
    assert set(frames) == set(P.BURST_FRAMES) and extras["labels"][0]["id"] == "D1" and extras["meta"] == {}
    rows = RD.rows_from_labels(extras["labels"], "X1")
    assert [r["defect_id"] for r in rows] == ["D1"] and rows[0]["u_mm"] == 1.5
    assert len(RD.footprint_from_row(rows[0])) == 1
    np.savez(tmp_path / "c.npz", B1=f, L=f)
    with pytest.raises(ValueError, match="missing frames"):
        RD.load_burst(tmp_path / "c.npz")
    assert np.allclose(RD.plate_order([[0, 0], [1, 0], [1, 1], [0, 1]], [1, 2, 3, 0]), [[0, 1], [0, 0], [1, 0], [1, 1]])
