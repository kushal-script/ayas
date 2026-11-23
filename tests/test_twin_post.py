"""Twin post-processing: EXR light groups -> capture arms -> plate grid -> labels, tiles, grids, bursts.

The fixture is rendered with numpy, not Blender: a plate albedo map in plate mm is warped
into the ROI with the plan's own camera, per light group, and written as a multipart EXR
laid out like Blender 5.2 writes it (one part per pass, R/G/B half channels).

The capture method (lightproc.process_burst, burst.simulate_burst / simulate_naive) belongs
to another build, so these tests run post against small deterministic stand-ins with the
same API (FAKE_LP, FAKE_B; gate RETAKE when r_ambient > 2). One test at the end runs the
real modules when they are importable.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from ayas import tiles as T
from ayas.twin import frames as F
from ayas.twin import post as P

ROOT = Path(__file__).resolve().parents[1]
MPP, BORDER = 0.08, 5.0
SIZE = (60.0, 45.0)                      # non-square, so the (W, H) / (H, W) grid choice is tested
RES = 0.04                               # texture mm per texel
DOT = (17.3, 26.2)                       # dark reference dot, never labelled
HOLE1 = (10.0, 10.0, 8.0)                # punched holes: part features, never labelled
HOLE2 = (50.0, 35.0, 4.1)
DENT = (40.0, 20.0, 1.4)                 # u, v, d: drawn dark -> visible dent
SCRATCH = (22.0, 36.0, 8.0, 0.3, 30.0)   # u, v, length, width, angle: labelled but never drawn
PIT = (12.0, 30.0, 1.0)                  # u, v, d: single pit, drawn dark
RUST = (48.0, 9.0, 6.0)                  # u, v, d: rust patch, drawn rust brown
STAIN = (30.0, 12.0, 8.0, 5.0, 20.0)     # u, v, major, minor, angle: oil stain, drawn a little darker
CRACK = (30.5, 26.8, 9.0, 0.3, 5.0)      # u, v, length, width, angle: open crack, drawn dark
INDICATION = (30.0, 12.0, 6.0, 0.4, 0.0)  # penetrant linear indication (u, v, length, width, angle)
DRAWN = {"D1": "dent", "D3": "pit", "D4": "corrosion", "D5": "stain", "D6": "crack"}
CENTRES = {"D1": DENT[:2], "D3": PIT[:2], "D4": RUST[:2], "D5": STAIN[:2], "D6": CRACK[:2]}


# ---- stand-ins for lightproc and burst (same API, DESIGN_STEEL E) --------------------

@dataclass
class GateResult:
    verdict: str
    reasons: list = field(default_factory=list)
    message: str = ""


@dataclass
class Burst:
    frames: dict
    meta: dict


@dataclass
class Processed:
    image_u8: np.ndarray
    M_lin: np.ndarray
    relief: np.ndarray
    sat_mask: np.ndarray
    shifts: dict
    metrics: dict
    gate: GateResult


def _load_capture_config(path="configs/capture.yaml"):
    return yaml.safe_load(Path(path).read_text())


def _process_burst(burst, roi_mask, cfg):
    f = burst.frames
    A = f["A"]
    Lc, Rc = np.maximum(f["L"] - A, 0), np.maximum(f["R"] - A, 0)
    yl, yr = P.luma_lin(Lc), P.luma_lin(Rc)
    M = np.where((yl <= yr)[..., None], Lc, Rc) / np.asarray(cfg["leds"]["colour"], np.float32)
    p = max(float(np.percentile(P.luma_lin(M)[roi_mask], 99.5)), 1e-6)
    img = P.to_u8(np.clip(M / p, 0, 1) ** (1 / 2.2))
    led = float(np.median(P.luma_lin(f["B1"] - A)[roi_mask]))
    r = float(np.median(P.luma_lin(A)[roi_mask])) / max(led, 1e-9)
    verdict = "RETAKE" if r > 2 else "PASS_FLAGGED" if r > 0.3 else "PASS"
    msg = cfg["gate"]["messages"]["too_bright"] if verdict == "RETAKE" else ""
    return Processed(img, M, (yl - yr) / (yl + yr + 1e-4), np.zeros(yl.shape, bool), {"B2": (0.0, 0.0)},
                     {"r_ambient": r, "led_level": led, "motion_px": 0.0},
                     GateResult(verdict, ["r_ambient"] if verdict == "RETAKE" else [], msg))


def _simulate_burst(groups, plan, rng, capture_cfg, roi_mask=None):
    seq, state = capture_cfg["burst"]["sequence"], capture_cfg["burst"]["led_state"]
    frames = {}
    for name in seq:
        l, r = state[name]
        s = 0.5 * (l * groups["L"] + r * groups["R"] + groups["AMB"])
        frames[name] = np.clip(s + 0.002 * rng.standard_normal(s.shape, dtype=np.float32), 0, 1).astype(np.float32)
    return Burst(frames, {"sequence": list(seq), "k": 0.5})


def _simulate_naive(groups, plan, rng, capture_cfg):
    img = groups["L"] + groups["R"] + groups["AMB"]
    y = P.luma_lin(img)
    img = img * np.float32(0.18 / np.median(y[y > 0.02]))
    return P.to_u8(P.srgb_oetf(np.clip(img, 0, 1)))


FAKE_LP = SimpleNamespace(Burst=Burst, Processed=Processed, GateResult=GateResult, process_burst=_process_burst,
                          load_capture_config=_load_capture_config, naive_process=lambda x: x)
FAKE_B = SimpleNamespace(simulate_burst=_simulate_burst, simulate_naive=_simulate_naive)


@pytest.fixture(autouse=True)
def fake_capture(monkeypatch):
    monkeypatch.setattr(P, "capture_modules", lambda: (FAKE_LP, FAKE_B))


# ---- fixture scene ------------------------------------------------------------------

def twin_cfg(**post) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "twin.yaml").read_text())
    cfg["capture_config"] = str(ROOT / "configs" / "capture.yaml")
    cfg["post"]["empty_tile_keep"] = 1.0
    cfg["post"].update(post)
    return cfg


def expected_grid(u: float, v: float, roll: float) -> tuple[float, float]:
    """Grid px of plate (u, v) worked out by hand: roll r turns the image by r clockwise,
    so the plate corner imaged top-left is datum (0), (0, H) (90), (W, H) (180), (W, 0) (270)."""
    W, H = SIZE
    k = {0: 0, 90: 3, 180: 2, 270: 1}[int(round(roll / 90)) * 90 % 360]   # a few degrees of jitter rectify away
    up, vp = {0: (u, v), 1: (v, W - u), 2: (W - u, H - v), 3: (H - v, u)}[k]
    return (up + BORDER) / MPP, (vp + BORDER) / MPP


def _defect(did, cls, kind, poly, measure, near=None):
    return {"id": did, "cls": cls, "kind": kind, "params": {},
            "footprint": {"polys_mm": [np.asarray(poly).tolist()], "measure": measure},
            "near": near or {"plate_edge_mm": 9.0, "hole_edge_mm": 12.0}, "label": True}


def make_plan(roll=0.0, tilt=(0.0, 0.0), mode="surface", seed=7, scene_id="s000007", distance=200.0,
              split="train", sheets=None, condition="C1_office", lux=450.0, device_mode="open") -> dict:
    W, H = SIZE
    K = F.nominal_K()
    R, t = F.camera_pose((W / 2, H / 2), distance, tilt[0], tilt[1], roll)
    c = F.apply_h(F.homography_plate_to_image(K, R, t), P.plate_corners(W, H))
    x0, y0 = (np.floor(c.min(0)) - 48).astype(int)
    x1, y1 = (np.ceil(c.max(0)) + 48).astype(int)
    plan = {
        "schema": "ayas.twin.scene/2", "scene_id": scene_id, "seed": seed, "split": split,
        "mode": mode, "source": "fixture",
        "plate": {"size_mm": [W, H], "thickness_mm": 3.0, "material": {"kind": "ms_cold_rolled"}},
        "sheets": sheets or [],
        "holes": [{"id": "H1", "u": HOLE1[0], "v": HOLE1[1], "d_mm": HOLE1[2], "ovality": 1.0, "angle_deg": 0.0,
                   "sheet": (sheets or [{}])[0].get("id"), "filled": False},
                  {"id": "H2", "u": HOLE2[0], "v": HOLE2[1], "d_mm": HOLE2[2], "ovality": 1.0, "angle_deg": 0.0,
                   "sheet": None, "filled": False}],
        "rivets": [],
        "defects": [
            _defect("D1", "dent", "punch", P.circle_poly(*DENT), {"diameter_mm": DENT[2], "depth_mm": 0.2},
                    {"plate_edge_mm": 19.3, "hole_edge_mm": 12.0}),
            _defect("D2", "scratch", "scriber", P.rect_poly(*SCRATCH), {"length_mm": SCRATCH[2], "width_mm": SCRATCH[3]}),
            _defect("D3", "pit", "single", P.circle_poly(*PIT), {"diameter_mm": PIT[2], "count": 1}),
            _defect("D4", "corrosion", "rust", P.circle_poly(*RUST), {"area_mm2": 28.3, "diameter_mm": RUST[2]}),
            _defect("D5", "stain", "oil", P.ellipse_poly(*STAIN), {"area_mm2": 31.4, "diameter_mm": 6.3}),
            _defect("D6", "crack", "open", P.rect_poly(*CRACK), {"length_mm": CRACK[2], "width_mm": CRACK[3]}),
        ],
        "distractors": [{"kind": "fingerprint", "params": {"u": 52, "v": 22, "r_mm": 5, "angle_deg": 0}}],
        "penetrant": None,
        "camera": {"K": K.tolist(), "full_res": [4056, 3040], "R": R.tolist(), "t": t.tolist(),
                   "roi": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], "fstop": 1.79, "focus_m": 0.2,
                   "pose": {"target_mm": [W / 2, H / 2], "distance_mm": distance, "tilt_deg": list(tilt), "roll_deg": roll}},
        "ambient": {"condition": condition, "lux": lux, "cct_k": 4500.0},
        "extras": {"device": {"mode": device_mode}},
        "capture": {"sensor": "full", "shake_px_per_frame": 0.0, "led_jitter": 0.0},
        "render": {"samples": 16, "texture_res_mm": RES, "clean_pass": False},
    }
    if mode == "penetrant":
        ind = P.rect_poly(*INDICATION)
        plan["penetrant"] = {"developer": {"albedo": 0.88}, "indications": [
            {"id": "I1", "cls": "linear_indication", "from_defect": "D2",
             "params": {}, "footprint": {"polys_mm": [ind.tolist()], "measure": {"length_mm": INDICATION[2], "width_mm": INDICATION[3]}}}]}
    return plan


def _inside(poly_mm, uu, vv) -> np.ndarray:
    pts = (np.asarray(poly_mm) / RES - 0.5).astype(np.float32)
    m = np.zeros(uu.shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    return m.astype(bool)


def plate_albedo(draw_dent=True, penetrant=False) -> np.ndarray:
    W, H = SIZE
    nu, nv = round(W / RES), round(H / RES)
    uu, vv = np.meshgrid((np.arange(nu) + 0.5) * RES, (np.arange(nv) + 0.5) * RES)
    a = 0.55 + 0.01 * np.random.default_rng(0).standard_normal((nv, nu))
    a[np.hypot(uu - DOT[0], vv - DOT[1]) <= 0.5] = 0.03
    for hu, hv, hd in (HOLE1, HOLE2):
        a[np.hypot(uu - hu, vv - hv) <= hd / 2] = 0.35
    img = a[..., None] * np.array([1.0, 0.98, 0.95])
    if draw_dent:
        img[np.hypot(uu - DENT[0], vv - DENT[1]) <= DENT[2] / 2] = 0.2
        img[np.hypot(uu - PIT[0], vv - PIT[1]) <= PIT[2] / 2] = 0.1
        img[np.hypot(uu - RUST[0], vv - RUST[1]) <= RUST[2] / 2] = [0.35, 0.15, 0.06]
        img[_inside(P.ellipse_poly(*STAIN), uu, vv)] *= 0.72
        img[_inside(P.rect_poly(*CRACK), uu, vv)] = 0.1
    if penetrant:
        u, v, length, width, _ = INDICATION
        red = (np.abs(uu - u) <= length / 2) & (np.abs(vv - v) <= width / 2)
        img[red] = [0.55, 0.05, 0.06]
    return img.astype(np.float32)


def render_groups(plan: dict, albedo: np.ndarray, K_img=None, size=None, amb=0.3) -> dict[str, np.ndarray]:
    """Warp the texture (texel centres at (c + 0.5) * RES mm) into the ROI per light group
    (or into an image of the given size with intrinsics K_img)."""
    c = plan["camera"]
    x0, y0, w, h = c["roi"]
    K_img = F.roi_K(c["K"], x0, y0) if K_img is None else K_img
    w, h = size or (w, h)
    S = np.array([[RES, 0, 0.5 * RES], [0, RES, 0.5 * RES], [0, 0, 1]])
    Hm = F.homography_plate_to_image(K_img, c["R"], c["t"]) @ S
    img = cv2.warpPerspective(albedo, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ramp = np.linspace(1.0, 0.0, w, dtype=np.float32)[None, :, None]
    return {"L": img * (0.6 + 0.4 * ramp), "R": img * (0.6 + 0.4 * (1 - ramp)), "AMB": img * np.float32(amb)}


def write_exr(path: Path, groups: dict[str, np.ndarray]) -> None:
    import OpenEXR

    def hdr():   # a fresh dict per part: Part() stores the part name in the header it is given
        return {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}

    parts = [OpenEXR.Part(hdr(), {f"ViewLayer.Combined_{g}": np.ascontiguousarray(a.astype(np.float16))}, f"ViewLayer.Combined_{g}")
             for g, a in groups.items()]
    comb = sum(groups.values())
    rgba = np.ascontiguousarray(np.dstack([comb, np.ones(comb.shape[:2], np.float32)]).astype(np.float16))
    parts.append(OpenEXR.Part(hdr(), {"ViewLayer.Combined": rgba}, "ViewLayer.Combined"))
    OpenEXR.File(parts).write(str(path))


def make_scene(tmp: Path, scene_id="s000007", draw_dent=True, clean=False, penetrant=False, amb=0.3, **plan_kw):
    plan = make_plan(scene_id=scene_id, mode="penetrant" if penetrant else "surface", **plan_kw)
    work = tmp / "work" / scene_id
    work.mkdir(parents=True)
    (work / "plan.json").write_text(json.dumps(plan))
    groups = render_groups(plan, plate_albedo(draw_dent, penetrant), amb=amb)
    write_exr(work / "render.exr", groups)
    if clean:
        write_exr(work / "render_clean.exr", render_groups(plan, plate_albedo(False, penetrant), amb=amb))
    return plan, work, groups


def read_labels(out: Path, split: str, stem: str) -> dict[str, list[list[float]]]:
    labels = {}
    for p in sorted((out / "yolo" / "labels" / split).glob(f"{stem}_r*c*.txt")):
        labels[p.stem[len(stem) + 1 :]] = [list(map(float, line.split())) for line in p.read_text().splitlines()]
    return labels


def dot_centroid(grid_lin: np.ndarray, guess: tuple[float, float], win: int = 25) -> tuple[float, float]:
    gx, gy = int(round(guess[0])), int(round(guess[1]))
    y = P.luma_lin(grid_lin)[gy - win : gy + win + 1, gx - win : gx + win + 1]
    wgt = np.clip(0.5 * np.median(y) - y, 0, None)
    ys, xs = np.mgrid[gy - win : gy + win + 1, gx - win : gx + win + 1]
    return float((wgt * xs).sum() / wgt.sum()), float((wgt * ys).sum() / wgt.sum())


def assert_labels_where_expected(labels: dict, tiles: list, sidecar: dict, roll: float) -> None:
    """Every drawn defect: one box of its class in every tile that holds it whole, centred
    where the plate puts it and as big as its mask box."""
    boxes = {d["id"]: d["bbox_grid"] for d in sidecar["defects"]}
    for did, cls in DRAWN.items():
        gx, gy = expected_grid(*CENTRES[did], roll)
        x0, y0, x1, y1 = boxes[did]
        assert abs((x0 + x1) / 2 - (gx + 0.5)) < 1.0 and abs((y0 + y1) / 2 - (gy + 0.5)) < 1.0, did
        hits = 0
        for t in tiles:
            if not (t.x0 + 2 < x0 and x1 < t.x1 - 2 and t.y0 + 2 < y0 and y1 < t.y1 - 2):
                continue
            hits += 1
            mine = [l for l in labels[t.name] if l[0] == T.CLASSES.index(cls)
                    and abs(l[1] * 640 - ((x0 + x1) / 2 - t.x0)) < 1.0 and abs(l[2] * 640 - ((y0 + y1) / 2 - t.y0)) < 1.0]
            assert len(mine) == 1, (did, t.name, labels[t.name])
            w, h = max(x1 - x0, 8), max(y1 - y0, 8)
            assert abs(mine[0][3] * 640 - w) < 1.0 and abs(mine[0][4] * 640 - h) < 1.0, did
        assert hits >= 1, did


# ---- EXR ---------------------------------------------------------------------------

def test_read_light_groups_round_trip_and_missing_group(tmp_path):
    rng = np.random.default_rng(1)
    groups = {g: rng.random((6, 9, 3)).astype(np.float32) for g in ("L", "R", "AMB")}
    write_exr(tmp_path / "r.exr", groups)
    out = P.read_light_groups(tmp_path / "r.exr")
    assert set(out) == {"L", "R", "UV", "AMB"}
    for g in groups:
        assert out[g].dtype == np.float32 and np.allclose(out[g], groups[g], atol=1e-3)
    assert not out["UV"].any() and out["UV"].shape == (6, 9, 3)


def test_read_light_groups_split_channels(tmp_path):
    import OpenEXR

    px = np.random.default_rng(2).random((5, 7, 3)).astype(np.float16)
    chans = {f"ViewLayer.Combined_R.{c}": np.ascontiguousarray(px[..., i]) for i, c in enumerate("RGB")}
    chans["ViewLayer.Combined.R"] = np.ascontiguousarray(px[..., 0] * 9)          # not a light group
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
                 chans).write(str(tmp_path / "s.exr"))
    out = P.read_light_groups(tmp_path / "s.exr", ("L", "R"))
    assert np.allclose(out["R"], px.astype(np.float32)) and not out["L"].any()


# ---- legacy compose and camera model (tools and the real-data photo fixture) -----------

def test_compose_both_rgb_and_lr3():
    one = np.ones((2, 2, 3), np.float32)
    groups = {"L": one * 0.2, "R": np.zeros_like(one), "AMB": one * 0.1, "UV": one * 5}
    gains = {"L": 1.0, "R": 1.0, "AMB": 1.0, "UV": 0.0}
    assert np.allclose(P.compose(groups, gains, "both_rgb"), 0.3)
    lr3 = P.compose(groups, gains, "lr3")
    assert np.allclose(lr3[..., 0], 0.3, atol=1e-5) and np.allclose(lr3[..., 1], 0.1, atol=1e-5)
    assert np.allclose(lr3[..., 2], 0.5 + 0.5 * 0.2 / 0.4, atol=1e-3)
    groups["R"] = groups["L"]
    assert np.allclose(P.compose(groups, gains, "lr3")[..., 2], 0.5)
    with pytest.raises(ValueError):
        P.compose(groups, gains, "rgbx")


def test_camera_model_exposure_deterministic_and_noise():
    img = np.full((64, 64, 3), 0.05, np.float32)
    img[:, 32:] = 0.4
    quiet = {"exposure_p50": [0.2, 0.2], "wb_jitter": 0.0, "noise": {"full_well_e": 1e9, "gain": [1, 1], "read_e": [0, 0]},
             "unsharp": [0, 0], "motion_blur_px": [0, 0]}
    mask = np.zeros((64, 64), bool)
    mask[:, 32:] = True
    out = P.camera_model(img, np.random.default_rng(0), quiet, mask)
    assert out.dtype == np.uint8 and out.shape == img.shape
    lin = P.srgb_eotf(out[:, 40:].astype(np.float32) / 255)
    assert np.median(lin) == pytest.approx(0.2, rel=0.02)          # plate median luma hits the target
    noisy = twin_cfg()["post"]
    a = P.camera_model(img, np.random.default_rng(5), noisy, mask)
    b = P.camera_model(img, np.random.default_rng(5), noisy, mask)
    assert np.array_equal(a, b) and a[:, 40:50].std() > 0.5          # deterministic, and noisy


def test_motion_kernel_normalised_and_long_enough():
    k = P.motion_kernel(3.0, 0.0)
    assert k.sum() == pytest.approx(1.0) and k.shape == (5, 5)
    assert np.allclose(k[[0, 1, 3, 4]], 0) and (k[2] > 0).sum() >= 3     # horizontal line
    assert P.motion_kernel(0.2, 10) is None


# ---- geometry ----------------------------------------------------------------------

@pytest.mark.parametrize("roll,tilt", [(0, (0, 0)), (90, (0, 0)), (180, (0, 0)), (270, (0, 0)),
                                       (90, (8, -6)), (3, (-5, 7))])
def test_rectified_grid_puts_the_dot_where_expected(roll, tilt):
    cfg = twin_cfg()
    plan = make_plan(roll=roll, tilt=tilt)
    geom = P.grid_geometry(plan, cfg["grid"])
    W, H = SIZE
    swapped = round(roll) % 180 == 90
    assert geom["grid_plate_mm"] == ((H, W) if swapped else (W, H))
    assert geom["size"] == (round((geom["grid_plate_mm"][0] + 10) / MPP), round((geom["grid_plate_mm"][1] + 10) / MPP))
    exp = expected_grid(*DOT, roll)
    assert np.allclose(F.apply_h(geom["H_grid_plate"], [DOT])[0], exp, atol=1e-3)
    datum = {0: "TL", 90: "TR", 180: "BR", 270: "BL"}[int(round(roll / 90) * 90) % 360]
    assert geom["datum_corner"] == datum
    groups = render_groups(plan, plate_albedo())
    grid = P.rectify(P.compose(groups, {"L": 1, "R": 1, "AMB": 1}), geom)
    assert grid.shape[:2] == geom["shape"]
    cx, cy = dot_centroid(grid, exp)
    assert math.hypot(cx - exp[0], cy - exp[1]) < 1.5


@pytest.mark.parametrize("res_scale", [0.5, 0.37])
def test_rectify_handles_a_reduced_resolution_render(res_scale):
    """render --res-scale: Blender (sensor_fit AUTO) scales pixel edges about the image
    centre by s = max(ws, hs) / max(w, h), pixel centres staying on integers."""
    plan = make_plan(roll=90, tilt=(4, 3))
    geom = P.grid_geometry(plan, twin_cfg()["grid"])
    x0, y0, w, h = plan["camera"]["roi"]
    ws, hs = math.floor(w * res_scale + 0.5), math.floor(h * res_scale + 0.5)
    s = max(ws, hs) / max(w, h)
    K = F.roi_K(plan["camera"]["K"], x0, y0)
    Ks = np.array([[K[0, 0] * s, 0, ws / 2 - 0.5 + (K[0, 2] + 0.5 - w / 2) * s],
                   [0, K[1, 1] * s, hs / 2 - 0.5 + (K[1, 2] + 0.5 - h / 2) * s], [0, 0, 1]])
    img = P.compose(render_groups(plan, plate_albedo(), Ks, (ws, hs)), {"L": 1, "R": 1, "AMB": 1})
    exp = expected_grid(*DOT, 90)
    cx, cy = dot_centroid(P.rectify(img, geom), exp)
    assert math.hypot(cx - exp[0], cy - exp[1]) < 0.5
    mask = P.plate_mask_image(geom, img.shape)
    assert mask.shape == img.shape[:2] and abs(mask.mean() - (P.luma_lin(img) > 0.05).mean()) < 0.01


def test_mirrored_corners_are_rejected():
    H_img_plate = np.diag([-10.0, 10.0, 1.0])            # u flipped: a view from behind the plate
    H_img_plate[0, 2] = 900
    with pytest.raises(ValueError, match="mirrored"):
        P.grid_geometry(H_img_plate, {"mm_per_px": 0.08, "border_mm": 5}, plate_size_mm=SIZE)


def test_raised_features_are_projected_through_the_camera():
    sheet = {"id": "S1", "rect_mm": [0.0, 0.0, 60.0, 20.0], "thickness_mm": 1.0}
    plan = make_plan(tilt=(10, 0), sheets=[sheet])
    geom = P.grid_geometry(plan, twin_cfg()["grid"])
    assert P.surface_height(plan, [[10, 10], [10, 30]]).tolist() == [1.0, 0.0]
    obj = {o["id"]: o for o in P.object_masks(plan, geom, geom["shape"])}
    c = plan["camera"]
    img = F.project(c["K"], c["R"], c["t"], F.plate_to_world([HOLE1[:2]], 1.0))
    on_sheet = F.apply_h(geom["H_grid_img"], img)[0]
    planar = F.apply_h(geom["H_grid_plate"], [HOLE1[:2]])[0]
    centre = obj["H1"]["poly_grid"][0].mean(axis=0)
    assert np.allclose(centre, on_sheet, atol=0.05) and np.linalg.norm(on_sheet - planar) > 0.5


def test_holes_are_unlabelled_part_features():
    plan = make_plan()
    plan["rivets"] = [{"id": "R1", "hole": "H2", "head_d_mm": 8.0}]         # a v1 plan's rivet
    plan["holes"][1]["filled"] = True
    items = {it["id"]: it for it in P.plan_items(plan)}
    assert items["H1"]["group"] == items["R1"]["group"] == "feature"
    assert items["H1"]["cls"] == "hole" and items["R1"]["cls"] == "rivet"
    assert not items["H1"]["label"] and not items["R1"]["label"] and "H2" not in items
    assert all(items[d]["label"] and items[d]["group"] == "defect" for d in ("D1", "D2", *DRAWN))


# ---- visibility ---------------------------------------------------------------------

def test_record_location_is_the_box_centre_like_the_pi():
    """Regression: record_fields' location_mm was the footprint's vertex mean, which for a rust
    streak (many vertices on its rounded head, one at the tip) lay 4-8 mm from the box centre
    the Pi reports (app detection_record) -- enough to flip the rules' pit-in-rust pairing."""
    plan = make_plan()
    a = np.radians(np.linspace(90, 270, 33))
    streak = np.r_[np.c_[12 + 3 * np.cos(a), 30 + 3 * np.sin(a)], [[30.0, 30.0]]]   # head at u 9-12, tip at u 30
    plan["defects"] = [_defect("S1", "corrosion", "rust_streak", streak, {"area_mm2": 40.0, "length_mm": 21.0})]
    geom = P.grid_geometry(plan, {"mm_per_px": MPP, "border_mm": BORDER})
    obj = next(o for o in P.object_masks(plan, geom, geom["shape"]) if o["id"] == "S1")
    assert obj["centre_mm"][0] < 13.0                                   # the vertex mean sits in the head
    rec = P.record_fields(obj, geom)
    x0, y0, x1, y1 = rec["box_px"]
    assert rec["location_mm"]["x"] == pytest.approx(((x0 + x1) / 2 - 0.5) * MPP - BORDER, abs=0.01)
    assert rec["location_mm"]["x"] == pytest.approx(19.5, abs=0.1) and rec["location_mm"]["y"] == pytest.approx(30.0, abs=0.1)


def _obj(mask_full, cls="dent", width_px=None):
    ys, xs = np.nonzero(mask_full)
    x0, y0 = xs.min(), ys.min()
    crop = mask_full[y0 : ys.max() + 1, x0 : xs.max() + 1]
    return {"id": "X", "cls": cls, "label": True, "mask": crop, "origin": (int(x0), int(y0)),
            "n_px": int(crop.sum()), "width_px": width_px}


@pytest.mark.parametrize("shape,width", [("disc", 150.0), ("line", 2.0)])
def test_local_contrast_visibility_small_and_wide(shape, width):
    cfg = twin_cfg()
    rng = np.random.default_rng(3)
    base = np.clip(120 + 3 * rng.standard_normal((600, 600)), 0, 255)
    yy, xx = np.mgrid[:600, :600]
    if shape == "disc":                                     # a 12 mm dent: wider than any 31-px median
        m = np.hypot(xx - 300, yy - 300) <= 75
    else:
        m = (np.abs(yy - 300) <= 1) & (np.abs(xx - 300) <= 150)
    obj = _obj(m, width_px=width)
    plain = np.dstack([base] * 3).astype(np.uint8)
    dark = plain.copy()
    dark[m] = np.clip(dark[m].astype(int) - 30, 0, 255)
    vis, con = P.visibility(dark, None, obj, cfg)
    assert vis and con > 10
    vis, con = P.visibility(plain, None, obj, cfg)
    assert not vis and abs(con) < cfg["visibility"]["local_threshold"]
    assert not P.visibility(plain, None, dict(obj, cls="stain"), cfg)[0]        # always_visible is empty
    cfg["visibility"]["always_visible"] = ["stain"]
    assert P.visibility(plain, None, dict(obj, cls="stain"), cfg)[0]


def test_ring_ignores_neighbouring_holes():
    """A faint scratch 3 px off a textured hole: the hole is not its background."""
    cfg = twin_cfg()
    plan = make_plan()
    geom = P.grid_geometry(plan, cfg["grid"])
    r = {o["id"]: o for o in P.object_masks(plan, geom, geom["shape"])}["H1"]
    grid = np.full((*geom["shape"], 3), 120, np.uint8)
    x0, y0 = r["origin"]
    sub = grid[y0 : y0 + r["mask"].shape[0], x0 : x0 + r["mask"].shape[1]]
    speckle = np.random.default_rng(4).choice([60, 180], size=r["mask"].shape)
    sub[r["mask"]] = speckle[r["mask"], None]                # the bench seen through the hole, its wall glints
    line = np.zeros(geom["shape"], bool)
    cx, cy = r["poly_grid"][0].mean(axis=0)
    line[int(cy) - 1 : int(cy) + 2, int(cx) + 53 : int(cx) + 90] = True
    grid[line] = 105                                         # 15 DN darker than the plate
    obj = _obj(line, "scratch", 3.0)
    vis_all, con_all = P.visibility(grid, None, obj, cfg, None)
    vis, con = P.visibility(grid, None, obj, cfg, P.ring_exclude([r, obj], geom))
    assert vis and con > 10
    assert not vis_all and con_all < con


def test_edge_nick_is_judged_on_the_plate_not_the_edge():
    """A half-disc nick on the top edge: the plate/background step is not its contrast."""
    cfg = twin_cfg()
    plan = make_plan()
    geom = P.grid_geometry(plan, cfg["grid"])
    a = np.linspace(0, np.pi, 33)
    nick = np.c_[30 + 1.25 * np.cos(a), 1.25 * np.sin(a)]          # d 2.5 mm, diameter on v = 0
    obj = P.rasterise_items([{"id": "N", "cls": "dent", "kind": "nick", "group": "defect", "label": True,
                              "polys_mm": [nick], "measure": {"diameter_mm": 2.5}}], geom, geom["shape"])[0]
    grid = np.zeros((*geom["shape"], 3), np.uint8)
    grid[geom["grid"].plate_mask()] = 120
    occ = P.ring_exclude([obj], geom)
    assert not P.visibility(grid, None, obj, cfg, occ)[0]
    x0, y0 = obj["origin"]
    sub = grid[y0 : y0 + obj["mask"].shape[0], x0 : x0 + obj["mask"].shape[1]]
    sub[obj["mask"] & (sub[..., 0] > 0)] = 90                       # the dent shades the plate inside it
    assert P.visibility(grid, None, obj, cfg, occ)[0]


def test_reference_image_takes_the_darker_pod_and_ignores_ambient():
    """min(L, R) by luma removes one pod's mirror highlight; ambient never enters."""
    cap = yaml.safe_load((ROOT / "configs" / "capture.yaml").read_text())
    L = np.full((40, 40, 3), 0.3, np.float32)
    R = L.copy()
    L[10:20, 10:20] = 5.0                                      # glare from the left pod only
    roi = np.ones((40, 40), bool)
    a, na = P.reference_image({"L": L, "R": R, "AMB": np.full_like(L, 9.0)}, roi, cap)
    b, nb = P.reference_image({"L": L, "R": R, "AMB": np.zeros_like(L)}, roi, cap)
    assert np.array_equal(a, b) and a.dtype == np.uint8 and na == nb
    assert abs(int(a[15, 15, 1]) - int(a[30, 30, 1])) <= 1       # no highlight left
    c, nc = P.reference_image({"L": L * 0.5, "R": R * 0.5}, roi, cap, norm=na)   # a clean pass: same scale
    assert nc == na and int(c[30, 30, 1]) < int(a[30, 30, 1])


# ---- labels ------------------------------------------------------------------------

def _line_obj(x0, x1, y, cls="scratch"):
    m = np.zeros((1, x1 - x0), bool)
    m[:] = True
    return {"id": "X", "cls": cls, "label": True, "visible": True, "mask": m, "origin": (x0, y), "n_px": int(m.sum())}


def test_tile_label_rules_keep_frac_min_px_and_padding():
    tiles = T.tile_grid(1375, 1375)
    lab = {"min_box_px": 8, "keep_frac": 0.3, "keep_min_px": 24}
    sc = T.CLASSES.index("scratch")
    out = P.tile_labels([_line_obj(600, 701, 100)], tiles, lab)     # 40 of 101 px in r0c0 (40 %)
    assert out["r0c0"] == [T.yolo_line(sc, (600, 96.5, 640, 104.5))]
    assert out["r0c1"] == [T.yolo_line(sc, (600 - 368, 96.5, 701 - 368, 104.5))]
    out = P.tile_labels([_line_obj(630, 736, 100)], tiles, lab)     # 10 px (9 %), diagonal 10 < 24
    assert out["r0c0"] == [] and len(out["r0c1"]) == 1
    out = P.tile_labels([_line_obj(612, 736, 100)], tiles, lab)     # 28 px (23 %) but diagonal 28 >= 24
    assert len(out["r0c0"]) == 1
    hidden = dict(_line_obj(600, 701, 100), visible=False)
    unlabelled = dict(_line_obj(600, 701, 100), label=False)
    hole = dict(_line_obj(600, 701, 100), cls="hole")
    assert not any(P.tile_labels([hidden, unlabelled, hole], tiles, lab).values())


# ---- ambient sweep ---------------------------------------------------------------------

def test_amb_multiplier_log_uniform_in_the_condition_range():
    post = twin_cfg()["post"]
    amb = {"condition": "C3_window", "lux": 5000.0}
    assert P.amb_multiplier(np.random.default_rng(0), amb, 0, post) == 1.0        # v0 keeps the render
    lux = np.array([5000.0 * P.amb_multiplier(np.random.default_rng([3, k]), amb, 1, post) for k in range(400)])
    assert lux.min() >= 1500 - 1e-6 and lux.max() <= 20000 + 1e-6
    assert np.median(np.log(lux)) == pytest.approx(0.5 * (math.log(1500) + math.log(20000)), abs=0.25)
    assert P.amb_multiplier(np.random.default_rng(0), {"condition": "C0_dark", "lux": 0}, 1, post) == 0.0
    assert P.amb_multiplier(np.random.default_rng(0), {"condition": "C0_dark", "lux": 0}, 0, post) == 0.0
    assert P.amb_multiplier(np.random.default_rng(0), {}, 1, post) == 1.0                 # v1 plan: no condition
    off = dict(post, amb_sweep={"enabled": False})
    assert P.amb_multiplier(np.random.default_rng(0), amb, 1, off) == 1.0
    a, b = np.random.default_rng(9), np.random.default_rng(9)       # one draw whatever the outcome
    P.amb_multiplier(a, amb, 0, post), P.amb_multiplier(b, amb, 1, post)
    assert a.random() == b.random()


# ---- one scene ---------------------------------------------------------------------

@pytest.mark.parametrize("roll,tilt", [(0, (0, 0)), (90, (8, -6))])
def test_process_scene_labels_the_steel_classes_in_the_right_tiles(tmp_path, roll, tilt):
    cfg = twin_cfg()
    plan, work, _ = make_scene(tmp_path, roll=roll, tilt=tilt)
    out = tmp_path / "out"
    sc = P.process_scene(work, out, cfg)
    assert sc["schema"] == "ayas.twin.sidecar/2" and sc["scene_id"] == "s000007" and sc["plate"] == "s000007"
    assert sc["condition"] == "C1_office" and sc["lux"] == 450.0 and sc["device_mode"] == "open"
    assert sc["part"] == {"material": "ms_cold_rolled", "thickness_mm": 3.0, "size_mm": list(SIZE)}
    geom = P.grid_geometry(plan, cfg["grid"])
    assert sc["grid"]["size_px"] == list(geom["size"])
    tiles = T.tile_grid(*geom["size"])
    labels = read_labels(out, "train", "s000007_v0")
    assert set(labels) == {t.name for t in tiles}                  # empty_tile_keep = 1
    for t in tiles:
        img = cv2.imread(str(out / "yolo" / "images" / "train" / f"s000007_v0_{t.name}.jpg"))
        assert img.shape == (640, 640, 3)
    assert_labels_where_expected(labels, tiles, sc, roll)

    every = [l for ls in labels.values() for l in ls]
    assert not any(l[0] == T.CLASSES.index("scratch") for l in every)   # the undrawn scratch is dropped
    assert {int(l[0]) for l in every} == {T.CLASSES.index(c) for c in DRAWN.values()}   # holes never labelled
    defects = {d["id"]: d for d in sc["defects"]}
    assert all(defects[d]["visible"] is True and defects[d]["contrast"] > 4 for d in DRAWN)
    assert defects["D2"]["visible"] is False and defects["D2"]["tiles"] == []
    assert np.allclose(defects["D1"]["centre_mm"], DENT[:2], atol=0.01)
    d1 = defects["D1"]                                            # record-3.0 ground truth for the rules
    assert d1["type"] == "dent" and d1["size"] == {"diameter_mm": DENT[2]} and d1["near_edge_mm"] == 12.0
    assert d1["box_px"] == d1["bbox_grid"]
    gx, gy = expected_grid(*DENT[:2], roll)
    assert d1["location_mm"]["x"] == pytest.approx(gx * MPP - BORDER, abs=0.02)
    feats = {f["id"]: f for f in sc["features"]}
    assert set(feats) == {"H1", "H2"} and feats["H1"]["cls"] == "hole" and feats["H1"]["label"] is False
    assert np.allclose(feats["H1"]["centre_px"], expected_grid(*HOLE1[:2], roll), atol=0.01)
    assert "fasteners" not in sc and np.allclose(np.array(sc["H_grid_plate"]), geom["H_grid_plate"])
    v = sc["variants"][0]
    assert v["arm"] == "burst" and v["gate"] == "PASS" and v["metrics"]["r_ambient"] < 0.3
    assert v["condition"] == "C1_office" and v["lux"] == 450.0 and v["amb_mult"] == 1.0
    assert set(v["arms"]) == {"burst"} and v["arms"]["burst"]["gated"] is True and v["arms"]["burst"]["grid"] is None
    assert 250 <= sc["variants"][1]["lux"] <= 700 and sc["variants"][1]["lux"] != 450.0
    assert (out / "debug" / "s000007_v0.jpg").exists()              # first 50 scenes get overlays
    assert not (out / "grids").exists() and not (out / "bursts").exists()
    assert json.loads((out / "sidecar" / "s000007.json").read_text())["defects"][1]["visible"] is False


def test_retake_variants_write_no_tiles_but_are_recorded(tmp_path):
    cfg = twin_cfg()
    _, work, _ = make_scene(tmp_path, condition="C6_sun", lux=50000.0, amb=6.0)   # ambient >> LEDs
    out = tmp_path / "out"
    sc = P.process_scene(work, out, cfg, debug=False)
    assert [v["gate"] for v in sc["variants"]] == ["RETAKE", "RETAKE"]
    for v in sc["variants"]:
        assert v["tiles_written"] == [] and v["n_labels"] == 0 and v["tiles"] == {}
        assert v["gate_message"].startswith("Too bright") and v["metrics"]["r_ambient"] > 2
        assert 30000 <= v["lux"] <= 100000
    assert not (out / "yolo" / "images" / "train").exists() or not list((out / "yolo" / "images" / "train").iterdir())
    assert all(d["visible"] for d in sc["defects"] if d["id"] in DRAWN)   # ground truth is not the gate's
    assert (out / "sidecar" / "s000007.json").exists()


def test_naive_train_arm_has_no_gate(tmp_path):
    """The old method never refuses: under the sun it still makes (bad) training tiles."""
    cfg = twin_cfg(train_arm="naive")
    _, work, _ = make_scene(tmp_path, condition="C6_sun", lux=50000.0, amb=6.0)
    sc = P.process_scene(work, tmp_path / "out", cfg, debug=False)
    v = sc["variants"][0]
    assert v["arm"] == "naive" and v["gate"] == "PASS" and v["arms"]["naive"]["gated"] is False
    assert v["tiles_written"] and set(v["arms"]) == {"naive"} and "median_u8" in v["metrics"]


def test_test_split_writes_full_grids_per_arm_with_sidecars(tmp_path):
    cfg = twin_cfg()
    plan, work, _ = make_scene(tmp_path, split="test", roll=90, condition="C4_side_lamp", lux=2000.0, device_mode="hood")
    plan["matrix"] = {"plate": "s000007", "condition": "C4_side_lamp", "device_mode": "hood", "row": "C4h"}
    (work / "plan.json").write_text(json.dumps(plan))
    out = tmp_path / "out"
    sc = P.process_scene(work, out, cfg, debug=False)
    assert not (out / "yolo" / "images").exists()                 # test scenes: grids, no tiles
    geom = P.grid_geometry(plan, cfg["grid"])
    grids = {}
    for k in range(2):
        for arm in ("burst", "naive"):
            png = out / "grids" / arm / "test" / f"s000007_v{k}.png"
            img = cv2.imread(str(png))
            assert img.shape[:2] == geom["shape"]
            grids[arm, k] = img
            g = json.loads(png.with_suffix(".json").read_text())
            assert g["schema"] == "ayas.twin.grid/1" and g["arm"] == arm and g["k"] == k
            assert g["scene_id"] == "s000007" and g["plate"] == "s000007" and g["split"] == "test"
            assert g["condition"] == "C4_side_lamp" and g["device_mode"] == "hood" and g["matrix"]["row"] == "C4h"
            assert g["gate"] in ("PASS", "PASS_FLAGGED", "RETAKE") and g["gated"] is (arm == "burst")
            assert g["lux"] == pytest.approx(2000.0 * sc["variants"][k]["amb_mult"], rel=1e-4)
            assert g["part"]["material"] == "ms_cold_rolled" and g["grid_image"] == png.name
            d = {x["id"]: x for x in g["defects"]}
            gx, gy = expected_grid(*DENT[:2], 90)
            x0, y0, x1, y1 = d["D1"]["bbox_grid"]                    # labels in grid px
            assert abs((x0 + x1) / 2 - (gx + 0.5)) < 1.0 and abs((y0 + y1) / 2 - (gy + 0.5)) < 1.0
            assert d["D1"]["visible"] is True and d["D2"]["visible"] is False   # the reference decides
            assert "visible_arm" in d["D1"] and d["D1"]["type"] == "dent"
            assert np.allclose(np.array(g["H_grid_plate"]), geom["H_grid_plate"])
        rec = sc["variants"][k]["arms"]
        assert rec["burst"]["grid"] == f"grids/burst/test/s000007_v{k}.png" and rec["naive"]["grid"].endswith(".png")
    assert np.abs(grids["burst", 0].astype(int) - grids["naive", 0].astype(int)).mean() > 2   # two methods


def _load_eval_plates(monkeypatch):
    path = ROOT / "training" / "eval_plates.py"
    spec = importlib.util.spec_from_file_location("eval_plates_for_post_tests", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001  (another build owns training/; skip if it cannot load here)
        pytest.skip(f"training/eval_plates.py does not load: {e}")
    return mod


def test_eval_plates_reads_the_grid_sidecars(tmp_path, monkeypatch):
    """training/eval_plates finds the per-grid sidecar next to each grid and reads its ground truth."""
    E = _load_eval_plates(monkeypatch)
    cfg = twin_cfg(variants_per_scene=1)
    plan, work, _ = make_scene(tmp_path, split="test")
    out = tmp_path / "out"
    P.process_scene(work, out, cfg, debug=False)
    png = out / "grids" / "naive" / "test" / "s000007_v0.png"
    side = E.find_sidecar(png, out / "sidecar")
    assert side == png.with_suffix(".json")
    sc = json.loads(side.read_text())
    gts = E.gt_objects(sc, P.grid_geometry(plan, cfg["grid"])["shape"], png.stem)
    by_key = {g.key.split(":")[-1]: g for g in gts}
    assert set(by_key) == {"D2", *DRAWN} and all(g.key.startswith("s000007:") for g in gts)
    assert all(by_key[d].cls == T.CLASSES.index(c) and by_key[d].visible for d, c in DRAWN.items())
    assert by_key["D2"].visible is False and by_key["D2"].cls == T.CLASSES.index("scratch")


def test_eval_arms_override_and_jpg_grids(tmp_path):
    cfg = twin_cfg(grid_format="jpg", variants_per_scene=1)
    _, work, _ = make_scene(tmp_path)                             # a train scene
    out = tmp_path / "out"
    sc = P.process_scene(work, out, cfg, debug=False, eval_arms=["naive"])
    assert (out / "grids" / "naive" / "train" / "s000007_v0.jpg").exists()
    assert (out / "grids" / "naive" / "train" / "s000007_v0.json").exists()
    assert not (out / "grids" / "burst").exists() and sc["variants"][0]["tiles_written"]
    with pytest.raises(ValueError, match="unknown arm"):
        P.process_scene(work, tmp_path / "x", cfg, debug=False, eval_arms=["hdr"])


def _npz_json(z, key):
    return json.loads(z[key].tobytes().decode())


def test_save_bursts_writes_the_pi_demo_npz(tmp_path):
    """lightproc.save_burst's layout: frame_<name> float16, meta_json bytes, extra arrays."""
    cfg = twin_cfg()
    plan, work, groups = make_scene(tmp_path, roll=180, condition="C2_warm_bulb", lux=900.0)
    out = tmp_path / "out"
    sc = P.process_scene(work, out, cfg, debug=False, save_bursts=True)
    assert sc["variants"][0]["burst_npz"] == "bursts/s000007_v0.npz" and sc["variants"][1]["burst_npz"] is None
    geom = P.grid_geometry(plan, cfg["grid"])
    with np.load(out / "bursts" / "s000007_v0.npz", allow_pickle=False) as z:
        assert [str(s) for s in z["sequence"]] == ["B1", "L", "A", "R", "B2"]
        for n in ("B1", "L", "A", "R", "B2"):
            f = z[f"frame_{n}"]
            assert f.dtype == np.float16 and f.shape == groups["L"].shape and f.max() <= 1
        assert np.allclose(z["corners_px"], geom["corners_roi"]) and list(z["corner_plate_idx"]) == geom["corner_plate_idx"]
        assert list(z["plate_size_mm"]) == list(SIZE) and list(z["grid_plate_mm"]) == list(geom["grid_plate_mm"])
        datum_first = geom["corners_roi"][np.argsort(geom["corner_plate_idx"])]
        assert np.allclose(F.apply_h(z["H_frame_plate"], P.plate_corners(*SIZE)), datum_first)
        assert str(z["condition"]) == "C2_warm_bulb" and float(z["lux"]) == 900.0 and str(z["device_mode"]) == "open"
        assert str(z["material"]) == "ms_cold_rolled" and float(z["thickness_mm"]) == 3.0
        assert float(z["mm_per_px"]) == MPP and float(z["border_mm"]) == BORDER
        labels = {d["id"]: d for d in _npz_json(z, "labels_json")}
        meta = _npz_json(z, "meta_json")
    assert set(labels) == {"D1", "D2", *DRAWN} and labels["D1"]["cls"] == "dent" and labels["D2"]["visible"] is False
    assert np.allclose(np.concatenate(labels["D1"]["polys_mm"]).mean(axis=0), DENT[:2], atol=0.01)
    assert labels["D1"]["bbox_grid"] == {d["id"]: d for d in sc["defects"]}["D1"]["bbox_grid"]
    assert meta["sequence"][0] == "B1" and meta["k"] == 0.5                 # the Burst meta (exposure k)
    tw = meta["twin"]
    assert tw["schema"] == "ayas.twin.burst/1" and tw["k"] == 0 and tw["scene_id"] == "s000007"
    assert tw["gate"]["gate"] == "PASS" and tw["lux"] == 900.0 and tw["part"]["material"] == "ms_cold_rolled"
    assert not list((out / "bursts").glob("*.part*"))


def test_process_scene_is_reproducible_and_variants_differ(tmp_path):
    cfg = twin_cfg(variants_per_scene=2)
    _, work, _ = make_scene(tmp_path, condition="C3_window", lux=4000.0)
    a = P.process_scene(work, tmp_path / "a", cfg, debug=False)
    b = P.process_scene(work, tmp_path / "b", cfg, debug=False)
    assert a["variants"] == b["variants"]
    assert a["variants"][0]["lux"] == 4000.0 and a["variants"][1]["lux"] != 4000.0
    ja = (tmp_path / "a" / "yolo" / "images" / "train" / "s000007_v1_r0c0.jpg").read_bytes()
    jb = (tmp_path / "b" / "yolo" / "images" / "train" / "s000007_v1_r0c0.jpg").read_bytes()
    assert ja == jb
    c = P.process_scene(work, tmp_path / "c", cfg, variant_seed=99, debug=False)
    assert c["variants"][1]["lux"] != a["variants"][1]["lux"]


def test_clean_pass_visibility_uses_the_difference(tmp_path):
    cfg = twin_cfg()
    _, work, _ = make_scene(tmp_path, clean=True)
    sc = P.process_scene(work, tmp_path / "out", cfg, debug=False)
    d = {x["id"]: x for x in sc["defects"]}
    assert sc["clean_pass"] is True
    assert d["D1"]["visible"] is True and d["D1"]["contrast"] > cfg["visibility"]["diff_threshold"]
    assert d["D2"]["visible"] is False and d["D2"]["contrast"] == pytest.approx(0.0, abs=1e-6)
    assert sc["variants"][0]["visible"]["D2"] is False and sc["variants"][0]["visible"]["D1"] is True


def test_penetrant_scene_writes_grids_not_tiles(tmp_path):
    cfg = twin_cfg()
    plan, work, _ = make_scene(tmp_path, scene_id="s000100", penetrant=True, roll=180)
    out = tmp_path / "out"
    sc = P.process_scene(work, out, cfg)
    png = cv2.imread(str(out / "penetrant" / "s000100_v0.png"))
    geom = P.grid_geometry(plan, cfg["grid"])
    assert png.shape[:2] == geom["shape"]
    assert not list((out / "yolo").rglob("*.jpg")) and not list((out / "yolo").rglob("*.txt"))
    ind = sc["indications"]
    assert len(ind) == 1 and ind[0]["cls"] == "linear_indication" and ind[0]["from_defect"] == "D2"
    gx, gy = expected_grid(INDICATION[0], INDICATION[1], 180)
    x0, y0, x1, y1 = ind[0]["bbox_grid"]
    assert abs((x0 + x1) / 2 - (gx + 0.5)) < 1.0 and abs((y0 + y1) / 2 - (gy + 0.5)) < 1.0
    rgb = png[int(gy), int(gx)][::-1].astype(int)                # the red dye is red in the grid
    assert rgb[0] > rgb[1] + 40 and rgb[0] > rgb[2] + 40
    assert sc["variants"][0]["penetrant_png"] == "penetrant/s000100_v0.png"


def test_data_yaml(tmp_path):
    p = P.write_data_yaml(tmp_path, list(T.CLASSES))
    d = yaml.safe_load(p.read_text())
    assert d == {"path": str((tmp_path / "yolo").resolve()), "train": "images/train", "val": "images/val",
                 "names": {i: c for i, c in enumerate(T.CLASSES)}}
    assert d["names"] == {0: "scratch", 1: "dent", 2: "pit", 3: "corrosion", 4: "crack", 5: "stain"}
    (tmp_path / "yolo" / "images" / "twin_test").mkdir(parents=True)
    assert yaml.safe_load(P.write_data_yaml(tmp_path).read_text())["test"] == "images/twin_test"
    with pytest.raises(ValueError):
        P.write_data_yaml(tmp_path, ["rivet", "open_hole", "dent", "scratch", "crack", "corrosion"])


def test_cli_post(tmp_path, capsys):
    cfg = twin_cfg()
    cfg_path = tmp_path / "twin.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    make_scene(tmp_path, scene_id="s000001", split="val")
    make_scene(tmp_path, scene_id="s000002", roll=270, split="test")
    out = tmp_path / "out"
    P.main(["--work", str(tmp_path / "work"), "--out", str(out), "--config", str(cfg_path), "--save-bursts"])
    assert (out / "sidecar" / "s000001.json").exists() and (out / "sidecar" / "s000002.json").exists()
    assert list((out / "yolo" / "labels" / "val").glob("s000001_v0_*.txt"))
    assert (out / "grids" / "naive" / "test" / "s000002_v1.png").exists()
    assert (out / "bursts" / "s000001_v0.npz").exists() and (out / "bursts" / "s000002_v0.npz").exists()
    assert yaml.safe_load((out / "yolo" / "data.yaml").read_text())["names"][5] == "stain"
    assert "C1_office/open" in capsys.readouterr().out


# ---- pipeline and CLI (Blender stubbed: render_batch writes the numpy fixture EXR) -----------

def _stub_blender(monkeypatch, calls: list) -> None:
    from ayas.twin import render as RN
    from ayas.twin import textures as TX

    def render_batch(works, cfg=None, device=None, samples=None, res_scale=None, timeout_s=None,
                     overview=False, overview_samples=None):
        calls.append({"n": len(works), "overview": overview, "samples": samples})
        metas = []
        for w in map(Path, works):
            plan = json.loads((w / "plan.json").read_text())
            write_exr(w / "render.exr", render_groups(plan, plate_albedo()))
            if overview:
                cv2.imwrite(str(w / "overview.png"), np.zeros((8, 8, 3), np.uint8))
            metas.append({"scene_id": plan["scene_id"], "error": "boom" if plan["scene_id"] == "s000009" else None})
        return metas

    monkeypatch.setattr(TX, "write_maps", lambda plan, work: None)
    monkeypatch.setattr(RN, "prepare", lambda work, plan, cfg=None: {})
    monkeypatch.setattr(RN, "render_batch", render_batch)


def _plan_files(folder: Path, specs) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for sid, split in specs:
        p = folder / f"{sid}.json"
        p.write_text(json.dumps(make_plan(scene_id=sid, split=split)))
        paths.append(p)
    return paths


def test_pipeline_run_plans_resumable_disk_guarded_with_bursts_and_overview(tmp_path, monkeypatch):
    from ayas.twin import pipeline as PL

    calls: list = []
    _stub_blender(monkeypatch, calls)
    cfg = twin_cfg(variants_per_scene=1)
    out = tmp_path / "out"
    paths = _plan_files(out / "plans", [("s000001", "train"), ("s000002", "test"), ("s000009", "train")])
    s = PL.run_plans(cfg, paths, out, batch=2, save_bursts=True, overview=True)
    assert s["done"] == 2 and s["skipped"] == 0 and s["stopped"] is None and s["retake_variants"] == 0
    assert s["failed"] == [{"scene": "s000009", "error": "boom"}] and (out / "work" / "s000009").exists()
    assert [c["n"] for c in calls] == [2, 1] and all(c["overview"] for c in calls)
    assert not (out / "work" / "s000001").exists() and (out / "overview" / "s000001.png").exists()
    assert (out / "bursts" / "s000001_v0.npz").exists() and (out / "bursts" / "s000002_v0.npz").exists()
    assert list((out / "yolo" / "images" / "train").glob("s000001_v0_*.jpg"))
    assert (out / "grids" / "naive" / "test" / "s000002_v0.png").exists()
    assert yaml.safe_load((out / "yolo" / "data.yaml").read_text())["names"][0] == "scratch"
    assert len((out / "run_log.jsonl").read_text().splitlines()) == 2
    again = PL.run_plans(cfg, paths, out, batch=2)                       # resumable: only the failed one
    assert again["skipped"] == 2 and again["done"] == 0 and len(calls) == 3
    (out / "sidecar" / "s000001.json").unlink()
    monkeypatch.setattr(PL, "free_gb", lambda path: 0.1)
    low = PL.run_plans(cfg, paths, out)
    assert low["done"] == 0 and "GB free" in low["stopped"] and len(calls) == 3


def test_pipeline_run_forces_condition_mode_split(tmp_path, monkeypatch):
    from ayas.twin import pipeline as PL

    calls, seen = [], {}
    _stub_blender(monkeypatch, calls)

    def write_plans(cfg, n, out_dir, start_seed=0, **kw):
        seen.update(kw, n=n, start_seed=start_seed)
        return _plan_files(Path(out_dir) / "plans", [(f"s{start_seed + i:06d}", "test") for i in range(n)])

    monkeypatch.setattr(PL.sampler, "write_plans", write_plans)
    cfg = twin_cfg(variants_per_scene=1)
    s = PL.run(cfg, n=1, start_seed=5, out=tmp_path / "o", condition="C6_sun", device_mode="hood", split="test")
    assert seen == {"condition": "C6_sun", "device_mode": "hood", "split": "test", "n": 1, "start_seed": 5}
    assert s["done"] == 1 and (tmp_path / "o" / "grids" / "burst" / "test" / "s000005_v0.png").exists()
    seen.clear()
    PL.run(cfg, n=1, start_seed=6, out=tmp_path / "o")
    assert seen == {"n": 1, "start_seed": 6}                     # nothing forced unless asked
    seen.clear()
    assert PL.run(cfg, n=0, out=tmp_path / "o")["done"] == 0 and seen["n"] == 0   # regression: 0 meant n_scenes (1500)


def test_cli_run_and_matrix_pass_the_options(tmp_path, monkeypatch, capsys):
    from ayas.twin import cli as C
    from ayas.twin import matrix as MX
    from ayas.twin import pipeline as PL

    got = {}
    monkeypatch.setattr(PL, "run", lambda cfg, **kw: got.update(kw) or {"done": 0})
    C.main(["run", "--config", str(ROOT / "configs" / "twin.yaml"), "--n", "3", "--condition", "C6_sun",
            "--device-mode", "hood", "--split", "test", "--save-bursts", "--overview", "--samples", "16"])
    assert got["condition"] == "C6_sun" and got["device_mode"] == "hood" and got["split"] == "test"
    assert got["save_bursts"] is True and got["overview"] is True and got["samples"] == 16 and got["n"] == 3

    calls: list = []
    _stub_blender(monkeypatch, calls)
    cfg_path = tmp_path / "twin.yaml"
    cfg_path.write_text(yaml.safe_dump(twin_cfg(variants_per_scene=1)))

    def write_matrix_plans(cfg, n_plates, out, conditions=None, hood_conditions=None, start_seed=900000):
        rows = [("C0_dark", "open"), ("C6_sun", "hood")][: len(conditions or ["a", "b"])]
        paths = _plan_files(Path(out) / "plans", [(f"s{start_seed}_{c[:2]}{m[0]}", "test") for c, m in rows])
        return {"plates": [{"rows": [{"plan": f"plans/{p.name}"} for p in paths]}]}

    monkeypatch.setattr(MX, "write_matrix_plans", write_matrix_plans)
    out = tmp_path / "mx"
    C.main(["matrix", "--config", str(cfg_path), "--n", "1", "--out", str(out), "--start-seed", "900003"])
    text = capsys.readouterr().out
    summary = json.loads(text[text.rindex("{\n"):])
    assert summary["plans"] == 2 and summary["plates"] == 1 and summary["done"] == 2
    assert (out / "sidecar" / "s900003_C0o.json").exists() and (out / "sidecar" / "s900003_C6h.json").exists()
    assert (out / "grids" / "burst" / "test" / "s900003_C6h_v0.png").exists() and len(calls) == 1
    C.main(["matrix", "--config", str(cfg_path), "--n", "1", "--out", str(tmp_path / "mx2"), "--plans-only",
            "--conditions", "C0_dark"])
    assert len(list((tmp_path / "mx2" / "plans").glob("*.json"))) == 1 and len(calls) == 1   # nothing rendered


def test_cli_matrix_plans_only_with_the_real_matrix(tmp_path, capsys):
    """The real twin.matrix + sampler write one plan per (condition, device mode) row."""
    from ayas.twin import cli as C

    C.main(["matrix", "--config", str(ROOT / "configs" / "twin.yaml"), "--n", "1", "--out", str(tmp_path),
            "--plans-only", "--conditions", "C0_dark", "C1_office", "C6_sun", "--hood-conditions", "C6_sun"])
    plans = sorted(p.stem for p in (tmp_path / "plans").glob("*.json"))
    assert len(plans) == 4 and all(p.startswith("s900000_") for p in plans)
    assert json.loads(capsys.readouterr().out)["plans"] == 4


def test_matrix_condition_subset_keeps_only_its_hood_rows(tmp_path, capsys):
    """Regression: `matrix --conditions C0_dark C1_office` (no --hood-conditions) crashed in
    twin.matrix.rows_for, because the default hood rows C4/C6 were not among the conditions.
    The hood rows are now the default hood conditions that were asked for: every one of them
    since device v2 (the hood head is the product configuration)."""
    from ayas.twin import cli as C
    from ayas.twin import pipeline as PL

    cfg = yaml.safe_load((ROOT / "configs" / "twin.yaml").read_text())
    assert PL.run_matrix(cfg, 1, tmp_path / "a", conditions=["C0_dark", "C1_office"], plans_only=True)["plans"] == 4
    PL.run_matrix(cfg, 1, tmp_path / "b", conditions=["C0_dark", "C6_sun"], plans_only=True)
    assert sorted(p.stem for p in (tmp_path / "b" / "plans").glob("*.json")) == \
        ["s900000_C0h", "s900000_C0o", "s900000_C6h", "s900000_C6o"]
    s = PL.run_matrix(cfg, 1, tmp_path / "c", conditions=["C0_dark", "C6_sun"], hood_conditions=[], plans_only=True)
    assert s["plans"] == 2                                          # an explicit empty list still means no hood rows
    C.main(["matrix", "--config", str(ROOT / "configs" / "twin.yaml"), "--n", "1", "--out", str(tmp_path / "d"),
            "--plans-only", "--conditions", "C0_dark", "C4_side_lamp"])
    assert json.loads(capsys.readouterr().out)["plans"] == 4        # C0o, C4o, C0h, C4h


def test_overview_needs_render_support(monkeypatch):
    from ayas.twin import pipeline as PL

    monkeypatch.setattr(PL.render, "render_batch", lambda works, cfg=None, device=None, samples=None, res_scale=None: [])
    with pytest.raises(SystemExit, match="overview"):
        PL._render_kwargs(None, None, None, True)
    assert PL._render_kwargs("CPU", 8, 0.5, False) == {"device": "CPU", "samples": 8, "res_scale": 0.5}


# ---- the real capture modules, when another build has written them -------------------------

def test_process_scene_with_the_real_capture_modules(tmp_path, monkeypatch):
    """Contract check against the real lightproc/burst/lighting: the fixture's LED passes are
    scaled to physical radiance (rho * E_led / pi, E_led from lighting.led_irradiance on two
    LED lights), so the real burst simulator's exposure is right. Only the contract is checked
    (arms, gate values, files), not the thresholds."""
    LP = pytest.importorskip("ayas.lightproc")
    B = pytest.importorskip("ayas.twin.burst")
    L = pytest.importorskip("ayas.twin.lighting")
    monkeypatch.setattr(P, "capture_modules", lambda: (LP, B))
    cfg = twin_cfg(variants_per_scene=1)
    cap = LP.load_capture_config(str(ROOT / "configs" / "capture.yaml"))
    plan = make_plan(split="test", condition="C1_office", lux=450.0)
    Rm, t = np.asarray(plan["camera"]["R"]), np.asarray(plan["camera"]["t"])
    C = F.camera_centre(Rm, t)
    power = L.led_power_w(cap["leds"]["white_lumens_per_led"]) * 9
    plan["lights"] = [{"name": f"WL_{g}_0", "group": g, "kind": "spot", "pos_m": (C + Rm.T @ [dx, 0, 0]).tolist(),
                       "dir": Rm[2].tolist(), "radius_m": 0.002, "spot_deg": 150, "blend": 0.6, "power_w": power,
                       "color": cap["leds"]["colour"]} for g, dx in (("L", -0.04), ("R", 0.04))]
    e_l, e_r = L.led_irradiance(plan, ("L",)), L.led_irradiance(plan, ("R",))
    assert e_l > 0 and e_r > 0
    work = tmp_path / "work" / plan["scene_id"]
    work.mkdir(parents=True)
    (work / "plan.json").write_text(json.dumps(plan))
    g = render_groups(plan, plate_albedo(), amb=0.0)
    alb = g["L"] / (0.6 + 0.4 * np.linspace(1.0, 0.0, g["L"].shape[1], dtype=np.float32)[None, :, None])
    amb = alb * np.float32(L.lux_to_w(450.0) / math.pi)
    write_exr(work / "render.exr", {"L": alb * np.float32(e_l / math.pi), "R": alb * np.float32(e_r / math.pi), "AMB": amb})
    out = tmp_path / "out"
    sc = P.process_scene(work, out, cfg, debug=False, save_bursts=True)
    v = sc["variants"][0]
    assert v["gate"] in ("PASS", "PASS_FLAGGED", "RETAKE") and set(v["arms"]) == {"burst", "naive"}
    for arm in ("burst", "naive"):
        assert (out / "grids" / arm / "test" / f"{plan['scene_id']}_v0.png").exists()
    burst, extra = LP.load_burst(out / "bursts" / f"{plan['scene_id']}_v0.npz")   # lightproc reads post's npz
    assert set(burst.frames) == {"B1", "L", "A", "R", "B2"} and burst.meta["twin"]["condition"] == "C1_office"
    assert burst.frames["B1"].dtype == np.float32 and extra["corners_px"].shape == (4, 2)
    again = LP.process_burst(burst, P.plate_mask_image(P.grid_geometry(plan, cfg["grid"]), burst.frames["B1"].shape),
                             cap)
    assert again.gate.verdict == v["gate"]                       # replaying the file gives the same verdict


def test_lux_ranges_match_the_lighting_module():
    L = pytest.importorskip("ayas.twin.lighting")
    post = twin_cfg()["post"]
    assert set(L.CONDITIONS) == set(P.LUX_RANGES) == set(post["amb_sweep"]["lux"])
    spec = getattr(L, "SPEC", None)
    if spec is None:
        pytest.skip("twin.lighting has no SPEC table")
    for c in L.CONDITIONS:
        assert tuple(map(float, spec[c]["lux"])) == P.LUX_RANGES[c] == P.lux_range(c, post)
