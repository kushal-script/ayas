"""Blender/Cycles twin renderer on hand-made scenes (skipped when Blender is missing).

The first Metal render on a machine compiles the Cycles kernels (about 200 s, once).
The lighting tests render one plate under conditions from ayas.twin.lighting with the
real rig LEDs (power from capture.yaml lumens): DESIGN_STEEL C, D.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ayas.twin import frames as F
from ayas.twin import lighting as Lt
from ayas.twin import render
from ayas.twin import rig as RIG

pytest.importorskip("OpenEXR")
pytestmark = pytest.mark.skipif(render.find_blender() is None, reason="Blender not installed")

RES_SCALE = 0.15
SQUARE = (70.0, 20.0, 80.0, 30.0)           # dark albedo square: u0, v0, u1, v1 (mm)
OPEN_HOLES = {"H2": (40.0, 45.0, 1.0, 0.0), "H3": (62.0, 60.0, 1.35, 30.0)}   # u, v, ovality, angle
DEFAULT_LEDS = (("WL_L_0", "L", -60, -20), ("WL_L_1", "L", -60, 20), ("WL_R_0", "R", 60, -20),
                ("WL_R_1", "R", 60, 20))   # name, group, x_cam, y_cam (mm), 5 mm in front of the lens
BOX_CFG = {"camera": {"pixel_um": 1.55}, "rig": {"device_box_mm": {"x": [-134, 134], "y": [-54, 80], "z": [-74, 0]}}}

# Second scene: 100 x 60 plate, rolled 100 deg and tilted, target off the plate centre. The ROI is
# taller than wide (Blender fits the sensor to the height) and off the optical axis (non-zero shift).
OFF_SIZE = (100.0, 60.0)
OFF_POSE = ((35.0, 22.0), 205.0, 7.0, -9.0, 100.0)  # target, distance, tilt x, tilt y, roll
OFF_SQUARE = (58.0, 14.0, 70.0, 24.0)
OFF_UV_SQUARE = (12.0, 34.0, 24.0, 46.0)            # uvemit.png (plan textures.uv_emit is null)
OFF_HOLES = {"H1": (22.0, 22.0, 1.0, 0.0), "H2": (82.0, 40.0, 1.35, 120.0)}


def _rgb16(rgb) -> np.ndarray:
    return np.round(np.asarray(rgb, float) * 65535).astype(np.uint16)


def _hole(hid, u, v, ov=1.0, ang=0.0, filled=False, sheet=None) -> dict:
    return {"id": hid, "u": u, "v": v, "d_mm": 4.1, "ovality": ov, "angle_deg": ang, "sheet": sheet, "filled": filled}


def write_fixture(work: Path, res_mm: float = 0.2, samples: int = 96, *, size=(100.0, 100.0),
                  pose=((50.0, 50.0), 200.0, 0.0, 0.0, 0.0), square=SQUARE, holes=None, rivets=None,
                  material=None, rough8: int = 89, leds=DEFAULT_LEDS, uv_square=None, metal8: int = 255,
                  albedo8: int = 200) -> dict:
    """A plate seen from the pose (default: 100 x 100 x 3 straight down from 200 mm), flat maps.

    Defaults: 1 rivet and 2 open holes (one oval), one dark albedo square, ROI = projected
    plate bbox + 48 px, LEDs placed in the camera frame and aimed at the target.
    """
    work.mkdir(parents=True, exist_ok=True)
    W, H = size
    target = pose[0]
    K = F.nominal_K()
    R, t = F.camera_pose(*pose)
    px = F.project(K, R, t, F.plate_to_world([(0, 0), (W, 0), (W, H), (0, H)]))
    x0, y0 = (max(0, int(math.floor(v)) - 48) for v in px.min(axis=0))
    x1, y1 = (min(lim, int(math.ceil(v)) + 48) for v, lim in zip(px.max(axis=0), (4056, 3040)))
    C = F.camera_centre(R, t)

    def led(name, group, x_mm, y_mm):
        pos = C + R.T @ (np.array([x_mm, y_mm, 5.0]) / 1000.0)
        d = F.plate_to_world([target])[0] - pos
        return {"name": name, "group": group, "kind": "spot", "pos_m": pos.tolist(),
                "dir": (d / np.linalg.norm(d)).tolist(), "radius_m": 0.002, "spot_deg": 150, "blend": 0.6,
                "power_w": 0.35, "color": [1.0, 0.96, 0.9]}

    if holes is None:
        holes = [_hole("H1", 30.0, 72.0, filled=True)]
        holes += [_hole(k, u, v, ov, a) for k, (u, v, ov, a) in OPEN_HOLES.items()]
    if rivets is None:
        rivets = [{"id": "R1", "hole": "H1", "head_d_mm": 8.0, "head_h_mm": 1.3, "mandrel_d_mm": 2.2,
                   "tilt_deg": 0.0, "material": {"base_color": [0.86, 0.87, 0.88], "roughness": 0.35}}]
    mat = {"base_color": [0.91, 0.92, 0.92], "roughness": 0.35, "anisotropy": 0.3, "aniso_rotation": 0.0,
           "grain_amp_um": 0.0, "grain_dir_deg": 0.0, **(material or {})}
    plan = {
        "schema": "ayas.twin.scene/1", "scene_id": work.name, "seed": 7, "split": "train",
        "mode": "surface", "source": "fixture",
        "plate": {"size_mm": [W, H], "thickness_mm": 3.0, "edge_bevel_mm": 0.3, "finish": "brushed",
                  "material": mat},
        "sheets": [], "holes": holes, "rivets": rivets,
        "defects": [], "distractors": [], "penetrant": None,
        "camera": {"K": K.tolist(), "full_res": [4056, 3040], "R": R.tolist(), "t": t.tolist(),
                   "roi": [x0, y0, x1 - x0, y1 - y0], "fstop": 1.79, "focus_m": pose[1] / 1000.0,
                   "pose": {"target_mm": list(target), "distance_mm": pose[1], "tilt_deg": list(pose[2:4]),
                            "roll_deg": pose[4]}},
        "lights": [led(*x) for x in leds],
        "ambient": {"world_color": [0.8, 0.8, 0.82], "world_strength": 0.25,
                    "ceiling": [{"pos_m": [0.3, -0.2, 2.0], "size_m": [1.2, 0.05], "rot_deg": 30,
                                 "power_w": 40, "color": [1.0, 0.98, 0.95]}]},
        "extras": {"background": {"kind": "noise", "color": [0.2, 0.2, 0.22], "roughness": 0.8, "scale": 40},
                   "gloves": [{"u": 4.0, "v": H - 4.0, "angle_deg": 160, "length_mm": 60, "radius_mm": 8}],
                   "device_box": True},
        "render": {"samples": samples, "noise_threshold": 0.02, "max_bounces": 6, "glossy_bounces": 4,
                   "texture_res_mm": res_mm, "clean_pass": False},
        "textures": {"normal": "normal.png", "albedo": "albedo.png", "roughness": "rough.png",
                     "metallic": "metal.png", "uv_emit": None},
    }
    (work / "plan.json").write_text(json.dumps(plan, indent=1))

    nu, nv = round(W / res_mm), round(H / res_mm)

    def cells(sq):
        u0, v0, u1, v1 = sq
        return slice(round(v0 / res_mm), round(v1 / res_mm)), slice(round(u0 / res_mm), round(u1 / res_mm))

    normal = np.empty((nv, nu, 3), np.uint16)
    normal[:] = _rgb16([0.5, 0.5, 1.0])                                  # n = (0, 0, 1)
    albedo = np.full((nv, nu, 3), albedo8, np.uint8)
    if square is not None:
        albedo[cells(square)] = 40
    cv2.imwrite(str(work / "normal.png"), normal[..., ::-1])            # cv2 wants BGR
    cv2.imwrite(str(work / "albedo.png"), albedo[..., ::-1])
    cv2.imwrite(str(work / "rough.png"), np.full((nv, nu), rough8, np.uint8))
    cv2.imwrite(str(work / "metal.png"), np.full((nv, nu), metal8, np.uint8))
    if uv_square is not None:
        uv = np.zeros((nv, nu), np.uint8)
        uv[cells(uv_square)] = 255
        cv2.imwrite(str(work / "uvemit.png"), uv)
    return plan


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    work = tmp_path_factory.mktemp("twin_render") / "fixture0"
    plan = write_fixture(work)
    cam = render.prepare(work, plan, BOX_CFG)
    meta = render.render_batch([work], BOX_CFG, device="METAL", samples=32, res_scale=RES_SCALE, timeout_s=1500)[0]
    return work, plan, cam, meta


@pytest.fixture(scope="module")
def offcentre(tmp_path_factory):
    work = tmp_path_factory.mktemp("twin_render_off") / "fixture_off"
    holes = [_hole(k, u, v, ov, a) for k, (u, v, ov, a) in OFF_HOLES.items()]
    plan = write_fixture(work, size=OFF_SIZE, pose=OFF_POSE, square=OFF_SQUARE, holes=holes, rivets=[],
                         uv_square=OFF_UV_SQUARE)
    cam = render.prepare(work, plan, BOX_CFG)
    meta = render.render_batch([work], BOX_CFG, device="METAL", samples=32, res_scale=RES_SCALE, timeout_s=1500)[0]
    return work, plan, cam, meta


def _luma(img):
    return img @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _render_K(plan, cam):
    w, h = cam["resolution"]
    return render.scale_K(cam["K_roi"], w, h, RES_SCALE)


def _square_centroid(Y, Hs, square):
    """Image centroid found and expected, pixel count found, expected area and contrast (<= 1) of a square.

    The square (u0, v0, u1, v1) mm differs from the plate around it (darker or brighter). Pixels
    are split at the midpoint between the projected square's interior and a ring around it, inside
    a window 8 px larger than the projection, so a misplacement of up to ~8 px shows up.
    """
    Y = cv2.medianBlur(np.asarray(Y, np.float32), 3)        # few samples: drop the sparkle
    u0, v0, u1, v1 = square
    corners = F.apply_h(Hs, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    g = np.stack(np.meshgrid(np.linspace(u0, u1, 41), np.linspace(v0, v1, 41)), -1).reshape(-1, 2)
    centre = F.apply_h(Hs, g).mean(axis=0)
    mask = np.zeros(Y.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(corners * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    grow = lambda m, r: cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))  # noqa: E731
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool)
    ring = (grow(mask, 8) & ~grow(mask, 3)).astype(bool)
    win = grow(mask, 8).astype(bool)
    a, b = np.median(Y[inner]), np.median(Y[ring])
    rows, cols = np.nonzero(win & ((Y < 0.5 * (a + b)) if a < b else (Y > 0.5 * (a + b))))
    found = np.array([cols.mean(), rows.mean()])
    area = 0.5 * abs(np.cross(np.r_[corners[2] - corners[0], 0], np.r_[corners[3] - corners[1], 0])[2])
    return found, centre, len(rows), area, min(a, b) / max(a, b, 1e-12)


def test_prepare_camera_is_physical(rendered):
    _, plan, cam, _ = rendered
    assert cam["lens_mm"] == pytest.approx(4.74, rel=1e-3)
    assert cam["sensor_width_mm"] == pytest.approx(max(cam["resolution"]) * 1.55e-3)
    assert np.allclose(np.asarray(cam["matrix_world"])[:3, 3], [0.05, -0.05, 0.2])
    assert cam["device_box_m"]["z"][1] < 0.005                # front face kept behind the LEDs


def test_exr_has_light_groups(rendered):
    work, plan, cam, meta = rendered
    assert not meta.get("error"), meta
    assert (work / "render.exr").is_file()
    names = render.pass_names(meta["exr_channels"])
    assert {"Combined", "L", "R", "AMB"} <= names
    w, h = render.scaled_size(*cam["resolution"], RES_SCALE)
    assert meta["resolution"] == [w, h]
    assert np.allclose(meta["K_render"], _render_K(plan, cam), atol=1e-6)
    p = render.read_passes(work / "render.exr")
    assert p["L"].shape == (h, w, 3) and np.isfinite(p["L"]).all()
    assert _luma(p["AMB"]).mean() > 1e-4                     # world + ceiling really land in AMB


def test_left_and_right_groups_light_their_own_side(rendered):
    work, *_ = rendered
    p = render.read_passes(work / "render.exr")
    L, R = _luma(p["L"]), _luma(p["R"])
    half = L.shape[1] // 2
    assert L[:, :half].mean() > 1.3 * L[:, half:].mean()
    assert R[:, half:].mean() > 1.3 * R[:, :half].mean()


def test_dark_albedo_square_lands_where_frames_projects_it(rendered):
    work, plan, cam, _ = rendered
    p = render.read_passes(work / "render.exr")
    Y = _luma(p["L"] + p["R"])
    Hs = F.homography_plate_to_image(_render_K(plan, cam), plan["camera"]["R"], plan["camera"]["t"])
    found, centre, n, area, contrast = _square_centroid(Y, Hs, SQUARE)
    assert np.abs(found - centre).max() < 1.0, (found, centre)
    assert 0.75 * area < n < 1.3 * area
    assert contrast < 0.3


def _open_holes_dark(Y, Hs, mm_px, holes):
    rr, cc = np.mgrid[: Y.shape[0], : Y.shape[1]]
    for u, v, _, _ in holes:
        x, y = F.apply_h(Hs, [(u, v)])[0]
        dist = np.hypot(cc - x, rr - y)
        r = 2.05 * mm_px
        inside = Y[dist < 0.6 * r].mean()
        ring = Y[(dist > 1.6 * r) & (dist < 2.4 * r)].mean()
        assert inside < 0.5 * ring, (u, v, inside, ring)


def test_open_holes_are_darker_than_the_plate(rendered):
    work, plan, cam, _ = rendered
    p = render.read_passes(work / "render.exr")
    Ks = _render_K(plan, cam)
    Hs = F.homography_plate_to_image(Ks, plan["camera"]["R"], plan["camera"]["t"])
    _open_holes_dark(_luma(p["L"] + p["R"]), Hs, Ks[0, 0] / 200.0, OPEN_HOLES.values())


def test_rolled_tilted_offcentre_view_lands_where_frames_projects_it(offcentre):
    """Catches shift sign errors, W/H swaps in the texture mapping and vertical sensor fit."""
    work, plan, cam, meta = offcentre
    assert not meta.get("error"), meta
    w, h = cam["resolution"]
    assert w < h and abs(cam["shift_x"]) > 0.05 and abs(cam["shift_y"]) > 0.05   # the case being tested
    p = render.read_passes(work / "render.exr")
    Ks = _render_K(plan, cam)
    Hs = F.homography_plate_to_image(Ks, plan["camera"]["R"], plan["camera"]["t"])
    found, centre, n, area, contrast = _square_centroid(_luma(p["L"] + p["R"]), Hs, OFF_SQUARE)
    assert np.abs(found - centre).max() < 1.0, (found, centre)
    assert 0.75 * area < n < 1.3 * area and contrast < 0.3
    # uvemit.png is used although plan textures.uv_emit is null; it lands in Combined_UV only
    uv = _luma(p["UV"])
    found, centre, n, area, _ = _square_centroid(uv, Hs, OFF_UV_SQUARE)
    assert np.abs(found - centre).max() < 1.0, (found, centre)
    assert 0.75 * area < n < 1.3 * area
    assert uv.max() > 0.3 and np.median(uv) < 1e-3
    # (holes are not checked photometrically here: seen obliquely, a lit hole wall can outshine
    # the plate; their world placement is checked in test_scene_geometry_follows_the_plan)


def _highlight_axes(Y: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """(angle of the major axis from image +x in degrees, major/minor std ratio) of the highlight."""
    Z = np.where(mask, Y, 0.0)
    Z = np.where(Z > 0.3 * np.percentile(Z[mask], 99.5), Z, 0.0)
    rr, cc = np.nonzero(Z)
    wgt = Z[rr, cc]
    m = np.array([np.average(cc, weights=wgt), np.average(rr, weights=wgt)])
    d = np.stack([cc - m[0], rr - m[1]], 1)
    cov = (d * wgt[:, None]).T @ d / wgt.sum()
    ev, evec = np.linalg.eigh(cov)
    major = evec[:, 1]
    return math.degrees(math.atan2(major[1], major[0])) % 180.0, math.sqrt(ev[1] / ev[0])


def test_anisotropic_highlight_runs_across_the_grain(tmp_path):
    """Brushed metal: the highlight stretches across the grain lines, whichever way the sampler encodes them.

    Straight down at roll 0, image x = +u. Grain along u (grain_dir 0, aniso_rotation 0) must give a
    highlight along image y; grain along v as the sampler writes it (grain_dir 90 with
    aniso_rotation 0.25, the same direction in Blender units) must give one along image x.
    """
    works = []
    for name, grain, rot in (("grain_u", 0.0, 0.0), ("grain_v", 90.0, 0.25)):
        plan = write_fixture(tmp_path / name, square=None, holes=[], rivets=[], rough8=46,
                             material={"anisotropy": 0.9, "grain_dir_deg": grain, "aniso_rotation": rot},
                             leds=(("T_L", "L", 0, -15),))
        render.prepare(tmp_path / name, plan, None)
        works.append(tmp_path / name)
    metas = render.render_batch(works, None, device="METAL", samples=16, res_scale=RES_SCALE, timeout_s=1500)
    assert not any(m.get("error") for m in metas), metas
    for work, want in zip(works, (90.0, 0.0)):
        plan = json.loads((work / "plan.json").read_text())
        cam = json.loads((work / "blender_cam.json").read_text())
        Y = _luma(render.read_passes(work / "render.exr")["L"])
        Hs = F.homography_plate_to_image(_render_K(plan, cam), plan["camera"]["R"], plan["camera"]["t"])
        mask = np.zeros(Y.shape, np.uint8)
        cv2.fillPoly(mask, [np.round(F.apply_h(Hs, [(3, 3), (97, 3), (97, 97), (3, 97)])).astype(np.int32)], 1)
        ang, ratio = _highlight_axes(Y, mask.astype(bool))
        off = min(abs(ang - want), 180.0 - abs(ang - want))
        assert off < 15.0 and ratio > 1.8, (work.name, ang, ratio)


def test_clean_pass_renders_the_clean_maps(tmp_path):
    work = tmp_path / "clean"
    plan = write_fixture(work, res_mm=0.5, holes=[], rivets=[])
    plan["render"]["clean_pass"] = True
    cv2.imwrite(str(work / "albedo_clean.png"), cv2.imread(str(work / "albedo.png"))[:] * 0 + 200)
    for key in ("normal", "rough", "metal"):                 # the same maps without defects
        (work / f"{key}_clean.png").write_bytes((work / f"{key}.png").read_bytes())
    cam = render.prepare(work, plan, BOX_CFG)
    meta = render.render_batch([work], BOX_CFG, device="METAL", samples=32, res_scale=RES_SCALE, timeout_s=900)[0]
    assert not meta.get("error"), meta
    assert {"render.exr", "render_clean.exr"} <= set(meta["files"]) and not meta["warnings"]
    Hs = F.homography_plate_to_image(_render_K(plan, cam), plan["camera"]["R"], plan["camera"]["t"])
    contrast = {}
    for name in ("render.exr", "render_clean.exr"):
        p = render.read_passes(work / name)
        contrast[name] = _square_centroid(_luma(p["L"] + p["R"]), Hs, SQUARE)[4]
    assert contrast["render.exr"] < 0.3 and contrast["render_clean.exr"] > 0.8, contrast


def test_preview_png(rendered):
    work, *_ = rendered
    out = render.preview(work)
    img = cv2.imread(str(out))
    assert img is not None and img.shape[2] == 3 and img.max() > 100


def test_bad_scene_does_not_kill_the_batch(rendered, tmp_path):
    work, plan, cam, _ = rendered
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "plan.json").write_text("{not json")
    good = tmp_path / "good"
    good.mkdir()
    for name in ("plan.json", "blender_cam.json", "normal.png", "albedo.png", "rough.png", "metal.png"):
        (good / name).write_bytes((work / name).read_bytes())
    # an empty scene: no fasteners, lights, gloves, ceiling or device box still gives every pass
    empty = tmp_path / "empty"
    ep = write_fixture(empty, res_mm=1.0, square=None, holes=[], rivets=[], leds=())
    ep["extras"].update(gloves=[], device_box=False)
    ep["ambient"]["ceiling"] = []
    render.prepare(empty, ep, None)
    metas = render.render_batch([bad, good, empty], None, device="METAL", samples=2, res_scale=0.05, timeout_s=900)
    assert metas[0]["error"] and not (bad / "render.exr").exists()
    assert not metas[1].get("error"), metas[1]
    assert {"L", "R", "AMB"} <= render.pass_names(metas[1]["exr_channels"])
    assert not metas[2].get("error"), metas[2]
    p = render.read_passes(empty / "render.exr")
    assert {"L", "R", "AMB"} <= set(p) and float(np.abs(p["L"]).max()) == 0.0 and _luma(p["AMB"]).mean() > 0


def test_batch_problem_only_fails_unfinished_scenes(rendered, tmp_path):
    """A crash or timeout after a scene finished must not throw that scene away."""
    work, *_ = rendered
    done = tmp_path / "done"
    done.mkdir()
    for name in ("render.exr", "render_meta.json"):
        (done / name).write_bytes((work / name).read_bytes())
    meta = json.loads((done / "render_meta.json").read_text())
    meta["error"] = None
    (done / "render_meta.json").write_text(json.dumps(meta))
    todo = tmp_path / "todo"
    todo.mkdir()
    a = render._finish(done, "log", "Blender timed out after 10 s")
    b = render._finish(todo, "log", "Blender timed out after 10 s")
    assert not a["error"] and any("timed out" in w for w in a["warnings"])
    assert b["error"] == "Blender timed out after 10 s"
    assert render._finish(tmp_path / "gone", "log", None)["error"]     # reported, never raised


def test_split_render_without_its_ambient_half_is_an_error(rendered, tmp_path):
    """Regression: a split render whose render_amb.exr is missing (render_scene.py run on its own, or
    the file lost) used to pass as a dark room (empty AMB pass); _finish now says so."""
    work, *_ = rendered
    meta = json.loads((work / "render_meta.json").read_text())
    assert meta.get("split_ambient") and meta.get("ambient_merged_into") == ["render.exr"]
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "render.exr").write_bytes((work / "render.exr").read_bytes())
    fresh = {k: v for k, v in meta.items() if k not in ("ambient_merged", "ambient_merged_into", "exr_channels")}
    (raw / "render_meta.json").write_text(json.dumps(fresh))
    out = render._finish(raw, "log", None)
    assert out["error"] and "render_amb.exr missing" in out["error"]
    # with its ambient half it merges, and a second _finish (re-run) keeps it valid
    (raw / "render_meta.json").write_text(json.dumps(fresh))
    (raw / "render_amb.exr").write_bytes((work / "render.exr").read_bytes())
    out = render._finish(raw, "log", None)
    assert not out["error"] and out["ambient_merged_into"] == ["render.exr"] and not (raw / "render_amb.exr").exists()
    assert not render._finish(raw, "log", None)["error"]


def test_prepare_rewrites_a_stale_plan(rendered, tmp_path):
    work, plan, *_ = rendered
    w = tmp_path / "stale"
    w.mkdir()
    (w / "plan.json").write_text(json.dumps({**plan, "seed": 999}))
    render.prepare(w, plan, BOX_CFG)
    assert json.loads((w / "plan.json").read_text()) == plan


DUMP = r"""
import json, sys
from pathlib import Path
import bpy
from mathutils import Vector
args = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, args[0])
import render_scene as RS
work = Path(args[1])
plan = json.loads((work / "plan.json").read_text())
cam = json.loads((work / "blender_cam.json").read_text())
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.engine = "CYCLES"
class A:
    res_scale = 1.0
RS.setup_render(plan, 1)
RS.build_scene(plan, cam, work, A, {"warnings": [], "errors": [], "files": []})
bpy.context.view_layer.update()
out = {}
for o in bpy.context.scene.objects:
    d = {"type": o.type}
    if o.type == "MESH":
        vs = [list(o.matrix_world @ v.co) for v in o.data.vertices]
        d["verts"] = vs
        d["mats"] = [m.name for m in o.data.materials]
        d["colors"] = []
        for m in o.data.materials:
            b = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            d["colors"].append(list(b.inputs["Base Color"].default_value)[:3] if b else None)
    if o.type == "LIGHT":
        d["minus_z"] = list(o.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0)))
        d["x_axis"] = list(o.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0)))
        d["loc"] = list(o.matrix_world.translation)
        d["group"] = o.lightgroup
        d["light_type"] = o.data.type
        d["energy"] = o.data.energy
        if o.data.type == "AREA":
            d["size"] = [o.data.size, o.data.size_y]
            d["shape"] = o.data.shape
        if o.data.type == "SUN":
            d["angle"] = o.data.angle
    out[o.name] = d
Path(args[2]).write_text(json.dumps(out))
"""


def _dump_scene(work: Path, tmp: Path) -> dict:
    script, out = tmp / "dump.py", tmp / f"{work.name}_dump.json"
    script.write_text(DUMP)
    r = subprocess.run([str(render.find_blender()), "-b", "--factory-startup", "--python", str(script), "--",
                        str(render.SCRIPT.parent), str(work), str(out)], capture_output=True, text=True, timeout=600)
    assert out.is_file(), r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads(out.read_text())


def test_scene_geometry_follows_the_plan(tmp_path):
    """No render: rivet tilt direction and seat, sheet, glove, device box, lights and oval hole in world metres."""
    work = tmp_path / "geo"
    plan = write_fixture(work, res_mm=1.0)
    plan["sheets"] = [{"id": "S1", "rect_mm": [0.0, 0.0, 100.0, 30.0], "thickness_mm": 1.0,
                       "material": {"base_color": [0.9, 0.9, 0.9], "roughness": 0.3, "anisotropy": 0.5,
                                    "aniso_rotation": 0.25, "grain_dir_deg": 90.0}}]
    plan["holes"].append(_hole("H4", 50.0, 15.0, filled=True, sheet="S1"))
    plan["rivets"].append({"id": "R2", "hole": "H4", "head_d_mm": 8.0, "head_h_mm": 1.3, "mandrel_d_mm": 2.2,
                           "tilt_deg": 6.0, "tilt_dir_deg": 90.0})
    plan["extras"]["gloves"] = [{"u": -5.0, "v": 60.0, "angle_deg": 180.0, "length_mm": 60.0, "radius_mm": 8.0,
                                 "color": [0.05, 0.12, 0.45]}]
    cam = render.prepare(work, plan, BOX_CFG)
    g = _dump_scene(work, tmp_path)
    V = {k: np.asarray(v["verts"]) for k, v in g.items() if "verts" in v}

    # rivet on the sheet leans down towards +v (world -Y) and its low rim sits on the sheet top
    r2, X, Y = V["rivet_R2"], 0.050, -0.015
    low = r2[np.argmin(r2[:, 2])]
    assert low[1] < Y - 0.003 and abs(low[0] - X) < 1e-4
    assert 0.001 - 2e-5 < r2[:, 2].min() < 0.001 + 1e-6
    assert r2[:, 2].max() > 0.001 + 0.0012
    r1 = V["rivet_R1"]
    assert abs(r1[:, 2].min()) < 2e-5                          # untilted rivet on the plate face
    # sheet box and plate box
    s1 = V["sheet_S1"]
    assert np.allclose(s1.min(0), [0, -0.030, 0], atol=1e-6) and np.allclose(s1.max(0), [0.1, 0, 0.001], atol=1e-6)
    pl = V["plate"]
    assert np.allclose(pl.min(0), [0, -0.1, -0.003], atol=1e-6) and np.allclose(pl.max(0), [0.1, 0, 0], atol=1e-6)
    # oval open hole H3 (u 62, v 60, major 4.1 x 1.35 along 30 deg from +u towards +v)
    front = pl[np.abs(pl[:, 2]) < 1e-9]
    c = np.array([0.062, -0.060])
    near = front[np.hypot(*(front[:, :2] - c).T) < 0.004]
    far = near[np.argmax(np.hypot(*(near[:, :2] - c).T))]
    assert np.hypot(*(far[:2] - c)) == pytest.approx(0.0041 * 1.35 / 2, rel=1e-3)
    ang = math.degrees(math.atan2(-(far[1] - c[1]), far[0] - c[0])) % 180.0    # plate frame: v = -Y
    assert ang == pytest.approx(30.0, abs=1.0)
    # glove: fingertip centre 5 mm left of the plate, finger along -u, resting on the plate face
    gl = V["glove_0"]
    assert gl[:, 0].max() == pytest.approx(0.003, abs=2e-4)    # tip reaches 3 mm onto the plate
    assert gl[:, 0].min() == pytest.approx(-0.057, abs=2e-4)
    assert gl[:, 2].min() == pytest.approx(0.0, abs=1e-4)
    assert np.allclose(g["glove_0"]["colors"][0], [0.05, 0.12, 0.45], atol=1e-4)
    # device box: rig box (camera frame, mm) -> world through cv2world
    R, t = np.asarray(plan["camera"]["R"]), np.asarray(plan["camera"]["t"])
    box = cam["device_box_m"]
    corners = np.array([[x, y, z] for x in box["x"] for y in box["y"] for z in box["z"]])
    want = corners @ R + F.camera_centre(R, t)                  # R.T @ p + C, row-wise
    assert np.allclose(np.sort(V["device_box"], 0), np.sort(want, 0), atol=1e-6)
    # lights point their -Z along the plan direction and keep their group
    for L in plan["lights"]:
        d = g[L["name"]]
        assert np.allclose(d["minus_z"], L["dir"], atol=1e-5) and np.allclose(d["loc"], L["pos_m"], atol=1e-7)
        assert d["group"] == L["group"]


def test_penetrant_scene_coats_rivets_and_sheets(tmp_path):
    work = tmp_path / "pen"
    plan = write_fixture(work, res_mm=1.0)
    plan["mode"] = "penetrant"
    plan["penetrant"] = {"developer": {"albedo": 0.88, "roughness": 0.9}, "indications": [], "edge_bleed": 0.0}
    plan["sheets"] = [{"id": "S1", "rect_mm": [0.0, 0.0, 100.0, 30.0], "thickness_mm": 1.0,
                       "material": {"base_color": [0.9, 0.9, 0.9], "roughness": 0.3}}]
    render.prepare(work, plan, BOX_CFG)
    g = _dump_scene(work, tmp_path)
    for name in ("rivet_R1", "shank_R1", "sheet_S1"):
        assert g[name]["mats"][0] == "developer", (name, g[name]["mats"])
        assert np.allclose(g[name]["colors"][0], [0.88] * 3)
    assert g["background"]["mats"] == ["background"]


def test_scale_K_matches_full_resolution():
    K = F.roi_K(F.nominal_K(), 1200, 700)
    assert np.allclose(render.scale_K(K, 1600, 1500, 1.0), K)
    Ks = render.scale_K(K, 1600, 1500, 0.5)
    # a pixel centre at (x, y) lands at ((x + 0.5) / 2 - 0.5, (y + 0.5) / 2 - 0.5)
    x = np.array([K[0, 2] + 100.0, K[1, 2] - 40.0])
    xs = (x - K[:2, 2]) * 0.5 + Ks[:2, 2]
    assert np.allclose(xs, (x + 0.5) / 2 - 0.5)


# ----------------------------------------------------------------------------- lighting conditions (DESIGN_STEEL C, D)
ROOT = Path(__file__).resolve().parents[1]
TWIN_CFG = yaml.safe_load((ROOT / "configs" / "twin.yaml").read_text())
FIX = {"steel": dict(albedo8=170, rough8=115, metal8=255,      # cold-rolled-like steel: linear 0.40, roughness 0.45
                     material={"base_color": [0.4, 0.4, 0.4], "roughness": 0.45, "anisotropy": 0.0}),
       "matte": dict(albedo8=188, rough8=255, metal8=0,        # matte grey card: linear albedo 0.503
                     material={"base_color": [0.5, 0.5, 0.5], "roughness": 1.0, "metallic": 0.0, "anisotropy": 0.0})}
MATTE_ALBEDO = ((188 / 255 + 0.055) / 1.055) ** 2.4
# name: condition, device mode, ambient seed, plate, hood seed (14: rim 0-1 mm over the plate face, leaking
# under it round the plate; 11: rim on the bench)
COND_SCENES = {"C0": ("C0_dark", "open", 11, "steel", 0), "C3": ("C3_window", "open", 12, "steel", 0),
               "C4": ("C4_side_lamp", "open", 13, "steel", 0), "C6": ("C6_sun", "open", 14, "steel", 0),
               "C6_hood_part": ("C6_sun", "hood", 14, "steel", 14), "C6_hood_bench": ("C6_sun", "hood", 14, "steel", 11),
               "C6_matte": ("C6_sun", "open", 15, "matte", 0)}


def lighting_fixture(work: Path, cond: str, mode: str, seed: int, kind: str, hood_seed: int | None = None,
                     samples: int = 16) -> dict:
    """A flat 100 x 100 plate, 200 mm straight down, the real rig LEDs, ambient and hood from ayas.twin.lighting."""
    plan = write_fixture(work, res_mm=0.5, samples=samples, square=None, holes=[], rivets=[], **FIX[kind])
    plan["schema"] = "ayas.twin.scene/2"
    plan["extras"]["gloves"] = []
    rig = RIG.load_rig(TWIN_CFG, ROOT)
    plan["lights"] = RIG.lights_world(rig, plan["camera"]["R"], plan["camera"]["t"])
    plan["ambient"] = Lt.ambient_for(np.random.default_rng(seed), plan, TWIN_CFG, cond)
    plan["extras"]["device"] = {"mode": mode}
    plan["extras"]["hood"] = Lt.hood_for(np.random.default_rng(seed if hood_seed is None else hood_seed), plan,
                                         TWIN_CFG, mode)
    (work / "plan.json").write_text(json.dumps(plan, indent=1))
    render.prepare(work, plan, TWIN_CFG)
    return plan


@pytest.fixture(scope="module")
def conditions(tmp_path_factory):
    root = tmp_path_factory.mktemp("twin_lighting")
    scenes = {}
    for name, (cond, mode, seed, kind, hood_seed) in COND_SCENES.items():
        scenes[name] = (root / name, lighting_fixture(root / name, cond, mode, seed, kind, hood_seed))
    metas = render.render_batch([w for w, _ in scenes.values()], TWIN_CFG, device="METAL", samples=16,
                                res_scale=RES_SCALE, timeout_s=1500, overview=True, overview_samples=8)
    return {k: (w, plan, m) for (k, (w, plan)), m in zip(scenes.items(), metas)}


def _plate_lumas(work: Path, plan: dict, margin_mm: float = 8.0) -> tuple[dict, np.ndarray, np.ndarray]:
    """Luma of every pass, a plate mask margin_mm inside the edges, and H image<-plate of the render."""
    cam = json.loads((work / "blender_cam.json").read_text())
    p = render.read_passes(work / "render.exr")
    Hs = F.homography_plate_to_image(_render_K(plan, cam), plan["camera"]["R"], plan["camera"]["t"])
    W, H = plan["plate"]["size_mm"]
    m = margin_mm
    mask = np.zeros(p["L"].shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(F.apply_h(Hs, [(m, m), (W - m, m), (W - m, H - m), (m, H - m)])).astype(np.int32)], 1)
    return {k: _luma(v) for k, v in p.items()}, mask.astype(bool), Hs


def test_lighting_scenes_render_and_merge_the_ambient_pass(conditions):
    for name, (work, plan, meta) in conditions.items():
        assert not meta.get("error"), (name, meta)
        assert {"L", "R", "AMB"} <= render.pass_names(meta["exr_channels"])
        assert not list(work.glob("*_amb.exr"))
        assert bool(meta.get("ambient_merged")) == (name != "C0"), (name, meta)
        assert "render.exr" in meta["files"] and "render_amb.exr" not in meta["files"]


def test_ambient_pass_follows_the_condition(conditions):
    amb = {}
    for name, (work, plan, _) in conditions.items():
        P, mk, _ = _plate_lumas(work, plan)
        amb[name] = (float(np.median(P["AMB"][mk])), float(np.median((P["L"] + P["R"])[mk])), P["AMB"])
    assert float(np.abs(amb["C0"][2]).max()) == 0.0                 # dark room: nothing
    c3, c4, c6 = (amb[c][0] for c in ("C3", "C4", "C6"))
    led = amb["C0"][1]
    print(f"\nsteel plate, open: median AMB / median(L+R) = C3 {c3 / led:.3f}, C4 {c4 / led:.3f}, C6 {c6 / led:.3f}")
    assert 0 < c3 < c6 and c4 > 0
    assert c6 > 0.3 * led                                       # direct sun rivals the LEDs on steel


def test_led_passes_do_not_depend_on_the_condition(conditions):
    """LEDs are rendered without the ambient, so L and R are the same render under every open condition."""
    w0, p0, _ = conditions["C0"]
    ref, mk, _ = _plate_lumas(w0, p0)
    lr0 = (ref["L"] + ref["R"])[mk]
    for cond in ("C3", "C4", "C6"):
        w, plan, _ = conditions[cond]
        P, _, _ = _plate_lumas(w, plan)
        for g in ("L", "R"):
            assert P[g][mk].mean() == pytest.approx(ref[g][mk].mean(), rel=0.01), (cond, g)
        lr = (P["L"] + P["R"])[mk]
        assert np.sqrt(np.mean((lr - lr0) ** 2)) / lr0.mean() < 0.03, cond


def test_hood_blocks_the_sun_but_not_the_leds(conditions):
    wo, po, _ = conditions["C6"]
    O, mk, _ = _plate_lumas(wo, po)
    for name, rests_on in (("C6_hood_part", "part"), ("C6_hood_bench", "bench")):
        wh, ph, _ = conditions[name]
        assert ph["extras"]["hood"]["rests_on"] == rests_on and ph["ambient"] == po["ambient"]
        Hd, _, _ = _plate_lumas(wh, ph)
        print(f"\nC6 mean AMB on the plate: open {O['AMB'][mk].mean():.4g}, hood on the {rests_on} "
              f"(gap {ph['extras']['hood']['gap_mm']} mm) {Hd['AMB'][mk].mean():.4g}")
        assert Hd["AMB"][mk].mean() < O["AMB"][mk].mean() / 20
        assert (Hd["L"] + Hd["R"])[mk].mean() == pytest.approx((O["L"] + O["R"])[mk].mean(), rel=0.05)


def test_side_lamp_arm_shadow_is_visible(conditions):
    """Plate points fully in the arm's shadow (every point of the lamp hidden) are much darker in AMB than fully lit ones."""
    work, plan, _ = conditions["C4"]
    amb = plan["ambient"]
    lamp = amb["lights"][0]
    assert len(amb["occluders"]) == 1
    P, _, Hs = _plate_lumas(work, plan)
    Y = cv2.blur(P["AMB"], (3, 3))
    r = lamp["radius_m"]
    disk = [np.asarray(lamp["pos_m"]) + r * np.array([math.cos(a), math.sin(a), 0.0]) * k
            for k in (0.0, 1.0) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    lit, dark = [], []
    for uv in np.stack(np.meshgrid(np.arange(8, 93, 3.0), np.arange(8, 93, 3.0)), -1).reshape(-1, 2):
        p = F.plate_to_world([uv])[0]
        hid = [Lt.shadowed(amb, p, dict(lamp, pos_m=list(q))) for q in disk]
        x, y = np.round(F.apply_h(Hs, [uv])[0]).astype(int)
        (dark if all(hid) else lit if not any(hid) else []).append(Y[y, x])
    assert len(lit) > 20 and len(dark) > 20, (len(lit), len(dark))
    print(f"\nC4 AMB: fully lit median {np.median(lit):.4g}, arm shadow median {np.median(dark):.4g}")
    assert np.median(lit) > 3 * np.median(dark)


def test_matte_card_reads_the_target_lux_and_the_led_irradiance(conditions):
    """End to end photometry: a matte grey plate reads albedo * E / pi for the sun (AMB) and the LEDs (L + R)."""
    work, plan, _ = conditions["C6_matte"]
    assert not plan["ambient"]["device_shadow"]
    P, _, Hs = _plate_lumas(work, plan)
    x, y = np.round(F.apply_h(Hs, [(50.0, 50.0)])[0]).astype(int)
    win = (slice(y - 3, y + 4), slice(x - 3, x + 4))
    amb = float(np.median(P["AMB"][win]))
    want_amb = MATTE_ALBEDO * Lt.lux_to_w(plan["ambient"]["lux"]) / math.pi
    led = float(np.median((P["L"] + P["R"])[win]))
    e_led = Lt.led_irradiance(plan)
    want_led = MATTE_ALBEDO * e_led * Lt.luma(plan["lights"][0]["color"]) / math.pi
    print(f"\nmatte card, C6 {plan['ambient']['lux']:.0f} lux: AMB {amb:.4g} (expected {want_amb:.4g}); "
          f"LEDs {led:.4g} (expected {want_led:.4g}, E_led {e_led:.3f} W/m^2 = {Lt.w_to_lux(e_led):.0f} lux)")
    assert amb == pytest.approx(want_amb, rel=0.15)              # the device hides part of the sky (5-15 % of the lux)
    assert led == pytest.approx(want_led, rel=0.12)


def test_overview_png_renders(conditions):
    work, plan, meta = conditions["C4"]
    assert "overview.png" in meta["files"] and not any("overview" in w for w in meta["warnings"])
    img = cv2.imread(str(work / "overview.png"))
    assert img is not None and img.shape == (720, 1280, 3) and img.mean() > 3
    # the main render was not disturbed by the overview objects
    assert meta["resolution"] == list(render.scaled_size(*json.loads((work / "blender_cam.json").read_text())["resolution"], RES_SCALE))


def test_ambient_scene_matches_lighting_geometry(tmp_path):
    """No render: area/sun/spot lights, the arm box and the hood land where ayas.twin.lighting puts them."""
    work = tmp_path / "amb_geo"
    plan = lighting_fixture(work, "C4_side_lamp", "hood", 21, "steel")
    rng = np.random.default_rng(5)
    extra = Lt.ambient_for(rng, plan, TWIN_CFG, "C3_window")["lights"] + Lt.ambient_for(rng, plan, TWIN_CFG, "C6_sun")["lights"]
    extra += [L for L in Lt.ambient_for(rng, plan, TWIN_CFG, "C5_reflected_lamp")["lights"] if L["name"] == "AMB_glare"]
    plan["ambient"]["lights"] += extra
    (work / "plan.json").write_text(json.dumps(plan))
    g = _dump_scene(work, tmp_path)
    for L in plan["ambient"]["lights"]:
        d = g[L["name"]]
        assert d["group"] == "AMB" and d["light_type"] == L["kind"].upper()
        assert np.allclose(d["minus_z"], np.asarray(L["dir"]) / np.linalg.norm(L["dir"]), atol=1e-5), L["name"]
        if L["kind"] == "area":
            assert np.allclose(d["x_axis"], np.asarray(L["x_dir"]) / np.linalg.norm(L["x_dir"]), atol=1e-5)
            assert d["shape"] == L["shape"] and d["size"][0] == pytest.approx(L["size_m"][0], rel=1e-6)
            assert d["energy"] == pytest.approx(L["power_w"], rel=1e-6)
        elif L["kind"] == "sun":
            assert d["energy"] == pytest.approx(L["strength"], rel=1e-6)
            assert d["angle"] == pytest.approx(math.radians(L["angle_deg"]), rel=1e-6)
        else:
            assert d["energy"] == pytest.approx(L["power_w"], rel=1e-6)
            assert np.allclose(d["loc"], L["pos_m"], atol=1e-7)
    occ = plan["ambient"]["occluders"][0]
    assert np.allclose(np.sort(np.asarray(g["occluder_0"]["verts"]), 0), np.sort(Lt.occluder_corners(occ), 0), atol=1e-6)
    hood = plan["extras"]["hood"]
    hv = np.asarray(g["hood"]["verts"])
    corners = Lt.hood_corners(plan, hood)
    bottom = hv[np.abs(hv[:, 2] - hood["gap_mm"] / 1000) < 1e-7]
    want = np.vstack([corners["inner_bottom"], corners["outer_bottom"]])
    assert len(bottom) == 8 and np.allclose(np.sort(bottom, 0), np.sort(want, 0), atol=1e-6)
    R, t = np.asarray(plan["camera"]["R"]), np.asarray(plan["camera"]["t"])
    top = hv[np.abs((hv @ R.T + t)[:, 2] - 0.017) < 1e-7]
    assert np.allclose(np.sort(top, 0), np.sort(np.vstack([corners["inner_top"], corners["outer_top"]]), 0), atol=1e-6)
    assert (hv @ R.T + t)[:, 2].min() == pytest.approx(-0.003, abs=1e-7)    # lid behind the lens, 2 mm thick
