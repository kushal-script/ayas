"""Lighting conditions, hood and LED calibration (trinetra.twin.lighting, DESIGN_STEEL C, D). No Blender.

The irradiance checks use their own brute-force integration here, independent of the module's
light_irradiance_w, so a wrong solver cannot agree with itself.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from trinetra.twin import frames as F
from trinetra.twin import lighting as Lt
from trinetra.twin import rig as RIG

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "twin.yaml").read_text())
CAPTURE = yaml.safe_load((ROOT / "configs" / "capture.yaml").read_text())
LUMA = np.array([0.2126, 0.7152, 0.0722])
UP = np.array([0.0, 0.0, 1.0])


def _plan(seed: int, size=(100.0, 100.0), t_mm=3.0, straight=False, target=None) -> dict:
    """A plate with an open-mode camera pose like the sampler's (or straight down from 200 mm)."""
    rng = np.random.default_rng(seed)
    W, H = size
    target = target or ((W / 2, H / 2) if straight else (rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H))
    pose = (target, 200.0, 0.0, 0.0, 0.0) if straight else \
        (target, rng.uniform(185, 215), rng.uniform(-12, 12), rng.uniform(-12, 12), rng.uniform(-20, 20) + 90 * int(rng.integers(4)))
    R, t = F.camera_pose(*pose)
    return {"plate": {"size_mm": [W, H], "thickness_mm": t_mm}, "camera": {"R": R.tolist(), "t": t.tolist()},
            "extras": {}}


def _brute_lux(light: dict, p) -> float:
    """Independent illuminance (lux) of one light at plate point p (normal +Z), unoccluded."""
    p = np.asarray(p, float)
    y = float(LUMA @ np.asarray(light["color"], float))
    kind = light["kind"]
    if kind == "sun":
        return 683 * light["strength"] * max(0.0, -float(np.dot(light["dir"], UP))) * y
    if kind in ("point", "spot"):
        v = np.asarray(light["pos_m"]) - p
        d = np.linalg.norm(v)
        mask = 1.0
        if kind == "spot":
            ca = float(np.dot(light["dir"], -v / d)) / np.linalg.norm(light["dir"])
            ch = math.cos(math.radians(light["spot_deg"]) / 2)
            f = np.clip((ca - ch) / ((1 - ch) * light["blend"]), 0, 1)
            mask = f * f * (3 - 2 * f)
        return 683 * light["power_w"] / (4 * math.pi * d * d) * max(0.0, v[2] / d) * mask * y
    # area: 300 x 300 midpoint grid over the emitter, Lambertian radiance P / (pi A)
    z = -np.asarray(light["dir"], float) / np.linalg.norm(light["dir"])
    x = np.asarray(light["x_dir"], float)
    x = x - x.dot(z) * z
    x /= np.linalg.norm(x)
    yv = np.cross(z, x)
    w, h = light["size_m"]
    disk = light["shape"] == "DISK"
    if disk:
        h = w
    g = (np.arange(300) + 0.5) / 300 - 0.5
    a, b = (q.ravel() for q in np.meshgrid(g * w, g * h))
    keep = (a / (w / 2)) ** 2 + (b / (h / 2)) ** 2 <= 1 if disk else np.ones(a.size, bool)
    area = math.pi * w * h / 4 if disk else w * h
    q = np.asarray(light["pos_m"]) + a[keep, None] * x + b[keep, None] * yv
    v = q - p
    r = np.linalg.norm(v, axis=1)
    cs, cl = v[:, 2] / r, -(v @ -z) / r
    ok = (cs > 0) & (cl > 0)
    dA = w * h / 300 ** 2
    return 683 * float(np.sum(light["power_w"] / (math.pi * area) * cs[ok] * cl[ok] * dA / r[ok] ** 2)) * y


def _world_lux(amb: dict) -> float:
    return 683 * math.pi * amb["world_strength"] * float(LUMA @ np.asarray(amb["world_color"], float))


# ----------------------------------------------------------------------------- units and LEDs
def test_units_and_led_power():
    assert Lt.lux_to_w(683.0) == pytest.approx(1.0)
    assert Lt.w_to_lux(Lt.lux_to_w(1234.5)) == pytest.approx(1234.5)
    # Lambertian: on-axis intensity lumens / pi cd; power = 4 pi I / 683
    assert Lt.led_power_w(36.7) == pytest.approx(4 * math.pi * (36.7 / math.pi) / 683)
    assert Lt.led_power_w(36.7) == pytest.approx(0.21493, rel=1e-4)


def test_spot_mask_matches_cycles():
    """Values measured on Cycles 5.2.3 renders (spot 150 deg, blend 0.6): see the module docstring."""
    for a_deg, want in ((0, 1.0), (45, 1.0), (53.4, 0.854), (60.9, 0.519), (67.3, 0.198), (74.4, 0.001), (76, 0.0)):
        assert Lt.spot_mask(math.cos(math.radians(a_deg)), 150, 0.6) == pytest.approx(want, abs=0.006), a_deg
    assert Lt.spot_mask(0.99, 30, 0.0) == 1.0 and Lt.spot_mask(0.9, 30, 0.0) == 0.0


def test_rig_white_power_comes_from_the_capture_lumens():
    rig = RIG.load_rig(CFG, ROOT)
    lm = CAPTURE["leds"]["white_lumens_per_led"]
    white = [led for led in rig.leds if led.kind == "white"]
    uv = [led for led in rig.leds if led.kind == "uv"]
    assert len(white) == 18 and all(led.params["power_w"] == pytest.approx(Lt.led_power_w(lm), rel=1e-5) for led in white)
    assert all(led.params["power_w"] == CFG["rig"]["uv"]["power_w"] for led in uv)       # UV unchanged
    # no capture config: the twin.yaml value is the fallback
    cfg = copy.deepcopy(CFG)
    cfg["capture_config"] = "configs/does_not_exist.yaml"
    assert RIG.load_rig(cfg, ROOT).leds[0].params["power_w"] == CFG["rig"]["white"]["power_w"]


def test_led_irradiance_is_a_few_klux_at_200_mm():
    rig = RIG.load_rig(CFG, ROOT)
    plan = _plan(0, straight=True)
    plan["lights"] = RIG.lights_world(rig, plan["camera"]["R"], plan["camera"]["t"])
    e = Lt.led_irradiance(plan)
    lux = Lt.w_to_lux(e)
    print(f"\nLED irradiance at the axis point, 200 mm straight down: {e:.3f} W/m^2 = {lux:.0f} lux (white-equivalent)")
    assert 3000 < lux < 8000                                  # research: 2.4-4.4 klux for 360-660 lm in total
    left, right = Lt.led_irradiance(plan, ("L",)), Lt.led_irradiance(plan, ("R",))
    assert left == pytest.approx(right, rel=0.02) and left + right == pytest.approx(e)
    assert Lt.led_irradiance(plan, ("UV",)) > 0
    # independent check: the same sum written out
    p = Lt.axis_point(plan)
    assert np.allclose(p, F.plate_to_world([[50.0, 50.0]])[0], atol=1e-9)
    brute = sum(_brute_lux(dict(L, color=[1, 1, 1]), p) for L in plan["lights"] if L["group"] in ("L", "R")) / 683 / LUMA.sum()
    assert e == pytest.approx(brute, rel=1e-6)


# ----------------------------------------------------------------------------- conditions
@pytest.mark.parametrize("cond", Lt.CONDITIONS)
def test_lux_round_trip_per_condition(cond):
    """Lights + world give the target lux at the plate centre (brute-force check, unoccluded) within 10 %."""
    lo, hi = Lt.SPEC[cond]["lux"]
    for seed in range(12):
        plan = _plan(seed)
        amb = Lt.ambient_for(np.random.default_rng(seed), plan, CFG, cond)
        assert amb["condition"] == cond and set(amb) >= {"lux", "cct_k", "world_color", "world_strength", "lights",
                                                         "occluders", "change"}
        assert all(L["group"] == "AMB" for L in amb["lights"])
        p0 = Lt.plate_centre(plan)
        lights = sum(_brute_lux(L, p0) for L in amb["lights"])
        world = _world_lux(amb)
        if cond == "C0_dark":
            assert amb["lux"] == 0 and not amb["lights"] and amb["world_strength"] == 0
            assert amb["change"] == {"frame_sigma": 0.0, "shadow_shift_mm": 0.0}
            continue
        assert lo <= amb["lux"] <= hi
        assert lights + world == pytest.approx(amb["lux"], rel=0.10), (seed, lights, world, amb["lux"])
        assert 0.045 <= world / amb["lux"] <= 0.155                 # the world gives 5-15 % of the lux
        assert Lt.ambient_lux(amb, p0) == pytest.approx(amb["lux"], rel=1e-3)
        cl, ch = Lt.SPEC[cond]["cct"]
        assert cl <= amb["cct_k"] <= ch


def _el_deg(v) -> float:
    v = np.asarray(v, float)
    return math.degrees(math.asin(v[2] / np.linalg.norm(v)))


def test_condition_scenes_follow_the_table():
    for seed in range(15):
        plan = _plan(100 + seed)
        p0 = Lt.plate_centre(plan)
        rng = np.random.default_rng(seed)
        office = Lt.ambient_for(rng, plan, CFG, "C1_office")
        assert 2 <= len(office["lights"]) <= 6
        assert all(L["kind"] == "area" and 2.15 <= L["pos_m"][2] <= 3.05 and L["dir"] == [0, 0, -1.0]
                   for L in office["lights"])
        assert len({L["power_w"] for L in office["lights"]}) == 1          # identical fixtures
        assert 0 <= office["change"]["frame_sigma"] <= 0.03
        bulbs = Lt.ambient_for(rng, plan, CFG, "C2_warm_bulb")
        assert 1 <= len(bulbs["lights"]) <= 2
        for L in bulbs["lights"]:
            v = np.asarray(L["pos_m"]) - p0
            assert L["kind"] == "point" and 0.4 <= np.linalg.norm(v) <= 1.2 and 20 <= _el_deg(v) <= 55
        win = Lt.ambient_for(rng, plan, CFG, "C3_window")["lights"]
        assert len(win) == 1 and win[0]["kind"] == "area" and win[0]["size_m"] == [1.5, 1.2]
        assert 1.5 <= np.linalg.norm(np.asarray(win[0]["pos_m"]) - p0) <= 4.0 and abs(win[0]["dir"][2]) < 1e-9
        lamp = Lt.ambient_for(rng, plan, CFG, "C4_side_lamp")
        L = lamp["lights"][0]
        v = np.asarray(L["pos_m"]) - p0
        assert L["kind"] == "spot" and 30 <= L["spot_deg"] <= 60 and 0.4 <= np.linalg.norm(v) <= 1.0
        assert 10 <= _el_deg(v) <= 30 and len(lamp["occluders"]) == 1
        assert lamp["change"]["frame_sigma"] == 0.01 and 0 <= lamp["change"]["shadow_shift_mm"] <= 1.5
        sun = Lt.ambient_for(rng, plan, CFG, "C6_sun")["lights"][0]
        assert sun["kind"] == "sun" and sun["angle_deg"] == 0.53 and 20 <= _el_deg(-np.asarray(sun["dir"])) <= 70
        assert 30000 <= Lt.ambient_for(np.random.default_rng(seed), plan, CFG, "C6_sun")["lux"] <= 100000


def test_directional_lamps_are_not_shadowed_by_the_device():
    for seed in range(25):
        plan = _plan(200 + seed)
        p0 = Lt.plate_centre(plan)
        for cond in ("C2_warm_bulb", "C3_window", "C4_side_lamp", "C6_sun"):
            amb = Lt.ambient_for(np.random.default_rng(seed), plan, CFG, cond)
            assert not amb["device_shadow"]
            for L in amb["lights"]:
                d = -np.asarray(L["dir"]) if L["kind"] == "sun" else np.asarray(L["pos_m"]) - p0
                tmax = np.inf if L["kind"] == "sun" else float(np.linalg.norm(d))
                assert not Lt.device_blocks(plan, CFG, p0, d, tmax), (cond, seed)


def test_side_lamp_arm_casts_an_edge_across_the_plate():
    """The arm's shadow edge runs through edge_uv_mm along the lamp's direction: shadow one side, light the other."""
    for seed in range(20):
        plan = _plan(300 + seed)
        amb = Lt.ambient_for(np.random.default_rng(seed), plan, CFG, "C4_side_lamp")
        occ, lamp = amb["occluders"][0], amb["lights"][0]
        assert Lt.occluder_corners(occ)[:, 2].min() > 0.03                  # above the plate
        C = Lt._pose(plan)[2]
        for uv in np.stack(np.meshgrid(np.linspace(0, 100, 11), np.linspace(0, 100, 11)), -1).reshape(-1, 2):
            p = F.plate_to_world([uv])[0]
            assert not Lt.occluder_blocks(occ, C, p - C, float(np.linalg.norm(p - C)))   # never hides the plate
        q = np.asarray(occ["edge_uv_mm"])
        a = math.radians(occ["edge_dir_deg"])
        across = np.array([-math.sin(a), math.cos(a)])                   # plate frame, perpendicular to the edge
        sides = []
        for k in (-1, 1):
            pts = [q + k * s * across for s in (3.0, 8.0, 15.0)]
            sides.append([Lt.shadowed(amb, F.plate_to_world([p])[0], lamp) for p in pts])
        assert sorted(map(all, sides)) == [False, True] and not any(sides[0][i] and sides[1][i] for i in range(3)), seed


def test_side_lamp_arm_never_shades_the_leds():
    """Regression: the arm used to be checked against camera rays only, so now and then (about 1 in 150
    sampler plans) it cut an LED's path to the plate and L/R differed between conditions."""
    rig = RIG.load_rig(CFG, ROOT)
    plan = _plan(0, straight=True)
    plan["lights"] = RIG.lights_world(rig, plan["camera"]["R"], plan["camera"]["t"])
    led = min((L for L in plan["lights"] if L["group"] in ("L", "R")), key=lambda L: L["pos_m"][0])
    corner = F.plate_to_world([[100.0, 50.0]])[0]
    c = corner + 0.85 * (np.asarray(led["pos_m"]) - corner)            # on the LED's ray, near the device
    occ = {"kind": "box", "center_m": c.tolist(), "size_m": [0.01, 0.01, 0.01], "rot_deg": [0.0, 0.0, 0.0]}
    C, pts = Lt._view_rays(plan)
    assert not any(Lt.occluder_blocks(occ, C, p - C, float(np.linalg.norm(p - C))) for p in pts)   # camera sees past it
    assert Lt._occluder_ok(occ, {k: v for k, v in plan.items() if k != "lights"}, CFG)
    assert not Lt._occluder_ok(occ, plan, CFG)
    for seed in range(60):
        p = _plan(500 + seed)
        p["lights"] = RIG.lights_world(rig, p["camera"]["R"], p["camera"]["t"])
        amb = Lt.ambient_for(np.random.default_rng(seed), p, CFG, "C4_side_lamp")
        assert len(amb["occluders"]) == 1, seed
        grid = F.plate_to_world(np.stack(np.meshgrid(np.linspace(0, 100, 9), np.linspace(0, 100, 9)), -1).reshape(-1, 2))
        for L in (L for L in p["lights"] if L["group"] in ("L", "R")):
            for q in grid:
                v = q - np.asarray(L["pos_m"])
                assert not Lt.occluder_blocks(amb["occluders"][0], L["pos_m"], v, float(np.linalg.norm(v))), seed


def test_reflected_lamp_mirror_image_lands_on_the_plate():
    for seed in range(15):
        plan = _plan(400 + seed)
        amb = Lt.ambient_for(np.random.default_rng(seed), plan, CFG, "C5_reflected_lamp")
        glare = next(L for L in amb["lights"] if L["name"] == "AMB_glare")
        u, v = amb["mirror_uv_mm"]
        assert 0 < u < 100 and 0 < v < 100 and not amb["mirror_blocked"]
        assert 5000 - 1 <= glare["luminance_cd_m2"] <= 40000 + 1
        # the camera ray to that point, mirrored by the plate, hits the panel's centre
        C = Lt._pose(plan)[2]
        p = F.plate_to_world([[u, v]])[0]
        r = p - C
        r = np.array([r[0], r[1], -r[2]]) / np.linalg.norm(r)
        to = np.asarray(glare["pos_m"]) - p
        assert np.allclose(to / np.linalg.norm(to), r, atol=1e-5)
        assert not Lt.device_blocks(plan, CFG, p, r)
        assert np.dot(glare["dir"], -r) == pytest.approx(1.0, abs=1e-5)   # faces the plate point
        assert sum(L["lux"] for L in amb["lights"] if L is not glare) > 0.3 * amb["lux"]


def test_ambient_is_reproducible_and_conditions_are_drawn_from_cfg():
    plan = _plan(5)
    a = Lt.ambient_for(np.random.default_rng(9), plan, CFG, "C4_side_lamp")
    b = Lt.ambient_for(np.random.default_rng(9), plan, CFG, "C4_side_lamp")
    assert a == b
    seen = {Lt.ambient_for(np.random.default_rng(s), plan, CFG)["condition"] for s in range(200)}
    assert seen == set(Lt.CONDITIONS)
    with pytest.raises(ValueError):
        Lt.ambient_for(np.random.default_rng(0), plan, CFG, "C9_moon")


def test_euler_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        M = q * np.sign(np.linalg.det(q))
        assert np.allclose(Lt.euler_xyz_matrix(Lt.euler_xyz_deg(M)), M, atol=1e-6)


# ----------------------------------------------------------------------------- hood
def test_hood_only_in_hood_mode():
    plan = _plan(1, straight=True)
    assert Lt.hood_for(np.random.default_rng(0), plan, CFG, "open") is None
    assert Lt.hood_for(np.random.default_rng(0), plan, CFG, None) is None
    h = Lt.hood_for(np.random.default_rng(0), plan, CFG, "hood")
    assert {"top_mm", "bottom_mm", "wall_mm", "gap_mm", "albedo", "rests_on"} <= set(h)
    assert h["top_mm"] == [170.0, 160.0] and h["bottom_mm"] == [200.0, 190.0] and h["albedo"] == 0.03


def test_hood_rests_on_the_bench_or_the_part():
    plan = _plan(2, straight=True, t_mm=4.0)
    hoods = [Lt.hood_for(np.random.default_rng(s), plan, CFG, "hood") for s in range(300)]
    bench = [h for h in hoods if h["rests_on"] == "bench"]
    assert 0.62 < len(bench) / len(hoods) < 0.78
    assert all(h["gap_mm"] == -4.0 for h in bench)                       # rim on the bench, sealed
    assert all(0.0 < h["gap_mm"] <= 1.0 for h in hoods if h["rests_on"] == "part")
    # a 150 x 100 plate seen far off its centre pokes out of the opening: the hood rests on it
    big = _plan(3, size=(150.0, 100.0), straight=True, target=(30.0, 50.0))
    assert not Lt.plate_fits_hood(big)
    assert all(Lt.hood_for(np.random.default_rng(s), big, CFG, "hood")["rests_on"] == "part" for s in range(20))
    assert Lt.plate_fits_hood(_plan(3, size=(150.0, 100.0), straight=True))     # centred, it fits


def test_side_lamp_arm_stays_outside_the_hood():
    """Regression: in hood plans the arm was checked against the device box only, and about 3 % of
    C4 hood scenes put it through the hood's wall (lit by the LEDs inside, seen in the plate's mirror)."""
    plan = _plan(4, straight=True)
    plan["extras"] = {"device": {"mode": "hood"}}
    plan["extras"]["hood"] = Lt.hood_for(np.random.default_rng(0), plan, CFG, "hood")
    C = Lt._pose(plan)[2]
    p0 = Lt.plate_centre(plan)
    inside = Lt.in_hood(plan, [p0 + [0, 0, 0.05], C + [0, 0, -0.03], p0 + [0.3, 0, 0.05], p0 + [0, 0, 0.4]])
    assert inside.tolist() == [True, True, False, False]
    corners = Lt.hood_corners(plan)
    assert Lt.in_hood(plan, corners["outer_bottom"], margin_m=1e-9).all()
    assert not Lt.in_hood(plan, corners["outer_bottom"] + [0, 0, -0.001]).any()                # below the rim
    rig = RIG.load_rig(CFG, ROOT)
    # a box through the hood's +u wall, out of the camera's and the LEDs' way: fine open, refused with the hood
    plan["lights"] = RIG.lights_world(rig, plan["camera"]["R"], plan["camera"]["t"])
    occ = {"kind": "box", "center_m": [0.14, -0.05, 0.05], "size_m": [0.04, 0.04, 0.04], "rot_deg": [0.0, 0.0, 0.0]}
    assert Lt.in_hood(plan, Lt.occluder_corners(occ)).any() and not Lt.in_hood(plan, Lt.occluder_corners(occ)).all()
    assert Lt._occluder_ok(occ, dict(plan, extras={"device": {"mode": "open"}}), CFG)
    assert not Lt._occluder_ok(occ, plan, CFG)
    for seed in range(80):
        rng = np.random.default_rng(seed)
        R, t = F.camera_pose((50 + rng.uniform(-5, 5), 50 + rng.uniform(-5, 5)), 200 + rng.uniform(-1, 1),
                             rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(0, 360))
        p = {"plate": {"size_mm": [100.0, 100.0], "thickness_mm": 3.0}, "camera": {"R": R.tolist(), "t": t.tolist()},
             "extras": {"device": {"mode": "hood"}}}
        p["extras"]["hood"] = Lt.hood_for(rng, p, CFG, "hood")
        p["lights"] = RIG.lights_world(rig, R, t)
        amb = Lt.ambient_for(rng, p, CFG, "C4_side_lamp")
        assert len(amb["occluders"]) == 1, seed
        occ = amb["occluders"][0]
        M, h = Lt.euler_xyz_matrix(occ["rot_deg"]), np.asarray(occ["size_m"]) / 2
        g = np.stack(np.meshgrid(*(np.linspace(-1, 1, 9) for _ in range(3)), indexing="ij"), -1).reshape(-1, 3)
        assert not Lt.in_hood(p, np.asarray(occ["center_m"]) + (g * h) @ M.T).any(), seed


def test_hood_corners_lie_on_the_bottom_plane_and_the_top_on_the_pods():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        R, t = F.camera_pose((50 + rng.uniform(-3, 3), 50 + rng.uniform(-3, 3)), 200 + rng.uniform(-1, 1),
                             rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(0, 360))
        plan = {"plate": {"size_mm": [100.0, 100.0], "thickness_mm": 3.0},
                "camera": {"R": R.tolist(), "t": t.tolist()}}
        hood = Lt.hood_for(rng, plan, CFG, "hood")
        c = Lt.hood_corners(plan, hood)
        zb = hood["gap_mm"] / 1000
        for k in ("inner_bottom", "outer_bottom"):
            assert np.allclose(c[k][:, 2], zb, atol=1e-12)
        for k in ("inner_top", "outer_top"):
            assert np.allclose((c[k] @ R.T + t)[:, 2], 0.017, atol=1e-12)    # camera-frame z = 17 mm
        # nearly straight down: the inner opening is about 200 x 190 mm, the wall 2 mm thick
        cam = (c["inner_bottom"] - F.camera_centre(R, t)) @ R.T
        assert np.allclose(np.abs(cam[:, 0]), 0.100, atol=0.003) and np.allclose(np.abs(cam[:, 1]), 0.095, atol=0.003)
        outer = (c["outer_bottom"] - F.camera_centre(R, t)) @ R.T
        assert np.allclose(np.abs(outer[:, :2]) - np.abs(cam[:, :2]), 0.002, atol=2e-4)
        # every LED of the rig is inside the hood (in front of the lid, inside the top rectangle)
        for led in RIG.load_rig(CFG, ROOT).leds:
            x, y, z = led.p_cam_mm
            assert abs(x) < 85 and abs(y) < 80 and z > -1


# ---- device v2: the hood's dark-field LED ring (DESIGN_STEEL C2) ------------------------------

def _hood_plan(seed: int, size=(100.0, 100.0), t_mm=3.0) -> dict:
    """A plate under the hood (sampler hood pose limits: 200 +- 1 mm, tilt +-1.5 deg) with its hood dict."""
    rng = np.random.default_rng(seed)
    W, H = size
    target = (W / 2 + rng.uniform(-0.05, 0.05) * W, H / 2 + rng.uniform(-0.05, 0.05) * H)
    R, t = F.camera_pose(target, rng.uniform(199, 201), rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5),
                         rng.uniform(-20, 20) + 90 * int(rng.integers(4)))
    plan = {"plate": {"size_mm": [W, H], "thickness_mm": t_mm}, "camera": {"R": R.tolist(), "t": t.tolist()},
            "extras": {"device": {"mode": "hood"}}}
    plan["extras"]["hood"] = Lt.hood_for(rng, plan, CFG, "hood")
    return plan


def test_ring_leds_sit_inside_the_hood_walls_aimed_at_the_centre():
    white = RIG.white_params(CFG, ROOT)
    rc = Lt.ring_config(CFG)
    assert rc == {"enabled": True, "n_leds": 18, "inset_mm": 6.0, "h_mm": 18.0}         # twin.yaml rig.ring
    for seed in range(12):
        plan = _hood_plan(seed, size=[(100.0, 100.0), (150.0, 100.0), (100.0, 60.0)][seed % 3], t_mm=1.5 + seed % 4)
        lights = Lt.ring_lights(plan, CFG, white)
        assert len(lights) == 18 and all(L["mount"] == "ring" and L["kind"] == "spot" for L in lights)
        pos = np.array([L["pos_m"] for L in lights])
        assert np.allclose(pos[:, 2], 0.018, atol=1e-9)                                 # 18 mm above the plate face
        # inside the hood: within its inner cross-section at that height, 6 mm in from the walls
        inner = Lt.hood_corners(plan, None, 0.018)["inner_bottom"][:, :2]
        assert Lt._inside(inner, pos[:, :2], 0.006 - 1e-6) and not Lt._inside(inner, pos[:, :2], 0.0062)
        assert np.all(Lt.in_hood(plan, pos))
        # aimed at the plate point on the optical axis, at a low (dark-field) elevation
        p0 = Lt.axis_point(plan)
        for L in lights:
            to = (p0 - np.asarray(L["pos_m"])) / np.linalg.norm(p0 - np.asarray(L["pos_m"]))
            assert np.dot(to, L["dir"]) > 1 - 1e-6                                         # dir rounded to 1e-7
        el = Lt.ring_elevations_deg(plan, lights)
        assert 8.0 <= el.min() and el.max() <= 25.0, el
        assert L["power_w"] == pytest.approx(white["power_w"], rel=1e-4) and L["spot_deg"] == white["spot_deg"]
        # both halves light every part of the plate: corners and edge middles get at least 0.1 x the
        # centre irradiance from each, and at most 8 x in all (the near LEDs of a 150 mm plate)
        W, H = plan["plate"]["size_mm"]
        e0 = sum(Lt.light_irradiance_w(L, p0) for L in lights)
        for c in F.plate_to_world([[0, 0], [W, 0], [W, H], [0, H], [W / 2, 0], [0, H / 2]]):
            e = {g: sum(Lt.light_irradiance_w(L, c) for L in lights if L["group"] == g) for g in "LR"}
            assert min(e.values()) > 0.1 * e0 and sum(e.values()) < 8.0 * e0, e


def test_ring_groups_split_at_the_image_centre():
    """L = the LEDs on the image left (camera x < 0), R = the rest: 9 + 9, mirror-symmetric, power per side."""
    white = RIG.white_params(CFG, ROOT)
    for seed in range(6):
        plan = _hood_plan(100 + seed)
        R, t = np.array(plan["camera"]["R"]), np.array(plan["camera"]["t"])
        lights = Lt.ring_lights(plan, CFG, white, {"white_L": 0.9, "white_R": 1.1})
        xs = {g: sorted(float((R @ np.asarray(L["pos_m"]) + t)[0]) for L in lights if L["group"] == g) for g in "LR"}
        assert len(xs["L"]) == len(xs["R"]) == 9 and max(xs["L"]) < 0 < min(xs["R"])
        assert np.allclose(sorted(-x for x in xs["L"]), xs["R"], atol=2e-3)          # symmetric about x = 0 (tilt)
        assert {L["name"] for L in lights} == {f"WL_{g}_{i}" for g in "LR" for i in range(9)}
        for L in lights:
            assert L["power_w"] == pytest.approx(white["power_w"] * (0.9 if L["group"] == "L" else 1.1), rel=1e-4)
    # the ring's irradiance at the centre is a few klux, as the pods' (the fixed exposure follows it)
    e = Lt.led_irradiance({**plan, "lights": lights})
    assert 2000 < Lt.w_to_lux(e) < 5000


def test_sampler_hood_plans_use_the_ring_and_open_plans_the_pods():
    from trinetra.twin import sampler as S

    for seed in range(4):
        h = S.make_plan(CFG, seed, "train", condition="C0_dark", device_mode="hood")
        o = S.make_plan(CFG, seed, "train", condition="C0_dark", device_mode="open")
        assert h["extras"]["device"] == {"mode": "hood", "illumination": "ring"}
        assert o["extras"]["device"] == {"mode": "open", "illumination": "pods"}
        ring = [L for L in h["lights"] if L["group"] in ("L", "R")]
        assert len(ring) == 18 and all(L.get("mount") == "ring" for L in ring)
        assert not any(L.get("mount") for L in o["lights"])
        assert [L["name"] for L in h["lights"] if L["group"] == "UV"] == [L["name"] for L in o["lights"] if L["group"] == "UV"]
        assert h["plate"] == o["plate"] and h["defects"] == o["defects"]                # only the view changes
    off = copy.deepcopy(CFG)
    off["rig"]["ring"] = {"enabled": False}
    h = S.make_plan(off, 1, "train", device_mode="hood")
    assert h["extras"]["device"]["illumination"] == "pods" and not any(L.get("mount") for L in h["lights"])
    assert Lt.illumination_for(CFG, "hood") == "ring" and Lt.illumination_for(CFG, "open") == "pods"


def test_side_lamp_shadow_moves_slowly():
    """A moving hand shadow is 0-1.5 mm per frame (a hand at <= 15 mm/s); 0-4 mm refused most C4 bursts."""
    assert Lt.SPEC["C4_side_lamp"]["shift"] == (0.0, 1.5)
    shifts = [Lt.ambient_for(np.random.default_rng(s), _plan(s), CFG, "C4_side_lamp")["change"]["shadow_shift_mm"]
              for s in range(40)]
    assert 0.0 <= min(shifts) and max(shifts) <= 1.5 and max(shifts) > 1.0
