"""twin/burst.py: the simulated LED burst and the naive baseline arm (DESIGN_STEEL.md E). No renders."""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from ayas import lightproc as LP
from ayas.twin import burst as BU

CFG = LP.load_capture_config()
SENSOR = CFG["sensor"]
E = 10.0                                                   # fake LED irradiance (W/m^2) at the plate


def card(h=120, w=120, albedo=0.9, amb=0.0):
    """Uniform matte card: each pod gives E/2, ambient amb x E."""
    rad = albedo * (E / 2) / math.pi
    one = np.full((h, w, 3), rad, np.float32)
    return {"L": one, "R": one.copy(), "AMB": one * np.float32(2 * amb)}


def plan(distance=200.0, shake=0.0, jitter=0.0, sigma=0.0, shadow=0.0, w=120):
    f = distance / 0.065
    return {"camera": {"K": [[f, 0, w / 2], [0, f, w / 2], [0, 0, 1]], "roi": [0, 0, w, w], "pose": {"distance_mm": distance}},
            "capture": {"shake_px_per_frame": shake, "led_jitter": jitter},
            "ambient": {"condition": "C1_office", "change": {"frame_sigma": sigma, "shadow_shift_mm": shadow}},
            "plate": {"material": {"kind": "ss_2b"}}}


def sim(groups, pl, seed=0, **kw):
    return BU.simulate_burst(groups, pl, np.random.default_rng(seed), CFG, e_led=E, **kw)


def test_fixed_exposure_white_card_reads_ref_full_scale():
    b = sim(card(), plan())
    ref = CFG["burst"]["ref_full_scale"]
    assert b.frames["B1"].mean() == pytest.approx(ref, abs=0.003)
    assert b.frames["B2"].mean() == pytest.approx(ref, abs=0.003)
    assert b.frames["L"].mean() == pytest.approx(ref / 2, abs=0.003)
    assert b.frames["R"].mean() == pytest.approx(ref / 2, abs=0.003)
    assert abs(b.frames["A"].mean()) < 0.001                # LEDs off, dark room
    assert b.meta["k"] == pytest.approx(ref * math.pi / (CFG["burst"]["ref_albedo"] * E))
    assert set(b.frames) == set(LP.SEQUENCE)
    assert all(f.dtype == np.float32 and f.shape == (120, 120, 3) for f in b.frames.values())


def test_exposure_does_not_follow_distance():
    """Exposure is calibrated at ref_distance_mm: a plate at 250 mm with the same pass reads 0.64x."""
    near = sim(card(), plan(200.0)).frames["B1"].mean()
    far = sim(card(), plan(250.0)).frames["B1"].mean()
    assert far / near == pytest.approx((200 / 250) ** 2, rel=0.01)
    k = BU.exposure_k(plan(250.0), CFG, E)
    assert k["E_ref"] == pytest.approx(E * (250 / 200) ** 2) and k["E_led"] == E
    with pytest.raises(ValueError):
        BU.exposure_k(plan(), CFG, 0.0)


def test_sensor_noise_statistics_and_quantisation():
    rng = np.random.default_rng(1)
    e_fs, levels = LP.electrons_per_fs(SENSOR), LP.raw_levels(SENSOR)
    for s in (0.002, 0.05, 0.4):
        S = np.full((400, 400, 3), s, np.float32)
        v = BU.expose(S, rng, SENSOR)
        want = s / e_fs + (SENSOR["read_noise_e"] / e_fs) ** 2 + 1 / (12 * levels ** 2)
        assert v.mean() == pytest.approx(s, rel=0.01, abs=2e-5)
        assert v.var() == pytest.approx(want, rel=0.06)
        assert np.allclose(v * levels, np.rint(v * levels), atol=1e-3)   # whole DN steps
    hot = BU.expose(np.full((50, 50, 3), 3.0, np.float32), rng, SENSOR)
    assert hot.max() == 1.0 and hot.min() == 1.0              # clips at full scale
    dark = BU.expose(np.zeros((200, 200, 3), np.float32), rng, SENSOR)
    assert abs(dark.mean()) < 2e-4 and dark.min() >= -SENSOR["black_level"] / levels
    assert dark.min() < 0                                     # read noise below black is kept


def test_led_states_and_jitter():
    g = card()
    g["L"] = np.zeros_like(g["R"])                           # only the right pod is lit
    b = sim(g, plan(jitter=0.05))
    assert abs(b.frames["L"].mean()) < 0.001 and abs(b.frames["A"].mean()) < 0.001
    assert b.frames["R"].mean() == pytest.approx(b.frames["B1"].mean(), rel=0.01)
    for seed in range(20):
        gains = sim(card(40, 40), plan(jitter=0.05, w=40), seed).meta["led_gains"]
        assert all(0.95 <= v <= 1.05 for v in gains.values())
    assert sim(card(), plan()).meta["led_gains"] == {"L": 1.0, "R": 1.0}


def test_shake_random_walk():
    rng = np.random.default_rng(3)
    steps = np.concatenate([np.diff(BU.shake_walk(rng, 5, 1.5), axis=0) for _ in range(3000)])
    assert np.sqrt((steps ** 2).sum(axis=1).mean()) == pytest.approx(1.5, rel=0.03)
    w = BU.shake_walk(rng, 5, 2.0)
    assert w.shape == (5, 2) and np.all(w[0] == 0)
    assert np.all(BU.shake_walk(rng, 5, 0.0) == 0)


def test_frames_move_by_meta_shifts():
    g, pl, roi, e = BU.toy_scene(seed=1, r=0.1, shake_px_per_frame=1.5)
    b = BU.simulate_burst(g, pl, np.random.default_rng(4), CFG, roi, e_led=e)
    assert b.meta["shifts"]["B1"] == [0.0, 0.0]
    for k in ("L", "R", "B2"):
        dx, dy, _ = LP.estimate_shift(b.frames["B1"], b.frames[k])
        assert abs(dx - b.meta["shifts"][k][0]) < 0.1 and abs(dy - b.meta["shifts"][k][1]) < 0.1, k
    forced = BU.simulate_burst(g, pl, np.random.default_rng(4), CFG, roi, e_led=e,
                               shifts={"B2": (2.3, -0.7)})
    assert forced.meta["shifts"]["B2"] == [2.3, -0.7] and forced.meta["shifts"]["L"] == [0.0, 0.0]


def test_ambient_factors_and_moving_shadow():
    rng = np.random.default_rng(5)
    a = np.concatenate([BU.ambient_factors(rng, 5, 0.03) for _ in range(2000)])
    assert a.mean() == pytest.approx(1.0, abs=0.003) and a.std() == pytest.approx(0.03, rel=0.05)
    assert np.all(BU.ambient_factors(rng, 5, 0.0) == 1.0)
    # ambient-only scene: textured plate x a soft hand-shadow edge. A moving shadow moves only the
    # illumination (2 x shadow_shift_mm in the A frame); the texture stays where it is
    h = w = 256
    xx = np.mgrid[0:h, 0:w][1].astype(np.float32)
    tex = 0.5 + 0.1 * BU._smooth_noise(np.random.default_rng(0), (h, w), 1.5)
    k = BU.exposure_k(plan(w=w), CFG, E)["k"]
    g = {"L": np.zeros((h, w, 3), np.float32), "R": np.zeros((h, w, 3), np.float32),
         "AMB": np.dstack([tex * (0.3 + 0.35 * (1 + np.tanh((xx - 128) / 30.0)))] * 3).astype(np.float32) / np.float32(k)}
    b = sim(g, plan(shadow=0.5, w=w), seed=6)
    ppm = b.meta["px_per_mm"]
    assert ppm == pytest.approx(1 / 0.065, rel=1e-6)
    sa = b.meta["shadow_px"]["A"]
    assert math.hypot(*sa) == pytest.approx(2 * 0.5 * ppm, rel=1e-6)
    dx, dy, _ = LP.estimate_shift(b.frames["B1"], b.frames["A"])
    assert math.hypot(dx, dy) < 0.15                                    # texture static (was: moved by sa)
    assert b.meta["shadow_px"]["B1"] == [0.0, 0.0]
    sig = sim(card(), plan(sigma=0.02), seed=7).meta["ambient_factors"]
    assert set(sig) == set(LP.SEQUENCE) and len(set(sig.values())) == 5


def test_deterministic_and_meta_json():
    import json
    g, pl, roi, e = BU.toy_scene(seed=2, r=0.3, shake_px_per_frame=1.0, frame_sigma=0.01, shadow_shift_mm=0.5)
    b1 = BU.simulate_burst(g, pl, np.random.default_rng(9), CFG, roi, e_led=e)
    b2 = BU.simulate_burst(g, pl, np.random.default_rng(9), CFG, roi, e_led=e)
    assert all(np.array_equal(b1.frames[k], b2.frames[k]) for k in LP.SEQUENCE)
    assert b1.meta == b2.meta
    json.dumps(b1.meta)
    assert b1.meta["r_true"] == pytest.approx(0.3, rel=0.25)
    assert b1.meta["material"] == "ms_cold_rolled"


def test_led_irradiance_comes_from_lighting_lazily(monkeypatch):
    calls = []

    def fake(plan, groups=("L", "R")):
        calls.append(tuple(groups))
        return 12.5

    mod = types.ModuleType("ayas.twin.lighting")
    mod.led_irradiance = fake
    monkeypatch.setitem(sys.modules, "ayas.twin.lighting", mod)
    b = BU.simulate_burst(card(), plan(), np.random.default_rng(0), CFG)
    assert calls == [("L", "R")] and b.meta["E_led"] == 12.5
    assert b.frames["B1"].mean() == pytest.approx(CFG["burst"]["ref_full_scale"] * E / 12.5, abs=0.003)


def test_missing_groups_are_zero_and_uv_ignored():
    g = card()
    g["UV"] = np.full_like(g["L"], 5.0)
    del g["AMB"]
    b = sim(g, plan())
    assert abs(b.frames["A"].mean()) < 0.001
    with pytest.raises(ValueError):
        sim({"UV": g["UV"]}, plan())


def test_naive_auto_exposure_and_grey_world():
    g, pl, roi, e = BU.toy_scene(seed=3, r=1.0, tint=(1.0, 0.7, 0.4))   # strongly warm ambient
    out = BU.simulate_naive(g, pl, np.random.default_rng(1), CFG, roi, e_led=e)
    assert out.shape == roi.shape + (3,) and out.dtype == np.uint8
    er = LP._roi(roi, roi.shape, 6)
    target = 255 * (1.055 * 0.18 ** (1 / 2.4) - 0.055)                  # sRGB of 0.18 FS
    assert np.median(LP.luma(out.astype(np.float32))[er]) == pytest.approx(target, abs=8)
    means = out[er].astype(float).mean(axis=0)
    assert np.ptp(means) < 3.0                                           # grey world removed the cast
    # the old arm is not fixed-exposure: a 4x brighter plate gives the same output level
    g2 = {k: v * 4 for k, v in g.items()}
    out2 = BU.simulate_naive(g2, pl, np.random.default_rng(1), CFG, roi, e_led=e)
    assert abs(float(np.median(out2[er])) - float(np.median(out[er]))) < 3


def test_toy_scene_contract():
    g, pl, roi, e = BU.toy_scene(64, 80, seed=0, r=0.5)
    assert set(g) == {"L", "R", "AMB"} and all(v.shape == (64, 80, 3) and v.dtype == np.float32 for v in g.values())
    assert roi.shape == (64, 80) and roi.dtype == bool and e == 10.0
    assert pl["ambient"]["change"] == {"frame_sigma": 0.0, "shadow_shift_mm": 0.0}
    assert BU.px_per_mm(pl, (64, 80)) == pytest.approx(1 / 0.065)
    assert BU.px_per_mm({}, (64, 80)) == pytest.approx(BU.DEFAULT_PX_PER_MM)


def test_with_sampler_plan_and_lighting():
    """Integration with the other twin modules: a sampler plan, the analytic LED irradiance of
    twin/lighting.py, passes at 1/8 of the ROI resolution (px_per_mm follows the pass size)."""
    yaml = pytest.importorskip("yaml")
    lighting = pytest.importorskip("ayas.twin.lighting")
    from ayas.twin import sampler
    cfg = yaml.safe_load(open(LP.ROOT / "configs" / "twin.yaml"))
    try:
        pl = sampler.make_plan(cfg, 3, "train")
    except Exception as exc:                                   # sampler owner's problem, not this module's
        pytest.skip(f"make_plan failed: {exc}")
    x0, y0, w, h = pl["camera"]["roi"]
    g, _, roi, _ = BU.toy_scene(h // 8, w // 8, seed=1, margin_px=4)
    b = BU.simulate_burst(g, pl, np.random.default_rng(0), CFG, roi)
    E = lighting.led_irradiance(pl, groups=("L", "R"))
    assert b.meta["E_led"] == pytest.approx(E) and 1.0 < E * 683 / 1000 < 20.0       # klux at the plate
    d = pl["camera"]["pose"]["distance_mm"]
    assert b.meta["E_ref"] == pytest.approx(E * (d / CFG["burst"]["ref_distance_mm"]) ** 2)
    assert b.meta["px_per_mm"] == pytest.approx(pl["camera"]["K"][0][0] / 8 * (w // 8 * 8 / w) / d, rel=0.02)
    p = LP.process_burst(b, roi, CFG)
    assert p.gate.verdict in LP.VERDICTS and p.image_u8.shape == roi.shape + (3,)


# ---- C4 moving shadow and shake scaling (device v2 redesign, 14 Nov) ----------------------------------

def test_moving_shadow_moves_only_the_illumination():
    """AMB = texture x illumination: shadow_shift moves the soft shadow edge by s, the texture stays put
    (the old model translated the whole pass, so alignment locked onto the A frame's texture)."""
    h = w = 320
    rng = np.random.default_rng(1)
    xx = np.mgrid[0:h, 0:w][1].astype(np.float32)
    tex = (0.5 + 0.12 * BU._smooth_noise(rng, (h, w), 1.2)).astype(np.float32)
    def illum(x0):
        return 0.25 + 0.375 * (1.0 + np.tanh((xx - x0) / 120.0))          # a ~1 cm penumbra (lamp 1-3 cm)
    amb = np.dstack([tex * illum(160.0)] * 3).astype(np.float32)
    sigma = 5.0 * 15.4                                       # 5 mm at 15.4 px/mm
    moved = BU.shadow_shift(amb, (12.0, 0.0), sigma)
    truth = np.dstack([tex * illum(172.0)] * 3)
    inner = (slice(40, 280), slice(40, 280))
    rel = np.abs(moved[inner] - truth[inner]) / truth[inner]
    assert np.median(rel) < 0.02                             # the shadow edge moved by 12 px, texture kept
    old = LP.translate(amb, 12.0, 0.0)
    assert np.median(np.abs(old[inner] - truth[inner]) / truth[inner]) > 5 * np.median(rel)
    dx, dy, _ = LP.estimate_shift(amb, moved)
    assert math.hypot(dx, dy) < 0.1                          # alignment sees no motion in the texture
    assert BU.shadow_shift(amb, (0.0, 0.0), sigma) is amb
    # with the plate mask, the bench's level does not leak across the outline when the light moves
    plate = np.zeros((h, w), bool)
    plate[60:260, 60:260] = True
    flat = np.where(plate[..., None], 0.5, 0.05).astype(np.float32) * np.ones((1, 1, 3), np.float32)
    m = BU.shadow_shift(flat, (20.0, 0.0), sigma, mask=plate)
    assert np.allclose(m[plate], 0.5, rtol=0.02) and np.allclose(m[~plate], 0.05, rtol=0.02)
    um = BU.shadow_shift(flat, (20.0, 0.0), sigma)            # no mask: the outline's blur moves too
    assert np.abs(um[plate] - 0.5).max() > 0.03 > 3 * np.abs(m[plate] - 0.5).max()


def test_shake_is_scaled_to_the_render_resolution():
    """capture.shake_px_per_frame is in full-resolution sensor px: a 0.25x render shakes 0.25x as many
    of its px (it was applied in render px, 4x too much); lightproc reports motion in sensor px."""
    w = 120
    full = plan(shake=2.0, w=w)
    quarter = plan(shake=2.0, w=w)
    quarter["camera"]["roi"] = [0, 0, 4 * w, 4 * w]          # the pass is 1/4 of the ROI: res_scale 0.25
    quarter["camera"]["K"] = [[4 * 200 / 0.065, 0, 2 * w], [0, 4 * 200 / 0.065, 2 * w], [0, 0, 1]]
    assert BU.res_scale(full, (w, w)) == 1.0 and BU.res_scale(quarter, (w, w)) == 0.25
    assert BU.px_per_mm(quarter, (w, w)) == pytest.approx(BU.px_per_mm(full, (w, w)))
    a = sim(card(w=w), full, seed=3).meta
    b = sim(card(w=w), quarter, seed=3).meta
    assert a["shake_px_per_frame"] == pytest.approx(2.0) and b["shake_px_per_frame"] == pytest.approx(0.5)
    assert a["px_scale"] == 1.0 and b["px_scale"] == 0.25
    assert np.allclose(np.array(b["shifts"]["B2"]), 0.25 * np.array(a["shifts"]["B2"]))   # same draws
    forced = sim(card(w=w), quarter, seed=3, shifts={"B2": (1.0, 0.0)}).meta              # overrides: pass px
    assert forced["shifts"]["B2"] == [1.0, 0.0]
    g, pl, roi, e = BU.toy_scene(seed=5, r=0.05)
    pl["capture"]["shake_px_per_frame"] = 1.0
    pl["camera"]["roi"] = [0, 0, 2 * roi.shape[1], 2 * roi.shape[0]]                     # a 0.5x render
    bb = BU.simulate_burst(g, pl, np.random.default_rng(7), CFG, roi, e_led=e)
    p = LP.process_burst(bb, roi, CFG)
    assert p.metrics["motion_px"] == pytest.approx(2 * p.metrics["motion_frame_px"])
    assert p.metrics["motion_frame_px"] == pytest.approx(math.hypot(*bb.meta["shifts"]["B2"]), abs=0.1)


def test_burst_meta_carries_the_illumination():
    assert sim(card(), plan()).meta["illumination"] == "pods"          # plans without it: the pods
    pl = plan()
    pl["extras"] = {"device": {"mode": "hood", "illumination": "ring"}}
    assert sim(card(), pl).meta["illumination"] == "ring"
    assert BU.illumination(pl) == "ring" and BU.illumination({}) == "pods"


def test_expose_keeps_the_mean_of_a_shifted_speckle_frame():
    """Review 15 Nov: the cubic sub-pixel shake shift rings negative next to sparse bright pixels
    (render speckle on black: light leaking under the hood). expose clipped those lobes to 0, so the
    ambient frame read +8 % and M = L - A came out 7 % dark (-7 DN). The lobes now stay as an offset."""
    import cv2
    rng = np.random.default_rng(3)
    S = np.zeros((300, 300, 3), np.float32)
    hits = rng.random((300, 300)) < 0.05
    S[hits] = rng.uniform(0.02, 0.2, (int(hits.sum()), 1)).astype(np.float32)
    shifted = LP.translate(S, 0.37, -0.21, interp=cv2.INTER_CUBIC, border=cv2.BORDER_REFLECT)
    assert shifted.min() < -1e-3                                  # the lobes exist
    means = [BU.expose(shifted, np.random.default_rng(s), SENSOR).mean() for s in range(4)]
    assert np.mean(means) == pytest.approx(float(S.mean()), rel=0.01)
    clipped = BU.expose(np.maximum(shifted, 0), np.random.default_rng(0), SENSOR).mean()
    assert clipped > 1.03 * float(S.mean())                      # what the old clipping did


def test_ring_frames_use_the_ring_gain():
    """capture.yaml sensor.by_illumination.ring: one exposure for both heads, the ring's lower irradiance
    taken up by analogue gain, so its frames carry fewer electrons per full scale (more noise)."""
    ring = LP.sensor_for(SENSOR, "ring")
    assert ring["analogue_gain"] == pytest.approx(1.5) and LP.sensor_for(SENSOR, "pods")["analogue_gain"] == 1.0
    assert LP.electrons_per_fs(SENSOR, "ring") == pytest.approx(SENSOR["full_well_e"] / 1.5)
    assert LP.electrons_per_fs(SENSOR) == pytest.approx(SENSOR["full_well_e"])
    sd = {}
    for illum in ("pods", "ring"):
        pl = plan()
        pl["extras"] = {"device": {"mode": "hood" if illum == "ring" else "open", "illumination": illum}}
        f = sim(card(), pl).frames["L"]
        sd[illum] = float(f.std())
        assert f.mean() == pytest.approx(CFG["burst"]["ref_full_scale"] / 2, abs=0.003)   # same FS level
    assert sd["ring"] / sd["pods"] == pytest.approx(math.sqrt(1.5), rel=0.08)
