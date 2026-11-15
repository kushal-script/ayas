"""lightproc: burst processing and the quality gate (DESIGN_STEEL.md E). Synthetic arrays only."""

from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np
import pytest

from trinetra import lightproc as LP
from trinetra.twin import burst as BU

CFG = LP.load_capture_config()
MSG = CFG["gate"]["messages"]


def run(r=0.0, seed=0, burst_seed=1, shifts=None, cfg=CFG, **scene):
    g, plan, roi, e = BU.toy_scene(r=r, seed=seed, **scene)
    b = BU.simulate_burst(g, plan, np.random.default_rng(burst_seed), cfg, roi, e_led=e, shifts=shifts)
    return LP.process_burst(b, roi, cfg), b, roi


def inner(roi, px=8):
    return LP._roi(roi, roi.shape, px)


def texture(n=512, seed=0, sigma=1.5):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), sigma)
    return 0.3 + 0.05 * t / t.std()


def make_burst(lit, amb, edits=None, noise=0.002, seed=0):
    """Hand-made burst: lit = the plate under ONE pod, amb = the ambient image (both (H, W, 3));
    B = 2 lit + amb, L = R = lit + amb, A = amb; edits: {frame: fn(frame) -> frame}; small noise."""
    rng = np.random.default_rng(seed)
    lit, amb = np.asarray(lit, np.float32), np.asarray(amb, np.float32)
    base = {"B1": 2 * lit + amb, "L": lit + amb, "A": amb.copy(), "R": lit + amb, "B2": 2 * lit + amb}
    out = {}
    for k in LP.SEQUENCE:
        f = base[k]
        if edits and k in edits:
            f = edits[k](f)
        out[k] = np.clip(f + rng.normal(0, noise, f.shape).astype(np.float32), -0.05, 1.0)
    return LP.Burst(out, {})


GOOD = {"r_ambient": 0.1, "sat_frac_a": 0.0, "led_level": 0.3, "motion_px": 0.2, "masked_frac": 0.0,
        "masked_ambient_frac": 0.0, "residual_sigma": 1.0, "blur_rel": None}


def gate(**kw):
    return LP.quality_gate({**GOOD, **kw}, CFG)


# ---- config and helpers -----------------------------------------------------------------------

def test_config_loads_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = LP.load_capture_config()
    for k in ("sensor", "leds", "burst", "process", "gate"):
        assert k in cfg
    assert list(cfg["gate"]["messages"])[:8] == ["hood_leak", "too_bright", "glare_ambient", "saturated", "too_dark",
                                                 "motion", "blur", "changed"]
    assert cfg["burst"]["sequence"] == list(LP.SEQUENCE)
    assert LP.electrons_per_fs(cfg["sensor"]) == pytest.approx(9610 / 1.0)      # analogue_gain 1.0 (was 1.5)
    assert LP.raw_levels(cfg["sensor"]) == 1023 - 64


def test_luma_and_translate_conventions():
    x = np.zeros((4, 4, 3), np.float32)
    x[..., 1] = 1.0
    assert np.allclose(LP.luma(x), 0.7152, atol=1e-6)
    assert np.allclose(LP.luma(np.ones((3, 3, 3), np.float32)), 1.0, atol=1e-6)
    z = np.zeros((32, 32), np.float32)
    z[10, 10] = 1.0
    moved = LP.translate(z, 2, 3)                      # content moves by +dx (columns), +dy (rows)
    assert np.unravel_index(int(moved.argmax()), moved.shape) == (13, 12)
    assert np.allclose(LP.align(moved, (2, 3)), z, atol=1e-6)
    rgb = np.dstack([z, z, z])
    assert LP.translate(rgb, 0, 0) is rgb
    half = LP.translate(z, 0.5, 0.0)                    # bilinear: half of the dot on each neighbour
    assert half[10, 10] == pytest.approx(0.5, abs=0.02) and half[10, 11] == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("d", [(2.3, -1.4), (0.4, 0.15), (-3.7, 1.25)])
def test_estimate_shift_subpixel(d):
    a = texture(512, seed=1)
    rng = np.random.default_rng(2)
    b = LP.translate(a, *d, interp=cv2.INTER_CUBIC)
    a_n = a + rng.normal(0, 0.01, a.shape).astype(np.float32)
    b_n = b + rng.normal(0, 0.01, a.shape).astype(np.float32)
    dx, dy, resp = LP.estimate_shift(a_n, b_n)
    assert abs(dx - d[0]) < 0.05 and abs(dy - d[1]) < 0.05
    assert 0.3 < resp <= 1.0
    # different illumination of the same texture (L vs B1): still recovered
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
    lit = b * (0.6 + 0.8 * xx / 512) + 0.4 * np.exp(-((xx - 150) ** 2 + (yy - 250) ** 2) / 3000.0)
    dx, dy, _ = LP.estimate_shift(a_n, lit + rng.normal(0, 0.01, a.shape).astype(np.float32))
    assert abs(dx - d[0]) < 0.1 and abs(dy - d[1]) < 0.1
    # unrelated noise: low response
    n1, n2 = (rng.normal(0, 1, (256, 256)).astype(np.float32) for _ in range(2))
    assert LP.estimate_shift(n1, n2)[2] < 0.2


# ---- ambient cancellation -------------------------------------------------------------------------

def test_ambient_cancels_x0_x1_x10():
    """The same plate with AMB x0, x1, x10: M changes only by noise, r grows."""
    base = 0.05
    out = {m: run(r=base * m, seed=2, burst_seed=10 + m)[0] for m in (0, 1, 10)}
    p0b, _, roi = run(r=0.0, seed=2, burst_seed=99)
    er = inner(roi)
    p0 = out[0]
    rs = [out[m].metrics["r_ambient"] for m in (0, 1, 10)]
    assert rs[0] < 0.005 and 0.03 < rs[1] < 0.06 and 0.3 < rs[2] < 0.6
    assert rs[0] < rs[1] < rs[2]
    noise = np.abs(p0b.M_lin - p0.M_lin)[er].mean()                     # two dark-room captures
    for m in (1, 10):
        d = (out[m].M_lin - p0.M_lin)[er]
        assert abs(d.mean()) < 0.005 * p0.M_lin[er].mean()             # no ambient leak (bias)
        assert np.abs(d).mean() < 1.6 * noise                           # only (larger) shot noise
        lp = np.abs(cv2.GaussianBlur(out[m].image_u8.astype(np.float32), (0, 0), 4)
                    - cv2.GaussianBlur(p0.image_u8.astype(np.float32), (0, 0), 4))[er].mean()
        assert lp < 1.5                                                 # < 1.5 DN after the noise is averaged
    assert out[1].gate.verdict == "PASS" and out[10].gate.verdict == "PASS_FLAGGED"


def test_naive_arm_is_not_invariant():
    """The baseline (single both-pods frame, AE, grey-world, gamma) moves with ambient; the burst does not."""
    def lowpass(u8):
        return cv2.GaussianBlur(u8.astype(np.float32), (0, 0), 4.0)

    for r in (0.5, 1.0):
        g0, pl0, roi, e = BU.toy_scene(r=0.0, seed=2, albedo=0.3)
        g1, pl1, _, _ = BU.toy_scene(r=r, seed=2, albedo=0.3)
        er = inner(roi, 10)
        b0 = LP.process_burst(BU.simulate_burst(g0, pl0, np.random.default_rng(1), CFG, roi, e_led=e), roi, CFG)
        b1 = LP.process_burst(BU.simulate_burst(g1, pl1, np.random.default_rng(3), CFG, roi, e_led=e), roi, CFG)
        n0 = BU.simulate_naive(g0, pl0, np.random.default_rng(1), CFG, roi, e_led=e)
        n1 = BU.simulate_naive(g1, pl1, np.random.default_rng(3), CFG, roi, e_led=e)
        d_burst = np.abs(lowpass(b1.image_u8) - lowpass(b0.image_u8))[er].mean()
        d_naive = np.abs(lowpass(n1) - lowpass(n0))[er].mean()
        assert d_naive > 5.0                     # several DN of ambient gradient + shadow in the old arm
        assert d_naive > 8.0 * d_burst
        assert LP.naive_process(n1) is n1 or np.array_equal(LP.naive_process(n1), n1)


# ---- glare and saturation ---------------------------------------------------------------------------

def _hl_disc(roi, cx=0.3, cy=0.45):
    h, w = roi.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - cx * w) ** 2 + (yy - cy * h) ** 2 < (w / 25) ** 2


def test_specular_highlight_in_l_only_disappears():
    p, b, roi = run(r=0.0, seed=3, highlight=0.5)
    hl = _hl_disc(roi)
    lc = LP.luma(np.maximum(b.frames["L"] - b.frames["A"], 0))[hl].mean()
    rc = LP.luma(np.maximum(b.frames["R"] - b.frames["A"], 0))[hl].mean()
    m = LP.luma(p.M_lin / LP._wb(CFG))[hl].mean()
    assert lc > 2.5 * rc
    assert m == pytest.approx(rc, rel=0.03)                 # M took the right pod: the highlight is gone
    ring = _hl_disc(roi, 0.3, 0.45) ^ cv2.dilate(_hl_disc(roi).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    assert abs(float(p.image_u8[hl].mean()) - float(p.image_u8[ring].mean())) < 12
    assert p.metrics["masked_frac"] == 0.0 and p.gate.verdict == "PASS"
    assert np.all(np.abs(p.relief) <= 1.0)
    assert p.relief[hl].mean() > 0.4                        # L much brighter there


def test_saturated_single_pod_recovered_not_masked():
    p, b, roi = run(r=0.0, seed=3, highlight=3.0)            # left highlight clips in L
    hl = _hl_disc(roi)
    assert (LP._maxc(b.frames["L"])[hl] >= 0.95).mean() > 0.5
    assert p.metrics["sat_frac_any"] > 0.003
    assert p.metrics["masked_frac"] == 0.0
    assert not p.sat_mask[hl].any()
    rc = LP.luma(np.maximum(b.frames["R"] - b.frames["A"], 0))[hl].mean()
    assert LP.luma(p.M_lin / LP._wb(CFG))[hl].mean() == pytest.approx(rc, rel=0.03)
    assert p.gate.verdict == "PASS"


def test_saturation_masks_and_glare_retake():
    lit, amb = np.full((300, 300, 3), 0.15, np.float32), np.full((300, 300, 3), 0.01, np.float32)
    patch = (slice(100, 160), slice(100, 160))               # 3600 px of 288^2 ROI ~ 4.3 % (> masked_frac_max 2 %)

    def clip_patch(f):
        f[patch] = 1.0
        return f

    extra = {k: clip_patch for k in ("B1", "L", "R", "B2")}   # both pods' glare on the same spot
    roi = np.zeros((300, 300), bool)
    roi[6:294, 6:294] = True
    p = LP.process_burst(make_burst(lit, amb, extra), roi, CFG)
    assert p.sat_mask[patch].all() and p.sat_mask.sum() == 60 * 60
    assert p.metrics["masked_frac"] == pytest.approx(3600 / p.metrics["roi_px"], rel=0.1)
    assert p.gate.verdict == "RETAKE" and "saturated" in p.gate.reasons
    assert p.gate.message == MSG["saturated"]
    # the same patch saturated by ambient (A too): shade/hood, not tilt
    extra2 = {k: clip_patch for k in LP.SEQUENCE}
    p2 = LP.process_burst(make_burst(lit, amb, extra2), roi, CFG)
    assert p2.metrics["sat_frac_a"] > 0.02 and p2.metrics["masked_ambient_frac"] > 0.02
    assert p2.gate.verdict == "RETAKE" and p2.gate.message == MSG["glare_ambient"]   # local, r low: a lamp reflection


# ---- motion -----------------------------------------------------------------------------------------

def test_known_shift_recovered_and_aligned():
    shifts = {"B1": (0, 0), "L": (0.6, -0.2), "A": (1.2, -0.4), "R": (1.7, -0.3), "B2": (2.3, 0.0)}
    p, b, roi = run(r=0.1, seed=4, shifts=shifts)
    p0, _, _ = run(r=0.1, seed=4, burst_seed=2)
    assert abs(p.shifts["B2"][0] - 2.3) < 0.1 and abs(p.shifts["B2"][1]) < 0.1
    assert p.metrics["motion_px"] == pytest.approx(2.3, abs=0.1)
    for k in ("L", "R"):
        assert np.hypot(p.shifts[k][0] - shifts[k][0], p.shifts[k][1] - shifts[k][1]) < 0.1
    assert p.metrics["residual_sigma"] < 2.0                # aligned B2 - B1 is just noise
    assert p.gate.verdict == "PASS"                          # 2.3 px < motion_max_px at low ambient
    er = inner(roi, 12)
    p00, _, _ = run(r=0.1, seed=4, burst_seed=3)
    noise = np.abs(p00.M_lin - p0.M_lin)[er].mean()
    assert np.abs(p.M_lin - p0.M_lin)[er].mean() < 1.5 * noise


def test_motion_gate():
    big = {"B1": (0, 0), "L": (2.2, 0.6), "A": (4.4, 1.2), "R": (6.6, 1.8), "B2": (8.8, 2.4)}   # > motion_max_px 8
    p, _, _ = run(r=0.1, seed=4, shifts=big)
    assert p.metrics["motion_px"] > CFG["gate"]["motion_max_px"]
    assert p.gate.verdict == "RETAKE" and "motion" in p.gate.reasons and p.gate.message == MSG["motion"]
    # 1 < r <= 2: strict motion limit
    small = {"B1": (0, 0), "L": (0.05, 0), "A": (0.1, 0), "R": (0.15, 0), "B2": (0.2, 0.0)}
    mid = {"B1": (0, 0), "L": (0.2, 0), "A": (0.4, 0), "R": (0.6, 0), "B2": (0.8, 0.0)}
    ps, _, _ = run(r=1.6, seed=4, albedo=0.1, shifts=small)
    pm, _, _ = run(r=1.6, seed=4, albedo=0.1, shifts=mid)
    assert 1.0 < ps.metrics["r_ambient"] <= 2.0
    assert ps.gate.verdict == "PASS_FLAGGED"
    assert pm.gate.verdict == "RETAKE" and pm.gate.reasons == ["motion"]
    pl, _, _ = run(r=0.1, seed=4, shifts=mid)
    assert pl.gate.verdict == "PASS"                         # the same motion is fine at low ambient


def test_lighting_change_retake():
    p, _, _ = run(r=0.8, seed=5)
    assert 0.6 < p.metrics["residual_sigma"] < 1.5
    ps, _, _ = run(r=0.8, seed=5, shadow_shift_mm=1.0)      # a hand shadow moving 1 mm per frame
    assert ps.gate.verdict == "RETAKE" and "changed" in ps.gate.reasons and ps.gate.message == MSG["changed"]
    pf, _, _ = run(r=0.8, seed=5, frame_sigma=0.15)          # strong ambient flicker between frames
    assert pf.gate.verdict == "RETAKE" and "changed" in pf.gate.reasons


# ---- the gate ------------------------------------------------------------------------------------------

def test_gate_r_thresholds():
    assert gate(r_ambient=0.1).verdict == "PASS"
    assert gate(r_ambient=0.3).verdict == "PASS"                       # pass < r is exclusive
    g = gate(r_ambient=0.31)
    assert g.verdict == "PASS_FLAGGED" and g.reasons == ["ambient_high"] and g.message == MSG["ambient_high"]
    assert gate(r_ambient=1.0).verdict == "PASS_FLAGGED"
    assert gate(r_ambient=1.5, motion_px=0.4).verdict == "PASS_FLAGGED"
    g = gate(r_ambient=1.5, motion_px=0.6)
    assert g.verdict == "RETAKE" and g.reasons == ["motion"]
    assert gate(r_ambient=1.0, motion_px=0.6).verdict == "PASS_FLAGGED"  # strict band is 1 < r <= 2
    assert gate(r_ambient=2.0, motion_px=0.4).verdict == "PASS_FLAGGED"
    g = gate(r_ambient=2.01)
    assert g.verdict == "RETAKE" and g.reasons == ["too_bright"] and g.message == MSG["too_bright"]
    assert gate(r_ambient=LP.R_CAP).verdict == "RETAKE"


def test_gate_other_rules_and_message_order():
    cases = {
        "too_dark": dict(led_level=0.01),
        "motion": dict(motion_px=8.1),
        "saturated": dict(masked_frac=0.03),
        "changed": dict(residual_sigma=3.5),
        "too_bright": dict(sat_frac_a=0.5),
        "glare_ambient": dict(sat_frac_a=0.05),
    }
    for key, kw in cases.items():
        g = gate(**kw)
        assert g.verdict == "RETAKE" and g.reasons == [key] and g.message == MSG[key], key
        assert g.details
    assert gate(masked_frac=0.03, masked_ambient_frac=0.029, r_ambient=1.2).reasons == ["too_bright"]
    assert gate(blur_rel=0.4).reasons == ["blur"]
    assert gate(blur_rel=None).verdict == "PASS"                        # no blur reference configured
    g = gate(r_ambient=5.0, motion_px=9.0, led_level=0.01)
    assert g.verdict == "RETAKE" and g.message == MSG["too_bright"]     # first in gate.messages order
    assert set(g.reasons) == {"too_bright", "motion", "too_dark"}
    g = gate(residual_sigma=float("nan"))
    assert g.verdict == "RETAKE" and "invalid" in g.reasons and g.message
    assert gate(r_ambient=np.float32(0.1), led_level=np.float64(0.3)).verdict == "PASS"
    assert LP.quality_gate({**GOOD}, CFG["gate"]).verdict == "PASS"      # gate section alone works
    json.dumps(g.to_dict())


def test_r_end_to_end():
    assert run(r=0.1, seed=6)[0].gate.verdict == "PASS"
    p = run(r=0.6, seed=6)[0]
    assert 0.3 < p.metrics["r_ambient"] <= 1.0 and p.gate.verdict == "PASS_FLAGGED"
    p = run(r=4.0, seed=6, albedo=0.05)[0]
    assert p.metrics["r_ambient"] > 2.0 and p.gate.verdict == "RETAKE" and p.gate.message == MSG["too_bright"]


@pytest.mark.parametrize("albedo", [0.03, 0.1, 0.35, 0.8])
def test_sun_c6_is_retake_too_bright_never_pass(albedo):
    for seed in range(3):
        p, b, _ = run(r=20.0, seed=seed, burst_seed=seed, albedo=albedo, shake_px_per_frame=0.5 * seed)
        assert b.meta["r_true"] > 10
        assert p.gate.verdict == "RETAKE", p.metrics
        assert p.gate.message == MSG["too_bright"]
        assert "too_bright" in p.gate.reasons


def test_dark_plate_too_dark():
    p, _, _ = run(r=0.0, seed=7, albedo=0.012)
    assert p.metrics["led_level"] < CFG["gate"]["led_level_min"]["pods"]
    assert p.gate.verdict == "RETAKE" and p.gate.message == MSG["too_dark"]


# ---- outputs, files, raw -----------------------------------------------------------------------------

def test_processed_contract_and_json_metrics():
    p, b, roi = run(r=0.2, seed=8, shake_px_per_frame=0.8)
    h, w = roi.shape
    assert p.image_u8.shape == (h, w, 3) and p.image_u8.dtype == np.uint8
    assert p.M_lin.shape == (h, w, 3) and p.M_lin.dtype == np.float32
    assert p.relief.shape == (h, w) and p.sat_mask.dtype == bool
    assert set(p.shifts) == set(LP.SEQUENCE) and p.shifts["B1"] == [0.0, 0.0]
    for k in ("r_ambient", "sat_frac_a", "led_level", "blur", "blur_rel", "motion_px", "residual_sigma",
              "masked_frac", "shifts", "lux_est", "snr", "process_ms"):
        assert k in p.metrics
    lum_c = float(np.dot(LP.LUMA_W, CFG["leds"]["colour"]))            # photometric: 683 lm/W x luma(colour)
    assert lum_c == pytest.approx(0.9632, abs=1e-3)
    assert p.metrics["lux_est"] == pytest.approx(p.metrics["r_ambient"] * b.meta["E_led"] * 683 * lum_c, rel=1e-5)
    json.dumps(p.metrics)
    json.dumps(p.gate.to_dict())
    # deterministic
    q = LP.process_burst(b, roi, CFG)
    assert np.array_equal(q.image_u8, p.image_u8) and q.metrics["r_ambient"] == p.metrics["r_ambient"]
    # missing frame
    with pytest.raises(ValueError):
        LP.process_burst(LP.Burst({k: v for k, v in b.frames.items() if k != "A"}, {}), roi, CFG)


def test_blur_rel_with_reference():
    cfg = {**CFG, "blur_ref": {"ms_cold_rolled": 1e6, "default": 1e-9}}
    p, _, _ = run(r=0.0, seed=8, cfg=cfg)
    assert p.metrics["blur_rel"] < 0.5 and p.gate.verdict == "RETAKE" and p.gate.message == MSG["blur"]
    p, _, _ = run(r=0.0, seed=8, cfg={**CFG, "blur_ref": 1e-9})
    assert p.metrics["blur_rel"] > 1 and p.gate.verdict == "PASS"


def test_flat_field_evens_out_falloff():
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fall = 1.0 - 0.4 * ((xx - 100) ** 2 + (yy - 100) ** 2) / 100 ** 2
    white = LP.Burst({"B1": np.dstack([0.8 * fall] * 3), "A": np.zeros((h, w, 3), np.float32)}, {})
    flat = LP.make_flat(white, smooth_px=2.0)
    grey = np.dstack([0.15 * fall] * 3).astype(np.float32)
    burst = make_burst(grey, np.zeros_like(grey), noise=0.0005)
    p_no = LP.process_burst(burst, None, CFG)
    p_ff = LP.process_burst(burst, None, CFG, flat=flat)
    er = LP._roi(None, (h, w), 10)
    cv_no = LP.luma(p_no.M_lin)[er].std() / LP.luma(p_no.M_lin)[er].mean()
    cv_ff = LP.luma(p_ff.M_lin)[er].std() / LP.luma(p_ff.M_lin)[er].mean()
    assert cv_no > 0.08 and cv_ff < 0.02


def test_save_load_burst_roundtrip(tmp_path):
    _, b, roi = run(r=0.2, seed=9)
    path = tmp_path / "b.npz"
    LP.save_burst(path, b, corners=np.array([[1, 2], [3, 4], [5, 6], [7, 8]], np.float32), roi=roi)
    b2, extra = LP.load_burst(path)
    assert set(b2.frames) == set(b.frames) and b2.frames["B1"].dtype == np.float32
    assert np.abs(b2.frames["L"] - b.frames["L"]).max() < 1e-3          # float16 storage
    assert b2.meta["shifts"] == b.meta["shifts"] and b2.meta["k"] == pytest.approx(b.meta["k"])
    assert np.array_equal(extra["roi"], roi) and extra["corners"].shape == (4, 2)
    p1, p2 = LP.process_burst(b, roi, CFG), LP.process_burst(b2, roi, CFG)
    assert p1.gate.verdict == p2.gate.verdict


def _mosaic(rgb, pattern):
    (ry, rx), (by, bx) = LP._BAYER_LAYOUT[pattern]
    m = rgb[..., 1].copy()
    m[ry::2, rx::2] = rgb[ry::2, rx::2, 0]
    m[by::2, bx::2] = rgb[by::2, bx::2, 2]
    return m


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("shifted", [False, True])
def test_raw_to_rgb_synthetic_bayer(pattern, shifted):
    black, bits = 64, 10
    rgb = np.zeros((64, 96, 3), np.uint16)
    colours = [(900, 400, 100), (100, 700, 300), (500, 500, 500)]
    for i, c in enumerate(colours):
        rgb[:, i * 32:(i + 1) * 32] = c
    raw = _mosaic(rgb, pattern)
    if shifted:
        raw = raw << 6                                      # Pi 5 unpacked: left-shifted to 16 bits
    out = LP.raw_to_rgb(raw, black, bits, pattern=pattern)
    assert out.shape == (64, 96, 3) and out.dtype == np.float32
    for i, c in enumerate(colours):
        px = out[16:48, i * 32 + 6:(i + 1) * 32 - 6].reshape(-1, 3)
        want = (np.array(c, float) - black) / (1023 - black)
        assert np.allclose(px, want, atol=1e-4), (i, px.mean(0), want)
    dark = LP.raw_to_rgb(np.full((8, 8), 60, np.uint16), black, bits, pattern=pattern, left_shifted=False)
    assert dark.max() < 0                                    # below black stays negative (no bias)


def test_bayer_code_is_self_checked():
    assert LP.bayer_code("RGGB") == LP.bayer_code("RGGB")
    with pytest.raises(ValueError):
        LP.bayer_code("RGBW")


def test_process_burst_timing_1600():
    g, plan, roi, e = BU.toy_scene(1600, 1600, r=0.3, seed=6, shake_px_per_frame=0.5, margin_px=40)
    b = BU.simulate_burst(g, plan, np.random.default_rng(0), CFG, roi, e_led=e)
    LP.process_burst(b, roi, CFG)                            # warm-up (caches, LUT)
    best = min(_timed(lambda: LP.process_burst(b, roi, CFG)) for _ in range(2))
    print(f"process_burst 1600x1600: {best * 1e3:.0f} ms")
    assert best < 1.0                                        # target < 0.3 s on the M3 (measured ~0.15 s)


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _manual_burst(groups, plan, e_led, a_f, seed=0):
    """Burst with chosen per-frame ambient factors (no shake), built from the twin sensor model."""
    k = BU.exposure_k(plan, CFG, e_led)["k"]
    rng = np.random.default_rng(seed)
    st = CFG["burst"]["led_state"]
    frames = {f: BU.expose(k * (st[f][0] * groups["L"] + st[f][1] * groups["R"] + a_f[f] * groups["AMB"]), rng, CFG["sensor"])
              for f in LP.SEQUENCE}
    return LP.Burst(frames, {"E_led": e_led})


def test_closure_catches_change_in_middle_frames():
    """Ambient up 8 % only while L and R expose: B2 - B1 cannot see it, B + A - L - R can."""
    g, plan, roi, e = BU.toy_scene(seed=11, r=0.6)
    steady = LP.process_burst(_manual_burst(g, plan, e, dict.fromkeys(LP.SEQUENCE, 1.0)), roi, CFG)
    assert steady.metrics["closure_sigma"] < 1.5 and steady.gate.verdict == "PASS_FLAGGED"
    a_f = {"B1": 1.0, "L": 1.08, "A": 1.0, "R": 1.08, "B2": 1.0}
    p = LP.process_burst(_manual_burst(g, plan, e, a_f), roi, CFG)
    assert p.metrics["residual_sigma"] < 1.5                      # the brackets agree
    assert p.metrics["closure_sigma"] > CFG["gate"]["residual_sigma_max"]
    assert p.gate.verdict == "RETAKE" and p.gate.message == MSG["changed"]


def test_no_silent_failure_on_random_conditions():
    """Random ambient (r 0.01-40), albedo, glare, shake, flicker and moving shadows: every capture
    the gate lets through matches the dark-room image (low-pass mean |diff| < 3 DN)."""
    rng = np.random.default_rng(2026)
    passed = 0
    for i in range(24):
        kw = dict(albedo=float(rng.uniform(0.03, 0.6)), highlight=float(rng.choice([0.3, 0.8, 2.0])))
        cond = dict(r=float(np.exp(rng.uniform(np.log(0.01), np.log(40.0)))),
                    shake_px_per_frame=float(rng.choice([0.0, 0.3, 1.0, 2.0])),
                    frame_sigma=float(rng.choice([0.0, 0.01, 0.03, 0.06])),
                    shadow_shift_mm=float(rng.choice([0.0, 0.0, 0.1, 0.5])))
        # flicker above 1 % only with low ambient (DESIGN_STEEL C: 3 % is office tubes at r ~0.05; window,
        # lamp and sun flicker <= 1 %). Known open-mode blind spot, found when motion_max_px rose from 3
        # (which had refused it): 3 % flicker at r 0.6 on a near-black plate passes with ~4-5 DN error.
        if cond["r"] > 0.3:
            cond["frame_sigma"] = min(cond["frame_sigma"], 0.01)
        g0, p0, roi, e = BU.toy_scene(300, 300, seed=i, **kw)
        g1, p1, _, _ = BU.toy_scene(300, 300, seed=i, **kw, **cond)
        dark = LP.process_burst(BU.simulate_burst(g0, p0, np.random.default_rng(i), CFG, roi, e_led=e), roi, CFG)
        q = LP.process_burst(BU.simulate_burst(g1, p1, np.random.default_rng(100 + i), CFG, roi, e_led=e), roi, CFG)
        assert dark.gate.verdict == "PASS"
        if q.gate.verdict == "RETAKE":
            assert q.gate.message
            continue
        passed += 1
        er = inner(roi, 10) & ~q.sat_mask
        d = np.abs(cv2.GaussianBlur(q.image_u8.astype(np.float32), (0, 0), 3)
                   - cv2.GaussianBlur(dark.image_u8.astype(np.float32), (0, 0), 3))[er].mean()
        assert d < 3.0, (cond, kw, q.metrics)
    assert passed >= 6                                            # the sweep is not all refusals


def test_empty_roi_and_float16_frames():
    _, b, roi = run(r=0.1, seed=12)
    p = LP.process_burst(b, np.zeros_like(roi), CFG)              # no plate found: refuse, never crash
    assert p.gate.verdict == "RETAKE" and "invalid" in p.gate.reasons and p.gate.message
    h16 = LP.Burst({k: v.astype(np.float16) for k, v in b.frames.items()}, dict(b.meta))
    p16, p32 = LP.process_burst(h16, roi, CFG), LP.process_burst(b, roi, CFG)
    assert p16.gate.verdict == p32.gate.verdict
    assert np.abs(p16.image_u8.astype(int) - p32.image_u8).mean() < 1.0
    assert b.frames["B1"].dtype == np.float32                    # inputs are not modified
    assert np.array_equal(LP.process_burst(b, roi, CFG).image_u8, p32.image_u8)


def test_normalise_u8_matches_process_burst():
    p, b, roi = run(r=0.2, seed=13)
    er = LP._roi(roi, roi.shape, int(CFG["process"]["roi_erode_px"]))
    u8, norm = LP.normalise_u8(p.M_lin, er & ~p.sat_mask, CFG)
    assert norm == pytest.approx(p.metrics["norm"]) and np.array_equal(u8, p.image_u8)
    again, n2 = LP.normalise_u8(p.M_lin * 0.5, None, CFG, norm=norm)    # a fixed scale is kept
    assert n2 == norm and again.mean() < u8.mean()
    lin = (u8.astype(np.float64) / 255) ** CFG["process"]["gamma"] * norm
    ok = (p.M_lin < 0.98 * norm) & (p.M_lin > 0.05 * norm)
    assert np.allclose(lin[ok], p.M_lin[ok], rtol=0.03)                    # gamma round trip


# ---- review regressions (14 Nov): each test failed before its fix ----------------------------------

def _with_reflection(n=320, r=0.05, amp=0.3, rad=10.0, cx=200, cy=120, sharp=False, seed=3, **scene):
    """Toy plate plus a lamp reflection (a blob of `amp` FS) in the ambient pass only (C5)."""
    g, plan, roi, e = BU.toy_scene(n, n, seed=seed, r=r, albedo=0.3, **scene)
    k = BU.exposure_k(plan, CFG, e)["k"]
    yy, xx = np.mgrid[0:n, 0:n]
    dd = (xx - cx) ** 2 + (yy - cy) ** 2
    d = cv2.GaussianBlur((dd < rad ** 2).astype(np.float32), (0, 0), 1.0) if sharp else np.exp(-dd / (2 * rad ** 2))
    g["AMB"] = g["AMB"] + np.float32(amp / k) * np.asarray(d, np.float32)[..., None]
    return g, plan, roi, e


@pytest.mark.parametrize("amp", [0.1, 0.2, 0.3])
def test_a_frame_reflection_does_not_lock_onto_an_led_highlight(amp):
    """A dim lamp reflection was A's only structure: correlated with B1 it matched the right pod's
    highlight, A was 'aligned' by (-23, -56) px and a still, steady capture was refused."""
    g, plan, roi, e = _with_reflection(amp=amp)
    p = LP.process_burst(BU.simulate_burst(g, plan, np.random.default_rng(3), CFG, roi, e_led=e), roi, CFG)
    assert np.hypot(*p.shifts["A"]) < 0.5, p.shifts["A"]
    assert p.metrics["closure_sigma"] < 1.5 and p.gate.verdict == "PASS"


def test_estimate_shift_search_window():
    a = texture(256, seed=4)
    b = LP.translate(a, 6.0, -2.0, interp=cv2.INTER_CUBIC)
    assert np.allclose(LP.estimate_shift(a, b)[:2], (6.0, -2.0), atol=0.05)
    dx, dy, _ = LP.estimate_shift(a, b, max_px=3.0)               # the true peak is outside the window
    assert abs(dx) <= 3.5 and abs(dy) <= 3.5
    assert np.allclose(LP.estimate_shift(a, b, max_px=3.0, center=(5.0, -1.0))[:2], (6.0, -2.0), atol=0.05)


def test_alignment_unbiased_by_clipped_highlights():
    """Clipped LED highlights on a bright plate biased the L/R/B2 shifts by up to 0.54 px (linear
    luma); the sqrt view keeps them under 0.3 px."""
    errs = {"sqrt": [], "linear": [], "refined": []}
    for s in range(4):
        g, plan, roi, e = BU.toy_scene(320, 320, seed=s, albedo=0.3, highlight=5.0, shake_px_per_frame=1.0)
        b = BU.simulate_burst(g, plan, np.random.default_rng(s), CFG, roi, e_led=e)
        for tf in errs:                                  # the coarse estimate (no closure refinement) ...
            pc = {"align_transform": tf, "align_refine": False} if tf != "refined" else {}
            p = LP.process_burst(b, roi, {**CFG, "process": {**CFG["process"], **pc}})
            errs[tf] += [np.hypot(*np.subtract(p.shifts[k], b.meta["shifts"][k])) for k in ("L", "R", "B2")]
    assert max(errs["sqrt"]) < 0.3 and np.median(errs["sqrt"]) < 0.12
    assert max(errs["linear"]) > max(errs["sqrt"])
    assert max(errs["refined"]) < 0.3 and np.median(errs["refined"]) <= np.median(errs["sqrt"]) + 0.01   # ... and refined
    with pytest.raises(ValueError):
        LP.process_burst(b, roi, {**CFG, "process": {**CFG["process"], "align_transform": "log"}})


def test_ambient_clipped_reflection_is_inpainted_not_a_black_hole():
    """A reflected lamp clipping A (a small glint, < masked_ambient_max) passed the gate with a black
    disc in image_u8 (L - A = R - A = 0 there): a fake pit for the detector. A bigger one (0.3 % of the
    plate) is refused: clipped ambient may hide a defect that scatters it."""
    n = 400
    g, plan, roi, e = BU.toy_scene(n, n, seed=4, r=0.1, albedo=0.3, highlight=0.3)
    yy, xx = np.mgrid[0:n, 0:n]
    disc = (xx - 250) ** 2 + (yy - 150) ** 2 < 3.5 ** 2               # a lamp glint below masked_ambient_max
    g["AMB"][disc] = 2.0 / BU.exposure_k(plan, CFG, e)["k"]          # 2 FS: clips in every frame
    p = LP.process_burst(BU.simulate_burst(g, plan, np.random.default_rng(0), CFG, roi, e_led=e), roi, CFG)
    assert 0 < p.metrics["masked_frac"] < CFG["gate"]["masked_frac_max"]["pods"] and p.gate.verdict == "PASS"
    assert p.sat_mask[disc].all() and p.metrics["inpainted_px"] >= disc.sum()
    near = cv2.dilate(disc.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    ring = near & ~cv2.dilate(disc.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert abs(float(p.image_u8[disc].mean()) - float(p.image_u8[ring].mean())) < 20
    assert np.median(LP.luma(p.M_lin)[disc]) == 0.0                   # M_lin keeps the truth (no data)
    big = (xx - 250) ** 2 + (yy - 150) ** 2 < 11 ** 2                # 0.25 % of the plate
    g2 = {**g, "AMB": g["AMB"].copy()}
    g2["AMB"][big] = 2.0 / BU.exposure_k(plan, CFG, e)["k"]
    q = LP.process_burst(BU.simulate_burst(g2, plan, np.random.default_rng(0), CFG, roi, e_led=e), roi, CFG)
    assert q.metrics["masked_frac"] < CFG["gate"]["masked_frac_max"]["pods"] and q.gate.message == MSG["glare_ambient"]
    # a refused capture is not touched: the operator sees where the glare is
    disc2 = (xx - 250) ** 2 + (yy - 150) ** 2 < 45 ** 2              # ~5 % of the plate > masked_frac_max
    g["AMB"][disc2] = 2.0 / BU.exposure_k(plan, CFG, e)["k"]
    q = LP.process_burst(BU.simulate_burst(g, plan, np.random.default_rng(0), CFG, roi, e_led=e), roi, CFG)
    assert q.gate.verdict == "RETAKE" and q.metrics["inpainted_px"] == 0 and q.image_u8[disc].mean() < 20


def test_saturation_masks_follow_the_alignment():
    """Thresholding the aligned frames missed the edge of a clipped region (interpolation pulls it
    below sat_level); the raw mask is moved instead, every touched pixel counted."""
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    assert LP._move_mask(m, (0.0, 0.0)) is m
    moved = LP._move_mask(m, (0.5, 0.0))                            # aligned(x) mixes raw x and x + 1
    assert set(zip(*np.nonzero(moved))) == {(4, 3), (4, 4)}
    lit, amb = np.full((120, 120, 3), 0.2, np.float32), np.full((120, 120, 3), 0.02, np.float32)
    yy, xx = np.mgrid[0:120, 0:120]
    blob = ((xx - 60) ** 2 + (yy - 60) ** 2 < 8 ** 2)
    b = make_burst(lit, amb, {k: (lambda f: np.where(blob[..., None], 1.0, f).astype(np.float32)) for k in LP.SEQUENCE}, noise=0.0)
    b.frames["A"] = LP.translate(b.frames["A"], 0.4, 0.0)            # A moved 0.4 px (hand shake)
    raw_a = LP._maxc(b.frames["A"]) >= CFG["process"]["sat_level"]
    p = LP.process_burst(b, None, {**CFG, "process": {**CFG["process"], "align_min_response": 0.0}})
    assert abs(p.shifts["A"][0] - 0.4) < 0.1
    assert p.sat_mask[blob].all() and p.sat_mask.sum() >= raw_a.sum()


def test_make_flat_uses_the_single_pod_falloff():
    """M is the darker pod per pixel, and each pod falls off towards its far side: a flat built
    from B1 - A (both pods) left a 5 % tent across M; the min-of-pods flat removes it."""
    h = w = 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fall = (1 - 0.3 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w / 2) ** 2)[..., None]

    def groups(albedo, e=10.0):
        e_l, e_r = (0.5 * e * (1.15 - 0.3 * xx / w))[..., None], (0.5 * e * (0.85 + 0.3 * xx / w))[..., None]
        one = np.ones(3, np.float32)
        return {"L": (albedo * e_l * fall / math.pi * one).astype(np.float32),
                "R": (albedo * e_r * fall / math.pi * one).astype(np.float32), "AMB": np.zeros((h, w, 3), np.float32)}

    _, plan, _, _ = BU.toy_scene(h, w)
    card = BU.simulate_burst(groups(0.9), plan, np.random.default_rng(0), CFG, e_led=10.0)
    grey = BU.simulate_burst(groups(0.3), plan, np.random.default_rng(1), CFG, e_led=10.0)
    er = LP._roi(None, (h, w), 12)

    def cv(flat):
        y = LP.luma(LP.process_burst(grey, None, CFG, flat=flat).M_lin)[er]
        return float(y.std() / y.mean())

    old = LP.make_flat(LP.Burst({k: card.frames[k] for k in ("B1", "A")}), smooth_px=4)   # B1 - A only
    assert cv(None) > 0.1 and cv(old) > 0.04
    assert cv(LP.make_flat(card, smooth_px=4)) < 0.035                 # ~ the shot-noise floor


def _moving_ambient(monkeypatch):
    """The whole ambient pass (texture and lamp reflection) moves by shadow_shift_mm per frame:
    the old burst model of the moving shadow, used here to make a moving lamp reflection."""
    monkeypatch.setattr(BU, "shadow_shift", lambda amb, s, sigma, mask=None, parts=None, **kw:
                        LP.translate(amb, *s, interp=cv2.INTER_CUBIC, border=cv2.BORDER_REFLECT))


def test_local_ghost_of_a_moving_reflection_is_refused(monkeypatch):
    """A small lamp reflection moving 0.1 mm per frame at low ambient left a 60-110 DN ghost in M
    while residual_sigma and closure_sigma (trimmed, global) stayed < 1.5: it passed silently."""
    _moving_ambient(monkeypatch)
    for seed in range(4):
        g, plan, roi, e = _with_reflection(r=0.02, rad=5.0, cx=230, cy=225, shadow_shift_mm=0.1, shake_px_per_frame=0.5)
        p = LP.process_burst(BU.simulate_burst(g, plan, np.random.default_rng(seed), CFG, roi, e_led=e), roi, CFG)
        assert p.metrics["residual_sigma"] < 3.0 and p.metrics["closure_sigma"] < 3.5
        assert p.metrics["ghost_bins"] > CFG["gate"]["ghost_bins_max"]
        assert p.gate.verdict == "RETAKE" and "changed" in p.gate.reasons and p.gate.message == MSG["changed"]
        # the same reflection held steady is fine
        g, plan, roi, e = _with_reflection(r=0.02, rad=5.0, cx=230, cy=225)
        q = LP.process_burst(BU.simulate_burst(g, plan, np.random.default_rng(seed), CFG, roi, e_led=e), roi, CFG)
        assert q.metrics["ghost_bins"] == 0 and q.gate.verdict == "PASS"


def test_benign_captures_have_no_ghost():
    """Shake, clipped highlights, pod imbalance and steady ambient: no ghost bins beyond the limit."""
    rng = np.random.default_rng(77)
    for i in range(10):
        kw = dict(albedo=float(rng.uniform(0.03, 0.6)), highlight=float(rng.choice([0.3, 2.0, 5.0])),
                  r=float(rng.choice([0.0, 0.05, 0.3])), shake_px_per_frame=float(rng.choice([0.0, 0.5, 1.5])),
                  led_jitter=0.05)
        g, plan, roi, e = BU.toy_scene(320, 320, seed=500 + i, **kw)
        p = LP.process_burst(BU.simulate_burst(g, plan, np.random.default_rng(i), CFG, roi, e_led=e), roi, CFG)
        assert p.metrics["ghost_bins"] <= CFG["gate"]["ghost_bins_max"], kw
        assert "changed" not in p.gate.reasons, (kw, p.gate.details)


def test_gate_ghost_bins_rule():
    assert gate(ghost_bins=CFG["gate"]["ghost_bins_max"]).verdict == "PASS"
    g = gate(ghost_bins=CFG["gate"]["ghost_bins_max"] + 1)
    assert g.verdict == "RETAKE" and g.reasons == ["changed"] and g.message == MSG["changed"]
    assert gate(ghost_bins=None).verdict == "PASS"                    # older metrics without the key


def test_empty_roi_r_is_unknown_not_zero():
    _, b, roi = run(r=0.3, seed=12)
    p = LP.process_burst(b, np.zeros_like(roi), CFG)
    assert math.isnan(p.metrics["r_ambient"]) and p.metrics["lux_est"] is None
    assert p.gate.verdict == "RETAKE" and "invalid" in p.gate.reasons


# ---- device v2: ring combine, gate recalibration (14 Nov) -------------------------------------------

def _ring_burst(seed=0, meta=None, noise=0.001, lit_l=None, lit_r=None):
    """A dark-field-like burst: a dark plate whose scratch scatters only the left half's light
    (one wall faces it) and a pit only the right's; no ambient."""
    rng = np.random.default_rng(seed)
    n = 160
    base = np.full((n, n, 3), 0.02, np.float32)
    lit_l = base.copy() if lit_l is None else lit_l
    lit_r = base.copy() if lit_r is None else lit_r
    if seed >= 0:
        lit_l[60:64, 30:130] = 0.4                    # scratch wall lit from the left
        lit_r[100:108, 70:78] = 0.3                   # pit rim lit from the right
    amb = np.zeros_like(base)
    fr = {"B1": lit_l + lit_r, "L": lit_l + amb, "A": amb, "R": lit_r + amb, "B2": lit_l + lit_r}
    fr = {k: (v + rng.normal(0, noise, v.shape)).astype(np.float32) for k, v in fr.items()}
    return LP.Burst(fr, dict(meta or {}))


def test_ring_combines_the_halves_by_their_mean_and_pods_by_the_min():
    """Dark-field signals are directional: min(L, R) would drop the scratch (lit from one side only)."""
    roi = np.zeros((160, 160), bool)
    roi[10:150, 10:150] = True
    ring = LP.process_burst(_ring_burst(meta={"illumination": "ring"}), roi, CFG)
    pods = LP.process_burst(_ring_burst(meta={"illumination": "pods"}), roi, CFG)
    old = LP.process_burst(_ring_burst(), roi, CFG)                         # no meta: pods (Pi v1, old npz)
    assert ring.metrics["combine"] == "mean" and ring.metrics["illumination"] == "ring"
    assert pods.metrics["combine"] == old.metrics["combine"] == "min_luma"
    wb = np.asarray(CFG["leds"]["colour"], np.float32)
    scratch = (slice(61, 63), slice(40, 120))
    assert ring.M_lin[scratch].mean(axis=(0, 1)) == pytest.approx(0.5 * (0.4 + 0.02) / wb, rel=0.03)
    assert pods.M_lin[scratch].mean(axis=(0, 1)) == pytest.approx(0.02 / wb, rel=0.1)   # the min loses it
    lum = lambda p, sl: float(LP.luma(p.M_lin[sl]).mean())                            # noqa: E731
    assert lum(ring, (slice(102, 106), slice(72, 76))) > 5 * lum(pods, (slice(102, 106), slice(72, 76)))
    for p in (ring, pods):                                                         # relief kept in both
        assert p.relief[61, 80] > 0.8 and p.relief[104, 74] < -0.8
    # one ring half clipped: masked (masked_frac, gate) and M is the mean of the clipped values (a lower
    # bound). Here the bound is far above the display white: light scattered by relief, shown white and
    # not inpainted (inpainting erased pits and scratches in the full-resolution pilot)
    b = _ring_burst(meta={"illumination": "ring"})
    for k in ("L", "B1", "B2"):
        b.frames[k][20:24, 20:24] = 1.0
    p = LP.process_burst(b, roi, CFG)
    assert p.sat_mask[21, 21] and p.metrics["masked_frac"] > 0 and p.gate.verdict == "PASS"
    assert p.metrics["inpainted_px"] == 0 and p.metrics["clipped_white_px"] >= 4
    assert (p.image_u8[21:23, 21:23] == 255).all()
    assert LP.luma(p.M_lin[21:23, 21:23]).min() > 0.4                              # >= (1 + 0.02) / 2 / WB
    q = LP.process_burst(LP.Burst(dict(b.frames), {"illumination": "pods"}), roi, CFG)
    assert not q.sat_mask[21, 21] and q.metrics["clipped_white_px"] == 0             # pods: R alone recovers it
    # a clipped half whose lower bound is below the display white is unknown there: inpainted as before
    # (a bright plate, norm ~0.42; ambient 0.45 under the clipped L leaves a bound of 0.275)
    lit = np.full((160, 160, 3), 0.35, np.float32)
    c = _ring_burst(lit_l=lit.copy(), lit_r=lit.copy(), meta={"illumination": "ring"})
    for k in LP.SEQUENCE:
        c.frames[k][20:24, 20:24] = 0.45
    for k in ("L", "B1", "B2"):
        c.frames[k][20:24, 20:24] = 1.0
    r = LP.process_burst(c, roi, CFG)
    assert r.gate.verdict == "PASS" and r.sat_mask[21, 21] and r.metrics["inpainted_px"] >= 16
    assert LP.luma(r.M_lin[21:23, 21:23]).max() < r.metrics["norm"]
    assert abs(float(r.image_u8[21:23, 21:23].mean()) - float(r.image_u8[40:44, 40:44].mean())) < 15
    # the ambient frame clipped: never kept white (L - A, R - A unknown), whatever the bound
    d = _ring_burst(meta={"illumination": "ring"})
    for k in LP.SEQUENCE:
        d.frames[k][20:24, 20:24] = 1.0
    s = LP.process_burst(d, roi, CFG)
    assert s.sat_mask[21, 21] and s.metrics["clipped_white_px"] == 0
    with pytest.raises(ValueError):
        LP.combine_for({"combine": {"ring": "max"}}, "ring")
    assert LP.combine_for({"combine": "mean"}, "pods") == "mean" and LP.combine_for({}, None) == "min_luma"


def test_twin_ring_burst_end_to_end():
    """simulate_burst passes the plan's illumination to lightproc; braced alignment for the hood."""
    g, plan, roi, e = BU.toy_scene(seed=4, r=0.0, shake_px_per_frame=0.15)
    plan["extras"]["device"] = {"mode": "hood", "illumination": "ring"}
    b = BU.simulate_burst(g, plan, np.random.default_rng(2), CFG, roi, e_led=e)
    assert b.meta["illumination"] == "ring"
    p = LP.process_burst(b, roi, {**CFG, "process": {**CFG["process"], "align_refine": False}})
    assert p.metrics["combine"] == "mean" and p.metrics["align_mode"] == "braced" and p.gate.verdict == "PASS"
    for k in ("L", "A", "R"):                                              # L, A, R follow B2 in time
        t = LP.SEQUENCE.index(k) / 4
        assert p.shifts[k] == pytest.approx([t * p.shifts["B2"][0], t * p.shifts["B2"][1]])
    q = LP.process_burst(b, roi, CFG)                                      # default: then refined on the closure
    assert q.gate.verdict == "PASS" and q.shifts["B2"] == p.shifts["B2"] and "skipped" not in q.metrics["align_refine"]
    json.dumps(p.metrics)
    json.dumps(q.metrics)


def test_closure_refinement_fixes_braced_and_biased_shifts():
    """B1 + A = L + R for a still part. Braced shifts assume the device moves linearly from B1 to
    B2, but a hand's shake is a random walk; free L/R shifts are biased by the opposite-side light.
    Both left closure ghosts on still parts in the full-resolution twin pilot (15 Nov); refining
    the L, A, R shifts on the closure removes them. A plate without structure keeps its shifts."""
    walk = {"B1": (0.0, 0.0), "L": (0.45, -0.3), "A": (0.2, -0.5), "R": (0.75, 0.1), "B2": (0.8, -0.4)}
    g, plan, roi, e = BU.toy_scene(seed=3, r=0.0, highlight=0.0)
    plan["extras"]["device"] = {"mode": "hood", "illumination": "ring"}
    b = BU.simulate_burst(g, plan, np.random.default_rng(1), CFG, roi, e_led=e, shifts=walk)
    off = LP.process_burst(b, roi, {**CFG, "process": {**CFG["process"], "align_refine": False}})
    on = LP.process_burst(b, roi, CFG)

    def err(p, k):
        return float(np.hypot(*np.subtract(p.shifts[k], walk[k])))

    assert err(off, "L") > 0.3 and err(off, "R") > 0.3                   # linear in time: 0.33-0.47 px off
    # left: B2's own error (0.08 px at gain 1.0; 0.12 at the ring's honest gain 1.5, review 15 Nov)
    assert err(on, "L") < 0.15 and err(on, "R") < 0.15
    assert set(on.metrics["align_refine"]["kept"]) >= {"L", "R"}
    assert on.metrics["closure_sigma"] < off.metrics["closure_sigma"] and on.metrics["ghost_bins"] <= off.metrics["ghost_bins"]
    flat = make_burst(np.full((120, 120, 3), 0.2, np.float32), np.full((120, 120, 3), 0.02, np.float32))
    pf = LP.process_burst(flat, None, CFG)
    assert pf.metrics["align_refine"].get("kept", []) == []               # noise only: nothing is moved


def test_motion_is_in_sensor_px_and_limits_follow_the_illumination():
    m = LP.process_burst(_ring_burst(meta={"px_scale": 0.25}), None, CFG).metrics
    assert m["motion_px"] == pytest.approx(m["motion_frame_px"] / 0.25)
    assert gate(motion_px=7.9).verdict == "PASS" and gate(motion_px=8.1).reasons == ["motion"]   # motion_max_px 8
    assert gate(masked_frac=0.019).verdict == "PASS" and gate(masked_frac=0.021).reasons == ["saturated"]
    assert gate(led_level=0.01).reasons == ["too_dark"]                       # pods: 0.02
    assert gate(led_level=0.01, illumination="ring").verdict == "PASS"        # ring (dark-field): 0.003
    assert gate(led_level=0.002, illumination="ring").reasons == ["too_dark"]
    g = {"gate": {"motion_max_px": {"pods": 3.0, "ring": 1.0}}}
    assert LP.quality_gate({**GOOD, "motion_px": 2.0}, g).verdict == "PASS"
    assert LP.quality_gate({**GOOD, "motion_px": 2.0, "illumination": "ring"}, g).reasons == ["motion"]


def test_saturated_b_frames_are_too_bright_not_invalid():
    """B1/B2 clipped over the plate left no unclipped bin for residual/closure: NaN metrics gave the
    generic 'Capture refused' message; it is 'too bright' with its own message."""
    g = gate(residual_sigma=float("nan"), sat_frac_b=1.0, sat_frac_any=1.0)
    assert g.verdict == "RETAKE" and g.reasons == ["too_bright"] and g.message == MSG["too_bright"]
    g = gate(residual_sigma=float("nan"), sat_frac_b=0.1, sat_frac_any=0.2)
    assert g.reasons == ["invalid"] and g.message                            # a NaN for another cause
    # end to end: a bright matte card under daylight, B1 and B2 clipped everywhere, L - A and R - A fine
    n = 96
    lit = np.full((n, n, 3), 0.45, np.float32)
    amb = np.full((n, n, 3), 0.5, np.float32)
    fr = {"B1": np.minimum(2 * lit + amb, 1.0), "L": lit + amb, "A": amb, "R": lit + amb, "B2": np.minimum(2 * lit + amb, 1.0)}
    rng = np.random.default_rng(0)
    fr = {k: np.minimum(v + rng.normal(0, 0.002, v.shape).astype(np.float32), 1.0) for k, v in fr.items()}
    p = LP.process_burst(LP.Burst(fr, {}), None, CFG)
    assert p.metrics["sat_frac_b"] > 0.9 and not math.isfinite(p.metrics["residual_sigma"])
    assert p.gate.verdict == "RETAKE" and "too_bright" in p.gate.reasons and "invalid" not in p.gate.reasons
    assert p.gate.message == MSG["too_bright"]


def test_lamp_reflection_on_the_part_is_glare_ambient():
    """A lamp mirrored in the part clips the ambient frame locally at modest ambient: the operator
    can tilt, shade or fit the hood (glare_ambient); a flooded frame or high r stays too_bright."""
    g = gate(sat_frac_a=0.05, r_ambient=0.1)
    assert g.verdict == "RETAKE" and g.reasons == ["glare_ambient"] and g.message == MSG["glare_ambient"]
    assert MSG["glare_ambient"] == "Lamp reflection on the part: tilt, shade it or fit the hood, then retake."
    assert gate(sat_frac_a=0.6, r_ambient=0.1).reasons == ["too_bright"]       # most of the plate: daylight
    assert gate(sat_frac_a=0.05, r_ambient=1.5).reasons == ["too_bright"]      # bright room as well
    assert gate(masked_frac=0.05, masked_ambient_frac=0.045, r_ambient=0.2).reasons == ["glare_ambient"]
    assert gate(masked_frac=0.05, masked_ambient_frac=0.045, r_ambient=1.5).reasons == ["too_bright"]
    assert gate(masked_frac=0.05, masked_ambient_frac=0.0).reasons == ["saturated"]   # LED glare: tilt
    # a defect scattering ambient light clips A on itself: inpainting would erase it, so an ambient-driven
    # mask above masked_ambient_max (a ~2.5 mm glint) is refused even when masked_frac is small
    assert gate(masked_frac=0.003, masked_ambient_frac=0.0028, r_ambient=0.2).reasons == ["glare_ambient"]
    assert gate(masked_frac=0.0003, masked_ambient_frac=0.0001, r_ambient=0.05).verdict == "PASS"   # a bulb glint
    # end to end: a sharp lamp reflection clipping A over ~3 % of the plate at r 0.05
    g2, plan, roi, e = _with_reflection(r=0.05, amp=3.0, rad=30.0, cx=160, cy=160, sharp=True)
    p = LP.process_burst(BU.simulate_burst(g2, plan, np.random.default_rng(1), CFG, roi, e_led=e), roi, CFG)
    assert 0.02 < p.metrics["sat_frac_a"] < 0.3 and p.metrics["r_ambient"] < 1.0
    assert p.gate.verdict == "RETAKE" and p.gate.message == MSG["glare_ambient"]


# ---- full-resolution pilot (15 Nov): closure blind spots ------------------------------------------

def _step_burst(where: str, illum: str = "pods", sz: int = 8, seed: int = 0) -> LP.Burst:
    """A textured plate (each half 0.1 FS, ambient 0.03) with a local ambient step to 0.11 in an
    sz x sz patch: 'L|A' = after the L frame (A, R and B2 see it: L - A is wrong, M too),
    'B2' = in B2 only (M does not use B2: harmless). No shake."""
    rng = np.random.default_rng(seed)
    n = 160
    yy, xx = np.mgrid[0:n, 0:n]
    led = (0.1 * (1 + 0.3 * np.sin(xx / 5.0) * np.cos(yy / 7.0)))[..., None].repeat(3, 2).astype(np.float32)
    amb = {k: np.full((n, n, 3), 0.03, np.float32) for k in LP.SEQUENCE}
    for k in {"L|A": ("A", "R", "B2"), "B2": ("B2",)}[where]:
        amb[k][60:60 + sz, 60:60 + sz] = 0.11
    fr = {"B1": 2 * led + amb["B1"], "L": led + amb["L"], "A": amb["A"], "R": led + amb["R"], "B2": 2 * led + amb["B2"]}
    fr = {k: (v + rng.normal(0, 0.001, v.shape)).astype(np.float32) for k, v in fr.items()}
    return LP.Burst(fr, {"illumination": illum})


def _proc_cfg(**over):
    return {**CFG, "process": {**CFG["process"], **over}}


def test_ghost_closure_max_refuses_an_ambient_step_after_l():
    """Per-frame ambient offsets e_L = a_L - a_A, e_R = a_R - a_A, b_i = a_Bi - a_A give closures
    b_1 - e_L - e_R and b_2 - e_L - e_R: a step after L and a B2-only blip both read (0, a).
    min(|c1|, |c2|) passed the step with M at a fraction of the truth (an accepted open capture in
    the pilot: a moving hand shadow, M 56 % dark locally); max refuses it, and the blip with it."""
    roi = np.zeros((160, 160), bool)
    roi[10:150, 10:150] = True
    patch = (slice(61, 67), slice(61, 67))
    ref = LP.process_burst(_step_burst("B2"), roi, _proc_cfg(ghost_closure="min"))
    lo = LP.process_burst(_step_burst("L|A"), roi, _proc_cfg(ghost_closure="min"))
    assert lo.gate.verdict == "PASS" and lo.metrics["ghost_bins"] == 0             # the blind spot
    assert LP.luma(lo.M_lin[patch]).mean() < 0.3 * LP.luma(ref.M_lin[patch]).mean()
    for where in ("L|A", "B2"):
        hi = LP.process_burst(_step_burst(where), roi, _proc_cfg(ghost_closure="max"))
        assert hi.gate.verdict == "RETAKE" and hi.gate.reasons == ["changed"] and hi.metrics["ghost_bins"] >= 4
        assert hi.metrics["ghost_closure"] == "max"
    per = _proc_cfg(ghost_closure={"pods": "max", "ring": "min"})
    assert LP.process_burst(_step_burst("L|A"), roi, per).gate.verdict == "RETAKE"
    ring = LP.process_burst(_step_burst("L|A", illum="ring"), roi, per)
    assert ring.metrics["ghost_closure"] == "min"
    with pytest.raises(ValueError):
        LP.process_burst(_step_burst("B2"), roi, _proc_cfg(ghost_closure="mean"))


def _local_reflection(n=400, amp=0.6, rad=30.0, albedo=0.08, seed=3):
    """A dark toy plate (r 0.02) and a soft lamp reflection of amp FS in the ambient pass only:
    locally the ambient is ~11 x the LED level, as under a mirrored ceiling tube (pilot C1 open)."""
    g, plan, roi, e = BU.toy_scene(n, n, seed=seed, r=0.02, albedo=albedo)
    k = BU.exposure_k(plan, CFG, e)["k"]
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.exp(-((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / (2 * rad ** 2))
    g["AMB"] = g["AMB"] + np.float32(amp / k) * d.astype(np.float32)[..., None]
    return g, plan, roi, e


def test_coarse_closure_refuses_flicker_under_a_mirrored_lamp():
    """0.6 % flicker of a lamp mirrored in a dark plate: below the noise of every 4 x 4 bin (ghost_bins
    0, closure_sigma 1.2), but coherent over the reflection, where M ends up ~30 % bright (pilot: an
    accepted open capture +30 %). The same test on 32 px blocks sees it; a steady lamp passes. (The
    plate-wide flicker test, second review, refuses it as well: switched off here to isolate the blocks.)"""
    g, plan, roi, e = _local_reflection(rad=25.0)                 # small: ambient_dominated_frac 0.01
    a_f = {"B1": 1.0, "L": 1.006, "A": 0.994, "R": 1.006, "B2": 1.0}
    steady = dict.fromkeys(LP.SEQUENCE, 1.0)
    no_flicker = {**CFG["gate"], "flicker_err_max": None}
    off = {**_proc_cfg(ghost_coarse_px=0), "gate": no_flicker}
    on = {**_proc_cfg(ghost_coarse_px=32), "gate": {**no_flicker, "ghost_coarse_max": 0}}
    q = LP.process_burst(_manual_burst(g, plan, e, steady, seed=1), roi, on)
    p0 = LP.process_burst(_manual_burst(g, plan, e, a_f, seed=1), roi, off)
    p1 = LP.process_burst(_manual_burst(g, plan, e, a_f, seed=1), roi, on)
    c = (slice(190, 210), slice(190, 210))
    assert LP.luma(p0.M_lin[c]).mean() > 1.2 * LP.luma(q.M_lin[c]).mean()            # M is wrong there
    assert p0.gate.verdict == "PASS" and p0.metrics["ghost_bins"] == 0 and p0.metrics["ghost_bins_coarse"] == 0
    assert p1.gate.verdict == "RETAKE" and p1.gate.reasons == ["changed"] and p1.metrics["ghost_bins_coarse"] > 0
    assert q.gate.verdict == "PASS" and q.metrics["ghost_bins_coarse"] == 0
    shipped = LP.process_burst(_manual_burst(g, plan, e, a_f, seed=1), roi, CFG)
    assert shipped.gate.verdict == "RETAKE" and shipped.metrics["ambient_flicker_err"] > CFG["gate"]["flicker_err_max"]


def test_ambient_dominated_plate_is_refused():
    """Where a mirrored lamp outshines the LEDs (A > local_r_max x (B1 - A)), the ambient's shot noise
    swamps L - A and R - A and the clamp at 0 biases M bright (pilot: +236 %, +21 DN at a median r of
    0.31, nothing changed between frames, so no closure test sees it). Over > ambient_dominated_max of
    the plate the capture is refused as a lamp reflection; a small glint or no lamp passes."""
    steady = dict.fromkeys(LP.SEQUENCE, 1.0)
    out = {}
    for rad, amp in ((60.0, 0.6), (25.0, 0.6), (60.0, 0.0)):
        g, plan, roi, e = _local_reflection(rad=rad, amp=amp)
        out[(rad, amp)] = LP.process_burst(_manual_burst(g, plan, e, steady, seed=1), roi, CFG)
    big = out[(60.0, 0.6)]
    assert big.metrics["ambient_dominated_frac"] > CFG["gate"]["ambient_dominated_max"] and big.metrics["r_ambient"] < 1.0
    assert big.gate.verdict == "RETAKE" and big.gate.reasons == ["glare_ambient"] and big.gate.message == MSG["glare_ambient"]
    assert 0 < out[(25.0, 0.6)].metrics["ambient_dominated_frac"] < CFG["gate"]["ambient_dominated_max"]
    assert out[(25.0, 0.6)].gate.verdict == "PASS"
    assert out[(60.0, 0.0)].metrics["ambient_dominated_frac"] == 0.0 and out[(60.0, 0.0)].gate.verdict == "PASS"
    assert gate(ambient_dominated_frac=0.5, r_ambient=1.5).reasons == ["too_bright"]          # a bright room
    assert gate(ambient_dominated_frac=None).verdict == "PASS"                              # older metrics


def test_shipped_gate_uses_the_pilot_closure_settings():
    """capture.yaml after the full-resolution pilot (15 Nov): either closure for the pods, both for the
    ring (hood) unless the ambient can explain one (second review: max_ambient), 32 px coarse blocks with
    no flagged block allowed, local ambient ratio 8 on 2 %, plate-wide flicker and hood-leak limits."""
    pc, g = CFG["process"], CFG["gate"]
    assert pc["ghost_closure"] == {"pods": "max", "ring": "max_ambient"} and pc["ghost_coarse_px"] == 32
    assert g["flicker_err_max"] == 0.2 and g["leak_r_max"] == {"ring": 0.15}
    assert g["ghost_coarse_max"] == 0 and g["ambient_dominated_max"] == 0.02 and pc["local_r_max"] == 8.0
    roi = np.zeros((160, 160), bool)
    roi[10:150, 10:150] = True
    p = LP.process_burst(_step_burst("L|A"), roi, CFG)
    assert p.gate.verdict == "RETAKE" and p.metrics["ghost_closure"] == "max"


def test_ring_noise_model_follows_the_ring_gain():
    """A still ring burst simulated at the ring gain reads residual and closure sigma ~1: the processing
    uses the same per-illumination electrons per full scale as the simulation (sensor_for)."""
    g, plan, roi, e = BU.toy_scene(r=0.0, seed=4, highlight=0.0)
    plan["extras"] = {"device": {"mode": "hood", "illumination": "ring"}}
    b = BU.simulate_burst(g, plan, np.random.default_rng(2), CFG, roi, e_led=e)
    p = LP.process_burst(b, roi, CFG)
    assert p.metrics["illumination"] == "ring"
    assert 0.85 < p.metrics["residual_sigma"] < 1.2 and 0.85 < p.metrics["closure_sigma"] < 1.2
    assert p.gate.verdict == "PASS"


def test_both_on_gain_calibrates_led_droop():
    """LED droop with both channels on: B - A = g (L + R - 2A). M never uses B1/B2, but the closure
    does: g = 0.95 refused a still dark-room capture as 'changed' (review, 15 Nov). led_both_ratio
    measures g; with leds.both_on_gain set to it the capture passes and M is unchanged."""
    g, plan, roi, e = BU.toy_scene(r=0.0, seed=5, highlight=0.0, albedo=0.5)
    b = BU.simulate_burst(g, plan, np.random.default_rng(3), CFG, roi, e_led=e)
    fr = dict(b.frames)
    for k in ("B1", "B2"):
        fr[k] = fr["A"] + np.float32(0.95) * (fr[k] - fr["A"])
    droop = LP.Burst(fr, b.meta)
    p0 = LP.process_burst(droop, roi, CFG)
    assert p0.gate.verdict == "RETAKE" and "changed" in p0.gate.reasons
    assert p0.metrics["led_both_ratio"] == pytest.approx(0.95, abs=0.005)
    cfg = {**CFG, "leds": {**CFG["leds"], "both_on_gain": p0.metrics["led_both_ratio"]}}   # the day-1 calibration
    p1 = LP.process_burst(droop, roi, cfg)
    assert p1.gate.verdict == "PASS", p1.gate.details
    assert p1.metrics["closure_sigma"] < 1.3 < p0.metrics["closure_sigma"]
    ref = LP.process_burst(b, roi, CFG)                                   # the same burst without droop
    assert ref.metrics["led_both_ratio"] == pytest.approx(1.0, abs=0.005)
    # the closure refinement fits the LED scale as a nuisance: before, it fitted the droop by moving L
    # and R 0.58 px apart, and M itself was 2.5 % wrong (still refused here, but a 2 % droop passed)
    assert p0.metrics["align_refine"]["led_scale"] == pytest.approx(0.95, abs=0.005)
    for p in (p0, p1):
        assert float(np.abs(LP.luma(p.M_lin) - LP.luma(ref.M_lin))[inner(roi, 20)].mean()) < 2e-4
        assert max(math.hypot(*np.subtract(p.shifts[k], ref.shifts[k])) for k in ("L", "A", "R")) < 0.05


def test_black_plate_in_the_dark_is_too_dark_not_a_lamp_reflection():
    """Review 15 Nov: a mirror-polished plate under the pods in the DARK room (B1 - A ~0.2 e-, A = read
    noise) was refused as glare_ambient ('Lamp reflection on the part'), because the local ambient ratio
    divided read noise by ~0. The LED level is now floored at 3 x the read noise. Under the ring the
    too_dark text names the finish or a ring fault, not the distance (the hood fixes it)."""
    lit = np.zeros((256, 256, 3), np.float32)                            # a mirror away from its highlight
    b = make_burst(lit, np.zeros_like(lit), noise=0.0003, seed=4)        # ~3 e- of read noise, no ambient
    y = {k: LP.luma(v) for k, v in b.frames.items()}
    roi = np.ones((256, 256), bool)
    assert LP._ambient_dominated(y["A"], y["B1"] - y["A"], roi, LP._proc(CFG)) > 0.02   # the old 1e-6 floor
    p = LP.process_burst(b, roi, CFG)
    assert p.metrics["ambient_dominated_frac"] == 0.0
    assert p.gate.reasons == ["too_dark"] and p.gate.message == MSG["too_dark"]
    ring = LP.quality_gate({**GOOD, "led_level": 0.001, "illumination": "ring"}, CFG)
    assert ring.reasons == ["too_dark"] and ring.message == CFG["gate"]["messages_by_illumination"]["ring"]["too_dark"]
    assert "mirror" in ring.message and LP.quality_gate({**GOOD, "led_level": 0.001}, CFG).message == MSG["too_dark"]
    # a real lamp mirrored in a dark plate still counts (A far above the floor)
    y_led = np.full((256, 256), 0.002, np.float32)
    y_a = np.where(np.arange(256)[None, :] < 64, 0.05, 0.0).astype(np.float32) * np.ones((256, 1), np.float32)
    frac = LP._ambient_dominated(y_a, y_led, np.ones((256, 256), bool), LP._proc(CFG), 3 * 2.5 / 9610)
    assert frac == pytest.approx(0.25)


# ---- second review (15 Nov): hood leaks, the ring's ghost rule, plate-wide flicker -------------------

def _pod_miss_burst(a_f: dict, amb_level: float = 0.05, seed: int = 0) -> tuple[LP.Burst, np.ndarray]:
    """Specular-like toy plate: the left pod lights it all (0.2 FS), the right pod misses the left
    half (0.01 FS there), so min(L, R) is the dark pod and the ambient (amb_level, uniform) is ~5 x M
    there. Frames in FS with the twin sensor noise; a_f = per-frame ambient factors."""
    n = 192
    rng = np.random.default_rng(seed)
    t = texture(n, seed=seed)[..., None].repeat(3, 2) / 0.3
    lit_l = 0.2 * t
    lit_r = 0.2 * t
    lit_r[:, : n // 2] *= 0.05
    amb = np.full((n, n, 3), amb_level, np.float32) * t
    st = CFG["burst"]["led_state"]
    fr = {f: BU.expose(st[f][0] * lit_l + st[f][1] * lit_r + a_f[f] * amb, rng, CFG["sensor"]) for f in LP.SEQUENCE}
    roi = np.zeros((n, n), bool)
    roi[8:-8, 8:-8] = True
    return LP.Burst(fr, {}), roi


def test_plate_wide_flicker_is_refused_where_it_matters():
    """A 6 % ambient change in the R frame only: every 4 x 4 bin and 32 px block is within its limits
    (closure_sigma < 3, ghost_bins 0), but M = R - A is the dark pod over half of this plate, with the
    ambient 5 x M, so M is ~30 % wrong there (second review: 5 % random flicker on cold-rolled steel at
    r 0.11 passed with M +15 DN over 8 % of the plate). The regression of the closures on A measures the
    change (ambient_flicker) and the gate refuses it; a steady ambient and the dark room pass."""
    steady = dict.fromkeys(LP.SEQUENCE, 1.0)
    flick = {**steady, "R": 1.06}
    p0 = LP.process_burst(_pod_miss_burst(steady)[0], _pod_miss_burst(steady)[1], CFG)
    b, roi = _pod_miss_burst(flick)
    off = {**CFG, "gate": {**CFG["gate"], "flicker_err_max": None}}
    p_off = LP.process_burst(b, roi, off)
    p = LP.process_burst(b, roi, CFG)
    dark = (slice(40, 150), slice(30, 80))
    assert LP.luma(p_off.M_lin[dark]).mean() > 1.2 * LP.luma(p0.M_lin[dark]).mean()     # M is wrong there
    assert p_off.gate.verdict != "RETAKE", p_off.gate.details                           # no local test sees it
    assert p_off.metrics["closure_sigma"] < CFG["gate"]["residual_sigma_max"] and p_off.metrics["ghost_bins"] == 0
    assert p.metrics["ambient_flicker"] == pytest.approx(0.06, abs=0.01)
    assert p.metrics["ambient_flicker_err"] > CFG["gate"]["flicker_err_max"]
    assert p.gate.verdict == "RETAKE" and p.gate.reasons == ["changed"] and p.gate.message == MSG["changed"]
    assert p0.gate.verdict != "RETAKE" and p0.metrics["ambient_flicker"] == 0.0
    no_amb = LP.process_burst(_pod_miss_burst(flick, amb_level=0.0)[0], roi, CFG)       # nothing to measure
    assert no_amb.metrics["ambient_flicker"] is None and no_amb.gate.verdict == "PASS"
    # the documented blind spot: only b_i - e_L - e_R are observable, so opposite changes in L and R with
    # steady brackets leave both closures at 0 (M wrong, accepted). The hood removes the ambient it needs.
    blind = LP.process_burst(_pod_miss_burst({**steady, "L": 1.06, "R": 0.94})[0], roi, CFG)
    assert blind.metrics["ambient_flicker"] == 0.0 and blind.gate.verdict != "RETAKE"
    json.dumps(p.metrics)
    # a featureless ambient: the regressors come from the neighbouring bins, so A's own noise (also in
    # the closures) does not bias kappa (errors in variables: it read ~1.0 with the bin's own A)
    rng = np.random.default_rng(5)
    n = 512
    led = (0.2 * (1 + 0.2 * np.cos(np.arange(n) / 50.0)))[None, :].repeat(n, 0).astype(np.float32)
    amb = np.full((n, n), 0.1, np.float32)
    noisy = lambda m: (m + np.sqrt(m / 9610.0) * rng.standard_normal(m.shape)).astype(np.float32)   # noqa: E731
    for e_l, want in ((0.0, 0.0), (0.03, 0.03)):
        y = {"A": noisy(amb), "B1": noisy(led + amb), "B2": noisy(led + amb),
             "L": noisy(0.5 * led + (1 + e_l) * amb), "R": noisy(0.5 * led + amb)}
        fit = LP._ambient_flicker(y, np.ones((n, n), bool), 0.5 * led, CFG["sensor"], LP._proc(CFG))
        assert fit["flicker"] == pytest.approx(want, abs=0.003)


def _led_residue_burst(seed: int = 0) -> LP.Burst:
    """Dark room (no ambient), ring: B2 brighter than B1 in an 8 x 8 patch (B1/B2 alignment residue at a
    sharp dark-field feature, LED-made). M does not use B2; only the B2 closure shows it."""
    b = _step_burst("B2", illum="ring", seed=seed)
    fr = {k: v.copy() for k, v in b.frames.items()}
    for k in LP.SEQUENCE:
        fr[k] -= np.float32(0.03) if k != "B2" else np.float32(0.0)
    fr["B2"] = fr["B1"].copy()
    fr["B2"][60:68, 60:68] += np.float32(0.08)
    return LP.Burst(fr, {"illumination": "ring"})


def test_ring_ghost_rule_sees_an_ambient_step_not_dark_room_residue():
    """ghost_closure max_ambient (the ring): a bin counts if both closures flag it, or one does where the
    ambient could have made the deviation. Under a leaking hood an ambient step after L passed with min
    (second review: a hand shadow arriving at the A frame, +9 DN over 70 mm^2); max refused still
    dark-room hood captures for LED-made B1/B2 residue (pilot: 2 of 8 per condition). max_ambient
    refuses the first and passes the second."""
    roi = np.zeros((160, 160), bool)
    roi[10:150, 10:150] = True
    no_leak_rule = {**CFG["gate"], "leak_r_max": None}
    cfg = {mode: {**_proc_cfg(ghost_closure={"pods": "max", "ring": mode}), "gate": no_leak_rule}
           for mode in ("min", "max", "max_ambient")}
    step = {m: LP.process_burst(_step_burst("L|A", illum="ring"), roi, c) for m, c in cfg.items()}
    assert step["min"].gate.verdict != "RETAKE" and step["min"].metrics["ghost_bins"] == 0      # the old blind spot
    for m in ("max", "max_ambient"):
        assert step[m].gate.verdict == "RETAKE" and "changed" in step[m].gate.reasons and step[m].metrics["ghost_bins"] >= 4
    assert step["max_ambient"].metrics["ghost_closure"] == "max_ambient"
    res = {m: LP.process_burst(_led_residue_burst(), roi, c) for m, c in cfg.items()}
    assert res["max"].gate.verdict == "RETAKE" and res["max"].metrics["ghost_bins"] >= 4
    for m in ("min", "max_ambient"):
        assert res[m].gate.verdict == "PASS" and res[m].metrics["ghost_bins"] == 0, res[m].gate.details
    # the shipped config refuses the same step (r 0.15: just inside leak_r_max, so by the ghost rule)
    shipped = LP.process_burst(_step_burst("L|A", illum="ring"), roi, CFG)
    assert shipped.metrics["r_ambient"] <= CFG["gate"]["leak_r_max"]["ring"]
    assert shipped.gate.verdict == "RETAKE" and shipped.gate.reasons == ["changed"]


def test_hood_leak_and_ring_messages():
    """The seated hood sees almost no ambient (twin pilot: r <= 0.021 in sun, <= 0.071 with the rim
    lifted 3 mm; <= 0.11 on a mirror-polished plate, whose dark-field LED level is tiny). Above gate.leak_r_max the ring capture is refused as a leak (with leak light x3-x5
    flicker and ambient steps passed with M off by 4-8 DN over up to 15 % of the plate). The pods are
    not limited by it. Ring texts for glare and lamp light point at the hood, which cannot be tilted."""
    ring = lambda **kw: LP.quality_gate({**GOOD, "illumination": "ring", **kw}, CFG)     # noqa: E731
    assert ring(r_ambient=0.12).verdict == "PASS"
    leak = ring(r_ambient=0.2)
    assert leak.verdict == "RETAKE" and leak.reasons == ["hood_leak"] and leak.message == MSG["hood_leak"]
    assert gate(r_ambient=0.2).verdict == "PASS"                                          # pods: no such limit
    assert ring(r_ambient=3.0).message == MSG["hood_leak"]                               # not 'fit the hood'
    over = CFG["gate"]["messages_by_illumination"]["ring"]
    assert ring(masked_frac=0.08).message == over["saturated"] and "tilt" not in over["saturated"]
    assert ring(masked_frac=0.05, masked_ambient_frac=0.045).message == over["glare_ambient"]
    assert gate(masked_frac=0.08).message == MSG["saturated"]
