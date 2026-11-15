"""Depth estimator, classical baseline and depth-aware severity (trinetra.physics)."""

import math

import numpy as np
import pytest
import yaml

from trinetra.physics import baselines as B
from trinetra.physics import depth_pinn as DP
from trinetra.physics import optics as O
from trinetra.physics import severity as SV
from trinetra.twin import frames as F
from trinetra.twin import lighting

SPEC = yaml.safe_load(open("configs/spec_steel.yaml"))


def test_dst1_matches_scipy():
    fft = pytest.importorskip("scipy.fft")
    x = np.random.default_rng(0).normal(size=(5, 9))
    assert np.allclose(B.dst1(x, 1), fft.dst(x, type=1, axis=1))
    assert np.allclose(B.dst1(x, 0), fft.dst(x, type=1, axis=0))


def test_poisson_dirichlet_recovers_height():
    s = 0.08
    yy, xx = np.mgrid[0:41, 0:47] * s
    h = -0.3 * np.sin(np.pi * xx / xx.max()) * np.sin(np.pi * yy / yy.max()) ** 2
    p = np.gradient(h, s, axis=1)
    q = np.gradient(h, s, axis=0)
    rec = B.poisson_dirichlet(p, q, s)
    assert abs(rec.min() - h.min()) < 0.02 * abs(h.min())
    assert np.allclose(rec[0], 0) and np.allclose(rec[:, -1], 0)


def _h_plate(angle_deg: float):
    """A plate -> grid homography rotated by a multiple of 90 deg, as post.grid_geometry produces."""
    s = 1 / 0.08
    a = math.radians(angle_deg)
    c, sn = round(math.cos(a)), round(math.sin(a))
    A = np.array([[c * s, -sn * s], [sn * s, c * s]])
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    H = np.eye(3)
    H[:2, :2] = A
    H[:2, 2] = 62.5 - (corners @ A.T).min(0)
    return H


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_crop_is_plate_aligned(angle):
    H = _h_plate(angle)
    ys, xs = np.mgrid[0:1375, 0:1375]
    uv = O.grid_to_plate(H, np.stack([xs.ravel(), ys.ravel()], 1)).reshape(1375, 1375, 2)
    frames = {"L": uv[..., 0], "R": uv[..., 1]}
    meta = {"H_grid_plate": H, "mm_per_px": 0.08, "plate_size_mm": [100, 100]}
    crop = DP.make_crop(frames, meta, [40, 30, 44, 33], dict(DP.DEFAULTS, max_px=400), ["L", "R"])
    nv, nu = crop.obs["L"].shape
    assert np.allclose(crop.obs["L"], crop.u0 + crop.s * np.arange(nu)[None, :], atol=1e-6)
    assert np.allclose(crop.obs["R"], crop.v0 + crop.s * np.arange(nv)[:, None], atol=1e-6)


def _synthetic_case():
    """Ring hood plan, a smooth 4 mm dent 0.25 mm deep rendered by optics itself (an inverse-crime
    smoke test: it checks the plumbing, not the accuracy on Blender data)."""
    R, t = F.camera_pose([50.0, 50.0], 200.0)
    plan = {"plate": {"size_mm": [100.0, 100.0]}, "camera": {"R": R.tolist(), "t": t.tolist()}}
    lights = lighting.ring_lights(plan, None, {"power_w": 0.18, "spot_deg": 150, "blend": 0.6})
    H = _h_plate(0)
    groups = O.group_lights(lights, ("L", "R"))
    C = F.camera_centre(R, t)
    n = 120
    u0 = 50.0 - 0.08 * n / 2
    lat = O.make_lattice(u0 - 0.02, u0 - 0.02, 2 * n, 2 * n, 0.04, 0.04, groups, C)
    U, V = lat.uv[..., 0], lat.uv[..., 1]
    r2 = ((U - 50) ** 2 + (V - 50) ** 2) / 2.0 ** 2
    h = np.where(r2 < 1, -0.25 * np.exp(-1.5 * r2) * (1 - r2) ** 2, 0.0)
    mat = O.Material(np.array([0.55]), 0.3, 1.0, 0.0, None)
    k = 0.4
    img = {g: k * a[..., 0] for g, a in O.render_height(h, lat, mat).items()}
    full = {g: np.zeros((1375, 1375)) for g in img}
    x0 = int(round((u0 + 5.0) / 0.08))
    for g in img:
        small = img[g].reshape(n, 2, n, 2).mean(axis=(1, 3))
        full[g][x0:x0 + n, x0:x0 + n] = small
    meta = {"lights": lights, "camera_centre": C.tolist(), "H_grid_plate": H, "mm_per_px": 0.08, "k": k,
            "plate_size_mm": [100, 100], "grain_deg": 0.0}
    return full, meta, [48.0, 48.0, 52.0, 52.0]


def test_estimate_depth_smoke():
    pytest.importorskip("torch")
    frames, meta, box = _synthetic_case()
    cfg = {"iters": 120, "iters_material": 30, "valid_reflection": False, "psf_sigma_mm": 0.0}
    r = DP.estimate_depth(frames, meta, box, "ms_cold_rolled", 3.0, n_seeds=2, threads=1, cfg=cfg)
    assert r.max_depth_mm is not None and math.isfinite(r.max_depth_mm)
    assert 0.12 < r.max_depth_mm < 0.38
    assert r.depth_ci_mm > 0 and len(r.members) == 2 and r.runtime_s > 0
    assert r.rel_thickness == pytest.approx(r.max_depth_mm / 3.0)
    js = r.to_json()
    assert set(js) >= {"max_depth_mm", "depth_ci_mm", "volume_mm3", "converged", "runtime_s"}


def test_classical_smoke():
    pytest.importorskip("torch")
    frames, meta, box = _synthetic_case()
    r = B.classical_depth(frames, meta, box, "ms_cold_rolled", 3.0, iters=80,
                          cfg={"iters_material": 30, "valid_reflection": False})
    assert r.max_depth_mm is not None and r.max_depth_mm >= 0 and r.method == "classical_2light"


PART = {"material": "ms_hot_rolled", "thickness_mm": 5.0, "size_mm": [100.0, 100.0]}


def _dent(d=5.0):
    return {"id": "D1", "type": "dent", "confidence": 0.9, "size": {"diameter_mm": d},
            "location_mm": {"x": 50, "y": 50}, "near_edge_mm": 30.0}


def test_today_is_hold_for_major_dent():
    from trinetra import rules
    _, j = rules.judge_steel([_dent()], SPEC, PART)
    assert j[0]["disposition"] == "HOLD_FOR_QA"


def test_shallow_dent_passes_with_measured_depth():
    j = SV.judge_with_depth(_dent(), SPEC, PART, {"max_depth_mm": 0.12, "depth_ci_mm": 0.04, "converged": True})
    assert j["depth"]["used"] and j["severity"] == "minor" and j["disposition"] == "PASS"
    assert "Table 1" in j["rule"]


def test_deep_dent_and_uncertain_cases():
    deep = SV.judge_with_depth(_dent(), SPEC, PART, {"max_depth_mm": 0.6, "depth_ci_mm": 0.05, "converged": True})
    assert deep["severity"] == "major" and deep["disposition"] == "HOLD_FOR_QA"
    crit = SV.judge_with_depth(_dent(), SPEC, PART, {"max_depth_mm": 1.5, "depth_ci_mm": 0.05, "converged": True})
    assert crit["severity"] == "critical" and crit["disposition"] == "REJECT"
    straddle = SV.judge_with_depth(_dent(), SPEC, PART, {"max_depth_mm": 0.38, "depth_ci_mm": 0.05, "converged": True})
    assert not straddle["depth"]["used"] and straddle["disposition"] == "HOLD_FOR_QA"
    wide = SV.judge_with_depth(_dent(), SPEC, PART, {"max_depth_mm": 0.1, "depth_ci_mm": 0.2, "converged": True})
    assert not wide["depth"]["used"]
    nc = SV.judge_with_depth(_dent(), SPEC, PART, {"max_depth_mm": 0.1, "depth_ci_mm": 0.02, "converged": False})
    assert not nc["depth"]["used"] and nc["disposition"] == "HOLD_FOR_QA"


def test_scratch_keeps_size_band():
    s = {"id": "S", "type": "scratch", "confidence": 0.9, "size": {"length_mm": 30.0, "width_mm": 0.3},
         "location_mm": {"x": 50, "y": 50}, "near_edge_mm": 30.0}
    j = SV.judge_with_depth(s, SPEC, PART, {"max_depth_mm": 0.05, "depth_ci_mm": 0.02, "converged": True})
    assert j["severity"] == "major" and j["disposition"] == "REWORK"


def test_depth_band_lookup():
    assert SV.depth_band(SPEC, 5.0)["imperfection_mm"] == 0.2
    assert SV.depth_band(SPEC, 10.0)["unrepaired_mm"] == 0.5
    assert SV.depth_band(SPEC, 2.0) is None
    sev, why, ref = SV.depth_severity(0.3, 2.0, 0.1, "pit", SPEC)
    assert sev == "major" and "agreement" in why
