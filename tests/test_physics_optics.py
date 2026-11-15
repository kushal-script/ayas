"""Forward image-formation model (trinetra.physics.optics): photometry, BRDF, numpy/torch agreement."""

import math

import numpy as np
import pytest

from trinetra.physics import optics as O
from trinetra.twin import frames as F
from trinetra.twin import lighting, textures


def _ring_plan():
    R, t = F.camera_pose([50.0, 50.0], 200.0, 0.5, -0.4, 10.0)
    K = F.nominal_K()
    plan = {"plate": {"size_mm": [100.0, 100.0], "thickness_mm": 3.0},
            "camera": {"R": R.tolist(), "t": t.tolist(), "K": K.tolist()},
            "extras": {"device": {"mode": "hood", "illumination": "ring"},
                       "hood": dict(lighting.HOOD, gap_mm=0.3)}}
    plan["lights"] = lighting.ring_lights(plan, None, {"power_w": 0.18, "spot_deg": 150, "blend": 0.6})
    return plan


def test_irradiance_matches_lighting_module():
    plan = _ring_plan()
    groups = O.group_lights(plan["lights"], ("L", "R"))
    rng = np.random.default_rng(1)
    uv = rng.uniform(5, 95, (12, 2))
    P = O.plate_points(uv)
    for g, ls in groups.items():
        l, phi = O.incident(ls, P)
        lights = [L for L in plan["lights"] if L["group"] == g]
        for k in range(len(uv)):
            n = np.array([0.2 * rng.normal(), 0.2 * rng.normal(), 1.0])
            n /= np.linalg.norm(n)
            ours = float(np.sum(phi[:, k] * np.maximum(l[:, k] @ n, 0)))
            ref = sum(lighting.light_irradiance_w(L, P[k], n) for L in lights)
            assert ours == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_spot_mask_matches_lighting():
    for c in np.linspace(-1, 1, 41):
        ref = lighting.spot_mask(float(c), 150.0, 0.6)
        ours = float(O.spot_mask(np.array([c]), math.cos(math.radians(75.0)), 0.6)[0])
        assert ours == pytest.approx(ref, abs=1e-12)


def test_normals_match_textures():
    rng = np.random.default_rng(2)
    h = rng.normal(0, 0.01, (17, 23)).astype(np.float32)
    ours = O.normals_from_height(h.astype(float), 0.04, 0.05)
    ref = textures.normal_from_height(h, (0.04, 0.05))
    assert np.allclose(ours, ref, atol=1e-5)


def test_lambert_part_is_base_over_pi():
    rng = np.random.default_rng(3)
    n = np.array([[0.0, 0.0, 1.0]])
    v = np.array([[0.05, -0.02, 1.0]]) / np.linalg.norm([0.05, -0.02, 1.0])
    l = rng.normal(size=(5, 1, 3)) * [1, 1, 0.3] + [0, 0, 0.6]
    l /= np.linalg.norm(l, axis=-1, keepdims=True)
    m1 = O.Material(np.array([0.5]), 0.6, 0.0)
    m0 = O.Material(np.array([0.0]), 0.6, 0.0)
    diff = O.brdf_cos(n, l, v, m1) - O.brdf_cos(n, l, v, m0)
    w = 1.0 - O.fresnel_dielectric(n[:, 2] * 0 + (n * v).sum(-1))
    expect = 0.5 / math.pi * np.maximum((l * n).sum(-1), 0) * w
    assert np.allclose(diff[..., 0], expect, rtol=1e-9, atol=1e-12)


def test_ggx_albedo_table_sane():
    E, Eavg = O.ggx_albedo_table()
    assert E.shape == (32, 32)
    assert E[1, -1] > 0.98                  # smooth GGX at normal incidence keeps its energy
    assert E[-1, -1] < E[8, -1]             # rough loses energy to masking
    assert np.all((Eavg > 0) & (Eavg <= 1))


def test_valid_reflection_caps_tilt():
    v = np.array([0.0, 0.0, 1.0])
    for deg in (10, 30, 44, 50, 70):
        a = math.radians(deg)
        n = np.array([[math.sin(a), 0.0, math.cos(a)]])
        m = O.valid_reflection_normals(n, v[None])
        R = 2 * (m @ v)[:, None] * m - v
        assert R[0, 2] >= 0.0099
        if deg <= 44:
            assert np.allclose(m, n)


def test_numpy_torch_agree():
    torch = pytest.importorskip("torch")
    plan = _ring_plan()
    groups = O.group_lights(plan["lights"], ("L", "R"))
    C = F.camera_centre(np.array(plan["camera"]["R"]), np.array(plan["camera"]["t"]))
    lat = O.make_lattice(40.0, 45.0, 24, 20, 0.04, 0.04, groups, C)
    yy, xx = np.mgrid[0:20, 0:24]
    h = -0.2 * np.exp(-((xx - 12) ** 2 + (yy - 10) ** 2) / 30.0)
    mat = O.Material(np.array([0.55]), 0.3, 0.8, 0.4, O.tangent_from_grain(20.0))
    a = O.render_height(h, lat, mat)
    tl = O.lattice_to(lat, dtype=torch.float64)
    mt = O.Material(torch.tensor([0.55], dtype=torch.float64), torch.tensor(0.3, dtype=torch.float64),
                    torch.tensor(0.8, dtype=torch.float64), 0.4,
                    torch.tensor(O.tangent_from_grain(20.0), dtype=torch.float64))
    b = O.render_height(torch.tensor(h, dtype=torch.float64), tl, mt)
    for g in a:
        assert np.allclose(a[g], b[g].numpy(), rtol=1e-6, atol=1e-12)
        assert np.all(a[g] >= 0)


def test_torch_gradient_finite():
    torch = pytest.importorskip("torch")
    plan = _ring_plan()
    groups = O.group_lights(plan["lights"], ("L", "R"))
    C = F.camera_centre(np.array(plan["camera"]["R"]), np.array(plan["camera"]["t"]))
    lat = O.lattice_to(O.make_lattice(48.0, 48.0, 16, 16, 0.08, 0.08, groups, C))
    h = torch.zeros(16, 16, requires_grad=True)
    mat = O.Material(torch.tensor([0.6]), torch.tensor(0.25), 1.0, 0.0, None)
    out = O.render_height(h, lat, mat)
    (out["L"].sum() - out["R"].sum()).backward()
    assert torch.isfinite(h.grad).all() and float(h.grad.abs().sum()) > 0


def test_hood_env_inside_and_positive():
    plan = _ring_plan()
    hood = plan["extras"]["hood"]
    R, t = np.array(plan["camera"]["R"]), np.array(plan["camera"]["t"])
    frustum, collar = O.hood_planes(hood, R, t)
    for L in plan["lights"]:                    # the ring LEDs sit inside the hood frustum
        p = np.asarray(L["pos_m"])
        assert all(np.dot(p - q, n) < 0 for q, n, name in frustum if name != "top_opening")
    groups = O.group_lights(plan["lights"], ("L", "R"))
    env = O.hood_env(groups, hood, R, t, O.plate_points(np.array([50.0, 50.0])), base_luma=0.6, metallic=1.0,
                     rough=0.25, plate_size_mm=(100.0, 100.0), n=24)
    for g in ("L", "R"):
        m = env.maps[g]
        assert m.shape == (24, 24) and np.all(m >= 0) and m.max() > 0
        assert env.irradiance[g] > 0
    # the L half lights the far (camera +x) side of the hood more than its own side
    xr = np.array([0.6, 0.0, -0.8])            # camera frame: +x, upwards (camera -z)
    xl = np.array([-0.6, 0.0, -0.8])
    Rw = np.asarray(R).T
    look = lambda d: float(np.ravel(O.env_lookup(env.maps["L"], (Rw @ d)[None]))[0])
    assert look(xr) > look(xl)


def test_env_lookup_numpy_torch_agree():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    m = rng.uniform(0, 1, (16, 16)).astype(np.float32)
    r = rng.uniform(-0.7, 0.7, (5, 7, 3))
    r[..., 2] = np.sqrt(1 - (r[..., :2] ** 2).sum(-1))
    a = O.env_lookup(m, r.astype(np.float32))
    b = O.env_lookup(torch.tensor(m), torch.tensor(r, dtype=torch.float32)).numpy()
    assert np.allclose(a, b, atol=2e-3)


def test_sector_partitions_nest_and_cover():
    from trinetra.physics import twin_eval as TE
    plan = _ring_plan()
    plan["lights"] = TE.per_led_lights(plan)
    parts = TE.led_partitions(plan, (2, 3, 4, 6))
    leds = sorted(L["group"] for L in plan["lights"])
    for n, part in parts.items():
        assert sorted(m for mem in part.values() for m in mem) == leds
        assert len(part) == n
    lr = parts[2]
    for mem in parts[6].values():               # 6 sectors (one per 3-LED module) nest in L / R
        assert set(mem) <= set(lr["L"]) or set(mem) <= set(lr["R"])


def test_device_meta_recovers_twin_geometry():
    """physics.device rebuilds the ring LEDs, camera and exposure from the plate corners alone."""
    from trinetra.physics import device as DV
    from trinetra.twin import burst
    plan = _ring_plan()
    K = np.array(plan["camera"]["K"])
    R, t = np.array(plan["camera"]["R"]), np.array(plan["camera"]["t"])
    corners = F.project(K, R, t, F.plate_to_world([[0, 0], [100, 0], [100, 100], [0, 100]]))
    cap = {"burst": {"ref_full_scale": 0.57, "ref_albedo": 0.9}}
    white = {"power_w": 0.18, "spot_deg": 150, "blend": 0.6}
    meta = DV.device_meta(K, corners, (100.0, 100.0), np.eye(3), 0.08, cap, {"rig": {"white": white}},
                          hood=plan["extras"]["hood"])
    assert np.allclose(meta["camera_centre"], F.camera_centre(R, t), atol=1e-6)
    pos = np.array([L["pos_m"] for L in meta["lights"]])
    ref = np.array([L["pos_m"] for L in plan["lights"]])
    assert np.allclose(pos, ref, atol=1e-5)
    assert meta["k"] == pytest.approx(burst.exposure_k(plan, cap)["k"], rel=1e-3)
