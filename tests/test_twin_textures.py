import logging
import math
import time

import cv2
import numpy as np
import pytest

from ayas.twin import textures as T

RES = 0.04
LOGGER = "ayas.twin.textures"


# ----------------------------------------------------------------------------- hand-made plans

def circle(u, v, r, n=72):
    rr = r / math.cos(math.pi / n)                      # circumscribed: contains the circle
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[u + rr * np.cos(a), v + rr * np.sin(a)].tolist()


def ellipse(u, v, A, B, ang_deg, n=72):
    k = 1 / math.cos(math.pi / n)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
    x, y = k * A * np.cos(a), k * B * np.sin(a)
    return np.c_[u + x * c - y * s, v + x * s + y * c].tolist()


def oriented_rect(u0, v0, ang_deg, s0, s1, hw):
    c, s = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
    return [[u0 + a * c - b * s, v0 + a * s + b * c] for a, b in ((s0, -hw), (s1, -hw), (s1, hw), (s0, hw))]


def bezier(ctrl, n=2000):
    P = np.asarray(ctrl, float)
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]


def buffer(pts, r):
    """Polyline buffered by r, as quads per segment plus discs at the vertices."""
    pts = np.asarray(pts, float)
    polys = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L = float(np.hypot(*d))
        if L > 0:
            polys.append(oriented_rect(a[0], a[1], math.degrees(math.atan2(d[1], d[0])), 0, L, r))
    return polys + [circle(p[0], p[1], r, 16) for p in pts]


def footprint(d, plan):
    """The natural analytic outline of each kind (what the sampler is expected to write)."""
    p, kind = d["params"], f"{d['cls']}.{d['kind']}"
    holes = {h["id"]: h for h in plan["holes"]}
    rivets = {r["id"]: r for r in plan["rivets"]}
    if kind in ("dent.punch", "dent.ball", "dent.nick"):
        return [circle(p["u"], p["v"], p["d_mm"] / 2)]
    if kind == "dent.smooth":
        return [ellipse(p["u"], p["v"], p["d_mm"] / 2, p["d_mm"] / 2 / p["aspect"], p["angle_deg"])]
    if kind in ("scratch.scriber", "scratch.gouge"):
        return buffer(bezier(p["ctrl_mm"], 200), p["width_mm"] / 2)
    if kind == "scratch.abrasion":
        c, s = math.cos(math.radians(p["angle_deg"])), math.sin(math.radians(p["angle_deg"]))
        return [oriented_rect(p["u"] - p["w_mm"] / 2 * c, p["v"] - p["w_mm"] / 2 * s, p["angle_deg"],
                              0, p["w_mm"], p["h_mm"] / 2)]
    if kind == "crack.slit":
        return [oriented_rect(p["u0"], p["v0"], p["angle_deg"], 0, p["length_mm"], p["width_mm"] / 2)]
    if kind == "crack.fatigue":
        return buffer(p["pts_mm"], p["width_mm"] / 2)
    if kind == "corrosion.pitting":
        return [circle(p["u"], p["v"], p["r_mm"])]
    if kind == "corrosion.galvanic":
        h = holes[p["hole"]]
        riv = next((r for r in plan["rivets"] if r["hole"] == h["id"]), None)
        r0 = riv["head_d_mm"] / 2 if h["filled"] and riv else h["d_mm"] / 2
        return [circle(h["u"], h["v"], r0 + p["ring_w_mm"])]
    if kind == "corrosion.fretting":
        riv = rivets[p["rivet"]]
        h = holes[riv["hole"]]
        rh, a = riv["head_d_mm"] / 2, math.radians(p["angle_deg"])
        return [oriented_rect(h["u"] + rh * math.cos(a), h["v"] + rh * math.sin(a), p["angle_deg"],
                              0, p["length_mm"], p["width_mm"] / 2)]
    raise KeyError(kind)


def make_plan(defects=(), distractors=(), size=(60.0, 40.0), seed=5, penetrant=None, finish="brushed",
              material=None, footprints=True, clean_pass=False):
    mat = {"base_color": [0.91, 0.92, 0.92], "roughness": 0.25, "anisotropy": 0.6, "aniso_rotation": 0.0,
           "grain_amp_um": 0.5, "grain_dir_deg": 0.0}
    mat.update(material or {})
    plan = {
        "schema": "ayas.twin.scene/1", "scene_id": f"t{seed}", "seed": seed, "split": "train",
        "mode": "penetrant" if penetrant else "surface", "source": "fixture",
        "plate": {"size_mm": list(size), "thickness_mm": 3.0, "edge_bevel_mm": 0.3, "finish": finish, "material": mat},
        "sheets": [],
        "holes": [{"id": "H1", "u": 8.0, "v": 8.0, "d_mm": 4.1, "ovality": 1.0, "angle_deg": 0.0, "filled": True},
                  {"id": "H2", "u": 52.0, "v": 8.0, "d_mm": 4.1, "ovality": 1.0, "angle_deg": 0.0, "filled": False},
                  {"id": "H3", "u": 52.0, "v": 32.0, "d_mm": 4.1, "ovality": 1.0, "angle_deg": 0.0, "filled": True}],
        "rivets": [{"id": "R1", "hole": "H1", "head_d_mm": 8.0, "head_h_mm": 1.3, "mandrel_d_mm": 2.2},
                   {"id": "R3", "hole": "H3", "head_d_mm": 8.0, "head_h_mm": 1.3, "mandrel_d_mm": 2.2}],
        "defects": [dict(d) for d in defects],
        "distractors": [dict(x) for x in distractors],
        "penetrant": penetrant,
        "render": {"texture_res_mm": RES, "clean_pass": clean_pass},
        "textures": {"normal": "normal.png", "albedo": "albedo.png", "roughness": "rough.png",
                     "metallic": "metal.png", "uv_emit": None},
    }
    if footprints:
        for d in plan["defects"]:
            d.setdefault("footprint", {"polys_mm": footprint(d, plan), "measure": {}})
    return plan


def defect(i, cls, kind, label=True, **params):
    return {"id": f"D{i}", "cls": cls, "kind": kind, "params": params, "label": label}


ALL_KINDS = [
    defect(1, "dent", "punch", u=20.0, v=8.0, d_mm=1.4, depth_mm=0.25, rim_h_mm=0.03),
    defect(2, "dent", "ball", u=30.0, v=8.0, d_mm=5.0, depth_mm=0.3),
    defect(3, "dent", "smooth", u=20.0, v=22.0, d_mm=10.0, depth_mm=0.5, aspect=1.6, angle_deg=30.0),
    defect(4, "dent", "nick", u=0.0, v=22.0, edge="left", d_mm=2.0, depth_mm=0.3),
    defect(5, "scratch", "scriber", ctrl_mm=[[4, 36], [14, 33], [22, 38], [32, 35]], width_mm=0.1, depth_mm=0.02, burr=0.5),
    defect(6, "scratch", "gouge", ctrl_mm=[[34, 14], [38, 17], [42, 13], [46, 16]], width_mm=0.8, depth_mm=0.2, burr=1.0),
    defect(7, "scratch", "abrasion", u=40.0, v=24.0, w_mm=8.0, h_mm=5.0, angle_deg=10.0, n_lines=40, depth_mm=0.01),
    defect(8, "crack", "fatigue", pts_mm=[[30, 18], [31, 18.4], [32, 17.8], [33, 18.3]], width_mm=0.03),
    defect(9, "corrosion", "galvanic", hole="H1", ring_w_mm=2.0, powder=0.7),
    defect(10, "crack", "slit", u0=54.05, v0=8.0, angle_deg=0.0, length_mm=4.0, width_mm=0.4),
    defect(11, "corrosion", "pitting", u=10.0, v=30.0, r_mm=3.0, n_pits=8, pit_d_mm=[0.1, 0.6], powder=0.5),
    defect(12, "corrosion", "fretting", rivet="R3", angle_deg=180.0, length_mm=8.0, width_mm=3.0, darkness=0.7),
    defect(13, "dent", "punch", label=False, u=56.0, v=37.0, d_mm=1.0, depth_mm=0.2),
]

DISTRACTORS = [
    {"kind": "fingerprint", "params": {"u": 30.0, "v": 28.0, "r_mm": 5.0, "angle_deg": 20.0}},
    {"kind": "oil", "params": {"u": 20.0, "v": 22.0, "r_mm": 3.0}},          # on top of the smooth dent
    {"kind": "water_spot", "params": {"u": 45.0, "v": 5.0, "r_mm": 2.0}},
    {"kind": "marker", "params": {"u": 12.0, "v": 16.0, "length_mm": 4.0, "width_mm": 0.8, "angle_deg": 45.0}},
    {"kind": "dust", "params": {"n": 80}},
    {"kind": "blend", "params": {"u": 46.0, "v": 34.0, "r_mm": 3.0}},
    {"kind": "burr", "params": {"hole": "H2", "height_mm": 0.05, "width_mm": 0.3}},
    {"kind": "glove", "params": {}},
]


def mask_of(polys, shape, dilate_mm=0.0):
    m = np.zeros(shape, np.uint8)
    pts = [np.round((np.asarray(p, float) / RES - 0.5) * 16).astype(np.int32) for p in polys]
    for q in pts:                       # one by one: a single fillPoly call XORs overlaps
        cv2.fillPoly(m, [q], 1, lineType=cv2.LINE_8, shift=4)
    if dilate_mm:
        m = (cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= dilate_mm / RES).astype(np.uint8)
    return m.astype(bool)


def changed(a, b):
    diff = np.zeros(a["height"].shape, bool)
    for k in a:
        d = a[k] != b[k]
        diff |= d.any(axis=-1) if d.ndim == 3 else d
    return diff


def texel_centres(mask):
    r, c = np.nonzero(mask)
    return np.c_[(c + 0.5) * RES, (r + 0.5) * RES]


def min_dist(q, pts, chunk=400):
    out = np.empty(len(q))
    for i in range(0, len(q), chunk):
        d = q[i: i + chunk, None, :] - pts[None, :, :]
        out[i: i + chunk] = np.sqrt((d ** 2).sum(-1)).min(1)
    return out


# ----------------------------------------------------------------------------- tests

def test_shapes_dtypes_and_ranges():
    plan = make_plan(ALL_KINDS, DISTRACTORS, size=(30.0, 20.0), footprints=False)
    plan["defects"] = [d for d in plan["defects"] if d["cls"] == "dent" and d["params"].get("u", 0) < 30]
    m = T.render_maps(plan)
    assert T.map_shape(plan) == (500, 750)
    assert set(m) == {"height", "albedo", "roughness", "metallic", "uv_emit"}
    assert m["height"].shape == (500, 750) and m["albedo"].shape == (500, 750, 3)
    for k, a in m.items():
        assert a.dtype == np.float32 and np.isfinite(a).all(), k
        if k != "height":
            assert a.min() >= 0 and a.max() <= 1, k
    assert m["uv_emit"].max() == 0
    assert T.map_shape(make_plan(size=(100.0, 100.0))) == (2500, 2500)


def test_normal_from_height_signs_and_flat():
    n = T.normal_from_height(np.zeros((4, 5), np.float32), RES)
    assert n.dtype == np.float32 and n.shape == (4, 5, 3) and np.allclose(n, [0, 0, 1])
    u = (np.arange(6) + 0.5) * RES
    k = 0.2
    nu = T.normal_from_height(np.tile(k * u, (5, 1)), RES)        # rising towards +u: tilts to -X
    assert np.allclose(nu, np.array([-k, 0, 1]) / math.hypot(k, 1), atol=1e-6)
    nv = T.normal_from_height(np.tile(k * u[:5, None], (1, 6)), RES)   # rising towards +v (= -Y): tilts to +Y
    assert np.allclose(nv, np.array([0, k, 1]) / math.hypot(k, 1), atol=1e-6)


def test_flat_plate_writes_neutral_normal_png(tmp_path):
    plan = make_plan(size=(8.0, 6.0), material={"grain_amp_um": 0.0, "waviness_um": 0.0})
    paths = T.write_maps(plan, tmp_path)
    img = cv2.imread(str(paths["normal"]), cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint16 and img.shape == (150, 200, 3)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert (rgb[..., 0] == 32768).all() and (rgb[..., 1] == 32768).all() and (rgb[..., 2] == 65535).all()


def test_written_files_formats_and_channel_order(tmp_path):
    plan = make_plan(size=(10.0, 8.0), material={"base_color": [0.8, 0.4, 0.1], "roughness": 0.3})
    paths = T.write_maps(plan, tmp_path)
    assert set(paths) == {"normal", "albedo", "roughness", "metallic"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["albedo.png", "metal.png", "normal.png", "rough.png"]
    alb = cv2.imread(str(paths["albedo"]), cv2.IMREAD_UNCHANGED)
    assert alb.dtype == np.uint8 and alb.shape == (200, 250, 3)
    lin = T.srgb8_to_linear(cv2.cvtColor(alb, cv2.COLOR_BGR2RGB)).reshape(-1, 3).mean(0)
    assert np.allclose(lin, [0.8, 0.4, 0.1], rtol=0.04)
    rough = cv2.imread(str(paths["roughness"]), cv2.IMREAD_UNCHANGED)
    metal = cv2.imread(str(paths["metallic"]), cv2.IMREAD_UNCHANGED)
    assert rough.shape == metal.shape == (200, 250) and rough.dtype == np.uint8
    assert abs(rough.mean() / 255 - 0.3) < 0.03 and (metal == 255).all()


def test_srgb_encoding_round_trip():
    assert T.linear_to_srgb8(np.array([0.0, 0.18, 0.5, 1.0])).tolist() == [0, 118, 188, 255]
    assert T.linear_to_srgb8(np.array([0.002])).tolist() == [round(0.002 * 12.92 * 255)]   # linear toe
    codes = np.arange(256, dtype=np.uint8)
    assert (T.linear_to_srgb8(T.srgb8_to_linear(codes)) == codes).all()
    x = np.linspace(0, 1, 1001)
    assert np.abs(T.srgb8_to_linear(T.linear_to_srgb8(x)) - x).max() < 0.005


def test_punch_dent_position_depth_and_extent():
    u, v, d = 7.313, 11.627, 1.4
    plan = make_plan([defect(1, "dent", "punch", u=u, v=v, d_mm=d, depth_mm=0.25, rim_h_mm=0.03)],
                     size=(20.0, 20.0), footprints=False)
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    dh = m["height"] - c["height"]
    for h in (dh, m["height"]):
        r, col = np.unravel_index(np.argmin(h), h.shape)
        assert abs(r - (v / RES - 0.5)) <= 1 and abs(col - (u / RES - 0.5)) <= 1
    assert dh.min() == pytest.approx(-0.25, rel=0.02)
    assert dh.max() == pytest.approx(0.03, rel=0.1)                     # rim
    q = texel_centres(changed(m, c))
    assert np.hypot(q[:, 0] - u, q[:, 1] - v).max() <= d / 2 + T.rim_width_mm(d) + 1e-9


def test_ball_and_smooth_dents_stay_inside_their_outlines():
    plan = make_plan([defect(1, "dent", "ball", u=10.0, v=10.0, d_mm=6.0, depth_mm=0.4),
                      defect(2, "dent", "smooth", u=30.0, v=10.0, d_mm=12.0, depth_mm=0.8, aspect=2.0, angle_deg=30.0)],
                     size=(40.0, 20.0), footprints=False)
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    dh = m["height"] - c["height"]
    q = texel_centres(changed(m, c))
    ball, smooth = q[q[:, 0] < 20], q[q[:, 0] >= 20]
    assert np.hypot(ball[:, 0] - 10, ball[:, 1] - 10).max() <= 3.0
    a = math.radians(30)
    x, y = smooth[:, 0] - 30, smooth[:, 1] - 10
    xs, ys = x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)
    assert ((xs / 6) ** 2 + (ys / 3) ** 2).max() <= 1.0
    assert dh[:, :500].min() == pytest.approx(-0.4, rel=0.02)
    assert dh[:, 500:].min() == pytest.approx(-0.8, rel=0.02)
    assert dh.max() <= 1e-6                                             # no pile-up for these kinds


@pytest.mark.parametrize("kind,width,depth", [("scriber", 0.05, 0.03), ("scriber", 0.3, 0.06), ("gouge", 1.0, 0.3)])
def test_scratch_pixels_lie_near_the_bezier(kind, width, depth):
    ctrl = [[3.0, 15.0], [12.0, 4.0], [22.0, 26.0], [37.0, 12.0]]
    plan = make_plan([defect(1, "scratch", kind, ctrl_mm=ctrl, width_mm=width, depth_mm=depth, burr=1.0)],
                     size=(40.0, 30.0), footprints=False)
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    q = texel_centres(changed(m, c))
    dist = min_dist(q, bezier(ctrl, 6000))
    assert len(q) > 500
    assert dist.max() <= width / 2 + 0.1
    assert (m["height"] - c["height"]).min() < -0.5 * depth            # a real groove, not just noise


def test_clean_maps_differ_only_inside_footprints(caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    plan = make_plan(ALL_KINDS, DISTRACTORS)
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    shape = m["height"].shape
    labelled = [d for d in plan["defects"] if d["label"]]
    allowed = mask_of([p for d in labelled for p in d["footprint"]["polys_mm"]], shape, 0.3)
    diff = changed(m, c)
    assert diff.any() and not (diff & ~allowed).any()
    for d in labelled:                                                  # every kind really rendered
        assert (diff & mask_of(d["footprint"]["polys_mm"], shape)).sum() > 3, d["id"]
    # the unlabelled defect is in both, and the textures stayed inside the natural outlines
    unl = mask_of(plan["defects"][-1]["footprint"]["polys_mm"], shape)
    assert m["height"][unl].min() < -0.15 and np.array_equal(m["height"][unl], c["height"][unl])
    assert not [r for r in caplog.records if r.name == LOGGER], [r.getMessage() for r in caplog.records]


def test_footprint_clips_and_warns(caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    d = defect(1, "dent", "punch", u=10.0, v=10.0, d_mm=2.0, depth_mm=0.2)
    d["footprint"] = {"polys_mm": [circle(10.0, 10.0, 0.3)], "measure": {}}
    plan = make_plan([d], size=(20.0, 20.0))
    q = texel_centres(changed(T.render_maps(plan), T.render_maps(plan, clean=True)))
    assert np.hypot(q[:, 0] - 10, q[:, 1] - 10).max() <= 0.3 + T.CLIP_TOL_MM + 1.5 * RES   # raster slack
    assert any("outside its footprint" in r.getMessage() for r in caplog.records)


def test_unknown_defect_kind_raises():
    plan = make_plan([defect(1, "dent", "laser", u=5.0, v=5.0)], size=(10.0, 10.0), footprints=False)
    with pytest.raises(ValueError, match="dent.laser"):
        T.render_maps(plan)


def test_determinism_and_seed():
    plan = make_plan(ALL_KINDS, DISTRACTORS)
    a, b = T.render_maps(plan), T.render_maps(plan)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    other = make_plan(ALL_KINDS, DISTRACTORS, seed=6)
    assert not np.array_equal(a["height"], T.render_maps(other)["height"])


@pytest.mark.parametrize("finish", ["brushed", "mill", "clad"])
def test_finishes_and_grain_direction(finish):
    gu, gv = [], []
    for ang in (0.0, 90.0):
        plan = make_plan(size=(20.0, 20.0), finish=finish, material={"grain_dir_deg": ang, "waviness_um": 0.0})
        h = T.render_maps(plan)["height"]
        assert h.std() == pytest.approx(0.5e-3, rel=0.05)                # grain_amp_um is the RMS
        gu.append(np.abs(np.diff(h, axis=1)).mean())
        gv.append(np.abs(np.diff(h, axis=0)).mean())
    ratio = 1.02 if finish == "clad" else 1.5                            # clad is mostly isotropic mottling
    assert gu[0] * ratio < gv[0] and gv[1] * ratio < gu[1]               # smooth along the grain


def test_penetrant_indications_developer_and_edge_bleed():
    pen = {"developer": {"albedo": 0.88, "roughness": 0.9, "speckle": 0.05, "thickness_var": 0.1},
           "indications": [{"id": "I1", "cls": "linear_indication", "from_defect": "D1",
                            "params": {"pts_mm": [[5, 10], [15, 10.5]], "bloom_mm": 0.8, "intensity": 0.9}},
                           {"id": "I2", "cls": "rounded_indication", "params": {"u": 22.0, "v": 6.0, "d_mm": 1.0}}],
           "edge_bleed": 0.5}
    pen["indications"][0]["footprint"] = {"polys_mm": buffer([[5, 10], [15, 10.5]], 0.85)}   # 0.05 + bloom
    pen["indications"][1]["footprint"] = {"polys_mm": [circle(22.0, 6.0, 0.5)]}
    plan = make_plan([defect(1, "dent", "punch", u=22.0, v=14.0, d_mm=1.5, depth_mm=0.25)], size=(30.0, 20.0),
                     penetrant=pen)
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)

    def at(maps, key, u, v):
        return maps[key][int(v / RES), int(u / RES)]

    for u, v in ((10.0, 10.25), (22.0, 6.0)):
        r, g, b = at(m, "albedo", u, v)
        assert r > g + 0.25 and g < 0.35 and b < 0.35                   # red dye
        assert min(at(c, "albedo", u, v)) > 0.7                         # clean: plain developer
    far = np.zeros(m["height"].shape, bool)
    far[150:-150, 150:-150] = True                                      # 6 mm in from the edge bleed
    far[int(9 / RES): int(12 / RES), : int(17 / RES)] = False           # around the indications
    far[: int(8 / RES), int(20 / RES): int(24 / RES)] = False
    dev = m["albedo"][far]
    assert dev.min() > 0.7 and dev.max() <= 0.97 + 1e-6
    assert np.abs(dev[:, 0] - dev[:, 1]).max() < 0.02                   # neutral white, no red
    assert (m["metallic"] == 0).all() and abs(float(np.median(m["roughness"])) - 0.9) < 0.03
    assert at(m, "height", 22.0, 14.0) < -0.2                           # dent kept under the coat
    band = np.zeros(m["height"].shape, bool)
    band[:12], band[-12:], band[:, :12], band[:, -12:] = True, True, True, True
    redness = m["albedo"][..., 0] - m["albedo"][..., 1]
    assert (redness[band] > 0.15).mean() > 0.05                          # bleed-out at the edges...
    assert np.array_equal(m["albedo"][band], c["albedo"][band])          # ...which is not a label


def test_uv_emit_and_clean_pass_files(tmp_path):
    dis = [{"kind": "oil", "params": {"u": 5.0, "v": 5.0, "r_mm": 2.0, "uv_emit": 0.6}}]
    plan = make_plan([defect(1, "dent", "punch", u=8.0, v=4.0, d_mm=1.0, depth_mm=0.2)], dis, size=(12.0, 8.0),
                     clean_pass=True)
    paths = T.write_maps(plan, tmp_path)
    assert set(paths) == {f"{k}{s}" for k in ("normal", "albedo", "roughness", "metallic", "uv_emit") for s in ("", "_clean")}
    assert all(p.exists() for p in paths.values())
    assert paths["uv_emit"].name == "uvemit.png" and paths["normal_clean"].name == "normal_clean.png"
    uv = cv2.imread(str(paths["uv_emit"]), cv2.IMREAD_UNCHANGED)
    assert uv.max() == round(0.6 * 255) and uv[100, 100] == 0
    assert np.array_equal(uv, cv2.imread(str(paths["uv_emit_clean"]), cv2.IMREAD_UNCHANGED))
    n, nc = (cv2.imread(str(paths[k]), cv2.IMREAD_UNCHANGED) for k in ("normal", "normal_clean"))
    diff = (n != nc).any(-1)
    assert diff.any() and texel_centres(diff)[:, 0].min() > 8.0 - 0.5 - 0.2


def test_full_plate_timing():
    holes = [{"id": "H1", "u": 20.0, "v": 15.0, "d_mm": 4.1, "filled": True}]
    plan = make_plan([
        defect(1, "dent", "punch", u=62.0, v=70.0, d_mm=1.4, depth_mm=0.25, rim_h_mm=0.03),
        defect(2, "dent", "smooth", u=40.0, v=60.0, d_mm=40.0, depth_mm=1.5, aspect=1.5, angle_deg=30.0),
        defect(3, "scratch", "scriber", ctrl_mm=[[5, 95], [30, 60], [60, 99], [95, 75]], width_mm=0.2, depth_mm=0.05, burr=0.5),
        defect(4, "scratch", "abrasion", u=70.0, v=40.0, w_mm=15.0, h_mm=15.0, angle_deg=15.0, n_lines=200, depth_mm=0.01),
        defect(5, "corrosion", "pitting", u=80.0, v=80.0, r_mm=8.0, n_pits=30, pit_d_mm=[0.1, 1.0], powder=0.8),
        defect(6, "corrosion", "galvanic", hole="H1", ring_w_mm=4.0, powder=0.7),
    ], [{"kind": "fingerprint", "params": {"u": 30.0, "v": 40.0, "r_mm": 7.0, "angle_deg": 20.0}},
        {"kind": "oil", "params": {"u": 50.0, "v": 20.0, "r_mm": 6.0}},
        {"kind": "dust", "params": {"n": 300}},
        {"kind": "blend", "params": {"u": 80.0, "v": 30.0, "r_mm": 8.0}}],
        size=(100.0, 100.0), footprints=False, material={"grain_dir_deg": 17.0})
    plan["holes"], plan["rivets"] = holes, [{"id": "R1", "hole": "H1", "head_d_mm": 8.0}]
    t0 = time.perf_counter()
    m = T.render_maps(plan)
    dt = time.perf_counter() - t0
    print(f"render_maps 100x100 mm @ {RES} mm, 6 defects + 4 distractors: {dt:.2f} s")
    assert m["height"].shape == (2500, 2500)
    assert dt < 15.0


# ----------------------------------------------------------------------------- review regressions

def _directional_roughness(h, ang_deg):
    gv, gu = np.gradient(h)
    c, s = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
    return np.abs(gu * c + gv * s).mean()


@pytest.mark.parametrize("finish", ["brushed", "mill"])
def test_grain_direction_sign(finish):
    """grain_dir_deg runs from +u toward +v: 30 and 150 deg are mirror images (0/90 cannot tell)."""
    for ang, other in ((30.0, 150.0), (150.0, 30.0)):
        plan = make_plan(size=(20.0, 12.0), finish=finish, material={"grain_dir_deg": ang, "waviness_um": 0.0})
        h = T.render_maps(plan)["height"]
        assert 2 * _directional_roughness(h, ang) < _directional_roughness(h, other)


def test_normal_png_axes_end_to_end(tmp_path):
    """A dent's +u wall tilts the normal to -X (R < 0.5) and its +v wall to +Y (G > 0.5) in normal.png."""
    plan = make_plan([defect(1, "dent", "ball", u=6.0, v=4.0, d_mm=4.0, depth_mm=0.4)], size=(12.0, 8.0),
                     material={"grain_amp_um": 0.0, "waviness_um": 0.0}, footprints=False)
    rgb = cv2.cvtColor(cv2.imread(str(T.write_maps(plan, tmp_path)["normal"]), cv2.IMREAD_UNCHANGED),
                       cv2.COLOR_BGR2RGB).astype(float) / 65535
    r0, c0 = int(4.0 / RES), int(6.0 / RES)
    off = int(1.2 / RES)
    assert rgb[r0, c0 + off, 0] < 0.45 and rgb[r0, c0 - off, 0] > 0.55      # +u wall rises toward +u
    assert rgb[r0 + off, c0, 1] > 0.55 and rgb[r0 - off, c0, 1] < 0.45      # +v wall rises toward +v = -Y
    assert abs(rgb[r0, c0 + off, 1] - 0.5) < 0.02 and abs(rgb[r0 + off, c0, 0] - 0.5) < 0.02


GEOM_KINDS = [   # non-trivial angles so that a u/v or sign flip leaves the footprint
    defect(1, "dent", "punch", u=20.0, v=8.0, d_mm=2.0, depth_mm=0.3, rim_h_mm=0.05),
    defect(2, "dent", "ball", u=30.0, v=8.0, d_mm=5.0, depth_mm=0.3),
    defect(3, "dent", "smooth", u=20.0, v=22.0, d_mm=10.0, depth_mm=0.5, aspect=2.0, angle_deg=120.0),
    defect(4, "dent", "nick", u=30.0, v=40.0, edge="bottom", d_mm=2.5, depth_mm=0.6),
    defect(5, "scratch", "scriber", ctrl_mm=[[4, 36], [10, 31], [18, 39], [26, 33]], width_mm=0.3, depth_mm=0.05, burr=0.2),
    defect(6, "scratch", "gouge", ctrl_mm=[[33, 30], [36, 34], [40, 33], [44, 38]], width_mm=1.5, depth_mm=0.3, burr=0.0),
    defect(7, "scratch", "abrasion", u=41.0, v=20.0, w_mm=12.0, h_mm=3.0, angle_deg=60.0, n_lines=60, depth_mm=0.01),
    defect(8, "crack", "fatigue", pts_mm=[[30, 14], [30.4, 14.5], [30.6, 15.1], [31.1, 15.5]], width_mm=0.04),
    defect(9, "corrosion", "galvanic", hole="H1", ring_w_mm=2.0, powder=0.7),
    defect(10, "crack", "slit", u0=52.0 + 2.04 * math.cos(math.radians(120)), v0=8.0 + 2.04 * math.sin(math.radians(120)),
           angle_deg=120.0, length_mm=4.0, width_mm=0.5),
    defect(11, "corrosion", "pitting", u=10.0, v=24.0, r_mm=3.0, n_pits=3, pit_d_mm=[0.3, 0.8], powder=0.5,
           pits=[[9.0, 23.0, 0.8], [11.5, 24.5, 0.6], [10.0, 26.0, 0.4]]),
    defect(12, "corrosion", "fretting", rivet="R3", angle_deg=250.0, length_mm=8.0, width_mm=3.0, darkness=0.7),
]


def test_kinds_stay_inside_geometry_footprints(caplog):
    """Contract with the sampler's footprints (geometry.footprint): every kind renders inside them,
    nothing is clipped (same maps without footprints) and no overflow warning is logged."""
    G = pytest.importorskip("ayas.twin.geometry")
    caplog.set_level(logging.WARNING, logger=LOGGER)
    plan = make_plan(GEOM_KINDS, footprints=False)
    for d in plan["defects"]:
        d["footprint"] = G.footprint(d, plan)
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    shape = m["height"].shape
    diff = changed(m, c)
    allowed = mask_of([p for d in plan["defects"] for p in d["footprint"]["polys_mm"]], shape, 0.2)
    assert diff.any() and not (diff & ~allowed).any()
    for d in plan["defects"]:
        assert (diff & mask_of(d["footprint"]["polys_mm"], shape)).sum() > 3, d["id"]
    free = make_plan(GEOM_KINDS, footprints=False)                          # no footprints: no clipping
    assert all(np.array_equal(m[k], T.render_maps(free)[k]) for k in m)
    assert not [r for r in caplog.records if r.name == LOGGER], [r.getMessage() for r in caplog.records]


def test_pitting_uses_the_plan_pit_list():
    """Pits sit where the sampler put them (penetrant rounded indications are drawn there)."""
    pits = [[9.0, 23.0, 0.8], [11.5, 24.5, 0.6], [10.0, 26.0, 0.4]]
    plan = make_plan([defect(1, "corrosion", "pitting", u=10.0, v=24.0, r_mm=3.0, n_pits=3, pit_d_mm=[0.3, 0.8],
                             powder=0.0, pits=pits)], size=(20.0, 40.0), footprints=False)
    dh = T.render_maps(plan)["height"] - T.render_maps(plan, clean=True)["height"]
    for u, v, d in pits:
        assert dh[int(v / RES), int(u / RES)] < -0.03
    q = texel_centres(dh < -0.02)
    assert len(q) and min_dist(q, np.asarray(pits)[:, :2]).max() <= 0.4 + RES   # no pits elsewhere


def test_indications_fill_their_footprint_and_use_dye_color():
    """Linear: dye over width/2 + bloom (the footprint's half-width); rounded: about d_mm across."""
    blue = [0.05, 0.1, 0.6]
    pen = {"developer": {"albedo": 0.88}, "dye_color": blue, "edge_bleed": 0.0,
           "indications": [{"id": "I1", "cls": "linear_indication",
                            "params": {"pts_mm": [[4.0, 10.0], [16.0, 10.0]], "width_mm": 0.4, "bloom_mm": 1.0,
                                       "intensity": 1.0},
                            "footprint": {"polys_mm": buffer([[4.0, 10.0], [16.0, 10.0]], 0.2 + 1.0)}},
                           {"id": "I2", "cls": "rounded_indication", "params": {"u": 22.0, "v": 5.0, "d_mm": 2.0,
                                                                                "intensity": 1.0},
                            "footprint": {"polys_mm": [circle(22.0, 5.0, 1.0)]}}]}
    plan = make_plan(size=(30.0, 20.0), penetrant=pen)
    alb = T.render_maps(plan)["albedo"]
    blueness = alb[..., 2] - alb[..., 0]

    def at(u, v):
        return blueness[int(v / RES), int(u / RES)]

    assert at(10.0, 10.0) > 0.25 and at(10.0, 10.0 + 0.2 + 0.5) > 0.1        # core, half the bloom out (+-40 % along)
    assert abs(at(10.0, 10.0 + 1.3)) < 0.02                                   # beyond the footprint
    assert at(22.0, 5.0) > 0.4 and at(22.0 + 0.7, 5.0) > 0.15 and abs(at(22.0 + 1.1, 5.0)) < 0.02
    assert (alb[..., 0] - alb[..., 2] < 0.05).all()                           # no default red anywhere


def test_distractors_honour_sampler_params():
    """Keys the sampler writes: burr h_mm/w_mm/start_deg/arc_deg (oval hole), strength, ring_w_mm, size_mm."""
    flat = {"grain_amp_um": 0.0, "waviness_um": 0.0}

    def render(dis, **kw):
        plan = make_plan(distractors=dis, size=(20.0, 20.0), material=flat, **kw)
        plan["holes"] = [{"id": "H1", "u": 10.0, "v": 10.0, "d_mm": 4.0, "ovality": 1.3, "angle_deg": 30.0,
                          "filled": False}]
        plan["rivets"] = []
        return T.render_maps(plan)

    h = render([{"kind": "burr", "params": {"hole": "H1", "start_deg": 0.0, "arc_deg": 90.0, "h_mm": 0.2,
                                            "w_mm": 0.3}}])["height"]
    assert h.max() == pytest.approx(0.2, rel=0.1)
    q = texel_centres(h > 0.01) - 10.0
    ang = np.degrees(np.arctan2(q[:, 1], q[:, 0])) % 360
    assert ang.min() >= 0.0 and ang.max() <= 90.0                             # only the arc, +u toward +v
    a, b, al = 2.6, 2.0, math.radians(30)
    xs, ys = q[:, 0] * math.cos(al) + q[:, 1] * math.sin(al), -q[:, 0] * math.sin(al) + q[:, 1] * math.cos(al)
    rho = np.hypot(xs / a, ys / b)
    assert rho.min() >= 1.0 and rho.max() < 1.2                               # just outside the hole ellipse

    base = render([])
    for kind, p in (("oil", {"r_mm": 3.0}), ("water_spot", {"r_mm": 2.0, "ring_w_mm": 0.3}), ("marker", {"r_mm": 1.5})):
        m = render([{"kind": kind, "params": {"u": 5.0, "v": 5.0, "strength": 0.0, **p}}])
        assert all(np.array_equal(m[k], base[k]) for k in m), kind               # strength 0: invisible

    def spot_area(w):
        alb = render([{"kind": "water_spot", "params": {"u": 5.0, "v": 5.0, "r_mm": 2.0, "ring_w_mm": w}}])["albedo"]
        return (np.abs(alb - base["albedo"]).max(-1) > 0.02).sum()

    assert spot_area(0.4) > 2.5 * spot_area(0.1)

    def dust_area(size):
        alb = render([{"kind": "dust", "params": {"u": 5.0, "v": 15.0, "r_mm": 4.0, "n": 40, "size_mm": size}}])["albedo"]
        return (np.abs(alb - base["albedo"]).max(-1) > 0.05).sum()

    assert dust_area([0.15, 0.2]) > 5 * dust_area([0.02, 0.03])


def test_galvanic_round_open_oval_hole_matches_visible_outline(caplog):
    """An open hole's visible radius is d·ovality/2 (geometry.visible_radius): no clip, no warning."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    d = defect(1, "corrosion", "galvanic", hole="H2", ring_w_mm=1.5, powder=0.8)
    plan = make_plan([d], footprints=False)
    plan["holes"][1].update(ovality=1.3, angle_deg=45.0)
    ri = 4.1 * 1.3 / 2
    d = plan["defects"][0]
    d["footprint"] = {"polys_mm": [circle(52.0, 8.0, ri + 1.5)], "measure": {}}
    q = texel_centres(changed(T.render_maps(plan), T.render_maps(plan, clean=True)))
    r = np.hypot(q[:, 0] - 52.0, q[:, 1] - 8.0)
    assert len(q) > 100 and r.min() >= ri - RES and r.max() <= ri + 1.5 + T.CLIP_TOL_MM
    assert not [x for x in caplog.records if x.name == LOGGER]


def test_nick_centre_snaps_to_its_edge():
    plan = make_plan([defect(1, "dent", "nick", u=0.4, v=20.0, edge="left", d_mm=2.0, depth_mm=0.4)],
                     footprints=False)
    dh = T.render_maps(plan)["height"] - T.render_maps(plan, clean=True)["height"]
    r, c = np.unravel_index(np.argmin(dh), dh.shape)
    assert c == 0 and abs(r - (20.0 / RES - 0.5)) <= 1                      # deepest at the plate edge
    q = texel_centres(dh != 0)
    assert np.hypot(q[:, 0], q[:, 1] - 20.0).max() <= 1.0 + T.rim_width_mm(2.0) + 1e-9


# ----------------------------------------------------------------------------- steel and metal (DESIGN_STEEL A, B)

STEEL_MATERIALS = {   # kind: (finish, material extras) as the sampler writes them
    "ms_hot_rolled": ("hot_rolled", {"base_color": [0.12, 0.125, 0.135], "roughness": 0.6, "metallic": 0.5,
                                     "anisotropy": 0.05, "grain_amp_um": 1.5,
                                     "scale": {"mottle": 0.2, "cell_mm": 4.0, "flake_frac": 0.2, "spall_frac": 0.01}}),
    "ms_cold_rolled": ("cold_rolled", {"base_color": [0.5, 0.505, 0.515], "roughness": 0.25, "metallic": 1.0,
                                       "anisotropy": 0.3, "grain_amp_um": 0.4}),
    "ss_brushed": ("brushed", {"base_color": [0.6, 0.595, 0.585], "roughness": 0.2, "metallic": 1.0,
                               "anisotropy": 0.6, "grain_amp_um": 0.6}),
    "ss_2b": ("2b", {"base_color": [0.55, 0.548, 0.54], "roughness": 0.32, "metallic": 1.0, "anisotropy": 0.1,
                     "grain_amp_um": 0.3}),
    "galvanised": ("spangle", {"base_color": [0.62, 0.625, 0.632], "roughness": 0.3, "metallic": 1.0,
                               "anisotropy": 0.1, "grain_amp_um": 0.4, "spangle": {"cell_mm": 3.0, "contrast": 0.06}}),
    "aluminium": ("brushed", {"base_color": [0.91, 0.92, 0.92], "roughness": 0.25, "metallic": 1.0}),
}


def steel_plan(kind, defects=(), distractors=(), size=(40.0, 30.0), seed=5, with_footprints=True, **extra):
    finish, mat = STEEL_MATERIALS[kind]
    mat = {"kind": kind, "aniso_rotation": 0.0, "grain_dir_deg": 0.0, **mat, **extra}
    plan = make_plan(defects, distractors, size=size, seed=seed, finish=finish, material=mat, footprints=False)
    plan["holes"], plan["rivets"] = [], []
    if with_footprints:
        G = pytest.importorskip("ayas.twin.geometry")
        for d in plan["defects"]:
            d["footprint"] = G.footprint(d, plan)
    return plan


STEEL_KINDS = [
    defect(1, "pit", "single", u=5.0, v=5.0, d_mm=1.0, depth_mm=0.3, halo_mm=0.6, halo=0.9),
    defect(2, "pit", "cluster", u=16.0, v=6.0, r_mm=4.0, n=5, pit_d_mm=[0.3, 0.9], depth_mm=0.25, halo_mm=0.3,
           halo=0.8, pits=[[14.5, 4.5, 0.6], [17.0, 7.0, 0.9], [18.5, 4.0, 0.3], [13.5, 8.0, 0.5], [16.0, 9.0, 0.4]]),
    defect(3, "corrosion", "rust", u=30.0, v=8.0, r_mm=6.0, coverage=0.6, pitted=True, aspect=1.4, angle_deg=20.0),
    defect(4, "corrosion", "rust_streak", u0=0.0, v0=16.0, angle_deg=10.0, length_mm=14.0, width_mm=3.0,
           source="edge", edge="left", fade=0.6),
    defect(5, "corrosion", "rust_streak", u0=34.0, v0=16.5, angle_deg=100.0, length_mm=11.0, width_mm=2.0,
           source="spot", head_r_mm=1.5, fade=0.5),
    defect(6, "crack", "open", pts_mm=[[17.0, 14.0], [18.0, 14.4], [19.1, 13.9], [20.2, 14.6], [21.0, 14.3]],
           width_mm=0.2, start="free"),
    defect(7, "corrosion", "white", u=24.0, v=22.0, r_mm=3.0, n_pits=6, pit_d_mm=[0.2, 0.6], powder=0.7),
    defect(8, "stain", "oil", u=8.0, v=24.0, r_mm=4.0, aspect=1.3, angle_deg=30.0, gloss=0.9, strength=0.9),
    defect(9, "stain", "water", u=16.0, v=24.0, r_mm=3.0, ring_w_mm=0.3, strength=0.9, minerals=0.8),
    defect(10, "stain", "fingerprint", u=30.0, v=24.0, r_mm=5.0, angle_deg=20.0, aspect=1.4, strength=0.9),
    defect(11, "crack", "open", pts_mm=[[0.0, 28.5], [3.0, 28.2], [6.0, 28.8]], width_mm=0.06, start="edge"),
]
STEEL_DISTRACTORS = [
    {"kind": "label_residue", "params": {"u": 12.0, "v": 15.0, "w_mm": 6.0, "h_mm": 3.0, "angle_deg": 15.0,
                                         "strength": 0.8}},
    {"kind": "marker", "params": {"u": 38.0, "v": 28.0, "r_mm": 1.0}},
    {"kind": "dust", "params": {"n": 60}},
]


@pytest.mark.parametrize("kind", sorted(STEEL_MATERIALS))
def test_steel_kinds_stay_inside_their_footprints(kind, caplog):
    """Every steel kind (stains included: labelled now) renders on every material, only inside the
    geometry footprints dilated by 0.3 mm, with no overflow warning."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    plan = steel_plan(kind, STEEL_KINDS, STEEL_DISTRACTORS)
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    shape = m["height"].shape
    for k, a in m.items():
        assert a.dtype == np.float32 and np.isfinite(a).all(), k
        if k != "height":
            assert a.min() >= 0 and a.max() <= 1, k
    diff = changed(m, c)
    allowed = mask_of([p for d in plan["defects"] for p in d["footprint"]["polys_mm"]], shape, 0.3)
    assert diff.any() and not (diff & ~allowed).any()
    for d in plan["defects"]:
        assert (diff & mask_of(d["footprint"]["polys_mm"], shape)).sum() > 3, d["id"]
    assert not [r for r in caplog.records if r.name == LOGGER], [r.getMessage() for r in caplog.records]
    a, b = T.render_maps(plan), T.render_maps(steel_plan(kind, STEEL_KINDS, STEEL_DISTRACTORS))
    assert all(np.array_equal(a[k], b[k]) for k in a)                      # deterministic


@pytest.mark.parametrize("kind", sorted(STEEL_MATERIALS))
def test_steel_bases_look_like_their_material(kind):
    finish, mat = STEEL_MATERIALS[kind]
    m = T.render_maps(steel_plan(kind, size=(30.0, 30.0)))
    alb = m["albedo"].reshape(-1, 3)
    grey = float(np.mean(mat["base_color"]))
    assert abs(float(alb.mean()) - grey) < 0.15 * grey + 0.01
    assert abs(float(np.median(m["roughness"])) - mat["roughness"]) < 0.06
    assert abs(float(m["metallic"].mean()) - mat["metallic"]) < 0.1
    if kind == "ms_hot_rolled":                                            # mottled blue-grey semi-metallic scale
        assert alb[:, 2].mean() > alb[:, 0].mean()
        assert m["metallic"].std() > 0.05 and 0.08 * 0.6 < float(np.median(alb.mean(1))) < 0.18 * 1.2
        lum = alb.mean(1)
        assert lum.std() / lum.mean() > 0.1
    elif kind == "galvanised":                                             # spangle: piecewise-constant cells
        lum = m["albedo"].mean(-1)
        step = np.abs(np.diff(lum, axis=1))
        assert (step > 0.005).mean() < 0.05 and lum.std() / lum.mean() > 0.02
    else:
        assert m["metallic"].min() > 0.99


def test_rust_is_red_rough_and_not_metallic():
    plan = steel_plan("ms_cold_rolled", [defect(1, "corrosion", "rust", u=15.0, v=15.0, r_mm=10.0, coverage=0.7,
                                                pitted=False, aspect=1.0, angle_deg=0.0)], size=(30.0, 30.0))
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    rust = m["metallic"] < 0.1
    frac = rust.sum() / (math.pi * 10 * 10 / RES ** 2)
    assert 0.45 < frac < 0.8                                               # about the coverage of the outline
    a = m["albedo"][rust]
    base = c["albedo"][rust]
    assert (a[:, 0] / a[:, 1]).mean() > 2.0 and (base[:, 0] / base[:, 1]).mean() < 1.1
    assert 0.25 * 0.9 <= a[:, 0].mean() <= 0.45 * 1.1 and 0.08 * 0.9 <= a[:, 1].mean() <= 0.18 * 1.1
    assert a[:, 2].mean() <= 0.08 * 1.1
    assert m["roughness"][rust].mean() > 0.7
    assert (m["height"] - c["height"])[rust].mean() > 0.005                 # a raised crust


def test_rust_on_mill_scale_and_pit_halo_is_redder():
    plan = steel_plan("ms_hot_rolled", [defect(1, "pit", "single", u=10.0, v=10.0, d_mm=1.0, depth_mm=0.4,
                                               halo_mm=0.8, halo=1.0)], size=(20.0, 20.0))
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    U, V = np.meshgrid((np.arange(500) + 0.5) * RES, (np.arange(500) + 0.5) * RES)
    r = np.hypot(U - 10.0, V - 10.0)
    ring = (r > 0.55) & (r < 0.85)
    assert (m["albedo"][ring][:, 0] / m["albedo"][ring][:, 1]).mean() > 1.5 > \
        (c["albedo"][ring][:, 0] / c["albedo"][ring][:, 1]).mean()
    dh = m["height"] - c["height"]
    assert dh.min() == pytest.approx(-0.4, rel=0.05) and r[np.unravel_index(np.argmin(dh), dh.shape)] < 0.1
    assert m["albedo"][r < 0.15].mean() < 0.5 * c["albedo"][r < 0.15].mean()          # a dark cavity
    nohalo = steel_plan("ss_2b", [defect(1, "pit", "single", u=10.0, v=10.0, d_mm=1.0, depth_mm=0.4, halo_mm=0.0)],
                        size=(20.0, 20.0))
    q = texel_centres(changed(T.render_maps(nohalo), T.render_maps(nohalo, clean=True)))
    assert np.hypot(q[:, 0] - 10, q[:, 1] - 10).max() <= 0.5 + 0.1


def test_oil_lowers_roughness_and_darkens_slightly():
    plan = steel_plan("ms_hot_rolled", [defect(1, "stain", "oil", u=10.0, v=10.0, r_mm=6.0, aspect=1.0, angle_deg=0.0,
                                               gloss=0.8, strength=1.0)], size=(20.0, 20.0))
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    U, V = np.meshgrid((np.arange(500) + 0.5) * RES, (np.arange(500) + 0.5) * RES)
    core = np.hypot(U - 10.0, V - 10.0) < 4.0
    assert m["roughness"][core].mean() < 0.65 * c["roughness"][core].mean()
    assert (m["roughness"] <= c["roughness"] + 1e-6).all()                 # never rougher
    ratio = m["albedo"][core].mean() / c["albedo"][core].mean()
    assert 0.85 < ratio < 0.99                                              # slightly darker, still metal
    assert np.array_equal(m["metallic"], c["metallic"])


def test_water_ring_and_fingerprint_stains():
    plan = steel_plan("ss_brushed", [defect(1, "stain", "water", u=8.0, v=10.0, r_mm=5.0, ring_w_mm=0.3,
                                            strength=1.0, minerals=0.5),
                                     defect(2, "stain", "fingerprint", u=25.0, v=10.0, r_mm=6.0, angle_deg=0.0,
                                            aspect=1.4, strength=1.0)], size=(35.0, 20.0))
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    U, V = np.meshgrid((np.arange(875) + 0.5) * RES, (np.arange(500) + 0.5) * RES)
    r = np.hypot(U - 8.0, V - 10.0)
    rim, mid = (r > 4.3) & (r < 4.9), r < 3.0
    d = m["roughness"] - c["roughness"]
    assert d[rim].mean() > 0.1 > d[mid].mean()                              # a ring, not a disc
    fp = (np.hypot((U - 25.0) / 6.0, (V - 10.0) * 1.4 / 6.0) < 0.6)
    assert np.abs(m["roughness"] - c["roughness"])[fp].std() > 0.02           # ridges


def test_open_crack_is_dark_with_raised_lips_and_through_scale_scratch_is_bright():
    pts = [[3.0, 10.0], [6.0, 10.4], [9.0, 9.8], [12.0, 10.3], [15.0, 10.0]]
    plan = steel_plan("ss_2b", [defect(1, "crack", "open", pts_mm=pts, width_mm=0.3, start="free")], size=(20.0, 20.0))
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    G = pytest.importorskip("ayas.twin.geometry")
    q = texel_centres(changed(m, c))
    dist = min_dist(q, bezier_free_polyline(pts))
    assert dist.max() <= 0.15 + G.crack_lip_mm(0.3) + 0.06
    at = (int(10.4 / RES), int(6.0 / RES))
    assert m["albedo"][at].mean() < 0.35 * c["albedo"][at].mean()
    dh = m["height"] - c["height"]
    assert dh.min() < -0.2 and 0.005 < dh.max() < 0.03
    hot = steel_plan("ms_hot_rolled", [defect(1, "scratch", "gouge", ctrl_mm=[[3, 10], [8, 10], [12, 10], [17, 10]],
                                              width_mm=0.8, depth_mm=0.2, burr=0.0)], size=(20.0, 20.0))
    m, c = T.render_maps(hot), T.render_maps(hot, clean=True)
    line = (slice(int(9.9 / RES), int(10.1 / RES)), slice(int(6 / RES), int(14 / RES)))
    assert m["albedo"][line].mean() > 2 * c["albedo"][line].mean() and m["metallic"][line].mean() > 0.9


def bezier_free_polyline(pts, step=0.005):
    """Dense samples along a polyline."""
    out = []
    for a, b in zip(np.asarray(pts[:-1], float), np.asarray(pts[1:], float)):
        n = max(2, int(np.hypot(*(b - a)) / step))
        out.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    return np.vstack(out)


def test_stains_are_labelled_and_label_residue_is_not():
    plan = steel_plan("ss_brushed", [defect(1, "stain", "oil", u=10.0, v=10.0, r_mm=4.0, aspect=1.0, angle_deg=0.0,
                                            gloss=0.8, strength=1.0)],
                      [{"kind": "label_residue", "params": {"u": 25.0, "v": 10.0, "w_mm": 8.0, "h_mm": 5.0,
                                                            "angle_deg": 10.0, "strength": 1.0}}], size=(35.0, 20.0))
    m, c = T.render_maps(plan), T.render_maps(plan, clean=True)
    diff = changed(m, c)
    q = texel_centres(diff)
    assert len(q) and q[:, 0].max() < 14.5                                  # only the oil left the clean maps
    residue = (slice(int(8.5 / RES), int(11.5 / RES)), slice(int(22 / RES), int(28 / RES)))
    assert np.array_equal(m["albedo"][residue], c["albedo"][residue])
    base = steel_plan("ss_brushed", size=(35.0, 20.0))
    assert (np.abs(c["albedo"] - T.render_maps(base)["albedo"]).max(-1)[residue] > 0.01).mean() > 0.3


def test_mill_scale_keeps_the_base_hue_outside_flakes():
    """Only the flakes are tinted bluer: the whole face used to be scaled by [0.97, 1, 1.06], so its
    hue drifted from plate.material.base_color (which Blender gives the edges and hole walls)."""
    plan = steel_plan("ms_hot_rolled", size=(30.0, 30.0),
                      scale={"mottle": 0.0, "cell_mm": 4.0, "flake_frac": 0.01, "spall_frac": 0.0})
    alb = T.render_maps(plan)["albedo"].reshape(-1, 3).mean(0)
    base = np.asarray(plan["plate"]["material"]["base_color"])
    assert (alb[2] / alb[0]) / (base[2] / base[0]) == pytest.approx(1.0, abs=0.01)
    assert alb.mean() == pytest.approx(base.mean(), rel=0.03)
    flaky = steel_plan("ms_hot_rolled", size=(30.0, 30.0),
                       scale={"mottle": 0.0, "cell_mm": 4.0, "flake_frac": 0.35, "spall_frac": 0.0})
    a2 = T.render_maps(flaky)["albedo"].reshape(-1, 3)
    lum = a2.mean(1)
    dark = lum < np.quantile(lum, 0.2)                                    # the flakes: darker and bluer
    assert (a2[dark, 2] / a2[dark, 0]).mean() > 1.04 * (a2[~dark, 2] / a2[~dark, 0]).mean()


@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 120.0])
def test_label_residue_stays_inside_the_sampler_cover_outline(angle):
    """The sampler keeps label residue (it paints over what is below) off labelled defects using
    sampler._cover_outline; the texture must never reach past that outline."""
    S = pytest.importorskip("ayas.twin.sampler")
    p = {"u": 20.0, "v": 15.0, "w_mm": 24.0, "h_mm": 10.0, "angle_deg": angle, "strength": 1.0}
    plan = steel_plan("ss_2b", distractors=[{"kind": "label_residue", "params": p}], size=(40.0, 30.0))
    diff = changed(T.render_maps(plan), T.render_maps(steel_plan("ss_2b", size=(40.0, 30.0))))
    allowed = mask_of([S._cover_outline("label_residue", p)], diff.shape, 0.05)
    assert diff.sum() > 0.5 * 24 * 10 / RES ** 2 and not (diff & ~allowed).any()


@pytest.mark.parametrize("kind", sorted(STEEL_MATERIALS))
def test_blend_grinds_through_mill_scale_and_zinc(kind):
    """Review fix: a blend (grind) mark used to keep the dark semi-metallic mill scale and the zinc
    spangle under its sanded dish. Grinding removes them: bright metallic steel on hot-rolled,
    plain uniform steel grey on galvanised; other finishes keep their own metal."""
    blend = [{"kind": "blend", "params": {"u": 15.0, "v": 15.0, "d_mm": 12.0, "aspect": 1.0, "angle_deg": 0.0,
                                          "depth_mm": 0.3}}]
    m = T.render_maps(steel_plan(kind, distractors=blend, size=(30.0, 30.0)))
    c = T.render_maps(steel_plan(kind, size=(30.0, 30.0)))
    U, V = np.meshgrid((np.arange(750) + 0.5) * RES, (np.arange(750) + 0.5) * RES)
    inner = np.hypot(U - 15.0, V - 15.0) < 4.0
    g, g0 = m["albedo"][..., 1][inner], c["albedo"][..., 1][inner]
    assert np.abs(m["height"][inner] - c["height"][inner]).max() > 0.1     # the dish is there
    if kind == "ms_hot_rolled":
        assert g.mean() > 0.35 > 0.2 > g0.mean() and m["metallic"][inner].mean() > 0.9
    elif kind == "galvanised":
        assert 0.4 < g.mean() < 0.6 and g.std() < 0.25 * g0.std() and m["metallic"][inner].min() > 0.99
    else:
        assert g.mean() == pytest.approx(1.02 * g0.mean(), rel=0.01)
        assert np.array_equal(m["metallic"], c["metallic"])


@pytest.mark.parametrize("kind", ["ms_hot_rolled", "galvanised"])
def test_full_steel_plate_timing(kind):
    """The slowest finishes with every steel kind, 100 x 100 mm at 0.04 mm: < 15 s (DESIGN budget)."""
    G = pytest.importorskip("ayas.twin.geometry")
    big = [
        defect(1, "corrosion", "rust", u=30.0, v=30.0, r_mm=25.0, coverage=0.8, pitted=True, aspect=1.2, angle_deg=10.0),
        defect(2, "corrosion", "rust_streak", u0=100.0, v0=80.0, angle_deg=180.0, length_mm=40.0, width_mm=6.0,
               source="edge", edge="right"),
        defect(3, "pit", "cluster", u=80.0, v=20.0, r_mm=8.0, n=30, pit_d_mm=[0.1, 1.2], depth_mm=0.4, halo_mm=0.5,
               halo=0.8, pits=[[80.0 + 7 * math.cos(a), 20.0 + 7 * math.sin(a), 0.8] for a in np.linspace(0, 6, 30)]),
        defect(4, "crack", "open", pts_mm=[[5.0 + 0.8 * i, 95.0 - 0.3 * (i % 3)] for i in range(50)], width_mm=0.4,
               start="free"),
        defect(5, "stain", "oil", u=75.0, v=60.0, r_mm=15.0, aspect=1.5, angle_deg=30.0, gloss=0.9, strength=1.0),
        defect(6, "stain", "water", u=60.0, v=85.0, r_mm=10.0, ring_w_mm=0.4, strength=1.0, minerals=1.0),
        defect(7, "stain", "fingerprint", u=10.0, v=70.0, r_mm=9.0, angle_deg=20.0, aspect=1.3, strength=1.0),
        defect(8, "scratch", "abrasion", u=45.0, v=70.0, w_mm=15.0, h_mm=15.0, angle_deg=15.0, n_lines=200,
               depth_mm=0.01),
    ]
    dis = [{"kind": "label_residue", "params": {"u": 90.0, "v": 45.0, "w_mm": 12.0, "h_mm": 8.0, "angle_deg": 5.0}},
           {"kind": "dust", "params": {"n": 300}}]
    plan = steel_plan(kind, big, dis, size=(100.0, 100.0), grain_dir_deg=17.0)
    t0 = time.perf_counter()
    m = T.render_maps(plan)
    dt = time.perf_counter() - t0
    print(f"render_maps {kind} 100x100 mm @ {RES} mm, 8 steel defects + 2 distractors: {dt:.2f} s")
    assert m["height"].shape == (2500, 2500) and dt < 15.0
    assert all(G.footprint(d, plan)["polys_mm"] for d in plan["defects"])
