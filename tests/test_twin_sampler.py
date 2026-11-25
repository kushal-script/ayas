import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from ayas.twin import frames as F
from ayas.twin import geometry as G
from ayas.twin import rig as RIG
from ayas.twin import sampler as S

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "twin.yaml").read_text())
CONDITIONS = ("C0_dark", "C1_office", "C2_warm_bulb", "C3_window", "C4_side_lamp", "C5_reflected_lamp", "C6_sun")

# DESIGN_STEEL A (steel kinds) and DESIGN 3.1 (kept kinds): measure keys per (cls, kind)
MEASURE_KEYS = {
    ("dent", "punch"): {"diameter_mm", "depth_mm"}, ("dent", "ball"): {"diameter_mm", "depth_mm"},
    ("dent", "smooth"): {"diameter_mm", "depth_mm"}, ("dent", "nick"): {"diameter_mm"},
    ("scratch", "scriber"): {"length_mm", "width_mm"}, ("scratch", "gouge"): {"length_mm", "width_mm"},
    ("scratch", "abrasion"): {"area_mm2", "length_mm", "width_mm"},
    ("crack", "slit"): {"length_mm", "width_mm"}, ("crack", "fatigue"): {"length_mm", "width_mm"},
    ("crack", "open"): {"length_mm", "width_mm"},
    ("pit", "single"): {"diameter_mm", "count"}, ("pit", "cluster"): {"diameter_mm", "count", "area_mm2", "pit_d_mm"},
    ("corrosion", "rust"): {"area_mm2", "diameter_mm"},
    ("corrosion", "rust_streak"): {"area_mm2", "diameter_mm", "length_mm"},
    ("corrosion", "white"): {"area_mm2", "diameter_mm"},
    ("corrosion", "pitting"): {"area_mm2", "diameter_mm"}, ("corrosion", "galvanic"): {"area_mm2", "diameter_mm"},
    ("corrosion", "fretting"): {"area_mm2", "length_mm"},
    ("stain", "oil"): {"area_mm2", "diameter_mm"}, ("stain", "water"): {"area_mm2", "diameter_mm"},
    ("stain", "fingerprint"): {"area_mm2", "diameter_mm"},
}
STEEL_CLASSES = {"scratch", "dent", "pit", "corrosion", "crack", "stain"}
# DESIGN 3 / DESIGN_STEEL D: key -> required sub-keys
DESIGN_KEYS = {
    "plate": {"size_mm", "thickness_mm", "edge_bevel_mm", "finish", "material"},
    "camera": {"K", "full_res", "R", "t", "roi", "fstop", "focus_m", "pose"},
    "ambient": {"condition", "lux", "cct_k", "world_color", "world_strength", "lights", "occluders", "change"},
    "extras": {"background", "gloves", "device_box", "device", "hood"},
    "render": {"samples", "noise_threshold", "max_bounces", "glossy_bounces", "texture_res_mm", "clean_pass"},
    "textures": {"normal", "albedo", "roughness", "metallic", "uv_emit"},
    "capture": {"sensor", "shake_px_per_frame", "led_jitter"},
}
ITEM_KEYS = {
    "holes": {"id", "u", "v", "d_mm", "ovality", "angle_deg", "sheet", "filled"},
    "defects": {"id", "cls", "kind", "params", "footprint", "near", "label"},
    "distractors": {"kind", "params"},
    "lights": {"name", "group", "kind", "pos_m", "dir", "spot_deg", "blend", "power_w", "color"},
}
PLATE_KEYS = ("schema", "seed", "split", "mode", "source", "plate", "sheets", "holes", "rivets", "fasteners", "empty",
              "defects", "distractors", "penetrant", "render", "textures")
MATERIAL_SPEC = {   # DESIGN_STEEL B: base grey, roughness, metallic, finish
    "ms_hot_rolled": ((0.08, 0.18), (0.45, 0.8), (0.3, 0.7), "hot_rolled"),
    "ms_cold_rolled": ((0.45, 0.6), (0.15, 0.35), (1.0, 1.0), "cold_rolled"),
    "ss_brushed": ((0.55, 0.65), (0.1, 0.3), (1.0, 1.0), "brushed"),
    "ss_2b": ((0.5, 0.6), (0.25, 0.4), (1.0, 1.0), "2b"),
    "galvanised": ((0.55, 0.7), (0.2, 0.4), (1.0, 1.0), "spangle"),
    "aluminium": ((0.88, 0.92), (0.12, 0.4), (1.0, 1.0), None),
}


# ----------------------------------------------------------------------------- lighting module
def stub_lighting(calls: list | None = None):
    """Minimal twin/lighting.py stand-in with the cross-module API (ambient_for, hood_for, CONDITIONS)."""

    def ambient_for(rng, plan, cfg, condition=None):
        if calls is not None:
            calls.append(("ambient", condition, "camera" in plan, (plan.get("extras") or {}).get("device")))
        return {"condition": condition, "lux": float(rng.uniform(0, 1e5)), "cct_k": 5000.0,
                "world_color": [1.0, 1.0, 1.0], "world_strength": float(rng.uniform(0, 1)), "lights": [],
                "occluders": [], "change": {"frame_sigma": 0.0, "shadow_shift_mm": 0.0}}

    def hood_for(rng, plan, cfg, device_mode):
        if calls is not None:
            calls.append(("hood", device_mode, "camera" in plan))
        if device_mode != "hood":
            return None
        return {"top_mm": [170.0, 160.0], "bottom_mm": [200.0, 190.0], "wall_mm": 2.0,
                "gap_mm": float(rng.uniform(0, 1)), "albedo": 0.03}

    return types.SimpleNamespace(CONDITIONS=CONDITIONS, ambient_for=ambient_for, hood_for=hood_for)


@pytest.fixture(scope="module", autouse=True)
def lighting():
    """The real twin/lighting.py when it imports with the agreed API, else the stub (monkeypatched)."""
    try:
        import ayas.twin.lighting as real
        ok = all(hasattr(real, k) for k in ("CONDITIONS", "ambient_for", "hood_for"))
    except Exception:
        real, ok = None, False
    mp = pytest.MonkeyPatch()
    mp.setattr(S, "lighting", real if ok else stub_lighting())
    yield S.lighting
    mp.undo()


@pytest.fixture(scope="module")
def plans(lighting):
    return [S.make_plan(CFG, s, "train", source="random") for s in range(120)]


def all_polys(plan):
    for d in plan["defects"]:
        yield d, d["footprint"]["polys_mm"]
    for ind in (plan["penetrant"] or {}).get("indications", []):
        yield ind, ind["footprint"]["polys_mm"]


def cfg_with(**sampler):
    cfg = json.loads(json.dumps(CFG))
    cfg["sampler"].update(sampler)
    return cfg


# ----------------------------------------------------------------------------- determinism, schema
def test_plan_is_deterministic_and_round_trips():
    a = S.make_plan(CFG, 42, "train")
    b = S.make_plan(CFG, 42, "train")
    assert a == b
    assert json.loads(json.dumps(a)) == a
    assert a["scene_id"] == "s000042" and a["seed"] == 42 and a["schema"] == "ayas.twin.scene/2"
    assert S.make_plan(CFG, 43, "train") != a
    # forcing the mode only overrides the choice; the plate and the view stay the same
    p = S.make_plan(CFG, 42, "train", mode="penetrant", source="random")
    q = S.make_plan(CFG, 42, "train", mode="surface", source="random")
    assert p["holes"] == q["holes"] and p["camera"] == q["camera"] and p["plate"] == q["plate"]


def test_every_design_key_is_present(plans):
    pen = [S.make_plan(CFG, s, "val", mode="penetrant", source="random") for s in range(3)]
    for plan in plans + pen:
        for k in ("schema", "scene_id", "seed", "split", "mode", "source", "penetrant", "fasteners", "empty"):
            assert k in plan
        for k, sub in DESIGN_KEYS.items():
            assert sub <= set(plan[k]), (k, sub - set(plan[k]))
        for k, sub in ITEM_KEYS.items():
            for item in plan[k]:
                assert sub <= set(item), (k, sub - set(item))
        mat = plan["plate"]["material"]
        assert {"kind", "base_color", "roughness", "metallic", "anisotropy", "aniso_rotation", "grain_amp_um",
                "grain_dir_deg"} <= set(mat)
        assert {"target_mm", "distance_mm", "tilt_deg", "roll_deg"} <= set(plan["camera"]["pose"])
        assert plan["extras"]["device"]["mode"] in ("open", "hood")
        assert plan["ambient"]["condition"] in CONDITIONS
        assert plan["render"]["samples"] == CFG["render"]["samples"] == 64
        assert not plan["rivets"] and not plan["sheets"]                     # steel plates: no rivets, no lap strip
        for d in plan["defects"]:
            assert d["cls"] in STEEL_CLASSES and d["label"] is True
            assert set(d["footprint"]["measure"]) == MEASURE_KEYS[(d["cls"], d["kind"])], (d["cls"], d["kind"])
            assert {"fastener_edge_mm", "plate_edge_mm", "near_edge_mm"} <= set(d["near"])
    for plan in pen:
        assert {"developer", "indications", "edge_bleed"} <= set(plan["penetrant"])


def test_class_kind_and_distractor_coverage(plans):
    kinds = {(d["cls"], d["kind"]) for p in plans for d in p["defects"]}
    new = {("pit", "single"), ("pit", "cluster"), ("corrosion", "rust"), ("corrosion", "rust_streak"),
           ("corrosion", "white"), ("crack", "open"), ("stain", "oil"), ("stain", "water"), ("stain", "fingerprint")}
    assert new <= kinds, new - kinds
    assert {c for c, _ in kinds} == STEEL_CLASSES
    dis = {x["kind"] for p in plans for x in p["distractors"]}
    assert dis <= set(S.DISTRACTORS) and {"marker", "dust", "blend", "label_residue"} <= dis
    assert not dis & {"fingerprint", "oil", "water_spot"}                    # stains are labelled defects now
    for p in plans:
        assert p["fasteners"]["pattern"] in ("none", "holes_row")
        for h in p["holes"]:                                                  # punched holes / slots: features
            assert not h["filled"] and h["kind"] in ("punched", "slot")
            assert (h["kind"] == "slot") == (h["ovality"] > 1.0)
    assert any(p["holes"] for p in plans) and any(h["kind"] == "slot" for p in plans for h in p["holes"])


def test_materials_sizes_and_thickness(plans):
    seen = set()
    sizes = {tuple(float(x) for x in k.split("x")) for k in CFG["sampler"]["sizes_mm"]}
    lo, hi = CFG["sampler"]["thickness_mm"]
    for p in plans:
        pl, mat = p["plate"], p["plate"]["material"]
        kind = mat["kind"]
        seen.add(kind)
        grey, rough, metal, finish = MATERIAL_SPEC[kind]
        assert tuple(pl["size_mm"]) in sizes and lo <= pl["thickness_mm"] <= hi
        assert finish is None and pl["finish"] in ("brushed", "mill", "clad") or pl["finish"] == finish
        assert grey[0] - 0.02 <= float(np.mean(mat["base_color"])) <= grey[1] + 0.02, (kind, mat["base_color"])
        assert rough[0] - 1e-6 <= mat["roughness"] <= rough[1] + 1e-6
        assert metal[0] - 1e-6 <= mat["metallic"] <= metal[1] + 1e-6
        if kind == "ms_hot_rolled":
            assert {"mottle", "cell_mm", "flake_frac", "spall_frac"} <= set(mat["scale"])
            assert mat["base_color"][2] > mat["base_color"][0]                # blue-grey mill scale
        if kind == "galvanised":
            assert {"cell_mm", "contrast"} <= set(mat["spangle"])
    assert seen == set(MATERIAL_SPEC)


def test_corrosion_follows_the_material():
    n = {"rust": 0, "white": 0, "halo": 0}
    for s in range(400):
        p = S.make_plan(CFG, s, "train", source="random")
        kind = p["plate"]["material"]["kind"]
        for d in p["defects"]:
            if d["kind"] in ("rust", "rust_streak"):
                assert kind in ("ms_hot_rolled", "ms_cold_rolled"), (s, kind)
                n["rust"] += 1
            if d["kind"] == "white":
                assert kind in ("aluminium", "galvanised"), (s, kind)
                n["white"] += 1
            if d["cls"] == "corrosion":
                assert not kind.startswith("ss_")                             # stainless: no corrosion class
            if d["cls"] == "pit" and d["params"]["halo_mm"] > 0:
                assert kind.startswith("ms_")                                 # rust halos only on mild steel
                n["halo"] += 1
    assert min(n.values()) >= 5, n
    for kind, allowed in (("ss_brushed", {}), ("ms_hot_rolled", {"rust", "rust_streak"}),
                          ("galvanised", {"white"})):
        assert set(S.kind_weights(CFG, "corrosion", kind)) == set(allowed)


# ----------------------------------------------------------------------------- the lighting matrix contract
def strip_view(plan):
    """Everything the condition / device mode may NOT change (gloves: none with the hood)."""
    out = {k: plan.get(k) for k in PLATE_KEYS}
    out["extras"] = {k: v for k, v in plan["extras"].items() if k not in ("device", "hood", "gloves")}
    return out


def test_condition_and_device_overrides_keep_the_plate(lighting):
    for seed in (3, 11, 900000, 900007):
        base = S.make_plan(CFG, seed, "test", source="random")
        rows = {(c, m): S.make_plan(CFG, seed, "test", source="random", condition=c, device_mode=m)
                for c in CONDITIONS for m in ("open", "hood")}
        ref = strip_view(base)
        for (c, m), plan in rows.items():
            assert strip_view(plan) == ref, (seed, c, m)
            assert plan["ambient"]["condition"] == c and plan["extras"]["device"]["mode"] == m
            assert (plan["extras"]["hood"] is None) == (m == "open")
            if m == "hood":
                assert plan["extras"]["gloves"] == []
        for m in ("open", "hood"):                     # same device mode: only the ambient changes
            first = rows[(CONDITIONS[0], m)]
            for c in CONDITIONS[1:]:
                for k in ("camera", "lights", "capture", "extras"):
                    assert rows[(c, m)][k] == first[k], (seed, c, m, k)
        # the drawn view is one of the rows, unchanged
        drawn = (base["ambient"]["condition"], base["extras"]["device"]["mode"])
        assert rows[drawn] == base
        assert rows[(CONDITIONS[0], "open")]["camera"] != rows[(CONDITIONS[0], "hood")]["camera"]


def test_overrides_are_validated_and_reach_lighting():
    calls = []
    mp = pytest.MonkeyPatch()
    mp.setattr(S, "lighting", stub_lighting(calls))
    try:
        p = S.make_plan(CFG, 5, "train", condition="C6_sun", device_mode="hood")
        assert p["ambient"]["condition"] == "C6_sun" and p["extras"]["hood"]["gap_mm"] >= 0
        # hood_for then ambient_for, both after the camera, with the device mode known to ambient_for
        assert calls == [("hood", "hood", True), ("ambient", "C6_sun", True, {"mode": "hood", "illumination": "pods"})]
        with pytest.raises(ValueError):
            S.make_plan(CFG, 5, "train", condition="C9_moon")
        with pytest.raises(ValueError):
            S.make_plan(CFG, 5, "train", device_mode="tripod")
    finally:
        mp.undo()


def test_device_mode_pose_and_capture_limits():
    hood_n = 0
    for s in range(150):
        for mode in ("open", "hood"):
            p = S.make_plan(CFG, s, "train", device_mode=mode)
            pose, cap = p["camera"]["pose"], p["capture"]
            W, H = p["plate"]["size_mm"]
            assert cap["sensor"] == "full" and 0 <= cap["led_jitter"] <= 0.05
            if mode == "hood":
                hood_n += 1
                assert 199.0 - 1e-6 <= pose["distance_mm"] <= 201.0 + 1e-6
                assert all(abs(a) <= 1.5 + 1e-6 for a in pose["tilt_deg"]) and not pose["wide"]
                assert abs(pose["target_mm"][0] - W / 2) <= 0.05 * W + 1e-6
                assert abs(pose["target_mm"][1] - H / 2) <= 0.05 * H + 1e-6
                assert 0 <= cap["shake_px_per_frame"] <= 0.2 + 1e-9
            else:
                assert 155 <= pose["distance_mm"] <= 250
                assert all(abs(a) <= CFG["sampler"]["tilt_deg"] for a in pose["tilt_deg"])
                sh = cap["shake_px_per_frame"]
                assert 0.2 - 1e-9 <= sh <= 1.5 + 1e-9 or 3.0 - 1e-9 <= sh <= 6.0 + 1e-9
    assert hood_n == 150
    big = [S.make_plan(CFG, s, "train", device_mode="open")["capture"]["shake_px_per_frame"] for s in range(400)]
    assert 0.01 < np.mean(np.asarray(big) >= 3.0) < 0.12                     # about 5 % shaky scenes


def test_drawn_device_modes_and_conditions_follow_cfg(plans):
    modes = [p["extras"]["device"]["mode"] for p in plans]
    assert 0.2 < modes.count("open") / len(modes) < 0.6                  # device_modes open 0.4 / hood 0.6
    assert {p["ambient"]["condition"] for p in plans} == set(CONDITIONS)
    only = cfg_with(device_modes={"hood": 1.0}, conditions={"C3_window": 1.0})
    p = S.make_plan(only, 1, "train")
    assert p["extras"]["device"]["mode"] == "hood" and p["ambient"]["condition"] == "C3_window"


# ----------------------------------------------------------------------------- footprints and placement
def test_footprints_inside_the_plate_and_off_holes(plans):
    n = 0
    for plan in plans:
        W, H = plan["plate"]["size_mm"]
        for d, polys in all_polys(plan):
            assert polys
            p = np.vstack(polys)
            assert p[:, 0].min() >= -1e-6 and p[:, 1].min() >= -1e-6
            assert p[:, 0].max() <= W + 1e-6 and p[:, 1].max() <= H + 1e-6
            if "kind" not in d:
                continue
            n += 1
            prm = d["params"]
            at_edge = d["kind"] == "nick" or prm.get("start") == "edge" or prm.get("source") == "edge"
            at_hole = prm.get("start") == "hole" or prm.get("source") == "hole"
            if not at_edge:
                assert d["near"]["plate_edge_mm"] >= 0.5 - 1e-3, (d["cls"], d["kind"])
            else:
                assert d["near"]["plate_edge_mm"] <= 0.05
            if not at_hole and d["near"]["fastener_edge_mm"] is not None:
                assert d["near"]["fastener_edge_mm"] >= 0.3 - 1e-3
            assert d["near"]["near_edge_mm"] == min(x for x in (d["near"]["plate_edge_mm"],
                                                                d["near"]["fastener_edge_mm"]) if x is not None)
        for a in plan["defects"]:                                             # no two defects overlap
            for b in plan["defects"]:
                if a["id"] < b["id"]:
                    assert G.polys_distance(a["footprint"]["polys_mm"], b["footprint"]["polys_mm"]) >= 0.5 - 1e-3
    assert n > 150


def test_edge_attached_defects_touch_only_their_own_edge(plans):
    """Nicks, edge cracks and edge rust streaks touch their start edge only: a streak from near a
    corner used to run off through the next edge (its length_mm then overstated the mark)."""
    n = 0
    for p in plans:
        W, H = p["plate"]["size_mm"]
        for d in p["defects"]:
            edge, _ = S._own_hole(d, p)
            if edge is None:
                continue
            n += 1
            q = np.vstack(d["footprint"]["polys_mm"])
            gaps = {"left": q[:, 0].min(), "right": W - q[:, 0].max(), "top": q[:, 1].min(), "bottom": H - q[:, 1].max()}
            assert gaps[edge] <= 0.05, (d["kind"], edge, gaps)
            assert all(g >= S.SHEET_GAP - 1e-3 for e, g in gaps.items() if e != edge), (d["kind"], edge, gaps)
    assert n > 15
    # the seed-69 case of the first build: top edge near the right corner, heading out at the right
    plan = {"plate": {"size_mm": [100.0, 60.0]}, "sheets": [], "holes": [], "rivets": []}
    d = {"cls": "corrosion", "kind": "rust_streak",
         "params": {"source": "edge", "edge": "top", "u0": 94.9, "v0": 0.0, "angle_deg": 68.9, "length_mm": 26.0,
                    "width_mm": 4.7}}
    edge, own = S._own_hole(d, plan)
    assert edge == "top" and own is None
    assert not S._placement_ok(G.footprint(d, plan)["polys_mm"], plan, [], edge, own)
    d["params"]["angle_deg"] = 90.0                                           # straight down: stays on the plate
    assert S._placement_ok(G.footprint(d, plan)["polys_mm"], plan, [], edge, own)
    assert not S._placement_ok(G.footprint(d, plan)["polys_mm"], plan, [], "left", own)   # not its edge


def test_cover_distractors_keep_off_labelled_defects(plans):
    """A marker dot or label residue paints over what lies below, so it never sits on a labelled
    defect (it would hide or relabel it); dust specks may."""
    n = 0
    for p in plans:
        for x in p["distractors"]:
            out = S._cover_outline(x["kind"], x["params"])
            if out is None:
                continue
            n += 1
            for d in p["defects"]:
                assert G.polys_distance([out], d["footprint"]["polys_mm"]) >= S.DEFECT_GAP - 1e-3, (x["kind"], d["kind"])
    assert n > 30
    plan = {"plate": {"size_mm": [100.0, 100.0]}, "sheets": [], "holes": [], "rivets": []}
    oil = {"cls": "stain", "kind": "oil", "label": True,
           "params": {"u": 30.0, "v": 30.0, "r_mm": 5.0, "aspect": 1.2, "angle_deg": 0.0}}
    oil["footprint"] = G.footprint(oil, plan)
    over = {"u": 36.0, "v": 30.0, "w_mm": 12.0, "h_mm": 8.0, "angle_deg": 0.0, "strength": 0.8}
    assert not S._distractor_ok("label_residue", over, plan, [oil])
    assert S._distractor_ok("label_residue", dict(over, u=60.0), plan, [oil])
    assert not S._distractor_ok("marker", {"u": 30.0, "v": 30.0, "r_mm": 1.0}, plan, [oil])
    assert S._distractor_ok("dust", {"u": 30.0, "v": 30.0, "r_mm": 10.0}, plan, [oil])
    assert S._distractor_ok("label_residue", over, plan, [dict(oil, label=False)])   # unlabelled: allowed


def test_pit_clusters_are_dense(plans):
    """A cluster's pits (with halos) cover >= ~10 % of its hull: sparse 'clusters' were mostly clean
    steel inside the label box (a third of them below 10 %), which post's p90 test drops."""
    cov = []
    for p in plans:
        for d in p["defects"]:
            if d["kind"] != "cluster":
                continue
            q = d["params"]
            r_hi = max(1.5, (q["pit_d_mm"][0] + q["pit_d_mm"][1]) / 4 * math.sqrt(q["n"] / S.CLUSTER_COVER))
            assert 1.5 - 1e-3 <= q["r_mm"] <= min(8.0, r_hi) + 1e-3
            disc = sum(math.pi * r * r for _, _, r in G.pit_discs(q))
            cov.append(disc / d["footprint"]["measure"]["area_mm2"])
    assert len(cov) > 10 and min(cov) >= 0.08 and np.mean(np.asarray(cov) < 0.1) <= 0.05


def test_gloves_never_with_the_hood_or_over_a_defect():
    """The hood sits over the part, so a finger at the plate edge would pass through its wall:
    hood plans have no gloves. Open plans keep a glove only off every labelled defect."""
    seen = 0
    for s in range(300):
        o = S.make_plan(CFG, s, "train", device_mode="open")
        assert S.make_plan(CFG, s, "train", device_mode="hood")["extras"]["gloves"] == []
        for g in o["extras"]["gloves"]:
            seen += 1
            out = S.glove_outline(g)
            ax = G.unit(g["angle_deg"])                       # the capsule is length_mm long along angle_deg
            assert np.ptp(out @ ax) == pytest.approx(g["length_mm"], abs=0.2)     # (polygonal round caps)
            assert np.ptp(out @ np.array([-ax[1], ax[0]])) == pytest.approx(2 * g["radius_mm"], abs=1e-3)
            for d in o["defects"]:
                assert G.polys_distance([out], d["footprint"]["polys_mm"]) >= S.DEFECT_GAP - 1e-3, (s, d["kind"])
    assert seen > 30


def test_some_compact_defects_sit_near_an_edge(plans):
    compact = [d for p in plans for d in p["defects"] if d["kind"] in S.NEAR_KINDS]
    near = [d for d in compact if d["near"]["near_edge_mm"] <= 5.0 + 1e-6]
    assert len(compact) > 100 and 0.1 < len(near) / len(compact) < 0.6


def test_steel_kind_params_ranges(plans):
    for p in plans:
        for d in p["defects"]:
            k, q, m = d["kind"], d["params"], d["footprint"]["measure"]
            if (d["cls"], k) == ("pit", "single"):
                assert 0.2 - 1e-6 <= q["d_mm"] <= 1.5 + 1e-6 and 0.05 - 1e-6 <= q["depth_mm"] <= 0.5 + 1e-6
                assert m["count"] == 1 and isinstance(m["count"], int)
            elif (d["cls"], k) == ("pit", "cluster"):
                assert 1.5 - 1e-6 <= q["r_mm"] <= 8 + 1e-6 and 3 <= q["n"] <= 30 and len(q["pits"]) == q["n"]
                assert 0.1 - 1e-6 <= q["pit_d_mm"][0] < q["pit_d_mm"][1] <= 1.2 + 1e-6
                assert m["count"] == q["n"]
                assert m["diameter_mm"] == pytest.approx(2 * math.sqrt(m["area_mm2"] / math.pi), abs=2e-3)
                assert m["pit_d_mm"] == pytest.approx(max(x[2] for x in q["pits"]), abs=1e-4)   # largest pit
                assert m["pit_d_mm"] <= q["pit_d_mm"][1] + 1e-4
            elif k == "rust":
                assert 2 - 1e-6 <= q["r_mm"] <= 25 + 1e-6 and 0.3 - 1e-6 <= q["coverage"] <= 1 + 1e-6
                assert isinstance(q["pitted"], bool)
                a, b = q["r_mm"], q["r_mm"] / q["aspect"]
                assert m["area_mm2"] == pytest.approx(q["coverage"] * math.pi * a * b, rel=1e-3)
            elif k == "rust_streak":
                assert 5 - 1e-6 <= q["length_mm"] <= 40 + 1e-6 and 1 - 1e-6 <= q["width_mm"] <= 6 + 1e-6
                assert q["source"] in ("edge", "hole", "spot") and (q["source"] == "spot") == ("head_r_mm" in q)
            elif k == "white":
                assert 1.5 - 1e-6 <= q["r_mm"] <= 10 + 1e-6 and len(q["pits"]) == q["n_pits"]
            elif k == "open":
                L = G.polyline_length(q["pts_mm"])
                assert 3 - 1e-3 <= L <= 40 + 1e-3 and 0.05 - 1e-6 <= q["width_mm"] <= 0.4 + 1e-6
                assert q["start"] in ("edge", "hole", "free")
            elif d["cls"] == "stain":
                lo, hi = {"oil": (2, 15), "water": (2, 12), "fingerprint": (5, 9)}[k]
                assert lo - 1e-6 <= q["r_mm"] <= hi + 1e-6
                if k == "oil":
                    assert 0.3 - 1e-6 <= q["gloss"] <= 1 + 1e-6


def test_empty_scenes_have_no_defects_or_holes():
    p = S.make_plan(cfg_with(empty_scene_frac=1.0), 5, "train", source="random")
    assert p["empty"] and not p["defects"] and not p["holes"] and not p["rivets"] and not p["sheets"]


def test_roi_contains_projected_plate(plans):
    for plan in plans:
        cam = plan["camera"]
        K, R, t = np.array(cam["K"]), np.array(cam["R"]), np.array(cam["t"])
        x0, y0, w, h = cam["roi"]
        Ws, Hs = cam["full_res"]
        assert 0 <= x0 and 0 <= y0 and x0 + w <= Ws and y0 + h <= Hs
        W, Hp = plan["plate"]["size_mm"]
        corners = F.project(K, R, t, F.plate_to_world([[0, 0], [W, 0], [W, Hp], [0, Hp]]))
        assert (corners >= 0).all() and (corners[:, 0] <= Ws - 1).all() and (corners[:, 1] <= Hs - 1).all()
        q = F.project(K, R, t, S.scene_points_world(plan))
        assert x0 <= q[:, 0].min() and q[:, 0].max() <= x0 + w - 1
        assert y0 <= q[:, 1].min() and q[:, 1].max() <= y0 + h - 1


def test_nominal_scale_near_0065_mm_per_px():
    plan = S.make_plan(CFG, 2, "train", device_mode="hood")
    cam = plan["camera"]
    Hm = F.homography_plate_to_image(np.array(cam["K"]), np.array(cam["R"]), np.array(cam["t"]))
    tu, tv = cam["pose"]["target_mm"]
    px = F.apply_h(Hm, [[tu, tv], [tu + 1, tv], [tu, tv + 1]])
    mm_per_px = 1.0 / np.sqrt(np.linalg.norm(px[1] - px[0]) * np.linalg.norm(px[2] - px[0]))
    assert mm_per_px == pytest.approx(0.065, rel=0.06)
    assert np.allclose(cam["K"], F.nominal_K(), atol=1e-4)


# ----------------------------------------------------------------------------- rig and lights
def test_rig_matches_fusion_check_values():
    rig = RIG.load_rig(CFG, ROOT)
    white = [led for led in rig.leds if led.kind == "white"]
    uv = [led for led in rig.leds if led.kind == "uv"]
    assert len(white) == 18 and len(uv) == 18
    assert {led.group for led in uv} == {"UV"}
    L = [led for led in white if led.group == "L"]
    assert len(L) == 9 and all(led.p_cam_mm[0] < 0 for led in L)
    by = {led.name: led for led in rig.leds}

    def model(name):          # back to model coordinates relative to the lens (upright board)
        p = by[name].p_cam_mm
        return [-p[0], -p[1], p[2]]

    xyz = np.array([model(f"WL_L_{k}") for k in range(9)])
    assert np.allclose(xyz[::3, 0], [32.17, 45.75, 59.33], atol=0.06)   # 40 mm modules at 14 mm pitch (24 Nov)
    assert np.allclose(xyz[::3, 2], [12.03, 15.42, 18.81], atol=0.06)   # pod base 8 mm
    assert np.allclose(xyz[:3, 1], [-0.9, 11.6, 24.1], atol=0.06)     # 3 LEDs 12.5 mm apart
    assert np.allclose(by["WL_L_0"].d_cam, [0.242, 0.0, 0.970], atol=2e-3)
    uv3 = [model(f"UV_L_{k}") for k in range(3)]
    assert np.allclose(uv3, [[29.86, -49.68, 13.02], [29.18, -59.44, 15.74], [28.5, -69.21, 18.45]], atol=0.06)


def test_plan_lights_l_group_on_image_left(plans):
    rig = S.get_rig(CFG, ROOT)
    for plan in plans[:20]:
        cam = plan["camera"]
        R, t = np.array(cam["R"]), np.array(cam["t"])
        lights = plan["lights"]
        nominal = {x["name"]: x["power_w"] for x in RIG.lights_world(rig, R, t)}
        ring = plan["extras"]["device"]["illumination"] == "ring"
        assert ring == (plan["extras"]["device"]["mode"] == "hood")         # device v2: the hood head has the ring
        assert len([x for x in lights if x["group"] in ("L", "R")]) == 18
        for x in lights:
            pc = R @ np.array(x["pos_m"]) + t                  # LED in the camera frame
            if x["group"] == "L":
                assert pc[0] < 0
            if x["group"] == "R":
                assert pc[0] > 0
            if ring and x["group"] in ("L", "R"):              # on the hood walls, 18 mm above the plate
                assert x["mount"] == "ring" and x["pos_m"][2] == pytest.approx(0.018, abs=1e-6)
                assert 0.15 < pc[2] < 0.2
            else:
                assert abs(pc[2]) < 0.03                        # LEDs sit near the lens plane
                assert np.dot(R @ np.array(x["dir"]), [0, 0, 1]) > 0.9
            assert x["power_w"] == pytest.approx(nominal[x["name"]], rel=0.151)   # +-15 % per pod


def test_rig_cache_follows_the_capture_config_lumens(tmp_path):
    """Review fix: the white LED power comes from capture.yaml leds.white_lumens_per_led, but the rig
    cache key left the capture config out, so a second cfg with other lumens got the first rig's power."""
    cap = yaml.safe_load((ROOT / "configs" / "capture.yaml").read_text())
    lm = float(cap["leds"]["white_lumens_per_led"])
    cap["leds"]["white_lumens_per_led"] = 2 * lm
    path = tmp_path / "capture_2x.yaml"
    path.write_text(yaml.safe_dump(cap))
    cfg2 = json.loads(json.dumps(CFG))
    cfg2["capture_config"] = str(path)
    a = S.make_plan(CFG, 11, "train", condition="C0_dark", device_mode="open")
    b = S.make_plan(cfg2, 11, "train", condition="C0_dark", device_mode="open")
    pa = {x["name"]: x["power_w"] for x in a["lights"] if x["group"] in ("L", "R")}
    pb = {x["name"]: x["power_w"] for x in b["lights"] if x["group"] in ("L", "R")}
    assert pa.keys() == pb.keys() and all(pb[k] == pytest.approx(2 * pa[k], rel=1e-4) for k in pa)
    assert S.get_rig(CFG, ROOT) is S.get_rig(json.loads(json.dumps(CFG)), ROOT)   # still cached


# ----------------------------------------------------------------------------- penetrant (kept, off in cfg)
def test_penetrant_plans_have_indications():
    n = 0
    for s in range(12):
        p = S.make_plan(CFG, s, "train", mode="penetrant", source="random")
        if p["empty"]:
            continue
        n += 1
        inds = p["penetrant"]["indications"]
        ids = {d["id"] for d in p["defects"]}
        cracks = {d["id"] for d in p["defects"] if d["cls"] == "crack"}
        assert cracks and cracks <= {i["from_defect"] for i in inds if i["cls"] == "linear_indication"}
        for i in inds:
            assert i["cls"] in ("linear_indication", "rounded_indication")
            assert i["from_defect"] is None or i["from_defect"] in ids
        assert 0 <= p["penetrant"]["edge_bleed"] <= 0.5
    assert n >= 8


# ----------------------------------------------------------------------------- aero patterns (kept via cfg)
AERO = {"fastener_patterns": {"none": 0.15, "single": 0.2125, "double": 0.2125, "staggered": 0.2125,
                              "perimeter": 0.2125}, "materials": {"aluminium": 1.0}}


def test_aero_fastener_patterns_still_work():
    cfg = cfg_with(**AERO)
    seen = set()
    for s in range(80):
        plan = S.make_plan(cfg, s, "train", source="random")
        fs = plan["fasteners"]
        if not plan["holes"]:
            continue
        D = fs["D_mm"]
        W, H = plan["plate"]["size_mm"]
        viol = {v["kind"]: v for v in fs["violations"]}
        seen |= set(viol)
        edge_bad = set(viol.get("edge_distance", {}).get("holes", []))
        for h in plan["holes"]:
            e = min(h["u"], h["v"], W - h["u"], H - h["v"])
            if h["id"] in edge_bad:
                assert 1.0 * D - 1e-3 <= e <= 1.8 * D + 1e-3
            else:
                assert e >= 2.0 * D - 1e-3
        filled = {h["id"] for h in plan["holes"] if h["filled"]}
        assert {r["hole"] for r in plan["rivets"]} == filled
    assert {"missing", "edge_distance"} <= seen


# ----------------------------------------------------------------------------- replicas (kept, off in cfg)
def test_replica_from_a_coupon_table():
    table = {
        "P2A": {"plate": {"size_mm": [100.0, 100.0], "thickness_mm": 3.0, "finish": "mill"},
                "sheets": [], "distractors": [{"kind": "fingerprint", "params": {"u": 30, "v": 40, "r_mm": 7}}],
                "holes": [{"id": "H1", "u": 20.0, "v": 10.0, "d_mm": 4.1, "filled": True},
                          {"id": "H2", "u": 40.0, "v": 5.0, "d_mm": 4.1, "filled": True},
                          {"id": "H3", "u": 60.0, "v": 10.0, "d_mm": 4.1, "filled": False}],
                "rivets": [{"id": "R1", "hole": "H1", "head_d_mm": 8.0, "head_h_mm": 1.3},
                           {"id": "R2", "hole": "H2", "head_d_mm": 8.0, "head_h_mm": 1.3}],
                "fastener_d_mm": 4.0, "expect": [{"type": "edge_distance", "hole": "H2"}],
                "defects": [{"id": "D1", "cls": "crack", "kind": "slit",
                             "params": {"hole": "H3", "start": "hole", "u0": 62.05, "v0": 10.0, "angle_deg": 0.0,
                                        "length_mm": 4.0, "width_mm": 0.4}},
                            {"id": "D2", "cls": "dent", "kind": "punch",
                             "params": {"u": 50.0, "v": 60.0, "d_mm": 1.5, "depth_mm": 0.2, "rim_h_mm": 0.03}}]},
        "P5A": {"plate": {"size_mm": [100.0, 100.0]}, "holes": [], "rivets": [], "defects": [], "distractors": []},
    }
    cfg = cfg_with(replica_coupons=["P2", "P5"])
    p = S.make_plan(cfg, 3, "train", source="replica:P2A", coupons=table)
    assert p["source"] == "replica:P2A" and p["plate"]["material"]["kind"] == "aluminium"
    assert [h["id"] for h in p["holes"]] == ["H1", "H2", "H3"]
    for h, h0 in zip(p["holes"], table["P2A"]["holes"]):
        assert abs(h["u"] - h0["u"]) <= 0.3 + 1e-9 and abs(h["v"] - h0["v"]) <= 0.3 + 1e-9
    assert p["fasteners"]["violations"][0]["holes"] == ["H2"]
    slit = p["defects"][0]
    assert slit["near"]["fastener_edge_mm"] == 0.0 and slit["near"]["fastener"] == "H3"
    assert 3.2 - 1e-6 <= slit["footprint"]["measure"]["length_mm"] <= 4.8 + 1e-6
    with pytest.raises(ValueError):
        S.make_plan(cfg, 3, "train", source="replica:P5A", coupons=table)
    for s in range(20):
        q = S.make_plan(cfg, s, "train", source="replica", coupons=table)
        assert "P5" not in q["source"]
    assert S._replica_ids(table, ["P2", "P5"]) == ["P2A"]


def test_replica_of_the_real_coupons():
    if not (ROOT / "configs" / "coupons.yaml").exists():
        pytest.skip("configs/coupons.yaml not written yet")
    table = S.load_coupon_table(ROOT)
    allowed = ["P1", "P2", "P3", "P4", "S1"]
    cfg = cfg_with(replica_coupons=allowed)
    ids = S._replica_ids(table, allowed)
    assert ids and not any(i.upper().startswith("P5") for i in ids) and "P5A" in table
    with pytest.raises(ValueError):
        S.make_plan(cfg, 1, "train", source="replica:P5A", coupons=table)
    for cid in ids:
        for seed in range(2):
            p = S.make_plan(cfg, seed, "train", mode="penetrant" if seed % 2 else "surface",
                            source=f"replica:{cid}", coupons=table)
            assert [h["id"] for h in p["holes"]] == [h["id"] for h in table[cid]["holes"]]
            assert [d["id"] for d in p["defects"]] == [d["id"] for d in table[cid]["defects"]]
            W, H = p["plate"]["size_mm"]
            for _, polys in all_polys(p):
                q = np.vstack(polys)
                assert q.min() >= -1e-6 and q[:, 0].max() <= W + 1e-6 and q[:, 1].max() <= H + 1e-6


# ----------------------------------------------------------------------------- writing plans
def test_write_plans(tmp_path):
    paths = S.write_plans(CFG, 3, tmp_path, start_seed=10)
    assert [p.name for p in paths] == ["s000010.json", "s000011.json", "s000012.json"]
    plan = json.loads(paths[1].read_text())
    assert plan["seed"] == 11 and plan["split"] == S.split_for(CFG, 11)
    splits = [S.split_for(CFG, s) for s in range(400)]
    assert 0.8 < splits.count("train") / 400 < 0.97
    # one plate under two views in the same directory: suffixed ids, same plate, forced split
    a = S.write_plans(CFG, 2, tmp_path, start_seed=900000, condition="C6_sun", device_mode="hood", split="test")
    b = S.write_plans(CFG, 2, tmp_path, start_seed=900000, condition="C0_dark", device_mode="open", split="test")
    assert [p.name for p in a] == ["s900000_C6h.json", "s900001_C6h.json"]
    assert [p.name for p in b] == ["s900000_C0o.json", "s900001_C0o.json"]
    pa, pb = json.loads(a[0].read_text()), json.loads(b[0].read_text())
    assert pa["seed"] == pb["seed"] == 900000 and pa["split"] == pb["split"] == "test"
    assert pa["defects"] == pb["defects"] and pa["plate"] == pb["plate"]
    assert pa["ambient"]["condition"] == "C6_sun" and pb["extras"]["device"]["mode"] == "open"


# ----------------------------------------------------------------------------- geometry
def test_bezier_length_of_a_straight_control_polygon_is_its_chord():
    ctrl = [[1.0, 2.0], [3.0, 3.0], [4.0, 3.5], [9.0, 6.0]]
    assert G.bezier_length(ctrl) == pytest.approx(math.hypot(8.0, 4.0), rel=1e-9)
    k = 4 / 3 * (math.sqrt(2) - 1)                              # quarter circle of radius 10
    quarter = [[10, 0], [10, 10 * k], [10 * k, 10], [0, 10]]
    assert G.bezier_length(quarter) == pytest.approx(math.pi * 5, rel=1e-3)


def test_near_on_a_known_case():
    plan = {"plate": {"size_mm": [100.0, 80.0]},
            "holes": [{"id": "H1", "u": 50.0, "v": 40.0, "d_mm": 4.1, "ovality": 1.0, "angle_deg": 0.0, "filled": True},
                      {"id": "H2", "u": 20.0, "v": 20.0, "d_mm": 4.0, "ovality": 1.25, "angle_deg": 90.0,
                       "filled": False}],
            "rivets": [{"id": "R1", "hole": "H1", "head_d_mm": 8.0, "head_h_mm": 1.3}]}
    dent = {"cls": "dent", "kind": "punch", "params": {"u": 62.0, "v": 40.0, "d_mm": 2.0, "depth_mm": 0.2}}
    nr = G.near(G.footprint(dent, plan)["polys_mm"], plan)
    assert nr["fastener_edge_mm"] == pytest.approx(7.0, abs=0.01) and nr["fastener"] == "H1"
    assert nr["plate_edge_mm"] == pytest.approx(37.0, abs=1e-6) and nr["near_edge_mm"] == nr["fastener_edge_mm"]
    dent2 = {"cls": "dent", "kind": "punch", "params": {"u": 20.0, "v": 25.0, "d_mm": 2.0, "depth_mm": 0.2}}
    nr2 = G.near(G.footprint(dent2, plan)["polys_mm"], plan)
    assert nr2["fastener"] == "H2" and nr2["fastener_edge_mm"] == pytest.approx(1.5, abs=0.01)
    dent3 = {"cls": "dent", "kind": "ball", "params": {"u": 55.0, "v": 40.0, "d_mm": 4.0, "depth_mm": 0.2}}
    assert G.near(G.footprint(dent3, plan)["polys_mm"], plan)["fastener_edge_mm"] == 0.0
    free = G.near([G.circle((5, 3), 1)], {"plate": {"size_mm": [10, 10]}})
    assert free["fastener_edge_mm"] is None and free["near_edge_mm"] == pytest.approx(2.0, abs=1e-6)


def test_steel_footprints_on_known_cases():
    plan = {"plate": {"size_mm": [100.0, 60.0]}, "holes": [], "rivets": []}

    def fp(cls, kind, **params):
        return G.footprint({"cls": cls, "kind": kind, "params": params}, plan)

    f = fp("pit", "single", u=10.0, v=10.0, d_mm=0.8, depth_mm=0.2, halo_mm=0.4)
    assert f["measure"] == {"diameter_mm": 0.8, "count": 1}
    assert np.ptp(np.asarray(f["polys_mm"][0])[:, 0]) == pytest.approx(2 * 0.8, abs=1e-3)     # d/2 + halo
    pits = [[29.0, 29.0, 0.5], [32.0, 31.0, 0.8], [30.0, 33.0, 0.3]]
    f = fp("pit", "cluster", u=30.0, v=30.0, r_mm=4.0, n=3, pits=pits, halo_mm=0.0)
    q = np.asarray(f["polys_mm"][0])
    for u, v, d in pits:                                         # every pit disc inside the hull
        assert G.point_in_polygon(G.circle((u, v), d / 2 * 0.99, 16), q).all()
    assert f["measure"]["count"] == 3 and f["measure"]["area_mm2"] == pytest.approx(G.polygon_area(q), rel=1e-3)
    assert f["measure"]["pit_d_mm"] == 0.8                                       # the largest pit, not a range bound
    f = fp("corrosion", "rust", u=50.0, v=30.0, r_mm=10.0, coverage=0.5, aspect=2.0, angle_deg=30.0)
    assert f["measure"]["area_mm2"] == pytest.approx(0.5 * math.pi * 10 * 5, rel=1e-4)
    f = fp("corrosion", "rust_streak", u0=0.0, v0=30.0, angle_deg=0.0, length_mm=20.0, width_mm=4.0)
    q = np.asarray(f["polys_mm"][0])
    assert q[:, 0].min() == pytest.approx(0.0) and q[:, 0].max() == pytest.approx(20.0, abs=1e-3)   # clipped at the edge
    assert np.ptp(q[:, 1]) == pytest.approx(4.0, abs=1e-3)
    f = fp("corrosion", "rust_streak", u0=40.0, v0=10.0, angle_deg=90.0, length_mm=20.0, width_mm=2.0, head_r_mm=1.6)
    q = np.asarray(f["polys_mm"][0])
    assert np.ptp(q[:, 0]) == pytest.approx(3.2, abs=1e-3) and q[:, 1].max() == pytest.approx(30.0, abs=1e-3)
    f = fp("crack", "open", pts_mm=[[5.0, 50.0], [8.0, 51.0], [12.0, 49.0]], width_mm=0.1)
    assert f["measure"]["length_mm"] == pytest.approx(math.hypot(3, 1) + math.hypot(4, 2), abs=1e-3)
    hw = 0.05 + G.crack_lip_mm(0.1)
    assert np.asarray(f["polys_mm"][0])[:, 0].min() == pytest.approx(5.0 - hw, abs=0.01)
    f = fp("stain", "oil", u=80.0, v=20.0, r_mm=5.0, aspect=1.5, angle_deg=0.0)
    assert f["measure"] == pytest.approx({"area_mm2": math.pi * 5 * 5 / 1.5, "diameter_mm": 10.0}, rel=1e-3)
    f = fp("stain", "fingerprint", u=60.0, v=30.0, r_mm=7.0, angle_deg=0.0)                       # widest default
    assert np.ptp(np.asarray(f["polys_mm"][0])[:, 1]) == pytest.approx(2 * 7 / G.FINGERPRINT_ASPECT, abs=1e-3)
    f = fp("stain", "water", u=80.0, v=45.0, r_mm=4.0)
    assert f["measure"]["diameter_mm"] == 8.0


def test_polygon_helpers():
    ring = G.ring((10, 10), 4.0, 6.0)
    assert G.polygon_area(ring) == pytest.approx(math.pi * 20, rel=5e-3)
    assert G.point_in_polygon([[10, 10], [15, 10], [17, 10]], ring).tolist() == [False, True, False]
    band = G.buffer_polyline([[0, 0], [10, 0]], 0.5)
    assert G.polygon_area(band) == pytest.approx(10 + math.pi * 0.25, rel=5e-3)
    thin = G.buffer_polyline([[0, 0], [10, 0]], 0.001)
    assert np.ptp(np.asarray(thin)[:, 1]) == pytest.approx(0.1, rel=1e-6)
    assert G.poly_distance(G.circle((0, 0), 1), G.circle((5, 0), 2)) == pytest.approx(2.0, abs=1e-6)
    assert G.poly_distance(G.circle((0, 0), 5), G.circle((0, 0), 1)) == 0.0
    clipped = G.clip_to_rect(G.rect((0, 5), 4, 2), 0, 0, 10, 10)
    assert G.polygon_area(clipped) == pytest.approx(4.0)
    hull = G.convex_hull([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0]])
    assert len(hull) == 4 and G.polygon_area(hull) == pytest.approx(4.0)
    assert G.equivalent_diameter(math.pi) == pytest.approx(2.0)
