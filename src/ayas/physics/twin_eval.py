"""Twin helpers for the physics work (docs/twin/PHYSICS_PINN.md sections 2-4).

- grid_frames_*: the ambient-subtracted LED frames on the rectified plate grid, from a twin
  render (EXR light groups) either noise-free (validation) or through the real burst path
  (burst.simulate_burst -> lightproc shifts and gate -> aligned L - A, R - A -> post.rectify).
- crop inputs for depth_pinn.estimate_depth (frames, meta, box) from a plan and grid frames.
- ground truth: textures.render_maps(plan) height (mm) and the depth of each labelled dent,
  pit or scratch (deepest point below the median of its crop border).
- depth coupons: plans with many dents and pits of known depth on each material, whose ring
  LEDs each carry their own light group (E00..E17), so L/R and any ring sector split can be
  summed from one Blender render (render_scene.py supports arbitrary groups: setup_render adds
  every plan light group as a view-layer light group, make_light sets obj.lightgroup).

Nothing here edits the twin; it only calls its public functions.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import optics as O

LED_GROUP_FMT = "E{:02d}"


# ----------------------------------------------------------------------------- geometry

def twin_cfg() -> dict:
    import yaml

    from ayas.lightproc import ROOT
    return yaml.safe_load(open(ROOT / "configs" / "twin.yaml"))


def capture_cfg() -> dict:
    from ayas import lightproc
    return lightproc.load_capture_config()


def grid_geom(plan: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> dict:
    from ayas.twin import post
    return post.grid_geometry(plan, (cfg or twin_cfg())["grid"])


def camera_centre(plan: Mapping[str, Any]) -> np.ndarray:
    cam = plan["camera"]
    R, t = np.asarray(cam["R"], float), np.asarray(cam["t"], float)
    return -R.T @ t


def meta_from_plan(plan: Mapping[str, Any], geom: Mapping[str, Any], cap: Mapping[str, Any] | None = None,
                   lights: Sequence[Mapping[str, Any]] | None = None, k: float | None = None) -> dict:
    """The depth estimator's 'meta' for a twin scene: LED dicts, camera centre, grid mapping,
    exposure k (FS per radiance unit), sensor, material nominal (for priors, not truth)."""
    from ayas.twin import burst
    cap = cap or capture_cfg()
    if k is None:
        k = burst.exposure_k(plan, cap)["k"]
    return {"lights": [dict(L) for L in (lights if lights is not None else plan["lights"])],
            "camera_centre": camera_centre(plan).tolist(),
            "camera_R": plan["camera"]["R"], "camera_t": plan["camera"]["t"],
            "H_grid_plate": np.asarray(geom["H_grid_plate"], float).tolist(),
            "mm_per_px": float(geom["mm_per_px"]), "k": float(k),
            "sensor": dict(cap.get("sensor") or {}),
            "illumination": str(((plan.get("extras") or {}).get("device") or {}).get("illumination") or "pods"),
            "hood": _hood_of(plan),
            "plate_size_mm": list(plan["plate"]["size_mm"])}


def _hood_of(plan: Mapping[str, Any]) -> dict | None:
    ex = plan.get("extras") or {}
    if (ex.get("device") or {}).get("mode") != "hood" or not isinstance(ex.get("hood"), dict):
        return None
    return dict(ex["hood"])


# ----------------------------------------------------------------------------- frames from renders

def read_groups(exr: str | Path, groups: Sequence[str]) -> dict[str, np.ndarray]:
    from ayas.twin import post
    return post.read_light_groups(exr, tuple(groups))


def rectify(img: np.ndarray, geom: Mapping[str, Any]) -> np.ndarray:
    from ayas.twin import post
    return post.rectify(img, geom)


def lum(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.float32) @ np.asarray(O.LUMA, np.float32)


def grid_frames_clean(groups: Mapping[str, np.ndarray], geom: Mapping[str, Any], k: float) -> dict[str, np.ndarray]:
    """Noise-free LED frames on the grid in FS (luma): k * rectified pass."""
    return {g: lum(rectify(np.asarray(a, np.float32), geom)) * np.float32(k) for g, a in groups.items()}


def grid_frames_burst(groups: Mapping[str, np.ndarray], plan: Mapping[str, Any], geom: Mapping[str, Any],
                      rng: np.random.Generator, cap: Mapping[str, Any] | None = None,
                      amb_mult: float = 1.0) -> tuple[dict[str, np.ndarray], dict]:
    """The real capture path: simulate_burst (noise, shake, ambient, LED jitter) -> lightproc
    process_burst (shifts, gate) -> aligned Lc = L - A, Rc = R - A (not clipped at 0, so the
    noise stays unbiased) -> rectified luma in FS. Returns (frames, info with gate and noise)."""
    from ayas import lightproc as LP
    from ayas.twin import burst, post
    cap = cap or capture_cfg()
    g = {k: v for k, v in groups.items() if k in ("L", "R", "AMB")}
    if amb_mult != 1.0:
        g["AMB"] = g["AMB"] * np.float32(amb_mult)
    roi = post.plate_mask_image(geom, g["L"].shape)
    b = burst.simulate_burst(g, plan, rng, cap, roi_mask=roi)
    pr = LP.process_burst(b, roi, cap)
    al = {f: (b.frames[f] if f == "B1" else LP.align(b.frames[f], pr.shifts[f])) for f in LP.SEQUENCE}
    Lc, Rc = al["L"] - al["A"], al["R"] - al["A"]
    A = al["A"]
    out = {"L": lum(rectify(Lc, geom)), "R": lum(rectify(Rc, geom))}
    # per-pixel noise variance of L - A in FS^2 (shot noise of both frames + read noise), grid luma
    e_fs = LP.electrons_per_fs(cap.get("sensor") or {})
    rn = float((cap.get("sensor") or {}).get("read_noise_e", 0.0))
    var = {}
    for name, F in (("L", al["L"]), ("R", al["R"])):
        v = (np.maximum(lum(F), 0) + np.maximum(lum(A), 0)) / e_fs + 2 * (rn / e_fs) ** 2
        var[name] = rectify(v.astype(np.float32), geom) * 0.55     # luma averages 3 channels (~0.55 x var)
    info = {"gate": pr.gate.verdict, "reasons": list(pr.gate.reasons), "message": pr.gate.message,
            "metrics": {k: pr.metrics.get(k) for k in ("r_ambient", "led_level", "motion_px", "masked_frac",
                                                        "sat_frac_a", "closure_sigma", "ghost_bins")},
            "gains": b.meta.get("led_gains"), "k": b.meta.get("k"), "var": var}
    return out, info


# ----------------------------------------------------------------------------- crops

def box_mm_of(defect: Mapping[str, Any]) -> list[float]:
    """[u0, v0, u1, v1] plate mm of a defect's footprint polygons."""
    P = np.concatenate([np.asarray(p, float) for p in defect["footprint"]["polys_mm"]])
    return [float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())]


def depth_targets(plan: Mapping[str, Any], maps: Mapping[str, np.ndarray] | None = None,
                  classes: Sequence[str] = ("dent", "pit", "scratch"), margin_mm: float = 1.0) -> list[dict]:
    """Ground truth per labelled dent/pit/scratch: max depth (mm) below the median height of a
    margin_mm ring around its footprint box, its location and the box (from the texture height)."""
    from ayas.twin import textures
    if maps is None:
        maps = textures.render_maps(plan)
    h = maps["height"]
    nv, nu = h.shape
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    ru, rv = W / nu, H / nv
    out = []
    for d in plan.get("defects") or []:
        if d.get("cls") not in classes or not d.get("label", True):
            continue
        u0, v0, u1, v1 = box_mm_of(d)
        c0, c1 = max(0, int((u0 - margin_mm) / ru)), min(nu, int(math.ceil((u1 + margin_mm) / ru)))
        r0, r1 = max(0, int((v0 - margin_mm) / rv)), min(nv, int(math.ceil((v1 + margin_mm) / rv)))
        win = h[r0:r1, c0:c1]
        cu0, cu1 = int((u0 - c0 * ru) / ru), int(math.ceil((u1 - c0 * ru) / ru))
        cv0, cv1 = int((v0 - r0 * rv) / rv), int(math.ceil((v1 - r0 * rv) / rv))
        ring = np.ones(win.shape, bool)
        ring[max(0, cv0):cv1, max(0, cu0):cu1] = False
        ref = float(np.median(win[ring])) if ring.any() else 0.0
        inner = win[max(0, cv0):cv1, max(0, cu0):cu1]
        depth = float(ref - inner.min()) if inner.size else 0.0
        vol = float(np.sum(np.maximum(ref - inner, 0.0)) * ru * rv) if inner.size else 0.0
        out.append({"id": d["id"], "cls": d["cls"], "kind": d["kind"], "box_mm": [u0, v0, u1, v1],
                    "depth_mm": depth, "volume_mm3": vol, "param_depth_mm": (d.get("params") or {}).get("depth_mm"),
                    "measure": (d.get("footprint") or {}).get("measure"), "near": d.get("near")})
    return out


def scratch_segments(plan: Mapping[str, Any], maps: Mapping[str, np.ndarray], half_mm: float = 1.5,
                     margin_mm: float = 1.0) -> list[dict]:
    """One window per labelled scratch, centred on its Bezier midpoint (half_mm a side): the
    depth target of a scratch is the groove depth there (a whole scratch is too long for one crop)."""
    from ayas.twin import geometry as G
    h = maps["height"]
    nv, nu = h.shape
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    ru, rv = W / nu, H / nv
    out = []
    for d in plan.get("defects") or []:
        if d.get("cls") != "scratch" or not d.get("label", True) or "ctrl_mm" not in (d.get("params") or {}):
            continue
        pts = G.bezier_points(d["params"]["ctrl_mm"], 65)
        u, v = (float(x) for x in pts[32])
        box = [u - half_mm, v - half_mm, u + half_mm, v + half_mm]
        c0, c1 = max(0, int((box[0] - margin_mm) / ru)), min(nu, int((box[2] + margin_mm) / ru))
        r0, r1 = max(0, int((box[1] - margin_mm) / rv)), min(nv, int((box[3] + margin_mm) / rv))
        win = h[r0:r1, c0:c1]
        ring = np.ones(win.shape, bool)
        k = int(round(margin_mm / ru))
        ring[k:-k, k:-k] = False
        inner = win[k:-k, k:-k]
        ref = float(np.median(win[ring & (win > np.percentile(win, 20))])) if ring.any() else 0.0
        out.append({"id": d["id"], "cls": "scratch", "kind": d["kind"], "box_mm": box,
                    "depth_mm": float(ref - inner.min()), "param_depth_mm": d["params"].get("depth_mm"),
                    "width_mm": d["params"].get("width_mm"), "measure": (d.get("footprint") or {}).get("measure"),
                    "near": d.get("near")})
    return out


# ----------------------------------------------------------------------------- depth coupons

COUPON_SEEDS = {"ms_hot_rolled": 910000, "aluminium": 910002, "ms_cold_rolled": 910003,
                "ss_brushed": 910007, "galvanised": 910011, "ss_2b": 910062}

# (cls, kind, d_mm, depth_mm, extra) laid out on a 4 x 4 grid of 15 mm pitch round the plate centre
COUPON_SET = [
    ("dent", "ball", 3.0, 0.10, {}), ("dent", "ball", 4.0, 0.25, {}), ("dent", "ball", 5.0, 0.40, {}),
    ("dent", "ball", 6.0, 0.15, {}), ("dent", "ball", 8.0, 0.60, {}), ("dent", "ball", 10.0, 0.30, {}),
    ("dent", "smooth", 6.0, 0.20, {"aspect": 2.0}), ("dent", "smooth", 8.0, 0.35, {"aspect": 1.5}),
    ("dent", "smooth", 5.0, 0.12, {"aspect": 1.8}), ("dent", "punch", 1.5, 0.30, {"rim_h_mm": 0.03}),
    ("dent", "punch", 2.5, 0.50, {"rim_h_mm": 0.04}), ("pit", "single", 0.4, 0.10, {}),
    ("pit", "single", 0.6, 0.25, {}), ("pit", "single", 0.8, 0.40, {}), ("pit", "single", 1.2, 0.30, {}),
    ("pit", "single", 1.5, 0.50, {}),
]


def coupon_defects(plan: Mapping[str, Any], seed: int, pitch_mm: float = 15.0) -> list[dict]:
    """The COUPON_SET placed on a 4 x 4 grid (pitch_mm) round the plate centre, shuffled per seed,
    with random angles and a +-15 % depth jitter; footprints from the sampler (finish_defect)."""
    from ayas.twin import sampler
    rng = np.random.default_rng(seed)
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    cells = [(W / 2 + (i - 1.5) * pitch_mm, H / 2 + (j - 1.5) * pitch_mm) for j in range(4) for i in range(4)]
    order = rng.permutation(len(COUPON_SET))
    out = []
    for n, (idx, (u, v)) in enumerate(zip(order, cells), 1):
        cls, kind, d, depth, extra = COUPON_SET[idx]
        depth = float(depth * rng.uniform(0.85, 1.15))
        p = {"u": round(u + rng.uniform(-1, 1), 4), "v": round(v + rng.uniform(-1, 1), 4), "d_mm": d,
             "depth_mm": round(depth, 4), **extra}
        if kind == "smooth":
            p["angle_deg"] = round(float(rng.uniform(0, 180)), 3)
        if cls == "pit":
            p.update({"halo_mm": 0.0, "halo": 0.0, "seed": int(rng.integers(2 ** 31))})
        dd = {"id": f"D{n}", "cls": cls, "kind": kind, "params": p, "label": True}
        out.append(sampler.finish_defect(dd, plan))
    return out


def per_led_lights(plan: Mapping[str, Any]) -> list[dict]:
    """The plan's ring LEDs with group E00.. (one Blender light group per LED); 'orig_group' keeps L/R."""
    out = []
    i = 0
    for L in plan["lights"]:
        L2 = dict(L)
        if L.get("mount") == "ring" and L.get("group") in ("L", "R"):
            L2["orig_group"] = L["group"]
            L2["group"] = LED_GROUP_FMT.format(i)
            i += 1
        out.append(L2)
    return out


def coupon_plan(material_kind: str, base_seed: int | None = None, cfg: Mapping[str, Any] | None = None,
                roi_margin_mm: float = 6.0) -> dict:
    """A hood (ring), dark-room plan of a 100 x 100 mm plate of this material with the coupon
    defects, no distractors, one light group per ring LED and the render ROI cut to the defects."""
    from ayas.twin import frames as F
    from ayas.twin import sampler
    cfg = copy.deepcopy(cfg or twin_cfg())
    seed = int(base_seed if base_seed is not None else COUPON_SEEDS[material_kind])
    plan = sampler.make_plan(cfg, seed, "test", condition="C0_dark", device_mode="hood")
    if plan["plate"]["material"]["kind"] != material_kind:
        raise ValueError(f"seed {seed} gives {plan['plate']['material']['kind']}, not {material_kind}")
    plan["scene_id"] = f"coupon_{material_kind}_{seed}"
    plan["defects"] = coupon_defects(plan, seed)
    plan["distractors"] = []
    plan["lights"] = per_led_lights(plan)
    plan["render"] = dict(plan.get("render") or {}, clean_pass=False, split_ambient=False)
    # ROI: the projected box of the defect area plus a margin (smaller EXR, faster render)
    cam = plan["camera"]
    K, R, t = (np.asarray(cam[k], float) for k in ("K", "R", "t"))
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    u0, u1 = W / 2 - 22.5 - roi_margin_mm - 6, W / 2 + 22.5 + roi_margin_mm + 6
    v0, v1 = H / 2 - 22.5 - roi_margin_mm - 6, H / 2 + 22.5 + roi_margin_mm + 6
    corners = F.project(K, R, t, F.plate_to_world([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]))
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    full = cam.get("full_res") or [F.IMX500["width"], F.IMX500["height"]]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(int(full[0]), int(x1)), min(int(full[1]), int(y1))
    cam["roi"] = [x0, y0, x1 - x0, y1 - y0]
    return plan


def prepare_coupon(work: str | Path, plan: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> Path:
    """Write plan.json, blender_cam.json and the texture maps into work (render with twin.render.render_batch)."""
    from ayas.twin import render, textures
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    render.prepare(work, dict(plan), cfg or twin_cfg())
    textures.write_maps(dict(plan), work)
    return work


def sum_groups(groups: Mapping[str, np.ndarray], members: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    """{new group: sum of member passes}."""
    out = {}
    for g, mem in members.items():
        acc = None
        for m in mem:
            acc = groups[m].astype(np.float32) if acc is None else acc + groups[m]
        if acc is not None:
            out[g] = acc
    return out


def led_partitions(plan: Mapping[str, Any], n_list: Sequence[int] = (2, 3, 4, 6)) -> dict[int, dict[str, list[str]]]:
    """Ring partitions as {n: {sector: [LED groups]}} for a per-LED plan (per_led_lights).

    n = 2 is the device's L/R (camera x < 0 / >= 0, as lighting.ring_lights); n >= 3 are equal
    azimuth sectors about the optical-axis plate point in the camera frame, the first centred on
    camera +x (image right)."""
    from ayas.twin import lighting
    cam = plan["camera"]
    R, t = np.asarray(cam["R"], float), np.asarray(cam["t"], float)
    c = lighting.axis_point(dict(plan))
    leds = [L for L in plan["lights"] if str(L.get("group", "")).startswith("E")]
    out: dict[int, dict[str, list[str]]] = {}
    for n in n_list:
        part: dict[str, list[str]] = {}
        for L in leds:
            if n == 2:
                key = L.get("orig_group") or ("L" if (R @ np.asarray(L["pos_m"]) + t)[0] < 0 else "R")
            else:
                az = O.ring_azimuth_deg(L, R, t, c)
                k = int(((az + 180.0 / n) % 360.0) // (360.0 / n))
                key = f"S{n}_{k}"
            part.setdefault(key, []).append(L["group"])
        out[n] = dict(sorted(part.items()))
    return out


def lights_for_partition(plan: Mapping[str, Any], part: Mapping[str, Sequence[str]]) -> list[dict]:
    """Plan light dicts regrouped by a partition (for optics / estimate_depth meta)."""
    byg = {L["group"]: L for L in plan["lights"]}
    out = []
    for g, mem in part.items():
        for m in mem:
            out.append(dict(byg[m], group=g))
    return out


def save_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
