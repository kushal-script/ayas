"""Scene sampler: every random decision of one twin scene, written as a plan JSON
(DESIGN 3, DESIGN_STEEL A-D; schema trinetra.twin.scene/2).

make_plan(cfg, seed, split) builds a scene from numpy Generators seeded by the scene seed, so
a plan plus its seed reproduces the scene. Each part draws from its own child stream of
SeedSequence(seed), so a change in one part does not reshuffle the others:
- pick (mode, source, empty, counts), coupon (plate, material, holes), defects, distractors,
  penetrant: the PLATE. These never depend on the lighting condition or the device mode;
- extras (background, gloves, render), device (device mode, hood), camera (pose), lights (pod
  power), lighting (condition, ambient), capture (shake, LED jitter): the VIEW.
make_plan(..., condition=, device_mode=) only overrides the drawn choice; every draw still
happens, so the same seed under another condition or device mode keeps the plate, defects and
distractors identical and changes only ambient, extras.device/hood, camera (and the device
LEDs that move with it) and capture; a device_mode override also changes extras.gloves (none
with the hood). The lighting test matrix (twin/matrix.py) relies on this.

Sources:
- random: a procedural steel or metal plate (DESIGN_STEEL B: material kind, size, thickness),
  an optional row of punched holes or slots (unlabelled part features), labelled defects
  (scratch, dent, pit, corrosion, crack, stain) and unlabelled distractors. The aero fastener
  patterns (lap sheet, rivets, rule violations) are still built when cfg names them;
- replica:<coupon id>: a jittered copy of a physical coupon from configs/coupons.yaml, drawn
  only from sampler.replica_coupons. Never P5: it is the held-out real test plate.

Material decides corrosion: rust and rust streaks only on mild steel (ms_*), white corrosion
only on aluminium and galvanised; stainless gets no corrosion class.

Defects live on the exposed plate face only (textures cover the plate front face, not the
sheet tops): at least 0.5 mm inside the plate and away from sheets, and clear of every visible
fastener (hole) outline, except nicks, edge cracks and edge rust streaks (on a plate edge) and
cracks, rust streaks, galvanic rings and fretting streaks that start at a hole. A quarter of
the compact defects are placed 0.5-5 mm from a plate or hole edge (the +1 severity band).

Angles in params are degrees from +u toward +v (clockwise on the front face).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import frames as F
from . import geometry as G
from . import rig as rig_mod

try:                                  # twin/lighting.py (lighting-render workstream)
    from . import lighting
except Exception:                     # not written yet or broken: _lighting() re-imports and raises
    lighting = None

SCHEMA = "trinetra.twin.scene/2"
REPO_ROOT = Path(__file__).resolve().parents[3]

STEELS = ("ms_hot_rolled", "ms_cold_rolled")
STAINLESS = ("ss_brushed", "ss_2b")
MATERIAL_WEIGHTS = {"ms_hot_rolled": 0.2, "ms_cold_rolled": 0.25, "ss_brushed": 0.15, "ss_2b": 0.1,
                    "galvanised": 0.1, "aluminium": 0.2}
CLASS_WEIGHTS = {"scratch": 0.22, "dent": 0.18, "pit": 0.15, "corrosion": 0.18, "crack": 0.12, "stain": 0.15}
KIND_WEIGHTS = {
    "scratch": {"scriber": 0.5, "gouge": 0.25, "abrasion": 0.25},
    "dent": {"punch": 0.3, "ball": 0.3, "smooth": 0.25, "nick": 0.15},
    "pit": {"single": 0.5, "cluster": 0.5},
    "corrosion": {"rust": 0.65, "rust_streak": 0.35, "white": 1.0},     # filtered by CORRODES
    "crack": {"open": 0.75, "slit": 0.15, "fatigue": 0.1},
    "stain": {"oil": 0.45, "water": 0.3, "fingerprint": 0.25},
}
# corrosion kind -> material kinds it may appear on (DESIGN_STEEL A/B)
CORRODES = {"rust": STEELS, "rust_streak": STEELS, "white": ("aluminium", "galvanised"),
            "pitting": ("aluminium", "galvanised"), "galvanic": ("aluminium", "galvanised"),
            "fretting": ("aluminium", "galvanised")}
NEAR_KINDS = ("punch", "ball", "smooth", "single", "cluster", "rust", "white", "oil", "water", "fingerprint")
NEAR_FRAC = 0.25      # share of compact defects placed 0.5-5 mm from a plate or hole edge
NEAR_GAP_MM = (0.5, 5.0)
CLUSTER_COVER = 0.08  # pit.cluster: least pit-disc area / patch-circle area (r_mm shrinks to keep it)
PATTERNS = {"none": 0.85, "holes_row": 0.15}     # steel default; aero: single, double, staggered, perimeter
AERO_PATTERNS = ("single", "double", "staggered", "perimeter")
DISTRACTORS = ("marker", "dust", "blend", "burr", "label_residue")
DEVICE_MODES = ("open", "hood")
EDGES = ("top", "bottom", "left", "right")
INWARD_DEG = {"top": 90.0, "bottom": 270.0, "left": 0.0, "right": 180.0}
ALUMINIUM = (0.91, 0.92, 0.92)
STEEL = (0.50, 0.51, 0.53)
# aluminium finishes: roughness, anisotropy, grain amplitude (um); inside 0.08-0.45 and 0-0.8
FINISHES = {"brushed": ((0.12, 0.35), (0.4, 0.8), (0.3, 1.0)),
            "mill": ((0.2, 0.45), (0.1, 0.4), (0.15, 0.6)),
            "clad": ((0.08, 0.25), (0.0, 0.2), (0.05, 0.3))}
ALU_ROUGHNESS = (0.12, 0.4)          # DESIGN_STEEL B
# steel and metal kinds (DESIGN_STEEL B): textures finish, linear grey, roughness, metallic,
# anisotropy, grain RMS (um), RGB tint of the grey
MATERIALS = {
    "ms_hot_rolled": {"finish": "hot_rolled", "grey": (0.08, 0.18), "rough": (0.45, 0.8), "metal": (0.3, 0.7),
                      "aniso": (0.0, 0.1), "grain_um": (1.0, 2.5), "tint": (0.93, 0.98, 1.08)},
    "ms_cold_rolled": {"finish": "cold_rolled", "grey": (0.45, 0.6), "rough": (0.15, 0.35), "metal": (1.0, 1.0),
                       "aniso": (0.1, 0.4), "grain_um": (0.2, 0.6), "tint": (0.98, 0.99, 1.02)},
    "ss_brushed": {"finish": "brushed", "grey": (0.55, 0.65), "rough": (0.1, 0.3), "metal": (1.0, 1.0),
                   "aniso": (0.5, 0.8), "grain_um": (0.3, 1.0), "tint": (1.0, 0.99, 0.975)},
    "ss_2b": {"finish": "2b", "grey": (0.5, 0.6), "rough": (0.25, 0.4), "metal": (1.0, 1.0),
              "aniso": (0.0, 0.15), "grain_um": (0.1, 0.4), "tint": (1.0, 0.995, 0.985)},
    "galvanised": {"finish": "spangle", "grey": (0.55, 0.7), "rough": (0.2, 0.4), "metal": (1.0, 1.0),
                   "aniso": (0.0, 0.2), "grain_um": (0.2, 0.6), "tint": (0.99, 1.0, 1.015)},
}
HOOD_POSE = {"distance_mm": (199.0, 201.0), "tilt_deg": 1.5, "target_frac": 0.05}   # DESIGN_STEEL C
MARKER_COLORS = ([0.02, 0.02, 0.025], [0.02, 0.04, 0.30], [0.02, 0.18, 0.04], [0.40, 0.02, 0.03])
GLOVE_COLORS = ([0.05, 0.12, 0.45], [0.03, 0.03, 0.03], [0.80, 0.80, 0.80])
TEXTURES = {"normal": "normal.png", "albedo": "albedo.png", "roughness": "rough.png",
            "metallic": "metal.png", "uv_emit": None}
STREAMS = ("pick", "coupon", "defects", "distractors", "penetrant", "camera", "lights", "extras",
           "device", "lighting", "capture")       # new streams go at the end: earlier ones keep their seeds
TRIES = 40
SHEET_GAP = 0.5       # mm between a defect outline and a sheet or the plate edge
FASTENER_GAP = 0.3    # mm between a free defect outline and a visible fastener outline
DEFECT_GAP = 0.5      # mm between two defect outlines


# ----------------------------------------------------------------------------- small helpers
def scene_id(seed: int) -> str:
    return "s%06d" % seed


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _u(rng, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _logu(rng, lo: float, hi: float) -> float:
    """Log-uniform in [lo, hi]: small sizes as common per octave as large ones."""
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _lighting():
    """The twin/lighting.py module (ambient_for, hood_for, CONDITIONS)."""
    global lighting
    if lighting is None:
        from . import lighting as mod          # raises the real error if it is missing or broken
        lighting = mod
    return lighting


def _pick(rng, weights: dict[str, float]) -> str:
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _tint(rng, base, spread: float = 0.03) -> list[float]:
    k = _u(rng, 1.0 - spread, 1.0 + spread / 3)
    return [_r(min(1.0, max(0.0, c * k + rng.normal(0.0, spread / 4))), 4) for c in base]


def _streams(seed: int) -> dict[str, np.random.Generator]:
    return {k: np.random.default_rng(s) for k, s in zip(STREAMS, np.random.SeedSequence(seed).spawn(len(STREAMS)))}


def _by_id(items: list[dict], value: str, key: str = "id") -> dict | None:
    return next((x for x in items if x.get(key) == value), None)


def _edge_point(edge: str, s: float, c: float, W: float, H: float) -> tuple[float, float]:
    """Point at s along an edge and c inward from it."""
    return {"top": (s, c), "bottom": (s, H - c), "left": (c, s), "right": (W - c, s)}[edge]


def _edge_len(edge: str, W: float, H: float) -> float:
    return W if edge in ("top", "bottom") else H


def kelvin_rgb(kelvin: float) -> list[float]:
    """Linear RGB (max channel 1) of a black body, from Tanner Helland's sRGB fit."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    srgb = np.clip(np.array([r, g, b]) / 255.0, 0.0, 1.0)
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return [_r(c, 4) for c in lin / lin.max()]


# ----------------------------------------------------------------------------- coupon
def _grain_dir(aniso_rotation: float) -> float:
    """Grain direction in the plate frame for a Blender anisotropic rotation (1 = 360 deg).

    Blender turns the tangent (+X = +u) counter-clockwise about +Z, i.e. from +u toward -v.
    """
    return _r((-360.0 * aniso_rotation) % 180.0, 3)


def _metal(rng, finish: str, base=ALUMINIUM, rough: tuple[float, float] | None = None) -> dict:
    rr, ar, gr = FINISHES[finish]
    if rough is not None:                              # intersect with the material's roughness band
        rr = (max(rr[0], rough[0]), min(rr[1], rough[1]))
    rot = ((0.0 if rng.random() < 0.5 else 0.25) + _u(rng, -0.03, 0.03)) % 1.0
    return {"base_color": _tint(rng, base), "roughness": _r(_u(rng, *rr)), "anisotropy": _r(_u(rng, *ar)),
            "aniso_rotation": _r(rot), "grain_amp_um": _r(_u(rng, *gr)), "grain_dir_deg": _grain_dir(rot)}


def _material(rng, kind: str) -> tuple[dict, str]:
    """(plate.material, plate.finish) of a material kind (DESIGN_STEEL B)."""
    if kind == "aluminium":
        finish = str(rng.choice(list(FINISHES)))
        mat = _metal(rng, finish, ALUMINIUM, ALU_ROUGHNESS)
        return {"kind": kind, **mat, "metallic": 1.0}, finish
    if kind not in MATERIALS:
        raise ValueError(f"unknown plate material {kind!r} (DESIGN_STEEL B: {sorted(MATERIALS) + ['aluminium']})")
    m = MATERIALS[kind]
    grey = _u(rng, *m["grey"])
    color = [_r(min(1.0, grey * t * (1 + rng.normal(0.0, 0.01))), 4) for t in m["tint"]]
    rot = ((0.0 if rng.random() < 0.5 else 0.25) + _u(rng, -0.03, 0.03)) % 1.0   # rolling / brushing direction
    mat = {"kind": kind, "base_color": color, "roughness": _r(_u(rng, *m["rough"])),
           "metallic": _r(_u(rng, *m["metal"])), "anisotropy": _r(_u(rng, *m["aniso"])), "aniso_rotation": _r(rot),
           "grain_amp_um": _r(_u(rng, *m["grain_um"])), "grain_dir_deg": _grain_dir(rot)}
    if kind == "ms_hot_rolled":                        # mill scale: mottling, thick flakes, spalled spots
        mat["scale"] = {"mottle": _r(_u(rng, 0.12, 0.3)), "cell_mm": _r(_u(rng, 2.0, 8.0)),
                        "flake_frac": _r(_u(rng, 0.05, 0.35)), "spall_frac": _r(_u(rng, 0.0, 0.02))}
    elif kind == "galvanised":                         # zinc spangle crystals
        mat["spangle"] = {"cell_mm": _r(_u(rng, 1.5, 8.0)), "contrast": _r(_u(rng, 0.03, 0.08))}
    return mat, m["finish"]


def _size(rng, sizes: dict[str, float]) -> list[float]:
    """Plate size [W, H] from sampler.sizes_mm weights keyed 'WxH'."""
    w, h = _pick(rng, sizes).lower().split("x")
    return [float(w), float(h)]


def _plate(rng, cfg: dict) -> dict:
    sc = cfg["sampler"]
    kind = _pick(rng, sc.get("materials") or MATERIAL_WEIGHTS)
    size = _size(rng, sc.get("sizes_mm") or {"100x100": 1.0})
    t = _u(rng, *(sc.get("thickness_mm") or (3.0, 3.0)))
    bevel = _u(rng, 0.1, 0.5)
    mat, finish = _material(rng, kind)
    return {"size_mm": size, "thickness_mm": _r(t, 3), "edge_bevel_mm": _r(min(bevel, 0.3 * t), 3),
            "finish": finish, "material": mat}


def _sheet_rect(edge: str, ws: float, W: float, H: float) -> list[float]:
    return {"top": [0.0, 0.0, W, ws], "bottom": [0.0, H - ws, W, ws],
            "left": [0.0, 0.0, ws, H], "right": [W - ws, 0.0, ws, H]}[edge]


def _row(edge: str, c: float, e: float, pitch: float, W: float, H: float, offset: float = 0.0) -> list:
    """Hole centres of a row parallel to edge, c inward from it, edge distance e at both ends."""
    L = _edge_len(edge, W, H) - 2 * e - 2 * offset
    if L < 0:
        return []
    n = int(math.floor(L / pitch + 1e-9))
    s0 = e + offset + (L - n * pitch) / 2
    return [_edge_point(edge, s0 + i * pitch, c, W, H) for i in range(n + 1)]


def _perimeter(e: float, pitch: float, W: float, H: float) -> list[list]:
    rows = []
    corners = [(e, e), (W - e, e), (W - e, H - e), (e, H - e)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        L = math.dist(a, b)
        n = max(1, int(round(L / pitch)))
        rows.append([(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n)])
    return rows


def _min_spacing_ok(holes: list[dict], k: int, min_c: float) -> bool:
    h = holes[k]
    return all(math.hypot(h["u"] - o["u"], h["v"] - o["v"]) >= min_c for j, o in enumerate(holes) if j != k)


def _layout_rows(rng, pattern: str, D: float, W: float, H: float, sheet_edge: str | None,
                 ws: float) -> tuple[list[list], float, float]:
    """Rows of hole centres for a pattern; returns (rows, pitch, edge distance)."""
    e, pitch = _u(rng, 2.0, 3.0) * D, _u(rng, 4.0, 6.0) * D
    if pattern == "perimeter":
        return _perimeter(e, pitch, W, H), pitch, e
    gap = _u(rng, 2.5, 4.0) * D
    if sheet_edge:                                     # rows inside the lap sheet, parallel to its edge
        edge = sheet_edge
        if pattern != "single" and ws - gap >= 4.0 * D:
            cs = [(ws - gap) / 2, (ws + gap) / 2]
        else:
            pattern, cs = "single", [ws / 2]
    else:
        edge = "top" if rng.random() < 0.5 else "left"
        across = H if edge == "top" else W
        span = gap if pattern != "single" else 0.0
        if rng.random() < 0.6:
            c0 = e if rng.random() < 0.5 else across - e - span
        else:
            c0 = _u(rng, 0.3 * across, max(0.3 * across, 0.7 * across - span))
        cs = [c0] if pattern == "single" else [c0, c0 + gap]
    rows = [_row(edge, cs[0], e, pitch, W, H)]
    if len(cs) > 1:
        rows.append(_row(edge, cs[1], e, pitch, W, H, offset=pitch / 2 if pattern == "staggered" else 0.0))
    return [r for r in rows if r], pitch, e


def _violate_edge(rng, holes: list[dict], D: float, W: float, H: float) -> dict | None:
    for k in rng.permutation(len(holes)):
        h = dict(holes[int(k)])
        dist = {"left": h["u"], "right": W - h["u"], "top": h["v"], "bottom": H - h["v"]}
        edge = min(dist, key=dist.get)
        c = _u(rng, 1.0, 1.8) * D
        s = h["u"] if edge in ("top", "bottom") else h["v"]
        h["u"], h["v"] = _edge_point(edge, s, c, W, H)
        trial = holes[: int(k)] + [h] + holes[int(k) + 1:]
        if _min_spacing_ok(trial, int(k), 2.05 * D):
            holes[int(k)] = h
            return {"kind": "edge_distance", "holes": [h["id"]], "edge": edge, "value_D": _r(c / D, 3)}
    return None


def _violate_pitch(rng, holes: list[dict], D: float) -> dict | None:
    rows: dict[int, list[int]] = {}
    for i, h in enumerate(holes):
        rows.setdefault(h["row"], []).append(i)
    pairs = [(r[j], r[j + 1]) for r in rows.values() for j in range(len(r) - 1)]
    for k in rng.permutation(len(pairs)):
        a, b = pairs[int(k)]
        ha, hb = holes[a], dict(holes[b])
        d = np.array([hb["u"] - ha["u"], hb["v"] - ha["v"]])
        s = _u(rng, 2.0, 2.8) * D
        hb["u"], hb["v"] = (float(ha["u"] + d[0] / np.linalg.norm(d) * s), float(ha["v"] + d[1] / np.linalg.norm(d) * s))
        trial = holes[:b] + [hb] + holes[b + 1:]
        if _min_spacing_ok(trial, b, 2.0 * D - 1e-6):
            holes[b] = hb
            return {"kind": "pitch", "holes": [ha["id"], hb["id"]], "value_D": _r(s / D, 3)}
    return None


def _rivet(rng, rid: str, hole: dict, D: float, screw: bool) -> dict:
    tilt = _u(rng, 3.0, 8.0) if rng.random() < 0.05 else _u(rng, 0.0, 2.0)
    if screw:
        mat = {"base_color": _tint(rng, STEEL, 0.08), "roughness": _r(_u(rng, 0.3, 0.5))}
        return {"id": rid, "hole": hole["id"], "kind": "screw", "head_d_mm": 7.0, "head_h_mm": _r(0.65 * D, 3),
                "mandrel_d_mm": 0.0, "tilt_deg": _r(tilt, 3), "tilt_dir_deg": _r(_u(rng, 0, 360), 2), "material": mat}
    mat = {"base_color": _tint(rng, (0.86, 0.87, 0.88)), "roughness": _r(_u(rng, 0.25, 0.45))}
    return {"id": rid, "hole": hole["id"], "kind": "blind_rivet", "head_d_mm": _r(2.0 * D, 3), "head_h_mm": _r(0.32 * D, 3),
            "mandrel_d_mm": _r(0.55 * D, 3), "tilt_deg": _r(tilt, 3), "tilt_dir_deg": _r(_u(rng, 0, 360), 2),
            "material": mat}


def _random_coupon(rng, plan: dict, cfg: dict, empty: bool) -> None:
    """Plate, holes (and for aero patterns lap sheet and rivets) and the fastener block of a random plate."""
    sc = cfg["sampler"]
    plate = _plate(rng, cfg)
    W, H = plate["size_mm"]
    fd = sc["fastener_d_mm"]
    w = np.asarray(fd["weights"], float)
    D = float(fd["choices"][int(rng.choice(len(w), p=w / w.sum()))])
    pattern = _pick(rng, sc.get("fastener_patterns") or PATTERNS)
    plan.update(plate=plate, sheets=[], holes=[], rivets=[])
    plan["fasteners"] = {"pattern": "none", "D_mm": None, "pitch_mm": None, "edge_mm": None, "violations": []}
    if empty or pattern == "none":
        return
    if pattern == "holes_row":
        _holes_row(rng, plan, D, W, H)
        return
    if pattern not in AERO_PATTERNS:
        raise ValueError(f"unknown fastener pattern {pattern!r}")
    _riveted(rng, plan, pattern, D, W, H)


def _holes_row(rng, plan: dict, D: float, W: float, H: float) -> None:
    """A row of punched holes, or of slots along the row, parallel to one edge: unlabelled part
    features (Blender cuts them; no rivets, no rule violations)."""
    edge = str(rng.choice(EDGES))
    slot = bool(rng.random() < 0.25)
    ov = _u(rng, 1.8, 3.0) if slot else 1.0
    d = D + 0.1
    major = d * ov
    pitch = max(_u(rng, 3.0, 6.0) * D, major + 2.0 * D)
    e_across = _u(rng, 2.0, 3.0) * D
    e_along = max(e_across, major / 2 + D)
    across = H if edge in ("top", "bottom") else W
    c = e_across if rng.random() < 0.7 else _u(rng, e_across, max(e_across, 0.4 * across))
    pts = _row(edge, c, e_along, pitch, W, H)
    ang = 0.0 if edge in ("top", "bottom") else 90.0
    plan["holes"] = [{"id": f"H{k + 1}", "u": _r(u), "v": _r(v), "d_mm": _r(d, 3), "ovality": _r(ov, 3),
                      "angle_deg": ang, "sheet": None, "filled": False, "row": 0,
                      "kind": "slot" if slot else "punched"} for k, (u, v) in enumerate(pts)]
    plan["fasteners"] = {"pattern": "holes_row", "D_mm": D, "pitch_mm": _r(pitch, 3), "edge_mm": _r(e_across, 3),
                         "violations": []}


def _riveted(rng, plan: dict, pattern: str, D: float, W: float, H: float) -> None:
    """Aero fastener patterns (DESIGN 3): optional lap sheet, rivet rows, deliberate rule violations."""
    want_sheet = rng.random() < 0.25
    sheet_edge, ws = None, 0.0
    if want_sheet:
        sheet_edge = str(rng.choice(EDGES))
        ws = _u(rng, 25.0, 40.0)
        plan["sheets"].append({"id": "S1", "rect_mm": [_r(x, 3) for x in _sheet_rect(sheet_edge, ws, W, H)],
                               "thickness_mm": 1.0, "edge": sheet_edge,
                               "material": _metal(rng, str(rng.choice(list(FINISHES))))})
        if pattern == "perimeter":
            pattern = "single"
    rows, pitch, e = _layout_rows(rng, pattern, D, W, H, sheet_edge, ws)
    holes = [{"id": "", "u": float(u), "v": float(v), "d_mm": _r(D + 0.1, 3), "ovality": 1.0, "angle_deg": 0.0,
              "sheet": "S1" if sheet_edge else None, "filled": True, "row": i}
             for i, row in enumerate(rows) for (u, v) in row]
    for k, h in enumerate(holes):
        h["id"] = f"H{k + 1}"
    violations = []
    if holes and rng.random() < 0.10:
        violations.append(_violate_edge(rng, holes, D, W, H))
    if holes and rng.random() < 0.10:
        violations.append(_violate_pitch(rng, holes, D))
    if holes and rng.random() < 0.30:
        n = min(len(holes), int(rng.integers(1, 3)))
        ids = [holes[int(k)]["id"] for k in rng.choice(len(holes), n, replace=False)]
        for h in holes:
            if h["id"] in ids:
                h["filled"] = False
        violations.append({"kind": "missing", "holes": ids})
    if holes and rng.random() < 0.10:
        open_ = [h for h in holes if not h["filled"]]
        h = open_[int(rng.integers(len(open_)))] if open_ else holes[int(rng.integers(len(holes)))]
        if h["filled"]:
            h["filled"] = False
            violations.append({"kind": "missing", "holes": [h["id"]]})
        h["ovality"], h["angle_deg"] = _r(_u(rng, 1.1, 1.4), 3), _r(_u(rng, 0, 180), 2)
        violations.append({"kind": "oval", "holes": [h["id"]], "value": h["ovality"]})
    for h in holes:
        h["u"], h["v"] = _r(h["u"]), _r(h["v"])
    plan["holes"] = holes
    rivets = []
    for h in holes:
        if h["filled"]:
            rivets.append(_rivet(rng, f"R{len(rivets) + 1}", h, D, screw=rng.random() < 0.10))
    plan["rivets"] = rivets
    plan["fasteners"] = {"pattern": pattern, "D_mm": D, "pitch_mm": _r(pitch, 3), "edge_mm": _r(e, 3),
                         "violations": [v for v in violations if v]}


# ----------------------------------------------------------------------------- placement checks
def _sheet_polys(plan: dict) -> list[np.ndarray]:
    out = []
    for s in plan.get("sheets", []):
        x, y, w, h = s["rect_mm"]
        out.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float))
    return out


def _gap_at_least(a: np.ndarray, b: np.ndarray, need: float) -> bool:
    """poly_distance(a, b) >= need, with a cheap bounding-box shortcut."""
    gx = max(b[:, 0].min() - a[:, 0].max(), a[:, 0].min() - b[:, 0].max())
    gy = max(b[:, 1].min() - a[:, 1].max(), a[:, 1].min() - b[:, 1].max())
    if max(gx, gy) >= need:
        return True
    return G.poly_distance(a, b) >= need


def _placement_ok(polys: list, plan: dict, placed: list[dict], edge: str | None = None,
                  own_hole: str | None = None) -> bool:
    """Footprint fits: SHEET_GAP inside every plate edge except `edge` (the one a nick, edge
    crack or edge rust streak starts on, which it may touch), clear of sheets, of every
    fastener but own_hole, and DEFECT_GAP from the placed defects."""
    if not polys:
        return False
    P = [np.asarray(p, float) for p in polys]
    W, H = plan["plate"]["size_mm"]
    m = {e: 0.0 if e == edge else SHEET_GAP for e in EDGES}
    allp = np.vstack(P)
    if allp[:, 0].min() < m["left"] - 1e-6 or allp[:, 1].min() < m["top"] - 1e-6 \
            or allp[:, 0].max() > W - m["right"] + 1e-6 or allp[:, 1].max() > H - m["bottom"] + 1e-6:
        return False
    for sp in _sheet_polys(plan):
        if not all(_gap_at_least(p, sp, SHEET_GAP) for p in P):
            return False
    for hid, outline in G.fastener_outlines(plan):
        if hid != own_hole and not all(_gap_at_least(p, outline, FASTENER_GAP) for p in P):
            return False
    for d in placed:
        for q in d["footprint"]["polys_mm"]:
            if not all(_gap_at_least(p, np.asarray(q, float), DEFECT_GAP) for p in P):
                return False
    return True


def _exposed_point(rng, plan: dict, margin: float) -> tuple[float, float]:
    W, H = plan["plate"]["size_mm"]
    return (_u(rng, margin, max(margin, W - margin)), _u(rng, margin, max(margin, H - margin)))


def _free_holes(plan: dict, filled: bool | None = None) -> list[dict]:
    """Holes not under a lap sheet (their surroundings are plate face), optionally by state."""
    return [h for h in plan.get("holes", []) if not h.get("sheet") and (filled is None or h["filled"] == filled)]


# ----------------------------------------------------------------------------- defect params
def _start(rng, plan: dict) -> dict:
    """Start of a slit or fatigue crack: a free hole's visible edge (60 %) or a plate edge."""
    W, H = plan["plate"]["size_mm"]
    holes = _free_holes(plan)
    if holes and rng.random() < 0.6:
        open_ = [h for h in holes if not h["filled"]]
        pool = open_ if open_ and rng.random() < 0.8 else holes
        h = pool[int(rng.integers(len(pool)))]
        ang = _u(rng, 0.0, 360.0)
        r = G.outline_radius_at(h, G.rivet_for(h, plan), ang) - 0.01   # just inside the inscribed outline
        p = np.array([h["u"], h["v"]]) + r * G.unit(ang)
        return {"start": "hole", "hole": h["id"], "u0": float(p[0]), "v0": float(p[1]), "angle_deg": ang}
    edge = str(rng.choice(EDGES))
    s = _u(rng, 3.0, _edge_len(edge, W, H) - 3.0)
    u0, v0 = _edge_point(edge, s, 0.0, W, H)
    return {"start": "edge", "edge": edge, "u0": u0, "v0": v0, "angle_deg": (INWARD_DEG[edge] + _u(rng, -30, 30)) % 360}


def _centre(rng, plan: dict, extent: float, near: bool = False) -> tuple[float, float]:
    """Centre of a compact defect of outline radius <= extent.

    near: its outline sits NEAR_GAP_MM (edge to edge, at least) from a free hole (40 % when
    the plate has holes) or a plate edge, the +1 severity band; else anywhere on the face.
    """
    if not near:
        return _exposed_point(rng, plan, extent + SHEET_GAP)
    W, H = plan["plate"]["size_mm"]
    gap = _u(rng, *NEAR_GAP_MM)
    holes = _free_holes(plan)
    if holes and rng.random() < 0.4:
        h = holes[int(rng.integers(len(holes)))]
        ang = _u(rng, 0.0, 360.0)
        r = G.outline_radius_at(h, G.rivet_for(h, plan), ang) + gap + extent
        p = np.array([h["u"], h["v"]]) + r * G.unit(ang)
        return float(p[0]), float(p[1])
    edge = str(rng.choice(EDGES))
    L = _edge_len(edge, W, H)
    return _edge_point(edge, _u(rng, extent + SHEET_GAP, max(extent + SHEET_GAP, L - extent - SHEET_GAP)),
                       gap + extent, W, H)


def _dent(rng, plan: dict, kind: str, near: bool = False) -> dict:
    if kind == "nick":
        W, H = plan["plate"]["size_mm"]
        edge = str(rng.choice(EDGES))
        d = _u(rng, 1.0, 3.0)
        u, v = _edge_point(edge, _u(rng, d + 1.0, _edge_len(edge, W, H) - d - 1.0), 0.0, W, H)
        return {"u": u, "v": v, "edge": edge, "d_mm": d, "depth_mm": _u(rng, 0.2, 0.5 * d)}
    if kind == "punch":
        d = _u(rng, 0.5, 2.5)
        p = {"d_mm": d, "depth_mm": _u(rng, 0.05, min(0.6, 0.5 * d)), "rim_h_mm": _u(rng, 0.02, 0.05)}
    elif kind == "ball":
        d = _u(rng, 3.0, 12.0)
        p = {"d_mm": d, "depth_mm": _u(rng, 0.1, min(0.8, 0.12 * d))}
    else:
        d = _u(rng, 5.0, 40.0)
        p = {"d_mm": d, "depth_mm": _u(rng, 0.1, min(1.5, 0.06 * d)), "aspect": _u(rng, 1.0, 2.0),
             "angle_deg": _u(rng, 0.0, 180.0)}
    p["u"], p["v"] = _centre(rng, plan, p["d_mm"] / 2, near)
    return p


def _scratch(rng, plan: dict, kind: str, near: bool = False) -> dict:
    if kind == "abrasion":
        u, v = _exposed_point(rng, plan, 5.0)
        return {"u": u, "v": v, "w_mm": _u(rng, 5, 15), "h_mm": _u(rng, 5, 15), "angle_deg": _u(rng, 0, 180),
                "n_lines": int(rng.integers(20, 201)), "depth_mm": _u(rng, 0.002, 0.02)}
    L = _u(rng, 5, 60) if kind == "scriber" else _u(rng, 3, 25)
    p0 = np.array(_exposed_point(rng, plan, 1.0))
    t = G.unit(_u(rng, 0, 360))
    n = np.array([-t[1], t[0]])
    bend = rng.normal(0.0, 0.06 * L, 2)
    ctrl = [p0, p0 + t * L / 3 + n * bend[0], p0 + t * 2 * L / 3 + n * bend[1], p0 + t * L]
    if kind == "scriber":
        w, depth = _u(rng, 0.05, 0.5), _u(rng, 0.005, 0.08)
    else:
        w, depth = _u(rng, 0.5, 2.0), _u(rng, 0.1, 0.5)
    return {"ctrl_mm": [[float(x), float(y)] for x, y in ctrl], "width_mm": w, "depth_mm": depth,
            "burr": _u(rng, 0, 1)}


def _jagged(rng, start, angle_deg: float, length: float, step=(0.2, 0.5), drift: float = 5.0,
            jag: float = 20.0) -> list[list[float]]:
    """Jagged crack path from start: steps of `step` mm, heading random-walking by `drift` deg and
    each step turned by up to +-2 `jag` deg about it."""
    main, pts, done = angle_deg, [[float(start[0]), float(start[1])]], 0.0
    while done < length - 1e-9:
        s = min(_u(rng, *step), length - done)
        main += rng.normal(0.0, drift)
        ang = main + float(np.clip(rng.normal(0.0, jag), -2 * jag, 2 * jag))
        q = np.array(pts[-1]) + s * G.unit(ang)
        pts.append([float(q[0]), float(q[1])])
        done += s
    return pts


def _crack_start(rng, plan: dict, margin: float) -> dict:
    """Start of an open crack: a plate edge (40 %), a free hole's edge (20 % with holes) or a free
    point heading roughly toward the plate centre."""
    W, H = plan["plate"]["size_mm"]
    r = rng.random()
    holes = _free_holes(plan)
    if 0.4 <= r < 0.6 and holes:
        h = holes[int(rng.integers(len(holes)))]
        ang = _u(rng, 0.0, 360.0)
        rr = G.outline_radius_at(h, G.rivet_for(h, plan), ang) - 0.01
        p = np.array([h["u"], h["v"]]) + rr * G.unit(ang)
        return {"start": "hole", "hole": h["id"], "u0": float(p[0]), "v0": float(p[1]),
                "angle_deg": (ang + _u(rng, -30, 30)) % 360}
    if r < 0.4:
        edge = str(rng.choice(EDGES))
        s = _u(rng, 3.0, _edge_len(edge, W, H) - 3.0)
        u0, v0 = _edge_point(edge, s, 0.0, W, H)
        return {"start": "edge", "edge": edge, "u0": u0, "v0": v0,
                "angle_deg": (INWARD_DEG[edge] + _u(rng, -30, 30)) % 360}
    u0, v0 = _exposed_point(rng, plan, margin)
    to_c = math.degrees(math.atan2(H / 2 - v0, W / 2 - u0))
    return {"start": "free", "u0": u0, "v0": v0, "angle_deg": (to_c + _u(rng, -60, 60)) % 360}


def _crack(rng, plan: dict, kind: str, near: bool = False) -> dict:
    if kind == "open":
        L = _logu(rng, 3.0, 40.0)
        w = 0.05 + 0.35 * float(rng.random()) ** 2          # mostly hairlines, some open up to 0.4 mm
        st = _crack_start(rng, plan, 1.0)
        out = {k: st[k] for k in ("start", "hole", "edge") if k in st}
        out.update(pts_mm=_jagged(rng, (st["u0"], st["v0"]), st["angle_deg"], L, (0.3, 1.0), 6.0, 12.0),
                   width_mm=w, seed=int(rng.integers(2**31)))
        return out
    st = _start(rng, plan)
    if kind == "slit":
        st.update(length_mm=_u(rng, 1, 10), width_mm=_u(rng, 0.25, 0.6))
        return st
    L = _u(rng, 1.0, 8.0)
    out = {k: st[k] for k in ("start", "hole", "edge") if k in st}
    out.update(pts_mm=_jagged(rng, (st["u0"], st["v0"]), st["angle_deg"], L), width_mm=_u(rng, 0.01, 0.08))
    return out


def _halo(rng, plan: dict) -> tuple[float, float]:
    """(halo_mm, strength) of the rust halo round a pit: mild steel only, 60 % of pits."""
    if plan["plate"]["material"].get("kind") in STEELS and rng.random() < 0.6:
        return _u(rng, 0.15, 0.8), _u(rng, 0.4, 1.0)
    return 0.0, 0.0


def _pit(rng, plan: dict, kind: str, near: bool = False) -> dict:
    if kind == "single":
        d = 0.2 + 1.3 * float(rng.random()) ** 1.5
        depth = _u(rng, 0.05, min(0.5, max(0.08, 1.2 * d)))
        halo, k = _halo(rng, plan)
        u, v = _centre(rng, plan, d / 2 + halo, near)
        return {"u": u, "v": v, "d_mm": d, "depth_mm": depth, "halo_mm": halo, "halo": k,
                "seed": int(rng.integers(2**31))}
    n = int(rng.integers(3, 31))
    dmin = _u(rng, 0.1, 0.5)
    dmax = _u(rng, dmin + 0.1, 1.2)
    # a cluster is a dense group: the pits cover >= CLUSTER_COVER of the patch circle, else most
    # of its label box is clean steel (post's p90 visibility test then drops it)
    r_hi = (dmin + dmax) / 4 * math.sqrt(n / CLUSTER_COVER)
    r = _u(rng, 1.5, min(8.0, max(1.5, r_hi)))
    halo, k = _halo(rng, plan)
    u, v = _centre(rng, plan, r + halo, near)
    return {"u": u, "v": v, "r_mm": r, "n": n, "pit_d_mm": [dmin, dmax], "depth_mm": _u(rng, 0.05, 0.4),
            "halo_mm": halo, "halo": k, "pits": _pits(rng, u, v, r, n, dmin, dmax), "seed": int(rng.integers(2**31))}


def _rust_streak(rng, plan: dict) -> dict:
    """Rust bleed streak from a plate edge (inward), a free hole (outward) or a rust spot."""
    W, H = plan["plate"]["size_mm"]
    L, w = _u(rng, 5.0, 40.0), _u(rng, 1.0, 6.0)
    holes = _free_holes(plan)
    src = _pick(rng, {"edge": 0.45, "hole": 0.15 if holes else 0.0, "spot": 0.4})
    p: dict[str, Any] = {"source": src, "length_mm": L, "width_mm": w}
    if src == "edge":
        edge = str(rng.choice(EDGES))
        p["u0"], p["v0"] = _edge_point(edge, _u(rng, w, _edge_len(edge, W, H) - w), 0.0, W, H)
        p.update(edge=edge, angle_deg=(INWARD_DEG[edge] + _u(rng, -25, 25)) % 360)
    elif src == "hole":
        h = holes[int(rng.integers(len(holes)))]
        ang = _u(rng, 0.0, 360.0)
        q = np.array([h["u"], h["v"]]) + (G.outline_radius_at(h, G.rivet_for(h, plan), ang) - 0.01) * G.unit(ang)
        p.update(hole=h["id"], u0=float(q[0]), v0=float(q[1]), angle_deg=(ang + _u(rng, -20, 20)) % 360)
    else:
        head = w / 2 * _u(rng, 1.0, 1.6)
        p["u0"], p["v0"] = _exposed_point(rng, plan, head + SHEET_GAP)
        p.update(angle_deg=_u(rng, 0.0, 360.0), head_r_mm=head)
    p.update(fade=_u(rng, 0.3, 0.9), seed=int(rng.integers(2**31)))
    return p


def _corrosion(rng, plan: dict, kind: str, near: bool = False) -> dict | None:
    W, H = plan["plate"]["size_mm"]
    if kind == "rust":
        r = _logu(rng, 2.0, min(25.0, 0.35 * min(W, H)))
        u, v = _centre(rng, plan, r, near)
        return {"u": u, "v": v, "r_mm": r, "coverage": _u(rng, 0.3, 1.0), "pitted": bool(rng.random() < 0.35),
                "aspect": _u(rng, 1.0, 2.0), "angle_deg": _u(rng, 0.0, 180.0), "seed": int(rng.integers(2**31))}
    if kind == "rust_streak":
        return _rust_streak(rng, plan)
    if kind in ("white", "pitting"):           # white = the aero pitting look (powder + pits)
        r = _u(rng, 1.5, 10.0 if kind == "white" else 8.0)
        u, v = _centre(rng, plan, r, near) if kind == "white" else _exposed_point(rng, plan, r + SHEET_GAP)
        dmin = _u(rng, 0.1, 0.4)
        dmax = _u(rng, dmin + 0.1, 1.0)
        n = int(rng.integers(5, 41))
        pits = _pits(rng, u, v, r, n, dmin, dmax)
        return {"u": u, "v": v, "r_mm": r, "n_pits": n, "pit_d_mm": [dmin, dmax], "powder": _u(rng, 0, 1),
                "pits": pits, "seed": int(rng.integers(2**31))}
    if kind == "galvanic":
        holes = _free_holes(plan, filled=True)
        if not holes:
            return None
        kinds = {r["hole"]: r.get("kind") for r in plan["rivets"]}
        w = np.array([4.0 if kinds.get(h["id"]) == "screw" else 1.0 for h in holes])
        h = holes[int(rng.choice(len(holes), p=w / w.sum()))]
        return {"hole": h["id"], "ring_w_mm": _u(rng, 1, 4), "powder": _u(rng, 0, 1), "seed": int(rng.integers(2**31))}
    free = {h["id"] for h in _free_holes(plan, filled=True)}
    rivets = [r for r in plan.get("rivets", []) if r["hole"] in free and r.get("kind") != "screw"]
    if not rivets:
        return None
    rv = rivets[int(rng.integers(len(rivets)))]
    return {"rivet": rv["id"], "angle_deg": _u(rng, 0, 360), "length_mm": _u(rng, 3, 15),
            "width_mm": _u(rng, 1.0, min(4.0, 0.9 * rv["head_d_mm"])), "darkness": _u(rng, 0.3, 0.9)}


def _stain(rng, plan: dict, kind: str, near: bool = False) -> dict:
    """Labelled stains (DESIGN_STEEL A): oil film, dried water ring, fingerprint."""
    if kind == "oil":
        r = _logu(rng, 2.0, 15.0)
        p = {"r_mm": r, "aspect": _u(rng, 1.0, 1.8), "angle_deg": _u(rng, 0.0, 180.0), "gloss": _u(rng, 0.3, 1.0),
             "strength": _u(rng, 0.5, 1.0)}
    elif kind == "water":
        r = _u(rng, 2.0, 12.0)
        p = {"r_mm": r, "ring_w_mm": _u(rng, 0.1, 0.5), "strength": _u(rng, 0.4, 1.0), "minerals": _u(rng, 0.2, 1.0)}
    else:
        r = _u(rng, 5.0, 9.0)
        p = {"r_mm": r, "angle_deg": _u(rng, 0.0, 180.0), "aspect": _u(rng, 1.25, 1.5), "strength": _u(rng, 0.4, 1.0)}
    p["u"], p["v"] = _centre(rng, plan, r, near)
    p["seed"] = int(rng.integers(2**31))
    return p


def _pits(rng, u: float, v: float, r: float, n: int, dmin: float, dmax: float) -> list[list[float]]:
    """[u, v, d] of n pits spread uniformly inside the patch circle."""
    pits = []
    for _ in range(n):
        d = _u(rng, dmin, dmax)
        rho, a = max(r - d / 2, 0.0) * math.sqrt(rng.random()), _u(rng, 0, 2 * math.pi)
        pits.append([_r(u + rho * math.cos(a)), _r(v + rho * math.sin(a)), _r(d)])
    return pits


GENERATORS = {"dent": _dent, "scratch": _scratch, "crack": _crack, "corrosion": _corrosion, "pit": _pit,
              "stain": _stain}


def _round_params(p: dict) -> dict:
    def rr(x):
        if isinstance(x, bool) or x is None or isinstance(x, (str, int)):
            return x
        if isinstance(x, float):
            return _r(x)
        if isinstance(x, (list, tuple)):
            return [rr(y) for y in x]
        if isinstance(x, dict):
            return {k: rr(v) for k, v in x.items()}
        return x
    return rr(p)


def _own_hole(defect: dict, plan: dict) -> tuple[str | None, str | None]:
    """(the plate edge it starts on and may touch, id of the hole it starts at or attaches to).

    Only that one edge: a streak or crack that would run off through another edge (near a
    corner) is redrawn, so its length_mm stays the length on the plate.
    """
    p, kind = defect["params"], defect["kind"]
    if kind == "nick":
        return p.get("edge"), None
    if kind in ("slit", "fatigue", "open"):
        return (p.get("edge") if p.get("start") == "edge" else None), p.get("hole")
    if kind == "rust_streak":
        return (p.get("edge") if p.get("source") == "edge" else None), p.get("hole")
    if kind == "galvanic":
        return None, p["hole"]
    if kind == "fretting":
        rv = _by_id(plan["rivets"], p["rivet"])
        return None, rv["hole"] if rv else None
    return None, None


def finish_defect(defect: dict, plan: dict) -> dict:
    """Add footprint, near and label to a defect with params."""
    defect["footprint"] = G.footprint(defect, plan)
    defect["near"] = G.near(defect["footprint"]["polys_mm"], plan)
    defect.setdefault("label", True)
    return defect


def _place(rng, plan: dict, cls: str, kind: str, placed: list[dict]) -> dict | None:
    """Draw params until the footprint fits (TRIES); NEAR_FRAC of compact kinds try near an edge first."""
    near_first = kind in NEAR_KINDS and rng.random() < NEAR_FRAC
    for near in ((True, False) if near_first else (False,)):
        for _ in range(TRIES):
            params = GENERATORS[cls](rng, plan, kind, near)
            if params is None:
                return None
            d = {"id": "", "cls": cls, "kind": kind, "params": _round_params(params)}
            d["footprint"] = G.footprint(d, plan)
            edge, own = _own_hole(d, plan)
            if _placement_ok(d["footprint"]["polys_mm"], plan, placed, edge, own):
                return finish_defect(d, plan)
    return None


def kind_weights(cfg: dict, cls: str, material: str | None) -> dict[str, float]:
    """Kinds of a class allowed on a material (CORRODES), with their weights (sampler.kind_weights overrides)."""
    w = dict((cfg["sampler"].get("kind_weights") or {}).get(cls) or KIND_WEIGHTS[cls])
    if cls == "corrosion":
        w = {k: x for k, x in w.items() if material in CORRODES.get(k, ())}
    return {k: x for k, x in w.items() if x > 0}


def _defects(rng, plan: dict, n: int, cfg: dict, need_crack: bool = False) -> list[dict]:
    material = plan["plate"]["material"].get("kind")
    kinds = {c: kind_weights(cfg, c, material) for c in KIND_WEIGHTS}
    cw = {c: x for c, x in (cfg["sampler"].get("class_weights") or CLASS_WEIGHTS).items() if kinds.get(c)}
    todo = []
    for _ in range(n):
        cls = _pick(rng, cw)
        todo.append((cls, _pick(rng, kinds[cls])))
    if need_crack and not any(c == "crack" for c, _ in todo):
        todo.append(("crack", _pick(rng, kinds["crack"])))
    placed: list[dict] = []
    for cls, kind in todo:
        d = _place(rng, plan, cls, kind, placed)
        if d is None and kind in ("galvanic", "fretting"):
            d = _place(rng, plan, "corrosion", "pitting", placed)
        if d is not None:
            placed.append(d)
    for k, d in enumerate(placed):
        d["id"] = f"D{k + 1}"
    return placed


# ----------------------------------------------------------------------------- distractors
def _distractor(rng, plan: dict, kind: str) -> dict | None:
    if kind == "burr":
        holes = _free_holes(plan, filled=False)
        if not holes:
            return None
        h = holes[int(rng.integers(len(holes)))]
        return {"hole": h["id"], "start_deg": _u(rng, 0, 360), "arc_deg": _u(rng, 60, 360),
                "h_mm": _u(rng, 0.05, 0.3), "w_mm": _u(rng, 0.1, 0.5)}
    u, v = _exposed_point(rng, plan, 2.0)
    p: dict[str, Any] = {"u": u, "v": v}
    if kind == "fingerprint":
        p.update(r_mm=_u(rng, 5, 9), angle_deg=_u(rng, 0, 180), strength=_u(rng, 0.3, 1.0))
    elif kind == "oil":
        p.update(r_mm=_u(rng, 2, 8), aspect=_u(rng, 1.0, 1.6), angle_deg=_u(rng, 0, 180), strength=_u(rng, 0.3, 1.0))
    elif kind == "water_spot":
        p.update(r_mm=_u(rng, 1, 4), ring_w_mm=_u(rng, 0.1, 0.4), strength=_u(rng, 0.3, 1.0))
    elif kind == "marker":
        p.update(r_mm=_u(rng, 0.8, 2.5), color=MARKER_COLORS[int(rng.integers(len(MARKER_COLORS)))],
                 strength=_u(rng, 0.6, 1.0))
    elif kind == "dust":
        p.update(r_mm=_u(rng, 5, 30), n=int(rng.integers(20, 301)), size_mm=[0.02, _u(rng, 0.05, 0.2)])
    elif kind == "blend":
        d = _u(rng, 4, 20)
        p.update(d_mm=d, aspect=_u(rng, 1.0, 2.0), angle_deg=_u(rng, 0, 180), depth_mm=d / 40.0)
    elif kind == "label_residue":            # sticker glue left on the part: a hard negative for stain
        p.update(w_mm=_u(rng, 8, 30), h_mm=_u(rng, 5, 20), angle_deg=_u(rng, 0, 180), strength=_u(rng, 0.3, 1.0))
    p["seed"] = int(rng.integers(2**31))
    return p


def _distractor_ok(kind: str, p: dict, plan: dict, defects: list[dict]) -> bool:
    if kind == "burr":
        return True
    c = np.array([[p["u"], p["v"]]])
    for _, outline in G.fastener_outlines(plan):
        if G.point_in_polygon(c, outline)[0]:
            return False
    for sp in _sheet_polys(plan):
        if G.point_in_polygon(c, sp)[0]:
            return False
    if kind == "blend":                      # a rework dish must not sit on a labelled defect
        disc = G.circle(c[0], p["d_mm"] / 2 + 0.5)
        return all(_gap_at_least(disc, np.asarray(q, float), 0.0 + 1e-9)
                   for d in defects for q in d["footprint"]["polys_mm"])
    outline = _cover_outline(kind, p)
    if outline is not None:                  # opaque paint or glue would hide (relabel) a defect
        return all(_gap_at_least(outline, np.asarray(q, float), DEFECT_GAP)
                   for d in defects if d.get("label", True) for q in d["footprint"]["polys_mm"])
    return True


def _cover_outline(kind: str, p: dict) -> np.ndarray | None:
    """Outline of a distractor that paints over what is below it (textures mix albedo, roughness
    and metallic toward its own): a marker dot, label residue (its ragged glue edge reaches up
    to ~1.17 x the half-sizes, textures._label_residue). Dust specks may lie on defects."""
    if kind == "marker":
        return G.circle((p["u"], p["v"]), p["r_mm"])
    if kind == "label_residue":
        return G.rect((p["u"], p["v"]), 1.2 * p["w_mm"], 1.2 * p["h_mm"], p["angle_deg"])
    return None


def _distractors(rng, plan: dict, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        kind = str(rng.choice(DISTRACTORS))
        for _ in range(TRIES):
            p = _distractor(rng, plan, kind)
            if p is None:
                break
            if _distractor_ok(kind, p, plan, plan.get("defects", [])):
                out.append({"kind": kind, "params": _round_params(p)})
                break
    return out


# ----------------------------------------------------------------------------- penetrant
def _indication(iid: str, cls: str, params: dict, plan: dict, from_defect: str | None) -> dict:
    ind = {"id": iid, "cls": cls, "from_defect": from_defect, "params": _round_params(params)}
    ind["footprint"] = G.footprint(ind, plan)
    return ind


def _penetrant(rng, plan: dict) -> dict:
    """Developer coat and red-dye indications from cracks and pits, plus random rounded ones."""
    dev = {"albedo": _r(_u(rng, 0.82, 0.92)), "roughness": _r(_u(rng, 0.85, 0.95)),
           "speckle": _r(_u(rng, 0.02, 0.08)), "thickness_var": _r(_u(rng, 0.05, 0.2))}
    inds: list[dict] = []
    for d in plan.get("defects", []):
        p = d["params"]
        if d["kind"] in ("slit", "fatigue", "open"):
            if d["kind"] == "slit":
                end = np.array([p["u0"], p["v0"]]) + p["length_mm"] * G.unit(p["angle_deg"])
                pts = [[p["u0"], p["v0"]], [float(end[0]), float(end[1])]]
            else:
                pts = p["pts_mm"]
            params = {"pts_mm": pts, "width_mm": p["width_mm"], "bloom_mm": _u(rng, 0.3, 1.5),
                      "intensity": _u(rng, 0.5, 1.0)}
            inds.append(_indication(f"I{len(inds) + 1}", "linear_indication", params, plan, d["id"]))
        elif d["kind"] in ("pitting", "white", "cluster", "single"):
            pits = [[p["u"], p["v"], p["d_mm"]]] if d["kind"] == "single" else p.get("pits", [])
            pits = sorted(pits, key=lambda q: -q[2])[:6]
            for (u, v, pd) in pits:
                params = {"u": u, "v": v, "d_mm": float(np.clip(pd * _u(rng, 1.2, 2.0), 0.3, 1.0)),
                          "intensity": _u(rng, 0.4, 1.0)}
                inds.append(_indication(f"I{len(inds) + 1}", "rounded_indication", params, plan, d["id"]))
    for _ in range(int(rng.integers(0, 3))):
        u, v = _exposed_point(rng, plan, 2.0)
        params = {"u": u, "v": v, "d_mm": _u(rng, 0.3, 1.0), "intensity": _u(rng, 0.3, 0.9)}
        inds.append(_indication(f"I{len(inds) + 1}", "rounded_indication", params, plan, None))
    return {"developer": dev, "indications": inds, "edge_bleed": _r(_u(rng, 0.0, 0.5)),
            "dye_color": [0.55, 0.05, 0.06]}


# ----------------------------------------------------------------------------- camera, lights, extras
def camera_K(cfg: dict) -> np.ndarray:
    cam = cfg["camera"]
    if cam.get("K"):
        return np.asarray(cam["K"], float)
    w, h = cam["full_res"]
    return F.nominal_K({"width": w, "height": h, "pixel_um": cam["pixel_um"], "focal_mm": cam["focal_mm"]})


def scene_points_world(plan: dict) -> np.ndarray:
    """World points bounding what the ROI must show: plate corners, sheet tops, rivet head tops."""
    W, H = plan["plate"]["size_mm"]
    pts = [F.plate_to_world([[0, 0], [W, 0], [W, H], [0, H]])]
    sheets = {s["id"]: s for s in plan.get("sheets", [])}
    for s in sheets.values():
        x, y, w, h = s["rect_mm"]
        pts.append(F.plate_to_world([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], s["thickness_mm"]))
    for rv in plan.get("rivets", []):
        hole = _by_id(plan["holes"], rv["hole"])
        top = rv["head_h_mm"] + (sheets[hole["sheet"]]["thickness_mm"] if hole.get("sheet") in sheets else 0.0)
        pts.append(F.plate_to_world(G.circle((hole["u"], hole["v"]), rv["head_d_mm"] / 2, 16), top))
    return np.vstack(pts)


def hood_pose(cfg: dict) -> dict:
    """Pose limits with the hood (DESIGN_STEEL C): distance 200 +- 1 mm, tilt +-1.5 deg, target
    within target_frac of the plate centre; sampler.hood overrides."""
    return {**HOOD_POSE, **(cfg["sampler"].get("hood") or {})}


def _pose(rng, W: float, H: float, sc: dict, hood: dict | None) -> tuple[list[float], float, float, float, float, bool]:
    """(target_mm, distance, tilt_x, tilt_y, roll, wide) of one handheld (hood None) or hood pose."""
    roll_j = sc["roll_jitter_deg"]
    if hood is None:
        dist, tilt = sc["distance_mm"], sc["tilt_deg"]
        target = [_u(rng, 0.2 * W, 0.8 * W), _u(rng, 0.2 * H, 0.8 * H)]
        wide = bool(rng.random() < dist["wide_frac"])
        distance = _u(rng, *dist["wide" if wide else "sharp"])
    else:                                       # the hood rests on the part or bench: near-straight, fixed range
        tilt, f = hood["tilt_deg"], hood["target_frac"]
        target = [_u(rng, (0.5 - f) * W, (0.5 + f) * W), _u(rng, (0.5 - f) * H, (0.5 + f) * H)]
        wide = False
        distance = _u(rng, *hood["distance_mm"])
    tx, ty = _u(rng, -tilt, tilt), _u(rng, -tilt, tilt)
    roll = _u(rng, -roll_j, roll_j) + 90.0 * int(rng.integers(0, 4))
    return target, distance, tx, ty, roll, wide


def _camera(rng, plan: dict, cfg: dict, device_mode: str = "open") -> dict:
    cam, sc = cfg["camera"], cfg["sampler"]
    K = camera_K(cfg)
    Ws, Hs = cam["full_res"]
    W, H = plan["plate"]["size_mm"]
    corners = F.plate_to_world([[0, 0], [W, 0], [W, H], [0, H]])
    hood = hood_pose(cfg) if device_mode == "hood" else None
    for _ in range(200):
        target, distance, tx, ty, roll, wide = _pose(rng, W, H, sc, hood)
        R, t = F.camera_pose(target, distance, tx, ty, roll)
        px = F.project(K, R, t, corners)
        if (px[:, 0] >= 0).all() and (px[:, 0] <= Ws - 1).all() and (px[:, 1] >= 0).all() and (px[:, 1] <= Hs - 1).all():
            break
    else:
        raise RuntimeError("no camera pose keeps the plate on the sensor")
    q = F.project(K, R, t, scene_points_world(plan))
    m = cam["roi_margin_px"]
    x0, y0 = max(0, int(math.floor(q[:, 0].min() - m))), max(0, int(math.floor(q[:, 1].min() - m)))
    x1, y1 = min(Ws, int(math.ceil(q[:, 0].max() + m)) + 1), min(Hs, int(math.ceil(q[:, 1].max() + m)) + 1)
    return {"K": [[_r(x, 6) for x in row] for row in K], "full_res": [int(Ws), int(Hs)],
            "pixel_um": cam["pixel_um"],
            "R": [[float(x) for x in row] for row in R], "t": [float(x) for x in t],
            "roi": [x0, y0, x1 - x0, y1 - y0], "fstop": cam["fstop"], "focus_m": cam["focus_mm"] / 1000.0,
            "pose": {"target_mm": [_r(x, 3) for x in target], "distance_mm": _r(distance, 3),
                     "tilt_deg": [_r(tx, 3), _r(ty, 3)], "roll_deg": _r(roll, 3), "wide": wide}}


_RIG_CACHE: dict[tuple, rig_mod.Rig] = {}


def get_rig(cfg: dict, root: Path) -> rig_mod.Rig:
    """The rig of cfg, cached. The white LED power comes from the capture config's lumens
    (rig.white_params), so the capture file and its lumens are part of the cache key."""
    rc = cfg["rig"]
    lumens = (capture_config(cfg, root).get("leds") or {}).get("white_lumens_per_led")
    key = (str(Path(root) / rc["fusion_generator"]), bool(rc.get("model_x_is_image_left", True)),
           json.dumps(rc.get("white")), json.dumps(rc.get("uv")),
           str(Path(root) / (cfg.get("capture_config") or "configs/capture.yaml")), json.dumps(lumens))
    if key not in _RIG_CACHE:
        _RIG_CACHE[key] = rig_mod.load_rig(cfg, Path(root))
    return _RIG_CACHE[key]


def _lights(rng, plan: dict, cfg: dict, root: Path) -> list[dict]:
    """The device LEDs in world coordinates (per-pod power 0.85-1.15). With the ring
    (extras.device.illumination 'ring', device v2) the pods' white LEDs are replaced by the
    hood's dark-field ring (lighting.ring_lights, the same LED and power draws); UV stays."""
    scale = {f"{k}_{pod}": _u(rng, 0.85, 1.15) for k in ("white", "uv") for pod in ("L", "R")}
    cam = plan["camera"]
    rig = get_rig(cfg, root)
    lights = rig_mod.lights_world(rig, cam["R"], cam["t"], scale)
    if ((plan.get("extras") or {}).get("device") or {}).get("illumination") == "ring":
        white = next((led.params for led in rig.leds if led.kind == "white"), None)
        lights = [L for L in lights if L["group"] not in ("L", "R")] + _lighting().ring_lights(plan, cfg, white, scale)
    return lights


def glove_outline(g: dict) -> np.ndarray:
    """Top view of a glove finger in plate mm: the capsule twin/blender/render_scene.glove_axis
    builds (fingertip centre (u, v), running length_mm along angle_deg, radius_mm)."""
    r = float(g["radius_mm"])
    a = np.array([g["u"], g["v"]], float)
    return G.buffer_polyline([a, a + max(float(g["length_mm"]) - 2 * r, 1e-3) * G.unit(g["angle_deg"])], r)


def _gloves(rng, plan: dict, device_mode: str = "open") -> list[dict]:
    """0-2 gloved fingers holding the plate at its edges (30 % of scenes), never over a labelled
    defect. None with the hood: it sits over the part and a finger would pass through its wall.
    The draws are the same in every case, so the rest of the extras stream does not move."""
    if rng.random() >= 0.3:
        return []
    W, H = plan["plate"]["size_mm"]
    out = []
    for edge in rng.choice(EDGES, int(rng.integers(1, 3)), replace=False):
        r = _u(rng, 7, 10)
        L = _edge_len(str(edge), W, H)
        u, v = _edge_point(str(edge), _u(rng, 0.15 * L, 0.85 * L), -(r - _u(rng, 0, 4)), W, H)
        out.append({"u": _r(u, 3), "v": _r(v, 3), "angle_deg": _r((INWARD_DEG[str(edge)] + 180 + _u(rng, -25, 25)) % 360, 2),
                    "length_mm": _r(_u(rng, 50, 80), 2), "radius_mm": _r(r, 3),
                    "color": GLOVE_COLORS[int(rng.integers(len(GLOVE_COLORS)))]})
    if device_mode == "hood":
        return []
    labelled = [np.asarray(q, float) for d in plan.get("defects", []) if d.get("label", True)
                for q in d["footprint"]["polys_mm"]]
    return [g for g in out if all(_gap_at_least(glove_outline(g), q, DEFECT_GAP) for q in labelled)]


def _extras(rng, plan: dict, device_mode: str = "open") -> dict:
    """Background, gloves, device box and the device mode; the hood is added by make_plan."""
    grey = _u(rng, 0.03, 0.35)
    return {"background": {"kind": "noise", "color": [_r(grey * _u(rng, 0.9, 1.1)) for _ in range(3)],
                           "roughness": _r(_u(rng, 0.5, 0.95)), "scale": _r(_u(rng, 10, 80), 2)},
            "gloves": _gloves(rng, plan, device_mode), "device_box": True, "device": {"mode": device_mode}}


def _render(rng, cfg: dict) -> dict:
    r = cfg["render"]
    return {"samples": r["samples"], "noise_threshold": r["noise_threshold"], "max_bounces": r["max_bounces"],
            "glossy_bounces": r["glossy_bounces"], "texture_res_mm": r["texture_res_mm"],
            "clean_pass": bool(rng.random() < r.get("clean_pass_frac", 0.0))}


_CAPTURE_CACHE: dict[str, dict] = {}


def capture_config(cfg: dict, root: Path) -> dict:
    """configs/capture.yaml (cfg capture_config), cached; {} when it does not exist."""
    path = str(Path(root) / (cfg.get("capture_config") or "configs/capture.yaml"))
    if path not in _CAPTURE_CACHE:
        p = Path(path)
        _CAPTURE_CACHE[path] = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
    return _CAPTURE_CACHE[path]


def _capture(rng, cfg: dict, root: Path, device_mode: str) -> dict:
    """plan['capture'] (DESIGN_STEEL D): hand-shake random-walk step and LED gain jitter.

    shake_px_per_frame (full-resolution sensor px; burst.py scales it to the render): open
    0.2-1.5 px (5 % of scenes 3-6 px), hood 0-0.2 px (it rests on the part). The open range
    was 0.2-2.5 (5 % 3-8): with B2 - B1 spanning 4 steps that refused a third of the open
    captures for motion alone (review_post.md); a braced handheld at ~10 frames/s moves
    about 0.2-1.5 px (15-100 um) per frame. The same three draws are made in both modes, so
    only the mapping differs.
    """
    big, s, j = float(rng.random()), float(rng.random()), float(rng.random())
    if device_mode == "hood":
        shake = 0.2 * s
    elif big < 0.05:
        shake = 3.0 + 3.0 * s
    else:
        shake = 0.2 + 1.3 * s
    sensor = ((capture_config(cfg, root).get("sensor") or {}).get("mode")) or "full"
    return {"sensor": str(sensor), "shake_px_per_frame": _r(shake), "led_jitter": _r(0.05 * j)}


# ----------------------------------------------------------------------------- replicas
EXPECT_VIOLATIONS = {"edge_distance": "edge_distance", "pitch": "pitch", "missing_fastener": "missing",
                     "elongated_hole": "oval"}
SIZE_KEYS = ("d_mm", "depth_mm", "length_mm", "width_mm", "w_mm", "h_mm", "r_mm", "ring_w_mm", "rim_h_mm")
DISTRACTOR_DEFAULTS = {
    "burr": {"start_deg": 0.0, "arc_deg": 360.0, "h_mm": 0.1, "w_mm": 0.3},
    "blend": {"aspect": 1.0, "angle_deg": 0.0},
    "fingerprint": {"r_mm": 7.0, "angle_deg": 0.0, "strength": 0.7},
    "oil": {"r_mm": 4.0, "aspect": 1.0, "angle_deg": 0.0, "strength": 0.7},
    "water_spot": {"r_mm": 2.0, "ring_w_mm": 0.2, "strength": 0.7},
    "marker": {"r_mm": 1.5, "color": MARKER_COLORS[0], "strength": 0.9},
    "dust": {"r_mm": 15.0, "n": 100, "size_mm": [0.02, 0.1]},
}


def _coupon_map(data) -> dict[str, dict] | None:
    """{id: coupon dict} from the coupons file layout (or a list of coupons with ids)."""
    if isinstance(data, dict) and isinstance(data.get("coupons"), (dict, list)):
        data = data["coupons"]
    if isinstance(data, list) and all(isinstance(c, dict) and "id" in c for c in data):
        return {c["id"]: c for c in data}
    if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
        return {k: dict(v, id=v.get("id", k)) for k, v in data.items()}
    return None


def resolve_coupon(coupon: dict, raw: dict) -> dict:
    """Merge the file's defaults into a coupon: plate, holes, rivets (by rivet kind)."""
    dflt = raw.get("defaults") or {}
    c = json.loads(json.dumps(coupon))
    c["plate"] = {**(dflt.get("plate") or {}), **(c.get("plate") or {})}
    c["holes"] = [{**(dflt.get("hole") or {}), **h} for h in c.get("holes") or []]
    kinds = dflt.get("rivet") or {}
    c["rivets"] = [{**(kinds.get(r.get("kind", "blind_rivet")) or {}), **r} for r in c.get("rivets") or []]
    c.setdefault("fastener_d_mm", raw.get("fastener_d_mm"))
    if dflt.get("datum") and "datum" not in c:
        c["datum"] = dflt["datum"]
    return c


def load_coupon_table(root: Path) -> dict[str, dict]:
    """Resolved coupons by face id (P2A, ...) from configs/coupons.yaml.

    Uses trinetra.twin.coupons.load_coupons when it exists and returns dicts; otherwise
    reads the YAML directly. Defaults are merged here either way (idempotent).
    """
    path = Path(root) / "configs" / "coupons.yaml"
    raw = (yaml.safe_load(path.read_text()) if path.exists() else None) or {}
    table = None
    try:
        from . import coupons as coupons_mod                     # the coupons workstream's loader
        table = _coupon_map(coupons_mod.load_coupons(path))
    except (ImportError, AttributeError, TypeError, ValueError, KeyError):
        table = None
    table = table or _coupon_map(raw) or {}
    return {cid: resolve_coupon(c, raw) for cid, c in table.items()}


def _replica_ids(table: dict[str, dict], allowed: list[str]) -> list[str]:
    """Coupon faces of the allowed plates (P2 -> P2A, P2B); never P5 or any role: test coupon."""
    ids = sorted(cid for cid in table if any(cid == a or cid.startswith(a) for a in allowed))
    return [cid for cid in ids if not cid.upper().startswith("P5") and table[cid].get("role") != "test"]


def _rot(v, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    v = np.asarray(v, float)
    return np.array([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]]).T


def _attachment(start, coupon: dict) -> tuple[str, str | None]:
    """Where an (unjittered) crack starts: ('hole', id) on or under a fastener, ('edge', name), or ('free', None)."""
    best = None
    view = {"rivets": coupon.get("rivets") or []}
    for h in coupon.get("holes") or []:
        r = G.visible_radius(h, G.rivet_for(h, view))
        dist = math.hypot(start[0] - h["u"], start[1] - h["v"])
        if dist <= r + 0.3 and (best is None or dist < best[1]):
            best = (h["id"], dist)
    if best:
        return "hole", best[0]
    W, H = coupon["plate"]["size_mm"]
    d = {"top": start[1], "bottom": H - start[1], "left": start[0], "right": W - start[0]}
    e = min(d, key=d.get)
    return ("edge", e) if d[e] <= 0.3 else ("free", None)


def _crack_ends(kind: str, p: dict) -> tuple[np.ndarray, np.ndarray]:
    if kind == "slit":
        s = np.array([p["u0"], p["v0"]], float)
        return s, s + p["length_mm"] * G.unit(p["angle_deg"])
    pts = np.asarray(p["pts_mm"], float)
    return pts[0], pts[-1]


def _jitter_defect(rng, d: dict, plan: dict, coupon: dict, moved: dict[str, dict]) -> tuple[dict, dict]:
    """One jittered copy of a coupon defect's params: +-1 mm position, +-20 % size, cracks stay
    attached to their hole, plate edge or parent crack (grows_from). Returns (params, crack info)."""
    kind, p0 = d["kind"], d["params"]
    p = json.loads(json.dumps(p0))
    k, off, dphi = _u(rng, 0.8, 1.2), rng.uniform(-1.0, 1.0, 2), _u(rng, -5.0, 5.0)
    for key in SIZE_KEYS:
        if isinstance(p.get(key), (int, float)) and not isinstance(p.get(key), bool):
            p[key] = p[key] * k
    W, H = plan["plate"]["size_mm"]
    info: dict = {}
    if kind in ("slit", "fatigue"):
        s_old, _ = _crack_ends(kind, p0)
        parent = moved.get(d.get("grows_from") or "")
        if parent:
            dphi = parent["dphi"]
            s_new = parent["tip_new"] + _rot(s_old - parent["tip_old"], dphi)
        else:
            att, ref = _attachment(s_old, coupon)
            if att == "hole":
                h_old = _by_id(coupon["holes"], ref)
                h_new = _by_id(plan["holes"], ref)
                c_old, c_new = np.array([h_old["u"], h_old["v"]]), np.array([h_new["u"], h_new["v"]])
                s_new = c_new + _rot(s_old - c_old, dphi)
                p.setdefault("start", "hole")
                p.setdefault("hole", ref)
            elif att == "edge":
                dphi = 0.0
                along = (s_old[0] if ref in ("top", "bottom") else s_old[1]) + float(off[0])
                s_new = np.array(_edge_point(ref, along, 0.0, W, H))
                p.setdefault("start", "edge")
                p.setdefault("edge", ref)
            else:
                s_new = s_old + off
        if kind == "slit":
            p["u0"], p["v0"] = float(s_new[0]), float(s_new[1])
            p["angle_deg"] = (p0["angle_deg"] + dphi) % 360.0
        else:
            rel = np.asarray(p0["pts_mm"], float) - s_old
            p["pts_mm"] = [list(map(float, q)) for q in s_new + k * _rot(rel, dphi)]
        info = {"tip_old": _crack_ends(kind, p0)[1], "tip_new": _crack_ends(kind, p)[1], "dphi": dphi}
    elif kind == "nick":
        along = (p0["u"] if p0["edge"] in ("top", "bottom") else p0["v"]) + float(off[0])
        p["u"], p["v"] = _edge_point(p0["edge"], along, 0.0, W, H)
    elif "ctrl_mm" in p:
        ctrl = np.asarray(p0["ctrl_mm"], float)
        c = ctrl.mean(axis=0)
        p["ctrl_mm"] = [list(map(float, q)) for q in c + off + k * (ctrl - c)]
    elif "u" in p and "v" in p:
        p["u"], p["v"] = p0["u"] + float(off[0]), p0["v"] + float(off[1])
    if kind == "pitting" and "pits" not in p:
        dmin, dmax = (float(x) * k for x in p.get("pit_d_mm", [0.1, 0.5]))
        p["pit_d_mm"] = [dmin, dmax]
        p["pits"] = _pits(rng, p["u"], p["v"], p["r_mm"], int(p.get("n_pits", 20)), dmin, dmax)
        p.setdefault("seed", int(rng.integers(2**31)))
    return p, info


def _status(defect: dict, plan: dict) -> tuple[bool, bool]:
    """(clear of every fastener, clear of the plate edges)."""
    fp = G.footprint(defect, plan)
    if not fp["polys_mm"]:
        return False, False
    nr = G.near(fp["polys_mm"], plan)
    return (nr["fastener_edge_mm"] is None or nr["fastener_edge_mm"] > 0.05), nr["plate_edge_mm"] > 0.05


def _replica(rng, plan: dict, coupon: dict) -> None:
    """A jittered copy of a resolved coupon: holes +-0.3 mm, defects +-1 mm and +-20 %."""
    plate = json.loads(json.dumps(coupon["plate"]))
    plate.setdefault("thickness_mm", 3.0)
    plate.setdefault("edge_bevel_mm", 0.3)
    plate.setdefault("finish", "mill")
    plate.setdefault("material", _metal(rng, plate["finish"]))
    plate["material"].setdefault("grain_dir_deg", _grain_dir(plate["material"].get("aniso_rotation", 0.0)))
    plate["material"].setdefault("kind", "aluminium")          # the 5 Nov coupons are aluminium
    plate["material"].setdefault("metallic", 1.0)
    plan["plate"] = plate
    plan["sheets"] = []
    for s in coupon.get("sheets") or []:
        s = json.loads(json.dumps(s))
        s.setdefault("thickness_mm", 1.0)
        s.setdefault("material", _metal(rng, str(rng.choice(list(FINISHES)))))
        plan["sheets"].append(s)
    D = float(coupon.get("fastener_d_mm") or 4.0)
    holes = []
    for h in coupon.get("holes") or []:
        h = json.loads(json.dumps(h))
        h.setdefault("d_mm", D + 0.1)
        h.setdefault("ovality", 1.0)
        h.setdefault("angle_deg", 0.0)
        h.setdefault("sheet", None)
        h.setdefault("filled", True)
        h["u"], h["v"] = _r(h["u"] + _u(rng, -0.3, 0.3)), _r(h["v"] + _u(rng, -0.3, 0.3))
        holes.append(h)
    by_hole = {r["hole"]: r for r in coupon.get("rivets") or []}
    rivets = []
    for h in holes:                       # a rivet only in a filled hole; a filled hole always gets one
        rv = by_hole.get(h["id"])
        if not h["filled"]:
            continue
        base = _rivet(rng, f"R_{h['id']}", h, D, screw=bool(rv and rv.get("kind") == "screw"))
        if rv:
            base.update(json.loads(json.dumps(rv)))
            base["tilt_deg"] = _r(float(rv.get("tilt_deg", 0.0)) + _u(rng, 0.0, 1.0), 3)
        rivets.append(base)
    plan["holes"], plan["rivets"] = holes, rivets
    violations = [{"kind": EXPECT_VIOLATIONS[e["type"]], "holes": e.get("holes") or [e["hole"]]}
                  for e in coupon.get("expect") or [] if e.get("type") in EXPECT_VIOLATIONS]
    plan["fasteners"] = {"pattern": "coupon", "D_mm": D, "pitch_mm": None, "edge_mm": None, "violations": violations}
    plan["expect"] = json.loads(json.dumps(coupon.get("expect") or []))
    if coupon.get("datum"):
        plan["datum"] = json.loads(json.dumps(coupon["datum"]))
    defects, moved = [], {}
    for d0 in coupon.get("defects") or []:
        d0 = {k: v for k, v in d0.items() if k not in ("footprint", "near")}
        want = _status(d0, plan)
        for attempt in range(10):
            params, info = _jitter_defect(rng, d0, plan, coupon, moved)
            d = dict(d0, params=_round_params(params))
            if attempt == 9 or _status(d, plan) == want:
                break
        d.setdefault("id", f"D{len(defects) + 1}")
        if info:
            moved[d["id"]] = info
        defects.append(finish_defect(d, plan))
    plan["defects"] = defects
    dis = []
    for x in coupon.get("distractors") or []:
        x = json.loads(json.dumps(x))
        prm = {**DISTRACTOR_DEFAULTS.get(x["kind"], {}), **(x.get("params") or {})}
        if "u" in prm and "v" in prm:
            off = rng.uniform(-1.0, 1.0, 2)
            prm["u"], prm["v"] = prm["u"] + float(off[0]), prm["v"] + float(off[1])
        k = _u(rng, 0.8, 1.2)
        for key in ("r_mm", "d_mm", "depth_mm", "h_mm", "w_mm", "ring_w_mm"):
            if isinstance(prm.get(key), (int, float)):
                prm[key] = prm[key] * k
        prm.setdefault("seed", int(rng.integers(2**31)))
        x["params"] = _round_params(prm)
        dis.append(x)
    plan["distractors"] = dis


# ----------------------------------------------------------------------------- plans
def _choice(name: str, given: str | None, drawn: str, allowed) -> str:
    if given is None:
        return drawn
    if allowed and given not in allowed:
        raise ValueError(f"unknown {name} {given!r}; expected one of {list(allowed)}")
    return given


def make_plan(cfg: dict, seed: int, split: str, mode: str | None = None, source: str | None = None,
              coupons: dict[str, dict] | None = None, condition: str | None = None,
              device_mode: str | None = None) -> dict:
    """One scene plan (DESIGN 3, DESIGN_STEEL D). mode: surface | penetrant; source: random |
    replica | replica:<id>; condition: a lighting.CONDITIONS id; device_mode: open | hood.

    Unset choices are drawn from sampler.modes / sources / conditions / device_modes. The draws
    always happen, so passing a choice explicitly only overrides it, not the rest of the scene:
    a condition or device_mode override changes only ambient, extras.device/hood, camera (and
    the LEDs on the device), and capture; device_mode also extras.gloves (none with the hood).
    """
    rs = _streams(seed)
    sc = cfg["sampler"]
    root = Path(cfg.get("root") or REPO_ROOT)
    drawn_mode, drawn_source = _pick(rs["pick"], sc["modes"]), _pick(rs["pick"], sc["sources"])
    empty = bool(rs["pick"].random() < sc["empty_scene_frac"])
    n_def, n_dis = int(rs["pick"].integers(0, 7)), int(rs["pick"].integers(0, 5))
    mode, source = mode or drawn_mode, source or drawn_source
    lt = _lighting()
    conditions = tuple(getattr(lt, "CONDITIONS", ()) or ())
    device = _choice("device_mode", device_mode, _pick(rs["device"], sc.get("device_modes") or {"open": 1.0}),
                     DEVICE_MODES)
    condition = _choice("condition", condition,
                        _pick(rs["lighting"], sc.get("conditions") or {c: 1.0 for c in conditions}), conditions)
    plan: dict[str, Any] = {"schema": SCHEMA, "scene_id": scene_id(seed), "seed": int(seed), "split": split,
                            "mode": mode, "source": "random"}
    if source.startswith("replica"):
        table = coupons if coupons is not None else load_coupon_table(root)
        allowed = list(sc["replica_coupons"])
        ids = _replica_ids(table, allowed)
        if ":" in source:
            cid = source.split(":", 1)[1]
            if cid.upper().startswith("P5") or (table.get(cid) or {}).get("role") == "test":
                raise ValueError(f"{cid} is the held-out real test plate and is never replicated")
            if cid not in ids:
                raise ValueError(f"coupon {cid} is not in sampler.replica_coupons or configs/coupons.yaml")
        elif ids:
            cid = ids[int(rs["pick"].integers(len(ids)))]
        else:
            cid = None
        if cid is not None:
            assert not cid.upper().startswith("P5"), "P5 must never be replicated"
            _replica(rs["coupon"], plan, table[cid])
            plan["source"] = f"replica:{cid}"
    if plan["source"] == "random":
        _random_coupon(rs["coupon"], plan, cfg, empty)
        plan["empty"] = empty
        plan["defects"] = [] if empty else _defects(rs["defects"], plan, n_def, cfg, need_crack=mode == "penetrant")
        plan["distractors"] = _distractors(rs["distractors"], plan, n_dis)
    plan["penetrant"] = _penetrant(rs["penetrant"], plan) if mode == "penetrant" else None
    # the view: everything below may depend on the device mode and the condition, nothing above does
    plan["extras"] = _extras(rs["extras"], plan, device)
    plan["render"] = _render(rs["extras"], cfg)
    plan["camera"] = _camera(rs["camera"], plan, cfg, device)
    plan["extras"]["hood"] = lt.hood_for(rs["device"], plan, cfg, device)
    illum = getattr(lt, "illumination_for", None)                                # pods | ring (device v2)
    plan["extras"]["device"]["illumination"] = illum(cfg, device) if illum else "pods"
    plan["lights"] = _lights(rs["lights"], plan, cfg, root)
    plan["ambient"] = lt.ambient_for(rs["lighting"], plan, cfg, condition)
    plan["capture"] = _capture(rs["capture"], cfg, root, device)
    plan["textures"] = dict(TEXTURES)
    return plan


def split_for(cfg: dict, seed: int) -> str:
    """Train/val split of a scene, fixed by its seed (independent of the scene contents)."""
    return _pick(np.random.default_rng([int(seed), 0x5EED]), cfg["sampler"]["splits"])


def view_tag(condition: str | None = None, device_mode: str | None = None) -> str:
    """Scene-id suffix of an overridden view, as twin/matrix.row_tag: C6_sun + hood -> 'C6h'."""
    code = (condition.split("_")[0] if condition else "")
    return code + (device_mode[0] if device_mode else "")


def write_plans(cfg: dict, n: int, out_dir: str | Path, start_seed: int = 0, condition: str | None = None,
                device_mode: str | None = None, split: str | None = None) -> list[Path]:
    """Write <out>/plans/<scene_id>.json for seeds start_seed .. start_seed + n - 1.

    condition / device_mode override the view of every plan (make_plan); the scene id then
    gets the view_tag suffix (s000010_C6h) so several views of one plate can share out_dir,
    while plan['seed'] stays the plate seed. split forces the split (default split_for(seed)).
    """
    out = Path(out_dir) / "plans"
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.get("root") or REPO_ROOT)
    table = load_coupon_table(root) if cfg["sampler"]["sources"].get("replica", 0) > 0 else {}
    tag = view_tag(condition, device_mode)
    paths = []
    for seed in range(start_seed, start_seed + n):
        plan = make_plan(cfg, seed, split or split_for(cfg, seed), coupons=table, condition=condition,
                         device_mode=device_mode)
        if tag:
            plan["scene_id"] = f"{plan['scene_id']}_{tag}"
        p = out / f"{plan['scene_id']}.json"
        p.write_text(json.dumps(plan))
        paths.append(p)
    return paths
