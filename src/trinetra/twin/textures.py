"""Texture maps of the plate front face for the twin (numpy; DESIGN sections 1, 3 and 4,
DESIGN_STEEL A and B).

render_maps(plan) turns a scene plan into float maps on the texel grid of DESIGN section 1:
Nu = round(W / res) columns, Nv = round(H / res) rows, texel (r, c) centred at
u = (c + 0.5)·W/Nu, v = (r + 0.5)·H/Nv (= (c + 0.5)·res when W is a multiple of res).
write_maps(plan, work_dir) encodes them as the PNGs Blender reads.

- Height is in mm (positive out of the face); the normal map is derived from it.
- Albedo is linear RGB; roughness, metallic and UV emission are 0..1.
- Every element (finish, each defect, each distractor, the developer, each indication)
  draws from its own generator seeded by (plan seed, element key). The clean maps
  therefore differ from the normal maps only where labelled defects (and penetrant
  indications) are.
- Defects are stamped in local windows and clipped to their footprint polygons
  (dilated by CLIP_TOL_MM): textures render a defect, they never move it.
- Angles (angle_deg, grain_dir_deg) are measured from +u towards +v (image convention,
  v down), i.e. direction (cos a, sin a) in (u, v).
- Holes and rivets are Blender geometry and are never painted.
- Finishes (plate.finish): aluminium brushed / mill / clad; steel and metal hot_rolled (mill
  scale: mottled blue-grey, thick flakes, spalled spots; plate.material.scale), cold_rolled
  (faint roll marks), brushed (stainless No.4), 2b (stainless matte), spangle (galvanised zinc
  crystals: Voronoi cells, plate.material.spangle). plate.material.metallic is the mean metallic.
- Stains (oil, water, fingerprint) are LABELLED defects: the clean maps leave them out.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import geometry as G

log = logging.getLogger(__name__)

DEFAULT_RES_MM = 0.04
CLIP_TOL_MM = 0.1                       # a stamp may reach this far past its footprint polygon
# The clip mask measures distance between texel centres of the rasterised polygon, whose
# edge is off by up to ~half a texel either way. Without slack, texels 0.08 mm from the true
# outline were cut (a height step at punch rims and nick lips). With it, everything within
# CLIP_TOL_MM of the outline is kept and nothing beyond CLIP_TOL_MM + 1.5 texels.
CLIP_SLACK_PX = 0.5
PENETRANT_RED = (0.55, 0.05, 0.06)      # Wesaf kit red, linear (DESIGN 3.2), until measured
TEXTURE_FILES = {"normal": "normal.png", "albedo": "albedo.png", "roughness": "rough.png",
                 "metallic": "metal.png", "uv_emit": "uvemit.png"}
PNG_COMPRESSION = 1                     # fast; work dirs are deleted after post

# Base finishes. streaks: (across_mm, along_mm, weight) octaves of grain-aligned noise;
# iso: (cell_mm, weight) isotropic octaves; line_mod: contrast between brush lines;
# roll_lines: sparse deeper roll marks per 100 mm; wave_cell_mm: waviness scale;
# mottle: albedo/roughness mid-frequency mottling (clad).
FINISHES = {
    "brushed": {"streaks": ((0.04, 2.4, 1.0), (0.12, 9.0, 0.55), (0.5, 36.0, 0.3)), "iso": (),
                "line_mod": 0.6, "roll_lines": 0, "wave_cell_mm": 12.0, "mottle": 0.0},
    "mill": {"streaks": ((0.08, 16.0, 1.0), (0.28, 60.0, 0.8), (1.2, 200.0, 0.6)), "iso": ((0.08, 0.3),),
             "line_mod": 0.9, "roll_lines": 12, "wave_cell_mm": 15.0, "mottle": 0.0},
    "clad": {"streaks": ((0.08, 8.0, 0.35),), "iso": ((0.08, 0.6), (0.24, 1.0), (0.8, 0.6)),
             "line_mod": 0.2, "roll_lines": 0, "wave_cell_mm": 10.0, "mottle": 0.025},
    "hot_rolled": {"streaks": ((0.15, 12.0, 0.35), (0.6, 60.0, 0.25)), "iso": ((0.06, 0.7), (0.2, 1.0), (0.8, 0.6)),
                   "line_mod": 0.0, "roll_lines": 0, "wave_cell_mm": 20.0, "mottle": 0.0},
    "cold_rolled": {"streaks": ((0.06, 20.0, 1.0), (0.25, 80.0, 0.6)), "iso": ((0.08, 0.35),),
                    "line_mod": 0.5, "roll_lines": 8, "wave_cell_mm": 15.0, "mottle": 0.01},
    "2b": {"streaks": ((0.1, 30.0, 0.35),), "iso": ((0.06, 1.0), (0.2, 0.6), (0.6, 0.3)),
           "line_mod": 0.1, "roll_lines": 3, "wave_cell_mm": 12.0, "mottle": 0.02},
    "spangle": {"streaks": (), "iso": ((0.06, 0.6), (0.2, 0.4)), "line_mod": 0.0, "roll_lines": 0,
                "wave_cell_mm": 10.0, "mottle": 0.0},
}
STEEL_KINDS = ("ms_hot_rolled", "ms_cold_rolled")
SPANGLE_BINS = 6                        # grain directions of the zinc crystals


# ----------------------------------------------------------------------------- helpers

def map_shape(plan: dict) -> tuple[int, int]:
    """(Nv, Nu) of the plate texture maps."""
    w, h = (float(x) for x in plan["plate"]["size_mm"])
    res = float((plan.get("render") or {}).get("texture_res_mm", DEFAULT_RES_MM))
    return max(1, int(round(h / res))), max(1, int(round(w / res)))


def rim_width_mm(d_mm: float) -> float:
    """Width of the raised rim just outside a punch dent or nick of diameter d (<= CLIP_TOL_MM)."""
    return min(0.1, max(0.05, 0.1 * d_mm))


def scratch_reach_mm(width_mm: float, burr: float, res_mm: float = DEFAULT_RES_MM) -> float:
    """How far a scratch's texture reaches from its centre line (groove + burr ridge)."""
    he, bw = _scratch_widths(width_mm, burr, res_mm)
    return he + bw


def _scratch_widths(w: float, burr: float, res: float) -> tuple[float, float]:
    he = max(w / 2, 0.75 * res)                       # groove half-width, at least 1.5 texels wide
    bw = min(0.08, max(res, 0.3 * w)) if burr > 0 else 0.0
    return he, bw


def linear_to_srgb8(x: np.ndarray) -> np.ndarray:
    """Linear 0..1 -> 8-bit sRGB (IEC 61966-2-1 OETF)."""
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.round(s * 255.0).astype(np.uint8)


def srgb8_to_linear(x: np.ndarray) -> np.ndarray:
    s = np.asarray(x, np.float32) / 255.0
    return np.where(s <= 0.04045, s / 12.92, np.power((s + 0.055) / 1.055, 2.4)).astype(np.float32)


def encode_normal16(n: np.ndarray) -> np.ndarray:
    """Unit normals in [-1, 1] -> 16-bit (n + 1) / 2, same channel order."""
    return np.clip(np.round((np.asarray(n, np.float32) + 1.0) * 32767.5), 0, 65535).astype(np.uint16)


def normal_from_height(height: np.ndarray, res_mm: float | tuple[float, float]) -> np.ndarray:
    """Object-space normals (Nv, Nu, 3) from height in mm: n = normalize(-dh/du, +dh/dv, 1).

    World X = u and Y = -v (DESIGN 1), so -dh/dY = +dh/dv. Central differences inside,
    one-sided at the borders. res_mm is the texel size, or (du, dv) if they differ.
    """
    ru, rv = (res_mm, res_mm) if np.isscalar(res_mm) else res_mm
    h = np.asarray(height, np.float32)
    gu, gv = np.zeros_like(h), np.zeros_like(h)
    if h.shape[1] > 1:
        gu[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * ru)
        gu[:, 0], gu[:, -1] = (h[:, 1] - h[:, 0]) / ru, (h[:, -1] - h[:, -2]) / ru
    if h.shape[0] > 1:
        gv[1:-1] = (h[2:] - h[:-2]) / (2 * rv)
        gv[0], gv[-1] = (h[1] - h[0]) / rv, (h[-1] - h[-2]) / rv
    inv = 1.0 / np.sqrt(gu * gu + gv * gv + 1.0)
    return np.stack([-gu * inv, gv * inv, inv], axis=-1).astype(np.float32)


def _smooth(e0: float, e1: float, x):
    """Smoothstep from 0 at e0 to 1 at e1 (e1 < e0 gives a falling edge)."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit(x: np.ndarray) -> np.ndarray:
    s = float(x.std())
    return x / s if s > 0 else x


def _noise(rng: np.random.Generator, shape: tuple[int, int], cell_px: float) -> np.ndarray:
    """Smooth noise with features of about cell_px texels (cubic-interpolated lattice)."""
    h, w = shape
    gh, gw = max(2, int(math.ceil(h / cell_px)) + 2), max(2, int(math.ceil(w / cell_px)) + 2)
    g = rng.standard_normal((gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _fbm(rng: np.random.Generator, shape: tuple[int, int], cell_px: float, octaves: int = 4,
         gain: float = 0.55) -> np.ndarray:
    """Unit-variance fractal noise; cell_px is the coarsest feature size."""
    out, amp = np.zeros(shape, np.float32), 1.0
    for k in range(octaves):
        out += amp * _noise(rng, shape, max(1.5, cell_px / 2 ** k))
        amp *= gain
    return _unit(out)


def _noise1d(rng: np.random.Generator, x, cell: float, lo: float, hi: float) -> np.ndarray:
    """Smooth 1-D noise of x in [lo, hi] (knots every `cell`), about unit variance."""
    n = int(math.ceil((hi - lo) / cell)) + 2
    k = rng.standard_normal(n)
    q = np.clip((np.asarray(x, float) - lo) / cell, 0.0, n - 1.001)
    i = q.astype(int)
    f = q - i
    f = f * f * (3 - 2 * f)
    return k[i] * (1 - f) + k[i + 1] * f


def _angular_noise(rng: np.random.Generator, theta: np.ndarray, terms: int = 8) -> np.ndarray:
    """Smooth periodic noise of an angle, about unit variance."""
    k = np.arange(1, terms + 1)
    a = rng.standard_normal(terms) / k
    ph = rng.uniform(0, 2 * np.pi, terms)
    out = sum(a[i] * np.cos(k[i] * theta + ph[i]) for i in range(terms))
    return out / math.sqrt(float((a * a).sum()) / 2)


# ----------------------------------------------------------------------------- containers

@dataclass
class _Maps:
    h: np.ndarray        # (Nv, Nu) mm
    albedo: np.ndarray   # (Nv, Nu, 3) linear
    rough: np.ndarray
    metal: np.ndarray
    uv: np.ndarray

    def copy(self) -> _Maps:
        return _Maps(self.h.copy(), self.albedo.copy(), self.rough.copy(), self.metal.copy(), self.uv.copy())

    def result(self) -> dict[str, np.ndarray]:
        for a in (self.albedo, self.rough, self.metal, self.uv):
            np.clip(a, 0.0, 1.0, out=a)
        return {"height": self.h, "albedo": self.albedo, "roughness": self.rough,
                "metallic": self.metal, "uv_emit": self.uv}


@dataclass(frozen=True)
class _Win:
    """Texel window rows [r0, r1), columns [c0, c1)."""

    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def sl(self) -> tuple[slice, slice]:
        return slice(self.r0, self.r1), slice(self.c0, self.c1)

    @property
    def shape(self) -> tuple[int, int]:
        return self.r1 - self.r0, self.c1 - self.c0


# An op is (channel, mode, value, weight): channel in h/albedo/rough/metal/uv; mode
# "add" (x += w·value), "mix" (x += w·(value - x)) or "mul" (x *= 1 + w·(value - 1)).
Op = tuple[str, str, Any, Any]


@dataclass
class _Stamp:
    win: _Win
    ops: list[Op]
    cover: np.ndarray | None = None   # where the element acts (0..1), for footprint checks


class _Ctx:
    def __init__(self, plan: dict):
        self.plan = plan
        self.W, self.H = (float(x) for x in plan["plate"]["size_mm"])
        self.nv, self.nu = map_shape(plan)
        self.ru, self.rv = self.W / self.nu, self.H / self.nv
        self.res = max(self.ru, self.rv)
        self.seed = int(plan.get("seed", 0)) & 0xFFFFFFFFFFFFFFFF
        self.finish = plan["plate"].get("finish", "brushed")
        self.kind = (plan["plate"].get("material") or {}).get("kind")
        self.holes = {h["id"]: h for h in plan.get("holes") or []}
        self.rivets = {r["id"]: r for r in plan.get("rivets") or []}
        self.rivet_in = {r["hole"]: r for r in plan.get("rivets") or [] if r.get("hole")}

    @property
    def shape(self) -> tuple[int, int]:
        return self.nv, self.nu

    def rng(self, key: str) -> np.random.Generator:
        k = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
        return np.random.default_rng([self.seed, k])

    def window(self, u0: float, v0: float, u1: float, v1: float, pad: int = 2) -> _Win | None:
        """Texels whose centres may lie in the mm box [u0, u1] x [v0, v1], clipped to the map."""
        c0 = max(0, int(math.floor(u0 / self.ru - 0.5)) - pad)
        c1 = min(self.nu, int(math.ceil(u1 / self.ru - 0.5)) + pad + 1)
        r0 = max(0, int(math.floor(v0 / self.rv - 0.5)) - pad)
        r1 = min(self.nv, int(math.ceil(v1 / self.rv - 0.5)) + pad + 1)
        return _Win(r0, r1, c0, c1) if r1 > r0 and c1 > c0 else None

    def disc(self, u: float, v: float, r: float) -> _Win | None:
        return self.window(u - r, v - r, u + r, v + r)

    def coords(self, win: _Win) -> tuple[np.ndarray, np.ndarray]:
        """Texel-centre u (1, n) and v (m, 1) in mm, broadcastable to the window."""
        u = ((np.arange(win.c0, win.c1) + 0.5) * self.ru)[None, :]
        v = ((np.arange(win.r0, win.r1) + 0.5) * self.rv)[:, None]
        return u, v

    def fastener(self, hole_id: str) -> tuple[float, float, float]:
        """Centre and radius of a fastener's visible outline: the rivet head if filled, else the
        hole's major semi-axis (d·ovality/2), as geometry.visible_radius and the footprints use."""
        hole = self.holes[hole_id]
        r = float(hole["d_mm"]) * max(1.0, float(hole.get("ovality", 1.0))) / 2
        riv = self.rivet_in.get(hole_id)
        if hole.get("filled", riv is not None) and riv is not None:
            r = float(riv["head_d_mm"]) / 2
        return float(hole["u"]), float(hole["v"]), r


def _local(ctx: _Ctx, win: _Win, sub: _Win | None):
    """Part of `sub` inside `win`: (slices into win, its _Win) or None."""
    if sub is None:
        return None
    r0, r1, c0, c1 = max(sub.r0, win.r0), min(sub.r1, win.r1), max(sub.c0, win.c0), min(sub.c1, win.c1)
    if r1 <= r0 or c1 <= c0:
        return None
    return (slice(r0 - win.r0, r1 - win.r0), slice(c0 - win.c0, c1 - win.c0)), _Win(r0, r1, c0, c1)


def _seg_local(ctx: _Ctx, win: _Win, a, b, reach: float):
    """Distance to segment a-b and its parameter t in [0, 1], over the texels of win near it."""
    loc = _local(ctx, win, ctx.window(min(a[0], b[0]) - reach, min(a[1], b[1]) - reach,
                                      max(a[0], b[0]) + reach, max(a[1], b[1]) + reach))
    if loc is None:
        return None
    sl, sub = loc
    U, V = ctx.coords(sub)
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    t = np.clip(((U - a[0]) * dx + (V - a[1]) * dy) / l2, 0.0, 1.0) if l2 > 0 else np.zeros_like(U * V)
    d = np.hypot(U - a[0] - t * dx, V - a[1] - t * dy)
    return sl, d, np.broadcast_to(t, d.shape)


def _polyline_field(ctx: _Ctx, win: _Win, pts: np.ndarray, reach: float):
    """Distance to a polyline and arc length at the nearest point, over win (inf beyond reach)."""
    D = np.full(win.shape, np.inf)
    S = np.zeros(win.shape)
    cum = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.hypot(*(b - a)))
        got = _seg_local(ctx, win, a, b, reach)
        if got is not None:
            sl, d, t = got
            better = d < D[sl]
            np.copyto(D[sl], d, where=better)
            np.copyto(S[sl], cum + t * L, where=better)
        cum += L
    return D, S, cum


def _bbox_window(ctx: _Ctx, pts: np.ndarray, reach: float) -> _Win | None:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return ctx.window(lo[0] - reach, lo[1] - reach, hi[0] + reach, hi[1] + reach)


def _bezier(ctrl, step: float = 0.2) -> np.ndarray:
    """Dense samples of a cubic Bezier (4 points); other point counts are used as a polyline."""
    P = np.asarray(ctrl, float).reshape(-1, 2)
    if len(P) != 4:
        return P
    edge = float(np.hypot(*np.diff(P, axis=0).T).max())
    n = max(8, int(math.ceil(3 * edge / step)))     # |B'| <= 3 max edge: segments <= step
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return ((1 - t) ** 3) * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]


# ----------------------------------------------------------------------------- applying

def _apply(maps: _Maps, st: _Stamp, clip: np.ndarray | None) -> None:
    sl = st.win.sl
    for ch, mode, value, w in st.ops:
        tgt = getattr(maps, ch)[sl]
        w = np.asarray(w, np.float32)
        if clip is not None:
            w = w * clip
        val = np.asarray(value, np.float32)
        if tgt.ndim == 3:
            if w.ndim == 2:
                w = w[..., None]
            if val.ndim == 2:
                val = val[..., None]
        if mode == "add":
            tgt += w * val
        elif mode == "mix":
            tgt += w * (val - tgt)
        elif mode == "mul":
            tgt *= 1.0 + w * (val - 1.0)
        else:
            raise ValueError(f"unknown op mode {mode!r}")


def _clip_mask(ctx: _Ctx, polys, win: _Win) -> np.ndarray | None:
    """1 inside the footprint polygons dilated by CLIP_TOL_MM, else 0 (None: no polygons)."""
    pts = []
    for poly in polys or []:
        a = np.asarray(poly, float).reshape(-1, 2)
        if len(a) >= 3:
            px = np.stack([a[:, 0] / ctx.ru - 0.5 - win.c0, a[:, 1] / ctx.rv - 0.5 - win.r0], axis=1)
            pts.append(np.round(px * 16).astype(np.int32))
    if not pts:
        return None
    m = np.zeros(win.shape, np.uint8)
    for q in pts:                     # one call per polygon: a single call XORs overlapping polygons
        cv2.fillPoly(m, [q], 1, lineType=cv2.LINE_8, shift=4)
    dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (dist <= CLIP_TOL_MM / ctx.res + CLIP_SLACK_PX).astype(np.float32)


def _put(maps: _Maps, ctx: _Ctx, stamps: list[_Stamp], polys=None, name: str = "",
         uv_emit: float = 0.0) -> None:
    for st in stamps:
        if uv_emit and st.cover is not None:
            st.ops.append(("uv", "mix", float(uv_emit), st.cover))
        clip = _clip_mask(ctx, polys, st.win) if polys else None
        if clip is not None and st.cover is not None:
            tot = float(st.cover.sum())
            out = float((st.cover * (1.0 - clip)).sum())
            if tot > 0 and out > 0.02 * tot:
                log.warning("%s: %.0f%% of the texture lies outside its footprint; clipped", name, 100 * out / tot)
        _apply(maps, st, clip)


# ----------------------------------------------------------------------------- base finish

def _rotate_to_plate(src: np.ndarray, nv: int, nu: int, angle_deg: float) -> np.ndarray:
    """Resample a square field whose +x is the grain onto the plate grid, grain at angle_deg."""
    L = src.shape[0]
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    cxd, cyd, cs = (nu - 1) / 2, (nv - 1) / 2, (L - 1) / 2
    M = np.array([[c, s, cs - c * cxd - s * cyd], [-s, c, cs + s * cxd - c * cyd]])
    return cv2.warpAffine(src, M, (nu, nv), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def _grain(ctx: _Ctx, rng: np.random.Generator, spec: dict, angle_deg: float) -> np.ndarray:
    """Unit-variance grain height: streaks along angle_deg plus isotropic octaves."""
    L = int(math.ceil(math.hypot(ctx.nu, ctx.nv))) + 4
    src = np.zeros((L, L), np.float32)
    for across_mm, along_mm, wgt in spec["streaks"]:
        a, b = max(1.0, across_mm / ctx.res), max(1.0, along_mm / ctx.res)
        g = rng.standard_normal((max(2, int(math.ceil(L / a))), max(2, int(math.ceil(L / b))))).astype(np.float32)
        src += wgt * cv2.resize(g, (L, L), interpolation=cv2.INTER_CUBIC)
    if spec["line_mod"]:
        m = _unit(_noise(rng, (L, 1), 2.0))
        src *= np.exp(spec["line_mod"] * m - spec["line_mod"] ** 2 / 2)
    n_lines = int(round(spec["roll_lines"] * L * ctx.res / 100.0))
    if n_lines:
        prof = 2.5 * float(src.std()) * np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)[:, None]
        along = np.clip(0.5 + 0.5 * _unit(_noise(rng, (1, L), 200.0)), 0, None)
        for row in rng.integers(4, L - 5, n_lines):
            src[row - 4: row + 5] -= prof * along
    out = _rotate_to_plate(src, ctx.nv, ctx.nu, angle_deg)
    for cell_mm, wgt in spec["iso"]:
        out += wgt * _unit(_noise(rng, ctx.shape, max(1.0, cell_mm / ctx.res)))
    return _unit(out)


def _base(ctx: _Ctx) -> _Maps:
    """Clean plate: finish grain, waviness, albedo/roughness variation, metallic 1."""
    plate = ctx.plan["plate"]
    mat = plate.get("material") or {}
    finish = plate.get("finish", "brushed")
    if finish not in FINISHES:
        raise ValueError(f"unknown plate finish {finish!r}")
    spec = FINISHES[finish]
    rng = ctx.rng("finish")
    amp = float(mat.get("grain_amp_um", 0.5)) / 1000.0
    grain = _grain(ctx, rng, spec, float(mat.get("grain_dir_deg", 0.0)))
    h = grain * amp
    wave_um = mat.get("waviness_um")
    wave_um = float(rng.uniform(0.5, 2.0) if wave_um is None else wave_um)
    if wave_um > 0:
        h += _unit(_noise(rng, ctx.shape, spec["wave_cell_mm"] / ctx.res)) * (wave_um / 1000.0)
    var = 0.03 * np.clip(_unit(_noise(rng, ctx.shape, 15.0 / ctx.res)) / 2, -1, 1)
    rough = float(mat.get("roughness", 0.25)) + 0.05 * np.clip(_unit(_noise(rng, ctx.shape, 10.0 / ctx.res)) / 2, -1, 1)
    rough += 0.02 * np.clip(grain / 2, -1, 1)          # streaky roughness along the grain
    if spec["mottle"]:
        mot = np.clip(_unit(_noise(rng, ctx.shape, 1.5 / ctx.res)) / 2, -1, 1)
        var += spec["mottle"] * mot
        rough += 1.5 * spec["mottle"] * mot
    base = np.asarray(mat.get("base_color", (0.91, 0.92, 0.92)), np.float32)
    albedo = (1.0 + var)[..., None] * base
    metal = np.full(ctx.shape, float(mat.get("metallic", 1.0)), np.float32)
    maps = _Maps(h.astype(np.float32), albedo.astype(np.float32), rough.astype(np.float32),
                 metal, np.zeros(ctx.shape, np.float32))
    if finish == "hot_rolled":
        _mill_scale(maps, ctx, rng, mat.get("scale") or {})
    elif finish == "spangle":
        _spangle(maps, ctx, rng, mat.get("spangle") or {}, float(mat.get("grain_amp_um", 0.4)) / 1000.0)
    return maps


def _threshold(x: np.ndarray, frac: float) -> float:
    """Value above which about `frac` of x lies (from a strided subsample: fast on full maps)."""
    sub = x[:: max(1, x.shape[0] // 400), :: max(1, x.shape[1] // 400)]
    return float(np.quantile(sub, 1.0 - float(np.clip(frac, 0.0, 1.0))))


def _mill_scale(maps: _Maps, ctx: _Ctx, rng: np.random.Generator, sc: dict) -> None:
    """Hot-rolled mill scale: mottled blue-grey semi-metallic oxide, thicker darker flakes with
    rough edges, and a few spalled spots where bright steel shows."""
    cell = float(sc.get("cell_mm", 4.0)) / ctx.res
    mot = np.clip(_fbm(rng, ctx.shape, cell, 4) / 2, -1, 1)
    k = float(sc.get("mottle", 0.2))
    tint = np.stack([1 - 0.03 * mot, np.ones_like(mot), 1 + 0.05 * mot], -1)     # bluer where lighter
    maps.albedo *= (1 + k * mot)[..., None] * tint
    maps.metal += 0.2 * np.clip(_fbm(rng, ctx.shape, 0.7 * cell, 3) / 2, -1, 1)
    maps.rough += 0.08 * np.clip(_fbm(rng, ctx.shape, 0.5 * cell, 3) / 2, -1, 1)
    fl = _fbm(rng, ctx.shape, max(2.0, 0.6 * cell), 5)
    th = _threshold(fl, float(sc.get("flake_frac", 0.2)))
    flake = _smooth(th - 0.05, th + 0.15, fl)                    # ragged: the fBm's own fine octaves
    blue = np.array([0.97, 1.0, 1.06], np.float32) - 1.0         # flakes only: the rest keeps base_color's hue
    maps.albedo *= (1 - 0.3 * flake)[..., None] * (1.0 + flake[..., None] * blue)
    maps.metal += 0.15 * flake
    maps.rough -= 0.1 * flake
    maps.h += 0.008 * flake
    spall_frac = float(sc.get("spall_frac", 0.0))
    if spall_frac > 0:
        sp = _fbm(rng, ctx.shape, max(2.0, 1.2 / ctx.res), 4)
        th = _threshold(sp, spall_frac)
        m = _smooth(th, th + 0.2, sp)
        steel = np.float32(rng.uniform(0.35, 0.45))
        maps.albedo += m[..., None] * (steel * np.array([1.0, 1.0, 1.02], np.float32) - maps.albedo)
        maps.metal += m * (1.0 - maps.metal)
        maps.rough += m * (0.35 - maps.rough)
        maps.h -= 0.005 * m
    np.clip(maps.metal, 0.0, 1.0, out=maps.metal)


def _voronoi_labels(ctx: _Ctx, rng: np.random.Generator, cell_mm: float) -> tuple[np.ndarray, int]:
    """Cell index per texel of a jittered Voronoi tessellation (mean cell size cell_mm), with the
    cell borders roughened by a domain warp (dendritic spangle boundaries)."""
    n = max(4, int(round(ctx.W * ctx.H / (cell_mm * cell_mm))))
    rows = rng.integers(0, ctx.nv, n)
    cols = rng.integers(0, ctx.nu, n)
    seeds = np.ones(ctx.shape, np.uint8)
    seeds[rows, cols] = 0
    _, lab = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    k = int(lab.max())
    amp = 0.08 * cell_mm / ctx.res
    dx = (amp * np.clip(_fbm(rng, ctx.shape, 0.3 * cell_mm / ctx.res, 3) / 2, -1, 1)).astype(np.float32)
    dy = (amp * np.clip(_fbm(rng, ctx.shape, 0.3 * cell_mm / ctx.res, 3) / 2, -1, 1)).astype(np.float32)
    gx, gy = np.meshgrid(np.arange(ctx.nu, dtype=np.float32), np.arange(ctx.nv, dtype=np.float32))
    lab = cv2.remap(lab.astype(np.float32), gx + dx, gy + dy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return lab.astype(np.int32) - 1, k


def _spangle(maps: _Maps, ctx: _Ctx, rng: np.random.Generator, sp: dict, amp_mm: float) -> None:
    """Galvanised zinc spangle: crystals (Voronoi cells) with slightly different albedo and
    roughness, each with its own micro-grain direction (shading differs per cell, like
    anisotropy), and faint grooves along the crystal boundaries."""
    cell_mm = float(sp.get("cell_mm", 4.0))
    contrast = float(sp.get("contrast", 0.05))
    lab, k = _voronoi_labels(ctx, rng, cell_mm)
    alb = (1 + contrast * np.clip(rng.standard_normal(k), -2.5, 2.5)).astype(np.float32)
    rgh = (0.05 * np.clip(rng.standard_normal(k), -2, 2)).astype(np.float32)
    bins = rng.integers(0, SPANGLE_BINS, k)
    maps.albedo *= alb[lab][..., None]
    maps.rough += rgh[lab]
    # No per-crystal tilt: piecewise-planar heights put steps on the boundaries, which shade
    # like cracks. Cells differ by albedo, roughness and the direction/strength of their grain.
    spec = {"streaks": ((0.04, 0.8, 1.0), (0.12, 2.5, 0.4)), "iso": (), "line_mod": 0.0, "roll_lines": 0}
    cell_bin = bins[lab]
    off = rng.uniform(0.0, 180.0)
    gain = rng.uniform(0.5, 1.5, k).astype(np.float32)[lab]
    for b in range(SPANGLE_BINS):
        m = cell_bin == b
        if m.any():
            g = _grain(ctx, rng, spec, off + b * 180.0 / SPANGLE_BINS)
            maps.h[m] += amp_mm * gain[m] * g[m]
    edge = np.zeros(ctx.shape, bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    e = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 0.8)
    maps.h -= 0.0005 * e                       # faint grain-boundary line, far below a crack's contrast
    maps.albedo *= (1 - 0.03 * e)[..., None]


# ----------------------------------------------------------------------------- dents

def _punch_profile(r, a: float, depth: float, tip_frac: float = 0.25):
    """Cone of surface radius a with a paraboloid tip; minimum exactly -depth, 0 for r >= a."""
    rt = tip_frac * a
    D = depth / (1 - tip_frac / 2)
    s = D / a
    h = np.where(r < rt, -D + s * rt / 2 + s * r * r / (2 * rt), -D + s * r)
    return np.minimum(h, 0.0)


def _rim(r, a: float, rw: float, height: float):
    t = (r - a) / rw
    return np.where((t > 0) & (t < 1), np.sin(np.pi * t) ** 2, 0.0) * height


def _dent_punch(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    u, v, d, depth = float(p["u"]), float(p["v"]), float(p["d_mm"]), float(p["depth_mm"])
    rim_h = float(p.get("rim_h_mm", 0.03))
    a, rw = d / 2, rim_width_mm(d)
    win = ctx.disc(u, v, a + rw)
    if win is None:
        return []
    U, V = ctx.coords(win)
    r = np.hypot(U - u, V - v)
    rim = _rim(r, a, rw, rim_h)
    inside = _smooth(a, a - ctx.res, r)
    rim_w = rim / rim_h if rim_h > 0 else np.zeros_like(r)
    ops = [("h", "add", _punch_profile(r, a, depth) + rim, 1.0),
           ("rough", "add", -0.05, inside),            # burnished by the hardened tip
           ("rough", "add", 0.06, rim_w)]
    return [_Stamp(win, ops, np.maximum(inside, rim_w))]


def _dent_ball(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    u, v, d, D = float(p["u"]), float(p["v"]), float(p["d_mm"]), float(p["depth_mm"])
    a = d / 2
    win = ctx.disc(u, v, a)
    if win is None or D <= 0:
        return []
    U, V = ctx.coords(win)
    r = np.hypot(U - u, V - v)
    R = (a * a + D * D) / (2 * D)                         # sphere through the rim and the bottom

    def cap(x):
        return -D + R - np.sqrt(np.maximum(R * R - x * x, 0.0))

    r1 = 0.85 * a                                         # rounded shoulder: Hermite to 0 at the rim
    z1, s1 = cap(r1), r1 / math.sqrt(R * R - r1 * r1)
    t = np.clip((r - r1) / (a - r1), 0.0, 1.0)
    herm = (2 * t ** 3 - 3 * t ** 2 + 1) * z1 + (t ** 3 - 2 * t ** 2 + t) * (a - r1) * s1
    h = np.where(r <= r1, cap(r), np.where(r < a, herm, 0.0))
    cover = (h < -1e-4 * D).astype(np.float32)
    return [_Stamp(win, [("h", "add", h, 1.0), ("rough", "add", -0.03, cover)], cover)]


def _dent_smooth(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    u, v, d, depth = float(p["u"]), float(p["v"]), float(p["d_mm"]), float(p["depth_mm"])
    A = d / 2
    B = A / max(1.0, float(p.get("aspect", 1.0)))
    th = math.radians(float(p.get("angle_deg", 0.0)))
    win = ctx.disc(u, v, A)
    if win is None:
        return []
    U, V = ctx.coords(win)
    x, y = U - u, V - v
    xs, ys = x * math.cos(th) + y * math.sin(th), -x * math.sin(th) + y * math.cos(th)
    rho2 = (xs / A) ** 2 + (ys / B) ** 2
    g = np.where(rho2 < 1, np.exp(-1.5 * rho2) * (1 - rho2) ** 2, 0.0)   # windowed Gaussian, C1 at the rim
    return [_Stamp(win, [("h", "add", -depth * g, 1.0)], (g > 1e-3).astype(np.float32))]


def _dent_nick(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Half-dent centred on a plate edge (the off-plate half has no texels) with a raised bright lip.

    The centre is snapped onto the named edge, as the footprint (geometry._nick) does.
    """
    u, v, d, depth = float(p["u"]), float(p["v"]), float(p["d_mm"]), float(p["depth_mm"])
    u = {"left": 0.0, "right": ctx.W}.get(p.get("edge"), u)
    v = {"top": 0.0, "bottom": ctx.H}.get(p.get("edge"), v)
    a, rw = d / 2, rim_width_mm(d)
    win = ctx.disc(u, v, a + rw)
    if win is None:
        return []
    U, V = ctx.coords(win)
    r = np.hypot(U - u, V - v)
    lip_h = min(0.04, 0.15 * depth)
    lip = _rim(r, a, rw, 1.0)
    inside = _smooth(a, a - ctx.res, r)
    ops = [("h", "add", _punch_profile(r, a, depth, 0.15) + lip_h * lip, 1.0),
           ("rough", "add", 0.08, inside), ("rough", "add", 0.15, lip), ("albedo", "mul", 1.08, lip)]
    return [_Stamp(win, ops, np.maximum(inside, lip))]


# ----------------------------------------------------------------------------- scratches

def _scratch(p: dict, ctx: _Ctx, rng: np.random.Generator, gouge: bool) -> list[_Stamp]:
    pts = _bezier(p["ctrl_mm"])
    w, depth, burr = float(p["width_mm"]), float(p["depth_mm"]), float(p.get("burr", 0.0))
    he, bw = _scratch_widths(w, burr, ctx.res)
    D = depth * (w / 2) / he                       # same groove cross-section area when widened
    reach = he + bw
    win = _bbox_window(ctx, pts, reach)
    if win is None:
        return []
    dist, s, length = _polyline_field(ctx, win, pts, reach)
    taper = min(2.0, 0.1 * length)
    along = _smooth(0, taper, s) * _smooth(length, length - taper, s) if taper > 0 else np.ones_like(s)
    var = 1 + 0.15 * np.clip(_noise1d(rng, s, 2.0, 0.0, max(length, 1e-6)), -2, 2)
    q = np.minimum(dist / he, 1.0)                  # dist is inf away from the line
    groove = np.where(q < 1, np.cos(np.pi / 2 * q) ** 2, 0.0) * along
    dh = -D * groove * var
    if gouge:                                       # torn, uneven bottom
        dh *= 1 + 0.1 * _noise(rng, win.shape, max(1.5, he / ctx.res))
    ridge = np.zeros_like(dist)
    if bw > 0:
        tb = np.clip((dist - he) / bw, 0.0, 1.0)
        ridge = np.where((tb > 0) & (tb < 1), np.sin(np.pi * tb) ** 2, 0.0) * along
        dh += burr * 0.3 * D * ridge * var
    ops = [("h", "add", dh, 1.0),
           ("rough", "add", 0.15 if gouge else 0.1, groove),
           ("albedo", "mul", 0.85 if gouge else 1.06, groove)]   # gouge: darker; scriber: fresh metal
    ops += _through_scale(ctx, rng, groove)
    if bw > 0:
        ops.append(("rough", "add", 0.05, ridge))
    return [_Stamp(win, ops, np.maximum(groove, ridge))]


def _through_scale(ctx: _Ctx, rng: np.random.Generator, m: np.ndarray) -> list[Op]:
    """A scratch through mill scale shows bright bare steel (hot-rolled plates only)."""
    if ctx.finish != "hot_rolled":
        return []
    steel = np.float32(rng.uniform(0.4, 0.55)) * np.array([1.0, 1.0, 1.02], np.float32)
    return [("albedo", "mix", steel, 0.85 * m), ("metal", "mix", 1.0, 0.85 * m), ("rough", "mix", 0.3, 0.6 * m)]


def _scratch_abrasion(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """n_lines short fine lines along the w axis (angle_deg), inside the w x h rectangle."""
    u, v, w, hh = float(p["u"]), float(p["v"]), float(p["w_mm"]), float(p["h_mm"])
    depth, n = float(p["depth_mm"]), int(p.get("n_lines", 60))
    th = math.radians(float(p.get("angle_deg", 0.0)))
    c, s = math.cos(th), math.sin(th)
    eu, ev = abs(c) * w / 2 + abs(s) * hh / 2, abs(s) * w / 2 + abs(c) * hh / 2
    win = ctx.window(u - eu, v - ev, u + eu, v + ev)
    if win is None:
        return []
    U, V = ctx.coords(win)
    x, y = U - u, V - v
    S, T = x * c + y * s, -x * s + y * c
    dh = np.zeros(win.shape)
    acc = np.zeros(win.shape)
    for _ in range(n):
        wl = rng.uniform(0.02, 0.08)
        hel = max(wl / 2, 0.6 * ctx.res)
        dl = depth * rng.uniform(0.3, 1.0) * (wl / 2) / hel
        room = w - 2 * hel
        ln = rng.uniform(0.15, 0.7) * room
        s0 = -w / 2 + hel + rng.uniform(0, room - ln)
        t0 = rng.uniform(-hh / 2 + hel, hh / 2 - hel)
        a = np.array([u + s0 * c - t0 * s, v + s0 * s + t0 * c])
        b = a + ln * np.array([c, s])
        got = _seg_local(ctx, win, a, b, hel)
        if got is None:
            continue
        sl, d, t = got
        q = d / hel
        f = np.where(q < 1, np.cos(np.pi / 2 * q) ** 2, 0.0) * _smooth(0, 0.2, t) * _smooth(1, 0.8, t)
        dh[sl] = np.minimum(dh[sl], -dl * f)
        acc[sl] = np.maximum(acc[sl], f)
    patch = _smooth(0.0, 0.5, np.minimum(w / 2 - np.abs(S), hh / 2 - np.abs(T)))
    ops = [("h", "add", dh, 1.0), ("rough", "add", 0.08, acc), ("rough", "add", 0.04, patch),
           ("albedo", "mul", 0.98, patch)] + _through_scale(ctx, rng, acc)
    return [_Stamp(win, ops, np.maximum(acc, patch))]


# ----------------------------------------------------------------------------- cracks

def _crack_slit(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Saw slit from (u0, v0) along angle_deg: a deep dark kerf (a normal map cannot occlude)."""
    u0, v0, L, w = float(p["u0"]), float(p["v0"]), float(p["length_mm"]), float(p["width_mm"])
    th = math.radians(float(p["angle_deg"]))
    c, s = math.cos(th), math.sin(th)
    pts = np.array([[u0, v0], [u0 + L * c, v0 + L * s]])
    win = _bbox_window(ctx, pts, w / 2 + ctx.res)
    if win is None:
        return []
    U, V = ctx.coords(win)
    x, y = U - u0, V - v0
    S, T = x * c + y * s, -x * s + y * c
    r = ctx.res
    cov = (np.clip((w / 2 - np.abs(T)) / r + 0.5, 0, 1) * np.clip(S / r + 0.5, 0, 1)
           * np.clip((L - S) / r + 0.5, 0, 1))
    ops = [("h", "add", -min(1.0, 3 * w) * cov, 1.0), ("albedo", "mix", 0.03, cov),
           ("metal", "mix", 0.0, cov), ("rough", "mix", 0.5, cov)]
    return [_Stamp(win, ops, cov)]


def _crack_fatigue(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Tight crack: a thin jagged dark line, contrast scaled by width / texel."""
    pts = np.asarray(p["pts_mm"], float).reshape(-1, 2)
    w = float(p["width_mm"])
    hw = max(w / 2, ctx.res / 2)
    win = _bbox_window(ctx, pts, hw + ctx.res)
    if win is None:
        return []
    dist, _, _ = _polyline_field(ctx, win, pts, hw + ctx.res)
    cov = np.clip((hw - dist) / ctx.res + 0.5, 0, 1)
    k = 0.6 * min(1.0, w / ctx.res)
    ops = [("albedo", "mul", 1.0 - k, cov), ("h", "add", -min(0.02, 2 * w) * cov, 1.0),
           ("rough", "add", 0.05, cov)]
    return [_Stamp(win, ops, cov)]


# ----------------------------------------------------------------------------- corrosion

def _powder_ops(rng: np.random.Generator, m: np.ndarray, shape: tuple[int, int]) -> list[Op]:
    """White-grey corrosion powder of coverage m: albedo 0.7-0.85, roughness 0.7-0.95, metallic 0 where m > 0.5."""
    speck = _noise(rng, shape, 1.5)
    col = rng.uniform(0.70, 0.85) * (1 + 0.05 * np.clip(speck, -2, 2))
    tint = np.array([1.0, 1.0, rng.uniform(0.95, 1.0)], np.float32)
    return [("albedo", "mix", col[..., None] * tint, m), ("rough", "mix", rng.uniform(0.7, 0.95), m),
            ("metal", "mix", 0.0, _smooth(0.4, 0.6, m)),
            ("h", "add", 0.008 * m * (0.6 + 0.4 * np.clip(speck, -1, 1)), 1.0)]


def _corrosion_pitting(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    u, v, r = float(p["u"]), float(p["v"]), float(p["r_mm"])
    powder = float(p.get("powder", 0.5))
    pd = p.get("pit_d_mm", [0.1, 0.5])
    win = ctx.disc(u, v, r)
    if win is None:
        return []
    U, V = ctx.coords(win)
    rho = np.hypot(U - u, V - v) / r
    edge = _smooth(1.0, 0.85, rho)
    raw = 0.6 * _fbm(rng, win.shape, 0.4 * r / ctx.res) + 1.6 * (powder - 0.5) + 0.8 * (1 - rho ** 2) - 0.4
    m = _smooth(-0.1, 0.4, raw) * edge
    halo = _smooth(-0.5, 0.0, raw) * (1 - m) * edge          # dulled, etched metal around the powder
    ops = [("albedo", "mul", 0.9, halo), ("rough", "add", 0.15, halo)] + _powder_ops(rng, m, win.shape)
    dh = np.zeros(win.shape)
    pit = np.zeros(win.shape)
    # The sampler's pit list ([u, v, d]) wins: penetrant rounded indications sit on those pits.
    given = p.get("pits")
    for k in range(len(given) if given else int(p.get("n_pits", 10))):
        if given:
            pu, pv, rp = float(given[k][0]), float(given[k][1]), float(given[k][2]) / 2
            dp = min(rng.uniform(0.05, 0.3), 1.2 * rp)
        else:
            rp = rng.uniform(float(pd[0]), float(pd[1])) / 2
            dp = min(rng.uniform(0.05, 0.3), 1.2 * rp)
            rc = (r - rp - ctx.res) * math.sqrt(rng.random())
            if rc < 0:
                continue
            ang = rng.uniform(0, 2 * np.pi)
            pu, pv = u + rc * math.cos(ang), v + rc * math.sin(ang)
        loc = _local(ctx, win, ctx.disc(pu, pv, rp))
        if loc is None:
            continue
        sl, sub = loc
        Us, Vs = ctx.coords(sub)
        q = np.hypot(Us - pu, Vs - pv) / rp
        dh[sl] = np.minimum(dh[sl], np.where(q < 1, -dp * (1 - q * q), 0.0))
        pit[sl] = np.maximum(pit[sl], _smooth(1.0, 0.6, q))
    ops += [("h", "add", dh, 1.0), ("albedo", "mix", rng.uniform(0.1, 0.25), pit),
            ("metal", "mix", 0.2, pit), ("rough", "mix", 0.85, pit)]
    return [_Stamp(win, ops, np.maximum(np.maximum(m, halo), pit))]


def _corrosion_galvanic(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Ring of powder round a fastener's visible outline, darkest at the inner edge."""
    fu, fv, r0 = ctx.fastener(p["hole"])
    rw = float(p["ring_w_mm"])
    powder = float(p.get("powder", 0.6))
    win = ctx.disc(fu, fv, r0 + rw)
    if win is None:
        return []
    U, V = ctx.coords(win)
    x, y = U - fu, V - fv
    r = np.hypot(x, y)
    t = (r - r0) / rw
    ring = (t > 0) * _smooth(1.0, 0.85, t)
    raw = (0.9 - 1.2 * t + 0.35 * _fbm(rng, win.shape, max(0.3, rw / 3) / ctx.res)
           + 0.25 * _angular_noise(rng, np.arctan2(y, x)) + 0.8 * (powder - 0.5))
    m = _smooth(0.0, 0.35, raw) * ring
    dark = np.exp(-(((r - r0) / (0.12 * rw + 0.08)) ** 2)) * ring
    ops = _powder_ops(rng, m, win.shape) + [("albedo", "mix", 0.12, 0.75 * dark), ("metal", "mix", 0.3, 0.7 * dark)]
    return [_Stamp(win, ops, np.maximum(m, dark))]


def _corrosion_fretting(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """'Smoking rivet': dark streak from the head edge along angle_deg, tapering."""
    riv = ctx.rivets[p["rivet"]]
    hole = ctx.holes[riv["hole"]]
    cu, cv_, rh = float(hole["u"]), float(hole["v"]), float(riv["head_d_mm"]) / 2
    L, W, dark = float(p["length_mm"]), float(p["width_mm"]), float(p.get("darkness", 0.6))
    th = math.radians(float(p["angle_deg"]))
    c, s = math.cos(th), math.sin(th)
    e1, e2 = np.array([c, s]), np.array([-s, c])
    ctr = np.array([cu, cv_])
    corners = np.array([ctr + (rh + a) * e1 + b * e2 for a in (0, L) for b in (-W / 2, W / 2)])
    win = _bbox_window(ctx, corners, 0.0)
    if win is None:
        return []
    U, V = ctx.coords(win)
    x, y = U - cu, V - cv_
    S, T = x * c + y * s - rh, -x * s + y * c
    q = np.clip(S / L, 0, 1)
    half = W / 2 * (1 - 0.7 * q)
    lat = _smooth(1.0, 0.4, np.abs(T) / half)
    along = _smooth(0, 0.05 * L, S) * _smooth(L, 0.85 * L, S) * (1 - 0.6 * q)
    streaks = 0.35 * np.clip(_noise1d(rng, T, 0.15, -W / 2, W / 2), -2, 2)
    m = np.clip(lat * along * (0.75 + streaks + 0.2 * _fbm(rng, win.shape, 1.0 / ctx.res, 3)), 0, 1)
    alb = 0.3 - 0.22 * float(np.clip((dark - 0.3) / 0.6, 0, 1))
    ops = [("albedo", "mix", alb, m * min(1.0, 0.4 + dark)), ("metal", "mix", 0.3, m), ("rough", "mix", 0.8, m)]
    return [_Stamp(win, ops, m)]


# ----------------------------------------------------------------------------- steel defects (DESIGN_STEEL A)

def _rust_colors(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(orange-brown, dark red-brown) iron oxide, linear, both inside [0.25-0.45, 0.08-0.18, 0.03-0.08]."""
    fresh = np.array([rng.uniform(0.33, 0.45), rng.uniform(0.12, 0.18), rng.uniform(0.04, 0.08)], np.float32)
    old = np.array([rng.uniform(0.25, 0.33), rng.uniform(0.08, 0.11), rng.uniform(0.03, 0.05)], np.float32)
    return fresh, old


def _rust_ops(rng: np.random.Generator, m: np.ndarray, shape: tuple[int, int], cell_px: float,
              thick_mm: float, metal_k: float = 1.0) -> list[Op]:
    """Iron-oxide layer of coverage m: orange-brown to dark red, roughness 0.7-0.95, metallic 0
    where dense (metal_k < 1 for a thin bleed film), a raised rough crust up to thick_mm."""
    fresh, old = _rust_colors(rng)
    age = _smooth(-0.8, 0.8, _fbm(rng, shape, max(2.0, cell_px), 3))[..., None]
    speck = 1 + 0.06 * np.clip(_noise(rng, shape, 1.5), -2, 2)[..., None]
    col = (fresh + (old - fresh) * age) * speck
    crust = 0.6 + 0.4 * np.clip(_fbm(rng, shape, max(2.0, 0.3 * cell_px), 3) / 2, -1, 1)
    return [("albedo", "mix", col, m), ("rough", "mix", rng.uniform(0.7, 0.95), m),
            ("metal", "mix", 0.0, metal_k * _smooth(0.25, 0.6, m)), ("h", "add", thick_mm * m * crust, 1.0)]


def _pit_stamps(ctx: _Ctx, rng: np.random.Generator, pits, halo_mm: float, halo_k: float,
                win: _Win) -> list[Op]:
    """Ops (on win) of conical dark pits [(u, v, d, depth)], each with an optional ragged rust halo
    reaching halo_mm beyond its rim (mild steel)."""
    dh = np.zeros(win.shape)
    dark = np.zeros(win.shape)
    halo = np.zeros(win.shape)
    for pu, pv, d, depth in pits:
        a = d / 2
        loc = _local(ctx, win, ctx.disc(pu, pv, a + halo_mm))
        if loc is None:
            continue
        sl, sub = loc
        U, V = ctx.coords(sub)
        r = np.hypot(U - pu, V - pv)
        dh[sl] = np.minimum(dh[sl], _punch_profile(r, a, depth, 0.3))
        k = float(np.clip(2.0 * depth / max(d, 1e-6), 0.7, 1.0))       # a cavity traps the light
        dark[sl] = np.maximum(dark[sl], k * _smooth(a, 0.4 * a, r))
        if halo_mm > 0 and halo_k > 0:
            t = np.clip((r - a) / halo_mm, 0.0, 1.0)
            th = np.arctan2(V - pv, U - pu)
            rag = np.clip(0.65 + 0.35 * _angular_noise(rng, th, 6), 0, 1)
            ring = (1 - t) ** 1.5 * _smooth(0.5 * a, a, r) * (r < a + halo_mm)      # from the rim outward
            halo[sl] = np.maximum(halo[sl], halo_k * ring * rag)
    ops: list[Op] = []
    if halo.any():
        ops += _rust_ops(rng, halo, win.shape, 0.5 / ctx.res, 0.004)
    pit_col = (np.array([0.035, 0.02, 0.012], np.float32) if ctx.kind in STEEL_KINDS
               else np.array([0.04, 0.04, 0.045], np.float32))
    ops += [("h", "add", dh, 1.0), ("albedo", "mix", pit_col, dark), ("metal", "mix", 0.3, dark),
            ("rough", "mix", 0.8, dark)]
    return ops


def _pit_single(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Small dark conical cavity of diameter d_mm and depth depth_mm, optional rust halo."""
    u, v, d = float(p["u"]), float(p["v"]), float(p["d_mm"])
    halo = float(p.get("halo_mm") or 0.0)
    win = ctx.disc(u, v, d / 2 + halo)
    if win is None:
        return []
    ops = _pit_stamps(ctx, rng, [(u, v, d, float(p["depth_mm"]))], halo, float(p.get("halo", 0.7)), win)
    U, V = ctx.coords(win)
    cover = (np.hypot(U - u, V - v) < d / 2 + halo).astype(np.float32)
    return [_Stamp(win, ops, cover)]


def _pit_cluster(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Cluster of pits at the sampler's pit list ([u, v, d]), depth scaled with pit size."""
    pits = p.get("pits") or []
    if not pits:
        return []
    halo = float(p.get("halo_mm") or 0.0)
    dmax = max(float(q[2]) for q in pits)
    depth = float(p.get("depth_mm", 0.2))
    arr = np.asarray(pits, float)[:, :3]
    reach = arr[:, 2].max() / 2 + halo
    win = _bbox_window(ctx, arr[:, :2], reach)
    if win is None:
        return []
    spec = [(float(q[0]), float(q[1]), float(q[2]), depth * math.sqrt(float(q[2]) / dmax) * rng.uniform(0.6, 1.0))
            for q in pits]
    ops = _pit_stamps(ctx, rng, spec, halo, float(p.get("halo", 0.7)), win)
    cover = np.zeros(win.shape, np.float32)
    U, V = ctx.coords(win)
    for pu, pv, d, _ in spec:
        cover = np.maximum(cover, (np.hypot(U - pu, V - pv) < d / 2 + halo).astype(np.float32))
    return [_Stamp(win, ops, cover)]


def _corrosion_rust(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Red rust patch inside the ellipse r_mm x r_mm/aspect: fBm-ragged edge, `coverage` of the
    outline area rusted (denser in the middle), optional pitting under the dense rust."""
    a, aspect = float(p["r_mm"]), float(p.get("aspect", 1.0))
    got = _ellipse_frame(ctx, p, a, aspect)
    if got is None:
        return []
    win, rho = got
    U, V = ctx.coords(win)
    th = math.radians(float(p.get("angle_deg", 0.0)))
    x, y = U - float(p["u"]), V - float(p["v"])
    phi = np.arctan2(-x * math.sin(th) + y * math.cos(th), x * math.cos(th) + y * math.sin(th))
    cell = max(0.5, 0.25 * a) / ctx.res
    edge_r = 1 - 0.22 * (0.5 + 0.5 * np.clip(_angular_noise(rng, phi, 10) / 1.5, -1, 1))
    rho_n = rho + 0.05 * _fbm(rng, win.shape, max(2.0, 0.3 / ctx.res), 3)
    env = _smooth(edge_r, edge_r - 0.08, rho_n) * _smooth(1.0, 0.97, rho)    # never past the outline
    raw = _fbm(rng, win.shape, cell, 5) + 1.2 * (1 - np.minimum(rho, 1.0) ** 2)
    inside = env > 0.5
    n_out = float((rho < 1).sum())
    c_eff = float(np.clip(float(p.get("coverage", 0.7)) * n_out / max(1.0, float(inside.sum())), 0.0, 1.0))
    t = float(np.quantile(raw[inside], 1.0 - c_eff)) if inside.any() else float(raw.max())
    m = _smooth(t - 0.1, t + 0.1, raw) * env if c_eff < 1.0 else env
    ops = _rust_ops(rng, m, win.shape, cell, rng.uniform(0.01, 0.04))
    if p.get("pitted"):
        dense = np.argwhere(m > 0.7)
        n = min(40, int(len(dense) * ctx.ru * ctx.rv * 0.3))
        if n:
            pick = dense[rng.choice(len(dense), n, replace=False)]
            pits = [(float((win.c0 + c + 0.5) * ctx.ru), float((win.r0 + r + 0.5) * ctx.rv),
                     rng.uniform(0.15, 0.6), rng.uniform(0.03, 0.15)) for r, c in pick]
            pits = [q for q in pits if _ellipse_rho(p, a, aspect, q[0], q[1]) + q[2] / (2 * a / aspect) < 0.98]
            ops += _pit_stamps(ctx, rng, pits, 0.0, 0.0, win)
    return [_Stamp(win, ops, m)]


def _ellipse_rho(p: dict, a: float, aspect: float, u: float, v: float) -> float:
    th = math.radians(float(p.get("angle_deg", 0.0)))
    x, y = u - float(p["u"]), v - float(p["v"])
    xs, ys = x * math.cos(th) + y * math.sin(th), -x * math.sin(th) + y * math.cos(th)
    return math.hypot(xs / a, ys * aspect / a)


def _corrosion_rust_streak(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Rust bleed streak from (u0, v0) along angle_deg, tapering and fading toward the tip, with
    fine runs along it; a dense rust spot at the start when head_r_mm is given."""
    hw0, hw1, D = G.streak_widths(p)
    u0, v0 = float(p["u0"]), float(p["v0"])
    th = math.radians(float(p["angle_deg"]))
    c, s = math.cos(th), math.sin(th)
    L = float(p["length_mm"])
    head = float(p.get("head_r_mm") or 0.0)
    reach = max(hw0, head)
    win = _bbox_window(ctx, np.array([[u0, v0], [u0 + D * c, v0 + D * s]]), reach)
    if win is None:
        return []
    U, V = ctx.coords(win)
    x, y = U - u0, V - v0
    S, T = x * c + y * s, -x * s + y * c
    r0 = np.hypot(x, y)
    hw = hw0 + (hw1 - hw0) * np.clip(S / D, 0.0, 1.0)
    q = np.where(S < 0, r0 / hw0, np.where(S > D, np.hypot(x - D * c, y - D * s) / hw1, np.abs(T) / hw))
    shape = np.exp(-1.2 * q * q) * _smooth(1.0, 0.85, q)            # soft across, zero at the outline
    fade = 1 - float(p.get("fade", 0.6)) * np.clip(S / L, 0.0, 1.0)
    runs = 1 + 0.3 * np.clip(_noise1d(rng, T, 0.12, -reach, reach), -2, 2)
    blotch = np.clip(0.55 + 0.45 * _fbm(rng, win.shape, max(2.0, 1.5 / ctx.res), 3), 0, 1)
    m = np.clip(shape * fade * runs * blotch * rng.uniform(0.7, 1.0), 0, 1)
    if head > 0:
        rag = 1 + 0.15 * np.clip(_fbm(rng, win.shape, max(2.0, 0.3 * head / ctx.res), 3), -2, 2)
        hm = _smooth(head, 0.6 * head, r0 * rag) * _smooth(head, 0.95 * head, r0)
        m = np.maximum(m, hm)
    ops = _rust_ops(rng, m, win.shape, max(2.0, hw0 / ctx.res), 0.005, metal_k=0.6)
    return [_Stamp(win, ops, m)]


def _crack_open(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Open crack: a jagged dark line of width_mm (darkness scaled by width / texel below one texel),
    a little groove, slightly raised lips (geometry.crack_lip_mm) and tips that narrow."""
    pts = np.asarray(p["pts_mm"], float).reshape(-1, 2)
    w = float(p["width_mm"])
    lip = G.crack_lip_mm(w)
    hc = max(w / 2, ctx.res / 2)
    reach = hc + lip
    win = _bbox_window(ctx, pts, reach)
    if win is None:
        return []
    dist, s, length = _polyline_field(ctx, win, pts, reach)
    tl = min(1.5, 0.2 * length)
    taper = _smooth(length, length - tl, s)
    if p.get("start") not in ("edge", "hole"):                    # a free crack narrows at both ends
        taper = taper * _smooth(0.0, tl, s)
    width_k = 0.35 + 0.65 * taper
    hcs = hc * width_k
    cov = np.clip((hcs - dist) / ctx.res + 0.5, 0, 1)
    var = 1 + 0.15 * np.clip(_noise1d(rng, s, 1.0, 0.0, max(length, 1e-6)), -2, 2)
    k = float(rng.uniform(0.7, 0.9)) * min(1.0, w / ctx.res)          # an open void returns little light
    tl_ = np.clip((dist - hcs) / lip, 0.0, 1.0)
    lips = np.where((tl_ > 0) & (tl_ < 1), np.sin(np.pi * tl_) ** 2, 0.0) * taper
    ops = [("albedo", "mul", 1.0 - k, np.clip(cov * var, 0, 1)), ("metal", "mix", 0.4, 0.5 * cov),
           ("rough", "add", 0.1, cov), ("h", "add", -min(0.25, 2 * w) * cov + min(0.015, 0.25 * w + 0.003) * lips, 1.0),
           ("rough", "add", 0.06, lips), ("albedo", "mul", 1.04, lips)]
    return [_Stamp(win, ops, np.maximum(cov, lips))]


def _stain_oil(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Oil film: slightly darker, LOWER roughness (glossy), a faint thin-film tint at the rim."""
    r = float(p.get("r_mm", 4.0))
    got = _ellipse_frame(ctx, p, r, float(p.get("aspect", 1.0)))
    if got is None:
        return []
    win, rho = got
    k, gloss = float(p.get("strength", 0.8)), float(p.get("gloss", 0.7))
    rho_n = rho / 0.92 + 0.05 * _fbm(rng, win.shape, max(2.0, 0.3 * r / ctx.res), 3)
    m = _smooth(1.0, 0.85, rho_n) * _smooth(1.0, 0.96, rho)
    film = 0.7 + 0.3 * np.clip(_fbm(rng, win.shape, max(2.0, 0.5 * r / ctx.res), 3) / 2, -1, 1)
    ring = np.exp(-(((rho_n - 0.9) / 0.06) ** 2)) * m
    ph = 2 * np.pi * (3 * rho_n + 0.3 * _fbm(rng, win.shape, max(2.0, 0.2 * r / ctx.res), 2))
    tint = 1 + 0.06 * k * np.stack([np.cos(ph), np.cos(ph + 2.1), np.cos(ph + 4.2)], -1)
    ops = [("rough", "mul", 1.0 - 0.65 * gloss, m * k), ("albedo", "mul", 1.0 - 0.08 * k, m * film),
           ("albedo", "mul", tint, ring)]
    return [_Stamp(win, ops, m * k)]


def _stain_water(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Dried water mark: whitish mineral ring just inside r_mm (ring_w_mm FWHM), a faint haze and
    scattered mineral specks inside."""
    u, v, r = float(p["u"]), float(p["v"]), float(p.get("r_mm", 4.0))
    sig = float(p.get("ring_w_mm", 0.25)) / 2.355
    k, minerals = float(p.get("strength", 0.7)), float(p.get("minerals", 0.5))
    win = ctx.disc(u, v, r)
    if win is None:
        return []
    U, V = ctx.coords(win)
    d = np.hypot(U - u, V - v)
    th = np.arctan2(V - v, U - u)
    re = (r - 2.5 * sig) * (1 - 0.05 * (0.5 + 0.5 * np.clip(_angular_noise(rng, th, 10) / 2, -1, 1)))
    lim = _smooth(r, 0.97 * r, d)
    ring = k * np.exp(-0.5 * ((d - re) / sig) ** 2) * lim
    haze = k * 0.2 * _smooth(re, 0.6 * re, d)
    sp = _noise(rng, win.shape, 2.0)
    speck = minerals * k * _smooth(1.8, 2.4, sp) * (d < re)
    dep = np.array([0.78, 0.78, 0.75], np.float32) * np.float32(rng.uniform(0.9, 1.05))
    ops = [("albedo", "mix", dep, np.clip(0.45 * ring + 0.06 * haze + 0.5 * speck, 0, 1)),
           ("rough", "add", 0.3 * ring + 0.06 * haze + 0.2 * speck, 1.0),
           ("metal", "mix", 0.0, np.clip(0.5 * ring + 0.3 * speck, 0, 1)), ("h", "add", 0.002 * ring, 1.0)]
    return [_Stamp(win, ops, np.clip(np.maximum(np.maximum(ring, haze), speck), 0, 1))]


DEFECTS: dict[tuple[str, str], Callable[[dict, _Ctx, np.random.Generator], list[_Stamp]]] = {
    ("dent", "punch"): _dent_punch,
    ("dent", "ball"): _dent_ball,
    ("dent", "smooth"): _dent_smooth,
    ("dent", "nick"): _dent_nick,
    ("scratch", "scriber"): lambda p, c, r: _scratch(p, c, r, gouge=False),
    ("scratch", "gouge"): lambda p, c, r: _scratch(p, c, r, gouge=True),
    ("scratch", "abrasion"): _scratch_abrasion,
    ("crack", "slit"): _crack_slit,
    ("crack", "fatigue"): _crack_fatigue,
    ("corrosion", "pitting"): _corrosion_pitting,
    ("corrosion", "galvanic"): _corrosion_galvanic,
    ("corrosion", "fretting"): _corrosion_fretting,
    ("pit", "single"): _pit_single,
    ("pit", "cluster"): _pit_cluster,
    ("corrosion", "rust"): _corrosion_rust,
    ("corrosion", "rust_streak"): _corrosion_rust_streak,
    ("corrosion", "white"): _corrosion_pitting,          # white powder + pits (the aluminium look)
    ("crack", "open"): _crack_open,
    ("stain", "oil"): _stain_oil,
    ("stain", "water"): _stain_water,
    ("stain", "fingerprint"): lambda p, c, r: _fingerprint(p, c, r),
}


# ----------------------------------------------------------------------------- distractors

def _ellipse_frame(ctx: _Ctx, p: dict, r: float, aspect: float = 1.0):
    u, v = float(p["u"]), float(p["v"])
    win = ctx.disc(u, v, r)
    if win is None:
        return None
    U, V = ctx.coords(win)
    th = math.radians(float(p.get("angle_deg", 0.0)))
    x, y = U - u, V - v
    xs, ys = x * math.cos(th) + y * math.sin(th), -x * math.sin(th) + y * math.cos(th)
    return win, np.hypot(xs / r, ys * aspect / r)


def _fingerprint(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Ridge pattern of concentric distorted ellipses: roughness +-0.1, faint albedo."""
    r = float(p.get("r_mm", 7.0))
    aspect = rng.uniform(1.25, 1.5)                    # always drawn: the sampler's aspect wins
    got = _ellipse_frame(ctx, p, r, float(p.get("aspect", aspect)))
    if got is None:
        return []
    win, rho = got
    period = rng.uniform(0.42, 0.55)
    phase = 2 * np.pi * rho * r / period + 2.0 * _fbm(rng, win.shape, 0.25 * r / ctx.res, 3)
    ridges = (0.5 + 0.5 * np.cos(phase)) ** 1.5
    cover = _smooth(1.0, 0.75, rho) * np.clip(0.7 + 0.6 * _fbm(rng, win.shape, 0.3 * r / ctx.res, 3), 0, 1)
    wgt = cover * ridges * float(p.get("strength", rng.uniform(0.6, 1.0)))
    sign = 1.0 if rng.random() < 0.75 else -1.0
    return [_Stamp(win, [("rough", "add", 0.1 * sign, wgt), ("albedo", "mul", 0.97, wgt)], wgt)]


def _oil(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    r = float(p.get("r_mm", 4.0))
    got = _ellipse_frame(ctx, p, r, float(p.get("aspect", 1.0)))
    if got is None:
        return []
    win, rho = got
    m = _smooth(1.0, 0.8, rho / 0.85 + 0.12 * _fbm(rng, win.shape, 0.4 * r / ctx.res, 3))
    m = m * float(p.get("strength", 1.0))
    return [_Stamp(win, [("rough", "add", -0.1, m), ("albedo", "mul", 0.95, m)], m)]


def _water_spot(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Thin whitish residue ring (coffee-ring) with a faint haze inside; ring_w_mm is its FWHM."""
    r = float(p.get("r_mm", 2.0))
    rw = rng.uniform(0.05, 0.2)
    if p.get("ring_w_mm") is not None:
        rw = float(p["ring_w_mm"]) / 2.355             # Gaussian sigma from the full width at half max
    k = float(p.get("strength", 1.0))
    got = _ellipse_frame(ctx, p, r + 4 * rw)
    if got is None:
        return []
    win, rho = got
    d = rho * (r + 4 * rw)
    win_u, win_v = ctx.coords(win)
    th = np.arctan2(win_v - float(p["v"]), win_u - float(p["u"]))
    re = r * (1 + 0.05 * np.clip(_angular_noise(rng, th), -2, 2))
    ring = k * np.exp(-0.5 * ((d - re) / rw) ** 2) * (d < r + 4 * rw)
    haze = k * 0.25 * _smooth(r, 0.6 * r, d)
    res_col = np.array([0.80, 0.80, 0.78], np.float32)
    ops = [("albedo", "mix", res_col, 0.35 * ring + 0.05 * haze), ("rough", "add", 0.25 * ring + 0.05 * haze, 1.0),
           ("metal", "mix", 0.0, 0.3 * ring)]
    return [_Stamp(win, ops, np.maximum(ring, haze))]


_INKS = {"black": (0.02, 0.02, 0.025), "blue": (0.02, 0.04, 0.25), "red": (0.35, 0.02, 0.03),
         "green": (0.02, 0.18, 0.05)}


def _marker(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Marker dot (r_mm) or stroke (length_mm, width_mm, angle_deg): ink, metallic 0, roughness 0.4."""
    u, v = float(p["u"]), float(p["v"])
    col = p.get("color")
    col = np.asarray(col if col is not None else _INKS[rng.choice(sorted(_INKS))], np.float32)
    if "length_mm" in p:
        hw = float(p.get("width_mm", 1.0)) / 2
        th = math.radians(float(p.get("angle_deg", 0.0)))
        half = float(p["length_mm"]) / 2 * np.array([math.cos(th), math.sin(th)])
        pts = np.array([[u, v] - half, [u, v] + half])
    else:
        hw = float(p.get("r_mm", 1.5))
        pts = np.array([[u, v], [u, v]])
    win = _bbox_window(ctx, pts, hw)
    if win is None:
        return []
    dist, _, _ = _polyline_field(ctx, win, pts, hw)
    m = _smooth(hw, hw - 0.05, dist) * np.clip(0.85 + 0.15 * _fbm(rng, win.shape, 0.5 / ctx.res, 3), 0, 1)
    m = m * float(p.get("strength", 1.0))
    return [_Stamp(win, [("albedo", "mix", col, m), ("metal", "mix", 0.0, m), ("rough", "mix", 0.4, m)], m)]


def _dust(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Tiny bright (lint) and dark (grit) specks over a disc (u, v, r_mm) or the whole plate.

    size_mm = [min, max] speck diameter (Gaussian sigma = d / 2.5); default sigma 0.01-0.06 mm.
    """
    size = p.get("size_mm")
    if "u" in p and "v" in p:
        cu, cv_, r = float(p["u"]), float(p["v"]), float(p.get("r_mm", 5.0))
        n = int(p.get("n", max(5, round(1.5 * r * r))))
    else:
        cu, cv_, r = None, None, 0.0
        n = int(p.get("n", rng.integers(50, 300)))
    out = []
    for _ in range(n):
        if cu is None:
            pu, pv = rng.uniform(0, ctx.W), rng.uniform(0, ctx.H)
        else:
            rr, a = r * math.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            pu, pv = cu + rr * math.cos(a), cv_ + rr * math.sin(a)
        sig = rng.uniform(float(size[0]), float(size[1])) / 2.5 if size else rng.uniform(0.01, 0.06)
        bright = rng.random() < 0.5
        win = ctx.disc(pu, pv, 3 * sig + ctx.res)
        if win is None:
            continue
        U, V = ctx.coords(win)
        g = np.exp(-0.5 * (np.hypot(U - pu, V - pv) / max(sig, 0.6 * ctx.res)) ** 2)
        ops = [("albedo", "mix", 0.85 if bright else 0.1, g), ("rough", "mix", 0.9, g),
               ("metal", "mix", 0.0, g), ("h", "add", 1.5 * sig * g, 1.0)]
        out.append(_Stamp(win, ops, g))
    return out


def _blend(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Blend-out rework: smooth 20:1 dish, sanded isotropic finish, roughness 0.35."""
    r = float(p["r_mm"]) if "r_mm" in p else float(p.get("d_mm", 10.0)) / 2
    depth = float(p.get("depth_mm", r / 20))
    got = _ellipse_frame(ctx, p, r, float(p.get("aspect", 1.0)))
    if got is None:
        return []
    win, rho = got
    dish = np.where(rho < 1, -depth * (1 - rho ** 2) ** 2, 0.0)
    sand = _smooth(1.0, 0.85, rho)
    iso = 0.0004 * _unit(_noise(rng, win.shape, 1.0) + 0.5 * _noise(rng, win.shape, 3.0))
    ops = [("h", "mul", 0.0, sand), ("h", "add", dish + sand * iso, 1.0), ("rough", "mix", 0.35, sand),
           ("albedo", "mul", 1.02, sand)] + _ground_coat(ctx, rng, sand)
    return [_Stamp(win, ops, sand)]


def _ground_coat(ctx: _Ctx, rng: np.random.Generator, m: np.ndarray) -> list[Op]:
    """A grind or blend takes the surface layer off: bright bare steel through mill scale
    (hot_rolled), plain steel grey without the zinc spangle (galvanised). Other finishes: none."""
    lo_hi = {"hot_rolled": (0.4, 0.55), "spangle": (0.45, 0.55)}.get(ctx.finish)
    if lo_hi is None:
        return []
    steel = np.float32(rng.uniform(*lo_hi)) * np.array([1.0, 1.0, 1.02], np.float32)
    return [("albedo", "mix", steel, 0.95 * m), ("metal", "mix", 1.0, 0.95 * m)]


def _burr(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Raised rough ring round an open hole's edge (an oval hole's ellipse), optionally an arc.

    Height h_mm (or height_mm; default 30-80 um), radial width w_mm (or width_mm), arc from
    start_deg over arc_deg (plate angles, +u toward +v; default the full ring).
    """
    a = b = alpha = 0.0
    if "hole" in p:
        hole = ctx.holes[p["hole"]]
        cu, cv_ = float(hole["u"]), float(hole["v"])
        b = float(hole["d_mm"]) / 2
        a = b * max(1.0, float(hole.get("ovality", 1.0)))       # major semi-axis along angle_deg
        alpha = math.radians(float(hole.get("angle_deg", 0.0)))
    else:
        cu, cv_, a = float(p["u"]), float(p["v"]), float(p["r_mm"])
        b = a
    hgt, bw = p.get("height_mm", p.get("h_mm")), p.get("width_mm", p.get("w_mm"))
    hgt = float(rng.uniform(0.03, 0.08) if hgt is None else hgt)
    bw = float(rng.uniform(0.15, 0.4) if bw is None else bw)
    arc = float(p.get("arc_deg", 360.0))
    win = ctx.disc(cu, cv_, a + bw)
    if win is None or arc <= 0 or hgt <= 0 or bw <= 0:
        return []
    U, V = ctx.coords(win)
    x, y = U - cu, V - cv_
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    r0 = a * b / np.maximum(np.hypot(b * np.cos(phi - alpha), a * np.sin(phi - alpha)), 1e-12)
    t = (r - r0) / bw
    prof = np.where((t > 0) & (t < 1), np.sin(np.pi * np.sqrt(np.clip(t, 0, 1))) ** 2, 0.0)  # steep at the edge
    jag = np.clip(0.55 + 0.45 * _angular_noise(rng, phi, 16)
                  + 0.2 * _noise(rng, win.shape, 2.0), 0, 1)
    m = prof * jag
    if arc < 360.0:
        rel = (np.degrees(phi) - float(p.get("start_deg", 0.0))) % 360.0
        e = min(10.0, arc / 4)
        m = m * _smooth(0.0, e, rel) * _smooth(arc, arc - e, rel)
    ops = [("h", "add", hgt * m, 1.0), ("rough", "add", 0.2, m), ("albedo", "mul", 1.05, m)]
    return [_Stamp(win, ops, m)]


def _label_residue(p: dict, ctx: _Ctx, rng: np.random.Generator) -> list[_Stamp]:
    """Sticker glue left on the part (unlabelled; a hard negative for stain): a rounded w x h
    patch along angle_deg, yellowish and duller, a darker glue line at its edge, stuck grit."""
    u, v, w, hh = float(p["u"]), float(p["v"]), float(p.get("w_mm", 15.0)), float(p.get("h_mm", 10.0))
    th = math.radians(float(p.get("angle_deg", 0.0)))
    c, s = math.cos(th), math.sin(th)
    eu, ev = abs(c) * w / 2 + abs(s) * hh / 2, abs(s) * w / 2 + abs(c) * hh / 2
    win = ctx.window(u - eu, v - ev, u + eu, v + ev)
    if win is None:
        return []
    U, V = ctx.coords(win)
    x, y = U - u, V - v
    S, T = x * c + y * s, -x * s + y * c
    rho = ((np.abs(S) / (w / 2)) ** 6 + (np.abs(T) / (hh / 2)) ** 6) ** (1 / 6)    # rounded rectangle
    # ragged by at most 0.1: nothing reaches past 1.15 x the half-sizes (sampler._cover_outline)
    rho_n = rho + 0.04 * np.clip(_fbm(rng, win.shape, max(2.0, 0.6 / ctx.res), 3), -2.5, 2.5)
    k = float(p.get("strength", 0.7))
    patch = np.clip(0.6 + 0.4 * _fbm(rng, win.shape, max(2.0, 2.0 / ctx.res), 3), 0, 1)
    m = k * _smooth(1.0, 0.92, rho_n) * patch
    edge = k * np.exp(-(((rho_n - 0.95) / 0.02) ** 2)) * _smooth(1.05, 1.0, rho_n)
    grit = m * _smooth(2.0, 2.6, _noise(rng, win.shape, 1.5))
    yellow = np.array([0.95, 0.93, 0.86], np.float32)
    ops = [("albedo", "mul", yellow, m), ("rough", "mix", rng.uniform(0.45, 0.65), 0.8 * m),
           ("metal", "mix", 0.4, 0.5 * m), ("h", "add", 0.004 * m, 1.0), ("albedo", "mul", 0.85, 0.6 * edge),
           ("rough", "add", 0.1, edge), ("albedo", "mix", 0.08, grit)]
    return [_Stamp(win, ops, np.maximum(m, edge))]


SHAPE_DISTRACTORS = {"blend": _blend, "burr": _burr}               # applied before the defects
SURFACE_DISTRACTORS = {"label_residue": _label_residue, "fingerprint": _fingerprint, "oil": _oil,
                       "water_spot": _water_spot,
                       "marker": _marker, "dust": _dust}          # applied on top of them
GEOMETRY_ONLY = {"glove"}                                         # Blender objects, nothing to paint


# ----------------------------------------------------------------------------- penetrant

def _bloom(dist: np.ndarray, core: float, bloom: float) -> np.ndarray:
    """1 within core of the line or centre, then a Gaussian falloff (sigma = bloom/2) reaching 0 at core + bloom."""
    x = np.maximum(dist - core, 0.0)
    if bloom <= 0:
        return (dist <= core).astype(np.float32)
    return np.exp(-0.5 * (x / (bloom / 2)) ** 2) * _smooth(bloom, 0.8 * bloom, x)


def _indication(ind: dict, ctx: _Ctx, rng: np.random.Generator, red=PENETRANT_RED) -> list[_Stamp]:
    """Red dye that fills the indication's footprint (geometry.footprint):

    - linear: full dye over the crack width, bleeding bloom_mm beyond it (outline: the line
      buffered by width/2 + bloom);
    - rounded: a dot of diameter d_mm (full dye in the inner 40 % of the radius).
    """
    p = ind.get("params") or {}
    inten = float(p.get("intensity", 0.85))
    if ind["cls"] == "linear_indication":
        pts = np.asarray(p["pts_mm"], float).reshape(-1, 2)
        core, bloom = float(p.get("width_mm", 0.0)) / 2, float(p.get("bloom_mm", 0.6))
        win = _bbox_window(ctx, pts, core + bloom)
        if win is None:
            return []
        dist, s, length = _polyline_field(ctx, win, pts, core + bloom)
        along = 1 + 0.2 * np.clip(_noise1d(rng, s, 1.0, 0.0, max(length, 1e-6)), -2, 2)
    elif ind["cls"] == "rounded_indication":
        u, v = float(p["u"]), float(p["v"])
        R = float(p.get("d_mm", 0.8)) / 2
        core, bloom = 0.4 * R, 0.6 * R
        win = ctx.disc(u, v, R)
        if win is None:
            return []
        U, V = ctx.coords(win)
        dist, along = np.hypot(U - u, V - v), 1.0
    else:
        raise ValueError(f"unknown penetrant indication class {ind['cls']!r}")
    m = np.clip(inten * _bloom(dist, core, bloom) * along, 0, 1)
    return [_Stamp(win, [("albedo", "mix", np.asarray(red, np.float32), m), ("rough", "mix", 0.7, m)], m)]


def _developer(maps: _Maps, ctx: _Ctx, dev: dict) -> None:
    """White developer over the whole face; the heights underneath are kept."""
    rng = ctx.rng("penetrant:developer")
    speck, tv = float(dev.get("speckle", 0.05)), float(dev.get("thickness_var", 0.1))
    fine = _unit(_noise(rng, ctx.shape, 1.0))
    low = np.clip(_unit(_noise(rng, ctx.shape, 4.0 / ctx.res)) / 2, -1, 1)
    alb = float(dev.get("albedo", 0.88)) * (1 + speck * np.clip(fine, -2, 2)) * (1 + 0.2 * tv * low)
    maps.albedo[...] = np.clip(alb, 0.0, 0.97)[..., None]           # a white coat, never a perfect reflector
    maps.rough[...] = np.clip(float(dev.get("roughness", 0.9)) + 0.03 * np.clip(fine, -2, 2), 0, 1)
    maps.metal[...] = 0.0
    maps.uv[...] = float(dev.get("uv_emit", 0.0))
    maps.h += 0.002 * _unit(_noise(rng, ctx.shape, 1.5)) + 0.004 * tv * low


def _edge_bleed(maps: _Maps, ctx: _Ctx, amount: float, red=PENETRANT_RED) -> None:
    """Unlabelled red bleed-out in a patchy band along the four plate edges."""
    rng = ctx.rng("penetrant:edge_bleed")
    red = np.asarray(red, np.float32)
    for edge in ("top", "bottom", "left", "right"):
        bw = rng.uniform(0.4, 1.2)
        depth = 5 * bw
        if edge == "top":
            win = ctx.window(0, 0, ctx.W, depth)
        elif edge == "bottom":
            win = ctx.window(0, ctx.H - depth, ctx.W, ctx.H)
        elif edge == "left":
            win = ctx.window(0, 0, depth, ctx.H)
        else:
            win = ctx.window(ctx.W - depth, 0, ctx.W, ctx.H)
        U, V = ctx.coords(win)
        U, V = np.broadcast_arrays(U, V)
        dist, pos, span = {"top": (V, U, ctx.W), "bottom": (ctx.H - V, U, ctx.W),
                           "left": (U, V, ctx.H), "right": (ctx.W - U, V, ctx.H)}[edge]
        width = bw * np.clip(1 + 0.4 * _noise1d(rng, pos, 2.0, 0.0, span), 0.3, 2.0)
        patch = np.clip(0.5 + 0.7 * _noise1d(rng, pos, 3.0, 0.0, span), 0, 1)
        m = np.clip(amount * patch * np.exp(-dist / width) * _smooth(depth, 0.8 * depth, dist), 0, 1)
        _apply(maps, _Stamp(win, [("albedo", "mix", red, m), ("rough", "mix", 0.75, m)]), None)


# ----------------------------------------------------------------------------- driver

def _distractors(maps: _Maps, ctx: _Ctx, table: dict) -> None:
    known = SHAPE_DISTRACTORS.keys() | SURFACE_DISTRACTORS.keys() | GEOMETRY_ONLY
    for i, dis in enumerate(ctx.plan.get("distractors") or []):
        kind = dis.get("kind")
        p = dis.get("params") or {}
        if kind in table:
            stamps = table[kind](p, ctx, ctx.rng(f"distractor:{i}:{kind}"))
            _put(maps, ctx, stamps, None, f"distractor {i} {kind}", float(p.get("uv_emit", 0.0)))
        elif kind not in known and table is SURFACE_DISTRACTORS:
            log.warning("distractor %d: unknown kind %r skipped", i, kind)


def _elements(maps: _Maps, ctx: _Ctx, clean: bool) -> _Maps:
    """Stamp distractors, defects and the penetrant coat onto the base maps (in place)."""
    _distractors(maps, ctx, SHAPE_DISTRACTORS)
    for d in ctx.plan.get("defects") or []:
        if clean and d.get("label", True):
            continue
        key = (d["cls"], d["kind"])
        if key not in DEFECTS:
            raise ValueError(f"defect {d.get('id')}: no texture generator for {key[0]}.{key[1]}")
        stamps = DEFECTS[key](d["params"], ctx, ctx.rng(f"defect:{d['id']}"))
        _put(maps, ctx, stamps, (d.get("footprint") or {}).get("polys_mm"), f"defect {d['id']} {key[0]}.{key[1]}")
    _distractors(maps, ctx, SURFACE_DISTRACTORS)
    pen = ctx.plan.get("penetrant")
    if pen:
        red = pen.get("dye_color") or PENETRANT_RED
        _developer(maps, ctx, pen.get("developer") or {})
        if not clean:
            for ind in pen.get("indications") or []:
                p = ind.get("params") or {}
                _put(maps, ctx, _indication(ind, ctx, ctx.rng(f"indication:{ind['id']}"), red),
                     (ind.get("footprint") or {}).get("polys_mm"), f"indication {ind['id']}",
                     float(p.get("uv_emit", 0.0)))
        if float(pen.get("edge_bleed") or 0.0) > 0:
            _edge_bleed(maps, ctx, float(pen["edge_bleed"]), red)
    return maps


def render_maps(plan: dict, clean: bool = False) -> dict[str, np.ndarray]:
    """Float maps of the front face: height (mm), albedo (linear RGB), roughness, metallic, uv_emit.

    clean=True leaves out labelled defects and penetrant indications; the finish,
    distractors, developer and edge bleed are identical to the normal maps.
    """
    ctx = _Ctx(plan)
    return _elements(_base(ctx), ctx, clean).result()


def _imwrite(path: Path, img: np.ndarray) -> None:
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
        raise OSError(f"could not write {path}")


def _write_pass(m: dict[str, np.ndarray], ctx: _Ctx, work: Path, names: dict[str, str], suffix: str,
                with_uv: bool) -> dict[str, Path]:
    def path(key: str) -> Path:
        p = Path(names[key])
        return work / f"{p.stem}{suffix}{p.suffix}"

    out = {}
    n16 = encode_normal16(normal_from_height(m["height"], (ctx.ru, ctx.rv)))
    out["normal"] = path("normal")
    _imwrite(out["normal"], cv2.cvtColor(n16, cv2.COLOR_RGB2BGR))
    out["albedo"] = path("albedo")
    _imwrite(out["albedo"], cv2.cvtColor(linear_to_srgb8(m["albedo"]), cv2.COLOR_RGB2BGR))
    for key, arr in (("roughness", m["roughness"]), ("metallic", m["metallic"])) + ((("uv_emit", m["uv_emit"]),) if with_uv else ()):
        out[key] = path(key)
        _imwrite(out[key], np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return {(k + suffix): v for k, v in out.items()}


def write_maps(plan: dict, work_dir: str | Path) -> dict[str, Path]:
    """Write normal/albedo/rough/metal (+ uvemit if non-zero) PNGs, and *_clean.png if render.clean_pass.

    Returns {map key: path}, keys "normal", "albedo", "roughness", "metallic", "uv_emit"
    and the same with a "_clean" suffix.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    names = dict(TEXTURE_FILES)
    names.update({k: v for k, v in (plan.get("textures") or {}).items() if k in names and v})
    ctx = _Ctx(plan)
    base = _base(ctx)
    want_clean = bool((plan.get("render") or {}).get("clean_pass"))
    m = _elements(base.copy() if want_clean else base, ctx, clean=False).result()
    with_uv = bool(m["uv_emit"].any())
    out = _write_pass(m, ctx, work, names, "", with_uv)
    if want_clean:
        del m
        out.update(_write_pass(_elements(base, ctx, clean=True).result(), ctx, work, names, "_clean", with_uv))
    return out
