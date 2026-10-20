"""Synthetic flat metal plate seen by the 3 cameras under each white light, plus room light.

Lets the whole pipeline run on a laptop before the rig exists. The physics is simple on
purpose, but it keeps the signatures the classifier relies on:
- relief (dent, bump) changes the shading (Lambertian, so photometric stereo works),
  in a direction set by each light;
- fine relief (scratch, dust) scatters light, more under grazing light, more towards the
  grazing camera, and a little more in blue;
- cracks and pits trap light, so they are dark from every camera;
- stains change the colour (albedo) by the same amount from every camera.
Room light ("ambient") is random per part - level, colour and gradient - and flickers a
little between captures. It is in every frame, including the dark frame, so subtracting
the dark frame removes it; that is how the open (unboxed) rig copes with the room.

The penetrant mode renders a plate after the red-dye process: white developer with red
indications bled out of cracks, slits and pores, plus a faint pink smear from poor removal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np

from .config import BGR_INDEX, CHANNELS, DARK, VIEWS, Config, Grid, LightSpec
from .hardware.cameras import Camera
from .hardware.lights import Lights

DEFECT_KINDS = ("scratch", "dent", "bump", "crack", "pit", "stain", "dust")
INDICATION_KINDS = ("crack", "slit", "pore", "pore_line", "smear")

# Size ranges in mm: length for scratch, crack and slit, radius for the rest (pore: diameter).
SIZE_RANGE = {
    "scratch": (4.0, 14.0),
    "crack": (5.0, 12.0),
    "dent": (2.5, 4.0),
    "bump": (2.0, 3.5),
    "pit": (0.35, 0.7),
    "stain": (2.5, 5.0),
    "dust": (0.2, 0.35),
    "slit": (3.0, 10.0),
    "pore": (0.6, 3.0),
    "pore_line": (1.8, 2.4),  # diameter of each of the 4 pores (relevant: > 1.6 mm)
    "smear": (4.0, 7.0),
}

BASE_ALBEDO = {"r": 0.60, "g": 0.61, "b": 0.63}
STAIN_LOSS = {"r": 0.15, "g": 0.40, "b": 0.55}  # rust-like: red is lost least
DEVELOPER = {"r": 0.66, "g": 0.67, "b": 0.68}  # white developer (exposure set lower for it on the rig)
DYE = {"r": 0.58, "g": 0.10, "b": 0.13}  # red penetrant bled into the developer
SCATTER_COLOUR = {"r": 0.78, "g": 1.0, "b": 1.33}  # roughness scatter ~ 1/wavelength^2
# Camera channel mixing under white light (rows: camera r, g, b; columns: light r, g, b).
MIX = np.array([[0.92, 0.06, 0.02], [0.05, 0.90, 0.05], [0.02, 0.06, 0.92]], np.float32)
BACKGROUND = 0.04  # matte black jig


@dataclass
class SimDefect:
    kind: str
    x_mm: float
    y_mm: float
    size_mm: float
    angle_deg: float = 0.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _segment_distance(xx, yy, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / max(dx * dx + dy * dy, 1e-12), 0.0, 1.0)
    return np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))


def _crack_path(d: SimDefect) -> list[np.ndarray]:
    """A wandering main line with one branch, centred on the defect position."""
    rng = np.random.default_rng(d.seed)
    step = d.size_mm / 12
    ang = np.radians(d.angle_deg)
    main = [np.zeros(2)]
    for _ in range(12):
        ang += rng.normal(0.0, np.radians(25))
        main.append(main[-1] + step * np.array([np.cos(ang), np.sin(ang)]))
    branch = [main[6]]
    bang = np.radians(d.angle_deg) + rng.choice([-1, 1]) * np.radians(60)
    for _ in range(5):
        bang += rng.normal(0.0, np.radians(20))
        branch.append(branch[-1] + 0.8 * step * np.array([np.cos(bang), np.sin(bang)]))
    offset = np.array([d.x_mm, d.y_mm]) - np.mean(main + branch, axis=0)
    return [np.array(main) + offset, np.array(branch) + offset]


class Surface:
    """Height, scatter and per-channel albedo of one plate, on the grid."""

    def __init__(self, grid: Grid, rng: np.random.Generator, base: dict[str, float] = BASE_ALBEDO, texture: float = 0.02):
        self.grid = grid
        self.rng = rng
        h, w = grid.shape
        mpp = grid.mm_per_px
        self.height = np.zeros((h, w), np.float32)
        self.scatter = np.zeros((h, w), np.float32)
        # Brushed finish: streaks a few mm long along x, each plate different.
        tex = rng.standard_normal((h, w)).astype(np.float32)
        tex = cv2.GaussianBlur(tex, (0, 0), sigmaX=2.5 / mpp, sigmaY=max(0.08 / mpp, 0.5))
        tex /= tex.std() + 1e-9
        self.albedo = {c: (base[c] * (1.0 + texture * tex)).astype(np.float32) for c in CHANNELS}
        self.plate = grid.plate_mask(0.0)
        xs = (np.arange(w, dtype=np.float32)) * mpp - grid.border_mm
        ys = (np.arange(h, dtype=np.float32)) * mpp - grid.border_mm
        self.xx, self.yy = np.meshgrid(xs, ys)

    def _window(self, x_mm: float, y_mm: float, r_mm: float):
        g = self.grid
        x0, y0 = g.mm_to_px(x_mm - r_mm, y_mm - r_mm)
        x1, y1 = g.mm_to_px(x_mm + r_mm, y_mm + r_mm)
        h, w = g.shape
        sl = (
            slice(max(int(y0), 0), min(int(np.ceil(y1)) + 1, h)),
            slice(max(int(x0), 0), min(int(np.ceil(x1)) + 1, w)),
        )
        return sl, self.xx[sl], self.yy[sl]

    def _darken(self, sl, p, amount):
        for c in CHANNELS:
            self.albedo[c][sl] *= 1.0 - amount * p

    def add(self, d: SimDefect) -> None:
        getattr(self, f"_add_{d.kind}")(d)

    def _add_scratch(self, d: SimDefect) -> None:
        th, half = np.radians(d.angle_deg), d.size_mm / 2
        x0, y0 = d.x_mm - half * np.cos(th), d.y_mm - half * np.sin(th)
        x1, y1 = d.x_mm + half * np.cos(th), d.y_mm + half * np.sin(th)
        sl, xx, yy = self._window(d.x_mm, d.y_mm, half + 1.0)
        p = np.exp(-0.5 * (_segment_distance(xx, yy, x0, y0, x1, y1) / 0.12) ** 2)
        self.height[sl] -= 0.004 * p
        self.scatter[sl] += 0.25 * p
        self._darken(sl, p, 0.03)

    def _add_crack(self, d: SimDefect) -> None:
        sl, xx, yy = self._window(d.x_mm, d.y_mm, d.size_mm)
        dist = np.full(xx.shape, np.inf, np.float32)
        for line in _crack_path(d):
            for a, b in zip(line[:-1], line[1:]):
                dist = np.minimum(dist, _segment_distance(xx, yy, a[0], a[1], b[0], b[1]))
        p = np.exp(-0.5 * (dist / 0.09) ** 2)
        self._darken(sl, p, 0.7)
        self.height[sl] -= 0.02 * p
        self.scatter[sl] += 0.03 * p

    def _relief(self, d: SimDefect, sign: float) -> None:
        sigma = d.size_mm / 2
        sl, xx, yy = self._window(d.x_mm, d.y_mm, 3.2 * sigma)
        g = np.exp(-0.5 * ((xx - d.x_mm) ** 2 + (yy - d.y_mm) ** 2) / sigma**2)
        self.height[sl] += sign * 0.18 * d.size_mm * g  # depth = 18 % of radius (punch-mark scale)
        self.scatter[sl] += 0.01 * g

    def _add_dent(self, d: SimDefect) -> None:
        self._relief(d, -1.0)

    def _add_bump(self, d: SimDefect) -> None:
        self._relief(d, +1.0)

    def _add_pit(self, d: SimDefect) -> None:
        sl, xx, yy = self._window(d.x_mm, d.y_mm, d.size_mm + 0.5)
        r = np.hypot(xx - d.x_mm, yy - d.y_mm)
        p = 1.0 / (1.0 + np.exp((r - d.size_mm) / 0.05))
        self._darken(sl, p, 0.6)
        self.height[sl] -= 0.05 * p
        self.scatter[sl] += 0.02 * p

    def _add_stain(self, d: SimDefect) -> None:
        rng = np.random.default_rng(d.seed)
        sl, xx, yy = self._window(d.x_mm, d.y_mm, 1.6 * d.size_mm)
        noise = rng.standard_normal(xx.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=d.size_mm / 3 / self.grid.mm_per_px)
        noise /= noise.std() + 1e-9
        base = np.exp(-0.5 * ((xx - d.x_mm) ** 2 + (yy - d.y_mm) ** 2) / (d.size_mm / 1.3) ** 2)
        m = np.clip((base * (1.0 + 0.5 * noise) - 0.35) / 0.4, 0.0, 1.0)
        for c in CHANNELS:
            self.albedo[c][sl] *= 1.0 - STAIN_LOSS[c] * m

    def _add_dust(self, d: SimDefect) -> None:
        sl, xx, yy = self._window(d.x_mm, d.y_mm, 4 * d.size_mm)
        p = np.exp(-0.5 * ((xx - d.x_mm) ** 2 + (yy - d.y_mm) ** 2) / d.size_mm**2)
        self.height[sl] += 0.03 * p
        self.scatter[sl] += 0.5 * p
        self._darken(sl, p, -0.05)

    # ---- red-dye penetrant indications (on a developed plate) -------------------------

    def _dye(self, sl, m) -> None:
        m = np.clip(m, 0.0, 1.0)
        for c in CHANNELS:
            self.albedo[c][sl] = self.albedo[c][sl] * (1.0 - m) + DYE[c] * m

    def add_indication(self, d: SimDefect) -> None:
        if d.kind == "crack":
            sl, xx, yy = self._window(d.x_mm, d.y_mm, d.size_mm)
            dist = np.full(xx.shape, np.inf, np.float32)
            for line in _crack_path(d):
                for a, b in zip(line[:-1], line[1:]):
                    dist = np.minimum(dist, _segment_distance(xx, yy, a[0], a[1], b[0], b[1]))
            self._dye(sl, 0.9 * np.exp(-0.5 * (dist / 0.2) ** 2))  # bleed-out ~0.5 mm wide
        elif d.kind == "slit":
            th, half = np.radians(d.angle_deg), d.size_mm / 2
            ends = (d.x_mm - half * np.cos(th), d.y_mm - half * np.sin(th), d.x_mm + half * np.cos(th), d.y_mm + half * np.sin(th))
            sl, xx, yy = self._window(d.x_mm, d.y_mm, half + 1.0)
            self._dye(sl, 0.9 * np.exp(-0.5 * (_segment_distance(xx, yy, *ends) / 0.18) ** 2))
        elif d.kind == "pore":
            r0 = d.size_mm / 2
            sl, xx, yy = self._window(d.x_mm, d.y_mm, r0 + 1.0)
            r = np.hypot(xx - d.x_mm, yy - d.y_mm)
            self._dye(sl, 0.9 / (1.0 + np.exp((r - r0) / 0.06)))
        elif d.kind == "pore_line":
            th, gap = np.radians(d.angle_deg), 1.0  # 4 pores, ~1 mm edge-to-edge
            pitch = d.size_mm + gap
            for i in range(4):
                o = (i - 1.5) * pitch
                self.add_indication(SimDefect("pore", d.x_mm + o * np.cos(th), d.y_mm + o * np.sin(th), d.size_mm))
        elif d.kind == "smear":
            sl, xx, yy = self._window(d.x_mm, d.y_mm, 2.5 * d.size_mm)
            self._dye(sl, 0.06 * np.exp(-0.5 * ((xx - d.x_mm) ** 2 + (yy - d.y_mm) ** 2) / d.size_mm**2))
        else:
            raise ValueError(f"unknown indication kind {d.kind!r}")

    def shading(self, light: LightSpec) -> np.ndarray:
        gy, gx = np.gradient(self.height, self.grid.mm_per_px)
        lx, ly, lz = light.direction()
        ndotl = (-gx * lx - gy * ly + lz) / np.sqrt(gx * gx + gy * gy + 1.0)
        return np.clip(ndotl, 0.0, None).astype(np.float32)


def random_defects(
    rng: np.random.Generator,
    grid: Grid,
    kinds: Iterable[str],
    margin_mm: float = 10.0,
    min_gap_mm: float = 16.0,
) -> list[SimDefect]:
    placed: list[SimDefect] = []
    for kind in kinds:
        for _ in range(500):
            x = rng.uniform(margin_mm, grid.plate_w_mm - margin_mm)
            y = rng.uniform(margin_mm, grid.plate_h_mm - margin_mm)
            if all(np.hypot(x - p.x_mm, y - p.y_mm) >= min_gap_mm for p in placed):
                break
        else:
            raise RuntimeError("no room left on the plate for another defect")
        lo, hi = SIZE_RANGE[kind]
        placed.append(
            SimDefect(kind, float(x), float(y), float(rng.uniform(lo, hi)), float(rng.uniform(0, 180)), int(rng.integers(2**31)))
        )
    return placed


class _SimLights(Lights):
    def __init__(self, rig: "SimRig"):
        self._rig = rig

    def on(self, name: str) -> None:
        if name not in self._rig.cfg.lights:
            raise KeyError(name)
        self._rig.lit = name

    def off(self) -> None:
        self._rig.lit = None


class _SimCamera(Camera):
    def __init__(self, rig: "SimRig", view: str, seed: int):
        self._rig, self._view = rig, view
        self._rng = np.random.default_rng(seed)  # one generator per camera: captures run in threads

    def capture(self) -> np.ndarray:
        return self._rig.frame(self._view, self._rng)


class SimRig:
    """Drop-in replacement for the real rig: .lights, .cameras, .place_part(), .place_developed()."""

    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.grid = cfg.grid
        sim = cfg.sim
        self.rng = np.random.default_rng(sim.get("seed", 0) if seed is None else seed)
        self.jitter_mm = float(sim.get("jitter_mm", 0.15))
        self.jitter_deg = float(sim.get("jitter_deg", 0.1))
        self.light_drift = float(sim.get("light_drift", 0.004))
        self.ambient_max = float(sim.get("ambient_max", 0.15))
        self.flicker = float(sim.get("flicker", 0.01))
        self.views = {}
        self.true_h = {}
        for view in VIEWS:
            cam = cfg.cameras[view]["sim"]
            corners = np.array(cam["plate_corners_px"], np.float32)
            H = cv2.getPerspectiveTransform(self.grid.plate_corners_px(), corners)
            self.true_h[view] = H
            self.views[view] = {
                "size": tuple(cam["image_size"]),
                "view_gain": float(cam["view_gain"]),
                "scatter_gain": float(cam["scatter_gain"]),
                "blur": self._antialias_sigma(H),
            }
        self.top_light = cfg.pair_light("top")
        self.lit: str | None = None
        self.fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.ambient = np.zeros((*self.grid.shape, 3), np.float32)
        self.hotspot = self._hotspot()
        self.defects: list[SimDefect] = []
        self.lights = _SimLights(self)
        self.cameras = {v: _SimCamera(self, v, int(self.rng.integers(2**31))) for v in VIEWS}

    def _antialias_sigma(self, H: np.ndarray) -> tuple[float, float]:
        """Blur that mimics pixel integration when a camera sees the grid at lower resolution."""
        cx, cy = self.grid.width_px / 2, self.grid.height_px / 2
        pts = cv2.perspectiveTransform(np.array([[[cx, cy], [cx + 1, cy], [cx, cy + 1]]], np.float32), H)[0]
        sx, sy = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[0])
        return (0.5 / sx if sx < 1 else 0.0, 0.5 / sy if sy < 1 else 0.0)

    def _hotspot(self) -> np.ndarray:
        """Glare of the top light next to the top camera, seen on the shiny plate (fixed in space)."""
        g = self.grid
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = g.mm_to_px(g.plate_w_mm / 2, g.plate_h_mm / 2)
        r_mm = np.hypot(xx - cx, yy - cy) * g.mm_per_px
        return (0.08 * np.exp(-0.5 * (r_mm / 12.0) ** 2) * g.plate_mask(0.0)).astype(np.float32)

    # ---- parts ---------------------------------------------------------------------------

    def place_part(self, defects: None | str | Iterable[str] | Iterable[SimDefect] = None) -> list[SimDefect]:
        """Render a new plate. defects: None = good part, "random" = 1-3 random
        defects, a list of kinds, or explicit SimDefect objects."""
        rng = np.random.default_rng(self.rng.integers(2**63))
        placed = self._choose(rng, defects, DEFECT_KINDS)
        surface = Surface(self.grid, rng)
        for d in placed:
            surface.add(d)
        self._render(surface, rng)
        self.defects = placed
        return placed

    def place_developed(self, indications: None | str | Iterable[str] | Iterable[SimDefect] = None) -> list[SimDefect]:
        """Render a plate after the red-dye penetrant process (white developer + red indications)."""
        rng = np.random.default_rng(self.rng.integers(2**63))
        placed = self._choose(rng, indications, INDICATION_KINDS)
        surface = Surface(self.grid, rng, base=DEVELOPER, texture=0.015)
        for d in placed:
            surface.add_indication(d)
        self._render(surface, rng)
        self.defects = placed
        return placed

    def _choose(self, rng, items, kinds) -> list[SimDefect]:
        if items is None:
            return []
        if isinstance(items, str) and items == "random":
            return random_defects(rng, self.grid, rng.choice(kinds, size=rng.integers(1, 4), replace=False))
        items = list(items)
        return items if all(isinstance(d, SimDefect) for d in items) else random_defects(rng, self.grid, items)

    def _render(self, surface: Surface, rng: np.random.Generator) -> None:
        g = self.grid
        cx, cy = g.width_px / 2, g.height_px / 2
        M = cv2.getRotationMatrix2D((cx, cy), float(rng.normal(0, self.jitter_deg)), 1.0)
        M[:, 2] += rng.normal(0, self.jitter_mm, 2) / g.mm_per_px

        def place(img: np.ndarray, border: float = 0.0) -> np.ndarray:
            return cv2.warpAffine(img, M, g.size, flags=cv2.INTER_LINEAR, borderValue=border)

        plate = place(surface.plate.astype(np.float32)) > 0.5
        albedo = np.dstack([surface.albedo[c] for c in CHANNELS])
        xn = (surface.xx - g.plate_w_mm / 2) / g.plate_w_mm
        yn = (surface.yy - g.plate_h_mm / 2) / g.plate_h_mm

        self.fields = {}
        for name, light in self.cfg.lights.items():
            drift = 1.0 + rng.normal(0, self.light_drift)
            away = light.away_xy()
            if light.elevation_deg > 80:
                illum = 1.0 - 0.4 * (xn**2 + yn**2)  # overhead light falls off towards the edges
            else:
                illum = 1.0 - 0.15 * (xn * away[0] + yn * away[1])  # falls off away from the lamp
            illum = (drift * illum).astype(np.float32)
            shade = surface.shading(light)
            base = illum[..., None] * albedo * shade[..., None]
            scat = illum * np.cos(np.radians(light.elevation_deg)) * surface.scatter
            base = np.dstack([place(base[..., i]) for i in range(3)])
            bg = BACKGROUND * np.sin(np.radians(light.elevation_deg))
            base[~plate] = bg
            scat = np.where(plate, place(scat), 0.0).astype(np.float32)
            self.fields[name] = (base.astype(np.float32), scat)

        # Room light: random level, colour and gradient per part, the same in every capture.
        level = rng.uniform(0.0, self.ambient_max)
        t = rng.uniform(-0.2, 0.2)
        tint = np.array([1 + t, 1.0, 1 - t], np.float32)
        gdir = rng.uniform(0, 2 * np.pi)
        grad = 1.0 + rng.uniform(0, 0.3) * (xn * np.cos(gdir) + yn * np.sin(gdir))
        amb = level * grad[..., None] * tint * albedo
        amb = np.dstack([place(amb[..., i]) for i in range(3)])
        amb[~plate] = level * tint * BACKGROUND
        self.ambient = amb.astype(np.float32)

    def frame(self, view: str, rng: np.random.Generator) -> np.ndarray:
        v = self.views[view]
        w, h = v["size"]
        img = self.ambient * (1.0 + rng.normal(0, self.flicker))
        if self.lit is not None:
            base, scat = self.fields[self.lit]
            led = v["view_gain"] * base + v["scatter_gain"] * scat[..., None] * np.array([SCATTER_COLOUR[c] for c in CHANNELS], np.float32)
            if view == "top" and self.lit == self.top_light:
                led = led + self.hotspot[..., None]
            img = img + led
        sx, sy = v["blur"]
        if sx or sy:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=max(sx, 0.01), sigmaY=max(sy, 0.01))
        img = cv2.warpPerspective(img, self.true_h[view], (w, h), flags=cv2.INTER_LINEAR, borderValue=(BACKGROUND,) * 3)
        noise = rng.standard_normal(img.shape).astype(np.float32) * np.sqrt(0.004**2 + 5e-5 * np.clip(img, 0, None))
        img = np.clip((img + noise) @ MIX.T, 0.0, 1.0)  # rgb
        frame = np.zeros((h, w, 3), np.float32)
        for i, c in enumerate(CHANNELS):
            frame[..., BGR_INDEX[c]] = img[..., i]
        return np.round(frame * 255).astype(np.uint8)

    def close(self) -> None:
        self.lights.off()


__all__ = ["DARK", "DEFECT_KINDS", "INDICATION_KINDS", "SimDefect", "SimRig", "Surface"]
