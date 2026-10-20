"""3D shape of the surface from the top camera under several white lights (photometric stereo).

How bright a point looks depends on how its surface faces each light. With three or
more lights from different directions (not all in one vertical plane) the surface
normal at every pixel can be solved for, and integrating the slopes gives a height map.

- "full": 3+ shape lights from different azimuths. Gives normal maps (ps_nx, ps_ny)
  and, during inspection, a height map relative to the good parts.
- "slope-only": the shape lights all lie in one vertical plane (e.g. every light on
  the camera arch). Only the slope along that plane can be recovered (ps_sa), so there
  is no height map; dents and bumps are still told apart by their shading direction.
- "none": fewer than two usable lights.

Images are normalised by their plate median first, so different LED brightnesses
cancel. Slow brightness fall-off across the plate shows up as a constant tilt that the
golden reference absorbs, because it is the same on every part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LUMA, SHAPE_FULL_KEYS, SHAPE_SLOPE_KEYS, Config, key


@dataclass
class ShapeSetup:
    mode: str  # "full", "slope-only" or "none"
    lights: tuple[str, ...]
    L: np.ndarray  # k x 3 unit vectors towards the lights
    pinv: np.ndarray | None  # 3 x k (full) or 2 x k (slope-only)
    axis: np.ndarray | None  # in-plane unit vector (x, y) of the slope for slope-only
    reason: str

    @property
    def map_keys(self) -> tuple[str, ...]:
        return {"full": SHAPE_FULL_KEYS, "slope-only": SHAPE_SLOPE_KEYS}.get(self.mode, ())

    def describe(self) -> dict:
        out = {"mode": self.mode, "lights": list(self.lights), "reason": self.reason}
        if self.axis is not None:
            out["slope_axis_deg"] = round(float(np.degrees(np.arctan2(self.axis[1], self.axis[0]))) % 180, 1)
        return out


def shape_setup(cfg: Config) -> ShapeSetup:
    names = cfg.shape_lights
    if len(names) < 2:
        return ShapeSetup("none", names, np.zeros((0, 3)), None, None, "fewer than 2 lights marked shape: true")
    L = np.array([cfg.lights[n].direction() for n in names])
    s = np.linalg.svd(L, compute_uv=False)
    if len(names) >= 3 and s[2] / s[0] > 0.15:
        return ShapeSetup("full", names, L, np.linalg.pinv(L), None, f"{len(names)} lights from different directions")

    horizontal = L[:, :2]
    _, hs, vt = np.linalg.svd(horizontal)
    if hs[0] < 1e-3:
        return ShapeSetup("none", names, L, None, None, "all shape lights are directly overhead")
    axis = vt[0] / np.linalg.norm(vt[0])
    basis = np.array([[axis[0], axis[1], 0.0], [0.0, 0.0, 1.0]])
    L2 = L @ basis.T
    s2 = np.linalg.svd(L2, compute_uv=False)
    if s2[1] / s2[0] < 0.1:
        return ShapeSetup("none", names, L, None, None, "shape lights are too close together")
    return ShapeSetup(
        "slope-only",
        names,
        L,
        np.linalg.pinv(L2),
        axis,
        "all shape lights lie in one vertical plane, so only the slope along it can be measured; "
        "add lights at other azimuths for a full 3D map",
    )


def luminance(stack: dict[str, np.ndarray], view: str, light: str) -> np.ndarray:
    """Brightness map of one camera under one light (built from r, g, b if stored in colour)."""
    k = key(view, light)
    if k in stack:
        return stack[k]
    return sum(LUMA[c] * stack[f"{k}_{c}"] for c in LUMA).astype(np.float32)


def shape_maps(stack: dict[str, np.ndarray], setup: ShapeSetup, roi: np.ndarray) -> dict[str, np.ndarray]:
    """Normal maps from the top camera's images under the shape lights."""
    if setup.mode == "none":
        return {}
    imgs = []
    for name, l in zip(setup.lights, setup.L):
        img = luminance(stack, "top", name)
        imgs.append(img * (l[2] / max(float(np.median(img[roi])), 1e-6)))
    I = np.stack(imgs).reshape(len(imgs), -1)  # k x N
    g = setup.pinv @ I
    shape = imgs[0].shape
    if setup.mode == "full":
        norm = np.maximum(np.linalg.norm(g, axis=0), 1e-6)
        nz = np.maximum(g[2] / norm, 0.05)
        # Store slopes n/nz (= -dh/dx, -dh/dy): linear in the surface gradient.
        return {
            "ps_nx": (g[0] / norm / nz).reshape(shape).astype(np.float32),
            "ps_ny": (g[1] / norm / nz).reshape(shape).astype(np.float32),
        }
    ga, gz = g
    return {"ps_sa": (ga / np.maximum(gz, 0.05)).reshape(shape).astype(np.float32)}


def integrate_slopes(p: np.ndarray, q: np.ndarray, mm_per_px: float) -> np.ndarray:
    """Height (mm) from slopes p = dh/dx, q = dh/dy (mm per mm), Frankot-Chellappa (FFT least squares)."""
    h, w = p.shape
    wx = np.fft.fftfreq(w) * 2 * np.pi
    wy = np.fft.fftfreq(h) * 2 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    denom = WX**2 + WY**2
    denom[0, 0] = 1.0
    Z = mm_per_px * (-1j * WX * np.fft.fft2(p) - 1j * WY * np.fft.fft2(q)) / denom
    Z[0, 0] = 0.0
    return np.real(np.fft.ifft2(Z)).astype(np.float32)


def height_map(stack: dict[str, np.ndarray], golden_mean: dict[str, np.ndarray], roi: np.ndarray, mm_per_px: float) -> np.ndarray | None:
    """Height of this part relative to the golden reference (mm, + = raised). Full mode only."""
    if "ps_nx" not in stack or "ps_nx" not in golden_mean:
        return None
    # ps_n* = -dh/dx; deviations from the golden mean remove the fixed tilt of the light fall-off.
    p = -(stack["ps_nx"] - golden_mean["ps_nx"])
    q = -(stack["ps_ny"] - golden_mean["ps_ny"])
    p = np.where(roi, p, 0.0)
    q = np.where(roi, q, 0.0)
    return integrate_slopes(p, q, mm_per_px)
