"""LED rig of the handheld, taken from the Fusion generator (the single source of truth).

The generator (hardware/enclosure/ayas_handheld_fusion.py) imports ``adsk``, which only
exists inside Fusion. We load it with stub ``adsk`` modules and call only its pure-geometry
helpers: P, DISPLAYS, layout(), pod_frame(), uv_frame(), white_columns(), uv_led_points().

Model frame (mm): X across (+X is the right-hand side seen from the FRONT), Y up, Z forward
(toward the part). The lens sits at model (0, disp_cy, z_lens).

Camera frame (OpenCV, mm, DESIGN 3.3):
- model_x_is_image_left true (upright camera board): x = -X, y = -(Y - disp_cy), z = Z - z_lens.
- false (board turned 180 deg about the optical axis): x = +X, y = +(Y - disp_cy), z = Z - z_lens.
Both are proper rotations, so directions transform the same way without the offset.

White LED power (DESIGN_STEEL C): each white LED is a Lambertian emitter of
capture.yaml leds.white_lumens_per_led, so its Blender power is lighting.led_power_w(lumens)
(4 pi I / 683 W with I = lumens / pi cd). twin.yaml rig.white.power_w is only the fallback
when no capture config is found. UV LEDs keep rig.uv.power_w.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .lighting import led_power_w

ADSK_STUBS = ("adsk", "adsk.core", "adsk.fusion")


@dataclass
class Led:
    name: str
    group: str               # L | R (white, by image side) | UV
    kind: str                # white | uv
    p_cam_mm: np.ndarray     # position in the camera frame (mm)
    d_cam: np.ndarray        # unit emission direction in the camera frame
    params: dict[str, Any]   # light params from configs/twin.yaml rig.white / rig.uv


@dataclass
class Rig:
    leds: list[Led]
    device_box_mm: dict[str, list[float]] = field(default_factory=dict)
    model_x_is_image_left: bool = True


def load_fusion_generator(path: Path) -> types.ModuleType:
    """Import the Fusion generator script with stub adsk modules (no Fusion needed)."""
    saved = {n: sys.modules.get(n) for n in ADSK_STUBS}
    stubs = {n: types.ModuleType(n) for n in ADSK_STUBS}
    stubs["adsk"].core, stubs["adsk"].fusion = stubs["adsk.core"], stubs["adsk.fusion"]
    sys.modules.update(stubs)
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                  # keep hardware/enclosure free of new .pyc files
    try:
        spec = importlib.util.spec_from_file_location("_ayas_fusion_generator", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    return module


def model_to_cam(d: dict, x_is_image_left: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(M, origin) with p_cam = M @ (p_model - origin) and d_cam = M @ d_model."""
    M = np.diag([-1.0, -1.0, 1.0]) if x_is_image_left else np.diag([1.0, 1.0, 1.0])
    return M, np.array([0.0, d["disp_cy"], d["z_lens"]])


def _frame_point(frame, x: float, y: float) -> np.ndarray:
    o, ex, ey, _ = (np.asarray(v, float) for v in frame)
    return o + x * ex + y * ey


def led_model_points(fusion: types.ModuleType, p: dict | None = None) -> tuple[list[dict], dict]:
    """Every LED in the model frame (side +1/-1, kind, index, pos mm, dir), and the layout dict."""
    p = dict(p or fusion.P)
    p.update(fusion.DISPLAYS[p["display"]])
    d = fusion.layout(p)
    out = []
    ymid = (d["ytop"] + d["y_wb"]) / 2
    for s in (1, -1):
        fr = fusion.pod_frame(d, s)
        k = 0
        for cx in fusion.white_columns(d):
            for j in (-1, 0, 1):                    # 3 LEDs per vertical module, white_l / 3 apart
                y = ymid + j * d["white_l"] / 3
                out.append({"side": s, "kind": "white", "index": k, "pos": _frame_point(fr, cx, y),
                            "dir": np.asarray(fr[3], float)})
                k += 1
        uf = fusion.uv_frame(d, s)
        for k, (x, y) in enumerate(fusion.uv_led_points(d)):
            out.append({"side": s, "kind": "uv", "index": k, "pos": _frame_point(uf, x, y),
                        "dir": np.asarray(uf[3], float)})
    return out, d


def white_lumens(cfg: dict, root: Path) -> float | None:
    """leds.white_lumens_per_led from the capture config (cfg capture_config, default configs/capture.yaml)."""
    path = Path(cfg.get("capture_config") or "configs/capture.yaml")
    path = path if path.is_absolute() else Path(root) / path
    try:
        cap = yaml.safe_load(path.read_text()) or {}
    except OSError:
        return None
    lm = (cap.get("leds") or {}).get("white_lumens_per_led")
    return float(lm) if lm else None


def white_params(cfg: dict, root: Path) -> dict:
    """rig.white light params with power_w from the LED flux in the capture config (when there is one)."""
    p = dict(cfg["rig"]["white"])
    lm = white_lumens(cfg, root)
    if lm is not None:
        p["power_w"] = round(led_power_w(lm), 6)
        p["lumens"] = lm
    return p


def load_rig(cfg: dict, root: Path) -> Rig:
    """Build the rig from configs/twin.yaml (cfg) and the Fusion generator under root."""
    rc = cfg["rig"]
    white = white_params(cfg, root)
    fusion = load_fusion_generator(Path(root) / rc["fusion_generator"])
    pts, d = led_model_points(fusion)
    left = bool(rc.get("model_x_is_image_left", True))
    M, origin = model_to_cam(d, left)
    for q in pts:
        q["p_cam"] = M @ (q["pos"] - origin)
        q["d_cam"] = M @ q["dir"] / np.linalg.norm(q["dir"])
    # the pod whose white LEDs sit on the image left (x_cam < 0) is group L
    side_x = {s: np.mean([q["p_cam"][0] for q in pts if q["side"] == s and q["kind"] == "white"]) for s in (1, -1)}
    side_group = {s: ("L" if side_x[s] < 0 else "R") for s in (1, -1)}
    leds = []
    for q in pts:
        pod = side_group[q["side"]]
        if q["kind"] == "white":
            leds.append(Led(f"WL_{pod}_{q['index']}", pod, "white", q["p_cam"], q["d_cam"], dict(white)))
        else:
            leds.append(Led(f"UV_{pod}_{q['index']}", "UV", "uv", q["p_cam"], q["d_cam"], dict(rc["uv"])))
    return Rig(leds, dict(rc.get("device_box_mm", {})), left)


def pod_of(led: Led) -> str:
    """Physical pod (L or R, by image side) of an LED; UV LEDs carry it in their name."""
    return led.name.split("_")[1]


def lights_world(rig: Rig, R, t, power_scale: dict[str, float] | None = None) -> list[dict]:
    """Plan 'lights' entries (DESIGN 3): pos_m = C + R.T @ p_cam / 1000, dir = R.T @ d_cam.

    power_scale multiplies power_w per key "<kind>_<pod>" (e.g. "white_L", "uv_R"); missing
    keys mean 1.
    """
    R = np.asarray(R, float)
    C = -R.T @ np.asarray(t, float)
    scale = power_scale or {}
    out = []
    for led in rig.leds:
        pr = led.params
        pos = C + R.T @ led.p_cam_mm / 1000.0
        dw = R.T @ led.d_cam
        out.append({
            "name": led.name, "group": led.group, "kind": pr.get("kind", "spot"),
            "pos_m": [round(float(v), 7) for v in pos],
            "dir": [round(float(v), 7) for v in dw / np.linalg.norm(dw)],
            "radius_m": round(pr["radius_mm"] / 1000.0, 6),
            "spot_deg": pr["spot_deg"], "blend": pr["blend"],
            "power_w": round(pr["power_w"] * scale.get(f"{led.kind}_{pod_of(led)}", 1.0), 5),
            "color": [float(c) for c in pr["color"]],
        })
    return out
