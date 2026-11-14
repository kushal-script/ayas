"""Drive Blender/Cycles for the twin: camera files, batch renders, EXR passes and previews.

prepare() turns a plan's OpenCV camera into blender_cam.json with trinetra.twin.frames
(the Blender side never imports trinetra). render_batch() runs one Blender process over
several scene work dirs with twin/blender/render_scene.py, then checks each render.exr
and records its channel names in render_meta.json. See docs/twin/DESIGN.md sections 1,
3.3, 4 and 7.1.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from . import frames as F

BLENDER_MAC = Path("/Applications/Blender.app/Contents/MacOS/Blender")
SCRIPT = Path(__file__).resolve().parents[3] / "twin" / "blender" / "render_scene.py"
REQUIRED_GROUPS = ("L", "R", "AMB")        # UV may be missing when a scene has no UV lights
PASS_RE = re.compile(r"(?:^|\.)Combined(?:_(L|R|UV|AMB))?(?:\.([RGBA]))?$")
TIMEOUT_EXTRA_S = 600.0                    # Blender start-up + first Metal kernel compile (about 200 s)


def find_blender() -> Path | None:
    """Blender binary: $TRINETRA_BLENDER, then the macOS app, then PATH."""
    env = os.environ.get("TRINETRA_BLENDER")
    if env and Path(env).is_file():
        return Path(env)
    if BLENDER_MAC.is_file():
        return BLENDER_MAC
    found = shutil.which("blender") or shutil.which("Blender")
    return Path(found) if found else None


def scaled_size(w: int, h: int, res_scale: float) -> tuple[int, int]:
    """Rendered size for a res_scale (same rounding as render_scene.py)."""
    return max(1, int(math.floor(w * res_scale + 0.5))), max(1, int(math.floor(h * res_scale + 0.5)))


def scale_K(K, w: int, h: int, res_scale: float) -> np.ndarray:
    """K of a (w, h) render scaled by res_scale with the same field of view (Blender sensor_fit AUTO).

    Pixel edges scale about the image centre by s = max(ws, hs) / max(w, h); pixel centres
    stay on integers, so cx' = ws/2 - 0.5 + (cx + 0.5 - w/2) * s.
    """
    K = np.asarray(K, float)
    ws, hs = scaled_size(w, h, res_scale)
    s = max(ws, hs) / max(w, h)
    Ks = K.copy()
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = ws / 2 - 0.5 + (K[0, 2] + 0.5 - w / 2) * s
    Ks[1, 2] = hs / 2 - 0.5 + (K[1, 2] + 0.5 - h / 2) * s
    return Ks


def _get(cfg: dict | None, *keys, default=None):
    node: Any = cfg or {}
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _device_box(plan: dict, cfg: dict | None, R: np.ndarray, t: np.ndarray) -> tuple[dict | None, list[str]]:
    """Device box in the OpenCV camera frame (metres), pulled back behind any LED it would swallow."""
    want = (plan.get("extras") or {}).get("device_box", False)
    box = want if isinstance(want, dict) else (_get(cfg, "rig", "device_box_mm") if want else None)
    if not box:
        return None, []
    (x0, x1), (y0, y1), (z0, z1) = (sorted(float(v) for v in box[k]) for k in ("x", "y", "z"))
    notes = []
    for L in plan.get("lights") or []:
        p = (R @ np.asarray(L["pos_m"], float) + t) * 1000.0
        if x0 < p[0] < x1 and y0 < p[1] < y1 and z0 < p[2] <= z1 + 1.0:
            z1 = min(z1, p[2] - 1.0)
            notes.append(f"device box front pulled to z = {z1:.1f} mm behind light {L.get('name')}")
    if z1 - z0 < 5.0:
        return None, notes + ["device box dropped: LEDs sit inside it"]
    return {"x": [x0 / 1000, x1 / 1000], "y": [y0 / 1000, y1 / 1000], "z": [z0 / 1000, z1 / 1000]}, notes


def prepare(work_dir: str | Path, plan: dict | str | Path, cfg: dict | None = None) -> dict:
    """Write <work_dir>/blender_cam.json (and plan.json if missing or different); returns the camera dict.

    The ROI window of the full sensor is rendered with the physical sensor width
    (max(w, h) * pixel_um), so lens_mm is the real focal length and DOF is physical.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if not isinstance(plan, dict):
        plan = json.loads(Path(plan).read_text())
    plan_path = work / "plan.json"
    try:
        same = json.loads(plan_path.read_text()) == plan
    except (OSError, ValueError):
        same = False
    if not same:                    # Blender reads plan.json: it must be the plan this camera came from
        plan_path.write_text(json.dumps(plan, indent=1))
    cam = plan["camera"]
    K = np.asarray(cam["K"], float)
    R = np.asarray(cam["R"], float)
    t = np.asarray(cam["t"], float)
    x0, y0, w, h = (int(round(v)) for v in cam["roi"])
    pixel_um = float(cam.get("pixel_um") or _get(cfg, "camera", "pixel_um", default=F.IMX500["pixel_um"]))
    K_roi = F.roi_K(K, x0, y0)
    out = {"schema": "trinetra.twin.blender_cam/1", "scene_id": plan.get("scene_id"),
           **F.blender_intrinsics(K_roi, w, h, sensor_width_mm=max(w, h) * pixel_um / 1000.0),
           "matrix_world": F.blender_cam2world(R, t).tolist(),
           "focus_m": float(cam.get("focus_m", _get(cfg, "camera", "focus_mm", default=200.0) / 1000.0)),
           "fstop": float(cam.get("fstop", _get(cfg, "camera", "fstop", default=1.79))),
           "roi": [x0, y0, w, h], "K_roi": K_roi.tolist()}
    cv2world = np.eye(4)
    cv2world[:3, :3] = R.T
    cv2world[:3, 3] = F.camera_centre(R, t)
    out["cv2world"] = cv2world.tolist()
    box, notes = _device_box(plan, cfg, R, t)
    if box:
        out["device_box_m"] = box
    out["notes"] = notes
    (work / "blender_cam.json").write_text(json.dumps(out, indent=1))
    return out


def exr_channels(path: str | Path) -> list[str]:
    import OpenEXR

    f = OpenEXR.File(str(path), header_only=True)
    return [ch.name for part in f.parts for ch in part.header["channels"]]


def pass_names(channels: list[str]) -> set[str]:
    """{'Combined', 'L', 'R', ...} present among EXR channel or part names."""
    out = set()
    for c in channels:
        m = PASS_RE.search(c)
        if m:
            out.add(m.group(1) or "Combined")
    return out


def read_passes(path: str | Path) -> dict[str, np.ndarray]:
    """Linear float32 (H, W, 3) arrays keyed 'Combined', 'L', 'R', 'UV', 'AMB' (those present)."""
    import OpenEXR

    split: dict[str, dict[str, np.ndarray]] = {}
    out: dict[str, np.ndarray] = {}
    for part in OpenEXR.File(str(path)).parts:
        for name, ch in part.channels.items():
            m = PASS_RE.search(name)
            if not m:
                continue
            key = m.group(1) or "Combined"
            px = np.asarray(ch.pixels)
            if px.ndim == 3:
                out[key] = px[..., :3].astype(np.float32)
            elif m.group(2):
                split.setdefault(key, {})[m.group(2)] = px
    for key, chans in split.items():
        if key not in out and all(c in chans for c in "RGB"):
            out[key] = np.stack([chans[c] for c in "RGB"], axis=-1).astype(np.float32)
    return out


def merge_ambient(exr: str | Path, amb_exr: str | Path) -> None:
    """Write the AMB pass of the ambient-only render amb_exr into exr (adding its light to Combined).

    render_scene.py renders the LEDs and the ambient apart when a scene has ambient light
    (see its docstring): exr then holds L, R, UV and an empty AMB, amb_exr the ambient.
    """
    import OpenEXR

    amb = read_passes(amb_exr).get("AMB")
    if amb is None:
        raise ValueError(f"{Path(amb_exr).name} has no AMB pass")
    f = OpenEXR.File(str(exr))
    done = False
    for part in f.parts:
        for name, ch in part.channels.items():
            m = PASS_RE.search(name)
            if not m or (m.group(1) or "Combined") not in ("AMB", "Combined"):
                continue
            px = np.asarray(ch.pixels)
            add = amb if px.ndim == 3 else (amb[..., "RGB".index(m.group(2))] if m.group(2) in ("R", "G", "B") else None)
            if add is None:
                continue
            if add.shape[:2] != px.shape[:2]:
                raise ValueError(f"{Path(amb_exr).name}: size {add.shape[:2]} differs from {px.shape[:2]}")
            out = px.astype(np.float32)
            if px.ndim == 3:                     # RGB(A) array: AMB is replaced, Combined gains the ambient
                out[..., :3] = add if m.group(1) == "AMB" else out[..., :3] + add
            else:                                # one split channel
                out = add.astype(np.float32) if m.group(1) == "AMB" else out + add
            done = done or m.group(1) == "AMB"
            ch.pixels = out.astype(px.dtype)
    if not done:
        raise ValueError(f"{Path(exr).name} has no AMB pass to fill")
    tmp = Path(str(exr) + ".tmp.exr")
    f.write(str(tmp))
    os.replace(tmp, exr)
    Path(amb_exr).unlink()


def _finish(work: Path, batch_log: str, problem: str | None) -> dict:
    """Read Blender's render_meta.json, add EXR channel names and pass checks, write it back.

    A batch problem (timeout, crash, non-zero exit) is this scene's error only if Blender
    never finished the scene (no render_meta.json): scenes completed before it stay usable.
    Split renders get their <name>_amb.exr merged (ambient_merged_into lists the EXRs done);
    a split render whose ambient half is missing is an error, never a silent dark room.
    """
    meta_path = work / "render_meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        meta = {"scene_id": work.name, "errors": [], "warnings": [],
                "error": problem or "Blender wrote no render_meta.json"}
    else:
        if problem:
            meta.setdefault("warnings", []).append(f"batch: {problem}")
    try:
        with open(work / "blender.log", "a") as f:
            f.write("\n--- batch log (trinetra.twin.render) ---\n" + batch_log)
    except OSError as e:            # e.g. the work dir is gone: still report the scene, never raise
        meta["error"] = meta.get("error") or f"work dir unusable: {e}"
        return meta
    for name in ("render", "render_clean"):          # split renders: put the ambient pass back
        amb = work / f"{name}_amb.exr"
        if amb.is_file() and (work / f"{name}.exr").is_file():
            try:
                merge_ambient(work / f"{name}.exr", amb)
                meta["files"] = [x for x in meta.get("files", []) if x != amb.name]
                meta["ambient_merged"] = True
                meta["ambient_merged_into"] = sorted(set(meta.get("ambient_merged_into", [])) | {f"{name}.exr"})
            except Exception as e:  # noqa: BLE001  (a bad merge is this scene's error)
                meta["error"] = meta.get("error") or f"cannot merge {amb.name}: {e}"
        elif (meta.get("split_ambient") and (work / f"{name}.exr").is_file() and not meta.get("error")
              and f"{name}.exr" not in meta.get("ambient_merged_into", [])):
            # an LEDs-only EXR without its ambient half: AMB would silently read as a dark room
            meta["error"] = f"split render: {amb.name} missing, {name}.exr has no ambient light"
    exr = work / "render.exr"
    if exr.is_file():
        try:
            meta["exr_channels"] = exr_channels(exr)
            meta["exr_mb"] = round(exr.stat().st_size / 1e6, 2)
            missing = [g for g in REQUIRED_GROUPS if g not in pass_names(meta["exr_channels"])]
            if missing and not meta.get("error"):
                meta["error"] = f"render.exr lacks light-group passes {missing}"
        except Exception as e:      # noqa: BLE001  (a broken EXR is this scene's error, not the batch's)
            meta["error"] = meta.get("error") or f"cannot read render.exr: {e}"
    elif not meta.get("error"):
        meta["error"] = "no render.exr written"
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def render_batch(work_dirs: list, cfg: dict | None = None, device: str | None = None, samples: int | None = None,
                 res_scale: float | None = None, timeout_s: float | None = None, overview: bool = False,
                 overview_samples: int | None = None) -> list[dict]:
    """Render prepared work dirs in one Blender process; returns their render_meta dicts.

    A meta with a truthy "error" means that scene failed; the others are still usable.
    overview=True also writes <work>/overview.png (third-person view of the twin for demos,
    1280 x 720, denoised); its failure is only a warning in render_meta.json.
    timeout_s None means cfg render.timeout_s_per_scene (default 900 s) per scene plus
    TIMEOUT_EXTRA_S for Blender start-up and the one-off Metal kernel compile, so a hung
    GPU cannot stall a long run; pass 0 for no timeout.
    """
    works = [Path(d).resolve() for d in work_dirs]
    if not works:
        return []
    blender = find_blender()
    if blender is None:
        raise FileNotFoundError("Blender not found: set TRINETRA_BLENDER or install Blender 5.2")
    if timeout_s is None:
        timeout_s = float(_get(cfg, "render", "timeout_s_per_scene", default=900.0)) * len(works) + TIMEOUT_EXTRA_S
    timeout_s = float(timeout_s) or None
    device = str(device or _get(cfg, "render", "device", default="METAL")).upper()
    cmd = [str(blender), "-b", "--factory-startup", "-P", str(SCRIPT), "--", "--scenes", *map(str, works),
           "--device", device]
    if samples:
        cmd += ["--samples", str(int(samples))]
    if res_scale and float(res_scale) != 1.0:
        cmd += ["--res-scale", repr(float(res_scale))]
    if overview:
        cmd += ["--overview"]
        if overview_samples:
            cmd += ["--overview-samples", str(int(overview_samples))]
    for w in works:
        for name in ("render_meta.json", "render.exr", "render_clean.exr", "render_amb.exr", "render_clean_amb.exr",
                     "blender.log", "overview.png"):
            (w / name).unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    problem = None
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              timeout=timeout_s, env=env)
        log = proc.stdout or ""
        if proc.returncode != 0:
            problem = f"Blender exited with code {proc.returncode}"
    except subprocess.TimeoutExpired as e:
        raw = e.stdout or b""
        log = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        problem = f"Blender timed out after {timeout_s} s"
    return [_finish(w, log, problem) for w in works]


def tone_map(rgb: np.ndarray, white_pct: float = 99.5) -> np.ndarray:
    """Linear RGB -> 8-bit sRGB for humans: the white_pct luma percentile maps to 1."""
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    white = float(np.percentile(luma, white_pct)) or 1.0
    x = np.clip(rgb / white, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (srgb * 255 + 0.5).astype(np.uint8)


def preview(work_dir: str | Path, out: str | Path | None = None, exr_name: str = "render.exr") -> Path:
    """Write a tone-mapped PNG of the Combined pass (default <work_dir>/preview.png)."""
    import cv2

    work = Path(work_dir)
    passes = read_passes(work / exr_name)
    rgb = passes.get("Combined")
    if rgb is None:
        rgb = sum(passes[k] for k in ("L", "R", "UV", "AMB") if k in passes)
    out = Path(out) if out else work / "preview.png"
    cv2.imwrite(str(out), tone_map(np.nan_to_num(rgb))[..., ::-1])
    return out
