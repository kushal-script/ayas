"""Capture sequence: switch on one white light, fire all three cameras together, repeat;
then one frame with every light off. The dark frame is subtracted later, which removes
the room light (it is the same in every state), so the station works in an open space."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np

from .config import BGR_INDEX, DARK, LUMA, VIEWS, key
from .hardware.cameras import Camera
from .hardware.lights import Lights


def capture_set(
    lights: Lights,
    cameras: Mapping[str, Camera],
    states: Iterable[str],
    settle_s: float = 0.05,
) -> dict[str, np.ndarray]:
    """Return BGR uint8 frames keyed "<view>_<state>"; a state is a light name or "dark"."""
    frames: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
        try:
            for state in states:
                if state == DARK:
                    lights.off()
                else:
                    lights.on(state)
                time.sleep(settle_s)
                futures = {view: pool.submit(cam.capture) for view, cam in cameras.items()}
                for view, fut in futures.items():
                    frames[key(view, state)] = fut.result()
        finally:
            lights.off()
    return frames


def lit(frames: Mapping[str, np.ndarray], view: str, light: str) -> np.ndarray:
    """(frame under the light - dark frame) as float BGR in 0..1: the LED's light only."""
    on = frames[key(view, light)].astype(np.float32)
    dark = frames.get(key(view, DARK))
    if dark is not None:
        on = on - dark.astype(np.float32)
    return np.clip(on / 255.0, 0.0, None)


def channel(img_bgr: np.ndarray, ch: str) -> np.ndarray:
    """One colour channel ("r", "g", "b") or luminance ("y") of a float BGR image."""
    if img_bgr.ndim == 2:
        return img_bgr.astype(np.float32)
    if ch == "y":
        return sum(LUMA[c] * img_bgr[..., BGR_INDEX[c]] for c in LUMA).astype(np.float32)
    return img_bgr[..., BGR_INDEX[ch]].astype(np.float32)


def bright_view(frames: Mapping[str, np.ndarray], view: str, lights: Iterable[str]) -> np.ndarray:
    """Sum of all lights for one camera, scaled to 0..1: a bright, evenly lit plate for calibration."""
    total = sum(channel(lit(frames, view, n), "y") for n in lights)
    return np.clip(total / max(float(np.percentile(total, 99.5)), 1e-6), 0.0, 1.0)


def save_frames(frames: Mapping[str, np.ndarray], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for k, frame in frames.items():
        cv2.imwrite(str(directory / f"{k}.png"), frame)


def load_frames(directory: Path) -> dict[str, np.ndarray]:
    frames = {}
    for path in sorted(Path(directory).glob("*.png")):
        view, _, _state = path.stem.rpartition("_")
        if view in VIEWS:
            frames[path.stem] = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return frames
