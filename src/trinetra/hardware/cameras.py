"""Camera drivers. Every camera returns a BGR uint8 frame with exposure, gain,
white balance and focus locked, so intensities are comparable between parts."""

from __future__ import annotations

import subprocess
from typing import Any, Sequence

import numpy as np


class Camera:
    def capture(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PiCamera(Camera):
    """CSI camera (AI Camera, HQ Camera, Camera Module 3) through Picamera2."""

    def __init__(
        self,
        index: int,
        size: Sequence[int],
        exposure_us: int,
        analogue_gain: float = 1.0,
        colour_gains: Sequence[float] = (1.0, 1.0),
        lens_position: float | None = None,
        flush_frames: int = 2,
    ):
        from picamera2 import Picamera2

        self._cam = Picamera2(index)
        config = self._cam.create_still_configuration(
            main={"size": tuple(size), "format": "RGB888"}, buffer_count=2
        )
        self._cam.configure(config)
        controls: dict[str, Any] = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": int(exposure_us),
            "AnalogueGain": float(analogue_gain),
            "ColourGains": tuple(float(g) for g in colour_gains),
        }
        if lens_position is not None:  # only for cameras with a focus motor (Camera Module 3)
            controls.update({"AfMode": 0, "LensPosition": float(lens_position)})
        self._cam.set_controls(controls)
        self._cam.start()
        self._flush = flush_frames

    def capture(self) -> np.ndarray:
        # Frames exposed while the previous light was still on are thrown away.
        for _ in range(self._flush):
            self._cam.capture_array("main")
        # Picamera2 "RGB888" is stored B, G, R - the same order OpenCV uses.
        return self._cam.capture_array("main")

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()


class UsbCamera(Camera):
    """UVC webcam. Controls go through v4l2-ctl because their names differ between
    models; list yours with `v4l2-ctl -d /dev/video0 --list-ctrls`."""

    def __init__(
        self,
        device: str,
        size: Sequence[int],
        v4l2_controls: dict[str, int] | None = None,
        flush_frames: int = 4,
    ):
        import cv2

        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open USB camera {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Applied after opening: some drivers reset controls when streaming starts.
        if v4l2_controls:
            args = ["v4l2-ctl", "-d", device]
            for name, value in v4l2_controls.items():
                args += ["-c", f"{name}={value}"]
            subprocess.run(args, check=True)
        self._flush = flush_frames

    def capture(self) -> np.ndarray:
        # USB cameras queue frames; drop the ones taken under the previous light.
        for _ in range(self._flush):
            self._cap.grab()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("USB camera returned no frame")
        return frame

    def close(self) -> None:
        self._cap.release()
