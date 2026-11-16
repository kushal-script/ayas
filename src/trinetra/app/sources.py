"""Inputs of the on-device app: twin bursts, the Pi camera burst, or a folder of single images.

Every source has next() -> (item, meta) or None when it is exhausted, plus close(). item is a
lightproc.Burst (arm 'burst': B1, L, A, R, B2 linear full-scale frames) or a uint8 RGB image
(arm 'single', the old one-frame method). meta is a JSON-safe dict the pipeline reads:
source, arm, mode (burst5 | naive), part_id, part {material, thickness_mm, size_mm},
device_mode, corners_px (4 image points TL, TR, BR, BL in frame px, or None = find them),
plate_size_mm, grid_plate_mm (image-aligned W x H, when known), capture_ms, and for the twin
condition, lux and the ground-truth labels. Optional through holes for the edge-or-hole rule:
holes_mm [[x, y, d]] in grid-aligned plate mm (corners.json, or app.yaml part.holes_mm for a
jig), or holes_plate_mm [[u, v, d]] in plate mm + H_frame_plate (a twin burst that has them).

- TwinSource reads out/bursts/*.npz written by twin/post.py (lightproc.save_burst layout): a
  virtual camera that replays the twin's bursts, optionally paced at `fps`.
- CameraSource is the Pi camera (IMX500 via picamera2, pods via gpiozero; both imported lazily):
  fixed exposure and gain, AE/AWB/NR off, an unpacked raw SRGGB10 stream; for each frame of the
  sequence it switches the pods, then capture_request(flush=True) so the frame was exposed after
  the switch; raw -> lightproc.raw_to_rgb (black level, demosaic, FS units). See docs/APP.md for
  the assumptions to verify on the Pi.
- FileSource reads PNG/JPEG photos (+ optional corners.json) for the single-image arm.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .. import lightproc as LP

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SENSOR_SIZES = {"full": (4056, 3040), "binned": (2028, 1520)}
DEFAULT_PART = {"material": "ms_cold_rolled", "thickness_mm": 3.0, "size_mm": [100.0, 100.0]}


def _part(part: Mapping[str, Any] | None) -> dict[str, Any]:
    p = {**DEFAULT_PART, **{k: v for k, v in (part or {}).items() if v is not None}}
    return {"material": str(p["material"]), "thickness_mm": float(p["thickness_mm"]),
            "size_mm": [float(x) for x in p["size_mm"]]}


def _scalar(extra: Mapping[str, np.ndarray], key: str):
    """A 0-d npz entry as a Python str/float (None when absent, empty or NaN)."""
    if key not in extra:
        return None
    v = np.asarray(extra[key])
    if v.ndim:
        return None
    v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str):
        return v or None
    v = float(v)
    return None if math.isnan(v) else v


def _list(extra: Mapping[str, np.ndarray], key: str):
    return np.asarray(extra[key], float).tolist() if key in extra else None


def _json(extra: Mapping[str, np.ndarray], key: str):
    if key not in extra:
        return None
    return json.loads(np.asarray(extra[key], np.uint8).tobytes().decode())


class _Pacer:
    """Holds next() to at most fps items per second (fps None or <= 0: no wait)."""

    def __init__(self, fps: float | None):
        self.period = 1.0 / float(fps) if fps else 0.0
        self._t = None

    def wait(self) -> None:
        if self.period > 0 and self._t is not None:
            dt = self._t + self.period - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
        self._t = time.perf_counter()


# ---- twin -------------------------------------------------------------------------------

def read_twin_npz(path: str | Path, part: Mapping[str, Any] | None = None) -> tuple[LP.Burst, dict[str, Any]]:
    """One out/bursts/<scene>_v<k>.npz -> (Burst with float32 frames, meta). See twin/post.save_burst."""
    path = Path(path)
    burst, extra = LP.load_burst(path)
    twin = dict(burst.meta.get("twin") or {})
    tpart = dict(twin.get("part") or {})
    size = _list(extra, "plate_size_mm") or tpart.get("size_mm") or (part or {}).get("size_mm")
    p = _part({**(part or {}), **tpart, "material": _scalar(extra, "material") or tpart.get("material"),
               "thickness_mm": _scalar(extra, "thickness_mm") or tpart.get("thickness_mm"), "size_mm": size})
    lux = _scalar(extra, "lux")
    meta = {
        "source": "twin", "arm": "burst", "mode": "burst5", "path": str(path), "part_id": path.stem,
        "scene_id": twin.get("scene_id"), "variant": twin.get("k"), "plate": twin.get("plate"),
        "split": twin.get("split"),
        "condition": _scalar(extra, "condition") or twin.get("condition"),
        "lux": lux if lux is not None else twin.get("lux"),
        "device_mode": _scalar(extra, "device_mode") or twin.get("device_mode") or "open",
        "corners_px": _list(extra, "corners_px"),
        "corner_plate_idx": [int(i) for i in extra["corner_plate_idx"]] if "corner_plate_idx" in extra else None,
        "plate_size_mm": p["size_mm"],
        "grid_plate_mm": _list(extra, "grid_plate_mm"),
        "part": p,
        "labels": _json(extra, "labels_json") or [],
        "twin_gate": twin.get("gate"),
    }
    if "holes_plate_mm" in extra and "H_frame_plate" in extra:   # optional: through holes [u, v, d] (plate mm)
        meta["holes_plate_mm"] = np.asarray(extra["holes_plate_mm"], float).reshape(-1, 3).tolist()
        meta["H_frame_plate"] = np.asarray(extra["H_frame_plate"], float).tolist()
    burst.meta.setdefault("material", p["material"])
    burst.meta.setdefault("demosaiced", False)
    missing = [k for k in LP.SEQUENCE if k not in burst.frames]
    if missing:
        raise ValueError(f"{path.name}: burst is missing frames {missing}")
    return burst, meta


class TwinSource:
    """Replays the twin's burst files (out/bursts/*.npz) as a virtual camera."""

    kind = "twin"

    def __init__(self, npz_dir: str | Path, loop: bool = True, fps: float | None = None,
                 part: Mapping[str, Any] | None = None):
        self.dir = Path(npz_dir)
        self.files = sorted(p for p in self.dir.glob("*.npz") if not p.name.endswith(".part.npz"))
        if not self.files:
            raise FileNotFoundError(f"no burst .npz files in {self.dir}")
        self.loop, self.part = bool(loop), part
        self._pace = _Pacer(fps)
        self._i = 0

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self):
        while (item := self.next()) is not None:
            yield item

    def next(self) -> tuple[LP.Burst, dict[str, Any]] | None:
        if self._i >= len(self.files):
            if not self.loop:
                return None
            self._i = 0
        path = self.files[self._i]
        self._i += 1
        self._pace.wait()
        t0 = time.perf_counter()
        burst, meta = read_twin_npz(path, self.part)
        meta["capture_ms"] = round((time.perf_counter() - t0) * 1e3, 1)   # file read: the twin's "exposure"
        sib = self._sibling(path)
        if sib is not None:                       # P1h: the scene's other variant is the second burst
            meta["partner"] = read_twin_npz(sib, self.part)
        return burst, meta

    def _sibling(self, path: Path) -> Path | None:
        """<scene>_v0.npz -> <scene>_v1.npz when it is in the folder (and the reverse). With only one
        variant per scene there is no partner and the inspection is judged on its single burst."""
        stem = path.stem
        if "_v" not in stem:
            return None
        base, k = stem.rsplit("_v", 1)
        if not k.isdigit():
            return None
        other = path.with_name(f"{base}_v{1 - int(k) if int(k) in (0, 1) else k}.npz")
        return other if other != path and other.exists() else None

    def describe(self) -> str:
        return f"twin:{self.dir} ({len(self.files)} bursts{', loop' if self.loop else ''})"

    def close(self) -> None:
        pass


# ---- single images ------------------------------------------------------------------------

class FileSource:
    """A folder of single RGB photos (arm 'single'). corners.json (optional) maps a file name to
    {"corners_px": [[x, y] x 4], "plate_size_mm", "material", "thickness_mm", "device_mode"}."""

    kind = "files"

    def __init__(self, folder: str | Path, loop: bool = False, part: Mapping[str, Any] | None = None,
                 fps: float | None = None):
        self.dir = Path(folder)
        self.files = sorted(p for p in self.dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.files:
            raise FileNotFoundError(f"no images in {self.dir}")
        cj = self.dir / "corners.json"
        self.entries = json.loads(cj.read_text()) if cj.exists() else {}
        self.loop, self.part = bool(loop), part
        self._pace = _Pacer(fps)
        self._i = 0

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self):
        while (item := self.next()) is not None:
            yield item

    def next(self) -> tuple[np.ndarray, dict[str, Any]] | None:
        if self._i >= len(self.files):
            if not self.loop:
                return None
            self._i = 0
        path = self.files[self._i]
        self._i += 1
        self._pace.wait()
        t0 = time.perf_counter()
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cannot read {path}")
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else
                           (cv2.COLOR_BGRA2RGB if img.shape[2] == 4 else cv2.COLOR_BGR2RGB))
        e = dict(self.entries.get(path.name) or {})
        p = _part({**(self.part or {}), **{k: e.get(k) for k in ("material", "thickness_mm")},
                   "size_mm": e.get("plate_size_mm") or (self.part or {}).get("size_mm")})
        meta = {"source": "files", "arm": "single", "mode": "naive", "path": str(path), "part_id": path.stem,
                "device_mode": e.get("device_mode") or (self.part or {}).get("device_mode") or "open",
                "corners_px": e.get("corners_px"), "plate_size_mm": p["size_mm"], "grid_plate_mm": None,
                "holes_mm": e.get("holes_mm"),
                "part": p, "labels": [], "capture_ms": round((time.perf_counter() - t0) * 1e3, 1)}
        return np.ascontiguousarray(img), meta

    def describe(self) -> str:
        return f"files:{self.dir} ({len(self.files)} images)"

    def close(self) -> None:
        pass


# ---- the Pi camera ----------------------------------------------------------------------------

CAMERA_DEFAULTS = {"index": 0, "raw_format": "SRGGB10", "left_shifted": "auto", "black_level": "config",
                   "roi": None, "corners_px": None, "settle_ms": 0, "warmup_frames": 2,
                   "colour_gains": [1.0, 1.0]}


def packed_or_compressed(raw_format: str) -> bool:
    """True for raw formats raw_u16 cannot read: CSI-2 packed (_CSI2P) or the Pi 5 PiSP
    compressed ones (e.g. BGGR16_PISP_COMP1, the Pi 5 default for raw streams)."""
    f = str(raw_format).upper()
    return f.endswith("_CSI2P") or "PISP_COMP" in f or "_COMP" in f


def bayer_pattern(raw_format: str) -> str:
    """'SRGGB10' / 'SBGGR10_CSI2P' / 'RGGB' -> 'RGGB' / 'BGGR' ..."""
    f = str(raw_format).upper()
    f = f[1:] if f.startswith("S") and f[1:5] in ("RGGB", "BGGR", "GRBG", "GBRG") else f
    pat = f[:4]
    if pat not in ("RGGB", "BGGR", "GRBG", "GBRG"):
        raise ValueError(f"cannot read the Bayer order from raw format {raw_format!r}")
    return pat


def raw_u16(arr: np.ndarray, width: int | None = None) -> np.ndarray:
    """picamera2 make_array('raw') of an unpacked 16-bit stream -> (H, W) uint16.

    The raw array is uint8 (H, stride in bytes) on most picamera2 versions: view it as
    little-endian uint16 and drop the stride padding."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        a = np.ascontiguousarray(a).view("<u2")
    elif a.dtype != np.uint16:
        a = a.astype(np.uint16)
    return a[:, :width] if width else a


class CameraSource:
    """The device's burst on the Pi: IMX500 raw + two LED pods on GPIO (DESIGN_STEEL H 1).

    Assumptions to VERIFY on the Pi (docs/APP.md lists the checks):
    - the requested raw format is an UNPACKED 16-bit Bayer stream (no _CSI2P suffix); its Bayer
      order is read back from the camera configuration;
    - Pi 5 unpacked raw is MSB-aligned (left-shifted to 16 bits): left_shifted 'auto' decides
      from the first frame (black level 64 << 6 = 4096 > 1023);
    - capture_request(flush=True) returns a frame whose exposure STARTED after the call, so a
      pod switch just before it never leaks into the frame (rolling shutter included);
    - with AeEnable/AwbEnable off, ExposureTime and AnalogueGain are applied exactly (checked
      against each frame's metadata: a mismatch goes to meta['camera']['warnings']);
    - libcamera clamps FrameDurationLimits to the sensor mode (about 100 ms at 4056 x 3040);
    - pods: one logic-level MOSFET per pod, active-high, BCM pins from configs/app.yaml.
    """

    kind = "camera"

    def __init__(self, capture_cfg: Mapping[str, Any], cam_cfg: Mapping[str, Any] | None = None,
                 pods_cfg: Mapping[str, Any] | None = None, part: Mapping[str, Any] | None = None):
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise ImportError("picamera2 not found: on the Pi `sudo apt install python3-picamera2` and create the "
                              "venv with --system-site-packages (docs/APP.md)") from e
        try:
            from gpiozero import DigitalOutputDevice
        except ImportError as e:
            raise ImportError("gpiozero not found: `sudo apt install python3-gpiozero` (venv with "
                              "--system-site-packages)") from e
        self.cfg = {**CAMERA_DEFAULTS, **(cam_cfg or {})}
        self.capture_cfg = capture_cfg
        b = capture_cfg.get("burst") or {}
        self.sequence = tuple(b.get("sequence") or LP.SEQUENCE)
        self.led_state = {k: tuple(int(x) for x in v) for k, v in (b.get("led_state") or {}).items()}
        missing = [k for k in self.sequence if k not in self.led_state]
        if missing:
            raise ValueError(f"capture.yaml burst.led_state has no entry for {missing}")
        self.part = _part(part)
        self.device_mode = str((part or {}).get("device_mode") or "open")
        # pods (handheld) | ring (device v2 hood head: the GPIO pins drive the ring's image-left
        # and image-right halves); lightproc picks its L/R combination from it
        self.illumination = str((part or {}).get("illumination") or ("ring" if self.device_mode == "hood" else "pods"))
        # sensor values may be per illumination (capture.yaml analogue_gain {pods, ring}): one exposure,
        # and the ring gives about 1/1.5 of the pods' irradiance at the plate (DESIGN_STEEL C2)
        self.sensor = LP.sensor_for(capture_cfg.get("sensor"), self.illumination)
        self.bit_depth = int(self.sensor.get("bit_depth", 10))
        pods = {"L": 17, "R": 27, "active_high": True, **(pods_cfg or {})}
        self.pods = {k: DigitalOutputDevice(int(pods[k]), active_high=bool(pods["active_high"]), initial_value=False)
                     for k in ("L", "R")}
        self.size = SENSOR_SIZES.get(str(self.sensor.get("mode", "full")), SENSOR_SIZES["full"])
        self.cam = None
        try:
            self.cam = Picamera2(int(self.cfg["index"]))
            self._configure(Picamera2)
        except BaseException:   # free the GPIO pins and the camera, so a retry in this process can claim them
            for p in self.pods.values():
                p.close()
            if self.cam is not None:
                self.cam.close()
            raise
        self._left: bool | None = None if self.cfg["left_shifted"] in ("auto", None) else bool(self.cfg["left_shifted"])
        self._n = 0
        self.warnings: list[str] = []

    # -- setup --
    def _configure(self, Picamera2) -> None:
        fmt = str(self.cfg["raw_format"])
        if packed_or_compressed(fmt):
            raise ValueError(f"raw_format {fmt!r} must be an unpacked stream (e.g. SRGGB10 or SRGGB16), "
                             "not a packed _CSI2P or compressed PISP_COMP one")
        kw = dict(main={"size": (640, 480), "format": "RGB888"}, raw={"format": fmt, "size": self.size},
                  buffer_count=2)
        try:
            conf = self.cam.create_still_configuration(**kw, sensor={"output_size": self.size, "bit_depth": self.bit_depth})
        except TypeError:   # older picamera2 without the sensor= argument
            conf = self.cam.create_still_configuration(**kw)
        self.cam.configure(conf)
        got = (self.cam.camera_configuration() or {}).get("raw") or {}
        self.raw_format = str(got.get("format") or fmt)
        if packed_or_compressed(self.raw_format):   # read as 16-bit Bayer it would demosaic to garbage
            raise RuntimeError(f"camera delivers packed or compressed raw {self.raw_format}; request an unpacked "
                               "format (camera.raw_format SRGGB10, or SRGGB16 on a Pi 5)")
        self.raw_size = tuple(got.get("size") or self.size)
        self.pattern = bayer_pattern(self.raw_format)
        self.controls = self._controls()
        self.cam.set_controls(self.controls)
        self.cam.start()
        for _ in range(int(self.cfg.get("warmup_frames") or 0)):
            self.cam.capture_request(flush=True).release()

    def _controls(self) -> dict[str, Any]:
        s = self.sensor
        exp = int(s.get("exposure_us", 10000))
        fd = max(int(s.get("frame_duration_us", exp)), exp)
        try:
            from libcamera import controls as lc
            nr_off = lc.draft.NoiseReductionModeEnum.Off
        except (ImportError, AttributeError):
            nr_off = 0
        return {"AeEnable": False, "AwbEnable": False, "ExposureTime": exp,
                "AnalogueGain": float(s.get("analogue_gain", 1.0)),
                "ColourGains": tuple(float(g) for g in self.cfg.get("colour_gains") or (1.0, 1.0)),
                "FrameDurationLimits": (fd, fd), "NoiseReductionMode": nr_off}

    # -- capture --
    def set_pods(self, left: int, right: int) -> None:
        for name, on in (("L", left), ("R", right)):
            (self.pods[name].on if on else self.pods[name].off)()

    def _black(self, md: Mapping[str, Any]) -> float:
        if self.cfg.get("black_level") == "metadata" and md.get("SensorBlackLevels"):
            return float(md["SensorBlackLevels"][0]) / float(1 << (16 - self.bit_depth))   # 16-bit scale -> DN
        return float(self.sensor.get("black_level", 64))

    def _crop(self, raw: np.ndarray) -> np.ndarray:
        roi = self.cfg.get("roi")
        if not roi:
            return raw
        x0, y0, w, h = (int(v) for v in roi)
        x0, y0 = x0 - x0 % 2, y0 - y0 % 2          # keep the Bayer phase
        return raw[y0:y0 + h, x0:x0 + w]

    def capture_burst(self) -> tuple[LP.Burst, dict[str, Any]]:
        """Switch the pods through the sequence and grab one raw frame per state."""
        raws, mds, t_frames = {}, {}, {}
        t0 = time.perf_counter()
        settle = float(self.cfg.get("settle_ms") or 0) / 1e3
        try:
            for name in self.sequence:
                self.set_pods(*self.led_state[name])
                if settle:
                    time.sleep(settle)
                req = self.cam.capture_request(flush=True)
                try:
                    raw = req.make_array("raw")
                    md = dict(req.get_metadata() or {})
                finally:
                    req.release()
                raws[name] = self._crop(raw_u16(raw, int(self.raw_size[0])))
                mds[name] = md
                t_frames[name] = round((time.perf_counter() - t0) * 1e3, 1)
        finally:
            self.set_pods(0, 0)
        t_cap = time.perf_counter()
        if self._left is None:
            self._left = int(raws[self.sequence[0]].max(initial=0)) > (1 << self.bit_depth) - 1
        frames = {k: LP.raw_to_rgb(v, self._black(mds[k]), self.bit_depth, self.pattern, self._left)
                  for k, v in raws.items()}
        warnings = self._check(mds)
        burst = LP.Burst(frames, {"demosaiced": True, "material": self.part["material"],
                                  "illumination": self.illumination, "sequence": list(self.sequence),
                                  "exposure_us": [mds[k].get("ExposureTime") for k in self.sequence],
                                  "analogue_gain": [mds[k].get("AnalogueGain") for k in self.sequence]})
        cam = {"raw_format": self.raw_format, "pattern": self.pattern, "left_shifted": self._left,
               "frame_ms": t_frames, "grab_ms": round((t_cap - t0) * 1e3, 1),
               "demosaic_ms": round((time.perf_counter() - t_cap) * 1e3, 1),
               "sensor_ts": [mds[k].get("SensorTimestamp") for k in self.sequence], "warnings": warnings}
        return burst, cam

    def _check(self, mds: Mapping[str, Mapping[str, Any]]) -> list[str]:
        """Frame metadata against the requested exposure, gain and frame order."""
        out = []
        exp, gain = float(self.controls["ExposureTime"]), float(self.controls["AnalogueGain"])
        for k, md in mds.items():
            e, g = md.get("ExposureTime"), md.get("AnalogueGain")
            if e is not None and abs(float(e) - exp) > 0.02 * exp:
                out.append(f"{k}: ExposureTime {e} us != {exp:.0f}")
            if g is not None and abs(float(g) - gain) > 0.02 * gain:
                out.append(f"{k}: AnalogueGain {g} != {gain}")
        ts = [md.get("SensorTimestamp") for md in mds.values()]
        if all(t is not None for t in ts) and any(b <= a for a, b in zip(ts, ts[1:])):
            out.append("SensorTimestamp not increasing: a frame was reused")
        return out

    def next(self) -> tuple[LP.Burst, dict[str, Any]]:
        t0 = time.perf_counter()
        burst, cam = self.capture_burst()
        self._n += 1
        h, w = next(iter(burst.frames.values())).shape[:2]
        meta = {"source": "camera", "arm": "burst", "mode": "burst5",
                "part_id": f"cam-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self._n:04d}",
                "device_mode": self.device_mode, "corners_px": self.cfg.get("corners_px"),
                "plate_size_mm": self.part["size_mm"], "grid_plate_mm": None, "part": dict(self.part),
                "labels": [], "frame_size": [w, h], "camera": cam,
                "capture_ms": round((time.perf_counter() - t0) * 1e3, 1)}
        if int(self.cfg.get("confirm_bursts", 2)) >= 2:      # P1h: one inspection = two bursts
            burst2, cam2 = self.capture_burst()
            meta["partner"] = (burst2, {k: v for k, v in meta.items() if k != "camera"} | {"camera": cam2})
            meta["capture_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        return burst, meta

    def describe(self) -> str:
        return f"camera:{self.raw_format} {self.raw_size[0]}x{self.raw_size[1]} {self.pattern}"

    def close(self) -> None:
        try:
            self.set_pods(0, 0)
            for p in self.pods.values():
                p.close()
        finally:
            try:
                self.cam.stop()
            finally:
                self.cam.close()


def open_source(spec: str, capture_cfg: Mapping[str, Any] | None = None, app_cfg: Mapping[str, Any] | None = None,
                loop: bool | None = None, fps: float | None = None):
    """'twin:<dir>' | 'camera' | 'files:<dir>' -> a source."""
    app_cfg = app_cfg or {}
    part = app_cfg.get("part")
    twin = app_cfg.get("twin") or {}
    kind, _, arg = str(spec).partition(":")
    if kind == "twin":
        return TwinSource(arg, loop=twin.get("loop", True) if loop is None else loop,
                          fps=twin.get("fps") if fps is None else fps, part=part)
    if kind == "files":
        return FileSource(arg, loop=bool(loop), part=part, fps=fps)
    if kind == "camera":
        if capture_cfg is None:
            raise ValueError("camera source needs the capture config")
        return CameraSource(capture_cfg, app_cfg.get("camera"), app_cfg.get("pods"), part)
    raise ValueError(f"unknown source {spec!r}: twin:<dir> | camera | files:<dir>")


def corners_array(corners: Sequence | None) -> np.ndarray | None:
    return None if corners is None else np.asarray(corners, float).reshape(4, 2)
