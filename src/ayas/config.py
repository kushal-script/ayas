"""Rig configuration: cameras, white lights, the common top-down grid and detection settings.

Every capture switches on ONE white light (or none, for the dark frame) and fires all
three cameras together. Frames are keyed "<view>_<light>", e.g. "grazing_L15" or
"top_dark". Each camera has a "pair" light next to it on the arch; extra lights can be
added around the plate for the 3D shape map (photometric stereo).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

VIEWS = ("top", "oblique45", "grazing")
CHANNELS = ("r", "g", "b")
# Channel index in an OpenCV / Picamera2 "RGB888" frame (byte order B, G, R).
BGR_INDEX = {"b": 0, "g": 1, "r": 2}
DARK = "dark"
# Luminance weights (Rec. 601) for turning a white-light colour frame into one brightness map.
LUMA = {"r": 0.299, "g": 0.587, "b": 0.114}

# Shape (photometric stereo) maps added to the stack when enough lights are set up.
SHAPE_FULL_KEYS = ("ps_nx", "ps_ny")
SHAPE_SLOPE_KEYS = ("ps_sa",)


def key(view: str, light: str) -> str:
    return f"{view}_{light}"


def split_key(k: str) -> tuple[str, str]:
    view, light = k.rsplit("_", 1)
    return view, light


@dataclass
class Grid:
    """Common top-down frame every image is warped into. Millimetres are measured
    from the plate's top-left corner; the grid adds a border so plate edges stay
    visible for alignment."""

    plate_w_mm: float
    plate_h_mm: float
    border_mm: float = 5.0
    mm_per_px: float = 0.1
    edge_margin_mm: float = 2.0

    @property
    def width_px(self) -> int:
        return int(round((self.plate_w_mm + 2 * self.border_mm) / self.mm_per_px))

    @property
    def height_px(self) -> int:
        return int(round((self.plate_h_mm + 2 * self.border_mm) / self.mm_per_px))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height_px, self.width_px)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) as OpenCV expects."""
        return (self.width_px, self.height_px)

    def mm_to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return ((x_mm + self.border_mm) / self.mm_per_px, (y_mm + self.border_mm) / self.mm_per_px)

    def px_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        return (x_px * self.mm_per_px - self.border_mm, y_px * self.mm_per_px - self.border_mm)

    def plate_corners_px(self) -> np.ndarray:
        """Plate corners TL, TR, BR, BL in grid pixels."""
        x0, y0 = self.mm_to_px(0.0, 0.0)
        x1, y1 = self.mm_to_px(self.plate_w_mm, self.plate_h_mm)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)

    def plate_mask(self, margin_mm: float = 0.0) -> np.ndarray:
        x0, y0 = self.mm_to_px(margin_mm, margin_mm)
        x1, y1 = self.mm_to_px(self.plate_w_mm - margin_mm, self.plate_h_mm - margin_mm)
        mask = np.zeros(self.shape, bool)
        mask[int(np.ceil(y0)) : int(np.floor(y1)), int(np.ceil(x0)) : int(np.floor(x1))] = True
        return mask


@dataclass
class LightSpec:
    """One white LED. Azimuth is measured in grid coordinates (x right, y down):
    90 deg = in front of the plate (+y), where the camera arch stands."""

    name: str
    elevation_deg: float
    azimuth_deg: float = 90.0
    pin: int | None = None
    pair: str | None = None  # the camera this light sits next to on the arch
    shape: bool = False  # used by the top camera for the 3D shape map

    def direction(self) -> np.ndarray:
        """Unit vector from the surface towards the light (x, y, z up)."""
        az, el = np.radians(self.azimuth_deg), np.radians(self.elevation_deg)
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def away_xy(self) -> np.ndarray:
        """Unit vector in the plate plane pointing away from the light."""
        az = np.radians(self.azimuth_deg)
        return -np.array([np.cos(az), np.sin(az)])


@dataclass
class Detection:
    z_threshold: float = 6.0
    sigma_floor: float = 0.01
    shape_sigma_floor: float = 0.01
    smooth_px: float = 1.0
    close_mm: float = 1.5
    group_mm: float = 2.5
    min_area_mm2: float = 0.05
    min_confidence: float = 0.6


@dataclass
class ImageMap:
    """One map in the analysis stack: a channel of (camera frame - dark frame)."""

    key: str
    view: str
    light: str
    channel: str  # "r", "g", "b" or "y" (luminance)


@dataclass
class Config:
    path: Path
    root: Path
    grid: Grid
    lights: dict[str, LightSpec]
    lights_active_high: bool
    detection: Detection
    cameras: dict[str, dict[str, Any]]
    capture: dict[str, Any]
    penetrant: dict[str, Any]
    llm: dict[str, Any]
    sim: dict[str, Any]
    data_dir: Path
    spec_path: Path
    classifier_path: Path

    @property
    def calibration_path(self) -> Path:
        return self.data_dir / "calibration.json"

    @property
    def golden_path(self) -> Path:
        return self.data_dir / "golden.npz"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ayas.db"

    @property
    def captures_dir(self) -> Path:
        return self.data_dir / "captures"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def is_sim(self) -> bool:
        return all(cam.get("backend") == "sim" for cam in self.cameras.values())

    def elevation_deg(self, view: str) -> float:
        return float(self.cameras[view].get("elevation_deg", {"top": 90, "oblique45": 45, "grazing": 15}[view]))

    # ---- light roles ---------------------------------------------------------------

    def pair_light(self, view: str) -> str:
        for name, spec in self.lights.items():
            if spec.pair == view:
                return name
        raise ValueError(f"no light has pair: {view}")

    @property
    def states(self) -> tuple[str, ...]:
        """Capture states: each light alone, then all lights off."""
        return (*self.lights, DARK)

    @property
    def align_key(self) -> str:
        """Image used to correct placement: the top camera's green channel under its own light."""
        return self.capture.get("align_key") or f"{key('top', self.pair_light('top'))}_g"

    @property
    def shape_lights(self) -> tuple[str, ...]:
        return tuple(n for n, s in self.lights.items() if s.shape)

    def image_maps(self) -> list[ImageMap]:
        """The images the analysis uses:
        - the top camera under its own light, in colour (r, g, b): material/colour cues;
        - the top camera under every other light, as brightness: relief and shading;
        - the 45 deg and grazing cameras under their own light, as brightness: scatter."""
        top_light = self.pair_light("top")
        maps = [ImageMap(key("top", top_light) + f"_{c}", "top", top_light, c) for c in CHANNELS]
        maps += [ImageMap(key("top", n), "top", n, "y") for n in self.lights if n != top_light]
        maps += [ImageMap(key(v, self.pair_light(v)), v, self.pair_light(v), "y") for v in VIEWS if v != "top"]
        return maps


def _dataclass_from(cls, raw: dict[str, Any] | None):
    raw = raw or {}
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)


def _lights_from(raw: dict[str, Any]) -> tuple[dict[str, LightSpec], bool]:
    leds = raw.get("leds") or {}
    if not leds:
        raise ValueError("config needs lights.leds (one entry per white LED)")
    lights = {}
    for name, spec in leds.items():
        if "_" in name or name == DARK:
            raise ValueError(f"light name {name!r}: no underscores, and 'dark' is reserved")
        lights[name] = _dataclass_from(LightSpec, {"name": name, **spec})
    for view in VIEWS:
        pairs = [n for n, s in lights.items() if s.pair == view]
        if len(pairs) != 1:
            raise ValueError(f"exactly one light needs pair: {view} (found {pairs})")
    return lights, bool(raw.get("active_high", True))


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    # Relative paths in the YAML are resolved against the project root.
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    paths = raw.get("paths", {})
    cameras = raw.get("cameras", {})
    missing = set(VIEWS) - set(cameras)
    if missing:
        raise ValueError(f"config {path} has no camera for views: {sorted(missing)}")
    lights, active_high = _lights_from(raw.get("lights", {}))
    return Config(
        path=path,
        root=root,
        grid=_dataclass_from(Grid, raw.get("grid")),
        lights=lights,
        lights_active_high=active_high,
        detection=_dataclass_from(Detection, raw.get("detection")),
        cameras=cameras,
        capture=raw.get("capture", {}),
        penetrant=raw.get("penetrant", {}),
        llm=raw.get("llm", {"backend": "mock"}),
        sim=raw.get("sim", {}),
        data_dir=root / paths.get("data_dir", "data"),
        spec_path=root / paths.get("spec", "configs/spec.yaml"),
        classifier_path=root / paths.get("classifier", "models/classifier.joblib"),
    )
