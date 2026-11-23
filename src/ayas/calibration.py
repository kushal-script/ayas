"""Map each camera onto the common top-down grid with a homography.

Default method: find the plate's own outline (bright plate on the black jig),
fit a line to each of its 4 edges and intersect them. The plate's known size
then fixes the mapping into millimetres. If detection fails for a camera
(e.g. the grazing view), put the 4 corner pixels in the config as `corners_px`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .config import VIEWS, Config, Grid


class CalibrationError(RuntimeError):
    pass


def find_plate_corners(gray: np.ndarray) -> np.ndarray:
    """Plate corners in image pixels, ordered TL, TR, BR, BL as they appear in the image."""
    img8 = gray if gray.dtype == np.uint8 else np.clip(gray * 255, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img8, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise CalibrationError("no bright plate found in the image")
    outline = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(outline)
    perimeter = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            break
    else:
        raise CalibrationError("plate outline is not a quadrilateral")
    corners = _order_corners(approx.reshape(4, 2).astype(np.float32))
    return _refine_corners(corners, outline.reshape(-1, 2).astype(np.float32))


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL by image position (the plate must be rotated less than ~45 deg)."""
    by_y = pts[np.argsort(pts[:, 1])]
    tl, tr = by_y[:2][np.argsort(by_y[:2, 0])]
    bl, br = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([tl, tr, br, bl], np.float32)


def _refine_corners(corners: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Fit a line to the middle of each edge and intersect neighbours (sub-pixel corners)."""
    lines = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        length = float(np.linalg.norm(b - a))
        direction = (b - a) / length
        rel = contour - a
        t = rel @ direction / length
        dist = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
        pts = contour[(t > 0.1) & (t < 0.9) & (dist < 4.0)]
        if len(pts) < 10:
            return corners
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    refined = []
    for i in range(4):
        # Corner i joins edge i-1 (corner i-1 -> i) and edge i (corner i -> i+1).
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        s = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
        refined.append(p1 + s[0] * d1)
    return np.array(refined, np.float32)


def homography_from_corners(image_corners: np.ndarray, grid: Grid, corner_order: Sequence[int] = (0, 1, 2, 3)) -> np.ndarray:
    """Image px -> grid px. corner_order[i] is the plate corner (0=TL, 1=TR, 2=BR, 3=BL
    of the plate) that appears at image position i (TL, TR, BR, BL)."""
    target = grid.plate_corners_px()[list(corner_order)]
    return cv2.getPerspectiveTransform(np.asarray(image_corners, np.float32), target)


@dataclass
class Calibration:
    homographies: dict[str, np.ndarray]
    image_corners: dict[str, list] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            view: {
                "H_image_to_grid": self.homographies[view].tolist(),
                "image_corners": self.image_corners.get(view),
                "method": self.methods.get(view),
            }
            for view in self.homographies
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `ayas calibrate` first")
        data = json.loads(path.read_text())
        return cls(
            homographies={v: np.array(d["H_image_to_grid"], np.float64) for v, d in data.items()},
            image_corners={v: d.get("image_corners") for v, d in data.items()},
            methods={v: d.get("method") for v, d in data.items()},
        )


def calibrate(bright: Mapping[str, np.ndarray], cfg: Config) -> Calibration:
    """bright: one evenly lit grey image (0..1) per view, with a good plate in the jig
    (see capture.bright_view: all lights summed, dark frame subtracted)."""
    cal = Calibration(homographies={})
    for view in VIEWS:
        cam = cfg.cameras[view]
        if cam.get("corners_px"):
            corners, method = np.array(cam["corners_px"], np.float32), "manual corners_px"
        else:
            corners, method = find_plate_corners(bright[view]), "plate outline"
        cal.homographies[view] = homography_from_corners(corners, cfg.grid, cam.get("corner_order", (0, 1, 2, 3)))
        cal.image_corners[view] = corners.tolist()
        cal.methods[view] = method
    return cal
