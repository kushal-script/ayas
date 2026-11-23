"""Coordinate frames shared by the twin, the real-capture tools and the Pi pipeline.

Plate frame (mm): origin at the plate's datum corner, u right and v down along the
front face; w is height out of the face. This is the frame of spec.yaml zones and of
the ground-truth CSV.

World frame (Blender, metres, right-handed): X = u/1000, Y = -v/1000, Z = w/1000.
The plate's front face is the plane Z = 0 and the plate body is -t <= Z <= 0.

Camera frame (OpenCV): x right, y down, z forward; X_cam = R @ X_world + t (metres).
Image pixels follow OpenCV: the centre of pixel (row i, col j) is at (x=j, y=i).
"""

from __future__ import annotations

import math

import numpy as np

# Raspberry Pi AI Camera (IMX500): 4056 x 3040 px of 1.55 um, 4.74 mm lens (nominal;
# replace with the ChArUco calibration).
IMX500 = {"width": 4056, "height": 3040, "pixel_um": 1.55, "focal_mm": 4.74}

# Straight-down camera at roll 0: image x along +u, image y along +v, looking at -Z.
R_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

# OpenCV camera axes -> Blender camera axes (Blender cameras look along -z with y up).
CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0])


def nominal_K(cam: dict | None = None) -> np.ndarray:
    c = cam or IMX500
    f = c["focal_mm"] / (c["pixel_um"] / 1000.0)
    return np.array([[f, 0.0, (c["width"] - 1) / 2], [0.0, f, (c["height"] - 1) / 2], [0.0, 0.0, 1.0]])


def plate_to_world(uv_mm, w_mm=0.0) -> np.ndarray:
    """(N, 2) plate mm (+ optional height) -> (N, 3) world metres."""
    uv = np.atleast_2d(np.asarray(uv_mm, float))
    w = np.broadcast_to(np.asarray(w_mm, float), uv.shape[:1])
    return np.stack([uv[:, 0] / 1000.0, -uv[:, 1] / 1000.0, w / 1000.0], axis=1)


def world_to_plate(xyz_m) -> np.ndarray:
    """(N, 3) world metres -> (N, 3) plate (u, v, w) mm."""
    p = np.atleast_2d(np.asarray(xyz_m, float))
    return np.stack([p[:, 0] * 1000.0, -p[:, 1] * 1000.0, p[:, 2] * 1000.0], axis=1)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def camera_pose(target_uv_mm, distance_mm: float, tilt_x_deg: float = 0.0, tilt_y_deg: float = 0.0,
                roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera whose optical axis hits the plate at target_uv_mm.

    Start from the straight-down view (image upright: u right, v down), then roll about
    the optical axis and tilt about the camera's own x and y axes. distance_mm is
    measured along the optical axis from the lens to the target point.
    """
    R = _rx(math.radians(tilt_x_deg)) @ _ry(math.radians(tilt_y_deg)) @ _rz(math.radians(roll_deg)) @ R_DOWN
    target = plate_to_world([target_uv_mm])[0]
    centre = target - (distance_mm / 1000.0) * R[2]   # R[2] = optical axis in world coordinates
    return R, -R @ centre


def camera_centre(R, t) -> np.ndarray:
    return -np.asarray(R).T @ np.asarray(t)


def project(K, R, t, xyz_m) -> np.ndarray:
    """World metres (N, 3) -> pixels (N, 2)."""
    Xc = np.atleast_2d(np.asarray(xyz_m, float)) @ np.asarray(R).T + np.asarray(t)
    x = Xc @ np.asarray(K).T
    return x[:, :2] / x[:, 2:3]


def homography_plate_to_image(K, R, t) -> np.ndarray:
    """3x3 H mapping plate (u, v, 1) mm on the front face (w = 0) to image pixels."""
    R = np.asarray(R, float)
    M = np.stack([R[:, 0] / 1000.0, -R[:, 1] / 1000.0, np.asarray(t, float)], axis=1)
    H = np.asarray(K, float) @ M
    return H / H[2, 2]


def homography_plate_to_grid(mm_per_px: float, border_mm: float) -> np.ndarray:
    """Plate mm -> rectified grid pixels, matching config.Grid.mm_to_px."""
    s = 1.0 / mm_per_px
    return np.array([[s, 0.0, border_mm * s], [0.0, s, border_mm * s], [0.0, 0.0, 1.0]])


def apply_h(H, pts) -> np.ndarray:
    p = np.atleast_2d(np.asarray(pts, float))
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:3]


def roi_K(K, x0: float, y0: float) -> np.ndarray:
    """Intrinsics of a window of the sensor whose top-left full-sensor pixel is (x0, y0)."""
    Kr = np.array(K, float)
    Kr[0, 2] -= x0
    Kr[1, 2] -= y0
    return Kr


def blender_cam2world(R, t) -> np.ndarray:
    """4x4 matrix_world for a Blender camera with OpenCV extrinsics (R, t)."""
    R = np.asarray(R, float)
    M = np.eye(4)
    M[:3, :3] = R.T @ CV_TO_BLENDER
    M[:3, 3] = camera_centre(R, t)
    return M


def blender_intrinsics(K, width: int, height: int, sensor_width_mm: float = 36.0) -> dict:
    """Blender camera data for pinhole K at (width, height), sensor_fit AUTO, square pixels.

    Same formulas as BlenderProc's set_intrinsics_from_K_matrix (pixel centres at integers).
    """
    K = np.asarray(K, float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if abs(fx - fy) > 1e-3 * fx:
        raise ValueError("non-square pixels are not supported (fx != fy)")
    view = float(max(width, height))
    return {
        "lens_mm": fx * sensor_width_mm / view,
        "sensor_width_mm": sensor_width_mm,
        "sensor_fit": "AUTO",
        "shift_x": -(cx - (width - 1) / 2) / view,
        "shift_y": (cy - (height - 1) / 2) / view,
        "resolution": [int(width), int(height)],
    }


def order_corners(pts) -> np.ndarray:
    """Order 4 image points TL, TR, BR, BL by image position (as the Pi pipeline does)."""
    p = np.asarray(pts, float).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])     # image y down: increasing angle is clockwise
    p = p[np.argsort(ang)]
    start = int(np.argmin(p[:, 0] + p[:, 1]))             # TL has the smallest x + y
    return np.roll(p, -start, axis=0)
