"""On-device inputs for depth_pinn.estimate_depth (docs/twin/PHYSICS_PINN.md 5.3, 7).

On the Pi there is no plan: the depth estimator's 'meta' comes from the device geometry (the
hood and its LED ring, twin.yaml rig / lighting.HOOD, the same numbers the twin renders), the
camera intrinsics K (ChArUco calibration) and the plate pose recovered from the four plate
corners the pipeline already finds (cv2.solvePnP on a planar target of known size). The fixed
exposure k (FS per radiance unit) is the white-card calibration: a matte card of albedo
ref_albedo under both LED groups reads ref_full_scale (capture.yaml burst), the same definition
the twin uses (burst.exposure_k). numpy + cv2 only.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from . import optics as O


def pose_from_corners(K, corners_img, plate_size_mm: Sequence[float], corner_plate_idx: Sequence[int] | None = None):
    """(R, t) world->camera (frames.py convention, metres) from the 4 imaged plate corners.

    corners_img: (4, 2) image px of the plate corners in plate order (0,0), (W,0), (W,H), (0,H),
    or in image order with corner_plate_idx giving each one's plate corner index."""
    import cv2
    W, H = (float(x) for x in plate_size_mm)
    plate = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    img = np.asarray(corners_img, float).reshape(4, 2)
    if corner_plate_idx is not None:
        obj = plate[np.asarray(corner_plate_idx, int)]
    else:
        obj = plate
    world = np.c_[obj[:, 0] / 1000.0, -obj[:, 1] / 1000.0, np.zeros(4)]
    ok, rvec, tvec = cv2.solvePnP(world, img, np.asarray(K, float), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP failed on the plate corners")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def exposure_k(lights: Sequence[Mapping[str, Any]], R, t, capture_cfg: Mapping[str, Any]) -> float:
    """FS per radiance unit of the fixed exposure (burst.exposure_k without the twin's plan)."""
    from ayas.twin import lighting
    R = np.asarray(R, float)
    C = -R.T @ np.asarray(t, float)
    a = R[2]
    p = C + (0.0 - C[2]) / a[2] * a                         # optical axis on the plate face
    E = sum(lighting.light_irradiance_w(L, p) for L in lights if L.get("group") in ("L", "R"))
    b = capture_cfg.get("burst") or {}
    return float(b.get("ref_full_scale", 0.57)) * math.pi / (float(b.get("ref_albedo", 0.9)) * E)


def device_meta(K, corners_img, plate_size_mm, H_grid_plate, mm_per_px: float, capture_cfg: Mapping[str, Any],
                twin_cfg: Mapping[str, Any] | None = None, hood: Mapping[str, Any] | None = None,
                corner_plate_idx=None, k: float | None = None, sectors: Mapping[str, str] | None = None) -> dict:
    """estimate_depth 'meta' for a real hood capture: ring LEDs placed from the hood geometry and the
    plate pose, the camera centre, the grid mapping and the exposure scale.

    sectors (optional): {ring LED name: group} for a depth burst with ring sectors."""
    from ayas.twin import lighting
    R, t = pose_from_corners(K, corners_img, plate_size_mm, corner_plate_idx)
    hd = dict(lighting.HOOD, **(hood or {}))
    plan = {"plate": {"size_mm": list(plate_size_mm)}, "camera": {"R": R.tolist(), "t": t.tolist(),
                                                                  "K": np.asarray(K, float).tolist()},
            "extras": {"device": {"mode": "hood", "illumination": "ring"}, "hood": hd}}
    white = ((twin_cfg or {}).get("rig") or {}).get("white") or {}
    lights = lighting.ring_lights(plan, twin_cfg, white, hood=hd)
    kk = float(k) if k is not None else exposure_k(lights, R, t, capture_cfg)
    if sectors:
        lights = [dict(L, group=sectors.get(L["name"], L["group"])) for L in lights]
    return {"lights": lights, "camera_centre": (-R.T @ t).tolist(), "camera_R": R.tolist(), "camera_t": t.tolist(),
            "H_grid_plate": np.asarray(H_grid_plate, float).tolist(), "mm_per_px": float(mm_per_px), "k": kk,
            "sensor": dict(capture_cfg.get("sensor") or {}), "illumination": "ring", "hood": hd,
            "plate_size_mm": list(plate_size_mm)}
