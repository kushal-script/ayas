"""3D surface of each detected defect, from the burst's left and right LED frames.

Photometric depth (docs/twin/PHYSICS_PINN.md): the left and right light-only frames (Lc = L - A,
Rc = R - A, aligned like the burst processing does) are mapped onto the plate grid, and for each
dent, pit or scratch the physics-informed height field (physics.depth_pinn) or the classical
slopes + Poisson baseline (physics.baselines) is fitted to that crop. The result is a small height
map per defect (mm, relative to the flat ring around it) for the station's 3D view, plus its depth.

Twin, 96 dents and pits, today's L/R lighting: PINN dent MAE 0.053 mm, classical 0.153 mm; pits are
under-estimated (reports/physics_pinn.md). It is a measurement aid shown next to the verdict: the
rules engine does not use it.

The fit needs the LED geometry and the exposure scale:
- twin bursts: the scene plan (data/twin_demo/plans/<scene>.json) and the burst's own 'k';
- the real device: physics.device.device_meta from the hood/pod geometry and a white-card
  calibration (not calibrated yet: 3D is then reported as unavailable).

torch is needed for the fit (the Pi app venv: pip install torch, CPU wheel). Everything else here
is numpy / OpenCV.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .. import lightproc as LP

KINDS = ("dent", "pit", "scratch")      # what the 3D view measures (relief defects)
VIEW_PX = 64                            # height-map side sent to the viewer


@dataclass
class SurfaceSource:
    """What one inspection keeps for an on-demand 3D measurement (no copies until it is asked for)."""
    frames: Mapping[str, np.ndarray]        # raw burst frames (frame px)
    shifts: Mapping[str, Sequence[float]]   # lightproc shifts relative to B1
    H_frame_to_grid: np.ndarray             # 3x3 (GridMap.H)
    grid_size: tuple[int, int]              # (W, H) px
    mm_per_px: float
    plate_corners_px: np.ndarray            # (4, 2) grid px: TL, TR, BR, BL
    plate_size_mm: Sequence[float]
    corner_plate_idx: Sequence[int] | None
    burst_meta: Mapping[str, Any]
    meta: Mapping[str, Any] = field(default_factory=dict)   # the source meta (scene, device mode, plan path)


def H_grid_plate(src: SurfaceSource) -> np.ndarray:
    """3x3 plate mm -> grid px, consistent with which plate corner the grid's TL is."""
    W, H = (float(x) for x in src.plate_size_mm)
    pc = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    idx = [int(i) for i in (src.corner_plate_idx if src.corner_plate_idx is not None else (0, 1, 2, 3))]
    return cv2.getPerspectiveTransform(pc[idx], np.asarray(src.plate_corners_px, np.float32)).astype(float)


def grid_frames(src: SurfaceSource) -> dict[str, np.ndarray]:
    """Aligned light-only luma frames on the plate grid: {'L': Lc, 'R': Rc} in FS (not clipped)."""
    f = src.frames
    al = {k: (np.asarray(f[k], np.float32) if k == "B1" else LP.align(np.asarray(f[k], np.float32), src.shifts[k]))
          for k in ("L", "A", "R") if k in f}
    out = {}
    for g in ("L", "R"):
        lc = LP.luma(al[g] - al["A"])
        out[g] = cv2.warpPerspective(lc, src.H_frame_to_grid, tuple(src.grid_size), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def plan_for(meta: Mapping[str, Any], plans_dir: str | Path | None) -> dict | None:
    """The twin scene plan of a twin burst (scene id from the burst meta or the file name)."""
    if not plans_dir:
        return None
    scene = meta.get("scene_id") or ((meta.get("twin") or {}).get("scene_id"))
    if not scene and meta.get("part_id"):
        scene = str(meta["part_id"]).rsplit("_v", 1)[0]
    p = Path(plans_dir) / f"{scene}.json"
    return json.loads(p.read_text()) if scene and p.exists() else None


def depth_meta(src: SurfaceSource, plans_dir: str | Path | None = None) -> tuple[dict | None, str]:
    """(meta for physics.depth_pinn, '') or (None, why 3D is unavailable)."""
    from ..physics import twin_eval as TE
    plan = plan_for({**dict(src.burst_meta.get("twin") or {}), **dict(src.meta)}, plans_dir)
    if plan is None:
        return None, ("3D needs the LED geometry: on the device this comes from the hood calibration "
                      "(white card), which is not done yet")
    k = src.burst_meta.get("k")
    m = {"lights": [dict(L) for L in plan["lights"]],
         "camera_centre": TE.camera_centre(plan).tolist(),
         "camera_R": plan["camera"]["R"], "camera_t": plan["camera"]["t"],
         "H_grid_plate": H_grid_plate(src).tolist(), "mm_per_px": float(src.mm_per_px),
         "k": float(k) if k is not None else float(TE.meta_from_plan(plan, TE.grid_geom(plan))["k"]),
         "sensor": dict((TE.capture_cfg().get("sensor") or {})),
         "illumination": str(((plan.get("extras") or {}).get("device") or {}).get("illumination") or "pods"),
         "hood": TE._hood_of(plan), "plate_size_mm": list(plan["plate"]["size_mm"])}
    return m, ""


def box_plate_mm(box_px: Sequence[float], Hgp: np.ndarray) -> list[float]:
    """Grid px box -> plate mm box [u0, v0, u1, v1] (via the inverse of H_grid_plate)."""
    x0, y0, x1, y1 = (float(v) for v in box_px)
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    uv = cv2.perspectiveTransform(pts[None], np.linalg.inv(Hgp))[0]
    return [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]


SCRATCH_HALF_MM = 1.5     # a scratch is measured in a window across its groove (twin_eval.scratch_segments)


def scratch_window(frames: Mapping[str, np.ndarray], box_px: Sequence[float], Hgp: np.ndarray,
                   half_mm: float = SCRATCH_HALF_MM) -> list[float]:
    """Plate-mm window across a scratch: centred on the brightest smoothed L + R pixel inside the
    detection box (low-angle light lights the groove walls), half_mm a side. A whole scratch is
    too long for one crop: averaged down to its box it is coarser than the groove itself."""
    x0, y0, x1, y1 = (int(round(v)) for v in box_px)
    img = frames["L"] + frames["R"]
    h, w = img.shape
    x0, x1, y0, y1 = max(0, x0), min(w, max(x1, x0 + 1)), max(0, y0), min(h, max(y1, y0 + 1))
    sub = cv2.GaussianBlur(np.ascontiguousarray(img[y0:y1, x0:x1], np.float32), (0, 0), 2.0)
    j, i = np.unravel_index(int(np.argmax(sub)), sub.shape)[::-1]
    uv = cv2.perspectiveTransform(np.array([[[x0 + j, y0 + i]]], np.float32), np.linalg.inv(Hgp))[0, 0]
    return [float(uv[0] - half_mm), float(uv[1] - half_mm), float(uv[0] + half_mm), float(uv[1] + half_mm)]


def _small(h: np.ndarray, n: int = VIEW_PX) -> np.ndarray:
    s = max(h.shape)
    if s <= n:
        return h
    f = n / s
    return cv2.resize(h.astype(np.float32), (max(2, round(h.shape[1] * f)), max(2, round(h.shape[0] * f))),
                      interpolation=cv2.INTER_AREA)


FAST = {"iters_material": 40, "iters": 110}   # half the default steps: twin dents 0.013 mm MAE (0.008 at full), 2x faster
_POOL = None


def _pool(workers: int):
    """A warm process pool (torch imported once per worker); one core per defect."""
    global _POOL
    if _POOL is None or getattr(_POOL, "_ayas_workers", 0) != workers:
        import concurrent.futures as cf
        import multiprocessing as mp
        if _POOL is not None:
            _POOL.shutdown(cancel_futures=True)
        _POOL = cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_warm)
        _POOL._ayas_workers = workers
        import atexit
        atexit.register(_shutdown)
    return _POOL


def _shutdown() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
        _POOL = None


def _warm() -> None:
    import torch  # noqa: F401
    from ..physics import depth_pinn  # noqa: F401
    torch.set_num_threads(1)


def warm_up(workers: int = 3) -> None:
    """Start the pool and import torch in every worker now (call at station start-up)."""
    p = _pool(workers)
    for f in [p.submit(_warm) for _ in range(workers)]:
        f.result()


def _one(args: tuple) -> dict:
    """Fit one defect (runs in a worker process)."""
    d, frames, meta, box, material, thickness_mm, method, n_seeds, threads, cfg = args
    from ..physics import baselines, depth_pinn as DP
    kind = "pit" if d["type"] == "pit" else "dent"
    if method == "classical":
        r = baselines.classical_depth(frames, meta, box, material, thickness_mm, threads=threads, return_map=True, cfg=cfg)
    else:
        r = DP.estimate_depth(frames, meta, box, material, thickness_mm, method="pinn", n_seeds=n_seeds,
                              threads=threads, return_map=True, kind=kind, cfg=cfg)
    h = r.height_map if r.height_map is not None else np.zeros((2, 2))
    hs = _small(np.asarray(h, float))
    s = float((r.crop or {}).get("s", meta["mm_per_px"])) * (max(h.shape) / max(hs.shape))
    return {"id": d["id"], "type": d["type"], "depth_mm": None if r.max_depth_mm is None else round(r.max_depth_mm, 3),
            "ci_mm": None if r.depth_ci_mm is None else round(r.depth_ci_mm, 3),
            "rel_thickness": None if r.rel_thickness is None else round(r.rel_thickness, 4),
            "box_mm": [round(v, 2) for v in box], "spacing_mm": round(s, 4),
            "height_mm": np.round(hs, 4).tolist(), "runtime_s": round(r.runtime_s, 1), "method": r.method,
            "notes": list(r.notes)}


def _crop_frames(frames: Mapping[str, np.ndarray], box_mm: Sequence[float], Hgp: np.ndarray, pad_mm: float
                 ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Only the grid pixels round one box (so a worker gets kB, not the whole plate), and the
    plate->grid matrix shifted to that sub-image."""
    u0, v0, u1, v1 = box_mm
    size = max(u1 - u0, v1 - v0)
    m = max(pad_mm, 0.6 * size) + 1.0
    pts = np.array([[u0 - m, v0 - m], [u1 + m, v0 - m], [u1 + m, v1 + m], [u0 - m, v1 + m]], np.float32)
    g = cv2.perspectiveTransform(pts[None], Hgp)[0]
    h, w = next(iter(frames.values())).shape
    x0, y0 = max(0, int(np.floor(g[:, 0].min()))), max(0, int(np.floor(g[:, 1].min())))
    x1, y1 = min(w, int(np.ceil(g[:, 0].max())) + 1), min(h, int(np.ceil(g[:, 1].max())) + 1)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    return {k: np.ascontiguousarray(v[y0:y1, x0:x1]) for k, v in frames.items()}, T @ Hgp


def measure(src: SurfaceSource, defects: Sequence[Mapping[str, Any]], material: str, thickness_mm: float | None,
            *, method: str = "pinn", n_seeds: int = 1, threads: int | None = None,
            plans_dir: str | Path | None = None, kinds: Sequence[str] = KINDS, fast: bool = True,
            workers: int = 1, on_item=None, first: str | None = None) -> dict[str, Any]:
    """3D of each relief defect: {'available', 'why', 'method', 'items': [...], 'runtime_s'}.
    Each item: id, type, depth_mm, ci_mm, box_mm, spacing_mm, height_mm (list of rows, <= VIEW_PX).
    A scratch is measured in a 3 x 3 mm window across its groove (box_mm is that window).
    fast: FAST step counts. workers > 1: defects in parallel in a warm process pool (1 thread each).
    on_item(item) is called as each defect finishes; 'first' (a defect id) is started first."""
    t0 = time.perf_counter()
    meta, why = depth_meta(src, plans_dir)
    targets = [d for d in defects if d.get("type") in kinds]
    if first is not None:
        targets.sort(key=lambda d: d["id"] != first)
    if meta is None:
        return {"available": False, "why": why, "method": method, "items": [], "runtime_s": 0.0}
    if not targets:
        return {"available": True, "why": "no dent, pit or scratch to measure", "method": method, "items": [],
                "runtime_s": 0.0}
    from ..physics import depth_pinn as DP
    frames = grid_frames(src)
    meta["grain_deg"] = DP.plate_grain_deg(frames["L"] + frames["R"], meta["H_grid_plate"], meta["plate_size_mm"])
    Hgp = np.asarray(meta["H_grid_plate"], float)
    cfg = dict(FAST) if fast else None
    jobs = []
    for d in targets:
        box = (scratch_window(frames, d["box_px"], Hgp) if d["type"] == "scratch" else box_plate_mm(d["box_px"], Hgp))
        sub, Hsub = _crop_frames(frames, box, Hgp, float(DP.DEFAULTS["margin_min_mm"]))
        jobs.append((d, sub, {**meta, "H_grid_plate": Hsub.tolist()}, box, material, thickness_mm, method, n_seeds,
                     1 if workers > 1 else threads, cfg))
    items = []
    if workers > 1 and len(jobs) > 1:
        import concurrent.futures as cf
        pool = _pool(workers)
        futs = {pool.submit(_one, j): j[0]["id"] for j in jobs}
        for f in cf.as_completed(futs):
            it = f.result()
            items.append(it)
            if on_item:
                on_item(it)
        order = {d["id"]: k for k, d in enumerate(targets)}
        items.sort(key=lambda it: order[it["id"]])
    else:
        for j in jobs:
            it = _one(j)
            items.append(it)
            if on_item:
                on_item(it)
    return {"available": True, "why": "", "method": method, "items": items,
            "runtime_s": round(time.perf_counter() - t0, 1)}
