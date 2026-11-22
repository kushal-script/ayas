"""Training pairs for the depth network: synthetic relief rendered by the physics forward model.

Each sample is one defect crop as physics.depth_pinn sees it (make_crop: <= 48 px working lattice,
grown by the same margin), rendered with depth_pinn.Renderer for the light geometry, camera and
exposure of a real TRAIN plan of the twin (hood ring or open pods), a material drawn round that
plate's prior, a smooth albedo/texture factor, micro-roughness and sensor noise.
Outputs (npz shards): obs (N, 2, 48, 48) L/R in FS, valid (N, 48, 48), height (N, 48, 48) mm on the
working lattice, meta per sample (spacing, kind, crop centre relative to the camera axis, illumination).

usage: python training/depthnet/datagen.py --plans <train plans dir> --out <dir> --n 20000 --shard 2000 --seed 0
Test plates (seed >= 900000) are refused.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from trinetra.physics import depth_pinn as DP, optics as O, twin_eval as TE

N = 48
GROUPS = ("L", "R")


def smooth_noise(rng, shape, sigma_px):
    import cv2
    a = rng.standard_normal(shape).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), max(0.5, sigma_px))
    return a / (a.std() + 1e-8)


def relief(rng, kind, uu, vv, size_mm):
    """Height (mm, negative = below the surface) of one synthetic defect centred at (0, 0)."""
    r = np.hypot(uu, vv)
    if kind == "dent":
        shape = rng.choice(["ball", "smooth", "punch"], p=[0.45, 0.35, 0.2])
        R = size_mm / 2
        depth = rng.uniform(0.04, min(0.8, 0.18 * size_mm))
        if shape == "ball":
            h = -depth * np.clip(1 - (r / R) ** 2, 0, None)
        elif shape == "smooth":
            a = rng.uniform(1.0, 2.2); th = rng.uniform(0, math.pi)
            x = uu * math.cos(th) + vv * math.sin(th); y = -uu * math.sin(th) + vv * math.cos(th)
            h = -depth * np.exp(-((x / (R / 1.6)) ** 2 + (y * a / (R / 1.6)) ** 2))
        else:
            h = -depth * (r < R * 0.8) + rng.uniform(0.0, 0.04) * np.exp(-((r - R * 0.85) / (0.12 * R + 0.05)) ** 2)
            h = h.astype(np.float64)
    elif kind == "pit":
        R = size_mm / 2
        depth = rng.uniform(0.05, 0.5)
        h = -depth * np.clip(1 - r / R, 0, None) ** rng.uniform(0.8, 1.6)
        if rng.random() < 0.3:                            # a small cluster
            for _ in range(rng.integers(1, 4)):
                c = rng.uniform(-1.5 * R, 1.5 * R, 2)
                rr = np.hypot(uu - c[0], vv - c[1]); R2 = R * rng.uniform(0.4, 0.9)
                h = np.minimum(h, -depth * rng.uniform(0.4, 1.0) * np.clip(1 - rr / R2, 0, None))
    else:                                                  # scratch groove across the window
        th = rng.uniform(0, math.pi); w = rng.uniform(0.1, 1.4); depth = rng.uniform(0.01, 0.45)
        d = -uu * math.sin(th) + vv * math.cos(th) + rng.uniform(-0.3, 0.3)
        prof = rng.choice(["v", "u"])
        h = -depth * (np.clip(1 - np.abs(d) / (w / 2), 0, None) if prof == "v" else np.exp(-(d / (w / 3)) ** 2))
        h = h + rng.uniform(0, 0.02) * np.exp(-((np.abs(d) - w / 2 - 0.05) / 0.06) ** 2)   # raised lips
    return h


def sample(rng, plans, cfg, sensor):
    import torch
    plan = plans[rng.integers(len(plans))]
    geom = TE.grid_geom(plan)
    meta = TE.meta_from_plan(plan, geom)
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    kind = rng.choice(["dent", "pit", "scratch"], p=[0.45, 0.25, 0.30])
    size = {"dent": rng.uniform(1.5, 12.0), "pit": rng.uniform(0.3, 2.5), "scratch": 3.0}[kind]
    m = max(cfg["margin_min_mm"], cfg["margin_frac"] * size)
    span = size + 2 * m
    s0 = float(geom["mm_per_px"])
    f = max(1, int(math.ceil(span / s0 / N)))
    s = s0 * f
    n = min(N, int(math.floor(span / s)))
    cu = rng.uniform(span / 2 + 1, W - span / 2 - 1); cv = rng.uniform(span / 2 + 1, H - span / 2 - 1)
    u0, v0 = cu - s * (n - 1) / 2, cv - s * (n - 1) / 2
    obs0 = {g: np.zeros((n, n)) for g in GROUPS}
    crop = DP.Crop(obs0, None, np.ones((n, n), bool), u0, v0, s, f, [cu - size / 2, cv - size / 2, cu + size / 2, cv + size / 2],
                   np.zeros((n, n), bool))
    prior = dict(DP.MATERIAL_PRIORS.get(plan["plate"]["material"]["kind"], DP.DEFAULT_PRIOR))
    meta = dict(meta); meta["env_material"] = prior
    grain = rng.uniform(0, 180)
    rend = DP.Renderer(crop, meta, list(GROUPS), cfg, O.tangent_from_grain(grain))
    ss = rend.ss
    nf = n * ss
    off = (np.arange(nf) + 0.5) / nf * (n * s) - n * s / 2
    uu, vv = np.meshgrid(off, off)
    h = relief(rng, kind, uu, vv, size)
    h = h + rng.uniform(0.0, 0.006) * smooth_noise(rng, h.shape, rng.uniform(0.6, 3.0) * ss)    # micro relief
    jit = lambda x, a: float(np.clip(x * math.exp(rng.normal(0, a)), 0.02, 0.98))
    mat = O.Material(torch.tensor([jit(prior["base"], 0.25)], dtype=rend.dtype),
                     torch.tensor(jit(prior["rough"], 0.25), dtype=rend.dtype),
                     torch.tensor(prior["metal"] if prior["metal"] >= 0.999 else jit(prior["metal"], 0.3), dtype=rend.dtype),
                     prior["aniso"] * rng.uniform(0.6, 1.4), rend.tangent)
    mfac = np.exp(rng.uniform(0.0, 0.35) * smooth_noise(rng, (n, n), rng.uniform(1.0, 6.0))).repeat(ss, 0).repeat(ss, 1)
    nuis = {"log_gain": {g: torch.tensor(rng.normal(0, 0.05), dtype=rend.dtype) for g in GROUPS},
            "env_b": {g: torch.tensor(abs(rng.normal(0, 0.003)), dtype=rend.dtype) for g in GROUPS},
            "env_gain": {g: torch.tensor(rng.normal(0, 0.2), dtype=rend.dtype) for g in GROUPS}}
    with torch.no_grad():
        out = rend.render(torch.as_tensor(h, dtype=rend.dtype), mat, nuis, torch.as_tensor(mfac, dtype=rend.dtype))
    e_fs = float(sensor.get("e_per_fs", 0)) or 3000.0
    rn = float(sensor.get("read_noise_e", 2.0)) / e_fs
    amb = rng.uniform(0, 0.02) if rng.random() < 0.5 else 0.0
    obs = []
    for g in GROUPS:
        I = out[g].numpy().astype(np.float64)
        var = (np.maximum(I + amb, 0) + amb) / e_fs + 2 * rn ** 2
        obs.append(I + rng.standard_normal(I.shape) * np.sqrt(var))
    hw = h.reshape(n, ss, n, ss).mean(axis=(1, 3)) if ss > 1 else h
    pad = lambda a: np.pad(a, ((0, N - n), (0, N - n)))
    cam = np.asarray(meta["camera_centre"], float) * 1000.0
    return (np.stack([pad(o) for o in obs]).astype(np.float32), pad(np.ones((n, n))).astype(np.uint8),
            pad(hw).astype(np.float32),
            {"plan": plan["scene_id"], "kind": kind, "size_mm": round(size, 3), "s_mm": s, "n": n,
             "rel_u_mm": cu - cam[0], "rel_v_mm": cv + cam[1], "illumination": meta["illumination"],
             "material": plan["plate"]["material"]["kind"], "depth_mm": float(-(hw.min() - np.median(np.r_[hw[0], hw[-1], hw[:, 0], hw[:, -1]])))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    plans = [json.loads(p.read_text()) for p in sorted(Path(a.plans).glob("*.json"))]
    plans = [p for p in plans if int(p.get("seed", 0)) < 900000 and p.get("split") in ("train", "val") and p.get("mode", "surface") == "surface"]
    assert plans, "no train plans"
    cfg = dict(DP.DEFAULTS); cfg["hood_env"] = True
    sensor = dict(TE.capture_cfg().get("sensor") or {})
    from trinetra import lightproc as LP
    sensor["e_per_fs"] = LP.electrons_per_fs(sensor)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    k, t0 = 0, time.time()
    while k * a.shard < a.n:
        X, V, Y, M = [], [], [], []
        while len(X) < a.shard:
            try:
                x, v, y, m = sample(rng, plans, cfg, sensor)
            except Exception as e:             # a degenerate geometry: skip it
                print("skip:", type(e).__name__, e); continue
            X.append(x); V.append(v); Y.append(y); M.append(m)
        np.savez_compressed(out / f"shard_{a.seed:03d}_{k:03d}.npz", obs=np.stack(X), valid=np.stack(V), height=np.stack(Y),
                            meta=np.frombuffer(json.dumps(M).encode(), np.uint8))
        k += 1
        print(f"shard {k}: {k * a.shard} samples, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
