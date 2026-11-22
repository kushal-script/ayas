"""Real-domain training crops for the depth network: Blender twin renders through the device's capture path.

For each TRAIN scene (work dir with plan.json + render.exr), per variant: twin_eval.grid_frames_burst
(sensor noise, shake, ambient, 5-frame burst processing, aligned Lc = L - A, Rc = R - A, rectified),
then for every labelled dent and pit (twin_eval.depth_targets) and scratch window (scratch_segments),
plus jittered copies of each box: the crop depth_pinn.make_crop gives the physics fit, and the true
height on that crop's lattice (textures.render_maps height, area-averaged to the crop spacing).
A capture the gate refuses (RETAKE) gives no crops. Test plates (seed >= 900000) are refused.

usage: python training/depthnet/blender_crops.py --work <dir of scene dirs> --out <dir> [--variants 3] [--jitter 2]
       [--workers 14] [--shard 1500]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

N = 48


def truth_on_crop(h: np.ndarray, W: float, H: float, crop) -> np.ndarray:
    """True height (mm) on the crop's working lattice: the texture height area-averaged to the spacing."""
    import cv2
    nv, nu = h.shape
    ru, rv = W / nu, H / nv
    n_v, n_u = next(iter(crop.obs.values())).shape
    sig = max(0.0, (crop.s / ru) / 2.355)
    u = crop.u0 + crop.s * np.arange(n_u)
    v = crop.v0 + crop.s * np.arange(n_v)
    c0, c1 = max(0, int((u[0] - 2 * crop.s) / ru)), min(nu, int((u[-1] + 2 * crop.s) / ru) + 2)
    r0, r1 = max(0, int((v[0] - 2 * crop.s) / rv)), min(nv, int((v[-1] + 2 * crop.s) / rv) + 2)
    sub = np.ascontiguousarray(h[r0:r1, c0:c1], np.float32)
    if sig > 0.3:
        sub = cv2.GaussianBlur(sub, (0, 0), sig)
    UU, VV = np.meshgrid(u / ru - 0.5 - c0, v / rv - 0.5 - r0)
    return cv2.remap(sub, UU.astype(np.float32), VV.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def scene_job(args):
    scene_dir, variants, jitter, seed = args
    import torch
    torch.set_num_threads(1)
    from trinetra.physics import depth_pinn as DP, twin_eval as TE
    from trinetra.twin import textures as TX
    scene_dir = Path(scene_dir)
    plan = json.loads((scene_dir / "plan.json").read_text())
    if int(plan.get("seed", 0)) >= 900000:
        return []
    geom = TE.grid_geom(plan)
    cap = TE.capture_cfg()
    meta = TE.meta_from_plan(plan, geom, cap)
    maps = TX.render_maps(plan)
    targets = TE.depth_targets(plan, maps, classes=("dent", "pit")) + TE.scratch_segments(plan, maps)
    if not targets:
        return []
    groups = TE.read_groups(scene_dir / "render.exr", ("L", "R", "AMB"))
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    cam = np.asarray(meta["camera_centre"], float) * 1000.0
    cfg = dict(DP.DEFAULTS)
    out = []
    for k in range(variants):
        rng = np.random.default_rng([int(plan.get("seed", 0)), seed, k])
        amb = float(rng.choice([0.0, 0.3, 1.0, 2.0]))
        frames, info = TE.grid_frames_burst(groups, plan, geom, rng, cap, amb_mult=amb)
        if info["gate"] == "RETAKE":
            continue
        for t in targets:
            for j in range(1 + jitter):
                box = list(t["box_mm"])
                if j:                                        # the detector's box is never exact
                    sz = max(box[2] - box[0], box[3] - box[1])
                    d = rng.normal(0, 0.12 * sz, 2); g = rng.uniform(0.85, 1.25)
                    cu, cv = (box[0] + box[2]) / 2 + d[0], (box[1] + box[3]) / 2 + d[1]
                    hw, hh = (box[2] - box[0]) / 2 * g, (box[3] - box[1]) / 2 * g
                    box = [cu - hw, cv - hh, cu + hw, cv + hh]
                try:
                    crop = DP.make_crop(frames, meta, box, cfg, ("L", "R"))
                except Exception:
                    continue
                o = np.stack([crop.obs["L"], crop.obs["R"]]).astype(np.float32)
                nv_, nu_ = o.shape[1:]
                if max(nv_, nu_) > N:
                    continue
                ht = truth_on_crop(maps["height"], W, H, crop)
                cu = crop.u0 + crop.s * (nu_ - 1) / 2; cv = crop.v0 + crop.s * (nv_ - 1) / 2
                pad = lambda a: np.pad(a, ((0, N - nv_), (0, N - nu_)))
                out.append((np.stack([pad(o[0]), pad(o[1])]), pad(crop.valid.astype(np.uint8)), pad(ht.astype(np.float32)),
                            {"plan": plan["scene_id"], "kind": t["cls"], "s_mm": crop.s, "n": max(nv_, nu_), "nv": nv_, "nu": nu_,
                             "rel_u_mm": cu - cam[0], "rel_v_mm": cv + cam[1], "illumination": meta["illumination"],
                             "material": plan["plate"]["material"]["kind"], "amb_mult": amb, "jitter": j,
                             "true_depth_mm": t["depth_mm"]}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--variants", type=int, default=3); ap.add_argument("--jitter", type=int, default=2)
    ap.add_argument("--workers", type=int, default=14); ap.add_argument("--shard", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    scenes = sorted(p.parent for p in Path(a.work).rglob("plan.json") if (p.parent / "render.exr").exists())
    if a.limit:
        scenes = scenes[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    buf, k, n_all, t0 = [], 0, 0, time.time()

    def flush(force=False):
        nonlocal buf, k
        while len(buf) >= a.shard or (force and buf):
            part, buf = buf[:a.shard], buf[a.shard:]
            np.savez_compressed(out / f"blender_{k:03d}.npz", obs=np.stack([p[0] for p in part]),
                                valid=np.stack([p[1] for p in part]), height=np.stack([p[2] for p in part]),
                                meta=np.frombuffer(json.dumps([p[3] for p in part]).encode(), np.uint8))
            k += 1

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(scene_job, (str(s), a.variants, a.jitter, a.seed)) for s in scenes]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                res = f.result()
            except Exception as e:
                print("scene failed:", type(e).__name__, e, flush=True); res = []
            buf += res; n_all += len(res)
            flush()
            if i % 20 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} scenes, {n_all} crops, {time.time() - t0:.0f} s", flush=True)
    flush(force=True)
    print("done", n_all, "crops in", k, "shards", flush=True)


if __name__ == "__main__":
    sys.exit(main())
