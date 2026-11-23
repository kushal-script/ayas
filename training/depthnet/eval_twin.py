"""Depth network on the Blender twin (out of domain: trained on physics-renderer crops only).

For each labelled dent, pit and scratch window of the given twin bursts: the same crop the PINN
fits (depth_pinn.make_crop), the network's height map (ONNX), and optionally the PINN refined from
it, against the twin's true depth.
usage: python training/depthnet/eval_twin.py --onnx <depthnet.onnx> --bursts <dir> --plans <dir> scene_v0 ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import model as M  # noqa: E402


def net_depth(sess, crop, meta):
    groups = ("L", "R")
    obs = np.stack([crop.obs[g] for g in groups]).astype(np.float32)
    n = obs.shape[-1]
    assert obs.shape[-2] == n or True
    cam = np.asarray(meta["camera_centre"], float) * 1000.0
    cu = crop.u0 + crop.s * (obs.shape[-1] - 1) / 2
    cv = crop.v0 + crop.s * (obs.shape[-2] - 1) / 2
    nn_ = max(obs.shape[-2:])
    o = np.zeros((2, nn_, nn_), np.float32); o[:, :obs.shape[-2], :obs.shape[-1]] = obs
    v = np.zeros((nn_, nn_), np.float32); v[:obs.shape[-2], :obs.shape[-1]] = crop.valid
    x = M.prepare(o, v, meta.get("illumination", "pods"), cu - cam[0], cv + cam[1], crop.s)
    t = time.perf_counter()
    h = sess.run(None, {"crop": x[None]})[0][0, 0] * M.H_UNIT_MM
    dt = time.perf_counter() - t
    hh = h[:obs.shape[-2], :obs.shape[-1]]
    return float(max(0.0, -hh.min())), hh, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("--bursts", required=True); ap.add_argument("--plans", required=True)
    ap.add_argument("scenes", nargs="+")
    a = ap.parse_args()
    import onnxruntime as ort
    from ayas import rules
    from ayas.app import pipeline as PL, sources as SR, surface3d as S3
    from ayas.physics import depth_pinn as DP, twin_eval as TE
    from ayas.twin import textures as TX
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    cfg = PL.load_app_config("configs/app.yaml")
    pipe = PL.InspectionPipeline(cfg["_capture"], cfg["_grid"], PL.FakeDetector(), rules.load_spec("configs/spec_steel.yaml"),
                                 cfg["part"], None, log=None, advise="off")
    rows = []
    for sc in a.scenes:
        insp = pipe.inspect(*SR.read_twin_npz(Path(a.bursts) / f"{sc}.npz", cfg["part"]))
        plan = json.loads((Path(a.plans) / f"{sc.rsplit('_v', 1)[0]}.json").read_text())
        maps = TX.render_maps(plan)
        gt = TE.depth_targets(plan, maps, classes=("dent", "pit")) + TE.scratch_segments(plan, maps)
        meta, why = S3.depth_meta(insp.surface, a.plans)
        frames = S3.grid_frames(insp.surface)
        c = dict(DP.DEFAULTS)
        for t in gt:
            crop = DP.make_crop(frames, meta, t["box_mm"], c, ("L", "R"))
            d_net, _, dt = net_depth(sess, crop, meta)
            rows.append((t["cls"], t["depth_mm"], d_net, dt))
            print(f"{sc} {t['id']:>3} {t['cls']:<7} true {t['depth_mm']:.3f}  net {d_net:.3f}  ({dt * 1e3:.1f} ms)", flush=True)
    for k in ("dent", "pit", "scratch"):
        e = [abs(n - g) for c_, g, n, _ in rows if c_ == k]
        if e:
            print(f"{k:<8} n {len(e)}  net MAE {np.mean(e):.3f} mm  median {np.median(e):.3f}")


def refine_compare(argv=None):
    """Network -> short grid fit, vs the fast PINN, on the same twin crops (depth error and time)."""
    import onnxruntime as ort
    from ayas import rules
    from ayas.app import pipeline as PL, sources as SR, surface3d as S3
    from ayas.physics import depth_pinn as DP, twin_eval as TE
    from ayas.twin import textures as TX
    import torch
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("--bursts", required=True); ap.add_argument("--plans", required=True)
    ap.add_argument("--iters", type=int, default=40); ap.add_argument("--iters-material", type=int, default=30)
    ap.add_argument("scenes", nargs="+")
    a = ap.parse_args(argv)
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    cfg = PL.load_app_config("configs/app.yaml")
    pipe = PL.InspectionPipeline(cfg["_capture"], cfg["_grid"], PL.FakeDetector(), rules.load_spec("configs/spec_steel.yaml"),
                                 cfg["part"], None, log=None, advise="off")
    rows = []
    for sc in a.scenes:
        insp = pipe.inspect(*SR.read_twin_npz(Path(a.bursts) / f"{sc}.npz", cfg["part"]))
        plan = json.loads((Path(a.plans) / f"{sc.rsplit('_v', 1)[0]}.json").read_text())
        maps = TX.render_maps(plan)
        gt = TE.depth_targets(plan, maps, classes=("dent", "pit")) + TE.scratch_segments(plan, maps)
        meta, _ = S3.depth_meta(insp.surface, a.plans)
        frames = S3.grid_frames(insp.surface)
        meta["grain_deg"] = DP.plate_grain_deg(frames["L"] + frames["R"], meta["H_grid_plate"], meta["plate_size_mm"])
        mat, th = plan["plate"]["material"]["kind"], plan["plate"]["thickness_mm"]
        for t in gt:
            kind = "pit" if t["cls"] == "pit" else "dent"
            crop = DP.make_crop(frames, meta, t["box_mm"], dict(DP.DEFAULTS), ("L", "R"))
            d_net, h_net, _ = net_depth(sess, crop, meta)
            t0 = time.perf_counter()
            r1 = DP.estimate_depth(frames, meta, t["box_mm"], mat, th, method="grid", n_seeds=1, threads=1, kind=kind,
                                   cfg={"h_init": h_net, "iters": a.iters, "iters_material": a.iters_material})
            t1 = time.perf_counter()
            r2 = DP.estimate_depth(frames, meta, t["box_mm"], mat, th, method="pinn", n_seeds=1, threads=1, kind=kind,
                                   cfg={"iters": 110, "iters_material": 40})
            t2 = time.perf_counter()
            rows.append((t["cls"], t["depth_mm"], d_net, r1.max_depth_mm, r2.max_depth_mm, t1 - t0, t2 - t1))
            print(f"{sc} {t['id']:>3} {t['cls']:<7} true {t['depth_mm']:.3f} | net {d_net:.3f} | net+fit {r1.max_depth_mm:.3f} ({t1 - t0:.2f} s)"
                  f" | fast PINN {r2.max_depth_mm:.3f} ({t2 - t1:.2f} s)", flush=True)
    for k in ("dent", "pit", "scratch"):
        R = [r for r in rows if r[0] == k]
        if R:
            f = lambda i: np.mean([abs(r[i] - r[1]) for r in R])
            print(f"{k:<8} n {len(R)} MAE: net {f(2):.3f} | net+fit {f(3):.3f} ({np.mean([r[5] for r in R]):.2f} s)"
                  f" | fast PINN {f(4):.3f} ({np.mean([r[6] for r in R]):.2f} s)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refine":
        refine_compare(sys.argv[2:])
    else:
        main()
