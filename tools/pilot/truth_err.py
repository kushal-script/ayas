"""Image-level corruption check of every burst capture of a rendered matrix: the device's M against
the noise-free, ambient-free LED image of the same render (what M would be with a perfect sensor in
the dark), both low-passed over a 15 x 15 px box (~1.1 mm on the grid, 15 sensor-ROI px) so pixel
noise averages out and only systematic (lighting) error is left. Reported per capture, inside the
plate eroded by 20 px: the 8-bit DN error after the device's own encoding (norm, gamma), p1 / p99
and mean |.|, and the relative error p1 / p99 of the linear luma. The burst is rebuilt from the
kept EXR exactly as post does (same rng streams), so gate and image equal the matrix outputs.
Run from the repo root:
  .venv/bin/python tools/pilot/truth_err.py <matrix dir> <exr keep dir> <out json> [--only prefix,...]
"""
import json, sys, time
from pathlib import Path
import cv2
import numpy as np
import yaml
from trinetra import lightproc as LP
from trinetra.twin import matrix as MX, post as P

ROOT = Path(__file__).resolve().parents[2]
a = sys.argv[1:]
mx, keep, outp = Path(a[0]), Path(a[1]), Path(a[2])
only = a[a.index("--only") + 1].split(",") if "--only" in a else None
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
cap = P.capture_config(cfg)
names = tuple((cfg.get("render") or {}).get("light_groups") or P.GROUPS)
post_cfg = cfg.get("post") or {}
LPm, B = P.capture_modules()
nvar = max(1, int(post_cfg.get("variants_per_scene", 1)))
res = []
t0 = time.time()
for pp in MX.plan_paths(mx):
    scene = pp.stem
    if only and not any(scene.startswith(x) for x in only):
        continue
    work = keep / scene
    plan = json.loads((work / "plan.json").read_text())
    lights = P.read_light_groups(work / "render.exr", names)
    geom = P.grid_geometry(plan, cfg.get("grid") or {})
    pmask = P.plate_mask_image(geom, next(iter(lights.values())).shape)
    comb = P.combine_of(plan, cap)
    inner = cv2.erode(pmask.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    for k in range(nvar):
        rng = np.random.default_rng([int(plan["seed"]), k])
        mult = P.amb_multiplier(rng, plan.get("ambient"), k, post_cfg)
        groups = P.scale_ambient(lights, mult)
        burst = B.simulate_burst(groups, plan, np.random.default_rng([int(plan["seed"]), k, 1 + P.ARMS.index("burst")]),
                                 cap, roi_mask=pmask)
        p = LP.process_burst(burst, pmask, cap)
        kexp = B.exposure_k(plan, cap, burst.meta.get("E_led"))["k"]
        g = burst.meta.get("led_gains") or {"L": 1.0, "R": 1.0}       # the burst's own pod imbalance (led_jitter):
        # review 15 Nov: without it the truth differed from M by up to the 0-5 % jitter (p1 -2 DN in the dark room)
        Mt = P._led_min({"L": groups["L"] * kexp * g["L"], "R": groups["R"] * kexp * g["R"]}, cap, comb).astype(np.float32)
        yM, yT = cv2.blur(LP.luma(p.M_lin), (15, 15)), cv2.blur(LP.luma(Mt), (15, 15))
        rel = (yM - yT) / np.maximum(yT, 1e-6)
        n = p.metrics["norm"]
        dn = 255.0 * (np.clip(yM / n, 0, 1) ** (1 / 2.2) - np.clip(yT / n, 0, 1) ** (1 / 2.2))
        # pixels the device masked (sat_mask, inpainted or kept white) are not M errors: leave them out
        ok = inner & ~cv2.dilate(p.sat_mask.view(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        m = p.metrics
        r = {"scene": scene, "k": k, "tag": (plan.get("matrix") or {}).get("row"), "gate": p.gate.verdict,
             "reasons": list(p.gate.reasons or []), "r_ambient": round(float(m["r_ambient"]), 4),
             "ghost_bins": m["ghost_bins"], "closure_sigma": round(float(m["closure_sigma"]), 3),
             "residual_sigma": round(float(m["residual_sigma"]), 3),
             "dn_p1": round(float(np.percentile(dn[ok], 1)), 2), "dn_p99": round(float(np.percentile(dn[ok], 99)), 2),
             "dn_absmean": round(float(np.abs(dn[ok]).mean()), 3),
             "dn_absmax": round(float(np.abs(dn[ok]).max()), 2),
             "rel_p1": round(float(np.percentile(rel[ok], 1)), 4), "rel_p99": round(float(np.percentile(rel[ok], 99)), 4)}
        res.append(r)
        print(json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
outp.write_text(json.dumps(res, indent=1))
print("DONE", len(res))
