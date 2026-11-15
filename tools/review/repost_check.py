"""Re-run the burst arm of kept pilot EXRs (exactly as post does: same rng streams) under the shipped
capture.yaml or with overrides, and report the gate and the error against the noise-free LED image
(tools/pilot/truth_err.py's measure, plus p0.1 / p99.9 and the share of the plate > 6 DN off).
  .venv/bin/python tools/review/repost_check.py <exr keep dir> <out json> [--set key.sub=value ...]
      [--only prefix,...] [--k 0,1]
Values are YAML (e.g. --set sensor.analogue_gain='{pods: 1.0, ring: 1.5}').
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from trinetra import lightproc as LP
from trinetra.twin import post as P

ROOT = Path(__file__).resolve().parents[2]


def main(argv):
    keep, outp = Path(argv[0]), Path(argv[1])
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    ks = [int(x) for x in argv[argv.index("--k") + 1].split(",")] if "--k" in argv else [0, 1]
    cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
    cap = copy.deepcopy(P.capture_config(cfg))
    for s in sets:
        key, val = s.split("=", 1)
        d = cap
        *head, last = key.split(".")
        for h in head:
            d = d.setdefault(h, {})
        d[last] = yaml.safe_load(val)
    names = tuple((cfg.get("render") or {}).get("light_groups") or P.GROUPS)
    post_cfg = cfg.get("post") or {}
    LPm, B = P.capture_modules()
    res, t0 = [], time.time()
    for work in sorted(p for p in keep.iterdir() if (p / "plan.json").exists()):
        scene = work.name
        if only and not any(scene.startswith(x) for x in only):
            continue
        plan = json.loads((work / "plan.json").read_text())
        lights = P.read_light_groups(work / "render.exr", names)
        geom = P.grid_geometry(plan, cfg.get("grid") or {})
        pmask = P.plate_mask_image(geom, next(iter(lights.values())).shape)
        comb = P.combine_of(plan, cap)
        inner = cv2.erode(pmask.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
        for k in ks:
            rng = np.random.default_rng([int(plan["seed"]), k])
            mult = P.amb_multiplier(rng, plan.get("ambient"), k, post_cfg)
            groups = P.scale_ambient(lights, mult)
            burst = B.simulate_burst(groups, plan, np.random.default_rng([int(plan["seed"]), k, 1 + P.ARMS.index("burst")]),
                                     cap, roi_mask=pmask)
            p = LP.process_burst(burst, pmask, cap)
            kexp = B.exposure_k(plan, cap, burst.meta.get("E_led"))["k"]
            g = burst.meta["led_gains"]
            Mt = P._led_min({"L": groups["L"] * kexp * g["L"], "R": groups["R"] * kexp * g["R"]}, cap, comb).astype(np.float32)
            yM, yT = cv2.blur(LP.luma(p.M_lin), (15, 15)), cv2.blur(LP.luma(Mt), (15, 15))
            n = p.metrics["norm"]
            dn = 255.0 * (np.clip(yM / n, 0, 1) ** (1 / 2.2) - np.clip(yT / n, 0, 1) ** (1 / 2.2))
            ok = inner & ~cv2.dilate(p.sat_mask.view(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
            # pixel noise of the encoded image (unfiltered luma DN against the truth)
            dpx = 255.0 * (np.clip(LP.luma(p.M_lin) / n, 0, 1) ** (1 / 2.2) - np.clip(LP.luma(Mt) / n, 0, 1) ** (1 / 2.2))
            m = p.metrics
            d = dn[ok]
            r = {"scene": scene, "k": k, "gate": p.gate.verdict, "reasons": list(p.gate.reasons),
                 "r_ambient": round(float(m["r_ambient"]), 4), "led_level": round(float(m["led_level"]), 4),
                 "masked": round(float(m["masked_frac"]), 4), "ghost": m["ghost_bins"], "ghost_c": m["ghost_bins_coarse"],
                 "closure": round(float(m["closure_sigma"]), 3), "resid": round(float(m["residual_sigma"]), 3),
                 "dn_p001": round(float(np.percentile(d, 0.1)), 2), "dn_p999": round(float(np.percentile(d, 99.9)), 2),
                 "dn_p1": round(float(np.percentile(d, 1)), 2), "dn_p99": round(float(np.percentile(d, 99)), 2),
                 "frac6": round(float((np.abs(d) > 6).mean()), 5),
                 "px_sd": round(float(np.std(dpx[ok])), 3)}
            res.append(r)
            print(json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
    outp.write_text(json.dumps(res, indent=1))
    print("DONE", len(res))


if __name__ == "__main__":
    main(sys.argv[1:])
