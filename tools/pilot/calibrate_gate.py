"""Gate calibration: every burst capture of a matrix rebuilt from its kept EXR (as post does) and
processed with the shipped capture.yaml, optionally with process.ghost_coarse_px overridden. Writes
[{scene, k, tag, gate, reasons, ghost_bins, ghost_bins_coarse, ambient_dominated_frac, r_ambient}]:
the change statistics behind gate.ghost_bins_max / ghost_coarse_max / ambient_dominated_max.
  .venv/bin/python tools/pilot/calibrate_gate.py <matrix dir> <exr keep dir> <out json> [coarse_px]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import yaml
from ayas import lightproc as LP
from ayas.twin import matrix as MX, post as P

ROOT = Path(__file__).resolve().parents[2]
mx, keep, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
cap = P.capture_config(cfg)
if len(sys.argv) > 4:
    cap = {**cap, "process": {**cap["process"], "ghost_coarse_px": int(sys.argv[4])}}
names = tuple((cfg.get("render") or {}).get("light_groups") or P.GROUPS)
post_cfg = cfg.get("post") or {}
LPm, B = P.capture_modules()
res, t0 = [], time.time()
for pp in MX.plan_paths(mx):
    scene = pp.stem
    work = keep / scene
    plan = json.loads((work / "plan.json").read_text())
    lights = P.read_light_groups(work / "render.exr", names)
    geom = P.grid_geometry(plan, cfg.get("grid") or {})
    pmask = P.plate_mask_image(geom, next(iter(lights.values())).shape)
    for k in range(max(1, int(post_cfg.get("variants_per_scene", 1)))):
        rng = np.random.default_rng([int(plan["seed"]), k])
        groups = P.scale_ambient(lights, P.amb_multiplier(rng, plan.get("ambient"), k, post_cfg))
        burst = B.simulate_burst(groups, plan, np.random.default_rng([int(plan["seed"]), k, 1 + P.ARMS.index("burst")]),
                                 cap, roi_mask=pmask)
        p = LP.process_burst(burst, pmask, cap)
        m = p.metrics
        r = {"scene": scene, "k": k, "tag": (plan.get("matrix") or {}).get("row"), "gate": p.gate.verdict,
             "reasons": list(p.gate.reasons or []), "ghost_bins": m["ghost_bins"],
             "ghost_bins_coarse": m["ghost_bins_coarse"],
             "ambient_dominated_frac": round(float(m["ambient_dominated_frac"]), 4), "r_ambient": round(float(m["r_ambient"]), 4)}
        res.append(r)
        print(json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
outp.write_text(json.dumps(res, indent=1))
print("DONE", len(res))
