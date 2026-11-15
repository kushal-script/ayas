"""Re-run post (burst + naive arms, grids, sidecars; npz bursts for the plates named --bursts) on the kept
EXRs of a rendered matrix, after a lightproc / capture.yaml change. Old outputs must be removed first.
  repost.py <out> <exr keep dir> [--bursts s900011] [--only prefix ...]
"""
import json, sys, time
from pathlib import Path
import yaml
from trinetra.twin import matrix as MX, post as P

ROOT = Path(__file__).resolve().parents[2]
a = sys.argv[1:]
out, keep = Path(a[0]), Path(a[1])
bursts = a[a.index("--bursts") + 1].split(",") if "--bursts" in a else []
only = a[a.index("--only") + 1].split(",") if "--only" in a else None
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
if "--no-refine" in a:                     # the method before the closure refinement (round 0)
    cap = P.capture_config(cfg)
    cfg["capture_config"] = {**cap, "process": {**cap["process"], "align_refine": False}}
t0 = time.time()
n = 0
for p in MX.plan_paths(out):
    if only and not any(p.stem.startswith(x) for x in only):
        continue
    work = keep / p.stem
    assert (work / "render.exr").exists(), work
    sb = any(p.stem.startswith(x) for x in bursts)
    sc = P.process_scene(work, out, cfg, save_bursts=sb)
    n += 1
    print(p.stem, [v["gate"] for v in sc["variants"]], f"{time.time() - t0:.0f} s", flush=True)
P.write_data_yaml(out, cfg["classes"])
print("DONE", n, round(time.time() - t0, 1))
