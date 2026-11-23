"""Print the test plans of seeds 900010-900199 (material, size, labelled defects) to pick the pilot plates.
  .venv/bin/python tools/pilot/scan_seeds.py   (from the repo root)
"""
import sys, yaml
from pathlib import Path
from collections import Counter
from ayas.twin import sampler
cfg = yaml.safe_load(Path("configs/twin.yaml").read_text())
for seed in range(900010, 900200):
    p = sampler.make_plan(cfg, seed, "test", mode="surface", source="random", condition="C0_dark", device_mode="hood")
    pl = p["plate"]; mat = pl["material"]
    labs = [d for d in p.get("defects") or [] if d.get("label", True)]
    cls = Counter(f"{d['cls']}/{d['kind']}" for d in labs)
    print(seed, mat["kind"], pl["size_mm"], round(pl.get("thickness_mm", 0),1), len(labs), dict(cls), "dist", len(p.get("distractors") or []), "holes", len(p.get("holes") or []), "rough", mat.get("roughness"))
