"""Write the pilot lighting matrix (explicit plate seeds) in trinetra.twin.matrix's manifest format.

Seeds: for each material asked, the first seed >= 900010 (honesty rule: 900000-900004 calibrated the
gate) whose plate is 100 x 100 mm with >= 3 labelled defects (scan_seeds.py).
  .venv/bin/python tools/pilot/make_plans.py <out> 900011 900027 900028 900068
"""
import json, sys
from pathlib import Path
import yaml
from trinetra.twin import matrix as MX, sampler

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
(out / "plans").mkdir(parents=True, exist_ok=True)
rows = MX.rows_for(MX.ALL)
plates = []
for seed in seeds:
    plans = MX.matrix_plans(cfg, seed, rows)
    entries = []
    for plan in plans:
        rel = f"plans/{plan['scene_id']}.json"
        (out / rel).write_text(json.dumps(plan))
        m = plan["matrix"]
        entries.append({"scene_id": plan["scene_id"], "condition": m["condition"], "device_mode": m["device_mode"],
                        "tag": m["row"], "plan": rel})
    ref = plans[0]
    defects = [{"id": d.get("id"), "cls": d.get("cls"), "kind": d.get("kind"), "label": d.get("label", True)}
               for d in ref.get("defects") or []]
    plates.append({"plate": sampler.scene_id(seed), "seed": seed, "part": MX.part_of(ref),
                   "clean": not any(d["label"] for d in defects), "defects": defects, "rows": entries})
manifest = {"schema": MX.SCHEMA, "split": "test", "start_seed": min(seeds), "n_plates": len(seeds), "min_clean": 0,
            "n_clean": sum(p["clean"] for p in plates), "seeds": seeds,
            "selection": "per material, the first seed >= 900010 with a 100 x 100 mm plate and >= 3 labelled defects",
            "conditions": list(MX.ALL), "hood_conditions": list(MX.ALL),
            "rows": [{"condition": c, "device_mode": m, "tag": MX.row_tag(c, m)} for c, m in rows], "plates": plates}
(out / MX.MANIFEST).write_text(json.dumps(manifest, indent=1))
for p in plates:
    print(p["plate"], p["part"], [f"{d['cls']}/{d['kind']}" for d in p["defects"] if d["label"]], len(p["rows"]), "rows")
