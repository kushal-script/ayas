"""Plans for the independent adversarial review (15 Nov) of the device v2 pilot (reports/pilot_lighting.md).

Seeds 99xxxx: outside the training range (< 900000) and far from the planned lighting matrix
(--start-seed 900100), so nothing here touches a future test plate. Plates:
  s990004 ss_2b (not in the pilot), as sampled: every hood row + C0o, C3o, C5o  (reproduction)
  s990029 ss_brushed made mirror-polished (roughness 0.02, no anisotropy): C0h, C3h, C6h, C0o
  s990014 ms_hot_rolled made very dark (base colour x 0.3, ~0.03-0.05):    C0h, C3h, C6h, C0o
Only plate.material changes for the adversarial plates; geometry, defects and lights are the sampler's.
  .venv/bin/python tools/review/make_review_plans.py <out>
"""
import copy, json, sys
from pathlib import Path
import yaml
from trinetra.twin import matrix as MX, sampler

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
H = [(c, "hood") for c in MX.ALL]
SETS = {
    990004: (H + [("C0_dark", "open"), ("C3_window", "open"), ("C5_reflected_lamp", "open")], None),
    990029: ([("C0_dark", "hood"), ("C3_window", "hood"), ("C6_sun", "hood"), ("C0_dark", "open")], "mirror"),
    990014: ([("C0_dark", "hood"), ("C3_window", "hood"), ("C6_sun", "hood"), ("C0_dark", "open")], "dark"),
}


def adversarial(plan, how):
    m = plan["plate"]["material"]
    if how == "mirror":
        m.update(roughness=0.02, anisotropy=0.0, grain_amp_um=0.05)
        plan["review"] = "mirror-polished stainless (roughness 0.02)"
    elif how == "dark":
        m["base_color"] = [round(0.3 * c, 4) for c in m["base_color"]]
        plan["review"] = "very dark mill scale (base colour x 0.3)"
    return plan


(out / "plans").mkdir(parents=True, exist_ok=True)
plates, all_rows = [], []
for seed, (rows, how) in SETS.items():
    plans = MX.matrix_plans(cfg, seed, rows)
    entries = []
    for plan in plans:
        plan = adversarial(copy.deepcopy(plan), how)
        rel = f"plans/{plan['scene_id']}.json"
        (out / rel).write_text(json.dumps(plan))
        m = plan["matrix"]
        entries.append({"scene_id": plan["scene_id"], "condition": m["condition"], "device_mode": m["device_mode"],
                        "tag": m["row"], "plan": rel})
    ref = plans[0]
    defects = [{"id": d.get("id"), "cls": d.get("cls"), "kind": d.get("kind"), "label": d.get("label", True)}
               for d in ref.get("defects") or []]
    plates.append({"plate": sampler.scene_id(seed), "seed": seed, "part": MX.part_of(ref), "review": how,
                   "clean": not any(d["label"] for d in defects), "defects": defects, "rows": entries})
    all_rows += [r for r in rows if r not in all_rows]
manifest = {"schema": MX.SCHEMA, "split": "test", "start_seed": min(SETS), "n_plates": len(SETS), "min_clean": 0,
            "n_clean": sum(p["clean"] for p in plates), "seeds": list(SETS),
            "selection": "independent review plates (tools/review/make_review_plans.py)",
            "conditions": list(MX.ALL), "hood_conditions": list(MX.ALL),
            "rows": [{"condition": c, "device_mode": d, "tag": MX.row_tag(c, d)} for c, d in all_rows], "plates": plates}
(out / MX.MANIFEST).write_text(json.dumps(manifest, indent=1))
for p in plates:
    print(p["plate"], p["part"], p["review"], [f"{d['cls']}/{d['kind']}" for d in p["defects"] if d["label"]], len(p["rows"]), "rows")
