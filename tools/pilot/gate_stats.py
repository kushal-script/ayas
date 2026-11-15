"""Gate statistics per (condition, device mode) row of a rendered matrix, from the scene sidecars."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

mx = Path(sys.argv[1])
man = json.loads((mx / "matrix_manifest.json").read_text())
rows = defaultdict(list)
for pl in man["plates"]:
    for r in pl["rows"]:
        p = mx / "sidecar" / f"{r['scene_id']}.json"
        if not p.exists():
            continue
        sc = json.loads(p.read_text())
        for v in sc["variants"]:
            b = v["arms"]["burst"]
            rows[(r["condition"], r["device_mode"])].append((pl["plate"], pl["part"]["material"], v["k"], v.get("lux"), b))
order = [(c, m) for m in ("hood", "open") for c in man["conditions"]]
def st(vals, f=max):
    vals = [x for x in vals if x is not None and np.isfinite(x)]
    return round(float(f(vals)), 4) if vals else None
out = {}
for key in order:
    rs = rows.get(key) or []
    if not rs:
        continue
    ms = [b["metrics"] for *_, b in rs]
    ret = [b for *_, b in rs if b["gate"] == "RETAKE"]
    reasons = Counter(x for b in ret for x in b["reasons"])
    out[f"{key[0]}/{key[1]}"] = d = {
        "n": len(rs), "retake": len(ret), "flagged": sum(b["gate"] == "PASS_FLAGGED" for *_, b in rs),
        "reasons": dict(reasons), "retaken": [f"{p}({mat[:6]})v{k}:{','.join(b['reasons'])}" for p, mat, k, _, b in rs if b["gate"] == "RETAKE"],
        "r_med": st([m.get("r_ambient") for m in ms], np.median), "r_max": st([m.get("r_ambient") for m in ms]),
        "masked_max": st([m.get("masked_frac") for m in ms]), "masked_amb_max": st([m.get("masked_ambient_frac") for m in ms]),
        "sat_a_max": st([m.get("sat_frac_a") for m in ms]),
        "led_min": st([m.get("led_level") for m in ms], min), "led_max": st([m.get("led_level") for m in ms]),
        "motion_max": st([m.get("motion_px") for m in ms]), "resid_max": st([m.get("residual_sigma") for m in ms]),
        "closure_max": st([m.get("closure_sigma") for m in ms]), "ghost_max": st([m.get("ghost_bins") for m in ms]),
        "lux": [round(l) if l is not None else None for *_, l, _ in rs],
    }
    print(f"{key[0]:18s} {key[1]:4s} n {d['n']:2d} retake {d['retake']:2d} flag {d['flagged']} r {d['r_med']}/{d['r_max']} "
          f"masked {d['masked_max']} amb {d['masked_amb_max']} satA {d['sat_a_max']} led {d['led_min']}-{d['led_max']} "
          f"mot {d['motion_max']} res {d['resid_max']} clo {d['closure_max']} gh {d['ghost_max']} {d['reasons']}")
    for x in d["retaken"]:
        print("      ", x)
(mx / "gate_stats.json").write_text(json.dumps(out, indent=1))
