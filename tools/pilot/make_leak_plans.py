"""Leak stress test of the hood head: the pilot's hood plans with the rim lifted gap_mm above the plate
face (a warped part, a weld bead, or the hood held off the part), so ambient light gets in under it.
Only extras.hood.gap_mm / rests_on and the ring LED positions (glued to the hood walls: recomputed on the
new frustum at the same h_mm) change. Writes <out>/plans/<plate>_<tag>.json + a matrix manifest with the
C0 hood row (no ambient: no leak to test, copied from the pilot by the caller) and the lifted rows.
  make_leak_plans.py <pilot dir> <out> <gap_mm> <cond> [<cond> ...]
"""
import copy, json, sys
from pathlib import Path
import numpy as np
from ayas.twin import lighting as LT, matrix as MX

src, out, gap = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
conds = sys.argv[4:]
man = json.loads((src / MX.MANIFEST).read_text())
(out / "plans").mkdir(parents=True, exist_ok=True)
plates = []
for pl in man["plates"]:
    rows = []
    for r in pl["rows"]:
        if r["device_mode"] != "hood" or (r["condition"] not in conds and r["condition"] != "C0_dark"):
            continue
        plan = json.loads((src / r["plan"]).read_text())
        if r["condition"] != "C0_dark":
            plan = copy.deepcopy(plan)
            hood = plan["extras"]["hood"]
            hood["gap_mm"], hood["rests_on"] = gap, f"lifted {gap:g} mm (leak test)"
            ring = [L for L in plan["lights"] if L.get("mount") == "ring"]
            pts = LT.ring_points(plan, None, hood)
            target = LT.axis_point(plan)
            assert len(pts) == len(ring)
            for L, p in zip(ring, pts):
                L["pos_m"] = [round(float(v), 7) for v in p]
                d = target - p
                L["dir"] = [round(float(v), 7) for v in d / np.linalg.norm(d)]
            plan["matrix"]["leak_gap_mm"] = gap
        (out / r["plan"]).write_text(json.dumps(plan))
        rows.append(dict(r))
    plates.append({**pl, "rows": rows})
m2 = {**man, "conditions": ["C0_dark", *conds], "hood_conditions": ["C0_dark", *conds],
      "rows": [x for x in man["rows"] if x["device_mode"] == "hood" and x["condition"] in ("C0_dark", *conds)],
      "plates": plates, "leak_gap_mm": gap}
(out / MX.MANIFEST).write_text(json.dumps(m2, indent=1))
print(sum(len(p["rows"]) for p in plates), "plans")
