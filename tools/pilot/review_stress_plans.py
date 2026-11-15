"""Second review: physical adversarial plans from the pilot plans (render with tools/pilot/run_pilot.py).
  mirror    s900068 (ss_brushed) made mirror-polished: roughness 0.02, no anisotropy, grain 0.05 um
  darkscale s900027 (ms_hot_rolled) with the mill-scale base colour x 0.35 (very dark scale)
  overhang  s900028 (aluminium) with the hood moved so its rim crosses the plate 15 mm in from one edge:
            the hood rests on the plate (gap 0.2 mm) and there is a 5.9 mm slit over the bench side
  .venv/bin/python tools/pilot/review_stress_plans.py data/twin_pilot <out root>   (then run_pilot.py on <out root>/adv_<set>)
"""
import copy, json, math, sys
from pathlib import Path
import numpy as np
import yaml
from trinetra.twin import frames as F, lighting as LT, matrix as MX

ROOT = Path(__file__).resolve().parents[2]
src, root = Path(sys.argv[1]), Path(sys.argv[2])
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
man = json.loads((src / MX.MANIFEST).read_text())
SETS = {
    "mirror": ("s900068", ["C0h", "C3h", "C6h", "C0o", "C1o"]),
    "darkscale": ("s900027", ["C0h", "C3h", "C6h", "C0o", "C2o"]),
    "overhang": ("s900028", ["C0h", "C2h", "C3h", "C6h"]),
}


def mirror(plan):
    m = plan["plate"]["material"]
    m.update(roughness=0.02, anisotropy=0.0, grain_amp_um=0.05)


def darkscale(plan):
    m = plan["plate"]["material"]
    m["base_color"] = [round(c * 0.35, 4) for c in m["base_color"]]


def overhang(plan):
    W, H = plan["plate"]["size_mm"]
    cam = plan["camera"]
    pose = cam["pose"]
    target = [-15.0, H / 2.0]                   # hood centre 65 mm left of the plate centre: rim at u = 85 mm
    R, t = F.camera_pose(target, pose["distance_mm"], 0.0, 0.0, 0.0)
    K = np.asarray(cam["K"], float)
    Ws, Hs = cam["full_res"]
    q = F.project(K, R, t, F.plate_to_world([[0, 0], [W, 0], [W, H], [0, H]]))
    assert (q[:, 0] >= 0).all() and (q[:, 0] < Ws).all() and (q[:, 1] >= 0).all() and (q[:, 1] < Hs).all(), q
    m = cfg["camera"]["roi_margin_px"]
    x0, y0 = max(0, int(math.floor(q[:, 0].min() - m))), max(0, int(math.floor(q[:, 1].min() - m)))
    x1, y1 = min(Ws, int(math.ceil(q[:, 0].max() + m)) + 1), min(Hs, int(math.ceil(q[:, 1].max() + m)) + 1)
    cam.update(R=[[float(x) for x in row] for row in R], t=[float(x) for x in t], roi=[x0, y0, x1 - x0, y1 - y0])
    pose.update(target_mm=target, tilt_deg=[0.0, 0.0], roll_deg=0.0)
    hood = plan["extras"]["hood"]
    hood.update(gap_mm=0.2, rests_on="part (overhang: rim crosses the plate, slit over the bench)", plate_fits=False)
    old = [L for L in plan["lights"] if L.get("mount") == "ring"]
    pw = {g: [L["power_w"] for L in old if L["group"] == g] for g in ("L", "R")}
    w = {"kind": old[0]["kind"], "spot_deg": old[0]["spot_deg"], "blend": old[0]["blend"],
         "radius_mm": old[0]["radius_m"] * 1000.0, "power_w": 1.0, "color": old[0]["color"]}
    new = LT.ring_lights(plan, cfg, white=w, hood=hood)
    idx = {"L": 0, "R": 0}
    for L in new:
        L["power_w"] = pw[L["group"]][idx[L["group"]] % len(pw[L["group"]])]
        idx[L["group"]] += 1
    plan["lights"] = [L for L in plan["lights"] if L.get("mount") != "ring"] + new


EDIT = {"mirror": mirror, "darkscale": darkscale, "overhang": overhang}
for name, (plate, tags) in SETS.items():
    out = root / f"adv_{name}"
    (out / "plans").mkdir(parents=True, exist_ok=True)
    pl = next(p for p in man["plates"] if p["plate"] == plate)
    rows = [r for r in pl["rows"] if r["tag"] in tags]
    for r in rows:
        plan = copy.deepcopy(json.loads((src / r["plan"]).read_text()))
        EDIT[name](plan)
        plan.setdefault("matrix", {})["review_case"] = name
        (out / r["plan"]).write_text(json.dumps(plan))
    m2 = {**man, "conditions": sorted({r["condition"] for r in rows}), "hood_conditions": sorted({r["condition"] for r in rows if r["device_mode"] == "hood"}),
          "rows": [x for x in man["rows"] if x["tag"] in tags], "plates": [{**pl, "rows": rows}], "review_case": name}
    (out / MX.MANIFEST).write_text(json.dumps(m2, indent=1))
    print(name, len(rows), "plans ->", out)
