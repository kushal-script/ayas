"""Hood partly off the plate (independent review, 15 Nov): the review plate s990004's hood rows with the
camera (and so the hood head and its ring) moved shift_mm along the plate's x axis, so the hood's
bottom opening covers only part of the plate: the near hood wall sits over the plate, the far side
overhangs the bench by the plate thickness (a gap of thickness + gap_mm all round off the plate). The
camera pose (target), render ROI, hood, ring LEDs and the pods' UV LEDs move; plate, defects and
ambient stay. Writes <out>/plans/<plate>_<tag>.json + a matrix manifest.
  .venv/bin/python tools/review/make_overhang_plans.py <review dir> <out> <shift_mm> <cond> [<cond> ...]
"""
import copy, json, math, sys
from pathlib import Path
import numpy as np
import yaml
from ayas.twin import frames as F, lighting as LT, matrix as MX, sampler as S

ROOT = Path(__file__).resolve().parents[2]
src, out, shift = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
conds = sys.argv[4:]
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
man = json.loads((src / MX.MANIFEST).read_text())
(out / "plans").mkdir(parents=True, exist_ok=True)
plates = []
for pl in man["plates"]:
    if pl["plate"] != "s990004":
        continue
    rows = []
    for r in pl["rows"]:
        if r["device_mode"] != "hood" or r["condition"] not in conds:
            continue
        plan = copy.deepcopy(json.loads((src / r["plan"]).read_text()))
        cam = plan["camera"]
        pose = cam["pose"]
        R0, t0 = np.asarray(cam["R"]), np.asarray(cam["t"])
        C0 = F.camera_centre(R0, t0)
        tgt = [pose["target_mm"][0] + shift, pose["target_mm"][1]]
        R, t = F.camera_pose(tgt, pose["distance_mm"], *pose["tilt_deg"], pose["roll_deg"])
        C = F.camera_centre(R, t)
        K = np.asarray(cam["K"])
        Ws, Hs = cam["full_res"]
        q = F.project(K, R, t, S.scene_points_world(plan))
        m = cfg["camera"]["roi_margin_px"]
        x0, y0 = max(0, int(math.floor(q[:, 0].min() - m))), max(0, int(math.floor(q[:, 1].min() - m)))
        x1, y1 = min(Ws, int(math.ceil(q[:, 0].max() + m)) + 1), min(Hs, int(math.ceil(q[:, 1].max() + m)) + 1)
        cam.update(R=[[float(x) for x in row] for row in R], t=[float(x) for x in t], roi=[x0, y0, x1 - x0, y1 - y0])
        pose["target_mm"] = [round(x, 3) for x in tgt]
        hood = plan["extras"]["hood"]
        hood.update(gap_mm=0.5, rests_on=f"part, overhanging its edge by {shift - 50:g} mm (review)", plate_fits=False)
        rot = R.T @ R0                                    # world rotation old -> new camera (identity here)
        for L in plan["lights"]:
            if L.get("mount") == "ring":
                continue
            L["pos_m"] = [round(float(v), 7) for v in C + rot @ (np.asarray(L["pos_m"]) - C0)]
            L["dir"] = [round(float(v), 7) for v in rot @ np.asarray(L["dir"])]
        ring = [L for L in plan["lights"] if L.get("mount") == "ring"]
        pts = LT.ring_points(plan, cfg, hood)
        target = LT.axis_point(plan)
        assert len(pts) == len(ring)
        for L, p in zip(ring, pts):
            L["pos_m"] = [round(float(v), 7) for v in p]
            d = target - p
            L["dir"] = [round(float(v), 7) for v in d / np.linalg.norm(d)]
        plan["matrix"]["overhang_shift_mm"] = shift
        (out / r["plan"]).write_text(json.dumps(plan))
        rows.append(dict(r))
        print(r["scene_id"], "roi", cam["roi"], "hood covers plate x from", round(tgt[0] - 100, 1), "mm")
    plates.append({**pl, "rows": rows})
m2 = {**man, "conditions": conds, "hood_conditions": conds, "seeds": [990004],
      "rows": [x for x in man["rows"] if x["device_mode"] == "hood" and x["condition"] in conds],
      "plates": plates, "overhang_shift_mm": shift}
(out / MX.MANIFEST).write_text(json.dumps(m2, indent=1))
