import json, sys
from pathlib import Path
import numpy as np, cv2
from trinetra import rules, lightproc as LP
from trinetra.app import pipeline as PL, sources as SR, surface3d as S3
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
cfg = PL.load_app_config("configs/app.yaml")
spec = rules.load_spec("configs/spec_steel.yaml")
det, name = PL.build_detector("onnx:models/v8n_burst.onnx", 0.25)
pipe = PL.InspectionPipeline(cfg["_capture"], cfg["_grid"], det, spec, cfg["part"], None, log=None, advise="off", vision_name=name, image_px=1080)
tone = lambda f: (np.clip(np.asarray(f, np.float32), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
# 1. dent plate: detections, record, plate texture, 3D
item = SR.read_twin_npz(Path("artifacts/restore/dent_bursts/bursts/s900104_C0h_v0.npz"), cfg["part"])
insp = pipe.inspect(*item)
open(OUT / "det_overlay.jpg", "wb").write(insp.jpeg)
json.dump(insp.record, open(OUT / "record_dent.json", "w"), indent=1, default=str)
burst = item[0]
corners = np.asarray(item[1]["corners_px"], float)
proc = LP.process_burst(burst, PL.roi_mask(corners, burst.frames["B1"].shape[:2]), cfg["_capture"])
gm = PL.grid_map(corners, item[1].get("plate_size_mm"), cfg["_grid"], item[1].get("grid_plate_mm"), item[1].get("corner_plate_idx"))
grid = PL.rectify(proc.image_u8, gm)
cv2.imwrite(str(OUT / "plate_texture.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print("verdict", insp.record["verdict"], [(d["id"], d["type"], d["disposition"]) for d in insp.record["defects"]])
res = S3.measure(insp.surface, insp.record["defects"], "aluminium", 5.817, method="pinn", fast=False, n_seeds=1, threads=4,
                 plans_dir="artifacts/restore/dent_bursts/plans")
json.dump(res, open(OUT / "surface_dent.json", "w"))
print("3D", [(i["id"], i["type"], i["depth_mm"]) for i in res["items"]])
fr = S3.grid_frames(insp.surface)
d1 = next(d for d in insp.record["defects"] if d["type"] == "dent")
x0, y0, x1, y1 = (int(v) for v in d1["box_px"]); m = int(0.6 * (x1 - x0))
for g in ("L", "R"):
    c = fr[g][max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
    c = (np.clip(c / np.percentile(c, 99.5), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    cv2.imwrite(str(OUT / f"dent_{g}.png"), c)
# 2. open-mode window-light burst: the raw frames, the processed image and the gate
for sc in ("s900100_C3o_v0", "s900100_C6o_v0", "s900100_C3h_v0"):
    it = SR.read_twin_npz(Path(f"artifacts/jarvislabs/trinetra-render_574738_final/twin_matrix/post_final/bursts/{sc}.npz"), cfg["part"])
    b = it[0]; cn = np.asarray(it[1]["corners_px"], float)
    pr = LP.process_burst(b, PL.roi_mask(cn, b.frames["B1"].shape[:2]), cfg["_capture"])
    for k in LP.SEQUENCE:
        cv2.imwrite(str(OUT / f"{sc}_{k}.jpg"), cv2.cvtColor(cv2.resize(tone(b.frames[k]), (720, 720), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / f"{sc}_M.jpg"), cv2.cvtColor(cv2.resize(pr.image_u8, (720, 720), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    print(sc, "gate", pr.gate.verdict, pr.gate.message[:80])
