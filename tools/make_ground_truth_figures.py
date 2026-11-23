import json, sys
from pathlib import Path
import cv2, numpy as np
from ayas.app.pipeline import grid_map, rectify
ROOT = Path(sys.argv[1]); OUT = ROOT / "reports/figures/scenes"
B = ROOT / "artifacts/jarvislabs/ayas-render_574738_final/twin_matrix/post_final/bursts"
COL = {"scratch": (0, 200, 255), "dent": (255, 120, 0), "pit": (200, 0, 200), "corrosion": (0, 90, 255), "crack": (0, 0, 255), "stain": (0, 200, 0)}
for name in ("s900100_C0h_v0", "s900101_C0h_v0", "s900100_C6o_v0"):
    z = np.load(B / f"{name}.npz", allow_pickle=True)
    labels = json.loads(bytes(z["labels_json"]).decode())
    f = np.clip(z["frame_B1"].astype(np.float32), 0, 1) ** (1 / 2.2)
    img = (f * 255).astype(np.uint8)
    gm = grid_map(z["corners_px"], z["plate_size_mm"], {"mm_per_px": float(z["mm_per_px"]), "border_mm": float(z["border_mm"])},
                  z["grid_plate_mm"], z["corner_plate_idx"])
    g = cv2.cvtColor(rectify(img, gm), cv2.COLOR_RGB2BGR)
    for d in labels:
        col = COL.get(d["cls"], (255, 255, 255))
        for poly in d.get("polys_grid") or []:
            cv2.polylines(g, [np.asarray(poly, np.int32)], True, col, 3, cv2.LINE_AA)
        x0, y0, x1, y1 = map(int, d["bbox_grid"])
        cv2.rectangle(g, (x0 - 12, y0 - 12), (x1 + 12, y1 + 12), col, 3, cv2.LINE_AA)
        t = f"{d['id']} {d['cls']}" + ("" if d.get("label", True) else " (below label threshold)")
        ty = y0 - 24 if y0 > 60 else y1 + 50
        cv2.putText(g, t, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(g, t, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 2, cv2.LINE_AA)
    g = cv2.resize(g, (900, int(900 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
    kinds = ", ".join(f"{d['id']} {d['cls']}" for d in labels) or "no defects; the marks are decoys (2 blend marks, label residue)"
    bar = np.full((44, g.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"{name[:7]} rectified to 0.08 mm/px, ground truth: {kinds}", (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"ground_truth_{name}.jpg"), np.vstack([bar, g]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(name, kinds)
