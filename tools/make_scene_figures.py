import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(sys.argv[1]); OUT = ROOT / "reports/figures/scenes"; OUT.mkdir(parents=True, exist_ok=True)
B = ROOT / "artifacts/jarvislabs/ayas-render_574738_final/twin_matrix/post_final/bursts"
COND = {"C0": "dark room", "C1": "office", "C2": "warm bulb", "C3": "window", "C4": "side lamp", "C5": "reflected lamp", "C6": "direct sun"}
MAT = {"ms_hot_rolled": "mild steel, hot-rolled (mill scale)", "ss_brushed": "stainless, brushed"}

def img(f, size):
    a = np.clip(f.astype(np.float32), 0, 1) ** (1 / 2.2)
    a = (a[..., ::-1] * 255).astype(np.uint8)          # RGB -> BGR
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)

def label(im, text, h=34, scale=0.62):
    bar = np.full((h, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, h - 11), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, im])

def title(w, text, h=46):
    bar = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(bar, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return bar

def pad(im, p=6):
    return cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def load(name):
    z = np.load(B / f"{name}.npz", allow_pickle=True)
    return z, json.loads(bytes(z["meta_json"]).decode()), json.loads(bytes(z["labels_json"]).decode())

# 1. one plate under all 7 lightings, open vs hood (frame L = left LED pod + whatever ambient light)
for plate in ("s900100", "s900101"):
    rows = []
    for mode, mname in (("o", "open handheld"), ("h", "hood")):
        cells = []
        for c in COND:
            z, m, _ = load(f"{plate}_{c}{mode}_v0")
            cells.append(pad(label(img(z["frame_L"], 300), f"{COND[c]}, {float(z['lux']):,.0f} lx")))
        rows.append(np.hstack(cells))
    z, _, _ = load(f"{plate}_C0h_v0")
    grid = np.vstack([label(rows[0], "OPEN (handheld, 2 LED pods)", 30, 0.7), label(rows[1], "HOOD (LED ring, light-tight)", 30, 0.7)])
    t = title(grid.shape[1], f"Blender twin, plate {plate}: {MAT.get(str(z['material']), z['material'])}, {float(z['thickness_mm']):.1f} mm, 100 x 100 mm. Same plate, 7 lightings, 2 device modes")
    cv2.imwrite(str(OUT / f"lighting_grid_{plate}.jpg"), np.vstack([t, grid]), [cv2.IMWRITE_JPEG_QUALITY, 90])

# 2. the 5-frame burst of one capture
for name in ("s900100_C3o_v0", "s900100_C3h_v0"):
    z, m, _ = load(name)
    names = {"B1": "B1: both LEDs", "L": "L: left LED", "A": "A: LEDs off (ambient)", "R": "R: right LED", "B2": "B2: both LEDs"}
    cells = [pad(label(img(z[f"frame_{k}"], 360), names[k])) for k in z["sequence"]]
    row = np.hstack(cells)
    mode = "open" if name[-4] == "o" else "hood"
    cv2.imwrite(str(OUT / f"burst_{name}.jpg"),
                np.vstack([title(row.shape[1], f"One capture = 5-frame burst ({mode} mode, window light, {float(z['lux']):,.0f} lx), plate {name[:7]}"), row]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

# 3. ground-truth defects drawn on the frame (labels_json polygons in plate mm -> frame px via H)
COL = {"scratch": (0, 200, 255), "dent": (255, 120, 0), "pit": (200, 0, 200), "corrosion": (0, 90, 255), "crack": (0, 0, 255), "stain": (0, 200, 0)}
for name in ("s900100_C0h_v0", "s900101_C0h_v0"):
    z, m, labels = load(name)
    f = np.clip(z["frame_B1"].astype(np.float32), 0, 1) ** (1 / 2.2)
    im = (f[..., ::-1] * 255).astype(np.uint8).copy()
    H = np.asarray(z["H_frame_plate"])
    Hinv = np.linalg.inv(H)
    def to_px(pts):
        p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Hinv.T
        return (p[:, :2] / p[:, 2:]).astype(np.int32)
    test = to_px([[0, 0]])[0]
    if not (0 <= test[0] < im.shape[1] and 0 <= test[1] < im.shape[0]):
        Hinv = H                                           # the stored matrix is plate -> frame
    for d in labels:
        if not d.get("label", True):
            continue
        col = COL.get(d["cls"], (255, 255, 255))
        for poly in d.get("polys_mm") or []:
            cv2.polylines(im, [to_px(poly)], True, col, 3, cv2.LINE_AA)
        x0, y0, x1, y1 = d["bbox_mm"]
        q = to_px([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        cv2.polylines(im, [q], True, col, 2, cv2.LINE_AA)
        tx, ty = q[:, 0].min(), max(q[:, 1].min() - 10, 20)
        cv2.putText(im, f"{d['id']} {d['cls']}", (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(im, f"{d['id']} {d['cls']}", (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2, cv2.LINE_AA)
    im = cv2.resize(im, (900, 900), interpolation=cv2.INTER_AREA)
    kinds = ", ".join(sorted({d["cls"] for d in labels if d.get("label", True)}))
    cv2.imwrite(str(OUT / f"ground_truth_{name}.jpg"),
                np.vstack([title(900, f"{name[:7]} ground truth from the twin: {kinds}", 44), im]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(name, [(d["id"], d["cls"], d.get("label")) for d in labels])
print("\n".join(sorted(p.name for p in OUT.iterdir())))
