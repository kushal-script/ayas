"""Report figures for reports/pilot_lighting.md from a rendered + evaluated pilot matrix.
  figures.py <matrix dir> <eval json> <figures dir> <plate for naive-vs-burst> [<plate> ...]
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np

mx, evj, figs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
plates = sys.argv[4:]
figs.mkdir(parents=True, exist_ok=True)
rep = json.loads(evj.read_text())
caps = {}
for c in rep["captures"]:
    caps.setdefault((c["arm"], c["scene"], c["k"]), c)
F = cv2.FONT_HERSHEY_SIMPLEX
CONDS = ["C0_dark", "C1_office", "C2_warm_bulb", "C3_window", "C4_side_lamp", "C5_reflected_lamp", "C6_sun"]
SHORT = {"C0_dark": "C0 dark", "C1_office": "C1 office", "C2_warm_bulb": "C2 warm bulb", "C3_window": "C3 window",
         "C4_side_lamp": "C4 side lamp", "C5_reflected_lamp": "C5 lamp reflection", "C6_sun": "C6 sun"}


def cell(path: Path, cap: dict | None, size: int, crop=None) -> np.ndarray:
    img = cv2.imread(str(path))
    if crop is not None:
        x0, y0, x1, y1 = crop
        img = img[y0:y1, x0:x1]
    s = size / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.full((size + 38, size, 3), 24, np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    if cap is not None:
        if cap["gate"] == "RETAKE":
            out[:size] = (out[:size] * 0.35).astype(np.uint8)
            reason = ",".join(cap.get("gate_reasons") or [])[:34]
            cv2.putText(out, "RETAKE (refused)", (8, size // 2 - 6), F, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(out, reason, (8, size // 2 + 18), F, 0.45, (200, 200, 255), 1, cv2.LINE_AA)
        lux = cap.get("lux")
        top = f"{cap['tag']} {'' if lux is None else f'{lux / 1000:.1f} klx'} gate {cap['gate']}"
        ok = cap["pred_verdict"] in ("RETAKE", cap["gt_verdict"])
        bot = "refused" if cap["pred_verdict"] == "RETAKE" else f"{cap['pred_verdict']} (truth {cap['gt_verdict']})"
        cv2.putText(out, top, (4, size + 14), F, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(out, bot, (4, size + 31), F, 0.4, (90, 220, 90) if ok else (60, 60, 255), 1, cv2.LINE_AA)
    return out


def label_col(text_rows, h, w=120):
    col = np.full((h, w, 3), 24, np.uint8)
    for i, t in enumerate(text_rows):
        cv2.putText(col, t, (6, h // 2 - 10 + 18 * i), F, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return col


def grid_path(arm, scene, k=0):
    return mx / "grids" / arm / "test" / f"{scene}_v{k}.png"


# 1. naive vs burst per plate: rows = (open naive, open burst, hood naive, hood burst), cols = conditions
size = 260
for plate in plates:
    rows = []
    for mode, arm, name in (("open", "naive", ["open", "old method"]), ("open", "burst", ["open", "new: burst"]),
                            ("hood", "naive", ["hood", "old method"]), ("hood", "burst", ["hood", "new: burst"])):
        cells = []
        for cond in CONDS:
            scene = f"{plate}_{cond.split('_')[0]}{mode[0]}"
            cells.append(cell(grid_path(arm, scene), caps.get((arm, scene, 0)), size))
        rows.append(np.hstack([label_col(name, size + 38)] + cells))
    head = np.full((28, rows[0].shape[1], 3), 24, np.uint8)
    for j, cond in enumerate(CONDS):
        cv2.putText(head, SHORT[cond], (120 + j * size + 6, 20), F, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(figs / f"pilot_naive_vs_burst_{plate}.png"), np.vstack([head] + rows))

# 2. overview renders of plate 1 (first plate given): 2 rows (open, hood) x 7 conditions
p1 = plates[0]
ov = []
for mode in ("open", "hood"):
    r = []
    for cond in CONDS:
        p = mx / "overview" / f"{p1}_{cond.split('_')[0]}{mode[0]}.png"
        img = cv2.imread(str(p)) if p.exists() else np.zeros((720, 1280, 3), np.uint8)
        img = cv2.resize(img, (384, 216), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{SHORT[cond]}, {mode}", (6, 18), F, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        r.append(img)
    ov.append(np.hstack(r))
cv2.imwrite(str(figs / f"pilot_overview_{p1}.jpg"), np.vstack(ov), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("ok")
