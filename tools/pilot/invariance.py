"""Invariance numbers per (row, arm) that separate noise from lighting residue.

For every accepted capture, against the same plate's C0 grid of the same device and arm (same k,
as eval_lighting's mad_c0), on the plate (RGB mean, 8-bit DN):
  mad      mean |grid - grid_C0| over pixels and channels    (= eval_lighting's mad_c0, |delta|)
  lp       mean |lowpass(grid - grid_C0)| (channel mean), a normalised 15 x 15 px box (~1.1 mm): what is left
           after averaging away pixel noise = the systematic (lighting) part of the change
  bias     mean (grid - grid_C0) (signed: a brightness offset)
  p99      99th percentile of |lowpass(grid - grid_C0)| (a local residue, e.g. a shadow edge)
Noise floor per (device, arm), from the two dark-room captures C0 v0 and v1 of each plate (an
independent noise and shake draw of the same scene): mad_floor, lp_floor, and sigma = the per-pixel,
per-channel noise std = std(v0 - v1) / sqrt(2) (after removing the 15 px low-pass: pixel noise only).
For independent Gaussian noise of std sigma in both images, E|a - b| = 2 sigma / sqrt(pi).
  invariance2.py <matrix dir> <eval json (lighting_matrix.json)> <out json>
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
import eval_lighting as EL
import eval_plates as EP

mx, evj, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rep = json.loads(evj.read_text())
caps = {}
for c in rep["captures"]:
    caps.setdefault((c["arm"], c["scene"], c["k"]), c)
K = 15
cache = {}


def side(c):
    return json.loads(Path(c["image"]).with_suffix(".json").read_text())


def img(c):
    p = c["image"]
    if p not in cache:
        if len(cache) > 40:
            cache.clear()
        cache[p] = EP.load_rgb(Path(p)).astype(np.float32)
    return cache[p]


def stats(a, b, mask):
    d3 = a - b
    d = d3.mean(axis=2)
    m = mask.astype(np.float32)
    den = np.maximum(cv2.boxFilter(m, -1, (K, K), normalize=False), 1e-6)
    lp = cv2.boxFilter(d * m, -1, (K, K), normalize=False) / den
    inner = cv2.erode(mask.astype(np.uint8), np.ones((K, K), np.uint8)).astype(bool)
    # per-channel pixel noise: the channel difference minus its own low-pass
    hp_std = float(np.mean([(d3[..., c] - cv2.boxFilter(d3[..., c] * m, -1, (K, K), normalize=False) / den)[inner].std()
                            for c in range(3)]))
    return {"mad": float(np.abs(d3).mean(axis=2)[mask].mean()), "lp": float(np.abs(lp[inner]).mean()),
            "bias": float(d[mask].mean()), "p99": float(np.percentile(np.abs(lp[inner]), 99)), "hp_std": hp_std}


rows = defaultdict(lambda: defaultdict(list))
floor = defaultdict(lambda: defaultdict(list))
for (arm, scene, k), c in caps.items():
    if c["gate"] == "RETAKE":
        continue
    ref_scene = f"{c['plate']}_C0{c['device_mode'][0]}"
    sc = side(c)
    if scene == ref_scene:
        if k == 0:
            r = caps.get((arm, ref_scene, 1))
            if r is not None and r["gate"] != "RETAKE":
                sr = side(r)
                b = img(r)
                a = EL.align_to(img(c), sc, sr, b.shape[:2])
                s = stats(a, b, EL.plate_mask(sr, b.shape[:2]))
                for key, v in s.items():
                    floor[(c["device_mode"], arm)][key].append(v)
                floor[(c["device_mode"], arm)]["sigma"].append(s["hp_std"] / math.sqrt(2))
        continue
    r = caps.get((arm, ref_scene, k))
    if r is None or r["gate"] == "RETAKE":
        continue
    sr = side(r)
    b = img(r)
    a = EL.align_to(img(c), sc, sr, b.shape[:2])
    for key, v in stats(a, b, EL.plate_mask(sr, b.shape[:2])).items():
        rows[(c["tag"], arm)][key].append(v)
out = {"rows": {}, "floor": {}}
order = sorted(rows, key=lambda t: (t[0][-1], t[0], t[1]))
for (dev, arm), d in sorted(floor.items()):
    out["floor"][f"{dev}/{arm}"] = {key: round(float(np.mean(v)), 3) for key, v in d.items()} | {"n": len(d["mad"])}
    f = out["floor"][f"{dev}/{arm}"]
    f["expected_mad_from_sigma"] = round(2 * f["sigma"] / math.sqrt(math.pi), 3)
    print("FLOOR", dev, arm, f)
for key in order:
    d = rows[key]
    out["rows"][f"{key[0]}/{key[1]}"] = {k2: round(float(np.mean(v)), 3) for k2, v in d.items()} | {
        "lp_max": round(float(np.max(d["lp"])), 3), "mad_max": round(float(np.max(d["mad"])), 3), "n": len(d["mad"])}
    print(key[0], key[1], out["rows"][f"{key[0]}/{key[1]}"])
outp.write_text(json.dumps(out, indent=1))
