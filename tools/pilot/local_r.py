"""Share of the plate where the ambient outshines the device's LEDs locally (noise-free, from the
render passes, per variant's ambient multiplier): r_loc = AMB / (L + R) on 32 x 32 px blocks; prints
the share of plate blocks with r_loc > 4, 8, 16 and the global median ratio.
  local_r.py <matrix dir> <exr keep dir>"""
import json, sys
from pathlib import Path
import cv2, numpy as np, yaml
from trinetra.twin import matrix as MX, post as P

ROOT = Path(__file__).resolve().parents[2]
mx, keep = Path(sys.argv[1]), Path(sys.argv[2])
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
names = tuple((cfg.get("render") or {}).get("light_groups") or P.GROUPS)
post_cfg = cfg.get("post") or {}
for pp in MX.plan_paths(mx):
    s = pp.stem
    if s.endswith("h"):
        continue
    plan = json.loads((keep / s / "plan.json").read_text())
    g = P.read_light_groups(keep / s / "render.exr", names)
    geom = P.grid_geometry(plan, cfg.get("grid") or {})
    pm = P.plate_mask_image(geom, next(iter(g.values())).shape)
    lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722], np.float32)   # noqa: E731
    for k in range(2):
        mult = P.amb_multiplier(np.random.default_rng([int(plan["seed"]), k]), plan.get("ambient"), k, post_cfg)
        a, led = lum(g["AMB"]) * mult, lum(g["L"]) + lum(g["R"])
        h, w = a.shape
        bs = 32
        cut = lambda x: cv2.resize(x[: h // bs * bs, : w // bs * bs], (w // bs, h // bs), interpolation=cv2.INTER_AREA)  # noqa: E731
        mb = cut(pm.astype(np.float32)) > 0.999
        rl = cut(a) / np.maximum(cut(led), 1e-9)
        print(s, k, "median r %.3f" % float(np.median(a[pm]) / max(np.median(led[pm]), 1e-9)),
              " ".join(f"r>{t}:{(rl[mb] > t).mean():.3f}" for t in (4, 8, 16)), flush=True)
