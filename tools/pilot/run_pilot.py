"""Render + post the pilot matrix at full resolution, 64 spp; keep each scene's plan + EXR in <keep>
(for re-running post after gate/processing changes without re-rendering), delete the rest of the work dir.
  python run_pilot.py <out> <keep> <scene-glob-prefix ...> [--bursts] [--overview]
"""
import json, shutil, sys, time
from pathlib import Path
import yaml
from trinetra.twin import matrix as MX, pipeline as PL

ROOT = Path(__file__).resolve().parents[2]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
out, keep, prefixes = Path(args[0]), Path(args[1]), args[2:]
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
paths = [p for p in MX.plan_paths(out) if any(p.stem.startswith(x) for x in prefixes)]
keep.mkdir(parents=True, exist_ok=True)
t0 = time.time()
B = 7
for i in range(0, len(paths), B):
    tb = time.time()
    s = PL.run_plans(cfg, paths[i:i + B], out, batch=B, samples=64, keep_work=True,
                     save_bursts=True if "--bursts" in flags else None, overview="--overview" in flags)
    for p in paths[i:i + B]:
        w = out / "work" / p.stem
        if not w.exists():
            continue
        d = keep / p.stem
        d.mkdir(exist_ok=True)
        for f in ("plan.json", "render.exr", "blender_cam.json", "render_meta.json", "blender.log"):
            if (w / f).exists():
                shutil.copy2(w / f, d / f)
        shutil.rmtree(w, ignore_errors=True)
    print(json.dumps({"batch": i // B, "scenes": [p.stem for p in paths[i:i + B]], "s": round(time.time() - tb, 1),
                      "total_s": round(time.time() - t0, 1), **{k: v for k, v in s.items() if k != "failed"},
                      "failed": s["failed"][:3]}), flush=True)
print("DONE", round(time.time() - t0, 1))
