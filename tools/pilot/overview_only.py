"""Re-render only the overview pictures of some matrix scenes (after the add_pod_glow ring fix):
a cheap main pass (res 0.25, 4 spp: thrown away) plus the normal overview camera; copies
overview.png to <out>/overview/<scene>.png and deletes the work dirs.
  overview_only.py <matrix out> <scratch work root> <scene> [<scene> ...]
"""
import json, shutil, sys
from pathlib import Path
import yaml
from ayas.twin import render, textures

ROOT = Path(__file__).resolve().parents[2]
out, root, scenes = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
works = []
for s in scenes:
    plan = json.loads((out / "plans" / f"{s}.json").read_text())
    w = root / s
    w.mkdir(parents=True, exist_ok=True)
    (w / "plan.json").write_text(json.dumps(plan))
    textures.write_maps(plan, w)
    render.prepare(w, plan, cfg)
    works.append(w)
metas = render.render_batch(works, cfg, samples=4, res_scale=0.25, overview=True)
for w, m in zip(works, metas):
    ok = (w / "overview.png").exists()
    print(w.name, "error" if m.get("error") else "ok", "overview" if ok else "NO overview", m.get("seconds_overview"), m.get("warnings"))
    if ok:
        shutil.copy2(w / "overview.png", out / "overview" / f"{w.name}.png")
    shutil.rmtree(w, ignore_errors=True)
