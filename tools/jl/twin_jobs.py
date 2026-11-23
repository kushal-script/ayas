"""Render / post driver for long twin jobs on the JarvisLabs GPU instance (instance copy:
/home/ayas/tools/jl/twin_jobs.py). It only calls the project's own building blocks
(textures.write_maps, render.prepare, render.render_batch, post.process_scene) and never
changes src/, twin/ or configs/.

Why not `ayas-twin run|matrix --keep-work`: pipeline.run_plans marks a scene done by its
sidecar, i.e. only after post. With post deferred (render now, post later with the final code)
a restart would re-render every scene, and render.render_batch deletes the old render.exr first.
So this driver tracks the two stages apart:

  render  done = out/work/<scene>/render_meta.json has exr_channels (written by render._finish
          after the ambient merge), no error, render.exr exists and no render_amb.exr is left.
          The texture maps (~20-40 MB per scene) are deleted after the render (post never reads
          them); plan.json, blender_cam.json, render.exr, render_meta.json, blender.log and
          overview.png stay. Failed scenes are retried on the next run.
  post    done = <post_out>/sidecar/<scene>.json (--force redoes it). Post a second time into a
          NEW --post-out: re-posting into the old dir leaves stale tiles behind.

Never run `ayas-twin run|matrix` or pipeline.run_plans on a render-only out dir: they see
no sidecar and re-render.

  python tools/jl/twin_jobs.py render --plans OUT/matrix_manifest.json --out OUT --device OPTIX --procs 2
  python tools/jl/twin_jobs.py render ... --post            # post each batch right after its render
  python tools/jl/twin_jobs.py post --out OUT [--post-out OUT2] [--plans ...] --procs 16
  python tools/jl/twin_jobs.py status --out OUT [--post-out OUT2] [--plans ...]

--plans takes plan dirs, plan files, globs or manifests (matrix_manifest.json /
pairs_manifest.json; --first-plates N keeps their first N plates). --jepa builds the config
the way training/jepa/pairs.py does (jepa_config of the current configs/twin.yaml).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("AYAS_REPO") or Path(__file__).resolve().parents[2])
KEEP_FILES = {"plan.json", "blender_cam.json", "render.exr", "render_clean.exr", "render_meta.json",
              "blender.log", "overview.png"}


# ----------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def jappend(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def hms(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def load_pairs_module():
    spec = importlib.util.spec_from_file_location("_jepa_pairs", REPO / "training" / "jepa" / "pairs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cfg(a: argparse.Namespace) -> dict[str, Any]:
    import yaml

    path = Path(a.config)
    path = path if path.is_absolute() else REPO / path
    cfg = yaml.safe_load(path.read_text())
    cap = str(cfg.get("capture_config") or "configs/capture.yaml")
    if not cap.startswith("/"):
        cfg["capture_config"] = str(REPO / cap)
    if a.jepa:
        cfg = load_pairs_module().jepa_config(cfg, a.jepa_arms, a.jepa_variants, a.jepa_grid_format)
    return cfg


def plan_list(specs: list[str], first_plates: int | None = None) -> list[Path]:
    out: list[Path] = []
    for s in specs:
        p = Path(s)
        if p.is_file() and p.suffix == ".json":
            d = json.loads(p.read_text())
            if isinstance(d, dict) and "plates" in d and "scene_id" not in d:      # a manifest
                plates = d["plates"][:first_plates] if first_plates else d["plates"]
                out += [p.parent / r["plan"] for pl in plates for r in pl["rows"]]
            else:
                out.append(p)
        elif p.is_dir():
            out += sorted(p.glob("*.json"))
        else:
            out += sorted(Path(x) for x in glob.glob(s))
    seen, uniq = set(), []
    for p in out:
        if p.stem not in seen:
            seen.add(p.stem)
            uniq.append(p)
    return uniq


def rendered(work: Path, want_overview: bool = False) -> bool:
    try:
        m = json.loads((work / "render_meta.json").read_text())
    except (OSError, ValueError):
        return False
    ok = (bool(m.get("exr_channels")) and not m.get("error") and (work / "render.exr").is_file()
          and not (work / "render_amb.exr").exists())
    return ok and (not want_overview or (work / "overview.png").is_file())


def prune(work: Path) -> None:
    """Delete everything a later post does not need (the texture maps)."""
    for f in work.iterdir():
        if f.is_file() and f.name not in KEEP_FILES:
            f.unlink(missing_ok=True)


def copy_overview(work: Path, post_out: Path) -> None:
    if (work / "overview.png").exists():
        (post_out / "overview").mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / "overview.png", post_out / "overview" / f"{work.name}.png")


# ----------------------------------------------------------------------------- workers

def _post_one(work: str, post_out: str, cfg: dict, save_bursts: bool | None) -> dict:
    from ayas.twin import post

    t0 = time.time()
    try:
        sc = post.process_scene(Path(work), Path(post_out), cfg, save_bursts=save_bursts)
    except Exception as e:  # noqa: BLE001  (one bad scene must not stop a long run)
        return {"scene": Path(work).name, "ok": False, "error": f"post: {type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-2000:], "post_s": round(time.time() - t0, 2)}
    copy_overview(Path(work), Path(post_out))
    return {"scene": Path(work).name, "ok": True, "post_s": round(time.time() - t0, 2),
            "gates": [v.get("gate") for v in sc.get("variants", [])]}


def _render_batch(plan_paths: list[str], out: str, cfg: dict, rkw: dict, post_out: str | None,
                  save_bursts: bool | None, keep_textures: bool) -> dict:
    from ayas.twin import render, textures

    out_p = Path(out)
    t0 = time.time()
    works = []
    for p in plan_paths:
        plan = json.loads(Path(p).read_text())
        work = out_p / "work" / plan["scene_id"]
        work.mkdir(parents=True, exist_ok=True)
        (work / "plan.json").write_text(json.dumps(plan))
        textures.write_maps(plan, work)
        render.prepare(work, plan, cfg)
        works.append(work)
    t1 = time.time()
    metas = render.render_batch(works, cfg, **rkw)
    t2 = time.time()
    res = []
    for w, m in zip(works, metas):
        r = {"scene": w.name, "ok": not m.get("error"), "error": m.get("error"), "device": m.get("device"),
             "seconds_render": m.get("seconds_render"), "seconds": m.get("seconds"), "exr_mb": m.get("exr_mb"),
             "warnings": (m.get("warnings") or [])[:3]}
        if not keep_textures:
            prune(w)
        if r["ok"] and post_out:
            pr = _post_one(str(w), post_out, cfg, save_bursts)
            r.update(post_ok=pr["ok"], post_s=pr["post_s"], post_error=pr.get("error"), gates=pr.get("gates"))
        res.append(r)
    return {"tex_s": round(t1 - t0, 1), "render_s": round(t2 - t1, 1), "post_s": round(time.time() - t2, 1),
            "scenes": res}


# ----------------------------------------------------------------------------- commands

def cmd_render(a: argparse.Namespace) -> int:
    from ayas.twin import pipeline

    cfg = load_cfg(a)
    out = Path(a.out)
    (out / "work").mkdir(parents=True, exist_ok=True)
    plans = plan_list(a.plans, a.first_plates)
    if a.limit:
        plans = plans[: a.limit]
    todo = [p for p in plans if not rendered(out / "work" / p.stem, a.overview)]
    batch = a.batch or int(cfg["render"]["batch_size"])
    dev = a.device.upper()
    rkw = pipeline._render_kwargs(dev, a.samples, a.res_scale, a.overview)
    post_out = str(Path(a.post_out or a.out)) if a.post else None
    batches = [todo[i: i + batch] for i in range(0, len(todo), batch)]
    log(f"[render] {len(plans)} plans, {len(plans) - len(todo)} already rendered, {len(todo)} to render in "
        f"{len(batches)} batches of {batch}, {a.procs} Blender process(es), device {dev}, "
        f"post {'inline -> ' + post_out if post_out else 'deferred'}, {free_gb(out):.0f} GB free")
    done = failed = 0
    t0 = time.time()
    stop = None
    jlog, flog = out / "jobs_log.jsonl", out / "render_failures.jsonl"
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.procs, mp_context=ctx) as ex:
        pending: dict = {}
        it = iter(enumerate(batches))
        bad_batches = 0

        def submit_next() -> bool:
            nonlocal stop
            if stop:
                return False
            if free_gb(out) < a.min_free_gb:
                stop = f"less than {a.min_free_gb} GB free"
                return False
            try:
                i, b = next(it)
            except StopIteration:
                return False
            fut = ex.submit(_render_batch, [str(p) for p in b], str(out), cfg, rkw, post_out,
                            True if a.save_bursts else None, a.keep_textures)
            pending[fut] = (i, len(b), time.time())
            return True

        for _ in range(a.procs):
            submit_next()
        while pending:
            fin, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in fin:
                i, n, tb = pending.pop(fut)
                try:
                    r = fut.result()
                except Exception as e:  # noqa: BLE001
                    r = {"scenes": [], "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
                ok = [s for s in r["scenes"] if s["ok"]]
                bad = [s for s in r["scenes"] if not s["ok"]]
                done += len(ok)
                failed += len(bad) + (n if r.get("error") else 0)
                for s in bad:
                    jappend(flog, {"t": time.time(), **s})
                if r.get("error"):
                    jappend(flog, {"t": time.time(), "batch": i, "error": r["error"], "trace": r.get("trace")})
                devs = {str(s.get("device")) for s in r["scenes"]}
                if ok and not all(str(s.get("device", "")).startswith(dev) for s in ok) and dev != "CPU":
                    stop = f"Blender did not render on {dev}: devices {sorted(devs)}"
                bad_batches = bad_batches + 1 if not ok else 0
                if bad_batches >= 2:
                    stop = f"two batches in a row with no good scene; last error: {(bad or [{}])[-1].get('error') or r.get('error')}"
                el = time.time() - t0
                rate = el / max(1, done)
                post_s = [s["post_s"] for s in ok if s.get("post_s") is not None]
                entry = {"t": time.time(), "batch": i, "scenes": n, "ok": len(ok), "failed": n - len(ok),
                         "tex_s": r.get("tex_s"), "render_s": r.get("render_s"), "post_s": r.get("post_s"),
                         "wall_s": round(time.time() - tb, 1), "done": done, "todo": len(todo),
                         "elapsed_s": round(el, 1), "free_gb": round(free_gb(out), 1), "devices": sorted(devs),
                         "post_gates": [s.get("gates") for s in ok] if post_out else None}
                jappend(jlog, entry)
                log(f"[render] batch {i + 1}/{len(batches)}: {len(ok)}/{n} ok, render {r.get('render_s')} s "
                    f"({(r.get('render_s') or 0) / max(1, n):.1f} s/scene in-process), "
                    f"{'post %.1f s/scene, ' % (sum(post_s) / len(post_s)) if post_s else ''}"
                    f"{done}/{len(todo)} done, {rate:.1f} s/scene wall, elapsed {hms(el)}, "
                    f"ETA {hms(rate * (len(todo) - done - failed))}, {entry['free_gb']} GB free"
                    + (f" | FAILED: {[s['scene'] + ': ' + str(s.get('error')) for s in bad][:3]}" if bad else "")
                    + (f" | BATCH ERROR {r['error']}" if r.get("error") else ""))
                submit_next()
    if post_out:
        from ayas.twin import post as post_mod
        post_mod.write_data_yaml(post_out, cfg["classes"])
    summary = {"plans": len(plans), "rendered_now": done, "failed": failed,
               "already": len(plans) - len(todo), "stopped": stop, "seconds": round(time.time() - t0, 1)}
    log("[render] summary " + json.dumps(summary))
    return 1 if stop or failed else 0


def cmd_post(a: argparse.Namespace) -> int:
    from ayas.twin import post as post_mod

    cfg = load_cfg(a)
    out = Path(a.out)
    post_out = Path(a.post_out or a.out)
    post_out.mkdir(parents=True, exist_ok=True)
    if a.plans:
        names = [p.stem for p in plan_list(a.plans, a.first_plates)]
        works = [out / "work" / n for n in names]
    else:
        works = sorted(p.parent for p in (out / "work").glob("*/plan.json"))
    ready = [w for w in works if rendered(w)]
    todo = [w for w in ready if a.force or not (post_out / "sidecar" / f"{w.name}.json").exists()]
    log(f"[post] {len(works)} scenes, {len(works) - len(ready)} not rendered (skipped), "
        f"{len(ready) - len(todo)} already posted, {len(todo)} to post -> {post_out} with {a.procs} processes")
    nthreads = str(max(1, (os.cpu_count() or 2) // max(1, a.procs)))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, nthreads)
    t0, done, failed = time.time(), 0, 0
    flog = post_out / "post_failures.jsonl"
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.procs, mp_context=ctx) as ex:
        futs = {ex.submit(_post_one, str(w), str(post_out), cfg, True if a.save_bursts else None): w for w in todo}
        last = 0.0
        for fut in futs:                       # in submission order; progress every 30 s
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                r = {"scene": futs[fut].name, "ok": False, "error": f"{type(e).__name__}: {e}"}
            if r["ok"]:
                done += 1
            else:
                failed += 1
                jappend(flog, {"t": time.time(), **r})
                log(f"[post] FAILED {r['scene']}: {r.get('error')}")
                if failed > a.max_failures:
                    log(f"[post] more than {a.max_failures} failures: stopping")
                    ex.shutdown(cancel_futures=True)
                    break
            if time.time() - last > 30 or done + failed == len(todo):
                last = time.time()
                el = time.time() - t0
                log(f"[post] {done + failed}/{len(todo)} ({failed} failed), {el / max(1, done + failed):.2f} s/scene wall, "
                    f"elapsed {hms(el)}, ETA {hms(el / max(1, done + failed) * (len(todo) - done - failed))}, "
                    f"{free_gb(post_out):.0f} GB free")
    post_mod.write_data_yaml(post_out, cfg["classes"])
    summary = {"posted": done, "failed": failed, "seconds": round(time.time() - t0, 1), "post_out": str(post_out)}
    log("[post] summary " + json.dumps(summary))
    return 1 if failed else 0


def cmd_status(a: argparse.Namespace) -> int:
    out = Path(a.out)
    post_out = Path(a.post_out or a.out)
    if a.plans:
        names = [p.stem for p in plan_list(a.plans, a.first_plates)]
    else:
        names = sorted(p.stem for p in (out / "plans").glob("*.json"))
    r = sum(rendered(out / "work" / n) for n in names)
    s = sum((post_out / "sidecar" / f"{n}.json").exists() for n in names)
    du = lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9 if p.exists() else 0.0  # noqa: E731
    print(json.dumps({"plans": len(names), "rendered": r, "posted": s, "work_gb": round(du(out / "work"), 2),
                      "free_gb": round(free_gb(out), 1)}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "post", "status"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="job dir: plans/, work/<scene>/ (EXRs), and post outputs by default")
        p.add_argument("--post-out", help="post outputs here instead of --out (re-post with final code)")
        p.add_argument("--plans", nargs="*", default=None, help="plan dirs/files/globs/manifests (render: required)")
        p.add_argument("--first-plates", type=int, help="manifests: only their first N plates")
        p.add_argument("--config", default="configs/twin.yaml")
        p.add_argument("--jepa", action="store_true", help="config = training/jepa/pairs.py jepa_config(config)")
        p.add_argument("--jepa-arms", nargs="+", default=["burst"])
        p.add_argument("--jepa-variants", type=int, default=2)
        p.add_argument("--jepa-grid-format", default="jpg")
        p.add_argument("--save-bursts", action="store_true")
        p.add_argument("--procs", type=int, default=3 if name == "render" else 16)
        if name == "render":
            p.add_argument("--device", default="OPTIX", help="OPTIX | CUDA | CPU (the config's METAL is Mac only)")
            p.add_argument("--samples", type=int)
            p.add_argument("--res-scale", type=float)
            p.add_argument("--batch", type=int, help="scenes per Blender process (default render.batch_size)")
            p.add_argument("--overview", action="store_true")
            p.add_argument("--post", action="store_true", help="post each scene right after its render")
            p.add_argument("--keep-textures", action="store_true")
            p.add_argument("--min-free-gb", type=float, default=10.0)
            p.add_argument("--limit", type=int, help="render at most N of the plans (benchmarks)")
        if name == "post":
            p.add_argument("--force", action="store_true", help="re-post scenes that have a sidecar")
            p.add_argument("--max-failures", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "render" and not a.plans:
        ap.error("render needs --plans")
    return {"render": cmd_render, "post": cmd_post, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
