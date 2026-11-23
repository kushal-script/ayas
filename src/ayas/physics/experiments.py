"""Experiment drivers behind reports/physics_pinn.md (python -m ayas.physics.experiments ...).

Sub-commands (all write under --out, a scratch directory; nothing in the repo is edited):
  coupons     build depth-coupon plans (twin_eval.coupon_plan) and render them with Blender
              (one light group per ring LED; dark room, hood).
  hood0       re-render a pilot hood scene with the hood albedo set to 0 (validation control:
              the hood's own inter-reflection is the only light path the direct model leaves out).
  validate    forward model vs Blender L/R passes, per material, flat areas and defects.
  depth       depth estimation runs (PINN and baselines) on pilot scenes and coupons.

Renders use ayas.twin.render (prepare, render_batch) unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import optics as O
from . import twin_eval as TE


# ----------------------------------------------------------------------------- renders

def render_coupons(out: Path, materials: Sequence[str], samples: int = 64) -> list[dict]:
    from ayas.twin import render
    cfg = TE.twin_cfg()
    works = []
    for m in materials:
        plan = TE.coupon_plan(m, cfg=cfg)
        work = out / plan["scene_id"]
        TE.prepare_coupon(work, plan, cfg)
        works.append(work)
    metas = []
    for w in works:                                   # one scene per Blender process: a failure stays local
        t0 = time.time()
        metas += render.render_batch([w], cfg, samples=samples)
        print(f"rendered {w.name} in {time.time() - t0:.0f} s: {metas[-1].get('error')}", flush=True)
    return metas


def render_hood0(out: Path, pilot_scene_dir: Path, samples: int = 64) -> dict:
    """Copy of a pilot hood scene with extras.hood.albedo = 0 (black hole hood)."""
    from ayas.twin import render, textures
    cfg = TE.twin_cfg()
    plan = json.loads((pilot_scene_dir / "plan.json").read_text())
    plan = copy.deepcopy(plan)
    plan["scene_id"] = plan["scene_id"] + "_hood0"
    plan["extras"]["hood"]["albedo"] = 0.0
    work = out / plan["scene_id"]
    work.mkdir(parents=True, exist_ok=True)
    render.prepare(work, plan, cfg)
    textures.write_maps(plan, work)
    t0 = time.time()
    meta = render.render_batch([work], cfg, samples=samples)[0]
    print(f"rendered {work.name} in {time.time() - t0:.0f} s: {meta.get('error')}", flush=True)
    return meta


# ----------------------------------------------------------------------------- depth runs

def _rules_defect(t: dict, plan: dict) -> dict:
    """A record-3.0 style defect (rules input) for a depth target, from its plan footprint measures."""
    m = dict(t.get("measure") or {})
    size = {}
    if t["cls"] == "pit":
        size = {"diameter_mm": m.get("diameter_mm"), "count": m.get("count", 1)}
        if m.get("pit_d_mm") is not None:
            size["pit_d_mm"] = m["pit_d_mm"]
        if m.get("area_mm2") is not None:
            size["area_mm2"] = m["area_mm2"]
    elif t["cls"] == "dent":
        size = {"diameter_mm": m.get("diameter_mm")}
    else:
        size = {"length_mm": m.get("length_mm"), "width_mm": m.get("width_mm")}
    u0, v0, u1, v1 = t["box_mm"]
    near = (t.get("near") or {}).get("near_edge_mm")
    return {"id": t["id"], "type": t["cls"], "confidence": 0.95, "size": {k: v for k, v in size.items() if v is not None},
            "location_mm": {"x": (u0 + u1) / 2, "y": (v0 + v1) / 2}, "near_edge_mm": near}


def _run_methods(frames, meta, t, plan, methods, threads, n_seeds, noise_var=None, cfg=None) -> dict:
    from . import baselines as B
    from . import depth_pinn as DP
    mat = plan["plate"]["material"]["kind"]
    th = float(plan["plate"]["thickness_mm"])
    kind = t["cls"]
    c = dict(cfg or {})
    if kind == "pit":
        c.setdefault("max_px", 80)              # pits are 2-20 px: keep the full 0.08 mm grid
    out = {}
    for m in methods:
        try:
            if m == "pinn":
                r = DP.estimate_depth(frames, meta, t["box_mm"], mat, th, threads=threads, n_seeds=n_seeds,
                                      noise_var=noise_var, cfg=c, kind=kind)
            elif m == "grid":
                r = DP.estimate_depth(frames, meta, t["box_mm"], mat, th, method="grid", threads=threads,
                                      noise_var=noise_var, cfg=c, kind=kind)
            elif m == "classical":
                r = B.classical_depth(frames, meta, t["box_mm"], mat, th, threads=threads, noise_var=noise_var, cfg=c)
            else:
                continue
            out[m] = r.to_json()
            out[m]["material_fit"] = r.material_fit
        except Exception as e:                                       # one failure must not stop a sweep
            out[m] = {"error": f"{type(e).__name__}: {e}"}
    return out


def pilot_scene_job(args: tuple) -> list[dict]:
    """Depth of every labelled dent, pit (and a scratch window) in one pilot scene through the
    real burst path (noise, shake, ambient, gate)."""
    scene_dir, methods, threads, n_seeds, seed = args
    import cv2
    import torch
    torch.set_num_threads(int(threads))
    cv2.setNumThreads(int(threads))
    from ayas.twin import textures
    from . import depth_pinn as DP
    scene_dir = Path(scene_dir)
    plan = json.loads((scene_dir / "plan.json").read_text())
    geom = TE.grid_geom(plan)
    cap = TE.capture_cfg()
    meta = TE.meta_from_plan(plan, geom, cap)
    groups = TE.read_groups(scene_dir / "render.exr", ("L", "R", "AMB"))
    rng = np.random.default_rng([int(plan.get("seed", 0)), int(seed)])
    t0 = time.time()
    frames, info = TE.grid_frames_burst(groups, plan, geom, rng, cap)
    var = info.pop("var")
    t_burst = time.time() - t0
    meta["grain_deg"] = DP.plate_grain_deg(frames["L"] + frames["R"], geom["H_grid_plate"], plan["plate"]["size_mm"])
    maps = textures.render_maps(plan)
    targets = TE.depth_targets(plan, maps, classes=("dent", "pit")) + TE.scratch_segments(plan, maps)
    rows = []
    ex = plan.get("extras") or {}
    for t in targets:
        row = {"scene": plan["scene_id"], "plate": plan["scene_id"].split("_")[0],
               "condition": (plan.get("ambient") or {}).get("condition"),
               "device_mode": (ex.get("device") or {}).get("mode"), "material": plan["plate"]["material"]["kind"],
               "thickness_mm": plan["plate"]["thickness_mm"], "gate": info["gate"], "gate_reasons": info["reasons"],
               "r_ambient": info["metrics"].get("r_ambient"), "id": t["id"], "cls": t["cls"], "kind": t["kind"],
               "gt_depth_mm": t["depth_mm"], "param_depth_mm": t.get("param_depth_mm"), "box_mm": t["box_mm"],
               "rules_defect": _rules_defect(t, plan), "t_burst_s": t_burst}
        if info["gate"] != "RETAKE":
            row.update(_run_methods(frames, meta, t, plan, methods, threads, n_seeds, var))
        rows.append(row)
    return rows


def coupon_job(args: tuple) -> list[dict]:
    """Depth of the 16 coupon defects of one material for each ring partition (2 = L/R, 3, 4, 6
    sectors), from the per-LED Blender passes plus dark-room sensor noise."""
    work, parts, methods, threads, n_seeds, seed = args
    import cv2
    import torch
    torch.set_num_threads(int(threads))
    cv2.setNumThreads(int(threads))
    from ayas import lightproc as LP
    from ayas.twin import textures
    from . import depth_pinn as DP
    work = Path(work)
    plan = json.loads((work / "plan.json").read_text())
    geom = TE.grid_geom(plan)
    cap = TE.capture_cfg()
    from ayas.twin import burst
    kplan = dict(plan, lights=[dict(L, group=L.get("orig_group", L["group"])) for L in plan["lights"]])
    k = burst.exposure_k(kplan, cap)["k"]                      # the device's fixed exposure (L + R on)
    leds = [L["group"] for L in plan["lights"] if str(L.get("group", "")).startswith("E")]
    passes = TE.read_groups(work / "render.exr", tuple(leds))
    from ayas.twin import post
    lum_led = {g: TE.lum(post.rectify(passes[g], geom)) for g in leds}
    del passes
    maps = textures.render_maps(plan)
    targets = TE.depth_targets(plan, maps, classes=("dent", "pit"))
    import os
    targets = targets[:int(os.environ.get("PHYS_MAX_TARGETS", "999"))]
    sensor = cap.get("sensor") or {}
    e_fs = LP.electrons_per_fs(sensor)
    rn = float(sensor.get("read_noise_e", 0.0))
    partitions = TE.led_partitions(plan, parts)
    rows = []
    for n, part in partitions.items():
        rng = np.random.default_rng([int(plan["seed"]), int(n), int(seed)])
        clean = {g: k * sum(lum_led[m] for m in mem) for g, mem in part.items()}
        var = {g: np.maximum(a, 0) / e_fs * 0.55 + 2 * (rn / e_fs) ** 2 * 0.55 for g, a in clean.items()}
        frames = {g: (a + rng.standard_normal(a.shape).astype(np.float32) * np.sqrt(var[g])).astype(np.float32)
                  for g, a in clean.items()}
        meta = TE.meta_from_plan(plan, geom, cap, lights=TE.lights_for_partition(plan, part), k=k)
        meta["grain_deg"] = DP.plate_grain_deg(sum(frames.values()), geom["H_grid_plate"], plan["plate"]["size_mm"])
        for t in targets:
            row = {"scene": plan["scene_id"], "material": plan["plate"]["material"]["kind"],
                   "thickness_mm": plan["plate"]["thickness_mm"], "n_groups": n, "groups": list(part),
                   "id": t["id"], "cls": t["cls"], "kind": t["kind"], "gt_depth_mm": t["depth_mm"],
                   "param_depth_mm": t.get("param_depth_mm"), "box_mm": t["box_mm"], "d_mm": (t.get("measure") or {}).get("diameter_mm"),
                   "rules_defect": _rules_defect(t, plan)}
            row.update(_run_methods(frames, meta, t, plan, methods, threads, n_seeds, var))
            rows.append(row)
            print(json.dumps({k_: row.get(k_) for k_ in ("material", "n_groups", "id", "cls", "gt_depth_mm")}),
                  {m: (row.get(m) or {}).get("max_depth_mm") for m in methods}, flush=True)
    return rows


def run_pool(jobs: list, fn, workers: int, out_jsonl: Path) -> None:
    import multiprocessing as mp
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool, open(out_jsonl, "a") as f:
        for rows in pool.imap_unordered(fn, jobs):
            for r in rows:
                f.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            f.flush()
            print(f"done {rows[0]['scene'] if rows else '?'} ({len(rows)} rows)", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ayas.physics.experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coupons")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--materials", nargs="+", default=list(TE.COUPON_SEEDS))
    c.add_argument("--samples", type=int, default=64)
    h = sub.add_parser("hood0")
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--scene", type=Path, required=True)
    h.add_argument("--samples", type=int, default=64)
    d = sub.add_parser("pilot")
    d.add_argument("--exr-root", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--scenes", nargs="*", default=None)
    d.add_argument("--methods", nargs="+", default=["pinn", "classical"])
    d.add_argument("--workers", type=int, default=6)
    d.add_argument("--threads", type=int, default=1)
    d.add_argument("--seeds", type=int, default=2)
    d.add_argument("--noise-seed", type=int, default=0)
    e = sub.add_parser("coupon-depth")
    e.add_argument("--renders", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--materials", nargs="*", default=None)
    e.add_argument("--parts", nargs="+", type=int, default=[2, 3, 4, 6])
    e.add_argument("--methods", nargs="+", default=["pinn", "classical"])
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--seeds", type=int, default=1)
    bn = sub.add_parser("bench")
    bn.add_argument("--work", type=Path, required=True)
    bn.add_argument("--out", type=Path, required=True)
    bn.add_argument("--ids", nargs="+", required=True)
    bn.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    bn.add_argument("--seeds", type=int, default=2)
    sm = sub.add_parser("summarise")
    sm.add_argument("--coupons", type=Path, required=True)
    sm.add_argument("--pilot", type=Path, required=True)
    sm.add_argument("--bench", type=Path, default=None)
    sm.add_argument("--out", type=Path, default=Path("reports/physics_pinn_summary.md"))
    sm.add_argument("--figs", type=Path, default=Path("reports/figures"))
    a = ap.parse_args(argv)
    if a.cmd == "bench":
        bench(a.work, a.out, a.ids, a.threads, a.seeds)
        return
    if a.cmd == "summarise":
        summarise(a.out, a.coupons, a.pilot, a.bench, a.figs)
        return
    if a.cmd == "pilot":
        dirs = sorted(p for p in a.exr_root.iterdir() if (p / "render.exr").is_file())
        if a.scenes:
            dirs = [p for p in dirs if any(p.name.startswith(s) for s in a.scenes)]
        jobs = [(str(p), a.methods, a.threads, a.seeds, a.noise_seed) for p in dirs]
        run_pool(jobs, pilot_scene_job, a.workers, a.out)
    elif a.cmd == "coupon-depth":
        dirs = sorted(p for p in a.renders.iterdir() if p.name.startswith("coupon_") and (p / "render.exr").is_file())
        if a.materials:
            dirs = [p for p in dirs if any(m in p.name for m in a.materials)]
        jobs = [(str(p), tuple(a.parts), a.methods, a.threads, a.seeds, 0) for p in dirs]
        run_pool(jobs, coupon_job, a.workers, a.out)
    elif a.cmd == "coupons":
        render_coupons(a.out, a.materials, a.samples)
    elif a.cmd == "hood0":
        render_hood0(a.out, a.scene, a.samples)



# ----------------------------------------------------------------------------- runtime

def bench(work: Path, out_json: Path, ids: Sequence[str], threads_list: Sequence[int] = (1, 4), n_seeds: int = 2) -> dict:
    """Wall time of estimate_depth / classical_depth on coupon crops (clean L/R frames), per thread count."""
    import os
    import platform
    import torch
    from ayas.twin import burst, post, textures
    from . import baselines as B
    from . import depth_pinn as DP
    plan = json.loads((work / "plan.json").read_text())
    geom = TE.grid_geom(plan)
    cap = TE.capture_cfg()
    kplan = dict(plan, lights=[dict(L, group=L.get("orig_group", L["group"])) for L in plan["lights"]])
    k = burst.exposure_k(kplan, cap)["k"]
    part = TE.led_partitions(plan, (2,))[2]
    leds = [m for mem in part.values() for m in mem]
    passes = TE.read_groups(work / "render.exr", tuple(leds))
    frames = {g: k * sum(TE.lum(post.rectify(passes[m], geom)) for m in mem) for g, mem in part.items()}
    del passes
    meta = TE.meta_from_plan(plan, geom, cap, lights=TE.lights_for_partition(plan, part), k=k)
    meta["grain_deg"] = DP.plate_grain_deg(frames["L"] + frames["R"], geom["H_grid_plate"], plan["plate"]["size_mm"])
    targets = {t["id"]: t for t in TE.depth_targets(plan, textures.render_maps(plan), classes=("dent", "pit"))}
    rows = []
    for th in threads_list:
        for i in ids:
            t = targets[i]
            c = {"max_px": 80} if t["cls"] == "pit" else {}
            crop = DP.make_crop(frames, meta, t["box_mm"], dict(DP.DEFAULTS, **c), ["L", "R"])
            t0 = time.perf_counter()
            r = DP.estimate_depth(frames, meta, t["box_mm"], plan["plate"]["material"]["kind"],
                                  plan["plate"]["thickness_mm"], threads=th, n_seeds=n_seeds, cfg=c)
            t1 = time.perf_counter()
            rc = B.classical_depth(frames, meta, t["box_mm"], plan["plate"]["material"]["kind"],
                                   plan["plate"]["thickness_mm"], threads=th, cfg=c)
            t2 = time.perf_counter()
            u0, v0, u1, v1 = t["box_mm"]
            grid_px = int(round(max(u1 - u0, v1 - v0) / 0.08 + 2 * max(0.8, 0.45 * max(u1 - u0, v1 - v0)) / 0.08))
            rows.append({"threads": th, "id": i, "cls": t["cls"], "kind": t["kind"], "defect_mm": round(max(u1 - u0, v1 - v0), 2),
                         "crop_grid_px": grid_px, "work_px": list(crop.obs["L"].shape), "block": crop.f,
                         "pinn_s": round(t1 - t0, 2), "pinn_seeds": n_seeds, "classical_s": round(t2 - t1, 2),
                         "pinn_depth": r.max_depth_mm, "gt": t["depth_mm"]})
            print(rows[-1], flush=True)
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    out = {"machine": platform.platform(), "processor": platform.processor(), "torch": torch.__version__,
           "loadavg_end": load, "rows": rows,
           "note": "Mac (Apple M3) wall time; other jobs shared the CPU (load average shown). Pi 5 is projected, not measured."}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- summaries (reports/physics_pinn.md)

def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()] if path.is_file() else []


def _err_stats(rows: list[dict], method: str, spec: dict | None = None) -> dict:
    from . import severity as SV
    e, rel, cov, conv, tl = [], [], [], [], []
    for r in rows:
        m = r.get(method) or {}
        if m.get("max_depth_mm") is None:
            continue
        g = float(r["gt_depth_mm"])
        d = float(m["max_depth_mm"])
        e.append(d - g)
        rel.append((d - g) / max(g, 1e-3))
        if m.get("depth_ci_mm") is not None:
            cov.append(abs(d - g) <= float(m["depth_ci_mm"]))
        conv.append(bool(m.get("converged")))
        if spec is not None:
            band = SV.depth_band(spec, float(r["thickness_mm"]))
            if band:
                tl.append(abs(d - g) / float(band["imperfection_mm"]))
    if not e:
        return {"n": 0}
    e, rel = np.array(e), np.array(rel)
    out = {"n": len(e), "mae_mm": float(np.mean(np.abs(e))), "median_abs_mm": float(np.median(np.abs(e))),
           "bias_mm": float(np.mean(e)), "median_rel": float(np.median(rel)), "mape": float(np.median(np.abs(rel))),
           "within_0p05mm": float(np.mean(np.abs(e) <= 0.05)), "within_25pct": float(np.mean(np.abs(rel) <= 0.25)),
           "converged": float(np.mean(conv)) if conv else None}
    if cov:
        out["ci_coverage"] = float(np.mean(cov))
    if tl:
        out["err_over_T1_median"] = float(np.median(tl))
    return out


def _dispositions(r: dict, method: str, spec: dict) -> tuple[str, str, str, dict]:
    """(GT-depth disposition, estimated-depth disposition, today's disposition, est depth info)."""
    from ayas import rules
    from . import severity as SV
    part = {"material": r["material"], "thickness_mm": r["thickness_mm"], "size_mm": [100.0, 100.0]}
    d = r["rules_defect"]
    gt = SV.judge_with_depth(d, spec, part, {"max_depth_mm": r["gt_depth_mm"], "depth_ci_mm": 0.0, "converged": True})
    today = rules.judge_steel_defect(dict(d), spec, rules.steel_part(spec, part), {})
    m = r.get(method) or {}
    est = SV.judge_with_depth(d, spec, part, m if m.get("max_depth_mm") is not None else None)
    return gt["disposition"], est["disposition"], today["disposition"], est.get("depth", {})


def _severity_table(rows: list[dict], method: str, spec: dict) -> dict:
    """Dispositions with the estimated depth (and today's) against the true-depth disposition.
    unsafe_used: the depth was used and the result is more lenient than with the true depth;
    unsafe_fallback: the depth was not used and the fallback (today's rule) is more lenient."""
    from ayas.rules import VERDICTS
    c = {"n": 0, "agree": 0, "unsafe_used": 0, "unsafe_fallback": 0, "stricter": 0, "depth_used": 0,
         "used_agree": 0, "today_agree": 0, "today_unsafe": 0, "today_stricter": 0}
    for r in rows:
        if (r.get(method) or {}).get("max_depth_mm") is None:
            continue
        g, e, t, info = _dispositions(r, method, spec)
        used = bool(info.get("used"))
        c["n"] += 1
        c["agree"] += g == e
        c["depth_used"] += used
        c["used_agree"] += used and g == e
        if VERDICTS.index(e) < VERDICTS.index(g):
            c["unsafe_used" if used else "unsafe_fallback"] += 1
        c["stricter"] += VERDICTS.index(e) > VERDICTS.index(g)
        c["today_agree"] += g == t
        c["today_unsafe"] += VERDICTS.index(t) < VERDICTS.index(g)
        c["today_stricter"] += VERDICTS.index(t) > VERDICTS.index(g)
    return c


def summarise(out_md: Path, coupons_jsonl: Path, pilot_jsonl: Path, bench_json: Path | None, fig_dir: Path) -> dict:
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ayas.lightproc import ROOT
    spec = yaml.safe_load(open(ROOT / "configs" / "spec_steel.yaml"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    S: dict[str, Any] = {}
    cp = _load(coupons_jsonl)
    pl = _load(pilot_jsonl)
    # CI calibration on the coupons (L/R, PINN): SYS_REL per class so that ~90 % of the coupon
    # errors fall inside sqrt((1.645 sd)^2 + (rel d)^2 + abs^2); then applied unchanged to the pilot
    # (out of sample) and to every coupon partition.
    from . import depth_pinn as DP
    cal = {}
    for meth in ("pinn", "classical"):
        for n in sorted({r["n_groups"] for r in cp}):
            for cls in ("dent", "pit"):
                rows = [r for r in cp if r["n_groups"] == n and r["cls"] == cls and (r.get(meth) or {}).get("max_depth_mm")]
                ab = 0.02
                need = []
                for r in rows:
                    d = float(r[meth]["max_depth_mm"])
                    mem = r[meth].get("members") or [d]
                    sd = float(np.std(mem, ddof=1)) if len(mem) > 1 else 0.0
                    err = d - float(r["gt_depth_mm"])
                    need.append(math.sqrt(max(0.0, err ** 2 - ab ** 2 - (DP.Z90 * sd) ** 2)) / max(d, 0.02))
                cal[(meth, cls, n)] = (float(np.quantile(need, 0.9)) if need else DP.SYS_REL, ab)
    S["ci_calibration"] = {f"{k[0]}_{k[1]}_{k[2]}groups": {"sys_rel": v[0], "sys_abs_mm": v[1]} for k, v in cal.items()}

    def recal(rows):
        for r in rows:
            for meth in ("pinn", "classical"):
                m = r.get(meth) or {}
                key = (meth, r["cls"], int(r.get("n_groups", 2)))
                if m.get("max_depth_mm") is None or key not in cal:
                    continue
                d = float(m["max_depth_mm"])
                mem = m.get("members") or [d]
                sd = float(np.std(mem, ddof=1)) if len(mem) > 1 else 0.0
                rel, ab = cal[key]
                m["depth_ci_mm_raw"] = m.get("depth_ci_mm")
                m["depth_ci_mm"] = math.sqrt((DP.Z90 * sd) ** 2 + (rel * d) ** 2 + ab ** 2)

    recal(cp)
    recal(pl)
    # coupons: error by partition x method x class
    tab = []
    for n in sorted({r["n_groups"] for r in cp}):
        for meth in ("pinn", "classical"):
            for cls in ("dent", "pit"):
                rows = [r for r in cp if r["n_groups"] == n and r["cls"] == cls]
                st = _err_stats(rows, meth, spec)
                st.update(n_groups=n, method=meth, cls=cls)
                tab.append(st)
    S["coupon_errors"] = tab
    # by material, L/R and best partition
    S["coupon_material"] = []
    for mat in sorted({r["material"] for r in cp}):
        for n in sorted({r["n_groups"] for r in cp}):
            rows = [r for r in cp if r["material"] == mat and r["n_groups"] == n and r["cls"] == "dent"]
            st = _err_stats(rows, "pinn", spec)
            st.update(material=mat, n_groups=n)
            S["coupon_material"].append(st)
    S["coupon_severity"] = []
    for n in sorted({r["n_groups"] for r in cp}):
        for meth in ("pinn", "classical"):
            st = _severity_table([r for r in cp if r["n_groups"] == n], meth, spec)
            st.update(n_groups=n, method=meth)
            S["coupon_severity"].append(st)
    # figure: est vs GT per partition (PINN and classical)
    ns = sorted({r["n_groups"] for r in cp})
    if ns:
        fig, ax = plt.subplots(2, len(ns), figsize=(4 * len(ns), 8), squeeze=False)
        for j, n in enumerate(ns):
            for i, meth in enumerate(("pinn", "classical")):
                a = ax[i, j]
                for cls, mk in (("dent", "o"), ("pit", "^")):
                    rr = [r for r in cp if r["n_groups"] == n and r["cls"] == cls and (r.get(meth) or {}).get("max_depth_mm") is not None]
                    a.scatter([r["gt_depth_mm"] for r in rr], [r[meth]["max_depth_mm"] for r in rr], marker=mk, s=18,
                              label=cls, alpha=0.75)
                a.plot([0, 0.8], [0, 0.8], "k--", lw=0.8)
                a.set_xlim(0, 0.8)
                a.set_ylim(0, 0.8)
                a.set_title(f"{meth}, {'L/R' if n == 2 else f'{n} sectors'}")
                a.set_xlabel("true depth (mm)")
                a.set_ylabel("estimated depth (mm)")
                a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "physics_coupons_scatter.png", dpi=90)
        plt.close(fig)
    # pilot: invariance across conditions
    S["pilot"] = []
    keys = sorted({(r["plate"], r["id"]) for r in pl})
    for plate, did in keys:
        rows = [r for r in pl if r["plate"] == plate and r["id"] == did]
        r0 = rows[0]
        ent = {"plate": plate, "id": did, "cls": r0["cls"], "kind": r0["kind"], "material": r0["material"],
               "thickness_mm": r0["thickness_mm"], "gt_depth_mm": r0["gt_depth_mm"]}
        for mode in ("hood", "open"):
            rm = [r for r in rows if r["device_mode"] == mode]
            ok = [r for r in rm if (r.get("pinn") or {}).get("max_depth_mm") is not None]
            ds = [r["pinn"]["max_depth_mm"] for r in ok]
            cs = [r["classical"]["max_depth_mm"] for r in ok if (r.get("classical") or {}).get("max_depth_mm") is not None]
            ent[mode] = {"n_conditions": len(rm), "retake": sum(r["gate"] == "RETAKE" for r in rm), "n_est": len(ds),
                         "pinn_median": float(np.median(ds)) if ds else None,
                         "pinn_min": float(min(ds)) if ds else None, "pinn_max": float(max(ds)) if ds else None,
                         "pinn_sd": float(np.std(ds)) if len(ds) > 1 else None,
                         "classical_median": float(np.median(cs)) if cs else None,
                         "converged": sum(bool(r["pinn"].get("converged")) for r in ok),
                         "by_condition": {r["condition"]: (r.get("pinn") or {}).get("max_depth_mm") if r["gate"] != "RETAKE"
                                          else "RETAKE" for r in rm},
                         "ci_median": float(np.median([r["pinn"]["depth_ci_mm"] for r in ok])) if ok else None}
            sev = []
            for r in ok:
                g, e, t, info = _dispositions(r, "pinn", spec)
                sev.append((g, e, t, bool(info.get("used"))))
            ent[mode]["severity"] = {"gt": sev[0][0] if sev else None, "today": sev[0][2] if sev else None,
                                     "est": dict(zip(*np.unique([s[1] for s in sev], return_counts=True))) if sev else {},
                                     "depth_used": sum(s[3] for s in sev)}
        S["pilot"].append(ent)
    S["pilot_severity"] = {mode: _severity_table([r for r in pl if r["device_mode"] == mode and r["cls"] in ("dent", "pit")
                                                  and r["gate"] != "RETAKE"], "pinn", spec) for mode in ("hood", "open")}
    S["pilot_errors"] = {mode: {meth: {cls: _err_stats([r for r in pl if r["device_mode"] == mode and r["cls"] == cls],
                                                       meth, spec) for cls in ("dent", "pit", "scratch")}
                                for meth in ("pinn", "classical")} for mode in ("hood", "open")}
    S["pilot_scratch"] = {mode: _err_stats([r for r in pl if r["device_mode"] == mode and r["cls"] == "scratch"], "pinn", spec)
                          for mode in ("hood", "open")}
    if pl:
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
        conds = ["C0_dark", "C1_office", "C2_warm_bulb", "C3_window", "C4_side_lamp", "C5_reflected_lamp", "C6_sun"]
        for a, mode in zip(axs, ("hood", "open")):
            for k_, (plate, did) in enumerate(keys):
                rows = {r["condition"]: r for r in pl if r["plate"] == plate and r["id"] == did and r["device_mode"] == mode}
                y = [((rows.get(c) or {}).get("pinn") or {}).get("max_depth_mm") for c in conds]
                gt = next(iter(rows.values()))["gt_depth_mm"] if rows else None
                line, = a.plot(range(7), [np.nan if v is None else v for v in y], "o-", ms=4, label=f"{plate} {did}")
                if gt is not None:
                    a.axhline(gt, color=line.get_color(), ls="--", lw=0.8)
            a.set_xticks(range(7))
            a.set_xticklabels([c.split("_")[0] for c in conds])
            a.set_title(f"PINN depth by lighting condition ({mode}); dashed = truth; gaps = RETAKE")
            a.set_ylabel("max depth (mm)")
        axs[1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "physics_pilot_invariance.png", dpi=90)
        plt.close(fig)
    if bench_json and bench_json.is_file():
        S["bench"] = json.loads(bench_json.read_text())
    out_md.parent.mkdir(parents=True, exist_ok=True)
    (out_md.with_suffix(".json")).write_text(json.dumps(S, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    out_md.write_text(tables_md(json.loads(out_md.with_suffix(".json").read_text())))
    return S


def _f(x, nd=3, pct=False):
    if x is None:
        return "–"
    if pct:
        return f"{100 * x:.0f} %"
    return f"{x:.{nd}f}"


def tables_md(S: dict) -> str:
    """Markdown tables of a summary dict (the narrative is written in reports/physics_pinn.md)."""
    L = []
    L.append("### Coupons: depth error by ring partition and method (dark room, hood, Blender per-LED renders + sensor noise)\n")
    L.append("| LED groups | method | class | n | MAE mm | median abs mm | bias mm | median rel | within 0.05 mm | within 25 % | median err / T1 | CI coverage | converged |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in S.get("coupon_errors", []):
        if not r.get("n"):
            continue
        g = "L/R" if r["n_groups"] == 2 else f"{r['n_groups']} sectors"
        L.append(f"| {g} | {r['method']} | {r['cls']} | {r['n']} | {_f(r['mae_mm'])} | {_f(r['median_abs_mm'])} | "
                 f"{_f(r['bias_mm'])} | {_f(r['median_rel'], 2)} | {_f(r['within_0p05mm'], pct=True)} | "
                 f"{_f(r['within_25pct'], pct=True)} | {_f(r.get('err_over_T1_median'), 2)} | "
                 f"{_f(r.get('ci_coverage'), pct=True)} | {_f(r.get('converged'), pct=True)} |")
    L.append("\n### Coupons: PINN dent depth error by material\n")
    L.append("| material | LED groups | n | MAE mm | bias mm | within 0.05 mm |")
    L.append("|---|---|---|---|---|---|")
    for r in S.get("coupon_material", []):
        if r.get("n"):
            g = "L/R" if r["n_groups"] == 2 else f"{r['n_groups']} sectors"
            L.append(f"| {r['material']} | {g} | {r['n']} | {_f(r['mae_mm'])} | {_f(r['bias_mm'])} | {_f(r['within_0p05mm'], pct=True)} |")
    L.append("\n### Coupons: disposition with the estimated depth vs with the true depth\n")
    L.append("| LED groups | method | n | depth used | used and agrees | agrees (all) | more lenient: depth used | more lenient: fallback | stricter | today agrees | today more lenient | today stricter |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in S.get("coupon_severity", []):
        if r.get("n"):
            g = "L/R" if r["n_groups"] == 2 else f"{r['n_groups']} sectors"
            L.append(f"| {g} | {r['method']} | {r['n']} | {r['depth_used']} | {r['used_agree']} | {r['agree']} | "
                     f"{r['unsafe_used']} | {r['unsafe_fallback']} | {r['stricter']} | {r['today_agree']} | "
                     f"{r['today_unsafe']} | {r['today_stricter']} |")
    if S.get("pilot"):
        L.append("\n### Pilot plates: PINN depth across the 7 lighting conditions (L/R, real burst path)\n")
        L.append("| plate | defect | material, t | true mm | hood: median [min–max] mm (n, retakes) | open: median [min–max] mm (n, retakes) | classical hood median | disposition true / today / estimated (hood) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for e in S["pilot"]:
            def cell(m):
                x = e[m]
                if not x["n_est"]:
                    return f"– (0, {x['retake']} retake)"
                return f"{_f(x['pinn_median'])} [{_f(x['pinn_min'])}–{_f(x['pinn_max'])}] ({x['n_est']}, {x['retake']})"
            sv = e["hood"]["severity"]
            est = ", ".join(f"{k} {v}" for k, v in (sv.get("est") or {}).items())
            L.append(f"| {e['plate']} | {e['id']} {e['cls']}.{e['kind']} | {e['material']}, {e['thickness_mm']:.1f} | "
                     f"{_f(e['gt_depth_mm'])} | {cell('hood')} | {cell('open')} | {_f(e['hood']['classical_median'])} | "
                     f"{sv.get('gt')} / {sv.get('today')} / {est} |")
        pe = S.get("pilot_errors", {})
        L.append("\n### Pilot plates: depth error per class (every accepted capture, all 7 conditions)\n")
        L.append("| mode | method | class | n | MAE mm | median abs mm | bias mm | within 0.05 mm | CI coverage | converged |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for mode, d in pe.items():
            for meth, dc in d.items():
                for cls, r in dc.items():
                    if r.get("n"):
                        L.append(f"| {mode} | {meth} | {cls} | {r['n']} | {_f(r['mae_mm'])} | {_f(r['median_abs_mm'])} | "
                                 f"{_f(r['bias_mm'])} | {_f(r['within_0p05mm'], pct=True)} | {_f(r.get('ci_coverage'), pct=True)} | "
                                 f"{_f(r.get('converged'), pct=True)} |")
    if S.get("pilot_severity"):
        L.append("\n### Pilot plates: dispositions over every accepted capture (dents and pits, PINN, L/R)\n")
        L.append("| mode | n | depth used | used and agrees | agrees (all) | more lenient: depth used | more lenient: fallback | stricter | today agrees | today more lenient | today stricter |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for mode, r in S["pilot_severity"].items():
            if r.get("n"):
                L.append(f"| {mode} | {r['n']} | {r['depth_used']} | {r['used_agree']} | {r['agree']} | {r['unsafe_used']} | "
                         f"{r['unsafe_fallback']} | {r['stricter']} | {r['today_agree']} | {r['today_unsafe']} | {r['today_stricter']} |")
    if S.get("ci_calibration"):
        L.append("\n### Interval calibration (coupons, per class and partition; SYS_ABS = 0.02 mm)\n")
        L.append("| key | SYS_REL |")
        L.append("|---|---|")
        for k, v in S["ci_calibration"].items():
            L.append(f"| {k} | {v['sys_rel']:.3f} |")
    if S.get("bench"):
        b = S["bench"]
        L.append(f"\n### Runtime ({b.get('machine')}, torch {b.get('torch')}; load average at the end {b.get('loadavg_end')})\n")
        L.append("| threads | defect | defect mm | crop grid px | working px (block) | PINN s (seeds) | classical s |")
        L.append("|---|---|---|---|---|---|---|")
        for r in b["rows"]:
            L.append(f"| {r['threads']} | {r['id']} {r['cls']}.{r['kind']} | {r['defect_mm']} | {r['crop_grid_px']} | "
                     f"{r['work_px'][0]}×{r['work_px'][1]} ({r['block']}×) | {r['pinn_s']} ({r['pinn_seeds']}) | {r['classical_s']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
