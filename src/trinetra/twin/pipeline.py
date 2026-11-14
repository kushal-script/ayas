"""Run the twin end to end in batches: plan -> textures -> Blender render -> post.

Work directories are deleted as soon as their scene is post-processed, because the
Mac has little free disk. A run can be stopped and restarted: scenes that already
have a sidecar are skipped. run() samples new plans (optionally forced to one lighting
condition, device mode or split); run_matrix() writes the lighting test matrix
(twin.matrix) and renders it; both go through run_plans().
"""

from __future__ import annotations

import inspect
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import yaml

from . import post, render, sampler, textures

MIN_FREE_GB = 2.0
MAX_POST_FAILURES = 5           # more post errors than this stop the run (kept work dirs fill the disk)


def load_twin_config(path: str | Path = "configs/twin.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _log(out: Path, entry: dict[str, Any]) -> None:
    with open(out / "run_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def with_post(cfg: dict[str, Any], save_bursts: bool | None = None) -> dict[str, Any]:
    """cfg with post.save_bursts overridden (a copy; None keeps the config's value)."""
    if save_bursts is None:
        return cfg
    return {**cfg, "post": {**(cfg.get("post") or {}), "save_bursts": bool(save_bursts)}}


def _render_kwargs(device, samples, res_scale, overview: bool) -> dict[str, Any]:
    kw: dict[str, Any] = {"device": device, "samples": samples, "res_scale": res_scale}
    if overview:
        if "overview" not in inspect.signature(render.render_batch).parameters:
            raise SystemExit("--overview: this render.render_batch has no 'overview' option yet")
        kw["overview"] = True
    return kw


def run_plans(cfg: dict[str, Any], plan_paths: Sequence[str | Path], out: str | Path, batch: int | None = None,
              device: str | None = None, samples: int | None = None, res_scale: float | None = None,
              keep_work: bool = False, save_bursts: bool | None = None, overview: bool = False,
              min_free_gb: float = MIN_FREE_GB) -> dict[str, Any]:
    """Textures -> Blender -> post for existing plan files, in batches. Resumable (a scene with
    out/sidecar/<scene_id>.json is skipped) and disk-guarded (stops below min_free_gb).
    overview asks render.render_batch for its overview-camera render too (kept as
    out/overview/<scene_id>.png)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = with_post(cfg, save_bursts)
    batch = batch or cfg["render"]["batch_size"]
    rkw = _render_kwargs(device, samples, res_scale, overview)
    paths = [Path(p) for p in plan_paths]
    todo = [p for p in paths if not (out / "sidecar" / f"{p.stem}.json").exists()]
    done, failed, retake, t0, stopped, post_failures = 0, [], 0, time.time(), None, 0
    for i in range(0, len(todo), batch):
        if free_gb(out) < min_free_gb:
            stopped = f"less than {min_free_gb} GB free on disk"
            print(f"stopping: {stopped}")
            break
        works = []
        for p in todo[i : i + batch]:
            plan = json.loads(p.read_text())
            work = out / "work" / plan["scene_id"]
            work.mkdir(parents=True, exist_ok=True)
            (work / "plan.json").write_text(json.dumps(plan))
            textures.write_maps(plan, work)
            render.prepare(work, plan, cfg)
            works.append(work)
        tb = time.time()
        metas = render.render_batch(works, cfg, **rkw)
        render_s = time.time() - tb
        for work, meta in zip(works, metas):
            if meta.get("error"):
                failed.append({"scene": work.name, "error": meta["error"]})
                continue
            try:
                sc = post.process_scene(work, out, cfg)
            except ImportError:
                raise                   # lightproc / burst missing: every scene would fail
            except Exception as e:  # noqa: BLE001  (one bad scene must not stop a long run; its work dir is kept)
                failed.append({"scene": work.name, "error": f"post: {type(e).__name__}: {e}"})
                post_failures += 1
                if post_failures > MAX_POST_FAILURES:
                    raise RuntimeError(f"post failed on {post_failures} scenes; last: {failed[-1]}") from e
                continue
            done += 1
            retake += sum(v.get("gate") == post.RETAKE for v in sc["variants"])
            if (work / "overview.png").exists():          # render --overview: keep it, the work dir goes
                (out / "overview").mkdir(exist_ok=True)
                shutil.copy2(work / "overview.png", out / "overview" / f"{work.name}.png")
            if not keep_work:
                shutil.rmtree(work, ignore_errors=True)
        _log(out, {"batch": i // batch, "scenes": len(works), "render_s": round(render_s, 1),
                   "elapsed_s": round(time.time() - t0, 1), "free_gb": round(free_gb(out), 2)})
        print(f"{done}/{len(todo)} scenes, {render_s / max(1, len(works)):.1f} s/scene render, "
              f"{free_gb(out):.1f} GB free")
    post.write_data_yaml(out, cfg["classes"])
    return {"done": done, "failed": failed, "skipped": len(paths) - len(todo), "retake_variants": retake,
            "stopped": stopped, "seconds": round(time.time() - t0, 1)}


def run(cfg: dict[str, Any], n: int | None = None, start_seed: int = 0, out: str | Path | None = None,
        batch: int | None = None, device: str | None = None, samples: int | None = None,
        res_scale: float | None = None, keep_work: bool = False, condition: str | None = None,
        device_mode: str | None = None, split: str | None = None, save_bursts: bool | None = None,
        overview: bool = False) -> dict[str, Any]:
    """Generate n scenes into out (default cfg['out_dir']); returns a short summary.
    condition / device_mode / split force those plan fields (sampler.write_plans)."""
    out = Path(out or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    n = cfg["sampler"]["n_scenes"] if n is None else int(n)     # --n 0 renders nothing (not all 1500)
    force = {k: v for k, v in (("condition", condition), ("device_mode", device_mode), ("split", split)) if v is not None}
    plan_paths = sampler.write_plans(cfg, n, out, start_seed=start_seed, **force)
    return run_plans(cfg, plan_paths, out, batch, device, samples, res_scale, keep_work, save_bursts, overview)


def _default_hood_conditions(matrix) -> tuple[str, ...]:
    """twin.matrix's default hood rows: matrix.HOOD_CONDITIONS (every condition, device v2), else
    write_matrix_plans' hood_conditions default, else C4_side_lamp and C6_sun (older matrix)."""
    if getattr(matrix, "HOOD_CONDITIONS", None):
        return tuple(matrix.HOOD_CONDITIONS)
    p = inspect.signature(matrix.write_matrix_plans).parameters.get("hood_conditions")
    if p is None or p.default is inspect.Parameter.empty or p.default is None:
        return ("C4_side_lamp", "C6_sun")
    return tuple(p.default)


def run_matrix(cfg: dict[str, Any], n_plates: int, out: str | Path, conditions: Sequence[str] | None = None,
               hood_conditions: Sequence[str] | None = None, start_seed: int | None = None,
               plans_only: bool = False, **kw) -> dict[str, Any]:
    """The lighting test matrix (DESIGN_STEEL G): twin.matrix.write_matrix_plans, then run_plans
    on its plans (split test: full grids per eval arm). kw goes to run_plans. With conditions
    and no hood_conditions, the hood rows are the default hood conditions among conditions
    (the matrix refuses hood rows for conditions it does not render)."""
    from . import matrix

    mkw: dict[str, Any] = {}
    if conditions is not None:
        mkw["conditions"] = list(conditions)
        if hood_conditions is None:
            hood_conditions = [c for c in _default_hood_conditions(matrix) if c in mkw["conditions"]]
    if hood_conditions is not None:
        mkw["hood_conditions"] = list(hood_conditions)
    if start_seed is not None:
        mkw["start_seed"] = int(start_seed)
    out = Path(out)
    manifest = matrix.write_matrix_plans(cfg, n_plates, out, **mkw)
    paths = matrix.plan_paths(out, manifest)
    summary = {"plans": len(paths), "plates": len(manifest.get("plates") or [])}
    if plans_only:
        return summary
    return {**summary, **run_plans(cfg, paths, out, **kw)}
