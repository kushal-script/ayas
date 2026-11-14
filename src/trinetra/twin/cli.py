"""trinetra-twin: the digital-twin commands.

  trinetra-twin coupons ...        validate coupons, print fastener tables, write templates + CSV
  trinetra-twin plan --n 20        write scene plans only (cheap; inspect them before rendering)
  trinetra-twin run --n 20         plan -> textures -> Blender -> post, in batches (resumable)
                                   [--condition C6_sun] [--device-mode hood] [--split test]
                                   [--save-bursts] [--overview]
  trinetra-twin matrix --n 10      lighting test matrix: every test plate x every condition
                                   (open) + hood rows -> plans, render, post (DESIGN_STEEL G)
  trinetra-twin post ...           post-process existing work dirs
  trinetra-twin real-tiles ...     real photos / LED bursts + ground_truth.csv -> the same tiles/labels
  trinetra-twin regress ...        advisor regression on twin defect records
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/twin.yaml")
    p.add_argument("--batch", type=int, help="scenes per Blender process")
    p.add_argument("--device", help="METAL | CPU | OPTIX | CUDA")
    p.add_argument("--samples", type=int)
    p.add_argument("--res-scale", type=float, help="render smaller for previews, e.g. 0.25")
    p.add_argument("--keep-work", action="store_true", help="keep textures and EXRs (needs disk)")
    p.add_argument("--save-bursts", action="store_true", help="also write out/bursts/<scene>_v<k>.npz (Pi demo)")
    p.add_argument("--overview", action="store_true", help="also render the Blender overview camera (demo)")


def _render_kw(a: argparse.Namespace) -> dict:
    return {"batch": a.batch, "device": a.device, "samples": a.samples, "res_scale": a.res_scale,
            "keep_work": a.keep_work, "save_bursts": True if a.save_bursts else None, "overview": a.overview}


def _run(argv: list[str]) -> None:
    from .pipeline import load_twin_config, run

    p = argparse.ArgumentParser(prog="trinetra-twin run")
    _render_args(p)
    p.add_argument("--n", type=int, help="number of scenes (default: sampler.n_scenes)")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--out", help="output dir (default: out_dir in the config)")
    p.add_argument("--condition", help="force one ambient condition, e.g. C6_sun (DESIGN_STEEL C)")
    p.add_argument("--device-mode", choices=["open", "hood"], help="force the device mode")
    p.add_argument("--split", help="force the plan split (train | val | test); test writes full grids per arm")
    a = p.parse_args(argv)
    cfg = load_twin_config(a.config)
    summary = run(cfg, n=a.n, start_seed=a.start_seed, out=a.out, condition=a.condition,
                  device_mode=a.device_mode, split=a.split, **_render_kw(a))
    print(json.dumps(summary, indent=1))


def _matrix(argv: list[str]) -> None:
    from .pipeline import load_twin_config, run_matrix

    p = argparse.ArgumentParser(prog="trinetra-twin matrix")
    _render_args(p)
    p.add_argument("--n", type=int, default=10, help="test plates")
    p.add_argument("--out", default="data/twin_matrix")
    p.add_argument("--start-seed", type=int, help="first plate seed (default: twin.matrix's, 900000)")
    p.add_argument("--conditions", nargs="+", help="default: every condition")
    p.add_argument("--hood-conditions", nargs="*", help="default: every condition of --conditions (device v2); none given: no hood rows")
    p.add_argument("--plans-only", action="store_true", help="write the plans and manifest, render nothing")
    a = p.parse_args(argv)
    cfg = load_twin_config(a.config)
    summary = run_matrix(cfg, a.n, a.out, a.conditions, a.hood_conditions, a.start_seed, a.plans_only,
                         **({} if a.plans_only else _render_kw(a)))
    print(json.dumps(summary, indent=1))


def _plan(argv: list[str]) -> None:
    from . import sampler
    from .pipeline import load_twin_config

    p = argparse.ArgumentParser(prog="trinetra-twin plan")
    p.add_argument("--config", default="configs/twin.yaml")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--out", help="output dir (default: out_dir in the config)")
    p.add_argument("--condition")
    p.add_argument("--device-mode", choices=["open", "hood"])
    p.add_argument("--split")
    a = p.parse_args(argv)
    cfg = load_twin_config(a.config)
    force = {k: v for k, v in (("condition", a.condition), ("device_mode", a.device_mode), ("split", a.split))
             if v is not None}
    paths = sampler.write_plans(cfg, a.n, Path(a.out or cfg["out_dir"]), start_seed=a.start_seed, **force)
    print(f"wrote {len(paths)} plans to {paths[0].parent if paths else '-'}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"run": _run, "plan": _plan, "matrix": _matrix}
    if argv and argv[0] == "coupons":
        from .coupons import main as coupons_main
        return coupons_main(argv[1:])
    if argv and argv[0] == "post":
        from .post import main as post_main
        return post_main(argv[1:])
    if argv and argv[0] == "real-tiles":
        from .realdata import main as real_main
        return real_main(argv[1:])
    if argv and argv[0] == "regress":
        from .records import main as records_main
        return records_main(argv[1:])
    if not argv or argv[0] not in commands:
        print(__doc__)
        raise SystemExit(0 if not argv or argv[0] in ("-h", "--help") else 2)
    commands[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
