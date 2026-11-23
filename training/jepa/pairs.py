"""Paired TRAINING plates for the cross-lighting JEPA: every plate under every (condition, device) row.

`ayas-twin matrix` only writes TEST plates (seeds >= 900000; twin.matrix refuses lower seeds
on purpose), so this module writes the same kind of rows for TRAINING seeds (100000..899999)
with sampler.make_plan, checks that plate/defects/distractors are identical across a plate's
rows (as twin.matrix does), and renders them with twin.pipeline.run_plans under a config copy
that keeps the output compact:

- plan split "jepa_train", which is post's only grid split here -> full rectified grids per arm,
  no YOLO tiles (post.tile_splits = []);
- arms: burst only by default (the deployed method; --arms burst naive adds the old arm);
- grids as JPEG q95 (~0.56 MB per 100 x 100 mm grid instead of ~2.9 MB PNG);
- variants_per_scene 2 by default (ambient lux sweep + sensor noise + shake, post only, no render cost).

Writes <out>/plans/*.json, <out>/pairs_manifest.json (same layout as matrix_manifest.json, split
"jepa_train"), <out>/jepa_twin.yaml (the config used) and, with --render, the twin outputs
(<out>/grids/<arm>/jepa_train/*.jpg + .json, <out>/sidecar/*.json).

  python training/jepa/pairs.py --n 2 --start-seed 100000 --out <dir> --render --res-scale 0.5 --samples 32
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "jepa_train"
MANIFEST = "pairs_manifest.json"
TRAIN_SEED_MIN = 100000
TEST_SEED_MIN = 900000            # = ayas.twin.matrix.TEST_SEED_MIN (never trained on)
SAME_KEYS = ("plate", "sheets", "holes", "rivets", "defects", "distractors", "empty")


def check_train_seed(seed: int) -> None:
    if not (TRAIN_SEED_MIN <= int(seed) < TEST_SEED_MIN):
        raise ValueError(f"seed {seed}: JEPA training plates use seeds {TRAIN_SEED_MIN}..{TEST_SEED_MIN - 1} "
                         "(>= 900000 are test plates and are never trained on)")


def jepa_config(cfg: dict[str, Any], arms: Sequence[str] = ("burst",), variants: int = 2,
                grid_format: str = "jpg", quality: int = 95) -> dict[str, Any]:
    """A copy of the twin config whose post writes compact full grids for the jepa_train split only."""
    c = copy.deepcopy(cfg)
    p = c.setdefault("post", {})
    p.update({"grid_splits": [SPLIT], "eval_arms": list(arms), "tile_splits": [], "train_arm": "burst",
              "grid_format": grid_format, "jpeg_quality": int(quality), "variants_per_scene": int(variants),
              "save_bursts": False})
    return c


def plate_plans(cfg: dict[str, Any], seed: int, rows: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    from ayas.twin import matrix, sampler

    check_train_seed(seed)
    base = sampler.scene_id(seed)
    plans: list[dict[str, Any]] = []
    for condition, mode in rows:
        plan = sampler.make_plan(cfg, seed, SPLIT, mode="surface", source="random",
                                 condition=condition, device_mode=mode)
        got_c = (plan.get("ambient") or {}).get("condition")
        got_m = ((plan.get("extras") or {}).get("device") or {}).get("mode")
        if (got_c, got_m) != (condition, mode):
            raise ValueError(f"sampler ignored the override: {condition}/{mode} -> {got_c}/{got_m}")
        tag = matrix.row_tag(condition, mode)
        plan["scene_id"] = f"{base}_{tag}"
        plan["split"] = SPLIT
        plan["matrix"] = {"plate": base, "plate_seed": int(seed), "condition": condition,
                          "device_mode": mode, "row": tag}
        if plans:
            for k in SAME_KEYS:
                if json.dumps(plans[0].get(k), sort_keys=True) != json.dumps(plan.get(k), sort_keys=True):
                    raise ValueError(f"{plan['scene_id']}: '{k}' differs across rows")
        plans.append(plan)
    return plans


def write_pair_plans(cfg: dict[str, Any], n_plates: int, out: str | Path, start_seed: int = TRAIN_SEED_MIN,
                     conditions: Sequence[str] | None = None,
                     device_modes: Sequence[str] = ("open", "hood"),
                     hood_conditions: Sequence[str] | None = None) -> dict[str, Any]:
    """hood_conditions None = the same conditions as open (14 rows per plate); e.g. [C0_dark,
    C4_side_lamp, C6_sun] gives 10 rows (the sealed hood rows differ from C0h by ~1.5 DN only)."""
    from ayas.twin import matrix

    out = Path(out)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    conditions = list(conditions or matrix.ALL)
    hood = list(hood_conditions) if hood_conditions is not None else conditions
    rows = ([(c, "open") for c in conditions] if "open" in device_modes else []) + \
           ([(c, "hood") for c in hood] if "hood" in device_modes else [])
    plates = []
    for seed in range(int(start_seed), int(start_seed) + int(n_plates)):
        check_train_seed(seed)
        plans = plate_plans(cfg, seed, rows)
        entries = []
        for plan in plans:
            rel = f"plans/{plan['scene_id']}.json"
            (out / rel).write_text(json.dumps(plan))
            m = plan["matrix"]
            entries.append({"scene_id": plan["scene_id"], "condition": m["condition"],
                            "device_mode": m["device_mode"], "tag": m["row"], "plan": rel})
        ref = plans[0]
        defects = [{"id": d.get("id"), "cls": d.get("cls"), "kind": d.get("kind"), "label": d.get("label", True)}
                   for d in ref.get("defects") or []]
        plates.append({"plate": ref["matrix"]["plate"], "seed": seed, "part": matrix.part_of(ref),
                       "clean": not any(d["label"] for d in defects), "defects": defects, "rows": entries})
    man = {"schema": "ayas.jepa.pairs/1", "split": SPLIT, "start_seed": int(start_seed),
           "n_plates": int(n_plates), "conditions": conditions,
           "rows": [{"condition": c, "device_mode": m, "tag": matrix.row_tag(c, m)} for c, m in rows],
           "plates": plates}
    (out / MANIFEST).write_text(json.dumps(man, indent=1))
    return man


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "twin.yaml"))
    ap.add_argument("--n", type=int, default=2, help="training plates")
    ap.add_argument("--start-seed", type=int, default=TRAIN_SEED_MIN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+", help="default: every condition")
    ap.add_argument("--device-modes", nargs="+", default=["open", "hood"])
    ap.add_argument("--hood-conditions", nargs="+", help="hood rows (default: the same as --conditions)")
    ap.add_argument("--arms", nargs="+", default=["burst"], help="grid arms to write (burst | naive)")
    ap.add_argument("--variants", type=int, default=2, help="post variants per render (lux sweep + noise)")
    ap.add_argument("--grid-format", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--res-scale", type=float)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if not cfg.get("capture_config", "").startswith("/"):
        cfg["capture_config"] = str(ROOT / cfg.get("capture_config", "configs/capture.yaml"))
    jcfg = jepa_config(cfg, a.arms, a.variants, a.grid_format)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "jepa_twin.yaml").write_text(yaml.safe_dump(jcfg, sort_keys=False))
    man = write_pair_plans(jcfg, a.n, out, a.start_seed, a.conditions, a.device_modes, a.hood_conditions)
    n_rows = sum(len(p["rows"]) for p in man["plates"])
    print(f"{len(man['plates'])} training plates x {len(man['rows'])} rows = {n_rows} plans -> {out}/plans")
    if a.render:
        from ayas.twin.pipeline import run_plans

        paths = [out / r["plan"] for p in man["plates"] for r in p["rows"]]
        res = run_plans(jcfg, paths, out, batch=a.batch, device=a.device, samples=a.samples,
                        res_scale=a.res_scale)
        print(json.dumps(res, indent=1))
        man["render"] = res
    return man


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
