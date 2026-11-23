"""The lighting test matrix: every test plate under every lighting condition (DESIGN_STEEL G).

For each test plate seed this writes one plan per (condition, device_mode) row, split "test":
every condition with the open device, plus `hood_conditions` with the hood (by default every
condition too: the hood head with its LED ring is device v2, the product configuration,
DESIGN_STEEL C2 and G). The plate,
defects and distractors of a plate are identical in all its rows, because
sampler.make_plan only changes ambient, extras.device/hood, camera and capture when
condition or device_mode is overridden; write_matrix_plans checks this and refuses
otherwise.

- Seeds start at TEST_SEED_MIN (900000). Training and validation scenes use seeds below
  it, so a matrix plate is never trained on (training/train_arms.py refuses such tiles).
  At least n // 4 plates are defect-free (pick_seeds; research: 2-3 of 8-10), so false
  rejects of clean parts are part of the experiment.
- Each row has its own scene id, `<plate>_<code><o|h>` (s900000_C0o, s900000_C6h), so the
  rows render, post-process and evaluate as separate scenes. plan["seed"] stays the plate
  seed (post draws the variants from it: common random numbers across conditions), and
  plan["matrix"] records the row.

Writes <out>/plans/<scene_id>.json and <out>/matrix_manifest.json. run_matrix renders the
plans (textures -> Blender -> post, pipeline.run_plans); pipeline.run_matrix does both steps.

  python -m ayas.twin.matrix --n 10 --out data/twin_matrix                       # plans + manifest
  python -m ayas.twin.matrix --n 10 --out data/twin_matrix --render --overview   # ... and render them
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from . import sampler

SCHEMA = "ayas.twin.matrix/1"
TEST_SEED_MIN = 900000          # training scenes use seeds below this
MANIFEST = "matrix_manifest.json"
DEVICE_MODES = ("open", "hood")
# DESIGN_STEEL C; the live list is ayas.twin.lighting.CONDITIONS (used when importable).
DEFAULT_CONDITIONS = ("C0_dark", "C1_office", "C2_warm_bulb", "C3_window", "C4_side_lamp",
                      "C5_reflected_lamp", "C6_sun")
SAME_KEYS = ("plate", "sheets", "holes", "rivets", "defects", "distractors", "empty")   # identical in a plate's rows


def _conditions() -> tuple[str, ...]:
    try:
        from .lighting import CONDITIONS
    except ImportError:          # lighting.py may not be importable yet
        return DEFAULT_CONDITIONS
    return tuple(CONDITIONS)


ALL = _conditions()
# DESIGN_STEEL C2/G: the conditions also rendered with the hood. Device v2 (the hood head and
# its dark-field ring) is the product, so it is tested in every condition (was C4 and C6 only).
HOOD_CONDITIONS = ALL


def is_matrix_seed(seed: int) -> bool:
    return int(seed) >= TEST_SEED_MIN


def row_tag(condition: str, device_mode: str) -> str:
    """Short row name: C6_sun + hood -> C6h."""
    if device_mode not in DEVICE_MODES:
        raise ValueError(f"device_mode {device_mode!r} not in {DEVICE_MODES}")
    code = condition.split("_")[0] if "_" in condition else condition
    return f"{code}{device_mode[0]}"


def rows_for(conditions: Sequence[str] = ALL,
             hood_conditions: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """(condition, device_mode) rows: every condition open, then the hood rows.

    hood_conditions None: the default HOOD_CONDITIONS (every condition) that are among
    `conditions` (a matrix of C0 and C6 gets C0h and C6h); given explicitly, each must be in
    `conditions`.
    """
    if hood_conditions is None:
        hood_conditions = [c for c in HOOD_CONDITIONS if c in conditions]
    unknown = [c for c in hood_conditions if c not in conditions]
    if unknown:
        raise ValueError(f"hood conditions {unknown} are not in conditions {list(conditions)}")
    rows = [(c, "open") for c in conditions] + [(c, "hood") for c in hood_conditions]
    tags = [row_tag(c, m) for c, m in rows]
    if len(set(tags)) != len(tags):
        raise ValueError(f"row tags collide: {tags}")
    return rows


def part_of(plan: dict[str, Any]) -> dict[str, Any]:
    """The part dict the steel rules take: material kind, thickness and size."""
    p = plan.get("plate") or {}
    mat = p.get("material") or {}
    return {"material": mat.get("kind") or p.get("finish"), "thickness_mm": p.get("thickness_mm"),
            "size_mm": list(p.get("size_mm") or [])}


def _check_row(plan: dict[str, Any], condition: str, device_mode: str) -> None:
    got_c = (plan.get("ambient") or {}).get("condition")
    got_m = ((plan.get("extras") or {}).get("device") or {}).get("mode")
    if got_c != condition or got_m != device_mode:
        raise ValueError(f"sampler ignored the override: asked {condition}/{device_mode}, plan has {got_c}/{got_m}")


def _check_same(ref: dict[str, Any], plan: dict[str, Any]) -> None:
    for k in SAME_KEYS:
        if json.dumps(ref.get(k), sort_keys=True) != json.dumps(plan.get(k), sort_keys=True):
            raise ValueError(f"{plan['scene_id']}: '{k}' differs from {ref['scene_id']}; the matrix needs "
                             "the same plate in every row (sampler per-part RNG streams)")


def matrix_plans(cfg: dict[str, Any], seed: int, rows: Sequence[tuple[str, str]],
                 coupons: dict[str, dict] | None = None) -> list[dict[str, Any]]:
    """The plans of one plate, one per row, checked for identical plate/defects/distractors."""
    if not is_matrix_seed(seed):
        raise ValueError(f"seed {seed} < {TEST_SEED_MIN}: matrix plates must never overlap training seeds")
    base = sampler.scene_id(seed)
    plans = []
    for condition, mode in rows:
        plan = sampler.make_plan(cfg, seed, "test", mode="surface", source="random", coupons=coupons,
                                 condition=condition, device_mode=mode)
        _check_row(plan, condition, mode)
        tag = row_tag(condition, mode)
        plan["scene_id"] = f"{base}_{tag}"
        plan["split"] = "test"
        plan["matrix"] = {"plate": base, "plate_seed": int(seed), "condition": condition,
                          "device_mode": mode, "row": tag}
        if plans:
            _check_same(plans[0], plan)
        plans.append(plan)
    return plans


def _is_clean(plan: dict[str, Any]) -> bool:
    return not any(d.get("label", True) for d in plan.get("defects") or [])


def pick_seeds(cfg: dict[str, Any], n_plates: int, start_seed: int, min_clean: int,
               probe: tuple[str, str] = ("C0_dark", "open"), max_scan: int | None = None) -> list[int]:
    """n_plates plate seeds from start_seed up with at least min_clean defect-free plates.

    The consecutive seeds start_seed .. start_seed + n_plates - 1 when they already hold
    min_clean clean plates; otherwise the next clean seeds after them replace the last
    defective ones (the research matrix: 8-10 plates, 2-3 of them defect-free, so false
    rejects of clean parts are measured too). One make_plan per seed decides (the plate does
    not depend on the row); at most max_scan seeds past the block are searched.
    """
    n, start = int(n_plates), int(start_seed)
    min_clean = max(0, min(int(min_clean), n))

    def clean(seed: int) -> bool:
        return _is_clean(sampler.make_plan(cfg, seed, "test", mode="surface", source="random",
                                           condition=probe[0], device_mode=probe[1]))

    seeds = list(range(start, start + n))
    flags = [clean(s) for s in seeds] if min_clean else []
    need = min_clean - sum(flags)
    if need <= 0:
        return seeds
    found = []
    for seed in range(start + n, start + n + (max_scan or 50 * n + 100)):
        if clean(seed):
            found.append(seed)
            if len(found) == need:
                break
    if len(found) < need:
        print(f"note: only {sum(flags) + len(found)} defect-free plates found for the matrix (wanted {min_clean})")
    drop = [s for s, f in zip(seeds, flags) if not f][::-1][: len(found)]
    return sorted([s for s in seeds if s not in drop] + found)


def write_matrix_plans(cfg: dict[str, Any], n_plates: int, out: str | Path, conditions: Sequence[str] = ALL,
                       hood_conditions: Sequence[str] | None = None,
                       start_seed: int = TEST_SEED_MIN, min_clean: int | None = None) -> dict[str, Any]:
    """Write <out>/plans/<plate>_<row>.json for n_plates seeds and <out>/matrix_manifest.json.

    hood_conditions None = HOOD_CONDITIONS (every condition) that are among `conditions`.
    min_clean None = n_plates // 4 defect-free plates at least (pick_seeds; 10 plates -> 2).

    Returns the manifest: {schema, split, start_seed, conditions, hood_conditions, rows,
    plates: [{plate, seed, part, clean, defects, rows: [{scene_id, condition, device_mode, tag, plan}]}]}.
    """
    if not is_matrix_seed(start_seed):
        raise ValueError(f"start_seed {start_seed} < {TEST_SEED_MIN}: it would overlap training seeds")
    out = Path(out)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    rows = rows_for(conditions, hood_conditions)
    hood_conditions = [c for c, m in rows if m == "hood"]
    plates = []
    min_clean = int(n_plates) // 4 if min_clean is None else int(min_clean)
    seeds = pick_seeds(cfg, n_plates, start_seed, min_clean, rows[0]) if min_clean > 0 else \
        list(range(int(start_seed), int(start_seed) + int(n_plates)))
    for seed in seeds:
        plans = matrix_plans(cfg, seed, rows)
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
        plates.append({"plate": sampler.scene_id(seed), "seed": seed, "part": part_of(ref),
                       "clean": not any(d["label"] for d in defects), "defects": defects, "rows": entries})
    manifest = {"schema": SCHEMA, "split": "test", "start_seed": int(start_seed), "n_plates": int(n_plates),
                "min_clean": min_clean, "n_clean": sum(p["clean"] for p in plates),
                "conditions": list(conditions), "hood_conditions": list(hood_conditions),
                "rows": [{"condition": c, "device_mode": m, "tag": row_tag(c, m)} for c, m in rows],
                "plates": plates}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1))
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads((p / MANIFEST if p.is_dir() else p).read_text())


def scene_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """scene_id -> {plate, seed, part, condition, device_mode, tag}."""
    idx = {}
    for pl in manifest.get("plates") or []:
        for r in pl["rows"]:
            idx[r["scene_id"]] = {"plate": pl["plate"], "seed": pl["seed"], "part": pl.get("part"),
                                  "condition": r["condition"], "device_mode": r["device_mode"], "tag": r["tag"]}
    return idx


def plan_paths(out: str | Path, manifest: dict[str, Any] | None = None) -> list[Path]:
    out = Path(out)
    manifest = manifest or load_manifest(out)
    return [out / r["plan"] for pl in manifest["plates"] for r in pl["rows"]]


def run_matrix(cfg: dict[str, Any], out: str | Path, only: Iterable[str] | None = None, **kw: Any) -> dict[str, Any]:
    """Render and post-process the matrix plans already in <out> with pipeline.run_plans (batches,
    resumable, disk-guarded). only: a subset of scene ids. kw: batch, device, samples, res_scale,
    keep_work, save_bursts, overview (see pipeline.run_plans)."""
    from .pipeline import run_plans

    paths = plan_paths(out)
    if only is not None:
        keep = set(only)
        paths = [p for p in paths if p.stem in keep]
    return run_plans(cfg, paths, out, **kw)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="python -m ayas.twin.matrix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--n", type=int, default=10, help="test plates")
    ap.add_argument("--out", default="data/twin_matrix")
    ap.add_argument("--start-seed", type=int, default=TEST_SEED_MIN)
    ap.add_argument("--conditions", nargs="+", default=list(ALL))
    ap.add_argument("--hood-conditions", nargs="*",
                    help="default: every condition in --conditions (device v2); none given: no hood rows")
    ap.add_argument("--min-clean", type=int, help="at least this many defect-free plates (default: n // 4)")
    ap.add_argument("--render", action="store_true", help="also render + post-process the plans")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--res-scale", type=float)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--overview", action="store_true", help="also keep the Blender overview renders (demo video)")
    ap.add_argument("--save-bursts", action="store_true", help="also write out/bursts/*.npz (Pi demo)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    man = write_matrix_plans(cfg, a.n, a.out, a.conditions, a.hood_conditions, a.start_seed, a.min_clean)
    n_rows = sum(len(p["rows"]) for p in man["plates"])
    print(f"{len(man['plates'])} plates x {len(man['rows'])} rows = {n_rows} plans -> {a.out}/plans")
    if a.render:
        print(json.dumps(run_matrix(cfg, a.out, batch=a.batch, device=a.device, samples=a.samples,
                                    res_scale=a.res_scale, keep_work=a.keep_work, overview=a.overview,
                                    save_bursts=True if a.save_bursts else None), indent=1))
    return man


if __name__ == "__main__":
    main()
