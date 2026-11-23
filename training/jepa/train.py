"""Train the cross-lighting I-JEPA on twin plate grids (docs/JEPA.md).

  .venv-train/bin/python training/jepa/train.py --config training/jepa/configs/smoke.yaml \
      --data <pairs dir> --out runs/jepa/smoke
  python training/jepa/train.py --config training/jepa/configs/full.yaml --data /home/ayas/data/jepa_train \
      --out /home/ayas/runs/jepa/full --device cuda            # JarvisLabs (jl_train.sh)

Outputs in --out: config.yaml (resolved), metrics.jsonl (one line per log step), last.pt (resume:
model + optimiser + step), encoder.pt (EMA target encoder + predictor + cfg, what eval and
export load), summary.json. Deterministic: every crop, mask and augmentation is drawn from
(seed, sample index); CUDA runs set CUBLAS_WORKSPACE_CONFIG and deterministic algorithms (warn
only), MPS has a few non-deterministic kernels (close, not bit-exact).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from jepa import grids as G  # noqa: E402
from jepa.data import MaskCollator, PairCrops  # noqa: E402
from jepa.masks import MaskCfg  # noqa: E402
from jepa.model import JEPA, ema_momentum, lr_at  # noqa: E402


def set_by_path(cfg: dict[str, Any], dotted: str, value: str) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def load_cfg(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    for o in overrides:
        k, v = o.split("=", 1)
        set_by_path(cfg, k, v)
    return cfg


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _worker_init(wid: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def param_groups(model: JEPA, wd: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for mod in (model.encoder, model.predictor):
        for n, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or "pos" in n or "token" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def save_encoder(path: Path, model: JEPA, cfg: dict[str, Any], step: int, extra: dict[str, Any]) -> None:
    """What eval/export need: the full JEPA state dict (both encoders + predictor) and the cfg."""
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, **extra}, path)


def train(cfg: dict[str, Any], out: Path, device: torch.device, resume: bool = False,
          max_minutes: float | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0))
    seed_all(seed, device)
    d, t = cfg["data"], cfg["train"]
    recs = G.index_grids(d["roots"], d.get("arms"), d.get("splits"))
    steps, batch = int(t["steps"]), int(t["batch"])
    ds = PairCrops(recs, cfg, length=steps * batch, seed=seed)
    n_acc = sum(len(g) for g in ds.groups)
    print(f"data: {len(recs)} grids indexed, {n_acc} accepted in {len(ds.groups)} pairing groups "
          f"({len({g[0].plate for g in ds.groups})} plates)", flush=True)
    model = JEPA(cfg).to(device)
    mcfg = MaskCfg.from_dict(cfg.get("mask"), model.encoder.grid)
    workers = int(t.get("workers", 4))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False, drop_last=True, num_workers=workers,
        collate_fn=MaskCollator(mcfg, seed), worker_init_fn=_worker_init, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
    opt = torch.optim.AdamW(param_groups(model, float(t.get("weight_decay", 0.05))), lr=float(t["lr"]),
                            betas=(0.9, 0.95))
    start = 0
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        print(f"resumed at step {start}", flush=True)
    amp = str(t.get("amp", "auto"))
    use_amp = device.type == "cuda" and amp in ("auto", "bf16", "true", "True")
    ema = tuple(float(x) for x in t.get("ema", [0.996, 1.0]))
    clip = float(t.get("grad_clip", 0) or 0)
    log_every, ckpt_every = int(t.get("log_every", 10)), int(t.get("ckpt_every", 1000))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    mlog = open(out / "metrics.jsonl", "a")
    t0, seen, last = time.time(), 0, {}
    model.train()
    it = iter(loader)
    if start:
        for _ in range(start):                  # deterministic resume: skip the consumed batches
            next(it)
    stop_reason = "steps"
    for step in range(start, steps):
        b = next(it)
        lr = lr_at(step, steps, float(t["lr"]), int(t.get("warmup", 0)), float(t.get("min_lr", 0)))
        for g in opt.param_groups:
            g["lr"] = lr
        ctx, tgt = b["ctx"].to(device, non_blocking=True), b["tgt"].to(device, non_blocking=True)
        ci, ti = b["ctx_idx"].to(device), b["tgt_idx"].to(device)
        vis, cross = b["vis"].to(device), b["cross"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            res = model(ctx, tgt, ci, ti, vis, cross)
        opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.predictor.parameters()),
                                            clip if clip > 0 else 1e9)
        opt.step()
        m = ema_momentum(step, steps, ema)
        model.ema_update(m)
        seen += batch
        if (step + 1) % log_every == 0 or step == start or step + 1 == steps:
            el = time.time() - t0
            last = {"step": step + 1, "loss": float(res["loss"]), "loss_hidden": float(res["loss_hidden"]),
                    "loss_visible_cross": float(res["loss_visible"]), "tgt_spread": float(res["tgt_spread"]),
                    "grad_norm": float(gn), "lr": lr, "ema": m, "cross_frac": float(b["cross"].float().mean()),
                    "img_per_s": round(seen / max(el, 1e-6), 1), "elapsed_s": round(el, 1)}
            mlog.write(json.dumps(last) + "\n")
            mlog.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in last.items()), flush=True)
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "cfg": cfg},
                       out / "last.pt")
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            stop_reason = f"time limit {max_minutes} min"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "cfg": cfg},
                       out / "last.pt")
            break
    mlog.close()
    final_step = int(last.get("step", start))
    summary = {"steps_done": final_step, "steps_planned": steps, "stop": stop_reason,
               "seconds": round(time.time() - t0, 1), "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
               "torch": torch.__version__, "amp_bf16": use_amp, "n_grids": len(recs), "n_accepted": n_acc,
               "n_plates": len({g[0].plate for g in ds.groups}), "last": last,
               "plates": sorted({g[0].plate for g in ds.groups})[:50]}
    save_encoder(out / "encoder.pt", model, cfg, final_step, {"summary": summary})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("summary " + json.dumps(summary), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("configs") / "smoke.yaml"))
    ap.add_argument("--data", nargs="+", help="override data.roots")
    ap.add_argument("--out", help="override train.out")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-minutes", type=float, help="stop (and checkpoint) after this wall time")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config, a.set)
    if a.data:
        cfg["data"]["roots"] = a.data
    for k, v in (("steps", a.steps), ("batch", a.batch), ("workers", a.workers)):
        if v is not None:
            cfg["train"][k] = v
    out = Path(a.out or cfg["train"].get("out", "runs/jepa/run"))
    cfg["train"]["out"] = str(out)
    return train(copy.deepcopy(cfg), out, pick_device(a.device), a.resume, a.max_minutes)


if __name__ == "__main__":
    main()
