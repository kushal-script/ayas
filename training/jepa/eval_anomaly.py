"""Anomaly maps and lighting invariance of the JEPA encoder vs baselines (docs/JEPA.md, section 5).

1. Memory bank (PatchCore-style) per feature extractor and scale, from TRAINING plates only
   (seeds < 900000; split test refused): every patch of accepted burst-arm grids whose window
   lies on the plate and whose 16 x 16 (scale 1) or 32 x 32 (scale 2) footprint is outside every
   defect (labelled or not) dilated by --bank-margin-mm; greedy k-centre coreset on a random
   projection; z-score = (NN distance - mu) / sigma with mu, sigma from held-out bank patches.
2. Methods: <extractor>_knn for jepa (EMA target encoder), vit_in1k (timm ViT-Tiny ImageNet, the
   same architecture without JEPA), resnet18 (timm, PatchCore layer2+3), pixels (raw 16 x 16 x 3
   patches); and jepa_pred: the JEPA predictor's error when 2 x 2-patch squares are masked in 4
   checkerboard groups and predicted from the rest of the crop (z-scored on training patches).
   Scales 1 and 2 are averaged (z units).
3. Test grids (split test, seeds >= 900000): one heat map per grid (canonical plate frame,
   saved as float16 at 1/4 resolution), pixel- and region-level AUROC/AP per defect class,
   device mode, condition, material and arm; recall at 0.5 false-alarm cells per plate and at the
   fixed threshold z > --z-thr (FA/plate reported); distractor alarms.
4. Invariance: per (plate, device, variant, arm) every condition's scale-1 patch embeddings
   against C0 of the same device: centred cosine (same patch / other patch), retrieval@1.

Burst-arm test grids whose gate is RETAKE are skipped (the app never inspects them); the naive
arm has no gate. Smoke runs (--smoke) label every output "SMOKE, NOT A RESULT".

  .venv-train/bin/python training/jepa/eval_anomaly.py --ckpt runs/jepa/smoke/encoder.pt \
      --train <pairs dir> --test data/twin_pilot --out reports --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jepa import grids as G  # noqa: E402
from jepa import metrics as MX  # noqa: E402
from jepa.data import normalize, to_input  # noqa: E402
from jepa.masks import checker_groups  # noqa: E402
from jepa.model import Encoder, gather_tokens, load_jepa  # noqa: E402
from trinetra import anomaly as A  # noqa: E402

CLASSES = ("scratch", "dent", "pit", "corrosion", "crack", "stain")
EXTRACTORS = ("jepa", "vit_in1k", "resnet18", "pixels")


# ---- feature extractors -----------------------------------------------------------------------

class Extractors:
    def __init__(self, names: Sequence[str], ckpt: str | None, device: torch.device, pred_block: int = 2):
        self.device, self.names = device, list(names)
        self.jepa = None
        self.models: dict[str, Any] = {}
        if ckpt and ("jepa" in names or "jepa_pred" in names):
            self.jepa, ck = load_jepa(ckpt)
            self.jepa.to(device).eval()
            self.jepa_step = int(ck.get("step", 0))
            grid = self.jepa.encoder.grid
            self.groups = [torch.as_tensor(g, device=device) for g in checker_groups(grid, pred_block)]
            n = grid[0] * grid[1]
            self.keeps = [torch.as_tensor(np.setdiff1d(np.arange(n), g.cpu().numpy()), device=device)
                          for g in self.groups]
        if "vit_in1k" in names:
            self.models["vit_in1k"] = Encoder("vit_tiny", "imagenet").to(device).eval()
        if "resnet18" in names:
            import timm

            self.models["resnet18"] = timm.create_model("resnet18.a1_in1k", pretrained=True, features_only=True,
                                                        out_indices=(2, 3)).to(device).eval()

    @torch.no_grad()
    def embed(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 196, D) patch embeddings on the 14 x 14 grid."""
        if name == "jepa":
            return self.jepa.target_encoder(x)
        if name == "vit_in1k":
            return self.models[name](x)
        if name == "resnet18":
            f2, f3 = self.models[name](x)
            f2 = F.avg_pool2d(f2, 3, 1, 1)
            f3 = F.avg_pool2d(f3, 3, 1, 1)
            f2 = F.adaptive_avg_pool2d(f2, f3.shape[-2:])
            return torch.cat([f2, f3], 1).flatten(2).transpose(1, 2)
        if name == "pixels":
            return F.unfold(x, 16, stride=16).transpose(1, 2)
        raise KeyError(name)

    @torch.no_grad()
    def pred_error(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 196): per-patch squared error of the predictor (target = LN(target encoder(x)))."""
        j = self.jepa
        h = j.targets(x)
        err = torch.zeros(h.shape[:2], device=h.device)
        B = x.shape[0]
        for keep, grp in zip(self.keeps, self.groups):
            k = keep[None].expand(B, -1)
            g = grp[None].expand(B, -1)
            p = j.predictor(j.encoder(x, k), k, g)
            e = ((p - gather_tokens(h, g)) ** 2).mean(-1)
            err.scatter_(1, g, e)
        return err


# ---- windows ------------------------------------------------------------------------------

def grid_windows(meta: dict[str, Any], scale: float, stride_frac: float = 1.0) -> tuple[int, list[tuple[int, int]]]:
    win = int(round(224 * scale))
    return win, A.windows(G.plate_rect(meta), win, int(win * stride_frac))


def batch_inputs(img: np.ndarray, wins: Sequence[tuple[int, int]], win: int) -> torch.Tensor:
    xs = [normalize(to_input(A.crop(img, x, y, win), 224)).transpose(2, 0, 1) for x, y in wins]
    return torch.from_numpy(np.stack(xs))


def patch_boxes(x: int, y: int, win: int, n: int = 14) -> np.ndarray:
    p = win / n
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.stack([x + c * p, y + r * p, x + (c + 1) * p, y + (r + 1) * p], -1).reshape(-1, 4)


def patches_clear(mask: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """True for patch boxes that do not touch `mask` (integral image)."""
    ii = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = mask.shape
    b = np.round(boxes).astype(int)
    x0, y0 = np.clip(b[:, 0], 0, w), np.clip(b[:, 1], 0, h)
    x1, y1 = np.clip(b[:, 2], 0, w), np.clip(b[:, 3], 0, h)
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return s == 0


# ---- memory bank --------------------------------------------------------------------------

@torch.no_grad()
def greedy_coreset(z: torch.Tensor, k: int, seed: int = 0, proj: int = 128) -> torch.Tensor:
    n = z.shape[0]
    if n <= k:
        return torch.arange(n, device=z.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if z.shape[1] > proj:
        P = (torch.randn(z.shape[1], proj, generator=g) / proj ** 0.5).to(z.device)
        z = z @ P
    first = int(torch.randint(n, (1,), generator=g))
    sel = torch.empty(k, dtype=torch.long, device=z.device)
    sel[0] = first
    dmin = ((z - z[first]) ** 2).sum(1)
    for i in range(1, k):
        j = torch.argmax(dmin)
        sel[i] = j
        dmin = torch.minimum(dmin, ((z - z[j]) ** 2).sum(1))
    return sel


@torch.no_grad()
def nn_dist(q: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    out = []
    for i in range(0, q.shape[0], chunk):
        out.append(torch.cdist(q[i:i + chunk], bank).min(1).values)
    return torch.cat(out) if out else q.new_zeros(0)


class Bank:
    def __init__(self, feats: torch.Tensor, k: int, seed: int, holdout: int = 4000):
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(feats.shape[0], generator=g).to(feats.device)
        ho = perm[:min(holdout, feats.shape[0] // 5)]
        pool = perm[len(ho):]
        sel = greedy_coreset(feats[pool], k, seed)
        self.bank = feats[pool][sel].contiguous()
        self.mean = feats.mean(0)
        d = nn_dist(feats[ho], self.bank)
        self.mu, self.sigma = float(d.mean()), float(d.std().clamp_min(1e-6))
        self.n_pool, self.n_holdout = int(feats.shape[0]), int(len(ho))

    def z(self, q: torch.Tensor) -> torch.Tensor:
        return (nn_dist(q, self.bank) - self.mu) / self.sigma


# ---- evaluation ---------------------------------------------------------------------------

def load_grid(rec: G.GridRec) -> tuple[np.ndarray, dict[str, Any]]:
    meta = rec.meta()
    return G.to_canonical(G.read_rgb(rec.image), meta), meta


def collect_bank(ex: Extractors, recs: Sequence[G.GridRec], methods: Sequence[str], scales: Sequence[float],
                 margin_mm: float, bs: int, max_per: int = 120000, seed: int = 0,
                 per_grid: int = 1500) -> dict[tuple, torch.Tensor]:
    """Normal patches per (method, scale): at most per_grid random clear patches of each grid, kept
    on the CPU as float16 (the Mac has 16 GB shared with Blender), at most max_per in total."""
    feats: dict[tuple, list[torch.Tensor]] = defaultdict(list)
    gsel = torch.Generator(device="cpu").manual_seed(seed + 7)
    for i, rec in enumerate(recs):
        img, meta = load_grid(rec)
        excl = G.any_defect_mask(meta, margin_mm) | ~G.plate_mask(meta)
        for s in scales:
            win, wins = grid_windows(meta, s)
            keep = np.concatenate([patches_clear(excl, patch_boxes(x, y, win)) for x, y in wins])
            idx = np.flatnonzero(keep)
            if len(idx) > per_grid:            # the same random subset for every method of this grid / scale
                pick = torch.randperm(len(idx), generator=gsel)[:per_grid].numpy()
                keep = np.zeros_like(keep)
                keep[idx[pick]] = True
            for b0 in range(0, len(wins), bs):
                xb = batch_inputs(img, wins[b0:b0 + bs], win).to(ex.device)
                kb = torch.from_numpy(keep[b0 * 196:(b0 + len(xb)) * 196]).to(ex.device)
                for m in methods:
                    if m == "jepa_pred":
                        f = ex.pred_error(xb).reshape(-1, 1)
                    else:
                        f = ex.embed(m.replace("_knn", ""), xb)
                        f = f.reshape(-1, f.shape[-1])
                    feats[(m, s)].append(f[kb].to("cpu", torch.float16))
        if (i + 1) % 10 == 0:
            print(f"  bank: {i + 1}/{len(recs)} grids", flush=True)
    out = {}
    g = torch.Generator(device="cpu").manual_seed(seed)
    for key, lst in feats.items():
        f = torch.cat(lst)
        if f.shape[0] > max_per:
            f = f[torch.randperm(f.shape[0], generator=g)[:max_per]]
        out[key] = f.to(ex.device, torch.float32)
    return out


def plate_groups(recs: Sequence[G.GridRec]) -> list[list[G.GridRec]]:
    """Test grids grouped by (plate, device, k, arm), the C0 row first."""
    gs = G.groups(recs)
    out = []
    for key in sorted(gs):
        rows = sorted(gs[key], key=lambda r: (not r.condition.startswith("C0"), r.tag))
        out.append(rows)
    return out


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    dev = torch.device(args.device if args.device != "auto" else
                       ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)
    scales = [float(s) for s in args.scales]
    extractors = [e for e in args.extractors if e != "jepa" or args.ckpt]
    methods = [f"{e}_knn" for e in extractors] + (["jepa_pred"] if args.ckpt and args.pred else [])
    ex = Extractors(extractors, args.ckpt, dev)

    # one memory bank per arm that has TRAINING grids (the naive arm's own bank when its grids were
    # rendered, --jepa-arms "burst naive"); an arm without one is scored against the burst bank (flagged)
    train_all = [r for r in G.index_grids(args.train, args.arms) if r.accepted]
    G.refuse_test_plates(train_all)
    by_arm: dict[str, list[G.GridRec]] = {}
    for a in args.arms:
        rs = [r for r in train_all if r.arm == a]
        if args.max_train_grids and len(rs) > args.max_train_grids:
            idx = rng.choice(len(rs), args.max_train_grids, replace=False)
            rs = [rs[i] for i in sorted(idx)]
        if rs:
            by_arm[a] = rs
    if "burst" not in by_arm:
        raise SystemExit("no accepted burst-arm TRAINING grids for the memory bank")
    bank_arms = list(by_arm)
    bank_for = {a: (a if a in by_arm else "burst") for a in args.arms}
    train = [r for rs in by_arm.values() for r in rs]
    test = [r for r in G.index_grids(args.test, args.arms) if r.seed >= max(G.TEST_SEED_MIN, args.min_test_seed)]
    if not test:
        raise SystemExit(f"no test grids with seed >= {args.min_test_seed} under {args.test}")
    if args.max_test_plates:
        keep = sorted({r.plate for r in test})[:args.max_test_plates]
        test = [r for r in test if r.plate in keep]
    print(f"banks from {', '.join(f'{a}: {len(rs)}' for a, rs in by_arm.items())} accepted training grids "
          f"({len({r.plate for r in train})} plates); {len(test)} test grids ({len({r.plate for r in test})} plates); "
          f"arm -> bank {bank_for}", flush=True)

    cache_key = json.dumps({"ckpt": str(args.ckpt), "train": sorted(str(r.image) for r in train), "methods": methods,
                            "scales": scales, "margin": args.bank_margin_mm, "coreset": args.coreset,
                            "per_grid": args.per_grid, "seed": args.seed, "bank_arms": bank_arms}, sort_keys=True)
    banks: dict[tuple, Any] = {}
    cache = Path(args.bank_cache) if args.bank_cache else None
    if cache is not None and cache.exists():
        ck = torch.load(cache, map_location=dev, weights_only=False)
        if ck.get("key") == cache_key:
            banks = ck["banks"]
            print(f"banks loaded from {cache}", flush=True)
    if not banks:
        for arm, rs in by_arm.items():
            feats = collect_bank(ex, rs, methods, scales, args.bank_margin_mm, args.batch, seed=args.seed,
                                 per_grid=args.per_grid)
            for (m, s) in list(feats):
                f = feats.pop((m, s))
                if m == "jepa_pred":
                    v = f.squeeze(1)
                    banks[(arm, m, s)] = {"mu": float(v.mean()), "sigma": float(v.std().clamp_min(1e-6)),
                                          "n": int(v.numel())}
                else:
                    banks[(arm, m, s)] = Bank(f, args.coreset, args.seed)
            del feats
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"key": cache_key, "banks": banks}, cache)
    bank_info = {f"{a}:{m}@{s:g}": ({"coreset": int(b.bank.shape[0]), "pool": b.n_pool, "mu": round(b.mu, 4),
                                     "sigma": round(b.sigma, 4)} if isinstance(b, Bank) else b)
                 for (a, m, s), b in banks.items()}
    print(f"banks ready ({time.time() - t0:.0f} s): " + json.dumps(bank_info), flush=True)
    means = {(a, e): banks[(a, f"{e}_knn", scales[0])].mean for a in bank_arms for e in extractors}

    heat_dir = Path(args.heat_dir)
    heat_dir.mkdir(parents=True, exist_ok=True)
    regions: list[dict[str, Any]] = []      # one row per (method, grid, defect)
    cells: list[dict[str, Any]] = []        # one row per (method, grid): normal-cell scores
    pix: list[dict[str, Any]] = []          # one row per (method, grid): sampled pixel scores
    dist: list[dict[str, Any]] = []
    inv: list[dict[str, Any]] = []
    examples: list[tuple[str, np.ndarray, dict[str, np.ndarray]]] = []
    n_grids = n_skipped = n_missing = 0
    for gi, rows in enumerate(plate_groups(test)):
        c0_emb: dict[str, np.ndarray] | None = None
        for rec in rows:
            gated_out = rec.arm == "burst" and not rec.accepted
            try:
                img, meta = load_grid(rec)
            except (FileNotFoundError, json.JSONDecodeError) as e:     # a grid rewritten by post meanwhile
                print(f"  skipped {rec.image.name}: {type(e).__name__}", flush=True)
                n_missing += 1
                continue
            plate = G.plate_mask(meta)
            maps: dict[str, list[np.ndarray]] = defaultdict(list)
            emb1: dict[str, list[np.ndarray]] = defaultdict(list)
            for s in scales:
                win, wins = grid_windows(meta, s)
                per_m: dict[str, list[np.ndarray]] = defaultdict(list)
                for b0 in range(0, len(wins), args.batch):
                    xb = batch_inputs(img, wins[b0:b0 + args.batch], win).to(dev)
                    for e in extractors:
                        f = ex.embed(e, xb)
                        if not gated_out:
                            per_m[f"{e}_knn"].append(banks[(bank_for[rec.arm], f"{e}_knn", s)].z(f.reshape(-1, f.shape[-1]).float())
                                                     .reshape(len(xb), -1).cpu().numpy())
                        if s == scales[0]:
                            emb1[e].append(f.float().cpu().numpy().astype(np.float16))
                    if "jepa_pred" in methods and not gated_out:
                        b = banks[(bank_for[rec.arm], "jepa_pred", s)]
                        per_m["jepa_pred"].append(((ex.pred_error(xb) - b["mu"]) / b["sigma"]).cpu().numpy())
                for m, lst in per_m.items():
                    sc = np.concatenate(lst).reshape(len(wins), 14, 14)
                    maps[m].append(A.assemble(img.shape[:2], [(x, y, sc[i]) for i, (x, y) in enumerate(wins)],
                                              win, plate, sigma_px=4 * s))
            emb = {e: np.concatenate(v) for e, v in emb1.items()}
            # invariance vs C0 of the same plate / device / variant / arm
            if rec.condition.startswith("C0"):
                c0_emb = emb
            elif c0_emb is not None:
                for e in extractors:
                    st = MX.invariance_stats(emb[e].astype(np.float32), c0_emb[e].astype(np.float32),
                                             means[(bank_for[rec.arm], e)].cpu().numpy(), rng)
                    inv.append({"extractor": e, "arm": rec.arm, "device": rec.device_mode, "condition": rec.condition,
                                "accepted": rec.accepted, "material": rec.material, "plate": rec.plate, **st})
            if gated_out:
                n_skipped += 1
                continue
            n_grids += 1
            dmasks = G.defect_masks(meta)
            excl_neg = G.any_defect_mask(meta, args.neg_margin_mm)
            excl_px = G.any_defect_mask(meta, 0.5)
            dpts = G.distractor_points(meta)
            heats = {}
            for m, lst in maps.items():
                heat = np.mean(lst, 0).astype(np.float32)
                heats[m] = heat
                small = cv2.resize(heat, (heat.shape[1] // 4, heat.shape[0] // 4), interpolation=cv2.INTER_AREA)
                np.savez_compressed(heat_dir / f"{rec.image.stem}_{rec.arm}_{m}.npz", heat=small.astype(np.float16))
                base = {"method": m, "arm": rec.arm, "device": rec.device_mode, "condition": rec.condition,
                        "material": rec.material, "plate": rec.plate, "stem": rec.image.stem, "k": rec.k}
                neg_cells = MX.normal_cells(heat, plate, excl_neg, args.cell)
                cells.append({**base, "scores": neg_cells})
                negpx = heat[plate & ~excl_px]
                prow = {**base, "neg": MX.subsample(negpx, 4000, rng), "pos": {}}
                for d in dmasks:
                    if not d["mask"].any():
                        continue
                    vis = bool(d["visible"]) if d["visible"] is not None else True
                    region = G.dilate(d["mask"], args.pos_margin_mm / 0.08)
                    regions.append({**base, "cls": d["cls"], "kind": d["kind"], "id": d["id"], "visible": vis,
                                    "score": float(heat[region].max())})
                    if vis:
                        prow["pos"].setdefault(d["cls"], []).append(MX.subsample(heat[d["mask"]], 2000, rng))
                pix.append(prow)
                for kind, x, y in dpts:
                    r = int(round(2.0 / 0.08))
                    yy, xx = int(round(y)), int(round(x))
                    win_ = heat[max(0, yy - r):yy + r + 1, max(0, xx - r):xx + r + 1]
                    if win_.size:
                        dist.append({**base, "kind": kind, "score": float(win_.max())})
            if len(examples) < args.n_examples and rec.k == 0 and rec.arm == "burst" and \
                    rec.condition in ("C0_dark", "C3_window", "C6_sun"):
                examples.append((f"{rec.image.stem}_{rec.arm}", img, heats))
        print(f"  test group {gi + 1}: {rows[0].plate} {rows[0].device_mode} k{rows[0].k} {rows[0].arm} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return {"methods": methods, "extractors": extractors, "scales": scales, "regions": regions, "cells": cells,
            "pix": pix, "dist": dist, "inv": inv, "examples": examples, "bank_info": bank_info,
            "n_test_grids": n_grids, "n_skipped_retake": n_skipped, "n_missing": n_missing, "n_train_grids": len(train),
            "train_plates": sorted({r.plate for r in train}), "test_plates": sorted({r.plate for r in test}),
            "bank_for": bank_for, "n_train_by_arm": {a: len(rs) for a, rs in by_arm.items()},
            "jepa_step": getattr(ex, "jepa_step", None), "seconds": round(time.time() - t0, 1), "device": str(dev)}


# ---- aggregation and report ---------------------------------------------------------------

def _f(x: float, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def aggregate(res: dict[str, Any], keys: Sequence[str], args: argparse.Namespace,
              cls: str | None = None) -> dict[tuple, dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    sel = lambda row: tuple(row[k] for k in keys)  # noqa: E731
    reg, cel, px, ds = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    grids_per = defaultdict(set)
    for r in res["regions"]:
        if r["visible"] and (cls is None or r["cls"] == cls):
            reg[sel(r)].append(r["score"])
    for c in res["cells"]:
        cel[sel(c)].append(c["scores"])
        grids_per[sel(c)].add(c["stem"])
    for p in res["pix"]:
        pos = [a for k, v in p["pos"].items() if cls is None or k == cls for a in v]
        px[sel(p)].append((pos, p["neg"]))
    for d in res["dist"]:
        ds[sel(d)].append(d["score"])
    for key in cel:
        neg = np.concatenate(cel[key]) if cel[key] else np.zeros(0)
        pos = np.asarray(reg.get(key, []), float)
        n_grid = max(1, len(grids_per[key]))
        cpp = len(neg) / n_grid
        rec05, thr05 = MX.recall_at_budget(pos, neg, cpp, 0.5)
        pp = [a for pos_l, _ in px[key] for a in pos_l]
        pn = [n for _, n in px[key]]
        ppos = np.concatenate(pp) if pp else np.zeros(0)
        pneg = np.concatenate(pn) if pn else np.zeros(0)
        dsc = np.asarray(ds.get(key, []), float)
        out[key] = {"n_def": int(len(pos)), "n_grids": n_grid, "cells_per_plate": round(cpp, 1),
                    "region_auroc": MX.auroc(pos, neg), "region_ap": MX.average_precision(pos, neg),
                    "pixel_auroc": MX.auroc(ppos, pneg), "pixel_ap": MX.average_precision(ppos, pneg),
                    "recall_fa05": rec05, "thr_fa05": thr05,
                    "recall_z": float((pos > args.z_thr).mean()) if len(pos) else float("nan"),
                    "fa_per_plate_z": float((neg > args.z_thr).sum() / n_grid),
                    "distractor_alarm_z": float((dsc > args.z_thr).mean()) if len(dsc) else float("nan")}
    return out


def inv_table(res: dict[str, Any], accepted_only: bool = True) -> dict[tuple, dict[str, float]]:
    acc = defaultdict(list)
    for r in res["inv"]:
        if accepted_only and not r["accepted"]:
            continue
        acc[(r["extractor"], r["arm"], r["device"], r["condition"])].append(r)
    out = {}
    for k, v in acc.items():
        out[k] = {m: float(np.mean([x[m] for x in v])) for m in ("cos_same", "cos_other", "gap", "retrieval_at1")}
        out[k]["n"] = len(v)
    return out


def save_examples(res: dict[str, Any], fig_dir: Path, methods: Sequence[str]) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, img, heats in res["examples"]:
        tiles = [cv2.resize(img, (360, 360 * img.shape[0] // img.shape[1]))]
        for m in methods:
            if m not in heats:
                continue
            h = heats[m]
            hn = np.clip((h + 1) / 6.0, 0, 1)                         # z in [-1, 5] -> [0, 1]
            col = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
            ov = (0.5 * img + 0.5 * col).astype(np.uint8)
            ov = cv2.resize(ov, tiles[0].shape[1::-1])
            cv2.putText(ov, m, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(ov)
        p = fig_dir / f"{name}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p.as_posix())
    return paths


def write_report(res: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    smoke = bool(args.smoke)
    tag = "SMOKE, NOT A RESULT" if smoke else "result (twin test plates)"
    methods = res["methods"]
    t_main = aggregate(res, ["method", "arm", "device"], args)
    t_cls = {c: aggregate(res, ["method", "arm", "device"], args, cls=c) for c in CLASSES}
    t_cond = aggregate(res, ["method", "arm", "device", "condition"], args)
    t_mat = aggregate(res, ["method", "arm", "material"], args)
    t_inv = inv_table(res, True)
    t_inv_all = inv_table(res, False)
    figs = save_examples(res, Path(args.fig_dir), methods)
    L: list[str] = []
    L.append(f"# JEPA anomaly map and lighting invariance: {tag}\n")
    if smoke:
        L.append("> **SMOKE, NOT A RESULT.** This run only proves the pipeline end to end. The JEPA was trained for "
                 f"{res['jepa_step']} steps on {len(res['train_plates'])} low-resolution TRAINING plates "
                 f"({', '.join(res['train_plates'])}; rendered at res-scale 0.5, 32 spp), and the memory bank comes "
                 "from the same plates, while the test plates are the full-resolution pilot (a resolution domain "
                 "shift on top of a tiny bank). The test set is 4 plates with a handful of defects per class. "
                 "None of these numbers says anything about the method; the full training waits for the paired "
                 "training renders (docs/JEPA.md, data plan).\n")
    L.append(f"- Test grids scored: {res['n_test_grids']} (burst-arm RETAKE skipped: {res['n_skipped_retake']}; "
             f"unreadable: {res.get('n_missing', 0)}); "
             f"test plates {', '.join(res['test_plates'])} (seeds >= {args.min_test_seed}, never trained on).")
    nb = res.get("n_train_by_arm") or {"burst": res["n_train_grids"]}
    fb = [a for a, b in (res.get("bank_for") or {}).items() if a != b]
    L.append(f"- Memory banks (one per arm with training grids): "
             + ", ".join(f"{a} {n} accepted grids" for a, n in nb.items())
             + f" of training plates {', '.join(res['train_plates'])}; defect margin {args.bank_margin_mm} mm; "
             f"coreset {args.coreset}."
             + (f" The {', '.join(fb)} arm had no training grids and is scored against the burst-arm bank "
                "(a domain mismatch by construction: read its rows with that in mind)." if fb else ""))
    L.append(f"- Scales {res['scales']} (1 = 1.28 mm patches, 2 = 2.56 mm), averaged in z units; device {res['device']}; "
             f"{res['seconds']:.0f} s.")
    L.append(f"- Region level: a defect's max score inside its footprint (+{args.pos_margin_mm} mm) vs the max of "
             f"each {args.cell} px ({args.cell * 0.08:.1f} mm) normal cell (>= {args.neg_margin_mm} mm from any defect). "
             f"recall@0.5FA: threshold at 0.5 false-alarm cells per plate (set on the test normals: an operating "
             f"point, not a deployable threshold). recall@z{args.z_thr:g} / FA@z: the fixed threshold z > {args.z_thr:g} "
             "set from the training bank only (deployable). Only defects visible in the C0 reference count.\n")

    def main_rows(tbl: dict[tuple, dict[str, Any]], key_names: Sequence[str]) -> list[str]:
        hdr = "| " + " | ".join(key_names) + " | defects | pixel AUROC | pixel AP | region AUROC | region AP | " \
              f"recall@0.5FA | recall@z{args.z_thr:g} | FA/plate@z | distractor alarms |"
        rows = [hdr, "|" + "---|" * (len(key_names) + 9)]
        for k in sorted(tbl):
            v = tbl[k]
            rows.append("| " + " | ".join(str(x) for x in k) + f" | {v['n_def']} | {_f(v['pixel_auroc'])} | "
                        f"{_f(v['pixel_ap'])} | {_f(v['region_auroc'])} | {_f(v['region_ap'])} | {_f(v['recall_fa05'], 2)} | "
                        f"{_f(v['recall_z'], 2)} | {_f(v['fa_per_plate_z'], 2)} | {_f(v['distractor_alarm_z'], 2)} |")
        return rows

    L.append(f"## 1. Overview by method, arm and device ({tag})\n")
    L += main_rows(t_main, ["method", "arm", "device"])
    L.append(f"\n## 2. Per defect class, region AUROC / pixel AUROC ({tag})\n")
    hdr = "| method | arm | device | " + " | ".join(CLASSES) + " |"
    L += [hdr, "|" + "---|" * (3 + len(CLASSES))]
    for k in sorted(t_main):
        cells_ = []
        for c in CLASSES:
            v = t_cls[c].get(k)
            cells_.append("–" if not v or v["n_def"] == 0 else
                          f"{_f(v['region_auroc'], 2)} / {_f(v['pixel_auroc'], 2)} (n={v['n_def']})")
        L.append("| " + " | ".join(k) + " | " + " | ".join(cells_) + " |")
    L.append(f"\n## 3. Per lighting condition, region AUROC ({tag})\n")
    conds = sorted({k[3] for k in t_cond})
    L += ["| method | arm | device | " + " | ".join(c.split("_")[0] for c in conds) + " |",
          "|" + "---|" * (3 + len(conds))]
    for k in sorted({k[:3] for k in t_cond}):
        L.append("| " + " | ".join(k) + " | " + " | ".join(
            _f((t_cond.get((*k, c)) or {}).get("region_auroc", float("nan")), 2) for c in conds) + " |")
    L.append(f"\n## 4. Per material, region AUROC ({tag})\n")
    mats = sorted({k[2] for k in t_mat})
    L += ["| method | arm | " + " | ".join(mats) + " |", "|" + "---|" * (2 + len(mats))]
    for k in sorted({k[:2] for k in t_mat}):
        L.append("| " + " | ".join(k) + " | " + " | ".join(
            _f((t_mat.get((*k, m)) or {}).get("region_auroc", float("nan")), 2) for m in mats) + " |")
    L.append(f"\n## 5. Lighting invariance vs C0 of the same device ({tag})\n")
    L.append("Scale-1 patch embeddings, centred by the training-bank mean of each extractor. cos same = the patch "
             "vs itself under C0; cos other = vs a random other patch of the window (the floor); retrieval@1 = the "
             "patch's nearest C0 patch in its window (196 candidates) is itself. Accepted captures only "
             "(burst RETAKE rows dropped; the naive arm has no gate). Higher retrieval@1 and gap = more invariant "
             "AND still discriminative.\n")
    conds_i = sorted({k[3] for k in t_inv})
    for metric in ("retrieval_at1", "gap", "cos_same"):
        L.append(f"**{metric}**\n")
        L += ["| extractor | arm | device | " + " | ".join(c.split("_")[0] for c in conds_i) + " | mean |",
              "|" + "---|" * (4 + len(conds_i))]
        for k in sorted({k[:3] for k in t_inv}):
            vals = [(t_inv.get((*k, c)) or {}).get(metric, float("nan")) for c in conds_i]
            fin = [v for v in vals if np.isfinite(v)]
            L.append("| " + " | ".join(k) + " | " + " | ".join(_f(v, 3) for v in vals) +
                     f" | {_f(float(np.mean(fin)) if fin else float('nan'), 3)} |")
        L.append("")
    if figs:
        L.append(f"## 6. Example heat maps ({tag})\n")
        L.append("Left: the canonical burst-arm grid; then one overlay per method (z from -1 to 5, inferno).\n")
        rel = Path(args.out).resolve()
        for p in figs:
            try:
                L.append(f"![{Path(p).stem}]({Path(p).resolve().relative_to(rel).as_posix()})")
            except ValueError:
                L.append(f"![{Path(p).stem}]({p})")
    L.append("\n## Bank statistics\n")
    L.append("```json\n" + json.dumps(res["bank_info"], indent=1) + "\n```")
    md = Path(args.report)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(L) + "\n")

    def jsonable(tbl: dict[tuple, dict[str, Any]]) -> dict[str, Any]:
        return {"|".join(k): {kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv)
                              for kk, vv in v.items()} for k, v in tbl.items()}

    summary = {"smoke": smoke, "label": tag, "methods": methods, "scales": res["scales"],
               "n_test_grids": res["n_test_grids"], "n_skipped_retake": res["n_skipped_retake"],
               "train_plates": res["train_plates"], "test_plates": res["test_plates"], "jepa_step": res["jepa_step"],
               "overview": jsonable(t_main), "per_class": {c: jsonable(t) for c, t in t_cls.items()},
               "per_condition": jsonable(t_cond), "per_material": jsonable(t_mat),
               "invariance_accepted": jsonable(t_inv), "invariance_all": jsonable(t_inv_all),
               "bank": res["bank_info"], "figures": figs, "seconds": res["seconds"],
               "regions": res["regions"]}
    md.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="train.py encoder.pt (omit: baselines only)")
    ap.add_argument("--train", nargs="+", required=True, help="dirs with TRAINING plate grids (memory bank)")
    ap.add_argument("--test", nargs="+", required=True, help="dirs with TEST plate grids (seeds >= 900000)")
    ap.add_argument("--arms", nargs="+", default=["burst", "naive"])
    ap.add_argument("--extractors", nargs="+", default=list(EXTRACTORS))
    ap.add_argument("--no-pred", dest="pred", action="store_false", help="skip the predictor-error method")
    ap.add_argument("--scales", nargs="+", type=float, default=[1, 2])
    ap.add_argument("--coreset", type=int, default=10000)
    ap.add_argument("--bank-margin-mm", type=float, default=2.0)
    ap.add_argument("--pos-margin-mm", type=float, default=0.5)
    ap.add_argument("--neg-margin-mm", type=float, default=2.0)
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--z-thr", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-train-grids", type=int)
    ap.add_argument("--per-grid", type=int, default=1500, help="bank patches kept per training grid and scale")
    ap.add_argument("--bank-cache", help="save / reuse the memory banks here (keyed by ckpt, grids and settings)")
    ap.add_argument("--min-test-seed", type=int, default=900100,
                    help="test plates from this seed up (the capture gate was tuned on 900000-900068; the pilot "
                         "smoke used 900000 explicitly)")
    ap.add_argument("--max-test-plates", type=int)
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="label every output SMOKE, NOT A RESULT")
    ap.add_argument("--report", default="reports/jepa_anomaly.md")
    ap.add_argument("--fig-dir", default="reports/figures/jepa")
    ap.add_argument("--heat-dir", default="runs/jepa/heat")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)
    res = evaluate(a)
    s = write_report(res, a)
    print(f"wrote {a.report} ({'SMOKE' if a.smoke else 'result'})")
    return s


if __name__ == "__main__":
    main()
