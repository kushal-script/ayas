"""I-JEPA multi-block masking (Assran et al. 2023, sec. 3) on the ViT patch grid (numpy only).

Per image: M = 4 target blocks (scale 0.15-0.2 of the patches, aspect 0.75-1.5) and one context
block (scale 0.85-1.0, aspect 1) with every target patch removed from it, so the context is
informative but never sees the targets. A batch needs equal lengths, so each target block j and
the context are cut to the batch minimum (a random subset of the context, not the first k
patches, to avoid a top-left bias). The four blocks' indices are concatenated: the predictor
predicts all of them in one pass (mask tokens may attend to each other; they carry only their
position, so this is a cost-saving simplification of the per-block passes in the paper).

Cross-lighting mode adds `n_visible` context-visible positions to the targets: the predictor must
also map the context (condition X) to the anchor condition's embedding at patches it can see,
which is the direct lighting-invariance term (docs/JEPA.md).

`checker_groups` gives the fixed masks of the predictor-error anomaly score: G groups of b x b
patch blocks in a checkerboard; each patch is masked in exactly one group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MaskCfg:
    grid: tuple[int, int] = (14, 14)
    n_targets: int = 4
    target_scale: tuple[float, float] = (0.15, 0.2)
    target_aspect: tuple[float, float] = (0.75, 1.5)
    ctx_scale: tuple[float, float] = (0.85, 1.0)
    ctx_aspect: tuple[float, float] = (1.0, 1.0)
    min_keep: int = 10
    n_visible: int = 0
    tries: int = 20

    @classmethod
    def from_dict(cls, d: dict | None, grid: tuple[int, int] = (14, 14)) -> "MaskCfg":
        d = dict(d or {})
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in cls.__dataclass_fields__}
        kw.setdefault("grid", tuple(grid))
        return cls(**kw)


def sample_block(rng: np.random.Generator, grid: tuple[int, int], scale: Sequence[float],
                 aspect: Sequence[float]) -> np.ndarray:
    """One rectangular block of patches (H x W bool)."""
    H, W = grid
    s = rng.uniform(*scale)
    n = max(1, int(round(H * W * s)))
    lo, hi = math.log(aspect[0]), math.log(aspect[1])
    ar = math.exp(rng.uniform(lo, hi)) if hi > lo else aspect[0]
    h = int(round(math.sqrt(n * ar)))
    w = int(round(math.sqrt(n / ar)))
    h, w = max(1, min(h, H)), max(1, min(w, W))
    top = int(rng.integers(0, H - h + 1))
    left = int(rng.integers(0, W - w + 1))
    m = np.zeros((H, W), bool)
    m[top:top + h, left:left + w] = True
    return m


def sample_masks(rng: np.random.Generator, cfg: MaskCfg) -> tuple[list[np.ndarray], np.ndarray]:
    """(target index arrays, context index array) for one image; the context excludes the targets."""
    H, W = cfg.grid
    for _ in range(cfg.tries):
        tgts = [sample_block(rng, cfg.grid, cfg.target_scale, cfg.target_aspect) for _ in range(cfg.n_targets)]
        ctx = sample_block(rng, cfg.grid, cfg.ctx_scale, cfg.ctx_aspect)
        union = np.zeros((H, W), bool)
        for t in tgts:
            union |= t
        ctx &= ~union
        if ctx.sum() >= cfg.min_keep:
            break
    else:                                   # fall back: everything not targeted
        ctx = ~union
    return [np.flatnonzero(t.ravel()) for t in tgts], np.flatnonzero(ctx.ravel())


def collate_masks(rng: np.random.Generator, cfg: MaskCfg, batch: int) -> dict[str, np.ndarray]:
    """Equal-length index arrays for a batch.

    Returns ctx (B, Lc), tgt (B, Lt) [the M blocks concatenated, then n_visible context
    positions], vis (B, Lt) bool (True where the target position is visible in the context).
    """
    per = [sample_masks(rng, cfg) for _ in range(batch)]
    lt = [min(len(p[0][j]) for p in per) for j in range(cfg.n_targets)]
    lc = min(len(p[1]) for p in per)
    nv = min(int(cfg.n_visible), lc)
    ctx, tgt, vis = [], [], []
    for tg, cx in per:
        c = np.sort(rng.choice(cx, lc, replace=False)) if len(cx) > lc else cx
        blocks = [np.sort(rng.choice(t, lt[j], replace=False)) if len(t) > lt[j] else t for j, t in enumerate(tg)]
        t = np.concatenate(blocks) if blocks else np.zeros(0, int)
        v = np.zeros(len(t), bool)
        if nv:
            extra = np.sort(rng.choice(c, nv, replace=False))
            t = np.concatenate([t, extra])
            v = np.concatenate([v, np.ones(nv, bool)])
        ctx.append(c)
        tgt.append(t)
        vis.append(v)
    return {"ctx": np.stack(ctx).astype(np.int64), "tgt": np.stack(tgt).astype(np.int64),
            "vis": np.stack(vis)}


def checker_groups(grid: tuple[int, int] = (14, 14), block: int = 2) -> list[np.ndarray]:
    """Fixed predictor-error masks: 4 groups of block x block patch squares in a checkerboard of
    period 2 * block; group g masks the squares with (row // block % 2, col // block % 2) == g.
    Every patch is masked in exactly one group; each masked square keeps visible neighbours."""
    H, W = grid
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    key = ((r // block) % 2) * 2 + (c // block) % 2
    return [np.flatnonzero((key == g).ravel()) for g in range(4) if (key == g).any()]
