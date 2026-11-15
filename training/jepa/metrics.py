"""Anomaly and invariance metrics (numpy only).

- auroc / average_precision: rank-based, ties averaged; NaN when a class is missing.
- pixel level: per defect class, positives = the class's labelled-defect pixels (visible in the
  C0 reference), negatives = plate pixels outside every defect dilated by a margin; the margin
  ring and invisible defects are ignored.
- region level: positives = each labelled defect's max score inside its footprint (dilated by
  `pos_margin`); negatives = the max score of each `cell` x `cell` px cell of the plate that is
  fully outside every defect dilated by `neg_margin` (a 64 px cell = 5.1 mm, about the size of
  the smallest box an operator would check).
- recall at a false-alarm budget: the threshold is the (1 - budget / cells_per_plate) quantile of
  the normal-cell scores (budget = false-alarm cells per plate), and recall is the share of
  defects whose region score exceeds it.
- invariance: per patch, the cosine similarity of its (dataset-mean centred) embedding under
  condition X to the same patch under C0; the same for a random other patch of the window
  (the floor: a constant encoder is perfectly 'invariant' and useless); and the share of patches
  whose nearest C0 patch in the window is itself (retrieval@1: invariant AND discriminative).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties get their average rank."""
    order = np.argsort(x, kind="mergesort")
    _, first, counts = np.unique(x[order], return_index=True, return_counts=True)
    r = np.empty(len(x), float)
    r[order] = np.repeat(first + (counts - 1) / 2.0 + 1.0, counts)
    return r


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float).ravel(), np.asarray(neg, float).ravel()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _ranks(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float).ravel(), np.asarray(neg, float).ravel()
    if len(pos) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / len(pos))


def subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x).ravel()
    return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]


def normal_cells(heat: np.ndarray, plate: np.ndarray, excl: np.ndarray, cell: int = 64) -> np.ndarray:
    """Max score of each cell x cell block fully on the plate and outside `excl`."""
    h, w = heat.shape
    out = []
    for y in range(0, h - cell + 1, cell):
        for x in range(0, w - cell + 1, cell):
            if plate[y:y + cell, x:x + cell].all() and not excl[y:y + cell, x:x + cell].any():
                out.append(float(heat[y:y + cell, x:x + cell].max()))
    return np.asarray(out, float)


def recall_at_budget(pos: np.ndarray, neg: np.ndarray, cells_per_plate: float, budget: float = 0.5) -> tuple[float, float]:
    """(recall, threshold) at `budget` false-alarm cells per plate."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan")
    q = max(0.0, 1.0 - budget / max(cells_per_plate, 1e-9))
    thr = float(np.quantile(neg, q))
    return float((pos > thr).mean()), thr


def summarize_scores(pos: Sequence[float], neg: Sequence[float]) -> dict[str, Any]:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "auroc": auroc(pos, neg),
            "ap": average_precision(pos, neg)}


def invariance_stats(ex: np.ndarray, e0: np.ndarray, mean: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """ex, e0: (N, P, D) embeddings of N windows x P patches under condition X and C0."""
    a = ex - mean
    b = e0 - mean
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    same = (a * b).sum(-1)                                   # (N, P)
    perm = rng.permutation(a.shape[1])
    other = (a * b[:, perm]).sum(-1)
    sim = np.einsum("npd,nqd->npq", a, b)
    r1 = (sim.argmax(-1) == np.arange(a.shape[1])[None]).mean()
    return {"cos_same": float(same.mean()), "cos_other": float(other.mean()),
            "gap": float(same.mean() - other.mean()), "retrieval_at1": float(r1)}
