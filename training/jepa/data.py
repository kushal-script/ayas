"""Torch datasets of plate-grid crops for I-JEPA, with cross-lighting pairs.

Each item is drawn deterministically from (seed, index), so a run is reproducible whatever the
number of loader workers:
1. a pairing group: one plate, one device mode (open | hood), one post variant k, one arm (burst);
   only rows whose gate is not RETAKE (the variant gate: a refused capture never reaches the app);
2. a scale (1 = 224 px at 0.08 mm/px = 17.9 mm; 2 = 448 px downsampled to 224 = 35.8 mm) and a crop
   whose centre lies on the plate;
3. with probability cross_prob a CROSS pair: context = the crop under a condition X, target = the
   same canonical pixels under the anchor (C0 of the same device; `anchor: any` = another
   accepted condition); otherwise a SELF pair (standard I-JEPA: context and target are one image);
4. the same flip / 90-degree turn on both images; photometric jitter (gain, gamma, smooth shading,
   noise) on the context only, as the last line of lighting robustness (DESIGN_STEEL, item 6).

Images are RGB, normalised with the ImageNet mean/std (docs/JEPA.md: why RGB).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import grids as G
from .masks import MaskCfg, collate_masks
from .model import IMAGENET_MEAN, IMAGENET_STD


class GridCache:
    """LRU cache of canonical uint8 grids (a 100 x 100 mm grid is 5.7 MB)."""

    def __init__(self, capacity: int = 64):
        self.cap = int(capacity)
        self.d: OrderedDict[str, np.ndarray] = OrderedDict()
        self.meta: dict[str, dict[str, Any]] = {}

    def get(self, rec: G.GridRec) -> np.ndarray:
        key = str(rec.image)
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        meta = self.meta.get(key) or rec.meta()
        self.meta[key] = meta
        img = G.load_canonical(rec, meta)
        if self.cap > 0:
            self.d[key] = img
            while len(self.d) > self.cap:
                self.d.popitem(last=False)
        return img

    def get_meta(self, rec: G.GridRec) -> dict[str, Any]:
        key = str(rec.image)
        if key not in self.meta:
            self.meta[key] = rec.meta()
        return self.meta[key]


def crop_at(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """size x size crop with top-left (x, y); zero padding outside the image."""
    h, w = img.shape[:2]
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + size), min(h, y + size)
    if x1 > x0 and y1 > y0:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = img[y0:y1, x0:x1]
    return out


def to_input(crop_u8: np.ndarray, out: int = 224) -> np.ndarray:
    """uint8 crop (any size) -> float32 [0, 1] at out x out (INTER_AREA when downsampling)."""
    if crop_u8.shape[0] != out:
        crop_u8 = cv2.resize(crop_u8, (out, out), interpolation=cv2.INTER_AREA)
    return crop_u8.astype(np.float32) / 255.0


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)


def photometric(rng: np.random.Generator, x: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Lighting jitter on a float [0, 1] HxWx3 image."""
    if rng.random() >= float(p.get("p", 0.0)):
        return x
    g0, g1 = p.get("gain", [0.6, 1.5])
    x = x * rng.uniform(g0, g1)
    if p.get("shading", 0.0):
        s = float(p["shading"])
        field = rng.uniform(1 - s, 1 + s, (3, 3)).astype(np.float32)
        x = x * cv2.resize(field, x.shape[1::-1], interpolation=cv2.INTER_CUBIC)[..., None]
    x = np.clip(x, 0, 1)
    gm0, gm1 = p.get("gamma", [0.7, 1.4])
    x = x ** rng.uniform(gm0, gm1)
    n = rng.uniform(0, float(p.get("noise", 0.02)))
    if n > 0:
        x = x + rng.normal(0, n, x.shape).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def geometric(x: np.ndarray, flip_h: bool, flip_v: bool, k: int) -> np.ndarray:
    if flip_h:
        x = x[:, ::-1]
    if flip_v:
        x = x[::-1]
    if k:
        x = np.rot90(x, k)
    return np.ascontiguousarray(x)


class PairCrops(Dataset):
    """Cross-lighting / self crops from accepted grids of TRAINING plates."""

    def __init__(self, recs: Sequence[G.GridRec], cfg: dict[str, Any], length: int, seed: int = 0,
                 allow_test: bool = False):
        d = cfg.get("data") or {}
        recs = [r for r in recs if r.arm in (d.get("arms") or ["burst"])]
        if d.get("device_modes"):
            recs = [r for r in recs if r.device_mode in d["device_modes"]]
        if d.get("skip_retake", True):
            recs = [r for r in recs if r.accepted]
        if not allow_test:
            G.refuse_test_plates(recs)
        self.groups = [v for _, v in sorted(G.groups(recs).items())]
        if not self.groups:
            raise ValueError("no accepted grids to train on")
        by_plate: dict[tuple, list[G.GridRec]] = {}
        for r in recs:
            by_plate.setdefault((r.plate, r.k, r.arm), []).append(r)
        self.by_plate = by_plate
        self.cfg, self.seed, self.length = cfg, int(seed), int(length)
        self.crop = int(d.get("crop", 224))
        self.scales = [float(s) for s in d.get("scales", [1, 2])]
        sp = d.get("scale_probs") or [1.0 / len(self.scales)] * len(self.scales)
        self.scale_p = np.asarray(sp, float) / float(np.sum(sp))
        pair = d.get("pair") or {}
        self.cross_prob = float(pair.get("cross_prob", 0.75))
        self.anchor = str(pair.get("anchor", "C0"))
        self.scope = str(pair.get("scope", "device"))
        self.aug = d.get("aug") or {}
        self.cache = GridCache(int(d.get("cache_grids", 64)))
        self.cpg = max(1, int(d.get("crops_per_group", 1)))

    def __len__(self) -> int:
        return self.length

    def _anchor(self, rng: np.random.Generator, rows: list[G.GridRec], ctx: G.GridRec) -> G.GridRec:
        pool = rows
        if self.scope == "all":
            pool = [r for r in self.by_plate[(ctx.plate, ctx.k, ctx.arm)]]
        if self.anchor == "C0":
            c0 = [r for r in pool if r.condition.startswith("C0") and r.device_mode == ctx.device_mode] \
                if self.scope != "all" else [r for r in pool if r.condition.startswith("C0")]
            c0 = [r for r in c0 if r.image != ctx.image]
            if c0:
                return c0[int(rng.integers(len(c0)))]
        others = [r for r in pool if r.image != ctx.image]
        return others[int(rng.integers(len(others)))] if others else ctx

    def __getitem__(self, i: int) -> dict[str, Any]:
        rng = np.random.default_rng([self.seed, 1, int(i)])
        # crops_per_group consecutive indices share a pairing group, so a loader worker (which builds
        # whole batches of consecutive indices) decodes few grids per batch: cache hits, no I/O bound
        g_rng = np.random.default_rng([self.seed, 3, int(i) // self.cpg])
        rows = self.groups[int(g_rng.integers(len(self.groups)))]
        cross = rng.random() < self.cross_prob and len(rows) > 1
        if cross:
            nonc0 = [r for r in rows if not r.condition.startswith("C0")] or rows
            ctx_rec = nonc0[int(rng.integers(len(nonc0)))]
            tgt_rec = self._anchor(rng, rows, ctx_rec)
            cross = tgt_rec.image != ctx_rec.image
        else:
            ctx_rec = tgt_rec = rows[int(rng.integers(len(rows)))]
        scale = float(rng.choice(self.scales, p=self.scale_p))
        size = int(round(self.crop * scale))
        meta = self.cache.get_meta(ctx_rec)
        x0, y0, x1, y1 = G.plate_rect(meta)
        cx, cy = int(rng.integers(x0, max(x0 + 1, x1))), int(rng.integers(y0, max(y0 + 1, y1)))
        x, y = cx - size // 2, cy - size // 2
        a = to_input(crop_at(self.cache.get(ctx_rec), x, y, size), self.crop)
        b = a if tgt_rec is ctx_rec else to_input(crop_at(self.cache.get(tgt_rec), x, y, size), self.crop)
        fh = bool(self.aug.get("flip", True)) and rng.random() < 0.5
        fv = bool(self.aug.get("flip", True)) and rng.random() < 0.5
        k = int(rng.integers(4)) if self.aug.get("rot90", True) else 0
        a, b = geometric(a, fh, fv, k), geometric(b, fh, fv, k)
        a = photometric(rng, a, self.aug.get("photometric") or {})
        return {"ctx": torch.from_numpy(normalize(a).transpose(2, 0, 1).copy()),
                "tgt": torch.from_numpy(normalize(b).transpose(2, 0, 1).copy()),
                "cross": bool(cross), "scale": scale, "index": int(i),
                "dt": f"{ctx_rec.tag}->{tgt_rec.tag}"}


class MaskCollator:
    def __init__(self, mcfg: MaskCfg, seed: int = 0):
        self.mcfg, self.seed = mcfg, int(seed)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        rng = np.random.default_rng([self.seed, 2, int(items[0]["index"])])
        m = collate_masks(rng, self.mcfg, len(items))
        cross = torch.tensor([it["cross"] for it in items])
        vis = torch.from_numpy(m["vis"])
        # the visible-position term only makes sense across lighting: on SELF pairs it is the identity
        vis_w = vis & cross[:, None]
        return {"ctx": torch.stack([it["ctx"] for it in items]), "tgt": torch.stack([it["tgt"] for it in items]),
                "ctx_idx": torch.from_numpy(m["ctx"]), "tgt_idx": torch.from_numpy(m["tgt"]), "vis": vis,
                "vis_cross": vis_w, "cross": cross, "scale": torch.tensor([it["scale"] for it in items])}
