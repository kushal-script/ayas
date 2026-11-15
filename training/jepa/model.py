"""I-JEPA model: timm ViT context encoder, EMA target encoder, narrow ViT predictor.

- Encoder: timm `vit_tiny_patch16_224` (192-d, 12 blocks, 5.5 M params) by default, `vit_small_patch16_224`
  (384-d, 22 M) as an option. The class token is never used: patch tokens + their position
  embeddings go through the blocks, so the encoder can run on any subset of patches (the
  I-JEPA context) and exports to ONNX as a plain ViT for the full image.
- Init: `imagenet` (timm pretrained weights, then JEPA) or `scratch`. With a few hundred twin
  plates, ImageNet init is the practical default; the ImageNet model itself is the baseline the
  JEPA must beat (eval_anomaly.py), so the comparison isolates what the JEPA objective adds.
- Predictor: Linear(D -> Dp), learned mask token, fixed 2-D sin-cos positions, `pred_depth` timm
  Blocks of width Dp (= D / 2 by default), LayerNorm, Linear(Dp -> D).
- Targets: LayerNorm(target_encoder(anchor image)) at the target positions (I-JEPA), smooth-L1 loss.
- EMA momentum: linear from ema[0] to ema[1] over training.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ARCHS = {"vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
         "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k"}


def sincos_2d(dim: int, grid: tuple[int, int]) -> torch.Tensor:
    """(1, H*W, dim) fixed 2-D sin-cos position embedding (MAE / I-JEPA)."""
    H, W = grid
    gy, gx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")

    def one(d: int, pos: np.ndarray) -> np.ndarray:
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate([one(dim // 2, gy), one(dim // 2, gx)], axis=1)
    return torch.from_numpy(emb).float().unsqueeze(0)


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B, N, D), idx (B, K) -> (B, K, D)."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class Encoder(nn.Module):
    """timm ViT that runs on a subset of patch tokens (no class token)."""

    def __init__(self, arch: str = "vit_tiny", init: str = "imagenet", img_size: int = 224,
                 in_chans: int = 3, drop_path: float = 0.0):
        super().__init__()
        import timm

        name = ARCHS.get(arch, arch)
        self.vit = timm.create_model(name, pretrained=(init == "imagenet"), num_classes=0, img_size=img_size,
                                     in_chans=in_chans, drop_path_rate=drop_path)
        self.embed_dim = int(self.vit.embed_dim)
        self.grid = tuple(int(g) for g in self.vit.patch_embed.grid_size)
        self.num_patches = self.grid[0] * self.grid[1]
        self.n_prefix = 0 if getattr(self.vit, "no_embed_class", False) else int(self.vit.num_prefix_tokens)

    def patch_pos(self) -> torch.Tensor:
        return self.vit.pos_embed[:, self.n_prefix:self.n_prefix + self.num_patches]

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        t = self.vit.patch_embed(x)
        if t.ndim == 4:                                    # NHWC output (some timm configs)
            t = t.flatten(1, 2)
        t = t + self.patch_pos()
        if keep is not None:
            t = gather_tokens(t, keep)
        t = self.vit.norm_pre(t)
        for blk in self.vit.blocks:
            t = blk(t)
        return self.vit.norm(t)


class Predictor(nn.Module):
    def __init__(self, embed_dim: int, grid: tuple[int, int], pred_dim: int | None = None, depth: int = 4,
                 heads: int = 3, mlp_ratio: float = 4.0):
        super().__init__()
        from timm.models.vision_transformer import Block

        pred_dim = int(pred_dim or embed_dim // 2)
        self.embed = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sincos_2d(pred_dim, grid), persistent=False)
        self.blocks = nn.ModuleList([Block(pred_dim, heads, mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.proj = nn.Linear(pred_dim, embed_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ctx: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
        B = ctx.shape[0]
        pos = self.pos.expand(B, -1, -1)
        x = self.embed(ctx) + gather_tokens(pos, ctx_idx)
        m = self.mask_token.expand(B, tgt_idx.shape[1], -1) + gather_tokens(pos, tgt_idx)
        z = torch.cat([x, m], dim=1)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z[:, x.shape[1]:])
        return self.proj(z)


class JEPA(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        m = cfg.get("model") or {}
        self.encoder = Encoder(m.get("arch", "vit_tiny"), m.get("init", "imagenet"), int(m.get("img_size", 224)),
                               int(m.get("in_chans", 3)), float(m.get("drop_path", 0.0)))
        D = self.encoder.embed_dim
        self.predictor = Predictor(D, self.encoder.grid, m.get("pred_dim"), int(m.get("pred_depth", 4)),
                                   int(m.get("pred_heads", 3)))
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.vis_weight = float((cfg.get("train") or {}).get("vis_weight", 0.5))

    @torch.no_grad()
    def targets(self, img: torch.Tensor) -> torch.Tensor:
        h = self.target_encoder(img)
        return F.layer_norm(h, (h.shape[-1],))

    def forward(self, ctx_img: torch.Tensor, tgt_img: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor,
                vis: torch.Tensor | None = None, cross: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """vis (B, Lt) bool: target position visible in the context; cross (B,) bool: the target image
        is another lighting condition. Hidden targets weigh 1; visible ones weigh vis_weight on
        CROSS pairs (the direct invariance term) and 0 on SELF pairs (identity, no signal)."""
        h = gather_tokens(self.targets(tgt_img), tgt_idx)
        z = self.encoder(ctx_img, ctx_idx)
        p = self.predictor(z, ctx_idx, tgt_idx)
        err = F.smooth_l1_loss(p.float(), h.float(), reduction="none").mean(-1)       # (B, Lt)
        w = torch.ones_like(err)
        shown = err.new_zeros(())
        hidden = torch.ones_like(err, dtype=torch.bool)
        if vis is not None and bool(vis.any()):
            vc = vis & (cross[:, None] if cross is not None else torch.ones_like(vis))
            w = torch.where(vis, torch.where(vc, torch.full_like(err, self.vis_weight), torch.zeros_like(err)), w)
            hidden = ~vis
            if bool(vc.any()):
                shown = err[vc].mean()
        hid = err[hidden].mean()
        loss = (err * w).sum() / w.sum().clamp_min(1e-6)
        with torch.no_grad():
            zs = F.normalize(h.float().mean(1), dim=-1)
            spread = zs.std(0).mean()                       # collapse monitor (-> 0 if collapsed)
        return {"loss": loss, "loss_hidden": hid.detach(), "loss_visible": shown.detach(), "tgt_spread": spread}

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for pt, pc in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(momentum).add_(pc.detach(), alpha=1.0 - momentum)


def ema_momentum(step: int, total: int, ema: tuple[float, float]) -> float:
    return float(ema[0] + (ema[1] - ema[0]) * min(1.0, step / max(1, total)))


def lr_at(step: int, total: int, lr: float, warmup: int, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, t)))


def load_jepa(ckpt_path: str, map_location: str = "cpu") -> tuple["JEPA", dict[str, Any]]:
    """A trained JEPA from a train.py checkpoint (weights only; no download: init is forced to scratch)."""
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = copy.deepcopy(ck["cfg"])
    cfg.setdefault("model", {})["init"] = "scratch"
    model = JEPA(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
