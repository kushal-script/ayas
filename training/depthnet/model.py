"""Depth network: one defect crop (left/right LED frames + geometry) -> height map (mm).

Input (N, 7, 48, 48), every channel in [0, 1] and quantised to 8 bits before the network sees it
(the Hailo-10H takes uint8; training on the same quantised input avoids a quantisation gap):
  0, 1  log-scaled L and R, normalised by the crop's flat ring (see prepare)
  2     valid mask (the crop is padded to 48 px)
  3     illumination: 1 = hood ring, 0 = open pods
  4, 5  crop centre relative to the camera axis, u and v (mm / 200 + 0.5)
  6     working spacing (mm / 0.8)
Output (N, 1, 48, 48): height in units of 0.1 mm, relative to the crop's flat ring.

Only convolutions, ReLU, max-pool, nearest upsampling and concatenation: all native on the Hailo.
"""
from __future__ import annotations

import numpy as np

N = 48
H_UNIT_MM = 0.1


def prepare(obs: np.ndarray, valid: np.ndarray, illumination: str, rel_u_mm: float, rel_v_mm: float,
            s_mm: float) -> np.ndarray:
    """(2, n, n) L/R in FS (n <= 48) -> (7, 48, 48) float32 in [0, 1], quantised to 1/255."""
    n = obs.shape[-1]
    o = np.zeros((2, N, N), np.float32)
    o[:, :n, :n] = obs[:, :n, :n]
    v = np.zeros((N, N), np.float32)
    v[:n, :n] = valid[:n, :n]
    ring = np.zeros((N, N), bool)
    k = max(2, n // 8)
    ring[:n, :n] = True
    ring[k:n - k, k:n - k] = False
    ref = float(np.median((o[0] + o[1])[ring & (v > 0)])) if (ring & (v > 0)).any() else float(np.median(o[0] + o[1]))
    ref = max(ref / 2, 1e-5)
    x = np.zeros((7, N, N), np.float32)
    for c in range(2):
        x[c] = np.clip((np.log2(np.maximum(o[c], 1e-6) / ref) + 6.0) / 12.0, 0, 1) * v
    x[2] = v
    x[3] = 1.0 if illumination == "ring" else 0.0
    x[4] = np.clip(rel_u_mm / 200.0 + 0.5, 0, 1)
    x[5] = np.clip(rel_v_mm / 200.0 + 0.5, 0, 1)
    x[6] = np.clip(s_mm / 0.8, 0, 1)
    return np.round(x * 255.0) / 255.0


def build(width: int = 24):
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.e1, self.e2, self.e3 = block(7, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.mid = block(4 * w, 8 * w)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.d3, self.d2, self.d1 = block(12 * w, 4 * w), block(6 * w, 2 * w), block(3 * w, w)
            self.out = nn.Conv2d(w, 1, 1)

        def forward(self, x):
            a = self.e1(x)                       # 48
            b = self.e2(self.pool(a))            # 24
            c = self.e3(self.pool(b))            # 12
            m = self.mid(self.pool(c))           # 6
            c = self.d3(torch.cat([self.up(m), c], 1))
            b = self.d2(torch.cat([self.up(c), b], 1))
            a = self.d1(torch.cat([self.up(b), a], 1))
            return self.out(a)

    return UNet(width)
