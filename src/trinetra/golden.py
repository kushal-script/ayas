"""Golden reference: per-pixel mean and spread of every map over many good parts.

A single golden image would flag every harmless variation (brushing marks,
placement, LED drift) as a defect; the spread tells us what "normal" looks like.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .align import prepare
from .calibration import Calibration
from .config import Config
from .shape import shape_setup


@dataclass
class Golden:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    valid: np.ndarray
    medians: dict[str, float]  # plate median of each image map, for LED drift correction
    count: int

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.mean)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"mean__{k}": self.mean[k].astype(np.float16) for k in self.keys}
        arrays.update({f"std__{k}": self.std[k].astype(np.float16) for k in self.keys})
        np.savez_compressed(
            path,
            valid=self.valid,
            count=self.count,
            keys=json.dumps(list(self.keys)),
            medians=json.dumps(self.medians),
            **arrays,
        )

    @classmethod
    def load(cls, path: Path) -> "Golden":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `trinetra golden` first")
        with np.load(path) as data:
            if "keys" not in data:
                raise ValueError(f"{path} is from an older rig layout - rebuild it with `trinetra golden`")
            keys = json.loads(str(data["keys"]))
            return cls(
                mean={k: data[f"mean__{k}"].astype(np.float32) for k in keys},
                std={k: data[f"std__{k}"].astype(np.float32) for k in keys},
                valid=data["valid"].astype(bool),
                medians=json.loads(str(data["medians"])),
                count=int(data["count"]),
            )


class GoldenBuilder:
    """Streams good parts in (Welford running mean/variance), so memory stays flat."""

    def __init__(self, cfg: Config, cal: Calibration):
        self.cfg, self.cal = cfg, cal
        self.grid = cfg.grid
        self.setup = shape_setup(cfg)
        self.roi = self.grid.plate_mask(self.grid.edge_margin_mm)
        self.n = 0
        self._mean: dict[str, np.ndarray] = {}
        self._m2: dict[str, np.ndarray] = {}
        self._valid = np.ones(self.grid.shape, bool)
        self._reference: np.ndarray | None = None
        self._medians: dict[str, float] | None = None

    def add(self, frames: Mapping[str, np.ndarray]) -> None:
        stack, valid = prepare(frames, self.cal, self.cfg, self._reference, self._medians, self.setup)
        if self._reference is None:
            self._reference = stack[self.cfg.align_key].copy()
            self._medians = {m.key: float(np.median(stack[m.key][self.roi])) for m in self.cfg.image_maps()}
        self.n += 1
        for k, img in stack.items():
            x = img.astype(np.float64)
            if k not in self._mean:
                self._mean[k] = x.copy()
                self._m2[k] = np.zeros_like(x)
                continue
            delta = x - self._mean[k]
            self._mean[k] += delta / self.n
            self._m2[k] += delta * (x - self._mean[k])
        self._valid &= valid

    def finalize(self) -> Golden:
        if self.n < 2:
            raise ValueError("need at least 2 good parts for a golden reference (20+ recommended)")
        return Golden(
            mean={k: v.astype(np.float32) for k, v in self._mean.items()},
            std={k: np.sqrt(self._m2[k] / (self.n - 1)).astype(np.float32) for k in self._mean},
            valid=self._valid,
            medians=self._medians or {},
            count=self.n,
        )
