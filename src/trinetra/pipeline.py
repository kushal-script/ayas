"""Frames in (each light + dark, three cameras) -> inspection report out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .align import prepare
from .calibration import Calibration
from .classify import Prediction, load_classifier
from .config import Config
from .detect import Anomalies, find_anomalies
from .features import MapRoles, RegionFeatures, region_features
from .golden import Golden
from .report import build_report, defect_entry
from .rules import judge_part, load_spec
from .shape import height_map, shape_setup

DISPOSITION_BGR = {
    "PASS": (80, 200, 80),
    "CLEAN_AND_REINSPECT": (0, 220, 255),
    "REWORK": (0, 140, 255),
    "HOLD_FOR_QA": (220, 80, 220),
    "REJECT": (40, 40, 255),
}


@dataclass
class Analysis:
    stack: dict[str, np.ndarray]
    anomalies: Anomalies
    features: list[RegionFeatures]
    predictions: list[Prediction]
    height: np.ndarray | None
    timing_ms: dict[str, float] = field(default_factory=dict)


class Inspector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.spec = load_spec(cfg.spec_path)
        self.calibration = Calibration.load(cfg.calibration_path)
        self.golden = Golden.load(cfg.golden_path)
        self.classifier = load_classifier(cfg.classifier_path)
        self.setup = shape_setup(cfg)
        self.roles = MapRoles.from_config(cfg)
        self.roi = cfg.grid.plate_mask(cfg.grid.edge_margin_mm)
        expected = {m.key for m in cfg.image_maps()} | set(self.setup.map_keys)
        if expected != set(self.golden.keys):
            raise ValueError("golden reference was built with a different light layout - run `trinetra golden` again")

    def analyse(self, frames: Mapping[str, np.ndarray]) -> Analysis:
        grid, t = self.cfg.grid, {}
        t0 = time.perf_counter()
        stack, valid = prepare(
            frames, self.calibration, self.cfg, self.golden.mean[self.cfg.align_key], self.golden.medians, self.setup
        )
        t1 = time.perf_counter()
        height = height_map(stack, self.golden.mean, self.roi & valid, grid.mm_per_px)
        t2 = time.perf_counter()
        anomalies = find_anomalies(stack, valid, self.golden, grid, self.cfg.detection)
        t3 = time.perf_counter()
        feats = [region_features(r, anomalies, grid, self.cfg, self.roles, height) for r in anomalies.regions]
        preds = [self.classifier.predict(f) for f in feats]
        t4 = time.perf_counter()
        t.update(align=(t1 - t0) * 1e3, shape3d=(t2 - t1) * 1e3, detect=(t3 - t2) * 1e3, classify=(t4 - t3) * 1e3)
        return Analysis(stack, anomalies, feats, preds, height, t)

    def inspect(
        self,
        frames: Mapping[str, np.ndarray],
        part_id: str,
        capture_dir: Path | None = None,
        overlay_path: Path | None = None,
        extra_timing_ms: dict[str, float] | None = None,
    ) -> tuple[dict[str, Any], Analysis]:
        analysis = self.analyse(frames)
        t0 = time.perf_counter()
        entries = [defect_entry(f, p) for f, p in zip(analysis.features, analysis.predictions)]
        verdict, judged = judge_part(entries, self.spec)
        timing = {**(extra_timing_ms or {}), **analysis.timing_ms, "rules": (time.perf_counter() - t0) * 1e3}
        if overlay_path is not None:
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), self.overlay(analysis, judged))
        report = build_report(
            part_id,
            verdict,
            judged,
            self.cfg,
            self.spec,
            self.classifier.name,
            self.golden.count,
            timing,
            capture_dir=str(capture_dir) if capture_dir else None,
            overlay_path=str(overlay_path) if overlay_path else None,
            shape=self.setup.describe(),
        )
        return report, analysis

    def overlay(self, analysis: Analysis, judged: list[dict[str, Any]]) -> np.ndarray:
        """Top view in natural colour with defect boxes; the 3D height map beside it when available."""
        r, g, b = (analysis.stack[k] for k in self.roles.top_rgb)
        scale = 0.5 / max(float(np.median(self.golden.mean[self.roles.top_rgb[1]][self.roi])), 1e-3)
        img = np.clip(np.dstack([b, g, r]) * scale * 255, 0, 255).astype(np.uint8)
        for region, d in zip(analysis.anomalies.regions, judged):
            x0, y0, x1, y1 = region.bbox
            color = DISPOSITION_BGR[d["disposition"]]
            pad = 6
            cv2.rectangle(img, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), color, 2)
            label = f"{d['id']} {d['type']}"
            if d.get("depth_mm") is not None and d["type"] in ("dent", "bump"):
                label += f" {d['depth_mm']:+.2f}mm"
            cv2.putText(img, label, (x0 - pad, max(y0 - pad - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if analysis.height is None:
            return img
        h = np.where(self.roi, analysis.height, 0.0)
        lim = max(float(np.percentile(np.abs(h[self.roi]), 99.9)), 0.02)
        h8 = np.clip(127.5 + 127.5 * h / lim, 0, 255).astype(np.uint8)
        colour = cv2.applyColorMap(h8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        cv2.putText(colour, f"3D height, +/-{lim:.2f} mm", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.hstack([img, colour])
