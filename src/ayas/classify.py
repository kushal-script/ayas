"""Defect type from region features.

Two classifiers share one interface:
- SignatureClassifier: explicit rules from the view/light/colour/3D signature. Works from
  day one, before any labelled data, and every decision has a readable reason.
- ForestClassifier: a random forest trained on labelled regions from the rig
  (`ayas train`). Used automatically once a model file exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import FEATURE_NAMES, RegionFeatures

CLASSES = ("scratch", "dent", "bump", "crack", "pit", "stain", "dust", "unknown")


@dataclass
class Prediction:
    label: str
    confidence: float
    basis: str


def _conf(margin: float) -> float:
    """Map how far past its threshold the deciding feature is (0..1+) to 0.6..0.95."""
    return round(0.6 + 0.35 * float(np.clip(margin, 0.0, 1.0)), 2)


@dataclass
class SignatureClassifier:
    stain_chroma: float = 0.08
    stain_view_spread: float = 0.25
    dark_all: float = -0.05
    thin: float = 3.5
    crack_solidity: float = 0.7
    relief_depth_mm: float = 0.08
    relief_min_area_mm2: float = 1.5
    scatter_ratio: float = 2.0
    scatter_balance: float = 0.5
    dust_max_area_mm2: float = 5.0
    shading_polarity: float = 0.3
    shading_balance: float = 0.6

    name = "signature-rules-v2"

    def predict(self, f: RegionFeatures) -> Prediction:
        v = f.values
        if v["chroma_shift"] > self.stain_chroma and v["view_spread"] < self.stain_view_spread:
            return Prediction(
                "stain",
                _conf(v["chroma_shift"] / self.stain_chroma - 1),
                "its colour changed by the same amount in every camera and its surface shape did not change, so it is a change in the material",
            )
        if v["brightest_view"] < self.dark_all:
            if v["elongation"] > self.thin or v["solidity"] < self.crack_solidity:
                return Prediction(
                    "crack",
                    _conf(-v["brightest_view"] / -self.dark_all - 1),
                    "it is darker under every light and from the top and 45-degree cameras, and thin or branched, so it is a deep opening that traps light",
                )
            return Prediction(
                "pit",
                _conf(-v["brightest_view"] / -self.dark_all - 1),
                "it is darker under every light and from the top and 45-degree cameras, and compact, so it is a small cavity that traps light",
            )
        if f.depth_available and abs(v["depth_mm"]) >= self.relief_depth_mm and v["area_mm2"] >= self.relief_min_area_mm2:
            label = "dent" if v["depth_mm"] < 0 else "bump"
            return Prediction(
                label,
                _conf(abs(v["depth_mm"]) / self.relief_depth_mm - 1),
                f"the 3D shape map shows the surface {'pushed in' if label == 'dent' else 'raised'} by {abs(v['depth_mm']):.2f} mm with no colour change",
            )
        if (
            v["geometry_ratio"] > self.scatter_ratio
            and v["grazing_balance"] > self.scatter_balance
            and v["c_grazing"] > 0
        ):
            if v["elongation"] > self.thin:
                return Prediction(
                    "scratch",
                    _conf(v["geometry_ratio"] / self.scatter_ratio - 1),
                    "it is a thin line that lights up at the grazing camera with no colour change, so it is a shallow cut scattering light",
                )
            if v["area_mm2"] < self.dust_max_area_mm2:
                return Prediction(
                    "dust",
                    _conf(v["geometry_ratio"] / self.scatter_ratio - 1),
                    "it is a small spot that lights up at the grazing camera, so it is a raised particle scattering light",
                )
        if abs(v["polarity"]) > self.shading_polarity and v["sign_balance"] < self.shading_balance:
            label = "dent" if v["polarity"] > 0 else "bump"
            side = "far from" if label == "dent" else "facing"
            return Prediction(
                label,
                _conf(abs(v["polarity"]) / self.shading_polarity - 1),
                f"under the 45-degree light its shading is bright on the side {side} the light with no colour change, so the surface is {'pushed in' if label == 'dent' else 'raised'}",
            )
        return Prediction("unknown", 0.3, "its signature does not match any known defect type")


class ForestClassifier:
    def __init__(self, model=None, n_samples: int = 0):
        self.model = model
        self.n_samples = n_samples

    @property
    def name(self) -> str:
        return f"random-forest({self.n_samples} labelled regions)"

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "ForestClassifier":
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=0)
        self.model.fit(X, list(y))
        self.n_samples = len(y)
        return self

    def predict(self, f: RegionFeatures) -> Prediction:
        proba = self.model.predict_proba(f.vector()[None, :])[0]
        i = int(np.argmax(proba))
        label = str(self.model.classes_[i])
        top = sorted(zip(self.model.feature_importances_, FEATURE_NAMES), reverse=True)[:3]
        basis = "trained model; most informative features: " + ", ".join(n for _, n in top)
        return Prediction(label, round(float(proba[i]), 2), basis)

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "n_samples": self.n_samples, "features": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path: Path) -> "ForestClassifier":
        import joblib

        data = joblib.load(path)
        if tuple(data["features"]) != FEATURE_NAMES:
            raise ValueError(f"{path} was trained on a different feature set - retrain it")
        return cls(data["model"], data["n_samples"])


def load_classifier(path: Path):
    return ForestClassifier.load(path) if path.exists() else SignatureClassifier()
