"""Physics for Ayas: the LED image-formation model and a per-defect depth estimator.

docs/twin/PHYSICS_PINN.md is the contract and the results. Modules:

- optics: forward model of the ambient-subtracted LED frames on the rectified plate grid
  (near-field LEDs, a Principled-BSDF approximation, the camera view per pixel). numpy core;
  the same code runs on torch tensors (differentiable) when torch is installed.
- depth_pinn: estimate_depth(), a physics-informed neural field h(u, v) fitted per detected
  dent or pit to the LED frames through optics (torch, CPU), with a seed-ensemble uncertainty.
- baselines: classical photometric slopes + Frankot-Chellappa / Poisson integration, and the
  'no depth' baseline (today's HOLD_FOR_QA).
- severity: the proposed depth-aware disposition (EN 10163-2 / ISO 7788 depth tables).
- corrosion: an advisory uniform-corrosion thickness-loss forecast (ISO 9223 / ISO 9224).
- twin_eval: helpers that build depth-estimator inputs from twin renders (EXR + plan).

Only numpy (and cv2 for resampling) are needed to import this package; torch is optional.
"""

from __future__ import annotations

__all__ = ["optics", "depth_pinn", "baselines", "severity", "corrosion", "twin_eval"]
