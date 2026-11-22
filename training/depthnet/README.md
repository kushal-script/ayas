# Depth network for the Hailo-10H (tried 22 Nov 2026, not adopted)

**Goal.** Measure a defect's height map in one pass on the AI HAT+ 2 instead of fitting it on the CPU. The physics fit (`physics/depth_pinn.py`, used by `app/surface3d.py`) takes about 5 s per defect on one Pi core.

**What was built.**
- `datagen.py`: 24,000 training crops rendered by the physics forward model (`depth_pinn.Renderer`):
  - synthetic dents (ball, smooth, punch), pits (single and cluster) and scratch grooves;
  - drawn on the light geometry, camera and exposure of the twin's TRAIN plans (hood ring and open pods);
  - each plate's material prior with jitter, a smooth albedo factor, micro-relief and sensor noise. Test plates are never used.
- `model.py`: a 0.5 M-parameter U-Net. It uses only convolutions, ReLU, max-pool, nearest upsampling and concatenation, so it is Hailo-native.
  - Its input is the same crop the physics fit uses: left and right light-only frames, log-scaled and quantised to 8 bits, plus the light type, the crop position and the pixel spacing.
  - Its output is the height in 0.1 mm units.
- `train.py`: 30 epochs on the Mac's GPU (about 25 min); exported to ONNX.
- `eval_twin.py`: scores it on the Blender twin, which it never saw.

**Results.**

| | Dents MAE | Pits MAE | Scratches MAE |
|---|---|---|---|
| Held-out synthetic crops | 0.089 mm | 0.048 mm | 0.059 mm |

On the Blender twin dent plate (s900104; hood in dark room and direct sun, plus open mode; Mac, 1 core):

| Method | Dents MAE | Pits MAE | Time per defect |
|---|---|---|---|
| Network alone | 0.140 mm | 0.150 mm | 1.7 ms (CPU ONNX) |
| Network, then a 40-step physics fit from it | 0.153 mm | 0.171 mm | 0.45 s |
| **Fast physics fit (shipped)** | **0.022 mm** | **0.034 mm** | 0.86 s |

- The network matches the large hood dent (0.300 mm against 0.331 true; 0.332 after the short fit).
- It under-reads the small dent by half, and over-reads one pit by 3.5×.
- In open mode it reads dents 2× too deep.
- A short fit started from it stays near the network's answer.

**Why.** The network learned the simplified physics renderer. Blender's images (and real ones) differ from it, most under the open pods. The physics fit is robust to that gap because it fits each crop's own material and gains.

**Decision.** Not adopted: by the rule set beforehand, it had to match the physics fit's accuracy. The 3D view keeps the physics fit, which was made 3.7× faster the same day (parallel per-defect workers, half the steps).

**What would fix it (Phase 2).**
- Train on crops from Blender renders of the TRAIN scenes (their EXRs are on Drive), and later on real gauged coupons, so the training domain matches the device.
- Then compile with `hailo/compile_hef_jl.sh`, exactly as for the detector.

The code is kept here; `depth_pinn.fit_once` accepts `cfg["h_init"]` to start the grid fit from any height map.
