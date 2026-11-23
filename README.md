# Ayas: on-device visual QC for steel and metal surfaces

Raspberry Pi 5 · 5-frame LED burst with a quality gate · YOLOv8n · rules informed by EN 10163-2 / ISO 7788 · Llama 3.2 1B via ExecuTorch on the Arm CPU. **Built with Llama.**

This is an entry to the Arm **Bharat AI-SoC Challenge 2026-27**, PS5 Manufacturing (real-time surface-defect detection and QC assistant).

**Phase 1** has two parts:
- a **Blender digital-twin proof of concept** of a capture method that is robust to lighting;
- the **full pipeline running on the Pi**, with twin captures as the camera.

Real-camera validation is Phase 2.

## What it does

1. **Capture.** There are two device modes:
   - the product is a **clip-on hood with a low-angle LED ring**;
   - **open handheld** with two LED pods.

   Each capture is a **5-frame raw burst** of lit and LEDs-off frames. From it the device:
   - subtracts the ambient light;
   - removes glare;
   - runs a **quality gate** that answers PASS / PASS_FLAGGED / **RETAKE**, so it asks again instead of guessing.
2. **Detect.** The plate is rectified to a 0.08 mm/px grid and cut into 640 px tiles. **YOLOv8n** finds 6 classes: scratch, dent, pit, corrosion, crack and stain. It runs on the **Hailo-10H** (AI HAT+ 2, `models/v8n_burst.hef`, 47 ms per plate), or in ONNX Runtime on the Cortex-A76 without the HAT.
3. **Judge.** A rules engine (`configs/spec_steel.yaml`) sets severity and the verdict: PASS / CLEAN_AND_REINSPECT / REWORK / HOLD_FOR_QA / REJECT.
   - The thresholds are illustrative, informed by EN 10163-2 / ISO 7788, ISO 8501 and IS 2062.
   - Confidence is not severity: a detection below 0.5 means HOLD_FOR_QA. A low-confidence crack keeps REJECT only when the inspection's second burst confirms it; otherwise it is held.
   - Open mode never certifies: PASS and CLEAN_AND_REINSPECT become HOLD_FOR_QA until the hood is fitted.
4. **Record.** Each inspection gives one JSON record (`ayas.inspection/3.0`), a row in the SQLite log and an overlay image.
5. **Advise.** **Llama 3.2 1B Instruct via ExecuTorch** on the CPU works from the JSON record. The rules engine writes the verdict, the measurements and the steps. Llama adds the likely cause and prevention, and answers the operator's follow-up questions.
   - It never changes the verdict.
   - If it is unavailable, a labelled template answers instead.

6. **Measure in 3D (on demand).** The **3D** button rebuilds the surface of each dent, pit and scratch from the burst's left and right LED frames (`app/surface3d.py`). A physics-informed height field is fitted per defect (`physics/depth_pinn.py`), in parallel worker processes. The screen shows a rotatable height map with the depth and its uncertainty. It is a measurement aid; the verdict never depends on it.
   - Twin dent plate, under the hood in dark room, window and direct sun: the 0.331 mm dent reads 0.30–0.34 mm and the 0.168 mm dent 0.15–0.16 mm; a 0.31 mm gouge reads 0.36 mm. Pits read shallow and are shown as indicative.
   - On the Pi: about 15 s for a plate with 5 relief defects, with the first surface on screen after about 8 s (it was 55 s before the speed-up).

Vision (Pipeline A) and Llama (Pipeline B) are **separate pipelines** that exchange only the JSON record.

## Results so far (digital twin)

15 unseen test plates, 7 lighting set-ups from a dark room to direct sun (~45,000 lux), open and hood, two capture methods: 840 captures in all (`reports/lighting_matrix.md`).

| Method | Device | Wrong, no warning | Unsafe | Retakes | F1 |
|---|---|---|---|---|---|
| New (burst) | **hood (product)** | **2 / 210** | **0** | 0 / 210 | 0.83–0.85 |
| New (burst) | open | 12 / 210 | 2 | 142 / 210 | 0.72–0.89 |
| Old (one frame) | hood | 14 / 210 | 0 | no gate | 0.91–0.92 |
| Old (one frame) | open | 16 / 210 | 0 | no gate | 0.80–0.91 |

**With the hood:**
- Lighting changed nothing beyond sensor noise, in every condition including direct sun. But in the twin the seated hood blocks ambient light by construction, so this tests repeatability; real-world light leakage is untested.
- The 2 wrong verdicts are a single low-confidence phantom "crack" at a plate edge. It gave REJECT where the truth was HOLD_FOR_QA: stricter than the truth, not unsafe.
- Verified cause: the model weakly scores unlabelled mill-scale texture at the plate edge as a crack (0.12–0.32 in every hood capture of that plate, including the dark room), and sensor noise sometimes lifts it over the 0.25 threshold.
- The pre-registered criterion allows **0** such errors, so the strict result is **FAIL by 2 captures**. Adopted fix (22 Nov): two-burst confirmation. A low-confidence detection the second burst does not confirm causes at most HOLD_FOR_QA; it still needs confirming on fresh plates.

**Without the hood:**
- The gate refuses most bright scenes.
- Defects on dark mill scale were missed, twice unsafely. So **the device certifies only with the hood** (adopted 22 Nov): in open mode PASS and CLEAN_AND_REINSPECT become HOLD_FOR_QA with "fit the hood to certify".

**The old method** has no gate. In the twin its hood rows reuse one noise draw, which flatters it.

**Other results:**
- **Demo video:** `reports/demo_lighting.mp4`.
- **JEPA anomaly head:** tried and not adopted (`docs/JEPA.md` §10).
- **PINN dent depth:** 0.02–0.05 mm error in the twin (`docs/twin/PHYSICS_PINN.md`). Not in the app yet.
- **On the Raspberry Pi 5** (22 Nov, `reports/pi_phase1_raspberrypi_20261122-170051/`):
  - the app gave the twin evaluation's verdict for 28 / 28 bursts;
  - camera to verdict: **1.7 s p50 with the detector on the Hailo-10H** (47 ms per inspection, 28× faster than the CPU's 1.3 s; 2.9 s on the CPU), 0.98 GB; Hailo verdicts match the CPU's on 27/28 bursts;
  - Llama 3.2 1B SpinQuant via ExecuTorch, 1.4 GB: in the first run, 3 of 7 Llama answers misstated facts, though the verdicts were right. Now the rules engine writes the verdict, the measurements and the steps, and Llama writes the cause-and-prevention note and answers follow-ups, all behind a stricter guardrail. Final run: 9 of 9 notes from Llama passed, advice in 13.5 s, first token after 5.4 s, 11 tokens/s. "How to fix?" and override requests get the rules engine's exact answer; open questions take Llama 30–35 s and are its weakest part.

## Quick start on a laptop (no hardware)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,twin]"
# models/v8n_burst.onnx: see models/README.md.  Twin bursts in data/twin_demo/bursts/: see DATASETS.md.
.venv/bin/ayas-app run --source twin:data/twin_demo/bursts --detector onnx:models/v8n_burst.onnx --advisor mock --count 5
.venv/bin/ayas-app run --source twin:data/twin_demo/bursts --detector onnx:models/v8n_burst.onnx --advisor mock --ui --auto 8
#   station page: http://127.0.0.1:8080
.venv/bin/ayas-app run --source twin:data/twin_demo/bursts --detector onnx:models/v8n_burst.onnx --advisor mock --bench 20
.venv/bin/python -m pytest -q
```

## On the Raspberry Pi 5

`docs/APP.md` section 2 covers the install (Raspberry Pi OS Trixie), the camera, the optional Hailo, and Llama via ExecuTorch (`--advisor executorch` or `executorch_py`).

The Phase 1 measurement run is one script. It runs:
- a system record;
- a smoke pass compared with the twin evaluation;
- a vision benchmark and a benchmark with Llama.

It bundles the results into one archive:

```bash
bash tools/pi/phase1_run.sh            # see its header for the files to copy first
```

## Digital twin

| What | Where |
|---|---|
| Design contract | `docs/twin/DESIGN_STEEL.md` |
| Status and decisions | `docs/twin/STATUS_STEEL.md` |
| Data and seeds | `DATASETS.md` |
| Rendering on a GPU box | `tools/jl/README.md` |
| Training and the lighting evaluation | `training/README.md` |

## Repo map

```
src/ayas/
  lightproc.py     5-frame burst: ambient subtraction, L/R combination, quality gate
  tiles.py npu.py  tiling, ONNX / Hailo / torch detectors, seam merge
  rules.py         severity and verdict (steel spec)
  report.py store.py  JSON record, SQLite log
  llm.py           Llama prompt, ExecuTorch runners, template fallback
  anomaly.py       ONNX patch-anomaly scorer (experimental, not in the verdict)
  physics/         optics model, PINN depth, baselines (twin experiments)
  twin/            Blender digital twin: sampler, textures, lighting, rig, render, post, matrix
  app/             ayas-app: sources, pipeline, advisor, station UI, benchmarks
  sim.py penetrant.py shape.py golden.py ...   the October 3-camera design, kept for reference
training/          YOLO per capture arm, export, eval_lighting, the JEPA experiment
tools/             jl/ (GPU-box jobs), pi/phase1_run.sh, make_demo_video.py
configs/           capture.yaml, twin.yaml, spec_steel.yaml, app.yaml
docs/              Phase 1 plan, app guide, twin design and status, research
reports/           lighting matrix, JEPA, pilot, figures, demo video
```

## Honest scope

- **All evidence so far is from the digital twin.** Real captures of steel specimens come in Phase 2.
- **This is decision support, not a certified inspection system.** The severity thresholds are illustrative.
- **The Phase 1 app does not measure depth.** Depth-governed rows (scratches, dents and pits above minor) go to HOLD_FOR_QA for a gauge check.

## Licence

- AGPL-3.0 (`LICENSE`).
- `NOTICE` lists the third-party software and the Llama 3.2 Community License.
- Built with Llama.
