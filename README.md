# Ayas · अयस्

*Every surface, seen true.*

**On-device visual QC for steel and metal surfaces.**

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

## Results

All accuracy numbers are from the physically based **digital twin** (Blender 5.2 renders with the device's sensor-noise model). Real steel specimens are Phase 2. All timing and memory numbers are **measured on the Raspberry Pi 5**. Full tables: `reports/` and the project report.

### 1. Headline

| What | Result |
|---|---|
| Lighting robustness (hood, dark room to direct sun) | Wrong verdict with no warning in **2 of 210** captures, **0 unsafe**; the old single-frame method had 14 of 210 |
| Defect recall (hood, all 7 conditions) | scratch 1.00 · dent 1.00 · crack 0.98 · corrosion 0.83 · stain 0.80 · pit 0.79 |
| Camera to verdict on the Pi 5 | **1.7 s p50** with the detector on the Hailo-10H (**47 ms**, 28× faster than the CPU) |
| Llama 3.2 1B via ExecuTorch on the Pi | First token **5.4 s**, **11 tokens/s**, cause note in 10.3 s; 9 of 9 notes passed the guardrail |
| 3D dent depth (PINN) | **0.053 mm** error with the L/R LEDs, **0.021 mm** with a 6-sector ring |
| Device software = evaluated pipeline | **28 / 28** verdicts on the Pi |
| Product hardware | 13 boards: **ERC 0, DRC 0, 0 unconnected**; **0 interference** across 318 component models in the housing |
| Tests | **869 passed**, 12 skipped |

### 2. Lighting experiment (pre-registered)

**Set-up:**
- 15 unseen test plates (seeds ≥ 900100: never trained, tuned or calibrated on), 6 materials, 3 defect-free.
- 7 lighting conditions: C0 dark room, C1 office (~350 lux), C2 warm bulb close by, C3 window daylight, C4 low side lamp with a hand shadow, C5 a lamp reflected in the plate, C6 direct sun (~45,000 lux).
- Hood and open device, 2 variants each, 2 capture methods: **840 captures**.

**Pass criterion (fixed before the test):**
- **Hood (the product), every condition:**
  - retake ≤ 5 %;
  - F1 ≥ F1(dark) − 0.05;
  - verdict agreement with the dark room ≥ 95 %;
  - **0 silent failures** (the gate accepts and the verdict is wrong).
- **Open device:** 0 silent failures.

| Method | Device | Wrong, no warning | Unsafe¹ | Retakes | F1 |
|---|---|---|---|---|---|
| New (5-frame burst + gate) | **hood (product)** | **2 / 210** | **0** | 0 / 210 | 0.83–0.85 |
| New (5-frame burst + gate) | open | 12 / 210 | 2 | 142 / 210 | 0.72–0.89 |
| Old (one auto-exposed frame) | hood | 14 / 210 | 0 | no gate | 0.91–0.92 |
| Old (one auto-exposed frame) | open | 16 / 210 | 0 | no gate | 0.80–0.91 |

¹ Milder than the truth and not held.

**New method, per condition** (P / R / F1 over accepted captures; verdict agreement with the dark room and with the truth; |Δ| = mean image difference from the dark room in sensor counts):

| Condition | Lux | Retakes (main reason) | Precision | Recall | F1 | Verdict = dark | Verdict = truth | Silent (unsafe) | \|Δ\| (DN) |
|---|---|---|---|---|---|---|---|---|---|
| C0 dark room, **hood** | 0 | 0/30 | 0.78 | 0.89 | 0.83 | 30/30 | 30/30 | 0 (0) | 0.0 |
| C1 office, **hood** | 353 | 0/30 | 0.78 | 0.91 | 0.84 | 29/30 | 29/30 | 1 (0) | 2.8 |
| C2 warm bulb, **hood** | 648 | 0/30 | 0.78 | 0.91 | 0.84 | 30/30 | 30/30 | 0 (0) | 2.7 |
| C3 window daylight, **hood** | 3,569 | 0/30 | 0.79 | 0.91 | 0.85 | 30/30 | 30/30 | 0 (0) | 2.7 |
| C4 side lamp + hand shadow, **hood** | 2,005 | 0/30 | 0.79 | 0.91 | 0.85 | 29/30 | 29/30 | 1 (0) | 2.8 |
| C5 lamp reflected in the plate, **hood** | 514 | 0/30 | 0.78 | 0.91 | 0.84 | 30/30 | 30/30 | 0 (0) | 2.8 |
| C6 direct sun, **hood** | 44,872 | 0/30 | 0.77 | 0.89 | 0.83 | 30/30 | 30/30 | 0 (0) | 2.7 |
| C0 dark room, open | 0 | 6/30 (saturated) | 0.81 | 0.82 | 0.82 | 24/24 | 20/24 | 4 (0) | 0.0 |
| C1 office, open | 353 | 17/30 (changed) | 0.80 | 0.72 | 0.76 | 11/13 | 9/13 | 4 (2) | 2.8 |
| C2 warm bulb, open | 648 | 13/30 (changed) | 0.77 | 0.85 | 0.81 | 16/17 | 16/17 | 1 (0) | 2.7 |
| C3 window daylight, open | 3,569 | 26/30 (glare + ambient) | 0.92 | 0.86 | 0.89 | 4/4 | 4/4 | 0 (0) | 3.4 |
| C4 side lamp + hand shadow, open | 2,005 | 27/30 (changed) | 0.89 | 0.73 | 0.80 | 3/3 | 3/3 | 0 (0) | 3.0 |
| C5 lamp reflected, open | 514 | 23/30 (glare + ambient) | 0.76 | 0.68 | 0.72 | 7/7 | 4/7 | 3 (0) | 3.1 |
| C6 direct sun, open | 44,872 | 30/30 (too bright) | – | – | – | – | – | 0 (0) | – |

**With the hood:**
- Every condition from dark to direct sun meets the retake, F1 and agreement limits.
- False alarms on defect-free plates: **0** (the old method: 0.33 per plate).
- The processed image differs from the dark room by 2.7–2.8 DN in every condition, about the sensor-noise level, even in direct sun.
- **Caveat:** in the twin a seated hood is light-tight by construction, so these rows test repeatability, not light leakage. Real-world leakage is the first Phase 2 measurement.

**Strict result: FAIL by 2 captures.**
- Both are one low-confidence (0.27 / 0.32) false "crack" on unlabelled mill-scale texture at a plate edge. It gave REJECT where the truth is HOLD_FOR_QA: stricter than the truth, not unsafe.
- The score does not follow the light level (Spearman ρ = 0.15), so the cause is the model, not the lighting.

**Fixes adopted (22 Nov):**
- **Two-burst confirmation (P1h).** Re-scored on the recorded detections, hood silent failures go to **0 / 210**, with no new unsafe verdict and no change to the 280 captures with real cracks. It still needs confirming on fresh plates.
- **Hood-only certification.** Without the hood, the gate refuses 142 of 210 bright scenes (every direct-sun capture), and dark mill scale caused 2 unsafe misses. Open mode now never certifies.

### 3. Detection quality (new method, hood, all 7 conditions pooled)

| Class | Recall |
|---|---|
| scratch | 126 / 126 = **1.00** |
| dent | 70 / 70 = **1.00** |
| crack | 82 / 84 = **0.98** |
| corrosion | 70 / 84 = 0.83 |
| stain | 56 / 70 = 0.80 |
| pit | 77 / 98 = 0.79 |

- **Precision** is 0.77–0.79 (the old method: 0.90). Part of the gap is detections of real defects that the ground truth marks "not visible" and counts as false positives.
- **Detector:** YOLOv8n trained from COCO for 150 epochs on 5,176 / 430 tiles from 800 twin scenes. ONNX and PyTorch agree to 0.0013 on the raw output.

### 4. On the Raspberry Pi 5

| Metric | Pi 5 + Hailo-10H | Pi 5, CPU only (ONNX Runtime) | Mac M3 CPU (reference) |
|---|---|---|---|
| Camera to verdict, p50 / p95 | **1,665 / 2,106 ms** | 2,856 / 3,331 ms | 617 / 716 ms |
| Detector per inspection | **47 ms** | 1,329 ms | – |
| Throughput | 0.57 inspections/s | 0.39 inspections/s | 1.8 inspections/s |
| Peak memory (app) | 0.98 GB | 1.08 GB | 1.7 GB |
| Verdicts equal to the twin evaluation | 27 / 28¹ | **28 / 28** | 28 / 28 |
| CPU temperature | 51–67 °C, no throttling | | |

¹ The one difference is the edge false positive of §2, which the INT8 model happens not to fire: a side effect of quantisation, not a fix.

**Vision stages on the CPU, p50 ms** (from the Llama benchmark run, `reports/pi_llm_final4_20261122-183748/`; the Hailo run is in `reports/pi_phase1_raspberrypi_20261122-201042/`):

| Stage | capture | process | rectify | tile | detect | merge | measure | rules | report | image | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ms | 561 | 1,135 | 5.4 | 4.1 | 1,337 | 0.3 | 8.8 | 0.2 | 0.3 | 7.5 | 2,866 |

### 5. Llama 3.2 1B Instruct via ExecuTorch (on the Pi's Cortex-A76)

**Speed** (`reports/pi_llm_final4_20261122-183748/`, 16 inspections):

| Metric | p50 | p95 |
|---|---|---|
| Time to first token (model clock) | 5,429 ms | 5,505 ms |
| First text on screen (wall, incl. 2.5 s load) | 8,029 ms | 8,100 ms |
| Decode speed | 11.0 tokens/s | 11.4 tokens/s |
| Prefill speed | 112.5 tokens/s | 113.3 tokens/s |
| Whole cause note | 10.3 s | 11.1 s |
| Camera to advice | 13.5 s | 13.9 s |
| Prompt / generated tokens | 611 / 29 | 611 / 41 |
| Peak memory of the runner | 1.38 GB | |

**First design against final design:**

| | First design | Final design |
|---|---|---|
| Who writes the facts | Llama restates the whole record | The rules engine; Llama writes cause and prevention |
| Prompt | ~1,340 tokens (JSON) | 611 tokens |
| First token | 15.8 s | 5.4 s |
| Decode | 6.7 tokens/s | 11.0 tokens/s |
| Time per answer | ~54 s | 10.3 s |
| Answers with a factual error | 3 of 7 | 0 of 9 |

- Every verdict on screen was right in both designs, because the verdict comes from the rules engine.
- In the final run, the 7 RETAKE and PASS results were answered by the rules engine.
- "How to fix?" and override requests ("Can I paint over the crack and ship it?") always get the rules engine's exact answer.
- Open questions take 31–35 s and are general: the weakest part, Phase 2 work.

### 6. 3D depth: physics-informed neural network (PINN)

**Forward model against Blender** (no fitted parameters):
- **Open pods:** flat metal within 0.1–0.9 % at the median; ball dents within 0.2–1.1 %, pixel correlation ≥ 0.99.
- **Hood:** the direct model under-predicts flat metal by 2–57 %. A black-hood control closes most of the gap (aluminium 0.85 / 0.76 → 0.96 / 0.94), which shows the cause is the hood's own inter-reflection. A one-bounce hood model reproduces it at dents (correlation 0.88–0.94).

**Depth on 96 twin dents and pits** (6 materials, sensor noise):

| | L/R halves (today) | 3 sectors | 6 sectors |
|---|---|---|---|
| PINN, dents, MAE | **0.053 mm** | 0.034 mm | **0.021 mm** |
| PINN, dents within 0.05 mm | 62 % | 79 % | 92 % |
| Classical slopes + Poisson, dents, MAE | 0.153 mm | 0.089 mm | 0.071 mm |
| PINN, pits, MAE (bias) | 0.154 mm (−0.154) | 0.114 mm | 0.107 mm |

- **Against the classical method:** the PINN is 2.1–3.4× better on dents and 1.5–1.8× on pits. Pits are under-estimated (28–88 %) and shown as indicative only.
- **Lighting-invariant with the hood:** across all 7 conditions, dent depth changes by at most 0.023 mm.
- **Twin dent plate, dark room to direct sun:** the 0.331 mm dent reads 0.30–0.34 mm, the 0.168 mm dent 0.15–0.16 mm, and a 0.31 mm gouge 0.36 mm.
- **Severity safety:** under the proposed rule (use a depth only if the fit converged and the 90 % interval is ≤ 0.08 mm), no depth-based decision was more lenient than the decision with the true depth: 0 of 89 coupon decisions and 0 of 7 pilot decisions.
- **On the Pi:** about 15 s for a plate with 5 relief defects, first surface after about 8 s (it was 55 s before the speed-up).

### 7. Experiments tried and not adopted

**I-JEPA anomaly head** (region AUROC, burst capture):

| Method | Hood | Open |
|---|---|---|
| jepa_pred (predictor error) | **0.975** | 0.957 |
| ResNet18 PatchCore | 0.965 | 0.957 |
| ViT-Tiny (ImageNet) kNN | 0.919 | 0.923 |
| Pixels kNN | 0.919 | 0.858 |
| jepa_knn (EMA target + memory bank) | 0.508 | 0.563 |

- Not adopted. It did not beat ResNet18 by the pre-set +0.02 (+0.010 hood, +0.000 open).
- It was less lighting-invariant (open retrieval@1: 0.853 against 0.967).
- It is too slow for the Pi (350–580 ms per crop against a 150 ms budget).

**Depth network:** a U-Net fine-tuned on Blender crops. Dent error 0.149 mm (pit 0.096, scratch 0.041) against 0.022 mm for the PINN: not adopted (`models/depthnet_blender.onnx`).

### 8. Optimisations (measured)

| Technique | Before → after |
|---|---|
| Detector: INT8 + Hailo-10H HEF with on-chip NMS | 1,329 → **47 ms** per inspection (28×); camera to verdict 2.86 → 1.67 s |
| Llama: SpinQuant INT4 + ExecuTorch XNNPACK | 1B model in a 1.08 GB `.pte`, 1.38 GB RAM, 11 tokens/s on the CPU |
| Llama: facts to the rules engine, a short plain-text prompt | Prompt 1,340 → 611 tokens; first token 15.8 → 5.4 s; answer ~54 → 10.3 s; errors 3/7 → 0/9 |
| 3D: fast mode, warm worker pool, one defect per core | Plate with 5 defects 55 → 15 s |
| Capture-side invariance instead of a bigger network | A ~3 M-parameter detector holds across 7 lighting conditions |

### 9. Product hardware (Phase 2)

| Check | Result |
|---|---|
| Schematics (ERC), 13 boards | 0 errors, 0 warnings |
| Layouts (DRC), 13 boards | 0 violations, 0 unconnected pads (LIGHT/IO 6-layer, 0.1 mm rules) |
| Design consistency (`tests/test_hardware_design.py`) | 43 passed: matched cable pinouts, palindrome flex, no swappable connectors, no dangling nets |
| Mechanical fit (Fusion 360) | 318 board component models, 0 interference with the housing, buttons, Pi, display, fan or bank; no floating parts |
| Housing | 268 mm wide, about 900 g of PETG; every service screw into a heat-set insert |

## Architecture and demo

- **Architecture** of the system, every model, the data contract and the electronics: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **2-minute product demo video:** [reports/Ayas_demo_1080p.mp4](reports/Ayas_demo_1080p.mp4) (sources in `tools/video/`).
- **Models:** `models/` holds the detector ONNX and Hailo HEF and the depth-network ONNX; `models/README.md` says how the Llama `.pte` was made.

## Project report

**[docs/report/Ayas_Project_Report.pdf](docs/report/Ayas_Project_Report.pdf)** (also `.docx`) covers the methodology, the AI models, the results, hardware utilisation, optimisation techniques, the 3D model and the PCBs. It is generated by `tools/report/` from the numbers in `reports/` and `docs/`.

Authors: Binay Verma, Dhanush D, Kushal S. Faculty mentor: Dr. Subir Roy.

## Phase 2 product hardware

![Ayas handheld](docs/hardware/renders/1_front.jpg)

- **Housing:** a parametric Fusion 360 model (`hardware/enclosure/ayas_handheld_fusion.py`): a two-grip handheld, 268 mm wide, with two toed-in LED pods, the camera on the centre line, a 5-inch touch display on the back and a clip-on light-tight inspection hood. Every service screw goes into a heat-set insert.
- **Full assembly:** `hardware/enclosure/ayas_assembly_fusion.py` places every KiCad board as its 3D model and draws the cable harness. All 318 board component models fit with no interference. Renders: `docs/hardware/renders/`.
- **13 PCBs** generated from one description, `hardware/pcb/design.py` (KiCad 10, auto-routed with Freerouting). Every board passes ERC and DRC with nothing unrouted:
  - **POWER** (2-layer): USB-C PD sink to 12 V, a 5 V / 5 A buck for the Pi, LED-rail monitor, power-hold with no standby drain, and charge-through.
  - **LIGHT/IO** (6-layer): an RP2040 with 10 constant-current LED sinks, the UV interlock, buttons, fan and buzzer.
  - **Pods:** UV pod hubs (9 plug-in UV LEDs each) and WHITE-L / WHITE-R carriers for three 12 V white LED modules each.
  - **Docking:** the pod flex with its docks, and the hood dock and pad on spring pins.
  - **SIDE-L / SIDE-R** button boards and the **CHARGE** port board.
- **Controls, power and export:** `docs/hardware/controls.md`. **Board specification:** `docs/hardware/pcb_phase2.md`. **Service manual** (fasteners, cable table, assembly order, repairs): `docs/hardware/service.md`.
- **Checked by** `tests/test_hardware_design.py`: both ends of every cable carry the same pinout, flex pinouts are palindromes, and no two different connectors on one board take the same plug.

## Gallery

Every image of the project: the product, how it works, the architecture, the results, and every PCB with its schematic. Click an image for full size.

### The product

<img src="docs/hardware/renders/1_front.jpg" width="860" alt="Front">

<table><tr><td><img src="docs/hardware/renders/2_back.jpg" width="420" alt=""><br><sub>Back: display, thumb buttons, USB-A export</sub></td><td><img src="docs/hardware/renders/3_hood.jpg" width="420" alt=""><br><sub>With the inspection hood fitted</sub></td></tr>
<tr><td><img src="docs/hardware/renders/4_xray_front.jpg" width="420" alt=""><br><sub>X-ray, front: boards, LED modules and harness</sub></td><td><img src="docs/hardware/renders/5_xray_back.jpg" width="420" alt=""><br><sub>X-ray, back: POWER, LIGHT/IO, Pi stack, bank, fan</sub></td></tr>
<tr><td><img src="docs/hardware/renders/6_xray_top.jpg" width="420" alt=""><br><sub>X-ray, top: harness routes</sub></td><td><img src="docs/hardware/renders/7_cutaway.jpg" width="420" alt=""><br><sub>Cutaway from the back, shell at 25 %</sub></td></tr>
<tr><td><img src="docs/report/figures/v_product.jpg" width="420" alt=""><br><sub>Product render (demo video)</sub></td><td><img src="docs/report/figures/v_exploded.jpg" width="420" alt=""><br><sub>Exploded view (demo video)</sub></td></tr></table>

### How it works

<table>
<tr><td><img src="docs/report/figures/v_burst.jpg" width="420" alt=""><br><sub>5-frame light burst and the quality gate</sub></td><td><img src="docs/report/figures/v_flow.jpg" width="420" alt=""><br><sub>End to end: scan → gate → tile → detect → 3D → rules → explanation</sub></td></tr>
<tr><td><img src="docs/report/figures/v_yolo.jpg" width="420" alt=""><br><sub>YOLOv8n on the Hailo-10H</sub></td><td><img src="docs/report/figures/v_rules.jpg" width="420" alt=""><br><sub>The rules engine decides the verdict</sub></td></tr>
<tr><td><img src="docs/report/figures/v_pinn.jpg" width="420" alt=""><br><sub>PINN: a light model fits the height map</sub></td><td><img src="docs/report/figures/v_surface.jpg" width="420" alt=""><br><sub>3D surface of a 0.32 mm dent</sub></td></tr>
<tr><td><img src="docs/report/figures/v_llama.jpg" width="420" alt=""><br><sub>Llama 3.2 1B via ExecuTorch explains the verdict</sub></td><td><img src="docs/report/figures/v_qa.jpg" width="420" alt=""><br><sub>Station functions and who answers each question</sub></td></tr>
</table>

### Architecture

<table><tr><td><img src="docs/report/figures/d_system.png" width="420" alt=""><br><sub>Software: the two pipelines</sub></td><td><img src="docs/report/figures/d_electronics.png" width="420" alt=""><br><sub>Electronics</sub></td></tr></table>

### Results

<table>
<tr><td><img src="docs/report/figures/c_lighting.png" width="420" alt=""><br><sub>Wrong verdicts with no warning, by method and device</sub></td><td><img src="docs/report/figures/c_recall.png" width="420" alt=""><br><sub>Detector recall per class</sub></td></tr>
<tr><td><img src="docs/report/figures/c_latency.png" width="420" alt=""><br><sub>Camera to verdict on the Pi 5: CPU against Hailo-10H</sub></td><td><img src="docs/report/figures/c_llama.png" width="420" alt=""><br><sub>Llama on the Pi: first design against final</sub></td></tr>
<tr><td><img src="docs/report/figures/c_pinn.png" width="420" alt=""><br><sub>Dent depth error: PINN against classical</sub></td><td><img src="docs/report/figures/c_speedups.png" width="420" alt=""><br><sub>Measured speed-ups</sub></td></tr>
</table>

**Digital twin and experiments**

<table>
<tr><td><img src="reports/figures/lighting/lighting_s900100.png" width="420" alt=""><br><sub>Test plate s900100 in all 7 lighting conditions</sub></td><td><img src="reports/figures/scenes/lighting_grid_s900100.jpg" width="420" alt=""><br><sub>Lighting grid, plate s900100</sub></td></tr>
<tr><td><img src="reports/figures/pilot_naive_vs_burst_s900011.png" width="420" alt=""><br><sub>Old single frame against the new burst</sub></td><td><img src="reports/figures/pilot_gate_fixes.png" width="420" alt=""><br><sub>Quality-gate decisions</sub></td></tr>
<tr><td><img src="reports/figures/scenes/burst_s900100_C3h_v0.jpg" width="420" alt=""><br><sub>Burst-processed plate, window daylight, hood</sub></td><td><img src="reports/figures/scenes/ground_truth_s900100_C0h_v0.jpg" width="420" alt=""><br><sub>Its ground truth</sub></td></tr>
<tr><td><img src="reports/figures/physics_validation_open_aluminium.png" width="420" alt=""><br><sub>Light model against Blender, open pods</sub></td><td><img src="reports/figures/physics_hood_bounce_D3.png" width="420" alt=""><br><sub>Hood inter-reflection model at a dent</sub></td></tr>
<tr><td><img src="reports/figures/physics_pilot_invariance.png" width="420" alt=""><br><sub>PINN depth across lighting conditions</sub></td><td><img src="reports/figures/physics_coupons_scatter.png" width="420" alt=""><br><sub>PINN depth against the truth on the coupons</sub></td></tr>
<tr><td><img src="reports/figures/jepa/s900100_C3h_v0_burst.jpg" width="420" alt=""><br><sub>JEPA anomaly maps (not adopted)</sub></td><td><img src="reports/figures/pilot_montage_s900011.png" width="420" alt=""><br><sub>Pilot plate montage</sub></td></tr>
</table>

### PCBs and schematics

All 13 boards, generated from `hardware/pcb/design.py` and routed with 0 DRC errors. For each board: 3D render of the top and bottom, the copper layout, and the schematic (click it for the vector PDF). KiCad sources: `hardware/pcb/kicad/<board>/`.

<img src="docs/report/figures/pcb_montage.jpg" width="860" alt="All boards">

#### POWER

2-layer, 46 × 55 mm. USB-C PD sink to 12 V (STUSB4500), 5 V / 5 A buck for the Pi (TPS565201), LED-rail monitor (INA226), power-hold with no standby drain, charge-through.

<table><tr><td><img src="docs/hardware/pcb/power_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/power_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/power_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_power.pdf"><img src="docs/hardware/schematics/ayas_power.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### LIGHT/IO

6-layer, 46 × 55 mm. RP2040 on USB to the Pi, 10 constant-current LED sinks, UV interlock, buttons, fan, buzzer, NTCs. The layout view shows F, In2–In4 and B copper.

<table><tr><td><img src="docs/hardware/pcb/light_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/light_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/light_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_light.pdf"><img src="docs/hardware/schematics/ayas_light.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### UV pod hub (×2)

2-layer, 43 × 30.5 mm. 9 plug-in 5 mm 395 nm UV LEDs on sockets marked + / −, ZIF for the pod flex, cable to the white carrier.

<table><tr><td><img src="docs/hardware/pcb/uv_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/uv_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/uv_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_uv.pdf"><img src="docs/hardware/schematics/ayas_uv.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### WHITE-L carrier

2-layer, 50 × 66 mm. Three 12 V white LED modules; tabs on M2 screws into SMD nuts, leads in push-in terminals.

<table><tr><td><img src="docs/hardware/pcb/white_l_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/white_l_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/white_l_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_white_l.pdf"><img src="docs/hardware/schematics/ayas_white_l.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### WHITE-R carrier

The mirror image of WHITE-L for the right pod.

<table><tr><td><img src="docs/hardware/pcb/white_r_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/white_r_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/white_r_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_white_r.pdf"><img src="docs/hardware/schematics/ayas_white_r.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### Pod flex (×2)

Flex, 11 × 60 mm. 10 straight conductors with a palindrome pinout: correct either way round.

<table><tr><td><img src="docs/hardware/pcb/podflex_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/podflex_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/podflex_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_podflex.pdf"><img src="docs/hardware/schematics/ayas_podflex.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### POD DOCK L

2-layer, 28 × 14 mm. ZIF for the pod flex behind the body front face; JST-SH 6 to LIGHT/IO.

<table><tr><td><img src="docs/hardware/pcb/pod_dock_l_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/pod_dock_l_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/pod_dock_l_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_pod_dock_l.pdf"><img src="docs/hardware/schematics/ayas_pod_dock_l.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### POD DOCK R

The same with JST-SH 7, so the two pod cables cannot be swapped.

<table><tr><td><img src="docs/hardware/pcb/pod_dock_r_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/pod_dock_r_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/pod_dock_r_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_pod_dock_r.pdf"><img src="docs/hardware/schematics/ayas_pod_dock_r.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### HOOD DOCK

2-layer, 34 × 13 mm. 9 spring pins up through the top wall; JST-SH 9 to LIGHT/IO.

<table><tr><td><img src="docs/hardware/pcb/hood_dock_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/hood_dock_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/hood_dock_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_hood_dock.pdf"><img src="docs/hardware/schematics/ayas_hood_dock.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### HOOD PAD

2-layer, 40 × 32 mm. Gold pads onto the dock, 6 push-in terminals for the ring modules, the interlock loop.

<table><tr><td><img src="docs/hardware/pcb/hood_pad_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/hood_pad_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/hood_pad_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_hood_pad.pdf"><img src="docs/hardware/schematics/ayas_hood_pad.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### SIDE-L buttons

2-layer, 36 × 20 mm. Push-in terminals TRIG, UPPER, LOWER; JST-SH 4 to LIGHT/IO.

<table><tr><td><img src="docs/hardware/pcb/btn_l_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/btn_l_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/btn_l_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_btn_l.pdf"><img src="docs/hardware/schematics/ayas_btn_l.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### SIDE-R buttons

The same with JST-SH 5.

<table><tr><td><img src="docs/hardware/pcb/btn_r_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/btn_r_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/btn_r_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_btn_r.pdf"><img src="docs/hardware/schematics/ayas_btn_r.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

#### CHARGE

2-layer, 24 × 12.5 mm. Power-only USB-C through the side wall; JST-PH 5 to POWER.

<table><tr><td><img src="docs/hardware/pcb/charge_top.png" width="280" alt=""><br><sub>top</sub></td><td><img src="docs/hardware/pcb/charge_bottom.png" width="280" alt=""><br><sub>bottom</sub></td><td><img src="docs/hardware/pcb/charge_layout.png" width="280" alt=""><br><sub>copper layout</sub></td></tr></table>

<a href="docs/hardware/schematics/ayas_charge.pdf"><img src="docs/hardware/schematics/ayas_charge.png" width="860" alt=""></a><br><sub>Schematic (click for the PDF)</sub>

## Quick start on a laptop (no hardware)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,twin]"
# models/v8n_burst.onnx is in the repo (models/README.md).  Twin bursts in data/twin_demo/bursts/: see DATASETS.md.
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
hardware/          enclosure/ (Fusion 360 housing + full assembly), pcb/ (design.py, generators, router, 13 KiCad boards)
tools/             jl/ (GPU-box jobs), pi/phase1_run.sh, make_demo_video.py, report/ (project report generator)
configs/           capture.yaml, twin.yaml, spec_steel.yaml, app.yaml
docs/              Phase 1 plan, app guide, twin design, hardware (spec, controls, service, renders), report, research
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
