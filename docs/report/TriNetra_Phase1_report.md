# TriNetra: on-device visual QC for steel and metal surfaces

**Technical report, Phase 1 (Open Qualifying).** Arm Bharat AI-SoC Challenge 2026-27, PS5 Manufacturing.
DRAFT, 22 Nov 2026. Items marked **TBD** are filled from the Raspberry Pi measurement run (`tools/pi/phase1_run.sh`).

---

## 1. Problem and mapping to the brief

Steel service centres and fabrication shops inspect plates and sheets by eye before coating or forming. They look for scratches, dents, pits, corrosion, cracks and contamination.

Two things make this hard to automate cheaply:
- **Shiny metal under uncontrolled light.** Glare, lamp reflections, shadows and sunlight change the image more than the defects do.
- **Turning a detection into a decision** that an operator can act on and that a QA record can defend.

TriNetra is a handheld inspection head on a Raspberry Pi 5. It captures a plate area, detects and measures defects, applies a written severity specification, and logs the result. It then explains the verdict through **Llama 3.2 1B running on the Arm CPU via ExecuTorch**. Everything runs on the device; no cloud.

| Brief requirement | How TriNetra meets it | Evidence |
|---|---|---|
| Detect surface defects, cracks, anomalies | YOLOv8n on 640 px tiles of a 0.08 mm/px rectified grid; 6 classes | §7.2 |
| Classify type and severity | Class from the detector; severity and verdict from `configs/spec_steel.yaml` (rules informed by EN 10163-2 / ISO 7788, ISO 8501, IS 2062) | §4.4, §7 |
| Alert with actionable recommendations | Verdict banner, numbered operator steps, Llama explanation and Q&A | §6, §8 |
| Log results | JSON record per inspection + SQLite + overlay image | §5 |
| Real time at the edge, no cloud | Pi 5 (Cortex-A76) CPU; optional Hailo-10H | §7.4 |
| Llama via ExecuTorch | Llama 3.2 1B Instruct `.pte`, ExecuTorch runner on the CPU, separate process | §4.5 |

The hard requirement we set ourselves was that **it must work across lighting conditions**. Phase 1 therefore centres on a capture method designed for that, proven in a physically based **digital twin** before real specimens (Phase 2).

## 2. System overview

```
 hood head (LED ring) ─┐
 or open head (2 pods) ├─> 5-frame raw burst ─> lightproc: ambient subtraction, glare removal, QUALITY GATE
 camera (raw, fixed)  ─┘                            │ PASS / PASS_FLAGGED              │ RETAKE -> "retake, because ..."
                                                    v
          rectify to 0.08 mm/px plate grid -> 640 px tiles -> YOLOv8n (ONNX Runtime CPU | HailoRT optional)
                                                    v
          merge seams -> measure in mm -> rules (spec_steel.yaml) -> verdict + operator steps
                                                    v
                         JSON record (trinetra.inspection/3.0) -> SQLite log, overlay image, station UI
                                                    |  (only the JSON crosses)
                                                    v
          Pipeline B: Llama 3.2 1B Instruct (.pte) via ExecuTorch on the Cortex-A76 -> explanation, Q&A
```

**Pipeline A (vision)** and **Pipeline B (Llama)** are separate processes that exchange only the JSON record. The verdict never waits for, and is never changed by, the language model.

## 3. Hardware

| Part | Role |
|---|---|
| Raspberry Pi 5 (8 GB) | All compute; Cortex-A76 ×4 |
| Raspberry Pi AI Camera (IMX500), raw mode | Capture. Fixed exposure 10 ms (one 100 Hz mains flicker period), fixed gain, no auto-exposure or ISP |
| Clip-on **hood** head (product) | Light-tight shroud resting on the plate; the camera sits ~200 mm above it. An 18-LED low-angle dark-field ring, 18 mm above the plate (8–12° grazing), in left/right halves. It blocks sun and room light |
| Open head | Two white LED pods (left/right), for quick looks without the hood (preview only, §7.1) |
| Hailo-10H (AI HAT+ 2) | Optional detector accelerator (`--detector hailo:<hef>`); not used for the Phase 1 numbers |
| Waveshare 5" DSI display | Station UI (800 × 480) |

Enclosure: a handheld body generated as parametric CAD in Fusion 360 (`hardware/enclosure/`). LEDs are switched by logic-level MOSFETs from GPIO.

## 4. Pipeline A: vision

### 4.1 Capture: a 5-frame burst

Each capture is B1 – L – A – R – B2 in raw:
- B = LEDs off (ambient only);
- L / R = the left or right LED half (pod);
- A = both.

Because exposure, gain and white balance are fixed and the data is linear raw, the device can:
- **subtract the ambient light**, using the mean of B1 and B2, which also cancels slow drifts;
- **remove glare** by combining L and R (a specular highlight from one side is absent from the other);
- **detect motion and flicker** from B1 vs B2 and from the closure of L + R vs A.

### 4.2 Quality gate

`src/trinetra/lightproc.py` with `configs/capture.yaml` scores every burst and returns:
- **PASS**;
- **PASS_FLAGGED** (result kept with a note, e.g. "ambient light is high");
- **RETAKE**, with a plain instruction ("Too bright: shade the part or fit the hood, then retake"; "Lighting changed during capture: retake").

The gate's checks:
- ambient ratio;
- saturated and masked fractions;
- motion (pixels);
- flicker across the frame;
- closure residual;
- a hood-leak test.

The design rule is: **when the image cannot be trusted, refuse instead of guessing.**

### 4.3 Detection

The processed image is rectified to the plate frame at 0.08 mm/px with a 5 mm border, then cut into 640 px tiles with at least 128 px overlap.

The detector is **YOLOv8n**, trained from COCO weights for 150 epochs. It uses native scale (no resizing) with light colour and scale augmentation, on 5,176 training / 430 validation tiles from 800 twin scenes.

On the device it is exported to ONNX (12.8 MB) and run by ONNX Runtime on the CPU. ONNX and PyTorch agree to within 0.0013 on the raw head output. Detections are merged across tile seams, then measured in mm from the box or mask (length, width, area, distance to the nearest edge or hole).

### 4.4 Severity and verdict

`configs/spec_steel.yaml` maps each defect to a severity tier and a disposition. The part verdict is the worst disposition:
PASS < CLEAN_AND_REINSPECT < REWORK < HOLD_FOR_QA < REJECT.

The rules include:
- size limits by plate thickness (EN 10163-2 / ISO 7788 tables);
- ISO 7788's 40 mm clustering;
- edge and hole proximity;
- rust grades;
- "contamination never rejects" (a stain means clean and re-inspect).

Three rules make the device honest about what it cannot see:
1. **The camera does not measure depth.** Scratches, dents and pits above minor go to HOLD_FOR_QA for a gauge check.
2. **Confidence is not severity.** A detection below 0.5 means HOLD_FOR_QA, "uncertain, operator to confirm". A crack keeps REJECT only if the second burst of the inspection confirms it (same class within 3 mm); an unconfirmed low-confidence crack is held (HOLD_FOR_QA), never dropped (P1h, adopted 22 Nov).
3. **Hood-only certification.** In open mode the device never outputs PASS or CLEAN_AND_REINSPECT; those become HOLD_FOR_QA with "fit the hood and inspect again to certify" (adopted 22 Nov).

All thresholds are **illustrative demo values** to be agreed with a customer.

### 4.5 Pipeline B: Llama via ExecuTorch

- **Model:** Llama 3.2 1B Instruct as an ExecuTorch `.pte` (XNNPACK); Meta's `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8` checkpoint (4-bit weights, group size 32, 8-bit embeddings), exported on the Pi with ExecuTorch v1.6.0-rc1 (`80264e11d`); 1.08 GB `.pte`, recorded in `models/README.md`.
- **Runner:** ExecuTorch's `llama_main` runner as a separate process (`--advisor executorch`), or the in-process `TextLLMRunner` (`--advisor executorch_py`). Both run on the Cortex-A76.
- **Who writes what (redesigned 22 Nov after the first Pi run, §8):**
  - The **rules engine** writes the first answer's facts: the verdict, the deciding defects with their measurements and the rule in words, and the numbered steps. They are exact by construction.
  - **Llama** writes the cause-and-prevention note for the main deciding defect, and answers "Why?" and open questions (Hindi too).
  - "How to fix?" and requests to ship, paint over, accept or override get the rules engine's exact answer. On the Pi the 1B model refused these questions or invented repair steps.
  - A RETAKE or a PASS has nothing to explain, so the rules engine answers alone; Llama still takes follow-ups.
- **Prompt:** built from the JSON record (never the image).
  - The first answer gets a short cause note (~610 tokens with the system prompt). The note's opening words ("The scratch most likely comes from") are pre-filled, so it names the right defect and blames no one.
  - Follow-ups get a plain-text brief of the record: verdict, the deciding defect and rule, steps, other findings, capture, cause. It is about 860 tokens, against about 1,340 for the earlier JSON prompt.
  - `--max_new_tokens=160`. The app builds the chat template and drops the oldest turns to fit the model's context.
- **Guardrails** (`llm.check_steel_reply`), on every model answer. An answer is rejected if it:
  - contradicts the verdict or a defect's disposition, says a held part "requires repair", or calls the verdict wrong;
  - names a defect type that is not in the record;
  - quotes a number that is not in the record, or a depth or depth rule that is not verbatim in the record (the Phase 1 camera measures no depth);
  - uses a banned phrase or cites a reference that is not on the whitelist;
  - in the cause note: contains any number, or does not name the deciding defect.

  A rejected answer gets one rewrite, with the broken rule named; if that also fails, the labelled template answers. Answers stream to the UI after the verdict is shown, and the UI always shows who answered ("Llama 3.2 1B · ExecuTorch", "rules engine" or "TEMPLATE").
- The UI and this report carry "Built with Llama", as the Llama 3.2 licence requires.

## 5. The interface: one JSON record

`trinetra.inspection/3.0` (`schemas/`) holds:
- the part (material, thickness, size);
- the capture (device mode, gate result and every gate metric);
- each defect (type, confidence, box, location in mm, size, edge distance, severity, disposition, the rule that fired);
- the verdict and its instruction;
- the backends.

It is the only thing Pipeline B receives, and it is what QA would archive.

## 6. Operator experience

`trinetra-app` serves an 800 × 480 station page with:
- an **Inspect** button;
- the verdict banner (colour + buzzer on REJECT / HOLD / RETAKE);
- the overlay with boxes and mm;
- the operator steps;
- the streaming Llama explanation with follow-up questions.

A history view shows the log. A RETAKE always says why and what to do. Screenshots: **TBD** (from the Pi run).

## 7. Results

All results are from the **digital twin**, a physically based Blender model of plates, defects, device and lighting (§9). No real-camera numbers are claimed.

### 7.1 Lighting experiment (pre-registered)

**Set-up:**
- 15 unseen test plates (seeds ≥ 900100: never trained, tuned or calibrated on; 6 materials; 3 defect-free plates).
- 7 lighting set-ups:
  - C0 dark;
  - C1 office (~350 lux);
  - C2 warm bulb close by;
  - C3 window daylight;
  - C4 low side lamp with a moving hand shadow;
  - C5 a lamp reflected in the plate;
  - C6 direct sun (~45,000 lux).
- Each with the hood and open, 2 variants each, for both capture methods: 840 captures.

**Criterion (fixed before the test):**
- **Hood**, every condition:
  - retake ≤ 5 %;
  - F1 ≥ F1(C0) − 0.05;
  - verdict agreement with C0 ≥ 95 %;
  - **0 silent failures** (the gate lets a capture through and the verdict is wrong).
- **Open:** 0 silent failures.

| Method | Device | Wrong verdict, no warning | Unsafe¹ | Retakes | F1 range |
|---|---|---|---|---|---|
| New (burst + gate) | **hood (product)** | **2 / 210** | **0** | 0 / 210 | 0.83–0.85 |
| New (burst + gate) | open | 12 / 210 | 2 | 142 / 210 | 0.72–0.89 |
| Old (one auto-exposed frame) | hood | 14 / 210 | 0 | no gate | 0.91–0.92 |
| Old (one auto-exposed frame) | open | 16 / 210 | 0 | no gate | 0.80–0.91 |

¹ Milder than the truth and not held.

**With the hood (the product):**
- Every condition from dark to direct sun meets the retake, F1 and agreement limits.
- The verdict equals the ground truth in 30/30 captures in five conditions and 29/30 in two (C1, C4).
- False alarms on defect-free plates: **0**, against 0.33 per plate for the old method.
- The processed hood images differ from the dark reference by 2.7–2.8 DN in every condition, about the sensor-noise level, including direct sun.
- **Important caveat: this is by construction.** In the twin a seated hood blocks ambient light almost completely (measured ambient ratio ≈ 0 in every hood row). The 7 hood rows are therefore 7 sensor-noise draws of the same image: they test **repeatability**, not light leakage. Real-world leakage (warped plate, rim gaps, the hood on a plate edge) is untested and is the first Phase 2 measurement.

**Result: FAIL by 2 captures.**
- Both silent failures are on one plate. A low-confidence (0.27 / 0.32) false "crack" at the plate edge appears in one variant only. By the never-downgrade-a-crack rule it gives REJECT, where the truth is HOLD_FOR_QA. That is stricter than the truth, but still a wrong verdict, and the criterion counts it.
- **Root cause (verified by independent re-analysis):** a model false positive, not lighting.
  - The spot is unlabelled mill-scale texture at the plate's left edge: a dark flake arm with a bright spalled speck on its rim. The twin's crack generator starts 40 % of cracks at a plate edge, so this texture looks like an edge crack.
  - The burst model scores it 0.12–0.32 in **all 14** hood captures of that plate, including the dark room. The score does not follow lux (Spearman ρ = 0.15).
  - Sensor noise pushes it over the 0.25 detection threshold about 15 % of the time.
- **Fix adopted (22 Nov):** two-burst confirmation (P1h, §4.4). It was chosen after seeing these 2 failures, so the result above stays the evaluated result and the fix must be confirmed on fresh plates. Re-scored on the recorded detections, hood silent failures go to 0/210, with no new unsafe verdict and no change to the 280 captures with real cracks. The root-cause fix, retraining with this texture as a hard negative, is Phase 2.

**Open (no hood):**
- The gate refuses most bright scenes: 142 of 210, including every direct-sun capture.
- On two dark mill-scale plates the open optics missed the defects even in the dark room. Twice the result was unsafe (PASS where the truth was REJECT).
- No verdict rule can fix a missed defect: every policy tested left 2–3 unsafe. So **the product gives verdicts only with the hood (adopted 22 Nov)**: in open mode PASS and CLEAN_AND_REINSPECT become HOLD_FOR_QA with "Open mode is screening only: fit the hood and inspect again to certify". The two unsafe PASS verdicts would therefore have been held.
- Likely cause (one part contested by a verifier): open-device burst images were under-represented in training (77 of 511 burst training scenes), and the failing plates are the only glossy hot-rolled ones.

**Old method:**
- It has no gate, so it never refuses.
- It fails silently in 6 of 7 open conditions.
- Its hood rows are identical across conditions because the twin reuses one noise draw for it. That flatters it, and is a known limit of this comparison.

**Detection quality, new method with the hood, pooled over 7 conditions:**

| Class | Recall |
|---|---|
| scratch | 126/126 = 1.00 |
| dent | 70/70 = 1.00 |
| crack | 82/84 = 0.98 |
| corrosion | 70/84 = 0.83 |
| stain | 56/70 = 0.80 |
| pit | 77/98 = 0.79 |

Precision is 0.77–0.79, lower than the old method's 0.90. Part of this is detections of real defects that the ground truth marks "not visible in the reference" (counted as false positives); the analysis is ongoing.

Full tables with Wilson intervals: `reports/lighting_matrix.md`. Video: `reports/demo_lighting.mp4`.

### 7.2 The device software equals the evaluated pipeline

The on-device app (`trinetra-app`), fed the 28 showcase twin bursts as its camera, reproduced the evaluation's verdict for **28 / 28** captures, including the failure and every RETAKE. On the Raspberry Pi 5 (22 Nov, `reports/pi_phase1_raspberrypi_20261122-170051/`) the same check gave **28 / 28**.

### 7.3 Experiments that were tried and not adopted

| Experiment | Result | Decision |
|---|---|---|
| **I-JEPA** (cross-lighting self-supervised ViT-Tiny) as an anomaly map | Region AUROC 0.975 hood / 0.957 open vs ResNet18 PatchCore 0.965 / 0.957. It missed the +0.02 adoption margin, was less lighting-invariant, and too slow for the Pi in its better form | Not adopted (`docs/JEPA.md` §10) |
| **PINN depth** (photometric stereo with a physics-informed height field) | Dent depth error 0.05 mm (ring halves) to 0.02 mm (6 sectors), 2–3× better than classical integration; pits unreliable | Twin only; candidate for Phase 2 depth (`docs/twin/PHYSICS_PINN.md`) |

### 7.4 On-device performance (Raspberry Pi 5)

| Metric | Pi 5 | Mac M3 CPU (reference) |
|---|---|---|
| Vision latency p50 / p95 (camera to verdict) | **Hailo-10H: 1665 / 2106 ms** (detect 47 ms, burst processing 1110 ms, twin file load 547 ms). CPU ONNX: 2856 / 3331 ms (detect 1329 ms) | 617 / 716 ms |
| Vision throughput | **0.57 inspections/s with the Hailo-10H** (0.39 on the CPU) | 1.8 inspections/s |
| Peak memory (app) | 0.98 GB with the Hailo (1.08 GB on the CPU) | 1.7 GB |
| Llama time to first token / decode tokens/s | Cause note: 5.4 s p50 (model clock; 8.0 s wall incl. 2.5 s load) / 11.0 tok/s at a 611-token prompt; whole note 10.3 s; camera to advice 13.5 s p50, 13.9 s p95 (`reports/pi_llm_final4_20261122-183748/`). The first design: 15.8 s / 6.7 tok/s at ~1,340 tokens, about 54 s per answer | – |
| Peak memory (Llama runner) | 1.38 GB | – |
| CPU temperature start / max | 58 / 67 °C, no throttling (`get_throttled=0x0`) | – |

Source: `reports/pi_phase1_<host>_<date>/bench_*.md` (Hailo: `reports/pi_phase1_raspberrypi_20261122-201042/`).

**Hailo-10H detector (22 Nov).** The ONNX model, compiled to a HEF with DFC 5.1.0 (`models/README.md`), detects in 47 ms per inspection instead of 1,329 ms: 28× faster. It gives the ONNX verdict on 27 of 28 twin bursts. The one difference is the hood capture `s900100_C1h`, where the CPU model's known edge false positive (a 0.32 "crack" on mill-scale texture, §7.1) is not detected after INT8 quantisation. That verdict therefore changes to the true HOLD_FOR_QA. This is a side effect of quantisation, not a fix: the hard-negative retraining in §7.1 remains the cure. Confidence scores move by up to about 0.07 after quantisation.

## 8. Example advice, and what the first Pi run taught us

**First Pi run** (`reports/pi_phase1_raspberrypi_20261122-170051/`, the earlier design where Llama restated the whole record): Llama answered 7 of 8 inspections, and **3 of the 7 answers had factual errors the guardrail then missed**:
- it explained a crack REJECT as a scratch;
- it inverted the scratch-depth rule ("0.2 mm or less, repair");
- it invented a measured depth.

The verdicts on screen were right every time, because they come from the rules engine. Two more rounds of prompt and guardrail changes on the Pi showed the same pattern: whenever the 1B model restates numbers or rules, it garbles some of them (for example "unsure because of the 88 mm width"). So the facts moved to the rules engine and Llama got the job it does reliably (§4.5).

**Final Pi run** (`reports/pi_llm_final4_20261122-183748/`, 16 inspections):
- 9 of 9 cause notes came from Llama, and none failed the guardrail;
- the 7 RETAKE and PASS results were answered by the rules engine;
- one slip remained in that run: with two deciding defects, Llama gave the scratch's cause to "scratch and corrosion". Fixed the same evening: the note now covers the main deciding defect only (checked on the Pi on two plates).

One answer from the Pi (part `s900100_C0o_v0`, open mode, illustrative thresholds):

```
Verdict: HOLD_FOR_QA. Decided by D1 scratch (critical): width 0.6 mm, at or over the 0.5 mm limit;
D2 scratch (major): width 0.28 mm, at or over the 0.2 mm limit; depth not measured.
What to do:
1. Do not paint over it.
2. QA checks the depth with a gauge. On 3-8 mm plate up to 0.2 mm deep may stay; 0.2-0.4 mm only if
   the affected area is 5 % or less; deeper must be repaired.
...
7. Put the part in the HOLD tray. QA checks the depth or confirms the finding with a gauge or dye penetrant.

Likely cause and prevention (Llama): The scratch most likely comes from parts sliding against each other
or on a dirty bench. To prevent it next time, you should make sure to deburr the fixtures and guides before
using them, and use separator sheets or protective film to prevent parts from sliding against each other.
```

**Follow-ups on the Pi** (crack REJECT `s900100_C1h_v0`):
- "How to fix?" gives the crack's steps from the rules engine.
- "Can I just paint over the crack and ship it?" gets "The verdict REJECT is final … call QA".
- "Why?" and "Why does a crack matter on a steel plate?" get Llama answers in 31–35 s that pass the guardrail. They are general and loosely worded: one says the unsure crack "is not a clear indication of a defect". Open-question quality is the weakest part of Pipeline B. A larger model, or a fine-tune on station Q&A, is Phase 2 work.

## 9. The digital twin

**Plates.** Blender 5.2 Cycles renders procedurally textured plates of six finishes:
- hot-rolled with mill scale;
- cold-rolled;
- stainless 2B and brushed;
- galvanised;
- aluminium.

Shading uses GGX microfacet models.

**Defects.** The six classes have physical geometry (height fields for dents, pits and scratches) and exact ground truth: class, mask, size in mm, depth.

**Device.** The device geometry, LED positions and hood are modelled from the CAD. The seven lighting conditions are modelled with real luminance and illuminance levels.

**Sensor and processing.** A shot-and-read noise model matching the camera's datasheet is applied in post. The same `lightproc.py` code then runs in the twin and on the Pi.

**Validation.** The rendered photometry matches analytic formulas to within 2.4 %.

**Data.**

| Set | Size | Use |
|---|---|---|
| Training | 800 scenes | Detector training |
| Test matrix | 15 plates × 14 rows | Lighting experiment |
| JEPA | 48 paired plates | JEPA experiment |

All renders were made on a rented L4 GPU (about 3.5 h). Seeds and regeneration: `DATASETS.md`.

## 10. Limitations and next steps

**Limitations:**
- **Twin only.** Real specimens (aluminium, mild and stainless steel coupons with seeded defects) are Phase 2. The sim-to-real gap is the main open risk.
- **Strict criterion not met in the hood**, by 2 of 210 captures (conservative errors). Adopted fix (P1h): two consecutive bursts form one inspection, and a low-confidence detection that the second burst does not confirm causes at most HOLD_FOR_QA. Confirmed cracks still REJECT, and a refused second burst means the pair is retaken. On the recorded data it removes both failures, adds no unsafe verdict, and changes none of the 280 captures with real cracks. Because it was chosen after the test, it needs confirmation on fresh test plates (seeds ≥ 900200).
- **Open mode must not give verdicts** (unsafe misses on dark mill scale).
- **No depth measurement in the app yet.** PINN depth is twin-validated only.
- **Illustrative thresholds.** Decision support, not a certified inspection system.

**Phase 2 roadmap:**
- real-specimen validation;
- a Hailo-10H detector;
- PINN depth with a 6-sector ring;
- the confirmation-burst policy;
- a ResNet18 PatchCore anomaly head after threshold calibration;
- for parts inspected at a distance: a telephoto overhead camera as a detection-only preview that never returns PASS. Depth and verdicts stay in the hood.

**Why not interferometry or profilometry at a distance** (`docs/research/2026-11-22_distance_profilometry.md`; desk research and simulation, not measured):
- **A green-LED interferometer cannot work at 0.3–3 m.** It needs the surface inside a ~4 µm coherence gate against 30–200 µm of hand tremor, a reference arm matched to ±2 µm over the standoff, and ~7,600 frames for a 0.5 mm scan (254 s at 30 fps).
- **The spec needs only about 25 µm of depth σ**, roughly 10⁵× coarser than interferometry.
- **Deflectometry** keeps the phase-shifting fringe idea. It grades bright finishes only on a fixed, shrouded station with the screen within ~0.3–0.6 m, and gives no usable fringes on mill scale.
- **Fringe projection** grades only at ~0.3 m in ≤ 10 klx.
- **The camera can be far** (an HQ camera + 50 mm lens gives 0.093 mm/px at 3 m), **but the lights cannot.** Light from the lamp falls as 1/D² while ambient light does not, so lighting robustness needs the illumination close to the plate, as in the hood.

## 11. Arm tools used

| Tool | Use |
|---|---|
| ExecuTorch | Llama 3.2 1B on the Cortex-A76 (XNNPACK backend; KleidiAI kernels where the build includes them) |
| Raspberry Pi 5 (Cortex-A76) | Both pipelines |
| ONNX Runtime (Arm64 CPU) | Detector |
| HailoRT | Optional NPU path |

## 12. Licences and references

**Licences.** The project is AGPL-3.0 (because of Ultralytics YOLO); `NOTICE` lists third parties.

**Built with Llama.** Llama 3.2 is licensed under the Llama 3.2 Community License, © Meta Platforms, Inc.

**Standards referenced** (not reproduced): EN 10163-2 / ISO 7788, ISO 8501-1/-3, IS 2062, IS 9954.

**Project documents:**
- `docs/twin/DESIGN_STEEL.md` (design contract)
- `docs/APP.md` (app)
- `reports/` (all results)
