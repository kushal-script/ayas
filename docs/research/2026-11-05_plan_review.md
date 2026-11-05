# Independent reviews of Phase1_build_plan.md (first draft, 5 Nov 2026)

All points below were folded into the revised plan the same day.



---

## Lens: compliance_schedule

# Review of Phase1_build_plan.md: rules, completeness and schedule

## Blockers

**B1. The Hailo go/no-go 2 comes after the TEST evaluation (§7.7 step 5, §11 Wed 18, §14).** The plan evaluates TEST on Mon 16 but decides on Wed 18 which detector ships. That is also freeze day, and the video is shot the next day. If the decision goes to the CPU fallback, the metrics, the benchmarks and report §8 all have to be redone. The CLIP-embedding fallback is new, unbudgeted work landing on freeze day.
*Fix:* "Go/no-go 2 is **Fri 13**. The custom HEF passes `hailortcli run`, and its mAP on a TRAIN-val split is within 5 points of ONNX FP32. If not, ship YOLO on the CPU and stop Hailo work. Delete the CLIP fallback." Run the TEST evaluation **once**, on Mon 16, with the frozen model.

**B2. The dataset can't be recaptured, and the TEST plate gets dyed (§4.3 step 5, §7.6, §11 Tue 10/Thu 12).**
- All 160 sequences are shot one day after the capture tool is written, before fusion, flat-field and golden have run on any real frame.
- Only warped grids are saved, so dark subtraction and the B+G channel choice can't be redone later.
- Penetrant goes onto P5 two days later. After that, P5 can't be recaptured, and the plate shown live in the video carries trapped dye.

*Fix:*
- **Mon 9:** pilot-capture 3 sequences of P3. Run fusion and check the contrast, then decide whether to do the full capture.
- **Capture:** save plate-ROI raw Bayer as uint16 (about 1700², 8 states, `savez_compressed`, about 5 GB in all, calc). Warp offline.
- **Penetrant:** S1 only on Thu 12, P4 on Sat 14 or later, **never on P5** before the video is shot.

**B3. There is no end-to-end build until Fri 13–Sat 14 (§11).** Every requirement M1–M7 depends on one late integration day.
*Fix:* add a **walking skeleton** milestone on the Pi for **Wed 11**: capture → anomaly map + geometry typing (elongation > 4 = scratch/crack, round = dent/pit, colour = corrosion) → rules → JSON → Llama → banner/buzzer → SQLite. YOLO only replaces the typing step later. This also cuts audit item 5 (features/classify, 13 h) to about 4 h.

## Major

**S1. The workload doesn't fit the calendar.**
- The 124 h audit estimate leaves out fusion, the penetrant rework, evaluation tooling, schema v2.1, hardware and specimens (about 35 h), the report (about 25 h) and the video (about 12 h). The real total is about 200–215 h (calc).
- Role B carries about 95 h between Sun 8 and Wed 18, roughly 9 h a day through Deepavali. Role A has nothing scheduled Mon 16–Wed 18.

*Fix:*
- A takes the label review, ground-truth counting for the evaluation, the vision and power benchmarks, and `bench_vision.py`.
- Write `npu.py` against the **stock** `yolov8n.hef` on Sat 7–Mon 9, not on Fri 13.
- C develops the advisor on the Mac. The macOS arm64 ExecuTorch wheel includes `TextLLMRunner` (executorch research, VERIFICATION). This takes C off the only Pi.
- Solo team: infeasible. Two people: use the cut order below.

**S2. Parts and holidays (§2 step 7, §11 Sat 7).** "Arrive by Sat 7" overlaps Deepavali: Naraka Chaturdashi on Sun 8, and Balipadyami, a Karnataka general holiday, on **Tue 10**, which is capture day. ([Sakshi Post](https://www.sakshipost.com/news/national/karnataka-2026-holiday-calendar-released-check-general-restricted-local-holidays), [ClearTax](https://cleartax.in/s/karnataka-governmnet-holidays-list-2026))
*Fix:* "Buy the 12 V adapter, MOSFETs, extrusion, white modules and MDF **in person on SP Road Thu 5/Fri 6**. Plan Sun 8 and Tue 10 at half capacity."

**R1. Llama export (§1.2 rank 6, §8.1, §14).**
- **Rule:** a third-party `.pte` meets the rule as written (Llama-family, ExecuTorch, on the device). It is weak on originality and on "use of Arm tools", and it depends on a HF repo that can change.
- **Wrong fallback:** the executorch-community `.pte` was built before 1.0, so it is *less* likely to load than the software-mansion file, which was built for 1.4.1.

*Fix:*
- Make the team's own export **mandatory (M1b), due Tue 10**.
- If Meta hasn't approved by Mon 9: export 8da4w from the ungated mirror `unsloth/Llama-3.2-1B-Instruct` with optimum-executorch. It ships HF safetensors only and no `original/` folder ([HF API](https://huggingface.co/api/models/unsloth/Llama-3.2-1B-Instruct), [optimum-executorch](https://github.com/huggingface/optimum-executorch)). Whether `TextLLMRunner` loads its output is UNVERIFIED, so test it the same day.
- If the prebuilt file won't load on 1.5.1, run executorch **1.4.1** with the torch version in v1.4.1's `torch_pin.py`.
- Pin `--revision <commit>` and record the SHA-256 in `models/README.md`. Credit software-mansion in NOTICE.

**R2. The AGPL licence decision is left open (§13.1).** Training uses Ultralytics, and the ONNX and HEF come from AGPL weights.
*Fix:* "LICENSE: **AGPL-3.0**." NOTICE should list Ultralytics (AGPL), Hailo Model Zoo (MIT), the Llama 3.2 licence and "Built with Llama", and the `.pte` source. Never commit the DFC wheel. In §15, add that a product version needs an Ultralytics Enterprise licence or a permissively licensed detector.

**C1. TEST data leaks into the golden reference (§7.4 step 2, §11 Tue 10).** The golden is "from P1/P2", but P2 is the TEST plate used for the false-reject rate in §12.
*Fix:* "Golden from **P1 only**: 2 faces × 2 orientations × 4 placements = 16 captures."

**C2. The class map has gaps (§4.1, §7.4, `spec.yaml`).**
- P4 and P5 carry a burr, an edge chip and abrasion patches, but none of them has a YOLO class.
- `spec.yaml` has no `corrosion` or `contamination` class.
- Nothing produces CLEAN_AND_REINSPECT unless UV mode is used.

*Fix:*
- Add an `eval_class` column to `ground_truth.csv`: abrasion → scratch, edge chip → dent, burr → unknown (expected result HOLD_FOR_QA).
- Add `corrosion: {measure: area_mm2, limits: {minor: 2, major: 5, critical: 20}, reworkable: false}`.
- Contamination comes only from UV mode; otherwise it should come out as HOLD_FOR_QA. The video shot at 2:20 already shows that.

**C3. "Real time" has no target (M7).**
*Fix:* add to M7: "verdict ≤ 3 s p95 after INSPECT; first Llama token ≤ 6 s; full advice ≤ 20 s; ≥ 10 parts/min" (est.). Measure these on Mon 16.

**C4. Judges can't run the application without the rig or weights (§13.1).**
*Fix:*
- Add `trinetra replay data/real/sample/`, which runs the full pipeline on recorded sequences.
- Attach the team's **own** `best.onnx` and `.hef` to GitHub Release `v1.0-phase1`. Fetch the `.pte` with a pinned script.
- Put the TEST `.npz` files on the Release or Drive so the evaluation can be checked.

**C5. Advice quality has no owner or date (§12).** It also depends on the mentor being free.
*Fix:* C runs it on **Tue 17**:
- 30 records, Llama against the template baseline, scored blind by 2 people (plus the mentor if free);
- automatic checks: every number Llama quotes matches the record, and 0 replies contradict the verdict.

## Minor

- **Calibration tiles:** 1,024 calibration tiles is more than the roughly 864 TRAIN tiles. Use `calibset_size=864` or flipped TRAIN tiles, never TEST tiles.
- **§6.1 example:** `spec.yaml` has no dent `zone_limits`. 3.4 mm is "major" → REJECT, not "critical".
- **Llama licence naming:** name the export `Llama-3.2-1B-Instruct-SpinQuant-TriNetra.pte` (derivative names must start with "Llama").
- **Show the backend:** the Result screen shows "ExecuTorch 1.5.1 · <pte> · 4 threads" or "TEMPLATE FALLBACK". Template text must never look like Llama output in the video.
- **Offline proof:** prove M7 with Wi-Fi off, Ethernet unplugged and an `ip link` readout on the System screen. Don't show the LAN laptop view.
- **Operator flow:** add **Acknowledge / QA sign-off** buttons (name + action) to the Result screen, logged to SQLite. This adds to both operator experience and logging.
- **Date mismatches:**
  - §4.4 says the bench tests are on Day 3–4; §11 has them on Wed 11.
  - The Fri 6 ground truth can't include corrosion, which is still growing. Measure the corrosion spots on Mon 9.
- **Video:** the penetrant indication on S1 fades. Do a fresh S1 penetrant run (about 45 min) on the morning of Thu 19.
- **IMX500:** the plan handles it correctly (camera only until the organisers confirm). Keep it that way.

## Critical path

Parts → rig and wiring → focus and exposures → capture tool → **pilot gate (Mon 9)** → dataset → labels → YOLO → HEF → `npu.py` → TEST evaluation → report results → video.

## Cut first if behind (in this order)

1. UV mode, hood and UV bench test
2. IMX500 and PatchCore
3. Hindi and trend hints (keep the template shift summary)
4. Thread sweep and the idle-vs-loaded matrix (keep one configuration)
5. Fusion ablation against white-only
6. Custom HEF (fall back to YOLO on the CPU)

**Never cut:** the skeleton, Llama via ExecuTorch, logging, the TEST evaluation, the "Two separate pipelines" section, the video.

## Corrected must-ship path

| Day | Work |
|---|---|
| Thu 5 | Admin; parts bought in person; defects made; corrosion started; HF and Hailo accounts |
| Fri 6 | Pi bring-up (morning); `.pte` load test on Mac and Pi; ground truth |
| Sat 7 | Rig build; stock HEF Colab→Pi (go/no-go 1); advisor service on the Mac |
| Sun 8 (½) | Single-view config; capture tool |
| Mon 9 | Focus and exposures; `npu.py` on the stock HEF; **pilot gate**; Meta deadline for the own export |
| Tue 10 (½) | Raw-ROI dataset capture; label projection; UI skeleton; own export |
| Wed 11 | **Walking skeleton on the Pi**; YOLO training; contamination session |
| Thu 12 | Custom HEF compile; S1 penetrant |
| Fri 13 | **Go/no-go 2**; penetrant code |
| Sat 14–Sun 15 | Integration; tuning on TRAIN; P4 penetrant |
| Mon 16 | TEST evaluation (once); benchmarks |
| Tue 17 | Advice evaluation; report results |
| Wed 18 | Freeze; README, LICENSE, Release assets |
| Thu 19 | Fresh S1 penetrant; video shoot |
| Fri 20 | Video edit; report proofread |
| Sat 21 | Buffer |
| Sun 22 | Submit |

Files reviewed:
- <repo>/docs/Phase1_build_plan.md
- <repo>/configs/spec.yaml
- <repo>/src/trinetra/rules.py
- <repo>/src/trinetra/llm.py
- the six research files in <repo>/docs/research/


---

## Lens: technical

# TriNetra Phase 1 plan: technical-correctness review

I found one blocker, nine major problems and ten minor ones. The checks that passed are listed at the end.

## Blocker

**B1. §7.1 Capture. Errors here can't be fixed later: the Tue 10 dataset can't be re-shot once P4 and P5 carry penetrant.**
- **Raw format:** `raw` is given with no format. On a Pi 5 that returns the compressed `PISP_COMP1` stream, so the "linear raw" dark subtraction will be wrong.
- **Exposure changes:** a new `ExposureTime` takes several frames to apply. `flush=True` only checks that `SensorTimestamp − ExposureTime` is after the call time. It does not check that the new exposure is in effect ([picamera2.py](https://raw.githubusercontent.com/raspberrypi/picamera2/main/picamera2/picamera2.py)).
- **Dark frames:** one dark pair can't serve lit frames at 10, 20 and 30 ms, because ambient light scales with exposure.
- **Fix:**
  - Use `raw={'format':'SRGGB10','size':(4056,3040)}` and `.view(np.uint16)`.
  - After `capture_request(flush=True)`, keep capturing until `abs(md['ExposureTime']-target)<100`.
  - Take a dark frame at every exposure in use, or use one exposure for every state.
  - Save each light's warped uint16 grid plus the homography and the dark frames, not just the fused image. Copy the TEST raw frames to a USB drive.

## Major

**M1. §3.1/§7.2 Plate outline from BLUE-40.** The 40° light is also off-specular. Its mirror point is at 150·200/325 = 92 mm, outside the 50 mm half-plate, so the plate is only diffusely lit, just as under grazing light. "Even view" and "albedo" are therefore wrong.
- Fix: test Otsu on BLUE-40 on Day 1.
- Fallback: a fixed homography from 4 white fiducial dots on the base beside the L-stop, refined on each capture.
- Rename channel B to "40° diffuse".

**M2. §7.3 Relief channel.** On flat bare aluminium the dark-field frames are close to zero, so (E−W)/(E+W) is noise, or a division by almost zero, over most of the plate. `abs()` also throws away the dent/bump sign that §7.4 says it measures.
- Fix: G = clip(255·(|E−W|+|N−S|)/(E+W+N+S+ε)), with ε = the median 4-bar sum on P1.
- Keep the signed (E−W)/(E+W+ε) and (N−S)/(N+S+ε) as separate maps for the relief sign.
- Scale R and G with fixed constants taken from TRAIN, never per-image min-max. Put the same floor on the flat-field divisor.

**M3. §3.1 Hood "only for UV".** The camera looks down at a near-mirror, so it sees the ceiling lights and its own silver arm reflected. That shot noise, plus room light changing over the 2 s sequence, will swamp the weak dark-field signal, and saturated pixels can't be subtracted.
- Fix: make the hood standard in every mode.
- Use black 2020 extrusion and put black card around the lens.
- Reject a sequence if the dark-frame mean, or the D0/D1 difference, goes over a threshold.

**M4. §7.4 and Tue 10: test data leaks into the golden reference.** "Golden from P1/P2" puts test plate P2 into the reference, and §12 then measures false rejects on P2. Fix: build the golden from P1 only.

**M5. §4.1 Specimen shapes.** After S1 and S2 are cut off, P4 is 100 × 60 mm, and S1 is 100 × 20 mm. The 100 × 100 plate fit, the 1,375 px grid, the L-stop and the golden don't fit either one, and S1 is the penetrant shot in the video.
- Fix: give each specimen its own `plate_w_mm`/`plate_h_mm` and a validity mask.
- For P4, use a 1375 × 875 px grid with 3 × 2 tiles.
- Image S1 on a 100 × 100 black carrier card with fiducials.

**M6. §7.6 Dataset arithmetic.**
- 2 orientations × 4 placements is 8, not 16. That gives 48 TRAIN sequences and 432 tiles.
- Even with 96 sequences, 864 tiles is fewer than the 1,024 calibration tiles §7.7 asks for.
- P1 is a third of the TRAIN sequences, so at least 33 % of tiles are good-only, not 5–10 %.
- Fix: write "2 orientations × 8 re-seated placements = 16". Make the calibration set 1,024 random 640 px crops from the TRAIN grids, and subsample P1 tiles to about 10 %.

**M7. §7.4 Tiles vs long scratches.** A 30–60 mm scratch is 375–750 px long at 0.08 mm/px, longer than one tile.
- Fragments from neighbouring tiles overlap with IoU = 272/750 ≈ 0.36.
- With an NMS threshold of 0.3, the shorter fragment is dropped and the length comes out short. With 0.5–0.7, both survive and the scratch is counted twice.
- Fix: after NMS, merge same-class boxes that touch across a tile seam. Measure length and width from the connected region of the whole-plate anomaly mask, not from the box.

**M8. §3.2 Buzzer.** The usual "5 V active buzzer module" is low-trigger, with an S8550 PNP whose emitter sits at VCC.
- With VCC at 5 V, a GPIO high of 3.3 V still leaves V_EB ≈ 1.7 V. The buzzer never switches off, and about 1 mA flows back into the GPIO pin.
- [SunFounder's PNP circuit](https://docs.sunfounder.com/projects/superkit-v3-pi/en/latest/Lesson_6_buzzer.html) ties the emitter to 3.3 V for this reason.
- Fix: power the module from 3.3 V (pin 1), or switch the buzzer with the spare MOSFET channel.

**M9. §7.4 classes vs §1.2 and video shot 2:20.** "Stains count as corrosion" plus "corrosion → REJECT" means the marker-dot/oil test will most likely come out as corrosion → REJECT, not unknown → HOLD. `spec.yaml` also has no `corrosion` entry, so `rules[kind]` raises a KeyError.
- Fix: label the P4-B stain as `stain` (CLEAN_AND_REINSPECT), or leave it unlabelled.
- Add `corrosion: {measure: area_mm2, …, reworkable: false}` to the spec.

## Minor

- **m1. §4.2/§7.2/§12 Resolution and tiling.**
  - The analysis grid is 0.08 mm/px, so a 0.3 mm defect is about 3.75 px, not 5. The blue Bayer plane has a 0.128 mm pitch.
  - In §12, write "0.08 mm/px grid (0.064 mm/px sensor)".
  - Stride 368 overruns the grid: 2·368 + 640 = 1376. Place the tiles at 0, 368 and 735.
  - Change `mm_per_px` in `rig.pi.yaml` from 0.1 to 0.08.
- **m2. §6.1 The record contradicts the code.**
  - `location_mm` is a list, but the schema and `rules.py` expect `{x, y}`.
  - y = 18.4 is outside `sealing_face` (0–15 mm), so the zone is `general`.
  - A 3.4 mm dent is "major" (limits 3.0/6.0), not "critical".
  - D2 has no `location_mm`, so `judge_defect` raises a KeyError.
- **m3. §7.8 Redness.** R/(G+B) is about 0.5 on clean developer, which is already above the existing 0.45/0.22 thresholds, so the whole plate would be flagged. Keep `penetrant.py`'s (r−(g+b)/2)/brightness, which is 0 on developer.
- **m4. §3.2 Ground pin.** The Touch Display 2 power lead uses pins 2 and 6, so use pin 14 or 20 for the 12 V ground. SPI panels such as the Waveshare 3.5" also use GPIO17 (touch IRQ).
- **m5. §7.6 Augmentation.** Ultralytics' defaults `hsv_h=0.015, hsv_s=0.7` mix the meanings of the fusion channels. Set `hsv_h=0 hsv_s=0` and `scale≤0.2`.
- **m6. §7.7 Model Zoo files.** Keep the copied yaml and alls inside `hailo_model_zoo/cfg/…`, because `base:` and `../../postprocess_config` are relative paths. Point the yaml's `alls_script` at the edited file.
- **m7. §7.7 CPU fallback.** The 67 ms figure is for YOLO26n, not YOLOv8n. Nine tiles is about 0.6–0.9 s per part on the CPU; say so.
- **m8. §8.2/§8.3 Token budget.**
  - A 400-token budget plus `reset()` can't hold system + record + previous answer + question for follow-ups. Either keep the KV cache within one inspection, or budget about 700 tokens.
  - Hindi takes about 3× as many tokens per word, so 160 new tokens will cut off an 80-word answer. Use 300 for Hindi.
- **m9. §8.4 Mac export prerequisites.** It needs Apple Silicon with macOS 14 or later, `brew install python@3.12`, `pip install -U huggingface_hub && hf auth login`, and copying `tokenizer.model` to the Pi.
- **m10. §11 Schedule.**
  - Cut S1 and S2 before the corrosion patch goes on P4-B, i.e. on Thu 5, not Fri 6.
  - Penetrant on P5 on Thu 12 puts dye into the held-out plate before the Thu 19 video. Film the P5 surface shots first, or use penetrant only on S1 and P4.

## Checked and correct

- **Optics:**
  - 1.55 µm ÷ (4.74/195.3) = 0.0638 mm/px.
  - The field is 259 × 194 mm and the plate spans 1,567 px.
  - 110 mm / 0.08 mm/px = 1,375 px.
- **Lighting angles:**
  - The bars give 14.9° at the centre and 11.3–21.8° across the plate.
  - The 40° tier gives 39.8° (32.0–51.3°).
- **Field of view and glare:**
  - At bar height the half-field is 104 × 78 mm, against a 150 mm offset, so the bars and posts are out of view.
  - On a flat plate the mirror points (125 mm and 92 mm) are off the plate, so there is no direct glare.
- **Electrical:**
  - Each UV string draws 16–21 mA, and all 12 draw 0.23 A.
  - Everything on draws about 2.2 A.
  - The header pin numbers are correct.
  - The AO3400A is fully on at 3.3 V. The D4184 and IRLZ44N aren't rated at 3.3 V, but are fine below 0.5 A.
- **Camera and Hailo:**
  - `flush=True` waits for a frame whose exposure began after the call, which is all the rolling shutter needs.
  - `Hailo(batch_size=9)` accepts a (9, 640, 640, 3) array and returns one output per tile ([hailo.py](https://raw.githubusercontent.com/raspberrypi/picamera2/main/picamera2/devices/hailo/hailo.py)).
- **Llama:** the chat template, `num_bos=0`, the CPU-index install pins and the `export_llm` overrides are all correct.
- **Penetrant:** the ASME VIII App. 8 criteria are correct as stated ([summary](https://amarineblog.com/2019/11/03/nde-penetrant-testing-pt-standards-and-criteria/)). ASME's own metric values are 1.5 mm and 5 mm.
- **Statistics:** the Wilson 95 % interval for 13/15 is 62.1–96.3 %.
- **Module count:** 12 + 4 + 4 = 20. The "2 spare" wording doesn't appear in this plan.
