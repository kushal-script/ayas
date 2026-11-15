# Pilot: the inspector under seven kinds of lighting (digital twin, full resolution)

TriNetra, Arm Bharat AI SoC Challenge PS5, Phase 1 proof of concept. Pilot run 14–15 Nov 2026 on the Blender digital twin. All numbers here come from simulation, not from real steel.

## 1. In plain words

**What works.** With the hood head (the product configuration), the new capture method gives the same image of the plate in all seven lighting conditions, from a dark room to direct sun. This holds on all four metals tested.
- The gate accepted 56 of 56 hood captures.
- Each image was compared with the noise-free, ambient-free image the device's own LEDs would give in the same render. After averaging over 1 mm, every hood capture is within −2.6 to +1.1 grey levels (DN, 8-bit). The dark room itself gives −2.1 to +0.8. The lighting adds nothing measurable.
- The detector stand-in (an "oracle", see section 3) scores F1 1.00 in every hood condition, with the dark-room verdict in 8/8 captures and no silent failure.

**The hood does most of the work.** The old method (one auto-exposed frame, the "naive" arm) is also unchanged under the hood: at most 1.8 DN from its dark-room image, in sun. The new method's own value shows without the hood:
- **Old method:** its images move 11–58 DN away from the dark room and it never refuses.
- **New method:** the captures it accepts stay within 2.4–4.1 DN of the dark room. It refuses 39 of 56 open captures, with a reason the operator can act on: too bright, lamp reflection, or lighting changed. None of its accepted captures gave a wrong verdict.

**What this pilot fixed.** Looking at the images, not only at the scores, found three defects in the method. Each is fixed and tested:
- **Pits and scratches erased (hood).** Under the hood the brightest defects were painted out. A pit cluster's contrast fell from 101 to 3 DN.
- **Moving shadow passed (open).** One accepted open capture carried a hand shadow's error (−14 DN).
- **Mirrored lamp passed (open).** One accepted open capture carried a mirrored ceiling tube (+21 DN).

**Second independent review (section 11).**
- **Reproduced.** A fresh full-resolution render of one plate gave the same results.
- **Broken, then fixed.** Adversarial bursts produced silent corruptions:
  - with the hood: one sealed and three leaking cases, now 0;
  - open: 5 % flicker on cold-rolled steel, +15 DN.
- **Three gate rules added.** None changes a pilot hood verdict.
- **Still open.**
  - In open mode, a static mirrored lamp at 2–8× the local LED level can still pass with a −10 DN error.
  - The ring LEDs are in the camera's view, and the twin cannot show flare.

**What is not shown yet.**
- **No trained detector.** No detector has been trained yet. The oracle only checks the gate and the inspection rules, so it cannot show whether a model finds defects in these images.
- **Open-mode naive result.** The old method's PASS without the hood says nothing about detection.
- **Not independent.** Four plates and two captures each is a small sample. The gate rules added here were tuned on these same plates.

## 2. Setup

**Plates.** For each material, the first test seed at or after 900010 whose plan has a 100 × 100 mm plate and at least 3 labelled defects (`tools/pilot/scan_seeds.py`). Seeds 900000–900004 had been used to calibrate the gate on 14 Nov.

| Plate | Material | Thickness | Labelled defects | Ground-truth verdict |
|---|---|---|---|---|
| s900011 | ms_cold_rolled | 2.5 mm | open crack, 2 scriber scratches, rust | REJECT |
| s900027 | ms_hot_rolled | 5.2 mm | punch dent, gouge, open crack, slit crack | REJECT |
| s900028 | aluminium | 5.7 mm | pit cluster, scriber scratch, 2 ball dents, white corrosion | REJECT |
| s900068 | ss_brushed | 4.9 mm | pit cluster, single pit, water stain (stain invisible to both arms) | HOLD_FOR_QA |

Galvanised steel is not in the pilot: cold-rolled filled the "cold-rolled or galvanised" slot.

**Conditions.** Every condition with two device modes, for 14 rows per plate and 56 rows in all. The median lux on the plate is over all captures:

| Condition | C0 dark | C1 office | C2 warm bulb | C3 window | C4 side lamp + moving hand shadow | C5 lamp reflected in the plate | C6 sun |
|---|---|---|---|---|---|---|---|
| Median lux | 0 | 504 | 1,453 | 8,866 | 4,152 | 901 | 68,202 |

- **Hood** (`h`, the product): the device v2 head, a matte-black hood resting on the part, with an 18-LED low-angle ring (dark-field). This is the product configuration.
- **Open** (`o`): the handheld without the hood, lit by the two near-coaxial LED pods, with hand shake of 0.2–1.5 px/frame.

**Render.** Blender 5.2.3 Cycles (Metal), full resolution, 64 samples. The plate ROI is about 1640 × 1630 px of the 4056 × 3040 sensor. It is rectified to a 1375 × 1375 px grid at 0.08 mm/px.
- Each scene has 2 variants (k = 0, 1). Each variant gets its own ambient multiplier, hand shake, LED ripple and sensor noise.
- That makes 112 captures per arm, 224 in all.

**Arms.**
- **burst** (new method): the five-frame LED burst B1, L, A, R, B2, then ambient subtraction, L/R combination (ring: mean; pods: luma-guided min), alignment, and the quality gate (`src/trinetra/lightproc.py`, `configs/capture.yaml`).
- **naive** (old method): one auto-exposed frame through the ISP. It never refuses.

**Cost.**
- Render: 22–55 s per scene, mean about 36 s.
- Render plus post: about 41 s per row. The 56 rows took about 39 min of wall time; one batch crashed in Blender (exit −6) and was re-run.
- Re-running post on the kept EXRs (both arms, 2 variants): 4.4 s per scene on an idle machine.
- `process_burst` on one full-resolution ROI burst: median 382 ms on the development Mac, which was shared with other jobs.

## 3. How it is measured, and what each number can and cannot say

**Oracle detector** (`training/eval_lighting.py --model oracle`).
- **What it outputs.** For each grid it returns the box of every labelled defect that post's visibility test judged visible in that grid: a local contrast of at least 4 DN in that arm's own image.
- **What it does not do.** It never looks at the pixels to find or classify anything, and it makes no false positives.
- **What its numbers test.** Oracle precision, recall, F1 and verdicts test only three things: the gate (a refused capture gives no verdict), the inspection rules, and that coarse visibility threshold. They do not show that a detector would find the defects.
- **Verdict agreement is easy here.** Three of the four plates are REJECT by truth (each has a crack, pit corrosion or similar), so any capture that keeps one serious defect visible gets the right verdict.
- **Measures.** "box" turns detections into millimetres from the padded boxes; "app" uses the app's own measurement path. Both arms were evaluated with both measures, and the verdicts were identical.

**Image-level evidence.** These are the numbers that actually test lighting invariance. All are computed on the plate, on accepted captures.
- **SSIM** of the grid against the same plate's dark-room grid (C0) from the same device.
- **|Δ|** = mean |grid − grid_C0| over pixels and RGB channels, in 8-bit DN (eval_lighting's `mad_c0`).
- **Low-pass |Δ|**: the same difference averaged over a 15 × 15 px box (1.1 mm) first. Pixel noise averages away and a lighting change does not, so this separates the two.
- **Truth error**: the burst's linear image M against the noise-free, ambient-free LED image of the same render (what a perfect sensor would see in the dark). Both are 1.1 mm low-passed and encoded with the device's own scale and gamma. This is the direct check for silent corruption: an image that changed for the wrong reason.

**The burst's 2.4 DN hood floor is subtraction noise, not lighting.**
- **Floor.** Two dark-room burst captures of the same plate, differing only in their noise and shake draw, differ by **2.53 DN**. The per-pixel noise is σ = 2.29 DN per channel, and pure Gaussian noise predicts 2σ/√π = **2.58 DN**. The hood rows C1h–C6h differ from C0h by 2.42–2.62 DN, which is the same.
- **After the 1.1 mm low-pass,** the hood rows differ from C0h by only 0.13–0.26 DN (floor 0.30 DN, worst single capture 1.06 DN). The 2.4 DN is unstructured pixel noise.
- **Why it is higher than naive.**
  - **Twice the noise.** The burst image is built from frame differences (B − A adds A's noise) at a fixed exposure, set on a white card at 57 % of full scale. Under dark-field a flat plate reads only 0.7–12 % of full scale, so the image has few electrons, and the per-image normalisation stretches the noise.
  - **Naive's own floor.** The naive arm's auto-exposure lifts the plate median to 18 % of full scale. Its own floor between two independent dark-room captures is 1.38 DN (σ 1.26 DN).
  - **Naive's 0.01–0.26 DN with the hood is not a lower noise level.** The naive variant k reuses the same noise draw in every condition and the hood blocks the ambient, so it compares an image with itself.
- **Possible fix.** A ring exposure set on the plate, 2–4× longer, should cut the burst noise by 1.4–2×. This is to be checked on the Pi.

## 4. Results with the hood (device v2, the product)

Criterion (DESIGN_STEEL G): retake rate ≤ 5 %, F1 ≥ F1(C0h) − 0.05, verdict = C0h in ≥ 95 % of captures, 0 silent failures. Result: **PASS for both arms** with the oracle (`data/twin_pilot/eval_oracle_box/lighting_matrix.md`, `.../eval_oracle_app/`).

| Row | Median lux | Burst retakes | Burst P / R / F1 (oracle) | Burst verdict = C0 | Burst silent | Naive P / R / F1 (oracle) | Naive verdict = C0 | Naive silent |
|---|---|---|---|---|---|---|---|---|
| C0h dark | 0 | 0/8 | 1.00 / 1.00 / 1.00 | 8/8 | 0 | 1.00 / 1.00 / 1.00 | 8/8 | 0 |
| C1h office | 504 | 0/8 | 1.00 / 1.00 / 1.00 | 8/8 | 0 | 1.00 / 1.00 / 1.00 | 8/8 | 0 |
| C2h warm bulb | 1,453 | 0/8 | 1.00 / 1.00 / 1.00 | 8/8 | 0 | 1.00 / 1.00 / 1.00 | 8/8 | 0 |
| C3h window | 8,866 | 0/8 | 1.00 / 1.00 / 1.00 | 8/8 | 0 | 1.00 / 1.00 / 1.00 | 8/8 | 0 |
| C4h side lamp + shadow | 4,152 | 0/8 | 1.00 / 1.00 / 1.00 | 8/8 | 0 | 1.00 / 1.00 / 1.00 | 8/8 | 0 |
| C5h lamp reflection | 901 | 0/8 | 1.00 / 1.00 / 1.00 | 8/8 | 0 | 1.00 / 1.00 / 1.00 | 8/8 | 0 |
| C6h sun | 68,202 | 0/8 | 1.00 / 1.00 / 1.00 | 8/8 | 0 | 1.00 / 1.00 / 1.00 | 8/8 | 0 |

Interval note: 30/30 detections gives a Wilson 95 % interval of [0.89, 1.00]. The variants of one plate are not independent.

Images against the dark room. Each cell is the mean over captures, with the worst single capture in brackets.

| Row | Burst SSIM | Burst \|Δ\| (DN) | Burst low-pass \|Δ\| (DN) | Burst truth error p1..p99 (DN) | Naive SSIM | Naive \|Δ\| (DN) | Naive low-pass \|Δ\| (DN) |
|---|---|---|---|---|---|---|---|
| C0h | 1 | reference | reference | −2.1 .. +0.8 | 1 | reference | reference |
| C1h | 0.955 | 2.42 (3.88) | 0.13 (0.19) | −2.1 .. +1.0 | 1.000 | 0.01 (0.03) | 0.00 (0.01) |
| C2h | 0.955 | 2.42 (3.89) | 0.14 (0.19) | −2.3 .. +1.1 | 1.000 | 0.05 (0.13) | 0.01 (0.02) |
| C3h | 0.954 | 2.44 (3.94) | 0.17 (0.49) | −2.0 .. +0.9 | 0.997 | 0.26 (0.70) | 0.05 (0.13) |
| C4h | 0.954 | 2.44 (4.04) | 0.24 (1.06) | −2.2 .. +0.9 | 0.999 | 0.15 (0.36) | 0.03 (0.07) |
| C5h | 0.954 | 2.43 (3.98) | 0.22 (0.94) | −2.2 .. +0.9 | 1.000 | 0.01 (0.04) | 0.00 (0.01) |
| C6h | 0.942 | 2.62 (4.37) | 0.26 (0.71) | −2.6 .. +0.7 | 0.934 | 1.78 (5.54) | 0.37 (1.19) |
| Floor: two dark-room captures | | 2.53 | 0.30 | | | 1.38 | 0.18 |

- **Truth-error range.** The p1..p99 range is the widest over the row's captures, as a relative error −6 % .. +3 %. The largest local truth error, 13–14 DN in every row including the dark room, sits at sharp features: it is the device's own interpolation and inpainting, not lighting.
- **Gate metrics with the hood.** r_ambient max 0.021 (sun), masked_frac max 0.0019, motion max 0.61 px, LED level 0.0074–0.12 of full scale, ghost_bins 0–1 (`data/twin_pilot/gate_stats.json`).
- **Sun under the sealed hood.** The naive arm moves 1.78 DN in sun, so a little light reaches the plate even then. The burst subtracts it: low-pass 0.26 DN, against the 0.30 DN floor.

## 5. Results without the hood (open handheld)

Criterion: 0 silent failures. Retakes are allowed and reported.
- **Result.** PASS for both arms: no wrong verdict was ever let through.
- **Refusals.** The burst arm refused 39 of 56 captures. The naive arm refuses nothing by design.

| Row | Median lux | Burst retakes (gate reasons, a capture can have several) | Burst P / R / F1 (accepted) | Burst verdict = C0 | Burst silent | Naive P / R / F1 | Naive silent |
|---|---|---|---|---|---|---|---|
| C0o dark | 0 | 0/8 | 1.00 / 0.93 / 0.97 | 8/8 | 0 | 1.00 / 1.00 / 1.00 | 0 |
| C1o office | 504 | 5/8 (changed 4, glare_ambient 4) | 1.00 / 0.92 / 0.96 | 3/3 | 0 | 1.00 / 1.00 / 1.00 | 0 |
| C2o warm bulb | 1,453 | 4/8 (changed 4, glare_ambient 2) | 1.00 / 0.89 / 0.94 | 4/4 | 0 | 1.00 / 1.00 / 1.00 | 0 |
| C3o window | 8,866 | 8/8 (changed 8, too_bright 4, glare_ambient 3, motion 1) | n/a | n/a | 0 | 1.00 / 1.00 / 1.00 | 0 |
| C4o side lamp + shadow | 4,152 | 8/8 (changed 8, motion 2, too_bright 2) | n/a | n/a | 0 | 1.00 / 1.00 / 1.00 | 0 |
| C5o lamp reflection | 901 | 6/8 (changed 6, too_bright 5, glare_ambient 3, motion 1) | 1.00 / 1.00 / 1.00 | 2/2 | 0 | 1.00 / 1.00 / 1.00 | 0 |
| C6o sun | 68,202 | 8/8 (too_bright 8, changed 4, too_dark 3) | n/a | n/a | 0 | 1.00 / 1.00 / 1.00 | 0 |

| Row | Burst SSIM | Burst \|Δ\| (DN) | Burst low-pass \|Δ\| (DN) | Burst truth error p1..p99 (DN) | Naive SSIM | Naive \|Δ\| (DN) | Naive low-pass \|Δ\| (DN) |
|---|---|---|---|---|---|---|---|
| C0o | 1 | reference | reference | −2.3 .. +2.8 | 1 | reference | reference |
| C1o | 0.944 | 2.38 (2.93) | 0.36 (0.70) | −2.0 .. +3.5 | 0.882 | 17.9 (35.2) | 17.7 (35.0) |
| C2o | 0.838 | 4.14 (6.08) | 0.93 (1.84) | −2.9 .. +3.5 | 0.932 | 11.3 (28.5) | 11.0 (27.8) |
| C3o | refused | | | | 0.846 | 25.6 (46.5) | 25.3 (46.4) |
| C4o | refused | | | | 0.900 | 17.5 (41.4) | 17.2 (41.0) |
| C5o | 0.938 | 2.63 (3.03) | 0.49 (0.78) | −2.5 .. +0.5 | 0.656 | 52.8 (77.9) | 52.8 (78.2) |
| C6o | refused | | | | 0.672 | 58.0 (94.0) | 57.7 (93.9) |
| Floor: two dark-room captures | | 2.25 | 0.26 | | | 1.17 | 0.12 |

How to read this:
- **Old method, open.** The naive images change by 11–58 DN, and the low-pass |Δ| is almost the whole |Δ|, so this is lighting, not noise. With the oracle the naive arm still scores F1 1.00, because the oracle reports whatever stays visible. Whether a trained model would cope with those images is exactly what the trained-detector evaluation must show (section 9).
- **New method, open.** Accepted burst captures stay near the floor: low-pass 0.36–0.93 DN, against naive's 11–18 DN in the same rows.
  - **Small residues remain.** The worst accepted capture reaches +3.5 DN at p99 against +2.8 DN in the open dark room, and +43 % relative error at p99 over small areas, against +7 %. These are local ambient residues below the gate's limits (`ghost_contrast` allows changes of up to about 10 % of the LED level). They are the main remaining limit of open mode.
  - **Open-mode recall below 1.** Under the pods' min combination, the rust patch on s900011 is invisible to the burst arm even in the dark room (contrast −6 DN, against +14 for naive). That is a property of the open illumination, not of the lighting.
- **Refusal reasons the operator sees.** "Too bright: shade the part or fit the hood", "Lamp reflection on the part: tilt, shade it or fit the hood", and "Lighting changed during capture". The dark room is never refused.

## 6. What the pilot changed, with the evidence

Each change is documented next to its setting in `configs/capture.yaml` and tested in `tests/test_lightproc.py`.

| Gate version | Hood retakes (all 7 conditions) | Open retakes C0 / C1 / C2 / C3 / C4 / C5 / C6 (of 8) |
|---|---|---|
| Round 0: the 14 Nov redesign as delivered | C3h 1/8 ('changed', ghost 3), others 0 | 2 / 6 / 2 / 8 / 8 / 8 / 8 |
| + closure refinement (14 Nov) | 0/56 | 0 / 3 / 3 / 8 / 7 / 6 / 8 |
| + the 15 Nov fixes (final) | 0/56 | 0 / 5 / 4 / 8 / 8 / 6 / 8 |

1. **Closure refinement of the L, A, R shifts** (14 Nov, `process.align_refine`).
   - **Why it was needed.** Braced shifts assume the device moves linearly from B1 to B2, but a hand's shake is a random walk. L was 0.16 px off on a still brushed plate under the hood, which is enough for 'changed' ghosts on a still capture. Open captures of hot-rolled plate were refused in the dark room (ghost_bins 38–50).
   - **What it does.** After the coarse alignment, the L, A and R shifts minimise B1 + A − L − R, which is zero for a still part. The L/R errors fall to 0.01–0.11 px.
   - **Review on 15 Nov.** The comments wrote the closure as "B1 = L + R + A"; it is B1 + A = L + R, and they are corrected. The 15 Nov session also made the code 4× faster: no per-iteration gathers, float32 BLAS. Its results are unchanged (largest shift difference 0.00005 px on 6 bursts), and `process_burst` at 1600² takes 0.44 s on an idle machine.
2. **Dark-field defects are no longer erased** (`process.inpaint_px`, ring).
   - **The defect.** Under the ring, a pixel where only L or R clipped was masked and inpainted. Those are exactly the brightest scattering defects. Contrast on s900028's pit cluster fell from 101 (reference) to 3–15 DN, and on s900011's second scratch from 191 to 16 DN. In 5 hood captures the pit dropped below the visibility threshold.
   - **The fix.** Such a pixel's mean is a lower bound. When that bound is at or above display white in every channel, image_u8 is exact there, so it is kept white. The gate's masked_frac is unchanged, so not a single gate decision changed: all 112 burst captures matched.
   - **Result.** The pit cluster reads 89–97 DN in all 14 hood captures, and burst hood recall went from 0.97 to 1.00.
3. **Ghost test on either closure for the pods** (`process.ghost_closure: {pods: max, ring: min}`).
   - **The blind spot.** With per-frame ambient offsets, the closures are b₁ − e_L − e_R and b₂ − e_L − e_R. A step in the ambient after the L frame, which makes M wrong, and a blip in B2 only, which is harmless, both read (0, a). The old min test passed both.
   - **The case found.** s900028 C4o v0 was accepted with a moving hand shadow, M 56 % dark locally (−14 DN). It is now refused with 142 ghost bins.
   - **Why the ring keeps min.** max on the ring refused still hood captures: 2 of 8 in every condition including the dark room, from B1/B2 alignment residue at sharp dark-field features. (Second review: the ring now uses `max_ambient`, section 11.3.)
4. **Coarse closure test** (`process.ghost_coarse_*`, `gate.ghost_coarse_max: 0`).
   - **What it does.** It applies the same test to 32 px blocks, which sees a change that is coherent over a region but below the noise of each 4 × 4 bin.
   - **The case found.** Flicker of a ceiling tube mirrored in cold-rolled steel left M up to +32 % over the reflection (s900011 C1o v1, 9 blocks). It is now refused.
   - **Two dark-room false alarms removed.** The coarse test needs both closures (min) and A ≥ the LED level in the block. A first version flagged 3–12 blocks on brushed stainless in the dark room (near-black pixels at about 7 electrons). Those blocks had no ambient, so they could not be a lighting ghost (`data/twin_pilot/gate_calibration_interim.json`).
5. **Ambient-dominated plate** (`process.local_r_*`, `gate.ambient_dominated_max: 0.02`).
   - **The mechanism.** Where a mirrored lamp is more than 8× the LED level, the ambient's shot noise swamps L − A and R − A. Clamping at zero then biases M bright. Nothing changes between frames, so no closure test can see it.
   - **The case found.** s900028 C1o v0 (median r 0.31, flagged and kept) had local r above 8 over 9 % of the plate (about 22 where it was worst) and M +236 % (+21 DN). It is now refused as glare_ambient.
   - **Separation.** Measured from the render passes, every accepted open capture with a correct image had 0–0.3 % of the plate above r 8. Every hood and leak capture had 0.
6. **Overview renders** (`twin/blender/render_scene.py add_pod_glow`). The overview-only "glowing pod" slab spanned the ring LEDs' bounding box and drew the hood as a lit white box. Ring LEDs now get no slab, and the 7 hood overviews were re-rendered (`tools/pilot/overview_only.py`).
7. **Evaluation** (14 Nov, `training/eval_lighting.py`): the oracle mode and the |Δ| (`mad_c0`) metric, reviewed on 15 Nov.

Nothing was loosened. Every 15 Nov gate change refuses more, and each was checked against the dark-room and hood captures for false refusals. The open captures newly refused all carried an image error against the truth: −56 %, +236 %, +32 % and +126 % relative error.

## 7. Hood light-leak stress test (rim lifted 3 mm)

The hood rows of the pilot were re-rendered with the hood held 3 mm above the part, so ambient light gets in under the rim. This models a warped part, a weld bead, or the operator lifting the hood. It used C3, C4, C5 and C6 (16 rows, 32 captures), with the ring LEDs moved with the hood (`tools/pilot/make_leak_plans.py`). Outputs are in `data/twin_pilot/leak3/`.

| Row | Burst retakes | Oracle F1 / verdict = C0 / silent | Burst \|Δ\| vs C0h (DN) | Burst low-pass \|Δ\| | Burst truth error p1..p99 (DN) | Naive \|Δ\| (DN) | Naive low-pass \|Δ\| |
|---|---|---|---|---|---|---|---|
| C3h window, lifted | 0/8 | 1.00 / 8/8 / 0 | 2.46 (3.97) | 0.14 (0.24) | −1.9 .. +0.8 | 0.54 (1.22) | 0.10 (0.22) |
| C4h side lamp, lifted | 0/8 | 1.00 / 8/8 / 0 | 2.44 (3.93) | 0.13 (0.22) | −2.1 .. +1.0 | 0.31 (0.66) | 0.06 (0.12) |
| C5h lamp reflection, lifted | 0/8 | 1.00 / 8/8 / 0 | 2.42 (3.89) | 0.16 (0.30) | −2.1 .. +0.9 | 0.12 (0.19) | 0.02 (0.04) |
| C6h sun, lifted | 0/8 | 1.00 / 8/8 / 0 | 2.73 (4.71) | 0.35 (1.41) | −3.1 .. +0.9 | 3.22 (7.53) | 0.66 (1.76) |

**Verdict: a 3 mm leak is processed correctly, not silently corrupted.** The gap lets little light in.
- **Window, lamp and shadow.** r_ambient stays at most 0.0035. Under the window, the side lamp with its moving shadow, and the reflected lamp, nothing changes measurably.
- **Direct sun.** On brushed stainless r reaches 0.05–0.07. The extra ambient shot noise raises that plate's |Δ| to 4.5–4.7 DN (SSIM 0.83–0.90). Its truth error stays within −3.1 .. +0.9 DN, against −2.6 .. +1.1 for the sealed hood.
- **Naive arm.** It drifts 3.2 DN in sun with the lifted hood, against 1.8 DN sealed.

**Limit of this test.** A 3 mm gap is too small to exercise the gate. Nothing came close to a limit, so this does not show that a badly leaking hood would be refused. The ring keeps the min closure (section 6, item 3), so an ambient step after the L frame under a leaking hood would not be seen by the fine ghost test. A toy case of exactly that passes with M at 53 % of the truth (`tests/test_lightproc.py::test_ghost_closure_max_refuses_an_ambient_step_after_l`). A larger lift (10 mm, or one side open) is the next stress test. (Second review, section 11: the ring rule is now `max_ambient`, which refuses that step, and a ring capture above r 0.15 is refused as a hood leak.)

## 8. Figures and images

**Montages.** `reports/figures/pilot_montage_s900011.png`, `pilot_montage_s900027.png`, `pilot_montage_s900028.png` and `pilot_montage_s900068.png`, one per plate. Columns are the 14 matrix rows; rows are the two arms; ground truth is green and oracle detections red. They come from eval_lighting.

**Naive vs burst.** `reports/figures/pilot_naive_vs_burst_<plate>.png`, one per plate. The rows are open/old, open/new, hood/old and hood/new; the columns are C0–C6. Each cell has the gate verdict and the verdict against the truth, and refused captures are dimmed with their reason.

**Gate fixes.** `reports/figures/pilot_gate_fixes.png`:
- top: the pit cluster with the hood, before and after the ring fix;
- bottom: the two formerly accepted open aluminium captures, each beside its error map against the noise-free LED image.

**Overview strip.** `reports/figures/pilot_overview_s900011.jpg`, with plate 1 seen from outside in all 14 rows.

**Overview images (plate 1, one per condition and device).**
- `data/twin_pilot/overview/s900011_C0o.png` … `s900011_C6o.png`: open handheld.
- `data/twin_pilot/overview/s900011_C0h.png` … `s900011_C6h.png`: hood, re-rendered after the glow fix.

**Pi demo bursts.** `data/twin_pilot/bursts/s900011_C{0..6}{h,o}_v0.npz`: 14 files, 345 MB. Each holds five 1641 × 1633 RGB frames plus labels and the plate geometry, for `trinetra-app --source twin:data/twin_pilot/bursts`.

**Data.**
- `data/twin_pilot/`: grids, sidecars and gate_stats.json.
- `invariance.json`: the noise-separated numbers.
- `truth_error.json`: the per-capture truth error.
- `eval_oracle_box/` and `eval_oracle_app/`.
- `leak3/`.

## 9. Honest limits, and what is not done yet

- **No trained detector yet.** Every F1 and verdict above is the oracle's, so the image-level numbers in sections 3–5 are the evidence for lighting invariance. Next step: train the two arms with `training/train_arms.py` (YOLOv8n per arm), then run `training/eval_lighting.py --model naive=... --model burst=...` on a fresh matrix. Until then:
  - the naive arm's open-mode PASS says nothing about detection;
  - nothing here measures false positives on clean plates, because there was no clean plate in the pilot.
- **The gate rules were tuned on these four plates.** The closure, coarse closure and local-ratio rules (15 Nov) were tuned on the pilot seeds 900011, 900027, 900028 and 900068, so this pilot is not an independent test of them. The real lighting matrix must use new seeds, `--start-seed 900100` or later (training/README.md, DESIGN_STEEL C2).
- **Small sample.**
  - Only 4 plates, with 2 captures per scene.
  - The two captures of a plate share its render; only the ambient, shake and noise are drawn again.
  - Galvanised steel is not covered.
  - Three of four plates are REJECT, so verdict agreement is an easy test.
- **Open mode still has small local residues.** Accepted open captures carry ambient residue over small areas, up to +3.5 DN at p99 and +43 % relative, below the gate's per-bin limits. The open handheld is not the product, and the hood removes this.
- **Burst noise is about 1.8× naive's** (σ 2.29 against 1.26 DN). This comes from the fixed white-card exposure under dark-field. A plate-based ring exposure should help; to be checked on the Pi.
- **Twin, not hardware.** The sensor noise model, the ring geometry (18 mm above the plate face, 8–11° elevation) and the exposure (analogue gain 1.0, white card at 57 % of full scale) all have to be verified on the Pi on day 1. `lux_est` is still not meaningful on specular metal.
- **Tests.**
  - The suite is green: 735 passed, 5 skipped, with `tests/test_twin_render.py` left out.
  - The Blender render tests were not run; the pilot exercised the real Blender path.
  - The 1600² timing test passed at 0.44 s, but failed at about 2 s while other jobs loaded the machine.

## 10. Reproduce

Run everything from the repo root. `<keep>` is a folder for the rendered EXRs, 3.5 GB for pilot plus leak. The EXRs used here are in this session's scratch folder and will not persist, so a fresh run must render again (about 40 min for the pilot).

```sh
# 0. plates: one seed per material, >= 900010, 100 x 100 mm, >= 3 labelled defects
.venv/bin/python tools/pilot/scan_seeds.py
.venv/bin/python tools/pilot/make_plans.py data/twin_pilot 900011 900027 900028 900068

# 1. render + post, full resolution, 64 spp, EXRs kept in <keep>; bursts and overviews for plate 1
.venv/bin/python tools/pilot/run_pilot.py data/twin_pilot <keep> s900011 --bursts --overview
.venv/bin/python tools/pilot/run_pilot.py data/twin_pilot <keep> s900027 s900028 s900068

# 1b. after a lightproc / capture.yaml change: delete grids/ sidecar/ debug/ bursts/ yolo/, then re-post
.venv/bin/python tools/pilot/repost.py data/twin_pilot <keep> --bursts s900011

# 2. evaluation (oracle; box and app measures)
.venv/bin/python training/eval_lighting.py --matrix data/twin_pilot --model oracle --figures 4 --out data/twin_pilot/eval_oracle_box
.venv/bin/python training/eval_lighting.py --matrix data/twin_pilot --model oracle --measure app --figures 0 --out data/twin_pilot/eval_oracle_app

# 3. image-level numbers and tables
.venv/bin/python tools/pilot/gate_stats.py data/twin_pilot
.venv/bin/python tools/pilot/invariance.py data/twin_pilot data/twin_pilot/eval_oracle_box/lighting_matrix.json data/twin_pilot/invariance.json
.venv/bin/python tools/pilot/truth_err.py data/twin_pilot <keep> data/twin_pilot/truth_error.json
.venv/bin/python tools/pilot/tables.py data/twin_pilot data/twin_pilot/eval_oracle_box/lighting_matrix.json \
    data/twin_pilot/eval_oracle_app/lighting_matrix.json data/twin_pilot/invariance.json data/twin_pilot/truth_error.json
.venv/bin/python tools/pilot/calibrate_gate.py data/twin_pilot <keep> <out.json>     # gate change statistics
.venv/bin/python tools/pilot/local_r.py data/twin_pilot <keep>                         # local ambient ratio (open rows)

# 4. figures
.venv/bin/python tools/pilot/figures.py data/twin_pilot data/twin_pilot/eval_oracle_box/lighting_matrix.json reports/figures \
    s900011 s900027 s900028 s900068
for p in s900011 s900027 s900028 s900068; do
  cp data/twin_pilot/eval_oracle_box/figures/lighting_$p.png reports/figures/pilot_montage_$p.png; done
# (pilot_gate_fixes.png: tools/pilot/fig_fixes.py, which also needs the pre-fix pit crop)

# 5. hood leak test, rim lifted 3 mm (the C0h plans are copied unchanged and render as in the pilot)
.venv/bin/python tools/pilot/make_leak_plans.py data/twin_pilot <leak> 3 C3_window C4_side_lamp C5_reflected_lamp C6_sun
.venv/bin/python tools/pilot/run_pilot.py <leak> <leak keep> s9000
.venv/bin/python training/eval_lighting.py --matrix <leak> --model oracle --figures 0 --out <leak>/eval_oracle_box
.venv/bin/python tools/pilot/invariance.py <leak> <leak>/eval_oracle_box/lighting_matrix.json <leak>/invariance.json
.venv/bin/python tools/pilot/truth_err.py <leak> <leak keep> <leak>/truth_error.json

# tests
.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_twin_render.py
```

## 11. Second independent review (15 Nov)

An adversarial review of the redesign and of this pilot, done separately from the 15 Nov review whose changes are already in sections 6–10. Earlier that day, that review changed the ring's analogue gain to 1.5, added the LED-droop calibration and fixed the sensor model's negative lobes. My scratch work is in the session scratch folder (`redesign/review_b/`), which will not persist. Its EXRs were deleted after the analysis; re-render them with `tools/pilot/run_pilot.py` on the plans that `tools/pilot/review_stress_plans.py` writes. The harness scripts are kept in `tools/pilot/review_adversarial.py` and `tools/pilot/review_stress_plans.py`; the adversarial harness runs on kept EXRs, for example the pilot's.

### 11.1 What I re-ran, and whether it reproduces

**Fresh render of one pilot plate.** I re-rendered s900028 (aluminium) from its pilot plans: full resolution, 64 spp, all 7 hood rows plus C0o, C1o and C4o. That is 10 scenes and 20 captures.
- **Render.** The gate verdicts match the pilot EXRs re-posted with today's code in 20 of 20 captures. The metrics are identical except in two captures, where Cycles noise differs slightly (C3h v1 motion 0.366 against 0.387 px; C4o v1 ghost_bins 1539 against 1511).
- **Pilot sidecars.** The verdicts also match.
- **Oracle evaluation.** Both arms PASS. With the hood: F1 1.00, and 2 of 2 verdicts equal C0h in every condition. Open: C1o refused 2 of 2 (glare_ambient), C4o refused 2 of 2 (changed).
- **Error against the LED-only truth.** With the hood: −0.59 .. +0.67 DN (p1..p99) in every condition, at most 2.4 DN anywhere.

**The hood noise numbers in sections 3–4 are out of date.** With today's ring gain 1.5 (the honest photon count), the hood |Δ| against C0h for this plate is 2.84–2.93 DN (SSIM 0.929–0.932). Sections 3–4 give 2.25–2.29 DN (0.954). This is pixel noise, about 1.27× more: the low-pass |Δ| and the verdicts do not change.

**A new plate that nothing was tuned on.** I rendered s990123, a galvanised plate (a material the pilot did not cover) with 6 labelled defects, at full resolution: 7 hood rows plus C0o, C1o and C5o. Results with the final gate:
- **Hood.** Accepted 14 of 14. Oracle F1 1.00, verdict = C0h 2 of 2 in every condition. Error against the truth −0.58 .. +0.54 DN (p1..p99), at most 1.3 DN. |Δ| against C0h 2.45–2.47 DN, which is noise (SSIM 0.958).
- **Open.**
  - C0o: accepted.
  - C1o v0: refused by the new flicker rule. Its measured error was real: −6.4 DN at p1, −28 %.
  - C1o v1: accepted, with +4.5 DN at p99 (+22 %), the small open-mode residue of section 9.
  - C5o: refused 2 of 2.
- **No silent failure** in either arm.

**Every pilot capture re-processed.** All 112 pilot captures and the 40 leak-test captures went through the final gate. Section 11.3 has the results.

### 11.2 Trying to break it

**Method.** I built adversarial bursts from the kept full-resolution renders, each with a clean twin: the same shake, LED gains and noise draws, with no perturbation.
- **Score.** I compared the detector image of the two (`image_u8`, each with its own normalisation, 1.1 mm low-pass) and checked it against the noise-free LED-only image.
- **Classes.**
  - SILENT: accepted, and more than 8 DN wrong over at least 1 mm², which is enough to create or hide a defect at post's 4 DN visibility threshold.
  - MARGINAL: accepted, 4–8 DN wrong.
  - Otherwise OK, or REFUSED.
- **27 cases per scene:**
  - ambient flicker of 5 % and 10 % (5 random draws);
  - an A-only change of +5 % or −10 %, and an L-only change of +10 %;
  - an ambient step after L or after A;
  - the A frame alone moved by 2–4 px;
  - LED droop of 2 % or 5 % with both channels on;
  - a mirrored tube in the ambient pass (40 × 4 mm at 3× the LED level), static, moving or flickering 10 %, plus a small one at 20× the LED level;
  - the whole ambient pass moving 1–3 px per frame;
  - a hand shadow arriving at the A or R frame;
  - shake of 3 or 6 px per frame.

| Scene | Before the fixes in 11.3 | After |
|---|---|---|
| Hood, sealed, s900068 C6h (stainless, sun, r 0.02) | 1 SILENT (flickering tube: −33 DN over 253 mm²), 3 MARGINAL | 0 SILENT; 3 MARGINAL ≤ 5.2 DN (static tube, hand shadows); 10 refused |
| Hood, sealed, s900027 C4h (hot-rolled, side lamp) | 1 MARGINAL (static tube) | the same; 0 SILENT |
| Leaking hood: 3 mm-lift sun render, leak light ×2 (r 0.09) | 1 SILENT (hand shadow at A: +9 DN over 70 mm²), 2 MARGINAL (of the 9 cases run before the fixes) | 0 SILENT, 3 MARGINAL ≤ 4.8 DN, 18 refused |
| Leaking hood, ×3 (r 0.14) | 5 MARGINAL (4–6 DN over up to 560 mm²; of 9 cases) | 0 SILENT, 1 MARGINAL (4.7 DN), 24 refused |
| Leaking hood, ×5 (r 0.22) | 2 SILENT (10 % flicker: +8 DN over 9 mm², over 4 DN on 15 % of the plate; flickering tube), 3 MARGINAL | 27 of 27 refused, shown as "Light is getting under the hood" |
| Open, s900011 C2o (cold-rolled, warm bulb, r 0.11) | 2 SILENT: 5 % flicker (+15 DN over 774 mm², 8 % of the plate) and the static tube | 1 SILENT: the static tube (−9.6 DN over 17 mm²) |
| Open, s900028 C2o (aluminium, warm bulb, r 0.10) | 1 SILENT (static tube), 3 MARGINAL (flicker, A +5 %: −7 DN over 25 % of the plate) | 1 SILENT: the static tube (−10.7 DN over 65 mm²) |

- **Always handled, before and after the fixes.** Every A-only misalignment of 2–4 px, every LED droop and every moving reflection or moving ambient pass was either processed correctly or refused as `changed`. Shake of 6 px was refused as `motion`.

**Physical stress renders.** These are new Blender renders at full resolution, 64 spp (`tools/pilot/review_stress_plans.py`).

| Case | Result |
|---|---|
| Mirror-polished stainless (s900068, roughness 0.02, no anisotropy); hood C0, C3, C6 and open C0, C1 | Hood: refused 6 of 6 as too_dark, with the message that names the mirror finish. The LED level was 0.0029–0.0030 against the limit of 0.003, so this is a hair-trigger, but the images were correct anyway (−2.4 .. +3.5 DN against the truth). Open: C0 refused as too_dark ("move to 200 mm", which is misleading for a mirror); C1 refused as too_bright (r 3.6–4.0). |
| Very dark mill scale (s900027, base colour ×0.35); hood C0, C3, C6 and open C0, C2 | Hood: accepted 6 of 6, within ±0.65 DN of the truth. Open: C0 accepted (−0.7 .. +1.2 DN); C2 refused 2 of 2 as changed. |
| Hood straddling the plate edge (s900028; the rim crosses the plate 15 mm in from its edge, with a 5.9 mm slit over the bench); C0, C2, C3, C6 | Refused 8 of 8 as saturated: the plate under the ring LEDs clips over 8 % of it. The leak stayed below r 0.0007, even in sun. Before the fix the message was "tilt the device about 10 degrees", which cannot be done with the hood on (fixed in 11.3). |

### 11.3 What I fixed

Each fix is documented in `configs/capture.yaml` and has a regression test in `tests/test_lightproc.py`.

1. **Ring ghost rule** (`process.ghost_closure.ring: max_ambient`).
   - **The defect.** With `min`, an ambient step after the L frame passed under a leaking hood: the hand shadow at the A frame in the table above.
   - **The fix.** A 4 × 4 bin now counts when both closures flag it, or when one closure flags it and the ambient frame there could have made the deviation (A ≥ the deviation).
   - **Why not `max`.** The pilot's dark-room residue is LED-made, with A = 0, so it still needs both closures. `max` refused those dark-room hood captures in the pilot (2 of 8 per condition).
   - Test: `test_ring_ghost_rule_sees_an_ambient_step_not_dark_room_residue`.
2. **Plate-wide flicker** (`lightproc._ambient_flicker`, `gate.flicker_err_max` 0.2).
   - **How it measures.** The two closures and B2 − B1 are regressed on [A, B1 − A] over 8 × 8 bins, with the regressors taken from the neighbouring bins so that A's own noise does not bias the fit. That gives the ambient change between frames to about 0.05 %, even when no bin or block shows it.
   - **The rule.** That change times the 99th percentile of A/M is the relative error it makes in M. The limit, 20 % of M, is the error that `ghost_contrast` already refuses locally.
   - **No effect without ambient.** The test has nothing to measure in the dark room or under a seated hood: the metric is None.
   - Test: `test_plate_wide_flicker_is_refused_where_it_matters`.
3. **Hood leak** (`gate.leak_r_max: {ring: 0.15}`, new reason `hood_leak`: "Light is getting under the hood: seat it flat on the part (or shade it), then retake.").
   - **Evidence.** The seated hood reads r ≤ 0.021 in the pilot, and ≤ 0.071 with the rim lifted 3 mm. A sealed hood on the mirror plate in sun read 0.074–0.108, because its dark-field LED level is tiny; that capture is refused as too_dark anyway. With the leak light ×3–×5 (r 0.14–0.29), flicker and ambient steps passed with 4–8 DN errors.
   - Test: `test_hood_leak_and_ring_messages`.
4. **Ring messages.** Under the ring, `saturated` now says to centre the part in the hood, away from the LEDs. `glare_ambient` now names the leak. `training/eval_lighting.py` ranks `hood_leak` first, like `gate.messages`.

**Re-check on every pilot capture with the final gate.**
- **Hood.** 56 of 56 pilot captures and 40 of 40 leak captures are still accepted.
- **Open.** One verdict changed: s900011 C1o v0 went from PASS to RETAKE ('changed').
  - Its predicted error was 0.35, and its measured error against the truth was +34 % at p99 (absolute max 30 DN).
- **The flicker estimate tracks the measured truth error.**

  | Accepted open capture | Predicted error | Measured truth error (p99, relative) |
  |---|---|---|
  | s900011 C2o, v0 / v1 | 0.16 / 0.19 | 0.32 / 0.24 |
  | s900027 C1o v0 (4.3 % flicker, hot-rolled) | 0.03 | 0.007 |

  So harmless flicker is not refused.
- **Tests.** `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_twin_render.py`: 766 passed, 10 skipped. In `.venv-train`, test_lightproc, test_eval, test_matrix and test_burst give 137 passed.

### 11.4 Thresholds and the pass criterion

**Loosened in the redesign.** Each was checked against the evidence in its `capture.yaml` comment.
- **`masked_frac_max` 0.005 → 0.02.** The largest masked share measured was 0.0066 (ring) and 0.0074 (pods), so 0.02 is a 3× margin. Masked pixels are inpainted, so a defect inside them is lost. The oracle's visibility test sees that loss.
- **`motion_max_px` 3 → 8.** Supported by the sweep up to 12 px, but only within the twin's shake model: translation only, and blur within 10 ms.
- **Ring `led_level_min` 0.003.** Its comment ("= 29 e-") assumed gain 1.0. At the ring's gain 1.5 it is 19 e-, and the mirror plate sits right at the limit.

**Tightened.** Every other gate change in the redesign, the pilot and this review is stricter.

**Pass criterion.** `training/eval_lighting.py` matches DESIGN_STEEL G:
- with the hood, in every condition C0–C6: retakes ≤ 5 %, F1 ≥ F1(C0h) − 0.05, verdict = C0h in ≥ 95 % of captures, and 0 silent failures;
- open: 0 silent failures;
- C0 open is required.

**A caveat on "silent failure".** The ground-truth verdict counts only defects visible in the same device's C0 capture (`--gt-visibility c0`). So "no silent failure" means the lighting never changed the verdict. It does not mean no defect was missed: for example, the rust that the pods cannot see in the dark room is not counted.

### 11.5 Ring physics and the fairness of the comparison

**The ring LEDs and the exposure reference were checked by hand.**
- **Position and aim.** The 18 LEDs sit on the inner walls, 6 mm in, and aim at the optical-axis plate point, at 8.2–11.2° elevation.
- **Irradiance.** Analytic: 3.2–3.4 klux at the centre and 8.5–10.2 klux at the corners. A hand calculation for 18 × 11.7 cd at about 95 mm and 10° gives about 4 klux.
- **White-card reference.** It now matches a single Pi exposure through the ring gain of 1.5.

**Ring height: the twin differs from the build spec.**
- **The twin** puts the ring 18 mm above the plate face.
- **A ring glued 18 mm above the rim** of a hood resting on the bench sits at 18 − t above the plate face. For a 5.7 mm plate that gives 5.6–7.7° elevation and about −22 % irradiance at the centre.

**The ring LEDs are inside the camera's view.** All 18 LEDs project onto the sensor, outside the plate. The camera is 63–76° off each LED's axis, inside its 150° cone.
- **The twin has no lens flare.**
- **The real hood needs** a lip or baffle over each LED on the camera side.

**The naive-arm comparison is fair about light.** Both arms use the same render and the same illumination, which is the ring under the hood. It is unfair in two ways, both in naive's favour:
- **Auto-exposure means more light.** Naive's auto-exposure amounts to a 1.5–25× longer exposure on dark-field plates, so it collects more light.
- **Naive reuses its noise draw.** Without shake, its draw is the same in every condition. So its hood |Δ| of 0.01–0.26 DN compares an image with itself. Compare only the low-pass |Δ|.

The oracle cannot penalise naive; only a trained detector can.

### 11.6 What is still open

- **Open: a static mirrored lamp is still a silent failure.**
  - **The case.** A static tube at 2–8× the local LED level, below `local_r_max`, over less than 2 % of the plate.
  - **The cause.** Ambient shot noise turns into bias through the pods' minimum, the clamp and the gamma (the error vanishes without noise). M ends up −10 DN over 17–65 mm².
  - **No rule yet.** A per-pixel ambient-noise rule would flag 50–85 % of ordinary warm-bulb captures, so it cannot single this case out.
  - **The consequence.** Open mode is not shown to "never fail silently" under mirrored lamps. The hood removes this case.
- **Some ambient changes cannot be seen at all.** Only b_i − e_L − e_R can be observed. An ambient change with e_L = −e_R and steady brackets passes with M wrong; a test pins this down.
- **The closure refinement can fit a flicker residual with a shift.** It keeps a correction that lowers the closure by only 2 %. On a still burst with structure in A, it moved A by 0.17 px.
  - With the hood that needs a leak.
  - A minimum required gain would stop it, but was not validated here.
- **B2 alignment.** With `align_lowpass` 0.05 the B2 alignment is 0.03–0.17 px off on still twin bursts; at 0.2 it is at most 0.04 px. Not changed here.
- **Unchanged from section 9:**
  - no trained detector;
  - gate rules tuned on pilot seeds (the real matrix starts at `--start-seed 900100`; this review tuned nothing on new seeds);
  - day-1 hardware checks.
