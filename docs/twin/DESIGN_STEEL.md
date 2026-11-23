# Ayas v2: steel and metal surfaces, lighting-invariant (design contract, 14 Nov 2026)

This file **supersedes** the aircraft parts of `docs/twin/DESIGN.md`: classes, rivets as labels, aero rules. These parts of DESIGN.md stay valid unless changed below: frames (section 1), the plan JSON skeleton (section 3), work-dir files (section 4), post rules (section 5), tiles and Blender facts (section 7.1).

Evidence for every choice is in:
- `docs/research/2026-11-14_lighting_invariance.md`
- `docs/research/2026-11-14_steel_datasets.md`
- `docs/research/2026-11-14_steel_severity_standards.md`
- `docs/research/2026-11-13_ps5_session_notes.md`

**Phase 1 goal (user, 14 Nov).** A Blender digital twin and a solid proof of concept, built by simulation, showing that the inspector **works in different lighting conditions**. The end-to-end pipeline runs on the Pi: vision → JSON → Llama via ExecuTorch → alert → log. Twin frames act as the camera; recorded input is allowed by the organisers.

**The lighting redesign.** Robustness comes from the capture method, not from the network:
1. A 5-frame LED burst in linear raw.
2. Ambient subtraction.
3. Combining the two LED halves: min(L, R) for the handheld's pods (removes their glare), mean(L, R) for the hood's dark-field ring.
4. A quality gate that refuses rather than fails silently.
5. **Device v2 (section C2, 14 Nov): the clip-on black hood head with a low-angle (dark-field) LED ring.** It is the product configuration and must work in every condition; the open handheld must never fail silently.
6. Augmentation and twin lighting randomisation as the last line.

**The old method is kept as the baseline arm:** a single both-pods frame through auto-exposure and an ISP. The experiment must show where it fails and where the new method holds.

## A. Classes and kinds

`ayas.tiles.CLASSES = ("scratch", "dent", "pit", "corrosion", "crack", "stain")`, with YOLO ids 0–5. `ELONGATED = ("scratch", "crack")`.

| Class | Kinds (params in plate mm) | `measure` keys |
|---|---|---|
| scratch | `scriber`, `gouge`, `abrasion` (as DESIGN 3.1) | length_mm, width_mm (abrasion: area_mm2 too) |
| dent | `punch`, `ball`, `smooth`, `nick` (as DESIGN 3.1) | diameter_mm, depth_mm |
| pit | `single` {u, v, d_mm 0.2–1.5, depth_mm 0.05–0.5}; `cluster` {u, v, r_mm 1.5–8, n 3–30, pit_d_mm [min, max] within 0.1–1.2, depth_mm} | single: diameter_mm, count = 1. cluster: diameter_mm (equivalent of the hull), count, area_mm2 |
| corrosion | `rust` {u, v, r_mm 2–25, coverage 0.3–1, pitted bool, aspect, angle_deg}, steel only; `rust_streak` {u0, v0, angle_deg, length_mm 5–40, width_mm 1–6}, from an edge, hole or rust patch, steel only; `white` {u, v, r_mm 1.5–10, powder}, aluminium/galvanised only (= old pitting/galvanic look) | area_mm2, diameter_mm (equivalent), length_mm (streak) |
| crack | `open` {pts_mm jagged polyline 3–40 mm, width_mm 0.05–0.4}; `slit` (as DESIGN 3.1); `fatigue` (tight, mostly invisible) | length_mm, width_mm |
| stain | `oil` {u, v, r_mm 2–15, aspect, angle_deg, gloss}; `water` {u, v, r_mm 2–12} (dried ring); `fingerprint` {u, v, r_mm 5–9, angle_deg} | area_mm2, diameter_mm |

- **Unlabelled distractors:** marker, dust, blend (a grind or blend mark), burr, label residue, and through holes or slots. Holes are part features, not defects; they are cut by Blender. Rivets are no longer used: sampler `fastener` patterns default to `none`, with 15 % plain punched-hole rows as distractor geometry.
- **Footprints.** `geometry.footprint` handles every new kind, with the same rules as before: the sampler computes the footprint, and textures must stay inside it.

## B. Materials (`plate.material.kind`)

| Kind | Look | Base colour (linear) | Roughness | Metallic | Notes |
|---|---|---|---|---|---|
| `ms_hot_rolled` | mill scale, blue-grey/black, mottled | 0.08–0.18 | 0.45–0.8 | 0.3–0.7 mottled | scale flakes; rust allowed |
| `ms_cold_rolled` | bright grey, faint roll marks | 0.45–0.6 | 0.15–0.35 | 1 | rust allowed |
| `ss_brushed` | stainless No.4 brushed | 0.55–0.65 | 0.1–0.3, anisotropic | 1 | no rust (light tea-staining allowed as stain) |
| `ss_2b` | stainless matte mill finish | 0.5–0.6 | 0.25–0.4 | 1 | |
| `galvanised` | zinc spangle crystals | 0.55–0.7 | 0.2–0.4 | 1 | white corrosion only |
| `aluminium` | as before | 0.88–0.92 | 0.12–0.4 | 1 | white corrosion only |

**Sampler weights:** ms_hot_rolled 0.2, ms_cold_rolled 0.25, ss_brushed 0.15, ss_2b 0.1, galvanised 0.1, aluminium 0.2.

`plate.size_mm` is mostly 100×100, plus some 150×100 and 100×60. The thickness is 1.5–6 mm and goes into the severity spec.

## C. Lighting conditions, units and the device

**Units.** Blender light power and strength are radiometric "W" with **683 lm/W**: `E_W = lux / 683` W/m² at the plate. EXR pixel values are scene radiance, so a matte surface of albedo ρ reads `ρ·E/π`.

**LEDs.** `rig.py` sets each white LED's Blender power from `capture.yaml leds.white_lumens_per_led`. Treat the LED as Lambertian, so on-axis intensity is `I = Φ/π` cd and `power_w = 4π·I/683`. Use a spot of 150° with blend 0.6.

**Ambient conditions.** `ambient.condition` is one of the following. The `lux` column is the target illuminance at the plate centre, sampled log-uniformly; CCT is in K.

| id | Scene | lux | CCT | burst `change` |
|---|---|---|---|---|
| `C0_dark` | none: world strength 0, no lights | 0 | – | 0 |
| `C1_office` | 2–6 ceiling LED panels or tubes 2.2–3 m up, plus a dim world | 250–700 | 4000–5000 | frame σ 0.00–0.03 (LED driver ripple) |
| `C2_warm_bulb` | 1–2 point bulbs 0.4–1.2 m away, low-ish angle, plus a dim world | 300–3000 | 2700–3000 | σ 0–0.01 |
| `C3_window` | a large area light 1.5×1.2 m, 1.5–4 m away to one side and above, plus a sky-tinted world | 1500–20000 | 5500–7000 | σ 0–0.01 |
| `C4_side_lamp` | a work lamp (spot, 30–60°) at 10–30° elevation, 0.4–1 m, plus a box occluder (hand or arm) casting a shadow edge across the plate | 1000–8000 | 3000–6500 | σ 0.01; `shadow_shift_mm` 0–1.5 per frame (a hand moving ≤ 15 mm/s; was 0–4) |
| `C5_reflected_lamp` | an office scene plus one bright panel placed on the specular-reflection path, so its mirror image falls inside the plate for the open device | 300–1500 | 4000–6000 | σ 0–0.02 |
| `C6_sun` | a sun lamp (angle 0.53°) at 20–70° elevation, plus a sky world | 30000–100000 | 5500–6500 | σ 0–0.01 |

- `lighting.ambient_for` turns lux into power with simple analytic formulas:
  - point or spot: `P = 4π d² E_W / cosθ`;
  - sun: strength `= E_W / cosθ`;
  - small area light: `P ≈ π d² E_W / (cosθ_l · cosθ_s)`;
  - world: strength `s = E_W_world / π` (unoccluded).
- The world contributes 5–15 % of the condition lux, except in C0.
- Every ambient light and the world are in light group **AMB**.

**Device modes** (`extras.device.mode`, with `extras.device.illumination` = `pods` | `ring`):
- `open`: the handheld as built, lit by its two near-coaxial LED pods (`illumination: pods`).
  - Pose as DESIGN: distance 185–215 mm sharp, 20 % wide 160–250 mm; tilt ±12°; roll ±20° + k·90°.
- `hood`: device v2 (section C2), a clip-on black **hood head** that rests on the bench or part, lit by its dark-field LED ring (`illumination: ring`; `twin.yaml rig.ring.enabled: false` gives the v1 hood with the pods).
  - Distance 200 ± 1 mm; tilt ±1.5°.
  - `extras.hood = {"top_mm": [170, 160], "bottom_mm": [200, 190], "wall_mm": 2, "gap_mm": g, "albedo": 0.03}`. Top and bottom are in the camera frame, centred on the optical axis; the top sits at camera-frame z = +17 mm (the front of the pods) and the bottom at the plate's front-face plane (world Z = 0) + `gap_mm`.
  - `g` = 0–1 mm if it rests on the part, or the plate thickness if it rests on the bench (≈70 %).
  - The hood is a matte-black frustum, visible to the camera and casting shadows. Ambient leaks in only through the gap.
- **Sampler weights:** open 0.4, hood 0.6 (was 0.7 / 0.3). The **test matrix** renders every test plate under every condition with both `open` and `hood` (14 rows; was open plus C4 and C6 with the hood).

## C2. Device v2 (redesign, 14 Nov)

**Why the v1 method failed (review reports in `docs/twin/build_reports_steel/`).** In the first real Blender matrix (2 plates × 9 rows, review_lighting-render.md) every burst variant was RETAKE, including the dark room:
- (a) **Pod glare.** The near-coaxial pods mirror into polished and brushed metal: both pods clipped over 0.8–5 % of the plate (gate limit 0.5 %), worst with the hood, which looks straight down. The "tilt 10°" message cannot help there. The pilot below found the cause: the fixed exposure was set on a white card at 85 % FS, and bare metal under the pods reflects more than the card (B1 clipped over 34–50 % of aluminium and cold-rolled plates).
- (b) **Hand shake.** The open shake of 0.2–2.5 px/frame gave B2 − B1 above `motion_max_px` 3.0 in about a third of captures (review_post.md, review_eval.md). It was also applied in render px, so a 0.25× render shook 4× too much.
- (c) **C4 model.** The moving hand shadow translated the whole AMB pass, so alignment locked onto the ambient-lit texture of the A frame (motion_px 16 at 1.5 px true shake; review_post.md, review_lightproc.md).
- (d) **C5 lamp.** The reflected panel clipped the ambient frame over 19–27 % of polished plates.
- (e) **Saturated B frames.** Fully clipped B1/B2 gave NaN statistics and the generic "Capture refused" message.

**The design.**
1. **The hood head carries a low-angle LED ring** (`lighting.ring_lights`, `twin.yaml rig.ring`). The same 18 white LEDs (36.7 lm, Lambertian, spot 150°) sit on the hood's inner walls, `h_mm` 18 above the plate face, `inset_mm` 6 in from the walls, spread evenly round the inner cross-section (about 40 mm apart). Each is aimed at the optical-axis plate point, giving 8–12° elevation there. Flat specular metal mirrors this light away from the lens, so there are no LED highlights. Scratches, dents, pits, rust and stains scatter, or change the scatter, into the lens. Group L = the LEDs on the image left (camera x < 0), R = the rest: 9 + 9 on the pods' two GPIO channels. Ring irradiance at the centre is about 3.1–3.5 klux (pods: 5.2 klux). Both halves light every plate point (≥ 0.1× the centre each); the near LEDs give up to 7× the centre at the edge of a 150 mm plate.
2. **Combination by illumination** (`capture.yaml process.combine: {pods: min_luma, ring: mean}`, from `burst.meta['illumination']`). Dark-field signals are directional: a scratch wall lit from one side scatters far more than from the other, so min(L, R) would drop it. The ring uses M = mean(Lc, Rc) = (B − A)/2 with less noise. Relief is kept.
3. **Braced alignment for the ring** (`process.align_braced: {ring: true}`). L and R are lit from opposite sides, so their correlation with B1 is biased by relief and directional scatter: 0.3–0.7 px with no motion at all in the pilot, enough for 'changed' ghosts. The hood rests on the part, so L, A and R follow the unbiased B1 → B2 shift linearly in time. Errors fell to 0.01–0.08 px, closure_sigma to about 1.0 and ghost_bins to 0.
4. **Exposure.** `sensor.analogue_gain` goes from 1.5 to 1.0 and `burst.ref_full_scale` from 0.85 to 0.57: the same photons, a 1.5× wider range. Both-pod clipping in the dark room fell from 1.1–2.2 % to 0–0.7 %. The ring's clipped band on a rough 150 × 100 hot-rolled plate (3–8 mm from its edge) fell from 2.6 % to 0.66 %.
5. **Gate recalibration** (each change is documented in `capture.yaml`):
   - masked pixels are inpainted, and `masked_frac_max` goes from 0.005 to 0.02;
   - pixels the ambient frame clips get their own limit, `masked_ambient_max` 0.0005 (a ~2.5 mm glint), refused as `glare_ambient`. A defect that scatters ambient light clips A on itself. In the smoke matrix, window light clipped 75–85 % of a scratch on aluminium, inpainting erased it and the capture passed with PASS against a HOLD_FOR_QA truth;
   - `motion_max_px` goes from 3 to 8 sensor px, now measured in full-resolution sensor px. Accepted captures stayed within 1.1 DN of the still capture up to 12 px of motion. Real misalignment is the closure/ghost checks' job: a roll walk of 0.05–0.4°/frame was refused in 24/24 bursts;
   - `led_level_min` is {pods: 0.02, ring: 0.003}, because dark-field is dark by design on flat metal (polished plates 0.008–0.019 FS);
   - non-finite statistics with ≥ 50 % of the plate clipped give `too_bright`, not `invalid`;
   - a local ambient clip at r ≤ 1 (≤ 30 % of the plate) gives the new `glare_ambient`: "Lamp reflection on the part: tilt, shade it or fit the hood, then retake.";
   - any gate threshold may be `{pods: x, ring: y}`;
   - second review (15 Nov): ring ghost rule `max_ambient`, plate-wide flicker `flicker_err_max` 0.2, ring `leak_r_max` 0.15 (`hood_leak`); see below.
6. **Twin models.** In C4, only the illumination (the AMB low-pass at `burst.shadow_sigma_mm` 5, normalised inside the plate outline) moves, by 0–1.5 mm/frame; texture and reflections stay. `capture.shake_px_per_frame` is in full-resolution sensor px and is scaled to the render (`burst.res_scale`): open 0.2–1.5 (5 % 3–6), hood 0–0.2.

**Twin pilot (5 matrix plates: aluminium, cold-rolled ×3, hot-rolled 150 × 100; 14 rows; 2 variants; 0.4× render, 16 spp; burst arm re-run on the kept EXRs):**

| Arm settings | Hood: retakes per condition C0–C6 | Open: retakes per condition (main reason) |
|---|---|---|
| v1 gate and exposure | 4/10 in every condition (saturated) | C0 8/10, C1 8/10, C2 7/10, C3 10/10, C4 8/10 (saturated), C5 9/10 (glare_ambient), C6 10/10 (too_bright) |
| **v2** | **0/10 in every condition** (the hood seals: identical verdicts in all 7 per plate) | C0 3/10, C1 5/10, C2 2/10 (changed), C3 9/10, C4 4/10, C5 7/10 (glare_ambient), C6 10/10 (too_bright) |

**Smoke matrix** (`ayas-twin matrix --n 1 --res-scale 0.25 --samples 16`; aluminium plate s900000, one scriber scratch). The burst arm passes every hood row (C0h–C6h, 2/2 each, SSIM 0.975 vs C0h) and refuses open C1, C3, C4 and C5 as `glare_ambient` and C6 as `too_bright`. With an oracle detector (the boxes post judged visible in each arm's grid), `eval_lighting` gives the burst arm PASS under the device-v2 criterion: hood F1 1.00 in all 7 conditions and no silent failure open. Under the ring the scratch's contrast is 56 against 11 under the pods.

The open handheld still refuses often on specular plates ('changed': the pods' L/R correlation bias, which a moving handheld cannot avoid by bracing). That is allowed by section G; the hood head is the product. The calibration used matrix seeds 900000–900004 and the full-resolution pilot tuned the gate on 900011, 900027, 900028 and 900068, so the lighting experiment must start at `--start-seed 900100` or later (training/README.md honesty rules). Known open-mode blind spot: ambient flicker of 3 % or more at r > 0.3 on a near-black plate passes with a 4–5 DN error (the old 3 px motion limit happened to refuse that toy case); the hood blocks it.

**Full-resolution pilot (14–15 Nov, `reports/pilot_lighting.md`).** 4 plates (cold-rolled, hot-rolled, aluminium, brushed stainless; seeds 900011/27/28/68) × 7 conditions × {hood, open}, 64 spp, 2 variants. With the hood the burst arm was accepted in 56/56 captures, and every image was within −2.6..+1.1 DN (1.1 mm low-pass, p1..p99) of the noise-free LED image of its render, the same as in the dark room. In the open the gate refused 39/56 and there was no silent failure. The pilot changed the method in five ways, each documented in `capture.yaml`:
- closure refinement of the L, A, R shifts (`process.align_refine`), so still captures leave no 'changed' ghosts;
- ring pixels clipped in L or R only are kept white instead of inpainted: inpainting had erased the brightest dark-field defects, for example a pit cluster's contrast 101 → 3 DN;
- `process.ghost_closure {pods: max, ring: min}`: with min, a local ambient step after the L frame passed (a moving hand shadow, M 56 % dark);
- a 32 px coarse closure test (`ghost_coarse_*`) for flicker under a mirrored lamp;
- `gate.ambient_dominated_max`: where a mirrored lamp outshines the LEDs more than 8× over more than 2 % of the plate, M is noise, biased bright by the clamp.

A 3 mm hood lift (`data/twin_pilot/leak3`) let in at most r 0.07 (sun) and was processed correctly. Detection is still measured with an oracle only.

**Second independent review (15 Nov, `reports/pilot_lighting.md` section 11).** A full-resolution re-render of one pilot plate reproduced the gate decisions and metrics; adversarial bursts on the kept renders found silent corruptions, and three gate rules were added (each documented in `capture.yaml`, none changes a pilot hood verdict):
- `process.ghost_closure.ring: max_ambient`: both closures, or either one where the ambient frame could have made the deviation. With min, a hand shadow arriving at the A frame under a leaking hood passed (+9 DN over 70 mm²); max would refuse dark-room hood captures (LED-made residue, no ambient).
- `gate.flicker_err_max` 0.2 (`lightproc._ambient_flicker`): the closures and B2 − B1 regressed on [A, B1 − A] measure a plate-wide ambient change between frames that no bin or block shows; times the 99th percentile of A/M it is the M error it makes. Open, 5 % random flicker at r 0.11 had passed with M +15 DN over 8 % of the plate.
- `gate.leak_r_max: {ring: 0.15}` → `hood_leak` ("Light is getting under the hood: seat it flat ..."). The seated hood sees r ≤ 0.021 (≤ 0.11 on a mirror-polished plate, whose dark-field LED level is tiny); with more leak light, flicker and ambient steps passed with 4–8 DN errors.
- Ring texts for `saturated` and `glare_ambient` (the hood cannot be tilted; a part reaching under the hood wall clips at the ring LEDs).
- Still open: only b_i − e_L − e_R are observable, so an ambient change with e_L = −e_R and steady brackets passes (open mode; the hood removes the ambient it needs). All 18 ring LEDs are inside the camera's field of view, outside the plate (the camera is 63–76° off each LED's axis, inside its 150° cone): the twin has no lens flare, the real hood needs a lip over the LEDs.

**Phase 2 hardware.** The hood head is a matte-black printed frustum/skirt, top 170 × 160 mm at the pods' front, bottom 200 × 190 mm resting on the part, 2 mm wall, about 183 mm tall at the 200 mm working distance (`lighting.HOOD`). The LED ring is the existing 12 V 3-LED white modules: six modules (18 LEDs) glued to the inner walls 15–20 mm above the rim (twin: 18 mm), about 6 mm proud of the wall and tilted to aim at the centre. Fit two on each long wall and one on each short wall, about 40 mm LED pitch. The three image-left modules go on the L pod channel and the three image-right on R (existing MOSFETs and GPIO; `app.yaml part.illumination: ring`).

## D. Plan JSON changes (`schema: ayas.twin.scene/2`)

- `plate.material.kind` (section B).
- `ambient`:

  ```json
  {"condition", "lux", "cct_k", "world_color", "world_strength",
   "lights": [{"name", "kind": "area|sun|point|spot", "pos_m", "dir",
               "size_m": [w, h], "shape": "RECTANGLE|DISK",
               "power_w" (area/point/spot) | "strength" (sun, W/m^2),
               "angle_deg" (sun), "spot_deg", "blend", "color"}],
   "occluders": [{"kind": "box", "center_m", "size_m", "rot_deg": [x, y, z]}],
   "change": {"frame_sigma", "shadow_shift_mm"}}
  ```

  The old `ceiling` key is still read by Blender, for v1 plans.
- `extras.device = {"mode": "open|hood", "illumination": "pods|ring"}` and `extras.hood` (section C, C2). With the ring, the L/R plan lights are `lighting.ring_lights` (`"mount": "ring"`) instead of the pods' white LEDs.
- `capture`:

  ```json
  {"sensor": "full", "shake_px_per_frame": s, "led_jitter": 0.0-0.05}
  ```

  - `shake_px_per_frame` (full-resolution sensor px; burst.py scales it to the render) is 0.2–1.5 when open (5 % of scenes get 3–6) and 0–0.2 with the hood (was 0.2–2.5, 5 % 3–8, and 0–0.3).
  - post draws the per-frame random walk from the plan seed.
- `defects[].cls/kind` (section A); `render.samples` defaults to 64.
- `make_plan(cfg, seed, split, mode=None, source=None, coupons=None, condition=None, device_mode=None)`. Overriding `condition` or `device_mode` must change ONLY `ambient`, `extras.hood/device`, `camera` (hood forces near-straight pose) and `capture`. Plate, defects and distractors stay identical for the same seed, which is what makes the test matrix possible. Use the existing per-part RNG streams.

## E. Capture burst: simulation (twin) and processing (shared with the Pi)

### `src/ayas/twin/burst.py` (twin only)

`simulate_burst(groups, plan, rng, capture_cfg, roi_mask=None) -> lightproc.Burst` takes the light-group passes (linear radiance, float32 H×W×3: `L`, `R`, `AMB`, maybe `UV`) and builds each frame.

For frame f in `B1, L, A, R, B2`, with LED state `(l, r)` from `capture.yaml burst.led_state`:
- `S_f = k · (l·gL·L + r·gR·R + a_f·AMB)`.
- `k = ref_full_scale · π / (ref_albedo · E_led)`, where `E_led` is the analytic LED irradiance (W/m²) at the plate point on the optical axis, summed over the plan's L and R lights. This is the fixed exposure of the real device; it does not change with the scene.
- `gL`, `gR` = 1 ± `led_jitter`.
- `a_f = 1 + N(0, frame_sigma)`. If `shadow_shift_mm > 0`, move only the AMB pass's illumination by a per-frame shift along a random direction (a moving shadow, `burst.shadow_shift`): AMB × E(x − s)/E(x), with E = its low-pass at `burst.shadow_sigma_mm`, normalised inside the plate outline.
- Translate the whole frame by the hand-shake random walk (`shake_px_per_frame` × `res_scale`: sensor px to render px).
- `meta` carries `illumination` (plan `extras.device.illumination`, default pods) and `px_scale` (render px per sensor px).
- Noise in electrons: `e = Poisson(S_f · full_well)` + `N(0, read_noise)`. Quantise to `bit_depth`, clip at full scale, and return the frame normalised to [0, 1] full scale. These are linear, black-level-free RGB frames, the same contract as the Pi after demosaic.

`simulate_naive(groups, plan, rng, capture_cfg) -> uint8 RGB` is the **baseline arm** (old method):
- one both-pods frame `L + R + AMB`;
- auto-exposure: plate median at 0.18 FS;
- grey-world AWB;
- the same noise;
- sRGB gamma and a mild unsharp mask.

### `src/ayas/lightproc.py` (shared; runs on the Pi; numpy + cv2 only)

**Data structures:**
- `Burst(frames: dict[str, float32 H×W×3], meta: dict)`.
- `Processed(image_u8, M_lin, relief, sat_mask, shifts, metrics: dict, gate: GateResult)`.
- `GateResult(verdict: "PASS"|"PASS_FLAGGED"|"RETAKE", reasons: list[str], message: str)`.

**`process_burst(burst, roi_mask, cfg) -> Processed`:**
1. `estimate_shift(a, b)`: phase correlation on luma, sub-pixel. Align L, A, R and B2 to B1 by translation (`cv2.warpAffine`, INTER_LINEAR). With `process.align_braced` for the illumination (ring: the hood rests on the part), only B2 is estimated and L, A, R follow it linearly in time (C2).
2. `motion_px` = |shift(B2 vs B1)| in full-resolution sensor px (frame px / `meta.px_scale`; `motion_frame_px` keeps the frame px). `residual_sigma` = robust σ of (B2 − B1) in the ROI, over the expected shot-noise σ.
3. `Lc = max(L − A, 0)`, `Rc = max(R − A, 0)`, `BA = B1 − A`.
4. `sat_mask` = A saturated, or (pods) both L and R saturated, or (ring) L or R saturated. `masked_frac` = its share of the ROI; masked pixels are inpainted in `image_u8` on accepted captures.
5. `M` by `process.combine[meta.illumination]`: `min_luma` (pods) = per pixel, the RGB of whichever of Lc and Rc has the lower luma, fully saturated pixels taking the other frame; `mean` (ring) = (Lc + Rc)/2, a clipped half replaced by the other.
6. `relief = (lum(Lc) − lum(Rc)) / (lum(Lc) + lum(Rc) + 1e-4)`.
7. Fixed white balance: divide by `leds.colour`. Optional flat-field (`flat` gain map from a white-card calibration, used if given). Divide by the ROI `percentile`, clip to [0, 1], apply `gamma`, convert to uint8.
8. **Metrics:**
   - `r_ambient` = median_ROI(lum A) / median_ROI(lum BA);
   - `sat_frac_a` = fraction of the ROI with A ≥ `sat_level`;
   - `led_level` = median_ROI(lum BA);
   - `blur` = variance of Laplacian of lum(M) in the ROI, plus `blur_rel` against `cfg.get('blur_ref')`;
   - `motion_px`, `residual_sigma`, `masked_frac`, `shifts`;
   - `sat_frac_b` (B1 or B2 clipped), `illumination`, `combine`, `align_mode` (braced | free).
9. **`quality_gate(metrics, cfg)`** applies `capture.yaml gate`:
   - **RETAKE** if any of these hold: r > `leak_r_max` (ring: `hood_leak`); r > `motion_strict_upto`; `sat_frac_a` > max (`glare_ambient` when it is local, ≤ `glare_frac_max` at r ≤ `glare_r_max`, else `too_bright`); `led_level` < min; `motion_px` > max; (1 < r ≤ 2 and `motion_px` > `motion_strict_px`); `masked_frac` > max; `masked_ambient_frac` > `masked_ambient_max`; `residual_sigma` or `closure_sigma` > max; `ghost_bins` > max (`process.ghost_closure` min | max | max_ambient); `ghost_bins_coarse` > max; `ambient_flicker_err` > `flicker_err_max`; `blur_rel` < min; a required metric not finite (`too_bright` when ≥ 50 % of the plate is clipped, else `invalid`).
   - Any threshold may be `{pods: x, ring: y}`, picked by `metrics.illumination`.
   - **PASS_FLAGGED** if `pass` < r ≤ `flag`.
   - **PASS** otherwise.
   - `message` = the first matching `gate.messages` entry.

Also provide `naive_process(frame_u8)` (identity: the ISP output is used as is) and `luma(x)`. **Raw-only maths:** never feed ISP or gamma output into the subtraction.

### Post integration

- `post.process_scene` runs, per variant:
  1. `burst.simulate_burst`;
  2. `lightproc.process_burst` with `roi_mask` = the projected plate mask;
  3. rectify `Processed.image_u8`;
  4. labels and tiles.
- **Sidecar fields:** `condition`, `lux` (target × the variant's ambient multiplier), `device_mode`, `gate`, `metrics`, `arm`.
- **Ambient sweep per variant:** post may scale the AMB pass by a multiplier drawn log-uniformly within the condition's lux range relative to the rendered lux, so one render gives several lux levels.
- **Training tiles** come from arm `burst` (config `post.train_arm`), and only from variants whose gate verdict is not RETAKE.
- **Evaluation output** (config `post.eval_arms: [burst, naive]`): full rectified grids per arm go to `out/grids/<arm>/<split>/<scene>_v<k>.png`, plus the sidecar.
- **Pi demo output:** if `post.save_bursts` is true, also write `out/bursts/<scene>_v<k>.npz`: the frames as float16, the image-order plate corners in the ROI image, the plate size and the ground-truth labels.

## F. Severity, record and advisor

**`configs/spec_steel.yaml` (domain: steel_surface).** The rows follow `docs/research/2026-11-14_steel_severity_standards.md` recommendations 1–7. Thresholds are ILLUSTRATIVE unless they are standard values.
- **Tiers:** Imperfection (log) → minor; Defect (repair) → major; Critical defect → critical.
- **Dispositions:** PASS, CLEAN_AND_REINSPECT (stain), REWORK (blend, grind or clean), HOLD_FOR_QA (a depth check is needed, or confidence is low), REJECT (crack, critical).
- **Modifiers:** +1 level within 5 mm of an edge or hole; cluster and count rules; depth is never measured, so above the minor band a "depth not measured" note sends the part to HOLD.
- **References** whitelist: EN 10163-2 / ISO 7788, ISO 8501-1 / IS 9954, ISO 8501-3, IS 2062 (Part 1):2025 cl. 7, ISO 4628-3 bands, IS 1477-1, ISO 8502-3/-4.
- Plate thickness comes from the part config.

**Hand-off record: `ayas.inspection/3.0`.** This is the organisers' "one JSON hand-off" (defect, severity, confidence, box).

```jsonc
{"schema": "ayas.inspection/3.0", "part_id", "timestamp", "material", "thickness_mm",
 "verdict", "verdict_instruction",
 "capture": {"mode": "burst5|naive", "device_mode", "r_ambient", "lux_est", "motion_px",
             "sat_pct", "blur", "gate": "PASS|PASS_FLAGGED|RETAKE", "gate_message"},
 "defects": [{"id", "type", "confidence", "box_px": [x0, y0, x1, y1], "location_mm": {"x", "y"},
              "size": {...mm}, "near_edge_mm", "severity": "minor|major|critical",
              "tier": "Imperfection|Defect|Critical defect", "disposition", "rule", "reference", "action"}],
 "timing_ms": {"capture", "process", "detect", "rules", "total"},
 "backend": {"vision", "advisor"}}
```

- A RETAKE gate gives verdict `RETAKE` with no defects. **This is not a silent failure.**
- **Advisor (`llm.py`), steel mode:**
  - a system prompt with the reference whitelist;
  - `check_reply` rejects any reference not on the whitelist, banned phrases (certified/compliant/approved, "measures depth", "works in any lighting", "zero defects") and any number not in the record;
  - capture-quality advice (shade, hood, tilt, hold still) comes from the `capture` block;
  - on any problem it falls back to the template answer and records which backend answered.

## G. Test matrix and the lighting experiment

**`src/ayas/twin/matrix.py`:** `write_matrix_plans(cfg, n_plates, out, conditions=ALL, hood_conditions=None, start_seed=900000)`; `hood_conditions` None = `matrix.HOOD_CONDITIONS` = every condition (device v2), those of them in `conditions`; the CLI's `--hood-conditions` with no value gives no hood rows.
- For each test plate seed, write one plan per (condition, device_mode), with split `test`.
- The plate, defects and distractors are identical across a plate's rows.
- The same plate seeds are never used for training.

**`training/eval_lighting.py`:** for each arm × model × condition × device_mode, on the full grids with sidecars, report:
- defect-level precision, recall and F1 with Wilson 95 % CIs (the `eval_plates` matching rules);
- false positives per clean plate;
- retake rate, the gate reasons counted and the main reason (the one whose message was shown);
- **silent-failure rate**: gate passed, but the verdict differs from the ground-truth verdict computed by the rules engine on the GT defects;
- SSIM of the rectified grid against C0 of the same plate and device (hood rows against C0 hood, open rows against C0 open), for every row and arm.

**PASS criterion (device v2, as stated in the report):**
- **HOOD** (device v2, the product configuration), for every condition C0–C6: retake rate ≤ 5 %, F1 ≥ F1(C0, hood) − 0.05, verdict agreement with C0 (hood) ≥ 95 %, and 0 silent failures;
- **OPEN** (the handheld without the hood): 0 % silent failures in every condition; retakes are allowed and reported per condition with the main gate reason.
- Every table keeps the naive arm (the old method) next to the burst arm; ground-truth visibility (`c0`) comes from the C0 capture of the same device.

**Models.** One YOLOv8n per arm: `naive` and `burst`. Both use the same training plans and the same tile pipeline. Outputs are `reports/lighting_matrix.md` and `reports/lighting_matrix.json`.

## H. On-device app (`src/ayas/app/`)

**Command:** `ayas-app --source twin:<dir of npz>|camera --detector onnx:<path>|hailo:<hef> --spec configs/spec_steel.yaml --advisor executorch|mock`

**Flow:**
1. **Source.**
   - `TwinSource` reads `out/bursts/*.npz`, a virtual camera that plays real-time-like.
   - `CameraSource` runs the picamera2 burst: fixed exposure and gain, AE/AWB off, raw SRGGB10 unpacked; GPIO toggles L/R pods with gpiozero; `capture_request(flush=True)` after every switch; black level subtracted; demosaic `cv2.COLOR_BayerRG2RGB`.
2. `lightproc.process_burst`.
3. Plate corners: ground-truth corners for the twin, or `calibration.find_plate_corners` on the processed image.
4. Rectify, tile, run `npu` detect, merge tiles.
5. mm measures, then `rules` with `spec_steel`, then the record.
6. Advisor: ExecuTorch via `llm.py` (runner binary, or the Python TextLLMRunner if available).
7. Output and storage:
   - the web UI: stdlib `http.server`, an auto-refresh page on the 5" display showing the image, boxes, verdict banner, advice and capture-quality line;
   - the SQLite log (`store.py`);
   - JSONL.
8. **Benchmarks:** per-stage ms, FPS, peak RSS, Llama TTFT, tokens/s and total time, written to `reports/bench_*.json`.

**Offline proof:** the app never imports network libraries, and the UI shows the network state.

## I. Ownership (build workflow, 14 Nov)

| Area | Files |
|---|---|
| steel-twin | `sampler.py`, `textures.py`, `geometry.py`, `coupons.yaml` (+ steel coupons), sampler/textures/coupons tests |
| lighting-render | `twin/lighting.py`, `rig.py`, `twin/blender/render_scene.py`, `twin/render.py`, render tests, plus a Blender overview-camera render for the demo |
| lightproc | `src/ayas/lightproc.py`, `src/ayas/twin/burst.py`, `configs/capture.yaml` (tune), tests |
| post | `post.py`, `realdata.py`, `tiles.py` (CLASSES), `pipeline.py`, `twin/cli.py`, `configs/twin.yaml`, post/realdata/frames-tiles tests |
| spec-advisor | `configs/spec_steel.yaml`, `rules.py`, `report.py`, `llm.py`, `schemas/`, `twin/records.py`, tests |
| eval | `twin/matrix.py`, `training/*`, `hailo/*` (classes 6), eval/npu tests, `tools/make_demo_video.py` |
| app | `src/ayas/app/*`, `pyproject` script, app tests |
