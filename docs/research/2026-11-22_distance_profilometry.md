# Green-LED optical profilometry on a Raspberry Pi, at 0.3-3 m, in any light? (research note)

Date: 22 Nov 2026. Status: desk research and simulation only. **Nothing here was measured on this project's hardware or on real metal.**
Labels: **[pub]** = published number with a source; **[snip]** = source seen only as a web-search snippet (lower confidence); **[calc]** = our own calculation or simulation; **[contested]** = a verification pass disagreed and the corrected value is given; **[assumed]** = a model input nobody has measured; **[memory]** = recalled, not checked against the primary document.

---

## 0. Bottom line (answer to the question)

1. **Interference itself on a Pi at 0.3-3 m: no.** The instrument needs the surface held inside a ~4 µm coherence gate while hand tremor is 30-200 µm. The reference arm has to match a 0.3-3 m path to ±2 µm. A 0.5 mm height scan is ~7,600 frames (254 s at 30 fps) [calc]. Optics and roughness are not the main blockers; motion, path matching and scan time are (section 2).
2. **What transfers is the phase-shifting fringe maths, with displayed or projected fringes:** deflectometry (PMD) for shiny plate and fringe projection (FPP) for diffuse plate. There the height ruler is geometry (millimetres per fringe), not λ/2 = 262 nm (section 3).
3. **None of them is independent of lighting or distance.** The calc envelopes against one criterion (depth σ ≤ 25 µm at the 0.2 mm limit, slope in range):
   - PMD grades bright finishes only as a fixed, shrouded station with the screen within ~0.3-0.6 m of the plate. The camera may be 1-3 m away.
   - FPP grades only at ~0.3 m in ≤ 10 klx, or at 1 m in the dark.
   - A line laser detects but does not grade.
   - Off-the-shelf depth cameras (ToF, stereo, LiDAR) are 50-1,600× short (section 3.0).
4. **Distance is cheap for the camera, expensive for the light.** A 50 mm lens on the HQ camera gives 0.093 mm/px at 3 m [calc]. But in-band irradiance from a lamp falls as 1/D² while ambient stays constant (section 4.1). So "a few metres" honestly means a detection and preview mode, with depth still measured in the hood.
5. **The hood is the only configuration that looked lighting-insensitive, and only in the Blender twin** (strict FAIL by 2 captures; unmeasured on hardware). **Phase 1 (due 23 Nov) builds nothing new.**

**Single criterion used throughout.** Grading 0.2 / 0.3 / 0.4 mm limits needs depth σ ≤ 25 µm [calc]. In the simulation's "smallest dent at SNR 5" terms (min dent = 5σ), that means:
- **grades** when the smallest dent is ≤ 125 µm and the 0.2-0.4 mm dent slopes are in range;
- **detects only** when the smallest dent is 125-200 µm (σ 25-40 µm);
- **fails** above 200 µm.

Correction to the framing. The hood does **not** hold the camera 18 mm from the plate. The AI Camera sits **200 mm** from the plate. **18 mm is the height of the low-angle ring LEDs** (configs/twin.yaml `rig.ring {n_leds 18, h_mm 18}`; the LEDs point 8-12° above the plate). The AI Camera cannot focus closer than 20 cm (docs/research/2026-11-05_camera_lighting.md).

---

## 1. What you saw

What you saw is almost certainly a **coherence-scanning interferometer** (CSI, also called VSI or white-light interferometry) or a **phase-shifting interferometer** (PSI). Examples are the Zygo NewView and the Bruker Contour GT, usually run with a green LED.

How it works:
- A Michelson, Mirau or Linnik objective splits the light into two paths: one to the part, one to an internal reference mirror. The recombined light forms fringes.
- **Each fringe is λ/2 = 262 nm of height** at λ = 525 nm [calc].
- **Coherence length** l_c = 0.441·λ²/Δλ: **4.1 µm** for a 30 nm LED, **12.2 µm** with a 10 nm filter [calc]. Fringes exist only where the two paths match to within l_c. That is why it scans in z.
- **PSI** reaches ~0.1 nm, but only on smooth parts with less than λ/4 between neighbouring pixels. Bruker quotes ~135 nm in green [snip] (https://inup.iitm.ac.in/assets/docus/manuals/Bruker_Contour_GT-K_Optical_Surface_Profiler_User_manual.pdf).
- **CSI** steps z by λ/8 = 65.6 nm, so a 0.5 mm scan is **7,619 frames** [calc]:
  - 9.4 s on a Zygo NewView 9000, which scans at 53 µm/s [pub] (https://www.lambdaphoto.co.uk/pdfs/Zygo/LAMBDA_NewView9000Specs.pdf);
  - **254 s** at 30 fps [calc].
- **Close proximity:** interference objectives have 3.4-10.3 mm working distance, NA 0.075-0.70 and a 0.06-2.5 mm field [snip] (https://www.edmundoptics.com/f/nikon-interferometry-objectives/13308/). The largest field is the Zygo at up to 17.5 mm.
- **Environment:** the Zygo specifies VC-C isolation and < 1 °C drift per 15 min [pub, same PDF].

What we need is much coarser. The EN 10163-2 / ISO 7788 limits in `configs/spec_steel.yaml` are 0.2 / 0.3 / 0.4 mm for 3-8 mm plate. Judging them needs 20-50 µm uncertainty, about **25 µm σ** [calc], roughly 10⁵× coarser than interferometry.

---

## 2. Interference at 0.3-3 m on a Pi: not feasible

All [calc] (interferometry-limits workflow summary; source file not available in the scratch dir) unless marked otherwise.

| CSI/PSI requirement | Handheld Pi at 0.3-3 m | Gap |
|---|---|---|
| Reference arm matched to the standoff within ±2 µm | Aluminium arm drifts 23 µm/K per metre, so it must be held to ±0.09 K | Lab only |
| Surface inside a 4 µm coherence gate | Hand tremor 30-200 µm rms at 8-12 Hz. Inside the gate **0.8-5 %** of the time | Far beyond VC-C |
| Fringes frozen during the exposure | Needs ≤ 3.7-25 µs exposure; the path moves 240-1,600 fringes between 30 fps frames | Phase steps become random |
| Scan time | 254 s per 0.5 mm. Finding the surface at ±5 mm takes 152,400 frames, **85 min** | Not real time |
| Air path | 0.92 µm apparent height per K per metre | Larger than the whole PSI range |

**Optics and roughness are not the main reason [contested, corrected].**
- **Mirror finishes.** The AI Camera's NA is 4.4×10⁻³ to 4.4×10⁻⁴ at 0.3-3 m. Fringes alias unless local tilt is < 0.91 / 0.27 / 0.089 mrad. EN 10029 class N allows up to 16-36 mrad [snip] (https://www.scribd.com/doc/308863699/en10029-pdf).
- **Rough finishes.** Single-wavelength phase is random. Coherence scanning still works in principle, with height uncertainty about equal to the roughness: ~0.1-0.75 µm for 2B, 4-9 µm for hot-rolled [calc]. The claim that this does not depend on the aperture follows Häusler & Willomitzer (https://arxiv.org/pdf/2112.05113); attribution not re-verified.
  - Even if the height σ is met, the lateral resolution at that NA is **73-726 µm** [calc], so small pits would not be resolved laterally.
  - It would also still need a fixtured, path-matched, vibration-free setup, which this note does not propose.

**Signal.** Interference rejects ambient light as a DC term, but an eye-safe 1 mW laser spread over 50×50 mm gives only **0.03-0.3×** the in-band sunlight (1-10 nm filter) [calc]. That calc uses ASTM G173 sun irradiance of ~1.4-1.54 W/m²/nm at 520-530 nm, which was seen only as a snippet [snip] (https://rredc.nrel.gov/solar/spectra/am1.5/).

**Coherent methods that do work at distance, and why they are out:**

| Method | Published | Why out |
|---|---|---|
| FMCW ranging (Aeva Eve 1D) | sub-µm claimed, 0.1-20 m, single point [pub] (https://www.aeva.com/eve-1d) | Single point, quote-only |
| Long-range swept-source OCT | 15 µm over 1.5 m, lab [pub] (https://pmc.ncbi.nlm.nih.gov/articles/PMC5325157) | Lab system |
| Multi-wavelength holography (Fraunhofer HoloTop) | inline industrial [pub] (https://www.ipm.fraunhofer.de/content/dam/ipm/en/PDFs/product-information/PK/GIM/HoloTop-3D-inline-measurement.pdf) | Quote-only. A 0.4 mm range needs two lines within 0.35 nm [calc] |
| Dynamic interferometry (4D InSpec) | 4 µm, 8×8 mm, ~1 s [snip] (https://www.photonics.com/pr60558/4D_InSpec_Surface_Gauge) | Its tip rests on the part; quote-only |

The one sourced optic price is a single interference objective at **USD 5,095, ≈ ₹4.5 lakh** [pub] (https://www.edmundoptics.com/p/5x-nikon-cf-ic-epi-plan-ti-interferometry-objective/16066/). That is ≈ 45× the ₹10k module rule [calc].

**Verdict:** use a lab interferometer or stylus profiler only off the device, as ground truth for coupon depths.

---

## 3. Alternatives, judged against σ ≤ 25 µm

### 3.0 Methods considered and rejected (one line each)

| Method | Number | vs 25 µm σ | Why rejected |
|---|---|---|---|
| ToF camera for Pi (Arducam) | ±2 cm far / ±4 cm near mode [snip] (https://www.arducam.com/blog/time-of-flight-camera-raspberry-pi/, https://linuxgizmos.com/arducam-launches-30-tof-depth-camera-for-raspberry-pi-devices/). Newer RGBD module: ~1 % of range [snip] (https://www.arducam.com/product/time-of-flight-camera-for-raspberry-pi/) | 800-1,600×; 200× at 1 % of 0.5 m [calc] | Specular steel also causes multipath and dropout |
| Active stereo (RealSense D405) | ±2 % at 50 cm [snip] (https://intelrealsense.com/depth-camera-d405/). "<1 mm at 7 cm" is a feature size, not a depth error | 1.4 mm at 7 cm = 56×; 10 mm at 50 cm = 400× [calc] | Needs texture; shiny plate has none |
| Phone LiDAR (iPhone Pro) | ~±1 cm on small objects in peer-reviewed tests [snip] (https://link.springer.com/10.1038/s41598-021-01763-9) | ~400× [calc] | Not a Pi part |
| Focus variation / depth-from-focus | DOF = 2·N·c·(1+m)/m², height res ≈ DOF/10-20. HQ + 50 mm at f/1.8: ~17-33 µm at 0.3 m, ~0.2-0.4 mm at 1 m, ~2-4 mm at 3 m [calc] | Passes only at ~0.3 m, and only on textured surfaces | Needs texture (fails on bright steel) and a focus sweep of tens of frames. Commercial units (Alicona) use microscope objectives |
| Chromatic confocal (Micro-Epsilon confocalDT) | nm-class resolution, mm range, standoff ~28-220 mm by model [snip] (https://www.micro-epsilon.com/distance-sensors/confocal-sensors/?sLang=en) | Meets σ | Point or line sensor at tens of mm, so it is a contact-distance tool; price far above the rule (quote-only) |
| Shape-from-polarisation | Needs a polarisation sensor (Sony IMX250MZR). Cheapest listed camera US$2,530 [snip] (https://machinevisionstore.com/catalog/details/12316) | Gives normals only, with ambiguities on metals; no validated σ | Not Pi-native; ≈ 24× the ₹10k rule [calc] |
| Photogrammetry / SfM | Needs matchable texture | (no number) | Bright and specular steel has no stable texture |

### 3.1 Phase-measuring deflectometry (PMD): bright finishes

**How it works.** A screen shows phase-shifted sinusoidal fringes, and the camera views their mirror image in the plate. Phase gives slope, and integrating slope gives height.
- Slope sensitivity: Δα = p·Δφ/(4π·d_s), for fringe period p and screen-to-plate distance d_s [calc].
- References: https://www.osti.gov/pages/biblio/1438323-review-phase-measuring-deflectometry ; https://archiv.optik.nat.fau.de/forschung/osmin-former-group/deflectometry-sub-nm-precision

**Noise is not the limit, with a green bandpass filter and inside the model.** Reference setup: Pi HQ camera, 24" monitor. Slope noise is 21-26 µrad per pixel at 0.3-3 m [calc, simulation/pmd_sim.py].
- In the ideal model, depth noise is **µm-class**, i.e. 10-50× inside the 25 µm criterion.
- Real error will be **calibration-limited**: about 2 µm per 1 % error in d_s, plus the height-slope ambiguity. Neither is modelled.
- Do not quote sub-µm figures. The floor rests on an assumed 0.3 mrad texture clutter, and the Monte Carlo matches the analytic model only within 0.59-1.28×.
- Without the bandpass filter, at d_c = d_s = 3 m in office or sky light, there are no valid fringes.

**What limits it:**
- **Slope range ≈ (usable screen height)/(4·d_s)** [calc; consistent with L_scr/(2d) in reflected angle, snip, https://www.bnl.gov/tcp/uploads/files/bnl-2023-010j.2.pdf].
  - A full 24" screen gives ±249 mrad at 0.3 m and ±25 mrad at 3 m. A 6.5 mm-wide dent is then measurable to 0.405 mm at 0.3 m and 0.040 mm at 3 m.
  - **To grade 0.2-0.4 mm dents, the screen must be within ~0.3-0.6 m of the plate** [contested, corrected].
  - Steeper walls give dropout: detected, no depth.
- **Camera far, screen near.** With d_s ≈ 0.3 m and the camera at 1-3 m, σ 5 mm dents are measured to 0.4 mm. σ 2 mm dents are measured only to ~0.2 mm (0.05-0.1 mm in some GS cases) [calc, corrected].
  - So 0.3-0.4 mm narrow dents do **not grade** with the 121 mm screen the script sizes.
  - The screen sits in the camera's line of sight, so the layout needs an oblique arrangement or a beamsplitter (not modelled).
- **Screen size** ≈ 1.1·W·(1 + d_s/d_c): 220 mm for a 100 mm plate, 660 mm for a 300 mm plate [calc].
- **Two different tilt numbers:**
  - (a) **Static placement budget** ≈ screen margin/(2·d_s): **17 mrad at d_s 0.3 m, 1.7 mrad at 3 m** [calc]. This is how far the patch may be tilted before its reflection leaves the screen. A rigid tilt inside it is removed by plane fitting.
  - (b) **Motion during the phase sequence:** Δα = Δφ·p/(4π·d_s) = **57 µrad** for Δφ = 0.05 rad, p = 4.3 mm, d_s = 0.3 m (17 µrad at 1 m) [calc]. Hand shake is ~70 mrad in 2 s, **≥ 1,200× too much**.
  - **So PMD is not handheld.** (The draft's "20 µrad, 3,200×" compared shake with the noise floor; corrected.)
- **Real plates bow.** EN 10029 class N allows up to 36 mrad [snip]. That moves the reflected patch 2α·d_s = **72 mm per metre of d_s** (22 mm at 0.3 m, against a 10 mm margin) [calc].
  - A whole plate therefore walks off the screen. PMD needs each patch seated on a datum, or a tilt-search step per patch.
  - Tilt budget versus plate bow is a station-design requirement.
- **Finish** (gloss-lobe widths θ_s are all [assumed]; the rule is p ≥ 5.3·θ_s·d_s for 50 % contrast):
  - bright cold-rolled, BA, polished or painted: good;
  - 2B: to ~0.6 m;
  - brushed across the lay: ~0.2 m with a 24" screen;
  - **mill-scale hot-rolled (the spec default) gives no usable visible fringes**; it would need LWIR (https://eprints.soton.ac.uk/428285/1/IR_deflectometry_R1_v3.pdf).
- **Grading verdict:** PMD grades bright and 2B finishes only on a fixed, shrouded station with d_s ≈ 0.3-0.6 m, a datum, and calibration that is not yet done.

Published anchors:
- Micro-Epsilon reflectCONTROL ±1 µm, 170×160 mm [snip] (https://www.micro-epsilon.com/2d-3d-measurement/reflectcontrol/?sLang=en);
- ISRA SpecGAGE3D, 70-100 µm defects [snip] (https://www.isravision.com/en-en/products/specgage3d);
- smartphone PMD, 50 nm rms (lab) [snip] (https://3dim.northwestern.edu/project/mobilepmd).

### 3.2 Fringe projection (FPP): diffuse plate (mill scale, matte galvanised)

- σ_z = p·σ_φ/(2π·tanθ). One fringe is p/tanθ of height [calc].
- Projector light on the plate falls as 1/D².
- Throw ratio of the DLP2000 EVM: TI documents give 1.6 or 2.2, and users measured ~1.7. **All three are snippet-only** (https://ti.com/lit/pdf/dlpu089 ; https://e2e.ti.com/support/dlp-products-group/dlp/f/dlp-products-forum/1384405/dlpdlcr2000evm-dlpdlcr2000evm-throw-ratio).

Corrected envelope [calc, verify_simulation_C3_source/fpp_t*.py]. Setup: HQ camera, 100 mm plate, ρ = 0.2 [assumed], green fraction 0.6 [assumed], θ = 20°, bandpass fitted, 20 lm. Each cell gives the smallest dent at SNR 5, the implied σ = value/5, and the verdict.

| Standoff | Dark | 500 lx office | 10 klx shade | 100 klx sun |
|---|---|---|---|---|
| 0.3 m | 10-60 µm (σ 2-12): **grades** | **grades** | 35-60 µm (σ 7-12): **grades** | Single capture fails at TR 1.7. With TR 2.2 [snip]: 110 µm (σ 22), **grades marginally**. Averaged 3-11 captures: **unverified** |
| 1 m | 54-95 µm (σ 11-19): **grades** | TR 1.7: 230-280 µm (σ 46-56), **fails**. TR 2.2: 110 µm, **grades marginally** | **fails** | **fails** (σ_φ ≈ 8 rad) |
| 3 m | 1.1-1.7 mm: **fails** | fails | fails | fails |

- **Every cell rests on a snippet-only throw ratio and on assumed ρ and green fraction.** Measure throw and image lux (E5) before using any of it.
- **The 100 lm DLP230NP EVM** reaches 0.05 mm/px at 1 m in dark and office light **only on paper**. Its catch:
  - it needs an F/1.4 lens that is not on the parts list;
  - its 100 lm rating is snippet-only;
  - it costs ≈ 3.3× the parts cap;
  - TI supports it on the Pi 4B only.
- Daylight is the real limit: at 3.6 klx ambient is ~18× the fringe signal [calc].
- **Gamma:** with 3-step patterns, projector gamma 2.2 gives 0.19-0.21 rad error, against 0.006-0.008 rad with 4-step [calc]. Use 4-step or more.
- Commercial cross-check: Zivid 2+ M130, 210 µm at 1.3 m [snip] (https://www.zivid.com/compare-and-select-3d). That is a structured-light product an order short of grading at 1 m, consistent with the table.

### 3.3 Single-shot fringe methods (FTP, cross or colour-multiplexed fringes, single-shot PMD)

These answer the motion objection: one 1-10 ms exposure freezes the hand.
- Fourier-transform profilometry (FTP) and similar methods band-limit the phase. That costs about **2-3× in lateral resolution**, because the carrier must sit about 3 fringe periods wide [calc, rule of thumb]. They also fail at dent edges and discontinuities (spectral leakage).
- They do **nothing for ambient shot noise**. Using one frame instead of N steps gives roughly **√2× worse σ_φ** than a 4-step sequence at equal per-frame exposure [calc, rough]. At r = 10 that is ~0.36 rad against 0.18 rad.
- Colour multiplexing breaks on coloured rust and stains, and Bayer crosstalk adds error.
- **Is handheld single-shot still ruled out? Yes, but for the right reasons:**
  - FPP: the lighting envelope in 3.2 does not grade beyond 0.3 m in office light or above 10 klx, and a single shot only makes it worse.
  - PMD: the screen must be near the plate at the mirror angle, which is incompatible with a handheld head, and the slope range is limited.
  - Both: the 18 Nov hood-only rule.
- So **"lighting and slope range", not "motion", is the limiting reason.**

### 3.4 Green line laser triangulation: diffuse plate, detects but does not grade

Speckle sets the noise: δz ≈ C·λ/(2π·sin u·sinθ), with θ = 30° and λ = 520 nm [calc].
- At 1 m with a 16 mm lens, it is **≈ 40 µm at F/2 up to ≈ 230 µm at F/11** for speckle contrast C = 1.
- Across the three sibling investigations the range at 1 m was **≈ 40-270 µm**, depending on aperture and C.
- Published anchors are ~2× worse than the optimistic end:
  - UCL/Tata: 97 µm at 1 m on as-cast steel [pub] (https://discovery.ucl.ac.uk/id/eprint/10028424/1/Muller_PID4194273.pdf);
  - Micro-Epsilon optoNCDT 1760: ~100 µm at 1-2 m [snip, from the workflow summary].
- Even the best case misses the criterion: ≈ 28 µm per profile at 0.3 m.
- Averaging several profiles helps only if the speckle decorrelates. The draft's 8-profile figures are dropped: they are **best case [assumed]**.
- **Verdict:**
  - detects 0.2 mm dents at 0.3-1 m on diffuse finishes; **does not grade**;
  - needs scanning motion;
  - fails on specular finishes;
  - beating the sun at 1-3 m needs 5-50 mW (Class 3R-3B [memory]), unacceptable around specular steel;
  - the Phase-1 build plan §1.3 already excludes a laser line (docs/Phase1_build_plan.md:154).

### 3.5 Inside the existing hood: sectors, not a new colour

- Inside the hood, ambient is already negligible: r ≤ 0.021 (r = ambient/LED signal) [calc, twin].
- Narrowband green adds only ~+3 % SNR (82.3 → 84.5) [calc, brief-fit summary]. It also removes the rust/stain colour cue and would invalidate the white-light YOLO.
- It does not fix open mode, which misses silently even at 0 lux (C0o 4/30, reports/lighting_matrix.md).
- **The depth gain comes from regrouping the existing 18 white LEDs into sectors** [calc, twin]:
  - PINN dent error falls **0.053 → 0.034 → 0.021 mm** (L/R → 3 → 6 sectors);
  - the 90 % interval at 0.2 mm narrows **94 → 50 → 28 µm**.
  - These results are taken to assume the unchanged 18-LED ring (twin.yaml `n_leds 18`), regrouped. The brief-fit source file was not available, so confirm before quoting.
  - Even 6 sectors is 28 µm against the 25 µm criterion: marginal.

---

## 4. "Irrespective of lighting" and "irrespective of distance"

### 4.1 Distance: the camera can be far; the illuminator cannot

- **Camera sampling is solved by focal length** [calc]:
  - The AI Camera at 1-3 m samples 0.33-0.98 mm/px.
  - The HQ camera + 50 mm lens gives **0.031 mm/px over 126×94 mm at 1 m, and 0.093 mm/px over 377×283 mm at 3 m**.
  - The Arducam 50 mm C-mount lens is **₹5,959 incl. GST** [snip] (https://zbotic.in/product/arducam-c-mount-lens-for-raspberry-pi-high-quality-camera-50mm-focal-length-with-manual-focus-and-adjustable-aperture/), inside the ₹10k rule.
  - Depth of field shrinks to a few mm at 1 m (section 3.0), so the plate must be presented at a known height.
- **The illumination signal comes from where the lights are, not where the camera is.**
  - Dark-field and photometric stereo need grazing light (8-12° in the hood).
  - Each hood LED is ~87-129 mm from the plate [calc from h = 18 mm].
  - Moving the same LED to 1 m cuts in-band irradiance **60-130×**; at 3 m, **540-1,200×** (1/D²). Ambient on the plate does not change [calc].
  - At 1 m, a 10° grazing angle needs the lamp 174 mm high and 985 mm to the side.
  - So lighting invariance is lost when the lamps move away, not when the camera does.
- **This is how industrial strip-inspection stations work:** an overhead camera with a long lens, and line lights or strobes close to the strip, usually shrouded. That is general industry practice; no specific product was verified in this run.
- **The honest form of "a few metres" for Ayas:**
  - a fixed overhead station (telephoto camera at 1-3 m, grazing strobes or a shroud near the plate) for detection and preview;
  - depth and verdicts still from the hood.

### 4.2 What each tool fixes

| Tool | Fixes | Does not fix |
|---|---|---|
| Phase shifting / on-off subtraction | The mean ambient offset, exactly | Ambient **shot noise**, saturation, flicker. σ_φ = √(2/N)·√(A_L(1+r)+σ_r²)/(γ·A_L) |
| Green bandpass filter | On a mono IMX296: net ~5-6× with a green LED; 22-68× with a 520 nm laser [calc] | On the colour Pi cameras the gain is only ~2.3-3×. It gives ~20 % worse noise in the dark. Interference filters shift 8-11 nm at 20° [calc] |
| Pulsed, overdriven LEDs gated by the camera's XTR trigger | r improves by G = min(T_frame/t_e, P_peak/P_cont); 3× assumed | Must be mains-locked against 100 Hz flicker. Exposure length alone never changes r |
| Shroud or hood | 1 % leakage keeps the PMD noise multiplier ≤ 1.13-1.23 at 100 klx [calc]. Twin hood flat from 0 lux to a 44,872 lux median | **Lighting-insensitive only in the twin; unmeasured on hardware.** Assumes a flat seal. Plate edges, bowed or curved plate, and the hood's own LED specular hotspots on bright finishes are untested |

Phase noise versus r, IMX296 (≈ 10.3 ke⁻ full well [snip], 4.8 e⁻ read noise), N = 4, γ = 0.5 [calc]:

| r = ambient/source | 0 | 1 | 10 | 100 |
|---|---|---|---|---|
| σ_φ (rad), single 5-frame burst | 0.019 | 0.035 | 0.18 | 1.6 |
| Bursts needed for 0.02 rad (time at 50 fps) | 1 (0.1 s) | 3 (0.3 s) | 77 (7.7 s) | 6,200 (10 min) |

Source power for r ≤ 1 on a matte target with a 25 nm filter [calc]:

| Ambient | 0.3 m | 1 m | 3 m |
|---|---|---|---|
| Office 500 lx | 0.02 W/sr | 0.24 W/sr | 2.2 W/sr |
| Overcast 10 klx | 0.5 W/sr | 5.8 W/sr | 52 W/sr |
| Sun 100 klx | 5.2 W/sr | 58 W/sr | 524 W/sr |

- One 0.4 W green LED in a 10° optic gives ~9.85 W/sr [calc].
- Sun at 3 m therefore needs ~53 LEDs continuous, or ~18 pulsed at 3×. That lights only a ~0.5 m spot.
- **Safety:**
  - Such an array needs an **IEC 62471 photobiological risk-group assessment** [memory: limits to be checked against the standard].
  - Any strobe visible to operators must run **outside 3-60 Hz** (photosensitive-seizure band [memory]) or be shrouded.

### 4.3 What nothing fixes

- **Sun glints on bright steel.** The reflected sun is ~1.6×10⁹ cd/m². It fills an IMX296 pixel 20-200× over at the ~30 µs minimum exposure [calc]. No subtraction recovers it.
- **Specular plate reflecting sky or luminaires:** r ≈ 6-19 at 3 m even with 18 pulsed LEDs [calc]. In the PMD simulation, direct sun leaves a no-data patch of 7-70 mm (mirror finish) or 45-452 mm (brushed).
- **A monitor as the PMD source outdoors:** r 3.3-63, fringe visibility 0.16-0.28× [calc].

### 4.4 Design targets, not results (each gated by a named test)

| Method | Target envelope (calc only) | Gate |
|---|---|---|
| Hood + 5-frame burst + quality gate (existing) | Twin: 0 lux to a 44,872 lux median, 0/30 retakes, \|Δ\| 2.7-2.8 DN; strict FAIL by 2 conservative captures | Real-hood lighting matrix (Phase 2 step 1) |
| PMD station, shrouded | Dark room to window daylight; grading only with d_s ≤ ~0.6 m and a datum | E3 + E4 |
| FPP at 0.3 m | Grades ≤ 10 klx; sun unverified | E5 |
| FPP at 1 m | Grades in the dark only | E5 |
| Line laser | Detects only, on diffuse finishes; Class 2 to ~1 m. Specular glints and partly specular galvanised plate not covered | (not planned) |

### 4.5 Gate logic carried over

Any non-hood mode must:
1. **measure r** from the off frame and log it;
2. **count** clipped pixels and low-modulation pixels;
3. **return RETAKE** if r is above the method's limit, clipping is more than a few per cent of the ROI, or σ_φ > 0.5 rad;
4. **never return PASS.** It may say "anomaly / measure in hood".

This keeps the 18 Nov hood-only rule, which rests on 2 unsafe open-mode misses.

---

## 5. Pi architecture for Phase 2

Order:
1. sector-ring depth in the hood;
2. PMD station or hood-lid "verify" mode for bright finishes;
3. telephoto overhead preview station;
4. FPP or line laser: low priority.

Interferometry: never on the device.

### 5.1 BOM (INR; prices from snippets, recheck at checkout; ₹93.9/USD; DigiKey/Mouser before 18 % GST)

**(i) Sector ring: the existing 18 white LEDs, geometry unchanged**

| Item | Price | Status |
|---|---|---|
| 4-6 extra logic-level MOSFET channels + gate resistors, regrouping the ring into 3×6 or 6×3 | a few hundred rupees | assumed |
| GPIO | 3-6 lines | Pi 5 has spare |

This keeps the twin calibration and the lighting-matrix geometry. It is the only change recommended for the hood.

**(ii) Optional add-on: separate green depth channel (not a replacement; YOLO stays on white)**

| Item | Price | Status |
|---|---|---|
| Cree XP-E2 green (DigiKey India) | ₹164.01 each (₹155.81 at 10+) | snip |
| Mean Well NLDD-350H drivers | ~US$4.15 each | snip |
| DigiKey delivery below ₹7k | ₹1,200 | snip |
| 525/25 nm filter (Edmund #87-777) | ₹12,767, above the cap: skip | snip, corrected |

At 12 V each NLDD-350H drives 2 LEDs. An all-on draw of 14.6-17.3 W [calc] is marginal on the 33 W bank.

**(iii) Telephoto preview station:**
- HQ camera ₹4,974 [snip];
- Arducam 50 mm lens ₹5,959 [snip];
- each module is under ₹10k.
- The AI Camera and the DSI display use both Pi 5 MIPI ports, so the HQ camera means unplugging the AI Camera or moving the UI to HDMI.

**(iv) PMD station:**

| Item | Price |
|---|---|
| HQ camera | ₹4,974 |
| 16 mm lens | ₹5,229 |
| 24" monitor | ₹7,499-7,999 |
| Rails | ~₹1,720 |
| **Total** | **₹19,422**; ₹11,923 with an owned monitor [calc, snip prices] |

Zero-cost trial: the owned 5" DSI as the fringe screen, giving a 50×30 mm patch [calc].

**(v) FPP:**
- DLP230NP EVM ≈ ₹33.4k before GST (Pi 4B only);
- DLP2000 EVM ₹11,071 (no Pi driver).
- Both are over the rule.

Imports will not arrive before 23 Nov.

### 5.2 Capture and processing

- **Sector burst:** the existing 5-frame burst plus 3 or 6 sector frames, about +0.4 s per the brief-fit summary (the frames alone are 0.06-0.12 s at 50 fps [calc]; not timed on the Pi).
- **PMD:** 34 patterns (8-step x/y + unwrapping + off), exposure at multiples of 10 ms, 1.7-3.4 s on a fixed stand with the patch on a datum.
- **Decoding** [calc, M3 benchmark ×4, not benchmarked on a Pi]:
  - phase: arctangent + temporal unwrapping, ≈ 0.23 s for 12 frames at 1.6 MP;
  - integration: Frankot-Chellappa, then plane/bow removal and a local dent fit.
- **The Hailo keeps running the existing YOLO** on the white-light frame.
- **New JSON fields:**
  - `depth_method`, `depth_mm`, `depth_ci90_mm`;
  - `r_ambient`, `clip_frac`, `low_mod_frac`, `sigma_phi_rad`;
  - `gate`;
  - `mode` (`hood` | `station` | `standoff_preview`).
- **Verdicts come only from `mode == hood`.**
- **The LLM path is unchanged:** Llama 3.2 via ExecuTorch on the Arm CPU writes alerts from the record.

---

## 6. Fit with the brief and the plan

- **Brief (PS5).** Profilometry improves only **severity**, since EN 10163-2 grades by depth. PMD curvature maps add a little to anomaly detection. Alerts, logging and the Llama/ExecuTorch path are untouched. The organisers favour a simple end-to-end on-device pipeline (docs/research/2026-11-13_ps5_session_notes.md).
- **Real time.** Depth severity is **not real time on the Pi today:**
  - PINN depth is projected at 1-1.5 min per dent;
  - the 34-frame PMD decode has not been benchmarked.
  - Verdicts do not wait on depth (`depth.enabled false`).
  - **Phase 2 acceptance criterion (proposed):** depth result ≤ 2 s per capture on the Pi 5, measured, before depth enters verdicts.

**Phase 1 (due 23 Nov, 1 day after this note): build nothing new.**
- Finish the Pi run (tools/pi/phase1_run.sh), the report, the demo, and the hood-only and P1h decisions.
- In the report:
  - present the hood + burst + gate with its *twin-only* envelope;
  - present PINN depth as twin-only "estimated depth";
  - add one paragraph on "why not interferometry, why not profilometry at a distance", using sections 0, 2 and 4.1;
  - add the Phase 2 roadmap.

**Phase 2 (to 5 Feb 2027):**
1. Build the real hood and validate burst + gate on real plates.
2. Add the sector ring. Recalibrate the PINN on coupons gauged with a 0.01 mm dial gauge or an off-device profiler. Pass when σ ≤ 25 µm at 0.2 mm, with 0 lenient decisions.
3. Add PMD (station or lid) only if E2 beats PINN L/R on polished dents with 0 lenient decisions.
4. Add a telephoto overhead or 0.3-1 m standoff preview that never returns PASS.
5. Green only as an extra channel. FPP and the laser are low priority. Interferometry never on the device.

**Contradictions to state openly:**
- "Few metres" against hood-only verdicts. Resolved as preview only, unless the 18 Nov decision is reopened.
- PMD's fixed station against the handheld F1-wheel enclosure. The compromise to test is a hood-lid display, but a 7" covers only 80×47 mm.
- Build plan §1.3 laser ban: kept.
- Parts rule: the sector ring, the telephoto lens and the HQ camera fit; PMD fits only with owned parts; FPP breaks it.
- Earlier finding that a bandpass filter is ineffective with white light (docs/research/2026-11-14): still true. The filter gains apply only to green LED or laser sources.

---

## 7. Risks and next experiments

**Risks and open questions:**
- **Finish parameters are unmeasured:** θ_s, ρ and Ra of real coupons.
- **Snippet-only specs:** projector throw and lumens, sensor full wells, ASTM G173 values.
- **[memory] values:** IEC 62471 limits, laser classes and the flicker band.
- **PMD calibration is unmodelled.**
- **Pi runtimes are not benchmarked.**
- **Power on the 33 W bank, and MIPI port conflicts.**
- **Ground-truth access:** college stylus or optical profiler?
- **Which instrument you saw.** If it was focus variation or confocal rather than CSI/PSI, section 3.0 applies instead.

**Next experiments (none needed for 23 Nov):**
- **E1 (highest value):** sector-ring PINN on real coupons in the real hood. Pass: 90 % interval at 0.2 mm ≤ 50 µm, σ ≤ 25 µm, 0 lenient decisions.
- **E2:** Blender PMD twin with a 5"/7" lid screen at d_s 0.1-0.3 m, on polished and 2B plates, dents 0.05-0.4 mm. Go/no-go against PINN L/R.
- **E3:** owned 5" DSI tabletop PMD. Measure B, σ_φ, and σ_φ against lux.
- **E4:** C0-C6 lighting matrix on E3, shrouded and open. This replaces the PMD target envelope.
- **E5:** DLP2000 EVM throw and lux; one 4-step capture at 0.3 m and 1 m on mill scale.
- **E6 (cheap, new):** HQ + 50 mm at 1-3 m over a coupon. Compare detection against the hood with grazing strobes near the plate, then with ambient only.

### Traceability

- PMD and FPP grid: (local working files, not in the repo) (README.md, pmd_sim.py, fpp_sim.py, out/tables.md).
- Corrected FPP throw runs: (local working files, not in the repo) and `.../verify_simulation_C3_physics/c3.py`.
- This revision's calcs (criterion conversion, telephoto, 1/D² lamp distance, PMD motion and bow, speckle versus F-number, depth-from-focus, depth-camera ratios, burst timing): (local working files, not in the repo) (output in `revision_calcs_out.txt`).
- Interferometry-limits, deflectometry, ambient-rejection, parts and brief-fit numbers: **from the workflow summary, source files not available** in the scratch dir.
