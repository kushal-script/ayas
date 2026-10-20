Status: COMPLETE, last updated after: final verification fetches (Zygo NewView 7300 datasheet, Keyence VK-X3100, GelSight official page, 4D InSpec specs) and consistency fixes

# Handheld 3D scanners vs micrometre-level defect measurement on shiny metal

Research date: 20 Oct 2026. Context: student team (Bengaluru), handheld device, room light, Raspberry Pi 5 + Hailo-8 AI HAT+ or Arm Android, Llama via ExecuTorch as advisor, PoC by 23 Nov 2026. New requirement: capture detail at the **micrometre (µm) level**.

## Confidence tags and source keys
- **H**: I read it on the maker's or primary page, datasheet, patent or paper (WebFetch or PDF text extraction).
- **M**: I read it on a reseller, review or trade page, or it is a search snippet from a maker or credible source.
- **L**: single snippet from a marketplace or aggregator, an undated page, or sources that conflict.
- `[KEY]` points to the numbered source list at the end. Every fact has a key.
- "calc" means my own arithmetic from sourced numbers. "judgement" means my engineering opinion, not a sourced fact.
- Limits: WebSearch hit its per-turn limit near the end. Artec, Revopoint, Amazon.in, Robu.in and the RealSense support site block or JS-render pages, so those facts are snippet-only (M/L).

---

## 0. Bottom line for the µm requirement

Target used below: **≤10 µm depth and ≤20–50 µm lateral, handheld.**

1. **No free-motion handheld 3D scanner reaches ≤10 µm depth.** The best metrology-grade handheld laser scanners quote about 12–25 µm accuracy and 20–25 µm resolution:
   - Creaform HandySCAN BLACK Elite: 25 µm [CR1, CR2]. BLACK Elite Limited: 12 µm accuracy [CR7].
   - Scantech SIMSCAN Gen2: 20 µm [SC1].
   - Artec Point: 20 µm [AR10].
   - FreeScan UE Pro: 20 µm [SH1].

   These scanners meet the lateral target but not the depth target. Prosumer scanners (Revopoint, Creality, EinScan) quote 20–200 µm, often as "single-frame precision" at the closest range. Revopoint also says to use scanning spray on highly reflective objects [RV1, RV3, RV4].
2. **Industry measures µm-deep scratches and pits with small-field "touch-down" sensors, not free-motion scanners:**
   - GelSight Mobile (elastomer pad + photometric stereo): Z accuracy 1 µm + 4%, XY 3 µm + 0.2%, FOV 17.0 × 14.2 mm [I15, M].
   - 4D InSpec (dynamic interferometry): defects 2.5–2,540 µm deep, 4 µm vertical resolution, 8×8 mm FOV, 1 s per measurement [I19, M].
   - GE/Optics for Hire handheld fringe-projection gauge: 3.2 mm FOV, about 2.5 µm resolution [I22, M].
   - Benchtop and inline references: Keyence LJ-X8060 profiler (0.4 µm Z, 5 µm X pitch, 16 mm FOV) [I1, H]; confocal (Keyence VK-X3100, 12 nm σ at 50×) [I8, H]; white-light interferometry (Zygo NewView 7300: <0.1 nm vertical, 0.36–9.5 µm lateral, FOV 0.03–14 mm, a 430 kg system on an isolation table) [I9, H]; deflectometry (Micro-Epsilon reflectCONTROL, 10 nm deviations, 100 µm XY) [I10, H].
3. **Physics (calc):** lateral sampling ≈ FOV ÷ pixels. With a ~1.5 MP camera (Pi Global Shutter Camera, 3.45 µm pixels [P14]):
   - ≤10 µm/px needs FOV ≤ ~14 mm.
   - ≤20–50 µm lateral needs FOV ≤ ~30–70 mm.

   This is why all µm tools have fields of view of a few mm up to ~17 mm.
4. **What fits a Pi 5 in 4 weeks (judgement):** a touch-down **"micro-patch" sensor**:
   - 5–15 mm FOV, a light-tight hood and a fixed standoff, so room light and hand shake do not matter.
   - Photometric stereo from a ring of 4–8 LEDs, optionally through a GelSight-style gel pad for shiny metal, and/or phase-shift fringe projection.
   - Pi Global Shutter camera with a macro lens.
   - Hailo-8 (26 TOPS [P15]) segments defects in the normal and height maps.
   - Llama gets JSON measurements (depth µm, width µm, length mm).

   Free-motion scanning with real-time fusion is PC/GPU work. Vendor software asks for an RTX 4060/4070 [AR2], and scanners that process on board use NVIDIA Jetson TX2 or Orin NX [AR3, SH8].

---

## 1. Working principles of handheld 3D scanners

| Principle | How it works | Products using it | Typical handheld lateral / depth | µm capability (≤10 µm depth, ≤20–50 µm lateral, handheld) |
|---|---|---|---|---|
| **Structured light, white/blue LED** (fringe phase-shift, Gray code, coded patterns) | Projector acts as an "inverse camera"; stripe displacement is triangulated to 3D. Phase shifting uses "at least 3, typically about 10 exposures" and resolves about 1/10 of the stripe pitch [P1, H]. Multi-shot capture is sensitive to motion. | Artec Space Spider (blue LED, 7.5 fps) [AR1, H]; Artec Spider II (450 nm LED, 15 fps) [AR2, H]; Revopoint MINI 2 (blue light) [RV1, M]; EinScan H2 white-LED mode [SH2, M]; Artec Micro II (desktop) [AR8, M] | 50–200 µm resolution; 50–100 µm accuracy (Spider II 50/50 µm; Space Spider 100/50 µm; MINI 2 20 µm point spacing / 50 µm accuracy; H2 200/50 µm) | **No** when handheld. Desktop Artec Micro II claims 5 µm accuracy and 2 µm repeatability, but parts must fit 20×20×15 cm [AR8, M]. |
| **NIR VCSEL dot / speckle / IR structured light** | Invisible NIR pattern from VCSEL arrays, often through a diffractive optic (e.g. a 940 nm module with a 15,372-dot pattern [P11, M]), triangulated with one or more cameras. Usually eye-safe Class 1. | Artec Leo (VCSEL) [AR3, M]; EinScan H2 IR (3 VCSEL projectors) [SH3, M]; FreeScan Combo IR mode [SH4, M]; Revopoint POP 3 Plus / MIRACO / INSPIRE 2 [RV6, RV3, RV8, M]; Creality Otter, Raptor NIR mode [CL4, CL2, M] | 50–200 µm point spacing; 50–200 µm accuracy (Leo 200/100 µm; H2 IR 200/100 µm; Raptor NIR 100 µm; POP 3 Plus 50/80 µm; MIRACO 50/50 µm at 150 mm) | **No** |
| **Laser-line triangulation** (many crossing blue laser lines + 2 cameras + retro-reflective targets) | Laser planes cut the surface; two calibrated cameras triangulate points on each line in a single exposure. The scanner finds its own pose from retro-reflective targets ("self-positioning"), with no arm or tracker [CR10, CR11, M]. Working close to the part gives accuracy but a smaller FOV, so targets must be closer together [CR12, M]. | Creaform HandySCAN BLACK Elite (22 lines + 1) [CR2, H], HandySCAN MAX (38 lines) [CR3, H]; Scantech SIMSCAN Gen2 (11/17/27 crosses + 7/17 parallel + 1) [SC1, H], SIMSCAN-E (63 lines) [SC2, M], KSCAN-Magic (blue + IR) [SC5, M]; Artec Point (17 crosses + 7 + 1) [AR11, M]; FreeScan UE Pro (26 + 7 + 1) [SH1, H]; FreeScan Combo [SH4, M]; EinScan Libre (101 blue lines + 38 IR) [SH8, M]; Creality Raptor (7 parallel blue lines) [CL2, M]; Revopoint INSPIRE 2 (IR multi-line) [RV8, M] | Metrology grade: 20–40 µm resolution, 12–25 µm accuracy. Consumer: "20 µm at 100 mm" (Raptor) [CL2, M]. | **Partly.** Lateral 20–25 µm meets the target. Depth is ≥12–25 µm, so it misses ≤10 µm. |
| **Active stereo** | Two cameras match patches; a projector adds texture so plain surfaces can be matched. RealSense D435/D455 have a Class 1 IR laser projector; the D405 has none and relies on ambient light [RS2, RS3, M]. | RealSense D405 (passive stereo, 7–50 cm) [RS1, H]. Vendor "dual/quad-camera IR structured light" scanners (Revopoint; Creality Otter "four-lens stereo" [CL4, M]) are projector-assisted stereo (my classification, judgement). | D405: "<1 mm at 7 cm" [RS1, H]; ~100–160 µm/px at 7 cm (calc, §4C); depth error is mm-level | **No** |
| **Time-of-flight / LiDAR** | Pulsed (direct) or modulated-wave phase (indirect). Accuracy is "usually estimated at 1% of the measured distance". Uses a bandpass filter and a background frame. Prone to multipath and to errors on specular surfaces [P2, H]. | Phone LiDAR / ToF depth sensors (not researched further in this pass) | mm to cm | **No** |
| **Photogrammetry** | Many overlapping photos and coded targets, solved by bundle adjustment. Used on its own (OpenScan Pi-camera kits, vendor claims "up to 0.02 mm" Mini / "up to 0.01 mm" Classic on its own benchmark; processed locally or in OpenScan Cloud [P17, M]). Also used as a global reference to tighten volumetric accuracy of laser scanners: SIMSCAN + MSCAN-L15 0.015 mm + 0.012 mm/m [SC1, H]; FreeScan UE Pro built-in 0.02 + 0.015 mm/m [SH1, H]; MIRACO Plus kit 0.02 mm + 0.05 mm×L [RV4, M]. | as listed | Large-volume accuracy tool | **No** for micro-defects (it targets part-scale accuracy) |
| **Micro-surface methods** (photometric stereo, deflectometry, confocal, interferometry, fringe-projection borescopes) | See §5 | Keyence LumiTrax, GelSight, Micro-Epsilon reflectCONTROL, Keyence VK-X, Zygo, 4D InSpec, Waygate 3DPM | nm to µm | Yes, but on a small FOV and mostly not free-motion handheld (§5) |

---

## 2. How scanners work in room light

| Technique | What it does | Evidence / examples |
|---|---|---|
| Narrowband source + matching bandpass filter | Passes only the projector or laser wavelength | Blue-LED scanners "use a narrow band of blue LED light" and "filters to block out non-blue light"; white light is "easily washed out by ambient light" [SC3, M, vendor]. MidOpt BN450 narrow indigo bandpass is aimed at blue-laser triangulation under heavy ambient light [P7, M]. ToF cameras use an optical bandpass filter [P2, H]. Shining 3D: laser mode is less sensitive to ambient light [SH1, H]. |
| NIR at 940 nm | Less solar energy reaches the ground at 940 nm | Patent: switching projector and sensor to 940 nm improves sunlight rejection [P8, M]. |
| Pulsed or strobed source + short global-shutter exposure synced to the pulse | Collects ambient light only during the brief pulse | Global-shutter exposure "pulsed and synchronized" with projector pulses limits ambient light [P9, M]. Global shutter suits short illumination and moving objects [P10, M]. Patent claims 10–1000× sunlight rejection (claim only) [P8, M]. Pi Global Shutter Camera minimum exposure is 30 µs [P14, H]. |
| High peak brightness + very short exposures | Swamps ambient light | Vision Components VC nano 3D-Z: 450 nm Class 2 laser with "extremely short exposure times" works under up to 100,000 lux [P6, H]. |
| Background subtraction | Capture with the illumination off, then subtract | ToF practice [P2, H]. |
| Physical enclosure (contact devices) | Removes room light entirely | GelSight Mobile claims to work regardless of reflectivity, transparency or ambient light [I17, M]. The gel skin "supplies its own BRDF" [I13, H]. |
| Blue rather than red laser on shiny metal | Less speckle and less penetration into the surface | Blue gives about 2–3× lower noise on polished surfaces than red; a 405 nm laser plus a filter rejects glow from hot metal [P5, M, Micro-Epsilon]. |

---

## 3. How scanners cope with hand motion

| Technique | Evidence / examples |
|---|---|
| Single-shot capture per frame | Laser lines are captured in one exposure per frame. Photoneo "parallel structured light" keeps the laser on with a per-pixel shutter of about 10 µs, so it gets coded patterns from one frame [P12, M]. GelSight: "complete information is recorded in a single frame" [I13, H]. |
| High frame rate | Artec Point 120 fps [AR11, M]; Creality Raptor blue mode 60 fps [CL2, M]; Leo 22 fps fusion / 44 fps recording / 80 fps streaming [AR3, M]; Spider II 15 fps [AR2, H]; Space Spider 7.5 fps [AR1, H]; MINI 2 16 fps [RV1, M]; INSPIRE 2 up to 90 fps in laser mode with a GPU [RV8, M]; EinScan H2 20 fps scan-and-align, 55 fps camera [SH2, M] |
| Frame-to-model registration (ICP, geometry/texture tracking) | KinectFusion aligns each live depth frame to a global model with coarse-to-fine ICP on commodity graphics hardware. Tracking against the model limits drift [P4, M]. Artec: "hybrid geometry and texture tracking", "target-free" [AR2, H; AR3, M]. Revopoint: feature/marker (MINI 2) [RV1, M]; feature/marker/global marker (INSPIRE 2, MIRACO Plus) [RV8, RV4, M]. Creality: geometry/marker/texture [CL5, M]. |
| Reflective position markers | HandySCAN "self-positioning with dynamic referencing", no tripod or external tracker [CR1, H]. FreeScan UE Pro FAQ: scanning without markers "No" [SH1, H]. |
| IMU | Artec Leo "built-in 9 DoF" [AR4, M]; MIRACO 9-axis IMU [RV3, M] |
| Detect motion and re-capture | GE/Baker Hughes structured-light borescope patent: a motion-detection module re-captures if the probe moved between phase images [I27, H] |
| Very short total capture | GE handheld fringe gauge takes 3 phase-shifted images in <0.5 s [I22, M] |
| **Implication for µm** | µm-class handheld tools avoid free motion by resting on the part. GelSight's elastomer is pressed onto the surface [I13, H]; 4D InSpec is "vibration-immune" and takes "one second to measure" [I19, M]. |

---

## 4. Product comparison

Lateral = published resolution / point spacing; depth = published accuracy (or precision where stated). "µm cap." = meets ≤10 µm depth and ≤20–50 µm lateral when handheld.

### 4A. Metrology-grade handheld laser scanners

| Product | Light / principle | Lateral (µm) | Depth / accuracy (µm) | Working distance | Rate | Laser class | Price | Shiny metal / spray | SDK, Linux-ARM, Android | µm cap. |
|---|---|---|---|---|---|---|---|---|---|---|
| **Creaform HandySCAN 3D BLACK Elite** | 22 blue laser lines (+1) [CR2, H] | 25 (measurement resolution) [CR1, H]; mesh 100 [CR5, M] | 25 accuracy; volumetric 20 + 40 µm/m; step capability 25 at 0.3 m [CR2, H]. BLACK Elite Limited: 12 [CR7, M] | 200–450 mm [CR2, H] | up to 1,300,000 measurements/s [CR1, H] | 2M [CR2, H] | ~$65,000 (2023-era; Aniwaa lists it as discontinued) [CR6, CR7, L] | "Automatic shutter" for difficult, contrasted finishes [CR1, H]. No explicit no-spray claim found. | Creaform.OS / Metrology Suite [CR2, H]. No SDK info found. | Lateral yes; depth **no** (12–25) |
| HandySCAN MAX EVO / EVO Elite | 38 blue lines [CR3, H] | 40 (mesh 400) [CR3, H] | 150 / 75 [CR3, H] | 0.45–1.6 m / 0.3–2.5 m [CR3, H] | 2,250,000 measurements/s [CR3, H] | 2M [CR3, H] | not found | – | – | **No** (large parts) |
| **Scantech SIMSCAN Gen2** (22/30/42) | 11/17/27 blue crosses + 7 or 17 parallel + 1 deep-hole line [SC1, H] | up to 20 [SC1, H] | up to 20; volumetric 15 + 35 µm/m (12 µm/m with MSCAN-L15) [SC1, H] | 300 mm standoff, 550 mm DOF [SC1, H] | 2.02 / 4.58 / 5.8 M measurements/s [SC1, H] | Class II (eye-safe) [SC1, H] | ~$30,000 [SC4, L]; SIMSCAN-E ₹35,00,000 on IndiaMART [SC7, L] | "even black and reflective surfaces" [SC1, H]; vendor guide says blue laser scans shiny parts without spray [SC3, M] | none found | Lateral yes; depth **no** (20) |
| Scantech KSCAN-Magic (II) | blue + IR lasers [SC5, M] | – | 20; volumetric 10 + 15 µm/m with MSCAN-L15 (reseller) [SC5, M] | – | – | not found | – | – | – | **No** |
| **Artec Point** | blue laser: 17 crosses + 7 parallel + 1 [AR11, M] | up to 20 [AR10, M] | up to 20; volumetric 15 + 35 µm/m (15 µm/m with metrology kit) [AR10, M] | >300 mm, DOF 550 mm [AR11, M] | 120 fps [AR11, M] | Class II [AR11, M] | ~$29,000–29,900 [AR12, L] | "black, shiny" surfaces claimed [AR13, M] | Artec SDK ships .lib/.dll/.h (Windows-style; OS not stated) [AR16, M] | Lateral yes; depth **no** (20) |
| **Shining 3D FreeScan UE Pro** | 26 lines (fast) / 7 lines (detail) / 1 line [SH1, H] | point distance setting 10–10,000 [SH1, H] (a software spacing, not optical resolution; judgement) | 20 "with markers"; volumetric 20 + 15 µm/m with photogrammetry [SH1, H] | – | 1,850,000 points/s [SH1, H] | Class II per brochure [SH5, M] (some resellers say 2M) | ~$24,000 [SH6, L] | Laser mode better on reflective and dark surfaces [SH1, H] | not found | Lateral yes; depth **no** (20) |
| Shining 3D FreeScan Combo | 26 blue lines (13 crosses) + 7 + 1, plus IR VCSEL [SH4, M] | 50 laser / 100 IR [SH4, M] | volumetric 20 + 33 µm/m (laser), 50 + 100 µm/m (IR) [SH4, M] | – | – | not confirmed | from $16,499 [SH4, M] | Reviewer scanned a shiny handrest without spray, "not flawless" [SH4, M] | – | **No** |
| Shining 3D EinScan Libre (standalone) | 101 blue laser lines + IR modes [SH8, M] | 50 max resolution [SH8, M] | volumetric 40 + 60 µm/m [SH8, M] | IR up to 1.5 m [SH8, M] | – | – | $28,999 [SH8, M] | – | Runs on a Jetson Orin NX inside the scanner [SH8, M] | **No** |
| Creality CR-Scan Raptor | 7 parallel blue lines + NIR [CL2, M] | 20–2,000 (blue) [CL2, M] | 20 at 100 mm (blue); 100 (NIR) [CL2, M] | 150–400 mm (blue); 170–1000 mm (NIR) [CL2, M] | 60 fps blue / 20 fps NIR [CL2, M] | Class I [CL2, M] | $879 sale (regular $1,499) on 20 Oct 2026 [CL1, H]; ₹1,85,130 incl. GST and duty at Tanotis [CL7, L] | not found | Listing says Windows/macOS [CL2, M]; CrealityScan also runs on Android, with no Linux build [CL11, M]; no public API [CL10, M] | Claims 20/20; **no** for ≤10 µm |
| Creality Raptor Pro / RaptorX | Pro: 22 cross + 7 parallel + NIR; X: multi-line + NIR [CL13, CL12, L] | – | 20 (claim) [CL5, M] | – | Pro 660k points/s [CL13, L] | – | RaptorX $3,299 [CL12, L]; Raptor Pro ₹2,76,239 on Amazon.in [CL8, L] | – | – | **No** |

Observation (judgement, unverified): the Artec Point spec sheet (570 g, 550 mm DOF, 700×600 mm area, 17 crosses [AR11]) closely matches the Scantech SIMSCAN 30 Gen2 (570 g, 550 mm DOF, 700×600 mm area option, 17 crosses [SC1]).

### 4B. Prosumer structured-light / IR handheld scanners

| Product | Light / principle | Lateral (µm) | Depth / accuracy (µm) | Working distance | fps | Eye-safety class | Price | Shiny metal / spray | SDK, Linux-ARM, Android | µm cap. |
|---|---|---|---|---|---|---|---|---|---|---|
| Artec Space Spider (legacy) | blue LED [AR1, H] | 100 [AR1, H] | 50 (+0.3 mm/m) [AR1, H] | 0.2–0.3 m [AR1, H] | 7.5 [AR1, H] | LED | ~$24,800 (listing may be stale) [AR7, L]; reseller URL now shows Spider II [AR17, M] | – | – | **No** |
| **Artec Spider II** | 450 nm LED structured light [AR2, H] | up to 50 [AR2, H] | up to 50 (+0.3 mm/m) [AR2, H] | 0.19–0.3 m; FOV 128×104 to 171×152 mm [AR2, H] | 15 (datasheet; resellers say 30) [AR2, H] | LED | $29,700 [AR5, M] | – | Windows 10/11 x64 only; needs RTX 4060 minimum, RTX 4070 recommended [AR2, H] | **No** |
| Artec Leo | VCSEL [AR3, M] | 200 [AR3, M] | 100 (+0.3 mm/m) [AR3, M] | 0.35–1.2 m; FOV 244×142 to 838×488 mm [AR3, M] | 22 / 44 / 80 [AR3, M] | – | ~$34,800 [AR5, L] | – | Onboard Jetson TX2 (Arm Cortex-A57 + Maxwell GPU) [AR3, M] | **No** |
| Artec Micro II (desktop, not handheld) | RGB LED, four 13 MP cameras [AR8, M] | not published | 5 accuracy, 2 repeatability [AR8, M] | objects ≤20×20×15 cm | – | – | $22,500 [AR9, L] | – | – | Near µm, but **not handheld** |
| Shining 3D EinScan H2 | white LED + IR VCSEL (3 projectors) [SH2, SH3, M] | 200–3,000 [SH2, M] | 50 (white) / 100 (IR) [SH2, M] | best 470 mm; DOF 200–700 / 200–1500 mm [SH2, M] | 20 scan-and-align, 55 camera [SH2, M] | LED eye-safe; IR Class I [SH2, M] | £4,545 ex VAT [SH2, M]; ~$5,299 [SH7, L]; ₹7,90,000 at Exxjet (Bengaluru, IndiaMART) [SH9, L] | – | Windows 10 64-bit + GTX 1060, i7-8700, ≥16 GB [SH2, M] | **No** |
| **Revopoint MINI 2** | blue structured light, Class 1 [RV1, M] | 20 point spacing [RV1, M] | 20 precision / 50 accuracy (single frame) [RV1, M] | 120–250 mm; 52×64 mm at 120 mm [RV1, M] | 16 [RV1, M] | Class 1 | $912 [RV1, M] | "Use scanning spray for transparent, dark, or highly reflective objects" [RV1, M] | Windows/macOS/Android/iOS apps [RV1, M]; SDK by request (1-month trial; C++/Python; snippets say it only exposes sensor settings, not point clouds) [RV10, RV11, M] | Lateral claim yes; depth **no**; needs spray |
| Revopoint POP 3 Plus | IR, Class 1 [RV6, M] | 50 [RV6, M] | 40 precision / 80 accuracy [RV6, RV7, M] | 150–400 mm [RV6, M] | 18 [RV7, M] | Class 1 [RV6, M] | $557–619 [RV7, M]; ₹69,159 plus shipping and customs at Ubuy India [RV16, L] | – | – | **No** |
| Revopoint MIRACO (standalone) | IR structured light, Class 1 [RV3, M] | 50 at 150 mm [RV3, M] | 20 precision / 50 accuracy at 150 mm (200/500 at 600 mm) [RV3, M] | 100–1000 mm [RV3, M] | 15 [RV3, M] | Class 1 [RV3, M] | $1,500 retail [RV3, M]; Pro $1,999 [RV5, M]; Plus CAD$2,999 [RV4, M] | Spray for transparent or highly reflective objects [RV3, RV4, M] | Runs Android 12 on an 8-core SoC [RV3, M]; Plus: Arm A76 + Mali-G52 [RV4, M] | **No** |
| Revopoint INSPIRE 2 | IR structured light + IR multi-line laser [RV8, M] | – | 30 precision / 50 accuracy [RV8, M] | 200–400 mm [RV8, M] | 18 (SL) / up to 90 (laser, with GPU) [RV8, M] | – | $459 (review) [RV8, M]; from $549 [RV9, M] | – | Android 9+ / iOS 14+ (mobile mode needs Premium edition); Windows minimum RTX 3060 8 GB [RV8, M] | **No** |
| Creality CR-Scan Otter | NIR structured light, four-lens stereo [CL4, M] | 50–2,000 [CL4, M] (Creality blog: 100–2,000 [CL5]) | 20 at 60 mm [CL4, M] (Creality blog: 50 [CL5]); conflicting | 110–1000 mm [CL4, M] | 20 [CL4, M] | – | $899 [CL3, H] | – | Windows/macOS/Android app; no Linux [CL11, M] | **No** |
| Creality CR-Scan Ferret Pro | not stated in source [CL5] | 160 [CL5, M] | 100 [CL5, M] | 150–700 mm [CL5, M] | 30 [CL5, M] | – | ~$329.99 [CL6, L]; Ferret ₹36,996 incl. taxes at Moglix [CL9, L] | – | Android (wireless) and iPhone 11+ supported [CL5, M] | **No** |

**Revopoint SDK notes** (consumer scanners vs industrial cameras):
- A developer-forum topic titled "SDK Authorization Fails on Ubuntu 22.04" (Jul 2026) suggests some Linux use of the scanner SDK, but no ARM build is documented [RV12, M].
- Revopoint's separate industrial "Surface" 3D cameras have C++, OpenNI and Halcon SDKs [RV14, H].
- The PLUS toolkit supports Revopoint 3D cameras on Windows 64-bit and Ubuntu 18.04+ (SDK 2.4.0) and reads depth maps and point clouds [RV13, M].
- I found no Raspberry Pi or aarch64 support for any Revopoint product.

### 4C. Depth camera

| Product | Principle | Lateral | Depth | Range | fps | Price | SDK / platforms | µm cap. |
|---|---|---|---|---|---|---|---|---|
| **RealSense D405** | Passive stereo with global shutter. No IR projector [RS2, M], so it needs good light [RS3, M]. | "<1 mm at 7 cm" min object detection [RS1, H]; about 104 µm/px at 7 cm (calc: 2×70 mm×tan 43.5° = 133 mm over 1280 px) | ±2% at 50 cm [RS1, H]; roughly 0.2 mm at 7 cm (calc, assumes error grows with Z²; not published) | 7–50 cm [RS1, H] | up to 90 [RS1, H] | $325 [RS1, H]; ₹31,999–38,031 at Robu.in / Amazon.in [RS10, L] | Open SDK 2.0 for Windows, Linux, Android, macOS [RS1, H]. Pi is unofficial: libuvc backend, basic depth/colour streams only [RS4, RS5, M]. The libuvc install method is "not officially validated" [RS7, H]. The old official guide covers Pi 3 + D435 [RS6, H]. | **No** |

Corporate status (RealSense): spun out of Intel in July 2025 with a $50M Series A [RS9, M]. Cognex announced a deal to buy it for about $500M cash on 22 Sep 2026, expected to close in Q4 2026 [RS8, M]. Long-term SDK continuity is a risk (judgement).

### 4D. Availability in India (Bengaluru first)

| Brand | Channel | Confidence / source |
|---|---|---|
| Creaform | **AMETEK Instruments India Pvt Ltd, Division FARO CREAFORM**, Bhoruka Tech Park, 3rd & 4th Floor, Hoody ITPL Main Rd, Mahadevapura, Bengaluru 560048, +91 7411566960 | H [CR4] |
| Artec | **Altem Technologies**, Artec's Gold-certified partner in India, headquartered in Bangalore (Koramangala). Artec Point was launched in India with Altem (Apr 2025). | M [AR14, AR15, AR13] |
| Scantech | Avanitec (Pune), "national distributor", with staff in Bangalore [SC6, L]. Scantech India office in New Delhi [SC2, M]. IndiaMART resellers include RND Tech (Delhi), Pheonix 3D Metrology (Coimbatore), Expert 3D (Delhi) [SC7, L]. | L/M |
| Shining 3D | No official India distributor found. Exxjet Systems LLP (Yelahanka, Bengaluru) lists EinScan H2 at ₹7,90,000; authorisation not confirmed. | L [SH9] |
| Creality | Tanotis (Raptor ₹1,85,130 incl. GST and duty) [CL7]; Amazon.in (Raptor Pro ₹2,76,239) [CL8]; Moglix (Ferret ₹36,996) [CL9]; WOL3D India (Mumbai) lists Raptor on IndiaMART [CL14]. The official store also ships (Raptor $879, Otter $899) [CL1, CL3, H]. | L |
| Revopoint | Amazon.in lists MINI 2, INSPIRE and MIRACO Pro (prices not captured; page blocked) [RV15, M]; Ubuy India POP 3 Plus bundle ₹69,159 plus shipping and customs [RV16, L] | M/L |
| RealSense | Robu.in, Amazon.in, ₹31,999–38,031 | L [RS10] |

Practical tip (judgement): Creaform's Bengaluru office and Altem (Artec) are local. A demo scan of the team's defect samples could give a ~25 µm-class reference to validate the PoC.

---

## 5. What industry uses for the tiniest surface defects, and how it scales

| Method | Example and published numbers | Depth (µm) | Lateral (µm) | FOV | Handheld? | µm cap. |
|---|---|---|---|---|---|---|
| Handheld laser scanner + deviation map | Creaform HandySCAN + Pipecheck for pipeline corrosion and dents, assessed against ASME B31G. Rebuilds the original surface to get pit depth. Vendor claims >10× faster than a pit gauge [CR8, CR9, M]. | ~25 (scanner accuracy class [CR2]) | ~25 [CR1] | 100s of mm | yes | **No** (≥12–25 µm) |
| Laser profilometer | **Keyence LJ-X8060**: Z repeatability 0.4 µm, X repeatability 0.5 µm, X pitch 5 µm, X width 15–16 mm, 405 nm 10 mW Class 2M, 3200 points/profile [I1, H]. LJ-X8900: Z 10 µm, X pitch 225 µm [I2, M]. Controller up to 16 kHz [I3, M]. Repeatability measured with averaging (4096×) [I4, M]. | 0.4–10 | 5–225 | 16 mm to wider | **no** (fixed head + motion) | Yes, inline only |
| Embedded laser profiler | Vision Components VC nano 3D-Z: X about 40 µm, Z about 10 µm, 2 kHz, Zynq Arm SoC, up to 100,000 lux [P6, H] | ~10 | ~40 | 60 mm to >3 m standoff | no | Borderline |
| Confocal microscope | Keyence VK-X3100 (404 nm): display resolution 0.1 nm; height σ 100 nm at 10×, 40 nm at 20×, 12 nm at 50×; width 3σ 40 nm at 50×; accuracy 0.2 + L/100 µm; FOV 11–7,398 µm. Now discontinued, replaced by VK-X4100 [I8, H]. | 0.01–0.1 | <1 | 0.011–7.4 mm | no (bench) | Yes (bench) |
| Portable chromatic confocal | NASA in-situ rig with a STIL OP300VM pen for Shuttle windows: spot <25 µm, 1.5 µm depth-accuracy goal [I31, M] | 1.5 | <25 | scanned | portable, not handheld | Yes |
| White-light / coherence-scanning interferometry | Zygo NewView 7300 (2009 datasheet): vertical resolution <0.1 nm; RMS repeatability <0.01 nm; step-height accuracy ≤0.75%; lateral resolution 0.36–9.5 µm (objective-dependent); FOV 0.03–14 mm; works on specular and non-specular surfaces (1–100% reflectivity); 430 kg system, needs vibration isolation [I9, H] | <0.001 | 0.36–9.5 | 0.03–14 mm | no (bench) | Yes (bench) |
| Photometric stereo, inline | **Keyence LumiTrax**: multiple images under different lighting; produces normal and shape images; finds hairline scratches, shallow dents and pits on glossy metal [I5, H; I6, I7, M]. No µm numbers published. | detection only | – | camera FOV | no | Detection, not metrology |
| Deflectometry (specular surfaces) | **Micro-Epsilon reflectCONTROL** (phase-measuring deflectometry): deviations "from 10 nm", XY 100 µm, field 170×160 mm, up to 5 M points [I10, H]. Origin: Knauer, Kaminski, Häusler 2004 [I11, M]. Fraunhofer system for painted car panels [I12, M]. | 0.01 | 100 | 170×160 mm | no | Depth yes; lateral coarse; mirror-like surfaces only |
| **Gel + photometric stereo, handheld** | **GelSight Mobile Series 2**: XY 3 µm + 0.2%, Z 1 µm + 4% (1–50 µm), FOV 17.0×14.2 mm, 400 g [I15, M]. GelSight Max: Z to 0.3 µm [I16, M]. Principle: elastomer with a reflective skin plus RGB photometric stereo; the skin supplies its own BRDF, so it works on specular surfaces [I13, H]. Boeing/Siemens/GelSight/CMU aerospace work: "sensitive to 1 micrometer" [I14, M]. Maker's page: handheld, "micron-level measurements", "any material — metal, glass…", "including reflective, transparent and translucent surfaces under any lighting conditions". Listed uses include scratch and defect measurement, pit detection / corrosion, and fastener flushness. Quote only; no price [I33, H]. Host is a Windows 11 PC over USB-C [I17, M]. | ~1 (1 µm + 4%) | XY accuracy 3 µm + 0.2% (pixel pitch not published) | 17×14 mm | **yes (contact)** | **Yes** |
| **Dynamic interferometry, handheld** | **4D InSpec**: handheld, non-contact, vibration-immune "Dynamic Interferometry". Corrosion pitting, pits, scratches, nicks, dents, bumps, edge break, rivet depth from 2.5 µm to 2,540 µm deep. Vertical resolution 4 µm; FOV 8×8 mm; "one second to measure" [I19, M]. NCMS: 2 µm to 2.5 mm, "micron-level resolution" [I21, M]. Adopted by airlines and MROs since 2017 [I18, L]. | 2.5–4 | not stated | 8×8 mm | **yes** | **Yes** (depth) |
| **Handheld fringe projection (research prototype)** | GE Global Research + Optics for Hire (2010): Ronchi grating, piezo mirror, 3 phase-shifted images in <0.5 s; FOV 0.125 in (3.2 mm); resolution about 0.0001 in (~2.5 µm); 1360×1024 CCD, 6.45 µm pixels; host PC [I22, M] | ~2.5 | ~2.3 (calc: 3.18 mm / 1360 px) | 3.2 mm | **yes** | **Yes**; the closest analogue the team could build |
| Aircraft dent mapper | 8tree dentCHECK: Z uncertainty ±50 µm, XY ±1 mm [I23, L]; damage assessment time 92 → 12 min in an airline case [I24, L] | 50 | 1000 | – | yes | **No** |
| Videoscope 3D (turbine blades) | **Waygate Mentor Visual iQ(+) Real3D**: "on-demand 3D Phase and Stereo Measurement", 3D stitching, point cloud, depth and size of pitting, cracking, corrosion [I25, H]. Blog claims "sub-millimeter accuracy" [I26, H]. 3D phase = LED fringe sets through line gratings at the tip, phase-shift analysis, motion check (GE/Baker Hughes patent US8107083) [I27, H]. Sold as an add-on package [I30, M]. **Evident IPLEX**: stereo measurement tips (4 and 6 mm) and "3DAssist" single-optical-path 3D. No accuracy figures published. I found "3D phase measurement" only as Waygate's term, not Evident's [I28, H; I29, M]. | not published | – | probe FOV | yes (probe) | Unproven at µm |

**FOV vs resolution trade-off**
- **Lateral sampling = FOV ÷ pixels (calc).** The Pi Global Shutter Camera has 1.58 MP and 3.45 µm pixels [P14]. That is about 1451×1089 px at 4:3 (calc), a sensor about 5.0 mm wide. At magnification m, lateral sampling = 3.45 µm ÷ m:

  | Magnification | FOV width | µm/px |
  |---|---|---|
  | 1× | 5 mm | 3.45 |
  | 0.5× | 10 mm | 6.9 |
  | 0.25× | 20 mm | 13.8 |
  | 0.1× | 50 mm | 34.5 |
  | 0.033× | 150 mm | ~103 |

  To resolve a 20 µm-wide scratch you need ≤10 µm/px (Nyquist), so FOV ≤ ~14 mm (calc).
- Same sensor, different FOV: the Keyence LJ-X8060 (16 mm, 5 µm pitch, 0.4 µm Z) versus the LJ-X8900 (225 µm pitch, 10 µm Z) [I1, I2].
- Phase shifting resolves about 1/10 of the fringe pitch, and accuracy falls over large volumes [P1, H].
- Deflectometry partly escapes the trade-off: it measures slope, giving 10 nm Z sensitivity with 100 µm XY over 170×160 mm [I10, H].
- Industry scaling pattern (synthesis, judgement):
  1. Wide-FOV fast **detection**: photometric stereo or deflectometry inline, or a videoscope in the field.
  2. Small-FOV **µm measurement** at flagged spots: profilometer, confocal or interferometer in the lab; GelSight or 4D InSpec in the field.
  3. **Disposition** against limits (e.g. Pipecheck / ASME B31G).

---

## 6. Laser safety

| Class (IEC 60825-1 summary) | Meaning |
|---|---|
| 1 | Safe under reasonably foreseeable use, including viewing through optical instruments [I32, M] |
| 2 | Visible (400–700 nm), ≤1 mW CW; the blink reflex (~0.25 s) protects [I32, M] |
| 2M | Like Class 2, but hazardous if viewed through magnifying optics [I32, M] |

| Device | Class |
|---|---|
| Creaform HandySCAN BLACK Elite | 2M "eye safe" [CR2, H] |
| Creaform HandySCAN MAX | 2M [CR3, H] |
| Scantech SIMSCAN Gen2 | Class II (eye-safe) [SC1, H] |
| Scantech SIMSCAN-E | Class II [SC2, M] |
| Artec Point | Class II [AR11, M] |
| FreeScan UE Pro | Class II per brochure [SH5, M]; some resellers say 2M |
| EinScan H2 | LED eye-safe; IR Class I [SH2, M] |
| Creality Raptor | Class I [CL2, M] |
| Revopoint MINI 2 | Class 1 blue light; "avoid direct eye exposure for extended periods" [RV1, M] |
| Revopoint MIRACO / POP 3 Plus | Class 1 IR [RV3, RV6, M] |
| Keyence LJ-X8060 | 2M (405 nm, 10 mW) [I1, H] |
| VC nano 3D-Z | Class 2 (450 nm) [P6, H] |
| RealSense D435/D455 | Class 1 IR projector; D405 has none [RS2, M] |
| Artec Space Spider / Spider II | LED, not laser [AR1, AR2, H] |

Note (judgement): "Class II" on Chinese spec sheets almost certainly means IEC Class 2. For a DIY device, use LEDs or Class 1/2 line lasers and avoid 2M/3R in demos.

---

## 7. Assessment: buy, build on Pi 5 in 4 weeks, or PC/GPU only

### 7.1 Buy and integrate
- **RealSense D405** is the only product with an open SDK on Linux/ARM/Android [RS1]. Pi support is unofficial [RS4]. It is far from µm (about 100 µm/px, mm-level depth). Use it only for coarse localisation, if at all.
- **Prosumer scanners** (Revopoint, Creality, EinScan) have closed apps.
  - Android capture via a phone works for MINI 2, Ferret Pro and INSPIRE 2 Premium; MIRACO runs Android 12 itself [RV1, CL5, RV8, RV3].
  - You could export a mesh (STL/PLY/OBJ) and analyse it on the Pi offline.
  - They are 20–100 µm class and need spray on shiny metal [RV1], so they cannot meet µm. They could serve dents or gouges of roughly ≥0.1 mm (judgement).
  - No Linux build or public API for Creality [CL10, CL11]. The Revopoint SDK is a limited trial by request [RV10, RV11].
- **Metrology handheld laser scanners** cost from ~$24k to $65k, or tens of lakhs in India [SH6, SC4, CR6, SC7]. Their software needs Windows and a GPU, and they still miss ≤10 µm depth. Use only as a borrowed reference.
- **GelSight Mobile / 4D InSpec** are the right class of tool for µm. Prices were not found, and GelSight needs a Windows PC host [I17]. Treat them as benchmark references.

### 7.2 Build on Pi 5 + Hailo-8 in 4 weeks: touch-down "micro-patch" sensor (judgement unless tagged)
- **Core (recommended): photometric stereo micro-patch.**
  - Hardware: Pi Global Shutter Camera (IMX296, 1.58 MP, 3.45 µm pixels, ≥30 µs exposure, C/CS mount) [P14, H]; a macro lens at 0.5–1× for 3.5–7 µm/px over 5–10 mm (calc); 4–8 LEDs at a fixed elevation switched from GPIO; a 3D-printed light-tight hood with a fixed standoff.
  - Shiny metal breaks the Lambertian assumption [P3, H]. Fix it with a GelSight-style elastomer pad, whose skin supplies its own BRDF [I13, H]. Making a good gel skin is the main schedule risk.
  - Output: normal map → integrated height map → scratch and pit depth, width and length. Calibrate on known steps or spheres and on samples measured by a lab profilometer.
  - Expected depth: GelSight reaches ~1 µm [I15]. A first DIY version is more likely a few µm to ~10 µm (judgement).
- **Option B: phase-shift fringe micro-projection** (GE gauge analogue [I22]).
  - 3–4 phase images over a 3–10 mm FOV.
  - Projector: TI DLPDLCR230NPEVM (1080p, up to 100 lm, "Raspberry PI 4 compatibility out of the box", Python package; Pi 5 not stated [P16, H]). It needs macro projection optics. A Ronchi grating plus a small mechanical shift is the alternative.
  - Specular highlights can exceed the camera's dynamic range [P1, H]. Use HDR exposure brackets or the gel pad.
- **Option C: blue line-laser micro-profiler** (LJ-X analogue [I1]).
  - Class 1/2 405/450 nm line laser with the GS camera at a triangulation angle, slid along a rail over the defect.
  - Single-shot per frame. Few-µm to ~10 µm depth is theoretically possible with sub-pixel peak finding, after calibration (judgement).
- **Compute fit.**
  - Per-pixel least squares and phase decoding on 1.6 MP images is light work for the Pi 5 CPU (judgement).
  - Hailo-8 runs CNN segmentation and classification on the normal, curvature and height maps. The AI HAT+ comes in 26 TOPS (Hailo-8) and 13 TOPS (Hailo-8L) versions, "from $70" (per-variant prices not listed), for the Pi 5, integrated with the Pi camera stack [P15, H].
  - Llama (ExecuTorch) receives JSON such as `{type, depth_um, width_um, length_mm, location, confidence}` and gives disposition advice.

### 7.3 Only feasible with a PC/GPU, or not in 4 weeks
- **Free-motion handheld scanning with real-time ICP and volumetric fusion.** KinectFusion used GPU hardware [P4]. Vendor requirements: Spider II RTX 4060/4070 on Windows x64 [AR2]; EinScan H2 GTX 1060 [SH2]; INSPIRE 2 RTX 3060 for 90 fps laser mode [RV8]. Scanners that process on board use Jetson TX2 (Leo) [AR3] or Orin NX (EinScan Libre) [SH8].
- **µm metrology over a whole part:** needs a profilometer on a motion stage, a confocal microscope or a white-light interferometer [I1, I8, I9].
- **Photogrammetry reconstruction:** OpenScan processes on a PC or in its cloud [P17].
- **Training learned photometric stereo for non-Lambertian metal:** needs a PC/GPU. Running the trained model on Hailo may be possible (judgement).

### 7.4 Caveats
- µm depth on shiny metal while handheld is realistic only as a touch-down, small-FOV device. Free-motion handheld scanners stay at ≥12–25 µm.
- Scanning spray adds a coating of unknown thickness (not researched). At µm scale, avoid spray for measurement (judgement).
- Many specs are vendor "up to" numbers under ideal conditions. Single-frame precision is not the same as accuracy.

### Gaps (not verified; search budget exhausted)
- Revopoint, Amazon.in and Robu.in live prices.
- Official RealSense support on Pi 5.
- TI DLP EVM on Pi 5.
- GelSight and 4D InSpec prices.
- Waygate and Evident accuracy figures.
- Creaform, Artec and Shining 3D SDK platforms.
- ToF handheld products.
- Raspberry Pi AI Camera (IMX500) specs.

---

## Sources
(H/M/L = confidence basis for that source.)

**Creaform**
- CR1 (H) https://www.creaform3d.com/en/products/portable-3d-scanners/portable-3d-scanner-handyscan-3d
- CR2 (H) https://cdn.goengineer.com/handyscan-3d_black_series_brochure.pdf (official V4 brochure, distributor copy; text extracted)
- CR3 (H) https://www.creaform3d.com/en/products/portable-3d-scanners/handyscan-3d-max-series-for-large-objects/technical-specifications
- CR4 (H) https://www.creaform3d.com/en/about-us/offices-worldwide
- CR5 (M) https://www.goengineer.com/3d-scanners/creaform/handyscan
- CR6 (L) https://www.aniwaa.com/product/3d-scanners/creaform-handyscan-black-elite/
- CR7 (M) https://mobile.engineering.com/amp/27495.html
- CR8 (M) https://www.creaform3d.com/en/about-us/press-releases/creaform-releases-new-pipecheck-software-pipeline-integrity
- CR9 (M) https://www.vision-systems.com/cameras-accessories/article/16743402/3d-imaging-software-and-scanner-inspect-corroded-oil-and-gas-pipelines
- CR10 (M) https://patents.google.com/patent/US8082120
- CR11 (M) https://www.theengineer.co.uk/going-steady/
- CR12 (M) https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8140295

**Scantech**
- SC1 (H) https://www.3d-scantech.com/product/simscan-gen2-smart-handheld-3d-scanner/
- SC2 (M) https://www.3d-scantech.com/upload/other/20250711/b68a2bdbd809c25d2d635823430cd9c1.pdf
- SC3 (M) https://www.3d-scantech.com/blue-vs-white-light-vs-laser-scanning-the-ultimate-guide/
- SC4 (L) https://www.aniwaa.com/product/3d-scanners/scantech-simscan/
- SC5 (M) https://top3dshop.com/blog/scantech-kscan-magic-hybrid-3d-scanners-review
- SC6 (L) https://www.slideshare.net/slideshow/avanitecpunepdf/251752925
- SC7 (L) https://m.indiamart.com/impcat/3d-laser-scanners.html

**Artec**
- AR1 (H) https://www.scitepress.org/PublishedPapers/2022/108869/pdf/index.html
- AR2 (H) https://cdn.goengineer.com/1.1_spider_ii_us_letter.pdf
- AR3 (M) https://www.goengineer.com/3d-scanners/artec/leo
- AR4 (M) https://www.scheimpflug.com/product-page/artec-leo-scanner
- AR5 (M/L) https://www.matterhackers.com/store/l/artec-spider-ii-3d-scanner
- AR7 (L) https://matterhackers.com/store/l/artec-space-spider-3d-scanner
- AR8 (M) https://www.artec3d.com/news/artec-micro-II-released
- AR9 (L) https://www.matterhackers.com/store/l/artec-micro-ii-3d-scanner/sk/MGYMWVD7
- AR10 (M) https://www.artec3d.com/portable-3d-scanners/laser-point
- AR11 (M) https://www.goengineer.com/3d-scanners/artec/point
- AR12 (L) https://all3dp.com/4/artec-point-3d-scanner/ and https://matterhackers.com/store/l/artec-point
- AR13 (M) https://manufactur3dmag.com/altem-artec-point-in-india-3d-scanning/
- AR14 (M) https://artec3D.com/reseller/altem-technologies-p-ltd
- AR15 (L) https://my.motherbase.ai/startup/93027-altem-technologies-pvt-ltd
- AR16 (M) https://docs.artec3d.com/sdk/1.1
- AR17 (M) https://hawkridgesys.com/artec-space-spider-3d-scanner

**Shining 3D**
- SH1 (H) https://www.shining3d.com/solutions/freescan-ue-pro
- SH2 (M) https://www.createeducation.com/?p=88618
- SH3 (M) https://3dprintingindustry.com/news/shining-3d-launches-next-generation-einscan-h2-3d-scanner-technical-specifications-and-pricing-223625/
- SH4 (M) https://www.aniwaa.com/review/3d-scanners/shining-3d-freescan-combo-review-small-but-mighty/
- SH5 (M) https://encdn.shining3d.com/2022/03/【EN】FreeScan-UE-Pro.pdf
- SH6 (L) https://www.aniwaa.com/product/3d-scanners/shining-3d-freescan-ue-pro/
- SH7 (L) https://www.aniwaa.com/product/3d-scanners/shining-3d-einscan-h2/
- SH8 (M) https://matterhackers.com/store/l/einscan-libre-laser-3d-scanner-wireless-markerless-full-color/sk/M10LPUPH and https://top3dshop.com/product/einscan-libre-3d-scanner
- SH9 (L) https://m.indiamart.com/exxjetsystems/3d-scanner.html

**Revopoint**
- RV1 (M) https://3dprintingindustry.com/news/a-closer-look-at-revopoints-new-mini-2-3d-scanner-technical-specifications-and-pricing-227920/
- RV3 (M) https://3dprintingindustry.com/?p=225376
- RV4 (M) https://spool3d.ca/revopoint-miraco-plus-standalone-3d-scanner-with-metrology/
- RV5 (M) https://www.revopoint3d.com/products/revopoint-miraco-pro-3d-scanner
- RV6 (M) https://spool3d.ca/revopoint-pop-3-plus-handheld-3d-scanner/
- RV7 (M) https://global.revopoint3d.com/products/portable-3d-scanner-pop3
- RV8 (M) https://3dwithus.com/?p=36366
- RV9 (M) https://www.revopoint3d.com/products/infrared-laser-3d-scanner-inspire2
- RV10 (M) https://forum.revopoint3d.com/t/is-there-any-sdk-or-api/2410/36
- RV11 (M) https://forum.revopoint3d.com/t/is-there-any-sdk-or-api/2410?page=2
- RV12 (M) https://forum.revopoint3d.com/c/developer/13
- RV13 (M) https://pluslib.readthedocs.io/en/latest/devices/DeviceRevopointSurface.html
- RV14 (H) https://3dcamera.revopoint3d.com/html/Areascan2/index.html
- RV15 (M) Amazon.in listings:
  - https://www.amazon.in/Revopoint-Printing-Handheld-Precision-Industrial/dp/B0CNYRRMRL
  - https://www.amazon.in/Revopoint-3D-Printing-Handheld-Portable/dp/B0CQPM8S5N
  - https://www.amazon.in/Revopoint-MIRACO-Printing-Handheld-Precision/dp/B0CNXGQHDD
- RV16 (L) https://www.ubuy.co.in/product/TMC22EIMA-pop3-plus-3d-scanner-3d-scanning-spray-bundle-handheld-3d-printer-scanner-0-04mm-precision-full-color-scan-for-medium-object-support-for

**Creality**
- CL1 (H for price) https://store.creality.com/products/cr-scan-raptor-3d-scanner
- CL2 (M) https://www.pbtech.com/pacific/product/PTRCRL0026/Creality-CR-Scan-Raptor-3D-Scanner-Blue-7-Line-Las
- CL3 (H for price) https://store.creality.com/products/cr-scan-otter-3d-scanner
- CL4 (M) https://www.pbtech.com/au/product/PTRCRL0025/Creality-CR-Scan-Otter-3D-Scanner-Accuracy-up-to-0
- CL5 (M) https://store.creality.com/uk/blog/creality-3d-scanners-technical-parameters
- CL6 (L) https://microcenter.com/category/4294828446,4294819530/3d-printer-tools
- CL7 (L) https://www.tanotis.com/collections/vendors/products/creality-cr-scan-raptor-portable-3d-scanner
- CL8 (L) https://www.pricebefore.com/creality-3d-scanner-cr-scan-raptor-p1037970.html
- CL9 (L) https://www.moglix.com/creality-cr-scan-3d-scanner-ferret-for-3d-printer/mp/msno5wnqz0xw51
- CL10 (M) https://forum.creality.com/t/automating-cr-otter-scanner-api-or-open-source-solutions/16450
- CL11 (M) https://blog.0x17.de/tags/scanning/
- CL12 (L) https://stemfinity.com/collections/middle-school/products/cr-scan-raptor-x-3d-scanner
- CL13 (L) https://makerlab.ph/products/creality-raptor-pro-scanner
- CL14 (L) https://m.indiamart.com/impcat/3d-white-light-scanner.html

**RealSense**
- RS1 (H) https://realsenseai.com/products/stereo-depth-camera-d405/
- RS2 (M) https://support.realsenseai.com/hc/en-us/community/posts/10327342028435/comments/10327801573779
- RS3 (M) https://support.realsenseai.com/hc/en-us/community/posts/22966312714899-D405-as-a-stereo-camera
- RS4 (M) https://support.realsenseai.com/hc/en-us/community/posts/24610226505363/comments/24630176398995
- RS5 (M) https://support.realsenseai.com/hc/en-us/community/posts/28433133128979/comments/28433952591123
- RS6 (H) https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_raspbian.md
- RS7 (H) https://github.com/IntelRealSense/librealsense/blob/master/doc/libuvc_installation.md
- RS8 (M) https://www.calcalistech.com/ctechnews/article/s1tqir19fl
- RS9 (M) https://pulse2.com/realsense-50-million-raised-and-spin-out-from-intel-complete/amp/
- RS10 (L) https://robu.in/product/intel-realsense-depth-camera-d405/ and https://www.amazon.in/Intel-RealSense-D405-Webcam-Computer/dp/B09JBBHVTY

**Principles, room light, motion, Pi hardware**
- P1 (H) https://en.wikipedia.org/wiki/Structured-light_3D_scanner
- P2 (H) https://en.wikipedia.org/wiki/Time-of-flight_camera
- P3 (H) https://en.wikipedia.org/wiki/Photometric_stereo
- P4 (M) https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-dense-surface-mapping-tracking/
- P5 (M) https://newequipment.com/research-and-development/article/21128808/a-comparison-of-blue-and-red-laser-triangulation-sensors
- P6 (H) https://www.vision-components.com/en/news/vc-nano-3d-z-laser-profilers-ambient-light-suppression-technology/
- P7 (M) https://www.vision-systems.com/directory/manufacturers/optics/filters/press-release/14196367/midwest-optical-systems-midopt-new-midopt-bp450-indigo-bandpass-filter-and-bn450-narrow-indigo-bandpass-filter
- P8 (M) https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11330199
- P9 (M) https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11774769
- P10 (M) https://data.epo.org/publication-server/rest/v1.2/patents/EP3474541NWA1/document.html
- P11 (M) https://lasermate.com/wp-content/uploads/2026/02/vdoe940cob15k-ds-rev1.pdf
- P12 (M) https://www.photoneo.com/welcome-to-the-future-of-3d-vision/
- P14 (H) https://www.raspberrypi.com/products/raspberry-pi-global-shutter-camera/
- P15 (H) https://www.raspberrypi.com/products/ai-hat/
- P16 (H) https://www.ti.com/tool/DLPDLCR230NPEVM
- P17 (M) https://openscan.eu/

**Industrial micro-defect practice**
- I1 (H) https://keyence.com/products/measure/laser-2d/lj-x8000/models/lj-x8060
- I2 (M) https://keyence.com.my/products/measure/laser-2d/lj-x8000/models/lj-x8900/
- I3 (M) https://www.keyence.com/products/measure/laser-2d/lj-x8000/specs/
- I4 (M) https://keyence.ca/products/measure/laser-2d/lj-x8000/models/lj-x8400
- I5 (H) https://www.keyence.com/products/vision/resources/vision-resources/detecting-hairline-scratches-and-stains-on-glossy-surfaces-with-lumitrax.jsp
- I6 (M) https://www.keyence.com/landing/lpc/yt_linescan_specular_reflection.jsp
- I7 (M) https://patents.google.com/patent/US10489900
- I8 (H) https://www.keyence.eu/products/microscope/macroscope/vk-x3000/models/vk-x3100
- I9 (H) https://www.lambdaphoto.co.uk/pdfs/Zygo/LAMBDA_Zygo_NewView_7300_specifications.pdf (Zygo NewView 7300 datasheet SS-0039 06/09; text extracted)
- I10 (H) https://www.micro-epsilon.com/2d-3d-measurement/reflectcontrol/?sLang=en
- I11 (M) https://spiedigitallibrary.org/conference-proceedings-of-spie/5457/1/Phase-measuring-deflectometry--a-new-approach-to-measure-specular/10.1117/12.545704.short
- I12 (M) https://www.vision-systems.com/factory/article/16736059/measurementgauging-phase-measurement-technique-detects-defects-in-specular-surfaces
- I13 (H) http://people.csail.mit.edu/kimo/publications/retrographic/
- I14 (M) https://www.cmsc.org/inspection-of-surface-discontinuities-on-aerostructures-and-other-flight-critical-components
- I15 (M) https://www.ndtec.net/fileadmin/user_upload/produkte/SellSheets/GelSight_Series2.pdf
- I16 (M) https://www.anzupartners.com/2023/05/03/introducing-gelsight-max-the-highest-resolution-mobile-surface-characterization-tool-on-the-market
- I17 (M) https://ncms.org/wp-content/uploads/Quad-Chart_GelSight_FRC-East_6.11.24.pdf
- I18 (L) https://www.avm-mag.com/?p=26759 (adoption since 2017; snippet)
- I19 (M) https://www.newequipment.com/product-directory/controls-instrumentation/inspection-equipment/product/55104851/68491-surface-gauge (fetched); also https://www.photonics.com/pr60558/4D_InSpec_Surface_Gauge (snippet)
- I21 (M) https://ncms.org/?p=49851 (fetched)
- I33 (H) https://www.gelsight.com/products/gelsight-mobile/
- I22 (M) https://vision-systems.com/cameras-accessories/article/16749722/defect-detection
- I23 (L) https://www.directindustry.com/prod/8tree/product-244259-2483024.html
- I24 (L) https://gbp.com.sg/stories/8tree-scores-with-dentcheck2-dentcheck-as-a-service/
- I25 (H) https://www.ndtproducts.ca/wp-content/uploads/Mentor-ViQ-Brochure.pdf
- I26 (H) https://www.bakerhughes.com/waygate-technologies/blog/how-3d-measurement-capabilities-are-advancing-remote-visual-inspection
- I27 (H) https://patents.google.com/patent/US8107083
- I28 (H) https://ims.evidentscientific.com/products/video-borescopes/iplex-one
- I29 (M) https://ims.evidentscientific.com/de/landing/videoscopes/iplex-nx
- I30 (M) https://www.instrumart.com/products/181787
- I31 (M) https://ntrs.nasa.gov/api/citations/20120006139/downloads/20120006139.pdf
- I32 (M) https://www.laserax.com/laser-safety-classification ; https://www.sciencebuddies.org/science-fair-projects/references/laser-safety-guide
