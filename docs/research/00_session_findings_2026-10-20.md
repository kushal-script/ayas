# Session findings and open research (saved 20 Oct 2026)

This file saves the state of the design discussion, so nothing is lost if the session stops. Research results are in this folder:

| File | Status |
|---|---|
| `01_handheld_3d_scanners.md` | **COMPLETE** |
| `02_gelsight_tactile_3d.md` | **COMPLETE.** Bottom line below |
| `03_handheld_inspection_architecture.md` | **COMPLETE** |
| `04_usb_camera_options.md` | COMPLETE (camera shortlist, Bengaluru sourcing, pitfalls) |
| `05_recommendation_handheld_um.md` | **RECOMMENDATION (21 Oct):** TriNetra-Handheld, find + touch-measure (+ optional gel tip) |
| `archive/` | earlier laser / microscope research (10 Oct) |

Related plans in `docs/`:
- `TriNetra_plan.md`: fixed 3-camera rig + 3D + red-dye
- `TriNetra_FPI.md`: fluorescent penetrant
- `TriNetra_parts.md`
- `SukshmaScope_plan.md`, `BeamTriage_plan.md`: older

## Where we are

**Challenge:** Bharat AI-SoC 2026-27, Manufacturing theme (PS5). **Open Qualifying is due Mon 23 Nov 2026** (report + source code + demo).
- Brief: detect surface defects, cracks and anomalies; classify type and severity; alert with recommendations; log; real time at the edge, no cloud.
- Llama via ExecuTorch on the Arm CPU is mandatory.
- If Hailo is used, vision runs via HailoRT, documented as a separate pipeline.

**User's latest requirements (20 Oct):**
1. **Handheld** product.
2. Works in **normal room light** (no enclosure).
3. Must capture details at the **micrometre level** ("tiniest of the tiniest").
4. Feasible as a product and scalable to industry.
5. Not restricted to an RGB camera + LEDs.
6. Hardware: Pi 5 (8 GB) + AI HAT+ 26 TOPS; parts genuine and buyable in Bengaluru.

The user said the box/rig ideas "don't work", and asked how handheld 3D scanners work.

## Hardware state
- **Owned / on the list:** Raspberry Pi 5 8 GB, AI HAT+ 26 TOPS (Hailo-8), Raspberry Pi AI Camera (IMX500, focuses from 20 cm), 6 × 12 V white LED strips (3 LEDs each), SME 3 W RGB LEDs, aluminium plates 100×100×3 mm, etc.
- **Camera under consideration:** Arducam B0471, 16 MP IMX519 with motorised (software) focus, USB 3. ₹13,186 at Fab.to.Lab Bengaluru, in stock. See `04_usb_camera_options.md`.

## Conclusions reached so far (with reasons)

| Topic | Conclusion | Why |
|---|---|---|
| Fixed 3-camera rig (TriNetra) | Works on the simulator. Code and plan are in the repo (50 tests) | Dark-frame subtraction handles room light. Photometric stereo gives depth (sim −0.69 vs −0.71 mm). Red-dye penetrant mode coded |
| One camera moving on a 0–90° arch | Possible, but adds a mechanism; stop-and-shoot is needed before video | Repeatable angle (about ±0.2°) needed; motion blur; light/frame sync with UVC |
| 90° vs 180° arch | 90° is enough when 3D comes from photometric stereo at the top position | The lights provide the angles; 180° only adds the opposite-side views |
| Gimbal | Not recommended | It stabilises rotation only, not distance or drift. A 2 mm drift is about 55 px at 0.036 mm/px (calc) during a 1–2 s multi-light sequence. A 1 kg payload gimbal costs about ₹30k+ |
| Box resting on the surface (skirt) | Technically sound (fixed geometry, blocks ambient). **User rejected it** ("doesn't work TBH") | — |
| Video sweep for 3D (multi-view / structure from motion) | Not suitable for plain shiny metal | Needs texture; reflections move with the viewpoint |
| 12 V white LED strips | 12 V 1–2 A regulated adapter, one AO3400 MOSFET per strip (100 Ω gate, 10 kΩ pull-down), 1000 µF capacitor, common ground, no PWM | Strips have built-in resistors; about 20–60 mA each (estimate) |
| Handheld power budget | About 12–15 W: 3S 18650 pack (about 33 Wh, 2–2.5 h) + 5 V 5 A buck for the Pi + regulated 12 V for the LEDs | Power banks rarely give 5 V 5 A |

## Leading ideas being researched for handheld + room light + µm detail

1. **How commercial handheld 3D scanners work** (structured light, blue laser lines, NIR VCSEL). They capture **single-shot** frames with **narrowband bright light + bandpass filter + short global-shutter exposure**, which handles room light and hand motion, and track motion between frames. Typical resolution is tens of µm (to verify). Study 1.
2. **GelSight-style contact 3D sensor:** a camera + RGB lights from 3 directions imaging a reflective silicone gel pressed on the surface. This is photometric stereo in a single shot (colour-multiplexed, as in the mentor's R/G/B idea).
   - Handheld, unaffected by room light, and independent of shiny or dark surfaces; µm-level depth claimed (to verify).
   - Commercial: GelSight Mobile (aerospace/automotive). Open: GelSight Mini, Meta DIGIT; DIY possible.
   - Limits: small area per press; no surface colour.
   - Study 2.
3. **Two-stage handheld product:** a wide "find" camera (AI detection in room light using flash + cross-polariser + flash on/off subtraction) plus a "measure" head for µm 3D depth of the flagged spot. Also evaluating handheld videoscopes with 3D phase measurement, handheld microscopes, laser-line gauges, and an Android-phone option (allowed by the brief; ExecuTorch Llama runs on Android). Study 3.

## Prompts of the running research (re-run these if results are lost)

**Study 1: handheld 3D scanners.** Working principles (structured light white/blue LED fringe or speckle, NIR VCSEL, multi-line blue laser triangulation + stereo, active stereo, ToF/LiDAR, photogrammetry). How they work in room light (narrowband + bandpass, NIR, pulsed, short global-shutter exposures). How they handle hand motion (single shot per frame, high fps, ICP/texture tracking, markers, IMU). Product specs (resolution, accuracy, working distance, fps, light source, price, scanning spray on shiny metal, Linux-ARM/Android SDK, India/Bengaluru availability) for:
- Artec Space Spider / Leo / Micro
- Creaform HandySCAN
- Scantech SIMSCAN / KSCAN
- Shining3D EinScan H2 / FreeScan
- Revopoint MINI 2 / POP 3 / MIRACO / INSPIRE
- Creality CR-Scan Raptor / Otter / Ferret
- Intel RealSense D405

Industry methods for micron-level defects (laser profilometers, confocal / white-light interferometry, Keyence LumiTrax photometric stereo, deflectometry, videoscopes with 3D phase measurement), the field-of-view vs resolution trade-off, and laser safety. What a student could build or buy and integrate on a Pi 5 in 4 weeks. **Add a µm-capability column.**

**Study 2: GelSight-style tactile sensors.**
- GelSight Mobile specs and industrial uses
- GelSight Mini, DIGIT, GelSlim (price, USB/UVC, SDK, Linux/Pi support, licences)
- DIY recipe: silicone types, reflective coating, acrylic, LED layout, close-focus camera, mm/px, ball-indenter calibration → lookup table / NN, Poisson integration, failure modes
- papers using it for defect inspection
- limits: area, tight cracks, no colour, pressure, wear
- Bengaluru/India sourcing and prices
- Pi 5 CPU speed and Hailo-8 for the networks
- **Published µm numbers, and how DIY reaches ≤10 µm depth**

**Study 3: handheld inspection architecture.**
- Industrial handheld tools (videoscopes with 3D phase measurement, handheld microscopes, weld/gap laser gauges, smartphone AI apps) and how they handle ambient light and motion
- Room-light techniques (flash + short exposure + on/off subtraction, cross-polarisation, narrowband + bandpass, global shutter, flash irradiance vs 300–500 lux)
- Resolution vs field of view (µm/px for macro / objective optics, pixels needed for a 20 µm scratch, two-stage find + measure)
- Android option (ExecuTorch Llama 3.2 on Android, YOLO/anomaly backends, clip-on macro/polariser, phone + USB-C accessory vs Pi 5 + Hailo for handheld)
- Datasets (MVTec AD, NEU, KolektorSDD2, Magnetic tile, MVTec 3D-AD, Real3D-AD)
- Student/startup lessons
- A recommended 4-week handheld PoC architecture covering every brief bullet, plus the industry scale-up path
- **A µm-level target**

## Result: GelSight-style contact gel sensor (02, COMPLETE)

**Only GelSight's own metrology probes are truly µm-level, and they're priced out.**
- GelSight Mobile: Z accuracy 1 µm + 4 %, XY 6.9 µm, field of view 17 × 14.2 mm.
- Cost about US$30–55k; gel cartridges US$600.
- It is not single-shot: 6 LEDs, 6 images at 50 fps (100 ms), 3D within seconds.

**The cheaper sensors are not metrology tools, by GelSight's own statement.**
- GelSight Mini ($510) and DIGIT ($355).
- Best published Mini result: 12.55 µm mean absolute error, with a custom gel and careful calibration.

**A 4-week DIY build is feasible as a demonstrator,** at about 10–30 µm depth error and about 5 µm/px on 20–200 µm-deep scratches, dents and pits. ≤10 µm is a stretch; sub-µm isn't achievable.
- Optics: IMX219 refocused to about 16 mm, or HQ Camera + 16 mm lens + extension. **The AI Camera (64 µm/px at 20 cm) can't image the gel.**
- Light 4–6 directions one at a time, with dark-frame subtraction.
- Soft clear silicone (Solaris 15A, or Indian Fortisil/Trisil around ₹1,200–1,300/kg), 1 µm aluminium powder coat.
- Ball-indenter calibration, LUT or MLP, then FFT Poisson integration (reuse `shape.integrate_slopes`).
- Go/no-go check about 3 Nov: a 50 µm shim step must measure within ±10 µm.

**Evidence for the two-stage design** (Agrawal et al. 2023, GelSight Mobile on Boeing panels):
- vision alone: precision 0.62 / recall 0.79
- touch: 1.0 / 1.0
- vision-then-touch covered the whole panel in 168 s, against 12,978 s for touch alone

**Limits:**
- 17–19 mm field per press
- can't see closed cracks with no relief (keep red-dye or the camera for those)
- no colour
- contact force sets the measured depth
- gel wear

**Compute:** gradients and Poisson on the Pi CPU (about 10–25 ms at 640×480, estimate); defect segmentation on the Hailo (UNet-MobileNetV2 256² at 662 FPS in the Model Zoo, measured over PCIe Gen3 x4; the Pi 5 has a single Gen 2 lane by default).

## Next step after the research
Synthesise one recommendation, with a feasibility verdict for 23 Nov: product architecture, parts (Bengaluru), the µm resolution actually achievable, and the code changes. The existing photometric stereo, rules, JSON and Llama modules are reusable.
