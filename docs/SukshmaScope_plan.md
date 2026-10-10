# SukshmaScope: merged build plan (verified revision)

**Context:** Bharat AI-SoC Challenge 2026-27, PS5 "Real-time industrial defect detection & quality-control assistant". Prepared 10 Oct 2026; week 1 starts Mon 12 Oct 2026.

**Revision note (adversarial verification, 10 Oct 2026).** The main changes from the previous draft:
- The Scheiner autofocus crossing is now biased, so the two spots stay resolved at best focus.
- The autofocus attenuation chain is corrected, and so are the pick-off ghosts.
- The bare 850 nm module is recognised as Class 3B.
- ALIGN mode and the motion interlock are redesigned.
- Week-1 laser tests are made safe.
- The 520 nm part type is corrected, and missing BOM items are added.
- The clash between the LED ring and the laser beams is resolved.
- Frame gating is made conservative.
- The Hailo-in-the-verdict-path fallback is added.
- The toolchain moves to native x86 instead of a VM on the Apple-silicon Mac.
- A Phase-1 minimum scope is defined.
- The station price now uses the sourced figure.

**Labels used throughout:**
- **(calc)** = my own calculation from the cited specs
- **(est.)** = an estimate to be measured
- **\*** = an unverified price
- Currency: ₹88 per US$. Prices are seller listings, not quotes.

---

## 1. Summary and positioning

**What it is.** SukshmaScope ("sukshma" means minute or subtle) is an upright reflected-light microscope that the team builds itself. It is interlocked and runs fully offline.

**Optics**
- The **unmodified Raspberry Pi AI Camera** looks afocally through a WF10x/18 eyepiece and 160 mm DIN RMS objectives.
- **4x** is for survey, **10x** for review, and **40x** only for bright-field teaching.

**Laser dark-field**
- Three grazing lasers, each 1 mW or less, sit at 0°, 120° and 240° azimuth.
- They make scratches, particles and pits smaller than the resolution limit glow on a black field.
- The ratio between the three azimuths is a physics feature that separates scratches from particles and gives each scratch's orientation.

**Laser autofocus**
- A two-spot (Scheiner) laser autofocus locks onto mirror-polished parts. On these parts image autofocus has no signal, and in dark-field the frame is black.
- The beam crossing point is **biased** away from the focal plane, so the two spots stay resolved at best focus.
- **Phase 1:** the AI Camera reads the spots.
- **Phase 2:** a dedicated 850 nm arm with an OV9281 global-shutter camera reads them.

**Three compute domains, documented separately**
1. **IMX500 on-sensor NPU:** an optional live-rate field gate.
2. **Hailo-8 through HailoRT:** defect classification and segmentation. A Hailo model is always in the verdict path.
3. **Arm Cortex-A76:**
   - deterministic measurement, rules, alerts and SQLite logging
   - Llama 3.2 1B through ExecuTorch, which writes advice and never produces a number.

**PS5 framing, to use verbatim in the report:**
> "SukshmaScope is an offline AI inspection microscope for metallurgical and electronics QC that doubles as a general-purpose teaching microscope for academic labs."

**What PS5 asks for:** detect surface defects, cracks and anomalies; classify type and severity; alert operators with actionable recommendations; log results; run in real time at the edge without cloud. It names QC operators on the factory floor as the user. Source: https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html

**How the UI meets it**
- The UI **opens in QC mode**: load → laser autofocus → bright-field plus laser dark-field → measure → spec check → PASS/FAIL with severity → stack light → SQLite → report.
- **Lab mode** is the second tab.

**Demos**
- **Headline:** mirror-finish manufactured parts. These are lapped or polished 304 stainless seal-face coupons, e-waste HDD platters and Si die pieces. Both laser modes are essential on them.
- **Second:** a polished ductile-iron section. Bright-field nodularity is measured behind a dark-field prep-quality gate.
- PCB and grain size are LED-only teaching modes.

**Who it is for**

| Segment | Why | Evidence |
|---|---|---|
| **Primary (PS5 user):** QC benches at MSME foundries and heat-treatment shops, precision lapping and polishing shops, Si/OSAT/EMS incoming inspection | Scratch, particle and dig cleanliness on polished parts; nodularity | India is the #2 casting producer, about 90% MSMEs: https://www.foundryinfo-india.org/profile_of_indian.aspx |
| **Secondary (market and validation channel):** college metallography, materials and electronics labs | AICTE PCC-MM207 covers "cast, wrought and welded structures with defects and failures" | https://fliphtml5.com/jfjx/uffx/Updated-Vol-02/83 |
| | Old, scarce instruments | 10 microscopes, 6 of them from 2000: https://Lbrce.ac.in/mech/mech_labs/06%20Metallurgy%20&%20Material%20Science%20Lab.pdf ; 3 microscopes for 2 sections: https://diet.ac.in/wp-content/uploads/2023/05/08.Metallurgy-Lab.pdf |
| | Manual E112 counting is "time-consuming and error-prone" | https://ira.lib.polyu.edu.hk/handle/10397/103898 |

**Price against Indian alternatives**

| Option | Price | Source |
|---|---|---|
| Metnation trinocular metallurgical microscope | ₹2,80,955–6,35,050 | https://m.indiamart.com/metnationtechnologies/laboratory-microscope.html |
| Image-analysis / nodularity software | ₹96,000 / ₹1,55,000 | same |
| 10 MP microscope camera | ₹70,790 | https://m.indiamart.com/uvsar-india/microscopes.html |
| Typical station (calc: lowest Metnation trinocular ₹2,80,955 + ₹70,790 + ₹96,000) | **₹4,47,745** | sourced listings above (the earlier ₹2,59,000 figure had no source) |
| Student compound microscope (no dark-field, calibration or AI) | ₹32,000–56,880 | https://m.indiamart.com/acmeinstrumentco/labomed-microscope.html , https://org.gem.gov.in/cart/cart_global/directBuy/754454 |
| USB "1000x" microscope (2 MP, uncalibrated) | ₹2,155–2,495 | https://www.moglix.com/microware-40-1000x-20mp-8-led-digital-microscope-for-inspection/mp/msne9j8zpmw09d |
| **SukshmaScope prototype, new spend (§12)** | ₹0.98–1.47 lakh, plus the owned Pi 5 and AI Camera | calc |
| **SukshmaScope target list price** (assembled, calibrated, with software) | **₹1.5–2.0 lakh**, about 34–45% of a station | calc |

- **Procurement route:** AICTE MODROB funds up to ₹20 lakh per proposal, so roughly 10 units: https://aicte.gov.in/sites/default/files/MODROB%20Scheme%20Guidelines%2024_12_2021.pdf
- **Buyer needs:** deliver it pre-assembled, with lockable settings and no Linux or Git burden: https://pmc.ncbi.nlm.nih.gov/articles/PMC11448745/

**Why not a DIY electron microscope (three lines)**
1. **A laser does not make an electron microscope.** A laser emits photons, so the instrument stays diffraction-limited (about 1.3 µm at NA 0.25). In an SEM a laser can only fire the cathode (UBC 2025); high voltage, vacuum, electron lenses, scan coils and a point detector are still needed: https://pmc.ncbi.nlm.nih.gov/articles/PMC12394587/
2. **The track record is poor.**
   - Ben Krasnow took about 4 months to reach about 50 µm resolution.
   - UBC took a multi-year, professor-led effort to get below 1 µm, at about US$5k plus a turbo pump.
   - Most hobby builds produced no image, and the IMX500 cannot act as an SEM detector.
3. **It would be regulated, and SEM time is cheap to rent.** A home-built 25–35 kV column falls under AERB X-ray licensing and type approval: https://www.aerb.gov.in/storage/uploads/documents/regdocTyjvJ.pdf . Real SEM time costs about ₹708 per sample including GST at IIT Delhi through I-STEM (tariff from about 2022): https://www.istem.gov.in/uploads/governmentorders/1647003348_IIT_Delhi.pdf

**Why it should score well** (1st-edition criteria as a proxy: alignment, correctness, novelty, criticality of Arm tools, performance, documentation, demo: https://arm-education.github.io/Arm-Developer-Labs/Challenge_Page.html)
- It covers all five PS5 capabilities.
- Its three pipelines are documented separately, as the rules require, and a HailoRT model is always in the verdict path.
- **Novelty:** the landscape survey found no microscope that runs IMX500 on-sensor inference, none built on Hailo, and none that combines an NPU with a local LLM. The closest IMX500 work is the medical Q-Segment: https://arxiv.org/abs/2312.09854
- It meets the 2nd edition's "engineering AI-enabled products on SoCs" theme with measured evidence: POD, autofocus statistics, latency, tok/s, and a CPU-only comparison against the accelerators.
- One PS5 rival does NEU-DET with a rule-based advisor: https://github.com/katyayanitiwari/EdgeInspect-AI

---

## 2. Key decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Imaging camera | AI Camera unmodified, lens locked at infinity, afocal behind WF10x/18 | The lens mount is "N/A" and the IR-cut is "Integrated" (https://www.raspberrypi.com/documentation/accessories/camera.html). The IMX500 is a bare chip (https://sony.mediaroom.com/2020-05-14-Sony-to-Release-Worlds-First-Intelligent-Vision-Sensors-with-AI-Processing-Functionality). Removing the lens gives pink shading (https://forums.raspberrypi.com/viewtopic.php?t=368940). Afocal coupling keeps the NA, because the 1.25 mm exit pupil is smaller than the 2.65 mm camera entrance pupil. All lens-removed numbers in research C1 are discarded. |
| D2 | Objectives | **160 mm DIN RMS achromats, both phases.** 4x/0.10 and 10x/0.25 for QC; 40x/0.65 for bright-field teaching only. The 100x oil objective in the set is not used (teaching or loan only). | **Overrides the red team's infinity recommendation.** My calculation (§3.3): 45° plates up to 1.05 mm thick in the slow DIN cone add λ/31–λ/32 RMS astigmatism, against the λ/14 limit. A Scheiner autofocus beam focused near an image-conjugate point gets its pupil offset from angle (Abbe sine condition), so no pupil relay is needed. DIN parts cost $104 for a set or can be borrowed. They avoid tube-lens brand mismatch and import lead time. |
| D3 | Orientation and Z | Upright. The **whole head rides Z**: objective, dark-field collar, epi arm, autofocus arm, eyepiece, camera. | PCBs, dies and mounts lie face-up, and the lasers need room above the sample. Parfocal 45 mm objectives keep the focal point fixed relative to the head, so the dark-field beams and the autofocus stay aimed. A 1 mm sample-height change would move a 20° beam about 2.7 mm off a 1.8 mm field (calc). |
| D4 | Dark-field magnifications | 4x survey, 10x review. Dark-field is software-blocked at 40x. | 40x working distance is 0.53–0.6 mm, so with an assumed 3 mm nose radius the elevation would have to be below 10–11° (calc; 6–7° if r = 5 mm), and a 4–5 mm beam cannot pass under the nose at all. BD objectives are rejected on cost and M32 thread (https://evidentscientific.com/en/microscope-resource/knowledge-hub/techniques/darkfieldreflect). |
| D5 | Dark-field geometry | 3 lasers at 0/120/240° azimuth, 20° elevation (printed slots 15–25°; 30° only if CAD shows clearance of the LED ring), defocused Gaussian spot about 5 mm (1/e²), polarisation set in the week-3 test (s baseline), one V-trap each | 4 beams at 90° give only 2 distinct orientations, because the 2φ terms collapse (red team). 3 beams at 120° give 3 orientations 60° apart. Odd counts measure as more consistent (https://opg.optica.org/ao/abstract.cfm?uri=ao-65-1-309). A static diffuser would imprint speckle on the illumination, so the spot is defocused instead. |
| D6 | Dark-field wavelength | **520 nm direct-diode DOT modules are the production source,** ordered in week 1. 650 nm ships in Phase 1 only if the week-1 IR-cut test passes. | My budget (§4.5) gives about 9x more electrons at 520 nm for 1 µm. Only about 1.7x comes from scattering; the rest is the camera (IR-cut, Bayer). No 405 nm (blue-light hazard, gain only below 0.3 µm), and no DPSS 532 nm (IR leak: https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1668.pdf). |
| D7 | Laser power | Run every module at nominal current. Fix an absorptive ND, rated at the module's wavelength, to its aperture. Measure ≤1 mW (visible) or ≤0.5 mW (850 nm). **A bare 5 mW 850 nm module is Class 3B** (Class 3R AEL ≈3.9 mW at 850 nm, calc): prefer a maker-rated ≤0.5 mW module, otherwise bond an ND1 whose OD is specified at 850 nm before first power, and use Class 3B handling until then. | This settles the red-team trap: derating near threshold is unstable, and PWM may not be used to lower the class. Gel and plastic NDs are often much less dense in the NIR, so they are never used at 850 nm. |
| D8 | Phase 1 autofocus | 650 nm **two-hole Scheiner** beam through the epi port, with the **crossing biased** about 30 mm (image space) from the field-diaphragm conjugate. The module has a fixed ND2 cap (≤50 µW, Class 1 by power). An ND slider (0/1/2/3) plus PWM sets brightness. The pick-off is a #1.5 coverslip or a wedged plate. Read by the AI Camera in dedicated frames. | Uses no new camera or dichroic. Tilt- and pointing-insensitive: a single-spot design would read 2 µm of pointing drift at 10x as 8.3 µm of defocus (calc). The bias keeps the spots about 3 spot diameters apart at best focus (calc, §5.2), so the fit is never degenerate. Fallback chain: reticle → image → manual. |
| D9 | Phase 2 autofocus | 850 nm, ≤0.5 mW (Class 1), Scheiner with the same bias. Edmund #69-196 800 nm shortpass dichroic (1.05 mm) in a slot **reserved from day 1** 15–25 mm above the objective. OV9281 on CAM1 behind an 800 nm longpass. The injection plate is wedged, and the 92% of the injection beam that passes the plate goes into a dump. | PureFocus-style layout; dichroic placed nearest the objective for dark-field (https://www.prior.com/blog/adaptive-autofocus-empowers-microscopy-a-technical-overview). Runs at 60–120 Hz and tracks during XY moves. Use 850 nm, never 780 nm, with an 800 nm shortpass edge; 780 nm sits in the transition band. |
| D10 | Ports and I/O | CAM0 = AI Camera, CAM1 = OV9281, display on HDMI. Both camera ribbons are 300–500 mm, 22-pin, routed through a light-tight brush gland with strain relief for 30 mm of Z travel. **Pi GPIO** drives light sources, stack light and interlock read-back. **Pico** drives steppers and the WS2812 ring. | The Pi 5 has only 2 MIPI ports (https://www.raspberrypi.com/products/raspberry-pi-5/). The Pi sits in the vented bay, so standard 200 mm ribbons will not reach. Light-switch times stay on the camera's clock. The Pico PIO avoids unverified WS2812 timing on the Pi 5's RP1. Stepping from Linux jitters. |
| D11 | Compute roles | IMX500 FieldGate is optional and no verdict depends on it. Hailo H1 is a 3-azimuth crop classifier (Phase 1) and tile segmenter (Phase 2), and its class label feeds the rule engine. The CPU computes every number and applies every rule. Llama 1B on cores 2–3 is advisory only. **A HailoRT model is always in the verdict path:** if the custom HEF fails, a precompiled Model Zoo HEF (ResNet-18-class backbone) runs as a crop embedder with a CPU linear head. If the organisers do not accept the IMX500, FieldGate is compiled to a HEF and runs on the Hailo. | The IMX500's PS5 acceptance is unconfirmed, and it sees single frames only. The PS5 hard rule puts vision on the Hailo NPU through HailoRT. Llama is never on the alert path. |
| D12 | "Real-time" | Per-field verdict ≤0.3 s after the last frame; field cycle 0.7–1.2 s (calc; up to about 1.3 s if T6 forces the conservative switch rule); IMX500 gate at 30 fps on the frames it is given. Part throughput is stated openly. | Avoids over-claiming. |
| D13 | Headline demo | D1: mirror-finish part QC. D2: polished ductile-iron (bright-field nodularity plus dark-field prep gate). D3 (Phase 2): Si die map and PCB at 4x under LED. Grain size = Lab mode. | Both laser modes are idle or harmful on PCB and etched grain (speckle, glints). They are essential on featureless polished parts. |
| D14 | Safety architecture | Opaque enclosure. 2 lid switches + camera-collar switch + key + latching laser relay K1 with manual reset. A separate motion relay K2 runs on the lid chain only. Emission LED. Camera-only viewing (eyepiece sealed in the collar). Principal Laser User; sign-off. | The eyepiece is a hidden eye port, and the 650 nm autofocus return travels straight up it. Microscopes count as optical aids (https://www.fda.gov/radiation-emitting-products/home-business-and-entertainment-products/laser-products-and-instruments). K2 keeps motion available in key-OFF LED teaching mode but stops it whenever the lid opens. |
| D15 | Licences and toolchain | Mechanics designed in-house (OpenFlexure cited only as prior art). **Native x86-64 Linux** (college lab PC) or a cloud x86 instance (Colab, JarvisLabs) in week 1 for the Hailo DFC and the IMX500 export. | CERN-OHL-S reciprocity and GPLv3 server code; the rules require entrants to own all materials. The team Mac is Apple-silicon (M3), so an x86-64 VM on it would be emulated and impractically slow for the DFC; the IMX500 converter needs x86-64 Linux with Java 17. |

### 2b. Red-team contradictions: rulings

| Topic | Ruling |
|---|---|
| Removing the AI Camera lens | A2 overrides C1. The camera stays unmodified and every lens-removed number is discarded. Afocal: 2.04/0.82 µm/px; 10x tensor 1.99 µm/px on the inscribed square. The 30 fps binned mode is for dark-field detection, autofocus and preview; measurement uses full resolution. |
| DIN vs infinity | **DIN is kept; I disagree with the red team's recommendation here.** (a) Astigmatism is λ/31 for the 1.05 mm dichroic and λ/32 for the 1 mm beamsplitter (calc). The two are tilted about orthogonal axes, which leaves a residual focal split of (1.05−1.00)×0.269 ≈ 0.013 mm. The Phase-1 blank spacer is tilted at 45° about Y for the same reason. (b) The Scheiner beams are focused to a point near the intermediate-image conjugate (offset by the bias). Their angle u′ = h/f_L maps to object angle sinθ = M·u′, so no pupil relay is needed. (c) Path budget: the 150 mm tube holds the dichroic at 15–25 mm and the beamsplitter at about 55 mm. Each 45° plate shifts the image about 0.5–0.75 mm axially (about 5–7.5 µm in object space at 10x, absorbed by Z) and about 0.33 mm laterally (about 33 µm at 10x, centred at assembly; the orthogonal tilts displace in orthogonal directions). (d) Infinity objectives stay a later upgrade, not a Phase-2 dependency. |
| Hot mirror vs afocal eyepoint | Nothing may sit between the eyepiece and the camera. Both plates sit below the eyepiece field stop. The collar sets the camera pupil on the eyepoint; eye relief and pupil depth are measured in test T1. |
| Wavelength 520 / 405 / 650 | 520 nm direct-diode dot modules; 650 nm only once IR-cut transmission is measured; no 405 nm; no fluorescence (outside the user's decisions). |
| Autofocus architecture | Through-the-lens only. An oblique 60° spot misses the objective on mirrors, and external specular triangulation reads 1 mrad of tilt as about 70 µm. Phase 1: 650 nm Scheiner read by the AI Camera. Phase 2: 850 nm with the OV9281. Both use a biased crossing, so the two spots stay resolved at best focus. |
| Knife edge vs Scheiner | Scheiner (two holes of unequal size, so the defocus sign is readable) in **both** phases, on a rigid aluminium block, with the bias of §5.2. |
| Number of dark-field lasers | 3 at 120°, with a 5-at-72° option in Phase 2 only if the week-3 orientation sweep shows dips. |
| Laser power and class | Class 2 visible (≤1 mW), Class 1 NIR (≤0.5 mW). Modules are attenuated by an ND rated at their wavelength, at nominal current. Never PWM to lower the class. A bare 5 mW 850 nm module is Class 3B until it is capped. |
| Stage speed vs "real-time" | Lead-screw XYZ instead of OFM flexures: 0.25–0.45 s per field step, against about 14–25 s with OFM. Real-time = per-field latency. A 10×10 mm area at 10x takes about 58–93 s (calc; up to about 107 s with the conservative switch rule). |
| Demos vs laser modes | The laser headline is polished-part QC. Nodularity is bright-field with a dark-field prep gate; nodules are masked out of dark-field using the bright-field image. PCB and grain size run under LED. |
| Detection-limit claims | Only measured POD is published, and "detected" is reported separately from "resolved". The C1 85–110 nm figure is dropped. |
| Eyepiece vs teaching | No binocular head; teaching happens on the screen. The eyepiece is sealed in a tool-fixed collar with an interlock switch. The key-OFF position gives an LED-only teaching mode in which motion stays enabled through K2. |
| Second MIPI port | CAM1 = OV9281; HDMI display; no HQ, holography or DSI. |
| Hailo throughput | Planned with an estimated Pi 5 app throughput well below the model-zoo figures (est., from the team's earlier BeamTriage work; to be measured in T9 and week 5). The crop classifier keeps the load small. |
| IMX500 role | Optional gate; its value is shown by an on/off ablation; asked about in the PS5 email. If it is rejected, FieldGate runs on the Hailo. |
| PiFocus | Only the centroid and control-loop ideas are used. Licence checked before reusing any code. |

### 2c. Must-decide items: answers

| Item | Answer |
|---|---|
| Optical train by 15 Oct | DIN 160 mm + WF10x/18 + afocal AI Camera, with a hot-mirror slot at 15–25 mm (D2). |
| Imaging camera | Unmodified AI Camera (D1). |
| Ports and display | CAM0 AI Camera, CAM1 OV9281, HDMI 7-inch display; 300–500 mm ribbons through a light-tight gland. |
| Orientation | Upright, head on Z, specimen tilt ≤0.2° using a levelling press or parallel-lapped mounts (0.2° over 1.8 mm = 6.3 µm, calc). |
| Headline demo | D13. |
| Dark-field sources | 3 × 520 nm dot modules (650 nm stopgap), 120°, 20°, ring baseline at about 26 mm height (§4.7). |
| Autofocus phasing | D8 and D9. |
| Real-time and stage | D12; lead screws, not OFM. |
| Licence path | In-house mechanics; "Built with Llama"; dataset licence audit; no ASTM text in the retrieval corpus. Ask the organisers about reusing CERN-OHL hardware only if it is ever used. |
| Timing and I/O controller | Pico for motion and the WS2812 ring; Pi GPIO for light sources (timestamp integrity). |
| Safety architecture | D14 and §11. |
| Thermal layout | Pi 5, HAT, PSUs and laser drivers sit in a vented bay outside the optical enclosure. Column and autofocus block are aluminium; prints are PETG/ASA, never PLA (Tg about 60 °C). DS18B20 probes on the column and the autofocus block. |
| Phase-1 scope freeze | §13, including the Phase-1 minimum and descope order. The 850 nm arm, focus map, POD table and grain size are deferred to Phase 2. |
| Build machine | Native x86-64 Linux PC or cloud x86 instance in week 1 (not a VM on the M3 Mac). |
| PS5 email | Sent by Wed 14 Oct (§13, test T15). |

The traps the red team listed are handled in §3.6 (drift and tilt), §3.5 (stray light), §4 (glints, speckle, beam spill), §6 (sync and AE/AWB), §5.6 (glass and PCB), §9–§10 (licences), §11 (modules, eyewear, eyepiece port), §12–§13 (imports) and §15.

---

## 3. Optical design

### 3.1 Train and sketch

The head is upright and rides the Z rail. Elements from the bottom up:

1. **Sample**, face-up on the XY stage.
2. **DIN RMS objective** in a kinematic quick-swap holder. The ID is selected in the UI in Phase 1 and read by a Hall sensor in Phase 2.
3. **Hot-mirror slot, 15–25 mm above the objective shoulder.**
   - Phase 1: a blank 1 mm glass spacer, tilted at 45° about the Y axis, which keeps the optical path and the astigmatism compensation identical.
   - Phase 2: the Edmund #69-196 dichroic, tilted about the Y axis, with the autofocus arm on +X.
4. **25 mm 50/50 plate beamsplitter** (back face AR-coated), about 1 mm thick, about 55 mm up, tilted about the X axis, with the epi arm on +Y.
5. **Intermediate image** at the FN 18 field stop, about 150 mm from the shoulder plus about 1–1.5 mm of plate shifts.
6. **WF10x/18 eyepiece** (f = 25 mm).
7. **AI Camera** with its lens on, focus locked at infinity and its entrance pupil at the eyepoint. A printed axial-adjust collar holds it; the collar is fixed with tools and carries an interlock microswitch.

```
                         CAM0 (AI Camera, lens ON, focus = inf, EP at eyepoint)  [collar switch]
                              | collimated (afocal)
                       [WF10x/18 eyepiece f=25]
        ----- intermediate image / FN18 field stop (~150 mm) -----
                              | slow cone, NA' = NA/M = 0.025 (4x,10x)
 +Y epi arm: LED>opal>f30 collector>iris(AS)>[field stop + slit reticle]>[coverslip pick-off]>[50/50 plate, tilt about X]--trap
                                                             ^ P1 AF: 650 module+ND2 > ND slider 0-3 > 2-hole mask (+-0.9 mm)
                                                               > f75 lens (focus biased ~30 mm off the field-stop conjugate)
                              |
 +X (P2): OV9281<800LP<[wedged 8/92 plate]<f75<2-hole<850nm+ND1(NIR-rated)  ->dump for 92%
                              ---[HOT-MIRROR SLOT 15-25 mm, tilt about Y] (45 deg blank in P1)
                              |
                   [DIN objective 4x | 10x | 40x(BF only)]   <- quick-swap RMS holder
     DF1 (az 0)  \  20 deg elev, ~5 mm spot     |       / V-trap (az 180)
     WS2812 ring  '-----------> [SAMPLE face-up] <----'     DF2 az 120 / DF3 az 240, traps at 300 / 60
     (~26 mm up, LEDs at r ~36-43 mm, ~33 deg, shrouded)   XY: MGN12+NEMA17+T8x8 (Pico PIO)   Z: whole head, MGN12+NEMA17+T8x2
```

### 3.2 Afocal formulas and the µm/px / FOV table (calc)

**Inputs**
- AI Camera: f_cam 4.74 mm, F1.79, 1.55 µm pixels, 4056×3040 at 10 fps, or 2028×1520 binned at 30 fps (https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf).
- Eyepiece: f_eye 25 mm, FN 18 (https://moticeurope.com/en/widefield-eyepiece-wf10x-18mm-18.html).

**Formulas**

| Quantity | Formula | Value or source |
|---|---|---|
| Sensor magnification | M_sensor = M_obj·f_cam/f_eye | 0.1896·M_obj (https://www.photomacrography.net/forum/viewtopic.php?p=99265) |
| Object-space pixel | p_obj = 1.55 µm / M_sensor | ×2 in binned mode |
| Image circle on sensor | FN·f_cam/f_eye | 3.41 mm = 2,202 px; inscribed square 1,557 px |
| Eyepiece apparent half-field | atan(9/25) | 19.8°, below the camera's 26.15° vertical half-FoV, so the whole circle is captured |
| Exit pupil | 500·NA/(10·M_obj) | must be < the camera entrance pupil 4.74/1.79 = 2.65 mm, so the NA is kept |
| Resolution | r = 0.61λ/NA, λ = 550 nm | https://svi.nl/RayleighCriterion ; Nyquist pixel = r/2 |
| Depth of field | MicroscopyU tabulated values (visual; the camera value is somewhat larger, see T4) | https://www.microscopyu.com/microscopy-basics/depth-of-field-and-depth-of-focus |
| IMX500 tensor pixel | p_obj·1557/640 (square) or p_obj·2202/640 (whole circle) | FieldGate at 224: p_obj·1557/224 |

**Per-objective values**

| | 4x/0.10 (survey) | 10x/0.25 (review) | 20x/0.40 (not bought) | 40x/0.65 (BF teaching) |
|---|---|---|---|---|
| M_sensor | 0.758 | 1.896 | 3.79 | 7.58 |
| µm/px, full / binned | 2.04 / 4.09 | 0.82 / 1.64 | 0.41 / 0.82 | 0.20 / 0.41 |
| Field circle / inscribed square | 4.50 / 3.18 mm | 1.80 / 1.27 mm | 0.90 / 0.64 mm | 0.45 / 0.32 mm |
| Rayleigh / Nyquist pixel | 3.35 / 1.68 µm | 1.34 / 0.67 µm | 0.84 / 0.42 µm | 0.52 / 0.26 µm |
| Exit pupil (camera EP 2.65 mm) | 1.25 mm | 1.25 mm | 1.00 mm | 0.81 mm |
| Depth of field | 55.5 µm | 8.5 µm | 5.8 µm | 1.0 µm |
| Working distance, DIN | 18–26 mm | 4.4–7 mm | 0.8–1.7 mm | 0.53–0.6 mm |
| IMX500 tensor pixel, square → 640 / whole circle → 640 | 4.97 / 7.03 µm | 1.99 / 2.81 µm | 0.99 / 1.41 µm | 0.50 / 0.70 µm |
| Native 640 ROI covers | 1.31 mm | 0.52 mm | 0.26 mm | 0.13 mm |
| FieldGate 224 input pixel | 14.2 µm | 5.7 µm | — | 1.4 µm |

Working-distance sources: https://magnifiers.nz/product/biological-microscope-objectives/ , https://www.edmundoptics.com/p/40x-din-achromactic-finite-intl-standard-objective/3136/ , https://www.edmundoptics.com/p/20x-din-achromactic-finite-intl-standard-objective/3135/

**What the table means**
- **Sampling:** 4x and 10x are slightly pixel-limited, and red and blue are sampled at twice the pixel pitch. So bright-field sizes are claimed only for features of 3 px or more (≥2.5 µm at 10x full resolution), and widths only above 2×Rayleigh (≥2.7 µm at 10x).
- **Binned mode:** at 10x the 1.64 µm binned pixel is close to the 1.34 µm resolution. Binned frames are fine for dark-field *detection* and autofocus, but measurement stills run at full resolution (10 fps).
- **Wasted readout:** the sensor has no documented crop modes, so about 69% of each frame read out is black surround (calc: the circle is 3.8 of 12.3 MP).

### 3.3 Plates in the DIN converging space (calc)

**Inputs:** plate index n = 1.517, tilt U = 45°.

**Astigmatic focal split (Smith):** Δ = t/√(n²−sin²U)·[n²cos²U/(n²−sin²U) − 1] = **0.269·t**.

**Wavefront error:** W_pv = Δ·NA′²/2, and W_rms = W_pv/(2√6).

| Plate | Image-side NA′ | W_pv | W_rms | Verdict (Maréchal limit λ/14) |
|---|---|---|---|---|
| 1.0 mm beamsplitter, 4x/10x | 0.025 | 84 nm | 17.2 nm = **λ/32** | ok |
| 1.05 mm #69-196, 4x/10x | 0.025 | 88 nm | 18.0 nm = **λ/31** | ok |
| 1.0 mm, 40x | 0.016 | 34 nm | 7.0 nm = λ/78 | ok |
| 3.0 mm hot mirror | 0.025 | 252 nm | 51.5 nm = λ/11 | **avoid**, or add an equal plate tilted in the orthogonal plane |

The two plates are tilted about orthogonal axes, so their astigmatism cancels to (1.05−1.00)×0.269 ≈ 0.013 mm (assuming similar substrate index; check the #69-196 substrate). Each plate also shifts the image about 0.5–0.75 mm axially and about 0.33 mm laterally (calc). Both shifts are static: Z absorbs the axial one, and the lateral one is centred at assembly. Coma from the plates at NA′ 0.025 is a few nm (calc), so it is negligible.

**Missing cover glass on bare metal (DIN objectives are corrected for 0.17 mm):** W040 = t(n²−1)/(8n³)·NA⁴.

| Objective | W040 | Balanced RMS | Verdict |
|---|---|---|---|
| 10x/0.25 | 31 nm | 2.3 nm | negligible |
| 40x/0.65 | 1.4 µm | 105 nm ≈ λ/5 | poor. 40x is used only on covered slides, which confirms D4 and D2. |

### 3.4 Beam envelope and slot sizes (calc)

**Estimate:** D(d) ≈ D_pupil(1 − d/150) + FN·d/150, with D_pupil ≈ 2·f·NA ≈ 8 mm at 10x (the largest of the three objectives).

| Height above shoulder | Envelope | Slot fit |
|---|---|---|
| 15 / 20 / 25 mm | 9.0 / 9.3 / 9.7 mm | **#69-196:** its 12.5 mm side × 17.6·cos45° = 12.4 mm clear aperture fits with about 1.4 mm margin per side |
| 55 mm | 11.7 mm | a **25 mm plate**. A 10 mm cube (Holmarc, https://www.holmarc.com/metalic_cube_beamsplitter.php) would clip the field. |

### 3.5 Epi bright-field and stray light

**Illumination path**
1. 1 W neutral-white LED on constant current; PWM at 20 kHz or more on GPIO12 (https://pi4j.com/blog/2024/20240423_pwm_rpi5).
2. Opal diffuser.
3. Collector lens, f ≈ 30 mm.
4. Aperture iris.
5. Field diaphragm at about 95 mm optical path from the beamsplitter. This plane is conjugate with the intermediate image.
   - It carries a removable **slit reticle**, which gives a Leica-style "focus finder" for mirror-like parts (https://www.mikroskop-center.de/media//wysiwyg/Leica_DM12000_M/Leica_DM8000_M_DM12000_M-TechnicalBrochure_en.pdf).
   - The Phase-1 autofocus focuses about 30 mm away from this plane (the bias of §5.2).

**Stray-light control.** Veiling glare from the beamsplitter is the known failure of low-cost epi builds (https://openflexure.discourse.group/t/refletion-microscope-without-beam-splitter/2268).
- Flocked light trap behind the beamsplitter for the transmitted half.
- Matte-black baffles and a lens snoot.
- **Per-session veil frame** (epi on, light trap instead of sample) subtracted from every bright-field frame.
- **Flat-field** correction from polished Si.
- **Per-azimuth dark-field illumination maps** from a ground-glass target at the focal plane.

OpenFlexure metallurgy users also needed flat-field correction: https://openflexure.discourse.group/t/ofm-optics-for-metallurgy-applications-epi-illumination-kohler-design/74

**Phase-2 option:** a 70T/30R plate in place of the 50/50, which sends 1.4x more dark-field light to the camera.

### 3.6 Mechanics, drift and tilt

**Z axis**
- Vertical MGN12 rail on a 2040 aluminium column.
- NEMA17 + TMC2209 + T8×2 lead screw with anti-backlash nut.
- At 1/32 microstepping, 0.31 µm per microstep (calc: 2 mm/6400). 30 mm travel at about 2 mm/s.
- Final approach always from below; closed loop on the autofocus signal, so microstep nonlinearity matters only at 40x, where a z-stack is used (§5.2).

**XY**
- 2 MGN12 rails with NEMA17 and T8×8 screws: 1.25 µm per microstep at 1/32, or 2.5 µm at 1/16 (calc).
- 75×50 mm travel at 10–20 mm/s.
- A 1.14 mm field pitch takes 0.25–0.45 s including 0.1–0.2 s of settling (est.).
- Steps come from Pico PIO, because Linux GPIO stepping jitters (https://github.com/byogaratchagan/stepper_motor).
- Compare OpenFlexure: about 14 s per 0.5 mm step (https://ar5iv.arxiv.org/html/2109.06842).

**Tilt**
- Specimens are levelled to 0.2° or better with the lab's hand levelling press or parallel-lapped mounts.
- 0.2° across the 1.8 mm field at 10x is 6.3 µm, inside the 8.5 µm depth of field; 0.27° equals the full depth of field (calc).
- The autofocus z-map also gives part tilt.

**Thermal**
- Printed parts have a CTE of about 8.5e-5/K, about 8.5 µm/K over 100 mm (https://iris.unipv.it//retrieve/handle/11571/1450203/671919/J%20Optics%20Laser%20Technology.pdf). Aluminium is about 2.3 µm/K over 100 mm, which is still about DOF/4 at 10x per kelvin.
- So the column, head plate and autofocus block are aluminium; prints are PETG/ASA only.
- Compute and PSUs sit outside the optical enclosure.
- Temperature is logged from two DS18B20 probes (column and autofocus block), and the autofocus zero is re-checked when temperature changes by more than 1 K.

**Afocal mechanics**
- Lock the camera's focus ring at infinity with a drop of removable thread-lock or a printed clamp.
- Set infinity by focusing the camera on a reticle placed at the eyepiece field stop, which collimates it.
- The AI Camera is thicker than standard Pi cameras, so the collar is custom.
- Both camera ribbons get strain relief for the 30 mm Z travel, and they leave the enclosure through a light-tight brush gland.

---

## 4. Dark-field laser design

### 4.1 Geometry (calc; nose radius r = 5 mm assumed for 4x/10x and about 3 mm for 40x, measured in T10)

**Two conditions**
- **Dark-field:** incidence from the normal must exceed asin(NA) + 2·tilt + divergence.
- **Clearance:** elevation α ≤ atan(WD/r), *and* the top edge of the beam must pass under the nose. The beam's top edge at the nose radius sits at r·tanα + (b/2)/cosα.

| Objective | asin(NA) | α_max from WD | Top edge at r = 5 mm (α = 20°, 4 mm / 5 mm beam) | Verdict |
|---|---|---|---|---|
| 4x/0.10, WD 18–26 | 5.7° | 74–79° | 3.95 / 4.48 mm, far below WD | easy |
| 10x/0.25, WD 4.4 | 14.5° | 41° | 3.95 / **4.48 mm, clips** | use 15° (3.41 / 3.93 mm) or a 4 mm beam |
| 10x/0.25, WD 6.6–7 | 14.5° | 53–55° | 4.48 mm, about 2 mm margin | **chosen: buy WD ≥6.6 mm** |
| 40x/0.65, WD 0.53–0.6 | 40.5° | 10–11° (r ≈ 3 mm; 6–7° at r = 5 mm) | impossible | dark-field blocked |

**At 20° elevation (70° from the normal)**
- The footprint stretches 2.92x.
- Haze is about cos³70° = 0.040 of normal incidence.
- This matches KLA oblique practice: about 70° on the SP1 (https://patents.google.com/patent/US7218392) and 60–85° in a KLA dark-field patent (https://patents.google.com/patent/US9176072B2/en).

The collar has slots at 15/20/25° (30° only if CAD clears the LED ring, §4.7). A week-3 sweep chooses between haze, irradiance and clearance.

### 4.2 Sources and beam shaping

**Phase 1:** Robu 650 nm 5 mW, 5 V dot modules (about ₹10–13 each, https://robu.in/?p=16339).
**Phase 2, or Phase 1 if they win:** 520 nm direct-diode **dot** modules (not line modules), confirmed to run from the 5 V laser bus. Rated fallback: Picotronic DI520-1-3, 1 mW.

**Each module**
- Runs at nominal 5 V, which keeps the operating point stable.
- Carries a glued absorptive ND (650 nm: about ND 1.0 to start, ND 0.7 only if the post-ND reading is low) so that it measures ≤1 mW on a borrowed calibrated meter.
- Is logged in the `laser_power` table.

**Beam shaping**
- Defocus the module lens to about a 5 mm (1/e²) Gaussian spot at the focal point.
- Peak irradiance at the field centre: 102 W/m² in the beam and 34.8 W/m² on the surface per mW (calc).
- Field edge: about 77% of peak at 10x, but about 20% at 4x, so per-azimuth flat maps are needed.
- No static diffuser (it would imprint speckle). No Powell lens (this is area imaging).

**Polarisation:** each module is rotated to a set polarisation and checked with polariser film. Baseline is s-polarisation. Week 3 compares s with p on a mirror and on polished steel. On a mirror, s-polarised light has a near-node in the field at the surface, so p tends to raise the signal from small particles, while s tends to lower haze on rough metal.

**Beam dumps**
- Flocked V-traps at 180/300/60° receive the specular beams.
- Matte-black collar, no shiny fasteners in the beam plane.
- Dumps are installed before any reflective part is loaded.

**Phase-2 options**
- Rotating ground-glass diffuser (48% throughput, more than about 400 rpm; https://arxiv.org/abs/1703.05311v2).
- Crossed analyser film (residual circular polarisation from metal limits it: https://patents.google.com/patent/US11493454).

### 4.3 Azimuths and the physics classifier

**Why the beam angle matters.** A long scratch scatters light onto a cone around its own axis, so it sends light up into the objective only when the beam crosses it roughly at right angles. Particles and pits scatter from every azimuth (https://patents.google.com/patent/US9841384). With beams at 0°, 120° and 240°, every scratch direction has a beam at least 60° away from it (calc), which meets the rule that light should be more than 30° from the scratch direction (https://ebooks.spiedigitallibrary.org/profile/Mingjie.Deng-4237696).

**CPU features per blob** (before any network):
- S = Σ I_k
- Z2 = Σ I_k·e^{i2φ_k}
- **A = |Z2|/S ∈ [0,1]**. It is 0 for isotropic scatterers and 1 when only one beam lights the feature.
  - Example: I = (1, 0.2, 0.2) gives A = 0.57.
  - Placeholders: particle-like A < 0.15; scratch A ≥ 0.35 with L/W ≥ 5.
- **Scratch orientation = ½·arg(Z2) + 90°.**
- **Z1 = Σ I_k e^{iφ_k}**, a forward/back asymmetry used as a bump vs pit hint.
- **Persistence:** the blob must be present in 2 or more frames (airborne dust moves).
- **Speckle check:** the summed image averages three independent speckle patterns, cutting speckle contrast by about √3 (calc). A blob that appears in only one azimuth *and* is round is down-weighted as speckle.

### 4.4 Wavelength decision

| λ | For | Against | Decision |
|---|---|---|---|
| 650 nm | ₹10 and in stock now | Sits on the IR-cut roll-off (HQ/GS filter about 50% at 600 nm and about 0 by 700 nm; no curve is published for the AI Camera: https://pip-assets.raspberrypi.com/categories/810-raspberry-pi-global-shutter-camera/documents/RP-008148-DS-1-camera-extended-spectral-sensitivity.pdf). Reaches only R pixels (¼ of the array). | Bench, alignment and the autofocus laser. Phase-1 dark-field **only if** test T2 shows usable R response at ≤1 mW. |
| **520 nm direct-diode** | Falls on G pixels (½ of the array), close to the 546 nm achromat design, away from the IR-cut edge | Import or third-party sourcing. The Amazon.in listing at about ₹3,027 is a *line* module, low confidence, and not suitable unless a dot version is offered: https://www.amazon.in/520nm-Laser-Module-Industrial-adjustable/dp/B0CSYSVPS9 . Rated Picotronic DI520-1-3 €129.33: https://www.picotronic.de/picopage/en/product/detail/id/357261 | **Production dark-field source (dot modules).** Order in week 1; reject any unit that shows IR on the OV9281 behind the IR-pass filter. |
| 405 nm | λ⁻⁴ gain only below about 0.3 µm (A2 Mie) | Blue-light hazard (https://cie.co.at/node/7261), ₹2,599+ for 5 mW (https://price-history.in/product/laser-diode-module-violet-purple-05nm-ZxugQAce), unknown IMX500 response, achromat focal shift | Rejected |
| 532 nm DPSS | — | IR leakage (NIST TN 1668) | Rejected |

### 4.5 Photon budget (calc; order of magnitude only)

**Model and assumptions**
- Free-space PSL sphere (n = 1.59) Mie scattering, s-polarised.
- 70° incidence; collection over the objective cone.
- 1 mW, 5 mm Gaussian spot, field centre, 30 ms frame.
- Optics: objective 0.85 × beamsplitter 0.5 × eyepiece 0.9 × camera lens 0.9.
- **Assumed** IR-cut transmission: 0.30 at 650 nm, 0.90 at 520 nm.
- **Assumed** area-averaged Bayer QE: 0.15 at 650 nm (R only), 0.33 at 520 nm.
- Resulting η = 0.0155 (650) and 0.102 (520).
- My Mie code reproduces the Bohren–Huffman reference (Q_sca 3.105 for m 1.55, x 5.213).
- Substrate effects are ignored; a mirror can change the signal by up to a few times. For s-polarisation, the near-node at a mirror surface means the 0.2–0.5 µm columns are probably overestimated on metal and Si.

**Electrons per 30 ms frame**

| | 0.2 µm | 0.3 µm | 0.5 µm | 1 µm | 3 µm | 10 µm |
|---|---|---|---|---|---|---|
| 10x, 650 nm | 8 | 36 | 66 | 229 | 2,066 | 8,834 |
| 10x, 520 nm | 77 | 74 | 767 | 2,016 | 6,060 | 33,976 |
| 4x, 650 nm | 1 | 6 | 11 | 48 | 458 | 1,010 |
| 4x, 520 nm | 12 | 10 | 130 | 300 | 483 | 6,289 |

The values are non-monotonic because of Mie resonances near 110° scattering.

**Comparison with the two designs**
- Design A's table is about 4x higher at 650 nm, because it left out the Bayer fill factor.
- Design B's figure of about 13,000 e⁻ for 1 µm is about 6x too high.

**Why 520 nm wins:** at 1 µm it gives about 9x the electrons of 650 nm, but only about 1.7x comes from scattering physics. The rest is the assumed camera response, so **T2 decides**.

**Haze per binned pixel** (Lambertian, TIS = (4πσcosθ_i/λ)², 1 mW). The same at 4x and 10x, because NA²·p² is nearly equal.

| RMS roughness σ | 650 nm | 520 nm | Example surface |
|---|---|---|---|
| 0.25 nm | 0.02 e⁻ | 0.2 e⁻ | HDD platter, Ra 0.13–0.25 nm (https://americas.fujielectric.com/files/r53-2-02.pdf) |
| 1 nm | 0.4 e⁻ | 3.2 e⁻ | |
| 3 nm | 3.5 e⁻ | 29 e⁻ | polished metallographic steel, fully speckled (grain λ/2NA ≈ 1.0 µm, below one 1.64 µm binned pixel at 10x) |

**Photon-limited floor.** Assuming about 3 e⁻ read noise (unpublished assumption) and a 5σ threshold of about 15 e⁻:
- 10x: about 0.3 µm at 650 nm, about 0.2 µm at 520 nm
- 4x: about 1 µm at 650 nm, about 0.5 µm at 520 nm

**The real floor** is set by speckled haze on metal, epi glare, glints, airborne dust and false calls. KLA itself calls speckle the main limit (https://patents.google.com/patent/US10067072). A2's finding governs: no DIY Pi dark-field build has a published measured limit.

### 4.6 Expected detection limits (targets; publish only measured POD)

| Surface | 10x target | 4x target | Not claimed |
|---|---|---|---|
| HDD platter, Si, glass | ≥0.5–1 µm scatter-equivalent (stretch 0.3) | ≥2–3 µm-eq | widths below 2.7 µm |
| Mirror-polished stainless, unetched polished metallographic sections | ≥2–5 µm-eq | ≥5–10 µm-eq | — |
| Etched, grainy, matte, solder mask, PCB | — | — | dark-field not used; bright-field or LED ring instead (https://averroes.ai/blog/dark-field-inspection) |

**Precedents**
- 60 nm PSL with a 405 nm laser, NA 0.45 and sCMOS camera: https://pmc.ncbi.nlm.nih.gov/articles/PMC12944180/
- SDES LED-ring dark-field at about 0.5 µm: https://opg.optica.org/ao/abstract.cfm?uri=ao-63-25-6686
- 2 µm scratches and 4 µm pits at a 7 µm pixel: https://doi.org/10.3390/photonics11070681
- Samsung LED dark-field, scratches over 1 µm wide at about 8–11 µm/px: https://www.scitepress.org/Papers/2021/102233/102233.pdf

**Measurement protocol**
- **Size:** reported as scatter-equivalent µm, from 1/3/9 µm diamond-paste calibration slides (₹260 per 5 g: https://m.indiamart.com/met-hr-tech/diamond-paste.html) on Si and glass, normalised to a daily reference reading.
- **POD:** N ≥ 30 sites per size and substrate.
- **False calls:** per cm² on clean references.
- **Dig number:** diameter in 0.01 mm when the dig is resolved (https://www.edmundoptics.com/resources/application-notes/optics/understanding-surface-quality-specifications/).

### 4.7 LED-ring baseline (the no-laser mode the lasers must beat)

- **Ring:** WS2812B, 24 LEDs, 86 mm across (C$3.99–£9.10, https://www.universal-solder.ca/product/24-led-ring-ws2812b-rgb-addressable-86mm/).
- **Geometry:** mounted on the collar about **26 mm** above focus. The LEDs sit at r ≈ 36–43 mm, giving about 33° elevation (calc); this is still dark-field at 10x (57° incidence > 14.5°). The ring is shrouded so it does not light the objective barrel.
  - *Why not 12 mm:* a 20° laser beam 5 mm wide crosses the 12 mm plane between r = 25.7 and 40.3 mm (calc), so it would clip the ring and its shroud. No single ring height clears every slot from 15° to 30°. At 26 mm the 15–25° beams clear; 30° clears only if CAD confirms. The calculation assumes the common 86/72 mm ring, so confirm the real ID in CAD.
- **Modes:**
  - full ring
  - single quadrants, as on the Evident MX63 (https://evidentscientific.com/en/products/semiconductor-inspection/mx63-mx63l)
  - RGB-coded azimuths (Rheinberg)
- **Drive:** channel values are only 0 or 255, driven from Pico PIO.
- **Report ablation:** laser dark-field vs LED-ring dark-field vs bright-field on the same fields, as background σ and defect SNR.

---

## 5. Autofocus

### 5.1 Why laser autofocus

- **Image autofocus fails on our parts.** Passive contrast autofocus fails on bare wafers (https://evidentscientific.com/en/learn/white-papers/bxc-series-modular-microscope-assembly-focus-sensing-unit), and in dark-field a clean field is black.
- **It is also slow.** OpenFlexure image autofocus takes 4.1 s (JPEG size) or 14.9 s (Laplacian) and fails on sparse samples (https://ar5iv.arxiv.org/html/2109.06842).
- **Nanometre locking is not needed.** PiFocus gets 2 nm with a piezo on coverslips only (https://www.biorxiv.org/content/10.1101/2024.01.15.575442v1.full); our depths of field need about a micron.
- **Commercial accuracy:** about 20% of depth of field (https://www.prior.com/product/purefocus850-laser-autofocus-system), or ±¼ of depth of field static (https://m.yiqi.com/zt6206/product_285052.html).

### 5.2 Accuracy needed vs depth of field (calc)

**Inputs:** Phase 1 holes at ±h = 0.9 mm with an f_L = 75 mm lens, so u′ = 0.012 rad and sinθ = M·u′. Phase 2 uses the same geometry with 3 µm OV9281 pixels at the image-conjugate plane.

**Bias (new; essential).** If the two beams cross exactly at the focal plane, they reflect back to one spot at best focus. Inside the ±DOF/4 lock window they would be less than 1 px apart, inside spots 15 px across, so they cannot be fitted separately. The AF lens therefore focuses the beams about 30 mm (image space) away from the field-diaphragm conjugate, so they cross z_c = 30 mm/M² from best focus (calc).

| Objective | z_c | s0 at best focus |
|---|---|---|
| 4x | 1.9 mm | about 44 binned px, about 3 spot diameters |
| 10x | 0.30 mm | about 44 binned px |
| 40x | about 16 µm | about 44 binned px |

- s0 is about the same for every objective because both s0 and the spot size scale as 1/M.
- At best focus the surface is imaged, and the spots sit where the beams strike it, so the zero stays independent of tilt and pointing.
- The spots merge only at z = z_c/2 (0.15 mm at 10x), which is outside the lock range. The unequal holes tell which side of it the surface is.

**Formulas**
- Spot separation s = 2·tanθ·|2z − z_c|, where z is the surface height from best focus. At best focus this gives s0 = 2·tanθ·z_c.
- Slope |ds/dz| = 4·tanθ, the same as without the bias.
- Phase 1 slope: 4tanθ/p_obj.
- Phase 2 slope: 4·M·tanθ / 3 µm.
- σz = centroid noise / slope (valid because the spots are resolved).

| Objective | DOF | Target ±DOF/4 | sinθ (% of NA) | P1 slope, binned (full) | P1 σz at 0.07 / 0.2 px | P2 slope (OV9281) | P2 σz at 0.07 px | Capture range P1 / P2 (est., asymmetric, measured in T4) |
|---|---|---|---|---|---|---|---|---|
| 4x | 55.5 µm | ±13.9 µm | 0.048 (48%) | 0.047 (0.094) px/µm | 1.5 / 4.3 µm | 0.26 px/µm | 0.27 µm | about ±2 mm / ±2.5 mm |
| 10x | 8.5 µm | ±2.1 µm | 0.120 (48%) | 0.296 (0.591) | 0.24 / 0.68 µm | 1.61 | 0.04 µm | about ±0.3 mm / ±0.4 mm |
| 40x | 1.0 µm | ±0.25 µm | 0.480 (74%) | 5.35 (10.7) | 0.013 / 0.037 µm | 29.2 | 0.002 µm | about ±45 µm / ±22 µm |

**Reading the table**
- Phase 1 meets the target with a 3–9x margin even at a pessimistic 0.2 px centroid noise, now that the spots are resolved at best focus.
- The 256 px ROI alone would allow about ±5 mm, ±0.9 mm and ±48 µm. The quoted ranges are conservative until T4 measures them.
- The practical floor is the 0.31 µm Z quantum, plus thermal drift and the chromatic offset.
- At 40x, autofocus is followed by a 5-slice z-stack at 0.31 µm and best-focus selection.

### 5.3 Phase 1 (by 30 Nov): visible Scheiner autofocus through the epi port, read by the AI Camera

**Beam path**
1. 650 nm module with a fixed **ND2** cap, so the module emits ≤50 µW (Class 1 by power at the module aperture).
2. **ND slider** with ND0/1/2/3 positions.
3. Two-hole mask, drilled in 0.1 mm blackened shim, holes at ±0.9 mm. Diameters of 0.5 and 0.35 mm make the two spots distinguishable, which gives the sign of defocus.
4. Lens, f = 75 mm, focused about 30 mm off the field-diaphragm conjugate (the bias). It reaches the epi path via a **#1.5 coverslip pick-off** (or a wedged plate).
5. 50/50 beamsplitter, then the objective.

**Pick-off and power**
- A pick-off reflects off both surfaces (about 4% each), so it sends about 8% of the AF beam on.
- With a 1 mm slide, the back-surface ghost would be displaced about 0.75 mm (about 45 binned px at 10x). That would put a second, equally bright spot pair in the fit ROI.
- A 0.17 mm coverslip keeps the ghost within about 8 px. The ghost is common-mode, so it cancels in the separation.
- The transmitted part goes into a dump.
- With the slider at ND0, about 2 µW or less reaches the objective: Class 1 by a wide margin.

**Brightness and exposure (calc, order of magnitude; IMX500 full well unpublished)**
- With about 1 µW at the objective, a mirror return at the fixed 30 ms exposure gives about 1.4e9 e⁻ in the two spots. They can hold only about 1e6 e⁻, so they would be about 10³ times saturated.
- Therefore exposure and gain stay fixed per session, and spot brightness is ranged with the ND slider plus ≥20 kHz hardware PWM on GPIO13.
- Starting slider settings: about ND3 for a mirror, about ND2.5 for Si (32%), about ND1.5 for glass (4%). They are verified in T4.
- PWM and slider set brightness only. The laser class is already set by the ND2 cap.

**Spot quality**
- Sub-aperture NA is M·0.25/75: 0.033 at 10x.
- Airy diameter is 24 µm, about 15 binned px. It is about 15 px at every magnification (calc), which is good for Gaussian centroiding. With the bias, the spots are about 3 diameters apart at best focus.

**Why two spots**
- Pointing drift of the mount moves both spots together, so it cancels.
- At best focus the surface is imaged and the spots sit where the beams strike it, so the zero point does not depend on tilt; tilt only changes the gain away from focus.
- The return vignettes at roughly ±1.5° (4x), ±3.7° (10x) and ±5° (40x). These are centre-ray estimates from (asin NA − θ)/2, lower if beam-edge clipping is counted, and measured in T4.
- A single-spot design would read 2 µm of drift at the sample (10x) as 8.3 µm of defocus (calc).

**Control loop**
1. Turn off every source except AF650.
2. Accept the first frame that passes the timestamp gate (§6).
3. Subtract the stored autofocus-ghost frame.
4. Fit the two (resolved) Gaussians in a 256 px ROI.
5. Compute Δz = (s − s0)/g.
6. Move Z, finishing the approach from below.
7. Run one verify read, then turn AF650 off.

Two iterations take 0.19–0.30 s (calc). The logic follows the ImSwitch focus lock (centroid plus PI control, handles double reflections): https://imswitchopt.readthedocs.io/en/latest/_sources/gui.rst.txt. Each new field is pre-positioned from a plane fit to earlier fields, which also produces the flatness/tilt map.

### 5.4 Phase 2 (December–January): dedicated 850 nm arm on CAM1

**Source**
- 850 nm module, preferably maker-rated ≤0.5 mW (Class 1).
- If only 5 mW modules can be sourced:
  - The bare module is **Class 3B**. The Class 3R AEL at 850 nm is 5×3.9e-4·C4 W ≈ 3.9 mW (calc).
  - A permanent NIR-rated **ND1** (OD specified at 850 nm; never gel or plastic) is bonded on before normal use.
  - Until then the module is handled only inside a closed tube, by the Principal Laser User, with EN 207 eyewear worn.
- Result: ≤0.5 mW at its own aperture and ≤0.05 mW after the 8% injection plate.
- The Class 1 AEL is 3.9e-4·10^(0.002(λ−700)) W = **0.78 mW at 850 nm** (calc: https://www.ieee802.org/3/ae/public/mar00/kolesar_1_0300.pdf). This matches PureFocus850 at <0.77 mW, Class 1 (https://microscopeworld.com/content/brochures/PureFocus850_Datasheet.pdf).
- There is no Class 2 in the IR.

**Combiner**
- **Edmund #69-196**: 800 nm shortpass, 12.5×17.6×1.05 mm, $145 (https://www.edmundoptics.com/p/800nm-125-x-176mm-dichroic-shortpass-filter/23538/).
- It transmits 520 nm and 650 nm and reflects 850 nm. The 50 nm margin is far more than diode drift (about 0.06 nm/°C plus 0.3 nm mode hops: https://www.newport.com/medias/sys_master/images/images/h4b/h48/8797049585694/AN08-Mode-Hopping-in-Semiconductor-Laser-Diodes.pdf).
- **Use 850 nm, never 780 nm**, with this part.
- Alternatives:
  - Edmund #43-955 25 mm hot mirror, $53, if its thickness is ≤1.5 mm (https://www.edmundoptics.com/p/25mm-diameter-45deg-aoi-hot-mirror/5424/).
  - The Edmund India ₹10,121 shortpass has a 600–700 nm cut, which would block the 650 nm fallback (https://www.edmundoptics.in/f/PdfExport/14198).

**Arm** (on one aluminium block)
- Injection: 850 nm module → two-hole mask → f = 75 mm lens (crossing biased like Phase 1) → **wedged** uncoated plate (about 8% reflected for injection) → dichroic → objective. The roughly 92% that passes the plate goes into a flocked dump inside the block.
- Return: back through the plate (92% transmitted) → 800 nm longpass (FELH0800 class) → **Arducam OV9281** at the image-conjugate plane, about 130 mm of folded optical path.
- Spots at the image plane are about 100 px across (calc: image-side sub-aperture NA 0.0033 at 850 nm). So the bias is set for s0 of about 3 spot diameters (about 300 px), and capture is asymmetric.
- Exposure on the OV9281 is set freely, in the tens of µs at 0.05 mW.
- OV9281: mono NoIR, 1280×800, 3 µm pixels, 120 fps, ₹5,841 (https://zbotic.in/product/arducam-ov9281-1mp-global-shutter-monochrome-noir-camera-module-with-m12-mount-lens-for-raspberry-pi-4-3b-3/). The M12 lens is removed so the bare sensor sits at the conjugate plane. Enable with `dtoverlay=ov9281,arducam` on CAM1 (https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/Global-Shutter/1MP-OV9281-OV9282/).

**Loop**
- PI control at 60–120 Hz on an ROI; PiFocus reaches about 150 Hz at 300×300 on a Pi 4 (https://github.com/PonjavicLab/PiFocus).
- Z converges during the XY settle.
- It frees about 0.2 s of AI Camera frames per field.
- It builds a dense focus and flatness map.
- **Gate G4:** if Phase 2 σz is not at least 2x better than Phase 1 on test materials, Phase 1 stays the primary autofocus.

### 5.5 Calibration (per objective, stored with its ID)

1. Find best visible focus by a Laplacian sweep at 1 µm steps under white bright-field on the stage micrometer. Record the biased autofocus separation s0 there. This absorbs the 650 or 850 nm chromatic offset; achromats are not corrected in the NIR.
2. Measure the gain g by stepping Z ±20 µm.
3. Capture the autofocus-ghost frame over a light trap.
4. Re-check at each session start and after a temperature change of more than 1 K.
5. Raise MAINTENANCE if s0 drifts by more than half the tolerance, or if the autofocus success rate falls below 95%.

### 5.6 Surface behaviour and fallback chain

**Expected behaviour by surface**
- **Polished metal and Si:** strong spots (ND slider high). Fresnel reflectance at about 850 nm is about 96% for Cu and about 32% for Si (calc: https://refractiveindex.info/?shelf=main&book=Si&page=Green-2008).
- **Glass:** about 4% per face and a second spot about t/n deeper (0.67 mm for a 1 mm slide, calc). Gate the expected Z range and lock on the first compact spot.
- **PCB:** chatter at step edges. Reject outliers; use multispot averaging later.
- **Matte surfaces:** weak, speckled spot. A spot-quality check (peak/background, FWHM, blob count) fails over, the way PureFocus disables itself.

**Fallback chain**
1. Laser Scheiner autofocus.
2. Reticle focus-finder contrast autofocus (incoherent; works with the key OFF).
3. Image autofocus (JPEG size or Laplacian) on bright-field.
4. Manual.

Every field logs `af_mode` and `af_err_um`.

---

## 6. Crosstalk, per-field sequence and timing

### 6.1 Defence in depth

| Layer | Phase 1 | Phase 2 |
|---|---|---|
| Geometry | Dark-field external at 70° from the normal, specular beams into V-traps. Autofocus enters through the objective. | same |
| Wavelength | None: autofocus and dark-field may both be 650 nm. **Time is the separator.** | Dark-field 520 nm, autofocus 850 nm, bright-field white |
| Filters | — | Hot mirror R > 90% at 850 nm; AI Camera integrated IR-cut ("not IR sensitive"); 800 nm longpass on the OV9281 blocks 520 nm and white light |
| Time | One source per frame. Autofocus is always off during image frames; lasers fire one at a time. | Autofocus may stay on during bright-field frames only if the week-9 leak test shows OD ≥4 at the AI Camera. Always off during dark-field frames. |
| Static ghosts | Autofocus-ghost frame; plates tilted slightly; coverslip or wedged pick-off | same, on the OV9281 |
| Stray light | Flocking, dumps, dark frame, per-session veil frame | same |

### 6.2 Frame gating (Pi cameras have no strobe output)

Only the HQ (XVS) and GS (XTR) cameras have sync pads (https://www.raspberrypi.com/documentation/accessories/camera.html), so gating is done in software:

1. Command the switch on the Pi GPIO, then record t_sw on the clock T6 confirms. Until then, log both CLOCK_MONOTONIC and CLOCK_BOOTTIME. The Pi maintainer says SensorTimestamp shares the monotonic clock; the libcamera proposal names CLOCK_BOOTTIME. The two are equal unless the system has suspended.
2. Accept a frame only if **SensorTimestamp − ExposureTime·1000 > t_sw + margin**.
   - The Pi maintainer says SensorTimestamp marks the start of readout (https://github.com/raspberrypi/picamera2/issues/787).
   - libcamera's wording says first-row exposure start (https://patchwork.libcamera.org/patch/12173/), so subtracting the exposure is the conservative choice for the start.
3. **Switch the next source only after SensorTimestamp + ExposureTime·1000 + FrameDuration** of the last accepted frame. This is conservative under both readings. Under the libcamera reading, the last rows keep exposing until about ts + 63 ms, so the faster rule (ts + FrameDuration) would contaminate them. Revert to the faster rule only if T6 confirms start-of-readout semantics. The conservative rule costs 0–1 frame per state.
4. Tag every frame, and its IMX500 tensor, with source state, power and timestamps. A tensor computed on a mixed-light frame is discarded.

**Fixed settings**
- ExposureTime, AnalogueGain and ColourGains are fixed per session and confirmed in the returned metadata.
- AE and AWB are off, because the black surround and dark-field frames make them hunt.
- Brightness is equalised at the source (PWM, ND slider).

**Sanity check:** a frame whose mean does not match its expected state is dropped.

**Pico commands** (ring, motion) are acknowledged over USB, plus a 5 ms margin.

### 6.3 I/O map

| Signal | Where |
|---|---|
| DF1–3 FET gates | Pi GPIO17/27/22 (100 Ω, 10 kΩ pull-down); GPIO23/24 reserved for DF4–5 |
| AF650 brightness | GPIO13 (PWM channel 1, ≥20 kHz) |
| AF850 enable | GPIO19 (PWM channel 3, used as a plain enable); the "IR ON" LED is wired across the switched AF850 supply, so it shows actual power |
| Bright-field LED | GPIO12 (PWM channel 0, ≥20 kHz). Three channels need a custom overlay; dtoverlay=pwm-2chan covers only one pair |
| Interlock read-back (opto, input only) | GPIO25 |
| Stack light R/A/G, buzzer (ULN2803 with 10 kΩ pull-downs on its inputs, because GPIO5/6 are pulled up at boot), ACK/OVERRIDE | GPIO5/6/16/26, GPIO20/21 |
| Temperature (2× DS18B20, 1-Wire) | GPIO4 |
| Watchdog heartbeat (Phase 2) | GPIO18 |
| Steppers X/Y/Z STEP/DIR, home switches, WS2812 data, BPW34 TIA (ADC) | Pico over USB. ENN is pulled high (driver disabled) by the motion relay K2 whenever the lid opens |

### 6.4 Session start

1. Confirm the interlock is closed.
2. Capture a dark frame.
3. Capture the veil frame.
4. Capture the autofocus-ghost frame.
5. Check dark-field power: BPW34 relative reading plus the 3 µm diamond reference slide.
6. Set the autofocus zero (biased s0) on the stage micrometer for each objective.
7. Log everything.

### 6.5 Per-field sequence (calc and est.; binned 2028×1520 at 30 fps, 30 ms exposure, 33.3 ms period)

Each light state takes 67–100 ms with the fast switch rule (one frame discarded and one accepted), or up to about 133 ms with the conservative rule.

| # | Step | Sources on | Phase 1 | Phase 2 |
|---|---|---|---|---|
| 1 | XY move (1.14 mm pitch) + settle | none | 0.25–0.45 s | same, with AF850 converging during the settle |
| 2 | Autofocus, 2 iterations | AF650 only | 0.19–0.30 s | about 0–0.1 s extra |
| 3 | Bright-field | epi LED | 0.07–0.10 s | 0.07–0.10 s |
| 4 | DF1, DF2, DF3 | one laser each | 0.20–0.30 s | 0.20–0.30 s |
| 5 | Ring dark-field (optional) | ring | +0.07–0.10 s | same |
| 6 | Dark frame | none | every 10 fields | same |
| 7 | Hand-off to CPU and Hailo | — | asynchronous, overlapped with the next move | same |
| | **Field cycle** | | **0.71–1.15 s** (fast rule; up to about 1.3 s with the conservative rule) | **about 0.5–0.95 s** |

- **Measurement fields** (nodularity, sizing a flagged defect) add a full-resolution bright-field still at 10 fps: 0.2–0.3 s plus the mode switch, measured in test T6. If the switch takes more than 1 s, measurement runs as a second pass.
- **Throughput (calc):**
  - 10×10 mm at 10x is 81 fields: about 58–93 s (up to about 107 s with the conservative rule).
  - A 4x survey of the same area is 16 fields: about 11–18 s, then 10x review of flagged fields only.
  - OpenFlexure-class motion would take about half an hour.

---

## 7. Compute pipeline

```
 DOMAIN 1  IMX500 ON-SENSOR NPU (AI Camera, CAM0)           [optional; verdict never depends on it]
   FieldGate: YOLO11n-cls 224, INT8 .rpk (<= 8,388,480 B), ROI = inscribed square (set_inference_roi_abs)
   classes: no_sample | featureless_blank | out_of_focus | under/over_exposed | usable_clean | usable_suspect
   -> tensor metadata per frame (~0 host CPU) + image stream
   (if the organisers reject the IMX500: the same model compiled to a HEF runs on Domain 2)
        | P1 capture/sequencer (venv A, core 0): picamera2, GPIO lights, AF loop, Pico serial
        v  shared-memory ring
 CPU core 1 (P1b): dark/veil/flat, Bayer plane (R in P1 / G at 520), log-compress, classical DF detector
   (bg + k*sigma, connected components), physics features S, A, orientation, Z1, persistence,
   calibrated um measurements, E2567 nodularity
        | blob crops + tile IDs (ZeroMQ ipc)
        v
 DOMAIN 2  HAILO-8 (AI HAT+ 26T, PCIe) via HailoRT 4.23 async, one VDevice + scheduler (P2, venv B)
   H1c (P1): ResNet-18 crop classifier, 128x128x3 uint8 = DF az0/120/240        [always in verdict path]
       fallback: precompiled Model Zoo backbone HEF as crop embedder + CPU linear head
   H1s (P2): yolov8n-seg on 640 tiles (2x2 binned circle, 4x4 full-res), SAHI-style overlap
   H2  (P2): U-Net-MobileNetV2 BF artefact rejection (nodularity) / grain boundaries (Lab)
   H3  (P2 opt): yolov8s PCB at 4x BF                                   -> detections JSON
        v
 CPU: YAML rule engine (physics features + H1 class) -> severity -> GPIO stack light/buzzer, kiosk UI, SQLite (WAL)   <- never waits for LLM
        | event (ZeroMQ)
        v
 DOMAIN 3  ARM CORTEX-A76 cores 2-3 (P3, venv C, taskset, nice +5)
   ExecuTorch 1.5.1 XNNPACK(+KleidiAI), Llama 3.2 1B Instruct SpinQuant .pte (TextLLMRunner / llama_main)
   + SQLite FTS5 over team-written SOP summaries -> slot template -> numeral validator -> template fallback
 Phase 2: OV9281 (CAM1) AF loop 60-120 Hz on core 0 (CPU, no NPU)
```

These are three processes in three venvs, reusing the BeamTriage split, because the numpy and torch versions conflict.

**Domain 1 notes**
- Ultralytics `format="imx"` export of YOLO11n-cls runs at 33.31 ms per frame and needs x86-64 Linux with Java 17 (https://docs.ultralytics.com/integrations/sony-imx500/).
- Training data: about 200–300 frames per class from the AI Camera dataset tool (https://www.raspberrypi.com/news/streamline-dataset-creation-for-the-raspberry-pi-ai-camera/).
- Used only on bright-field and ring frames, selected by state tag.
- Never used for subtraction or fusion, because it sees single frames (A2).
- Phase-1 status: stretch goal, first in the descope order (§13).

**Domain 2 notes**
- The Phase-1 crop classifier needs only per-blob labels and uses robust Model Zoo ResNet recipes.
- The 3 azimuths fit exactly 3 uint8 channels. That avoids both the 16-bit-input DFC crash (https://community.hailo.ai/t/compiling-with-16-bit-inputs-crashes/19078) and the risk of more than 3 channels.
- Build: DFC 3.33 on native x86-64 Linux (lab PC or cloud) to match HailoRT 4.23, with ≥1,024 in-domain calibration crops; compare the quantised HAR against the HEF early.
- **PS5 rule:** a HailoRT model stays in the verdict path whatever happens. If the custom HEF fails at G3, the precompiled Model Zoo backbone runs on HailoRT as an embedder with a CPU linear head. CPU-only is a benchmark configuration, never the demo.
- Zoo speeds: yolov8n_seg 528 FPS, yolov8s_seg 107 FPS. These depend on the host (https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_instance_segmentation.rst). Plan with a much lower Pi 5 application rate (est., from earlier BeamTriage work), measured with hailortcli and the real app in T9 and week 5.

**Domain 3 notes**
- Export with `export_llm` and `llama_xnnpack_spinquant.yaml`, max_context_length 1024 (https://docs.pytorch.org/executorch/main/raspberry_pi_llama_tutorial.html).
- Llama speed on a Pi 5 is unpublished. Reference points: a OnePlus 12 ran 1B SpinQuant at 50.2 tok/s with 0.3 s TTFT and 1,921 MB RSS (https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md). On a Pi 5, an older ExecuTorch post measured Llama3-8B at about 2 tok/s and Stories110M at 31.6 tok/s ("not a rigorous benchmarking effort").
- Budget: t = TTFT + N/r. Prompt ≤350 tokens, output ≤80–120 tokens, ≤15 s per part, asynchronous, cached by (rule_id, class, zone). On 2 threads, prefill of 350 tokens may dominate. If T8 shows that, run the advisor on 4 threads between parts, when vision is idle.

**Latency budget (est.; measured in weeks 1, 5 and 6)**

| Stage | Domain | Time |
|---|---|---|
| FieldGate, live | IMX500 | 33 ms per frame, on sensor |
| Autofocus, Phase 1 / Phase 2 | CPU + camera | 0.19–0.30 s / about 0.05–0.1 s overlapped |
| Capture, 4–5 states | camera | 0.3–0.5 s (up to about 0.65 s with the conservative switch rule) |
| Preprocess, per binned frame | CPU core 1 | 10–30 ms |
| Classical dark-field + physics | CPU core 1 | 10–40 ms |
| H1c, about 50 crops, batched | Hailo | 30–60 ms (est.) |
| Rules, GPIO, SQLite | CPU | <20 ms |
| **Verdict after the last frame** | — | **≤0.15–0.3 s** |
| Llama advice per part | cores 2–3 | ≤15 s target, asynchronous |

**Benchmark matrix for the report**
- CPU-only, with H1c on ExecuTorch XNNPACK INT8. The Arm model card gives about 8.75 ms per 224 crop for ResNet-18 (device as stated on the card; re-measure on the Pi 5: https://huggingface.co/Arm/resnet-18-int8-xnnpack-executorch). That is about 0.15–0.45 s for 50 crops (calc).
- IMX500 + CPU.
- IMX500 + Hailo + CPU.
- Report fps, latency histograms, power, temperature and get_throttled, plus Llama tok/s with vision idle vs busy.
- As a proxy only: the 1st-edition PS5 (an FPGA problem statement) asked for at least a 2x speed-up over CPU-only (https://arm-education.github.io/Arm-Developer-Labs/Challenge_Page.html).

**CPU-only fallback:** one config flag, for benchmarking. The classical detector plus physics classifier is a complete path from week 4, but the demo always runs with a HailoRT model in the verdict path.

---

## 8. QC demos, defect classes, severity rules, alerts, logging

### 8.1 Demos (all manufactured specimens)

**D1. Mirror-finish part QC (headline; Phase 1, extended in Phase 2)**

Specimens:
- lapped and polished 304 stainless seal-face coupons and washers
- polished Cu and Al
- e-waste HDD platters
- Si wafer pieces and dies

Video beats:
1. Image autofocus hunts on a blank mirror while the laser autofocus locks in about 0.3 s.
2. A field that looks clean in bright-field lights up under laser dark-field.
3. Three-azimuth colour coding shows each scratch's orientation and anisotropy.
4. Wipe and re-image: particles vanish and fixed defects stay (the Samsung labelling trick).
5. Verdict, stack light, Llama action, SQLite log.
6. Flatness map from the autofocus z values.

**D2. Polished metallographic section QC (Phase 1 basic, Phase 2 validated)**
- Unetched ductile iron at 10x bright-field, ASTM E2567 nodularity and nodules/mm². At least 500 particles are required, and PASS/FAIL uses a spec the user enters, because the standard sets no limits (https://store.astm.org/e2567-16a.html).
- **Dark-field prep gate:** the bright-field image segments the graphite nodules, which are masked out of the dark-field image. Then scratches, pull-outs, comet tails and embedded abrasive are counted on the matrix. Too many gives **RE-POLISH**, not FAIL.
- Story: India is the #2 casting producer, with about 90% MSMEs.

**D3. Electronics (Phase 2)**
- Si die or wafer-piece particle and scratch map: a 4x survey plus 10x review, as a microscope-scale "mini-Surfscan" (https://wiki.nanofab.ucsb.edu/wiki/Surface_Analysis_(KLA/Tencor_Surfscan)).
- Die-edge chipping in bright-field.
- PCB and solder inspection at 4x under bright-field and the LED ring, with no laser. Overkill and escape rates are reported as SolderNet does (https://arxiv.org/pdf/2211.10274).

**Lab mode:** E1382 grain size on etched steel, under LED.

### 8.2 Recipe zones

- **A, critical:** seal face, data zone, active area or measurement area.
- **B, cosmetic.**
- **E, 1 mm edge band:** masked for dark-field point defects because of edge glints; judged from bright-field.

Curved or filleted parts are kept out of the laser demo, because their glints are about 100x brighter than the scatter from a 1 µm scratch.

### 8.3 Defect classes and default severity

**Levels:** S0 PASS · S1 minor (logged) · S2 major (clean or rework, then re-inspect) · S3 critical (scrap or quarantine).

Thresholds are placeholders in versioned YAML, tuned with a mentor. Every decision logs its rule ID.

| Class | Recognised by (physics + H1) | Default rule |
|---|---|---|
| particle | A < 0.15; gone after an air-blow or wipe and re-image | S1 CLEAN. S2 CLEAN+REINSPECT if >20/mm² in zone A or any >10 µm-eq in zone A. Reclassified as pit or embedded if it survives cleaning. |
| fiber | elongated, curved, removable | S1 CLEAN |
| scratch | A ≥ 0.35, L/W ≥ 5, survives cleaning; graded R1–R5 against our own scribed ladder | Zone A: ≥R3 or length ≥0.5 mm → S3 on seal, data or active areas, else S2 RE-POLISH. Zone B: S1 (S2 if ≥R4). |
| pit / dig | isotropic, survives cleaning; diameter measured if ≥2×Rayleigh, otherwise µm-eq | Zone A: ≥25 µm → S2; ≥100 µm → S3 |
| crack | linear or branching, very high scatter, also seen in bright-field | always S3 QUARANTINE + REFER (cross-section or SEM) |
| stain / film | diffuse low-A haze >0.05 mm² | S1 CLEAN (fingerprint pattern adds a glove hint) |
| pull-out / comet tail | dark-field tail along the polish direction plus a bright-field pit | S2 RE-POLISH (lower load, fresh cloth) |
| embedded abrasive | bright isotropic point that survives ultrasonic cleaning (lab ultrasonic bath), on soft metal | S2 RE-POLISH |
| Si edge chip | bright-field outline | >50 µm → S2; entering the active area → S3 |
| speckle / glint nuisance | one azimuth, round, not persistent, or in zone E | not counted; logged for false-call statistics |
| unknown | high scatter, no confident class | S2 HOLD + optional REFER-TO-SEM-EDS via I-STEM |
| nodularity / count | H2 mask (Phase 2) or threshold (Phase 1) + shape factor on CPU | Below spec: S2 FAIL (check the Mg treatment and inoculation log). <500 particles: INSUFFICIENT. Prep gate failed: RE-POLISH. |
| grain size G (Lab, Phase 2) | E1382 intercepts; G = −6.643856·log10(l̄) − 3.288 | Outside spec ±0.5: S2. <5 fields or <400 intersections: INSUFFICIENT (https://www.metkon.com/what-is-astm-e112-a-practical-guide-to-measuring-average-grain-size/) |

**Measurement-honesty rule**
- No part fails on a reading below the *measured* detection limit; such readings are logged as "indicative".
- No width or depth is claimed for sub-pixel defects.
- The rules borrow MIL-PRF-13830B and ISO 10110-7 concepts and are not a certified scratch-dig test.

### 8.4 Lot and process flags

- **LOT HOLD:** 3 or more same-class S2+ defects in the last 50 parts.
- **PROCESS-SCRATCH:** at least 60% of scratches in the last 20 parts within ±10° of one orientation. This points to handling damage, or the polishing direction not being rotated.
- **CLEANLINESS:** rolling particle density doubles against baseline.
- **FLATNESS:** autofocus z-map peak-to-valley or tilt over spec.
- **MAINTENANCE:** any of these:
  - haze on the clean reference up by more than 30%
  - dark-field reference power down by more than 20%
  - autofocus s0 drift above half the tolerance
  - autofocus success rate below 95%

### 8.5 Alerts (BeamTriage stack light via ULN2803)

| Pattern | Meaning |
|---|---|
| Green | PASS |
| Amber | CLEAN, REWORK or RE-POLISH |
| Red + 1 s buzz | SCRAP or QUARANTINE |
| Red flashing | LOT HOLD until supervisor ACK |
| Amber slow blink | MAINTENANCE |

OVERRIDE requires a reason, and every override becomes a training label.

### 8.6 Logging (SQLite WAL, hash-chained rows; BeamTriage schema extended)

- `inspections`: part, lot, severity, disposition, ruleset_ver, rpk/hef/pte hashes, latency per domain
- `fields`: xy_mm, objective_id, calib_id, um_per_px, af_mode, af_z_um, af_err_um, haze_index, temperature
- `frames`: source_id, azimuth_deg, power_rel, exposure_us, sensor_ts, clock_id, state_ok
- `defects`: class, A, orientation, I_az1..3, length_um, width_um (or null), scatter_eq_um, zone, rule_id, severity
- `measurements`: method, value, CI, n, fields
- `calibrations`
- `laser_power`
- `interlock_events`
- `advice`: ttft_ms, tok_s, validator_ok, fallback_used
- `alerts`
- `model_registry`
- `system_health`

**Exports:** CSV per lot; PDF/HTML certificate of inspection. No data leaves the box.

---

## 9. Academic-lab features and the Llama assistant

### 9.1 Lab mode (second tab, lockable)

**Imaging modes**
- epi bright-field with field and aperture diaphragms
- LED-ring dark-field: full, quadrant or rotating sector
- laser dark-field: per azimuth or summed
- RGB Rheinberg
- oblique illumination
- Phase 2: polariser film and a sub-stage LED for transmitted bright-field with the coverslip-corrected 40x (needs a stage aperture)

**Viewing:** camera-only, on screen. Key OFF gives LED-only teaching with the lasers hardware-disabled. Motion stays enabled through K2 while the lid is closed, so reticle and image autofocus still work.

**Calibration and measurement**
- A stage-micrometer wizard for each objective stores µm/px, ID, date and hash.
- Scale bars are burned into exports.
- µm values are refused when the objective is uncalibrated. Labs pay licences for exactly this (https://support.proscitech.com.au/article/44001696357).
- Tools: line, polyline, area, angle, count; an E112 intercept helper; particle size distribution; a scratch-orientation rose diagram.
- Phase 2: automated nodularity and grain size, the modules commercial software sells (https://www.selectscience.net/article/advancing-materialographic-automation-obtain-a-fully-integrated-workflow-with-olympus-stream-1-8).

**Automation**
- Laser-autofocus tiling with a focus map, then stitching.
- Extended depth of field with focus-stack (MIT licence). It needs about 100 MB/MP, so about 0.3 GB for binned stacks (https://github.com/PetteriAimonen/focus-stack).

**Reports:** HTML/PDF with method, magnification, calibration ID, field count and operator.

**Usability** (https://pmc.ncbi.nlm.nih.gov/articles/PMC11448745/):
- pre-assembled
- kiosk UI with no Linux exposure
- instructor and student roles
- one-click session checks

### 9.2 What Llama 3.2 1B does (offline; it explains, it never measures)

- Explains the physics in context using the computed record, for example why dark-field shows particles smaller than the 1.34 µm resolution, or why a scratch glows under only one azimuth.
- Diagnoses "why is my image blurry, dark or washed out" from metrics the code has already computed: autofocus error, Laplacian, clipping, haze index, the FieldGate class. Rules pick the cause; Llama phrases it.
- Coaches preparation, cleaning, levelling and calibration from team-written SOP summaries with clause references.
- Drafts the lab report or inspection certificate from the JSON record.
- Summarises a shift or class log, in Hindi if the quality is acceptable.

### 9.3 Guardrails

- **(a) Code writes the record.** It is a signed JSON with IDs, calib_id, µm/px, method, fields, counts, values, spec, rule IDs and verdict.
- **(b) Llama sees only that record** plus the team's SOP snippets. No ASTM text is used, because the rules require owning all materials.
- **(c) Slot templates** such as {{nodularity_pct}} are filled by code.
- **(d) A validator** rejects any numeral or unit not in the record, regenerates once, then uses the template fallback.
- **(e) Whitelisted actions per rule.** Severity and verdict always come from rules.
- **(f) Hard refusals:** no calibration means no µm; fewer than 500 nodules means INSUFFICIENT; Llama must name the spec it used.
- **(g) Advice is shown beside the measured values,** against automation bias (https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf).

**Licence:** show "Built with Llama". Any fine-tuned model name must start with "Llama"; Phase 1 does no fine-tuning. The model card says the models are "not designed to be deployed in isolation".

### 9.4 Built-in teaching experiments, which double as report ablations

1. Bright-field vs LED dark-field vs laser dark-field on one field.
2. Image autofocus vs laser autofocus on mirror, Si, glass and PCB.
3. Detected vs resolved: 1 µm diamond is invisible in bright-field and bright in dark-field.
4. Rayleigh resolution vs NA.
5. Depth of field vs NA, from the autofocus z log.
6. Manual vs automatic grain intercepts.

**Curriculum fit:** AICTE PCC-MM207, the DIET experiment list (steels, cast irons, heat-treated steels, welds/HAZ), and electronics PCB labs.

---

## 10. Datasets, data collection and licences

### 10.1 Calibration and reference artefacts

- **Reflective (chrome-on-glass) stage micrometer:** µm/px per objective. Check against the formula values of 2.04 and 0.82 µm, with ±5% acceptance.
- **Diamond paste 1/3/9 µm in IPA on Si and glass:** POD curves and scatter-equivalent calibration.
- **Scribed scratch ladder R1–R5:** widths from 40x bright-field, and SEM through I-STEM (about ₹708 per sample with GST, about 2022 tariff).
- **Clean references:** first-surface mirror, HDD platter, Si piece. Used for haze, false calls per cm² and daily MAINTENANCE.
- **Ground-glass target** at the focal plane for the per-azimuth dark-field illumination maps.

### 10.2 Own captures (the core dataset; collection starts in week 2)

**Stored for every field**
- raw Bayer planes for bright-field, DF1–3, ring, dark and autofocus frames
- metadata: z, sensor_ts, exposure and gain, powers, objective, calibration, specimen and zone

**Labels**
- **Automatic:** wipe-and-re-image (particle vs fixed defect, the Samsung method) plus physics labels (A, orientation, persistence).
- **Manual:** checked in CVAT or Label Studio, offline.

**Splits:** by specimen, never by field.

**Targets**

| Phase | Minimum (gate) | Target |
|---|---|---|
| Phase 1 | ≥15 specimens and 200 fields; ≥600 labelled dark-field blobs; ≥1,024 calibration crops (augmented if needed); ≥100 clean-reference fields as negatives | ≥40 specimens and 600 fields; ≥2,000 labelled dark-field blobs; ≥300 clean-reference fields; 6 ductile-iron specimens × 10 fields with manual counts by 2 students, checked by the instructor; 200–300 FieldGate frames per class |
| Phase 2 | Re-capture at 520 nm; ≥3,000 tiles; 5 steels × 5 fields for grain size | about 500 PCB images at 4x; 20–50 SEM-verified defects |

No public labelled optical nodularity dataset was found, so publishing ours is itself a contribution.

### 10.3 Public data (pretraining and ablation only; validate on own captures and report separately)

| Dataset | Use | Licence |
|---|---|---|
| MVTec AD 2: 8 scenarios, includes dark-field (https://arxiv.org/abs/2503.21622) | anomaly and dark-field pretraining | CC BY-NC-SA 4.0 (challenge only) |
| VisA PCB1–4 (https://github.com/amazon-science/spot-diff) | PCB | CC BY 4.0 |
| DeepPCB (https://github.com/tangsanli5201/DeepPCB) | PCB | research only |
| FICS-PCB, includes a microscope subset (https://eprint.iacr.org/2020/366) | PCB at microscope scale | check |
| NEU (named in PS5), KolektorSDD2 (https://www.vicos.si/resources/kolektorsdd2/) | steel-surface ablation | KSDD2 CC BY-NC-SA |
| MicroSteel: 82 optical micrographs (https://arxiv.org/pdf/2606.19934) | bright-field segmentation | MIT |
| ExOne 316L grain boundaries (https://arxiv.org/html/2307.05911v1) | grain pretraining | none stated; internal use, ask the authors |
| UHCS (https://arxiv.org/pdf/1702.01117), MetalDAM (https://github.com/ari-dasci/OD-MetalDAM) | SEM ablations only | MetalDAM none stated |
| BACG-YOLOv8: dark-field scratches, 450 images (https://pmc.ncbi.nlm.nih.gov/articles/PMC12735104/) | comparison point | availability unknown |

Every image taken from a dataset is captioned as such in the video and the report.

---

## 11. Laser safety and interlock wiring

### 11.1 Inventory and class

Limits under IEC 60825-1, visible CW: Class 2 ≤1 mW, Class 3R ≤5 mW (https://www.seibersdorf-laboratories.at/fileadmin/user_upload/docs/le/las/publ/whitepaper_iec-60825-1.pdf). At 850 nm: Class 1 AEL ≈0.78 mW, Class 3R AEL ≈3.9 mW (calc); above that is Class 3B.

| Source | Phase | How the class is set | At the aperture | Class |
|---|---|---|---|---|
| DF1–3, 650 nm or 520 nm direct-diode dot, nominal 5 mW | 1 and 2 | glued absorptive ND rated at its wavelength, nominal current, measured | ≤1 mW each | 2 (the bare module is 3R or worse) |
| AF650 | 1 | ND2 cap → ND slider → coverslip pick-off (about 8%) → beamsplitter | module ≤50 µW; ≤2 µW at the objective (slider ND0) | 1 by power |
| AF850 | 2 | Preferably a maker-rated ≤0.5 mW module. Otherwise a 5 mW module + bonded NIR-rated ND1 (OD specified at 850 nm), then an 8% wedged plate | ≤0.5 mW module; ≤0.05 mW at the objective | 1 (AEL 0.78 mW, calc). **A bare 5 mW module is Class 3B** (>3.9 mW). No Class 2 in the IR: https://www.mikrocontroller.net/attachment/283492/00018294_0_Safety_Information.pdf |

**Measuring every module**
- Measure every module after its ND, on a borrowed calibrated meter (calibrated at 850 nm for AF850). Point the module into the meter through a closed black tube, after safety sign-off. Record it in `laser_power`, on a label, and in the report.
- Do not run bare modules open-beam to "find" the right ND. Start at ND1.0 and step down only if the post-ND reading is low.
- NIST found 44.4% of red and 89.7% of green "3R" pointers over the limit, up to 66.5 mW, and more than 75% leaked IR (https://www.nist.gov/news-events/news/2013/03/nist-tests-underscore-potential-hazards-green-laser-pointers).
- Check each "green" module for IR leakage on the OV9281 behind the IR-pass filter. Check the wavelength with a DVD-grating spectroscope.

**Rules**
- Never use PWM duty to lower the class.
- No 405 nm; no DPSS.
- Never use gel or plastic NDs at 850 nm.
- AF850 is never powered with the lid open.
- Never power a bare 850 nm module outside a closed tube without EN 207 eyewear and the Principal Laser User present.

### 11.2 Enclosure

- Opaque, light-tight and flocked, with no viewing window.
- **The eyepiece is sealed inside the camera collar**, which is fixed with tools and carries an interlock switch. Microscopes are optical aids that raise the hazard even at Class 1 or 2 (FDA), and the AF650 return travels straight up the eyepiece.
- All beams stay within about 35 mm of the sample plane under the head (module exits at r ≈ 50–85 mm sit 18–31 mm up at 20°, calc) and end in V-traps. The AF850 transmitted beam ends in a dump inside the AF block.
- Camera ribbons leave through a light-tight brush gland.
- "Class 1 by containment" is a design target, not a certification (https://en.wikipedia.org/wiki/Laser_safety).
- Compute and PSUs are in a separate vented bay.

### 11.3 Interlock wiring (hardware; software is never in the safety path)

```
 Mains -> certified 5 V LASER PSU (separate from Pi rail) -> F1
   -> KEY S1 [OFF | ALIGN | RUN]   (key held by the Principal Laser User)
   -> LID-A  positive-opening (contact 1), closed only when lid shut
   -> LID-B  positive-opening (contact 1), second, differently mounted
   -> COLLAR microswitch, closed only while camera collar is seated
   -> coil of laser relay K1 (5 V coil), RESET pushbutton (momentary NO) in parallel with K1 aux NO = seal-in latch
      (any break drops K1; restart needs manual RESET; AERB/SG/IS-7 7.2.5 practice)
 K1 main NO -> LASER BUS +5 V --+-- EMISSION LED (hard-wired)
                                +-- opto-isolator -> GPIO25 (read-only, logs interlock_events)
                                +-- DF1..DF3 (+ND) -- N-FET -- GND   gates GPIO17/27/22 via 100R, 10k pull-down
                                +-- AF650 (+ND2)  -- N-FET (GPIO13)
                                +-- [KEY S1 contact: open in ALIGN] -- AF850 (+ND1) -- N-FET (GPIO19)
                                                   'IR ON' LED across the switched AF850 supply
 MOTION: 12/24 V PSU -> LID-A (contact 2) -> LID-B (contact 2) -> coil of motion relay K2
   K2 NO -> connects stepper ENN to the Pico enable line; K2 dropped -> ENN pulled HIGH via 10k to VIO
   (drivers disabled when lid open: pinch hazard). Key position does NOT affect motion, so key-OFF LED teaching keeps Z/XY.
 ALIGN: bypasses LID-A/B contact 1 only (never COLLAR, never K2), flashes emission LED;
        only the ND-capped Class 2 visible modules (<=1 mW, nominal current, optional slip-on ND0.5) can run;
        AF850 supply is opened by the key contact; EN 208 alignment eyewear mandatory; PLU present.
 Phase 2: 555 missing-pulse watchdog on GPIO18 heartbeat in series with K1 coil
```

- **Basis:**
  - AERB/SG/IS-7 §7.2.5 requires fail-safe, positive-break interlocks with a manual restart for enclosures containing Class 3B/4 lasers (https://aerb.gov.in/images/PDF/CodesGuides/NuclearFacility/FuelCycleFacilities/1.pdf). We apply it voluntarily to this Class 2 product. It is directly relevant whenever a bare (3B) 850 nm module is present.
  - 21 CFR 1040.10 (a US manufacturer rule, used here as a design checklist) calls for a housing, an interlock and an emission indicator (https://www.law.cornell.edu/cfr/text/21/1040.10).
- **Software:** refuses to sequence lasers when GPIO25 reads open, and can only switch lasers off within an enabled rail. The watchdog can only drop K1.
- **Testing:** test both chains (K1 lasers, K2 motion) 20 times before first light and after every wiring change, and log each test.

### 11.4 Labels, eyewear, people

**Labels**
- yellow laser-warning triangle
- wavelength, power and class label at each module
- on the door: "CAUTION – LASER RADIATION WHEN OPEN AND INTERLOCK DEFEATED. DO NOT STARE INTO BEAM."
- Phase 2: "INVISIBLE LASER RADIATION 850 nm"

**Eyewear**
- Certified EN 208 alignment eyewear for 520 nm and for 650 nm (https://www.lasermet.com/resources/laser-alignment-adjustment-eyewear-european-standard-en-208/). EN 208 attenuates visible beams to Class 2 levels, so the beam stays visible and nobody lifts their goggles to align.
- Optional full-protection 520 nm eyewear (OD ≥4) only from a certified supplier. The $19.99 OD6+ listing has no certification evidence.
- EN 207 at 850 nm (https://www.lasermet.com/?p=1181), ordered with the 850 nm modules in week 5 and in hand before any module is powered.
- Goggles rated 200–540 nm do **not** cover 650 or 850 nm (https://m.indiamart.com/adlabs-instruments/laser-safety-goggle.html). OD is log₁₀ of attenuation.
- The 850 nm beam is aligned only with an IR card or the OV9281 view.

**People**
- Name one Principal Laser User; AERB guidance appoints one for installations with Class 2 or higher lasers. Keep a log book (our own practice).
- No jewellery; never look along a beam; RUN mode only with the lid closed.
- Get written sign-off from the institute's safety officer or physics laser in-charge before first light, and before any laser power measurement. Request it in week 1; target the end of week 2.
- Standards: cite IEC 60825-1:2014. Indian adoption of Part 1 as IS 14624 is unconfirmed (https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/Indian_standards/isdetails/MzMzMDQ=).

**Other hazards**
- No high voltage, vacuum or X-rays, so AERB radiation licensing does not arise.
- Electrical: fused 12/24 V rails, mains only inside certified adapters.
- Chemicals: IPA kept away from electronics; gloves for diamond paste. Etching only in the college lab, under its SOP.
- HDD teardown: neodymium magnets pinch and some platters are glass, so wear safety glasses.

**The event**
- The finals are a live virtual demo, so the lid stays closed on camera.
- Never run beams outdoors. Local BNSS s.163 orders apply, for example on Mumbai public roads from 6 Oct to 4 Dec 2026 (https://curlytales.com/mumbai-bans-firecrackers-high-intensity-lasers-on-public-roads-till-dec-what-you-need-to-know/).

**Report dossier:** risk assessment, measured powers, class rationale (including the bare-850 nm Class 3B note), wiring diagram for both chains and the test log, alignment SOP, eyewear specification, and the no-HV/vacuum/ionising-radiation statement.

---

## 12. BOM

₹88/US$. \* = estimate, unverified; get quotes before the report. **Owned** items are excluded from the subtotals.

| # | Item | Qty | Phase / order week | USD | INR | Source / note |
|---|---|---|---|---|---|---|
| O1 | Raspberry Pi 5 8 GB | 1 | **OWNED** | (175 list) | — | BeamTriage |
| O2 | Raspberry Pi AI Camera (IMX500), unmodified | 1 | **OWNED** | (70 list) | (₹6,889–10,144 street) | https://imaging-resource.com/news/2024/10/07/raspberry-pi-launches-a-70-ai-powered-camera-module |
| 1 | AI HAT+ 26 TOPS (Hailo-8) | 1 | P1 / wk1 | 110 | 9,950–11,309 | https://www.silverlineelectronics.in/products/official-raspberry-pi-ai-hat-26-tops-artificial-intelligence-authentically-impressive-silverline-india |
| 2 | Active Cooler, 27 W PSU, 64 GB A2 microSD | 1 set | P1 / wk1 | 27\* | 2,400\* | BeamTriage |
| 3 | DIN 160 mm achromat set 4x/10x/40x/100x (check 10x WD ≥6.6 mm; 100x not used) | 1 | P1 / wk1 (or borrow) | 104 + shipping | 9,200–12,000\* | https://www.walmart.com/ip/178289600 ; single 10x $45.90–51: https://magnifiers.nz/product/biological-microscope-objectives/ |
| 4 | WF10x/18 eyepiece, 23.2 mm | 1 | P1 / wk1 | 9–17\* | 800–1,500\* | https://moticeurope.com/en/widefield-eyepiece-wf10x-18mm-18.html |
| 5 | 25 mm 50/50 plate beamsplitter, about 1 mm, back face AR-coated | 1 | P1 / wk1 | 6–23\* | 500–2,000\* | research C |
| 6 | Epi illuminator: 1 W LED + CC driver, opal, f30/f50 lenses, iris, printed slit reticle | 1 | P1 / wk2 | 15–25\* | 1,300–2,200\* | est. |
| 7 | 650 nm 5 mW 5 V dot modules (3 DF + 1 AF + 6 spares) | 10 | P1 / wk1 | ~1.5 | 100–130 | https://robu.in/?p=16339 |
| 8 | 520 nm direct-diode **dot** modules, adjustable focus, 5 V confirmed (3 + spare) | 4 | P1 / wk1 | 91–136\* | 8,000–12,000\* | generic dot modules; the cited Amazon listing is a line module (do not buy unless a dot version is offered): https://www.amazon.in/520nm-Laser-Module-Industrial-adjustable/dp/B0CSYSVPS9 ; rated fallback Picotronic DI520-1-3 |
| 9 | ND attenuator assortment, absorptive glass rated at 520/650 nm (ND 0.3–2), plus ND slider parts (0/1/2/3) | 1 set | P1 / wk1 | 6–17\* | 500–1,500\* | est.; gel or plastic NDs are visible-only |
| 9a | 780–850 nm IR-pass / longpass filter for week-1 IR-leak tests on the OV9281 | 1 | P1 / wk1 | 6–17\* | 500–1,500\* | est. |
| 9b | Linear polariser film (polarisation check) | 1 | P1 / wk1 | 6\* | 500\* | moved from options |
| 10 | AF injection: #1.5 coverslip or wedged pick-off, f = 75 mm lens, drilled two-hole shim mask | 1 | P1 / wk1 | 9–28\* | 800–2,500\* | est. |
| 11 | WS2812B 24-LED ring + printed shroud (mounted about 26 mm up) | 1 | P1 / wk1 | 3–11 | 300–1,000 | https://www.universal-solder.ca/product/24-led-ring-ws2812b-rgb-addressable-86mm/ |
| 12 | XY: 2× MGN12H, 2× NEMA17, 2× TMC2209, 2× T8×8, couplers, home switches | 1 | P1 / wk1 | 63–91\* | 5,500–8,000\* | TMC2209 ₹429, MGN12H 0.5 m ₹2,194: https://stgaws.robu.in/product/tmc2209-v3-0-stepper-motor-driver-module ; NEMA17 ₹350–1,000: https://probots.co.in/nema17-stepper-motor-4-2-kg-cm-single-shaft-1-8-degree.html |
| 13 | Z: MGN12 200 mm, NEMA17, TMC2209, T8×2 + anti-backlash nut, home switch | 1 | P1 / wk1 | 31–45\* | 2,700–4,000\* | same + est. |
| 14 | Raspberry Pi Pico (PIO stepping, WS2812, ADC) | 1 | P1 / wk1 | 5–8 | 400–700 | research A1 |
| 15 | Frame: 2040/2020 extrusion, aluminium head plate and autofocus block, PETG/ASA prints | 1 | P1 / wk1–2 | 34–45\* | 3,000–4,000\* | est. |
| 16 | Light-tight enclosure, black flocking, 3 V-traps | 1 | P1 / wk2 | 17–34 | 1,500–3,000 | research C |
| 17 | Interlock: 2 positive-opening lid switches (2 contacts each) + collar microswitch, key switch (with ALIGN contact), latching relay K1, motion relay K2, RESET, emission LED, labels | 1 | P1 / wk1 | 11–17 | 1,000–1,500 | research C + est. |
| 18 | MOSFETs, pull-downs, optocoupler, fuses; 5 V laser PSU; 12/24 V motor PSU | 1 | P1 / wk1 | 15–25\* | 1,300–2,200\* | est. |
| 18a | 2× 300–500 mm Pi 5 22-pin camera ribbons, light-tight brush gland, strain relief | 1 set | P1 / wk1 | 7–17\* | 600–1,500\* | est. |
| 18b | 2× DS18B20 temperature probes (column, autofocus block) | 2 | P1 / wk1 | 1–2\* | 100–200\* | est. |
| 19 | 12 V stack light + buzzer + ULN2803, ACK and OVERRIDE buttons | 1 | P1 / wk3 | 33\* | 2,900\* | BeamTriage |
| 20 | 7-inch HDMI touch display (CAM1 kept for the OV9281) | 1 | P1 / wk3 | 45–68\* | 4,000–6,000\* | est. |
| 21 | Calibration: reflective stage micrometer, diamond paste 1/3/9 µm (₹780), slides, first-surface mirror offcut, ground-glass target | 1 set | P1 / wk1 | 20–38\* | 1,800–3,300\* | https://m.indiamart.com/met-hr-tech/diamond-paste.html |
| 22 | Specimens: HDD platters, Si pieces, polished SS shims, ductile-iron coupons (borrowed where possible) | — | P1 / wk1–2 | 11–23\* | 1,000–2,000\* | college lab, foundry, e-waste |
| 23 | Eyewear: certified EN 208 alignment eyewear for 520 nm and 650 nm (optional certified OD ≥4 520 nm pair) | 2 | P1 / wk1 | 57–102\* | 5,000–9,000\* | https://www.lasermet.com/resources/laser-alignment-adjustment-eyewear-european-standard-en-208/ ; Indian listing: https://m.indiamart.com/adlabs-instruments/laser-safety-goggle.html |
| 24 | BPW34 photodiode + op-amp TIA (Pico ADC) | 1 | P1 / wk2 | 2\* | 150\* | est. |
| 25 | GPU and x86 cloud time for training, DFC compiles and the IMX500 export | — | P1 | 14–40 | 1,200–3,500 | BeamTriage |
| | **Phase 1 subtotal** | | | **~$760–1,165** | **₹67,000–1,02,500** | Lean path: about ₹47–73k (borrow objectives, eyepiece, display, 650 nm goggles; drop the stack light) |
| 26 | Arducam OV9281 mono global-shutter NoIR + Pi 5 22-pin cable | 1 | P2 / **order wk1** (IR-leak and fit tests) | ~70 | 6,141 | https://zbotic.in/product/arducam-ov9281-1mp-global-shutter-monochrome-noir-camera-module-with-m12-mount-lens-for-raspberry-pi-4-3b-3/ |
| 27 | 850 nm module, preferably maker-rated ≤0.5 mW (Class 1); otherwise 5 mW (bare = Class 3B) + NIR-rated ND1 caps with OD specified at 850 nm | 2 | P2 / **order wk5** | 12–20 | 1,100–1,800 | generic $12–20 (MOQ 10); TME €108.17 rated: https://www.tme.eu/en/details/fp-d-850-5-e-f/laser-modules/laser-components/ |
| 28 | Edmund #69-196 800 nm shortpass dichroic, 12.5×17.6×1.05 mm | 1 | P2 / **order by wk3** | 145 + shipping/duty | 13,000–17,000\* | https://www.edmundoptics.com/p/800nm-125-x-176mm-dichroic-shortpass-filter/23538/ |
| 29 | Autofocus arm: wedged 8/92 plate, f = 75 mm achromat, 800 nm longpass (FELH0800 class), beam dump, IR viewer card, aluminium block | 1 | P2 / wk6 | 45–102\* | 4,000–9,000\* | research C |
| 30 | EN 207 850 nm eyewear | 1 | P2 / **order wk5** (in hand before any module is powered) | ~40\* | 3,500\* | est. |
| 31 | SEM widths for the scratch ladder via I-STEM (about ₹708 per sample with GST) | 5–10 | P2 / book wk2 | 40–80 | 3,500–7,000 | https://www.istem.gov.in/uploads/governmentorders/1647003348_IIT_Delhi.pdf |
| | **Phase 2 subtotal** | | | **~$355–505** | **₹31,200–44,400** | |
| | **Total new spend** | | | **~$1,110–1,670** | **₹0.98–1.47 lakh** | against ₹4.48 lakh for a typical metallography station |

**Options, not in the subtotals**
- 2 more 520 nm modules for 5 azimuths: ₹4,000–6,000\*.
- Rotating diffuser: about ₹500\*.
- Sub-stage LED: about ₹300\*.
- Hall-sensor objective ID: about ₹200\*.
- 555 watchdog: about ₹100.
- Rated 520 nm Picotronic DI520-1-3: €129.33 each.
- Motic LM Plan 10x/0.25, WD 17.5 mm: $321–378 (https://www.microscopeworld.com/p-2192-mitutoyo-m-plan-apo-10x-objective.aspx).
- Fallback camera, Arducam M12 IMX500: $139.99 (https://www.arducam.com/arducam-imx500-m12-mount-ai-camera-professional-optical-expansion-kit-multi-fov-m12-lens-bundle-20-to-180-hfov.html).
- Edmund #43-955 hot mirror: $53.

---

## 13. Week-by-week schedule (12 Oct 2026 – 10 Feb 2027)

**Key dates:** registration closes Mon 19 Oct; Phase 1 (report, source code and demo on GitHub) is due Mon 30 Nov; Phase 2 is due Wed 10 Feb 2027; finals are early March. Diwali (Sun 8 Nov) and end-semester exams (late Nov to Dec) are light weeks.

**Phase-1 minimum (must ship by 30 Nov):**
- the afocal head with 4x/10x
- laser dark-field (3 azimuths) and the biased Scheiner laser autofocus
- the interlock
- the classical detector plus physics features
- the Hailo crop classifier in the verdict path
- rules → severity → stack light → SQLite
- Llama advice with the validator
- the dataset minimum (§10.2)
- report and video

**Descope order, if a gate slips:**
1. FieldGate (IMX500)
2. nodularity (D2)
3. motorised XY: fall back to manual XY with motorised Z, decided at the end of week 3
4. dataset size, down to the minimum

| Wk | Dates | Deliverables | Gate / descope |
|---|---|---|---|
| 1 | 12–18 Oct | **Admin:** register (before Mon 19 Oct). Email tech-queries@armbharatchallenge.com with "PS5" in the subject by Wed 14 Oct (T15). **Orders:** place the week-1 orders (including 520 nm dot modules, IR-pass filter, ribbons, EN 208 eyewear) and get written quotes and ETAs by Fri 16 Oct. **Toolchain:** set up native x86-64 Linux (lab PC) or a cloud x86 instance for DFC 3.33 and the IMX500 export; flash Trixie (check hailo-all availability, Bookworm fallback); install hailo-all and ExecuTorch 1.5.1. **Tests:** T1 afocal coupling on a borrowed microscope, T2 IR-cut (LEDs and the ND-capped 650 nm only), T6 timing and clock, T7 ROI, T8 Llama gate, T14 toolchain. **People:** name the Principal Laser User, request safety sign-off, book I-STEM. | **G1:** afocal µm/px within 5%; IR-cut response at 650 nm known; Llama tok/s known; email sent. Descope: if afocal coupling fails, order the Arducam M12 IMX500. |
| 2 | 19–25 Oct | **Head v1:** column, Z rail with whole head, afocal collar, epi arm with reticle, beamsplitter, 45° blank hot-mirror spacer. Calibrate 4x/10x on the micrometer. **Safety:** build the enclosure and both interlock chains (K1, K2) and run the 20× test. **After sign-off:** T3 (post-ND power only, closed tube). **First light:** one 650 nm dark-field laser (lid closed) on a platter and SS. **Tests:** T4 autofocus bench, T5 dark-field look, T10 objective geometry, T11 Z mechanics, T12 veil, T13 drift, T9 when the HAT arrives. **Motion:** Pico firmware for XY/Z and the ring. | Laser first light and any laser power measurement only after sign-off and the interlock log. Until then, LED modes only. |
| 3 | 26 Oct–1 Nov | Phase-1 Scheiner autofocus closed loop with the bias (s0 and gain per objective, ND slider settings) plus the image-autofocus baseline. 3-laser collar with dumps; ring at about 26 mm; elevation sweep 15–25°; orientation sweep on a scratch; s vs p polarisation test. Timestamp-gated sequencer. XY stage. Capture tool. **Order the #69-196.** | **G2:** spots resolved at best focus (s0 ≈ 44 binned px ±25%); autofocus locks on a blank mirror in ≥95% of tries within ±DOF/4; 3 µm diamond visible in dark-field at 10x. **Wavelength decision:** 650 vs 520 nm for Phase 1. Descope: autofocus on mirror and Si only, reticle autofocus for the rest. If XY is not reliable, Phase 1 runs on manual XY with motorised Z. |
| 4 | 2–8 Nov (Diwali) | Light week. Data sprint 1: D1 specimens with wipe-and-re-image; ductile-iron fields. Classical detector + anisotropy. Rule engine v1, SQLite, stack light. FieldGate training and .rpk conversion: stretch only. | CPU physics path complete. |
| 5 | 9–15 Nov | H1c crop classifier HEF (DFC 3.33, ≥1,024 crops); hailortcli and app benchmark on the Pi; Hailo process with shared memory. Llama service with validator, FTS5 SOPs and fallback. Basic E2567 nodularity with the dark-field prep gate (descope item 2). **Order the 850 nm modules and EN 207 eyewear together.** | **G3:** end-to-end on one part. If the custom HEF fails, ship the precompiled Model Zoo backbone HEF as a crop embedder with a CPU linear head, so a HailoRT model stays in the verdict path. |
| 6 | 16–22 Nov | Integration; CPU-only flag (benchmark only); first POD points at 1/3/9 µm; latency histograms; tok/s idle vs busy; thermals; QC-mode UI. | **Feature freeze Sun 22 Nov.** Dataset minimum met. |
| 7 | 23–29 Nov | Report draft, short demo video, GitHub cleanup with model hashes. **Submit by Sat 28 Nov** (deadline Mon 30 Nov). | — |
| 8 | 30 Nov–6 Dec (exams) | Light week. Switch to 520 nm if Phase 1 shipped at 650 nm; re-measure powers; re-capture the reference set. Bring up the OV9281 on CAM1 alongside the HAT ribbon; CMA check. | — |
| 9 | 7–13 Dec | 850 nm autofocus arm on its aluminium block. Before first power: EN 207 eyewear in hand and the NIR-rated ND1 bonded (or a rated ≤0.5 mW module). Then: dichroic in the reserved slot, two-hole mask with bias, wedged 8/92 plate, dump, 800 nm longpass, OV9281 at the conjugate plane. IR-card alignment. 850 nm leak test into the AI Camera. | If the dichroic is late, keep AF650 plus the focus-map prediction. |
| 10 | 14–20 Dec | NIR autofocus loop at 60–120 Hz; chromatic offsets per objective; tracking during XY; flatness map; comparison with Phase 1 by material. | **G4:** NIR σz at least 2x better than Phase 1, or Phase 1 stays primary. |
| 11 | 21–27 Dec | POD study at 520 nm by substrate and objective; false calls per cm²; scratch ladder with SEM widths. | — |
| 12 | 28 Dec–3 Jan | Light / buffer week. Retrain H1c on 520 nm data; H1s tile segmenter ablation. FieldGate, if descoped from Phase 1. | — |
| 13 | 4–10 Jan | D2 validated: nodularity against manual counts, plus H2 artefact rejection. D3: Si die map, edge chipping, PCB at 4x under LED. | Descope order: PCB → grain size → 5th laser. |
| 14 | 11–17 Jan | Lab mode: calibration wizard, stitching, focus stacking, Rheinberg, reports, E1382 grain size. Llama student coach and report drafting; Hindi optional. Watchdog; Hall ID. | — |
| 15 | 18–24 Jan | 3-hour soak test. Benchmarks and ablations: CPU-only vs IMX500+CPU vs all three; laser vs LED dark-field; laser vs image autofocus; power and thermals. | — |
| 16 | 25–31 Jan | Student user test in the metallurgy lab (\|ΔG\|, nodularity bias). Technical report, safety dossier, licence audit. | — |
| 17 | 1–7 Feb | Demo video (lid closed, dataset clips captioned). GitHub release with rpk/hef/pte hashes. Jury Q&A rehearsal. | — |
| — | 8–10 Feb | **Submit the Phase 2 proof of concept by Wed 10 Feb 2027.** Then prepare for the finals. | — |

---

## 14. Week-1 test list

Each test gives the method, the pass criterion and the fallback. Tests run in weeks 1–2 as parts arrive. Any test that powers a laser runs only after safety sign-off.

| # | Test | Method | Pass | Fallback |
|---|---|---|---|---|
| T1 | Afocal coupling | Lock the AI Camera at infinity by focusing on a reticle at the eyepiece field stop. Mount it behind the WF10x/18 on a borrowed microscope or a printed tube. Find the eyepoint and eye relief and the camera pupil depth. Image a stage micrometer. | Full circle of about 2,202 px; 2.04 µm (4x) and 0.82 µm (10x) within ±5%; acceptable edge shading; 2 µm lines resolved at 10x | Arducam M12 IMX500, or a relay lens |
| T2 | IR-cut response and leak | At equal measured power on a diffuse card, compare raw R per µW at 650 nm (ND-capped module, after sign-off) against G at 520 nm (or a green LED until the modules arrive), with fixed exposure and gain. Put an **850 nm LED** (not a module) and a 940 nm remote into the camera at maximum gain and exposure. | 650 nm dark-field usable at ≤1 mW (R response ≥20% of the 520 nm G response; placeholder); 850/940 nm reads about 0 DN | Make 520 nm the Phase-1 dark-field source |
| T3 | Laser modules (after sign-off) | Post-ND power only, with the module pointed into the meter through a closed black tube, starting at ND1.0. Power against current after the ND, for every module; 10-minute stability with the BPW34; DVD-grating wavelength; IR leak of green modules on the OV9281 behind the IR-pass filter | ≤1 mW stable within ±10%; no IR; direct-diode | Change ND, reject the module, or buy Picotronic |
| T4 | Autofocus physics | (a) Confirm image autofocus fails on a blank first-surface mirror. (b) Biased Scheiner spots through 10x on mirror, Si and glass: log 50-frame centroid noise, s0, separation per 10 µm of Z, ND slider setting per material, tilt vignetting and capture range. | Spots resolved at best focus (s0 ≈ 44 binned px ±25%); slope within ±20% of 0.296 px/µm (binned 10x); sign readable; no ghost pair from the pick-off; glass double spot resolvable; σ ≤0.2 px; no saturation at the chosen slider setting | Reticle autofocus primary; enlarge h; adjust bias |
| T5 | Dark-field quick look | One 650 nm laser at 20° plus the ring (at about 26 mm, check for beam clipping), on an HDD platter, polished SS, Si and glass at 10x and 4x | Background σ and defect SNR recorded for laser vs ring; 3 µm diamond visible; no ring or shroud glints in the beam | Re-scope where laser dark-field is claimed to win; rotating diffuser; adjust ring height |
| T6 | picamera2 timing | Toggle an LED from GPIO at logged times on both CLOCK_MONOTONIC and CLOCK_BOOTTIME. Measure SensorTimestamp semantics (start of exposure vs start of readout), rolling readout in both modes, latency of exposure/gain changes, binned↔full mode switch, and IMX500 network alongside the raw stream. | 2 frames per state at 30 fps with the rule T6 validates; clock and timestamp semantics known; mode switch time known | Conservative switch rule (§6.2); separate measurement pass; empirical timestamp offset |
| T7 | IMX500 ROI | Dump input tensors over a micrometer in both modes; change set_inference_roi_abs over 100 frames | Binned mode: native vs binned ROI pixels known; no stalls | Fixed ROI; FieldGate on full-res stills only |
| T8 | Llama gate | 1B SpinQuant, ExecuTorch 1.5.1 XNNPACK, 2 threads on cores 2–3 (and 4 for reference). Measure TTFT, decode tok/s and RSS, idle and with synthetic vision load. | TTFT + 80 tokens ≤15 s busy | Shorter prompts, cached advice, 4 threads between parts, llama_main |
| T9 | Hailo + two ribbons | `apt install dkms hailo-all`, `hailortcli benchmark`. Fit the HAT with both 300–500 mm camera ribbons through the gland; run the AI Camera full-res plus OV9281 at 120 fps plus Hailo together; check dmesg for CMA errors and get_throttled; measure real-app Hailo throughput. | Both cameras plus Hailo stable for 1 h, no throttling | Longer header or different ribbons; raise CMA; fan |
| T10 | Objective geometry | Measure nose radius, WD and parfocality of the bought 4x and 10x | 10x WD ≥6.6 mm, or recompute for 15° elevation | Buy Accu-Scope or Kern 10x |
| T11 | Z mechanics | Dial gauge (borrowed), later the autofocus signal: microstep repeatability, backlash, hold with power off | Repeatability ≤1 µm after unidirectional approach | Anti-backlash nut, 1/32 stepping |
| T12 | Epi veil | Ratio of a light-trap frame to a mirror frame | Veil <2% of mirror signal after baffles | Wedged plate, more flocking |
| T13 | Drift | Hold a 10x focus on the micrometer for 1 h near a warm source, logging DS18B20 temperature and focus metric; PETG vs aluminium | Drift <DOF/4 per K after re-zero | Aluminium parts, re-zero rule |
| T14 | Toolchain | Native x86-64 Linux PC or cloud x86: compile a toy ResNet-18 with 3-channel uint8 input to a HEF and run it on the Pi; also run a precompiled Model Zoo HEF on the Pi (the G3 fallback); Ultralytics IMX500 export of YOLO11n-cls; hailo-all on Trixie | All run | Colab or JarvisLabs; Model Zoo recipe; Bookworm |
| T15 | Organisers | Email: is microscope-scale QC of manufactured specimens in scope; is the IMX500 pipeline acceptable alongside HailoRT; any rule on Class 2 lasers in an interlocked box; what Phase 1 must include | Replies logged | Keep the IMX500 optional (FieldGate on the Hailo if rejected); lead with QC mode |
| T16 | Access and logistics | Ductile-iron, steel and Si specimens; a reflective micrometer with contrast in epi; a power meter (calibrated at 650/520 nm, and 850 nm for Phase 2); an ultrasonic bath; quotes and ETAs for 520 nm dot modules, #69-196 and OV9281 by 16 Oct; I-STEM booking; safety officer meeting | All confirmed | Anything arriving after 10 Nov moves to Phase 2 |

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Afocal coupling: pupil depth vs eye relief, vignetting, colour shading | T1 on day 1–3; printed axial collar; measure on the inscribed 1,557 px square; Arducam M12 IMX500 fallback. Never open the owned camera. |
| IR-cut attenuates 650 nm | T2; 520 nm ordered in week 1; autofocus unaffected (it needs only nW–µW) |
| Scheiner spots overlap at best focus (degenerate fit) | Biased crossing (§5.2) with s0 ≈ 3 spot diameters; T4 and G2 check that the spots are resolved |
| Autofocus spots saturate or ghost | ND slider plus PWM; coverslip or wedged pick-off; T4 per-material settings |
| Speckle, haze, glints or dust set a high dark-field floor | 3-azimuth sum and persistence; V-traps; flocking; dark and veil frames; clean-reference negatives; ring baseline; rotating diffuser in Phase 2; only measured POD published |
| LED ring clips laser beams | Ring at about 26 mm; slots 15–25°; CAD check; T5 |
| Epi veiling glare | Light trap, baffles, veil subtraction, flat-field, wedged plate |
| Autofocus fails on glass, PCB or matte surfaces | Range gate, first-spot lock, quality metric, fallback chain, success rate logged per material |
| Autofocus pointing or thermal drift | Scheiner design; aluminium; DS18B20 re-zero above 1 K; MAINTENANCE rule |
| DIN 40x aberration on bare metal; no 40x dark-field | 40x for bright-field on covered slides only; software block |
| 10x working distance too short for 20° | T10; 15° slot or a WD ≥6.6 mm objective |
| picamera2 timing, timestamp semantics, mode switch, IMX500 + raw | T6/T7; conservative switch rule until confirmed; separate measurement pass; binned survey |
| Two ribbons + HAT: fit, length, CMA, heat | T9; 300–500 mm ribbons through a light-tight gland; ribbons fitted before the HAT; CMA increase; vented bay and fan |
| HEF compile or accuracy | Crop classifier, uint8 3-channel input, Model Zoo recipe, HAR vs HEF; precompiled Model Zoo HEF fallback keeps HailoRT in the verdict path; physics CPU path ships from week 4 |
| IMX500 not accepted for PS5 | Optional by design; FieldGate compiled to a HEF and run on the Hailo |
| Llama slow | Week-1 gate; short prompts; cache; 4 threads between parts; template fallback; never on the alert path |
| Imports late (520 nm, dichroic) | Ordered in weeks 1–3; Phase 1 can run on 650 nm; Phase 2 can keep AF650 |
| Cheap modules mislabelled or unstable; wrong type (line vs dot) | Specify dot modules; measure every one post-ND; ND at nominal current; spares; BPW34 and reference slide normalisation |
| Bare 850 nm module is Class 3B (invisible) | Prefer rated ≤0.5 mW modules; NIR-rated ND1 bonded before normal use; closed-tube handling; EN 207 eyewear in hand first; AF850 disabled in ALIGN and with the lid open |
| Safety approval delayed | Request in week 1; LED-only modes are a complete instrument meanwhile; no laser measurement before sign-off |
| Toolchain host | Native x86-64 Linux or cloud x86, not an emulated VM on the M3 Mac |
| Judges doubt "factory floor" | Week-1 email; QC mode first; manufactured specimens only; MSME foundry story |
| Over-claiming sensitivity | Measurement-honesty rule; scatter-equivalent units; detected vs resolved; POD only |
| Throughput looks slow | 4x survey then 10x review; per-field latency reported; part times stated |
| Scope creep with 1–3 people and exams | Phase-1 minimum and descope order; gates G1–G4; feature flags; light weeks |
| Toolchain drift | Wheels downloaded now; apt-mark hold; hashes in `model_registry` |
| Licence conflicts | In-house mechanics; no OpenFlexure STL; PiFocus and ImSwitch licences checked before copying any code; dataset audit |

---

## 16. Sources (accessed 10 Oct 2026)

**Challenge and positioning**
- https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html
- https://arm-education.github.io/Arm-Developer-Labs/Challenge_Page.html
- https://github.com/katyayanitiwari/EdgeInspect-AI
- https://www.foundryinfo-india.org/profile_of_indian.aspx
- https://fliphtml5.com/jfjx/uffx/Updated-Vol-02/83
- https://Lbrce.ac.in/mech/mech_labs/06%20Metallurgy%20&%20Material%20Science%20Lab.pdf
- https://diet.ac.in/wp-content/uploads/2023/05/08.Metallurgy-Lab.pdf
- https://ira.lib.polyu.edu.hk/handle/10397/103898
- https://m.indiamart.com/metnationtechnologies/laboratory-microscope.html
- https://m.indiamart.com/uvsar-india/microscopes.html
- https://m.indiamart.com/acmeinstrumentco/labomed-microscope.html
- https://org.gem.gov.in/cart/cart_global/directBuy/754454
- https://www.moglix.com/microware-40-1000x-20mp-8-led-digital-microscope-for-inspection/mp/msne9j8zpmw09d
- https://aicte.gov.in/sites/default/files/MODROB%20Scheme%20Guidelines%2024_12_2021.pdf
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11448745/
- https://arxiv.org/abs/2312.09854

**Electron-microscope feasibility**
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12394587/
- https://www.popsci.com/diy/article/2011-07/you-built-what-scanning-electron-microscope/
- https://www.aerb.gov.in/storage/uploads/documents/regdocTyjvJ.pdf
- https://www.istem.gov.in/uploads/governmentorders/1647003348_IIT_Delhi.pdf

**Camera and optics**
- https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf
- https://pip-assets.raspberrypi.com/categories/1044-raspberry-pi-ai-camera/documents/RP-008147-DS-1-ai-camera-product-brief.pdf
- https://www.raspberrypi.com/documentation/accessories/camera.html
- https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- https://sony.mediaroom.com/2020-05-14-Sony-to-Release-Worlds-First-Intelligent-Vision-Sensors-with-AI-Processing-Functionality
- https://forums.raspberrypi.com/viewtopic.php?t=368940
- https://arxiv.org/abs/1911.13295
- https://www.photomacrography.net/forum/viewtopic.php?p=99265
- https://moticeurope.com/en/widefield-eyepiece-wf10x-18mm-18.html
- https://www.microscopyu.com/microscopy-basics/depth-of-field-and-depth-of-focus
- https://svi.nl/RayleighCriterion
- https://www.walmart.com/ip/178289600
- https://magnifiers.nz/product/biological-microscope-objectives/
- https://www.edmundoptics.com/p/20x-din-achromactic-finite-intl-standard-objective/3135/
- https://www.edmundoptics.com/p/40x-din-achromactic-finite-intl-standard-objective/3136/
- https://www.mikroskop-center.de/media//wysiwyg/Leica_DM12000_M/Leica_DM8000_M_DM12000_M-TechnicalBrochure_en.pdf
- https://openflexure.discourse.group/t/refletion-microscope-without-beam-splitter/2268
- https://openflexure.discourse.group/t/ofm-optics-for-metallurgy-applications-epi-illumination-kohler-design/74
- https://ar5iv.arxiv.org/html/2109.06842
- https://iris.unipv.it//retrieve/handle/11571/1450203/671919/J%20Optics%20Laser%20Technology.pdf
- https://www.arducam.com/arducam-imx500-m12-mount-ai-camera-professional-optical-expansion-kit-multi-fov-m12-lens-bundle-20-to-180-hfov.html

**Dark-field**
- https://evidentscientific.com/en/microscope-resource/knowledge-hub/techniques/darkfieldreflect
- https://patents.google.com/patent/US7218392
- https://patents.google.com/patent/US9176072B2/en
- https://patents.google.com/patent/US9841384
- https://opg.optica.org/ao/abstract.cfm?uri=ao-65-1-309
- https://ebooks.spiedigitallibrary.org/profile/Mingjie.Deng-4237696
- https://patents.google.com/patent/US10067072
- https://patents.google.com/patent/US11493454
- https://arxiv.org/abs/1703.05311v2
- https://pip-assets.raspberrypi.com/categories/810-raspberry-pi-global-shutter-camera/documents/RP-008148-DS-1-camera-extended-spectral-sensitivity.pdf
- https://en.wikipedia.org/wiki/Rayleigh_scattering
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12944180/
- https://opg.optica.org/ao/abstract.cfm?uri=ao-63-25-6686
- https://doi.org/10.3390/photonics11070681
- https://www.scitepress.org/Papers/2021/102233/102233.pdf
- https://americas.fujielectric.com/files/r53-2-02.pdf
- https://averroes.ai/blog/dark-field-inspection
- https://evidentscientific.com/en/products/semiconductor-inspection/mx63-mx63l
- https://www.universal-solder.ca/product/24-led-ring-ws2812b-rgb-addressable-86mm/
- https://m.indiamart.com/met-hr-tech/diamond-paste.html
- https://www.edmundoptics.com/resources/application-notes/optics/understanding-surface-quality-specifications/
- https://wiki.nanofab.ucsb.edu/wiki/Surface_Analysis_(KLA/Tencor_Surfscan)

**Autofocus**
- https://evidentscientific.com/en/learn/white-papers/bxc-series-modular-microscope-assembly-focus-sensing-unit
- https://www.prior.com/product/purefocus850-laser-autofocus-system
- https://microscopeworld.com/content/brochures/PureFocus850_Datasheet.pdf
- https://www.prior.com/blog/adaptive-autofocus-empowers-microscopy-a-technical-overview
- https://arxiv.org/pdf/2006.08610
- https://asiimaging.com/docs/crisp_manual
- https://m.yiqi.com/zt6206/product_285052.html
- https://www.biorxiv.org/content/10.1101/2024.01.15.575442v1.full
- https://github.com/PonjavicLab/PiFocus
- https://imswitchopt.readthedocs.io/en/latest/_sources/gui.rst.txt
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10953376/
- https://ar5iv.labs.arxiv.org/html/2302.14628
- https://www.edmundoptics.com/p/800nm-125-x-176mm-dichroic-shortpass-filter/23538/
- https://www.edmundoptics.com/p/25mm-diameter-45deg-aoi-hot-mirror/5424/
- https://www.edmundoptics.in/f/PdfExport/14198
- https://zbotic.in/product/arducam-ov9281-1mp-global-shutter-monochrome-noir-camera-module-with-m12-mount-lens-for-raspberry-pi-4-3b-3/
- https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/Global-Shutter/1MP-OV9281-OV9282/
- https://refractiveindex.info/?shelf=main&book=Si&page=Green-2008
- https://www.newport.com/medias/sys_master/images/images/h4b/h48/8797049585694/AN08-Mode-Hopping-in-Semiconductor-Laser-Diodes.pdf

**Synchronisation and I/O**
- https://github.com/raspberrypi/picamera2/issues/787
- https://patchwork.libcamera.org/patch/12173/
- https://pi4j.com/blog/2024/20240423_pwm_rpi5
- https://github.com/byogaratchagan/stepper_motor
- https://www.raspberrypi.com/products/raspberry-pi-5/

**Compute**
- https://docs.ultralytics.com/integrations/sony-imx500/
- https://raw.githubusercontent.com/raspberrypi/picamera2/main/picamera2/devices/imx500/imx500.py
- https://www.raspberrypi.com/news/streamline-dataset-creation-for-the-raspberry-pi-ai-camera/
- https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_instance_segmentation.rst
- https://community.hailo.ai/t/compiling-with-16-bit-inputs-crashes/19078
- https://arxiv.org/pdf/2202.06934
- https://docs.pytorch.org/executorch/main/raspberry_pi_llama_tutorial.html
- https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
- https://huggingface.co/Arm/resnet-18-int8-xnnpack-executorch
- https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf

**Standards and datasets**
- https://store.astm.org/e2567-16a.html
- https://store.astm.org/e0112-25.html
- https://www.metkon.com/what-is-astm-e112-a-practical-guide-to-measuring-average-grain-size/
- https://www.selectscience.net/article/advancing-materialographic-automation-obtain-a-fully-integrated-workflow-with-olympus-stream-1-8
- https://support.proscitech.com.au/article/44001696357
- https://github.com/PetteriAimonen/focus-stack
- https://arxiv.org/abs/2503.21622
- https://github.com/amazon-science/spot-diff
- https://github.com/tangsanli5201/DeepPCB
- https://eprint.iacr.org/2020/366
- https://www.vicos.si/resources/kolektorsdd2/
- https://arxiv.org/pdf/2606.19934
- https://arxiv.org/html/2307.05911v1
- https://arxiv.org/pdf/1702.01117
- https://github.com/ari-dasci/OD-MetalDAM
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12735104/
- https://arxiv.org/pdf/2211.10274

**Safety**
- https://www.seibersdorf-laboratories.at/fileadmin/user_upload/docs/le/las/publ/whitepaper_iec-60825-1.pdf
- https://ehs.weill.cornell.edu/faq/how-are-classes-lasers-defined
- https://www.ieee802.org/3/ae/public/mar00/kolesar_1_0300.pdf
- https://www.mikrocontroller.net/attachment/283492/00018294_0_Safety_Information.pdf
- https://www.nist.gov/news-events/news/2013/03/nist-tests-underscore-potential-hazards-green-laser-pointers
- https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1668.pdf
- https://cie.co.at/node/7261
- https://www.fda.gov/radiation-emitting-products/home-business-and-entertainment-products/laser-products-and-instruments
- https://aerb.gov.in/images/PDF/CodesGuides/NuclearFacility/FuelCycleFacilities/1.pdf
- https://www.law.cornell.edu/cfr/text/21/1040.10
- https://www.lasermet.com/resources/laser-alignment-adjustment-eyewear-european-standard-en-208/
- https://www.lasermet.com/?p=1181
- https://photonics.laser2000.co.uk/blogs/optical-density-in-laser-safety/
- https://m.indiamart.com/adlabs-instruments/laser-safety-goggle.html
- https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/Indian_standards/isdetails/MzMzMDQ=
- https://en.wikipedia.org/wiki/Laser_safety
- https://curlytales.com/mumbai-bans-firecrackers-high-intensity-lasers-on-public-roads-till-dec-what-you-need-to-know/

**Parts**
- https://robu.in/?p=16339
- https://www.amazon.in/520nm-Laser-Module-Industrial-adjustable/dp/B0CSYSVPS9
- https://www.picotronic.de/picopage/en/product/detail/id/357261
- https://price-history.in/product/laser-diode-module-violet-purple-05nm-ZxugQAce
- https://www.tme.eu/en/details/fp-d-850-5-e-f/laser-modules/laser-components/
- https://stgaws.robu.in/product/tmc2209-v3-0-stepper-motor-driver-module
- https://probots.co.in/nema17-stepper-motor-4-2-kg-cm-single-shaft-1-8-degree.html
- https://www.holmarc.com/metalic_cube_beamsplitter.php
- https://www.silverlineelectronics.in/products/official-raspberry-pi-ai-hat-26-tops-artificial-intelligence-authentically-impressive-silverline-india
- https://imaging-resource.com/news/2024/10/07/raspberry-pi-launches-a-70-ai-powered-camera-module
- https://www.microscopeworld.com/p-2192-mitutoyo-m-plan-apo-10x-objective.aspx
