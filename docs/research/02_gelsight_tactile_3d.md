# GelSight-style tactile 3D sensors for µm-level surface-defect measurement

Status: COMPLETE, last updated 20 Oct 2026 after the µm-level analysis and the feasibility verdict.

**Confidence tags**
- **[H]**: read on the source page, or extracted from the PDF text.
- **[M]**: search-engine snippet only, or a third-party aggregator.
- **[L]**: a single weak snippet, or an internally inconsistent listing.
- **Estimate**: my own calculation, not a published figure.

**Search limit.** The shared WebSearch budget ran out partway through. The later sections rely on pages already found, plus direct fetches of known URLs. The gaps this left are listed in §11.

---

## TL;DR (µm-level requirement)

1. **Only the commercial GelSight metrology probes are genuinely µm-level.**
   - **GelSight Mobile:** Z accuracy 1 µm + 4% for 1–50 µm features, Z sensitivity < 1 µm, 6.9 µm XY resolution, 17 × 14.2 mm field of view.
   - **GelSight Max:** Z accuracy 0.3 µm + 4%.
   - **Modulus HR:** Z accuracy 0.4 µm + 4%.
   - These cost roughly **US $30–55K per system** (US government award records), plus **$600 gel cartridges**.
2. **GelSight Mini ($510) and DIGIT ($355) are robotics sensors, not metrology tools.**
   - GelSight's own FAQ says the Mini "is not a metrology device", and that DIGIT "is not designed for quantitative measurements". [H]
   - With a custom gel and careful calibration, one academic group reached **12.55 µm mean absolute error** (SD 8–24 µm) against a profilometer on 12–96 µm features, using Mini hardware. [H]
   - Generic learned calibrations give **156–205 µm** errors. [H]
3. **A 4-week DIY build can realistically reach about 10–30 µm depth error, with 5–8 µm/px lateral sampling, on scratches, dents and pits 20–200 µm deep.**
   - **≤ 10 µm is a stretch.** It needs multi-direction sequential lighting, close-focus optics at ≤ 5 µm/px, uncompressed frames, controlled press force, small-ball calibration and local detrending (§4.4).
   - **Sub-µm is not achievable** in this timeframe.
4. **Best fit with your brief and TriNetra.** Use the gel probe as an optional "measure" mode:
   - The camera system finds and classifies defects.
   - The probe measures µm depth on flagged spots. This is the published two-stage pattern: vision-only P 0.62 / R 0.79 vs touch P = R = 1.0 on Boeing panels.
5. **The probe cannot see closed cracks with no relief, and it gives no surface colour.** Keep the red-dye penetrant and the camera channel for those.

---

## 1. How it works

- **Principle.** A clear, soft elastomer has an opaque reflective "skin". Pressed on a part, the skin copies the surface. A camera behind it images the skin under lights from known directions. Photometric stereo gives surface gradients, and integrating the gradients gives a height map.
  - Because the skin "supplies its own BRDF", shiny, transparent and matte parts measure the same. [H] https://people.csail.mit.edu/kimo/gelsight/faq.html
  - The camera views the inside of the opaque skin in a closed probe, so GelSight claims operation "regardless of … ambient lighting". [H] https://www.gelsight.com/products/gelsight-mobile/
- **Classic single-shot recipe.** Three coloured lights, an RGB camera, a lookup table from (R,G,B) to gradient built by pressing a calibration target, then Poisson integration. [H] FAQ
- **The commercial metrology unit is multi-shot, not single-shot RGB.**
  - GelSight Mobile has a 5 MP CMOS camera, a **0.5X telecentric lens** and **six LEDs**. It captures **six images at 50 fps**, one per light direction, and computes 3D "within seconds". [H] M. K. Johnson (GelSight co-founder), *The Shot Peener*, Fall 2022: https://www.shotpeener.com/library/pdf/2022041.pdf
  - MTU describes the same "six LEDs … six images" design. [H] https://cdn.aeroreport.de/en/innovation/with-gelsight-mtu-can-measure-even-the-tiniest-of-flaws
- **Robotics sensors are single-shot RGB at video rate.** This covers Mini, DIGIT, Wedge and GelSlim (§3).
- **Origin of the µm resolution.**
  - The 2011 MIT microgeometry system reached "spatial resolution as small as 2 microns" through improved materials, illumination and reconstruction. [H] https://people.csail.mit.edu/kimo/publications/microgeometry/
  - Press coverage quotes the high-res version as registering "less than a micrometer in depth and about two micrometers across". Reaching that needed smaller metal flecks in the paint plus new lighting and algorithms. [H] https://www.edge-ai-vision.com/2011/08/surface-visualization-no-fiscal-exaggeration/ ; [H] https://www.industryweek.com/information-technology/portable-super-high-resolution-3-d-imaging

---

## 2. GelSight Mobile (industrial handheld)

### 2.1 Specs (µm numbers in bold)

| Item | GelSight Mobile Series 2 | Source |
|---|---|---|
| Field of view | **17.0 × 14.2 mm** | [H] https://www.gelsight.com/products/gelsight-max/ (comparison table); [H] NDTec sheet 10.07.24 https://www.ndtec.net/_files/ugd/043502_2a00239a94aa447292d5d726e694634b.pdf |
| X-Y accuracy | **3 µm + 0.2%** (G520 plate); 3 µm + 1% (G500 plate) | [H] both |
| Z accuracy | **1 µm + 4% (1–50 µm)**; 3 µm + 4% (51–500 µm); 3 µm + 4.5% (501–1000 µm) | [H] gelsight.com |
| Z sensitivity | **< 1 µm** | [H] NDTec |
| X-Y resolution (0.5X lens) | **6.9 µm** | [H] Shot Peener; [H] Agrawal et al. https://ar5iv.labs.arxiv.org/html/2309.04590 |
| Z resolution | **4 µm** | [H] Agrawal et al. |
| As described by MTU | **≈ 4 µm axial, ≈ 8 µm lateral**; gel pad about 2 cm | [H] aeroreport |
| Roughness range (Ra) | 2–20 µm | [H] |
| Capture | **100 ms**; 6 images at 50 fps; 3D "within seconds" | [H] NDTec; [H] Shot Peener |
| Optics and lights | 5 MP CMOS, 0.5X telecentric fixed lens, 6 LEDs | [H] Shot Peener |
| Size and weight | 5 × 5 × 15.5 cm, 400 g (Shock Protected: 464 g) | [H] |
| Interface and OS | USB-C (data + power), Windows 10+, optional Surface Pro 12.3" | [H] NDTec |
| Exports | PDF, STL, CSV, TMD, X3P | [H] NDTec |
| Repeatability | No published depth-repeatability spec found | — |
| Repeatability evidence | Shot-peen coverage study: expanded uncertainty **< 1%** over 20 repeats; 3-operator **Gage R&R < 20% of tolerance**; each gel cartridge calibrated with the standard procedure | [H] Shot Peener |
| Gel cartridge | **US $600** (MB012-4, Series 2) | [H] https://www.gelsight.com/online-store/ |

Sister products (same table) [H] https://www.gelsight.com/products/gelsight-max/:

| Product | Field of view | Z accuracy | X-Y accuracy | Weight |
|---|---|---|---|---|
| GelSight Max | 14.6 × 8.3 mm | **0.3 µm + 4% (1–50 µm)** | — | 727 g |
| Modulus HR | 6.8 × 5.6 mm | **0.4 µm + 4% (1–11 µm)** | 1 µm + 0.3% | — |
| Modulus | 6.8 × 5.6 mm | 1 µm + 4% (5–50 µm) | — | — |

### 2.2 Named industrial uses

- **Product page** [H] https://www.gelsight.com/products/gelsight-mobile/:
  - hole diameter; scratch and defect measurement; fastener flushness (rivets); XYZ dimensions
  - **weld inspection**; **pit detection and corrosion analysis**
  - roughness; radius of curvature; replica measurement; shot-peen analysis
  - Industries: aerospace and defence, automotive, energy, forensics, additive manufacturing, chemical.
- **Mobile-metrology page** [H] https://www.gelsight.com/applications/tactile-sensing-mobile-metrology/:
  - Production QC, flight-line and MRO inspection of **dents, scratches and corrosion on airframes, engine components and landing gear**.
  - Oil and gas piping and sealing surfaces.
  - Moulds.
- **NDTec sheet** [H] adds:
  - dents, notches and scratches; pitting and corrosion; coating profile; fillet-weld relief; porosity; weld seam; **cracks**
  - **painted surfaces** as a measurable material.
  - I found no automotive-paint case study (§11).
- **MTU Aero Engines** [H]: used since 2019 in PW1100G-JM and TP400-D6 final assembly to judge scratches and dents in place. It replaces the old impression → lab → report loop.
- **Boeing panels** [H]: scratches as shallow as **0.01 mm deep and 0.051 mm wide** (Agrawal et al. 2023, §6).
- **Kalitta Air** [H]: teaser claims "$350K in end-customer savings"; the details are behind a download. https://www.gelsight.com/resource/kalitta-air-case-study/
- **Rolls-Royce** [M]: gives the device to its maintenance shops to assess engine-part damage. https://forbes.com.br/negocios/2020/01/startup-arrecada-us-10-milhoes-para-tecnologia-de-sensor-artificial/
  - The same source says more than 100 systems and 2,500 gel cartridges had been sold. [M]
- **Mobile 3.5 software** (Oct 2023) [H]: one-click automated detection of **dents, scratches and pits**. https://s24.q4cdn.com/622300748/files/doc_news/GelSight-Revolutionizes-Defect-Detection-with-GelSight-Mobile-3.5-Software-Update-2023.pdf

### 2.3 Price

There is no list price; GelSight sells on quote. US government award records, via an aggregator [M] https://govconinabox.com/explore/contractors/profile/gelsight-inc-f52xq7:

| Contract | Date | Amount | Item |
|---|---|---|---|
| N6660423P0165 | Mar 2023 | **$40K** | GelSight Mobile Series-2 system |
| W50S8324PA002 | Sep 2024 | $46K | Handheld 3D surface analyser |
| FA857124C0047 | Sep 2024 | $50K | GS Max system + Surface Pro |
| N0016725P1111 | Aug 2025 | $53K | "GelSight system" |
| NASA | Sep 2023 | $31K | — |

A 2020 Forbes article reportedly cited "$40,000" handheld systems. [L] The page returned 403: https://www.forbes.com/sites/jeremybogaisky/2020/01/23/vision-through-touch-startup-gelsight-hones-measurement-system-that-could-speed-up-aerospace-work/

**Working range: US $30–55K per system, plus $600 per cartridge.**

---

## 3. Commercial and open sensors

| Sensor | Price | Camera / optics | Sensing area | Interface | SDK, licence | Pi 5 / ARM | Sources |
|---|---|---|---|---|---|---|---|
| **GelSight Mini** | **$510** Mini System; $560 Robotics Package (markers); gel $57 / $70; launched at $499 | **8 MP, 25 fps** (FAQ also says 24 Hz); RGB LEDs; gel **4.25 ± 0.20 mm** "Lambertian silicone gel"; durability "**1000 coin presses**"; full frames 3264×2448 | **18.6 × 14.3 mm** | USB (Micro-B to USB-C), **UVC** | **gsrobotics**: official, GPL-3.0, Python 3.12, Win/Ubuntu, NN normals → point cloud. **gs_sdk**: GPL-3.0, Ubuntu 22.04, Python ≥ 3.9, calibration tools; OpenCV path may drop to ~10 Hz, FFmpeg path holds 25 Hz. PyTouch; ROS Noetic/Foxy listed | No ARM/Pi statement; UVC + Python suggests it should work (not verified) | [H] sheet https://www.gelsight.com/wp-content/uploads/productsheet/Mini/GS_Mini_4.3.24.pdf ; [H] https://www.gelsight.com/online-store/ ; [H] https://github.com/gelsightinc/gsrobotics ; [H] https://github.com/joehjhuang/gs_sdk ; [H] https://www.eenewseurope.com/en/human-resolution-tactile-sensor-for-engineers-scientists ; [H] https://arxiv.org/html/2509.11385v1 |
| **DIGIT (Meta, sold by GelSight)** | **$355**; gel $42 | OmniVision OVM7692, 640×480 at 60 fps (driver default VGA at 30 fps; QVGA option); 3 RGB LEDs, ≤ 4 lm; ~USD 15 parts cost at 1000 units | **19 × 16 mm** | USB | Hardware **CC BY-NC 4.0** (digit-design); digit-interface CC BY-NC, **archived Nov 2024**; PyTouch MIT, **archived Apr 2024** | Linux udev setup; ARM not stated | [H] https://arxiv.org/pdf/2005.14679 ; [H] https://github.com/facebookresearch/digit-design ; [H] https://github.com/facebookresearch/digit-interface ; [H] https://github.com/facebookresearch/PyTouch |
| DIGIT 360 | Not sold; free units through a call for proposals | Hemispherical fingertip; "~8.3 M taxels"; "7 µm" claimed | Fingertip (not flat) | USB | digit360 repo (licence file not named) | — | [H] https://github.com/facebookresearch/digit360 ; [H] https://digit.ml/ |
| GelSlim 3.0 | ~$122 materials | RPi 160° wide-angle camera, 640×480 at 90 fps | 675 mm² | CSI | Design files **CC BY-NC**; MIT MCube lab closed | Pi camera | [H] https://ar5iv.labs.arxiv.org/html/2103.12269 ; [H] https://github.com/mcubelab/gelslim |
| GelSlim 4.0 | **~$65** materials; laser-cut lens $0.13 | — | — | — | Open docs; U-Net RGB → depth (MMintLab); licence not stated | — | [H] https://arxiv.org/pdf/2409.19770 ; [H] https://www.mmintlab.com/research/gelslim-4-0/ |
| GelSight Wedge (MIT) | Camera ~$16 | RPi mini camera 120°, 640×480 at 60 fps or 320×240 at 90 fps; MLP 5-32-32-32-2 | Curved finger | Pi streams to PC | Project page | Pi used only as camera | [H] https://ar5iv.labs.arxiv.org/html/2106.08851 |
| 9DTact | — | Focus at ~15 mm; 4 mm-radius calibration ball | Compact, 32.5 × 25.5 × 25.5 mm | — | **MIT licence**, ROS, Python 3.8 | — | [H] https://github.com/linchangyi1/9DTact |
| Fingertip GelSight (2017) | ~$30 components | Logitech C310 at 35 mm, 640×480 at 30 Hz | **18 × 14 mm** | USB webcam | — | — | [H] https://pmc.ncbi.nlm.nih.gov/articles/PMC5751610 ; [H] https://ar5iv.labs.arxiv.org/html/1708.00922 |

GelSlim 4.0 also quotes, at the time of its writing [H]: Mini $499 + $49 gels; Mini Robotics $549 + $69 gels; DIGIT $350 + $40 gels.

---

## 4. µm-level performance (new requirement)

### 4.1 Published µm numbers

| System | Lateral | Depth | Accuracy evidence | Conf. |
|---|---|---|---|---|
| **GelSight Mobile S2 (0.5X)** | 6.9 µm XY resolution; XY accuracy 3 µm + 0.2% | Z accuracy 1 µm + 4% (1–50 µm); sensitivity < 1 µm; Z resolution 4 µm | Vendor spec; shot-peen Gage R&R study | [H] |
| GelSight Max | XY accuracy 3 µm + 0.2% | **0.3 µm + 4%** (1–50 µm) | Vendor spec | [H] |
| Modulus HR | XY accuracy 1 µm + 0.3% | **0.4 µm + 4%** (1–11 µm) | Vendor spec | [H] |
| MIT 2011 bench / high-res | "as small as 2 µm" | "< 1 µm in depth" (press quote) | Research prototype | [H] |
| **GelSight Mini** | Native ≈ 5.7 µm/px (estimate: 18.6 mm / 3264 px); **7.7 µm/px** measured on a ~1 cm crop; SDK default 320×240 = **63.4 µm/px** | Vendor: "**not a metrology device**", "mm-level surface data" | Custom gel + CNN normals + FFT Poisson: **MAE 12.55 µm** (test) / 14.0 µm (validation) vs Keyence profilometer on 12–96 µm channels; per-feature SD **8–24 µm**; gel-to-gel ICC 0.993. 3D Cal generic CNN: **205 µm** average error | [H] |
| **DIGIT** | ≈ 30 µm/px (estimate: 19 mm / 640 px) | "**not designed for quantitative measurements**" | 3D Cal: **156 µm** average error | [H] |
| Fingertip GelSight (2017) | "around **20–30 µm**" spatial | — | — | [H] |
| GelSight Wedge | — | Gradient RMSE ≈ 0.043 (no µm figure) | — | [H] |
| DIGIT 360 | "7 µm" spatial detail (marketing) | — | — | [H] |

**Sources for this table:**
- Skin probe: https://arxiv.org/html/2509.11385v1
- 3D Cal: https://rohankotanu.github.io/3DCal/
- Mini/DIGIT FAQ: https://github.com/gelsightinc/gsrobotics

### 4.2 What limits µm performance

| Factor | Effect | Evidence |
|---|---|---|
| **Camera µm/px and optics** | You need ≤ 5–7 µm/px for 10 µm features. Wide-angle M12 lenses distort; the Mobile uses a 0.5X **telecentric** lens. | [H] Shot Peener; GelSlim 3.0 corrected spherical distortion from its 160° lens [H] |
| Diffraction vs depth of field (estimate) | At object-side NA ≈ 0.05: Rayleigh limit ≈ 7 µm, DOF ≈ 0.25 mm. At NA ≈ 0.10: ≈ 3.4 µm, but DOF only ≈ 55 µm. You can't have both very fine lateral detail and deep dents in focus. | Estimate (0.61λ/NA; λ/NA², λ = 0.55 µm) |
| **Coating** | Pigment particle and flake size, and coating thickness, set the finest copyable detail. 1 µm Al powder (matte) is standard. Semi-specular flake is more sensitive to small slopes but harder to calibrate. | [H] Yuan 2017; [H] FAQ ("metal flake … high contrast for fine details"); [M] RainbowSight https://arxiv.org/pdf/2409.13649 ; [H] edge-ai-vision (smaller flecks for µm) |
| **Gel thickness and hardness** | A thick transfer layer deforms less and loses spatial resolution; thin layers tear. Typical gels are Shore A 5–20; the Mini gel is 4.25 mm. | [H] DIGIT paper; [H] Yuan 2017; [H] Mini sheet |
| **Press force** | Low force means the gel doesn't fully conform, so depth is **underestimated** (skin probe used ~19.6 N). Soft samples (0.6 MPa) also read shallow. | [H] skin probe |
| **Calibration** | Indenter size should match the feature scale (0.5–1.25 mm-radius balls for µm work). Each gel needs its own calibration. Contact-depth labelling error was ≤ 15 µm even with a 3D-printer rig. | [H] skin probe; [H] Shot Peener (per-cartridge calibration) |
| **Integration error** | Depth comes from integrating gradients, so errors accumulate; there is no direct absolute depth. | [H] https://arxiv.org/abs/2606.17438 (abstract) |
| **Illumination non-uniformity and interreflection** | GelSlim 4.0 abandoned analytic photometric stereo because of dark spots. The skin probe saw x/y asymmetry from uneven LEDs. | [H] GelSlim 3.0/4.0; [H] skin probe |
| **Image quantisation and compression** | The classic LUT is for 8-bit images. H.264 adds latency. The gs_sdk FFmpeg path can duplicate frames. | [H] Yuan 2017; [H] gs_sdk |
| **Thermal drift** | The Mini case rises by 22.7 °C at 25 fps; preferred range 0–30 °C. | [H] Mini sheet |
| **Dirt and wear** | Dirt on the gel caused misclassifications. Worn coating degrades the image. | [H] Agrawal 2023; [H] DIGIT abrasion table |

### 4.3 Error-budget insight (estimate)

Integrating a gradient error σ_g over a feature n pixels wide at pixel pitch Δx:
- **Random errors** grow as σ_g·Δx·√n.
- **Bias** grows as σ_g·Δx·n.

Taking σ_g ≈ 0.043 (the Wedge figure) and Δx = 5 µm:

| Feature | Random error | Bias error |
|---|---|---|
| **50 µm-wide scratch** (10 px) | ≈ 0.7 µm | ≈ 2 µm |
| **1 mm-wide shallow dent** (200 px) | ≈ 3 µm | ≈ 43 µm |

**Conclusion.** Narrow scratches and pits are the easy case for µm depth. Broad, shallow dents are the hard case, and bias dominates: calibration bias, lighting non-uniformity and force. Measure depth against a locally fitted reference plane.

### 4.4 How a DIY build can approach ≤ 10 µm depth (checklist)

1. **Optics: ≤ 5 µm/px over a 10–16 mm field.** All values below are estimates.

   | Camera | Setup | Sampling | Notes |
   |---|---|---|---|
   | IMX219 (Pi Camera v2 / Arducam M12) | Stock 3.04 mm lens refocused to ~16 mm | **≈ 4.9 µm/px** over 16 mm | Diffraction ≈ 7 µm |
   | IMX477 HQ | 16 mm C-mount lens + ~6 mm extension, at ~57 mm | **≈ 4 µm/px** | Stop down to keep DOF |
   | Camera Module 3 (software autofocus) | At its 10 cm / 5 cm minimum focus | Only **≈ 28 / 24 µm/px** | Too coarse without close-up lenses |
   | AI Camera | At its 20 cm minimum focus | **≈ 64 µm/px** | Unusable for the probe |

   Focus distances: [H] https://www.raspberrypi.com/documentation/accessories/camera.html

   Focus once, then lock the lens. This conflicts with your "no manual focus ring" preference, but focus is fixed for a gel at a fixed distance. The GelSight Mini also ships "focus set" with a focus tool. [H]
2. **Lighting: ≥ 4–6 directions, one at a time,** like the Mobile, with a dark-frame subtraction (your `capture.py` already does this). This avoids RGB crosstalk and gives an over-determined photometric stereo (reasoning).
   - Keep single-shot RGB only as a live-preview mode.
   - Your 3 W LEDs are far brighter than needed (DIGIT uses ≤ 4 lm). Diffuse them, run them at low current, and watch gel heating.
3. **Sensor settings.** Lock exposure, gain and white balance. Avoid MJPEG/H.264, grab uncompressed or RAW frames where the camera stack allows, and average 4–16 frames per light (reasoning).
4. **Gel**
   - Soft (≈ Shore 10–20 A), optically clear platinum silicone, about 1.5–4 mm thick.
   - **Matte 1 µm Al powder** coat, thin and even, with a thin protective silicone overcoat (skin-probe recipe).
5. **Force.** Use a spring-loaded or load-cell handle with a constant press force (skin probe: ~19.6 N, load cell + HX711).
6. **Calibration**
   - Use precision steel balls (0.5–1.25 mm radius for fine work, 2–4 mm for the general map) at many positions.
   - A 3D printer is a good automatic prober (3D Cal; skin probe used 49 presses per ball).
   - Fit either a LUT or a small MLP mapping RGBXY to gradients (gs_sdk). Recalibrate for every new gel.
7. **Reconstruction**
   - Use DST Poisson on a local ROI around the defect.
   - Fit and subtract the undamaged surround (detrend / high-pass, as in Agrawal and the skin probe).
   - Report depth, width and length in µm.
8. **Validation.** Use step-height references such as stacked feeler-gauge shims of 10–50 µm (my suggestion; no source checked) and, if you can borrow one, a college profilometer. Report MAE and SD the way the skin probe does.
9. **Expectation**
   - About **10–30 µm MAE** in 4 weeks.
   - **≤ 10 µm** only if items 1–8 all work.
   - **Sub-µm**: commercial only.

---

## 5. DIY recipe (materials, prices, time)

### 5.1 Materials with published data

| Layer | Option | Data | Source |
|---|---|---|---|
| Clear gel | **Smooth-On Solaris** (DIGIT's gel) | 1A:1B, **Shore 15A**, 240 min pot life, **24 h cure** at 23 °C, "ultra-transparent", 1,200 cps, degas at 29 inHg. US trial unit (2 lb + primer) $80.76 [M] | [H] https://www.smooth-on.com/products/solaris/ ; [M] https://shop.smooth-on.com/platinum-cure-silicone-rubber/solaris-trade |
| Clear gel | Silicones Inc. **XP-565** (Wedge, GelSlim, skin probe) | Hardness set by the activator:base ratio: 1:10 (GelSlim 3.0), **1:22 cured at 55 °C** (skin probe). Several hours at RT, or ~15 min at 100 °C. No datasheet found | [H] GelSlim 3.0; [H] skin probe; [H] Yuan 2017 |
| Clear gel | **SORTA-Clear 18** | 100A:10B, 18A, "water clear translucent", **21,000 cps** (traps bubbles, so degas), 24 h | [H] https://www.smooth-on.com/products/sorta-clear-18/ |
| Gel (not ideal) | Ecoflex 00-30 | 00-30, "water white **translucent**" (hazy), 4 h | [H] https://www.smooth-on.com/products/ecoflex-00-30/ |
| Gel (too hard) | Sylgard 184 | 10:1, **43 Shore A**, clear; 48 h at 25 °C or 35 min at 100 °C | [H] https://www.dow.com/en-us/pdp.sylgard-184-silicone-elastomer-kit.01064291z.html |
| Coating, matte | **1 µm spherical Al powder**, brushed on with a makeup brush, plus a thin silicone overcoat (the earliest bench sensor used 1 µm silver) | Skin probe overcoat: SORTA-Clear 18 + NOVOCS Matte at 1:16.5:33 | [H] Yuan 2017; [H] skin probe |
| Coating, semi-specular | Al or bronze **flake** in silicone paint base (Cirius / Psycho Paint) + solvent, airbrushed in thin layers | — | [H] Yuan 2017 |
| Paint base | **Psycho Paint** | 1A:1B + 200–500 parts NOVOCS for airbrushing; pigment ≤ 5% of A+B; ~2 h cure; bonds only to platinum silicone; latex gloves inhibit cure. US pint $41.81; Silc Pig 4 oz $21.69 [M] | [H] https://www.smooth-on.com/products/psycho-paint/ |
| Avoid | Spray paint | "will crack eventually when pressed" | [H] https://hira.hope.ac.uk/2869/1/Paper2_LowCostGelsight_camera-ready.pdf |
| Support | Clear acrylic window (Wedge used a 4.5 mm block); glue with **Sil-Poxy** (DIGIT) or adhesive silicone sheet | Sil-Poxy: 5 min work time, 24 h full cure | [H] DIGIT paper; [H] https://www.smooth-on.com/products/sil-poxy/ ; [H] Wedge |
| Quick hack (week 1) | Clear silicone **cosmetic sponge** (Shore A 7) + Psycho Paint | Skips casting and degassing; qualitative only | [H] Abad et al. 2019 |

### 5.2 Build steps and time

1. **Housing.**
   - 3D-print the housing and laser-cut the acrylic window.
   - Mount the camera at the focus distance: about 15 mm (9DTact), 35 mm (C310) or ~16 mm (IMX219 estimate).
   - Fit LEDs on 3–6 sides, aimed through the acrylic edge or at a grazing angle (Mini: RGB on 3 sides via waveguides; Yuan: 2×4 arrays tilted 71°). [H]
   - About 2–3 days (estimate).
2. **Gel.**
   - Cast 1.5–4 mm of clear platinum silicone on a flat, clean mould, degas, and cure.
   - About 1 day per cast (Solaris: 24 h). Pour 3–4 variants in parallel (thickness and hardness).
3. **Coating.**
   - Brush 1 µm Al powder onto the cured surface, or airbrush a pigmented Psycho Paint layer.
   - Add a thin overcoat and cure: about 2 h to 1 day.
4. **Bond and assemble.** Sil-Poxy (24 h full cure).
5. **Calibrate.** Ball presses, then fit a LUT or MLP: under 1 h per gel once the tooling exists (gs_sdk).
6. **Reconstruct.** DST/FFT Poisson. Reuse your `src/trinetra/shape.py: integrate_slopes()` (Frankot–Chellappa FFT).

**Total (estimate):** about 1–1.5 weeks to a first calibrated sensor, assuming the materials are already in hand. Expect 2–3 gel iterations.

### 5.3 Failure modes

| Failure | Evidence |
|---|---|
| Bubbles (degas needed) | [H] Smooth-On pages; [H] Abad 2019 |
| Coating cracks (spray paint) | [H] Abad 2019 |
| Coating abrasion: GelSight Inc. gel 475 / 662 / 918% degradation after 5 / 10 / 15 passes; DIGIT gel 0 / 0.3 / 0.3% | [H] DIGIT paper |
| Delamination and tearing | [M] https://arxiv.org/pdf/2504.19341 |
| Dust in the coating; viscoelastic hysteresis; surface bumps adding noise | [H] Yuan 2017 |
| Cure inhibition (latex gloves) | [H] Psycho Paint page |
| Uneven lighting | [H] GelSlim 4.0; [H] skin probe |

---

## 6. Evidence for defect inspection

| Study | Sensor | Parts / defects | Results | Source |
|---|---|---|---|---|
| **Agrawal et al. 2023** (CMU + Siemens + Boeing + GelSight) | GelSight Mobile 0.5X on a robot with a compliant mount | Boeing curved panels; scratches ≥ **10 µm deep, 51 µm wide**; drill runs and gouges ≥ 20 µm deep | See below | [H] https://ar5iv.labs.arxiv.org/html/2309.04590 |
| Jiang et al. 2021 | GelSight-type, 14 × 10.5 mm, 20–30 µm | Mock-up cracks (gaps) + painted fakes | Segmentation IoU 0.376 → **0.636** with touch; profile error 0.82 → **0.24 mm**; 35 s active vs 400 s passive. **Hairline closed cracks not tested** | [H] https://ar5iv.labs.arxiv.org/html/2105.06325 |
| GelSight / Shot Peener 2022 | Mobile 0.5X | Almen strips, peening coverage | Uncertainty < 1%; Gage R&R < 20% of tolerance | [H] https://www.shotpeener.com/library/pdf/2022041.pdf |
| MTU Aero Engines | GelSight handheld | Engine-assembly scratches and dents | In production use since 2019 | [H] aeroreport |
| Skin probe 2025 | Mini hardware + custom gel | Channels 12–96 µm vs Keyence | MAE **12.55 µm**; R² 0.94–0.98 | [H] https://arxiv.org/html/2509.11385v1 |
| 3D Cal 2025 | DIGIT / Mini + CNN | mm-scale 3D-printed shapes | **156 / 205 µm** average error | [H] https://rohankotanu.github.io/3DCal/ |

**Agrawal et al. 2023 results**
- Tactile classification: **95.75%**.
- Robot test: vision-only P 0.62 / R 0.79; tactile or two-stage **P = R = 1.0**.
- Time per capture: tactile **22.3 s** vs visual 6.5 s.
- Whole panel: tactile-only **12,978 s** vs two-stage **168 s**.
- The tactile stage used classical Canny + Hough + depth profile + thresholds, not a CNN.

---

## 7. Limits

- **Area per press**
  - 17 × 14 mm (Mobile), 18.6 × 14.3 mm (Mini), 19 × 16 mm (DIGIT). [H]
  - A flat sensor covered 8 × 11 cm in **196 s**, vs **10 s** for the TouchRoller roller sensor. [H] https://arxiv.org/abs/2103.00595
  - That flat sensor needed "≥ 49 presses". [M] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10007629/
  - **Strategy:** vision localises, touch measures. That turns roughly 3.6 h into roughly 3 min per panel (Agrawal).
  - GelSight added automated image stitching in Dec 2024. [M] press-release title only
- **Tight cracks with no relief**
  - The method measures only topography. Open cracks, or cracks with edge relief, are measurable: the vendor lists "cracks", and Jiang used open gaps.
  - **I found no evidence for closed, zero-opening cracks.** Keep the red-dye penetrant (reasoning).
- **No colour.** The opaque skin hides the surface's optics by design [H FAQ]. Discolouration, heat tint and corrosion colour need the camera channel.
- **Contact force.**
  - Low force means depth is underestimated. [H] skin probe
  - Use a compliant mount or force control. [H] Agrawal; skin probe
- **Hard vs soft parts.** Hard metal is the good case. Soft parts deform under the gel and read shallow (0.6 MPa samples). [H] skin probe
- **Gel life and cost**
  - Mini: 1000 coin presses [H]; the Jan 2025 gels last up to 40% longer. [H] https://www.anzupartners.com/2025/01/06/gelsight-extends-life-and-durability-of-gelsight-mini-with-new-gels/
  - Replacement cost: Mobile $600, Mini $57, DIGIT $42. [H]
  - Clean the part first; GelSight sells a $50 cleaning kit. [H]
- **Temperature.** Mini case Δ +22.7 °C; preferred 0–30 °C. [H]

---

## 8. Sourcing in India and Bengaluru

**Smooth-On authorised distributors (Full Line)** [H] https://www.smooth-on.com/distributor-regions/middle-east-india/:
- **ONS Impex, Mumbai**: +91 9594023001, opmishra@onsimpex.net. Site lists Ecoflex and Silc Pig, prices on quote. [H] https://onsimpex.in/
- **Pearl Impex, Delhi and Chennai**: +91 9717868972, pearlimpex22@gmail.com.

**No Bengaluru distributor.** Ask for Solaris, SORTA-Clear 18, Psycho Paint, NOVOCS and Sil-Poxy quotes.

| Item | Seller (city) | Price | Conf. | URL |
|---|---|---|---|---|
| **Fortisil 20A Clear LV** / 40A Clear platinum RTV-2 (the spec table says "translucent") | Fortius Marketing (Pune) | **₹1,300/kg + taxes**; 1+1 kg kit | [H] | https://m.indiamart.com/proddetail/fortisi-40a-clear-platinum-cure-liquid-silicone-rubber-rtv2-2858222303773.html |
| Trisil A-20 translucent platinum, 10–25 ShA | Triotex (Pune) | ₹1,200/kg | [M] | https://m.indiamart.com/triotex-multiservices |
| Zhermack ZA 35-15 GLASS (25 ShA) | IndiaMART | ₹2,440/kg | [M] | https://m.indiamart.com/impcat/silicone-rubber.html |
| "White Smooth On Silicone Rubber" (ONS listing; product unnamed) | IndiaMART | ₹7,000/kg | [M] | https://www.indiamart.com/proddetail/smooth-on-silicone-rubber-2850951920262.html |
| TLS-130 translucent platinum | Tinflix (**Bengaluru**) | ₹850/kg (pack info inconsistent) | [L] | https://m.indiamart.com/tinflix-enterprises/ |
| Sylgard 184 | — | No INR price found; Indian labs buy it by tender. Farnell BE €262.59 per 1.1 kg | [M] | https://be.farnell.com/dowsil-formerly-dow-corning/sylgard-184-1-1kg/elastomer-clear-184-1-1kg/dp/101697 |
| 1–2 µm spherical Al powder, 99.9% | Nanografi (EU; India shipping unconfirmed) | €62/100 g | [M] | https://shop.nanografi.com/microparticles/aluminum-al-micron-powder-purity-99-9-size-1-2-m/ |
| Paint-grade aluminium paste / powder (particle size unknown) | IndiaMART | ₹250–600/kg, mostly bulk | [M] | https://m.indiamart.com/impcat/aluminium-paste.html |
| Arducam IMX219 8 MP + **M12** lens LS40136 (CSI) | ThinkRobotics | ₹3,999.99 (out of stock at capture) | [M] | https://thinkrobotics.com/ar/products/arducam-8-mp-sony-imx219-camera-with-ls40136 |
| Arducam 8 MP IMX219 USB **UVC** module | The Pi Hut (UK) | £28.80 (no Indian price found) | [M] | https://thepihut.com/collections/raspberry-pi-camera/products/8mp-1080p-usb-camera-module-1-4-cmos-imx219-mini-uvc |
| Raspberry Pi Camera Module 3 | Thingbits / KSP | ≈ ₹3,005–3,099 | [M] | https://thingbits.in/products/raspberry-pi-camera-module-3 ; https://kspelectronics.in/product/raspberry-pi-camera-module-3-standard/ |
| GelSight Mini System | GelSight store (USA) | **$510** + shipping and duty (not quoted; India not mentioned) | [H] | https://www.gelsight.com/online-store/ |

Not found (search budget): acrylic sheet, precision steel balls, small RGB LEDs, vacuum degassing kit and feeler-gauge prices in Bengaluru.

---

## 9. Compute (Pi 5 + Hailo-8 AI HAT+)

- **Poisson / Frankot–Chellappa on the Pi 5 CPU.**
  - FFT/DST, O(N log N). Yuan 2017 says the DST solver "allows online operation". [H]
  - I found no published Pi 5 timing. Estimates:
    - 320×240: ~12 MFLOP, a few ms.
    - 640×480: ~10–25 ms, so **video rate is feasible**.
    - Full 8 MP for µm work: roughly 0.5–1 s. That is acceptable for press-and-measure; GelSight Mobile itself takes "seconds". [H]
  - A ~1000×1000 ROI is fast.
- **RGB→gradient.**
  - A LUT is trivial on the CPU.
  - A per-pixel MLP 5-32-32-32-2 costs ≈ 0.35 GFLOP per frame at 320×240 (estimate), roughly 10–30 fps on the CPU.
- **Hailo-8.**
  - Accepts ONNX/TFLite. Supports convolution, depthwise convolution and **tanh**. [M] Hailo community post https://community.hailo.ai/t/supported-operators/5046/2
  - The MLP can be written as 1×1 convolutions, so it should compile. But Hailo runs INT8, which may cost µm-level gradient precision, so keep gradients on the CPU for metrology (reasoning).
  - STFT is listed as unsupported, so keep FFT/Poisson on the CPU.
  - **Defect segmentation on Hailo is easy.** Hailo-8 Model Zoo [H] https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_semantic_segmentation.rst:
    - unet_mobilenet_v2 at 256×256: **662 FPS** (float mIoU 77.3 vs quantised 77.2)
    - deeplab_v3_mobilenet_v2 at 513×513: **121 FPS**
  - Those numbers were measured on an x86 host over PCIe Gen3 x4. The Pi 5 has a **single-lane PCIe, Gen 2.0 by default**; Gen 3 is optional and "isn't certified". [H] https://www.raspberrypi.com/documentation/computers/raspberry-pi.html
- **Pipeline split (fits the brief).**
  - Hailo: defect segmentation.
  - CPU: gradients, Poisson and metrics.
  - JSON (depth_um, width_um, length_um, area, gel_id, calibration_id, force_N), then Llama on the CPU via ExecuTorch for operator advice.
  - Llama runs after the measurement, so it doesn't compete with capture.

---

## 10. Feasibility verdict: proof of concept due 23 Nov 2026 (~34 days)

| Option | µm capability | Cost / lead | Risk | Verdict |
|---|---|---|---|---|
| **A. DIY gel probe** (IMX219/HQ optics, 4–6-direction sequential LEDs, 1 µm Al coating, ball calibration, Poisson) | ~5 µm/px sampling; **~10–30 µm depth MAE** realistic; ≤ 10 µm is a stretch | Core sourced items: silicone ₹2,600 + taxes (2 kg Fortisil) + camera ₹3–4k + pigment; plus acrylic, LEDs, balls and printing (prices not found) | Gel iterations (24 h cures), bubbles, coating, calibration, Pune/Mumbai/EU shipping | **Feasible as a tens-of-µm demonstrator.** Not a validated ≤ 10 µm instrument in 4 weeks |
| B. GelSight Mini ($510) + own calibration | Same class as A, with better gel and lighting quality; skin probe reached ~12.5 µm with custom work | $510 + import (timing unknown); not buyable in Bengaluru | Import lead time; not a metrology device; ARM support unverified | Good **backup or reference** if it arrives by about 10 Nov |
| C. GelSight Mobile / Max | **1 µm + 4%** / 0.3 µm + 4% | **$30–55K** | — | Not realistic for a student PoC |

**Recommended plan (estimate)**

| Week | Dates | Work |
|---|---|---|
| 1 | 21–27 Oct | Order silicone (Fortius) and Smooth-On quotes (ONS/Pearl), 1 µm Al powder and an IMX219 M12 camera. Build the housing. Develop software on a sponge-hack gel. |
| 2 | 28 Oct–3 Nov | Cast 3–4 gels in parallel, coat, assemble, calibrate with steel balls. Reuse `integrate_slopes`. |
| 3 | 4–10 Nov | Validate on shim steps and on scratched or dented Al plates. Train the Hailo segmentation model. Build the JSON → Llama step. |
| 4 | 11–17 Nov | Integrate with TriNetra: the camera flags a defect, the probe measures µm depth. Record the demo video and write the report. |
| Buffer | to 23 Nov | — |

**Go / no-go on about 3 Nov:** a 50 µm shim step must measure within ±10 µm. If it doesn't, keep the probe as roadmap and ship TriNetra camera-only.

**Fit with the brief and your saved preferences**
- The probe is camera-based, but it is contact and small-area.
- Present it as an **optional measurement mode** under the brief's camera inspection, not a replacement for it.
- Its sourcing doesn't meet your "buy in Bengaluru" rule: Smooth-On is in Mumbai/Delhi/Chennai, the gel options are in Pune, and the Mini comes from the USA. Check GST invoices.
- Focus is set once by hand, which conflicts with your "software focus" camera rule.

---

## 11. Gaps and unverified items

- **Pi 5 runtime.** No published Pi 5 timings for FFT/Poisson or for GelSight software on ARM. Benchmark with `numpy.fft` / `scipy.fft.dstn` on 320×240, 640×480 and full frames.
- **Indian prices still missing.**
  - XP-565: no datasheet or Indian seller found.
  - Smooth-On prices from ONS/Pearl Impex.
  - Sylgard 184.
  - Bengaluru prices for acrylic, LEDs, balls and shims.
- **Commercial details.**
  - Whether GelSight has an Indian distributor, or ships the Mini to India.
  - Full Kalitta Air case study details.
  - Automotive-paint, lightning-strike or rivet case studies.
  - GelSight Mobile depth repeatability (Gage R&R) figures.
  - Johnson 2011 paper internals (the PDF is 27 MB, too large to fetch).
- **Low-confidence items.** The Forbes $40K figure [L]. The TouchRoller "49 presses" [M]. The image-stitching release [M].
