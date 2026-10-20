Status: COMPLETE, last updated 21 Oct 2026 after merging the scanner (file 01) and gel-probe (file 02) findings into the summary, §3.5-3.6 and §7

# 03 - Handheld, room-light, µm-level defect inspection: industry practice and a 4-week PoC architecture

Prepared 20 Oct 2026 for the Bharat AI-SoC Challenge, Manufacturing theme, PS5 (real-time industrial defect detection and QC assistant). Deadline for report + code + demo: **23 Nov 2026**.

**Confidence tags**
- **[H]** I read the fact on the source page or PDF myself.
- **[M]** Search-result snippet, reseller or secondary source only. Treat as medium confidence.
- **[L]** My own calculation or inference. The inputs are cited; the result is not a published spec.

**Method note:** the shared web-search budget ran out partway through this research. Facts after that point come from fetching known URLs directly. Gaps are listed in §9.

---

## Executive summary

1. **Every handheld tool that reaches µm-scale detail references the part mechanically and brings its own light.** Dino-Lite uses caps that touch the part [H], GapGun a contact standoff [H], GelSight presses a gel on the part [M], and dermatoscopes use contact plates [M]. I found no handheld µm tool that beats room light and hand motion with optics alone.
2. **Room light is beaten by distance, not by cleverness.**
   - A phone-flash-class LED (260 lm) gives about 33,000-92,000 lx at 3-5 cm. That is 66-184× a 500 lx office.
   - At 10 cm it gives only about 17×, and at 1 m about 0.17× [L, from the Lumileds datasheet, H].
   - Flash-on/off subtraction, cross-polarisation (costs 6-10× light [H]), bandpass filters and global shutter are clean-up tools on top of that.
3. **Hand tremor (about 50-100 µm RMS, ~10 Hz [M]) rules out free-hand µm imaging.** Hovering at 1.5 µm/px needs ≤0.17-0.34 ms exposures for ≤1 px blur, and the depth of field is only 0.07-0.4 mm [L]. Resting the nose on the part fixes both problems.
4. **µm-level sampling is easy; µm-level resolving is not.**
   - The Raspberry Pi HQ camera (1.55 µm pixels) with a 16 mm lens on extension gives 1.55-3.1 µm/px over 6.3-12.6 mm. The lens limits real resolution to about 5-8 µm at f/4 [L].
   - Truly resolving 1-2 µm needs a 10x/0.25 objective: about 2.3 × 1.7 mm field and about 11 µm depth of field [L].
   - A Dino-Lite 1.3 MP handheld microscope gives 6.1 µm/px at 50x and 1.6 µm/px at 200x [H datasheet + L].
5. **Detection is not measurement.** Low-angle (dark-field) light makes scratches far narrower than a pixel visible [M]. So a wide "find" view can flag µm-wide scratches that only a close "measure" view can size. Industry does exactly this two-stage split: wafer inspection then SEM review, AOI then a verification station, Waygate's "locate then measure" [M/H].
6. **µm-level depth, handheld, is published only for tools that rest on the part over a small field.**
   - GelSight Mobile: Z 1 µm + 4%, 17 × 14 mm [M, vendor].
   - 4D InSpec interferometer: 4 µm vertical resolution, 8 × 8 mm [M].
   - GapGun FOV7 laser head: 3 µm resolution over 6.5 mm [H].
   - No free-motion handheld 3D scanner reaches ≤10 µm depth. The best metrology laser scanners quote 12-25 µm, cost $24-65k and need Windows + GPU; prosumer ones want spray on shiny metal (file 01).
   - Videoscope 3D phase/stereo is about ±50 µm at best [H].
   - Focus variation needs a vertical scan and I found no handheld version [H].
   - A **DIY GelSight-style gel cap** is feasible in about 1-1.5 weeks at about 10-30 µm depth error and about 5 µm/px (file 02 estimate). It is the only µm-ish depth option for 23 Nov.
   - The published two-stage "vision then touch" result: whole panel in 168 s vs 12,978 s touch-only, with P = R = 1.0 vs vision-only P 0.62 / R 0.79 ([Agrawal et al. 2023](https://ar5iv.labs.arxiv.org/html/2309.04590), via file 02).
7. **Android vs Pi.**
   - A phone wins on Llama speed (Llama 3.2 1B at 50.2 tok/s decode on a OnePlus 12 [H]) and on size.
   - The Pi 5 + AI HAT+ wins on camera and lighting control, Hailo vision (YOLO26n 21.7 ms [H]) and reuse of the team's existing Python code.
   - Recommendation: Pi for 23 Nov; phone + USB-C "nose" accessory as the scale-up path (a shipped product category, e.g., FLIR ONE Pro [H]).
8. **Recommended PoC: "TriNetra-Handheld: find + touch-measure".**
   - A Camera Module 3 "find" view at about 10 cm (≈28 µm/px over 130 × 73 mm) with its own LED ring.
   - A Raspberry Pi HQ camera "measure" nose (≈1.55-3.1 µm/px, 6-12 mm field) with **two swappable caps**:
     - **Optical cap:** 15-30 mm, light-tight, with bright-field, dark-field, 4-direction and cross-polarised LEDs. Gives width, length and colour.
     - **Gel cap (optional, week 2-3):** a DIY GelSight-style pad lit by the same 4-direction LEDs at about 4-5 µm/px. Gives depth.
   - HailoRT for vision; Llama 3.2 1B via ExecuTorch on the CPU for advice; a rules engine for severity; SQLite logging.
9. **Honest µm statement.** In 4 weeks the device can:
   - *sample* at 1.55 µm/px and *resolve* about 5-8 µm;
   - *detect* thinner scratches with dark-field;
   - give *relative* depth maps from the optical cap;
   - with the gel cap, give *calibrated* depth at about 10-30 µm error (≤10 µm is a stretch).

   True 1 µm depth needs GelSight Mobile-class hardware (US $30-55k, file 02) or interferometry (4D InSpec, 4 µm). That is roadmap.
10. **The cap is not a box.** The user rejected a box resting on the part. The nose here is a Dino-Lite-style 15-30 mm cap at the tip of a handheld tool. Without any contact, expect roughly 5-10 µm/px effective sampling and no µm depth.

---

## 1. Handheld inspection in industry, and how each tool copes with room light and hand motion

### 1.1 Videoscopes with 3D measurement

| Item | Fact | Conf. | Source |
|---|---|---|---|
| Waygate Real3D Phase (3DPM) | Structured-light fringe patterns projected from the tip give a fully surfaced XYZ point cloud. The white-light image can be out of focus and still give valid XYZ data. | H | [Waygate white paper BHCS38986](https://dam.bakerhughes.com/m/54cb0022a30973e7/original/Surface-Indication-Measurem_BHCS38986-1.pdf) |
| Waygate Real3D Stereo | Matches pixels between two stereoscopic images. Needs sharp focus, a feature-rich surface and no glare. | H | same |
| 3D Phase "ideal conditions" | Trained technician, tip very close to the indication, non-perpendicular incidence, matte surface, minimal reflections, and **"Absence of ambient light"**. | H | same |
| Accuracy | Minimum recommended feature 0.125 mm. Under ideal conditions, errors of ±0.05 mm or better on length and depth. All 3D borescopes use triangulation, so depth error grows with distance. Avoid incidence above about 60°. Run a Gage R&R on representative parts. | H | same |
| MViQ hardware | 1/6" CCD. HD 6.1 mm probe: 1,200k pixels (SD: 440k). White LED illumination. Auto long exposure up to 16 s. 73 Wh Li-ion, 2 h battery. 3.0 kg out of case. | H | [MViQ datasheet](https://www.instrumart.com/assets/Waygate-MVIQ-datasheet.pdf) |
| Probe motion (GE patent family) | Inspection light is switched off in measurement mode so the fringes keep contrast. At least one image per fringe set (the example uses 3). Motion is checked by comparing images before and after the fringe sequence (one example limit: 1 px). On motion it re-captures until motion is low or a timeout hits. Repeated fringe images can be summed; multiple exposures help on reflective parts. | H | [US8107083B2](https://patents.google.com/patent/US8107083) |
| On-device AI | Everest Mentor Visual iQ+ "uses computer vision and machine learning". The analytics "operate entirely standalone within the device". ADR (assisted defect recognition) annotates automatically for the user to approve. Menu Directed Inspection gives standard templates and reports. | H | [Everest MViQ+ page](https://www.bakerhughes.com/waygate-technologies/remote-visual-inspection-rvi/video-borescopes/everest-mentor-visual-iq) |
| Evident IPLEX NX stereo | Triangulation and parallax from the perspective shift across the CCD (two views in one frame). Tip-to-target 4-60 mm, about 40 mm recommended. On a highly reflective 100 mm calibrated rule: 99.79% accuracy where a standard videoscope gave no measurement. | H | [Evident application note](https://ims.evidentscientific.com/en/applications/rvi-exploring-the-latest-advances-3d-measurement) |
| IPLEX NX light and power | Laser-diode illumination "up to 4 times brighter than conventional video borescopes". Multi-spot ranging gives live distances to several points. 14.8 V battery, about 100 min. 7.1-7.6 kg. | H | [IPLEX NX page](https://ims.evidentscientific.com/en/products/video-borescopes/iplex-nx) |
| Evident and phase measurement | I found no fringe/phase measurement in Evident's current lineup, only stereo plus spot ranging. | M | [Evident measurement intro](https://www.olympus-ims.com/remote-visual-inspection/videoscope/introduction-to-visual-inspection-measurement-solutions/) |

**How they cope:** they work inside dark machinery with their own LED or laser light. Phase measurement takes several frames, so it needs no ambient light and a motion check with re-capture. Stereo is single-shot, so it tolerates motion but needs texture and no glare. At about ±50 µm best case, **videoscope 3D is not a µm tool**.

### 1.2 Handheld digital microscopes

| Item | Fact | Conf. | Source |
|---|---|---|---|
| Dino-Lite AM4515ZT (Edge) | 1.3 MP (1280×1024), 20-220x, 8 white LEDs, built-in linear polarizer "reduces glare and reflection on shiny objects", UVC, USB 2.0, 30 fps. | H | [Dino-Lite Europe datasheet (TME)](https://www.tme.com/Document/f744b6938c16708c8bc59c5eb0376a27/AM4515ZT.pdf) |
| FOV and DOF table, same datasheet | 20x: 19.5 × 15.6 mm, DOF 2.5 mm. 50x: 7.8 × 6.3 mm. 100x: 3.9 × 3.1 mm. 200x: 2.0 × 1.6 mm. 220x: 1.8 × 1.4 mm, DOF 0.1 mm. Working distance at 50x is 6.4 mm with the front cap. | H | same |
| Dino-Lite caps | The 9.5 mm open cap "eases the task of finding and holding a fixed focus when using the Dino-Lite in handheld mode". | H | [Dino-Lite N3C-O](https://dino-lite.eu/en/products/accessories/caps-extensions/n3c-o-open-cap) |
| Other caps | Extended 12.5 mm cap: standard models focus at about 50x and 200x when the cap edge touches. Long 30 mm cap: about 30x. Sidelight and diffuser caps. | M | [LFC blog](https://www.lfc.com.sg/blog/detail/dino-lite-caps) |
| Dino-Lite Edge 5 MP | 2592×1944, 10-140x / 10-220x; FLC = 4 LED quadrants switchable in software, 6 levels. | M | [TEquipment AM73915MZTL](https://www.tequipment.net/Dino-Lite/AM73915MZTL/Digital-Microscopes/) |
| AI in handheld microscopes | DinoCapture has measurement and calibration but no AI detection that I could find. Evident DSX2000 (Jan 2025) has "Live AI" but is a motorized benchtop unit. | M | [DinoCapture](https://www.dinolite.us/features/dinocapture/), [Evident DSX2000](https://evidentscientific.com/en/news/new-dsx2000-digital-microscope) |
| Keyence VHX | Lens and camera can be removed from the stand for handheld use. | M | [Keyence glossary](https://www.keyence.com.sg/ss/products/microscope/microscope_glossary/related_devices/free-angle_observation_system.jsp) |

**Sampling of the 1.3 MP Dino-Lite [L, from the datasheet FOV]:** 15.2 µm/px at 20x, 7.7 at 40x, 6.1 at 50x, 4.4 at 70x, 3.0 at 100x, 2.0 at 150x, 1.6 at 200x, 1.4 at 220x.

**How they cope:** the cap rests on the part. That fixes focus (DOF is only 0.1 mm at 220x), shades the field, and removes most tremor. LEDs a few mm away swamp room light, and the polarizer cuts glare.

### 1.3 Contact-gel 3D (GelSight; details in `02_gelsight_tactile_3d.md`)

| Item | Fact | Conf. | Source |
|---|---|---|---|
| Principle | An elastomer gel conforms to the surface and is imaged by an internal camera and lights. Vendor: works "regardless of composition, reflectivity, transparency, or ambient lighting conditions". | M | [Hexagon GelSight Mobile](https://nexus.hexagon.com/home/product/gelsight-mobile) |
| GelSight Mobile Series 2 | FOV about 17.0 × 14.2 mm, about 100 ms capture, X-Y accuracy 3 µm + 0.2%, Z accuracy 1 µm + 4% (1-50 µm). | M | [NDTEC listing](https://shop.ndtec.net/en-us/gelsight-mobile-system-series-2-surface-analysis-i-5100--10000--30502) |
| GelSight Max | Measures scratches and pitting "down to the submicron level" (vendor). | M | [Anzu Partners](https://www.anzupartners.com/2023/05/03/introducing-gelsight-max-the-highest-resolution-mobile-surface-characterization-tool-on-the-market) |
| GelSight Mini | Vendor page: "mm-level surface data in three dimensions"; aimed at robotics; handles reflective and transparent materials. Its product sheet (GS_Mini 4.3.24, linked from the page): 8 MP camera, 25 fps, RGB LEDs, 18.6 × 14.3 mm field, Lambertian silicone gel, USB Micro-B to USB-C, Ubuntu/ROS. That is about 5.7 µm/px if 3264 px wide [L]. | H (page) / M (sheet) | [GelSight Mini](https://www.gelsight.com/gelsightmini/) |

**How it copes:** the opaque gel blocks room light and turns shiny metal matte; the sensor is pressed on, so hand motion doesn't matter. This is the only handheld family with published µm depth, and only for the Mobile/Max units.

### 1.4 Handheld laser profile gauges

| Item | Fact | Conf. | Source |
|---|---|---|---|
| GapGun Pro | Laser triangulation; swappable VChange heads; patented adjustable laser cap; internal battery about 4 h plus 6.5 h clip-on; Wi-Fi to PC; "up to 20x" faster than manual methods. | H | [third.com GapGun Pro](https://www.third.com/products/gapgun-pro/) |
| Head FOV vs resolution | FOV7: 6.5 mm, 3 µm. FOV15: 25 mm, 11 µm. FOV40: 55 mm, 24 µm. FOV80: 90 mm, 40 µm. | H | [Vectro brochure (ftr.co.jp)](https://ftr.co.jp/n/eng/common/pdf/Vectro.pdf) |
| Positioning and workflow | A removable standoff allows contact measurement "from a certain distance and position"; the head rotates for a stable posture; per-head laser modulation handles carbon fibre, gloss coat and chrome; check plans synced to many units over Wi-Fi; data sent automatically and reports generated. | H | [FTR GapGun features](https://ftr.co.jp/n/eng/products/gapgun/feature.html) |
| Laser | Class 2M with automatic laser intensity adjustment. | M | [IPROS listing](https://pr.mono.ipros.com/en/ftr/product/detail/2000319384/) |
| Servo-Robot WiKi-SCAN | Handheld laser weld gauge: hold to the joint, pull the trigger, GO/NO-GO against tolerances; measures leg size, concavity, undercut, mismatch. One distributor lists it as discontinued. | M | [Servo-Robot article](https://servo-robot.com/wiki-scan-a-revolution-in-weld-quality-management/), [Berg listing](https://www.bergeng.com/product/P13-0100P.html) |
| Ambient light in laser sensors | Bandpass filter around the laser line. Laser drift of about 0.3 nm/°C forces wider passbands (e.g., 31.5 nm). Strong sunlight can still overwhelm it. | M | [US10627493](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10627493) |

**How they cope:** narrowband laser + bandpass filter for room light; contact standoff and posture design for motion. I found no published GapGun ambient-light spec.

### 1.5 Smartphone / tablet inspection apps with AI

| App | Fact | Conf. | Source |
|---|---|---|---|
| Enao Vision (iPhone); article by its CEO, VSD, 1 Apr 2026 | The 48 MP iPhone 15 Pro is "more than good enough for most surface inspection tasks". Users expect no light box, so software must compensate for ambient light. Reflective and transparent parts need substantial work. 100 ms per part. An industrial setup costs "lower to mid 5 digits". Vendor-authored. | H | [Vision Systems Design](https://www.vision-systems.com/cameras-accessories/article/55368033/is-smartphone-based-visual-inspection-the-future-of-quality-control) |
| Telit deviceWISE VI Assistant (30 Jul 2025) | iPhone/iPad, iOS 18.2+, predictions on locally stored models. | H | [Telit press release](https://www.telit.com/press/telit-cinterion-devicewise-vi-assistant-app-for-iphone-and-ipad-extends-ai-powered-visual-inspection/) |
| TOIVIS | Learns from 4-20 good images, heatmaps, on device. | M | [Product Hunt](https://www.producthunt.com/posts/1241309) |
| Musashi AI Cendiant Gear Inspect Mobile | iPad bevel-gear defect detection on device; images not stored or sent. | M | [App Store](https://apps.apple.com/us/app/id1616240540) |
| IBM Maximo Visual Inspection Mobile | Collect, train and run models on iOS. | M | [Capgemini](https://www.capgemini.com/us-en/insights/expert-perspectives/simplifying-visual-inspection/) |

**How they cope:** no lux or motion specs are published. They rely on computational photography and the operator. They are "find"-class tools (mm-scale), not µm-scale.

---

## 2. Making camera inspection robust to room light, handheld

### 2.1 Overpower room light: get close, use a bright pulse, keep exposure short

| Quantity | Value | Conf. | Source |
|---|---|---|---|
| Office lighting | EN 12464-1: 500 lx maintained for writing, typing, reading. | M | [EN 12464-1 code finder](https://www.performanceinlighting.com/ww/en/en-12464-1) |
| Phone flash LED | Lumileds LUXEON Flash 7 (LXCL-EY17): 220 min / 260 typ lm at 1000 mA, single 20 ms pulse, 120° viewing angle. 1.5 A max pulse (≤200 ms, ≤30% duty). 500 mA continuous. | H | [LUXEON Flash 7 datasheet DS121](https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/196/LXCL_2D00_EY17.pdf) |
| Illuminance from one such LED (Lambertian, on axis) | about 83 lx at 1 m, 920 lx at 30 cm, 8,300 lx at 10 cm (17×), 33,000 lx at 5 cm (66×), 92,000 lx at 3 cm (184×). | L | computed from the datasheet |
| Cross-check | About 70 lx at 1 m per older Lumileds flash LED at 600 mA; two LEDs give 110-140 lx at 1 m. | M | [MaxLinear ANP-26](https://maxlinear.com/appnote/anp-26_poweringflashledssp7685_031306.pdf) |
| Strobe overdrive example | LED array strobed 1 ms at 500%, exposure 250 µs. LED output can fall by up to 40% from 25 to 90 °C. Test: unplug the light and trigger; a black image means ambient doesn't matter. | H | [VSD, Lockard, 5 Jun 2019](https://www.vision-systems.com/cameras-accessories/article/16739383/when-to-invest-in-led-strobe-controllers) |
| Overdrive arithmetic | 200% overdrive lets you halve exposure, cutting the ambient effect 4×. | M | [1stVision (Gardasoft)](https://www.1stvision.com/machine-vision-solutions/tag/gardasoft) |
| Overdrive limit | 6-8× for most high-brightness LEDs at about 150 µs pulse and 0.15% duty. | M | [Quality Magazine](https://qualitymag.com/articles/89618-strobe-lighting-for-machine-vision) |
| Hand tremor | About ≤50 µm RMS per axis at 8-12 Hz (commonly quoted); Riviere 2006: ≤100 µm RMS, 6-12 Hz. | M | [Ang slides](https://www.lirmm.fr/uee07/presentations/lecturers/Ang.pdf), [CMU RI](https://www.ri.cmu.edu/app/uploads/2018/01/ProcIEEE-RIcopy.pdf) |
| Exposure for ≤1 px blur while hovering | Peak tremor speed 4.4-8.9 mm/s. At 1.5 µm/px: 0.17-0.34 ms. At 5 µm/px: 0.56-1.1 ms. At 20 µm/px: 2.3-4.5 ms. At 60 µm/px: 6.8-13.5 ms. | L | computed |

**Take-away [L]:** your own light must be within about 3-5 cm to dominate a 500 lx room, or within 10 cm with several LEDs. In find mode at 10 cm, a ring of 6+ LEDs gets to roughly 50-100× ambient. In a contact nose, room light is near zero anyway.

### 2.2 Flash-on / flash-off subtraction

| Fact | Conf. | Source |
|---|---|---|
| Flash-only image = (flash + ambient) − ambient, each scaled by its exposure. | M | [Petschnigg et al. 2004](https://www.cs.princeton.edu/courses/archive/fall14/cos526/papers/petschnigg04.pdf), [US7457477B2](https://patents.google.com/patent/US7457477B2) |
| If the camera or scene moves between shots, the two images must be registered first. | M | [US7457477B2](https://patents.google.com/patent/US7457477B2) |
| Short pulses + filtering + subtraction recovered a flash image at 1/37,000 of ambient power (UBC thesis claim). | M | [UBC thesis](https://open.library.ubc.ca/collections/24/items/1.0071707) |
| 50 Hz mains → 100 Hz light ripple; rolling shutter + short exposure → bands; multiples of 10 ms average it out. | M | [Kasson](https://blog.kasson.com/nikon-z6-7/ameliorating-lighting-banding-with-the-z7-silent-shutter), [Verkada](https://help.verkada.com/en/articles/3943201-black-banding-on-camera-feed) |

**Take-away [L]:** take the off-frame within a few ms of the on-frame, or make ambient small so the 100 Hz ripple is irrelevant. The existing `src/trinetra/capture.py` already does dark-frame subtraction; keep it as a correction, not the main defence.

### 2.3 Cross-polarisation

| Fact | Conf. | Source |
|---|---|---|
| Polarizer on the light + polarizer on the lens at 90° removes glare and specular reflections and raises fine-detail contrast. Light loss: tenfold on a reflective part (exposure 1 → 10 ms); about 6-10× more light needed on diffuse parts. | H | [VA Imaging, 18 Jul 2025](https://va-imaging.com/blogs/machine-vision-knowledge-center/polarization-in-your-machine-vision-application) |
| A polarizer acts like about 32% transmission ND (MidOpt PL032). | H | [Novus Light, Warzak 2020](https://www.novuslight.com/selecting-the-best-optical-filters-for-your-machine-vision-application_N11020.html) |
| Polarization of the reflection varies with material and angle, so metal needs tuning. | M | [Syron Optics](https://syronoptics.com/blogs/faqs/how-do-polarizing-filters-improve-image-acquisition-in-machine-vision) |

**Physics note [L]:** near normal incidence, bare metal keeps the linear polarisation of the glint, so crossed polarisers remove it while scatter from scratches and pits (partly depolarised) passes. At oblique incidence metal makes the reflection elliptical, so suppression is partial. Use cross-pol in the contact nose, where light is plentiful.

### 2.4 Narrowband LED + bandpass filter

| Fact | Conf. | Source |
|---|---|---|
| A matching bandpass filter on the lens limits ambient; monochromatic LEDs give repeatable images. | H | [VSD 2019](https://www.vision-systems.com/cameras-accessories/article/16739383/when-to-invest-in-led-strobe-controllers) |
| MidOpt BN880: 855-890 nm useful range, 45 nm FWHM; BP450 suits "higher than normal ambient lighting". | M | [BN880](https://marketing.dev.machinevisiondirect.com/products/filters/product/BN880-67), [BP450](https://www.vision-systems.com/directory/manufacturers/optics/filters/press-release/14196367/midwest-optical-systems-midopt-new-midopt-bp450-indigo-bandpass-filter-and-bn450-narrow-indigo-bandpass-filter) |
| Passbands should be broad enough for LED spectral variation (Gaussian-shaped to match LEDs). | H | [Novus Light 2020](https://www.novuslight.com/selecting-the-best-optical-filters-for-your-machine-vision-application_N11020.html) |

**Take-away [L]:** a 40-50 nm passband passes most of a matched LED but only a slice of white room light (400-700 nm), roughly another 5-8× rejection. It loses colour, so use it on the find camera only if room light is still a problem after §2.1.

### 2.5 Global-shutter sensors

| Fact | Conf. | Source |
|---|---|---|
| RPi Global Shutter Camera: IMX296, 1456 × 1088, 3.45 µm, global shutter. Camera Module 3 and HQ are rolling shutter. | H | [RPi camera docs](https://www.raspberrypi.com/documentation/accessories/camera.html) |
| $50 at launch (9 Mar 2023), C/CS mount, shortest exposure about 30 µs (staff). | H | [RPi GS launch](https://www.raspberrypi.com/news/new-raspberry-pi-global-shutter-camera/) |
| External trigger: pulse XTR (1.8 V); exposure = low-pulse width + 14.26 µs; remove R11 if Q2 is fitted. | H | [RPi camera docs](https://www.raspberrypi.com/documentation/accessories/camera.html) |

**Take-away [L]:** global shutter helps a *moving* camera (no row skew; a short LED pulse lights the whole frame). But 1456 px across means 5 µm/px covers only 7.3 mm. Not needed in a contact nose, where nothing moves during capture.

### 2.6 What handheld tools actually do
Dino-Lite caps [H], GapGun standoff [H], GelSight contact [M] and dermatoscope contact plates [M] all touch the part. A small light-tight cap at the tool tip removes room light **and** hand tremor and fixes focus. That is cheaper and more reliable than any purely optical trick. It is a 15-30 mm cap, not a box.

---

## 3. Resolution vs field of view, at the micrometre level

### 3.1 Sampling rules

| Rule | Conf. | Source |
|---|---|---|
| 2 px across an edge with excellent contrast; 3-4 px across a feature to be robust. Worked example: 0.25 mm hole, 4 px, 40 mm FOV → 640 px. | H | [1stVision, Chouinard 2015](https://www.1stvision.com/machine-vision-solutions/?p=96) |
| 3 px per feature (Edmund); 2-3 px (Imaging Source). | M | [Edmund Optics](https://www.edmundoptics.com/knowledge-center/application-notes/imaging/lens-selection-based-on-resolution/), [Imaging Source](https://www.theimagingsource.com/en-us/resource/glossary/image-resolution-in-machine-vision/) |

**20 µm scratch [L]:** at 3-4 px per feature you need 5-6.7 µm/px. A 20 mm field then needs 3,000-4,000 px across; 50 mm needs 7,500-10,000 px; 100 mm needs 15,000-20,000 px. A 12 MP sensor can *measure* a 20 µm scratch only over about 20 mm.

### 3.2 Field of view at 1-10 µm/px (if the optics deliver that sampling) [L]

| Sensor (px across) | 1 µm/px | 2 µm/px | 5 µm/px | 10 µm/px |
|---|---|---|---|---|
| Pi HQ / AI Camera, IMX477/IMX500 (4056) | 4.1 × 3.0 mm | 8.1 × 6.1 | 20.3 × 15.2 | 40.6 × 30.4 |
| Pi Camera Module 3, IMX708 (4608) | 4.6 × 2.6 | 9.2 × 5.2 | 23.0 × 13.0 | 46.1 × 25.9 |
| Pi Global Shutter, IMX296 (1456) | 1.5 × 1.1 | 2.9 × 2.2 | 7.3 × 5.4 | 14.6 × 10.9 |
| Dino-Lite 1.3 MP (1280) | 1.3 × 1.0 | 2.6 × 2.0 | 6.4 × 5.1 | 12.8 × 10.2 |
| Dino-Lite 5 MP (2592) | 2.6 × 1.9 | 5.2 × 3.9 | 13.0 × 9.7 | 25.9 × 19.4 |

Sensor pixel counts: [RPi docs, H](https://www.raspberrypi.com/documentation/accessories/camera.html); Dino-Lite: [datasheet, H](https://www.tme.com/Document/f744b6938c16708c8bc59c5eb0376a27/AM4515ZT.pdf) and [TEquipment, M](https://www.tequipment.net/Dino-Lite/AM73915MZTL/Digital-Microscopes/).

### 3.3 What realistic optics actually deliver [L]
Assumptions: λ = 0.55 µm; object-side NA ≈ m / (2N(1+m)); Rayleigh = 0.61λ/NA; DOF ≈ λ/NA² + pixel/(m·NA).

| Optics | µm/px | FOV (mm) | Optical resolution | DOF |
|---|---|---|---|---|
| AI Camera, 66° HFOV, at 20 cm min focus | 64 | 260 × 195 | — | large |
| Camera Module 3, 66°, at 10 cm min focus | 28 | 130 × 73 | — | large |
| HQ + 16 mm lens, m = 0.25, f/4 | 6.2 | 25 × 19 | 13 µm | ~1.1 mm |
| HQ + 16 mm, m = 0.5, f/4 | 3.1 | 12.6 × 9.4 | 8 µm | ~0.4 mm |
| HQ + 16 mm, m = 1, f/4 | 1.55 | 6.3 × 4.7 | 5.4 µm | ~0.17 mm |
| HQ + 16 mm, m = 1, f/8 | 1.55 | 6.3 × 4.7 | 10.7 µm | ~0.6 mm |
| HQ + 4x/0.10 infinity objective + 50 mm tube lens (m ≈ 1.1) | 1.4 | 5.7 × 4.2 | 3.4 µm | ~70 µm |
| HQ + 4x/0.10 DIN objective, 160 mm tube (m = 4) | 0.39 | 1.6 × 1.2 | 3.4 µm | ~60 µm |
| HQ + 10x/0.25 infinity objective + 50 mm tube lens (m ≈ 2.8) | 0.56 | 2.3 × 1.7 | 1.3 µm | ~11 µm |
| GS + lens, m = 1, f/4 | 3.45 | 5.0 × 3.8 | 5.4 µm | ~0.2 mm |
| Dino-Lite 1.3 MP at 50x / 200x (datasheet FOV) | 6.1 / 1.6 | 7.8 × 6.3 / 2.0 × 1.6 | not published | 0.1 mm at 220x (H) |
| Reversed phone lens on iPhone 4S (published) | 1.4 | 4.57 × 3.43 | ≤5 µm over ~10 mm² | — |

The last row is published [H]: [Switz et al., PLOS ONE 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4031072/).

Extension needed with the 16 mm lens (thin-lens estimate [L]): about 8 mm for m = 0.5 and 16 mm for m = 1. Working distance is about 48 mm and 32 mm from the lens.

**What this means [L]:**
- Below about 3 µm/px the lens, not the sensor, limits detail. Resolving 1-2 µm needs NA ≥ 0.25, which gives a ~2 mm field and ~10 µm DOF.
- At 1.55 µm/px one touch covers 6.3 × 4.7 mm. A full 100 × 100 mm plate would take about 420 touches (about 100 at m = 0.5). That is why a wide find view must choose where to touch.
- DOF of 0.07-0.4 mm means focus must come from the cap length, not the hand.

### 3.4 Detection is not measurement

| Fact | Conf. | Source |
|---|---|---|
| Reflected dark-field reveals a scratch or flaw "of 8nm level smaller than the resolving power limit" (vendor claim). | M | [Evident dark-field](https://evidentscientific.com/en/learn/microscope/terms/feature14) |
| Scratch grooves scatter like a grating; best camera-to-illumination separation was 8-20° in one study. | M | [Rochester senior design](https://hajim.rochester.edu/optics/undergraduate/senior-design/pdf/spr17/scratch-dig-dddfinal.pdf) |
| Weak scratches sit just above noise; dark-field + adaptive smoothing gave about 82% detection of shallow scratches. | M | [Chinese Optics Letters 2017](https://www.researching.cn/ArticlePdf/m00005/2017/15/8/COL201715081202.pdf) |

**Take-away [L]:** with grazing light, a scratch a few µm wide can light up as a line in a 28 µm/px find view. Its *width and depth* still need the measure nose.

### 3.5 Two-stage "find then measure" in industry

| Example | Conf. | Source |
|---|---|---|
| Wafer fabs: optical inspection finds defects; an SEM review tool re-images them for classification (KLA eDR series). | M | [KLA press release](https://ir.kla.com/news-events/press-releases/detail/364/kla-tencor-completes-its-45nm-defect-portfolio-with-new), [US9293298](https://patents.google.com/patent/US9293298) |
| PCB AOI: suspected faults go to a separate verification station; deep learning can double-check flagged defects in-line. | M | [US5369431](https://patents.google.com/patent/US5369431), [Mycronic](https://www.mycronic.com/de/product-areas/pcb-assembly/webinars-and-events/webinars/fighting-aoi-false-calls/) |
| Waygate: locate, measure and analyse with the same tip; measurement on demand. | H | [Waygate white paper](https://dam.bakerhughes.com/m/54cb0022a30973e7/original/Surface-Indication-Measurem_BHCS38986-1.pdf) |
| GapGun: choose the head by FOV vs resolution (6.5 mm / 3 µm … 90 mm / 40 µm). | H | [Vectro brochure](https://ftr.co.jp/n/eng/common/pdf/Vectro.pdf) |
| **Vision then touch on Boeing panels** (CMU + Siemens + Boeing + GelSight). Vision-only P 0.62 / R 0.79. Tactile or two-stage P = R = 1.0. Per capture: tactile 22.3 s vs visual 6.5 s. Whole panel: **12,978 s tactile-only vs 168 s two-stage**. Scratches ≥10 µm deep, 51 µm wide. | H (from file 02) | [Agrawal et al. 2023](https://ar5iv.labs.arxiv.org/html/2309.04590) |

The last row is the strongest published case for the find + touch-measure split in §7. Vision chooses where to touch, and touch supplies the µm-level truth.

### 3.6 Depth at µm level, handheld

| Method | Handheld depth performance | Conf. | Source |
|---|---|---|---|
| Contact gel + multi-direction light (GelSight Mobile) | Z 1 µm + 4% (1-50 µm), XY 3 µm + 0.2%, ~17 × 14 mm, ~100 ms | M | [NDTEC listing](https://shop.ndtec.net/en-us/gelsight-mobile-system-series-2-surface-analysis-i-5100--10000--30502) |
| Contact gel, robotics grade (GelSight Mini) | Vendor says "mm-level" 3D and "not a metrology device". With a custom gel and careful calibration, one group reached 12.55 µm MAE (SD 8-24 µm) vs a profilometer on 12-96 µm channels. | H | [GelSight Mini](https://www.gelsight.com/gelsightmini/), [arXiv 2509.11385](https://arxiv.org/html/2509.11385v1) (via file 02) |
| **DIY gel cap (this team, 4 weeks)** | About 10-30 µm depth MAE realistic, ≤10 µm a stretch, about 5 µm/px. About 1-1.5 weeks to a first calibrated sensor, with 2-3 gel iterations (24 h cures). | L (file-02 estimate) | `02_gelsight_tactile_3d.md` §4.4, §5 |
| Handheld interferometer (4D InSpec) | Defects 2.5-2,540 µm deep, 4 µm vertical resolution, 8 × 8 mm FOV, about 1 s, "vibration-immune" | M | [New Equipment Digest](https://www.newequipment.com/product-directory/controls-instrumentation/inspection-equipment/product/55104851/68491-surface-gauge) (via file 01) |
| Free-motion handheld laser scanners (Creaform, Scantech, Artec, Shining 3D) | Best quoted accuracy 12-25 µm; about $24-65k; Windows + GPU (e.g., Artec Spider II needs Windows x64 + RTX 4060 or better). Prosumer scanners quote 20-200 µm and want spray on reflective parts. **None reaches ≤10 µm depth.** | H/M | [Creaform BLACK brochure](https://cdn.goengineer.com/handyscan-3d_black_series_brochure.pdf), [Artec Spider II](https://cdn.goengineer.com/1.1_spider_ii_us_letter.pdf), [Revopoint MINI 2](https://3dprintingindustry.com/news/a-closer-look-at-revopoints-new-mini-2-3d-scanner-technical-specifications-and-pricing-227920/) (via file 01) |
| Handheld laser line (GapGun FOV7) | 3 µm resolution over a 6.5 mm profile, with contact standoff | H | [Vectro brochure](https://ftr.co.jp/n/eng/common/pdf/Vectro.pdf) |
| Videoscope phase / stereo (Waygate) | ±0.05 mm best case; 0.125 mm min feature | H | [Waygate white paper](https://dam.bakerhughes.com/m/54cb0022a30973e7/original/Surface-Indication-Measurem_BHCS38986-1.pdf) |
| Focus variation | Needs the optics scanned along the axis; claims µm and sub-µm tolerances; struggles with very smooth or transparent parts; no handheld version described ("PortableRL" named only). Handheld use would need the device braced during the sweep. | H (method) / L (handheld) | [Bruker Alicona](https://www.alicona.com/en/focus-variation/) |
| Non-contact photometric stereo on shiny metal | Glare and mirror images break it (Waygate lists these as error sources for its own 3D); on matte/brushed metal it gives slopes and relative height. | H (problem) / L (conclusion) | [Waygate white paper](https://dam.bakerhughes.com/m/54cb0022a30973e7/original/Surface-Indication-Measurem_BHCS38986-1.pdf) |

---

## 4. The Android option

### 4.1 Llama via ExecuTorch

| Fact | Conf. | Source |
|---|---|---|
| OnePlus 12, ExecuTorch, prompt 64 tokens: Llama 3.2 1B SpinQuant 50.2 tok/s decode, 260.5 tok/s prefill, 1,921 MiB RSS. 3B SpinQuant 19.7 / 89.7 tok/s, 3,726 MiB. 1B BF16 19.2 / 60.3 tok/s, 3,185 MiB. | H | [ExecuTorch PR #6434](https://github.com/pytorch/executorch/pull/6434/files) |
| Quantised Llama 3.2 (24 Oct 2024): 4-bit groupwise weights (group 32), 8-bit dynamic activations; short context up to 8K; on average 2.5× faster decode, 4.2× faster prefill, 56% smaller, 41% less memory. | H | [Meta AI blog](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/) |
| ExecuTorch 0.7: KleidiAI on by default; Galaxy S24+ >350 tok/s prefill, >40 tok/s decode (1B); SDOT path for older phones and the Pi 5 (13 Aug 2025). | H | [PyTorch blog](https://pytorch.org/blog/bringing-generative-ai-to-the-masses-with-executorch-and-kleidiai/) |
| Android demo app at `examples/demo-apps/android/LlamaDemo`; delegates XNNPACK, QNN, MediaTek; Llama 3.2 1B/3B, Llama Guard 3 1B, Qwen3. | H | [ExecuTorch 0.7 docs](https://docs.pytorch.org/executorch/0.7/llm/llama-demo-android.html) |
| Pi tutorial (Pi 4/5, `llama_main`, SpinQuant XNNPACK config, cross-compile on Linux x86_64). Gives no tok/s. Known GLIBC 2.38 vs 2.36 mismatch. | H | [ExecuTorch on Raspberry Pi](https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html) |
| No published ExecuTorch tok/s for 1B on Pi 5. Proxies on Pi 5: PyTorch INT8 2.42, LiteRT 5.08, llama.cpp INT8 6.86 tok/s; Ollama Q4 about 11 tok/s. | M | [ICCVW 2025 supplement](https://openaccess.thecvf.com/content/ICCV2025W/AIM/supplemental/Yoon_LiteRT-Optimized_INT8_LLM_ICCVW_2025_supplemental.pdf), [benchmarklist](https://benchmarklist.com/benchmarks/edge_llm_raspberry_pi_5/) |

### 4.2 Vision models

| Fact | Conf. | Source |
|---|---|---|
| ExecuTorch backends: XNNPACK (all, CPU), Vulkan (Android GPU), Qualcomm and MediaTek (Android NPU), Samsung Exynos, Arm VGF, Arm Ethos-U, Core ML, OpenVINO and more. | H | [Backends overview](https://docs.pytorch.org/executorch/stable/backends-overview.html) |
| Ultralytics ExecuTorch export (v8.3.220, Oct 2025): YOLOv8/11/26; XNNPACK default; **FP32 only**. Pi 5: YOLO26n 142 ms (ExecuTorch) vs 314.8 ms (PyTorch). Android: `org.pytorch:executorch-android`, `Module.load`. | H | [Ultralytics ExecuTorch](https://docs.ultralytics.com/integrations/executorch) |
| Hailo on Pi 5 + AI HAT+ (Hailo-8): YOLO26n 21.7 ms vs 364.6 ms PyTorch CPU (inference only). Compile needs Linux x86_64; Hailo-8/8L use DFC 3.x, Hailo-10H 5.x; ≥1,024 calibration images recommended. | H | [Ultralytics Hailo](https://docs.ultralytics.com/integrations/hailo) |
| EfficientAD: about 2 ms per image on an RTX A6000; I found no phone, Pi or Hailo anomaly-detection benchmarks. | M | [EfficientAD paper](https://arxiv.org/abs/2303.14535v3) |

### 4.3 Attachments and USB-C accessories

| Fact | Conf. | Source |
|---|---|---|
| Android supports external UVC cameras via Camera2 (EXTERNAL level from API 28) only if the OEM enables it. Meant for light uses, not manual ISP/sensor/lens control. Sample config: 640×480@30, 1280×720@15, 1920×1080@10 fps. | H | [AOSP external USB cameras](https://source.android.com/docs/core/camera/external-usb-cameras) |
| DermLite HÜD2: 10x, 12 cross-polarised white LEDs, 25 mm effective focal length, universal phone clamp. DermLite DL1: polarised and non-polarised modes, 10 mm scale. | M | [Harmanis](https://harmanis.com.gr/en/?p=132347), [Macquarie Medical](https://www.macquariemed.com.au/dermlite-dl1-hand-held-dermatoscope-with-phone-adaptor) |
| Moment 10x (25 mm) macro: about 18.5 mm working distance; its 1-inch hood sets the distance. | M | [Unique Photo](https://www.uniquephoto.com/product/momentmacro10xlens), [B&H](https://www.bhphotovideo.com/c/product/1357296-REG/moment_120_002_new_macro_lens_no.html) |
| Reversed phone lens at ~1:1: ≤5 µm over ~10 mm², 4.57 × 3.43 mm, iPhone 4S 1.4 µm pixels, lens module <$6. | H | [Switz 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4031072/) |
| FLIR ONE Pro: phone attachment with an Android USB-C version, own battery (~1 h), app; $349. | H | [FLIR ONE Pro](https://www.flir.com/products/flir-one-pro/) |

**Phone + clip-on 10x [L, assumptions not specs]:** for a main camera of about 6-7 mm focal length at f/1.8 with a 25 mm close-up lens, m ≈ 0.25-0.28 and NA ≈ 0.07. That gives about 4-5 µm/px (at 1.0-1.2 µm pixels), about 5 µm resolution, a field about 3 cm wide, and DOF about 0.1 mm, so a hood or contact plate is again needed.

### 4.4 Pi 5 + Hailo-8 vs phone + USB-C nose for a handheld

| Criterion | Pi 5 8GB + AI HAT+ (Hailo-8) | Android phone + USB-C nose accessory |
|---|---|---|
| Vision compute | Hailo-8 26 TOPS; YOLO26n 21.7 ms [H] | Phone NPU through ExecuTorch QNN/MediaTek (not benchmarked here) or CPU XNNPACK; Ultralytics export is FP32 only [H] |
| Llama 3.2 1B | Pi CPU; no official tok/s; proxies 2.4-11 tok/s [M] | 50.2 tok/s decode, 260.5 prefill on a OnePlus 12 [H] |
| Power | Pi 5 8GB 3.2 W idle, 9.8 W stress [H]; Hailo-8 ~2.5 W typ, 8.65 W max [M]; whole device ~10-15 W [L]. 5V 5A is an optional PD profile; without it USB is limited to 600 mA [H] | Phone battery; accessory carries its own (FLIR ONE ~1 h) [H] |
| Size | Pi + HAT stack + display + battery + 2 cameras: a "brick" handle [L] | Phone-sized + nose [L] |
| Cost | Pi 5 8GB already owned; now $80 → $95 → $125 [H] then +$50 Apr 2026 [H]. AI HAT+ "from $70" [H]; 26 TOPS ₹14,999 at Probots Bengaluru (team's parts doc, 19 Oct) | Phone not priced here; accessory BOM is LEDs, optics, MCU [L] |
| Camera and light control | Picamera2 full manual control on CSI; GPIO LED sequencing; GS external trigger [H] | Manual exposure is device-dependent; external UVC is OEM-optional and not for manual control [H]; accessory needs its own MCU [L] |
| 4-week dev risk | Reuse `src/trinetra` Python (capture with dark frames, photometric stereo, rules, JSON, Llama wrapper). HEF compile and `.pte` export need an x86 Linux host [H] | New Kotlin app + ExecuTorch AAR (LlamaDemo as a base) + accessory firmware + per-phone camera quirks |
| Brief fit | "Pi 5 + AI HAT+" path; HailoRT and ExecuTorch pipelines documented separately | "Arm Android phone" path; no Hailo |

**Verdict:** Pi 5 + AI HAT+ for the 23 Nov PoC. The Android phone + USB-C nose (optics + LEDs + MCU, using the phone's own camera) is a strong product/scale-up path, not a 4-week task.

---

## 5. Public datasets for small surface defects

| Dataset | What it is | Licence | Demo use | Conf. | Source |
|---|---|---|---|---|---|
| MVTec AD | 5,000+ high-res images, 15 categories, pixel-precise masks | CC BY-NC-SA 4.0 | Benchmark anomaly models (metal_nut, screw, grid) | H | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| MVTec AD 2 | 8,000+ images, 8 scenarios; test images under lighting not seen in training; private test split via server | CC BY-NC-SA 4.0 | **Best public test of lighting robustness** | H | [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) |
| MVTec 3D-AD | 4,000+ high-res 3D scans, 10 categories | CC BY-NC-SA 4.0 | 3D roadmap only | H | [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) |
| Real3D-AD (NeurIPS 2023) | 1,254 point clouds, 12 categories (toys, candy, shells), PMAX-S130 scanner, 4 training prototypes per class | CC BY 4.0 | Not useful for an RGB metal demo | H | [Real3D-AD GitHub](https://github.com/M-3LAB/Real3D-AD) |
| Real-IAD D3 (CVPR 2025) | RGB + µm-level point clouds + photometric-stereo pseudo-3D, 20 categories, small fine defects | research use | Closest public match to the photometric-stereo nose | M | [arXiv 2504.14221](https://arxiv.org/abs/2504.14221) |
| NEU surface defect | 1,800 grayscale 200×200, 6 hot-rolled steel classes (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches); NEU-DET has boxes | research, cite | Pre-train the defect-type classifier | M | [Song faculty page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm), [IEEE DataPort](https://ieee-dataport.org/documents/neu-det) |
| KolektorSDD2 | 3,335 colour images (~230×630), 356 defective; train 246/2,085, test 110/894; scratches, minor spots, imperfections | CC BY-NC-SA 4.0 | Small-defect segmentation benchmark | H | [ViCoS](https://www.vicos.si/resources/kolektorsdd2/) |
| Magnetic tile | 1,344 images, 5 defect classes + free, saliency masks | no formal licence, research use | Cracks/blowholes | M | [Dataset Ninja](https://datasetninja.com/magnetic-tile-surface-defect) |
| MPDD | 6 metal-part categories, pixel masks, varied position and rotation; methods drop up to 30% AUROC | CC BY-NC-SA 4.0 | Metal parts with pose variation | M | [anomalib MPDD](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/data/datamodules/image/mpdd.html) |
| VisA | 10,821 images (9,621 normal / 1,200 anomalous), 12 subsets, 4000×6000 sensor | CC BY 4.0 (anomalib says BY-NC-SA) | Small anomalies (PCBs etc.) | M | [spot-diff](https://github.com/amazon-science/spot-diff) |

All are fine for a non-commercial challenge demo. None is handheld, room-light, µm-scale imagery. The demo model **must** also be trained on the team's own captures from both heads.

---

## 6. Lessons from startups, students and shipped tools

1. **Fix the distance physically.** Smartphone crack-width apps mostly fail on distance and per-phone calibration; good results use fixed rigs or markers. [M] [WUJNS 2024](https://wujns.edpsciences.org/articles/wujns/pdf/2024/02/wujns-1007-1202-2024-02-0154-11.pdf), [MDPI Eng 2025](https://www.mdpi.com/2673-4117/6/6/122). Dino-Lite caps and the GapGun standoff solve the same problem [H].
2. **Users won't build a light box**, so the device must bring its own light and the software must cope with the rest. [H] [VSD 2026](https://www.vision-systems.com/cameras-accessories/article/55368033/is-smartphone-based-visual-inspection-the-future-of-quality-control)
3. **Guided workflow + automatic reports sell.** GapGun syncs check plans and auto-reports [H]; Waygate ships Menu Directed Inspection and AI annotation that the operator approves [H].
4. **Measurement claims need Gage R&R** on representative parts (Waygate's own advice) [H].
5. **False calls tire operators**, who then miss real defects at verification. [M] [STERC](https://sterc.org/pdf/31193040.pdf)
6. **CPU-only CNNs on a Pi are slow.** One Pi study found MobileNetV2 too slow for real time while YOLO-tiny worked [M] ([UTeM](https://semarakilmu.com.my/journals/index.php/applied_sciences_eng_tech/article/view/6309)). Hailo cuts YOLO26n from 365 ms to 22 ms [H].
7. **Motion check and retake** (GE patent) is cheap insurance in any multi-frame capture [H].

---

## 7. Recommended architecture: "TriNetra-Handheld" (find + touch-measure), 4-week PoC

### 7.1 Concept
One handheld body (pistol-grip or "torch" style) with two heads looking the same way:
- **FIND head (hover, about 10 cm):** live video, AI detection of candidate defects over about 130 × 73 mm, room light beaten by an LED ring at close range.
- **MEASURE nose (touch):** the operator touches the flagged spot. A 15-30 mm cap sets focus and blocks room light. The same HQ camera and LED ring take one of two caps:
  - **Optical cap** (default): a 7-frame lighting sequence at 1.55-3.1 µm/px gives width, length, area, colour and relative slopes.
  - **Gel cap** (optional): a DIY GelSight-style pad with a reflective skin. The 4-direction LEDs give photometric stereo on the gel and a calibrated height map, at about 4-5 µm/px and about 10-30 µm depth error (file 02).

This is the published "vision then touch" pattern. On Boeing panels, a whole panel took 168 s two-stage vs 12,978 s touch-only, at P = R = 1.0 ([Agrawal et al. 2023](https://ar5iv.labs.arxiv.org/html/2309.04590)).

```
            ┌──────────────────────────── Pi 5 8GB ─────────────────────────────┐
 CM3 ──CSI0─┤ vision_svc (Python): Picamera2 → HailoRT (find YOLO / nose seg)    │
 HQ  ──CSI1─┤    + OpenCV metrology (µm), photometric slopes (trinetra/shape.py) │
 LEDs ─GPIO─┤ rules.py → verdict + severity → JSON v2.x ──┐                      │
 AI HAT+ ─PCIe   (Hailo-8 runs NNs only)                   │                      │
            │ advisor_svc (C++/Python): ExecuTorch llama_main, Llama 3.2 1B      │
            │    reads JSON → recommendation text (never changes verdict)        │
            │ app: HDMI/USB touch UI, buzzer + RGB LED alerts, SQLite + images   │
            └────────────────────────────────────────────────────────────────────┘
```
The Pi 5 has "2 × 4-lane MIPI camera/display transceivers" [H, [Pi 5 page](https://www.raspberrypi.com/products/raspberry-pi-5/)]. Both go to cameras, so the display must be HDMI or USB (as in the team's parts list).

### 7.2 Hardware (genuine, buyable in Bengaluru; re-check stock before paying)

| Block | Part | Why | Price / where |
|---|---|---|---|
| Compute | Raspberry Pi 5 8GB + Active Cooler | owned | — |
| Vision NPU | AI HAT+ 26 TOPS (Hailo-8), or 13 TOPS | vision on HailoRT, as the brief requires | ₹14,999 Probots / 13 TOPS ₹7,539 Robu (team parts doc, 14-19 Oct) |
| FIND camera | Raspberry Pi Camera Module 3 (IMX708, motorised focus, 10 cm min) | 28 µm/px at 10 cm [L]; software focus/exposure | ₹3,249 Robu (parts doc) |
| MEASURE camera | Raspberry Pi HQ Camera (IMX477, 1.55 µm) + official 16 mm C-mount lens | 1.55-3.1 µm/px in the nose [L] | HQ ₹6,493 Robu (was out of stock) or Arducam HQ ₹9,324; lens ₹5,979 Robu (parts doc) |
| Extension | C-mount extension ring(s): 8 mm (m ≈ 0.5) or 16 mm (m ≈ 1). Stacking 5 mm C-CS adapters also works | sets magnification [L] | verify availability |
| Nose lighting | ring of 4-8 small white LEDs at ~30-45° (bright-field, cross-pol film); 4 LEDs at ~10° (dark-field); 4 directional LEDs N/E/S/W at ~30° (photometric slopes); AO3400 MOSFETs from GPIO | §2.3, §3.4 | ~₹500-1,000 [L] |
| Find lighting | 6-8 × 1-3 W white LEDs around the CM3, pulsed per frame; 2 grazing LEDs | ~50-100× room light at 10 cm [L] | team already has 3 W LEDs and strips |
| Cap | 3D-printed matte-black cone, 15-30 mm, felt or foam lip | blocks room light; sets focus | printer |
| Gel cap (optional) | Clear silicone pad on acrylic, with a thin reflective (aluminium-pigment) skin, in a cap matching the optical cap's focus. Calibrate with a small steel ball. Recipe, suppliers and prices are in `02_gelsight_tactile_3d.md` §5 | µm-ish depth (about 10-30 µm error) | silicone ₹2,600 + taxes, plus pigment and acrylic (file 02) |
| Polariser | linear polariser film (light) + lens analyser | glare on metal | verify brand |
| Alerts | buzzer + RGB LED + on-screen banner | "alert operators" | ~₹100 (parts doc) |
| Display | Waveshare 7" HDMI capacitive (parts doc) or smaller HDMI | UI | ₹4,800 (parts doc) |
| Power | 3S 18650 pack + 5 V 5 A buck (session notes) or a 5 V 3 A PD bank with CSI cameras only | Pi limits USB to 600 mA without a 5 A profile [H] | — |
| Calibration | glass stage micrometer (10 µm divisions), USAF-1951 target, feeler gauges | proves the µm claims | verify brand |

*Off-the-shelf alternative measure head:* a Dino-Lite (UVC, caps, polariser, 6.1 µm/px at 50x and 1.6 µm/px at 200x) [H]. Linux LED control and Indian price are unverified; custom dark-field and photometric lighting is limited.

### 7.3 Capture sequences

**FIND (live):**
- CM3 at 2304×1296 for speed (the 4608 mode when it matters), exposure 0.5-1 ms, gain fixed, white balance locked, LED ring on.
- Every Nth frame with LEDs off, to measure the room-light share. If it exceeds ~5%, the UI says "move closer or shade".
- Hailo YOLO gives boxes plus a confidence/anomaly score. The UI marks "touch here".
- 1 ms at 28 µm/px keeps tremor blur below 0.3 px [L].

**MEASURE (touch, about 1 s):**
1. HQ full-res frames: dark, bright-field cross-pol, dark-field, N, E, S, W.
2. Phase-correlate the first and a repeated last bright-field frame. If the shift is more than 1 px, retake (the GE patent idea) [H].
3. Dark-subtract and flat-field.
4. Segment the defect (Hailo YOLO-seg on 640 px tiles at native resolution, or classical thresholding of the dark-field image).
5. Measure length, width and area in µm using the calibration.
6. Get slopes and relative height from N/E/S/W with the existing `shape.py` (photometric stereo + Frankot-Chellappa).
7. Classify the type (scratch/crack = linear, pit/dent = round, stain = colour, burr = edge).

**MEASURE with the gel cap (optional, about 1-2 s):**
1. Press the cap with steady force (a spring stop in the cap helps).
2. Take a dark frame, then one frame each under N, E, S, W (6 directions if the LEDs allow).
3. Photometric stereo on the gel's known reflectance gives surface normals. Poisson / Frankot-Chellappa integration (existing `shape.py`) gives height.
4. Apply the ball-calibration lookup (file 02) and local detrending.
5. Report depth in µm, plus profile plots across the defect.

The gel shows relief only. Closed cracks with no relief, and colour or stain defects, still need the optical cap or the red-dye penetrant (file 02).

### 7.4 Software, as two separately documented pipelines (the brief's requirement)
- **Vision pipeline (HailoRT):** `vision_svc`. Picamera2 capture, LED sequencing, HailoRT inference (find detector, nose segmenter), OpenCV metrology, photometric slopes, `rules.py` verdict and severity. Output: JSON (schema v2.x + `head: find|measure`, `um_per_px`, `width_um`, `length_um`, `rel_depth`, `confidence`).
- **Advisory pipeline (ExecuTorch on CPU):** `advisor_svc`. Llama 3.2 1B Instruct (quantised `.pte`, XNNPACK/KleidiAI), prompt = compact JSON + part's tolerance profile, output = 2-3 line action plus answers to follow-up questions. It never overrides the verdict; a template fallback covers slow runs. Export and cross-compile on an x86 Linux host [H].
- **App:** modes (Find / Measure / Review), alerts, SQLite log (ID, time, operator, crops, JSON, verdict, Llama text), CSV/PDF report, and operator confirm/reject (the Waygate ADR-style approval) to grow the dataset.

### 7.5 Brief bullet → design

| Brief bullet | How the PoC meets it |
|---|---|
| Detect surface defects, cracks, anomalies | Find head: Hailo YOLO + golden-statistics anomaly score; dark-field view for thin scratches and cracks |
| Classify type and severity | Measure nose: µm width/length/area + shape class (+ depth with the gel cap); `rules.py` severity from a tolerance profile (JSON) |
| Alert operators with recommendations | Buzzer, RGB LED and banner within about 1 s of the touch; Llama 3.2 1B text follows |
| Log results | SQLite + images + JSON; CSV/PDF export |
| Real time at the edge, no cloud | Find at ≥15 fps (Hailo YOLO ~22 ms [H]); measure ≤2 s; Llama on device |
| Llama via ExecuTorch | `advisor_svc` on the Pi CPU |
| Hailo vision separate from Llama | `vision_svc` (HailoRT) and `advisor_svc` (ExecuTorch) are separate processes, documented separately |

### 7.6 µm-level target: what is achievable by 23 Nov

| Capability | User target | PoC (4 weeks), honest | Evidence | Beyond the PoC |
|---|---|---|---|---|
| Lateral sampling | µm level | **1.55 µm/px** (m ≈ 1) or 3.1 µm/px (m ≈ 0.5) in the nose | 1.55 µm pixel [H] + optics [L] | 0.4-0.6 µm/px with a 4x-10x objective |
| Lateral resolution | µm level | **≈5-8 µm** at f/4; verify on a USAF target | NA 0.04-0.06 [L] | 1.3 µm (10x/0.25; 2.3 mm field; ~11 µm DOF; focus stacking) |
| Field per touch | — | 6.3 × 4.7 mm (m ≈ 1) or 12.6 × 9.4 mm (m ≈ 0.5) | [L] | interchangeable heads |
| Smallest scratch *detected* | "tiniest" | a few µm wide, with dark-field (detection only; to be shown on samples) | [M]/[L] | — |
| Smallest width *measured* | — | ≈10-15 µm (≥2 resolution elements), length ±~3-5 µm after calibration | [L] | ≈3 µm with a 10x objective |
| Depth, optical cap | µm level | **relative** height/slope maps only; calibrate against known dents/holes and report the measured error | `shape.py`; Waygate glare caveat [H] | — |
| Depth, gel cap (optional) | µm level | **about 10-30 µm MAE** on 20-200 µm-deep scratches, dents and pits; ≤10 µm is a stretch | file 02 estimate [L]; published 12.55 µm MAE with custom gel [H via file 02] | 1 µm + 4% with GelSight Mobile-class hardware [M] (US $30-55k, file 02); 4 µm with a 4D InSpec-class interferometer [M]; 3 µm profile with a GapGun FOV7-class laser line [H] |
| Depth, free-motion 3D scanner | — | not used: none reaches ≤10 µm; best 12-25 µm at $24-65k with Windows + GPU | file 01 [H/M] | — |
| Room light | normal office | nose: near zero ambient (cap); find: ≥50× ambient at 10 cm | §2.1 [L] | strobe + global shutter in find mode |
| Hand motion | handheld | nose: rests on the part; motion check + retake; find: 0.5-1 ms exposures | [H]/[L] | IMU gating |

**If the user still refuses any contact:** a hover-only measure mode (burst + sharpness selection + ≤0.3 ms LED pulses at 2-3 cm) can work at about 5-10 µm/px effective. It cannot give µm depth, and focus will hunt because DOF is about 0.2-0.4 mm [L].

### 7.7 Acceptance tests for the report
1. **Scale:** stage micrometer, scale error <1%.
2. **Resolution:** USAF-1951 smallest resolved element (target ≤8 µm line width).
3. **Room light:** same parts at 0 / 300 / 500 / 1,000 lx (phone lux meter); detection F1 change <5%; dark-frame share in nose images <1%.
4. **Repeatability:** mini Gage R&R (3 operators × 10 defects × 3 trials) on width and length.
5. **Accuracy:** widths and depths of scribed scratches, punched dents and drilled holes vs a lab microscope or dial gauge. With the gel cap: depth MAE on 20-200 µm features, reported honestly even if above 10 µm.
6. **Latency:** find fps; touch-to-alert time; Llama time to first token and tok/s on the Pi.
7. **Power:** runtime on battery; watts from a USB-C meter.
8. **Public benchmark:** detector/anomaly scores on KolektorSDD2 / NEU / MVTec AD 2 for context.

### 7.8 Timeline (21 Oct - 23 Nov)
- **Week 1 (21-27 Oct):**
  - Buy CM3, HQ + 16 mm + extension, LEDs/MOSFETs, polariser, power. Print the handle and nose.
  - Bring up both cameras in Picamera2. Calibrate µm/px.
  - Room-light "unplug" test [H].
  - Export the Llama `.pte` on an x86 host and benchmark it on the Pi.
- **Week 2 (28 Oct-3 Nov):**
  - Make specimens: scribe and sandpaper scratches, punch dents, 0.5-2 mm holes, saw slits.
  - Capture ~500 find frames and ~200 nose sets; label.
  - Train YOLO (find) and YOLO-seg (nose) on a GPU; compile HEFs on x86 (≥1,024 calibration images recommended [H]).
- **Gel cap, parallel track (one person, 28 Oct-10 Nov):** cast 2-3 gels (24 h cures), coat, ball-calibrate, integrate with `shape.py`. Drop it without harm if it misses 10 Nov; the optical cap still meets the brief.
- **Week 3 (4-10 Nov):** integrate the services, UI, rules profiles, Llama prompts, logging, reports, motion retake and ambient warnings.
- **Week 4 (11-17 Nov):** run the §7.7 tests; record the demo video.
- **18-23 Nov:** technical report (two pipelines documented separately), code cleanup.

### 7.9 Main risks and mitigations
| Risk | Mitigation |
|---|---|
| Nose DOF too shallow on curved parts | Use m ≈ 0.5 and f/5.6 (DOF ~0.5 mm); flat plates for the demo |
| Glare on polished aluminium | Cross-pol bright-field + dark-field frames; slopes only qualitative on mirror finishes |
| Llama slow on the Pi | Short JSON prompt, template fallback, alert never waits for Llama |
| Hailo compile toolchain | x86 Docker host in week 1; CPU fallback (ExecuTorch YOLO 142 ms [H]) |
| Rolling-shutter skew in find mode | Short exposure; detection only (no measuring in find mode) |
| User dislikes contact | Explain cap vs box (§2.6). Every handheld µm tool found rests on the part (files 01 and 02). Offer hover mode with the lower spec (§7.6) |
| Gel cap: bubbles, coating, cure time, press force, wear | Treat as optional; 2-3 casts; spring stop for force; spare pads (file 02 §5) |

---

## 8. Scale-up path to industry
1. **Interchangeable heads** (the GapGun VChange model):
   - Find (wide)
   - Measure (1.5 µm/px)
   - Micro (10x objective, 0.5 µm/px, focus stacking)
   - Gel (µm depth)
   - Laser line (profiles)
2. **Global shutter + triggered strobe + IMU** for motion-gated capture in find mode.
3. **Metrology discipline:** per-unit calibration file, traceable targets, Gage R&R per part family (Waygate's advice) [H].
4. **Guided inspection plans and reports:** check plans pushed over Wi-Fi like GapGun [H]; MES/QMS export; operator-approved AI annotations as in Waygate ADR [H], which feed retraining; false-call tracking [M].
5. **Platform options:**
   - An Android phone + USB-C nose accessory (FLIR ONE-style product [H]), running ExecuTorch Llama at 20-50 tok/s on phone CPUs/NPUs [H].
   - Or a CM5-based custom handheld.
6. **Productisation:** IP rating, ESD, eye-safe light classes, secure local storage.

## 9. Gaps and what I could not verify
- No published ExecuTorch tok/s for Llama 3.2 1B on a Pi 5; benchmark it in week 1.
- No ambient-light spec for GapGun; no Dino-Lite Linux LED control; no Indian Dino-Lite price.
- GelSight Mobile figures come from a reseller listing [M]; the Mini sheet was read from a copy, linked from the vendor page.
- The gel-cap depth (10-30 µm) and the 1-1.5-week build time are file 02's estimates [L], not measurements; the two-stage panel timing comes from one robot study (Agrawal et al. 2023).
- No anomaly-detection (PatchCore/EfficientAD) benchmark on Hailo or Android.
- Phone + clip-on numbers in §4.3 rest on assumed phone camera parameters [L].
- The web-search budget ran out midway, so some items (KLA/AOI two-stage, NEU, VisA, MPDD, Real-IAD D3) rest on snippets [M].
