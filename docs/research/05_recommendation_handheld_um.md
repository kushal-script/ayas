# Recommendation: a handheld, room-light, micrometre-level inspection device (21 Oct 2026)

Status: RECOMMENDATION, awaiting the user's go-ahead. Based on `01_handheld_3d_scanners.md`, `02_gelsight_tactile_3d.md` and `03_handheld_inspection_architecture.md` (all COMPLETE); see those files for sources. **(calc)** = my own arithmetic.

## 1. How handheld 3D scanners work, and why they don't give µm

**How they work** (Artec, Creaform, Scantech, Shining 3D, Revopoint, Creality):
- **Light:** they project their own narrowband light (blue LED fringes, NIR VCSEL dots, or crossing blue laser lines) and put a matching **bandpass filter** on the cameras. That is how they work in room light.
- **Motion:** they take **one short exposure per frame** at 7.5–120 fps, so hand motion doesn't matter.
- **Stitching:** they register every frame to the growing model (ICP, texture tracking, IMU), often with reflective **markers** stuck on the part.

**Why they don't give µm:**
- The best metrology laser scanners reach **12–25 µm depth accuracy**. They cost $24–65k and need a Windows PC with a GPU.
- Prosumer scanners (Revopoint, Creality) are **20–200 µm** and need **scanning spray on shiny metal**.
- None has an SDK that runs on a Pi.

## 2. What does give µm handheld

Every handheld tool that reaches µm detail **rests on the part and brings its own light**:

| Tool | Method | Published figures |
|---|---|---|
| Dino-Lite microscopes | Caps that touch the part | — |
| GapGun | Contact standoff, laser line | 3 µm over a 6.5 mm profile |
| GelSight Mobile | Gel + photometric stereo | Z 1 µm + 4 %, 17 × 14 mm; $30–55k |
| 4D InSpec | Interferometry | 4 µm, 8 × 8 mm |

**The physics (calc):**
- Hand tremor is about 50–100 µm RMS.
- At µm magnification the depth of field is only about 0.07–0.4 mm.
- 1.5–5 µm/px covers only a 6–20 mm field with a 12–16 MP sensor.

So µm detail comes from a small field held steady by **touching**. A wide view finds where to touch. Industry works the same way: wafer optical inspection then SEM review, AOI then a verification station, Waygate's "locate then measure". Published vision-then-touch on Boeing panels covered a whole panel in 168 s, against 12,978 s for touch alone, with touch at precision/recall 1.0/1.0.

## 3. Recommended product: "TriNetra-Handheld", find + touch-measure

One handheld "torch" body with two heads facing forward. No box, no gimbal, no arch.

| Head | Hardware | What it does | Spec (honest) |
|---|---|---|---|
| **FIND** (hover at about 10 cm) | Raspberry Pi **Camera Module 3** (IMX708, software autofocus, from 10 cm) + ring of the user's 12 V white strips | Live AI detection of candidate defects on the **Hailo-8**; "touch here" marker | About 28 µm/px over about 130 × 73 mm (calc). The ring is about 50–100× room light at 10 cm. A dark-field (grazing) frame makes µm-wide scratches *visible* (detection only) |
| **MEASURE** nose (touch) | Raspberry Pi **HQ Camera** (IMX477, 1.55 µm pixels) + official 16 mm C-mount lens + extension ring, in a **15–30 mm matte-black cap** with small LEDs: cross-polarised bright-field ring, 4 dark-field (about 10°), 4 directional N/E/S/W (about 30°) | About a 1 s lighting sequence: µm measurement of width/length/area, a relative height map (existing `shape.py`), a motion check with retake | **1.55–3.1 µm/px**, **about 5–8 µm resolution** at f/4, field 6–13 mm. Room light is near zero inside the cap. Focus is fixed by the cap (set once and locked) |
| **GEL tip** (optional snap-on) | Clear soft silicone pad with a 1 µm aluminium-powder skin on the same nose | Turns the relative height into **calibrated depth** on shiny metal (GelSight principle) | DIY **about 10–30 µm depth error**. Go/no-go 3 Nov: a 50 µm shim must read within ±10 µm |

**Compute and software (two pipelines, as the brief requires):**
- **`vision_svc`:** Picamera2 + LED sequencing → **HailoRT** (find YOLO, nose segmentation) → OpenCV metrology in µm + `shape.py` photometric stereo → `rules.py` severity → JSON.
- **`advisor_svc`:** **ExecuTorch** `llama_main`, Llama 3.2 1B on the Pi CPU. It reads the JSON, gives the recommendation and answers follow-ups, and never changes the verdict.
- **App:** Find / Measure / Review modes, buzzer + RGB LED + banner, SQLite + images, CSV/PDF report.
- **Ports:** both Pi 5 MIPI ports are used by the cameras, so the display goes on HDMI. Alternatively, if the Arducam B0471 (USB) is the find camera, one MIPI port is free for an official DSI touch display.

**Reused from the repo:** dark-frame capture, photometric stereo + Frankot–Chellappa (`shape.py`), rules, JSON, the Llama wrapper, SQLite, CLI.

**New work:**
- two-camera handheld capture
- µm/px calibration (stage micrometer)
- width/length metrology
- Hailo models
- gel calibration (ball indenter → lookup table)
- UI

## 4. Honest µm statement for 23 Nov

| Capability | Achievable |
|---|---|
| Lateral sampling | 1.55–3.1 µm/px |
| Lateral resolution | about 5–8 µm (verify on a USAF-1951 target) |
| Smallest scratch *detected* | a few µm wide (dark-field) |
| Smallest width *measured* | about 10–15 µm |
| Depth | relative without the gel; with the DIY gel about 10–30 µm error |

True 1 µm depth needs GelSight-class optics and gel ($30–55k) or a lab instrument: roadmap.

**If touching is refused:** a hover-only measure mode is possible. It takes a burst of frames, keeps the sharpest, and uses LED pulses of ≤0.3 ms at 2–3 cm. Expect about 5–10 µm/px effective with **no µm depth**, and focus will hunt because the depth of field is only about 0.2–0.4 mm (file 03, §7.6).

## 5. Parts (Bengaluru; prices from the team's parts doc and research; recheck stock)

| Part | Price |
|---|---|
| Camera Module 3 | about ₹3,249 |
| HQ Camera | ₹6,493, or Arducam HQ ₹9,324 |
| 16 mm C-mount lens | ₹5,979 |
| C-mount extension rings | unverified |
| Small white LEDs + AO3400 MOSFETs, polariser film | about ₹1–2k |
| Stage micrometer (10 µm divisions) + USAF-1951 target | unverified |
| 3D-printed handle and cap | — |
| HDMI display | ₹4,800 (7" Waveshare, pre-order) |
| Battery (3S 18650 + 5 V 5 A buck + 12 V regulator) | — |
| Gel: clear platinum silicone (Fortisil / Trisil, Pune) + 1 µm aluminium powder | about ₹1,200–1,300/kg |

Already owned: Pi 5 8 GB, AI HAT+ 26 TOPS, 12 V strips. **New spend about ₹20–30k** (calc, rough).

## 6. Timeline (21 Oct – 23 Nov)

| Week | Work |
|---|---|
| Week 1 (21–27 Oct) | Buy parts; print handle and nose; bring up both cameras; µm/px calibration; room-light test; export the Llama `.pte` on x86 and benchmark it on the Pi |
| Week 2 (28 Oct–3 Nov) | Specimens (scribed scratches, punched dents, 0.5–2 mm holes, slits); capture about 500 find frames + about 200 nose sets; label; train YOLO / YOLO-seg on a GPU; compile HEFs on x86; **gel go/no-go 3 Nov** |
| Week 3 (4–10 Nov) | Integrate services, UI, rules, Llama, logging, motion retake, ambient warning |
| Week 4 (11–17 Nov) | Acceptance tests: scale, USAF resolution, 0–1,000 lx room light, mini Gage R&R, accuracy against microscope / dial gauge, latency, power; demo video |
| 18–23 Nov | Technical report (two pipelines documented separately); submit |

## 7. Scale-up path

1. **Interchangeable heads** (GapGun style): Find / Measure / Micro (10× objective, 0.5 µm/px) / Gel (µm depth) / Laser line.
2. **Find mode:** global-shutter camera + triggered strobe + IMU.
3. **Calibration:** per-unit calibration files, traceable targets, Gage R&R per part family.
4. **Workflow:** guided inspection plans, MES export, operator-approved AI labels that feed retraining.
5. **Platform:** an Android phone + USB-C nose accessory (FLIR ONE-style), since ExecuTorch Llama 3.2 1B decodes at 50 tok/s on a OnePlus 12; or a CM5 custom handheld.
