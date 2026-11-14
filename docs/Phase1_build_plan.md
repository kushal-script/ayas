# TriNetra Phase 1: what to build and what to do (5 → 23 Nov 2026)

**Context:**
- Bharat AI SoC Challenge 2026-27, problem statement PS5 (Manufacturing & Industry 4.0).
- Written Thu 5 Nov 2026. Revised the same day after two independent reviews: one on compliance and schedule, one on technical correctness.
- Phase 1 (Open Qualifying) is due **Mon 23 Nov 2026**. The time of day and time zone are not published, so **submit on Sun 22 Nov**.

**Scope.** This plan **supersedes** the 3-camera arch, the moving camera and the handheld-µm proposals for Phase 1; they move to the Phase 2 roadmap (section 15). It uses only what the team owns or has just bought, plus about ₹3–5k of small parts.

**Labels used:**
- **(calc)**: our own arithmetic.
- **(est.)**: an estimate still to be measured.
- **(UNVERIFIED)**: not confirmed.

The sourced research behind every number is in `docs/research/2026-11-05_*.md`.

---

## 0. The solution in one paragraph

**TriNetra** is a bench-top, AI-assisted visual QC station for flat metal parts. The demo part is a 100 × 100 × 3 mm aluminium plate. The **Raspberry Pi AI Camera** looks straight down from 20 cm, inside a black cloth shroud on an open frame.

**Lighting.** Four low-angle blue light bars (N, E, S and W) light the plate one at a time. A white, a blue and a UV light at 40° complete the set. Grazing light makes scratches, dents, pits and cracks glow or cast shadows, and the bar that lights a defect gives its direction and relief. A lights-off frame is subtracted from every frame.

**Detection.** The frames are fused into one **defect-contrast image**. A **custom YOLO detector on the Hailo-10H** (AI HAT+ 2, run through HailoRT) finds and types the defects: scratch, dent, pit, crack, corrosion. A **CPU anomaly map** flags anything the detector doesn't know.

**Verdict.** Every defect is measured in **mm**. A **rules engine** sets its severity and the part's verdict: PASS / CLEAN_AND_REINSPECT / REWORK / HOLD_FOR_QA / REJECT.

**Advice.** The structured result goes to **Llama 3.2 1B-Instruct, running on the Pi's Arm Cortex-A76 through ExecuTorch**. Llama writes the operator's instruction, the likely cause and the prevention, and answers follow-up questions on the **touchscreen**. It never changes the verdict.

**Alerts and logging.** A buzzer and a full-screen colour banner alert the operator. Everything is logged to SQLite with images, and can be exported to CSV.

**Cracks too tight to see** go through a **red-dye penetrant mode** using the Wesaf kit, graded with ASME VIII App. 8 rules. An optional **UV mode** tells removable oil from real corrosion.

---

## 0B. CURRENT (Sat 14 Nov): steel and metal surfaces, lighting-invariant, twin proof of concept

**This section supersedes 0A.** After the organisers' PS5 session on 13 Nov (notes: `docs/research/2026-11-13_ps5_session_notes.md`), the team chose the **steel and metal surfaces** application. The user's hard requirement is that it **works in different lighting conditions**. For Phase 1, the deliverable is a **Blender digital-twin proof of concept**, plus the full pipeline running on the Pi with twin frames as the camera. The organisers allow recorded input.

**Contract:** `docs/twin/DESIGN_STEEL.md`. **Research:** `docs/research/2026-11-14_{lighting_invariance,steel_datasets,steel_severity_standards}.md`.

| Topic | Decision |
|---|---|
| Application | Pre-coating surface inspection of steel and metal plates and sheets (mild steel hot- and cold-rolled, stainless, galvanised, aluminium) in fabrication shops and steel service centres. Decision support |
| Classes (6) | scratch, dent, pit, corrosion, crack, stain (oil/water/fingerprint → clean and re-inspect) |
| Lighting method (redesign) | Fixed 10 ms exposure, fixed gain and WB, raw linear. A **5-frame LED burst** (both / left / ambient / right / both) gives **ambient subtraction** plus **min(L, R) to remove glare**. A **quality gate** refuses captures instead of failing silently. A **clip-on black hood/standoff** handles direct sun. Augmentation and twin lighting randomisation come last. The old single-frame method is kept as the **baseline arm** |
| Lighting conditions tested (twin) | C0 dark · C1 office · C2 warm bulb · C3 window · C4 side lamp + moving shadow · C5 lamp reflected in the plate · C6 direct sun (open and hood) |
| Pass criterion | C0–C5: F1 ≥ F1(C0) − 0.05 and verdict agreement ≥ 95 %. All conditions: 0 % silent failures. C6 with the hood passes |
| Severity | `configs/spec_steel.yaml`, rule-based, informed by EN 10163-2 / ISO 7788, ISO 8501-1 / IS 9954, ISO 8501-3 and IS 2062 (Part 1) cl. 7. Thresholds are illustrative; the camera does not measure depth |
| Hand-off | One JSON record `trinetra.inspection/3.0` (defect, severity, confidence, box, mm, capture quality) → Llama 3.2 1B via ExecuTorch → alert + SQLite log |
| On the Pi | `trinetra-app`: twin bursts or the camera → lightproc → Hailo/ONNX YOLOv8n → rules → record → Llama → web UI on the 5" display. Benchmarks: P/R/F1, FPS, latency, memory, power, TTFT, tok/s |

**Plan to 22 Nov:**

| Date | Work |
|---|---|
| Sat 14 | Build (twin for steel, lighting, lightproc, rules, eval, app) |
| Sun 15 | Pilot, then the overnight render (training set + lighting matrix) |
| Mon 16 | Train the naive and burst arms; run the lighting experiment; redesign if a condition fails |
| Tue 17 | ONNX/HEF; Pi end-to-end + benchmarks |
| Wed 18 | Optional real sanity captures; demo video (Blender overview + matrix) |
| Thu 19 – Fri 20 | Report + README |
| Sat 21 | Buffer |
| **Sun 22** | **Submit** |

---

## 0A. Update (Thu 5 Nov, evening): aircraft-parts framing + digital twin (SUPERSEDED by 0B)

**Decision.** The team adopted the aircraft framing and a digital twin.

**Problem statement.** TriNetra is a **handheld AI visual-inspection assistant for riveted aluminium aerostructure details at Tier-2/3 aerospace suppliers**. The Bengaluru cluster includes HAL, Dynamatic, Aequs and TASL. It is human-in-the-loop decision support: it flags, measures and documents, and a qualified inspector decides.

The research behind every rule and number is in `docs/research/2026-11-05_aero_*.md`.

### What changes

| Area | Before | Now |
|---|---|---|
| Test parts | Plain plates P1–P5 with dents and scratches | **Aircraft-representative riveted coupons** made from the same 5 plates: 4 mm blind rivets at 20 mm pitch, plus fastener faults (missing rivet, oval hole, edge-distance and pitch violations), dents near and far from rivets, scratches, slits from hole edges, galvanic and pitting corrosion, and fatigue-crack strips. Single source of truth: `configs/coupons.yaml`. Make guide and 1:1 drill templates: `docs/twin/coupon_fabrication.md`, `docs/twin/templates/`. **P5A is the held-out test coupon** |
| Detector classes | scratch, dent, pit, crack, corrosion | **rivet, open_hole, dent, scratch, crack, corrosion** (6). Fastener geometry (edge distance ≥ 2D, pitch ≥ 3D, missing fastener, elongated hole) is measured by **rules**, not by YOLO |
| Acceptance rules | `configs/spec.yaml` (generic) | `configs/spec_aero.yaml`: a **DEMO** table built only from public sources (FAA AC 43.13-1B, Piper SL 1095A), with paragraph references |
| Dispositions | PASS … REJECT | Adds **HOLD_FOR_MRB**. Follows Boeing Q219P: a supplier may only rework or scrap; use-as-is and repair are customer-MRB only. An AS9102-style **NCR draft** is generated |
| Training data | Real tiles only | **Digital twin** of the handheld (camera intrinsics, LED pods placed from the Fusion model) and of the coupons, rendered in Blender 5.2 LTS / Cycles on the M3. Used for **synthetic pretraining, then fine-tuning on real tiles**. Contract: `docs/twin/DESIGN.md` |
| Proof | TEST metrics | TEST metrics **plus a 5-row ablation** on real P5: real-only / 25 % real / synth-only / synth→real / mixed, with Wilson 95 % CIs |
| Llama advisor | Generic QC prompt | Aerospace mode: cites the reference ids, never says accept/release/airworthy, and refers out-of-scope items to NDT. Twin-generated defect records **regression-test the prompts** |

### Time gate for the twin

The twin gets **at most about 4 person-days (Fri 6 to Thu 12)**.
- **Fri 13 go/no-go:** synth→real must beat real-only on P5. If it does not, ship the real-only model and report the ablation honestly; a negative result is still a result.
- The twin never delays these: Llama via ExecuTorch, the Hailo stock-HEF proof (Sat 7), the real capture (Mon 9–Tue 10) or the video.

### Words to use and to avoid

- **Use:** "inspector's assistant", "decision support", "aircraft-representative coupons (commercial alloy)", "demo acceptance criteria from public sources", "simulation-pretrained, validated on real coupons".
- **Never say:**
  - airworthy, certified, approves/releases parts, SRM-compliant;
  - replaces NDT or the inspector;
  - "trained in simulation, works directly in the real world".
- **Penetrant:** the red dye is Type II. ASTM E1417-21 §6.9.2 bars it from aerospace final acceptance, so call it a **demonstration of an indication-reading aid**.
- **UV:** the 395 nm UV is outside the 347–382 nm FPI band.

---

## 1. What MUST be done and what CAN be done

### 1.1 Must do (to qualify; from the rules and the PS5 brief)

| # | Requirement | How TriNetra meets it | Proof for the judges |
|---|---|---|---|
| M1 | Llama-family model for the advisory part, **on the device, through ExecuTorch** | Llama 3.2 1B-Instruct (SpinQuant INT4) `.pte`, run by ExecuTorch's `TextLLMRunner` on the Cortex-A76 | Video: Wi-Fi off and Ethernet unplugged, advice streaming. Report: versions, export recipe, tokens/s table |
| M1b | *(our own must)* **Export the `.pte` ourselves** by Tue 10 | A third-party `.pte` is within the rules, but weak on originality and on "use of Arm tools". Section 8.4 | Export YAML, command and log in the repo |
| M2 | If the Hailo is used: vision through **HailoRT on the NPU** and Llama through ExecuTorch on the CPU, **documented as separate pipelines** | Pipeline A (vision) and Pipeline B (Llama) are separate processes that exchange only a JSON defect record | Report section "Two separate pipelines", with a diagram and numbers for each pipeline |
| M3 | Detect surface defects, cracks, anomalies | Multi-light capture, YOLO, CPU anomaly map; penetrant mode for tight cracks | Per-defect metrics on held-out plates |
| M4 | Classify defect **type and severity** | YOLO class plus geometry; severity from mm size, zone and count against `configs/spec.yaml` | Confusion matrix; size error in mm |
| M5 | Alert operators with **actionable recommendations** | Instant colour banner and buzzer; Llama instruction, cause and prevention; Q&A; acknowledge and QA sign-off buttons | Video |
| M6 | Log inspection results | SQLite + JSON + overlay image per part; CSV export; history screen | Video and a sample log |
| M7 | **Real time** at the edge, **no cloud** | All on the Pi 5 + AI HAT+ 2. **Targets (est.):** verdict ≤ 4 s p95 after INSPECT; first Llama token ≤ 6 s after the verdict; full advice ≤ 20 s; ≥ 10 parts/min | Measured on Mon 16; live System screen showing `ip link` with Wi-Fi and Ethernet down |
| M8 | **Deliverables:** source code on **GitHub**, the **application**, a **demo video**, a **technical report**, submitted via the challenge form | Section 13 | The form link comes from the organisers (section 2) |
| M9 | Original work; the team owns IP and licence compliance | LICENSE **AGPL-3.0**, NOTICE ("Built with Llama", Ultralytics, Hailo Model Zoo, the `.pte` source), DATASETS.md; no weights or datasets committed | The repo |

In their PS5 mentoring video (2 Nov) the organisers list what judges look at: **detection quality, real-time performance on the device, usefulness of the advice, operator experience and documentation**. Everything below aims at those five.

### 1.2 Can do (optional, ranked by marks per hour)

| Rank | Optional item | Why it scores | Cost | If it slips |
|---|---|---|---|---|
| 1 | **Custom YOLO on the Hailo-10H** (instead of the CPU) | Real-time performance; use of the HAT | About 2 days; needs Colab | **Go/no-go Fri 13**: ship YOLO on the CPU (NCNN) and say so in the report |
| 2 | **Measured benchmark table plus live on-screen stats** | Performance, documentation | ½ day | Never cut |
| 3 | **Penetrant crack mode** with ASME grading | "Cracks" in the brief; a strong video moment | ½ day code (mostly exists) + chemistry | Show only notch indications |
| 4 | **Unknown anomaly → HOLD_FOR_QA** | "Anomalies" in the brief; honesty about what it can't recognise | Included in the walking skeleton | Never cut |
| 5 | **Shift summary and trend hints by Llama**, e.g. "dents in the same corner in 3 of the last 10 parts → check the clamp" | Usefulness of the advice | ½ day | Template summary |
| 6 | Thread sweep, plus Llama speed idle vs. while vision runs | Optimisation | 2 h | Keep one configuration |
| 7 | **UV contamination check:** oil fluoresces → CLEAN_AND_REINSPECT; corrosion doesn't | Gives the UV LEDs a real job | ½ day | Cut first |
| 8 | Hindi answers ("Explain in Hindi" button) | Operator experience | 1 h | Cut |
| 9 | Ablation: fusion image vs. white-only | Detection quality, documentation | ½ day | Cut |
| 10 | IMX500 on-sensor "part present" gate | The "AI SoC" story | 1–2 days | **Only if the organisers confirm it is allowed**; otherwise Phase 2 |

**Cut order if behind:**
1. UV mode
2. IMX500 and PatchCore
3. Hindi and trend hints
4. Thread sweep
5. Fusion ablation
6. Custom HEF (fall back to YOLO on the CPU)

**Never cut:** the walking skeleton, Llama via ExecuTorch, logging, the TEST evaluation, the "Two separate pipelines" section, and the video.

### 1.3 Do NOT do for Phase 1

These move to section 15:
- multiple cameras, a moving camera, the handheld µm scanner, a laser line;
- photometric-stereo depth in mm on bare aluminium (too shiny to trust);
- **Llama on the Hailo** (hailo-ollama would break the rule that Llama runs through ExecuTorch on the CPU);
- Llama 3B as the default;
- an Android app.

---

## 2. Admin: do it TODAY (Thu 5 Nov)

1. **Team leader emails support@armbharatchallenge.com** (subject mentions PS5). Ask for:
   - the submission form link, and the exact deadline time and time zone;
   - any report template or page limit;
   - the demo-video length and hosting (YouTube unlisted or Drive);
   - whether the repo must be public.
2. **Email tech-queries@armbharatchallenge.com** (subject starting "PS5"). Ask them to confirm that this split is acceptable: vision on the Hailo-10H through HailoRT, and Llama 3.2 1B on the CPU through ExecuTorch. Also ask whether IMX500 on-sensor inference may be added.
3. **Hugging Face:** every member requests access to `meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8` and `meta-llama/Llama-3.2-1B-Instruct`. Approval is manual and can take days.
4. **Hailo Developer Zone** (hailo.ai): register for free. Check that the **Dataflow Compiler 5.1.0** wheel can still be downloaded. If not, post on community.hailo.ai today.
5. **Watch** the challenge page and the commits on `arm-education/Arm-Developer-Labs` every day. The deadline has already moved once (30 → 23 Nov).
6. Whole team: watch the 5-minute PS5 video at https://www.youtube.com/watch?v=AsISV9eDakY. Its reference architecture is camera → vision model → structured defect record → Llama → alert + log, and TriNetra follows it.
7. **Buy the small parts in person on SP Road today or tomorrow** (section 3.4). Deepavali falls on Sun 8 Nov, and **Balipadyami (Tue 10 Nov) is a Karnataka holiday**, so online orders may arrive late. Plan Sun 8 and Tue 10 at half capacity.

---

## 3. The device: physical build

> **Update (5 Nov 2026, later):**
> - **Device:** the team chose a **handheld, F1-steering-wheel-style device** instead of the bench station described below. The Fusion 360 model, print files and assembly guide are in [hardware/enclosure/](../hardware/enclosure/README.md).
> - **Lighting:** front-facing pods beside the camera, each with **3 white 3-LED strips + 3 UV strips of 3 x 5 mm LEDs**. They replace the 4 grazing bars and the 40-degree tier.
> - **Power:** a 33 W USB-PD bank feeds a 12 V trigger, which feeds a 5 V 5 A buck to the Pi.
> - **Display:** the team's Waveshare 5inch DSI LCD (800x480) carries the Pi on its own standoffs; it needs `dtoverlay=vc4-kms-dsi-7inch`.
> - **Software:** the capture sequence becomes "one light group per frame, handheld". There is no fixed jig, so alignment comes from the part outline instead of base fiducials.
> - **Still valid:** the rest of this plan (pipelines, schedule, deliverables).

### 3.1 Layout

```
SIDE VIEW                                          TOP VIEW (camera's view)

          AI Camera (lens 200 mm above plate)                N bar (3 blue)
                 |  [black card around lens]         +-----------------------+
   40° tier  *   |   *  40° tier                     |   * 40° tier (N)      |
  (blue+white    |    (blue+white       W bar        |   +---------------+   | E bar
   +UV)  \       |       /  +UV)       (3 blue)      | * |  plate 100 mm | * | (3 blue)
          \      |      /                            |  o|   in L-stop   |o  |  o = white
 15° bar ==\=====|=====/== 15° bar                   |   +---------------+   |  fiducial
  (40 mm up) \___|___/                               |  o * 40° tier (S) o   |  dots
 ===========[ plate ]=========== matte black base    +-----------------------+
   (whole frame inside a black cloth shroud)                 S bar (3 blue)
```

| Item | Position (from plate centre) | Why |
|---|---|---|
| AI Camera, straight down | Lens front **200 mm** above the plate top | The rated focus range is 20 cm to infinity. That gives **0.064 mm/px on the sensor** and a 259 × 194 mm field, with the plate about 1,567 px wide (calc) |
| **Dark-field bars** DF-N/E/S/W, 3 blue modules each (12 total) | One bar parallel to each plate edge, **150 mm out and 40 mm up**, faces tilted 15° down | Grazing light at 11–22° across the plate (calc). Scratches, pits, crack edges and dent rims light up. Which bar lights a scratch tells its direction. The bars are outside the camera's view (calc) |
| **40° tier**: one post per side, each with 1 blue module, 1 white module and 1 UV cluster (6 LEDs) | **150 mm out and 125 mm up**, tilted 40° down | **Diffuse** 40° light: its mirror point falls off the plate, so there is no glare (calc). Blue-40 makes red dye look dark. White-40 gives true colour and the dye's redness. UV is for the contamination check |
| Base | 450 × 450 mm MDF with matte black EVA foam. An **L-stop** of two black strips in one corner, and **4 white fiducial dots** around the plate area | Repeatable placement to about 0.5 mm. The fiducials give a fixed camera-to-jig mapping, so no plate-edge detection is needed |
| **Shroud (standard in every mode)** | Black cloth or black foam board around the frame, with a front flap. Black (or black-taped) stand, and black card around the lens | Seen from above, shiny aluminium mirrors the ceiling lights and the stand. That glare and room-light changes would swamp the weak dark-field signal |

**Module count:**
- **Blue:** 12 for dark-field + 4 at 40° = 16 used, **4 spare**.
- **White:** the 4 white modules are a new purchase.

**Plate orientation:** if a plate has a visible rolling or brushing grain, draw an arrow on it and always place it with the grain **left to right**.

### 3.2 Electrical

```
12 V 2 A adapter --+-- 1000 µF/25 V --+-- +12 V rail -- (+) of every light bank
                   |                  |
                  GND ----------------+-- Pi GND (pin 14 or 20)   <- common ground is essential

each channel:  Pi GPIO --100 Ω-- gate [logic-level N-MOSFET] drain -- (−) of that bank
                                 gate --10 kΩ-- source -- GND
```

| Channel | Load | GPIO (header pin) |
|---|---|---|
| DF-N | 3 blue modules in parallel (about 0.2–0.3 A, est.) | GPIO17 (pin 11) |
| DF-E | 3 blue | GPIO27 (pin 13) |
| DF-S | 3 blue | GPIO22 (pin 15) |
| DF-W | 3 blue | GPIO23 (pin 16) |
| BLUE-40 | 4 blue | GPIO24 (pin 18) |
| WHITE-40 | 4 white | GPIO25 (pin 22) |
| UV | 12 strings of (2 × UV LED + 270 Ω) from 12 V, 16–21 mA each, about 0.23 A total (calc) | GPIO26 (pin 37) |
| Buzzer | Active-buzzer module **powered from 3.3 V (pin 1)**, or a bare 12 V buzzer on a spare MOSFET channel | GPIO12 (pin 32) |

- **MOSFETs:** prefer **AO3400A** boards, which are fully on at 3.3 V. D4184 (non-opto) and IRLZ44N modules are not rated at 3.3 V gate drive, but are fine below 0.5 A.
- **Buzzer:** common "5 V buzzer modules" use a PNP transistor. Powered from 5 V, they **never switch off** from a 3.3 V GPIO, which is why the table powers the module from 3.3 V.
- **Pins to avoid:** GPIO0–8 (HAT ID EEPROM, I²C, SPI). The Touch Display 2 power lead uses pins 2 and 6, so take ground from pin 14 or 20. If the touchscreen is an **SPI** panel, it uses GPIO17 and GPIO24/25, so remap.
- **Software rule:** at most **2 channels on at once**. All channels together draw about 2.2 A (calc), which is over the adapter's rating. The "all bars on" image is the sum of the 4 dark-field frames.
- **No PWM dimming:** it would put bands in the image through the rolling shutter.
- **Day 1:** measure the current of one blue module at 12.0 V. Expect 60–100 mA; listings say 0.72–1.2 W.
- **Pi power:** official **27 W (5 V/5 A) PSU**, **Active Cooler**, AI HAT+ 2 on its 16 mm stacking header with its heatsink.

### 3.3 Safety

- **Aerosols:** the penetrant and developer are flammable aerosols. Spray **outdoors or at an open window, at least 3 m from the Pi, the PSUs and the LEDs**. Let parts flash off for at least 5 min before they go under the camera. Wear nitrile gloves and safety glasses. Store the cans below 30 °C.
- **UV (395 nm) and bright blue LEDs:** never look into them. Wear **UV400 glasses** when setting up UV. Lights are on only during capture.
- **Corrosion step:** never mix bleach with vinegar.

### 3.4 Bill of materials

**Owned:**
- Pi 5 8 GB, AI HAT+ 2 (Hailo-10H), AI Camera (IMX500), touchscreen
- 20 blue 12 V 3-LED modules, 50 UV 5 mm LEDs
- auto centre punch, 5 aluminium plates, Wesaf DPT kit, nitrile gloves

**Check you have:**
- the 27 W PSU, the Active Cooler, a 64 GB A2 microSD and a USB drive for backups
- vise, hammer, drill (1/2/3 mm), hacksaw, needle file, scriber, 80/220 sandpaper, caliper

**Buy** (genuine brands; SP Road in person, or Robu / Robocraze / Probots; prices approximate):

| Item | Approx. ₹ |
|---|---|
| 12 V 2 A adapter + DC jack-to-screw-terminal | 470–550 |
| 8-channel AO3400A board, or 8 × D4184 (non-opto) modules | 320–400 |
| Resistors (100 Ω ×10, 10 kΩ ×10, 270 Ω ×15) + 1000 µF/25 V capacitor | 100–150 |
| **4 × white 12 V 3-LED modules** (same type as the blue ones) | 100–200 |
| Active-buzzer module (3.3 V-capable) | 30–60 |
| 2 × black 2020 V-slot 500 mm + brackets/T-nuts (or wood) for the camera stand | 700–950 |
| Pi 5 camera FPC 500 mm (if the Pi sits beside the rig) | 370 |
| MDF + matte black EVA foam + black foam board/cloth for base and shroud | 500–700 |
| UV400 safety glasses | 500–1,500 |
| Hookup wire, heat-shrink, perfboard (UV clusters), white dot stickers (fiducials) | 150–250 |
| **Total** | **about ₹3,200–5,100** (calc) |

**Why white as well as blue:**
- Under blue light, red dye, a dark stain and a shadow all look dark. White separates them by colour.
- White gives the natural-colour picture for the operator.
- Under blue light a Bayer sensor samples mainly a quarter of its pixel sites.

---

## 4. Test specimens

> **Superseded (5 Nov evening) by the riveted aircraft-representative coupons (section 0A):**
> - definitions in `configs/coupons.yaml`;
> - step-by-step make guide in `docs/twin/coupon_fabrication.md`;
> - 1:1 drill and marking templates in `docs/twin/templates/`.
>
> The principles below still hold:
> - split by plate;
> - P1 is never damaged and gives the golden reference;
> - corrosion starts first;
> - no penetrant on the test coupon before the video;
> - ground truth is measured in plate mm.

### 4.1 Plate plan (split by plate, so no physical defect appears in both train and test)

| Plate | Use | What goes on it |
|---|---|---|
| **P1** (faces A and B) | TRAIN, good | Nothing. Gloves only. **The golden reference comes from P1 only.** No penetrant, ever |
| **P2** (A/B) | TEST, good | Nothing permanent. Face B later gets **removable** contamination for the test (fingerprint, oil drop, marker dot, water spot), cleaned off afterwards. No penetrant |
| **P3** | TRAIN | **Face A, dents:** a 3 × 3 grid of centre-punch marks (3 impact settings × 1/2/3 hits), 3 heavy punch-plus-hammer dents, 3 blind holes (1, 2 and 3 mm drill, 0.5–1 mm deep). **Face B, scratches:** 6 scriber scratches (light/medium/heavy × 1 and 5 passes, 30–60 mm long, at different angles), 1 needle-file scratch, 80- and 220-grit abrasion patches (10 × 10 mm) |
| **P4** | TRAIN | **First cut off two 100 × 20 mm strips, S1 and S2** (P4 becomes 100 × 60 mm). **Face A and edges:** 3 hacksaw edge slits (3, 6 and 10 mm deep), 2 file V-notches, 1 edge chip, one burred edge. **Face B:** 3 corrosion spots |
| **P5** | TEST | A smaller copy of every type: 4 dents, 1 hole, 3 scratches, 1 abrasion patch, 2 slits, 1 notch, 1 burr, 1 corrosion spot. **No penetrant before the video is shot** |
| **S1** | Real crack, for the penetrant demo | File a 1 mm V-notch at mid-length. Clamp just above the vise jaws and **reverse-bend ±15–20° by hand** until a 2–5 mm hairline runs from the notch root, then unload. Call it a "notch-initiated fatigue crack" |
| **S2** | Punch calibration | The same 9 punch marks as on P3. Saw across them, sand the cut face, and take a macro photo beside a ruler to get the **true depth for each punch setting** |

**Label for evaluation.** Give `ground_truth.csv` an `eval_class` column:
- dent, hole → `dent` / `pit`
- scratch, abrasion → `scratch`
- slit, notch, fatigue crack → `crack`
- corrosion → `corrosion`
- edge chip → `dent`
- **burr → `unknown`**, expected HOLD_FOR_QA

**Contamination on P2-B is expected to come out `unknown` → HOLD_FOR_QA**, or `contamination` → CLEAN_AND_REINSPECT if UV mode is used. It must **not** be labelled as corrosion.

### 4.2 How each defect is made

- **Alloy check (Thu 5):** fold one end of a strip flat in the vise.
  - If it cracks: hard alloy. An over-bend crack is possible too.
  - If it folds clean: use only the S1 fatigue method.
- **Corrosion (start Thu 5):** soak a tissue patch in 5 % salt in white vinegar and keep it wet in a sealed box. Mask the rest of the face with tape. It needs 24–72 h. Accept white, pitted spots. **Measure the corrosion ground truth on Mon 9.**
- **Size:** make defects **at least about 0.3 mm wide**. That is about 3.75 px on the 0.08 mm/px analysis grid (calc). Finer scratches still glint but can't be measured.
- **Ground-truth table** (`data/real/ground_truth.csv`), one row per defect:
  - columns: ID, plate/face, centre x,y in mm from the L-stop corner, type, eval_class, tool and setting, length, width, depth, method, uncertainty;
  - length: caliper;
  - width: phone macro photo with a 0.5 mm rule;
  - dent depth: from the S2 section; hole depth: from the drill stop;
  - scratch depth: categorical only (light/medium/heavy).

### 4.3 Order of operations (this matters)

1. **Thu 5:** cut S1/S2, make the mechanical defects, start corrosion.
2. Measure the ground truth (mechanical defects Fri 6, corrosion Mon 9).
3. Pilot capture (Mon 9), then the **whole surface dataset** (Tue 10), including **video-quality surface shots of P5**.
4. Contamination session on P2-B, then clean it.
5. Penetrant: **S1 on Thu 12, P4 on Sat 14, P5 only after the video.** Do a fresh S1 run on the morning of the video shoot (Thu 19), because the indications fade.

### 4.4 Two bench tests (Wed 11, 10 min each)

**Does Wesaf Penetrant-97 fluoresce under 395 nm?**
1. In a dark room, put five spots on bare aluminium and black card:
   - penetrant;
   - penetrant under developer;
   - developer only;
   - a yellow highlighter streak (positive control);
   - clean metal (negative control).
2. Capture UV on and UV off at a fixed exposure, and compare the R channel.
3. Read the result. Visible red dyes are normally not fluorescent, so don't count on it.
4. Don't use printer paper: its brighteners glow.

**Blue vs. white contrast** of a developed slit: this confirms the "dye looks dark under blue" trick and sets the thresholds.

---

## 5. Inspection modes (what the operator does)

| Mode | Operator steps | Capture | Output |
|---|---|---|---|
| **SURFACE** (default) | Put the plate in the L-stop, close the flap, tap **INSPECT** | dark → DF-N → DF-E → DF-S → DF-W → BLUE-40 → WHITE-40 → dark. About 1.5–2.5 s (est.), one exposure for every state | Boxes, types, mm sizes, severity, verdict, banner + buzzer, Llama advice, Acknowledge / QA sign-off |
| **PENETRANT** | Dye process off the rig (section 7.8). At 10 min of development, put the part in and tap **PENETRANT** | dark → WHITE-40 → BLUE-40 → dark (own exposure; developer is bright) | Indications, linear/rounded, ASME App. 8 verdict, Llama explanation |
| **UV** (optional) | Tap **UV CHECK** on a suspicious stain | dark → UV → dark | Fluoresces → `contamination` → CLEAN_AND_REINSPECT. Doesn't fluoresce → left to the surface result |

---

## 6. Software architecture: two separate pipelines

```
            PIPELINE A: VISION (HailoRT on Hailo-10H)               PIPELINE B: ADVISOR (ExecuTorch on Cortex-A76)
 process: trinetra-vision (system Python 3.13 + apt picamera2,      process: trinetra-advisor (venv: executorch 1.5.1,
          python3-h10-hailort, OpenCV)                                       torch 2.14.0 CPU build)

 GPIO lights + AI Camera (Picamera2, SRGGB10 raw, AE/AWB off)
   -> dark subtraction, fixed jig homography (fiducials) + ECC, part mask   [CPU]
   -> fusion image (dark-field sum | relief | 40° diffuse)                   [CPU]
   -> 9 tiles of 640 px -> YOLOv8n .hef  ------------------>  [Hailo-10H NPU via HailoRT]
   -> tile NMS + seam merge + anomaly map + mm measurement      [CPU]
   -> rules engine (spec.yaml) -> severity + verdict            [CPU, deterministic]
   -> defect record JSON  ==== localhost HTTP ====>  Llama 3.2 1B-Instruct SpinQuant .pte
   -> banner + buzzer (instant)                               TextLLMRunner (XNNPACK + KleidiAI, 4 threads)
   -> SQLite + JSON + images                                  -> instruction / cause / prevention, Q&A,
   <=========== streamed tokens to the touchscreen ==========    shift summary (streamed)

 UI: FastAPI in trinetra-vision -> Chromium kiosk on the touchscreen
```

**Why separate processes:**
- the brief asks for separately documented pipelines;
- the apt Picamera2/HailoRT Python and the ExecuTorch/torch venv can't conflict over numpy;
- the alert never waits for the LLM.

**Order:** vision runs first, then Llama, so they never compete for the 4 cores.

**Developing without the Pi.** The macOS arm64 ExecuTorch wheel also includes `TextLLMRunner`, so the advisor can be built and tested on the Mac. The vision code runs on the Mac with the simulator.

### 6.1 The JSON defect record (schema v2.0 → v2.1)

```json
{
  "part_id": "P5-A-0007", "mode": "surface", "timestamp": "2026-11-16T14:02:11",
  "verdict": "REJECT", "inspect_ms": 2140,
  "defects": [
    {"id": "D1", "type": "dent", "confidence": 0.91, "size": {"diameter_mm": 3.4},
     "location_mm": {"x": 72.1, "y": 18.4}, "zone": "general", "severity": "major",
     "disposition": "REJECT", "rule": "dent 3.4 mm > major limit 3.0 mm (general)",
     "sop_action": "Do not hammer it flat on the line. Set it aside for QA.",
     "likely_causes": ["plate dropped or struck by a tool", "chip trapped under the plate in a clamp"]},
    {"id": "D2", "type": "unknown", "confidence": 0.0, "anomaly_score": 7.8,
     "location_mm": {"x": 30.2, "y": 64.0}, "size": {"area_mm2": 1.9}, "disposition": "HOLD_FOR_QA"}
  ],
  "history": {"last_n": 20, "same_type_count": 3, "same_location_count": 2},
  "backend": {"vision": "hailo10h:yolov8n_trinetra.hef", "advisor": "executorch-1.5.1:Llama-3.2-1B-Instruct-SpinQuant-TriNetra.pte"}
}
```

---

## 7. Pipeline A (vision) in detail

### 7.1 Capture (gets the dataset right the first time)

**Configuration:**
- `create_still_configuration(main={'size': (2028, 1520), 'format': 'RGB888'}, raw={'format': 'SRGGB10', 'size': (4056, 3040)}, buffer_count=2, queue=False, controls={...})`.
- **Ask for the raw format explicitly.** Otherwise the Pi 5 returns a compressed raw stream that can't be used for linear maths.
- Read it with `req.make_array('raw').view(np.uint16)`.
- Confirm the Bayer order with `picam2.camera_configuration()['raw']`.

**Controls:**
- `AeEnable False`, `AwbEnable False`, fixed `ColourGains`, `AnalogueGain 1.0`.
- `ExposureTime` of **10, 20 or 30 ms only**. Whole multiples of 10 ms cancel the 100 Hz flicker of 50 Hz mains.
- **One exposure for every state in a mode.** Ambient light scales with exposure, so one dark pair can't serve several exposures.

**Per state:**
1. Switch the GPIO.
2. `req = picam2.capture_request(flush=True)`.
3. If the exposure was changed, keep capturing until `abs(md['ExposureTime'] - target) < 100`. A new exposure takes a few frames to apply, and `flush` only guarantees the frame started after the call.
4. Keep the frame, `req.release()`.

**Dark frames and sanity checks:**
- Take dark frames before and after the sequence and subtract their mean in linear raw.
- **Reject the sequence** if the dark mean or the difference between the two dark frames is over a threshold (room light changed, or the shroud is open).

**What to save:**
- For every sequence, save the **plate-region raw Bayer** of each state as uint16 (`np.savez_compressed`, about 5 GB for the whole dataset, calc), plus the homography and metadata.
- Never save only the fused image: then the processing can be redone if the specimens change.
- **Copy the TEST data to a USB drive the same day.**

**Day-1 checks:**
- Focus at 200 mm with the lens tool on a fine-print target, then fix it with a dot of nail varnish.
- Try 150 mm only if the lens turns that far without coming loose.

### 7.2 Pre-processing (CPU, OpenCV)

- **Geometry:**
  - Find the 4 fiducial dots once, giving a fixed homography to a **0.08 mm/px grid** covering the 110 × 110 mm jig area (1,375 × 1,375 px).
  - Refine each capture with ECC (existing `align.estimate_motion`).
  - The same grid serves the 100 × 60 P4 and the strips, with a **part mask** from the WHITE-40 frame (aluminium is brighter than matte black EVA).
  - Check Otsu on WHITE-40 on Day 1. The 40° lights are diffuse on a mirror-like plate, so the plate is not "bright" under them.
- **Flat-field:** divide each light's image by a smoothed P1 image under that light, with the divisor floored at its median × 0.2. This cancels the 2–4× fall-off across the plate.

### 7.3 The fusion image

The fusion image has three channels. Scale each with **fixed constants taken from TRAIN**, never with per-image min–max.

| Channel | Formula | What it shows |
|---|---|---|
| **R = dark-field sum** | S = DF-N + DF-E + DF-S + DF-W | Scatter from scratches, pits, crack edges and dent rims |
| **G = relief** | clip(k · (\|E−W\| + \|N−S\|) / (S + ε)), with ε = median of S on P1 | A dent shows a bright and dark crescent pair; a scratch lights only one bar. ε stops division by almost zero on flat bare metal |
| **B = 40° diffuse (WHITE-40 luminance)** | — | Colour and albedo changes: stains, corrosion |

Also keep the **signed** maps (E−W)/(S+ε) and (N−S)/(S+ε), which tell dent from bump, and the **anisotropy and direction** (max/min over the 4 bars), which gives scratch orientation.

### 7.4 Detection

1. **YOLOv8n on the Hailo-10H.**
   - **3 × 3 tiles of 640 px at x, y = 0, 368, 735** (calc; covers 1,375 px), sent as one batch with `picamera2.devices.Hailo(hef, batch_size=9)`.
   - NMS across tiles, then **merge same-class boxes that touch across a tile seam.** A 30–60 mm scratch is longer than a tile.
   - Classes (updated 5 Nov, section 0A): `rivet, open_hole, dent, scratch, crack, corrosion`. Fastener geometry checks run in `aero.py` on the rivet/hole detections.
   - Trained on tiles at the **same scale** as inspection. Shrinking a 12 MP frame to 640 px would lose the defects (calc).
2. **Anomaly map (CPU).**
   - Robust z-score per map against the **P1-only golden** (mean over 16 P1 captures; variance box-filtered over 31 × 31 px).
   - Plus a self-reference: the map minus a 51 px median blur, divided by the plate MAD.
   - Threshold inside the part mask to get connected regions.
3. **Fusion of the two.**
   - **Length, width, diameter and area come from the connected anomaly region**, not from the box.
   - Regions with no YOLO box above 0.3 → **`unknown` → HOLD_FOR_QA**.
   - Boxes with no anomaly support get lower confidence; below `min_confidence` → HOLD_FOR_QA.
4. **Walking-skeleton typing**, used until YOLO exists (Wed 11):
   - elongation > 4 → scratch, or crack if it touches an edge or is very dark;
   - round with a relief dipole → dent;
   - small and round → pit;
   - strong B-channel change → corrosion.

### 7.5 Severity and verdict

`rules.py` + `configs/spec.yaml` (existing, deterministic):
- per-type mm limits per zone; `reworkable` → REWORK, otherwise REJECT; `unknown` → HOLD_FOR_QA; the part takes the worst verdict.

**Add to `spec.yaml`:**
- `corrosion: {measure: area_mm2, limits: {minor: 2, major: 5, critical: 20}, reworkable: false, …}`
- `contamination: {disposition: CLEAN_AND_REINSPECT, …}`, used by UV mode.

`rules[kind]` currently raises KeyError for a missing class. **Llama never changes the verdict.**

### 7.6 Dataset capture and labels

**Pilot gate (Mon 9).**
1. Capture 3 sequences of P3.
2. Run the fusion code on them.
3. Check that the dents, scratches and holes stand out.
4. Only then capture the full set.

**Full capture (Tue 10).** Each face × **2 orientations (0°/180°, grain left-right) × 8 re-seated placements = 16 sequences per face**.

| Split | Faces | Sequences |
|---|---|---|
| TRAIN | P1 A/B, P3 A/B, P4 A/B | 96 |
| TEST | P2 A/B, P5 A/B | 64 |

**Labels, without drawing every image:**
1. Draw each face's defects **once** in plate-mm (the ground-truth CSV, plus one Label Studio pass per face).
2. A script projects them into every capture and writes the YOLO `.txt` files. The grid is in mm and 180° is a fixed flip.
3. Review in Label Studio (about 2–3 h, versus about 12 person-hours of hand drawing, calc).

**Tiles (calc):**
- 96 × 9 ≈ 860 TRAIN tiles.
- **Subsample P1 tiles to about 10 % background.** Otherwise a third of all tiles are defect-free.
- Hold out one placement per face as TRAIN-val.

**Training** (Mac `device=mps` or Colab):

```
yolo detect train model=yolov8n.pt data=plates.yaml imgsz=640 epochs=150 patience=50 hsv_h=0 hsv_s=0 hsv_v=0.2 scale=0.2 fliplr=0.5 flipud=0.5
```

- No hue/saturation augmentation: the fusion channels have fixed meanings.
- YOLOv8n holds INT8 accuracy best. **Not YOLO26**, which has no Hailo 5.1.0 support.

**Export:** `yolo export model=best.pt format=onnx imgsz=640 opset=11`.

### 7.7 Compiling the HEF (the Mac can't do it, so use Colab)

**The version trap.** The Pi's apt ships **HailoRT 5.1.1** while Hailo upstream is at 5.5.0. Compile **only with DFC 5.1.0 + Model Zoo v5.1.0**.

**Colab steps:**
1. Use a **T4 GPU**, and set **Runtime → Change runtime type → Runtime version "2026.07"** (Python 3.12). The current default runtime has Python 3.13, which DFC 5.1.0 rejects. Check with `!python --version`.
2. Install the DFC 5.1.0 wheel, then:

   ```
   apt-get install -y graphviz graphviz-dev
   git clone -b v5.1.0 https://github.com/hailo-ai/hailo_model_zoo.git
   pip install -e hailo_model_zoo
   ```

3. If you use Hailo's `hailo-apps` `compilation.ipynb`, fix two things: it clones master, and it has a `dfc_whl = 'dfc'` bug.
4. **Prove the chain by Sat 7 (go/no-go 1).** Compile the **stock COCO yolov8n**, then on the Pi run `hailortcli parse-hef`, `hailortcli run` and `detect.py -m <hef>`. Write `npu.py` against this stock HEF straight away.
5. **Custom model (Thu 12):**
   - Calibration set: **1,024 random 640 px crops from the TRAIN grids** (raw 0–255, not normalised; never TEST).
   - Copy `yolov8n.yaml` and `yolov8n.alls` **inside `hailo_model_zoo/cfg/…`**, because their paths are relative. Point `alls_script` at the copy and add `model_optimization_config(calibration, calibset_size=1024)`.
   - Compile:

     ```
     hailomz compile --ckpt best.onnx --calib-path calib/ --yaml <copy>.yaml --classes 6 --hw-arch hailo10h
     ```

   - Leave out `--performance` on the first pass. Check that SNR is at least 10 dB. Save the HAR and HEF to Drive.
6. **Go/no-go 2 (Fri 13).** The custom HEF must pass `hailortcli run` on the Pi, and its mAP on TRAIN-val must be within 5 points of ONNX FP32.
   - **If not:** ship YOLO on the CPU (NCNN export; about 0.6–0.9 s for 9 tiles, est.), stop Hailo work, and document it honestly.
   - An x86 Ubuntu 22.04 GPU VM is the only other compile option worth a few hours.

### 7.8 Penetrant mode

**Process** (Wesaf kit; read the can labels first; no datasheet is published online, so these are the generic Method C times):
1. Wipe with Cleaner-96 and let it dry for 5 min.
2. Penetrant-97: **dwell for 10 min**, keeping the surface wet.
3. Wipe with a dry lint-free cloth, then with a cloth *damp* with cleaner. Never spray cleaner directly on the part.
4. Let it dry for 2–5 min.
5. Shake the developer, then apply one **thin** coat from 20–30 cm.
6. **Read at 10 min** (allowed range 10–60). For the video time-lapse, also capture at 2, 5, 10, 20 and 30 min.
7. Post-clean.

**On the rig:**
- **An indication must be dark under BLUE-40 *and* red under WHITE-40.** Use the existing `penetrant.py` redness, (r − (g+b)/2) / brightness, which is ~0 on white developer. Do not use R/(G+B), which is about 0.5 on clean developer.
- **Linear:** length > 3 × width. **Relevant:** larger than 1.6 mm.
- **REJECT** if:
  - there is any relevant linear indication;
  - a rounded indication is larger than 4.8 mm;
  - 4 or more rounded indications lie in a line, ≤ 1.6 mm apart (ASME VIII Div. 1 App. 8).

---

## 8. Pipeline B (Llama via ExecuTorch) in detail

### 8.1 Day-1 path (no export, no C++ build): Mac and Pi

```bash
python3 -m venv ~/et15 && source ~/et15/bin/activate      # Pi: Trixie Python 3.13; Mac: python3.12 is fine
pip install -U pip
pip install executorch==1.5.1 torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install -U huggingface_hub && hf auth login
hf download software-mansion/Llama-3.2-ExecuTorch 1b/xnnpack/llama_3_2_1b_xnnpack_spinquant.pte tokenizer.json --local-dir ~/models/llama32
```

- **Pin `torch==2.14.0` explicitly.** executorch 1.5.1 doesn't declare it, and the runner imports it.
- **Always use the CPU index.** Plain PyPI torch pulls CUDA packages on Linux.
- About the prebuilt `.pte`:
  - It is ungated, about 1 GiB, and built for ExecuTorch 1.4.1. Loading it on 1.5.1 is the best-supported case, but not guaranteed (it uses custom ops). **Test on Fri 6.**
  - **If it fails:** use **executorch 1.4.1** with the torch version in v1.4.1's `torch_pin.py`. The old `executorch-community` `.pte` predates 1.0 and is *less* likely to load.
  - Record the HF revision and the SHA-256 in `models/README.md`, and credit software-mansion in NOTICE.

### 8.2 The advisor service

New file `src/trinetra/advisor_svc.py`, run in the `et15` venv:

```python
from executorch.extension.llm.custom_ops import custom_ops      # registers custom ops
from executorch.kernels import quantized
from executorch.extension.llm.runner import TextLLMRunner, GenerationConfig

runner = TextLLMRunner(PTE_PATH, TOKENIZER_PATH)

def turn(role, text):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"

def prompt(system, user):
    return ("<|begin_of_text|>" + turn("system", system) + turn("user", user) +
            "<|start_header_id|>assistant<|end_header_id|>\n\n")

cfg = GenerationConfig(max_new_tokens=160, temperature=0.0, echo=False)   # Hindi: 300
runner.reset()                                  # once per NEW part (KV position carries over)
runner.generate(prompt(SYSTEM, record_json), cfg,
                token_callback=send_to_ui, stats_callback=log_stats)
```

- Load the model once at boot and run one warm-up generation.
- Serve `POST /advise`, `/ask` and `/summary` on `127.0.0.1`, streaming tokens.
- Neither `TextLLMRunner` nor `llama_main` applies the chat template. The app builds it, with **one BOS only**.
- **Follow-up questions** on the same part:
  - First try sending only the new user turn and the assistant header, without a reset (the KV cache keeps the context).
  - If that misbehaves, reset and re-send the whole conversation, kept to about **700 tokens**.
  - The software-mansion `.pte` treats its context as a sliding window, so **reset on every new part**.
- **Fixes to the existing `llm.py`:**
  - replace the blocking `llama_main` subprocess with this service;
  - fix the BOS assumption (`llm.py:51`);
  - add a token budget;
  - log which backend answered.

### 8.3 What Llama is asked to do

- **System prompt:** "You are a QC assistant on a factory floor. The verdict and severity are decided by the inspection rules and must not be changed. Using only the JSON record and SOP text given, write: ACTION, WHY (which rule), LIKELY CAUSE, PREVENT. Max 80 words. Plain words."
- **Input:** the minified record (section 6.1), including the SOP action and likely causes from `spec.yaml` and the history counts from SQLite. Llama reasons over facts instead of inventing them.
- **Buttons:** "Why this verdict?", "How do I rework it?", "What caused it?", "Shift summary", "Explain in Hindi". The on-screen keyboard allows free text.
- **Guardrails:**
  - If a reply contradicts the verdict, or quotes a number that isn't in the record, replace it with the template answer and log the event.
  - The Result screen always shows **which backend answered**, either "Llama 3.2 1B · ExecuTorch 1.5.1 · 4 threads" or "TEMPLATE FALLBACK", so template text never looks like Llama output.
  - Also show **"Built with Llama"**.
- **Speed budget (est., to measure):**
  - prefill about 50–100 tokens/s and decode **5–13 tokens/s**, so the first token comes about 3–6 s after a ~300-token prompt;
  - the banner and buzzer fire **immediately**; Llama text streams in afterwards.

### 8.4 Own export (M1b, due Tue 10)

On the Mac: Apple Silicon, macOS 14 or later, `brew install python@3.12`.

```bash
python3.12 -m venv ~/et15x && source ~/et15x/bin/activate
pip install executorch==1.5.1 torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install -U huggingface_hub && hf auth login
hf download meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8 --local-dir ~/llama/1b-sq
# config: examples/models/llama/config/llama_xnnpack_spinquant.yaml from the release/1.5 branch
python -m executorch.extension.llm.export.export_llm --config llama_xnnpack_spinquant.yaml \
  +base.model_class=llama3_2 +base.checkpoint=$HOME/llama/1b-sq/consolidated.00.pth \
  +base.params=$HOME/llama/1b-sq/params.json +export.output_name=Llama-3.2-1B-Instruct-SpinQuant-TriNetra.pte
```

- **The name must start with "Llama"** (Llama 3.2 licence).
- Copy the `.pte` and `tokenizer.model` to the Pi.
- **Use the same ExecuTorch version on Mac and Pi.** If the release/1.5 YAML is missing, use 1.6.0 + torch 2.15.0 on both.
- Make sure `max_seq_length` and `max_context_length` are 2048; the default is 128.
- **Never export without `base.checkpoint`.** Without it the weights are random.
- Don't export on the Pi; it would swap.
- **If Meta hasn't approved access by Mon 9:** try the ungated mirror `unsloth/Llama-3.2-1B-Instruct` with optimum-executorch (8da4w). Whether `TextLLMRunner` loads that output is UNVERIFIED, so test it the same day. Otherwise keep the prebuilt `.pte` and say so.

### 8.5 Benchmark protocol

`tools/bench_llm.py` uses the `stats_callback`.

**Inputs:**
- Prompts of about 64 and 256 tokens, plus a real ~350-token record.
- 128 new tokens, temperature 0.
- 1 warm-up run plus 10 measured runs.

**Report:**
- Median and p90 of time to first token, prefill tokens/s and decode tokens/s.
- Load time, `.pte` size and peak RSS.
- Threads 1–4.
- Each case idle and while vision is running.
- `vcgencmd measure_temp`, `get_throttled`, `measure_clock arm` and `pmic_read_adc`, before and after.

---

## 9. The application, alerts and logs

**Stack:**
- FastAPI and plain HTML/JS, served by `trinetra-vision`, shown in Chromium kiosk mode on the touchscreen.
- MJPEG live view.
- Server-Sent Events for Llama streaming.

**Screens:**
1. **Inspect:** live view with the jig outline, a big **INSPECT** button, and the mode selector.
2. **Result:**
   - **full-width verdict banner:** green PASS, blue CLEAN & RE-INSPECT, amber REWORK, purple HOLD FOR QA, red REJECT;
   - overlay with numbered boxes coloured by severity, and a defect table (type, mm, zone, severity, rule);
   - streamed Llama advice with its backend label, plus the question buttons;
   - **Acknowledge** and **QA sign-off** (name + action), logged.
3. **History:** last N parts, defect Pareto chart, trend, **Export CSV**, **Shift summary**.
4. **System:** live Hailo ms, Llama tokens/s, CPU temperature, throttle flags, and `ip link` state (Wi-Fi and Ethernet **down** in the video).

**Alerts:**
- The banner appears at once.
- **Buzzer:** 1 short beep for PASS, 2 for REWORK or CLEAN, 3 long for REJECT or HOLD.

**Logs:**
- `trinetra.db`: inspections, defects, chat turns and sign-offs. Open one SQLite connection per thread; `store.py` currently isn't thread-safe.
- JSON and an overlay PNG per part.
- CSV export.
- **`trinetra replay <dir>`** runs the whole pipeline on recorded sequences, so judges can run it without the rig.

---

## 10. Toolchain bring-up (Pi day: Fri 6, once the Pi is on the network)

```bash
# Raspberry Pi OS Trixie 64-bit (Raspberry Pi Imager: set hostname, user, SSH)
sudo apt update && sudo apt full-upgrade -y && sudo rpi-eeprom-update -a && sudo reboot
sudo apt install -y dkms hailo-h10-all imx500-all python3-picamera2 python3-opencv python3-gpiozero && sudo reboot
hailortcli fw-control identify          # expect: Device Architecture HAILO10H, firmware 5.1.1
sudo apt-mark hold h10-hailort h10-hailort-pcie-driver python3-h10-hailort hailo-h10-all hailo-tappas-core python3-hailo-tappas rpicam-apps-hailo-postprocess
rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/hailo_yolov8_inference.json   # AI Camera -> Hailo demo
git clone https://github.com/raspberrypi/picamera2-examples && cd picamera2-examples/examples/hailo && python3 detect.py
```

**Hailo packages:**
- **Never install `hailo-all`, `hailort` or `hailo-gen-ai-model-zoo`.** They pull in the Hailo-8 stack and break the H10 driver. Ignore any hailo-apps docs that say otherwise.
- **After it works, stop running `full-upgrade`.** Kernel updates have broken the Hailo DKMS driver before. Image the SD card as a backup.
- Use precompiled HEFs **only** from `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/`.

**Repo on the Pi:**
- `python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e . --no-deps`
- Trixie ships numpy 2.2 and OpenCV 4.10.
- In `rig.pi.yaml`: set `mm_per_px: 0.08`, delete the USB-camera placeholder, and fix the `/home/pi` paths.

**Versions to pin and write in the README:**

| Component | Version |
|---|---|
| OS | Raspberry Pi OS Trixie 64-bit |
| HailoRT | 5.1.1 |
| TAPPAS | 5.1.0 |
| DFC / Model Zoo | 5.1.0 / v5.1.0 |
| ExecuTorch | 1.5.1 |
| torch | 2.14.0 (CPU) |
| Ultralytics | pinned release ≥ 14 days old |
| Llama | Llama 3.2 1B-Instruct SpinQuant INT4 |

---

## 11. Day-by-day schedule

**Roles:**

| Role | Owns |
|---|---|
| **A** | hardware, specimens, capture, labels review, measurements, video |
| **B** | vision, Hailo |
| **C** | Llama, app, report |

**Realistic workload: about 200 h** (est.). That is the code audit's ~124 h plus the hardware, specimens, evaluation, report and video.
- **3 people:** the table below.
- **2 people:** merge A and C, and follow the cut order in section 1.2.
- **1 person:** do the walking skeleton, Llama + ExecuTorch, logging, the TEST evaluation, the video and the report. Run YOLO on the CPU and skip Hailo, UV and Hindi.

| Date | A (hardware/specimens/measure) | B (vision/Hailo) | C (Llama/app/report) |
|---|---|---|---|
| **Thu 5** | Admin with the leader (section 2); **buy parts on SP Road**; alloy bend test; **cut S1/S2**; mechanical defects; start corrosion | Hailo Dev Zone + DFC 5.1.0 check; Colab 2026.07 check; start the single-view config refactor (Mac) | HF access; Mac: et15 venv + `.pte` load test; advisor_svc skeleton |
| **Fri 6** | Grow the S1 crack; S2 section photos; ground-truth CSV (mechanical) | **Pi bring-up** (morning): Trixie, `hailo-h10-all`, identify, AI Camera + Hailo demo | **Pi:** et15 venv, `.pte` load, first tokens/s |
| **Sat 7** | Build the base, stand, bars and shroud; fiducials; wire MOSFETs; measure module current | Colab: stock yolov8n → HEF → Pi (**go/no-go 1**); `npu.py` on the stock HEF | Advisor service: streaming, reset, warm-up, prompts, guardrail |
| **Sun 8** (½, Deepavali) | — | Capture tool (SRGGB10, exposure wait, darks, raw ROI saving) | UI skeleton (Inspect/Result), SSE |
| **Mon 9** | Focus at 200 mm; one exposure per mode; dark checks; corrosion ground truth; **pilot capture (3 sequences)** | Fusion, flat-field, golden from P1, fiducial homography → **pilot gate** | **Meta-access deadline** → own export or fallback; Result screen |
| **Tue 10** (½, Balipadyami) | **Full capture: 160 sequences + P5 video shots**; copy to USB | Label projection; Label Studio | **Own `.pte` export done (M1b)**; `bench_llm.py` |
| **Wed 11** | **Walking skeleton on the Pi** (all three); label review; P2-B contamination session; UV + blue/white bench tests | YOLO training; walking-skeleton vision (anomaly map + geometry typing) | History screen, CSV, buzzer, Acknowledge/QA sign-off |
| **Thu 12** | **S1 penetrant** (time-lapse 2–30 min) | **Custom HEF compile** (1,024 TRAIN crops); NCNN CPU export in parallel | Shift summary; Hindi; backend label; "Built with Llama" |
| **Fri 13** | Build photos for the report; `bench_vision.py` + power meter setup | **Go/no-go 2**; `npu.py` with the chosen model; seam merge; fusion with the anomaly map | Integrate the advisor with real records |
| **Sat 14** | **P4 penetrant** | Penetrant mode (blue-dark + white-red) | End-to-end runs on the Pi |
| **Sun 15** | Video dry run | Tune thresholds on **TRAIN only** | Report skeleton; two-pipeline diagrams |
| **Mon 16** | Count ground truth for the evaluation; vision and power benchmarks | **TEST evaluation (once, frozen model)** | Llama benchmarks; real-time targets (M7) |
| **Tue 17** | Export `replay` samples | Bug fixes | **Advice evaluation** (section 12); report results |
| **Wed 18** | — | **Freeze.** GitHub Release: own `best.onnx`, `.hef`, sample `.npz` | README, LICENSE (AGPL-3.0), NOTICE, DATASETS.md, `models/README.md` |
| **Thu 19** | **Fresh S1 penetrant (morning); shoot the video** | Support the shoot | Report final draft |
| **Fri 20** | Edit the video (≤ 5 min, assumed) | Tag `v1.0-phase1` | Proofread the report |
| **Sat 21** | Buffer | Buffer | Buffer |
| **Sun 22** | **SUBMIT** (repo link, video link, report PDF via the form) | | |
| Mon 23 | Official deadline | | |

**Critical path:** parts → rig → focus/exposure → capture tool → **pilot gate (Mon 9)** → dataset → labels → YOLO → HEF → `npu.py` → TEST evaluation → report results → video.

**Changes from section 0A (5 Nov evening).**

*Role A:*
- **Thu 5 / Fri 6:** buy the rivet kit (4 mm blind rivets, hand riveter, 4.1 mm drill, saw blades, files, M4 steel screws, salt, vinegar).
- Print the 1:1 templates from `docs/twin/templates/` and check the 50 mm bar.
- **Photograph every face clean first.**
- **Start corrosion (P4A) on Fri 6.**
- Drill and rivet P1A/P2A/P3A/P5A on Fri 6 and Sat 7, then make the defects in the order given in `coupon_fabrication.md`.

**Digital-twin track** (role B, or a helper; at most about 4 person-days):

| Date | Twin track |
|---|---|
| Thu 5 (eve) | Twin code built (sampler, textures, Blender renderer, post, training/ablation, Hailo kit, aero rules + advisor guardrails) |
| Fri 6 | Pilot: 20 scenes; measure s/scene; check labels on the debug overlays |
| Sat 7 | Match materials and pod brightness to the first real photos of **P1A/P3A only (never P5A)**; start the long render (overnight on the M3, or a JarvisLabs L4) |
| Sun 8 | Long render finishes; synth-only training (M3 `mps` or Colab) |
| Wed 11 | Real tiles from `realdata.py`; train real-only, 25 % real, synth→real and mixed |
| Thu 12 | `training/ablation.py` on real P5A; pick the model for the HEF compile (calibrate with **real** tiles) |
| Fri 13 | **Twin go/no-go** (section 0A), together with Hailo go/no-go 2 |
| Tue 17 | Advisor regression on 200 twin records (`twin/records.py`): guardrail violations must be 0 |

**Repo work by file** (lines and estimates are in `docs/research/2026-11-05_code_audit.md`):

| Area | Change | Est. |
|---|---|---|
| config | Single-view config with light roles | 9 h |
| capture | Raw SRGGB10, flush plus exposure wait, raw ROI saving | — |
| fusion | New `fusion.py` | — |
| golden | P1-only golden with pooled variance, plus self-reference | 6 h |
| channels | Blue-channel handling | 3 h |
| features | Geometry typing (cut from 13 h by the skeleton approach) | about 4 h |
| npu | `npu.py` + seam merge, plus training and compile | 34 h |
| advisor | Advisor service and `llm.py` fixes | 13 h |
| app | App, alerts, export, replay | 22 h |
| benchmarks | Benchmark scripts | 3.5 h |
| dataset | Dataset tools | 5.5 h |
| simulator | Single-view simulator and tests | 5 h |

---

## 12. What to measure (results section of the report)

| Area | Metric | How |
|---|---|---|
| Detection | Per-defect recall and precision on TEST plates (P2, P5), **with Wilson 95 % intervals** (e.g. 13/15 = 62–96 %) | Split **by plate**; count physical defects, not images |
| Classification | Confusion matrix over 5 classes + unknown (using `eval_class`) | TEST |
| Measurement | Size error (mm MAE) against caliper or section | Ground-truth CSV |
| False alarms | False-reject rate on good plates | P2, all placements |
| **Twin ablation (required since 5 Nov)** | real-only / 25 % real / synth-only / synth→real / mixed on real P5: per-class recall with Wilson CI, FP per clean plate, tile mAP50 (`training/ablation.py`) | Same split; P5 never rendered or tuned on |
| Twin realism | Render vs. photo of the same coupon under the same pod; clean-plate patch statistics | `docs/twin/` figures |
| Ablation (optional) | Fusion vs. white-only; Hailo INT8 vs. ONNX FP32 mAP | Same split |
| Vision speed | Capture, pre-processing, **Hailo ms per 9-tile batch**, end-to-end ms; p50 and p95 | `perf_counter` per stage; `hailortcli run` |
| Real-time targets | Verdict ≤ 4 s p95, first token ≤ 6 s, full advice ≤ 20 s, ≥ 10 parts/min (est. targets) | 20 consecutive parts |
| Llama | `.pte` size, load s, time to first token, prefill and decode tokens/s, peak RSS, idle vs. loaded | `bench_llm.py` |
| System | Board watts (inline USB-C meter), CPU and NPU °C, throttling | `vcgencmd` |
| Penetrant | Indications on S1/P4 vs. ground truth; linear/rounded correct | Photos + ground truth |
| **Advice quality** (Tue 17, owner C) | 30 records: Llama vs. the template baseline, **scored blind by 2 people** (plus the mentor if free) for correctness, actionability and clarity, 1–5. **Automatic checks:** every number Llama quotes matches the record; **answers contradicting the verdict = 0** | A rubric table in the report |

**Honest scope for the report:**
- Make **no industrial-accuracy claims**.
- State the dataset: about 50 physical seeded defects, one material, one jig, a 0.08 mm/px grid (0.064 mm/px sensor).
- Penetrant chemistry takes about 35–45 min. Only the AI step takes seconds.

---

## 13. Deliverables

### 13.1 GitHub repo

```
README.md          photo, BOM, wiring, install (exact pins/versions), run + replay commands, results table
LICENSE            AGPL-3.0 (Ultralytics YOLO is used for training/weights)
NOTICE             Llama 3.2 Community License + "Built with Llama"; Ultralytics (AGPL-3.0);
                   Hailo Model Zoo (MIT); source of any prebuilt .pte (software-mansion) and its revision
DATASETS.md        own data description; public datasets only via download scripts, with licences
docs/              Phase1_build_plan.md, report source, diagrams, wiring photo, research/
configs/           rig.pi.yaml (single camera, 7 light channels), spec.yaml
src/trinetra/      vision pipeline, fusion.py, npu.py, advisor_svc.py, app/
tools/             capture_dataset.py, project_labels.py, train_yolo.md, compile_hef.ipynb,
                   bench_llm.py, bench_vision.py, fetch_models.sh (pinned revision + SHA-256)
models/README.md   how to get the .hef and .pte (no weights committed; never commit the DFC wheel)
data/real/sample/  a few recorded sequences for `trinetra replay`
```

**GitHub Release `v1.0-phase1` assets:**
- the team's own `best.onnx` and `.hef`;
- the TEST `.npz` files (or a Drive link), so the evaluation can be checked.

For a future product: Ultralytics needs an Enterprise licence, or the detector must be replaced with a permissively licensed one (Phase 2 note).

### 13.2 Technical report (about 12–15 pages; there is no official limit, so ask)

1. Problem and mapping to the brief (table 1.1).
2. System overview: device photo and block diagram.
3. Hardware: optics, lighting geometry, wiring, safety.
4. **Pipeline A, vision on HailoRT/Hailo-10H:** capture, fusion image, tiling, YOLO, anomaly map, rules.
5. **Pipeline B, Llama on ExecuTorch/Cortex-A76:** model, quantisation, export recipe, runner, prompt, guardrails.
6. The interface between them (the JSON record) and why they run as separate processes.
7. Dataset: specimens, ground truth, split by plate.
8. Results (section 12), with honest limits.
9. Operator experience: screens, alerts, Q&A examples, sign-off.
10. Penetrant mode.
11. Limitations and Phase 2 roadmap.
12. References and licences.

Frame the discussion around the judges' five aspects. Add an **"Arm tools used"** box: ExecuTorch, XNNPACK, KleidiAI, Cortex-A76, HailoRT.

### 13.3 Demo video (≤ 5 min; the length is an assumption, so ask)

| Time | Shot |
|---|---|
| 0:00 | Title and the problem in one line |
| 0:10 | Hardware tour; **Wi-Fi off and Ethernet unplugged**, shown on the System screen |
| 0:30 | Good plate → **PASS**, with the time shown |
| 0:50 | P5 (dents + scratches) → boxes, mm, severity, **REJECT** + buzzer; Llama advice streams, with its backend label visible |
| 1:50 | "Why?" / "How do I fix it?" → Llama answers; one answer in Hindi; operator taps Acknowledge |
| 2:20 | Marker dot or oil on a good plate → **unknown → HOLD FOR QA** (optional UV → contamination → CLEAN & RE-INSPECT) |
| 2:40 | Crack: penetrant process (sped up) → fresh S1 → **linear indication → REJECT (ASME App. 8)** |
| 3:30 | History: log, Pareto, CSV export, **Llama shift summary** |
| 4:00 | Under the hood: the two-pipeline diagram and live numbers (Hailo ms, tokens/s, °C, W) |
| 4:40 | Close: Phase 2 roadmap |

---

## 14. Risks and fallbacks

| Risk | Fallback / decision point |
|---|---|
| A Hailo HEF won't load (version mismatch) | DFC 5.1.0 / Model Zoo v5.1.0 only; stock-HEF proof by **Sat 7** |
| No DFC 5.1.0 wheel, or Colab can't run it | x86 Ubuntu 22.04 GPU VM; otherwise **Fri 13**: YOLO on the CPU (NCNN), documented |
| The dataset turns out unusable after the plates are dyed | Pilot gate Mon 9; save raw Bayer for every state; no penetrant on P5 before the video; TEST data backed up on USB |
| Prebuilt `.pte` won't load on 1.5.1 | ExecuTorch 1.4.1 + its torch pin, then our own export (M1b) |
| Meta approval is late | Unsloth mirror + optimum-executorch (UNVERIFIED with `TextLLMRunner`); otherwise the prebuilt `.pte`, stated in the report |
| Llama slow (< 5 tokens/s with vision running) | Shorter prompt, `max_new_tokens` 100, pinned threads, Llama runs after vision; the banner never waits for it |
| Dark-field too noisy on brushed plates | Shroud, grain left-right, flat-field with a floor, pooled golden, 20/30 ms exposure |
| Plate outline not found under the diffuse lights | Fiducial homography (primary method) plus ECC |
| No real crack (soft alloy) | S1 notched-strip fatigue crack; slits labelled "crack-like notch" |
| Thermal or power trouble | Active Cooler, heatsink, 27 W PSU; log `get_throttled` |
| A driver breaks after an update | Freeze apt after bring-up; SD-card image |
| Deadline moves or the form arrives late | Check daily; submit Sun 22 |
| Licence problems | AGPL-3.0, NOTICE, "Built with Llama", datasets never committed |
| Synthetic data does not transfer (the literature says synth-only < real-only) | Synth is pretraining only; **Fri 13 go/no-go**: if synth→real does not beat real-only on P5, ship real-only and report the ablation |
| Twin eats the schedule or the disk (about 8 GB free) | Cap at about 4 person-days; batches delete their work dirs; stop at < 2 GB free; render on a JarvisLabs L4 if the M3 is too slow |
| Overclaiming in aerospace wording | Section 0A word list; the advisor's banned-phrase guardrail |

---

## 15. Phase 2 roadmap (mention it in the report; don't build it now)

- **Throughput:** turntable or conveyor feed with a reject flap; an asynchronous Hailo pipeline; continuous video.
- **3D:** structured light, or photometric stereo on developer-coated parts, for dent depth in mm. The multi-camera and moving-camera ideas belong here.
- **Resolution:** HQ or Global Shutter camera with a 16 mm lens, about 20 µm/px; the handheld scanner concept (docs/research/06).
- **AI:**
  - PatchCore backbone on the Hailo; CLIP-embedding anomaly scoring;
  - IMX500 on-sensor "part present" gate (if allowed);
  - LoRA fine-tune of Llama 3.2 1B on SOP Q&A;
  - Hindi/Kannada voice.
- **Coverage:** more part types and materials, plus NEU-DET/GC10-DET pretraining; a permissively licensed detector for productisation.
- **Fluorescent penetrant:** 365 nm with a long-pass filter.

---

## 16. Questions the team must answer this week

1. **Touchscreen:** which model is it? Touch Display 2 (DSI), HDMI or SPI? This decides the port and the GPIO map.
2. **Team:** how many people (1–3), and who is the registered team leader? Did the registration confirmation arrive?
3. **Equipment:** do you have the 27 W PSU, the Active Cooler, a vise, a drill, a hacksaw and a caliper?
4. **HAT:** is it the official AI HAT+ 2? Confirm with `hailortcli fw-control identify` on Pi day.

---

### Sources (main ones; full lists in docs/research/2026-11-05_*.md)

**Challenge**
- Challenge page: https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html
- Deadline change: https://github.com/arm-education/Arm-Developer-Labs/pull/188
- PS5 mentoring video: https://www.youtube.com/watch?v=AsISV9eDakY

**Hardware**
- AI HAT+ 2: https://www.raspberrypi.com/products/ai-hat-plus-2/ ; https://www.raspberrypi.com/documentation/computers/ai.html
- AI Camera: https://www.raspberrypi.com/documentation/accessories/ai-camera.html

**Hailo**
- Model Zoo v5.1.0: https://github.com/hailo-ai/hailo_model_zoo/tree/v5.1.0
- hailo-apps: https://github.com/hailo-ai/hailo-apps

**ExecuTorch and Llama**
- ExecuTorch on PyPI: https://pypi.org/project/executorch/
- LLM runner: https://github.com/pytorch/executorch/tree/release/1.6/extension/llm/runner
- Raspberry Pi tutorial: https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html
- Prebuilt `.pte`: https://huggingface.co/software-mansion/Llama-3.2-ExecuTorch

**Arm learning resources**
- Course: https://learn.arm.com/courses/executorch-on-arm/
- Learning Path: https://learn.arm.com/learning-paths/embedded-and-microcontrollers/rpi-llama3/
