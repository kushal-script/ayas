# TriNetra FPI module: fluorescent penetrant crack inspection

Prepared 19 Oct 2026 for the Open Qualifying submission (due **23 Nov 2026**). Prices are seller listings checked on 19 Oct; confidence is marked **H / M / L**. "(calc)" marks my own arithmetic and "(suggestion)" marks a design idea that has not been tested.

## 1. What it adds and how it maps to the brief

FPI is the standard industrial test for surface-breaking cracks in aluminium and other non-magnetic metals (ASTM E1417, ISO 3452-1). A fluorescent dye seeps into cracks and pores, a developer draws it back out, and under UV-A the defects glow yellow-green on a dark plate. TriNetra captures that glow with the top camera and measures every indication. The rules engine then grades it with the same linear/rounded and size rules industry uses, and Llama explains the verdict.

| Brief bullet | What the FPI module does |
|---|---|
| Detect surface defects, cracks, anomalies | Finds surface-breaking cracks and pores, including tight cracks the RGB views may miss |
| Classify type and severity | Linear (crack-like) vs rounded (pore-like), size in mm, relevance threshold, accept/reject profiles modelled on ASME VIII App 8 / ISO 23277 (§6) |
| Alert with actionable recommendations | Verdict, stack-light/buzzer, Llama explanation and next steps (quarantine, confirm with a wipe-off re-bleed, check the process step) |
| Log results | Indications, sizes, rule IDs and the process record (dwell, development time, UV source) go to SQLite with the images |
| Real time at the edge | The AI inspection after development takes about 1 s per plate (calc). Penetrant dwell and development (about 35–50 min) are chemical process times; the report should say so |

**What it doesn't change:** the Hailo pipeline (vision models through HailoRT) and the Llama pipeline (ExecuTorch on the Arm CPU) stay separate. FPI adds one more Hailo model (§6.3) and new JSON fields (§6.5).

**How to describe it:** "lighting modelled on ASTM E1417 §6.6.1", not "E1417-compliant". Compliance also needs QPL-listed penetrants, certified personnel and an E3022-certified lamp.

## 2. What to buy

| # | Item | Qty | Price (INR) | Seller | Conf. |
|---|---|---|---|---|---|
| 1 | **ZChek-FP12** fluorescent penetrant, Type 1, Level 3, 520 ml aerosol (usable with Method C, solvent removal) | 2 | 850 each + GST | Zchem Specialities, Bengaluru: [IndiaMART](https://m.indiamart.com/proddetail/zchek-fp12-post-emulsifiable-level-3-fluorescent-penetrant-level-3-sensitivity-2854089976212.html), [system chart](https://zchem.in/products/liquid-penetrant-inspection/consumables-lpi/) | M |
| 2 | **ZChek-CR** cleaner/remover, aerosol (it runs out first: pre-clean, removal, post-clean) | 3 | 220 each + GST | [Zchem store](https://m.indiamart.com/zchemspecialities) | M |
| 3 | **ZChek-DL** non-aqueous developer, aerosol | 2 | 320 each + GST | same | M |
| | *Alternative to 1–3:* Magnaflux ZYGLO **ZL-27A** penetrant ₹723 + **SKD-S2** developer ₹450 (+ ask for SKC-S cleaner) | | + GST | Ansai Tooling, Gurugram: [IndiaMART](https://m.indiamart.com/proddetail/magnaflux-zl-27a-penetrant-aerosol-19969849262.html) | M |
| 4 | **Backup: visible red-dye kit** (cleaner + penetrant + developer, 400 ml each). Works under normal light with the RGB cameras, no UV needed, less sensitive | 1 | 1,175, ships in about 7 days | [Industrybuying (WESAF)](https://www.industrybuying.com/lubricants-wesaf-FAM197650); alt. [Molygraph CDK ₹1,593](https://www.moglix.com/molygraph-cdk-crack-detection-kit/mp/msn2qgpco3r5zl) | M |
| 5 | **365 nm filtered UV torch** ("Wood's lamp"). Your 12-LED torch is almost certainly 395 nm; keep it as a backup | 1 | 439 (delivery est. 26 Oct) | [Flipkart (Clefairy)](https://www.flipkart.com/clefairy-uv-torch-filtered-365nm-black-light-wood-s-lamp-usb-rechargeable-blacklight-flambe/p/itm62e125f25a87d); backup [amiciVision 365 nm 3 W ₹753](https://www.flipkart.com/amicivision-365nm-uv-flashlight-rechargeable-ultraviolet-black-light-4-modes-torch/p/itmcf6c37d443519?pid=TOHH4FY3HWMRY4QR) | M price, L quality |
| 6 | **3 W 365 nm LED on 16 mm PCB**, for GPIO-switched UV like the RGB LEDs (§4); needs a heatsink and a resistor | 2 | 299 each | [Probots](https://probots.co.in/3w-uv-led-365nm-16mm-pcb.html) | M |
| 7 | **SCHOTT GG-495 long-pass filter, 12.5 mm × 3 mm** (Edmund #54-652). Blocks reflected 365/395 nm and violet; passes the 520–550 nm glow. Fits over the AI Camera's 5.75 mm lens | 1 | 3,734 incl. tax, in stock | [Edmund Optics India](https://www.edmundoptics.in/p/gg-495-125mm-dia-longpass-filter/11321/) | H |
| 8 | **3M Virtua 11326** UV safety glasses (rated 200–380 nm) | 2 | 250 each | [DigiKey India](https://www.digikey.in/en/products/detail/3m/11326-00000-100/6828695) | H |
| 9 | Black cloth hood for the UV shot (photographer's black cloth; E1417 §6.6.1 names this method for portable work) | 1 | 200–500 (est.) | local | — |
| 10 | Nitrile gloves, lint-free wipes, closed metal bin for used wipes | — | ~500 (est.) | local | — |
| 11 | *Optional:* **Lutron UV-340A** UV-A meter, so the report can state the measured µW/cm² at the plate | 1 | 3,900 | Vector Technologies, Hyderabad: [IndiaMART](https://m.indiamart.com/impcat/uv-meter.html) | M |
| 12 | Test-sample materials (§7): 1 mm 7075-T6 sheet (~₹489/kg), 2024 plate (~₹300–315/kg), drill bits 0.5/1.0/1.5/2.0 mm, a jeweller's saw, scrap machined aluminium castings | — | a few hundred (est.) | IndiaMART metal sellers, scrap dealer | M/L |

**Total (calc):** about **₹11,000** for items 1–10 (₹3,540 kit incl. GST + ₹1,175 backup + ₹439 + ₹598 + ₹3,734 + ₹500 + ~₹850 local). Add ₹3,900 for the meter.

**Ordering notes:**
- IndiaMART prices exclude 18% GST. Zchem sells aerosols in cases of 12, so ask for single cans. Aerosols are UN1950 dangerous goods and many couriers refuse them, so ask for surface shipping. **Phone Zchem and Ansai this week.** The kit should arrive by about 1 Nov.
- Order the visible-dye backup kit at the same time. It is cheap and protects the demo if the fluorescent kit is late.

## 3. Process SOP for the demo (Type I, Method C, solvent-removable)

| Step | What to do | Time | Source |
|---|---|---|---|
| 1. Pre-clean | Spray cleaner on a wipe and clean the plate plus 25 mm beyond the area. No wire brush or abrasives, which smear metal over cracks | dry ≥5 min | LANL ITM-1306; NASA PT-TE-1426; ISO 3452-1 §8.2 |
| 2. Penetrant | Spray an even coat and keep it wet | dwell **10 min** for cracks in aluminium (NASA table); ISO allows 5–60 min, Magnaflux Method C 10–30 min | NASA PT-TE-1426 Table 3; ISO 3452-1 §8.4.2 |
| 3. Remove excess | Wipe with a dry lint-free cloth, then a cloth *lightly moistened* with cleaner. **Never spray cleaner directly on the plate.** Check under UV that the background is dark | then dry 5–10 min | ISO 3452-1 §8.5.3; Magnaflux process guide |
| 4. Developer | Shake well. Spray a thin, even, slightly wet coat from ~30 cm, with no drips or blotches | — | ISO 3452-1 §8.6.4 |
| 5. Develop | Leave it | **image at exactly 10 min** (allowed 10–60 min) | ISO 3452-1 §8.6.1 |
| 6. Inspect | Put the plate in the jig, hood down, UV on, capture (§5) | ~1 s | — |
| 7. Post-clean | Solvent clean, because residue causes corrosion | — | NASA PT-TE-1426; ISO 3452-1 §8.8 |

- **Per plate:** about 35–50 min (calc). Run 4–5 plates in a staggered batch to save time.
- **Why the time is fixed:** indications grow as dye bleeds out. One study saw 3.26 mm at 2 min become 3.73 mm at 30 min, so always measure at the same development time and log it.
- **Where to work:** do steps 1–5 and 7 **away from the rig** (§8).

## 4. Rig integration (open arch)

- **Camera:** the top AI Camera at 90°, 20–25 cm above the plate, full resolution, about 0.065 mm/px (calc). The 1.5 mm relevance threshold is then about 23 px, which is enough.
- **Filter mount (suggestion):** a 3D-printed matte-black **flip-up cap** holding the GG-495 within 3–4 mm of the lens, fixed with M2 screws through the camera board holes. Give it a hard stop in both positions, since the RGB shots need it fully out of the way. Board holes: ø2.2 mm, 2 mm in from the edges, about 21 × 12.5 mm apart; lens barrel ø5.75 mm (AI Camera product brief).
- **UV source:**
  - Prototype: the 365 nm torch in a clamp next to the top camera, aimed at the plate centre, switched by hand.
  - Automated (suggestion): two 365 nm 3 W LEDs on a heatsink, about 20 cm from the plate on opposite sides of the top camera for an even beam, each switched by a free MOSFET channel. At 5 V a resistor sets the current. The 10 Ω resistors you have give about 0.1–0.14 A, which is dim but safe for a first test. For about 0.35–0.45 A use about 3.3 Ω 5 W. Check the LED's forward voltage on the datasheet first; it is typically 3.4–4.0 V for 365 nm.
- **Darkness:** drop the black cloth hood over the arch for the UV shot only. In the open rig, ambient light would otherwise swamp the glow; E1417 asks for ≤21.5 lx at the surface.
- **Exposure:** locked: `AeEnable=False`, fixed `ExposureTime` and `AnalogueGain`, fixed `ColourGains` (Picamera2 manual, App. C). Start at 100 ms–1 s with gain 2–4, in multiples of 10 ms. Bracket on day 1, and longer exposures are needed with the 395 nm torch. Widen `FrameDurationLimits` to allow long exposures.

## 5. Capture protocol

1. **Clean plate:** run the normal TriNetra RGB capture (all cameras, all LEDs, dark frames). The surface-defect pipeline and its golden model only ever see clean plates, because the white developer coat changes how the plate looks.
2. **FPI processing** away from the rig (§3).
3. **Developed plate:** push it into the same L-stop in the jig, so it lands on the same grid as its RGB capture. Flip the filter down and lower the hood.
4. Capture a **UV-off frame** (filter in), then a **UV-on frame**, at identical settings. Use raw or linear data, not gamma JPEGs, for the subtraction.
5. *Optional* **wipe-off check** on a doubtful indication (ISO 3452-1 §8.7.3): swab it with a little solvent, re-apply a light developer coat, then re-image right away and again at 10 min. A real flaw bleeds back in the same place. The JSON records it.
6. **Post-clean.** Keep a separate set of FPI plates, because residue can leave faint glow and affect later RGB golden data.

## 6. Image processing, classification and severity

### 6.1 Finding indications (CPU, classical)
This follows the published pattern: Yang et al. 2022 (threshold, then ResNet; F1 92.9%) and the RTX US10054552 patent (HSV/Otsu, then shape analysis).

1. **Glow image:** `G = green(UV_on) − green(UV_off)`. Reflected violet lands mostly in blue, so `G − k·blue` also helps if the filter isn't fitted yet.
2. **Uneven UV:** divide by a blurred background estimate.
3. **Candidates:** an adaptive or Otsu threshold, then morphological open/close, then connected components.
4. **Merge rule (ISO 23277):** neighbouring indications closer than the major dimension of the smaller one count as one.
5. **Measure** each indication in mm on the top-camera grid: length *l*, width *w*, area, orientation and peak brightness.

### 6.2 Type and relevance (rules engine, deterministic)
- **Linear** if *l* > 3·*w*, otherwise **rounded** (ISO 23277 §3; ASME).
- **Relevant** if the major dimension is > 1.5 mm (1/16 in, ASME VIII App 8); smaller ones are logged as non-relevant.
- **Selectable acceptance profiles** in `spec.yaml`. No code is written for plain aluminium plate, and ASTM E433 leaves the limits to the user:

| Profile | Reject if |
|---|---|
| `asme_app8` (default) | any relevant **linear** indication; a relevant **rounded** indication > 3/16 in (4.8 mm); **≥4 rounded in a line** with ≤1/16 in (1.6 mm) edge-to-edge spacing |
| `iso23277_al1/al2/al3` | linear *l* > 2 / 4 / 8 mm; rounded *d* > 4 / 6 / 8 mm |
| optional cluster rule | ≥10 rounded indications in any 37.5 cm² (6 in²) |

- **Severity mapping (suggestion):**
  - **Critical:** a relevant linear indication (likely crack). Reject and quarantine.
  - **Major:** rounded above the limit, or in-line rounded. Reject.
  - **Minor:** relevant rounded within the limit. Accept with a note.
  - **Info:** non-relevant.
  
  Every verdict records the rule ID that fired.

### 6.3 False-call filter (Hailo, HailoRT)
- **Model:** a small CNN (ResNet-18 class) on 64–128 px crops of each candidate, sorting them into **true indication** vs **false** (lint, developer clump, removal smear, scratch bleed, edge glow).
- **Why:** false calls are the main problem in camera FPI. A threshold-plus-CNN design is the published way to fix them.
- **Data:** label crops from your own runs. Stage the false types on purpose (§7).
- **Pipeline:** it runs as a separate HailoRT pipeline. If it isn't ready by 23 Nov, ship the classical path with the rules and show the Hailo model in Phase 2.

### 6.4 Timing (calc)
Subtraction, threshold, components and rules take well under 1 s on the Pi CPU. Hailo crop inference is milliseconds per crop. Llama runs after the verdict and never blocks it.

### 6.5 JSON fields to add
```json
"fpi": {
  "performed": true,
  "system": {"penetrant": "ZChek-FP12", "type": "I", "method": "C", "level": 3},
  "process": {"dwell_min": 10, "development_min": 10, "temp_c": 28},
  "uv": {"source": "365nm torch", "irradiance_uW_cm2": null, "ambient_lux": null},
  "acceptance_profile": "asme_app8",
  "indications": [
    {"id": "F1", "type": "linear", "length_mm": 6.2, "width_mm": 0.4,
     "bbox_mm": [12.1, 40.3, 18.3, 40.9], "relevant": true,
     "nn_class": "true_indication", "nn_conf": 0.93, "wipeoff_rebleed": null}
  ],
  "verdict": "REJECT", "severity": "critical", "rule_ids": ["APP8-LINEAR"]
}
```

### 6.6 What Llama says (it explains and never changes the verdict)
> "Plate P-07 is **rejected**. FPI found a 6.2 mm linear indication near the left edge (F1). Under the ASME App 8 profile any relevant linear indication is a reject, and a linear indication like this is typical of a crack. Next steps: quarantine the plate; if you want to confirm, do a wipe-off re-bleed check on F1; check the bending or heat-treatment step that this batch went through. Two small rounded indications under 1.5 mm were logged as non-relevant."

Follow-up questions use the same JSON plus the SOP text in `spec.yaml`, for example "why is a 6 mm crack rejected but a 3 mm pore accepted?"

## 7. Making test defects

The 3 mm 1050/6061 plates are ductile and won't crack easily, so use a mix:

| Defect | How | Gives | Notes |
|---|---|---|---|
| **Known-size pores** (suggestion) | Drill holes 0.5 / 1.0 / 1.5 / 2.0 mm, 0.5–1 mm deep, then deburr lightly | Rounded indications either side of the 1.5 mm threshold, plus an in-line row of 4 for the cluster rule | Repeatable ground truth; larger and cleaner than real porosity. Say so in the report |
| **Known-length slits** (suggestion) | Jeweller's-saw cuts 3 / 6 / 10 mm long | Linear indications, length checked with a ruler | Wider than real cracks |
| **Real cracks: bending** | Bend 1 mm 7075-T6 or 2024-T3 strip over a tight radius until it cracks | Real tearing cracks | Rougher than fatigue cracks |
| **Real cracks: quench** | ASME V T-653-style: heat a spot on a 2024 block to ~510–525 °C (950–975 °F temp crayon), then quench in cold water; repeat (Colorado DOT QAP 5942) | Crazed network of fine cracks | Proven recipe for 9.5 mm blocks; unverified on 3 mm plate. Needs a propane torch, so do it with lab supervision |
| **Real porosity** | Machined scrap aluminium die-castings from a scrap dealer | Natural rounded and cluster indications | Not repeatable between parts |
| **False calls** (staged) | Lint on the developer, a fingerprint, poor removal, a fine scratch, the plate edge | Training and test cases for the Hailo false-call filter | — |

Commercial reference panels (PSM-5, NiCr twin) cost $700–2,300 with long lead times and no confirmed Indian stock, so skip them for qualifying.

## 8. Safety

- **Aerosols** are extremely flammable (Category 1) and under pressure.
  - Spray outdoors or in a well-ventilated area, **away from the Pi rig, the hot LEDs and soldering irons.**
  - Store below 50 °C, out of the sun.
  - Never pierce or burn cans.
  - Put used wipes in a closed metal bin and dispose of them through the institute's hazardous-waste route.
- **Skin and eyes:** nitrile gloves and safety glasses for all penetrant work.
- **UV:**
  - Never look into the LED or torch, and point it down inside the hood.
  - Wear UV-blocking glasses. The 3M 11326 is rated to 380 nm, so it covers 365 nm but only partly blocks 395 nm.
  - Keep hands out of the beam. The ACGIH-type limit is about 1 mW/cm² for longer than 1,000 s, so a hand in a 1,000 µW/cm² beam reaches it in about 17 min (calc).

## 9. Schedule to 23 Nov (alongside the RGB core)

| Week | FPI work |
|---|---|
| 19–25 Oct | Phone Zchem/Ansai and order the kit (single cans, surface shipping). Order the backup visible kit, the GG-495, the 365 nm torch and LEDs, glasses and gloves. Make the drilled-hole and saw-slit plates; get 7075/2024 strip and scrap castings |
| 26 Oct–1 Nov | First manual FPI run, with the visible kit if the fluorescent one hasn't arrived. Exposure and gain bracketing under the hood with the UV-off/on subtraction. Print the filter cap |
| 2–8 Nov (Diwali 8 Nov) | Classical detection, linear/rounded rules, profiles in `spec.yaml`, JSON fields, Llama prompts |
| 9–15 Nov | FPI dataset: about 10–15 plates × staged false calls; label crops; train and compile the Hailo false-call CNN (x86 host); measure detection of known holes and slits and false calls per plate |
| 16–22 Nov | Freeze. Demo-video segment (spray, dwell and develop shown as time-lapse; inspection in real time); report section. Submit by **23 Nov** |

**Cut order if late:**
1. Hailo false-call CNN (ship classical + rules)
2. GPIO-switched UV LEDs (use the torch by hand)
3. Fluorescent kit (use the visible red-dye kit with the RGB cameras)

## 10. Risks

| Risk | Mitigation |
|---|---|
| Kit delivery (aerosol shipping) | Order this week; visible backup kit; Bengaluru/Gurugram sellers |
| Cheap 365 nm torch is weak or leaks visible light | GG-495 filter; the 365 nm LEDs as a second source; report measured µW/cm² if you buy the meter |
| No real cracks in 3 mm plate | Drilled and saw-cut defects for ground truth; bent 7075 strip and quench-cracked 2024 for realism |
| Uneven UV across the plate | Two-sided LEDs; background normalisation (§6.1) |
| Indication size drifts with time | Image at exactly 10 min of development; log it |
| Developer or residue spoils RGB data | RGB capture before FPI; separate FPI plates; post-clean |
| Judges question "real time" | State that process time is chemical; the AI inspection after development takes about 1 s |

## 11. Sources
- ISO 3452-1:2021 copy: https://weldcalc.ssab.com/sisStandards/ISO%203452-1.pdf
- ISO 23277:2006 preview: https://cdn.standards.iteh.ai/samples/41428/8e55bba667e74817bf33663a9ec3df20/ISO-23277-2006.pdf
- ASTM E1417/E1417M-21e1 preview (§1.2, §6.6.1): https://www.hsenode.com/NewSamples/ASTM/158745421/ASTM-E1417-E1417M-21e1-1.pdf
- Magnaflux process guide (Methods A–D): https://magnaflux.com/Files/Instructions/Penetrant-Testing_Process-Guide_Methods-A-B-C-D.pdf
- NASA PT-TE-1426: https://extapps.ksc.nasa.gov/Reliability/Documents/Preferred_Practices/1426.pdf
- NASA aluminium POD study: https://ntrs.nasa.gov/api/citations/20100015401/downloads/20100015401.pdf
- LANL ITM-1306-NDE-PT-101: https://engstandards.lanl.gov/esm/welding/vol6/ITM-1306-NDE-PT-101-R1.pdf
- ASME App 8 / B31 acceptance summaries: https://www.materialwelding.com/liquid-dye-penetrant-testing-acceptance-criteria/
- Colorado DOT QAP 5942 (quench-crack block): https://www.codot.gov/programs/bridge/bridge-manuals/quality-assurance-procedures/qap_5942_2018_09.pdf/@@download/file/QAP%205942%202018%2009.pdf
- Bleed-out growth study: https://bulletin.is.gliwice.pl/node/414
- Yang et al. 2022, automated FPI with ResNet34: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2022.1081805/full
- Shipway et al., FPI with deep learning (NDT&E Int. 2021): https://www.sciencedirect.com/science/article/abs/pii/S0963869520308008
- RTX patent US10054552: https://patents.google.com/patent/US10054552
- GE patent US10726543: https://patents.google.com/patent/US10726543
- Magnaflux E3022 lamp certification example: https://www.magnaflux.com/Files/COC/LED-UV-Lamp-Certification-Report.pdf
- Picamera2 manual: https://pip-assets.raspberrypi.com/categories/1044-raspberry-pi-ai-camera/documents/RP-008156-DS-6-picamera2-manual.pdf
- AI Camera product brief: https://pip-assets.raspberrypi.com/categories/1044-raspberry-pi-ai-camera/documents/RP-008147-DS-1-ai-camera-product-brief.pdf
- Magnaflux SKC-S aerosol SDS: https://ecolink.com/wp-content/uploads/2015/11/skc-s-aerosol-sds.pdf
