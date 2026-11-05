# Making the TriNetra test coupons

**Status:** build guide (5 Nov 2026). For the students making the coupons in Bengaluru.

The coupons are the only ground truth of the project. Every metric in the report is measured on real photos of these plates, and the digital twin copies their layout (except P5). Make them carefully, measure them honestly, and write down what you actually made.

**Single source of truth.**
- `configs/coupons.yaml` describes every imaged face: holes, rivets, defects, distractors, the fabrication steps (`make`) and the rule findings each coupon is designed to trigger (`expect`).
- The printed templates in `docs/twin/templates/` (one SVG per face, plus `index.html`) and the ground-truth CSV template are generated from it:

  ```
  .venv/bin/python -m trinetra.twin.coupons validate
  .venv/bin/python -m trinetra.twin.coupons table P2A
  .venv/bin/python -m trinetra.twin.coupons templates          # SVGs, index.html, ground_truth_template.csv
  ```

- If you change anything on a plate, change `configs/coupons.yaml` too, then re-run `validate` and `templates`.

**Coordinates.** Millimetres from the datum corner **A**, the top-left corner of face A: u to the right, v down. Angles are measured from +u towards +v (0 = right, 90 = down, 180 = left, 270 = up).

## 1. What you are making

Five 3 mm aluminium plates, 100 × 100 mm, alloy unknown. Each row below is one imaged face.

| Face | Name | Role | Size (mm) | What is on it |
|---|---|---|---|---|
| P1A | Reference (golden) | train | 100 × 100 | 2 rows of 4 blind rivets at v = 15 and 85, u = 20…80 (pitch 20 mm = 5D, edge 15 mm). No defects, ever |
| P2A | Fastener faults | train | 100 × 100 | P1 layout plus: missing rivet (H3), filed oval hole (H6), extra hole at 10 mm pitch (H9), rivet 5 mm from the edge (H10), burrs on the open holes, simulated fretting streak, 4 saw slits of 1/2/4/6 mm from rivet holes |
| P3A | Dents and scratches | train | 100 × 100 | 1 rivet row; 3 punch dents near rivets, 3 far; 2 ball dents; 3 scriber scratches; 1 gouge; 1 abrasion patch; 2 edge nicks; 1 blend (rework mark) |
| P4A | Corrosion | train | 100 × 60 | 2 M4 steel screws with galvanic rings, 2 pitting patches, 2 hacksaw edge slits |
| S1A | Fatigue crack | calib | 100 × 20 | 1 hole, 1 mm starter notch, a 2–5 mm notch-initiated fatigue crack; the penetrant demo strip |
| S2A | Punch calibration | calib | 100 × 20 | 9 punch marks (3 settings × 1/2/3 hits); later sectioned to measure true depth |
| P5A | Held-out test plate | **test** | 100 × 100 | A smaller copy of every type in a different layout. Never rendered for training, never used to tune anything |

S1 and S2 are cut from P4. D = 4.0 mm is the fastener diameter used by the demo rules:
- **AC 43.13-1B 4-57c:** edge distance (hole centre to plate edge) at least 2D = 8 mm; pitch (centre to centre) at least 3D = 12 mm.
- **Piper SL 1095A:** a dent closer than 2D to the nearest fastener is "minor" (needs action); one farther away (and shallow, with no scratch or crack within 2× its diameter) is "negligible".

These are public, illustrative limits only (section 11).

## 2. Buy list

You already have: the five plates, an automatic centre punch, the red-dye penetrant kit (Type II, visible), black nitrile gloves.

| Item | Notes |
|---|---|
| 4.0 mm aluminium blind rivets, dome head, pack of 50 | You need 25 (8 + 8 + 4 + 5) plus spares. **Grip range:** packs are sold as diameter × length (for example 4.0 × 8). The pack or seller's table gives the grip range, the total thickness the rivet clamps. It must include 3.0 mm, ideally mid-range. If 3.0 mm is at the end of a range or outside it, add a 1 mm aluminium backing strip on face B under each rivet row. The stack is then 4 mm and the joint becomes a skin + doubler, which is more aircraft-like anyway. The strip is on the back, so face A and the twin do not change |
| Hand riveter with a 4.0 mm nosepiece | Any standard hand blind-rivet tool |
| HSS drill bits: 4.1 mm (two, they break) and 2.5 mm (pilot) | Plus an electric hand drill (borrow one) or a drill stand. One 4.1 mm drill serves both the rivets and the M4 screws |
| Files | A needle-file set (round, triangular, flat) and one 150–200 mm flat file |
| Scriber | Hardened steel or carbide tip |
| Saws | Jeweller's saw frame with fine blades; junior hacksaw with fine (32 TPI) blades |
| Abrasive paper | 220 and 400 grit (wet-and-dry) |
| Steel ball, 20–25 mm | A bearing ball from a bearing shop |
| M4 × 10 mm pan-head steel screws and M4 nuts, 4 each | Stainless or plain steel. **Not zinc-plated:** zinc corrodes first and protects the aluminium. If only zinc-plated screws are available, soak them in white vinegar overnight to strip the zinc |
| Salt, white vinegar, tissue paper, an airtight plastic box | For the corrosion patches |
| Masking tape, isopropyl alcohol (IPA), cotton buds, lint-free cloth | Cleaning and masking |
| Soft pencil (6B) or graphite powder, a drop of light oil | The simulated fretting streak |
| Measuring | Digital caliper (150 mm, 0.01 mm), steel rule with 0.5 mm divisions, 10× loupe |
| Workshop | Safety glasses, small bench vise with soft jaws (card or aluminium angle), a flat steel block (any flat piece about 50 mm, or the vise anvil), a 300 g hammer, a scrap wooden board, C-clamp |
| Optional | 1 mm aluminium strip (backing strip); 3 mm steel letter punch "A"; dial indicator on a stand (dent depth); feeler gauges (slit width) |

Buy from local hardware and tool shops or online industrial suppliers. Buy only genuine, labelled items.

## 3. Alloy check (before the S1 fatigue crack)

The alloy is unknown. It decides whether a real fatigue crack will grow in S1 and how we describe the coupons.

**Reference hardness (AC 43.13-1B Table 4-5, Brinell):** 1100-O 23, 1100-H18 44, 5052-H36 73, 6061-T6 95, 2024-T4 120, 7075-T6 135.

1. **File test (any plate edge, 2–3 strokes).**
   - Soft alloys (1100, 3003) clog the file and leave long curly chips.
   - Harder heat-treated alloys (6061-T6, 2024, 7075) file crisply with fine chips.
2. **Bend test (the S2 offcut sliver from the section cut, section 6.6).**
   - Clamp it in the vise, bend it to 90°, then hammer it flat on itself.
   - It folds flat without cracking: a soft alloy (1xxx, 3xxx or 5xxx).
   - It cracks on the outside of the bend: a harder heat-treated alloy (6xxx-T6, 2xxx, 7xxx).
3. **Caustic-soda test (AC 43.13-1B 4-23b). Optional and dangerous; skip it unless a lab supervisor runs it.**
   - One drop of 10 % sodium hydroxide on the sliver. Copper-bearing alloys (2014, 2017, 2024) turn black.
   - No blackening does NOT separate 1100, 6061 and 7075.
   - It causes burns and gives off hydrogen. Outdoors only, goggles and gloves, rinse with lots of water.
4. **Record the result** in the report and in `configs/coupons.yaml` (a comment next to `plate`).
   - Soft alloy: the S1 crack may tear instead of forming a tight crack (section 6.5). Describe what you got honestly. If the team wants real fatigue cracks, buy one 6061-T6 plate with a mill certificate.

## 4. Order of operations (this matters)

Work in this order. Gloves on whenever you handle a plate; skin oil shows up in the photos.

1. **Day 1, first: mark and photograph everything clean.**
   1. Choose the cleaner face of each plate as face A. Punch a small "A" in its top-left corner, about 4 mm in from both edges: use a 3 mm letter punch, or 5–7 light dots (lowest punch setting) in an A shape. Keep the 8 × 8 mm corner clear of everything else.
   2. Photograph faces A and B of P1–P5 **bare**, before any work: the bare-plate set. Use the TriNetra capture tool if it is running. Otherwise use a phone on a stand, for documentation only.
2. **Day 1: cut P4 and start the corrosion.** Corrosion needs 48–72 h, so it starts first.
   1. Cut S1 and S2 off P4 (section 6.4).
   2. Drill P4 and fit the screws. Drill P5's holes (no rivets yet).
   3. Mask both plates and start the salt + vinegar patches on P4 (D1–D4) and P5 (D8).
3. **Day 1: P1, the golden reference.** Drill, deburr and rivet P1 (section 6.1).
   - Capture P1A with the TriNetra device as soon as it runs, at every pose and light setting used for the dataset. P1 never changes after riveting, so the golden set can be re-shot at any time.
   - **Never** damage, mark, corrode or penetrant-test P1.
4. **Day 1–2: one punch session with marked settings.**
   1. Mark the punch cap at three positions (low, mid, high).
   2. Punch S2A, then the P3A dents (before P3's rivets).
   3. Punch the P5A dents later, after P5's corrosion, with the same cap marks.
5. **Day 1–2: S2 sectioning and the alloy check (section 3).** Then P3 (rest), P2 and the S1 notch. Grow the S1 crack after the alloy check.
6. **Day 3–4: finish the corroded plates.**
   1. Stop the corrosion and rinse.
   2. Cut P4's edge slits.
   3. Finish P5: oval, slit, dents, rivets, scratches, nick.
7. **Measure the ground truth** (section 8) as each coupon is finished. Mechanical defects first (Fri 6), corrosion when it is done (Mon 9).
8. **Capture the datasets.** Pilot on Mon 9, the whole surface dataset on Tue 10, including video-quality shots of P5. Record which imaged corner is A in every capture.
9. **Penetrant last** (section 7).
   - **S1 only after its surface photos** (Thu 12). P4 optionally, after its photos (Sat 14).
   - **P5 only after the demo video** is shot. **Never P1.**
   - Do a fresh S1 run on the morning of the video shoot (Thu 19): the indications fade.

**P5 rule.** Touch P5 only as its template says. Its photos are the test set. They are never used for training, tuning thresholds, choosing models or rendering twin scenes (`coupons.for_replica` refuses P5). Store it in a bag labelled TEST.

## 5. Workshop basics (used on every coupon)

- **Print the template.**
  - Open the face's SVG from `docs/twin/templates/` in a browser and print it at **100 % / actual size**, never "fit to page".
  - Check the 50 mm bar with the caliper; it must read 50.0 ± 0.3 mm.
  - Tape the print on face A with the outline on the plate edges and the "A" on the datum corner.
- **Mark through the paper.**
  - Centre-punch hole centres and punch dents through the paper at the crosshairs.
  - For scratches, prick the start dot and end cross lightly with the scriber, then remove the paper.
  - Use marker only on tape, never on the plate.
- **Drill.**
  - Clamp the plate on scrap wood; never hold it by hand.
  - Drill a 2.5 mm pilot, then 4.1 mm, square to the face, at moderate speed with a drop of oil.
- **Deburr.** Twist a larger drill bit (8–10 mm) by hand in each hole edge, on both faces: just enough to remove the burr, not a countersink. Leave burrs only where the template says "burr (leave)".
- **Rivet.**
  - Rivet from face A (heads on face A). Hold the riveter square and pressed onto the head, and squeeze until the mandrel snaps.
  - The head must sit flat. Re-drill and redo a crooked one on P1.
- **Punch dents.**
  - Plate flat on the steel block, punch vertical, with the cap at the marked setting.
  - For "x2" and "x3", fire again with the tip kept in the same mark.
- **Clean before photographing.** Wipe with IPA on a lint-free cloth, with gloves on. Do not wipe the fretting streak or the corrosion powder.

## 6. Coupon by coupon

The numbered `make` steps are printed on each template and listed in `docs/twin/templates/index.html`. This section adds the how and the why.

### 6.1 P1A Reference (golden)

- 8 holes at u = 20, 40, 60, 80 on rows v = 15 and v = 85. All riveted.
- Edge distance 15 mm (3.75D) and pitch 20 mm (5D): comfortably inside the 2D and 3D minimums.
- This is the "good" class and the golden reference. Make it neatly; it is the standard every other plate is compared against.

### 6.2 P2A Fastener faults

| Id | What | How |
|---|---|---|
| H3 | Missing rivet (open hole) | Drill from **face B** so the exit burr is on face A; do not deburr face A (burr X1) |
| H6 | Oval open hole, 5.1 × 4.1 mm (ovality 1.25) | Drill from face B (burr X2). File along the row with a round needle file, about 0.5 mm each side; check major and minor with the caliper's inside jaws |
| H9 | Extra rivet 10 mm from H8 (2.5D < 3D) | Normal hole and rivet. The two heads end up 2 mm apart |
| H10 | Rivet 5 mm from the left edge (1.25D < 2D) | Normal hole and rivet. The edge may bulge a little when you set it; that is realistic |
| D1–D4 | Saw slits from rivet-hole edges: 1, 2, 4, 6 mm | Cut before riveting. Release one end of the jeweller's-saw blade, thread it through the hole, re-tension it, and cut outwards along the line to the length mark (measured from the hole edge) |
| D5 | Black "smoking rivet" streak at R7, 8 × 2 mm upwards | **Simulated** with graphite: rub a 6B lead on 400 grit, mix the powder with a trace of oil on a cotton bud, lay one streak from the head edge, and smudge it once in that direction |

- **Slits hide under heads.** The 8 mm head covers about 2 mm beyond the hole edge, so D1 (1 mm) is invisible and D2 (2 mm) barely reaches the head edge. That is realistic: cracks under heads cannot be seen. Only D3 and D4 are expected to give visible indications.
- **Burrs are not defects in the YOLO sense.** The rules must send them to HOLD (unknown).

### 6.3 P3A Dents and scratches

1. **Drill the 4 holes; do not rivet yet.**
2. **Punch the dents first**, plate flat on the steel block (rivet tails would stop it lying flat).

   | Dent | Where | Setting |
   |---|---|---|
   | D1 | 2.5 mm from the R1 head edge | low x1 |
   | D2 | 3.5 mm from the R2 head edge | mid x2 |
   | D3 | 4.5 mm from the R3 head edge | high x3 |
   | D4–D6 | more than 15 mm from any head | low x3, mid x1, high x2 |

   - D1–D3 are closer than 2D = 8 mm to a head ("minor"); D4–D6 are "negligible".
3. **Ball dents D7 and D8.**
   - Steel ball on the mark (hold it with pliers or in a ring of tape), plate on the steel block, one hammer blow.
   - Practise on face B of P4 first (never imaged) to learn the force: aim for 5.5 mm (light) and 7.5 mm (firm) across.
4. **Rivet R1–R4.**
5. **Scratches.**
   - Hold a steel rule firmly between the start dot and end cross. Pull the scriber towards you at about 60°.
   - D9: 1 light pass. D10: 3 passes. D11: 5 firm passes.
   - D12 gouge: the corner of a flat file, 3 passes.
6. **Abrasion D13.** Mask a 10 × 10 mm window, then 220 grit on a small flat block (an eraser works), 30 strokes along u.
7. **Blend X1 (a rework mark, not a defect).** Wrap 220 grit over the steel ball and twist it to a 10 mm dish about 0.25 mm deep (20:1, radius ÷ depth). Finish with 400 grit.
8. **Edge nicks D14 (left) and D15 (right).** 3–5 strokes of a triangular needle file across the face-A edge corner, about 2 mm wide.

Keep scratches away from dents, as the template does. Piper's rule makes a dent with a scratch within 2× its diameter "major", which would change the designed result.

### 6.4 P4A Corrosion (and cutting S1, S2)

1. **Cut the strips.**
   1. Scribe lines across P4 at v = 60 and v = 80, on face A.
   2. In the vise with soft jaws, hacksaw just below each line, so the kerf comes off the strip below it.
   3. File the sawn edges straight and square, back to the line; break the sharp edges with 400 grit. Crisp edges matter, because the device finds the plate by its corners.
   4. Strip v 60–80 is S1 and strip v 80–100 is S2. Keep face A up and punch each strip's own "A" top-left.
   5. Measure all three pieces: expect P4A about 59.5 mm and the strips about 19 mm wide. Update `size_mm` in `configs/coupons.yaml`.
      - If a strip is under 20 mm, move its features to mid-width (S1: H1, D1 and D2; S2: the 9 marks).
      - Re-run `validate` and `templates`, and print the new templates before drilling or punching the strips.
2. **Screws.** Drill H1 and H2 at 4.1 mm. Fit the M4 screws with heads on face A and nuts on face B, finger-tight plus a quarter turn. The steel head touching the aluminium makes the galvanic couple.
3. **Corrosion (Day 1).**
   1. Mix 5 g of salt (one level teaspoon) in 100 ml of white vinegar: about 5 % salt in dilute acetic acid, a home version of the acidified salt solution of ASTM G85 A1 (vinegar is more acidic).
   2. Cover face A with masking tape. Cut round windows at D3 (8 mm) and D4 (10 mm), and rings around the screw heads, outer diameter 12 mm (D1) and 14 mm (D2). Expose the screw heads too.
   3. Lay tissue soaked in the solution on the windows. Close the plate in the plastic box, **away from all electronics**.
   4. Re-wet morning and evening (it dries a little in between, which helps), for 48–72 h.
   5. Rinse gently with water, dab dry, and leave the white powder. Remove the tape.
   6. Accept white, pitted spots as corrosion. If nothing shows after 72 h, continue to 5 days.
4. **Edge slits after the corrosion**, with the fine hacksaw: D5 3 mm into the top edge at u = 88, D6 6 mm into the left edge at v = 40.
   - A hacksaw kerf is often 0.6–0.9 mm. Record the measured width; the plan assumes 0.6.

### 6.5 S1A Fatigue crack (penetrant demo)

1. Drill H1 (4.1 mm) at mid-length; deburr.
2. Cut a 1 mm starter notch D1 with the jeweller's saw, from the hole edge towards the bottom long edge (+v).
3. **Photograph S1A with the notch only** (a useful extra image).
4. **Grow the crack.**
   1. Clamp S1 upright in the vise with soft jaws, by one end, with the hole centre 2–3 mm above the jaw tops.
   2. Reverse-bend the free end through the face (towards face A, then face B), about ±15–20° by hand, counting cycles. The bending stress runs along the strip, so the crack grows across it from the notch, towards the long edge.
   3. Look with the 10× loupe every 10 cycles.
   4. Stop as soon as a hairline runs 2–5 mm past the notch tip (planned 3 mm). Bend the strip back to flat and record the cycles.
5. **Measure the crack** under the loupe, with a scale photo. Replace D2's `pts_mm` with the measured path.
6. **Photograph the surface before any penetrant**, then do the penetrant demonstration (section 7).

On a soft alloy the crack may tear or the strip may neck. Report it as it is: "notch-initiated crack in a commercial alloy". Evaluate S1 separately from the saw slits.

### 6.6 S2A Punch calibration

1. **Mark the punch cap at three positions**: low (cap fully out), mid, high (fully in). Use the same marks on P3 and P5.
2. **Punch D1–D9 along v = 10** at u = 10…90, strip flat on the steel block: low x1, x2, x3; mid x1, x2, x3; high x1, x2, x3.
3. **Photograph S2A** and measure each mark's diameter (loupe and 0.5 mm scale).
4. **Section the strip.**
   1. Hacksaw along the strip at about v = 12.
   2. File and sand the cut face up to v = 10, through the mark centres.
   3. Stop when each notch on the cut face is as wide as that mark's diameter on face A. A cone-shaped mark loses at least 0.1 mm of depth for every 0.1 mm you are off-centre, so creep up on the line.
   4. Finish with 400 grit.
5. **Read the depths.** Macro-photograph the cut face beside a 0.5 mm scale and read each depth.
   - Update `depth_mm` for S2A and for every P3/P5 punch dent made with the same setting.
   - The offcut sliver (v 12–20) is the alloy bend-test piece (section 3).

### 6.7 P5A Held-out test plate

The same defect types as P2–P4 in a different layout, so the test measures generalisation, not memory.

| Id | What |
|---|---|
| H1–H7 | A rivet row at v = 80 |
| H1 | Rivet 6 mm from the left edge (1.5D) |
| H3 | Open hole: missing rivet |
| H5 | Filed oval 4.9 × 4.1, open |
| H6–H7 | Pitch 11 mm (2.75D) |
| D1, D2 | Punch dents 2.5 and 4 mm from heads |
| D3, D4 | A punch dent and a ball dent far from fasteners |
| D5, D6 | Scriber scratch and gouge |
| D7 | 4 mm slit from H4 towards H5 |
| D8 | Pitting patch |
| D9 | Edge nick on the top edge |

Order:
1. Drill and deburr all 7 holes.
2. Start the D8 pitting patch on Day 1, with everything else masked (holes and edges too).
3. When the patch is done, finish the plate:
   1. File the oval.
   2. Cut the slit.
   3. Punch the dents with the marked cap settings, then the ball dent.
   4. Rivet H1, H2, H4, H6, H7.
   5. Scratch, gouge and nick.
4. Clean, photograph, measure, and bag it.

## 7. Penetrant demonstration (S1 first; never P1; P5 only after the video)

The kit is a **Type II visible red-dye** penetrant. Its maker data (dwell times, chemistry) is not published, so read the can labels first. Until then, use the generic solvent-removable procedure.

1. **Photograph the surface first.** Penetrant and developer change the surface for good, as far as the dataset is concerned.
2. **Pre-clean** with the cleaner and wipe dry. Work at 16–52 °C.
3. **Penetrant:** spray and dwell 10 min (5–30 min). Keep it wet; re-spray if it starts to dry.
4. **Remove the excess.** Wipe with a dry cloth, then with a cloth moistened with cleaner. Never spray cleaner straight on the part; it washes penetrant out of the crack.
5. **Developer:** shake until the ball rattles, then one thin, even coat from 20–30 cm until the part is just white.
6. **Read and photograph** 10–60 min after developing, with the device (white light, and the blue modules: the red dye looks dark under blue).
7. **Post-clean** with cleaner. Residue attracts moisture and causes corrosion.
8. **Log** temperature, dwell and development times.

The red dye is not expected to fluoresce under the 395 nm LEDs; that is a bench test, not an assumption.

## 8. Measuring the ground truth

Measure every defect after it is made, and photograph it beside a scale. Use the planned values in the template only as a guide.

| What | How | Typical uncertainty |
|---|---|---|
| Position (u, v) | Caliper from the left edge (u) and the top edge (v) of face A, to the feature centre (to the edge of the feature, plus its radius) | ±0.1–0.2 mm |
| Hole diameter, ovality | Caliper inside jaws: major and minor | ±0.05 mm |
| Punch dent diameter | Loupe with a 0.5 mm scale, or a macro photo with a scale | ±0.05 mm |
| Punch dent depth | From the S2 section for the same setting | ±0.03 mm |
| Ball dent diameter | Raking light, caliper or photo with a scale | ±0.2 mm |
| Ball dent depth | Dial indicator zeroed on the flat plate beside it, or the ball formula `depth ≈ R − sqrt(R² − (d/2)²)` (an upper bound, because the metal springs back) | ±0.05 mm |
| Scratch, gouge | Length: caliper, start to end. Width: macro photo with a scale. Depth: light, medium or heavy only | ±0.2 mm / ±0.03 mm |
| Abrasion | Caliper on the patch bounds | ±0.3 mm |
| Slit | Length from the hole or plate edge: loupe and scale. Width (kerf): feeler gauge or photo | ±0.1 mm |
| Fatigue crack | Loupe and scale photo before penetrant; record the penetrant bleed length separately | ±0.2 mm |
| Corrosion | Diameter of the circle that encloses the visible attack: photo with a scale. Galvanic: outer ring diameter | ±0.5 mm |
| Fretting streak | Length from the head edge, and width | ±0.3 mm |

Write the measured values into `configs/coupons.yaml`: positions, sizes, depths, slit lengths and the fatigue path. Add `measured: "<date> <method> <uncertainty>"` to each measured defect, for example `measured: "2026-11-06 caliper +-0.1"`. Then re-run `validate`.

`validate` will flag a measured position that moves a near dent out of its 2–5 mm band, or a hole out of its intended violation. That is a real result. Then either:
- correct the `expect` list, with a note saying why; or
- re-make the defect on a spare area. Never on P5.

## 9. Filling `data/real/ground_truth.csv`

The CSV has the columns of DESIGN section 6: `defect_id, plate, face, cls, kind, u_mm, v_mm, length_mm, width_mm, diameter_mm, angle_deg, poly_mm, eval_class, notes`.

1. **After measuring, regenerate it from the YAML**, so the CSV and the twin never disagree:

   ```
   .venv/bin/python -m trinetra.twin.coupons templates --csv data/real/ground_truth.csv
   ```

2. **Check it.** There is one row per defect (`defect_id` is the coupon's id, e.g. D3, with `plate` P2 and `face` A).
   - Dents and corrosion: a circle (`diameter_mm`).
   - Scratches, slits, abrasion and fretting: an oriented rectangle (`length_mm` × `width_mm` at `angle_deg`, 0–180 from +u towards +v, centred on `u_mm`, `v_mm`).
   - The fatigue crack also has `poly_mm`, a JSON outline in plate mm.
   - `eval_class` = `cls`. Distractors (burrs, the blend) are not rows: they are not defects.
   - `notes` starts with `measured …` for measured defects and `planned` otherwise. **No row may still say `planned` when the dataset is evaluated.**
3. **If you must hand-edit the CSV**, keep the columns and ids, and copy the same numbers back into `configs/coupons.yaml`.

Keep the scale photos with the captures, named `<face>_<defect_id>.jpg`.

## 10. Safety

- **Eyes.** Safety glasses for drilling, punching, hammering, sawing, filing and the bend test. Chips, snapped saw blades and a flying steel ball are the main risks.
- **Clamp the work.** Never hold a plate by hand while drilling: a 100 mm plate that catches on the drill spins like a blade. Saw in a vise, with the free hand away from the blade.
- **Punch and hammer.** Keep fingers out of the punch line. Hold the steel ball with pliers or in a ring of tape.
- **Sharp edges.** Sawn edges, burrs and slits cut. Wear gloves and break every edge that is not a designed defect.
- **Salt and vinegar.** Corrosive to electronics.
  - Keep the corrosion box on a separate table or in another room, never near the Pi, the Hailo HAT, the power bank, the LEDs or laptops.
  - Wash your hands after handling. Pour the used solution down the drain with plenty of water.
- **Penetrant, cleaner and developer aerosols.** Extremely flammable.
  - Spray outdoors or at an open window, at least 3 m from the device, chargers and power supplies, with nothing switched on or off nearby. No flames or smoking.
  - Let parts flash off for at least 5 min before they go back under the camera.
  - Store the cans below 30 °C. Nitrile gloves; change them when wet.
- **Caustic soda (optional alloy test).** Burns skin and eyes and gives off hydrogen with aluminium. Only outdoors, with goggles and gloves, and with a supervisor; otherwise skip it.
- **UV LEDs on the device.** Never look into the 395 nm LEDs.
- **Graphite and aluminium dust.** Wipe it up; do not blow it. Wash your hands.

## 11. Honest wording (report, video, UI)

- **Coupons.**
  - Say "aircraft-representative aluminium coupons (commercial alloy, 3 mm)", never "aerospace-grade".
  - The plates are 2–4× thicker than light-aircraft skins. State the alloy result from section 3.
- **Slits.** They are "crack-like notches (an EDM-notch analogue)", not cracks.
  - The only real crack is the S1 "notch-initiated fatigue crack", and only if it grew. Evaluate it separately: notch and natural-crack responses differ.
- **Simulated fretting.** P2's black streak is graphite, not real fretting residue. Say so.
- **Corrosion.** It is salt + vinegar accelerated pitting and galvanic attack. It is not intergranular or exfoliation corrosion; those cannot be made in 18 days.
- **Penetrant.** Say "a Type II visible red-dye demonstration of an indication-reading aid".
  - Aerospace final acceptance uses Type I fluorescent penetrant (ASTM E1417).
  - The 395 nm LEDs are not a compliant UV-A source.
- **Rules.** They are demo acceptance criteria from public sources, FAA AC 43.13-1B (4-57c) and Piper SL 1095A. Real limits come from the customer drawing or specification. The device is "an inspector's assistant: decision support, human in the loop". Never say airworthy, certifies, approves or releases.
- **Twin.**
  - Say "simulation-pretrained, real-calibrated". Every metric is measured on real photos of these coupons.
  - P5 is never rendered or used for tuning. Never claim "trained in simulation, works directly".
