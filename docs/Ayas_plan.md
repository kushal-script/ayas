# Ayas: 3 cameras × 3 white lights, 3D surface map and red-dye crack check, with a Llama operator assistant

**Context.** Bharat AI SoC Challenge 2026-27, Manufacturing & Industry 4.0 theme (real-time industrial defect detection and quality-control assistant). Revised **Tue 20 Oct 2026** for the **Open Qualifying submission due Mon 23 Nov 2026** (report, source code, demo).

**What changed since the 14 Oct version:**
- White light instead of one-colour-at-a-time RGB.
- An open (unboxed) rig, made to work by subtracting a lights-off frame.
- A 3D surface map from photometric stereo.
- Visible red-dye penetrant for cracks.

UV and fluorescent penetrant are on the roadmap only. This plan supersedes BeamTriage, SukshmaScope and Drishti.

**Labels used throughout**
- **(calc)**: my own calculation
- **(est.)**: an estimate to be measured on the rig
- **(sim)**: measured on the simulator, which proves the code paths, not real accuracy
- **\***: an unverified price

**Official brief:** https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html

**Name.** *Ayas* (अयस्). Tagline: "Every surface, seen true." The earlier working name was TriNetra.

---

## 1. What it is, in one paragraph

A flat aluminium plate (100 × 100 × 3 mm) sits against two stops on a matte-black base. A 0–90° arch over the base carries three cameras, at 90° (top), 45° and 15° (grazing), each with a white LED beside it. Two more white LEDs sit on short posts at 45°, 120° either side of the arch.

1. **Capture.** Each LED switches on alone while all three cameras fire together, then one frame is taken with every LED off. That is 6 captures and 18 frames in about 2 s (est.).
2. **Remove room light.** The lights-off frame is subtracted from every other frame. Room light is the same in all of them, so it cancels, and the station works in an open space.
3. **Align.** Every image is warped into one top-down millimetre grid, so each point on the plate has one signature: its colour (top camera), its brightness from each camera/light pair, and its slope from the 3D map.
4. **3D map.** The top camera's images under three lights from different directions give the surface slope at every pixel. Integrating the slope gives a height map, so dents and bumps get a **depth in mm**.
5. **Compare.** The signature is compared against a **golden reference**: the mean and spread of 20+ good plates.
6. **Classify.** Where the plate differs, the pattern across cameras, lights, colour and 3D tells us **what** the defect is. Its size and depth tell us **how bad** it is.
7. **Judge.** A **rules engine** applies the acceptance spec: PASS / CLEAN_AND_REINSPECT / REWORK / HOLD_FOR_QA / REJECT.
8. **Explain.** Everything goes into a **JSON report**. **Llama 3.2 1B-Instruct, running on the Pi's Arm CPU through ExecuTorch**, explains it to the operator and answers follow-up questions. It never changes the verdict.
9. **Crack check (red-dye penetrant).** For cracks too tight to see, the plate goes through the visible-dye process: clean → red penetrant → wipe → white developer. The same top camera then finds red marks, measures them, sorts them into linear (crack-like) or rounded (pore-like), and grades them with ASME- or ISO-style acceptance rules. Llama explains that result the same way.

## 2. How it maps to the brief

| Brief asks | Ayas |
|---|---|
| Real-time camera-based visual inspection | 18 frames per part in about 2 s (est.). Processing about 0.15 s per part on the Mac (sim); to be measured on the Pi |
| Detect surface defects, cracks, anomalies | Golden-reference anomaly detection over every image and the 3D slope map. Red-dye penetrant for tight cracks. Anything unrecognised becomes `unknown` → HOLD_FOR_QA |
| Classify type and severity | Surface: scratch, dent, bump, crack, pit, stain, dust. Penetrant: linear / rounded indication. Severity comes from spec limits per zone, dent/bump depth from the 3D map, and penetrant grading from ASME VIII App. 8 / ISO 23277-style profiles |
| Alert operators with actionable recommendations | Verdict, instruction, per-defect action, likely causes and prevention from the SOP in `spec.yaml`, explained by Llama in plain words, with follow-up Q&A |
| Log results | JSON report + raw frames + overlay (with the 3D height map) per part; SQLite table of inspections and chats |
| Operating in real time at the edge, no cloud | Everything runs on the Pi 5, with no network calls. Works with room lights on (dark-frame subtraction) |
| **Llama via ExecuTorch (mandatory)** | Llama 3.2 1B-Instruct `.pte` (XNNPACK, 8da4w), run by ExecuTorch's `llama_main` on the Cortex-A76 |
| **Hailo and Llama documented as separate pipelines** | Section 11 |
| Optional technologies the brief names | OpenCV (all image processing), Hailo SDK (defect CNN), anomaly detection (golden-reference z-scores), NEU / KolektorSDD (benchmark for the crack/defect CNN) |
| Deliverables: code, app, demo video, technical report | This repo (CLI working now; touchscreen app by 20 Nov), video and report for 23 Nov |

## 3. Rig design

### 3.1 Geometry

```
SIDE VIEW (along the arch)                         TOP VIEW (as the top camera sees it)

        top camera + L90 (90°)                                  P330 (45°)
              |  .                                                 *
              |     .  45° camera + L45                  +-----------------+
              |        \                                 |                 |
              |         \       15° camera + L15         |   plate 100 mm  |
              |          \   ___/                        |                 |
   ___________|___________\_/__________                  +-----------------+
         [ plate against the stops ]               *            ^             *
                                                 P210     arch (front, 90°)   (P330 is
                                                          all three cameras    top-right)
                                                          and L90/L45/L15
```

**Each camera has a light beside it on the arch** (the mentor's design):

| Pair | What it sees best |
|---|---|
| Top camera + L90 | The plate's colour and its overall appearance; the best view for size measurement |
| 45° camera + L45 | The mirror reflection goes away from the camera, so scratches and edges stand out |
| Grazing camera + L15 | Light skims the surface, so scratches, particles and raised edges scatter brightly (a dark-field view) |

Every camera also captures under every other light. This costs no extra time, since the cameras fire together. For example, the top camera under the 15° light shows long shadows from any relief.

**Why two extra LEDs (P210, P330).** Getting 3D shape from one camera needs lights from at least three directions that are **not all in one vertical plane**. The three arch lights are all in the arch's plane. From them alone, only the slope *along* the arch can be measured, so there is no full height map. L45, P210 and P330 are all at 45° elevation, 120° apart, which is the textbook arrangement for photometric stereo.
- With them, `shape: full` gives a height map and a depth for every dent and bump.
- Without them, the code falls back to `shape: slope-only`: dents and bumps are still told apart by shading, but there is no depth number. The fallback is tested.
- To build the 3-light-only version, delete P210/P330 from `configs/rig.pi.yaml`.

**Glare.** On shiny aluminium, a light directly next to a camera reflects straight back into it. That only affects the top pair:
- the reflection of L90 appears as a bright patch in the middle of the plate;
- the golden reference absorbs a fixed patch;
- keep exposure low enough that it never clips;
- a cheap diffuser over L90, or moving L90 about 10 cm along the arch, makes it smaller.

No other light sits at another camera's mirror angle (calc: the nearest is about 40° away).

### 3.2 Cameras and ports

The Pi 5 has **two** CSI camera connectors, so:

| View | Camera | Port | Notes |
|---|---|---|---|
| Top 90° | Raspberry Pi **AI Camera** (owned), used as a plain 12 MP camera | CAM0 | Its lens focuses from 20 cm, so mount it 20–25 cm above the plate at full resolution (about 0.065 mm/px at 20 cm, calc) |
| Grazing 15° | **HQ Camera + 16 mm C-mount lens** | CAM1 | At 15° the plate is about 97 mm deep along the line of sight (calc). Mount it about **45 cm** from the plate at f/11–16 and focus mid-plate. Expect about 0.3 mm sharpness across the plate: enough to see scratches light up, not to measure them. Sizes come from the top camera |
| Oblique 45° | **Waveshare IMX335 5MP USB (B)** | USB | Fixed-focus **175° fisheye**: mount it about 15 cm from the plate. **Check on day 1 that exposure and white balance can be set manually:** `v4l2-ctl --list-ctrls`. If not, use a C920-class webcam |

- Every image is compared only against **its own camera's** golden images, so mixing camera models is fine.
- The plate-to-grid mapping is a homography, which assumes a straight-line lens. Correct the fisheye first: print a checkerboard and use OpenCV's fisheye calibration. That needs a small undistortion step before `calibrate` (not written yet; section 17).
- On a Pi 5, address the USB camera by `/dev/v4l/by-id/...`. The CSI pipeline owns `/dev/video0..N`.

### 3.3 Lights

- **5 white 3 W LEDs** on star boards, each on a small heatsink, about 20–25 cm from the plate centre.
  - The two posts (P210, P330) are anything rigid; the angle matters more than the exact distance.
  - Zero-cost alternative: use the 5 RGB LEDs you own with all three colours on. Each colour then needs its own resistor, with the three cathodes switched by one MOSFET.
- **Wiring:** one logic-level N-MOSFET per LED (AO3400 board), gates on GPIO 17/27/22/23/24, one resistor per LED from a separate 5 V supply, common ground with the Pi.
  - At 5 V a white LED (Vf about 3.1 V) with **4.7 Ω** runs at about 0.4 A (calc).
  - Only one LED is on at a time, for about 0.3 s each.
- **Do not dim with PWM.** It beats against the rolling shutter and paints bands across the image. Switch fully on or off; set brightness with the resistor and the exposure.
- **No box.** The lights-off frame removes room light. Two rules make it work:
  1. Set exposure so **LED + room light never clips**, because clipped pixels can't be subtracted.
  2. Keep exposure a **multiple of 10 ms**, so 50 Hz mains flicker (100 Hz) averages out the same in every frame.
  
  A black cloth over the arch is an optional extra for very bright floors or direct sunlight.

### 3.4 Jig

- A matte-black base with two hard stops, an L-corner the plate is pushed into. This gives repeatable placement within about 0.5 mm.
- Software (ECC alignment on the plate edges) removes the rest.
- The plate must be **brighter than the base.** Calibration finds the plate outline automatically.
- Stick a small **grey card** in a corner of the top camera's view, off the plate. It lets you check white balance and room-light drift (optional).

## 4. Bill of materials

The list with buy links is in **[Ayas_parts.md](Ayas_parts.md)**.
- **Already owned:** Pi 5, AI HAT+ 26 TOPS, AI Camera, HQ Camera + 16 mm, IMX335 USB camera, plates, cables, ball heads, 5 V supply.
- **Still to buy:** about **₹3–5k**. That covers the white LEDs and resistors, a visible red-dye penetrant kit (≈ ₹1,175–1,593), gloves and wipes, heatsinks, and two posts.

## 5. Capture protocol (`capture.py`)

1. All cameras run with **auto-exposure, auto-white-balance and autofocus off**: fixed exposure, gain, colour gains and lens position. Anything automatic changes intensities and breaks the comparison with the golden reference.
2. For each light: LED on → wait 50 ms → fire all three cameras in parallel → drop the first 2 frames (CSI) or 4 (USB), which were exposed under the previous light → keep the next frame.
3. Last state: **all LEDs off**, the same capture. That gives the dark frame per camera.
4. Analysis uses `(frame − dark frame)`: the LED's light only.

**Set exposure** so a good plate sits at 40–60 % of full scale under its camera's own light, with nothing clipped even with the room lights on. The penetrant check uses the same top camera. Its white developer is brighter than bare aluminium, so on the rig give `penetrant` its own lower exposure if it clips (section 17).

## 6. Processing pipeline (surface inspection)

| Step | Module | What happens |
|---|---|---|
| Calibrate (once) | `calibration.py` | Sums all lights per camera into one evenly lit image, finds the plate outline, fits its 4 edges to sub-pixel accuracy and maps them onto the known plate size: one homography per camera. Manual `corners_px` fallback |
| Image maps | `align.py` | From the dark-subtracted frames: the top camera under its own light **in colour** (R, G, B); the top camera under every other light, and the 45°/grazing cameras under their own light, **as brightness**. With 5 LEDs that is 9 maps |
| Warp + align | `align.py` | Into the common grid (0.1 mm/px now, 0.05 later), anti-aliased. ECC placement correction on the top green map; LED drift cancelled by matching each map's median to the golden |
| 3D shape | `shape.py` | Photometric stereo on the top camera's images under the shape lights → slope maps `ps_nx`, `ps_ny` (or `ps_sa` in slope-only mode). At inspection: slopes minus golden slopes → **Frankot–Chellappa FFT integration → height map in mm** |
| Golden | `golden.py` | Streaming mean and spread per pixel for every map, including the slope maps, over 20+ good plates |
| Detect | `detect.py` | z-score per map; maximum over all maps; threshold; close gaps; group pieces less than 2.5 mm apart |
| Features | `features.py` | Colour contrast (top), brightness contrast per camera/light, geometry and grazing-scatter ratios, shading polarity under the 45° light, slope change, **depth from the height map**, shape |
| Classify | `classify.py` | Signature rules now; random forest once real labels exist; a Hailo CNN as the next step (section 11) |
| Judge | `rules.py` + `configs/spec.yaml` | Severity and disposition per defect, verdict per part |
| Report | `report.py` | JSON (`schemas/inspection.schema.json`, v2.0) + overlay: the natural-colour top view with boxes, next to the 3D height map |
| Explain / log | `llm.py`, `store.py` | Llama on ExecuTorch (or the template fallback); SQLite |

### 6.1 How the images tell defects apart

**Contrast** means (image − golden) / golden, averaged over the defect.

| Defect | Cameras and lights | Colour (top) | 3D map | Rule that decides |
|---|---|---|---|---|
| Stain / corrosion | The same change from every camera | **Strong shift** (red kept, blue lost) | Flat | `chroma_shift` > 0.08 and `view_spread` < 0.25 |
| Crack | **Darker from top and 45° under every light** (light trapped). The grazing camera is ignored, because crack edges scatter into it | None | Small | `brightest_view` < −0.05 and thin or branched |
| Pit | Same as crack | None | Small | `brightest_view` < −0.05 and compact |
| Dent | Shading dark towards the 45° light | None | **Pushed in, depth in mm** | `depth_mm` ≤ −0.08 (full 3D) or `polarity` > +0.3 (slope-only) |
| Bump | Shading bright towards the 45° light | None | **Raised** | `depth_mm` ≥ +0.08 or `polarity` < −0.3 |
| Scratch | **Brighter at grazing**, uniformly (scatter) | None | Tiny | `geometry_ratio` > 2, `grazing_balance` > 0.5, long and thin |
| Dust / particle | Brighter at grazing | Little | Tiny | Same scatter test, compact and < 5 mm² |

**Simulator check (sim), 20 Oct:**
- `ayas evaluate --parts 100`: 145/145 defects found and correctly typed, 0 good plates flagged.
- Dent and bump depths from the 3D map came within about 3 % of the simulated depth: −0.69 mm measured against −0.71 mm true.
- With the 3-light arch only, all types were still classified, without depth.
- Brighter, tinted room light at inspection than during the golden build still gives PASS on good plates.

**What this proves:** only that the code paths work. The simulator follows the same physics the rules assume, so real accuracy must be measured on real plates (section 14).

## 7. Red-dye penetrant check (`penetrant.py`, `ayas penetrant`)

**Why.** A tight crack in aluminium can be too fine for any camera to see directly. Penetrant inspection is the standard industrial method for cracks in non-magnetic metals (ISO 3452-1, ASTM E165/E1417). The visible red-dye version needs no UV and no dark room, so it suits this open rig. It is less sensitive than fluorescent penetrant (that is the roadmap upgrade).

**Process off the rig**, in a ventilated area away from the Pi and hot LEDs, because the aerosols are flammable. Times follow ISO 3452-1 and NASA PT-TE-1426:
1. Pre-clean with the kit's cleaner on a cloth, then let it dry 5 min. No wire brush or abrasives: they smear metal over cracks.
2. Spray penetrant and keep it wet for a **10 min** dwell (aluminium cracks).
3. Wipe off the excess: first a dry lint-free cloth, then a cloth *lightly moistened* with cleaner. **Never spray cleaner directly on the plate.** Let it dry 5–10 min.
4. Spray a thin, even coat of developer.
5. **Image at exactly 10 min of development**, because indications grow as dye bleeds out. Log the time with `--develop`.
6. Post-clean afterwards: residue causes corrosion.

Run the **RGB/white surface inspection on the clean plate first**. The developer coat changes how the plate looks.

**On the rig.** The top camera, under L90, minus the dark frame. Each colour channel is scaled so the white developer is neutral, which cancels LED colour and white balance. Then:
- **Redness** = (R − (G+B)/2) / brightness. Seeds above 0.45 grow into the faint bleed-out edge above 0.22, which ignores faint pink smears from poor removal.
- **Measure** each indication in mm: length, width, area and orientation.
- **Linear** if length > 3 × width (crack-like), otherwise **rounded** (pore-like).
- **Relevant** only if the major dimension is > 1.6 mm (1/16 in). Smaller ones are logged only.

**Acceptance profiles** (`spec.yaml → penetrant`). These are templates: no code covers plain plate, so use your customer's or drawing's limits.

| Profile | Reject if | Merge rule |
|---|---|---|
| `asme_app8` (default) | any relevant linear; rounded > 4.8 mm (3/16 in); ≥ 4 relevant rounded in a line, ≤ 1.6 mm apart | — |
| `iso23277_al1 / al2 / al3` | linear > 2 / 4 / 8 mm; rounded > 4 / 6 / 8 mm | Neighbours closer than the smaller one's major dimension count as one (ISO 23277) |

**Test defects to make:**
- drilled holes of 0.5 / 1.0 / 1.5 / 2.0 mm, for pores either side of the 1.6 mm threshold
- a row of four 2 mm holes about 1 mm apart, for the in-line rule
- jeweller's-saw slits of 3 / 6 / 10 mm, for linear indications of known length
- real cracks from bending a thin 1 mm 7075 or 2024 strip

**Real-time note.** Dwell and development (about 35–50 min per plate) are chemical process times. The AI inspection after development takes about 30 ms (sim). Say this plainly in the report.

## 8. Severity and verdict (`configs/spec.yaml`, `rules.py`)

- **Zones.** A 15 mm `sealing_face` band along the top edge with tighter limits, and `general` for the rest.
- **Ladder.** Each surface defect type uses one measure (length, diameter or area) with minor / major / critical limits.
  - Dents currently use diameter.
  - The JSON also carries `depth_mm`, so a depth limit can be added as soon as your mentor gives one.
- **Dispositions:**
  - critical → REJECT
  - major → REWORK if reworkable, otherwise REJECT
  - minor or negligible → PASS (logged)
  - dust → CLEAN_AND_REINSPECT
  - unknown or low confidence → HOLD_FOR_QA
- **Part verdict** = the worst defect.
- **The numbers are placeholders.** Replace them with real limits agreed with your mentor. Any surface crack is critical.

## 9. The JSON contract

Full schema: `schemas/inspection.schema.json` (v2.0, `mode: surface | penetrant`). One dent from a simulated run:

```json
{
  "id": "D3", "type": "dent", "confidence": 0.95,
  "location_mm": {"x": 66.3, "y": 18.9}, "zone": "general",
  "size": {"length_mm": 6.36, "width_mm": 5.39, "area_mm2": 34.29, "diameter_mm": 6.61},
  "orientation_deg": 135, "relief": "recessed", "depth_mm": -0.531,
  "severity": "critical", "disposition": "REJECT",
  "rule": "dent diameter_mm 6.61 >= 6.0 (critical, zone general)",
  "action": "Do not try to hammer it flat on the line. Set it aside for QA to decide on scrap or straightening.",
  "evidence": {
    "contrast": {"top_rgb": [-0.006, -0.006, -0.006], "top_under_45deg_light": -0.005,
                 "top_under_15deg_light": 0.0, "oblique45": -0.001, "grazing": 0.062},
    "geometry_ratio": 13.61, "color_shift": 0.0, "slope_change": 0.152, "shading_polarity": 0.81,
    "strongest": "grazing",
    "basis": "the 3D shape map shows the surface pushed in by 0.53 mm with no colour change"
  }
}
```

**At part level** the report also carries:
- `mode`, `verdict` and `verdict_instruction`
- `actions`: only the deciding defects' actions
- `rig`: the lights and the 3D mode (`full` / `slope-only`)
- `summary`, `models`, `timing_ms` per stage, and paths to frames and overlay

**Penetrant reports** use the same structure with `linear_indication` / `rounded_indication` entries (`I1`, `I2`, ...). Each entry has `relevant` and `evidence.redness` / `aspect_ratio`, and the report adds a `penetrant` block: profile, process times, in-line groups and warnings.

The LLM gets a **compact** copy (`compact_for_llm`), about 2 kB for 3 defects, because on a CPU every prompt token costs time.

## 10. The LLM part

### 10.1 Model and runtime

**Model:** Llama 3.2 **1B**-Instruct as the default; try 3B if the latency budget allows. Quantised **8da4w** (8-bit dynamic activations, 4-bit weights), with the XNNPACK backend on the Cortex-A76.

**Toolchain** (from the official ExecuTorch Raspberry Pi tutorial and the `export_llm` docs):
- **Host:** Linux x86-64 (Ubuntu 20.04+), Python 3.10–3.14, CMake 3.26+. The Apple-silicon Mac can't be the host. Use a lab PC or a cloud VM. The same machine is needed for the Hailo Dataflow Compiler.
- **Build the runner:** `examples/raspberry_pi/setup.sh pi5` produces `llama_main`. Copy it to the Pi.
- **Weights:** Llama 3.2 1B-Instruct (`consolidated.00.pth`, `params.json`, `tokenizer.model`) from Meta's Llama site or Hugging Face, after accepting the license.
- **Export:** `python -m executorch.extension.llm.export.export_llm --config llama_pi.yaml`, with:

```yaml
base:
  model_class: llama3_2
  checkpoint: path/to/consolidated.00.pth
  params: path/to/params.json
  metadata: '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
model:
  use_kv_cache: True
  use_sdpa_with_kv_cache: True
quantization:
  embedding_quantize: 4,32
  qmode: 8da4w
backend:
  xnnpack:
    enabled: True
    extended_ops: True
```

**Also raise the export's maximum sequence/context length to ≥ 2048.** The default is small, and our prompt plus answer is about 1,500 tokens. Check the field names in `LlmConfig` for your ExecuTorch version.

**Run on the Pi:** `./llama_main --model_path llama3_2.pte --tokenizer_path tokenizer.model --seq_len 2048 --prompt "..."`. Ayas calls this binary through `ExecuTorchBackend` (`configs/rig.pi.yaml → llm:`).

### 10.2 Prompt design (`llm.py`)

- **System prompt** with 4 rules:
  1. The verdict is final.
  2. Use only facts from the report and SOP.
  3. How to read the evidence: contrast per camera/light, colour, 3D depth (surface mode); linear/rounded, relevance and profile (penetrant mode).
  4. Short numbered steps.
- **First user turn:** the compact JSON + SOP notes for what was found (action, likely causes, prevention; for penetrant, the acceptance profile).
- **Follow-ups:** the report turn always stays in context. Only the last 3 Q&A pairs are kept, which bounds prompt length.
- **Llama 3 chat template.** The runner adds BOS itself.

**Guardrails**
- Pass/fail and all numbers come from the rules engine and the JSON.
- If the runner fails or times out, the station falls back to a **template answer** built from the same JSON and says so. The line never stops because of the LLM.

**Example follow-ups the demo should show**
- "Why is D1 a scratch and not a crack?" → evidence (bright at grazing, no colour change, not dark under every light)
- "How deep is the dent?" → `depth_mm` from the 3D map
- "Why is I1 rejected but I2 accepted?" (penetrant) → linear vs rounded, the relevance threshold and the profile's limits
- "What caused it / how do we prevent it?" → SOP causes and prevention
- "Can I just pass it?" → "The verdict is final; call the QA supervisor."
- "Where exactly is it, and how long is it?" → location, size, zone

**Latency budget (est., to measure).** About 1,200 prompt tokens + ~150 answer tokens. Prompt processing dominates on a CPU. If time-to-first-token is too long:
- send only the deciding defects;
- shorten the SOP notes;
- move to a persistent runner (C++ `TextLLMRunner`) that keeps the model loaded and reuses the KV cache of the system prompt.

## 11. The two pipelines, kept separate (the brief requires this)

```
 Pipeline A: vision (HailoRT on the Hailo-8 NPU)            Pipeline B: reasoning (ExecuTorch on the Arm CPU)
 ───────────────────────────────────────────────            ───────────────────────────────────────────────────
 3 cameras x (5 lights + dark) -> dark subtraction (CPU)     inspection JSON (compact) + SOP notes
   -> warp/align (OpenCV, CPU) -> 3D slope/height (NumPy)      -> Llama 3 chat prompt
   -> golden z-score detection (NumPy, CPU)                    -> llama_main + Llama 3.2 1B .pte (XNNPACK, 8da4w)
   -> region crops -> CNN classifier          [Hailo-8]        -> explanation + follow-up answers
   -> features + signature rules / random forest (CPU)         (never changes the verdict)
   -> rules engine (CPU) -> inspection JSON  ──────────────>
 penetrant: top frame -> redness -> indications -> profile rules -> JSON ──>
```

| Block | Silicon | Runtime | Status (20 Oct) |
|---|---|---|---|
| Capture, dark subtraction, warp, align, golden, detect, features | Cortex-A76 | Python, OpenCV, NumPy | written, tested on the simulator |
| 3D shape (photometric stereo + FFT integration) | Cortex-A76 | NumPy | written, tested on the simulator |
| Red-dye penetrant indications + profiles | Cortex-A76 | OpenCV, NumPy | written, tested on the simulator |
| Signature rules / random forest | Cortex-A76 | scikit-learn | written, tested on the simulator |
| CNN region classifier (defect type; penetrant true/false indication) | **Hailo-8** | **HailoRT** | to do: needs real crops and the x86 host for the Dataflow Compiler |
| Llama 3.2 1B advisor | Cortex-A76 | **ExecuTorch** (`llama_main`) | wrapper written and tested with a fake runner; real model pending export |

**Hailo for 23 Nov.** Using the Hailo is optional in the brief. If the CNN is used, it must be documented as its own pipeline, as above.
- The submission is complete with the CPU vision pipeline plus Llama on ExecuTorch.
- If there is time, train a small crop classifier (ResNet-18/MobileNet class) on real crops and compile it with the Dataflow Compiler on an x86 machine. Run it through HailoRT, keeping the signature rules as a cross-check.
- Pretrain on NEU / KolektorSDD crops, then fine-tune on your own plates.

## 12. Making the dataset

Use the 10 aluminium plates now; get about 30 more cut by a local fabricator.

**Good parts.** 20 clean plates make the golden reference. Re-place one plate 10× to measure placement repeatability.

**Make defects on purpose,** aiming for 20+ examples per type over time:
- **Scratch:** scriber or needle along a ruler. Vary the length and the **angle (0/45/90/135°)**.
- **Dent:** centre punch or a ball bearing tapped from the front. Measure the depth with a dial gauge to check the 3D map.
- **Bump:** punch from the back.
- **Pit:** 0.5–1.5 mm drill, shallow.
- **Stain:** salt-water drop left overnight (aluminium corrosion), oil fingerprint, marker.
- **Dust:** metal filings sprinkled on.
- **Crack:** hard to make in 3 mm plate. Use bent thin 7075/2024 strip, or a broken casting. Then the penetrant check (section 7).

**Label.** `ayas dataset --out data/rig/regions.csv` writes one row per region. Fill in `label` using the overlays, then run `ayas train`.

**Hold out** about 25 % of plates (whole plates) for the final numbers.

## 13. Bring-up checklist (rig)

1. Bring up each camera on its own: `rpicam-hello --list-cameras` (CSI); `v4l2-ctl --list-ctrls` (USB). **Confirm manual exposure and white balance on the IMX335.**
2. Fisheye calibration of the IMX335 (checkerboard) before anything else (section 17).
3. Set exposure, in multiples of 10 ms: a good plate at 40–60 % under its own light, **nothing clipped with the room lights on**. Lock the colour gains on a grey card.
4. Focus. Top and 45°: sharp at the plate centre. Grazing: mid-plate at f/11–16; check the near and far edges.
5. **Dark-frame check:** all LEDs off → the image is just room light. Then with an LED on, `(on − dark)` must not change when you switch the room lights on and off.
6. Run `ayas calibrate`. Check the printed corners. If a view fails, put `corners_px` in the config.
7. Run `ayas golden --count 20`. It prints the 3D mode: `full` with P210/P330 fitted.
8. Inspect 10 good plates. Lower `z_threshold` until just before good plates get flagged.
9. Inspect the defect plates; collect a dataset; train.
10. Penetrant: develop one plate with a known slit, run `ayas penetrant`, and tune `redness_high`/`redness_low` until the slit is found and clean developer is not.
11. Export Llama, set `llm:` and time the first answer.

## 14. Metrics for the technical report

| Area | What to report |
|---|---|
| Detection | Recall and precision per type on held-out plates; false-flag rate on good plates |
| Classification | Confusion matrix: signature rules vs random forest (vs Hailo CNN if done) |
| 3D | Dent/bump depth against a dial gauge (n ≥ 10); `full` vs `slope-only` |
| Measurement | Length/diameter error against calipers (n ≥ 20) |
| Room light | The same plates with room lights off / on / a lamp moved: verdicts must not change |
| Penetrant | Known holes and slits found and sized (indication vs true size); false calls on clean developer |
| Repeatability | The same plate inspected 10× gives the same verdict |
| Latency | Per stage (capture, align, 3D, detect, classify, rules); LLM time-to-first-token and tokens/s; end-to-end |
| Resources | RAM, CPU %, NPU use; power at idle, during inspection and during LLM (USB-C meter) |
| LLM quality | 30 scripted operator questions scored by the mentor; answers contradicting the verdict (target 0) |

## 15. Timeline to the Open Qualifying submission (Mon 23 Nov 2026)

| When | What |
|---|---|
| 20 – 25 Oct | Order the white LEDs, resistors, red-dye kit and gloves. Build the base, stops, arch and the two LED posts; wire the LEDs. Bring up the three cameras (IMX335 manual-exposure check). **x86 Linux host: build `llama_main`, export Llama 3.2 1B, run it on the Pi with a sample JSON, measure tok/s** |
| 26 Oct – 1 Nov | IMX335 fisheye undistortion; calibrate; lock exposures; golden from 20 plates; first real inspections; tune thresholds |
| 2 – 8 Nov | Make defect plates; first penetrant runs (holes, slits, bent strip); tune the redness thresholds; collect and label a dataset (Diwali 8 Nov) |
| 9 – 15 Nov | Train the random forest on real labels; measure the section 14 metrics; touchscreen app (verdict, overlay with 3D map, chat). Optional: Hailo CNN |
| 16 – 20 Nov | Freeze. Demo video: live surface inspection with the room lights on, 3D dent depth, penetrant crack reject, Llama explaining and answering. Technical report with both pipelines documented |
| **by Mon 23 Nov** | **Submit:** report, source code, demo |

## 16. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Room light clips the sensor or changes between frames | Exposure set with room lights on; multiples of 10 ms; dark frame per part; black cloth for very bright floors |
| Glare of L90 on shiny aluminium | Low exposure, diffuser, or move L90 about 10 cm along the arch; golden absorbs a fixed patch |
| Photometric stereo assumes a matte-ish surface | Mill/brushed aluminium is semi-matte. Report depths against a dial gauge; very glossy parts are a known limit |
| IMX335 has no manual exposure/WB | Check day 1; swap to a C920-class webcam |
| Grazing view soft at near/far edges | f/11–16 at about 45 cm; that view gives scatter cues, not measurements |
| Scratches parallel to the arch scatter weakly | Orientation test (section 14); the off-arch lights also help; re-inspect rotated 90° if needed |
| Real defects don't match the rule table | Random forest on real labels; `unknown` → HOLD_FOR_QA |
| Penetrant false calls (smears, edges, lint) | Two-level redness threshold, relevance size, wipe-off check for doubtful marks; Hailo true/false classifier later |
| LLM too slow on CPU | 1B model, compact JSON, bounded history; template fallback |
| LLM says something wrong | It never sets the verdict; prompt restricted to report + SOP; scripted question test |

## 17. State of the code (20 Oct 2026)

**Written and tested on the simulator** (50 tests, `pytest -q`):
- capture with one light at a time plus a dark frame; room-light removal
- calibration, warp and alignment, golden reference, detection
- 3D shape (full and slope-only), height map, depth per defect
- features, signature rules, random forest
- red-dye penetrant indications, linear/rounded, ASME / ISO profiles, in-line rule
- rules engine, JSON + schema v2.0, overlays, SQLite
- Llama prompt and ExecuTorch runner wrapper (fake runner), template fallback
- CLI: `calibrate`, `golden`, `inspect`, `penetrant`, `chat`, `dataset`, `train`, `evaluate`, `history`

**Written but not yet run on hardware:**
- Picamera2 and USB camera drivers
- GPIO LED switching
- the Pi config (camera indices, USB device path and exposures are placeholders)

**Not written yet:**
- fisheye undistortion for the IMX335
- a separate exposure for the penetrant shot
- the Hailo CNN
- the touchscreen app
- an optional buzzer / stack light

## 18. Roadmap (not for 23 Nov)

- **Fluorescent penetrant (FPI):** a 365 nm UV-A source, a GG-495 filter on the top camera and a black hood for the UV shot. Higher sensitivity than red dye; full notes in [Ayas_FPI.md](Ayas_FPI.md).
- **Industrial version:**
  - global-shutter cameras with strobed lights, for moving parts on a conveyor
  - a shroud
  - PLC output and a reject mechanism
  - per-product "teach mode" from about 20 good parts
