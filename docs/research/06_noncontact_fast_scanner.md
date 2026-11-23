# Non-contact, fast, handheld scanner: design and honest limits (21 Oct 2026)

Status: PROPOSAL, awaiting the user's go-ahead.

**User's requirements (21 Oct):**
- **no contact with the material at all**
- **fast scanning**
- handheld, room light, micrometre-level detail

This supersedes the touch-measure proposal in `05_...`. Sources for the facts used are in `01`–`03`. **(calc)** = my own arithmetic, to be confirmed in the bench test (§6).

## 1. The honest trade-off

**Without contact you can *see* µm-scale defects, but you can't *measure* them to 1 µm.**
- At µm resolution the depth of field is about 0.1 mm, while a hovering hand moves 0.05–0.1 mm (tremor) and drifts by millimetres.
- Every handheld tool that measures in µm touches the part (file 03).
- Non-contact handheld 3D scanners reach only 12–25 µm at best (file 01).

**The best honest non-contact handheld target (calc):**
- **Lateral:** about **20 µm per pixel**, about 18–26 µm real resolution.
- **Depth:** about **10 µm** precision from a laser-line profile, limited by speckle.
- **Detection:** scratches and particles **a few µm wide**, made visible by dark-field scatter. This is detection only; their width isn't measured.

## 2. How it is fast and works in room light (same principles as commercial handheld scanners)

| Principle | How it's used | Number (calc) |
|---|---|---|
| **Global-shutter camera + LED strobe** | Each frame is one 50–150 µs flash: no motion blur, and room light barely registers | ≤1 px blur at 20 µm/px needs ≤133 µs at 150 mm/s |
| **Bright light up close** | Overdriven LED pulses at 6–11 cm; 2–3 LEDs per light | Each 260 lm LED gives about 13,000 lx at 8 cm at nominal current, about 5× that in pulses. Room light (500 lx) is about 1–2 % of the frame |
| **Single-shot frames, cycled** | 60 fps, cycling **bright-field → dark-field → laser line**, so each mode runs at 20 fps | Sweep up to about 300–400 mm/s with no gaps; a 100×100 mm plate in about 5–10 s |
| **Large depth of field instead of autofocus** | Fixed focus at f/4–5.6. A ToF distance sensor beeps when the scanner is in range; blurred frames are dropped automatically | DOF about ±1.3 mm (f/4) to ±2.3 mm (f/5.6) |
| **Laser line + (optional) bandpass** | The blue line is far brighter than room light in its own band | About 200× room light (radiometric), about 2,000× with a 450 nm bandpass |

## 3. The device: "Ayas-Scan"

| Part | Choice | Why |
|---|---|---|
| Camera | **Raspberry Pi Global Shutter Camera** (IMX296 colour, 1456×1088, 3.45 µm pixels, 60 fps, about 30 µs minimum exposure, external trigger) + **16 mm C-mount lens** at about 11 cm (or a 12 mm at about 8 cm), f/4–5.6 | No motion blur; software-controlled exposure and gain; trigger for strobe sync |
| Field of view | about **29 × 22 mm** at **20 µm/px** | Wide enough to sweep a plate in 4 passes |
| Timing | **Raspberry Pi Pico**: triggers the camera (XTR pin, 1.8 V) and fires the LED / laser pulses in the same µs window | Precise strobes; frames tagged by mode |
| Bright-field light | Diffused white LED ring (pulsed) | Overview, colour (stains, corrosion), and motion tracking between frames |
| Dark-field light | **The user's 12 V white strips** as a low-angle ring (about 10–15°), pulsed through MOSFETs | Scratches, pits, particles and crack edges light up: µm-wide detection |
| Laser line | Blue (about 450 nm) **Class 2** (≤1 mW) line laser at about 30–45° to the camera | Depth profile across the line in one shot: about 10 µm depth precision; hand motion doesn't matter within one frame |
| Distance guide | VL53L1X ToF sensor (or a sharpness score) | Beep / LED when within ±2 mm; frames outside are discarded |
| Optional, stretch | RGB LEDs from 3 directions (the user's SME 3 W) for **single-shot colour photometric stereo** | Relative 3D slope map per frame. Risky on shiny metal; brushed/mill finish is OK |
| Compute | Pi 5 + **Hailo-8** (detector / segmenter on bright-field and dark-field frames, about 20 ms) + Pi CPU (laser-line extraction, motion tracking by phase correlation, mosaic, metrology) | Live detection while sweeping |
| Advisor | **Llama 3.2 1B via ExecuTorch** on the CPU | Explains the scan results and answers questions; never changes the verdict |
| UI and alerts | HDMI display (both MIPI ports free except one camera, so DSI is also possible), buzzer, RGB LED; SQLite log + report | Brief bullets |

**Scan flow:**
1. Hold about 11 cm above the part (the beep confirms range) and sweep at walking-hand speed.
2. Live view: Hailo boxes on detected defects; dark-field highlights; the laser profile shows height.
3. After the sweep: a stitched mosaic with defect positions. Each defect gets length (mm), width class (dark-field brightness and width), depth (µm, from laser profiles crossing it) and severity (`rules.py`), then goes into the JSON and on to Llama.

## 4. Achievable numbers (calc, to verify in the bench test)

| Quantity | Value | Basis |
|---|---|---|
| Sampling | 20 µm/px over 29 × 22 mm | Lens geometry for the 1456 px sensor |
| Resolution | 18 µm (f/4) to 26 µm (f/5.6) | Rayleigh, NA ≈ m/(2N(1+m)) |
| Depth of field | ±1.3 to ±2.3 mm | λ/NA² + pixel term |
| Depth precision (laser line) | about 8–11 µm | Speckle σz ≈ λ/(2π·NA·sinθ), λ = 0.45 µm, θ = 30° |
| Smallest scratch *seen* | a few µm wide (dark-field) | Literature in file 03 §3.4 (detection, not measurement) |
| Scan speed | 150–300 mm/s sweep; 100×100 mm plate in about 5–10 s | 20 fps per mode, about 22 mm frame height |
| Room light share | about 1–2 % (LEDs), much less for the laser | Strobe intensity vs 500 lx |

## 5. Risks

| Risk | Mitigation |
|---|---|
| **Light budget at 50–150 µs** | Overdriven pulses (≤5× for short pulses, ≤1 % duty); f/4; gain ×4–8; 2–3 LEDs per light |
| Shiny (mirror) metal | Dark-field and the laser work best on brushed/mill finish. Mirror-polished parts give weak laser returns: state this as a limit |
| Motion tracking on plain metal | Bright-field texture (brushed lines) with phase correlation; fallback to per-frame detection without a mosaic |
| Photometric stereo on metal | Stretch goal only |
| GS camera trigger mod | Follow the Raspberry Pi docs (XTR pin, resistor change); fallback: free-running camera with a longer strobe |
| Laser safety | Class 2 only; never point at eyes; label it |

## 6. Bench test before buying everything (1–2 days)

Parts: Pi GS camera + lens + Pico + 2 LEDs + MOSFET, and an aluminium plate with scribed scratches of known width (measured under a microscope).

| Test | Pass criterion |
|---|---|
| 1. Strobe exposure | A 100 µs pulse at f/5.6 gives a usable image (plate at ≥30 % of full scale at gain ≤8) |
| 2. Sweep blur | Sharp frames at a 150 mm/s hand sweep |
| 3. Dark-field | Visibility of the smallest scribed scratch |
| 4. Laser line | Depth noise on a flat plate ≤10 µm σ; a 50 µm shim step reads within ±10 µm |

If 1 fails, slow the sweep or add LEDs. If 4 fails, depth becomes relative only.

## 7. Scale-up

1. **Same head on a robot or linear stage:** automated, repeatable scans.
2. **Several heads side by side, or a line-scan camera:** for conveyors at m/s.
3. **Fixtured micro-head:** for single-µm verification where needed. This is the only route to true µm measurement.
4. **Android + USB-C head version:** for field use.
