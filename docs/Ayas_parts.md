# Ayas parts list (with buy links)

## Current build (20 Oct 2026): 3 cameras × 3 white lights + 3D map + red-dye penetrant

This replaces the lists further down (RGB lights, FPI/UV), which are kept for reference only. Design: [Ayas_plan.md](Ayas_plan.md).

**Already on your prototype list (bought or ordered):**
- Raspberry Pi 5 8 GB, AI HAT+ 26 TOPS
- Raspberry Pi AI Camera (top), HQ Camera + 16 mm lens (grazing), Waveshare IMX335 5MP USB Camera (B) (45°)
- 2 × 500 mm Pi 5 camera cables, 3 mini ball heads
- 5 V 3 A adapter (LED supply), 3 × 10 Ω 5 W resistors, AO3400 MOSFET board
- SME 3 W RGB LEDs (5 pcs), aluminium plates 100 × 100 × 3 mm (10 pcs), automatic centre punch
- UV torch: not needed for this build; keep it for the FPI roadmap

**Still to buy** (≈ ₹3,000–5,000; \* = unverified price, check before paying):

| # | Part | Qty | Why | Price | Where |
|---|---|---|---|---|---|
| 1 | 3 W white LED on 20 mm star board (neutral or cool white) | 5 + 2 spare | L90, L45, L15 on the arch; P210, P330 on posts. *Zero-cost option:* use the 5 RGB LEDs you have with all three colours on (3 resistors per LED) | ~₹40–80 each\* | Robu / local electronics shop |
| 2 | 4.7 Ω 5 W resistor | 5 | One per white LED: about 0.4 A at 5 V (calc) | ~₹17 each\* | Robu (same range as the 10 Ω you bought) |
| 3 | Small heatsink for 20 mm star LED (or cut the 100 mm bar) | 5 | Keeps the LEDs cool during setup | ~₹20–50 each\* | Robu / local |
| 4 | **Visible red-dye penetrant kit** (cleaner + red penetrant + white developer, aerosols) | 1 | The crack check. Solvent-removable, no UV, works in room light | ₹1,175, ships in about 7 days | [Industrybuying (WESAF DPT kit)](https://www.industrybuying.com/lubricants-wesaf-FAM197650); alternative [Molygraph CDK ₹1,593 (Moglix)](https://www.moglix.com/molygraph-cdk-crack-detection-kit/mp/msn2qgpco3r5zl) |
| 5 | Nitrile gloves + lint-free wipes + safety glasses | 1 set | Penetrant handling; wiping off excess with a cloth | ~₹500\* | local / pharmacy |
| 6 | Matte-black base board + 2 hard stops (L-corner) + 2 short posts for P210/P330 | — | The jig and the off-arch lights | ~₹300–600\* | local hardware shop / 3D print |
| 7 | Drill bits 0.5–2 mm + jeweller's saw | 1 set | Known-size pores and slits for the penetrant test plates | ~₹500\* | local hardware |
| 8 | *Optional:* 18 % grey card | 1 | White-balance and room-light check in the corner of the top view | ~₹300–500\* | Amazon / camera shop |
| 9 | *Optional:* 1 mm 7075-T6 or 2024 aluminium strip | small piece | Bends until it cracks: real cracks for the penetrant demo | ~₹489/kg\* | IndiaMART metal sellers |
| 10 | *If you don't have them:* official 27 W USB-C PSU + Active Cooler | 1 each | AI HAT+ plus three cameras; sustained CPU load from the LLM | ₹1,255 + ₹519 | [Robu PSU](https://robu.in/product/official-27w-usb-c-pd-power-supply-for-raspberry-pi-5-black/), [Robu cooler](https://robu.in/product/official-raspberry-pi-5-active-cooler/) |
| 11 | *Fallback only:* Logitech C920 | 1 | Only if the IMX335 can't lock exposure / white balance (`v4l2-ctl --list-ctrls`) | ₹7,995 | [Amazon](https://www.amazon.in/dp/B006JH8T3S) |

**Check your AO3400 board has 5 channels** (one per LED). If not, add a second board.

**Free:** print a checkerboard for the IMX335 fisheye calibration.

---

## Earlier lists (14–19 Oct), kept for reference

Prices were checked on the sellers' pages on **14 Oct 2026** and include GST; shipping is extra. Prices and stock change daily, so check before you pay.

**Already owned (then):** Raspberry Pi 5 8 GB, Raspberry Pi AI Camera, 64 GB microSD, AO3400 MOSFET board.

## Wave 1: cheap smoke test (~₹2,100)

Buy these first. They let you check that the optics work, using the AI Camera you already own moved by hand to 90°, 45° and 15°, before spending on the other cameras (see "Will it work?" in the chat answer).

| # | Part | Why | Price | Link |
|---|---|---|---|---|
| 1 | SME 3 W RGB LED, 6-pin, 5 pcs | The light: red, green and blue in one package | ₹269 | [Amazon](https://www.amazon.in/dp/B09G1X9XCT) |
| 2 | Aluminium heatsink for high-power LED, 100×40×8 mm | LED cooling (stick the LED on with thermal adhesive) | ₹299 | [Robu](https://robu.in/product/aluminium-heat-sink-for-high-power-led-amplifier-100-x-40-x-8-mm/) |
| 3 | 10 Ω 5 W wire-wound resistor × 3 | One per colour: about 280 mA red, 170 mA green/blue from 5 V | ₹17 each | [Robu](https://robu.in/product/10-ohm-5-watt-wire-wound-resistor/) |
| 4 | 5 V 3 A adapter, 5.5 mm DC plug | Separate supply for the LEDs (common ground with the Pi). Add a DC female-to-screw-terminal adapter from any local shop | ₹239 | [Robu](https://robu.in/product/orange-5v-3a-power-supply-adapter-charger-with-5-5mm-dc-plug/) |
| 5 | Milky-white acrylic, A4, 3 mm | Diffuser in front of the LED | ₹250 | [Amazon](https://www.amazon.in/dp/B0F7GQPTR6) |
| 6 | Aluminium plate 100×100×3 mm, 5 pcs | First test parts. Set `plate_w_mm`/`plate_h_mm` to 100 in the config | ₹750 | [Amazon](https://www.amazon.in/dp/B09PKCMPBJ) |
| 7 | Automatic centre punch | Making dents and bumps | ₹284 | [Amazon](https://www.amazon.in/dp/B0CG8YJZND) |

## Wave 2: the full rig

| # | Part | Why | Price | Link |
|---|---|---|---|---|
| 8 | **Arducam HQ Camera (IMX477)**, Pi 5 compatible | Grazing camera. Comes with C-CS adapter, tripod mount and Pi 5 cable. Needs `dtoverlay=imx477` (see `configs/rig.pi.yaml`) | ₹9,324 | [Amazon](https://www.amazon.in/dp/B0D3WYQF2Q) |
| | *or* Official Raspberry Pi HQ Camera | Cheaper, but **out of stock** at Robu today, so click "notify me" | ₹6,493 | [Robu](https://robu.in/product/raspberry-pi-hq-camera/) |
| 9 | Official Raspberry Pi 16 mm telephoto lens (C-mount, f/1.4–16) | Lens for the grazing camera; stop down to f/11–16 for depth of field | ₹5,979 | [Robu](https://robu.in/product/16mm-telephoto-lens-for-raspberry-pi-high-quality-camera/) |
| 10 | Logitech C920 HD Pro | 45° camera (USB) with manual exposure, focus and white balance | ₹7,995 | [Amazon](https://www.amazon.in/dp/B006JH8T3S) |
| 11 | Pi 5 camera FPC cable 500 mm × 2 | Reach for the top (20–25 cm high) and grazing cameras | ₹372 each | [Robu](https://robu.in/product/official-raspberry-pi-5-camera-fpc-cable-500mm/) |
| 12 | Mini ball head, 1/4" × 3 | Camera mounts (HQ and C920 have 1/4" threads; 3D-print a small plate for the AI Camera) | ₹148 each | [Amazon](https://www.amazon.in/dp/B0CR3K3527) |
| 13 | Black EVA foam A4 5 mm, 5 sheets (buy 2 packs) | Matte-black lining for the dark box (box: cardboard/MDF from a local shop) | ₹249 per pack | [Amazon](https://www.amazon.in/dp/B0D2NJD4DK) |
| 14 | USB-C power meter | Power numbers for the technical report | ₹920 | [Amazon](https://www.amazon.in/dp/B0GR3GMC4Z) |
| 15 | Official 27 W USB-C PSU (skip if owned) | Needed for the HAT, 3 cameras and USB current | ₹1,255 | [Robu](https://robu.in/product/official-27w-usb-c-pd-power-supply-for-raspberry-pi-5-black/) |
| 16 | Official Active Cooler (skip if owned) | Sustained CPU load from the LLM | ₹519 | [Robu](https://robu.in/product/official-raspberry-pi-5-active-cooler/) |

## Wave 3: NPU, needed for Phase 2 (December)

| # | Part | Why | Price | Link |
|---|---|---|---|---|
| 17 | **Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L)**, in stock | Runs the Phase-2 defect CNN through HailoRT. A few crops per part is a light load, so 13 TOPS is plenty, and the software is the same as the 26 TOPS version | ₹7,539 | [Robu](https://robu.in/product/raspberry-pi-ai-hat-13-tops/) |
| | *or* AI HAT+ 26 TOPS (Hailo-8) | More headroom. **In stock at Probots Bengaluru** (store pickup possible); sold out at Robu (₹12,099) and Robocraze (₹11,309) | ₹14,999 | [Probots](https://probots.co.in/official-raspberry-pi-ai-hat-26-tops.html) |

## Totals (approximate)

| Option | Total |
|---|---|
| Waves 1–3 with Arducam HQ + 13 TOPS HAT (PSU and cooler included) | **≈ ₹37,000** |
| Same with the 26 TOPS HAT | ≈ ₹44,500 |
| Budget: Camera Module 3 ([Robu, ₹3,249](https://robu.in/product/raspberry-pi-camera-module-3/)) instead of HQ + 16 mm lens at grazing | ≈ ₹25,000 |

**About the budget option:** Camera Module 3 has a fixed f/1.8 aperture, so put it 40–45 cm away to keep the whole plate in focus at 15°. You get about 3× fewer pixels on the plate than with HQ + 16 mm. That's fine for brightness cues, but weaker for very fine scratches.

**Also needed (you probably have these, or buy locally):**
- jumper wires and a small perfboard or terminal block
- thermal adhesive for the LED
- a scriber or needle for making scratches
- a 0.5–1.5 mm drill for pits
- salt water for rust
- M2/M3 screws

## Added 19 Oct 2026: items the plan uses but the list above missed

| # | Part | Why | Price | Link |
|---|---|---|---|---|
| 18 | Waveshare 7" capacitive touch LCD (C), 1024×600, HDMI + USB touch | Touchscreen app (December). Must be HDMI: both CSI/DSI ports are taken by the AI Camera and the HQ Camera. **Pre-order, 2–4 weeks**, so order by early November | ₹4,800 | [ThinkRobotics](https://thinkrobotics.com/products/7inch-capacitive-touch-screen-lcd-c-1024-600-hdmi-ips) |
| 19 | Lever microswitch | Lid switch for the dark box (plan §3.3) | ~₹30–50 (est.) | local / Robu |
| 20 | 5 V active buzzer + 3 LEDs (green/amber/red) + resistors | Operator alert (the brief's "alert operators"); a stack light is optional | ~₹50–100 (est.) | local / Robu |
| 21 | Box material: cardboard or 6 mm MDF, ~30×30×30 cm | The dark enclosure (lined with item 13) | ~₹200–500 (est.) | local shop |

**Stock check on 19 Oct:** the AI HAT+ 26 TOPS is in stock at Probots for ₹14,999 (the store reopens and dispatches from 21 Oct). Robu and Amazon pages couldn't be fetched, so recheck those prices and stock in a browser before paying.

## Added 19 Oct 2026: FPI (fluorescent penetrant) module

Full list, process and reasoning are in [Ayas_FPI.md](Ayas_FPI.md) §2. In short, about ₹11,000:
- **Penetrant kit:** ZChek fluorescent kit (2 penetrant, 3 cleaner, 2 developer), about ₹3,540 incl. GST.
- **Backup:** visible red-dye kit, ₹1,175.
- **UV source:** a 365 nm filtered torch (₹439) and 2 × 365 nm 3 W LEDs (₹598).
- **Camera filter:** SCHOTT GG-495 long-pass, ₹3,734.
- **Safety and consumables:** UV glasses ₹500, a black cloth hood, gloves and wipes.
- **Optional:** a UV meter, ₹3,900.

**More plates:** ~30 more plates, cut to the same size, from a local fabricator (mild steel + aluminium) are much cheaper per piece than Amazon.
