# Imaging and lighting design: one Raspberry Pi AI Camera (IMX500) looking straight down at a 100x100x3 mm aluminium plate, using the blue 12 V LED modules, 395 nm LEDs and DPT kit the team owns

SUMMARY: The AI Camera uses a fixed 4.74 mm F1.79 lens, with 66.3 x 52.3 deg field of view and 1.55 um pixels. Focus is set by hand with the included lens tool, and the rated range is 20 cm to infinity. At 20 cm, full resolution gives 0.064 mm/px (calc). The field is 259 x 194 mm and the plate spans about 1566 px (calc). Depth of field is about +/-5 to 10 mm (calc), so a flat 3 mm plate is easy. Mount the lens 20 cm above the plate first, and only try 15 cm (0.048 mm/px) after checking on the bench that the lens will turn that far without coming loose.

In Picamera2, run full resolution (10 fps, 73 ms rolling-shutter readout per the driver, calc) with AE and AWB off and fixed ExposureTime/AnalogueGain/ColourGains. Use ExposureTime of 10, 20 or 30 ms so the 100 Hz flicker from Indian mains lighting cancels. Capture each light state with capture_request(flush=True), which guarantees the exposure started after the LED switched. Use the uncompressed 16-bit raw stream for linear maths (dark subtraction, photometric stereo).

The IMX500 NPU attaches tensor metadata to the same frames, stills included. It is best used as a cheap 'part present and seated' gate or a coarse OK/NG triage, while the main defect model runs on the Hailo-10H through HailoRT and Llama runs on the CPU through ExecuTorch. Treat it as optional and document it as its own pipeline. Custom models need Edge-MDT/MCT plus imxconv on Linux with Python 3.10 to 3.12 and Java 17, then imx500-package on the Pi.

Lighting design: 12 of the blue modules form four low-angle dark-field banks (N/E/S/W, about 15 deg). Four more, one per side, form a 40 deg photometric-stereo tier. Each group is switched by its own logic-level MOSFET (AO3400A, or a D4184 module) from a 12 V 2 A adapter, with DC drive only and no PWM. Adding four images in software gives the 'all-on' ring image, so only one bank ever needs to be lit (at most 0.4 A).

Blue light suits fine scratches (more micro-roughness scatter, calc 1.8x vs red) and makes red dye on white developer look dark. But on a Bayer sensor blue light samples mainly a quarter of the pixel sites, and under blue a red indication looks the same as a dark stain. So add about four white 12 V modules for true colour and a redness ratio.

The 395 nm LEDs are for checking oil/grease residue, testing whether the Wesaf red penetrant fluoresces, and cleaning verification. Bare aluminium should stay dark. Wire them 2 per string with 270 ohm from 12 V. They need UV400 glasses and the hood, because aluminium is a near-mirror for UV.

Room light is handled by a black hood plus a dark frame (all LEDs off) at the same exposure, subtracted in raw. The full sequence (dark, 4 DF, 4 PS, white, UV, dark) takes about 2.4 s at full resolution, or about 0.8 s binned (calc). Extra spend is roughly Rs 3,000 to 5,000.

## Findings
- [high] AI Camera specification: Sony IMX500, 12.3 MP, 4056x3040, 1.55 um pixels, 7.857 mm (type 1/2.3) sensor, integrated IR cut filter. Focus is 'Manual adjustable focus', range 20 cm to infinity. Focal length 4.74 mm, F1.79, HFOV 66.3 +/-3 deg, VFOV 52.3 +/-3 deg. Modes: 2028x1520 10-bit 30 fps (2x2 binned) and 4056x3040 10-bit 10 fps. Maximum input tensor 640x640, int8/uint8. 8,388,480 bytes for firmware, weights and working memory. Operating range 0-50 C; in production until at least Jan 2028; $70. (https://pip-assets.raspberrypi.com/categories/1044-raspberry-pi-ai-camera/documents/RP-008147-DS-1-ai-camera-product-brief.pdf)
- [high] The camera documentation lists the AI Camera image area as 6.287 x 4.712 mm, focus 'Adjustable', and focus range 'Approx 20 cm to ∞'. (https://www.raspberrypi.com/documentation/accessories/camera.html)
- [medium] Diagonal FOV is 78.3 +/-3 deg. The retail package includes 'Lens adjustment tool x1', used to turn the lens for manual focus. (https://waveshare.com/raspberry-pi-ai-camera.htm)
- [high] mm per pixel at 4056x3040 (thin lens, lens-to-plate distance d):
- 15 cm: 0.0475 mm/px, field 193x144 mm, plate spans about 2105 px.
- 20 cm: 0.0639 mm/px, field 259x194 mm, plate about 1566 px.
- 25 cm: 0.0802 mm/px, field 325x244 mm, plate about 1247 px.

These agree with the 66.3 deg FOV (261 mm wide at 20 cm). Under single-colour light, one Bayer colour plane samples at twice this pitch: 0.095, 0.128 and 0.160 mm. (calc)
- [medium] Depth of field at F1.79, f = 4.74 mm, total, for a circle of confusion of 1 px (1.55 um) and 2 px (3.1 um):
- 15 cm: 5.4 mm and 10.8 mm
- 20 cm: 9.7 mm and 19.3 mm
- 25 cm: 15.2 mm and 30.4 mm

The Airy disk at 465 nm is about 2.0 um, close to the pixel pitch, so it is not limiting. Real lens sharpness (MTF) is unpublished (UNVERIFIED); expect about 2 px effective resolution. (calc)
- [low] No source documents AI Camera focus closer than 20 cm. For other Pi camera modules, forum users report unscrewing the lens to focus closer, with the risk that it comes out of the thread. Whether the AI Camera lens can be turned to about 15 cm is UNVERIFIED. (https://forums.raspberrypi.com/viewtopic.php?t=365590)
- [medium] The Raspberry Pi blog suggests adding +2D/+4D/+10D close-up lenses in front of Pi camera modules for macro work. Calc for the AI Camera with its lens set to 20 cm (5 D): +2 D focuses at about 14.3 cm, +4 D at about 11.1 cm. With the lens set to infinity, +4 D focuses at 25 cm. (https://www.raspberrypi.com/news/macro-photography-with-the-camera-board/)
- [high] Picamera2 manual controls:
- AeEnable=False stops automatic gain/exposure updates.
- AwbEnable=False stops colour-gain updates; setting ColourGains (red, blue; 0-32) also disables AWB.
- ExposureTime is in microseconds; FrameDurationLimits bounds the frame time.
- Sharpness 0 means no extra sharpening.
- NoiseReductionMode accepts Off.
- For 50 Hz mains the flicker is 100 Hz, so AeFlickerPeriod is 10000 us. (https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-7-picamera2-manual.pdf)
- [high] Controls set before picam2.start() apply from the first frame. Changing ExposureTime/AnalogueGain while running takes several frames, so the frame metadata must be checked. (https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-7-picamera2-manual.pdf)
- [high] picam2.capture_request(flush=True) returns a frame whose first pixel began exposing no earlier than the call. This is the documented way to capture after a light is switched on or off. flush can also take a monotonic_ns timestamp. (https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-7-picamera2-manual.pdf)
- [high] On a Pi 5 the default raw stream is compressed (e.g. BGGR16_PISP_COMP1). Requesting an unpacked format such as 'SRGGB10' gives uncompressed, left-aligned 16-bit pixels. A still configuration defaults to buffer_count=1. (https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-7-picamera2-manual.pdf)
- [high] IMX500 kernel driver values: IMX500_PIXEL_RATE 744 MHz, full-res line_length_pix 17900, binned 9398, VBLANK_MIN 1117.

Calc:
- Full resolution: 24.06 us per line, 73.1 ms rolling-shutter readout, 100.0 ms minimum frame (10 fps).
- Binned: 12.63 us per line, 19.2 ms readout, 33.3 ms frame (30 fps). (https://raw.githubusercontent.com/raspberrypi/linux/rpi-6.12.y/drivers/media/i2c/imx500.c)
- [medium] The product brief says the AI Camera 'transparently augments captured still images or video with tensor metadata'. The driver computes inference lines with a 'Full-res mode' comment. So the on-sensor network should run alongside 4056x3040 frames, at up to 10 inferences/s. Not yet confirmed on hardware. (https://pip-assets.raspberrypi.com/categories/1044-raspberry-pi-ai-camera/documents/RP-008147-DS-1-ai-camera-product-brief.pdf)
- [high] IMX500 runtime:
- Install with sudo apt install imx500-all; models go in /usr/share/imx500-models/.
- The first firmware load can take several minutes; the RP2040 on the camera caches firmware.
- Create IMX500(model) before Picamera2().
- The ISP crops and scales the inference ROI (default: full sensor) to the network input size. (https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [high] Custom IMX500 model flow:
1. pip install edge-mdt[pt] (or [tf]).
2. Quantise with Sony MCT.
3. Convert with imxconv-pt -i <quantised.onnx> -o <dir>; --no-input-persistency saves on-sensor memory. This outputs packerOut.zip.
4. Package on a Raspberry Pi: sudo apt install imx500-tools, then imx500-package -i packerOut.zip -o <dir>. This produces network.rpk. (https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [high] PyPI versions: edge-mdt 1.3.0 (2026-04-23) pins model-compression-toolkit ~=2.5.1 and imx500-converter ~=3.18.2. Latest standalone releases are MCT 2.6.1 (2026-09-17) and imx500-converter 3.19.2 (2026-10-22). uni-pytorch 3.19.0 requires Python >=3.10,<3.13. (https://pypi.org/pypi/edge-mdt/json)
- [medium] Converter host requirements: 'Linux PC (recommended)' with 4 GB RAM, Python 3.11, JVM 17, tested on Ubuntu 20.04/22.04. A Raspberry Pi 4+ with 4 GB, Python 3.11 and JVM 17 is also listed. macOS is not listed. (https://pypi.org/project/edge-mdt-nightly/)
- [high] Ultralytics IMX500 export:
- Supports only YOLOv8n and YOLO11n (detect, segment, classify, pose), via model.export(format='imx', data=...).
- 'only supported on Linux with Python 3.10–3.12', Java 17+.
- Outputs packerOut.zip and labels.txt.
- On the AI Camera: YOLO11n-cls at 224 takes 33.31 ms/image; YOLO11n at 640 takes 58.82 ms/image. (https://docs.ultralytics.com/integrations/sony-imx500/)
- [high] Sony MCT is tested with Python 3.10-3.12 and PyTorch 2.3-2.6. imx500-converter 3.19 pairs with TPC 1.0/4.0/5.0/6.0. (https://github.com/sony/model_optimization)
- [high] Picamera2 IMX500 examples now live in raspberrypi/picamera2-examples/examples/imx500, including imx500_classification_demo.py. The demo uses create_preview_configuration(controls={'FrameRate': intrinsics.inference_rate}, buffer_count=12). (https://github.com/raspberrypi/picamera2-examples)
- [medium] Listings for 12 V 3-LED 5730 injection modules give 0.72 W, 0.96 W (blue, 110 lm) and 1.2 W per module. That is about 60-100 mA at 12 V (calc). Some modules use a constant-current driver, so measure actual current at 12.0 V. (https://poliledsigns.com/shop/20-modules-3dots-smd5730-led-module-blue-12vdc-0-96watt-110lm-ip67-inject-shell)
- [high] AO3400A (AOS): VDS 30 V, ID 5.7 A, VGS max +/-12 V, VGS(th) 0.65/1.05/1.45 V (min/typ/max), RDS(on) <48 mOhm at VGS 2.5 V and <32 mOhm at 4.5 V, SOT-23. It is fully on from a 3.3 V Pi GPIO. (https://www.aosmd.com/res/datasheets/AO3400A.pdf)
- [medium] IRLZ44N: VGS(th) 1.0-2.0 V, RDS(on) 0.035 Ohm at 4 V and 0.025 Ohm at 5 V. There is no 3.3 V rating; it is fine for 0.1-0.5 A LED banks in practice, but not datasheet-guaranteed. Robu lists it at Rs 39-44, sometimes out of stock. (https://robu.in/product/irlz44n/)
- [medium] The D4184 MOSFET module is about Rs 39 at Robu and is advertised for 3.3/5 V logic, active high. The SmartElex variant manual mentions optocoupler isolation on the input, which needs more GPIO current: check the variant. (https://robu.in/wp-content/uploads/2025/01/USER-MANUAL-1.pdf)
- [medium] Robu 12 V 2 A adapters: Pro-range 5.5x2.5 mm, Rs 519, in stock; 5.5x2.1 mm, Rs 439, out of stock; 'Standard' 12 V 2 A 5.5 mm, Rs 389-399. Prices are from search snippets; recheck. (https://robu.in/product/orange-12v-2a-power-adapter-with-dc-plug-bis-approved/)
- [medium] PWM-dimmed LEDs seen by a rolling shutter give horizontal banding when exposure is shorter than, or not a whole multiple of, the PWM period. Each sensor row sees a different on/off phase. (https://support.enttec.com/pixel/pixel-general-knowledge/why-do-leds-flicker-on-camera)
- [medium] Low-angle dark-field light, 'typically below 15 degrees', makes scratches on polished metal stand out. Blue or UV light 'can reveal fine surface defects'. A light colour boosts similar colours and suppresses complementary ones. (https://mvpromedia.com/machine-vision-lighting-the-fundamentals-behind-reliable-inspection/)
- [medium] Total integrated scatter of a smooth surface is TIS ≈ (4π σ cosθ / λ)². Micro-roughness therefore scatters (630/465)² = 1.84x more under 465 nm blue than under 630 nm red (calc). Base reflectance of aluminium is roughly flat across the visible (UNVERIFIED for this alloy). (https://api.creol.ucf.edu/Publications/5412.pdf)
- [medium] Colour rule for monochrome-style inspection: use the same colour to make a feature light and the opposite colour to make it dark. A red dye indication on white developer should therefore look dark under blue or green light. The actual contrast for the Wesaf dye is unmeasured (UNVERIFIED). (https://roboticstomorrow.com/article/2011/08/eight-tips-for-optimal-machine-vision-lighting/12)
- [high] A Bayer colour filter array is 50% green, 25% red and 25% blue sites. Under blue-only light most of the signal comes from 1/4 of the pixels, plus partial response from green pixels, so effective resolution drops toward the 0.128 mm colour-plane pitch at 20 cm (calc). (https://en.wikipedia.org/wiki/Bayer_filter)
- [high] Magnaflux supplies its SKD-S2 non-aqueous developer as a matte white 3D-scanning spray for shiny metal. A thin developer coat therefore makes the plate close to Lambertian, which suits photometric stereo of dents. It also fills very fine scratches. (https://magnaflux.com/Magnaflux/Resources/Blog/3D-Scanning)
- [medium] Generic 5 mm 395-400 nm LEDs: about 3.2 V at 20 mA, 'light power up to 40mW', 220 ohm suggested from 5 V. Another listing gives Vf 3.2-3.8 V, 30 mA max continuous and 75 mA peak. DigiKey's LED5-UV400 lists 3.4 V typical at 20 mA and 20 deg viewing angle. Invento's own datasheet was not found (UNVERIFIED). (https://sigmanortec.ro/en/5mm-uv-led-395-400nm)
- [medium] Bivar UV5TZ-395-30, a branded 5 mm 395 nm reference part, is listed at 40 mW and 30 deg (RS) with Vf 3.2-3.4 V at 20 mA. Another listing says 20 mW. Assume about 10-40 mW per LED. (https://th.rs-online.com/web/p/uv-leds/7134894)
- [high] UV LED series resistors (calc):
- 12 V, 2 LEDs per string, 270 ohm: 20.7 / 19.3 / 16.3 mA at Vf 3.2 / 3.4 / 3.8 V; resistor about 0.10 W, so 1/4 W is fine.
- 12 V, 3 LEDs, 100 ohm: 24-6 mA spread. Too little headroom; avoid.
- 5 V, 1 LED, 82 ohm: 22.0-14.6 mA; 100 ohm: 18-12 mA.
- Twelve strings (24 LEDs) draw about 0.23 A at 12 V. (calc)
- [medium] EN 62471 UV-A (315-400 nm) limits for the unprotected eye: radiant exposure 10000 J/m² for t < 1000 s, and 10 W/m² for longer exposures. The EU occupational directive lists the eye lens as the tissue at risk (cataract). (https://legislation.gov.uk/eudr/2006/25/annex/I/data.html)
- [high] ams OSRAM AN002 (2025-09-16) says a blue LED of 5 cd or 20 cd at 470 nm dominant 'falls into Risk Group 2' for blue-light hazard, assessed at 200 mm. Do not stare into the blue banks. (https://look.ams-osram.com/m/1d720a7b18ab3fce/original/Details-on-photobiological-safety-of-LED-light-sources.pdf)
- [medium] 3M clear polycarbonate safety lenses block UV between 200 and 380 nm, which does not cover a 395 nm source. Use lenses marketed as UV400 (blocking to 400 nm), or amber lenses. (https://3m.com/3M/sl_SI/p/d/v101736201)
- [medium] Oils and greases luminesce under UV, and this is used commercially to measure part cleanliness (SITA CleanoSpector). A SITA inline monitor uses 365 nm excitation and 460 nm detection. Oil fluorescence is therefore mostly blue, and 395 nm excitation for this purpose is UNVERIFIED. (https://www.productionmachining.com/products/sitas-cleanospector-measures-part-cleanliness)
- [medium] Some red visible penetrants are dual-mode and fluoresce under UV, for example Ardrox 9VF2 (EN ISO 3452 Type II red plus Type III fluorescent) and Magnaflux SKL-4C. No datasheet was found for Wesaf Penetrant-97, so whether it fluoresces is UNVERIFIED and must be tested. (https://silmid.com/es/non-destructive-testing-ndt/penetrants/Ardrox-9VF2-Variant)
- [low] No IMX500 spectral sensitivity curve was found. Whether the sensor, lens and IR-cut filter record 395 nm itself, rather than the LED's 400-420 nm violet tail, is UNVERIFIED. Related Sony back-illuminated sensors keep more than 50% QE down to about 387 nm. (https://arxiv.org/pdf/2211.04913)
- [high] AI HAT+ 2 confirmed: Hailo-10H, '40 TOPS (INT4)', 8 GB on-board RAM, $200. It ships with a GPIO stacking header, so header GPIOs stay available for LED switching. (https://www.raspberrypi.com/products/ai-hat-plus-2/)
- [medium] Robu sells the EasyMech 500 mm 20x20 4V-slot aluminium extrusion (6063, black) for Rs 369. The 20x20 T-slot silver is Rs 188-199, often out of stock. (https://robu.in/product/easymech-500-mm-20x20-4v-slot-aluminium-extrusion-profile-silver/)
- [high] Bank geometry (calc):
- Dark-field bank: line 150 mm horizontally from the plate centre, 40 mm high. Elevation is 14.9 deg at the centre, 21.8 deg at the near edge and 11.3 deg at the far edge. Point-source falloff near/far is 3.6x, less for a line of modules.
- Photometric-stereo bank: 150 mm out, 125 mm high, giving 39.8 deg at the centre (51.3 / 32.0 deg at the edges), 195 mm from the centre.

To reflect light into a straight-down camera, a surface facet must tilt (90-θ)/2: 37.5 deg for 15 deg light, 25 deg for 40 deg light. Flat mill-finish areas therefore stay dark and scratch walls light up. At 200 mm the plate corner is only 19.5 deg off-axis. (calc)
- [low] Exposure estimate (calc): one module emitting about 0.4 W optical (UNVERIFIED) at 250 mm and 40 deg elevation gives about 1.3 W/m² on the plate. On a white matte (developer) surface that is about 2.9e5 e-/s per pixel at 50% QE. Exposures of 10-30 ms at gain 1-4 are therefore the right order; bare aluminium under dark-field is much darker. (calc)
- [medium] Capture cost (calc):
- A full-resolution raw16 frame is 24.7 MB; an RGB888 frame is 37.0 MB.
- A 12-frame sequence (dark, 4 DF, 4 PS, white, UV, dark) takes about 2.4 s at 10 fps with flush (up to 2 frames each), or about 0.8 s in the 30 fps binned mode.
- Twelve raw frames are about 300 MB of RAM. (calc)

## Recommendations
- CAMERA DISTANCE AND FOCUS:
- Mount the AI Camera facing straight down, with the lens front 200 mm above the plate top (0.064 mm/px, calc).
- Focus with the included lens tool on a printed fine-text target placed on the plate. Lock the setting with a dab of nail varnish.
- Only then try about 150 mm (0.048 mm/px) by turning the lens further, a quarter turn at a time, and stop if the lens feels loose.
- A close-up lens (+2 D gives about 143 mm) is optional. The 100 mm plate already fits at 150-200 mm.
- PICAMERA2 CONFIGURATION (load-bearing):

```python
cfg = picam2.create_still_configuration(
    main={'size': (2028, 1520), 'format': 'RGB888'},
    raw={'format': 'SRGGB10', 'size': (4056, 3040)},
    buffer_count=2, queue=False,
    controls={'AeEnable': False, 'AwbEnable': False,
              'ExposureTime': 20000, 'AnalogueGain': 1.0,
              'ColourGains': (r, b),
              'FrameDurationLimits': (100000, 100000),
              'Sharpness': 0.0,
              'NoiseReductionMode': controls.draft.NoiseReductionModeEnum.Off})
```

Then picam2.configure(cfg) and picam2.start().
- Per light state: switch the LED, then call req = picam2.capture_request(flush=True), then raw = req.make_array('raw').view(np.uint16), then req.release().
- Read SensorBlackLevels and ExposureTime from req.get_metadata().
- Confirm the Bayer order from picam2.camera_configuration()['raw']['format'].
- FLICKER: keep ExposureTime at 10000, 20000 or 30000 us (whole multiples of the 10 ms period of 100 Hz light from 50 Hz Indian mains). Set brightness with exposure, gain and the number of modules per bank, never with PWM. If PWM is ever unavoidable, use hardware PWM at 10 kHz or more so each exposure spans 100+ periods (ripple at most 1%, calc).
- CAPTURE SEQUENCE PER PART (about 2.4 s at full resolution):
1. Dark frame D0
2. Dark-field DF-N, DF-E, DF-S, DF-W
3. Photometric-stereo PS-N, PS-E, PS-S, PS-W
4. WHITE
5. UV
6. Dark frame D1

Then:
- Subtract the mean of D0 and D1 from every lit frame in linear raw.
- Ring dark-field image = sum of the four DF frames (by superposition, identical to all banks on, so only one bank is ever lit).
- Scratch orientation comes from which direction is brightest.
- Dent cue = (I_E − I_W)/(I_E + I_W) and (I_N − I_S)/(I_N + I_S). On a developer-coated plate, also do full 4-light photometric stereo.
- BLUE ON A BAYER SENSOR: build a near-full-resolution mono image from raw. Keep the B sites and scale the G sites by k = mean(B)/mean(G), measured on a flat developer-coated plate under each bank. Interpolate the R sites. Flat-field each light with a reference image of a plain (golden) plate to cancel the about 2-3.6x falloff across the plate.
- LED BANK LAYOUT (20 blue modules):
- Dark-field tier: 3 modules per side (12 total) on a bar parallel to each plate edge. Centre line 150 mm out from the plate centre, 40 mm above the plate. Tilt the faces 15 deg down toward the centre (elevation about 15 deg; 11-22 deg across the plate).
- Photometric-stereo tier: 1 module per side (4 total), 150 mm out and 125 mm up, faces tilted 40 deg down (about 40 deg elevation).
- Keep 4 modules as spares.
- For more uniform angles at the cost of light, use 200 mm out and 54 mm up (12-20 deg).
- Add a 10-15 mm black paper lens hood. Banks sit outside the field of view at these positions (calc).
- WHITE LIGHT: add 4 white 12 V 3-LED modules of the same type (or reuse the team's earlier 12 V white 3-LED strips if still owned). Put one per side at the 40 deg tier and switch all four on one MOSFET (about 0.4 A). Use them to:
- calibrate ColourGains once on the developer-coated plate, then freeze them;
- separate red dye from dirt with redness = R/(G+B), or Lab a*;
- give natural colour on the touchscreen and RGB data comparable to MVTec AD.
- UV CLUSTERS: use 24 of the 50 LEDs as 12 strings of 2 LEDs + 270 ohm 1/4 W from 12 V (about 19 mA each, 0.23 A total, calc). Make 4 clusters of 6 on perfboard at the 40 deg positions, aimed at the plate centre; a 30 deg LED gives about a 104 mm spot at 195 mm (calc). Switch all four clusters on one MOSFET.

Uses:
- oil/grease residue check before penetrant;
- a 10-minute test of whether Wesaf Penetrant-97 fluoresces (dab it on the plate, compare the R/G channels of the UV frame minus the dark frame);
- cleaning verification after wiping off excess penetrant.

Keep white paper (optical brighteners) away from the field. For blue oil fluorescence, an optional GG-495 long-pass filter (Edmund India, about Rs 3,734 per the team's earlier quote) blocks the reflected violet.
- SWITCHING HARDWARE: one logic-level N-MOSFET per channel, 10 channels (DF-N/E/S/W, PS-N/E/S/W, WHITE, UV).
- Preferred: the AO3400 board the team already listed as owned (check its channel count), or D4184 modules (about Rs 39 each, Robu; choose the non-opto variant), or IRLZ44N on perfboard.
- Per channel: GPIO to a 100 ohm resistor to the gate; a 10 kohm resistor from gate to source; source to GND; drain to the bank's negative lead; bank's positive lead to +12 V.
- 12 V 2 A adapter through a DC jack to the +12 V rail, with a 1000 uF 25 V capacitor across the rail.
- Tie the 12 V negative to Pi GND (physical pin 6).
- No flyback diodes are needed (LED loads).
- Use gpiozero DigitalOutputDevice (lgpio backend on the Pi 5).
- GPIO MAP (avoid GPIO0-8, which include the HAT ID EEPROM and I2C pull-ups and could turn a bank on at boot):
- GPIO17 (pin 11) DF-N; GPIO27 (pin 13) DF-E; GPIO22 (pin 15) DF-S; GPIO23 (pin 16) DF-W
- GPIO24 (pin 18) PS-N; GPIO25 (pin 22) PS-E; GPIO12 (pin 32) PS-S; GPIO13 (pin 33) PS-W
- GPIO16 (pin 36) WHITE; GPIO26 (pin 37) UV
- GND on pins 6/9/14/20

In software, allow at most 2 channels on at once; all on would draw about 2.2 A (calc), over the 2 A supply. Turn UV on only during capture, so its duty cycle is tiny.
- MEASURE ON DAY 1:
- Current per blue module at 12.0 V (expect 60-100 mA).
- Dark-frame mean with the hood open and closed.
- Blue-channel contrast of a red-dye line on developer under blue versus white light.
- Whether a UV frame minus dark shows anything on bare aluminium (expect nothing).
- IMX500 ROLE (optional, stretch):
- Keep the main defect detector or classifier on the Hailo-10H through HailoRT, and Llama on the CPU through ExecuTorch.
- Use the IMX500 NPU only for a 'part present and seated' or coarse OK/NG classifier, read from the WHITE frame or idle frames.
- Fastest path: Ultralytics YOLO11n-cls at 224, model.export(format='imx', data=<dataset dir>), on Linux x86 (a Linux machine or Colab) with Python 3.10-3.12 and Java 17. Then on the Pi: sudo apt install imx500-all imx500-tools, imx500-package -i packerOut.zip -o out, and load out/network.rpk with IMX500(...) before Picamera2().
- Verify on device that imx500.get_outputs(metadata) is not None in the 4056x3040 configuration.
- Document it in the report as a third, separate pipeline.
- CONVERSION HOST: the Mac is not a listed converter host. Use Google Colab or any Ubuntu x86 machine, or the Pi 5 itself with a Python 3.11 venv (for example via uv) plus OpenJDK 17. Pi OS Trixie ships Python 3.13, which uni-pytorch (<3.13) rejects. Use Colab only for training and conversion; runtime stays fully on-device, as the brief requires.
- PHYSICAL BUILD:
- Base: 450x450 mm board of 6-9 mm MDF or plywood, covered with 2 mm matte black EVA foam or matte black paint.
- Plate stops: two 10x10 mm black strips forming an L at one corner, so the plate always sits in the same place for golden-reference comparison.
- Camera stand: one 500 mm 2020 V-slot upright at a rear corner on the diagonal, so it shades no bank, with a 2020 arm of about 300 mm over the plate centre. Mount the camera on a small plate with M2 screws, squared to the plate edges and levelled.
- Light banks: on 25x25 mm aluminium angle or wooden wedges cut at 15 deg and 40 deg, held on four 150 mm posts.
- Hood: a 450x450x350 mm box of black foam board or blackout cloth with a front flap.
- A 500 mm Pi 5 camera FPC if the Pi sits beside the rig.
- SHOPPING LIST (approx. INR, recheck live prices):
- 12 V 2 A adapter, Robu Pro-range: Rs 439-519
- DC jack to screw terminal: about Rs 30 (est.)
- 10x D4184 module: about Rs 390, or 10x IRLZ44N at Rs 39-44 plus perfboard (skip if the AO3400 board has enough channels)
- Resistor and capacitor set (100 ohm x10, 10 kohm x10, 270 ohm x12, 1000 uF 25 V): about Rs 100-150 (est.)
- 4 white 12 V 3-LED modules: about Rs 100-200 (est., SP Road or Amazon)
- 2x EasyMech 2020 V-slot 500 mm: Rs 738 (Robu), plus brackets and T-nuts at about Rs 200 (est.)
- Pi 5 camera FPC 500 mm: Rs 372 (Robu, from the team's earlier quote)
- MDF, EVA foam and foam board for base and hood: about Rs 400-600 (est.)
- UV400 safety glasses (Uvex i-vo UV400 or 3M amber): about Rs 500-1,500 (UNVERIFIED)

Total about Rs 3,300-4,700 (calc). Optional: GG-495 filter, about Rs 3,734.

## Risks
- The AI Camera lens may not focus closer than 20 cm, or may come loose when turned past its rated range. => Design for 200 mm, where specs are guaranteed. Try 150 mm only with small turns and a fine-text target. If it fails, keep 200 mm or add a +2 D close-up lens.
- Blue-only light on a Bayer sensor lowers effective resolution to about 0.128 mm per colour-plane pixel at 20 cm (calc), so fine scratch widths are not measurable. => Use raw and combine B with gain-scaled G sites. Report scratch detection rather than width below about 2 px. Use white light for colour and measurement frames.
- Under blue light, red dye indications and dark stains, oxide or shadows all look dark, causing false positives. => Add a white bank. Classify an indication as dye only if it is dark under blue and has a high R/(G+B) under white.
- The rolling shutter (73 ms readout) mixes light states if an LED changes mid-frame, giving split frames. => Switch the LED, then capture_request(flush=True). Keep the LED on until the request returns. Use DC drive only, no PWM.
- Ambient room light and 100 Hz mains flicker contaminate images and change between frames. => Use the black hood, exposures of 10/20/30 ms, dark frames before and after each sequence subtracted in raw, and a warning if the dark-frame mean exceeds a threshold.
- Running all LED channels together (about 2.2 A, calc) overloads the 12 V 2 A adapter. Module current is also unmeasured (0.72-1.2 W listings). => Limit simultaneous channels in software (at most 2) and use superposition sums for the ring image. Measure module current at 12.0 V on day 1.
- GPIO0-8 have pull-ups or HAT/I2C roles and could switch a bank on at boot. The D4184 opto variants may not trigger reliably from 3.3 V. => Use GPIO12, 13, 16, 17 and 22-27, add a 10 kohm gate pull-down, and buy or keep non-opto modules or the AO3400 board.
- UV-A eye exposure: aluminium reflects 395 nm strongly. The estimated plate irradiance of tens of W/m² (calc) exceeds the 10 W/m² long-term limit if viewed. Blue LEDs of this class can be Risk Group 2. => Hood closed during capture, UV and blue on only during capture (under about 0.5 s per part), UV400 or amber glasses for anyone setting up, and no looking into the LEDs.
- The IMX500 path may eat schedule: first firmware load takes minutes, conversion needs Linux, Python <3.13 and Java 17, and running alongside full-res stills is unverified. => Treat the IMX500 network as a stretch goal after the core pipeline (lighting, defect model on Hailo, Llama advisory) works. Use the Ultralytics YOLO11n-cls export on Colab. Verify get_outputs() in the full-res configuration in the first hour.
- Photometric stereo on bare mill-finish aluminium violates the Lambertian assumption, and near-field lights vary the light direction by more than 10 deg across the plate. => Run full photometric stereo only on the developer-coated plate. On bare metal use the normalised east/west and north/south difference images as dent cues. Flat-field each light, and high-pass the normal or curvature maps so only local dents remain.
- Wesaf penetrant, developer and LED specifications are undocumented: fluorescence, wavelength, beam angle. => Run quick bench tests (UV on a drop of dye, multimeter current, beam pattern on a wall) before finalising the geometry and code.

## Open questions
- Can the AI Camera lens be turned with the included tool to focus at about 150 mm without coming loose? This needs a bench test.
- Does the IMX500 return valid output tensors in Picamera2 when configured at 4056x3040 (10 fps)? Check imx500.get_outputs(metadata) on the device.
- What is the actual current and peak wavelength of the bought blue 12 V modules (0.72, 0.96 or 1.2 W variant), and their beam angle?
- Does Wesaf Penetrant-97 fluoresce under 395 nm? Does the Wesaf non-aqueous developer fluoresce (optical brighteners)?
- Exactly which touchscreen is it (DSI or HDMI)? It decides whether the second MIPI port is used.
- How many channels does the team's existing AO3400 MOSFET board have? Are the earlier 12 V white 3-LED strips still on hand?
- Is a Linux x86 machine or Colab acceptable for training and conversion? The runtime stays fully on-device.
- Does the IMX500 lens plus IR-cut filter pass meaningful 395 nm light, or only the LED's 400-420 nm tail? This decides whether a long-pass filter is needed for blue oil-fluorescence imaging.