# Raspberry Pi 5 (8 GB) + Active Cooler + AI HAT+ 2 stack: dimensions for a simplified Fusion 360 dummy and enclosure keep-out zones
- Pi 5 PCB | Board outline, corner radius, datum | 85.0 (X) x 56.0 (Y); 4 corners R3.0 (arc centres (3,3), (82,3), (3,53), (82,53)). Datum O = corner at USB-C/HDMI edge (Y=0) and microSD/PCIe edge (X=0). | high
- Pi 5 PCB | PCB thickness | 1.34. STEP: bottom-side parts end at STEP z=0.00 and top-side parts start at z=1.34; laminate solid is 1.28. Drawing side view measures about 1.41 (calc). Model 1.4 and allow 1.6. | medium
- Pi 5 PCB | Mounting holes | 4 x dia 2.70 (M2.5 clearance) at (3.5,3.5), (61.5,3.5), (3.5,52.5), (61.5,52.5); pattern 58.0 x 49.0. Keep a dia 6.0 copper pad area clear around each (drawn dia 5.8–6.0). | high
- Pi 5 PCB | Active Cooler push-pin holes | 2 x dia 3.10 (STEP; drawing labels dia 3) at (3.5, 9.5) and (61.5, 46.5): each is 6.0 from the adjacent mounting hole. | high
- Pi 5 PCB | RJ45 locating-peg holes (info) | 2 x dia 3.30 at (77.25, 4.54) and (77.25, 15.96) | high
- Pi 5 | Overall envelope (no cables) | X -0.45 (power button) .. 88.05 (RJ45 face); X -1.73 with microSD card inserted. Y -1.63 (micro-HDMI faces) .. 56.0. Z -2.79 .. +16.19 (USB 3.0 stack). | high
- Pi 5 | Bottom side (underside) keep-out | Lowest SMD part at Z -2.79 (X 15.1–17.9, Y 32–35), i.e. 1.45 below the PCB bottom. microSD slot body X 2.25–13.65, Y 22.11–34.06, Z -2.76..-1.34. Through-hole tails: GPIO to Z -1.90, RJ45 to -2.00, USB to -1.50 (STEP). The drawing shows RJ45/USB peg/pin features reaching about 1.9 below the PCB bottom (Z about -3.35). Design 2.5 mm clear below the PCB bottom (Z -3.9). | medium
- Pi 5 | USB-C power (J1), mid-mount | Centre X 11.2. Shell X 6.73–15.67 (8.94 wide). Front face at Y -1.20 (1.2 past front edge). Outer Z -1.00..3.26; receptacle shell Z -0.90..2.93, centre about Z 1.0. Drawing: top 3.2 above PCB. | high
- Pi 5 | 2 x micro-HDMI (HDMI0, HDMI1) | HDMI0 centre X 25.8 (body X 22.20–29.40); HDMI1 centre X 39.2 (body X 35.60–42.80). Front face Y -1.63. Z -1.10..3.34, centre about Z 1.1. Drawing: top 3.4. | high
- Pi 5 | USB-A stacks (2 x USB 3.0 + 2 x USB 2.0) | USB 3.0 (blue, next to RJ45): X 70.47–88.00, Y 21.75–36.25 (centre Y 29.0–29.1), top Z 16.19. USB 2.0 (GPIO corner): X 70.33–87.78, Y 39.75–54.25 (centre Y 47.0), top Z 16.02. Faces about 3.0 past the right edge (X 88). Drawing: centres 29.1 / 47, overhang 3. | high
- Pi 5 | Gigabit Ethernet RJ45 | X 66.45–88.05, Y 2.28–18.21 (centre Y 10.2), top Z 13.30. Face 3.0 past the right edge. | high
- Pi 5 | CAM/DISP 22-pin 0.5 mm FPC connectors (vertical entry, cable leaves +Z) | Nearer HDMI, silkscreen CAM/DISP 1: X 47.10–50.65, Y 0.15–16.85, cable slot X 47.85–50.15 / Y 0.5–16.5. Nearer RJ45, CAM/DISP 0: X 53.30–56.85, Y 0.15–16.85, slot X 54.05–56.36. Top Z 3.95 (STEP); drawing label 4.1. Long axis along Y, perpendicular to the front edge. | medium
- Pi 5 | PCIe 16-pin 0.5 mm FPC connector J20 (vertical) | X 1.10–4.65, Y 23.15–36.84 (centre Y 30.0), top Z 3.95 (drawing 4.1), at the left (-X) edge. Cable contacts face +X (toward USB). Ribbon (FFC) max 50 mm, opposite-side-contact type. 5 V on pins 1–2, 500 mA each (1 A total). | high
- Pi 5 | 40-pin GPIO header | Plastic X 7.10–57.90, Y 49.96–55.04; centre (32.5, 52.5), on the top mounting-hole line. Pin 1 at (8.37, 51.23) (calc). Plastic top Z 2.54; pin tops Z 8.54 (drawing about 8.57); tails to Z -1.90. | high
- Pi 5 | FAN connector (4-pin JST-SH, vertical, labelled FAN) | Body X 65.30–68.20 (pins to 69.10), Y 49.00–55.00, top Z 4.30. Right of the top-right mounting hole, outside the HAT outline (HAT ends X 65.0). | high
- Pi 5 | Power button (left edge) | Body X 0.35–2.95, Y 16.15–20.65, top Z 3.30. Actuator X -0.45..0.60, Y 17.40–19.40 (centre Y 18.4), Z 0.70–2.70 (centre Z 1.7). Sticks out 0.45 past the left edge (X=0). | high
- Pi 5 | Status LED (single bi-colour PWR/ACT, side-looking, at left edge) | X 0.16–1.64, Y 11.70–14.90 (centre Y 13.3), Z 0–1.0 (centre about Z 0.5). Emits toward -X. | medium
- Pi 5 | microSD (underside, left edge) | Slot body X 2.25–13.65, Y 22.11–34.06, Z -2.76..-1.34. A fitted card sticks out to X -1.73 (1.73 past the left edge), Y 22.59–33.59 (11.0 wide), under the PCB. | medium
- Pi 5 | RTC battery connector J5 (2-pin JST-SH) and J2 power-switch pads | J5: X 17.0–21.0, Y 3.3–7.1, top Z 4.30, right of USB-C. J2 pads lie between J5 and the board edge. | high
- Pi 5 | UART (3-pin JST-SH) and PoE 2x2 header | UART: X 30.0–35.0, Y 2.5–6.3, top Z 4.30 (between the HDMIs). PoE header: X 58.96–64.04, Y 6.96–12.04, centre (61.5, 9.5), pin tops Z 8.54. | high
- Pi 5 | Main ICs under the cooler (for thermal pads / dummy) | BCM2712 SoC 17 x 17: X 24.65–41.65, Y 14.25–31.25, top Z 2.31. RAM: X 25.9–40.4, Y 34.2–44.2, top 1.0. RP1: X 52.5–64.5, Y 29.1–41.1, top 1.2. Wi-Fi can: X 6.9–17.7, Y 36.2–49.2, top 1.70. PMIC area about X 8.7–13.7, Y 12.7–17.7, top 0.55. Tallest inductor: X 65.35–69.65, Y 40.35–47.65, top 1.90. | high
- Active Cooler | Overall size, fan and fan specification | 63.50 x 42.50 footprint, 13.70 overall height including push pins (free state). Fan 30 x 30. 5 V from the 4-pin fan header; PWM + tach; max 8000 rpm +/-15%; max 1.09 CFM; blower attached by 3 screws. | high
- Active Cooler | Footprint on Pi (registered by its push pins at (3.5,9.5) and (61.5,46.5)) | Bbox X 0.33–64.2, Y 6.5–49.6 including hex pin heads (about 6.3 across corners, 5.5 across flats). Main body starts X about 6.4 (clears PCIe J20 and the power button). Fin block X about 6.8–26.6, Y about 9.5–49. Fan housing X 27.5–57.4, Y about 18–48; clears the CAM/DISP connectors (Y<17). Top intake hole dia about 21.2, centre (41.7, 33.7). | medium
- Active Cooler | Heights | Measured from drawing: base plate 1.49 thick; base underside to fan top 9.28, to fin tips 9.21, to pin-head top 10.48 (free state); barb 3.21 below base underside. Installed estimate: base underside about Z 2.8–3.3 (SoC top 2.31 + thermal pad), fan top about Z 12.1–12.6, fins about 12.0–12.5. Push-pin heads about Z 9.6 (calc: latch 10.98 below head top, seated at PCB bottom Z -1.34). | low
- Active Cooler | Airflow | Intake: top circular hole (dia about 21 at (41.7,33.7), Z about 12.4). Exhaust: sideways toward -X into the pin-fin block (X 6.8–26.6), leaving through the fin tops and sides toward the PCIe/microSD/power-button end. Inferred from the drawing and doc illustration (solid housing walls on the -Y and +X faces); official pages do not state it and forum posts conflict. Fan switches on at 60 C, steps up at 67.5 C, full speed at 75 C. CPU throttles at 80 C, harder at 85 C. | low
- AI HAT+ 2 | Board outline and holes (HAT+ form factor) | 65.0 (X 0–65) x 56.5 (Y 0–56.5), R3.0 corners. Overhangs the Pi's GPIO edge by 0.5 (Pi Y max 56.0). 4 x dia 2.7 holes at (3.5,3.5), (61.5,3.5), (3.5,52.5), (61.5,52.5), coincident with the Pi. Heatsink push-pin holes (drawn dia 3.7) at (3.5, 9.5) and (61.5, 46.5), directly above the Active Cooler pins. | high
- AI HAT+ 2 | PCIe slot and HAT-side FPC connector | Left-edge slot Y 21.0–39.0 (18.0 long; drawing 17.5 / 18 / 21 from top), 5.0 deep (X 0–4.9), about R1 corners. HAT FPC connector (horizontal, side-slide retaining clips; cable enters from -X through the slot) at X 6.1–10.8, Y 23.3–36.7. The ribbon drops vertically from here to Pi J20 directly below (X 1.1–4.65). | high
- AI HAT+ 2 | Camera/display FFC notch | X 47.5–57.5 (47.5 / 10 / 7.5), Y 0–16.0, open to the front edge; sits over both CAM/DISP connectors. HAT+ spec M.2 example uses 16.5. | high
- AI HAT+ 2 | Chips | Hailo-10H 15.0 x 15.0 at X 19.2–34.2, Y 19.6–34.6 (centre 26.7, 27.1). LPDDR4X about 10 x 14.6 at X 37.6–47.75, Y 19.8–34.4. Package heights UNVERIFIED. | high
- AI HAT+ 2 | GPIO pass-through header J1 (on HAT top) | 2x20 female socket at X 7.4–57.6, Y 50.0–55.0 (centre 32.5, 52.5). Height UNVERIFIED: a standard socket is 8.5, giving top about Z 26.1 if the HAT is 1.6 thick. | low
- AI HAT+ 2 | Stack spacing (spacers / 16 mm stacking header) | Board-to-board 16.0, so the HAT PCB bottom is at Z 16.0. The HAT+ spec asks for at least 15 mm with the Active Cooler, 16 mm ideal, more if the HAT underside has parts. Supplied: 4 threaded spacers, 4 long screws (into the Pi), 4 short screws (HAT), GPIO stacking header, ribbon cable. HAT PCB thickness UNVERIFIED (assume 1.6, HAT top Z about 17.6). | medium
- AI HAT+ 2 heatsink | Official size | 42.5 x 64 across the push-pin corners. About 3.2 high without push pins and 14 high with them (free state). Thermal pads sit over the power regulator, the Hailo NPU and the SDRAM. 2 black spring push pins in diagonally opposite corners. | high
- AI HAT+ 2 heatsink | Footprint polygon on the HAT, same XY (calc, +/-0.3) | (3.6,6.5) -> (47.45,6.5) -> (47.45,15.7) -> (48.0,16.2) -> (62.6,16.2) -> (64.65,18.2) -> (64.65,46.5) -> arc about pin (61.5,46.5) R about 3.2 -> (60.4,49.0) -> (11.2,49.0) -> (11.2,20.0) -> (6.8,19.1) -> (1.2,11.3) -> arc about pin (3.5,9.5) R about 3.2 -> back to start. The notch clears the FFC cut-out; the X<11.2, Y>20 region clears the PCIe connector. | medium
- AI HAT+ 2 heatsink | Side-view details and installed heights | Measured from drawing: fins/body 3.2 above heatsink underside; plate 1.2 thick at the pins; thermal pads 0.9–1.1 (free). Barb hangs 3.6 below the underside, max dia about 4.0. Pin head dia about 5.6, 1.4 thick, head top 10.4 above the underside. Installed: barbs below the HAT down to Z about 12.4 (2.8 above the Active Cooler pin heads at Z about 9.6). Pin-head tops Z about 26.4 (16.0 + 10.4, calc). Fin tops Z about 22.3–23.3 (est.: HAT top 17.6 + chip/pad 1.5–2.5 + 3.2). | low
- Whole stack | Height summary (Z from Pi PCB top) | Bottom -2.79 (PCB bottom -1.34). HAT region top about +26.4 (heatsink pin heads; J1 about 26.1) -> about 29.2 total (calc). Over the USB/RJ45 (X>66.45): top 16.19. Fan-top to HAT-underside gap about 3.4–3.9 (calc). | medium
- Power | Supply and draw (non-mm values) | n/a.
Official Raspberry Pi figures:
- Pi 5 recommended PSU is 5 V / 5 A, i.e. the 27 W supply.
- Downstream USB current is 1.6 A on a 5 A supply and 600 mA on any other.
- Typical bare-board active current is 800 mA (about 4 W).
- Powered off, the board draws 1–1.4 W.
- EEPROM setting PSU_MAX_CURRENT=5000 skips USB-PD negotiation.
- The PCIe ribbon (FFC) carries 5 V at up to 1 A.
- The AI HAT+ 2 brief states no PSU requirement; its ambient rating is 0–50 C.
Third-party figures:
- Hailo-10H max about 3 W (Geerling); Hailo spec about 2.5 W (heise).
- Pi 5 8 GB + Hailo-10H averaged 7.59 W system power (Sixfab, vendor).
Stack peak estimate: about 15–16 W without the display or LEDs (calc). | medium
RULES
- Fusion 360 setup:
- Make one component 'Stack_Dummy' with its origin at the datum: Pi PCB corner at the USB-C/HDMI and microSD/PCIe edges, Z=0 on the PCB top.
- Add user parameters: PCB_T=1.34, SPACER=16.0, HAT_T=1.6, HAT_HS_PIN_TOP=10.4, J1_H=8.5, CLR=0.5.
- Optionally use 'Insert > Insert Mesh/Design' with the official STEP (rpi-5b_no_graphics.step) as a reference body. Shift it -1.34 in Z, because the STEP has Z=0 at the PCB bottom.
- Dummy bodies for the Pi:
- PCB 85 x 56 x 1.34 with R3 corners, 4 holes dia 2.7 and 2 holes dia 3.1.
- Boxes for the USB3 / USB2 / RJ45 / GPIO / CAM-DISP / PCIe J20 / FAN / J5 / UART / power button / LED / microSD, using the dimension-list XYZ.
- Underside slab to Z -2.8.
- Active Cooler as a block X 6.4–64.2, Y 9.5–49.6, Z 2.8–12.6, plus pin-head cylinders dia 6 to Z 9.6 at (3.5,9.5) and (61.5,46.5).
- Dummy bodies for the AI HAT+ 2:
- Plate 65 x 56.5 x 1.6 at Z 16.0–17.6, with the PCIe slot (X 0–5, Y 21–39) and the FFC notch (X 47.5–57.5, Y 0–16) cut out.
- Heatsink polygon extruded Z 17.6 -> 23.5.
- Pin-head cylinders dia 5.6 to Z 26.5 at the two pin XYs.
- Barb cylinders dia 4 from Z 16.0 down to 12.4.
- J1 block X 7.4–57.6, Y 50–55, Z 17.6 -> 26.1.
- Spacer cylinders dia 5 (M2.5 hex spacer, about 5 across flats, est.) Z 0–16 at the 4 holes.
- Mounting:
- Seat the Pi on 4 printed bosses on the 58 x 49 pattern, OD 6.0 max (inside the pad keep-out).
- Boss height under the PCB bottom: at least 3.0, recommend 4.0. The underside parts reach 1.45 below the PCB (STEP) and the drawing shows up to about 1.9.
- Through-bolt M2.5 from outside into the supplied 16 mm threaded spacers. Size the screw for boss + 1.34 + at least 4 mm of thread engagement; supplied 'long screw' length is UNVERIFIED. Or use M2.5 heat-set inserts plus male-female standoffs.
- Spacing: keep the 16 mm board-to-board spacing. Never go below 15 mm (HAT+ spec); the cooler fan top sits at about Z 12.4.
- FDM fit: allow 0.5 mm per side around both PCB outlines in pockets. The internal pocket is at least 86 x 57.5 at the HAT level, because the HAT runs to Y 56.5. Add 0.5 per side to every port window.
- Ceiling clearance:
- Put the lid or display-back inner surface over the HAT at Z 29.5 or higher (at least 3 mm above the pin heads at about 26.4).
- Lowest internal floor: Z -3.9 under the board.
- Total internal height for the stack zone is at least 33.5 mm (calc). Over X > 65 (USB/RJ45 zone) only Z 16.2 is needed if those ports are not used.
- Edge port windows:
- Left end (-X): power-button plunger at Y 18.4, Z 1.7 (actuator 2.0 x 2.0, tip at X -0.45).
- LED light-pipe or window at Y 13.3, Z 0.5.
- microSD access slot about 14 x 4 at Y 22–34, Z -3.9..-1.3; the card sticks out 1.73.
- Front (-Y): USB-C window centred X 11.2, Z 1.0, at least 12 x 7 for plug overmolds (est.).
- micro-HDMI windows centred X 25.8 / 39.2, Z 1.1, if an HDMI display is used. Add a right-angle adapter allowance of about 15 mm (est.).
- Ribbon cables:
- PCIe: keep X -3..+6, Y 20–40, Z 4–18 free for the ribbon loop between J20 and the HAT connector. Do not swap in a ribbon longer than 50 mm (spec).
- Camera FFC: it leaves CAM/DISP 0 or 1 vertically (+Z) at X 47.9–50.2 or 54.1–56.4, Y 0.5–16.5, and rises through the HAT notch. Leave at least 10 mm bend space above the HAT there, or bend it out toward -Y in the gap below the HAT.
- Connect camera cables before fitting the Active Cooler (official note).
- Fan cable: keep X 65–70, Y 48–56, Z 0–10 free for the fan plug and wiring.
- Airflow, lower layer:
- The Active Cooler breathes from its top, but the HAT sits only about 3.6 mm above it. Feed air sideways into the Z 4–16 gap between the boards.
- Intake: vents on the enclosure wall facing the Pi's front edge (-Y) over X 20–65, Z 4–16, and/or vents above the HAT FFC notch (X 47.5–57.5, Y 0–16).
- The GPIO + stacking header blocks the +Y side (X 7–58, Y 50–55, Z 0–17.6).
- Exhaust: vents at the -X end over Y 8–50, Z 3–16 (fin-block outlet). Keep intake and exhaust on different faces so air does not recirculate.
- Rule of thumb: at least 700 mm^2 open area for each of intake and exhaust, about 2x the dia 21 fan inlet (est.).
- Airflow, top layer: the HAT heatsink is passive, so put vents directly over its footprint (X 11–65, Y 6.5–49). If a display back covers the HAT, keep at least 5 mm of air above the pin heads (Z 31.5 or higher) and open the sides. Official guidance: run in a well-ventilated environment, and if in a case, the case should not be covered.
- Keep the 12 V LED modules and their drivers (up to 12 x 1.2 W = 14.4 W, calc) thermally separate from the stack's intake path.
- Power path: 5 V / 5 A USB-PD (27 W class) to the Pi 5 USB-C is preferred. With a 5 V / 3 A bank, USB peripherals are limited to 600 mA, and headroom for an estimated 15–20 W system with display is thin. Do not power the HAT separately; it takes 5 V from the Pi.
RISKS
- Several Z values are estimates and must be measured with calipers before the CAD is frozen: AI HAT+ 2 PCB thickness (assumed 1.6), J1 socket height (assumed 8.5), Hailo-10H package height plus pad (assumed 1.5–2.5), and the installed Active Cooler fan-top height (estimated 12.1–12.6). Keep at least 2 mm margin above Z 26.5.
- The drawing and the STEP disagree in places. PCB is 1.41 (drawing) vs 1.34 (STEP). CAM/DISP and PCIe connector heights are 4.1 vs 3.95. RJ45/USB underside protrusion is about 1.9 vs 0.66 below the PCB. Use the larger value. Raspberry Pi marks both drawing and STEP as approximate, 'for guidance only'.
- Active Cooler intake is starved by the HAT (about 3.6 mm gap). In a closed handheld the CPU (throttles at 80/85 C) and the AI HAT+ 2 (max 50 C ambient) can overheat under Bengaluru ambient plus LED heat. Plan a thermal test with vcgencmd measure_temp and the Hailo temperature, and consider an extra small case fan.
- The Active Cooler exhaust direction (-X through the fins) is inferred from the official drawing and illustration, not stated officially. Confirm with a smoke or tissue test before fixing vent positions.
- The microSD card, power button and LED all sit on the -X end, where the exhaust vents also go. Provide access without blocking the airflow.
- The PCIe ribbon loop and the vertical camera FFCs need bend space. Pinching the 50 mm PCIe ribbon can cause PCIe link errors.
- Power bank: most do not offer 5 V / 5 A PD. The Pi 5 then limits USB to 600 mA and may brown out under combined CPU + NPU + display load. PSU_MAX_CURRENT=5000 only skips negotiation and does not add capacity.
- If the team's '40 TOPS HAT' is a third-party Hailo-10H board rather than the official AI HAT+ 2, its outline, heatsink and stack heights will differ.
VERIFY: I re-checked 21 items (17 dimensional) against fresh downloads of the official PDFs, which have the same SHA1 as the colleague's copies. I rasterised the drawings at 600 dpi (1200 for the button detail) and checked the scale against labelled dimensions: '3' measures 2.96 mm and '13.70' measures 13.71 mm.

Most of the colleague's XY numbers hold within about 0.2 mm, so the STEP-derived positions agree with the drawings. Confirmed:
- Pi 5 outline and holes.
- Cooler-pin holes at (3.5, 9.5) and (61.5, 46.5).
- Positions of the front, left and right connectors.
- CAM/DISP and PCIe connectors.
- GPIO header.
- Active Cooler footprint, fan and intake.
- AI HAT+ 2 outline, PCIe slot and FFC notch.
- HAT+ 2 heatsink 42.5 x 64 x 3.2 / 14 and its pin holes.
- 16 mm board spacing.
- PCIe ribbon limits and power figures.

Errors that matter for the CAD:
1. **Heights over X>66.** The drawing side view shows the RJ45 top at 13.9, not the STEP's 13.30, and a USB shell tab reaching about 16.4-16.5, not 16.19. Use a ceiling of at least 16.5 plus clearance over the USB zone.
2. **Active Cooler dummy.** It leaves out the three fan screw heads (about dia 3.5), which stand 1.2 mm above the fan top at about (31.5, 46.3), (54.5, 45.5) and (54.5, 20.5). Installed tops are about Z 13.3-13.8 (est.), not 12.6. Clearance to the HAT underside is about 2.2-2.7 mm at those points; the fan inlet gap stays about 3.5-4 mm.
3. **Stack bottom.** The drawing's RJ45 pegs put the bottom at about Z -3.3, not -2.8, so the stack is about 30.3 mm before clearance.
4. **HAT width conflict.** The AI HAT+ 2 docs text says '66 mm x 56.5 mm' while both dimensioned drawings say 65. Make the pocket 66 + 0.5 per side.

J1 on the HAT is a real surface-mount female header, apparently bottom-entry (hero photo; the stacking header's long pins enter it from below), so its existence is confirmed. Its height is unverified and probably under 8.5 mm. It does not set the ceiling.

Still uncertain:
- Installed heatsink pin-head Z: about 26.4-26.9 (est.). Design to 27 + 3 mm.
- HAT PCB thickness.
- Lengths of the spacers and screws.
- Which connector is CAM/DISP 0 and which is 1.
- Active Cooler exhaust direction.

Renders and working files are in (local working files, not in the repo) (pi5side-1.png, acfront-1.png, actop-1.png, hsside-4.png, docs/hdr_end2.jpg).
- CONFIRMED | Pi 5 PCB outline 85.0 x 56.0 mm, R3.0 corners; datum at the USB-C/HDMI x microSD/PCIe corner | 85 x 56. Both are labelled dimensions on the official drawing. The R3.0 corners come only from the STEP; the drawing does not dimension them but they look like about R3 by eye. The drawing orientation matches the datum: GPIO at the top, USB/RJ45 on the right, USB-C/HDMI at the bottom, PCIe/button/LED on the left.
- CONFIRMED | Pi 5 mounting holes: dia 2.7 at (3.5,3.5), (61.5,3.5), (3.5,52.5), (61.5,52.5); 58 x 49 pattern; dia 6 pad keep-out | Drawing labels: dia 2.7, 3.5 from the edges, 58 and 49. The pad ring measures about dia 5.8 on the raster (calc); a dia 6.0 keep-out is fine.
- CONFIRMED | Active Cooler push-pin holes on the Pi: dia 3 (STEP 3.1) at (3.5, 9.5) and (61.5, 46.5) | Drawing labels dia 3 and '6' twice, each from the adjacent mounting hole. My pixel measure gives 6.1 (calc). The holes are at (3.5, 9.5) and (61.5, 46.5).
- CONFIRMED | Pi 5 PCB thickness 1.34 (STEP) vs about 1.41 (drawing) | Drawing side view, my measure: 1.42 (line centres at 600 dpi) and 1.38 (1200 dpi) (calc). The STEP gives 1.34. The real board is not officially stated; model 1.4 and let pockets accept 1.6.
- CONFIRMED | Front-edge ports: USB-C centre X 11.2, top 3.2; micro-HDMI centres X 25.8 / 39.2, top 3.4 | Drawing labels 11.2, 25.8, 39.2, 3.2 and 3.4. Side-view measure: USB-C top 3.16, HDMI top 3.4 above the PCB top (calc).
- CONFIRMED | Right-edge ports: USB centres Y 29.1 / 47.0, RJ45 centre Y 10.2, port faces 3.0 past X=85 | Drawing labels 29.1, 47, 10.2 and 3. The overhang measures 2.96 (calc). The USB 3 (blue) stack is the one next to the RJ45, which the hero photo confirms.
- REFUTED | Heights over X>66: USB stack top Z 16.19, RJ45 top Z 13.30 (from STEP); 'only Z 16.2 needed over USB/RJ45' | The official drawing side view (calc, PCB top = 0) disagrees with the STEP. RJ45 top is 13.9 (vs 13.30): the body stands about 0.6 above the PCB. USB shell body top is 15.8, and a hooked shell tab at the port face reaches about 16.4-16.5 (vs 16.19). Use 16.5 + clearance as the ceiling over the USB zone and 14.0 + clearance over the RJ45. Smaller related difference: the drawing gives the UART JST 4
- CONFIRMED | Pi underside keep-out: lowest part Z -2.79 (STEP); RJ45/USB pegs about 1.9 below the PCB in the drawing; design 2.5 mm clear below the PCB bottom | Drawing measures: RJ45 locating pegs 1.88 below the PCB bottom, microSD socket about 1.4 below, GPIO tails about 0.6 below (calc). The microSD card sticks out about 1.8 past X=0 (calc; colleague 1.73). The 2.5 mm under-board clearance covers all of these.
- CONFIRMED | CAM/DISP FPC connectors: X 47.10-50.65 and 53.30-56.85, Y 0.15-16.85, top 3.95 (STEP) / 4.1 (drawing), vertical entry | Drawing top-view measure: X 47.2-50.3 and 53.5-56.4, Y 0.7-16.1 (calc). Side view: top 4.09, matching the 4.1 label. Both sit inside the HAT notch, apart from a 0.4 overlap of the left connector body under the notch edge at X 47.5; its cable slot (X 47.85-50.15) is clear. Which connector is CAM/DISP 0 and which is 1 is UNCERTAIN: the silkscreen in the docs photo is ambiguous. This does not affect 
- CONFIRMED | PCIe J20 vertical 16-pin 0.5 mm FPC at the left edge (X 1.1-4.65, Y 23.15-36.84), top 4.1; FFC <= 50 mm; 5 V on pins 1-2 at 500 mA each | Drawing: top 4.1. The top-view body outline reads X about 1.2-4.6, Y about 24.6-35.4, narrower than the STEP's latch-inclusive 23.15-36.84; keep the STEP envelope. RP-008298-DS states: 16-pin 0.5 mm, vertical on Pi 5, FFC 50 mm or shorter, opposite-side-contact cable, 5 V on pins 1-2 at 500 mA each. Docs: contacts face the USB ports.
- CONFIRMED | Left-edge features: power button centre Y 18.4, protruding 0.45; status LED centre Y 13.3 | Drawing labels 18.4, 13.3 and 0.45. My 1200 dpi measure of the button protrusion is about 0.5 (calc).
- CONFIRMED | 40-pin GPIO: centre (32.5, 52.5), pin 1 at (8.37, 51.23), pin tops Z 8.54 | Drawing: '29' from the left mounting hole gives X 32.5. The pin-1 square pad measures (8.3, 51.1), on the left end of the inner row (calc). Side view: pin tops 8.6 (calc).
- CONFIRMED | Active Cooler 63.50 x 42.50 x 13.70, fan 30 x 30; footprint X 0.33-64.2, Y 6.5-49.6; fan X 27.5-57.4, Y 18-48; intake dia 21.2 at (41.7, 33.7) | Labels: 63.50, 42.50, 13.70 (pin-head top to barb tip, free state), 30 and 30. Registering the pins at (3.5, 9.5)/(61.5, 46.5) (measured spacing 57.98 x 36.98) gives (calc):
- Overall: X 0.47-64.27 including the hex heads; body right edge 63.97. Y 6.52-49.65 including the hex; body top 49.0.
- Fan: X 27.48-57.45, Y 18.0-48.0.
- Intake: dia 21.17 at (41.75, 33.69).
- Fins: X about 7.0-26.5.
The bod
- REFUTED | Active Cooler installed height: fan top Z 12.1-12.6; dummy block Z 2.8-12.6; 3.6 mm gap to the HAT underside | Front view, from the base underside (calc): fan-housing top +9.22 (colleague 9.28), barb 3.26 below, base plate 1.44. The colleague missed the three fan screw heads, about dia 3.5, which sit 1.2 above the fan top and reach +10.45, the same height as the free pin heads. They are at about (31.5, 46.3), (54.5, 45.5) and (54.5, 20.5) (calc). Installed (est.), with the base underside at Z 2.8-3.3: fan-
- CONFIRMED | AI HAT+ 2 outline 65.0 x 56.5, R3, holes on the Pi's 58 x 49 pattern, 0.5 overhang at the GPIO edge | The product-brief drawing and the docs drawing PNG both label 65, 56.5, 58, 49, 29 and 3.5. Measured hole margins are about 3.4-3.5 from the bottom edge and 3.9-4.0 from the GPIO edge (calc). Conflict: the docs TEXT says 'approximately 66 mm x 56.5 mm' while the dimensioned drawing says 65. Model 65 but size the pocket for 66 + 0.5 per side.
- CONFIRMED | AI HAT+ 2 PCIe edge slot Y 21-39, 5 deep; camera FFC notch X 47.5-57.5, Y 0-16; HAT FPC connector X 6.1-10.8, Y 23.3-36.7 | Drawing labels on the left edge: 17.5 / 18 / 21 from top to bottom, giving the slot at Y 21-39, and slot depth 5. Bottom edge: 47.5 / 10 / 7.5 with notch depth 16, giving X 47.5-57.5 and Y 0-16. HAT connector measures X 6.1-10.9, Y 23.1-36.6 (calc). The hero photo shows a horizontal FPC on the HAT, with the ribbon wrapping round the slot edge down to the vertical J20.
- CONFIRMED | AI HAT+ 2 heatsink: 42.5 x 64, 3.2 without pins and 14 with pins; pin holes dia 3.7 at (3.5, 9.5) and (61.5, 46.5), above the Active Cooler pins | Drawing labels 42.5, 64, 14 and 3.2; the docs text repeats them. Side view (calc): 3.2 runs from the plate underside to the fin tips; plate 1.25; pads about 0.9; barb 3.6 below the underside, widest just under the plate; head 1.4 thick, top 10.4 above the underside. HAT holes measured at (3.47, 9.6) and (61.5, 46.4), dia about 3.7 (calc). The heatsink pin XYs coincide with the Pi's cooler-pin hole
- CONFIRMED | Board-to-board spacing 16.0 (HAT+ spec: at least 15, 16 ideal); HAT PCB bottom at Z 16.0 | The HAT+ spec ch. 7 says 'at least 15mm board-to-board spacers; 16mm spacers are ideal'. The AI HAT+ 2 brief says it is 'Supplied with ... 16mm stacking header, spacers, and screws'. Lengths of the spacers and screws are not stated anywhere (UNVERIFIED).
- UNCERTAIN | Installed heatsink pin-head top Z about 26.4; whole stack about 29.3 (Z -2.8 to +26.5) | The geometry supports the method: in the free state the barb's latch shoulder sits about 0-0.5 below the plate underside, so latch-to-head is about 10.4-10.9 and the head lands at Z about 26.4-26.9 with the latch at the HAT bottom (Z 16.0) (calc, est.). No official installed height exists. Design to Z 27.0 + 3 mm. The bottom should be Z -3.3 (drawing RJ45 pegs: -1.42 - 1.88), not -2.8, so the stac
- UNCERTAIN | J1 2x20 female socket on the HAT top at X 7.4-57.6, Y 50-55, 8.5 tall (top Z 26.1); HAT PCB 1.6 thick | XY is CONFIRMED: drawing measure X 7.5-57.7, Y 49.9-55.1 (calc). Existence is CONFIRMED: the hero photo shows a black 2x20 female header labelled J1 on the HAT top. It is a surface-mount part with gull-wing legs, apparently bottom-entry: the installation illustration shows the separate 16 mm stacking header with a female body on the Pi pins and long male pins pointing up into it. Its height is UNV
- CONFIRMED | Power: 27 W 5 V/5 A recommended; downstream USB 1.6 A on a 5 A supply, 600 mA otherwise; 800 mA typical active | As stated. Docs quote: 'Raspberry Pi 5 provides 1.6 A ... when connected to a power supply capable of 5 A at +5 V (25 W) ... any other ... 600 mA'.
