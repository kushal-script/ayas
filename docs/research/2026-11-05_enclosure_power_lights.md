# Power budget, battery/power-module bays and LED-strip options for the two-hand Pi 5 + AI HAT+ 2 inspection device (Fusion 360 / FDM enclosure)

(Research for the handheld enclosure, 5 Nov 2026. Every number carries its source; UNVERIFIED and (calc) are marked.)

## Summary

POWER BUDGET (calc, all values are load power)
- Pi 5 8 GB: idle 3.0 W, full 4-core stress 8.8 W (raspberry.tips). Official "typical bare-board active" figure is 800 mA.
- AI HAT+ 2 (Hailo-10H): 2.5 W typical (Hailo), "maximum of 3W" (Jeff Geerling). The official product brief gives NO power figure (UNVERIFIED at board level).
- Whole system, Pi 5 + Hailo-10H running vision (Sixfab, inline USB-PD meter): average 7.59 W, idle 3.6-3.9 W, peaks 9.2-17.8 W across chips and models.
- Active Cooler: about 0.3-0.5 W.
- AI Camera (IMX500): about 1.87 W while running an on-sensor model (Hackster, measured share of the system). Estimate about 1 W if used only as a sensor (est.).
- Display:
  - RPi Touch Display 2, 5-inch: 1.4 W (official brief; TFT 3.3 V at 55 mA, backlight 5 V at 250 mA).
  - Waveshare 5-inch DSI: about 1.2 W.
  - Waveshare 5-inch HDMI (B): 385 mA at 5 V, about 1.9 W.
  - Waveshare 5-inch HDMI (H) V4: 2 W.
- 5 V side totals (calc):
  - Typical: about 11 W (7.6 + 1.9 + 1.4).
  - Peak (Llama on the CPU + NPU + camera + display): 8.8 + 3.0 + 0.5 + 1.9 + 1.4 = 15.6 W, about 3.1 A at 5.1 V.
- 12 V LEDs (calc):
  - 6 white modules at 0.72-1.2 W = 4.3-7.2 W (0.36-0.60 A). One PickTheDeal-type listing says 1.5 W per module, so budget up to 9 W.
  - 6 UV strips of 3 LEDs in series with 150 ohm, about 16 mA: 6 x 0.19 W = 1.15 W. At 20 mA: 1.44 W.
- Everything on at once: about 24.2 W at the loads. With a 90%-efficient 5 V buck that is about 26 W from the bank, or about 2.2 A from a 12 V PD contract.
- Typical, with LEDs on only during capture: about 14-15 W from the bank.

CAN A POWER BANK RUN IT?
- Raspberry Pi's own white paper (RP-009856, Mar 2026) says:
  - The Pi 5 asks only for 5 V PDOs. 5 V/5 A is an OPTIONAL PDO.
  - Without it, the USB ports are limited to 600 mA, USB boot is off and the desktop warns, but the Pi "will function perfectly well with a 3A supply" if peripherals are low-power.
  - With no PD at all, the Pi assumes 5 V/3 A.
  - EEPROM PSU_MAX_CURRENT=5000 skips PD and assumes 5 A. config.txt usb_max_current_enable=1 raises the USB limit to 1.6 A.
- The 600 mA limit hardly matters here: the camera is on CSI, the display on DSI and the HAT on PCIe/GPIO, so nothing uses USB.
- The real limit is about 3 A (15.3 W) total on the 5 V rail. The 15.6 W peak exceeds that, and the voltage drop in the cable at 3 A gets close to the Pi's under-voltage threshold of 4.63 V ±5%.
- So a 5 V/3 A bank works for typical loads but is marginal at peak. Stability with the AI HAT+ 2 at peak is UNVERIFIED and must be measured.
- I found no bank sold in India with a verified 5 V/5 A USB-C PDO. Listicle claims (aswayke, INIU 140 W, NOBIS) are UNVERIFIED.

RECOMMENDED ARCHITECTURE
- A 30-33 W PD bank feeds a PD trigger board fixed at 12 V.
- The 12 V rail drives the LEDs directly, through low-side logic-level MOSFETs.
- A 5 V 5 A buck (Pololu D36V50F5 or D24V50F5, or the SmartElex TPS565201 / 7Semi AP64500 modules sold in India) feeds the Pi's 5 V GPIO pins or a USB-C pigtail. Set PSU_MAX_CURRENT=5000.
- 12 V is NOT one of the normative PD voltages (5/9/15/20 V; Renesas). The bank must actually list 12 V at 2.5 A or more on USB-C. If it does not, set the trigger to 15 V (normative for sources over 27 W) and add a 15 to 12 V buck. Leave room for it in the bay.

INDIA BANKS THAT LIST 12 V/2.5 A AND 15 V/2 A ON USB-C (Xiaomi Pocket Pro is the pick)
| Bank | Size, weight | Energy | Runtime, typical / peak (calc) |
|---|---|---|---|
| Xiaomi Pocket Power Bank Pro 10000 mAh 33 W (PB1030IN, mi.com/in) | 90 x 63.9 x 24.4 mm, 199 g | 37 Wh | about 2.0 h / 1.1 h |
| Anker Nano 10000 30 W (A1259); India SKU UNVERIFIED | 104 x 52.3 x 26 mm, 215 g | 37 Wh | about 2.1 h / 1.1 h |
| Anker Zolo 20K 30 W (A1689) | 120 x 73 x 31.5 mm, 356 g | measured 61 Wh at 9 V/3 A | about 4.3 h / 2.3 h |

- The Anker Zolo throttles to 20 W when hot.
- The Xiaomi 4i 20000 mAh 33 W (PB2030IN) also lists 12 V/2.5 A, but at 440 g it is over budget.
- Common 20-22.5 W banks are NOT enough: they give only 12 V/1.67 A, e.g. the Xiaomi Compact 20000 (PB2020IN): 128 x 73 x 32 mm, 342 g.

LIGHTS
- White: keep the 12 V injection-moulded 3-LED 5730 modules.
  - Outline per listings: 77 x 12.5 x 6 mm or 78 x 15 x 5 mm. Adhesive back, often with end screw tabs.
  - Similar 2835 modules range from 60 x 13 x 4 mm to 68 x 17 x 11 mm.
  - MEASURE the owned blue modules with calipers before modelling.
- UV: build DIY strips from the 50 owned 5 mm 395 nm LEDs, 3 per perfboard strip.
  - Bivar UV5TZ drawing: body diameter 5.0/4.9 mm, flange diameter 5.8 mm, flange 1.0 mm thick, 8.7 mm from flange underside to dome top, leads 0.5 mm square at 2.54 mm pitch, 25.4 mm long minimum.
  - Vf 2.9-3.5 V (3.2 V typical) at 20 mA; 30 mA absolute maximum; about 40 mW at 20 mA; 15 or 30 degree beam.
  - Use 3 in series + 150 ohm from 12 V. 120 ohm can exceed 30 mA at Vf 2.9 V and 12.6 V supply.
- Ready-made 395 nm 12 V 2835 tape (120 LEDs/m, cut every 25 mm per 3 LEDs, 8/5/10 mm wide, 2.5 mm thick, about 0.3 W per segment) exists overseas (SuperLightingLED). I found no confirmed 395 nm 2835 tape listing in India. Probots lists a 12 V 5050 UV strip, out of stock, wavelength not stated.

THERMAL AND SAFETY
- PLA's heat limit is about 50-60 °C (HDT, about 57 °C Tg) and it has the poorest UV resistance. Use PETG or ASA for the LED face, or put an aluminium strip behind the modules.
- The AI HAT+ 2 is rated for 0-50 °C ambient, so the inside of the enclosure must stay at or below 50 °C.
- UV-A eye limit (IEC 62471): 10 kJ/m² for exposures under 1000 s, or 10 W/m² for longer. A bystander looking straight into all 18 LEDs from 200 mm could reach it in about 3 minutes (calc, est.).
- Mitigations: recess or hood the LEDs, interlock them in software, and always aim them away from the user (the display faces the user, the LEDs face forward).

## Dimensions

- **Raspberry Pi Touch Display 2, 5-inch (portrait)**, Unit size / active area (for the display bay; power is 1.4 W): Unit 91.5 x 143.4; active area 62.1 x 110.4 (power 1.4 W; TFT 3.3 V at 55 mA; backlight 5 V at 250 mA; 500 cd/m²; operating temperature -20 to +70 °C) (high; https://pip-assets.raspberrypi.com/categories/1083-raspberry-pi-touch-display-2/documents/RP-010430-MM-1-touch-display-2-5-inch-product-brief.pdf)
- **Xiaomi Pocket Power Bank Pro 10000 mAh 33 W (PB1030IN, India)**, Outline L x W x H, weight, PD profiles: 90.0 x 63.9 x 24.4 mm; 199 g; 37 Wh. USB-C out: 5V3A, 9V3A, 11V3A, 12V2.5A, 15V2A, 20V1.5A (33 W max). USB-A: 12 V 1.67 A max. Dual-port use drops to 5V3A. Note: the global PB1030ZM 'Pocket Edition Pro' is a different body (105 x 55.8 x 25.5 mm; rated 5600 mAh at 5V3A, i.e. 28 Wh, 76% (calc)) (high; https://www.mi.com/in/product/xiaomi-pocket-power-bank-pro-10000-mah-33w/specs/ ; https://www.mi.com/global/product/xiaomi-33w-power-bank-10000mah-pocket-edition-pro/specs/)
- **Anker Nano Power Bank 10000 mAh 30 W, built-in USB-C cable (A1259)**, Outline, weight, PD profiles: 104 x 52.3 x 26 mm; 215 g. USB-C (cable and port): 5V3A, 9V3A, 10V2.25A, 12V2.5A, 15V2A, 20V1.5A (30 W). Page also says 'Total 24W Max' (conflict; probably multi-port). India availability UNVERIFIED (Flipkart lists Anker A1680011 30 W 10000 mAh: 109.9 x 65.5 x 25 mm, about 217 g, per-port profiles not stated) (medium; https://www.anker.com/uk/products/a1259-built-in-cable-power-bank-10000mah ; https://www.flipkart.com/anker-10000-mah-30-w-compact-power-bank/p/itm435ff847074a2)
- **Anker Zolo Power Bank 20K 30 W (A1689)**, Outline, weight, PD profiles, measured capacity: 120 x 73 x 31.5 mm; 356 g (retailer says 320 g). USB-C: 5V3A, 9V3A, 12V2.5A, 15V2A, 20V1.5A. Measured 60.93 Wh at 9V/3A (82%). Exceeded 70 °C and throttled to 20 W after about 25% discharge (mainly at 20 V). Multi-port use drops to 5V3A only (medium; https://techtest.org/anker-zolo-power-bank-20k-30w-a1689-preiswerte-powerbank-mit-integriertem-kabel-aber-hitzeproblemen/)
- **Xiaomi Power Bank 4i 20000 mAh 33 W (PB2030IN, India)**, Outline, weight, PD profiles: 150.6 x 72.2 x 26.3 mm; 440 g (over the about 350 g budget); 74 Wh; USB-C 12V2.5A, 15V2A, 20V1.5A (high; https://www.mi.com/in/product/xiaomi-power-bank-4i-20000mah-33w-sonic-charge/specs/)
- **Xiaomi Compact Power Bank 20000 mAh 22.5 W, built-in cable (PB2020IN) - NOT recommended**, Outline, weight, 12 V limit: 128 x 73 x 32 mm; 342 g; 74 Wh; every output 12 V 1.67 A max (20 W), which is below the about 26 W peak (high; https://www.mi.com/in/product/xiaomi-power-bank-20000/specs/)
- **Anker PowerCore 10000 PD Redux 25 W - NOT suitable for a 12 V rail**, Outline, weight: 107 x 52 x 27 mm; about 194 g; USB-C out 5V3A / 9V2.78A plus PPS only (no 12 V) (medium; https://service.anker.com/product-description/a085g000004x2B1AAI/powercore-10000-pd-redux)
- **PD trigger: ZY12PDN (button-select 5/9/12/15/20 V, STM32F030 based)**, Board outline: Bare PCB 30 x 15; screw-terminal version COM-ZY12PDN-ST 31.5 x 15 x 14 H (5 A max, 100 W) (high; https://siliconchip.com.au/Issue/2021/August/El+Cheapo+Modules%3A+USB-PD+Triggers ; https://www.joy-it.net/files/files/Produkte/COM-ZY12PDN-ST/COM-ZY12PDN-ST_Datenblatt_2021-11-10.pdf)
- **PD trigger: small fixed-voltage solder-pad boards (IP2721/CH224-class, chip unnamed)**, Board outline: Delinx PP2080: 19 x 10 x 4.3 H (pads select 20/15/12/9/5 V; default 20 V; 5 A; falls back to the highest voltage at or below the target). Joy-it ZYPDS-02: 19 x 10 x 5 H (5/9/12 V by removing resistors; 5 A, 60 W) (high; https://media.jaycar.com.au/product/resources/PP2080_datasheetMain_171073.pdf ; https://asset.conrad.com/media10/add/160267/c1/-/en/003588332DS01/tehnicni-podatki-3588332-joy-it-zypds-02-napajalni-modul-1-kos.pdf)
- **PD trigger: IP2721 fixed-12 V decoy (PDC004-PD type; Robu SKU R175169, Rs 99, out of stock)**, Board outline: UNVERIFIED (no dimensions published); 12 V fixed, 5 A, no PPS, 'may attempt to negotiate for the next available voltage level' (low; https://www.oomipood.ee/en/product/usb_pd_trigger_moodul_jooteklemmid_12v_05a_valjundiga ; https://aws.robu.in/?p=2276006)
- **PD trigger: CH224K boards**, Board outline / config: Board size UNVERIFIED. Tindie DIP-switch trigger (chip not named) is about 25.7 x 11. CH224K CFG-to-VDD resistor: 10k = 5V, 20k = 9V, 47k = 12V, 100k = 15V, 200k = 20V (UNVERIFIED; Chinese doc). One reviewed board died during a 20 V load test (low; https://www.beyondlogic.org/review-usb-c-power-delivery-trigger-board-ch224/ ; https://w2.electrodragon.com/Chip-cn-dat/WCH-dat/CH224-dat/CH224-dat.md ; https://www.tindie.com/products/pcb-studios/type-c-pd-decoy-dip-switch-trigger-module/)
- **PD trigger: Adafruit HUSB238 (#5807)**, Board outline: 24.5 x 20.2 x 4.9 H; jumper-select 5/9/12/15/18/20 V or I2C; 2.2 g (high; https://www.adafruit.com/product/5807)
- **5 V buck: Pololu D36V50F5 (#4091)**, Outline / holes: 25.4 x 25.4 x 9.5 H; three Ø2.18 mm (0.086 in) holes for M2; input 5.5-50 V; 5 V ±4%; 5.5 A typical (at 36 V in); efficiency 80-95%. India: only via TME (Poland) per Pololu (high; https://www.pololu.com/product/4091 ; https://www.pololu.com/product/4091/distributors)
- **5 V buck: Pololu D24V50F5 (#2851)**, Outline / holes: 17.8 x 20.3 x 8.8 H; two Ø2.18 mm holes (M2) spaced 13.5 mm (0.53 in) horizontally and 16.0 mm (0.63 in) vertically (calc from inches); input 6-38 V; 5 A; 85-95% (high; https://www.pololu.com/product/2851)
- **5 V buck: MINI560 (5 V or 12 V fixed versions)**, Outline: Listings disagree: 29 x 18 x 5.4 / 30 x 18 x 5.5 / 33 x 18 x 5.4. Rated '5A (continuous 3A with heat dissipation)', which is marginal for a 3.1 A Pi peak (low; https://robu.in/product/mini560-dc-5v-5a-step-down-stabilized-module/ ; https://quartzcomponents.com/products/mini560-dc-dc-step-down-stabilized-voltage-module)
- **5 V buck: XL4015 (bulky) / LM2596 (too weak)**, Outline / rating: XL4015 module about 54 x 24 x 12-16 H; about 3 A continuous without airflow. LM2596 IC is rated 3 A maximum (4.5-40 V in), so it is too weak for a 3.1 A peak (medium; https://www.phippselectronics.com/product/xl4015-5a-dc-dc-step-down-adjustable-power-supply-module-with-heatsink/ ; https://www.ti.com/product/LM2596)
- **5 V 5 A buck modules sold in India (SmartElex TPS565201, 7Semi AP64500)**, Outline: UNVERIFIED (pages rate-limited). TPS565201 IC: 4.5-17 V in, 5 A, 500 kHz (fine for 12 or 15 V in, NOT 20 V). AP64500 module: 7-40 V in, 5 V 5 A (low; https://www.ti.com/product/TPS565201 ; https://robocraze.com/products/smartelex-buck-module-5v-5a-tps565201ddcr ; https://robocraze.com/products/7semi-dc-dc-buck-converter-7v-40v-input-to-5v-5a-output-ap64500)
- **All-in-one PD to 5 V 5 A boards for Pi 5 (RegalDream/Pichondria, India)**, Outline: UNVERIFIED. Takes 9/12/15/20 V PD in (needs a 27 W or larger source), gives 5 V 5 A out on USB-C or pads. Sold out since 13 Oct 2026 (Tindie, Ernakulam) (low; https://tindie.com/products/regaldreamtech/usb-pd-2030-to-5v-5a-converter-board-for-rpi5 ; https://pichondria.com/2024/08/06/power-rpi5-using-powerbank/)
- **MOSFET switch: D4184 dual-MOS module (one channel per module)**, Outline: 34 x 17 x 12 H (EPFL sheet); trigger 3.3-20 V; load 5-36 V; 15 A. Other listings: 23 x 16 with Ø2 holes (Robu) or 32.6 x 16.5 (Zbotic), so measure the unit you get. Opto variants may not trigger reliably from 3.3 V (earlier notes) (medium; https://make.epfl.ch/storage/media/1426/Module_D4184.pdf ; https://robu.in/product/d4184-mosfet-control-module-replacement-relay)
- **MOSFET switch: 4-channel boards**, Outline: No AO3400 4-channel board with published size was found. Avoid the common IRF540 4-channel board: 66 x 66, four Ø3 holes, IRF540 not specified for 3.3 V gate drive. DIY AO3400A (SOT-23, specified for 3.3 V drive) 4-channel perfboard about 40 x 25 x 10 (est.) (medium; https://www.tinytronics.nl/en/switches/transistors-and-mosfets/mosfets/high-power-mosfet-module-5-20v-10a-4-channels ; https://components101.com/mosfets/ao3400a-n-channel-mosfet-pinout-datasheet)
- **12 V 3-LED 5730 injection module (white/blue)**, Outline L x W x H: 77 x 12.5 x 6 (Makers Electronics 5730 IP65) or 78 x 15 x 5 (Poliled 5730 PVC shell). 0.72 W/60 mA, 0.96 W/110 lm or 1.2 W variants; one PickTheDeal-style listing says 1.5 W per module. Mounting: adhesive tape; screw tabs on some (UNVERIFIED for the owned part). Wire exits at both ends (daisy-chain) (medium; https://makerselectronics.com/product/high-brightness-5730-led-ip65-module-12v/ ; https://poliledsigns.com/shop/20-modules-3dots-leds-red-module-smd5730-12v-0-72watts-60ma-110lum-ip67-pvc-shell-78x15x5mm ; https://www.desertcart.ni/products/102218408)
- **12 V 3-LED 2835 injection modules (reference family)**, Outline, joiner wire, beam: 60 x 13 x 4 (Makers 2835 IP65); 74 x 12 x 4.5 (EC21, tape plus screw hole, 115°); 68 x 17 x 11 with 80 mm joiners (Lights and Parts, 1.14 W); SA-LED S-Mini 3x2835: 40.75 x 9.4 x 10.35, 0.72 W, 60 mA, 160°, -25 to 60 °C (medium; https://makerselectronics.com/product/led-module-2835-12v-ip65/ ; https://www.ec21.com/product-details/Back-lit-2835-Injection-strong--9239040.html ; https://cdn.shopify.com/s/files/1/0082/4118/4847/files/415457-LED_Backlighting_for_sign-Spec_Sheet.pdf ; https://sa-led.com/wp-content/uploads/2020/10/SA-S-Mini-3LED2835Series_ver2.pdf)
- **Sign module datasheet example (LinearLux, 2x2835)**, Outline, wire pitch, mounting, operating temperature: 40.0 x 15.0 x 11.0; 90 mm wire between modules; 3M tape plus screw holes; 160°; operating -30 to +70 °C (indicates the family's temperature limits) (high; https://linearlux.com.au/wp-content/uploads/2021/03/Box-Letter-0.72W-Sign-Module-1.pdf)
- **12 V 2835 LED tape, 60 LEDs/m**, 3-LED cut segment: 50 mm long per 3 LEDs at 12 V (100 mm at 24 V). Width 6 mm (Perspex sheet, 12 W/m, about 0.6 W per segment, calc) or 8 mm (common). QLT catalogue: 4.8 W/m, i.e. about 0.24 W per segment (calc) (medium; https://www.perspex.co.uk/Perspex/media/General/technical-library/PDFs/2835-LED-60m-S-LED.pdf ; https://grobotronics.com/led-strip-2835-ip20-60l-m-8mm-warm-white-5m.html?sl=en ; https://viva-interiors.com/en/catalogs/qlt/collection-2024/18754/page-28)
- **12 V 2835 LED tape, 120 LEDs/m (incl. 395/365 nm UV)**, 3-LED cut segment: 25 mm long per 3 LEDs; PCB width 8 / 5 / 10 mm; thickness 2.5 mm; 12 W/m max, i.e. about 0.3 W per segment (calc); IP20/IP65/IP67/IP68 options (high; https://www.superlightingled.com/product_spec.php?products_id=5140)
- **12 V UV LED strip, India (Probots, 5050, 5 m)**, Availability / segment: Rs 2,499, out of stock, wavelength not stated. 5050 60/m tape normally cuts every 50 mm and is about 10 mm wide (est.) (low; https://probots.co.in/uv-led-strip-waterproof-12v-5050-5m-length.html)
- **5 mm 395 nm through-hole LED (Bivar UV5TZ-395-15/-30 drawing; generic 5 mm UV similar)**, Package: Body Ø5.0 (4.9 min); flange Ø5.8; flange thickness 1.0; flange underside to dome tip 8.7; leads 0.5 square at 2.54 (0.100 in) pitch, 25.4 min long; PCB lead hole Ø0.81 (0.032 in); flat on flange = cathode. Electrical: Vf 2.9/3.2/3.5 V min/typ/max at 20 mA; recommended 10-20 mA; 30 mA DC absolute max; 120 mW Pd max; about 40 mW optical at 20 mA; 15° or 30° (high; https://www.bivar.com/parts_content/Datasheets/UV5TZ-XXX-XX.pdf)
- **DIY UV strip (3 x 5 mm LED on 2.54 mm perfboard)**, Board and panel-hole geometry (calc): LED pitch 10.16 (4 holes) gives a 26.1 span over the flanges; board 30.5 x 10.2 (12 x 4 holes) x 1.6 thick; series resistor and wires on the back; about 3 mm solder/lead height behind. Front-panel hole Ø5.2 drilled (print Ø5.3, FDM holes come out undersize); inside counterbore Ø6.3 x 1.2 deep for the flange stop. Web between Ø6.3 counterbores is 3.86 mm (calc); at 7.62 pitch it would be only 1.32 mm. Dome protrudes 8.7 - 1.0 - panel thickness (medium; calc from https://www.bivar.com/parts_content/Datasheets/UV5TZ-XXX-XX.pdf)
- **Universal battery bay (Xiaomi Pocket Pro 33 W OR Anker Nano A1259)**, Internal pocket (calc): 106 x 66 x 28 internal (covers 90 x 63.9 x 24.4 and 104 x 52.3 x 26 with at least 1 mm clearance; foam shim for the smaller bank), plus about 20 mm at the port end for a right-angle USB-C plug (est.). If designing for the Pocket Pro only: 92 x 66 x 26.5 plus 20 mm plug zone (medium; calc from the mi.com/in and anker.com specs above)
- **Power-electronics bay (trigger + 5 V buck + optional 12 V buck + 4 switch channels)**, Volume (calc): About 95 x 50 x 16 with 4 x D4184 modules; about 80 x 40 x 16 with a DIY 4-channel AO3400A board (sum of footprints x 1.2 for wiring) (low; calc from the module dimensions above)
- **UV beam geometry (calc)**, Spot size and aim: 30° LED at 200 mm gives a 107 mm spot diameter (2 x 200 x tan 15°). LEDs offset 70 mm from the camera axis need about 19° toe-in to aim at the centre at 200 mm (atan 70/200); 15-20° wedge in the holder (medium; calc)

## Design rules

- ARCHITECTURE (recommended): PD bank USB-C port -> fixed 12 V PD trigger (19 x 10 mm class, or ZY12PDN 31.5 x 15 x 14) -> fuse (3.15 A slow-blow, est.) + main switch -> 12 V rail. The 12 V rail feeds (a) the LED MOSFET channels directly and (b) a 5 V 5 A buck to the Pi. Peak 12 V current about 2.2 A, within a 2.5 A PDO (calc).
- FALLBACK provision: reserve about 30 x 18 x 6 mm next to the trigger for a 15 V to 12 V buck. 12 V is not a normative PD voltage, so if a bank lacks a 12 V PDO, bridge the trigger to 15 V (normative for sources over 27 W) and add the buck. Many triggers fall back to the next lower voltage (9 V), which leaves the white modules dim and the 3-series UV strips dark (fails safe).
- Use ONLY the bank's USB-C port (USB-A tops out at 12 V 1.67 A). Never charge another device from the bank at the same time: multi-port use drops to 5 V 3 A total on Xiaomi and Anker banks.
- Pi feed: EITHER solder 5 V to GPIO pin 4 (5V) plus GND pins 9/14 on the AI HAT+ 2's 16 mm stacking-header passthrough, OR use a short right-angle USB-C pigtail into the Pi's USB-C. NEVER do both, and never feed the GPIO while the USB-C is connected to another supply. The Touch Display 2 lead already uses pins 2/6. Use 20 AWG for 5 V at 4 A and keep the run at or below 100 mm (drop about 26 mV at 4 A per 100 mm pair, calc).
- EEPROM (sudo rpi-eeprom-config -e): PSU_MAX_CURRENT=5000 (skips PD and enables the 1.6 A USB mode), and POWER_OFF_ON_HALT=1. With that set, off-state draw falls from about 1-1.4 W to about 0.01 W. The bank may then switch itself off at near-zero load; that is fine, but plan a wake method (bank button or main switch).
- Set the 5 V buck to 5.1 V if it is adjustable. Pololu D36V50F5 is 5.0 V ±4%. The Pi logs under-voltage below 4.63 V ±5%. Verify under load with `vcgencmd pmic_read_adc EXT5V_V` and `vcgencmd get_throttled`.
- Bring the Pi 5 J2 power-button pads (between the RTC battery connector and the board edge) to a normally-open momentary button on the enclosure top edge; it does the same as the onboard button.
- Bench-test with the official RPi 27 W PSU (12 V 2.25 A PDO) or 45 W PSU (12 V 3.75 A) before running from a bank.
- Bank choice and placement: Xiaomi Pocket Power Bank Pro 10000 33 W (90 x 63.9 x 24.4 mm, 199 g) is first choice. Mount it low and central between the grips for balance, as a slide-in drawer or hatch from the bottom for swapping. Give it vent slots on two faces and keep it out of the Pi/HAT exhaust path (the Anker Zolo throttled when above 70 °C).
- Bays to model in Fusion 360: battery bay 106 x 66 x 28 internal + 20 mm plug zone (universal) or 92 x 66 x 26.5 + 20 mm (Pocket Pro only). Power-electronics bay about 95 x 50 x 16 (4 x D4184) or about 80 x 40 x 16 (DIY AO3400A). Bosses: M2 for Pololu (Ø2.18 holes), M2 or M3 for the others; standoffs at least 3 mm so solder joints clear the plastic.
- LED channels: at least 4 low-side channels (white-left, white-right, UV-left, UV-right), on GPIO12/13/16/17/22-27 with 100 ohm gate resistor + 10 kohm gate pull-down (earlier project notes). Never light all white + all UV at once: sequence captures and keep at most one side's white bank on (keeps peak at or below about 25 W from the bank, calc).
- White LED pocket: model 80 x 16.5 x 7 mm per module (from the 77-78 x 12.5-15 x 5-6 mm listings + 1 mm clearance). Add a 3 x 3 mm wire slot at each end and allow for the 70-90 mm daisy-chain leads in a rear channel. Re-measure the owned blue PickTheDeal modules and adjust before printing. Fix with the module's foam tape and add an M3 screw boss at each end if the module has tabs.
- UV DIY strip: 3 x 5 mm 395 nm LEDs at 10.16 mm pitch on a 30.5 x 10.2 mm perfboard. Front holes Ø5.2 drilled (Ø5.3 as printed) with an inside counterbore Ø6.3 x 1.2 mm as the flange stop. Panel 2-3 mm thick; add a 15-20° toe-in wedge so the 30° beams overlap at 150-250 mm. Use the 30° variant (or wider) rather than 15°.
- UV resistor math (calc), 3 LEDs in series from 12 V: R = (Vs - 3Vf)/I. At typical Vf 3.2 V and 20 mA, R = 120 ohm, but at Vs 12.6 V and Vf 2.9 V the current reaches 32.5 mA, above the 30 mA maximum. Use 150 ohm: about 16 mA typical, 26 mA worst-high, 6 mA worst-low; resistor 0.10 W, so 1/4 W is fine. If strip-to-strip spread is too large, switch to 3 LEDs in parallel, each with its own 470 ohm: 16.8-20.6 mA each, 0.165 W per resistor (1/2 W preferred), 56 mA per strip (0.67 W).
- Budget numbers to design the thermal and electrical path for (calc): 5 V rail 11 W typical, 15.6 W peak. White LEDs 7.2 W (9 W if the modules are really 1.5 W). UV 1.15-1.44 W in series form (4.0 W in parallel form). Bank output about 14-15 W typical, about 26 W absolute peak. Runtime about 2 h typical / 1.1 h peak on a 37 Wh 10000 mAh bank, about 4 h / 2.2 h on a 20000 mAh bank.
- Thermal: the AI HAT+ 2 is rated for 0-50 °C ambient and the Pi/HAT stack dissipates about 11-16 W, so give the central body a through-flow path: inlet on the bottom edge, outlet on the top edge, away from the palms and the bank. Keep internal air at or below 50 °C, i.e. at most about 15 °C rise over a 35 °C Bengaluru ambient (est.).
- Materials: print the front face, LED holders and UV bezels in PETG (HDT about 70 °C) or preferably ASA (about 90-110 °C, best UV resistance), not PLA (about 50-60 °C, least UV-stable). Optionally back the white modules with a 1-1.5 mm aluminium flat strip; module operating limits are -25 to 60 °C or -30 to 70 °C per datasheets.
- UV/optical safety: recess the UV domes 3-5 mm, or add a 10-15 mm black hood around the camera and LEDs. Turn UV on only while a trigger is held and during capture, with a software auto-off at 2 s or less. Put a 'UV radiation - do not look into the light' label on the front, as the Bivar datasheet asks. Keep the display on the user side and all emitters on the front.

## Risks

- 5 V/3 A power-bank-direct option: the 5 V side peaks at about 15.6 W (about 3.1 A) against a 3 A limit, plus USB-C cable drop. Under-voltage, throttling or reboots are likely when Llama on the CPU and the Hailo run together. UNVERIFIED in practice, so measure EXT5V_V. PSU_MAX_CURRENT=5000 on a 3 A source only removes the software guard; it does not add current.
- 12 V is not a normative PD voltage. A bank without a 12 V PDO makes a fixed-12 V trigger fall back (often to 9 V) or not start. Confirm '12V⎓2.5A' on the bank's USB-C spec (Xiaomi PB1030IN, Anker A1259/A1689 list it). Most 20-22.5 W banks list only 12 V 1.67 A.
- The bank's thermal throttling or PD renegotiation (when hot, when a charger is plugged in for pass-through, or when a second device is plugged in) can briefly drop or collapse the 12 V rail and brown out the Pi (est.). Add bulk capacitance on the 12 V rail (e.g. 470-1000 µF, est.) and use a filesystem setup that tolerates power loss.
- Weight and balance: 20000 mAh 30 W banks weigh 356-440 g, at or above the about 350 g target. The 10000 mAh banks (199-215 g) give only about 2 h typical and about 1.1 h at peak, so plan swappable banks.
- LED module power is uncertain (0.72 / 0.96 / 1.2 / 1.5 W listings). At 1.5 W the white bank is 9 W and the all-on peak reaches about 2.3 A at 12 V, close to the 2.5 A PDO. Measure each module at 12.0 V on day 1.
- Cheap trigger boards: a CH224K board was killed by a 20 V load-off transient (Beyondlogic). Fix the trigger to 12 V (never 20 V) and verify the output with a meter before connecting the Pi.
- MINI560-class 5 V modules are realistically about 3 A continuous, which is marginal. Prefer Pololu D36V50F5/D24V50F5 (genuine, but only via TME in India) or TPS565201/AP64500-based 5 A modules (dimensions UNVERIFIED).
- Opto-isolated D4184 variants may not switch reliably from 3.3 V GPIO, and IRF540 4-channel boards are not specified for 3.3 V gate drive. Use AO3400A (specified for 3.3 V drive) or a verified non-opto D4184.
- UV exposure (calc, est.): about 40 mW per LED in a 30° beam is about 3.2 W/m² on-axis at 200 mm; 18 LEDs converging gives about 50-60 W/m², which reaches the IEC 62471 UV-A eye limit (10 kJ/m²) in about 3 minutes for someone staring into the front. Reflections off polished metal parts can redirect the beam. 395 nm also adds blue-light (retinal) hazard.
- UV (395 nm) slowly yellows and embrittles PLA (and to a lesser degree PETG) near the emitters. Heat from the LED modules plus a 35 °C ambient approaches PLA's about 55 °C softening point.
- Pi + AI HAT+ 2 inside a closed FDM body: the HAT is rated only to 50 °C ambient and the Hailo-10H reached 68.1 °C on an open bench (Sixfab). Insufficient venting will cause throttling.
- Many numbers rest on retailer listings (LED module outlines, MINI560/D4184 sizes, PD-trigger chip identity). Caliper-check the parts you actually own or buy before freezing the CAD.

## Open questions

- Which display will be used? It sets 1.2-2 W of load and the bay geometry (Touch Display 2 5-inch 91.5 x 143.4 mm vs Waveshare DSI or HDMI).
- What are the exact outline, screw-tab spacing, lead length, power and beam angle of the owned blue PickTheDeal modules, and of the white version to be bought (0.72 vs 1.2 vs 1.5 W)?
- Which bank will the team buy (Xiaomi Pocket Pro 33 W PB1030IN recommended)? Can Anker A1259 or A1689 be bought in Bengaluru with the same PDO list?
- Will Llama (ExecuTorch on the CPU) run at the same time as Hailo vision and the lights? That decides whether the 15.6 W (5 V) / about 26 W (bank) peak is real.
- GPIO 5 V feed via the HAT passthrough header, or a USB-C pigtail? This affects bay layout and the cable exit near the Pi's USB-C edge.
- How many independent light channels are needed: 4 (side x type) or 12 (each strip), for directional and photometric-stereo use? This sets the size of the MOSFET board.
- Measured Vf spread of the 50 owned 5 mm UV LEDs at 20 mA. Is it tight enough for 3-series + 150 ohm, or should the strips use 3-parallel + 470 ohm?
