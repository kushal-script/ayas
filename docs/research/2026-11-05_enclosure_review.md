# Independent reviews of the handheld enclosure (5 Nov 2026)

These reviews were of the first export, which was 80 mm deep and had a simplified heatsink. Every blocker and major item, and most minor items, are folded into `hardware/enclosure/trinetra_handheld_fusion.py`. The README lists what is still open.

---

## Lens: assembly

**TriNetra handheld enclosure: mechanical, assembly and print review**

Two problems stop assembly. The HAT clamp bosses land on the AI HAT+ 2 heatsink push pins, so the FrontCentre cannot close. The CamHood screws cannot be put in or driven. I checked every number by ray-casting the exported STLs (scripts in (local working files, not in the repo)). Positions use `pi_usb_dir=+1`.

**Assembly walk-through**
1. **Inserts:** the lid, pod, wing and tab inserts can all be pressed in. The shelf inserts cannot be pressed in straight (M4).
2. **Display into the BackLid:** the glass fits, with 0.85 mm to the lid screw columns. The only thing holding the display forward is the HAT clamp, which fails (B1).
3. **Pi 5 onto the display standoffs:** the same four holes are needed twice (M1). Plug the display cable and the J1 power lead into the display first, because its connector sits under the Pi.
4. **Flat cables:** both must be plugged into the Pi before the HAT goes on.
   - The Pi 5's CAM/DISP connectors are 47.2–56.4 mm from the non-USB end and 0.5–16.2 mm from the HDMI edge, i.e. X 5.9–15.1, Y −13.5–2.2 in the enclosure ([Pi 5 drawing](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-mechanical-drawing.pdf)). They sit directly under the HAT's 10×16 mm cable slot (X 6.2–16.2).
   - Camera cable: the path is about 70 mm. The 200 mm cable reaches, with about 130 mm of slack to fold behind the camera.
   - Display cable: the path is about 120 mm. It leaves the display toward −X at about X −42, wraps around the Pi's −X end outside the PCIe cable loop and SD card, runs over the heatsink, then down the slot. The length of the cable supplied with the display is unknown, so buy the 200 mm Pi 5 display cable.
5. **HAT onto its header:** fits.
6. **Camera onto its bosses:** M2 screws go in from the open back and can be reached.
7. **FrontCentre over the Pi stack:** fails (B1).
8. **LED pods:** fit (screw lengths in m5).
9. **Power bank into the bay:** fits (m6).
10. **Wings onto the centre and halves together:** all screws can be reached, but see M5 and m1.
11. **Lid:** all 4 screws can be reached (M2).
12. **CamHood:** fails (B2).

## BLOCKER

**B1. `detail_front_centre` HAT clamp bosses hit the heatsink push pins, and the dummy that should catch it is wrong**
- **What is wrong:** the real heatsink is 64×42.5 mm, with fins 3.2 mm tall. Its two push pins sit 6.0 mm inboard of the top-right and bottom-left HAT holes. Each pin head is 5.6 mm across and stands about 10.5 mm above the HAT ([AI HAT+ 2 product brief](https://www.raspberrypi.com/products/ai-hat-plus-2/), p.3). In the enclosure that puts them at (20.21, 32.5) and (−37.79, −4.5).
- **Numbers:** the bosses are 8 mm across, so each one overlaps a pin head by 0.8 mm. The boss tip at Z −2.7 meets a pin top at about Z +6.8, so the FrontCentre stops about 9.5 mm before it closes.
- **Other clashes:**
  - The boss at (−37.79, 38.5) hits the display's J1 power plug on GPIO pins 2 and 6: the plug comes 3.6 mm from the boss centre, inside its 4 mm radius.
  - The two top bosses clear the stacking-header pins by only 0.55 mm.
- **Why it was missed:** `hat_heatsink=(42.5,42.5,14)` is modelled as a centred block, which misses both pins, so the fit check passed.
- **Fix:**
  - Delete the `hat_points` boss loop.
  - Hold the display on its four case-mount bracket blocks with posts from the front wall. Put them at `(sx*(gw/2-19.85), cy±25.5)`, i.e. (±51.85, −11.5/39.5). Run each from Z `z_glass+10.4+1.0` (−27.4) to `zf_in+0.5`, 7 mm across, with a 1 mm foam pad on the tip. This leaves 1.6 mm to the Ethernet jack.
  - Also put VHB tape on the 13.7–15.5 mm overlap of the bezel lip.
  - Remodel the heatsink dummy as 64×42.5×4 mm plus the two pins (5.6 mm across, 10.5 mm tall).

**B2. `build_cam_hood`: the hood screws cannot be fitted**
- **What is wrong:** the cone flares out over the screw holes at X ±13.5. Directly above each hole the cone wall fills Z 48.05–50.47. There is only 6.0 mm between the flange top and the cone, and 3.7 mm at the edge of the screw head. An M2 screw cannot be dropped in, and no driver can reach it.
- **Why it matters:** the hood cannot be removed and refitted, and it is the cover you take off to focus the camera.
- **Fix:**
  - Move the screws (and the matching pilot bosses in `detail_front_centre`) to `(0, cy±20)`. That is outside the cone tip radius of 17.46 mm, with 1 mm left for the driver shaft.
  - Make the flange a 26×48 mm stadium.
  - Change the front vent limits from `cy±19` to `cy±24`; the top screw would otherwise land in a vent slot.

## MAJOR

**M1. The Pi's four holes are needed twice.** The display's standoffs take M2.5 screws through those holes. The kit spacers screw in from the Pi's underside through the same holes ([official install drawing](https://github.com/raspberrypi/documentation/tree/master/documentation/asciidoc/accessories/ai-hat-plus/images)). Fix: use 4× M2.5 male-female 16 mm standoffs, with at least 4 mm of male thread, screwed into the display standoffs. Check the thread depth in the display standoffs before buying.

**M2. The lid insert bosses (`lid_screw_points` in FrontCentre) are barely attached.** At X 76.6, a 7.6 mm boss overlaps the side wall by only 0.6 mm. That joins it along a 4.1×12 mm strip. Its Z −18 end is an unsupported overhang 58 mm above the bed. All four lid screws pull on these bosses. Fix: run each boss all the way to the front wall: `g.cyl((x,y,zs),(x,y,zf_in+0.5),...)`. Nothing is in the way; the nearest tab is 0.6 mm away.

**M3. HAT stack and clamp tolerances.** These are resolved by B1, because the clamp is removed.

**M4. Shelf electronics bosses (`elec_points`) cannot take inserts.**
- The insert axis is along Y. There is only 90 mm of straight room (Y −24.8 to 65.8) and the only opening is the back, so a soldering iron cannot line up.
- They print as sideways cantilevered cylinders.
- The cable slot through the shelf eats into the base of the (+62, Z 0) boss.
- Fix:
  - Use 2.5 mm pilot holes for M3 thread-forming screws; a stubby driver fits.
  - Move the cable slot to `bay_x1-30 … bay_x1-16`.

**M5. The grip-end wing bosses (`wing_screw_points` y=±64.2) break out of the rounded edges.**
- WingFront: the boss top at Z 16.30 sits outside the edge rounding (surface at 15.40), leaving a 0.9 mm bump.
- WingBack: the screw-head counterbore floor (−14.8) is above the rounded surface (−16.24), so the counterbore opens out through the side.
- Fix: use `grip_y_top-9.0` (±61).

## MINOR

**m1. Thumb buttons (`thumb_button_points` x=98).** The 12 mm button's nut (about 15 mm across flats, 17.3 across corners) overhangs the x=91 split line by up to 1.7 mm. It would hit the FrontCentre's back skin when WingBack is pushed on along +Z. The lower button also sits under the front-tab insert at (97, 7), leaving only 25.6 mm of depth. Fix: x=108.

**m2. Tab fit.**
- The top and bottom tabs touch the wing inner walls with zero clearance. Shrink their wall-facing faces by 0.3 mm.
- The tab screw holes are 0.3 mm from the wing split face, a one-layer sliver. Set `z_split_wing=-4`.
- The front-tab countersink leaves only 0.4 mm of skin.
- Joint strength under hand loads is not the issue (about 5 N per grip, under 1 N·m).

**m3. LED pods have only 2 screws, both at y=−22.** The pod top is 84 mm from them, so a 120 mm PETG pod will lift or warp at the top. Fix: add a 3 mm tongue on the pod back at y 54–58 that drops into the existing top wire window.

**m4. Focus access is unconfirmed.** The lens sits 1.0 mm below the front face inside a 14 mm hole. Neither the research nor my web search found the size of the supplied lens tool ([Waveshare listing](https://www.waveshare.com/raspberry-pi-ai-camera.htm)). Fix: measure the tool, then set `tool_hole_d` = tool OD + 1 and `cam_set_back` ≤ 0.5.

**m5. Screw lengths (longer ones bottom out in the blind holes).**

| Joint | Screw | Do not use |
|---|---|---|
| Pods | M3×8 | M3×10 |
| Lid | M3×12 | M3×14 |
| Wing halves | M3×18 | M3×20 |
| Tabs | M3×8 | — |

**m6. Power bank.**
- The Xiaomi bank is loose by 11.7 mm in Z and about 16 mm in X even with the 20 mm plug, so it needs foam or a spacer.
- The USB charge slot (X 45–61, bottom wall) does not line up with any bank port. Either add a panel-mount USB-C pigtail or delete the slot.

**m7. Lid seal.** All 4 lid screws are at the sides, and the bottom edge of the lid is 52 mm from the nearest one on a plain butt joint. Add a 1.2×2 mm locating lip around the split.

**m8. Print orientation and supports.**

| Part | Orientation | Supports / notes |
|---|---|---|
| FrontCentre | Front face down | 202 mm wide, needs a bed of at least 220 mm. Supports from the plate under the arm stubs and tabs (X 82–101, 22–31 mm tall). Use a teardrop for the switch hole. Change the r=4 bed-side edge rounding to a chamfer. |
| BackLid | Display face down | None. |
| WingFront / WingBack | Split face down | Tree supports from the plate for the roof; they come out through the open face. |
| LedPod | Angled face down | None. |
| CamHood | Flange down | None. |

The M3 insert bosses (4.0 mm hole, 7.6 mm OD) leave a 1.8 mm wall. That meets the CNC Kitchen/ruthex minimum of 1.6 mm.

---

## Lens: function

**TriNetra handheld enclosure review: device function and requirements (trinetra_handheld_fusion.py, out/*.stl)**

The part counts match the request: per side, 3 white modules and 3 UV strips of 3 LEDs, all facing +Z. The display faces −Z, the camera is on the display centre, and there are two side grips.

I checked these and they are fine:
- **Toe-in:** 14° vs atan(49/200)=13.8°.
- **Camera view:** nothing enters the camera field of view.
- **Hood vignetting:** the 42.15° cone clears the diagonal +3° tolerance, with 0.9 mm to spare for a pupil 1 mm behind the lens.
- **Direct light into the lens:** no LED has a direct line to the lens. Every emitter is more than 60° off the lens axis.
- **Vent areas:** 697 mm² in and 1152 mm² out, both above the research minimum of 600 mm².
- **Centre of mass:** estimated Z ≈ +2 mm, i.e. on the grip axis.

The problems are below.

---

## BLOCKER

**B1. The LED pods cannot be assembled or wired as printed (`build_led_pod`, `pod_rows`, `detail_front_centre`).**
- **UV board does not fit.** The UV cavity height is `8 + 2·uv_pitch = 28.32 mm`; a ray-cast on LedPod_L.stl shows it spans y −54.32 to −26.0. The specified perfboard (power_lights research: 12 holes, 30.5 × 10.2 mm) overruns it by 2.2 mm. The reference dummies model only the LED cylinders, so the "no interference" check could not catch this.
- **White module wires have nowhere to go.** The research says these modules bring their wires out of both end faces. The only wire holes are Ø4 at `ytop−4` (y = 56), which is 4.4 mm inside the 78.8 mm pocket, so they sit under the module body. There is no exit at the bottom end (y ≈ −18) and no space for the 70–90 mm leads between modules. The top end of the pocket (y 60.4) is 1.6 mm from the pod edge, so the wires would have to run over the outside of the pod with no way into the body.
- **Fix:** about 8 mm more pod height is needed.
  - Set `pod_margin` 2 → −2. The pod becomes y −62..66 and hangs slightly over the 4 mm edge fillet, which is fine for a separate part.
  - In `pod_rows`:
    - `ytop = pod_y1 − 4.4`
    - `y_screw = y_wb − 6.9`
    - `y_uv = y_wb − 10.8`
  - **White wiring:**
    - Replace the Ø4 holes with a 3 × (white_w+0.8) slot through the skin beyond each pocket end: ytop+0.4…+3.4 and y_wb−3.4…−0.4.
    - Make the rear hollow `white_l + 6` tall so it covers both slots.
    - Add a matching front-wall window at y_wb−2.6…y_wb+3.
  - **UV board:**
    - Use `uv_bot = y_uv − 31.3`.
    - Centre the LEDs on the board at `y_uv − 5.49 − j·10.16`.
    - Update the UV check in `checks()`.
  - Stopgap without the redesign: cut the boards to 11 holes (27.94 mm).

---

## MAJOR

**M1. The power bank cannot be charged or swapped without taking the device apart (bay, `usb_slot`, `z_split_centre`).**
- **Removal means lifting out the electronics.** The bank spans z −31.95..31.95, which crosses the lid split at −30. The only way out is to remove the BackLid. The lid holds the display, the Pi + HAT hang off the display standoffs, and the stack is held only by the front-wall clamp bosses. So every swap lifts the whole stack out on its cables. The research puts runtime at 1.1–2 h, so this would happen every session.
- **No charge path.** The bank's single USB-C port (charge in and power out) is taken by the right-angle plug, and USB-A tops out at 12 V 1.67 A. `usb_slot` (x 45–61, bottom face) does not line up with any port; the bank's port face is at x = 28.
- **The bank rattles.** The Xiaomi has 36 mm of play in X (126 mm bay) and 11.7 mm in Z, so its 199 g lands on the plug.
- **Fix:**
  - Add a BayDoor part: split the shell at y = −58 for |x| ≤ 77, giving a 154 × 80 × 6 tray that includes both lid halves of the bottom. Fix it with 4 M3 screws into inserts in bosses on the bay end stops (x ±64, z ±24), and start the end stops at y −57.8.
  - Add a 16 mm spacer at the −X end, and Z ribs at ±32.5.
  - Delete `usb_slot`.
  - Charge the bank through the open door or outside the device.

**M2. The UV is aimed about 54 mm below the image centre (`build_led_pod`).** The UV rows sit 44–64 mm below the lens axis and have no vertical tilt. Modelling the 30° LEDs at 190 mm, the image centre gets 34–59% of peak and 50 mm above centre gets 5–12%. The top of the frame is effectively dark, which matters for fluorescence capture.
- **Fix:** build the UV holes and cavity in a sub-frame rotated about the pod's `ex` axis by atan((cy − y_uv_mid)/190) ≈ 16°, LEDs pointing up. Add a wedge under the UV block that is 0 mm at its top and about 9 mm at `uv_bot`.

**M3. Heat cannot get out (vents in `detail_front_centre`).** These figures are estimates:
- Natural flow through the vents (596 mm² effective, 50 mm stack) is about 0.08 L/s, which removes about 1.4 W at a 15 K rise.
- The walls (about 0.08 m², U ≈ 3–4.5 W/m²K) remove about 4–5.5 W.
- The total is about 6–7 W against the research's 11–16 W. Internal air would reach roughly 50–60 °C at 35 °C ambient, above the AI HAT+ 2's 50 °C rating.
- The ergonomics research rule "30 x 30 x 7 mm pocket … optional 5 V exhaust fan" is not implemented. The bank bay is also sealed.
- **Fix:**
  - Add 4 bosses on the underside of the top wall at (pi_cx ± 12, z = 2 ± 12) for a 30 × 30 × 7 mm 5 V fan exhausting through the top slots. About 1 L/s keeps the rise to 11 K or less at 13 W.
  - Run the fan from a fifth MOSFET channel.
  - Put about 600 mm² of slots in the BayDoor (|x| < 50).

**M4. The hood screws cannot be driven (`build_cam_hood`).** The M2 screws at x = ±13.5 sit under the cone, whose outer radius at the tip is 17.46. A ray-cast on CamHood.stl shows the screw axis passes through cone wall at z 48.1–50.5, so a straight screwdriver cannot reach. Only a ball-end hex tilted 22° or more works. The hood must come off to refocus the lens.
- **Fix:**
  - Move the screws to (0, cy ± 21) on two lugs (|x| ≤ 6, r 23.5) and move the matching pilot bosses.
  - Move the intake slots to y ≤ cy−25 and ≥ cy+25 using `range(−4, 5)`. That gives 9 slots and 680 mm².

**M5. The touchscreen cannot be reached while holding the grips (`slab_t`, `body_z_back`).**
- The display glass (z −38.8) sits 22 mm behind the arm's back face (z −18).
- The active-area edge is 64.8 mm in from the grip axis (x 120), beyond the research's estimated 50–65 mm thumb reach, and the thumb would also have to climb over the body side wall.
- The thumb buttons at x 98 are 22 mm in from the grip axis; the research recommends 30–45 mm.
- **Fix:**
  - Make the physical buttons the main controls, with on-screen controls only in the outer 15 mm.
  - Add `slab_zc = −6`. Extrude the slab from −24 to 12, set `z_split_wing = slab_zc − 2`, and offset every ±sz / ±9 / sz_in reference in the wing and tab code. This cuts the step to 16 mm and leaves the centre of mass about 8 mm in front of the grip axis.

---

## MINOR

1. **Power electronics have nowhere proper to mount (`elec_points`).**
   - **Mounting:** each side has 2 M3 bosses 24 mm apart, which matches none of the listed modules. Pololu bucks use M2 holes, the trigger is 19 × 10 mm with no holes, and the MOSFETs go on perfboard.
   - **Missing items:** there is no space reserved for the fuse, the 470–1000 µF capacitor, the fallback 15→12 V buck, or the Pi J2 power button.
   - **Switch rating:** typical 12 mm latching switches are rated about 2 A against a 2.2 A all-on peak.
   - **Room is there:** about 34 × 56 mm each side (x 45–80, z −28..28).
   - **Fix:**
     - Use 4 bosses per side at (±49.5/±74.5, z −22/24) for a 30 × 50 perfboard carrier.
     - Add a Ø7 hole on top for the J2 button.
     - Use a switch rated 3 A or more, or switch a P-MOSFET instead.
2. **UV LEDs stick out and are poorly located.** The domes stand 5.7 mm proud (8.7 − 1 − `uv_skin`); the research says recess them 3–5 mm. The flange stop is only 0.25 mm per side.
   - **Fix:** set `uv_skin=3` with a Ø6.3 × 1.2 counterbore from the back, and add a black frame 9 mm tall with 1.6 mm walls around each UV block.
   - Add a UV warning label and allow UV only while the trigger is held.
3. **Hood rim is below the LEDs.** The rim is at z 53 and the LED tops at 59–61, against the research rule of 3 mm or more above every LED. There is no direct path into the lens, but the outer LEDs light the inner hood wall, which can cause flare.
   - **Fix:** use matte black and add 2–3 internal 0.6 mm baffle rings.
4. **White modules sit directly on 2 mm PETG** with dead air behind, at 1.2–1.5 W each.
   - **Fix:** set `white_pocket` 0.8 → 1.8 for a 1 mm aluminium strip under each module.
5. **Mass is slightly over target:** about 1.03 kg estimated against the 1.0 kg target. Going from 2.2 to 2.0 mm walls on the wings and BackLid saves about 25 g.

Sources:
- `<repo>/docs/research/2026-11-05_enclosure_power_lights.md`
- `<repo>/docs/research/2026-11-05_enclosure_ergonomics_printing.md`
- `<repo>/docs/research/2026-11-05_enclosure_camera_display.md`

Analysis scripts: (local working files, not in the repo) (stl.py, pod.py, com.py)
