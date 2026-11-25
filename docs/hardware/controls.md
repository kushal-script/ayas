# Ayas handheld: controls, power and data (Phase 2)

The physical controls cover what an operator does with both hands on the grips, often in gloves. Review tasks stay on the touchscreen: Why?, How to fix?, 3D, Acknowledge and Shift summary. The mapping lives in `configs/app.yaml` (`controls`, `power`, `export`), so it can change without a code change.

## Buttons

| Control | Where | Press | Hold ≥ 1 s |
|---|---|---|---|
| **Triggers** ×2 | front of each grip, index fingers | **Inspect**: 5-frame burst, then the verdict. Both triggers within 300 ms are needed, which forces a steady two-handed capture. | |
| **Left, upper** | back of the left arm root, thumb | **Scan part ID**: the camera reads the QR / DataMatrix / barcode on the part or job card. Every later inspection is tagged with it. | Clear part ID |
| **Left, lower** | | **Light mode**: Standard burst → Hood 3D ring → UV check (UV only with the hood on; also interlocked in hardware) | **Aim light**: a steady low white light to position the device |
| **Right, upper** | back of the right arm root | **Next region**: R1, R2, … for plates larger than the 100 × 100 mm field, with a plate map on screen so coverage is recorded | Back one region |
| **Right, lower** | | **Mark**: the operator's own flag on this capture; saved and sent to QA even if the verdict is PASS | **Retake** |
| **Power** | top of the body, right; 12 mm with ring LED | Off: **power on**. On: no action (avoids accidental presses). | **2 s: save and shut down**. **10 s: forced off.** |

- **Feedback:** 1 beep per press and 2 per hold, from the buzzer, so the operator keeps their eyes on the part. The power ring LED is lit while the device is on.

## Power sequence (POWER board, `hardware/pcb/design.py`)

1. **Off.** The bank's USB-C CC lines reach nothing, because the CC router (TS5A23157) is unpowered. The bank sees no device and its output is off, so there is no standby drain.
2. **Press power.**
   - The button connects 5.1 kΩ Rd to both bank CC lines (through BAT54C diodes). The bank sees a device and switches its 5 V on.
   - The bank's 3.3 V always-on rail (LM2936) powers BSS84 Q2, which drives DEVICE_MODE high.
   - The router connects the bank's CC lines to the STUSB4500, which negotiates 12 V.
   - The 12 V rail starts the 5 V buck, then the Pi and the RP2040.
3. **RP2040 starts** (under 1 s). It drives HOLD (GPIO24), so DEVICE_MODE stays high after the button is released. The Pi boots into the app in about 20–25 s.
4. **Hold power 2 s.**
   - The RP2040 (it reads the button on GPIO23) tells the app to finish writing the current record and shut down the Pi.
   - When the Pi's USB 5 V drops (PI_ON, GPIO22), or after `pi_off_timeout_s`, the RP2040 releases HOLD.
   - DEVICE_MODE goes low and the router loses its supply. The bank sees a detach and turns off.
5. **Hold power 10 s.** The RP2040 releases HOLD at once (forced off). If its firmware hangs, the hardware watchdog resets it; that also releases HOLD, so a fault ends in off, never stuck on.

## Charging (charge-through port)

- **Port.** A USB-C port on the right side wall, by the battery bay: the CHARGE board, cabled to POWER J6. Use a normal USB-C (PD) charger.
- **Device off.**
  - The port's Rd makes the charger supply 5 V. That powers the router from its own 3.3 V rail (3V3_CHG).
  - The router connects the charger's CC lines straight to the bank's, so they negotiate PD fast charging directly.
  - Back-to-back P-FETs (Q3, Q4) pass VBUS. A 10 V Zener limits their gate voltage at a 20 V contract.
- **Device on.** Charge-through is blocked (Q6). Run the device, or charge it off; the bank cannot charge and power the device at once on a single USB-C port.

## Saving and exporting readings

- **Automatic.** Every inspection is saved the moment the verdict appears, with no save step:
  - the record (`ayas.db` SQLite + `inspections.jsonl`);
  - the burst images and the 3D map;
  - the Llama note.
  
  These are tagged with the part ID, region, light mode, operator mark and time, and stored on the Pi's microSD (`log_dir`, `data/app/`).
- **Export.** Plug a pen drive into the USB-A port on the top of the body, left (a panel-mount extension from a Pi USB port). Tap **Export** and choose today / this shift / all. The app writes CSV + PDF report + images (`export` in `app.yaml`).

## RP2040 firmware (to be written)

- **USB CDC serial to the Pi.**
  - Events: `BTN trigger press`, `BTN thumb_lu hold`, `PWR off_request`, `HOOD 0|1`, `TEMP <ntc_l> <ntc_r> <board>`.
  - Commands from the Pi: light set-points, burst steps, buzzer, fan, `PWR halt_ok`.
- **GPIO map.**
  - Light channels: GPIO0–9.
  - Triggers and thumbs: GPIO10–15.
  - HOOD_N GPIO16, FAN GPIO17, BUZ GPIO18, ALERT GPIO19, I2C GPIO20/21.
  - PI_ON GPIO22, BTN GPIO23, HOLD GPIO24.
  - NTCs: ADC0–2.
- **HOLD is asserted first thing in `main()`, then the watchdog is armed.**
