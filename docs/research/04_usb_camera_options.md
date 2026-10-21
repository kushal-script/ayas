# USB camera options for the Pi 5 (research saved 20 Oct 2026)

Status: COMPLETE. These are the findings of the camera research on 20 Oct, plus follow-up checks. Confidence: **H** = read on the page; **M** = search snippet of a vendor or retailer page; **L** = third-party snippet or forum.

## Requirements used
- Plug-and-play USB (UVC) on Linux / Raspberry Pi 5, no vendor driver.
- Software control of exposure, gain, white balance and focus, locked via `v4l2-ctl`. Auto modes must switch off.
- Colour, 5 MP or more, a low-distortion lens (not fisheye), USB 3 preferred, uncompressed YUYV preferred over MJPEG.
- User's added constraints:
  - fully software-controlled (no manual focus ring)
  - Pi 5 + AI HAT+ 26 TOPS compatible
  - about ₹10k or less if possible
  - buyable in **Bengaluru**, genuine brand-name parts only

## Shortlist

| Camera | Key specs | Software control | Price / where (India) | Notes |
|---|---|---|---|---|
| **Arducam B0471: 16 MP IMX519, motorised focus, USB 3** (user is considering this) | 16 MP; uncompressed YUY2 at 9 fps full res; 1080p60; 78° diagonal field of view | Exposure, white balance and focus (manual) listed [H, Fab.to.Lab spec table] | **₹13,186** headline (₹15,006 struck through; GST inclusion not stated); +₹1,267 with enclosure; **In Stock** at Fab.to.Lab, Bengaluru (checked 20 Oct) [H] | The VCM (motorised) lens sags with gravity when tilted: store a focus value per orientation; don't autofocus on plain metal |
| Arducam B0477: 20 MP IMX283 1", USB 3 | 5472×3648, YUY2 at 9 fps; 1080p60; **16 mm C-mount lens, F1.4–F16 iris**, manual focus ring | Exposure and white balance manual listed; focus is manual on the lens [H] | ₹26,625, In Stock at Fab.to.Lab [H] | Best optics (aperture for depth of field), but focus is not software-controlled |
| Arducam B0510: 5 MP AR0521, USB 3 | YUY2 at 30 fps full res; 1080p60; M12 3.6 mm F3.0 95° "low distortion" | Exposure and white balance manual listed [H] | ₹12,425, In Stock at Fab.to.Lab [H] | Swap to an 8–12 mm M12 lens (₹887–2,662 at Fab.to.Lab). Fixed aperture |
| e-con See3CAM_50CUG: 5 MP Sony IMX264, **global shutter**, C-mount, trigger/strobe | 2448×2048 YUY2 at 35 fps; EMVA data published | Exposure and white balance listed as UVC controls [H] | USD 349 sample price (e-con webstore); e-con HQ in Chennai. Ask for an INR quote | Pick this for continuous video / moving capture |
| Logitech C920 | 1080p, USB 2; focus_absolute 0–250 (steps of 5), exposure_time_absolute, white_balance_temperature [L] | Full software control; firmware drops exposure set before streaming (kernel workaround) [M] | ₹7,100–9,000 [L]; Croma / Reliance Digital / Vijay Sales (Bengaluru stores); Amazon.in only from Logitech or an authorised seller | Lower detail (about 0.1 mm/px at about 15 cm, my estimate); USB vendor ID 046d |
| Arducam 12 MP IMX708 motorised focus, USB 3 (B0474) | 12 MP, 4608×2592, 75° diagonal field of view [M] | Focus listed as motorised [M] | About $130–145 (Arducam) [M] | Likely over ₹10k in India |

## Avoid
- **Waveshare IMX335 5MP USB (B):** 175° fisheye, distortion < −36 %, USB 2, runs hot; fixed focus; manual exposure not stated [H, Waveshare wiki].
- **Arducam B0459 (IMX477P USB):** exposure missing from the UVC control list [H].
- **Logitech Brio / C920 / IMX519 (when tilted):** the autofocus lens shifts with gravity. Use a fixed focus value per orientation.
- **Arducam IMX298 / IMX179 / IMX219 autofocus USB 2:** full resolution is MJPEG only (compression artefacts).
- **ELP:** inconsistent listings [L].

## Pitfalls (v4l2 / Pi 5)
1. **Exposure units:** 1 = 100 µs, and exposure is capped by the frame interval. Set the frame rate first. `auto_exposure=1` means manual. Set `exposure_dynamic_framerate=0`.
2. **Control names vary** (old: `exposure_auto` / `exposure_absolute`; new: `auto_exposure` / `exposure_time_absolute`). Always run `v4l2-ctl -d /dev/videoN -L`.
3. **Settings can be lost when streaming starts.** Set them after the stream starts, read back with `-C`, and monitor a grey patch.
4. **Product pages sometimes list controls the camera doesn't have.** Ask the seller for `v4l2-ctl --list-ctrls` and `--list-formats-ext` output before paying.
5. **Use YUYV, not MJPEG,** for pixel comparisons.
6. **USB:** a blue USB 3 port directly, no hub. 20 MP YUY2 at 9 fps is about 2.9 Gbit/s (calc), within the port's 5 Gbit/s.
7. **Pi 5 USB power:** 600 mA total by default, 1.6 A with a 5 A supply. The cameras draw 1.1–1.4 W. Never power LED strips from Pi USB.
8. **Moving cable:** use a high-flex USB 3 cable with strain relief.
9. **Pi 5 confirmation:** none of these cameras has a published Raspberry Pi 5 test. They are UVC "no driver" devices, so they should work, but test on day one.

## Day-one test
```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 -L
v4l2-ctl -d /dev/video0 -c auto_exposure=1 -c exposure_time_absolute=300 -c white_balance_automatic=0
v4l2-ctl -d /dev/video0 -C exposure_time_absolute
```
Then capture 20 frames of a still plate and check the brightness doesn't drift.

## Bengaluru sourcing (checked 20 Oct 2026)

| Seller | Details |
|---|---|
| **Fab.to.Lab** | 192/24, 2nd Floor, Dommasandra, Sarjapur Road, Bangalore 562125 (unit of ReCreational Hardware Systems LLP)<br>Sales +91-8050032228; warehouse +91-9110845041; tech +91-6361942383 (WhatsApp)<br>Mon–Fri 09:30–18:30, Sat 09:30–15:00<br>In-stock items ship within 8 business hours; transit 1–3 days; no pickup mentioned (ask) |
| **Probots** | GF & 1F, #530, Sankalpa, 5th Cross Rd, Vinayaka Layout, Nagarabhavi 9th Block, 2nd Stage, Bangalore 560072<br>**Store pickup:** Mr. Bharath, +91 9364881670; support +91 8929839098<br>Prices include GST<br>Camera listings couldn't be read on 20 Oct |
| Robocraze | Search pages not readable on 20 Oct |
| Croma / Reliance Digital | Pages blocked automated checks; check pincode stock or call |

**Authorised Arducam distributor in India: not confirmed.**
- Arducam's [distributors page](https://www.arducam.com/distributors/) lists no India distributor.
- [ThinkRobotics](https://thinkrobotics.com/es/collections/arducam-official-page) says it is an "official distributor".
- [Zbotic](https://zbotic.in/shop/brand/arducam/) says it is an "Arducam stockist".
- Ask Fab.to.Lab to confirm original stock and give a GST invoice.

**Checking a part is genuine:**
- **Arducam:** Arducam logo and model number on the board and box; the expected v4l2 controls; Arducam support can confirm from the invoice.
- **Logitech:** sealed box; the serial number on the box matches the camera; warranty registration; `lsusb` shows vendor ID 046d.

## Sources
- Fab.to.Lab:
  - [B0471](https://www.fabtolab.com/arducam-b0471-16mp-imx519-motorized-focus-usb-3-camera-module)
  - [B0477](https://www.fabtolab.com/arducam-b0477-20mp-usb-3-camera-module-16mm-c-mount-lens)
  - [B0510](https://www.fabtolab.com/arducam-b0510-b0510c-5mp-ar0521-color-usb3-camera-module)
  - [B0459](https://www.fabtolab.com/arducam-b0459-12mp-usb-3-camera-m12-manual-focus-lens)
  - [lenses](https://www.fabtolab.com/optical-lens)
  - [contact](https://www.fabtolab.com/index.php?route=information/contact)
  - [shipping](https://www.fabtolab.com/fabtolab-shipping-and-delivery-info)
- Arducam:
  - [B0477 datasheet](https://www.arducam.com/downloads/datasheet/B0477_20MP_IMX283_USB3.0_Camera_Datasheet.pdf)
  - [IMX519 motorised USB 3](https://www.arducam.com/product/arducam-16mp-imx519-motorized-focus-usb-3-0-camera-module/)
  - [IMX708 motorised USB 3](https://www.arducam.com/product/arducam-12mp-imx708-motorized-focus-usb-3-0-camera-module-b0474/)
  - [B0510](https://www.arducam.com/blog/product/arducam-5mp-ar0521-color-usb-3-0-camera-module/)
- e-con:
  - [See3CAM_50CUG](https://www.e-consystems.com/sony-imx264-camera.asp)
  - [webstore](https://e-consystems.com/webstore.asp)
  - [USB 3 cameras on Pi 4](https://www.e-consystems.com/blog/camera/amp/page/51/)
- Waveshare: [IMX335 (B) wiki](https://www.waveshare.com/wiki/IMX335_5MP_USB_Camera_(B))
- Kernel: [V4L2 camera controls](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html)
- Probots: [contact](https://probots.co.in/contact)
- Raspberry Pi: [Pi 5 product brief (USB power)](https://cdn.sparkfun.com/assets/9/9/0/d/e/raspberry-pi-5-product-brief.pdf)
