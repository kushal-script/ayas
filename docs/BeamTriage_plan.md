# BeamTriage: recommendation report
*Bharat AI-SoC Challenge 2026-27, PS5. Prepared 10 Oct 2026. All prices are approximate as of 10 Oct 2026.*

---
## 1. Which AI HAT to buy

**Buy the Raspberry Pi AI HAT+ 26 TOPS (Hailo-8).** Do not buy the AI HAT+ 2.

| Option | Chip | INT8 TOPS | On-board RAM | Vision speed on a Pi 5 | Toolchain maturity (Pi) | USD | INR (10 Oct 2026) |
|---|---|---|---|---|---|---|---|
| **AI HAT+ 26T (pick)** | Hailo-8 | 26 | none (on-die) | YOLOv8m ~50 FPS[^ce]; MobileNetV2 2,396 / DeepLabV3 122 FPS async[^sx]; YOLOv8s ~491 FPS in hailortcli[^hc] | **Mature.** `hailo-all` (HailoRT 4.23.0 in apt), DFC 3.33/3.34, Model Zoo v2.18/2.19, Ultralytics `format="hailo"` | $110 list ($132 SparkFun) | ₹9,950 (Silverline, in stock) to ₹11,309 (Robocraze) |
| AI HAT+ 13T | Hailo-8L | 13 | none | YOLOv8m ~20–25 FPS[^ce]; MobileNetV2 175, DeepLabV3 43 FPS[^sx] | Mature (same stack) | $70 | ₹6,350 (e2u, in stock) |
| AI HAT+ 2 | Hailo-10H | 20 (40 INT4) | 8 GB LPDDR4X | YOLOv8m ~45 FPS[^ce]; MobileNetV2 1,626, DeepLabV3 51; wins only on big models (YOLOv8l 40 vs 19.6)[^sx] | **Lagging.** `hailo-h10-all` stuck at 5.1.1 (upstream 5.4.0); HEF/driver mismatches reported | $200 | ₹21,400 (Silverline) |
| No HAT (AI Camera + CPU) | IMX500 + Cortex-A76 | — | — | YOLOv8m ~1 FPS on CPU; YOLO26n with NCNN ~67 ms | ExecuTorch / NCNN | $0 | ₹0 |
| AI Kit | Hailo-8L M.2 | 13 | — | — | Discontinued | — | — |

**Why the 26T:**
- Llama must run on the Arm CPU through ExecuTorch whatever you buy, because ExecuTorch has no Hailo backend. So the AI HAT+ 2's LLM features earn nothing toward the requirement. The Pi CPU also beats the 10H at LLMs anyway: 4.78 vs 2.60 tok/s for Llama 3.2 3B.
- The Hailo-8 is fastest on the single-context CNNs this design uses.
- It costs less than half the AI HAT+ 2's Indian price.
- It is the only accelerator here that robustly handles full-resolution SEM images with several models.

**When to pick something else:**
- **13T:** a very tight budget and one small model.
- **AI HAT+ 2:** large multi-context vision models, which this design does not need.
- **No HAT:** money or shipping is blocked. This is still compliant, because Hailo is optional.

**Pi RAM:** run `free -h`. 8 GB is ideal (1B or 3B). With 4 GB, use Llama 1B only. With 2 GB or less, buy an 8 GB board ($175). Skip 16 GB ($305).

**Setup checklist:**
- 27 W PSU
- Active Cooler under the HAT on the 16 mm header
- Camera ribbon fitted **before** the HAT
- Raspberry Pi OS Trixie 64-bit, `sudo apt install dkms hailo-all`, then `apt-mark hold` the packages
- Boot from microSD or a USB SSD (no NVMe)

---
## 2. What "electron microscopy" can honestly mean here

- **It is not an electron microscope.** The AI Camera resolves about 64 µm/px at 20 cm, or roughly 20–35 µm/px with a +10 D macro lens (my estimate). An SEM resolves nanometres.
- **It is a device in the SEM workflow,** in the same order fabs use: optical inspection, then e-beam review, then automatic defect classification. KLA, Applied and Hitachi sell this only to fabs; Applied reports 10–100x more e-beam review candidates at advanced nodes.
  - Before the SEM: decide which parts deserve SEM/EDS time and issue a work order.
  - Beside the SEM: read-only analysis of SEM TIFFs or HDMI frames covering image quality, defect class and size in µm from the metadata.
  - After the SEM: SEM verdicts become labels for the optical model.
- **Read-only, always.** Nothing is installed on the SEM PC, the box never controls the beam or stage, and USB is mounted read-only.
- **Real time where it is honest:** the optical stage (every frame), live SEM-screen IQ (about 10 Hz), and per-micrograph analysis in about 1–2 s, which is faster than the SEM's own slow scan.
- **The IMX500 can now take injected images.** picamera2 0.3.32 added tensor injection, but it is experimental, limited to 640x640 or smaller, and holds one model at a time. The design keeps SEM work on the Hailo.
- **SEM access is the critical path.** Book a college or I-STEM session in week 1. Caption anything from a dataset or replay.

---
## 3. Recommended product: BeamTriage

This merges TriageScope's real-time optical core with the SEM-review module from BeamSight and SEM Sidecar. The merge is strictly better than either alone: the live inspection that satisfies PS5 is ready for Phase 1, and the authentic SEM analysis is added for Phase 2. About 70% of the code is shared.

**Problem.** SEM and SEM-EDS failure analysis answers "what is this and why", but SEM time is scarce, booked and expensive.
- Floor inspection is by eye, unlogged and inconsistent.
- Parts are sent to the SEM arbitrarily, with a sticky note instead of coordinates.
- New SEM operators save defocused, astigmatic or charging images and classify defects by hand.
- Fab-grade automatic classification costs crores. Avizo's deep-learning features need Windows and an NVIDIA GPU. LLM microscope agents such as LBNL's TEM Agent use cloud models.

**Customer.**
- QA managers and FA labs in Indian electronics and OSAT: Micron Sanand ATMP (opened 28 Feb 2026), Kaynes Sanand OSAT (31 Mar 2026), CG Semi, and EMS/PCB-assembly MSMEs.
- University and I-STEM shared SEM facilities, often with XP-era, air-gapped instrument PCs.
- The buyer is whoever pays for SEM hours.

**Where it connects to the electron microscope.**
1. **Referral.** Rules flag REFER-TO-SEM for an unknown anomaly, an unidentified particle (needs EDS), a suspected crack (needs a cross-section per IPC-TM-650 2.1.1), or a trend. The box prints a QR work order and exports a USB JSON/CSV queue. Each entry carries the crop, location in mm from the fixture datum, material and a Llama-phrased imaging plan.
2. **SEM ingest.** The operator saves TIFFs under the case ID to a "SEM DROP" stick, which is auto-mounted read-only. Optionally, a point-to-point Ethernet SMB share or an HDMI splitter feeding an MS2109 USB dongle provides live frames. tifffile parses kV, working distance, magnification, detector and pixel size from FEI/Thermo tags 34680/34682, Zeiss tag 34118, or Hitachi/JEOL `.txt` sidecars. A manual scale-bar calibration covers SEMs that write no metadata.
3. **Closing the loop.** The FA verdict, picked from a dropdown, is attached to the optical image as a training label.

### Block diagram: three separate compute domains

```
                        Raspberry Pi 5 (8 GB), Raspberry Pi OS Trixie 64-bit
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ DOMAIN 1: IMX500 ON-SENSOR NPU (AI Camera, CSI/DSI port 0)                     │
 │  TinyGLASS anomaly network.rpk (MCT INT8, ≤640x640 ROI via set_inference_roi)│
 │  -> per-frame heatmap + score as CnnOutputTensor METADATA (~0 host CPU)        │
 │  -> full-res 4056x3040 @10 fps image stream                                  │
 └───────────────┬──────────────────────────────────────────────────────────────┘
                 │ Picamera2 process (venv A, core 0) ── shared memory frames ──┐
 SEM TIFF (USB ro / P2P SMB) ─► ingest proc: tifffile metadata, 16→8 bit, CLAHE,│
 SEM PC HDMI ─► splitter ─► MS2109 UVC ─► MJPEG decode, viewport crop, 10 Hz    │
 OpenCV (CPU core 0): trigger, ROI crops, 640/224 tiles, classical IQ metrics ◄─┘
                 │ crops / tiles (batched)
 ┌───────────────▼──────────────────────────────────────────────────────────────┐
 │ DOMAIN 2: HAILO-8 NPU (AI HAT+ 26T, PCIe Gen3 x1) via HailoRT 4.23 async API │
 │  one shared VDevice + scheduler (venv B, own process)                        │
 │  H1 YOLOv8s-det (optical defects) │ H2 IQ-Net ResNet-18 (SEM quality)        │
 │  H3 ResNet-18 Carinthia classifier (+ optional U-Net-MobileNetV2 masks)  .hef│
 └───────────────┬──────────────────────────────────────────────────────────────┘
                 │ detections JSON
 CPU core 0: deterministic YAML rule engine → severity / disposition
   → GPIO stack light + buzzer │ kiosk UI │ SQLite (WAL)      ◄── NEVER waits for LLM
                 │ ZeroMQ event (ipc://)
 ┌───────────────▼──────────────────────────────────────────────────────────────┐
 │ DOMAIN 3: ARM CORTEX-A76 CPU (cores 1–3, taskset, nice +5) via ExecuTorch    │
 │  1.5.1, XNNPACK + KleidiAI SDOT; venv C (numpy≥2); TextLLMRunner or llama_main│
 │  Llama 3.2 1B Instruct SpinQuant .pte (3B for SEM reports if 8 GB)           │
 │  + SQLite FTS5 SOP retrieval → operator_action / root_cause_hint /           │
 │    sem_request / FA report → validator → template fallback                   │
 └──────────────────────────────────────────────────────────────────────────────┘
```

**How the processes split:**
- The IMX500 only ever runs inference on its own sensor in this design.
- All SEM pixels go to the Hailo or the CPU.
- The Hailo never runs Llama.
- ExecuTorch never runs on the Hailo.
- The three domains run as separate processes in separate venvs, because numpy/torch versions conflict and IMX500 + Hailo in one process is unconfirmed.

### Defect taxonomy and severity rules

All rules are deterministic, live in a versioned YAML file, are tuned with an FA mentor, and are logged with every decision.

**A. Optical stage (H1 YOLOv8s on the Hailo, plus the IMX500 anomaly score)**

| Class | Default severity | Disposition |
|---|---|---|
| bent_lead | S1 if within the straightening limit, else S2 | REWORK |
| cut_or_missing_lead | S3 | SCRAP |
| damaged_case (chip or crack) | S3 | SCRAP + REFER-TO-SEM if crack-like |
| misplaced_or_rotated | S2 | REWORK |
| missing_component (PCB coupon) | S3 | REWORK / HOLD |
| solder_bridge | S2 | REWORK |
| foreign_particle | S1 for flux or dust (CLEAN); S2 above the size threshold | REFER-TO-SEM-EDS |
| scratch | S1 cosmetic below the length threshold and away from leads, else S2 | — |
| unknown_anomaly (high anomaly score, no confident class) | S2 provisional | REFER-TO-SEM |

S0 = PASS, S1 = minor (logged), S2 = major (rework or clean, then re-inspect), S3 = critical (scrap or quarantine).

Flags:
- **LOT HOLD:** 3 or more same-class S2+ defects in the last 50 parts.
- **MAINTENANCE:** the mean anomaly score of passing parts drifts upward (lens, lighting or fixture changed).

**B. SEM image quality (H2 IQ-Net plus classical metrics: Laplacian/Tenengrad focus, FFT anisotropy for astigmatism, saturation and streaks for charging, MAD noise, phase-correlation drift)**
- Classes: ok, defocus, astigmatism (with orientation), charging, low_SNR, drift.
- **Q0 OK:** measurements allowed.
- **Q1 MARGINAL:** measurements flagged.
- **Q2 RE-ACQUIRE:** defect results suppressed and marked untrusted.

**C. SEM defects (H3: Carinthia's 6 classes plus no_defect and unknown)**
- Size = mask or box extent × pixel size from the TIFF, compared with a per-layer critical dimension (CD).
- **S1 nuisance:** below 0.25 CD.
- **S2 minor:** 0.25–1 CD.
- **S3 major:** above 1 CD, or any killer class.
- **S4 excursion:** a killer class above 1 CD, N or more defects per field, or the lot is above the c-chart upper control limit.
- **unknown:** human review plus an EDS suggestion.

### Operator alerts

| Pattern | Meaning |
|---|---|
| Green steady / blinking | PASS / PASS with S1 note, or SEM Q0 |
| Amber steady | REWORK or CLEAN (S2), or SEM Q1 / S2 |
| Amber flashing | SEM RE-ACQUIRE (Q2) |
| Red + 1 s buzz | SCRAP (S3), or SEM S3/S4 |
| Amber/red alternating + 3 beeps | REFER-TO-SEM (QR work order prints) |
| Red flashing + continuous buzz | LOT HOLD / excursion until supervisor ACK |
| Amber slow blink | MAINTENANCE (lighting, lens or fixture drift) |

**Hardware:**
- 12 V 3-colour stack light with buzzer on GPIO, via gpiozero/lgpio and an opto or ULN2803 driver.
- ACK and OVERRIDE buttons. OVERRIDE forces a reason, and every override becomes a training label.

**UI (7-inch kiosk):**
- live heatmap and boxes, a disposition banner, the rule ID that fired, and size in mm/µm
- Llama advice streaming underneath, with the SOP ID and a "Built with Llama" badge
- lot Pareto, SEM queue, and SEM card (metadata, IQ gauges, astigmatism compass)

### Logging schema (SQLite, WAL mode, hash-chained rows)

- `inspections`: id, ts, part_id, lot, img_path, sha256, anomaly_score, heatmap_path, detections_json, size_mm, zone, severity, disposition, ruleset_ver, model_hashes(rpk/hef/pte), latency_ms_{imx,hailo,rules}, override, override_reason
- `sem_cases`: case_id (QR), part_id, referral_reason, work_order_text, status (open/imaged/closed), fa_verdict, closed_ts
- `sem_images`: id, case_id, path, sha256, source (usb/smb/hdmi), kV, WD_mm, mag, pixel_size_nm, detector, dwell
- `iq_results`: image_id, class, conf, focus, astig_ratio, astig_deg, sat_pct, snr, drift_px
- `sem_defects`: image_id, class, conf, bbox, mask_path, area_um2, max_dim_um, severity
- `advice`: event_id, model (1B/3B), prompt_hash, sop_ids, text, ttft_ms, decode_tps, validator_ok, fallback_used
- `alerts`: id, ts, pattern, ack_ts, ack_by
- `model_registry`: file, sha256, dfc_ver, hailort_ver, executorch_ver, mct_ver
- `system_health` (every 10 s): soc_temp, get_throttled, cpu_pct, ram_mb, npu_util

Exports: CSV per lot, a daily Pareto PNG, a per-case HTML/PDF FA report, and the SEM queue as JSON on USB. No data leaves the box.

### Physical design

- Bench station about 30 x 25 x 35 cm.
- 2020-extrusion gantry holding the AI Camera, pointing down 7–10 cm above a 3D-printed part nest that has a printed mm scale and a datum corner. Add a +10 D clip-on macro lens and a diffuse LED ring with a dome.
- Vented base with the Pi 5, Active Cooler and AI HAT+.
- Angled 7-inch Touch Display 2 (uses the second CSI/DSI port).
- Stack light on a pole, ACK and OVERRIDE buttons, an optional 58 mm thermal printer, and a front "SEM DROP" USB port.
- Rear: Ethernet labelled "point-to-point only" and USB HDMI-capture input.

### BOM (new spend; Pi 5 and AI Camera already owned)

| # | Item | USD | INR |
|---|---|---|---|
| 1 | AI HAT+ 26 TOPS (Hailo-8) | 110 | 9,950–11,309 (sourced) |
| 2 | Active Cooler | 5 | ~500* |
| 3 | Official 27 W USB-C PSU | 12 | ~1,100* |
| 4 | 64 GB A2 microSD + 256 GB USB 3 SSD (archive) | 42 | ~3,700* |
| 5 | Raspberry Pi Touch Display 2 (7-inch) | 60 | ~6,000* |
| 6 | 12 V 3-colour stack light + buzzer, ULN2803/opto board, 12 V adapter | 30 | ~2,500* |
| 7 | 2 x 22 mm buttons + wiring | 5 | ~400* |
| 8 | +10 D clip-on macro lens + printed holder | 8 | ~600* |
| 9 | LED ring + diffuser dome | 12 | ~1,000* |
| 10 | 2020 gantry, printed nest/mount, vented PETG enclosure | 35 | ~3,000* |
| 11 | 300–500 mm Pi 5 camera ribbon | 4 | ~400* |
| 12 | Demo parts (~300 TO-92 transistors, ~20 PCB coupons, solder, flux) | 12 | ~1,000* |
| 13 | SEM module: HDMI 1x2 EDID splitter + MS2109 dongle + VGA/DVI adapter | 35 | ~3,000* |
| 14 | 2 USB sticks ("SEM DROP") + Cat6 cable | 12 | ~1,000* |
| 15 | Optional 58 mm USB ESC/POS thermal printer | 35 | ~3,000* |
| 16 | x86 GPU time for training and DFC compiles (Colab or rented VM) | 15–40 | ~1,200–3,500* |
| | **Total (core, no printer)** | **~$400–425** | **~₹35,000–38,000** |

\*INR estimates are unverified; check Robu, e2u, Silverline or Aryabot. A no-HAT build costs about ₹25k. If you only own a 2 GB Pi, add an 8 GB Pi 5 ($175).

---
## 4. Models and datasets

**Models**

| Domain | Model | How to build | URL |
|---|---|---|---|
| IMX500 | TinyGLASS (ResNet-18 GLASS, MCT INT8; 94.2% AUROC on MVTec AD, 20 FPS) | Train on good parts only, then ONNX → Edge-MDT 1.3.0 / MCT → imxconv-pt → imx500-package `network.rpk` (on the Pi) | https://github.com/ETH-PBL/TinyGLASS, https://arxiv.org/abs/2603.16451 |
| IMX500 fallback | YOLO11n detection at 640 (~58.8 ms/img) | Ultralytics `format="imx"` (Linux) | https://docs.ultralytics.com/integrations/sony-imx500/ |
| Hailo H1 | YOLOv8s detection at 640 (single context), 8 optical classes | Ultralytics `format="hailo", name="hailo8"` on x86 Ubuntu, DFC 3.33, ≥1,024 in-domain calibration images | https://docs.ultralytics.com/integrations/hailo |
| Hailo H2 | IQ-Net: ResNet-18 / EfficientNet-Lite0 at 224, grey copied to 3 channels, 6 classes | Model Zoo v2.18 recipe + `.alls` script, normalisation inside the network | https://github.com/hailo-ai/hailo_model_zoo/tree/v2.18 |
| Hailo H3 | ResNet-18 Carinthia classifier; optional U-Net-MobileNetV2 at 256 (upsample+conv decoder, a16_w16 outputs) | Model Zoo | same |
| CPU fallback | YOLO26n with NCNN (~67 ms); ResNet-18 INT8 with ExecuTorch (~8.75 ms); PatchCore-R18 (Anomalib, ~1.1 s/part) | — | https://docs.ultralytics.com/guides/raspberry-pi/, https://huggingface.co/Arm/resnet-18-int8-xnnpack-executorch, https://github.com/open-edge-platform/anomalib |
| Arm CPU | Llama 3.2 1B Instruct SpinQuant (3B optional) | `export_llm` + `llama_xnnpack_spinquant.yaml`, max_context_length 1024, ExecuTorch 1.5.1; run with TextLLMRunner (pip cp313 aarch64 wheel) or cross-compiled `llama_main` | https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md, https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html, https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct |

**Datasets**

| Dataset | Use | Licence | URL |
|---|---|---|---|
| Own captures (~300 good + 40–60 per defect class, via Raspberry Pi's AI Camera dataset tool) | TinyGLASS + YOLOv8s | yours | https://www.raspberrypi.com/news/streamline-dataset-creation-for-the-raspberry-pi-ai-camera/ |
| MVTec AD (5,354 images; the "transistor" category maps to classes 1–4) | Reproduce the AUROC, seed classes | CC BY-NC-SA 4.0 (challenge only) | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| Carinthia / Carinthia-S (4,591 SEM wafer-defect images, 6 classes, masks) | H3 SEM defect classification and masks | CC BY 4.0 | https://zenodo.org/records/10715190 |
| DeepFocus 1.0 (9.5 GB SEM focus/astigmatism series) | H2 IQ-Net | CC BY 4.0 | https://zenodo.org/records/8416524 |
| NFFA-Europe SEM (18,577 images, 10 categories) | SEM pretraining + synthetic charging, noise and drift | CC BY | https://b2share.eudat.eu/records/qtq9v-ys023 |
| DeepPCB / PKU-Market-PCB | Optional PCB pretraining | research-only / unknown | https://github.com/tangsanli5201/DeepPCB |
| Own SEM session images (20–50 labelled + deliberately degraded) | Real-instrument test set; report separately | yours | — |

Show "Built with Llama". Any fine-tuned model name must start with "Llama". Credit the CC BY datasets.

---
## 5. Demo video storyboard (about 3:30)

1. **0:00–0:15 Hook.** A tray of parts: "Which of these deserve two hours of SEM time? BeamTriage decides in under a second, offline." Title card with "Built with Llama".
2. **0:15–0:40 Architecture.** Hardware tour, then the three-colour block diagram (IMX500 `.rpk`, Hailo `.hef` via HailoRT, Llama `.pte` via ExecuTorch on the A76). Unplug Ethernet and turn Wi-Fi off on camera.
3. **0:40–1:10 Optical pass and rework.** A good part goes green. A bent lead lights up the heatmap, the YOLO box says "bent_lead S1", amber REWORK, and Llama's fix streams in.
4. **1:10–1:40 Referral.** A glue blob or metal shaving gives a high anomaly score with no class. Amber/red, 3 beeps, REFER-TO-SEM-EDS. The QR work order prints with location and imaging plan (low kV or coat; EDS needed).
5. **1:40–2:20 At the SEM** (real session, or a captioned replay). The TIFF of the referred part arrives by USB stick. The metadata card shows kV, WD and pixel size. Q2 "RE-ACQUIRE: astigmatism" appears with the compass and Llama's fix; after re-acquiring, it goes green. Optional live HDMI focus meter peaks as the operator turns the knob.
6. **2:20–2:40 SEM defect review.** A Carinthia image, captioned "public Infineon dataset". Class plus size in µm gives red S3, Llama writes the FA note, and the case closes. The verdict is shown becoming a training label.
7. **2:40–3:00 Trend.** Three cut-lead parts trigger LOT HOLD, red flashing until supervisor ACK, followed by a Llama shift summary (Hindi too, if the quality is acceptable).
8. **3:00–3:20 Numbers.** Latency histogram, per-stage FPS, Llama TTFT and tok/s with vision idle vs busy, temperature and throttle log. Then switch to CPU-only mode and show it still works.
9. **3:20–3:30 Close.** SQLite export, PDF report, GitHub link, and an on-screen note saying which results are real and which are from datasets.

---
## 6. Build plan (week 1 starts 12 Oct 2026)

| Week | Dates | Deliverables |
|---|---|---|
| 1 | 12–18 Oct | **Register (closes 19 Oct).** Order the HAT. Email tech-queries@armbharatchallenge.com with "PS5" in the subject to confirm IMX500 vision and SEM-image input are acceptable and to ask what Phase 1 requires. Install Trixie, hailo-all and `apt policy hailort`. Get the DFC 3.33 wheel from the Hailo Developer Zone. `pip install executorch==1.5.1`, export Llama 1B, **measure tok/s, TTFT and RSS**. Run the TinyGLASS demo unchanged. Ruler-test the macro lens. Book SEM sessions. |
| 2 | 19–25 Oct | Fixture, lighting, gantry. Capture the optical dataset with the AI Camera tool. Collect sample SEM TIFFs to test the parsers. |
| 3 | 26 Oct–1 Nov | Train TinyGLASS and convert to `.rpk`. Train YOLOv8s and compile the HEF on x86. `hailortcli benchmark` on the Pi. |
| 4 | 2–8 Nov | Integrate domains 1 and 2: camera process and HailoRT process over shared memory. Rule engine, GPIO, SQLite. |
| 5 | 9–15 Nov | Llama service (cores 1–3), FTS5 SOP retrieval, validator and fallback, QR work orders. |
| 6 | 16–22 Nov | CPU-only fallback switch. Evaluation (AUROC, mAP, confusion matrix, latency, tok/s, thermals). |
| 7 | 23–30 Nov | **Phase 1 submission (30 Nov):** optical core, report draft, short video. |
| 8 | 1–7 Dec | SEM module: read-only USB/SMB ingest, tifffile/sidecar parsers, classical IQ metrics. Train IQ-Net (DeepFocus + synthetic) and the Carinthia classifier, compile HEFs. |
| 9 | 8–14 Dec | SEM session 1 (real IQ test set, screen recording). Closed-loop FA verdict. Optional IMX500 tensor-injection experiment. |
| 10 | 15–21 Dec | Live HDMI mode (stretch), 3B report mode on 8 GB, enclosure polish, 3-hour soak test. |
| Buffer | Jan–10 Feb 2027 | SEM session 2 for filming, final benchmarks, technical report, video, GitHub. **Phase 2 is due 10 Feb 2027.** |

---
## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Llama on the Pi is slow, or TextLLMRunner from the wheel doesn't work (inferred, untested) | Week-1 benchmark gate. 1B by default, short fields, cached advice, streaming. Cross-compiled `llama_main` on an x86 Linux VM (not the Mac) as backup. The LLM is never on the alert path. |
| HEF fails to compile, or HEF accuracy drops | Use only CNN recipes (YOLOv8, ResNet-18, MobileNet). Calibrate with ≥1,024 in-domain uint8 images. Compare the quantised HAR against the HEF early. Use a16_w16 on output layers. Match DFC 3.33 to HailoRT 4.23. |
| Anomaly model on the Hailo is unproven | Anomaly runs on the IMX500 (TinyGLASS). If conversion fails, use YOLO11n `imx` + PatchCore on the CPU at event rate. |
| IMX500 + Hailo in one process is unconfirmed | Separate processes with shared memory, tested in week 1. |
| Optics too coarse | Use 4056x3040 frames with ROI crops. Choose parts sized for the optics (TO-92, SOIC). Refer anything below about 0.3 mm to the SEM instead of claiming it. Stretch: afocal stereo-microscope adapter. |
| No SEM access | Book in week 1. Fallback: a laptop acting as the "SEM PC" replays dataset TIFFs and HDMI, clearly captioned. The optical core stands alone. |
| Carinthia doesn't match the user's SEM images | Present it as a recipe. The IQ gate is instrument-generic. Fine-tune on 20–50 of your own images and keep the "unknown" class. |
| Thermal throttling (80–85 °C) | Active Cooler under the HAT, vented case, 40 mm fan, `get_throttled` logging. |
| numpy/torch conflicts | Three venvs and three processes. |
| Judges question the EM framing | PS5 email in week 1. Lead with live optical inspection; the SEM module is the differentiator. |
| Toolchain drift, or DFC access changes after the Microchip deal | Download the wheels now, `apt-mark hold`, record every model hash. |

**CPU-only fallback (one config flag):**
- The IMX500 still runs TinyGLASS.
- H1 moves to NCNN YOLO26n (about 67 ms) on triggered frames only.
- IQ and SEM classifiers move to ExecuTorch XNNPACK INT8 (ResNet-18 about 8.75 ms per crop) for stills; the live HDMI view uses classical metrics only.
- Llama stays on cores 1–3.
- Report tok/s with vision on vs off for both modes.

---
## 8. Runner-up: BeamSight (standalone SEM review copilot)

- **What it is:** a desk console beside the SEM. An HDMI splitter feeds a USB dongle, and TIFFs come in through a read-only USB port.
  - Hailo runs an IQ classifier that gates defect classification, so bad images aren't classified.
  - YOLOv8s-seg finds Carinthia wafer defects, sized in µm from the metadata.
  - The IMX500 checks the stub load tray.
  - Llama 1B writes the corrective action.
- **Standout result:** an ablation counting false defects on defocused frames with the IQ gate off vs on.
- **Score:** 65/80, tied with SEM Sidecar.
- **Pick it instead if** you have guaranteed, repeated SEM access and want the most EM-native story.
- **Why it isn't the winner:** it depends on someone else's instrument, its "manufactured component" fit is weaker, and the AI Camera you own plays only a minor role.

---
## 9. Sources (accessed 10 Oct 2026 unless dated)

- AI HAT+ 2 product page ($200): https://www.raspberrypi.com/products/ai-hat-plus-2/ ; brief (Jul 2026): https://pip.raspberrypi.com/documents/RP-009655-MM-raspberry-pi-ai-hat-plus-2-product-brief.pdf
- AI HAT+ ("from $70"): https://www.raspberrypi.com/products/ai-hat/ ; launch prices (24 Oct 2024): https://www.phoronix.com/news/Raspberry-Pi-AI-HAT-Plus
- Price rise, 1 Apr 2026: https://www.raspberrypi.com/news/a-new-3gb-raspberry-pi-4-for-83-75-and-more-memory-driven-price-increases/
- Pi AI software docs: https://www.raspberrypi.com/documentation/computers/ai.html ; apt index: http://archive.raspberrypi.com/debian/dists/trixie/main/binary-arm64/Packages.gz
- Sixfab Pi 5 benchmark (13 Jul 2026): https://sixfab.com/blog/raspberry-pi-5-ai-hat-benchmark-deepx-vs-hailo/
- Core Electronics comparison (26 Jan 2026): https://core-electronics.com.au/guides/raspberry-pi-ai-hat-1-vs-ai-hat-2-which-should-you-buy/
- CNX AI HAT+ 2 review (20 Jan 2026): https://www.cnx-software.com/2026/01/20/raspberry-pi-ai-hat-2-review-a-40-tops-ai-accelerator-tested-with-computer-vision-llm-and-vlm-workloads/
- Jeff Geerling (15 Jan 2026): https://jeffgeerling.com/blog/2026/raspberry-pi-ai-hat-2
- Hailo community, Pi 5 FPS (Sep–Nov 2025; Apr 2026): https://community.hailo.ai/t/the-performance-on-the-raspberry-pi-5-with-the-hailo-8-chip-seems-not-good-as-he-official-results/17473 , https://community.hailo.ai/t/official-fps-benchmark-on-hailo-8-using-raspberry-pi-5/18873
- Hailo Model Zoo: https://github.com/hailo-ai/hailo_model_zoo ; HailoRT releases: https://github.com/hailo-ai/hailort/releases
- Ultralytics Hailo (Oct 2026): https://docs.ultralytics.com/integrations/hailo ; IMX500: https://docs.ultralytics.com/integrations/sony-imx500/
- AI Camera brief (Sep 2024): https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf ; docs: https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- picamera2 v0.3.32 tensor injection (27 Nov 2025): https://github.com/raspberrypi/picamera2/releases/tag/v0.3.32
- AI Camera + AI HAT coexistence (Nov 2024): https://forums.raspberrypi.com/viewtopic.php?t=378673
- TinyGLASS (AICAS 2026): https://arxiv.org/abs/2603.16451 , https://github.com/ETH-PBL/TinyGLASS
- ExecuTorch backends: https://docs.pytorch.org/executorch/stable/backends-overview.html ; PyPI 1.5.1 (22 Sep 2026): https://pypi.org/project/executorch/ ; Llama README: https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md ; Pi tutorial: https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html
- Llama 3.2 1B model card and licence: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- India prices: Silverline 26T https://www.silverlineelectronics.in/products/official-raspberry-pi-ai-hat-26-tops-artificial-intelligence-authentically-impressive-silverline-india ; Silverline HAT+ 2 https://www.silverlineelectronics.in/products/raspberry-pi-official-ai-hat-official-raspberry-pi ; e2u 13T https://e2u.in/products/official-raspberry-pi-ai-hat-13-tops-e2u-india
- Pi 5 prices (2 Oct 2026): https://atomiccreations.co/blogs/news/raspberry-pi-price-increase
- TC358743 fix PR #7406 (merged 15 Jun 2026): https://github.com/raspberrypi/linux/pull/7406
- Datasets: Carinthia https://zenodo.org/records/10715190 ; DeepFocus https://zenodo.org/records/8416524 ; NFFA https://b2share.eudat.eu/records/qtq9v-ys023 ; MVTec AD https://www.mvtec.com/company/research/datasets/mvtec-ad ; KolektorSDD2 https://www.vicos.si/resources/kolektorsdd2/
- SEM metadata (tifffile): https://github.com/cgohlke/tifffile
- Applied SEMVision H20 (19 Feb 2025): https://www.nasdaq.com/press-release/applied-materials-accelerates-chip-defect-review-next-gen-ebeam-system-2025-02-19
- Challenge page (19 Sep 2026): https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html

[^ce]: Core Electronics, Pi 5 measurements, Jan 2026.
[^sx]: Sixfab async NPU-only, Pi 5 8 GB, Jul 2026 (vendor benchmark).
[^hc]: Hailo community, hailortcli at batch 1 on Pi 5; a C++ app measured about 105 FPS.