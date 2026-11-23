# Ayas repo audit: reuse for 1x IMX500 + N/E/S/W banks + Hailo-10H

**Tests:** `.venv/bin/pytest -q` gave **50 passed in 29.14 s** (31.0 s wall, measured with `/usr/bin/time`). I ran it with `-p no:cacheprovider` so nothing in the repo was written. No files were changed. Local env: Python 3.11.14, numpy 2.4.6, opencv-python-headless 5.0.0.93, scikit-learn 1.9.1, macOS arm64.

**Hardware check:** the "40 TOPS HAT" is the AI HAT+ 2 (Hailo-10H). It needs the `hailo-h10-all` package (not `hailo-all`) and Raspberry Pi OS Trixie 64-bit ([RPi AI docs](https://www.raspberrypi.com/documentation/computers/ai.html); 40 TOPS/8 GB per [CNX review](https://www.cnx-software.com/2026/01/20/raspberry-pi-ai-hat-2-review-a-40-tops-ai-accelerator-tested-with-computer-vision-llm-and-vlm-workloads/)).

## (a) Works unchanged for one camera + N lights

- **`rules.py`** (whole file) and `configs/spec.yaml`: they contain no geometry.
- **`store.py`** (whole file).
- **`detect.py:34-71`**: it loops over whatever keys the golden has.
- **`golden.py:64-105`**: the Welford mean/variance code is generic over keys.
- **`align.py`**: `warp_to_grid` (:31), `estimate_motion` (:56), `match_gain` (:78) and `prepare` (:87) work once `image_maps()` is fixed.
- **`capture.py`**: `capture_set` (:20-41; a 1-worker pool is fine), `lit`, `channel`, `bright_view`.
- **`calibration.py`**: `find_plate_corners` (:26) and `homography_from_corners` (:80). Manual `corners_px` (:123) is the fallback.
- **`hardware/lights.py:22-43`**: `GpioLights`, one pin per bank.
- **`hardware/cameras.py:20-62`**: `PiCamera` works for the IMX500. Focus controls are skipped because there is no `lens_position` (:47).
- **`penetrant.py`**: it uses only the top camera (:225, :232-233), but see the blue-light problem in (b).
- **`llm.py`**: `Advisor`, `MockBackend` and the template answers (:140-261).
- **`classify.py`**: `ForestClassifier` (:107-145) is generic over `FEATURE_NAMES`.
- **`shape.py:48-75`**: `shape_setup` is generic. With 4 banks at one elevation e, singular-value ratio s3/s1 = √2·tan e (calc): 0.38 at 15°, 0.82 at 30°. Both are above the 0.15 cut at :54, so it gives "full" mode. Set `shape: true` on all 4 banks; adding the top ring as a 5th light helps conditioning.

## (b) What must change

**3-camera assumptions**
- `config.py:18` hard-codes `VIEWS`.
- `config.py:176` has a default elevation dict that throws KeyError for any new view name.
- `config.py:230-233` requires exactly one `pair` light per view.
- `config.py:244-246` requires all 3 cameras.
- `config.py:205-208`: `image_maps()` adds `oblique45`/`grazing` maps.
- Other places that loop over `VIEWS`: `rig.py:19`, `calibration.py:121`, `capture.py:78` (`load_frames` silently drops unknown views), `cli.py:54,58`, `report.py:76`.
- Fix: read views from YAML (`views: [top]`). Make `pair` optional and drive roles from a `role:` field per light (`brightfield` / `darkfield_N/E/S/W` / `uv`). Effort: **9 h**.

**Homographies**
- A single camera means one homography. `calibrate()` works once it loops over configured views.
- Under grazing-only light, shiny aluminium looks dark (the mirror reflection goes away from the camera). `bright_view` (capture.py:62) then may not show a bright plate, so Otsu (calibration.py:30) fails. Keep a top/ring light for calibration, or use `corners_px`.
- Lens distortion is ignored (homography only). Effort: **2 h**.

**Features, classifier, report, prompt**
- `features.py:67-88` (`MapRoles`) and `:153-169` assume a 45° camera and a grazing camera.
- Remap the "grazing" role to "top camera under the dark-field banks", e.g. the max over N/E/S/W.
- Add a directional-anisotropy feature (max/min response across banks). A scratch lights up only under the bank across its direction, which is a strong cue and also gives its orientation.
- Thresholds that need retuning on real data: `classify.py:36-48`.
- `report.py:38-44` (contrast keys) and `schemas/inspection.schema.json:24,36` (requires top/oblique45/grazing) need a schema bump to v2.1.
- `llm.py:26` (SYSTEM_PROMPT) and `llm.py:197` describe three cameras.
- Effort: **13 h** (7 h code, 6 h tuning).

**Blue LEDs (they bought blue, not white)**
- `LUMA` (config.py:24) weights blue at only 0.114.
- `chroma_shift` for stains (features.py:165) means nothing under one colour of light.
- `penetrant.redness()` (penetrant.py:238-244) white-balances on the developer. Under blue light the red channel is roughly zero, so this amplifies noise.
- Fixes:
  - Add a per-light `channel: b` so blue banks use the B plane. This affects `align.py:46` and `shape.luminance` (shape.py:78-83).
  - For penetrant, either use a white top light, or redefine the dye signal as "dark in B" (the dye absorbs blue).
- The 390-395 nm UV LEDs are not used anywhere. The kit is visible Type II dye, which is not fluorescent (UNVERIFIED that UV adds anything).
- Effort: **3 h**.

**Photometric stereo on aluminium (`shape.py`)**
- The code assumes a Lambertian surface. Each image is normalised by plate median × l_z (:93), then `pinv` solves I = ρ·n·l (:94-95).
- A camera looking down with grazing banks on brushed aluminium sees dark-field scatter, which is not n·l and is very anisotropic. Treat `ps_nx`/`ps_ny` as a relief cue only.
- So the mm values from `integrate_slopes` (:109-119), `depth_mm` (report.py:61) and `relief_depth_mm=0.08` (classify.py:42) cannot be trusted. Report a unitless relief index instead.
- Light directions assume distant lights: one vector per light (config.py:101-104). For a bank 150 mm from the plate centre, azimuth swings ±18.4° across the plate and irradiance varies 4× (calc). Subtracting the golden mean (shape.py:127-128) absorbs this only if a jig keeps placement fixed.
- Exposure is fixed at init (cameras.py:40-49). Grazing banks need longer exposure, so add a per-state `set_controls`.
- Effort: **6 h**, including centre-punch dent tests.

**`sim.py`**
- `SimRig` loops over views (:298, :316).
- `self.top_light = cfg.pair_light("top")` (:309) will break once `pair` is optional.
- Shading is Lambertian (:231-236), so green sim tests prove nothing about real aluminium.
- Fix: a one-view `rig.sim.yaml` (top `scatter_gain` 0.3 → about 2.5) plus updated `conftest.py:44-59` fixtures. Effort: **5 h**.

**Golden reference with only 5 plates**
- The code allows n ≥ 2 (golden.py:97) but per-pixel std from 5 parts is noise. The 95% CI of σ̂/σ is [0.60, 2.87] (calc, χ² with 4 dof), so z > 6 (detect.py:43) will fire falsely.
- Options:
  1. Capture all 5 plates × 2 faces × 0°/180° (180° keeps the brushing direction) = 20 captures (calc). **Do this before any centre-punching.**
  2. Pool the variance spatially: box-filter `m2/(n-1)` over 31×31 px, about 3.8k dof (calc).
  3. Self-reference fallback with no golden: map minus a 51-px median blur, divided by the plate MAD.
  4. Leave-one-out script to measure the false-positive rate.
- Effort: **6 h**.

## (c) Missing for the brief

- **Hailo module: none.**
  - Write `npu.py` using `hailo_platform` `VDevice` → `create_infer_model` → `configure` → `run_async`, as in [hailo-apps hailo_inference.py:7-44,119-139](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/core/common/hailo_inference.py).
  - Tile the 1100×1100 grid (100+2×5 mm at 0.1 mm/px) as 2×2 tiles of 640 px with 180 px overlap (calc), then NMS across tiles.
  - Day-1 smoke test: the precompiled COCO `yolov8n.hef` for hailo10h at v5.5.0 ([Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst)).
  - A custom defect model needs DFC 5.x on **x86_64 Linux** ([Ultralytics Hailo](https://docs.ultralytics.com/integrations/hailo)). The Mac cannot do it, so use a cloud VM.
  - Do not use hailo-ollama for Llama: the brief requires ExecuTorch on the CPU.
  - Effort: **34 h** (wrapper 10, dataset/labels 12, train 4, compile 8).
- **ExecuTorch integration in `llm.py`** (checked against v1.5.1, released 2026-09-23):
  - `--seq_len` (:95) is marked DEPRECATED. There is no `--max_new_tokens`, so a 2048 seq_len allows about 548 output tokens after a ~1500-token prompt (calc, from [irunner.h](https://github.com/pytorch/executorch/blob/main/extension/llm/runner/irunner.h) `resolve_max_new_tokens`). Pass `--max_new_tokens=160`.
  - **BOS bug:** `llm.py:51` assumes the runner adds BOS. In v1.5.1 `num_bos` defaults to 0 ([main.cpp:63-66](https://github.com/pytorch/executorch/blob/v1.5.1/examples/models/llama/main.cpp)), so prompts go in without `<|begin_of_text|>`. Set `bos_in_prompt: true` or pass `--num_bos=1`.
  - The header/eot template (:50-56) is correct. Special tokens written in the prompt text are parsed ([tokenizers bpe_tokenizer_base.cpp:271](https://github.com/meta-pytorch/tokenizers/blob/main/src/bpe_tokenizer_base.cpp)).
  - `echo` defaults to true, so the prompt-stripping in `llm.py:114` works.
  - There is no token budget. A prompt that is too long fails with "Prompt exceeds KV cache capacity" ([text_llm_runner.cpp](https://github.com/pytorch/executorch/blob/main/extension/llm/runner/text_llm_runner.cpp)), and the error is silently replaced by the template fallback (:165-167). Log which backend actually answered.
  - No streaming: `subprocess.run` (:105) blocks. Use `Popen`; the runner flushes after each token.
  - Use `--prompt_file` instead of a very long argv.
  - The model reloads on every answer (UNVERIFIED cost).
  - `setup.sh pi5` supports a macOS host ([setup.sh](https://github.com/pytorch/executorch/blob/v1.5.1/examples/raspberry_pi/setup.sh)).
  - `rig.pi.yaml:93` uses `/home/pi`, which is a likely wrong user, and `runner` is not root-resolved (llm.py:133).
  - Effort: **13 h**.
- **Touchscreen GUI: none.** `rig.py:32,39` blocks on `input()`. Build a PyQt5 app with the picamera2 Qt preview (UNVERIFIED API), a capture QThread, a verdict banner, and preset-question buttons. Effort: **16 h**.
- **Operator alert:** nothing beyond the overlay colours (pipeline.py:24-30). Add a gpiozero `Buzzer` + `TrafficLights` on free pins. LED pins in use are 17/27/22/23/24 (rig.pi.yaml:24-28). Avoid GPIO0/1 (HAT EEPROM) and SPI pins if the touchscreen is SPI (UNVERIFIED, model unknown). Effort: **2.5 h**.
- **Log export:** only `history` printing exists (cli.py:335-342). Add CSV via `csv` and PDF via reportlab. Effort: **3.5 h**.
- **Benchmark:** stage timings exist (pipeline.py:71,86; cli.py:153). But the `PyTorchObserver` stats are thrown away (llm.py:122); parse them for tok/s. Read temperature from `/sys/class/thermal/thermal_zone0/temp` and throttling from `vcgencmd get_throttled`; get Hailo FPS from `hailortcli run`. Effort: **3.5 h**.
- **Dataset capture/labelling:** `cmd_dataset` (cli.py:232-261) only writes feature CSVs. Build a raw multi-light capture tool with plate/face/rotation metadata. Export `detect.py` regions as YOLO prelabels, then correct them in CVAT/Label Studio. Effort: **5.5 h**.

## (d) Pi 5 breakage risks

- **GPIO:** OK. The code uses gpiozero (lights.py:26) and there is no RPi.GPIO anywhere. The docs say only lgpio works on the Pi 5 ([gpiozero](https://gpiozero.readthedocs.io/en/latest/api_pins.html)).
- **picamera2:** `AeEnable`/`AwbEnable`/`ExposureTime`/`AnalogueGain`/`ColourGains` (cameras.py:40-46) are valid controls. Full-res 4056×3040 runs at **10 fps** ([camera specs](https://www.raspberrypi.com/documentation/accessories/camera.html)). With 2 flush frames that is about 300 ms per state, roughly 1.8 s for 6 states (calc). Each RGB888 buffer is 37 MB (calc); a CMA allocation failure is UNVERIFIED. Converting to float32 costs 148 MB per frame (capture.py:46-50; calc), so crop to the plate before conversion. Saving PNGs (capture.py:68-71) will fill the SD card. Install `imx500-all` ([AI Camera docs](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)). Flush by `SensorTimestamp`, not by a fixed frame count.
- **Threading:** `sqlite3.connect` (store.py:33) refuses calls from a thread other than the one that created it, which will break a GUI worker. `llama_main` with 4 threads plus OpenCV plus Qt will compete for the CPU; use `cpu_threads: 3` and add a fan.
- **Versions:** Trixie ships numpy 2.2.4 ([Debian](https://packages.debian.org/trixie/python3-numpy)) and OpenCV 4.10.0 ([Debian](https://packages.debian.org/trixie/python3-opencv)), on Python 3.13. The repo was only tested on OpenCV 5.0. Use a venv with `--system-site-packages` and `pip install -e . --no-deps`. Retrain the classifier joblib on the Pi, because sklearn versions differ.
- **Paths:** `rig.pi.yaml:80` (USB camera placeholder) will crash; delete it. `cli.py:22` defaults to the sim config.

## (e) Order and effort (about 124 h total, calc)

1. Pi bring-up: Trixie, `hailo-h10-all`, `imx500-all`, pytest on the Pi — 3.5 h.
2. Llama/ExecuTorch fixes and real `.pte` — 13 h. This is mandatory, so de-risk it first.
3. Capture tool, then capture the golden set **before damaging the plates** — 5.5 h.
4. Config/single-view refactor, sim and tests — 16 h.
5. Few-plate golden method, then photometric-stereo and blue-light changes — 15 h.
6. Features/classifier retuning — 13 h.
7. GUI, alerts and export — 22 h.
8. Hailo module, in parallel by a second member — 34 h.
9. Benchmark script and a write-up of the two separate pipelines — 3.5 h.

Release facts: ExecuTorch v1.5.1 and HailoRT v5.5.0 (2026-10-28) come from `gh api` on [executorch releases](https://github.com/pytorch/executorch/releases) and [hailort releases](https://github.com/hailo-ai/hailort/releases).