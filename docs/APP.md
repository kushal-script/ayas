# TriNetra on-device app (`trinetra-app`)

The station program that closes the loop on the Raspberry Pi 5 (and on the Mac for development):

```
camera burst (or twin .npz) -> lightproc.process_burst (ambient subtraction, glare removal, quality gate)
  -> RETAKE? -> record with verdict RETAKE + the gate message (nothing is detected)
  -> rectify to the plate grid (0.08 mm/px) -> 640 px tiles -> detector (Hailo / ONNX / fake) -> merge
  -> mm measures -> rules.judge_steel (configs/spec_steel.yaml) -> record trinetra.inspection/3.0
  -> ALERT (screen banner, console line, optional buzzer) -> LOG (SQLite + JSONL)
  -> Llama 3.2 via ExecuTorch explains the record (after the alert: the verdict never waits for it)
```

Design contract: `docs/twin/DESIGN_STEEL.md` section H. Code: `src/trinetra/app/`.

| File | What it does |
|---|---|
| `sources.py` | `TwinSource` (replays `out/bursts/*.npz`), `CameraSource` (IMX500 raw burst + LED pods on GPIO), `FileSource` (single photos) |
| `pipeline.py` | `InspectionPipeline`, geometry, mm measures, `FakeDetector`, the log (`AppLog`), the alert |
| `advisor.py` | `AdvisorService`: mock template, the `llama_main` runner (streamed, timed), the in-process `TextLLMRunner` |
| `ui.py` | the station page (stdlib `http.server`, no external JS/CSS) for the 5" 800 x 480 display |
| `bench.py` | per-stage latency, FPS, memory, CPU temperature, Llama TTFT and tokens/s |
| `cli.py` | `trinetra-app run ...` and `trinetra-app check` |
| `configs/app.yaml` | the app's own settings (pods, camera, advisor, UI, log); shared settings stay in `capture.yaml`, `twin.yaml`, `spec_steel.yaml` |

`trinetra-app` is registered in `pyproject.toml`. After pulling, run `pip install -e .` once so the command exists; until then use `python -m trinetra.app.cli` with the same arguments.

## 1. Run it on the Mac

```bash
source .venv/bin/activate

# Twin bursts for the demo: every test plate x every lighting condition (+ hood rows), with .npz bursts.
trinetra-twin matrix --n 2 --save-bursts --out data/twin_demo          # -> data/twin_demo/bursts/*.npz (30-50 MB each)

# Console: each burst once, verdict line + advice
trinetra-app run --source twin:data/twin_demo/bursts --detector fake --advisor mock

# Station page: http://127.0.0.1:8080 (Inspect button), or an automatic demo loop every 5 s
trinetra-app run --source twin:data/twin_demo/bursts --detector onnx:models/best.onnx --advisor mock --ui --auto 5

# Benchmark (writes reports/bench_<host>_<date>.json and .md)
trinetra-app run --source twin:data/twin_demo/bursts --detector onnx:models/best.onnx --advisor mock --bench 20
```

- `--detector fake` replays the twin's ground-truth boxes through the real tile, merge, measure and rules code. The record says `backend.vision = "fake (ground-truth replay)"`, so it is never mistaken for a detector result. `fake:empty` returns no boxes.
- `--detector onnx:<best.onnx>` is the trained YOLOv8n on onnxruntime (CPU). `pt:<best.pt>` uses Ultralytics (training venv only).
- `--source files:<dir>` inspects single photos (the old one-frame arm, `capture.mode = naive`, no gate). A `corners.json` there can give `{"<file>": {"corners_px": [[x, y] x 4], "material": ..., "thickness_mm": ..., "plate_size_mm": [W, H]}}`; otherwise the plate is found automatically.
- Part settings come from the burst (twin) or `configs/app.yaml part:` (camera, photos). Override with `--material ms_hot_rolled --thickness 4 --device-mode hood`.

## 2. Run it on the Pi 5

### Install (Raspberry Pi OS Trixie 64-bit)

```bash
sudo apt install -y python3-picamera2 python3-gpiozero python3-libcamera
sudo apt install -y hailo-h10-all            # only for --detector hailo:<hef> (HailoRT 5.x, AI HAT+ 2)
git clone <repo> ~/trinetra && cd ~/trinetra
python3 -m venv --system-site-packages .venv  # picamera2, libcamera, gpiozero and hailo_platform come from apt
source .venv/bin/activate
pip install -e .                              # keeps apt's numpy and PyYAML; adds opencv-python-headless, scikit-learn
trinetra-app check                            # lists camera, GPIO, Hailo, onnxruntime, ExecuTorch, network, model files
```

### Llama via ExecuTorch (mandatory for the advice)

Two ways; both get the same prompt, guardrail check and template fallback from `trinetra/llm.py`.

1. **The `llama_main` runner the team already has** (`--advisor executorch`). Set in `configs/app.yaml` `llm:` the `runner` (full path, or a wrapper script that sets `LD_LIBRARY_PATH`), `model_path` (`.pte`) and `tokenizer_path`, or pass `--runner`, `--pte`, `--tokenizer`. The app streams the runner's stdout to the screen and reads its `PyTorchObserver {...}` stats line for TTFT and tokens/s. The runner loads the `.pte` on every answer, so the wall time includes the model load (reported separately as `load_ms`).
2. **In-process `TextLLMRunner`** (`--advisor executorch_py`): `pip install executorch==1.5.1 torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple` in the same venv (docs/research/2026-11-05_executorch.md; match the version the `.pte` was exported with). The model stays loaded and one warm-up answer runs at start-up. Paths come from `llm.py.model_path` / `llm.py.tokenizer_path` or `--pte` / `--tokenizer`.

If the configured backend cannot start, the app prints a WARNING, the page shows "(... unavailable ...)" next to the advisor name, and the template answers. For `executorch` this is checked at start-up: the runner executable, the `.pte` and the tokenizer must exist. The record's `backend.advisor` always says who answered (`mock-template` on any fallback). The page carries "Built with Llama" as the Llama 3.2 licence asks.

Prompt length: `llm.Advisor` drops the oldest follow-up turns so that prompt + answer fit `llm.seq_len` (2048 in `app.yaml`). Both backends report it: `executorch` from `seq_len`, `executorch_py` from `py.seq_len`, or from `llm.seq_len` when that is -1. Set it to the context the `.pte` was exported with.

### Copy twin bursts to the Pi (before going offline)

```bash
rsync -av --progress data/twin_demo/bursts/ pi@<pi-host>:~/trinetra/data/twin_demo/bursts/
# one file per (plate, condition, device mode); a handful is enough for the demo (30-50 MB each)
```

### Run

```bash
# twin bursts as the camera, Hailo detector, Llama in-process, station page on the display
trinetra-app run --source twin:data/twin_demo/bursts --detector hailo:models/yolov8n_steel.hef \
                 --advisor executorch_py --ui --auto 8
chromium --kiosk --app=http://127.0.0.1:8080 &      # the 5" display (Wayland desktop)

# the real camera (AI Camera raw burst + both LED pods)
trinetra-app run --source camera --detector hailo:models/yolov8n_steel.hef --advisor executorch --ui
```

The UI binds to 127.0.0.1 by default (the Pi's own browser). Use `--host 0.0.0.0` only to watch from a laptop on the same network, never in the offline demo.

## 3. The station page (800 x 480)

- **Banner**: the verdict, coloured by verdict: PASS green, CLEAN + REINSPECT blue, REWORK orange, HOLD FOR QA purple, REJECT red, **RETAKE amber**. Next to it is the verdict instruction. REJECT, HOLD and RETAKE pulse until **Acknowledge** is pressed (the acknowledgement is logged).
- **Image**: the rectified plate grid with the boxes coloured by disposition and their ids, or, on RETAKE, the refused frame with the plate outline.
- **Capture-quality line**: gate, r (ambient / LED), estimated ambient lux, motion px, saturated %, device mode; the gate message under it when it is not PASS.
- **Defect table**: id, type (`?` = uncertain detection, operator to confirm), severity, disposition, detector confidence (not severity), location (plate mm), size, distance to the edge (scroll sideways on the 5" display).
- **Advice**: Llama's text as it streams in, then who answered, TTFT, tokens/s and the call time. While it streams, the text is grey italic and marked "draft: checked when complete". The guardrail check (`llm.check_reply`) runs on the whole answer, and a reply that fails it is replaced by the template.
- The panel only shows advice about the part on screen. With `--auto` and a slow model, an explanation still running when the next part is inspected is no longer shown. A queued explanation of a part that is no longer the latest is skipped (logged as `explain_skipped` in `advice.jsonl`), so the advice never falls further and further behind.
- **Buttons**: Inspect (one capture from the source), Why?, How to fix?, Shift summary (the last 50 records), Acknowledge.

API (for scripts): `GET /api/latest`, `GET /api/record` (the full 3.0 record), `GET /api/image.jpg`, `GET /api/history`, `POST /api/inspect`, `POST /api/ask {"kind": "why"|"fix"|"shift"}` or `{"question": "..."}`, `POST /api/ack`.

## 4. Record, log and measures

- **Record**: `trinetra.inspection/3.0` (`schemas/inspection.v3.schema.json`), built by `report.build_steel_record`. The app adds `source` (twin scene, condition, lux, and `truth`, the rules verdict on the twin's visible ground truth), `inspection_id`, and after the advisor answers `advice`, `backend.advisor`, `timing_ms.advisor` and `advisor_stats`.
- **timing_ms**: capture (camera grab + demosaic, or the twin file read), process (lightproc), rectify, tile, detect (model inference only), merge, measure, rules, report, image, **total** (camera-to-result, without the advisor), advisor.
- **Log** (`log_dir`, default `data/app/`): `trinetra.db` (the `store.py` tables `inspections` and `chats`; the row is updated with the advice), `inspections.jsonl` (one record per line, written at alert time), `advice.jsonl` (advisor answers, questions, acknowledgements, with stats), `images/<id>_<part>.jpg`.
- **Locations** are plate mm in the grid's own frame: origin at the imaged top-left plate corner, x right, y down, as in the twin's records. `near_edge_mm` is the gap from the detector box (the defect's extent) to the nearest plate edge or known hole, so the edge-or-hole rule errs on the safe side.
- **Holes.** The device does not detect holes, so give them when the part has any:
  - for a jig, in `app.yaml part.holes_mm: [[x, y, d], ...]` (grid-aligned plate mm);
  - for single photos, in `corners.json` (`holes_mm`);
  - in a twin burst, as `holes_plate_mm` ([u, v, d] in plate mm, mapped through `H_frame_plate`). `post.save_burst` does not write it yet, so twin plates with holes are judged on the plate edges only. Their `source.truth` verdict counts the holes and may be one level higher.

  Without a hole list, only the plate edges count. A defect near a hole is then judged one level too low.
- **Sizes** (the camera never measures depth). Detector boxes are padded to at least 8 px (0.64 mm) in training, so sizes the rules compare with sub-mm limits (widths, pit diameters) are measured on the defect's own pixels inside the box. These are the pixels that differ from the ring around the box, at half of the peak difference for geometric classes. Lengths of scratches and cracks come from the box: on a rendered twin plate the pixel mask missed the faint end of a curved 36 mm scriber mark (25 mm), while the box diagonal gave 36.2 mm.

| Class | Measured | Fallback (no clear pixels) |
|---|---|---|
| scratch, crack | length = box diagonal (the mark's extent); width = pixel area / the pixels' own long side (mean visible width); abrasion-like boxes add area | width = short box side |
| pit | one blob: diameter (equivalent circle), count 1; several: count, largest pit, hull diameter and area | long box side, count 1 |
| corrosion | area of the changed pixels, equivalent diameter, length for a streak | ellipse in the box |
| dent | long box side (a dent shades its whole footprint) | |
| stain | ellipse in the box (faint rings fill it only in part) | |

## 5. Offline demo checklist (organisers: "show the full system works without the internet")

1. Before: copy the bursts and the `.pte`/tokenizer to the Pi, run `trinetra-app check` (every file found).
2. Pull the Ethernet cable and turn Wi-Fi off (`nmcli radio wifi off`). `trinetra-app check` must print `network: OFFLINE: no default route`.
3. Start the station (`--ui --auto 8`, or press Inspect) on the Hailo detector and the ExecuTorch advisor; the console prints each alert line.
4. Show the lighting story on twin bursts: C0-C5 open give verdicts, C6 sun without the hood gives amber **RETAKE** with "Too bright: shade the part or fit the hood" (a refusal, never a silent pass), and C6 with the hood gives a verdict again.
5. Llama: the advice streams in after the banner; press Why?, How to fix?, Shift summary. Point at the TTFT and tokens/s on the page.
6. Show the log: `sqlite3 data/app/trinetra.db "select id, part_id, verdict, defect_count from inspections order by id desc limit 5"` and `tail -n 1 data/app/inspections.jsonl | python3 -m json.tool` (the one JSON hand-off).
7. Run the benchmark (section 6) and show the markdown table.
8. The code never imports a network client: `tests/test_app.py::test_app_imports_no_network_client`. The only socket is the local UI server on 127.0.0.1.

## 6. Benchmarks

```bash
trinetra-app run --source twin:data/twin_demo/bursts --detector hailo:models/yolov8n_steel.hef \
                 --advisor executorch_py --bench 20 --warmup 1
trinetra-app run --source twin:data/twin_demo/bursts --detector hailo:models/yolov8n_steel.hef --no-advice --bench 50   # vision only
```

- Each inspection runs vision first, then the advisor, one after the other, so Llama never overlaps the vision timings.
- The report has: per-stage p50/p95/mean/max ms; FPS, both vision (1000 / mean total) and loop (including the source and the log); camera-to-result and camera-to-advice latency; peak RSS of the app and of child processes (the `llama_main` runner); CPU temperature (start, max, end, from `/sys/class/thermal`); and Llama TTFT on the model clock, first text on the wall clock (including model load), decode and prefill tokens/s, generated tokens, call time and who answered. The warm-up inspection is kept separately as `cold_start`.
- For the report, state the cooling (Active Cooler), the PSU (27 W), the OS, and the ExecuTorch, torch, HailoRT and model versions. Run idle and again with the camera source. Use `vcgencmd get_throttled` before and after.
- `ru_maxrss` is the peak of the whole process, including the loaded `.pte` for `executorch_py`. Measure memory early (organisers: 8 GB is safer).

## 7. Pi bring-up checks (assumptions in `CameraSource` to verify)

| # | Assumption | How to check |
|---|---|---|
| 1 | `raw={"format": "SRGGB10"}` gives an **unpacked** 16-bit stream (not `_CSI2P`, not the Pi 5's compressed `*_PISP_COMP1`), and its Bayer order is read back. The app refuses packed and compressed formats, because they would demosaic to garbage. | print `cam.camera_configuration()["raw"]`; if the Pi 5 does not offer `SRGGB10` unpacked, set `camera.raw_format: SRGGB16` (MSB-aligned, read the same way); a red card must come out red |
| 2 | Pi 5 unpacked raw is MSB-aligned (`left_shifted: auto` checks the first frame) | `raw.max()` of a dark frame is about 4096 (64 << 6) |
| 3 | black level 64 DN (10-bit) as `capture.yaml`; or `camera.black_level: metadata` uses `SensorBlackLevels` (16-bit scale) | dark frame (lens cap) mean after `raw_to_rgb` is about 0 |
| 4 | `capture_request(flush=True)` returns a frame exposed entirely **after** the pod switch (rolling shutter included) | L frame: the right half must not carry the B1 brightness; if it does, set `camera.settle_ms` to one frame time |
| 5 | AE/AWB off, `ExposureTime` 10000 us and `AnalogueGain` 1.5 are applied exactly | `meta.camera.warnings` stays empty (it compares every frame's metadata) |
| 6 | `FrameDurationLimits` is clamped by libcamera to the mode (about 100 ms at 4056 x 3040) | burst `grab_ms` about 5-10 frame times |
| 7 | pods: BCM 17 = the pod on the IMAGE LEFT, 27 = right, active-high MOSFETs (`configs/app.yaml pods`) | L-only frame is brighter on the left; otherwise swap the pins (and check `twin.yaml rig.model_x_is_image_left`) |
| 8 | memory: 5 full-resolution frames are about 0.75 GB as float32 | crop with `camera.roi` (e.g. `[1004, 496, 2048, 2048]`: the plate is about 1530 px wide at 200 mm) |
| 9 | plate corners from `B1 - A` (bright plate on a dark mat; the hood interior is black) | for a jig, set fixed `camera.corners_px` |
| 10 | the runner adds BOS or not (ET 1.6 `llama_main --num_bos` defaults to 0) | set `llm.bos_in_prompt: true` if answers start oddly |
| 11 | the runner's flags (`--seq_len` deprecated in 1.6; `--max_new_tokens` may exist) | `llama_main --help`; extra flags go in `llm.extra_args` |

## 8. Tests

`.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_app.py` uses:
- synthetic bursts in the `post.save_burst` npz layout, plus one written by the real `post.save_burst`;
- a FakeDetector and the mock advisor;
- a fake `llama_main` script (streaming plus the `PyTorchObserver` stats line);
- a fake `executorch.extension.llm.runner` (the documented `TextLLMRunner` API);
- fake `picamera2` / `gpiozero` modules for the camera burst order and the raw-format checks.

It also checks the UI server (including that stale advice never shows under a newer part), the skipping of stale explanations, the benchmark report and the CLI.
