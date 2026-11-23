# Hailo-10H HEF: compile on Colab, verify on the Pi

## Versions (the version trap)

| Where | What |
|---|---|
| Pi | Raspberry Pi OS Trixie 64-bit, apt `hailo-h10-all` = **HailoRT 5.1.1** (`hailo_platform` Python bindings) |
| Compile | **DFC 5.1.0 + Hailo Model Zoo v5.1.0**, Colab **T4**, Runtime version **2026.07** (Python 3.12) |
| Model | **YOLOv8n**, 640 x 640, 6 steel classes: `scratch, dent, pit, corrosion, crack, stain` (ids 0-5 = `ayas.tiles.CLASSES`). Not YOLO26: it has no v5.1.0 support |

- Never compile with DFC 5.5 or Model Zoo master: those HEFs target the 5.5 runtime.
- Precompiled HEFs only from `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/`.
- The DFC does not run on macOS, so the compile happens on Colab.

## Compile (`compile_hef_colab.ipynb`)

1. **ONNX** (Mac): `training/export_onnx.py --weights runs/train/<row>/weights/best.pt --name <name>` gives `models/<name>.onnx` (opset 11, 1 x 3 x 640 x 640, no NMS) and checks that the six head convolutions the Model Zoo cuts at are present. Upload it to Drive as `ayas_hailo/best.onnx`.
2. **Calibration** (Mac): `training/calib_set.py --images data/real/yolo/images/train --out runs/calib --no-npy`, then `cd runs/calib && zip -qr calib_640.zip calib_640`. These are 1,024 raw 0-255 RGB crops of TRAIN images from the burst pipeline (what the deployed model sees), never a test plate. Upload `calib_640.zip`.
   - **Phase 1 (steel, no real burst captures yet):** calibrate on the twin's burst-arm TRAIN tiles: `training/calib_set.py --images data/twin/arm_burst/yolo/images/train --allow-twin --out runs/calib --no-npy`. Lighting-matrix plates (seed >= 900000) are always refused. Recalibrate on real burst captures as soon as they exist.
3. **DFC wheel:** download `hailo_dataflow_compiler-5.1.0-*.whl` yourself from the Hailo Developer Zone (manual login) into `ayas_hailo/`. No credentials go in the notebook.
4. **Run the notebook** with Runtime > Change runtime type > T4 GPU, Runtime version 2026.07.
   - **Step 0 (go/no-go 1):** it compiles the stock COCO yolov8n and saves Hailo's precompiled v5.1.0 HEF as a reference.
   - **Custom model:** it copies `yolov8n.yaml` and its `.alls` next to the originals inside `hailo_model_zoo/cfg/` (their paths are relative) and appends `model_optimization_config(calibration, calibset_size=1024)`. Then it runs:
     ```
     hailomz compile --ckpt best.onnx --calib-path calib_640/ --yaml <copy>.yaml --classes 6 --hw-arch hailo10h
     ```
     There is no `--performance` on the first pass.
   - **Check:** every output's SNR must be at least 10 dB. For a low output, add `quantization_param(<layer>, precision_mode=a16_w16)`.
5. The HEF, HARs, configs, log and `compile_info.json` land in `ayas_hailo/out/`, plus `<hef stem>.json` with the class names.
   - **Keep that JSON next to the HEF on the Pi.** `ayas.npu.HailoDetector` takes its class names and count from it (else `ayas.tiles.CLASSES`); `export_onnx.py` writes the same file next to the ONNX.

## Verify on the Pi

```bash
hailortcli fw-control identify                 # Device Architecture HAILO10H, firmware 5.1.1
hailortcli parse-hef ayas_v8n.hef          # input 640x640x3 UINT8; NMS output, 6 classes (scratch..stain)
hailortcli run ayas_v8n.hef                # throughput; repeat with --batch-size 9
```

Python, in the repo venv: create it with `--system-site-packages` so the apt `hailo_platform` is visible, then `pip install -e . --no-deps`.

```python
import time

import cv2
from ayas.npu import HailoDetector, detect_grid

grid = cv2.cvtColor(cv2.imread("grid.png"), cv2.COLOR_BGR2RGB)   # rectified 0.08 mm/px grid, RGB
with HailoDetector("ayas_v8n.hef", conf=0.25, batch_size=9) as det:   # classes from ayas_v8n.json
    t = time.perf_counter()
    dets = detect_grid(grid, det)                                  # grid px; mm via config.Grid.px_to_mm
    print(f"{(time.perf_counter() - t) * 1000:.0f} ms", dets[:5])
```

**Do the stock HEF first (go/no-go 1).** Run `HailoDetector("yolov8n_stock_dfc5.1.0.hef", n_classes=80)` on a 640 x 640 photo. Resizing is fine for this check only. You should see sensible COCO boxes (class 0 = person). This proves the assumptions `ayas/npu.py` makes about HailoRT 5.1.1. If any of them is wrong, fix `HailoDetector` / `parse_nms_by_class` against the stock HEF before trusting a custom one:

- `VDevice(params)` with `HailoSchedulingAlgorithm.ROUND_ROBIN`, `create_infer_model(hef)`, `set_batch_size(9)`.
- `input().set_format_type(UINT8)` and `output().set_format_type(FLOAT32)`.
- `configure()` used as a context manager.
- Per frame: `create_bindings()`, `input().set_buffer(uint8 HxWx3)`, `output().set_buffer(np.empty(output().shape, float32))`.
- `run(bindings_list, timeout_ms)`, then `output().get_buffer()`.
- **Output format:** HAILO_NMS_BY_CLASS. Either a list of per-class `(k, 5)` arrays `[y_min, x_min, y_max, x_max, score]` normalised 0..1, or the packed flat buffer. Both are parsed.
- **Short batches** are padded with black tiles to the full batch size.

**Go/no-go 2.** The custom HEF passes `hailortcli run`. Then run `training/eval_plates.py --model ayas_v8n.hef` and `--model ayas_v8n.onnx` on the same TRAIN-val grids. The HEF must be within 5 mAP points of ONNX FP32. Never tune on a test plate (P5, or the lighting matrix). `eval_plates.py` needs only numpy and OpenCV on the Pi, plus `onnxruntime` for the `.onnx` run.

The lighting experiment can also run on the HEF: `training/eval_lighting.py --matrix <dir> --model burst=ayas_v8n.hef --arms burst` (the burst arm only; the naive arm has its own model).

## CPU fallback (if go/no-go 2 fails)

- **ONNX Runtime:** run `pip install onnxruntime` in the Pi venv, then use `ayas.npu.OnnxDetector("model.onnx")`. It is the same code path as the Mac. It takes about 130 ms per tile on the M3; it has not been measured on the Pi.
- **NCNN:** `yolo export model=best.pt format=ncnn imgsz=640` (training venv). On the Pi, `pip install ultralytics ncnn` (this pulls in CPU torch), then `UltralyticsDetector("best_ncnn_model")`. The plan estimates about 0.6-0.9 s for 9 tiles (est.).
- Either way, stop Hailo work, ship YOLO on the CPU and state it plainly in the report.

## Compiling on JarvisLabs instead of Colab (`compile_hef_jl.sh`, used 22 Nov 2026)

The HEF in `models/v8n_burst.hef` came from this script on an A30 box (~20 min, ~₹17). Put the DFC wheel, `best.onnx` and `calib_640.zip` in `/home/hailo_job`, then run `jl run --on <id> -- bash /home/hailo_job/compile_hef_jl.sh`. Four traps it handles, each found the hard way:
1. **Python 3.10 venv**: DFC 5.1.0 pins TensorFlow 2.18 and onnxruntime 1.18. The script installs `tensorflow[and-cuda]==2.18.0` so the GPU is used.
2. **Model Zoo install**: `pip install --no-build-isolation -e hailo_model_zoo`. Its `setup.py` checks that the DFC is installed, which an isolated build cannot see.
3. **`setuptools<70`**: the DFC SDK still imports `pkg_resources`.
4. **End nodes**: Model Zoo v5.1.0's `yolov8.yaml` has no parser end nodes. The automatic guess cuts at `/model.22/dfl/Reshape`, which the yolov8 NMS post-process rejects ("expected conv but found depth_to_space"). Pass `--start-node-names images --end-node-names` with the six head convs (`/model.22/cv{2,3}.{0,1,2}/...2/Conv`).

**Result on the Pi:**
- `hailortcli parse-hef` shows 3 contexts, UINT8 640×640×3 input and on-chip NMS for 6 classes.
- `HailoDetector` takes 47 ms per inspection.
- The verdict matches the ONNX model on 27/28 twin bursts. The difference is the known edge false positive, which disappears.
- `hailortcli run` is not supported on the Hailo-10H: use `run2`.
