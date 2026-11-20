# Models (not committed)

Weights are not in git (see `.gitignore` and NOTICE). Put them in this folder.

| File | What | How it was made | SHA-256 |
|---|---|---|---|
| `v8n_burst.onnx` (+ `.json`) | YOLOv8n, 6 steel classes, 640 px tiles of the burst-processed grid. **The device model** | `training/train_arms.py` on JarvisLabs (L4), 150 epochs from COCO `yolov8n.pt`, run `r_82a2be7e` (15 Nov 2026); `training/export_onnx.py` | `84933c6145ca01f8d868c6728b65d62ee833d0ed0f80a42740df18ca95cc6dde` |
| `v8n_naive.onnx` (+ `.json`) | Same, trained on the old single-frame capture. Baseline arm of the lighting experiment only | same run | `43db93e43e8742c9332edab6511d6f0ff58f30c011cf27e444fbc6f6ee1e0756` |
| `llama3_2_1b_instruct.pte` + `tokenizer.model` | Llama 3.2 1B Instruct for ExecuTorch (advisor) | Meta `meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8` (4-bit weights, group 32, 8-bit embeddings), exported on the Pi 5 on 1 Nov 2026 with ExecuTorch v1.6.0-rc1 (`80264e11d`), XNNPACK; on the Pi it is `~/executorch/llama32_1b_spinquant.pte` (1.08 GB), runner `~/executorch/cmake-out/examples/models/llama/llama_main` | `85f3af53e1cf4ec6d418b8402e9120ca59e13d538602c6facea7269d4710c6ec` |
| `v8n_burst.hef` | `v8n_burst.onnx` compiled for the **Hailo-10H** (AI HAT+ 2), `--detector hailo:models/v8n_burst.hef`; class names from `v8n_burst.json` | 22 Nov 2026 on JarvisLabs (A30) with `hailo/compile_hef_jl.sh`: DFC 5.1.0 + Model Zoo v5.1.0 (`c13e20e`), Python 3.10, 1,024 twin TRAIN burst tiles for INT8 calibration (`training/calib_set.py --allow-twin`), cut at the six head convs, on-chip NMS (6 classes, score 0.2). Outputs, logs and HAR in `artifacts/jarvislabs/trinetra-hailo_583671/` | `4b3ae346ebe5a51510cbd6d5a6041db9819c5449d02b874e00efd48a62ea9840` |

Each `.json` next to an ONNX records the export:
- the classes;
- the input size (640);
- a torch-vs-ONNX check on sample tiles;
- the SHA-256.

The PyTorch checkpoints (`best.pt`, `last.pt`), training curves and per-arm datasets lists are archived with the run outside git.

Check a download:

```bash
shasum -a 256 models/v8n_burst.onnx   # must print the hash above
```
