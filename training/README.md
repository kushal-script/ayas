# Training, the lighting experiment and evaluation of the surface-defect detector

YOLOv8n, 6 steel classes (`scratch, dent, pit, corrosion, crack, stain` = `trinetra.tiles.CLASSES`, ids 0-5), trained on **640 px tiles cut from the rectified 0.08 mm/px plate grid**. Tiles are never resized, here or on the Pi. The binding contract is `docs/twin/DESIGN_STEEL.md` (section G for the lighting experiment), with `docs/twin/DESIGN.md` for frames and tiles. No script hard-codes the class list: they read `trinetra.tiles.CLASSES` or the model's own names.

All scripts run with the training venv and have `--help` (`eval_lighting.py` and `eval_plates.py` also run in the dev venv with an `.onnx` model):

| Script | Does |
|---|---|
| `train_arms.py` | **Phase 1.** One YOLOv8n per capture arm (`naive` = old single frame, `burst` = new 5-frame method) from the same twin plans; export; optional lighting evaluation |
| `eval_lighting.py` | **Phase 1.** The lighting experiment on the test matrix (`trinetra.twin.matrix`): P/R/F1, retakes, silent failures, verdict agreement, SSIM vs C0, PASS/FAIL; `reports/lighting_matrix.{md,json}` + `reports/figures/` |
| `build_datasets.py` | Image lists for the 5 sim-to-real ablation rows: `runs/datasets/<row>/{train,val,test}.txt` + `data.yaml` + `manifest.json` |
| `train.py` | Ultralytics training with the native-scale settings; `--test` adds tile mAP50 on the real test tiles |
| `export_onnx.py` | `best.pt` -> `models/<name>.onnx` + `<name>.json` (class names); checks onnxruntime against PyTorch and the Hailo cut nodes |
| `eval_plates.py` | Plate-level metrics on full test grids (matching rules shared with `eval_lighting.py`) |
| `ablation.py` | The 5 rows end to end, then `reports/ablation.md` (Phase 2, once real steel captures exist) |
| `calib_set.py` | 1,024 TRAIN tiles for the Hailo INT8 calibration (`--allow-twin` in Phase 1) |

## The lighting experiment (Phase 1)

The claim to test: **the inspector works in different lighting**, because of the capture method (burst, ambient subtraction, min(L, R), quality gate, hood), not because of the network. The old method is the baseline arm. Both arms get the same plans, renders and tile pipeline; only the capture processing differs.

1. **Training renders, both arms.** Post writes the tiles of `post.train_arm` and deletes each work dir after it succeeds, so render in chunks with `--keep-work` and post the kept work dirs a second time with `train_arm: naive`. A kept work dir is about 63 MB (36 MB EXR + 27 MB maps, build reports), so a chunk of 100 scenes needs about 6.3 GB; `run` stops below 2 GB free, and re-running the same loop fills the gaps (scenes with a sidecar are skipped, and only new work dirs get the naive pass):

   ```bash
   .venv/bin/python training/train_arms.py --write-naive-config runs/twin_naive.yaml
   for s in $(seq 0 100 1400); do
     .venv/bin/python -m trinetra.twin.cli run --n 100 --start-seed $s --out data/twin/arm_burst --keep-work
     .venv/bin/python -m trinetra.twin.cli post --work data/twin/arm_burst/work --out data/twin/arm_naive --config runs/twin_naive.yaml
     rm -rf data/twin/arm_burst/work
   done
   ```

   Training seeds are always below 900000.
2. **The test matrix.** Every test plate under every condition with the open device and with the hood head (device v2, DESIGN_STEEL C2): 14 rows per plate. The plate, defects and distractors are identical across a plate's rows; seeds start at 900000 and are refused by `train_arms.py` and `calib_set.py`. At least a quarter of the plates are defect-free (`--min-clean` of `python -m trinetra.twin.matrix`; research: 2-3 of 8-10), so false rejects of clean parts are measured too:

   ```bash
   .venv/bin/python -m trinetra.twin.cli matrix --n 10 --start-seed 900100 --out data/twin_matrix --overview   # 10 x 14 rows
   # (= python -m trinetra.twin.matrix --n 10 --out data/twin_matrix --render --overview)
   ```

   Post must have `eval_arms: [burst, naive]` (the default) so that it writes `grids/<arm>/test/<scene>_v<k>.png` for both arms. Cost: 140 renders at about 41 s each (full resolution, 96 samples) is about 1.6 h on the M3, plus post. **Start at seed 900100 or later**: the device v2 gate recalibration (14 Nov, `configs/capture.yaml`) used a pilot on matrix seeds 900000-900004, and the full-resolution pilot (15 Nov, `reports/pilot_lighting.md`) tuned the closure, coarse-closure and local-ambient gate rules on seeds 900011, 900027, 900028 and 900068, so those plates are no longer untouched test plates.
3. **Train, export, evaluate** (training venv):

   ```bash
   .venv-train/bin/python training/train_arms.py --burst data/twin/arm_burst/yolo --naive data/twin/arm_naive/yolo \
       --matrix data/twin_matrix          # add --dry-run to print the commands, --quick for a smoke run
   ```

   Each arm trains on its own tiles of the same renders (`--match none`, the default; the burst arm has no tiles of refused variants). `--match scenes` would also drop those scenes from the naive arm, which weakens the baseline, so it is only an option. Both trees' sidecar scene lists are compared and a mismatch is reported in the dataset manifest. This trains `runs/train/arm_{naive,burst}`, exports `models/v8n_{naive,burst}.onnx` and runs:

   ```bash
   .venv/bin/python training/eval_lighting.py --matrix data/twin_matrix \
       --model naive=models/v8n_naive.onnx --model burst=models/v8n_burst.onnx --out reports
   .venv/bin/python tools/make_demo_video.py --report reports/lighting_matrix.json \
       --overviews data/twin_matrix/overview --out reports/demo_lighting.mp4
   ```

**What `eval_lighting.py` reports**, per model x arm x (condition, device mode):
- defect P/R/F1 with Wilson 95 % intervals, over the captures the gate accepted (F1 interval approximate);
- FP per clean plate;
- retake rate (the naive arm has no gate);
- **silent failures**: the gate let the capture through, but the verdict (`rules.judge_steel` on the detections) differs from the ground-truth verdict (`judge_steel` on the ground-truth defects, same box -> mm measures). "Unsafe" is the subset where the device was milder than the truth and did not hold the part for a person (PASS on a REWORK part is unsafe; HOLD_FOR_QA on a REJECT part is wrong but not unsafe). A RETAKE is never a silent failure;
- verdict agreement with the truth and with the same plate's C0 capture of the same device (hood rows against C0 hood);
- SSIM of the grid against C0 for the same plate and device, over the accepted captures (`ssim_c0_all` also counts the refused ones);
- the gate reasons of refused captures and the main one (the reason whose message was shown).

Two ways to turn boxes into mm (`--measure`):
- `box` (default): detections and ground-truth boxes are measured the same way (`box_defect`): every box is padded to `label.min_box_px` (8 px) per side first, as post pads every training label, so a perfect detector gets the ground-truth verdict. This isolates what the lighting does to detection. A box only bounds a thin defect, so a diagonal 0.1 mm scratch gets the width of its box on both sides: the verdicts are consistent, not physical.
- `app`: detections are measured exactly as the on-device app measures them (`trinetra.app.pipeline.detection_record`: sizes from the defect's own pixels in the grid, so the lighting can change them), and the truth uses the analytic measures (`--gt-measure true`, converted as `twin.records.gt_verdict` does, plan params such as pitted rust included). This is the verdict the device would really show against the physical truth; run it as well and quote both.

A defect counts as ground truth when post found it visible in the plate's C0 (dark-room) capture with the same device (`--gt-visibility c0`; the ring and the pods show different defects), so a defect that glare hides in sun is a miss, not "invisible".

**PASS criterion** (device v2, 14 Nov; fixed before the test, stated in the report): HOOD (the hood head with its LED ring, the product configuration), every condition C0-C6: retake rate <= 5 %, F1 >= F1(C0, hood) - 0.05, verdict agreement with C0 (hood) >= 95 % and 0 silent failures. OPEN (the handheld without the hood): 0 silent failures in every condition; retakes are allowed and reported per condition with the main gate reason. A hood row where every capture was refused fails. Missing hood rows (or C0 open) make the result INCOMPLETE. Every table keeps the naive arm (the old method).

**Honesty rules for this experiment.** Matrix plates are never trained, tuned or calibrated on. Thresholds (`conf=0.25`, matching, the criterion) are fixed before looking at the matrix. These are twin numbers: they show the method works in simulation, not on real steel; say so wherever they are quoted.

## Setup (Mac)

```bash
python3.11 -m venv .venv-train
.venv-train/bin/pip install -r requirements-train.txt
.venv-train/bin/pip install -e . --no-deps
```

## The sim-to-real flow (Phase 2, real captures)

```bash
P=.venv-train/bin/python
$P training/build_datasets.py --synth data/twin/yolo --real data/real/yolo
$P training/train.py --data runs/datasets/real_only/data.yaml --name real_only --test
$P training/export_onnx.py --weights runs/train/real_only/weights/best.pt --name v8n_real_only \
    --check-tiles data/real/yolo/images/val
$P training/eval_plates.py --model models/v8n_real_only.onnx --grids <realdata out>/grids/test \
    --sidecars <realdata out>/sidecar --out reports/eval_real_only
$P training/calib_set.py --images data/real/yolo/images/train --out runs/calib   # for hailo/ (Phase 1: twin, --allow-twin)
```

Or all five rows at once: `$P training/ablation.py --test-grids <realdata out>/grids/test` (add `--dry-run` to see the commands, `--report-only` to rebuild the table).

## Training settings and why

- `imgsz=640`: tiles are already 640 px, so nothing is resized. `scale=0.1` keeps mm/px nearly fixed.
- `hsv_h=0.01, hsv_s=0.4, hsv_v=0.4` (lighting research, training recipe): a little hue jitter only (rust and stain colours carry meaning), more saturation and brightness jitter as the last line of defence after the capture method. Tiles are plain RGB, so flips are fine.
- `translate=0.1, degrees=0, fliplr=0.5, flipud=0.5, mosaic=1.0, mixup=0, close_mosaic=10`, fixed `seed`, `deterministic=True`.
- Fine-tuning (`--weights`) uses `optimizer=SGD, lr0=0.002, freeze=10` (backbone frozen). With Ultralytics' `optimizer=auto` the `lr0` is silently replaced (AdamW at about 0.001), so the optimizer is set explicitly.
- `--device auto` picks CUDA `0`, else Apple `mps`, else `cpu`. On MPS Ultralytics warns that two ops have no deterministic kernel; runs are close to, not bit-for-bit, reproducible.
- `--model yolov8n.pt` (COCO, downloads 6 MB on first use) is the default start. `yolov8n.yaml` trains from scratch and needs no download (smoke tests).

## Where to run

**Mac M3 (MPS).** Measured on 5 Nov with toy 640 px tiles: 30 epochs on 20 tiles took 134 s, about 1.2-1.5 s per batch of 8 after a 5-20 s warm-up. That suggests roughly 2-3 min per epoch for 860 real tiles, so 150 epochs is about 5-7 h (estimate): fine for one row overnight, use a GPU for the full ablation.

**Colab (T4/L4).** Zip `src/ training/ configs/ pyproject.toml requirements-train.txt` and the two YOLO trees to Drive, then:

```bash
!cd /content && unzip -q /content/drive/MyDrive/trinetra_train.zip -d trinetra
!cd /content/trinetra && pip install -q ultralytics==8.4.202 onnx==1.23.2 onnxruntime==1.31.1 onnxslim==0.1.99 && pip install -q -e . --no-deps
!cd /content/trinetra && python training/ablation.py --epochs 150 --test-grids <grids> --device 0
```

Keep Colab's CUDA torch if it is 2.15.x; otherwise note the version (every run records it in `trinetra_train.json`). Copy `runs/`, `models/` and `reports/` back to Drive before the session ends.

**JarvisLabs GPU (`jl` CLI).** Use a bundle rather than a directory run, so the venvs and `data/` are not rsynced:

```bash
tar czf /tmp/trinetra_train.tgz src training configs pyproject.toml requirements-train.txt data/twin/yolo data/real/yolo
jl create --gpu L4 --storage 50 --yes --json                       # note the machine_id
jl upload <id> /tmp/trinetra_train.tgz /home/trinetra_train.tgz
jl run --on <id> --json --yes -- sh -lc 'mkdir -p /home/trinetra && cd /home/trinetra && tar xzf ../trinetra_train.tgz && pip install -q ultralytics==8.4.202 onnx==1.23.2 onnxruntime==1.31.1 onnxslim==0.1.99 && pip install -q -e . --no-deps && python training/ablation.py --epochs 150 --device 0'
jl run logs <run_id> --tail 50                                     # poll every few minutes
jl download <id> /home/trinetra/runs ./runs -r && jl download <id> /home/trinetra/reports ./reports -r
jl pause <id> --yes --json                                         # stop billing
```

## Ablation protocol (5 rows)

| Row | Train | Notes |
|---|---|---|
| `real_only` | real train | the baseline |
| `real_25` | 25 % of the real train **captures**, stratified by coupon | how much real data the twin can replace |
| `synth_only` | twin train | expected to be worst on real plates |
| `synth_then_real` | stage 1 twin, stage 2 real (fine-tune) | the claim we test |
| `mixed` | twin + real (`--real-oversample k`) | |

- Same seed, epochs, val set and test set for every row. Val is the real TRAIN-val split (a held-out placement of the training plates); it only picks `best.pt`.
- Test is the real test split (P5, plus whatever `realdata.py` puts there). It is used once per row, at the end.
- Report from `reports/ablation.md`: per-class recall per **physical** defect with Wilson 95 % intervals, defect precision, false positives per clean plate, and tile mAP50 as a secondary number.

## Honesty rules

- **P5 is never used for training, validation, calibration, threshold choice or twin tuning.** `build_datasets.py` refuses a test image that is also in train/val; `calib_set.py` refuses test-split and P5 images.
- Thresholds (`conf=0.25`, match `IoU>=0.25` or `cover>=50 %`) are fixed before the test plates are looked at.
- Every metric we quote comes from **real photos**. The claim is "simulation-pretrained, real-calibrated", never "trained in sim, works directly".
- Count physical defects, not captures, for intervals: 16 captures of one dent are one dent.
- If `synth_then_real` does not beat `real_only` on P5 by 13 Nov, ship `real_only` and keep the ablation as a result.
- The Hailo INT8 calibration uses TRAIN tiles only, never test. Real burst captures when they exist; in Phase 1 the twin burst-arm tiles (`calib_set.py --allow-twin`), because that is what the deployed pipeline feeds the model.

## Evaluation rule (`eval_plates.py`)

Full rectified grids go through `trinetra.npu.detect_grid` (same tiling and seam merge as the Pi). A detection matches a same-class object if IoU with its box is at least 0.25, or the box covers at least 50 % of the object's mask (thin scratches and cracks have sliver boxes, so IoU alone is unfair). Duplicates on one object are true positives and are counted separately. Detections on `visible: false` defects are ignored. Clean plates are grids with no labelled defect.

## Pitfalls seen so far

- Ultralytics reads numpy images as **BGR**; every `trinetra.npu` detector takes **RGB** tiles (convert after `cv2.imread`).
- Ultralytics writes label caches (`labels/<split>.cache`) next to the label folders.
- With an untrained model, ONNX and PyTorch detections can differ at very low `conf` because dozens of anchors tie at the same score; the raw-output check is what matters (measured max differences: 2e-6 score, 6e-4 px).
- Disk is tight: never use `cache=disk`; `calib_set.py` for 1,024 tiles needs about 1.9 GB (`--no-npy` saves 1.26 GB).
