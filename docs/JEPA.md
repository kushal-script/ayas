# JEPA image models for Ayas (cross-lighting I-JEPA, anomaly map)

Status, 15 Nov 2026: **phase 1 of 2.** The pipeline is built and proven end to end, both on the Mac and on one paid JarvisLabs L4 smoke run. Every number in this file is a **SMOKE number, NOT A RESULT**: the model was trained on 2 low-resolution training plates. The full training waits for the paired training renders (section 4).

Contract: `docs/twin/DESIGN_STEEL.md`. Code: `training/jepa/` (training side) and `src/ayas/anomaly.py` (inference side; not wired into the app). Tests: `tests/test_jepa.py` and `tests/test_jepa_anomaly.py`. The torch tests skip in `.venv`; run them with `.venv-train`. Reports: `reports/jepa_anomaly.md` and `.json`, `reports/jepa_onnx.md`.

## 1. Why JEPA here, mapped to the brief

| Brief | What YOLO gives | What the JEPA adds |
|---|---|---|
| "Detect surface defects, cracks and **anomalies**" | 6 trained classes: scratch, dent, pit, corrosion, crack, stain | (a) A **patch anomaly map**. Each patch is scored against a memory bank of normal-surface patches (PatchCore-style), or by the JEPA predictor's error when the patch is masked. It needs no labels, so it can flag a surface anomaly outside the 6 classes, e.g. a weld spatter, an inclusion, a lamination edge or an unseen corrosion form. It is reported as "unexplained" when no YOLO box covers it. |
| "Classify type and severity" | class, box, size → rules | (c) Optional: a linear probe on the JEPA patch features, as a second opinion on YOLO's class for each box. It is not built yet; it is a phase-2 check. |
| "**Must work in every lighting condition**" (user's hard requirement) | The capture method does the work: burst, ambient subtraction, min/mean of L/R, the quality gate and the hood | (b) **Lighting-invariant features** from a **cross-lighting** objective. The twin renders the SAME plate under C0–C6 on pixel-aligned grids, so the model is trained to predict the same region's embedding under C0 (dark) from a capture under condition X. If the invariance holds, the anomaly map does not fire on a lighting change. |
| "Real time at the edge" | YOLOv8n on Hailo-10H | ViT-Tiny/16 (5.5 M parameters; the Hailo Model Zoo compiles `vit_tiny` for Hailo-10H with DFC 5.1.0). The CPU fallback is about 15 ms (INT8) to 40 ms (fp32) per 224 crop on 4 Mac threads, projected to 44–72 ms (INT8) on the Pi 5 (section 6); the anomaly map runs after the alert, like the advisor. |

Why JEPA rather than a pixel autoencoder or contrastive learning:
- I-JEPA predicts in **embedding space**, so it does not spend capacity reproducing pixel noise, sensor grain or brushed-steel texture. Those are exactly what changes between captures.
- It needs **no hand-made augmentations** to define "the same thing". The twin supplies real positives instead: the same plate region under another light.
- It is the method the user asked for.

The honest framing stays the same as in DESIGN_STEEL: **robustness comes from the capture method; the network is the last line.** Section 7 says what JEPA may and may not claim.

## 2. Architecture and objective

```
plate grid (0.08 mm/px, canonical plate frame)
   │  crop 224 px (scale 1: 17.9 mm)  or  448 px -> 224 (scale 2: 35.8 mm)
   ├── context image  = the crop under condition X (+ photometric jitter)
   │      └─ context encoder f_θ (ViT-Tiny/16, only the context patches) ─┐
   │                                                                    ├─ predictor g_φ (narrow ViT, 96-d, 4–6 blocks)
   │                                                 mask tokens @ target positions ─┘        │
   └── target image   = the SAME pixels under the anchor (C0 of the same device)              ▼
          └─ target encoder f_θ̄ (EMA of f_θ, full image) → LayerNorm → h at target positions  ↔  ŷ
loss = smooth-L1(ŷ, h): hidden targets weight 1; context-visible targets weight 0.5 (cross pairs only)
```

- **Encoder.** timm `vit_tiny_patch16_224.augreg_in21k_ft_in1k`: 192-d, 12 blocks, 3 heads, 5.52 M parameters. `vit_small` (384-d, 22 M) is a config option. The class token is never used: patch tokens plus their position embeddings go through the blocks. So the encoder runs on any subset of patches (the I-JEPA context), and exports to ONNX as a plain ViT on the full image (`model.Encoder`).
- **Init.** `imagenet` (default) or `scratch`. With tens of training plates, an ImageNet start is the practical choice. The same ImageNet ViT-Tiny, without JEPA, is the baseline that the JEPA must beat (section 5), so the comparison isolates what the JEPA objective adds.
- **Predictor.** Linear 192→96, a learned mask token, fixed 2-D sin-cos positions, 4 (smoke) or 6 (full) timm blocks of width 96 with 3 heads, LayerNorm, then Linear 96→192 (0.48 M parameters at depth 4).
- **Masking** (I-JEPA multi-block, `masks.py`):
  - 4 target blocks, each covering 15–20 % of the 14×14 patch grid with aspect 0.75–1.5;
  - 1 context block covering 85–100 % at aspect 1, with every target patch removed;
  - lengths are cut to the batch minimum, using a random subset rather than the first k patches, so there is no top-left bias.
  - Simplification: the 4 blocks are predicted in one predictor pass, not 4. The mask tokens can attend to each other, but they carry only their position.
- **EMA.** The target encoder's momentum rises linearly from 0.996 to 1.0. Its weights never get gradients.
- **Loss.** Smooth-L1 between the prediction and the LayerNorm-ed target embedding (I-JEPA). The collapse monitor `tgt_spread` (the std of the normalised target embeddings across the batch) is logged; it stayed at 0.035–0.06 in the smoke runs, i.e. no collapse.
- **Cross-lighting variant (b).**
  - With probability `cross_prob` = 0.75, the **context** crop comes from a non-C0 accepted row X (C1–C6) and the **target** crop is the same canonical pixels of the anchor row: C0 of the same plate, device mode and post variant. `anchor: any` uses another accepted condition instead; `scope: all` also pairs open↔hood.
  - The rest are standard self pairs: context and target come from one image.
  - On cross pairs, the predictor must also output the anchor's embedding at 16 **context-visible** positions (`n_visible`, weight 0.5). This is the direct invariance term. The hidden-block term keeps the predictive, semantic part of I-JEPA.
  - On self pairs, visible targets get weight 0, because they would be the identity.
- **Geometric augmentation** (the same flip and 90° turn on both images). **Photometric jitter on the context only**: gain 0.6–1.5, gamma 0.7–1.4, a smooth 3×3 shading field of ±25 % and noise σ ≤ 0.02, with p = 0.5. This is DESIGN_STEEL item 6, "the last line".
- **RGB, not grayscale (decision).**
  - Colour carries class information here: orange-brown rust against an oil stain; white zinc corrosion on galvanised or aluminium; the blue-black of mill scale.
  - `training/README.md` already limits hue jitter for YOLO for the same reason.
  - In the burst arm, colour is lighting-stable by construction: ambient is subtracted, and the white balance is fixed to the LED white point (`capture.yaml leds.colour`). So RGB does not reintroduce the ambient CCT.
  - The IMX500 delivers RGB, and the ImageNet init and baselines are RGB.
  - The cost of the 3-channel patch embedding is negligible: 0.15 M multiply-accumulates per patch against about 5.5 M for the blocks.
  - `model.in_chans: 1` remains possible for an ablation with `init: scratch`.
- **Mixed scales.** Scale 1 is 224 px at 0.08 mm/px: 1.28 mm patches, for pits, thin scratches and crack tips. Scale 2 is 448 px downsampled to 224: 2.56 mm patches and 35.8 mm of context, for rust patches, stains and dents. Each batch mixes them 50/50. At inference both scales are scored and their z-maps averaged; the app proposal uses scale 2 only, which needs 9 crops per 100 mm plate.
- **Determinism.** Each crop, row choice, mask and augmentation is drawn from `(seed, sample index)`, so a run does not depend on the number of loader workers. CUDA runs use `use_deterministic_algorithms(warn_only)` plus `CUBLAS_WORKSPACE_CONFIG`; flash-attention backward is non-deterministic and the run says so. MPS is close but not bit-exact.
- **Checkpoints.** `last.pt` holds the model, optimiser and step, for `--resume`. `encoder.pt` holds the model and cfg and is what eval and export load. AMP is bf16 autocast on CUDA. The config yaml lives in `training/jepa/configs/{smoke,full}.yaml`, with dotted `--set` overrides.

## 3. Data: pairing, the variant gate, splits

- **Pixel alignment, checked on the pilot.**
  - Within one device mode, a plate's rows share the camera pose. Phase correlation of the grids against C0 gave shifts ≤ 0.02 px on hood rows and ≤ 0.43 px on accepted open rows. Open C6 (RETAKE) was not aligned, as expected for a refused capture.
  - Between the open and hood rows the grid can be **turned by k × 90°**: it follows the image's corner order (`datum_corner` and `H_grid_plate` in the sidecar).
  - `grids.canonical_affine` maps every grid exactly to one plate frame (datum top-left, x = u/0.08 + 62.5). That is an integer affine, so the warp is exact. The same region therefore has the same pixels in all 14 rows. Tested for all 4 turns in `tests/test_jepa.py`.
- **The variant gate.** Only rows whose burst-arm gate is not RETAKE are used. A refused capture never reaches the app, so it must not shape the features. In the smoke render, 9 of 56 variants were refused, all open-mode C1/C3/C5/C6 rows on brushed or 2B stainless plates.
- **Arms.** Training uses `burst` grids only, the deployed method. The naive arm is the evaluation baseline (section 5).
- **Seeds and splits.**
  - JEPA training plates use seeds 100000–899999.
  - `pairs.check_train_seed`, `grids.refuse_test_plates` and `PairCrops` refuse seeds ≥ 900000 and split `test`.
  - The memory bank is built from training plates only.
  - The test matrix is never trained, tuned or calibrated on. **Its plates must have seeds ≥ 900100**: the capture gate was tuned on 900000–900068, and `eval_anomaly.py --min-test-seed` defaults to 900100.
  - The pilot plates (900011, 900027, 900028, 900068) are used for the smoke evaluation only, with `--min-test-seed 900000`. They are tuning plates, so they can never give a result.
- **`ayas-twin matrix --start-seed 100000` does NOT work.** `twin/matrix.py` raises for any start seed below 900000, on purpose: the matrix is the test set. `training/jepa/pairs.py` writes the same rows for training seeds instead. It uses `sampler.make_plan` with the condition and device overrides, checks that plate, defects and distractors are identical across rows, and renders through `pipeline.run_plans` with a config copy that writes compact grids:
  - plan split `jepa_train`, which is post's only grid split here;
  - no YOLO tiles;
  - burst arm (add `--arms burst naive` for the baseline arm);
  - JPEG q95;
  - 2 post variants.

  It also writes `pairs_manifest.json`, which has the same layout as `matrix_manifest.json`.

## 4. Data plan (training renders)

**Recommendation: K = 48 paired training plates**, seeds 100000–100047, with all 14 rows (7 conditions × open/hood). This is the default of the render tooling:

```bash
# on the Mac, with ayas-render RUNNING (tools/jl/README.md)
jl run --on <id> --json --yes -- sh -lc 'sh /home/ayas_jobs/render_all.sh --jobs jepa --jepa-n 48'
jl run --on <id> --json --yes -- sh -lc 'sh /home/ayas_jobs/render_all.sh --stage post --jobs jepa --jepa-n 48 --post-tag final'
#   -> /home/data/jepa_train/pairs_manifest.json, plans/, work/ (EXRs kept), post_final/grids/burst/jepa_train/*.jpg + .json
```

- `ayas-twin matrix --start-seed 100000 --n K` does **not** work: `twin/matrix.py` refuses any start seed below 900000. `render_all.sh` writes the plans with `python training/jepa/pairs.py --n K --start-seed 100000 --out /home/data/jepa_train --arms burst` instead. It renders through `tools/jl/twin_jobs.py` with the EXRs kept and post deferred, then posts with `pairs.jepa_config`.
- Add `--jepa-arms "burst naive"` if the naive-arm memory bank is wanted (section 5); it adds about 1.1 MB per scene.
- **Why 48.**
  - At least 4–5 plates of each of the rarest materials (ss_2b and galvanised, 10 % each).
  - About 1,340 grids, about 1,150 of them accepted after the open-mode gate.
  - Each material × device mode gets enough normal patches for a 4096–10000-patch coreset.
- **Why all 14 rows.** The sealed hood rows (C1h–C6h) differ from C0h by only about 1.5 DN, so they carry almost no cross-lighting signal. But the hood is the product configuration, and its rows still give the memory bank hood-mode normal patches with independent sensor noise. On the L4 they cost little.
- **A cheaper 10-row variant** (7 open + hood C0, C4, C6; −29 % render time) is possible. Write the plans before `render_all.sh`, which reuses an existing manifest: `python training/jepa/pairs.py --n 48 --start-seed 100000 --out /home/data/jepa_train --arms burst --hood-conditions C0_dark C4_side_lamp C6_sun`.
- **Render all 48 in one go** (about 1.5 h, about ₹66). A 16-plate first pass is not worth it at this cost: extending it to 48 would mean rewriting the manifest, which the render README forbids once scenes are rendered.

**Random (unpaired) training scenes: reuse the YOLO training renders; render nothing extra.**
- The render job `train` (800 random scenes, seeds 0–799, `/home/data/twin_train`) writes YOLO tiles only. If its post also wrote full burst-arm grids (`post.grid_splits: [test, twin_test, train]`, `grid_format: jpg`: +1.1 MB per scene, 0.9 GB for 800 scenes), the JEPA could use them as self pairs and as memory-bank material: `--remote-data /home/data/jepa_train/post_final /home/data/twin_train/post_final` plus, after `--`, `--set "data.splits=[jepa_train,train]"`.
- That post setting belongs to the render job's config, so it is a proposal here. Without it, the 48 paired plates are the whole training set. That is enough for the full run; the extra scenes mainly widen the bank's material and pose coverage.

| Item | Measured / estimated |
|---|---|
| Render time, L4 (render job, `artifacts/jarvislabs/ayas-render_setup/MANIFEST.md`) | 6.6–8.7 s/scene inside Blender per process, 3 processes per L4. Their estimate for **40–60 plates × 14 rows: 1.25–1.9 h, ₹55–82**. **K = 48: 672 scenes ≈ 1.5 h ≈ ₹66.** The post pass is CPU only, about 5–7 min |
| Render time, Mac M3 | 17–20 s/scene full resolution (STATUS_STEEL), 23–37 s/scene in the render job's comparison. K = 48 would be 672 × (20–30 s + about 5 s post) ≈ 4.7–6.5 h. The smoke render here (0.5× scale, 32 spp, under contention from another Blender job): 28 scenes in 817 s including post |
| Disk per scene, compact (what JEPA reads) | Burst-arm JPEG q95 grids: 0.56 MB per 100×100 mm grid at full resolution (pilot PNG re-encoded), 0.47–0.52 MB at 0.5× (smoke). 2 variants + grid JSONs (11 kB each) + sidecar (25 kB) + plan (18 kB) ≈ **1.2 MB/scene**; 2.3 MB with the naive arm. **K = 48: ≈ 0.8 GB** (1.6 GB with naive) |
| Disk per scene, default post | PNG grids for both arms: 2 × 2 × 2.9 MB ≈ 11.6 MB/scene, plus bursts (`--save-bursts`) at 32 MB/scene. The pilot was 1.0 GB for 56 scenes |
| EXRs kept on the instance | About 55 MB/scene with the textures deleted (render job): about 37 GB for K = 48. They are needed only to re-post; the 200 GB disk holds them |
| On the Mac (about 8–12 GB free) | Never download the renders. Only the compact grids would fit (0.8 GB), and the full run trains on the instance. The local smoke uses 2 plates at 0.5× in the scratch dir (45 MB) |
| Upload when training on a fresh instance | `pack.py` builds a `.tgz` of the JPEG grids and JSON only; the smoke bundle was 29.6 MB for 56 grids |

**If `pairs.py --render` is ever run on the Linux instance directly, pass `--device OPTIX`.** `configs/twin.yaml` says METAL, which falls back to the CPU on Linux without any error. `render_all.sh` passes OPTIX itself. On the Mac (METAL) the direct call is fine:

```bash
.venv/bin/python training/jepa/pairs.py --n 48 --start-seed 100000 --out data/jepa_train --render   # Mac, ~5-6 h
```

The test data for the evaluation is the lighting matrix the render job makes: `render_all.sh --jobs matrix` (15 test plates × 14 rows, both arms, PNG grids, posted to `/home/data/twin_matrix/post_final`). It is never trained on. **It must start at seed 900100** (`ayas-twin matrix --start-seed 900100`), because 900000–900068 tuned the gate. `eval_anomaly.py` ignores test plates below `--min-test-seed` (default 900100) and stops if none are left.

## 5. Training and evaluation commands

**Local smoke (Mac, MPS)**:

```bash
.venv-train/bin/python training/jepa/pairs.py --n 2 --start-seed 100000 --out $SCRATCH/jepa/train_pairs --render --res-scale 0.5 --samples 32
.venv-train/bin/python training/jepa/train.py --config training/jepa/configs/smoke.yaml --data $SCRATCH/jepa/train_pairs \
    --out $SCRATCH/jepa/runs/smoke_local --workers 3          # 300 steps x 32 in 370 s (26 img/s, MPS under load)
.venv-train/bin/python training/jepa/eval_anomaly.py --ckpt <encoder.pt> --train $SCRATCH/jepa/train_pairs \
    --test data/twin_pilot --min-test-seed 900000 --coreset 4000 --smoke   # -> reports/jepa_anomaly.md (SMOKE)
.venv-train/bin/python training/jepa/export_onnx.py --ckpt <encoder.pt> --train $SCRATCH/jepa/train_pairs --name jepa_smoke
```

**JarvisLabs smoke (fresh instance, proven 15 Nov)**: this mode uploads a packed dataset, trains, collects and verifies the artifacts, then destroys the instance.

```bash
training/jepa/jl_train.sh fresh --data $SCRATCH/jepa/train_pairs --gpu L4 --name jepa-smoke --max-min 10 \
    -- --steps 2000 --batch 64 --workers 6
```

**JarvisLabs full run (on the render instance `ayas-render`; the data is already there, nothing is uploaded but code)**:

```bash
# 0. the lead resumes the instance (this runner never resumes it) and passes the id `jl resume` printed
# 1. train + evaluate on the test matrix + export ONNX, then collect, verify and leave the instance running
training/jepa/jl_train.sh on --machine <ayas-render id> \
    --remote-data /home/data/jepa_train/post_final \
    --eval-test /home/data/twin_matrix/post_final --export \
    --config training/jepa/configs/full.yaml --name jepa-full --max-min 120
# 2. latency of the final ONNX on the Mac (and later on the Pi): copy run_out/onnx from the artifacts dir
.venv-train/bin/python training/jepa/export_onnx.py --ckpt x --train x --out artifacts/jarvislabs/ayas-render_<run_id>/run_out/onnx \
    --name jepa --bench-only --runs 40 --report reports/jepa_onnx.md
```

What `on` mode does on that box, and why:
- **Code.** It runs `sh tools/jl/sync_code.sh <id>`, the render job's sync, to `/home/ayas` (`--no-sync` skips it). docs/, reports/ and artifacts/ are not synced and not needed. The runner also keeps its own `code.tgz` and sha256 of `training/jepa` + `src/ayas/anomaly.py` in the artifacts, as the code record.
- **Python.** The pytorch template's system Python is 3.10, and the project needs ≥ 3.11. `/home/venvs/ayas` (3.11) has no torch and is never touched. So the job makes `/home/venvs/jepa`:
  - uv Python 3.11 under `/home/.uv`, exactly as `bootstrap.sh` does, so it survives pause;
  - `torch==2.11.0` + torchvision from the cu130 index (`--torch` / `--torch-index` to change). That is the CUDA 13 build the JarvisLabs template shipped in the smoke run, and the L4 driver 595.58.03 supports it;
  - timm 1.0.31, onnx 1.23.2, onnxruntime, opencv, numpy and scikit-learn pinned by `tools/jl/constraints-mac.txt`;
  - the project `-e . --no-deps`.

  This is idempotent, and later jobs reuse the venv. timm downloads its weights to `HF_HOME=/home/.cache/huggingface`.
- **Data.** `--remote-data` and `--eval-test` point at the post dirs that hold `grids/`. The job stops with exit 3 if one has no grid JSON. The runner refuses an instance that is not `Running`, refuses `--after destroy`, and leaves the instance running by default; `--after pause` pauses only after the artifacts verify. If the verification fails, the instance stays running and the runner says so, so that nothing is lost.
- **Artifacts** go to `artifacts/jarvislabs/ayas-render_<run_id>/` (instance name + run id) with `MANIFEST.md`.

- `full.yaml`: 30000 steps × 128 crops, AdamW 1e-3 cosine with 1500 warm-up steps, wd 0.05, EMA 0.996→1, predictor depth 6, `crops_per_group` 16 (consecutive crops share a grid group, so the loader decodes few grids per batch), 8 workers.
- Estimate: the smoke run reached 932 img/s on the L4 (batch 64, bf16), so 3.84 M crops take about 70 min if the loader keeps up. Then `eval_anomaly.py` runs on the instance; the test grids and heat maps stay there, and the report, figures and JSON come back. `export_onnx.py` writes the ONNX files there (no benchmark). In total about 2 h, or ≈ ₹85.
- `--max-min` stops the training cleanly and checkpoints it. `--resume` continues from `last.pt`.
- **Artifact rule (standing).** Before any destroy or pause, the runner downloads into `artifacts/jarvislabs/<instance-name>_<run_id>/`:
  - the whole run output dir (`encoder.pt`, `last.pt`, `metrics.jsonl`, `summary.json`, `config.yaml`, eval report, figures, ONNX);
  - `<home>/jl-runs/<run_id>/` and the full `jl run logs`;
  - `jl run status`;
  - the remote environment (nvidia-smi, torch/timm, pip freeze);
  - the data manifests (never bulk data);
  - the exact code bundle with its sha256;
  - every jl command it ran.

  Then `training/jepa/artifacts.py` checks sizes, parses every JSON, **loads every checkpoint and runs a forward pass**, and writes `MANIFEST.md`. Only a passing check lets the runner destroy; a failed check pauses instead (in `on` mode it leaves the box running). In `on` mode the instance is never destroyed. pip installs go into the runner's own venvs, `/home/jepa_venv` (fresh mode, template torch) or `/home/venvs/jepa` (on mode), so the render job's packages are never touched. All three branches were dry-run offline with a stub `jl`: fresh pass → destroy, fresh fail → pause, and on → collect + verify + leave running. On mode also refuses a paused instance.

**Evaluation protocol** (`eval_anomaly.py`; thresholds and the gate are fixed here, before the test matrix is looked at):
- **Memory bank.**
  - One bank per arm that has TRAINING grids (the naive arm's own bank needs `--jepa-arms "burst naive"` renders; otherwise the naive test grids are scored against the burst bank, and the report flags it).
  - It holds every patch of accepted TRAINING grids whose window lies on the plate and whose footprint is ≥ 2 mm from any defect, labelled or not. At most 1500 patches per grid and scale are kept (16 GB Mac).
  - Greedy k-centre coreset (10 000; 4096 for the ONNX head) on a 128-d random projection.
  - z = (NN distance − μ) / σ, with μ and σ taken from held-out bank patches.
  - Methods: `jepa_knn` (EMA target encoder), `jepa_pred` (predictor error over 4 checkerboard groups of 2×2-patch squares), and the baselines `vit_in1k_knn` (the same ViT-Tiny with ImageNet weights only), `resnet18_knn` (PatchCore layer2+3) and `pixels_knn` (raw 16×16×3 patches).
- **Heat map per grid.** Built in the canonical plate frame, scales 1 and 2 averaged in z; saved as float16 at 1/4 resolution.
- **Pixel level.** Per class: the class's visible labelled-defect pixels against plate pixels > 0.5 mm from any defect.
- **Region level.** A defect's max inside its footprint (+0.5 mm) against the max of each 64 px (5.1 mm) normal cell ≥ 2 mm from any defect. Reported as AUROC and AP, and:
  - recall at 0.5 false-alarm cells per plate (an operating point set on the test normals);
  - **recall and false alarms per plate at the fixed z > 3** (deployable: set from the training bank only);
  - distractor alarms (marker, dust, blend, burr: unlabelled marks that should not fire).
- **Breakdown.** Per method × arm × device, per class, per condition, per material. Burst-arm RETAKE grids are skipped; the naive arm has no gate.
- **Invariance.** Scale-1 patch embeddings of each condition against C0 of the same plate, device and variant, centred by each model's training-bank mean:
  - cos(same patch);
  - cos(random other patch), the floor (a constant encoder is perfectly "invariant");
  - retrieval@1 (the patch's nearest C0 patch among the 196 in its window is itself).

  For the JEPA, the ImageNet ViT-Tiny, ResNet18 and raw pixels.

**Adoption gate.** The anomaly head goes into the app only if ALL of these hold on the twin TEST matrix (seeds ≥ 900100) with the trained YOLO burst model:
1. **Adds detection.** Among the visible ground-truth defects that YOLO misses (the `eval_plates` matching rules, conf 0.25), the anomaly head at the fixed threshold catches **≥ 30 %**, with a Wilson 95 % lower bound above 0, counted per physical defect, not per capture. The fixed threshold is z\* = 3.0 for now. Before the test it is recalibrated as the 99.5th percentile of normal-cell scores on 8 TRAINING plates held out of the bank, never on test plates. Hood rows are what count; open rows are reported.
2. **At an acceptable false-positive rate.** Unexplained regions at z\* number **≤ 0.5 per clean plate (hood)** and ≤ 1 per clean plate (open). Distractor alarms are ≤ 20 %.
3. **Beats the simpler baselines.** The region AUROC of the shipped JEPA method (knn or pred) is ≥ the best of {vit_in1k_knn, resnet18_knn, pixels_knn} + 0.02 on the burst arm, hood and open. A paired bootstrap over plates must give a 95 % CI above 0. If a baseline passes 1 and 2 but the JEPA does not beat it, **the baseline ships**, not the JEPA.
4. **More lighting-invariant than the baselines.** Mean retrieval@1 vs C0 over C1–C6, on accepted open-mode burst grids, is ≥ the best of (ImageNet ViT-Tiny, ResNet18, raw pixels) + 0.05. No condition may be worse than the ImageNet ViT by more than 0.02. Hood rows are reported, but they are nearly lighting-free already (section 8).
5. **Edge budget.**
   - ≤ 150 ms per 224 crop on the Pi 5 CPU (4 threads) or on Hailo;
   - ≤ 1.5 s per 100 mm plate at scale 2;
   - it runs after the alert and never delays the verdict.

If 1–3 fail, the head stays a report-only experiment. If 4 fails, nothing is claimed about JEPA invariance: the capture method is the lighting story, as now.

## 6. Edge deployment check

`export_onnx.py` exports opset 17 at batch 1. The models: `jepa_encoder` (patches → embeddings), `jepa_knn` (encoder + coreset head → (1, 14, 14) z), `jepa_knn_head` (the head alone, for an NPU encoder), `jepa_pred` (predictor error, 1 + 4 + 4 passes) and dynamic-INT8 variants. It also writes `latency.json` and `reports/jepa_onnx.md`, and checks onnxruntime against PyTorch.

Measured on the Mac M3 (onnxruntime 1.31.1 CPU EP, batch 1, 224×224). Other Blender renders and post jobs shared the machine (load average 13–50 on 8 cores), so the medians are upper-side numbers; the min of 40 runs is the better estimate for an idle core. Full table and op counts: `reports/jepa_onnx.md`. The model is the smoke model: latency does not depend on the training.

| model | MB | Mac 1 thread ms median (min) | Mac 4 threads ms median (min) | Pi 5 1 thread ms (proj.) | Pi 5 4 threads ms (proj.) | max abs err vs torch | ops outside the ViT set |
|---|---|---|---|---|---|---|---|
| jepa_smoke_encoder.onnx | 22.19 | 106.95 (90.55) | 62.81 (32.18) | 272–453 | 97–161 | 1.5e-05 | – |
| jepa_smoke_knn.onnx | 25.35 | 117.79 (103.1) | 63.71 (39.78) | 309–516 | 119–199 | 1.5e-05 | Clip, ReduceMin, ReduceSum |
| jepa_smoke_knn_head.onnx | 3.16 | 9.23 (8.24) | 2.93 (2.52) | 25–41 | 8–13 | nan | Clip, ReduceMin, ReduceSum |
| jepa_smoke_pred.onnx | 46.92 | 392.86 (228.43) | 122.58 (116.11) | 685–1142 | 348–581 | 2.3e-06 | ConstantOfShape, Equal, Expand, Gather, GatherElements, Where |
| jepa_smoke_encoder_int8.onnx | 6.36 | 34.12 (31.5) | 16.69 (14.64) | 94–158 | 44–73 | 0.63 | – |
| jepa_smoke_knn_int8.onnx | 7.16 | 31.03 (30.53) | 16.07 (14.5) | 92–153 | 44–72 | 0.71 | – |


**Pi 5 projection (estimate, not measured).**
- The Pi 5's Cortex-A76 at 2.4 GHz has 2 × 128-bit FMA pipes against the M3 P-core's 4 at 4.05 GHz: about 3.4× less peak fp32 per core, and about 6× less memory bandwidth (LPDDR4X ~17 GB/s).
- Published ONNX Runtime CPU numbers for small ViTs put a Pi 5 core at 3–5× slower than an Apple M-series P-core.
- FLOP check: ViT-Tiny/16 at 224 is about 1.25 G multiply-accumulates, i.e. 2.5 GFLOP (Hailo's table says 2.2 GOPs). At an achievable 12–15 GFLOP/s per A76 core in ORT fp32, that is 165–210 ms on 1 thread and 50–65 ms on 4.

**What this means for the ≤ 150 ms per 224 crop target on the Pi 5 CPU:**
- **fp32** encoder + kNN head (`jepa_knn`), 4 threads: about 120–200 ms projected from the Mac minimum. The FLOP estimate (1.25 GMAC ≈ 2.5 GFLOP at 12–15 GFLOP/s per A76 core in ORT, 4 cores at about 3.3× scaling) gives about 50–65 ms. So it is **borderline-to-OK**. The fp32 single-thread Mac number (90 ms) looks inflated by contention or E-core scheduling.
- **Dynamic INT8** (`jepa_knn_int8`): **44–72 ms on 4 threads**, 92–153 ms on 1 thread (projected). This **meets the target**. Its z-scores differ from fp32 by up to 0.7 (correlation 0.994), so the memory bank must be rebuilt with the INT8 encoder. The A76 has the ARMv8.2 dot-product instructions that the ORT INT8 kernels use, as the M3 does.
- **Per plate at scale 2** (9 crops of 448 → 224, 100 × 100 mm): about 0.4–0.65 s with INT8 on 4 threads. With scale 1 added (36 more crops), about 2–3 s. That is acceptable only because the map runs **after** the alert, like the advisor.
- **The predictor-error head** (`jepa_pred`, 1 target pass + 4 masked context passes + 4 predictor passes): about 350–580 ms per crop on 4 threads. It does **not** meet the budget on the CPU and needs gathers, so it is not Hailo-friendly either. It would ship only if it clearly wins the gate.
- **What it would take** to be safe: INT8 (CPU) or the Hailo HEF (about 7 ms per crop), scale 2 only, and measuring on the Pi itself:

  ```bash
  .venv/bin/python -c "from ayas.anomaly import OnnxPatchScorer; import numpy as np, time; s=OnnxPatchScorer('jepa_knn_int8.onnx', 4); x=np.zeros((448,448,3),np.uint8); s.patch_scores(x); t=time.perf_counter(); [s.patch_scores(x) for _ in range(20)]; print((time.perf_counter()-t)/20*1e3, 'ms')"
  ```

  `ayas.anomaly` needs only numpy, cv2 and onnxruntime, so it runs in the Pi venv without torch. On the Mac, the dev venv ran it on a full pilot grid: 9 windows in 1.6 s with fp32 on 4 threads, under load.

**Hailo-10H (AI HAT+ 2) notes.**
- **Supported architecture.** Hailo Model Zoo v5.1.0 lists **vit_tiny for Hailo-10H**: 137 FPS at batch 1, 2.2 GOPs, top-1 75.5 float / 74.6 on hardware. That table was compiled with **DFC 5.1.0**, the version this project pins (`hailo/README.md`: the Pi's HailoRT 5.1.1; never compile with 5.5). vit_small runs at 113 FPS. So the ViT-Tiny encoder is in the compiler's comfort zone: conv patch-embed, MatMul attention, Softmax, LayerNorm, GELU.
- **Graph differences from the zoo model (risk).** Ours has no class token and no head. Its output is the 196×192 patch-token tensor, not logits. The DFC must accept a Reshape/Transpose-free output, or a final reshape to 14×14×192 (NHWC). That is likely fine, but it is untested until a Colab compile with DFC 5.1.0 (the same notebook as the YOLO HEF).
- **Ops in our ONNX** (`reports/jepa_onnx.md`). The encoder graph has Conv 1, MatMul 72, LayerNormalization 25, Softmax 12, Erf 12 (GELU), Add/Mul/Div/Sqrt, Reshape/Transpose/Split/Squeeze/Slice/Concat, Shape 13 and Cast 24: nothing outside the ViT op set. The Shape/Slice/Cast nodes come from timm's attention reshape; fold them with onnxsim (fixed 224 input) before the DFC parses the graph.
  - The kNN head (`ReduceMin` over a 4096-entry constant bank, `Sqrt`, a large `MatMul` against constants) should stay on the **CPU**: 196×4096×192 MACs is about 154 M, or 5–10 ms on the Pi.
  - The predictor-error head needs `Gather` with fixed indices and 9 encoder/predictor passes. It is CPU-only, or HEF × 5 with host-side gathers. Not worth it unless it clearly wins the gate.
- **INT8 risk.** The zoo's vit_tiny loses 0.9 top-1 points. The anomaly score relies on small embedding distances, so the **memory bank must be rebuilt from the quantised (HEF) encoder's own outputs** on training plates. A float bank against a quantised encoder would shift every z. LayerNorm/Softmax-heavy blocks may need 16-bit activations (DFC `quantization_param ... precision_mode=a16_w16`) at some FPS cost.
- **Calibration** uses TRAINING crops only (as `calib_set.py` does for YOLO), at both scales and from accepted burst-arm grids.
- **Throughput budget.** At about 7 ms per crop (zoo FPS), a plate at scale 2 (9 crops) takes about 65 ms on Hailo. Scale 1 adds 36 crops, about 260 ms. YOLO on the same NPU needs 9 tiles of 640, so the HEFs would be time-shared (HailoRT scheduler).

## 7. Integration proposal (NOT applied; exact diffs)

The anomaly head runs **after the alert** in the same advisor-thread slot, never delaying the verdict. Its regions that no YOLO box covers go into the record as `anomaly.unexplained`. The verdict is **not** changed in phase 1: an unexplained region makes the advisor say "a person should look at x, y". A rules change (anomaly → HOLD_FOR_QA) is a phase-2 decision after the gate.

**Record field** (`ayas.inspection/3.0`; the v3 schema has no `additionalProperties: false`, so the field is additive; `report.build_steel_record(..., extra=...)` already merges extra keys):

```jsonc
"anomaly": {"schema": "ayas.anomaly/1", "model": "jepa_knn vit_tiny s2 (coreset 4096)", "threshold": 3.0,
            "score_max": 4.71, "score_p99": 2.10, "area_mm2_above": 18.4, "n_regions": 2,
            "regions": [{"x_mm": 41.2, "y_mm": 63.0, "w_mm": 4.1, "h_mm": 3.3, "area_mm2": 9.6, "peak": 4.71,
                         "box_px": [...]}],
            "unexplained": [ /* the regions no detector box covers (>= 30 % of the region) */ ],
            "ms": 412.0}
```

`llm.check_reply` already allows any number that is in the record, so the advisor can quote these numbers without changing the guardrail.

```diff
--- a/src/ayas/app/pipeline.py
+++ b/src/ayas/app/pipeline.py
@@ class InspectionPipeline:
     def __init__(self, capture_cfg: Mapping[str, Any], grid_cfg: Mapping[str, Any], detector, spec: Mapping[str, Any],
                  part: Mapping[str, Any] | None = None, advisor=None, *, log: AppLog | None = None,
                  alerter: Callable[[Mapping[str, Any]], None] | None = None, advise: str = "thread",
-                 vision_name: str | None = None, image_px: int = 480, measure: Mapping[str, Any] | None = None):
+                 vision_name: str | None = None, image_px: int = 480, measure: Mapping[str, Any] | None = None,
+                 anomaly=None):
         self.capture_cfg, self.grid_cfg = dict(capture_cfg), dict(grid_cfg or {})
         self.detector, self.spec, self.part = detector, dict(spec), dict(part or {})
+        self.anomaly = anomaly            # ayas.anomaly.OnnxPatchScorer or None (docs/JEPA.md section 7)
@@ def inspect(self, item, meta: Mapping[str, Any] | None = None) -> Inspection:
         judged: list[dict[str, Any]] = []
         verdict = "RETAKE"
         view, view_corners = image, corners
+        grid_for_anomaly, gm_for_anomaly = None, None
         if capture["gate"] != "RETAKE":
@@
             verdict, judged = rules.judge_steel(defects, self.spec, part)
             lap("rules")
             view, view_corners = grid_u8, None
+            grid_for_anomaly, gm_for_anomaly = grid_u8, gm
@@
         self.alerter(record)                                     # the alert: before the log and the model
+        if self.anomaly is not None and grid_for_anomaly is not None:      # after the alert, never before it
+            if self.advise == "sync":
+                self._anomaly(record, grid_for_anomaly, gm_for_anomaly)
+            else:                                                          # the single advisor thread, queued
+                self.submit(self._anomaly, record, grid_for_anomaly, gm_for_anomaly)   # before the explanation
         with self.lock:
@@
+    def _anomaly(self, record: dict[str, Any], grid_u8: np.ndarray, gm) -> None:
+        from ayas import anomaly as AN
+        t0 = time.perf_counter()
+        mask = gm.grid.plate_mask()
+        ys, xs = np.nonzero(mask)
+        rect = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
+        heat = self.anomaly.score_grid(grid_u8, rect, mask)
+        mmpp = float(self.grid_cfg.get("mm_per_px", 0.08))
+        b = float(self.grid_cfg.get("border_mm", 5.0)) / mmpp
+        block = AN.summarize(heat, mmpp, self.anomaly.threshold, (b, b), mask, model=self.anomaly.meta.get("model", "jepa_knn"))
+        block["unexplained"] = AN.unexplained(block, [d["box_px"] for d in record.get("defects", [])])
+        block["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
+        with self.lock:
+            record["anomaly"] = block
+        self._notify("anomaly", self.last)
```

(The advisor job is queued after `_anomaly` on the same single advisor thread, so the explanation already sees `record["anomaly"]`. The `AppLog` row update that stores the advice must also store `record["anomaly"]`: the JSONL line written at alert time does not have it yet, and that is correct, because the alert never waits for the anomaly map.)

```diff
--- a/src/ayas/app/cli.py
+++ b/src/ayas/app/cli.py
@@ def build(args, advise: str) -> tuple[InspectionPipeline, object, dict]:
+    anomaly = None
+    if getattr(args, "anomaly", None):
+        from ayas.anomaly import OnnxPatchScorer
+        anomaly = OnnxPatchScorer(repo_path(args.anomaly.split(":", 1)[-1]),
+                                  threads=int((cfg.get("anomaly") or {}).get("threads", 4)))
     pipe = InspectionPipeline(cfg["_capture"], cfg["_grid"], detector, spec, cfg["part"], advisor, log=log,
                               alerter=alerter, advise=advise, vision_name=vision,
-                              image_px=int((cfg.get("ui") or {}).get("image_px", 480)))
+                              image_px=int((cfg.get("ui") or {}).get("image_px", 480)), anomaly=anomaly)
@@
     r.add_argument("--conf", type=float, default=None, help="detector score threshold (default app.yaml detector.conf)")
+    r.add_argument("--anomaly", default=None, help="onnx:<jepa_knn.onnx> patch anomaly map (after the alert; docs/JEPA.md)")
--- a/configs/app.yaml
+++ b/configs/app.yaml
@@
 detector:
   conf: 0.25                              # detector score threshold (the rules mark < 0.5 as uncertain)
+
+anomaly:                                  # docs/JEPA.md: only after the adoption gate passes
+  threads: 4                              # onnxruntime intra-op threads (Pi 5: 4 A76 cores)
```

```diff
--- a/src/ayas/llm.py
+++ b/src/ayas/llm.py
@@ def explain_steel(report: dict[str, Any], spec: dict[str, Any] | None = None) -> str:
     lines += [_steel_line(d) for d in defects]
+    an = (report.get("anomaly") or {}).get("unexplained") or []
+    if an:
+        r = an[0]
+        lines.append(f"The anomaly map also marks {len(an)} area(s) no defect class explains; the largest is at "
+                     f"x {r['x_mm']} mm, y {r['y_mm']} mm ({r['area_mm2']} mm2). A person should look at it. "
+                     "The anomaly score is not a defect type or a severity.")
     lines.append("What to do:")
```

(`system_prompt` gets one sentence for the Llama backend: "anomaly.unexplained lists surface areas that no defect class explains; mention them as 'look here', never as a defect type, severity or reason to pass".)

```diff
--- a/src/ayas/app/ui.py
+++ b/src/ayas/app/ui.py
@@ def summarise(rec: Mapping[str, Any], key: Any = None) -> dict[str, Any]:
         "timing_ms": rec.get("timing_ms") or {}, "source": rec.get("source") or {},
         "advice": rec.get("advice"), "backend": rec.get("backend") or {},
+        "anomaly": {k: (rec.get("anomaly") or {}).get(k) for k in ("score_max", "n_regions", "unexplained", "ms")},
     }
@@
   $("gatemsg").textContent = (i.gate !== "PASS" && i.gate_message) ? i.gate_message : "";
+  const an = i.anomaly || {};
+  $("anom").textContent = (an.unexplained && an.unexplained.length)
+      ? "anomaly map: " + an.unexplained.length + " unexplained area(s), look at x " + an.unexplained[0].x_mm + " y " + an.unexplained[0].y_mm + " mm"
+      : (an.score_max == null ? "" : "anomaly map: nothing outside the detected defects");
@@
     <div id="gatemsg"></div>
+    <div id="anom" style="font-size:13px;color:#c9a0ff"></div>
```

The overlay JPEG gets dashed purple boxes for `unexplained` regions, drawn in `overlay_jpeg` after the record update, or on the next page refresh.

## 8. Honest limits

- **Nothing here is a result yet.** The smoke model saw 2 training plates at 0.5× render resolution. It is scored against full-resolution pilot plates, a resolution domain shift on top of a tiny bank, so its z threshold is mis-calibrated (hundreds of false-alarm cells per plate at z > 3 in the smoke). The smoke tables prove only that every stage runs and produces the right shapes and files.
- **Where cross-lighting can matter at all.** In the **hood** (the product configuration), the burst grids under C1–C6 already differ from C0h by about 1.5 DN, essentially sensor noise. A raw-pixel feature is already almost lighting-invariant there; that is the capture method's success. The JEPA invariance can only add something on **open-mode** captures that pass the gate (residual glare edges, shadow-edge leakage, the pods' L/R bias) and on the naive arm. Section 5, gate 4, is set on open rows for that reason.
- **Twin-only.** Twin surfaces are procedural textures. Real mill scale, brushed grain and real anomalies (inclusions, laminations, weld spatter) are not in the twin, so "detects anomalies outside the 6 classes" can only be shown here on unlabelled twin marks: distractors, unlabelled defects, held-out kinds. That says nothing about real parts. Phase 2 needs real captures of normal plates for the bank.
- **No out-of-class twin defects yet.** A clean test of (a) needs test plates carrying a defect kind that YOLO was never trained on, e.g. a held-out `kind` or a new texture such as weld spatter. Post's labels make "YOLO misses" measurable, but "outside the 6 classes" is only approximated by YOLO's misses and the distractors.
- **The kNN bank grows with material diversity.** A 4096-patch coreset from 48 plates may under-cover rare finishes (galvanised spangle), which would give false alarms on a new spangle pattern. More plates or a per-material bank fix that; the material is known from the part config.
- **Gate 1 is not computable yet.** It needs the trained YOLO burst model. `eval_anomaly.py` already gives each defect's region score; the missing join is about 50 lines: run `ayas.npu.detect_grid` on each test grid, apply the `eval_plates` matching, then count the missed defects whose region score exceeds z\*. It will be added when `models/v8n_burst.onnx` exists. The linear probe (c) is also not built yet.
- **Depth is still not measured.** An anomaly score is not a severity. The record keeps them separate, and the advisor text says so.
- **Determinism** is per (seed, index) on the data side. GPU kernels (flash-attention backward on the L4, MPS) are not bit-exact.
- **Phase-1 smoke artifacts.** The first JarvisLabs smoke run (r_25fe5b78) used the runner version before the standing artifact rule. Its `last.pt` (the optimiser-state resume checkpoint) and the raw `jl-runs` folder were not kept. The full `output.log`, `encoder.pt` (the full model), metrics, config, summary and a rebuilt code bundle are kept and verified in `artifacts/jarvislabs/jepa-smoke_r_25fe5b78/`.

## 9. What has been run (15 Nov): SMOKE, NOT A RESULT

| Step | What ran | Outcome |
|---|---|---|
| Training render | `pairs.py --n 2 --start-seed 100000 --render --res-scale 0.5 --samples 32` in the scratch dir | 2 training plates (s100000 ss_2b with 2 stains, a pit and a crack; s100001 ss_brushed with a dent and a scratch) × 14 rows × 2 variants = 56 burst grids. 9 RETAKE, all open-mode stainless; 47 accepted. 817 s including post; 29 MB of JPEG grids |
| Local training | `train.py` smoke.yaml, MPS, 300 steps × 32 | loss 0.413 → 0.086; `tgt_spread` 0.03–0.04 (no collapse); 26 img/s with the Mac shared with Blender |
| **JarvisLabs smoke** | `jl_train.sh fresh`, **run r_25fe5b78**, machine 574712 (`jepa-smoke`, L4 on demand, IN2) | 2000 steps × 64 in 137.5 s, **932 img/s** (bf16, torch 2.11.0+cu130). Loss 0.421 → 0.059; hidden 0.058, visible-cross 0.065. **Instance up for 226 s → ≈ ₹2.6** (balance 1,038.83 → 1,036.24). **Destroyed: yes**; afterwards `jl status`: running_instances 0 (the only instance listed is `ayas-render`, paused, not created here). Artifacts: `artifacts/jarvislabs/jepa-smoke_r_25fe5b78/`, verified (`MANIFEST.md`: encoder.pt loads and runs) |
| Anomaly + invariance eval | `eval_anomaly.py --smoke` with the JarvisLabs checkpoint, bank from the 47 training grids, test = the pilot (4 plates × 14 rows × 2 variants × 2 arms = 224 grids, a frozen copy, because another job re-posts `data/twin_pilot`) | `reports/jepa_anomaly.md` and `.json`, `reports/figures/jepa/*.jpg`; one heat map per grid in the scratch dir. Highlights below |
| ONNX + latency | `export_onnx.py` | `reports/jepa_onnx.md`: encoder, knn, knn_head, pred and INT8 graphs; ONNX vs torch max error ≤ 3e-5 (fp32); latency in section 6 |
| Inference side without torch | `ayas.anomaly.OnnxPatchScorer` in `.venv` on a pilot grid | 9 windows at scale 2 in 1.6 s (fp32, 4 threads, Mac under load). The smoke bank's z is wildly off (z ≈ 111 everywhere), the expected result of a bank built at 0.5× resolution and scored at full resolution |

**Smoke eval highlights (SMOKE, NOT A RESULT: 2 low-resolution training plates for the JEPA and the bank, 4 test plates, defect counts are captures, not physical defects).**
- **The pipeline runs end to end.** 185 test grids were scored (39 burst-arm RETAKE skipped), 5 methods × 2 scales, with one heat map per grid, every table and 4 example figures. On the Mac, under heavy shared load, it took 39 min.
- **The smoke JEPA is the weakest method, as expected.**

  | Method | Region AUROC, burst hood | Region AUROC, burst open |
  |---|---|---|
  | jepa_knn | 0.82 | 0.68 |
  | jepa_pred | 0.75 | 0.66 |
  | ImageNet ViT-Tiny | 0.87 | 0.90 |
  | raw pixels | 0.92 | 0.92 |
  | ResNet18 PatchCore | **0.97** | **0.96** |

  Fine-tuning the ImageNet ViT on 2 half-resolution plates made it *worse* than its own ImageNet starting point. This is the clearest sign that the full run must beat ResNet18-PatchCore (gate 3), not just a pixel baseline. The z thresholds are mis-calibrated by the resolution shift: hundreds of false-alarm cells per plate at z > 3 for jepa_knn and ResNet18, and 0 for the ImageNet ViT.
- **Invariance.** Mean retrieval@1 vs C0, accepted grids:

  | Arm, device | raw pixels | ResNet18 | smoke JEPA | ImageNet ViT |
  |---|---|---|---|---|
  | burst, hood | 0.997 | 0.992 | 0.931 | 0.745 |
  | burst, open | 0.842 | 0.894 | 0.707 | 0.612 |
  | naive, open | 0.51 | 0.73 | 0.31 | 0.36 |

  This is consistent with section 8:
  - the hood plus burst already makes raw pixels near-perfectly lighting-invariant;
  - the open arm is where invariance can still be gained;
  - the naive arm shows how much the old capture method loses (pixels 0.51).

  The smoke JEPA's centred-cosine gap is only 0.03 (others 0.28–0.68): its embeddings are nearly collapsed in direction after 2000 steps on 2 plates. The full run must watch `tgt_spread` and this gap. A variance/covariance (VICReg-style) regulariser on the target embeddings is the planned fallback if the gap stays small.
- **A side observation.** In the ImageNet ViT-Tiny heat maps there are periodic low-score dots, the known high-norm "register" tokens of supervised ViTs. They are harmless here, but they are one more reason the JEPA (no class token, patch-only training) may beat it once trained on real data.



## 10. Full run (15 Nov): RESULT, and the adoption-gate verdict

**Run.** `jl_train.sh on`, run `r_dd9195e8` on the render instance 574738 (L4, IN2):
- `full.yaml`: 30,000 steps × 128 in 4,773 s, about 805 img/s, bf16, torch 2.11.0+cu130.
- 1,344 paired grids of 48 training plates; 836 grids were accepted by the gate.
- Final loss 0.113 (hidden 0.113, visible-cross 0.098). `tgt_spread` rose 0.04 → 0.068: no collapse.
- Then `eval_anomaly.py` on the test matrix: 15 plates from seed 900100, 698 grids scored, 142 burst RETAKEs skipped. Then ONNX export.
- Artifacts: `artifacts/jarvislabs/ayas-render_r_dd9195e8/` (VERIFY OK; encoder.pt and last.pt load and run).
- Report: `reports/jepa_anomaly.md` and `.json`. The smoke report is kept as `reports/jepa_anomaly_smoke.md`.

Region AUROC on the burst arm (the product arm):

| method | hood | open |
|---|---|---|
| jepa_pred (predictor error) | **0.975** | 0.957 |
| jepa_knn (EMA target encoder + memory bank) | 0.508 | 0.563 |
| resnet18_knn (PatchCore) | 0.965 | 0.957 |
| vit_in1k_knn (the same ViT-Tiny, ImageNet only) | 0.919 | 0.923 |
| pixels_knn | 0.919 | 0.858 |

On the other measures:
- Region AP: jepa_pred 0.375 / 0.429 against ResNet18 0.687 / 0.586.
- Recall at 0.5 false-alarm cells per plate: 0.24 / 0.28 against 0.58 / 0.37.

**Gate verdict: JEPA is NOT adopted.** It stays a report-only experiment.

| gate | result |
|---|---|
| 3. Beats the baselines by ≥ 0.02 region AUROC, hood and open | **FAIL**. jepa_pred is +0.010 over ResNet18 in the hood and +0.000 open. jepa_knn is at chance (below chance on the naive arm), so its EMA target embeddings are not usable for a memory bank. |
| 4. More lighting-invariant than the baselines (open burst retrieval@1, ≥ best + 0.05) | **FAIL**. JEPA 0.853 against ResNet18 0.967 and raw pixels 0.905. Nothing is claimed about JEPA invariance: **the capture method is the lighting story**, as before. In the hood every extractor is flat across C1–C6. |
| 2. False alarms at z\* ≤ 0.5 per clean plate (hood) | **FAIL for every method at the uncalibrated z\* = 3.** jepa_pred 21.9, ResNet18 16.3 FA/plate. z\* must be recalibrated on 8 held-out TRAINING plates before any anomaly head can be judged. |
| 5. Edge budget ≤ 150 ms per crop | jepa_pred needs 1 + 4 + 4 passes (350–580 ms projected): FAIL. jepa_knn INT8 fits (7.2 MB) but scores at chance. |
| 1. Catches ≥ 30 % of YOLO misses | Not computed yet; it needs `v8n_burst.onnx`, which is training. |

**Consequence.**
- If an anomaly head ships in Phase 1, it is the **ResNet18 PatchCore baseline**, and only if it passes gates 1 and 2 after z\* recalibration. Otherwise none ships.
- The JEPA work is reported honestly as tried and not adopted. jepa_pred's 0.975 hood AUROC is close to ResNet18, but it is too slow for the Pi and adds nothing measurable.
