"""I-JEPA for Ayas (docs/JEPA.md).

A self-supervised ViT encoder trained on the twin's rectified plate grids (0.08 mm/px), with a
cross-lighting objective (context from the plate under condition X, targets from the same
plate region under C0 dark or another condition). It feeds two things the YOLO detector does
not give: a patch anomaly map (PatchCore memory bank or predictor error) and lighting-invariant
patch features.

Modules (torch-free unless noted):
- grids.py       grid index from sidecars, canonical plate orientation, masks (numpy + cv2)
- pairs.py       paired TRAINING plans (seeds 100000..899999) + render through the twin pipeline
- pack.py        compact dataset (canonical grids as JPEG q95 + index.json) for upload
- metrics.py     AUROC/AP, region scores, invariance statistics (numpy)
- masks.py       I-JEPA multi-block masks (numpy; torch collate in data.py)
- data.py        torch datasets (self and cross-lighting pairs)                    [torch]
- model.py       timm ViT encoder (masked forward), narrow predictor, EMA, JEPA     [torch]
- train.py       training CLI (deterministic seeds, AMP on CUDA, checkpoints)       [torch]
- eval_anomaly.py memory bank, heat maps, AUROC/AP, invariance, reports/jepa_anomaly.md [torch]
- export_onnx.py encoder + anomaly heads -> ONNX, latency (Mac CPU 1/4 threads)    [torch]
- jl_train.sh    JarvisLabs runner: fresh instance (smoke) or --on the render box; collect + verify, then destroy/pause
- artifacts.py   verify a downloaded artifact folder (checkpoints load and run) and write its MANIFEST.md
"""
