#!/bin/sh
# YOLOv8n capture-arm training (naive = old single frame, burst = new 5-frame method) on the JarvisLabs
# instance, next to the twin data. Instance copy: /home/trinetra_jobs/train_yolo.sh (repo: tools/jl/train_yolo.sh)
#
# Run it tracked, from the Mac, after render_all.sh --stage post --post-tag final has finished:
#   jl run --on <id> --json --yes -- sh -lc 'sh /home/trinetra_jobs/train_yolo.sh'
#
# Steps (each one is skipped when its output is already there, so re-run the same command after a stop):
#   venv   /home/venvs/yolo: uv Python 3.11, CUDA torch (cu130 wheel, as training/jepa/jl_train.sh; the L4
#          driver runs it), ultralytics/onnx/onnxruntime/onnxslim/opencv at the requirements-train.txt pins,
#          the rest at tools/jl/constraints-mac.txt, the project --no-deps. /home/venvs/trinetra is not touched.
#   naive  post the training renders a second time with post.train_arm: naive -> <train>/post_naive (CPU)
#   train  training/train_arms.py: per-arm datasets, train.py, export_onnx.py, then eval_lighting.py on the
#          test matrix (<matrix>/post_<tag> + a copy of matrix_manifest.json) -> /home/data/yolo_runs/
#
# Options:
#   --post-tag T     the post dirs of the final code               (final)
#   --epochs N       (150 = train_arms default)   --batch N (16)   --workers N (8)
#   --skip-eval      train and export only
#   --venv-only      build/check the venv and stop (can run while post is still going)
#   --quick          smoke: 1 epoch, batch 4, into /home/data/yolo_runs_quick
#   --torch SPEC     ("torch==2.11.0 torchvision")   --torch-index URL (https://download.pytorch.org/whl/cu130)
set -eu

POST_TAG=final; EPOCHS=150; BATCH=16; WORKERS=8; EVAL=1; QUICK=""; VENV_ONLY=0
TORCH_SPEC="torch==2.11.0 torchvision"; TORCH_INDEX="https://download.pytorch.org/whl/cu130"
while [ $# -gt 0 ]; do
  case "$1" in
    --post-tag) POST_TAG=$2; shift 2;;
    --epochs) EPOCHS=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --workers) WORKERS=$2; shift 2;;
    --skip-eval) EVAL=0; shift;;
    --venv-only) VENV_ONLY=1; shift;;
    --quick) QUICK=--quick; shift;;
    --torch) TORCH_SPEC=$2; shift 2;;
    --torch-index) TORCH_INDEX=$2; shift 2;;
    -h|--help) sed -n '2,23p' "$0"; exit 0;;
    *) echo "unknown option $1"; exit 2;;
  esac
done

. /home/trinetra_jobs/env.sh
sh /home/setup_sys.sh            # libgl1 for opencv; apt installs do not survive a pause
cd "$TRINETRA_REPO"
D=$TRINETRA_DATA
TRAIN=$D/twin_train; MATRIX=$D/twin_matrix
BURST_POST=$TRAIN/post_$POST_TAG; NAIVE_POST=$TRAIN/post_naive; MATRIX_POST=$MATRIX/post_$POST_TAG
OUT=$D/yolo_runs; [ -n "$QUICK" ] && OUT=$D/yolo_runs_quick
V=/home/venvs/yolo; PY=$V/bin/python
echo "=== train_yolo $(date -u +%FT%TZ) post_tag=$POST_TAG epochs=$EPOCHS batch=$BATCH out=$OUT $QUICK"

# --- venv -------------------------------------------------------------------------------------------
if ! "$PY" -c 'import torch, ultralytics, onnxruntime, trinetra' 2>/dev/null; then
  echo "--- venv $V"
  [ -x /home/.uvtool/bin/uv ] || { python3 -m venv /home/.uvtool && /home/.uvtool/bin/pip install -q uv; }
  [ -x "$PY" ] || { /home/.uvtool/bin/uv python install 3.11 && /home/.uvtool/bin/uv venv --python 3.11 --seed "$V"; }
  "$PY" -c 'import torch, torchvision' 2>/dev/null || "$PY" -m pip install -q $TORCH_SPEC --index-url "$TORCH_INDEX"
  "$PY" -m pip install -q -c tools/jl/constraints-mac.txt ultralytics==8.4.202 onnx==1.23.2 onnxruntime==1.31.1 \
      onnxslim==0.1.99 opencv-python==5.0.0.93 OpenEXR PyYAML numpy scikit-learn joblib 2>&1 | tail -3
  "$PY" -m pip install -q --no-deps -e . 2>&1 | tail -2
fi
"$PY" -c 'import sys, torch, ultralytics; print("python", sys.version.split()[0], "torch", torch.__version__, "ultralytics", ultralytics.__version__, "cuda", torch.cuda.is_available() and torch.cuda.get_device_name(0))'
[ "$VENV_ONLY" = 1 ] && { echo "=== venv ready"; exit 0; }

# --- inputs ------------------------------------------------------------------------------------------
[ -d "$BURST_POST/yolo" ] || { echo "missing $BURST_POST/yolo: run render_all.sh --stage post --post-tag $POST_TAG first"; exit 3; }
if [ ! -d "$NAIVE_POST/yolo" ] || [ "$(ls "$NAIVE_POST/sidecar" 2>/dev/null | wc -l)" -lt "$(ls "$BURST_POST/sidecar" | wc -l)" ]; then
  echo "--- naive post -> $NAIVE_POST"
  mkdir -p runs
  "$TRINETRA_VENV/bin/python" training/train_arms.py --write-naive-config runs/twin_naive.yaml
  "$TRINETRA_VENV/bin/python" tools/jl/twin_jobs.py post --plans "$TRAIN/plans" --out "$TRAIN" --post-out "$NAIVE_POST" \
      --config runs/twin_naive.yaml --procs 24
fi
for t in "$BURST_POST" "$NAIVE_POST"; do
  echo "tiles $(basename "$t"): train $(ls "$t/yolo/images/train" 2>/dev/null | wc -l), val $(ls "$t/yolo/images/val" 2>/dev/null | wc -l)"
done
MATRIX_ARG=""
if [ "$EVAL" = 1 ]; then
  [ -d "$MATRIX_POST/grids" ] || { echo "missing $MATRIX_POST/grids"; exit 3; }
  cp -f "$MATRIX/matrix_manifest.json" "$MATRIX_POST/matrix_manifest.json"
  MATRIX_ARG="--matrix $MATRIX_POST"
fi

# --- train, export, evaluate ---------------------------------------------------------------------------
mkdir -p "$OUT"
# shellcheck disable=SC2086
"$PY" training/train_arms.py --burst "$BURST_POST/yolo" --naive "$NAIVE_POST/yolo" $MATRIX_ARG \
    --datasets "$OUT/datasets" --runs "$OUT/train" --models "$OUT/models" --reports "$OUT/reports" \
    --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS" --device 0 $QUICK
ls -la "$OUT/models" "$OUT/reports" 2>/dev/null || true
echo "=== train_yolo done $(date -u +%FT%TZ)"
