#!/usr/bin/env bash
# JarvisLabs job: Drive training renders -> real-domain crops -> fine-tune the depth network.
# Needs /home/.config/rclone/rclone.conf (uploaded by the user), /home/ayas (code), /home/venv,
# /home/depthnet/{data,run1} (synthetic shards and the synthetic model).
set -euo pipefail
RC="/home/bin/rclone --config /home/.config/rclone/rclone.conf"
SRC="gdrive:Ayas twin data (rclone)/render_2026-11-15/twin_train"
D=/home/data; mkdir -p $D/tars $D/work
. /home/venv/bin/activate; export PYTHONPATH=/home/ayas/src; cd /home/ayas
echo "== $(date -Is) download"
$RC copy "$SRC" $D/tars --include "work_s*.tar" --transfers 4 --checkers 8 --drive-chunk-size 64M --stats 60s --stats-one-line 2>&1 | { grep -v NOTICE || true; }
ls -la $D/tars | tail -3
echo "== $(date -Is) unpack plan.json + render.exr"
for t in $D/tars/work_s*.tar; do
  tar -xf "$t" -C $D/work --wildcards "*/plan.json" "*/render.exr" && rm -f "$t"
done
echo "scenes: $(find $D/work -name render.exr | wc -l)"; df -h /home | tail -1
echo "== $(date -Is) crops"
python training/depthnet/blender_crops.py --work $D/work --out /home/depthnet/blender --variants 3 --jitter 2 --workers 15 --shard 1500
echo "== $(date -Is) fine-tune"
python training/depthnet/train.py --data /home/depthnet/data --blender /home/depthnet/blender --init /home/depthnet/run1/best.pt \
  --out /home/depthnet/run2 --epochs 25 --lr 1e-3 --bs 128
echo "== $(date -Is) done"
