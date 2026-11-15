#!/bin/sh
# Full TriNetra twin render on the JarvisLabs L4 instance.
# Instance copy: /home/trinetra_jobs/render_all.sh   (repo copy: tools/jl/render_all.sh)
#
# Run it tracked, from the Mac (see tools/jl/README.md):
#   jl run --on <id> --json --yes -- sh -lc 'sh /home/trinetra_jobs/render_all.sh --jobs matrix,train'
#
# Resumable: re-run the same command after a stop, a crash or a pause/resume. Rendered scenes
# (good render_meta.json + render.exr in <job>/work/<scene>/) and posted scenes (<post dir>/sidecar/
# <scene>.json) are skipped. EXRs are always kept (keep-work), texture maps are deleted.
#
# Jobs and their dirs under --data (default /home/data):
#   matrix  twin_matrix/  `trinetra-twin matrix --plans-only`: test plates (seeds from --matrix-start-seed,
#                         900100: the gate was tuned on 900000-900068, so those are not test plates) x 14 rows
#                         (7 conditions x open/hood); the first --matrix-showcase plates also get the
#                         overview render and --save-bursts post (Pi demo bursts).
#   train   twin_train/   `trinetra-twin plan`: random training/val scenes, seeds from --train-start-seed.
#   jepa    jepa_train/   `python training/jepa/pairs.py` (no --render): paired TRAINING plates
#                         (seeds 100000..899999) x 14 rows, split jepa_train; post uses pairs.jepa_config
#                         (burst grids as JPEG q95, no YOLO tiles).
#
# Options:
#   --jobs LIST            comma list of matrix,train,jepa         (default matrix,train)
#   --stage S              render | post | both                     (default render; post later with final code)
#   --data DIR             /home/data
#   --matrix-n N           test plates                              (15)
#   --matrix-showcase N    plates with overview + save-bursts       (2)
#   --matrix-start-seed S  first test plate seed                    (900100; must be >= 900100)
#   --train-n N            training scenes                          (800)
#   --train-start-seed S   (0)
#   --jepa-n K             JEPA training plates; required for jobs=jepa (docs/JEPA.md gives K)
#   --jepa-start-seed S    (100000)
#   --jepa-arms "A [B]"    grid arms for JEPA post                  ("burst"; "burst naive" adds the old arm)
#   --procs N              Blender processes in parallel on the one L4 (3: measured best; 2 = -10 %, 4 = no gain)
#   --post-procs N         post processes for --stage post          (24 of the 28 vCPUs)
#   --post-tag TAG         post into <job>/post_<TAG>/ instead of <job>/ (re-post with the final code)
#   --force-post           re-post scenes that already have a sidecar in the post dir
#   --device D             OPTIX | CUDA                             (OPTIX)
#   --samples N            override the plans' samples (64)
#   --batch N              scenes per Blender process               (12 = render.batch_size)
#   --min-free-gb G        stop rendering below this free disk      (10)
set -u

JOBS=matrix,train; STAGE=render; DATA=/home/data
MATRIX_N=15; MATRIX_SHOW=2; MATRIX_SEED=900100; TRAIN_N=800; TRAIN_SEED=0
JEPA_N=""; JEPA_SEED=100000; JEPA_ARMS="burst"
PROCS=3; POST_PROCS=24; POST_TAG=""; FORCE_POST=""; DEVICE=OPTIX; SAMPLES=""; BATCH=""; MIN_FREE=10
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS=$2; shift 2;;
    --stage) STAGE=$2; shift 2;;
    --data) DATA=$2; shift 2;;
    --matrix-n) MATRIX_N=$2; shift 2;;
    --matrix-showcase) MATRIX_SHOW=$2; shift 2;;
    --matrix-start-seed) MATRIX_SEED=$2; shift 2;;
    --train-n) TRAIN_N=$2; shift 2;;
    --train-start-seed) TRAIN_SEED=$2; shift 2;;
    --jepa-n) JEPA_N=$2; shift 2;;
    --jepa-start-seed) JEPA_SEED=$2; shift 2;;
    --jepa-arms) JEPA_ARMS=$2; shift 2;;
    --procs) PROCS=$2; shift 2;;
    --post-procs) POST_PROCS=$2; shift 2;;
    --post-tag) POST_TAG=$2; shift 2;;
    --force-post) FORCE_POST=--force; shift;;
    --device) DEVICE=$2; shift 2;;
    --samples) SAMPLES=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --min-free-gb) MIN_FREE=$2; shift 2;;
    -h|--help) sed -n '2,48p' "$0"; exit 0;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
case "$STAGE" in render|post|both) ;; *) echo "--stage must be render, post or both"; exit 2;; esac
[ "$MATRIX_SEED" -ge 900100 ] || { echo "--matrix-start-seed must be >= 900100 (900000-900068 tuned the gate)"; exit 2; }

. /home/trinetra_jobs/env.sh
mkdir -p "$DATA/logs"
LOG="$DATA/logs/render_all.log"
RC_FILE="$DATA/logs/.render_all.rc"

main() {
  set -e
  echo "=== render_all $(date -u +%FT%TZ) jobs=$JOBS stage=$STAGE procs=$PROCS device=$DEVICE data=$DATA"
  sh /home/setup_sys.sh
  [ -x "$TRINETRA_BLENDER" ] || { echo "Blender missing: run /home/trinetra_jobs/bootstrap.sh"; exit 3; }
  "$TRINETRA_BLENDER" -b --factory-startup --version 2>/dev/null | head -1
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  cd "$TRINETRA_REPO"
  PY="$TRINETRA_VENV/bin/python"
  TJ="$PY tools/jl/twin_jobs.py"
  RARGS="--device $DEVICE --procs $PROCS --min-free-gb $MIN_FREE"
  [ -n "$SAMPLES" ] && RARGS="$RARGS --samples $SAMPLES"
  [ -n "$BATCH" ] && RARGS="$RARGS --batch $BATCH"
  [ "$STAGE" = both ] && RARGS="$RARGS --post"
  rc=0
  for job in $(echo "$JOBS" | tr ',' ' '); do
    case "$job" in
      matrix)
        D="$DATA/twin_matrix"; MAN="$D/matrix_manifest.json"; EXTRA=""
        [ -f "$MAN" ] || "$TRINETRA_VENV/bin/trinetra-twin" matrix --n "$MATRIX_N" --start-seed "$MATRIX_SEED" --out "$D" --plans-only
        ;;
      train)
        D="$DATA/twin_train"; MAN="$D/plans"; EXTRA=""
        have=$(ls "$D/plans" 2>/dev/null | grep -c '\.json$' || true)
        if [ "$have" -lt "$TRAIN_N" ]; then
          [ "$have" -gt 0 ] && echo "WARNING: $D/plans has $have < $TRAIN_N plans; rewriting seeds $TRAIN_SEED.. (same code -> same plans)"
          "$TRINETRA_VENV/bin/trinetra-twin" plan --n "$TRAIN_N" --start-seed "$TRAIN_SEED" --out "$D"
        fi
        ;;
      jepa)
        [ -n "$JEPA_N" ] || { echo "jobs=jepa needs --jepa-n K (see docs/JEPA.md)"; exit 2; }
        D="$DATA/jepa_train"; MAN="$D/pairs_manifest.json"
        # shellcheck disable=SC2086
        EXTRA="--jepa --jepa-arms $JEPA_ARMS"
        # shellcheck disable=SC2086
        [ -f "$MAN" ] || "$PY" training/jepa/pairs.py --n "$JEPA_N" --start-seed "$JEPA_SEED" --out "$D" --arms $JEPA_ARMS
        ;;
      *) echo "unknown job $job"; exit 2;;
    esac
    POUT="$D"; [ -n "$POST_TAG" ] && POUT="$D/post_$POST_TAG"
    echo "--- job $job: dir $D, post dir $POUT"
    if [ "$STAGE" != post ]; then
      if [ "$job" = matrix ] && [ "$MATRIX_SHOW" -gt 0 ]; then       # showcase plates first: overview (+ bursts)
        SB=""; [ "$STAGE" = both ] && SB="--save-bursts"
        # shellcheck disable=SC2086
        $TJ render --plans "$MAN" --first-plates "$MATRIX_SHOW" --overview --out "$D" --post-out "$POUT" $SB $RARGS || rc=$?
      fi
      # shellcheck disable=SC2086
      $TJ render --plans "$MAN" --out "$D" --post-out "$POUT" $EXTRA $RARGS || rc=$?
    fi
    if [ "$STAGE" = post ]; then
      if [ "$job" = matrix ] && [ "$MATRIX_SHOW" -gt 0 ]; then
        # shellcheck disable=SC2086
        $TJ post --plans "$MAN" --first-plates "$MATRIX_SHOW" --save-bursts --out "$D" --post-out "$POUT" --procs "$POST_PROCS" $FORCE_POST || rc=$?
      fi
      # shellcheck disable=SC2086
      $TJ post --plans "$MAN" --out "$D" --post-out "$POUT" --procs "$POST_PROCS" $FORCE_POST $EXTRA || rc=$?
    fi
    $TJ status --plans "$MAN" --out "$D" --post-out "$POUT"
  done
  df -h /home | tail -1
  echo "=== render_all done $(date -u +%FT%TZ) rc=$rc"
  return $rc
}

# one run at a time (the lock goes with the process, so a pause/crash never leaves it stuck)
exec 9>"$DATA/logs/.render_all.lock"
if ! flock -n 9; then echo "another render_all.sh is running"; exit 4; fi
rm -f "$RC_FILE"
( main; echo $? > "$RC_FILE" ) 2>&1 | tee -a "$LOG"     # a set -e abort leaves no rc file -> exit 1
exit "$(cat "$RC_FILE" 2>/dev/null || echo 1)"
