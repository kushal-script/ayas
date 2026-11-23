#!/usr/bin/env bash
# JarvisLabs runner for the cross-lighting I-JEPA (docs/JEPA.md, section 5).
#
# Two modes:
#   fresh  create ONE L4 container, upload the code + a packed dataset, train, collect + verify the
#          artifacts, then destroy it (default; --after pause keeps the disk). Smoke / small jobs.
#          Python: the template's CUDA torch in a --system-site-packages venv (/home/jepa_venv).
#     training/jepa/jl_train.sh fresh --data <local grid dir> [<dir> ...] [--spot] \
#         [--config training/jepa/configs/smoke.yaml] [--name jepa-smoke] [--max-min 15] [-- <train.py args>]
#   on     run on an EXISTING, RUNNING instance whose disk already holds the renders (the shared
#          "ayas-render" box: resume it yourself first; this script never resumes, and never
#          destroys an instance it did not create). No dataset upload. The code goes up with
#          tools/jl/sync_code.sh (-> /home/ayas). Python: /home/venvs/jepa, a Python 3.11 venv
#          (uv, kept under /home like /home/venvs/ayas) with a CUDA torch wheel (--torch,
#          --torch-index; cu130 works with the L4 driver 595.58.03), timm, onnx, onnxruntime and the
#          project installed --no-deps; /home/venvs/ayas is never touched.
#     training/jepa/jl_train.sh on --machine <id> --remote-data /home/data/jepa_train/post_final [<dir> ...] \
#         [--config training/jepa/configs/full.yaml] [--name jepa-full] [--max-min 120] \
#         [--eval-test /home/data/twin_matrix/post_final] [--export] [--after none|pause] [-- <train.py args>]
#   --remote-data / --eval-test are post output dirs (the ones holding grids/<arm>/<split>/).
#   --eval-test runs eval_anomaly.py on the instance after training (the report, figures and json come
#   back; the test grids and heat maps stay remote); --export writes the ONNX files there (no timing).
#
# Flow: jl run (detached, --json --yes) -> poll `jl run logs <run> --tail 25` every --poll s ->
# COLLECT to artifacts/jarvislabs/<instance-name>_<run_id>/: the whole run output dir (encoder.pt,
# last.pt, metrics.jsonl, summary.json, config.yaml, eval/, onnx/), <home>/jl-runs/<run_id>/, the
# full `jl run logs`, `jl run status --json`, the remote env (nvidia-smi, torch/timm, pip freeze),
# the data manifests (pairs_manifest.json / matrix_manifest.json: they regenerate the data; bulk
# data is never downloaded), the exact JEPA code bundle + sha256, every jl command (commands.txt)
# -> VERIFY with training/jepa/artifacts.py (sizes, every checkpoint loads and runs, MANIFEST.md)
# -> only then the lifecycle action. A failed verification turns destroy into pause.
# `jl status --json` / `jl get --json` are read for resources, balance, status and name only (no
# account details are printed or stored). Only twin-rendered data and project code are uploaded.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv-train/bin/python"
MODE="${1:-}"; shift || true
GPU="L4"; SPOT=""; CONFIG=""; NAME=""; MAXMIN=""; POLL=60; AFTER=""; MACHINE=""; SYNC=1
TORCH_SPEC="torch==2.11.0 torchvision"; TORCH_INDEX="https://download.pytorch.org/whl/cu130"   # the smoke run's template torch
DATA=(); RDATA=(); EVAL=(); EXTRA=(); MAXWAIT_MIN=60; STORAGE=20; EXPORT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) shift; while [[ $# -gt 0 && "$1" != --* ]]; do DATA+=("$1"); shift; done ;;
    --remote-data) shift; while [[ $# -gt 0 && "$1" != --* ]]; do RDATA+=("$1"); shift; done ;;
    --eval-test) shift; while [[ $# -gt 0 && "$1" != --* ]]; do EVAL+=("$1"); shift; done ;;
    --export) EXPORT=1; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    --spot) SPOT="--spot"; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --max-min) MAXMIN="$2"; shift 2 ;;
    --max-wait-min) MAXWAIT_MIN="$2"; shift 2 ;;
    --poll) POLL="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --machine) MACHINE="$2"; shift 2 ;;
    --storage) STORAGE="$2"; shift 2 ;;
    --no-sync) SYNC=0; shift ;;
    --torch) TORCH_SPEC="$2"; shift 2 ;;
    --torch-index) TORCH_INDEX="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ "$MODE" == "fresh" || "$MODE" == "on" ]] || { sed -n '2,34p' "$0"; exit 2; }
case "$GPU" in L4) ;; *) echo "refusing GPU $GPU: this runner is capped at L4" >&2; exit 2 ;; esac

STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="${NAME:-jepa-$MODE}"
[[ "$MODE" == "fresh" && "$NAME" == "ayas-render" ]] && { echo "refusing: a fresh instance may not take the render box's name" >&2; exit 2; }
WORK="$(mktemp -d "${TMPDIR:-/tmp}/jepa_jl.XXXXXX")"
ART_BASE="${JEPA_ARTIFACTS:-$ROOT/artifacts/jarvislabs}"   # override only for offline tests
ART="$ART_BASE/${NAME}_pending_${STAMP}"                   # renamed to <instance-name>_<run_id> once known
mkdir -p "$ART"
LOG="$ART/jl_runner.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
jlx() { echo "jl $*" >> "$ART/commands.txt"; jl "$@"; }      # every jl call is recorded
jget() {  # jget <key> : first scalar value of <key> anywhere in the JSON on stdin
  python3 -c '
import json, sys
key = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception:
    print(""); sys.exit(0)
def find(x):
    if isinstance(x, dict):
        if key in x and not isinstance(x[key], (dict, list)):
            return x[key]
        for v in x.values():
            r = find(v)
            if r is not None:
                return r
    elif isinstance(x, list):
        for v in x:
            r = find(v)
            if r is not None:
                return r
    return None
r = find(d)
print("" if r is None else r)' "$1"
}
resources() {  # running/paused counts + balance only (jl status also returns account details: never printed)
  jl status --json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
r, b = d.get("resources", {}), d.get("balance", {})
print(json.dumps({"running_instances": r.get("running_instances"), "paused_instances": r.get("paused_instances"),
                  "balance_inr": b.get("balance")}))'
}

# ---- code bundle (the record of exactly which JEPA code ran) ------------------------------
say "code bundle"
( cd "$ROOT" && COPYFILE_DISABLE=1 tar czf "$ART/code.tgz" --no-mac-metadata --no-xattrs \
    --uid 0 --gid 0 --uname root --gname root --exclude '__pycache__' --exclude '*.pyc' \
    pyproject.toml src/ayas/__init__.py src/ayas/anomaly.py training/jepa configs/twin.yaml )
( cd "$WORK" && mkdir code && tar xzf "$ART/code.tgz" -C code && cd code && \
  find . -type f -print0 | sort -z | xargs -0 shasum -a 256 > "$ART/code_sha256.txt" )

RDATA_ARGS=(); MANIFESTS=(); INST_NAME="$NAME"; CREATED=0
if [[ "$MODE" == "fresh" ]]; then
  [[ ${#DATA[@]} -gt 0 ]] || { echo "fresh mode needs --data <local grid dir>" >&2; exit 2; }
  REMOTE="/home/jepa_job_${STAMP}"; REPO="$REMOTE"; RUN_OUT="$REMOTE/runs/$NAME"
  say "data bundle from ${DATA[*]}"
  "$PY" "$ROOT/training/jepa/pack.py" --src "${DATA[@]}" --out "$WORK/data.tgz" --splits jepa_train --arms burst \
      | tee "$ART/data_manifest.json" | tee -a "$LOG"
  for d in "${DATA[@]}"; do
    RDATA_ARGS+=("$REMOTE/data/$(basename "$d")")
    MANIFESTS+=("$REMOTE/data/$(basename "$d")/pairs_manifest.json")
  done
  CONFIG="${CONFIG:-training/jepa/configs/smoke.yaml}"; MAXMIN="${MAXMIN:-12}"; AFTER="${AFTER:-destroy}"
else
  [[ -n "$MACHINE" && ${#RDATA[@]} -gt 0 ]] || { echo "on mode needs --machine <id> --remote-data <post dir>" >&2; exit 2; }
  [[ "$AFTER" == "destroy" ]] && { echo "refusing --after destroy in on mode (not this script's instance)" >&2; exit 2; }
  GJ="$(jl get "$MACHINE" --json 2>/dev/null || true)"
  STATUS="$(echo "$GJ" | jget status)"; INST_NAME="$(echo "$GJ" | jget name)"; INST_NAME="${INST_NAME:-machine$MACHINE}"
  echo "{\"machine_id\": \"$MACHINE\", \"name\": \"$INST_NAME\", \"status\": \"$STATUS\"}" > "$ART/instance_before.json"
  [[ "$STATUS" == "Running" ]] || { echo "instance $MACHINE ($INST_NAME) is '$STATUS': resume it first (jl resume; use the id it returns)" >&2; exit 2; }
  REMOTE="/home/jepa_runs/${NAME}_${STAMP}"; REPO="/home/ayas"; RUN_OUT="$REMOTE"
  RDATA_ARGS=("${RDATA[@]}")
  for d in "${RDATA[@]}" "${EVAL[@]:-}"; do
    [[ -n "$d" ]] || continue
    MANIFESTS+=("$d/pairs_manifest.json" "$d/matrix_manifest.json" "$(dirname "$d")/pairs_manifest.json" "$(dirname "$d")/matrix_manifest.json")
  done
  CONFIG="${CONFIG:-training/jepa/configs/full.yaml}"; MAXMIN="${MAXMIN:-120}"; AFTER="${AFTER:-none}"
  MAXWAIT_MIN=$(( ${MAXMIN%.*} + 90 ))                 # training + eval + export
fi
say "mode $MODE, instance $INST_NAME, before: $(resources)"

# ---- collect + verify, then the lifecycle action -------------------------------------------
RUN_ID=""; T_CREATE=$(date +%s); VERIFIED=0
VENV_ACT='. /home/jepa_venv/bin/activate 2>/dev/null'
[[ "$MODE" == "on" ]] && VENV_ACT='. /home/venvs/jepa/bin/activate 2>/dev/null'
collect() {
  [[ -n "$MACHINE" ]] || return 0
  say "collecting artifacts into $ART"
  if [[ -n "$RUN_ID" ]]; then
    jlx run logs "$RUN_ID" --tail 1000000 > "$ART/run_logs_full.txt" 2>&1 || say "run logs failed"
    jlx run status "$RUN_ID" --json > "$ART/run_status.json" 2>&1 || say "run status failed"
    mkdir -p "$ART/jl-runs"
    jlx download "$MACHINE" "/home/jl-runs/$RUN_ID" "$ART/jl-runs/$RUN_ID" -r >> "$LOG" 2>&1 || say "jl-runs download failed"
  fi
  jlx download "$MACHINE" "$RUN_OUT" "$ART/run_out" -r >> "$LOG" 2>&1 || say "run output download failed"
  jlx exec "$MACHINE" -- sh -lc "$VENV_ACT; nvidia-smi; python -c 'import sys, torch, timm; print(sys.version.split()[0], torch.__version__, timm.__version__)'; pip freeze 2>/dev/null" \
      > "$ART/remote_env.txt" 2>&1 || true
  mkdir -p "$ART/data_manifests"
  for m in "${MANIFESTS[@]:-}"; do
    [[ -n "$m" ]] || continue
    jlx download "$MACHINE" "$m" "$ART/data_manifests/$(echo "${m#/home/}" | tr '/' '_')" >> "$LOG" 2>&1 || true
  done
}
verify() {
  say "verifying artifacts (sizes, checkpoints load and run, MANIFEST.md)"
  local rc=0
  "$PY" "$ROOT/training/jepa/artifacts.py" "$ART" --require encoder.pt summary.json metrics.jsonl output.log \
      --note "JarvisLabs run $RUN_ID on machine $MACHINE ($INST_NAME, $MODE mode, $GPU ${SPOT:-on-demand}); remote dir $RUN_OUT; config $CONFIG; data ${RDATA_ARGS[*]}; train args: ${EXTRA[*]:-none}" \
      > "$WORK/verify.out" 2>&1 || rc=$?
  cat "$WORK/verify.out" >> "$LOG"
  tail -25 "$WORK/verify.out"
  [[ $rc -eq 0 ]] && VERIFIED=1 || say "verification exit code $rc"
}
finish() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ $CREATED -eq 1 || ( "$MODE" == "on" && -n "$RUN_ID" ) ]]; then
    collect
    if [[ -n "$RUN_ID" ]]; then
      local final="$ART_BASE/${INST_NAME}_${RUN_ID}"
      [[ -d "$final" ]] || { mv "$ART" "$final" && ART="$final" && LOG="$ART/jl_runner.log"; }
    fi
    verify
  fi
  local action="$AFTER"
  if [[ $VERIFIED -ne 1 && "$action" == "destroy" ]]; then
    say "VERIFICATION FAILED: pausing instead of destroying (inspect $ART, then destroy $MACHINE by hand)"
    action="pause"
  fi
  if [[ $VERIFIED -ne 1 && "$action" == "pause" && "$MODE" == "on" ]]; then
    say "verification failed: leaving $MACHINE running so nothing is lost (download by hand, then pause)"
    action="none"
  fi
  if [[ $CREATED -eq 1 || "$MODE" == "on" ]]; then
    case "$action" in
      destroy) say "jl destroy $MACHINE"; jlx destroy "$MACHINE" --yes --json >> "$LOG" 2>&1 || say "destroy FAILED: destroy $MACHINE by hand" ;;
      pause)   say "jl pause $MACHINE";   jlx pause "$MACHINE" --yes --json >> "$LOG" 2>&1 || say "pause FAILED" ;;
      *)       say "instance $MACHINE left as it is ($action)" ;;
    esac
  fi
  local t=$(( $(date +%s) - T_CREATE ))
  echo "{\"seconds_since_start\": $t, \"machine_id\": \"$MACHINE\", \"name\": \"$INST_NAME\", \"created_here\": $CREATED, \"after\": \"$action\", \"verified\": $VERIFIED}" > "$ART/instance.json"
  say "wall time ${t}s; after: $(resources)"
  "$PY" "$ROOT/training/jepa/artifacts.py" "$ART" --require encoder.pt summary.json metrics.jsonl output.log \
      --note "JarvisLabs run $RUN_ID on machine $MACHINE ($INST_NAME, $MODE mode); after: $action, verified before it: $VERIFIED" \
      >/dev/null 2>&1 || true                          # the MANIFEST also lists instance.json and the runner log
  rm -rf "$WORK"
  say "artifacts: $ART"
  exit $rc
}
trap finish EXIT

# ---- instance and code -------------------------------------------------------------------
if [[ "$MODE" == "fresh" ]]; then
  say "jl create --gpu $GPU $SPOT --storage $STORAGE --name $NAME"
  CJ="$(jlx create --gpu "$GPU" $SPOT --storage "$STORAGE" --name "$NAME" --yes --json)"
  MACHINE="$(echo "$CJ" | jget machine_id)"
  [[ -n "$MACHINE" ]] || { say "create failed: $CJ"; exit 1; }
  CREATED=1
  T_CREATE=$(date +%s)
  echo "$MACHINE" > "$ART/machine_id"
  say "created machine $MACHINE; upload to $REMOTE"
  jlx exec "$MACHINE" -- mkdir -p "$REMOTE/data" >> "$LOG" 2>&1
  jlx upload "$MACHINE" "$ART/code.tgz" "$REMOTE/code.tgz" >> "$LOG" 2>&1
  jlx upload "$MACHINE" "$WORK/data.tgz" "$REMOTE/data.tgz" >> "$LOG" 2>&1
  # a separate venv (inherits the template's CUDA torch) so pip never touches the template packages
  SETUP="cd $REMOTE && tar xzf code.tgz && tar xzf data.tgz -C data && rm data.tgz; \
[ -x /home/jepa_venv/bin/python ] || python -m venv --system-site-packages /home/jepa_venv; \
. /home/jepa_venv/bin/activate; pip install -q timm==1.0.31 pyyaml opencv-python-headless 2>&1 | tail -2"
else
  echo "$MACHINE" > "$ART/machine_id"
  if [[ $SYNC -eq 1 ]]; then
    say "code sync: sh tools/jl/sync_code.sh $MACHINE (-> /home/ayas)"
    echo "sh tools/jl/sync_code.sh $MACHINE" >> "$ART/commands.txt"
    sh "$ROOT/tools/jl/sync_code.sh" "$MACHINE" >> "$LOG" 2>&1 || { say "code sync failed"; exit 1; }
  fi
  # /home/venvs/jepa: Python 3.11 from uv (as bootstrap.sh makes /home/venvs/ayas), CUDA torch,
  # timm, onnx/onnxruntime, the project --no-deps; versions of the shared packages from constraints-mac.txt
  SETUP="mkdir -p $REMOTE; . /home/ayas_jobs/env.sh; export HF_HOME=/home/.cache/huggingface \
UV_PYTHON_INSTALL_DIR=/home/.uv/python UV_CACHE_DIR=/home/.cache/uv; cd $REPO; \
[ -x /home/.uvtool/bin/uv ] || { python3 -m venv /home/.uvtool && /home/.uvtool/bin/pip install -q uv; }; \
[ -x /home/venvs/jepa/bin/python ] || { /home/.uvtool/bin/uv python install 3.11 && /home/.uvtool/bin/uv venv --python 3.11 --seed /home/venvs/jepa; }; \
. /home/venvs/jepa/bin/activate; \
python -c 'import torch, torchvision' 2>/dev/null || pip install -q $TORCH_SPEC --index-url $TORCH_INDEX; \
pip install -q -c tools/jl/constraints-mac.txt timm==1.0.31 pyyaml numpy opencv-python-headless scikit-learn onnx==1.23.2 onnxruntime 2>&1 | tail -2; \
pip install -q --no-deps -e . 2>&1 | tail -2; \
for d in ${RDATA_ARGS[*]} ${EVAL[*]:-}; do n=\$(find \$d/grids -name '*.json' 2>/dev/null | wc -l); echo \"grids under \$d: \$n\"; [ \$n -gt 0 ] || { echo \"no grids under \$d (point at the post dir that holds grids/)\"; exit 3; }; done"
fi

CMD="set -e; $SETUP; \
python -c 'import sys, torch, timm; print(\"python\", sys.version.split()[0], \"torch\", torch.__version__, \"timm\", timm.__version__, torch.cuda.get_device_name(0))'; \
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; \
python training/jepa/train.py --config $CONFIG --data ${RDATA_ARGS[*]} --out $RUN_OUT --device cuda \
  --max-minutes $MAXMIN ${EXTRA[*]:-}"
if [[ ${#EVAL[@]} -gt 0 ]]; then        # anomaly + invariance eval on the instance (test grids never leave it)
  CMD="$CMD; PYTHONPATH=$REPO/src python training/jepa/eval_anomaly.py --ckpt $RUN_OUT/encoder.pt \
  --train ${RDATA_ARGS[*]} --test ${EVAL[*]} --device cuda --report $RUN_OUT/eval/jepa_anomaly.md \
  --fig-dir $RUN_OUT/eval/figures --heat-dir $REMOTE/../heat_${NAME}_${STAMP} --out $RUN_OUT/eval"
fi
if [[ $EXPORT -eq 1 ]]; then             # ONNX files (latency is measured on the Mac / Pi, not here)
  CMD="$CMD; PYTHONPATH=$REPO/src python training/jepa/export_onnx.py --ckpt $RUN_OUT/encoder.pt \
  --train ${RDATA_ARGS[*]} --out $RUN_OUT/onnx --name jepa --runs 0"
fi
say "jl run: $CMD"
RJ="$(jlx run --on "$MACHINE" --json --yes -- sh -lc "$CMD")"
RUN_ID="$(echo "$RJ" | jget run_id)"
[[ -n "$RUN_ID" ]] || { say "run start failed: $RJ"; exit 1; }
echo "$RUN_ID" > "$ART/run_id"
say "run $RUN_ID started"

# ---- poll --------------------------------------------------------------------------------
T_RUN=$(date +%s); STATE=""
sleep 20
while true; do
  jl run logs "$RUN_ID" --tail 25 2>&1 | tee -a "$LOG" | tail -6 || true
  STATE="$(jl run status "$RUN_ID" --json 2>/dev/null | jget state)"
  say "state=$STATE elapsed=$(( $(date +%s) - T_RUN ))s"
  case "$STATE" in succeeded|failed|stopped|error|killed) break ;; esac
  if (( $(date +%s) - T_RUN > MAXWAIT_MIN * 60 )); then
    say "max wait reached: stopping run"; jlx run stop "$RUN_ID" --json >> "$LOG" 2>&1 || true; STATE="stopped"; break
  fi
  sleep "$POLL"
done
echo "{\"run_id\": \"$RUN_ID\", \"state\": \"$STATE\", \"run_seconds\": $(( $(date +%s) - T_RUN )), \"mode\": \"$MODE\", \"gpu\": \"$GPU\", \"spot\": \"${SPOT:-no}\"}" \
  > "$ART/run.json"
say "run finished: $STATE"
[[ "$STATE" == "succeeded" ]]          # the EXIT trap collects, verifies, then applies the lifecycle action
