#!/usr/bin/env bash
# Phase 1 on-device run (Raspberry Pi 5): system record, a smoke pass checked against the twin evaluation,
# a vision-only benchmark and a full benchmark with Llama (ExecuTorch); everything bundled in one .tgz.
#
# Run ON THE PI from the repo root, inside the venv from docs/APP.md section 2 (after `pip install -e .`):
#   bash tools/pi/phase1_run.sh                                   # defaults below
#   bash tools/pi/phase1_run.sh --detector hailo:models/yolov8n_steel.hef --advisor executorch_py --n 50
#
# Before it, copy from the Mac (replace <pi-host>; ~/ayas = the repo on the Pi):
#   rsync -av models/v8n_burst.onnx models/v8n_burst.json <pi-host>:~/ayas/models/
#   rsync -av reports/lighting_matrix.json <pi-host>:~/ayas/reports/
#   rsync -av artifacts/jarvislabs/ayas-render_574738_final/twin_matrix/post_final/bursts/ \
#         <pi-host>:~/ayas/data/twin_demo/bursts/            # 28 twin bursts, ~1 GB
# After it, copy the bundle back:  rsync -av <pi-host>:~/ayas/reports/pi_phase1_*.tgz reports/
#
# Options:
#   --bursts DIR       twin bursts used as the camera            (data/twin_demo/bursts)
#   --detector SPEC    onnx:<path> | hailo:<hef>                (onnx:models/v8n_burst.onnx)
#   --advisor NAME     executorch | executorch_py | mock        (executorch: the llama_main runner in app.yaml)
#   --n N              vision-only benchmark inspections        (30)
#   --n-llm N          benchmark inspections with Llama advice  (8; each waits for the full answer)
#   --skip-llm         vision only (e.g. while the .pte is not on the Pi yet)
set -uo pipefail

BURSTS=data/twin_demo/bursts; DET=onnx:models/v8n_burst.onnx; ADV=executorch; N=30; NLLM=8; SKIP_LLM=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bursts) BURSTS=$2; shift 2;;
    --detector) DET=$2; shift 2;;
    --advisor) ADV=$2; shift 2;;
    --n) N=$2; shift 2;;
    --n-llm) NLLM=$2; shift 2;;
    --skip-llm) SKIP_LLM=1; shift;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) echo "unknown option $1"; exit 2;;
  esac
done

APP="ayas-app"; command -v ayas-app >/dev/null 2>&1 || APP="python -m ayas.app.cli"
[ -d "$BURSTS" ] && ls "$BURSTS"/*.npz >/dev/null 2>&1 || { echo "no .npz bursts in $BURSTS (copy them first, see the header)"; exit 3; }
OUT=reports/pi_phase1_$(hostname)_$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
echo "=== Phase 1 Pi run -> $OUT (detector $DET, advisor $ADV)"
rc=0

# 1. system record (what this run was measured on)
{
  echo "## date";      date -Iseconds
  echo "## model";     tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo
  echo "## uname";     uname -a
  echo "## os";        grep -E '^(PRETTY_NAME|VERSION)=' /etc/os-release 2>/dev/null
  echo "## memory";    free -m
  echo "## disk";      df -h .
  echo "## temp";      vcgencmd measure_temp 2>/dev/null || cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null
  echo "## throttled"; vcgencmd get_throttled 2>/dev/null
  echo "## python";    python -V 2>&1
  echo "## links";     ip -brief link 2>/dev/null
  echo "## ayas-app check"; $APP check 2>&1
} > "$OUT/system.txt" 2>&1
echo "--- 1/4 system record: $OUT/system.txt"

# 2. smoke: every burst once, template advice; then compare each verdict with the twin evaluation
$APP run --source "twin:$BURSTS" --detector "$DET" --advisor mock --no-loop --json --quiet \
    --log-dir "$OUT/log_smoke" > "$OUT/smoke.jsonl" 2> "$OUT/smoke.err" || { echo "SMOKE FAILED (see $OUT/smoke.err)"; rc=1; }
python - "$OUT/smoke.jsonl" reports/lighting_matrix.json > "$OUT/smoke_vs_twin.txt" 2>&1 <<'EOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
try:
    caps = json.load(open(sys.argv[2]))["captures"]
except FileNotFoundError:
    print(f"{len(recs)} records; reports/lighting_matrix.json not on the Pi, so no comparison"); sys.exit()
ref = {(c["scene"], c["k"]): c for c in caps if c["arm"] == "burst"}
same = n = 0
for r in recs:
    scene, k = r["part_id"].rsplit("_v", 1)
    c = ref.get((scene, int(k)))
    if c is None:
        continue
    want = "RETAKE" if c["gate"] == "RETAKE" else c["pred_verdict"]
    got = "RETAKE" if r["capture"]["gate"] == "RETAKE" else r["verdict"]
    n += 1; same += got == want
    if got != want:
        print(f"DIFFERS {r['part_id']}: Pi {got}, twin evaluation {want}")
print(f"Pi verdicts equal to the twin evaluation (burst arm, same captures): {same}/{n} ({len(recs)} records)")
EOF
echo "--- 2/4 smoke: $(tail -1 "$OUT/smoke_vs_twin.txt")"

# 3. vision-only benchmark (camera-to-verdict latency, FPS, memory, temperature)
$APP run --source "twin:$BURSTS" --detector "$DET" --no-advice --bench "$N" --warmup 2 \
    --reports "$OUT" --log-dir "$OUT/log_vision" --quiet > "$OUT/bench_vision.log" 2>&1 || { echo "VISION BENCH FAILED (see $OUT/bench_vision.log)"; rc=1; }
echo "--- 3/4 vision bench: $(grep -m1 'vision p50' "$OUT/bench_vision.log")"

# 4. full benchmark with the Llama advisor (TTFT, tokens/s, runner memory)
if [ "$SKIP_LLM" = 1 ]; then
  echo "--- 4/4 Llama bench skipped (--skip-llm)"
else
  $APP run --source "twin:$BURSTS" --detector "$DET" --advisor "$ADV" --bench "$NLLM" --warmup 1 \
      --reports "$OUT" --log-dir "$OUT/log_llama" --quiet > "$OUT/bench_llama.log" 2>&1 || { echo "LLAMA BENCH FAILED (see $OUT/bench_llama.log)"; rc=1; }
  grep -q "WARNING" "$OUT/bench_llama.log" && echo "    NOTE: the app printed a WARNING (advisor fallback?): grep WARNING $OUT/bench_llama.log"
  echo "--- 4/4 Llama bench: $(grep -m1 'vision p50' "$OUT/bench_llama.log")"
fi

tar czf "$OUT.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")" && echo "=== bundle: $OUT.tgz ($(du -h "$OUT.tgz" | cut -f1)); copy it back to the Mac (see the header)"
exit $rc
