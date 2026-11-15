#!/bin/sh
# Sync the code to the JarvisLabs instance (run on the Mac, instance must be Running):
#   sh tools/jl/sync_code.sh <machine_id>
# repo (minus tools/jl/rsync_filter.txt excludes: .venv*, data/, backups/, docs/, reports/, hardware/
# except the Fusion generator the twin rig reads, caches, *.tar.gz) -> /home/trinetra
# tools/jl/{env.sh,bootstrap.sh,render_all.sh,backup_drive.sh,train_yolo.sh} -> /home/trinetra_jobs/
# tools/jl/setup_sys.sh                                             -> /home/setup_sys.sh
# The Fusion generator's REPO_DIR (a local absolute path, used only for Fusion exports) is
# rewritten to /home/trinetra in the staged copy; the staged tree is scanned for local paths
# and key-like strings before anything is sent.
set -eu
ID=${1:?usage: sync_code.sh <machine_id>}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/trinetra_stage.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
rsync -a --filter="merge $REPO/tools/jl/rsync_filter.txt" "$REPO/" "$STAGE/trinetra/"
python3 - "$STAGE/trinetra/hardware/enclosure/trinetra_handheld_fusion.py" <<'EOF'
import re, sys
p = sys.argv[1]
t = open(p).read()
t = re.sub(r"(?m)^REPO_DIR = .*$", "REPO_DIR = '/home/trinetra'", t)
open(p, "w").write(t)
EOF
if grep -rIl --exclude=sync_code.sh -E "/Users/|@gmail\.com|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{20}|JL_API_KEY=" "$STAGE/trinetra"; then
  echo "refusing to upload: the files above hold local paths or key-like strings"; exit 3
fi
mkdir -p "$STAGE/jobs"
cp "$REPO/tools/jl/env.sh" "$REPO/tools/jl/bootstrap.sh" "$REPO/tools/jl/render_all.sh" \
   "$REPO/tools/jl/backup_drive.sh" "$REPO/tools/jl/train_yolo.sh" "$STAGE/jobs/"
SSHCMD=$(jl get "$ID" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["ssh_command"])')
eval "set -- $SSHCMD"
shift                                   # drop "ssh"
TARGET=""; OPTS=""
while [ $# -gt 0 ]; do
  case "$1" in
    -p|-i|-o|-J|-F) OPTS="$OPTS $1 $2"; shift 2;;
    -*) OPTS="$OPTS $1"; shift;;
    *) TARGET=$1; shift;;
  esac
done
RSH="ssh$OPTS -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
$RSH "$TARGET" 'mkdir -p /home/trinetra /home/trinetra_jobs /home/data && command -v rsync >/dev/null || sh -c "apt-get update -qq >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null"'
rsync -az --delete --filter="merge $REPO/tools/jl/rsync_filter.txt" -e "$RSH" "$STAGE/trinetra/" "$TARGET:/home/trinetra/"
rsync -az -e "$RSH" "$STAGE/jobs/" "$TARGET:/home/trinetra_jobs/"
rsync -az -e "$RSH" "$REPO/tools/jl/setup_sys.sh" "$TARGET:/home/setup_sys.sh"
echo "synced to $ID: /home/trinetra, /home/trinetra_jobs, /home/setup_sys.sh"
