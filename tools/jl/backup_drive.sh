#!/usr/bin/env bash
# Back up TriNetra twin data from the JarvisLabs instance to the user's Google Drive (rclone remote `gdrive`).
# Instance copy: /home/trinetra_jobs/backup_drive.sh   (repo copy: tools/jl/backup_drive.sh)
#
# Run it tracked, from the Mac:
#   jl run --on <id> --json --yes -- sh -lc 'bash /home/trinetra_jobs/backup_drive.sh --what renders'
#
# Why tar bundles: rclone's shared Google client id is throttled per file. A file-by-file copy of the
# ~10k render files stalled at bytes/s after ~300 files (15 Nov), while one 100 MB file went at 14 MiB/s.
# So every job dir goes up as a few ~2.5 GB tars:
#   <tag>/<job>/meta.tar                   everything in the job dir except work/ and post_*/ (plans, manifests, logs)
#   <tag>/<job>/work_<first>-<last>.tar    --chunk scenes' work/<scene>/ dirs (EXR + render_meta + plan + logs)
#   <tag>/<job>/post_<p>.tar               one per post dir (--what post)
#   <tag>/logs.tar                         /home/data/logs + /home/jl-runs (--what logs; always added)
#   <tag>/MANIFEST_<what>.txt              md5, bytes and name of every tar, plus each tar's file list (lists_<what>.tar)
# Each tar is built in --tmp, uploaded with rclone copyto, and its local MD5 is compared with the MD5 Google
# computes. Only then is <state>/<name>.ok written; a re-run skips tars with an .ok file (resumable). Copy only:
# nothing on Drive is deleted. Restore: rclone copy the tars back, then `tar -C <job dir> -xf <tar>`.
#
# Needs /home/.config/rclone/rclone.conf: the USER uploads it from the Mac (their token: never commit or print it):
#   ~/.local/bin/jl upload <id> ~/.config/rclone/rclone.conf /home/.config/rclone/rclone.conf
# Delete it before destroying the instance:  jl exec <id> -- rm -f /home/.config/rclone/rclone.conf
#
# Options:
#   --what renders|post|logs   renders = meta + work chunks; post = the post_* dirs                 (renders)
#   --jobs "A B C"             job dirs under --data          ("twin_matrix jepa_train twin_train")
#   --post-dirs "P Q"          post dir names for --what post (default: every post_* dir of each job)
#   --tag T                    Drive folder                    (render_2026-11-15)
#   --root NAME                top Drive folder                ("TriNetra twin data (rclone)"; with a NEW client id and
#                              scope drive.file the old folder is invisible, so use "TriNetra twin data")
#   --chunk N                  scenes per work tar             (50, about 2.5 GB)
#   --parallel N               tars built/uploaded at a time   (3)
#   --chunk-size S             Drive upload chunk              (256M: 4x fewer API requests than 64M)
#   --attempts N               upload attempts per tar         (6; waits 90 s, 180 s, ... between them)
#   --upload-timeout D         cap per attempt (timeout(1))    (40m; a healthy 2.5 GB tar takes 2-3 min)
#   --data DIR (/home/data)   --tmp DIR (/home/data/.backup_tmp)
#
# Google 403 "Quota exceeded ... defaultPerMinutePerProject": rclone's SHARED client id is one Google project
# for all rclone users, so its per-minute quota runs out globally (15 Nov: two tars hung for 1-2.5 h inside
# one rclone attempt). The bounded attempts + bigger chunks ride it out; the real fix is the user's own
# client id (rclone.org/drive/#making-your-own-client-id), set in their rclone.conf on the Mac.
set -uo pipefail

RCLONE=/home/bin/rclone; CONF=/home/.config/rclone/rclone.conf
DEST_ROOT="gdrive:TriNetra twin data (rclone)"
WHAT=renders; TAG=render_2026-11-15; DATA=/home/data; JOBS="twin_matrix jepa_train twin_train"; POST_DIRS=""
CHUNK=50; PAR=3; TMPD=""; CHUNK_SIZE=256M; ATTEMPTS=6; UP_TIMEOUT=40m
while [ $# -gt 0 ]; do
  case "$1" in
    --what) WHAT=$2; shift 2;;
    --jobs) JOBS=$2; shift 2;;
    --post-dirs) POST_DIRS=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    --root) DEST_ROOT="gdrive:$2"; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --parallel) PAR=$2; shift 2;;
    --data) DATA=$2; shift 2;;
    --tmp) TMPD=$2; shift 2;;
    --chunk-size) CHUNK_SIZE=$2; shift 2;;
    --attempts) ATTEMPTS=$2; shift 2;;
    --upload-timeout) UP_TIMEOUT=$2; shift 2;;
    -h|--help) sed -n '2,41p' "$0"; exit 0;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
case "$WHAT" in renders|post|logs) ;; *) echo "--what must be renders, post or logs"; exit 2;; esac
[ -x "$RCLONE" ] || { echo "rclone missing at $RCLONE"; exit 3; }
[ -s "$CONF" ] || { echo "no $CONF: upload it from the Mac first (see the header)"; exit 3; }
TMPD=${TMPD:-$DATA/.backup_tmp}
STATE=$DATA/logs/backup_state/$TAG                      # the default root keeps its original state path
[ "$DEST_ROOT" = "gdrive:TriNetra twin data (rclone)" ] || \
  STATE=$DATA/logs/backup_state/$(printf '%s' "${DEST_ROOT#gdrive:}" | tr -c 'A-Za-z0-9._-' '_')/$TAG
LISTS=$STATE/lists
mkdir -p "$TMPD" "$STATE" "$LISTS"
RC=("$RCLONE" --config "$CONF")
FLAGS=(--drive-chunk-size "$CHUNK_SIZE" --retries 1 --low-level-retries 30 --tpslimit 4 --log-level NOTICE)
DEST="$DEST_ROOT/$TAG"

# rclone with bounded attempts: each capped by --upload-timeout, then a growing wait (quota windows are per minute)
rc_try() {   # label, rclone args...
  local label=$1 a; shift
  for ((a = 1; a <= ATTEMPTS; a++)); do
    # --foreground keeps rclone in the job's process group, so `jl run stop` reaches it (plain timeout makes its own
    # group: on 17 Nov a stopped run left rclone uploading). nice execs rclone, so the time limit still hits rclone.
    timeout --foreground "$UP_TIMEOUT" nice -n 5 "${RC[@]}" "$@" && return 0
    [ "$a" -lt "$ATTEMPTS" ] && { echo "attempt $a/$ATTEMPTS failed: $label; retry in $((a * 90)) s"; sleep $((a * 90)); }
  done
  return 1
}
remote_md5() {   # Drive path -> its MD5 (retried: the md5 query can hit the same quota)
  local a h
  for ((a = 1; a <= ATTEMPTS; a++)); do
    h=$("${RC[@]}" md5sum "$1" 2>/dev/null | cut -d' ' -f1)
    [ -n "$h" ] && { echo "$h"; return 0; }
    sleep $((a * 30))
  done
  return 1
}

# one tar: name (Drive path below $DEST, without .tar), base dir, list file (paths relative to base)
put_tar() {
  local name=$1 base=$2 list=$3 key ok tmp md5 sz remote t0
  key=${name//\//__}; ok="$STATE/$key.ok"; tmp="$TMPD/$key.tar"
  [ -f "$ok" ] && { echo "skip $name (done: $(cut -c1-12 "$ok"))"; return 0; }
  t0=$(date +%s)
  if ! tar -C "$base" -cf "$tmp" -T "$list"; then echo "TAR FAILED $name"; rm -f "$tmp"; return 1; fi
  sz=$(stat -c %s "$tmp"); md5=$(md5sum "$tmp" | cut -d' ' -f1)
  tar -tf "$tmp" > "$LISTS/$key.txt"
  if ! rc_try "$name" copyto "$tmp" "$DEST/$name.tar" "${FLAGS[@]}"; then
    echo "UPLOAD FAILED $name (after $ATTEMPTS attempts)"; rm -f "$tmp"; return 1
  fi
  remote=$(remote_md5 "$DEST/$name.tar")
  rm -f "$tmp"
  if [ "$remote" != "$md5" ]; then echo "MD5 MISMATCH $name (local $md5, drive ${remote:-none})"; return 1; fi
  echo "$md5  $sz  $name.tar" > "$ok"
  echo "ok $name.tar  $((sz / 1048576)) MiB  md5 $md5  $(( $(date +%s) - t0 )) s"
}

TASKS=()   # "name|base|list"
add_task() { TASKS+=("$1|$2|$3"); }

for j in $JOBS; do
  D=$DATA/$j
  [ -d "$D" ] || { echo "skip (missing): $D"; continue; }
  case "$WHAT" in
    renders)
      ( cd "$D" && ls -A | grep -v -e '^work$' -e '^post_' ) > "$LISTS/$j.meta.list"
      add_task "$j/meta" "$D" "$LISTS/$j.meta.list"
      mapfile -t scenes < <(ls "$D/work" | sort)
      for ((i = 0; i < ${#scenes[@]}; i += CHUNK)); do
        part=("${scenes[@]:i:CHUNK}")
        lf="$LISTS/$j.work_$i.list"; printf 'work/%s\n' "${part[@]}" > "$lf"
        add_task "$j/work_${part[0]}-${part[${#part[@]}-1]}" "$D" "$lf"
      done;;
    post)
      dirs=$POST_DIRS; [ -n "$dirs" ] || dirs=$(cd "$D" && ls -d post_* 2>/dev/null)
      for p in $dirs; do
        [ -d "$D/$p" ] || { echo "skip (missing): $D/$p"; continue; }
        echo "$p" > "$LISTS/$j.$p.list"; add_task "$j/$p" "$D" "$LISTS/$j.$p.list"
      done;;
  esac
done
printf 'data/logs\njl-runs\n' > "$LISTS/logs.list"
add_task "logs_$WHAT" /home "$LISTS/logs.list"
rm -f "$STATE/logs_${WHAT//\//__}.ok"        # logs change: always re-upload them

echo "=== backup_drive $(date -u +%FT%TZ) what=$WHAT tag=$TAG jobs=$JOBS: ${#TASKS[@]} tars, $PAR at a time"
for t in "${TASKS[@]}"; do
  IFS='|' read -r name base list <<< "$t"
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  put_tar "$name" "$base" "$list" &
done
wait

# manifest: md5/size/name of every finished tar + the per-tar file lists
fail=0; man="$TMPD/MANIFEST_$WHAT.txt"
{ echo "# TriNetra twin data backup, tag $TAG, what=$WHAT, $(date -u +%FT%TZ)"
  echo "# md5  bytes  tar   (restore: rclone copy the tars back, then tar -C <job dir> -xf <tar>)"
  for t in "${TASKS[@]}"; do
    name=${t%%|*}; ok="$STATE/${name//\//__}.ok"
    if [ -f "$ok" ]; then cat "$ok"; else echo "MISSING  $name.tar"; fail=1; fi
  done; } > "$man"
tar -C "$LISTS" -cf "$TMPD/lists_$WHAT.tar" .
rc_try manifest copyto "$man" "$DEST/MANIFEST_$WHAT.txt" "${FLAGS[@]}" || fail=1
rc_try lists copyto "$TMPD/lists_$WHAT.tar" "$DEST/lists_$WHAT.tar" "${FLAGS[@]}" || fail=1
cat "$man"
"${RC[@]}" size "$DEST" 2>/dev/null | sed 's/^/drive total: /'
rm -f "$man" "$TMPD/lists_$WHAT.tar"
echo "=== backup_drive done $(date -u +%FT%TZ) rc=$fail"
exit $fail
