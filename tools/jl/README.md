# Twin renders on JarvisLabs (`tools/jl/`)

The Blender twin renders and post-processing run on one JarvisLabs **L4 on-demand container**
named `ayas-render` (region IN2, 200 GB disk). The data stays on the instance for the later
JEPA and YOLO training.

| On the instance | What it is |
|---|---|
| `/home/blender/blender-5.2.3-linux-x64/blender` | Blender 5.2.3 LTS, the Mac's version (official tarball, sha256 checked) |
| `/home/venvs/ayas` | Python 3.11 venv: `pip install -e ".[twin,dev]"`, pinned to the Mac's versions (`constraints-mac.txt`) |
| `/home/ayas` | the code (`sync_code.sh`) |
| `/home/ayas_jobs/{env.sh,bootstrap.sh,render_all.sh,backup_drive.sh,train_yolo.sh}` | the job scripts from this folder |
| `/home/setup_sys.sh` | apt packages headless Blender needs. Container system installs are lost on pause, so every job runs it first. |
| `/home/data/{twin_matrix,twin_train,jepa_train}` | the job dirs: `plans/`, `work/<scene>/` (EXRs kept), post outputs |
| `/home/data/logs/render_all.log` | the log of every `render_all.sh` run (also in `jl run logs`) |
| `/home/.cache` | the Cycles kernel cache (`XDG_CACHE_HOME`) |

Everything is under `/home`, so it all survives pause and resume. `AYAS_BLENDER` (in
`env.sh`) is how `ayas.twin.render.find_blender` finds Blender. Nothing in `src/`, `twin/`
or `configs/` was changed for the cloud.

**Device.** The config says `render.device: METAL`, which is Mac only. On Linux,
`render_scene.setup_device` quietly falls back to the **CPU** when the device is missing. So
every command here passes `--device OPTIX`. `twin_jobs.py` also stops a job if a scene reports
any other device. If you run `ayas-twin run|matrix` or `training/jepa/pairs.py --render`
yourself on the instance, always add `--device OPTIX`.

**Render now, post later.** `pipeline.run_plans` (used by `ayas-twin run|matrix` and
`pairs.py --render`) only counts a scene as done once it has a **sidecar**, which is written by
post. If post is deferred, a restart would re-render everything. Worse, `render_batch` deletes the
old `render.exr` before it renders. So `twin_jobs.py` tracks the two stages separately:

- **Rendered:** `work/<scene>/render_meta.json` has `exr_channels`, no error, and the ambient is
  merged. The texture maps are deleted after the render; the EXR is kept.
- **Posted:** `<post dir>/sidecar/<scene>.json` exists.

Never run `ayas-twin run|matrix` or `run_plans` on a job dir that was rendered with
post deferred.

## Commands (run on the Mac; `ID` = the machine_id; every `jl` command takes `--json`)

```bash
ID=574713   # resume can return a NEW id: always use the one jl resume prints

# 0. Start the instance, and check that the resume did not move it to a new id
jl resume $ID --yes --json            # use the returned machine_id from here on
jl status --json

# 1. Sync the code. This copies the repo to /home/ayas, and tools/jl/* to /home/ayas_jobs and /home/setup_sys.sh.
#    It skips .venv*, data/, backups/, docs/, reports/, hardware/ (all but the Fusion generator the
#    twin rig reads), caches and *.tar.gz, and it scans the copy for local paths and keys first.
sh tools/jl/sync_code.sh $ID
#    Only after a code change that touches dependencies (bootstrap is idempotent):
jl run --on $ID --json --yes -- sh -lc 'sh /home/ayas_jobs/bootstrap.sh'

# 2. Render (EXRs kept, post deferred). This is resumable: if it stops, run the same command again.
jl run --on $ID --json --yes -- sh -lc 'sh /home/ayas_jobs/render_all.sh --jobs matrix,train,jepa --jepa-n K'
#    K = the number of JEPA training plates in docs/JEPA.md. Leave jepa out until K is known.
#    The defaults are: --matrix-n 15 --matrix-start-seed 900100 (x 14 rows; the first 2 plates get overview renders), --train-n 800,
#    --jepa-start-seed 100000, --procs 3 (3 Blender processes on the one L4: the best measured), --device OPTIX.
#    To post each batch straight after its render, add --stage both (it uses the current code).

# 3. Post with the final code (CPU only; a cheaper GPU is fine for this too).
sh tools/jl/sync_code.sh $ID
jl run --on $ID --json --yes -- sh -lc 'sh /home/ayas_jobs/render_all.sh --stage post --jobs matrix,train,jepa --jepa-n K --post-tag final'
#    This writes to <job>/post_final/, so each post run gets a fresh dir and no stale tiles.
#    The showcase matrix plates get --save-bursts. To re-post again, use a new --post-tag, or add --force-post.

# 3b. Back up to the user's Google Drive (rclone remote gdrive; copy only, MD5-verified, resumable).
#     The USER uploads their rclone sign-in first (their token: never print, commit or copy it elsewhere):
#       ~/.local/bin/jl upload $ID ~/.config/rclone/rclone.conf /home/.config/rclone/rclone.conf
jl run --on $ID --json --yes -- sh -lc 'sh /home/ayas_jobs/backup_drive.sh --what renders'   # EXRs, plans, manifests
jl run --on $ID --json --yes -- sh -lc 'sh /home/ayas_jobs/backup_drive.sh --what post'      # <job>/post_final
#     -> "Ayas twin data (rclone)/render_2026-11-15/<job>/..." plus logs/. Before destroying the instance:
#       jl exec $ID -- rm -f /home/.config/rclone/rclone.conf

# 3c. YOLO capture arms (naive vs burst): venv, naive re-post of the training renders, train, export, eval.
jl run --on $ID --json --yes -- sh -lc 'sh /home/ayas_jobs/train_yolo.sh --venv-only'   # optional, during post
jl run --on $ID --json --yes -- sh -lc 'sh /home/ayas_jobs/train_yolo.sh'               # -> /home/data/yolo_runs/
#     JEPA: training/jepa/jl_train.sh on --machine $ID ... (docs/JEPA.md).

# 4. Follow the progress
jl run logs <run_id> --tail 30         # the footer shows still running / succeeded / failed
jl run status <run_id> --json
jl exec $ID -- sh -lc 'tail -20 /home/data/logs/render_all.log'
jl exec $ID -- sh -lc 'cd /home/ayas && /home/venvs/ayas/bin/python tools/jl/twin_jobs.py status --out /home/data/twin_matrix --plans /home/data/twin_matrix/matrix_manifest.json'
jl exec $ID -- sh -lc 'tail -3 /home/data/twin_matrix/jobs_log.jsonl; cat /home/data/twin_matrix/render_failures.jsonl 2>/dev/null | tail -3'
jl exec $ID -- nvidia-smi
jl exec $ID -- df -h /home

# 5. Stop a run (it resumes from where it stopped)
jl run stop <run_id> --json

# 6. Pause (compute billing stops; storage billing continues)
jl pause $ID --yes --json
jl status --json                        # check: running_instances 0, paused_instances 1
```

Plans are written once and then reused. `render_all.sh` never rewrites `plans/` or a manifest that already exists. If the code that writes plans changes before the render (the sampler, matrix or pairs code), delete `<job>/plans` and its manifest, so the next run writes them again. Never do this after scenes are rendered.

Run only one `render_all.sh` at a time; it takes a lock. To render one job by hand, use
`/home/venvs/ayas/bin/python tools/jl/twin_jobs.py render --help`. It does the same thing
as `render_all.sh`, with full control of plans, procs, batch, `--overview`, `--post` and `--limit`.

## Job definitions (what `render_all.sh` runs)

| Job | Plans | Render | Post |
|---|---|---|---|
| matrix | `ayas-twin matrix --n 15 --start-seed 900100 --out /home/data/twin_matrix --plans-only` (test plates from seed 900100: 900000-900068 tuned the gate; 14 rows each) | the first 2 plates with `--overview`, then all | the first 2 plates with `--save-bursts`, then all (test split: full grids for the burst and naive arms) |
| train | `ayas-twin plan --n 800 --start-seed 0 --out /home/data/twin_train` | all | YOLO tiles (train arm) |
| jepa | `python training/jepa/pairs.py --n K --start-seed 100000 --out /home/data/jepa_train --arms burst` (training plates 100000..899999; `matrix` refuses seeds below 900000 on purpose) | all | `--jepa`: `pairs.jepa_config` of the current `configs/twin.yaml` (burst grids as JPEG q95, split `jepa_train`, no tiles) |

`pairs.py --render` would also work. But it posts inline, deletes the EXRs (no keep-work),
and renders on the config's METAL device unless you pass `--device OPTIX`. So `render_all.sh`
only uses `pairs.py` to write the plans.

## Measured on the L4, and the estimate

See `artifacts/jarvislabs/ayas-render_setup/MANIFEST.md` for the benchmark numbers, the
Mac consistency test and the cost estimate.
