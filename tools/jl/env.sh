# Sourced by every job on the instance (instance copy: /home/trinetra_jobs/env.sh).
# Everything lives under /home, which survives pause/resume.
export TRINETRA_BLENDER=/home/blender/blender-5.2.3-linux-x64/blender   # trinetra.twin.render.find_blender reads this
export XDG_CACHE_HOME=/home/.cache                                       # Cycles kernel cache survives pause
export TRINETRA_VENV=/home/venvs/trinetra
export TRINETRA_REPO=/home/trinetra
export TRINETRA_DATA=/home/data
export PYTHONUNBUFFERED=1
export PATH="$TRINETRA_VENV/bin:$PATH"
