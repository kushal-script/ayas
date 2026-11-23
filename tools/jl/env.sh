# Sourced by every job on the instance (instance copy: /home/ayas_jobs/env.sh).
# Everything lives under /home, which survives pause/resume.
export AYAS_BLENDER=/home/blender/blender-5.2.3-linux-x64/blender   # ayas.twin.render.find_blender reads this
export XDG_CACHE_HOME=/home/.cache                                       # Cycles kernel cache survives pause
export AYAS_VENV=/home/venvs/ayas
export AYAS_REPO=/home/ayas
export AYAS_DATA=/home/data
export PYTHONUNBUFFERED=1
export PATH="$AYAS_VENV/bin:$PATH"
