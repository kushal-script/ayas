#!/bin/sh
# One-off (idempotent) instance setup, everything under /home (survives pause):
#   Blender 5.2.3 LTS (official tarball, sha256-checked) -> /home/blender
#   venv with the project installed (pip install -e ".[twin,dev]") -> /home/venvs/trinetra
# Run after the code is synced to /home/trinetra:
#   jl run --on <id> --json --yes -- sh -lc 'sh /home/trinetra_jobs/bootstrap.sh'
set -eu
sh /home/setup_sys.sh
. /home/trinetra_jobs/env.sh

BV=5.2.3
TARBALL=blender-$BV-linux-x64.tar.xz
URL=https://download.blender.org/release/Blender5.2
SHA=78486c018b4443efe5e75ab54c290ed8c15abafd249b862c2c9ce2530a918eec   # from $URL/blender-$BV.sha256
mkdir -p /home/blender "$XDG_CACHE_HOME"
if [ ! -x "$TRINETRA_BLENDER" ]; then
  cd /home/blender
  [ -f "$TARBALL" ] || curl -fsSL -o "$TARBALL.part" "$URL/$TARBALL"
  [ -f "$TARBALL" ] || mv "$TARBALL.part" "$TARBALL"
  curl -fsSL -o blender-$BV.sha256 "$URL/blender-$BV.sha256"
  official=$(grep " $TARBALL\$" blender-$BV.sha256 | cut -d' ' -f1)
  got=$(sha256sum "$TARBALL" | cut -d' ' -f1)
  echo "sha256 official=$official pinned=$SHA got=$got"
  [ "$got" = "$official" ] && [ "$got" = "$SHA" ] || { echo "CHECKSUM MISMATCH"; exit 3; }
  tar -xJf "$TARBALL"
  rm -f "$TARBALL"
fi
"$TRINETRA_BLENDER" -b --factory-startup --version 2>/dev/null | head -3

# Python 3.11 (the Mac's version; the pytorch template only has 3.10) from uv, kept under /home
if [ ! -x "$TRINETRA_VENV/bin/python" ]; then
  [ -x /home/.uvtool/bin/uv ] || { python3 -m venv /home/.uvtool && /home/.uvtool/bin/pip install -q uv; }
  export UV_PYTHON_INSTALL_DIR=/home/.uv/python UV_CACHE_DIR=/home/.cache/uv
  /home/.uvtool/bin/uv python install 3.11
  /home/.uvtool/bin/uv venv --python 3.11 --seed "$TRINETRA_VENV"
fi
cd "$TRINETRA_REPO"
# same package versions as the Mac .venv (tools/jl/constraints-mac.txt = its pip freeze)
"$TRINETRA_VENV/bin/python" -m pip install -q -c tools/jl/constraints-mac.txt -e ".[twin,dev]"
"$TRINETRA_VENV/bin/python" -c 'import sys, trinetra, OpenEXR, cv2, numpy; print("venv ok", sys.version.split()[0], numpy.__version__, cv2.__version__)'
