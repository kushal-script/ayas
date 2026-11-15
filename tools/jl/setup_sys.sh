#!/bin/sh
# System packages headless Blender 5.2 needs on a JarvisLabs container (instance copy: /home/setup_sys.sh).
# Container system installs are lost on pause, so every job runs this first (idempotent, ~20-40 s
# after a resume, ~1 s when the packages are already there).
set -eu
PKGS="libxi6 libxrender1 libxkbcommon0 libsm6 libice6 libgl1 libegl1 libxxf86vm1 libxfixes3 libx11-6 libxext6 libxkbcommon-x11-0 libwayland-client0 libdbus-1-3 xz-utils rsync"
missing=""
for p in $PKGS; do
  dpkg -s "$p" >/dev/null 2>&1 || missing="$missing $p"
done
if [ -n "$missing" ]; then
  echo "setup_sys: installing$missing"
  export DEBIAN_FRONTEND=noninteractive
  SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
  $SUDO apt-get update -qq >/dev/null
  # shellcheck disable=SC2086
  $SUDO apt-get install -y -qq --no-install-recommends $missing >/dev/null
fi
echo "setup_sys: ok"
