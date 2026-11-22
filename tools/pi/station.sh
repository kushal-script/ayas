#!/bin/bash
# Start or restart the TriNetra station UI on the Pi (twin bursts as the camera), bound to localhost.
cd /home/rp5/trinetra || exit 1
if [ -f runs/station.pid ] && kill -0 "$(cat runs/station.pid)" 2>/dev/null; then kill "$(cat runs/station.pid)"; sleep 2; fi
. .venv/bin/activate
mkdir -p runs
nohup trinetra-app run --source "${SOURCE:-twin:data/twin_demo/bursts}" --detector "${DETECTOR:-$( [ -f models/v8n_burst.hef ] && echo hailo:models/v8n_burst.hef || echo onnx:models/v8n_burst.onnx )}" \
  --advisor executorch --ui --host 127.0.0.1 --port 8080 --log-dir runs/station > runs/station.log 2>&1 < /dev/null &
echo $! > runs/station.pid
sleep 5
ss -ltn | grep -q ":8080 " && echo "station up, pid $(cat runs/station.pid)" || { echo "station failed"; tail -20 runs/station.log; }
