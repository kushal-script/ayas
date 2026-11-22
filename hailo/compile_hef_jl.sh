#!/usr/bin/env bash
# Compile the TriNetra YOLOv8n (6 steel classes) to a Hailo-10H HEF on a JarvisLabs GPU box.
# Same steps as hailo/compile_hef_colab.ipynb: DFC 5.1.0 + Model Zoo v5.1.0 (the Pi runs HailoRT 5.1.1).
# Inputs in $JOB (default /home/hailo_job): hailo_dataflow_compiler-5.1.0-*.whl, best.onnx, calib_640.zip
# Outputs in $JOB/out: the HEF, its <stem>.json (class names), HARs, compile.log, compile_info.json.
set -euo pipefail
export OMP_NUM_THREADS=8   # onnxruntime thread-affinity spam on containers
JOB=${JOB:-/home/hailo_job}; OUT=$JOB/out; mkdir -p "$OUT"; cd "$JOB"
echo "== $(date -Is) system"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y graphviz graphviz-dev git unzip >/dev/null
python3 -m pip -q install uv
SYSPY=$(command -v python3)
[ -d "$JOB/venv" ] || "$SYSPY" -m uv venv -q -p 3.10 "$JOB/venv"
UVPIP() { "$SYSPY" -m uv pip install -q --python "$JOB/venv/bin/python" "$@"; }
. "$JOB/venv/bin/activate"
python -V
echo "== $(date -Is) install DFC 5.1.0 (+ TensorFlow 2.18 CUDA libs)"
UVPIP "$(ls $JOB/hailo_dataflow_compiler-5.1.0*.whl)" "tensorflow[and-cuda]==2.18.0" pip "setuptools<70"   # the SDK imports pkg_resources
python - <<'PY'
from importlib.metadata import version
v = version("hailo-dataflow-compiler"); print("DFC", v); assert v.startswith("5.1.0"), v
import tensorflow as tf; print("TF", tf.__version__, "GPUs", tf.config.list_physical_devices("GPU"))
PY
echo "== $(date -Is) Model Zoo v5.1.0"
[ -d "$JOB/hailo_model_zoo" ] || git clone -q -b v5.1.0 --depth 1 https://github.com/hailo-ai/hailo_model_zoo.git "$JOB/hailo_model_zoo"
python -m pip install -q --no-build-isolation -e "$JOB/hailo_model_zoo"   # its setup.py checks for the DFC, so no isolated build
python -c "from importlib.metadata import version; v=version('hailo-dataflow-compiler'); assert v.startswith('5.1.0'), v; print('DFC still', v)"
rm -rf "$JOB/calib_640" && unzip -q "$JOB/calib_640.zip" -d "$JOB"
echo "== $(date -Is) checks and config copies"
python - <<'PY'
import ast, glob, os, pathlib, re, urllib.request, zipfile, json
import onnx
JOB = os.environ.get("JOB", "/home/hailo_job")
CKPT = f"{JOB}/best.onnx"; CALIB = f"{JOB}/calib_640"
n = len(glob.glob(f"{CALIB}/*.png")); print(n, "calibration tiles"); assert n >= 1024
m = onnx.load(CKPT)
dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]; assert dims == [1, 3, 640, 640], dims
od = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]; assert od[:2] == [1, 10], od
meta = {p.key: p.value for p in m.metadata_props}
names = list(ast.literal_eval(meta["names"]).values()) if "names" in meta else None
print("classes", names); assert names in (None, ["scratch", "dent", "pit", "corrosion", "crack", "stain"])
END = [f"/model.22/cv{k}.{i}/cv{k}.{i}.2/Conv" for i in range(3) for k in (2, 3)]
missing = [e for e in END if e not in {nd.name for nd in m.graph.node}]; assert not missing, missing
CFG = pathlib.Path(f"{JOB}/hailo_model_zoo/hailo_model_zoo/cfg")
net = CFG / "networks" / "yolov8n.yaml"; text = net.read_text()
alls_name = re.search(r"alls_script:\s*(\S+)", text).group(1)
alls = sorted(CFG.glob(f"alls/**/{alls_name}")); assert alls, alls_name
new_yaml = net.with_name("trinetra_yolov8n.yaml")
new_yaml.write_text(re.sub(r"(alls_script:\s*)\S+", r"\g<1>trinetra_yolov8n.alls", text))
for a in alls:
    lines = [ln for ln in a.read_text().splitlines() if not ln.startswith("model_optimization_config(calibration")]
    lines.append("model_optimization_config(calibration, calibset_size=1024)")
    new = a.with_name("trinetra_yolov8n.alls"); new.write_text("\n".join(lines) + "\n")
    nms = re.search(r'nms_postprocess\("([^"]+)"', new.read_text())
    if nms:
        target = (a.parent / nms.group(1)).resolve()
        print(a.parent.name, "NMS config", target, target.exists())
        url = re.search(r"url:\s*(\S+)", text)
        if not target.exists() and url:
            urllib.request.urlretrieve(url.group(1), f"{JOB}/yolov8n_mz.zip")
            with zipfile.ZipFile(f"{JOB}/yolov8n_mz.zip") as z:
                js = [x for x in z.namelist() if x.endswith(target.name)]; assert js
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(z.read(js[0]))
            print("fetched", target)
print(new_yaml.read_text()); print(new.read_text())
PY
echo "== $(date -Is) compile (hailo10h, 6 classes, no --performance)"
mkdir -p "$JOB/custom" && cd "$JOB/custom"
hailomz compile --ckpt "$JOB/best.onnx" --calib-path "$JOB/calib_640" \
  --yaml "$JOB/hailo_model_zoo/hailo_model_zoo/cfg/networks/trinetra_yolov8n.yaml" \
  --classes 6 --hw-arch hailo10h \
  --start-node-names images \
  --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv /model.22/cv2.1/cv2.1.2/Conv \
                   /model.22/cv3.1/cv3.1.2/Conv /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv \
  2>&1 | grep -v pthread_setaffinity | tee "$JOB/custom/compile.log"
# Model Zoo v5.1.0's yolov8 yaml has no parser end nodes, and the auto-guess cuts at /model.22/dfl/Reshape,
# which the yolov8 NMS post-process rejects: cut at the six head convs, as older Model Zoo releases did.
echo "== $(date -Is) collect"
python - <<'PY'
import glob, json, os, platform, re, shutil, subprocess
from importlib.metadata import version
JOB = os.environ.get("JOB", "/home/hailo_job"); OUT = f"{JOB}/out"
log = open(f"{JOB}/custom/compile.log").read()
snr = [(a, float(b)) for a, b in re.findall(r"(\S+)\s+SNR:?\s*(-?[\d.]+)\s*dB", log)]
hefs = glob.glob(f"{JOB}/custom/*.hef"); assert hefs, "no HEF: read compile.log"
for f in hefs + glob.glob(f"{JOB}/custom/*.har") + [f"{JOB}/custom/compile.log"]:
    shutil.copy(f, OUT)
for a in glob.glob(f"{JOB}/hailo_model_zoo/hailo_model_zoo/cfg/**/trinetra_yolov8n.*", recursive=True):
    shutil.copy(a, f"{OUT}/{os.path.basename(os.path.dirname(a))}_{os.path.basename(a)}")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
mz = subprocess.run(["git", "-C", f"{JOB}/hailo_model_zoo", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
classes = ["scratch", "dent", "pit", "corrosion", "crack", "stain"]
info = {"dfc": version("hailo-dataflow-compiler"), "model_zoo": "v5.1.0", "model_zoo_commit": mz, "python": platform.python_version(),
        "gpu": gpu, "classes": classes, "n_classes": 6, "calib_tiles": len(glob.glob(f"{JOB}/calib_640/*.png")),
        "snr_db": snr, "low_snr": [s for s in snr if s[1] < 10.0], "ckpt": "models/v8n_burst.onnx", "hw_arch": "hailo10h", "performance_flag": False}
json.dump(info, open(f"{OUT}/compile_info.json", "w"), indent=2)
for h in hefs:
    json.dump({"classes": classes, "imgsz": 640, "source": "models/v8n_burst.onnx"},
              open(f"{OUT}/{os.path.splitext(os.path.basename(h))[0]}.json", "w"), indent=2)
print(json.dumps(info, indent=1)); print(sorted(os.listdir(OUT)))
PY
echo "== $(date -Is) done"
