"""ayas-app: the on-device inspection station (docs/APP.md).

  ayas-app run --source twin:<dir of npz>|camera|files:<dir> --detector fake|onnx:<path>|hailo:<hef>|pt:<path>
                   --spec configs/spec_steel.yaml --advisor mock|executorch|executorch_py
                   [--ui [--auto SEC]] [--bench N] [--count N]
  ayas-app check      what this machine has (camera, GPIO, Hailo, onnxruntime, ExecuTorch, network)

Without --ui or --bench it inspects --count captures and prints each verdict, then the advice.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Sequence

from .. import rules
from . import repo_path
from .advisor import AdvisorService
from .bench import cpu_temp_c, run_bench
from .pipeline import Alerter, AppLog, InspectionPipeline, build_detector, load_app_config
from .sources import open_source
from .ui import Station, make_server, network_state, url_of


def _llm_cfg(cfg: dict, args) -> dict:
    llm = dict(cfg.get("llm") or {})
    py = dict(llm.get("py") or {})
    if args.advisor:
        llm["backend"] = args.advisor
    for flag, key in ((args.pte, "model_path"), (args.tokenizer, "tokenizer_path")):
        if flag:
            llm[key] = py[key] = str(Path(flag).expanduser().resolve())
    if args.runner:
        llm["runner"] = args.runner
    llm["py"] = py
    return llm


def _part(cfg: dict, args) -> dict:
    part = dict(cfg.get("part") or {})
    if args.material:
        part["material"] = args.material
    if args.thickness:
        part["thickness_mm"] = args.thickness
    if args.device_mode:
        part["device_mode"] = args.device_mode
    return part


def build(args, advise: str) -> tuple[InspectionPipeline, object, dict]:
    """(pipeline, source, config) from the command line."""
    cfg = load_app_config(args.config)
    spec = rules.load_spec(repo_path(args.spec)) if args.spec else cfg["_spec"]
    cfg["part"] = _part(cfg, args)
    detector, vision = build_detector(args.detector, float(args.conf if args.conf is not None
                                                           else (cfg.get("detector") or {}).get("conf", 0.25)))
    advisor = None
    if not args.no_advice:
        advisor = AdvisorService.from_config(_llm_cfg(cfg, args), spec)
        if advisor.degraded:
            print(f"WARNING: {advisor.degraded}; the template answers instead of Llama", file=sys.stderr)
    log = AppLog(repo_path(args.log_dir or cfg.get("log_dir", "data/app")), bool(cfg.get("save_images", True)))
    al = cfg.get("alert") or {}
    alerter = Alerter(al.get("buzzer_pin"), al.get("beep_ms", 120), al.get("beep_on") or ("REJECT",), echo=not args.quiet)
    loop = args.loop if args.loop is not None else (True if (args.ui or args.bench) else False)
    source = open_source(args.source, cfg["_capture"], cfg, loop=loop, fps=args.fps)
    pipe = InspectionPipeline(cfg["_capture"], cfg["_grid"], detector, spec, cfg["part"], advisor, log=log,
                              alerter=alerter, advise=advise, vision_name=vision,
                              image_px=int((cfg.get("ui") or {}).get("image_px", 480)))
    pipe.surface_cfg = dict(cfg.get("surface") or {})
    if pipe.surface_cfg.get("plans_dir"):
        pipe.surface_cfg["plans_dir"] = str(repo_path(pipe.surface_cfg["plans_dir"]))
    return pipe, source, cfg


def cmd_run(args) -> int:
    advise = "thread" if args.ui else "sync"
    pipe, source, cfg = build(args, advise)
    print(f"source {getattr(source, 'describe', lambda: args.source)()} | vision {pipe.vision_name} | "
          f"advisor {getattr(pipe.advisor, 'name', 'off')} | log {pipe.log.dir} | {network_state()['detail']}",
          flush=True)
    try:
        if args.bench:
            rep = run_bench(pipe, source, int(args.bench), warmup=int(args.warmup), out_dir=repo_path(args.reports),
                            echo=not args.quiet,
                            config={"source": args.source, "detector": args.detector,
                                    "advisor": getattr(pipe.advisor, "name", None), "spec": args.spec or cfg.get("spec")})
            v = rep["vision"]["total"]
            print(f"vision p50 {v['p50']} ms, p95 {v['p95']} ms, {rep['fps']['vision']} FPS; peak RSS "
                  f"{rep['memory_mb']['peak_rss']} MB; wrote {rep['json']} and {rep['md']}")
            return 0
        if args.ui:
            station = Station(pipe, source, auto_s=args.auto)
            ui = cfg.get("ui") or {}
            srv = make_server(station, args.host or ui.get("host", "127.0.0.1"), args.port or ui.get("port", 8080))
            print(f"station UI at {url_of(srv)} (Ctrl+C to stop)", flush=True)
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                srv.server_close()
                station.close()
            return 0
        n = args.count if args.count is not None else (len(source) if hasattr(source, "__len__") else 1)
        done = 0
        while done < n:
            item = source.next()
            if item is None:
                break
            insp = pipe.inspect(*item)
            done += 1
            if args.json:
                print(json.dumps(insp.record, ensure_ascii=False))
            elif insp.advice and not args.quiet:
                print("  " + insp.advice[-1].text.replace("\n", "\n  "), flush=True)
        print(f"{done} inspection(s) logged to {pipe.log.jsonl}")
        return 0
    finally:
        pipe.close()
        source.close()


def cmd_check(args) -> int:
    """What this machine offers the app (nothing is started)."""
    def has(mod: str) -> str:
        try:
            return "yes" if importlib.util.find_spec(mod) is not None else "no"
        except (ImportError, ValueError):
            return "no"

    cfg = load_app_config(args.config)
    llm = cfg.get("llm") or {}
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"network: {network_state()['detail']}")
    print(f"cpu temp: {cpu_temp_c()} C")
    for mod in ("picamera2", "gpiozero", "hailo_platform", "onnxruntime", "executorch", "ultralytics"):
        print(f"module {mod}: {has(mod)}")
    for key in ("runner", "model_path", "tokenizer_path"):
        if llm.get(key):
            p = Path(llm[key]) if Path(llm[key]).is_absolute() else repo_path(llm[key])
            print(f"llm {key}: {p} ({'found' if p.exists() else 'MISSING'})")
    print(f"capture: {cfg.get('capture_config')} grid: {cfg['_grid']} spec: {cfg.get('spec')}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ayas-app", description="Ayas on-device steel surface inspection station")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="inspect from a source: console, UI (--ui) or benchmark (--bench N)")
    r.add_argument("--source", required=True, help="twin:<dir of out/bursts npz> | camera | files:<dir of images>")
    r.add_argument("--config", default="configs/app.yaml")
    r.add_argument("--detector", default="fake", help="fake (ground-truth replay) | onnx:<path> | hailo:<hef> | pt:<path>")
    r.add_argument("--conf", type=float, default=None, help="detector score threshold (default app.yaml detector.conf)")
    r.add_argument("--spec", default=None, help="severity spec (default app.yaml spec: configs/spec_steel.yaml)")
    r.add_argument("--advisor", choices=("mock", "executorch", "executorch_py"), default=None,
                   help="Llama backend (default app.yaml llm.backend)")
    r.add_argument("--pte", help="Llama .pte (overrides app.yaml llm.model_path)")
    r.add_argument("--tokenizer", help="tokenizer.model / tokenizer.json (overrides llm.tokenizer_path)")
    r.add_argument("--runner", help="llama_main command (overrides llm.runner)")
    r.add_argument("--no-advice", action="store_true", help="no advisor at all (vision + rules + log only)")
    r.add_argument("--ui", action="store_true", help="serve the station page")
    r.add_argument("--host", default=None)
    r.add_argument("--port", type=int, default=None)
    r.add_argument("--auto", type=float, default=None, help="with --ui: inspect every AUTO seconds (demo loop)")
    r.add_argument("--bench", type=int, default=None, metavar="N", help="benchmark N inspections, write reports/bench_*")
    r.add_argument("--warmup", type=int, default=1)
    r.add_argument("--reports", default="reports")
    r.add_argument("--count", type=int, default=None, help="console mode: number of inspections (default: every file once)")
    r.add_argument("--fps", type=float, default=None, help="pace the twin/files source (captures per second)")
    r.add_argument("--loop", dest="loop", action="store_true", default=None)
    r.add_argument("--no-loop", dest="loop", action="store_false")
    r.add_argument("--log-dir", default=None)
    r.add_argument("--material", default=None, help="part material when the source does not say")
    r.add_argument("--thickness", type=float, default=None, help="part thickness mm when the source does not say")
    r.add_argument("--device-mode", choices=("open", "hood"), default=None)
    r.add_argument("--json", action="store_true", help="console mode: print each record as JSON")
    r.add_argument("--quiet", action="store_true")
    c = sub.add_parser("check", help="list what this machine has for the app")
    c.add_argument("--config", default="configs/app.yaml")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return {"run": cmd_run, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
