"""Benchmark N inspections from a source (the organisers' system and Llama metrics).

Per inspection, in order: source.next() (capture or file read) -> pipeline.inspect (advisor off)
-> advisor.explain on the record (sequential, so Llama never overlaps the vision timings).
Reported:
- vision: per-stage p50 / p95 / mean / max ms (record timing_ms: capture, process, rectify, tile,
  detect, merge, measure, rules, report, image, total) and FPS (1000 / mean total; the loop FPS
  includes the source and the logging);
- camera-to-result latency (timing_ms.total) and camera-to-advice (total + the advisor call);
- memory: peak RSS of this process and of child processes (the llama_main runner);
- CPU temperature (/sys/class/thermal, when present) at the start, the maximum and the end;
- Llama: TTFT (model clock and wall clock), decode and prefill tokens/s, total, who answered.
Writes reports/bench_<host>_<YYYYmmdd-HHMMSS>.json and a .md table next to it.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pipeline import InspectionPipeline

STAGES = ("capture", "process", "rectify", "tile", "detect", "merge", "measure", "rules", "report", "image", "total")
LLM_KEYS = ("ttft_ms", "wall_ttft_ms", "decode_tokens_per_s", "prefill_tokens_per_s", "total_ms", "call_ms",
            "prompt_tokens", "generated_tokens", "load_ms")


def _rss_mb(who: int) -> float:
    """Peak RSS in MB (ru_maxrss is KB on Linux, bytes on macOS)."""
    r = resource.getrusage(who).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


def cpu_temp_c() -> float | None:
    """The hottest /sys/class/thermal zone in deg C (the Pi's cpu-thermal), None elsewhere."""
    temps = []
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(p.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return round(max(temps), 1) if temps else None


def stats(values: Sequence[float]) -> dict[str, float | int | None]:
    v = np.asarray([x for x in values if x is not None], float)
    if not v.size:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "max": None}
    return {"n": int(v.size), "p50": round(float(np.percentile(v, 50)), 2), "p95": round(float(np.percentile(v, 95)), 2),
            "mean": round(float(v.mean()), 2), "max": round(float(v.max()), 2)}


def platform_info() -> dict[str, Any]:
    info = {"host": platform.node().split(".")[0] or "host", "system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(), "cpus": os.cpu_count()}
    model = Path("/proc/device-tree/model")
    if model.exists():
        try:
            info["model"] = model.read_bytes().rstrip(b"\x00").decode(errors="replace")
        except OSError:
            pass
    return info


def run_bench(pipeline: InspectionPipeline, source, n: int, *, advisor=None, warmup: int = 1,
              out_dir: str | Path = "reports", label: str | None = None, config: Mapping[str, Any] | None = None,
              echo: bool = True) -> dict[str, Any]:
    """Benchmark n inspections (after `warmup` untimed ones); returns the report dict with
    'json' and 'md' paths. advisor defaults to pipeline.advisor (None: vision only)."""
    advisor = advisor if advisor is not None else pipeline.advisor
    mode, pipeline.advise = pipeline.advise, "off"
    rows: list[dict[str, Any]] = []
    temps = [cpu_temp_c()]
    warm: list[dict[str, Any]] = []
    t_loop = None
    try:
        for i in range(int(warmup) + int(n)):
            if i == int(warmup):
                t_loop = time.perf_counter()
            t0 = time.perf_counter()
            item = source.next()
            if item is None:
                break
            insp = pipeline.inspect(*item)
            row = {"part_id": insp.record["part_id"], "verdict": insp.record["verdict"],
                   "gate": insp.record["capture"]["gate"], "defects": len(insp.record["defects"]),
                   "condition": (insp.record.get("source") or {}).get("condition"),
                   "timing_ms": dict(insp.record["timing_ms"]), "loop_ms": None, "llm": None}
            row["loop_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            if advisor is not None:
                res = advisor.explain(insp.record, key=insp.key)
                pipeline.record_advice(insp, res)
                row["llm"] = {"backend": res.backend, "model": res.model, "problems": len(res.problems),
                              **{k: res.stats.get(k) for k in LLM_KEYS}}
            temps.append(cpu_temp_c())
            (warm if i < int(warmup) else rows).append(row)
            if echo:
                tag = "warm-up" if i < int(warmup) else f"{len(rows)}/{n}"
                llm_ms = (row["llm"] or {}).get("call_ms")
                print(f"[{tag}] {row['part_id']}: {row['verdict']} vision {row['timing_ms'].get('total', 0):.0f} ms"
                      + (f", advisor {llm_ms:.0f} ms" if llm_ms is not None else ""), flush=True)
    finally:
        pipeline.advise = mode
    wall = time.perf_counter() - t_loop if t_loop is not None and rows else None
    totals = [r["timing_ms"].get("total") for r in rows]
    mean_total = stats(totals)["mean"]
    llm_rows = [r["llm"] for r in rows if r["llm"]]
    rep = {
        "schema": "ayas.bench/1",
        "label": label,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform_info(),
        "config": dict(config or {}),
        "n": len(rows), "warmup": len(warm),
        "vision": {s: stats([r["timing_ms"].get(s) for r in rows]) for s in STAGES},
        "fps": {"vision": round(1000.0 / mean_total, 3) if mean_total else None,
                "loop": round(len(rows) / wall, 3) if wall else None},
        "latency_ms": {"camera_to_result": stats(totals),
                       "camera_to_advice": stats([r["timing_ms"].get("total", 0) + r["llm"]["call_ms"]
                                                  for r in rows if r["llm"] and r["llm"].get("call_ms") is not None])},
        "memory_mb": {"peak_rss": _rss_mb(resource.RUSAGE_SELF), "children_peak_rss": _rss_mb(resource.RUSAGE_CHILDREN)},
        "cpu_temp_c": {"start": temps[0], "max": max([t for t in temps if t is not None], default=None),
                       "end": temps[-1]},
        "llm": {"n": len(llm_rows),
                "backend": getattr(advisor, "name", None),
                "answered_by": dict(Counter(r["backend"] for r in llm_rows)),
                **{k: stats([r.get(k) for r in llm_rows]) for k in LLM_KEYS}} if advisor is not None else None,
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "gates": dict(Counter(r["gate"] for r in rows)),
        "cold_start": warm,
        "runs": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"bench_{rep['platform']['host']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    jp, mp = out / f"{stem}.json", out / f"{stem}.md"
    jp.write_text(json.dumps(rep, indent=1))
    mp.write_text(markdown(rep))
    rep["json"], rep["md"] = str(jp), str(mp)
    return rep


def _c(x, nd: int = 1) -> str:
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def markdown(rep: Mapping[str, Any]) -> str:
    """The benchmark as markdown tables (for the report)."""
    p, cfg = rep["platform"], rep.get("config") or {}
    lines = [f"# Ayas benchmark: {p['host']} ({p.get('model') or p['machine']}), {rep['date']}", "",
             f"{rep['n']} inspections after {rep['warmup']} warm-up; source `{cfg.get('source', '?')}`, "
             f"detector `{cfg.get('detector', '?')}`, advisor `{cfg.get('advisor', '?')}`. "
             f"Python {p['python']}, {p['cpus']} CPUs.", "",
             "## Vision pipeline (ms per inspection)", "", "| Stage | p50 | p95 | mean | max |", "|---|---|---|---|---|"]
    for s, v in rep["vision"].items():
        if v["n"]:
            lines.append(f"| {s} | {_c(v['p50'])} | {_c(v['p95'])} | {_c(v['mean'])} | {_c(v['max'])} |")
    lat, mem, t = rep["latency_ms"], rep["memory_mb"], rep["cpu_temp_c"]
    lines += ["", "## System", "", "| Metric | Value |", "|---|---|",
              f"| FPS (vision, 1000 / mean total) | {_c(rep['fps']['vision'], 2)} |",
              f"| FPS (loop incl. source + log) | {_c(rep['fps']['loop'], 2)} |",
              f"| Camera-to-result p50 / p95 ms | {_c(lat['camera_to_result']['p50'])} / {_c(lat['camera_to_result']['p95'])} |",
              f"| Camera-to-advice p50 / p95 ms | {_c(lat['camera_to_advice']['p50'])} / {_c(lat['camera_to_advice']['p95'])} |",
              f"| Peak RSS (app) MB | {_c(mem['peak_rss'])} |",
              f"| Peak RSS (child: llama runner) MB | {_c(mem['children_peak_rss'])} |",
              f"| CPU temp start / max / end C | {_c(t['start'])} / {_c(t['max'])} / {_c(t['end'])} |",
              f"| Verdicts | {', '.join(f'{k} {v}' for k, v in rep['verdicts'].items()) or '-'} |",
              f"| Capture gates | {', '.join(f'{k} {v}' for k, v in rep['gates'].items()) or '-'} |"]
    llm = rep.get("llm")
    if llm:
        lines += ["", f"## Llama advisor ({llm.get('backend')})", "",
                  f"Answered by: {', '.join(f'{k} {v}' for k, v in llm['answered_by'].items()) or '-'}.", "",
                  "| Metric | p50 | p95 | n |", "|---|---|---|---|"]
        names = {"ttft_ms": "TTFT ms (model clock)", "wall_ttft_ms": "First text ms (wall, incl. load)",
                 "decode_tokens_per_s": "Decode tokens/s", "prefill_tokens_per_s": "Prefill tokens/s",
                 "total_ms": "Generation ms (model clock)", "call_ms": "Advisor call ms (wall)",
                 "prompt_tokens": "Prompt tokens", "generated_tokens": "Generated tokens", "load_ms": "Model load ms"}
        for k, name in names.items():
            v = llm.get(k) or {}
            if v.get("n"):
                lines.append(f"| {name} | {_c(v['p50'])} | {_c(v['p95'])} | {v['n']} |")
        if not any((llm.get(k) or {}).get("n") for k in ("ttft_ms", "decode_tokens_per_s")):
            lines.append("")
            lines.append("No model token timings: the template (mock) answered, or the runner printed no stats line.")
    return "\n".join(lines) + "\n"
