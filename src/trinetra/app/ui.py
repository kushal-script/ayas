"""Operator station UI: a stdlib http.server page for the 5" 800 x 480 display (no external JS/CSS).

Routes:
- GET  /             the station page (polls /api/latest every second)
- GET  /api/latest   station state: verdict banner, defects, capture-quality line, advice (streamed
                     text while Llama answers), network state, busy flags
- GET  /api/record   the latest full record (trinetra.inspection/3.0, the JSON hand-off)
- GET  /api/image.jpg  the latest result image (rectified grid with boxes, or the refused frame)
- GET  /api/history  the last inspections from the SQLite log
- POST /api/inspect  run one inspection from the source (on the inspection thread)
- POST /api/ask      {"kind": "why" | "fix" | "shift"} or {"question": "..."} -> the advisor thread
- POST /api/ack      acknowledge the alert (logged)

Station owns the threads: one runs inspections, the pipeline's own advisor thread runs Llama
after each alert. The page shows OFFLINE when the device has no default route.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from .pipeline import COLOURS, Inspection, InspectionPipeline

ALERT_VERDICTS = ("REJECT", "HOLD_FOR_QA", "RETAKE")
VERDICT_TEXT = {"PASS": "PASS", "CLEAN_AND_REINSPECT": "CLEAN + REINSPECT", "REWORK": "REWORK",
                "HOLD_FOR_QA": "HOLD FOR QA", "REJECT": "REJECT", "RETAKE": "RETAKE"}


# ---- network state (offline proof) -----------------------------------------------------------

def _default_route_linux(route_text: str, ipv6_text: str = "") -> bool:
    """True if /proc/net/route (or ipv6_route) has an UP default route."""
    for line in route_text.splitlines()[1:]:
        f = line.split()
        if len(f) >= 4 and f[1] == "00000000" and int(f[3], 16) & 0x1:
            return True
    for line in ipv6_text.splitlines():
        f = line.split()
        if len(f) >= 10 and f[0] == "0" * 32 and f[1] == "00" and int(f[8], 16) & 0x1 and f[9] != "lo":
            return True
    return False


def network_state(_cache: dict[str, Any] = {}) -> dict[str, Any]:   # noqa: B006 (5 s cache)
    """{'offline': bool, 'detail': str}: offline = no default route (no way off the local network).

    Linux reads /proc/net/route and /proc/net/ipv6_route; macOS asks `route -n get default`."""
    now = time.monotonic()
    if _cache.get("t") and now - _cache["t"] < 5.0:
        return _cache["v"]
    try:
        if Path("/proc/net/route").exists():
            v6 = Path("/proc/net/ipv6_route")
            online = _default_route_linux(Path("/proc/net/route").read_text(), v6.read_text() if v6.exists() else "")
        elif platform.system() == "Darwin":
            online = subprocess.run(["route", "-n", "get", "default"], capture_output=True, timeout=2).returncode == 0
        else:
            online = None
    except (OSError, subprocess.SubprocessError, ValueError):
        online = None
    if online is None:
        v = {"offline": None, "detail": "network state unknown"}
    else:
        v = {"offline": not online, "detail": "OFFLINE: no default route" if not online else "NETWORK: default route present"}
    _cache.update(t=now, v=v)
    return v


# ---- station state -----------------------------------------------------------------------------

class Station:
    """The source, the pipeline and the UI state; thread-safe."""

    def __init__(self, pipeline: InspectionPipeline, source=None, auto_s: float | None = None):
        self.pipeline, self.source = pipeline, source
        self.lock = threading.RLock()
        self.seq = 0
        self.busy = False
        self.error: str | None = None
        self.acked = True
        self.advice: dict[str, Any] = {"status": "idle", "text": "", "kind": None, "question": None,
                                       "backend": None, "stats": {}, "for": None}
        self._shown: Any = None             # key of the inspection on screen
        self._active = False                # the running advisor job is about the part on screen (or shift)
        self._jobs: list[str] = []
        self.surface: dict[str, Any] = {"status": "idle", "for": None}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.auto_s = auto_s
        pipeline.listeners.append(self._on_event)
        w = int((getattr(pipeline, "surface_cfg", {}) or {}).get("workers", 1))
        if w > 1:                                   # warm the 3D worker pool (imports torch) in the background
            def _warm() -> None:
                try:
                    from . import surface3d
                    surface3d.warm_up(w)
                except Exception as e:              # 3D then reports why when asked
                    print(f"3D workers not started: {e}", file=sys.stderr)
            threading.Thread(target=_warm, name="surface3d-warm", daemon=True).start()
        self._thread = threading.Thread(target=self._loop, name="inspect", daemon=True)
        self._thread.start()

    # -- events from the pipeline (any thread) --
    def _on_event(self, event: str, insp: Inspection | None, **kw: Any) -> None:
        """Station state from pipeline events. Advice events of a job about another part than the
        one on screen (an answer still streaming when the next part was inspected) are dropped,
        so the panel never shows one part's advice under another part's verdict."""
        with self.lock:
            key = None if insp is None else insp.key
            if event == "inspection":
                self._shown, self._active = key, False
                self.surface = {"status": "idle", "for": key}
                self.acked = insp.record["verdict"] not in ALERT_VERDICTS
                self.advice = {"status": "waiting" if self.pipeline.advisor and self.pipeline.advise != "off" else "idle",
                               "text": "", "kind": "explain", "question": None, "backend": None, "stats": {},
                               "for": insp.key}
            elif event == "advice_start":
                self._active = insp is None or key == self._shown
                if not self._active:
                    return
                self.advice = {"status": "thinking", "text": "", "kind": kw.get("kind"), "question": kw.get("question"),
                               "backend": None, "stats": {}, "for": key}
            elif not self._active or (insp is not None and key != self._shown):
                return
            elif event == "advice_text":
                self.advice["status"], self.advice["text"] = "streaming", kw.get("text", "")
            elif event == "advice_done":
                r = kw["result"]
                self.advice.update(status="error" if r.error else "done", text=r.text, backend=r.backend,
                                   stats={k: v for k, v in r.stats.items() if k != "summary"}, problems=r.problems)
            self.seq += 1

    # -- requests from the UI --
    def request(self, job: str) -> bool:
        """Queue 'inspect' (False when the source is missing or exhausted, or one is already queued)."""
        with self.lock:
            if job == "inspect" and (self.source is None or "inspect" in self._jobs or self.busy):
                return False
            self._jobs.append(job)
            self._wake.set()
            return True

    def ask(self, kind: str | None = None, question: str | None = None) -> bool:
        p = self.pipeline
        if p.advisor is None:
            return False
        if kind == "shift":
            p.submit(p.shift)
            return True
        insp = p.last
        if insp is None:
            return False
        q = question or kind
        if not q:
            return False
        p.submit(p.ask, insp, q)
        return True

    def surface_request(self) -> bool:
        """Start the 3D measurement of the part on screen (one at a time, in the background)."""
        insp = self.pipeline.last
        with self.lock:
            if insp is None or self.surface.get("status") == "measuring":
                return False
            if insp.surface is None:
                self.surface = {"status": "unavailable", "for": insp.key,
                                "why": "3D needs an accepted burst capture (not a refused or single-image one)"}
                self.seq += 1
                return False
            kinds = ("dent", "pit", "scratch")
            self.surface = {"status": "measuring", "for": insp.key, "started": time.time(), "items": [],
                            "total": sum(1 for d in insp.record.get("defects", []) if d.get("type") in kinds)}
            self.seq += 1
        threading.Thread(target=self._measure_surface, args=(insp,), name="surface3d", daemon=True).start()
        return True

    def _measure_surface(self, insp: Inspection) -> None:
        from . import surface3d as S3
        cfg = dict(getattr(self.pipeline, "surface_cfg", {}) or {})
        rec = insp.record
        def on_item(it: dict) -> None:            # stream each defect to the viewer as it finishes
            with self.lock:
                if self.surface.get("for") == insp.key and self.surface.get("status") == "measuring":
                    self.surface["items"] = [*self.surface.get("items", []), it]
                    self.seq += 1

        deciding = next((d["id"] for d in rec.get("defects", []) if d.get("disposition") == rec.get("verdict")
                         and d.get("type") in S3.KINDS), None)
        try:
            res = S3.measure(insp.surface, rec.get("defects", []), rec.get("material") or "", rec.get("thickness_mm"),
                             method=cfg.get("method", "pinn"), n_seeds=int(cfg.get("n_seeds", 1)),
                             threads=cfg.get("threads"), plans_dir=cfg.get("plans_dir"), fast=bool(cfg.get("fast", True)),
                             workers=int(cfg.get("workers", 1)), on_item=on_item, first=deciding)
            out = {"status": "done" if res["available"] else "unavailable", "for": insp.key, **res}
        except ModuleNotFoundError as e:
            out = {"status": "unavailable", "for": insp.key, "why": f"3D needs {e.name} on this device (pip install torch)"}
        except Exception as e:   # shown on the page; the station stays up
            out = {"status": "error", "for": insp.key, "why": f"{type(e).__name__}: {e}"}
            print(f"3D measurement failed: {e}", file=sys.stderr)
        with self.lock:
            if self.surface.get("for") == insp.key:      # still the part on screen
                self.surface = out
                self.seq += 1

    def surface_state(self) -> dict[str, Any]:
        with self.lock:
            out = dict(self.surface)
        if out.get("status") == "measuring":
            out["elapsed_s"] = round(time.time() - out.get("started", time.time()), 0)
        return out

    def acknowledge(self) -> bool:
        insp = self.pipeline.last
        with self.lock:
            self.acked = True
            self.seq += 1
        if insp is not None:
            self.pipeline.acknowledge(insp)
        return insp is not None

    def _loop(self) -> None:
        next_auto = time.monotonic() + (self.auto_s or 0)
        while not self._stop.is_set():
            timeout = None if not self.auto_s else max(0.0, next_auto - time.monotonic())
            self._wake.wait(timeout)
            self._wake.clear()
            if self._stop.is_set():
                return
            with self.lock:
                job = self._jobs.pop(0) if self._jobs else None
            if job is None and self.auto_s and time.monotonic() >= next_auto:
                job = "inspect"
            if job == "inspect":
                next_auto = time.monotonic() + (self.auto_s or 0)
                self._inspect_once()
            with self.lock:
                if self._jobs:
                    self._wake.set()

    def _inspect_once(self) -> None:
        with self.lock:
            self.busy, self.error = True, None
            self.seq += 1
        try:
            item = self.source.next()
            if item is None:
                raise RuntimeError("the source has no more captures")
            self.pipeline.inspect(*item)
        except Exception as e:   # shown on the page; the station stays up
            with self.lock:
                self.error = f"{type(e).__name__}: {e}"
            print(f"inspection failed: {e}", file=sys.stderr)
        finally:
            with self.lock:
                self.busy = False
                self.seq += 1

    def state(self) -> dict[str, Any]:
        p = self.pipeline
        with p.lock:
            insp = p.last
            rec = json.loads(json.dumps(insp.record)) if insp is not None else None
        with self.lock:
            out = {"seq": self.seq, "busy": self.busy, "error": self.error, "acked": self.acked,
                   "advice": dict(self.advice), "network": network_state(), "count": p.count,
                   "source": getattr(self.source, "describe", lambda: None)() if self.source else None,
                   "vision": p.vision_name, "advisor": getattr(p.advisor, "name", None),
                   "advisor_degraded": getattr(p.advisor, "degraded", None), "inspection": None,
                   "surface": {k: v for k, v in self.surface.items() if k in ("status", "for", "why")}}
        if rec is not None:
            out["inspection"] = summarise(rec, insp.key)
        return out

    def image(self) -> bytes | None:
        insp = self.pipeline.last
        return insp.jpeg if insp is not None else None

    def record(self) -> dict[str, Any] | None:
        with self.pipeline.lock:
            insp = self.pipeline.last
            return json.loads(json.dumps(insp.record)) if insp is not None else None

    def history(self, n: int = 20) -> list[dict[str, Any]]:
        log = self.pipeline.log
        if log is None:
            return []
        return [{"id": r[0], "part_id": r[1], "ts": r[2], "verdict": r[3], "defects": r[4]} for r in log.store.recent(n)]

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=5.0)


def _fmt(x, nd: int = 2, unit: str = "") -> str:
    return "?" if x is None else f"{x:.{nd}f}{unit}"


def capture_line(cap: Mapping[str, Any]) -> str:
    """'Gate PASS | r 0.05 | ~220 lux | motion 0.3 px | sat 0.0 %' for the capture block."""
    if cap.get("mode") == "naive":
        return f"Gate {cap.get('gate')} | single image (no capture check)"
    parts = [f"Gate {cap.get('gate')}", f"r {_fmt(cap.get('r_ambient'), 2)}"]
    lux = cap.get("lux_est")
    parts.append(f"~{lux:.0f} lux" if lux is not None else "lux ?")
    parts.append(f"motion {_fmt(cap.get('motion_px'), 1, ' px')}")
    parts.append(f"sat {_fmt(cap.get('sat_pct'), 1, ' %')}")
    if cap.get("blur") is not None:
        parts.append(f"blur {_fmt(cap.get('blur'), 2)}")
    parts.append(cap.get("device_mode") or "open")
    return " | ".join(parts)


def _size_short(size: Mapping[str, Any]) -> str:
    names = (("length_mm", "L {} mm"), ("width_mm", "W {} mm"), ("diameter_mm", "D {} mm"), ("area_mm2", "{} mm2"),
             ("count", "n {}"))
    return ", ".join(f.format(round(size[k], 2) if isinstance(size[k], float) else size[k]) for k, f in names if k in size)


def summarise(rec: Mapping[str, Any], key: Any = None) -> dict[str, Any]:
    """What the page shows of a record."""
    v = rec["verdict"]
    cap = rec.get("capture") or {}
    return {
        "id": key, "part_id": rec["part_id"], "timestamp": rec.get("timestamp"), "verdict": v,
        "verdict_text": VERDICT_TEXT.get(v, v), "colour": "#%02x%02x%02x" % COLOURS.get(v, (90, 90, 90)),
        "instruction": rec.get("verdict_instruction"), "material": rec.get("material_name") or rec.get("material"),
        "thickness_mm": rec.get("thickness_mm"), "capture_line": capture_line(cap),
        "gate": cap.get("gate"), "gate_message": cap.get("gate_message") or "",
        "defects": [{"id": d["id"], "type": d["type"], "conf": round(d["confidence"], 2), "severity": d["severity"],
                     "disposition": d["disposition"], "x": d["location_mm"]["x"], "y": d["location_mm"]["y"],
                     "size": _size_short(d.get("size") or {}), "edge": d.get("near_edge_mm"),
                     "status": d.get("status"), "colour": "#%02x%02x%02x" % COLOURS.get(d["disposition"], (200, 200, 200))}
                    for d in rec.get("defects", [])],
        "timing_ms": rec.get("timing_ms") or {}, "source": rec.get("source") or {},
        "advice": rec.get("advice"), "backend": rec.get("backend") or {},
    }


# ---- HTTP -----------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "TriNetra/1"
    station: Station = None   # set by make_server

    def log_message(self, fmt: str, *args: Any) -> None:   # quiet: the console is for alerts
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:   # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0]
        st = self.station
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/latest":
            self._json(st.state())
        elif path == "/api/record":
            self._json(st.record())
        elif path == "/api/history":
            self._json(st.history())
        elif path == "/api/surface":
            self._json(st.surface_state())
        elif path == "/api/image.jpg":
            img = st.image()
            if img:
                self._send(200, img, "image/jpeg")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"no image yet", "text/plain")
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:   # noqa: N802
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            body = {}
        st = self.station
        if path == "/api/inspect":
            self._json({"queued": st.request("inspect")})
        elif path == "/api/ask":
            self._json({"queued": st.ask(body.get("kind"), body.get("question"))})
        elif path == "/api/ack":
            self._json({"acknowledged": st.acknowledge()})
        elif path == "/api/surface":
            self._json({"started": st.surface_request()})
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")


def make_server(station: Station, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """The HTTP server (port 0 = a free port: read server.server_address[1])."""
    handler = type("StationHandler", (Handler,), {"station": station})
    srv = ThreadingHTTPServer((host, int(port)), handler)
    srv.daemon_threads = True
    return srv


def serve_in_thread(station: Station, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    srv = make_server(station, host, port)
    t = threading.Thread(target=srv.serve_forever, name="ui", daemon=True)
    t.start()
    return srv, t


def url_of(srv: ThreadingHTTPServer) -> str:
    host, port = srv.server_address[:2]
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TriNetra station</title>
<style>
:root { --bg:#101214; --panel:#1b1e22; --line:#2c3138; --text:#eceff1; --muted:#9aa4ad; --banner:#3a3f45; --btn:#2b3f57; }
* { box-sizing:border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
  font:14px/1.3 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; overflow:hidden; }
#app { display:grid; grid-template-rows:64px 1fr 60px; height:100vh; max-height:100vh; }
#top { display:flex; align-items:center; gap:10px; padding:0 10px; background:var(--banner); transition:background .2s; }
#top.alert { animation:pulse 1s infinite; }
@keyframes pulse { 50% { filter:brightness(1.35); } }
#verdict { font-size:32px; font-weight:800; letter-spacing:.5px; white-space:nowrap; }
#instr { flex:1; font-size:13px; line-height:1.2; max-height:56px; overflow:hidden; }
#meta { text-align:right; font-size:12px; white-space:nowrap; }
#main { display:grid; grid-template-columns:auto 1fr; gap:8px; padding:6px 8px; min-height:0; }
#imgbox { height:100%; aspect-ratio:1/1; max-width:46vw; background:#000; border:1px solid var(--line);
  display:flex; align-items:center; justify-content:center; overflow:hidden; }
#img { max-width:100%; max-height:100%; object-fit:contain; }
#right { display:flex; flex-direction:column; gap:5px; min-width:0; min-height:0; }
#cap { font-size:11px; color:var(--muted); line-height:1.25; max-height:28px; overflow:hidden; }
#gatemsg { font-size:13px; color:#ffcc66; min-height:0; }
#tbl { max-height:34%; overflow:auto; border:1px solid var(--line); background:var(--panel); }
table { border-collapse:collapse; width:100%; font-size:12px; }
th, td { padding:2px 4px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }
th { color:var(--muted); font-weight:600; position:sticky; top:0; background:var(--panel); }
td.sw { width:6px; padding:0; }
#adv { flex:1; min-height:0; overflow:auto; background:var(--panel); border:1px solid var(--line); padding:5px 7px;
  white-space:pre-wrap; font-size:13px; -webkit-overflow-scrolling:touch; }
#adv .llama { border-left:4px solid #8ab4f8; background:#1d2633; padding:4px 7px; margin:0 0 6px; border-radius:3px; }
#adv .llama b, #adv .rules b { display:block; font-size:10px; letter-spacing:.4px; color:#8ab4f8; text-transform:uppercase; }
#adv .rules b { color:var(--muted); }
#adv .rules { margin:0; }
#advhead { font-size:11px; color:var(--muted); display:flex; justify-content:space-between; gap:6px; }
#bar { display:grid; grid-template-columns:1.15fr 1fr 1fr 0.8fr 1fr 1.1fr; gap:6px; padding:5px 8px 7px; }
#imgbox { position:relative; }
#s3d { position:absolute; inset:0; background:#0b0d10; display:none; flex-direction:column; z-index:5; }
#s3d.open { display:flex; }
#s3dhead { display:flex; align-items:center; gap:4px; padding:3px 4px; font-size:12px; flex-wrap:wrap; }
#s3dhead button { font-size:13px; padding:3px 8px; border-radius:5px; }
#s3dhead button.on { background:#8ab4f8; color:#111; }
#s3dclose { margin-left:auto; }
#s3dcv { flex:1; min-height:0; width:100%; touch-action:none; }
#s3dinfo { font-size:12px; padding:2px 6px 4px; color:var(--text); }
#b-3d { background:#3a2f5a; border-color:#6b58a8; }
button { font:600 17px system-ui, sans-serif; color:var(--text); background:var(--btn); border:1px solid #3d5a7a;
  border-radius:8px; touch-action:manipulation; }
button:active { filter:brightness(1.4); }
button:disabled { opacity:.45; }
#b-inspect { background:#1f5f3a; border-color:#2f8a55; }
#b-ack { background:#5a4a1d; border-color:#8a7330; }
.muted { color:var(--muted); }
.draft { color:var(--muted); font-style:italic; }
</style></head>
<body><div id="app">
<div id="top"><div id="verdict">READY</div><div id="instr">Press Inspect.</div>
  <div id="meta"><div id="part">&nbsp;</div></div></div>
<div id="main">
  <div id="imgbox"><img id="img" alt="">
    <div id="s3d"><div id="s3dhead"><span id="s3dtitle">3D surface</span><span id="s3dtabs"></span>
      <button id="s3dclose" onclick="close3d()">Close</button></div>
      <canvas id="s3dcv"></canvas><div id="s3dinfo"></div></div></div>
  <div id="right">
    <div id="cap">capture: -</div>
    <div id="gatemsg"></div>
    <div id="tbl"><table><thead><tr><th></th><th>id</th><th>type</th><th>sev</th><th>action</th><th>conf</th><th>x,y</th><th>size mm (A mm&sup2;, e edge)</th></tr></thead>
      <tbody id="rows"><tr><td></td><td colspan="7" class="muted">no inspection yet</td></tr></tbody></table></div>
    <div id="advhead"><span id="advwho">advisor</span><span id="advstats"></span></div>
    <div id="adv" class="muted">Advice from Llama appears here after the verdict.</div>
    <div class="muted" style="font-size:10px">Built with Llama. Decision support only; the rules set the verdict.</div>
  </div>
</div>
<div id="bar">
  <button id="b-inspect" onclick="post('/api/inspect')">Inspect</button>
  <button onclick="post('/api/ask',{kind:'why'})">Why?</button>
  <button onclick="post('/api/ask',{kind:'fix'})">How to fix?</button>
  <button onclick="post('/api/ask',{kind:'shift'})">Shift summary</button>
  <button id="b-3d" onclick="open3d()">3D</button>
  <button id="b-ack" onclick="post('/api/ack')">Acknowledge</button>
</div></div>
<script>
"use strict";
let lastSeq = -1, lastImg = null;
function $(id) { return document.getElementById(id); }
function post(url, body) {
  fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body||{})})
    .then(() => tick()).catch(() => {});
}
const SHORT = {HOLD_FOR_QA:"HOLD", CLEAN_AND_REINSPECT:"CLEAN", REJECT:"REJECT", REWORK:"REWORK", PASS:"PASS"};
const CAUSE = "Likely cause and prevention (Llama): ";
function who(b) {
  if (!b) return "advisor";
  if (/llama/i.test(b) && /executorch/i.test(b)) return "Llama 3.2 1B \u00b7 ExecuTorch (on device)";
  if (b === "mock-template") return "Template (model answer refused or unavailable)";
  if (b === "rules-engine") return "Rules engine";
  return b;
}
function esc(t) { return t.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function adviceHtml(t) {
  // first answer = rules-engine header + Llama's cause note: show the note first, then the steps, then the facts
  const k = t.indexOf(CAUSE);
  if (k < 0) return null;
  const head = t.slice(0, k).trim(), note = t.slice(k + CAUSE.length).trim();
  const w = head.indexOf("What to do:");
  const facts = w > 0 ? head.slice(0, w).trim() : "", steps = w > 0 ? head.slice(w + 11).trim() : head;
  return '<div class="llama"><b>Llama: likely cause and prevention</b>' + esc(note) + '</div>'
       + '<div class="rules"><b>What to do (rules engine)</b>' + esc(steps) + '</div>'
       + (facts ? '<div class="rules" style="margin-top:6px"><b>Why (rules engine)</b>' + esc(facts.replace(/^Verdict: [A-Z_]+\. /, "")) + '</div>' : "");
}
function cell(tr, text, cls) { const td = document.createElement("td"); td.textContent = text; if (cls) td.className = cls; tr.appendChild(td); return td; }
function render(s) {
  const top = $("top"), i = s.inspection;
  $("b-inspect").disabled = !!s.busy || !s.source;
  if (!i) { $("verdict").textContent = s.busy ? "INSPECTING" : "READY"; $("instr").textContent = s.error || "Press Inspect."; return; }
  $("verdict").textContent = s.busy ? "INSPECTING" : i.verdict_text;
  top.style.background = i.colour;
  top.style.color = (i.verdict === "RETAKE") ? "#111" : "#fff";
  top.className = (!s.acked && !s.busy) ? "alert" : "";
  $("instr").textContent = s.error ? ("Error: " + s.error) : (i.instruction || "");
  $("part").textContent = i.part_id + "  #" + (i.id == null ? "" : i.id) + "  " + Math.round(i.timing_ms.total || 0) + " ms";
  $("cap").textContent = "capture: " + i.capture_line + (i.source && i.source.condition ? "  [" + i.source.condition + "]" : "");
  $("gatemsg").textContent = (i.gate !== "PASS" && i.gate_message) ? i.gate_message : "";
  const rows = $("rows"); rows.textContent = "";
  if (!i.defects.length) {
    const tr = document.createElement("tr"); cell(tr, ""); const td = cell(tr, i.verdict === "RETAKE" ? "not judged: capture refused" : "no defects found", "muted"); td.colSpan = 7; rows.appendChild(tr);
  }
  for (const d of i.defects) {
    const tr = document.createElement("tr");
    const sw = cell(tr, "", "sw"); sw.style.background = d.colour;
    cell(tr, d.id); cell(tr, d.type + (d.status ? " ?" : "")); cell(tr, d.severity); cell(tr, SHORT[d.disposition] || d.disposition);
    cell(tr, d.conf.toFixed(2)); cell(tr, d.x.toFixed(0) + "," + d.y.toFixed(0));
    cell(tr, d.size.replace(/([\d.]+) mm2/g, (m, v) => "A " + (+v).toFixed(1)).replace(/ mm/g, "").replace(/, /g, " ") + (d.edge != null ? " e" + Math.round(d.edge) : "")); rows.appendChild(tr);
  }
  const a = s.advice || {};
  const whoName = who(a.backend || s.advisor);
  const draft = a.status === "streaming";   // model text before the guardrail check (llm.check_reply)
  $("advwho").textContent = (a.kind === "shift" ? "shift summary \u00b7 " : "") + whoName + " \u00b7 " + (a.status || "idle")
    + (draft ? " (draft: checked when complete)" : "") + (s.advisor_degraded ? " (" + s.advisor_degraded + ")" : "");
  const st = a.stats || {};
  const parts = [];
  if (st.ttft_ms != null) parts.push("TTFT " + Math.round(st.ttft_ms) + " ms");
  else if (st.wall_ttft_ms != null) parts.push("first text " + Math.round(st.wall_ttft_ms) + " ms");
  if (st.decode_tokens_per_s != null) parts.push(st.decode_tokens_per_s.toFixed(1) + " tok/s");
  if (st.call_ms != null) parts.push(Math.round(st.call_ms) + " ms");
  $("advstats").textContent = parts.join("  ");
  const adv = $("adv");
  const txt = (a.text || "").replace(/\*\*/g, "") || (a.status === "thinking" || a.status === "waiting" ? "Llama is preparing advice..." : (i.advice || ""));
  if (adv.dataset.txt !== txt) {
    adv.dataset.txt = txt;
    const html = adviceHtml(txt);
    if (html) adv.innerHTML = html; else adv.textContent = txt;
    if (!draft) adv.scrollTop = 0;
  }
  adv.className = draft ? "draft" : (a.text ? "" : "muted");
  const imgKey = i.id + ":" + s.count;
  if (lastImg !== imgKey) { lastImg = imgKey; $("img").src = "/api/image.jpg?k=" + i.id + "&n=" + s.count; }
}
// ---- 3D surface view (app/surface3d.py via /api/surface) ----
const V3 = {open:false, data:null, forKey:null, sel:0, yaw:-35, pitch:58, drag:null};
function open3d() {
  V3.open = true; $("s3d").className = "open";
  if (!V3.data || V3.data.status !== "done") fetch("/api/surface", {method:"POST"}).then(() => poll3d());
  else draw3d();
}
function close3d() { V3.open = false; $("s3d").className = ""; }
function poll3d() {
  fetch("/api/surface", {cache:"no-store"}).then(r => r.json()).then(d => { V3.data = d; render3dHead(); draw3d(); }).catch(() => {});
}
function render3dHead() {
  const d = V3.data || {}, tabs = $("s3dtabs"); tabs.textContent = "";
  if ((d.status === "done" || d.status === "measuring") && d.items && d.items.length) {
    d.items.forEach((it, k) => { const b = document.createElement("button"); b.textContent = it.id + " " + it.type;
      b.className = k === V3.sel ? "on" : ""; b.onclick = () => { V3.sel = k; render3dHead(); draw3d(); }; tabs.appendChild(b); });
  }
  let info = "";
  if (d.status === "measuring" && !(d.items || []).length) info = "Measuring the surface from the left and right LED frames... " + (d.elapsed_s || 0) + " s";
  else if (d.status === "unavailable" || d.status === "error") info = d.why || "3D not available";
  else if (d.status === "done" && (!d.items || !d.items.length)) info = d.why || "Nothing to measure";
  else if (d.status === "done" || d.status === "measuring") {
    const it = d.items[Math.min(V3.sel, d.items.length - 1)];
    info = it.id + " " + it.type + ": " + (it.depth_mm == null ? "depth ?" : it.depth_mm.toFixed(2) + " mm deep")
      + (it.ci_mm != null ? " (\u00b1" + it.ci_mm.toFixed(2) + ")" : "") + (it.rel_thickness != null ? ", " + (100 * it.rel_thickness).toFixed(1) + " % of thickness" : "")
      + " \u00b7 " + (it.method || "") + " \u00b7 " + it.runtime_s + " s. Drag to turn. Not used for the verdict."
      + (it.type === "pit" ? " Pit depth is indicative only: small pits read shallow." : "")
      + (d.status === "measuring" ? "  [" + d.items.length + " of " + (d.total || "?") + " measured, " + (d.elapsed_s || 0) + " s]" : "");
  }
  $("s3dinfo").textContent = info;
}
function draw3d() {
  const cv = $("s3dcv"), box = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(10, box.width * dpr); cv.height = Math.max(10, box.height * dpr);
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, box.width, box.height);
  const d = V3.data; if (!d || !d.items || !d.items.length) return;
  const it = d.items[Math.min(V3.sel, d.items.length - 1)], H = it.height_mm, ny = H.length, nx = H[0].length, s = it.spacing_mm;
  let hmax = 1e-6; for (const row of H) for (const v of row) hmax = Math.max(hmax, Math.abs(v));
  const ext = Math.max(nx, ny) * s, ex = Math.max(1, Math.min(60, Math.round(0.28 * ext / hmax)));
  const cy = Math.cos(V3.yaw * Math.PI / 180), sy = Math.sin(V3.yaw * Math.PI / 180);
  const cp = Math.cos(V3.pitch * Math.PI / 180), sp = Math.sin(V3.pitch * Math.PI / 180);
  const sc = 0.62 * Math.min(box.width, box.height) / ext, ox = box.width / 2, oy = box.height / 2 + 10;
  const P = (i, j) => { const x = (j - (nx - 1) / 2) * s, y = (i - (ny - 1) / 2) * s, z = H[i][j] * ex;
    const xr = x * cy - y * sy, yr = x * sy + y * cy, ys = yr * cp - z * sp, depth = yr * sp + z * cp;
    return [ox + xr * sc, oy + ys * sc, depth]; };
  const pts = []; for (let i = 0; i < ny; i++) { const r = []; for (let j = 0; j < nx; j++) r.push(P(i, j)); pts.push(r); }
  const quads = [];
  for (let i = 0; i < ny - 1; i++) for (let j = 0; j < nx - 1; j++) {
    const a = pts[i][j], b = pts[i][j + 1], c = pts[i + 1][j + 1], e = pts[i + 1][j];
    const hz = (H[i][j] + H[i][j + 1] + H[i + 1][j + 1] + H[i + 1][j]) / 4;
    const dzx = (H[i][j + 1] - H[i][j] + H[i + 1][j + 1] - H[i + 1][j]) / (2 * s), dzy = (H[i + 1][j] - H[i][j] + H[i + 1][j + 1] - H[i][j + 1]) / (2 * s);
    const n = 1 / Math.hypot(dzx * ex, dzy * ex, 1), lit = 0.55 + 0.45 * Math.max(0, (-dzx * ex * 0.5 - dzy * ex * 0.5 + 0.7) * n);
    quads.push([(a[2] + b[2] + c[2] + e[2]) / 4, a, b, c, e, hz, lit]);
  }
  quads.sort((p, q) => q[0] - p[0]);
  for (const [, a, b, c, e, hz, lit] of quads) {
    const t = Math.max(-1, Math.min(1, hz / hmax));
    const base = t < 0 ? [235 + t * 175, 235 + t * 105, 240 + t * 10] : [235, 235 - t * 95, 240 - t * 200];
    g.fillStyle = "rgb(" + base.map(v => Math.round(Math.max(0, Math.min(255, v * lit)))).join(",") + ")";
    g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.lineTo(c[0], c[1]); g.lineTo(e[0], e[1]); g.closePath(); g.fill();
  }
  g.fillStyle = "#9aa4ad"; g.font = "11px system-ui";
  g.fillText(ext.toFixed(1) + " mm across \u00b7 relief \u00d7" + ex + " \u00b7 blue = below the surrounding surface", 6, box.height - 6);
}
(function () {
  const cv = $("s3dcv");
  cv.addEventListener("pointerdown", e => { V3.drag = [e.clientX, e.clientY, V3.yaw, V3.pitch]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => { if (!V3.drag) return;
    V3.yaw = V3.drag[2] + (e.clientX - V3.drag[0]) * 0.6; V3.pitch = Math.max(5, Math.min(89, V3.drag[3] - (e.clientY - V3.drag[1]) * 0.4)); draw3d(); });
  cv.addEventListener("pointerup", () => { V3.drag = null; });
})();
function tick() {
  fetch("/api/latest", {cache:"no-store"}).then(r => r.json()).then(s => { lastSeq = s.seq; render(s);
      const sf = s.surface || {};
      if (V3.open && (sf.for !== V3.forKey || sf.status !== (V3.data || {}).status || sf.status === "measuring")) {
        if (sf.for !== V3.forKey) V3.sel = 0; V3.forKey = sf.for; poll3d(); }
      $("b-3d").textContent = sf.status === "measuring" ? "3D ..." : (sf.status === "done" ? "3D \u2713" : "3D"); })
    .catch(() => { $("part").textContent = "station not responding"; });
}
setInterval(tick, 1000); tick();
</script></body></html>
"""
