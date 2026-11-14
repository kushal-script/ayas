"""The advisor of the app: Llama 3.2 via ExecuTorch explains the record (trinetra.llm does the
prompting, the guardrail check and the template fallback; this module adds measurement and
streaming).

Backends (configs/app.yaml llm.backend):
- mock: llm.MockBackend, the deterministic template (development, and every fallback);
- executorch: the llama_main runner binary through llm.build_backend, run here with its stdout
  streamed (live UI text, wall-clock TTFT) and its 'PyTorchObserver {json}' stats line parsed
  (model TTFT, prefill and decode tokens/s, model load time);
- executorch_py: the in-process TextLLMRunner of the executorch pip package (imported lazily,
  only when this backend is chosen): the model stays loaded between answers.

Stats per answer (AdviceResult.stats): ttft_ms (first generated token after the prompt, model
clock), decode_tokens_per_s, prefill_tokens_per_s, prompt_tokens, generated_tokens, total_ms
(inference), load_ms, wall_ttft_ms / wall_ms (process start or call -> first text / end, which
include the runner's model load), source (runner_stats | wall | template).

The advisor never blocks the record: any failure ends in llm.Advisor's template fallback, and
the caller (pipeline) runs it after the alert.
"""

from __future__ import annotations

import codecs
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .. import llm
from . import REPO_ROOT

OBSERVER = "PyTorchObserver"
EOT_RE = re.compile(r"<\|eot_id\|>|<\|end_of_text\|>|<\|eom_id\|>")
QUESTIONS = {   # the UI buttons; worded to hit the template branches of llm.answer_steel too
    "why": "Why this verdict? Show the rule and the evidence for each finding.",
    "fix": "How do I fix it? Give the repair steps.",
}
APP_KEYS = ("backend", "history_turns", "stream", "py")
NO_ANSWER = "none (advisor failed)"   # AdviceResult.backend when no backend, not even the template, answered


@dataclass
class AdviceResult:
    kind: str                      # explain | ask | shift
    text: str
    backend: str                   # who answered: the model, or mock-template on a fallback
    model: str                     # the configured backend
    question: str | None = None
    problems: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- runner stats -----------------------------------------------------------------------------

def parse_observer(text: str) -> dict[str, Any]:
    """The last 'PyTorchObserver {json}' line of runner output -> timing stats (empty if none).

    ExecuTorch's stats.h: TTFT = first_token_ms - inference_start_ms; prefill = prompt_tokens /
    (prompt_eval_end_ms - inference_start_ms); decode = generated_tokens / (inference_end_ms -
    prompt_eval_end_ms); times are in 1/SCALING_FACTOR_UNITS_PER_SECOND s (ms by default)."""
    raw = None
    for line in reversed(text.splitlines()):
        i = line.find(OBSERVER)
        if i >= 0:
            try:
                raw = json.loads(line[i + len(OBSERVER):].strip())
                break
            except ValueError:
                continue
    return observer_stats(raw) if isinstance(raw, dict) else {}


def observer_stats(s: Mapping[str, Any]) -> dict[str, Any]:
    """Timing stats from the runner's Stats JSON (also TextLLMRunner Stats.to_json_string())."""
    unit = 1000.0 / float(s.get("SCALING_FACTOR_UNITS_PER_SECOND") or 1000.0)   # -> ms

    def span(a: str, b: str) -> float | None:
        if s.get(a) is None or s.get(b) is None:
            return None
        return max(0.0, (float(s[b]) - float(s[a])) * unit)

    out: dict[str, Any] = {"source": "runner_stats", "prompt_tokens": s.get("prompt_tokens"),
                           "generated_tokens": s.get("generated_tokens"),
                           "ttft_ms": span("inference_start_ms", "first_token_ms"),
                           "total_ms": span("inference_start_ms", "inference_end_ms"),
                           "load_ms": span("model_load_start_ms", "model_load_end_ms")}
    pre, dec = span("inference_start_ms", "prompt_eval_end_ms"), span("prompt_eval_end_ms", "inference_end_ms")
    if s.get("prefill_token_per_sec") is not None:
        out["prefill_tokens_per_s"] = float(s["prefill_token_per_sec"])
    elif pre and s.get("prompt_tokens"):
        out["prefill_tokens_per_s"] = float(s["prompt_tokens"]) / (pre / 1e3)
    if s.get("decode_token_per_sec") is not None:
        out["decode_tokens_per_s"] = float(s["decode_token_per_sec"])
    elif dec and s.get("generated_tokens"):
        out["decode_tokens_per_s"] = float(s["generated_tokens"]) / (dec / 1e3)
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}


def answer_part(text: str, prompt: str) -> str | None:
    """The answer so far in streamed runner stdout (None while the prompt is still being echoed)."""
    i = text.find(prompt)
    if i >= 0:
        rest = text[i + len(prompt):]
    else:
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker not in text:
            return None
        rest = text.rsplit(marker, 1)[1]
    rest = EOT_RE.split(rest, maxsplit=1)[0]
    return "\n".join(ln for ln in rest.splitlines() if OBSERVER not in ln).strip()


def run_streaming(cmd: Sequence[str], prompt: str, timeout_s: float,
                  on_text: Callable[[str], None] | None = None) -> tuple[int, str, str, dict[str, float]]:
    """Run the runner, reading stdout as it comes -> (returncode, stdout, stderr, wall stats).

    A watchdog kills the process after timeout_s (returncode then -9 or similar, and
    'timeout' in the wall stats)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    err: list[bytes] = []
    drain = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)
    drain.start()
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        proc.kill()

    dog = threading.Timer(float(timeout_s), kill)
    dog.daemon = True
    dog.start()
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    out, first, last = [], None, ""
    try:
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            out.append(dec.decode(chunk))
            ans = answer_part("".join(out), prompt)
            if ans:
                if first is None:
                    first = time.perf_counter()
                if on_text is not None and ans != last:
                    last = ans
                    on_text(ans)
        out.append(dec.decode(b"", final=True))
        proc.wait()
    finally:
        dog.cancel()
        drain.join(timeout=2.0)
    wall = {"wall_ms": round((time.perf_counter() - t0) * 1e3, 1),
            "wall_ttft_ms": None if first is None else round((first - t0) * 1e3, 1)}
    if timed_out.is_set():
        wall["timeout"] = True
    return proc.returncode, "".join(out), b"".join(err).decode(errors="replace"), wall


# ---- backends ---------------------------------------------------------------------------------

class StreamingRunner:
    """llm.ExecuTorchBackend (llama_main) with streamed stdout and parsed stats.

    Uses the backend's own command() and output cleaning, so prompt format and flags stay
    trinetra.llm's; falls back to the backend's plain generate() when it has no command()."""

    def __init__(self, base):
        self.base = base
        self.name = getattr(base, "name", type(base).__name__)
        # llm.Advisor sizes its prompt budget from backend.seq_len (prompt + answer tokens): without
        # it, follow-up questions grow past the runner's context and every answer falls back
        self.seq_len = getattr(base, "seq_len", None)
        self.listener: Callable[[str], None] | None = None
        self.last_stats: dict[str, Any] = {}

    def generate(self, messages: Sequence[dict[str, str]], context: dict[str, Any] | None = None) -> str:
        self.last_stats = {}
        if not hasattr(self.base, "command"):
            t0 = time.perf_counter()
            text = self.base.generate(messages, context)
            self.last_stats = {"source": "wall", "wall_ms": round((time.perf_counter() - t0) * 1e3, 1)}
            return text
        prompt = llm.format_llama3(messages, bos=bool(getattr(self.base, "bos_in_prompt", False)))
        code, out, err, wall = run_streaming(self.base.command(prompt), prompt,
                                             float(getattr(self.base, "timeout_s", 180.0)), self.listener)
        self.last_stats = {**parse_observer(out), **wall} or wall
        self.last_stats.setdefault("source", "wall")
        if wall.get("timeout"):
            raise RuntimeError(f"llama runner timed out after {getattr(self.base, 'timeout_s', 180)} s")
        if code != 0:
            raise RuntimeError(f"llama runner exited {code}: {err[-500:]}")
        return llm.clean_runner_output(out, prompt)


class PyRunnerBackend:
    """In-process ExecuTorch TextLLMRunner (executorch pip package, >= 1.2): the model stays loaded.

    API (docs/research/2026-11-05_executorch.md): TextLLMRunner(model_path, tokenizer_path[,
    data_path]).generate(prompt, GenerationConfig, token_callback, stats_callback), reset();
    the prompt carries <|begin_of_text|> itself (num_bos = 0); reset() before every answer.
    self.seq_len (what llm.Advisor sizes its prompt budget from) is the generation seq_len when
    set, else context_len (app.yaml llm.seq_len: the context the .pte was exported with)."""

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path, data_path: str | Path | None = None,
                 max_new_tokens: int = 160, temperature: float = 0.0, seq_len: int = -1, warmup: bool = False,
                 context_len: int = 2048):
        try:
            runner_mod = importlib.import_module("executorch.extension.llm.runner")
        except ImportError as e:
            raise ImportError("executorch_py backend: `import executorch.extension.llm.runner` failed "
                              f"({e}); pip install executorch (and torch, which its runner imports)") from e
        for mod in ("executorch.extension.llm.custom_ops.custom_ops", "executorch.kernels.quantized"):
            try:   # register the custom and quantized ops a SpinQuant / XNNPACK .pte needs
                importlib.import_module(mod)
            except ImportError:
                pass
        self._GenerationConfig = runner_mod.GenerationConfig
        args = [str(model_path), str(tokenizer_path)] + ([str(data_path)] if data_path else [])
        t0 = time.perf_counter()
        self._runner = runner_mod.TextLLMRunner(*args)
        self.load_ms = round((time.perf_counter() - t0) * 1e3, 1)
        self.gen = {"max_new_tokens": int(max_new_tokens), "temperature": float(temperature),
                    "seq_len": int(seq_len), "echo": False, "num_bos": 0, "num_eos": 0}
        self.seq_len = int(seq_len) if int(seq_len) > 0 else int(context_len)
        self.name = f"llama via executorch TextLLMRunner ({Path(model_path).name})"
        self.listener: Callable[[str], None] | None = None
        self.last_stats: dict[str, Any] = {}
        self._lock = threading.Lock()
        if warmup:
            self.generate([{"role": "user", "content": "Say OK."}], None, max_new_tokens=4)

    def _config(self, **over: Any):
        cfg = self._GenerationConfig()
        for k, v in {**self.gen, **over}.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def generate(self, messages: Sequence[dict[str, str]], context: dict[str, Any] | None = None, **over: Any) -> str:
        prompt = llm.format_llama3(messages, bos=True)
        pieces: list[str] = []
        stats: dict[str, Any] = {}
        first: list[float] = []
        listener = self.listener

        def on_token(tok: str) -> None:
            if not first:
                first.append(time.perf_counter())
            pieces.append(tok)
            if listener is not None:
                listener(EOT_RE.split("".join(pieces), maxsplit=1)[0].strip())

        def on_stats(s) -> None:
            try:
                stats.update(observer_stats(json.loads(s.to_json_string())))
            except Exception:   # stats are a nice-to-have; never fail the answer for them
                pass

        with self._lock:
            t0 = time.perf_counter()
            self._runner.reset()
            self._runner.generate(prompt, self._config(**over), on_token, on_stats)
            t1 = time.perf_counter()
        self.last_stats = {**stats, "wall_ms": round((t1 - t0) * 1e3, 1),
                           "wall_ttft_ms": round((first[0] - t0) * 1e3, 1) if first else None,
                           "load_ms": stats.get("load_ms") or self.load_ms}
        self.last_stats.setdefault("source", "wall")
        return EOT_RE.split("".join(pieces), maxsplit=1)[0].strip()


def _executorch_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The llm section minus the app's own keys, limited to what llm.ExecuTorchBackend accepts."""
    out = {k: v for k, v in cfg.items() if k not in APP_KEYS and v is not None}
    try:
        params = inspect.signature(llm.ExecuTorchBackend.__init__).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            out = {k: v for k, v in out.items() if k in params}
    except (TypeError, ValueError):
        pass
    return out


def runner_problems(base) -> list[str]:
    """What stops llm.ExecuTorchBackend from ever answering: the runner executable, the .pte or
    the tokenizer missing (checked at start-up, so the station warns once instead of falling
    back on every answer)."""
    out = []
    runner = list(getattr(base, "runner", []) or [])
    exe = runner[0] if runner else ""
    if not exe or not (shutil.which(exe) or (Path(exe).is_file() and os.access(exe, os.X_OK))):
        out.append(f"runner not found: {exe or '(none)'}")
    for key in ("model_path", "tokenizer_path"):
        p = getattr(base, key, None)
        if not p or not Path(p).exists():
            out.append(f"{key} not found: {p}")
    return out


def make_backend(llm_cfg: Mapping[str, Any] | None, root: Path = REPO_ROOT):
    """Backend object from the app's llm config (mock | executorch | executorch_py); raises if the
    configured model cannot run (AdvisorService.from_config then falls back to the template)."""
    cfg = dict(llm_cfg or {})
    kind = str(cfg.get("backend", "mock"))
    if kind == "mock":
        return llm.MockBackend()
    if kind == "executorch":
        base = llm.build_backend({"backend": "executorch", **_executorch_kwargs(cfg)}, root)
        problems = runner_problems(base)
        if problems:
            raise FileNotFoundError("; ".join(problems))
        return StreamingRunner(base) if cfg.get("stream", True) else base
    if kind == "executorch_py":
        py = {**{k: cfg[k] for k in ("model_path", "tokenizer_path") if k in cfg}, **(cfg.get("py") or {})}
        if cfg.get("seq_len"):
            py.setdefault("context_len", int(cfg["seq_len"]))
        for k in ("model_path", "tokenizer_path", "data_path"):
            if py.get(k) and not Path(py[k]).is_absolute():
                py[k] = str(root / py[k])
        return PyRunnerBackend(**{k: v for k, v in py.items()
                                  if k in inspect.signature(PyRunnerBackend.__init__).parameters})
    raise ValueError(f"unknown llm backend {kind!r} (mock | executorch | executorch_py)")


# ---- the service ----------------------------------------------------------------------------------

class AdvisorService:
    """One llm.Advisor conversation per inspection (kept for the follow-up questions), timed.

    on_text(text_so_far) gets the streamed answer when the backend streams; calls are
    serialised (one model run at a time)."""

    def __init__(self, backend, spec: Mapping[str, Any], history_turns: int = 3, keep: int = 8,
                 degraded: str | None = None):
        self.backend, self.spec = backend, dict(spec)
        self.history_turns, self.keep = int(history_turns), int(keep)
        self.degraded = degraded            # why the configured model is not in use (fallback to mock)
        self._conv: dict[Any, llm.Advisor] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, llm_cfg: Mapping[str, Any] | None, spec: Mapping[str, Any], root: Path = REPO_ROOT,
                    fallback: bool = True) -> "AdvisorService":
        """Build the configured backend; with fallback, a backend that cannot start becomes the template."""
        cfg = dict(llm_cfg or {})
        try:
            backend, degraded = make_backend(cfg, root), None
        except Exception as e:
            if not fallback:
                raise
            backend, degraded = llm.MockBackend(), f"{cfg.get('backend')} unavailable: {e}"
        return cls(backend, spec, int(cfg.get("history_turns", 3)), degraded=degraded)

    @property
    def name(self) -> str:
        return getattr(self.backend, "name", type(self.backend).__name__)

    def _advisor(self, record: Mapping[str, Any], key: Any) -> llm.Advisor:
        key = key if key is not None else record.get("part_id")
        adv = self._conv.get(key)
        if adv is None:
            adv = llm.Advisor(self.backend, self.spec, dict(record), history_turns=self.history_turns)
            self._conv[key] = adv
            while len(self._conv) > self.keep:
                self._conv.pop(next(iter(self._conv)))
        return adv

    def _call(self, kind: str, fn: Callable[[], str], on_text, question: str | None = None) -> AdviceResult:
        with self._lock:
            if hasattr(self.backend, "listener"):
                self.backend.listener = on_text
            if hasattr(self.backend, "last_stats"):
                self.backend.last_stats = {}
            t0 = time.perf_counter()
            error = None
            try:
                text = fn()
            except Exception as e:   # llm.Advisor already falls back; this is the last line of defence
                text, error = f"(Advisor failed: {e})", str(e)
            wall = round((time.perf_counter() - t0) * 1e3, 1)
            if hasattr(self.backend, "listener"):
                self.backend.listener = None
            stats = dict(getattr(self.backend, "last_stats", {}) or {})
        stats.setdefault("source", "template" if isinstance(self.backend, llm.MockBackend) else "wall")
        stats["call_ms"] = wall
        who = self.name if error is None else NO_ANSWER     # nobody answered: never credit the model
        return AdviceResult(kind, text, who, self.name, question, [], stats, error)

    def explain(self, record: Mapping[str, Any], key: Any = None, on_text=None) -> AdviceResult:
        box: dict[str, llm.Advisor] = {}

        def run() -> str:   # the conversation is built inside _call: a bad record is an error result too
            box["adv"] = self._advisor(record, key)
            return box["adv"].explain()

        return self._who(self._call("explain", run, on_text), box.get("adv"))

    def ask(self, record: Mapping[str, Any], question: str, key: Any = None, on_text=None) -> AdviceResult:
        q = QUESTIONS.get(question, question)
        box: dict[str, llm.Advisor] = {}

        def run() -> str:
            box["adv"] = self._advisor(record, key)
            return box["adv"].ask(q)

        return self._who(self._call("ask", run, on_text, q), box.get("adv"))

    def shift(self, records: Sequence[Mapping[str, Any]], last_n: int = 50, on_text=None) -> AdviceResult:
        summary = llm.shift_summary([dict(r) for r in records], last_n=last_n)
        who: list[str] = []

        def run() -> str:
            text, by = llm.explain_shift(summary, self.backend, self.spec)
            who.append(by)
            return text

        res = self._call("shift", run, on_text, "shift summary")
        if who:
            res.backend = who[0]
        res.stats["summary"] = {k: summary[k] for k in ("inspections", "by_verdict", "retakes")}
        return res

    def _who(self, res: AdviceResult, adv: llm.Advisor | None) -> AdviceResult:
        if res.error is not None or adv is None:   # adv.last would be the previous answer's, not this one's
            return res
        last = getattr(adv, "last", {}) or {}
        res.backend = last.get("backend") or res.backend
        res.problems = list(last.get("problems") or [])
        return res
