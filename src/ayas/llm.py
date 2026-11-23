"""Operator assistant: Llama (via the ExecuTorch runner) explains the inspection
JSON and answers follow-up questions. It never changes the verdict.

Backends:
- executorch: runs the ExecuTorch `llama_main` binary with a Llama 3.2 .pte.
- mock: deterministic template answers built from the JSON. Used for development
  on a laptop, and as the fallback if the model fails during a shift.

Domains: the original station and penetrant reports (2.0), aerospace (2.1) and steel and metal
surfaces (record 3.0, spec_steel.yaml). In steel and aerospace mode every model reply is checked by
check_reply; on any problem the template answers instead and Advisor.last records which backend
answered. shift_summary / explain_shift summarise the last N records for the operator.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .report import compact_for_llm, is_steel_record, llm_items
from .rules import VERDICTS

SYSTEM_PROMPT = """You are Ayas, the quality-control assistant at a metal-part inspection station. You talk to a shop-floor operator.
You get an INSPECTION REPORT (JSON from the camera system and rules engine) and SOP NOTES for what was found.
Rules:
1. The verdict, severity and disposition in the report are final. Never change or soften them. If the operator disagrees, tell them to call the QA supervisor.
2. Use only facts from the report and SOP notes. If the answer is not there, say you do not know and suggest asking QA.
3. Explain evidence in plain words.
   - mode "surface": three cameras (top, 45-degree, grazing), each with a white light beside it. "contrast" is the brightness change against defect-free parts; positive means brighter. "top_rgb" is the colour change seen from the top. Bright at the grazing camera with no colour change means a scratch or a particle scattering light. A colour change seen the same from every camera means a stain or corrosion. Darker under every light from the top and 45-degree cameras means a crack or pit trapping light. "depth_mm" comes from the 3D shape map: negative is pushed in (dent), positive is raised (bump).
   - mode "penetrant": red dye on a white developer. A linear indication (longer than 3 times its width) is crack-like; a rounded one is pore-like. Indications at or below the relevance size are only logged. The acceptance profile sets the limits.
4. Give short numbered steps for what to do next. Use mm. Keep answers under 120 words unless asked for detail."""

SYSTEM_PROMPT_AERO = """You are Ayas, an inspector's assistant for riveted aluminium coupons (aircraft-representative parts) at a Tier-2/3 aerospace supplier. You talk to a shop-floor inspector.
You get an INSPECTION REPORT (JSON from the camera, the detector and the rules engine) and SOP NOTES with actions and public references (FAA AC 43.13-1B, Piper SL 1095A, Boeing Q219P, AS9100D).
Rules:
1. The verdict, severity and disposition in the report are final. Never change or soften them. If the inspector disagrees, they raise it with QA or the customer MRB.
2. Use only facts and numbers from the report and SOP notes. If the answer is not there, say you do not know.
3. You give decision support only. Never say a part is accepted, released, approved, airworthy or certified. The qualified inspector decides.
4. Supplier options for a nonconforming part are rework to drawing or scrap. Use-as-is and repair are decided only by the customer MRB.
5. Cite the reference ids from the SOP notes in brackets, for example [AC 43.13-1B 4-57c].
6. Out-of-scope checks (cracks under fastener heads or below the surface, intergranular corrosion, elongated holes under installed rivets, depth) go to NDT by a qualified NAS 410 / EN 4179 Level 2.
7. "findings" are fastener checks: edge distance and pitch from hole centres, missing fastener, elongated hole, lone open hole.
8. Give short numbered steps. Use mm. Keep answers under 120 words unless asked for detail."""


SYSTEM_PROMPT_STEEL = """You are Ayas, the quality-control assistant at a steel and metal fabrication inspection station. You talk to a shop-floor operator.
You get an INSPECTION BRIEF from the rules engine. The brief is always right.
The screen already shows the verdict, the measurements and the steps. When you get a CAUSE NOTE, write 2 plain sentences for the operator: what most likely caused the defect and how to prevent it next time. No numbers.
Rules:
1. The verdict, severity and disposition are final. Never change them. If the operator disagrees, tell them to call QA.
2. Use only facts and numbers from the brief. Name only the defects in the brief. If the answer is not in the brief, say you do not know.
3. The camera does not measure depth. Never state a depth; say QA checks it with a gauge.
4. Confidence is how sure the detector is, not severity. "uncertain, operator to confirm" means the operator checks the spot.
5. Cite only these references, exactly as written: {references}. Never name any other standard, clause or table.
6. Make no promises about quality or certification.
7. Use mm and plain words, not field names. Under 80 words. A follow-up question gets 1 to 4 short sentences.
8. If the operator writes in Hindi or asks for Hindi, answer in simple Hindi; keep ids, numbers, units and verdict words in English."""


def is_aero(spec: dict[str, Any] | None = None, report: dict[str, Any] | None = None) -> bool:
    return (spec or {}).get("domain") == "aerospace" or (report or {}).get("domain") == "aerospace"


def is_steel(spec: dict[str, Any] | None = None, report: dict[str, Any] | None = None) -> bool:
    return (spec or {}).get("domain") == "steel_surface" or (report is not None and is_steel_record(report))


def system_prompt(spec: dict[str, Any], report: dict[str, Any] | None = None) -> str:
    """The system prompt for the spec's domain; in steel mode it lists the record's reference ids."""
    if is_steel(spec, report):
        refs = [r["id"] for r in (report or {}).get("references", [])]
        listed = "; ".join(f'"{r}"' for r in refs) if refs else "none for this record (do not cite any standard)"
        return SYSTEM_PROMPT_STEEL.replace("{references}", listed)
    return SYSTEM_PROMPT_AERO if is_aero(spec, report) else SYSTEM_PROMPT


def _aero_sop_notes(report: dict[str, Any], spec: dict[str, Any]) -> str:
    """Per defect or finding type: its reference ids, two likely causes and the prevention; then
    the out-of-scope referral and one line per cited reference."""
    lines = []
    fr = spec.get("fastener_rules", {}).get("findings", {})
    shown, _ = llm_items(report)
    kinds = [(k, fr.get(k, {})) for k in dict.fromkeys(i["type"] for i in shown if "fasteners" in i)]
    kinds += [(k, spec["defects"].get(k, {})) for k in dict.fromkeys(i["type"] for i in shown if "fasteners" not in i)]
    cited = {r for i in shown for r in i.get("reference") or []} | {"Boeing Q219P"}
    for kind, rule in kinds:
        refs = f" [{'; '.join(rule['reference'])}]" if rule.get("reference") else ""
        causes = "; ".join(rule.get("likely_causes", [])[:2]) or "unknown"
        lines.append(f"- {kind.replace('_', ' ')}{refs}: likely causes: {causes}. Prevention: {rule.get('prevention', '')}")
    if not kinds:
        lines.append("- no defects or fastener findings")
    oos = "; ".join(o["item"] for o in spec.get("out_of_scope", [])[:3])
    lines.append(f"- out of scope, refer to NDT by a qualified NAS 410 / EN 4179 Level 2: {oos}")
    lines.append("- demo limits from public sources; real limits come from the customer drawing")
    lines += [f"- [{r['id']}]: {r['text']}" for r in report.get("references", []) if r["id"] in cited]
    return "\n".join(lines)


def sop_notes(report: dict[str, Any], spec: dict[str, Any]) -> str:
    if is_steel(spec, report):
        return _steel_sop_notes(report, spec)
    if is_aero(spec, report):
        return _aero_sop_notes(report, spec)
    lines = []
    if report.get("mode") == "penetrant":
        pen = spec.get("penetrant", {})
        profile = report.get("penetrant", {}).get("profile", pen.get("profile"))
        prof = pen.get("profiles", {}).get(profile, {})
        lines.append(f"- acceptance profile {profile}: {prof.get('source', '')}; relevance threshold {pen.get('relevant_mm')} mm")
        for kind in ("linear", "rounded"):
            if any(d["type"] == f"{kind}_indication" for d in report["defects"]):
                causes = "; ".join(pen.get("likely_causes", {}).get(kind, [])) or "unknown"
                lines.append(f"- {kind} indication: action: {pen['actions'][kind]} Likely causes: {causes}. Prevention: {pen.get('prevention', {}).get(kind, '')}")
        return "\n".join(lines) or "- no indications found"
    for kind in dict.fromkeys(d["type"] for d in report["defects"]):
        rule = spec["defects"].get(kind, {})
        causes = "; ".join(rule.get("likely_causes", [])) or "unknown"
        lines.append(f"- {kind}: action: {rule.get('action', '')} Likely causes: {causes}. Prevention: {rule.get('prevention', '')}")
    return "\n".join(lines) or "- no defects found"


def format_llama3(messages: Sequence[dict[str, str]], bos: bool = False) -> str:
    """Llama 3 chat template. The ExecuTorch runner adds BOS itself by default."""
    out = ["<|begin_of_text|>"] if bos else []
    prefill = None
    if messages and messages[-1].get("prefill"):   # the answer's opening words, continued by the model
        messages, prefill = messages[:-1], messages[-1]["content"]
    for m in messages:
        out.append(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>")
    out.append("<|start_header_id|>assistant<|end_header_id|>\n\n" + (prefill or ""))
    return "".join(out)


class MockBackend:
    name = "mock-template"

    def generate(self, messages: Sequence[dict[str, str]], context: dict[str, Any]) -> str:
        report, spec = context["report"], context["spec"]
        question = messages[-1]["content"]
        if is_steel(spec, report):
            return explain_steel(report, spec) if len(messages) <= 2 else answer_steel(question, report, spec)
        aero = is_aero(spec, report)
        if len(messages) <= 2:
            return explain_aero(report) if aero else explain(report)
        return answer_aero(question, report, spec) if aero else answer(question, report, spec)


class ExecuTorchBackend:
    """Calls the ExecuTorch Llama runner (`llama_main`) once per answer."""

    def __init__(
        self,
        runner: str | Sequence[str],
        model_path: str,
        tokenizer_path: str,
        seq_len: int = 1536,
        temperature: float = 0.2,
        cpu_threads: int | None = 4,
        extra_args: Sequence[str] = (),
        timeout_s: float = 180.0,
        bos_in_prompt: bool = False,
    ):
        self.runner = shlex.split(runner) if isinstance(runner, str) else list(runner)
        self.model_path, self.tokenizer_path = model_path, tokenizer_path
        self.seq_len, self.temperature, self.cpu_threads = seq_len, temperature, cpu_threads
        self.extra_args, self.timeout_s, self.bos_in_prompt = list(extra_args), timeout_s, bos_in_prompt
        self.name = f"llama via executorch ({model_path})"

    def command(self, prompt: str) -> list[str]:
        cmd = self.runner + [
            f"--model_path={self.model_path}",
            f"--tokenizer_path={self.tokenizer_path}",
            f"--seq_len={self.seq_len}",
            f"--temperature={self.temperature}",
            f"--prompt={prompt}",
        ]
        if self.cpu_threads:
            cmd.append(f"--cpu_threads={self.cpu_threads}")
        return cmd + self.extra_args

    def generate(self, messages: Sequence[dict[str, str]], context: dict[str, Any] | None = None) -> str:
        prompt = format_llama3(messages, bos=self.bos_in_prompt)
        proc = subprocess.run(self.command(prompt), capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"llama runner exited {proc.returncode}: {proc.stderr[-500:]}")
        return clean_runner_output(proc.stdout, prompt)


def clean_runner_output(stdout: str, prompt: str) -> str:
    """The runner echoes the prompt, then the answer, then stats lines."""
    text = stdout
    i = text.find(prompt)
    if i >= 0:
        text = text[i + len(prompt) :]
    else:
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker in text:
            text = text.rsplit(marker, 1)[1]
    text = re.split(r"<\|eot_id\|>|<\|end_of_text\|>", text, maxsplit=1)[0]
    lines = [ln for ln in text.splitlines() if not ln.startswith("PyTorchObserver")]
    return "\n".join(lines).strip()


def build_backend(llm_cfg: dict[str, Any], root: Path | None = None):
    kind = llm_cfg.get("backend", "mock")
    if kind == "mock":
        return MockBackend()
    if kind == "executorch":
        args = {k: v for k, v in llm_cfg.items() if k not in ("backend", "history_turns")}
        if root is not None:  # relative paths in the config are relative to the project root
            for name in ("model_path", "tokenizer_path"):
                if not Path(args[name]).is_absolute():
                    args[name] = str(root / args[name])
        return ExecuTorchBackend(**args)
    raise ValueError(f"unknown llm backend {kind!r}")


# Prompt budget (Advisor.max_prompt_bytes): tokens kept for the answer, and a conservative UTF-8
# bytes-per-token for the Llama 3 tokenizer (about 4 on this JSON-heavy English; bytes, not
# characters, so Devanagari, 3 bytes a character and about 1-2 characters a token, is not undercounted).
ANSWER_TOKENS = 256
PROMPT_BYTES_PER_TOKEN = 3.5


class Advisor:
    """One conversation about one inspected part.

    guard (default: on for an aerospace or steel spec) checks every model reply with check_reply
    and answers with the template instead when it finds problems. ``last`` records which backend
    answered, the problems found and the rejected reply; ``log`` keeps one entry per answer.

    max_prompt_bytes bounds the formatted prompt (UTF-8): the oldest Q&A turns are dropped until it
    fits (the report message always stays). By default it comes from the backend's seq_len
    (llama_main --seq_len counts prompt + answer): (seq_len - ANSWER_TOKENS) x PROMPT_BYTES_PER_TOKEN; a
    backend without one (the mock) is not limited. A worst-case steel record is about 1,300
    tokens, so 3 turns of history would overflow a 2048-token context.
    """

    def __init__(self, backend, spec: dict[str, Any], report: dict[str, Any], history_turns: int = 4,
                 guard: bool | None = None, max_prompt_bytes: int | None = None):
        self.backend, self.spec, self.report = backend, spec, report
        self.history_turns = history_turns
        if max_prompt_bytes is None:
            seq = getattr(backend, "seq_len", None)
            if isinstance(seq, int) and not isinstance(seq, bool) and seq > ANSWER_TOKENS:
                max_prompt_bytes = int((seq - ANSWER_TOKENS) * PROMPT_BYTES_PER_TOKEN)
        self.max_prompt_bytes = max_prompt_bytes
        self.guard = (is_aero(spec, report) or is_steel(spec, report)) if guard is None else guard
        self.retries = 1 if is_steel(spec, report) else 0   # steel: one guarded rewrite before the template
        self.retried: dict[str, Any] | None = None
        self.system = system_prompt(spec, report)
        self.last: dict[str, Any] = {}
        self.log: list[dict[str, Any]] = []
        if is_steel(spec, report):
            content = (f"INSPECTION BRIEF:\n{steel_brief(report, spec)}\n\n"
                       "In 2 or 3 sentences, tell the operator why the part got this verdict, "
                       "and the likely cause and prevention.")
        else:
            content = (f"INSPECTION REPORT:\n{compact_for_llm(report)}\n\nSOP NOTES:\n{sop_notes(report, spec)}\n\n"
                       "Explain this result to the operator and say what to do next.")
        self.first = {"role": "user", "content": content}
        self.turns: list[dict[str, str]] = []

    def _messages(self, question: str | None) -> list[dict[str, str]]:
        # The report message always stays; older Q&A is dropped to bound prompt length.
        recent = self.turns[-2 * self.history_turns :] if self.history_turns else []
        tail = [{"role": "user", "content": question}] if question is not None else []
        msgs = [{"role": "system", "content": self.system}, self.first, *recent, *tail]
        while recent and self.max_prompt_bytes and len(format_llama3(msgs).encode()) > self.max_prompt_bytes:
            recent = recent[1:] if recent[0]["role"] == "assistant" else recent[2:]   # oldest turn first
            msgs = [{"role": "system", "content": self.system}, self.first, *recent, *tail]
        return msgs

    def _run(self, messages: list[dict[str, str]], first: bool = False, opening: str = "") -> str:
        context = {"report": self.report, "spec": self.spec}
        model = getattr(self.backend, "name", type(self.backend).__name__)
        answered_by, problems, rejected = model, [], None
        try:
            reply = self.backend.generate(messages, context)
        except Exception as exc:  # keep the station usable if the model fails
            fallback = MockBackend().generate(messages, context)
            reply = f"{fallback}\n\n(Assistant model unavailable: {exc}. Showing a template answer.)"
            answered_by, problems = MockBackend.name, [f"backend error: {exc}"]
        else:
            if is_steel(self.spec, self.report):
                reply = _whole_sentences(PREAMBLE_RE.sub("", reply.strip(), count=1))
                if opening and not reply.startswith(opening):
                    reply = f"{opening} {reply}"
            if self.guard:
                question = messages[-1]["content"] if messages[-1]["role"] == "user" and len(messages) > 2 else ""
                problems = check_reply(reply, self.report, self.spec, context=question, first=first)
                for _ in range(self.retries if problems and not isinstance(self.backend, MockBackend) else 0):
                    # one guarded rewrite: the model is told which rule its answer broke
                    fix = {"role": "user", "content": "Your answer broke these rules: " + "; ".join(problems)
                           + ". Write it again, 2 or 3 sentences, without that. Use only the brief."}
                    base = [m for m in messages if not m.get("prefill")]
                    tail = [{"role": "assistant", "content": opening, "prefill": True}] if opening else []
                    retry = self.backend.generate([*base, {"role": "assistant", "content": reply}, fix, *tail], context)
                    if opening and not retry.strip().startswith(opening):
                        retry = f"{opening} {retry.strip()}"
                    if is_steel(self.spec, self.report):
                        retry = _whole_sentences(PREAMBLE_RE.sub("", retry.strip(), count=1))
                    first_problems = problems
                    problems = check_reply(retry, self.report, self.spec, context=question, first=first)
                    self.retried = {"problems": first_problems, "rejected": reply}
                    reply = retry
                    if not problems:
                        break
                if problems:
                    rejected = reply
                    reply = (MockBackend().generate(messages, context)
                             + "\n\n(Template answer: the model reply failed the guardrail check.)")
                    answered_by = MockBackend.name
        self.last = {"backend": answered_by, "model": model, "problems": problems, "rejected": rejected}
        if self.retried:
            self.last["retried"], self.retried = self.retried, None
        self.log.append(self.last)
        return reply.strip()

    @property
    def answered_by(self) -> str | None:
        """Which backend gave the last answer (the model's name, or the template on a fallback)."""
        return self.last.get("backend")

    def explain(self) -> str:
        if not is_steel(self.spec, self.report) or isinstance(self.backend, MockBackend):
            reply = self._run(self._messages(None))     # the template is already the whole answer
        else:
            # The verdict and steps come verbatim from the rules engine; the model adds only the why
            # and the prevention (checked by the guardrail). A failed check gives the full template.
            head = steel_header(self.report, self.spec)
            if self.report["verdict"] in ("RETAKE", "PASS"):   # nothing to explain: the rules engine answers
                self.last = {"backend": "rules-engine", "model": None, "problems": [], "rejected": None}
                self.log.append(self.last)
                self.turns = [{"role": "assistant", "content": head}]
                return head
            listener = getattr(self.backend, "listener", None)
            opening = cause_opening(self.report)
            if listener is not None:
                self.backend.listener = lambda t: listener(f"{head}\n\n{CAUSE_LABEL}{opening} {t}")
            try:
                para = self._run([{"role": "system", "content": self.system},
                                  {"role": "user", "content": cause_brief(self.report, self.spec)},
                                  {"role": "assistant", "content": opening, "prefill": True}], first=True,
                                 opening=opening)
            finally:
                if listener is not None:
                    self.backend.listener = listener
            reply = para if self.last.get("backend") == MockBackend.name else f"{head}\n\n{CAUSE_LABEL}{para}"
        self.turns = [{"role": "assistant", "content": reply}]
        return reply

    def ask(self, question: str) -> str:
        if not self.turns:
            self.explain()
        if (is_steel(self.spec, self.report) and not isinstance(self.backend, MockBackend)
                and RULES_QUESTION_RE.search(question)):
            # repair steps and override requests get the rules engine's exact answer (on the Pi the 1B
            # model refused them or invented steps); "why" and open questions go to the model
            reply = answer_steel(question, self.report, self.spec)
            self.last = {"backend": "rules-engine", "model": None, "problems": [], "rejected": None}
            self.log.append(self.last)
        else:
            reply = self._run(self._messages(question))
        self.turns += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        return reply


# ---- template answers (mock backend and fallback) ---------------------------------------


def _why(d: dict[str, Any]) -> str:
    ev = d["evidence"]
    if d["type"].endswith("_indication"):
        return (
            f"{d['id']} is a {d['type'].replace('_', ' ')}: {ev['basis']}. "
            f"It is {d['size']['length_mm']} mm long and {d['size']['width_mm']} mm wide (aspect ratio {ev['aspect_ratio']}), redness {ev['redness']}."
        )
    c = ev["contrast"]
    depth = f", depth {d['depth_mm']:+.2f} mm from the 3D map" if d.get("depth_mm") is not None else ""
    return (
        f"{d['id']} is classed as a {d['type']}: {ev['basis']}. "
        f"Brightness change: top {c['top_rgb']} (red/green/blue), 45-degree camera {c['oblique45']}, grazing camera {c['grazing']}; "
        f"strongest at {ev['strongest']}, colour shift {ev['color_shift']}{depth}."
    )


def explain(report: dict[str, Any]) -> str:
    defects = report["defects"]
    penetrant = report.get("mode") == "penetrant"
    if not defects:
        what = "No penetrant indications found" if penetrant else "No surface defects found"
        return f"Verdict: {report['verdict']}. {what}. {report['verdict_instruction']}"
    noun = "indication(s)" if penetrant else "defect(s)"
    lines = [f"Verdict: {report['verdict']}. {len(defects)} {noun} found."]
    for d in defects:
        loc = d["location_mm"]
        kind = d["type"].replace("_", " ")
        size = f", {d['size']['length_mm']} mm long" if penetrant else ""
        lines.append(
            f"- {d['id']}: {kind}{size} ({d['confidence']:.0%} confidence) at x={loc['x']} mm, y={loc['y']} mm "
            f"({d['zone']}), {d['severity']} -> {d['disposition']}. Rule: {d['rule']}."
        )
    lines.append("What to do:")
    steps = [*report["actions"], report["verdict_instruction"]]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    return "\n".join(lines)


def _causes(d: dict[str, Any], report: dict[str, Any], spec: dict[str, Any]) -> tuple[str, str]:
    if d["type"].endswith("_indication"):
        kind = d["type"].split("_")[0]
        pen = spec.get("penetrant", {})
        return ", ".join(pen.get("likely_causes", {}).get(kind, [])) or "not known", pen.get("prevention", {}).get(kind, "")
    rule = spec["defects"].get(d["type"], {})
    return ", ".join(rule.get("likely_causes", [])) or "not known", rule.get("prevention", "")


def answer(question: str, report: dict[str, Any], spec: dict[str, Any]) -> str:
    q = question.lower()
    defects = report["defects"]
    ids = re.findall(r"\b[di]\d+\b", q)
    named = [d for d in defects if d["type"].split("_")[0] in q]
    chosen = [d for d in defects if d["id"].lower() in ids] or named or defects[:1]
    if not defects:
        return f"Nothing was found on {report['part_id']}. {report['verdict_instruction']}"
    if any(w in q for w in ("override", "accept", "ship", "pass it", "ignore")):
        return f"The verdict {report['verdict']} is final and set by the QA rules. If you disagree, call the QA supervisor."
    if any(w in q for w in ("cause", "why did", "how did", "prevent", "avoid")):
        out = []
        for d in chosen:
            causes, prevention = _causes(d, report, spec)
            out.append(f"{d['id']} ({d['type'].replace('_', ' ')}): likely causes: {causes}. Prevention: {prevention}")
        return "\n".join(out)
    if any(w in q for w in ("fix", "rework", "repair", "do now", "next", "what should")):
        return "\n".join(f"{d['id']}: {d['action']}" for d in chosen) + f"\n{report['verdict_instruction']}"
    if any(w in q for w in ("why", "how do you know", "sure", "evidence", "not a")):
        return "\n".join(_why(d) for d in chosen)
    if any(w in q for w in ("where", "size", "how big", "long", "location", "deep")):
        return "\n".join(
            f"{d['id']}: x={d['location_mm']['x']} mm, y={d['location_mm']['y']} mm from the top-left corner, "
            f"length {d['size']['length_mm']} mm, area {d['size']['area_mm2']} mm2, zone {d['zone']}"
            + (f", depth {d['depth_mm']:+.2f} mm" if d.get("depth_mm") is not None else "")
            + "."
            for d in chosen
        )
    return explain(report) + "\n\n(Template mode: free-form questions need the Llama backend.)"


# ---- aerospace guardrail ----------------------------------------------------------------

NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
VERDICT_RE = re.compile(r"\b(" + "|".join(v.replace("_", "[_ ]") for v in VERDICTS) + r")\b")
SOFT_PASS = (
    r"\b(?:the |this )?part (?:passes|passed|is (?:fine|ok|okay|good|conforming|serviceable))\b",
    r"\b(?:ok|okay|safe|good|fine) to (?:ship|use|fly|deliver|install)\b",
    r"\bship (?:it|the part)\b",
    r"\bno (?:action|rework|nonconformance)s? (?:is |are )?(?:needed|required|found)\b",
)


def _numbers(text: str) -> set[float]:
    return {round(float(m), 6) for m in NUM_RE.findall(text)}


def _phrase_re(phrase: str) -> re.Pattern:
    words = [re.escape(w) for w in re.split(r"[\s\-]+", phrase.strip()) if w]
    return re.compile(r"(?<!\w)" + r"[\s\-]+".join(words) + r"(?!\w)", re.IGNORECASE)


def check_reply(text: str, report: dict[str, Any], spec: dict[str, Any], context: str = "",
                first: bool = False) -> list[str]:
    """Problems with an advisor reply (empty list = OK).

    - contradicts the verdict: names a verdict that is neither the part's nor any item's
      disposition, or says the part passes / can ship while the verdict is not PASS;
    - suggests use-as-is outside the customer MRB;
    - uses a banned phrase or a reserved decision word (spec banned_phrases, reserved_words);
    - quotes a number that is in neither the report, the SOP notes nor the question. Counts and
      step numbers 0-9, percentages of fractions in the report and numbers inside the spec's
      reference ids are allowed.
    Steel records (3.0) get check_steel_reply instead.
    """
    if is_steel(spec, report):
        return check_steel_reply(text, report, spec, context, first=first)
    problems = []
    verdict = report["verdict"]
    allowed_v = {verdict, *(i["disposition"] for i in [*report.get("defects", []), *report.get("findings", [])])}
    for m in VERDICT_RE.findall(text):
        v = m.replace(" ", "_")
        if v not in allowed_v:
            problems.append(f"contradicts the verdict: says {v}, the report says {verdict}")
    if verdict != "PASS":
        for pat in SOFT_PASS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                problems.append(f"contradicts the verdict {verdict}: '{m.group(0)}'")
    for sentence in re.split(r"(?<=[.!?\n])\s+", text):
        if re.search(r"use[\s\-]+as[\s\-]+is", sentence, re.IGNORECASE) and "MRB" not in sentence:
            problems.append("mentions use-as-is outside the customer MRB")
    for phrase in [*spec.get("banned_phrases", []), *spec.get("reserved_words", [])]:
        if _phrase_re(phrase).search(text):
            problems.append(f"banned phrase: '{phrase}'")

    ref_ids = sorted(spec.get("references", {}), key=len, reverse=True)
    stripped = text
    for rid in ref_ids:
        stripped = re.sub(re.escape(rid), " ", stripped, flags=re.IGNORECASE)
    allowed = (_numbers(json.dumps(report, default=str)) | _numbers(sop_notes(report, spec)) | _numbers(context)
               | _numbers(" ".join(ref_ids)) | {float(k) for k in range(10)})
    allowed |= {float(f(x * 100)) for x in list(allowed) if 0 < x <= 1 for f in (math.floor, math.ceil)}
    invented = sorted({m for m in NUM_RE.findall(stripped) if round(float(m), 6) not in allowed}, key=float)
    if invented:
        problems.append(f"numbers not in the report: {', '.join(invented)}")
    return problems


# ---- aerospace template answers (mock backend and fallback) -----------------------------


def _items(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [*report.get("findings", []), *report["defects"]]


def _line(it: dict[str, Any]) -> str:
    refs = f" [{', '.join(it['reference'])}]" if it.get("reference") else ""
    if "fasteners" in it:
        what = f"{it['type'].replace('_', ' ')} at {', '.join(it['fasteners'])}"
    else:
        loc = it["location_mm"]
        what = (f"{it['type'].replace('_', ' ')} ({it['confidence']:.0%} confidence) "
                f"at x={loc['x']} mm, y={loc['y']} mm")
    return f"- {it['id']}: {what}, {it['severity']} -> {it['disposition']}. Rule: {it['rule']}.{refs}"


def explain_aero(report: dict[str, Any]) -> str:
    defects, findings = report["defects"], report.get("findings", [])
    head = f"Verdict: {report['verdict']}."
    if not defects and not findings:
        return f"{head} No surface defects or fastener findings. {report['verdict_instruction']}"
    lines = [f"{head} {len(defects)} surface defect(s), {len(findings)} fastener finding(s)."]
    lines += [_line(it) for it in _items(report)]
    lines.append("What to do:")
    steps = [*report["actions"][:8], report["verdict_instruction"]]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    lines.append("Decision support only: the qualified inspector confirms against the drawing.")
    return "\n".join(lines)


def _aero_rule(it: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if "fasteners" in it:
        return spec.get("fastener_rules", {}).get("findings", {}).get(it["type"], {})
    return spec["defects"].get(it["type"], {})


def answer_aero(question: str, report: dict[str, Any], spec: dict[str, Any]) -> str:
    q = question.lower()
    items = _items(report)
    verdict = report["verdict"]
    if (any(w in q for w in ("override", "accept", "ship", "pass it", "ignore", "release", "sign off", "airworth",
                             "certif")) or re.search(r"\bas[\s-]+is\b", q)):
        return (f"The verdict {verdict} is final and set by the rules from the demo table; the assistant cannot change it. "
                "Only the qualified inspector decides, and use-as-is or repair only by the customer MRB. Call QA.")
    if any(w in q for w in ("ndt", "eddy", "under the head", "subsurface", "intergranular", "depth", "deep")):
        out = "; ".join(report.get("out_of_scope", []))
        return f"The camera cannot decide these; refer them to NDT by a qualified NAS 410 / EN 4179 Level 2: {out}."
    if not items:
        return f"Nothing was found on {report['part_id']}. {report['verdict_instruction']}"
    ids = re.findall(r"\b[dfh]\d+\b", q)
    chosen = [it for it in items if it["id"].lower() in ids or any(f.lower() in ids for f in it.get("fasteners", []))]
    chosen = chosen or [it for it in items if it["type"].replace("_", " ") in q or it["type"].split("_")[0] in q]
    chosen = chosen or items[:1]
    if any(w in q for w in ("cause", "why did", "how did", "prevent", "avoid")):
        out = []
        for it in chosen:
            rule = _aero_rule(it, spec)
            causes = ", ".join(rule.get("likely_causes", [])) or "not known"
            out.append(f"{it['id']} ({it['type'].replace('_', ' ')}): likely causes: {causes}. "
                       f"Prevention: {rule.get('prevention', '')}")
        return "\n".join(out)
    if any(w in q for w in ("fix", "rework", "repair", "scrap", "do now", "next", "what should", "ncr")):
        return "\n".join(f"{it['id']}: {it['action']}" for it in chosen) + f"\n{report['verdict_instruction']}"
    if any(w in q for w in ("why", "how do you know", "sure", "evidence", "rule", "reference", "limit")):
        return "\n".join(_line(it)[2:] for it in chosen)
    if any(w in q for w in ("where", "size", "how big", "long", "location", "distance", "pitch", "edge")):
        out = []
        for it in chosen:
            loc = it["location_mm"]
            if "fasteners" in it:
                out.append(f"{it['id']}: {', '.join(it['fasteners'])} at x={loc['x']} mm, y={loc['y']} mm; {it['rule']}.")
            else:
                size = ", ".join(f"{k.replace('_mm2', ' mm2').replace('_mm', ' mm').replace('_', ' ')} {v}"
                                 for k, v in it["size"].items() if v)
                out.append(f"{it['id']}: x={loc['x']} mm, y={loc['y']} mm from the datum corner; {size}.")
        return "\n".join(out)
    return explain_aero(report) + "\n\n(Template mode: free-form questions need the Llama backend.)"


# ---- steel and metal surfaces (record 3.0, configs/spec_steel.yaml) ---------------------------

CAPTURE_KEYWORDS = {   # capture.yaml gate.messages keys -> words in gate reasons or messages
    "too_bright": ("too_bright", "too bright", "r_ambient", "sat_frac_a", "ambient", "sun"),
    "saturated": ("saturat", "glare", "masked", "reflection"),
    "too_dark": ("too_dark", "too dark", "too far", "led_level"),
    "motion": ("motion", "hold still", "shake"),
    "blur": ("blur", "focus"),
    "changed": ("changed", "residual", "flicker", "shadow"),
}
DEFAULT_STEEL_BANNED = ("certified", "certify", "compliant", "approved", "guaranteed", "measures depth",
                        "works in any lighting", "zero defects")
STEEL_VERDICT_RE = re.compile(r"\b(" + "|".join(v.replace("_", "[_ ]") for v in (*VERDICTS, "RETAKE")) + r")\b")
STEEL_SOFT_PASS = (
    *SOFT_PASS,
    r"\bno (?:defects?|damage|problems?|issues?) (?:was |were |is |are )?(?:found|detected|seen)\b",
    r"\buse (?:it |the part )?as[\s\-]+is\b",
    r"\b(?:the |this )?part (?:is|was|has been|can be|may be|will be) (?:accepted|released)\b",
    r"\baccept (?:it|the part|this part)\b",
    r"\bOK bin\b",
    r"\bgood to go\b",
    r"\(\s*no action(?: needed| required)?\s*\)",
    r"\b(?:send|deliver|dispatch) (?:it|the part|this part) to the customer\b",
)
# Said while the verdict is not REJECT (the action texts say "or the part is scrapped" as an option,
# so "scrapped" is not matched).
STEEL_SOFT_REJECT = (
    r"\b(?:the |this )?part (?:is|was|has been|must be|should be|will be) rejected\b",
    r"\breject (?:it|the part|this part)\b",
    r"\bREJECT bin\b",
)
# "verdict is X" / "Verdict: X" / "**Verdict:** X" names the PART's verdict, so a defect disposition
# does not excuse it.
VERDICT_CLAIM_RE = re.compile(r"\bverdict\b[\s*:=\-]*(?:(?:is|was|remains|stays)\b[\s*:]*)?['\"]?([A-Za-z_]+)",
                              re.IGNORECASE)
SEVERITY_WORD_RE = re.compile(r"\b(minor|major|critical)\b", re.IGNORECASE)
DEFECT_ID_RE = re.compile(r"\bD\d+\b")
VERDICT_WORDS = {"pass": "PASS", "passed": "PASS", "accept": "PASS", "accepted": "PASS", "ok": "PASS",
                 "okay": "PASS", "clean": "CLEAN_AND_REINSPECT", "clean_and_reinspect": "CLEAN_AND_REINSPECT",
                 "rework": "REWORK", "hold": "HOLD_FOR_QA", "held": "HOLD_FOR_QA", "hold_for_qa": "HOLD_FOR_QA",
                 "reject": "REJECT", "rejected": "REJECT", "scrap": "REJECT", "retake": "RETAKE"}
# A banned phrase is allowed right after a negation of it: at most 2 words between, no punctuation
# ("is not certified", "never say it works in any lighting"), so "No problem, the part is certified"
# and idioms such as "no doubt" or "not only" do not excuse it.
NEGATION_RE = re.compile(r"\b(?:not|never|cannot|can't|no|without|nor|neither|doesn't|don't|isn't|aren't|won't|"
                         r"didn't|wasn't|shouldn't)\s+"
                         r"(?:(?!(?:doubt|only|problem|problems|worry|worries|question|matter|hesitate|forget|wait)\b)"
                         r"[\w'\-]+\s+){0,2}$",
                         re.IGNORECASE)
# Standards bodies named without a number ("ASME Section VIII", "per ASTM"): never on the whitelist.
FOREIGN_DESIG_RE = re.compile(r"\b(ASTM|ASME|AWS|API|NACE|SSPC|DIN|JIS|SAE|ANSI|MIL|NAS|AMS|NORSOK|BIS)\b")
DESIG_RE = re.compile(r"\b(EN|ISO|IS|ASTM|ASME|IACS|BS|DIN|JIS|AWS|API|NACE|SSPC|SIS|IEC|SAE|ANSI|MIL|NAS|AMS|AS)"
                      r"(?:[\s\-]+Rec\.?)?[\s\-]*([A-Z]{0,3}[\s\-]?\d+(?:[\-/.]\d+)*)")
DETAIL_RE = re.compile(r"(?<!\w)(Table|Tbl\.?|cl\.|clause|§|Annex)\s*([A-Z]?\d+(?:\.\d+)*)", re.IGNORECASE)
HINDI_RE = re.compile("[\u0900-\u097F]")   # Devanagari
CAPTURE_WORDS = ("light", "lighting", "sun", "bright", "glare", "dark", "shadow", "retake", "capture", "photo",
                 "picture", "image", "blur", "hood", "shade", "reflection", "lux", "lamp", "window")

HI_VERDICT = {
    "PASS": "भाग ठीक है: इसे OK बिन में रखें। दर्ज की गई छोटी खामियों की मरम्मत नहीं चाहिए।",
    "CLEAN_AND_REINSPECT": "चिह्नित जगह साफ़ करें, भाग सुखाएँ, फिर दोबारा जाँच करें।",
    "REWORK": "भाग को रीवर्क के लिए भेजें (ब्लेंड, ग्राइंड या सफ़ाई), फिर दोबारा जाँच करें।",
    "HOLD_FOR_QA": "भाग को HOLD ट्रे में रखें। QA गेज या डाई पेनिट्रेंट से जाँच करेगा।",
    "REJECT": "भाग को लाल REJECT बिन में रखें, दोष ID का टैग लगाएँ और QA को बताएँ।",
    "RETAKE": "तस्वीर उपयोग लायक नहीं थी, इसलिए कोई दोष नहीं जाँचा गया। नीचे दिया उपाय करें और दोबारा जाँच करें।",
}
HI_TYPE = {"scratch": "खरोंच", "dent": "डेंट (दबा हुआ निशान)", "pit": "पिट (छोटे गड्ढे)", "corrosion": "जंग",
           "crack": "दरार", "stain": "दाग", "unknown": "अज्ञात निशान"}
HI_SEVERITY = {"minor": "मामूली", "major": "बड़ा", "critical": "गंभीर", "unclassified": "अवर्गीकृत"}


def _capture_keys(cap: dict[str, Any]) -> list[str]:
    """capture_advice reason keys that match the gate reasons or message."""
    text = " ".join([*(cap.get("gate_reasons") or []), cap.get("gate_message") or ""]).lower()
    return [k for k, words in CAPTURE_KEYWORDS.items() if any(w in text for w in words)]


def capture_advice(report: dict[str, Any], spec: dict[str, Any], general: bool = True) -> list[str]:
    """Lighting and handling advice from the record's capture block (empty unless flagged or RETAKE):
    the spec's general line for the gate (unless general=False), the gate message, then the advice
    for every other gate reason (the reason the message already covers is not repeated).

    too_dark is dropped next to too_bright or saturated: under sun or glare B and A both clip, so
    B - A (the LED level) collapses and "move to 200 mm" would be the wrong advice."""
    cap = report.get("capture") or {}
    gate = cap.get("gate", "PASS")
    if gate not in ("PASS_FLAGGED", "RETAKE") and report.get("verdict") != "RETAKE":
        return []
    adv = spec.get("capture_advice") or {}
    reasons = adv.get("reasons") or {}
    message = cap.get("gate_message") or ""
    covered = set(_capture_keys({"gate_message": message})) if message else set()
    keys = [k for k in _capture_keys({"gate_reasons": cap.get("gate_reasons") or []}) if k not in covered]
    if not message and not keys:
        keys = _capture_keys(cap)
    if {"too_bright", "saturated"} & (covered | set(keys)):
        keys = [k for k in keys if k != "too_dark"]
    out = [adv.get(gate if gate in ("PASS_FLAGGED", "RETAKE") else "RETAKE", "") if general else "", message]
    out += [reasons[k] for k in keys if k in reasons]
    return [x for x in dict.fromkeys(out) if x]


def _plain(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _capture_numbers(cap: dict[str, Any]) -> str:
    parts = []
    if cap.get("r_ambient") is not None:
        parts.append(f"ambient-to-LED ratio {cap['r_ambient']}")
    if cap.get("lux_est") is not None:
        parts.append(f"about {_plain(cap['lux_est'])} lux ambient")
    if cap.get("motion_px") is not None:
        parts.append(f"motion {cap['motion_px']} px")
    if cap.get("sat_pct") is not None:
        parts.append(f"saturated {cap['sat_pct']} %")
    if cap.get("blur") is not None:
        parts.append(f"sharpness {cap['blur']}")
    return ", ".join(parts)


def _steel_sop_notes(report: dict[str, Any], spec: dict[str, Any]) -> str:
    """Capture advice (flagged or RETAKE), then per shown defect type the first likely cause and the
    prevention, and the one-line disclaimer. Kept short: the Pi's Llama context is 2048 tokens."""
    lines = []
    advice = capture_advice(report, spec)
    if advice:
        lines.append("- capture: " + " ".join(advice))
    shown, _ = llm_items(report)
    classes = spec.get("classes") or {}
    for kind in dict.fromkeys(d["type"] for d in shown):
        rule = classes.get(kind, {})
        cause = (rule.get("likely_causes") or ["unknown"])[0]
        lines.append(f"- {kind}: likely cause: {cause}. Prevention: {rule.get('prevention', '')}")
    if not shown and report.get("verdict") != "RETAKE":
        lines.append("- no defects found")
    lines.append("- thresholds are illustrative, to be agreed with the customer; the camera does not measure depth")
    return "\n".join(lines)


def _flex(literal: str) -> re.Pattern:
    """A literal string as a regex with flexible whitespace, case-insensitive."""
    return re.compile(r"\s+".join(re.escape(w) for w in literal.split()), re.IGNORECASE)


def _negated(text: str, start: int) -> bool:
    return bool(NEGATION_RE.search(text[max(0, start - 50):start]))


def _desig_key(m: re.Match) -> str:
    return f"{m.group(1)} {' '.join(m.group(2).split())}".upper()


def _detail_key(m: re.Match) -> str:
    word = m.group(1).lower().rstrip(".")
    word = {"tbl": "table", "clause": "cl"}.get(word, word)
    return f"{word} {m.group(2).upper()}"


def steel_whitelist(report: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Reference ids the advisor may cite: the spec's references plus the record's."""
    return list(dict.fromkeys([*(spec.get("references") or {}), *(r["id"] for r in report.get("references", []))]))


def _allowed_numbers(*texts: str) -> set[float]:
    allowed: set[float] = set()
    for t in texts:
        allowed |= _numbers(t)
    allowed |= {float(k) for k in range(10)}
    allowed |= {float(f(x * 100)) for x in list(allowed) if 0 < x <= 1 for f in (math.floor, math.ceil)}
    allowed |= {round(x, 1) for x in list(allowed)} | {float(round(x)) for x in list(allowed)}
    return allowed


def _severity_problems(text: str, report: dict[str, Any]) -> list[str]:
    """'D1 is a minor scratch' when D1 is major: per sentence, the first severity word after a defect
    id (up to the next id) must be that defect's severity ('not minor' is skipped)."""
    sev = {d["id"]: d.get("severity") for d in report.get("defects", [])}
    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        ids = list(DEFECT_ID_RE.finditer(sentence))
        for k, m in enumerate(ids):
            if m.group(0) not in sev:
                continue
            end = ids[k + 1].start() if k + 1 < len(ids) else len(sentence)
            seg = sentence[m.end():end]
            w = SEVERITY_WORD_RE.search(seg)
            if w and w.group(1).lower() != sev[m.group(0)] and not re.search(r"\bnot\s+(?:an?\s+)?$", seg[:w.start()]):
                problems.append(f"contradicts the severity: says {m.group(0)} is {w.group(1).lower()}, "
                                f"the record says {sev[m.group(0)]}")
    return problems


def check_steel_reply(text: str, report: dict[str, Any], spec: dict[str, Any], context: str = "",
                      first: bool = False) -> list[str]:
    """Problems with a steel-mode advisor reply (empty list = OK).

    - contradicts the verdict: names a verdict that is neither the record's nor any defect's
      disposition (the capture gate may be named next to 'capture'/'gate'); says "verdict is X" /
      "Verdict: X" with X not the record's verdict (a defect disposition does not excuse it); says
      the part passes, is accepted, goes to the OK bin, can ship, had no defects or can be used as
      is while the verdict is not PASS; or says the part is rejected while the verdict is not REJECT;
    - gives a defect id another severity than the record's ('D1 is minor' when D1 is major);
    - uses a banned phrase (spec banned_phrases; allowed right after a negation of it);
    - cites a standard, table or clause that is not on the reference whitelist (spec + record), or
      names a standards body that is not on it (ASTM, ASME, ...);
    - quotes a number that is in neither the record, the SOP notes, the system prompt nor the
      question (0-9, percentages of fractions and 0/1-decimal roundings are allowed);
    - names a defect type that is in neither the record nor the question ("crack" on a scratch-only
      record), or gives a defect id another disposition than the record's;
    - states a depth, or a depth rule, that is not a verbatim part of the record (the Phase 1 camera
      measures no depth: "a depth of 0.2 mm", an inverted "0.2 mm or less, repair");
    - first answer only (first=True, the model's cause note under the rules engine's header): has a
      number, or does not name any of the defects that decide the verdict.
    """
    problems: list[str] = []
    verdict = report["verdict"]
    cap = report.get("capture") or {}
    gate = str(cap.get("gate") or "")
    gate_token = "PASS" if gate.startswith("PASS") else gate
    allowed_v = {verdict, *(d["disposition"] for d in report.get("defects", []))}

    def about_capture(start: int) -> bool:
        return bool(re.search(r"capture|gate|image|photo|picture|quality", text[max(0, start - 30):start].lower()))

    for m in STEEL_VERDICT_RE.finditer(text):
        v = m.group(1).replace(" ", "_")
        if v in allowed_v or (v == gate_token and about_capture(m.start())):
            continue
        problems.append(f"contradicts the verdict: says {v}, the record says {verdict}")
    for m in VERDICT_CLAIM_RE.finditer(text):
        v = VERDICT_WORDS.get(m.group(1).lower())
        if v is not None and v != verdict and not (v == gate_token and about_capture(m.start())):
            problems.append(f"contradicts the verdict {verdict}: '{m.group(0)}'")
    soft = [*(STEEL_SOFT_PASS if verdict != "PASS" else ()), *(STEEL_SOFT_REJECT if verdict != "REJECT" else ())]
    for pat in soft:   # "No, you cannot ship it" answers the question correctly
        m = next((m for m in re.finditer(pat, text, re.IGNORECASE) if not _negated(text, m.start())), None)
        if m:
            problems.append(f"contradicts the verdict {verdict}: '{m.group(0)}'")
    if verdict == "HOLD_FOR_QA":   # held parts are decided by QA, not by the advisor
        m = re.search(r"\b(?:requires?|needs?|must (?:be|have))\s+(?:an?\s+)?(?:(?:immediate|urgent|further)\s+)?(?:repair|scrap|rework|to be (?:repaired|scrapped))\b",
                      text, re.IGNORECASE)
        if m and not _negated(text, m.start()):
            problems.append(f"contradicts the verdict HOLD_FOR_QA (QA decides): '{m.group(0)}'")
    if REFUSAL_RE.search(text):   # the 1B model sometimes refuses "how to fix" or "can I ship it"
        problems.append("the model refused to answer")
    m = re.search(r"\b(?:verdict|result|decision)\s+(?:is|was)\s+(?:incorrect|wrong|a mistake|mistaken)\b", text, re.IGNORECASE)
    if m and not _negated(text, m.start()):
        problems.append(f"contradicts the verdict {verdict}: '{m.group(0)}'")
    problems += _severity_problems(text, report)
    for phrase in spec.get("banned_phrases") or DEFAULT_STEEL_BANNED:
        if any(not _negated(text, m.start()) for m in _phrase_re(phrase).finditer(text)):
            problems.append(f"banned phrase: '{phrase}'")

    whitelist = steel_whitelist(report, spec)
    stripped = text
    for rid in sorted(whitelist, key=len, reverse=True):
        stripped = _flex(rid).sub(" ", stripped)
    allowed_des = {_desig_key(m) for t in [*whitelist, spec.get("disclaimer", "")] for m in DESIG_RE.finditer(t)}
    allowed_det = {_detail_key(m) for t in whitelist for m in DETAIL_RE.finditer(t)}
    bad = [m.group(0).strip() for m in DESIG_RE.finditer(stripped) if _desig_key(m) not in allowed_des]
    bad += [m.group(0).strip() for m in DETAIL_RE.finditer(stripped) if _detail_key(m) not in allowed_det]
    bodies = {k.split()[0] for k in allowed_des}
    bad += [m.group(1) for m in FOREIGN_DESIG_RE.finditer(stripped)
            if m.group(1) not in bodies and not any(b.startswith(m.group(1)) for b in bad)]
    if bad:
        problems.append(f"reference not on the whitelist: {', '.join(dict.fromkeys(bad))}")

    allowed = _allowed_numbers(json.dumps(report, default=str, ensure_ascii=False), sop_notes(report, spec), context,
                               " ".join(whitelist), system_prompt(spec, report))
    plain = re.sub(r"(?<=\d),(?=\d{3}\b)", "", stripped)
    invented = sorted({m for m in NUM_RE.findall(plain) if round(float(m), 6) not in allowed}, key=float)
    if invented:
        problems.append(f"numbers not in the record: {', '.join(invented)}")
    problems += _steel_fact_problems(text, report, spec, context, first)
    return problems


DEFECT_WORD_RE = {k: re.compile(rf"\b{k}(?:s|es|ed|ing)?\b", re.IGNORECASE)
                  for k in ("scratch", "dent", "pit", "corrosion", "crack", "stain")}
DEFECT_WORD_RE["corrosion"] = re.compile(r"\b(?:corrosion|rust(?:y|ed|ing)?)\b", re.IGNORECASE)
DEPTH_WORD_RE = re.compile(r"\b(?:depth|deep)\b", re.IGNORECASE)


def _norm(t: str) -> str:
    t = t.lower().replace("\u2013", "-").replace("\u2014", "-").replace("\u2264", "<=")
    t = re.sub(r"[*_`\"\u201c\u201d']", "", t)
    return " ".join(t.split())


def _steel_fact_problems(text: str, report: dict[str, Any], spec: dict[str, Any], context: str,
                         first: bool) -> list[str]:
    problems: list[str] = []
    verdict = report["verdict"]
    defects = report.get("defects", [])
    source = json.dumps(report, default=str, ensure_ascii=False) + "\n" + steel_brief(report, spec)
    src_norm = _norm(source.replace("\\n", " "))
    named = " ".join([steel_brief(report, spec), context,
                      *(f"{d['type']} {d.get('rule', '')} {d.get('action', '')}" for d in defects)])
    in_source = {k for k, rx in DEFECT_WORD_RE.items() if rx.search(named)}
    for k, rx in DEFECT_WORD_RE.items():
        if k in in_source:
            continue
        if any(not _negated(text, m.start()) for m in rx.finditer(text)):
            problems.append(f"names a defect that is not in the record: {k}")
    disp = {d["id"]: d["disposition"] for d in defects}
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        ids = list(DEFECT_ID_RE.finditer(sentence))
        for k, m in enumerate(ids):
            if m.group(0) not in disp:
                continue
            end = ids[k + 1].start() if k + 1 < len(ids) else len(sentence)
            seg = sentence[m.end():end]
            w = STEEL_VERDICT_RE.search(seg)
            if (w and w.group(1).replace(" ", "_") != disp[m.group(0)]
                    and not re.search(r"\b(?:part|verdict|so|therefore|overall)\b", seg[:w.start()], re.IGNORECASE)):
                problems.append(f"contradicts the disposition: says {m.group(0)} is {w.group(1)}, "
                                f"the record says {disp[m.group(0)]}")
    for clause in re.split(r"[.;:!?\n]+(?:\s+|$)|,\s+(?=(?:and|but|so|which|the|then)\b)", text):
        if not (DEPTH_WORD_RE.search(clause) and NUM_RE.search(re.sub(r"^\W*\d+\W*\s", "", clause))):
            continue
        c = _norm(re.sub(r"^\W*\d+[.)]\s*", "", clause.strip()))
        if c and c not in src_norm:
            problems.append(f"states a depth or depth rule that is not in the record: '{clause.strip()[:80]}'")
    if first:
        if NUM_RE.search(re.sub(r"\bD\d+\b", "", text)):
            problems.append("numbers in the cause note (the rules engine states the measurements)")
        deciding = _deciding(report)
        if deciding and not any(d["id"] in text or DEFECT_WORD_RE.get(d["type"], re.compile(re.escape(d["type"]),
                                re.IGNORECASE)).search(text) for d in deciding):
            names = ", ".join(f"{d['id']} {d['type']}" for d in deciding)
            problems.append(f"does not name the defect that decides the verdict: {names}")
    return problems


def _split_steps(action: str) -> list[str]:
    """'1. Do X. 2. Do Y to St 3. 3. Z' -> ['Do X.', 'Do Y to St 3.', 'Z']: a number starts a step only
    when it is the next step number; a 'Rule action:' prefix after an uncertain status is dropped."""
    action = action.split(" Rule action: ", 1)[-1] if " Rule action: " in action else action
    if not re.match(r"\s*1\.\s", action):
        return [action.strip()]
    steps, k, rest = [], 1, re.sub(r"^\s*1\.\s+", "", action)
    while True:
        m = re.search(rf"\s{k + 1}\.\s+", rest)
        if not m:
            steps.append(rest.strip())
            return [x for x in steps if x]
        steps.append(rest[:m.start()].strip())
        rest, k = rest[m.end():], k + 1


RULES_QUESTION_RE = re.compile(r"\b(?:fix|repair|rework|steps?|ship|accept|override|ignore|paint(?:ed)? over|pass it|"
                               r"sign off|release|use it as[\s\-]+is)\b", re.IGNORECASE)
REFUSAL_RE = re.compile(r"\bI (?:can(?:no|')t|cannot|am (?:not able|unable) to|won't) (?:provide|help|give|assist|answer|offer)"
                        r"|\bIs there anything else I can help you with\b", re.IGNORECASE)
PREAMBLE_RE = re.compile(r"^(?:here (?:is|are)|sure|okay|ok)\b[^\n:]{0,60}:\s*", re.IGNORECASE)
VERDICT_MEANING = {
    "HOLD_FOR_QA": "the part waits in the HOLD tray until QA checks it with a gauge or a second look; QA alone decides what happens next.",
    "REJECT": "the part does not go on; QA confirms and decides repair or scrap.",
    "REWORK": "the part is repaired as in the steps, then inspected again.",
    "CLEAN_AND_REINSPECT": "the surface is cleaned, then inspected again.",
    "PASS": "no finding needs action.",
}


def _plain_rule(d: dict[str, Any]) -> str:
    """A rule string in words: 'scratch: width_mm 0.88 >= 0.5 -> critical' -> 'width 0.88 mm, at or over
    the 0.5 mm limit'; the confidence clause becomes one 'detector unsure' note."""
    names = {"area_pct": ("area", "%"), "area_mm2": ("area", "mm2")}
    out = []
    for part in _short_rule(d).split(";"):
        part = re.sub(rf"^\s*{re.escape(d['type'])}:\s*", "", part.strip())
        if not part or part.startswith("confidence") or part.startswith("estimated rust grade"):
            continue
        part = re.sub(r"\s*->\s*[A-Za-z_]+\s*$", "", part)

        def cmp(m: re.Match) -> str:
            key, v, op, lim = m.groups()
            name, unit = names.get(key, (key.rsplit("_", 1)[0], key.rsplit("_", 1)[-1] if "_" in key else ""))
            word = "at or over" if op == ">=" else "over" if op == ">" else "under" if op == "<" else "within"
            return f"{name} {v} {unit}, {word} the {lim} {unit} limit".replace("  ", " ")

        part = re.sub(r"\b([a-z]+(?:_[a-z0-9]+)+) ([\d.]+) (>=|<=|>|<) ([\d.]+)", cmp, part)
        out.append(part.replace("_", " "))
    text = "; ".join(out)
    if d.get("status"):
        text += (f"; the detector is unsure ({d['confidence']:.0%} confidence), so the operator checks the spot"
                 + ("; a crack is still rejected" if d["disposition"] == "REJECT" else ""))
    return text


def _whole_sentences(text: str) -> str:
    """Drop a trailing fragment cut off by the token limit (keeps the text if it has no full sentence)."""
    t = text.rstrip()
    if not t or t[-1] in ".!?)\"'":
        return t
    cut = max(t.rfind(". "), t.rfind(".\n"), t.rfind("! "), t.rfind("? "))
    return t[:cut + 1] if cut > 0 else t


CAUSE_LABEL = "Likely cause and prevention (Llama): "


def cause_opening(report: dict[str, Any]) -> str:
    """The first words of the cause note, so it names the deciding defect and blames no one."""
    return f"The {_cause_kind(report) or 'defect'} most likely comes from"


def _cause_kind(report: dict[str, Any]) -> str | None:
    """The one defect type the cause note is about: the first deciding defect (on the Pi the 1B model
    gave two types one shared cause, 'scratch and corrosion ... from sliding')."""
    deciding = _deciding(report)
    return deciding[0]["type"] if deciding else None


def cause_brief(report: dict[str, Any], spec: dict[str, Any]) -> str:
    """The first-answer prompt: only the deciding defect types with their likely cause and prevention,
    no numbers. The rules engine writes the verdict, the measurements and the steps (steel_header);
    on the Pi the 1B model misquoted numbers and rules whenever it had to restate them."""
    classes = spec.get("classes") or {}
    verdict = report["verdict"]
    lines = ["CAUSE NOTE"]
    for kind in [k for k in [_cause_kind(report)] if k]:
        rule = classes.get(kind, {})
        lines.append(f"DEFECT: {kind} (verdict {verdict}: {VERDICT_MEANING.get(verdict, '')})")
        lines.append(f"LIKELY CAUSE: {(rule.get('likely_causes') or ['unknown'])[0]}")
        lines.append(f"PREVENTION: {rule.get('prevention', '')}")
    lines.append("Finish the sentence with the likely cause, then add one sentence on how to prevent it next time, "
                 "speaking to the operator as 'you'. No numbers.")
    return "\n".join(lines)


def steel_header(report: dict[str, Any], spec: dict[str, Any]) -> str:
    """The part of the first steel answer the rules engine writes: verdict, the deciding defects and
    the numbered steps, exactly as in the record."""
    verdict = report["verdict"]
    deciding = _deciding(report)
    if verdict == "RETAKE":
        lines = ["Verdict: RETAKE. The capture check refused this image, so no defects were judged."]
        steps = [*(capture_advice(report, spec, general=False) or ["Check the light and hold still."]),
                 "Inspect the part again."]
    else:
        if deciding:
            who = "; ".join(f"{d['id']} {d['type']} ({d['severity']}): {_plain_rule(d)}" for d in deciding)
            lines = [f"Verdict: {verdict}. Decided by {who}."]
        else:
            note = (report.get("capture") or {}).get("policy_note")
            lines = [f"Verdict: {verdict}." + (f" {note}" if note else
                     " No defects found." if not report.get("defects") else "")]
        steps = [x for a in report.get("actions", []) for x in _split_steps(a)] + [report["verdict_instruction"]]
    lines.append("What to do:")
    lines += [f"{i}. {x}" for i, x in enumerate(dict.fromkeys(steps), 1)]
    return "\n".join(lines)


def _deciding(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The defects whose disposition is the part's verdict (none for PASS, RETAKE or a policy hold)."""
    v = report["verdict"]
    return [] if v in ("PASS", "RETAKE") else [d for d in report.get("defects", []) if d["disposition"] == v]


def _short_rule(d: dict[str, Any]) -> str:
    return str(d.get("rule", "")).rsplit(" => ", 1)[0]


def steel_brief(report: dict[str, Any], spec: dict[str, Any]) -> str:
    """The steel record as a short fact sheet for the 1B model: the verdict, the deciding defect(s)
    with their rule, the steps to copy, at most 3 other findings without numbers, the capture and one
    cause and prevention line. About a third of the JSON's tokens, and ordered so the model starts
    from what decides the verdict."""
    verdict, cap = report["verdict"], report.get("capture") or {}
    lines = [f"VERDICT: {verdict}"]
    deciding = _deciding(report)
    if verdict == "RETAKE":
        lines.append("WHY: the capture check refused the image, so no defects were judged.")
        if cap.get("gate_message"):
            lines.append(f"CAPTURE: {cap['gate_message']}")
        steps = [*(capture_advice(report, spec, general=False) or ["Check the light and hold still."]),
                 "Inspect the part again."]
    else:
        if deciding:
            whys = []
            for d in deciding:
                whys.append(f"{d['id']} {d['type']}, {d['severity']}: {_plain_rule(d)}")
            lines.append("WHY: " + "; ".join(whys) + ".")
        elif cap.get("policy_note"):
            lines.append(f"WHY: {cap['policy_note']}")
        else:
            lines.append("WHY: no defects found." if not report.get("defects") else "WHY: no finding needs action.")
        steps = [x for a in report.get("actions", []) for x in _split_steps(a)]
        steps.append(report["verdict_instruction"])
    if verdict in VERDICT_MEANING:
        lines.append(f"MEANING: {verdict}: {VERDICT_MEANING[verdict]}")
    lines.append("STEPS:")
    lines += [f"{i}. {x}" for i, x in enumerate(dict.fromkeys(steps), 1)]
    others = [d for d in report.get("defects", []) if d not in deciding and d["disposition"] != "PASS"][:3]
    if others:
        lines.append("OTHER FINDINGS:")
        lines += [f"- {d['id']} {d['type']}, {d['severity']}, {d['disposition']}"
                  f"{' (' + d['status'] + ')' if d.get('status') else ''}" for d in others]
    n_pass = sum(d["disposition"] == "PASS" for d in report.get("defects", []))
    if n_pass:
        lines.append(f"ALSO: {n_pass} minor finding(s) within the limits.")
    if verdict != "RETAKE":
        advice = capture_advice(report, spec)
        lines.append(f"CAPTURE: {cap.get('device_mode', 'unknown')} mode, gate {cap.get('gate', 'PASS')}"
                     + (f". {' '.join(advice)}" if advice else "."))
    classes = spec.get("classes") or {}
    for d in (deciding or others)[:1]:
        rule = classes.get(d["type"], {})
        cause = (rule.get("likely_causes") or ["unknown"])[0]
        lines.append(f"CAUSE: {d['type']}: {cause}. Prevention: {rule.get('prevention', '')}")
    refs = list(dict.fromkeys(d["reference"] for d in deciding if d.get("reference")))
    if refs:
        lines.append("REFERENCE: " + "; ".join(refs))
    lines.append("NOTE: decision support only; thresholds are illustrative; the camera does not measure depth.")
    return "\n".join(lines)


def _size_text(d: dict[str, Any]) -> str:
    names = {"length_mm": "length {} mm", "width_mm": "width {} mm", "diameter_mm": "diameter {} mm",
             "area_mm2": "area {} mm2", "count": "{} pit(s)", "pit_d_mm": "largest pit {} mm"}
    return ", ".join(names[k].format(v) for k, v in (d.get("size") or {}).items() if k in names and v)


def _steel_line(d: dict[str, Any]) -> str:
    loc = d["location_mm"]
    size = _size_text(d)
    edge = f", {d['near_edge_mm']} mm from an edge or hole" if d.get("near_edge_mm") is not None else ""
    ref = f" [{d['reference']}]" if d.get("reference") else ""
    status = f" Status: {d['status']}." if d.get("status") else ""
    grade = f" Estimated rust grade {d['est_rust_grade']} (visual analogue)." if d.get("est_rust_grade") else ""
    return (f"- {d['id']}: {d['type']} ({d['confidence']:.0%} confidence) at x={loc['x']} mm, y={loc['y']} mm"
            f"{', ' + size if size else ''}{edge}; {d['severity']} ({d['tier']}) -> {d['disposition']}. "
            f"Rule: {d['rule']}.{ref}{grade}{status}")


def _steel_steps(report: dict[str, Any]) -> list[str]:
    """Per deciding action: the defect ids it covers and the numbered action text."""
    ids: dict[str, list[str]] = {}
    for d in report["defects"]:
        if d.get("action") in report.get("actions", []):
            ids.setdefault(d["action"], []).append(f"{d['id']} {d['type']}")
    return [f"{', '.join(ids.get(a, [])) or 'all'}: {a}" for a in report.get("actions", [])]


def explain_steel(report: dict[str, Any], spec: dict[str, Any] | None = None) -> str:
    """Template explanation of a steel record (mock backend and fallback)."""
    spec = spec or {}
    verdict, cap = report["verdict"], report.get("capture") or {}
    if verdict == "RETAKE":
        lines = ["Verdict: RETAKE. The capture check refused this image, so no defects were judged."]
        nums = _capture_numbers(cap)
        if nums:
            lines.append(f"Capture: {nums}.")
        lines.append("What to do:")
        steps = [*(capture_advice(report, spec, general=False) or ["Check the light and hold still."]),
                 "Inspect the part again."]
        lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
        return "\n".join(lines)
    defects = report["defects"]
    what = f"{len(defects)} finding(s)" if defects else "No defects found"
    lines = [f"Verdict: {verdict}. {what} on {report.get('material_name') or report.get('material')}, "
             f"{report.get('thickness_mm')} mm thick."]
    advice = capture_advice(report, spec)
    if advice:
        lines.append("Capture flagged: " + " ".join(advice))
    lines += [_steel_line(d) for d in defects]
    lines.append("What to do:")
    lines += _steel_steps(report)
    lines.append(f"Then: {report['verdict_instruction']}")
    lines.append("Decision support only: thresholds are illustrative and the camera does not measure depth.")
    return "\n".join(lines)


def explain_steel_hindi(report: dict[str, Any], spec: dict[str, Any] | None = None) -> str:
    """Short Hindi template; ids, numbers, units and verdict words stay in English."""
    verdict, defects = report["verdict"], report["defects"]
    if verdict == "RETAKE":
        lines = [f"निर्णय: RETAKE. {HI_VERDICT['RETAKE']}"]
        lines += [f"- {a}" for a in capture_advice(report, spec or {})]
        return "\n".join(lines)
    lines = [f"निर्णय: {verdict}. " + (f"{len(defects)} दोष मिले।" if defects else "कोई दोष नहीं मिला।")]
    for d in defects:
        loc = d["location_mm"]
        lines.append(f"- {d['id']}: {HI_TYPE.get(d['type'], d['type'])} ({d['type']}), "
                     f"{HI_SEVERITY.get(d['severity'], d['severity'])} ({d['severity']}) -> {d['disposition']}, "
                     f"x={loc['x']} mm, y={loc['y']} mm.")
    lines.append(f"क्या करें: {HI_VERDICT.get(verdict, report['verdict_instruction'])}")
    if report.get("actions"):
        lines.append("विस्तृत कदम (English):")
        lines += _steel_steps(report)
    return "\n".join(lines)


def _capture_answer(report: dict[str, Any], spec: dict[str, Any]) -> str:
    cap = report.get("capture") or {}
    nums = _capture_numbers(cap)
    gate = cap.get("gate", "PASS")
    head = {"PASS": "The capture check passed for this image",
            "PASS_FLAGGED": "The capture check flagged this image, but the result is valid",
            "RETAKE": "The capture check refused this image, so no defects were judged"}.get(gate, "Capture")
    lines = [f"{head}{': ' + nums if nums else ''}."]
    advice = capture_advice(report, spec, general=gate != "RETAKE")
    lines += [f"{i}. {a}" for i, a in enumerate(advice, 1)]
    if gate == "PASS":
        lines.append("The device lights the part with its own LEDs and subtracts the ambient frame; "
                     "if a later capture is refused, follow its message.")
    return "\n".join(lines)


def answer_steel(question: str, report: dict[str, Any], spec: dict[str, Any]) -> str:
    """Template answers for follow-up questions on a steel record."""
    q = question.lower()
    verdict, defects = report["verdict"], report["defects"]
    if (any(w in q for w in ("override", "accept", "ship", "pass it", "ignore", "certif", "approve", "sign off",
                             "paint over"))
            or re.search(r"\bas[\s\-]+is\b", q)):
        return (f"The verdict {verdict} is final and set by the rules; the assistant cannot change it. "
                "If you disagree, call QA.")
    if HINDI_RE.search(question) or "hindi" in q:
        return explain_steel_hindi(report, spec)
    if verdict == "RETAKE" or any(re.search(rf"\b{w}", q) for w in CAPTURE_WORDS):
        return _capture_answer(report, spec)
    if not defects:
        return f"Nothing was found on {report['part_id']}. {report['verdict_instruction']}"
    ids = re.findall(r"\bd\d+\b", q)
    chosen = ([d for d in defects if d["id"].lower() in ids] or [d for d in defects if d["type"] in q]
              or _deciding(report) or defects[:1])   # by default the defects that decide the verdict
    classes = spec.get("classes") or {}
    if any(w in q for w in ("depth", "deep")):
        out = ["The camera measures length, width, area and distance to an edge, not depth. The operator checks depth with a gauge."]
        out += [f"{d['id']} ({d['type']}): {d['action']}" for d in chosen]
        return "\n".join(out)
    if any(w in q for w in ("cause", "why did", "how did", "prevent", "avoid")):
        out = []
        for d in chosen:
            rule = classes.get(d["type"], {})
            causes = ", ".join(rule.get("likely_causes", [])) or "not known"
            out.append(f"{d['id']} ({d['type']}): likely causes: {causes}. Prevention: {rule.get('prevention', '')}")
        return "\n".join(out)
    if any(w in q for w in ("fix", "rework", "repair", "clean", "do now", "next", "what should", "what do")):
        return "\n".join(f"{d['id']} ({d['type']}): {d['action']}" for d in chosen) + f"\nThen: {report['verdict_instruction']}"
    if any(w in q for w in ("why", "how do you know", "sure", "evidence", "rule", "reference", "standard", "limit", "severity")):
        return "\n".join(_steel_line(d)[2:] for d in chosen)
    if any(w in q for w in ("where", "size", "how big", "long", "wide", "location", "edge")):
        out = []
        for d in chosen:
            loc = d["location_mm"]
            edge = f"; {d['near_edge_mm']} mm from an edge or hole" if d.get("near_edge_mm") is not None else ""
            out.append(f"{d['id']}: x={loc['x']} mm, y={loc['y']} mm from the datum corner; {_size_text(d)}{edge}.")
        return "\n".join(out)
    return explain_steel(report, spec) + "\n\n(Template mode: free-form questions need the Llama backend.)"


# ---- shift summary over the last N records -------------------------------------------------------

SHIFT_PROMPT = """You are Ayas, the quality-control assistant at a steel and metal inspection station. You get a SHIFT SUMMARY (JSON counts over the last inspections).
Summarise it for the operator in at most 6 short numbered lines: the verdict counts, the most common defect types and severities, retakes and their causes, and recurring locations with what to check there (fixtures, guides, handling, storage). Use only numbers from the summary. Never say certified, compliant, approved or guaranteed."""


def shift_summary(records: Iterable[dict[str, Any]], last_n: int | None = 50, cell_mm: float = 20.0,
                  min_parts: int = 2, edge_mm: float = 5.0) -> dict[str, Any]:
    """Counts over the last N inspection records (3.0; older records count by verdict and type).

    recurring = defect types seen in the same cell_mm x cell_mm square of the plate on at least
    min_parts different parts: a fixture, guide or handling problem at one spot.
    """
    recs = [r for r in records if r]
    if last_n:
        recs = recs[-last_n:]
    by_verdict, by_type, by_sev, gates, retake_msgs = Counter(), Counter(), Counter(), Counter(), Counter()
    by_type_sev: dict[str, Counter] = {}
    cells: dict[tuple[str, int, int], set[int]] = {}
    near_edge = uncertain = with_defects = 0
    for i, r in enumerate(recs):
        by_verdict[r.get("verdict", "?")] += 1
        cap = r.get("capture")
        if cap:
            gates[cap.get("gate", "PASS")] += 1
        if r.get("verdict") == "RETAKE":
            retake_msgs[(cap or {}).get("gate_message") or "capture refused"] += 1
        with_defects += bool(r.get("defects"))
        for d in r.get("defects", []):
            t, s = d.get("type", "?"), d.get("severity", "?")
            by_type[t] += 1
            by_sev[s] += 1
            by_type_sev.setdefault(t, Counter())[s] += 1
            loc = d.get("location_mm")
            if loc:
                key = (t, int(float(loc["x"]) // cell_mm), int(float(loc["y"]) // cell_mm))
                cells.setdefault(key, set()).add(i)
            if d.get("near_edge_mm") is not None and float(d["near_edge_mm"]) <= edge_mm:
                near_edge += 1
            uncertain += bool(d.get("status"))
    recurring = [{"type": t, "x_mm": [cx * cell_mm, (cx + 1) * cell_mm], "y_mm": [cy * cell_mm, (cy + 1) * cell_mm],
                  "parts": len(parts)}
                 for (t, cx, cy), parts in cells.items() if len(parts) >= min_parts]
    recurring.sort(key=lambda x: (-x["parts"], x["type"], x["x_mm"], x["y_mm"]))
    stamps = [r.get("timestamp") for r in recs if r.get("timestamp")]
    return {"inspections": len(recs), "from": stamps[0] if stamps else None, "to": stamps[-1] if stamps else None,
            "by_verdict": dict(by_verdict), "parts_with_defects": with_defects, "by_type": dict(by_type),
            "by_severity": dict(by_sev), "by_type_severity": {t: dict(c) for t, c in by_type_sev.items()},
            "gate": dict(gates), "retakes": {"count": by_verdict.get("RETAKE", 0), "messages": dict(retake_msgs)},
            "near_edge": near_edge, "uncertain": uncertain, "recurring": recurring[:5], "cell_mm": cell_mm}


def _fmt_num(x: float) -> str:
    return f"{x:g}"


def shift_summary_text(summary: dict[str, Any], spec: dict[str, Any] | None = None) -> str:
    """The operator's shift summary as numbered lines (template; explain_shift may use Llama)."""
    n = summary["inspections"]
    if not n:
        return "Shift summary: no inspections yet."
    lines = [f"Shift summary: {n} inspection(s), {summary['parts_with_defects']} with defects."]
    verdicts = ", ".join(f"{v} {c}" for v, c in sorted(summary["by_verdict"].items(), key=lambda x: -x[1]))
    lines.append(f"1. Verdicts: {verdicts}.")
    if summary["by_type"]:
        parts = []
        for t, c in sorted(summary["by_type"].items(), key=lambda x: -x[1]):
            sev = ", ".join(f"{s} {k}" for s, k in summary["by_type_severity"].get(t, {}).items())
            parts.append(f"{t} {c} ({sev})")
        lines.append(f"2. Defects: {'; '.join(parts)}.")
    else:
        lines.append("2. Defects: none.")
    rt = summary["retakes"]
    if rt["count"]:
        msgs = "; ".join(f"{m} ({c})" for m, c in rt["messages"].items())
        lines.append(f"3. Retakes: {rt['count']}: {msgs}")
    else:
        lines.append("3. Retakes: none.")
    lines.append(f"4. Near an edge or hole: {summary['near_edge']}; uncertain detections for the operator to confirm: {summary['uncertain']}.")
    classes = (spec or {}).get("classes") or {}
    if summary["recurring"]:
        rec = []
        for r in summary["recurring"][:3]:
            tip = classes.get(r["type"], {}).get("prevention", "check fixtures, guides and handling at that spot.")
            rec.append(f"{r['type']} at x {_fmt_num(r['x_mm'][0])}-{_fmt_num(r['x_mm'][1])} mm, "
                       f"y {_fmt_num(r['y_mm'][0])}-{_fmt_num(r['y_mm'][1])} mm on {r['parts']} parts ({tip})")
        lines.append(f"5. Recurring locations: {'; '.join(rec)}")
    else:
        lines.append("5. Recurring locations: none.")
    return "\n".join(lines)


def check_shift_reply(text: str, summary: dict[str, Any], spec: dict[str, Any] | None = None) -> list[str]:
    """Problems with a model's shift summary: banned phrases, verdict words and numbers not in the summary."""
    problems = []
    for phrase in (spec or {}).get("banned_phrases") or DEFAULT_STEEL_BANNED:
        if any(not _negated(text, m.start()) for m in _phrase_re(phrase).finditer(text)):
            problems.append(f"banned phrase: '{phrase}'")
    for m in STEEL_VERDICT_RE.finditer(text):
        v = m.group(1).replace(" ", "_")
        if v not in summary.get("by_verdict", {}):
            problems.append(f"verdict not in the summary: {v}")
    allowed = _allowed_numbers(json.dumps(summary, ensure_ascii=False), shift_summary_text(summary, spec), SHIFT_PROMPT)
    plain = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    invented = sorted({m for m in NUM_RE.findall(plain) if round(float(m), 6) not in allowed}, key=float)
    if invented:
        problems.append(f"numbers not in the summary: {', '.join(invented)}")
    return problems


def explain_shift(summary: dict[str, Any], backend=None, spec: dict[str, Any] | None = None) -> tuple[str, str]:
    """(text, backend that answered): Llama summarises the shift when a model backend is given and its
    reply passes check_shift_reply; otherwise the template text."""
    template = shift_summary_text(summary, spec)
    if backend is None or isinstance(backend, MockBackend):
        return template, MockBackend.name
    messages = [{"role": "system", "content": SHIFT_PROMPT},
                {"role": "user", "content": f"SHIFT SUMMARY:\n{json.dumps(summary, separators=(',', ':'))}\n\n"
                                            "Summarise this shift for the operator."}]
    try:
        reply = backend.generate(messages, {"report": None, "spec": spec, "shift": summary})
    except Exception as exc:  # the station keeps working without the model
        return f"{template}\n\n(Assistant model unavailable: {exc}. Showing the template summary.)", MockBackend.name
    if check_shift_reply(reply, summary, spec):
        return f"{template}\n\n(Template summary: the model reply failed the guardrail check.)", MockBackend.name
    return reply.strip(), getattr(backend, "name", type(backend).__name__)
