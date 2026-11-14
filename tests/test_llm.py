import json
import stat
import sys
from pathlib import Path

import pytest

from trinetra.llm import CAUSE_LABEL, answer_steel

from trinetra.llm import Advisor, ExecuTorchBackend, MockBackend, clean_runner_output, format_llama3
from trinetra.report import compact_for_llm
from trinetra.rules import load_spec

SPEC = load_spec(Path(__file__).resolve().parents[1] / "configs" / "spec.yaml")

REPORT = {
    "mode": "surface",
    "part_id": "P1",
    "part": {"name": "plate", "material": "steel"},
    "verdict": "REWORK",
    "verdict_instruction": "Send the part to rework with the listed actions, then re-inspect.",
    "actions": ["Polish the scratch out."],
    "defects": [
        {
            "id": "D1",
            "type": "scratch",
            "confidence": 0.91,
            "location_mm": {"x": 42.5, "y": 18.0},
            "zone": "general",
            "size": {"length_mm": 7.8, "width_mm": 0.3, "area_mm2": 2.1, "diameter_mm": 1.6},
            "orientation_deg": 32,
            "relief": "recessed",
            "severity": "major",
            "disposition": "REWORK",
            "rule": "scratch length_mm 7.80 >= 5.0 (major, zone general)",
            "action": "Polish the scratch out.",
            "depth_mm": -0.004,
            "evidence": {
                "contrast": {
                    "top_rgb": [0.04, 0.05, 0.06],
                    "top_under_45deg_light": 0.08,
                    "top_under_15deg_light": 0.21,
                    "oblique45": 0.19,
                    "grazing": 0.55,
                },
                "geometry_ratio": 10.1,
                "color_shift": 0.012,
                "slope_change": 0.008,
                "shading_polarity": 0.05,
                "strongest": "grazing",
                "basis": "it is a thin line that lights up at the grazing camera",
            },
        }
    ],
}


def test_llama3_template():
    p = format_llama3([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert p == (
        "<|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_compact_report_is_small_and_keeps_verdict():
    text = compact_for_llm(REPORT)
    assert json.loads(text)["verdict"] == "REWORK"
    assert len(text) < 1000


def test_clean_runner_output():
    prompt = "PROMPT<|start_header_id|>assistant<|end_header_id|>\n\n"
    out = prompt + "Polish D1.<|eot_id|>\nPyTorchObserver {\"tokens\": 5}\n"
    assert clean_runner_output(out, prompt) == "Polish D1."


def _fake_runner(tmp_path: Path, body: str) -> str:
    script = tmp_path / "llama_main"
    script.write_text(f"#!{sys.executable}\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_executorch_backend_with_fake_runner(tmp_path):
    runner = _fake_runner(
        tmp_path,
        "import sys\n"
        "prompt = [a for a in sys.argv if a.startswith('--prompt=')][0][len('--prompt='):]\n"
        "print(prompt + 'Part needs rework: polish D1.<|eot_id|>')\n"
        "print('PyTorchObserver {}')",
    )
    backend = ExecuTorchBackend(runner, "m.pte", "tokenizer.model")
    advisor = Advisor(backend, SPEC, REPORT)
    assert advisor.explain() == "Part needs rework: polish D1."
    assert advisor.ask("what next?") == "Part needs rework: polish D1."


def test_advisor_falls_back_when_runner_fails(tmp_path):
    runner = _fake_runner(tmp_path, "import sys; sys.exit(3)")
    advisor = Advisor(ExecuTorchBackend(runner, "m.pte", "t.model"), SPEC, REPORT)
    text = advisor.explain()
    assert "Verdict: REWORK" in text and "unavailable" in text


@pytest.mark.parametrize(
    "question, expect",
    [
        ("Can I just pass it?", "final"),
        ("What caused D1?", "parts sliding"),
        ("why is D1 a scratch?", "grazing"),
        ("where is it?", "x=42.5"),
    ],
)
def test_template_answers(question, expect):
    advisor = Advisor(MockBackend(), SPEC, REPORT)
    assert expect in advisor.ask(question)


def test_history_is_bounded():
    advisor = Advisor(MockBackend(), SPEC, REPORT, history_turns=2)
    for i in range(6):
        advisor.ask(f"where {i}?")
    msgs = advisor._messages("last?")
    assert msgs[1] is advisor.first  # the report always stays in context
    assert len(msgs) == 2 + 2 * 2 + 1


PENETRANT_REPORT = {
    "mode": "penetrant",
    "part_id": "P2-PT",
    "part": {"name": "plate", "material": "aluminium"},
    "verdict": "REJECT",
    "verdict_instruction": "Put the part in the red REJECT bin and tag it with the defect IDs.",
    "actions": [SPEC["penetrant"]["actions"]["linear"]],
    "penetrant": {"profile": "asme_app8", "relevant_mm": 1.6, "process": {"development_min": 10}, "aligned_groups": [], "warnings": []},
    "defects": [
        {
            "id": "I1",
            "type": "linear_indication",
            "confidence": 0.95,
            "location_mm": {"x": 30.0, "y": 22.5},
            "zone": "general",
            "size": {"length_mm": 6.2, "width_mm": 0.6, "area_mm2": 3.1, "diameter_mm": 2.0},
            "orientation_deg": 40,
            "relevant": True,
            "severity": "critical",
            "disposition": "REJECT",
            "rule": "relevant linear indication 6.20 mm: any relevant linear indication rejects (asme_app8)",
            "action": SPEC["penetrant"]["actions"]["linear"],
            "evidence": {"redness": 1.4, "aspect_ratio": 10.3, "basis": "red penetrant bled out along a line"},
        }
    ],
}


def test_penetrant_report_in_template_mode():
    advisor = Advisor(MockBackend(), SPEC, PENETRANT_REPORT)
    text = advisor.explain()
    assert "REJECT" in text and "linear indication" in text and "6.2 mm" in text
    assert "aspect ratio 10.3" in advisor.ask("why is I1 a crack?")
    assert "fatigue" in advisor.ask("what caused I1?")
    assert json.loads(compact_for_llm(PENETRANT_REPORT))["penetrant"]["profile"] == "asme_app8"


# ---- steel mode (record 3.0, spec_steel.yaml) -----------------------------------------------------

from trinetra.llm import (SYSTEM_PROMPT, SYSTEM_PROMPT_AERO, check_reply, explain_shift, shift_summary,  # noqa: E402
                          shift_summary_text, system_prompt)
from trinetra.report import build_steel_record  # noqa: E402
from trinetra.rules import judge_steel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEEL = load_spec(ROOT / "configs" / "spec_steel.yaml")
AERO = load_spec(ROOT / "configs" / "spec_aero.yaml")
PART = {"material": "ms_hot_rolled", "thickness_mm": 3.0, "size_mm": [100.0, 100.0]}
TOO_BRIGHT = "Too bright: shade the part or fit the hood, then retake."


def steel_defect(kind, did, x, y, conf=0.9, near=30.0, **size):
    return {"id": did, "type": kind, "confidence": conf, "box_px": [100.0, 100.0, 200.0, 140.0],
            "location_mm": {"x": x, "y": y}, "size": size, "near_edge_mm": near}


def steel_record(defects=None, gate="PASS", message="", part_id="P7", stamp="2026-11-14T10:00:00+05:30", reasons=()):
    if defects is None:
        defects = [steel_defect("scratch", "D1", 42.5, 18.0, conf=0.91, width_mm=0.31, length_mm=12.4),
                   steel_defect("stain", "D2", 70.0, 72.0, conf=0.8, area_mm2=60.0, diameter_mm=8.7)]
    verdict, judged = judge_steel(defects, STEEL, PART)
    cap = {"mode": "burst5", "device_mode": "open", "r_ambient": 0.62 if gate != "RETAKE" else 7.5,
           "lux_est": 1800, "motion_px": 0.4, "sat_pct": 0.1, "blur": 0.92, "gate": gate, "gate_message": message,
           "gate_reasons": list(reasons)}
    return build_steel_record(part_id, PART, judged, verdict, cap, {"capture": 400.0, "detect": 300.0},
                              {"vision": "onnx:best.onnx", "advisor": None}, spec=STEEL, timestamp=stamp)


def test_steel_system_prompt_lists_the_record_references():
    rec = steel_record()
    prompt = system_prompt(STEEL, rec)
    assert "steel and metal fabrication" in prompt and "never change" in prompt.lower()
    assert '"EN 10163-2 / ISO 7788 Table 1 (3-8 mm: 0.2 mm)"' in prompt
    assert "Hindi" in prompt and "{references}" not in prompt
    assert Advisor(MockBackend(), STEEL, rec).system == prompt
    assert system_prompt(SPEC) == SYSTEM_PROMPT and system_prompt(AERO) == SYSTEM_PROMPT_AERO


@pytest.mark.parametrize("question", [None, "What should I do next?", "Can I ship it anyway?", "How deep is it?",
                                      "Explain in Hindi.", "मुझे हिंदी में समझाओ", "Why is the light a problem?",
                                      "What caused D1?", "why is D2 a stain?", "where is D1?", "tell me a joke"])
@pytest.mark.parametrize("gate", ["PASS", "PASS_FLAGGED", "RETAKE"])
def test_steel_template_replies_pass_the_guardrail(question, gate):
    rec = steel_record(gate=gate, message=TOO_BRIGHT if gate == "RETAKE" else "")
    advisor = Advisor(MockBackend(), STEEL, rec)
    text = advisor.explain() if question is None else advisor.ask(question)
    assert advisor.last["problems"] == [] and advisor.answered_by == "mock-template"
    assert check_reply(text, rec, STEEL, context=question or "") == []
    if gate == "RETAKE":
        assert "REWORK" not in text and "HOLD_FOR_QA" not in text


def test_steel_explain_says_verdict_rule_and_action():
    rec = steel_record()
    text = Advisor(MockBackend(), STEEL, rec).explain()
    assert text.startswith("Verdict: HOLD_FOR_QA.")
    assert "D1: scratch (91% confidence)" in text and "major (Defect) -> HOLD_FOR_QA" in text
    assert "Check the depth with a fingernail" in text and "does not measure depth" in text


def test_retake_record_gets_the_gate_message():
    rec = steel_record(gate="RETAKE", message=TOO_BRIGHT, reasons=["too_bright", "motion"])
    assert rec["verdict"] == "RETAKE" and rec["defects"] == []
    advisor = Advisor(MockBackend(), STEEL, rec)
    text = advisor.explain()
    advice = STEEL["capture_advice"]["reasons"]
    assert text.startswith("Verdict: RETAKE.") and TOO_BRIGHT in text
    assert advice["motion"] in text and advice["too_bright"] not in text    # the message already says it
    assert TOO_BRIGHT in advisor.ask("What should I do next?")
    dark = steel_record(gate="RETAKE", message="", reasons=["too_dark"])
    assert advice["too_dark"] in Advisor(MockBackend(), STEEL, dark).explain()


def test_flagged_capture_gets_advice_first():
    text = Advisor(MockBackend(), STEEL, steel_record(gate="PASS_FLAGGED")).explain()
    assert text.splitlines()[1].startswith("Capture flagged: Ambient light is high")


def test_steel_hindi_answer():
    text = Advisor(MockBackend(), STEEL, steel_record()).ask("मुझे हिंदी में बताओ")
    assert "निर्णय: HOLD_FOR_QA" in text and "खरोंच" in text


@pytest.mark.parametrize("reply, problem", [
    ("Grind it per ISO 9001.", "reference not on the whitelist"),
    ("Use the EN 10163-2 Table 4 limit.", "reference not on the whitelist"),
    ("Check against ASTM A6 before painting.", "reference not on the whitelist"),
    ("The scratch is 0.45 mm deep.", "numbers not in the record"),
    ("About 2,000 lux.", "numbers not in the record"),
    ("The part is certified.", "banned phrase"),
    ("The plate is compliant with IS 2062.", "banned phrase"),
    ("Approved: move on.", "banned phrase"),
    ("The camera measures depth here.", "banned phrase"),
    ("TriNetra works in any lighting.", "banned phrase"),
    ("You will get zero defects.", "banned phrase"),
    ("This result is guaranteed.", "banned phrase"),
    ("The verdict is REJECT.", "contradicts the verdict"),
    ("D1 needs REWORK.", "contradicts the verdict"),
    ("The part passes, ship it.", "contradicts the verdict"),
    ("No defects were found.", "contradicts the verdict"),
])
def test_check_reply_catches(reply, problem):
    problems = check_reply(reply, steel_record(), STEEL)
    assert any(p.startswith(problem) for p in problems), problems


@pytest.mark.parametrize("reply", [
    "1. Put the part in the HOLD tray. 2. Check D1 with a gauge [EN 10163-2 / ISO 7788 Table 1 (3-8 mm: 0.2 mm)].",
    "EN 10163-2 Table 2 allows up to 0.4 mm on 3-8 mm plate.",
    "The part is not certified; the camera does not measure depth.",
    "The capture check was PASS FLAGGED; D1 is HOLD_FOR_QA and D2 is CLEAN_AND_REINSPECT.",
    "D1 is about 12 mm long and 0.3 mm wide; about 1,800 lux of ambient light.",
    "D1 is 91% sure.",
])
def test_check_reply_allows(reply):
    assert check_reply(reply, steel_record(), STEEL) == []


@pytest.mark.parametrize("reply", [
    "No problem, the part is certified.",            # the negation is about something else
    "Do not worry, this result is guaranteed.",
    "No, it is compliant with the spec.",
    "Without doubt the camera measures depth.",
    "Not only is it certified, it is also fine.",
    "There is no rust and the part is approved.",
])
def test_banned_phrase_after_an_unrelated_negation_is_caught(reply):
    assert any(p.startswith("banned phrase") for p in check_reply(reply, steel_record(), STEEL))


@pytest.mark.parametrize("reply", [
    "The part is not certified and cannot be guaranteed.",
    "Never say it works in any lighting.",
    "This is not a certification.",
])
def test_banned_phrase_right_after_its_negation_is_allowed(reply):
    assert check_reply(reply, steel_record(), STEEL) == []


def test_verdict_claim_is_checked_against_the_part_verdict():
    # REJECT part with a PASS defect: "Verdict: PASS" must not be excused by D1's own disposition.
    rec = steel_record([steel_defect("scratch", "D1", 20.0, 20.0, width_mm=0.1, length_mm=5.0),
                        steel_defect("crack", "D2", 60.0, 60.0, length_mm=3.0)])
    assert rec["verdict"] == "REJECT" and rec["defects"][0]["disposition"] == "PASS"
    for reply in ("Verdict: PASS. Move the part on.", "The verdict is pass.", "Final verdict - hold for QA."):
        assert any(p.startswith("contradicts the verdict") for p in check_reply(reply, rec, STEEL)), reply
    for reply in ("Verdict: REJECT.", "The verdict REJECT is final. D1 alone would be PASS.",
                  "The verdict for D1 is PASS, but the part verdict is REJECT."):
        assert check_reply(reply, rec, STEEL) == [], reply


def test_markdown_verdict_claim_is_checked():
    assert any(p.startswith("contradicts the verdict") for p in check_reply("**Verdict:** PASS", steel_record(), STEEL))


def test_severity_of_a_named_defect_must_match_the_record():
    rec = steel_record()                                   # D1 major scratch, D2 minor stain
    for reply in ("D1 is a minor scratch, so ignore it.", "D1 is critical.", "D2 (stain) is major."):
        assert any(p.startswith("contradicts the severity") for p in check_reply(reply, rec, STEEL)), reply
    for reply in ("D1 is a major scratch; D2 is a minor stain.", "D1 is not minor: it is major.",
                  "D2 is minor. D1 is major and needs a depth check."):
        assert check_reply(reply, rec, STEEL) == [], reply


@pytest.mark.parametrize("reply", [
    "The part is rejected.", "Reject the part and tag it.", "Put it in the reject bin.",
    "The part is accepted.", "Accept the part.", "Move it to the OK bin.", "This part is good to go.",
    "You can send it to the customer.",
])
def test_plain_words_that_contradict_a_hold(reply):
    assert any(p.startswith("contradicts the verdict") for p in check_reply(reply, steel_record(), STEEL)), reply


def test_negated_ship_or_accept_answers_the_question_correctly():
    rec = steel_record()
    for reply in ("No, you cannot ship it. The verdict HOLD_FOR_QA is final.", "Do not accept the part yet.",
                  "Do not reject the part: QA checks it."):
        assert check_reply(reply, rec, STEEL, context="Can I ship it anyway?") == [], reply
    for reply in ("Don't hesitate to ship it.", "No problem, ship it."):
        assert any(p.startswith("contradicts the verdict") for p in check_reply(reply, rec, STEEL)), reply


def test_reject_words_are_fine_on_a_reject_record():
    rec = steel_record([steel_defect("crack", "D1", 60.0, 60.0, length_mm=3.0)])
    assert check_reply("The part is rejected: put it in the red REJECT bin.", rec, STEEL) == []


@pytest.mark.parametrize("reply", ["Check it to ASME Section VIII.", "Follow the ASTM standards for plate."])
def test_standards_body_without_a_number_is_not_whitelisted(reply):
    assert any(p.startswith("reference not on the whitelist") for p in check_reply(reply, steel_record(), STEEL))


def test_sun_retake_does_not_say_move_closer():
    # C6 sun: B and A both clip, so lightproc also reports too_dark; "move to 200 mm" would be wrong.
    rec = steel_record(gate="RETAKE", message=TOO_BRIGHT, reasons=["invalid", "too_bright", "too_dark"])
    text = Advisor(MockBackend(), STEEL, rec).explain()
    assert TOO_BRIGHT in text and STEEL["capture_advice"]["reasons"]["too_dark"] not in text
    assert check_reply(text, rec, STEEL) == []


def test_retake_reply_must_not_judge_the_part():
    rec = steel_record(gate="RETAKE", message=TOO_BRIGHT)
    assert check_reply("No defects were found; the part is fine.", rec, STEEL)
    assert check_reply("D1 needs REWORK.", rec, STEEL)
    assert check_reply("RETAKE: shade the part or fit the hood, then retake.", rec, STEEL) == []


def test_steel_advisor_falls_back_on_a_bad_model_reply(tmp_path):
    runner = _fake_runner(tmp_path, "print('The part is certified to ISO 9001 and passes.<|eot_id|>')")
    rec = steel_record()
    advisor = Advisor(ExecuTorchBackend(runner, "m.pte", "t.model"), STEEL, rec)
    text = advisor.explain()
    assert advisor.answered_by == "mock-template" and "The part is certified" in advisor.last["rejected"]
    assert any("banned phrase" in p for p in advisor.last["problems"])
    assert "failed the guardrail check" in text and check_reply(text, rec, STEEL) == []


def test_steel_advisor_keeps_a_good_model_reply(tmp_path):
    good = "parts sliding on a dirty bench. Lift parts instead of sliding them."
    runner = _fake_runner(tmp_path, f"print({good!r} + '<|eot_id|>')")
    advisor = Advisor(ExecuTorchBackend(runner, "m.pte", "t.model"), STEEL, steel_record())
    assert advisor.explain().endswith(CAUSE_LABEL + "The scratch most likely comes from " + good)
    assert advisor.answered_by.startswith("llama via executorch") and advisor.last["problems"] == []


def test_aero_prompt_path_still_guarded():
    report = {"domain": "aerospace", "part_id": "A1", "part": {"name": "coupon", "thickness_mm": 3.0},
              "verdict": "PASS", "verdict_instruction": "Log it.", "defects": [], "findings": [], "actions": [],
              "references": [], "out_of_scope": []}
    advisor = Advisor(MockBackend(), AERO, report)
    assert advisor.system == SYSTEM_PROMPT_AERO and advisor.guard
    assert "Verdict: PASS" in advisor.explain()
    assert any("banned phrase" in p for p in check_reply("The part is airworthy.", report, AERO))


def test_surface_mode_is_unguarded_by_default():
    assert not Advisor(MockBackend(), SPEC, REPORT).guard


# ---- shift summary -----------------------------------------------------------------------------------

def _shift():
    recs = []
    for k in range(6):
        defects = [steel_defect("scratch", "D1", 45.0 + k % 2, 12.0, width_mm=0.1, length_mm=6.0)]
        if k % 3 == 0:
            defects.append(steel_defect("stain", "D2", 80.0, 80.0, area_mm2=30.0))
        recs.append(steel_record(defects, part_id=f"P{k}", stamp=f"2026-11-14T10:0{k}:00+05:30"))
    recs.append(steel_record(gate="RETAKE", message=TOO_BRIGHT, part_id="P6", stamp="2026-11-14T10:06:00+05:30"))
    recs.append(steel_record([], part_id="P7", stamp="2026-11-14T10:07:00+05:30"))
    return recs


def test_shift_summary_counts_and_recurring_locations():
    s = shift_summary(_shift())
    assert s["inspections"] == 8 and s["parts_with_defects"] == 6
    assert s["by_verdict"] == {"CLEAN_AND_REINSPECT": 2, "PASS": 5, "RETAKE": 1}
    assert s["by_type"] == {"scratch": 6, "stain": 2}
    assert s["by_type_severity"]["scratch"] == {"minor": 6}
    assert s["retakes"] == {"count": 1, "messages": {TOO_BRIGHT: 1}}
    assert s["recurring"][0] == {"type": "scratch", "x_mm": [40.0, 60.0], "y_mm": [0.0, 20.0], "parts": 6}
    assert shift_summary(_shift(), last_n=2)["inspections"] == 2
    text = shift_summary_text(s, STEEL)
    assert "8 inspection(s)" in text and "scratch at x 40-60 mm, y 0-20 mm on 6 parts" in text
    assert STEEL["classes"]["scratch"]["prevention"] in text


def test_explain_shift_uses_the_model_only_when_it_passes(tmp_path):
    s = shift_summary(_shift())
    text, who = explain_shift(s, MockBackend(), STEEL)
    assert who == "mock-template" and text == shift_summary_text(s, STEEL)
    bad = _fake_runner(tmp_path, "print('42 inspections, all certified.<|eot_id|>')")
    text, who = explain_shift(s, ExecuTorchBackend(bad, "m.pte", "t.model"), STEEL)
    assert who == "mock-template" and "failed the guardrail" in text
    (tmp_path / "ok").mkdir()
    good = _fake_runner(tmp_path / "ok", "print('1. 8 inspections, 1 retake. 2. Scratches recur at x 40-60 mm.<|eot_id|>')")
    text, who = explain_shift(s, ExecuTorchBackend(good, "m.pte", "t.model"), STEEL)
    assert who.startswith("llama via executorch") and text.startswith("1. 8 inspections")


class _Recorder:
    """A model backend with a seq_len (like ExecuTorchBackend) that records its prompts."""
    name = "recorder"

    def __init__(self, seq_len):
        self.seq_len, self.prompts = seq_len, []

    def generate(self, messages, context=None):
        self.prompts.append(format_llama3(messages))
        return "Put the part in the HOLD tray and call QA."


def test_advisor_keeps_the_prompt_inside_the_context():
    # a worst-case record (3 shown defects, long rules and actions, flagged capture) plus history
    # must stay inside seq_len minus the answer, or llama_main fails and the template answers
    ds = [steel_defect("pit", "D1", 3.0, 50.0, conf=0.4, near=1.0, diameter_mm=6, count=10, area_mm2=20),
          steel_defect("corrosion", "D2", 4.0, 54.0, near=2.0, area_mm2=900, diameter_mm=30),
          steel_defect("scratch", "D3", 50.0, 3.0, near=0.2, width_mm=0.6, length_mm=30),
          steel_defect("dent", "D4", 80.0, 80.0, diameter_mm=7)]
    rec = steel_record(ds, gate="PASS_FLAGGED", reasons=["ambient_high"])
    be = _Recorder(2048)
    advisor = Advisor(be, STEEL, rec, history_turns=3)
    advisor.explain()
    for q in ("What should I do next?", "Explain in Hindi.", "मुझे हिंदी में समझाओ", "How deep is it?"):
        advisor.ask(q)
    budget = (2048 - 256) * 3.5
    assert advisor.max_prompt_bytes == int(budget)
    assert all(len(p.encode()) <= budget for p in be.prompts)
    asks = [p for p in be.prompts if "INSPECTION BRIEF:" in p]          # the first answer (and its rewrite) is the cause note
    assert "INSPECTION BRIEF:" not in be.prompts[0] and "LIKELY CAUSE:" in be.prompts[0] and len(asks) >= 4
    assert all(advisor.first["content"] in p for p in asks)              # the record always stays for follow-ups
    assert Advisor(MockBackend(), STEEL, rec).max_prompt_bytes is None    # the template is not limited


# ---- 22 Nov: guardrail checks from the first Pi run (3 of 7 Llama answers had factual slips) ----

from trinetra.llm import steel_brief  # noqa: E402


def crack_record():
    return steel_record([steel_defect("scratch", "D1", 20.0, 20.0, width_mm=0.9, length_mm=20.0),
                         steel_defect("crack", "D2", 60.0, 60.0, conf=0.32, length_mm=5.0)])


def test_first_answer_must_name_the_verdict_and_the_deciding_defect():
    rec = crack_record()
    assert rec["verdict"] == "REJECT"
    blames_scratch = ("The scratch is a critical defect. 1. Tag the part and inform QA. "
                      "2. Put the part in the RED REJECT bin.")
    probs = check_reply(blames_scratch, rec, STEEL, first=True)
    assert any(p.startswith("does not name the defect that decides") for p in probs)
    good = ("The crack D2 decides it: any crack is critical, even an uncertain one. "
            "It likely comes from bending strain at a tight radius.")
    assert check_reply(good, rec, STEEL, first=True) == []
    assert check_reply(blames_scratch, rec, STEEL) == []      # follow-ups need not name the deciding defect


def test_first_steel_answer_is_the_rules_header_plus_the_model_why(tmp_path):
    rec = crack_record()
    para = "bending strain at a tight radius. You can ask QA to check the bend radius before forming"
    runner = _fake_runner(tmp_path, f"print({para!r} + '<|eot_id|>')")
    advisor = Advisor(ExecuTorchBackend(runner, "m.pte", "t.model"), STEEL, rec)
    text = advisor.explain()
    assert text.startswith("Verdict: REJECT. Decided by D2 crack") and "What to do:\n1. Stop:" in text
    assert text.endswith(CAUSE_LABEL + "The crack most likely comes from bending strain at a tight radius.")   # cut fragment dropped
    assert advisor.answered_by.startswith("llama via executorch")
    assert "(No action)" not in text
    bad = _fake_runner(tmp_path, "print('1. Do not paint over it. (No action)<|eot_id|>')")
    adv2 = Advisor(ExecuTorchBackend(bad, "m.pte", "t.model"), STEEL, rec)
    assert "failed the guardrail" in adv2.explain() and adv2.answered_by == "mock-template"


def test_invented_or_inverted_depth_is_rejected():
    rec = steel_record()
    for bad in ("If the depth is 0.2 mm or less, repair the part.",
                "The operator has also identified a depth of 0.2 mm, which is below the limit."):
        assert any(p.startswith("states a depth") for p in check_reply(bad, rec, STEEL)), bad
    quoted = "QA checks the depth with a gauge. On 3-8 mm plate up to 0.2 mm deep may stay."
    assert not any(p.startswith("states a depth") for p in check_reply(quoted, rec, STEEL))


def test_naming_a_defect_that_is_not_there_is_rejected():
    rec = steel_record()                                       # a scratch and a stain
    assert any("not in the record: crack" in p for p in check_reply("Check the crack near D1.", rec, STEEL))
    assert not any("not in the record" in p for p in check_reply("There is no crack.", rec, STEEL))
    assert not any("not in the record: dent" in p
                   for p in check_reply("A dent would need a gauge.", rec, STEEL, context="is it a dent?"))


def test_defect_id_with_the_wrong_disposition_is_rejected():
    rec = crack_record()
    probs = check_reply("D1 is REJECT because it is deep.", rec, STEEL)
    assert any(p.startswith("contradicts the disposition") for p in probs)
    assert not any(p.startswith("contradicts the disposition")
                   for p in check_reply("D2 crack: REJECT.", rec, STEEL))


def test_steel_brief_is_short_and_leads_with_the_decision():
    rec = crack_record()
    brief = steel_brief(rec, STEEL)
    lines = brief.splitlines()
    assert lines[0] == "VERDICT: REJECT" and lines[1].startswith("WHY: D2 crack")
    assert "STEPS:" in lines and rec["verdict_instruction"] in brief
    advisor = Advisor(MockBackend(), STEEL, rec)
    assert len(advisor.first["content"]) < 1500 and "INSPECTION BRIEF" in advisor.first["content"]
    retake = steel_brief(steel_record(gate="RETAKE", message=TOO_BRIGHT, reasons=["too_bright"]), STEEL)
    assert retake.startswith("VERDICT: RETAKE") and "Inspect the part again." in retake


@pytest.mark.parametrize("gate", ["PASS", "PASS_FLAGGED", "RETAKE"])
def test_templates_pass_the_guardrail_for_crack_records_too(gate):
    for rec in (steel_record(gate=gate, message=TOO_BRIGHT if gate == "RETAKE" else ""), crack_record()):
        assert check_reply(Advisor(MockBackend(), STEEL, rec).explain(), rec, STEEL) == []


def test_brief_rule_in_words_and_retake_needs_no_model(tmp_path):
    rec = crack_record()
    brief = steel_brief(rec, STEEL)
    assert "width_mm" not in brief and "MEANING: REJECT" in brief
    assert "detector is unsure (32% confidence)" in brief and "a crack is still rejected" in brief
    hold = steel_brief(steel_record(), STEEL)
    assert "width 0.31 mm, at or over the 0.2 mm limit" in hold
    retake = steel_record(gate="RETAKE", message=TOO_BRIGHT, reasons=["too_bright"])
    runner = _fake_runner(tmp_path, "raise SystemExit('must not be called')")
    adv = Advisor(ExecuTorchBackend(runner, "m.pte", "t.model"), STEEL, retake)
    text = adv.explain()
    assert text.startswith("Verdict: RETAKE.") and adv.answered_by == "rules-engine"


def test_preamble_dropped_and_wrong_verdict_claim_rejected(tmp_path):
    rec = crack_record()
    para = "Here is the note:\n\nbending strain. Check the bend radius."
    adv = Advisor(ExecuTorchBackend(_fake_runner(tmp_path, f"print({para!r} + '<|eot_id|>')"), "m.pte", "t.model"),
                  STEEL, rec)
    assert adv.explain().endswith(CAUSE_LABEL + "The crack most likely comes from bending strain. Check the bend radius.")
    assert any("contradicts the verdict" in p for p in check_reply("The verdict is incorrect, the crack is fine.", rec, STEEL))


def test_hold_answer_must_not_decide_repair():
    rec = steel_record()
    assert rec["verdict"] == "HOLD_FOR_QA"
    assert any("QA decides" in p for p in check_reply("The scratch is critical and requires repair or scrap.", rec, STEEL))
    assert any("QA decides" in p for p in check_reply("It is critical and requires immediate repair.", rec, STEEL))
    assert not any("QA decides" in p for p in check_reply("QA checks the scratch with a gauge and decides.", rec, STEEL))


def test_one_guarded_rewrite_before_the_template(tmp_path):
    rec = steel_record()
    body = ("import sys\np = [a for a in sys.argv if a.startswith('--prompt=')][0]\n"
            "print('parts sliding on a dirty bench, so QA checks it with a gauge.' if 'broke these rules' in p "
            "else 'a bench, and it requires repair.', end='<|eot_id|>')")
    adv = Advisor(ExecuTorchBackend(_fake_runner(tmp_path, body), "m.pte", "t.model"), STEEL, rec)
    text = adv.explain()
    assert text.endswith(CAUSE_LABEL + "The scratch most likely comes from parts sliding on a dirty bench, so QA checks it with a gauge.")
    assert adv.answered_by.startswith("llama") and "requires repair" in adv.last["retried"]["rejected"]
    bad = _fake_runner(tmp_path, "print('The scratch D1 requires repair.', end='<|eot_id|>')")
    adv2 = Advisor(ExecuTorchBackend(bad, "m.pte", "t.model"), STEEL, rec)
    assert "failed the guardrail" in adv2.explain() and adv2.answered_by == "mock-template"


def test_cause_note_covers_one_defect_type():
    from trinetra.llm import cause_brief, cause_opening
    ds = [steel_defect("scratch", "D1", 20.0, 20.0, width_mm=0.6, length_mm=20.0),
          steel_defect("corrosion", "D2", 60.0, 60.0, conf=0.4, area_mm2=10.0, diameter_mm=3.6)]
    rec = steel_record(ds)
    assert rec["verdict"] == "HOLD_FOR_QA" and {d["type"] for d in rec["defects"] if d["disposition"] == "HOLD_FOR_QA"} == {"scratch", "corrosion"}
    assert cause_opening(rec) == "The scratch most likely comes from"
    brief = cause_brief(rec, STEEL)
    assert brief.count("DEFECT:") == 1 and "corrosion" not in brief


def test_a_refusal_is_not_an_answer():
    rec = crack_record()
    reply = "I cannot provide repair steps for a crack in a metal part. Is there anything else I can help you with?"
    assert "the model refused to answer" in check_reply(reply, rec, STEEL, context="How do I fix it?")
    adv = Advisor(MockBackend(), STEEL, rec)
    no = adv.ask("Can I just paint over the crack and ship it?")
    assert "REJECT" in no and check_reply(no, rec, STEEL, context="Can I just paint over the crack and ship it?") == []


def test_fix_and_override_questions_get_the_rules_answer(tmp_path):
    rec = crack_record()
    fix = answer_steel("How do I fix it? Give the repair steps.", rec, STEEL)
    assert fix.startswith("D2 (crack)") and "red-dye penetrant" in fix     # the deciding crack, not the scratch
    runner = _fake_runner(tmp_path, "print('bending strain. Check the bend radius.', end='<|eot_id|>')")
    adv = Advisor(ExecuTorchBackend(runner, "m.pte", "t.model"), STEEL, rec)
    adv.explain()
    for q in ("How do I fix it? Give the repair steps.", "Can I paint over it and ship it?"):
        text = adv.ask(q)
        assert adv.answered_by == "rules-engine" and check_reply(text, rec, STEEL, context=q) == [], q
    assert "REJECT" in adv.ask("Can I paint over it and ship it?")
    adv.ask("What is mill scale?")
    assert adv.answered_by.startswith("llama")
