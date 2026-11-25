"""Narration with Indic Parler-TTS (ai4bharat/indic-parler-tts) on a GPU box.

1. Speaker pick: each candidate speaks the first line; keep speakers with a male pitch (median F0 < 165 Hz),
   rank by Whisper word error rate; samples are saved for listening.
2. Every sentence: up to N takes with different seeds; keep the take with the lowest WER.
usage: python tts_parler.py narration_tts.json out_dir [--speaker NAME]
"""
import json, re, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, pipeline

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CANDS = ["Thoma", "Dinesh", "Aakash", "Raghav", "Jatin", "Kabir", "Pranav", "Rahul"]
DESC = ("{s}'s voice is warm, confident and enthusiastic, with expressive, animated intonation, like a product presenter "
        "at a launch. He speaks Indian English at a moderately fast pace. The recording is very clear and close-up, "
        "with no background noise.")
TAKES = int(__import__("os").environ.get("TAKES", 3))


def norm(t):
    t = t.lower().replace("-", " ")
    return re.sub(r"[^a-z0-9 ]", " ", t).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def f0_median(a, sr):
    import librosa
    y = librosa.resample(a, orig_sr=sr, target_sr=16000)
    f0, vf, _ = librosa.pyin(y, fmin=60, fmax=400, sr=16000)
    f0 = f0[vf] if vf is not None else f0
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else 0.0


def main():
    narr = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    forced = sys.argv[sys.argv.index("--speaker") + 1] if "--speaker" in sys.argv else None
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(DEV)
    tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0 if DEV == "cuda" else -1)

    def say(text, spk, seed):
        torch.manual_seed(seed)
        d = dtok(DESC.format(s=spk), return_tensors="pt").to(DEV)
        p = tok(text, return_tensors="pt").to(DEV)
        g = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=p.input_ids,
                           prompt_attention_mask=p.attention_mask, do_sample=True, temperature=1.0)
        return g.cpu().numpy().squeeze().astype(np.float32)

    def score(a, text):
        import librosa
        y = librosa.resample(a, orig_sr=sr, target_sr=16000)
        return wer(text, asr({"raw": y, "sampling_rate": 16000})["text"])

    report = {"speakers": {}, "lines": {}}
    first = narr[0][1].split(". ")[1] if ". " in narr[0][1] else narr[0][1]
    if forced:
        spk = forced
    else:
        rows = []
        for s in CANDS:
            a = say(first, s, 1)
            sf.write(out / f"sample_{s}.wav", a, sr)
            f0, w = f0_median(a, sr), score(a, first)
            report["speakers"][s] = {"f0_hz": round(f0, 1), "wer": round(w, 3), "dur_s": round(len(a) / sr, 2)}
            print("speaker", s, report["speakers"][s], flush=True)
            if 0 < f0 < 165:
                rows.append((w, CANDS.index(s), s))
        spk = sorted(rows)[0][2] if rows else CANDS[0]
    report["chosen"] = spk
    print("chosen", spk, flush=True)
    gap = np.zeros(int(0.16 * sr), np.float32)
    for key, text in narr:
        sents = [x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
        parts, info = [], []
        for si, s in enumerate(sents):
            best = None
            for k in range(TAKES):
                a = say(s, spk, 100 + 10 * si + k)
                # trim leading/trailing near-silence
                e = np.abs(a) > 0.01
                if e.any():
                    i0, i1 = max(0, e.argmax() - int(0.03 * sr)), len(a) - max(0, e[::-1].argmax() - int(0.08 * sr))
                    a = a[i0:i1]
                w = score(a, s)
                if best is None or w < best[0]:
                    best = (w, a, k)
                if w == 0:
                    break
            parts += [best[1], gap]; info.append({"sentence": s, "wer": round(best[0], 3), "take": best[2]})
            print(key, si, "wer", round(best[0], 3), flush=True)
        a = np.concatenate(parts[:-1])
        a = a / max(1e-6, np.abs(a).max()) * 0.89
        sf.write(out / f"{key}.wav", a, sr)
        report["lines"][key] = info
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print("done", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
