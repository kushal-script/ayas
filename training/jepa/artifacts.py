"""Verify a JarvisLabs artifact folder and write its MANIFEST.md (standing rule: download everything
worth keeping and check it BEFORE an instance is destroyed or paused).

  .venv-train/bin/python training/jepa/artifacts.py artifacts/jarvislabs/<instance-name>_<run_id> \
      [--require encoder.pt summary.json metrics.jsonl] [--note "..."]

Checks: every listed file exists and is non-empty; every *.pt loads with torch.load, and JEPA
checkpoints (a "model" state dict + "cfg") rebuild with jepa.model.load_jepa and run one forward
pass; every *.json parses. Writes MANIFEST.md (file, size, what it is, check result) and
verify.json. Exit code 0 only if everything required is present and loads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DESCRIBE = [
    ("encoder.pt", "JEPA checkpoint for eval/export: context + EMA target encoder + predictor state dict, cfg, step, summary"),
    ("last.pt", "resume checkpoint: model + AdamW state + step + cfg"),
    ("summary.json", "train.py summary: steps, seconds, GPU, torch version, data counts, last metrics"),
    ("metrics.jsonl", "train.py log: one JSON line per log step (loss, hidden / visible-cross loss, spread, lr, ema, img/s)"),
    ("config.yaml", "resolved training config of the run"),
    ("output.log", "jl managed-run stdout+stderr (<home>/jl-runs/<run_id>/output.log)"),
    ("run_logs_full.txt", "`jl run logs <run_id>` output (full)"),
    ("run_log_tail.txt", "`jl run logs <run_id> --tail 400` output"),
    ("run_status.json", "`jl run status <run_id> --json`"),
    ("jl_runner.log", "the local runner's log (jl_train.sh): every step with timestamps"),
    ("commands.txt", "the exact jl commands the runner executed, in order"),
    ("code.tgz", "the exact code bundle uploaded to the instance"),
    ("code_sha256.txt", "sha256 of every file in the code bundle"),
    ("remote_env.txt", "nvidia-smi, torch / timm versions and pip freeze on the instance"),
    ("pairs_manifest.json", "the paired-plate manifest that regenerates the training data (training/jepa/pairs.py)"),
    ("matrix_manifest.json", "the test-matrix manifest that regenerates the test data"),
    ("instance.json", "machine id, instance wall time, lifecycle action"),
    ("run.json", "run id, final state, run seconds, GPU, spot"),
    ("machine_id", "JarvisLabs machine id"),
    ("run_id", "jl managed-run id"),
    ("data_manifest.json", "what the packed dataset held (grid count and bytes); the data itself is not kept"),
    ("verify.json", "this check's machine-readable result"),
]


def describe(name: str) -> str:
    for pat, d in DESCRIBE:
        if name == pat or name.endswith(pat):
            return d
    if name.endswith(".jpg") or name.endswith(".png"):
        return "sample output / figure"
    if name.endswith(".log") or name.endswith(".txt"):
        return "log"
    return ""


def check_pt(p: Path) -> str:
    try:
        import torch
    except ImportError:
        return "not checked (no torch in this python)"
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        return f"FAILED to load: {type(e).__name__}: {e}"
    if isinstance(ck, dict) and "model" in ck and "cfg" in ck:
        try:
            from jepa.model import load_jepa

            m, _ = load_jepa(str(p))
            with torch.no_grad():
                out = m.target_encoder(torch.zeros(1, 3, 224, 224))
            ok = bool(torch.isfinite(out).all())
            return f"loads; JEPA rebuilt (step {ck.get('step')}), forward {tuple(out.shape)} {'finite' if ok else 'NON-FINITE'}"
        except Exception as e:  # noqa: BLE001
            return f"FAILED to rebuild JEPA: {type(e).__name__}: {e}"
    return f"loads ({type(ck).__name__})"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(folder: Path, require: Sequence[str], note: str = "") -> dict:
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name not in ("MANIFEST.md", "verify.json"))
    rows, problems = [], []
    for p in files:
        rel = p.relative_to(folder).as_posix()
        size = p.stat().st_size
        chk = "ok" if size > 0 else "EMPTY"
        if p.suffix == ".pt":
            chk = check_pt(p)
        elif p.suffix == ".json":
            try:
                json.loads(p.read_text())
            except Exception as e:  # noqa: BLE001
                chk = f"FAILED to parse: {e}"
        if chk.startswith("FAILED") or chk == "EMPTY" or "NON-FINITE" in chk:
            problems.append(f"{rel}: {chk}")
        rows.append({"file": rel, "bytes": size, "what": describe(p.name), "check": chk,
                     "sha256": sha256(p) if size < 500e6 else None})
    names = {Path(r["file"]).name for r in rows}
    missing = [r for r in require if r not in names]
    problems += [f"missing required file {m}" for m in missing]
    res = {"folder": str(folder), "ok": not problems, "problems": problems, "files": rows}
    L = [f"# Artifacts: {folder.name}\n", note + "\n" if note else "",
         f"Verified: **{'OK' if res['ok'] else 'PROBLEMS'}**" + ("" if res["ok"] else ": " + "; ".join(problems)) + "\n",
         "| file | size | what it is | check |", "|---|---|---|---|"]
    for r in rows:
        sz = r["bytes"]
        h = f"{sz / 1e6:.2f} MB" if sz >= 1e5 else f"{sz / 1e3:.1f} kB"
        L.append(f"| `{r['file']}` | {h} | {r['what']} | {r['check']} |")
    L.append(f"\nTotal: {sum(r['bytes'] for r in rows) / 1e6:.1f} MB in {len(rows)} files. sha256 in verify.json.")
    (folder / "MANIFEST.md").write_text("\n".join(x for x in L if x is not None) + "\n")
    (folder / "verify.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--require", nargs="*", default=["encoder.pt", "summary.json", "metrics.jsonl"])
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    res = verify(Path(a.folder), a.require, a.note)
    for r in res["files"]:
        print(f"{r['bytes']:>12,d}  {r['file']}  [{r['check']}]")
    print("VERIFY", "OK" if res["ok"] else "FAILED: " + "; ".join(res["problems"]))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
