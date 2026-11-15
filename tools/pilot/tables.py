"""Markdown tables for reports/pilot_lighting.md from the evaluated matrix:
  tables.py <matrix dir> <eval box json> <eval app json | -> <invariance2 json> <truth json> [<truth json> ...]
Gate + oracle table (both arms), image table (both arms), truth-error table (burst), per row."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

mx, box = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
app = json.loads(Path(sys.argv[3]).read_text()) if sys.argv[3] != "-" else None
inv = json.loads(Path(sys.argv[4]).read_text())
truth = [t for p in sys.argv[5:] for t in json.loads(Path(p).read_text())]
NAME = {"C0": "C0 dark", "C1": "C1 office", "C2": "C2 warm bulb", "C3": "C3 window", "C4": "C4 side lamp + shadow",
        "C5": "C5 lamp reflection", "C6": "C6 sun"}


def rows_of(rep, arm):
    for res in rep["results"]:
        if res["arm"] == arm:
            return {r["tag"]: r for r in res["rows"]}
    return {}


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def frac(d):
    return f"{d['k']}/{d['n']}" if d and d.get("n") else "n/a"


# final gate per capture from the matrix sidecars (truth json gates may predate a gate change)
gate = {}
for sc in (mx / "sidecar").glob("*.json"):
    s = json.loads(sc.read_text())
    for v in s["variants"]:
        b = v["arms"]["burst"]
        gate[(s.get("scene_id") or sc.stem, v["k"])] = (b["gate"], b["reasons"])
caps = defaultdict(list)
for c in box["captures"]:
    if c["arm"] == "burst":
        caps[c["tag"]].append(c)
B, N = rows_of(box, "burst"), rows_of(box, "naive")
BA, NA = (rows_of(app, "burst"), rows_of(app, "naive")) if app else ({}, {})
tags = sorted(B, key=lambda t: (t[-1] != "h", t))
for mode in ("h", "o"):
    print(f"\n### {'HOOD' if mode == 'h' else 'OPEN'}: gate, oracle detection and verdicts\n")
    print("| Row | Median lux | Burst retakes (reasons) | Burst P / R / F1 | Burst verdict = C0 | Burst silent | "
          "Naive P / R / F1 | Naive verdict = C0 | Naive silent | App measures: burst / naive verdict = C0, silent |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t in [x for x in tags if x.endswith(mode)]:
        b, n = B[t], N[t]
        reasons = Counter(r for c in caps[t] if c["gate"] == "RETAKE" for r in c["gate_reasons"])
        rs = ", ".join(f"{k} {v}" for k, v in reasons.most_common()) or "-"
        prf = lambda r: "n/a" if r["f1"]["value"] is None else f"{f(r['precision']['value'])} / {f(r['recall']['value'])} / {f(r['f1']['value'])}"   # noqa: E731
        a = (f"{frac(BA[t]['verdict_agreement_c0'])}, {BA[t]['silent']['k']} / {frac(NA[t]['verdict_agreement_c0'])}, "
             f"{NA[t]['silent']['k']}") if app else "-"
        print(f"| {t} {NAME[t[:2]]} | {b['lux_median']:,.0f} | {b['retake']['k']}/{b['retake']['n']} ({rs}) | {prf(b)} | "
              f"{frac(b['verdict_agreement_c0'])} | {b['silent']['k']} | {prf(n)} | {frac(n['verdict_agreement_c0'])} | "
              f"{n['silent']['k']} | {a} |")
    print(f"\n### {'HOOD' if mode == 'h' else 'OPEN'}: images against the dark room (accepted captures)\n")
    print("| Row | Burst SSIM vs C0 | Burst mean abs diff vs C0 (DN) | Burst low-pass abs diff (DN) | Naive SSIM vs C0 | "
          "Naive mean abs diff vs C0 (DN) | Naive low-pass abs diff (DN) |")
    print("|---|---|---|---|---|---|---|")
    for t in [x for x in tags if x.endswith(mode)]:
        b, n = B[t], N[t]
        ib, inn = inv["rows"].get(f"{t}/burst"), inv["rows"].get(f"{t}/naive")
        sb = (b.get("ssim_c0") or {}).get("mean")
        sn = (n.get("ssim_c0") or {}).get("mean")
        cell = lambda i, key: "0 (reference)" if t.startswith("C0") else ("n/a" if i is None else f"{i[key]:.2f} (max {i[key + '_max'] if key + '_max' in i else i[key]:.2f})")  # noqa: E731
        print(f"| {t} | {f(sb, 3)} | {cell(ib, 'mad')} | {cell(ib, 'lp')} | {f(sn, 3)} | {cell(inn, 'mad')} | {cell(inn, 'lp')} |")
print("\n### Burst images against the noise-free LED image of the same render (1.1 mm low-pass)\n")
print("| Row | Accepted | DN error p1 .. p99 (worst capture) | Largest local DN error | Relative error p1 .. p99 (worst) |")
print("|---|---|---|---|---|")
tr = defaultdict(list)
for t in truth:
    g = gate.get((t["scene"], t["k"]), (t["gate"], t["reasons"]))
    tr[t["tag"]].append({**t, "gate": g[0]})
for tag in sorted(tr, key=lambda t: (t[-1] != "h", t)):
    acc = [x for x in tr[tag] if x["gate"] != "RETAKE"]
    if not acc:
        print(f"| {tag} | 0/{len(tr[tag])} | - | - | - |")
        continue
    print(f"| {tag} | {len(acc)}/{len(tr[tag])} | {min(x['dn_p1'] for x in acc):+.1f} .. {max(x['dn_p99'] for x in acc):+.1f} | "
          f"{max(x['dn_absmax'] for x in acc):.0f} | {min(x['rel_p1'] for x in acc):+.0%} .. {max(x['rel_p99'] for x in acc):+.0%} |")
print("\nFloor (two dark-room captures, independent noise and shake):",
      {k: {kk: v[kk] for kk in ("mad", "lp", "sigma", "expected_mad_from_sigma", "n")} for k, v in inv["floor"].items()})
