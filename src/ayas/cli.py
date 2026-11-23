"""Command line: calibrate -> golden -> inspect (+ chat); penetrant for red-dye crack checks;
dataset/train for the classifier.

The same commands drive the real rig and the simulator; the config decides which.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import VIEWS, Config, load_config

DEFAULT_CONFIG = os.environ.get("AYAS_CONFIG", "configs/rig.sim.yaml")


def _rig(cfg: Config, seed: int | None = None):
    from .rig import build_rig

    return build_rig(cfg, seed=seed)


def _defects_arg(value: str | None):
    if value in (None, "none"):
        return None
    if value == "random":
        return "random"
    return [v.strip() for v in value.split(",") if v.strip()]


def _capture(cfg: Config, rig, states=None):
    from .capture import capture_set

    return capture_set(rig.lights, rig.cameras, states or cfg.states, settle_s=cfg.capture.get("settle_s", 0.05))


def cmd_calibrate(cfg: Config, args) -> None:
    from .calibration import calibrate
    from .capture import bright_view

    rig = _rig(cfg, args.seed)
    try:
        print("Calibration: a good plate in the jig; every light in turn, then dark.")
        rig.place_part(None)
        frames = _capture(cfg, rig)
        cal = calibrate({v: bright_view(frames, v, cfg.lights) for v in VIEWS}, cfg)
    finally:
        rig.close()
    cal.save(cfg.calibration_path)
    for v in VIEWS:
        print(f"  {v:10s} {cal.methods[v]:16s} corners {np.round(cal.image_corners[v], 1).tolist()}")
    print(f"saved {cfg.calibration_path}")


def cmd_golden(cfg: Config, args) -> None:
    from .calibration import Calibration
    from .golden import GoldenBuilder
    from .shape import shape_setup

    cal = Calibration.load(cfg.calibration_path)
    builder = GoldenBuilder(cfg, cal)
    rig = _rig(cfg, args.seed)
    try:
        for i in range(args.count):
            print(f"good part {i + 1}/{args.count}")
            rig.place_part(None)
            builder.add(_capture(cfg, rig))
    finally:
        rig.close()
    golden = builder.finalize()
    golden.save(cfg.golden_path)
    roi = cfg.grid.plate_mask(cfg.grid.edge_margin_mm)
    setup = shape_setup(cfg)
    print(f"saved {cfg.golden_path} from {golden.count} parts; 3D shape: {setup.mode} ({setup.reason})")
    for k in golden.keys:
        print(f"  {k:18s} mean {np.median(golden.mean[k][roi]):+.3f}  spread {np.median(golden.std[k][roi]):.4f}")


def _print_report(report: dict) -> None:
    mode = report.get("mode", "surface")
    print(f"\n{report['part_id']} [{mode}]: {report['verdict']}  ({report['summary']['defect_count']} found)")
    for d in report["defects"]:
        loc = d["location_mm"]
        extra = ""
        if d.get("depth_mm") is not None and d["type"] in ("dent", "bump"):
            extra = f" depth {d['depth_mm']:+.2f} mm"
        if mode == "penetrant":
            extra = f" {d['size']['length_mm']:.1f} x {d['size']['width_mm']:.1f} mm" + ("" if d["relevant"] else " (non-relevant)")
        print(
            f"  {d['id']:4s} {d['type']:18s} conf {d['confidence']:.2f}  at ({loc['x']:5.1f}, {loc['y']:5.1f}) mm  "
            f"{d['zone']:12s} {d['severity']:12s} {d['disposition']}{extra}"
        )
    print("  timing ms: " + ", ".join(f"{k} {v:.0f}" for k, v in report["timing_ms"].items()))


def _chat(cfg: Config, report: dict, questions: list[str], interactive: bool, inspection_id: int | None = None) -> None:
    from .llm import Advisor, build_backend
    from .rules import load_spec
    from .store import Store

    advisor = Advisor(build_backend(cfg.llm, cfg.root), load_spec(cfg.spec_path), report, cfg.llm.get("history_turns", 4))
    store = Store(cfg.db_path) if inspection_id is not None else None

    def say(role: str, text: str) -> None:
        if store:
            store.add_chat(inspection_id, role, text)

    t = time.perf_counter()
    text = advisor.explain()
    print(f"\n[assistant, {time.perf_counter() - t:.1f} s]\n{text}")
    say("assistant", text)
    for q in questions:
        print(f"\n[operator] {q}")
        t = time.perf_counter()
        reply = advisor.ask(q)
        print(f"[assistant, {time.perf_counter() - t:.1f} s]\n{reply}")
        say("operator", q)
        say("assistant", reply)
    while interactive:
        try:
            q = input("\n[operator] ").strip()
        except EOFError:
            break
        if q.lower() in ("", "q", "quit", "exit"):
            break
        reply = advisor.ask(q)
        print(f"[assistant]\n{reply}")
        say("operator", q)
        say("assistant", reply)
    if store:
        store.close()


def cmd_inspect(cfg: Config, args) -> None:
    from .capture import save_frames
    from .pipeline import Inspector
    from .store import Store

    inspector = Inspector(cfg)
    rig = _rig(cfg, args.seed)
    part_id = args.part_id or datetime.now().strftime("P%Y%m%d-%H%M%S")
    try:
        truth = rig.place_part(_defects_arg(args.defects)) if cfg.is_sim else rig.place_part(None)
        t = time.perf_counter()
        frames = _capture(cfg, rig)
        capture_ms = (time.perf_counter() - t) * 1e3
    finally:
        rig.close()
    capture_dir = cfg.captures_dir / part_id
    if not args.no_save:
        save_frames(frames, capture_dir)
    report, _ = inspector.inspect(
        frames, part_id, capture_dir=capture_dir, overlay_path=cfg.reports_dir / f"{part_id}.png", extra_timing_ms={"capture": capture_ms}
    )
    if cfg.is_sim:
        report["sim_ground_truth"] = [d.to_dict() for d in truth]
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / f"{part_id}.json"
    report_path.write_text(json.dumps(report, indent=2))
    store = Store(cfg.db_path)
    inspection_id = store.add_inspection(report)
    store.close()
    _print_report(report)
    if cfg.is_sim and truth:
        print("  simulated truth: " + ", ".join(f"{d.kind} at ({d.x_mm:.1f}, {d.y_mm:.1f})" for d in truth))
    print(f"  report {report_path}\n  overlay {report['overlay']}")
    if not args.no_llm:
        _chat(cfg, report, args.ask or [], args.chat, inspection_id)


def cmd_penetrant(cfg: Config, args) -> None:
    """Red-dye penetrant check of a developed plate: find, measure and grade indications."""
    from .capture import save_frames
    from .penetrant import PenetrantInspector
    from .store import Store

    inspector = PenetrantInspector(cfg, args.profile)
    rig = _rig(cfg, args.seed)
    part_id = args.part_id or datetime.now().strftime("P%Y%m%d-%H%M%S") + "-PT"
    process = {
        "system": args.system,
        "type": "visible (Type II)",
        "method": "solvent-removable",
        "dwell_min": args.dwell,
        "development_min": args.develop,
    }
    try:
        truth = rig.place_developed(_defects_arg(args.indications)) if cfg.is_sim else rig.place_developed(None)
        t = time.perf_counter()
        frames = _capture(cfg, rig, (cfg.pair_light("top"), "dark"))
        capture_ms = (time.perf_counter() - t) * 1e3
    finally:
        rig.close()
    capture_dir = cfg.captures_dir / part_id
    if not args.no_save:
        save_frames(frames, capture_dir)
    report = inspector.inspect(
        frames, part_id, process, capture_dir=capture_dir, overlay_path=cfg.reports_dir / f"{part_id}.png", extra_timing_ms={"capture": capture_ms}
    )
    if cfg.is_sim:
        report["sim_ground_truth"] = [d.to_dict() for d in truth]
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / f"{part_id}.json"
    report_path.write_text(json.dumps(report, indent=2))
    store = Store(cfg.db_path)
    inspection_id = store.add_inspection(report)
    store.close()
    _print_report(report)
    for w in report["penetrant"]["warnings"]:
        print(f"  warning: {w}")
    if cfg.is_sim and truth:
        print("  simulated truth: " + ", ".join(f"{d.kind} at ({d.x_mm:.1f}, {d.y_mm:.1f})" for d in truth))
    print(f"  profile {report['penetrant']['profile']}; report {report_path}\n  overlay {report['overlay']}")
    if not args.no_llm:
        _chat(cfg, report, args.ask or [], args.chat, inspection_id)


def cmd_chat(cfg: Config, args) -> None:
    report = json.loads(Path(args.report).read_text())
    _print_report(report)
    _chat(cfg, report, args.ask or [], interactive=True)


def cmd_dataset(cfg: Config, args) -> None:
    """Simulator: auto-labelled regions. Real rig: unlabelled rows to label by hand."""
    from .features import FEATURE_NAMES
    from .pipeline import Inspector

    inspector = Inspector(cfg)
    rig = _rig(cfg, args.seed)
    rows = []
    try:
        for i in range(args.parts):
            truth = rig.place_part("random") if cfg.is_sim else rig.place_part(None)
            analysis = inspector.analyse(_capture(cfg, rig))
            for f in analysis.features:
                label = ""
                if cfg.is_sim:
                    near = [d for d in truth if np.hypot(d.x_mm - f.x_mm, d.y_mm - f.y_mm) <= max(3.0, d.size_mm)]
                    if not near:
                        continue
                    label = near[0].kind
                rows.append({"part": i, "region": f.region_id, "x_mm": round(f.x_mm, 2), "y_mm": round(f.y_mm, 2), "label": label, **f.values})
            print(f"part {i + 1}/{args.parts}: {len(analysis.features)} regions")
    finally:
        rig.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["part", "region", "x_mm", "y_mm", "label", *FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} regions to {out}" + ("" if cfg.is_sim else " - fill in the label column"))


def cmd_evaluate(cfg: Config, args) -> None:
    """Simulator only: detection recall/precision and the type confusion matrix on fresh random parts."""
    from .classify import CLASSES
    from .pipeline import Inspector
    from .sim import DEFECT_KINDS

    if not cfg.is_sim:
        raise SystemExit("evaluate needs the simulator; on the rig, label a dataset and use train's cross-validation")
    inspector = Inspector(cfg)
    rig = _rig(cfg, args.seed)
    confusion = {t: {p: 0 for p in CLASSES} for t in DEFECT_KINDS}
    missed = {t: 0 for t in DEFECT_KINDS}
    false_pos = good_parts_flagged = 0
    for i in range(args.parts):
        truth = rig.place_part("random" if i % 5 else None)  # every 5th part is a good one
        analysis = inspector.analyse(_capture(cfg, rig))
        used = set()
        for d in truth:
            near = [
                (np.hypot(d.x_mm - f.x_mm, d.y_mm - f.y_mm), j)
                for j, f in enumerate(analysis.features)
                if np.hypot(d.x_mm - f.x_mm, d.y_mm - f.y_mm) <= max(3.0, d.size_mm)
            ]
            if not near:
                missed[d.kind] += 1
                continue
            j = min(near)[1]
            used.add(j)
            confusion[d.kind][analysis.predictions[j].label] += 1
        extra = len(analysis.features) - len(used)
        false_pos += extra
        good_parts_flagged += int(not truth and extra > 0)
    print(f"{args.parts} parts ({args.parts // 5 + (args.parts % 5 > 0)} good), classifier {inspector.classifier.name}")
    print("truth \\ predicted " + " ".join(f"{c[:7]:>7s}" for c in CLASSES) + "   missed  recall")
    total = correct = 0
    for t in DEFECT_KINDS:
        row = confusion[t]
        n = sum(row.values()) + missed[t]
        total += n
        correct += row[t]
        recall = row[t] / n if n else float("nan")
        print(f"{t:17s} " + " ".join(f"{row[p]:7d}" for p in CLASSES) + f"   {missed[t]:6d}  {recall:6.2f}")
    print(f"overall correct type: {correct}/{total} = {correct / max(total, 1):.3f}")
    print(f"extra regions (false positives): {false_pos}; good parts wrongly flagged: {good_parts_flagged}")


def cmd_train(cfg: Config, args) -> None:
    from sklearn.model_selection import cross_val_score

    from .classify import ForestClassifier
    from .features import FEATURE_NAMES

    X, y = [], []
    for path in args.csv:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row["label"]:
                    X.append([float(row[n]) for n in FEATURE_NAMES])
                    y.append(row["label"])
    X = np.array(X, np.float32)
    clf = ForestClassifier().fit(X, y)
    counts = {c: y.count(c) for c in sorted(set(y))}
    print(f"trained on {len(y)} regions: {counts}")
    folds = min(5, min(counts.values()))
    if folds >= 2:
        scores = cross_val_score(clf.model, X, y, cv=folds)
        print(f"{folds}-fold accuracy {scores.mean():.3f} +/- {scores.std():.3f}")
    clf.save(cfg.classifier_path)
    print(f"saved {cfg.classifier_path}")


def cmd_history(cfg: Config, args) -> None:
    from .store import Store

    store = Store(cfg.db_path)
    for row in store.recent(args.n):
        print("  ".join(str(x) for x in row))
    print(store.verdict_counts())
    store.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ayas", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"rig config (default {DEFAULT_CONFIG})")
    parser.add_argument("--seed", type=int, default=None, help="simulator seed")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("calibrate", help="map each camera onto the top-down grid")
    p = sub.add_parser("golden", help="build the golden reference from good parts")
    p.add_argument("--count", type=int, default=20)
    p = sub.add_parser("inspect", help="capture every camera under every light, find and judge defects, explain")
    p.add_argument("--part-id")
    p.add_argument("--defects", help="simulator only: none | random | comma list, e.g. scratch,stain")
    p.add_argument("--ask", action="append", help="follow-up question (repeatable)")
    p.add_argument("--chat", action="store_true", help="interactive follow-up questions")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-save", action="store_true", help="do not keep the raw frames")
    p = sub.add_parser("penetrant", help="red-dye penetrant check of a developed plate")
    p.add_argument("--part-id")
    p.add_argument("--profile", help="acceptance profile from spec.yaml (default: spec's penetrant.profile)")
    p.add_argument("--system", default="visible red-dye kit", help="penetrant kit used, for the log")
    p.add_argument("--dwell", type=float, default=10.0, help="penetrant dwell time, min")
    p.add_argument("--develop", type=float, default=10.0, help="development time before imaging, min")
    p.add_argument("--indications", help="simulator only: none | random | comma list of crack,slit,pore,pore_line,smear")
    p.add_argument("--ask", action="append", help="follow-up question (repeatable)")
    p.add_argument("--chat", action="store_true", help="interactive follow-up questions")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-save", action="store_true", help="do not keep the raw frames")
    p = sub.add_parser("chat", help="ask about a saved report")
    p.add_argument("report")
    p.add_argument("--ask", action="append")
    p = sub.add_parser("dataset", help="collect region features for training")
    p.add_argument("--parts", type=int, default=60)
    p.add_argument("--out", default="data/regions.csv")
    p = sub.add_parser("evaluate", help="simulator: accuracy on fresh random parts")
    p.add_argument("--parts", type=int, default=100)
    p = sub.add_parser("train", help="train the random-forest classifier")
    p.add_argument("csv", nargs="+")
    p = sub.add_parser("history", help="recent inspections")
    p.add_argument("-n", type=int, default=10)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    commands = {
        "calibrate": cmd_calibrate,
        "golden": cmd_golden,
        "inspect": cmd_inspect,
        "penetrant": cmd_penetrant,
        "chat": cmd_chat,
        "dataset": cmd_dataset,
        "evaluate": cmd_evaluate,
        "train": cmd_train,
        "history": cmd_history,
    }
    commands[args.command](cfg, args)


if __name__ == "__main__":
    main(sys.argv[1:])
