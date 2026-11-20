"""Assemble the lighting demo video: one 1280 x 720 card per lighting condition, a few seconds each.

Each condition card shows the Blender overview render of the scene (the device over the
plate in that light, if one is found), the old method's grid (naive arm) and the new
method's grid (burst arm, + hood where used) with ground-truth boxes (green) and
detections (red), the verdicts, the quality-gate message and a caption such as
"Direct sun: old method fails / new method + hood passes". A title card and a verdict
card open and close the video.

Inputs:
- --report reports/lighting_matrix.json (training/eval_lighting.py): captures with grid
  paths, boxes, gate and verdicts. --plate picks the plate (default: the first plate with
  a defect).
- --overviews DIR: overview renders, found by scene id (<scene>.png, as pipeline.run_plans
  --overview keeps them in out/overview/; <scene>*overview*.png; <scene>/overview.png) or by
  condition (overview_<condition>.png); optional.
- or --cards cards.json: a list of cards {title, caption, overview, gate, panels:
  [{label, image, boxes: [[x0, y0, x1, y1, "label", "gt"|"det"]], status, ok}]}.

ffmpeg (default /opt/homebrew/bin/ffmpeg, else on PATH) encodes the cards, piped as raw
frames, to H.264 yuv420p at 25 fps.

Example:
  .venv/bin/python tools/make_demo_video.py --report reports/lighting_matrix.json \\
      --overviews data/twin_matrix/overview --out reports/demo_lighting.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

W, H = 1280, 720
BG, FG, DIM = (24, 24, 28), (240, 240, 240), (170, 170, 170)
GREEN, RED, AMBER = (60, 200, 60), (60, 60, 230), (0, 170, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CLASSES_FALLBACK = ("scratch", "dent", "pit", "corrosion", "crack", "stain")
NAMES = {"C0_dark": "Dark room", "C1_office": "Office ceiling light", "C2_warm_bulb": "Warm bulb close by",
         "C3_window": "Window daylight", "C4_side_lamp": "Low side lamp + hand shadow",
         "C5_reflected_lamp": "Lamp reflected in the plate", "C6_sun": "Direct sun"}
ASCII = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-",
                       "≥": ">=", "≤": "<=", "°": " deg", "×": "x", "→": "->"})


def ascii_text(s: Any) -> str:
    """OpenCV's Hershey fonts draw ASCII only."""
    return str(s).translate(ASCII).encode("ascii", "replace").decode()


def find_ffmpeg(path: str | None = None) -> str | None:
    for cand in (path, "/opt/homebrew/bin/ffmpeg", shutil.which("ffmpeg")):
        if cand and Path(cand).exists():
            return str(cand)
    return None


# --- Drawing -----------------------------------------------------------------------------

def put(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.6, colour=FG, thick: int = 1) -> None:
    cv2.putText(img, ascii_text(text), xy, FONT, scale, colour, thick, cv2.LINE_AA)


def wrap(text: str, width_px: int, scale: float, thick: int = 1) -> list[str]:
    """Greedy word wrap by measured text width; blank lines keep paragraph breaks."""
    out: list[str] = []
    for para in ascii_text(text).split("\n"):
        line = ""
        for word in para.split():
            cand = f"{line} {word}".strip()
            if line and cv2.getTextSize(cand, FONT, scale, thick)[0][0] > width_px:
                out.append(line)
                line = word
            else:
                line = cand
        out.append(line)
    return out


def put_wrapped(img: np.ndarray, text: str, x: int, y: int, width_px: int, scale: float = 0.55, colour=FG,
                line_h: int = 24, max_lines: int = 20) -> int:
    """Word-wrapped text (paragraphs on newlines); returns the y after the last line."""
    for line in wrap(text, width_px, scale)[:max_lines]:
        if line:
            put(img, line, (x, y), scale, colour)
        y += line_h
    return y


def fit(img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float]:
    """Scale an image to fit in w x h, keeping its aspect; returns (image, scale)."""
    s = min(w / img.shape[1], h / img.shape[0])
    size = (max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR), s


def read_bgr(path: str | Path | None) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    dst[y : y + h, x : x + w] = src


def panel(p: dict[str, Any], w: int, h: int) -> np.ndarray:
    """One labelled image with boxes and a status line, w x h, BGR."""
    out = np.full((h, w, 3), BG, np.uint8)
    put(out, p.get("label", ""), (6, 22), 0.6, FG, 1)
    img = read_bgr(p.get("image"))
    body_h = h - 32 - 62                    # label above, two status lines below
    if img is not None:
        small, s = fit(img, w - 8, body_h)
        x0, y0 = (w - small.shape[1]) // 2, 32
        for b in p.get("boxes") or []:
            colour = GREEN if (len(b) > 5 and b[5] == "gt") else RED
            bx0, by0, bx1, by1 = (int(round(v * s)) for v in b[:4])
            cv2.rectangle(small, (bx0, by0), (max(bx1, bx0 + 1), max(by1, by0 + 1)), colour, 2)
            if len(b) > 4 and b[4] and colour == RED:
                put(small, b[4], (bx0, max(12, by0 - 4)), 0.4, colour)
        paste(out, small, x0, y0)
    else:
        put(out, "(no image)", (w // 2 - 50, body_h // 2), 0.6, DIM)
    ok = p.get("ok")
    colour = GREEN if ok else (AMBER if ok is None else RED)
    put_wrapped(out, p.get("status", ""), 6, h - 38, w - 12, 0.55, colour, 22, 2)
    return out


def render_table(img: np.ndarray, table: dict[str, Any], y: int) -> int:
    """A text table: {"cols": [x, ...], "header": [...], "rows": [[cell, ...], ...]}; a cell is a string or
    [string, "good" | "bad" | "dim"]. Returns the y below the table."""
    cols = table["cols"]
    tone = {"good": GREEN, "bad": RED, "dim": DIM}
    for i, h in enumerate(table.get("header") or []):
        put(img, h, (cols[i], y), 0.55, AMBER, 1)
    y += 12
    cv2.line(img, (cols[0], y), (W - 30, y), DIM, 1)
    for row in table.get("rows") or []:
        y += 40
        for i, cell in enumerate(row):
            text, colour = (cell, FG) if isinstance(cell, str) else (cell[0], tone.get(cell[1], FG))
            put(img, text, (cols[i], y), 0.6, colour, 1)
    return y + 12


def render_card(card: dict[str, Any]) -> np.ndarray:
    """A 1280 x 720 BGR card: title, caption, optional overview, panels side by side, gate message."""
    img = np.full((H, W, 3), BG, np.uint8)
    put(img, card.get("title", ""), (30, 50), 1.1, FG, 2)
    if card.get("caption"):
        put(img, card["caption"], (30, 88), 0.7, AMBER, 2)
    y_text = 140
    if card.get("table"):
        y_text = render_table(img, card["table"], 140) + 24
    if card.get("text"):
        put_wrapped(img, card["text"], 30, y_text, W - 60, 0.6, FG, 28, 18)
    x, top, bottom = 30, 110, H - 60
    ov = read_bgr(card.get("overview"))
    if ov is not None:
        small, _ = fit(ov, 360, bottom - top - 24)
        put(img, "Twin render (device + light)", (x, top + 16), 0.5, DIM)
        paste(img, small, x, top + 24)
        x += 380
    panels = card.get("panels") or []
    if panels:
        pw = (W - x - 30) // len(panels)
        for i, p in enumerate(panels):
            paste(img, panel(p, pw - 10, bottom - top), x + i * pw, top)
    if card.get("gate"):
        put_wrapped(img, "Device says: " + card["gate"], 30, H - 28, W - 60, 0.6, AMBER, 24, 1)
    return img


# --- Cards from the lighting report ------------------------------------------------------

def _classes(rep: dict[str, Any]) -> Sequence[str]:
    return rep.get("classes") or CLASSES_FALLBACK


def capture_panel(cap: dict[str, Any], label: str, classes: Sequence[str]) -> dict[str, Any]:
    boxes = [[*g[1:5], "", "gt"] for g in cap.get("gt") or [] if g[-1]]
    boxes += [[*d[2:6], classes[d[0]] if 0 <= d[0] < len(classes) else str(d[0]), "det"] for d in cap.get("dets") or []]
    if cap["pred_verdict"] == "RETAKE":
        status, ok = f"RETAKE (refused safely): {cap.get('gate_message') or 'retake'}", None
    else:
        right = cap["pred_verdict"] == cap["gt_verdict"]
        status = f"{cap['pred_verdict']} (truth {cap['gt_verdict']})" + ("" if right else " - WRONG, no warning")
        ok = right
    return {"label": label, "image": cap["image"], "boxes": boxes, "status": status, "ok": ok}


def find_overview(overviews: Path | None, scene: str, condition: str) -> Path | None:
    if overviews is None or not Path(overviews).is_dir():
        return None
    d = Path(overviews)
    for cand in (d / f"{scene}.png", d / scene / "overview.png", d / f"{scene}_overview.png", d / f"overview_{scene}.png",
                 d / f"overview_{condition}.png", d / f"{condition}.png"):
        if cand.exists():
            return cand
    hits = sorted(p for p in d.rglob("*.png") if scene in str(p) and "overview" in str(p).lower())
    return hits[0] if hits else None


def _verb(cap: dict[str, Any] | None) -> str:
    if cap is None:
        return "not run"
    if cap["pred_verdict"] == "RETAKE":
        return "refuses safely (retake)"
    return "passes" if cap["pred_verdict"] == cap["gt_verdict"] else "fails"


def cards_from_report(rep: dict[str, Any], plate: str | None = None, overviews: Path | None = None,
                      notes: str | None = None) -> list[dict]:
    """Title card, one card per matrix row of one plate (old vs new method), then the results over all
    plates (summary table + optional notes) and the pre-registered criterion with the verdict."""
    caps = rep["captures"]
    if plate is None:
        withdef = sorted({c["plate"] for c in caps if c.get("n_gt")})
        plate = withdef[0] if withdef else sorted({c["plate"] for c in caps})[0]
    mine = [c for c in caps if c["plate"] == plate]
    if not mine:
        raise SystemExit(f"plate {plate} is not in the report")
    conds = list(rep.get("conditions") or NAMES)
    keys = sorted({(c["condition"], c["device_mode"]) for c in mine},
                  key=lambda k: (conds.index(k[0]) if k[0] in conds else 99, k[1] != "open"))
    classes = _classes(rep)

    def pick(arm: str, key) -> dict[str, Any] | None:
        """First variant of this arm and row, from the arm's own model when several models ran on it
        (eval_lighting --pairs all): the naive panel must show the naive model on naive grids."""
        cs = [c for c in mine if c["arm"] == arm and (c["condition"], c["device_mode"]) == key]
        return min(cs, key=lambda c: (c.get("model", arm) != arm, c["k"] if c["k"] is not None else -1)) if cs else None

    cards = [{"title": "TriNetra: steel surface inspection across lighting",
              "caption": f"One test plate ({plate}), {len(keys)} lighting set-ups, two capture methods",
              "text": ("Old method: one frame with auto-exposure and the camera ISP. New method: a 5-frame LED "
                       "burst in raw (lit frames and LEDs-off frames), ambient subtraction, glare removal, and a "
                       "quality gate that asks for a retake instead of guessing. The product fits a clip-on hood "
                       "with an LED ring. Green boxes: truth. Red boxes: what the detector found. The plates are "
                       "digital-twin renders, never trained on. The last cards give the results over all test plates."),
              "seconds": 7}]
    for key in keys:
        cond, mode = key
        old, new = pick("naive", key), pick("burst", key)
        hood = " + hood" if mode == "hood" else ""
        lux = next((c.get("lux") for c in (new, old) if c and c.get("lux") is not None), None)
        title = f"{NAMES.get(cond, cond)}{hood}" + (f"  ({lux:,.0f} lux)" if lux else "")
        # both arms are captured through the same hood in a hood row: say so for the old method too
        caption = f"old method{hood} {_verb(old)} / new method{hood} {_verb(new)}"
        panels = []
        if old:
            panels.append(capture_panel(old, f"Old method (single frame, auto-exposure{hood})", classes))
        if new:
            panels.append(capture_panel(new, f"New method (burst{hood})", classes))
        gate = (new or {}).get("gate_message") if (new or {}).get("gate") != "PASS" else ""
        scene = (new or old)["scene"]
        cards.append({"title": title, "caption": caption, "overview": find_overview(overviews, scene, cond),
                      "panels": panels, "gate": gate or ""})
    cards += summary_cards(rep, notes)
    return cards


ARM_LABEL = {"burst": "New method (burst)", "naive": "Old method (one frame)"}


def _sum(rows: Sequence[dict[str, Any]], key: str, part: str = "k") -> int:
    return sum(int((r.get(key) or {}).get(part) or 0) for r in rows)


def summary_cards(rep: dict[str, Any], notes: str | None = None) -> list[dict[str, Any]]:
    """Results over ALL test plates (a table from the report rows, when it has them), then the verdict card
    with the pre-registered criterion. A long verdict_text is left out: the table carries the numbers."""
    n_plates = rep.get("n_plates") or len({c.get("plate") for c in rep.get("captures") or []})
    table_rows = []
    for res in sorted(rep.get("results") or [], key=lambda r: r["arm"] != "burst"):
        for mode in ("hood", "open"):
            rows = [r for r in res.get("rows") or [] if r.get("device_mode") == mode]
            if not rows:
                continue
            n = _sum(rows, "silent", "n")
            silent, unsafe, retake = _sum(rows, "silent"), _sum(rows, "silent_unsafe"), _sum(rows, "retake")
            f1 = [(r.get("f1") or {}).get("value") for r in rows]
            f1 = [v for v in f1 if isinstance(v, (int, float))]
            gated = any((r.get("retake") or {}).get("k") for r in res.get("rows") or [])
            table_rows.append([
                ARM_LABEL.get(res["arm"], res["arm"]),
                "hood (product)" if mode == "hood" else "open (no hood)",
                [f"{silent} / {n}", "good" if silent == 0 else "bad"],
                [str(unsafe), "good" if unsafe == 0 else "bad"],
                [f"{retake} / {n}", "dim"] if gated else ["no gate", "dim"],
                f"{min(f1):.2f}-{max(f1):.2f}" if f1 else "-",
            ])
    cards = []
    if table_rows:
        all_rows = [r for res in rep.get("results") or [] for r in res.get("rows") or []]
        lux = max((r.get("lux_median") or 0 for r in all_rows), default=0)
        n_cond = len({r.get("condition") for r in all_rows})
        cards.append({"title": f"Results over all {n_plates} test plates",
                      "caption": f"{len(rep.get('captures') or [])} captures, {n_cond} lighting set-ups from 0 to "
                                 f"{lux:,.0f} lux",
                      "table": {"cols": [30, 330, 560, 820, 960, 1130],
                                "header": ["Method", "Device", "Wrong, no warning", "Unsafe", "Retakes", "F1"],
                                "rows": table_rows},
                      "text": notes or "", "seconds": 12})
    verdicts = ", ".join(f"{ARM_LABEL.get(r['arm'], r['arm'])}: {r['verdict']}" for r in rep.get("results") or [])
    text = (rep.get("criterion") or {}).get("text", "")
    vt = rep.get("verdict_text") or ""
    if vt and len(vt) <= 300:
        text += "\n\n" + vt
    cards.append({"title": "Result: " + verdicts, "caption": "Pass criterion (stated before the test)",
                  "text": text, "seconds": 9})
    return cards


# --- Video -------------------------------------------------------------------------------

def build_video(cards: Sequence[dict[str, Any]], out: str | Path, seconds: float = 3.0, fps: int = 25,
                ffmpeg: str | None = None, last_seconds: float | None = None) -> Path:
    """Render the cards and pipe them to ffmpeg as raw frames (exact durations), H.264 yuv420p.

    A card may carry its own "seconds"; last_seconds (if given) holds the last card longer.
    """
    exe = find_ffmpeg(ffmpeg)
    if exe is None:
        raise SystemExit("ffmpeg not found (brew install ffmpeg)")
    if not cards:
        raise SystemExit("no cards to render")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i, card in enumerate(cards):
            dur = last_seconds if (i == len(cards) - 1 and last_seconds) else card.get("seconds", seconds)
            frame = np.ascontiguousarray(render_card(card)).tobytes()
            for _ in range(max(1, int(round(float(dur) * fps)))):
                proc.stdin.write(frame)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed: {err.strip()[-500:]}")
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--report", type=Path, help="reports/lighting_matrix.json")
    src.add_argument("--cards", type=Path, help="cards JSON (see the docstring)")
    ap.add_argument("--plate", help="plate id (default: the first plate with a defect)")
    ap.add_argument("--overviews", type=Path, help="directory of Blender overview renders")
    ap.add_argument("--out", type=Path, default=Path("reports/demo_lighting.mp4"))
    ap.add_argument("--seconds", type=float, default=3.5, help="seconds per card")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--ffmpeg", help="ffmpeg binary")
    ap.add_argument("--save-cards", type=Path, help="also write the card PNGs here")
    ap.add_argument("--notes", type=Path, help="plain-text notes shown under the results table (the interpretation)")
    a = ap.parse_args(argv)
    if a.report:
        notes = a.notes.read_text().strip() if a.notes else None
        cards = cards_from_report(json.loads(a.report.read_text()), a.plate, a.overviews, notes)
    else:
        cards = json.loads(a.cards.read_text())
    if a.save_cards:
        a.save_cards.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(cards):
            cv2.imwrite(str(a.save_cards / f"card_{i:03d}.png"), render_card(c))
    out = build_video(cards, a.out, a.seconds, a.fps, a.ffmpeg)
    total = sum(float(c.get("seconds", a.seconds)) for c in cards)
    print(f"wrote {out} ({len(cards)} cards, {total:.1f} s)")
    return out


if __name__ == "__main__":
    main()
