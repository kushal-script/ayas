"""Ayas demo video compositor: Blender clips + real pipeline assets + narration -> 1920x1080 30 fps MP4.

usage: python3 compose.py [out.mp4] [--scene sNN --still t]   (still: write one PNG for checking)
"""
import json, math, os, subprocess, sys, wave
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

V = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(V))   # repo root (this file is in tools/video/)
W, H, FPS = 1920, 1080, 30
PRE, POST = 0.2, 0.25
BG = (11, 15, 22, 255)
PANEL = (22, 28, 38, 235)
CYAN, AMBER, GREEN, RED, PURPLE, ORANGE = (63, 208, 255), (255, 186, 40), (61, 220, 132), (255, 92, 92), (176, 102, 255), (255, 140, 50)
WHITE, GREY = (238, 242, 248), (150, 160, 175)
SEV = {"critical": RED, "major": ORANGE, "minor": AMBER}
DISP = {"PASS": GREEN, "CLEAN": GREEN, "REWORK": ORANGE, "HOLD_FOR_QA": PURPLE, "REJECT": RED}

# ---------------------------------------------------------------- helpers
AV = "/System/Library/Fonts/Avenir Next.ttc"
KIND = {"heavy": 8, "bold": 0, "demi": 2, "med": 5, "reg": 7}


@lru_cache(None)
def font(size, kind="demi"):
    if kind == "mono":
        return ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", size)
    if kind == "deva":
        return ImageFont.truetype("/System/Library/Fonts/Kohinoor.ttc", size, index=2)   # Kohinoor Devanagari Semibold
    return ImageFont.truetype(AV, size, index=KIND[kind])


def wrap(text, f, maxw):
    out = []
    for para in text.split("\n"):
        line = ""
        for w in para.split(" "):
            t = (line + " " + w).strip()
            if f.getlength(t) <= maxw or not line:
                line = t
            else:
                out.append(line); line = w
        out.append(line)
    return out


@lru_cache(None)
def txt(text, size, kind="demi", color=WHITE, maxw=None, spacing=1.25, align="left"):
    f = font(size, kind)
    lines = wrap(text, f, maxw) if maxw else text.split("\n")
    asc, desc = f.getmetrics()
    lh = int(size * spacing)
    w = int(max(f.getlength(l) for l in lines)) + 4
    h = lh * (len(lines) - 1) + asc + desc + 4
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        x = 2 if align == "left" else (w - f.getlength(l)) / 2
        d.text((x, 2 + i * lh), l, font=f, fill=color + (255,) if len(color) == 3 else color)
    return im


@lru_cache(None)
def box(w, h, fill=PANEL, r=16, outline=None, width=2):
    im = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, int(w) - 1, int(h) - 1], r, fill=fill, outline=outline, width=width)
    return im


@lru_cache(None)
def img(path, w=None, h=None, crop=None, r=0):
    im = Image.open(path).convert("RGBA")
    if crop:
        im = im.crop(crop)
    if w and h:
        im = im.resize((int(w), int(h)), Image.LANCZOS)
    elif w:
        im = im.resize((int(w), int(round(im.height * w / im.width))), Image.LANCZOS)
    elif h:
        im = im.resize((int(round(im.width * h / im.height)), int(h)), Image.LANCZOS)
    if r:
        m = Image.new("L", im.size, 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, im.width - 1, im.height - 1], r, fill=255)
        a = np.minimum(np.asarray(im.getchannel("A")), np.asarray(m)); im.putalpha(Image.fromarray(a))
    return im


def put(cv, spr, x, y, a=1.0):
    if a <= 0.01:
        return
    x, y = int(round(x)), int(round(y))
    if a < 0.99:
        spr = spr.copy()
        spr.putalpha(Image.fromarray((np.asarray(spr.getchannel("A")) * a).astype(np.uint8)))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + spr.width), min(H, y + spr.height)
    if x1 <= x0 or y1 <= y0:
        return
    if (x0, y0, x1, y1) != (x, y, x + spr.width, y + spr.height):
        spr = spr.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    cv.alpha_composite(spr, (x0, y0))


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ramp(t, t0, d=0.45):
    return ease((t - t0) / d)


def canvas(color=BG):
    return Image.new("RGBA", (W, H), color)


@lru_cache(None)
def grad_left(w=1150, a0=225):
    g = np.zeros((H, w, 4), np.uint8)
    g[..., 3] = (a0 * (1 - np.linspace(0, 1, w)) ** 1.4).astype(np.uint8)[None, :]
    g[..., :3] = 8
    return Image.fromarray(g, "RGBA")


@lru_cache(None)
def grad_bottom(h=330, a0=200):
    g = np.zeros((h, W, 4), np.uint8)
    g[..., 3] = (a0 * np.linspace(0, 1, h) ** 1.3).astype(np.uint8)[:, None]
    g[..., :3] = 6
    return Image.fromarray(g, "RGBA")


CLIPN = {}


def clip(name, t):
    d = f"{V}/clips/{name}.mp4_frames"
    if name not in CLIPN:
        CLIPN[name] = len([f for f in os.listdir(d) if f.endswith(".jpg")])
    i = min(max(1, int(t * FPS) + 1), CLIPN[name])
    a = cv2.imread(f"{d}/f_{i:04d}.jpg")[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(a)).convert("RGBA")


def heading(cv, t, num, title, sub=None):
    a = ramp(t, 0.15)
    put(cv, txt(num, 26, "bold", CYAN), 110, 62, a)
    put(cv, txt(title, 46, "bold"), 160, 48, a)
    if sub:
        put(cv, txt(sub, 24, "med", GREY), 162, 108, ramp(t, 0.35))


def chip(text, color, size=24, kind="demi", pad=16, fill=None):
    s = txt(text, size, kind, color)
    b = box(s.width + 2 * pad, s.height + 14, fill or (color[0] // 6 + 10, color[1] // 6 + 12, color[2] // 6 + 16, 235), r=(s.height + 14) // 2,
            outline=color + (200,), width=2).copy()
    b.alpha_composite(s, (pad, 6))
    return b


def tag(cv, text, color, x, y, a=1.0, size=24):
    put(cv, chip(text, color, size), x, y, a)


# ---------------------------------------------------------------- narration / subtitles
NARR = json.load(open(f"{V}/narration.json"))
DISPLAY = [("Hailo ten H", "Hailo-10H"), ("AI HAT plus two", "AI HAT+ 2"), ("five inch", "5-inch"), ("USB C", "USB-C"),
           ("YOLO v8 nano", "YOLOv8n"), ("eight hundredths of a millimetre", "0.08 mm"), ("forty seven milliseconds", "47 ms"),
           ("physics informed", "physics-informed"), ("five frame", "five-frame"), ("low angle", "low-angle"),
           ("clip-on", "clip-on"), ("hold for QA", "hold for QA")]


def disp(t):
    for a, b in DISPLAY:
        t = t.replace(a, b)
    return t


def adur(key):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                          f"{V}/audio/{key}.aiff"]).decode())


def chunks(text, maxc=78):
    import re
    parts = re.split(r"(?<=[.,:;])\s+", text)
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > maxc:
            out.append(cur); cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        out.append(cur)
    res = []
    for c in out:                                  # split long ones at a word boundary
        while len(c) > maxc + 20:
            k = c.rfind(" ", 0, maxc); res.append(c[:k]); c = c[k + 1:]
        res.append(c)
    return res


def subtitle_track(text, dur):
    cs = chunks(disp(text))
    n = sum(len(c) for c in cs)
    t, out = PRE, []
    for c in cs:
        d = dur * len(c) / n
        out.append((t, t + d, c)); t += d
    return out


def draw_sub(cv, subs, t):
    for t0, t1, c in subs:
        if t0 <= t < t1:
            s = txt(c, 31, "med", WHITE)
            b = box(s.width + 44, s.height + 18, (0, 0, 0, 165), r=12)
            x = (W - b.width) // 2; y = H - 64 - b.height
            put(cv, b, x, y); put(cv, s, x + 22, y + 9)
            return


# ---------------------------------------------------------------- scenes
A = f"{V}/assets"
REC = json.load(open(f"{A}/record_dent.json"))
SURF = json.load(open(f"{V}/clips/surface.mp4.json")) if os.path.exists(f"{V}/clips/surface.mp4.json") else {"depth_mm": 0.32, "exaggeration": 6.0}
EXP = json.load(open(f"{V}/clips/explode.mp4.json"))


SAFFRON = (255, 128, 32)


@lru_cache(None)
def lockup(size=150, center=False):
    """Brand lockup: Ayas | अयस् over the tagline."""
    a = txt("Ayas", size, "heavy", WHITE)
    d = txt("अयस्", int(size * 0.78), "deva", SAFFRON)
    sub = txt("E V E R Y   S U R F A C E ,   S E E N   T R U E .", int(size * 0.17), "demi", (190, 198, 210))
    gap = int(size * 0.22)
    w = max(a.width + gap * 2 + 4 + d.width, sub.width)
    h = a.height + int(size * 0.18) + sub.height
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x0 = (w - (a.width + gap * 2 + 4 + d.width)) // 2 if center else 0
    im.alpha_composite(a, (x0, 0))
    ImageDraw.Draw(im).rectangle([x0 + a.width + gap, int(size * 0.28), x0 + a.width + gap + 3, int(size * 1.05)], fill=SAFFRON + (255,))
    im.alpha_composite(d, (x0 + a.width + 2 * gap + 4, int(size * 0.02)))
    im.alpha_composite(sub, ((w - sub.width) // 2 if center else 4, a.height + int(size * 0.12)))
    return im


def s01(t, D):
    fr = clip("title", t)
    cv = Image.new("RGBA", (W, H))
    cv.alpha_composite(fr.crop((0, 0, 400, H)).transpose(Image.FLIP_LEFT_RIGHT), (0, 0))
    cv.alpha_composite(fr.crop((0, 0, W - 400, H)), (400, 0))
    put(cv, grad_left(1000, 210), 0, 0)
    put(cv, lockup(150), 104, 250, ramp(t, 0.4, 0.7))
    put(cv, txt("On-device visual quality control\nfor steel and metal surfaces", 42, "demi", (205, 215, 230), spacing=1.3), 112, 555, ramp(t, 3.0, 0.6))
    x = 112
    for i, (c, col) in enumerate([("Raspberry Pi 5", WHITE), ("AI HAT+ 2 · Hailo-10H", CYAN), ("Llama 3.2 · ExecuTorch", AMBER), ("Fully offline", GREEN)]):
        ch = chip(c, col, 24)
        put(cv, ch, x, 700, ramp(t, 4.2 + 0.3 * i))
        x += ch.width + 14
    put(cv, txt("Bharat AI SoC Challenge · demo", 24, "med", GREY), 112, 780, ramp(t, 5.4))
    return cv


DEFECTS = [("Scratch", "lines from sliding parts"), ("Dent", "impact, trapped chips"), ("Pit", "corrosion attack"),
           ("Corrosion", "rust, oxidation"), ("Crack", "bending, fatigue"), ("Stain", "oil, water marks")]


def s02(t, D):
    cv = canvas()
    heading(cv, t, "01", "The problem", "Surface defects on steel and metal plate")
    for i, (n, d) in enumerate(DEFECTS):
        x, y = 110 + (i % 2) * 400, 200 + (i // 2) * 128
        a = ramp(t, 0.6 + 0.45 * i)
        b = box(380, 110, PANEL, r=14).copy()
        ImageDraw.Draw(b).rounded_rectangle([0, 0, 8, 109], 4, fill=[RED, ORANGE, RED, ORANGE, RED, AMBER][i] + (255,))
        b.alpha_composite(txt(n, 34, "bold"), (28, 12)); b.alpha_composite(txt(d, 22, "med", GREY), (30, 62))
        put(cv, b, x, y + 18 * (1 - a), a)
    a = ramp(t, 5.0)
    b = box(780, 150, (40, 22, 26, 235), r=14, outline=RED + (160,)).copy()
    b.alpha_composite(txt("Manual inspection", 32, "bold", (255, 170, 170)), (28, 16))
    b.alpha_composite(txt("slow, inconsistent between operators and shifts, and it leaves no record", 24, "med", WHITE, maxw=720), (30, 64))
    put(cv, b, 110, 600, a)
    for i, (f, cap, msg) in enumerate([("s900100_C6o_v0_B1.jpg", "Direct sun, 51,301 lux", "glare washes the plate out"),
                                       ("s900100_C3o_v0_B1.jpg", "Window daylight, 4,758 lux", "uneven light across the plate")]):
        x = 1010 + i * 420; a = ramp(t, 7.6 + 0.6 * i)
        put(cv, img(f"{A}/{f}", 400, 400, r=12), x, 220, a)
        put(cv, txt(cap, 26, "bold"), x, 636, a)
        put(cv, txt(msg, 22, "med", (255, 150, 150)), x, 674, a)
    put(cv, txt("Plain camera frames from the Blender digital twin", 20, "med", GREY), 1010, 720, ramp(t, 8.6))
    return cv


LABEL_T = {"Display": 2.0, "PiStack": 3.6, "Camera": 7.0, "Leds_L": 8.6, "InspectionHood": 11.0, "PowerBank": 12.0, "BackLid": 12.4, "WingFront_R": 12.8}


def anchors(fr):
    k = str(1 + 3 * ((max(1, fr) - 1) // 3))
    an = EXP["anchors"]
    while k not in an and int(k) > 1:
        k = str(int(k) - 3)
    return an[k]


def spread(items, lo=200, hi=900, gap=74):
    items = sorted(items, key=lambda r: r[1])
    ys = [min(max(y, lo), hi) for _, y in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - hi if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    return [(n, y) for (n, _), y in zip(items, ys)]


def s03(t, D):
    cv = clip("explode", t)
    put(cv, grad_bottom(), 0, H - 330)
    heading(cv, t, "02", "The device", "Handheld inspector, exploded view")
    an = anchors(int(t * FPS) + 1)
    vis = [n for n in EXP["labels"] if t >= LABEL_T[n] and n in an]
    left = [(n, an[n][1]) for n in vis if an[n][0] < 960]
    right = [(n, an[n][1]) for n in vis if an[n][0] >= 960]
    d = ImageDraw.Draw(cv, "RGBA")
    for side, items in (("L", left), ("R", right)):
        for n, y in spread(items):
            a = ramp(t, LABEL_T[n], 0.5)
            s = txt(EXP["labels"][n], 27, "demi")
            b = box(s.width + 36, s.height + 16, (12, 18, 28, 220), r=10, outline=CYAN + (170,)).copy(); b.alpha_composite(s, (18, 8))
            bx = 110 if side == "L" else W - 110 - b.width
            ax, ay = an[n]
            ex = bx + b.width if side == "L" else bx
            al = int(220 * a)
            d.line([(ex, y), (ax, ay)], fill=CYAN + (al,), width=3)
            d.ellipse([ax - 7, ay - 7, ax + 7, ay + 7], fill=CYAN + (al,), outline=(255, 255, 255, al), width=2)
            put(cv, b, bx, y - b.height // 2, a)
    return cv


BURST_DESC = {"B1": "both LEDs", "L": "left LED only", "A": "all off: room light", "R": "right LED only", "B2": "both again"}


def s04(t, D):
    cv = clip("burst", t)
    heading(cv, t, "03", "Five-frame light burst", "Every inspection, in under a second")
    js = json.load(open(f"{V}/clips/burst.mp4.json"))
    step, seq = js["step_frames"] / FPS, js["seq"]
    k = int(t / step) % 5
    p1 = 1 - ramp(t, 5.4, 0.5)
    x0 = 1340
    if p1 > 0:
        put(cv, box(500, 680, PANEL, r=18), x0, 190, p1)
        n = seq[k]
        put(cv, img(f"{A}/s900100_C3h_v0_{n}.jpg", 452, 452, r=10), x0 + 24, 214, p1)
        put(cv, txt(n, 40, "heavy", CYAN), x0 + 26, 680, p1)
        put(cv, txt(BURST_DESC[n], 28, "demi"), x0 + 26 + (90 if len(n) == 2 else 60), 688, p1)
        for i, m in enumerate(seq):
            tx = x0 + 24 + i * 92
            th = img(f"{A}/s900100_C3h_v0_{m}.jpg", 84, 84, r=6)
            put(cv, th, tx, 760, p1 * (1.0 if i == k else 0.45))
            if i == k:
                ImageDraw.Draw(cv, "RGBA").rounded_rectangle([tx - 3, 757, tx + 87, 847], 8, outline=CYAN + (int(255 * p1),), width=3)
    p2 = ramp(t, 5.6, 0.5)
    if p2 > 0:
        put(cv, box(500, 690, PANEL, r=18), x0, 190, p2)
        put(cv, txt("L − A,  R − A", 34, "bold", CYAN), x0 + 26, 210, p2)
        put(cv, txt("dark frame subtracted: room light gone", 22, "med", GREY), x0 + 26, 254, p2)
        put(cv, img(f"{A}/s900100_C3h_v0_M.jpg", 230, 230, r=8), x0 + 26, 300, p2)
        put(cv, txt("Compare left and right:\nglare removed", 24, "demi", WHITE), x0 + 274, 330, ramp(t, 7.0))
        put(cv, txt("Quality gate", 30, "bold"), x0 + 26, 560, ramp(t, 8.0))
        g1 = chip("PASS  · hood, window daylight", GREEN, 22)
        put(cv, g1, x0 + 26, 610, ramp(t, 8.4))
        g2 = chip("RETAKE  · open, direct sun", RED, 22)
        put(cv, g2, x0 + 26, 668, ramp(t, 9.0))
        put(cv, txt("“Too bright: shade the part or fit the hood, then retake.”", 21, "med", (255, 170, 170), maxw=450), x0 + 28, 724, ramp(t, 9.4))
        put(cv, txt("Real gate messages, run on digital-twin captures", 18, "med", GREY), x0 + 28, 828, ramp(t, 9.4))
    return cv


def s05(t, D):
    cv = clip("uv", t)
    heading(cv, t, "04", "UV burst for cracks")
    tag(cv, "ROADMAP · ILLUSTRATION", AMBER, W - 110 - chip("ROADMAP · ILLUSTRATION", AMBER).width, 60, ramp(t, 0.3))
    b = box(640, 120, PANEL, r=14).copy()
    b.alpha_composite(txt("UV LEDs + fluorescent penetrant", 30, "bold", (190, 160, 255)), (26, 16))
    b.alpha_composite(txt("the crack glows; white light alone can miss it", 22, "med", GREY), (28, 66))
    put(cv, b, 110, 760, ramp(t, 1.6))
    return cv


DET_CROP = (54, 55, 1026, 632)


def s06(t, D):
    cv = canvas()
    heading(cv, t, "05", "Detect on the AI HAT+ 2", "Hailo-10H, 40 TOPS")
    ov = img(f"{A}/det_overlay.jpg", 1120, 665, crop=DET_CROP, r=12)
    x, y = 110, 190
    put(cv, ov, x, y, ramp(t, 0.3))
    d = ImageDraw.Draw(cv, "RGBA")
    # 640 px tiles with overlap over the 1250 x 750 px plate grid: 3 x 2
    a = ramp(t, 1.8)
    if a > 0:
        for r in range(2):
            for c in range(3):
                tx = x + c * (1120 - 640 * 1120 / 1250) / 2; ty = y + r * (665 - 640 * 665 / 750)
                d.rectangle([tx, ty, tx + 640 * 1120 / 1250, ty + 640 * 665 / 750], outline=CYAN + (int(150 * a),), width=2)
    # scan line reveals the detections
    sx = (t - 4.0) / 2.0
    if sx < 1:
        lx0 = x + int(1120 * max(0.0, sx))
        d.rectangle([lx0, y, x + 1120, y + 665], fill=(8, 12, 18, 105))
        if 0 <= sx:
            lx = x + 1120 * sx
            d.rectangle([lx - 2, y, lx + 2, y + 665], fill=CYAN + (230,))
    rows = [("0.08 mm / px", "rectified measuring grid", 2.2), ("640 px tiles", "with overlap, full resolution", 3.2),
            ("YOLOv8n", "compiled to a Hailo HEF", 4.6), ("6 classes", "scratch · dent · pit · corrosion · crack · stain", 6.0)]
    for i, (big, small, t0) in enumerate(rows):
        a = ramp(t, t0)
        put(cv, txt(big, 38, "bold", CYAN if i != 2 else WHITE), 1300, 200 + i * 118, a)
        put(cv, txt(small, 22, "med", GREY, maxw=500), 1302, 248 + i * 118, a)
    a = ramp(t, 7.2)
    put(cv, txt("47 ms", 96, "heavy", GREEN), 1296, 690, a)
    put(cv, txt("per frame on the Hailo-10H", 24, "med", WHITE), 1302, 810, a)
    return cv


def s07(t, D):
    cv = canvas()
    heading(cv, t, "06", "Rules decide the verdict", "Every rule cites a standard (EN 10163-2 / ISO 7788)")
    p = 1 - ramp(t, 6.0, 0.5)
    if p > 0:
        cols = [("ID", 110), ("Type", 210), ("Size", 420), ("Severity", 780), ("Rule", 980), ("Action", 1500)]
        for n, cx in cols:
            put(cv, txt(n, 22, "bold", GREY), cx, 180, p)
        for i, dft in enumerate(REC["defects"]):
            a = p * ramp(t, 0.6 + 0.4 * i, 0.35)
            y = 222 + i * 62
            put(cv, box(1700, 54, (22, 28, 38, 200), r=10), 96, y - 6, a)
            sz = dft["size"]
            s = ", ".join(f"{k.split('_')[0]} {v:.1f} mm" for k, v in sz.items() if k.endswith("_mm") and k != "area_mm2")[:34]
            put(cv, txt(dft["id"], 26, "bold"), 110, y, a)
            put(cv, txt(dft["type"], 26, "demi"), 210, y, a)
            put(cv, txt(s, 24, "med", GREY), 420, y + 2, a)
            put(cv, chip(dft["severity"], SEV[dft["severity"]], 20), 780, y - 2, a)
            put(cv, txt(dft["rule_id"], 24, "mono", WHITE), 980, y + 2, a)
            put(cv, chip(dft["disposition"].replace("_", " "), DISP[dft["disposition"]], 20), 1500, y - 2, a)
        a = p * ramp(t, 3.4)
        b = box(1700, 128, (60, 30, 96, 245), r=16).copy()
        b.alpha_composite(txt("HOLD FOR QA", 54, "heavy"), (30, 26))
        b.alpha_composite(txt(REC["verdict_instruction"], 24, "med", WHITE, maxw=1020), (520, 28))
        put(cv, b, 96, 620, a)
        put(cv, txt("Verdicts: PASS · CLEAN · REWORK · HOLD FOR QA · REJECT", 24, "demi", GREY), 110, 770, a)
    a = ramp(t, 6.3, 0.6)
    if a > 0:
        put(cv, img(f"{REPO}/reports/figures/scenes/lighting_grid_s900100.jpg", 1700, r=10), 110, 190, a)
        put(cv, chip("Hood: same verdict from a dark room to direct sun", GREEN, 26), 110, 840, ramp(t, 7.0))
        put(cv, txt("Blender digital twin: one plate, 7 lightings × 2 device modes", 22, "med", GREY), 110, 900, ramp(t, 7.0))
    return cv


def s08(t, D):
    cv = canvas()
    heading(cv, t, "07", "3D surface map", "Photometric stereo with a physics-informed fit")
    for i, (f, lab) in enumerate([("dent_L.png", "Left light"), ("dent_R.png", "Right light")]):
        a = ramp(t, 0.8 + 0.5 * i)
        put(cv, img(f"{A}/{f}", 270, 270, r=10), 110, 190 + i * 330, a)
        put(cv, txt(lab, 26, "bold"), 112, 466 + i * 330, a)
    a = ramp(t, 3.2)
    b = box(280, 300, PANEL, r=14, outline=CYAN + (140,)).copy()
    b.alpha_composite(txt("Physics-informed fit", 26, "bold", CYAN, maxw=240), (20, 18))
    b.alpha_composite(txt("Light model × height map, fitted to both images and the plate's material", 21, "med", WHITE, maxw=240), (20, 100))
    put(cv, b, 420, 350, a)
    d = ImageDraw.Draw(cv, "RGBA")
    if a > 0:
        al = int(230 * a)
        for y0 in (325, 655):
            d.line([(390, y0), (418, 500 if y0 < 500 else 500)], fill=CYAN + (al,), width=3)
        d.line([(705, 500), (745, 500)], fill=CYAN + (al,), width=4); d.polygon([(745, 490), (760, 500), (745, 510)], fill=CYAN + (al,))
    a = ramp(t, 4.0, 0.6)
    if a > 0:
        fr = clip("surface", max(0.0, t - 3.6)).resize((1060, 596), Image.LANCZOS)
        put(cv, fr, 770, 210, a)
        d.rounded_rectangle([770, 210, 1830, 806], 12, outline=(60, 72, 90, int(255 * a)), width=2)
    a = ramp(t, 8.6)
    put(cv, txt(f"{SURF['depth_mm']:.2f} mm", 80, "heavy", WHITE), 790, 828, a)
    put(cv, txt("deepest point of dent D1", 24, "med", GREY), 1110, 846, a)
    put(cv, txt("Digital-twin truth 0.33 mm", 24, "demi", GREEN), 1110, 880, a)
    put(cv, txt(f"Relief exaggerated ×{SURF['exaggeration']:.0f}", 20, "med", GREY), 1600, 770, ramp(t, 5.0))
    # depth colour bar
    a = ramp(t, 5.0)
    if a > 0:
        bar = np.zeros((24, 300, 4), np.uint8)
        c_top, c_deep = np.array([0.42, 0.46, 0.52]), np.array([0.02, 0.18, 0.75])
        tt = np.linspace(0, 1, 300) ** 0.7
        col = (c_top[None] + (c_deep - c_top)[None] * tt[:, None]) ** (1 / 2.2) * 255
        bar[..., :3] = col[None].astype(np.uint8); bar[..., 3] = 255
        put(cv, Image.fromarray(bar, "RGBA"), 1500, 224, a)
        put(cv, txt("0", 18, "med", WHITE), 1500, 250, a); put(cv, txt(f"−{SURF['depth_mm']:.2f} mm", 18, "med", WHITE), 1720, 250, a)
    return cv


LLAMA_NOTE = ("The scratch most likely comes from parts sliding against each other or on a dirty bench. "
              "To prevent it next time, you should make sure to deburr the fixtures and guides before assembling the parts.")


def s09(t, D):
    cv = canvas()
    heading(cv, t, "08", "Llama explains, on the device", "Llama 3.2 1B (SpinQuant) running in ExecuTorch on the Pi 5")
    a = ramp(t, 0.3)
    put(cv, box(1032, 632, (40, 46, 58, 255), r=22), 94, 194, a)
    put(cv, img(f"{A}/ui_hold.jpg", 1000, 600, r=10), 110, 210, a)
    put(cv, txt("Station UI on the 5-inch screen (real capture from the Pi)", 22, "med", GREY), 112, 842, a)
    a = ramp(t, 1.0)
    b = box(660, 400, PANEL, r=16, outline=AMBER + (150,)).copy()
    b.alpha_composite(txt("LIKELY CAUSE AND PREVENTION", 22, "bold", AMBER), (26, 20))
    put(cv, b, 1160, 210, a)
    n = int(len(LLAMA_NOTE) * min(1, max(0, (t - 1.4) / 4.2)))
    if n:
        put(cv, txt(LLAMA_NOTE[:n], 26, "med", WHITE, maxw=600, spacing=1.35), 1186, 262)
    for i, (c, col, t0) in enumerate([("Answers “Why?” and “How to fix?”", CYAN, 5.6), ("Never changes the verdict", GREEN, 6.6), ("No network needed", WHITE, 7.4)]):
        put(cv, chip(c, col, 24), 1160, 640 + i * 66, ramp(t, t0))
    return cv


def s10(t, D):
    cv = canvas()
    stats = [("1.7 s", "camera to verdict on the Pi 5"), ("47 ms", "YOLOv8n on the Hailo-10H"),
             ("0.02 mm", "mean dent-depth error (twin)"), ("Offline", "no cloud, no network")]
    p = 1 - ramp(t, 2.6, 0.5)
    for i, (big, small) in enumerate(stats):
        a = p * ramp(t, 0.2 + 0.25 * i)
        x = 110 + i * 430
        b = box(400, 230, PANEL, r=18).copy()
        b.alpha_composite(txt(big, 66, "heavy", [CYAN, GREEN, AMBER, WHITE][i]), (28, 30))
        b.alpha_composite(txt(small, 23, "med", GREY, maxw=350), (30, 140))
        put(cv, b, x, 400 + 20 * (1 - a), a)
    a = ramp(t, 2.9, 0.6)
    lk = lockup(150, True)
    put(cv, lk, (W - lk.width) // 2, 300, a)
    s = txt("Built with Llama", 44, "demi", AMBER)
    put(cv, s, (W - s.width) // 2, 640, ramp(t, 3.3))
    s = txt("Raspberry Pi 5 · AI HAT+ 2 (Hailo-10H) · Raspberry Pi AI Camera", 26, "med", GREY)
    put(cv, s, (W - s.width) // 2, 720, ramp(t, 3.6))
    return cv


# --- architecture / network / rules / PINN / Llama scenes (spliced into compose.py before SCENES) ---


def arrow(d, p0, p1, color, a=1.0, width=3, head=12):
    if a <= 0.01:
        return
    al = int(255 * a)
    d.line([p0, p1], fill=color + (al,), width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    l, r = ang + 2.6, ang - 2.6
    d.polygon([p1, (p1[0] + head * math.cos(l), p1[1] + head * math.sin(l)), (p1[0] + head * math.cos(r), p1[1] + head * math.sin(r))],
              fill=color + (al,))


def pulse(d, pts, f, color, a=1.0, r=7):
    """A glowing dot at fraction f along a polyline."""
    if a <= 0.01:
        return
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    s = f * sum(seg)
    for i, L in enumerate(seg):
        if s <= L or i == len(seg) - 1:
            u = 0 if L == 0 else min(1, s / L)
            x = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * u; y = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * u
            for rr, aa in ((r * 2.4, 50), (r * 1.6, 90), (r, 255)):
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=color + (int(aa * a),))
            return
        s -= L


def node(title, sub, tagtxt, color, w=250, h=112):
    b = box(w, h, (20, 27, 38, 240), r=14, outline=color + (190,)).copy()
    s = txt(title, 25, "bold", WHITE, maxw=w - 24); b.alpha_composite(s, ((w - s.width) // 2, 10))
    s = txt(sub, 18, "med", GREY, maxw=w - 24, align="center"); b.alpha_composite(s, ((w - s.width) // 2, 44))
    s = txt(tagtxt, 17, "bold", color); b.alpha_composite(s, ((w - s.width) // 2, h - 28))
    return b


FLOW = [  # title, sub, hardware, colour, t_on
    ("1  Scan", "5-frame LED burst", "AI CAMERA", WHITE, 0.4),
    ("2  Clean + gate", "room light, glare removed", "PI 5 CPU", CYAN, 1.4),
    ("3  Rectify + tile", "0.08 mm grid · 640 px", "PI 5 CPU", CYAN, 2.4),
    ("4  Detect", "YOLOv8n", "HAILO-10H NPU", GREEN, 3.4),
    ("5  Measure + rules", "severity › verdict", "PI 5 CPU", CYAN, 4.8),
    ("6  3D depth", "physics-informed fit", "PI 5 CPU · 3 workers", CYAN, 6.3),
    ("7  Explain", "Llama 3.2 1B", "PI 5 CPU · EXECUTORCH", AMBER, 7.6),
    ("8  Output", "screen · alert · record", "5-INCH SCREEN", PURPLE, 8.9),
]
FW, FH, FIMG = 380, 300, 200


def flow_card(i):
    title, sub, hw, col, _ = FLOW[i]
    b = box(FW, FH, (20, 27, 38, 240), r=14, outline=col + (170,)).copy()
    b.alpha_composite(txt(title, 24, "bold"), (16, 10))
    b.alpha_composite(txt(sub, 18, "med", GREY), (16, 44))
    s = txt(hw, 16, "bold", col); b.alpha_composite(s, (FW - s.width - 14, 14))
    ix, iy = 16, 80
    if i == 0:
        for j, n in enumerate(["B1", "L", "A", "R", "B2"]):
            b.alpha_composite(img(f"{A}/s900100_C3h_v0_{n}.jpg", 120, 120, r=6), (ix + j * 58, iy + (j % 2) * 40 + 20))
    elif i == 1:
        b.alpha_composite(img(f"{A}/s900100_C3h_v0_M.jpg", 200, 200, r=8), (ix, iy))
        b.alpha_composite(chip("PASS", GREEN, 20), (ix + 220, iy + 20))
        b.alpha_composite(txt("quality\ngate", 18, "med", GREY), (ix + 222, iy + 70))
    elif i == 2:
        t_ = img(f"{V}/plate_tex.png", 340, 200, r=8).copy(); dd = ImageDraw.Draw(t_, "RGBA")
        for c in range(4):
            dd.line([(c * 113, 0), (c * 113, 200)], fill=CYAN + (200,), width=2)
        dd.line([(0, 100), (340, 100)], fill=CYAN + (200,), width=2)
        b.alpha_composite(t_, (ix, iy))
    elif i == 3:
        b.alpha_composite(img(f"{A}/det_overlay.jpg", 340, 200, crop=DET_CROP, r=8), (ix, iy))
    elif i == 4:
        for j, dft in enumerate(REC["defects"][:4]):
            y = iy + j * 50
            b.alpha_composite(txt(f"{dft['id']} {dft['type']}", 19, "demi"), (ix, y + 6))
            b.alpha_composite(chip(dft["disposition"].replace("_", " "), DISP[dft["disposition"]], 16), (ix + 180, y))
    elif i == 5:
        fr = clip("surface", 9.0).crop((360, 120, 1560, 960)).resize((240, 168), Image.LANCZOS)
        b.alpha_composite(fr, (ix, iy)); b.alpha_composite(txt("0.32 mm", 20, "heavy"), (ix + 254, iy + 4))
    elif i == 6:
        b.alpha_composite(txt(LLAMA_NOTE, 17, "med", WHITE, maxw=340, spacing=1.3), (ix, iy))
    elif i == 7:
        b.alpha_composite(img(f"{A}/ui_hold.jpg", 340, 204, r=8), (ix, iy))
    return b


@lru_cache(None)
def flow_card_c(i):
    return flow_card(i)


def s05b(t, D):
    cv = canvas()
    heading(cv, t, "05", "End to end: scan › processing › output", "One trigger press, about two seconds, all on the handheld")
    d = ImageDraw.Draw(cv, "RGBA")
    xs = [110, 540, 970, 1400]
    pos = [(xs[i], 175) for i in range(4)] + [(xs[3 - i], 525) for i in range(4)]
    for i in range(8):
        t_on = FLOW[i][4]
        a = ramp(t, t_on)
        x, y = pos[i]
        if i:
            px, py = pos[i - 1]
            if py == y:
                p0 = (px + FW + 4, y + FH // 2) if x > px else (px - 4, y + FH // 2)
                p1 = (x - 6, y + FH // 2) if x > px else (x + FW + 6, y + FH // 2)
            else:
                p0, p1 = (px + FW // 2, py + FH + 4), (x + FW // 2, y - 6)
            arrow(d, p0, p1, (120, 140, 165), a, 3, 10)
            if 0 < t - t_on < 0.9:
                pulse(d, [p0, p1], (t - t_on) / 0.9, FLOW[i][3], 1)
        put(cv, flow_card_c(i), x, y + 12 * (1 - a), a)
        if t_on <= t < (FLOW[i + 1][4] if i < 7 else 99):
            d.rounded_rectangle([x - 4, y - 4, x + FW + 4, y + FH + 4], 16, outline=FLOW[i][3] + (int(255 * a),), width=4)
    put(cv, chip("≈1.7 s camera to verdict on the Pi 5 · no network", GREEN, 24), 110, 850, ramp(t, 9.6))
    return cv


QA_BTNS = [("Inspect", (46, 110, 60)), ("Why?", (40, 52, 72)), ("How to fix?", (40, 52, 72)), ("Shift summary", (40, 52, 72)), ("3D", (40, 52, 72)), ("Acknowledge", (110, 90, 30))]


@lru_cache(None)
def ui_btn(label, fill, hot=False):
    s = txt(label, 24, "bold")
    b = box(s.width + 48, 60, fill + (255,), r=10, outline=(255, 255, 255, 230) if hot else None, width=3).copy()
    b.alpha_composite(s, (24, 14))
    return b


def s09b(t, D):
    cv = canvas()
    heading(cv, t, "10", "Questions and every station function", "Each question goes to the engine that can answer it safely")
    d = ImageDraw.Draw(cv, "RGBA")
    press = {"Why?": 0.9, "How to fix?": 4.2, "Shift summary": 7.0, "3D": 9.8, "Acknowledge": 10.6, "Inspect": 11.4}
    x = 110
    bx = {}
    for lab, fill in QA_BTNS:
        hot = press[lab] <= t < press[lab] + 0.6
        b = ui_btn(lab, fill, hot)
        put(cv, b, x, 168 + (3 if hot else 0), ramp(t, 0.2))
        bx[lab] = (x + b.width // 2, 228); x += b.width + 14
    lanes = [
        ("Why?  ·  open questions", "Llama 3.2 1B", "Guardrail", "checks every number and defect type; retries once, else a template", AMBER, 0.9, 290),
        ("How to fix?  ·  ship it? override?", "Rules engine", "Exact SOP steps", "“1. Do not paint over it. 2. QA checks the depth with a gauge …”", CYAN, 4.2, 470),
        ("Shift summary", "Last 50 records", "Llama summary", "counts by verdict and defect, a heat map of where they occur; guardrail-checked", AMBER, 7.0, 650),
    ]
    for q, eng, step, outtxt, col, t0, y in lanes:
        a = ramp(t, t0)
        if a <= 0:
            continue
        qb = box(380, 120, (20, 27, 38, 240), r=12, outline=WHITE + (120,)).copy()
        qb.alpha_composite(txt("QUESTION", 16, "bold", GREY), (16, 12)); qb.alpha_composite(txt(q, 22, "bold", WHITE, maxw=350), (16, 40))
        put(cv, qb, 110, y, a)
        eb = box(300, 120, (20, 27, 38, 240), r=12, outline=col + (200,)).copy()
        eb.alpha_composite(txt("ANSWERED BY", 16, "bold", GREY), (16, 12)); eb.alpha_composite(txt(eng, 26, "bold", col), (16, 44))
        put(cv, eb, 560, y, ramp(t, t0 + 0.5))
        ob = box(900, 120, (20, 27, 38, 240), r=12).copy()
        ob.alpha_composite(txt(step.upper(), 16, "bold", GREY), (16, 12)); ob.alpha_composite(txt(outtxt, 21, "med", WHITE, maxw=860), (16, 42))
        put(cv, ob, 930, y, ramp(t, t0 + 1.0))
        arrow(d, (494, y + 60), (554, y + 60), (120, 140, 165), ramp(t, t0 + 0.5), 3, 10)
        arrow(d, (864, y + 60), (924, y + 60), (120, 140, 165), ramp(t, t0 + 1.0), 3, 10)
        if 0 < t - t0 < 1.4:
            pulse(d, [(110 + 190, y), (300, y + 60), (930, y + 60)], (t - t0) / 1.4, col)
    for i, (c, col) in enumerate([("3D surface on demand", CYAN), ("Acknowledge clears the alert", ORANGE), ("Inspect: next part, RETAKE if refused", GREEN),
                                  ("JSON record with rule references", WHITE)]):
        put(cv, chip(c, col, 21), 110 + [0, 330, 760, 1260][i], 830, ramp(t, 9.8 + 0.5 * i))
    return cv


# YOLOv8n diagram: (label, kind, spatial, channels)
YOLO = [("640²", "in", 640, 3), ("320²", "b", 320, 16), ("160²", "b", 160, 32), ("80²", "b", 80, 64), ("40²", "b", 40, 128), ("20²", "b", 20, 256)]


def slab(d, x, y, w, h, dep, col, a, glow=0.0):
    """A feature map drawn as a 3D slab: front face w x h, depth dep."""
    al = int(255 * a)
    c = tuple(int(v * (0.55 + 0.45 * glow)) for v in col)
    top = [(x, y), (x + dep, y - dep * 0.6), (x + dep + w, y - dep * 0.6), (x + w, y)]
    side = [(x + w, y), (x + w + dep, y - dep * 0.6), (x + w + dep, y + h - dep * 0.6), (x + w, y + h)]
    d.polygon(top, fill=tuple(min(255, int(v * 1.25)) for v in c) + (al,))
    d.polygon(side, fill=tuple(int(v * 0.7) for v in c) + (al,))
    d.rectangle([x, y, x + w, y + h], fill=c + (al,), outline=(255, 255, 255, int(al * (0.25 + 0.6 * glow))), width=2)


def s06(t, D):
    cv = canvas()
    heading(cv, t, "06", "Detection: YOLOv8n on the Hailo-10H")
    pA = 0
    if pA > 0:
        ov = img(f"{A}/det_overlay.jpg", 1120, 665, crop=DET_CROP, r=12)
        x, y = 110, 190
        put(cv, ov, x, y, pA)
        d = ImageDraw.Draw(cv, "RGBA")
        a = ramp(t, 1.0) * pA
        for r in range(2):
            for c in range(3):
                tx = x + c * (1120 - 640 * 1120 / 1250) / 2; ty = y + r * (665 - 640 * 665 / 750)
                d.rectangle([tx, ty, tx + 640 * 1120 / 1250, ty + 640 * 665 / 750], outline=CYAN + (int(150 * a),), width=2)
        sx = (t - 2.2) / 1.8
        if sx < 1:
            lx0 = x + int(1120 * max(0.0, sx))
            d.rectangle([lx0, y, x + 1120, y + 665], fill=(8, 12, 18, int(105 * pA)))
            if sx >= 0:
                d.rectangle([lx0 - 2, y, lx0 + 2, y + 665], fill=CYAN + (int(230 * pA),))
        for i, (big, small, t0) in enumerate([("0.08 mm / px", "rectified measuring grid", 0.8), ("640 px tiles", "3 × 2, overlapping", 1.6)]):
            put(cv, txt(big, 40, "bold", CYAN), 1300, 230 + i * 130, ramp(t, t0) * pA)
            put(cv, txt(small, 23, "med", GREY), 1302, 282 + i * 130, ramp(t, t0) * pA)
    pB = ramp(t, 0.2, 0.5)
    if pB <= 0:
        return cv
    d = ImageDraw.Draw(cv, "RGBA")
    # activation sweep through the stages
    stages = 12
    k = ((t - 0.6) / 0.16) % (stages + 4)
    gl = lambda i: max(0.0, 1 - abs(k - i) / 1.5)
    cy = 470
    # input tile
    tile = img(f"{V}/plate_tex.png", 150, 150)
    put(cv, tile, 110, cy - 75, pB)
    put(cv, txt("tile 640×640", 18, "med", GREY), 120, cy + 86, pB)
    # backbone
    put(cv, txt("BACKBONE  ·  CSP (C2f) + SPPF", 20, "bold", CYAN), 330, 200, pB)
    xs = 320
    bb = []
    for i, (lab, kind, sp, ch) in enumerate(YOLO[1:]):
        h = 230 - i * 34; dep = 10 + i * 9; w = 26 + i * 6
        slab(d, xs, cy - h / 2, w, h, dep, (40, 150, 200), pB, gl(i + 1))
        put(cv, txt(lab, 17, "med", GREY), xs - 2, cy + h / 2 + 8, pB)
        put(cv, txt(f"{ch} ch", 15, "med", (110, 120, 135)), xs - 2, cy + h / 2 + 30, pB)
        bb.append((xs + w + dep, sp))
        if i:
            arrow(d, (xs - 30, cy), (xs - 6, cy), (120, 140, 165), pB, 2, 8)
        xs += w + dep + 60
    arrow(d, (268, cy), (314, cy), (120, 140, 165), pB, 2, 8)
    put(cv, txt("SPPF", 17, "bold", WHITE), xs - 70, cy - 100, pB)
    # neck: FPN (top-down) + PAN (bottom-up) at 80, 40, 20
    put(cv, txt("NECK  ·  FPN + PAN", 20, "bold", (190, 160, 255)), 1000, 200, pB)
    ny = {80: 300, 40: 470, 20: 640}
    nx = 1030
    for j, sp in enumerate((80, 40, 20)):
        slab(d, nx, ny[sp] - 55, 30, 110, 22, (130, 90, 210), pB, gl(7 + j))
        put(cv, txt(f"{sp}²", 17, "med", GREY), nx + 60, ny[sp] - 10, pB)
    for sp in (80, 40, 20):                                   # backbone taps
        src = [b for b in bb if b[1] == sp][0]
        d.line([(src[0] + 4, cy - 20), (src[0] + 40, cy - 20), (src[0] + 40, ny[sp]), (nx - 6, ny[sp])], fill=(130, 90, 210, int(140 * pB)), width=2)
    d.line([(nx + 15, ny[20] - 60), (nx + 15, ny[80] + 60)], fill=(190, 160, 255, int(160 * pB)), width=2)
    # heads
    put(cv, txt("HEADS  ·  anchor-free", 20, "bold", GREEN), 1260, 200, pB)
    for j, sp in enumerate((80, 40, 20)):
        hb = box(220, 78, (20, 40, 30, 240), r=10, outline=GREEN + (int(120 + 135 * gl(10 + j)),)).copy()
        hb.alpha_composite(txt(f"P{3 + j} · {sp}×{sp}", 21, "bold"), (14, 8))
        hb.alpha_composite(txt("box (64) + 6 classes", 17, "med", GREY), (14, 42))
        put(cv, hb, 1250, ny[sp] - 39, pB)
        arrow(d, (nx + 60, ny[sp] + 25), (1244, ny[sp]), (120, 140, 165), pB, 2, 8)
        arrow(d, (1472, ny[sp]), (1540, cy), (120, 140, 165), pB, 2, 8)
    nb = box(120, 70, PANEL, r=10, outline=WHITE + (160,)).copy()
    nb.alpha_composite(txt("NMS", 22, "bold"), (34, 8)); nb.alpha_composite(txt("Pi CPU", 16, "med", GREY), (32, 40))
    put(cv, nb, 1546, cy - 35, pB)
    put(cv, img(f"{A}/det_overlay.jpg", 150, crop=DET_CROP, r=6), 1690, cy - 45, pB * ramp(t, 1.8))
    arrow(d, (1668, cy), (1686, cy), (120, 140, 165), pB, 2, 8)
    pulse(d, [(268, cy), (1030, cy), (1250, cy), (1700, cy)], min(1, max(0, (k - 0) / (stages + 1))), CYAN, pB * (1 if k < stages + 1 else 0))
    for i, (c, col) in enumerate([("≈3 M parameters", WHITE), ("8-bit, compiled to a Hailo HEF", GREEN), ("47 ms per frame", GREEN), ("NMS on the Pi CPU", CYAN)]):
        put(cv, chip(c, col, 21), 110 + [0, 250, 650, 920][i], 820, ramp(t, 2.6 + 0.4 * i))
    return cv


RANK = ["PASS", "CLEAN", "REWORK", "HOLD_FOR_QA", "REJECT"]


def meas_str(dft):
    s = dft["size"]
    if dft["type"] == "scratch":
        return f"L {s['length_mm']:.1f} · W {s['width_mm']:.2f} mm"
    if dft["type"] == "corrosion":
        return f"area {s['area_mm2']:.0f} mm²"
    return f"Ø {s['diameter_mm']:.2f} mm"


def s07(t, D):
    cv = canvas()
    heading(cv, t, "07", "Decision making: the rules engine", "Deterministic, auditable, cites EN 10163-2 / ISO 7788")
    d = ImageDraw.Draw(cv, "RGBA")
    d1 = REC["defects"][0]
    chain = [(f"D1 dent  ·  {meas_str(d1)}", WHITE, 0.3), ("Rule DNT-MAJ:  Ø ≥ 4 mm", CYAN, 0.8), ("major", ORANGE, 1.3), ("HOLD FOR QA", PURPLE, 1.8)]
    x = 110
    for i, (c, col, t0) in enumerate(chain):
        ch = chip(c, col, 26)
        a = ramp(t, t0)
        if i:
            arrow(d, (x - 46, 222), (x - 10, 222), (120, 140, 165), a)
        put(cv, ch, x, 222 - ch.height // 2, a)
        x += ch.width + 58
    put(cv, txt("one defect, one rule, one disposition", 20, "med", GREY), 112, 262, ramp(t, 2.0))
    # all six defects -> worst wins
    vx, vy = 1330, 470
    worst = max(REC["defects"], key=lambda q: RANK.index(q["disposition"]))
    for i, dft in enumerate(REC["defects"]):
        y = 350 + i * 76
        a = ramp(t, 2.3 + 0.22 * i, 0.35)
        rb = box(470, 60, (22, 28, 38, 225), r=10).copy()
        rb.alpha_composite(txt(f"{dft['id']}  {dft['type']}", 24, "bold"), (16, 14))
        rb.alpha_composite(txt(meas_str(dft), 20, "med", GREY), (220, 18))
        put(cv, rb, 110, y, a)
        put(cv, chip(dft["severity"], SEV[dft["severity"]], 19), 600, y + 12, a)
        dc = chip(dft["disposition"].replace("_", " "), DISP[dft["disposition"]], 19)
        put(cv, dc, 760, y + 12, a)
        la = ramp(t, 3.9 + 0.1 * i)
        if la > 0:
            hot = dft is worst or dft["disposition"] == worst["disposition"]
            col = DISP[dft["disposition"]] if hot else (90, 100, 115)
            pts = [(770 + dc.width, y + 30), (1150, y + 30), (vx - 8, vy + 100)]
            d.line(pts[:2], fill=col + (int(200 * la),), width=3 if hot else 2)
            arrow(d, pts[1], pts[2], col, la * (1 if hot else 0.6), 3 if hot else 2, 10)
            if hot:
                pulse(d, pts, ((t - 4.0) / 1.1) % 1, col, la)
    a = ramp(t, 4.8, 0.5)
    vb = box(480, 200, (60, 30, 96, 245), r=18, outline=PURPLE + (255,), width=3).copy()
    vb.alpha_composite(txt("VERDICT", 22, "bold", (220, 200, 255)), (30, 22))
    vb.alpha_composite(txt("HOLD FOR QA", 56, "heavy"), (28, 56))
    vb.alpha_composite(txt("worst disposition wins", 22, "med", (220, 200, 255)), (30, 140))
    put(cv, vb, vx, vy, a)
    a = ramp(t, 5.6)
    xr = 1330
    for i, r in enumerate(RANK):
        c = chip(r.replace("_", " "), DISP[r], 17)
        put(cv, c, xr, 700, a)
        xr += c.width + 6
    put(cv, txt("least  ›  most severe", 18, "med", GREY), 1332, 744, a)
    return cv


# ---- PINN
HMAP = np.array(json.load(open(f"{A}/surface_dent.json"))["items"][0]["height_mm"], np.float32)
HMAP -= np.median(np.concatenate([HMAP[0], HMAP[-1], HMAP[:, 0], HMAP[:, -1]]))


def depth_rgb(h, hmin):
    c_top, c_deep = np.array([0.42, 0.46, 0.52]), np.array([0.02, 0.18, 0.75])
    q = np.clip(h / hmin, 0, 1) ** 0.7
    col = (c_top[None, None] + (c_deep - c_top)[None, None] * q[..., None]) ** (1 / 2.2)
    return (col * 255).astype(np.uint8)


@lru_cache(None)
def pinn_imgs(pq):
    p = pq / 40.0
    h = HMAP * p
    hm = depth_rgb(h, HMAP.min())
    s = 0.24
    gy, gx = np.gradient(h * 5.0, s)                      # slopes, exaggerated for display
    n = np.stack([-gx, -gy, np.ones_like(h)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    outs = []
    for lx in (-0.75, 0.75):                                # left / right LED
        l = np.array([lx, 0.0, 0.66]); l /= np.linalg.norm(l)
        sh = np.clip(n @ l, 0, 1) ** 1.4
        g = (40 + 200 * sh / max(1e-6, (np.array([0, 0, 1.0]) @ l) ** 1.4 * 1.15)).clip(0, 255).astype(np.uint8)
        outs.append(Image.fromarray(np.stack([g, g, g], -1)).convert("RGBA").resize((140, 140), Image.BICUBIC))
    return Image.fromarray(hm).convert("RGBA").resize((170, 170), Image.BICUBIC), outs[0], outs[1]


def mlp_draw(d, x, y, w, h, a, k):
    cols = [6, 8, 8, 1]
    pts = []
    for ci, n in enumerate(cols):
        cx = x + ci * w / (len(cols) - 1)
        pts.append([(cx, y + h / 2 + (j - (n - 1) / 2) * (h / 8.5)) for j in range(n)])
    for ci in range(len(cols) - 1):
        for p in pts[ci]:
            for q in pts[ci + 1]:
                d.line([p, q], fill=(110, 130, 160, int(70 * a)), width=1)
    for ci, col in enumerate(pts):
        g = max(0.0, 1 - abs(k - ci) / 1.2)
        for (px, py) in col:
            r = 8
            d.ellipse([px - r, py - r, px + r, py + r], fill=(int(40 + 180 * g), int(150 + 80 * g), 255, int(255 * a)))


def s08(t, D):
    cv = canvas()
    heading(cv, t, "08", "3D surface: physics-informed neural network", "Fit per defect on the Pi CPU, no training data needed")
    d = ImageDraw.Draw(cv, "RGBA")
    for i, (f, lab) in enumerate([("dent_L.png", "Camera: left LED"), ("dent_R.png", "Camera: right LED")]):
        a = ramp(t, 0.4 + 0.3 * i)
        put(cv, img(f"{A}/{f}", 230, 230, r=10), 110, 190 + i * 300, a)
        put(cv, txt(lab, 22, "bold"), 112, 426 + i * 300, a)
    pA = 1 - ramp(t, 5.9, 0.6)
    it = int(110 * min(1, max(0, (t - 1.6) / 4.0)))
    if pA > 0:
        G = (120, 140, 165)
        a1, a2, a3, a4, a5 = (ramp(t, x_) * pA for x_ in (0.5, 0.8, 1.1, 1.4, 1.7))
        put(cv, node("x, y", "crop coords", "", WHITE, 120, 100), 400, 220, a1)
        put(cv, node("Fourier", "sin, cos 2πBx", "", WHITE, 150, 100), 560, 220, a1)
        arrow(d, (522, 270), (554, 270), G, a1, 2, 8)
        put(cv, txt("MLP", 22, "bold", CYAN), 800, 186, a2)
        mlp_draw(d, 760, 205, 200, 130, a2, ((t - 2.4) / 0.22) % 5)
        arrow(d, (712, 270), (746, 270), G, a2, 2, 8)
        put(cv, node("× (1−x²)(1−y²)", "h = 0 at the edge", "", WHITE, 210, 100), 1000, 220, a3)
        arrow(d, (970, 270), (996, 270), G, a3, 2, 8)
        hm, rl, rr = pinn_imgs(int(40 * ease(min(1, max(0, (t - 1.6) / 4.0)))))
        put(cv, hm, 1270, 185, a3)
        put(cv, txt("height map h(u, v)", 20, "bold"), 1270, 362, a3)
        arrow(d, (1214, 270), (1264, 270), G, a3, 2, 8)
        lb = node("Light model", "normals · LED positions · GGX gloss · material", "", WHITE, 300, 110)
        put(cv, lb, 1500, 470, a4)
        arrow(d, (1440, 300), (1600, 466), G, a4, 2, 8)
        put(cv, rl, 1120, 470, a4); put(cv, rr, 1275, 470, a4)
        put(cv, txt("rendered left", 19, "med", GREY), 1122, 616, a4); put(cv, txt("rendered right", 19, "med", GREY), 1277, 616, a4)
        arrow(d, (1496, 525), (1420, 525), G, a4, 2, 8)
        lo = node("Loss", "Cauchy(render − camera) + smoothness", "", AMBER, 300, 110)
        put(cv, lo, 720, 470, a5)
        arrow(d, (1114, 525), (1026, 525), G, a5, 2, 8)
        arrow(d, (346, 525), (714, 525), G, a5, 2, 8)
        # back-propagation
        bp = [(870, 466), (870, 400), (860, 345)]
        d.line(bp[:2], fill=AMBER + (int(230 * a5),), width=3); arrow(d, bp[1], bp[2], AMBER, a5, 3, 10)
        put(cv, txt("Adam: gradients update the MLP", 19, "bold", AMBER), 884, 410, a5)
        if a5 > 0.5:
            pulse(d, [(1650, 470), (1420, 525), (1026, 525), (870, 466), (870, 345), (860, 270), (1270, 270), (1600, 466)],
                  ((t - 2.6) / 1.6) % 1, AMBER, a5)
        # loss curve + iteration counter
        a6 = ramp(t, 1.8) * pA
        if a6 > 0:
            x0, y0, w_, h_ = 720, 660, 420, 150
            d.rectangle([x0, y0, x0 + w_, y0 + h_], outline=(60, 72, 90, int(255 * a6)), width=2)
            n = max(2, it)
            xs_ = np.linspace(0, 1, 111)[:n]
            ys_ = 0.08 + 0.9 * np.exp(-xs_ * 5.5)
            d.line([(x0 + 6 + v * (w_ - 12), y0 + h_ - 6 - yv * (h_ - 12)) for v, yv in zip(xs_, ys_)], fill=AMBER + (int(255 * a6),), width=3)
            put(cv, txt("loss", 18, "med", GREY), x0 + 8, y0 + 6, a6)
            put(cv, txt(f"iteration {it:3d} / 110", 24, "mono", WHITE), 1170, 690, a6)
            put(cv, txt("~4 s per defect, 3 defects in parallel", 20, "med", GREY), 1170, 734, a6)
    pB = ramp(t, 5.8, 0.6)
    if pB > 0:
        fr = clip("surface", max(0.0, t - 3.0)).resize((1180, 664), Image.LANCZOS)
        put(cv, fr, 420, 170, pB)
        d.rounded_rectangle([420, 170, 1600, 834], 12, outline=(60, 72, 90, int(255 * pB)), width=2)
        a = ramp(t, 7.0)
        put(cv, txt(f"{SURF['depth_mm']:.2f} mm", 58, "heavy"), 1626, 300, a)
        put(cv, txt("deepest point,\ndent D1", 24, "med", GREY), 1634, 400, a)
        put(cv, txt("Twin truth\n0.33 mm", 26, "bold", GREEN), 1634, 490, a)
        put(cv, txt(f"Relief ×{SURF['exaggeration']:.0f}", 20, "med", GREY), 1634, 600, a)
        bar = np.zeros((20, 200, 4), np.uint8)
        bar[..., :3] = depth_rgb(-np.linspace(0, 1, 200)[None].repeat(20, 0), -1.0); bar[..., 3] = 255
        put(cv, Image.fromarray(bar, "RGBA"), 1634, 650, a)
        put(cv, txt("0", 17, "med"), 1634, 674, a); put(cv, txt(f"−{SURF['depth_mm']:.2f}", 17, "med"), 1790, 674, a)
    return cv


# ---- Llama
PROMPT = "D1 scratch · critical · width 0.6 mm\nVerdict HOLD FOR QA (rules engine)\nTask: likely cause and prevention"


def s09(t, D):
    cv = canvas()
    heading(cv, t, "09", "Llama 3.2 1B on the Pi: how it explains", "Decision support only, the rules keep the verdict")
    d = ImageDraw.Draw(cv, "RGBA")
    a = ramp(t, 0.3)
    pb = box(480, 190, PANEL, r=14, outline=CYAN + (150,)).copy()
    pb.alpha_composite(txt("FROM THE RULES ENGINE", 18, "bold", CYAN), (20, 14))
    pb.alpha_composite(txt(PROMPT, 18, "mono", WHITE, spacing=1.45), (20, 48))
    put(cv, pb, 110, 190, a)
    # decoder stack
    x0, w, top, nl, lh = 700, 420, 262, 16, 26
    a2 = ramp(t, 0.9)
    tk = max(0.0, t - 1.6)
    sweep = (tk / 0.9) % 1.3
    for i in range(nl):
        y = top + (nl - 1 - i) * lh
        g = max(0.0, 1 - abs(sweep * nl - i) / 2.0) if tk > 0 else 0
        c = (int(60 + 160 * g), int(70 + 110 * g), int(110 + 40 * g))
        d.rounded_rectangle([x0, y, x0 + w, y + lh - 6], 5, fill=c + (int(255 * a2),))
        d.rounded_rectangle([x0 + 8, y + 4, x0 + 190, y + lh - 10], 3, fill=(255, 186, 40, int(110 * a2 * (0.4 + g))))
        d.rounded_rectangle([x0 + 200, y + 4, x0 + w - 8, y + lh - 10], 3, fill=(63, 208, 255, int(90 * a2 * (0.4 + g))))
    put(cv, txt("16 decoder layers", 20, "bold"), x0, top - 34, a2)
    put(cv, txt("attention (GQA)      SwiGLU MLP", 17, "med", GREY), x0 + 14, top + nl * lh + 2, a2)
    eb = box(w, 40, (30, 36, 48, 240), r=8).copy(); eb.alpha_composite(txt("token embeddings · 128K vocabulary", 18, "med"), (16, 9))
    put(cv, eb, x0, top + nl * lh + 30, a2)
    hb = box(w, 40, (30, 36, 48, 240), r=8).copy(); hb.alpha_composite(txt("LM head › next token", 18, "med"), (16, 9))
    put(cv, hb, x0, top - 80, a2)
    arrow(d, (596, 285), (694, top + nl * lh + 48), (120, 140, 165), a2, 2, 9)
    for j in range(4):                                         # tokens flowing in
        f = ((t - 1.0) / 1.6 + j / 4) % 1
        pulse(d, [(596, 285), (694, top + nl * lh + 48)], f, CYAN, a2 * ramp(t, 1.2), 5)
    # output stream
    a3 = ramp(t, 1.6)
    ob = box(640, 330, PANEL, r=16, outline=AMBER + (160,)).copy()
    ob.alpha_composite(txt("LIKELY CAUSE AND PREVENTION", 20, "bold", AMBER), (24, 18))
    put(cv, ob, 1170, 190, a3)
    arrow(d, (x0 + w + 6, top - 60), (1164, 260), (120, 140, 165), a3, 2, 9)
    words = LLAMA_NOTE.split(" ")
    nw = int(len(words) * min(1, max(0, (t - 1.4) / 5.0)))
    if nw:
        put(cv, txt(" ".join(words[:nw]), 25, "med", WHITE, maxw=590, spacing=1.38), 1194, 236)
    for i, (c, col, t0) in enumerate([("SpinQuant: 4-bit weights, 8-bit activations", AMBER, 2.2), ("ExecuTorch · XNNPACK on the Pi 5 CPU", CYAN, 3.0),
                                      ("Measured on the Pi: first token 4.7 s, 11.3 tokens/s", WHITE, 3.8)]):
        put(cv, chip(c, col, 21), 1170, 548 + i * 58, ramp(t, t0))
    put(cv, chip("Never changes the verdict", GREEN, 24), 1170, 730, ramp(t, 5.0))
    a4 = 0.0
    put(cv, img(f"{A}/ui_hold.jpg", 430, 258, r=8), 110, 420, a4)
    put(cv, txt("Real Pi screen: Why? and How to fix? ask Llama", 18, "med", GREY), 112, 688, a4)
    return cv


SCENES = [("s01_title", s01), ("s02_problem", s02), ("s03_product", s03), ("s04_burst", s04), ("s05_uv", s05), ("s05b_flow", s05b),
          ("s06_detect", s06), ("s07_rules", s07), ("s08_3d", s08), ("s09_llama", s09), ("s09b_qa", s09b), ("s10_close", s10)]


def frame(fn, t, D, subs, last):
    cv = fn(t, D)
    draw_sub(cv, subs, t)
    put(cv, txt("Ayas", 22, "bold", (200, 210, 225)), W - 150, H - 46, 0.6)
    f = min(ramp(t, 0.0, 0.25), 1.0 if not last else 1 - ramp(t, D - 0.9, 0.8))
    if t > D - 0.2 and not last:
        f = min(f, 1 - ramp(t, D - 0.2, 0.2))
    rgb = np.asarray(cv.convert("RGB"), np.float32)
    if f < 0.999:
        rgb = rgb * f
    return rgb.astype(np.uint8)


def build_audio(durs, path):
    sr = 48000
    total = sum(durs)
    mix = np.zeros(int(total * sr) + sr, np.float32)
    t0 = 0.0
    for (key, _), d in zip(SCENES, durs):
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", f"{V}/audio/{key}.aiff", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"])
        a = np.frombuffer(raw, np.float32)
        i = int((t0 + PRE) * sr); mix[i:i + len(a)] += a
        t0 += d
    # soft ambient pad under the voice
    n = len(mix); tt = np.arange(n) / sr
    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 246.94, 293.66), (164.81, 207.65, 246.94)]
    pad = np.zeros(n, np.float32)
    seg = 8.0
    for k in range(int(total / seg) + 2):
        c = chords[k % 4]; s0 = k * seg
        idx = (tt >= s0 - 1.5) & (tt < s0 + seg + 1.5)
        x = tt[idx] - s0
        env = np.clip((x + 1.5) / 3.0, 0, 1) * np.clip((seg + 1.5 - x) / 3.0, 0, 1)
        pad[idx] += env * sum(np.sin(2 * np.pi * f * tt[idx]) + 0.3 * np.sin(4 * np.pi * f * tt[idx]) for f in c) / 4
    pad *= 0.035
    fade = np.clip(tt / 2.0, 0, 1) * np.clip((total - tt) / 2.5, 0, 1)
    mix = mix * 0.92 + pad * fade
    mix = mix[:int(total * sr)]
    mix = np.clip(mix, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((mix * 32000).astype(np.int16).tobytes())


def main():
    args = sys.argv[1:]
    durs = [round(adur(k) + PRE + POST, 3) for k, _ in SCENES]
    durs[-1] += 1.6
    text = dict(NARR)
    if "--scene" in args:
        k = args[args.index("--scene") + 1]; ts = float(args[args.index("--still") + 1])
        i = [s[0].split('_')[0] for s in SCENES].index(k)
        key, fn = SCENES[i]
        Image.fromarray(frame(fn, ts, durs[i], subtitle_track(text[key], adur(key)), i == len(SCENES) - 1)).save(f"{V}/stills/comp_{k}_{ts}.png")
        print("saved", f"{V}/stills/comp_{k}_{ts}.png"); return
    out = args[0] if args else f"{V}/trinetra_demo.mp4"
    wav = f"{V}/mix.wav"; build_audio(durs, wav)
    print("scenes", [f"{k}:{d:.2f}" for (k, _), d in zip(SCENES, durs)], "total", round(sum(durs), 2), flush=True)
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-profile:v", "high",
                          "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out], stdin=subprocess.PIPE)
    for i, ((key, fn), D) in enumerate(zip(SCENES, durs)):
        subs = subtitle_track(text[key], adur(key))
        nf = int(round(D * FPS))
        for j in range(nf):
            p.stdin.write(frame(fn, j / FPS, D, subs, i == len(SCENES) - 1).tobytes())
        print("done", key, nf, "frames", flush=True)
    p.stdin.close(); p.wait()
    print("wrote", out, p.returncode)


if __name__ == "__main__":
    main()
