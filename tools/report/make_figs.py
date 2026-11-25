"""Charts and diagrams for the Ayas project report (numbers from reports/ and docs/ in the repo).

The PCB renders it tiles are made first with, per board:
    kicad-cli pcb render --side top --width 900 --height 900 --quality high -o docs/report/figures/pcb_<b>.png \
        hardware/pcb/kicad/<b>/ayas_<b>.kicad_pcb
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'docs/report/figures')
INK, MUTED, ORANGE, BLUE, GREEN, PURPLE, GREY = '#1d1f24', '#5c6068', '#e8761e', '#2f6db5', '#2e8b57', '#7b4bc4', '#b8bcc4'
plt.rcParams.update({'font.family': ['Helvetica', 'DejaVu Sans'], 'font.size': 10, 'axes.edgecolor': '#9aa0a8', 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def bar_labels(ax, bars, fmt='{:g}', dy=0.0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha='center', va='bottom', fontsize=8.5, color=INK)


# 1. lighting experiment
fig, ax = plt.subplots(figsize=(7.2, 3.0))
labels = ['Burst + gate\nhood (product)', 'Burst + gate\nopen', 'Single frame\nhood', 'Single frame\nopen']
silent = [2, 12, 14, 16]
unsafe = [0, 2, 0, 0]
x = range(4)
b1 = ax.bar([i - 0.18 for i in x], silent, 0.36, color=[ORANGE, GREY, GREY, GREY], label='Wrong verdict, no warning (of 210)')
b2 = ax.bar([i + 0.18 for i in x], unsafe, 0.36, color='#c0392b', label='Unsafe (milder than truth)')
bar_labels(ax, b1, dy=0.2); bar_labels(ax, b2, dy=0.2)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel('captures'); ax.set_ylim(0, 19)
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
ax.set_title('840 twin captures: 15 unseen plates × 7 lighting conditions × 2 devices × 2 methods', fontsize=9.5, color=MUTED)
save(fig, 'c_lighting.png')

# 2. per-class recall (burst, hood, pooled over 7 conditions)
fig, ax = plt.subplots(figsize=(7.2, 2.6))
cls = ['scratch', 'dent', 'crack', 'corrosion', 'stain', 'pit']
rec = [1.00, 1.00, 0.98, 0.83, 0.80, 0.79]
n = ['126/126', '70/70', '82/84', '70/84', '56/70', '77/98']
bars = ax.bar(cls, rec, color=[BLUE] * 3 + ['#6a8fc4'] * 3)
for b, t in zip(bars, n):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, '%.2f\n%s' % (b.get_height(), t), ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, 1.25); ax.set_ylabel('recall')
ax.set_title('YOLOv8n recall per class, burst capture with the hood, pooled over 7 lighting conditions', fontsize=9.5, color=MUTED)
save(fig, 'c_recall.png')

# 3. Pi 5 latency, camera to verdict (p50, ms)
fig, ax = plt.subplots(figsize=(7.2, 2.4))
stages = [('capture (twin file load)', [561, 547], '#9aa0a8'), ('burst processing + gate', [1135, 1110], BLUE),
          ('detect (YOLOv8n)', [1337, 47], ORANGE), ('rectify, tile, merge, measure, rules, report', [35, 35], GREEN)]
rows = ['Cortex-A76 CPU\n(ONNX Runtime)', 'Hailo-10H NPU\n(INT8 HEF)']
left = [0, 0]
for name, v, c in stages:
    ax.barh(rows, v, left=left, color=c, label=name, height=0.55)
    left = [l + a for l, a in zip(left, v)]
for i, tot in enumerate([2856, 1665]):
    ax.text(left[i] + 40, i, '%d ms p50' % tot, va='center', fontsize=9, color=INK, fontweight='bold')
ax.set_xlim(0, 3500); ax.set_xlabel('ms per inspection (p50), Raspberry Pi 5')
ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.32))
ax.invert_yaxis()
save(fig, 'c_latency.png')

# 4. Llama optimisation, before / after
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3))
data = [('Prompt tokens', 1340, 611, '{:.0f}'), ('Time to first token (s)', 15.8, 5.4, '{:.1f}'),
        ('Decode (tokens/s)', 6.7, 11.0, '{:.1f}'), ('Time per answer (s)', 54, 10.3, '{:.1f}')]
for ax, (t, a, b, f) in zip(axs, data):
    bars = ax.bar(['first\ndesign', 'final'], [a, b], color=[GREY, PURPLE], width=0.6)
    for bb, v in zip(bars, [a, b]):
        ax.text(bb.get_x() + bb.get_width() / 2, v * 1.02, f.format(v), ha='center', va='bottom', fontsize=8.5)
    ax.set_title(t, fontsize=8.8); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_ylim(0, max(a, b) * 1.25)
fig.suptitle('Llama 3.2 1B (SpinQuant INT4) via ExecuTorch on the Pi 5 Cortex-A76', fontsize=9.5, color=MUTED, y=1.04)
save(fig, 'c_llama.png')

# 5. PINN depth
fig, ax = plt.subplots(figsize=(7.2, 2.6))
cfg = ['L/R halves (today)', '3 sectors', '6 sectors']
pinn = [0.053, 0.034, 0.021]
cls_ = [0.153, 0.089, 0.071]
x = range(3)
b1 = ax.bar([i - 0.18 for i in x], pinn, 0.36, color=ORANGE, label='PINN (physics-informed height field)')
b2 = ax.bar([i + 0.18 for i in x], cls_, 0.36, color=GREY, label='Classical slopes + Poisson integration')
bar_labels(ax, b1, '{:.3f}', 0.002); bar_labels(ax, b2, '{:.3f}', 0.002)
ax.set_xticks(list(x)); ax.set_xticklabels(cfg); ax.set_ylabel('dent depth MAE (mm)'); ax.set_ylim(0, 0.19)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title('96 twin dents and pits, 6 materials, sensor noise: PINN is 2.1–3.4× better on dents', fontsize=9.5, color=MUTED)
save(fig, 'c_pinn.png')

# 6. optimisation speed-ups
fig, ax = plt.subplots(figsize=(7.2, 2.5))
items = [('Detector: CPU ONNX → Hailo-10H INT8', 1329, 47, 'ms'), ('3D surface of a plate (5 defects)', 55, 15, 's'),
         ('Llama advice per inspection', 54, 10.3, 's'), ('Llama time to first token', 15.8, 5.4, 's')]
y = range(len(items))
ratios = [a / b for _, a, b, _ in items]
bars = ax.barh([t for t, *_ in items], ratios, color=[ORANGE, GREEN, PURPLE, PURPLE], height=0.55)
for b, (t, a, bb, u) in zip(bars, items):
    ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, '%.1f×   (%g %s → %g %s)' % (a / bb, a, u, bb, u), va='center', fontsize=8.5)
ax.set_xlim(0, 40); ax.set_xlabel('speed-up (×)'); ax.invert_yaxis()
save(fig, 'c_speedups.png')


# 7. block diagrams
def box(ax, x, y, w, h, text, fc='#ffffff', ec=INK, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal', wrap=True)


def arrow(ax, x0, y0, x1, y1, c=MUTED, txt=None):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.1))
    if txt:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.07, txt, ha='center', fontsize=7, color=MUTED)


fig, ax = plt.subplots(figsize=(7.6, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis('off')
ax.text(0.1, 5.35, 'Pipeline A: vision (deterministic, < 2 s)', fontsize=9, color=BLUE, fontweight='bold')
A = [('5-frame raw burst\nB1 · L · A · R · B2', '#e8f0fb'), ('Ambient subtraction\nglare removal\nquality gate', '#e8f0fb'),
     ('Rectify 0.08 mm/px\n640 px tiles', '#e8f0fb'), ('YOLOv8n\nHailo-10H (INT8)\nor CPU ONNX', '#fdeedd'),
     ('Measure in mm\nrules engine\nverdict', '#e8f0fb')]
for i, (t, c) in enumerate(A):
    box(ax, 0.1 + i * 1.98, 3.85, 1.75, 1.15, t, fc=c)
    if i:
        arrow(ax, 0.1 + i * 1.98 - 0.23, 4.42, 0.1 + i * 1.98, 4.42)
box(ax, 2.1, 2.15, 2.6, 0.95, 'RETAKE with a reason\n("too bright: fit the hood")', fc='#fbe9e7')
arrow(ax, 2.95, 3.85, 3.3, 3.1)
box(ax, 5.3, 2.15, 4.55, 0.95, 'JSON record  ayas.inspection/3.0\nSQLite log  ·  overlay image  ·  station UI', fc='#eef6ee', bold=True)
arrow(ax, 8.95, 3.85, 8.4, 3.1)
ax.text(3.0, 1.62, 'Pipeline B: Llama (separate process, only the JSON crosses)', fontsize=9, color=PURPLE, fontweight='bold')
box(ax, 0.1, 0.15, 3.0, 1.2, 'Llama 3.2 1B Instruct\nSpinQuant INT4 · ExecuTorch\nXNNPACK on Cortex-A76', fc='#f1ebfa')
box(ax, 3.45, 0.15, 3.0, 1.2, 'Guardrail: verdict, defect types,\nnumbers, depth, banned phrases\n→ 1 rewrite → labelled template', fc='#f1ebfa')
box(ax, 6.8, 0.15, 3.05, 1.2, 'Cause & prevention note\n"Why?", open questions, Hindi\n(never changes the verdict)', fc='#f1ebfa')
arrow(ax, 5.3, 2.4, 1.6, 1.35, c=PURPLE); arrow(ax, 3.1, 0.75, 3.45, 0.75, c=PURPLE); arrow(ax, 6.45, 0.75, 6.8, 0.75, c=PURPLE)
save(fig, 'd_system.png')

fig, ax = plt.subplots(figsize=(7.6, 4.9)); ax.set_xlim(0, 10); ax.set_ylim(-0.5, 6.2); ax.axis('off')
box(ax, 0.1, 4.6, 1.9, 0.9, 'Power bank\nUSB-C PD 33 W', fc='#eceef2')
box(ax, 0.1, 3.25, 1.9, 0.9, 'CHARGE board\nUSB-C in (side wall)', fc='#eceef2')
box(ax, 2.6, 3.55, 2.4, 2.1, 'POWER  46 × 55, 2-layer\nSTUSB4500 PD sink → 12 V\nTPS565201 5 V / 5 A buck\nINA226 LED-rail monitor\nCC router, power-hold,\ncharge-through', fc='#fdeedd', fs=7.6)
box(ax, 5.6, 4.7, 1.8, 0.95, 'Pi 5 + AI HAT+ 2\n(Hailo-10H)', fc='#e8f0fb')
box(ax, 8.0, 4.7, 1.9, 0.95, '5" DSI display\nAI Camera', fc='#e8f0fb')
box(ax, 5.6, 2.0, 2.5, 2.2, 'LIGHT/IO  46 × 55, 6-layer\nRP2040 (USB to the Pi)\n10 constant-current sinks\nUV interlock, buttons,\nfan, buzzer, NTCs', fc='#fdeedd', fs=7.6)
box(ax, 8.5, 3.2, 1.4, 0.8, 'Fan 30 mm', fc='#eceef2')
box(ax, 2.6, 2.1, 2.4, 0.9, 'Power button\nUSB-A export', fc='#eceef2')
for i, (t, c) in enumerate([('Pod L: UV hub +\nWHITE-L carrier', '#f6f0e6'), ('Pod R: UV hub +\nWHITE-R carrier', '#f6f0e6'),
                            ('Hood: dock + pad\n6 ring modules', '#f6f0e6'), ('SIDE-L / SIDE-R\nbutton boards', '#f6f0e6')]):
    box(ax, 0.1 + i * 2.5, 0.15, 2.2, 0.95, t, fc=c)
arrow(ax, 2.0, 5.05, 2.6, 5.05, txt='USB-C'); arrow(ax, 2.0, 3.7, 2.6, 3.9, txt='CHG')
arrow(ax, 5.0, 5.15, 5.6, 5.15, txt='5 V'); arrow(ax, 7.4, 5.15, 8.0, 5.15)
arrow(ax, 5.0, 3.9, 5.6, 3.7); ax.text(5.3, 4.2, 'LINK', ha='center', fontsize=7, color=MUTED); arrow(ax, 6.5, 4.2, 6.5, 4.7, txt='USB')
arrow(ax, 8.1, 3.6, 8.5, 3.6); arrow(ax, 3.8, 3.55, 3.8, 3.0)
for i in range(4):
    arrow(ax, 6.85, 2.0, 1.2 + i * 2.5, 1.1, c=ORANGE)
ax.text(5.0, -0.35, 'Orange: keyed JST-SH cables (6, 7, 9, 4, 5 way), pod flex into ZIF docks, hood on spring pins', fontsize=7.4, color=MUTED, ha='center')
save(fig, 'd_electronics.png')

# 8. PCB montage
names = [('pcb_power', 'POWER (2-layer, 46 × 55)'), ('pcb_light', 'LIGHT/IO top (6-layer)'), ('pcb_light_b', 'LIGHT/IO bottom'),
         ('pcb_uv', 'UV pod hub (43 × 30.5)'), ('pcb_white_l', 'WHITE-L carrier (50 × 66)'), ('pcb_btn_l', 'SIDE-L buttons (36 × 20)'),
         ('pcb_charge', 'CHARGE (24 × 12.5)'), ('pcb_pod_dock_l', 'POD DOCK (28 × 14)'), ('pcb_hood_dock', 'HOOD DOCK (34 × 13)')]
tiles = []
for n, lab in names:
    im = Image.open(os.path.join(OUT, n + '.png')).convert('RGB')
    bg = im.getpixel((2, 2))
    diff = Image.eval(im.convert('L'), lambda v: 255 if abs(v - sum(bg) // 3) > 10 else 0)
    bb = diff.getbbox()
    if bb:
        im = im.crop((max(0, bb[0] - 20), max(0, bb[1] - 20), min(im.width, bb[2] + 20), min(im.height, bb[3] + 20)))
    im.thumbnail((560, 430))
    t = Image.new('RGB', (600, 500), 'white')
    t.paste(im, ((600 - im.width) // 2, 10 + (430 - im.height) // 2))
    d = ImageDraw.Draw(t)
    try:
        f = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 24)
    except Exception:
        f = ImageFont.load_default()
    w = d.textlength(lab, font=f)
    d.text(((600 - w) / 2, 455), lab, fill=INK, font=f)
    tiles.append(t)
sheet = Image.new('RGB', (1800, 1500), 'white')
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 3) * 600, (i // 3) * 500))
sheet.save(os.path.join(OUT, 'pcb_montage.jpg'), quality=90)
print('figures done')
