"""Build the Ayas project report (DOCX); LibreOffice converts it to PDF.

    python tools/report/make_figs.py        # charts, diagrams and the PCB montage -> docs/report/figures/
    python tools/report/build_report.py     # -> docs/report/Ayas_Project_Report.docx
    soffice --headless --convert-to pdf --outdir docs/report docs/report/Ayas_Project_Report.docx

Needs python-docx, matplotlib and Pillow. Numbers come from reports/ and docs/; the v_*.jpg figures are frames of
the project demo video and the pcb_*.png renders come from `kicad-cli pcb render` (both kept in docs/report/figures/).
"""
import os
import re
import sys

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(REPO, 'docs/report/figures')
RENDERS = os.path.join(REPO, 'docs/hardware/renders')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'docs/report/Ayas_Project_Report.docx')

INK = RGBColor(0x1D, 0x1F, 0x24)
MUTED = RGBColor(0x5C, 0x60, 0x68)
ORANGE = RGBColor(0xE8, 0x76, 0x1E)
BODY_FONT, HEAD_FONT, DEVA_FONT = 'Helvetica Neue', 'Helvetica Neue', 'Kohinoor Devanagari'

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
TEXT_W = 17.0

st = doc.styles['Normal']
st.font.name, st.font.size, st.font.color.rgb = BODY_FONT, Pt(10), INK
st.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)
st.paragraph_format.space_after = Pt(5)
st.paragraph_format.line_spacing = 1.15
for lvl, size, col in ((1, 17, INK), (2, 13, INK), (3, 11, MUTED)):
    h = doc.styles['Heading %d' % lvl]
    h.font.name, h.font.size, h.font.bold, h.font.color.rgb = HEAD_FONT, Pt(size), True, col
    h.element.rPr.rFonts.set(qn('w:eastAsia'), HEAD_FONT)
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    h.paragraph_format.space_after = Pt(6 if lvl == 1 else 4)
    h.paragraph_format.keep_with_next = True
for name in ('List Bullet', 'List Bullet 2', 'List Number'):
    s = doc.styles[name]
    s.font.name, s.font.size = BODY_FONT, Pt(10)
    s.paragraph_format.space_after = Pt(2)

FIGN = [0]
TABN = [0]


def runs(par, text, size=None, color=None):
    """Add text with **bold** and *italic* markup."""
    for part in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)', text):
        if not part:
            continue
        if part.startswith('**'):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith('*'):
            r = par.add_run(part[1:-1]); r.italic = True
        else:
            r = par.add_run(part)
        if size:
            r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color
    return par


def P(text, size=None, color=None, align=None, after=None):
    par = doc.add_paragraph()
    runs(par, text, size, color)
    if align == 'c':
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if after is not None:
        par.paragraph_format.space_after = Pt(after)
    return par


def H(text, lvl=1):
    return doc.add_heading(text, level=lvl)


def B(items, lvl=1):
    for it in items:
        par = doc.add_paragraph(style='List Bullet' if lvl == 1 else 'List Bullet 2')
        runs(par, it)


def N(items):
    for k, it in enumerate(items, 1):
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Cm(0.63)
        par.paragraph_format.first_line_indent = Cm(-0.63)
        par.paragraph_format.space_after = Pt(2)
        runs(par, '%d.	' % k + it)


def shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement('w:shd')
    sh.set(qn('w:val'), 'clear'); sh.set(qn('w:color'), 'auto'); sh.set(qn('w:fill'), hex_)
    tcPr.append(sh)


def T(rows, widths=None, caption=None, size=8.8, head=True):
    if caption:
        TABN[0] += 1
        cp = P('**Table %d.** %s' % (TABN[0], caption), size=8.8, color=MUTED, after=3)
        cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = ''
            par = c.paragraphs[0]
            par.paragraph_format.space_after = Pt(1)
            par.paragraph_format.line_spacing = 1.05
            runs(par, str(val), size=size, color=RGBColor(0xFF, 0xFF, 0xFF) if (head and i == 0) else None)
            if head and i == 0:
                shade(c, '2B2E35')
                for r in par.runs:
                    r.bold = True
            elif i % 2 == 0:
                shade(c, 'F3F4F6')
    if widths:
        t.autofit = False
        tw = sum(widths)
        widths = [w * TEXT_W / tw for w in widths]
        for j, w in enumerate(widths):
            for i in range(len(rows)):
                t.cell(i, j).width = Cm(w)
        for gc, w in zip(t._tbl.tblGrid.findall(qn('w:gridCol')), widths):
            gc.set(qn('w:w'), str(int(w * 567)))
        lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); t._tbl.tblPr.append(lay)
    # borders light grey
    tbl = t._tbl
    borders = OxmlElement('w:tblBorders')
    for e in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement('w:' + e)
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '4'); el.set(qn('w:color'), 'C8CBD0')
        borders.append(el)
    tbl.tblPr.append(borders)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def F(path, caption, width=TEXT_W):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.keep_with_next = True
    par.add_run().add_picture(path, width=Cm(width))
    FIGN[0] += 1
    cp = P('**Figure %d.** %s' % (FIGN[0], caption), size=8.8, color=MUTED, align='c', after=10)
    return cp


def F2(p1, p2, caption, w=8.3):
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, p in zip(t.rows[0].cells, (p1, p2)):
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraphs[0].add_run().add_picture(p, width=Cm(w))
    FIGN[0] += 1
    P('**Figure %d.** %s' % (FIGN[0], caption), size=8.8, color=MUTED, align='c', after=10)


def callout(text, fill='FDF1E6'):
    t = doc.add_table(rows=1, cols=1)
    c = t.cell(0, 0)
    shade(c, fill)
    c.paragraphs[0].paragraph_format.space_after = Pt(2)
    runs(c.paragraphs[0], text, size=9.6)
    c.width = Cm(TEXT_W)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def field(par, instr):
    r = par.add_run()
    f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = instr
    f2 = OxmlElement('w:fldChar'); f2.set(qn('w:fldCharType'), 'end')
    r._r.append(f1); r._r.append(it); r._r.append(f2)
    return r


# ------------------------------------------------------------------ footer
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run('Ayas · Project Report · Arm Bharat AI-SoC Challenge 2026-27 · page ')
r.font.size, r.font.color.rgb = Pt(8), MUTED
pr = field(fp, 'PAGE'); pr.font.size = Pt(8)
sec.different_first_page_header_footer = True

# ------------------------------------------------------------------ title page
P('', after=30)
tp = doc.add_paragraph()
r1 = tp.add_run('Ayas'); r1.font.size, r1.bold, r1.font.color.rgb = Pt(46), True, INK
r2 = tp.add_run('  |  '); r2.font.size, r2.font.color.rgb = Pt(36), RGBColor(0xB8, 0xBC, 0xC4)
r3 = tp.add_run('अयस्'); r3.font.size, r3.font.color.rgb = Pt(40), ORANGE
r3.font.name = DEVA_FONT
r3._r.rPr.rFonts.set(qn('w:cs'), DEVA_FONT)
szcs = OxmlElement('w:szCs'); szcs.set(qn('w:val'), '80'); r3._r.rPr.append(szcs)
P('EVERY SURFACE, SEEN TRUE.', size=10, color=MUTED, after=18)
P('**Project Report**', size=20, after=2)
P('On-device visual quality control for steel and metal surfaces', size=13, color=MUTED, after=16)
par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
par.add_run().add_picture(os.path.join(FIG, 'v_product.jpg'), width=Cm(TEXT_W))
P('', after=8)
T([['Competition', 'Arm Bharat AI-SoC Challenge 2026-27, Problem Statement 5: Manufacturing (real-time surface-defect detection and QC assistant)'],
   ['Platform', 'Raspberry Pi 5 (Arm Cortex-A76 ×4) · Raspberry Pi AI HAT+ 2 (Hailo-10H) · Raspberry Pi AI Camera'],
   ['AI models', 'YOLOv8n defect detector (INT8 on the Hailo-10H) · Llama 3.2 1B Instruct via ExecuTorch on the Arm CPU · physics-informed neural network (PINN) for 3D depth'],
   ['Authors', 'Binay Verma, Dhanush D, Kushal S'],
   ['Faculty mentor', 'Dr. Subir Roy'],
   ['Date', '25 November 2026'],
   ['Repository', 'github.com/kushal-script/ayas (private) · AGPL-3.0 · Built with Llama']],
  widths=[3.2, 13.8], head=False, size=9)
page_break()

# ------------------------------------------------------------------ contents
H('Contents', 1)
toc = ['1. Executive summary', '2. The problem and how Ayas answers the brief', '3. System architecture',
       '4. Methodology', '5. AI models developed', '6. Results', '7. Hardware utilisation',
       '8. Optimisation techniques', '9. The product: 3D model and housing', '10. Electronics and PCB design',
       '11. Serviceability, modularity and assembly', '12. Design choices and trade-offs',
       '13. Verification and software quality', '14. Limitations and roadmap', '15. Arm tools, licences and references',
       'Appendix A. Board summary and key parts', 'Appendix B. Repository map']
for t in toc:
    P(t, size=10.5, after=3)
page_break()

# ------------------------------------------------------------------ 1
H('1. Executive summary', 1)
P('Steel service centres and fabrication shops still check plates and sheets by eye for scratches, dents, pits, '
  'corrosion, cracks and stains. Manual inspection is slow, inconsistent between operators and shifts, and leaves no '
  'defensible record. **Ayas** is a handheld inspector that captures a plate area, finds and measures defects, applies a '
  'written severity specification, logs the result and explains it to the operator, entirely on a Raspberry Pi 5 with no cloud.')
P('The hard requirement set at the start was that the inspector **must work in any lighting**, from a dark room to direct '
  'sun on a shiny plate. Robustness comes from the capture method rather than from the network: a 5-frame LED burst in '
  'raw, ambient subtraction, glare removal and a quality gate that refuses a capture it cannot trust. A light-tight clip-on '
  'hood with a low-angle LED ring is the certified configuration.')
T([['What', 'Result'],
   ['Lighting robustness (twin, hood)', 'Wrong verdict with no warning in **2 of 210** captures across 7 conditions from dark to ~45,000 lux, and **0 unsafe**; the old single-frame method had 14 of 210. Both residual errors trace to one model false positive, fixed by two-burst confirmation (P1h).'],
   ['Detector (YOLOv8n)', 'Recall 1.00 scratch, 1.00 dent, 0.98 crack, 0.83 corrosion, 0.80 stain, 0.79 pit; F1 0.83–0.85 with the hood'],
   ['Camera to verdict on the Pi 5', '**1.7 s p50** with the detector on the Hailo-10H (detect **47 ms**, 28× faster than the CPU); 0.98 GB RAM'],
   ['Llama 3.2 1B via ExecuTorch', 'First token 5.4 s, 11 tokens/s decode, cause note in 10.3 s; 9 of 9 notes passed the factual guardrail'],
   ['3D depth (PINN)', 'Dent depth error 0.053 mm with the L/R LEDs, 0.021 mm with a 6-sector ring: 2.1–3.4× better than classical integration'],
   ['Device software', 'The on-device app reproduced the evaluation verdict for **28 / 28** bursts on the Pi'],
   ['Product hardware', '13 KiCad boards (POWER, 6-layer LIGHT/IO, pods, docks, buttons, charge port), all routed with 0 ERC / 0 DRC errors, fitted in a Fusion 360 housing with 0 interferences across 318 component models'],
   ['Engineering', '~46,500 lines of Python, ~870 automated tests, parametric CAD and PCB generators, a physically based digital twin, 2-minute demo video']],
  widths=[4.6, 12.4], caption='Headline results. All accuracy numbers are from the physically based digital twin; real-specimen validation is Phase 2.')
callout('**Honest scope.** All accuracy evidence so far comes from a physically based Blender digital twin; real steel specimens are the first '
        'Phase 2 task. Severity thresholds are illustrative, informed by EN 10163-2 / ISO 7788, ISO 8501 and IS 2062. Ayas is decision '
        'support, not a certified inspection system.')

# ------------------------------------------------------------------ 2
H('2. The problem and how Ayas answers the brief', 1)
P('Two things make automated metal inspection hard to do cheaply:')
B(['**Shiny metal under uncontrolled light.** Glare, lamp reflections, shadows and sunlight change the image more than the defects do.',
   '**Turning a detection into a decision** that an operator can act on and a QA record can defend.'])
F(os.path.join(FIG, 'v_burst.jpg'), 'The capture method: each scan is a 5-frame LED burst. The dark frame removes room light, left and right remove glare, '
  'and a quality gate refuses any capture it cannot trust (frame from the project demo video).')
T([['Brief requirement', 'How Ayas meets it'],
   ['Detect surface defects, cracks, anomalies', 'YOLOv8n on 640 px tiles of a 0.08 mm/px rectified plate grid; 6 classes'],
   ['Classify type and severity', 'Class from the detector; severity and verdict from configs/spec_steel.yaml, informed by EN 10163-2 / ISO 7788, ISO 8501, IS 2062'],
   ['Alert with actionable recommendations', 'Verdict banner and buzzer, numbered operator steps, Llama cause-and-prevention note and Q&A (Hindi too)'],
   ['Log results', 'One JSON record per inspection (ayas.inspection/3.0), SQLite log, overlay image; USB export to CSV/PDF'],
   ['Real time at the edge, no cloud', 'Pi 5 Cortex-A76 + Hailo-10H NPU; 1.7 s camera to verdict; fully offline'],
   ['Llama via ExecuTorch on Arm', 'Llama 3.2 1B Instruct SpinQuant INT4 as an ExecuTorch .pte, XNNPACK backend, on the Cortex-A76']],
  widths=[5.2, 11.8], caption='Mapping to the PS5 brief.')

# ------------------------------------------------------------------ 3
H('3. System architecture', 1)
P('Ayas runs two separate pipelines that exchange only one JSON record. **Pipeline A** (vision) is deterministic and '
  'produces the verdict. **Pipeline B** (Llama) explains it. The verdict never waits for, and is never changed by, the language model.')
F(os.path.join(FIG, 'd_system.png'), 'The two pipelines. Only the JSON record crosses from vision to Llama.')
F(os.path.join(FIG, 'v_flow.jpg'), 'End to end on the device: scan → clean and gate → rectify and tile → detect → 3D depth → measure and rules → explanation and output.')
T([['Stage', 'Where it runs', 'What it does'],
   ['Capture', 'AI Camera (raw, fixed exposure 10 ms)', '5-frame burst B1 · L · A · R · B2 with the LEDs switched per frame; no auto-exposure or ISP'],
   ['Light processing + gate', 'Cortex-A76 (NumPy)', 'Ambient subtraction from mean(B1, B2); L/R combination; motion, flicker, saturation, closure and hood-leak checks → PASS / PASS_FLAGGED / RETAKE'],
   ['Rectify and tile', 'Cortex-A76', 'Plate homography to 0.08 mm/px with a 5 mm border; 640 px tiles, ≥ 128 px overlap'],
   ['Detect', 'Hailo-10H (INT8 HEF) or CPU (ONNX Runtime)', 'YOLOv8n, 6 classes, on-chip NMS; seam merge across tiles'],
   ['Measure + rules', 'Cortex-A76', 'Length, width, area, edge distance in mm; severity tier and disposition; worst disposition = verdict'],
   ['Record', 'Cortex-A76', 'JSON record, SQLite row, overlay JPEG; station UI on the 5-inch display'],
   ['3D surface (on demand)', 'Cortex-A76, 3 worker processes', 'PINN height field per dent, pit or scratch with depth and uncertainty'],
   ['Advise', 'Cortex-A76, separate process', 'Llama 3.2 1B via ExecuTorch writes the cause note and answers questions, behind a guardrail']],
  widths=[3.2, 4.6, 9.2], caption='Processing stages and where each runs.')

# ------------------------------------------------------------------ 4
H('4. Methodology', 1)
H('4.1 Robust capture: the 5-frame burst and the quality gate', 2)
P('Because exposure, gain and white balance are fixed and the data is linear raw, the device can:')
B(['**subtract the ambient light** using the mean of the two LEDs-off frames B1 and B2, which also cancels slow drift;',
   '**remove glare** by combining the left and right LED frames: a specular highlight from one side is absent from the other '
   '(min(L, R) for the open pods, mean(L, R) for the hood ring);',
   '**detect motion and flicker** from B1 against B2 and from the closure of L + R against A.'])
P('The quality gate (src/ayas/lightproc.py with configs/capture.yaml) scores every burst on ambient ratio, saturated and masked '
  'fractions, motion, flicker, closure residual and a hood-leak test. It returns PASS, PASS_FLAGGED (kept with a note) or '
  'RETAKE with a plain instruction such as "Too bright: shade the part or fit the hood, then retake". The design rule is: '
  '**when the image cannot be trusted, refuse instead of guessing.**')
H('4.2 The physically based digital twin', 2)
P('Real defective steel specimens with exact ground truth are slow and costly to make, so Phase 1 was proved in a digital twin '
  'built in Blender 5.2 (Cycles):')
B(['**Plates** in six finishes: hot-rolled with mill scale, cold-rolled, stainless 2B and brushed, galvanised, aluminium; GGX microfacet shading.',
   '**Defects** in six classes with physical geometry (height fields for dents, pits and scratches) and exact ground truth: class, mask, size in mm, depth.',
   '**Unlabelled distractors** (marker, dust, grind marks, burrs, label residue, holes) so the model learns what not to flag.',
   '**The device** modelled from the CAD: LED positions, hood and camera; seven lighting conditions with real lux and colour temperature, from a dark room to ~45,000 lux sun.',
   '**Sensor and processing**: a shot-and-read noise model from the camera datasheet; the same lightproc.py code runs in the twin and on the Pi.',
   '**Validation**: rendered photometry matches analytic formulas within 2.4 %; the optical forward model matches Blender within 0.1–1.1 % on flat metal and dents.'])
T([['Set', 'Size', 'Use'],
   ['Training', '800 scenes → 5,176 training / 430 validation tiles', 'Detector training'],
   ['Lighting test matrix', '15 unseen plates (seeds ≥ 900100) × 7 conditions × hood/open × 2 variants × 2 methods = 840 captures', 'Pre-registered lighting experiment'],
   ['JEPA set', '48 paired plates', 'Self-supervised anomaly experiment'],
   ['Depth set', '96 dents and pits, 18 per-LED passes', 'PINN and depth-network evaluation']],
  widths=[3.4, 8.6, 5.0], caption='Twin datasets. Rendering ran on rented JarvisLabs GPUs (about 3.5 h on an L4); seeds in DATASETS.md.')
F2(os.path.join(REPO, 'reports/figures/scenes/burst_s900100_C3h_v0.jpg'), os.path.join(REPO, 'reports/figures/scenes/ground_truth_s900100_C0h_v0.jpg'),
   'A twin test plate after burst processing under window daylight with the hood (left) and its ground truth (right).')
H('4.3 Severity rules and the verdict', 2)
P('configs/spec_steel.yaml maps each defect to a severity tier and a disposition; the part verdict is the worst disposition: '
  '**PASS < CLEAN_AND_REINSPECT < REWORK < HOLD_FOR_QA < REJECT**. The rules include size limits by plate thickness '
  '(EN 10163-2 / ISO 7788), ISO 7788 clustering within 40 mm, edge and hole proximity, rust grades and "contamination never rejects".')
P('Three rules make the device honest about what it cannot see:')
N(['**The camera does not measure depth** in the verdict path: depth-governed scratches, dents and pits above minor go to HOLD_FOR_QA for a gauge check.',
   '**Confidence is not severity**: a detection below 0.5 means HOLD_FOR_QA; a low-confidence crack keeps REJECT only if the second burst of the inspection confirms it (P1h).',
   '**Hood-only certification**: in open mode the device never outputs PASS or CLEAN_AND_REINSPECT; those become HOLD_FOR_QA with "fit the hood to certify".'])
F(os.path.join(FIG, 'v_rules.jpg'), 'The rules engine: each defect gets a rule, a severity and a disposition; the worst disposition decides the verdict.')
H('4.4 Experimental protocol', 2)
P('The lighting experiment was **pre-registered**: the pass criterion was fixed before the test and the test plates were never used for '
  'training, tuning or calibration. Hood (product), every condition: retake ≤ 5 %, F1 ≥ F1(dark) − 0.05, verdict agreement with the dark room ≥ 95 % '
  'and **zero silent failures** (the gate accepts a capture and the verdict is wrong). Open device: zero silent failures. Fixes found after '
  'the test are reported separately and must be confirmed on fresh plates.')

# ------------------------------------------------------------------ 5
H('5. AI models developed', 1)
H('5.1 YOLOv8n surface-defect detector', 2)
F(os.path.join(FIG, 'v_yolo.jpg'), 'YOLOv8n: CSP backbone, feature-pyramid neck and three anchor-free detection heads, compiled to a Hailo HEF.')
T([['Item', 'Value'],
   ['Architecture', 'YOLOv8n (~3 M parameters), anchor-free, 3 heads (P3–P5)'],
   ['Classes', 'scratch, dent, pit, corrosion, crack, stain'],
   ['Training', 'From COCO weights, 150 epochs, native scale (no resizing), light colour and scale augmentation; JarvisLabs L4 GPU (run r_82a2be7e)'],
   ['Data', '5,176 training / 430 validation 640 px tiles from 800 twin scenes, burst-processed'],
   ['Baseline arm', 'Same network trained on single auto-exposed frames (the "old method")'],
   ['CPU export', 'ONNX (12.8 MB); ONNX vs PyTorch agree to 0.0013 on the raw head output'],
   ['NPU export', 'Hailo DFC 5.1.0 + Model Zoo 5.1.0; INT8 with 1,024 calibration tiles; cut at the six head convolutions with on-chip NMS (score 0.2)']],
  widths=[3.6, 13.4], caption='Detector summary.')
H('5.2 Llama 3.2 1B Instruct via ExecuTorch (the QC assistant)', 2)
F(os.path.join(FIG, 'v_llama.jpg'), 'Pipeline B: the rules engine writes the facts; Llama 3.2 1B (SpinQuant INT4, ExecuTorch, XNNPACK) writes the likely cause and prevention.')
B(['**Model:** Meta Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8 (4-bit weights, group size 32, 8-bit embeddings), exported on the Pi with ExecuTorch v1.6.0 (XNNPACK); 1.08 GB .pte.',
   '**Runner:** ExecuTorch llama_main as a separate process, or the in-process TextLLMRunner; both on the Cortex-A76 with 4 threads.',
   '**Division of labour (redesigned after the first Pi run):** the rules engine writes the verdict, the deciding defects with their measurements and rule, and the numbered steps, which are exact by construction. Llama writes the cause-and-prevention note for the main deciding defect and answers "Why?" and open questions, including in Hindi.',
   '**Prompt:** built from the JSON record, never the image. Cause note ~611 tokens with its opening words pre-filled; follow-ups use a plain-text brief of ~860 tokens instead of ~1,340 for raw JSON; max 160 new tokens.',
   '**Guardrail** on every answer: it rejects contradicting the verdict, naming a defect type not in the record, quoting a number or depth not in the record, banned phrases and unlisted references. A rejected answer gets one rewrite naming the broken rule; then a labelled template answers. The UI always shows who answered.'])
F(os.path.join(FIG, 'v_qa.jpg'), 'Station functions: each question goes to the engine that can answer it safely; "How to fix?" and override requests always get the rules engine\'s exact answer.')
H('5.3 Physics-informed neural network (PINN) for 3D depth', 2)
P('For dents, pits and scratches, a small coordinate network proposes a height map per defect; a differentiable light model renders what each LED '
  'would see, and the error against the camera frames trains the network until they match. No training data is needed: the physics is the supervision.')
F2(os.path.join(FIG, 'v_pinn.jpg'), os.path.join(FIG, 'v_surface.jpg'),
   'PINN depth: the light model compares rendered and captured left/right images (left); the fitted surface of a 0.32 mm dent (right).')
B(['Fitted per defect in parallel worker processes (3 workers on the Pi); the screen shows a rotatable height map with the depth and its uncertainty.',
   'The proposed severity rule uses a depth only when the fit converged and the calibrated 90 % interval is ≤ 0.08 mm and does not cross a limit.',
   'Measurement aid only: the verdict never depends on it in Phase 1.'])
H('5.4 Models tried and not adopted', 2)
T([['Model', 'What it was', 'Result', 'Decision'],
   ['I-JEPA anomaly head', 'Cross-lighting self-supervised ViT-Tiny as an anomaly map', 'Region AUROC 0.975 hood / 0.957 open vs ResNet18 PatchCore 0.965 / 0.957', 'Not adopted: missed the pre-set +0.02 margin, less lighting-invariant, too slow for the Pi'],
   ['Depth network', 'A CNN trained to predict height from the burst, fine-tuned on Blender crops', 'Dent error 0.150 mm vs 0.022 mm for the physics fit', 'Not adopted: the PINN is 7× better'],
   ['ResNet18 PatchCore', 'Memory-bank anomaly scorer', 'Competitive AUROC; thresholds uncalibrated', 'Phase 2 candidate']],
  widths=[3.0, 4.6, 4.8, 4.6], caption='Experiments reported honestly as tried and not adopted.')

# ------------------------------------------------------------------ 6
H('6. Results', 1)
H('6.1 Lighting experiment (pre-registered, digital twin)', 2)
F(os.path.join(FIG, 'c_lighting.png'), 'Wrong verdicts with no warning, by method and device (of 210 captures each).')
T([['Method', 'Device', 'Wrong, no warning', 'Unsafe', 'Retakes', 'F1'],
   ['Burst + gate (new)', '**hood (product)**', '**2 / 210**', '**0**', '0 / 210', '0.83–0.85'],
   ['Burst + gate (new)', 'open', '12 / 210', '2', '142 / 210', '0.72–0.89'],
   ['Single frame (old)', 'hood', '14 / 210', '0', 'no gate', '0.91–0.92'],
   ['Single frame (old)', 'open', '16 / 210', '0', 'no gate', '0.80–0.91']],
  widths=[3.6, 3.4, 3.0, 1.8, 2.4, 2.8], caption='Lighting experiment: 15 unseen plates, 7 conditions (dark room to direct sun ~45,000 lux).')
B(['**With the hood**, every condition from dark to direct sun meets the retake, F1 and agreement limits; verdict = truth in 30/30 captures in five conditions and 29/30 in two; '
   '0 false alarms on defect-free plates (0.33 per plate for the old method).',
   '**Strict result: FAIL by 2 captures.** Both are one low-confidence (0.27 / 0.32) false "crack" on unlabelled mill-scale texture at a plate edge, giving REJECT where the truth is HOLD_FOR_QA: '
   'stricter than the truth, not unsafe. The score does not follow lux (Spearman ρ = 0.15), so the cause is the model, not the lighting.',
   '**Fix adopted:** two-burst confirmation (P1h). Re-scored on the recorded detections, hood silent failures go to 0 / 210 with no new unsafe verdict and no change to the 280 captures with real cracks; it must be confirmed on fresh plates.',
   '**Without the hood**, the gate refuses most bright scenes (142 of 210, including every direct-sun capture), and dark mill scale caused 2 unsafe misses; hence **hood-only certification**.',
   '**Caveat:** in the twin a seated hood is light-tight by construction, so the hood rows test repeatability, not real-world light leakage; leakage is the first Phase 2 measurement.'])
F(os.path.join(FIG, 'c_recall.png'), 'Detector recall per class (burst capture, hood, all 7 conditions). Precision 0.77–0.79.')
H('6.2 The device software equals the evaluated pipeline', 2)
P('The on-device app, fed the 28 showcase twin bursts as its camera, reproduced the evaluation verdict for **28 / 28** captures on the Mac and **28 / 28** on '
  'the Raspberry Pi 5, including the known failure and every RETAKE. With the detector on the Hailo-10H the verdict matched on 27 of 28: the one difference '
  'is the edge false positive, which INT8 quantisation happens not to fire (a side effect, not a fix).')
H('6.3 On-device performance (Raspberry Pi 5)', 2)
F(os.path.join(FIG, 'c_latency.png'), 'Camera-to-verdict latency on the Pi 5, p50, by stage: CPU ONNX against the Hailo-10H NPU.')
T([['Metric', 'Raspberry Pi 5', 'Mac M3 CPU (reference)'],
   ['Camera to verdict p50 / p95', '**1,665 / 2,106 ms** (Hailo-10H); 2,856 / 3,331 ms (CPU)', '617 / 716 ms'],
   ['Detector per inspection', '**47 ms** (Hailo-10H) vs 1,329 ms (CPU ONNX)', '–'],
   ['Throughput', '0.57 inspections/s (Hailo), 0.39 (CPU)', '1.8 inspections/s'],
   ['Peak memory', 'app 0.98 GB (Hailo) / 1.08 GB (CPU); Llama runner 1.38 GB', '1.7 GB'],
   ['Llama cause note', 'first token 5.4 s (model clock), 11.0 tok/s decode, 112 tok/s prefill, note in 10.3 s', '–'],
   ['Camera to advice', '13.5 s p50, 13.9 s p95', '–'],
   ['3D surface, 5 relief defects', '~15 s, first surface on screen after ~8 s', '–'],
   ['CPU temperature', '51–67 °C, no throttling (get_throttled = 0x0)', '–']],
  widths=[4.2, 8.8, 4.0], caption='Measured on the Pi (reports/pi_phase1_raspberrypi_20261122-*, reports/pi_llm_final4_20261122-183748).')
H('6.4 Llama answer quality', 2)
B(['**First Pi run** (Llama restated the whole record): it answered 7 of 8 inspections, and 3 of the 7 answers had factual errors (a crack explained as a scratch, an inverted depth rule, an invented depth). The verdicts on screen were right every time because they come from the rules engine.',
   '**Redesign:** facts moved to the rules engine; Llama got the job it does reliably.',
   '**Final Pi run** (16 inspections): 9 of 9 cause notes came from Llama and none failed the guardrail; the 7 RETAKE and PASS results were answered by the rules engine.',
   '**Open questions** ("Why does a crack matter?") take 31–35 s and are general and loosely worded: the weakest part of Pipeline B, and Phase 2 work (a larger model or a fine-tune on station Q&A).'])
H('6.5 3D depth accuracy (twin)', 2)
F(os.path.join(FIG, 'c_pinn.png'), 'Dent-depth error: the PINN against classical photometric integration, for the L/R LEDs and 3- or 6-sector rings.')
B(['Lighting-invariant with the hood: depth changes across all seven conditions by at most 0.023 mm for dents.',
   'Twin dent plate under the hood, dark room to direct sun: the 0.331 mm dent reads 0.30–0.34 mm and the 0.168 mm dent 0.15–0.16 mm.',
   'Pits are under-estimated (28–88 %) and are shown as indicative only; a 6-sector ring halves the dent error again.',
   'Under the proposed rule, no depth-based decision was more lenient than the decision with the true depth (0 of 89 coupon and 7 pilot decisions).'])

# ------------------------------------------------------------------ 7
H('7. Hardware utilisation', 1)
P('Each part of the SoC platform does the work it is best at; nothing waits on anything it does not need.')
T([['Resource', 'What runs on it', 'Utilisation and notes'],
   ['Arm Cortex-A76 ×4 (Pi 5, 8 GB)', 'Burst processing and gate (NumPy), rectify, tile, measure, rules, UI; Llama via ExecuTorch (4 threads, XNNPACK); PINN workers', 'Vision CPU path ~1.1 s/inspection; Llama 11 tok/s; peak RAM 0.98 GB (app) + 1.38 GB (Llama) of 8 GB; 51–67 °C, no throttling'],
   ['Hailo-10H NPU (AI HAT+ 2)', 'YOLOv8n INT8 with on-chip NMS', '47 ms per inspection (all tiles); frees the CPU for Llama and 3D; verdicts equal the CPU on 27 / 28'],
   ['Raspberry Pi AI Camera (IMX500)', 'Raw capture with fixed exposure, gain and white balance', 'ISP and auto-exposure bypassed by design so ambient subtraction is linear; 10 ms exposure = one 100 Hz flicker period'],
   ['RP2040 MCU (LIGHT/IO board, Phase 2)', 'LED burst timing, 10 constant-current set-points, buttons, power-hold, UV interlock, fan, buzzer, NTCs', 'Hard real-time light switching off the Linux CPU; USB serial to the Pi'],
   ['Separate processes', 'Pipeline A and Pipeline B', 'The verdict never waits for Llama; the Llama runner can be restarted without touching vision']],
  widths=[3.8, 6.4, 6.8], caption='How the platform is used.')
P('**Memory budget.** Model files: YOLOv8n ONNX 12.8 MB, Hailo HEF, Llama .pte 1.08 GB. With the app, Llama and the PINN pool loaded, '
  'peak use stays under 2.5 GB of 8 GB, leaving room for logs, images and the station UI.')

# ------------------------------------------------------------------ 8
H('8. Optimisation techniques', 1)
F(os.path.join(FIG, 'c_speedups.png'), 'Measured speed-ups from the optimisations below.')
T([['Technique', 'Where', 'Effect'],
   ['INT8 post-training quantisation + compilation to a Hailo HEF (1,024 calibration tiles), cut at the head convolutions with on-chip NMS', 'Detector', '1,329 → 47 ms per inspection (28×); camera to verdict 2.86 → 1.67 s'],
   ['SpinQuant INT4 weights (group 32) with 8-bit embeddings; ExecuTorch XNNPACK delegate (KleidiAI kernels where built in)', 'Llama', '1B model in a 1.08 GB .pte and 1.38 GB RAM; 11 tok/s on the Cortex-A76'],
   ['Prompt engineering: facts from the rules engine, plain-text brief instead of JSON, pre-filled opening words, max 160 new tokens', 'Llama', 'Prompt 1,340 → 611 tokens; first token 15.8 → 5.4 s; answer ~54 → 10.3 s; factual errors 3/7 → 0/9'],
   ['Native-scale tiling at 0.08 mm/px with 128 px overlap, instead of resizing whole plates', 'Detector', 'Small defects keep their pixels; seam merge removes duplicates'],
   ['Capture-side invariance (burst, ambient subtraction, L/R glare removal) instead of a bigger network', 'Vision', 'A 3 M-parameter model holds across 7 lighting conditions; the gate refuses instead of failing'],
   ['PINN fast mode (half the fit steps), warm process pool, one defect per core, first surface streamed early', '3D', 'Plate with 5 defects 55 → 15 s; first surface after ~8 s'],
   ['Process isolation; streaming tokens to the UI after the verdict is shown', 'System', 'Verdict latency unaffected by Llama; perceived wait shorter'],
   ['Analogue constant-current LED sinks (not PWM dimming) and a 6-layer LIGHT/IO board', 'Hardware', 'No rolling-shutter banding; repeatable bursts; the dense RP2040 board routes completely']],
  widths=[7.6, 2.2, 7.2], caption='Optimisation techniques and their measured effect.')

# ------------------------------------------------------------------ 9
page_break()
H('9. The product: 3D model and housing', 1)
P('The handheld is a parametric Fusion 360 model generated from one Python script (hardware/enclosure/ayas_handheld_fusion.py), '
  'so every dimension, clearance and fastener is a parameter and the whole product rebuilds in one run. A second script '
  '(ayas_assembly_fusion.py) places every KiCad board as its real 3D model and draws the cable harness between the actual connectors.')
F(os.path.join(RENDERS, '1_front.jpg'), 'The complete Ayas handheld, front: two LED pods (3 white modules and 9 UV LEDs each), the camera with its hood, index triggers and the power button.')
F2(os.path.join(RENDERS, '2_back.jpg'), os.path.join(RENDERS, '3_hood.jpg'), 'Back (5-inch touch display, thumb buttons, USB-A export) and with the clip-on inspection hood fitted.')
F(os.path.join(FIG, 'v_exploded.jpg'), 'Exploded view: display, Raspberry Pi 5 with the AI HAT+ 2, the AI Camera between the LED pods, and the light-tight hood with its low-angle LED ring.')
T([['Feature', 'Design'],
   ['Form', 'Two-grip handheld ("F1-wheel" form): 268 mm wide, grips 224 mm apart, 12° rake, 7° splay; ~900 g of PETG with solid walls (708 cm³ printed)'],
   ['Display and compute', 'Waveshare 5-inch DSI touch display (800 × 480) on the back lid; Pi 5 + AI HAT+ 2 stacked behind it on posts; 30 mm fan under the top vents'],
   ['Camera and light', 'AI Camera on the centre line with a removable cone hood; two toed-in (14°) LED pods: 3 white 12 V modules and 9 plug-in 395 nm UV LEDs each, tilted toward the image centre'],
   ['Hood', 'Clip-on light-tight hood with a 6-sector low-angle white ring; thumb screws; docks on spring pins (12 V, 6 ring channels, interlock loop)'],
   ['Power', 'USB-C PD power bank (33 W) behind a 4-screw bay door; USB-C charge port in the side wall; 12 mm ring-LED power button'],
   ['Controls', '2 index triggers (both within 300 ms = Inspect), 4 thumb buttons (part ID, light mode, region, mark/retake), power button; touch screen for review'],
   ['Colours', 'Rugged industrial palette: graphite shell, black rubberised grips, orange accents']],
  widths=[3.4, 13.6], caption='Housing summary.')
F(os.path.join(RENDERS, '7_cutaway.jpg'), 'Cutaway from the back with the shell at 25 % opacity: the boards on the battery shelf, the Pi stack, the battery, the button boards in the wings and the harness.')

# ------------------------------------------------------------------ 10
page_break()
H('10. Electronics and PCB design', 1)
P('Phase 1 used jumper wiring and LEDs switched through resistors. For the product, the light must be identical from burst to burst '
  '(the depth fit and the glare removal assume it), the UV must be interlocked, and the device must be robust when carried and dropped. '
  'Phase 2 therefore has its own boards: **13 PCBs**, all generated from one Python description (hardware/pcb/design.py) so the schematic and the layout always agree.')
F(os.path.join(FIG, 'd_electronics.png'), 'Electronics architecture: power, control and the light modules.')
H('10.1 Design flow', 2)
N(['**design.py** describes every board: parts, footprints, manufacturer part numbers and nets, referenced by pin name.',
   '**gen_sch.py** writes the KiCad 10 schematic; **gen_pcb.py** places the parts in each board frame, matched to the Fusion housing; **gen_pro.py** writes the fab rules.',
   '**route.py** exports Specctra DSN, auto-routes with Freerouting 2.5, imports the session, pours ground zones and enforces the minimum track width.',
   '**kicad-cli** runs ERC and DRC; **tests/test_hardware_design.py** checks the design itself; the Fusion assembly checks the mechanical fit.'])
H('10.2 Power (POWER board)', 2)
B(['**USB-C PD sink** (ST STUSB4500) negotiates 12 V from the power bank; a TI TPS565201 buck makes 5.1 V / 5 A for the Pi; a TI INA226 measures the LED rail in every burst frame.',
   '**Power on / off with no standby drain:** the button wakes the bank through 5.1 kΩ CC terminations; the RP2040 then holds power (HOLD). Holding the button 2 s saves the record and shuts down the Pi; 10 s forces off. A watchdog reset also releases HOLD, so a fault ends in off, never stuck on.',
   '**Charge-through:** with the device off, a TS5A23157 router passes the charger\'s CC lines to the bank so they negotiate PD fast charging; back-to-back AO3401A P-FETs with a 10 V Zener gate clamp pass VBUS.',
   '**Protection:** SMBJ24A/SMAJ24A TVS on every VBUS input and a 2.6 A resettable fuse on the 12 V rail.'])
H('10.3 Light and I/O (LIGHT/IO board, 6-layer)', 2)
B(['**RP2040** on USB to the Pi: burst timing, buttons, buzzer, fan, three NTCs, power-hold.',
   '**10 constant-current sinks** (op-amp + logic-level MOSFET + sense resistor): white L/R (137 mA full scale), UV L/R (62 mA), 6 hood-ring channels (62 mA). The current is set in analogue, not PWM-dimmed, so the rolling shutter never bands.',
   '**UV interlock in hardware:** the UV sinks are disabled unless the hood\'s loop-back is present.',
   '**Layout:** 46 × 55 mm, 6 layers (inner layer 1 a solid ground plane); 0.1 mm track and space and 0.2 mm vias so the RP2040\'s 0.4 mm-pitch QFN fans out; decoupling and crystal parts pinned beside their pins. Every connection routed, 0 DRC errors.'])
F(os.path.join(FIG, 'pcb_montage.jpg'), 'The routed boards (KiCad 3D renders): POWER, LIGHT/IO top and bottom, UV pod hub, WHITE-L carrier, SIDE-L buttons, CHARGE, POD DOCK and HOOD DOCK.')
H('10.4 Light modules and small boards', 2)
T([['Board', 'Size (mm)', 'Purpose'],
   ['UV pod hub ×2', '43 × 30.5', '9 plug-in 5 mm 395 nm UV LEDs on 2-pin sockets marked + / −, 3 strings of 3 with 47 Ω ballast; pod NTC; ZIF for the pod flex'],
   ['WHITE-L / WHITE-R', '50 × 66', '3 owned 12 V white modules (3 × 2835 LEDs, 40 × 10 mm) per pod, in parallel; tabs on M2 screws into SMD nuts; leads in push-in terminals'],
   ['Pod flex ×2', '11 × 60', '10 straight conductors, palindrome pinout: correct either way round'],
   ['POD DOCK L / R', '28 × 14', 'ZIF for the pod flex behind the body front face; keyed JST-SH cable to LIGHT/IO (6-way L, 7-way R)'],
   ['HOOD DOCK / HOOD PAD', '34 × 13 / 40 × 32', 'Spring pins on the body, gold pads on the hood; 6 push-in terminals for the ring modules; interlock loop'],
   ['SIDE-L / SIDE-R', '36 × 20', 'Push-in terminals TRIG, UPPER, LOWER for the panel buttons; JST-SH 4 / 5 to LIGHT/IO'],
   ['CHARGE', '24 × 12.5', 'Power-only USB-C through the side wall; JST-PH 5 to POWER']],
  widths=[3.6, 2.6, 10.8], caption='Light modules and small boards.')
F2(os.path.join(REPO, 'docs/hardware/renders/4_xray_front.jpg'), os.path.join(REPO, 'docs/hardware/renders/5_xray_back.jpg'),
   'X-ray views with the housing hidden: every board, the LED modules, the buttons and the colour-coded cable harness.')

# ------------------------------------------------------------------ 11
H('11. Serviceability, modularity and assembly', 1)
P('The product is designed to be repaired in the field without ambiguity or tangling:')
B(['**Every service screw goes into a brass heat-set insert**, never bare plastic, so each joint can be opened any number of times (M3 for boards, pods, lids and wings; M2 for small boards, camera and modules).',
   '**No plug fits a wrong socket:** every cable on LIGHT/IO has a different pin count (JST-SH 6, 7, 9, 4, 5, 2), and the USB and power-button cables use JST-PH. An automated test checks this.',
   '**No solder in the field:** white modules and panel buttons use push-in terminals; UV LEDs plug into sockets and pull out from the front of the pod.',
   '**Modules dock, not wire:** each pod connects by its flex into a ZIF on the body (the dock decides the side, so the two pods are identical parts); the hood docks on spring pins.',
   '**Labelled:** every connector\'s function is printed beside it on both faces of its board.'])
T([['Fault', 'Repair'],
   ['A UV LED dead', 'Pull it out from the pod front, push a new one in (long leg to "+"). Nothing opened.'],
   ['A white module dead', 'Pod off (2 × M3), 2 × M2 tab screws, press 2 terminal buttons, swap, push the leads in (red to +).'],
   ['A button dead', 'Wing back off, unscrew the button nut, release its 2 push-in terminals, swap.'],
   ['A pod damaged', 'Swap the whole pod: either pod fits either side.'],
   ['No light / will not power on', 'Swap LIGHT/IO (4 × M3, 7 different plugs) or POWER (4 × M3, 5 plugs).']],
  widths=[4.2, 12.8], caption='Field repairs (docs/hardware/service.md has the full manual, fastener list and assembly order).')

# ------------------------------------------------------------------ 12
H('12. Design choices and trade-offs', 1)
T([['Decision', 'Chosen', 'Alternative', 'Why'],
   ['Lighting robustness', 'Capture method: burst, ambient subtraction, gate', 'A bigger network with heavy augmentation', 'Physics removes the ambient term exactly; a network can only guess it'],
   ['Certification', 'Hood only', 'Open handheld verdicts', 'Open mode missed defects on dark mill scale; refusing is safer than guessing'],
   ['Facts vs language', 'Rules engine writes facts; Llama explains', 'Llama writes the whole answer', '3 of 7 early Llama answers misstated facts; the verdict must be exact'],
   ['Detector accelerator', 'Hailo-10H INT8', 'CPU ONNX', '28× faster detection, frees the CPU for Llama and 3D'],
   ['Depth', 'PINN fit per defect', 'Trained depth network', 'PINN 0.022 vs 0.150 mm; needs no labelled depth data'],
   ['Anomaly head', 'Not shipped', 'I-JEPA', 'Did not beat ResNet18 by the pre-set margin; too slow for the Pi'],
   ['LED drive', 'Analogue constant current', 'PWM dimming', 'PWM bands with a rolling shutter; constant current repeats burst to burst'],
   ['White light', 'Owned 12 V 3-LED modules on screwed carriers', 'Custom LED strip boards', 'Parts already owned, buyable in Bengaluru, replaceable with no solder'],
   ['LIGHT/IO stack-up', '6 layers', 'I2C DACs for set-points on 4 layers', 'Keeps the proven circuit; the dense RP2040 board routes completely'],
   ['Distance inspection', 'Not pursued', 'Interferometry / profilometry at 0.3–3 m', 'Coherence gate vs hand tremor; ambient light swamps distant lamps (desk study)']],
  widths=[3.0, 4.0, 4.2, 5.8], caption='Key design decisions.', size=8.3)

# ------------------------------------------------------------------ 13
H('13. Verification and software quality', 1)
T([['Check', 'Result'],
   ['Automated test suite (pytest)', '~870 tests over 28 test files: lighting processing, rules and policies, twin rendering and post, PINN, LLM guardrails, app, hardware design consistency'],
   ['Hardware consistency tests', '43 passed: both ends of every cable carry the same pinout, flex pinouts are palindromes, no swappable connectors, no dangling nets, LED currents below the module current'],
   ['Schematics (ERC)', '13 boards: 0 errors, 0 warnings'],
   ['Layouts (DRC)', '13 boards: 0 violations, 0 unconnected pads'],
   ['Mechanical fit (Fusion 360)', '318 board component models: 0 interference with the housing, buttons, Pi, display, fan or battery; no floating printed bodies'],
   ['Device = evaluation', '28 / 28 verdicts on the Pi (27 / 28 with the Hailo, the difference explained)'],
   ['Reproducibility', 'Seeds and regeneration in DATASETS.md; model SHA-256 in models/README.md; CAD and PCBs regenerate from scripts']],
  widths=[4.4, 12.6], caption='Verification summary.')

# ------------------------------------------------------------------ 14
H('14. Limitations and roadmap', 1)
P('**Limitations.**')
B(['**Twin only.** Real specimens (mild and stainless steel and aluminium coupons with seeded defects) are Phase 2; the sim-to-real gap is the main open risk.',
   '**Strict lighting criterion not met in the hood** by 2 of 210 conservative captures; the adopted P1h fix must be confirmed on fresh plates.',
   '**Hood light leakage untested** in reality (the twin hood is light-tight by construction).',
   '**Depth not in the verdict yet**; pits are measured unreliably.',
   '**Llama open questions** are slow (30–35 s) and general.',
   '**Hardware open items:** the hood contact board must be redesigned to fit the hood collar; the white module\'s tab pitch, height and current must be measured; the 6-layer fab quote confirmed.'])
P('**Phase 2 roadmap.**')
B(['Real-specimen validation and the hood light-leakage measurement.',
   'Hard-negative retraining on mill-scale edge texture (the root-cause fix for the phantom crack).',
   'PINN depth in the verdict with the 6-sector hood ring, under the calibrated-interval rule.',
   'Fabricate and bring up the 13 boards; RP2040 firmware (USB CDC, power-hold, burst timing).',
   'A larger or fine-tuned language model for station Q&A; ResNet18 PatchCore anomaly head after threshold calibration.',
   'UV burst over a fluorescent penetrant for crack confirmation.'])

# ------------------------------------------------------------------ 15
H('15. Arm tools, licences and references', 1)
T([['Tool', 'Use'],
   ['ExecuTorch (XNNPACK, KleidiAI kernels where built in)', 'Llama 3.2 1B on the Cortex-A76'],
   ['Raspberry Pi 5 (Arm Cortex-A76)', 'Both pipelines, the 3D fit and the UI'],
   ['ONNX Runtime (Arm64)', 'CPU detector path'],
   ['HailoRT / Hailo DFC 5.1', 'NPU detector path'],
   ['RP2040 (Arm Cortex-M0+)', 'Real-time light and I/O controller on the LIGHT/IO board']],
  widths=[7.0, 10.0], caption='Arm platform and tools used.')
P('**Licences.** The project is AGPL-3.0 (because of Ultralytics YOLO); NOTICE lists third parties. **Built with Llama:** Llama 3.2 is licensed under the Llama 3.2 Community License, © Meta Platforms, Inc.')
P('**Standards referenced** (not reproduced): EN 10163-2 / ISO 7788, ISO 8501-1/-3, IS 2062, IS 9954, ISO 9223 / ISO 9224 (corrosion forecast, advisory).')
P('**Project documents:** docs/report/Ayas_Phase1_report.md, docs/twin/DESIGN_STEEL.md, docs/twin/PHYSICS_PINN.md, docs/APP.md, docs/hardware/pcb_phase2.md, '
  'docs/hardware/controls.md, docs/hardware/service.md, reports/ (all results), and the 2-minute demo video.')

# ------------------------------------------------------------------ appendices
page_break()
H('Appendix A. Board summary and key parts', 1)
T([['Board', 'Layers', 'Size (mm)', 'Key parts (manufacturer part numbers)'],
   ['POWER', '2', '46 × 55', 'STUSB4500 (ST), TPS565201 (TI), INA226 (TI), TS5A23157 (TI), LM2936-3.3 ×2, AO3401A ×3, BAT54C, BSS84, SMBJ24A, GCT USB4105 USB-C'],
   ['LIGHT/IO', '6', '46 × 55', 'RP2040 (Raspberry Pi), W25Q16JV flash (Winbond), 12 MHz crystal, AP2112K-3.3, 10 × op-amp + MOSFET constant-current sinks, CMT-8504 buzzer'],
   ['UV pod hub ×2', '2', '43 × 30.5', '9 × Mill-Max 2-pin sockets, 47 Ω ballast ×3, NCP18 NTC, Molex 200528 ZIF 10'],
   ['WHITE-L / WHITE-R', '2', '50 × 66', 'Phoenix PTSM push-in terminals ×3, Würth WA-SMSI M2 SMD nuts ×6, JST-PH 2 side entry'],
   ['POD DOCK L / R', '2', '28 × 14', 'Molex 200528 ZIF 10, JST-SH 6 / 7'],
   ['HOOD DOCK / PAD', '2', '34 × 13 / 40 × 32', 'Mill-Max 0906 spring pins ×9 / gold pads ×9, Phoenix PTSM ×6'],
   ['SIDE-L / SIDE-R', '2', '36 × 20', 'Phoenix PTSM ×3, JST-SH 4 / 5'],
   ['CHARGE', '2', '24 × 12.5', 'GCT USB4125 power-only USB-C, SMAJ24A, JST-PH 5'],
   ['Pod flex ×2', '1 (flex)', '11 × 60', 'Polyimide, 10 straight conductors, stiffened fingers']],
  widths=[3.0, 1.4, 2.6, 10.0], caption='The 13 boards (fab rules 0.15 mm, LIGHT/IO 0.1 mm; all parts genuine brand-name parts).', size=8.4)
T([['Platform part', 'Role'],
   ['Raspberry Pi 5 (8 GB)', 'All compute'],
   ['Raspberry Pi AI HAT+ 2 (Hailo-10H)', 'Detector NPU'],
   ['Raspberry Pi AI Camera (IMX500)', 'Raw capture'],
   ['Waveshare 5-inch DSI display', 'Station UI (800 × 480)'],
   ['USB-C PD power bank (33 W, 12 V PDO)', 'Power'],
   ['12 V white 3-LED modules (40 × 10 mm) ×12', 'White light: 3 per pod, 6 in the hood ring'],
   ['5 mm 395 nm UV LEDs ×18', 'UV light: 9 per pod'],
   ['12 mm and 7 mm panel push buttons, 12 mm ring-LED power button', 'Controls']],
  widths=[7.4, 9.6], caption='Platform and bought-in parts.')
H('Appendix B. Repository map', 1)
T([['Path', 'Contents'],
   ['src/ayas/', 'lightproc (burst + gate), tiles and npu (detectors), rules, report and store, llm (prompts, ExecuTorch runners, guardrail), physics (optics, PINN), twin (Blender digital twin), app (ayas-app, UI, benchmarks, 3D surface)'],
   ['training/', 'YOLO training per capture arm, ONNX export, lighting evaluation, JEPA and depth-network experiments'],
   ['hailo/', 'HEF compilation scripts (DFC 5.1)'],
   ['configs/', 'capture.yaml, twin.yaml, spec_steel.yaml, app.yaml (controls, power, export)'],
   ['hardware/enclosure/', 'Fusion 360 generator for the housing and the full assembly'],
   ['hardware/pcb/', 'design.py (single source), schematic/PCB/project generators, router, 13 KiCad boards'],
   ['docs/', 'Plans, app guide, twin design and status, hardware spec, controls, service manual, renders, research'],
   ['reports/', 'Lighting matrix, JEPA, PINN, Pi benchmarks, figures, demo video'],
   ['tests/', 'pytest suite including the hardware design consistency test']],
  widths=[3.6, 13.4], caption='Where everything is.')
P('')
P('Ayas · Every surface, seen true. · Built with Llama.', size=9, color=MUTED, align='c')

doc.save(OUT)
print('saved', OUT)
