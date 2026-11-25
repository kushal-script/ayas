"""Full Ayas Phase 2 assembly in Fusion: housing + every KiCad board (STEP) + buttons + cable harness,
then rendered views. Run inside Fusion after the board STEPs are exported:

  kicad-cli pcb export step --subst-models --user-origin 100x100mm -o <STEP_DIR>/<board>.step <board>.kicad_pcb

and the connector table (<STEP_DIR>/conn.json: board -> ref -> centre (u, v), size, side, missing model).
Board placement comes from assembly_frames() in ayas_handheld_fusion.py, so the two files always agree.
"""
import json
import math
import os

import adsk.core
import adsk.fusion

MM = 0.1
HERE = os.path.dirname(os.path.abspath(__file__))
STEP_DIR = os.environ.get('AYAS_STEP_DIR', '/tmp/ayas_step')

BOARDS = {   # placement name -> STEP file
    'power': 'power', 'light': 'light', 'charge': 'charge', 'uv_l': 'uv', 'uv_r': 'uv',
    'white_l': 'white_l', 'white_r': 'white_r', 'pod_dock_l': 'pod_dock_l', 'pod_dock_r': 'pod_dock_r',
    'hood_dock': 'hood_dock', 'hood_pad': 'hood_pad', 'btn_l': 'btn_l', 'btn_r': 'btn_r',
}

# cable: (name, colour rgb, end A, end B); an end is ('board', ref) or ('body', name, face)
CABLES = [
    ('LINK', (230, 120, 30), ('power', 'J2'), ('light', 'J1')),
    ('CHARGE', (200, 40, 40), ('charge', 'J2'), ('power', 'J6')),
    ('BANK USB-C', (30, 30, 30), ('body', 'PowerBank', '+x'), ('power', 'J1')),
    ('PI 5V', (200, 40, 40), ('power', 'J3'), ('body', 'PiStack', '-y')),
    ('PWR BTN', (240, 200, 40), ('power', 'J5'), ('body', 'PwrButton', '-y')),
    ('USB to Pi', (40, 90, 200), ('light', 'J2'), ('body', 'PiStack', '-y')),
    ('USB export', (40, 90, 200), ('body', 'UsbExport', '-y'), ('body', 'PiStack', '-y')),
    ('POD-L', (60, 170, 80), ('light', 'J3'), ('pod_dock_l', 'J2')),
    ('POD-R', (60, 170, 80), ('light', 'J4'), ('pod_dock_r', 'J2')),
    ('HOOD', (150, 60, 190), ('light', 'J5'), ('hood_dock', 'J2')),
    ('SIDE-L', (240, 200, 40), ('light', 'J6'), ('btn_l', 'J4')),
    ('SIDE-R', (240, 200, 40), ('light', 'J7'), ('btn_r', 'J4')),
    ('FAN', (120, 120, 120), ('light', 'J9'), ('body', 'Fan30', '-y')),
    ('WHITE-L', (240, 240, 240), ('white_l', 'J4'), ('uv_l', 'J2')),
    ('WHITE-R', (240, 240, 240), ('white_r', 'J4'), ('uv_r', 'J2')),
    ('POD FLEX L', (205, 130, 40), ('uv_l', 'J1'), ('pod_dock_l', 'J1')),
    ('POD FLEX R', (205, 130, 40), ('uv_r', 'J1'), ('pod_dock_r', 'J1')),
    ('TRIG-L', (230, 60, 60), ('btn_l', 'J1'), ('body', 'Trigger_L', 'c')),
    ('TRIG-R', (230, 60, 60), ('btn_r', 'J1'), ('body', 'Trigger_R', 'c')),
    ('THUMBS-L', (60, 60, 230), ('btn_l', 'J2'), ('body', 'Thumbs_L', 'c')),
    ('THUMBS-R', (60, 60, 230), ('btn_r', 'J2'), ('body', 'Thumbs_R', 'c')),
]

BOARD_COLOUR = (20, 70, 40)          # solder-mask green for blocks of connectors with no 3D model
BLOCK_COLOUR = (40, 140, 60)


def load_generator():
    path = os.path.join(HERE, 'ayas_handheld_fusion.py')
    g = {'__name__': 'ayas_gen', '__file__': path}
    exec(compile(open(path).read(), path, 'exec'), g)
    return g


def appearance(des, name, rgb):
    ap = des.appearances.itemByName(name)
    if ap:
        return ap
    app = adsk.core.Application.get()
    base = None
    for lib in app.materialLibraries:
        if 'Appearance' not in lib.name:
            continue
        for i in range(lib.appearances.count):
            a = lib.appearances.item(i)
            if 'Plastic' in a.name and 'Glossy' in a.name:
                base = a
                break
        if base:
            break
    ap = des.appearances.addByCopy(base, name)
    for i in range(ap.appearanceProperties.count):
        pr = ap.appearanceProperties.item(i)
        if pr.objectType == adsk.core.ColorProperty.classType():
            try:
                pr.value = adsk.core.Color.create(rgb[0], rgb[1], rgb[2], 255)
            except Exception:
                pass
    return ap


def V(a, b, k=1.0):
    return tuple(a[i] + k * b[i] for i in range(3))


def run(_context: str):
    app = adsk.core.Application.get()
    gen = load_generator()
    gen['build'](export=False)
    des = adsk.fusion.Design.cast(app.activeProduct)
    root = des.rootComponent
    p = dict(gen['P'])
    p.update(gen['DISPLAYS'][p['display']])
    d = gen['layout'](p)
    frames = gen['assembly_frames'](d)
    conn = json.load(open(os.path.join(STEP_DIR, 'conn.json')))
    P3, V3 = adsk.core.Point3D.create, adsk.core.Vector3D.create

    # ---- boards
    occs = {}
    for name, step in BOARDS.items():
        o, xa, ya, za = frames[name]
        im = app.importManager
        im.importToTarget(im.createSTEPImportOptions(os.path.join(STEP_DIR, step + '.step')), root)
        occ = root.occurrences.item(root.occurrences.count - 1)
        occ.component.name = 'PCB_' + name.upper()
        m = adsk.core.Matrix3D.create()
        m.setWithCoordinateSystem(P3(*(c * MM for c in o)), V3(*xa), V3(*ya), V3(*za))
        occ.transform2 = m
        occs[name] = occ
    if des.snapshots.hasPendingSnapshot:
        des.snapshots.add()

    bodies = {}
    for occ in root.allOccurrences:
        for b in occ.bRepBodies:
            bodies.setdefault(b.name, b)
    for b in root.bRepBodies:
        bodies.setdefault(b.name, b)
    for n in ('PcbPower', 'PcbLight', 'ChargeBoard', 'FFC_keepout', 'Plate100'):
        if n in bodies:
            bodies[n].isLightBulbOn = False

    def world(name, u, v, z):
        o, xa, ya, za = frames[name]
        return tuple(o[i] + u * xa[i] - v * ya[i] + z * za[i] for i in range(3))

    def conn_info(name):
        return conn[BOARDS[name]]

    # ---- connector blocks where KiCad has no 3D model (push-in terminals, ZIFs, pogo pins, pads), per board
    tb = adsk.fusion.TemporaryBRepManager.get()
    comp = root
    U = adsk.fusion.BooleanTypes.UnionBooleanType

    def add_body(body, nm, rgb):
        bf = comp.features.baseFeatures.add()
        bf.startEdit()
        comp.bRepBodies.add(body, bf)
        bf.finishEdit()
        nb = comp.bRepBodies.item(comp.bRepBodies.count - 1)
        nb.name = nm
        nb.appearance = appearance(des, 'ayas_' + nm.replace(' ', '_'), rgb)
        return nb

    def merge(acc, part):
        if acc is None:
            return part
        tb.booleanOperation(acc, part, U)
        return acc

    def box(name, cu, cv, z0, z1, su, sv):
        o, xa, ya, za = frames[name]
        ctr = world(name, cu, cv, (z0 + z1) / 2)
        return tb.createBox(adsk.core.OrientedBoundingBox3D.create(P3(*(k * MM for k in ctr)), V3(*xa), V3(*ya),
                                                                   su * MM, sv * MM, abs(z1 - z0) * MM))

    def cyl(name, cu, cv, z0, z1, r):
        return tb.createCylinderOrCone(P3(*(k * MM for k in world(name, cu, cv, z0))), r * MM,
                                       P3(*(k * MM for k in world(name, cu, cv, z1))), r * MM)

    for name in BOARDS:
        blocks = None
        for ref, c in conn_info(name).items():
            if not c['miss']:
                continue
            h = 0.3 if name == 'hood_pad' and ref == 'J1' else (5.0 if name == 'hood_dock' else 4.0)
            z0, z1 = (1.6, 1.6 + h) if c['side'] == 'F' else (-h, 0.0)
            blocks = merge(blocks, box(name, c['c'][0], c['c'][1], z0, z1, c['sz'][0], c['sz'][1]))
        if blocks:
            add_body(blocks, 'Blocks ' + name, BLOCK_COLOUR)

    # ---- the owned white modules (40 x 10 x 6 mm, 3 LEDs, screw tabs) on the carriers, and the 5 mm UV LEDs
    hp, mh = d['white_hole_pitch'], d['white_mod_h']
    for name in ('white_l', 'white_r'):
        W = 50.0
        cols = [W / 2 + (i - 1) * 14.0 for i in range(3)]
        if name == 'white_r':
            cols = [W - c for c in cols]
        mods, leds = None, None
        for cu in cols:
            vc = 15.0 + hp / 2
            mods = merge(mods, box(name, cu, vc, 1.6, 1.6 + mh, 10.0, 40.0))
            for tv in (15.0, 15.0 + hp):
                mods = merge(mods, box(name, cu, tv, 1.6, 3.6, 6.0, 6.0))
            for k in (-1, 0, 1):
                leds = merge(leds, box(name, cu, vc + k * d['white_led_pitch'], 1.6 + mh, 1.6 + mh + 0.6, 3.5, 2.8))
        add_body(mods, 'Modules ' + name, (232, 232, 226))
        add_body(leds, 'WhiteLEDs ' + name, (250, 220, 90))
    for name in ('uv_l', 'uv_r'):
        uv = None
        for c in range(3):
            for r in range(3):
                cu, cv = 5.5 + 16.0 * c, 5.09 + 10.16 * r
                uv = merge(uv, cyl(name, cu, cv, 1.6 + d['uv_sock_h'], 1.6 + d['uv_sock_h'] + 6.2, 2.5))
                uv = merge(uv, tb.createSphere(P3(*(k * MM for k in world(name, cu, cv, 1.6 + d['uv_sock_h'] + 6.2))), 2.5 * MM))
        add_body(uv, 'UvLEDs ' + name, (120, 50, 210))
    for n in ('Leds_L', 'Leds_R'):
        if n in bodies:
            bodies[n].isLightBulbOn = False

    # ---- cable harness
    def end_point(e):
        if e[0] == 'body':
            bb = bodies[e[1]].boundingBox
            lo, hi = [bb.minPoint.x / MM, bb.minPoint.y / MM, bb.minPoint.z / MM], [bb.maxPoint.x / MM, bb.maxPoint.y / MM, bb.maxPoint.z / MM]
            c = [(lo[i] + hi[i]) / 2 for i in range(3)]
            face = e[2]
            if face != 'c':
                ax = 'xyz'.index(face[1])
                c[ax] = lo[ax] if face[0] == '-' else hi[ax]
            n = [0, 0, 0]
            if face != 'c':
                n['xyz'.index(face[1])] = -1 if face[0] == '-' else 1
            return tuple(c), tuple(n)
        name, ref = e
        c = conn_info(name)[ref]
        za = frames[name][3]
        z = 1.6 + 2.0 if c['side'] == 'F' else -2.0
        n = za if c['side'] == 'F' else tuple(-k for k in za)
        return world(name, c['c'][0], c['c'][1], z), n

    for (nm, rgb, ea, eb) in CABLES:
        (a, na), (b, nb_) = end_point(ea), end_point(eb)
        a1, b1 = V(a, na, 6.0), V(b, nb_, 6.0)
        mid = tuple((a1[i] + b1[i]) / 2 for i in range(3))
        pts = [a, a1, mid, b1, b]
        r = 1.4 if 'FLEX' not in nm else 1.0
        tube = None
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            if math.dist(p0, p1) < 0.2:
                continue
            cyl = tb.createCylinderOrCone(P3(*(k * MM for k in p0)), r * MM, P3(*(k * MM for k in p1)), r * MM)
            sph = tb.createSphere(P3(*(k * MM for k in p1)), r * MM)
            for part in (cyl, sph):
                if tube is None:
                    tube = part
                else:
                    tb.booleanOperation(tube, part, adsk.fusion.BooleanTypes.UnionBooleanType)
        add_body(tube, 'Cable ' + nm, rgb)

    # ---- colours for the reference parts
    paint = {'Trigger_L': (200, 40, 30), 'Trigger_R': (200, 40, 30), 'Thumbs_L': (40, 40, 45), 'Thumbs_R': (40, 40, 45),
             'PwrCap': (200, 40, 30), 'PowerBank': (35, 35, 38), 'PiStack': (30, 110, 60), 'Display': (15, 15, 18),
             'Fan30': (60, 60, 62), 'UsbExport': (150, 150, 155), 'Camera': (30, 90, 50), 'Leds_L': (235, 235, 225),
             'Leds_R': (235, 235, 225), 'RingLeds': (235, 235, 225), 'PwrButton': (150, 150, 155),
             # rugged housing: graphite shell, black rubber grips, orange accents (the demo-video palette)
             'FrontCentre': (46, 48, 52), 'BackLid': (40, 42, 46), 'BayDoor': (40, 42, 46),
             'WingFront_L': (28, 28, 30), 'WingFront_R': (28, 28, 30), 'WingBack_L': (28, 28, 30), 'WingBack_R': (28, 28, 30),
             'LedPod_L': (58, 60, 64), 'LedPod_R': (58, 60, 64), 'CamHood': (204, 70, 10), 'InspectionHood': (52, 54, 58)}
    for n, rgb in paint.items():
        if n in bodies:
            bodies[n].appearance = appearance(des, 'ayas_' + n, rgb)
    print('assembly done:', len(occs), 'boards,', len(CABLES), 'cables')


def render(out_dir, views):
    """views: {name: {'eye': (x,y,z) mm, 'target': (x,y,z), 'hide': [...], 'show': [...], 'opacity': {...}}}"""
    app = adsk.core.Application.get()
    des = adsk.fusion.Design.cast(app.activeProduct)
    root = des.rootComponent
    bodies = {}
    for occ in root.allOccurrences:
        for b in occ.bRepBodies:
            bodies.setdefault(b.name, b)
    for b in root.bRepBodies:
        bodies.setdefault(b.name, b)
    vp = app.activeViewport
    vp.visualStyle = adsk.core.VisualStyles.ShadedVisualStyle
    try:                                               # no layout grid in the renders
        app.executeTextCommand('NaViewCube.ShowGrid false')
    except Exception:
        pass
    for name, v in views.items():
        state = {}
        for n in v.get('hide', []):
            if n in bodies:
                state[n] = bodies[n].isLightBulbOn
                bodies[n].isLightBulbOn = False
        op = {}
        for n, a in v.get('opacity', {}).items():
            if n in bodies:
                op[n] = bodies[n].opacity
                bodies[n].opacity = a
        for occ in root.occurrences:
            occ.isLightBulbOn = not any(occ.component.name.startswith(h) for h in v.get('hide_occ', []))
        cam = vp.camera
        cam.isFitView = False
        cam.eye = adsk.core.Point3D.create(*(k * MM for k in v['eye']))
        cam.target = adsk.core.Point3D.create(*(k * MM for k in v.get('target', (0, 0, 0))))
        cam.upVector = adsk.core.Vector3D.create(0, 1, 0)
        cam.cameraType = adsk.core.CameraTypes.PerspectiveCameraType
        vp.camera = cam
        vp.fit()
        vp.refresh()
        adsk.doEvents()
        opts = adsk.core.SaveImageFileOptions.create(os.path.join(out_dir, name + '.png'))
        opts.width, opts.height = 1800, 1200
        opts.isBackgroundTransparent = True
        opts.isAntiAliased = True
        vp.saveAsImageFileWithOptions(opts)
        for n, s in state.items():
            bodies[n].isLightBulbOn = s
        for n, a in op.items():
            bodies[n].opacity = a
        for occ in root.occurrences:
            occ.isLightBulbOn = True
