"""Build a KiCad PCB (.kicad_pcb) for a board in design.py, inside KiCad's Python (pcbnew).

Outline and mounting holes match the Fusion enclosure fit (hardware/enclosure/ayas_handheld_fusion.py:
pcb_main_x, pcb_main_z, pcb_hole_in, pcb_low). Placement: fixed positions for connectors, ICs and holes
(PLACE below), then every other part is clustered round the part it shares the most nets with. Nets are
assigned to pads from design.py, so the board and schematic agree by construction.

usage: <KiCad python> hardware/pcb/gen_pcb.py power|light [out_dir]
"""
from __future__ import annotations

import math
import os
import sys

import pcbnew

sys.path.insert(0, os.path.dirname(__file__))
import design  # noqa: E402

FP_DIR = os.environ.get('KICAD_FOOTPRINT_DIR', '/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints')
W, H = 46.0, 55.0                       # current board size (set per board by GEOM)
HOLE_IN = 3.5
MAIN_HOLES = [(HOLE_IN, HOLE_IN), (46.0 - HOLE_IN, HOLE_IN), (HOLE_IN, 55.0 - HOLE_IN), (46.0 - HOLE_IN, 55.0 - HOLE_IN)]
# per-board outline: size, corner radius, copper layers, hole centres, low-profile strip
GEOM = {
    'power': dict(W=46.0, H=55.0, r=2.0, layers=2, holes=MAIN_HOLES, low=True),
    'light': dict(W=46.0, H=55.0, r=2.0, layers=6, holes=MAIN_HOLES, low=True, fine=True,
                  b_keep=(15.0, 22.0, 27.0, 34.0)),   # no auto-placed underside parts round the RP2040 (fan-out vias)   # 6-layer: In1 GND plane, 5 signal layers
    # UV socket board in the pod's UV cavity: u = x' + 21.5, v = -y' - 0.4 (UV sub-frame of the pod generator)
    'uv': dict(W=43.0, H=30.5, r=1.0, layers=2, holes=[(13.5, 10.17), (29.5, 20.33)], low=False),
    # charge-port board: in the bay corner, receptacle mouth on the v = 0 edge facing the side wall
    'charge': dict(W=24.0, H=12.5, r=1.0, layers=2, holes=[(21.5, 6.25)], low=False),   # u along z, v along -x from the wall
    'podflex': dict(W=11.0, H=60.0, r=0.5, layers=2, holes=[], low=False, flex=True),
    # WHITE-L / WHITE-R carriers: 3 x 40 x 10 mm modules (columns at u 11/25/39), push-in terminals on top,
    # module tab nuts on the back; 4 x M2 into the pod. v = 0 at the top of the pod.
    # 2 x M2 at the OUTER edge only; the inner edge (2.5 mm, no parts) slides under a lip in the pod (thin there).
    # The R board is the mirror image, so its outer edge is u = 0.
    'white_l': dict(W=50.0, H=66.0, r=1.5, layers=2, holes=[(47.0, 3.5), (47.0, 62.5)], low=False),
    'white_r': dict(W=50.0, H=66.0, r=1.5, layers=2, holes=[(3.0, 3.5), (3.0, 62.5)], low=False),
    # docking pairs: pads (module side) and pogo pins (body side), each 2 x M2
    'pod_dock_l': dict(W=28.0, H=14.0, r=1.0, layers=2, holes=[(2.6, 7.0), (25.4, 7.0)], low=False),
    'pod_dock_r': dict(W=28.0, H=14.0, r=1.0, layers=2, holes=[(2.6, 7.0), (25.4, 7.0)], low=False),
    # button boards in the wings: 3 push-in terminals (TRIG, UPPER, LOWER) on top, cable to LIGHT/IO on the back
    'btn_l': dict(W=36.0, H=20.0, r=1.0, layers=2, holes=[(2.8, 7.0), (33.2, 7.0)], low=False),
    'btn_r': dict(W=36.0, H=20.0, r=1.0, layers=2, holes=[(2.8, 7.0), (33.2, 7.0)], low=False),
    'hood_pad': dict(W=40.0, H=32.0, r=1.0, layers=2, holes=[(2.8, 4.0), (37.2, 4.0)], low=False),
    'hood_dock': dict(W=34.0, H=13.0, r=1.0, layers=2, holes=[(2.6, 6.5), (31.4, 6.5)], low=False),
}
OX, OY = 100.0, 100.0                   # board origin on the KiCad sheet
# Board frame (u, v): u = KiCad +X, v = KiCad +Y (down the sheet); components on F.Cu face +y in the enclosure
# (up, away from the bank). KiCad's 3D/STEP Y is -v, so the right-handed mapping to the enclosure model is:
#   POWER (x > 0): u = 0 at the inner edge (x = 21), +u -> +x;  v = 0 at the BACK (z = -32), FRONT at v = H (z = 23)
#   LIGHT (x < 0): u = 0 at the inner edge (x = -21), +u -> -x; v = 0 at the FRONT (z = 23), BACK at v = H (z = -32)
# Low-profile strip (parts <= 5 mm) under the display posts, on both boards:
LOW = (31.0, 40.0)                      # u range on both boards (|x| 52..61 minus 21)

# fixed placements: ref -> (u, v, rotation deg, side). Everything else is clustered.
PLACE = {
    'power': {
        # inner edge (u = 0): bank USB-C + PD sink + CC router + power path; back edge: always-on regulators;
        # outer edge: LINK + power button; front edge: charge port + Pi 5 V. Nothing tall in the low strip.
        'J1': (3.5, 27.5, 270, 'F'), 'U1': (13.0, 27.5, 0, 'F'), 'U5': (8.5, 17.5, 0, 'F'),
        'Q1': (12.0, 36.5, 0, 'F'), 'F1': (17.5, 37.0, 90, 'F'), 'D1': (6.0, 39.0, 90, 'F'),
        'U4': (12.0, 8.5, 0, 'F'), 'U6': (22.5, 8.5, 0, 'F'),
        'R8': (27.0, 15.5, 0, 'F'), 'U3': (27.0, 21.5, 0, 'F'),
        'U2': (25.0, 30.0, 0, 'F'), 'L1': (26.0, 41.0, 0, 'F'),
        'J2': (43.0, 23.5, 270, 'F'), 'J5': (43.0, 42.8, 270, 'F'),
        'J3': (13.0, 49.0, 0, 'F'), 'J6': (25.5, 51.6, 0, 'F'),
    },
    'light': {                           # FRONT edge is v = 0 on this board (see the frame note)
        # JST-SH side-entry: rot 0 opens to +v (back), 180 to -v (front), 90 to +u (outer), 270 to -u (inner)
        'J3': (12.0, 3.4, 180, 'F'), 'J4': (24.0, 3.4, 180, 'F'),                 # front: POD-L, POD-R docks
        'J5': (3.4, 29.5, 270, 'F'),                                               # inner edge: HOOD dock
        'J6': (3.4, 13.0, 270, 'F'), 'J7': (3.4, 43.0, 270, 'F'),                  # inner edge: SIDE-L, SIDE-R wings
        'J1': (42.8, 35.0, 270, 'F'), 'J9': (42.6, 11.5, 90, 'F'),               # outer edge: LINK, FAN
        'J2': (13.0, 51.6, 0, 'F'),                                                 # back edge: USB to the Pi (PH-4)
        'U2': (21.0, 28.0, 270, 'F'), 'U3': (31.5, 28.0, 0, 'F'), 'Y1': (12.0, 29.0, 0, 'F'), 'U1': (35.0, 45.0, 0, 'F'),   # RP2040 GPIO0-11 face the sinks
        'SW1': (11.0, 45.5, 0, 'F'), 'SW2': (17.0, 45.5, 0, 'F'),
        'U4': (18.0, 12.0, 0, 'F'), 'U5': (30.5, 17.5, 90, 'F'), 'U6': (35.0, 38.0, 90, 'F'),
        'C39': (25.5, 43.5, 0, 'F'),      # 7.7 mm electrolytic: top side, outside the low strip
        # RP2040 support, per the hardware design guide: crystal beside XIN/XOUT, decoupling on the back
        # right under the pins it serves, flash pull-up and decoupler at the flash, USB series R at the USB pins
        'Y1': (14.0, 28.5, 0, 'F'),
        'C12': (18.7, 28.6, 90, 'B'), 'C13': (23.3, 28.6, 90, 'B'),             # 1V1, 0402, pins 23 and 45/50
        'C11': (25.8, 30.0, 90, 'B'), 'C10': (25.8, 33.7, 90, 'B'),             # VREG out (1V1) and in (3V3)
        'R2': (16.6, 26.6, 0, 'B'), 'C16': (13.8, 24.6, 0, 'B'), 'C15': (12.4, 31.0, 0, 'B'),   # crystal
        'R3': (16.0, 31.8, 0, 'B'),                                               # RUN pull-up
        'R1': (28.4, 23.8, 0, 'B'), 'C14': (35.1, 23.6, 0, 'B'),                # flash: SS pull-up, decoupler
        'R5': (28.8, 31.6, 0, 'B'), 'R6': (28.8, 33.4, 0, 'B'),                 # USB 27R at pins 46/47
        'BZ1': (12.0, 38.6, 0, 'F'),                                              # buzzer, below the crystal
    },
}


def white_cols(name):
    m = design.WHITE_MOD
    W_ = GEOM[name]['W']
    cols = [W_ / 2 + (i - 1) * m['pitch'] for i in range(3)]
    return cols if name == 'white_l' else [W_ - c for c in cols]


WHITE_TOP = 15.0                             # v of the upper tab holes (the terminals sit above)


def white_place(name):
    m = design.WHITE_MOD
    pl = {}
    for i, cx in enumerate(white_cols(name)):
        pl['J%d' % (i + 1)] = (cx, 6.2, 0, 'F')                                   # push-in terminal above module i
        pl['N%d' % (2 * i + 1)] = (cx, WHITE_TOP, 0, 'B')                           # tab nuts on the back
        pl['N%d' % (2 * i + 2)] = (cx, WHITE_TOP + m['hole_pitch'], 0, 'B')
    c = white_cols(name)
    pl['J4'] = ((c[1] + c[2]) / 2, 34.0, 0, 'B')     # side entry on the back, outer gap (deeper pod), opening down to the hub
    return pl


def place_for(name):
    if name == 'uv':                       # each socket centred on its LED hole; rot 90 puts pin 1 (anode, +) on the left
        pl = {}
        n = 1
        for c in range(design.UV_COLS):
            for r in range(design.UV_ROWS):
                pl['X%d' % n] = (5.5 + 16.0 * c, 5.09 + 10.16 * r, 90, 'F')
                n += 1
        pl['J1'] = (29.5, 10.0, 90, 'B')  # ZIF on the back, in the clear band between the middle and right columns
        pl['J2'] = (13.5, 24.0, 90, 'B')   # WHITE carrier cable
        return pl
    if name == 'charge':
        return {'J1': (6.0, 4.2, 180, 'F'), 'J2': (15.5, 6.25, 90, 'F'), 'D1': (6.0, 10.6, 0, 'F')}
    if name.startswith('white_'):
        return white_place(name)
    if name.startswith('btn_'):
        return {'J1': (9.5, 7.6, 0, 'F'), 'J2': (18.0, 7.6, 0, 'F'), 'J3': (26.5, 7.6, 0, 'F'), 'J4': (18.0, 16.6, 0, 'B')}
    if name.startswith('pod_dock'):         # ZIF on the outer face (opening up, toward the pod hub); cable to LIGHT behind
        return {'J1': (14.0, 3.2, 180, 'F'), 'J2': (14.0, 9.8, 180, 'B')}
    if name == 'hood_pad':
        pl = {'J1': (20.0, 3.4, 0, 'F'), 'R1': (34.0, 3.4, 0, 'B')}
        for i in range(6):                     # ring module leads, push-in, on the back: two rows of three
            pl['J%d' % (2 + i)] = (8.0 + 12.0 * (i % 3), 13.0 + 12.0 * (i // 3), 0, 'B')
        return pl
    if name == 'hood_dock':
        return {'J1': (17.0, 3.4, 0, 'F'), 'J2': (17.0, 9.4, 180, 'B')}
    if name.startswith('uvflex') or name == 'podflex':
        L = design.BOARDS[name].length
        w = GEOM[name]['W'] / 2
        return {'P1': (w, 2.6, 0, 'F'), 'P2': (w, L - 2.6, 180, 'F')}
    return PLACE[name]


def mm(v):
    return pcbnew.FromMM(v)


def to_xy(u, v):
    return pcbnew.VECTOR2I(mm(OX + u), mm(OY + v))


def load_fp(fp_id):
    lib, name = fp_id.split(':')
    base = os.path.dirname(os.path.abspath(__file__)) if lib == 'ayas' else FP_DIR
    fp = pcbnew.FootprintLoad(os.path.join(base, lib + '.pretty'), name)
    if fp is None:
        raise KeyError('footprint %s not found' % fp_id)
    return fp


def outline(board, r=2.0, low=True):
    pts = [(r, 0), (W - r, 0), (W, r), (W, H - r), (W - r, H), (r, H), (0, H - r), (0, r)]
    for i in range(0, 8, 2):
        a, b = pts[i], pts[(i + 1) % 8]
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(to_xy(*a)); seg.SetEnd(to_xy(*b))
        seg.SetLayer(pcbnew.Edge_Cuts); seg.SetWidth(mm(0.1))
        board.Add(seg)
    for (cu, cv, sa) in ((r, r, 180), (W - r, r, 270), (W - r, H - r, 0), (r, H - r, 90)):
        arc = pcbnew.PCB_SHAPE(board)
        arc.SetShape(pcbnew.SHAPE_T_ARC)
        arc.SetCenter(to_xy(cu, cv))
        a0 = math.radians(sa)
        arc.SetStart(to_xy(cu + r * math.cos(a0), cv + r * math.sin(a0)))
        arc.SetArcAngleAndEnd(pcbnew.EDA_ANGLE(90, pcbnew.DEGREES_T), True)
        arc.SetLayer(pcbnew.Edge_Cuts); arc.SetWidth(mm(0.1))
        board.Add(arc)
    # keep-out strips along the edges: Freerouting does not know the copper-to-edge rule
    k = 0.6
    for (u0, v0, u1, v1) in ((0, 0, W, k), (0, H - k, W, H), (0, 0, k, H), (W - k, 0, W, H)):
        ra = pcbnew.ZONE(board)
        ra.SetIsRuleArea(True)
        ra.SetDoNotAllowTracks(True); ra.SetDoNotAllowVias(True)
        ra.SetDoNotAllowZoneFills(False); ra.SetDoNotAllowPads(False); ra.SetDoNotAllowFootprints(False)
        ra.SetLayerSet(pcbnew.LSET.AllCuMask())
        ol = ra.Outline(); ol.NewOutline()
        for (uu, vv) in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
            ol.Append(mm(OX + uu), mm(OY + vv))
        board.Add(ra)
    # low-profile strip marker on User.1 (fab note)
    for u in (LOW if low else ()):
        ln = pcbnew.PCB_SHAPE(board)
        ln.SetShape(pcbnew.SHAPE_T_SEGMENT)
        ln.SetStart(to_xy(u, 0)); ln.SetEnd(to_xy(u, H))
        ln.SetLayer(pcbnew.User_1); ln.SetWidth(mm(0.15))
        board.Add(ln)


def bbox_mm(fp):
    bb = fp.GetCourtyard(pcbnew.F_CrtYd).BBox() if fp.GetCourtyard(pcbnew.F_CrtYd).OutlineCount() else fp.GetBoundingBox(False)
    return (pcbnew.ToMM(bb.GetX()) - OX, pcbnew.ToMM(bb.GetY()) - OY, pcbnew.ToMM(bb.GetWidth()), pcbnew.ToMM(bb.GetHeight()))


def overlaps(a, b, gap=0.3):
    return not (a[0] + a[2] + gap <= b[0] or b[0] + b[2] + gap <= a[0] or a[1] + a[3] + gap <= b[1] or b[1] + b[3] + gap <= a[1])


def build(board_name, out_dir):
    global W, H
    geo = dict(GEOM[board_name])
    if geo.get('flex'):
        geo['H'] = design.BOARDS[board_name].length
    W, H = geo['W'], geo['H']
    parts = design.BOARDS[board_name]()
    board = pcbnew.BOARD()
    ds = board.GetDesignSettings()
    ds.SetCopperLayerCount(geo['layers'])
    board.SetCopperLayerCount(geo['layers'])
    outline(board, geo['r'], geo['low'])
    nc = ds.m_NetSettings.GetDefaultNetclass() if hasattr(ds.m_NetSettings, 'GetDefaultNetclass') else None
    fine = geo.get('fine', False)                     # 6-layer LIGHT/IO: 0.1 mm rules for the QFN fan-out (gen_pro FINE)
    ds.m_MinClearance = mm(0.1 if fine else 0.15); ds.m_HoleClearance = mm(0.15 if fine else 0.18)
    ds.m_CopperEdgeClearance = mm(0.3)
    ds.m_TrackMinWidth = mm(0.1 if fine else 0.15)
    if nc is not None:
        nc.SetClearance(mm(0.1 if fine else 0.15)); nc.SetTrackWidth(mm(0.15 if fine else 0.25))
        if fine:
            nc.SetViaDiameter(mm(0.45)); nc.SetViaDrill(mm(0.2))
    nets = {}

    def net(name):
        if name not in nets:
            ni = pcbnew.NETINFO_ITEM(board, name)
            board.Add(ni)
            nets[name] = ni
        return nets[name]

    fps = {}
    for p in parts:
        fp = load_fp(p.footprint)
        fp.SetReference(p.ref)
        fp.SetValue(p.value)
        fp.SetFPIDAsString(p.footprint)
        board.Add(fp)
        fps[p.ref] = (fp, p)
    # pad nets: design pins are symbol pins; footprint pad numbers follow the symbol pin numbers
    import gen_sch
    for ref, (fp, p) in fps.items():
        sym = gen_sch.flat_symbol(p.symbol)
        pins = gen_sch.pins_of(sym)
        m = gen_sch.resolve(p, pins)
        for (unit, number), n in m.items():
            hit = False
            for pad in fp.Pads():
                if pad.GetNumber() == number:
                    pad.SetNet(net(n)); hit = True
            if not hit and number not in ('EP',):
                print('WARN %s pin %s (%s) has no pad' % (ref, number, n))
        if 'EP' in p.pins or any(k == 'EP' for k in p.pins):
            for pad in fp.Pads():
                if pad.GetNumber() in ('', '57', '25', '29', 'EP') and not pad.GetNet().GetNetname():
                    pad.SetNet(net('GND'))
    # fixed placements
    placed = []
    for ref, (u, v, rot, side) in place_for(board_name).items():
        fp = fps[ref][0]
        fp.SetPosition(to_xy(u, v)); fp.SetOrientationDegrees(rot)
        if side == 'B':
            fp.Flip(fp.GetPosition(), False)
        bx, by, bw, bh = bbox_mm(fp)                                   # (u, v) is the body centre
        fp.Move(pcbnew.VECTOR2I(mm(u - (bx + bw / 2)), mm(v - (by + bh / 2))))
        placed.append(ref)
    holes = [r for r in fps if r.startswith('H')]
    for r, (u, v) in zip(sorted(holes), geo['holes']):
        fps[r][0].SetPosition(to_xy(u, v)); placed.append(r)
    boxes = [bbox_mm(fps[r][0]) for r in placed]
    thru = [bbox_mm(fps[r][0]) for r in placed if any(pd.GetAttribute() == pcbnew.PAD_ATTRIB_PTH for pd in fps[r][0].Pads())]
    bottom = list(thru) + [bbox_mm(fps[r][0]) for r in placed if fps[r][0].IsFlipped()]
    # cluster the rest round the placed part they share most nets with
    def netset(ref):
        return {pd.GetNetname() for pd in fps[ref][0].Pads()} - {'', 'GND', 'V5', '3V3', 'V12', 'V12_LED', '1V1'}
    rest = [r for r in fps if r not in placed]
    rest.sort(key=lambda r: (-len(fps[r][0].Pads()), r))
    for r in rest:
        ns = netset(r)
        anchors = sorted(placed, key=lambda a: -len(ns & netset(a)))
        anchor = anchors[0] if anchors and ns & netset(anchors[0]) else None
        if anchor:
            ax, ay = pcbnew.ToMM(fps[anchor][0].GetPosition().x) - OX, pcbnew.ToMM(fps[anchor][0].GetPosition().y) - OY
        else:
            ax, ay = W / 2, H / 2
        fp = fps[r][0]
        ok = False
        for k in range(1, 400):
            ang, rad = k * 2.39996, 1.2 * math.sqrt(k)            # golden-angle spiral
            u, v = ax + rad * math.cos(ang), ay + rad * math.sin(ang)
            fp.SetPosition(to_xy(u, v))
            b = bbox_mm(fp)
            if b[0] < 0.5 or b[1] < 0.5 or b[0] + b[2] > W - 0.5 or b[1] + b[3] > H - 0.5:
                continue
            if any(overlaps(b, o) for o in boxes):
                continue
            if geo['low'] and r.startswith('J') and b[0] < LOW[1] and b[0] + b[2] > LOW[0]:
                continue                                            # connectors are too tall for the low strip
            boxes.append(b); placed.append(r); ok = True
            break
        if not ok and len(fp.Pads()) <= 3:                          # small passives may go on the underside
            fp.Flip(fp.GetPosition(), False)
            for k in range(1, 600):
                ang, rad = k * 2.39996, 1.2 * math.sqrt(k)
                u, v = ax + rad * math.cos(ang), ay + rad * math.sin(ang)
                fp.SetPosition(to_xy(u, v))
                b = bbox_mm(fp)
                if b[0] < 0.5 or b[1] < 0.5 or b[0] + b[2] > W - 0.5 or b[1] + b[3] > H - 0.5:
                    continue
                if any(overlaps(b, o) for o in bottom):
                    continue
                if any(overlaps(b, o, 0.0) for o in thru):                # keep clear of through-hole parts
                    continue
                if geo.get('b_keep'):
                    k0, k1, k2, k3 = geo['b_keep']
                    if overlaps(b, (k0, k1, k2 - k0, k3 - k1), 0.0):
                        continue
                bottom.append(b); placed.append(r); ok = True
                break
        if not ok:
            print('WARN could not place', r)
    for n in nets.values():
        pass
    if board_name == 'uv':                             # '+' (anode, square pad) and '-' beside every LED socket
        for ref, (fp, part) in fps.items():
            if not ref.startswith('X'):
                continue
            for pad in fp.Pads():
                sign = '+' if pad.GetNumber() == '1' else '-'
                pos = pad.GetPosition()
                dx = -1 if pad.GetPosition().x < fp.GetPosition().x or (pad.GetNumber() == '1') else 1
                t = pcbnew.PCB_TEXT(board)
                t.SetText(sign); t.SetLayer(pcbnew.F_SilkS)
                t.SetTextSize(pcbnew.VECTOR2I(mm(1.4), mm(1.4))); t.SetTextThickness(mm(0.25))
                t.SetPosition(pcbnew.VECTOR2I(pos.x, pos.y - mm(2.2)))
                board.Add(t)
    if board_name.startswith('white_'):                # module outlines, M1-M3, + / - at every terminal, board name
        m = design.WHITE_MOD

        def text(s_, u, v, size=1.2, layer=pcbnew.F_SilkS):
            t = pcbnew.PCB_TEXT(board)
            t.SetText(s_); t.SetLayer(layer)
            t.SetTextSize(pcbnew.VECTOR2I(mm(size), mm(size))); t.SetTextThickness(mm(0.2))
            t.SetPosition(to_xy(u, v)); board.Add(t)

        def rect(u0, v0, u1, v1):
            for a_, b_ in (((u0, v0), (u1, v0)), ((u1, v0), (u1, v1)), ((u1, v1), (u0, v1)), ((u0, v1), (u0, v0))):
                sg = pcbnew.PCB_SHAPE(board)
                sg.SetShape(pcbnew.SHAPE_T_SEGMENT); sg.SetLayer(pcbnew.F_SilkS); sg.SetWidth(mm(0.15))
                sg.SetStart(to_xy(*a_)); sg.SetEnd(to_xy(*b_)); board.Add(sg)

        vc = WHITE_TOP + m['hole_pitch'] / 2
        for i, cx in enumerate(white_cols(board_name)):
            rect(cx - m['W'] / 2, vc - m['L'] / 2, cx + m['W'] / 2, vc + m['L'] / 2)
            text('M%d' % (i + 1), cx, vc)
            pad1 = [pd for pd in fps['J%d' % (i + 1)][0].Pads() if pd.GetNumber() == '1'][0]
            pu = pcbnew.ToMM(pad1.GetPosition().x) - OX
            text('+', pu, 1.2); text('-', pu + 2.5, 1.2)
        text('WHITE-%s' % board_name[-1].upper(), W / 2, H - 2.0, 1.5)
        text('WHITE-%s  J4 to pod hub' % board_name[-1].upper(), W / 2, H - 2.0, 1.2, pcbnew.B_SilkS)
    if board_name.startswith('btn_'):                  # function under every terminal, board name on both faces
        for i, lab in enumerate(design.BTN_LABEL):
            t = pcbnew.PCB_TEXT(board); t.SetText(lab); t.SetLayer(pcbnew.F_SilkS)
            t.SetTextSize(pcbnew.VECTOR2I(mm(1.0), mm(1.0))); t.SetTextThickness(mm(0.18))
            t.SetPosition(to_xy(9.5 + 8.5 * i, 14.8)); board.Add(t)
        for ly, txt in ((pcbnew.F_SilkS, 'SIDE-%s' % board_name[-1].upper()),
                        (pcbnew.B_SilkS, 'SIDE-%s  J4 to LIGHT/IO' % board_name[-1].upper())):
            t = pcbnew.PCB_TEXT(board); t.SetText(txt); t.SetLayer(ly)
            t.SetTextSize(pcbnew.VECTOR2I(mm(1.0), mm(1.0))); t.SetTextThickness(mm(0.18))
            t.SetPosition(to_xy(W / 2, 18.6 if ly == pcbnew.F_SilkS else 1.4)); board.Add(t)
    board.BuildConnectivity()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'ayas_%s.kicad_pcb' % board_name)
    pcbnew.SaveBoard(path, board)
    print('wrote', path, len(fps), 'footprints', len(nets), 'nets')
    return path


if __name__ == '__main__':
    b = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), 'kicad', b)
    build(b, out)
