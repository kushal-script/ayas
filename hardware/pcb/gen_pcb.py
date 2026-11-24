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
W, H = 46.0, 55.0                       # board size (enclosure fit: x 21..67, z -32..23)
HOLE_IN = 3.5
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
        'J1': (3.5, 27.5, 270, 'F'),    # USB-C on the inner edge, facing the bank cable slot
        'U1': (13.0, 22.0, 0, 'F'), 'Q1': (11.0, 13.0, 0, 'F'), 'F1': (19.0, 11.0, 0, 'F'), 'D1': (11.0, 37.5, 90, 'F'),
        'U2': (24.0, 31.0, 0, 'F'), 'L1': (24.0, 42.0, 0, 'F'),
        'R8': (27.5, 14.0, 0, 'F'), 'U3': (24.0, 21.0, 0, 'F'),
        'J2': (43.0, 27.5, 270, 'F'),    # link to LIGHT/IO on the outer edge (PH is 6 mm tall: outside the low strip)
        'J3': (13.0, 47.0, 0, 'F'),      # 5 V out to the Pi (VH, 9.8 mm tall)
    },
    'light': {                           # FRONT edge is v = 0 on this board (see the frame note)
        # JST-SH side-entry: rot 0 opens to +v (back), 180 to -v (front), 90 to +u (outer), 270 to -u (inner)
        'J4': (11.0, 3.4, 180, 'F'), 'J3': (21.0, 3.4, 180, 'F'), 'J5': (32.5, 3.4, 180, 'F'),   # front: pods, hood ring
        'J8': (3.4, 41.0, 270, 'F'),                                                         # inner edge: thumbs
        'J1': (42.8, 33.0, 270, 'F'), 'J10': (42.6, 18.5, 90, 'F'), 'J9': (42.6, 11.5, 90, 'F'),   # outer edge
        'J2': (13.0, 51.6, 0, 'F'), 'J6': (21.0, 51.6, 0, 'F'), 'J7': (28.0, 51.6, 0, 'F'),     # back edge
        'U2': (20.0, 29.0, 0, 'F'), 'U3': (11.0, 22.0, 90, 'F'), 'Y1': (27.0, 38.0, 0, 'F'), 'U1': (35.0, 45.0, 0, 'F'),
        'SW1': (11.0, 45.5, 0, 'F'), 'SW2': (17.0, 45.5, 0, 'F'),
        'U4': (18.0, 12.0, 0, 'F'), 'U5': (30.5, 19.0, 90, 'F'), 'U6': (34.5, 34.0, 90, 'F'),
        'C39': (8.0, 11.0, 0, 'F'),      # 7.7 mm electrolytic: top side, outside the low strip
    },
}


def mm(v):
    return pcbnew.FromMM(v)


def to_xy(u, v):
    return pcbnew.VECTOR2I(mm(OX + u), mm(OY + v))


def load_fp(fp_id):
    lib, name = fp_id.split(':')
    fp = pcbnew.FootprintLoad(os.path.join(FP_DIR, lib + '.pretty'), name)
    if fp is None:
        raise KeyError('footprint %s not found' % fp_id)
    return fp


def outline(board):
    r = 2.0
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
    # low-profile strip marker on User.1 (fab note)
    for u in LOW:
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
    parts = design.BOARDS[board_name]()
    board = pcbnew.BOARD()
    ds = board.GetDesignSettings()
    ds.SetCopperLayerCount(4 if board_name == 'light' else 2)
    board.SetCopperLayerCount(4 if board_name == 'light' else 2)
    outline(board)
    nc = ds.m_NetSettings.GetDefaultNetclass() if hasattr(ds.m_NetSettings, 'GetDefaultNetclass') else None
    ds.m_MinClearance = mm(0.15); ds.m_HoleClearance = mm(0.18); ds.m_CopperEdgeClearance = mm(0.3)
    ds.m_TrackMinWidth = mm(0.15)
    if nc is not None:
        nc.SetClearance(mm(0.15)); nc.SetTrackWidth(mm(0.25))
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
    for ref, (u, v, rot, side) in PLACE[board_name].items():
        fp = fps[ref][0]
        fp.SetPosition(to_xy(u, v)); fp.SetOrientationDegrees(rot)
        if side == 'B':
            fp.Flip(fp.GetPosition(), False)
        bx, by, bw, bh = bbox_mm(fp)                                   # (u, v) is the body centre
        fp.Move(pcbnew.VECTOR2I(mm(u - (bx + bw / 2)), mm(v - (by + bh / 2))))
        placed.append(ref)
    holes = [r for r in fps if r.startswith('H')]
    for r, (u, v) in zip(sorted(holes), [(HOLE_IN, HOLE_IN), (W - HOLE_IN, HOLE_IN), (HOLE_IN, H - HOLE_IN), (W - HOLE_IN, H - HOLE_IN)]):
        fps[r][0].SetPosition(to_xy(u, v)); placed.append(r)
    boxes = [bbox_mm(fps[r][0]) for r in placed]
    thru = [bbox_mm(fps[r][0]) for r in placed if any(pd.GetAttribute() == pcbnew.PAD_ATTRIB_PTH for pd in fps[r][0].Pads())]
    bottom = list(thru)
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
                bottom.append(b); placed.append(r); ok = True
                break
        if not ok:
            print('WARN could not place', r)
    for n in nets.values():
        pass
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
