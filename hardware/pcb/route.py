"""Auto-route a generated board with Freerouting, then add ground pours (KiCad Python).

1. export Specctra DSN from ayas_<board>.kicad_pcb (rules from the .kicad_pro next to it)
2. run Freerouting 2.5 headless (Java 25)
3. import the session (.ses), add GND zones (both outer layers; on 4+ layer boards In1 is poured as a GND plane before routing), fill, save

usage: <KiCad python> hardware/pcb/route.py power|light [passes]
env: FREEROUTING_JAR (default ~/.local/share/freerouting/freerouting-2.5.0.jar), JAVA (default Homebrew openjdk)
"""
from __future__ import annotations

import os
import subprocess
import sys

import pcbnew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
JAR = os.environ.get('FREEROUTING_JAR', os.path.expanduser('~/.local/share/freerouting/freerouting-2.5.0.jar'))
JAVA = os.environ.get('JAVA', '/opt/homebrew/opt/openjdk/bin/java')


# high-current pours (board u, v in mm; priority over GND): 5 V / 3 A from L1 and the output caps to the Pi lead J3
POWER_ZONES = {'power': [('V5', 'F_Cu', [(8, 39), (31, 39), (31, 53.5), (8, 53.5)])]}


def add_zone(board, layer, net_name, priority=0, uv=None):
    net = board.FindNet(net_name)
    zone = pcbnew.ZONE(board)
    zone.SetLayer(layer)
    zone.SetNet(net)
    bb = board.GetBoardEdgesBoundingBox()
    pts = [(bb.GetLeft(), bb.GetTop()), (bb.GetRight(), bb.GetTop()), (bb.GetRight(), bb.GetBottom()), (bb.GetLeft(), bb.GetBottom())]
    if uv:
        pts = [(pcbnew.FromMM(100.0 + u), pcbnew.FromMM(100.0 + v)) for u, v in uv]
    ol = zone.Outline()
    ol.NewOutline()
    for x, y in pts:
        ol.Append(x, y)
    zone.SetAssignedPriority(priority)
    zone.SetLocalClearance(pcbnew.FromMM(0.25))
    zone.SetMinThickness(pcbnew.FromMM(0.2))
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
    zone.SetThermalReliefGap(pcbnew.FromMM(0.3))
    zone.SetThermalReliefSpokeWidth(pcbnew.FromMM(0.4))
    zone.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
    board.Add(zone)
    return zone


def main():
    name = sys.argv[1]
    passes = sys.argv[2] if len(sys.argv) > 2 else '40'
    d = os.path.join(HERE, 'kicad', name)
    pcb = os.path.join(d, 'ayas_%s.kicad_pcb' % name)
    dsn, ses = os.path.join(d, 'route.dsn'), os.path.join(d, 'route.ses')
    import gen_pcb                                     # always start from a freshly generated, unrouted board
    gen_pcb.build(name, d)
    board = pcbnew.LoadBoard(pcb)
    if name.startswith('uvflex') or name == 'podflex':   # flex: straight single-layer conductors, no vias
        fps = {f.GetReference(): f for f in board.GetFootprints()}
        p1 = sorted(fps['P1'].Pads(), key=lambda q: q.GetPosition().x)
        p2 = sorted(fps['P2'].Pads(), key=lambda q: q.GetPosition().x)
        for a_, b_ in zip(p1, p2):
            assert a_.GetNetname() == b_.GetNetname(), (a_.GetNetname(), b_.GetNetname())   # palindrome pinout
            t = pcbnew.PCB_TRACK(board)
            t.SetStart(a_.GetPosition()); t.SetEnd(b_.GetPosition())
            t.SetWidth(pcbnew.FromMM(0.5)); t.SetLayer(pcbnew.F_Cu); t.SetNet(a_.GetNet())
            board.Add(t)
        pcbnew.SaveBoard(pcb, board)
        print('saved', pcb, 'flex: 8 straight conductors on F.Cu')
        return
    plane = board.GetCopperLayerCount() >= 4
    if plane:                                           # GND plane on In1 BEFORE routing: GND pads just drop a via to it
        add_zone(board, pcbnew.In1_Cu, 'GND')
        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    for rnd in range(3):                                # later rounds keep what is routed and finish the rest
        if not pcbnew.ExportSpecctraDSN(board, dsn):
            raise SystemExit('DSN export failed')
        if os.path.exists(ses):
            os.remove(ses)
        cmd = [JAVA, '-jar', JAR, '-de', dsn, '-do', ses, '-mp', passes, '--gui.enabled=false']
        print(' '.join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = [l for l in (r.stdout + r.stderr).splitlines() if 'INFO' in l or 'WARN' in l or 'ERROR' in l][-6:]
        print('\n'.join(tail), flush=True)
        if not os.path.exists(ses):
            raise SystemExit('Freerouting produced no session')
        if not pcbnew.ImportSpecctraSES(board, ses):
            raise SystemExit('SES import failed')
        board.BuildConnectivity()
        left = board.GetConnectivity().GetUnconnectedCount(True)
        print('round', rnd + 1, 'unconnected', left, flush=True)
        if left == 0:
            break
    min_w = pcbnew.FromMM(0.1)                         # Freerouting necks down into small pads: hold the fab minimum,
    for t in board.GetTracks():                        # or as wide as the clearance allows (>= 0.09 mm)
        if t.GetClass() == 'PCB_TRACK' and t.GetWidth() < min_w:
            t.SetWidth(min_w)
    board.BuildConnectivity()
    for t in board.GetTracks():
        if t.GetClass() == 'PCB_TRACK' and t.GetWidth() == min_w:
            for w in (0.1, 0.095, 0.093, 0.09):
                t.SetWidth(pcbnew.FromMM(w))
                bad = any(o.GetNetCode() != t.GetNetCode() and o.IsOnLayer(t.GetLayer()) and
                          t.GetEffectiveShape().Collide(o.GetEffectiveShape(t.GetLayer()), pcbnew.FromMM(0.1) - 1)
                          for fp in board.GetFootprints() for o in fp.Pads())
                if not bad:
                    break
    layers = [] if (name.startswith('uvflex') or name == 'podflex') else \
        [pcbnew.F_Cu, pcbnew.B_Cu]                       # (In1 was poured before routing)
    for ly in layers:
        add_zone(board, ly, 'GND')
    for net, ly, uv in POWER_ZONES.get(name, []):
        add_zone(board, getattr(pcbnew, ly), net, priority=1, uv=uv)
    for fp in board.GetFootprints():                    # shield / mounting pads: solid to the pour (no starved thermals)
        for pad in fp.Pads():
            if pad.GetNumber() in ('SH', 'S1', 'MP') or fp.GetReference().startswith('H') or pad.GetNetname() == 'GND':
                pad.SetLocalZoneConnection(pcbnew.ZONE_CONNECTION_FULL)
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())
    pcbnew.SaveBoard(pcb, board)
    n_tracks = sum(1 for t in board.GetTracks() if t.GetClass() == 'PCB_TRACK')
    n_vias = sum(1 for t in board.GetTracks() if t.GetClass() == 'PCB_VIA')
    print('saved', pcb, 'tracks', n_tracks, 'vias', n_vias)


if __name__ == '__main__':
    main()
