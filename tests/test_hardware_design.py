"""Consistency of the Phase 2 electronics description (hardware/pcb/design.py).

Pure Python: no KiCad needed. ERC/DRC of the generated boards are run with kicad-cli (docs/hardware/pcb_phase2.md).
Checks that every cable and contact pair carries the same pinout at both ends, that the flex pinouts are
palindromes (either way round is correct), that no two different connectors on a board can be swapped, and
that no net is left with a single pin unless it leaves the board on a connector.
"""
import os
import re
import sys
from collections import Counter, defaultdict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hardware', 'pcb'))
import design  # noqa: E402

BOARDS = {name: f() for name, f in design.BOARDS.items()}


def part(board, ref):
    return {p.ref: p for p in BOARDS[board]}[ref]


def pinout(board, ref):
    p = part(board, ref)
    nums = sorted(int(k[1:]) for k in p.pins if k.startswith('#'))
    return [p.pins['#%d' % n] for n in nums]


def unsided(nets):
    return [re.sub(r'_(L|R)$', '', n) for n in nets]


# cables are straight (pin n to pin n); docking contacts face each other in the same order
PAIRS = [
    (('power', 'J2'), ('light', 'J1')),          # LINK
    (('charge', 'J2'), ('power', 'J6')),         # CHARGE port
    (('pod_dock_l', 'J2'), ('light', 'J3')),     # POD-L
    (('pod_dock_r', 'J2'), ('light', 'J4')),     # POD-R
    (('hood_dock', 'J2'), ('light', 'J5')),      # HOOD
    (('btn_l', 'J4'), ('light', 'J6')),          # SIDE-L buttons
    (('btn_r', 'J4'), ('light', 'J7')),          # SIDE-R buttons
    (('white_l', 'J4'), ('uv', 'J2')),           # WHITE-L carrier -> pod hub
    (('white_r', 'J4'), ('uv', 'J2')),           # WHITE-R carrier -> pod hub
    (('hood_pad', 'J1'), ('hood_dock', 'J1')),   # hood pads on the dock's spring pins
]


@pytest.mark.parametrize('a,b', PAIRS, ids=['%s.%s-%s.%s' % (a + b) for a, b in PAIRS])
def test_cable_pinouts_match(a, b):
    assert pinout(*a) == pinout(*b)


@pytest.mark.parametrize('dock', ['pod_dock_l', 'pod_dock_r'])
def test_pod_flex_matches_hub(dock):
    # the pod hub is side-agnostic; the dock adds the side to every net
    assert unsided(pinout(dock, 'J1')) == pinout('uv', 'J1')


def test_flex_pinouts_are_palindromes():
    assert design.POD_FLEX == design.POD_FLEX[::-1]
    assert design.uv_pins() == design.uv_pins()[::-1]
    flex = BOARDS['podflex']
    p1, p2 = pinout('podflex', 'P1'), pinout('podflex', 'P2')
    assert p1 == p2[::-1]                          # straight conductors: the far end is turned round
    assert len(flex) == 2


def test_link_is_the_shared_constant():
    assert part('power', 'J2').pins == design.LINK == part('light', 'J1').pins


def _family(fp):
    return re.sub(r'_1x\d+.*', '', fp.split(':')[1])


@pytest.mark.parametrize('board', sorted(BOARDS))
def test_no_swappable_connectors(board):
    """Connectors of the same type and size on one board must be truly interchangeable (identical pinout,
    like the three module terminals on a WHITE carrier), or a plug could go in the wrong one. Push-in
    terminals for loose module/button leads are exempt: each is labelled on the silkscreen (R1-R6, TRIG...)."""
    seen = defaultdict(list)
    for p in BOARDS[board]:
        if p.ref.startswith(('J', 'P')) and p.pins and 'PTSM' not in p.footprint and 'Fingers' not in p.footprint:
            seen[(p.footprint, len(p.pins))].append(p)
    for (fp, n), parts in seen.items():
        maps = {tuple(sorted(q.pins.items())) for q in parts}
        assert len(maps) == 1, (board, fp, [q.ref for q in parts])


def test_off_board_connectors_are_unique_across_the_light_board():
    fams = Counter((_family(p.footprint), len(p.pins)) for p in BOARDS['light'] if p.ref.startswith('J'))
    assert max(fams.values()) == 1, fams


@pytest.mark.parametrize('board', sorted(BOARDS))
def test_no_dangling_nets(board):
    pins = defaultdict(list)
    for p in BOARDS[board]:
        for k, n in p.pins.items():
            pins[n].append(p.ref)
    for n, refs in pins.items():
        if len(refs) == 1:
            assert refs[0].startswith(('J', 'P', 'TP')), (board, n, refs)


def test_white_channel_currents():
    """White and ring sinks run below the modules' own current so the sink regulates (see design.CH)."""
    full_scale = 3.3 * 10 / (150 + 10)
    ch = dict(design.CH)
    assert full_scale / ch['WHITE_L'] <= 0.15         # 3 modules per pod
    assert all(full_scale / ch['RING%d' % i] <= 0.065 for i in range(1, 7))   # one module per ring channel


def test_white_carriers_are_mirror_images():
    l_, r_ = BOARDS['white_l'], BOARDS['white_r']
    assert [(p.ref, p.footprint, p.pins) for p in l_] == [(p.ref, p.footprint, p.pins) for p in r_]
