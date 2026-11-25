"""Write a KiCad schematic (.kicad_sch) for a board in design.py.

Every part (and every unit of a multi-unit part) is placed on a grid; each connected pin gets a short wire
stub and a net label, unused pins get a no-connect flag, and power nets get a PWR_FLAG. Library symbols are
copied from the installed KiCad libraries (derived symbols flattened), so the schematic is self-contained.

usage: python3 hardware/pcb/gen_sch.py power|light [out_dir]
"""
from __future__ import annotations

import copy
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))
import design  # noqa: E402
from kisexp import Sym, dump, find, find_all, parse  # noqa: E402

KICAD_SYM = os.environ.get('KICAD_SYMBOL_DIR', '/Applications/KiCad/KiCad.app/Contents/SharedSupport/symbols')
GRID = 2.54
POWER_NETS = ('GND', 'VBUS', 'V12', 'V12_LED', 'V5', '3V3', '1V1', 'VPLUS', 'CHG_VBUS')
_LIBS = {}


def U():
    return str(uuid.uuid4())


def lib(name):
    if name not in _LIBS:
        _LIBS[name] = parse(open(os.path.join(KICAD_SYM, name + '.kicad_sym')).read())
    return _LIBS[name]


def raw_symbol(lib_name, sym_name):
    for s in find_all(lib(lib_name), 'symbol'):
        if s[1] == sym_name:
            return s
    raise KeyError('%s:%s not in the KiCad library' % (lib_name, sym_name))


def flat_symbol(lib_id):
    """The symbol with any `extends` resolved, renamed to its lib_id (as KiCad stores it in a schematic)."""
    lib_name, sym_name = lib_id.split(':')
    s = copy.deepcopy(raw_symbol(lib_name, sym_name))
    ext = find(s, 'extends')
    if ext:
        parent = copy.deepcopy(raw_symbol(lib_name, ext[1]))
        props = {p[1]: p for p in find_all(s, 'property')}
        body = [e for e in parent[2:] if not (isinstance(e, list) and e and e[0] == 'property')]
        pprops = [props.get(p[1], p) for p in find_all(parent, 'property')]
        pprops += [p for k, p in props.items() if k not in {q[1] for q in pprops}]
        for e in body:
            if isinstance(e, list) and e and e[0] == 'symbol':
                e[1] = e[1].replace(ext[1] + '_', sym_name + '_', 1)
        s = ['symbol', sym_name] + pprops + body
    s[1] = lib_id
    return s


def pins_of(sym):
    """[(unit, number, name, x, y, angle, length, etype)] in symbol coordinates (y up)."""
    out = []
    for sub in find_all(sym, 'symbol'):
        parts = sub[1].rsplit('_', 2)
        unit = int(parts[-2]) if len(parts) >= 3 and parts[-2].isdigit() else 0
        for pin in find_all(sub, 'pin'):
            at, ln = find(pin, 'at'), find(pin, 'length')
            nm, nu = find(pin, 'name'), find(pin, 'number')
            out.append((unit, nu[1], nm[1], float(at[1]), float(at[2]), float(at[3]) if len(at) > 3 else 0.0,
                        float(ln[1]), str(pin[1])))
    return out


def resolve(part, pins):
    """{(unit, number): net} from the part's name/number map; KeyError for an unknown pin."""
    out = {}
    for key, net in part.pins.items():
        if key.startswith('#'):
            hits = [p for p in pins if p[1] == key[1:]]
        else:
            hits = [p for p in pins if p[2] == key]
        if not hits:
            raise KeyError('%s (%s): no pin %r; pins are %s' % (part.ref, part.symbol, key,
                                                                sorted({p[2] for p in pins})))
        for p in hits:
            out[(p[0], p[1])] = net
    return out


def units(pins):
    us = sorted({p[0] for p in pins if p[0] != 0})
    return us or [1]


def bbox(pins, unit):
    pts = [(p[3], p[4]) for p in pins if p[0] in (0, unit)] or [(0, 0)]
    xs, ys = [x for x, _ in pts], [y for _, y in pts]
    return min(xs), max(xs), min(ys), max(ys)


def snap(v):
    return round(v / GRID) * GRID


def prop(name, value, x, y, hide=False, size=1.27):
    eff = ['effects', ['font', ['size', size, size]]]
    if hide:
        eff.append(['hide', Sym('yes')])
    return ['property', name, value, ['at', x, y, 0], eff]


def build(board, out_dir):
    parts = design.BOARDS[board]()
    root = U()
    proj = 'ayas_' + board
    lib_syms, items, used_ids = [], [], set()
    x0, y0, row_h, x, y, maxw = 25.4, 25.4, 0.0, 25.4, 25.4, 380.0
    flags, driven = set(), set()
    for part in parts:
        sym = flat_symbol(part.symbol)
        if part.symbol not in used_ids:
            lib_syms.append(sym)
            used_ids.add(part.symbol)
        pins = pins_of(sym)
        nets = resolve(part, pins)
        for unit in units(pins):
            bx0, bx1, by0, by1 = bbox(pins, unit)
            w, h = (bx1 - bx0) + 20 * GRID, (by1 - by0) + 8 * GRID
            if x + w > x0 + maxw:
                x, y, row_h = x0, y + row_h, 0.0
            sx, sy = snap(x - bx0 + 8 * GRID), snap(y + by1 + 3 * GRID)
            props = [prop('Reference', part.ref, sx, sy - (by1 + 2 * GRID) + 0.0),
                     prop('Value', part.value, sx, sy - by0 + 2 * GRID),
                     prop('Footprint', part.footprint, sx, sy, hide=True),
                     prop('Datasheet', '', sx, sy, hide=True),
                     prop('MPN', part.mpn, sx, sy, hide=True)]
            if part.note:
                props.append(prop('Note', part.note, sx, sy, hide=True))
            inst = ['symbol', ['lib_id', part.symbol], ['at', sx, sy, 0], ['unit', unit],
                    ['exclude_from_sim', Sym('no')], ['in_bom', Sym('yes')], ['on_board', Sym('yes')],
                    ['dnp', Sym('no')], ['uuid', U()]] + props
            for p in pins:
                if p[0] in (0, unit):
                    inst.append(['pin', p[1], ['uuid', U()]])
            inst.append(['instances', ['project', proj, ['path', '/' + root, ['reference', part.ref], ['unit', unit]]]])
            items.append(inst)
            for p in pins:
                if p[0] not in (0, unit):
                    continue
                px, py = sx + p[3], sy - p[4]                     # schematic y grows downward
                ang = p[5] % 360
                d = {0: (-1, 0), 180: (1, 0), 90: (0, 1), 270: (0, -1)}[int(ang)]
                net = nets.get((p[0], p[1]))
                if net is None:
                    if p[7] not in ('no_connect', 'power_in') or True:
                        items.append(['no_connect', ['at', px, py], ['uuid', U()]])
                    continue
                ex, ey = px + d[0] * GRID * 2, py + d[1] * GRID * 2
                items.append(['wire', ['pts', ['xy', px, py], ['xy', ex, ey]],
                              ['stroke', ['width', 0], ['type', Sym('default')]], ['uuid', U()]])
                lang = {(-1, 0): 180, (1, 0): 0, (0, 1): 270, (0, -1): 90}[d]
                items.append(['label', net, ['at', ex, ey, lang],
                              ['effects', ['font', ['size', 1.27, 1.27]],
                               ['justify', Sym('right' if lang == 180 else 'left' if lang == 0 else 'left')]],
                              ['uuid', U()]])
                if net in POWER_NETS:
                    flags.add(net)
                if p[7] == 'power_out':
                    driven.add(net)
            x += w
            row_h = max(row_h, h)
    # PWR_FLAG per power net (ERC: these nets are driven by connectors / regulators outside the sheet)
    pf = flat_symbol('power:PWR_FLAG')
    lib_syms.append(pf)
    x, y = x0, y + row_h + 10 * GRID
    for i, net in enumerate(sorted(flags - driven)):
        fx, fy = snap(x + i * 12 * GRID), snap(y)
        items.append(['symbol', ['lib_id', 'power:PWR_FLAG'], ['at', fx, fy, 0], ['unit', 1],
                      ['exclude_from_sim', Sym('no')], ['in_bom', Sym('no')], ['on_board', Sym('no')], ['dnp', Sym('no')],
                      ['uuid', U()], prop('Reference', '#FLG%02d' % (i + 1), fx, fy - 5, hide=True),
                      prop('Value', 'PWR_FLAG', fx, fy - 3.5, hide=True), prop('Footprint', '', fx, fy, hide=True),
                      ['pin', '1', ['uuid', U()]],
                      ['instances', ['project', proj, ['path', '/' + root, ['reference', '#FLG%02d' % (i + 1)], ['unit', 1]]]]])
        items.append(['wire', ['pts', ['xy', fx, fy], ['xy', fx, fy + 2 * GRID]],
                      ['stroke', ['width', 0], ['type', Sym('default')]], ['uuid', U()]])
        items.append(['label', net, ['at', fx, fy + 2 * GRID, 270], ['effects', ['font', ['size', 1.27, 1.27]],
                                                                       ['justify', Sym('left')]], ['uuid', U()]])
    title = ['title_block', ['title', 'Ayas %s board' % board.upper()], ['date', '2026-11-24'], ['rev', 'P2-A'],
             ['company', 'Ayas'], ['comment', 1, 'Generated by hardware/pcb/gen_sch.py from design.py']]
    doc = ['kicad_sch', ['version', 20250114], ['generator', 'eeschema'], ['generator_version', '9.0'],
           ['uuid', root], ['paper', 'A3' if board == 'power' else 'A2'], title, ['lib_symbols'] + lib_syms] + items + \
          [['sheet_instances', ['path', '/', ['page', '1']]]]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, proj + '.kicad_sch')
    with open(path, 'w') as f:
        f.write(dump(doc) + '\n')
    return path, parts


if __name__ == '__main__':
    b = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), 'kicad', b)
    path, parts = build(b, out)
    print('wrote', path, len(parts), 'parts')
