"""Minimal S-expression reader/writer for KiCad files (.kicad_sym, .kicad_sch)."""
from __future__ import annotations

import re

_TOK = re.compile(r'\s*(?:(\()|(\))|("(?:\\.|[^"\\])*")|([^\s()"]+))')


class Sym(str):
    """An unquoted atom (keyword or number), as opposed to a quoted string."""


def parse(text: str):
    stack, cur, pos = [], [], 0
    while True:
        m = _TOK.match(text, pos)
        if not m or m.end() == pos:
            break
        pos = m.end()
        lp, rp, s, a = m.groups()
        if lp:
            stack.append(cur)
            cur = []
        elif rp:
            done = cur
            cur = stack.pop()
            cur.append(done)
        elif s is not None:
            cur.append(s[1:-1].replace('\\"', '"').replace('\\\\', '\\'))
        else:
            cur.append(Sym(a))
    return cur[0] if len(cur) == 1 else cur


def q(s: str) -> str:
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') + '"'


def dump(x, ind: int = 0) -> str:
    if isinstance(x, list):
        if not x:
            return '()'
        simple = all(not isinstance(e, list) for e in x)
        hd = str(x[0]) if isinstance(x[0], str) else dump(x[0])     # the keyword is never quoted
        if simple and len(x) < 12:
            return '(' + ' '.join([hd] + [dump(e) for e in x[1:]]) + ')'
        head = hd
        inner = '\n'.join(' ' * (ind + 2) + dump(e, ind + 2) for e in x[1:])
        return '(' + head + '\n' + inner + '\n' + ' ' * ind + ')'
    if isinstance(x, Sym):
        return str(x)
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, (int, float)):
        return ('%.4f' % x).rstrip('0').rstrip('.') if isinstance(x, float) else str(x)
    return q(x)


def find(node, key):
    """First child list whose head is `key`."""
    for e in node:
        if isinstance(e, list) and e and e[0] == key:
            return e
    return None


def find_all(node, key):
    return [e for e in node if isinstance(e, list) and e and e[0] == key]
