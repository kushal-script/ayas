"""Uniform-corrosion thickness-loss forecast for the advisor (ISO 9223 / ISO 9224). ADVISORY ONLY.

This is a UNIFORM-corrosion forecast from the site's atmospheric corrosivity category. It is not
pit growth, not crevice or galvanic corrosion, and not a remaining-life certificate: the advisor
must say so (docs/twin/PHYSICS_PINN.md 6).

    D(t) = r_corr t^b                                   t <= 20 years    ISO 9224:2012 Eq. (1)
    D(t) = r_corr [20^b + b 20^(b-1) (t - 20)]          t > 20 years     ISO 9224:2012 Eq. (3) (upper estimate)

r_corr: first-year corrosion rate of the category (ISO 9223:2012 Table 2, um/a; aluminium is given
in g/(m^2 a) only and is converted with 2.70 g/cm^3). b: ISO 9224:2012 Table 2 time exponents,
B1 (average of the ISO CORRAG flat-panel regressions) and B2 (B1 + 2 standard deviations, for an
upper limit). Sources (checked 15 Nov 2026):
- ISO 9224:2012 preview, Clause 4 Eq. (1), Clause 5, Clause 7 Eq. (3), Table 2
  (https://cdn.standards.iteh.ai/samples/53500/48d0ad5886f74a05a0d0faf4a71da09e/ISO-9224-2012.pdf);
- ISO 9223:2012 Table 2 as reproduced in J. Tidblad, "Classification: outdoor (ISO 9223-9226)"
  (https://www.rrcap.ait.ac.th/male/Meeting%20Document/14%20JT%20Classification.pdf); the zinc
  column also matches the American Galvanizers Association's table
  (https://galvanizeit.org/knowledgebase/article/hdg-corrosion-rates-for-iso-categories-c1-c5-x).
Stainless steels are not covered by ISO 9224: the forecast returns None for them.
"""

from __future__ import annotations

import math
from typing import Any

CATEGORIES = ("C1", "C2", "C3", "C4", "C5", "CX")

# ISO 9223:2012 Table 2, first-year corrosion rate bands r_corr (lower, upper), um/a.
RATE_UM_A: dict[str, dict[str, tuple[float, float]]] = {
    "carbon_steel": {"C1": (0.0, 1.3), "C2": (1.3, 25.0), "C3": (25.0, 50.0), "C4": (50.0, 80.0),
                     "C5": (80.0, 200.0), "CX": (200.0, 700.0)},
    "zinc": {"C1": (0.0, 0.1), "C2": (0.1, 0.7), "C3": (0.7, 2.1), "C4": (2.1, 4.2),
             "C5": (4.2, 8.4), "CX": (8.4, 25.0)},
    "copper": {"C1": (0.0, 0.1), "C2": (0.1, 0.6), "C3": (0.6, 1.3), "C4": (1.3, 2.8),
               "C5": (2.8, 5.6), "CX": (5.6, 10.0)},
}
# aluminium, g/(m^2 a): C1 negligible, C2 <= 0.6, C3 0.6-2, C4 2-5, C5 5-10, CX > 10 (no upper bound given)
AL_G_M2_A = {"C1": (0.0, 0.0), "C2": (0.0, 0.6), "C3": (0.6, 2.0), "C4": (2.0, 5.0), "C5": (5.0, 10.0),
             "CX": (10.0, None)}
AL_DENSITY_G_CM3 = 2.70
RATE_UM_A["aluminium"] = {c: (lo / AL_DENSITY_G_CM3, (hi / AL_DENSITY_G_CM3) if hi is not None else None)
                          for c, (lo, hi) in AL_G_M2_A.items()}

# ISO 9224:2012 Table 2
B_EXP = {"carbon_steel": {"B1": 0.523, "B2": 0.575}, "zinc": {"B1": 0.813, "B2": 0.873},
         "copper": {"B1": 0.667, "B2": 0.726}, "aluminium": {"B1": 0.728, "B2": 0.807}}

# plate.material.kind -> ISO 9224 metal (stainless: not covered)
METAL_OF = {"ms_hot_rolled": "carbon_steel", "ms_cold_rolled": "carbon_steel", "galvanised": "zinc",
            "aluminium": "aluminium", "ss_brushed": None, "ss_2b": None}

LABEL = ("Uniform-corrosion forecast (ISO 9223 category, ISO 9224 power law): advisory only. "
         "It does not predict pit growth, crevice or galvanic attack; check pits and edges separately.")


def attack_um(t_years: float, r_um_a: float, b: float) -> float:
    """Corrosion attack D (um) after t years: Eq. (1) up to 20 years, Eq. (3) beyond (upper estimate)."""
    t = max(0.0, float(t_years))
    if t <= 20.0:
        return r_um_a * t ** b
    return r_um_a * (20.0 ** b + b * 20.0 ** (b - 1.0) * (t - 20.0))


def years_to_attack(d_um: float, r_um_a: float, b: float) -> float:
    """Inverse of attack_um: years until the attack reaches d_um (inf when r = 0)."""
    if r_um_a <= 0:
        return math.inf
    x = float(d_um) / r_um_a
    if x <= 20.0 ** b:
        return x ** (1.0 / b)
    return 20.0 + (x - 20.0 ** b) / (b * 20.0 ** (b - 1.0))


def forecast(material: str, thickness_mm: float, category: str, *, min_thickness_mm: float | None = None,
             allowed_loss_frac: float = 0.04, sides: int = 1, zinc_coating_um: float = 85.0) -> dict[str, Any] | None:
    """Years until the remaining thickness reaches the limit under uniform corrosion.

    Limit: min_thickness_mm if given, else thickness x (1 - allowed_loss_frac) (default 4 %, the IS 2062
    cl. 7 local grinding allowance used by spec_steel.yaml). sides: faces exposed (1 or 2).
    Galvanised: the zinc coating (zinc_coating_um a side; 85 um is the ISO 1461 minimum mean for
    steel over 6 mm, an assumption here) is consumed first, then the carbon-steel rates apply.
    Returns years as a range: fastest (upper rate, B2) / typical (mid rate, B1) / slowest (lower rate, B1),
    or None for stainless steels (not covered by ISO 9224)."""
    cat = str(category).upper()
    if cat not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    metal = METAL_OF.get(material, material if material in RATE_UM_A else None)
    if metal is None:
        return None
    t = float(thickness_mm)
    limit = float(min_thickness_mm) if min_thickness_mm is not None else t * (1.0 - float(allowed_loss_frac))
    loss_um = max(0.0, (t - limit) * 1000.0 / max(1, int(sides)))
    base = "carbon_steel" if metal == "zinc" else metal

    def years(metal_: str, d_um: float, which: str) -> float:
        lo, hi = RATE_UM_A[metal_][cat]
        if hi is None:
            hi = lo * 2.0
        r = {"fast": hi, "mid": 0.5 * (lo + hi), "slow": lo}[which]
        b = B_EXP[metal_]["B2" if which == "fast" else "B1"]
        return years_to_attack(d_um, r, b)

    out = {}
    for which in ("fast", "mid", "slow"):
        y = years(base, loss_um, which)
        if metal == "zinc":
            y = years("zinc", zinc_coating_um, which) + y
        out[which] = y

    def rnd(y):
        return None if not math.isfinite(y) else round(y, 1)

    return {"model": "ISO 9224:2012 Eq. (1)/(3), ISO 9223:2012 Table 2 rates", "label": LABEL,
            "material": material, "metal": metal, "category": cat, "thickness_mm": t,
            "limit_mm": round(limit, 3), "loss_allowed_um_per_side": round(loss_um, 1), "sides": int(sides),
            "years": {"fastest": rnd(out["fast"]), "typical": rnd(out["mid"]), "slowest": rnd(out["slow"])},
            "assumptions": (["zinc coating %g um a side consumed first" % zinc_coating_um] if metal == "zinc" else [])
            + (["aluminium rates converted from g/(m2 a) at 2.70 g/cm3; aluminium corrodes locally (pits), "
                "so the uniform rate understates the damage"] if metal == "aluminium" else [])
            + (["beyond 20 years the linear extension (Eq. 3) is used: an upper estimate of the attack"])}
