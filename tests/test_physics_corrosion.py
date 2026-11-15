"""Uniform-corrosion forecast (ISO 9223 / ISO 9224): numbers checked against ISO 9224:2012 Table 3."""

import math

import pytest

from trinetra.physics import corrosion as C


def test_power_law_matches_iso9224_table3():
    # ISO 9224:2012 Table 3: t^b at t = 10 years: steel B1 3.334, zinc B1 6.501, aluminium B2 6.412
    assert C.attack_um(10, 1.0, C.B_EXP["carbon_steel"]["B1"]) == pytest.approx(3.334, abs=2e-3)
    assert C.attack_um(10, 1.0, C.B_EXP["zinc"]["B1"]) == pytest.approx(6.501, abs=2e-3)
    assert C.attack_um(10, 1.0, C.B_EXP["aluminium"]["B2"]) == pytest.approx(6.412, abs=2e-3)
    assert C.attack_um(20, 1.0, 0.523) == pytest.approx(4.791, abs=2e-3)


def test_inverse_and_linear_extension():
    for t in (0.5, 3.0, 19.0, 25.0, 60.0):
        d = C.attack_um(t, 30.0, 0.523)
        assert C.years_to_attack(d, 30.0, 0.523) == pytest.approx(t, rel=1e-9)
    # beyond 20 years the attack grows linearly (Eq. 3), faster than the pure power law
    assert C.attack_um(40, 1.0, 0.523) > 40 ** 0.523


def test_forecast_ranges_and_labels():
    f = C.forecast("ms_hot_rolled", 5.0, "C3")
    y = f["years"]
    assert y["fastest"] < y["typical"] < y["slowest"]
    assert "advisory" in f["label"] and "pit" in f["label"]
    assert f["loss_allowed_um_per_side"] == pytest.approx(200.0)
    assert C.forecast("ss_brushed", 3.0, "C5") is None
    g = C.forecast("galvanised", 3.0, "C4")
    assert g["years"]["fastest"] > C.forecast("ms_cold_rolled", 3.0, "C4")["years"]["fastest"]
    assert C.forecast("ms_cold_rolled", 3.0, "C1")["years"]["slowest"] is None     # C1 lower rate 0: never
    with pytest.raises(ValueError):
        C.forecast("ms_hot_rolled", 3.0, "C9")
