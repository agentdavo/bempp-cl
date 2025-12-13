from __future__ import annotations

import pytest

from bempp_audio.driver.generic25 import parse_generic25, _parse_number_with_unit


def test_parse_number_with_unit_basic():
    assert _parse_number_with_unit("1mm") == pytest.approx(1e-3)
    assert _parse_number_with_unit("2.5cm") == pytest.approx(2.5e-2)
    assert _parse_number_with_unit("3kHz") == pytest.approx(3000.0)
    assert _parse_number_with_unit("4") == pytest.approx(4.0)


def test_parse_number_with_unit_unknown_unit():
    with pytest.raises(ValueError):
        _parse_number_with_unit("1bananas")


def test_parse_generic25_minimal_system():
    text = r"""
    // comment
    Def_Driver 'Drv1'
      dD=25mm Mms=0.01kg Cms=2.0e-4m/N Rms=1.0Ns/m Bl=10Tm Re=6ohm Le=0.1mH
    System 'Sys1'
      Driver 'D1' Def='Drv1' Node=1=0=10=20
      Enclosure 'Rear'
        Vb=200cm3
      Duct 'FrontDuct'
        dD=25mm Len=20mm
      Waveguide 'Exit'
        Len=30mm dTh=25mm dMo=50mm Conical
    """
    system = parse_generic25(text)
    assert system.system_name == "Sys1"
    assert system.driver.name == "Drv1"
    assert system.rear_volume is not None and system.rear_volume.volume_m3 > 0
    assert system.front_duct is not None and system.front_duct.length_m > 0
    assert system.waveguides is not None and len(system.waveguides) == 1

