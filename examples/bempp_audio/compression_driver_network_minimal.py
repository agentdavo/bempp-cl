#!/usr/bin/env python3
"""
Compression driver lumped network (generic25-style) - minimal example.

This does not run BEM. It parses a minimal generic25 text block, converts it to
`CompressionDriverConfig`, then evaluates the electro-acoustic network driven by
2.83 Vrms.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


GENERIC25_MINIMAL = r"""
// Minimal-ish generic25-style snippet (units included)
Def_Driver 'Drv1'
  dD=25mm
  Mms=0.010kg
  Cms=2.0e-4m/N
  Rms=1.0Ns/m
  Bl=10Tm
  Re=6ohm
  Le=0.10mH

System 'Sys1'
  Driver 'D1' Def='Drv1' Node=1=0=10=20
  Enclosure 'Rear'
    Vb=200cm3
  Duct 'FrontDuct'
    dD=25mm Len=20mm
  Waveguide 'Exit'
    Len=30mm dTh=25mm dMo=50mm Conical
"""


def main() -> int:
    _ensure_repo_on_path()

    from bempp_audio.driver import generic25_to_compression_driver_config
    from bempp_audio.driver.generic25 import parse_generic25
    from bempp_audio.driver.network import (
        CompressionDriverNetwork,
        CompressionDriverExcitation,
        AcousticMedium,
    )

    system = parse_generic25(GENERIC25_MINIMAL)
    cfg = generic25_to_compression_driver_config(system)

    medium = AcousticMedium(c=343.0, rho=1.225)
    network = CompressionDriverNetwork(cfg, medium=medium)
    excitation = CompressionDriverExcitation(voltage_rms=2.83)

    freqs = np.logspace(np.log10(200.0), np.log10(20000.0), 20)
    u_rms = np.array([network.solve_volume_velocity(float(f), excitation=excitation, z_external=0.0) for f in freqs])
    print(f"Computed volume velocity RMS: min={u_rms.min():.3e}, max={u_rms.max():.3e} m^3/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

