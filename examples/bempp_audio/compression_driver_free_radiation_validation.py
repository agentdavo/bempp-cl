#!/usr/bin/env python3
"""
Compression driver "free radiation" validation workflow (Panzer-style).

This script intentionally uses a low-damping termination to expose internal
reactive resonances:
- Solve BEM radiation impedance for a simple circular piston with unit velocity
- Use that impedance as `z_external` for the compression-driver internal network

This is not a strict model of a real compression driver aperture, but it mirrors
the workflow: (1) get a radiation load, (2) drive the lumped network with it.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import numpy as np


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_on_path()
    os.environ.setdefault("BEMPP_DEVICE_INTERFACE", "numba")

    from bempp_audio import Loudspeaker
    from bempp_audio.driver import (
        CompressionDriverConfig,
        DriverElectroMechConfig,
        RearVolumeConfig,
        FrontDuctConfig,
        CompressionDriverNetwork,
        CompressionDriverExcitation,
        AcousticMedium,
    )

    # Radiation load: simple piston
    throat_diameter = 0.036
    piston = (
        Loudspeaker()
        .circular_piston(diameter=throat_diameter, element_size=0.004)
        .free_space()
        .frequency_range(200.0, 20000.0, num=30, spacing="log")
        .velocity(mode="piston", amplitude=1.0)  # 1 m/s peak
    )
    piston_response = piston.solve(n_workers=1)

    # Lumped driver network
    cfg = CompressionDriverConfig(
        name="free_radiation_validation",
        driver=DriverElectroMechConfig(
            diaphragm_diameter_m=0.086,
            mms_kg=0.02,
            cms_m_per_n=1.5e-4,
            rms_ns_per_m=1.0,
            bl_tm=12.0,
            re_ohm=6.0,
            le_h=0.00015,
        ),
        rear_volume=RearVolumeConfig(volume_m3=200e-6),
        front_duct=FrontDuctConfig(diameter_m=0.03, length_m=0.01),
        suspension_volume_m3=50e-6,
        suspension_diameter_m=0.09,
        voice_coil_slit_resistance_pa_s_per_m3=2.0e6,
        voice_coil_slit_mass_pa_s2_per_m3=5.0e2,
    )

    medium = AcousticMedium(c=float(piston_response.results[0].c), rho=float(piston_response.results[0].rho))
    net = CompressionDriverNetwork(cfg, medium=medium)
    ex = CompressionDriverExcitation(voltage_rms=2.83)

    freqs = np.asarray(piston_response.frequencies, dtype=float)
    z_ext = np.array([r.radiation_impedance().acoustic() for r in piston_response.results], dtype=complex)
    u = np.array(
        [net.solve_with_metrics(float(f), excitation=ex, z_external=complex(z))["volume_velocity"] for f, z in zip(freqs, z_ext)],
        dtype=complex,
    )

    print(f"Computed throat U_rms: min={np.min(np.abs(u)):.3e}, max={np.max(np.abs(u)):.3e} m^3/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
