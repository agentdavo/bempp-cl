#!/usr/bin/env python3
"""
Compression driver network validation using a plane-wave tube termination.

This does not run BEM. It uses a clean, frequency-independent load:
    Z = ρc / S
which is useful to debug internal network resonances and coupling paths.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_on_path()

    from bempp_audio.driver import (
        CompressionDriverConfig,
        DriverElectroMechConfig,
        RearVolumeConfig,
        FrontDuctConfig,
        CompressionDriverNetwork,
        CompressionDriverExcitation,
        AcousticMedium,
        plane_wave_tube_load_impedance,
    )

    cfg = CompressionDriverConfig(
        name="plane_wave_tube_demo",
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
        # Enable V1 coupling (example values)
        suspension_volume_m3=50e-6,
        suspension_diameter_m=0.09,
        voice_coil_slit_resistance_pa_s_per_m3=2.0e6,
        voice_coil_slit_mass_pa_s2_per_m3=5.0e2,
    )

    medium = AcousticMedium(c=343.0, rho=1.225)
    net = CompressionDriverNetwork(cfg, medium=medium)
    ex = CompressionDriverExcitation(voltage_rms=2.83)

    area = net.exit_area_m2 or (np.pi * (0.5 * 0.03) ** 2)
    z_load = plane_wave_tube_load_impedance(medium=medium, area_m2=float(area))

    freqs = np.logspace(np.log10(200.0), np.log10(20000.0), 40)
    u = np.array([net.solve_with_metrics(float(f), excitation=ex, z_external=z_load)["volume_velocity"] for f in freqs])
    print(f"U_rms throat: min={np.min(np.abs(u)):.3e}, max={np.max(np.abs(u)):.3e} m^3/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

