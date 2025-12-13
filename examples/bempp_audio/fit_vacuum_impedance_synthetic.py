#!/usr/bin/env python3
"""
Synthetic demo: fit vacuum electro-mechanical parameters from a complex Z_elec curve.

This example generates a vacuum impedance curve from known parameters, adds mild noise,
fits the parameters back, and prints diagnostics. Replace the synthetic generator with
your measured data loader as needed.
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

    from bempp_audio.driver import fit_vacuum_impedance, vacuum_electrical_impedance

    freqs = np.logspace(np.log10(50.0), np.log10(20000.0), 200)

    true = dict(
        re_ohm=6.2,
        le_h=1.7e-4,
        bl_tm=11.5,
        mms_kg=0.018,
        cms_m_per_n=1.2e-4,
        rms_ns_per_m=1.1,
        voice_coil_eddy_rmax_ohm=3.0,
        voice_coil_eddy_fcorner_hz=900.0,
    )

    z = np.array([vacuum_electrical_impedance(frequency_hz=float(f), **true) for f in freqs], dtype=complex)
    rng = np.random.default_rng(0)
    z_meas = z * (1.0 + 0.01 * rng.standard_normal(z.shape) + 1j * 0.01 * rng.standard_normal(z.shape))

    initial = dict(
        re_ohm=5.5,
        le_h=1.0e-4,
        bl_tm=9.0,
        mms_kg=0.012,
        cms_m_per_n=2.0e-4,
        rms_ns_per_m=0.7,
        voice_coil_eddy_rmax_ohm=1.0,
        voice_coil_eddy_fcorner_hz=1200.0,
    )

    fit = fit_vacuum_impedance(frequencies_hz=freqs, z_measured_ohm=z_meas, initial=initial, fit_eddy=True)
    print("Fit success:", fit.diagnostics.success)
    print("Cost:", fit.diagnostics.cost)
    print("RMS resid:", fit.diagnostics.rms)
    print("cond(JTJ):", fit.diagnostics.cond_jtj)
    print("rank(JTJ):", fit.diagnostics.rank)
    for k in sorted(fit.x):
        print(f"{k:28s} = {fit.x[k]:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

