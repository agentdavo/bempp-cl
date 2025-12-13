"""
Validation helpers for compression-driver lumped networks.
"""

from __future__ import annotations

import numpy as np

from bempp_audio.driver.network import AcousticMedium


def plane_wave_tube_load_impedance(
    *,
    medium: AcousticMedium,
    area_m2: float,
) -> complex:
    """
    Plane-wave tube acoustic load impedance: Z = ρc / S (Pa·s/m³).

    This is real and frequency independent, and is useful as a clean termination
    for debugging and validation of internal networks.
    """
    s = float(area_m2)
    if s <= 0:
        return np.inf + 0j
    return (float(medium.rho) * float(medium.c)) / s


def vacuum_electrical_impedance(
    *,
    frequency_hz: float,
    re_ohm: float,
    le_h: float,
    bl_tm: float,
    mms_kg: float,
    cms_m_per_n: float,
    rms_ns_per_m: float,
    voice_coil_eddy_rmax_ohm: float = 0.0,
    voice_coil_eddy_fcorner_hz: float = 1000.0,
) -> complex:
    """
    Electrical input impedance under vacuum (no acoustic loading).

    Implements:
        Z_elec = (Re + jωLe + Reddy(f)) + Bl^2 / Z_mech
    with mechanical impedance:
        Z_mech = Rms + jωMms + 1/(jωCms)
    """
    omega = 2 * np.pi * float(frequency_hz)
    if omega <= 0:
        return complex(float(re_ohm), 0.0)

    z_mech = float(rms_ns_per_m) + 1j * omega * float(mms_kg) + 1.0 / (1j * omega * float(cms_m_per_n))
    z_coil = float(re_ohm) + 1j * omega * float(le_h)

    f = float(frequency_hz)
    f0 = max(1e-9, float(voice_coil_eddy_fcorner_hz))
    x = f / f0
    r_eddy = float(voice_coil_eddy_rmax_ohm) * (x * x) / (1.0 + x * x)
    z_coil = z_coil + r_eddy

    if abs(z_mech) < 1e-30:
        z_motional = 0.0 + 0.0j
    else:
        z_motional = (float(bl_tm) ** 2) / z_mech
    return z_coil + z_motional


def fit_vacuum_electro_mech(
    *,
    frequencies_hz: np.ndarray,
    z_measured_ohm: np.ndarray,
    initial: dict,
) -> dict:
    """
    Fit a vacuum electro-mechanical model to a measured complex impedance curve.

    Requires SciPy. Returns a dict of fitted parameters.
    """
    try:
        from scipy.optimize import least_squares
    except Exception as e:  # pragma: no cover
        raise ImportError("fit_vacuum_electro_mech requires SciPy (`pip install scipy`).") from e

    freqs = np.asarray(frequencies_hz, dtype=float)
    z_meas = np.asarray(z_measured_ohm, dtype=complex)

    keys = [
        "re_ohm",
        "le_h",
        "bl_tm",
        "mms_kg",
        "cms_m_per_n",
        "rms_ns_per_m",
        "voice_coil_eddy_rmax_ohm",
        "voice_coil_eddy_fcorner_hz",
    ]

    x0 = np.array([float(initial.get(k, 0.0)) for k in keys], dtype=float)

    def _resid(x: np.ndarray) -> np.ndarray:
        params = dict(zip(keys, x))
        z_hat = np.array(
            [
                vacuum_electrical_impedance(frequency_hz=float(f), **params)
                for f in freqs
            ],
            dtype=complex,
        )
        # Stack real/imag residuals for stable fitting.
        r = np.concatenate([np.real(z_hat - z_meas), np.imag(z_hat - z_meas)])
        return r

    res = least_squares(_resid, x0=x0, method="trf")
    out = dict(zip(keys, res.x.tolist()))
    out["success"] = bool(res.success)
    out["cost"] = float(res.cost)
    return out

