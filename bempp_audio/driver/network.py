"""
Lumped-element electro-acoustic network for compression drivers.

Implements a frequency-domain model that maps an input voltage to the
volume velocity at the driver exit (throat), given an external acoustic load.

This is intended to couple to BEM radiation by using the BEM-computed
input acoustic impedance at the throat as the network termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence
import numpy as np

from bempp_audio.driver.generic25 import (
    Generic25SystemConfig,
    Generic25WaveguideConfig,
)
from bempp_audio.driver.compression_config import (
    CompressionDriverConfig,
    VoiceCoilImpedanceModel,
    FrequencyWeighting,
    WaveguideMatrixMethod,
)


@dataclass(frozen=True)
class AcousticMedium:
    c: float = 343.0
    rho: float = 1.225
    # Thermo-viscous properties (air defaults). Used when duct_loss_model != "Lossless".
    gamma: float = 1.4
    prandtl: float = 0.71
    dynamic_viscosity_pa_s: float = 1.84e-5


def _tube_abcd(k: float, rho: float, c: float, area: float, length: float) -> np.ndarray:
    """ABCD matrix for a uniform lossless tube."""
    if area <= 0 or length <= 0:
        return np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
    zc = rho * c / area
    kl = k * length
    cos = np.cos(kl)
    j_sin = 1j * np.sin(kl)
    return np.array([[cos, j_sin * zc], [j_sin / zc, cos]], dtype=complex)


def _cascade(mats: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
    for m in mats:
        out = out @ m
    return out


def _abcd_to_zin(abcd: np.ndarray, z_load: complex) -> complex:
    a, b, c, d = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
    if np.isinf(z_load):
        # Open circuit load: U2 = 0 -> Z_in = p1/U1 = (A p2)/(C p2) = A/C
        if abs(c) < 1e-30:
            return np.inf + 0j
        return a / c
    if abs(z_load) < 1e-30:
        # Short circuit load: p2 = 0 -> Z_in = p1/U1 = (B U2)/(D U2) = B/D
        if abs(d) < 1e-30:
            return np.inf + 0j
        return b / d
    denom = (c * z_load + d)
    if abs(denom) < 1e-30:
        return np.inf + 0j
    return (a * z_load + b) / denom


def _abcd_input_to_load_volume_velocity(abcd: np.ndarray, z_load: complex, u_in: complex) -> complex:
    """
    Convert input volume velocity (at port 1) to load-side volume velocity (port 2).

    Using [p1; U1] = [[A,B],[C,D]] [p2; U2] and p2 = ZL * U2, we get:
        U2 = U1 / (C*ZL + D)
    """
    c, d = abcd[1, 0], abcd[1, 1]
    if np.isinf(z_load):
        # Open circuit at load: U2 = 0 by definition.
        return 0.0 + 0.0j
    denom = (c * z_load + d)
    if abs(denom) < 1e-30:
        return np.inf + 0j
    return u_in / denom


def _area_from_diameter(d: float) -> float:
    return np.pi * (0.5 * d) ** 2


def _diameter_from_area(a: float) -> float:
    return np.sqrt(4.0 * a / np.pi)


def _parallel(z1: complex, z2: complex) -> complex:
    if np.isinf(z1):
        return z2
    if np.isinf(z2):
        return z1
    denom = z1 + z2
    if abs(denom) < 1e-30:
        return np.inf + 0j
    return (z1 * z2) / denom


def _interp_gain_db(weighting: FrequencyWeighting, frequency_hz: float) -> float:
    knots = list(weighting.knots_hz_db)
    if not knots:
        return 0.0
    knots = sorted(((float(f), float(g)) for f, g in knots), key=lambda t: t[0])
    f = max(1e-12, float(frequency_hz))
    fk = np.array([k[0] for k in knots], dtype=float)
    gk = np.array([k[1] for k in knots], dtype=float)
    if fk.size == 1:
        return float(gk[0])
    x = np.log10(fk)
    y = gk
    return float(np.interp(np.log10(f), x, y))


def _tube_y_params(k: float, rho: float, c: float, area: float, length: float) -> np.ndarray:
    """
    Y-parameter matrix for a uniform lossless tube (pressure/volume-velocity convention).

    Returns Y such that [U1; U2] = Y [p1; p2].
    """
    if area <= 0 or length <= 0:
        return np.zeros((2, 2), dtype=complex)
    zc = rho * c / area
    kl = k * length
    a = np.cos(kl)
    d = a
    b = 1j * zc * np.sin(kl)
    c_ = 1j * (1.0 / zc) * np.sin(kl)
    if abs(b) < 1e-30:
        return np.zeros((2, 2), dtype=complex)
    y11 = d / b
    y21 = 1.0 / b
    y22 = -a / b
    y12 = c_ - (a * d) / b
    return np.array([[y11, y12], [y21, y22]], dtype=complex)


def _tube_abcd_from_k_zc(*, k: complex, zc: complex, length: float) -> np.ndarray:
    """ABCD matrix for a uniform tube given (possibly complex) k and Zc."""
    if length <= 0:
        return np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
    kl = complex(k) * float(length)
    cos = np.cos(kl)
    j_sin = 1j * np.sin(kl)
    return np.array([[cos, j_sin * complex(zc)], [j_sin / complex(zc), cos]], dtype=complex)


def _tube_y_params_from_k_zc(*, k: complex, zc: complex, length: float) -> np.ndarray:
    """Y-parameter matrix for a uniform tube given (possibly complex) k and Zc."""
    if length <= 0:
        return np.zeros((2, 2), dtype=complex)
    kl = complex(k) * float(length)
    a = np.cos(kl)
    d = a
    b = 1j * complex(zc) * np.sin(kl)
    c_ = 1j * (1.0 / complex(zc)) * np.sin(kl)
    if abs(b) < 1e-30:
        return np.zeros((2, 2), dtype=complex)
    y11 = d / b
    y21 = 1.0 / b
    y22 = -a / b
    y12 = c_ - (a * d) / b
    return np.array([[y11, y12], [y21, y22]], dtype=complex)


def _kirchhoff_circular_duct_characteristics(
    *,
    omega: float,
    rho: float,
    c: float,
    radius_m: float,
    gamma: float,
    prandtl: float,
    dynamic_viscosity_pa_s: float,
) -> tuple[complex, complex]:
    """
    Return (k_eff, zc_eff) for a circular duct using Kirchhoff thermo-viscous corrections.

    Notes
    -----
    Uses the classical correction factors:
      F = 2*J1(mu)/(mu*J0(mu)) with mu=(1-i)*a/delta.
    """
    if omega <= 0 or radius_m <= 0:
        return 0.0 + 0.0j, np.inf + 0.0j

    area = np.pi * float(radius_m) ** 2
    k0 = float(omega) / float(c)
    zc0 = float(rho) * float(c) / float(area)

    nu = float(dynamic_viscosity_pa_s) / float(rho)
    if nu <= 0:
        return k0 + 0.0j, zc0 + 0.0j

    pr = max(1e-9, float(prandtl))
    delta_v = np.sqrt(2.0 * nu / float(omega))
    delta_t = delta_v / np.sqrt(pr)

    mu_v = (1.0 - 1.0j) * (float(radius_m) / float(delta_v))
    mu_t = (1.0 - 1.0j) * (float(radius_m) / float(delta_t))

    if abs(mu_v) < 1e-6 or abs(mu_t) < 1e-6:
        return k0 + 0.0j, zc0 + 0.0j

    try:
        # jv supports complex arguments across SciPy versions more reliably than j0/j1.
        from scipy.special import jv  # type: ignore
    except Exception:
        return k0 + 0.0j, zc0 + 0.0j

    def _F(mu: complex) -> complex:
        j0_mu = jv(0, mu)
        denom = mu * j0_mu
        if abs(denom) < 1e-30:
            return 0.0 + 0.0j
        return 2.0 * jv(1, mu) / denom

    Fv = _F(mu_v)
    Ft = _F(mu_t)

    denom = (1.0 - Fv)
    therm = (1.0 + (float(gamma) - 1.0) * Ft)
    if abs(denom) < 1e-30 or abs(therm) < 1e-30:
        return k0 + 0.0j, zc0 + 0.0j

    k_eff = k0 * np.sqrt(therm / denom)
    zc_eff = zc0 / np.sqrt(denom * therm)
    return complex(k_eff), complex(zc_eff)


def conical_waveguide_y_matrix(
    *,
    frequency_hz: float,
    rho: float,
    c: float,
    throat_area_m2: float,
    mouth_area_m2: float,
    length_m: float,
) -> np.ndarray:
    """
    Y-parameter matrix for a conical (linearly-varying area) waveguide.

    Implements the analytical form from Panzer (ICA 2019, eq. 8) for a duct
    with linearly rising cross-section area from throat (Sth) to mouth (Smo).

    The matrix satisfies: [q1; q2] = Y [p1; p2]

    Parameters
    ----------
    frequency_hz : float
        Frequency in Hz.
    rho : float
        Air density in kg/m³.
    c : float
        Speed of sound in m/s.
    throat_area_m2 : float
        Cross-section area at throat (input) in m².
    mouth_area_m2 : float
        Cross-section area at mouth (output) in m².
    length_m : float
        Length of waveguide section in m.

    Returns
    -------
    np.ndarray
        2x2 complex Y-parameter matrix.

    Notes
    -----
    At low frequencies where kL << 1, this reduces to a uniform tube.
    At the limits (Sth → Smo), this also reduces to a uniform tube.

    References
    ----------
    Panzer J., "Modeling of a Compression Driver using Lumped Elements",
    ICA 2019, Aachen, Germany.
    """
    sth = float(throat_area_m2)
    smo = float(mouth_area_m2)
    length = float(length_m)

    if sth <= 0 or smo <= 0 or length <= 0:
        return np.zeros((2, 2), dtype=complex)

    omega = 2 * np.pi * float(frequency_hz)
    if omega <= 0:
        return np.zeros((2, 2), dtype=complex)

    k = omega / float(c)
    kl = k * length

    # Handle very small kL (low frequency / short duct)
    if abs(kl) < 1e-6:
        # Fall back to a uniform-tube approximation with an effective area to
        # preserve the kL -> 0 inertance behavior.
        area_eff = float(np.sqrt(sth * smo))
        return _tube_y_params(k, float(rho), float(c), area_eff, length)

    # Panzer notation
    # y20 = sqrt(Smo/Sth) - 1
    # y0L = 1 / (y20 + 1) = sqrt(Sth/Smo)
    # y2L = y20 * y0L = sqrt(Smo/Sth) - sqrt(Sth/Smo) = (Smo - Sth) / sqrt(Smo*Sth)

    ratio = np.sqrt(smo / sth)
    y20 = ratio - 1.0
    y0L = 1.0 / ratio  # = sqrt(Sth/Smo)
    y2L = y20 * y0L  # = 1 - 1/ratio = (ratio - 1) / ratio

    # sinc(x) = sin(x) / x
    sin_kl = np.sin(kl)
    cos_kl = np.cos(kl)
    if abs(kl) < 1e-6:
        sinc_kl = 1.0 - (kl * kl) / 6.0  # Taylor expansion
    else:
        sinc_kl = sin_kl / kl

    # y0 = Smo / (j * rho * c * sin(kL))
    if abs(sin_kl) < 1e-30:
        # At resonance (kL = n*pi), matrix becomes singular
        return np.zeros((2, 2), dtype=complex)

    y0 = smo / (1j * float(rho) * float(c) * sin_kl)

    # Y-matrix elements (Panzer eq. 8)
    # y11 = y0L * (cos(kL) + y20 * sinc(kL))
    # y12 = -y0L
    # y21 = -y0L
    # y22 = cos(kL) - y2L * sinc(kL)

    # Note: The paper's y12, y21 = -y0L appears to be a simplification.
    # The full form involves y0 scaling. Let me use the corrected form:
    # [q1; q2] = y0 * [[y11, y12], [y21, y22]] * [p1; p2]

    y11 = y0L * (cos_kl + y20 * sinc_kl)
    y12 = -y0L
    y21 = -y0L
    y22 = cos_kl - y2L * sinc_kl

    return y0 * np.array([[y11, y12], [y21, y22]], dtype=complex)


def conical_waveguide_abcd_matrix(
    *,
    frequency_hz: float,
    rho: float,
    c: float,
    throat_area_m2: float,
    mouth_area_m2: float,
    length_m: float,
) -> np.ndarray:
    """
    ABCD (transfer) matrix for a conical waveguide, derived from Y-parameters.

    The ABCD matrix relates input (p1, U1) to output (p2, U2):
        [p1]   [A  B] [p2]
        [U1] = [C  D] [U2]

    Parameters
    ----------
    frequency_hz : float
        Frequency in Hz.
    rho : float
        Air density in kg/m³.
    c : float
        Speed of sound in m/s.
    throat_area_m2 : float
        Cross-section area at throat (input) in m².
    mouth_area_m2 : float
        Cross-section area at mouth (output) in m².
    length_m : float
        Length of waveguide section in m.

    Returns
    -------
    np.ndarray
        2x2 complex ABCD matrix.
    """
    y = conical_waveguide_y_matrix(
        frequency_hz=frequency_hz,
        rho=rho,
        c=c,
        throat_area_m2=throat_area_m2,
        mouth_area_m2=mouth_area_m2,
        length_m=length_m,
    )

    # Convert Y to ABCD
    # For Y: [U1; U2] = [[y11, y12], [y21, y22]] [p1; p2]
    # ABCD: p1 = A*p2 + B*U2, U1 = C*p2 + D*U2
    #
    # From Y: U1 = y11*p1 + y12*p2, U2 = y21*p1 + y22*p2
    # Solving for ABCD:
    #   A = -y22/y21, B = -1/y21, C = -det(Y)/y21, D = -y11/y21
    # where det(Y) = y11*y22 - y12*y21

    y11, y12, y21, y22 = y[0, 0], y[0, 1], y[1, 0], y[1, 1]

    if abs(y21) < 1e-30:
        # Fallback to identity (no transformation)
        return np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])

    det_y = y11 * y22 - y12 * y21
    a = -y22 / y21
    b = -1.0 / y21
    c_elem = -det_y / y21
    d = -y11 / y21

    return np.array([[a, b], [c_elem, d]], dtype=complex)


def _solve_nodal_pressures(y: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve p from y p = q for a small nodal system.

    Returns (p, z_mat) where z_mat = inv(y).
    """
    z_mat = np.linalg.inv(y)
    p = z_mat @ q
    return p, z_mat


def _voice_coil_impedance_from_model(
    *,
    frequency_hz: float,
    re_ohm: float,
    le_h: float,
    model: VoiceCoilImpedanceModel | None,
) -> complex:
    omega = 2 * np.pi * float(frequency_hz)
    z = float(re_ohm) + 1j * omega * float(le_h)
    if model is None or model.kind == "Ideal":
        return z
    if model.kind == "EddyLoss":
        f = float(frequency_hz)
        f0 = max(1e-9, float(model.f_corner_hz))
        x = f / f0
        r_eddy = float(model.r_eddy_max_ohm) * (x * x) / (1.0 + x * x)
        return z + r_eddy
    return z


# ---------------------------------------------------------------------------
# Pure helper functions for separation of concerns
# ---------------------------------------------------------------------------


def compute_blocked_electrical_impedance(
    *,
    frequency_hz: float,
    re_ohm: float,
    le_h: float,
    eddy_r_max_ohm: float = 0.0,
    eddy_f_corner_hz: float = 1000.0,
) -> complex:
    """
    Compute blocked voice-coil electrical impedance (no motional contribution).

    This is the electrical impedance with the voice coil mechanically blocked:
        Z_blocked = Re + jωLe + R_eddy(f)

    The eddy-current loss model adds:
        R_eddy(f) = R_max * (x² / (1 + x²)), where x = f / f_corner

    Parameters
    ----------
    frequency_hz : float
        Frequency in Hz.
    re_ohm : float
        DC resistance in ohms.
    le_h : float
        Voice coil inductance in henries.
    eddy_r_max_ohm : float, optional
        Maximum eddy-current resistance (asymptotic at high frequency).
    eddy_f_corner_hz : float, optional
        Corner frequency for eddy-current rolloff.

    Returns
    -------
    complex
        Blocked electrical impedance in ohms.
    """
    omega = 2 * np.pi * float(frequency_hz)
    z = float(re_ohm) + 1j * omega * float(le_h)

    if eddy_r_max_ohm > 0:
        f = float(frequency_hz)
        f0 = max(1e-9, float(eddy_f_corner_hz))
        x = f / f0
        r_eddy = float(eddy_r_max_ohm) * (x * x) / (1.0 + x * x)
        z = z + r_eddy

    return z


def compute_mechanical_impedance(
    *,
    frequency_hz: float,
    mms_kg: float,
    cms_m_per_n: float,
    rms_ns_per_m: float,
) -> complex:
    """
    Compute mechanical impedance of the moving system.

    Z_mech = Rms + jωMms + 1/(jωCms)

    Parameters
    ----------
    frequency_hz : float
        Frequency in Hz.
    mms_kg : float
        Moving mass in kg.
    cms_m_per_n : float
        Mechanical compliance in m/N.
    rms_ns_per_m : float
        Mechanical resistance in N·s/m.

    Returns
    -------
    complex
        Mechanical impedance in N·s/m.
    """
    omega = 2 * np.pi * float(frequency_hz)
    if omega <= 0:
        return float(rms_ns_per_m) + 0j
    return (
        float(rms_ns_per_m)
        + 1j * omega * float(mms_kg)
        + 1.0 / (1j * omega * float(cms_m_per_n))
    )


@dataclass(frozen=True)
class AcousticNodeResult:
    """Result from solving a two-node acoustic network."""

    p0: complex  # Pressure at V0 (front chamber) in Pa
    p1: complex  # Pressure at V1 (suspension chamber) in Pa
    u_front: complex  # Volume velocity into front chain in m³/s
    z_front: complex  # Input impedance looking into front chain in Pa·s/m³
    z_mat: np.ndarray  # 2x2 nodal impedance matrix


def solve_two_node_acoustic_network(
    *,
    q0: complex,
    q1: complex,
    y_front: complex,
    y_v0: complex,
    y_v1: complex,
    y_coupling: np.ndarray,
    z_front: complex,
) -> AcousticNodeResult:
    """
    Solve the two-node (V0, V1) acoustic network for pressures and flow.

    This implements the nodal admittance equation: Y·p = q

    where:
    - q0: volume velocity injected at V0 (from diaphragm front)
    - q1: volume velocity injected at V1 (from suspension radiator)
    - y_front: admittance from V0 to the front chain (1/Z_front)
    - y_v0: shunt admittance at V0 (front chamber compliance)
    - y_v1: shunt admittance at V1 (suspension chamber compliance)
    - y_coupling: 2x2 admittance matrix for V0-V1 coupling element

    Parameters
    ----------
    q0 : complex
        Volume velocity source at V0 in m³/s.
    q1 : complex
        Volume velocity source at V1 in m³/s.
    y_front : complex
        Admittance into the front acoustic chain (m³/s/Pa).
    y_v0 : complex
        Shunt admittance at V0 node (front chamber compliance).
    y_v1 : complex
        Shunt admittance at V1 node (suspension chamber compliance).
    y_coupling : np.ndarray
        2x2 admittance matrix for the V0-V1 coupling element.
    z_front : complex
        Impedance into the front chain (Pa·s/m³).

    Returns
    -------
    AcousticNodeResult
        Nodal pressures, front flow, and impedance matrix.
    """
    # Build nodal admittance matrix
    y = np.array(
        [[y_front + y_v0, 0.0 + 0j], [0.0 + 0j, y_v1]], dtype=complex
    ) + y_coupling

    q = np.array([q0, q1], dtype=complex)

    # Fast path: no V1 coupling -> single-node solve for V0
    if abs(y[0, 1]) < 1e-30 and abs(y[1, 0]) < 1e-30 and abs(y[1, 1]) < 1e-30:
        if abs(y[0, 0]) < 1e-30:
            p0 = 0.0 + 0j
        else:
            p0 = q0 / y[0, 0]
        p1 = 0.0 + 0j
        u_front = p0 / z_front if not np.isinf(z_front) else 0.0 + 0j
        z_mat = np.array(
            [
                [1.0 / y[0, 0] if abs(y[0, 0]) > 1e-30 else np.inf + 0j, 0.0 + 0j],
                [0.0 + 0j, np.inf + 0j],
            ]
        )
        return AcousticNodeResult(
            p0=p0, p1=p1, u_front=complex(u_front), z_front=z_front, z_mat=z_mat
        )

    # General case: solve coupled system
    try:
        p, z_mat = _solve_nodal_pressures(y, q)
    except Exception:
        # Fallback: treat as uncoupled shunt at V0 only
        y0 = y_front + y_v0
        if abs(y0) < 1e-30:
            p0 = 0.0 + 0j
        else:
            p0 = q0 / y0
        p1 = 0.0 + 0j
        u_front = p0 / z_front if not np.isinf(z_front) else 0.0 + 0j
        z_mat = np.array(
            [
                [1.0 / y0 if abs(y0) > 1e-30 else np.inf + 0j, 0.0 + 0j],
                [0.0 + 0j, np.inf + 0j],
            ]
        )
        return AcousticNodeResult(
            p0=p0, p1=p1, u_front=complex(u_front), z_front=z_front, z_mat=z_mat
        )

    p0 = complex(p[0])
    p1 = complex(p[1])
    u_front = p0 / z_front if not np.isinf(z_front) else 0.0 + 0j

    return AcousticNodeResult(
        p0=p0, p1=p1, u_front=complex(u_front), z_front=z_front, z_mat=z_mat
    )


@dataclass(frozen=True)
class CompressionDriverExcitation:
    """Electrical drive conditions."""

    voltage_rms: float = 2.83


@dataclass(frozen=True)
class CompressionDriverNetworkOptions:
    """Numerical options for waveguide discretization inside the driver."""

    waveguide_segments: int = 40
    waveguide_matrix_method: WaveguideMatrixMethod = "CascadedTubes"
    duct_loss_model: Literal["Lossless", "Kirchhoff"] = "Lossless"


class CompressionDriverNetwork:
    """
    Electro-acoustic network for a Generic25-defined compression driver system.

    Model:
    - Standard electro-mechanical driver (Re, Le, Bl, Mms, Cms, Rms)
    - Rear load: compliance from rear volume
    - Front load: cascaded duct + waveguide sections terminated by external load
    - Coupling between mechanical and acoustic via diaphragm area Sd

    Sign convention:
    - Acoustic "voltage" is pressure, "current" is volume velocity.
    - Positive diaphragm velocity produces positive volume velocity into the front network.
    """

    def __init__(
        self,
        system: Generic25SystemConfig | CompressionDriverConfig,
        medium: AcousticMedium = AcousticMedium(),
        options: CompressionDriverNetworkOptions = CompressionDriverNetworkOptions(),
    ):
        self.system = system
        self.medium = medium
        self.options = options

        if isinstance(system, CompressionDriverConfig):
            self._sd = _area_from_diameter(system.driver.diaphragm_diameter_m)
        else:
            self._sd = _area_from_diameter(system.driver.diaphragm_diameter_m)

        # Optional suspension area (S1)
        self._s1 = 0.0
        if isinstance(system, CompressionDriverConfig):
            if system.suspension_diameter_m is not None and float(system.suspension_diameter_m) > 0:
                self._s1 = _area_from_diameter(float(system.suspension_diameter_m))

        # Determine exit (throat) area from the last waveguide (if present)
        exit_area = None
        if isinstance(system, CompressionDriverConfig):
            if system.exit_cone is not None:
                exit_area = _area_from_diameter(system.exit_cone.mouth_diameter_m)
            elif system.phase_plug is not None:
                exit_area = _area_from_diameter(system.phase_plug.mouth_diameter_m)
        else:
            if system.waveguides:
                last = system.waveguides[-1]
                if last.mouth_diameter_m is not None:
                    exit_area = _area_from_diameter(last.mouth_diameter_m)
        self._exit_area = exit_area

    @property
    def diaphragm_area_m2(self) -> float:
        return self._sd

    @property
    def suspension_area_m2(self) -> float:
        return float(self._s1)

    @property
    def exit_area_m2(self) -> Optional[float]:
        return self._exit_area

    def rear_acoustic_impedance(self, frequency_hz: float) -> complex:
        """Rear volume impedance to ground (compliance)."""
        if isinstance(self.system, CompressionDriverConfig):
            rear = self.system.rear_volume
            if rear is None:
                return np.inf + 0j
            v = rear.volume_m3
        else:
            if self.system.rear_volume is None:
                return np.inf + 0j
            v = self.system.rear_volume.volume_m3
        if v <= 0:
            return np.inf + 0j
        omega = 2 * np.pi * frequency_hz
        if omega <= 0:
            return np.inf + 0j
        c = self.medium.c
        rho = self.medium.rho
        compliance = v / (rho * c * c)  # m^5/N
        # Zc = 1/(jωC) in acoustic domain (Pa / (m^3/s))
        return 1.0 / (1j * omega * compliance)

    def _tube_abcd_section(self, *, frequency_hz: float, diameter_m: float, length_m: float) -> np.ndarray:
        """ABCD for a circular duct with optional thermo-viscous losses."""
        omega = 2 * np.pi * float(frequency_hz)
        if omega <= 0 or float(diameter_m) <= 0 or float(length_m) <= 0:
            return np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])

        radius = 0.5 * float(diameter_m)
        area = _area_from_diameter(float(diameter_m))

        if self.options.duct_loss_model == "Kirchhoff":
            k_eff, zc_eff = _kirchhoff_circular_duct_characteristics(
                omega=omega,
                rho=float(self.medium.rho),
                c=float(self.medium.c),
                radius_m=radius,
                gamma=float(self.medium.gamma),
                prandtl=float(self.medium.prandtl),
                dynamic_viscosity_pa_s=float(self.medium.dynamic_viscosity_pa_s),
            )
            return _tube_abcd_from_k_zc(k=k_eff, zc=zc_eff, length=float(length_m))

        k = omega / float(self.medium.c)
        zc = float(self.medium.rho) * float(self.medium.c) / float(area)
        return _tube_abcd_from_k_zc(k=k, zc=zc, length=float(length_m))

    def _tube_y_params_section(self, *, frequency_hz: float, diameter_m: float, length_m: float) -> np.ndarray:
        """Y-params for a circular duct with optional thermo-viscous losses."""
        omega = 2 * np.pi * float(frequency_hz)
        if omega <= 0 or float(diameter_m) <= 0 or float(length_m) <= 0:
            return np.zeros((2, 2), dtype=complex)

        radius = 0.5 * float(diameter_m)
        area = _area_from_diameter(float(diameter_m))

        if self.options.duct_loss_model == "Kirchhoff":
            k_eff, zc_eff = _kirchhoff_circular_duct_characteristics(
                omega=omega,
                rho=float(self.medium.rho),
                c=float(self.medium.c),
                radius_m=radius,
                gamma=float(self.medium.gamma),
                prandtl=float(self.medium.prandtl),
                dynamic_viscosity_pa_s=float(self.medium.dynamic_viscosity_pa_s),
            )
            return _tube_y_params_from_k_zc(k=k_eff, zc=zc_eff, length=float(length_m))

        k = omega / float(self.medium.c)
        zc = float(self.medium.rho) * float(self.medium.c) / float(area)
        return _tube_y_params_from_k_zc(k=k, zc=zc, length=float(length_m))

    @staticmethod
    def _eval_external_load(frequency_hz: float, z_external: complex | Callable[[float], complex]) -> complex:
        if callable(z_external):
            return complex(z_external(float(frequency_hz)))
        return complex(z_external)

    def _suspension_volume_impedance(self, frequency_hz: float) -> complex:
        """Suspension chamber impedance to ground (compliance)."""
        if not isinstance(self.system, CompressionDriverConfig):
            return np.inf + 0j
        v = self.system.suspension_volume_m3
        if v is None or float(v) <= 0:
            return np.inf + 0j
        omega = 2 * np.pi * frequency_hz
        if omega <= 0:
            return np.inf + 0j
        compliance = float(v) / (self.medium.rho * self.medium.c * self.medium.c)
        return 1.0 / (1j * omega * compliance)

    def _front_volume_impedance(self, frequency_hz: float) -> complex:
        """Front compression chamber impedance to ground (compliance V0)."""
        if not isinstance(self.system, CompressionDriverConfig):
            return np.inf + 0j
        v = self.system.front_volume_m3
        if v is None or float(v) <= 0:
            return np.inf + 0j
        omega = 2 * np.pi * frequency_hz
        if omega <= 0:
            return np.inf + 0j
        compliance = float(v) / (self.medium.rho * self.medium.c * self.medium.c)
        return 1.0 / (1j * omega * compliance)

    def _voice_coil_slit_impedance(self, frequency_hz: float) -> complex:
        """Series impedance between V0 (front chamber) and V1 (suspension chamber)."""
        if not isinstance(self.system, CompressionDriverConfig):
            return np.inf + 0j
        r = self.system.voice_coil_slit_resistance_pa_s_per_m3
        m = self.system.voice_coil_slit_mass_pa_s2_per_m3
        if r is None and m is None:
            return np.inf + 0j
        omega = 2 * np.pi * frequency_hz
        r_val = float(r) if r is not None else 0.0
        m_val = float(m) if m is not None else 0.0
        return r_val + 1j * omega * m_val

    def _voice_coil_slit_admittance_matrix(self, frequency_hz: float) -> np.ndarray:
        """
        Admittance matrix Y for the V0-V1 coupling element.

        Returns Y such that [U0; U1] = Y [p0; p1].
        """
        if not isinstance(self.system, CompressionDriverConfig):
            return np.zeros((2, 2), dtype=complex)

        omega = 2 * np.pi * float(frequency_hz)
        if omega <= 0:
            return np.zeros((2, 2), dtype=complex)

        if self.system.voice_coil_slit_model == "Tube":
            d = self.system.voice_coil_slit_diameter_m
            l = self.system.voice_coil_slit_length_m
            if d is None or l is None or float(d) <= 0 or float(l) <= 0:
                return np.zeros((2, 2), dtype=complex)
            return self._tube_y_params_section(frequency_hz=float(frequency_hz), diameter_m=float(d), length_m=float(l))

        z = self._voice_coil_slit_impedance(frequency_hz)
        if np.isinf(z):
            return np.zeros((2, 2), dtype=complex)
        y = 1.0 / z
        return np.array([[y, -y], [-y, y]], dtype=complex)

    def _rear_radiation_impedance(self, frequency_hz: float) -> complex:
        """
        Approximate rear radiation acoustic impedance (Pa·s/m³).

        Uses a piston-in-baffle impedance when available (SciPy); otherwise uses
        a low-frequency approximation for all frequencies.
        """
        if not isinstance(self.system, CompressionDriverConfig):
            return np.inf + 0j
        mode = self.system.rear_radiation_mode
        if mode == "None":
            return np.inf + 0j

        d = self.system.rear_radiator_diameter_m
        if d is None or float(d) <= 0:
            d = self.system.driver.diaphragm_diameter_m
        radius = 0.5 * float(d)
        area = _area_from_diameter(float(d))

        rho = float(self.medium.rho)
        c = float(self.medium.c)
        f = float(frequency_hz)
        k = 2 * np.pi * f / c
        x = 2.0 * k * radius
        if x < 1e-6:
            r1 = (x * x) / 8.0
            x1 = 4.0 * x / (3.0 * np.pi)
            z_mech = rho * c * area * (r1 + 1j * x1)
        else:
            try:
                from scipy.special import jv, struve  # type: ignore

                j1 = jv(1, x)
                h1 = struve(1, x)
                r1 = 1.0 - 2.0 * j1 / x
                x1 = 2.0 * h1 / x
                z_mech = rho * c * area * (r1 + 1j * x1)
            except Exception:
                # Fallback: keep the low-frequency approximation even out of range.
                r1 = min(1.0, (x * x) / 8.0)
                x1 = 4.0 * x / (3.0 * np.pi)
                z_mech = rho * c * area * (r1 + 1j * x1)

        if mode == "FreeSpaceApprox":
            z_mech *= 0.5

        # Convert mechanical impedance (F/v) to acoustic impedance (p/U): Z = Z_mech / S^2
        if area <= 0:
            return np.inf + 0j
        return z_mech / (area * area)

    def _rear_acoustic_impedance_total(self, frequency_hz: float) -> complex:
        """Rear acoustic load impedance to ground (rear volume || rear radiation)."""
        z_vol = self.rear_acoustic_impedance(frequency_hz)
        z_rad = self._rear_radiation_impedance(frequency_hz)
        return _parallel(z_vol, z_rad)

    def front_radiation_impedance(self, frequency_hz: float) -> complex:
        """
        Front (exit) radiation acoustic impedance for free-radiation validation mode.

        This implements the Panzer validation workflow where the driver is tested
        without a waveguide attached, revealing internal resonances clearly due
        to minimal radiation damping.

        Parameters
        ----------
        frequency_hz : float
            Frequency in Hz.

        Returns
        -------
        complex
            Acoustic radiation impedance at the front aperture (Pa·s/m³).
            Returns np.inf if front_radiation_mode is "None" or "BEMLoad".
        """
        if not isinstance(self.system, CompressionDriverConfig):
            return np.inf + 0j

        mode = self.system.front_radiation_mode
        if mode in ("None", "BEMLoad"):
            return np.inf + 0j

        # Determine front radiator diameter
        d = self.system.front_radiator_diameter_m
        if d is None or float(d) <= 0:
            # Use exit area from phase plug or exit cone
            if self._exit_area is not None and self._exit_area > 0:
                d = _diameter_from_area(self._exit_area)
            else:
                # Fallback to diaphragm diameter
                d = self.system.driver.diaphragm_diameter_m

        radius = 0.5 * float(d)
        area = _area_from_diameter(float(d))

        rho = float(self.medium.rho)
        c = float(self.medium.c)
        f = float(frequency_hz)
        k = 2 * np.pi * f / c
        x = 2.0 * k * radius

        # Piston-in-infinite-baffle radiation impedance
        if x < 1e-6:
            r1 = (x * x) / 8.0
            x1 = 4.0 * x / (3.0 * np.pi)
            z_mech = rho * c * area * (r1 + 1j * x1)
        else:
            try:
                from scipy.special import jv, struve  # type: ignore

                j1 = jv(1, x)
                h1 = struve(1, x)
                r1 = 1.0 - 2.0 * j1 / x
                x1 = 2.0 * h1 / x
                z_mech = rho * c * area * (r1 + 1j * x1)
            except Exception:
                r1 = min(1.0, (x * x) / 8.0)
                x1 = 4.0 * x / (3.0 * np.pi)
                z_mech = rho * c * area * (r1 + 1j * x1)

        # Convert mechanical impedance to acoustic impedance: Z = Z_mech / S^2
        if area <= 0:
            return np.inf + 0j
        return z_mech / (area * area)

    def get_effective_external_load(self, frequency_hz: float, z_external: complex = np.inf + 0j) -> complex:
        """
        Get the effective external acoustic load impedance.

        If front_radiation_mode is "FreeRadiation", uses the computed front
        radiation impedance instead of z_external.

        Parameters
        ----------
        frequency_hz : float
            Frequency in Hz.
        z_external : complex
            External load impedance (used if front_radiation_mode is not "FreeRadiation").

        Returns
        -------
        complex
            Effective acoustic load impedance at the driver exit.
        """
        if isinstance(self.system, CompressionDriverConfig):
            if self.system.front_radiation_mode == "FreeRadiation":
                return self.front_radiation_impedance(frequency_hz)
        return z_external

    def _voice_coil_impedance(self, frequency_hz: float) -> complex:
        """Electrical impedance of the voice coil (including optional eddy losses)."""
        d = self.system.driver
        model: VoiceCoilImpedanceModel | None = None
        if isinstance(self.system, CompressionDriverConfig):
            model = self.system.driver.voice_coil_model
        return _voice_coil_impedance_from_model(
            frequency_hz=float(frequency_hz),
            re_ohm=float(d.re_ohm),
            le_h=float(d.le_h),
            model=model,
        )

    def _apply_output_weighting(self, frequency_hz: float, u: complex) -> complex:
        """
        Apply optional frequency-dependent output weighting to volume velocities.

        This is intended as a pragmatic high-frequency correction hook (e.g. bending-mode
        shaping) without changing the electro-mechanical impedance solution.
        """
        if not isinstance(self.system, CompressionDriverConfig):
            return u
        w = self.system.output_velocity_weighting
        if w is None:
            return u
        gain_db = _interp_gain_db(w, float(frequency_hz))
        gain = 10.0 ** (gain_db / 20.0)
        return complex(u) * float(gain)

    def _front_abcd(self, frequency_hz: float) -> np.ndarray:
        """ABCD from diaphragm front node to exit node."""
        mats: list[np.ndarray] = []

        if isinstance(self.system, CompressionDriverConfig):
            duct = self.system.front_duct
            if duct is not None and duct.length_m > 0:
                mats.append(
                    self._tube_abcd_section(
                        frequency_hz=float(frequency_hz),
                        diameter_m=float(duct.diameter_m),
                        length_m=float(duct.length_m),
                    )
                )
        else:
            if self.system.front_duct is not None and self.system.front_duct.length_m > 0:
                mats.append(
                    self._tube_abcd_section(
                        frequency_hz=float(frequency_hz),
                        diameter_m=float(self.system.front_duct.diameter_m),
                        length_m=float(self.system.front_duct.length_m),
                    )
                )

        if isinstance(self.system, CompressionDriverConfig):
            if self.system.phase_plug is not None:
                mats.extend(self._waveguide_cfg_abcd_segments(self.system.phase_plug, frequency_hz))
            if self.system.exit_cone is not None:
                mats.extend(self._waveguide_cfg_abcd_segments(self.system.exit_cone, frequency_hz))
        else:
            if self.system.waveguides:
                for wg in self.system.waveguides:
                    mats.extend(self._waveguide_abcd_segments(wg, frequency_hz))

        return _cascade(mats)

    def _waveguide_cfg_abcd_segments(self, section, frequency_hz: float) -> list[np.ndarray]:
        """
        Compute ABCD matrix(es) for a waveguide section.

        Supports two methods controlled by `options.waveguide_matrix_method`:
        - "CascadedTubes": Discretize into N uniform tube segments (default)
        - "AnalyticalConical": Use Panzer eq. 8 for conical waveguides
        """
        if section.length_m <= 0:
            return []

        if hasattr(section, "throat_area_m2") and section.throat_area_m2 is not None:
            throat_area = float(section.throat_area_m2)
            d1 = _diameter_from_area(throat_area)
        else:
            d1 = getattr(section, "throat_diameter_m", None)
            throat_area = _area_from_diameter(float(d1)) if d1 and float(d1) > 0 else 0.0

        d2 = section.mouth_diameter_m
        mouth_area = _area_from_diameter(float(d2)) if d2 and float(d2) > 0 else 0.0

        if d1 is None or d2 is None or d1 <= 0 or d2 <= 0:
            if d2 and d2 > 0:
                return [self._tube_abcd_section(frequency_hz=float(frequency_hz), diameter_m=float(d2), length_m=float(section.length_m))]
            return []

        # Use analytical conical matrix if configured
        if self.options.waveguide_matrix_method == "AnalyticalConical":
            abcd = conical_waveguide_abcd_matrix(
                frequency_hz=float(frequency_hz),
                rho=float(self.medium.rho),
                c=float(self.medium.c),
                throat_area_m2=throat_area,
                mouth_area_m2=mouth_area,
                length_m=float(section.length_m),
            )
            return [abcd]

        # Default: cascaded uniform tubes
        n = max(1, int(self.options.waveguide_segments))
        lengths = np.full(n, section.length_m / n)
        t = (np.arange(n) + 0.5) / n
        diam = d1 + (d2 - d1) * t
        return [
            self._tube_abcd_section(frequency_hz=float(frequency_hz), diameter_m=float(di), length_m=float(li))
            for di, li in zip(diam, lengths)
        ]

    def _waveguide_abcd_segments(self, wg: Generic25WaveguideConfig, frequency_hz: float) -> list[np.ndarray]:
        """
        Discretize a waveguide section into short uniform tube segments.

        Supports conical-like sections by interpolating diameter linearly.
        """
        n = max(1, int(self.options.waveguide_segments))
        if wg.length_m <= 0:
            return []

        # Determine throat/mouth diameters
        d1 = wg.throat_diameter_m
        if d1 is None and wg.throat_area_m2 is not None:
            d1 = _diameter_from_area(wg.throat_area_m2)
        d2 = wg.mouth_diameter_m

        if d1 is None or d2 is None:
            # Fall back: treat as uniform tube with exit diameter if available
            d = d2 or d1 or 0.0
            if d > 0:
                return [self._tube_abcd_section(frequency_hz=float(frequency_hz), diameter_m=float(d), length_m=float(wg.length_m))]
            return []

        lengths = np.full(n, wg.length_m / n)
        # Diameters at segment midpoints
        t = (np.arange(n) + 0.5) / n
        diam = d1 + (d2 - d1) * t
        mats: list[np.ndarray] = []
        for di, li in zip(diam, lengths):
            mats.append(self._tube_abcd_section(frequency_hz=float(frequency_hz), diameter_m=float(di), length_m=float(li)))
        return mats

    def _acoustic_node_pressures_and_front_flow(
        self,
        frequency_hz: float,
        *,
        diaphragm_velocity_rms: complex,
        z_external: complex,
    ) -> tuple[complex, complex, complex, complex, np.ndarray]:
        """
        Solve coupled front/suspension acoustic nodes (V0 and V1).

        This method delegates to the pure function `solve_two_node_acoustic_network()`
        after computing the admittances from the driver configuration.

        Returns
        -------
        p0 : complex
            Pressure at V0 node (front chamber) in Pa.
        p1 : complex
            Pressure at V1 node (suspension chamber) in Pa.
        u_front_in : complex
            Volume velocity into the front two-port (port 1) in m³/s RMS.
        z_front : complex
            Input impedance at V0 into the front chain in Pa·s/m³.
        z_mat : np.ndarray
            2x2 node impedance matrix mapping [q0,q1] -> [p0,p1].
        """
        # Volume velocity sources from diaphragm and suspension
        q0 = self._sd * diaphragm_velocity_rms
        q1 = self._s1 * diaphragm_velocity_rms if self._s1 > 0 else 0.0 + 0.0j

        # Front chain impedance
        front_abcd = self._front_abcd(frequency_hz)
        z_front = _abcd_to_zin(front_abcd, z_external)

        # Convert impedances to admittances
        z_v0 = self._front_volume_impedance(frequency_hz)
        z_v1 = self._suspension_volume_impedance(frequency_hz)
        y_front = 0.0 + 0.0j if np.isinf(z_front) else 1.0 / z_front
        y_v0 = 0.0 + 0.0j if np.isinf(z_v0) else 1.0 / z_v0
        y_v1 = 0.0 + 0.0j if np.isinf(z_v1) else 1.0 / z_v1
        y_coupling = self._voice_coil_slit_admittance_matrix(frequency_hz)

        # Delegate to pure function
        result = solve_two_node_acoustic_network(
            q0=q0,
            q1=q1,
            y_front=y_front,
            y_v0=y_v0,
            y_v1=y_v1,
            y_coupling=y_coupling,
            z_front=z_front,
        )

        return result.p0, result.p1, result.u_front, result.z_front, result.z_mat

    def front_input_impedance(self, frequency_hz: float, z_external: complex | Callable[[float], complex]) -> complex:
        """Input impedance at diaphragm front node looking into the front chain."""
        f = float(frequency_hz)
        z_ext_raw = self._eval_external_load(f, z_external)
        z_ext = self.get_effective_external_load(f, z_ext_raw)
        abcd = self._front_abcd(f)
        return _abcd_to_zin(abcd, z_ext)

    def solve_volume_velocity(
        self,
        frequency_hz: float,
        excitation: CompressionDriverExcitation = CompressionDriverExcitation(),
        z_external: complex | Callable[[float], complex] = np.inf + 0j,
    ) -> complex:
        """
        Return complex volume velocity at the driver exit (throat) at `frequency_hz`.

        Parameters
        ----------
        excitation : CompressionDriverExcitation
            RMS input voltage.
        z_external : complex
            Acoustic load impedance at the driver exit (Pa·s/m^3).
        """
        f = float(frequency_hz)
        z_ext_raw = self._eval_external_load(f, z_external)
        z_ext = self.get_effective_external_load(f, z_ext_raw)

        omega = 2 * np.pi * f
        if omega <= 0:
            return 0.0 + 0.0j

        z_rear = self._rear_acoustic_impedance_total(f)

        # Mechanical impedance of moving system (N·s/m)
        d = self.system.driver
        z_mech = d.rms_ns_per_m + 1j * omega * d.mms_kg + 1.0 / (1j * omega * d.cms_m_per_n)

        # Electrical impedance (ohm)
        z_e = self._voice_coil_impedance(f)

        # Compute total mechanical load impedance from acoustics under v=1 m/s
        v_trial = 1.0 + 0.0j
        p0_trial, p1_trial, _, _, _ = self._acoustic_node_pressures_and_front_flow(
            f, diaphragm_velocity_rms=v_trial, z_external=z_ext
        )
        z_mech_load_front = (p0_trial * self._sd + p1_trial * self._s1) / v_trial
        q_rear = self._sd * v_trial
        p_rear = z_rear * q_rear if not np.isinf(z_rear) else 0.0 + 0.0j
        z_mech_load_rear = (p_rear * self._sd) / v_trial
        z_mech_load = z_mech_load_front + z_mech_load_rear

        # Motional impedance reflected into electrical domain: Bl^2 / (Z_mech + Z_load)
        denom = z_mech + z_mech_load
        if abs(denom) < 1e-30:
            return 0.0 + 0.0j
        z_motional = (d.bl_tm ** 2) / denom

        # Current from applied voltage
        v_in = excitation.voltage_rms
        i = v_in / (z_e + z_motional)

        # Diaphragm velocity (m/s)
        v = d.bl_tm * i / denom

        # Volume velocity into the front chain (port 1) from coupled acoustics
        _, _, u_front_in, _, _ = self._acoustic_node_pressures_and_front_flow(
            f, diaphragm_velocity_rms=v, z_external=z_ext
        )
        u_front_in = self._apply_output_weighting(frequency_hz, u_front_in)

        # Volume velocity at the external load (driver exit / throat)
        if np.isinf(z_ext):
            return 0.0 + 0.0j
        front_abcd = self._front_abcd(f)
        return _abcd_input_to_load_volume_velocity(front_abcd, z_ext, u_front_in)

    def solve_with_metrics(
        self,
        frequency_hz: float,
        excitation: CompressionDriverExcitation = CompressionDriverExcitation(),
        z_external: complex | Callable[[float], complex] = np.inf + 0j,
    ) -> dict:
        """
        Solve and return full metrics including electrical impedance and excursion.

        Parameters
        ----------
        frequency_hz : float
            Frequency in Hz.
        excitation : CompressionDriverExcitation
            RMS input voltage.
        z_external : complex
            Acoustic load impedance at the driver exit (Pa·s/m^3).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'volume_velocity': complex volume velocity (m³/s RMS)
            - 'diaphragm_velocity': complex diaphragm velocity (m/s RMS)
            - 'electrical_impedance': complex electrical impedance (ohms)
            - 'excursion_peak_mm': peak excursion in mm
            - 'current': complex current (A RMS)
        """
        f = float(frequency_hz)
        z_ext_raw = self._eval_external_load(f, z_external)
        z_ext = self.get_effective_external_load(f, z_ext_raw)

        omega = 2 * np.pi * f
        if omega <= 0:
            return {
                'volume_velocity': 0.0 + 0.0j,
                'volume_velocity_diaphragm': 0.0 + 0.0j,
                'volume_velocity_suspension': 0.0 + 0.0j,
                'volume_velocity_front': 0.0 + 0.0j,
                'diaphragm_velocity': 0.0 + 0.0j,
                'pressure_v0_pa': 0.0 + 0.0j,
                'pressure_v1_pa': 0.0 + 0.0j,
                'electrical_impedance': self.system.driver.re_ohm + 0.0j,
                'excursion_peak_mm': 0.0,
                'current': 0.0 + 0.0j,
            }

        z_rear = self._rear_acoustic_impedance_total(f)

        # Mechanical impedance of moving system (N·s/m)
        d = self.system.driver
        z_mech = d.rms_ns_per_m + 1j * omega * d.mms_kg + 1.0 / (1j * omega * d.cms_m_per_n)

        # Electrical blocked impedance (ohm)
        z_e = self._voice_coil_impedance(f)

        # Compute total mechanical load impedance from acoustics under v=1 m/s
        v_trial = 1.0 + 0.0j
        p0_trial, p1_trial, _, _, _ = self._acoustic_node_pressures_and_front_flow(
            f, diaphragm_velocity_rms=v_trial, z_external=z_ext
        )
        z_mech_load_front = (p0_trial * self._sd + p1_trial * self._s1) / v_trial
        q_rear = self._sd * v_trial
        p_rear = z_rear * q_rear if not np.isinf(z_rear) else 0.0 + 0.0j
        z_mech_load_rear = (p_rear * self._sd) / v_trial
        z_mech_load = z_mech_load_front + z_mech_load_rear

        # Motional impedance reflected into electrical domain: Bl^2 / (Z_mech + Z_load)
        denom = z_mech + z_mech_load
        if abs(denom) < 1e-30:
            return {
                'volume_velocity': 0.0 + 0.0j,
                'volume_velocity_diaphragm': 0.0 + 0.0j,
                'volume_velocity_suspension': 0.0 + 0.0j,
                'volume_velocity_front': 0.0 + 0.0j,
                'diaphragm_velocity': 0.0 + 0.0j,
                'pressure_v0_pa': 0.0 + 0.0j,
                'pressure_v1_pa': 0.0 + 0.0j,
                'electrical_impedance': z_e,
                'excursion_peak_mm': 0.0,
                'current': 0.0 + 0.0j,
            }

        z_motional = (d.bl_tm ** 2) / denom

        # Total electrical input impedance
        z_in = z_e + z_motional

        # Current from applied voltage
        v_in = excitation.voltage_rms
        i_rms = v_in / z_in

        # Diaphragm velocity (m/s RMS)
        v_rms = d.bl_tm * i_rms / denom

        # Peak velocity and excursion
        v_peak = abs(v_rms) * np.sqrt(2.0)
        x_peak = v_peak / omega  # peak displacement in meters
        x_peak_mm = x_peak * 1000.0

        # Volume velocities injected by the moving assembly
        u_diaphragm = self._sd * v_rms
        u_susp = self._s1 * v_rms if self._s1 > 0 else 0.0 + 0.0j

        front_abcd = self._front_abcd(f)
        p0, p1, u_front_in, _, _ = self._acoustic_node_pressures_and_front_flow(
            f, diaphragm_velocity_rms=v_rms, z_external=z_ext
        )
        u_front_in = self._apply_output_weighting(frequency_hz, u_front_in)

        if np.isinf(z_ext):
            u_throat = 0.0 + 0.0j
        else:
            u_throat = _abcd_input_to_load_volume_velocity(front_abcd, z_ext, u_front_in)

        return {
            'volume_velocity': u_throat,
            'volume_velocity_diaphragm': u_diaphragm,
            'volume_velocity_suspension': u_susp,
            'volume_velocity_front': u_front_in,
            'diaphragm_velocity': v_rms,
            'pressure_v0_pa': p0,
            'pressure_v1_pa': p1,
            'electrical_impedance': z_in,
            'excursion_peak_mm': x_peak_mm,
            'current': i_rms,
        }
