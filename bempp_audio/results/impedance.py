"""
Radiation impedance computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import weakref
import numpy as np

from bempp_audio._optional import optional_import

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")

if TYPE_CHECKING:
    from bempp_audio.results.radiation_result import RadiationResult


_MASS_MATRIX_CACHE: "weakref.WeakKeyDictionary[object, object]" = weakref.WeakKeyDictionary()


def _mass_matrix(space):
    """Return (and cache) the L2 mass matrix for `space`."""
    if not BEMPP_AVAILABLE:
        raise ImportError("bempp_cl is required for mass matrix assembly")

    try:
        cached = _MASS_MATRIX_CACHE.get(space)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    identity = bempp.operators.boundary.sparse.identity(space, space, space)
    mass = identity.weak_form()
    try:
        _MASS_MATRIX_CACHE[space] = mass
    except TypeError:
        pass
    return mass


class RadiationImpedance:
    """
    Compute acoustic radiation impedance on the radiator surface.

    The radiation impedance characterizes how the acoustic medium
    loads the vibrating surface. It affects both the radiated power
    and the mechanical impedance seen by the driver.

    Parameters
    ----------
    result : RadiationResult
        The radiation solution.

    Attributes
    ----------
    mechanical : complex
        Mechanical radiation impedance Z_rad = F/v (N·s/m).
    specific : complex
        Specific acoustic impedance p/v (Pa·s/m).

    References
    ----------
    For a piston in an infinite baffle, the analytical radiation
    impedance is:
        Z_rad = ρ·c·S·(R1(2ka) + j·X1(2ka))

    where R1 and X1 are the radiation resistance and reactance
    functions, and a is the piston radius.
    """

    def __init__(self, result: "RadiationResult"):
        self.result = result
        self._mechanical = None
        self._specific = None

    def mechanical(self) -> complex:
        """
        Compute mechanical radiation impedance.

        Z_rad = F / v_avg = ∫p dS / v_avg

        For geometries with multiple domains (e.g., waveguide with throat + walls),
        only integrates over the active (vibrating) domain where velocity is non-zero.

        Returns
        -------
        complex
            Mechanical impedance in N·s/m.
        """
        if self._mechanical is not None:
            return self._mechanical

        space = self.result.surface_pressure.space

        # Get velocity
        velocity_gf = self.result.velocity.to_grid_function(space)

        # Get mass matrix for integration
        mass = _mass_matrix(space)

        # For DP0 space, identify active elements (where velocity != 0)
        v_coeffs = velocity_gf.coefficients
        active_mask = np.abs(v_coeffs) > 1e-15  # Elements with non-zero velocity

        # If all elements are active (simple piston), use full integration
        if np.all(active_mask):
            ones = np.ones(space.global_dof_count)
            area = np.real(ones @ (mass @ ones))
            v_integral = ones @ (mass @ v_coeffs)
            v_avg = v_integral / area
            p_coeffs = self.result.surface_pressure.coefficients
            force = ones @ (mass @ p_coeffs)
        else:
            # Multi-domain case: only integrate over active elements
            # Create mask vector (1 for active, 0 for inactive)
            mask = active_mask.astype(float)
            
            # Area of active region: ∫_active 1 dS
            area = np.real(mask @ (mass @ mask))
            
            # Average velocity over active region: ∫_active v dS / S_active
            v_integral = mask @ (mass @ v_coeffs)
            v_avg = v_integral / area if area > 1e-15 else 0.0
            
            # Force over active region: ∫_active p dS
            p_coeffs = self.result.surface_pressure.coefficients
            force = mask @ (mass @ p_coeffs)

        # Impedance
        if abs(v_avg) < 1e-20:
            self._mechanical = np.inf + 0j
        else:
            self._mechanical = force / v_avg

        return self._mechanical

    def specific(self) -> complex:
        """
        Compute specific acoustic impedance (average p/v).

        For geometries with multiple domains (e.g., waveguide with throat + walls),
        only integrates over the active (vibrating) domain where velocity is non-zero.

        Returns
        -------
        complex
            Specific impedance in Pa·s/m.
        """
        if self._specific is not None:
            return self._specific

        space = self.result.surface_pressure.space

        # Get velocity
        velocity_gf = self.result.velocity.to_grid_function(space)

        # Get mass matrix
        mass = _mass_matrix(space)

        # Identify active elements (where velocity != 0)
        v_coeffs = velocity_gf.coefficients
        active_mask = np.abs(v_coeffs) > 1e-15

        # If all elements are active, use full integration
        if np.all(active_mask):
            ones = np.ones(space.global_dof_count)
            area = np.real(ones @ (mass @ ones))
            p_avg = (ones @ (mass @ self.result.surface_pressure.coefficients)) / area
            v_avg = (ones @ (mass @ v_coeffs)) / area
        else:
            # Multi-domain: only integrate over active region
            mask = active_mask.astype(float)
            area = np.real(mask @ (mass @ mask))
            
            if area > 1e-15:
                p_avg = (mask @ (mass @ self.result.surface_pressure.coefficients)) / area
                v_avg = (mask @ (mass @ v_coeffs)) / area
            else:
                p_avg = 0.0
                v_avg = 0.0

        if abs(v_avg) < 1e-20:
            self._specific = np.inf + 0j
        else:
            self._specific = p_avg / v_avg

        return self._specific

    def acoustic(self) -> complex:
        """
        Compute acoustic input impedance Z = p_avg / U_avg over the active region.

        Returns
        -------
        complex
            Acoustic impedance in Pa·s/m³.
        """
        area = self.active_area()
        if area <= 1e-20:
            return np.inf + 0j
        return self.specific() / area

    def active_area(self) -> float:
        """
        Area of the active (v≠0) region used for impedance averaging.

        Returns
        -------
        float
            Active area in m².
        """
        space = self.result.surface_pressure.space
        velocity_gf = self.result.velocity.to_grid_function(space)
        mass = _mass_matrix(space)

        v_coeffs = velocity_gf.coefficients
        active_mask = np.abs(v_coeffs) > 1e-15
        mask = active_mask.astype(float)
        area = float(np.real(mask @ (mass @ mask)))
        return max(0.0, area)

    def radiation_resistance(self) -> float:
        """
        Real part of mechanical impedance (power dissipation).

        Returns
        -------
        float
            Radiation resistance in N·s/m.
        """
        return np.real(self.mechanical())

    def radiation_reactance(self) -> float:
        """
        Imaginary part of mechanical impedance (reactive loading).

        Returns
        -------
        float
            Radiation reactance in N·s/m.
        """
        return np.imag(self.mechanical())

    def radiation_mass(self) -> float:
        """
        Equivalent mass from radiation reactance.

        m_rad = X_rad / omega

        Returns
        -------
        float
            Radiation mass in kg.
        """
        omega = 2 * np.pi * self.result.frequency
        if omega < 1e-10:
            return 0.0
        return self.radiation_reactance() / omega

    @staticmethod
    def analytical_piston_in_baffle(
        radius: float,
        frequency: float,
        c: float = 343.0,
        rho: float = 1.225,
    ) -> complex:
        """
        Analytical radiation impedance for piston in infinite baffle.

        Uses the exact series solution for the radiation impedance
        of a circular piston set in an infinite rigid baffle.

        Parameters
        ----------
        radius : float
            Piston radius in meters.
        frequency : float
            Frequency in Hz.
        c : float
            Speed of sound.
        rho : float
            Air density.

        Returns
        -------
        complex
            Mechanical radiation impedance.

        Notes
        -----
        Z = rho * c * S * (R1(2ka) + j*X1(2ka))

        where:
        R1(x) = 1 - J1(x) / (x/2)
        X1(x) = H1(x) / (x/2)

        J1 is Bessel function of first kind, order 1
        H1 is Struve function of order 1
        """
        from scipy.special import jv, struve

        k = 2 * np.pi * frequency / c
        a = radius
        S = np.pi * radius**2
        ka = k * a
        x = 2 * ka

        if x < 1e-6:
            # Low frequency approximation
            R1 = x**2 / 8
            X1 = x / (3 * np.pi)
        else:
            # Bessel J1
            J1 = jv(1, x)
            # Struve H1
            H1 = struve(1, x)

            R1 = 1 - 2 * J1 / x
            X1 = 2 * H1 / x

        Z = rho * c * S * (R1 + 1j * X1)
        return Z

    def __repr__(self) -> str:
        return (
            f"RadiationImpedance(f={self.result.frequency:.1f}Hz, "
            f"Z={self.mechanical():.2e})"
        )
