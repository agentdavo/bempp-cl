"""
Container for radiation solution at a single frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import numpy as np

from bempp_audio._optional import optional_import
from bempp_audio.acoustic_reference import AcousticReference

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")

if TYPE_CHECKING:
    from bempp_audio.mesh import LoudspeakerMesh
    from bempp_audio.velocity import VelocityProfile


@dataclass
class RadiationResult:
    """
    Container for radiation solution at a single frequency.

    Stores the surface pressure solution and provides methods for
    computing derived quantities like field pressure, directivity,
    and radiated power.

    Attributes
    ----------
    frequency : float
        Frequency in Hz.
    wavenumber : complex
        Complex wavenumber k = ω/c.
    surface_pressure : bempp.GridFunction
        Surface pressure solution.
    solver_info : int
        GMRES convergence info (0 = success).
    iterations : int
        Number of GMRES iterations.
    mesh : LoudspeakerMesh
        The mesh used for the solution.
    velocity : VelocityProfile
        The velocity profile used.
    c : float
        Speed of sound.
    rho : float
        Air density.
    """

    frequency: float
    wavenumber: complex
    surface_pressure: object  # bempp.GridFunction
    solver_info: int
    iterations: int
    mesh: "LoudspeakerMesh"
    velocity: "VelocityProfile"
    c: float = 343.0
    rho: float = 1.225
    baffle: Optional[object] = None
    baffle_plane_z: float = 0.0
    reference: Optional[AcousticReference] = None
    _neumann_cache_key: Optional[tuple] = field(default=None, init=False, repr=False)
    _neumann_cache_gf: Optional[object] = field(default=None, init=False, repr=False)

    def neumann_grid_function(self, space=None):
        """
        Return Neumann boundary data grid function: ∂p/∂n = -iωρ vₙ.

        This is cached per `(space, frequency, rho, velocity)` to avoid repeated
        velocity projection and GridFunction construction during field/directivity
        evaluations.
        """
        if not BEMPP_AVAILABLE:
            raise ImportError("bempp_cl is required for Neumann data evaluation")

        if space is None:
            space = self.surface_pressure.space

        key = (id(space), float(self.frequency), float(self.rho), id(self.velocity))
        if self._neumann_cache_key == key and self._neumann_cache_gf is not None:
            return self._neumann_cache_gf

        omega = 2 * np.pi * float(self.frequency)
        velocity_gf = self.velocity.to_grid_function(space)
        neumann_coeffs = -1j * omega * float(self.rho) * velocity_gf.coefficients
        neumann_gf = bempp.GridFunction(space, coefficients=neumann_coeffs)

        self._neumann_cache_key = key
        self._neumann_cache_gf = neumann_gf
        return neumann_gf

    def pressure_at(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate pressure at arbitrary field points.

        Uses Green's representation formula with potential operators.

        Parameters
        ----------
        points : np.ndarray
            Field point positions, shape (3, n_points).

        Returns
        -------
        np.ndarray
            Complex pressure at each point, shape (n_points,).
        """
        from bempp_audio.results.pressure_field import PressureField

        pf = PressureField(self)
        return pf.at_points(points)

    def far_field(self, directions: np.ndarray) -> np.ndarray:
        """
        Evaluate far-field pattern at given directions.

        Parameters
        ----------
        directions : np.ndarray
            Unit direction vectors, shape (3, n_directions).

        Returns
        -------
        np.ndarray
            Far-field pattern values, shape (n_directions,).
        """
        from bempp_audio.results.directivity import DirectivityPattern

        dp = DirectivityPattern(self)
        return dp.far_field_at(directions)

    def directivity(self) -> "DirectivityPattern":
        """
        Get a DirectivityPattern object for this result.

        Returns
        -------
        DirectivityPattern
            Object for computing and visualizing directivity.
        """
        from bempp_audio.results.directivity import DirectivityPattern

        return DirectivityPattern(self)

    def radiation_impedance(self) -> "RadiationImpedance":
        """
        Compute radiation impedance.

        Returns
        -------
        RadiationImpedance
            Object containing mechanical and specific impedance.
        """
        from bempp_audio.results.impedance import RadiationImpedance

        return RadiationImpedance(self)

    def radiated_power(self) -> float:
        """
        Compute total radiated acoustic power.

        Uses the formula: P = ½ Re(∫ p · v_n* dS)

        Returns
        -------
        float
            Radiated power in Watts.
        """
        # Get velocity grid function
        space = self.surface_pressure.space
        velocity_gf = self.velocity.to_grid_function(space)

        # Compute power = ½ Re(∫ p v* dS)
        # Using L2 inner product: <p, v> = ∫ p v* dS
        p_coeffs = self.surface_pressure.coefficients
        v_coeffs = velocity_gf.coefficients

        # Get mass matrix for proper integration
        identity = bempp.operators.boundary.sparse.identity(space, space, space)
        mass = identity.weak_form()

        # Power = ½ Re(p^H M v)
        integrand = np.conj(v_coeffs) @ (mass @ p_coeffs)
        power = 0.5 * np.real(integrand)

        return max(0.0, power)  # Ensure non-negative

    def sound_power_level(self, ref: float = 1e-12) -> float:
        """
        Compute sound power level in dB re reference power.

        Parameters
        ----------
        ref : float
            Reference power in Watts. Default 1e-12 W (1 pW).

        Returns
        -------
        float
            Sound power level in dB.
        """
        power = self.radiated_power()
        if power <= 0:
            return -np.inf
        return 10 * np.log10(power / ref)

    def spl_at(
        self,
        point: np.ndarray,
        ref: float = 20e-6,
    ) -> float:
        """
        Compute SPL at a single point.

        Parameters
        ----------
        point : np.ndarray
            Position (3,) or (3, 1).
        ref : float
            Reference pressure (default 20 µPa).

        Returns
        -------
        float
            SPL in dB.
        """
        point = np.asarray(point).reshape(3, 1)
        pressure = self.pressure_at(point)
        p_rms = np.abs(pressure[0]) / np.sqrt(2)
        if p_rms <= 0:
            return -np.inf
        return 20 * np.log10(p_rms / ref)

    def on_axis_spl(
        self,
        distance: float = 1.0,
        ref: float = 20e-6,
    ) -> float:
        """
        Compute on-axis SPL at given distance.

        Parameters
        ----------
        distance : float
            Distance from radiator center in meters.
        ref : float
            Reference pressure.

        Returns
        -------
        float
            On-axis SPL in dB.
        """
        reference = self.reference or AcousticReference.from_mesh(self.mesh)
        point = reference.point_on_axis(distance_m=float(distance))
        return self.spl_at(point, ref)

    def converged(self) -> bool:
        """Check if the solver converged successfully."""
        return self.solver_info == 0

    def __repr__(self) -> str:
        status = "converged" if self.converged() else f"info={self.solver_info}"
        return (
            f"RadiationResult(f={self.frequency:.1f}Hz, "
            f"k={np.real(self.wavenumber):.4f}, {status})"
        )
