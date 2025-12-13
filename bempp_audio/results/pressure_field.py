"""
Pressure field evaluation from BEM surface solution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from bempp_audio._optional import optional_import

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")
from bempp_audio.baffles import InfiniteBaffle
from bempp_audio.results._eval_utils import cache_key_for_array, mask_points_in_front_of_plane

if TYPE_CHECKING:
    from bempp_audio.results.radiation_result import RadiationResult


class PressureField:
    """
    Evaluate acoustic pressure field from boundary solution.

    Uses the Green's representation formula with potential operators
    to compute pressure at arbitrary points in the exterior domain.

    Parameters
    ----------
    result : RadiationResult
        The radiation solution from which to compute field values.

    Examples
    --------
    >>> field = PressureField(result)
    >>> points = np.array([[0, 0, 1], [0, 0, 2]]).T  # Two points
    >>> pressure = field.at_points(points)
    """

    def __init__(self, result: "RadiationResult"):
        self.result = result
        self.space = result.surface_pressure.space
        self.k = result.wavenumber
        self._slp_pot = None
        self._dlp_pot = None
        self._last_points_key = None

    def at_points(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate pressure at arbitrary points.

        Parameters
        ----------
        points : np.ndarray
            Field point positions, shape (3, n_points).

        Returns
        -------
        np.ndarray
            Complex pressure at each point, shape (n_points,).

        Notes
        -----
        Points should be in the exterior domain (not on the surface).
        For points very close to the surface, accuracy may be reduced.
        """
        if not BEMPP_AVAILABLE:
            raise ImportError("bempp_cl is required for pressure field evaluation")

        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(3, 1)

        # Infinite-baffle post-processing: suppress rear half-space.
        # This is a pragmatic approximation layered on top of a full-space solve.
        if isinstance(getattr(self.result, "baffle", None), InfiniteBaffle):
            plane_z = float(getattr(self.result, "baffle_plane_z", 0.0))
            mask, points_eval = mask_points_in_front_of_plane(points, plane_z)
            out = np.zeros(points.shape[1], dtype=complex)
            if not np.any(mask):
                return out
        else:
            points_eval = points

        # Cache potential operators when repeatedly evaluating the same points.
        points_key = cache_key_for_array(points_eval)
        if self._last_points_key == points_key and self._slp_pot is not None and self._dlp_pot is not None:
            slp_pot = self._slp_pot
            dlp_pot = self._dlp_pot
        else:
            slp_pot = bempp.operators.potential.helmholtz.single_layer(
                self.space, points_eval, self.k
            )
            dlp_pot = bempp.operators.potential.helmholtz.double_layer(
                self.space, points_eval, self.k
            )
            self._slp_pot = slp_pot
            self._dlp_pot = dlp_pot
            self._last_points_key = points_key

        # Get Neumann data from velocity
        neumann_gf = self.result.neumann_grid_function(self.space)

        # Green's representation formula:
        # p(x) = ∫ G(x,y) ∂p/∂n(y) dS - ∫ ∂G/∂n(x,y) p(y) dS
        # where ∂p/∂n = -iωρv_n
        slp_contribution = slp_pot @ neumann_gf
        dlp_contribution = dlp_pot @ self.result.surface_pressure

        # Note: sign convention depends on normal direction
        # For exterior problem: p = SLP(∂p/∂n) - DLP(p)
        field = slp_contribution - dlp_contribution

        field = field.ravel()
        if isinstance(getattr(self.result, "baffle", None), InfiniteBaffle):
            out[mask] = field
            return out
        return field

    def on_axis(self, distances: np.ndarray) -> np.ndarray:
        """
        Evaluate pressure along the principal axis.

        Parameters
        ----------
        distances : np.ndarray
            Distances from the radiator center along the axis.

        Returns
        -------
        np.ndarray
            Complex pressure at each distance.
        """
        distances = np.atleast_1d(distances)
        from bempp_audio.acoustic_reference import AcousticReference

        reference = self.result.reference or AcousticReference.from_mesh(self.result.mesh)
        origin = reference.origin
        axis = reference.axis

        # Create points along axis
        points = np.array([
            origin + float(d) * axis for d in distances
        ]).T  # Shape (3, n_distances)

        return self.at_points(points)

    def on_plane(
        self,
        plane: str = "xz",
        extent: float = 1.0,
        n_points: int = 50,
        offset: float = 0.0,
    ) -> tuple:
        """
        Evaluate pressure on a 2D plane.

        Parameters
        ----------
        plane : str
            Plane orientation: 'xz', 'xy', or 'yz'.
        extent : float
            Half-size of the plane in meters.
        n_points : int
            Number of points per dimension.
        offset : float
            Offset from origin in the third dimension.

        Returns
        -------
        tuple
            (X, Y, pressure) where X, Y are coordinate grids and
            pressure has shape (n_points, n_points).
        """
        coord1 = np.linspace(-extent, extent, n_points)
        coord2 = np.linspace(-extent, extent, n_points)
        C1, C2 = np.meshgrid(coord1, coord2)

        if plane == "xz":
            X, Z = C1, C2
            Y = np.full_like(X, offset)
            points = np.array([X.ravel(), Y.ravel(), Z.ravel()])
        elif plane == "xy":
            X, Y = C1, C2
            Z = np.full_like(X, offset)
            points = np.array([X.ravel(), Y.ravel(), Z.ravel()])
        elif plane == "yz":
            Y, Z = C1, C2
            X = np.full_like(Y, offset)
            points = np.array([X.ravel(), Y.ravel(), Z.ravel()])
        else:
            raise ValueError(f"Unknown plane: {plane}")

        pressure = self.at_points(points)
        pressure_grid = pressure.reshape(n_points, n_points)

        return C1, C2, pressure_grid

    def spl_db(
        self,
        points: np.ndarray,
        ref: float = 20e-6,
    ) -> np.ndarray:
        """
        Compute SPL in dB at given points.

        Parameters
        ----------
        points : np.ndarray
            Field point positions, shape (3, n_points).
        ref : float
            Reference pressure (default 20 µPa).

        Returns
        -------
        np.ndarray
            SPL values in dB, shape (n_points,).
        """
        pressure = self.at_points(points)
        p_rms = np.abs(pressure) / np.sqrt(2)

        # Avoid log of zero
        p_rms = np.maximum(p_rms, 1e-20)

        return 20 * np.log10(p_rms / ref)

    def phase_deg(self, points: np.ndarray) -> np.ndarray:
        """
        Compute phase at given points in degrees.

        Parameters
        ----------
        points : np.ndarray
            Field point positions, shape (3, n_points).

        Returns
        -------
        np.ndarray
            Phase values in degrees, shape (n_points,).
        """
        pressure = self.at_points(points)
        return np.degrees(np.angle(pressure))

    def __repr__(self) -> str:
        return f"PressureField(f={self.result.frequency:.1f}Hz)"
