"""
Velocity profile definitions for acoustic radiators.

Provides various methods to define the normal velocity distribution
on a radiator surface, from simple piston motion to complex modal patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union, List
import weakref
import numpy as np

from bempp_audio._optional import optional_import

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")


@dataclass
class ModalShape:
    """Container for a single mode shape."""
    frequency: float  # Natural frequency in Hz
    shape: np.ndarray  # Mode shape values at mesh vertices
    damping: float = 0.0  # Modal damping ratio


class VelocityProfile:
    """
    Define normal velocity distribution on a radiator surface.

    Provides factory methods to create common velocity patterns and
    conversion to bempp GridFunctions for use in BEM solvers.

    Attributes
    ----------
    amplitude : complex
        Overall velocity amplitude in m/s.
    phase : float
        Phase offset in radians.
    """

    def __init__(
        self,
        velocity_func: Callable[[np.ndarray, np.ndarray], complex],
        amplitude: complex = 1.0,
        phase: float = 0.0,
    ):
        """
        Initialize a velocity profile.

        Parameters
        ----------
        velocity_func : callable
            Function that takes (position, normal) and returns complex velocity.
            Position shape: (3,), Normal shape: (3,).
        amplitude : complex
            Overall amplitude multiplier.
        phase : float
            Phase offset in radians.
        """
        self._func = velocity_func
        self.amplitude = amplitude
        self.phase = phase
        self._grid_function_cache: "weakref.WeakKeyDictionary[object, dict[object, object]]" = weakref.WeakKeyDictionary()

    def __call__(self, x: np.ndarray, n: np.ndarray) -> complex:
        """
        Evaluate velocity at a point.

        Parameters
        ----------
        x : np.ndarray
            Position vector (3,).
        n : np.ndarray
            Normal vector (3,).

        Returns
        -------
        complex
            Normal velocity at the point.
        """
        base_vel = self._func(x, n)
        return self.amplitude * base_vel * np.exp(1j * self.phase)

    @classmethod
    def piston(
        cls,
        amplitude: complex = 1.0,
        phase: float = 0.0,
    ) -> "VelocityProfile":
        """
        Create a uniform piston velocity profile.

        All points on the surface move with the same velocity in the
        normal direction (rigid piston mode).

        Parameters
        ----------
        amplitude : complex
            Velocity amplitude in m/s.
        phase : float
            Phase offset in radians.

        Returns
        -------
        VelocityProfile
            Uniform velocity profile.

        Examples
        --------
        >>> profile = VelocityProfile.piston(amplitude=0.01)
        >>> profile(np.array([0, 0, 0]), np.array([0, 0, 1]))
        0.01
        """
        def func(x, n):
            return 1.0

        return _mark_profile_type(cls(func, amplitude, phase), "piston", {})

    @classmethod
    def gaussian(
        cls,
        center: np.ndarray,
        width: float,
        amplitude: complex = 1.0,
        phase: float = 0.0,
    ) -> "VelocityProfile":
        """
        Create a Gaussian-weighted velocity distribution.

        Velocity falls off as exp(-r^2 / (2*width^2)) from the center.
        Useful for modeling focused excitation or soft edges.

        Parameters
        ----------
        center : np.ndarray
            Center point of the Gaussian (3,).
        width : float
            Standard deviation of the Gaussian in meters.
        amplitude : complex
            Peak velocity amplitude.
        phase : float
            Phase offset in radians.

        Returns
        -------
        VelocityProfile
            Gaussian velocity profile.
        """
        center = np.asarray(center)

        def func(x, n):
            r2 = np.sum((x - center)**2)
            return np.exp(-r2 / (2 * width**2))

        return _mark_profile_type(
            cls(func, amplitude, phase),
            "gaussian",
            {"center": center.tolist(), "width": float(width)},
        )

    @classmethod
    def radial_taper(
        cls,
        center: np.ndarray,
        inner_radius: float,
        outer_radius: float,
        amplitude: complex = 1.0,
        phase: float = 0.0,
        taper_type: str = "linear",
    ) -> "VelocityProfile":
        """
        Create a radially-tapered velocity distribution.

        Useful for modeling cone drivers where velocity decreases
        toward the surround.

        Parameters
        ----------
        center : np.ndarray
            Center point (axis of revolution).
        inner_radius : float
            Radius where velocity is maximum.
        outer_radius : float
            Radius where velocity reaches zero.
        amplitude : complex
            Peak velocity amplitude.
        phase : float
            Phase offset in radians.
        taper_type : str
            Taper function: 'linear', 'cosine', 'quadratic'.

        Returns
        -------
        VelocityProfile
            Radially-tapered velocity profile.
        """
        center = np.asarray(center)

        def func(x, n):
            r = np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
            if r <= inner_radius:
                return 1.0
            elif r >= outer_radius:
                return 0.0
            else:
                t = (r - inner_radius) / (outer_radius - inner_radius)
                if taper_type == "linear":
                    return 1.0 - t
                elif taper_type == "cosine":
                    return 0.5 * (1 + np.cos(np.pi * t))
                elif taper_type == "quadratic":
                    return (1.0 - t)**2
                else:
                    raise ValueError(f"Unknown taper type: {taper_type}")

        return _mark_profile_type(
            cls(func, amplitude, phase),
            "radial_taper",
            {
                "center": center.tolist(),
                "inner_radius": float(inner_radius),
                "outer_radius": float(outer_radius),
                "taper_type": str(taper_type),
            },
        )

    @classmethod
    def from_modal_coefficients(
        cls,
        mode_shapes: List[ModalShape],
        coefficients: np.ndarray,
        mesh_vertices: np.ndarray,
        amplitude: complex = 1.0,
    ) -> "VelocityProfile":
        """
        Create velocity from weighted sum of mode shapes.

        Used when modal analysis results (from FEA) are available.
        Velocity = sum(coefficient_i * mode_shape_i).

        Parameters
        ----------
        mode_shapes : list of ModalShape
            Mode shapes from FEA analysis.
        coefficients : np.ndarray
            Complex modal participation factors.
        mesh_vertices : np.ndarray
            Vertex positions shape (3, n_vertices) matching mode shapes.
        amplitude : complex
            Overall amplitude multiplier.

        Returns
        -------
        VelocityProfile
            Modal velocity profile.

        Notes
        -----
        For accurate modal superposition, coefficients should include
        frequency-dependent scaling: H(f) = 1 / (fn^2 - f^2 + 2j*zeta*fn*f)
        """
        # Build interpolation from vertices
        from scipy.interpolate import LinearNDInterpolator

        # Compute combined velocity field at vertices
        n_vertices = mesh_vertices.shape[1]
        velocity_at_vertices = np.zeros(n_vertices, dtype=complex)

        for mode, coef in zip(mode_shapes, coefficients):
            velocity_at_vertices += coef * mode.shape

        # Create interpolator for arbitrary points
        points = mesh_vertices.T  # (n_vertices, 3)
        interpolator = LinearNDInterpolator(
            points, velocity_at_vertices, fill_value=0.0
        )

        def func(x, n):
            return complex(interpolator(x.reshape(1, 3))[0])

        return cls(func, amplitude, 0.0)

    @classmethod
    def from_array(
        cls,
        vertices: np.ndarray,
        velocity_values: np.ndarray,
        amplitude: complex = 1.0,
    ) -> "VelocityProfile":
        """
        Create velocity profile from pre-computed values at vertices.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex positions shape (3, n_vertices) or (n_vertices, 3).
        velocity_values : np.ndarray
            Complex velocity at each vertex (n_vertices,).
        amplitude : complex
            Overall amplitude multiplier.

        Returns
        -------
        VelocityProfile
            Interpolated velocity profile.
        """
        from scipy.interpolate import LinearNDInterpolator

        if vertices.shape[0] == 3:
            vertices = vertices.T  # Convert to (n_vertices, 3)

        interpolator = LinearNDInterpolator(
            vertices, velocity_values, fill_value=0.0
        )

        def func(x, n):
            return complex(interpolator(x.reshape(1, 3))[0])

        return cls(func, amplitude, 0.0)

    @classmethod
    def zero(cls) -> "VelocityProfile":
        """
        Create a zero-velocity profile (rigid boundary).

        Useful for baffle surfaces that don't radiate.

        Returns
        -------
        VelocityProfile
            Zero velocity everywhere.
        """
        return _mark_profile_type(cls(lambda x, n: 0.0, amplitude=0.0), "zero", {})

    @classmethod
    def by_domain(
        cls,
        profiles: dict,
        default: Optional["VelocityProfile"] = None,
    ) -> "VelocityProfile":
        """
        Create a velocity profile that varies by domain index.

        Useful for combined meshes (e.g., radiator + baffle) where different
        regions have different velocity distributions.

        Parameters
        ----------
        profiles : dict
            Mapping from domain_index to VelocityProfile.
        default : VelocityProfile, optional
            Default profile for unspecified domains. If None, uses zero.

        Returns
        -------
        VelocityProfile
            Domain-dependent velocity profile.

        Examples
        --------
        >>> profiles = {
        ...     0: VelocityProfile.piston(amplitude=0.01),  # Radiator
        ...     1: VelocityProfile.zero(),  # Baffle
        ... }
        >>> combined = VelocityProfile.by_domain(profiles)
        """
        if default is None:
            default = cls.zero()

        # Store in closure for to_grid_function to access
        profile = cls(lambda x, n: 0.0)
        profile._domain_profiles = profiles
        profile._default_profile = default
        profile._is_domain_dependent = True
        return profile

    def to_grid_function(
        self,
        space,
        wavenumber: Optional[complex] = None,
        domain_indices: Optional[np.ndarray] = None,
    ):
        """
        Convert velocity profile to a bempp GridFunction.

        Parameters
        ----------
        space : bempp.FunctionSpace
            The function space for the GridFunction.
        wavenumber : complex, optional
            Wavenumber for frequency-dependent profiles.
        domain_indices : np.ndarray, optional
            Domain index for each element (for domain-dependent profiles).

        Returns
        -------
        bempp.GridFunction
            The velocity as a GridFunction suitable for BEM RHS.
        """
        if not BEMPP_AVAILABLE:
            raise ImportError("bempp_cl required for to_grid_function")

        # Cache GridFunctions per space (and wavenumber, when provided). This matters for:
        # - frequency sweeps where velocity does not change with frequency (common)
        # - impedance / post-processing that re-projects velocity multiple times
        #
        # Without caching, domain-dependent profiles can trigger an O(n_elements) projection
        # for every frequency, and callable-based profiles can trigger repeated backend/JIT
        # compilation or projection work when coefficients are accessed.
        wkey = complex(wavenumber) if wavenumber is not None else None
        try:
            per_space = self._grid_function_cache.get(space)
        except TypeError:
            # Some space objects might not support weakrefs; fall back to no caching.
            per_space = None
        if per_space is not None and wkey in per_space:
            return per_space[wkey]

        # Check if this is a domain-dependent profile
        if hasattr(self, "_is_domain_dependent") and self._is_domain_dependent:
            gf = self._to_grid_function_by_domain(space, domain_indices)
            if per_space is not None:
                per_space[wkey] = gf
            else:
                try:
                    self._grid_function_cache[space] = {wkey: gf}
                except TypeError:
                    pass
            return gf

        # For simple profiles, use coefficient-based approach
        # This avoids numba JIT issues with closures
        profile_type = getattr(self, "_profile_type", "custom")

        if profile_type == "piston":
            vel_value = complex(self.amplitude) * np.exp(1j * float(self.phase))
            coeffs = np.full(int(space.global_dof_count), vel_value, dtype=complex)
            gf = bempp.GridFunction(space, coefficients=coeffs)
            if per_space is not None:
                per_space[wkey] = gf
            else:
                try:
                    self._grid_function_cache[space] = {wkey: gf}
                except TypeError:
                    pass
            return gf

        elif profile_type == "zero":
            coeffs = np.zeros(int(space.global_dof_count), dtype=complex)
            gf = bempp.GridFunction(space, coefficients=coeffs)
            if per_space is not None:
                per_space[wkey] = gf
            else:
                try:
                    self._grid_function_cache[space] = {wkey: gf}
                except TypeError:
                    pass
            return gf

        else:
            # For complex profiles, compute coefficients by projection
            gf = self._to_grid_function_by_projection(space)
            if per_space is not None:
                per_space[wkey] = gf
            else:
                try:
                    self._grid_function_cache[space] = {wkey: gf}
                except TypeError:
                    pass
            return gf

    def _to_grid_function_by_projection(self, space):
        """Create grid function by evaluating at DOF locations."""
        # Get grid information
        grid = space.grid
        n_elements = grid.number_of_elements

        # For DP0 space: one DOF per element at centroid
        # For P1 space: one DOF per vertex
        n_dofs = space.global_dof_count

        if n_dofs != n_elements:
            raise NotImplementedError(
                "VelocityProfile coefficient projection is only implemented for spaces with "
                "one DOF per element (e.g., DP0). Use a callable-based profile (e.g. piston/zero) "
                "or switch solver space to DP0."
            )

        # Compute coefficients by evaluating velocity at element centers
        # This is an approximation - proper L2 projection would be more accurate
        coefficients = np.zeros(n_dofs, dtype=complex)

        # Vectorized centroid/normal computation; the velocity evaluation itself
        # remains per-element because user callbacks are typically scalar.
        vertices = np.asarray(grid.vertices)
        elements = np.asarray(grid.elements)

        p0 = vertices[:, elements[0, :]]
        p1 = vertices[:, elements[1, :]]
        p2 = vertices[:, elements[2, :]]
        centroids = (p0 + p1 + p2) / 3.0  # (3, n_elements)

        e1 = (p1 - p0).T  # (n_elements, 3)
        e2 = (p2 - p0).T
        normals = np.cross(e1, e2)  # (n_elements, 3)
        norms = np.linalg.norm(normals, axis=1)
        good = norms > 0
        if np.any(good):
            normals[good] = normals[good] / norms[good, None]
        if np.any(~good):
            normals[~good] = np.array([0.0, 0.0, 1.0])

        for i in range(min(n_elements, n_dofs)):
            coefficients[i] = self(centroids[:, i], normals[i])

        return bempp.GridFunction(space, coefficients=coefficients)

    def _to_grid_function_by_domain(self, space, domain_indices):
        """Create grid function with domain-dependent velocity."""
        profiles = self._domain_profiles
        default = self._default_profile

        # Fast path: DP0 (one DOF per element) + only piston/zero sub-profiles.
        # This avoids centroid/normal computation and per-element Python callbacks.
        grid = space.grid
        n_elements = grid.number_of_elements
        n_dofs = space.global_dof_count

        if n_dofs != n_elements:
            # Fallback for spaces like P1 (OSRC requires P1): use a callable GridFunction
            # that dispatches by `domain_index` at quadrature points.
            def _piston_value(p: "VelocityProfile") -> complex:
                return complex(p.amplitude) * np.exp(1j * float(p.phase))

            def fun(x, n, domain_index, result, _params):  # bempp vectorized signature
                # domain_index is per-quadrature-point, but constant over each element.
                dom_ids = np.asarray(domain_index, dtype=int)

                result[...] = 0.0 + 0.0j
                if dom_ids.size == 0:
                    return

                # Fast path for piston/zero domain profiles
                unique = np.unique(dom_ids)
                for dom in unique:
                    prof = profiles.get(int(dom), default)
                    if prof is None:
                        val = 0.0 + 0.0j
                    else:
                        pt = getattr(prof, "_profile_type", "custom")
                        if pt == "piston":
                            val = _piston_value(prof)
                        elif pt == "zero":
                            val = 0.0 + 0.0j
                        else:
                            mask = dom_ids == int(dom)
                            # Evaluate only on the masked quadrature points.
                            result[0, mask] = prof(x[:, mask], n[:, mask])
                            continue
                    result[0, dom_ids == int(dom)] = val

            # bempp-cl expects the callback to specify real/complex.
            fun.bempp_type = "complex"  # type: ignore[attr-defined]
            fun.bempp_vectorized = True  # type: ignore[attr-defined]
            return bempp.GridFunction(space, fun=fun)

        domain_ids = np.asarray(grid.domain_indices)

        def _piston_value(p: "VelocityProfile") -> complex:
            return complex(p.amplitude) * np.exp(1j * float(p.phase))

        only_simple = True
        for p in list(profiles.values()) + ([default] if default is not None else []):
            if p is None:
                continue
            pt = getattr(p, "_profile_type", "custom")
            if pt not in ("piston", "zero"):
                only_simple = False
                break

        if only_simple:
            coefficients = np.zeros(int(n_dofs), dtype=complex)
            for i in range(int(n_elements)):
                dom_idx = int(domain_ids[i])
                prof = profiles.get(dom_idx, default)
                if prof is None:
                    coefficients[i] = 0.0 + 0.0j
                    continue
                pt = getattr(prof, "_profile_type", "custom")
                if pt == "piston":
                    coefficients[i] = _piston_value(prof)
                else:
                    coefficients[i] = 0.0 + 0.0j
            return bempp.GridFunction(space, coefficients=coefficients)

        # Generic fallback: evaluate each domain's profile at element centroid/normal.
        coefficients = np.zeros(n_dofs, dtype=complex)

        vertices = np.asarray(grid.vertices)
        elements = np.asarray(grid.elements)

        p0 = vertices[:, elements[0, :]]
        p1 = vertices[:, elements[1, :]]
        p2 = vertices[:, elements[2, :]]
        centroids = (p0 + p1 + p2) / 3.0  # (3, n_elements)

        e1 = (p1 - p0).T
        e2 = (p2 - p0).T
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1)
        good = norms > 0
        if np.any(good):
            normals[good] = normals[good] / norms[good, None]
        if np.any(~good):
            normals[~good] = np.array([0.0, 0.0, 1.0])

        for i in range(min(n_elements, n_dofs)):
            dom_idx = domain_ids[i]
            prof = profiles.get(dom_idx, default)
            coefficients[i] = prof(centroids[:, i], normals[i])

        return bempp.GridFunction(space, coefficients=coefficients)

    def to_dict(self) -> dict:
        """Serialize velocity profile for multiprocessing."""
        # Check if this is a domain-dependent profile
        if hasattr(self, "_is_domain_dependent") and self._is_domain_dependent:
            # Serialize each domain's profile
            domain_profiles_data = {}
            for dom_idx, profile in self._domain_profiles.items():
                domain_profiles_data[dom_idx] = profile.to_dict()

            default_data = None
            if self._default_profile:
                default_data = self._default_profile.to_dict()

            return {
                "type": "by_domain",
                "domain_profiles": domain_profiles_data,
                "default": default_data,
            }

        # Note: Only works for simple profiles (piston, gaussian, etc.)
        # Modal profiles need special handling
        amp = complex(self.amplitude)
        return {
            "type": self._profile_type if hasattr(self, "_profile_type") else "custom",
            "amplitude_real": float(amp.real),
            "amplitude_imag": float(amp.imag),
            "phase": float(self.phase),
            "params": getattr(self, "_params", {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VelocityProfile":
        """Reconstruct velocity profile from dictionary."""
        profile_type = data.get("type", "piston")

        # Handle domain-dependent profiles
        if profile_type == "by_domain":
            domain_profiles = {}
            for dom_idx_str, profile_data in data["domain_profiles"].items():
                dom_idx = int(dom_idx_str)  # JSON keys are strings
                domain_profiles[dom_idx] = cls.from_dict(profile_data)

            default = None
            if data.get("default"):
                default = cls.from_dict(data["default"])

            return cls.by_domain(domain_profiles, default=default)

        # Reconstruct complex amplitude
        if "amplitude_real" in data:
            amplitude = complex(data["amplitude_real"], data.get("amplitude_imag", 0.0))
        else:
            # Legacy format
            amplitude = data.get("amplitude", 1.0)
            if isinstance(amplitude, str):
                amplitude = complex(amplitude.replace("(", "").replace(")", ""))

        phase = data.get("phase", 0.0)
        params = data.get("params", {})

        if profile_type == "piston":
            profile = cls.piston(amplitude=amplitude, phase=phase)
        elif profile_type == "gaussian":
            profile = cls.gaussian(
                center=np.array(params["center"]),
                width=params["width"],
                amplitude=amplitude,
                phase=phase,
            )
        elif profile_type == "zero":
            profile = cls.zero()
        else:
            # Fallback to piston
            profile = cls.piston(amplitude=amplitude, phase=phase)

        return profile

    def __repr__(self) -> str:
        profile_type = getattr(self, "_profile_type", "custom")
        return f"VelocityProfile({profile_type}, amplitude={self.amplitude})"


# Helper to mark profile types for serialization
def _mark_profile_type(profile: VelocityProfile, type_name: str, params: dict):
    """Add type information for serialization."""
    profile._profile_type = type_name
    profile._params = params
    return profile
