"""
Shell FEM to Interior Acoustic FEM coupling utilities.

This module provides functions to transfer data between:
- Shell FEM (2D manifold mesh for dome diaphragm)
- Interior acoustic FEM (3D volume mesh for phase plug domain)

The key transfer is velocity: dome surface normal velocity becomes
the Neumann boundary condition on the dome-facing surface of the
acoustic domain.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union
import numpy as np


def create_dome_velocity_bc(
    dome_velocity_profile: Union[Callable, np.ndarray, complex],
    dome_radius_m: float,
    frequency_hz: float,
    rho: float = 1.225,
) -> Callable:
    """
    Create a velocity boundary condition function for the dome interface.

    Converts a dome velocity profile (function of radius) into a
    DOLFINx-compatible boundary condition function.

    Parameters
    ----------
    dome_velocity_profile : Callable, np.ndarray, or complex
        Velocity profile. Can be:
        - A constant complex value (piston motion)
        - A callable f(r) -> v_n returning normal velocity at radius r
        - An array of (r, v) pairs for interpolation
    dome_radius_m : float
        Dome radius for normalization [m]
    frequency_hz : float
        Excitation frequency [Hz]
    rho : float
        Air density [kg/m³]

    Returns
    -------
    Callable
        Function g(x) returning Neumann data -iωρv_n at point x
    """
    omega = 2 * np.pi * frequency_hz

    if isinstance(dome_velocity_profile, (int, float, complex)):
        # Constant piston velocity
        v_n = complex(dome_velocity_profile)
        g_val = -1j * omega * rho * v_n

        def neumann_func(x):
            """Constant Neumann BC (piston motion)."""
            return np.full(x.shape[1], g_val, dtype=np.complex128)

    elif callable(dome_velocity_profile):
        # Velocity varies with radius

        def neumann_func(x):
            """Spatially-varying Neumann BC from velocity profile."""
            # x has shape (3, N) where N is number of points
            r = np.sqrt(x[0]**2 + x[1]**2)
            v_n = np.array([dome_velocity_profile(ri) for ri in r], dtype=np.complex128)
            return -1j * omega * rho * v_n

    elif isinstance(dome_velocity_profile, np.ndarray):
        # Interpolation from array of (r, v) pairs
        r_data = dome_velocity_profile[:, 0]
        v_data = dome_velocity_profile[:, 1]

        def neumann_func(x):
            """Interpolated Neumann BC from velocity data."""
            r = np.sqrt(x[0]**2 + x[1]**2)
            v_n = np.interp(r, r_data, v_data)
            return -1j * omega * rho * v_n

    else:
        raise TypeError(
            f"dome_velocity_profile must be Callable, array, or scalar, got {type(dome_velocity_profile)}"
        )

    return neumann_func


def piston_velocity_profile(
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Callable:
    """
    Create a uniform (piston) velocity profile.

    All points on the dome move with the same velocity.

    Parameters
    ----------
    amplitude : float
        Velocity amplitude [m/s]
    phase : float
        Phase angle [rad]

    Returns
    -------
    Callable
        f(r) -> v_n
    """
    v_n = amplitude * np.exp(1j * phase)
    return lambda r: v_n


def first_mode_velocity_profile(
    dome_radius_m: float,
    amplitude: float = 1.0,
) -> Callable:
    """
    Create a first bending mode velocity profile (approximate).

    Uses a parabolic approximation: v(r) = v0 * (1 - (r/R)²)
    Maximum at center, zero at edge.

    Parameters
    ----------
    dome_radius_m : float
        Dome radius [m]
    amplitude : float
        Peak velocity at center [m/s]

    Returns
    -------
    Callable
        f(r) -> v_n
    """
    R = dome_radius_m

    def profile(r):
        rho = min(r / R, 1.0)
        return amplitude * (1 - rho**2)

    return profile


def ring_source_velocity_profile(
    ring_radius_m: float,
    ring_width_m: float,
    amplitude: float = 1.0,
) -> Callable:
    """
    Create a ring-shaped velocity profile.

    Velocity is concentrated in an annular ring, zero elsewhere.
    Models voice coil excitation at a specific radius.

    Parameters
    ----------
    ring_radius_m : float
        Center radius of the active ring [m]
    ring_width_m : float
        Width of the ring [m]
    amplitude : float
        Velocity in the ring [m/s]

    Returns
    -------
    Callable
        f(r) -> v_n
    """
    r_inner = ring_radius_m - ring_width_m / 2
    r_outer = ring_radius_m + ring_width_m / 2

    def profile(r):
        if r_inner <= r <= r_outer:
            return amplitude
        return 0.0

    return profile


def interpolate_shell_to_acoustic(
    shell_mesh_coords: np.ndarray,
    shell_velocity: np.ndarray,
    acoustic_boundary_coords: np.ndarray,
) -> np.ndarray:
    """
    Interpolate velocity from shell mesh to acoustic boundary mesh.

    Uses radial interpolation (assumes axisymmetric velocity distribution).

    Parameters
    ----------
    shell_mesh_coords : np.ndarray
        (N_shell, 3) shell mesh vertex coordinates
    shell_velocity : np.ndarray
        (N_shell,) complex velocity at each shell vertex
    acoustic_boundary_coords : np.ndarray
        (N_acoustic, 3) acoustic boundary vertex coordinates

    Returns
    -------
    np.ndarray
        (N_acoustic,) interpolated velocity at acoustic boundary vertices
    """
    from scipy.interpolate import interp1d

    # Compute radii for shell mesh
    shell_r = np.sqrt(shell_mesh_coords[:, 0]**2 + shell_mesh_coords[:, 1]**2)

    # Sort by radius for interpolation
    sort_idx = np.argsort(shell_r)
    shell_r_sorted = shell_r[sort_idx]
    shell_v_sorted = shell_velocity[sort_idx]

    # Handle potential duplicate radii by averaging
    unique_r, inverse = np.unique(shell_r_sorted, return_inverse=True)
    unique_v = np.zeros(len(unique_r), dtype=np.complex128)
    counts = np.zeros(len(unique_r))
    for i, idx in enumerate(inverse):
        unique_v[idx] += shell_v_sorted[i]
        counts[idx] += 1
    unique_v /= counts

    # Create interpolator
    interpolator = interp1d(
        unique_r, unique_v,
        kind='linear',
        bounds_error=False,
        fill_value=(unique_v[0], unique_v[-1])  # Extrapolate with edge values
    )

    # Interpolate to acoustic boundary
    acoustic_r = np.sqrt(
        acoustic_boundary_coords[:, 0]**2 +
        acoustic_boundary_coords[:, 1]**2
    )
    return interpolator(acoustic_r)


def compute_acoustic_impedance_at_dome(
    pressure_at_dome: np.ndarray,
    velocity_at_dome: np.ndarray,
    areas: np.ndarray,
) -> complex:
    """
    Compute the acoustic radiation impedance seen by the dome.

    Z_rad = ∫ p dS / ∫ v_n dS

    Parameters
    ----------
    pressure_at_dome : np.ndarray
        Complex pressure at dome interface elements
    velocity_at_dome : np.ndarray
        Complex normal velocity at dome interface elements
    areas : np.ndarray
        Area of each element

    Returns
    -------
    complex
        Radiation impedance [Pa·s/m]
    """
    # Weighted integrals
    p_integral = np.sum(pressure_at_dome * areas)
    v_integral = np.sum(velocity_at_dome * areas)

    if np.abs(v_integral) < 1e-12:
        return complex('inf')

    return p_integral / v_integral


def compute_acoustic_power_flow(
    pressure: np.ndarray,
    velocity: np.ndarray,
    areas: np.ndarray,
) -> float:
    """
    Compute time-averaged acoustic power through a surface.

    P = (1/2) Re{ ∫ p v_n^* dS }

    Parameters
    ----------
    pressure : np.ndarray
        Complex pressure at surface elements
    velocity : np.ndarray
        Complex normal velocity at surface elements
    areas : np.ndarray
        Area of each element

    Returns
    -------
    float
        Time-averaged acoustic power [W]
    """
    # p * v_n^* gives instantaneous power density
    # Time average of Re{p * v_n^* * e^{i2ωt}} is (1/2) Re{p * v_n^*}
    power_density = pressure * np.conj(velocity)
    total_power = 0.5 * np.real(np.sum(power_density * areas))
    return total_power
