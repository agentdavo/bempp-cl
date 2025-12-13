"""
Boundary conditions and configuration helpers for the `Loudspeaker` fluent API.

These helpers mutate a `Loudspeaker` instance and return it for chaining.
"""

from __future__ import annotations

from typing import Optional, List
from dataclasses import replace
import numpy as np

from bempp_audio.velocity import VelocityProfile
from bempp_audio.baffles import FreeSpace, InfiniteBaffle, CircularBaffle
from bempp_audio.api.types import (
    BCPolicyLike,
    FrequencySpacingLike,
    VelocityModeLike,
    normalize_bc_policy,
    normalize_frequency_spacing,
    normalize_velocity_mode,
)


def infinite_baffle(speaker: "Loudspeaker") -> "Loudspeaker":
    return speaker._with_state(baffle=InfiniteBaffle())


def circular_baffle(
    speaker: "Loudspeaker",
    radius: float,
    element_size: Optional[float] = None,
) -> "Loudspeaker":
    return speaker._with_state(baffle=CircularBaffle(radius=float(radius), element_size=element_size))


def free_space(speaker: "Loudspeaker") -> "Loudspeaker":
    return speaker._with_state(baffle=FreeSpace())


def velocity(
    speaker: "Loudspeaker",
    mode: VelocityModeLike = "piston",
    amplitude: complex = 1.0,
    phase: float = 0.0,
    **kwargs,
) -> "Loudspeaker":
    profile: VelocityProfile
    mode_norm = normalize_velocity_mode(mode)
    if mode_norm == "piston":
        profile = VelocityProfile.piston(amplitude, phase)
    elif mode_norm == "gaussian":
        center = np.array(kwargs.get("center", [0, 0, 0]))
        width = kwargs.get("width", 0.01)
        profile = VelocityProfile.gaussian(center, width, amplitude, phase)
    elif mode_norm == "radial_taper":
        center = np.array(kwargs.get("center", [0, 0, 0]))
        inner_r = kwargs.get("inner_radius", 0.01)
        outer_r = kwargs.get("outer_radius", 0.05)
        taper = kwargs.get("taper_type", "linear")
        profile = VelocityProfile.radial_taper(center, inner_r, outer_r, amplitude, phase, taper)
    elif mode_norm == "zero":
        profile = VelocityProfile.zero()
    else:
        raise ValueError(f"Unknown velocity mode: {mode}")

    return speaker._with_state(velocity=profile)


def velocity_profile(speaker: "Loudspeaker", profile: VelocityProfile) -> "Loudspeaker":
    return speaker._with_state(velocity=profile)


def domain_names(speaker: "Loudspeaker", names: dict) -> "Loudspeaker":
    """
    Assign human-readable names to mesh domain IDs.

    This enables using `.velocity_by_name()` instead of numeric domain IDs.

    Parameters
    ----------
    names : dict
        Mapping of domain ID (int) to name (str).
        Example: {1: "throat", 2: "walls", 3: "enclosure"}

    Returns
    -------
    Loudspeaker
        Self for chaining.
    """
    # Validate: keys must be int, values must be str
    validated = {}
    for k, v in names.items():
        if not isinstance(k, int):
            raise TypeError(f"Domain ID must be int, got {type(k).__name__}: {k}")
        if not isinstance(v, str):
            raise TypeError(f"Domain name must be str, got {type(v).__name__}: {v}")
        validated[int(k)] = str(v)
    return speaker._with_state(domain_names=validated)


def velocity_by_name(
    speaker: "Loudspeaker",
    profiles: dict,
    default: Optional[VelocityProfile] = None,
) -> "Loudspeaker":
    """
    Assign velocity profiles by domain name instead of numeric ID.

    Requires `.domain_names()` to be called first, or waveguide metadata
    to provide automatic names ("throat", "walls").

    Parameters
    ----------
    profiles : dict
        Mapping of domain name (str) to VelocityProfile.
        Example: {"throat": VelocityProfile.piston(0.01), "walls": VelocityProfile.zero()}
    default : VelocityProfile, optional
        Default profile for domains not in the mapping.
        If None, uses the `default_bc_policy` setting.

    Returns
    -------
    Loudspeaker
        Self for chaining.
    """
    # Build name -> ID mapping from state + waveguide metadata
    name_to_id: dict = {}

    # From explicit domain_names
    if speaker.state.domain_names:
        for dom_id, name in speaker.state.domain_names.items():
            name_to_id[name] = dom_id

    # From waveguide metadata (auto-assign "throat" and "walls")
    if speaker.state.waveguide:
        wg = speaker.state.waveguide
        name_to_id.setdefault("throat", wg.throat_domain)
        name_to_id.setdefault("walls", wg.wall_domain)

    if not name_to_id:
        raise ValueError(
            "No domain names configured. Call .domain_names({...}) first, "
            "or use a waveguide which provides automatic names."
        )

    # Convert name-based profiles to ID-based
    id_profiles: dict = {}
    for name, profile in profiles.items():
        if name not in name_to_id:
            available = ", ".join(sorted(name_to_id.keys()))
            raise KeyError(f"Unknown domain name: {name!r}. Available: {available}")
        id_profiles[name_to_id[name]] = profile

    # Handle default based on policy
    if default is None and speaker.state.default_bc_policy == "rigid":
        default = VelocityProfile.zero()

    combined = VelocityProfile.by_domain(id_profiles, default=default)
    return speaker._with_state(velocity=combined)


def default_bc(speaker: "Loudspeaker", policy: BCPolicyLike = "rigid") -> "Loudspeaker":
    """
    Set the default boundary condition policy for unassigned domains.

    Parameters
    ----------
    policy : str
        - "rigid": Unassigned domains get zero velocity (rigid wall). Default.
        - "error": Raise an error if any domain is not explicitly assigned.

    Returns
    -------
    Loudspeaker
        Self for chaining.
    """
    policy_norm = normalize_bc_policy(policy)
    if policy_norm not in ("rigid", "error"):
        raise ValueError(f"Unknown BC policy: {policy!r}. Use 'rigid' or 'error'.")
    return speaker._with_state(default_bc_policy=policy_norm)


def frequencies(speaker: "Loudspeaker", freqs: np.ndarray) -> "Loudspeaker":
    return speaker._with_state(frequencies=np.asarray(freqs))


def single_frequency(speaker: "Loudspeaker", frequency: float) -> "Loudspeaker":
    return speaker._with_state(frequencies=np.array([frequency]))


def frequency_range(
    speaker: "Loudspeaker",
    f1: float = 200.0,
    f2: float = 20000.0,
    num: int = 20,
    spacing: FrequencySpacingLike = "log",
    points_per_octave: Optional[int] = None,
) -> "Loudspeaker":
    spacing_norm = normalize_frequency_spacing(spacing)
    if spacing_norm == "octave" and points_per_octave is None:
        points_per_octave = int(num)

    if points_per_octave is not None:
        if f1 <= 0 or f2 <= 0 or f2 <= f1:
            raise ValueError("For points_per_octave, require 0 < f1 < f2.")
        ppo = int(points_per_octave)
        if ppo < 1:
            raise ValueError("points_per_octave must be >= 1.")
        octaves = float(np.log2(float(f2) / float(f1)))
        n = max(2, int(round(octaves * float(ppo))) + 1)
        t = np.linspace(0.0, octaves, n)
        freqs = float(f1) * (2.0 ** t)
        return speaker._with_state(frequencies=freqs)

    if spacing_norm in ("log", "logarithmic", "1"):
        freqs = np.logspace(np.log10(f1), np.log10(f2), int(num))
    elif spacing_norm in ("linear", "lin", "2"):
        freqs = np.linspace(f1, f2, int(num))
    else:
        raise ValueError(f"Unknown spacing '{spacing}'. Use 'log' or 'linear'.")
    return speaker._with_state(frequencies=freqs)


def octave_bands(
    speaker: "Loudspeaker",
    f_start: float = 63.0,
    f_end: float = 16000.0,
    fraction: int = 3,
) -> "Loudspeaker":
    bands: list[float] = []
    n_start = int(np.ceil(fraction * np.log2(f_start / 1000)))
    n_end = int(np.floor(fraction * np.log2(f_end / 1000)))
    for n in range(n_start, n_end + 1):
        bands.append(1000 * (2 ** (n / fraction)))
    return speaker._with_state(frequencies=np.array(bands))


def polar_angles(speaker: "Loudspeaker", start: float = 0.0, end: float = 180.0, num: int = 37) -> "Loudspeaker":
    return speaker._with_state(polar_start=start, polar_end=end, polar_num=num)


def normalize_to(speaker: "Loudspeaker", angle: float = 0.0) -> "Loudspeaker":
    return speaker._with_state(norm_angle=angle)


def acoustic_reference(
    speaker: "Loudspeaker",
    *,
    origin: Optional[np.ndarray] = None,
    axis: Optional[np.ndarray] = None,
    default_distance_m: Optional[float] = None,
) -> "Loudspeaker":
    from bempp_audio.acoustic_reference import AcousticReference

    mesh = speaker.state.mesh
    if mesh is None and (origin is None or axis is None):
        raise ValueError("Mesh not configured. Provide `origin` and `axis` explicitly or configure geometry first.")

    if origin is None:
        origin = np.asarray(mesh.center, dtype=float)  # type: ignore[union-attr]
    if axis is None:
        axis = np.asarray(mesh.axis, dtype=float)  # type: ignore[union-attr]
    if default_distance_m is None:
        default_distance_m = float(speaker.state.measurement_distance)

    return speaker._with_state(
        reference=AcousticReference.from_origin_axis(origin, axis, default_distance_m=float(default_distance_m))
    )


def reference_mode(speaker: "Loudspeaker", mode: str = "mouth") -> "Loudspeaker":
    """
    Set the acoustic reference origin for waveguides.

    For waveguides/horns, the acoustic reference can be placed at either:
    - "mouth" (default): The mouth center at z=0. This is the standard for
      directivity measurements where angles are measured from the mouth plane.
    - "throat": The throat center at z=-length. This is sometimes used when
      treating the throat as the acoustic source location.

    The axis is always +z (forward direction).
    """
    from bempp_audio.acoustic_reference import AcousticReference

    wg = speaker.state.waveguide
    mesh = speaker.state.mesh

    if wg is None and mesh is None:
        raise ValueError(
            "No waveguide or mesh configured. Call waveguide() or waveguide_from_config() first, "
            "or use acoustic_reference() directly."
        )

    # Get axis from mesh (always +z for our waveguides)
    axis = np.asarray(mesh.axis, dtype=float) if mesh is not None else np.array([0.0, 0.0, 1.0])

    mode_lower = mode.lower()
    if mode_lower == "mouth":
        if wg is not None:
            origin = np.array(wg.mouth_center, dtype=float)
        elif mesh is not None:
            origin = np.asarray(mesh.center, dtype=float)
        else:
            raise ValueError("Cannot determine mouth center without waveguide metadata or mesh")
    elif mode_lower == "throat":
        if wg is None:
            raise ValueError(
                "Throat reference requires waveguide metadata. "
                "Use waveguide() or waveguide_from_config() to set up a waveguide first."
            )
        # Throat is at z=-length from mouth
        mouth_center = np.array(wg.mouth_center, dtype=float)
        origin = mouth_center.copy()
        origin[2] = -wg.length  # Throat is at z=-length
    else:
        raise ValueError(f"Unknown reference mode: {mode!r}. Use 'mouth' or 'throat'.")

    distance = float(speaker.state.measurement_distance)
    return speaker._with_state(
        reference=AcousticReference.from_origin_axis(origin, axis, default_distance_m=distance)
    )


def measurement_distance(speaker: "Loudspeaker", distance: float = 1.0) -> "Loudspeaker":
    ref = speaker.state.reference
    if ref is None:
        return speaker._with_state(measurement_distance=distance)
    return speaker._with_state(measurement_distance=distance, reference=ref.with_default_distance(distance))


def spl_angles(speaker: "Loudspeaker", angles: List[float]) -> "Loudspeaker":
    return speaker._with_state(spl_angles=list(angles))


def medium(speaker: "Loudspeaker", c: float = 343.0, rho: float = 1.225) -> "Loudspeaker":
    return speaker._with_state(c=c, rho=rho)


def solver_options(
    speaker: "Loudspeaker",
    tol: float = 1e-5,
    maxiter: int = 1000,
    use_fmm: bool = False,
    fmm_expansion_order: int | None = None,
) -> "Loudspeaker":
    return speaker._with_state(
        solver_options=replace(
            speaker.state.solver_options,
            tol=tol,
            maxiter=maxiter,
            use_fmm=use_fmm,
            fmm_expansion_order=speaker.state.solver_options.fmm_expansion_order
            if fmm_expansion_order is None
            else int(fmm_expansion_order),
        )
    )


def solver_progress(
    speaker: "Loudspeaker",
    *,
    gmres_log_every: int = 10,
    gmres_log_residuals: bool = True,
) -> "Loudspeaker":
    """Enable lightweight per-iteration GMRES progress logs."""
    return speaker._with_state(
        solver_options=replace(
            speaker.state.solver_options,
            gmres_log_every=int(gmres_log_every),
            gmres_log_residuals=bool(gmres_log_residuals),
        )
    )


def use_osrc(speaker: "Loudspeaker", npade: int = 2) -> "Loudspeaker":
    return speaker._with_state(use_osrc=True, osrc_npade=npade)
