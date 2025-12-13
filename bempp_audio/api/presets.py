"""
Preset configuration helpers for the `Loudspeaker` fluent API.

These functions apply directivity/frequency/solver presets to a `Loudspeaker`
instance, returning it for chaining.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np


def preset(speaker: "Loudspeaker", name: str) -> "Loudspeaker":
    from bempp_audio.config import ConfigPresets

    config = ConfigPresets.get_preset(name)

    if config.frequency:
        freq_cfg = config.frequency
        if freq_cfg.spacing == "log":
            speaker.frequency_range(freq_cfg.f_start, freq_cfg.f_end, num=freq_cfg.num_points, spacing="log")
        elif freq_cfg.spacing == "linear":
            speaker.frequency_range(freq_cfg.f_start, freq_cfg.f_end, num=freq_cfg.num_points, spacing="linear")
        elif freq_cfg.spacing == "octave":
            speaker.frequency_range(
                freq_cfg.f_start,
                freq_cfg.f_end,
                points_per_octave=int(freq_cfg.num_points),
            )

    if config.directivity:
        dir_cfg = config.directivity
        speaker.polar_angles(dir_cfg.polar_start, dir_cfg.polar_end, dir_cfg.polar_num)
        if dir_cfg.measurement_distance:
            speaker.measurement_distance(dir_cfg.measurement_distance)
        if dir_cfg.normalize_angle is not None:
            speaker.normalize_to(dir_cfg.normalize_angle)
        if dir_cfg.spl_angles:
            speaker.spl_angles(dir_cfg.spl_angles)

    if config.solver:
        solver_cfg = config.solver
        solver_options = speaker.state.solver_options
        if solver_cfg.tol:
            solver_options = replace(solver_options, tol=solver_cfg.tol)
        if solver_cfg.maxiter:
            solver_options = replace(solver_options, maxiter=solver_cfg.maxiter)
        if solver_cfg.use_fmm is not None:
            solver_options = replace(solver_options, use_fmm=solver_cfg.use_fmm)
        # Only apply expansion order when FMM is enabled to avoid unnecessary
        # overrides of the default setting.
        if bool(solver_cfg.use_fmm) and getattr(solver_cfg, "fmm_expansion_order", None) is not None:
            solver_options = replace(solver_options, fmm_expansion_order=int(solver_cfg.fmm_expansion_order))
        speaker._with_state(solver_options=solver_options)

    return speaker


def solver_preset(speaker: "Loudspeaker", name: str) -> "Loudspeaker":
    """
    Apply an iterative-solver performance preset.

    This is intentionally narrow (tol/maxiter/use_fmm) so it can be combined
    with geometry/mesh choices independently of the full `ConfigPresets` system.
    """
    from dataclasses import replace

    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    presets = {
        "ultra-fast": {"tol": 3e-4, "maxiter": 400, "use_fmm": False},
        "super-fast": {"tol": 1e-4, "maxiter": 600, "use_fmm": False},
        "fast": {"tol": 3e-5, "maxiter": 800, "use_fmm": False},
        "standard": {"tol": 1e-5, "maxiter": 1000, "use_fmm": False},
        "slow": {"tol": 3e-6, "maxiter": 1500, "use_fmm": False},
    }
    if key not in presets:
        raise ValueError(f"Unknown solver preset '{name}'. Available: {sorted(presets.keys())}")

    cfg = presets[key]
    options = speaker.state.solver_options
    options = replace(options, tol=float(cfg["tol"]), maxiter=int(cfg["maxiter"]), use_fmm=bool(cfg["use_fmm"]))
    speaker._with_state(solver_options=options)
    return speaker


def performance_preset(speaker: "Loudspeaker", name: str, *, mode: str = "horn") -> "Loudspeaker":
    """
    Apply a unified performance preset for common waveguide workflows.

    Sets:
    - `speaker.state.mesh_preset` (used as a default by waveguide builders)
    - iterative solver settings via `solver_preset(...)`
    - common frequency/directivity defaults (selectable by `mode`)
    """
    from bempp_audio.mesh.validation import MeshResolutionPresets

    preset_key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    # Validate preset spelling early.
    MeshResolutionPresets.get_preset(preset_key)

    speaker._with_state(mesh_preset=preset_key)
    solver_preset(speaker, preset_key)

    mode_key = str(mode).strip().lower().replace("_", "-").replace(" ", "-")
    if mode_key in ("horn", "waveguide"):
        speaker.frequency_range(200.0, 20000.0, num=20, spacing="log")
        preset_horn(speaker, distance=1.0, max_angle=90.0, resolution_deg=5.0, normalize_angle=10.0)
        return speaker
    if mode_key in ("infinite-baffle", "baffle"):
        speaker.frequency_range(200.0, 20000.0, num=20, spacing="log")
        preset_infinite_baffle(speaker, distance=1.0, resolution_deg=5.0, normalize_angle=0.0)
        return speaker
    if mode_key in ("quick", "smoke"):
        speaker.frequency_range(500.0, 2000.0, num=3, spacing="log")
        preset_horn(speaker, distance=1.0, max_angle=90.0, resolution_deg=10.0, normalize_angle=10.0)
        return speaker

    raise ValueError("Unknown performance preset mode. Use 'horn', 'infinite-baffle', or 'quick'.")


def preset_horn(
    speaker: "Loudspeaker",
    distance: float = 1.0,
    max_angle: float = 90.0,
    resolution_deg: float = 5.0,
    normalize_angle: float = 10.0,
) -> "Loudspeaker":
    num = int(round(float(max_angle) / float(resolution_deg))) + 1
    return (
        speaker.measurement_distance(distance)
        .polar_angles(0.0, max_angle, num)
        .spl_angles([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        .normalize_to(normalize_angle)
    )


def preset_infinite_baffle(
    speaker: "Loudspeaker",
    distance: float = 1.0,
    resolution_deg: float = 5.0,
    normalize_angle: float = 0.0,
) -> "Loudspeaker":
    num = int(round(90.0 / resolution_deg)) + 1
    return (
        speaker.measurement_distance(distance)
        .polar_angles(0.0, 90.0, num)
        .spl_angles([0, 10, 20, 30, 45, 60, 90])
        .normalize_to(normalize_angle)
    )


def preset_cabinet_free_space(
    speaker: "Loudspeaker",
    distance: float = 2.0,
    resolution_deg: float = 5.0,
    normalize_angle: float = 0.0,
) -> "Loudspeaker":
    num = int(round(180.0 / resolution_deg)) + 1
    return (
        speaker.measurement_distance(distance)
        .polar_angles(0.0, 180.0, num)
        .spl_angles([0, 15, 30, 45, 60, 90, 120, 150, 180])
        .normalize_to(normalize_angle)
    )


def preset_nearfield(speaker: "Loudspeaker") -> "Loudspeaker":
    return (
        speaker.measurement_distance(0.5)
        .polar_angles(0, 90, 19)
        .spl_angles([0, 15, 30, 45, 60])
        .normalize_to(0)
    )


def preset_far_field(speaker: "Loudspeaker") -> "Loudspeaker":
    return (
        speaker.measurement_distance(10.0)
        .polar_angles(0, 180, 37)
        .spl_angles([0, 10, 20, 30, 45, 60, 90])
        .normalize_to(0)
    )


def preset_anechoic(speaker: "Loudspeaker") -> "Loudspeaker":
    return (
        speaker.measurement_distance(2.0)
        .polar_angles(0, 180, 73)
        .spl_angles([0, 10, 15, 20, 30, 40, 45, 60, 90])
        .normalize_to(0)
    )


def preset_cea2034(speaker: "Loudspeaker") -> "Loudspeaker":
    return preset_anechoic(speaker)
