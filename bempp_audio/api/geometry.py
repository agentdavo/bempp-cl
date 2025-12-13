"""
Geometry builders for the `Loudspeaker` fluent API.

These helpers mutate a `Loudspeaker` instance and return it for chaining.
Keeping the implementation outside `api/loudspeaker.py` reduces the size of
the façade module and clarifies separation of concerns.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional
from pathlib import Path
import numpy as np

from bempp_audio.mesh import LoudspeakerMesh
from bempp_audio.mesh.waveguide import WaveguideMeshConfig
from bempp_audio.mesh.unified_enclosure import UnifiedMeshConfig
from bempp_audio.velocity import VelocityProfile
from bempp_audio.progress import get_logger
from bempp_audio.api.state import WaveguideMetadata
from bempp_audio.baffles import FreeSpace
from bempp_audio.acoustic_reference import AcousticReference


def circular_piston(
    speaker: "Loudspeaker",
    radius: Optional[float] = None,
    *,
    diameter: Optional[float] = None,
    element_size: Optional[float] = None,
) -> "Loudspeaker":
    if radius is None and diameter is None:
        raise TypeError("Provide `radius` or `diameter`.")
    if radius is not None and diameter is not None:
        raise TypeError("Provide only one of `radius` or `diameter`.")
    if radius is None:
        radius = float(diameter) / 2.0

    mesh = LoudspeakerMesh.circular_piston(radius=float(radius), element_size=element_size)
    return speaker._with_state(
        mesh=mesh,
        reference=AcousticReference.from_mesh(mesh, default_distance_m=float(speaker.state.measurement_distance)),
        waveguide=None,
    )


def annular_piston(
    speaker: "Loudspeaker",
    inner_radius: float,
    outer_radius: float,
    element_size: Optional[float] = None,
) -> "Loudspeaker":
    mesh = LoudspeakerMesh.annular_piston(inner_radius, outer_radius, element_size=element_size)
    return speaker._with_state(
        mesh=mesh,
        reference=AcousticReference.from_mesh(mesh, default_distance_m=float(speaker.state.measurement_distance)),
        waveguide=None,
    )


def cone(
    speaker: "Loudspeaker",
    inner_r: float,
    outer_r: float,
    height: float,
    curvature: float = 0.0,
    element_size: Optional[float] = None,
) -> "Loudspeaker":
    logger = get_logger()
    logger.info(
        f"Creating cone mesh: inner_r={inner_r*1000:.1f}mm, "
        f"outer_r={outer_r*1000:.1f}mm, height={height*1000:.1f}mm"
    )
    mesh = LoudspeakerMesh.cone_profile(inner_r, outer_r, height, curvature, element_size=element_size)
    mesh_info = mesh.info()
    logger.info(f"Mesh created: {mesh_info.n_vertices} vertices, {mesh_info.n_elements} elements")
    return speaker._with_state(
        mesh=mesh,
        reference=AcousticReference.from_mesh(mesh, default_distance_m=float(speaker.state.measurement_distance)),
        waveguide=None,
    )


def dome(
    speaker: "Loudspeaker",
    radius: float,
    depth: float,
    profile: str = "spherical",
    element_size: Optional[float] = None,
) -> "Loudspeaker":
    mesh = LoudspeakerMesh.dome(radius, depth, profile, element_size=element_size)
    return speaker._with_state(
        mesh=mesh,
        reference=AcousticReference.from_mesh(mesh, default_distance_m=float(speaker.state.measurement_distance)),
        waveguide=None,
    )


def waveguide(
    speaker: "Loudspeaker",
    throat_diameter: float,
    mouth_diameter: float,
    length: float,
    profile: str = "exponential",
    mesh_preset: Optional[str] = None,
    throat_element_size: Optional[float] = None,
    mouth_element_size: Optional[float] = None,
    throat_velocity_amplitude: float = 0.01,
    # Default corresponds to 96 axial points (n_axial_slices + 1).
    n_axial_slices: int = 95,
    n_circumferential: int = 36,
    corner_resolution: int = 0,
    export_mesh: bool = False,
    export_mesh_path: str | Path | None = None,
    show_mesh_quality: bool = True,
    throat_edge_refine: bool = False,
    throat_edge_size: Optional[float] = None,
    throat_edge_dist_min: Optional[float] = None,
    throat_edge_dist_max: Optional[float] = None,
    throat_edge_sampling: int = 80,
    lock_throat_boundary: bool = False,
    throat_circle_points: Optional[int] = None,
    mesh_optimize: Optional[str] = None,
    morph: Optional["MorphConfig"] = None,
    morph_rate: Optional[float] = None,
    morph_fixed_part: Optional[float] = None,
    morph_end_part: Optional[float] = None,
    os_opening_angle_deg: Optional[float] = None,
    cts_driver_exit_angle_deg: Optional[float] = None,
    cts_throat_blend: Optional[float] = None,
    cts_transition: Optional[float] = None,
    cts_throat_angle_deg: Optional[float] = None,
    cts_tangency: Optional[float] = None,
    cts_mouth_roll: Optional[float] = None,
    cts_curvature_regularizer: Optional[float] = None,
    cts_mid_curvature: Optional[float] = None,
) -> "Loudspeaker":
    from bempp_audio.mesh import create_waveguide_mesh

    logger = get_logger()
    if (morph_rate is not None or morph_fixed_part is not None or morph_end_part is not None) and morph is None:
        raise ValueError("morph_rate/morph_fixed_part/morph_end_part require `morph=...` to be set")
    if morph is not None:
        if morph_rate is not None:
            morph = replace(morph, rate=float(morph_rate))
        if morph_fixed_part is not None:
            morph = replace(morph, fixed_part=float(morph_fixed_part))
        if morph_end_part is not None:
            morph = replace(morph, end_part=float(morph_end_part))

    preset_name = mesh_preset or speaker.state.mesh_preset
    if throat_element_size is None:
        from bempp_audio.mesh.validation import MeshResolutionPresets

        throat_element_size = MeshResolutionPresets.get_preset(preset_name or "standard")["h_throat"]
    if mouth_element_size is None:
        from bempp_audio.mesh.validation import MeshResolutionPresets

        mouth_element_size = MeshResolutionPresets.get_preset(preset_name or "standard")["h_mouth"]

    if preset_name is not None:
        from bempp_audio.mesh.validation import MeshResolutionPresets

        preset = MeshResolutionPresets.get_preset(preset_name)
        if int(n_axial_slices) == 95 and "n_axial_slices" in preset:
            n_axial_slices = int(preset["n_axial_slices"])
        if int(n_circumferential) == 36 and "n_circumferential" in preset:
            n_circumferential = int(preset["n_circumferential"])
        if int(corner_resolution) == 0 and "corner_resolution" in preset:
            corner_resolution = int(preset["corner_resolution"])

    logger.info(
        f"Creating {profile} waveguide: "
        f"throat={throat_diameter*1000:.1f}mm, "
        f"mouth={mouth_diameter*1000:.1f}mm, "
        f"length={length*1000:.1f}mm"
    )
    if morph is not None:
        try:
            from bempp_audio.mesh.morph import MorphTargetShape

            if morph.target_shape != MorphTargetShape.KEEP:
                shape = morph.target_shape.name.lower()
                details = []
                if float(morph.target_width) > 0 and float(morph.target_height) > 0:
                    details.append(f"{morph.target_width*1000:.0f}×{morph.target_height*1000:.0f}mm")
                if float(getattr(morph, "corner_radius", 0.0)) > 0:
                    details.append(f"r={float(morph.corner_radius)*1000:.1f}mm")
                details_str = (" " + " ".join(details)) if details else ""
                logger.info(
                    f"Morph: circle → {shape}{details_str} "
                    f"(mode={getattr(morph, 'profile_mode', 'blend')}, fixed={float(morph.fixed_part):.2f}, rate={float(morph.rate):.2f})"
                )
        except Exception:
            pass

    wg_mesh, throat_mask = create_waveguide_mesh(
        throat_diameter=throat_diameter,
        mouth_diameter=mouth_diameter,
        length=length,
        profile_type=profile,
        flare_constant=0.0,
        os_opening_angle_deg=os_opening_angle_deg,
        cts_driver_exit_angle_deg=cts_driver_exit_angle_deg,
        cts_throat_blend=cts_throat_blend if cts_throat_blend is not None else 0.0,
        cts_transition=cts_transition if cts_transition is not None else 0.75,
        cts_throat_angle_deg=cts_throat_angle_deg,
        cts_tangency=cts_tangency if cts_tangency is not None else 1.0,
        cts_mouth_roll=float(cts_mouth_roll) if cts_mouth_roll is not None else 0.0,
        cts_curvature_regularizer=float(cts_curvature_regularizer) if cts_curvature_regularizer is not None else 1.0,
        cts_mid_curvature=float(cts_mid_curvature) if cts_mid_curvature is not None else 0.0,
        h_throat=throat_element_size,
        h_mouth=mouth_element_size,
        grading_type="exponential",
        n_axial_slices=n_axial_slices,
        n_circumferential=n_circumferential,
        corner_resolution=int(corner_resolution),
        throat_edge_refine=bool(throat_edge_refine),
        throat_edge_size=throat_edge_size,
        throat_edge_dist_min=throat_edge_dist_min,
        throat_edge_dist_max=throat_edge_dist_max,
        throat_edge_sampling=int(throat_edge_sampling),
        lock_throat_boundary=bool(lock_throat_boundary),
        throat_circle_points=throat_circle_points,
        mesh_optimize=mesh_optimize,
        morph=morph,
    )

    domain_indices = wg_mesh.grid.domain_indices
    throat_domains = domain_indices[throat_mask]
    wall_domains = domain_indices[~throat_mask]
    throat_domain = int(np.bincount(throat_domains.astype(int)).argmax()) if throat_domains.size else 1
    wall_domain = int(np.bincount(wall_domains.astype(int)).argmax()) if wall_domains.size else 2

    n_throat = int(np.sum(throat_mask))
    n_walls = int(np.sum(~throat_mask))
    mesh_info = wg_mesh.info()
    logger.info(
        f"Mesh created: {mesh_info.n_elements} elements "
        f"({n_throat} throat [domain {throat_domain}], "
        f"{n_walls} walls [domain {wall_domain}])"
    )

    velocity = VelocityProfile.by_domain(
        {
            throat_domain: VelocityProfile.piston(amplitude=throat_velocity_amplitude),
            wall_domain: VelocityProfile.zero(),
        }
    )

    logger.info(f"Velocity: throat={throat_velocity_amplitude*1000:.2f} mm/s (piston), walls=0 (rigid)")

    if show_mesh_quality:
        wg_mesh.print_quality_report(by_domain=True, units="mm")

    if export_mesh_path is not None:
        export_mesh = True

    if mesh_info.n_vertices > 5000:
        preset_hint = f" (mesh_preset={preset_name})" if preset_name is not None else ""
        logger.warning(
            f"Large mesh{preset_hint}: {mesh_info.n_vertices} vertices, {mesh_info.n_elements} elements. "
            "Consider a coarser mesh preset (e.g., super_fast/ultra_fast) for faster iteration."
        )

    if export_mesh:
        try:
            from bempp_audio.viz import mesh_3d_html

            out_path = Path(export_mesh_path) if export_mesh_path is not None else Path("waveguide_mesh_3d.html")
            logger.info(f"Generating interactive 3D mesh visualization ({mesh_info.n_vertices} vertices)...")
            mesh_3d_html(
                wg_mesh,
                filename=str(out_path),
                title=f"{profile.capitalize()} Waveguide "
                f"({throat_diameter*1000:.0f}mm → {mouth_diameter*1000:.0f}mm)",
                color_by_domain=True,
                domain_names={
                    throat_domain: f"Throat ({throat_diameter*1000:.0f}mm)",
                    wall_domain: f"Walls ({profile})",
                },
                domain_colors={
                    throat_domain: "#FF4444",
                    wall_domain: "#4488FF",
                },
                show_edges=True,
                opacity=0.95,
            )
            logger.success(f"Saved: {out_path}")
        except Exception as e:
            logger.warning(f"Mesh export skipped: {e}")

    return speaker._with_state(
        mesh=wg_mesh,
        velocity=velocity,
        reference=AcousticReference.from_mesh(wg_mesh, default_distance_m=float(speaker.state.measurement_distance)),
        waveguide=WaveguideMetadata(
            throat_diameter=float(throat_diameter),
            mouth_diameter=float(mouth_diameter),
            length=float(length),
            profile=str(profile),
            throat_domain=int(throat_domain),
            wall_domain=int(wall_domain),
            mouth_center=(0.0, 0.0, 0.0),
            cts_throat_blend=float(cts_throat_blend) if cts_throat_blend is not None else 0.0,
            cts_transition=float(cts_transition) if cts_transition is not None else 0.75,
            cts_tangency=float(cts_tangency) if cts_tangency is not None else 1.0,
            cts_driver_exit_angle_deg=cts_driver_exit_angle_deg,
            cts_throat_angle_deg=cts_throat_angle_deg,
            os_opening_angle_deg=os_opening_angle_deg,
        ),
    )


def waveguide_from_config(
    speaker: "Loudspeaker",
    config: WaveguideMeshConfig,
    throat_velocity_amplitude: float = 0.01,
    export_mesh: bool = False,
    export_mesh_path: str | Path | None = None,
    show_mesh_quality: bool = True,
) -> "Loudspeaker":
    """Build waveguide from a WaveguideMeshConfig, preserving all profile parameters."""
    from bempp_audio.mesh.waveguide import WaveguideMeshGenerator

    logger = get_logger()

    logger.info(
        f"Creating {config.profile_type} waveguide: "
        f"throat={config.throat_diameter*1000:.1f}mm, "
        f"mouth={config.mouth_diameter*1000:.1f}mm, "
        f"length={config.length*1000:.1f}mm"
    )

    # Build mesh using the config directly (preserves CTS params)
    generator = WaveguideMeshGenerator(config)
    wg_mesh, throat_mask = generator.generate()

    domain_indices = wg_mesh.grid.domain_indices
    throat_domains = domain_indices[throat_mask]
    wall_domains = domain_indices[~throat_mask]
    throat_domain = int(np.bincount(throat_domains.astype(int)).argmax()) if throat_domains.size else 1
    wall_domain = int(np.bincount(wall_domains.astype(int)).argmax()) if wall_domains.size else 2

    n_throat = int(np.sum(throat_mask))
    n_walls = int(np.sum(~throat_mask))
    mesh_info = wg_mesh.info()
    logger.info(
        f"Mesh created: {mesh_info.n_elements} elements "
        f"({n_throat} throat [domain {throat_domain}], "
        f"{n_walls} walls [domain {wall_domain}])"
    )

    velocity = VelocityProfile.by_domain(
        {
            throat_domain: VelocityProfile.piston(amplitude=throat_velocity_amplitude),
            wall_domain: VelocityProfile.zero(),
        }
    )

    logger.info(f"Velocity: throat={throat_velocity_amplitude*1000:.2f} mm/s (piston), walls=0 (rigid)")

    if show_mesh_quality:
        wg_mesh.print_quality_report(by_domain=True, units="mm")

    if export_mesh_path is not None:
        export_mesh = True

    if mesh_info.n_vertices > 5000:
        logger.warning(
            f"Large mesh: {mesh_info.n_vertices} vertices, {mesh_info.n_elements} elements. "
            "Consider increasing element sizes or using fewer axial/circumferential samples for faster iteration."
        )

    if export_mesh:
        try:
            from bempp_audio.viz import mesh_3d_html

            out_path = Path(export_mesh_path) if export_mesh_path is not None else Path("waveguide_mesh_3d.html")
            logger.info(f"Generating interactive 3D mesh visualization ({mesh_info.n_vertices} vertices)...")
            mesh_3d_html(
                wg_mesh,
                filename=str(out_path),
                title=f"{config.profile_type.upper()} Waveguide "
                f"({config.throat_diameter*1000:.0f}mm → {config.mouth_diameter*1000:.0f}mm)",
                color_by_domain=True,
                domain_names={
                    throat_domain: f"Throat ({config.throat_diameter*1000:.0f}mm)",
                    wall_domain: f"Walls ({config.profile_type})",
                },
                domain_colors={
                    throat_domain: "#FF4444",
                    wall_domain: "#4488FF",
                },
                show_edges=True,
                opacity=0.95,
            )
            logger.success(f"Saved: {out_path}")
        except Exception as e:
            logger.warning(f"Mesh export skipped: {e}")

    return speaker._with_state(
        mesh=wg_mesh,
        velocity=velocity,
        reference=AcousticReference.from_mesh(wg_mesh, default_distance_m=float(speaker.state.measurement_distance)),
        waveguide=WaveguideMetadata(
            throat_diameter=float(config.throat_diameter),
            mouth_diameter=float(config.mouth_diameter),
            length=float(config.length),
            profile=str(config.profile_type),
            throat_domain=int(throat_domain),
            wall_domain=int(wall_domain),
            mouth_center=(0.0, 0.0, 0.0),
            cts_throat_blend=float(config.cts_throat_blend),
            cts_transition=float(config.cts_transition),
            cts_tangency=float(config.cts_tangency),
            cts_driver_exit_angle_deg=config.cts_driver_exit_angle_deg,
            cts_throat_angle_deg=config.cts_throat_angle_deg,
            cts_mouth_roll=float(getattr(config, "cts_mouth_roll", 0.0)),
            cts_curvature_regularizer=float(getattr(config, "cts_curvature_regularizer", 1.0)),
            cts_mid_curvature=float(getattr(config, "cts_mid_curvature", 0.0)),
            os_opening_angle_deg=config.os_opening_angle_deg,
        ),
    )


def from_stl(speaker: "Loudspeaker", filename: str, scale: float = 1.0) -> "Loudspeaker":
    mesh = LoudspeakerMesh.from_stl(filename, scale)
    return speaker._with_state(
        mesh=mesh,
        reference=AcousticReference.from_mesh(mesh, default_distance_m=float(speaker.state.measurement_distance)),
        waveguide=None,
    )


def from_mesh(speaker: "Loudspeaker", mesh: LoudspeakerMesh) -> "Loudspeaker":
    return speaker._with_state(
        mesh=mesh,
        reference=AcousticReference.from_mesh(mesh, default_distance_m=float(speaker.state.measurement_distance)),
        waveguide=None,
    )


def waveguide_on_box(
    speaker: "Loudspeaker",
    throat_diameter: float,
    mouth_diameter: float,
    waveguide_length: float,
    box_width: float,
    box_height: float,
    box_depth: float,
    profile: str = "exponential",
    mount_position: Optional[tuple[float, float]] = None,
    mesh_preset: Optional[str] = None,
    throat_element_size: Optional[float] = None,
    mouth_element_size: Optional[float] = None,
    box_element_size: Optional[float] = None,
    baffle_element_size: Optional[float] = None,
    side_element_size: Optional[float] = None,
    back_element_size: Optional[float] = None,
    cabinet_chamfer_bottom: Optional["ChamferSpec"] = None,
    cabinet_chamfer_top: Optional["ChamferSpec"] = None,
    cabinet_chamfer_left: Optional["ChamferSpec"] = None,
    cabinet_chamfer_right: Optional["ChamferSpec"] = None,
    cabinet_fillet_bottom_radius: float = 0.0,
    cabinet_fillet_top_radius: float = 0.0,
    cabinet_fillet_left_radius: float = 0.0,
    cabinet_fillet_right_radius: float = 0.0,
    # Default corresponds to 96 axial points (n_axial_slices + 1).
    n_axial_slices: int = 95,
    n_circumferential: int = 36,
    corner_resolution: int = 0,
    throat_velocity_amplitude: float = 0.01,
    export_mesh: bool = False,
    export_mesh_path: str | Path | None = None,
    show_mesh_quality: bool = True,
    lock_throat_boundary: bool = False,
    throat_circle_points: Optional[int] = None,
    mesh_optimize: Optional[str] = "Netgen",
    morph: Optional["MorphConfig"] = None,
    morph_rate: Optional[float] = None,
    morph_fixed_part: Optional[float] = None,
    morph_end_part: Optional[float] = None,
    os_opening_angle_deg: Optional[float] = None,
    cts_driver_exit_angle_deg: Optional[float] = None,
    cts_throat_blend: Optional[float] = None,
    cts_transition: Optional[float] = None,
    cts_throat_angle_deg: Optional[float] = None,
    cts_tangency: Optional[float] = None,
    cts_mouth_roll: Optional[float] = None,
    cts_curvature_regularizer: Optional[float] = None,
    cts_mid_curvature: Optional[float] = None,
) -> "Loudspeaker":
    from bempp_audio.mesh.unified_enclosure import create_waveguide_on_box_unified

    logger = get_logger()
    if (morph_rate is not None or morph_fixed_part is not None or morph_end_part is not None) and morph is None:
        raise ValueError("morph_rate/morph_fixed_part/morph_end_part require `morph=...` to be set")
    if morph is not None:
        if morph_rate is not None:
            morph = replace(morph, rate=float(morph_rate))
        if morph_fixed_part is not None:
            morph = replace(morph, fixed_part=float(morph_fixed_part))
        if morph_end_part is not None:
            morph = replace(morph, end_part=float(morph_end_part))

    mount_x: Optional[float] = None
    mount_y: Optional[float] = None
    if mount_position is not None:
        mount_x, mount_y = mount_position

    preset_name = mesh_preset or speaker.state.mesh_preset
    if (
        throat_element_size is None
        or mouth_element_size is None
        or box_element_size is None
        or baffle_element_size is None
        or side_element_size is None
        or back_element_size is None
    ):
        from bempp_audio.mesh.validation import MeshResolutionPresets

        defaults = MeshResolutionPresets.get_preset(preset_name or "standard")
        throat_element_size = defaults["h_throat"] if throat_element_size is None else throat_element_size
        mouth_element_size = defaults["h_mouth"] if mouth_element_size is None else mouth_element_size
        box_element_size = defaults["h_box"] if box_element_size is None else box_element_size
        baffle_element_size = defaults.get("h_baffle", defaults["h_mouth"]) if baffle_element_size is None else baffle_element_size
        side_element_size = defaults.get("h_sides", defaults["h_box"]) if side_element_size is None else side_element_size
        back_element_size = defaults.get("h_back", 1.5 * defaults["h_box"]) if back_element_size is None else back_element_size

    if preset_name is not None:
        from bempp_audio.mesh.validation import MeshResolutionPresets

        preset = MeshResolutionPresets.get_preset(preset_name)
        if int(n_axial_slices) == 95 and "n_axial_slices" in preset:
            n_axial_slices = int(preset["n_axial_slices"])
        if int(n_circumferential) == 36 and "n_circumferential" in preset:
            n_circumferential = int(preset["n_circumferential"])
        if int(corner_resolution) == 0 and "corner_resolution" in preset:
            corner_resolution = int(preset["corner_resolution"])

    config = UnifiedMeshConfig(
        throat_diameter=throat_diameter,
        mouth_diameter=mouth_diameter,
        waveguide_length=waveguide_length,
        box_width=box_width,
        box_height=box_height,
        box_depth=box_depth,
        profile_type=profile,
        flare_constant=0.0,
        os_opening_angle_deg=os_opening_angle_deg,
        cts_driver_exit_angle_deg=cts_driver_exit_angle_deg,
        cts_throat_blend=float(cts_throat_blend) if cts_throat_blend is not None else 0.0,
        cts_transition=float(cts_transition) if cts_transition is not None else 0.75,
        cts_throat_angle_deg=cts_throat_angle_deg,
        cts_tangency=float(cts_tangency) if cts_tangency is not None else 1.0,
        cts_mouth_roll=float(cts_mouth_roll) if cts_mouth_roll is not None else 0.0,
        cts_curvature_regularizer=float(cts_curvature_regularizer) if cts_curvature_regularizer is not None else 1.0,
        cts_mid_curvature=float(cts_mid_curvature) if cts_mid_curvature is not None else 0.0,
        mount_x=mount_x,
        mount_y=mount_y,
        h_throat=throat_element_size,
        h_mouth=mouth_element_size,
        h_box=box_element_size,
        h_baffle=baffle_element_size,
        h_sides=side_element_size,
        h_back=back_element_size,
        chamfer_bottom=cabinet_chamfer_bottom,
        chamfer_top=cabinet_chamfer_top,
        chamfer_left=cabinet_chamfer_left,
        chamfer_right=cabinet_chamfer_right,
        fillet_bottom_radius=float(cabinet_fillet_bottom_radius),
        fillet_top_radius=float(cabinet_fillet_top_radius),
        fillet_left_radius=float(cabinet_fillet_left_radius),
        fillet_right_radius=float(cabinet_fillet_right_radius),
        n_axial_slices=n_axial_slices,
        n_circumferential=n_circumferential,
        corner_resolution=corner_resolution,
        lock_throat_boundary=bool(lock_throat_boundary),
        throat_circle_points=throat_circle_points,
        mesh_optimize=mesh_optimize,
        morph=morph,
    )

    logger.info(
        f"Creating waveguide-on-box (unified mesh): throat={throat_diameter*1000:.1f}mm, "
        f"mouth={mouth_diameter*1000:.1f}mm, L={waveguide_length*1000:.1f}mm, "
        f"box={box_width*1000:.0f}x{box_height*1000:.0f}x{box_depth*1000:.0f}mm"
    )

    mesh = create_waveguide_on_box_unified(config)

    if show_mesh_quality:
        mesh.print_quality_report(by_domain=True, units="mm")

    if export_mesh_path is not None:
        export_mesh = True

    if export_mesh:
        try:
            from bempp_audio.viz import mesh_3d_html

            out_path = (
                Path(export_mesh_path) if export_mesh_path is not None else Path("waveguide_on_box_mesh_3d.html")
            )
            mesh_3d_html(
                mesh,
                filename=str(out_path),
                color_by_domain=True,
                domain_names={1: "Throat (Driver)", 2: "Waveguide Walls", 3: "Box Enclosure"},
                domain_colors={1: "#FF4444", 2: "#4488FF", 3: "#44FF44"},
                show_edges=True,
                title="Waveguide on Box (Unified Mesh)",
            )
            logger.success(f"Saved: {out_path}")
        except Exception as e:
            logger.warning(f"Mesh export skipped: {e}")

    mouth_x = config.mount_x if config.mount_x is not None else (box_width / 2)
    mouth_y = config.mount_y if config.mount_y is not None else (box_height / 2)
    reference = AcousticReference.from_origin_axis(
        (float(mouth_x), float(mouth_y), 0.0),
        (0.0, 0.0, 1.0),
        default_distance_m=float(speaker.state.measurement_distance),
    )
    return speaker._with_state(
        mesh=mesh,
        velocity=VelocityProfile.by_domain(
            {
                1: VelocityProfile.piston(amplitude=throat_velocity_amplitude),
                2: VelocityProfile.zero(),
                3: VelocityProfile.zero(),
            }
        ),
        waveguide=WaveguideMetadata(
            throat_diameter=float(throat_diameter),
            mouth_diameter=float(mouth_diameter),
            length=float(waveguide_length),
            profile=str(profile),
            throat_domain=1,
            wall_domain=2,
            mouth_center=(float(mouth_x), float(mouth_y), 0.0),
            cts_throat_blend=float(cts_throat_blend) if cts_throat_blend is not None else 0.0,
            cts_transition=float(cts_transition) if cts_transition is not None else 0.75,
            cts_tangency=float(cts_tangency) if cts_tangency is not None else 1.0,
            cts_driver_exit_angle_deg=cts_driver_exit_angle_deg,
            cts_throat_angle_deg=cts_throat_angle_deg,
            cts_mouth_roll=float(cts_mouth_roll) if cts_mouth_roll is not None else 0.0,
            cts_curvature_regularizer=float(cts_curvature_regularizer) if cts_curvature_regularizer is not None else 1.0,
            cts_mid_curvature=float(cts_mid_curvature) if cts_mid_curvature is not None else 0.0,
            os_opening_angle_deg=os_opening_angle_deg,
        ),
        baffle=FreeSpace(),
        reference=reference,
    )


def waveguide_on_box_from_config(
    speaker: "Loudspeaker",
    config: UnifiedMeshConfig,
    throat_velocity_amplitude: float = 0.01,
    export_mesh: bool = False,
    export_mesh_path: str | Path | None = None,
    show_mesh_quality: bool = True,
) -> "Loudspeaker":
    return waveguide_on_box(
        speaker,
        throat_diameter=config.throat_diameter,
        mouth_diameter=config.mouth_diameter,
        waveguide_length=config.waveguide_length,
        box_width=config.box_width,
        box_height=config.box_height,
        box_depth=config.box_depth,
        profile=config.profile_type,
        mount_position=(config.mount_x, config.mount_y)
        if config.mount_x is not None and config.mount_y is not None
        else None,
        throat_element_size=config.h_throat,
        mouth_element_size=config.h_mouth,
        box_element_size=config.h_box,
        baffle_element_size=config.h_baffle,
        side_element_size=config.h_sides,
        back_element_size=config.h_back,
        n_axial_slices=config.n_axial_slices,
        n_circumferential=config.n_circumferential,
        lock_throat_boundary=bool(getattr(config, "lock_throat_boundary", False)),
        throat_circle_points=getattr(config, "throat_circle_points", None),
        mesh_optimize=getattr(config, "mesh_optimize", "Netgen"),
        throat_velocity_amplitude=throat_velocity_amplitude,
        export_mesh=export_mesh,
        export_mesh_path=export_mesh_path,
        show_mesh_quality=show_mesh_quality,
    )
