"""
Unified waveguide-on-box mesh generation for BEM.

This module provides the CORRECT approach for creating waveguide mounted on
box enclosure with shared edges (no duplicate surfaces).

Key difference from enclosure.py:
- Everything generated in ONE Gmsh session
- Waveguide mouth edge and box hole edge are THE SAME curve object
- No boolean cutting, no vertex welding, no duplicate surfaces
- Topologically correct by construction

Coordinate System
-----------------
See `bempp_audio.mesh.conventions`. The front face (baffle) is `z=0` and the
box extends into negative `z` (depth). Mount positions (`mount_x`, `mount_y`)
are specified in the box-frame on the front face with `x ∈ [0, box_width]`,
`y ∈ [0, box_height]` and origin at the lower-left corner `(0, 0, 0)` when
viewed from the front.

For background on why the boolean cut approach fails, see:
MESH_CUTTING_ISSUE_ANALYSIS.md
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from bempp_audio._optional import optional_import

gmsh, GMSH_AVAILABLE = optional_import("gmsh")
bempp, _BEMPP_AVAILABLE = optional_import("bempp_cl.api")
from bempp_audio.mesh.loudspeaker_mesh import LoudspeakerMesh
from bempp_audio.mesh.backend import gmsh_session, write_temp_msh, import_msh_and_cleanup
from bempp_audio.progress import get_logger
from bempp_audio.mesh.domains import Domain
from bempp_audio.mesh.profiles import conical_profile, exponential_profile, tractrix_profile, cts_profile
from bempp_audio.mesh.morph import MorphConfig, MorphTargetShape, theta_for_morph, morphed_sections_xy
from bempp_audio.mesh.design import os_opening_angle_bounds_deg
from bempp_audio.mesh.cabinet import CabinetConfig, ChamferSpec, create_cabinet_geometry


@dataclass
class UnifiedMeshConfig:
    """Configuration for unified waveguide-on-box mesh generation."""
    
    # Waveguide geometry (required fields first)
    throat_diameter: float
    mouth_diameter: float
    waveguide_length: float
    
    # Box geometry (required fields)
    box_width: float
    box_height: float
    box_depth: float
    
    # Profile (optional fields after required)
    profile_type: str = 'exponential'  # 'conical', 'exponential', 'tractrix'(legacy), 'tractrix_horn', 'os', 'cts'
    flare_constant: float = 0.0        # For exponential (0 = auto)
    os_opening_angle_deg: Optional[float] = None  # For OS profile: half-angle from z-axis
    # CTS profile parameters (match WaveguideMeshConfig where possible)
    cts_driver_exit_angle_deg: Optional[float] = None
    cts_throat_blend: float = 0.0
    cts_transition: float = 0.75       # Normalized transition point t_c in [0, 1)
    cts_throat_angle_deg: Optional[float] = None
    cts_tangency: float = 1.0
    cts_mouth_roll: float = 0.0
    cts_curvature_regularizer: float = 1.0
    cts_mid_curvature: float = 0.0
    
    # Mounting
    mount_x: Optional[float] = None  # None = centered
    mount_y: Optional[float] = None  # None = centered
    
    # Mesh resolution (optimized for 16-20 kHz near the source)
    h_throat: float = 0.002          # Element size at throat (2mm)
    h_mouth: float = 0.008           # Element size at mouth (8mm)
    h_box: float = 0.012             # Element size on box faces (12mm) - λ/6 @ 4.8 kHz
    h_baffle: Optional[float] = None  # Front face (baffle) target size (defaults to h_mouth)
    h_sides: Optional[float] = None   # Side/top/bottom target size at z=0 (defaults to h_box)
    h_back: Optional[float] = None    # Back wall target size at z=-depth (defaults to 1.5*h_sides)

    # Optional baffle-edge blends (front perimeter edges, where baffle meets enclosure walls).
    # Distances in meters. Chamfers support asymmetric/angled variants via ChamferSpec.
    chamfer_bottom: Optional[ChamferSpec] = None
    chamfer_top: Optional[ChamferSpec] = None
    chamfer_left: Optional[ChamferSpec] = None
    chamfer_right: Optional[ChamferSpec] = None
    fillet_bottom_radius: float = 0.0
    fillet_top_radius: float = 0.0
    fillet_left_radius: float = 0.0
    fillet_right_radius: float = 0.0
    
    # Waveguide discretization
    # Profile slices along the waveguide axis. Note: geometry uses (n_axial_slices + 1) axial points.
    # Default 95 slices -> 96 axial points.
    n_axial_slices: int = 95
    n_circumferential: int = 36      # Throat/mouth circumference (10° resolution)
    corner_resolution: int = 0       # Per-corner extra samples for rounded rectangles (0 disables)

    # Throat boundary locking (for stable throat discretization / coupling).
    lock_throat_boundary: bool = False
    throat_circle_points: Optional[int] = None

    # Optional mesh optimization (can move nodes). Default keeps previous behavior.
    mesh_optimize: Optional[str] = "Netgen"

    # Cross-section morphing (non-axisymmetric waveguides via lofting)
    morph: Optional[MorphConfig] = None
    
    def __post_init__(self):
        """Set defaults and validate."""
        if self.mount_x is None:
            self.mount_x = self.box_width / 2
        if self.mount_y is None:
            self.mount_y = self.box_height / 2
        
        # Validation
        if self.throat_diameter >= self.mouth_diameter:
            raise ValueError("Throat diameter must be < mouth diameter")
        if self.waveguide_length <= 0:
            raise ValueError("Waveguide length must be > 0")
        if self.mount_x < 0 or self.mount_x > self.box_width:
            raise ValueError("mount_x must be within box width")
        if self.mount_y < 0 or self.mount_y > self.box_height:
            raise ValueError("mount_y must be within box height")
        if self.profile_type == "cts":
            if not (0.0 <= float(self.cts_throat_blend) < 1.0):
                raise ValueError("cts_throat_blend must be in [0, 1)")
            if not (0.0 <= float(self.cts_transition) < 1.0):
                raise ValueError("cts_transition must be in [0, 1)")
            if float(self.cts_throat_blend) >= float(self.cts_transition):
                raise ValueError("cts_throat_blend must be < cts_transition")
            if not (0.0 <= float(self.cts_tangency) <= 1.0):
                raise ValueError("cts_tangency must be in [0, 1]")
            if not (0.0 <= float(self.cts_mouth_roll) < 1.0):
                raise ValueError("cts_mouth_roll must be in [0, 1)")
            if float(self.cts_curvature_regularizer) < 0.0:
                raise ValueError("cts_curvature_regularizer must be >= 0")
            if not (0.0 <= float(self.cts_mid_curvature) <= 1.0):
                raise ValueError("cts_mid_curvature must be in [0, 1]")
        if self.profile_type in ("os", "oblate_spheroidal") and self.os_opening_angle_deg is not None:
            a = float(self.os_opening_angle_deg)
            if not (0.0 < a < 89.9):
                raise ValueError("os_opening_angle_deg must be in (0, 89.9)")
            throat_r = float(self.throat_diameter) / 2.0
            mouth_r = float(self.mouth_diameter) / 2.0
            amin, amax = os_opening_angle_bounds_deg(
                throat_radius=throat_r, mouth_radius=mouth_r, length=float(self.waveguide_length)
            )
            if a + 1e-12 < float(amin) or a - 1e-12 > float(amax):
                raise ValueError(f"os_opening_angle_deg must be in [{amin:.6g}, {amax:.6g}] for this geometry")
        if int(self.corner_resolution) < 0:
            raise ValueError("corner_resolution must be >= 0")
        if self.morph is not None:
            self.morph.validate()

        # Mesh sizing defaults (front/side/back)
        if self.h_baffle is None:
            self.h_baffle = float(self.h_mouth)
        if self.h_sides is None:
            self.h_sides = float(self.h_box)
        if self.h_back is None:
            self.h_back = float(self.h_sides) * 1.5

        # Edge-blend validation
        for name, r in [
            ("fillet_bottom_radius", self.fillet_bottom_radius),
            ("fillet_top_radius", self.fillet_top_radius),
            ("fillet_left_radius", self.fillet_left_radius),
            ("fillet_right_radius", self.fillet_right_radius),
        ]:
            if float(r) < 0:
                raise ValueError(f"{name} must be >= 0")

        if self.chamfer_top is not None and float(self.fillet_top_radius) > 0:
            raise ValueError("Cannot specify both chamfer_top and fillet_top_radius")
        if self.chamfer_bottom is not None and float(self.fillet_bottom_radius) > 0:
            raise ValueError("Cannot specify both chamfer_bottom and fillet_bottom_radius")
        if self.chamfer_left is not None and float(self.fillet_left_radius) > 0:
            raise ValueError("Cannot specify both chamfer_left and fillet_left_radius")
        if self.chamfer_right is not None and float(self.fillet_right_radius) > 0:
            raise ValueError("Cannot specify both chamfer_right and fillet_right_radius")

        if self.throat_circle_points is not None:
            n = int(self.throat_circle_points)
            if n < 8:
                raise ValueError("throat_circle_points must be >= 8")
        if self.mesh_optimize is not None and not str(self.mesh_optimize).strip():
            raise ValueError("mesh_optimize must be a non-empty string or None")


def create_waveguide_on_box_unified(config: UnifiedMeshConfig) -> LoudspeakerMesh:
    """
    Generate waveguide mounted on box enclosure with SHARED edges.
    
    This is the correct BEM approach - no duplicate surfaces!
    
    Parameters
    ----------
    config : UnifiedMeshConfig
        Complete configuration for waveguide and box.
    
    Returns
    -------
    LoudspeakerMesh
        Unified mesh with 3 domains:
        - Domain 1: Throat cap (vibrating)
        - Domain 2: Waveguide walls (rigid)
        - Domain 3: Box faces (rigid)
    
    Examples
    --------
    >>> config = UnifiedMeshConfig(
    ...     throat_diameter=0.025,
    ...     mouth_diameter=0.15,
    ...     waveguide_length=0.10,
    ...     box_width=0.30,
    ...     box_height=0.40,
    ...     box_depth=0.25,
    ... )
    >>> mesh = create_waveguide_on_box_unified(config)
    >>> # mesh is watertight with shared edges, no duplicates!
    """
    if not GMSH_AVAILABLE:
        raise ImportError("gmsh required. Install with: pip install gmsh")

    logger = get_logger()

    logger.info(
        "Generating unified waveguide-on-box mesh "
        f"(throat={config.throat_diameter*1e3:.1f}mm mouth={config.mouth_diameter*1e3:.1f}mm "
        f"L={config.waveguide_length*1e3:.0f}mm box={config.box_width*1e3:.0f}x{config.box_height*1e3:.0f}x{config.box_depth*1e3:.0f}mm "
        f"mount=({float(config.mount_x)*1e3:.0f},{float(config.mount_y)*1e3:.0f})mm profile={config.profile_type})"
    )
    
    throat_r = config.throat_diameter / 2
    mouth_r = config.mouth_diameter / 2
    
    with gmsh_session("waveguide_on_box_unified", terminal=0):
        # ====================================================================
        # STEP 1: Create waveguide profile curve
        # ====================================================================
        logger.debug("Step 1: Creating waveguide profile curve")
        
        # Generate profile points (throat to mouth)
        x = np.linspace(0, config.waveguide_length, config.n_axial_slices + 1)
        
        if config.profile_type == 'exponential':
            r = exponential_profile(
                x,
                throat_r,
                mouth_r,
                config.waveguide_length,
                flare_constant=config.flare_constant if config.flare_constant > 0 else None,
            )
        elif config.profile_type == 'conical':
            r = conical_profile(x, throat_r, mouth_r, config.waveguide_length)
        elif config.profile_type == 'tractrix':
            r = tractrix_profile(x, throat_r, mouth_r, config.waveguide_length)
        elif config.profile_type == "tractrix_horn":
            from bempp_audio.mesh.profiles import tractrix_horn_profile

            r = tractrix_horn_profile(x, throat_r, mouth_r, config.waveguide_length)
        elif config.profile_type in ("os", "oblate_spheroidal"):
            from bempp_audio.mesh.profiles import os_profile

            r = os_profile(
                x,
                throat_r,
                mouth_r,
                config.waveguide_length,
                opening_angle_deg=config.os_opening_angle_deg,
            )
        elif config.profile_type == "cts":
            r = cts_profile(
                x,
                throat_r,
                mouth_r,
                config.waveguide_length,
                throat_blend=float(config.cts_throat_blend),
                transition=float(config.cts_transition),
                driver_exit_angle_deg=config.cts_driver_exit_angle_deg,
                throat_angle_deg=config.cts_throat_angle_deg,
                tangency=float(config.cts_tangency),
                mouth_roll=float(getattr(config, "cts_mouth_roll", 0.0)),
                curvature_regularizer=float(getattr(config, "cts_curvature_regularizer", 1.0)),
                mid_curvature=float(getattr(config, "cts_mid_curvature", 0.0)),
            )
        else:
            raise ValueError(f"Unknown profile type: {config.profile_type}")
        
        # Create profile points in 3D (will revolve around z-axis)
        # Coordinate system:
        #   - Mouth at z=0 (flush with box front face)
        #   - Throat at z=-waveguide_length (inside box)
        #   - Centered at (mount_x, mount_y) in box coordinates

        morph = config.morph
        if morph is not None and morph.target_shape != MorphTargetShape.KEEP:
            logger.debug("  Morph enabled: generating lofted waveguide walls + morphed mouth cutout")
            n_circ = int(config.n_circumferential)
            theta = theta_for_morph(
                n_total=n_circ,
                mouth_radius=float(mouth_r),
                morph=morph,
                corner_resolution=int(getattr(config, "corner_resolution", 0)),
            )

            xs_all, ys_all = morphed_sections_xy(
                x=x,
                r=r,
                length=float(config.waveguide_length),
                theta=theta,
                throat_radius=float(throat_r),
                mouth_radius=float(mouth_r),
                morph=morph,
                enforce_n_directions_default=n_circ,
                name_prefix="unified morph cross-section",
            )

            wire_tags = []
            section_curves: list[list[int]] = []
            for i, (xi, xs, ys) in enumerate(zip(x, xs_all, ys_all)):
                z_coord = -config.waveguide_length + float(xi)
                h_here = (
                    float(config.h_throat)
                    if i == 0
                    else (float(config.h_mouth) if i == len(x) - 1 else (float(config.h_throat) + float(config.h_mouth)) / 2.0)
                )
                pt_tags = [
                    gmsh.model.occ.addPoint(float(config.mount_x + px), float(config.mount_y + py), float(z_coord), float(h_here))
                    for px, py in zip(xs, ys)
                ]
                curves = []
                for a, b in zip(pt_tags, pt_tags[1:] + [pt_tags[0]]):
                    curves.append(gmsh.model.occ.addLine(a, b))
                wire_tags.append(gmsh.model.occ.addWire(curves))
                section_curves.append(curves)

            # Throat cap as vertex-matched planar surface.
            throat_loop = gmsh.model.occ.addCurveLoop(section_curves[0])
            throat_cap_tag = gmsh.model.occ.addPlaneSurface([throat_loop])
            logger.debug(
                f"  Throat cap: planar surface at Z={-config.waveguide_length * 1e3:.0f}mm (morphed polyline)"
            )

            # Lofted waveguide walls (non-axisymmetric).
            lofted = gmsh.model.occ.addThruSections(wire_tags, makeSolid=False, makeRuled=True)
            waveguide_wall_tags = [tag for dim, tag in lofted if dim == 2]
            logger.debug(f"  Waveguide walls: {len(waveguide_wall_tags)} surfaces from loft")

            # The mouth loop is the last cross-section (shared edge for baffle hole).
            mouth_loop = gmsh.model.occ.addCurveLoop(section_curves[-1])
        else:
            profile_pts = []
            for i, (xi, ri) in enumerate(zip(x, r)):
                z_coord = -config.waveguide_length + xi  # Throat at -L, mouth at 0

                pt = gmsh.model.occ.addPoint(
                    config.mount_x + ri,
                    config.mount_y,
                    z_coord,
                    config.h_throat if i == 0 else (config.h_mouth if i == len(x) - 1 else (config.h_throat + config.h_mouth) / 2),
                )
                profile_pts.append(pt)

            logger.debug(f"  Profile: {len(profile_pts)} points, {config.profile_type}")
            logger.debug(
                f"  Throat {throat_r * 1e3:.1f}mm @ Z={-config.waveguide_length * 1e3:.0f}mm; mouth {mouth_r * 1e3:.1f}mm @ Z=0mm"
            )

            # ====================================================================
            # STEP 2: Create waveguide walls by revolution
            # ====================================================================
            logger.debug("Step 2: Creating waveguide walls (revolve)")

            profile_curve = gmsh.model.occ.addSpline(profile_pts)
            revolved = gmsh.model.occ.revolve(
                [(1, profile_curve)],
                config.mount_x,
                config.mount_y,
                0,
                0,
                0,
                1,
                2 * np.pi,
            )
            waveguide_wall_tags = [tag for dim, tag in revolved if dim == 2]
            logger.debug(f"  Waveguide walls: {len(waveguide_wall_tags)} surfaces from revolution")

            # ====================================================================
            # STEP 3: Extract mouth/throat curves from the walls (shared edges)
            # ====================================================================
            logger.debug("Step 3: Extracting mouth/throat curves (shared edges)")
            gmsh.model.occ.synchronize()

            boundary = gmsh.model.getBoundary([(2, int(s)) for s in waveguide_wall_tags], oriented=False, recursive=False)
            curves = [int(tag) for dim, tag in boundary if int(dim) == 1]
            if not curves:
                raise RuntimeError("Failed to find boundary curves for revolved waveguide wall")

            def _pick_circle_at_z(z_target: float, r_target: float) -> int:
                tol_z = max(1e-9, float(config.h_throat) * 0.5)
                candidates: list[tuple[float, int]] = []
                for tag in curves:
                    bbox = gmsh.model.getBoundingBox(1, tag)
                    z_min, z_max = float(bbox[2]), float(bbox[5])
                    if abs(z_min - z_target) > tol_z or abs(z_max - z_target) > tol_z:
                        continue
                    x_min, y_min = float(bbox[0]), float(bbox[1])
                    x_max, y_max = float(bbox[3]), float(bbox[4])
                    cx = 0.5 * (x_min + x_max)
                    cy = 0.5 * (y_min + y_max)
                    rx = 0.5 * (x_max - x_min)
                    ry = 0.5 * (y_max - y_min)
                    avg_r = 0.5 * (rx + ry)
                    center_err = float(np.hypot(cx - float(config.mount_x), cy - float(config.mount_y)))
                    radius_err = abs(avg_r - float(r_target))
                    candidates.append((center_err + radius_err, int(tag)))
                if not candidates:
                    raise RuntimeError(f"Failed to identify waveguide boundary curve at z={z_target:g}")
                return min(candidates, key=lambda p: p[0])[1]

            mouth_curve_tag = _pick_circle_at_z(0.0, float(mouth_r))
            throat_curve_tag = _pick_circle_at_z(-float(config.waveguide_length), float(throat_r))

            mouth_loop = gmsh.model.occ.addCurveLoop([int(mouth_curve_tag)])
            throat_loop = gmsh.model.occ.addCurveLoop([int(throat_curve_tag)])
            throat_cap_tag = gmsh.model.occ.addPlaneSurface([int(throat_loop)])
        
        # ====================================================================
        # STEP 5: Create box faces with front annulus using SHARED mouth curve
        # ====================================================================
        logger.debug("Step 5: Creating box faces with shared mouth edge")

        w = float(config.box_width)
        h = float(config.box_height)
        d = float(config.box_depth)

        h_baffle = float(config.h_baffle)
        h_sides = float(config.h_sides)
        h_back = float(config.h_back)

        h_front_edge = float(min(h_baffle, h_sides))

        has_edge_blends = (
            (config.chamfer_bottom is not None and config.chamfer_bottom.is_active())
            or (config.chamfer_top is not None and config.chamfer_top.is_active())
            or (config.chamfer_left is not None and config.chamfer_left.is_active())
            or (config.chamfer_right is not None and config.chamfer_right.is_active())
            or float(config.fillet_bottom_radius) > 0
            or float(config.fillet_top_radius) > 0
            or float(config.fillet_left_radius) > 0
            or float(config.fillet_right_radius) > 0
        )

        box_edge_info = None
        cabinet_non_baffle_surfaces: list[int]

        if has_edge_blends:
            cab_cfg = CabinetConfig(
                width=w,
                height=h,
                depth=d,
                chamfer_bottom=config.chamfer_bottom if config.chamfer_bottom is not None else ChamferSpec.none(),
                chamfer_top=config.chamfer_top if config.chamfer_top is not None else ChamferSpec.none(),
                chamfer_left=config.chamfer_left if config.chamfer_left is not None else ChamferSpec.none(),
                chamfer_right=config.chamfer_right if config.chamfer_right is not None else ChamferSpec.none(),
                fillet_bottom_radius=float(config.fillet_bottom_radius),
                fillet_top_radius=float(config.fillet_top_radius),
                fillet_left_radius=float(config.fillet_left_radius),
                fillet_right_radius=float(config.fillet_right_radius),
                h_baffle=h_baffle,
                h_sides=h_sides,
                h_back=h_back,
            )
            cab_cfg.validate()

            cab_geom = create_cabinet_geometry(cab_cfg)
            gmsh.model.occ.synchronize()

            if int(cab_geom.volume_tag) <= 0 or int(cab_geom.front_surface) <= 0:
                raise RuntimeError("Cabinet geometry generation failed (missing volume/front surface)")

            # Get all cabinet boundary surfaces from the volume so we don't have to
            # guess which ones are chamfer/fillet transition faces.
            cabinet_boundary = gmsh.model.getBoundary([(3, int(cab_geom.volume_tag))], oriented=False, recursive=False)
            cabinet_surfaces = [int(tag) for dim, tag in cabinet_boundary if int(dim) == 2]
            cabinet_surfaces = [abs(int(s)) for s in cabinet_surfaces]

            # Use the *existing* front curve loop as the outer boundary for the annulus.
            loop_tags, _loop_curves = gmsh.model.occ.getCurveLoops(int(cab_geom.front_surface))
            if len(loop_tags) < 1:
                raise RuntimeError("Failed to read cabinet front face curve loop")
            outer_loop = int(loop_tags[0])

            front_annulus = gmsh.model.occ.addPlaneSurface([outer_loop, int(mouth_loop)])
            gmsh.model.occ.remove([(2, int(cab_geom.front_surface))], recursive=False)

            # Remove the solid volume; keep surfaces (we mesh 2D for BEM).
            gmsh.model.occ.remove([(3, int(cab_geom.volume_tag))], recursive=False)

            cabinet_non_baffle_surfaces = [int(s) for s in cabinet_surfaces if int(s) != int(cab_geom.front_surface)]
            box_face_tags = [int(front_annulus)] + cabinet_non_baffle_surfaces
            logger.info(
                f"  Box faces: {len(box_face_tags)} surfaces (with baffle-edge blends) "
                f"({config.box_width*1000:.0f}x{config.box_height*1000:.0f}x{config.box_depth*1000:.0f}mm)"
            )
        else:
            # Build the box using *shared* corner points and edges so the surface mesh is conformal.
            p000 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, h_front_edge)
            pW00 = gmsh.model.occ.addPoint(w, 0.0, 0.0, h_front_edge)
            pWH0 = gmsh.model.occ.addPoint(w, h, 0.0, h_front_edge)
            p0H0 = gmsh.model.occ.addPoint(0.0, h, 0.0, h_front_edge)

            p00D = gmsh.model.occ.addPoint(0.0, 0.0, -d, h_back)
            pW0D = gmsh.model.occ.addPoint(w, 0.0, -d, h_back)
            pWHD = gmsh.model.occ.addPoint(w, h, -d, h_back)
            p0HD = gmsh.model.occ.addPoint(0.0, h, -d, h_back)

            e_front_bottom = gmsh.model.occ.addLine(p000, pW00)
            e_front_right = gmsh.model.occ.addLine(pW00, pWH0)
            e_front_top = gmsh.model.occ.addLine(pWH0, p0H0)
            e_front_left = gmsh.model.occ.addLine(p0H0, p000)

            e_back_bottom = gmsh.model.occ.addLine(p00D, pW0D)
            e_back_right = gmsh.model.occ.addLine(pW0D, pWHD)
            e_back_top = gmsh.model.occ.addLine(pWHD, p0HD)
            e_back_left = gmsh.model.occ.addLine(p0HD, p00D)

            e_vert_000 = gmsh.model.occ.addLine(p000, p00D)
            e_vert_W00 = gmsh.model.occ.addLine(pW00, pW0D)
            e_vert_WH0 = gmsh.model.occ.addLine(pWH0, pWHD)
            e_vert_0H0 = gmsh.model.occ.addLine(p0H0, p0HD)

            # Store edge info for transfinite meshing (applied after synchronize)
            box_edge_info = {
                # Front edges (baffle perimeter): use the stricter of baffle/side sizing
                e_front_bottom: (w, h_front_edge),
                e_front_right: (h, h_front_edge),
                e_front_top: (w, h_front_edge),
                e_front_left: (h, h_front_edge),
                # Back edges: length, target_h
                e_back_bottom: (w, h_back),
                e_back_right: (h, h_back),
                e_back_top: (w, h_back),
                e_back_left: (h, h_back),
                # Vertical edges: length, average_h (graded from h_sides to h_back)
                e_vert_000: (d, (h_sides + h_back) / 2),
                e_vert_W00: (d, (h_sides + h_back) / 2),
                e_vert_WH0: (d, (h_sides + h_back) / 2),
                e_vert_0H0: (d, (h_sides + h_back) / 2),
            }

            outer_loop = gmsh.model.occ.addCurveLoop([e_front_bottom, e_front_right, e_front_top, e_front_left])
            front_annulus = gmsh.model.occ.addPlaneSurface([outer_loop, mouth_loop])

            logger.debug(
                f"  Front annulus: hole at ({float(config.mount_x) * 1e3:.0f}, {float(config.mount_y) * 1e3:.0f})mm; "
                "mouth edge shared (no duplicates)"
            )

            back_loop = gmsh.model.occ.addCurveLoop([-e_back_left, -e_back_top, -e_back_right, -e_back_bottom])
            back = gmsh.model.occ.addPlaneSurface([back_loop])

            left_loop = gmsh.model.occ.addCurveLoop([e_vert_000, -e_back_left, -e_vert_0H0, e_front_left])
            left = gmsh.model.occ.addPlaneSurface([left_loop])

            right_loop = gmsh.model.occ.addCurveLoop([e_front_right, e_vert_WH0, -e_back_right, -e_vert_W00])
            right = gmsh.model.occ.addPlaneSurface([right_loop])

            bottom_loop = gmsh.model.occ.addCurveLoop([e_vert_000, e_back_bottom, -e_vert_W00, -e_front_bottom])
            bottom = gmsh.model.occ.addPlaneSurface([bottom_loop])

            top_loop = gmsh.model.occ.addCurveLoop([-e_front_top, e_vert_WH0, e_back_top, -e_vert_0H0])
            top = gmsh.model.occ.addPlaneSurface([top_loop])

            cabinet_non_baffle_surfaces = [int(back), int(left), int(right), int(bottom), int(top)]
            box_face_tags = [int(front_annulus)] + cabinet_non_baffle_surfaces

            logger.info(
                f"  Box faces: 6 surfaces ({config.box_width*1000:.0f}x{config.box_height*1000:.0f}x{config.box_depth*1000:.0f}mm)"
            )
        
        # ====================================================================
        # STEP 6: Synchronize and define physical groups (BEM domains)
        # ====================================================================
        gmsh.model.occ.synchronize()
        
        logger.debug("Step 6: Defining physical groups (BEM domains)")
        
        # Domain: Throat cap (vibrating)
        gmsh.model.addPhysicalGroup(2, [throat_cap_tag], int(Domain.THROAT))
        gmsh.model.setPhysicalName(2, int(Domain.THROAT), "Throat")
        
        # Domain: Waveguide walls (rigid)
        gmsh.model.addPhysicalGroup(2, waveguide_wall_tags, int(Domain.WALLS))
        gmsh.model.setPhysicalName(2, int(Domain.WALLS), "WaveguideWalls")
        
        # Domain: Box faces (rigid)
        gmsh.model.addPhysicalGroup(2, box_face_tags, int(Domain.ENCLOSURE))
        gmsh.model.setPhysicalName(2, int(Domain.ENCLOSURE), "BoxEnclosure")
        
        logger.info(f"  Domain 1 (Throat): 1 surface")
        logger.info(f"  Domain 2 (Waveguide): {len(waveguide_wall_tags)} surfaces")
        logger.info(f"  Domain 3 (Box): {len(box_face_tags)} surfaces")
        
        # ====================================================================
        # STEP 7: Set adaptive mesh sizes and generate
        # ====================================================================
        logger.debug("Step 7: Setting adaptive mesh sizes")
        
        # CABINET RESOLUTION (z-dependent) with per-face targets
        # - Baffle/front face: h_baffle
        # - Side/top/bottom: linearly graded from h_sides at z=0 to h_back at z=-depth
        # - Back wall: h_back (achieved by the z-grading)

        h_baffle = float(config.h_baffle)
        h_sides = float(config.h_sides)
        h_back = float(config.h_back)

        # Base z-grading for cabinet walls (non-baffle): h(z) = h_sides + (h_back-h_sides)*clamp(|z|/depth, 0..1)
        math_field = gmsh.model.mesh.field.add("MathEval")
        formula = f"{h_sides} + ({h_back} - {h_sides}) * Min(1.0, Abs(z) / {config.box_depth})"
        gmsh.model.mesh.field.setString(math_field, "F", formula)

        # Restrict the z-grading to non-baffle enclosure surfaces so h_baffle can differ from h_sides.
        cab_restrict = gmsh.model.mesh.field.add("Restrict")
        gmsh.model.mesh.field.setNumber(cab_restrict, "InField", int(math_field))
        gmsh.model.mesh.field.setNumbers(cab_restrict, "SurfacesList", [int(s) for s in cabinet_non_baffle_surfaces])
        cab_field_restricted = int(cab_restrict)

        # Waveguide z-grading (restricted to waveguide surfaces only) so the throat→mouth
        # sizing taper is honored even when the enclosure background field is present.
        wg_field_restricted = None
        try:
            L = float(config.waveguide_length)
            ht = float(config.h_throat)
            hm = float(config.h_mouth)
            ratio = hm / ht if ht > 0 else 1.0

            wg_z = gmsh.model.mesh.field.add("MathEval")
            if ratio <= 0 or abs(ratio - 1.0) < 1e-12 or L <= 0:
                wg_formula = f"{ht}"
            else:
                # z in [-L, 0] -> (z + L)/L in [0, 1]
                wg_formula = f"{ht} * Exp(Log({ratio}) * ((z + {L}) / {L}))"
            gmsh.model.mesh.field.setString(wg_z, "F", str(wg_formula))

            wg_restrict = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(wg_restrict, "InField", int(wg_z))
            gmsh.model.mesh.field.setNumbers(
                wg_restrict,
                "SurfacesList",
                [int(throat_cap_tag)] + [int(s) for s in waveguide_wall_tags],
            )
            wg_field_restricted = int(wg_restrict)
        except Exception:
            wg_field_restricted = None

        logger.info(
            f"  Cabinet resolution: mouth={float(config.h_mouth)*1000:.1f}mm, "
            f"baffle={h_baffle*1000:.1f}mm, sides={h_sides*1000:.1f}mm, back={h_back*1000:.1f}mm"
        )

        # -----------------------------------------------------------------------
        # BAFFLE RADIAL GRADING: h_mouth near waveguide opening → h_baffle at edges
        # Uses MathEval for efficiency (no extra synchronize needed).
        # -----------------------------------------------------------------------
        mouth_r = float(config.mouth_diameter) / 2.0
        mx, my = float(config.mount_x), float(config.mount_y)

        # Compute max distance from mouth center to baffle corners
        corner_dists = [
            np.sqrt((mx - 0)**2 + (my - 0)**2),
            np.sqrt((mx - w)**2 + (my - 0)**2),
            np.sqrt((mx - w)**2 + (my - h)**2),
            np.sqrt((mx - 0)**2 + (my - h)**2),
        ]
        max_corner_dist = max(corner_dists)

        # Radial grading on baffle (z ≈ 0): grades from h_mouth at mouth to h_baffle at edges
        # r = sqrt((x - mx)^2 + (y - my)^2)
        # t = clamp((r - r_min) / (r_max - r_min), 0, 1)
        # h = h_mouth + t * (h_baffle - h_mouth)
        r_min = mouth_r * 1.2  # Start grading just outside mouth
        r_max = max_corner_dist * 0.8  # Reach h_baffle before corners
        hm = float(config.h_mouth)

        # Only apply radial grading near z=0 (baffle); elsewhere use z-grading
        # Formula: if |z| < small, use radial; else use large (will be overridden by z-grading)
        baffle_field = gmsh.model.mesh.field.add("MathEval")
        baffle_formula = (
            f"{hm} + ({h_baffle} - {hm}) * "
            f"Min(1.0, Max(0.0, (Sqrt((x - {mx})^2 + (y - {my})^2) - {r_min}) / {max(1e-9, r_max - r_min)}))"
        )
        gmsh.model.mesh.field.setString(baffle_field, "F", baffle_formula)

        # Restrict to baffle surface only
        baffle_restrict = gmsh.model.mesh.field.add("Restrict")
        gmsh.model.mesh.field.setNumber(baffle_restrict, "InField", baffle_field)
        gmsh.model.mesh.field.setNumbers(baffle_restrict, "SurfacesList", [front_annulus])

        logger.debug(
            f"  Baffle radial grading: {hm*1e3:.1f}mm at r={r_min*1e3:.0f}mm → "
            f"{h_baffle*1e3:.1f}mm at r={r_max*1e3:.0f}mm"
        )

        # -----------------------------------------------------------------------
        # Combine all fields: cabinet z-grading, waveguide grading, baffle radial
        # -----------------------------------------------------------------------
        base_fields = [cab_field_restricted]
        if wg_field_restricted is not None:
            base_fields.append(int(wg_field_restricted))
        base_fields.append(int(baffle_restrict))

        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [int(x) for x in base_fields])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        # Optional transfinite refinement for the simple (non-blended) box.
        if box_edge_info is not None:
            box_edge_nodes = {}
            for edge_tag, (length, target_h) in box_edge_info.items():
                n_nodes = max(3, int(np.ceil(length / target_h)) + 1)
                box_edge_nodes[int(edge_tag)] = n_nodes
                try:
                    gmsh.model.mesh.setTransfiniteCurve(int(edge_tag), int(n_nodes))
                except Exception:
                    pass

            logger.debug(
                f"  Box edge subdivision: front={box_edge_nodes.get(int(e_front_bottom), 0)} nodes, "
                f"vertical={box_edge_nodes.get(int(e_vert_000), 0)} nodes, "
                f"back={box_edge_nodes.get(int(e_back_bottom), 0)} nodes"
            )

        # Fine mesh at throat
        throat_points = gmsh.model.getEntities(0)
        for dim, tag in throat_points:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            z = bbox[2]
            if abs(z + config.waveguide_length) < config.h_throat:
                gmsh.model.mesh.setSize([(0, tag)], config.h_throat)
        
        logger.debug("Step 7b: Generating mesh with adaptive sizing")

        # Optional: lock throat boundary discretization (stable node count).
        if bool(getattr(config, "lock_throat_boundary", False)):
            try:
                gmsh.option.setNumber("Mesh.RandomFactor", 0.0)
            except Exception:
                pass
            try:
                boundary = gmsh.model.getBoundary([(2, int(throat_cap_tag))], oriented=False, recursive=False)
                rim_curves = [int(tag) for dim, tag in boundary if int(dim) == 1]
            except Exception:
                rim_curves = []

            if rim_curves:
                if len(rim_curves) == 1:
                    n = (
                        int(config.throat_circle_points)
                        if config.throat_circle_points is not None
                        else int(config.n_circumferential)
                    )
                    gmsh.model.mesh.setTransfiniteCurve(int(rim_curves[0]), max(8, int(n)))
                else:
                    for c in rim_curves:
                        gmsh.model.mesh.setTransfiniteCurve(int(c), 2)
            else:
                logger.warning("lock_throat_boundary requested, but no throat rim curves were found")
        
        # Generate 2D mesh
        gmsh.model.mesh.generate(2)
        
        # Optional optimization pass (can move nodes; disabled when throat is locked).
        if config.mesh_optimize is not None and not bool(getattr(config, "lock_throat_boundary", False)):
            gmsh.model.mesh.optimize(str(config.mesh_optimize))
        
        # Count elements
        elements = gmsh.model.mesh.getElements(dim=2)
        n_total = sum(len(e) for e in elements[1])
        
        logger.info(f"Mesh generated: {n_total} elements")
        
        # ====================================================================
        # STEP 8: Export to bempp
        # ====================================================================
        logger.debug("Step 8: Exporting to Bempp-CL")
        
        tmp_file = write_temp_msh()
        gmsh.write(tmp_file)
        logger.debug(f"  Temp file: {tmp_file}")

    # Import into bempp
    grid = import_msh_and_cleanup(tmp_file)
    
    mesh = LoudspeakerMesh(grid)
    # Set a meaningful acoustic reference for downstream "on-axis" helpers.
    # For waveguide-on-box, the mouth center is the natural reference point.
    mesh.center = np.array([float(config.mount_x), float(config.mount_y), 0.0])
    mesh.axis = np.array([0.0, 0.0, 1.0])
    
    # Validate
    n_verts = grid.vertices.shape[1]
    n_elems = grid.elements.shape[1]
    domains = np.unique(grid.domain_indices)
    
    logger.info(
        f"Unified mesh complete: vertices={n_verts} elements={n_elems} "
        f"domains={list(domains)} (watertight, shared edges)"
    )
    
    return mesh


# Convenience function for quick usage
def waveguide_on_box(
    throat_diameter: float,
    mouth_diameter: float,
    waveguide_length: float,
    box_width: float,
    box_height: float,
    box_depth: float,
    profile_type: str = 'exponential',
    **kwargs
) -> LoudspeakerMesh:
    """
    Quick interface for waveguide-on-box mesh generation.
    
    Parameters
    ----------
    throat_diameter : float
        Throat diameter in meters (e.g., 0.025 for 1" driver).
    mouth_diameter : float
        Mouth diameter in meters.
    waveguide_length : float
        Waveguide length in meters.
    box_width : float
        Box width in meters.
    box_height : float
        Box height in meters.
    box_depth : float
        Box depth in meters.
    profile_type : str
        'exponential', 'conical', or 'tractrix'.
    **kwargs
        Additional config options (h_throat, h_mouth, h_box, etc.)
    
    Returns
    -------
    LoudspeakerMesh
        Unified mesh with 3 domains (throat, walls, box).
    
    Examples
    --------
    >>> mesh = waveguide_on_box(
    ...     throat_diameter=0.025,
    ...     mouth_diameter=0.15,
    ...     waveguide_length=0.10,
    ...     box_width=0.30,
    ...     box_height=0.40,
    ...     box_depth=0.25,
    ...     profile_type='exponential',
    ...     h_throat=0.005,
    ...     h_box=0.025
    ... )
    """
    config = UnifiedMeshConfig(
        throat_diameter=throat_diameter,
        mouth_diameter=mouth_diameter,
        waveguide_length=waveguide_length,
        box_width=box_width,
        box_height=box_height,
        box_depth=box_depth,
        profile_type=profile_type,
        **{k: v for k, v in kwargs.items() if hasattr(UnifiedMeshConfig, k)}
    )
    
    return create_waveguide_on_box_unified(config)
