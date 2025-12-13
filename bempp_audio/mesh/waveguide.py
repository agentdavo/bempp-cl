"""
Advanced waveguide mesh generation with adaptive element sizing.

This module provides sophisticated waveguide mesh generation using Gmsh OCC
kernel with:
- Multiple profile types (conical, exponential, tractrix, hyperbolic)
- Adaptive mesh sizing (finer at throat, coarser at mouth)
- Controlled circumferential and axial discretization
- Proper sealed throat cap with vertex matching
- Physical group assignment for multi-domain velocity BCs

Key Concepts
------------
- **Mouth plane**: z=0 (front/baffle plane)
- **Throat cap**: Sealed circular disk at z=-length (vibrating driver surface)
- **Waveguide walls**: Axisymmetric surface of revolution (rigid)
- **Mesh grading**: Element size varies from throat to mouth
- **Profile control**: Number of slices along axis for smooth curves

Coordinate System
-----------------
See `bempp_audio.mesh.conventions`. In particular:
- Mouth is at `z=0`
- Throat is at `z=-length`

Physics Considerations
----------------------
For accurate BEM analysis, element size should be < λ/6 where λ is the
shortest wavelength:
- At 20 kHz in air (343 m/s): λ = 17.15 mm → h < 2.86 mm
- Throat typically needs finer mesh (5 mm) than mouth (10 mm)
- Circumferential resolution: 32-48 divisions for smooth circular sections
- Axial resolution: 30-50 slices for smooth exponential/tractrix profiles
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable
import numpy as np

from bempp_audio._optional import optional_import

gmsh, GMSH_AVAILABLE = optional_import("gmsh")
bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")

from bempp_audio.mesh.loudspeaker_mesh import LoudspeakerMesh
from bempp_audio.mesh.backend import gmsh_session, write_temp_msh, import_msh_and_cleanup
from bempp_audio.progress import get_logger
from bempp_audio.mesh.domains import Domain
from bempp_audio.mesh.profiles import (
    conical_profile,
    exponential_profile,
    tractrix_profile,
    tractrix_horn_profile,
    os_profile,
    hyperbolic_profile,
    cts_profile,
)
from bempp_audio.mesh.morph import (
    MorphConfig,
    MorphTargetShape,
    SuperFormulaConfig,
    theta_for_morph,
    morphed_sections_xy,
)
from bempp_audio.mesh.design import os_opening_angle_bounds_deg


def custom_profile(
    x: np.ndarray,
    profile_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    User-defined profile function.
    
    Parameters
    ----------
    x : np.ndarray
        Axial positions (0 to length).
    profile_func : callable
        Function mapping x → radius.
    
    Returns
    -------
    np.ndarray
        Radius at each axial position.
    """
    return profile_func(x)


# =============================================================================
# Mesh Sizing Functions (Adaptive Grading)
# =============================================================================

def linear_grading(x: np.ndarray, h_throat: float, h_mouth: float, length: float) -> np.ndarray:
    """
    Linear element size grading from throat to mouth.
    
    h(x) = h_throat + (h_mouth - h_throat) * x/L
    """
    return h_throat + (h_mouth - h_throat) * (x / length)


def exponential_grading(x: np.ndarray, h_throat: float, h_mouth: float, length: float) -> np.ndarray:
    """
    Exponential element size grading (smoother transition).
    
    h(x) = h_throat * (h_mouth / h_throat)^(x/L)
    """
    if h_throat <= 0 or h_mouth <= 0:
        return np.full_like(x, h_throat)
    ratio = h_mouth / h_throat
    return h_throat * (ratio ** (x / length))


def sigmoid_grading(x: np.ndarray, h_throat: float, h_mouth: float, length: float) -> np.ndarray:
    """
    Sigmoid grading for very smooth transitions.
    
    Uses tanh for smooth S-curve from throat to mouth.
    """
    t = x / length
    # Map tanh from [-2, 2] to [0, 1]
    sigmoid = 0.5 + 0.5 * np.tanh(4 * (t - 0.5))
    return h_throat + (h_mouth - h_throat) * sigmoid


# =============================================================================
# Advanced Waveguide Mesh Generator
# =============================================================================

@dataclass
class WaveguideMeshConfig:
    """Configuration for waveguide mesh generation."""
    # Geometry
    throat_diameter: float
    mouth_diameter: float
    length: float

    # Profile
    profile_type: str = 'exponential'  # 'conical', 'exponential', 'tractrix'(legacy), 'tractrix_horn', 'os', 'hyperbolic', 'cts'
    flare_constant: float = 0.0        # For exponential profile (m⁻¹). 0 = auto-calculate from geometry
    hyperbolic_sharpness: float = 2.0  # For hyperbolic profile

    # OS (oblate spheroidal) profile parameter
    os_opening_angle_deg: Optional[float] = None  # Half-angle from z-axis (often ~target coverage)

    # CTS profile parameters (3-section: throat blend → conical → mouth blend)
    cts_driver_exit_angle_deg: Optional[float] = None  # Driver exit angle (computed from ExitConeConfig if available)
    cts_throat_blend: float = 0.0      # Where throat blend ends [0=no blend, 0.15=15% along length]
    cts_transition: float = 0.75       # Where mouth blend starts [0, 1)
    cts_throat_angle_deg: Optional[float] = None  # Conical section angle (if None, computed to hit mouth_r)
    cts_tangency: float = 1.0          # Mouth tangency [0=conical exit, 1=max achievable toward horizontal]
    cts_mouth_roll: float = 0.0        # Mouth roll length in [0,1) (bias mouth blend to spend longer near-horizontal)
    cts_curvature_regularizer: float = 1.0  # >=0: trades curvature vs tangency target (higher = closer to target)
    cts_mid_curvature: float = 0.0     # Mid-section OS-like curvature amount in [0,1] (0 = pure conical)
    
    # Mesh resolution (optimized for 16-20 kHz near the source)
    h_throat: float = 0.002            # Element size at throat (2 mm) - ~λ/8.6 @ 20 kHz
    h_mouth: float = 0.008             # Element size at mouth (8 mm) - graded coarser
    grading_type: str = 'exponential'  # 'linear', 'exponential', 'sigmoid'
    
    # Discretization
    # Number of profile slices along axis. Note: the geometry uses (n_axial_slices + 1) axial points.
    # Default 95 slices -> 96 axial points.
    n_axial_slices: int = 95
    n_circumferential: int = 36        # Divisions around circumference (10° resolution)
    corner_resolution: int = 0         # When morphing to rounded rectangles: min points per corner arc (0=off)

    # Revolve meridian curve type (avoid spline overshoot by using a polyline).
    meridian_curve: str = "spline"     # "spline" (default) or "polyline"

    # Advanced options
    throat_cap_refinement: float = 0.8  # Refinement factor for throat cap (< 1 = finer)
    use_structured_mesh: bool = False   # Use transfinite meshing (more regular)
    debug: bool = False                 # Enable verbose debug logging

    # Optional mouth-edge refinement (captures aperture edge diffraction region better).
    mouth_edge_refine: bool = False
    mouth_edge_size: Optional[float] = None  # If None: 0.8*h_mouth
    mouth_edge_dist_min: Optional[float] = None  # If None: 0.1*mouth_diameter
    mouth_edge_dist_max: Optional[float] = None  # If None: 0.3*mouth_diameter
    mouth_edge_sampling: int = 100

    # Optional throat-edge refinement (bias quality near the driver/throat interface).
    # This concentrates elements near the throat rim curve at z=-length, where velocity
    # boundary data is applied and where geometric curvature is often highest.
    throat_edge_refine: bool = False
    throat_edge_size: Optional[float] = None  # If None: 0.6*h_throat
    throat_edge_dist_min: Optional[float] = None  # If None: 0.05*throat_diameter
    throat_edge_dist_max: Optional[float] = None  # If None: 0.20*throat_diameter
    throat_edge_sampling: int = 80

    # Throat boundary locking (for stable throat discretization / FEM↔BEM coupling).
    # When enabled, Gmsh curve discretization is constrained so the throat rim node
    # count is reproducible. This is most meaningful for axisymmetric (revolved)
    # waveguides; lofted morph waveguides already have an explicit polyline rim.
    lock_throat_boundary: bool = False
    throat_circle_points: Optional[int] = None  # If None: uses n_circumferential (axisymmetric case)

    # Optional mesh optimization pass (Gmsh). Note: this can move nodes.
    # Set `lock_throat_boundary=True` to disable optimization by default.
    mesh_optimize: Optional[str] = None  # e.g. "Netgen"

    # Cross-section morphing (non-axisymmetric waveguides via lofting)
    morph: Optional[MorphConfig] = None

    def validate(self):
        """Validate configuration parameters."""
        if self.throat_diameter >= self.mouth_diameter:
            raise ValueError("Throat diameter must be less than mouth diameter")
        if self.length <= 0:
            raise ValueError("Length must be positive")
        if self.h_throat <= 0 or self.h_mouth <= 0:
            raise ValueError("Element sizes must be positive")
        if self.n_axial_slices < 10:
            raise ValueError("n_axial_slices must be >= 10 for smooth curves")
        if self.n_circumferential < 16:
            raise ValueError("n_circumferential must be >= 16 for circular sections")
        if int(self.corner_resolution) < 0:
            raise ValueError("corner_resolution must be >= 0")
        if int(self.corner_resolution) > 0 and int(self.corner_resolution) * 4 > int(self.n_circumferential):
            raise ValueError("corner_resolution too large for n_circumferential")
        if str(self.meridian_curve) not in ("spline", "polyline"):
            raise ValueError("meridian_curve must be 'spline' or 'polyline'")
        if self.profile_type == "cts":
            if not (0.0 <= float(self.cts_throat_blend) < 1.0):
                raise ValueError("cts_throat_blend must be in [0, 1)")
            if not (0.0 <= float(self.cts_transition) < 1.0):
                raise ValueError("cts_transition must be in [0, 1)")
            if self.cts_throat_blend >= self.cts_transition:
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
            amin, amax = os_opening_angle_bounds_deg(throat_radius=throat_r, mouth_radius=mouth_r, length=float(self.length))
            if a + 1e-12 < float(amin) or a - 1e-12 > float(amax):
                raise ValueError(f"os_opening_angle_deg must be in [{amin:.6g}, {amax:.6g}] for this geometry")
        if bool(self.mouth_edge_refine):
            if self.mouth_edge_size is not None and float(self.mouth_edge_size) <= 0:
                raise ValueError("mouth_edge_size must be > 0")
            if self.mouth_edge_dist_min is not None and float(self.mouth_edge_dist_min) <= 0:
                raise ValueError("mouth_edge_dist_min must be > 0")
            if self.mouth_edge_dist_max is not None and float(self.mouth_edge_dist_max) <= 0:
                raise ValueError("mouth_edge_dist_max must be > 0")
            dmin = float(self.mouth_edge_dist_min) if self.mouth_edge_dist_min is not None else 0.1 * float(self.mouth_diameter)
            dmax = float(self.mouth_edge_dist_max) if self.mouth_edge_dist_max is not None else 0.3 * float(self.mouth_diameter)
            if dmin >= dmax:
                raise ValueError("mouth_edge_dist_min must be < mouth_edge_dist_max")
            if int(self.mouth_edge_sampling) <= 0:
                raise ValueError("mouth_edge_sampling must be > 0")
        if bool(getattr(self, "throat_edge_refine", False)):
            if self.throat_edge_size is not None and float(self.throat_edge_size) <= 0:
                raise ValueError("throat_edge_size must be > 0")
            if self.throat_edge_dist_min is not None and float(self.throat_edge_dist_min) <= 0:
                raise ValueError("throat_edge_dist_min must be > 0")
            if self.throat_edge_dist_max is not None and float(self.throat_edge_dist_max) <= 0:
                raise ValueError("throat_edge_dist_max must be > 0")
            dmin = (
                float(self.throat_edge_dist_min)
                if self.throat_edge_dist_min is not None
                else 0.05 * float(self.throat_diameter)
            )
            dmax = (
                float(self.throat_edge_dist_max)
                if self.throat_edge_dist_max is not None
                else 0.20 * float(self.throat_diameter)
            )
            if dmin >= dmax:
                raise ValueError("throat_edge_dist_min must be < throat_edge_dist_max")
            if int(self.throat_edge_sampling) <= 0:
                raise ValueError("throat_edge_sampling must be > 0")
        if self.throat_circle_points is not None:
            n = int(self.throat_circle_points)
            if n < 8:
                raise ValueError("throat_circle_points must be >= 8")
        if self.mesh_optimize is not None and not str(self.mesh_optimize).strip():
            raise ValueError("mesh_optimize must be a non-empty string or None")
        if self.morph is not None:
            self.morph.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of this config."""

        def jsonify(value: Any) -> Any:
            if isinstance(value, np.generic):
                return value.item()
            # Enums first: IntEnum is also an int.
            if isinstance(value, Enum):
                return str(value.name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [jsonify(v) for v in value]
            if isinstance(value, dict):
                return {str(k): jsonify(v) for k, v in value.items()}
            if is_dataclass(value):
                return jsonify(asdict(value))
            return value

        return jsonify(asdict(self))

    def to_json(self, filepath: str | Path) -> None:
        """Write this config to a JSON file."""
        path = Path(filepath)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WaveguideMeshConfig":
        """Reconstruct a config from `to_dict()` output."""
        if "morph" in data and data["morph"] is not None:
            morph_data = dict(data["morph"])
            ts = morph_data.get("target_shape", MorphTargetShape.KEEP)
            if isinstance(ts, str):
                morph_data["target_shape"] = MorphTargetShape[ts]
            else:
                morph_data["target_shape"] = MorphTargetShape(int(ts))
            if "superformula" in morph_data and morph_data["superformula"] is not None:
                morph_data["superformula"] = SuperFormulaConfig(**dict(morph_data["superformula"]))
            data = {**data, "morph": MorphConfig(**morph_data)}
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: str | Path) -> "WaveguideMeshConfig":
        path = Path(filepath)
        return cls.from_dict(json.loads(path.read_text()))


class WaveguideMeshGenerator:
    """
    Generate high-quality waveguide meshes using Gmsh OCC kernel.
    
    Features
    --------
    - Adaptive mesh sizing (finer at throat, coarser at mouth)
    - Multiple profile types (conical, exponential, tractrix, hyperbolic)
    - Sealed throat cap with proper vertex matching
    - Physical groups for multi-domain BEM (throat = domain 1, walls = domain 2)
    - Control over circumferential and axial resolution
    
    Examples
    --------
    >>> config = WaveguideMeshConfig(
    ...     throat_diameter=0.025,  # 25 mm
    ...     mouth_diameter=0.150,   # 150 mm
    ...     length=0.100,           # 100 mm
    ...     profile_type='exponential',
    ...     h_throat=0.005,         # 5 mm at throat
    ...     h_mouth=0.010,          # 10 mm at mouth
    ...     n_axial_slices=40,
    ...     n_circumferential=36
    ... )
    >>> generator = WaveguideMeshGenerator(config)
    >>> mesh, throat_mask = generator.generate()
    """
    
    def __init__(self, config: WaveguideMeshConfig):
        """
        Initialize waveguide mesh generator.
        
        Parameters
        ----------
        config : WaveguideMeshConfig
            Mesh generation configuration.
        """
        if not GMSH_AVAILABLE:
            raise ImportError("gmsh required for waveguide mesh generation")
        if not BEMPP_AVAILABLE:
            raise ImportError("bempp_cl required for waveguide mesh generation")
        
        config.validate()
        self.config = config
        
        # Derived parameters
        self.throat_r = config.throat_diameter / 2
        self.mouth_r = config.mouth_diameter / 2
    
    def generate(self) -> Tuple[LoudspeakerMesh, np.ndarray]:
        """
        Generate waveguide mesh with adaptive sizing.
        
        Returns
        -------
        mesh : LoudspeakerMesh
            Waveguide mesh with physical groups:
            - Domain 1: Throat cap (driver surface, vibrating)
            - Domain 2: Waveguide walls (rigid)
        throat_mask : np.ndarray
            Boolean array where True indicates throat cap elements.
        """
        with gmsh_session("waveguide", terminal=0):
            # Compute profile arrays (throat -> mouth)
            x, r, h = self._compute_profile_arrays()

            if self._use_lofted_morph():
                throat_disk_tag, wall_surface_tags = self._create_lofted_geometry(x, r, h)
            else:
                # Generate axisymmetric profile curve
                profile_points = self._create_profile_curve(x, r, h)
                # Create waveguide wall by revolution
                wall_surface_tags = self._create_revolved_wall_surface(profile_points)
                # Create throat cap from the wall boundary so the mesh is conformal at the throat.
                throat_disk_tag = self._create_throat_cap_from_wall(wall_surface_tags)

            # Synchronize OCC kernel
            gmsh.model.occ.synchronize()

            # Apply adaptive mesh sizing
            self._apply_mesh_sizing()

            # Optional: lock throat boundary discretization (stable node count).
            # This is done after OCC sync and before meshing so it affects the 1D mesh.
            self._apply_throat_boundary_lock(throat_disk_tag)

            # Define physical groups (domains for BEM)
            gmsh.model.addPhysicalGroup(2, [throat_disk_tag], int(Domain.THROAT), "throat")
            gmsh.model.addPhysicalGroup(2, wall_surface_tags, int(Domain.WALLS), "walls")

            # Generate mesh
            gmsh.model.mesh.generate(2)

            # Optional optimization (can move nodes; disabled when throat is locked).
            if self.config.mesh_optimize is not None and not bool(self.config.lock_throat_boundary):
                gmsh.model.mesh.optimize(str(self.config.mesh_optimize))

            # Export and import into bempp
            mesh, throat_mask = self._export_to_bempp()

            return mesh, throat_mask

    def _use_lofted_morph(self) -> bool:
        morph = self.config.morph
        return morph is not None and morph.target_shape != MorphTargetShape.KEEP

    def _compute_profile_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute x/r/h arrays along the axis (pure numeric)."""
        cfg = self.config

        x = np.linspace(0, cfg.length, cfg.n_axial_slices + 1)

        if cfg.profile_type == 'conical':
            r = conical_profile(x, self.throat_r, self.mouth_r, cfg.length)
        elif cfg.profile_type == 'exponential':
            r = exponential_profile(
                x,
                self.throat_r,
                self.mouth_r,
                cfg.length,
                flare_constant=cfg.flare_constant if cfg.flare_constant > 0 else None,
            )
        elif cfg.profile_type == 'tractrix':
            r = tractrix_profile(x, self.throat_r, self.mouth_r, cfg.length)
        elif cfg.profile_type == 'tractrix_horn':
            r = tractrix_horn_profile(x, self.throat_r, self.mouth_r, cfg.length)
        elif cfg.profile_type in ("os", "oblate_spheroidal"):
            r = os_profile(
                x,
                self.throat_r,
                self.mouth_r,
                cfg.length,
                opening_angle_deg=cfg.os_opening_angle_deg,
            )
        elif cfg.profile_type == 'hyperbolic':
            r = hyperbolic_profile(x, self.throat_r, self.mouth_r, cfg.length, cfg.hyperbolic_sharpness)
        elif cfg.profile_type == "cts":
            r = cts_profile(
                x,
                self.throat_r,
                self.mouth_r,
                cfg.length,
                throat_blend=cfg.cts_throat_blend,
                transition=cfg.cts_transition,
                driver_exit_angle_deg=cfg.cts_driver_exit_angle_deg,
                throat_angle_deg=cfg.cts_throat_angle_deg,
                tangency=cfg.cts_tangency,
                mouth_roll=cfg.cts_mouth_roll,
                curvature_regularizer=cfg.cts_curvature_regularizer,
                mid_curvature=cfg.cts_mid_curvature,
            )
        else:
            raise ValueError(f"Unknown profile type: {cfg.profile_type}")

        if cfg.grading_type == 'linear':
            h = linear_grading(x, cfg.h_throat, cfg.h_mouth, cfg.length)
        elif cfg.grading_type == 'exponential':
            h = exponential_grading(x, cfg.h_throat, cfg.h_mouth, cfg.length)
        elif cfg.grading_type == 'sigmoid':
            h = sigmoid_grading(x, cfg.h_throat, cfg.h_mouth, cfg.length)
        else:
            h = np.full_like(x, cfg.h_throat)

        return x, r, h

    def _create_profile_curve(self, x: np.ndarray, r: np.ndarray, h: np.ndarray) -> list:
        """
        Create the waveguide profile curve (meridian line to be revolved).
        
        Returns
        -------
        list
            Gmsh point tags defining the profile.
        """
        cfg = self.config
        
        # Create Gmsh points along profile
        # Z coordinate: throat at -length, mouth at 0
        # So z = -length + x (where x goes from 0 to length)
        # Or equivalently: z = -(length - x)
        point_tags = []
        for i, (xi, ri, hi) in enumerate(zip(x, r, h)):
            # Profile is in (r, 0, z) plane (will be revolved around z-axis)
            # Throat (x=0) should be at z=-length, mouth (x=length) at z=0
            z = -cfg.length + xi
            pt = gmsh.model.occ.addPoint(ri, 0, z, hi)
            point_tags.append(pt)
        if cfg.debug:
            logger = get_logger()
            logger.debug("Waveguide profile debug:")
            logger.debug(f"  Points: {len(point_tags)}")
            logger.debug(f"  Orientation: mouth at z=0, throat at z={-cfg.length:.6g} m")
            logger.debug(f"  Throat radius: {r[0]*1000:.2f} mm, mouth radius: {r[-1]*1000:.2f} mm")
        
        return point_tags
    
    def _create_throat_cap(self) -> int:
        """
        Create sealed throat cap (circular disk at z=-length).
        
        This is the vibrating driver surface at the back of the waveguide.
        The cap must be properly sealed (vertex-matched) to the waveguide wall profile.
        
        Returns
        -------
        int
            Gmsh surface tag for throat disk.
        """
        cfg = self.config
        # Create disk at z=-length (throat is inside the box)
        throat_disk = gmsh.model.occ.addDisk(0, 0, -cfg.length, self.throat_r, self.throat_r)
        
        return throat_disk

    def _create_throat_cap_from_wall(self, wall_surface_tags: list[int]) -> int:
        """
        Create a planar throat cap surface that shares the throat boundary curve with the wall.

        Using `addDisk(...)` creates a second, unshared throat circle and produces a non-conformal
        seam in the resulting surface mesh. This helper instead builds the cap from the revolved
        wall's own throat boundary curve.
        """
        cfg = self.config
        if not wall_surface_tags:
            raise ValueError("wall_surface_tags must be non-empty")

        gmsh.model.occ.synchronize()
        boundary = gmsh.model.getBoundary([(2, int(s)) for s in wall_surface_tags], oriented=False, recursive=False)
        curves = [int(tag) for dim, tag in boundary if int(dim) == 1]
        if not curves:
            raise RuntimeError("Could not find boundary curves for waveguide wall surfaces")

        throat_z = -float(cfg.length)
        tol = max(1e-9, float(cfg.h_throat) * 0.5)

        candidates: list[tuple[float, int]] = []
        for tag in curves:
            bbox = gmsh.model.getBoundingBox(1, tag)
            z_min, z_max = float(bbox[2]), float(bbox[5])
            if abs(z_min - throat_z) > tol or abs(z_max - throat_z) > tol:
                continue
            x_min, y_min = float(bbox[0]), float(bbox[1])
            x_max, y_max = float(bbox[3]), float(bbox[4])
            rx = 0.5 * (x_max - x_min)
            ry = 0.5 * (y_max - y_min)
            avg_r = 0.5 * (rx + ry)
            candidates.append((abs(avg_r - float(self.throat_r)), int(tag)))

        if not candidates:
            raise RuntimeError("Could not identify throat boundary curve on the revolved wall surface")

        _, throat_curve_tag = min(candidates, key=lambda p: p[0])
        throat_loop = gmsh.model.occ.addCurveLoop([int(throat_curve_tag)])
        return gmsh.model.occ.addPlaneSurface([int(throat_loop)])
    
    def _create_revolved_wall_surface(self, profile_points: list) -> list:
        """
        Create waveguide wall surface by revolving profile curve.
        
        Parameters
        ----------
        profile_points : list
            Gmsh point tags defining the profile curve.
        
        Returns
        -------
        list
            Gmsh surface tags for wall surfaces.
        """
        cfg = self.config
        
        if str(cfg.meridian_curve) == "polyline":
            segs = [gmsh.model.occ.addLine(int(a), int(b)) for a, b in zip(profile_points, profile_points[1:])]
            revolved = gmsh.model.occ.revolve(
                [(1, int(s)) for s in segs],
                0, 0, 0,
                0, 0, 1,
                2 * np.pi,
            )
        else:
            # Create spline through profile points
            profile_curve = gmsh.model.occ.addSpline(profile_points)

            # Revolve curve around z-axis (full 2π rotation)
            revolved = gmsh.model.occ.revolve(
                [(1, profile_curve)],  # 1D curve entity
                0, 0, 0,               # Point on rotation axis
                0, 0, 1,               # Axis direction (z-axis)
                2 * np.pi              # Full revolution
            )
        
        # Extract surface tags from revolution result
        surface_tags = [tag for dim, tag in revolved if dim == 2]
        
        return surface_tags

    def _create_lofted_geometry(
        self, x: np.ndarray, r: np.ndarray, h: np.ndarray
    ) -> Tuple[int, list]:
        """Create a non-axisymmetric waveguide wall via lofted morphed cross-sections.

        Returns
        -------
        throat_cap_tag : int
            Gmsh surface tag for the throat cap at z=-length.
        wall_surface_tags : list[int]
            Surface tags for the waveguide walls.
        """
        cfg = self.config
        morph = cfg.morph
        if morph is None or morph.target_shape == MorphTargetShape.KEEP:
            raise ValueError("lofted geometry requested but morph is not enabled")

        n_circ = int(cfg.n_circumferential)
        theta = theta_for_morph(
            n_total=n_circ,
            mouth_radius=float(self.mouth_r),
            morph=morph,
            corner_resolution=int(getattr(cfg, "corner_resolution", 0)),
        )

        xs_all, ys_all = morphed_sections_xy(
            x=x,
            r=r,
            length=float(cfg.length),
            theta=theta,
            throat_radius=float(self.throat_r),
            mouth_radius=float(self.mouth_r),
            morph=morph,
            enforce_n_directions_default=n_circ,
            name_prefix="morph cross-section",
        )

        wire_tags = []
        wire_curve_tags: list[list[int]] = []
        for (xi, hi), xs, ys in zip(zip(x, h), xs_all, ys_all):
            z = -cfg.length + float(xi)

            pt_tags = [
                gmsh.model.occ.addPoint(float(px), float(py), float(z), float(hi))
                for px, py in zip(xs, ys)
            ]

            curves = []
            for a, b in zip(pt_tags, pt_tags[1:] + [pt_tags[0]]):
                curves.append(gmsh.model.occ.addLine(a, b))
            wire = gmsh.model.occ.addWire(curves)
            # Keep both (wire for loft, curves for throat cap creation).
            wire_tags.append(wire)
            wire_curve_tags.append(curves)

        # Throat cap: planar surface from the first wire loop (vertex-matched)
        throat_loop = gmsh.model.occ.addCurveLoop(wire_curve_tags[0])
        throat_cap_tag = gmsh.model.occ.addPlaneSurface([throat_loop])

        # Loft through all section wires (open surface, no volume)
        # Use ruled sections to avoid spline overshoot between cross-sections.
        lofted = gmsh.model.occ.addThruSections(wire_tags, makeSolid=False, makeRuled=True)
        wall_surface_tags = [tag for dim, tag in lofted if dim == 2]

        if cfg.debug:
            logger = get_logger()
            logger.debug(
                f"Lofted morph waveguide: sections={len(wire_tags)}, "
                f"circum={int(cfg.n_circumferential)}, wall_surfaces={len(wall_surface_tags)}"
            )

        return throat_cap_tag, wall_surface_tags
    
    def _apply_mesh_sizing(self):
        """
        Apply adaptive mesh sizing fields using Gmsh distance-based sizing.
        
        Uses Gmsh MathEval and Distance fields for smooth grading.
        """
        cfg = self.config
        
        # Get all points
        all_points = gmsh.model.getEntities(0)
        
        # Identify throat/mouth points.
        # Coordinate system used by this generator:
        # - throat at z=-length
        # - mouth at z=0
        throat_z = -cfg.length
        mouth_z = 0.0
        throat_pts = []
        mouth_pts = []
        
        for dim, tag in all_points:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            z_coord = bbox[2]  # z-min (same as z-max for points)
            
            if abs(z_coord - throat_z) < cfg.h_throat:
                throat_pts.append(tag)
            elif abs(z_coord - mouth_z) < cfg.h_mouth:
                mouth_pts.append(tag)
        
        # Set characteristic length at specific points
        if throat_pts:
            h_cap = float(cfg.h_throat) * float(cfg.throat_cap_refinement)
            gmsh.model.mesh.setSize([(0, pt) for pt in throat_pts], h_cap)
        
        if mouth_pts:
            gmsh.model.mesh.setSize([(0, pt) for pt in mouth_pts], cfg.h_mouth)
        
        # Global mesh algorithm
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cfg.h_throat * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cfg.h_mouth * 1.5)
        # Make mesh generation deterministic when requested.
        if bool(getattr(cfg, "lock_throat_boundary", False)):
            # RandomFactor introduces tiny perturbations to avoid degeneracies;
            # turning it off helps reproducibility for coupling workflows.
            gmsh.option.setNumber("Mesh.RandomFactor", 0.0)

        # Always set a background field for z-grading so the throat→mouth taper is honored
        # even when no edge-refinement options are enabled.
        base_field: Optional[int] = None
        try:
            L = float(cfg.length)
            ht = float(cfg.h_throat)
            hm = float(cfg.h_mouth)
            ratio = hm / ht if ht > 0 else 1.0

            z_field = gmsh.model.mesh.field.add("MathEval")
            if ratio <= 0 or abs(ratio - 1.0) < 1e-12 or L <= 0:
                formula = f"{ht}"
            else:
                # z in [-L, 0] -> (z + L)/L in [0, 1]
                formula = f"{ht} * Exp(Log({ratio}) * ((z + {L}) / {L}))"
            gmsh.model.mesh.field.setString(z_field, "F", str(formula))
            base_field = int(z_field)
        except Exception:
            base_field = None

        # Optional refinement near the mouth rim curves (z=0) and/or throat rim curves (z=-length).
        # These are where edge diffraction and BC/application quality are typically most sensitive.
        try:
            all_curves = gmsh.model.getEntities(dim=1)
            mouth_curves: list[int] = []
            throat_curves: list[int] = []

            mouth_tol_z = max(1e-9, float(cfg.h_mouth) * 0.5)
            throat_tol_z = max(1e-9, float(cfg.h_throat) * 0.5)
            throat_z = -float(cfg.length)

            for _dim, tag in all_curves:
                bbox = gmsh.model.getBoundingBox(1, int(tag))
                z_min, z_max = float(bbox[2]), float(bbox[5])
                # Filter: curve should be near the aperture/throat rim (avoid selecting the rotation axis).
                x_min, y_min = float(bbox[0]), float(bbox[1])
                x_max, y_max = float(bbox[3]), float(bbox[4])
                rx = 0.5 * (x_max - x_min)
                ry = 0.5 * (y_max - y_min)
                max_r = max(rx, ry)

                if bool(getattr(cfg, "mouth_edge_refine", False)):
                    if abs(z_min - 0.0) <= mouth_tol_z and abs(z_max - 0.0) <= mouth_tol_z:
                        if max_r >= 0.25 * float(self.mouth_r):
                            mouth_curves.append(int(tag))

                if bool(getattr(cfg, "throat_edge_refine", False)):
                    if abs(z_min - throat_z) <= throat_tol_z and abs(z_max - throat_z) <= throat_tol_z:
                        if max_r >= 0.25 * float(self.throat_r):
                            throat_curves.append(int(tag))

            fields: list[int] = [int(base_field)] if base_field is not None else []

            if mouth_curves:
                h_edge = float(cfg.mouth_edge_size) if cfg.mouth_edge_size is not None else (0.8 * float(cfg.h_mouth))
                dmin = float(cfg.mouth_edge_dist_min) if cfg.mouth_edge_dist_min is not None else (0.1 * float(cfg.mouth_diameter))
                dmax = float(cfg.mouth_edge_dist_max) if cfg.mouth_edge_dist_max is not None else (0.3 * float(cfg.mouth_diameter))

                dist_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", mouth_curves)
                gmsh.model.mesh.field.setNumber(dist_field, "Sampling", int(cfg.mouth_edge_sampling))

                thr = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(thr, "InField", dist_field)
                gmsh.model.mesh.field.setNumber(thr, "SizeMin", float(h_edge))
                gmsh.model.mesh.field.setNumber(thr, "SizeMax", float(cfg.h_mouth))
                gmsh.model.mesh.field.setNumber(thr, "DistMin", float(dmin))
                gmsh.model.mesh.field.setNumber(thr, "DistMax", float(dmax))
                fields.append(int(thr))

            if throat_curves:
                h_edge = (
                    float(cfg.throat_edge_size)
                    if getattr(cfg, "throat_edge_size", None) is not None
                    else (0.6 * float(cfg.h_throat))
                )
                dmin = (
                    float(cfg.throat_edge_dist_min)
                    if getattr(cfg, "throat_edge_dist_min", None) is not None
                    else (0.05 * float(cfg.throat_diameter))
                )
                dmax = (
                    float(cfg.throat_edge_dist_max)
                    if getattr(cfg, "throat_edge_dist_max", None) is not None
                    else (0.20 * float(cfg.throat_diameter))
                )

                dist_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", throat_curves)
                gmsh.model.mesh.field.setNumber(dist_field, "Sampling", int(getattr(cfg, "throat_edge_sampling", 80)))

                thr = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(thr, "InField", dist_field)
                gmsh.model.mesh.field.setNumber(thr, "SizeMin", float(h_edge))
                gmsh.model.mesh.field.setNumber(thr, "SizeMax", float(cfg.h_throat))
                gmsh.model.mesh.field.setNumber(thr, "DistMin", float(dmin))
                gmsh.model.mesh.field.setNumber(thr, "DistMax", float(dmax))
                fields.append(int(thr))

            if len(fields) == 1:
                gmsh.model.mesh.field.setAsBackgroundMesh(int(fields[0]))
            else:
                fmin = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", [int(x) for x in fields])
                gmsh.model.mesh.field.setAsBackgroundMesh(int(fmin))
        except Exception:
            # Best-effort: if Gmsh field setup fails, fall back to point sizing only.
            return

    def _apply_throat_boundary_lock(self, throat_cap_tag: int) -> None:
        cfg = self.config
        if not bool(getattr(cfg, "lock_throat_boundary", False)):
            return

        try:
            boundary = gmsh.model.getBoundary([(2, int(throat_cap_tag))], oriented=False, recursive=False)
            rim_curves = [int(tag) for dim, tag in boundary if int(dim) == 1]
        except Exception:
            rim_curves = []

        if not rim_curves:
            logger = get_logger()
            logger.warning("Throat boundary lock requested, but no throat rim curves were found")
            return

        # Axisymmetric revolve typically yields a single circular curve. Lofted morph
        # yields a polyline loop with many line segments (already vertex-defined).
        if len(rim_curves) == 1:
            n = int(cfg.throat_circle_points) if cfg.throat_circle_points is not None else int(cfg.n_circumferential)
            n = max(8, n)
            gmsh.model.mesh.setTransfiniteCurve(int(rim_curves[0]), int(n))
            return

        # Polyline case: force exactly one segment per line (endpoints only),
        # keeping the rim vertex set stable.
        for c in rim_curves:
            gmsh.model.mesh.setTransfiniteCurve(int(c), 2)
    
    def _export_to_bempp(self) -> Tuple[LoudspeakerMesh, np.ndarray]:
        """
        Export Gmsh mesh and import into Bempp.
        
        Returns
        -------
        mesh : LoudspeakerMesh
            Bempp-wrapped mesh.
        throat_mask : np.ndarray
            Boolean mask for throat elements.
        """
        msh_name = write_temp_msh()
        gmsh.write(msh_name)
        grid = import_msh_and_cleanup(msh_name)
        
        # Create LoudspeakerMesh wrapper
        mesh = LoudspeakerMesh(grid)
        mesh.center = np.array([0, 0, 0])
        mesh.axis = np.array([0, 0, 1])
        
        # Create throat mask.
        # By construction in `generate()`, physical group 1 is throat and 2 is walls.
        domain_indices = grid.domain_indices
        unique_domains = np.unique(domain_indices)
        if 1 in unique_domains:
            throat_mask = domain_indices == 1
        else:
            # Fallback: infer throat as the smallest domain (legacy behavior)
            logger = get_logger()
            logger.warning(
                "Waveguide mesh import did not preserve physical group id 1 for throat; "
                "falling back to heuristic throat-domain detection."
            )
            if len(unique_domains) > 1:
                domain_counts = {d: int(np.sum(domain_indices == d)) for d in unique_domains}
                throat_domain = min(domain_counts, key=domain_counts.get)
                throat_mask = domain_indices == throat_domain
            else:
                throat_mask = domain_indices == unique_domains[0]
        
        return mesh, throat_mask


# =============================================================================
# Convenience function
# =============================================================================

def create_waveguide_mesh(
    throat_diameter: float,
    mouth_diameter: float,
    length: float,
    profile_type: str = 'exponential',
    h_throat: float = 0.005,
    h_mouth: float = 0.010,
    # Default 95 slices -> 96 axial points.
    n_axial_slices: int = 95,
    n_circumferential: int = 36,
    **kwargs
) -> Tuple[LoudspeakerMesh, np.ndarray]:
    """
    Convenience function to create a waveguide mesh.
    
    Parameters
    ----------
    throat_diameter : float
        Throat diameter in meters (e.g., 0.025 for 25mm).
    mouth_diameter : float
        Mouth diameter in meters (e.g., 0.150 for 150mm).
    length : float
        Waveguide length in meters (e.g., 0.100 for 100mm).
    profile_type : str
        Profile type: 'conical', 'exponential', 'tractrix', 'hyperbolic'.
    h_throat : float
        Element size at throat in meters (default 5mm).
    h_mouth : float
        Element size at mouth in meters (default 10mm).
    n_axial_slices : int
        Number of profile slices along axis (default 95 -> 96 axial points).
    n_circumferential : int
        Divisions around circumference (default 36 = 10° resolution).
    **kwargs
        Additional WaveguideMeshConfig parameters.
    
    Returns
    -------
    mesh : LoudspeakerMesh
        Waveguide mesh with domains (1=throat, 2=walls).
    throat_mask : np.ndarray
        Boolean mask for throat cap elements.
    
    Examples
    --------
    >>> # Exponential waveguide with adaptive sizing
    >>> mesh, throat_mask = create_waveguide_mesh(
    ...     throat_diameter=0.025,
    ...     mouth_diameter=0.150,
    ...     length=0.100,
    ...     profile_type='exponential',
    ...     h_throat=0.005,  # 5mm at throat
    ...     h_mouth=0.010,   # 10mm at mouth
    ...     n_axial_slices=40,
    ...     n_circumferential=36
    ... )
    >>> 
    >>> # Tractrix (constant directivity) waveguide
    >>> mesh, mask = create_waveguide_mesh(
    ...     throat_diameter=0.025,
    ...     mouth_diameter=0.200,
    ...     length=0.150,
    ...     profile_type='tractrix',
    ...     h_throat=0.004,
    ...     h_mouth=0.012,
    ...     n_axial_slices=50
    ... )
    """
    config = WaveguideMeshConfig(
        throat_diameter=throat_diameter,
        mouth_diameter=mouth_diameter,
        length=length,
        profile_type=profile_type,
        h_throat=h_throat,
        h_mouth=h_mouth,
        n_axial_slices=n_axial_slices,
        n_circumferential=n_circumferential,
        **kwargs
    )
    
    generator = WaveguideMeshGenerator(config)
    return generator.generate()
