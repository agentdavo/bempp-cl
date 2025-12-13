"""
Cabinet geometry builder for BEM mesh generation.

Provides a fluent API for constructing box enclosures with:
- Per-edge chamfers (symmetric, asymmetric, or angled)
- Cutouts for waveguide/driver mounting
- Tapered mesh resolution (front to back)

Example usage:

    cabinet = (
        CabinetBuilder()
        .dimensions(width=0.52, height=0.44, depth=0.30)
        .chamfer_bottom(d1=0.020, d2=0.015)   # 20mm on baffle, 15mm on bottom
        .chamfer_top(d1=0.020)                 # 20mm symmetric
        .chamfer_left(d1=0.015, angle_deg=45)  # 15mm at 45 degrees
        .chamfer_right(d1=0.015, angle_deg=45)
        .resolution(baffle=0.010, sides=0.020, back=0.030)
        .build()
    )

Coordinate System:
- Front face (baffle) at z=0
- Box extends into -z (depth)
- Origin at lower-left corner of front face when viewed from front
- x: left to right (0 to width)
- y: bottom to top (0 to height)
- z: back to front (-depth to 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np


class ChamferType(Enum):
    """Chamfer specification type."""
    NONE = 0           # No chamfer
    SYMMETRIC = 1      # Single distance (45-degree)
    ASYMMETRIC = 2     # Two distances (d1 on baffle, d2 on side)
    ANGLED = 3         # Distance + angle


@dataclass(frozen=True)
class ChamferSpec:
    """Specification for a single edge chamfer.

    Attributes
    ----------
    chamfer_type : ChamferType
        Type of chamfer specification.
    d1 : float
        Primary distance in meters:
        - SYMMETRIC: chamfer size (both faces)
        - ASYMMETRIC: distance on the baffle face
        - ANGLED: distance on the baffle face
    d2 : float
        Secondary distance in meters (ASYMMETRIC only):
        - Distance on the adjacent side face
    angle_deg : float
        Chamfer angle in degrees (ANGLED only):
        - 45 = symmetric, <45 = steeper toward side, >45 = shallower
    """
    chamfer_type: ChamferType = ChamferType.NONE
    d1: float = 0.0
    d2: float = 0.0
    angle_deg: float = 45.0

    @classmethod
    def none(cls) -> "ChamferSpec":
        """No chamfer."""
        return cls(chamfer_type=ChamferType.NONE)

    @classmethod
    def symmetric(cls, distance: float) -> "ChamferSpec":
        """Symmetric chamfer (45-degree).

        Parameters
        ----------
        distance : float
            Chamfer size in meters (same on both faces).
        """
        if distance <= 0:
            raise ValueError("distance must be > 0")
        return cls(chamfer_type=ChamferType.SYMMETRIC, d1=float(distance), d2=float(distance))

    @classmethod
    def asymmetric(cls, d_baffle: float, d_side: float) -> "ChamferSpec":
        """Asymmetric chamfer with different distances on each face.

        Parameters
        ----------
        d_baffle : float
            Distance on the baffle (front) face in meters.
        d_side : float
            Distance on the adjacent side face in meters.
        """
        if d_baffle <= 0 or d_side <= 0:
            raise ValueError("distances must be > 0")
        return cls(chamfer_type=ChamferType.ASYMMETRIC, d1=float(d_baffle), d2=float(d_side))

    @classmethod
    def angled(cls, distance: float, angle_deg: float) -> "ChamferSpec":
        """Chamfer with specified angle.

        Parameters
        ----------
        distance : float
            Distance on the baffle face in meters.
        angle_deg : float
            Angle in degrees (45 = symmetric, <45 = steeper).
        """
        if distance <= 0:
            raise ValueError("distance must be > 0")
        if not (5.0 <= angle_deg <= 85.0):
            raise ValueError("angle_deg must be in [5, 85]")
        # Compute d2 from angle: d2 = d1 * tan(angle)
        d2 = distance * np.tan(np.radians(angle_deg))
        return cls(chamfer_type=ChamferType.ANGLED, d1=float(distance), d2=float(d2), angle_deg=float(angle_deg))

    def is_active(self) -> bool:
        """Return True if this chamfer should be applied."""
        return self.chamfer_type != ChamferType.NONE and self.d1 > 0

    def distances(self) -> Tuple[float, float]:
        """Return (d1, d2) distances for OCC chamfer operation."""
        if self.chamfer_type == ChamferType.SYMMETRIC:
            return (self.d1, self.d1)
        return (self.d1, self.d2)


@dataclass
class CabinetConfig:
    """Complete cabinet configuration.

    Attributes
    ----------
    width : float
        Cabinet width (x-direction) in meters.
    height : float
        Cabinet height (y-direction) in meters.
    depth : float
        Cabinet depth (z-direction) in meters.
    chamfer_top : ChamferSpec
        Chamfer on top edge of front face.
    chamfer_bottom : ChamferSpec
        Chamfer on bottom edge of front face.
    chamfer_left : ChamferSpec
        Chamfer on left edge of front face.
    chamfer_right : ChamferSpec
        Chamfer on right edge of front face.
    fillet_top_radius : float
        Fillet (roundover) radius on top edge of front face.
    fillet_bottom_radius : float
        Fillet (roundover) radius on bottom edge of front face.
    fillet_left_radius : float
        Fillet (roundover) radius on left edge of front face.
    fillet_right_radius : float
        Fillet (roundover) radius on right edge of front face.
    h_baffle : float
        Mesh element size on baffle near the opening.
    h_sides : float
        Mesh element size on side faces at z=0.
    h_back : float
        Mesh element size on back face.
    """
    width: float = 0.0
    height: float = 0.0
    depth: float = 0.0

    # Per-edge chamfers
    chamfer_top: ChamferSpec = field(default_factory=ChamferSpec.none)
    chamfer_bottom: ChamferSpec = field(default_factory=ChamferSpec.none)
    chamfer_left: ChamferSpec = field(default_factory=ChamferSpec.none)
    chamfer_right: ChamferSpec = field(default_factory=ChamferSpec.none)

    # Per-edge fillets (roundovers). Mutually exclusive with chamfers on the same edge.
    fillet_top_radius: float = 0.0
    fillet_bottom_radius: float = 0.0
    fillet_left_radius: float = 0.0
    fillet_right_radius: float = 0.0

    # Mesh resolution
    h_baffle: float = 0.010   # 10mm default
    h_sides: float = 0.020    # 20mm default
    h_back: float = 0.030     # 30mm default

    def validate(self) -> None:
        """Validate configuration."""
        if self.width <= 0 or self.height <= 0 or self.depth <= 0:
            raise ValueError("width, height, and depth must be > 0")

        # Mutual exclusivity (per edge)
        if self.chamfer_top.is_active() and self.fillet_top_radius > 0:
            raise ValueError("top edge: cannot specify both chamfer and fillet")
        if self.chamfer_bottom.is_active() and self.fillet_bottom_radius > 0:
            raise ValueError("bottom edge: cannot specify both chamfer and fillet")
        if self.chamfer_left.is_active() and self.fillet_left_radius > 0:
            raise ValueError("left edge: cannot specify both chamfer and fillet")
        if self.chamfer_right.is_active() and self.fillet_right_radius > 0:
            raise ValueError("right edge: cannot specify both chamfer and fillet")

        # Check chamfers don't exceed face dimensions
        if self.chamfer_top.is_active() or self.chamfer_bottom.is_active():
            max_d = min(self.height / 4, self.depth / 4)
            for name, spec in [("top", self.chamfer_top), ("bottom", self.chamfer_bottom)]:
                if spec.is_active():
                    d1, d2 = spec.distances()
                    if d1 > max_d or d2 > max_d:
                        raise ValueError(f"{name} chamfer too large (max {max_d*1e3:.1f}mm)")

        if self.chamfer_left.is_active() or self.chamfer_right.is_active():
            max_d = min(self.width / 4, self.depth / 4)
            for name, spec in [("left", self.chamfer_left), ("right", self.chamfer_right)]:
                if spec.is_active():
                    d1, d2 = spec.distances()
                    if d1 > max_d or d2 > max_d:
                        raise ValueError(f"{name} chamfer too large (max {max_d*1e3:.1f}mm)")

        # Fillet bounds (simple conservative guards)
        if any(r < 0 for r in [self.fillet_top_radius, self.fillet_bottom_radius, self.fillet_left_radius, self.fillet_right_radius]):
            raise ValueError("fillet radii must be >= 0")
        if self.fillet_top_radius > 0 or self.fillet_bottom_radius > 0:
            max_r = min(self.height / 4, self.depth / 4)
            for name, r in [("top", self.fillet_top_radius), ("bottom", self.fillet_bottom_radius)]:
                if r > max_r:
                    raise ValueError(f"{name} fillet too large (max {max_r*1e3:.1f}mm)")
        if self.fillet_left_radius > 0 or self.fillet_right_radius > 0:
            max_r = min(self.width / 4, self.depth / 4)
            for name, r in [("left", self.fillet_left_radius), ("right", self.fillet_right_radius)]:
                if r > max_r:
                    raise ValueError(f"{name} fillet too large (max {max_r*1e3:.1f}mm)")

        if self.h_baffle <= 0 or self.h_sides <= 0 or self.h_back <= 0:
            raise ValueError("mesh sizes must be > 0")

    def has_chamfers(self) -> bool:
        """Return True if any chamfers are specified."""
        return any(spec.is_active() for spec in [
            self.chamfer_top, self.chamfer_bottom,
            self.chamfer_left, self.chamfer_right
        ])

    def has_fillets(self) -> bool:
        """Return True if any fillets are specified."""
        return any(
            r > 0 for r in [self.fillet_top_radius, self.fillet_bottom_radius, self.fillet_left_radius, self.fillet_right_radius]
        )

    def has_edge_blends(self) -> bool:
        """Return True if any chamfer or fillet is specified."""
        return self.has_chamfers() or self.has_fillets()


class CabinetBuilder:
    """Fluent builder for cabinet geometry configuration.

    Example
    -------
    >>> config = (
    ...     CabinetBuilder()
    ...     .dimensions(width=0.52, height=0.44, depth=0.30)
    ...     .chamfer_bottom(d1=0.020)
    ...     .chamfer_top(d1=0.020)
    ...     .resolution(baffle=0.010, sides=0.020, back=0.030)
    ...     .build()
    ... )
    """

    def __init__(self) -> None:
        self._config = CabinetConfig()

    def dimensions(
        self,
        width: float,
        height: float,
        depth: float,
    ) -> "CabinetBuilder":
        """Set cabinet dimensions.

        Parameters
        ----------
        width : float
            Cabinet width (x-direction) in meters.
        height : float
            Cabinet height (y-direction) in meters.
        depth : float
            Cabinet depth (z-direction) in meters.
        """
        self._config.width = float(width)
        self._config.height = float(height)
        self._config.depth = float(depth)
        return self

    def dimensions_mm(
        self,
        width_mm: float,
        height_mm: float,
        depth_mm: float,
    ) -> "CabinetBuilder":
        """Set cabinet dimensions in millimeters."""
        return self.dimensions(
            width=width_mm * 1e-3,
            height=height_mm * 1e-3,
            depth=depth_mm * 1e-3,
        )

    def chamfer_top(
        self,
        d1: Optional[float] = None,
        d2: Optional[float] = None,
        angle_deg: Optional[float] = None,
        d1_mm: Optional[float] = None,
        d2_mm: Optional[float] = None,
    ) -> "CabinetBuilder":
        """Set chamfer on top edge of front face.

        Parameters
        ----------
        d1 : float, optional
            Primary distance (on baffle) in meters.
        d2 : float, optional
            Secondary distance (on top face) in meters. If None, symmetric.
        angle_deg : float, optional
            Chamfer angle. Overrides d2 if specified.
        d1_mm, d2_mm : float, optional
            Distances in millimeters (alternative to d1, d2).
        """
        self._config.chamfer_top = self._make_chamfer_spec(d1, d2, angle_deg, d1_mm, d2_mm)
        return self

    def chamfer_bottom(
        self,
        d1: Optional[float] = None,
        d2: Optional[float] = None,
        angle_deg: Optional[float] = None,
        d1_mm: Optional[float] = None,
        d2_mm: Optional[float] = None,
    ) -> "CabinetBuilder":
        """Set chamfer on bottom edge of front face."""
        self._config.chamfer_bottom = self._make_chamfer_spec(d1, d2, angle_deg, d1_mm, d2_mm)
        return self

    def chamfer_left(
        self,
        d1: Optional[float] = None,
        d2: Optional[float] = None,
        angle_deg: Optional[float] = None,
        d1_mm: Optional[float] = None,
        d2_mm: Optional[float] = None,
    ) -> "CabinetBuilder":
        """Set chamfer on left edge of front face."""
        self._config.chamfer_left = self._make_chamfer_spec(d1, d2, angle_deg, d1_mm, d2_mm)
        return self

    def chamfer_right(
        self,
        d1: Optional[float] = None,
        d2: Optional[float] = None,
        angle_deg: Optional[float] = None,
        d1_mm: Optional[float] = None,
        d2_mm: Optional[float] = None,
    ) -> "CabinetBuilder":
        """Set chamfer on right edge of front face."""
        self._config.chamfer_right = self._make_chamfer_spec(d1, d2, angle_deg, d1_mm, d2_mm)
        return self

    def fillet_top(self, radius: float) -> "CabinetBuilder":
        """Set fillet (roundover) radius on top edge of front face."""
        self._config.fillet_top_radius = float(radius)
        return self

    def fillet_bottom(self, radius: float) -> "CabinetBuilder":
        """Set fillet (roundover) radius on bottom edge of front face."""
        self._config.fillet_bottom_radius = float(radius)
        return self

    def fillet_left(self, radius: float) -> "CabinetBuilder":
        """Set fillet (roundover) radius on left edge of front face."""
        self._config.fillet_left_radius = float(radius)
        return self

    def fillet_right(self, radius: float) -> "CabinetBuilder":
        """Set fillet (roundover) radius on right edge of front face."""
        self._config.fillet_right_radius = float(radius)
        return self

    def fillet_all(self, radius: float) -> "CabinetBuilder":
        """Set same fillet on all four front face edges."""
        r = float(radius)
        self._config.fillet_top_radius = r
        self._config.fillet_bottom_radius = r
        self._config.fillet_left_radius = r
        self._config.fillet_right_radius = r
        return self

    def chamfer_all(
        self,
        d1: Optional[float] = None,
        d2: Optional[float] = None,
        angle_deg: Optional[float] = None,
        d1_mm: Optional[float] = None,
        d2_mm: Optional[float] = None,
    ) -> "CabinetBuilder":
        """Set same chamfer on all four front face edges."""
        spec = self._make_chamfer_spec(d1, d2, angle_deg, d1_mm, d2_mm)
        self._config.chamfer_top = spec
        self._config.chamfer_bottom = spec
        self._config.chamfer_left = spec
        self._config.chamfer_right = spec
        return self

    def chamfer_horizontal(
        self,
        d1: Optional[float] = None,
        d2: Optional[float] = None,
        angle_deg: Optional[float] = None,
        d1_mm: Optional[float] = None,
        d2_mm: Optional[float] = None,
    ) -> "CabinetBuilder":
        """Set chamfer on top and bottom edges."""
        spec = self._make_chamfer_spec(d1, d2, angle_deg, d1_mm, d2_mm)
        self._config.chamfer_top = spec
        self._config.chamfer_bottom = spec
        return self

    def chamfer_vertical(
        self,
        d1: Optional[float] = None,
        d2: Optional[float] = None,
        angle_deg: Optional[float] = None,
        d1_mm: Optional[float] = None,
        d2_mm: Optional[float] = None,
    ) -> "CabinetBuilder":
        """Set chamfer on left and right edges."""
        spec = self._make_chamfer_spec(d1, d2, angle_deg, d1_mm, d2_mm)
        self._config.chamfer_left = spec
        self._config.chamfer_right = spec
        return self

    def resolution(
        self,
        baffle: Optional[float] = None,
        sides: Optional[float] = None,
        back: Optional[float] = None,
    ) -> "CabinetBuilder":
        """Set mesh resolution for cabinet faces.

        Parameters
        ----------
        baffle : float, optional
            Element size on baffle (front face) near the opening in meters.
        sides : float, optional
            Element size on side faces at z=0 in meters.
        back : float, optional
            Element size on back face in meters.
        """
        if baffle is not None:
            self._config.h_baffle = float(baffle)
        if sides is not None:
            self._config.h_sides = float(sides)
        if back is not None:
            self._config.h_back = float(back)
        return self

    def resolution_mm(
        self,
        baffle_mm: Optional[float] = None,
        sides_mm: Optional[float] = None,
        back_mm: Optional[float] = None,
    ) -> "CabinetBuilder":
        """Set mesh resolution in millimeters."""
        return self.resolution(
            baffle=baffle_mm * 1e-3 if baffle_mm is not None else None,
            sides=sides_mm * 1e-3 if sides_mm is not None else None,
            back=back_mm * 1e-3 if back_mm is not None else None,
        )

    def build(self) -> CabinetConfig:
        """Build and validate the cabinet configuration."""
        self._config.validate()
        return self._config

    def _make_chamfer_spec(
        self,
        d1: Optional[float],
        d2: Optional[float],
        angle_deg: Optional[float],
        d1_mm: Optional[float],
        d2_mm: Optional[float],
    ) -> ChamferSpec:
        """Create a ChamferSpec from the given parameters."""
        # Handle mm units
        if d1_mm is not None:
            d1 = d1_mm * 1e-3
        if d2_mm is not None:
            d2 = d2_mm * 1e-3

        if d1 is None:
            return ChamferSpec.none()

        if angle_deg is not None:
            return ChamferSpec.angled(d1, angle_deg)
        elif d2 is not None:
            return ChamferSpec.asymmetric(d1, d2)
        else:
            return ChamferSpec.symmetric(d1)


@dataclass
class CabinetGeometry:
    """Result of cabinet geometry generation.

    Contains references to Gmsh entities for further operations
    (e.g., cutting waveguide mouth hole, assigning physical groups).
    """
    volume_tag: int                    # Solid box volume
    front_surface: int                 # Front face (baffle) surface
    back_surface: int                  # Back face surface
    side_surfaces: Dict[str, int]      # {'left', 'right', 'top', 'bottom'} -> surface tag
    chamfer_surfaces: Dict[str, int]   # Edge name -> chamfer surface tag (if chamfered)
    front_edges: Dict[str, int]        # {'top', 'bottom', 'left', 'right'} -> edge tag
    config: CabinetConfig              # Original configuration


def create_cabinet_geometry(
    config: CabinetConfig,
    gmsh_model: Optional[Any] = None,
) -> CabinetGeometry:
    """Create cabinet box geometry with chamfers in Gmsh.

    This function creates a solid box and applies chamfers to the specified
    front-face edges. The geometry is created in the current Gmsh model.

    Parameters
    ----------
    config : CabinetConfig
        Cabinet configuration from CabinetBuilder.
    gmsh_model : optional
        Gmsh model to use. If None, uses gmsh.model directly.

    Returns
    -------
    CabinetGeometry
        Container with Gmsh entity references.

    Notes
    -----
    Requires gmsh to be initialized and a model to be active.
    Call gmsh.model.occ.synchronize() after this function if needed.
    """
    from bempp_audio._optional import optional_import
    gmsh, GMSH_AVAILABLE = optional_import("gmsh")
    if not GMSH_AVAILABLE:
        raise ImportError("gmsh required for cabinet geometry generation")

    model = gmsh_model if gmsh_model is not None else gmsh.model
    occ = model.occ

    w = float(config.width)
    h = float(config.height)
    d = float(config.depth)

    config.validate()

    # Create solid box: origin at (0,0,-d), extends to (w,h,0)
    # Front face at z=0, back at z=-d
    box = occ.addBox(0, 0, -d, w, h, d)
    occ.synchronize()

    # Tolerance for geometry identification (1e-6 meters = 1 micron)
    tol = 1e-6

    # Identify surfaces and edges
    def _boundary_surfaces(volume_tag: int) -> list[int]:
        boundary = model.getBoundary([(3, int(volume_tag))], combined=False, oriented=False, recursive=False)
        return [abs(int(tag)) for dim, tag in boundary if int(dim) == 2]

    def _get_surfaces(volume_tag: int) -> Dict[str, int]:
        """Identify the volume's boundary surfaces by position."""
        surfaces: Dict[str, int] = {}
        for tag in _boundary_surfaces(volume_tag):
            bbox = model.getBoundingBox(2, tag)
            z_min, z_max = bbox[2], bbox[5]
            x_min, x_max = bbox[0], bbox[3]
            y_min, y_max = bbox[1], bbox[4]

            # Front face (baffle plane, z=0)
            # (Chamfer/fillet faces can have z_max≈0 but z_min<0; exclude those.)
            if abs(z_min) < tol and abs(z_max) < tol:
                surfaces['front'] = tag
            # Back face (z=-d)
            elif abs(z_min + d) < tol and abs(z_max + d) < tol:
                surfaces['back'] = tag
            # Left face (x=0)
            elif abs(x_min) < tol and abs(x_max) < tol:
                surfaces['left'] = tag
            # Right face (x=w)
            elif abs(x_min - w) < tol and abs(x_max - w) < tol:
                surfaces['right'] = tag
            # Bottom face (y=0)
            elif abs(y_min) < tol and abs(y_max) < tol:
                surfaces['bottom'] = tag
            # Top face (y=h)
            elif abs(y_min - h) < tol and abs(y_max - h) < tol:
                surfaces['top'] = tag
        return surfaces

    def _get_front_edges(front_surface_tag: int) -> Dict[str, int]:
        """Identify front face boundary curves by position (restricted to the front face)."""
        edges: Dict[str, int] = {}
        loop_tags, loop_curves = occ.getCurveLoops(int(front_surface_tag))
        if len(loop_tags) < 1:
            return edges
        curve_tags = {abs(int(c)) for arr in loop_curves for c in np.asarray(arr, dtype=int).tolist()}
        for tag in curve_tags:
            bbox = model.getBoundingBox(1, tag)
            z_min, z_max = bbox[2], bbox[5]
            # Only edges at z=0 (both endpoints)
            if abs(z_min) > tol or abs(z_max) > tol:
                continue

            x_min, x_max = bbox[0], bbox[3]
            y_min, y_max = bbox[1], bbox[4]

            # Bottom edge (y=0, x spans)
            if abs(y_min) < tol and abs(y_max) < tol:
                edges['bottom'] = tag
            # Top edge (y=h, x spans)
            elif abs(y_min - h) < tol and abs(y_max - h) < tol:
                edges['top'] = tag
            # Left edge (x=0, y spans)
            elif abs(x_min) < tol and abs(x_max) < tol:
                edges['left'] = tag
            # Right edge (x=w, y spans)
            elif abs(x_min - w) < tol and abs(x_max - w) < tol:
                edges['right'] = tag
        return edges

    surfaces = _get_surfaces(box)
    front_tag = surfaces.get("front")
    if front_tag is None:
        raise RuntimeError("Failed to identify cabinet front surface")
    front_edges = _get_front_edges(front_tag)

    chamfer_surfaces: Dict[str, int] = {}

    # Apply edge blends (one at a time; order matters)
    if config.has_edge_blends():
        edge_blends = [
            ("bottom", config.fillet_bottom_radius, config.chamfer_bottom, surfaces.get("bottom")),
            ("top", config.fillet_top_radius, config.chamfer_top, surfaces.get("top")),
            ("left", config.fillet_left_radius, config.chamfer_left, surfaces.get("left")),
            ("right", config.fillet_right_radius, config.chamfer_right, surfaces.get("right")),
        ]

        for edge_name, fillet_r, chamfer_spec, side_surface in edge_blends:
            if edge_name not in front_edges:
                continue
            edge_tag = int(front_edges[edge_name])

            # Fillet (roundover)
            if float(fillet_r) > 0:
                try:
                    result = occ.fillet(
                        volumeTags=[box],
                        curveTags=[edge_tag],
                        radii=[float(fillet_r)],
                        removeVolume=False,
                    )
                    occ.synchronize()
                    if result:
                        new_box = int(result[0][1])
                        occ.remove([(3, int(box))], recursive=True)
                        box = new_box
                        occ.synchronize()
                    surfaces = _get_surfaces(box)
                    front_tag = surfaces.get("front")
                    if front_tag is not None:
                        front_edges = _get_front_edges(front_tag)
                except Exception as e:
                    from bempp_audio.progress import get_logger

                    logger = get_logger()
                    logger.warning(f"Fillet on {edge_name} edge failed: {e}")
                continue

            # Chamfer
            if chamfer_spec.is_active():
                if side_surface is None:
                    continue
                front_tag = surfaces.get("front")
                if front_tag is None:
                    continue

                d1, d2 = chamfer_spec.distances()

                try:
                    before = set(_boundary_surfaces(box))
                    result = occ.chamfer(
                        volumeTags=[box],
                        curveTags=[edge_tag],
                        surfaceTags=[front_tag],
                        distances=[d1, d2],
                        removeVolume=False,
                    )
                    occ.synchronize()

                    if result:
                        new_box = int(result[0][1])
                        occ.remove([(3, int(box))], recursive=True)
                        box = new_box
                        occ.synchronize()

                    after = set(_boundary_surfaces(box))
                    surfaces = _get_surfaces(box)
                    # Best-effort: record any new boundary surface not matching the named faces.
                    for tag in sorted(after - before):
                        if tag not in surfaces.values():
                            chamfer_surfaces[edge_name] = int(tag)
                            break

                    front_tag = surfaces.get("front")
                    if front_tag is not None:
                        front_edges = _get_front_edges(front_tag)

                except Exception as e:
                    from bempp_audio.progress import get_logger

                    logger = get_logger()
                    logger.warning(f"Chamfer on {edge_name} edge failed: {e}")

    # Get final surface references
    surfaces = _get_surfaces(box)

    return CabinetGeometry(
        volume_tag=box,
        front_surface=surfaces.get('front', -1),
        back_surface=surfaces.get('back', -1),
        side_surfaces={
            'left': surfaces.get('left', -1),
            'right': surfaces.get('right', -1),
            'top': surfaces.get('top', -1),
            'bottom': surfaces.get('bottom', -1),
        },
        chamfer_surfaces=chamfer_surfaces,
        front_edges=front_edges,
        config=config,
    )
