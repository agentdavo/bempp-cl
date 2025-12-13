"""
Mesh generation for dome diaphragms using Gmsh + OpenCASCADE.

Creates triangular surface meshes suitable for shell FEM analysis
and BEM coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from bempp_audio.fea.dome_geometry import DomeGeometry
from bempp_audio.fea.materials import ShellMaterial, recommended_mesh_size
from bempp_audio.fea.mesh_quality import (
    MeshQualityReport,
    validate_mesh_quality,
    compute_aspect_ratios,
    compute_scaled_jacobians,
    compute_element_sizes,
    find_poor_quality_elements,
)

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    gmsh = None


@dataclass
class DomeMesh:
    """
    Triangular surface mesh of a dome.

    Coordinate convention:
    - Apex at z = 0 (center of dome, smallest radius)
    - Base at z = dome_height (largest radius)
    - Normals point outward (away from concave side, toward radiation direction)

    Coordinates are in meters. The mesh is suitable for shell FEM
    analysis and can be exported to bempp-cl for BEM coupling.
    """

    vertices: np.ndarray  # (N, 3) vertex coordinates
    triangles: np.ndarray  # (M, 3) triangle vertex indices (0-indexed)
    normals: Optional[np.ndarray] = None  # (M, 3) face normals (outward)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_triangles(self) -> int:
        return len(self.triangles)

    @property
    def apex_z(self) -> float:
        """Z-coordinate of apex (minimum radius point)."""
        r = np.sqrt(self.vertices[:, 0]**2 + self.vertices[:, 1]**2)
        return self.vertices[np.argmin(r), 2]

    @property
    def base_z(self) -> float:
        """Z-coordinate of base (maximum radius point)."""
        r = np.sqrt(self.vertices[:, 0]**2 + self.vertices[:, 1]**2)
        return self.vertices[np.argmax(r), 2]

    def compute_normals(self, outward_direction: str = "positive_z") -> None:
        """
        Compute face normals with consistent orientation.

        Parameters
        ----------
        outward_direction : str
            How to orient "outward" normals:
            - "positive_z": normals point in +z direction (toward radiation)
            - "negative_z": normals point in -z direction
            - "away_from_axis": normals point radially outward from z-axis
        """
        normals = np.zeros((self.num_triangles, 3))
        for i, tri in enumerate(self.triangles):
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm

            # Orient normal based on specified convention
            centroid = (v0 + v1 + v2) / 3
            if outward_direction == "positive_z":
                # For dome with apex at z=0, outward means toward +z
                if normal[2] < 0:
                    normal = -normal
            elif outward_direction == "negative_z":
                if normal[2] > 0:
                    normal = -normal
            elif outward_direction == "away_from_axis":
                # Radial direction in xy-plane
                r_vec = np.array([centroid[0], centroid[1], 0])
                r_norm = np.linalg.norm(r_vec)
                if r_norm > 1e-10:
                    r_vec /= r_norm
                    if np.dot(normal[:2], r_vec[:2]) < 0:
                        normal = -normal
            else:
                raise ValueError(f"Unknown outward_direction: {outward_direction}")

            normals[i] = normal
        self.normals = normals

    def normalize_coordinates(self, dome_height_m: float) -> "DomeMesh":
        """
        Normalize coordinates to standard convention (apex at z=0, base at z=h).

        If the mesh has apex at z=h and base at z=0 (inverted), this will
        flip the z-coordinates and adjust triangle winding to maintain
        consistent normal orientation.

        Parameters
        ----------
        dome_height_m : float
            Expected dome height [m]

        Returns
        -------
        DomeMesh
            Self (modified in place)
        """
        # Detect current convention by finding apex (min radius) position
        r = np.sqrt(self.vertices[:, 0]**2 + self.vertices[:, 1]**2)
        apex_idx = np.argmin(r)
        apex_z = self.vertices[apex_idx, 2]

        z_min = self.vertices[:, 2].min()
        z_max = self.vertices[:, 2].max()

        # Check if apex is at z_max (inverted convention)
        if abs(apex_z - z_max) < abs(apex_z - z_min):
            # Inverted: flip z-coordinates
            # Transform: z_new = z_max - z_old
            self.vertices[:, 2] = z_max - self.vertices[:, 2]

            # Flip triangle winding to maintain consistent normals
            # Swapping two vertices reverses the normal direction
            self.triangles[:, [1, 2]] = self.triangles[:, [2, 1]]

            # Clear cached normals (will be recomputed)
            self.normals = None

        return self

    def to_bempp_grid(self):
        """
        Convert to bempp-cl Grid.

        Returns
        -------
        bempp_cl.api.Grid
            BEM surface mesh
        """
        import bempp_cl.api

        # bempp expects (3, N_vertices) and (3, N_triangles)
        coords = self.vertices.T.astype(np.float64)
        cells = self.triangles.T.astype(np.uint32)

        return bempp_cl.api.Grid(coords, cells)

    def validate_quality(
        self,
        material: Optional["ShellMaterial"] = None,
        max_frequency_hz: Optional[float] = None,
        max_aspect_ratio: float = 4.0,
        min_jacobian: float = 0.3,
        min_elements_per_wavelength: int = 6,
    ) -> MeshQualityReport:
        """
        Validate mesh quality for shell FEM analysis.

        Checks:
        - Aspect ratio (important for thin shells to avoid shear locking)
        - Scaled Jacobian (element shape quality)
        - Element size vs wavelength (acoustic and bending)

        Parameters
        ----------
        material : ShellMaterial, optional
            Material for bending wavelength calculation
        max_frequency_hz : float, optional
            Maximum analysis frequency for wavelength checks [Hz]
        max_aspect_ratio : float
            Maximum acceptable aspect ratio (default: 4.0)
        min_jacobian : float
            Minimum acceptable scaled Jacobian (default: 0.3)
        min_elements_per_wavelength : int
            Minimum elements per wavelength (default: 6)

        Returns
        -------
        MeshQualityReport
            Comprehensive quality report with metrics and warnings

        Example
        -------
        >>> mesh = DomeMesher(geometry).generate(element_size_m=0.001)
        >>> report = mesh.validate_quality(
        ...     material=ShellMaterial.titanium(),
        ...     max_frequency_hz=20000
        ... )
        >>> print(report.summary())
        >>> if not report.is_valid:
        ...     print("Mesh quality issues detected!")
        """
        return validate_mesh_quality(
            vertices=self.vertices,
            triangles=self.triangles,
            material=material,
            max_frequency_hz=max_frequency_hz,
            max_aspect_ratio=max_aspect_ratio,
            min_jacobian=min_jacobian,
            min_elements_per_wavelength=min_elements_per_wavelength,
        )

    def get_aspect_ratios(self) -> np.ndarray:
        """
        Get per-element aspect ratios.

        Returns
        -------
        np.ndarray
            (M,) aspect ratios (1.0 = ideal equilateral)
        """
        return compute_aspect_ratios(self.vertices, self.triangles)

    def get_element_sizes(self) -> np.ndarray:
        """
        Get per-element characteristic sizes (incircle diameter).

        Returns
        -------
        np.ndarray
            (M,) element sizes [m]
        """
        return compute_element_sizes(self.vertices, self.triangles)

    def find_poor_elements(
        self,
        max_aspect_ratio: float = 4.0,
        min_jacobian: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find indices of elements with poor quality.

        Parameters
        ----------
        max_aspect_ratio : float
            Maximum acceptable aspect ratio
        min_jacobian : float
            Minimum acceptable scaled Jacobian

        Returns
        -------
        high_ar_indices : np.ndarray
            Indices of elements with aspect ratio > max_aspect_ratio
        low_j_indices : np.ndarray
            Indices of elements with Jacobian < min_jacobian
        """
        return find_poor_quality_elements(
            self.vertices, self.triangles, max_aspect_ratio, min_jacobian
        )


class DomeMesher:
    """
    Mesh generator for dome geometries using Gmsh + OCC.

    Creates triangular surface meshes suitable for shell FEM analysis.
    The mesh can be directly converted to a bempp-cl Grid for BEM coupling.
    """

    def __init__(self, geometry: DomeGeometry):
        """
        Initialize mesher with dome geometry.

        Parameters
        ----------
        geometry : DomeGeometry
            Parametric dome specification
        """
        if not GMSH_AVAILABLE:
            raise ImportError(
                "gmsh is required for dome meshing. Install with: pip install gmsh"
            )
        self.geometry = geometry

    def generate(
        self,
        element_size_m: Optional[float] = None,
        material: Optional[ShellMaterial] = None,
        max_frequency_hz: float = 20000.0,
        elements_per_wavelength: int = 6,
    ) -> DomeMesh:
        """
        Generate triangular surface mesh.

        Parameters
        ----------
        element_size_m : float, optional
            Target element size [m]. If not provided, computed from material
            and max_frequency_hz.
        material : ShellMaterial, optional
            Material for computing recommended mesh size
        max_frequency_hz : float
            Maximum analysis frequency for mesh sizing [Hz]
        elements_per_wavelength : int
            Elements per wavelength for mesh sizing

        Returns
        -------
        DomeMesh
            Triangulated dome surface
        """
        # Determine element size
        if element_size_m is not None:
            h = element_size_m
        elif material is not None:
            h = recommended_mesh_size(max_frequency_hz, material, elements_per_wavelength)
        else:
            # Default: λ_acoustic / 6 at 20 kHz
            h = 343.0 / max_frequency_hz / elements_per_wavelength

        # Generate based on profile type
        if self.geometry.profile == "spherical":
            return self._generate_spherical(h)
        elif self.geometry.profile == "elliptical":
            return self._generate_elliptical(h)
        else:
            return self._generate_by_revolution(h)

    def _generate_spherical(self, element_size_m: float) -> DomeMesh:
        """Generate mesh for spherical dome using OCC sphere primitive."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("spherical_dome")

        try:
            R = self.geometry.sphere_radius_m
            h = self.geometry.dome_height_m

            # Sphere centered at (0, 0, R) places apex at south pole (z=0)
            # OCC addSphere uses latitude angles: z = center_z + R*sin(lat), r = R*cos(lat)
            # At lat = -π/2 (south pole): z = R + R*(-1) = 0 (apex), r = 0
            # At lat = arcsin((h-R)/R): z = R + (h-R) = h (base), r = r_base
            center_z = R
            angle1 = -np.pi / 2  # South pole = apex at z=0
            angle2 = np.arcsin((h - R) / R)  # Base at z=h

            # Create partial sphere from apex to base
            sphere = gmsh.model.occ.addSphere(
                0, 0, center_z, R,
                angle1=angle1,
                angle2=angle2
            )
            gmsh.model.occ.synchronize()

            # Set mesh size
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size_m * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size_m * 1.5)

            # Generate 2D mesh
            gmsh.model.mesh.generate(2)

            # Extract mesh data
            return self._extract_mesh()

        finally:
            gmsh.finalize()

    def _generate_elliptical(self, element_size_m: float) -> DomeMesh:
        """Generate mesh for elliptical dome using scaled sphere."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("elliptical_dome")

        try:
            a = self.geometry.base_radius_m  # Horizontal semi-axis
            b = self.geometry.dome_height_m  # Vertical semi-axis

            # Create upper hemisphere (angle1=0 is equator, angle2=π/2 is pole)
            sphere = gmsh.model.occ.addSphere(0, 0, 0, 1.0, angle1=0, angle2=np.pi / 2)

            # Scale to ellipsoid
            gmsh.model.occ.dilate([(3, sphere)], 0, 0, 0, a, a, b)

            gmsh.model.occ.synchronize()

            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size_m * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size_m * 1.5)

            gmsh.model.mesh.generate(2)

            return self._extract_mesh()

        finally:
            gmsh.finalize()

    def _generate_by_revolution(self, element_size_m: float) -> DomeMesh:
        """Generate mesh by revolving profile curve (for parabolic/conical)."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("dome_revolution")

        try:
            # Generate profile points
            num_profile_points = max(20, int(self.geometry.dome_height_m / element_size_m * 2))
            r_profile, z_profile = self.geometry.generate_profile_points(num_profile_points)

            # Create profile as spline in OCC
            point_tags = []
            for r, z in zip(r_profile, z_profile):
                pt = gmsh.model.occ.addPoint(r, 0, z)
                point_tags.append(pt)

            # Create spline through points
            spline = gmsh.model.occ.addSpline(point_tags)

            # Create axis for revolution
            axis_start = gmsh.model.occ.addPoint(0, 0, 0)
            axis_end = gmsh.model.occ.addPoint(0, 0, self.geometry.dome_height_m)

            # Create wire from spline
            wire = gmsh.model.occ.addWire([spline])

            # Revolve around z-axis
            revolved = gmsh.model.occ.revolve(
                [(1, wire)],
                0, 0, 0,  # Origin
                0, 0, 1,  # Axis direction
                2 * np.pi  # Full revolution
            )

            gmsh.model.occ.synchronize()

            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size_m * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size_m * 1.5)

            gmsh.model.mesh.generate(2)

            return self._extract_mesh()

        finally:
            gmsh.finalize()

    def _extract_mesh(self) -> DomeMesh:
        """Extract mesh data from current Gmsh model and normalize coordinates."""
        nodes = gmsh.model.mesh.getNodes()
        node_tags = nodes[0]
        coords = nodes[1].reshape(-1, 3)

        # Find triangular elements
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        triangles = None

        for et, tags, ntags in zip(elem_types, elem_tags, elem_node_tags):
            props = gmsh.model.mesh.getElementProperties(et)
            # props = (name, dim, order, numNodes, localNodeCoord, numPrimaryNodes)
            num_nodes = props[3]
            if num_nodes == 3:  # 3-node triangle
                triangles = ntags.reshape(-1, 3).astype(int)
                break

        if triangles is None:
            raise RuntimeError("No triangular elements found in mesh")

        # Create node tag to index mapping
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

        # Reindex triangles
        triangles_reindexed = np.array([
            [tag_to_idx[t] for t in tri]
            for tri in triangles
        ])

        mesh = DomeMesh(vertices=coords, triangles=triangles_reindexed)

        # Normalize to standard convention: apex at z=0, base at z=dome_height
        mesh.normalize_coordinates(self.geometry.dome_height_m)

        # Compute normals with consistent orientation
        mesh.compute_normals(outward_direction="positive_z")

        return mesh
