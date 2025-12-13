"""
Mesh generation and management for acoustic radiators.

Provides methods to create meshes for loudspeaker simulations including
circular pistons, cones, and arbitrary geometries from STL files.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from bempp_audio._optional import optional_import, require_gmsh

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")
meshio, MESHIO_AVAILABLE = optional_import("meshio")

from bempp_audio.progress import get_logger
from bempp_audio.mesh.backend import gmsh_session, write_temp_msh, import_msh_and_cleanup


@dataclass
class MeshInfo:
    """Information about a mesh."""
    n_vertices: int
    n_elements: int
    min_edge_length: float
    max_edge_length: float
    mean_edge_length: float
    bounding_box: Tuple[np.ndarray, np.ndarray]


class LoudspeakerMesh:
    """
    Mesh generation and management for acoustic radiators.

    This class provides methods to create meshes suitable for BEM acoustic
    simulations of loudspeaker drivers, including circular pistons, cone
    profiles, and arbitrary geometries.

    Parameters
    ----------
    grid : bempp.Grid, optional
        A bempp grid object. If not provided, use class methods to create.

    Attributes
    ----------
    grid : bempp.Grid
        The underlying bempp grid object.
    """

    def __init__(self, grid=None):
        if not BEMPP_AVAILABLE:
            raise ImportError("bempp_cl is required for LoudspeakerMesh")
        self._grid = grid
        self._center = np.array([0.0, 0.0, 0.0])
        self._axis = np.array([0.0, 0.0, 1.0])
        self._info_cache: Optional[MeshInfo] = None

    @property
    def grid(self):
        """Return the bempp Grid object."""
        return self._grid

    @property
    def center(self) -> np.ndarray:
        """Center point of the radiator."""
        return self._center

    @center.setter
    def center(self, value):
        self._center = np.asarray(value)

    @property
    def axis(self) -> np.ndarray:
        """Principal axis direction (unit vector)."""
        return self._axis

    @axis.setter
    def axis(self, value):
        self._axis = np.asarray(value)
        n = np.linalg.norm(self._axis)
        if float(n) <= 0:
            raise ValueError("axis must be non-zero")
        self._axis = self._axis / n

    @classmethod
    def circular_piston(
        cls,
        radius: float,
        element_size: Optional[float] = None,
        center: Tuple[float, float, float] = (0, 0, 0),
    ) -> "LoudspeakerMesh":
        """
        Create a flat circular piston mesh.

        Parameters
        ----------
        radius : float
            Radius of the piston in meters.
        element_size : float, optional
            Target max edge length (meters). If None, defaults to `min(radius/10, 0.002)`.
        center : tuple
            Center point (x, y, z) of the piston.

        Returns
        -------
        LoudspeakerMesh
            Mesh object representing a circular piston.

        Examples
        --------
        >>> mesh = LoudspeakerMesh.circular_piston(radius=0.05)
        >>> mesh.grid.number_of_elements
        """
        if element_size is None:
            # Default targets HF accuracy; users can override for faster coarse runs.
            element_size = min(radius / 10, 0.002)

        grid = _generate_disk_grid_gmsh(center=center, radius=radius, h=float(element_size))

        mesh = cls(grid)
        mesh.center = np.array(center)
        mesh.axis = np.array([0, 0, 1])
        return mesh

    @classmethod
    def annular_piston(
        cls,
        inner_radius: float,
        outer_radius: float,
        element_size: Optional[float] = None,
        center: Tuple[float, float, float] = (0, 0, 0),
    ) -> "LoudspeakerMesh":
        """
        Create an annular (ring-shaped) piston mesh.

        Useful for modeling the radiating surface of a cone driver excluding
        the dust cap region.

        Parameters
        ----------
        inner_radius : float
            Inner radius in meters.
        outer_radius : float
            Outer radius in meters.
        element_size : float, optional
            Target max edge length (meters). If None, defaults to `min((outer-inner)/10, 0.002)`.
        center : tuple
            Center point of the piston.

        Returns
        -------
        LoudspeakerMesh
            Mesh object representing an annular piston.
        """
        if element_size is None:
            # Default targets HF accuracy; users can override for faster coarse runs.
            element_size = min((outer_radius - inner_radius) / 10, 0.002)

        grid = _generate_annulus_grid_gmsh(
            center=center,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            h=float(element_size),
        )

        mesh = cls(grid)
        mesh.center = np.array(center)
        mesh.axis = np.array([0, 0, 1])
        return mesh

    @classmethod
    def cone_profile(
        cls,
        inner_radius: float,
        outer_radius: float,
        height: float,
        curvature: float = 0.0,
        element_size: Optional[float] = None,
        center: Tuple[float, float, float] = (0, 0, 0),
    ) -> "LoudspeakerMesh":
        """
        Create an axisymmetric cone profile mesh via revolution.

        The cone profile is generated by revolving a 2D curve around the z-axis.
        The curve runs from (inner_radius, 0, height) to (outer_radius, 0, 0)
        with optional curvature.

        Parameters
        ----------
        inner_radius : float
            Inner radius at the voice coil (top of cone) in meters.
        outer_radius : float
            Outer radius at the surround (base of cone) in meters.
        height : float
            Height of the cone in meters.
        curvature : float
            Curvature parameter (-1 to 1). 0 is straight, positive is convex.
        element_size : float, optional
            Target max edge length (meters). If None, auto-calculated from geometry.
        center : tuple
            Center point at the base of the cone.

        Returns
        -------
        LoudspeakerMesh
            Mesh object representing an axisymmetric cone.
        """
        if element_size is None:
            # Estimate element size based on cone surface
            arc_length = np.sqrt((outer_radius - inner_radius)**2 + height**2)
            element_size = min(arc_length / 20, 0.002)

        grid = _generate_cone_grid_gmsh(
            inner_radius, outer_radius, height, curvature, float(element_size), center
        )

        mesh = cls(grid)
        mesh.center = np.array(center)
        mesh.axis = np.array([0, 0, 1])
        return mesh

    @classmethod
    def dome(
        cls,
        radius: float,
        depth: float,
        profile: str = "spherical",
        element_size: Optional[float] = None,
        center: Tuple[float, float, float] = (0, 0, 0),
    ) -> "LoudspeakerMesh":
        """
        Create a dome radiator mesh.

        Parameters
        ----------
        radius : float
            Radius at the base of the dome in meters.
        depth : float
            Height/depth of the dome in meters.
        profile : str
            Dome profile type: 'spherical', 'elliptical', 'parabolic'.
        element_size : float, optional
            Target max edge length (meters). If None, auto-calculated.
        center : tuple
            Center point at the base of the dome.

        Returns
        -------
        LoudspeakerMesh
            Mesh object representing a dome.
        """
        if element_size is None:
            element_size = min(radius / 15, 0.002)

        grid = _generate_dome_grid_gmsh(radius, depth, profile, float(element_size), center)

        mesh = cls(grid)
        mesh.center = np.array(center)
        mesh.axis = np.array([0, 0, 1])
        return mesh

    @classmethod
    def from_stl(
        cls,
        filename: str,
        scale: float = 1.0,
    ) -> "LoudspeakerMesh":
        """
        Import mesh from STL file.

        Parameters
        ----------
        filename : str
            Path to the STL file.
        scale : float
            Scale factor to apply (e.g., 0.001 to convert mm to m).

        Returns
        -------
        LoudspeakerMesh
            Mesh object from the STL file.
        """
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio is required for STL import")

        mesh_data = meshio.read(filename)
        vertices = mesh_data.points * scale

        # Get triangular cells
        cells = None
        for cell_block in mesh_data.cells:
            if cell_block.type == "triangle":
                cells = cell_block.data
                break

        if cells is None:
            raise ValueError("No triangular elements found in STL file")

        grid = bempp.Grid(vertices.T, cells.T)
        return cls(grid)

    @classmethod
    def from_meshio(
        cls,
        filename: str,
        scale: float = 1.0,
    ) -> "LoudspeakerMesh":
        """
        Import mesh from any meshio-supported format.

        Supports: OBJ, VTK, MSH (Gmsh), PLY, OFF, and 100+ other formats.

        Parameters
        ----------
        filename : str
            Path to the mesh file.
        scale : float
            Scale factor to apply.

        Returns
        -------
        LoudspeakerMesh
            Mesh object from the file.
        """
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio is required for mesh import")

        mesh_data = meshio.read(filename)
        vertices = mesh_data.points * scale

        # Prefer linear triangles; fall back to higher-order triangles by dropping extra nodes.
        cells = None
        for cell_block in mesh_data.cells:
            if cell_block.type == "triangle":
                cells = cell_block.data
                break
        if cells is None:
            for cell_block in mesh_data.cells:
                if cell_block.type == "triangle6":
                    cells = cell_block.data[:, :3]
                    break

        if cells is None:
            raise ValueError("No triangular elements found in mesh file")

        grid = bempp.Grid(vertices.T, cells.T)
        return cls(grid)

    def with_baffle(
        self,
        baffle_radius: float,
        element_size: Optional[float] = None,
    ) -> "LoudspeakerMesh":
        """
        Add a circular baffle around the radiator.

        Creates a grid union of the radiator and a surrounding baffle surface.
        The baffle has zero normal velocity in simulations.

        Parameters
        ----------
        baffle_radius : float
            Outer radius of the baffle in meters.
        element_size : float, optional
            Target max edge length (meters) for the baffle mesh.

        Returns
        -------
        LoudspeakerMesh
            New mesh with radiator (domain_index=0) and baffle (domain_index=1).
        """
        domains = np.unique(self._grid.domain_indices)
        if len(domains) != 1:
            raise ValueError(
                "Circular baffle cannot be added to a multi-domain mesh. "
                "Apply the baffle during mesh generation (e.g. waveguide-on-box) "
                "or provide an explicit domain mapping."
            )
        radiator_domain = int(domains[0])
        baffle_domain = radiator_domain + 1

        # Get the bounding radius of the current mesh
        vertices = self._grid.vertices
        r_max = np.max(np.sqrt(vertices[0]**2 + vertices[1]**2))

        if baffle_radius <= r_max:
            raise ValueError(
                f"Baffle radius ({baffle_radius}) must be larger than "
                f"radiator radius ({r_max})"
            )

        if element_size is None:
            element_size = (baffle_radius - r_max) / 10

        # Create annular baffle mesh (small gap to avoid overlapping)
        baffle_grid = _generate_annulus_grid_gmsh(
            center=tuple(self._center),
            inner_radius=r_max * 1.001,
            outer_radius=baffle_radius,
            h=float(element_size),
        )

        # Union with domain indices
        combined_grid = bempp.grid.union(
            [self._grid, baffle_grid],
            domain_indices=[radiator_domain, baffle_domain]
        )

        mesh = LoudspeakerMesh(combined_grid)
        mesh.center = self._center
        mesh.axis = self._axis
        return mesh

    def refine(
        self,
        max_element_size: Optional[float] = None,
        max_frequency: Optional[float] = None,
        c: float = 343.0,
        elements_per_wavelength: int = 6,
    ) -> "LoudspeakerMesh":
        """
        Refine mesh based on element size or target frequency.

        Parameters
        ----------
        max_element_size : float, optional
            Maximum element size in meters.
        max_frequency : float, optional
            Maximum frequency for the simulation. Element size will be
            calculated as wavelength / elements_per_wavelength.
        c : float
            Speed of sound in m/s. Default 343.0.
        elements_per_wavelength : int
            Number of elements per wavelength. Default 6.

        Returns
        -------
        LoudspeakerMesh
            New refined mesh.
        """
        if max_frequency is not None:
            wavelength = c / max_frequency
            target_h = wavelength / elements_per_wavelength
        elif max_element_size is not None:
            target_h = max_element_size
        else:
            raise ValueError("Specify either max_element_size or max_frequency")

        # Check current mesh density
        info = self.info()
        if info.mean_edge_length <= target_h:
            return self  # Already fine enough

        # TODO: Implement mesh refinement via Gmsh re-meshing
        # For now, raise an informative error
        raise NotImplementedError(
            f"Mesh refinement not yet implemented. Current mean edge: "
            f"{info.mean_edge_length:.4f}m, target: {target_h:.4f}m. "
            f"Re-create mesh with smaller h parameter."
        )

    def info(self) -> MeshInfo:
        """
        Get information about the mesh.

        Returns
        -------
        MeshInfo
            Dataclass with mesh statistics.
        """
        if self._info_cache is not None:
            return self._info_cache

        vertices = self._grid.vertices  # Shape (3, n_vertices)
        elements = self._grid.elements  # Shape (3, n_elements)

        # Calculate edge lengths
        edge_lengths = []
        for i in range(elements.shape[1]):
            v0, v1, v2 = elements[:, i]
            p0 = vertices[:, v0]
            p1 = vertices[:, v1]
            p2 = vertices[:, v2]
            edge_lengths.extend([
                np.linalg.norm(p1 - p0),
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p0 - p2),
            ])
        edge_lengths = np.array(edge_lengths)

        # Bounding box
        bbox_min = vertices.min(axis=1)
        bbox_max = vertices.max(axis=1)

        info = MeshInfo(
            n_vertices=vertices.shape[1],
            n_elements=elements.shape[1],
            min_edge_length=edge_lengths.min(),
            max_edge_length=edge_lengths.max(),
            mean_edge_length=edge_lengths.mean(),
            bounding_box=(bbox_min, bbox_max),
        )
        self._info_cache = info
        return info

    def clear_info_cache(self) -> None:
        """Clear cached mesh statistics computed by `info()`."""
        self._info_cache = None

    def to_dict(self) -> dict:
        """Serialize mesh to dictionary for multiprocessing."""
        return {
            "vertices": self._grid.vertices.copy(),
            "elements": self._grid.elements.copy(),
            "domain_indices": self._grid.domain_indices.copy(),
            "center": self._center.copy(),
            "axis": self._axis.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoudspeakerMesh":
        """Reconstruct mesh from dictionary."""
        domain_indices = data.get("domain_indices", None)
        grid = bempp.Grid(data["vertices"], data["elements"], domain_indices=domain_indices)
        mesh = cls(grid)
        mesh.center = data["center"]
        mesh.axis = data["axis"]
        return mesh

    def print_quality_report(self, by_domain: bool = True, units: str = 'mm') -> None:
        """
        Print comprehensive mesh quality report.
        
        Parameters
        ----------
        by_domain : bool
            If True, report quality for each domain separately (default True).
        units : str
            Units for display: 'mm' (default), 'm', 'um'.
        
        Examples
        --------
        >>> mesh.print_quality_report()
        Mesh Quality Report:
          Vertices: 1474
          Elements: 2824
          ...
        """
        logger = get_logger()
        scale = {'mm': 1000, 'm': 1, 'um': 1e6}[units]
        
        info = self.info()
        
        logger.subsection("Mesh Quality Report")
        logger.config("Vertices", info.n_vertices)
        logger.config("Elements", info.n_elements)
        logger.info(f"  Overall edge lengths ({units}):")
        logger.info(f"    Min:  {info.min_edge_length * scale:.2f}")
        logger.info(f"    Max:  {info.max_edge_length * scale:.2f}")
        logger.info(f"    Mean: {info.mean_edge_length * scale:.2f}")
        
        if by_domain and len(np.unique(self.grid.domain_indices)) > 1:
            logger.blank()
            logger.info("  By domain:")
            for domain_id in np.unique(self.grid.domain_indices):
                stats = self._compute_domain_stats(domain_id)
                n_elem = stats['n_elements']
                pct = 100 * n_elem / info.n_elements
                logger.info(f"    Domain {domain_id}: {n_elem} elements ({pct:.1f}%)")
                logger.info(f"      Min edge:  {stats['min_edge'] * scale:.2f} {units}")
                logger.info(f"      Max edge:  {stats['max_edge'] * scale:.2f} {units}")
                logger.info(f"      Mean edge: {stats['mean_edge'] * scale:.2f} {units}")
    
    def _compute_domain_stats(self, domain_id: int) -> dict:
        """Compute edge statistics for a specific domain."""
        vertices = self.grid.vertices
        elements = self.grid.elements
        domain_mask = self.grid.domain_indices == domain_id
        
        edge_lengths = []
        for i, elem in enumerate(elements.T):
            if domain_mask[i]:
                v0 = vertices[:, elem[0]]
                v1 = vertices[:, elem[1]]
                v2 = vertices[:, elem[2]]
                edge_lengths.extend([
                    np.linalg.norm(v1 - v0),
                    np.linalg.norm(v2 - v1),
                    np.linalg.norm(v0 - v2)
                ])
        
        edge_lengths = np.array(edge_lengths)
        return {
            'n_elements': int(np.sum(domain_mask)),
            'min_edge': float(edge_lengths.min()),
            'max_edge': float(edge_lengths.max()),
            'mean_edge': float(edge_lengths.mean()),
        }

    def __repr__(self) -> str:
        if self._grid is None:
            return "LoudspeakerMesh(empty)"
        info = self.info()
        return (
            f"LoudspeakerMesh(vertices={info.n_vertices}, "
            f"elements={info.n_elements}, "
            f"mean_edge={info.mean_edge_length:.4f}m)"
        )


# ============================================================================
# Private helper functions for Gmsh geometry generation
# ============================================================================


def _generate_disk_grid_gmsh(center: Tuple[float, float, float], radius: float, h: float):
    """Generate a 2D disk mesh using the Gmsh Python API (OCC)."""
    gmsh = require_gmsh()

    if radius <= 0:
        raise ValueError("radius must be positive")
    if h <= 0:
        raise ValueError("h must be positive")

    cx, cy, cz = center

    with gmsh_session("disk", terminal=0):
        surf = gmsh.model.occ.addDisk(cx, cy, cz, radius, radius)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surf], 1)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(2)
        msh_name = write_temp_msh()
        gmsh.write(msh_name)

    return import_msh_and_cleanup(msh_name)


def _generate_annulus_grid_gmsh(
    center: Tuple[float, float, float],
    inner_radius: float,
    outer_radius: float,
    h: float,
):
    """Generate a 2D annulus mesh using the Gmsh Python API (OCC)."""
    gmsh = require_gmsh()

    if inner_radius < 0:
        raise ValueError("inner_radius must be non-negative")
    if outer_radius <= 0:
        raise ValueError("outer_radius must be positive")
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be larger than inner_radius")
    if h <= 0:
        raise ValueError("h must be positive")

    cx, cy, cz = center

    with gmsh_session("annulus", terminal=0):
        outer = gmsh.model.occ.addDisk(cx, cy, cz, outer_radius, outer_radius)
        inner = gmsh.model.occ.addDisk(cx, cy, cz, inner_radius, inner_radius)
        cut, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        surfaces = [tag for dim, tag in cut if dim == 2]
        if not surfaces:
            raise RuntimeError("Gmsh failed to create annulus surface")
        gmsh.model.addPhysicalGroup(2, surfaces, 1)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(2)
        msh_name = write_temp_msh()
        gmsh.write(msh_name)

    return import_msh_and_cleanup(msh_name)


def _generate_cone_grid_gmsh(
    inner_r: float,
    outer_r: float,
    height: float,
    curvature: float,
    h: float,
    center: tuple,
):
    """Generate cone mesh using Gmsh Python API with OCC kernel."""
    gmsh = require_gmsh()

    cx, cy, cz = center

    with gmsh_session("cone", terminal=0):
        # Generate profile points
        n_profile_points = 20
        t = np.linspace(0, 1, n_profile_points)

        # Linear interpolation with curvature
        r_profile = inner_r + (outer_r - inner_r) * t
        if abs(curvature) < 1e-6:
            z_profile = height * (1 - t)
        else:
            z_linear = height * (1 - t)
            z_deviation = 4 * curvature * height * 0.1 * t * (1 - t)
            z_profile = z_linear + z_deviation

        # Create profile points
        point_tags = []
        for r, z in zip(r_profile, z_profile):
            tag = gmsh.model.occ.addPoint(cx + r, cy, cz + z)
            point_tags.append(tag)

        # Create spline through points
        spline_tag = gmsh.model.occ.addSpline(point_tags)

        # Revolve around z-axis (full 2*pi rotation)
        result = gmsh.model.occ.revolve(
            [(1, spline_tag)],  # Entities to revolve (1D curve)
            cx, cy, cz,         # Point on axis
            0, 0, 1,            # Axis direction
            2 * np.pi           # Angle (full revolution)
        )

        # Find the created surface
        surface_tags = [tag for dim, tag in result if dim == 2]
        if not surface_tags:
            raise RuntimeError("Gmsh OCC revolution did not create any surfaces")

        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, surface_tags, 1)

        # Set mesh parameters
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

        gmsh.model.mesh.generate(2)

        msh_name = write_temp_msh()
        gmsh.write(msh_name)

    return import_msh_and_cleanup(msh_name)


def _generate_dome_grid_gmsh(
    radius: float,
    depth: float,
    profile: str,
    h: float,
    center: tuple,
):
    """Generate dome mesh using Gmsh Python API with OCC kernel."""
    gmsh = require_gmsh()

    cx, cy, cz = center

    with gmsh_session("dome", terminal=0):
        if profile == "spherical":
            # For a spherical cap: R^2 = radius^2 + (R-depth)^2
            R = (radius**2 + depth**2) / (2 * depth)
            sphere_center_z = cz + depth - R

            # Create sphere and cut it
            sphere = gmsh.model.occ.addSphere(cx, cy, sphere_center_z, R)

            # Create cutting box to remove the part below z=cz
            box = gmsh.model.occ.addBox(
                cx - 2*R, cy - 2*R, cz - 2*R,
                4*R, 4*R, 2*R
            )

            # Cut sphere with box
            result, _ = gmsh.model.occ.cut([(3, sphere)], [(3, box)])
            gmsh.model.occ.synchronize()

            # Get surface boundary of the remaining volume
            surfaces = gmsh.model.getBoundary(result, combined=False, oriented=False)
            surface_tags = [abs(tag) for dim, tag in surfaces if dim == 2]

            # Add physical surface
            if surface_tags:
                gmsh.model.addPhysicalGroup(2, surface_tags, 1)

        elif profile in ("elliptical", "parabolic"):
            # Use OCC kernel for proper full-revolution surfaces
            n_points = 20
            t = np.linspace(0, np.pi/2, n_points)

            if profile == "elliptical":
                r_profile = radius * np.sin(t)
                z_profile = depth * np.cos(t)
            else:  # parabolic
                r_profile = np.linspace(0, radius, n_points)
                z_profile = depth * (1 - (r_profile / radius)**2)

            # Create profile points using OCC kernel
            point_tags = []
            for r, z in zip(r_profile, z_profile):
                tag = gmsh.model.occ.addPoint(cx + r, cy, cz + z)
                point_tags.append(tag)

            # Create spline through points
            spline_tag = gmsh.model.occ.addSpline(point_tags)

            # Revolve around z-axis using OCC
            result = gmsh.model.occ.revolve(
                [(1, spline_tag)],
                cx, cy, cz,
                0, 0, 1,
                2 * np.pi
            )

            # Find created surfaces
            surface_tags = [tag for dim, tag in result if dim == 2]
            if not surface_tags:
                raise RuntimeError("Gmsh OCC revolution did not create any surfaces")

            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2, surface_tags, 1)

        else:
            raise ValueError(f"Unknown dome profile: {profile}")

        # Set mesh parameters
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(2)

        msh_name = write_temp_msh()
        gmsh.write(msh_name)

    return import_msh_and_cleanup(msh_name)
