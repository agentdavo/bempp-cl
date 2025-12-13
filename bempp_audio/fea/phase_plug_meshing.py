"""
3D tetrahedral mesh generation for phase plug acoustic domains.

Creates meshes suitable for Helmholtz FEM analysis of the acoustic
volume between a compression driver dome and the throat exit.

Uses Gmsh with OpenCASCADE kernel for geometry construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

from bempp_audio.fea.phase_plug_geometry import PhasePlugGeometry, AnnularChannel

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    gmsh = None


@dataclass
class PhasePlugMesh:
    """
    3D tetrahedral mesh of the phase plug acoustic domain.

    Contains the mesh and boundary markers for applying BCs.
    """

    # Mesh file path (Gmsh .msh format)
    msh_file: Optional[str] = None

    # Mesh statistics
    num_nodes: int = 0
    num_elements: int = 0

    # Boundary marker mapping
    boundary_markers: Dict[str, int] = None

    def to_dolfinx_mesh(self):
        """
        Load mesh into DOLFINx.

        Returns
        -------
        Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]
            (mesh, cell_tags, facet_tags)
        """
        try:
            from dolfinx.io import gmshio
            from mpi4py import MPI
        except ImportError:
            raise ImportError(
                "DOLFINx and mpi4py are required. "
                "Install with: pip install fenics-dolfinx mpi4py"
            )

        if self.msh_file is None:
            raise ValueError("No mesh file available. Call generate() first.")

        mesh, cell_tags, facet_tags = gmshio.read_from_msh(
            self.msh_file, MPI.COMM_WORLD, gdim=3
        )
        return mesh, cell_tags, facet_tags


class PhasePlugMesher:
    """
    Mesh generator for phase plug acoustic domains.

    Creates 3D tetrahedral meshes of the air volume inside the phase plug,
    with marked boundaries for:
    - Dome interface (velocity BC)
    - Throat exit (radiation BC)
    - Hard walls (channel walls, phase plug body)
    """

    # Boundary markers (consistent with PhasePlugGeometry.get_boundary_markers())
    DOME_INTERFACE = 1
    THROAT_EXIT = 2
    HARD_WALL = 3

    def __init__(self, geometry: PhasePlugGeometry):
        """
        Initialize mesher with phase plug geometry.

        Parameters
        ----------
        geometry : PhasePlugGeometry
            Phase plug specification
        """
        if not GMSH_AVAILABLE:
            raise ImportError(
                "gmsh is required for phase plug meshing. "
                "Install with: pip install gmsh"
            )
        self.geometry = geometry

    def generate(
        self,
        element_size_m: float = 0.5e-3,
        output_file: Optional[str] = None,
        html_file: Optional[str] = None,
        refine_channels: bool = True,
        channel_refinement_factor: float = 0.5,
    ) -> PhasePlugMesh:
        """
        Generate 3D tetrahedral mesh of the acoustic domain.

        Parameters
        ----------
        element_size_m : float
            Target element size [m]
        output_file : str, optional
            Path to save .msh file. If None, uses temp file.
        html_file : str, optional
            If provided, also writes an interactive Plotly HTML preview of the boundary
            triangles (requires `plotly` + `meshio`).
        refine_channels : bool
            Apply finer mesh in narrow channels
        channel_refinement_factor : float
            Refinement factor for channels (smaller = finer)

        Returns
        -------
        PhasePlugMesh
            Mesh container with file path and statistics
        """
        import tempfile
        import os

        if output_file is None:
            fd, output_file = tempfile.mkstemp(suffix=".msh")
            os.close(fd)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("phase_plug_acoustic")

        try:
            # Cheap validation before constructing CAD.
            try:
                self.geometry.validate()
            except Exception:
                pass

            # Build geometry
            self._build_geometry()

            # Set mesh sizes
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size_m * 0.3)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size_m * 1.5)

            if refine_channels:
                self._apply_channel_refinement(
                    element_size_m * channel_refinement_factor
                )

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)

            # Get statistics
            nodes = gmsh.model.mesh.getNodes()
            num_nodes = len(nodes[0])

            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
            num_elements = sum(len(tags) for tags in elem_tags)

            # Save mesh
            gmsh.write(output_file)

            mesh_out = PhasePlugMesh(
                msh_file=output_file,
                num_nodes=num_nodes,
                num_elements=num_elements,
                boundary_markers=self.geometry.get_boundary_markers(),
            )
            if html_file:
                try:
                    from bempp_audio.viz.plotly_3d import gmsh_msh_boundary_html

                    gmsh_msh_boundary_html(
                        msh_file=output_file,
                        filename=html_file,
                        title="Phase Plug Acoustic Domain (Boundary Preview)",
                    )
                except Exception:
                    # Keep meshing usable even if plotly/meshio are unavailable.
                    pass

            return mesh_out

        finally:
            gmsh.finalize()

    def _build_geometry(self) -> None:
        """Build the OCC geometry for the acoustic domain."""
        geom = self.geometry
        occ = gmsh.model.occ

        # The acoustic domain consists of:
        # 1. Dome-facing gap region (z=0..gap_height). If channels exist, model this
        #    as annular gaps aligned with each channel to avoid collapsing the geometry
        #    into a single open cylinder.
        # 2. Channel regions: annular ducts (cylinders) from entry to exit.
        # 3. Throat collector/plenum: a conical (or cylindrical) manifold that connects
        #    channel exits to the throat diameter.

        volumes = []

        # 1. Gap region (dome interface to channel entries)
        if geom.channels:
            for ch in geom.channels:
                outer = occ.addCylinder(
                    0, 0, 0,
                    0, 0, geom.gap_height_m,
                    ch.outer_radius_m,
                )
                inner = occ.addCylinder(
                    0, 0, 0,
                    0, 0, geom.gap_height_m,
                    max(0.0, ch.inner_radius_m),
                )
                gap_ann, _ = occ.cut([(3, outer)], [(3, inner)])
                volumes.extend([v[1] for v in gap_ann])
        else:
            gap_cyl = occ.addCylinder(
                0, 0, 0,  # base center
                0, 0, geom.gap_height_m,  # axis vector (height in z)
                geom.dome_radius_m,  # radius
            )
            volumes.append(gap_cyl)

        # 2. Channel regions
        for ch in geom.channels:
            # Create annular cylinder for each channel
            # Outer cylinder minus inner cylinder
            outer_cyl = occ.addCylinder(
                0, 0, ch.entry_z_m,
                0, 0, ch.depth_m,
                ch.outer_radius_m,
            )
            inner_cyl = occ.addCylinder(
                0, 0, ch.entry_z_m,
                0, 0, ch.depth_m,
                ch.inner_radius_m,
            )
            # Cut inner from outer to get annulus
            channel_vol, _ = occ.cut([(3, outer_cyl)], [(3, inner_cyl)])
            volumes.extend([v[1] for v in channel_vol])

        # 3. Throat plenum region
        # Connect channel exits to throat via a simple axisymmetric collector.
        if geom.channels:
            max_exit_z = max(ch.exit_z_m for ch in geom.channels)
            plenum_r0 = max(ch.outer_radius_m for ch in geom.channels)
            plenum_r0 = max(float(plenum_r0), float(geom.throat_radius_m))
        else:
            max_exit_z = geom.gap_height_m
            plenum_r0 = float(geom.throat_radius_m)

        # Plenum from max channel exit to throat exit
        plenum_height = geom.throat_z_m - max_exit_z
        if plenum_height > 1e-6:
            if abs(plenum_r0 - geom.throat_radius_m) < 1e-12:
                plenum = occ.addCylinder(
                    0, 0, max_exit_z,
                    0, 0, plenum_height,
                    geom.throat_radius_m,
                )
            else:
                plenum = occ.addCone(
                    0, 0, max_exit_z,
                    0, 0, plenum_height,
                    plenum_r0,
                    geom.throat_radius_m,
                )
            volumes.append(plenum)

        # Fuse all volumes into single acoustic domain
        if len(volumes) > 1:
            vol_dimtags = [(3, v) for v in volumes]
            fused, _ = occ.fuse([vol_dimtags[0]], vol_dimtags[1:])
        else:
            fused = [(3, volumes[0])]

        occ.synchronize()

        # Mark boundaries
        self._mark_boundaries()

    def _mark_boundaries(self) -> None:
        """Assign physical groups to boundary surfaces."""
        geom = self.geometry

        # Get all boundary surfaces
        surfaces = gmsh.model.getEntities(dim=2)

        dome_surfaces = []
        throat_surfaces = []
        wall_surfaces = []

        for dim, tag in surfaces:
            # Get surface bounding box to identify it
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox

            # Dome interface: z ≈ 0
            if abs(zmin) < 1e-6 and abs(zmax) < 1e-6:
                dome_surfaces.append(tag)
            # Throat exit: z ≈ throat_z
            elif abs(zmin - geom.throat_z_m) < 1e-6 and abs(zmax - geom.throat_z_m) < 1e-6:
                throat_surfaces.append(tag)
            # Everything else is hard wall
            else:
                wall_surfaces.append(tag)

        # Create physical groups
        if dome_surfaces:
            gmsh.model.addPhysicalGroup(2, dome_surfaces, self.DOME_INTERFACE)
            gmsh.model.setPhysicalName(2, self.DOME_INTERFACE, "dome_interface")

        if throat_surfaces:
            gmsh.model.addPhysicalGroup(2, throat_surfaces, self.THROAT_EXIT)
            gmsh.model.setPhysicalName(2, self.THROAT_EXIT, "throat_exit")

        if wall_surfaces:
            gmsh.model.addPhysicalGroup(2, wall_surfaces, self.HARD_WALL)
            gmsh.model.setPhysicalName(2, self.HARD_WALL, "hard_wall")

        # Physical group for the volume
        volumes = gmsh.model.getEntities(dim=3)
        if volumes:
            vol_tags = [v[1] for v in volumes]
            gmsh.model.addPhysicalGroup(3, vol_tags, 1)
            gmsh.model.setPhysicalName(3, 1, "acoustic_domain")

    def _apply_channel_refinement(self, channel_element_size: float) -> None:
        """Apply mesh refinement in narrow channels."""
        geom = self.geometry

        # Create mesh size fields for channel regions
        fields = []

        for i, ch in enumerate(geom.channels):
            # Box field for channel region
            field_id = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(field_id, "VIn", channel_element_size)
            gmsh.model.mesh.field.setNumber(field_id, "VOut", channel_element_size * 3)

            # Box bounds (cylindrical approximation as box)
            gmsh.model.mesh.field.setNumber(field_id, "XMin", -ch.outer_radius_m)
            gmsh.model.mesh.field.setNumber(field_id, "XMax", ch.outer_radius_m)
            gmsh.model.mesh.field.setNumber(field_id, "YMin", -ch.outer_radius_m)
            gmsh.model.mesh.field.setNumber(field_id, "YMax", ch.outer_radius_m)
            gmsh.model.mesh.field.setNumber(field_id, "ZMin", ch.entry_z_m)
            gmsh.model.mesh.field.setNumber(field_id, "ZMax", ch.exit_z_m)

            fields.append(field_id)

        if fields:
            # Combine fields with minimum
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)


def create_simple_test_mesh(
    dome_diameter_m: float = 0.035,
    throat_diameter_m: float = 0.025,
    element_size_m: float = 1.0e-3,
) -> Tuple[PhasePlugGeometry, PhasePlugMesh]:
    """
    Create a simple single-channel phase plug mesh for testing.

    Returns
    -------
    Tuple[PhasePlugGeometry, PhasePlugMesh]
        Geometry and mesh objects
    """
    geom = PhasePlugGeometry.single_annular(
        dome_diameter_m=dome_diameter_m,
        throat_diameter_m=throat_diameter_m,
    )
    mesher = PhasePlugMesher(geom)
    mesh = mesher.generate(element_size_m=element_size_m)
    return geom, mesh
