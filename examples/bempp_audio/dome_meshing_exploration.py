#!/usr/bin/env python3
"""
Exploration: Dome meshing with Gmsh + OCC for thin-shell FEM-BEM coupling.

This script demonstrates how to generate a shell mesh for a compression driver
dome using Gmsh's OpenCASCADE (OCC) kernel. The output is a triangular surface
mesh suitable for shell FEM analysis and subsequent BEM coupling.

This is a proof-of-concept for TASKS.md Part 7 (Thin-Shell Dome Modeling).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    gmsh = None


def create_spherical_dome_mesh(
    *,
    dome_radius_m: float = 0.020,  # Radius of curvature
    dome_height_m: float = 0.008,  # Height from base to apex
    base_diameter_m: float = 0.035,  # Dome base diameter
    mesh_size_m: float = 0.002,  # Target element size
    output_file: str | None = None,
) -> tuple:
    """
    Create a spherical dome shell mesh using Gmsh + OCC.

    Returns (vertices, triangles) as numpy arrays.
    """
    if not GMSH_AVAILABLE:
        raise ImportError("gmsh is required for dome meshing")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("dome")

    # Calculate sphere parameters from dome geometry
    # For a spherical cap: h = R - sqrt(R^2 - r^2), where r = base_radius
    base_radius = base_diameter_m / 2
    # Solve for R: R = (r^2 + h^2) / (2h)
    sphere_radius = (base_radius**2 + dome_height_m**2) / (2 * dome_height_m)
    center_z = -(sphere_radius - dome_height_m)

    # Create only the upper portion of the sphere using parametric bounds
    # Sphere is parameterized as: x = R*sin(phi)*cos(theta), y = R*sin(phi)*sin(theta), z = R*cos(phi)
    # phi goes from 0 (north pole) to pi (south pole)
    # We want phi from 0 to phi_max where z >= 0
    # cos(phi_max) = -center_z / sphere_radius

    phi_max = np.arccos(-center_z / sphere_radius) if abs(center_z) < sphere_radius else np.pi / 2

    # Use addSphere with angle limits
    # addSphere(x, y, z, r, angle1=-pi/2, angle2=pi/2, angle3=2*pi)
    # angle1 and angle2 are latitude angles from -pi/2 to pi/2
    # For a cap from north pole down to phi_max, we want angle2 = pi/2 - phi_max

    angle2 = np.pi / 2 - phi_max

    sphere_tag = gmsh.model.occ.addSphere(0, 0, center_z, sphere_radius, angle1=angle2, angle2=np.pi/2)
    gmsh.model.occ.synchronize()

    # Get all surfaces
    surfaces = gmsh.model.getEntities(dim=2)
    if not surfaces:
        gmsh.finalize()
        raise RuntimeError("No surfaces created")

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_m * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_m * 1.5)

    # Generate 2D surface mesh
    gmsh.model.mesh.generate(2)

    # Extract mesh data
    nodes = gmsh.model.mesh.getNodes()
    node_tags = nodes[0]
    coords = nodes[1].reshape(-1, 3)

    # Get triangular elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    triangles = None
    for et, tags, ntags in zip(elem_types, elem_tags, elem_node_tags):
        props = gmsh.model.mesh.getElementProperties(et)
        # props = (name, dim, order, numNodes, localNodeCoord, numPrimaryNodes)
        if int(props[3]) == 3:  # 3-node triangle
            triangles = ntags.reshape(-1, 3).astype(int)
            break

    if triangles is None:
        gmsh.finalize()
        raise RuntimeError("No triangular elements found")

    # Build mapping from node tags to sequential indices
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    triangles_reindexed = np.array([[tag_to_idx[int(t)] for t in tri] for tri in triangles], dtype=int)

    vertices = coords

    if output_file:
        gmsh.write(output_file)
        print(f"Saved mesh to {output_file}")

    gmsh.finalize()

    return vertices, triangles_reindexed


def create_elliptical_dome_mesh(
    *,
    major_axis_m: float = 0.020,  # Semi-major axis (horizontal)
    minor_axis_m: float = 0.010,  # Semi-minor axis (vertical = dome height)
    mesh_size_m: float = 0.002,
    output_file: str | None = None,
) -> tuple:
    """
    Create an elliptical dome shell mesh using Gmsh + OCC.

    The dome is the upper half of an oblate ellipsoid.
    """
    if not GMSH_AVAILABLE:
        raise ImportError("gmsh is required for dome meshing")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("elliptical_dome")

    # Create upper hemisphere only using angle limits
    # angle1=0 starts at equator, angle2=pi/2 goes to north pole
    sphere = gmsh.model.occ.addSphere(0, 0, 0, 1.0, angle1=0, angle2=np.pi/2)

    # Scale to ellipsoid dimensions
    gmsh.model.occ.dilate([(3, sphere)], 0, 0, 0, major_axis_m, major_axis_m, minor_axis_m)

    gmsh.model.occ.synchronize()

    # Get all surfaces
    surfaces = gmsh.model.getEntities(dim=2)
    if not surfaces:
        gmsh.finalize()
        raise RuntimeError("No surfaces created")

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_m * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_m * 1.5)

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Extract mesh data
    nodes = gmsh.model.mesh.getNodes()
    node_tags = nodes[0]
    coords = nodes[1].reshape(-1, 3)

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    triangles = None
    for et, tags, ntags in zip(elem_types, elem_tags, elem_node_tags):
        props = gmsh.model.mesh.getElementProperties(et)
        if int(props[3]) == 3:  # 3-node triangle
            triangles = ntags.reshape(-1, 3).astype(int)
            break

    if triangles is None:
        gmsh.finalize()
        raise RuntimeError("No triangular elements found")

    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    triangles_reindexed = np.array([[tag_to_idx[int(t)] for t in tri] for tri in triangles], dtype=int)

    vertices = coords

    if output_file:
        gmsh.write(output_file)

    gmsh.finalize()

    return vertices, triangles_reindexed


def estimate_first_bending_mode(
    *,
    thickness_m: float,
    radius_m: float,
    youngs_modulus_pa: float,
    density_kg_m3: float,
    poissons_ratio: float,
) -> float:
    """
    Estimate first bending mode frequency for a clamped circular plate.

    Uses the classical formula: f1 = (10.21 / 2π) * (h / R²) * sqrt(D / (ρh))
    where D = Eh³ / (12(1-ν²)) is the flexural rigidity.

    For a dome (curved), the actual frequency will be higher due to membrane stiffening.
    """
    h = thickness_m
    r = radius_m
    e = youngs_modulus_pa
    rho = density_kg_m3
    nu = poissons_ratio

    # Flexural rigidity
    d = e * h**3 / (12 * (1 - nu**2))

    # First mode eigenvalue for clamped circular plate
    lambda_01 = 10.21  # First axisymmetric mode

    # Frequency
    f1 = (lambda_01 / (2 * np.pi)) * (1 / r**2) * np.sqrt(d / (rho * h))

    return f1


def main():
    print("Dome Meshing Exploration for Thin-Shell FEM-BEM Coupling")
    print("=" * 60)

    # Typical titanium dome parameters
    thickness_um = 50
    diameter_mm = 35
    dome_height_mm = 8

    thickness_m = thickness_um * 1e-6
    diameter_m = diameter_mm * 1e-3
    dome_height_m = dome_height_mm * 1e-3

    # Material properties (Grade 1 Titanium)
    e_titanium = 110e9  # Pa
    rho_titanium = 4500  # kg/m³
    nu_titanium = 0.34

    print(f"\nDome parameters:")
    print(f"  Thickness: {thickness_um} μm")
    print(f"  Diameter: {diameter_mm} mm")
    print(f"  Dome height: {dome_height_mm} mm")

    print(f"\nMaterial (Titanium):")
    print(f"  Young's modulus: {e_titanium / 1e9:.0f} GPa")
    print(f"  Density: {rho_titanium:.0f} kg/m³")
    print(f"  Poisson's ratio: {nu_titanium:.2f}")

    # Estimate first bending mode (flat plate approximation)
    f1_flat = estimate_first_bending_mode(
        thickness_m=thickness_m,
        radius_m=diameter_m / 2,
        youngs_modulus_pa=e_titanium,
        density_kg_m3=rho_titanium,
        poissons_ratio=nu_titanium,
    )
    print(f"\nFirst bending mode estimate (flat plate): {f1_flat:.0f} Hz")
    print("  (Actual dome frequency will be higher due to membrane stiffening)")

    # Meshing parameters
    # At 20 kHz, wavelength in air is ~17mm, so λ/6 ≈ 2.8mm
    # But bending wavelength in titanium dome is shorter
    acoustic_wavelength_20khz = 343 / 20000  # ~17mm
    mesh_size_acoustic = acoustic_wavelength_20khz / 6  # ~2.8mm

    # Bending wavelength estimate
    omega_20khz = 2 * np.pi * 20000
    d_flex = e_titanium * thickness_m**3 / (12 * (1 - nu_titanium**2))
    bending_wavelength = (2 * np.pi) * (d_flex / (rho_titanium * thickness_m * omega_20khz**2))**0.25
    mesh_size_bending = bending_wavelength / 6

    print(f"\nMesh size recommendations:")
    print(f"  Acoustic wavelength at 20 kHz: {acoustic_wavelength_20khz * 1000:.1f} mm")
    print(f"  Mesh size (acoustic, λ/6): {mesh_size_acoustic * 1000:.2f} mm")
    print(f"  Bending wavelength at 20 kHz: {bending_wavelength * 1000:.2f} mm")
    print(f"  Mesh size (bending, λ/6): {mesh_size_bending * 1000:.3f} mm")
    print(f"  Use: min of both ≈ {min(mesh_size_acoustic, mesh_size_bending) * 1000:.2f} mm")

    if GMSH_AVAILABLE:
        print("\n" + "=" * 60)
        print("Generating dome meshes...")

        # Conservative mesh size for demonstration
        mesh_size = max(0.001, min(mesh_size_acoustic, mesh_size_bending) * 2)

        # Spherical dome
        try:
            verts_sph, tris_sph = create_spherical_dome_mesh(
                dome_radius_m=0.025,
                dome_height_m=dome_height_m,
                base_diameter_m=diameter_m,
                mesh_size_m=mesh_size,
            )
            print(f"\nSpherical dome mesh:")
            print(f"  Vertices: {len(verts_sph)}")
            print(f"  Triangles: {len(tris_sph)}")
            print(f"  Z range: {verts_sph[:, 2].min():.4f} to {verts_sph[:, 2].max():.4f} m")
        except Exception as e:
            print(f"\nSpherical dome meshing failed: {e}")

        # Elliptical dome
        try:
            verts_ell, tris_ell = create_elliptical_dome_mesh(
                major_axis_m=diameter_m / 2,
                minor_axis_m=dome_height_m,
                mesh_size_m=mesh_size,
            )
            print(f"\nElliptical dome mesh:")
            print(f"  Vertices: {len(verts_ell)}")
            print(f"  Triangles: {len(tris_ell)}")
            print(f"  Z range: {verts_ell[:, 2].min():.4f} to {verts_ell[:, 2].max():.4f} m")
        except Exception as e:
            print(f"\nElliptical dome meshing failed: {e}")

        print("\nDome meshing exploration complete.")
    else:
        print("\ngmsh not available - skipping mesh generation")
        print("Install with: pip install gmsh")


if __name__ == "__main__":
    main()
