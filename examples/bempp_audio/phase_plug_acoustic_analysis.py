#!/usr/bin/env python3
"""
Phase Plug Acoustic Analysis Example

Demonstrates the interior acoustic FEM workflow for compression driver
phase plug modeling:

1. Create parametric phase plug geometry (dual annular slots)
2. Generate 3D tetrahedral mesh with Gmsh
3. Set up Helmholtz FEM with DOLFINx
4. Apply boundary conditions:
   - Velocity BC on dome interface (piston motion)
   - Radiation impedance BC at throat exit
   - Hard walls elsewhere
5. Solve at multiple frequencies
6. Analyze pressure distribution and impedance transformation

This example requires DOLFINx to be installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

# Check for DOLFINx availability
try:
    import dolfinx
    from mpi4py import MPI
    DOLFINX_AVAILABLE = True
except ImportError:
    DOLFINX_AVAILABLE = False
    print("DOLFINx not available - running geometry/mesh demo only")

from bempp_audio.fea import (
    PhasePlugGeometry,
    PhasePlugMesher,
    piston_velocity_profile,
    first_mode_velocity_profile,
)


def main():
    print("=" * 60)
    print("Phase Plug Acoustic Analysis")
    print("=" * 60)
    print()

    # 1. Create phase plug geometry
    print("1. Creating phase plug geometry...")
    geometry = PhasePlugGeometry.dual_annular(
        dome_diameter_m=0.044,       # 44mm dome (1.75" driver)
        throat_diameter_m=0.0254,    # 1" throat
        gap_height_m=0.5e-3,         # 0.5mm air gap
        channel_width_m=1.0e-3,      # 1mm channel width
        channel_depth_m=6.0e-3,      # 6mm channel depth
    )
    print(geometry.summary())
    print()

    # 2. Generate mesh
    print("2. Generating 3D mesh...")
    mesher = PhasePlugMesher(geometry)
    mesh = mesher.generate(
        element_size_m=0.8e-3,       # 0.8mm elements
        refine_channels=True,
        channel_refinement_factor=0.5,
    )
    print(f"   Nodes: {mesh.num_nodes:,}")
    print(f"   Elements: {mesh.num_elements:,}")
    print(f"   Mesh file: {mesh.msh_file}")
    print()

    # 3. Acoustic analysis (requires DOLFINx)
    if DOLFINX_AVAILABLE:
        print("3. Running Helmholtz FEM analysis...")
        run_acoustic_analysis(geometry, mesh)
    else:
        print("3. Skipping FEM analysis (DOLFINx not installed)")
        print("   Install with: pip install fenics-dolfinx")
        print()
        demonstrate_velocity_profiles(geometry)


def run_acoustic_analysis(geometry: PhasePlugGeometry, mesh_container):
    """Run full Helmholtz FEM analysis (requires DOLFINx)."""
    from bempp_audio.fea import HelmholtzFEMSolver, AcousticMedium
    from bempp_audio.fea import compute_phase_plug_metrics

    # Load mesh into DOLFINx
    print("   Loading mesh into DOLFINx...")
    dfx_mesh, cell_tags, facet_tags = mesh_container.to_dolfinx_mesh()

    # Create solver
    medium = AcousticMedium()  # Air at 20°C
    solver = HelmholtzFEMSolver(dfx_mesh, degree=1, medium=medium)
    solver.set_facet_tags(facet_tags)

    # Boundary conditions
    # - Dome interface (marker 1): uniform velocity (piston motion)
    # - Throat exit (marker 2): radiation impedance BC
    # - Hard walls (marker 3): ∂p/∂n = 0 (default)

    # Piston velocity: 1 mm/s amplitude
    v_n = 1.0e-3  # m/s
    solver.add_velocity_bc(v_n, marker=1)

    # Radiation BC at throat: plane wave impedance
    Z0 = medium.characteristic_impedance
    solver.add_impedance_bc(Z0, marker=2)

    # Frequency sweep
    frequencies = [1000, 2000, 4000, 8000, 16000]  # Hz
    print(f"   Solving at {len(frequencies)} frequencies...")

    results = []
    metrics_list = []
    for f in frequencies:
        result = solver.solve(f)
        results.append(result)
        try:
            metrics_list.append(compute_phase_plug_metrics(result, facet_tags, 1, 2, v_n))
        except Exception:
            pass
        print(f"   f = {f:5d} Hz: solved")

    # Analyze results
    print()
    print("4. Analyzing results...")
    analyze_results(geometry, results, medium)

    # Export dashboard (best-effort)
    if metrics_list:
        try:
            out_dir = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs")).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_html = out_dir / f"{Path(__file__).stem}_dashboard.html"
            from bempp_audio.viz.plotly_phase_plug import save_phase_plug_dashboard_html

            save_phase_plug_dashboard_html(
                metrics_list,
                filename=str(out_html),
                title="Phase Plug Acoustic Analysis — Metrics Dashboard",
            )
            print(f"\nDashboard exported: {out_html}")
        except Exception as e:
            print(f"\nDashboard export skipped: {e}")


def analyze_results(geometry: PhasePlugGeometry, results, medium):
    """Analyze pressure distribution from FEM results."""
    print()
    print("   Frequency Response Summary:")
    print("   " + "-" * 50)
    print(f"   {'Freq (Hz)':<10} {'k (rad/m)':<12} {'λ (mm)':<10}")
    print("   " + "-" * 50)

    for result in results:
        k = result.wavenumber
        wavelength = 2 * np.pi / k * 1000  # mm
        print(f"   {result.frequency_hz:<10.0f} {k:<12.2f} {wavelength:<10.1f}")

    print()
    print("   Acoustic domain metrics:")
    print(f"   - Dome area: {geometry.dome_area_m2 * 1e6:.1f} mm²")
    print(f"   - Throat area: {geometry.throat_area_m2 * 1e6:.1f} mm²")
    print(f"   - Compression ratio: {geometry.compression_ratio:.1f}:1")
    print(f"   - Total channel area: {geometry.total_channel_area_m2 * 1e6:.1f} mm²")


def demonstrate_velocity_profiles(geometry: PhasePlugGeometry):
    """Show velocity profile options when DOLFINx not available."""
    print()
    print("Velocity Profile Examples")
    print("-" * 40)

    # Sample radii
    r = np.linspace(0, geometry.dome_radius_m, 20)

    # Piston profile
    piston = piston_velocity_profile(amplitude=1.0e-3)
    v_piston = np.array([piston(ri) for ri in r])

    # First mode profile
    first_mode = first_mode_velocity_profile(
        dome_radius_m=geometry.dome_radius_m,
        amplitude=1.0e-3
    )
    v_mode = np.array([first_mode(ri) for ri in r])

    print(f"{'r (mm)':<10} {'Piston (mm/s)':<15} {'1st Mode (mm/s)':<15}")
    print("-" * 40)
    for ri, vp, vm in zip(r * 1000, np.abs(v_piston) * 1000, np.abs(v_mode) * 1000):
        print(f"{ri:<10.2f} {vp:<15.3f} {vm:<15.3f}")


if __name__ == "__main__":
    main()
