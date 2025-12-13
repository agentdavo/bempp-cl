"""
Finite Element Analysis module for bempp_audio.

This module provides FEM capabilities for compression driver modeling:
- Shell geometry and meshing for thin diaphragms (domes)
- Interior acoustic FEM for phase plug channels (Helmholtz equation)
- FEM-BEM coupling utilities for vibroacoustic simulations

Key components:
- DomeGeometry: Parametric dome profile generation
- DomeMesher: Gmsh + OCC meshing for thin shells
- HelmholtzFEMSolver: 3D interior acoustic FEM (DOLFINx)
- ShellMaterial: Material property definitions
- FEM-BEM coupling utilities

Example usage:
    from bempp_audio.fea import DomeGeometry, DomeMesher, ShellMaterial

    # Create dome geometry
    geometry = DomeGeometry.spherical(
        base_diameter_m=0.035,
        dome_height_m=0.008,
    )

    # Mesh for shell FEM
    mesh = DomeMesher(geometry).generate(element_size_m=0.001)

    # Material properties
    titanium = ShellMaterial.titanium(thickness_m=50e-6)

    # Convert to bempp grid for BEM coupling
    bempp_grid = mesh.to_bempp_grid()

References:
    - Leissa, A.W. "Vibration of Plates" (1969)
    - Kirkup, S. "The Boundary Element Method in Acoustics" (2019)
    - bempp-cl FEM-BEM coupling documentation
"""

from __future__ import annotations

import importlib

__all__ = [
    # Benchmarks
    "Benchmark3TiConfig_v1",
    "Benchmark3TiFormerConfig_v1",
    "Benchmark3TiSurroundConfig_v1",
    "sweep_benchmark3ti_v1",
    # Dome geometry
    "DomeGeometry",
    "CorrugationSpec",
    # Dome meshing
    "DomeMesher",
    "DomeMesh",
    # Phase plug geometry
    "PhasePlugGeometry",
    "AnnularChannel",
    "RadialSlot",
    # Phase plug meshing
    "PhasePlugMesher",
    "PhasePlugMesh",
    # Materials
    "ShellMaterial",
    "estimate_first_bending_mode",
    "recommended_mesh_size",
    "bending_wavelength",
    "TITANIUM",
    "BERYLLIUM",
    "ALUMINUM",
    # Mesh quality validation
    "MeshQualityReport",
    "validate_mesh_quality",
    "compute_aspect_ratios",
    "compute_scaled_jacobians",
    "compute_element_sizes",
    "find_poor_quality_elements",
    # FEM-BEM coupling
    "shell_surface_to_bempp_grid",
    "shell_displacement_to_velocity",
    "create_neumann_grid_function",
    "compute_radiation_impedance",
    "pressure_to_fem_surface_load",
    "iterative_fem_bem_coupling",
    "assess_coupling_strength",
    "get_element_areas",
    "get_element_centroids",
    # Interior acoustic FEM
    "HelmholtzFEMSolver",
    "HelmholtzResult",
    "AcousticMedium",
    # Shell-acoustic coupling
    "create_dome_velocity_bc",
    "piston_velocity_profile",
    "first_mode_velocity_profile",
    "ring_source_velocity_profile",
    "interpolate_shell_to_acoustic",
    # Phase plug coupling and metrics
    "ThroatExitData",
    "PhasePlugMetrics",
    "PressureFieldVisualization",
    "extract_throat_data",
    "extract_dome_data",
    "compute_phase_plug_metrics",
    "compute_metrics_sweep",
    "throat_to_bem_monopole",
    "throat_to_bem_piston",
    "apply_radiation_bc_at_throat",
    "sample_pressure_field",
    "create_axial_sample_points",
    "create_azimuthal_sample_points",
    "summarize_metrics_sweep",
    # Elastic dome BEM solver
    "ElasticDomeBEMSolver",
    "ElasticDomeSolverOptions",
    "ElasticDomeResult",
    "ElasticDomeFrequencyResponse",
    "ModalVelocityProfile",
    "compute_modal_radiation_efficiency",
    "compute_modal_spl",
    # Elastic dome network integration
    "ElasticDomeProfile",
    "ElasticDomeNetworkAdapter",
    "RigidVsElasticComparison",
    "create_validation_workflow",
    # Structural FEM (Mindlin–Reissner plate)
    "MindlinReissnerPlateSolver",
    "RayleighDamping",
    "ModalResult",
    "HarmonicResult",
    # Curved shell (spherical cap)
    "SphericalCapMindlinReissnerShellSolver",
]

from bempp_audio.fea.benchmarks import (
    Benchmark3TiConfig_v1,
    Benchmark3TiFormerConfig_v1,
    Benchmark3TiSurroundConfig_v1,
    sweep_benchmark3ti_v1,
)
from bempp_audio.fea.dome_geometry import DomeGeometry, CorrugationSpec
from bempp_audio.fea.phase_plug_geometry import PhasePlugGeometry, AnnularChannel, RadialSlot
from bempp_audio.fea.materials import (
    ShellMaterial,
    estimate_first_bending_mode,
    recommended_mesh_size,
    bending_wavelength,
    TITANIUM,
    BERYLLIUM,
    ALUMINUM,
)
from bempp_audio.fea.mesh_quality import (
    MeshQualityReport,
    validate_mesh_quality,
    compute_aspect_ratios,
    compute_scaled_jacobians,
    compute_element_sizes,
    find_poor_quality_elements,
)

# Avoid importing MPI-sensitive (dolfinx/petsc4py) or optional heavy dependencies
# (gmsh/occ) at module import time. These symbols are resolved lazily via
# __getattr__ when accessed.
_LAZY: dict[str, tuple[str, str]] = {
    # Meshing (gmsh)
    "DomeMesher": ("bempp_audio.fea.dome_meshing", "DomeMesher"),
    "DomeMesh": ("bempp_audio.fea.dome_meshing", "DomeMesh"),
    "PhasePlugMesher": ("bempp_audio.fea.phase_plug_meshing", "PhasePlugMesher"),
    "PhasePlugMesh": ("bempp_audio.fea.phase_plug_meshing", "PhasePlugMesh"),
    # FEM-BEM coupling (dolfinx)
    "shell_surface_to_bempp_grid": ("bempp_audio.fea.fem_bem_coupling", "shell_surface_to_bempp_grid"),
    "shell_displacement_to_velocity": ("bempp_audio.fea.fem_bem_coupling", "shell_displacement_to_velocity"),
    "create_neumann_grid_function": ("bempp_audio.fea.fem_bem_coupling", "create_neumann_grid_function"),
    "compute_radiation_impedance": ("bempp_audio.fea.fem_bem_coupling", "compute_radiation_impedance"),
    "pressure_to_fem_surface_load": ("bempp_audio.fea.fem_bem_coupling", "pressure_to_fem_surface_load"),
    "iterative_fem_bem_coupling": ("bempp_audio.fea.fem_bem_coupling", "iterative_fem_bem_coupling"),
    "assess_coupling_strength": ("bempp_audio.fea.fem_bem_coupling", "assess_coupling_strength"),
    "get_element_areas": ("bempp_audio.fea.fem_bem_coupling", "get_element_areas"),
    "get_element_centroids": ("bempp_audio.fea.fem_bem_coupling", "get_element_centroids"),
    # Interior acoustic FEM (dolfinx)
    "HelmholtzFEMSolver": ("bempp_audio.fea.helmholtz_fem", "HelmholtzFEMSolver"),
    "HelmholtzResult": ("bempp_audio.fea.helmholtz_fem", "HelmholtzResult"),
    "AcousticMedium": ("bempp_audio.fea.helmholtz_fem", "AcousticMedium"),
    # Shell-acoustic coupling (dolfinx)
    "create_dome_velocity_bc": ("bempp_audio.fea.shell_acoustic_coupling", "create_dome_velocity_bc"),
    "piston_velocity_profile": ("bempp_audio.fea.shell_acoustic_coupling", "piston_velocity_profile"),
    "first_mode_velocity_profile": ("bempp_audio.fea.shell_acoustic_coupling", "first_mode_velocity_profile"),
    "ring_source_velocity_profile": ("bempp_audio.fea.shell_acoustic_coupling", "ring_source_velocity_profile"),
    "interpolate_shell_to_acoustic": ("bempp_audio.fea.shell_acoustic_coupling", "interpolate_shell_to_acoustic"),
    # Phase plug coupling and metrics (dolfinx)
    "ThroatExitData": ("bempp_audio.fea.phase_plug_coupling", "ThroatExitData"),
    "PhasePlugMetrics": ("bempp_audio.fea.phase_plug_coupling", "PhasePlugMetrics"),
    "PressureFieldVisualization": ("bempp_audio.fea.phase_plug_coupling", "PressureFieldVisualization"),
    "extract_throat_data": ("bempp_audio.fea.phase_plug_coupling", "extract_throat_data"),
    "extract_dome_data": ("bempp_audio.fea.phase_plug_coupling", "extract_dome_data"),
    "compute_phase_plug_metrics": ("bempp_audio.fea.phase_plug_coupling", "compute_phase_plug_metrics"),
    "compute_metrics_sweep": ("bempp_audio.fea.phase_plug_coupling", "compute_metrics_sweep"),
    "throat_to_bem_monopole": ("bempp_audio.fea.phase_plug_coupling", "throat_to_bem_monopole"),
    "throat_to_bem_piston": ("bempp_audio.fea.phase_plug_coupling", "throat_to_bem_piston"),
    "apply_radiation_bc_at_throat": ("bempp_audio.fea.phase_plug_coupling", "apply_radiation_bc_at_throat"),
    "sample_pressure_field": ("bempp_audio.fea.phase_plug_coupling", "sample_pressure_field"),
    "create_axial_sample_points": ("bempp_audio.fea.phase_plug_coupling", "create_axial_sample_points"),
    "create_azimuthal_sample_points": ("bempp_audio.fea.phase_plug_coupling", "create_azimuthal_sample_points"),
    "summarize_metrics_sweep": ("bempp_audio.fea.phase_plug_coupling", "summarize_metrics_sweep"),
    # Elastic dome BEM solver (bempp-cl)
    "ElasticDomeBEMSolver": ("bempp_audio.fea.elastic_dome_solver", "ElasticDomeBEMSolver"),
    "ElasticDomeSolverOptions": ("bempp_audio.fea.elastic_dome_solver", "ElasticDomeSolverOptions"),
    "ElasticDomeResult": ("bempp_audio.fea.elastic_dome_solver", "ElasticDomeResult"),
    "ElasticDomeFrequencyResponse": ("bempp_audio.fea.elastic_dome_solver", "ElasticDomeFrequencyResponse"),
    "ModalVelocityProfile": ("bempp_audio.fea.elastic_dome_solver", "ModalVelocityProfile"),
    "compute_modal_radiation_efficiency": ("bempp_audio.fea.elastic_dome_solver", "compute_modal_radiation_efficiency"),
    "compute_modal_spl": ("bempp_audio.fea.elastic_dome_solver", "compute_modal_spl"),
    # Elastic dome network integration
    "ElasticDomeProfile": ("bempp_audio.fea.elastic_dome_network", "ElasticDomeProfile"),
    "ElasticDomeNetworkAdapter": ("bempp_audio.fea.elastic_dome_network", "ElasticDomeNetworkAdapter"),
    "RigidVsElasticComparison": ("bempp_audio.fea.elastic_dome_network", "RigidVsElasticComparison"),
    "create_validation_workflow": ("bempp_audio.fea.elastic_dome_network", "create_validation_workflow"),
    # Structural FEM (dolfinx)
    "MindlinReissnerPlateSolver": ("bempp_audio.fea.shell_fem", "MindlinReissnerPlateSolver"),
    "RayleighDamping": ("bempp_audio.fea.shell_fem", "RayleighDamping"),
    "ModalResult": ("bempp_audio.fea.shell_fem", "ModalResult"),
    "HarmonicResult": ("bempp_audio.fea.shell_fem", "HarmonicResult"),
    "SphericalCapMindlinReissnerShellSolver": ("bempp_audio.fea.curved_shell_fem", "SphericalCapMindlinReissnerShellSolver"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(name)
    module_name, attr = _LAZY[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(list(globals().keys()) + list(_LAZY.keys())))
