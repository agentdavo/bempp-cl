"""Mesh generation for acoustic radiators."""

from bempp_audio.mesh.loudspeaker_mesh import LoudspeakerMesh
from bempp_audio.mesh.conventions import MOUTH_PLANE_Z, throat_plane_z
from bempp_audio.mesh.checks import (
    MeshTopologyReport,
    topology_report,
    require_domains,
    require_manifold,
)
from bempp_audio.mesh.waveguide import (
    create_waveguide_mesh,
    WaveguideMeshConfig,
    WaveguideMeshGenerator,
    # Grading functions
    linear_grading,
    exponential_grading,
    sigmoid_grading,
)
from bempp_audio.mesh.waveguide_profile import (
    WaveguideProfileCheck,
    check_profile,
    auto_tune_morph_for_expansion,
    MouthDiffractionProxy,
    mouth_diffraction_proxy,
)
from bempp_audio.mesh.profiles import (
    conical_profile,
    exponential_profile,
    tractrix_profile,
    tractrix_horn_profile,
    sqrt_ease_profile,
    oblate_spheroidal_profile,
    os_profile,
    hyperbolic_profile,
    cts_profile,
    cts_mouth_angle_deg,
)
from bempp_audio.mesh.morph import MorphConfig, MorphTargetShape, SuperFormulaConfig
from bempp_audio.mesh.unified_enclosure import (
    UnifiedMeshConfig,
    create_waveguide_on_box_unified,
)
from bempp_audio.mesh.cabinet import (
    CabinetBuilder,
    CabinetConfig,
    CabinetGeometry,
    ChamferSpec,
    ChamferType,
    create_cabinet_geometry,
)
from bempp_audio.mesh.validation import (
    MeshResolutionValidator,
    MeshResolutionPresets,
)
from bempp_audio.mesh.reporting import log_profile_check
from bempp_audio.mesh.design import (
    conical_half_angle_deg,
    conical_half_angle_rad,
    mouth_radius_from_half_angle,
    mouth_diameter_from_half_angle,
    rectangular_mouth_from_half_angles,
    os_opening_angle_bounds_deg,
)

__all__ = [
    "LoudspeakerMesh",
    "MOUTH_PLANE_Z",
    "throat_plane_z",
    "MeshTopologyReport",
    "topology_report",
    "require_domains",
    "require_manifold",
    "create_waveguide_mesh",
    "check_profile",
    "auto_tune_morph_for_expansion",
    "WaveguideProfileCheck",
    "MouthDiffractionProxy",
    "mouth_diffraction_proxy",
    "WaveguideMeshConfig",
    "WaveguideMeshGenerator",
    "conical_profile",
    "exponential_profile",
    "tractrix_profile",
    "tractrix_horn_profile",
    "sqrt_ease_profile",
    "oblate_spheroidal_profile",
    "os_profile",
    "hyperbolic_profile",
    "cts_profile",
    "cts_mouth_angle_deg",
    "MorphConfig",
    "MorphTargetShape",
    "SuperFormulaConfig",
    "linear_grading",
    "exponential_grading",
    "sigmoid_grading",
    "UnifiedMeshConfig",
    "create_waveguide_on_box_unified",
    "CabinetBuilder",
    "CabinetConfig",
    "CabinetGeometry",
    "ChamferSpec",
    "ChamferType",
    "create_cabinet_geometry",
    "MeshResolutionValidator",
    "MeshResolutionPresets",
    "log_profile_check",
    "conical_half_angle_deg",
    "conical_half_angle_rad",
    "mouth_radius_from_half_angle",
    "mouth_diameter_from_half_angle",
    "rectangular_mouth_from_half_angles",
    "os_opening_angle_bounds_deg",
]
