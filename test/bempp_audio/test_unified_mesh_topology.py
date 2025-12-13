import pytest

from bempp_audio._optional import optional_import


gmsh, GMSH_AVAILABLE = optional_import("gmsh")
bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")


pytestmark = pytest.mark.skipif(
    not (GMSH_AVAILABLE and BEMPP_AVAILABLE),
    reason="Requires gmsh + bempp_cl",
)


def _closed(mesh):
    from bempp_audio.mesh.validation import MeshTopologyValidator

    ok, info = MeshTopologyValidator.validate_mesh(mesh, verbose=False)
    return ok, info


@pytest.mark.parametrize("use_morph", [False, True])
@pytest.mark.parametrize("blend_mode", ["none", "chamfer", "fillet"])
def test_unified_waveguide_on_box_mesh_is_closed(use_morph: bool, blend_mode: str) -> None:
    from bempp_audio.mesh.unified_enclosure import UnifiedMeshConfig, create_waveguide_on_box_unified

    morph = None
    corner_resolution = 0
    if use_morph:
        from bempp_audio.mesh.morph import MorphConfig

        morph = MorphConfig.rectangle(
            width=0.15,
            height=0.10,
            corner_radius=0.01,
            fixed_part=0.0,
            end_part=1.0,
            rate=2.0,
        )
        corner_resolution = 2

    chamfer_kwargs = {}
    fillet_kwargs = {}
    if blend_mode == "chamfer":
        from bempp_audio.mesh.cabinet import ChamferSpec

        chamfer_kwargs = {
            "chamfer_bottom": ChamferSpec.asymmetric(0.008, 0.010),
            "chamfer_top": ChamferSpec.symmetric(0.006),
            "chamfer_left": ChamferSpec.symmetric(0.004),
            "chamfer_right": ChamferSpec.angled(0.005, 35.0),
        }
    elif blend_mode == "fillet":
        fillet_kwargs = {
            "fillet_bottom_radius": 0.006,
            "fillet_top_radius": 0.004,
            "fillet_left_radius": 0.005,
            "fillet_right_radius": 0.003,
        }

    cfg = UnifiedMeshConfig(
        throat_diameter=0.025,
        mouth_diameter=0.15,
        waveguide_length=0.10,
        box_width=0.30,
        box_height=0.40,
        box_depth=0.25,
        # Keep meshes coarse so the topology check runs fast.
        h_throat=0.02,
        h_mouth=0.04,
        h_box=0.06,
        n_axial_slices=8,
        n_circumferential=20,
        corner_resolution=corner_resolution,
        morph=morph,
        **chamfer_kwargs,
        **fillet_kwargs,
    )

    mesh = create_waveguide_on_box_unified(cfg)
    ok, info = _closed(mesh)
    assert ok, info
