from __future__ import annotations


def test_radial_profile_mode_is_monotone_in_support_directions():
    from bempp_audio.mesh.waveguide import WaveguideMeshConfig
    from bempp_audio.mesh.waveguide_profile import check_profile
    from bempp_audio.mesh.morph import MorphConfig

    cfg = WaveguideMeshConfig(
        throat_diameter=0.0254,
        mouth_diameter=0.300,
        length=0.100,
        profile_type="cts",
        cts_throat_blend=0.15,
        cts_transition=0.75,
        cts_driver_exit_angle_deg=10.0,
        cts_tangency=1.0,
        morph=MorphConfig.rectangle(
            width=0.300,
            height=0.100,
            corner_radius=0.006,
            allow_shrinkage=False,
            profile_mode="radial",
        ),
    )

    rep = check_profile(cfg, n_axial=160, n_directions=36, tol=0.0)
    assert rep.ok
