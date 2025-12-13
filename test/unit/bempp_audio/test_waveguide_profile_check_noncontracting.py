from __future__ import annotations

import numpy as np


def test_check_profile_with_noncontracting_morph_passes_for_strong_aspect_ratio():
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
            allow_shrinkage=True,
            enforce_noncontracting=True,
            enforce_mode="directions",
            enforce_n_directions=24,
        ),
    )

    rep = check_profile(cfg, n_axial=120, n_directions=24, tol=0.0)
    assert rep.ok
    assert rep.max_scale_y is not None and rep.max_scale_y >= 1.0
