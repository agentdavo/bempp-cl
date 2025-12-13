import numpy as np


def test_mouth_diffraction_proxy_conical_has_infinite_curvature_radius():
    from bempp_audio.mesh import WaveguideMeshConfig, mouth_diffraction_proxy

    cfg = WaveguideMeshConfig(
        throat_diameter=0.025,
        mouth_diameter=0.200,
        length=0.100,
        profile_type="conical",
    )
    rep = mouth_diffraction_proxy(cfg, n_axial=200, fit_points=9)

    assert rep.mouth_angle_deg > 0.0
    assert rep.tangency_error_deg > 0.0
    # Linear profiles have zero curvature; numeric fitting may return a very large
    # (but finite) radius.
    assert rep.curvature_radius_m > 1e3


def test_mouth_diffraction_proxy_cts_tangency_reduces_tangency_error():
    from bempp_audio.mesh import WaveguideMeshConfig, mouth_diffraction_proxy

    base = dict(
        throat_diameter=0.025,
        mouth_diameter=0.300,
        length=0.100,
        profile_type="cts",
        cts_throat_blend=0.15,
        cts_transition=0.75,
        cts_driver_exit_angle_deg=10.0,
    )

    rep0 = mouth_diffraction_proxy(WaveguideMeshConfig(**base, cts_tangency=0.0), n_axial=300, fit_points=11)
    rep1 = mouth_diffraction_proxy(WaveguideMeshConfig(**base, cts_tangency=1.0), n_axial=300, fit_points=11)

    assert rep1.tangency_error_deg <= rep0.tangency_error_deg
