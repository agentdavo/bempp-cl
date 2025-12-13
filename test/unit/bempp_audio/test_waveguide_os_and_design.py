from __future__ import annotations

import numpy as np


def test_os_profile_endpoints_and_monotone():
    from bempp_audio.mesh.profiles import os_profile

    throat_r = 0.0127
    mouth_r = 0.150
    L = 0.10
    x = np.linspace(0.0, L, 401)

    # Must be >= the conical half-angle implied by the endpoints (here ~54°),
    # and within the feasible range for this OS parameterization.
    r = os_profile(x, throat_r, mouth_r, L, opening_angle_deg=56.0)
    assert np.isclose(r[0], throat_r, rtol=0, atol=1e-10)
    assert np.isclose(r[-1], mouth_r, rtol=0, atol=1e-10)
    assert np.all(np.diff(r) >= -1e-12)


def test_os_profile_default_angle_matches_endpoints():
    from bempp_audio.mesh.profiles import os_profile

    throat_r = 0.01
    mouth_r = 0.05
    L = 0.10
    x = np.linspace(0.0, L, 51)
    r = os_profile(x, throat_r, mouth_r, L)
    assert np.isclose(r[0], throat_r, atol=1e-10)
    assert np.isclose(r[-1], mouth_r, atol=1e-10)


def test_design_helpers_roundtrip_conical_angle():
    from bempp_audio.mesh.design import conical_half_angle_deg, mouth_diameter_from_half_angle

    throat_d = 0.0254
    L = 0.10
    half_angle = 30.0

    mouth_d = mouth_diameter_from_half_angle(throat_diameter=throat_d, length=L, half_angle_deg=half_angle)
    inferred = conical_half_angle_deg(
        throat_radius=throat_d / 2.0,
        mouth_radius=mouth_d / 2.0,
        length=L,
    )
    assert abs(inferred - half_angle) < 1e-10


def test_design_helpers_rectangular_mouth_sizes():
    from bempp_audio.mesh.design import rectangular_mouth_from_half_angles

    throat_d = 0.0254
    L = 0.10
    w, h = rectangular_mouth_from_half_angles(throat_diameter=throat_d, length=L, half_angle_h_deg=40.0, half_angle_v_deg=20.0)
    assert w > h
    assert w > throat_d
    assert h > throat_d


def test_os_opening_angle_bounds_are_consistent():
    from bempp_audio.mesh.design import os_opening_angle_bounds_deg

    throat_r = 0.0127
    mouth_r = 0.150
    L = 0.10
    amin, amax = os_opening_angle_bounds_deg(throat_radius=throat_r, mouth_radius=mouth_r, length=L)
    assert amin > 0
    assert amax > amin
