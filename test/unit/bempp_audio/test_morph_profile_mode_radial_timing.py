from __future__ import annotations

import numpy as np


def test_radial_profile_mode_uses_linear_blend_by_default():
    from bempp_audio.mesh.morph import MorphConfig, morphed_cross_section_xy

    theta = np.array([0.0])
    throat_r = 0.01
    mouth_radius = 0.05
    r_mid = throat_r + 0.5 * (mouth_radius - throat_r)  # s=0.5

    morph = MorphConfig.rectangle(
        width=0.30,
        height=0.10,
        corner_radius=0.006,
        profile_mode="radial",
    )

    x, y = morphed_cross_section_xy(
        theta,
        t=0.5,
        radius=r_mid,
        throat_radius=throat_r,
        mouth_radius=mouth_radius,
        morph=morph,
    )

    # In radial mode, the target shape is scaled to the current profile radius.
    a_mouth = 0.30 / 2.0  # along theta=0, the mouth support is the half-width
    scale = r_mid / mouth_radius
    a_here = a_mouth * scale
    expected = 0.5 * r_mid + 0.5 * a_here
    assert np.isclose(float(x[0]), expected, rtol=0, atol=1e-12)
    assert np.isclose(float(y[0]), 0.0, rtol=0, atol=1e-12)


def test_radial_profile_mode_applies_timing_overrides_when_set():
    from bempp_audio.mesh.morph import MorphConfig, morphed_cross_section_xy

    theta = np.array([0.0])
    throat_r = 0.01
    mouth_radius = 0.05
    r_mid = throat_r + 0.5 * (mouth_radius - throat_r)  # s=0.5

    morph = MorphConfig.rectangle(
        width=0.30,
        height=0.10,
        corner_radius=0.006,
        profile_mode="radial",
        rate=2.0,  # u = s^2
    )

    x, y = morphed_cross_section_xy(
        theta,
        t=0.5,
        radius=r_mid,
        throat_radius=throat_r,
        mouth_radius=mouth_radius,
        morph=morph,
    )

    a_mouth = 0.30 / 2.0
    scale = r_mid / mouth_radius
    a_here = a_mouth * scale
    u = 0.5**2
    expected = (1.0 - u) * r_mid + u * a_here
    assert np.isclose(float(x[0]), expected, rtol=0, atol=1e-12)
    assert np.isclose(float(y[0]), 0.0, rtol=0, atol=1e-12)
