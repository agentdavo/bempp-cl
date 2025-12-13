from __future__ import annotations

import numpy as np


def test_polar_directions_horizontal_plane_matches_xz_cut():
    from bempp_audio.results.polar import polar_directions

    dirs = polar_directions(np.array([0.0, 0.0, 1.0]), np.array([0.0, 90.0]), plane="horizontal")
    assert dirs.shape == (3, 2)
    assert np.allclose(dirs[:, 0], np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert np.allclose(dirs[:, 1], np.array([1.0, 0.0, 0.0]), atol=1e-12)


def test_polar_directions_vertical_plane_matches_yz_cut():
    from bempp_audio.results.polar import polar_directions

    dirs = polar_directions(np.array([0.0, 0.0, 1.0]), np.array([0.0, 90.0]), plane="vertical")
    assert dirs.shape == (3, 2)
    assert np.allclose(dirs[:, 0], np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert np.allclose(dirs[:, 1], np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_directivity_index_solid_angle_controls_hemisphere_convention():
    from bempp_audio.results.directivity import DirectivityBalloon

    theta_1d = np.linspace(0.0, np.pi, 361)
    phi_1d = np.linspace(0.0, 2.0 * np.pi, 721)
    theta, phi = np.meshgrid(theta_1d, phi_1d, indexing="ij")

    # Unit pattern on the front hemisphere (theta <= pi/2), zero elsewhere.
    pattern = np.where(theta <= (0.5 * np.pi), 1.0 + 0j, 0.0 + 0j)
    balloon = DirectivityBalloon(theta=theta, phi=phi, pattern=pattern, frequency=1000.0)

    di_hemisphere = balloon.directivity_index(solid_angle=2.0 * np.pi)
    di_fullspace = balloon.directivity_index(solid_angle=4.0 * np.pi)

    # For a constant-intensity hemisphere, hemispherical DI should be ~0 dB.
    assert abs(di_hemisphere) < 0.1
    # Full-space DI is +3 dB higher (since average is diluted by the silent rear hemisphere).
    assert abs(di_fullspace - 10.0 * np.log10(2.0)) < 0.1

