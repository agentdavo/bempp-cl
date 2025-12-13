from __future__ import annotations

import numpy as np

from bempp_audio.mesh.morph import (
    MorphConfig,
    MorphTargetShape,
    morphed_cross_section_xy,
    rounded_rectangle_xy,
    require_simple_polygon_xy,
    theta_with_corner_refinement,
    superellipse_exponent_for_corner_radius,
    superformula_xy,
    superellipse_xy,
)


def test_superellipse_endpoints():
    theta = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    a = 0.2
    b = 0.1
    x, y = superellipse_xy(theta, a, b, n=2.0)
    assert np.isclose(x[0], a)
    assert np.isclose(y[0], 0.0)
    assert np.isclose(x[1], 0.0)
    assert np.isclose(y[1], b)
    assert np.isclose(x[2], -a)
    assert np.isclose(y[3], -b)


def test_corner_radius_mapping_is_monotone():
    a = 0.15
    b = 0.10
    n0 = superellipse_exponent_for_corner_radius(a, b, corner_radius=0.0, n_max=80.0)
    n1 = superellipse_exponent_for_corner_radius(a, b, corner_radius=0.02, n_max=80.0)
    n2 = superellipse_exponent_for_corner_radius(a, b, corner_radius=min(a, b), n_max=80.0)
    assert n0 > n1 > n2
    assert np.isclose(n2, 2.0)


def test_morph_is_circle_at_throat_and_target_at_mouth():
    theta = np.linspace(0.0, 2 * np.pi, 361, endpoint=False)
    throat_r = 0.01
    mouth_r = 0.05

    morph = MorphConfig(
        target_shape=MorphTargetShape.RECTANGLE,
        target_width=0.12,
        target_height=0.06,
        corner_radius=0.005,
        fixed_part=0.0,
        rate=3.0,
        allow_shrinkage=True,
    )

    x0, y0 = morphed_cross_section_xy(theta, t=0.0, radius=throat_r, mouth_radius=mouth_r, morph=morph)
    r0 = np.sqrt(x0**2 + y0**2)
    assert np.allclose(r0, throat_r, atol=1e-10)

    x1, y1 = morphed_cross_section_xy(theta, t=1.0, radius=mouth_r, mouth_radius=mouth_r, morph=morph)
    # Superellipse target should reach ±a, ±b at axes (approximately, by sampling).
    assert np.max(x1) > 0.95 * (morph.target_width / 2)
    assert np.max(y1) > 0.95 * (morph.target_height / 2)


def test_morph_fixed_part_keeps_circle_initially():
    theta = np.linspace(0.0, 2 * np.pi, 181, endpoint=False)
    throat_r = 0.01
    mouth_r = 0.05
    morph = MorphConfig(
        target_shape=MorphTargetShape.RECTANGLE,
        target_width=0.10,
        target_height=0.07,
        corner_radius=0.0,
        fixed_part=0.3,
        rate=2.0,
    )

    x, y = morphed_cross_section_xy(theta, t=0.2, radius=0.02, mouth_radius=mouth_r, morph=morph)
    r = np.sqrt(x**2 + y**2)
    assert np.allclose(r, 0.02, atol=1e-10)


def test_superformula_basic_scaling_extents():
    theta = np.linspace(0.0, 2 * np.pi, 721, endpoint=False)
    # Simple "rounded square-ish" superformula.
    x0, y0 = superformula_xy(theta, a=1.0, b=1.0, m1=4.0, m2=4.0, n1=2.5, n2=2.5, n3=2.5)
    assert np.max(np.abs(x0)) > 0.1
    assert np.max(np.abs(y0)) > 0.1

    morph = MorphConfig.superformula_f([1.0, 1.0, 4.0, 2.5, 2.5, 2.5], width=0.12, aspect_ratio=2.0, allow_shrinkage=True)
    x, y = morphed_cross_section_xy(theta, t=1.0, radius=0.05, mouth_radius=0.05, morph=morph)
    assert abs(float(x.max() - x.min()) - 0.12) < 1e-3
    assert abs(float(y.max() - y.min()) - (0.12 / 2.0)) < 2e-3


def test_require_simple_polygon_detects_bowtie():
    # Self-intersecting "bowtie" quadrilateral.
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    with np.testing.assert_raises(ValueError):
        require_simple_polygon_xy(x, y, name="bowtie")


def test_rounded_rectangle_respects_corner_radius():
    theta = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)
    a = 0.05
    b = 0.03
    rc = 0.01

    x, y = rounded_rectangle_xy(theta, a, b, corner_radius=rc)
    ax = np.abs(x)
    ay = np.abs(y)

    assert np.isclose(ax.max(), a, atol=1e-10)
    assert np.isclose(ay.max(), b, atol=1e-10)

    eps = 5e-10
    on_vertical = np.isclose(ax, a, atol=1e-10)
    assert np.all(ay[on_vertical] <= (b - rc) + eps)

    on_horizontal = np.isclose(ay, b, atol=1e-10)
    assert np.all(ax[on_horizontal] <= (a - rc) + eps)

    corner = ~(on_vertical | on_horizontal)
    cx = a - rc
    cy = b - rc
    dist = np.sqrt((ax[corner] - cx) ** 2 + (ay[corner] - cy) ** 2)
    assert np.allclose(dist, rc, atol=2e-10)


def test_theta_with_corner_refinement_gives_all_corners_samples():
    # Use enough points that each quadrant can allocate an arc-interior sample.
    n_total = 20
    a = 0.05
    b = 0.03
    rc = 0.01
    theta = theta_with_corner_refinement(n_total=n_total, a=a, b=b, rc=rc, corner_points=1)
    assert len(theta) == n_total
    assert np.all(theta >= 0.0)
    assert np.all(theta < 2 * np.pi)

    x, y = rounded_rectangle_xy(theta, a, b, corner_radius=rc)
    ax = np.abs(x)
    ay = np.abs(y)
    on_vertical = np.isclose(ax, a, atol=1e-10)
    on_horizontal = np.isclose(ay, b, atol=1e-10)
    on_corner = ~(on_vertical | on_horizontal)

    sx = np.sign(x)
    sy = np.sign(y)
    assert np.all(sx[on_corner] != 0)
    assert np.all(sy[on_corner] != 0)

    for sxv, syv in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]:
        sel = on_corner & (sx == sxv) & (sy == syv)
        assert np.count_nonzero(sel) >= 1


def test_radial_mode_keeps_profile_radius_before_morph_starts():
    theta = np.linspace(0.0, 2 * np.pi, 256, endpoint=False)
    throat_r = 0.02
    mouth_r = 0.06
    # Choose a slice where the axial profile radius has already expanded.
    r_here = 0.03

    morph = MorphConfig.rectangle(
        width=0.12,
        height=0.06,
        corner_radius=0.006,
        fixed_part=0.5,  # keep circular for the early expansion region
        rate=1.0,
        profile_mode="radial",
        allow_shrinkage=True,
    )
    x, y = morphed_cross_section_xy(
        theta,
        t=0.1,
        radius=r_here,
        throat_radius=throat_r,
        mouth_radius=mouth_r,
        morph=morph,
    )
    r_max = float(np.sqrt(x**2 + y**2).max())
    assert abs(r_max - r_here) < 1e-10
