import numpy as np

from bempp_audio.mesh.morph import MorphConfig, MorphTargetShape, morphed_cross_section_xy


def _wrap_angle(diff: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(diff, dtype=float)))


def test_blend_ellipse_is_ray_aligned_at_mouth() -> None:
    theta = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)

    morph = MorphConfig(
        target_shape=MorphTargetShape.ELLIPSE,
        target_width=0.20,
        target_height=0.10,
        fixed_part=0.0,
        end_part=1.0,
        rate=1.0,
        allow_shrinkage=True,
        profile_mode="blend",
    )

    x, y = morphed_cross_section_xy(
        theta,
        t=1.0,
        radius=0.10,
        mouth_radius=0.10,
        morph=morph,
    )
    ang = np.arctan2(y, x)
    err = _wrap_angle(ang - theta)
    assert float(np.max(np.abs(err))) < 1e-10


def test_blend_superellipse_is_ray_aligned_at_mouth() -> None:
    theta = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)

    morph = MorphConfig(
        target_shape=MorphTargetShape.SUPERELLIPSE,
        target_width=0.20,
        target_height=0.10,
        superellipse_n=6.0,
        fixed_part=0.0,
        end_part=1.0,
        rate=1.0,
        allow_shrinkage=True,
        profile_mode="blend",
    )

    x, y = morphed_cross_section_xy(
        theta,
        t=1.0,
        radius=0.10,
        mouth_radius=0.10,
        morph=morph,
    )
    ang = np.arctan2(y, x)
    err = _wrap_angle(ang - theta)
    assert float(np.max(np.abs(err))) < 1e-10

