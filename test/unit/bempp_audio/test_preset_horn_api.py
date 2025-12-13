import pytest


def test_preset_horn_accepts_kwargs():
    from bempp_audio import Loudspeaker

    spk = Loudspeaker().preset_horn(distance=2.0, max_angle=60.0, resolution_deg=10.0, normalize_angle=0.0)
    assert spk.state.measurement_distance == pytest.approx(2.0)

