from __future__ import annotations


def test_loudspeaker_clone_is_independent():
    from bempp_audio import Loudspeaker

    spk = Loudspeaker().measurement_distance(1.0).frequency_range(200.0, 20000.0, num=5)
    clone = spk.clone()

    assert clone is not spk
    assert clone.state == spk.state

    clone.measurement_distance(2.0)
    assert spk.state.measurement_distance == 1.0
    assert clone.state.measurement_distance == 2.0

