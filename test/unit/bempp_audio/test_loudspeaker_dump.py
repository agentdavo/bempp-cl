from __future__ import annotations

import json

from bempp_audio import Loudspeaker, VelocityProfile


def test_loudspeaker_to_dict_is_json_serializable():
    speaker = (
        Loudspeaker()
        .frequency_range(100.0, 1000.0, num=3, spacing="linear")
        .infinite_baffle()
        .velocity_profile(VelocityProfile.piston(amplitude=0.01))
    )

    data = speaker.to_dict()
    assert data["baffle"]["type"] == "InfiniteBaffle"
    assert data["frequencies"] == [100.0, 550.0, 1000.0]
    assert data["mesh"] is None

    # Should round-trip through JSON without needing any custom encoders.
    json.loads(json.dumps(data))

