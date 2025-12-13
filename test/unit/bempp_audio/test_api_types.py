from __future__ import annotations

import numpy as np


def test_frequency_spacing_enum_supported():
    from bempp_audio import Loudspeaker
    from bempp_audio.api.types import FrequencySpacing

    spk = Loudspeaker().frequency_range(100.0, 200.0, num=3, spacing=FrequencySpacing.LINEAR)
    assert spk.state.frequencies is not None
    assert np.allclose(spk.state.frequencies, np.linspace(100.0, 200.0, 3))


def test_frequency_spacing_octave_supported():
    from bempp_audio import Loudspeaker
    from bempp_audio.api.types import FrequencySpacing

    spk = Loudspeaker().frequency_range(100.0, 800.0, num=6, spacing=FrequencySpacing.OCTAVE)
    freqs = spk.state.frequencies
    assert freqs is not None
    assert len(freqs) == 19  # 3 octaves * 6 ppo + 1 endpoint
    assert np.isclose(freqs[0], 100.0)
    assert np.isclose(freqs[-1], 800.0)


def test_default_bc_policy_enum_supported():
    from bempp_audio import Loudspeaker
    from bempp_audio.api.types import BoundaryConditionPolicy

    spk = Loudspeaker().default_bc(BoundaryConditionPolicy.ERROR)
    assert spk.state.default_bc_policy == "error"


def test_velocity_mode_enum_supported():
    from bempp_audio import Loudspeaker
    from bempp_audio.api.types import VelocityMode

    spk = Loudspeaker().velocity(mode=VelocityMode.ZERO)
    assert spk.state.velocity is not None
    assert spk.state.velocity.to_dict()["type"] == "zero"
