from __future__ import annotations

import numpy as np

from bempp_audio.velocity import VelocityProfile


def test_velocity_profile_roundtrip_piston():
    prof = VelocityProfile.piston(amplitude=0.123 + 0.1j, phase=0.0)
    data = prof.to_dict()
    prof2 = VelocityProfile.from_dict(data)
    assert prof2.to_dict() == data


def test_velocity_profile_roundtrip_gaussian():
    prof = VelocityProfile.gaussian(center=np.array([0.0, 0.0, 0.0]), width=0.01, amplitude=1.0, phase=0.2)
    data = prof.to_dict()
    prof2 = VelocityProfile.from_dict(data)
    assert prof2.to_dict() == data

