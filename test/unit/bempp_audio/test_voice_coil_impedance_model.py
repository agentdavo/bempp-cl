from __future__ import annotations

import numpy as np

from bempp_audio.driver.compression_config import (
    CompressionDriverConfig,
    DriverElectroMechConfig,
    VoiceCoilImpedanceModel,
)
from bempp_audio.driver.network import CompressionDriverNetwork, AcousticMedium


def test_voice_coil_eddy_loss_increases_real_part_with_frequency():
    cfg = CompressionDriverConfig(
        name="vc",
        driver=DriverElectroMechConfig(
            diaphragm_diameter_m=0.025,
            mms_kg=0.01,
            cms_m_per_n=2.0e-4,
            rms_ns_per_m=1.0,
            bl_tm=10.0,
            re_ohm=6.0,
            le_h=0.0001,
            voice_coil_model=VoiceCoilImpedanceModel(kind="EddyLoss", r_eddy_max_ohm=4.0, f_corner_hz=500.0),
        ),
        rear_volume=None,
    )
    net = CompressionDriverNetwork(cfg, medium=AcousticMedium())

    z_low = net._voice_coil_impedance(10.0)
    z_high = net._voice_coil_impedance(20000.0)

    assert np.real(z_high) > np.real(z_low)

