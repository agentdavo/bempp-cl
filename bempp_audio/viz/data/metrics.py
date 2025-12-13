from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bempp_audio.results import FrequencyResponse
    from bempp_audio.results.frequency_response import Plane2DLike


def directivity_index_vs_frequency(
    response: "FrequencyResponse",
    *,
    angle_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Directivity index vs frequency.

    This is a lightweight computation helper for visualization, delegating to the
    canonical implementation on `FrequencyResponse`.
    """
    return response.directivity_index_vs_freq(angle=float(angle_deg))


def beamwidth_vs_frequency(
    response: "FrequencyResponse",
    *,
    level_db: float = -6.0,
    plane: "Plane2DLike" = "xz",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Beamwidth vs frequency.

    This is a lightweight computation helper for visualization, delegating to the
    canonical implementation on `FrequencyResponse`.
    """
    return response.beamwidth_vs_freq(level_db=float(level_db), plane=plane)

