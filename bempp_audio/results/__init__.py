"""Result containers and post-processing for acoustic simulations."""

from bempp_audio.results.radiation_result import RadiationResult
from bempp_audio.results.pressure_field import PressureField
from bempp_audio.results.directivity import (
    DirectivityPattern,
    DirectivityBalloon,
    DirectivitySweepMetrics,
    compute_directivity_sweep_metrics,
)
from bempp_audio.results.objectives import (
    BeamwidthTarget,
    DirectivityObjectiveConfig,
    DirectivityObjectiveResult,
    evaluate_directivity_objective,
    evaluate_directivity_objective_from_metrics,
)
from bempp_audio.results.frequency_response import FrequencyResponse
from bempp_audio.results.impedance import RadiationImpedance
from bempp_audio.results.polar import PolarSweep, compute_polar_sweep

__all__ = [
    "RadiationResult",
    "PressureField",
    "DirectivityPattern",
    "DirectivityBalloon",
    "DirectivitySweepMetrics",
    "compute_directivity_sweep_metrics",
    "BeamwidthTarget",
    "DirectivityObjectiveConfig",
    "DirectivityObjectiveResult",
    "evaluate_directivity_objective",
    "evaluate_directivity_objective_from_metrics",
    "FrequencyResponse",
    "RadiationImpedance",
    "PolarSweep",
    "compute_polar_sweep",
]
