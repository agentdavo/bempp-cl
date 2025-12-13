from __future__ import annotations

from bempp_audio.config import (
    SimulationConfig,
    FrequencyConfig,
    DirectivityConfig,
    MediumConfig,
    SolverConfig,
    ExecutionConfig,
)


def test_simulation_config_validate_ok():
    cfg = SimulationConfig(
        frequency=FrequencyConfig(f_start=200.0, f_end=20000.0, num_points=5, spacing="log"),
        directivity=DirectivityConfig(polar_start=0.0, polar_end=180.0, polar_num=5, measurement_distance=1.0),
        medium=MediumConfig(c=343.0, rho=1.225),
        solver=SolverConfig(tol=1e-5, maxiter=50, space_type="DP", space_order=0),
        execution=ExecutionConfig(n_workers=None),
    )
    assert cfg.validate() == []


def test_frequency_config_octave_spacing_to_array():
    cfg = FrequencyConfig(f_start=100.0, f_end=800.0, num_points=2, spacing="octave")
    freqs = cfg.to_array()
    # 100->800 is 3 octaves; 2 points per octave => 7 points including endpoints.
    assert len(freqs) == 7
    assert freqs[0] == 100.0
    assert freqs[-1] == 800.0


def test_simulation_config_validate_errors():
    cfg = SimulationConfig(
        frequency=FrequencyConfig(f_start=-1.0, f_end=0.0, num_points=0, spacing="bad"),
        directivity=DirectivityConfig(polar_start=10.0, polar_end=5.0, polar_num=1, measurement_distance=-1.0),
        medium=MediumConfig(c=-1.0, rho=-1.0, temperature=-400.0),
        solver=SolverConfig(tol=2.0, maxiter=0, space_type="BAD", space_order=-1),
        execution=ExecutionConfig(n_workers=0),
    )
    errors = cfg.validate()
    assert errors
