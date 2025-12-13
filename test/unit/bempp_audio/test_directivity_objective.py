import numpy as np


def test_directivity_objective_constant_targets_smoke():
    from bempp_audio.results.directivity import DirectivitySweepMetrics
    from bempp_audio.results.objectives import BeamwidthTarget, DirectivityObjectiveConfig, evaluate_directivity_objective_from_metrics

    freqs = np.array([1000.0, 2000.0, 4000.0, 8000.0, 16000.0], dtype=float)
    bw_h = np.array([90.0, 90.0, 90.0, 90.0, 90.0], dtype=float)
    bw_v = np.array([60.0, 60.0, 60.0, 60.0, 60.0], dtype=float)
    di_h = np.array([10.0, 10.2, 10.1, 10.0, 10.1], dtype=float)
    di_v = np.array([8.0, 8.1, 8.2, 8.1, 8.0], dtype=float)

    m_h = DirectivitySweepMetrics(freqs, di_h, bw_h)
    m_v = DirectivitySweepMetrics(freqs, di_v, bw_v)
    cfg = DirectivityObjectiveConfig(
        f_lo_hz=1000.0,
        f_hi_hz=16000.0,
        target_h=BeamwidthTarget(kind="constant", value_deg=90.0),
        target_v=BeamwidthTarget(kind="constant", value_deg=60.0),
        di_ripple_target_db=2.0,
        monotonicity_target=0.8,
    )
    out = evaluate_directivity_objective_from_metrics(m_h, m_v, cfg=cfg)
    assert out.J >= 0.0
    assert out.bw_rmse_h_deg == 0.0
    assert out.bw_rmse_v_deg == 0.0


def test_directivity_objective_proxy_di_mode_smoke():
    from bempp_audio.results.objectives import DirectivityObjectiveConfig, evaluate_directivity_objective

    class _FakeResult:
        def __init__(self, f: float) -> None:
            self.frequency = float(f)
            self.baffle = None

        def directivity(self):
            raise RuntimeError("not used")

    # This test only checks the code path wiring; DirectivityPattern requires bempp for real evaluation.
    # So we just validate the config accepts di_mode="proxy".
    cfg = DirectivityObjectiveConfig(di_mode="proxy")
    assert cfg.di_mode == "proxy"
