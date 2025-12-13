import numpy as np


class _FakePolarSweep:
    def __init__(self, freqs: np.ndarray, angles: np.ndarray) -> None:
        self.freqs = freqs
        self.angles = angles

    def spl_db(self, ref_pa: float = 20e-6) -> np.ndarray:
        ff, aa = np.meshgrid(self.freqs, self.angles, indexing="ij")
        pressure = (1.0 / (1.0 + ff / 1000.0)) * (1.0 + 0.2 * np.cos(np.deg2rad(aa)))
        p_rms = np.abs(pressure) / np.sqrt(2.0)
        return 20.0 * np.log10(np.maximum(p_rms, 1e-20) / float(ref_pa))


class _FakeResponse:
    def __init__(self) -> None:
        self._freqs = np.array([500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0], dtype=float)

    @property
    def frequencies(self) -> np.ndarray:
        return self._freqs

    @property
    def results(self) -> list[object]:
        return []

    def polar_sweep(self, *, angles_deg, distance_m: float = 1.0, plane: str = "horizontal", show_progress: bool = False):
        return _FakePolarSweep(self._freqs, np.asarray(angles_deg, dtype=float))

    def directivity_index_vs_freq(self, angle: float = 0.0):
        return self._freqs, np.zeros_like(self._freqs)


def test_compute_design_scorecard_smoke(tmp_path):
    from bempp_audio.viz import compute_design_scorecard, save_scorecard_json

    sc = compute_design_scorecard(_FakeResponse(), show_progress=False)
    assert sc.n_freqs == 6
    assert len(sc.bands) == 4

    out = tmp_path / "scorecard.json"
    save_scorecard_json(sc, out)
    assert out.exists()

