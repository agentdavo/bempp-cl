from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest

from bempp_audio._optional import optional_import


def _file_contains(path: str, needle: bytes) -> bool:
    needle = bytes(needle)
    if not needle:
        return True
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                return False
            if needle in chunk:
                return True


@dataclass(frozen=True)
class _FakePolarSweep:
    freqs: np.ndarray
    angles: np.ndarray

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

    @property
    def driver_impedance(self):
        return None

    @property
    def driver_excursion_mm(self):
        return None

    def polar_sweep(self, *, angles_deg, distance_m: float = 1.0, plane: str = "horizontal", show_progress: bool = False):
        return _FakePolarSweep(self._freqs, np.asarray(angles_deg, dtype=float))

    def directivity_index_vs_freq(self, angle: float = 0.0):
        return self._freqs, np.zeros_like(self._freqs)

    def beamwidth_vs_freq(self, level_db: float = -6.0, plane: str = "xz"):
        return self._freqs, np.ones_like(self._freqs) * 60.0


def test_save_driver_dashboard_html_smoke_without_bem(tmp_path):
    from bempp_audio.viz import save_driver_dashboard_html

    out = tmp_path / "dashboard.html"
    save_driver_dashboard_html(_FakeResponse(), filename=str(out), title="offline smoke")

    assert out.exists()
    assert not _file_contains(str(out), b'src="https://cdn.plot.ly/')
    assert _file_contains(str(out), b"Plotly.newPlot")
    assert _file_contains(str(out), b"SPL 0")


@pytest.mark.skipif(os.environ.get("BEMPPAUDIO_RUN_SLOW") != "1", reason="set BEMPPAUDIO_RUN_SLOW=1 to enable")
def test_save_driver_dashboard_html_smoke_real_bem(tmp_path):
    gmsh, has_gmsh = optional_import("gmsh")
    bempp, has_bempp = optional_import("bempp_cl.api")
    if not has_gmsh or not has_bempp:
        pytest.skip("requires gmsh and bempp_cl")

    os.environ.setdefault("BEMPP_DEVICE_INTERFACE", "numba")

    from bempp_audio import Loudspeaker
    from bempp_audio.viz import save_driver_dashboard_html

    response = (
        Loudspeaker()
        .circular_piston(radius=0.02, element_size=0.01)
        .free_space()
        .single_frequency(1000.0)
        .solve(n_workers=1)
    )
    assert len(response.results) == 1

    out = tmp_path / "bem_dashboard.html"
    save_driver_dashboard_html(response, filename=str(out), title="real bem smoke", max_angle=60.0, normalize_angle=0.0, show_progress=False)

    assert out.exists()
    assert not _file_contains(str(out), b'src="https://cdn.plot.ly/')
    assert _file_contains(str(out), b"Plotly.newPlot")
    assert _file_contains(str(out), b"SPL 0")

