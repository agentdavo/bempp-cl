#!/usr/bin/env python3
"""
CI-friendly visualization smoke test.

Default mode (no BEM deps):
  - Builds a fake `FrequencyResponse`-like object
  - Writes `logs/viz_smoke_dashboard.html`
  - Verifies the HTML contains Plotly traces

Optional real-BEM mode:
  - Requires `gmsh` and `bempp_cl.api`
  - Runs a tiny piston solve and writes `logs/viz_smoke_bem_dashboard.html`
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _file_contains(path: str, needle: bytes) -> bool:
    needle = bytes(needle)
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


def _write_fake_dashboard(*, out_dir: Path) -> Path:
    from bempp_audio.viz import save_driver_dashboard_html

    out = out_dir / "viz_smoke_dashboard.html"
    save_driver_dashboard_html(_FakeResponse(), filename=str(out), title="Viz Smoke — Fake Response", show_progress=False)
    return out


def _write_real_bem_dashboard(*, out_dir: Path) -> Path:
    from bempp_audio._optional import optional_import

    gmsh, has_gmsh = optional_import("gmsh")
    bempp, has_bempp = optional_import("bempp_cl.api")
    if not has_gmsh or not has_bempp:
        raise RuntimeError("Real-BEM smoke requires `gmsh` and `bempp_cl.api` to be importable.")

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
    out = out_dir / "viz_smoke_bem_dashboard.html"
    save_driver_dashboard_html(response, filename=str(out), title="Viz Smoke — Real BEM", max_angle=60.0, normalize_angle=0.0, show_progress=False)
    return out


def main() -> int:
    _ensure_repo_on_path()
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.environ.get("BEMPPAUDIO_OUT_DIR", "logs"), help="Output directory (default: logs)")
    p.add_argument("--real-bem", action="store_true", help="Also run a tiny real BEM solve (requires gmsh+bempp_cl)")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out = _write_fake_dashboard(out_dir=out_dir)
    ok = _file_contains(str(out), b"Plotly.newPlot") and _file_contains(str(out), b"SPL 0")
    print(f"Wrote: {out}  ok={ok}")
    if not ok:
        return 2

    if args.real_bem:
        out2 = _write_real_bem_dashboard(out_dir=out_dir)
        ok2 = _file_contains(str(out2), b"Plotly.newPlot") and _file_contains(str(out2), b"SPL 0")
        print(f"Wrote: {out2}  ok={ok2}")
        if not ok2:
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
