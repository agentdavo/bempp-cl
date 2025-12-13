"""
Lightweight scorecard computation/export for design iteration.

This is intended to be CI- and sweep-friendly (JSON artifact), separate from
Plotly dashboards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from bempp_audio.viz.data.polar_data import PolarMapData
from bempp_audio.viz.plotly_dashboard import DashboardOptions


def _band_edges() -> list[tuple[float, float, str]]:
    return [
        (1000.0, 2000.0, "1–2 kHz"),
        (2000.0, 5000.0, "2–5 kHz"),
        (5000.0, 10000.0, "5–10 kHz"),
        (10000.0, 20000.0, "10–20 kHz"),
    ]


def _pp_ripple_in_band(freqs: np.ndarray, y: np.ndarray, f_lo: float, f_hi: float) -> float:
    freqs = np.asarray(freqs, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (freqs >= float(f_lo)) & (freqs <= float(f_hi)) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    return float(np.nanmax(y[mask]) - np.nanmin(y[mask]))


def _grade(value: float, *, pass_thr: float, warn_thr: float, higher_is_better: bool = False) -> str:
    if not np.isfinite(value):
        return "N/A"
    if higher_is_better:
        if value >= pass_thr:
            return "PASS"
        if value >= warn_thr:
            return "WARN"
        return "FAIL"
    if value <= pass_thr:
        return "PASS"
    if value <= warn_thr:
        return "WARN"
    return "FAIL"


def _listening_window_spl_db(
    freqs_hz: np.ndarray,
    angles_deg_grid: np.ndarray,
    spl_map_db: np.ndarray,
    angles_deg: tuple[float, ...],
    ref_pa: float = 20e-6,
) -> np.ndarray:
    if spl_map_db.size == 0:
        return np.zeros_like(freqs_hz, dtype=float)

    angles_req = np.asarray([float(a) for a in angles_deg], dtype=float)
    idxs = [int(np.argmin(np.abs(angles_deg_grid - a))) for a in angles_req]
    spl_sel = np.asarray(spl_map_db[idxs, :], dtype=float)  # (n_angles, n_freqs)

    p_rms = float(ref_pa) * (10.0 ** (spl_sel / 20.0))
    p2_mean = np.mean(p_rms * p_rms, axis=0)
    p_rms_mean = np.sqrt(np.maximum(p2_mean, (float(ref_pa) * 1e-10) ** 2))
    return 20.0 * np.log10(p_rms_mean / float(ref_pa))


@dataclass(frozen=True)
class BandScore:
    band: str
    f_lo_hz: float
    f_hi_hz: float
    on_axis_ripple_db: float
    listening_window_ripple_db: float
    di_ripple_db: Optional[float]
    grade_on_axis: str
    grade_listening_window: str
    grade_di: str


@dataclass(frozen=True)
class DesignScorecard:
    title: str
    n_freqs: int
    bands: tuple[BandScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_design_scorecard(
    response: "FrequencyResponse",
    *,
    title: str = "Design Scorecard",
    options: DashboardOptions = DashboardOptions(),
    show_progress: bool = False,
) -> DesignScorecard:
    freqs = np.asarray(response.frequencies, dtype=float)
    polar = PolarMapData.from_response(
        response,
        distance_m=float(options.distance_m),
        max_angle_deg=float(options.max_angle_deg),
        step_deg=float(options.step_deg),
        plane=str(options.plane),
        ref_pa=20e-6,
        show_progress=bool(show_progress),
    )

    curves = polar.spl_curves_db(
        response,
        angles_deg=[0.0],
        show_progress=bool(show_progress),
    )
    on_axis_db = np.asarray(curves[0.0], dtype=float) if 0.0 in curves else np.full_like(freqs, np.nan)

    lw_angles = tuple(
        float(a)
        for a in options.listening_window_angles_deg
        if float(a) <= float(options.max_angle_deg) + 1e-9
    ) or (0.0,)
    lw_db = _listening_window_spl_db(
        freqs_hz=np.asarray(polar.frequencies_hz, dtype=float),
        angles_deg_grid=np.asarray(polar.angles_deg, dtype=float),
        spl_map_db=np.asarray(polar.spl_map_db, dtype=float),
        angles_deg=lw_angles,
        ref_pa=float(polar.ref_pa),
    )

    di_ripple = None
    try:
        f_di, di = response.directivity_index_vs_freq(angle=0.0)
        di_ripple = (np.asarray(f_di, dtype=float), np.asarray(di, dtype=float))
    except Exception:
        di_ripple = None

    bands: list[BandScore] = []
    for f_lo, f_hi, label in _band_edges():
        r0 = _pp_ripple_in_band(freqs, on_axis_db, f_lo, f_hi)
        rlw = _pp_ripple_in_band(freqs, lw_db, f_lo, f_hi)
        rdi = float("nan")
        if di_ripple is not None:
            rdi = _pp_ripple_in_band(di_ripple[0], di_ripple[1], f_lo, f_hi)

        bands.append(
            BandScore(
                band=label,
                f_lo_hz=float(f_lo),
                f_hi_hz=float(f_hi),
                on_axis_ripple_db=float(r0),
                listening_window_ripple_db=float(rlw),
                di_ripple_db=float(rdi) if np.isfinite(rdi) else None,
                grade_on_axis=_grade(
                    float(r0),
                    pass_thr=float(options.spl_ripple_pass_db),
                    warn_thr=float(options.spl_ripple_warn_db),
                ),
                grade_listening_window=_grade(
                    float(rlw),
                    pass_thr=float(options.spl_ripple_pass_db),
                    warn_thr=float(options.spl_ripple_warn_db),
                ),
                grade_di=_grade(
                    float(rdi),
                    pass_thr=float(options.di_ripple_pass_db),
                    warn_thr=float(options.di_ripple_warn_db),
                )
                if np.isfinite(rdi)
                else "N/A",
            )
        )

    return DesignScorecard(title=str(title), n_freqs=int(freqs.size), bands=tuple(bands))


def save_scorecard_json(scorecard: DesignScorecard, filename: str | Path) -> str:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(scorecard.to_dict(), indent=2) + "\n")
    return str(path)

