"""
Objective functions for waveguide/directivity design.

The intent is to provide a single scalar objective J that can be minimized in
sweeps/optimizers, built from directivity metrics in horizontal/vertical planes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from bempp_audio.api.types import Plane2DLike, normalize_plane_2d
from bempp_audio.results.directivity import DirectivitySweepMetrics, DirectivityPattern
from bempp_audio.baffles import InfiniteBaffle


@dataclass(frozen=True)
class BeamwidthTarget:
    """
    Beamwidth target model (degrees).

    Supported kinds:
    - constant: target is a fixed number (deg)
    - piecewise_log: interpolate targets in log-frequency space
    """

    kind: Literal["constant", "piecewise_log"] = "constant"
    value_deg: float = 90.0
    freqs_hz: tuple[float, ...] = ()
    values_deg: tuple[float, ...] = ()

    def at(self, freqs_hz: np.ndarray) -> np.ndarray:
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        if self.kind == "constant":
            return np.full_like(freqs_hz, float(self.value_deg), dtype=float)
        if self.kind == "piecewise_log":
            if len(self.freqs_hz) < 2 or len(self.freqs_hz) != len(self.values_deg):
                raise ValueError("piecewise_log requires freqs_hz and values_deg of equal length >= 2")
            xf = np.asarray(self.freqs_hz, dtype=float)
            yf = np.asarray(self.values_deg, dtype=float)
            if np.any(xf <= 0):
                raise ValueError("freqs_hz must be > 0")
            xq = np.log(np.maximum(freqs_hz, 1e-30))
            return np.interp(xq, np.log(xf), yf)
        raise ValueError("Unknown BeamwidthTarget.kind")


@dataclass(frozen=True)
class DirectivityObjectiveConfig:
    """
    Weights/thresholds for a directivity-driven objective.

    The objective is intentionally simple and robust:
    - match beamwidth to target curves (dominant term)
    - penalize DI ripple (smoothness)
    - penalize non-monotone beamwidth vs frequency
    """

    beamwidth_level_db: float = -6.0
    f_lo_hz: float = 1000.0
    f_hi_hz: float = 16000.0

    # Targets (can be distinct horizontal vs vertical).
    target_h: BeamwidthTarget = BeamwidthTarget(kind="constant", value_deg=90.0)
    target_v: BeamwidthTarget = BeamwidthTarget(kind="constant", value_deg=90.0)

    # Weights
    w_bw: float = 1.0
    w_di_ripple: float = 0.25
    w_mono: float = 0.25

    # Soft thresholds
    di_ripple_target_db: float = 2.0
    monotonicity_target: float = 0.8

    # Normalization
    bw_scale_deg: float = 30.0  # error of 30° -> ~1.0 in normalized units

    # DI evaluation strategy:
    # - "proxy": approximate DI from horizontal/vertical beamwidths (fast, robust for objectives)
    # - "balloon": true DI via 3D balloon integration (slow; use for final validation)
    di_mode: Literal["proxy", "balloon"] = "proxy"


@dataclass(frozen=True)
class DirectivityObjectiveResult:
    J: float
    bw_rmse_h_deg: float
    bw_rmse_v_deg: float
    di_ripple_h_db: float
    di_ripple_v_db: float
    mono_h: float
    mono_v: float


def _band_mask(freqs_hz: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    return (freqs_hz >= float(f_lo)) & (freqs_hz <= float(f_hi)) & np.isfinite(freqs_hz)


def evaluate_directivity_objective_from_metrics(
    metrics_h: DirectivitySweepMetrics,
    metrics_v: DirectivitySweepMetrics,
    *,
    cfg: DirectivityObjectiveConfig = DirectivityObjectiveConfig(),
) -> DirectivityObjectiveResult:
    """
    Evaluate a scalar objective using already-computed metrics.
    """
    fh = np.asarray(metrics_h.frequencies, dtype=float)
    fv = np.asarray(metrics_v.frequencies, dtype=float)
    if fh.size == 0 or fv.size == 0:
        raise ValueError("metrics must have non-empty frequencies")

    # Intersect on frequency grid by interpolating targets and the opposite plane.
    # Use horizontal grid as reference.
    mask = _band_mask(fh, cfg.f_lo_hz, cfg.f_hi_hz)
    if int(np.count_nonzero(mask)) < 3:
        mask = np.isfinite(fh)
    f = fh[mask]

    bw_h = np.asarray(metrics_h.beamwidth_values, dtype=float)[mask]
    bw_v = np.interp(f, fv, np.asarray(metrics_v.beamwidth_values, dtype=float))

    tgt_h = cfg.target_h.at(f)
    tgt_v = cfg.target_v.at(f)

    err_h = bw_h - tgt_h
    err_v = bw_v - tgt_v
    bw_rmse_h = float(np.sqrt(np.mean(err_h * err_h))) if f.size else float("nan")
    bw_rmse_v = float(np.sqrt(np.mean(err_v * err_v))) if f.size else float("nan")

    # Smoothness penalties.
    di_ripple_h = float(metrics_h.di_ripple_db)
    di_ripple_v = float(metrics_v.di_ripple_db)
    mono_h = float(metrics_h.beamwidth_monotonicity)
    mono_v = float(metrics_v.beamwidth_monotonicity)

    bw_term = (bw_rmse_h + bw_rmse_v) / max(1e-9, float(cfg.bw_scale_deg))
    di_term = max(0.0, di_ripple_h - cfg.di_ripple_target_db) + max(0.0, di_ripple_v - cfg.di_ripple_target_db)
    mono_term = max(0.0, cfg.monotonicity_target - mono_h) + max(0.0, cfg.monotonicity_target - mono_v)

    J = float(cfg.w_bw) * float(bw_term) + float(cfg.w_di_ripple) * float(di_term) + float(cfg.w_mono) * float(mono_term)

    return DirectivityObjectiveResult(
        J=J,
        bw_rmse_h_deg=bw_rmse_h,
        bw_rmse_v_deg=bw_rmse_v,
        di_ripple_h_db=di_ripple_h,
        di_ripple_v_db=di_ripple_v,
        mono_h=mono_h,
        mono_v=mono_v,
    )


def evaluate_directivity_objective(
    results: Sequence[object],
    *,
    cfg: DirectivityObjectiveConfig = DirectivityObjectiveConfig(),
    plane_h: Plane2DLike = "xz",
    plane_v: Plane2DLike = "yz",
) -> DirectivityObjectiveResult:
    """
    Compute directivity metrics in 2 planes and evaluate the objective.
    """
    plane_h = normalize_plane_2d(plane_h)
    plane_v = normalize_plane_2d(plane_v)

    # Sort by frequency once for consistent sweeps.
    sorted_results = sorted(list(results), key=lambda r: float(getattr(r, "frequency")))
    freqs = np.asarray([float(r.frequency) for r in sorted_results], dtype=float)

    bw_h = np.zeros_like(freqs)
    bw_v = np.zeros_like(freqs)
    for i, r in enumerate(sorted_results):
        dp = DirectivityPattern(r)
        bw_h[i] = dp.beamwidth(level_db=float(cfg.beamwidth_level_db), plane=plane_h)
        bw_v[i] = dp.beamwidth(level_db=float(cfg.beamwidth_level_db), plane=plane_v)

    # DI values: either true balloon DI (slow) or beamwidth-derived proxy.
    if str(cfg.di_mode) == "balloon":
        di_h = np.zeros_like(freqs)
        di_v = np.zeros_like(freqs)
        for i, r in enumerate(sorted_results):
            di = DirectivityPattern(r).directivity_index()
            di_h[i] = di
            di_v[i] = di
    else:
        # DI proxy from beamwidths (assumes separable-ish main lobe).
        # Approximate solid angle of a rectangular patch on the sphere:
        # Ω ≈ 4 * sin(bw_h/2) * sin(bw_v/2).
        # Use 2π reference in infinite-baffle convention, else 4π.
        solid = 2.0 * np.pi if isinstance(getattr(sorted_results[0], "baffle", None), InfiniteBaffle) else 4.0 * np.pi
        ah = 0.5 * np.deg2rad(np.clip(bw_h, 1e-6, 360.0))
        av = 0.5 * np.deg2rad(np.clip(bw_v, 1e-6, 360.0))
        omega = 4.0 * np.sin(ah) * np.sin(av)
        omega = np.maximum(omega, 1e-9)
        di_proxy = 10.0 * np.log10(float(solid) / omega)
        di_h = di_proxy
        di_v = di_proxy

    metrics_h = DirectivitySweepMetrics(freqs, di_h, bw_h)
    metrics_v = DirectivitySweepMetrics(freqs, di_v, bw_v)
    return evaluate_directivity_objective_from_metrics(metrics_h, metrics_v, cfg=cfg)
