"""
Plotly HTML dashboards for design-stage inspection.

The matplotlib ReportBuilder is great for static summaries. For iterative design work
we also support interactive Plotly dashboards (HTML) that are easier to share and
inspect (hover, zoom, toggle traces).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np

from bempp_audio.viz.data.polar_data import PolarMapData

if TYPE_CHECKING:  # pragma: no cover
    from bempp_audio.results import FrequencyResponse


@dataclass(frozen=True)
class DashboardOptions:
    distance_m: float = 1.0
    max_angle_deg: float = 90.0
    step_deg: float = 5.0
    plane: str = "horizontal"
    normalize_angle_deg: float = 10.0
    angles_deg: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0)
    beamwidth_level_db: float = -6.0
    beamwidth_plane: str = "xz"
    listening_window_angles_deg: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)
    crossover_hz: float = 1000.0
    upper_hz: float = 20000.0

    # Resonance/mode markers (optional)
    structural_mode_hz: tuple[float, ...] = ()
    max_auto_markers: int = 4

    # Score thresholds (very conservative defaults; tune to taste per project)
    spl_ripple_pass_db: float = 2.0
    spl_ripple_warn_db: float = 3.0
    di_ripple_pass_db: float = 2.0
    di_ripple_warn_db: float = 3.0
    beamwidth_monotonicity_pass: float = 0.70
    beamwidth_monotonicity_warn: float = 0.60
    excursion_pass_mm: float = 0.20
    excursion_warn_mm: float = 0.30


def _band_edges() -> list[tuple[float, float, str]]:
    return [
        (1000.0, 2000.0, "1–2 kHz"),
        (2000.0, 5000.0, "2–5 kHz"),
        (5000.0, 10000.0, "5–10 kHz"),
        (10000.0, 20000.0, "10–20 kHz"),
    ]


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _listening_window_spl_db(
    *,
    freqs_hz: np.ndarray,
    angles_deg_grid: np.ndarray,
    spl_map_db: np.ndarray,
    angles_deg: Sequence[float],
    ref_pa: float = 20e-6,
) -> np.ndarray:
    """
    Compute a listening-window SPL curve as an energy average across angles.

    Uses mean(p_rms^2) across selected angles, then converts back to SPL.
    """
    if spl_map_db.size == 0:
        return np.zeros_like(freqs_hz, dtype=float)

    angles_req = np.asarray([float(a) for a in angles_deg], dtype=float)
    idxs = [int(np.argmin(np.abs(angles_deg_grid - a))) for a in angles_req]
    spl_sel = np.asarray(spl_map_db[idxs, :], dtype=float)  # (n_angles, n_freqs)

    # Convert dB SPL to p_rms, energy-average, back to dB.
    p_rms = float(ref_pa) * (10.0 ** (spl_sel / 20.0))
    p2_mean = np.mean(p_rms * p_rms, axis=0)
    p_rms_mean = np.sqrt(np.maximum(p2_mean, (float(ref_pa) * 1e-10) ** 2))
    return 20.0 * np.log10(p_rms_mean / float(ref_pa))


def _peak_indices(y: np.ndarray, *, max_peaks: int = 4, min_prominence: float = 0.05) -> list[int]:
    """
    Simple local-maximum picker (no SciPy dependency).
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return []
    # Normalize for scale invariance
    finite = np.isfinite(y)
    if not np.any(finite):
        return []
    y0 = y[finite]
    scale = float(np.nanmax(y0) - np.nanmin(y0))
    if scale <= 0:
        return []
    yn = (y - float(np.nanmin(y0))) / scale

    peaks = []
    for i in range(1, yn.size - 1):
        if not np.isfinite(yn[i - 1 : i + 2]).all():
            continue
        if yn[i] >= yn[i - 1] and yn[i] >= yn[i + 1] and (yn[i] - 0.5 * (yn[i - 1] + yn[i + 1])) >= min_prominence:
            peaks.append(i)
    # Keep strongest peaks
    peaks.sort(key=lambda i: yn[i], reverse=True)
    return peaks[: int(max_peaks)]


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
    # lower is better
    if value <= pass_thr:
        return "PASS"
    if value <= warn_thr:
        return "WARN"
    return "FAIL"


def driver_dashboard_figure(
    response: "FrequencyResponse",
    *,
    title: str = "Driver + Waveguide Dashboard",
    options: DashboardOptions = DashboardOptions(),
    phase_plug_metrics: Optional[Sequence[object]] = None,
    show_progress: bool = True,
):
    """
    Build a Plotly multi-panel dashboard for compression-driver design.

    Panels (when data is available):
    - Scorecard (directivity metrics, excursion, etc.)
    - SPL curves (selected angles)
    - Polar SPL heatmap
    - DI vs frequency
    - Beamwidth vs frequency
    - Driver electrical impedance (|Z| + phase)
    - Driver excursion (mm)
    - Throat acoustic impedance (Re/Im/|Z|)
    - Radiation specific impedance (Re/Im)
    - Radiated power (W) (when available)
    - On-axis group delay (ms) (when multiple frequencies)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        raise ImportError("plotly is required for HTML dashboards (`pip install plotly`).") from e

    freqs = _as_float_array(response.frequencies)

    polar = PolarMapData.from_response(
        response,
        distance_m=float(options.distance_m),
        max_angle_deg=float(options.max_angle_deg),
        step_deg=float(options.step_deg),
        plane=str(options.plane),
        ref_pa=20e-6,
        show_progress=bool(show_progress),
    )

    norm, theta_sym, spl_sym = polar.normalized_symmetric(normalize_angle_deg=float(options.normalize_angle_deg))

    # SPL curves from polar map (reuse where possible)
    angles_deg = tuple(float(a) for a in options.angles_deg if float(a) <= float(options.max_angle_deg) + 1e-9)
    if not angles_deg:
        angles_deg = (0.0,)
    curves = polar.spl_curves_db(response, angles_deg=list(angles_deg), show_progress=bool(show_progress))
    # Listening window SPL (energy average across angles)
    lw_angles = tuple(float(a) for a in options.listening_window_angles_deg if float(a) <= float(options.max_angle_deg) + 1e-9)
    if not lw_angles:
        lw_angles = (0.0,)
    spl_lw_db = _listening_window_spl_db(
        freqs_hz=_as_float_array(polar.frequencies_hz),
        angles_deg_grid=_as_float_array(polar.angles_deg),
        spl_map_db=_as_float_array(polar.spl_map_db),
        angles_deg=lw_angles,
        ref_pa=float(polar.ref_pa),
    )

    # Directivity metrics
    try:
        from bempp_audio.results.directivity import compute_directivity_sweep_metrics

        directivity_metrics = compute_directivity_sweep_metrics(
            response.results, beamwidth_level_db=float(options.beamwidth_level_db), plane=str(options.beamwidth_plane)
        )
    except Exception:
        directivity_metrics = None

    # Driver metrics
    z_drv = response.driver_impedance
    x_mm = response.driver_excursion_mm

    # Impedance traces (best-effort; computed directly from results)
    z_spec_re = np.full(freqs.shape, np.nan, dtype=float)
    z_spec_im = np.full(freqs.shape, np.nan, dtype=float)
    z_thr_re = np.full(freqs.shape, np.nan, dtype=float)
    z_thr_im = np.full(freqs.shape, np.nan, dtype=float)
    z_thr_mag = np.full(freqs.shape, np.nan, dtype=float)

    for i, result in enumerate(response.results):
        try:
            z = result.radiation_impedance()
            z_spec = z.specific()
            z_ac = z.acoustic()
            z_spec_re[i] = float(np.real(z_spec))
            z_spec_im[i] = float(np.imag(z_spec))
            z_thr_re[i] = float(np.real(z_ac))
            z_thr_im[i] = float(np.imag(z_ac))
            z_thr_mag[i] = float(np.abs(z_ac))
        except Exception:
            pass

    # Plane-wave reference (if waveguide metadata exists)
    z_pw = None
    z0_spec = None
    try:
        if bool(getattr(response, "has_waveguide", False)) and response.results:
            wg = response.waveguide_metadata
            rho = float(getattr(response.results[0], "rho", np.nan))
            c = float(getattr(response.results[0], "c", np.nan))
            if np.isfinite(rho) and np.isfinite(c):
                z0_spec = float(rho * c)  # Pa·s/m
                area = np.pi * (0.5 * float(wg.throat_diameter)) ** 2
                if area > 0:
                    z_pw = float((rho * c) / area)
    except Exception:
        z_pw = None
        z0_spec = None

    # Radiated power (best-effort)
    p_rad_w = None
    try:
        _, p_rad_w = response.radiated_power_vs_freq()
        p_rad_w = _as_float_array(p_rad_w)
    except Exception:
        p_rad_w = None

    # On-axis group delay (best-effort; uses field evaluation)
    gd_ms = None
    try:
        if freqs.size >= 2:
            sweep_on = response.polar_sweep(
                angles_deg=[0.0],
                distance_m=float(options.distance_m),
                plane=str(options.plane),
                show_progress=bool(show_progress),
            )
            p_on = np.asarray(sweep_on.pressure, dtype=complex).reshape(-1)
            phase = np.unwrap(np.angle(p_on))
            omega = 2.0 * np.pi * _as_float_array(freqs)
            gd_s = -np.gradient(phase, omega)
            gd_ms = 1000.0 * _as_float_array(gd_s)
    except Exception:
        gd_ms = None

    # -----------------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------------
    specs = [
        [{"type": "xy"}, {"type": "xy", "secondary_y": True}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [{"type": "heatmap", "colspan": 2}, None, {"type": "xy", "secondary_y": False}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    ]

    subplot_titles = (
        "Scorecard",
        "Driver Z<sub>elec</sub>",
        "Excursion",
        "SPL Curves",
        "Directivity Index",
        "Beamwidth (-6 dB)",
        "Polar Map (normalized)",
        "Throat Z<sub>ac</sub>",
        "Radiation Z<sub>spec</sub>",
        "Radiated Power",
        "Group Delay (on-axis)",
        "",
    )

    fig = make_subplots(
        rows=4,
        cols=3,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    # Scorecard table
    score_rows: list[str] = []
    score_vals: list[str] = []

    score_rows.extend(["Freqs", "Range", "Normalize", "Max angle"])
    score_vals.extend(
        [
            f"{len(freqs)}",
            f"{float(freqs[0]):.0f}–{float(freqs[-1]):.0f} Hz" if freqs.size else "n/a",
            f"{float(options.normalize_angle_deg):g}°",
            f"{float(options.max_angle_deg):g}°",
        ]
    )
    if directivity_metrics is not None:
        score_rows.extend(["DI ripple", "Beamwidth mono", "Beamwidth range"])
        score_vals.extend(
            [
                f"{directivity_metrics.di_ripple_db:.2f} dB",
                f"{100.0*directivity_metrics.beamwidth_monotonicity:.0f}%",
                f"{np.min(directivity_metrics.beamwidth_values):.0f}–{np.max(directivity_metrics.beamwidth_values):.0f}°",
            ]
        )
    if x_mm is not None and np.size(x_mm):
        score_rows.append("Max excursion")
        score_vals.append(f"{float(np.nanmax(_as_float_array(x_mm))):.3g} mm")

    # Banded ripple grading (listening window)
    for f0, f1, label in _band_edges():
        ripple = _pp_ripple_in_band(freqs, spl_lw_db, f0, f1)
        grade = _grade(ripple, pass_thr=float(options.spl_ripple_pass_db), warn_thr=float(options.spl_ripple_warn_db))
        score_rows.append(f"SPL ripple LW ({label})")
        score_vals.append(f"{ripple:.2f} dB ({grade})" if np.isfinite(ripple) else "N/A")

    if directivity_metrics is not None:
        di_grade = _grade(
            float(directivity_metrics.di_ripple_db),
            pass_thr=float(options.di_ripple_pass_db),
            warn_thr=float(options.di_ripple_warn_db),
        )
        bw_grade = _grade(
            float(directivity_metrics.beamwidth_monotonicity),
            pass_thr=float(options.beamwidth_monotonicity_pass),
            warn_thr=float(options.beamwidth_monotonicity_warn),
            higher_is_better=True,
        )
        score_rows.append("DI ripple grade")
        score_vals.append(di_grade)
        score_rows.append("Beamwidth monotonicity grade")
        score_vals.append(bw_grade)

    if x_mm is not None and np.size(x_mm):
        x_max = float(np.nanmax(_as_float_array(x_mm)))
        x_grade = _grade(x_max, pass_thr=float(options.excursion_pass_mm), warn_thr=float(options.excursion_warn_mm))
        score_rows.append("Excursion grade")
        score_vals.append(x_grade)

    score_lines = [f"<b>{r}</b>: {v}" for r, v in zip(score_rows, score_vals)]
    score_html = "<br>".join(score_lines)
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="text", text=[score_html], showlegend=False, hoverinfo="skip"),
        row=1,
        col=1,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1, visible=False)
    fig.update_yaxes(showticklabels=False, row=1, col=1, visible=False)

    # Driver electrical impedance (|Z| + phase)
    if z_drv is not None:
        z_re, z_im = z_drv
        z_re = _as_float_array(z_re)
        z_im = _as_float_array(z_im)
        z_mag = np.sqrt(z_re * z_re + z_im * z_im)
        z_phase = np.degrees(np.arctan2(z_im, z_re))

        fig.add_trace(go.Scatter(x=freqs, y=z_mag, name="|Z| (Ω)", mode="lines"), row=1, col=2, secondary_y=False)
        fig.add_trace(
            go.Scatter(x=freqs, y=z_phase, name="Phase (°)", mode="lines", line=dict(dash="dash")),
            row=1,
            col=2,
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Ω", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="°", row=1, col=2, secondary_y=True)

    # Excursion
    if x_mm is not None:
        fig.add_trace(go.Scatter(x=freqs, y=_as_float_array(x_mm), name="Excursion (mm)", mode="lines"), row=1, col=3)
        fig.update_yaxes(title_text="mm", row=1, col=3)

    # SPL curves
    for j, a in enumerate(angles_deg):
        fig.add_trace(
            go.Scatter(x=freqs, y=curves[:, j], name=f"SPL {a:g}°", mode="lines"),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=freqs, y=spl_lw_db, name=f"Listening window ({', '.join([f'{a:g}°' for a in lw_angles])})", mode="lines", line=dict(width=3)),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="dB re 20µPa", row=2, col=1)

    # DI
    try:
        f_di, di = response.directivity_index_vs_freq(angle=0.0)
        fig.add_trace(go.Scatter(x=_as_float_array(f_di), y=_as_float_array(di), name="DI (dB)", mode="lines"), row=2, col=2)
        fig.update_yaxes(title_text="dB", row=2, col=2)
    except Exception:
        pass

    # Beamwidth
    try:
        f_bw, bw = response.beamwidth_vs_freq(level_db=float(options.beamwidth_level_db), plane=str(options.beamwidth_plane))
        fig.add_trace(go.Scatter(x=_as_float_array(f_bw), y=_as_float_array(bw), name="Beamwidth", mode="lines"), row=2, col=3)
        fig.update_yaxes(title_text="deg", row=2, col=3)
    except Exception:
        pass

    # Polar heatmap (normalized, symmetric angles)
    fig.add_trace(
        go.Heatmap(
            x=_as_float_array(norm.frequencies_hz),
            y=_as_float_array(theta_sym),
            z=_as_float_array(spl_sym),
            colorscale="Jet",
            colorbar=dict(title="dB"),
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Angle (deg)", row=3, col=1)

    # Throat impedance
    if np.any(np.isfinite(z_thr_re)):
        if z_pw is not None and float(z_pw) > 0:
            z_thr_re_n = z_thr_re / float(z_pw)
            z_thr_im_n = z_thr_im / float(z_pw)
            z_thr_mag_n = z_thr_mag / float(z_pw)

            fig.add_trace(go.Scatter(x=freqs, y=z_thr_re_n, name="Re(Z/Z0)", mode="lines"), row=3, col=3)
            fig.add_trace(go.Scatter(x=freqs, y=z_thr_im_n, name="Im(Z/Z0)", mode="lines", line=dict(dash="dash")), row=3, col=3)
            fig.add_trace(go.Scatter(x=freqs, y=z_thr_mag_n, name="|Z/Z0|", mode="lines", line=dict(dash="dot")), row=3, col=3)
            fig.add_hline(y=1.0, line=dict(color="#AA0000", dash="dash"), row=3, col=3)
            fig.update_yaxes(title_text="Z/Z0", row=3, col=3, range=[0.0, 2.0])
        else:
            fig.add_trace(go.Scatter(x=freqs, y=z_thr_re, name="Re(Zac)", mode="lines"), row=3, col=3)
            fig.add_trace(go.Scatter(x=freqs, y=z_thr_im, name="Im(Zac)", mode="lines", line=dict(dash="dash")), row=3, col=3)
            fig.add_trace(go.Scatter(x=freqs, y=z_thr_mag, name="|Zac|", mode="lines", line=dict(dash="dot")), row=3, col=3)
            fig.update_yaxes(title_text="Pa·s/m³", row=3, col=3)

    # Radiation impedance (specific)
    if np.any(np.isfinite(z_spec_re)):
        if z0_spec is not None and float(z0_spec) > 0:
            z_spec_re_n = z_spec_re / float(z0_spec)
            z_spec_im_n = z_spec_im / float(z0_spec)
            fig.add_trace(go.Scatter(x=freqs, y=z_spec_re_n, name="Re(Zspec/ρc)", mode="lines"), row=4, col=1)
            fig.add_trace(go.Scatter(x=freqs, y=z_spec_im_n, name="Im(Zspec/ρc)", mode="lines", line=dict(dash="dash")), row=4, col=1)
            fig.add_hline(y=1.0, line=dict(color="#AA0000", dash="dash"), row=4, col=1)
            fig.update_yaxes(title_text="Zspec/(ρc)", row=4, col=1, range=[0.0, 2.0])
        else:
            fig.add_trace(go.Scatter(x=freqs, y=z_spec_re, name="Re(Zspec)", mode="lines"), row=4, col=1)
            fig.add_trace(go.Scatter(x=freqs, y=z_spec_im, name="Im(Zspec)", mode="lines", line=dict(dash="dash")), row=4, col=1)
            fig.update_yaxes(title_text="Pa·s/m", row=4, col=1)

    # Radiated power / group delay (use col=2/3 only when phase-plug metrics are absent)
    if not phase_plug_metrics:
        if p_rad_w is not None and np.size(p_rad_w) and np.any(np.isfinite(p_rad_w)):
            fig.add_trace(go.Scatter(x=freqs, y=p_rad_w, name="P_rad (W)", mode="lines"), row=4, col=2)
            fig.update_yaxes(title_text="W", row=4, col=2)
        if gd_ms is not None and np.size(gd_ms) and np.any(np.isfinite(gd_ms)):
            fig.add_trace(go.Scatter(x=freqs, y=gd_ms, name="Group delay (ms)", mode="lines"), row=4, col=3)
            fig.update_yaxes(title_text="ms", row=4, col=3)

    # Optional phase-plug metrics (if provided)
    # Accept either bempp_audio.fea.phase_plug_coupling.PhasePlugMetrics or dict-like.
    if phase_plug_metrics:
        try:
            f_pp = np.array([float(getattr(m, "frequency_hz", m["frequency_hz"])) for m in phase_plug_metrics], dtype=float)
            eff = np.array([float(getattr(m, "transmission_efficiency", m["transmission_efficiency"])) for m in phase_plug_metrics], dtype=float)
            uni = np.array([float(getattr(m, "pressure_uniformity", m["pressure_uniformity"])) for m in phase_plug_metrics], dtype=float)
            phs = np.array([float(getattr(m, "phase_spread_deg", m["phase_spread_deg"])) for m in phase_plug_metrics], dtype=float)

            fig.add_trace(go.Scatter(x=f_pp, y=eff, name="Phase plug η (power)", mode="lines"), row=4, col=2)
            fig.update_yaxes(title_text="η", row=4, col=2, range=[0, 1.05])

            fig.add_trace(go.Scatter(x=f_pp, y=uni, name="Throat uniformity", mode="lines"), row=4, col=3)
            fig.add_trace(go.Scatter(x=f_pp, y=phs, name="Phase spread (deg)", mode="lines", line=dict(dash="dash")), row=4, col=3)
            fig.update_yaxes(title_text="(unitless / deg)", row=4, col=3)
        except Exception:
            pass

    # -----------------------------------------------------------------------------
    # Cross-linked markers (resonances, modes, band edges)
    # -----------------------------------------------------------------------------
    marker_freqs: list[tuple[float, str, str]] = []  # (f, label, color)

    # Auto markers from Z_elec |Z| peaks
    if z_drv is not None and freqs.size:
        z_re, z_im = z_drv
        z_mag = np.sqrt(_as_float_array(z_re) ** 2 + _as_float_array(z_im) ** 2)
        for idx in _peak_indices(z_mag, max_peaks=int(options.max_auto_markers)):
            marker_freqs.append((float(freqs[idx]), "Z_elec peak", "#444444"))

    # Auto markers from throat |Z|
    if np.any(np.isfinite(z_thr_mag)) and freqs.size:
        for idx in _peak_indices(z_thr_mag, max_peaks=int(options.max_auto_markers)):
            marker_freqs.append((float(freqs[idx]), "Z_throat peak", "#AA0000"))

    # Structural mode markers
    for f0 in options.structural_mode_hz:
        if float(f0) > 0:
            marker_freqs.append((float(f0), "FEM mode", "#0066CC"))

    # Crossovers / bounds
    if float(options.crossover_hz) > 0:
        marker_freqs.append((float(options.crossover_hz), "XO", "#008800"))
    if float(options.upper_hz) > 0:
        marker_freqs.append((float(options.upper_hz), "20k", "#008800"))

    # Apply as global shapes
    for f0, label, color in marker_freqs:
        fig.add_vline(x=float(f0), line_width=1, line_dash="dot", line_color=color, annotation_text=label, annotation_position="top")

    fig.update_xaxes(type="log")
    fig.update_layout(
        title=str(title),
        height=1100,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="left", x=0.0),
        margin=dict(l=40, r=20, t=60, b=80),
    )

    return fig


def save_driver_dashboard_html(
    response: "FrequencyResponse",
    filename: str = "driver_dashboard.html",
    *,
    title: str = "Driver + Waveguide Dashboard",
    options: DashboardOptions = DashboardOptions(),
    phase_plug_metrics: Optional[Sequence[object]] = None,
    show_progress: bool = True,
    include_plotlyjs: str | bool = True,
) -> str:
    """
    Save an interactive Plotly HTML dashboard to disk.
    """
    fig = driver_dashboard_figure(
        response,
        title=title,
        options=options,
        phase_plug_metrics=phase_plug_metrics,
        show_progress=show_progress,
    )
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs)
    return str(path)
