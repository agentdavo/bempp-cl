from __future__ import annotations

import numpy as np

try:
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

from bempp_audio.viz.style import DEFAULT_PLOT_CONFIG, setup_freq_axis as _setup_freq_axis


def render_xsection_panel(
    ax,
    response: "FrequencyResponse",
) -> object:
    """
    Render waveguide cross-section (full-width panel).
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    wg = response.waveguide_metadata
    from bempp_audio.mesh.profiles import cts_profile, cts_mouth_angle_deg

    throat_r = float(wg.throat_diameter) / 2.0
    mouth_r = float(wg.mouth_diameter) / 2.0
    length = float(wg.length)

    x = np.linspace(0.0, length, 200)

    if wg.profile == "cts":
        r = cts_profile(
            x,
            throat_r,
            mouth_r,
            length,
            throat_blend=wg.cts_throat_blend,
            transition=wg.cts_transition,
            driver_exit_angle_deg=wg.cts_driver_exit_angle_deg,
            tangency=wg.cts_tangency,
        )
        mouth_angle = cts_mouth_angle_deg(
            throat_r,
            mouth_r,
            length,
            throat_blend=wg.cts_throat_blend,
            transition=wg.cts_transition,
            driver_exit_angle_deg=wg.cts_driver_exit_angle_deg,
            tangency=wg.cts_tangency,
        )
        profile_label = f"CTS (tangency={wg.cts_tangency:.1f})"
    else:
        r = throat_r + (mouth_r - throat_r) * (x / length)
        mouth_angle = np.rad2deg(np.arctan((mouth_r - throat_r) / length))
        profile_label = str(wg.profile).capitalize()

    z = x - length  # z goes from -length to 0
    ax.plot(r * 1000, z * 1000, "b-", linewidth=2, label=profile_label)
    ax.plot(-r * 1000, z * 1000, "b-", linewidth=2)

    ax.axhline(0, color="gray", linestyle="-", linewidth=1.5, label="Baffle")

    ax.annotate(
        f"Throat: {float(wg.throat_diameter) * 1000:.1f}mm",
        xy=(throat_r * 1000, -length * 1000),
        xytext=(throat_r * 1000 + 30, -length * 1000 + 10),
        fontsize=DEFAULT_PLOT_CONFIG.font_size_tick,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    )
    ax.annotate(
        f"Mouth: {float(wg.mouth_diameter) * 1000:.0f}mm\nAngle: {float(mouth_angle):.0f}°",
        xy=(mouth_r * 1000, 0),
        xytext=(mouth_r * 1000 - 40, -length * 1000 * 0.4),
        fontsize=DEFAULT_PLOT_CONFIG.font_size_tick,
    )

    ax.set_xlabel("Radius (mm)")
    ax.set_ylabel("Axial position (mm)")
    ax.set_title(f"Waveguide Cross-Section ({profile_label}, L={length*1000:.0f}mm)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick)
    return ax


def render_throat_impedance(
    ax,
    response: "FrequencyResponse",
    *,
    include_magnitude: bool = True,
    include_plane_wave_reference: bool = True,
    normalize_to_plane_wave: bool = True,
    y_min: float = 0.0,
    y_max: float = 2.0,
) -> object:
    """
    Render throat acoustic input impedance vs frequency.

    This uses `RadiationImpedance.acoustic()` on each `RadiationResult`. For
    waveguide geometries where the active region corresponds to the throat, this
    yields the desired throat input impedance in Pa·s/m³.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs = np.asarray(response.frequencies, dtype=float)
    z_re = np.zeros_like(freqs, dtype=float)
    z_im = np.zeros_like(freqs, dtype=float)
    z_mag = np.zeros_like(freqs, dtype=float)

    for i, result in enumerate(response.results):
        z = result.radiation_impedance().acoustic()
        z_re[i] = float(np.real(z))
        z_im[i] = float(np.imag(z))
        z_mag[i] = float(np.abs(z))

    z_pw = np.nan
    if include_plane_wave_reference or normalize_to_plane_wave:
        try:
            if response.results:
                rho = float(getattr(response.results[0], "rho", np.nan))
                c = float(getattr(response.results[0], "c", np.nan))
            else:
                rho = np.nan
                c = np.nan
            if np.isfinite(rho) and np.isfinite(c):
                if bool(getattr(response, "has_waveguide", False)):
                    wg = response.waveguide_metadata
                    area = np.pi * (0.5 * float(wg.throat_diameter)) ** 2
                else:
                    area = float(response.results[0].radiation_impedance().active_area()) if response.results else 0.0
                if area > 0:
                    z_pw = float((rho * c) / area)
        except Exception:
            z_pw = np.nan

    use_norm = bool(normalize_to_plane_wave) and np.isfinite(z_pw) and float(z_pw) > 0.0
    if use_norm:
        z_re = z_re / float(z_pw)
        z_im = z_im / float(z_pw)
        z_mag = z_mag / float(z_pw)

        ax.semilogx(freqs, z_re, "k-", linewidth=2, label="Re(Z/Z0)")
        ax.semilogx(freqs, z_im, "k--", linewidth=1.5, label="Im(Z/Z0)")
        if include_magnitude:
            ax.semilogx(freqs, z_mag, color="#444444", linestyle=":", linewidth=1.5, label="|Z/Z0|")
        if include_plane_wave_reference:
            ax.axhline(1.0, color="#AA0000", linestyle="--", linewidth=1.2, label="Z0 = ρc/S")

        ax.set_ylabel("Normalized Acoustic Impedance (Z/Z0)")
        ax.set_ylim(float(y_min), float(y_max))
        ax.legend(loc="best", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick)
    else:
        ax.semilogx(freqs, z_re, "k-", linewidth=2, label="Re(Z)")
        ax.semilogx(freqs, z_im, "k--", linewidth=1.5, label="Im(Z)")
        if include_magnitude:
            ax.semilogx(freqs, z_mag, color="#444444", linestyle=":", linewidth=1.5, label="|Z|")
        if include_plane_wave_reference and np.isfinite(z_pw) and float(z_pw) > 0.0:
            ax.axhline(float(z_pw), color="#AA0000", linestyle="--", linewidth=1.2, label="ρc/S (plane-wave)")
        ax.set_ylabel("Acoustic Impedance (Pa·s/m³)")
        ax.legend(loc="best", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("Throat Impedance")
    ax.grid(True, which="both", alpha=0.3)
    _setup_freq_axis(ax)
    return ax
