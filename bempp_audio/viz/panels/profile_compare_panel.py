from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

try:
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

from bempp_audio.viz.style import DEFAULT_PLOT_CONFIG


def _extract_profile_params(obj: object) -> dict:
    """
    Best-effort extraction of waveguide profile parameters from WaveguideMetadata-like objects.
    """
    def _get(name: str, default=None):
        return getattr(obj, name, default)

    return {
        "throat_diameter": float(_get("throat_diameter")),
        "mouth_diameter": float(_get("mouth_diameter")),
        "length": float(_get("length")),
        "profile": str(_get("profile")),
        "cts_throat_blend": float(_get("cts_throat_blend", 0.0)),
        "cts_transition": float(_get("cts_transition", 0.75)),
        "cts_tangency": float(_get("cts_tangency", 1.0)),
        "cts_driver_exit_angle_deg": _get("cts_driver_exit_angle_deg", None),
    }


def _radius_profile(x: np.ndarray, params: dict) -> np.ndarray:
    from bempp_audio.mesh.profiles import (
        conical_profile,
        exponential_profile,
        hyperbolic_profile,
        tractrix_horn_profile,
        oblate_spheroidal_profile,
        cts_profile,
    )

    throat_r = 0.5 * float(params["throat_diameter"])
    mouth_r = 0.5 * float(params["mouth_diameter"])
    length = float(params["length"])
    p = str(params["profile"]).lower().strip()

    if p == "cts":
        return cts_profile(
            x,
            throat_r,
            mouth_r,
            length,
            throat_blend=float(params.get("cts_throat_blend", 0.0)),
            transition=float(params.get("cts_transition", 0.75)),
            driver_exit_angle_deg=params.get("cts_driver_exit_angle_deg", None),
            tangency=float(params.get("cts_tangency", 1.0)),
        )
    if p in {"conical", "cone"}:
        return conical_profile(x, throat_r, mouth_r, length)
    if p in {"exponential", "exp"}:
        return exponential_profile(x, throat_r, mouth_r, length, flare_constant=None)
    if p in {"tractrix"}:
        return tractrix_horn_profile(x, throat_r, mouth_r, length)
    if p in {"hyperbolic"}:
        return hyperbolic_profile(x, throat_r, mouth_r, length, sharpness=2.0)
    if p in {"os", "oblate", "oblate_spheroidal"}:
        return oblate_spheroidal_profile(x, throat_r, mouth_r, length, opening_angle_deg=None)

    return conical_profile(x, throat_r, mouth_r, length)


def render_profile_compare(
    ax,
    response: "FrequencyResponse",
    *,
    profiles: Optional[Sequence[object]] = None,
    labels: Optional[Sequence[str]] = None,
    include_current: bool = True,
) -> object:
    """
    Overlay 2–3 waveguide profiles (geometry-only) on a single axes.

    Parameters
    ----------
    profiles:
        Sequence of WaveguideMetadata-like objects (or compatible objects with
        throat_diameter/mouth_diameter/length/profile attributes).
    labels:
        Optional labels corresponding to `profiles` (excluding the current response unless included in the list).
    include_current:
        If True and `response.has_waveguide`, include `response.waveguide_metadata` as the first profile.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    objs: list[object] = []
    if include_current and bool(getattr(response, "has_waveguide", False)):
        try:
            objs.append(response.waveguide_metadata)
        except Exception:
            pass

    if profiles:
        objs.extend(list(profiles))

    if not objs:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No waveguide profiles to compare", ha="center", va="center")
        return ax

    colors = ["#0066CC", "#CC0000", "#00AA66", "#6600CC"]
    if labels is None:
        labels_list: list[str] = []
        for obj in objs:
            p = _extract_profile_params(obj)
            labels_list.append(f"{p['profile']} ({p['throat_diameter']*1000:.0f}→{p['mouth_diameter']*1000:.0f}mm)")
    else:
        labels_list = list(labels)
        if len(labels_list) != len(objs):
            # Best-effort: pad or truncate.
            while len(labels_list) < len(objs):
                labels_list.append(f"profile_{len(labels_list)+1}")
            labels_list = labels_list[: len(objs)]

    max_len = max(_extract_profile_params(o)["length"] for o in objs)
    x = np.linspace(0.0, float(max_len), 300)

    for i, (obj, label) in enumerate(zip(objs, labels_list)):
        p = _extract_profile_params(obj)
        length = float(p["length"])
        if length <= 0:
            continue
        x_i = x[x <= length]
        r = _radius_profile(x_i, p)
        z = x_i - length
        color = colors[i % len(colors)]
        ax.plot(r * 1000, z * 1000, color=color, linewidth=2, label=label)
        ax.plot(-r * 1000, z * 1000, color=color, linewidth=2)

    ax.axhline(0, color="gray", linestyle="-", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Radius (mm)")
    ax.set_ylabel("Axial position (mm)")
    ax.set_title("Profile Comparison")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick)
    return ax

