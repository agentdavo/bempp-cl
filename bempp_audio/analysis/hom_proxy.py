"""
HOM proxy metrics from the radiated field.

This does not "solve HOMs" inside the waveguide. Instead, it provides a practical
proxy based on *mouth-plane pressure non-uniformity* sampled just outside the
aperture (z = z_mouth + epsilon). Strong non-uniformity often correlates with
reflections/diffraction and with designs that excite internal mode conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Iterable
import numpy as np

from bempp_audio.progress import ProgressTracker

if TYPE_CHECKING:
    from bempp_audio.api.loudspeaker import Loudspeaker
    from bempp_audio.results.frequency_response import FrequencyResponse


@dataclass(frozen=True)
class HOMProxyMetrics:
    """
    Summary metrics for mouth-plane pressure non-uniformity.

    All metrics are computed from |p| sampled on a disk, and are intended as
    *comparative* indicators across profiles/geometries (not absolute physics).
    """

    mean_mag: float
    cv_mag: float
    rms_rel: float
    azimuthal_rms_rel: float
    radial_rms_rel: float


@dataclass(frozen=True)
class HOMProxyReport:
    """Per-frequency HOM proxy report."""

    frequencies_hz: np.ndarray
    metrics: list[HOMProxyMetrics]

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return metrics as arrays keyed by metric name."""
        return {
            "frequency_hz": np.asarray(self.frequencies_hz, dtype=float),
            "mean_mag": np.array([m.mean_mag for m in self.metrics], dtype=float),
            "cv_mag": np.array([m.cv_mag for m in self.metrics], dtype=float),
            "rms_rel": np.array([m.rms_rel for m in self.metrics], dtype=float),
            "azimuthal_rms_rel": np.array([m.azimuthal_rms_rel for m in self.metrics], dtype=float),
            "radial_rms_rel": np.array([m.radial_rms_rel for m in self.metrics], dtype=float),
        }


def disk_sample_points(
    *,
    center: tuple[float, float, float],
    radius: float,
    z_offset: float = 1e-4,
    n_r: int = 12,
    n_theta: int = 72,
    include_center: bool = True,
    radial_distribution: str = "area",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sampling points on a disk in the plane z=center_z+z_offset.

    Parameters
    ----------
    center : tuple[float, float, float]
        Mouth center (x, y, z_mouth).
    radius : float
        Mouth radius.
    z_offset : float
        Offset to sample in front of the mouth plane (avoid singular/near-boundary issues).
    n_r : int
        Number of radial bins (rings). Total points ~ n_r * n_theta (+ center).
    n_theta : int
        Number of angular samples per ring.
    include_center : bool
        If True, include a single point at the center.
    radial_distribution : str
        "area" (default) makes points roughly uniform in area (r ~ sqrt(u));
        "linear" spaces r linearly.

    Returns
    -------
    points : np.ndarray
        Shape (3, n_points).
    radii : np.ndarray
        Radial coordinate per point (meters), shape (n_points,).
    thetas : np.ndarray
        Angular coordinate per point (radians), shape (n_points,).
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_r < 1 or n_theta < 4:
        raise ValueError("n_r must be >= 1 and n_theta must be >= 4")

    cx, cy, cz = center
    z = float(cz) + float(z_offset)

    if radial_distribution not in ("area", "linear"):
        raise ValueError("radial_distribution must be 'area' or 'linear'")

    # Radii for rings (exclude 0 unless include_center is False and n_r=1 etc).
    if n_r == 1:
        ring_r = np.array([float(radius)], dtype=float)
    else:
        u = np.linspace(0.0, 1.0, n_r)
        if radial_distribution == "area":
            ring_r = float(radius) * np.sqrt(u)
        else:
            ring_r = float(radius) * u
        ring_r = ring_r[1:]  # exclude zero ring; center handled separately

    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)

    pts = []
    rs = []
    ts = []

    if include_center:
        pts.append([cx, cy, z])
        rs.append(np.array([0.0], dtype=float))
        ts.append(np.array([0.0], dtype=float))

    for rr in ring_r:
        x = cx + rr * np.cos(theta)
        y = cy + rr * np.sin(theta)
        zz = np.full_like(x, z, dtype=float)
        pts.append(np.vstack([x, y, zz]).T)
        rs.append(np.full_like(theta, rr, dtype=float))
        ts.append(theta.copy())

    pts_arr = np.vstack([p if isinstance(p, np.ndarray) else np.array(p)[None, :] for p in pts])
    radii_arr = np.concatenate([np.asarray(r, dtype=float).ravel() for r in rs])
    thetas_arr = np.concatenate([np.asarray(t, dtype=float).ravel() for t in ts])
    return pts_arr.T, radii_arr, thetas_arr


def _ring_groups(radii: np.ndarray, tol: float = 1e-12) -> list[np.ndarray]:
    """Group indices by unique radii (for ringwise metrics)."""
    r = np.asarray(radii, dtype=float)
    uniq = np.unique(np.round(r / tol) * tol)
    groups = []
    for u in uniq:
        groups.append(np.where(np.abs(r - u) <= tol)[0])
    return groups


def _metrics_from_magnitude(mag: np.ndarray, radii: np.ndarray) -> HOMProxyMetrics:
    mag = np.asarray(mag, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if mag.size == 0:
        raise ValueError("no samples")

    mean_mag = float(np.mean(mag))
    if mean_mag <= 0:
        mean_mag = float(np.finfo(float).tiny)

    rel = (mag / mean_mag) - 1.0
    rms_rel = float(np.sqrt(np.mean(rel**2)))
    cv_mag = float(np.std(mag) / mean_mag)

    # Ringwise (azimuthal) non-uniformity: std within each ring.
    groups = _ring_groups(radii)
    ring_stds = []
    ring_means = []
    for idx in groups:
        if idx.size <= 1:
            continue
        ring_stds.append(float(np.std(mag[idx])))
        ring_means.append(float(np.mean(mag[idx])))

    if ring_stds:
        azimuthal_rms_rel = float(np.sqrt(np.mean((np.array(ring_stds) / mean_mag) ** 2)))
        radial_rms_rel = float(np.sqrt(np.mean(((np.array(ring_means) / mean_mag) - 1.0) ** 2)))
    else:
        azimuthal_rms_rel = 0.0
        radial_rms_rel = 0.0

    return HOMProxyMetrics(
        mean_mag=mean_mag,
        cv_mag=cv_mag,
        rms_rel=rms_rel,
        azimuthal_rms_rel=azimuthal_rms_rel,
        radial_rms_rel=radial_rms_rel,
    )


def hom_proxy_for_response(
    response: "FrequencyResponse",
    *,
    mouth_center: tuple[float, float, float],
    mouth_radius: float,
    z_offset: float = 1e-4,
    n_r: int = 12,
    n_theta: int = 72,
    show_progress: bool = True,
) -> HOMProxyReport:
    """
    Compute HOM proxy metrics vs frequency for a solved response.

    Parameters
    ----------
    response : FrequencyResponse
        Solved frequency response (must contain surface solutions).
    mouth_center : tuple[float, float, float]
        (x, y, z_mouth) in meters.
    mouth_radius : float
        Mouth radius in meters.
    z_offset : float
        Sampling plane offset in meters (forward from z_mouth).
    n_r, n_theta : int
        Disk sampling resolution.
    show_progress : bool
        Show progress bar during evaluation.
    """
    points, radii, _ = disk_sample_points(
        center=mouth_center,
        radius=mouth_radius,
        z_offset=z_offset,
        n_r=n_r,
        n_theta=n_theta,
        include_center=True,
        radial_distribution="area",
    )

    metrics: list[HOMProxyMetrics] = []
    with ProgressTracker(
        total=len(response.results),
        desc="HOM proxy",
        unit="freq",
        disable=not show_progress,
    ) as pbar:
        for res in response.results:
            p = res.pressure_at(points)
            mag = np.abs(p)
            metrics.append(_metrics_from_magnitude(mag, radii))
            pbar.update(item=f"{res.frequency:.0f} Hz")

    return HOMProxyReport(frequencies_hz=response.frequencies, metrics=metrics)


def hom_proxy_for_speaker(
    speaker: "Loudspeaker",
    response: "FrequencyResponse",
    *,
    z_offset: float = 1e-4,
    n_r: int = 12,
    n_theta: int = 72,
    show_progress: bool = True,
) -> HOMProxyReport:
    """
    Convenience wrapper that uses `speaker.state.waveguide` for mouth geometry.
    """
    wg = speaker.state.waveguide
    if wg is None:
        raise ValueError("No waveguide metadata on speaker.state.waveguide; provide mouth_center/mouth_radius.")
    return hom_proxy_for_response(
        response,
        mouth_center=tuple(wg.mouth_center),
        mouth_radius=float(wg.mouth_diameter) / 2.0,
        z_offset=z_offset,
        n_r=n_r,
        n_theta=n_theta,
        show_progress=show_progress,
    )
