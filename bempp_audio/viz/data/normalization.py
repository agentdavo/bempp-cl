from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NormalizedSPLMap:
    angles_deg: np.ndarray  # (n_a,)
    frequencies_hz: np.ndarray  # (n_f,)
    spl_map_db: np.ndarray  # (n_a, n_f) normalized
    spl_ref_db: np.ndarray  # (n_f,)
    norm_angle_used_deg: float
    norm_angle_index: int


def normalize_spl_map_by_angle(
    *,
    angles_deg: np.ndarray,
    frequencies_hz: np.ndarray,
    spl_map_db: np.ndarray,
    normalize_angle_deg: float,
) -> NormalizedSPLMap:
    """
    Normalize an SPL map by subtracting the curve at a chosen reference angle.

    Parameters
    ----------
    angles_deg:
        Angle grid in degrees, shape (n_a,).
    frequencies_hz:
        Frequencies in Hz, shape (n_f,).
    spl_map_db:
        SPL map in dB, shape (n_a, n_f).
    normalize_angle_deg:
        Desired reference angle in degrees. The nearest available sample is used.
    """
    angles = np.asarray(angles_deg, dtype=float)
    freqs = np.asarray(frequencies_hz, dtype=float)
    spl = np.asarray(spl_map_db, dtype=float)

    if spl.size == 0 or angles.size == 0 or freqs.size == 0:
        return NormalizedSPLMap(
            angles_deg=angles,
            frequencies_hz=freqs,
            spl_map_db=spl,
            spl_ref_db=np.zeros_like(freqs, dtype=float),
            norm_angle_used_deg=float(normalize_angle_deg),
            norm_angle_index=0,
        )

    if spl.shape != (angles.size, freqs.size):
        raise ValueError(f"Expected spl_map_db shape {(angles.size, freqs.size)}; got {spl.shape}")

    norm_idx = int(np.argmin(np.abs(angles - float(normalize_angle_deg))))
    norm_angle_used = float(angles[norm_idx])
    spl_ref = spl[norm_idx, :].copy()

    return NormalizedSPLMap(
        angles_deg=angles,
        frequencies_hz=freqs,
        spl_map_db=spl - spl_ref[None, :],
        spl_ref_db=spl_ref,
        norm_angle_used_deg=norm_angle_used,
        norm_angle_index=norm_idx,
    )


def symmetric_polar_map(
    *,
    angles_deg: np.ndarray,
    spl_map_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mirror a 0..max angle polar map into a symmetric -max..max map.

    Parameters
    ----------
    angles_deg:
        Non-negative angle grid, shape (n_a,).
    spl_map_db:
        SPL map, shape (n_a, n_f), corresponding to `angles_deg`.
    """
    angles = np.asarray(angles_deg, dtype=float)
    spl = np.asarray(spl_map_db, dtype=float)

    if angles.size == 0 or spl.size == 0:
        return angles, spl

    if spl.shape[0] != angles.size:
        raise ValueError(f"Expected spl_map_db first dimension {angles.size}; got {spl.shape}")

    # Avoid duplicating 0° in the symmetric array.
    angles_sym = np.concatenate([-angles[::-1][:-1], angles])
    spl_sym = np.concatenate([spl[::-1, :][:-1, :], spl], axis=0)
    return angles_sym, spl_sym

