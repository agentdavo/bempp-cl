"""
Reporting helpers for mesh-related diagnostics.

These functions keep example scripts small and ensure consistent formatting
across the codebase by routing output through `bempp_audio.progress.Logger`.
"""

from __future__ import annotations

from typing import Optional

from bempp_audio.progress import Logger, get_logger
from bempp_audio.mesh.waveguide_profile import WaveguideProfileCheck


def log_profile_check(tag: str, rep: WaveguideProfileCheck, *, logger: Optional[Logger] = None) -> None:
    """Log a `WaveguideProfileCheck` in a consistent, human-readable format."""
    logger = logger or get_logger()
    status = "OK" if rep.ok else "FAILED"

    logger.subsection(f"{tag} ({status})")
    logger.config("Axisymmetric", f"min Δr={rep.min_dr:.3e} (violations={rep.n_r_violations})")

    if rep.min_support_delta is not None:
        worst = f"{rep.worst_direction_deg}°" if rep.worst_direction_deg is not None else "n/a"
        logger.config(
            "Morphed support",
            f"min Δs={rep.min_support_delta} (violations={rep.n_support_violations}, worst_dir≈{worst})",
        )

    if rep.min_area_delta is not None:
        logger.config(
            "Cross-section area",
            f"min ΔA={rep.min_area_delta} (violations={rep.n_area_violations})",
        )

    if rep.max_scale_x is not None or rep.max_scale_y is not None:
        logger.config("Applied scaling", f"max sx={rep.max_scale_x} max sy={rep.max_scale_y}")

    for note in rep.notes:
        logger.info(f"  note: {note}")
