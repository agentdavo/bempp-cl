from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from bempp_audio.viz.data.normalization import normalize_spl_map_by_angle, symmetric_polar_map, NormalizedSPLMap

if TYPE_CHECKING:
    from bempp_audio.results import FrequencyResponse
    from bempp_audio.results.polar import PolarPlane


@dataclass(frozen=True)
class PolarMapData:
    """
    Frequency/angle polar SPL map computed via `FrequencyResponse.polar_sweep`.

    Storage convention:
    - `spl_map_db` has shape (n_angles, n_freqs)
    - `frequencies_hz` has shape (n_freqs,)
    - `angles_deg` has shape (n_angles,)
    """

    frequencies_hz: np.ndarray
    angles_deg: np.ndarray
    spl_map_db: np.ndarray
    distance_m: float
    plane: "PolarPlane"
    ref_pa: float = 20e-6

    @classmethod
    def from_response(
        cls,
        response: "FrequencyResponse",
        *,
        distance_m: float = 1.0,
        max_angle_deg: float = 180.0,
        step_deg: float = 5.0,
        plane: "PolarPlane" = "horizontal",
        ref_pa: float = 20e-6,
        show_progress: bool = True,
    ) -> "PolarMapData":
        freqs = np.asarray(response.frequencies, dtype=float)
        n_angles = max(int(float(max_angle_deg) / float(step_deg)) + 1, 5)
        angles_deg = np.linspace(0.0, float(max_angle_deg), n_angles)

        sweep = response.polar_sweep(
            angles_deg=angles_deg.tolist(),
            distance_m=float(distance_m),
            plane=str(plane),
            show_progress=bool(show_progress),
        )

        spl_db = sweep.spl_db(ref_pa=float(ref_pa)).T  # (n_angles, n_freqs)
        return cls(
            frequencies_hz=freqs,
            angles_deg=angles_deg,
            spl_map_db=spl_db,
            distance_m=float(distance_m),
            plane=plane,
            ref_pa=float(ref_pa),
        )

    def normalized(self, *, normalize_angle_deg: float = 0.0) -> NormalizedSPLMap:
        return normalize_spl_map_by_angle(
            angles_deg=self.angles_deg,
            frequencies_hz=self.frequencies_hz,
            spl_map_db=self.spl_map_db,
            normalize_angle_deg=float(normalize_angle_deg),
        )

    def normalized_symmetric(
        self,
        *,
        normalize_angle_deg: float = 0.0,
    ) -> tuple[NormalizedSPLMap, np.ndarray, np.ndarray]:
        """
        Return (normalized_map, symmetric_angles_deg, symmetric_spl_map_db).
        """
        norm = self.normalized(normalize_angle_deg=float(normalize_angle_deg))
        angles_sym, spl_sym = symmetric_polar_map(angles_deg=norm.angles_deg, spl_map_db=norm.spl_map_db)
        return norm, angles_sym, spl_sym

    def spl_curves_db(
        self,
        response: "FrequencyResponse",
        *,
        angles_deg: list[float],
        show_progress: bool = True,
        tol_deg: float = 1e-9,
        logger: Optional[object] = None,
    ) -> np.ndarray:
        """
        SPL curves at selected angles using the cached map when possible.

        Returns an array of shape (n_freqs, n_angles).
        """
        freqs = self.frequencies_hz
        angles_req = np.asarray([float(a) for a in angles_deg], dtype=float)
        curves = np.zeros((freqs.size, angles_req.size), dtype=float)

        missing_angles: list[float] = []
        missing_cols: list[int] = []

        for j, a in enumerate(angles_req):
            if self.angles_deg.size == 0:
                missing_angles.append(float(a))
                missing_cols.append(int(j))
                continue
            idx = int(np.argmin(np.abs(self.angles_deg - float(a))))
            if abs(float(self.angles_deg[idx]) - float(a)) <= float(tol_deg):
                curves[:, j] = self.spl_map_db[idx, :]
            else:
                missing_angles.append(float(a))
                missing_cols.append(int(j))

        if missing_angles:
            if logger is not None:
                try:
                    logger.info(
                        f"Computing SPL curves: {len(missing_angles)} angles "
                        "(not on polar-map grid; requires extra field evaluations)"
                    )
                except Exception:
                    pass

            sweep_extra = response.polar_sweep(
                angles_deg=missing_angles,
                distance_m=float(self.distance_m),
                plane=str(self.plane),
                show_progress=bool(show_progress),
            )
            spl_extra = sweep_extra.spl_db(ref_pa=float(self.ref_pa))  # (n_freqs, n_missing)
            for k, j in enumerate(missing_cols):
                curves[:, j] = spl_extra[:, k]
        else:
            if logger is not None:
                try:
                    logger.info(f"SPL curves: {len(angles_req)} angles (reused polar map)")
                except Exception:
                    pass

        return curves

