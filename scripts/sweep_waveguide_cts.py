#!/usr/bin/env python3
"""
CTS waveguide sweep harness (minimal).

Writes one folder of artifacts per sweep point:
  logs/sweeps/<name>/case_<i>_<tag>/
    - snapshot.json
    - scorecard.json
    - dashboard.html (if plotly installed)

This intentionally keeps caching simple: if `scorecard.json` exists, the case is skipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SweepCase:
    throat_blend: float
    transition: float
    tangency: float

    def tag(self) -> str:
        return f"tb{self.throat_blend:.2f}_tc{self.transition:.2f}_tg{self.tangency:.2f}".replace(".", "p")


def _cases() -> Iterable[SweepCase]:
    tb = [0.05, 0.10, 0.15]
    tc = [0.65, 0.75, 0.85]
    tg = [0.0, 0.5, 1.0]
    for a in tb:
        for b in tc:
            for c in tg:
                if a >= b:
                    continue
                yield SweepCase(a, b, c)


def main() -> int:
    os.environ.setdefault("BEMPP_DEVICE_INTERFACE", "numba")

    from bempp_audio import Loudspeaker
    from bempp_audio.mesh import WaveguideMeshConfig, mouth_diffraction_proxy
    from bempp_audio.viz import compute_design_scorecard, save_scorecard_json, save_driver_dashboard_html
    from bempp_audio.results import BeamwidthTarget, DirectivityObjectiveConfig, evaluate_directivity_objective

    out_root = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs")).resolve()
    sweep_name = os.environ.get("BEMPPAUDIO_SWEEP_NAME", "cts_sweep")
    base_dir = out_root / "sweeps" / sweep_name
    base_dir.mkdir(parents=True, exist_ok=True)

    mesh_preset = os.environ.get("BEMPPAUDIO_MESH_PRESET", "ultra-fast").strip()
    perf_mode = os.environ.get("BEMPPAUDIO_PERF_MODE", "horn").strip()
    n_workers = int(os.environ.get("BEMPPAUDIO_N_WORKERS", "1").strip())

    throat_d = float(os.environ.get("BEMPPAUDIO_THROAT_D", "0.0254"))
    mouth_d = float(os.environ.get("BEMPPAUDIO_MOUTH_W", "0.300"))  # axisymmetric diameter
    length = float(os.environ.get("BEMPPAUDIO_WG_LEN", "0.070"))
    driver_exit_angle = float(os.environ.get("BEMPPAUDIO_DRIVER_EXIT_DEG", "10.0"))

    for i, case in enumerate(_cases(), start=1):
        case_dir = base_dir / f"case_{i:03d}_{case.tag()}"
        case_dir.mkdir(parents=True, exist_ok=True)

        scorecard_path = case_dir / "scorecard.json"
        if scorecard_path.exists():
            continue

        # Pre-solve proxy metrics (cheap).
        cfg = WaveguideMeshConfig(
            throat_diameter=throat_d,
            mouth_diameter=mouth_d,
            length=length,
            profile_type="cts",
            cts_throat_blend=case.throat_blend,
            cts_transition=case.transition,
            cts_tangency=case.tangency,
            cts_driver_exit_angle_deg=driver_exit_angle,
        )
        proxy = mouth_diffraction_proxy(cfg)
        (case_dir / "proxy.json").write_text(json.dumps(asdict(proxy), indent=2) + "\n")

        speaker = (
            Loudspeaker()
            .performance_preset(mesh_preset, mode=perf_mode)
            .waveguide(
                throat_diameter=throat_d,
                mouth_diameter=mouth_d,
                length=length,
                profile="cts",
                mesh_preset=mesh_preset,
                cts_throat_blend=case.throat_blend,
                cts_transition=case.transition,
                cts_tangency=case.tangency,
                cts_driver_exit_angle_deg=driver_exit_angle,
                throat_edge_refine=True,
            )
            .infinite_baffle()
        )

        # Persist snapshot for reproducibility.
        speaker.to_snapshot_json(case_dir / "snapshot.json", include_mesh_data=False)

        response = speaker.solve(n_workers=n_workers)

        # Directivity objective (optional; defaults target to 90x60 if env vars are set).
        try:
            tgt_h = float(os.environ.get("BEMPPAUDIO_TARGET_BW_H_DEG", "90.0"))
            tgt_v = float(os.environ.get("BEMPPAUDIO_TARGET_BW_V_DEG", "60.0"))
            cfg_obj = DirectivityObjectiveConfig(
                f_lo_hz=float(os.environ.get("BEMPPAUDIO_OBJ_F_LO", "1000")),
                f_hi_hz=float(os.environ.get("BEMPPAUDIO_OBJ_F_HI", "16000")),
                target_h=BeamwidthTarget(kind="constant", value_deg=tgt_h),
                target_v=BeamwidthTarget(kind="constant", value_deg=tgt_v),
            )
            obj = evaluate_directivity_objective(response.results, cfg=cfg_obj, plane_h="xz", plane_v="yz")
            (case_dir / "directivity_objective.json").write_text(json.dumps(asdict(obj), indent=2) + "\n")
        except Exception:
            pass

        sc = compute_design_scorecard(response, title=f"{sweep_name} {case.tag()}", show_progress=False)
        save_scorecard_json(sc, scorecard_path)

        try:
            save_driver_dashboard_html(
                response,
                filename=str(case_dir / "dashboard.html"),
                title=f"{sweep_name}: {case.tag()}",
                max_angle=90.0,
                normalize_angle=10.0,
                show_progress=False,
            )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
