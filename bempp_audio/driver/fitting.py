"""
Parameter fitting / identification utilities for compression-driver lumped networks.

This module is intentionally lightweight: it fits the existing network/validation
models to complex impedance data and reports basic identifiability metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal, Optional

import numpy as np

from bempp_audio.driver.compression_config import CompressionDriverConfig, DriverElectroMechConfig, VoiceCoilImpedanceModel
from bempp_audio.driver.network import AcousticMedium, CompressionDriverNetwork, CompressionDriverNetworkOptions
from bempp_audio.driver.validation import vacuum_electrical_impedance


TransformKind = Literal["linear", "log10"]


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    transform: TransformKind = "linear"
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass(frozen=True)
class FitDiagnostics:
    success: bool
    cost: float
    rms: float
    cond_jtj: float
    rank: int
    nfev: int
    message: str


@dataclass(frozen=True)
class FitResult:
    x: dict[str, float]
    covariance: Optional[np.ndarray]
    stderr: Optional[dict[str, float]]
    diagnostics: FitDiagnostics


def _transform_forward(kind: TransformKind, x: float) -> float:
    if kind == "log10":
        return float(np.log10(max(1e-300, float(x))))
    return float(x)


def _transform_inverse(kind: TransformKind, y: float) -> float:
    if kind == "log10":
        return float(10.0 ** float(y))
    return float(y)


def _pack_params(x: dict[str, float], specs: list[ParameterSpec]) -> np.ndarray:
    return np.array([_transform_forward(s.transform, float(x[s.path])) for s in specs], dtype=float)


def _unpack_params(v: np.ndarray, specs: list[ParameterSpec]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s, y in zip(specs, v.tolist()):
        out[s.path] = _transform_inverse(s.transform, float(y))
    return out


def _bounds(specs: list[ParameterSpec]) -> tuple[np.ndarray, np.ndarray]:
    lo: list[float] = []
    hi: list[float] = []
    for s in specs:
        if s.lower is None:
            lo.append(-np.inf)
        else:
            lo.append(_transform_forward(s.transform, float(s.lower)))
        if s.upper is None:
            hi.append(np.inf)
        else:
            hi.append(_transform_forward(s.transform, float(s.upper)))
    return np.array(lo, dtype=float), np.array(hi, dtype=float)


def _covariance_from_jacobian(
    jac: np.ndarray,
    residual: np.ndarray,
    dof: int,
) -> tuple[Optional[np.ndarray], float, int]:
    """
    Estimate covariance = s^2 * (J^T J)^{-1}.
    Returns (cov, cond(JTJ), rank(J)).
    """
    if jac.size == 0:
        return None, np.inf, 0
    try:
        jtj = jac.T @ jac
        rank = int(np.linalg.matrix_rank(jtj))
        cond = float(np.linalg.cond(jtj)) if jtj.size else np.inf
        if dof <= 0 or rank < jtj.shape[0]:
            return None, cond, rank
        s2 = float((residual @ residual) / float(dof))
        cov = s2 * np.linalg.inv(jtj)
        return cov, cond, rank
    except Exception:
        return None, np.inf, 0


def fit_vacuum_impedance(
    *,
    frequencies_hz: np.ndarray,
    z_measured_ohm: np.ndarray,
    initial: dict[str, float],
    fit_eddy: bool = True,
    robust_loss: Literal["linear", "soft_l1", "huber"] = "soft_l1",
) -> FitResult:
    """
    Fit vacuum electro-mechanical parameters to complex electrical impedance.

    Parameters fit (always):
      - re_ohm, le_h, bl_tm, mms_kg, cms_m_per_n, rms_ns_per_m
    Optional eddy loss parameters:
      - voice_coil_eddy_rmax_ohm, voice_coil_eddy_fcorner_hz
    """
    from scipy.optimize import least_squares  # type: ignore

    freqs = np.asarray(frequencies_hz, dtype=float)
    z_meas = np.asarray(z_measured_ohm, dtype=complex)
    if freqs.shape != z_meas.shape:
        raise ValueError("frequencies_hz and z_measured_ohm must have the same shape.")

    specs: list[ParameterSpec] = [
        ParameterSpec("re_ohm", "log10", lower=1e-4),
        ParameterSpec("le_h", "log10", lower=1e-9),
        ParameterSpec("bl_tm", "log10", lower=1e-6),
        ParameterSpec("mms_kg", "log10", lower=1e-6),
        ParameterSpec("cms_m_per_n", "log10", lower=1e-12),
        ParameterSpec("rms_ns_per_m", "log10", lower=1e-6),
    ]
    if fit_eddy:
        specs.extend(
            [
                ParameterSpec("voice_coil_eddy_rmax_ohm", "log10", lower=0.0),
                ParameterSpec("voice_coil_eddy_fcorner_hz", "log10", lower=1.0, upper=1e6),
            ]
        )

    x0 = {s.path: float(initial.get(s.path, 0.0)) for s in specs}
    v0 = _pack_params(x0, specs)
    lo, hi = _bounds(specs)

    def _model(params: dict[str, float]) -> np.ndarray:
        z_hat = np.array(
            [
                vacuum_electrical_impedance(
                    frequency_hz=float(f),
                    re_ohm=float(params.get("re_ohm", 0.0)),
                    le_h=float(params.get("le_h", 0.0)),
                    bl_tm=float(params.get("bl_tm", 0.0)),
                    mms_kg=float(params.get("mms_kg", 0.0)),
                    cms_m_per_n=float(params.get("cms_m_per_n", 0.0)),
                    rms_ns_per_m=float(params.get("rms_ns_per_m", 0.0)),
                    voice_coil_eddy_rmax_ohm=float(params.get("voice_coil_eddy_rmax_ohm", 0.0)),
                    voice_coil_eddy_fcorner_hz=float(params.get("voice_coil_eddy_fcorner_hz", 1000.0)),
                )
                for f in freqs
            ],
            dtype=complex,
        )
        return z_hat

    def _resid(v: np.ndarray) -> np.ndarray:
        params = _unpack_params(v, specs)
        z_hat = _model(params)
        # Stack Re/Im residuals for stable fitting.
        r = np.concatenate([np.real(z_hat - z_meas), np.imag(z_hat - z_meas)])
        return r

    res = least_squares(_resid, x0=v0, bounds=(lo, hi), loss=robust_loss)
    x_hat = _unpack_params(res.x, specs)
    r = _resid(res.x)
    dof = int(max(1, r.size - res.x.size))
    cov, cond, rank = _covariance_from_jacobian(res.jac, r, dof)

    stderr = None
    if cov is not None:
        stderr = {s.path: float(np.sqrt(max(0.0, cov[i, i]))) for i, s in enumerate(specs)}

    diagnostics = FitDiagnostics(
        success=bool(res.success),
        cost=float(res.cost),
        rms=float(np.sqrt(np.mean(r * r))),
        cond_jtj=float(cond),
        rank=int(rank),
        nfev=int(res.nfev),
        message=str(res.message),
    )
    return FitResult(x=x_hat, covariance=cov, stderr=stderr, diagnostics=diagnostics)


def _replace_nested_dataclass(obj, path: str, value: float):
    parts = path.split(".")
    if not parts:
        raise ValueError("Empty parameter path.")
    if len(parts) == 1:
        return replace(obj, **{parts[0]: value})
    head, rest = parts[0], ".".join(parts[1:])
    child = getattr(obj, head)
    if child is None:
        raise ValueError(f"Cannot set nested path '{path}': '{head}' is None.")
    new_child = _replace_nested_dataclass(child, rest, value)
    return replace(obj, **{head: new_child})


def fit_network_impedance(
    *,
    frequencies_hz: np.ndarray,
    z_measured_ohm: np.ndarray,
    config: CompressionDriverConfig,
    specs: list[ParameterSpec],
    z_external: complex | Callable[[float], complex],
    medium: AcousticMedium = AcousticMedium(),
    options: CompressionDriverNetworkOptions = CompressionDriverNetworkOptions(),
    robust_loss: Literal["linear", "soft_l1", "huber"] = "soft_l1",
) -> tuple[CompressionDriverConfig, FitResult]:
    """
    Fit arbitrary `CompressionDriverConfig` parameters to complex electrical impedance.

    This is intended for identifying acoustic sub-network parameters from tube/termination
    measurements once the vacuum electro-mechanical parameters are known.
    """
    from scipy.optimize import least_squares  # type: ignore

    freqs = np.asarray(frequencies_hz, dtype=float)
    z_meas = np.asarray(z_measured_ohm, dtype=complex)
    if freqs.shape != z_meas.shape:
        raise ValueError("frequencies_hz and z_measured_ohm must have the same shape.")

    # Build initial parameter vector from current config values.
    def _get(cfg: CompressionDriverConfig, path: str) -> float:
        cur = cfg
        for p in path.split("."):
            cur = getattr(cur, p)
        return float(cur)

    x0 = {s.path: _get(config, s.path) for s in specs}
    v0 = _pack_params(x0, specs)
    lo, hi = _bounds(specs)

    def _cfg_from_params(v: np.ndarray) -> CompressionDriverConfig:
        updates = _unpack_params(v, specs)
        cfg = config
        for k, val in updates.items():
            cfg = _replace_nested_dataclass(cfg, k, float(val))
        return cfg

    def _resid(v: np.ndarray) -> np.ndarray:
        cfg = _cfg_from_params(v)
        net = CompressionDriverNetwork(cfg, medium=medium, options=options)
        z_hat = np.array(
            [net.solve_with_metrics(float(f), z_external=z_external)["electrical_impedance"] for f in freqs],
            dtype=complex,
        )
        r = np.concatenate([np.real(z_hat - z_meas), np.imag(z_hat - z_meas)])
        return r

    res = least_squares(_resid, x0=v0, bounds=(lo, hi), loss=robust_loss)
    x_hat = _unpack_params(res.x, specs)
    r = _resid(res.x)
    dof = int(max(1, r.size - res.x.size))
    cov, cond, rank = _covariance_from_jacobian(res.jac, r, dof)
    stderr = None
    if cov is not None:
        stderr = {s.path: float(np.sqrt(max(0.0, cov[i, i]))) for i, s in enumerate(specs)}
    diagnostics = FitDiagnostics(
        success=bool(res.success),
        cost=float(res.cost),
        rms=float(np.sqrt(np.mean(r * r))),
        cond_jtj=float(cond),
        rank=int(rank),
        nfev=int(res.nfev),
        message=str(res.message),
    )
    cfg_hat = _cfg_from_params(res.x)
    return cfg_hat, FitResult(x=x_hat, covariance=cov, stderr=stderr, diagnostics=diagnostics)


def fit_acoustic_from_tube_impedance(
    *,
    frequencies_hz: np.ndarray,
    z_measured_ohm: np.ndarray,
    config: CompressionDriverConfig,
    z_external: complex | Callable[[float], complex],
    medium: AcousticMedium = AcousticMedium(),
    options: CompressionDriverNetworkOptions = CompressionDriverNetworkOptions(),
) -> tuple[CompressionDriverConfig, FitResult]:
    """
    Convenience wrapper to fit common acoustic parameters against tube/termination Z_elec.

    Fits (when present / non-None):
      - rear_volume.volume_m3
      - front_volume_m3
      - voice_coil_slit_resistance_pa_s_per_m3
      - voice_coil_slit_mass_pa_s2_per_m3
    """
    specs: list[ParameterSpec] = []
    if config.rear_volume is not None and float(config.rear_volume.volume_m3) > 0:
        specs.append(ParameterSpec("rear_volume.volume_m3", "log10", lower=1e-9, upper=1e-2))
    if config.front_volume_m3 is not None and float(config.front_volume_m3) > 0:
        specs.append(ParameterSpec("front_volume_m3", "log10", lower=1e-9, upper=1e-2))
    if config.voice_coil_slit_resistance_pa_s_per_m3 is not None and float(config.voice_coil_slit_resistance_pa_s_per_m3) > 0:
        specs.append(ParameterSpec("voice_coil_slit_resistance_pa_s_per_m3", "log10", lower=1e2, upper=1e10))
    if config.voice_coil_slit_mass_pa_s2_per_m3 is not None and float(config.voice_coil_slit_mass_pa_s2_per_m3) >= 0:
        specs.append(ParameterSpec("voice_coil_slit_mass_pa_s2_per_m3", "log10", lower=1e-6, upper=1e8))
    if not specs:
        raise ValueError("No acoustic parameters available to fit in the provided config.")
    return fit_network_impedance(
        frequencies_hz=frequencies_hz,
        z_measured_ohm=z_measured_ohm,
        config=config,
        specs=specs,
        z_external=z_external,
        medium=medium,
        options=options,
    )


def apply_vacuum_fit_to_config(
    cfg: CompressionDriverConfig,
    fit: FitResult,
    *,
    enable_eddy: bool = True,
) -> CompressionDriverConfig:
    """
    Apply a vacuum fit result to a `CompressionDriverConfig` (returns a new config).
    """
    x = fit.x
    vc_model = None
    if enable_eddy and ("voice_coil_eddy_rmax_ohm" in x or "voice_coil_eddy_fcorner_hz" in x):
        vc_model = VoiceCoilImpedanceModel(
            kind="EddyLoss",
            r_eddy_max_ohm=float(x.get("voice_coil_eddy_rmax_ohm", 0.0)),
            f_corner_hz=float(x.get("voice_coil_eddy_fcorner_hz", 1000.0)),
        )
    driver = replace(
        cfg.driver,
        re_ohm=float(x.get("re_ohm", cfg.driver.re_ohm)),
        le_h=float(x.get("le_h", cfg.driver.le_h)),
        bl_tm=float(x.get("bl_tm", cfg.driver.bl_tm)),
        mms_kg=float(x.get("mms_kg", cfg.driver.mms_kg)),
        cms_m_per_n=float(x.get("cms_m_per_n", cfg.driver.cms_m_per_n)),
        rms_ns_per_m=float(x.get("rms_ns_per_m", cfg.driver.rms_ns_per_m)),
        voice_coil_model=vc_model if vc_model is not None else cfg.driver.voice_coil_model,
    )
    return replace(cfg, driver=driver)

