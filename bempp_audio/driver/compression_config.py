"""
Structured configuration for compression driver lumped-element networks.

This provides a stable fluent/config API independent of any particular
external file format (e.g. generic25.txt).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Literal, List, Tuple


WaveguideKind = Literal["Conical", "Exponential", "Tractrix", "Hyperbolic"]


VoiceCoilImpedanceKind = Literal["Ideal", "EddyLoss"]

RearRadiationMode = Literal["None", "InfiniteBaffle", "FreeSpaceApprox"]
FrontRadiationMode = Literal["None", "BEMLoad", "FreeRadiation"]
VoiceCoilSlitModel = Literal["Lumped", "Tube"]
WaveguideMatrixMethod = Literal["CascadedTubes", "AnalyticalConical"]


@dataclass(frozen=True)
class FrequencyWeighting:
    """
    Simple frequency-dependent gain curve for post-processing outputs.

    The gain is specified as a set of (frequency_hz, gain_db) knots and is
    interpolated linearly in log-frequency between knots.
    """

    knots_hz_db: List[Tuple[float, float]]


@dataclass(frozen=True)
class VoiceCoilImpedanceModel:
    """
    Electrical impedance model for the voice coil.

    Default behavior is the ideal series model: Z = Re + jωLe.

    The optional 'EddyLoss' model adds a frequency-dependent series resistance:
        R_eddy(f) = R_eddy_max * (x^2 / (1 + x^2)), where x = f / f_corner
    which transitions from ~0 at low frequency to ~R_eddy_max at high frequency.
    """

    kind: VoiceCoilImpedanceKind = "Ideal"
    r_eddy_max_ohm: float = 0.0
    f_corner_hz: float = 1000.0


@dataclass(frozen=True)
class DriverElectroMechConfig:
    """Electro-mechanical moving system parameters (SI units)."""

    diaphragm_diameter_m: float
    mms_kg: float
    cms_m_per_n: float
    rms_ns_per_m: float
    bl_tm: float
    re_ohm: float
    le_h: float
    voice_coil_model: Optional[VoiceCoilImpedanceModel] = None


@dataclass(frozen=True)
class RearVolumeConfig:
    """Rear (back) cavity volume."""

    volume_m3: float


@dataclass(frozen=True)
class FrontDuctConfig:
    """Front cavity / duct between diaphragm and phase plug entrance."""

    diameter_m: float
    length_m: float


@dataclass(frozen=True)
class PhasePlugConfig:
    """
    Simplified phase plug represented as a short waveguide section.

    Some tools specify the throat by area (STh) rather than diameter.
    """

    throat_area_m2: Optional[float]
    mouth_diameter_m: float
    length_m: float
    kind: WaveguideKind = "Conical"


@dataclass(frozen=True)
class ExitConeConfig:
    """Waveguide section from phase plug exit to driver exit (e.g. 1\" throat)."""

    throat_diameter_m: float
    mouth_diameter_m: float
    length_m: float
    kind: WaveguideKind = "Conical"


@dataclass(frozen=True)
class CompressionDriverConfig:
    """
    Full compression driver internal acoustic network configuration.

    This is intended to be coupled to BEM by using the BEM throat acoustic
    input impedance as the external termination.
    """

    name: str = "compression_driver"
    driver: DriverElectroMechConfig = None  # type: ignore[assignment]
    rear_volume: Optional[RearVolumeConfig] = None
    front_duct: Optional[FrontDuctConfig] = None
    phase_plug: Optional[PhasePlugConfig] = None
    exit_cone: Optional[ExitConeConfig] = None

    # Front compression chamber volume (V0) as an acoustic compliance shunt at the V0 node.
    front_volume_m3: Optional[float] = None

    # Optional suspension-side acoustic network (Panzer-style V1 coupling)
    suspension_volume_m3: Optional[float] = None
    suspension_diameter_m: Optional[float] = None
    # Voice-coil slit coupling from V1 -> V0 as either:
    # - 'Lumped': series (Ra + jωMa), in acoustic units.
    # - 'Tube': short duct two-port between nodes.
    voice_coil_slit_model: VoiceCoilSlitModel = "Lumped"
    voice_coil_slit_resistance_pa_s_per_m3: Optional[float] = None
    voice_coil_slit_mass_pa_s2_per_m3: Optional[float] = None
    voice_coil_slit_diameter_m: Optional[float] = None
    voice_coil_slit_length_m: Optional[float] = None

    # Optional rear radiation loading hooks (analytic first; BEM coupling can be added later).
    rear_radiation_mode: RearRadiationMode = "None"
    rear_radiator_diameter_m: Optional[float] = None

    # Optional front radiation mode for validation workflows.
    # - "None": Use external z_external (from BEM or user-supplied)
    # - "BEMLoad": Use BEM-computed throat impedance (default behavior)
    # - "FreeRadiation": Use analytic piston radiation impedance at front aperture
    #                    (Panzer validation mode - reveals internal resonances clearly)
    front_radiation_mode: FrontRadiationMode = "None"
    front_radiator_diameter_m: Optional[float] = None  # If None, uses exit/throat diameter

    # Optional output weighting hook (e.g. bending-mode magnitude correction).
    output_velocity_weighting: Optional[FrequencyWeighting] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressionDriverConfig":
        driver_data = dict(data["driver"])
        vc_model = None
        if driver_data.get("voice_coil_model"):
            vc_model = VoiceCoilImpedanceModel(**driver_data["voice_coil_model"])
        driver_data["voice_coil_model"] = vc_model
        driver = DriverElectroMechConfig(**driver_data)
        rear = RearVolumeConfig(**data["rear_volume"]) if data.get("rear_volume") else None
        duct = FrontDuctConfig(**data["front_duct"]) if data.get("front_duct") else None
        plug = PhasePlugConfig(**data["phase_plug"]) if data.get("phase_plug") else None
        exit_cone = ExitConeConfig(**data["exit_cone"]) if data.get("exit_cone") else None
        output_weighting = None
        if data.get("output_velocity_weighting"):
            output_weighting = FrequencyWeighting(**data["output_velocity_weighting"])
        return cls(
            name=data.get("name", "compression_driver"),
            driver=driver,
            rear_volume=rear,
            front_duct=duct,
            phase_plug=plug,
            exit_cone=exit_cone,
            front_volume_m3=data.get("front_volume_m3"),
            suspension_volume_m3=data.get("suspension_volume_m3"),
            suspension_diameter_m=data.get("suspension_diameter_m"),
            voice_coil_slit_model=data.get("voice_coil_slit_model", "Lumped"),
            voice_coil_slit_resistance_pa_s_per_m3=data.get("voice_coil_slit_resistance_pa_s_per_m3"),
            voice_coil_slit_mass_pa_s2_per_m3=data.get("voice_coil_slit_mass_pa_s2_per_m3"),
            voice_coil_slit_diameter_m=data.get("voice_coil_slit_diameter_m"),
            voice_coil_slit_length_m=data.get("voice_coil_slit_length_m"),
            rear_radiation_mode=data.get("rear_radiation_mode", "None"),
            rear_radiator_diameter_m=data.get("rear_radiator_diameter_m"),
            front_radiation_mode=data.get("front_radiation_mode", "None"),
            front_radiator_diameter_m=data.get("front_radiator_diameter_m"),
            output_velocity_weighting=output_weighting,
        )
