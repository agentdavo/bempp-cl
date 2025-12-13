"""
Parser for "generic25.txt"-style compression driver + cavity definitions.

The format resembles AkAbak-style blocks defining:
- a lumped electro-mechanical driver
- rear volume, front duct, waveguide sections

This module focuses on robust parsing + unit conversion to SI, so the
resulting config can be used by a future acoustic network model and by the
fluent API (JSON-serializable dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re

from bempp_audio.driver.compression_config import (
    CompressionDriverConfig,
    DriverElectroMechConfig,
    RearVolumeConfig,
    FrontDuctConfig,
    PhasePlugConfig,
    ExitConeConfig,
)

_UNIT_SCALE = {
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
    "cm2": 1e-4,
    "m2": 1.0,
    "cm3": 1e-6,
    "m3": 1.0,
    "g": 1e-3,
    "kg": 1.0,
    "mH": 1e-3,
    "H": 1.0,
    "Hz": 1.0,
    "kHz": 1e3,
    "ohm": 1.0,
    "Tm": 1.0,
    "Ns/m": 1.0,
    "m/N": 1.0,
}


def _parse_number_with_unit(raw: str) -> float:
    raw = raw.strip()
    m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z0-9/]+)?", raw)
    if not m:
        raise ValueError(f"Could not parse value '{raw}'")
    value = float(m.group(1))
    unit = m.group(2)
    if unit is None:
        return value
    if unit not in _UNIT_SCALE:
        raise ValueError(f"Unknown unit '{unit}' in '{raw}'")
    return value * _UNIT_SCALE[unit]


def _tokenize_kv_and_words(line: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse tokens like `a=1 b=2 Conical` into ({'a':'1','b':'2'}, ['Conical']).
    """
    kv: Dict[str, str] = {}
    words: List[str] = []
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            kv[k.strip()] = v.strip()
        else:
            words.append(token.strip())
    return kv, words


@dataclass(frozen=True)
class Generic25DriverConfig:
    name: str
    diaphragm_diameter_m: float
    mms_kg: float
    cms_m_per_n: float
    rms_ns_per_m: float
    bl_tm: float
    re_ohm: float
    le_h: float
    fre_hz: Optional[float] = None
    expo_re: Optional[float] = None
    expo_le: Optional[float] = None


@dataclass(frozen=True)
class Generic25EnclosureConfig:
    name: str
    volume_m3: float
    qb_over_fo: Optional[float] = None


@dataclass(frozen=True)
class Generic25DuctConfig:
    name: str
    diameter_m: float
    length_m: float


@dataclass(frozen=True)
class Generic25WaveguideConfig:
    name: str
    length_m: float
    kind: str  # e.g. "Conical"
    throat_diameter_m: Optional[float] = None
    throat_area_m2: Optional[float] = None
    mouth_diameter_m: Optional[float] = None


@dataclass(frozen=True)
class Generic25SystemConfig:
    system_name: str
    driver: Generic25DriverConfig
    rear_volume: Optional[Generic25EnclosureConfig] = None
    front_duct: Optional[Generic25DuctConfig] = None
    waveguides: Optional[List[Generic25WaveguideConfig]] = None
    radimp_name: Optional[str] = None


def parse_generic25(text: str) -> Generic25SystemConfig:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("//"):
            continue
        lines.append(line)

    driver_defs: Dict[str, Dict[str, str]] = {}
    current_driver_def: Optional[str] = None
    system_name: Optional[str] = None
    driver_ref_name: Optional[str] = None

    rear_volume: Optional[Generic25EnclosureConfig] = None
    front_duct: Optional[Generic25DuctConfig] = None
    waveguides: List[Generic25WaveguideConfig] = []
    radimp_name: Optional[str] = None

    current_enclosure_name: Optional[str] = None
    current_duct_name: Optional[str] = None
    current_waveguide_name: Optional[str] = None

    for line in lines:
        if line.startswith("Def_Driver"):
            m = re.search(r"'([^']+)'", line)
            if not m:
                raise ValueError(f"Invalid Def_Driver line: {line}")
            current_driver_def = m.group(1)
            driver_defs[current_driver_def] = {}
            continue

        if line.startswith("System"):
            m = re.search(r"'([^']+)'", line)
            if not m:
                raise ValueError(f"Invalid System line: {line}")
            system_name = m.group(1)
            current_driver_def = None
            continue

        if line.startswith("Driver"):
            # Example: Driver 'D1' Def='Drv1' Node=1=0=10=20
            m = re.search(r"Def='([^']+)'", line)
            if m:
                driver_ref_name = m.group(1)
            continue

        if current_driver_def is not None:
            kv, _ = _tokenize_kv_and_words(line)
            driver_defs[current_driver_def].update(kv)
            continue

        if line.startswith("Enclosure"):
            m = re.search(r"'([^']+)'", line)
            name = m.group(1) if m else "Enclosure"
            current_enclosure_name = name
            current_duct_name = None
            current_waveguide_name = None
            rear_volume = Generic25EnclosureConfig(name=name, volume_m3=0.0, qb_over_fo=None)
            continue

        if line.startswith("Duct"):
            m = re.search(r"'([^']+)'", line)
            name = m.group(1) if m else "Duct"
            current_duct_name = name
            current_enclosure_name = None
            current_waveguide_name = None
            front_duct = Generic25DuctConfig(name=name, diameter_m=0.0, length_m=0.0)
            continue

        if line.startswith("Waveguide"):
            m = re.search(r"'([^']+)'", line)
            name = m.group(1) if m else "Waveguide"
            current_waveguide_name = name
            current_enclosure_name = None
            current_duct_name = None
            waveguides.append(Generic25WaveguideConfig(name=name, length_m=0.0, kind="Conical"))
            continue

        if line.startswith("RadImp"):
            m = re.search(r"'([^']+)'", line)
            if m:
                radimp_name = m.group(1)
            continue

        # Parameter continuation lines for Enclosure, etc.
        if current_enclosure_name is not None and rear_volume is not None and ("Vb=" in line or "Qb/fo=" in line):
            kv, _ = _tokenize_kv_and_words(line)
            vol = rear_volume.volume_m3
            q = rear_volume.qb_over_fo
            if "Vb" in kv:
                vol = _parse_number_with_unit(kv["Vb"])
            if "Qb/fo" in kv:
                q = float(kv["Qb/fo"])
            rear_volume = Generic25EnclosureConfig(name=rear_volume.name, volume_m3=vol, qb_over_fo=q)
            continue

        if current_duct_name is not None and front_duct is not None and ("dD=" in line or "Len=" in line):
            kv, _ = _tokenize_kv_and_words(line)
            d_m = front_duct.diameter_m
            l_m = front_duct.length_m
            if "dD" in kv:
                d_m = _parse_number_with_unit(kv["dD"])
            if "Len" in kv:
                l_m = _parse_number_with_unit(kv["Len"])
            front_duct = Generic25DuctConfig(name=front_duct.name, diameter_m=d_m, length_m=l_m)
            continue

        if current_waveguide_name is not None and waveguides and ("Len=" in line or "dMo=" in line or "dTh=" in line or "STh=" in line):
            kv, words = _tokenize_kv_and_words(line)
            # The kind is often the last word on this continuation line (e.g. "Conical")
            kind = words[-1] if words else waveguides[-1].kind
            length_m = waveguides[-1].length_m
            throat_area_m2 = waveguides[-1].throat_area_m2
            throat_diameter_m = waveguides[-1].throat_diameter_m
            mouth_diameter_m = waveguides[-1].mouth_diameter_m

            if "Len" in kv:
                length_m = _parse_number_with_unit(kv["Len"])
            if "STh" in kv:
                throat_area_m2 = _parse_number_with_unit(kv["STh"])
            if "dTh" in kv:
                throat_diameter_m = _parse_number_with_unit(kv["dTh"])
            if "dMo" in kv:
                mouth_diameter_m = _parse_number_with_unit(kv["dMo"])

            waveguides[-1] = Generic25WaveguideConfig(
                name=waveguides[-1].name,
                length_m=length_m,
                kind=kind,
                throat_diameter_m=throat_diameter_m,
                throat_area_m2=throat_area_m2,
                mouth_diameter_m=mouth_diameter_m,
            )
            continue

    if system_name is None:
        raise ValueError("No System block found")
    if driver_ref_name is None:
        raise ValueError("No Driver reference found in System block")
    if driver_ref_name not in driver_defs:
        raise ValueError(f"Driver def '{driver_ref_name}' not found")

    d = driver_defs[driver_ref_name]
    required = ["dD", "Mms", "Cms", "Rms", "Bl", "Re", "Le"]
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"Missing driver parameters: {missing}")

    driver = Generic25DriverConfig(
        name=driver_ref_name,
        diaphragm_diameter_m=_parse_number_with_unit(d["dD"]),
        mms_kg=_parse_number_with_unit(d["Mms"]),
        cms_m_per_n=_parse_number_with_unit(d["Cms"]),
        rms_ns_per_m=_parse_number_with_unit(d["Rms"]),
        bl_tm=_parse_number_with_unit(d["Bl"]),
        re_ohm=_parse_number_with_unit(d["Re"]),
        le_h=_parse_number_with_unit(d["Le"]),
        fre_hz=_parse_number_with_unit(d["fre"]) if "fre" in d else None,
        expo_re=float(d["ExpoRe"]) if "ExpoRe" in d else None,
        expo_le=float(d["ExpoLe"]) if "ExpoLe" in d else None,
    )

    return Generic25SystemConfig(
        system_name=system_name,
        driver=driver,
        rear_volume=rear_volume if rear_volume and rear_volume.volume_m3 > 0 else None,
        front_duct=front_duct if front_duct and front_duct.length_m > 0 else None,
        waveguides=waveguides or None,
        radimp_name=radimp_name,
    )


def load_generic25(path: str | Path) -> Generic25SystemConfig:
    path = Path(path)
    return parse_generic25(path.read_text(encoding="utf-8", errors="replace"))


def generic25_to_compression_driver_config(system: Generic25SystemConfig) -> CompressionDriverConfig:
    """
    Convert parsed generic25 system into the structured `CompressionDriverConfig`.
    """
    d = system.driver
    driver_cfg = DriverElectroMechConfig(
        diaphragm_diameter_m=d.diaphragm_diameter_m,
        mms_kg=d.mms_kg,
        cms_m_per_n=d.cms_m_per_n,
        rms_ns_per_m=d.rms_ns_per_m,
        bl_tm=d.bl_tm,
        re_ohm=d.re_ohm,
        le_h=d.le_h,
    )

    rear = RearVolumeConfig(volume_m3=system.rear_volume.volume_m3) if system.rear_volume else None
    duct = (
        FrontDuctConfig(diameter_m=system.front_duct.diameter_m, length_m=system.front_duct.length_m)
        if system.front_duct
        else None
    )

    phase_plug = None
    exit_cone = None
    if system.waveguides:
        if len(system.waveguides) >= 1:
            w1 = system.waveguides[0]
            phase_plug = PhasePlugConfig(
                throat_area_m2=w1.throat_area_m2,
                mouth_diameter_m=w1.mouth_diameter_m or 0.0,
                length_m=w1.length_m,
                kind=w1.kind,
            )
        if len(system.waveguides) >= 2:
            w2 = system.waveguides[1]
            exit_cone = ExitConeConfig(
                throat_diameter_m=w2.throat_diameter_m or 0.0,
                mouth_diameter_m=w2.mouth_diameter_m or 0.0,
                length_m=w2.length_m,
                kind=w2.kind,
            )

    return CompressionDriverConfig(
        name=system.system_name,
        driver=driver_cfg,
        rear_volume=rear,
        front_duct=duct,
        phase_plug=phase_plug,
        exit_cone=exit_cone,
    )
