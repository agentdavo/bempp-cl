"""
Stable, versioned snapshot schema for the bempp_audio fluent API.

This module provides helpers to serialize/deserialize a `Loudspeaker` builder
state in a way intended for persistence and interoperability (as opposed to
internal multiprocessing dicts).
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional
import numpy as np

from bempp_audio.acoustic_reference import AcousticReference
from bempp_audio.baffles import FreeSpace, InfiniteBaffle, CircularBaffle, Baffle
from bempp_audio.config import ExecutionConfig


SNAPSHOT_SCHEMA = "bempp_audio.LoudspeakerSnapshot"
SNAPSHOT_VERSION = 1


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return _jsonify(value.to_dict())
    return repr(value)


def serialize_baffle(baffle: Baffle) -> Dict[str, Any]:
    if isinstance(baffle, FreeSpace):
        return {"type": "FreeSpace"}
    if isinstance(baffle, InfiniteBaffle):
        return {"type": "InfiniteBaffle", "plane_z": float(baffle.plane_z), "pressure_scale": float(baffle.pressure_scale)}
    if isinstance(baffle, CircularBaffle):
        return {"type": "CircularBaffle", "radius": float(baffle.radius), "element_size": _jsonify(baffle.element_size)}
    return {"type": type(baffle).__name__, "repr": repr(baffle)}


def deserialize_baffle(data: Dict[str, Any]) -> Baffle:
    kind = str(data.get("type", "")).strip()
    if kind == "FreeSpace":
        return FreeSpace()
    if kind == "InfiniteBaffle":
        return InfiniteBaffle(
            plane_z=float(data.get("plane_z", 0.0)),
            pressure_scale=float(data.get("pressure_scale", 2.0)),
        )
    if kind == "CircularBaffle":
        return CircularBaffle(
            radius=float(data["radius"]),
            element_size=data.get("element_size", None),
        )
    raise ValueError(f"Unknown baffle type in snapshot: {kind!r}")


def serialize_reference(reference: Optional[AcousticReference]) -> Optional[Dict[str, Any]]:
    if reference is None:
        return None
    return {
        "origin": _jsonify(reference.origin),
        "axis": _jsonify(reference.axis),
        "default_distance_m": float(reference.default_distance_m),
    }


def deserialize_reference(data: Optional[Dict[str, Any]]) -> Optional[AcousticReference]:
    if data is None:
        return None
    return AcousticReference.from_origin_axis(
        origin=np.asarray(data["origin"], dtype=float),
        axis=np.asarray(data["axis"], dtype=float),
        default_distance_m=float(data["default_distance_m"]),
    )


def serialize_execution_config(cfg: Optional[object]) -> Optional[Dict[str, Any]]:
    if cfg is None:
        return None
    if isinstance(cfg, ExecutionConfig):
        return _jsonify(asdict(cfg))
    if is_dataclass(cfg):
        return _jsonify(asdict(cfg))
    if isinstance(cfg, dict):
        return _jsonify(cfg)
    return {"repr": repr(cfg)}

