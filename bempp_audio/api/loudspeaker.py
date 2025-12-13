"""
High-level fluent API for loudspeaker acoustic simulation.

Provides a simple, chainable interface for common loudspeaker simulation tasks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List
import numpy as np

from bempp_audio._optional import optional_import

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")

from bempp_audio.mesh import LoudspeakerMesh
from bempp_audio.mesh.domains import Domain
from bempp_audio.mesh.waveguide import WaveguideMeshConfig
from bempp_audio.mesh.unified_enclosure import UnifiedMeshConfig
from bempp_audio.velocity import VelocityProfile
from bempp_audio.solver import RadiationSolver, OSRCRadiationSolver
from bempp_audio.solver.base import SolverOptions, solve_frequencies_parallel
from bempp_audio.results import FrequencyResponse, RadiationResult
from bempp_audio.progress import get_logger
from bempp_audio.driver import load_generic25, generic25_to_compression_driver_config
from bempp_audio.driver.compression_config import CompressionDriverConfig
from bempp_audio.driver.network import (
    AcousticMedium,
    CompressionDriverExcitation,
    CompressionDriverNetwork,
    CompressionDriverNetworkOptions,
)
from bempp_audio.api.driver_coupling import solve_with_compression_driver_network
from bempp_audio.api import geometry as _geometry
from bempp_audio.api import presets as _presets
from bempp_audio.api import solve as _solve
from bempp_audio.api import boundary_conditions as _bc
from bempp_audio.api.state import LoudspeakerState, WaveguideMetadata
from bempp_audio.api.types import BCPolicyLike, FrequencySpacingLike, VelocityModeLike
from bempp_audio.api.request import SimulationRequest
from bempp_audio.api.snapshot import (
    SNAPSHOT_SCHEMA,
    SNAPSHOT_VERSION,
    _jsonify as _snapshot_jsonify,
    deserialize_baffle,
    deserialize_reference,
    serialize_baffle,
    serialize_execution_config,
    serialize_reference,
)


class Loudspeaker:
    """
    Fluent API/builder for loudspeaker acoustic simulation.

    Provides a chainable interface for defining geometry, velocity,
    and computing acoustic response using BEM.

    Examples
    --------
    Basic circular piston simulation:

    >>> from bempp_audio import Loudspeaker
    >>> result = (Loudspeaker()
    ...     .circular_piston(radius=0.05)
    ...     .infinite_baffle()
    ...     .velocity(mode='piston', amplitude=0.01)
    ...     .frequency_range(20, 20000)
    ...     .solve())
    >>> freqs, spl = result.on_axis_response(distance=1.0)

    Cone driver with parallel solving:

    >>> result = (Loudspeaker()
    ...     .cone(inner_r=0.019, outer_r=0.085, height=0.015)
    ...     .infinite_baffle()
    ...     .frequency_range(20, 5000, points_per_octave=6)
    ...     .solve(n_workers=4))
    """

    def __init__(self):
        object.__setattr__(self, "_state", LoudspeakerState())

    def clone(self) -> "Loudspeaker":
        """
        Return a new `Loudspeaker` instance with the same internal state.

        Notes
        -----
        This is a shallow clone: referenced objects (e.g. mesh, velocity profiles)
        are shared, but the `LoudspeakerState` container is immutable, so further
        fluent mutations on either instance will not affect the other.
        """
        other = type(self)()
        object.__setattr__(other, "_state", self.state)
        return other

    def with_defaults(self) -> "Loudspeaker":
        """
        Return a cloned builder with default velocity/frequencies filled in.

        This is a convenience to make defaulting explicit without mutating the
        original builder.
        """
        s = self.state
        other = self.clone()

        if s.velocity is None:
            other._with_state(velocity=VelocityProfile.piston(amplitude=1.0))
        if s.frequencies is None:
            other._with_state(frequencies=np.logspace(np.log10(200.0), np.log10(20000.0), 20))
        if s.mesh is not None and s.reference is None:
            other._with_state(reference=AcousticReference.from_mesh(s.mesh, default_distance_m=float(s.measurement_distance)))

        return other

    def build(self, *, strict: bool = False) -> SimulationRequest:
        """
        Build an immutable `SimulationRequest` snapshot.

        Parameters
        ----------
        strict : bool
            If True, require velocity and frequencies to be explicitly configured.
            If False, fill missing values with defaults in the returned request.
        """
        s = self.state
        if s.mesh is None:
            raise ValueError("Mesh not configured. Call circular_piston(), cone(), etc.")

        if strict:
            if s.velocity is None:
                raise ValueError("Velocity not configured. Call .velocity(...) or .velocity_profile(...).")
            if s.frequencies is None:
                raise ValueError("Frequencies not configured. Call .frequency_range(...), .frequencies(...), etc.")

        velocity = s.velocity or VelocityProfile.piston(amplitude=1.0)
        frequencies = s.frequencies
        if frequencies is None:
            frequencies = np.logspace(np.log10(200.0), np.log10(20000.0), 20)

        reference = s.reference or AcousticReference.from_mesh(
            s.mesh, default_distance_m=float(s.measurement_distance)
        )

        return SimulationRequest(
            mesh=s.mesh,
            velocity=velocity,
            frequencies=np.asarray(frequencies, dtype=float),
            c=float(s.c),
            rho=float(s.rho),
            baffle=s.baffle,
            solver_options=s.solver_options,
            use_osrc=bool(s.use_osrc),
            osrc_npade=int(s.osrc_npade),
            measurement_distance=float(s.measurement_distance),
            reference=reference,
            waveguide=s.waveguide,
            driver_network=s.driver_network,
            driver_excitation=s.driver_excitation,
            execution_config=s.execution_config,
        )

    def validate(self, *, strict: bool = False) -> list[str]:
        """
        Validate the current builder state.

        Parameters
        ----------
        strict : bool
            If True, require velocity and frequencies to be explicitly configured.
            If False, missing velocity/frequencies are allowed (they will default at solve time).
        """
        s = self.state
        errors: list[str] = []

        if s.mesh is None:
            errors.append("Mesh not configured.")

        if strict:
            if s.velocity is None:
                errors.append("Velocity not configured.")
            if s.frequencies is None:
                errors.append("Frequencies not configured.")

        if s.frequencies is not None:
            freqs = np.asarray(s.frequencies, dtype=float)
            if freqs.size == 0:
                errors.append("Frequencies array is empty.")
            if np.any(freqs <= 0):
                errors.append("Frequencies must be > 0 Hz.")

        # Cross-field consistency checks (when mesh is present).
        if s.mesh is not None:
            domain_ids = set(int(d) for d in np.unique(s.mesh.grid.domain_indices))

            # Circular baffle incompatibility is a common footgun.
            from bempp_audio.baffles import CircularBaffle

            if isinstance(s.baffle, CircularBaffle) and len(domain_ids) != 1:
                errors.append("Circular baffle cannot be combined with a multi-domain mesh.")

            # If using a by-domain velocity profile, validate domains and policy coverage.
            if s.velocity is not None:
                vdict = s.velocity.to_dict()
                if vdict.get("type") == "by_domain":
                    profiles = vdict.get("domain_profiles", {}) or {}
                    try:
                        used = set(int(k) for k in profiles.keys())
                    except Exception:
                        used = set()
                        errors.append("Domain-dependent velocity profile has non-integer domain keys.")

                    extra = sorted(used - domain_ids)
                    if extra:
                        errors.append(f"Velocity specifies unknown domain IDs: {extra}.")

                    if s.default_bc_policy == "error":
                        missing = sorted(domain_ids - used)
                        if missing:
                            errors.append(
                                f"default_bc_policy='error' requires explicit velocity assignment for domains: {missing}."
                            )

            # If domain_names are present, validate they reference existing domains.
            if s.domain_names:
                bad = sorted(int(k) for k in s.domain_names.keys() if int(k) not in domain_ids)
                if bad:
                    errors.append(f"domain_names references unknown domain IDs: {bad}.")

        return errors

    def require_ready(self, *, strict: bool = False) -> "Loudspeaker":
        """
        Raise a ValueError if the builder is not ready to solve.

        This is a convenience wrapper around `validate()`.
        """
        errors = self.validate(strict=strict)
        if errors:
            raise ValueError("Invalid Loudspeaker configuration:\n- " + "\n- ".join(errors))
        return self

    @property
    def state(self) -> LoudspeakerState:
        """Current immutable configuration/state snapshot."""
        return object.__getattribute__(self, "_state")

    def to_dict(
        self,
        *,
        include_mesh: bool = False,
        include_mesh_data: bool = False,
    ) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot of the current builder state.

        Notes
        -----
        - By default, mesh data (vertices/elements) is not included because it
          can be very large and `LoudspeakerMesh.to_dict()` returns numpy arrays.
        - Set `include_mesh=True` to include a mesh summary, and
          `include_mesh_data=True` to include full arrays (as nested lists).
        """

        def jsonify(value: Any) -> Any:
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Enum):
                return value.name
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [jsonify(v) for v in value]
            if isinstance(value, dict):
                return {str(k): jsonify(v) for k, v in value.items()}
            if is_dataclass(value):
                return jsonify(asdict(value))
            if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
                return jsonify(value.to_dict())
            # Fallback: keep it inspectable rather than failing JSON conversion.
            return repr(value)

        st = self.state
        out: Dict[str, Any] = {
            "c": float(st.c),
            "rho": float(st.rho),
            "frequencies": jsonify(st.frequencies) if st.frequencies is not None else None,
            "baffle": {"type": type(st.baffle).__name__, **jsonify(st.baffle)},
            "solver_options": jsonify(st.solver_options),
            "use_osrc": bool(st.use_osrc),
            "osrc_npade": int(st.osrc_npade),
            "polar_start": float(st.polar_start),
            "polar_end": float(st.polar_end),
            "polar_num": int(st.polar_num),
            "norm_angle": float(st.norm_angle),
            "measurement_distance": float(st.measurement_distance),
            "spl_angles": jsonify(st.spl_angles),
            "default_bc_policy": str(st.default_bc_policy),
            "domain_names": jsonify(st.domain_names),
            "waveguide": jsonify(st.waveguide),
            "execution_config": jsonify(st.execution_config),
        }

        if st.reference is not None:
            out["reference"] = {
                "origin": jsonify(st.reference.origin),
                "axis": jsonify(st.reference.axis),
                "default_distance_m": float(st.reference.default_distance_m),
            }
        else:
            out["reference"] = None

        out["velocity"] = jsonify(st.velocity.to_dict()) if st.velocity is not None else None

        if include_mesh and st.mesh is not None:
            info = st.mesh.info()
            mesh_out: Dict[str, Any] = {
                "summary": {
                    "n_vertices": int(info.n_vertices),
                    "n_elements": int(info.n_elements),
                    "min_edge_length": float(info.min_edge_length),
                    "max_edge_length": float(info.max_edge_length),
                    "mean_edge_length": float(info.mean_edge_length),
                },
                "domains": jsonify(sorted(set(int(d) for d in np.unique(st.mesh.grid.domain_indices)))),
            }
            if include_mesh_data:
                md = st.mesh.to_dict()
                mesh_out["data"] = {k: jsonify(v) for k, v in md.items()}
            out["mesh"] = mesh_out
        else:
            out["mesh"] = None

        return out

    def to_json(
        self,
        filepath: str | Path,
        *,
        include_mesh: bool = False,
        include_mesh_data: bool = False,
    ) -> None:
        """Write `to_dict()` output to a JSON file."""
        path = Path(filepath)
        data = self.to_dict(include_mesh=include_mesh, include_mesh_data=include_mesh_data)
        path.write_text(json.dumps(data, indent=2) + "\n")

    def to_snapshot(
        self,
        *,
        include_mesh_data: bool = False,
        include_mesh_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        Return a stable, versioned snapshot dict for persistence.

        Notes
        -----
        - This is intended as a long-lived schema (`schema`/`version` fields).
        - `include_mesh_data=True` requires the mesh to be serializable and can be large.
        """
        st = self.state
        out: Dict[str, Any] = {
            "schema": SNAPSHOT_SCHEMA,
            "version": int(SNAPSHOT_VERSION),
            "performance": {"mesh_preset": str(st.mesh_preset) if st.mesh_preset is not None else None},
            "medium": {"c": float(st.c), "rho": float(st.rho)},
            "frequency": {"frequencies": _snapshot_jsonify(st.frequencies) if st.frequencies is not None else None},
            "baffle": serialize_baffle(st.baffle),
            "solver": {
                "solver_options": _snapshot_jsonify(st.solver_options),
                "use_osrc": bool(st.use_osrc),
                "osrc_npade": int(st.osrc_npade),
            },
            "directivity": {
                "polar_start": float(st.polar_start),
                "polar_end": float(st.polar_end),
                "polar_num": int(st.polar_num),
                "norm_angle": float(st.norm_angle),
                "measurement_distance": float(st.measurement_distance),
                "spl_angles": _snapshot_jsonify(st.spl_angles),
            },
            "boundary_conditions": {
                "default_bc_policy": str(st.default_bc_policy),
                "domain_names": _snapshot_jsonify(st.domain_names),
                "velocity": _snapshot_jsonify(st.velocity.to_dict()) if st.velocity is not None else None,
            },
            "reference": serialize_reference(st.reference),
            "waveguide": _snapshot_jsonify(st.waveguide),
            "execution": serialize_execution_config(st.execution_config),
        }

        if st.mesh is None:
            out["mesh"] = None
            return out

        mesh_out: Dict[str, Any] = {"format": "bempp_audio.mesh.v1"}
        if include_mesh_summary:
            info = st.mesh.info()
            mesh_out["summary"] = {
                "n_vertices": int(info.n_vertices),
                "n_elements": int(info.n_elements),
                "min_edge_length": float(info.min_edge_length),
                "max_edge_length": float(info.max_edge_length),
                "mean_edge_length": float(info.mean_edge_length),
                "domain_ids": sorted(set(int(d) for d in np.unique(st.mesh.grid.domain_indices))),
            }
        if include_mesh_data:
            md = st.mesh.to_dict()
            mesh_out["data"] = {k: _snapshot_jsonify(v) for k, v in md.items()}
        out["mesh"] = mesh_out
        return out

    def to_snapshot_json(
        self,
        filepath: str | Path,
        *,
        include_mesh_data: bool = False,
        include_mesh_summary: bool = True,
    ) -> None:
        """Write `to_snapshot()` output to a JSON file."""
        path = Path(filepath)
        data = self.to_snapshot(include_mesh_data=include_mesh_data, include_mesh_summary=include_mesh_summary)
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "Loudspeaker":
        """
        Reconstruct a `Loudspeaker` builder from a snapshot dict.

        Notes
        -----
        If `snapshot["mesh"]["data"]` is present, this requires `bempp_cl.api`
        to be importable to reconstruct `LoudspeakerMesh`.
        """
        schema = snapshot.get("schema")
        version = snapshot.get("version")
        if schema != SNAPSHOT_SCHEMA:
            raise ValueError(f"Unsupported snapshot schema: {schema!r}")
        if int(version) != SNAPSHOT_VERSION:
            raise ValueError(f"Unsupported snapshot version: {version!r}")

        speaker = cls()

        perf = snapshot.get("performance", {}) or {}
        if perf.get("mesh_preset", None) is not None:
            speaker._with_state(mesh_preset=str(perf["mesh_preset"]))

        medium = snapshot.get("medium", {}) or {}
        speaker._with_state(c=float(medium.get("c", speaker.state.c)), rho=float(medium.get("rho", speaker.state.rho)))

        freq = snapshot.get("frequency", {}) or {}
        freqs = freq.get("frequencies", None)
        if freqs is not None:
            speaker._with_state(frequencies=np.asarray(freqs, dtype=float))

        speaker._with_state(baffle=deserialize_baffle(snapshot.get("baffle", {}) or {"type": "FreeSpace"}))

        solver = snapshot.get("solver", {}) or {}
        if "solver_options" in solver and solver["solver_options"] is not None:
            from bempp_audio.solver.base import SolverOptions

            speaker._with_state(solver_options=SolverOptions(**solver["solver_options"]))
        speaker._with_state(
            use_osrc=bool(solver.get("use_osrc", False)),
            osrc_npade=int(solver.get("osrc_npade", 2)),
        )

        directivity = snapshot.get("directivity", {}) or {}
        speaker._with_state(
            polar_start=float(directivity.get("polar_start", speaker.state.polar_start)),
            polar_end=float(directivity.get("polar_end", speaker.state.polar_end)),
            polar_num=int(directivity.get("polar_num", speaker.state.polar_num)),
            norm_angle=float(directivity.get("norm_angle", speaker.state.norm_angle)),
            measurement_distance=float(directivity.get("measurement_distance", speaker.state.measurement_distance)),
            spl_angles=directivity.get("spl_angles", None),
        )

        bc = snapshot.get("boundary_conditions", {}) or {}
        speaker._with_state(
            default_bc_policy=str(bc.get("default_bc_policy", speaker.state.default_bc_policy)),
            domain_names=bc.get("domain_names", None),
        )
        vel = bc.get("velocity", None)
        if vel is not None:
            speaker._with_state(velocity=VelocityProfile.from_dict(vel))

        speaker._with_state(reference=deserialize_reference(snapshot.get("reference", None)))

        # Waveguide metadata (if present)
        wg = snapshot.get("waveguide", None)
        if wg is not None:
            from bempp_audio.api.state import WaveguideMetadata

            speaker._with_state(waveguide=WaveguideMetadata(**wg))

        # Execution config (if present)
        exec_cfg = snapshot.get("execution", None)
        if isinstance(exec_cfg, dict) and exec_cfg:
            try:
                from bempp_audio.config import ExecutionConfig

                speaker._with_state(execution_config=ExecutionConfig(**exec_cfg))
            except Exception:
                speaker._with_state(execution_config=exec_cfg)

        # Mesh reconstruction (optional)
        mesh = snapshot.get("mesh", None)
        if isinstance(mesh, dict) and isinstance(mesh.get("data", None), dict):
            from bempp_audio.mesh import LoudspeakerMesh

            md = mesh["data"]
            # LoudspeakerMesh.from_dict expects numpy arrays (vertices/elements/domain_indices).
            speaker._with_state(
                mesh=LoudspeakerMesh.from_dict(
                    {
                        "vertices": np.asarray(md["vertices"]),
                        "elements": np.asarray(md["elements"]),
                        "domain_indices": np.asarray(md.get("domain_indices", None)) if md.get("domain_indices", None) is not None else None,
                        "center": np.asarray(md.get("center", [0.0, 0.0, 0.0]), dtype=float),
                        "axis": np.asarray(md.get("axis", [0.0, 0.0, 1.0]), dtype=float),
                    }
                )
            )

        return speaker

    @classmethod
    def from_snapshot_json(cls, filepath: str | Path) -> "Loudspeaker":
        """Load a snapshot JSON file produced by `to_snapshot_json()`."""
        path = Path(filepath)
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Snapshot JSON must contain an object at the top level.")
        return cls.from_snapshot(data)

    def export_mesh_html(
        self,
        filepath: str | Path = "mesh_3d.html",
        *,
        title: str | None = None,
        color_by_domain: bool = True,
        show_edges: bool = True,
        opacity: float = 0.95,
        domain_colors: dict | None = None,
        domain_names: dict | None = None,
    ) -> str:
        """
        Export the currently configured mesh as interactive 3D HTML.

        This is the explicit, parameterized alternative to `export_mesh=True`
        flags used in some geometry builders.
        """
        if self.state.mesh is None:
            raise ValueError("Mesh not configured.")

        try:
            from bempp_audio.viz import mesh_3d_html
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Interactive mesh export requires bempp_audio.viz (plotly). "
                "Install with `pip install bempp-cl[audio]`."
            ) from e

        if title is None:
            title = "Mesh Geometry"

        if domain_names is None:
            domain_names = {}
            if self.state.domain_names:
                domain_names.update({int(k): str(v) for k, v in self.state.domain_names.items()})
            if self.state.waveguide is not None:
                domain_names.setdefault(int(self.state.waveguide.throat_domain), "throat")
                domain_names.setdefault(int(self.state.waveguide.wall_domain), "walls")

        out_path = str(Path(filepath))
        return mesh_3d_html(
            self.state.mesh,
            filename=out_path,
            title=title,
            color_by_domain=color_by_domain,
            show_edges=show_edges,
            opacity=float(opacity),
            domain_colors=domain_colors,
            domain_names=domain_names or None,
        )

    def _with_state(self, **kwargs) -> "Loudspeaker":
        """Internal helper: replace immutable state and return `self`."""
        object.__setattr__(self, "_state", replace(self.state, **kwargs))
        return self

    @classmethod
    def from_config(cls, config: "SimulationConfig") -> "Loudspeaker":
        """
        Create a `Loudspeaker` instance from a `SimulationConfig`.

        Notes
        -----
        This sets medium, frequency, directivity, solver, and execution defaults.
        Geometry/mesh and velocity boundary conditions are still configured via
        fluent geometry/velocity methods.
        """
        speaker = cls()
        speaker.apply_config(config)
        return speaker

    def apply_config(self, config: "SimulationConfig") -> "Loudspeaker":
        """
        Apply a `SimulationConfig` to this instance (excluding geometry/mesh).
        """
        errors = config.validate()
        if errors:
            raise ValueError("Invalid SimulationConfig:\n- " + "\n- ".join(errors))

        # Medium
        c = config.medium.compute_speed_of_sound()
        rho = float(config.medium.rho)

        # Frequencies
        freqs = config.frequency.to_array()

        # Directivity
        polar_start = float(config.directivity.polar_start)
        polar_end = float(config.directivity.polar_end)
        polar_num = int(config.directivity.polar_num)
        norm_angle = float(config.directivity.normalize_angle)
        measurement_distance = float(config.directivity.measurement_distance)
        spl_angles = list(config.directivity.spl_angles) if config.directivity.spl_angles else None

        # Solver
        solver_options = replace(
            self.state.solver_options,
            tol=float(config.solver.tol),
            maxiter=int(config.solver.maxiter),
            space_type=str(config.solver.space_type),
            space_order=int(config.solver.space_order),
            use_fmm=bool(config.solver.use_fmm),
            fmm_expansion_order=int(config.solver.fmm_expansion_order),
            coupling_parameter=config.solver.coupling_parameter,
        )

        # Execution (currently informational; `solve()` still takes explicit args)
        execution_config = config.execution

        return self._with_state(
            c=float(c),
            rho=rho,
            frequencies=freqs,
            polar_start=polar_start,
            polar_end=polar_end,
            polar_num=polar_num,
            norm_angle=norm_angle,
            measurement_distance=measurement_distance,
            spl_angles=spl_angles,
            solver_options=solver_options,
            execution_config=execution_config,
        )

    @property
    def mesh(self) -> Optional[LoudspeakerMesh]:
        """Return the currently configured mesh (if any)."""
        return self.state.mesh

    @property
    def velocity_profile_obj(self) -> Optional[VelocityProfile]:
        """Return the currently configured velocity profile (if any)."""
        return self.state.velocity

    def compression_driver_generic25(
        self,
        path: str,
        voltage_rms: float = 2.83,
        waveguide_segments: int = 40,
    ) -> "Loudspeaker":
        """
        Attach a full lumped-element compression-driver network from a generic25 file.

        The network will be coupled to BEM by treating the BEM-computed throat
        acoustic input impedance as the external termination.
        """
        system = load_generic25(path)
        cfg = generic25_to_compression_driver_config(system)
        medium = AcousticMedium(c=self.state.c, rho=self.state.rho)
        options = CompressionDriverNetworkOptions(waveguide_segments=waveguide_segments)
        return self._with_state(
            driver_network=CompressionDriverNetwork(cfg, medium=medium, options=options),
            driver_excitation=CompressionDriverExcitation(voltage_rms=voltage_rms),
        )

    def compression_driver(
        self,
        config: CompressionDriverConfig,
        voltage_rms: float = 2.83,
        waveguide_segments: int = 40,
    ) -> "Loudspeaker":
        """Attach a compression-driver network from `CompressionDriverConfig`."""
        medium = AcousticMedium(c=self.state.c, rho=self.state.rho)
        options = CompressionDriverNetworkOptions(waveguide_segments=waveguide_segments)
        return self._with_state(
            driver_network=CompressionDriverNetwork(config, medium=medium, options=options),
            driver_excitation=CompressionDriverExcitation(voltage_rms=voltage_rms),
        )

    # =========================================================================
    # Geometry methods
    # =========================================================================

    def circular_piston(
        self,
        radius: Optional[float] = None,
        *,
        diameter: Optional[float] = None,
        element_size: Optional[float] = None,
    ) -> "Loudspeaker":
        """
        Define radiator as a flat circular piston.

        Parameters
        ----------
        radius : float, optional
            Piston radius in meters.
        diameter : float, optional
            Convenience alternative to `radius` (meters).
        element_size : float, optional
            Target max edge length (meters).
        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _geometry.circular_piston(self, radius, diameter=diameter, element_size=element_size)

    def annular_piston(
        self,
        inner_radius: float,
        outer_radius: float,
        element_size: Optional[float] = None,
    ) -> "Loudspeaker":
        """
        Define radiator as an annular (ring-shaped) piston.

        Parameters
        ----------
        inner_radius : float
            Inner radius in meters.
        outer_radius : float
            Outer radius in meters.
        element_size : float, optional
            Target max edge length (meters).
        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _geometry.annular_piston(self, inner_radius, outer_radius, element_size=element_size)

    def cone(
        self,
        inner_r: float,
        outer_r: float,
        height: float,
        curvature: float = 0.0,
        element_size: Optional[float] = None,
    ) -> "Loudspeaker":
        """
        Define radiator as an axisymmetric cone.

        Parameters
        ----------
        inner_r : float
            Inner radius at voice coil in meters.
        outer_r : float
            Outer radius at surround in meters.
        height : float
            Cone height in meters.
        curvature : float
            Curvature parameter (-1 to 1). 0 is straight.
        element_size : float, optional
            Target max edge length (meters).
        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _geometry.cone(self, inner_r, outer_r, height, curvature, element_size=element_size)

    def dome(
        self,
        radius: float,
        depth: float,
        profile: str = "spherical",
        element_size: Optional[float] = None,
    ) -> "Loudspeaker":
        """
        Define radiator as a dome.

        Parameters
        ----------
        radius : float
            Dome base radius in meters.
        depth : float
            Dome depth in meters.
        profile : str
            Profile type: 'spherical', 'elliptical', 'parabolic'.
        element_size : float, optional
            Target max edge length (meters).
        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _geometry.dome(self, radius, depth, profile, element_size=element_size)

    def waveguide(
        self,
        throat_diameter: float,
        mouth_diameter: float,
        length: float,
        profile: str = 'exponential',
        mesh_preset: Optional[str] = None,
        throat_element_size: Optional[float] = None,
        mouth_element_size: Optional[float] = None,
        throat_velocity_amplitude: float = 0.01,
        # Default corresponds to 96 axial points (n_axial_slices + 1).
        n_axial_slices: int = 95,
        n_circumferential: int = 36,
        corner_resolution: int = 0,
        export_mesh: bool = False,
        export_mesh_path: str | Path | None = None,
        show_mesh_quality: bool = True,
        throat_edge_refine: bool = False,
        throat_edge_size: Optional[float] = None,
        throat_edge_dist_min: Optional[float] = None,
        throat_edge_dist_max: Optional[float] = None,
        throat_edge_sampling: int = 80,
        lock_throat_boundary: bool = False,
        throat_circle_points: Optional[int] = None,
        mesh_optimize: Optional[str] = None,
        morph: Optional["MorphConfig"] = None,
        morph_rate: Optional[float] = None,
        morph_fixed_part: Optional[float] = None,
        morph_end_part: Optional[float] = None,
        os_opening_angle_deg: Optional[float] = None,
        cts_driver_exit_angle_deg: Optional[float] = None,
        cts_throat_blend: Optional[float] = None,
        cts_transition: Optional[float] = None,
        cts_throat_angle_deg: Optional[float] = None,
        cts_tangency: Optional[float] = None,
        cts_mouth_roll: Optional[float] = None,
        cts_curvature_regularizer: Optional[float] = None,
        cts_mid_curvature: Optional[float] = None,
    ) -> "Loudspeaker":
        """
        Create waveguide with automatic setup.
        
        This method handles all waveguide configuration automatically:
        - Mesh generation with adaptive sizing
        - Domain detection (throat vs walls)
        - Velocity profile (throat vibrates, walls rigid)
        - Mesh quality reporting (optional)
        - 3D HTML export (optional)
        
        Parameters
        ----------
        throat_diameter : float
            Driver throat diameter in meters (e.g., 0.025 for 1" compression driver).
        mouth_diameter : float
            Mouth opening diameter in meters.
        length : float
            Waveguide length in meters.
        profile : str
            Profile type: 'exponential' (default), 'conical', 'tractrix' (legacy), 'tractrix_horn', 'os', 'hyperbolic', 'cts'.
        os_opening_angle_deg : float, optional
            For `profile='os'`: OS opening half-angle from z-axis (must be feasible for the given endpoints).
        cts_* : optional
            For `profile='cts'`: pass CTS throat blend/transition/tangency and optional driver angle.
        throat_element_size : float, optional
            Element size at throat. Default: 0.002 m (2 mm).
        mouth_element_size : float, optional
            Element size at mouth. Default: 0.008 m (8 mm).
        throat_velocity_amplitude : float
            Throat velocity amplitude in m/s (default 0.01).
        n_axial_slices : int
            Number of profile slices along axis (default 95 -> 96 axial points).
        n_circumferential : int
            Circumferential divisions (default 36, 10° resolution).
        corner_resolution : int
            For rounded-rectangle mouths: points per corner arc (default 0 = preset/default).
        export_mesh : bool
            Auto-export 3D HTML visualization (default False).
        show_mesh_quality : bool
            Print mesh quality report (default True).
        
        Returns
        -------
        Loudspeaker
            Self for chaining, configured with mesh and velocity profile.
        
        Examples
        --------
        >>> # Simple exponential waveguide
        >>> spk = (Loudspeaker()
        ...     .waveguide(throat_diameter=0.025, mouth_diameter=0.15, length=0.1)
        ...     .infinite_baffle()
        ...     .preset_horn()
        ...     .frequency_range(500, 20000, num=20)
        ...     .solve())
        
        >>> # Custom tractrix profile with fine mesh
        >>> spk = (Loudspeaker()
        ...     .waveguide(
        ...         throat_diameter=0.025,
        ...         mouth_diameter=0.15,
        ...         length=0.1,
        ...         profile='tractrix',
        ...         throat_element_size=0.002,  # 2mm (very fine)
        ...         n_axial_slices=60,           # High resolution
        ...     )
        ...     .infinite_baffle()
        ...     .solve())
        
        Notes
        -----
        Coordinate convention: the waveguide mouth lies on the plane `z=0` and
        the throat cap lies on the plane `z=-length` (see `bempp_audio.mesh.conventions`).
        This method automatically:
        1. Creates optimized mesh with adaptive sizing
        2. Detects domains (throat=vibrating, walls=rigid)
        3. Sets up velocity profile (piston at throat, zero at walls)
        4. Exports 3D visualization (if enabled)
        5. Prints mesh quality report (if enabled)
        """
        return _geometry.waveguide(
            self,
            throat_diameter=throat_diameter,
            mouth_diameter=mouth_diameter,
            length=length,
            profile=profile,
            mesh_preset=mesh_preset,
            throat_element_size=throat_element_size,
            mouth_element_size=mouth_element_size,
            throat_velocity_amplitude=throat_velocity_amplitude,
            n_axial_slices=n_axial_slices,
            n_circumferential=n_circumferential,
            corner_resolution=corner_resolution,
            export_mesh=export_mesh,
            export_mesh_path=export_mesh_path,
            show_mesh_quality=show_mesh_quality,
            throat_edge_refine=throat_edge_refine,
            throat_edge_size=throat_edge_size,
            throat_edge_dist_min=throat_edge_dist_min,
            throat_edge_dist_max=throat_edge_dist_max,
            throat_edge_sampling=throat_edge_sampling,
            lock_throat_boundary=lock_throat_boundary,
            throat_circle_points=throat_circle_points,
            mesh_optimize=mesh_optimize,
            morph=morph,
            morph_rate=morph_rate,
            morph_fixed_part=morph_fixed_part,
            morph_end_part=morph_end_part,
            os_opening_angle_deg=os_opening_angle_deg,
            cts_driver_exit_angle_deg=cts_driver_exit_angle_deg,
            cts_throat_blend=cts_throat_blend,
            cts_transition=cts_transition,
            cts_throat_angle_deg=cts_throat_angle_deg,
            cts_tangency=cts_tangency,
            cts_mouth_roll=cts_mouth_roll,
            cts_curvature_regularizer=cts_curvature_regularizer,
            cts_mid_curvature=cts_mid_curvature,
        )

    def waveguide_cts_rect(
        self,
        *,
        throat_diameter: float,
        mouth_width: float,
        mouth_height: float,
        length: float,
        corner_radius: float = 0.006,
        mesh_preset: Optional[str] = None,
        throat_velocity_amplitude: float = 0.01,
        # Default corresponds to 96 axial points (n_axial_slices + 1).
        n_axial_slices: int = 95,
        n_circumferential: int = 36,
        corner_resolution: int = 0,
        morph_fixed_part: Optional[float] = None,
        morph_rate: Optional[float] = None,
        morph_end_part: Optional[float] = None,
        lock_throat_boundary: bool = False,
        throat_circle_points: Optional[int] = None,
        mesh_optimize: Optional[str] = None,
        cts_driver_exit_angle_deg: Optional[float] = 10.0,
        cts_throat_blend: float = 0.15,
        cts_transition: float = 0.80,
        cts_tangency: float = 1.0,
        cts_mouth_roll: float = 0.6,
        cts_mid_curvature: float = 0.2,
        cts_curvature_regularizer: float = 1.0,
        export_mesh: bool = False,
        export_mesh_path: str | Path | None = None,
        show_mesh_quality: bool = True,
    ) -> "Loudspeaker":
        """
        Convenience wrapper for a CTS waveguide with a rounded-rectangle mouth.

        This is the most common "compression driver + rectangular CD waveguide" use-case.
        Defaults are chosen to support a directivity-driven iteration workflow.
        """
        from bempp_audio.mesh import MorphConfig

        morph = MorphConfig.rectangle(
            width=float(mouth_width),
            height=float(mouth_height),
            corner_radius=float(corner_radius),
            profile_mode="axes",
        )
        return self.waveguide(
            throat_diameter=float(throat_diameter),
            mouth_diameter=float(max(mouth_width, mouth_height)),  # axisymmetric envelope for sizing; morph sets final mouth
            length=float(length),
            profile="cts",
            mesh_preset=mesh_preset,
            throat_velocity_amplitude=float(throat_velocity_amplitude),
            n_axial_slices=int(n_axial_slices),
            n_circumferential=int(n_circumferential),
            corner_resolution=int(corner_resolution),
            lock_throat_boundary=bool(lock_throat_boundary),
            throat_circle_points=throat_circle_points,
            mesh_optimize=mesh_optimize,
            export_mesh=export_mesh,
            export_mesh_path=export_mesh_path,
            show_mesh_quality=show_mesh_quality,
            morph=morph,
            morph_fixed_part=morph_fixed_part,
            morph_rate=morph_rate,
            morph_end_part=morph_end_part,
            cts_driver_exit_angle_deg=cts_driver_exit_angle_deg,
            cts_throat_blend=float(cts_throat_blend),
            cts_transition=float(cts_transition),
            cts_tangency=float(cts_tangency),
            cts_mouth_roll=float(cts_mouth_roll),
            cts_mid_curvature=float(cts_mid_curvature),
            cts_curvature_regularizer=float(cts_curvature_regularizer),
        )

    def waveguide_from_config(
        self,
        config: WaveguideMeshConfig,
        throat_velocity_amplitude: float = 0.01,
        export_mesh: bool = False,
        export_mesh_path: str | Path | None = None,
        show_mesh_quality: bool = True,
    ) -> "Loudspeaker":
        return _geometry.waveguide_from_config(
            self,
            config=config,
            throat_velocity_amplitude=throat_velocity_amplitude,
            export_mesh=export_mesh,
            export_mesh_path=export_mesh_path,
            show_mesh_quality=show_mesh_quality,
        )

    def from_stl(
        self,
        filename: str,
        scale: float = 1.0,
    ) -> "Loudspeaker":
        """
        Load radiator geometry from STL file.

        Parameters
        ----------
        filename : str
            Path to STL file.
        scale : float
            Scale factor (e.g., 0.001 for mm to m).

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _geometry.from_stl(self, filename, scale)

    def from_mesh(self, mesh: LoudspeakerMesh) -> "Loudspeaker":
        """
        Use an existing LoudspeakerMesh.

        Parameters
        ----------
        mesh : LoudspeakerMesh
            Pre-configured mesh.

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _geometry.from_mesh(self, mesh)

    # =========================================================================
    # Baffle configuration
    # =========================================================================

    def infinite_baffle(self) -> "Loudspeaker":
        """
        Model as if mounted in an infinite rigid baffle.

        Uses a pragmatic image-source-inspired post-processing:
        - scales the solved surface pressure by ×2, and
        - suppresses rear-hemisphere evaluation (z < 0) for field/directivity.

        This is exact for planar sources lying in the baffle plane (z=0), but
        is not a full half-space BEM formulation for arbitrary geometries.

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.infinite_baffle(self)

    def circular_baffle(
        self,
        radius: float,
        element_size: Optional[float] = None,
    ) -> "Loudspeaker":
        """
        Add a finite circular baffle around the radiator.

        Parameters
        ----------
        radius : float
            Outer radius of the baffle in meters.
        element_size : float, optional
            Target max edge length (meters).
        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.circular_baffle(self, radius, element_size=element_size)

    def free_space(self) -> "Loudspeaker":
        """
        Model as radiating into free space (unbaffled).

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.free_space(self)

    def waveguide_on_box(
        self,
        throat_diameter: float,
        mouth_diameter: float,
        waveguide_length: float,
        box_width: float,
        box_height: float,
        box_depth: float,
        profile: str = "exponential",
        mount_position: Optional[tuple[float, float]] = None,
        mesh_preset: Optional[str] = None,
        throat_element_size: Optional[float] = None,
        mouth_element_size: Optional[float] = None,
        box_element_size: Optional[float] = None,
        baffle_element_size: Optional[float] = None,
        side_element_size: Optional[float] = None,
        back_element_size: Optional[float] = None,
        cabinet_chamfer_bottom: Optional["ChamferSpec"] = None,
        cabinet_chamfer_top: Optional["ChamferSpec"] = None,
        cabinet_chamfer_left: Optional["ChamferSpec"] = None,
        cabinet_chamfer_right: Optional["ChamferSpec"] = None,
        cabinet_fillet_bottom_radius: float = 0.0,
        cabinet_fillet_top_radius: float = 0.0,
        cabinet_fillet_left_radius: float = 0.0,
        cabinet_fillet_right_radius: float = 0.0,
        # Default corresponds to 96 axial points (n_axial_slices + 1).
        n_axial_slices: int = 95,
        n_circumferential: int = 36,
        corner_resolution: int = 0,
        throat_velocity_amplitude: float = 0.01,
        export_mesh: bool = False,
        export_mesh_path: str | Path | None = None,
        show_mesh_quality: bool = True,
        lock_throat_boundary: bool = False,
        throat_circle_points: Optional[int] = None,
        mesh_optimize: Optional[str] = "Netgen",
        morph: Optional["MorphConfig"] = None,
        morph_rate: Optional[float] = None,
        morph_fixed_part: Optional[float] = None,
        morph_end_part: Optional[float] = None,
        os_opening_angle_deg: Optional[float] = None,
        cts_driver_exit_angle_deg: Optional[float] = None,
        cts_throat_blend: Optional[float] = None,
        cts_transition: Optional[float] = None,
        cts_throat_angle_deg: Optional[float] = None,
        cts_tangency: Optional[float] = None,
        cts_mouth_roll: Optional[float] = None,
        cts_curvature_regularizer: Optional[float] = None,
        cts_mid_curvature: Optional[float] = None,
    ) -> "Loudspeaker":
        """
        Create a waveguide mounted on a box enclosure (free-space radiation).

        Generates a *unified* watertight BEM mesh with a shared edge at the
        waveguide-mouth / box-hole junction (no duplicate surfaces), and sets:
        - Domain 1: throat cap (vibrating piston velocity)
        - Domain 2: waveguide walls (rigid, v=0)
        - Domain 3: box faces (rigid, v=0)

        Parameters
        ----------
        throat_diameter, mouth_diameter, waveguide_length : float
            Waveguide geometry in meters.
        box_width, box_height, box_depth : float
            Box dimensions in meters.
        profile : str
            'exponential', 'conical', or 'tractrix'.
        mount_position : tuple[float, float], optional
            (x, y) position of mouth center on the front face (z=0). Default: centered.
            Coordinates are in the box-frame: `x ∈ [0, box_width]`, `y ∈ [0, box_height]`
            with origin at the lower-left corner when viewed from the front.
        throat_element_size, mouth_element_size, box_element_size : float
            Target element sizes in meters.
        baffle_element_size, side_element_size, back_element_size : float, optional
            Enclosure sizing controls for the unified mesh:
            - `baffle_element_size`: front face (z=0) target size
            - `side_element_size`: side/top/bottom target size at z=0 (grades toward back)
            - `back_element_size`: back face (z=-depth) target size
        cabinet_chamfer_* / cabinet_fillet_* : optional
            Optional baffle-edge blends (front perimeter edges), where the baffle meets
            the enclosure walls. You can set all four independently; each edge can be
            either chamfered (via `ChamferSpec`) or filleted (via radius), but not both.
        n_axial_slices, n_circumferential : int
            Waveguide discretization controls.
        corner_resolution : int
            Extra mesh samples per corner arc for rounded rectangle morphs.
            0 disables corner refinement; preset values (e.g., 8 for "standard")
            are used when not explicitly set.
        throat_velocity_amplitude : float
            Throat velocity amplitude in m/s.
        export_mesh : bool
            Export interactive HTML mesh visualization.
        show_mesh_quality : bool
            Print mesh quality report.
        """
        return _geometry.waveguide_on_box(
            self,
            throat_diameter=throat_diameter,
            mouth_diameter=mouth_diameter,
            waveguide_length=waveguide_length,
            box_width=box_width,
            box_height=box_height,
            box_depth=box_depth,
            profile=profile,
            mount_position=mount_position,
            mesh_preset=mesh_preset,
            throat_element_size=throat_element_size,
            mouth_element_size=mouth_element_size,
            box_element_size=box_element_size,
            baffle_element_size=baffle_element_size,
            side_element_size=side_element_size,
            back_element_size=back_element_size,
            cabinet_chamfer_bottom=cabinet_chamfer_bottom,
            cabinet_chamfer_top=cabinet_chamfer_top,
            cabinet_chamfer_left=cabinet_chamfer_left,
            cabinet_chamfer_right=cabinet_chamfer_right,
            cabinet_fillet_bottom_radius=cabinet_fillet_bottom_radius,
            cabinet_fillet_top_radius=cabinet_fillet_top_radius,
            cabinet_fillet_left_radius=cabinet_fillet_left_radius,
            cabinet_fillet_right_radius=cabinet_fillet_right_radius,
            n_axial_slices=n_axial_slices,
            n_circumferential=n_circumferential,
            corner_resolution=corner_resolution,
            throat_velocity_amplitude=throat_velocity_amplitude,
            export_mesh=export_mesh,
            export_mesh_path=export_mesh_path,
            show_mesh_quality=show_mesh_quality,
            lock_throat_boundary=lock_throat_boundary,
            throat_circle_points=throat_circle_points,
            mesh_optimize=mesh_optimize,
            morph=morph,
            morph_rate=morph_rate,
            morph_fixed_part=morph_fixed_part,
            morph_end_part=morph_end_part,
            os_opening_angle_deg=os_opening_angle_deg,
            cts_driver_exit_angle_deg=cts_driver_exit_angle_deg,
            cts_throat_blend=cts_throat_blend,
            cts_transition=cts_transition,
            cts_throat_angle_deg=cts_throat_angle_deg,
            cts_tangency=cts_tangency,
            cts_mouth_roll=cts_mouth_roll,
            cts_curvature_regularizer=cts_curvature_regularizer,
            cts_mid_curvature=cts_mid_curvature,
        )

    def waveguide_on_box_from_config(
        self,
        config: UnifiedMeshConfig,
        throat_velocity_amplitude: float = 0.01,
        export_mesh: bool = False,
        export_mesh_path: str | Path | None = None,
        show_mesh_quality: bool = True,
    ) -> "Loudspeaker":
        return _geometry.waveguide_on_box_from_config(
            self,
            config=config,
            throat_velocity_amplitude=throat_velocity_amplitude,
            export_mesh=export_mesh,
            export_mesh_path=export_mesh_path,
            show_mesh_quality=show_mesh_quality,
        )

    def describe_mesh(self) -> str:
        """
        Print and return a concise mesh summary (elements/vertices, domain stats, reference).

        Returns
        -------
        str
            Multi-line description suitable for logs/bug reports.
        """
        if self.state.mesh is None:
            raise ValueError("Mesh not configured.")

        logger = get_logger()
        mesh = self.state.mesh
        grid = mesh.grid
        info = mesh.info()

        def fmt_vec3(v) -> str:
            return "[" + ",".join(f"{float(x):g}" for x in np.asarray(v, dtype=float).reshape(3)) + "]"

        lines: list[str] = []
        lines.append(f"Mesh: {info.n_elements} elements, {info.n_vertices} vertices")

        domain_indices = np.asarray(grid.domain_indices)
        elements = np.asarray(grid.elements)
        vertices = np.asarray(grid.vertices)
        domain_ids = [int(d) for d in np.unique(domain_indices)]

        domain_labels: dict[int, str] = {}
        if self.state.waveguide is not None:
            domain_labels[int(self.state.waveguide.throat_domain)] = "throat"
            domain_labels[int(self.state.waveguide.wall_domain)] = "walls"
            if int(Domain.ENCLOSURE) in domain_ids:
                domain_labels.setdefault(int(Domain.ENCLOSURE), "enclosure")

        for domain_id in domain_ids:
            mask = domain_indices == domain_id
            n_elements = int(np.sum(mask))
            if n_elements <= 0:
                continue
            tri = elements[:, mask]
            vert_ids = np.unique(tri)
            z = vertices[2, vert_ids]
            z_min = float(np.min(z))
            z_max = float(np.max(z))
            label = domain_labels.get(domain_id)
            if label:
                lines.append(
                    f"Domain {domain_id} ({label}): {n_elements} elements, z ∈ [{z_min:.2f}, {z_max:.2f}]"
                )
            else:
                lines.append(f"Domain {domain_id}: {n_elements} elements, z ∈ [{z_min:.2f}, {z_max:.2f}]")

        if self.state.reference is not None:
            ref = self.state.reference
            lines.append(f"Acoustic reference: origin={fmt_vec3(ref.origin)}, axis={fmt_vec3(ref.axis)}")

        for line in lines:
            logger.info(line)
        return "\n".join(lines)

    def validate_topology(self) -> bool:
        """
        Validate mesh topology (closedness/manifoldness).

        Checks that the mesh is suitable for BEM analysis:
        - All edges shared by exactly 2 triangles (no holes, no non-manifold edges)

        Returns
        -------
        bool
            True if mesh passes topology validation.

        Raises
        ------
        ValueError
            If mesh is not configured.

        Notes
        -----
        BEM requires a closed surface for exterior radiation problems.
        Boundary edges (holes) or non-manifold edges will cause incorrect results.
        """
        if self.state.mesh is None:
            raise ValueError("Mesh not configured.")

        from bempp_audio.mesh.validation import MeshTopologyValidator

        is_valid, info = MeshTopologyValidator.validate_mesh(self.state.mesh, verbose=True)
        return is_valid

    def _get_mouth_diameter(self) -> float:
        """
        Get the mouth diameter of the current driver/waveguide.

        Derives the mouth size from the waveguide mesh geometry by finding
        vertices at z=0 (the mouth plane) and computing maximum radius.

        Returns
        -------
        float
            Mouth diameter in meters.

        Notes
        -----
        Currently assumes circular mouth geometry. Future versions will
        support rectangular mouths with corner radii, where this method
        will return the bounding diameter or equivalent aperture.
        """
        # For waveguides, get the maximum radius at z=0 (mouth)
        if self.state.mesh is None:
            raise ValueError("Mesh not configured.")
        vertices = self.state.mesh.grid.vertices

        # Find vertices at the mouth (z ≈ 0)
        z_coords = vertices[2, :]
        mouth_mask = np.abs(z_coords) < 1e-6

        if not np.any(mouth_mask):
            # Fallback: use max radius overall
            mouth_mask = np.ones(len(z_coords), dtype=bool)

        mouth_vertices = vertices[:, mouth_mask]

        cx, cy, _ = self._get_mouth_center()
        r = np.sqrt((mouth_vertices[0, :] - cx) ** 2 + (mouth_vertices[1, :] - cy) ** 2)

        mouth_diameter = 2.0 * np.max(r)

        return mouth_diameter

    def _get_mouth_center(self) -> Tuple[float, float, float]:
        """
        Estimate mouth center for circular mouth helpers.

        Uses stored metadata when available (e.g., waveguide-on-box mounting),
        otherwise estimates from vertices at z≈0.
        """
        if self.state.waveguide is not None:
            x, y, z = self.state.waveguide.mouth_center
            return float(x), float(y), float(z)

        if self.state.mesh is None:
            raise ValueError("Mesh not configured.")
        vertices = self.state.mesh.grid.vertices
        z_coords = vertices[2, :]
        mouth_mask = np.abs(z_coords) < 1e-6
        if np.any(mouth_mask):
            mouth_vertices = vertices[:, mouth_mask]
            return float(np.mean(mouth_vertices[0, :])), float(np.mean(mouth_vertices[1, :])), 0.0

        # Fallback: mesh center projected to z=0
        c = getattr(self.state.mesh, "center", np.array([0.0, 0.0, 0.0]))
        return float(c[0]), float(c[1]), 0.0

    def _get_mouth_vertices(self) -> np.ndarray:
        """
        Get the mouth edge vertices of the current driver/waveguide.

        Extracts vertices at z=0 that lie on the outer edge (maximum radius).
        These vertices define the exact polyline shape of the mouth opening.

        Returns
        -------
        np.ndarray
            Mouth edge vertices, shape (3, N), sorted by angle.
        """
        if self.state.mesh is None:
            raise ValueError("Mesh not configured.")
        vertices = self.state.mesh.grid.vertices

        # Find vertices at the mouth (z ≈ 0)
        z_coords = vertices[2, :]
        mouth_mask = np.abs(z_coords) < 1e-6

        if not np.any(mouth_mask):
            return np.array([]).reshape(3, 0)

        mouth_verts = vertices[:, mouth_mask]

        cx, cy, _ = self._get_mouth_center()
        r = np.sqrt((mouth_verts[0, :] - cx) ** 2 + (mouth_verts[1, :] - cy) ** 2)
        max_r = np.max(r)

        # Select vertices near the edge (within 1% of max radius)
        edge_mask = r > max_r * 0.99
        edge_verts = mouth_verts[:, edge_mask]

        # Sort by angle for proper polygon ordering
        angles = np.arctan2(edge_verts[1, :] - cy, edge_verts[0, :] - cx)
        order = np.argsort(angles)
        sorted_verts = edge_verts[:, order]

        return sorted_verts

    # =========================================================================
    # Velocity configuration
    # =========================================================================

    def velocity(
        self,
        mode: VelocityModeLike = "piston",
        amplitude: complex = 1.0,
        phase: float = 0.0,
        **kwargs,
    ) -> "Loudspeaker":
        """
        Set the velocity distribution.

        Parameters
        ----------
        mode : str
            Velocity mode: 'piston', 'gaussian', 'radial_taper', 'zero'.
        amplitude : complex
            Velocity amplitude in m/s.
        phase : float
            Phase offset in radians.
        **kwargs
            Additional arguments for specific modes.

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Examples
        --------
        >>> spk.velocity(mode='piston', amplitude=0.01)
        >>> spk.velocity(mode='gaussian', center=[0,0,0], width=0.02)
        """
        return _bc.velocity(self, mode=mode, amplitude=amplitude, phase=phase, **kwargs)

    def velocity_profile(self, profile: VelocityProfile) -> "Loudspeaker":
        """
        Use a custom VelocityProfile.

        Parameters
        ----------
        profile : VelocityProfile
            Pre-configured velocity profile.

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.velocity_profile(self, profile)

    def domain_names(self, names: dict) -> "Loudspeaker":
        """
        Assign human-readable names to mesh domain IDs.

        This enables using `.velocity_by_name()` instead of numeric domain IDs.
        Waveguides automatically get "throat" and "walls" names from metadata.

        Parameters
        ----------
        names : dict
            Mapping of domain ID (int) to name (str).
            Example: {1: "throat", 2: "walls", 3: "enclosure"}

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Examples
        --------
        >>> speaker.domain_names({1: "driver", 2: "cabinet", 3: "port"})
        >>> speaker.velocity_by_name({"driver": VelocityProfile.piston(0.01)})
        """
        return _bc.domain_names(self, names)

    def velocity_by_name(
        self,
        profiles: dict,
        default: Optional[VelocityProfile] = None,
    ) -> "Loudspeaker":
        """
        Assign velocity profiles by domain name instead of numeric ID.

        Requires `.domain_names()` to be called first, or waveguide metadata
        to provide automatic names ("throat", "walls").

        Parameters
        ----------
        profiles : dict
            Mapping of domain name (str) to VelocityProfile.
        default : VelocityProfile, optional
            Default profile for domains not in the mapping.
            If None, uses the `default_bc_policy` setting ("rigid" by default).

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Examples
        --------
        >>> # Waveguides auto-provide "throat" and "walls" names
        >>> speaker.waveguide(...).velocity_by_name({
        ...     "throat": VelocityProfile.piston(0.01),
        ...     "walls": VelocityProfile.zero(),
        ... })
        """
        return _bc.velocity_by_name(self, profiles, default=default)

    def default_bc(self, policy: BCPolicyLike = "rigid") -> "Loudspeaker":
        """
        Set the default boundary condition policy for unassigned domains.

        Parameters
        ----------
        policy : str
            - "rigid": Unassigned domains get zero velocity (rigid wall). Default.
            - "error": Raise an error if any domain is not explicitly assigned.

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Examples
        --------
        >>> # Only assign vibrating surfaces; everything else is rigid
        >>> speaker.default_bc("rigid").velocity_by_name({"throat": VelocityProfile.piston()})

        >>> # Require explicit assignment of all domains (catches mistakes)
        >>> speaker.default_bc("error")
        """
        return _bc.default_bc(self, policy)  # type: ignore[arg-type]

    # =========================================================================
    # Frequency configuration
    # =========================================================================

    def frequencies(self, freqs: np.ndarray) -> "Loudspeaker":
        """
        Set specific frequencies for analysis.

        Parameters
        ----------
        freqs : np.ndarray
            Array of frequencies in Hz.

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.frequencies(self, freqs)

    def single_frequency(self, frequency: float) -> "Loudspeaker":
        """
        Set a single frequency for analysis.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.single_frequency(self, frequency)

    def frequency_range(
        self,
        f1: float = 200.0,
        f2: float = 20000.0,
        num: int = 20,
        spacing: FrequencySpacingLike = "log",
        points_per_octave: Optional[int] = None,
    ) -> "Loudspeaker":
        """
        Set frequency range with specified spacing.

        Parameters
        ----------
        f1 : float
            Start frequency in Hz (default 200 Hz).
        f2 : float
            End frequency in Hz (default 20 kHz).
        num : int
            Number of frequency points (default 20).
        spacing : str
            Frequency spacing: 'log' (logarithmic) or 'linear' (default 'log').
        points_per_octave : int, optional
            If provided, overrides `num`/`spacing` and generates points spaced
            uniformly in octaves between `f1` and `f2`.

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Examples
        --------
        >>> # Use defaults: 200 Hz to 20 kHz, 20 points, logarithmic
        >>> speaker.frequency_range()
        >>>
        >>> # Logarithmic spacing from 100 Hz to 10 kHz (20 points)
        >>> speaker.frequency_range(100, 10000, num=20, spacing='log')
        >>>
        >>> # Linear spacing from 1 kHz to 5 kHz (50 points)
        >>> speaker.frequency_range(1000, 5000, num=50, spacing='linear')

        >>> # Octave-spaced points: 6 points per octave
        >>> speaker.frequency_range(20, 20000, points_per_octave=6)
        """
        return _bc.frequency_range(
            self,
            f1=f1,
            f2=f2,
            num=num,
            spacing=spacing,
            points_per_octave=points_per_octave,
        )

    def octave_bands(
        self,
        f_start: float = 63.0,
        f_end: float = 16000.0,
        fraction: int = 3,
    ) -> "Loudspeaker":
        """
        Set frequencies to ISO 266 octave band centers.

        Parameters
        ----------
        f_start : float
            Lowest band center frequency (default 63 Hz).
        f_end : float
            Highest band center frequency (default 16 kHz).
        fraction : int
            Octave fraction: 1 = full octave, 3 = 1/3 octave (default 3).

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Examples
        --------
        >>> # 1/3 octave bands from 100 Hz to 10 kHz
        >>> speaker.octave_bands(100, 10000, fraction=3)
        >>>
        >>> # Full octave bands
        >>> speaker.octave_bands(125, 8000, fraction=1)

        Notes
        -----
        ISO 266 band centers: f = 1000 * 2^(n/fraction)
        For 1/3 octave, ratio between bands is 2^(1/3) ≈ 1.26
        """
        return _bc.octave_bands(self, f_start=f_start, f_end=f_end, fraction=fraction)

    # =========================================================================
    # Polar map / directivity configuration
    # =========================================================================

    def polar_angles(
        self,
        start: float = 0.0,
        end: float = 180.0,
        num: int = 37,
    ) -> "Loudspeaker":
        """
        Set angle range for polar map / directivity analysis.

        Parameters
        ----------
        start : float
            Start angle in degrees (default 0).
        end : float
            End angle in degrees (default 180 for full hemisphere).
        num : int
            Number of angle points (default 37 for 5° resolution).

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Examples
        --------
        >>> # Half sphere for infinite baffle (0-90°, 19 points)
        >>> speaker.polar_angles(0, 90, 19)
        >>>
        >>> # Full hemisphere with high resolution
        >>> speaker.polar_angles(0, 180, 73)  # 2.5° resolution
        """
        return _bc.polar_angles(self, start=start, end=end, num=num)

    def normalize_to(self, angle: float = 0.0) -> "Loudspeaker":
        """
        Set normalization angle for directivity plots.

        Parameters
        ----------
        angle : float
            Angle in degrees to normalize SPL curves to (default 0 = on-axis).

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Examples
        --------
        >>> # Normalize to 20° off-axis (common for horn measurements)
        >>> speaker.normalize_to(20)
        """
        return _bc.normalize_to(self, angle)

    def measurement_distance(self, distance: float = 1.0) -> "Loudspeaker":
        """
        Set measurement distance for SPL calculations.

        Parameters
        ----------
        distance : float
            Distance from source center in meters (default 1.0).

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.measurement_distance(self, distance)

    def acoustic_reference(
        self,
        *,
        origin: Optional[np.ndarray] = None,
        axis: Optional[np.ndarray] = None,
        default_distance_m: Optional[float] = None,
    ) -> "Loudspeaker":
        """
        Set the acoustic reference (origin/axis/default distance) used by evaluation helpers.

        This is useful when the mesh is offset (e.g. waveguide-on-box) and you
        want a stable definition of "on-axis" and measurement distance.
        """
        return _bc.acoustic_reference(
            self,
            origin=origin,
            axis=axis,
            default_distance_m=default_distance_m,
        )

    def reference_mode(self, mode: str = "mouth") -> "Loudspeaker":
        """
        Set the acoustic reference origin mode for waveguides.

        For waveguides/horns, the reference origin can be placed at:

        - **"mouth"** (default): The mouth center at z=0. Standard for directivity
          measurements where angles are measured from the mouth plane. This is the
          correct choice for comparing with measured speaker data.

        - **"throat"**: The throat center at z=-length. Sometimes used when
          treating the throat as the acoustic source location (e.g., compression
          driver analysis where you want to reference the driver position).

        Parameters
        ----------
        mode : str
            Either "mouth" or "throat".

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Notes
        -----
        The axis is always +z (forward direction) regardless of mode.

        For most loudspeaker directivity work, "mouth" is the correct choice
        since that's where sound radiates into the listening space. Use "throat"
        only for specialized driver-referenced analysis.

        Examples
        --------
        >>> # Standard directivity (mouth-referenced)
        >>> speaker.waveguide(...).reference_mode("mouth")

        >>> # Driver-referenced analysis
        >>> speaker.waveguide(...).reference_mode("throat")
        """
        return _bc.reference_mode(self, mode)

    def spl_angles(self, angles: List[float]) -> "Loudspeaker":
        """
        Set specific angles for SPL vs frequency curves.

        Parameters
        ----------
        angles : list of float
            Angles in degrees for SPL curves.

        Returns
        -------
        Loudspeaker
            Self for chaining.

        Examples
        --------
        >>> # Standard listening angles
        >>> speaker.spl_angles([0, 15, 30, 45, 60, 90])
        """
        return _bc.spl_angles(self, angles)

    @property
    def plot_settings(self) -> dict:
        """
        Get current plot/directivity settings as a dictionary.

        Returns settings suitable for passing to `ReportBuilder.polar_options()` and SPL panels.

        Returns
        -------
        dict
            Dictionary with keys: max_angle, normalize_angle, distance, angles
        """
        s = self.state

        # Default SPL angles if not set
        if s.spl_angles is None:
            # Generate sensible defaults based on polar range
            max_angle = float(s.polar_end)
            if max_angle <= 90:
                angles = [0, 15, 30, 45, 60, 90][: int(max_angle / 15) + 1]
            else:
                angles = [0, 15, 30, 45, 60, 90]
        else:
            angles = list(s.spl_angles)

        return {
            "max_angle": float(s.polar_end),
            "normalize_angle": float(s.norm_angle),
            "distance": float(s.measurement_distance),
            "angles": angles,
            "n_polar_angles": int(s.polar_num),
        }

    # =========================================================================
    # Configuration presets
    # =========================================================================

    def preset(self, name: str) -> "Loudspeaker":
        """
        Apply a full simulation configuration preset.
        
        Loads a complete configuration including frequency range, directivity
        settings, and solver options from the config preset system.
        
        Parameters
        ----------
        name : str
            Preset name: 'quick_test', 'standard', 'high_resolution',
            'horn', 'woofer', or 'nearfield'.
        
        Returns
        -------
        Loudspeaker
            Self for chaining.
        
        Examples
        --------
        >>> # Use standard preset (200-20kHz, full hemisphere)
        >>> spk = (Loudspeaker()
        ...     .circular_piston(radius=0.05)
        ...     .infinite_baffle()
        ...     .preset('standard')
        ...     .solve())
        
        >>> # Use horn-optimized preset
        >>> spk = (Loudspeaker()
        ...     .waveguide(throat_diameter=0.025, mouth_diameter=0.15, length=0.1)
        ...     .infinite_baffle()
        ...     .preset('horn')
        ...     .solve())
        
        >>> # Quick test preset for development
        >>> spk = (Loudspeaker()
        ...     .circular_piston(radius=0.05)
        ...     .infinite_baffle()
        ...     .preset('quick_test')
        ...     .solve())
        
        Notes
        -----
        Available presets:
        - 'quick_test': Fast, 500-2000 Hz, 3 points
        - 'standard': General purpose, 200-20000 Hz, 20 points
        - 'high_resolution': Detailed, 100-20000 Hz, 50 points
        - 'horn': Horn-optimized, 500-20000 Hz, 0-90°
        - 'woofer': Low-freq, 20-1000 Hz, 30 points
        - 'nearfield': Nearfield measurement, 0.5m distance
        
        See Also
        --------
        preset_horn : Preset for horn directivity (no freq range)
        preset_nearfield : Preset for nearfield directivity
        preset_far_field : Preset for far-field measurements
        preset_anechoic : Preset for anechoic chamber (CEA-2034A)
        """
        return _presets.preset(self, name)

    def solver_preset(self, name: str) -> "Loudspeaker":
        """
        Apply an iterative-solver performance preset (tol/maxiter/use_fmm).

        Presets are independent of geometry; use `mesh_preset=...` on waveguide
        builders to control meshing.
        """
        return _presets.solver_preset(self, name)

    def performance_preset(self, name: str, *, mode: str = "horn") -> "Loudspeaker":
        """
        Apply a unified performance preset (mesh + solver + common sweep defaults).

        This is aimed at waveguide workflows. Geometry builders will use the
        preset's mesh defaults unless explicit element sizes are provided.
        """
        return _presets.performance_preset(self, name, mode=mode)

    def preset_horn(
        self,
        distance: float = 1.0,
        max_angle: float = 90.0,
        resolution_deg: float = 5.0,
        normalize_angle: float = 10.0,
    ) -> "Loudspeaker":
        """
        Configure for horn/waveguide measurements (typical CD horn normalization).
        
        Pre-configured settings:
        - Measurement distance: `distance` (default 1 m)
        - Polar angles: 0..`max_angle` at `resolution_deg` spacing (default 5°)
        - SPL angles: [0, 10, 20, 30, 45, 60, 90]
        - Normalize to: `normalize_angle` (default 10°; common for compression driver horns)
        
        Returns
        -------
        Loudspeaker
            Self for chaining.
        
        Examples
        --------
        >>> # Waveguide measurement
        >>> spk = (Loudspeaker()
        ...     .waveguide(throat_diameter=0.025, mouth_diameter=0.15, length=0.1)
        ...     .infinite_baffle()
        ...     .preset_horn()  # Sets all measurement parameters
        ...     .frequency_range(500, 20000)
        ...     .solve())
        
        Notes
        -----
        Horn-loaded speakers often normalize to 10° off-axis because the
        on-axis response can show HF irregularities while the listening
        window (0-30°) is smoother.
        """
        return _presets.preset_horn(
            self,
            distance=float(distance),
            max_angle=float(max_angle),
            resolution_deg=float(resolution_deg),
            normalize_angle=float(normalize_angle),
        )

    def preset_infinite_baffle(
        self,
        distance: float = 1.0,
        resolution_deg: float = 5.0,
        normalize_angle: float = 0.0,
    ) -> "Loudspeaker":
        """
        Configure for infinite-baffle (half-space) measurements.

        Parameters
        ----------
        distance : float
            Measurement distance in meters (default 1.0).
        resolution_deg : float
            Angular resolution in degrees for 0–90° (default 5.0).
        normalize_angle : float
            Normalization angle in degrees (default 0.0).
        """
        return _presets.preset_infinite_baffle(
            self,
            distance=distance,
            resolution_deg=resolution_deg,
            normalize_angle=normalize_angle,
        )

    def preset_cabinet_free_space(
        self,
        distance: float = 2.0,
        resolution_deg: float = 5.0,
        normalize_angle: float = 0.0,
    ) -> "Loudspeaker":
        """
        Configure for cabinet (finite baffle) radiation into free space.

        Uses full 0–180° polar coverage in a plane; for full 3D coverage,
        use 3D far-field balloon evaluation from individual frequency results.

        Parameters
        ----------
        distance : float
            Measurement distance in meters (default 2.0).
        resolution_deg : float
            Angular resolution in degrees for 0–180° (default 5.0).
        normalize_angle : float
            Normalization angle in degrees (default 0.0).
        """
        return _presets.preset_cabinet_free_space(
            self,
            distance=distance,
            resolution_deg=resolution_deg,
            normalize_angle=normalize_angle,
        )

    def preset_nearfield(self) -> "Loudspeaker":
        """
        Configure for nearfield measurements (studio monitors, desktop speakers).
        
        Pre-configured settings:
        - Measurement distance: 0.5 m (typical desktop listening)
        - Polar angles: 0-90° (19 points, 5° resolution)
        - SPL angles: [0, 15, 30, 45, 60]
        - Normalize to: 0° (on-axis)
        
        Returns
        -------
        Loudspeaker
            Self for chaining.
        
        Examples
        --------
        >>> # Studio monitor measurement
        >>> spk = (Loudspeaker()
        ...     .circular_piston(radius=0.0825)  # 6.5" woofer
        ...     .infinite_baffle()
        ...     .preset_nearfield()
        ...     .frequency_range(50, 5000)
        ...     .solve())
        
        Notes
        -----
        Nearfield monitors are typically used at 0.5-1m distance in
        studio environments where off-axis response up to 60° is relevant
        for the listening sweet spot.
        """
        return _presets.preset_nearfield(self)

    def preset_far_field(self) -> "Loudspeaker":
        """
        Configure for far-field measurements (PA systems, concert audio).
        
        Pre-configured settings:
        - Measurement distance: 10 m
        - Polar angles: 0-180° (37 points, 5° resolution)
        - SPL angles: [0, 10, 20, 30, 45, 60, 90]
        - Normalize to: 0° (on-axis)
        
        Returns
        -------
        Loudspeaker
            Self for chaining.
        
        Examples
        --------
        >>> # Line array element
        >>> spk = (Loudspeaker()
        ...     .cone(inner_r=0.025, outer_r=0.15, height=0.03)
        ...     .free_space()  # Unbaffled for array element
        ...     .preset_far_field()
        ...     .frequency_range(100, 10000)
        ...     .solve())
        
        Notes
        -----
        Far-field measurements require full hemisphere (0-180°) coverage
        to capture rear radiation, especially important for array elements
        and outdoor sound reinforcement systems.
        """
        return _presets.preset_far_field(self)

    def preset_anechoic(self) -> "Loudspeaker":
        """
        Configure for anechoic chamber measurements (CEA-2034A standard).
        
        Pre-configured settings:
        - Measurement distance: 2 m (CEA-2034A standard)
        - Polar angles: 0-180° (73 points, 2.5° resolution)
        - SPL angles: [0, 10, 15, 20, 30, 40, 45, 60, 90]
        - Normalize to: 0° (on-axis)
        
        Returns
        -------
        Loudspeaker
            Self for chaining.
        
        Examples
        --------
        >>> # High-resolution anechoic measurement
        >>> spk = (Loudspeaker()
        ...     .dome(radius=0.013, depth=0.005, profile='spherical')
        ...     .infinite_baffle()
        ...     .preset_anechoic()
        ...     .frequency_range(2000, 20000, num=40)
        ...     .solve())
        
        Notes
        -----
        CEA-2034A standard specifies 2m measurement distance with
        high angular resolution for comprehensive directivity analysis.
        This preset generates data suitable for spinorama charts and
        industry-standard comparisons.
        """
        return _presets.preset_anechoic(self)

    def preset_cea2034(self) -> "Loudspeaker":
        """
        Alias for a high-resolution anechoic-style setup (2 m).

        Note: Full CEA-2034A spinorama requires multiple planes; this preset
        configures a good default sweep for the current 2D polar workflow.
        """
        return _presets.preset_cea2034(self)

    # =========================================================================
    # Solver configuration
    # =========================================================================

    def medium(
        self,
        c: float = 343.0,
        rho: float = 1.225,
    ) -> "Loudspeaker":
        """
        Set acoustic medium properties.

        Parameters
        ----------
        c : float
            Speed of sound in m/s (default 343.0).
        rho : float
            Air density in kg/m³ (default 1.225).

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.medium(self, c=c, rho=rho)

    def solver_options(
        self,
        tol: float = 1e-5,
        maxiter: int = 1000,
        use_fmm: bool = False,
        fmm_expansion_order: int | None = None,
    ) -> "Loudspeaker":
        """
        Configure solver options.

        Parameters
        ----------
        tol : float
            GMRES tolerance (default 1e-5).
        maxiter : int
            Maximum iterations (default 1000).
        use_fmm : bool
            Use Fast Multipole Method (default False).
        fmm_expansion_order : int, optional
            FMM expansion order (passed through to the backend). If not provided,
            leaves the current value unchanged.

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.solver_options(
            self,
            tol=tol,
            maxiter=maxiter,
            use_fmm=use_fmm,
            fmm_expansion_order=fmm_expansion_order,
        )

    def solver_progress(
        self,
        *,
        gmres_log_every: int = 10,
        gmres_log_residuals: bool = True,
    ) -> "Loudspeaker":
        """Log per-iteration GMRES progress during solves."""
        return _bc.solver_progress(
            self,
            gmres_log_every=gmres_log_every,
            gmres_log_residuals=gmres_log_residuals,
        )

    def use_osrc(self, npade: int = 2) -> "Loudspeaker":
        """
        Use OSRC preconditioning for high-frequency efficiency.

        Parameters
        ----------
        npade : int
            Padé approximation order (default 2).

        Returns
        -------
        Loudspeaker
            Self for chaining.
        """
        return _bc.use_osrc(self, npade=npade)

    # =========================================================================
    # Solve
    # =========================================================================

    def solve(
        self,
        n_workers: Optional[int] = None,
        progress_callback=None,
        show_progress: Optional[bool] = None,
        *,
        strict: bool = False,
    ) -> FrequencyResponse:
        """
        Execute the acoustic simulation.

        Parameters
        ----------
        n_workers : int, optional
            Number of parallel workers. Default None = auto (min(8, CPUs)).
            Set to 1 for sequential execution.
        progress_callback : callable, optional
            Function called with (freq, index, total) for progress.
        show_progress : bool
            If True, display progress bar. If None, uses `ExecutionConfig.show_progress`
            (if available), otherwise defaults to True.
        strict : bool
            If True, require mesh/velocity/frequencies to be explicitly configured.
            If False (default), fill missing velocity/frequencies with defaults.

        Returns
        -------
        FrequencyResponse
            Container with results at all frequencies.

        Raises
        ------
        ValueError
            If mesh, velocity, or frequencies are not configured.
        """
        return _solve.solve(
            self,
            n_workers=n_workers,
            progress_callback=progress_callback,
            show_progress=show_progress,
            strict=strict,
        )

    def solve_single(self, frequency: float) -> RadiationResult:
        """
        Solve at a single frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        RadiationResult
            Result at the specified frequency.
        """
        return _solve.solve_single(self, frequency)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def info(self) -> dict:
        """
        Get configuration summary.

        Returns
        -------
        dict
            Configuration details.
        """
        s = self.state
        mesh_info = s.mesh.info() if s.mesh else None
        return {
            "mesh": mesh_info,
            "baffle": s.baffle,
            "velocity": repr(s.velocity) if s.velocity else None,
            "frequencies": s.frequencies.tolist() if s.frequencies is not None else None,
            "n_frequencies": len(s.frequencies) if s.frequencies is not None else 0,
            "c": s.c,
            "rho": s.rho,
            "use_osrc": s.use_osrc,
        }

    def __repr__(self) -> str:
        s = self.state
        parts = ["Loudspeaker("]
        if s.mesh:
            parts.append(f"  mesh={s.mesh},")
        if s.baffle:
            parts.append(f"  baffle={s.baffle},")
        if s.frequencies is not None:
            parts.append(f"  freqs={len(s.frequencies)} points,")
        parts.append(")")
        return "\n".join(parts)
