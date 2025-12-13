"""
Configuration classes for bempp_audio simulations.

This module provides structured configuration management for acoustic simulations,
supporting both fluent API usage and batch/reproducible workflows.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import numpy as np
from pathlib import Path
import json


@dataclass
class FrequencyConfig:
    """Frequency sweep configuration."""
    
    # Explicit frequency array (if provided, overrides range)
    frequencies: Optional[np.ndarray] = None
    
    # Frequency range parameters
    f_start: float = 200.0    # Hz
    f_end: float = 20000.0    # Hz
    num_points: int = 20
    spacing: str = 'log'      # 'log', 'linear', or 'octave' (see notes below)
    
    def to_array(self) -> np.ndarray:
        """Generate frequency array based on configuration."""
        if self.frequencies is not None:
            return self.frequencies
        
        if self.spacing == 'log':
            return np.logspace(
                np.log10(self.f_start),
                np.log10(self.f_end),
                self.num_points
            )
        if self.spacing == 'linear':
            return np.linspace(self.f_start, self.f_end, self.num_points)
        if self.spacing == "octave":
            if self.f_start <= 0 or self.f_end <= 0 or self.f_end <= self.f_start:
                raise ValueError("For spacing='octave', require 0 < f_start < f_end.")
            ppo = int(self.num_points)
            if ppo < 1:
                raise ValueError("For spacing='octave', num_points must be >= 1 (points per octave).")
            octaves = float(np.log2(float(self.f_end) / float(self.f_start)))
            n = max(2, int(round(octaves * float(ppo))) + 1)
            t = np.linspace(0.0, octaves, n)
            return float(self.f_start) * (2.0 ** t)
        raise ValueError(f"Unknown spacing '{self.spacing}'.")
    
    def validate(self) -> List[str]:
        """Validate configuration. Returns list of error messages."""
        errors = []
        if self.f_start <= 0:
            errors.append("f_start must be positive")
        if self.f_end <= self.f_start:
            errors.append("f_end must be greater than f_start")
        if self.num_points < 1:
            errors.append("num_points must be at least 1")
        if self.spacing not in ('log', 'linear', 'octave'):
            errors.append(f"spacing must be 'log', 'linear', or 'octave', got '{self.spacing}'")
        return errors


@dataclass
class DirectivityConfig:
    """Directivity measurement configuration."""
    
    # Polar angle sweep
    polar_start: float = 0.0      # degrees
    polar_end: float = 180.0      # degrees
    polar_num: int = 37           # number of angles
    
    # Specific angles for SPL reporting
    spl_angles: Optional[List[float]] = None  # degrees
    
    # Normalization
    normalize_angle: float = 0.0  # degrees (reference for normalization)
    measurement_distance: float = 1.0  # meters
    
    def polar_angles(self) -> np.ndarray:
        """Generate polar angle array."""
        return np.linspace(self.polar_start, self.polar_end, self.polar_num)
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if self.polar_start < 0 or self.polar_start > 180:
            errors.append("polar_start must be between 0 and 180 degrees")
        if self.polar_end < 0 or self.polar_end > 180:
            errors.append("polar_end must be between 0 and 180 degrees")
        if self.polar_end <= self.polar_start:
            errors.append("polar_end must be greater than polar_start")
        if self.polar_num < 2:
            errors.append("polar_num must be at least 2")
        if self.measurement_distance <= 0:
            errors.append("measurement_distance must be positive")
        if self.normalize_angle < 0 or self.normalize_angle > 180:
            errors.append("normalize_angle must be between 0 and 180 degrees")
        return errors


@dataclass
class MediumConfig:
    """Acoustic medium properties."""
    
    c: float = 343.0         # speed of sound (m/s)
    rho: float = 1.225       # air density (kg/m³)
    temperature: Optional[float] = None  # Celsius (if provided, overrides c)
    
    def compute_speed_of_sound(self) -> float:
        """Compute speed of sound from temperature if provided."""
        if self.temperature is not None:
            # c = 331.3 + 0.606 * T
            return 331.3 + 0.606 * self.temperature
        return self.c
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if self.c <= 0:
            errors.append("speed of sound must be positive")
        if self.rho <= 0:
            errors.append("density must be positive")
        if self.temperature is not None and self.temperature < -273.15:
            errors.append("temperature must be above absolute zero")
        return errors


@dataclass
class SolverConfig:
    """BEM solver configuration."""
    
    tol: float = 1e-5           # GMRES tolerance
    maxiter: int = 1000         # Maximum iterations
    space_type: str = "DP"      # Function space: "DP" or "P"
    space_order: int = 0        # Function space order
    use_fmm: bool = False       # Fast multipole method
    fmm_expansion_order: int = 5
    coupling_parameter: Optional[float] = None  # Burton-Miller η
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if self.tol <= 0 or self.tol >= 1:
            errors.append("tolerance must be between 0 and 1")
        if self.maxiter < 1:
            errors.append("maxiter must be at least 1")
        if self.space_type not in ("DP", "P"):
            errors.append(f"space_type must be 'DP' or 'P', got '{self.space_type}'")
        if self.space_order < 0:
            errors.append("space_order must be non-negative")
        return errors


@dataclass
class ExecutionConfig:
    """Execution and parallelization settings."""
    
    n_workers: Optional[int] = None  # None = auto-detect
    show_progress: bool = True
    cache_operators: bool = True
    verbose: bool = False
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if self.n_workers is not None and self.n_workers < 1:
            errors.append("n_workers must be at least 1 or None")
        return errors


@dataclass
class SimulationConfig:
    """
    Complete configuration for acoustic simulation.
    
    This class provides a structured way to define all simulation parameters,
    supporting validation, serialization, and integration with the fluent API.
    
    Examples
    --------
    Create configuration programmatically:
    
    >>> config = SimulationConfig(
    ...     frequency=FrequencyConfig(f_start=500, f_end=16000, num_points=10),
    ...     directivity=DirectivityConfig(polar_start=0, polar_end=90),
    ...     medium=MediumConfig(c=343.0, rho=1.225)
    ... )
    
    Save/load configuration:
    
    >>> config.to_json('simulation.json')
    >>> config = SimulationConfig.from_json('simulation.json')
    
    Use with fluent API:
    
    >>> from bempp_audio import Loudspeaker
    >>> spk = Loudspeaker.from_config(config)
    >>> spk.circular_piston(0.05).solve()
    """
    
    # Configuration sections
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    directivity: DirectivityConfig = field(default_factory=DirectivityConfig)
    medium: MediumConfig = field(default_factory=MediumConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Metadata
    name: str = "acoustic_simulation"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """
        Validate entire configuration.
        
        Returns
        -------
        List[str]
            List of validation error messages. Empty if valid.
        """
        errors = []
        errors.extend([f"Frequency: {e}" for e in self.frequency.validate()])
        errors.extend([f"Directivity: {e}" for e in self.directivity.validate()])
        errors.extend([f"Medium: {e}" for e in self.medium.validate()])
        errors.extend([f"Solver: {e}" for e in self.solver.validate()])
        errors.extend([f"Execution: {e}" for e in self.execution.validate()])
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns
        -------
        dict
            Configuration as nested dictionary.
        """
        result = {
            'name': self.name,
            'description': self.description,
            'tags': self.tags,
            'frequency': asdict(self.frequency),
            'directivity': asdict(self.directivity),
            'medium': asdict(self.medium),
            'solver': asdict(self.solver),
            'execution': asdict(self.execution),
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if self.frequency.frequencies is not None:
            result['frequency']['frequencies'] = self.frequency.frequencies.tolist()
        
        return result
    
    def to_json(self, filepath: str | Path) -> None:
        """
        Save configuration to JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path.
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationConfig:
        """
        Load configuration from dictionary.
        
        Parameters
        ----------
        data : dict
            Configuration dictionary.
        
        Returns
        -------
        SimulationConfig
            Configuration object.
        """
        # Extract section configs
        freq_data = data.get('frequency', {})
        if 'frequencies' in freq_data and freq_data['frequencies'] is not None:
            freq_data['frequencies'] = np.array(freq_data['frequencies'])
        
        return cls(
            name=data.get('name', 'acoustic_simulation'),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            frequency=FrequencyConfig(**freq_data),
            directivity=DirectivityConfig(**data.get('directivity', {})),
            medium=MediumConfig(**data.get('medium', {})),
            solver=SolverConfig(**data.get('solver', {})),
            execution=ExecutionConfig(**data.get('execution', {})),
        )
    
    @classmethod
    def from_json(cls, filepath: str | Path) -> SimulationConfig:
        """
        Load configuration from JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Input file path.
        
        Returns
        -------
        SimulationConfig
            Configuration object.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def summary(self) -> str:
        """
        Generate human-readable summary of configuration.
        
        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            f"Configuration: {self.name}",
            f"Description: {self.description or '(none)'}",
            "",
            "Frequency Sweep:",
            f"  {self.frequency.f_start} - {self.frequency.f_end} Hz",
            f"  {self.frequency.num_points} points ({self.frequency.spacing} spacing)",
            "",
            "Directivity:",
            f"  Polar: {self.directivity.polar_start}° - {self.directivity.polar_end}° ({self.directivity.polar_num} angles)",
            f"  Measurement distance: {self.directivity.measurement_distance} m",
            f"  Normalize to: {self.directivity.normalize_angle}°",
            "",
            "Medium:",
            f"  Speed of sound: {self.medium.compute_speed_of_sound():.1f} m/s",
            f"  Density: {self.medium.rho} kg/m³",
            "",
            "Solver:",
            f"  Tolerance: {self.solver.tol}",
            f"  Max iterations: {self.solver.maxiter}",
            f"  Space: {self.solver.space_type}{self.solver.space_order}",
            f"  FMM: {'enabled' if self.solver.use_fmm else 'disabled'}",
            "",
            "Execution:",
            f"  Workers: {self.execution.n_workers or 'auto'}",
            f"  Progress: {'enabled' if self.execution.show_progress else 'disabled'}",
        ]
        return "\n".join(lines)


# Preset configurations for common scenarios
class ConfigPresets:
    """Pre-defined configurations for common use cases."""

    @staticmethod
    def available() -> List[str]:
        """Return the list of available preset names."""
        return ["high_resolution", "horn", "nearfield", "quick_test", "standard", "woofer"]

    @staticmethod
    def get_preset(name: str) -> SimulationConfig:
        """Get a preset by name (used by the fluent API)."""
        key = str(name).strip().lower()
        presets = {
            "quick_test": ConfigPresets.quick_test,
            "standard": ConfigPresets.standard,
            "high_resolution": ConfigPresets.high_resolution,
            "horn": ConfigPresets.horn,
            "woofer": ConfigPresets.woofer,
            "nearfield": ConfigPresets.nearfield,
        }
        if key not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {', '.join(sorted(presets.keys()))}")
        return presets[key]()
    
    @staticmethod
    def quick_test() -> SimulationConfig:
        """Quick test with minimal computation."""
        return SimulationConfig(
            name="quick_test",
            description="Fast test configuration with coarse resolution",
            frequency=FrequencyConfig(f_start=500, f_end=2000, num_points=3),
            directivity=DirectivityConfig(polar_start=0, polar_end=90, polar_num=10),
            solver=SolverConfig(tol=1e-3, maxiter=250, use_fmm=False),
        )
    
    @staticmethod
    def standard() -> SimulationConfig:
        """Standard configuration for typical simulations."""
        return SimulationConfig(
            name="standard",
            description="Standard resolution for general-purpose simulation",
            frequency=FrequencyConfig(f_start=200, f_end=20000, num_points=20),
            directivity=DirectivityConfig(polar_start=0, polar_end=180, polar_num=37),
            solver=SolverConfig(tol=1e-5, maxiter=1000, use_fmm=False),
        )
    
    @staticmethod
    def high_resolution() -> SimulationConfig:
        """High-resolution configuration for detailed analysis."""
        return SimulationConfig(
            name="high_resolution",
            description="High resolution for detailed frequency and directivity analysis",
            frequency=FrequencyConfig(f_start=100, f_end=20000, num_points=50),
            directivity=DirectivityConfig(polar_start=0, polar_end=180, polar_num=73),
            solver=SolverConfig(tol=3e-6, maxiter=1500, use_fmm=False),
        )
    
    @staticmethod
    def horn() -> SimulationConfig:
        """Configuration optimized for horn simulations."""
        return SimulationConfig(
            name="horn",
            description="Optimized for horn/waveguide simulations (higher frequencies)",
            frequency=FrequencyConfig(f_start=500, f_end=20000, num_points=30),
            directivity=DirectivityConfig(
                polar_start=0,
                polar_end=90,
                polar_num=37,
                spl_angles=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            ),
            solver=SolverConfig(tol=1e-5, maxiter=1200, use_fmm=False),
        )
    
    @staticmethod
    def woofer() -> SimulationConfig:
        """Configuration optimized for woofer simulations."""
        return SimulationConfig(
            name="woofer",
            description="Optimized for woofer simulations (lower frequencies)",
            frequency=FrequencyConfig(f_start=20, f_end=1000, num_points=30),
            directivity=DirectivityConfig(polar_start=0, polar_end=180, polar_num=37),
            solver=SolverConfig(tol=1e-5, maxiter=800, use_fmm=False),
        )
    
    @staticmethod
    def nearfield() -> SimulationConfig:
        """Configuration for near-field measurements."""
        return SimulationConfig(
            name="nearfield",
            description="Near-field measurement configuration",
            frequency=FrequencyConfig(f_start=200, f_end=20000, num_points=20),
            directivity=DirectivityConfig(
                polar_start=0,
                polar_end=45,
                polar_num=19,
                measurement_distance=0.5,
                normalize_angle=0
            ),
            solver=SolverConfig(tol=1e-5, maxiter=1000, use_fmm=False),
        )
