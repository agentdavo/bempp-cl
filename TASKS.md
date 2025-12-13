## Roadmap: Compression Driver + Waveguide (LEM + FEM + BEM)

This repository is converging on a design-grade workflow for advanced compression drivers with smooth SPL/directivity.

### Conventions

- **`[x]`** = code exists, unit tests pass, and example script runs in "quick mode" without crashing
- **`[ ] Validation`** = analytical/benchmark comparison pending (scaffolding works, physics not yet verified)
- **Quick mode**: set `BEMPPAUDIO_NUM_FREQS=1` (or equivalent) for fast smoke tests

### Quick-Mode Smoke Tests (per Part)

| Part | Example Script | Quick-Mode Command |
|------|---------------|-------------------|
| 1 | `compression_driver_network_minimal.py` | `python examples/bempp_audio/compression_driver_network_minimal.py` |
| 2 | `waveguide_designer_report.py` | `python examples/bempp_audio/waveguide_designer_report.py` |
| 3 | `waveguide_infinite_baffle.py` | `BEMPPAUDIO_NUM_FREQS=1 python examples/bempp_audio/waveguide_infinite_baffle.py` |
| 3b | `waveguide_cts_rect_320x240_objective.py` | `BEMPPAUDIO_NUM_FREQS=1 python examples/bempp_audio/waveguide_cts_rect_320x240_objective.py` |
| 4 | `piston_minimal.py` | `python examples/bempp_audio/piston_minimal.py` |
| 5 | `phase_plug_acoustic_analysis.py` | `python examples/bempp_audio/phase_plug_acoustic_analysis.py` |
| 6 | `check_installed.sh` | `./check_installed.sh` |
| 7 | (pending) | `python examples/bempp_audio/verdict_scorecard_demo.py` |

---

### Layered Toolchain

- **LEM** (lumped electro-acoustic network): fast, interpretable, and fit-to-measurement.
- **FEM (interior)**: phase plug channels + throat region (complex-valued Helmholtz).
- **FEM (structural shell/plate)**: diaphragm breakup and controlled stiffening features.
- **BEM (exterior)**: radiation, directivity, impedance, and baffle/waveguide loading.

The goal is not "a single best solver", but a layered toolchain where higher-fidelity components can replace approximations as needed.

### Technology Stack

| Library | Role | Used For |
|---------|------|----------|
| **bempp_cl** | Boundary Element Method (this repo's core) | Exterior acoustics: radiation, directivity, impedance. Solves Helmholtz integral equations on surface meshes. Burton-Miller formulation with OSRC preconditioning. Numba (CPU) and OpenCL (GPU) backends. |
| **DOLFINx** (FEniCSx) | Finite Element Method | Interior acoustics (phase plug Helmholtz FEM), structural dynamics (Mindlin-Reissner shell/plate for diaphragm breakup). Requires PETSc/SLEPc for eigenvalue problems. |
| **Gmsh** | Mesh generation | Waveguide surface meshes (revolved profiles, lofted morphing), phase plug volume meshes, dome shell meshes. OCC kernel for CAD-like geometry. |
| **NumPy/SciPy** | Numerics | LEM network solver (sparse matrices, eigenvalues), curve fitting, signal processing. |
| **Plotly** | Visualization | Interactive HTML mesh previews, polar plots, frequency response charts. |

**Dependency tiers:**
- **Core** (always required): `numpy`, `scipy`, `bempp_cl`
- **Meshing** (for geometry): `gmsh` (with OpenCASCADE)
- **FEM** (for interior/structural): `dolfinx`, `petsc4py`, `slepc4py`, `mpi4py`
- **Visualization**: `plotly`, `matplotlib`

### Design Targets (for "smooth SPL" drivers)

- Low ripple in on-axis and listening-window SPL across band
- Controlled directivity (smooth DI, predictable beamwidth transitions)
- Smooth input impedance (avoid sharp reactive features that couple to the motor/network)
- Low HOM content at throat (or at least well-characterized)
- Robustness to tolerances (slot width/land width, dome stiffness variations, alignment)

---

## Part 1: Lumped-Element Driver Network (Panzer-style)

- [x] Implement compression driver lumped network backend (`bempp_audio/driver/*`)
- [x] Add validation examples (vacuum, free radiation, plane-wave tube)
- [x] Add unit/regression tests for key transfer functions
- [x] Parameter fitting / identification (`bempp_audio/driver/fitting.py`)
  - [x] Fit vacuum impedance (Re/Le/Bl/Mms/Cms/Rms + eddy loss) from complex Z_elec
  - [x] Fit acoustic volumes/ducts from tube/termination measurements (config-parameter fitting)
  - [x] Report confidence/identifiability metrics (cond(JᵀJ), covariance when well-conditioned)
- [x] Network extensions for "smooth SPL" design knobs
  - [x] Frequency-dependent losses (Kirchhoff thermo-viscous option for circular ducts via `CompressionDriverNetworkOptions.duct_loss_model`)
  - [x] Frequency-dependent radiation load injection (pass `z_external` as a callable `z(f)` from BEM/FEM)
  - [ ] Non-linear / level-dependent models (high-excursion validation)
    - [ ] `Bl(x)` force factor vs displacement (motor non-linearity)
    - [ ] `Cms(x)` suspension compliance vs displacement (stiffness non-linearity)
    - [ ] Power compression model (`Re(T)` thermal rise → resistance increase)
- [x] Validation suite (golden references)
  - [x] Panzer qualitative behavior checks: peak shifts, coupling/damping trends (unit tests)
  - [ ] Comparison to datasheet curves where available

---

## Part 2: Waveguides (Profiles, Morphing, Meshing)

- [x] CTS-style profiles and morphing (circle → rounded-rect)
- [x] Numeric guardrails for contraction/termination (`check_profile()`, `auto_tune_morph_for_expansion()`)
- [x] Plotly HTML mesh previews (required dependency)
- [x] Designer knobs for smooth directivity
  - [x] Flare schedule: CTS profile (`cts_throat_blend`, `cts_transition`, `cts_tangency`)
  - [x] Throat blend: `cts_throat_blend`, `cts_driver_exit_angle_deg`
  - [x] Mouth tangency: `cts_tangency` in [0,1]
  - [x] Aspect ratio schedule: `MorphConfig` with `profile_mode` options
  - [x] OS coverage angle: `os_opening_angle_deg`
- [x] Geometry constraints
  - [x] Min curvature radius check: `check_min_curvature_radius()` in `profiles.py`
  - [x] Curvature-based mesh refinement: `curvature_based_element_size()`
- [x] Objective metrics (post-BEM solve)
  - [x] DI ripple: `DirectivitySweepMetrics.di_ripple_db`
  - [x] Beamwidth monotonicity: `DirectivitySweepMetrics.beamwidth_monotonicity`
  - [x] Smoothness check: `DirectivitySweepMetrics.is_smooth()`
  - [x] Mouth diffraction proxy (geometric, pre-solve): `mouth_diffraction_proxy()`
  - [x] Directivity objective (scalar): `evaluate_directivity_objective()` (beamwidth target + smoothness penalties)
- [x] Meshing infrastructure
  - [x] Presets: `MeshResolutionPresets` (ultra_fast, super_fast, fast, standard, slow)
  - [x] λ/N frequency rule: `MeshResolutionValidator` with quality thresholds (N≥6=good)
  - [x] Curvature-based adaptive sizing: `curvature_based_element_size()`
  - [x] Element aspect ratio metrics: `ElementQualityMetrics` class
  - [x] Topology validation: `MeshTopologyValidator` (manifold checks)
  - [x] Basic quality report: `LoudspeakerMesh.print_quality_report()`
  - [x] Throat boundary locking: `lock_throat_boundary` + `throat_circle_points` (stable coupling / impedance comparisons)
  - [x] Optional Gmsh optimize pass (guarded when throat is locked)
- [ ] Mesh quality enhancements (optional)
  - [x] Element size histogram visualization (Plotly HTML)
  - [x] Per-element quality color map export (Plotly HTML)

---

### Fluent API ergonomics

- [x] Solver preset helper: `Loudspeaker.solver_preset("ultra-fast"|"super-fast"|...)`
- [x] Mesh preset passthrough: `mesh_preset=...` for `Loudspeaker.waveguide(...)` and `Loudspeaker.waveguide_on_box(...)`
- [x] Unified performance preset (single knob sets mesh + solver + common polar/freq defaults)
- [x] Expose waveguide throat-edge refinement knobs in fluent builders (`throat_edge_refine`, sizes/distances)

---

## Part 3: Exterior Acoustics (BEM)

### Solver Infrastructure
- [x] Burton–Miller solver wrapper (`bempp_audio/solver/base.py`)
  - [x] Operator caching per wavenumber
  - [x] GMRES with configurable tolerance/maxiter
  - [x] DP and P1 function space support
- [x] OSRC preconditioning (`bempp_audio/solver/osrc_solver.py`)
  - [x] Padé approximation (npade=1-4)
  - [x] `AdaptiveSolver` auto-switches at crossover frequency
- [x] FMM backend available (`SolverOptions.use_fmm=True`)
  - [x] Configurable expansion order
  - [x] ExaFMM integration via bempp_cl

### Results Pipeline
- [x] `RadiationResult` container (pressure, impedance, directivity)
- [x] `FrequencyResponse` aggregation with metadata
- [x] `DirectivityPattern` with DI, beamwidth, polar patterns
- [x] `DirectivitySweepMetrics` for design evaluation (DI ripple, monotonicity)

### Parallel Execution
- [x] `solve_frequencies_parallel()` with ProcessPoolExecutor
- [x] Automatic fallback to sequential on PermissionError/OSError
- [x] Progress tracking across workers

### Runtime Robustness
- [x] Examples default to `n_workers=1` (env: `BEMPPAUDIO_N_WORKERS`)
- [x] Examples default to Numba backend (`BEMPP_DEVICE_INTERFACE=numba`)
- [x] Explicit WSL2 detection and `/dev/shm` workaround (low priority)
- [ ] GPU memory overflow graceful fallback (low priority)

### Baffle Modeling
- [x] `InfiniteBaffle` post-processing (2× pressure scaling)
- [x] `CircularBaffle` mesh augmentation (adds rigid baffle domain)
- [ ] True half-space Green's function kernels
  - Requires bempp_cl core changes (`numba_kernels.py`, `opencl_kernels.py`)
  - Would add `helmholtz_*_halfspace` kernel variants with image source
  - Current post-processing is adequate for waveguide mouth near z=0
  - **Status**: Optional enhancement, not blocking for typical use cases

---

## Part 4: Vibroacoustics (Diaphragm FEM + FEM↔BEM + LEM integration)

### Phase 7.1: Dome Geometry + Meshing

- [x] Dome profile generation (`bempp_audio/fea/dome_geometry.py`)
- [x] Dome surface meshing (`bempp_audio/fea/dome_meshing.py`)
- [x] Material definitions and mesh sizing utilities (`bempp_audio/fea/materials.py`)
- [x] Mesh quality validation (`bempp_audio/fea/mesh_quality.py`)

### Phase 7.2: Shell FEM with DOLFINx (Structural)

Implemented baseline MR **flat plate** infrastructure; curved dome shells remain a major next step.

- [x] Canonical validation geometry: `Benchmark3TiConfig_v1` (3" Ti dome, finite clamp ring)
- [x] Implement Mindlin–Reissner formulation (flat plate baseline)
  - [x] 5 DOF per node: (u, v, w, θx, θy)
  - [x] Selective reduced integration for shear locking prevention
  - [x] Material: isotropic elastic (E, ν, ρ, thickness)
- [x] Add modal analysis capability (SLEPc)
  - [x] Eigenvalue solve for natural frequencies and mode shapes
  - [x] Mode shape export for visualization (XDMF)
  - [ ] Validate against analytical circular plate solution (Leissa tables; matching BCs/geometry)
- [x] Add frequency-response solve
  - [x] Harmonic excitation on tagged boundary (ring marker supported)
  - [x] Damping model (Rayleigh)
  - [x] Output: displacement field; velocity via `v=jωw`

**Next for real domes / breakup control**

- [x] Diaphragm benchmark configs (`bempp_audio/fea/benchmarks.py`)
  - [x] `Benchmark3TiConfig_v1` (14.5mm rise spherical cap, 0.75mm clamp band at 36.5–37.25mm)
  - [x] Add “Benchmark3Ti+Former” variant (bonded Kapton/Mylar former + adhesive ring parameters)
  - [x] Add “Benchmark3Ti+Surround” variant (hybrid surround geometry/material for realistic compliance + seal)
  - [x] Parametric sweep helpers (rise, clamp stiffness, bond loss factor, former mass/stiffness)
- [ ] Extend from flat plate → curved shell dome
  - [ ] Curved Mindlin–Reissner / Kirchhoff–Love formulation on 2D manifold meshes (triangulated surfaces)
  - [ ] Local tangent frames + surface gradients/curvatures; invariance to element orientation
  - [ ] Validation: spherical cap / shallow dome benchmarks (literature)
  - [x] Spherical-cap MR shell scaffolding (constant curvature 1/R) for Benchmark3TiConfig_v1
    - [x] `SphericalCapMindlinReissnerShellSolver` (lazy MPI-safe import)
    - [x] Example: `examples/bempp_audio/benchmark3ti_modal_validation.py` (writes mesh; solves if DOLFINx works)
    - [ ] Validation: confirm first 3 modal frequencies land in the expected 10–15 kHz window (and compare to literature/measurement)
- [ ] Corrugations / stiffening ridges (thin-shell, not 3D solid)
  - [ ] Spatial thickness field `h(x)` (and/or section-property map) on shell mesh
  - [ ] Corrugation generator → thickness map + facet tags (rings, radial beads)
  - [ ] Validation: modal shifts scale ~h³ in ridged regions
- [ ] Boundary/interface realism
  - [ ] Clamped/simply-supported/elastic-ring support models
  - [ ] Voice-coil former + glue joint model (critical for HF “harsh vs smooth”)
    - [ ] Former ring as shell/beam (Kapton/Mylar) or rigid ring (lumped) with mass/inertia
    - [ ] Adhesive annulus as viscoelastic interface (normal/tangential stiffness + loss factor)
    - [ ] Eddy-current avoidance lives in the LEM (coil model), not the shell model
    - [ ] “Hot state” sensitivity hooks (adhesive softening + Re(T) power compression)
  - [ ] Ring excitation helpers (distributed traction/moment on tagged ring)
- [ ] Material/damping extensions
  - [ ] Orthotropic/composite shells (coatings, laminates)
  - [ ] Frequency-dependent damping model (complex modulus `E(ω)` / Kelvin–Voigt / SLS / modal loss factors)
    - [ ] Presets: Titanium (near-constant low loss) vs polymer/ketone (rising loss at HF, minimal change near 800 Hz)
    - [ ] Use as the primary toggle for “harsh vs smooth” behavior in Benchmark3TiConfig_v1 comparisons
- [ ] Suspension realism (beyond “dome-only” benchmark)
  - [ ] Embossed suspension patterns (tangential ribs, diamond/M patterns, concentric ripples) as stiffness maps or explicit geometry
  - [ ] Airtight seal modeling: clamp + surround leak sensitivity (ties into low-frequency efficiency)
  - [ ] Fatigue/stress checks (suspension ridges vs yield/fatigue limits) as an optional FEA post-process
- [ ] Modal-acoustic coupling
  - [ ] Modal participation factor: weight structural modes by *acoustic* coupling (throat radiation efficiency / volume-velocity participation)
  - [ ] Prevent “solver distraction”: identify high-Q structural modes that radiate weakly into the throat
  - [ ] Export: `v_mode(x)` + `η_mode(f)` + modal participation vs SPL ripple correlation
- [ ] Performance + export
  - [ ] Modal superposition option for fast FRF sweeps
  - [ ] Export dome velocity `v(x,f)=jωw(x,f)` as a coupling-ready artifact

### Phase 7.3: FEM↔BEM Coupling Interface

- [x] `shell_surface_to_bempp_grid()` for 2D manifold meshes (unit tests pass)
- [x] Velocity transfer utilities (`shell_displacement_to_velocity`, `create_neumann_grid_function`)
- [x] Radiation impedance computation scaffolding (`compute_radiation_impedance`)
- [x] Pressure feedback utilities scaffolding (`iterative_fem_bem_coupling`, `assess_coupling_strength`)
- [ ] Validation
  - [ ] `compute_radiation_impedance` vs analytical piston/sphere impedance
  - [ ] Coupling strength assessment vs known lightweight/heavy radiator cases
- [ ] End-to-end coupled solve demo (stable, documented)
  - [ ] One-way: FEM velocity → BEM pressure/radiation impedance
  - [ ] Two-way: iterate pressure load ↔ structural response (with convergence criteria)

### Phase 7.4: Radiation from Elastic Dome (BEM post-processing)

- [x] `ElasticDomeBEMSolver` scaffolding (unit tests for API, not physics)
- [x] `ModalVelocityProfile` for non-uniform velocity distributions
- [x] `compute_modal_radiation_efficiency()` scaffolding
- [x] `compute_modal_spl()` scaffolding
- [ ] Validation
  - [ ] Pulsating sphere analytical comparison (uniform velocity)
  - [ ] Baffled piston analytical comparison
  - [ ] Mesh convergence study for impedance and directivity
  - [ ] Compare rigid piston vs first-mode velocity profile

### Phase 7.5: Integration with Driver Network (LEM ↔ elastic diaphragm)

- [x] `ElasticDomeProfile` for frequency-dependent effective area (unit tests pass)
- [x] `ElasticDomeNetworkAdapter` scaffolding (wraps network with elastic correction)
- [x] `RigidVsElasticComparison` for breakup analysis scaffolding
- [ ] Validation
  - [ ] Verify `ElasticDomeProfile.from_fem_results()` produces sensible ratios
  - [ ] Compare adapter SPL output to rigid-piston baseline
- [ ] Close the loop with measured data
  - [ ] Replace rigid piston `Sd` with frequency-dependent effective piston `Sd_eff(f)`
  - [ ] Validate network predictions vs measured SPL/impedance changes

---

## Part 5: Interior Acoustics (Phase Plug FEM) + Throat Coupling

### Phase 7.6: Interior Acoustic FEM (Phase Plug Modeling)

- [x] 3D Helmholtz FEM solver scaffolding (`HelmholtzFEMSolver`, unit tests for API)
- [x] Phase plug geometry classes (`PhasePlugGeometry`, `AnnularChannel`, `RadialSlot`)
  - [x] Factory methods: `single_annular`, `dual_annular`, `triple_annular`, `quad_annular`
  - [x] Factory methods: `radial`, `tangerine`, `exponential_annular`
  - [x] Serialization (`to_dict`/`from_dict`) with unit tests
- [x] Phase plug meshing (`PhasePlugMesher`, `PhasePlugMesh`)
- [x] Shell-to-acoustic coupling helpers (`shell_acoustic_coupling.py`)
- [ ] Validation (Helmholtz solver)
  - [ ] Plane wave in duct (analytical)
  - [ ] Cylindrical cavity modes (analytical)

**Throat exit coupling (scaffolding complete, validation pending)**

- [x] `extract_throat_data()` - extract pressure/velocity from FEM (unit tests pass)
- [x] `extract_dome_data()` - extract dome interface data
- [x] `throat_to_bem_monopole()` - equivalent monopole source
- [x] `throat_to_bem_piston()` - equivalent piston source data
- [x] `apply_radiation_bc_at_throat()` - radiation impedance BC helper
- [ ] Validation
  - [ ] Compare monopole strength to expected volume velocity
  - [ ] End-to-end: FEM phase plug → BEM exterior → SPL prediction

**Acoustic metrics (scaffolding complete, validation pending)**

- [x] `ThroatExitData` dataclass (power, impedance, volume velocity properties)
- [x] `PhasePlugMetrics` dataclass (transmission efficiency, uniformity, phase spread)
- [x] `compute_phase_plug_metrics()` - metrics at single frequency
- [x] `compute_metrics_sweep()` - metrics over frequency range
- [x] `PressureFieldVisualization` - pressure field export helpers
- [x] Sample point generators (`create_axial_sample_points`, `create_azimuthal_sample_points`)
- [ ] Validation
  - [ ] Power conservation check (dome input ≈ throat output for lossless)
  - [ ] Impedance transformation vs analytical duct formula
  - [ ] Uniformity metric sanity check on known geometries

**Next for smooth SPL / robust design**
- [ ] Thermo-viscous loss models (critical for smoothing ripple in narrow gaps)
  - [ ] Boundary-layer damping in slits (Stinson effective properties model; “secret sauce” for smooth HF)
    - [ ] Map slit hydraulic diameter → complex k(f), Zc(f) → effective acoustic resistance
    - [ ] Validate vs analytic viscous duct impedance (straight slot) and compare to “lossless” baseline
  - [ ] Narrow gap solver: at 50-100 μm clearances, viscous/thermal losses dominate
  - [ ] Frequency-dependent wall impedance on phase plug walls and throat
  - [ ] Acoustic resistance mapping: visualize where air is "squeezed" (non-linear compression risk)
  - [ ] Validation vs analytic viscous duct impedance (straight slot)
- [ ] Two-port characterization of the phase plug
  - [ ] Transfer matrix / scattering (S11/S21) between dome interface and throat exit
  - [ ] Export equivalent network parameters vs frequency (for Part 1/Phase 7.5)
- [ ] HOM metrics at the throat (Higher-Order Mode analysis)
  - [ ] Project throat field onto duct eigenmodes; compute modal power fractions
  - [ ] `HOM_ratio = (Total_Power - Plane_Wave_Power) / Total_Power`
  - [ ] Target: HOM < 5% up to 15 kHz for "smooth" high-frequency performance
  - [ ] Phase spread metric: `Δφ` variance across throat exit (target < 60° above 10 kHz)
- [ ] Robust meshing + solver diagnostics for thin channels
  - [ ] Automated refinement rules for hydraulic diameter / ka targets
  - [ ] Preconditioner options + convergence/stagnation diagnostics
  - [ ] Regression tests across mesh presets (super-fast/fast/medium)
- [ ] Manufacturing tolerance / design sweeps
  - [ ] Param sweeps: slot width, land width, depth, exit blend, misalignment
  - [ ] Metrics: throat SPL ripple, impedance smoothness, group delay smoothness

---

## Part 6: Validation, Optimization, and Automation

- [ ] Phase 8: Visualization Outputs (Designer Dashboards)
  - [x] Interactive Plotly dashboard export: `bempp_audio.viz.save_driver_dashboard_html()`
  - [x] Default multi-panel layout (SPL+listening-window/polar/DI/beamwidth/Z_elec/excursion/throat Z/radiation Z + scorecard)
  - [x] Scorecard thresholds + banded pass/warn/fail (SPL ripple in 1–2/2–5/5–10/10–20 kHz bands)
  - [x] Cross-linked resonance markers (auto peaks from `|Z_elec|` and `|Z_throat|` + XO/20k + optional FEM modes)
  - [x] Optional phase-plug panels when metrics are provided (η, uniformity, phase spread)
  - [ ] Add optional panels (power, group delay, HOM fractions)
    - [x] Radiated power panel
    - [x] Group delay panel
    - [ ] HOM fractions panel
  - [x] Standardize key examples to write `logs/<example>_dashboard.html` (waveguide infinite baffle, Panzer validation, phase plug metrics)
  - [x] Standardize remaining examples to write `logs/<example>_dashboard.html`
  - [x] Documentation: “how to read the dashboard” + recommended targets for smooth SPL/DI
- [ ] Define standard output “scorecards” for each design iteration
  - [x] Ripple metrics (SPL/DI) and thresholds (JSON export)
  - [ ] HOM proxy metrics at throat
  - [ ] Manufacturing sensitivity summary
- [ ] Parameter sweep harness
  - [x] Reproducible configs (snapshot JSON per case)
  - [ ] Caching of meshes/operators
  - [x] Parallel sweep execution with WSL2-safe defaults
  - [x] Report generation (Plotly dashboards + JSON scorecards)
- [x] CI-friendly smoke tests
  - [x] `check_installed.sh` always reports required deps (dolfinx/bempp/exafmm/plotly)
  - [x] Run a minimal waveguide solve (single freq) in a predictable mode
  - [x] Offline `save_driver_dashboard_html` smoke (fake response; embedded Plotly JS, no CDN)
  - [x] Optional real-BEM `save_driver_dashboard_html` smoke (tiny piston solve; gated by `BEMPPAUDIO_RUN_SLOW=1`)

---

## Part 7: Synthesis & Verdict Engine

This phase moves beyond simulation output into **Decision Science**. The goal is to identify destructive interference between electricity, mechanics, and acoustics.

### Three Lens Filters for Analysis

**Filter 1: Energy Conversion Lens (Efficiency & Thermal)**
- [ ] Transduction efficiency η = P_acoustic / P_electrical (target: 20-35% for world-class)
- [ ] Power compression PC_dB tracking (thermal model → Re(T) rise → resistance doubling at 230°C)
- [ ] Thermal runaway detection (if Re ramps too fast → flag former material change)

**Filter 2: Acoustic Integrity Lens (Phase Plug & Cavity)**
- [ ] HOM ratio at throat (target < 5% up to 15 kHz)
- [ ] Phase spread Δφ at throat (target < 60° above 10 kHz)
- [ ] Phase plug acts as "temporal lens": all wavefront parts arrive at throat simultaneously

**Filter 3: Vibroacoustic Lens (Structural Breakup)**
- [ ] Modal participation vs acoustic ripple correlation
  - Good: structural mode exists but phase plug damps it → smooth SPL
  - Bad: structural mode correlates with 3dB+ SPL peak/dip → harsh sound
- [ ] Settling time analysis (CSD/spectrogram): high-Q ringing at 15 kHz = fail
- [ ] Controlled breakup: many small peaks preferred over one giant resonance

### Golden Verdict Scorecard

| Priority | Metric | Target (3" Ti) | Impact if Failed |
|----------|--------|----------------|------------------|
| **High** | HOM Ratio @ 15kHz | < 5% | Muddiness, loss of HF detail |
| **High** | SPL Ripple (1k-20k) | ± 1.5 dB | Colored, non-neutral sound |
| **Med** | Power Compression | < 1 dB @ rated | Dynamic flattening |
| **Med** | f₁ Modal Q | < 10 | Harshness, metallic ringing |
| **Low** | Max Excursion | < 0.5 mm | Mechanical distortion |

### Automated Verdict Logic
- [ ] Parse results from Parts 1-5 → populate scorecard
- [ ] Pass/Warn/Fail classification with configurable thresholds
- [ ] Hard flag: `HOM_CRITICAL` if HOM ratio > 5% at any point above 10 kHz
- [ ] Subjective↔objective correlation mapping (“Harshness Score” calibration)
  - [ ] Harshness Score = `w_Q * max(Q_breakup)` + `w_ripple * std(SPL_LW)` + `w_HOM * max(HOM)`
  - [ ] Diagnosis routing:
    - High HOM + smooth structural → **fix Phase Plug**
    - Low HOM + sharp SPL spikes → **fix Diaphragm/Former bond**
- [ ] Auto cross-annotate FEM modal frequencies on BEM SPL/impedance plots (vertical markers)
  - Note: dashboard already supports manual `structural_mode_hz=[...]` overlay

### Sensitivity & Tolerance Analysis
- [ ] Tolerance sweep framework
  - [ ] Titanium thickness ±5 μm → 15 kHz response impact
  - [ ] Phase plug alignment ±1 mm → HOM/phase spread impact
  - [ ] Dome height ±0.5 mm → modal frequency shifts
  - [ ] Bondline loss/stiffness variation → modal Q + ripple impact
- [ ] Manufacturing robustness score (% of tolerance space that passes verdict)

### Quick-Mode Guidance
- [ ] `verdict_scorecard_demo.py` should load cached full-band sweeps (20 Hz–20 kHz) for DI/HOM/ripple validity
  - [ ] Quick-mode smoke test only verifies data plumbing and should print a warning:
        “Metrics based on cached 20 Hz–20 kHz data for valid DI/HOM analysis.”

---

## References

- Leissa, A.W. “Vibration of Plates” (1969)
- Panzer, J. “Modeling of a Compression Driver using Lumped Elements”, ICA 2019
- Kirkup, S. “The Boundary Element Method in Acoustics” (2019)
- Antoine & Darbas (OSRC), M2AN 2007
