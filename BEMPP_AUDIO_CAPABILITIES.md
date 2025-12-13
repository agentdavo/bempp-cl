# bempp_audio Capabilities (Concise)

`bempp_audio` is a loudspeaker / acoustic radiation simulation layer built on top of `bempp_cl`. It focuses on practical geometry pipelines (especially waveguides), boundary-condition assignment by domain, and a fluent API for running BEM frequency sweeps and extracting acoustically meaningful outputs.

## What you can do

### Geometry + meshing (Gmsh OCC where needed)
- Create primitive radiator meshes: circular pistons, simple shapes, imports via supported mesh paths.
- Generate **axisymmetric waveguides** by revolving a meridian curve.
- Generate **non-axisymmetric waveguides** (e.g. rounded-rectangle mouths) by cross-section sampling + lofting.
- Create **waveguide-on-box** (“unified enclosure”) meshes where the waveguide mouth loop is shared with the baffle cutout (shared edge by construction).
- Control mesh sizing and discretization:
  - graded element size from throat→mouth
  - circumferential and axial resolution controls
  - optional mouth-edge refinement targeting the aperture rim
  - avoid spline overshoot via polyline-meridian option

### Waveguide design profiles (radius vs axis)
- Conical, exponential, hyperbolic.
- CTS (“constant-directivity style”): throat blend → conical mid → mouth blend toward baffle tangency.
- Oblate-spheroidal (“OS”) family with feasibility bounds for a given geometry.
- Tractrix-like legacy curve and a tractrix-horn-like contour option.

### Mouth-shape morphing (non-axisymmetric cross-sections)
- Morph circular throat cross-sections to ellipse / rounded rectangle / superellipse / superformula targets.
- Choose morph timing models:
  - `profile_mode="axes"` (recommended for high aspect ratios; avoids short-axis contraction without changing mouth size)
  - `profile_mode="radial"` (morph timing tied to scalar profile radius; can show short-axis contraction near the mouth for rectangular targets)
- Optional enforcement to prevent contraction (axes or multi-direction), with the tradeoff that it can inflate the final mouth dimensions.

### Pure-math guardrails + diagnostics (no Gmsh required)
- `check_profile(...)`: numeric monotone-expansion checks (axisymmetric `r(x)`, directional support, and area monotonicity).
- `auto_tune_morph_for_expansion(...)`: searches morph timing parameters to reduce contraction without changing target dims.
- CSV export for diagnosing XZ/YZ short-axis behavior and baffle termination:
  - `python -m bempp_audio.mesh.profile_export --config-json <cfg.json> --out <file.csv>`

### Boundary conditions
- Assign normal velocity by domain ID (physical groups) or by name mappings.
- Use standard velocity profiles (piston, zero/rigid) and domain-dependent compositions.

### BEM solve orchestration (bempp_cl)
- Burton–Miller exterior Helmholtz solve for prescribed normal velocity.
- Frequency sweeps (serial; parallel when environment supports it).
- Optional OSRC preconditioning path (when enabled/configured).
- Optional fast multipole (FMM) operator assembly path (when enabled/configured).

### Environments / baffles
- Free-space radiation.
- Infinite-baffle approximation (post-process scaling + hemisphere suppression).
- Finite baffle workflows (where used by mesh/domain modeling).

### Results + post-processing
- Field evaluation at points and far-field/directivity evaluation.
- Polar sweeps, directivity index, beamwidth curves.
- Radiated power and SPL conventions (complex peak → RMS for SPL).
- Radiation impedance (specific / mechanical) derived from the surface solution.
- Plot/export helpers (matplotlib/plotly optional) including summary reports and CEA-2034-style views where applicable.

