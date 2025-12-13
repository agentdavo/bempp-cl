# AI Agent Guide (bempp-cl + bempp_audio)

This repository contains:
- `bempp_cl`: a Boundary Element Method (BEM) library (Laplace/Helmholtz/modified Helmholtz/Maxwell) with OpenCL kernels and a Numba fallback.
- `bempp_audio`: a loudspeaker/acoustic radiation simulation layer built on top of `bempp_cl` (mesh generation, solver orchestration, results, plotting, and driver coupling).

## Quick Orientation

**Core idea**: solve an exterior Helmholtz problem on a triangulated surface mesh with boundary integral operators; in `bempp_audio` the usual boundary condition is prescribed normal velocity (Neumann BC).

**Domain indices matter**: many `bempp_audio` workflows assign velocities/BCs by Gmsh physical groups (“domains”). These are typically `1,2,...` (not guaranteed to be zero-indexed).

## Repo Layout (high signal)

- `bempp_cl/api/`: public API (grids, spaces, operators, solvers, shapes)
- `bempp_cl/core/`: computational backend (OpenCL/Numba kernels, assembly)
- `bempp_audio/api/`: fluent façade (`Loudspeaker`) + presets + solve orchestration
- `bempp_audio/mesh/`: piston/box/waveguide/unified enclosure mesh generators, profiles, morphing
- `bempp_audio/solver/`: frequency sweeps (serial/parallel), Burton–Miller/OSRC variants
- `bempp_audio/results/`: response containers and derived metrics
- `bempp_audio/viz/`: matplotlib/plotly backends and summary reports
- `bempp_audio/driver/`: compression-driver lumped network + parsers
- `examples/`: runnable example scripts
- `test/`: unit + validation tests

## Common Commands

```bash
python -m pip install -e .
python -m pip install -e ".[test]"
python -m pytest test/unit
python -m pytest test/validation
python -m ruff check bempp_cl bempp_audio
python -m ruff format --check bempp_cl bempp_audio
```

Pytest options are defined in `test/conftest.py` (device backend, precision, vectorization, optional dependency toggles).

## bempp_audio “How it Fits Together”

Typical high-level flow (fluent API):
1) Build/import a mesh (`LoudspeakerMesh` / waveguide/unified enclosure generators)
2) Assign an environment (free space / infinite baffle approximation / enclosure)
3) Define a `VelocityProfile` (often by domain)
4) Solve a frequency sweep to get a `FrequencyResponse`
5) Plot/export derived metrics

Key entry points:
- Fluent API: `bempp_audio/api/loudspeaker.py`
- Orchestration: `bempp_audio/api/solve.py`
- Mesh generation: `bempp_audio/mesh/*` (notably `waveguide.py`, `unified_enclosure.py`, `morph.py`)
- Velocity profiles: `bempp_audio/velocity/profiles.py`
- Plotting: `bempp_audio/viz/mpl.py` and `bempp_audio/viz/plotter.py`

## Current Work / Roadmap

Current capabilities and high-level direction are summarized in the BEMPP_ md files.

Maintainability direction:
- Prefer explicit/typed APIs, immutable state updates, and small modules with minimal side effects.
- Breaking changes are acceptable when removing legacy shims and reducing duplication.

## Known Sharp Edges

If you touch these areas, prioritize correctness and add/adjust tests:
- Frequency presets vs config mismatch: presets reference `spacing="octave"` / `points_per_octave`, but config and `Loudspeaker.frequency_range()` don’t currently support that shape consistently.
- Driver metrics ordering: `FrequencyResponse.add()` sorts results, but driver metric arrays may not be kept in the same order if inserts happen out of order.
- `LoudspeakerMesh.from_meshio()` behavior: currently routes to STL import; docstrings may overpromise.

## Examples

- Run all examples: `./run_all_examples.sh`
- Individual examples: `examples/bempp_audio/` (e.g., `waveguide_infinite_baffle.py`)

## Development Conventions (lightweight)

- Keep optional dependencies importable: prefer “import at use-site” and consistent optional-import helpers where patterns already exist.
- Don’t assume domain IDs are contiguous or start at 0; preserve physical group semantics.
- When adding features to results containers, keep data keyed/sorted consistently by frequency.
