bempp_audio
==========

``bempp_audio`` is a loudspeaker/acoustics-oriented API layer built on top of
``bempp_cl``. It provides:

- a fluent “builder” interface (``Loudspeaker``) for assembling simulations
- Gmsh-based mesh generation (including waveguides and enclosures)
- BEM solvers (Burton–Miller; optional OSRC preconditioning)
- results and post-processing (fields, directivity, impedance, plotting/export)

Quickstart
----------

Install (Debian/Ubuntu)
~~~~~~~~~~~~~~~~~~~~~~~

**First-time system install (installs apt deps + Gmsh SDK + DOLFINx):**

.. code-block:: bash

   ./install_dolfinx_debian.sh --venv ./venv

This will use ``sudo`` internally for apt packages, but installs Python packages
into your venv as the current user.

**Complex Helmholtz FEM note (PETSc):**

For complex-valued Helmholtz FEM with DOLFINx/PETSc, you need a *complex-scalar*
On Ubuntu 24.04, ``python3-dolfinx-complex`` and ``python3-petsc4py-complex`` are
available via apt, but your *default* PETSc selection can still be real-scalar
depending on ``PETSC_DIR`` / system alternatives. If ``./check_installed.sh``
reports a real scalar type, set ``PETSC_DIR`` to the distro complex PETSc tree:

.. code-block:: bash

   export PETSC_DIR=/usr/lib/petscdir/petsc*/x86_64-linux-gnu-complex

Optional smoke tests (end of installer):

.. code-block:: bash

   ./install_dolfinx_debian.sh --venv ./venv --run-smoke-tests --run-fem-bem-example

After installing, run:

.. code-block:: bash

   ./check_installed.sh

Run the end-to-end waveguide example (generates a mesh, solves a frequency sweep,
and produces a summary plot):

.. code-block:: bash

   ./waveguide_infinite_baffle.sh

Or directly:

.. code-block:: bash

   python examples/bempp_audio/waveguide_infinite_baffle.py

Useful environment variables used by the example:

- ``BEMPP_DEVICE_INTERFACE``: compute backend (often ``numba`` for CPU).
- ``BEMPPAUDIO_MESH_PRESET``: mesh sizing preset (e.g. ``super-fast``).

Fluent API presets (recommended for new code):

.. code-block:: python

   from bempp_audio import Loudspeaker

   speaker = (
       Loudspeaker()
       .performance_preset("ultra-fast", mode="horn")   # mesh + solver + common sweep defaults
       .waveguide(throat_diameter=0.025, mouth_diameter=0.2, length=0.12, profile="cts")
       .infinite_baffle()
   )

You can also pass mesh presets directly to waveguide builders:

.. code-block:: python

   Loudspeaker().waveguide(..., mesh_preset="fast")

Directivity-driven workflow tip:

- Use `bempp_audio.results.evaluate_directivity_objective()` to score a candidate waveguide against
  horizontal/vertical beamwidth targets over a frequency band, then sweep depth (`length`) and CTS knobs.
- For cheap “are we on track?” iterations, run only a few frequencies (e.g. 5 log-spaced points in 1–16 kHz) and
  use `di_mode="proxy"` (beamwidth-derived DI proxy). Switch to `di_mode="balloon"` for final validation.

Waveguides: profiles, morphing, and sanity checks
-------------------------------------------------

Coordinate conventions (important):

- ``+z`` is forward (radiation direction)
- mouth/baffle plane is ``z = 0``
- throat plane is at ``z = -length``

Profiles (axisymmetric radius vs axial position) live in ``bempp_audio/mesh/profiles.py``.
Common waveguide options include:

- ``profile_type="cts"``: a practical “constant-directivity style” contour
  (throat blend → conical section → mouth blend toward baffle tangency)
- ``profile_type="os"`` / ``"oblate_spheroidal"``: oblate-spheroidal family
- ``profile_type="tractrix_horn"``: tractrix-horn-like contour

Non-axisymmetric mouths (e.g. rounded rectangles) are produced by morphing and lofting:

- Prefer ``MorphConfig(..., profile_mode="axes")`` for high-aspect-ratio mouths to
  avoid short-axis “waist” contraction without changing the final mouth dimensions.
- ``profile_mode="radial"`` can exhibit short-axis contraction near the mouth for
  rectangular targets; enforcement can prevent contraction but can inflate the mouth.

Numeric guardrail (pure-math, no Gmsh required):

.. code-block:: python

   from bempp_audio.mesh.waveguide_profile import check_profile
   rep = check_profile(waveguide_cfg, n_axial=200, n_directions=36)
   assert rep.ok

CTS advanced knobs (for HF ripple / termination shaping):

- ``cts_tangency``: how strongly the mouth tends toward baffle tangency (0..1)
- ``cts_mouth_roll``: spends more of the mouth-blend length near-horizontal (0..1)
- ``cts_curvature_regularizer``: trades tangency vs peak mouth curvature (>=0)
- ``cts_mid_curvature``: gentle OS-like curvature in the mid-section (0..1)

CSV export for debugging contraction/termination (pure-math):

.. code-block:: bash

   python -m bempp_audio.mesh.profile_export \
     --config-json logs/waveguide_cfg.json \
     --out logs/profile.csv \
     --n-axial 250

Cabinet geometry (chamfers and fillets)
---------------------------------------

When using ``waveguide_on_box()``, the enclosure's front-face perimeter edges
(where the baffle meets the side walls) can be chamfered or filleted for more
realistic diffraction behavior:

.. code-block:: python

   from bempp_audio import Loudspeaker
   from bempp_audio.mesh.cabinet import ChamferSpec

   speaker = (
       Loudspeaker()
       .performance_preset("standard", mode="horn")
       .waveguide_on_box(
           throat_diameter=0.025,
           mouth_diameter=0.32,
           waveguide_length=0.18,
           box_width=0.52,
           box_height=0.44,
           box_depth=0.30,
           profile="cts",
           # Symmetric 20mm chamfer on all four front-face edges
           cabinet_chamfer_top=ChamferSpec.symmetric(0.020),
           cabinet_chamfer_bottom=ChamferSpec.symmetric(0.020),
           cabinet_chamfer_left=ChamferSpec.symmetric(0.020),
           cabinet_chamfer_right=ChamferSpec.symmetric(0.020),
       )
   )

**ChamferSpec factory methods:**

- ``ChamferSpec.symmetric(distance)``: 45-degree chamfer, same distance on both faces
- ``ChamferSpec.asymmetric(d_baffle, d_side)``: different distances on baffle vs side
- ``ChamferSpec.angled(distance, angle_deg)``: specified angle (45 = symmetric)

**Fillet (roundover) alternative:**

Use ``cabinet_fillet_*_radius`` parameters instead of chamfers:

.. code-block:: python

   .waveguide_on_box(
       ...,
       cabinet_fillet_top_radius=0.015,    # 15mm roundover
       cabinet_fillet_bottom_radius=0.015,
   )

**Note:** Chamfers and fillets are mutually exclusive per edge.

**Environment variables (for example scripts):**

The ``waveguide_cts_rect_320x240_objective.py`` example supports:

- ``BEMPPAUDIO_CHAMFER_ALL_MM``: uniform chamfer on all edges (default: 20)
- ``BEMPPAUDIO_CHAMFER_TOP_MM``, ``BEMPPAUDIO_CHAMFER_BOTTOM_MM``, etc.: per-edge overrides

Infinite baffle note
--------------------

``InfiniteBaffle`` is currently implemented as a pragmatic post-process layered on
top of a full-space solve:

- rear half-space is suppressed in field/far-field evaluations
- surface pressure can be scaled by a factor (default 2)

This is not a half-space Green’s function / kernel-based solve.

Package map (architecture)
--------------------------

- ``bempp_audio/api/``: Fluent API façade (``Loudspeaker``) and orchestration helpers.
- ``bempp_audio/api/state.py``: Immutable state dataclasses used by the fluent API.
- ``bempp_audio/mesh/``: Mesh generation + validation utilities (Gmsh-based where needed).
- ``bempp_audio/fea/``: FEM utilities (thin-shell domes, phase plug geometry/meshing, FEM-BEM coupling scaffolding).
- ``bempp_audio/solver/``: BEM solve wrappers (Burton–Miller + OSRC preconditioner).
- ``bempp_audio/results/``: Result containers and derived quantities (directivity, fields, impedance).
- ``bempp_audio/driver/``: Lumped electro-acoustic compression-driver network + generic25 parsing.
- ``bempp_audio/io/``: Serialization and file I/O helpers.
- ``bempp_audio/viz/``: Plotting/export helpers (optional deps: matplotlib/plotly).
- ``bempp_audio/config.py``: Structured config dataclasses + presets.

Side effects policy
-------------------

To keep simulation code predictable and testable:

- Computation-oriented code (``api/``, ``mesh/``, ``solver/``, ``results/``) should not
  write files by default.
- File writing belongs in ``io/`` and ``viz/`` (explicit calls like ``save_*`` / ``*_html``).
- Optional dependencies (Gmsh, matplotlib, plotly) should be imported lazily/guarded so
  that core imports work.

Logging and reporting
---------------------

Examples and long-running workflows should route all user-facing output through
``bempp_audio.progress`` (not ``print()``) so console output and log files remain consistent.
