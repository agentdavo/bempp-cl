# CTS Waveguide Profile (Conical-to-Tangent Smoothstep)

This note documents the **CTS** (“Conical-to-Tangent Smoothstep”) waveguide meridian profile used by `bempp_audio`.

Primary implementation: `bempp_audio/mesh/profiles.py` (`cts_profile`).

## Coordinate conventions

The profile function is written as an **axisymmetric meridian** radius `r(x)`:

- `x ∈ [0, L]` where `x=0` is the **throat plane** and `x=L` is the **mouth plane**.
- In the mesh conventions, the mouth lies on the baffle plane `z=0` and the throat is at `z=-L`, but the profile code itself uses `x` measured from throat→mouth.

Define the **wall angle** `α` measured from the axis (`+z` direction):

- `α = 0°` → wall is parallel to axis (vertical).
- `α = 90°` → wall is horizontal → **tangent to the baffle plane** at the mouth.

Because the curve is `r(x)`, the geometric slope relation is:

`dr/dx = tan(α(x))`  ⇒  `α(x) = arctan(dr/dx)`

## Design intent (3 stages)

CTS is designed to be practical for “constant-directivity style” waveguides:

1. **Throat blend** (match driver exit angle): smooth transition from a driver exit cone into the waveguide.
2. **Conical mid-section** (quasi-conical propagation): maintain a mostly constant wall angle over much of the length.
3. **Mouth blend** (toward baffle tangency): smoothly increase wall angle near the mouth to reduce a hard termination.

## Parameters

In `cts_profile(x, throat_r, mouth_r, length, ...)`:

- `throat_blend = t_tb` (normalized, `[0, 1)`)  
  Where the throat blend ends. `0` disables the throat blend stage.
- `transition = t_c` (normalized, `[0, 1)`)  
  Where the conical stage ends and the mouth blend begins.
- `driver_exit_angle_deg`  
  Driver exit half-angle at the throat (used only when `throat_blend > 0`).
- `throat_angle_deg`  
  Optional explicit conical mid-section angle override.
- `tangency = τ` in `[0, 1]`  
  A knob to make the mouth more (or less) baffle-tangent:
  - `τ=0`: pure conical exit (no extra mouth flare)
  - `τ=1`: strong tangency preference, still subject to curvature regularization and endpoint feasibility
- `mouth_roll` in `[0, 1)`  
  4th knob controlling **how much of the mouth-blend length** is spent near the final (near-horizontal) angle:
  - `0`: symmetric smoothstep blend
  - `→1`: reaches the final angle earlier and stays near-horizontal longer (often reduces HF ripple sensitivity at the termination)
- `curvature_regularizer ≥ 0`  
  Curvature/tangency tradeoff weight:
  - `0`: ignore tangency preference when selecting the mid conical angle (tends toward “minimum curvature” / more conical)
  - higher values: track the tangency target more closely
- `mid_curvature` in `[0, 1]`  
  Adds a gentle zero-endpoint “bump” to the conical section wall angle to emulate **OS-like mid-section curvature** while keeping endpoints conical.

Constraints enforced in the config layer as well:

- `0 ≤ throat_blend < transition < 1`
- `0 ≤ tangency ≤ 1`
- `mouth_r > throat_r`, `L > 0`

## Core mathematical construction

### Normalized coordinate

Let:

- `t = x/L` in `[0, 1]`
- `t_tb = throat_blend`
- `t_c = transition`

CTS is defined by specifying the wall angle `α(t)` and integrating `tan(α)` to get radius.

### Smoothstep blend function

CTS uses a **quintic smoothstep** (zero 1st and 2nd derivatives at endpoints):

`S(u) = 10u³ − 15u⁴ + 6u⁵`, for `u ∈ [0, 1]`

This ensures that when `α(t)` transitions between angles, it does so smoothly (no kinks in slope/curvature in the *angle schedule*).

### Piecewise angle schedule (3 sections)

Let:

- `α_d` = driver exit angle
- `α_c` = conical mid-section angle
- `α_m` = mouth angle (chosen to satisfy the geometry constraint)

Then the CTS wall angle schedule is:

**Section 1 (throat blend):** `0 ≤ t < t_tb` (only if `t_tb > 0`)

`α(t) = α_d + (α_c − α_d) * S(t/t_tb)`

**Section 2 (mid-section):** `t_tb ≤ t < t_c`

Baseline CTS uses a constant mid-angle:

`α(t) = α_c`

If `mid_curvature > 0`, CTS adds a gentle “OS-like” bump with zero endpoints:

`α(t) = α_c + 0.15 * mid_curvature * α_c * sin(π u)`, with `u = (t − t_tb)/(t_c − t_tb)`

**Section 3 (mouth blend):** `t_c ≤ t ≤ 1`

Baseline CTS uses a smoothstep blend:

`α(t) = α_c + (α_m − α_c) * S(u)`, with `u = (t − t_c)/(1 − t_c)`

If `mouth_roll > 0`, CTS first “rolls” the mouth-blend coordinate so the profile reaches the final angle earlier:

`u' = 1 − (1 − u)^γ`, with `γ = 1/(1 − mouth_roll)`

Then it evaluates the smoothstep at `u'`:

`α(t) = α_c + (α_m − α_c) * S(u')`

### Radius as an integral of slope

From `dr/dx = tan(α)` and `x = Lt`:

`dr/dt = L * tan(α(t))`

So:

`r(t) = r_throat + L * ∫₀ᵗ tan(α(s)) ds`

In code, this integral is computed numerically (trapezoid rule on a fine `t` grid, `n_fine=1000`), then interpolated to the caller’s `x` samples.

## How CTS chooses the conical and mouth angles

The profile must satisfy the endpoint constraint:

`r(1) = mouth_r`  ⇔  `∫₀¹ tan(α(s)) ds = (mouth_r − throat_r)/L`

CTS does this in two coupled steps:

### Step 1: baseline “simple conical” angle

Define:

`α_simple = arctan((mouth_r − throat_r)/L)`

This is the wall angle of a pure cone matching both endpoints.

### Step 2: choose `α_d`, select `α_c`, then solve for `α_m`

**Driver angle `α_d`:**

- If `driver_exit_angle_deg` is given: `α_d = that`
- Otherwise: `α_d = α_simple`

**Conical angle `α_c` (adaptive + curvature-regularized):**

- If `throat_angle_deg` is specified: `α_c = that` (user override)
- Otherwise, CTS computes an **adaptive minimum conical factor** based on:
  - expansion ratio `mouth_r/throat_r`
  - available mouth-blend length `(1 − t_c)`

  This replaces the older fixed `0.70` floor with a geometry-aware lower bound. In code, the bound is expressed as a *minimum factor* in a conservative range:

  `α_c ∈ [α_simple * min_factor(expansion, 1−t_c), α_simple]`

- CTS then performs a **1D scan** over candidate `α_c` values in that interval. For each candidate:
  1) solve `α_m` by binary search so that `r(L)=mouth_r` (with the current `mouth_roll` and optional `mid_curvature` shaping),
  2) compute peak curvature over the mouth-blend region (`t ≥ t_c`),
  3) add a soft penalty for deviating from the tangency-driven mouth-angle target.

This yields a practical tradeoff: strong tangency preference when `τ→1`, but avoiding extreme mouth curvature spikes that often correlate with DI/beamwidth ripple at high frequencies.

### Curvature-regularized selection (informal)

In implementation terms:

- For `τ>0`, define a target mouth angle `α_m_target` that interpolates between “pure conical” and “strong tangency preference”.
- For each candidate `α_c`:
  - solve `α_m(α_c)` from the endpoint constraint,
  - compute `k_max` = max curvature in the mouth-blend region,
  - minimize a cost of the form:

`cost(α_c) = log(1 + k_max * L) + curvature_regularizer * (angle_error)^2`

where `angle_error` is the normalized deviation of `α_m(α_c)` from `α_m_target`.

**Mouth angle `α_m`:**

Given (`α_d`, `α_c`), CTS finds `α_m` by **binary search**:

- Search interval: `α_m ∈ [α_c, 89.9°]`
- Objective: `r(1; α_d, α_c, α_m) = mouth_r`
- Monotonicity: increasing `α_m` increases `tan(α)` near the mouth, increasing the integral, increasing `r(1)` → binary search is valid.

Special case:

- If `τ=0`: CTS sets `α_m = α_c = α_simple` → pure conical.

## What is “maximum achievable tangency” here?

CTS does not try to force `α_m → 90°`. Instead it:

- uses geometry-aware bounds for how much the mid-section can be “relaxed”, and
- chooses the final `(α_c, α_m)` pair to satisfy `r(L)=mouth_r` while keeping mouth curvature reasonable.

Geometries with large expansion ratios and/or long lengths will allow `α_m` to be closer to 90°.
Shallow geometries may be constrained to a smaller `α_m` even when `τ=1`.

## Practical tuning guidance

Recommended starting values (typical horn/waveguide workflows):

- `throat_blend` ≈ `0.10–0.20` (10–20% of length)
- `transition` ≈ `0.70–0.85` (start mouth tangency blend late)
- `driver_exit_angle_deg`: match your compression driver exit cone (often 5–15°)
- `tangency`:
  - start with `1.0` for baffle-friendly termination
  - reduce toward `0.5` if the mouth region becomes too “hockey-stick” for your tastes
- `mouth_roll`:
  - start with `0.0` (baseline CTS)
  - try `0.5–0.8` if you see HF ripple associated with the mouth termination; it tends to make the last section “settle” earlier
- `curvature_regularizer`:
  - increase if you want `tangency` to dominate more strongly
  - decrease if you want the algorithm to prioritize minimizing curvature spikes
- `mid_curvature`:
  - keep `0.0` if you want a strictly conical mid-section
  - try `0.1–0.3` for a mild OS-like “bulge” while preserving conical endpoints

## Throat mesh / throat impedance considerations

High-frequency directivity and throat impedance estimates are often limited by **throat discretization**:

- Circumferential sampling on the throat rim should be fine enough to represent the shortest wavelength in the band.
- A simple rule of thumb is to target a rim edge length comparable to the **throat element size** `h_throat`.

For an axisymmetric 1" throat (`d=25.4mm`, circumference ≈ 79.8mm):

- If you want ~2.5mm spacing around the rim (typical “good” HF behavior up to ~16 kHz),
  `N_points ≈ 79.8 / 2.5 ≈ 32` points around the circle.

In `bempp_audio.mesh.waveguide.WaveguideMeshConfig`:

- `lock_throat_boundary=True` constrains the Gmsh discretization of the throat rim so the node count is reproducible
  (useful for FEM↔BEM coupling and for stable impedance comparisons across iterations).
- `throat_circle_points=N` sets the target rim point count for **axisymmetric/revolved** waveguides.
  For lofted/morphed waveguides, the rim vertices are already explicitly defined by `n_circumferential`.

Note: Gmsh “optimize” passes can move nodes; the generator disables optimization when `lock_throat_boundary=True`.

## Directivity objective (recommended workflow)

CTS parameter tuning is best done with an explicit **directivity objective**:

- target beamwidth in horizontal and vertical planes (e.g. -6 dB beamwidth targets),
- plus penalties for DI ripple and beamwidth non-monotonicity over the design band.

This repo provides a scalar objective helper in `bempp_audio.results.objectives` that can be used in sweeps.

If you want strict control over the mid-section wall angle (e.g., a known coverage strategy), set `throat_angle_deg` explicitly and let CTS solve only for `α_m`.

Concrete recommendation: for a rectangular 320×240mm mouth with a 1" throat, depth `L` should be chosen by *minimizing the directivity objective* over a depth sweep (because shallow vs deep is a trade between low-frequency pattern “set” and high-frequency smoothness). Use the objective module to compare candidates rather than relying on visual intuition about “shallow CD waveguides”.

### Cheap “are we on track?” probe runs

Directivity objectives are only as expensive as the frequency solves you perform.
A good workflow is:

1. **Probe** with a very small number of frequencies (e.g. 5 log-spaced points in 1–16 kHz) to rank variants cheaply.
2. **Validate** the best few candidates with a denser frequency grid (e.g. 28–64 points).

For objective evaluation speed, `evaluate_directivity_objective` supports:

- `di_mode="proxy"` (default): uses a DI proxy derived from horizontal/vertical beamwidths (fast).
- `di_mode="balloon"`: computes true DI from a 3D balloon integration (slow; use for final validation).

## Relationship to meshing and non-axisymmetric mouths

`cts_profile` only defines the **axisymmetric meridian** `r(x)`.

If `MorphConfig` is enabled (e.g. circle→rounded-rectangle), the mesh generator samples cross-sections along `x` and lofts them; the **three-stage CTS schedule still governs the radial envelope** used to size/morph the section.

## Known limitations / implementation notes

- CTS uses numeric integration and interpolation, not a closed-form `r(x)`. This is intentional to keep the angle schedule simple and robust across parameter ranges.
- When `throat_angle_deg` is not specified and `tangency>0`, CTS does a small scan over candidate `α_c` values (each evaluation includes a binary search for `α_m`), so it is more expensive than a simple analytic `r(x)` formula. This is still negligible compared to meshing/BEM, but it matters if you call `cts_profile()` millions of times in an optimizer.
- `mid_curvature` is an *angle-bump heuristic* (not a true OSWG family). It is meant to add “OS-like” smooth curvature in the mid-section without breaking endpoint constraints.
- For “angle at mouth” reporting, `cts_mouth_angle_deg` in `bempp_audio/mesh/profiles.py` uses a last-segment finite difference to estimate `dr/dx` at the mouth.
