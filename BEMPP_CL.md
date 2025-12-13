  bempp-cl Capabilities for Waveguide/Loudspeaker Design

  Core Operators Available

  | Category            | Operators                                                       | Loudspeaker Application                       |
  |---------------------|-----------------------------------------------------------------|-----------------------------------------------|
  | Helmholtz Boundary  | single_layer, double_layer, adjoint_double_layer, hypersingular | Core acoustic BEM - radiation, scattering     |
  | Helmholtz Potential | single_layer, double_layer                                      | Field evaluation at arbitrary points          |
  | Helmholtz Far-Field | single_layer, double_layer                                      | Directivity patterns, RCS/radiation patterns  |
  | OSRC                | osrc_ntd, osrc_dtn                                              | High-frequency preconditioning (faster GMRES) |
  | Multitrace          | multitrace_operator                                             | Multi-domain coupling (enclosure + horn)      |
  | Maxwell             | electric_field, magnetic_field, multitrace                      | Voice coil EM fields, shielding               |
  | Laplace             | single_layer, double_layer, hypersingular                       | Static/low-frequency problems                 |

  ---
  Key Examples & Their Loudspeaker Applications

  1. Combined Field Integral Equation (helmholtz_combined_exterior.py)

  (½Id + K' - iηV) u_ν = ∂u_inc/∂n - iη u_inc
  Application: Resonance-free exterior radiation - avoids spurious interior resonances that plague simple BEM formulations. Essential for accurate horn/waveguide simulation across all frequencies.

  2. OSRC Burton-Miller (osrc_burton_miller.py)

  ntd = bempp.operators.boundary.helmholtz.osrc_ntd(space, k)
  osrc_bm = 0.5 * identity - dlp - ntd * hyp
  Application: High-frequency waveguide simulation. OSRC preconditioning dramatically reduces GMRES iterations at high frequencies (10kHz+), making large horn meshes tractable.

  3. BEM-BEM Multitrace Coupling (bem_bem_multitrace_coupling.py)

  Ai = bempp.operators.boundary.helmholtz.multitrace_operator(grid, k_interior)
  Ae = bempp.operators.boundary.helmholtz.multitrace_operator(grid, k_exterior)
  op = Ai + Ae  # Self-regularizing!
  Application:
  - Enclosure + port: Different acoustic domains (sealed box interior vs. exterior radiation)
  - Phase plug proximity: Interior dome chamber vs. horn throat
  - Transmission through panels: Enclosure wall vibration coupling

  4. FEM-BEM Coupling (simple_helmholtz_fem_bem_coupling_dolfinx.py)

  Application: Interior FEM (complex geometry, absorbers, variable properties) + exterior BEM (infinite domain radiation). Ideal for:
  - Horn interior pressure visualization
  - Absorptive throat treatment
  - Driver-horn coupled simulation

  5. Maxwell Dielectric Scattering (maxwell_dielectric.py)

  Application: Voice coil electromagnetic analysis, motor structure shielding, eddy current effects in pole pieces.

  ---
  Built-in Shapes

  bempp.api.shapes.sphere(r, origin, h)      # Dome approximation
  bempp.api.shapes.ellipsoid(a, b, c, h)     # Elongated sources
  bempp.api.shapes.cube(h)                    # Box enclosures
  bempp.api.shapes.cuboid(a, b, c, h)        # Rectangular enclosures
  bempp.api.shapes.screen(a, b, c, h)        # Baffle approximation
  bempp.api.shapes.cylinders(...)            # Port tubes

  ---
  Specific Waveguide/Loudspeaker Use Cases

  | Use Case               | bempp-cl Feature                 | Benefit                            |
  |------------------------|----------------------------------|------------------------------------|
  | Far-field directivity  | far_field.helmholtz.double_layer | Compute DI, beamwidth, polar plots |
  | Near-field pressure    | potential.helmholtz.single_layer | Pressure at any point in space     |
  | High-frequency horns   | OSRC preconditioner              | 3-10× fewer iterations at 10kHz+   |
  | Enclosure + horn       | Multitrace coupling              | Single solve for coupled domains   |
  | Resonance-free         | Combined field (Burton-Miller)   | No spurious resonances             |
  | Interior visualization | FEM-BEM coupling                 | Pressure inside horn throat        |
  | Multi-scatterer arrays | GeneralizedBlockedOperator       | Line arrays, multiple drivers      |
  | Absorptive treatment   | Complex wavenumber k + iα        | Lossy boundaries                   |
  | Variable sound speed   | FEM interior                     | Temperature gradients in horn      |

  ---
  Advanced Capabilities Not Yet in bempp_audio

  1. OSRC Preconditioning - Already in bempp-cl, could speed up high-freq sweeps
  2. Multitrace for enclosures - Model box + port + horn as coupled domains
  3. Maxwell for motors - Voice coil field analysis, Bl(x) prediction
  4. Far-field operators - Direct far-field computation (more efficient than potential eval at distance)
  5. Complex wavenumber - Absorptive boundary conditions for damped horns
  6. Multi-body scattering - Line arrays, multiple driver configurations

  ---
  Integration Opportunities for bempp_audio

  # Proposed extensions
  speaker = (
      Loudspeaker()
      .waveguide_from_config(cfg)
      .use_osrc_preconditioner()           # High-frequency speedup
      .enable_interior_fem()               # DOLFINx interior field
      .enclosure_multitrace(box_mesh)      # Coupled box + horn
      .frequency_range(200, 20000, num=50)
  )

  response = speaker.solve()

  # New outputs
  response.far_field_pattern(freq=4000)    # Direct far-field computation
  response.interior_pressure(freq=2000)    # FEM interior field
  response.directivity_3d(freq=8000)       # 3D balloon from far-field ops

  The library is indeed powerful - bempp_audio currently uses only a fraction of what's available. The OSRC preconditioner alone could provide significant speedup for high-frequency horn simulation.