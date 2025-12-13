Summary: DOLFINx + bempp-cl for Waveguide Design

  What Already Exists in bempp-cl

  API modules (bempp_cl/api/external/):
  | File       | Purpose                              |
  |------------|--------------------------------------|
  | fenicsx.py | Modern DOLFINx interface (active)    |
  | fenics.py  | Legacy DOLFIN interface (deprecated) |

  Key functions provided:
  - boundary_grid_from_fenics_mesh() - Extract bempp surface grid from FEM volumetric mesh
  - fenics_to_bempp_trace_data() - Create trace space + DOF mapping matrix
  - FenicsOperator - Wrap UFL weak forms as bempp-compatible operators
  - BlockedDiscreteOperator - Construct coupled FEM-BEM system

  Working example (examples/helmholtz/simple_helmholtz_fem_bem_coupling_dolfinx.py):
  - Helmholtz transmission problem (different k inside/outside)
  - Blocked 2×2 system: [A-k²M, -M_Γ; ½I-K, V]
  - Block-diagonal preconditioning + GMRES solve
  - Interior (FEM) + exterior (BEM representation formula) field evaluation

  ---
  Benefits for Waveguide Design

  | Capability              | Current (BEM only)             | With FEM-BEM Coupling                                  |
  |-------------------------|--------------------------------|--------------------------------------------------------|
  | Interior pressure field | Not available                  | Full 3D pressure distribution inside horn              |
  | Near-field              | Expensive potential evaluation | Efficient FEM evaluation                               |
  | Variable sound speed    | Not supported                  | FEM handles n(x) (temperature gradients, gas mixtures) |
  | Absorbing materials     | Not supported                  | Equivalent fluid models in FEM domain                  |
  | Driver coupling         | Velocity BC only               | Full vibroacoustic: elastic dome → acoustic field      |
  | Visualization           | Surface only                   | Volume rendering, cross-sections, streamlines          |

  Specific waveguide applications:

  1. Interior pressure visualization - Show standing waves, pressure gain, resonances inside horn throat/mouth
  2. Phase plug modeling - FEM for complex phase plug geometry + acoustic coupling
  3. Compression driver integration - Couple elastic dome (shell FEM) → interior air (acoustic FEM) → radiation (BEM)
  4. Throat chamber acoustics - Model complex back-chambers with absorptive treatment
  5. Temperature/humidity effects - Spatially varying sound speed c(x)

  ---
  Integration Path for bempp_audio

  The infrastructure is ready. A natural integration would be:

  # Proposed bempp_audio API extension
  speaker = (
      Loudspeaker()
      .waveguide_from_config(cfg)
      .enable_fem_interior(mesh_size=0.005)  # Generate interior FEM mesh
      .infinite_baffle()
      .frequency_range(200, 10000, num=20)
  )

  response = speaker.solve()

  # New capabilities:
  response.plot_interior_pressure(freq=2000, slice_z=0.0)  # Cross-section
  response.plot_pressure_3d(freq=4000)  # Volume rendering
  response.export_interior_field("pressure_field.vtu")  # ParaView export

  The existing fenicsx.py module handles DOF mapping and operator assembly - the main work would be volumetric mesh generation for the horn interior and result visualization.