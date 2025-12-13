# Bempp-CL Benchmarks

**Note:** Benchmarks have been updated to use `bempp_cl.api` (December 2025).

## Running Benchmarks

```bash
# Run all benchmarks
pytest benchmarks/ --benchmark-only

# Run with specific device
pytest benchmarks/ --benchmark-only --device=opencl
pytest benchmarks/ --benchmark-only --device=numba

# Run with specific precision
pytest benchmarks/ --benchmark-only --precision=double
pytest benchmarks/ --benchmark-only --precision=single

# Run specific benchmark
pytest benchmarks/benchmark_dense_assembly.py::laplace_single_layer_dense_benchmark --benchmark-only
```

## Available Options

- `--device`: `numba` | `opencl` | `auto` (default: auto)
- `--precision`: `single` | `double` (default: double)  
- `--vec`: `auto` | `novec` | `vec4` | `vec8` | `vec16` (default: auto)

## Benchmarks

### Dense Assembly (`benchmark_dense_assembly.py`)
- `laplace_single_layer_dense_benchmark` - Sphere refinement 4, DP0
- `laplace_single_layer_dense_large_benchmark` - Sphere refinement 5, DP0
- `laplace_single_layer_dense_p1_disc_benchmark` - Sphere refinement 4, DP1
- `laplace_single_layer_dense_p1_cont_benchmark` - Sphere refinement 4, P1
- `helmholtz_single_layer_dense_p1_cont_large_benchmark` - Helmholtz, sphere 5, P1
- `maxwell_electric_field_dense_large_benchmark` - Maxwell E-field, sphere 5
- `maxwell_magnetic_field_dense_large_benchmark` - Maxwell H-field, sphere 5

### Sparse Assembly (`benchmark_sparse_assembly.py`)
- Sparse operator benchmarks

### Potential Operators (`benchmark_dense_potential.py`)
- Potential evaluation benchmarks

## Changes from Original

The benchmarks were updated to use the new `bempp_cl.api` import path instead of the deprecated `bempp.api`. All functionality remains the same.
