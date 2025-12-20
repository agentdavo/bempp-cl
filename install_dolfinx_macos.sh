#!/usr/bin/env bash
#
# DOLFINx + Bempp-cl Installation Script for macOS (Homebrew)
#
# Primary target: macOS 13+ (Ventura) with Homebrew installed.
#
# Usage:
#   ./install_dolfinx_macos.sh
#   ./install_dolfinx_macos.sh --venv ./venv --skip-brew
#
# Notes:
# - This script uses Homebrew for system dependencies.
# - DOLFINx is installed via pip (fenics-dolfinx) since Homebrew doesn't have it.
# - PETSc with complex scalar support is built from source or installed via pip.
#

set -euo pipefail
IFS=$'\n\t'

GMSH_SDK_VERSION="${GMSH_SDK_VERSION:-4.11.1}"
# ExaFMM is required for bempp_cl fast evaluation workflows.
EXAFMM_GIT_URL="https://github.com/exafmm/exafmm-t.git"

# Detect architecture
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
  BREW_PREFIX="/opt/homebrew"
  GMSH_ARCH="MacOSARM-sdk"
else
  BREW_PREFIX="/usr/local"
  GMSH_ARCH="MacOSX-sdk"
fi

VENV_PATH=""
SKIP_BREW=false
BREW_UPGRADE=false
INSTALL_GMSH_SDK=false
RUN_SMOKE_TESTS=false
RUN_FEM_BEM_EXAMPLE=false

LOG_FILE="install_dolfinx_macos.log"
NO_LOG=false

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --venv PATH             Install Python deps into venv at PATH (created if missing)
  --skip-brew             Skip Homebrew installs (for already installed)
  --brew-upgrade          Run brew upgrade (default: off)
  --install-gmsh-sdk      Install Gmsh SDK under /usr/local (default: off)
  --run-smoke-tests       Run quick bempp_cl + dolfinx smoke test
  --run-fem-bem-example   Run examples/helmholtz FEM-BEM example (if present)
  --log PATH              Write output to PATH (default: ${LOG_FILE})
  --no-log                Disable log file output
  --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV_PATH="${2:-}"; shift 2 ;;
    --skip-brew) SKIP_BREW=true; shift ;;
    --brew-upgrade) BREW_UPGRADE=true; shift ;;
    --install-gmsh-sdk) INSTALL_GMSH_SDK=true; shift ;;
    --run-smoke-tests) RUN_SMOKE_TESTS=true; shift ;;
    --run-fem-bem-example) RUN_FEM_BEM_EXAMPLE=true; shift ;;
    --log) LOG_FILE="${2:-}"; shift 2 ;;
    --no-log) NO_LOG=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "Error: $*" >&2; exit 1; }

if [[ "$NO_LOG" == false ]]; then
  if [[ -z "${BEMPPAUDIO_INSTALL_LOGGING:-}" ]]; then
    export BEMPPAUDIO_INSTALL_LOGGING=1
    : >"$LOG_FILE" 2>/dev/null || true
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Logging to: $LOG_FILE"
  fi
fi

echo "========================================"
echo "DOLFINx + Bempp-cl Installer (macOS/Homebrew)"
echo "========================================"
echo ""
echo "Detected architecture: $ARCH"
echo "Homebrew prefix: $BREW_PREFIX"
echo ""

# Check for Homebrew
if ! command -v brew &>/dev/null; then
  die "Homebrew not found. Install it from https://brew.sh"
fi

if [[ "$SKIP_BREW" == false ]]; then
  echo "[1/5] Installing system dependencies (Homebrew)..."

  if [[ "$BREW_UPGRADE" == true ]]; then
    brew update
    brew upgrade
  fi

  # Core tools
  brew install --quiet \
    cmake \
    ninja \
    pkg-config \
    autoconf \
    automake \
    libtool \
    gcc \
    git \
    curl

  # Python (use Homebrew Python for consistency)
  brew install --quiet python@3.12 || brew install --quiet python@3.11 || brew install --quiet python

  # MPI + HDF5 (DOLFINx requires MPI builds)
  brew install --quiet \
    open-mpi \
    hdf5-mpi

  # Graph partitioners (DOLFINx dependencies)
  brew install --quiet \
    scotch \
    metis \
    parmetis || echo "Note: parmetis may not be available; SCOTCH should be sufficient."

  # PETSc (Homebrew version is real-scalar; we'll build complex via pip)
  # Install anyway for headers/libraries that might be useful
  brew install --quiet petsc || echo "Note: petsc formula may not exist; will use pip."

  # Boost (required by DOLFINx)
  brew install --quiet boost

  # BLAS/LAPACK (use Accelerate framework on macOS, but openblas for ExaFMM)
  brew install --quiet openblas

  # Gmsh
  brew install --quiet gmsh

  # pybind11 for building Python bindings
  brew install --quiet pybind11

  # NOTE: Do NOT install llvm from Homebrew for numba/llvmlite.
  # They ship pre-built wheels with bundled LLVM that work better.

  # OpenCL is built into macOS (via Metal/OpenCL framework), no separate install needed
  echo "   Note: OpenCL is provided by macOS system frameworks."

  echo "   Homebrew dependencies installed."
else
  echo "[1/5] Skipping Homebrew installation (--skip-brew)"
fi

# NOTE: Do NOT set LLVM_CONFIG here. numba/llvmlite ship pre-built wheels
# with bundled LLVM. Setting LLVM_CONFIG causes pip to build from source
# against Homebrew's LLVM, which often has version mismatches.

echo ""
echo "[2/5] Setting up Python environment..."

# Find Python
PYTHON_CMD=""
for py in python3.12 python3.11 python3; do
  if command -v "$py" &>/dev/null; then
    PYTHON_CMD="$py"
    break
  fi
done
[[ -n "$PYTHON_CMD" ]] || die "Python 3 not found"
echo "   Using system Python: $PYTHON_CMD"

if [[ -n "$VENV_PATH" ]]; then
  if [[ ! -d "$VENV_PATH" ]]; then
    echo "   Creating venv at $VENV_PATH"
    "$PYTHON_CMD" -m venv "$VENV_PATH"
  fi
else
  VENV_PATH="$HOME/dolfinx-env"
  if [[ ! -d "$VENV_PATH" ]]; then
    echo "   Creating venv at $VENV_PATH"
    "$PYTHON_CMD" -m venv "$VENV_PATH"
  fi
fi

# Convert to absolute path
VENV_PATH="$(cd "$VENV_PATH" && pwd)"
echo "   Using venv: $VENV_PATH"

VENV_PYTHON=""
if [[ -x "$VENV_PATH/bin/python3" ]]; then
  VENV_PYTHON="$VENV_PATH/bin/python3"
elif [[ -x "$VENV_PATH/bin/python" ]]; then
  VENV_PYTHON="$VENV_PATH/bin/python"
else
  die "Venv python not found under: $VENV_PATH"
fi

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo ""
echo "[3/5] Installing Python packages (venv)..."

# Set environment for building packages that need to find Homebrew libs
# Append to existing LDFLAGS/CPPFLAGS (may already have LLVM paths)
export LDFLAGS="${LDFLAGS:-} -L${BREW_PREFIX}/lib"
export CPPFLAGS="${CPPFLAGS:-} -I${BREW_PREFIX}/include"
export PKG_CONFIG_PATH="${BREW_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# OpenBLAS for ExaFMM
export OPENBLAS="${BREW_PREFIX}/opt/openblas"
export LDFLAGS="${LDFLAGS} -L${OPENBLAS}/lib"
export CPPFLAGS="${CPPFLAGS} -I${OPENBLAS}/include"

# Unset LLVM_CONFIG to ensure numba/llvmlite use pre-built wheels
unset LLVM_CONFIG 2>/dev/null || true

# Core scientific Python packages
# Pin numba/llvmlite to versions with pre-built wheels for older macOS
"$VENV_PYTHON" -m pip install --no-cache-dir \
  "numpy<1.27" \
  scipy \
  matplotlib \
  plotly \
  "llvmlite<0.43" \
  "numba<0.60" \
  meshio \
  mpi4py \
  "scikit-build-core[pyproject]>=0.10" \
  nanobind \
  pybind11 \
  cython

# PyOpenCL for bempp-cl
"$VENV_PYTHON" -m pip install --no-cache-dir pyopencl

# Gmsh Python bindings
"$VENV_PYTHON" -m pip install --no-cache-dir gmsh

# FEniCSx stack (complex scalar build)
# Install PETSc with complex scalar support first
echo "   Installing PETSc with complex scalar support..."
"$VENV_PYTHON" -m pip install --no-cache-dir petsc petsc4py

# Check if we got complex PETSc; if not, try building from source
"$VENV_PYTHON" - <<'PY' || NEED_COMPLEX_PETSC=true
from petsc4py import PETSc
st = str(PETSc.ScalarType)
if "complex" not in st.lower():
    raise SystemExit("PETSc is real-valued, need complex")
print(f"   PETSc ScalarType: {st}")
PY

if [[ "${NEED_COMPLEX_PETSC:-false}" == true ]]; then
  echo "   Note: pip petsc4py is real-valued. Building complex PETSc..."
  echo "   This may take a while..."

  # Build PETSc with complex scalar support
  PETSC_CONFIGURE_OPTIONS="--with-scalar-type=complex --with-fortran-bindings=0" \
    "$VENV_PYTHON" -m pip install --no-cache-dir --no-binary petsc,petsc4py petsc petsc4py \
    || echo "   Warning: Complex PETSc build failed. Some FEM features may not work."
fi

# FEniCSx packages
echo "   Installing FEniCSx (basix, ufl, ffcx, dolfinx)..."
"$VENV_PYTHON" -m pip install --no-cache-dir \
  fenics-basix \
  fenics-ufl \
  fenics-ffcx

# DOLFINx requires building from source on macOS for best results
# Try the pip package first
"$VENV_PYTHON" -m pip install --no-cache-dir fenics-dolfinx \
  || echo "   Note: fenics-dolfinx pip install failed; may need manual build."

# Install this repo in editable mode if we're in the bempp-cl workspace
if [[ -f "pyproject.toml" ]]; then
  "$VENV_PYTHON" -m pip install -e .
fi

echo "   Python packages installed."

if [[ "$INSTALL_GMSH_SDK" == true ]]; then
  echo ""
  echo "[4/5] Installing Gmsh SDK..."
  GMSH_DIR="/usr/local/gmsh-${GMSH_SDK_VERSION}-${GMSH_ARCH}"
  if [[ ! -d "$GMSH_DIR" ]]; then
    cd /tmp
    curl -fsSLo "gmsh-${GMSH_SDK_VERSION}-${GMSH_ARCH}.tgz" \
      "https://gmsh.info/bin/macOS/gmsh-${GMSH_SDK_VERSION}-${GMSH_ARCH}.tgz"
    sudo tar -xf "gmsh-${GMSH_SDK_VERSION}-${GMSH_ARCH}.tgz" -C /usr/local/
    rm -f "gmsh-${GMSH_SDK_VERSION}-${GMSH_ARCH}.tgz"
  fi
  export PATH="${GMSH_DIR}/bin:$PATH"
  export PYTHONPATH="${GMSH_DIR}/lib:${PYTHONPATH:-}"
  echo "   Gmsh SDK ${GMSH_SDK_VERSION} installed."
else
  echo ""
  echo "[4/5] Skipping Gmsh SDK install (use --install-gmsh-sdk)"
fi

echo ""
echo "[5/5] Installing required extras..."

echo "Installing ExaFMM (required)..."
# ExaFMM needs OpenBLAS on macOS
LDFLAGS="-L${OPENBLAS}/lib" \
CPPFLAGS="-I${OPENBLAS}/include" \
"$VENV_PYTHON" -m pip install --no-cache-dir "git+${EXAFMM_GIT_URL}" \
  || die "ExaFMM install failed. Ensure network access to GitHub and OpenBLAS is installed."

"$VENV_PYTHON" - <<'PY'
import importlib

for name in ("exafmm", "exafmm_t"):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "unknown")
        print(f"   ExaFMM import OK: {name} (version={v})")
        break
    except Exception:
        pass
else:
    raise SystemExit("ExaFMM installed but not importable as exafmm/exafmm_t")
PY

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Activate the environment:"
echo "  source ${VENV_PATH}/bin/activate"
echo ""
echo "Quick checks:"
echo "  ./check_installed.sh"
echo ""

echo "Running post-install checks..."

"$VENV_PYTHON" - <<'PY'
import sys

def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    raise SystemExit(2)

# Check core imports
modules_to_check = [
    ("mpi4py", True),
    ("gmsh", True),
    ("plotly", True),
    ("bempp_cl.api", True),
    ("bempp_cl.api.fmm", True),
    ("exafmm", True),
    ("dolfinx", False),  # Optional - may fail on some macOS setups
]

for mod, required in modules_to_check:
    try:
        __import__(mod)
        print(f"  OK import: {mod}")
    except Exception as e:
        if required:
            fail(f"import {mod!r} failed: {e}")
        else:
            print(f"  SKIP import: {mod} (optional, error: {e})")

# Check PETSc scalar type
try:
    from petsc4py import PETSc
    st = str(PETSc.ScalarType)
    print(f"  PETSc ScalarType: {st}")
    if "complex" not in st.lower():
        print("  WARNING: PETSc ScalarType is real; complex Helmholtz FEM may not work.")
        print("           Consider rebuilding with: PETSC_CONFIGURE_OPTIONS='--with-scalar-type=complex'")
except Exception as e:
    print(f"  NOTE: PETSc check failed: {e}")
PY

if [[ "$RUN_SMOKE_TESTS" == true ]]; then
  echo ""
  echo "Running bempp_cl + DOLFINx smoke tests..."
  "$VENV_PYTHON" - <<'PY'
import sys

def fail(msg: str) -> None:
    print(f"SMOKE TEST FAILED: {msg}", file=sys.stderr)
    raise SystemExit(3)

try:
    from mpi4py import MPI
    from dolfinx.mesh import create_unit_cube
except Exception as e:
    fail(f"dolfinx/mpi4py not importable: {e}")

try:
    from bempp_cl.api.external.fenicsx import boundary_grid_from_fenics_mesh
except Exception as e:
    fail(f"bempp_cl fenicsx coupling not importable: {e}")

m = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
grid, node_map = boundary_grid_from_fenics_mesh(m)
if grid.number_of_elements == 0:
    fail("bempp boundary grid has zero elements")
if len(node_map) == 0:
    fail("node map is empty")
print(f"  OK: boundary_grid_from_fenics_mesh (elements={grid.number_of_elements}, nodes={len(node_map)})")
PY
fi

if [[ "$RUN_FEM_BEM_EXAMPLE" == true ]]; then
  echo ""
  echo "Running FEM-BEM coupling example (dolfinx + bempp_cl)..."
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  EXAMPLE_PATH="${SCRIPT_DIR}/examples/helmholtz/simple_helmholtz_fem_bem_coupling_dolfinx.py"
  if [[ ! -f "$EXAMPLE_PATH" ]]; then
    echo "NOTE: Example not found at: $EXAMPLE_PATH"
  else
    tmpdir="$(mktemp -d)"
    echo "  Using temp dir: $tmpdir"
    (
      cd "$tmpdir"
      export MPLBACKEND=Agg
      "$VENV_PYTHON" "$EXAMPLE_PATH"
    )
    echo "  Example completed."
    echo "  Output (if generated): $tmpdir/example-simple_helmholtz_fem_bem_coupling_dolfinx.png"
  fi
fi
