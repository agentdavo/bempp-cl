#!/usr/bin/env bash
#
# DOLFINx + Bempp-cl Installation Script for Debian/Ubuntu (apt-first)
#
# Primary target: Ubuntu 24.04 with `python3-dolfinx-complex` available.
#
# Usage:
#   sudo ./install_dolfinx_debian.sh
#   ./install_dolfinx_debian.sh --venv ./venv --skip-apt
#
# Notes:
# - This script prefers distro packages for DOLFINx/PETSc to avoid version mismatches.
# - When creating a venv, it defaults to `--system-site-packages` so the venv can see
#   apt-installed dolfinx/petsc4py.
#

set -euo pipefail
IFS=$'\n\t'

GMSH_SDK_VERSION="${GMSH_SDK_VERSION:-4.11.1}"
# ExaFMM is required for bempp_cl fast evaluation workflows.
# Always installed from GitHub via pip:
#   pip install git+https://github.com/exafmm/exafmm-t.git
EXAFMM_GIT_URL="https://github.com/exafmm/exafmm-t.git"

orig_user() {
  if [[ "${SUDO_USER:-}" != "" && "${SUDO_USER}" != "root" ]]; then
    echo "${SUDO_USER}"
  else
    id -un
  fi
}

orig_home() {
  local u
  u="$(orig_user)"
  getent passwd "$u" | cut -d: -f6
}

as_user() {
  # Run a command as the invoking (non-root) user when possible.
  if [[ "${EUID}" -eq 0 && "${SUDO_USER:-}" != "" && "${SUDO_USER}" != "root" ]]; then
    sudo -u "${SUDO_USER}" -H "$@"
  else
    "$@"
  fi
}

detect_petsc_complex_dir() {
  # Prefer Debian/Ubuntu complex PETSc tree when present.
  local d
  d="$(ls -d /usr/lib/petscdir/petsc*/x86_64-linux-gnu-complex 2>/dev/null | head -n 1 || true)"
  if [[ -n "$d" ]]; then
    echo "$d"
    return 0
  fi
  return 1
}

VENV_PATH=""
SKIP_APT=false
APT_UPGRADE=false
INSTALL_GMSH_SDK=false
RUN_SMOKE_TESTS=false
RUN_FEM_BEM_EXAMPLE=false

LOG_FILE="install_dolfinx_debian.log"
NO_LOG=false

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --venv PATH             Install Python deps into venv at PATH (created if missing)
  --skip-apt              Skip apt installs (for non-root / already installed)
  --apt-upgrade           Run apt upgrade (default: off)
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
    --skip-apt) SKIP_APT=true; shift ;;
    --apt-upgrade) APT_UPGRADE=true; shift ;;
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
echo "DOLFINx + Bempp-cl Installer (apt-first)"
echo "========================================"
echo ""

SUDO=""
if [[ "$SKIP_APT" == false ]]; then
  if [[ "$EUID" -ne 0 ]]; then
    SUDO="sudo"
  fi
fi

if [[ "$SKIP_APT" == false ]]; then
  echo "[1/5] Installing system dependencies (apt)..."

  export DEBIAN_FRONTEND=noninteractive

  $SUDO apt-get -qq update
  if [[ "$APT_UPGRADE" == true ]]; then
    $SUDO apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade
  fi

  # Core tools + venv support
  $SUDO apt-get -y install --no-install-recommends \
    ca-certificates \
    curl \
    git \
    pkg-config \
    build-essential \
    cmake \
    ninja-build \
    gfortran \
    autoconf \
    automake \
    libtool \
    python3-dev \
    python3-venv \
    python3-pip

  # MPI + HDF5 (DOLFINx uses MPI builds; examples use mpi4py)
  $SUDO apt-get -y install --no-install-recommends \
    libopenmpi-dev \
    openmpi-bin \
    libhdf5-mpi-dev \
    python3-mpi4py

  # Graph partitioners required by DOLFINx
  $SUDO apt-get -y install --no-install-recommends \
    libscotch-dev \
    libptscotch-dev \
    libparmetis-dev \
    libmetis-dev || echo "Note: ParMETIS/METIS not available via apt (SCOTCH should be sufficient)."

  # DOLFINx (complex PETSc)
  PETSC4PY_PKG="python3-petsc4py-64-complex"
  SLEPC4PY_PKG="python3-slepc4py-64-complex"

  $SUDO apt-get -y install --no-install-recommends \
    python3-dolfinx-complex \
    "${PETSC4PY_PKG}"

  # SLEPc bindings are optional but recommended for eigenproblems.
  if ! $SUDO apt-get -y install --no-install-recommends "${SLEPC4PY_PKG}"; then
    echo "Note: ${SLEPC4PY_PKG} not available (optional)."
    if ! $SUDO apt-get -y install --no-install-recommends python3-slepc4py-complex; then
      $SUDO apt-get -y install --no-install-recommends python3-slepc4py \
        || echo "Note: python3-slepc4py not available (optional)."
    fi
  fi

  # Gmsh (system)
  $SUDO apt-get -y install --no-install-recommends \
    gmsh \
    python3-gmsh || echo "Note: python3-gmsh not available; will rely on pip gmsh."

  # OpenCL for bempp-cl
  $SUDO apt-get -y install --no-install-recommends \
    libpocl-dev \
    ocl-icd-opencl-dev

  # ExaFMM build deps (Python binding uses pybind11 and links to BLAS)
  $SUDO apt-get -y install --no-install-recommends \
    libopenblas-dev || echo "Note: libopenblas-dev not available; ExaFMM build may fail."

  $SUDO apt-get clean
  echo "   apt dependencies installed."
else
  echo "[1/5] Skipping apt installation (--skip-apt)"
fi

echo ""
echo "[2/5] Setting up Python environment..."

if [[ -n "$VENV_PATH" ]]; then
  if [[ ! -d "$VENV_PATH" ]]; then
    echo "   Creating venv at $VENV_PATH (with system-site-packages)"
    as_user python3 -m venv --system-site-packages "$VENV_PATH"
  fi
else
  VENV_PATH="$(orig_home)/dolfinx-env"
  if [[ ! -d "$VENV_PATH" ]]; then
    echo "   Creating venv at $VENV_PATH (with system-site-packages)"
    as_user python3 -m venv --system-site-packages "$VENV_PATH"
  fi
fi

# Convert to absolute path (needed when cd'ing elsewhere, e.g., for FEM-BEM example)
VENV_PATH="$(cd "$VENV_PATH" && pwd)"
echo "   Using venv: $VENV_PATH"

# Add WSL2 workaround for Numba parallel/semaphore issues
if grep -qi microsoft /proc/version 2>/dev/null; then
  if ! grep -q "NUMBA_NUM_THREADS" "$VENV_PATH/bin/activate" 2>/dev/null; then
    echo "   Adding WSL2 Numba workaround to activate script..."
    cat >> "$VENV_PATH/bin/activate" << 'WSLFIX'

# Workaround for WSL2 /dev/shm semaphore issues with Numba parallel
# See: https://github.com/numba/numba/issues/4315
if grep -qi microsoft /proc/version 2>/dev/null; then
    export NUMBA_NUM_THREADS=1
fi
WSLFIX
  fi
fi

VENV_PYTHON=""
if [[ -x "$VENV_PATH/bin/python3" ]]; then
  VENV_PYTHON="$VENV_PATH/bin/python3"
elif [[ -x "$VENV_PATH/bin/python" ]]; then
  VENV_PYTHON="$VENV_PATH/bin/python"
else
  die "Venv python not found under: $VENV_PATH"
fi

as_user "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo ""
echo "[3/5] Installing Python packages (venv)..."

# Clean up any conflicting FEniCS dev packages that might shadow system apt packages.
# The venv uses --system-site-packages so we want the apt-installed versions.
echo "   Cleaning up any conflicting FEniCS packages from venv..."
for pkg in dolfinx basix ffcx ufl; do
  rm -rf "$VENV_PATH/lib/python"*"/site-packages/${pkg}" \
         "$VENV_PATH/lib/python"*"/site-packages/${pkg}-"* \
         "$VENV_PATH/lib/python"*"/site-packages/fenics_${pkg}"* \
         "$VENV_PATH/lib/python"*"/site-packages/fenics-${pkg}"* 2>/dev/null || true
done

# Keep Python-side tooling in the venv even if system-site-packages is enabled.
as_user "$VENV_PYTHON" -m pip install --no-cache-dir \
  numpy \
  scipy \
  matplotlib \
  plotly \
  numba \
  meshio \
  pyopencl \
  "scikit-build-core[pyproject]>=0.10" \
  nanobind \
  pybind11

# If gmsh Python module is missing from apt, ensure it's available for bempp_audio meshing.
as_user "$VENV_PYTHON" - <<'PY' || as_user "$VENV_PYTHON" -m pip install --no-cache-dir gmsh
import gmsh  # noqa: F401
PY

# Install this repo in editable mode if we're in the bempp-cl workspace.
if [[ -f "pyproject.toml" ]]; then
  as_user "$VENV_PYTHON" -m pip install -e .
fi

echo "   Python packages installed."

if [[ "$INSTALL_GMSH_SDK" == true ]]; then
  echo ""
  echo "[4/5] Installing Gmsh SDK..."
  GMSH_DIR="/usr/local/gmsh-${GMSH_SDK_VERSION}-Linux64-sdk"
  if [[ ! -d "$GMSH_DIR" ]]; then
    cd /tmp
    curl -fsSLo "gmsh-${GMSH_SDK_VERSION}-Linux64-sdk.tgz" \
      "https://gmsh.info/bin/Linux/gmsh-${GMSH_SDK_VERSION}-Linux64-sdk.tgz"
    if [[ "$EUID" -ne 0 ]]; then
      sudo tar -xf "gmsh-${GMSH_SDK_VERSION}-Linux64-sdk.tgz" -C /usr/local/
    else
      tar -xf "gmsh-${GMSH_SDK_VERSION}-Linux64-sdk.tgz" -C /usr/local/
    fi
    rm -f "gmsh-${GMSH_SDK_VERSION}-Linux64-sdk.tgz"
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
as_user "$VENV_PYTHON" -m pip install --no-cache-dir "git+${EXAFMM_GIT_URL}" \
  || die "ExaFMM install failed. Ensure network access to GitHub."

as_user "$VENV_PYTHON" - <<'PY'
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
PETSC_COMPLEX_DIR="$(detect_petsc_complex_dir || true)"
if [[ -z "${PETSC_DIR:-}" && -n "${PETSC_COMPLEX_DIR}" ]]; then
  export PETSC_DIR="${PETSC_COMPLEX_DIR}"
fi

as_user "$VENV_PYTHON" - <<'PY'
import sys
import os
import subprocess
import pathlib

def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    raise SystemExit(2)

for mod in ("dolfinx", "mpi4py", "gmsh", "plotly", "bempp_cl.api", "bempp_cl.api.fmm", "exafmm"):
    try:
        __import__(mod)
        print(f"  OK import: {mod}")
    except Exception as e:
        fail(f"import {mod!r} failed: {e}")

def probe_scalar(extra_env: dict[str, str] | None = None) -> tuple[int, str, str]:
    cmd = [sys.executable, "-c", "from petsc4py import PETSc; print(PETSc.ScalarType)"]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()

rc, out, err = probe_scalar()
if rc == 0 and out:
    print(f"  PETSc ScalarType: {out}")
    if "complex" not in out.lower():
        print("  WARNING: PETSc ScalarType does not look complex; complex Helmholtz FEM may not work.")
else:
    msg = err.splitlines()[-1] if err else "unknown error"
    print("  NOTE: PETSc ScalarType probe failed (likely MPI init issue).")
    print(f"        stderr: {msg}")

complex_dir = next(pathlib.Path('/usr/lib/petscdir').glob('petsc*/x86_64-linux-gnu-complex'), None)
if complex_dir is not None:
    rc2, out2, err2 = probe_scalar({"PETSC_DIR": str(complex_dir)})
    if rc2 == 0 and out2:
        print(f"  PETSc ScalarType (PETSC_DIR={complex_dir}): {out2}")
        if "complex" in out2.lower() and (rc == 0 and out and "complex" not in out.lower()):
            print(f"  NOTE: export PETSC_DIR={complex_dir} to use complex PETSc with petsc4py.")
PY

if [[ "$RUN_SMOKE_TESTS" == true ]]; then
  echo ""
  echo "Running bempp_cl + DOLFINx smoke tests..."
  as_user "$VENV_PYTHON" - <<'PY'
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
      as_user "$VENV_PYTHON" "$EXAMPLE_PATH"
    )
    echo "  Example completed."
    echo "  Output (if generated): $tmpdir/example-simple_helmholtz_fem_bem_coupling_dolfinx.png"
  fi
fi
