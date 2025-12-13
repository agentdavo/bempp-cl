#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage: ./check_installed.sh [--venv PATH] [--system-python PATH] [--include-system] [--mpi-smoke]

Reports whether key dependencies (DOLFINx/FEniCSx, PETSc, bempp_cl, optional FMM, etc.)
are importable under a local virtualenv (default: ./venv).

Optionally, it can also check the "system" Python if you pass --include-system.

Examples:
  ./check_installed.sh
  ./check_installed.sh --venv ./venv
  ./check_installed.sh --include-system --system-python /usr/bin/python3 --venv ~/dolfinx-env
  ./check_installed.sh --mpi-smoke
  ./check_installed.sh --import-mpi
EOF
}

VENV_PATH="./venv"
SYSTEM_PYTHON=""
INCLUDE_SYSTEM=false
MPI_SMOKE=false
IMPORT_MPI=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="${2:-}"
      shift 2
      ;;
    --system-python)
      SYSTEM_PYTHON="${2:-}"
      shift 2
      ;;
    --include-system)
      INCLUDE_SYSTEM=true
      shift
      ;;
    --mpi-smoke)
      MPI_SMOKE=true
      shift
      ;;
    --import-mpi)
      IMPORT_MPI=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SYSTEM_PYTHON}" ]]; then
  SYSTEM_PYTHON="$(command -v python3 || true)"
fi

VENV_PYTHON=""
if [[ -d "${VENV_PATH}" && -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PYTHON="${VENV_PATH}/bin/python"
elif [[ -d "${VENV_PATH}" && -x "${VENV_PATH}/bin/python3" ]]; then
  VENV_PYTHON="${VENV_PATH}/bin/python3"
fi

COMPLEX_PETSC_DIR="$(ls -d /usr/lib/petscdir/petsc*/x86_64-linux-gnu-complex 2>/dev/null | head -n 1 || true)"

run_python_checks() {
  local py="$1"
  local label="$2"
  echo ""
  echo "========================================"
  echo "${label}"
  echo "Python: ${py}"
  echo "========================================"

  local import_mpi_env="0"
  if [[ "${IMPORT_MPI}" == true ]]; then
    import_mpi_env="1"
  fi

  local auto_petsc_env="0"
  local petsc_dir_env="${PETSC_DIR:-}"
  if [[ -z "${petsc_dir_env}" && -n "${COMPLEX_PETSC_DIR}" ]]; then
    petsc_dir_env="${COMPLEX_PETSC_DIR}"
    auto_petsc_env="1"
  fi

  MPI_SMOKE="${MPI_SMOKE}" \
    BEMPPAUDIO_IMPORT_MPI="${import_mpi_env}" \
    BEMPPAUDIO_COMPLEX_PETSC_DIR="${COMPLEX_PETSC_DIR}" \
    BEMPPAUDIO_AUTO_PETSC_DIR="${auto_petsc_env}" \
    PETSC_DIR="${petsc_dir_env}" \
    "$py" - <<'PY'
from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
from typing import Any


def _version(mod: Any) -> str:
    for attr in ("__version__", "VERSION_TEXT", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
        if isinstance(v, (tuple, list)) and all(isinstance(x, int) for x in v):
            return ".".join(str(x) for x in v)
    return "unknown"


def _try_import(name: str) -> tuple[bool, str, str]:
    try:
        mod = importlib.import_module(name)
        return True, _version(mod), ""
    except Exception as e:
        return False, "", f"{type(e).__name__}: {e}"


def _fmt(ok: bool, version: str, err: str) -> str:
    if ok:
        return f"OK   {version}"
    return f"MISS {err}"


modules = [
    # MPI-sensitive modules: prefer "spec" by default to avoid hard MPI aborts
    # on misconfigured systems. Set BEMPPAUDIO_IMPORT_MPI=1 to actually import.
    ("dolfinx", "DOLFINx (FEniCSx)", "spec"),
    ("mpi4py", "mpi4py", "spec"),
    ("petsc4py", "petsc4py", "spec"),
    ("slepc4py", "slepc4py (optional)", "spec"),
    # Pure-Python packages (safe to import)
    ("ufl", "UFL", "import"),
    ("basix", "Basix", "import"),
    ("ffcx", "FFCx", "import"),
    ("bempp_cl.api", "bempp_cl", "import"),
    ("bempp_cl.api.fmm", "bempp_cl FMM module", "import"),
    ("bempp_cl.api.external.fenicsx", "bempp_cl DOLFINx coupling", "import"),
    ("bempp_audio", "bempp_audio", "import"),
    ("gmsh", "gmsh (Python)", "import"),
    ("plotly", "plotly", "import"),
    ("meshio", "meshio", "import"),
    ("pyopencl", "pyopencl", "import"),
    ("numba", "numba", "import"),
]

print(f"sys.executable: {sys.executable}")
print(f"sys.version:    {sys.version.splitlines()[0]}")
print("")

import_mpi = os.environ.get("BEMPPAUDIO_IMPORT_MPI", "0") == "1"
mpi_smoke = os.environ.get("MPI_SMOKE", "false").lower() in ("1", "true", "yes")
complex_petsc_dir = os.environ.get("BEMPPAUDIO_COMPLEX_PETSC_DIR") or ""
auto_petsc_dir = os.environ.get("BEMPPAUDIO_AUTO_PETSC_DIR", "0") == "1"

if auto_petsc_dir and complex_petsc_dir:
    print(f"NOTE: Auto-set PETSC_DIR={complex_petsc_dir} for this check.")
    print("      To persist, export PETSC_DIR in your shell or venv activation.\n")

def _try_spec(name: str) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(name)
        return (spec is not None), ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _fmt_spec(ok: bool, err: str) -> str:
    return "FOUND (not imported)" if ok else f"MISS {err}"

width = max(len(label) for _, label, _ in modules)
for name, label, mode in modules:
    if mode == "import" or (mode == "spec" and import_mpi):
        ok, ver, err = _try_import(name)
        print(f"{label.ljust(width)}  {_fmt(ok, ver, err)}")
    else:
        ok, err = _try_spec(name)
        print(f"{label.ljust(width)}  {_fmt_spec(ok, err)}")

# ExaFMM Python binding: distribution name is `exafmm-t`, import is usually
# `exafmm` (sometimes `exafmm_t`).
exafmm_label = "exafmm-t (FMM backend)"
width = max(width, len(exafmm_label))
exafmm_version = "unknown"
try:
    import importlib.metadata as _ilm

    try:
        exafmm_version = _ilm.version("exafmm-t")
    except Exception:
        try:
            exafmm_version = _ilm.version("exafmm")
        except Exception:
            exafmm_version = "unknown"
except Exception:
    pass

exafmm_ok = False
errs = []
for modname in ("exafmm", "exafmm_t"):
    ok, ver, err = _try_import(modname)
    if ok:
        exafmm_ok = True
        if ver != "unknown":
            exafmm_version = ver
        break
    if err:
        errs.append(f"{modname}: {err}")

if exafmm_ok:
    print(f"{exafmm_label.ljust(width)}  OK   {exafmm_version}")
else:
    # Fallback: maybe installed but not importable (wrong ABI / build deps).
    print(f"{exafmm_label.ljust(width)}  MISS {('; '.join(errs)) if errs else 'not importable'}")

# PETSc scalar type (if petsc4py available)
ok, _, _ = _try_import("petsc4py") if import_mpi else (False, "", "")
if ok:
    try:
        from petsc4py import PETSc  # type: ignore
        scalar = getattr(PETSc, "ScalarType", None)
        scalar_s = str(scalar).lower()
        note = "complex" if "complex" in scalar_s else "real"
        print(f"\nPETSc ScalarType: {scalar} ({note})")
    except Exception as e:
        print(f"\nPETSc ScalarType: unknown ({e})")
else:
    # Even if we don't import petsc4py in-process, we can still probe ScalarType
    # in a subprocess (safer on MPI-sensitive setups), and try the complex PETSc
    # tree if it exists.
    import subprocess
    import pathlib

    def _probe_scalar(extra_env: dict[str, str] | None = None) -> tuple[int, str, str]:
        cmd = [
            sys.executable,
            "-c",
            "from petsc4py import PETSc; print(PETSc.ScalarType)",
        ]
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=env)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()

    def _probe_scalar_unset_petsc_dir() -> tuple[int, str, str]:
        cmd = [
            sys.executable,
            "-c",
            "from petsc4py import PETSc; print(PETSc.ScalarType)",
        ]
        env = os.environ.copy()
        env.pop("PETSC_DIR", None)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=env)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()

    rc, out, err = _probe_scalar()
    if rc == 0 and out:
        note = "complex" if "complex" in out.lower() else "real"
        print(f"\nPETSc ScalarType (subprocess): {out} ({note})")
    elif rc != 0:
        msg = err.splitlines()[-1] if err else "unknown error"
        print("\nPETSc ScalarType (subprocess): FAILED")
        print(f"  stderr: {msg}")
        print("  NOTE: This often indicates an MPI runtime/init problem (common on WSL2 misconfigs).")

    complex_dir = pathlib.Path("/usr/lib/petscdir").glob("petsc*/x86_64-linux-gnu-complex")
    complex_dir = next(complex_dir, None)
    if complex_dir is not None:
        if auto_petsc_dir:
            rc0, out0, err0 = _probe_scalar_unset_petsc_dir()
            if rc0 == 0 and out0 and out and out0.strip() != out.strip():
                note0 = "complex" if "complex" in out0.lower() else "real"
                print(f"PETSc ScalarType (baseline, PETSC_DIR unset): {out0} ({note0})")

        rc2, out2, err2 = _probe_scalar({"PETSC_DIR": str(complex_dir)})
        if rc2 == 0 and out2:
            note2 = "complex" if "complex" in out2.lower() else "real"
            print(f"PETSc ScalarType w/ PETSC_DIR={complex_dir}: {out2} ({note2})")
            if "complex" in out2.lower() and (rc == 0 and out and "complex" not in out.lower()):
                print("WARNING: Your default petsc4py is using a REAL PETSc.")
                print(f"         Set PETSC_DIR={complex_dir} (or export it in your venv activate) for complex Helmholtz FEM.")
        elif rc2 != 0:
            msg2 = err2.splitlines()[-1] if err2 else "unknown error"
            print(f"PETSc ScalarType w/ PETSC_DIR={complex_dir}: FAILED")
            print(f"  stderr: {msg2}")

# bempp_cl FMM capability (best-effort)
ok, _, _ = _try_import("bempp_cl.api")
if ok:
    try:
        import bempp_cl.api  # type: ignore
        # Probe for any FMM-related module presence.
        candidates = [
            "bempp_cl.api.fmm",
            "bempp_cl.api.assembly.fmm",
            "bempp_cl.fmm",
        ]
        found = []
        for c in candidates:
            ok2, _, _ = _try_import(c)
            if ok2:
                found.append(c)
        if found:
            print(f"\nBEMPP FMM modules: {', '.join(found)}")
        else:
            print("\nBEMPP FMM modules: not found (may still be available via runtime backends)")
    except Exception as e:
        print(f"\nBEMPP FMM probe failed: {e}")

# DOLFINx->bempp_cl coupling availability (kept import-only; should not init MPI).
ok_bempp_fenicsx, _, _ = _try_import("bempp_cl.api.external.fenicsx")
if ok_bempp_fenicsx:
    try:
        from bempp_cl.api.external import fenicsx  # type: ignore
        have_boundary = hasattr(fenicsx, "boundary_grid_from_fenics_mesh")
        have_trace = hasattr(fenicsx, "fenics_to_bempp_trace_data")
        print(f"\nFEM-BEM coupling module: OK (boundary_grid={have_boundary}, trace_data={have_trace})")
        if os.environ.get("BEMPPAUDIO_IMPORT_MPI", "0") == "1":
            print("NOTE: BEMPPAUDIO_IMPORT_MPI=1 set; MPI-sensitive imports were attempted above.")
    except Exception as e:
        print(f"\nFEM-BEM coupling module: FAILED ({type(e).__name__}: {e})")

# Optional environment context
for var in ("PETSC_DIR", "PETSC_ARCH", "SLEPC_DIR", "BEMPP_DEVICE_INTERFACE"):
    if var in os.environ:
        print(f"{var}={os.environ[var]}")

if mpi_smoke:
    print("\nMPI smoke test:")
    # Avoid importing MPI in-process: on some setups an MPI runtime failure can
    # abort the interpreter. Run the init in a subprocess instead.
    import subprocess

    cmd = [
        sys.executable,
        "-c",
        "from mpi4py import MPI; print(f'rank {MPI.COMM_WORLD.rank} size {MPI.COMM_WORLD.size}')",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if p.returncode == 0:
            print(f"  MPI init OK ({p.stdout.strip()})")
        else:
            out = (p.stdout or "").strip()
            err = (p.stderr or "").strip()
            print(f"  MPI init FAILED (exit={p.returncode})")
            if out:
                print(f"    stdout: {out}")
            if err:
                print(f"    stderr: {err.splitlines()[-1]}")
    except subprocess.TimeoutExpired:
        print("  MPI init FAILED (timeout)")
    except Exception as e:
        print(f"  MPI init FAILED ({type(e).__name__}: {e})")

    mpirun = shutil.which("mpirun")
    if mpirun:
        print("\nmpirun smoke test (-n 2):")
        cmd2 = [
            mpirun,
            "-n",
            "2",
            sys.executable,
            "-c",
            "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)",
        ]
        try:
            p2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
            if p2.returncode == 0:
                out = (p2.stdout or "").strip()
                print(f"  mpirun OK (stdout: {out})")
            else:
                out = (p2.stdout or "").strip()
                err = (p2.stderr or "").strip()
                print(f"  mpirun FAILED (exit={p2.returncode})")
                if out:
                    print(f"    stdout: {out}")
                if err:
                    print(f"    stderr: {err.splitlines()[-1]}")
        except subprocess.TimeoutExpired:
            print("  mpirun FAILED (timeout)")
        except Exception as e:
            print(f"  mpirun FAILED ({type(e).__name__}: {e})")
PY
}

run_shell_checks() {
  local label="$1"
  echo ""
  echo "----------------------------------------"
  echo "${label} (shell)"
  echo "----------------------------------------"
  if command -v gmsh >/dev/null 2>&1; then
    echo -n "gmsh: "
    gmsh -version 2>/dev/null || gmsh --version 2>/dev/null || echo "(version check failed)"
  else
    echo "gmsh: (not found on PATH)"
  fi
  if command -v mpirun >/dev/null 2>&1; then
    echo -n "mpirun: "
    mpirun --version 2>/dev/null | head -n 1 || echo "(version check failed)"
  else
    echo "mpirun: (not found on PATH)"
  fi
}

run_shell_checks "System"

if [[ "${INCLUDE_SYSTEM}" == true ]]; then
  if [[ -z "${SYSTEM_PYTHON}" ]]; then
    echo ""
    echo "========================================"
    echo "System environment"
    echo "========================================"
    echo "python3 not found on PATH and --system-python not provided."
  else
    run_python_checks "${SYSTEM_PYTHON}" "System environment"
  fi
fi

if [[ -n "${VENV_PYTHON}" ]]; then
  run_python_checks "${VENV_PYTHON}" "Venv environment (${VENV_PATH})"
else
  echo ""
  echo "========================================"
  echo "Venv environment"
  echo "========================================"
  echo "No Python found under: ${VENV_PATH} (expected ${VENV_PATH}/bin/python)"
fi

echo ""
echo "Done."
