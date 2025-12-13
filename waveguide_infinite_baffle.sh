#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/waveguide_infinite_baffle_${TS}.log}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

echo "======================================================================" | tee "$LOG_FILE"
echo "Waveguide Infinite Baffle Run" | tee -a "$LOG_FILE"
echo "Script: examples/bempp_audio/waveguide_infinite_baffle.py" | tee -a "$LOG_FILE"
echo "Python: $PYTHON_BIN" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# Favor the numba backend unless overridden.
export BEMPP_DEVICE_INTERFACE="${BEMPP_DEVICE_INTERFACE:-numba}"
export BEMPPAUDIO_MESH_PRESET="${BEMPPAUDIO_MESH_PRESET:-ultra-fast}"
export BEMPPAUDIO_SOLVER_PRESET="${BEMPPAUDIO_SOLVER_PRESET:-$BEMPPAUDIO_MESH_PRESET}"
export BEMPPAUDIO_N_WORKERS="${BEMPPAUDIO_N_WORKERS:-auto}"
export BEMPPAUDIO_OUT_DIR="${BEMPPAUDIO_OUT_DIR:-$LOG_DIR}"

"$PYTHON_BIN" -u examples/bempp_audio/waveguide_infinite_baffle.py 2>&1 | tee -a "$LOG_FILE"
