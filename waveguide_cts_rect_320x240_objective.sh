#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/waveguide_cts_rect_320x240_objective_${TS}.log}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/venv/bin/python"
    # Add venv/bin to PATH so gmsh and other tools are found
    export PATH="$ROOT_DIR/venv/bin:$PATH"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

echo "======================================================================" | tee "$LOG_FILE"
echo "CTS Rect Waveguide (320x240) Objective Run" | tee -a "$LOG_FILE"
echo "Script: examples/bempp_audio/waveguide_cts_rect_320x240_objective.py" | tee -a "$LOG_FILE"
echo "Python: $PYTHON_BIN" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# Compute/backend
export BEMPP_DEVICE_INTERFACE="${BEMPP_DEVICE_INTERFACE:-numba}"

# Outputs
export BEMPPAUDIO_OUT_DIR="${BEMPPAUDIO_OUT_DIR:-$LOG_DIR}"

# Performance
export BEMPPAUDIO_MESH_PRESET="${BEMPPAUDIO_MESH_PRESET:-super_fast}"
export BEMPPAUDIO_SOLVER_PRESET="${BEMPPAUDIO_SOLVER_PRESET:-$BEMPPAUDIO_MESH_PRESET}"
export BEMPPAUDIO_N_WORKERS="${BEMPPAUDIO_N_WORKERS:-auto}"

# OSRC (recommended)
export BEMPPAUDIO_USE_OSRC="${BEMPPAUDIO_USE_OSRC:-0}"
export BEMPPAUDIO_OSRC_NPADE="${BEMPPAUDIO_OSRC_NPADE:-2}"

# Mesh export (interactive HTML)
export BEMPPAUDIO_EXPORT_MESH="${BEMPPAUDIO_EXPORT_MESH:-1}"

# FRD export (for crossover tools / post-analysis)
export BEMPPAUDIO_EXPORT_FRD="${BEMPPAUDIO_EXPORT_FRD:-1}"
export BEMPPAUDIO_FRD_ANGLES="${BEMPPAUDIO_FRD_ANGLES:-spl}"

# Geometry
export BEMPPAUDIO_WG_LEN_M="${BEMPPAUDIO_WG_LEN_M:-0.100}"

# Objective band + targets
export BEMPPAUDIO_OBJ_F_LO="${BEMPPAUDIO_OBJ_F_LO:-1000}"
export BEMPPAUDIO_OBJ_F_HI="${BEMPPAUDIO_OBJ_F_HI:-16000}"
export BEMPPAUDIO_TARGET_BW_H_DEG="${BEMPPAUDIO_TARGET_BW_H_DEG:-90}"
export BEMPPAUDIO_TARGET_BW_V_DEG="${BEMPPAUDIO_TARGET_BW_V_DEG:-90}"
export BEMPPAUDIO_OBJ_DI_MODE="${BEMPPAUDIO_OBJ_DI_MODE:-proxy}"

# Frequency sampling
export BEMPPAUDIO_NUM_FREQS="${BEMPPAUDIO_NUM_FREQS:-28}"
export BEMPPAUDIO_PROBE_5F="${BEMPPAUDIO_PROBE_5F:-0}"
export BEMPPAUDIO_PROBE_MESH_PRESET="${BEMPPAUDIO_PROBE_MESH_PRESET:-standard}"

# Morph timing
export BEMPPAUDIO_MORPH_FIXED_MM="${BEMPPAUDIO_MORPH_FIXED_MM:-5}"
export BEMPPAUDIO_MORPH_RATE="${BEMPPAUDIO_MORPH_RATE:-3}"

# Discretization overrides (optional)
# export BEMPPAUDIO_N_CIRC=64
# export BEMPPAUDIO_CORNER_RES=4

# Cabinet chamfers (symmetric 45-degree, millimeters)
# Set CHAMFER_ALL_MM for uniform chamfer on all four front-face edges.
# Override individual edges with TOP/BOTTOM/LEFT/RIGHT variants.
export BEMPPAUDIO_CHAMFER_ALL_MM="${BEMPPAUDIO_CHAMFER_ALL_MM:-20}"
# export BEMPPAUDIO_CHAMFER_TOP_MM=25
# export BEMPPAUDIO_CHAMFER_BOTTOM_MM=25
# export BEMPPAUDIO_CHAMFER_LEFT_MM=15
# export BEMPPAUDIO_CHAMFER_RIGHT_MM=15

"$PYTHON_BIN" -u examples/bempp_audio/waveguide_cts_rect_320x240_objective.py 2>&1 | tee -a "$LOG_FILE"
