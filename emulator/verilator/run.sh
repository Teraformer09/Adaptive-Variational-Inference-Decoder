#!/usr/bin/env bash
# =============================================================================
# emulator/verilator/run.sh — Run Verilator Standalone Test or Co-Sim Bridge
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"

MODE="${1:---standalone}"
shift || true

if [ "${MODE}" = "--bridge" ]; then
    exec "${MODEL_DIR}/avid_cosim_bridge" "$@"
else
    exec "${MODEL_DIR}/standalone_rtl_tb" "$@"
fi
