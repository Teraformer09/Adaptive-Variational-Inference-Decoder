#!/usr/bin/env bash
# =============================================================================
# scripts/run_all.sh — Execute Configured AVID V1 Dual-Emulator Experiment
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CFG_PATH="${1:-${REPO_ROOT}/configs/system/v1_smoke.yaml}"
if [ ! -f "${CFG_PATH}" ] && [ -f "${REPO_ROOT}/configs/system/$(basename "${CFG_PATH}")" ]; then CFG_PATH="${REPO_ROOT}/configs/system/$(basename "${CFG_PATH}")"; fi
if [ ! -f "${CFG_PATH}" ] && [ -f "${REPO_ROOT}/configs/rtl/$(basename "${CFG_PATH}")" ]; then CFG_PATH="${REPO_ROOT}/configs/rtl/$(basename "${CFG_PATH}")"; fi

PYTHONPATH="${REPO_ROOT}" python3 "${REPO_ROOT}/scripts/collect_artifacts.py" "${CFG_PATH}"
