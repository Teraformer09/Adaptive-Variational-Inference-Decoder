#!/usr/bin/env bash
# =============================================================================
# scripts/build_all.sh — Compile QEMU ARM64 Daemon, Verilator RTL & Bridge
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${REPO_ROOT}/emulator/qemu/build.sh"
bash "${REPO_ROOT}/emulator/verilator/build.sh"
echo "[Build All] Emulator 1 (QEMU), Emulator 2 (Verilator), and C++ Co-Sim Bridge ready."
