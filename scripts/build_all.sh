#!/usr/bin/env bash
# =============================================================================
# scripts/build_all.sh — Compile QEMU ARM64 Daemon, Verilator RTL & Bridge
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# In Google Cloud Shell, container restarts reset /usr while preserving $HOME.
# Automatically run setup_cloudshell.sh if Verilator or ARM cross-toolchains are missing.
if [ "$(id -u)" -eq 0 ] || (command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null); then
    if ! command -v verilator >/dev/null 2>&1 || \
       ! command -v aarch64-linux-gnu-g++ >/dev/null 2>&1 || \
       ! command -v arm-linux-gnueabihf-g++ >/dev/null 2>&1 || \
       [ ! -e /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 ]; then
        bash "${REPO_ROOT}/scripts/setup_cloudshell.sh"
    fi
fi

bash "${REPO_ROOT}/emulator/qemu/build.sh"
bash "${REPO_ROOT}/emulator/verilator/build.sh"
echo "[Build All] Emulator 1 (QEMU), Emulator 2 (Verilator), and C++ Co-Sim Bridge ready."
