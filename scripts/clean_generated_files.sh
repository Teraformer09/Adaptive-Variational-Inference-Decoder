#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "${REPO_ROOT}/software/build_native" "${REPO_ROOT}/software/build_cortex_a9" "${REPO_ROOT}/software/build_arm64"
rm -rf "${REPO_ROOT}/emulator/bridge/build" "${REPO_ROOT}/emulator/verilator/build"
echo "[Clean] Removed temporary build directories."
