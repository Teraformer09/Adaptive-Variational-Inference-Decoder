#!/usr/bin/env bash
# =============================================================================
# scripts/setup_cloudshell.sh — Install QEMU (Cortex-A9 & ARM64), Verilator & Toolchain
# =============================================================================
set -euo pipefail

echo "[M0 Setup] Checking Cloud Shell toolchain for AVID V1 Dual-Emulator..."
if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 -o -n "$(command -v sudo || true)" ]; then
    NEED_INSTALL=0
    for bin in verilator qemu-arm qemu-aarch64 arm-linux-gnueabihf-g++ aarch64-linux-gnu-g++; do
        if ! command -v "${bin}" >/dev/null 2>&1; then
            NEED_INSTALL=1
        fi
    done
    if [ "${NEED_INSTALL}" -eq 1 ]; then
        echo "[M0 Setup] Installing verilator, qemu-user, g++-arm-linux-gnueabihf (Cortex-A9), and g++-aarch64-linux-gnu..."
        sudo apt-get update -qq || true
        sudo apt-get install -y -qq \
            verilator \
            qemu-user \
            qemu-user-static \
            gcc-arm-linux-gnueabihf \
            g++-arm-linux-gnueabihf \
            gcc-aarch64-linux-gnu \
            g++-aarch64-linux-gnu \
            cmake \
            python3-yaml || true
    fi
fi

python3 -c "import yaml" 2>/dev/null || pip install -q pyyaml || true

echo "[M0 Setup] Toolchain summary:"
echo "  - Python3      : $(python3 --version 2>&1)"
echo "  - CMake        : $(cmake --version 2>&1 | head -n1)"
echo "  - Host G++     : $(g++ --version 2>&1 | head -n1)"
echo "  - Cortex-A9 G++: $(arm-linux-gnueabihf-g++ --version 2>&1 | head -n1 || echo 'not-installed')"
echo "  - ARM64 G++    : $(aarch64-linux-gnu-g++ --version 2>&1 | head -n1 || echo 'not-installed')"
echo "  - QEMU CortexA9: $(qemu-arm --version 2>&1 | head -n1 || echo 'not-installed')"
echo "  - QEMU ARM64   : $(qemu-aarch64 --version 2>&1 | head -n1 || echo 'not-installed')"
echo "  - Verilator    : $(verilator --version 2>&1 | head -n1 || echo 'cycle-exact-c++-model')"
