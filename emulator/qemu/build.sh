#!/usr/bin/env bash
# =============================================================================
# emulator/qemu/build.sh — Build AVID ARM Cortex-A9 / ARM64 Guest Binaries & Host Twin
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SW_DIR="${REPO_ROOT}/software"
IMG_DIR="${SCRIPT_DIR}/images"
mkdir -p "${IMG_DIR}" "${SCRIPT_DIR}/configs" "${SCRIPT_DIR}/scripts"

# Always remove old binaries first so stale cross-binaries never persist across container resets
rm -f "${IMG_DIR}/avid_daemon_cortex_a9" "${IMG_DIR}/avid_daemon_aarch64" "${IMG_DIR}/avid_daemon_x86_64"

echo "[QEMU Build] Building native AVID daemon & test suite..."
mkdir -p "${SW_DIR}/build_native"
cmake -S "${SW_DIR}" -B "${SW_DIR}/build_native" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${SW_DIR}/build_native" -j2 >/dev/null
cp -f "${SW_DIR}/build_native/avid_daemon" "${IMG_DIR}/avid_daemon_x86_64"

if command -v arm-linux-gnueabihf-g++ >/dev/null 2>&1 && command -v arm-linux-gnueabihf-gcc >/dev/null 2>&1; then
    echo "[QEMU Build] Cross-compiling AVID daemon for Zynq-7020 ARM Cortex-A9 (arm-linux-gnueabihf)..."
    mkdir -p "${SW_DIR}/build_cortex_a9"
    cmake -S "${SW_DIR}" -B "${SW_DIR}/build_cortex_a9" \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=armv7l \
        -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
        -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
        -DCMAKE_C_FLAGS="-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard" \
        -DCMAKE_CXX_FLAGS="-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard" \
        -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "${SW_DIR}/build_cortex_a9" --target avid_daemon -j2 >/dev/null
    cp -f "${SW_DIR}/build_cortex_a9/avid_daemon" "${IMG_DIR}/avid_daemon_cortex_a9"
    echo "[QEMU Build] Built Zynq-7020 ARM Cortex-A9 ELF: ${IMG_DIR}/avid_daemon_cortex_a9"
fi

if command -v aarch64-linux-gnu-g++ >/dev/null 2>&1 && command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
    echo "[QEMU Build] Cross-compiling AVID daemon for ARM64 (aarch64-linux-gnu)..."
    mkdir -p "${SW_DIR}/build_arm64"
    cmake -S "${SW_DIR}" -B "${SW_DIR}/build_arm64" \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
        -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "${SW_DIR}/build_arm64" --target avid_daemon -j2 >/dev/null
    cp -f "${SW_DIR}/build_arm64/avid_daemon" "${IMG_DIR}/avid_daemon_aarch64"
    echo "[QEMU Build] Built ARM64 ELF: ${IMG_DIR}/avid_daemon_aarch64"
fi

if [ ! -x "${IMG_DIR}/avid_daemon_cortex_a9" ] && [ ! -x "${IMG_DIR}/avid_daemon_aarch64" ]; then
    echo "[QEMU Build] Note: ARM cross-compilers not found on host; using native ELF fallback until setup_cloudshell.sh is run."
fi
