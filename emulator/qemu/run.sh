#!/usr/bin/env bash
# =============================================================================
# emulator/qemu/run.sh — Execute AVID Daemon in QEMU ARM Cortex-A9 / ARM64
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG_DIR="${SCRIPT_DIR}/images"
PREF_ARCH="${AVID_QEMU_ARCH:-cortex-a9}"

if [ "${PREF_ARCH}" = "cortex-a9" ] && [ -x "${IMG_DIR}/avid_daemon_cortex_a9" ] && [ -e /usr/arm-linux-gnueabihf/lib/ld-linux-armhf.so.3 ] && command -v qemu-arm >/dev/null 2>&1; then
    exec qemu-arm -cpu cortex-a9 -L /usr/arm-linux-gnueabihf "${IMG_DIR}/avid_daemon_cortex_a9" "$@"
elif [ "${PREF_ARCH}" = "cortex-a9" ] && [ -x "${IMG_DIR}/avid_daemon_cortex_a9" ] && [ -e /usr/arm-linux-gnueabihf/lib/ld-linux-armhf.so.3 ] && command -v qemu-arm-static >/dev/null 2>&1; then
    exec qemu-arm-static -cpu cortex-a9 -L /usr/arm-linux-gnueabihf "${IMG_DIR}/avid_daemon_cortex_a9" "$@"
elif [ -x "${IMG_DIR}/avid_daemon_aarch64" ] && [ -e /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 ] && command -v qemu-aarch64 >/dev/null 2>&1; then
    exec qemu-aarch64 -L /usr/aarch64-linux-gnu "${IMG_DIR}/avid_daemon_aarch64" "$@"
elif [ -x "${IMG_DIR}/avid_daemon_aarch64" ] && [ -e /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 ] && command -v qemu-aarch64-static >/dev/null 2>&1; then
    exec qemu-aarch64-static -L /usr/aarch64-linux-gnu "${IMG_DIR}/avid_daemon_aarch64" "$@"
elif [ -x "${IMG_DIR}/avid_daemon_x86_64" ]; then
    exec "${IMG_DIR}/avid_daemon_x86_64" "$@"
else
    echo "[QEMU Run Error] avid_daemon binary not built. Run ./scripts/build_all.sh first." >&2
    exit 1
fi
