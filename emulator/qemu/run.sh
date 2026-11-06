#!/usr/bin/env bash
# =============================================================================
# emulator/qemu/run.sh — Execute AVID Daemon in QEMU ARM64 Linux Environment
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG_DIR="${SCRIPT_DIR}/images"

if [ -x "${IMG_DIR}/avid_daemon_aarch64" ] && command -v qemu-aarch64 >/dev/null 2>&1; then
    exec qemu-aarch64 -L /usr/aarch64-linux-gnu "${IMG_DIR}/avid_daemon_aarch64" "$@"
elif [ -x "${IMG_DIR}/avid_daemon_aarch64" ] && command -v qemu-aarch64-static >/dev/null 2>&1; then
    exec qemu-aarch64-static -L /usr/aarch64-linux-gnu "${IMG_DIR}/avid_daemon_aarch64" "$@"
elif [ -x "${IMG_DIR}/avid_daemon_x86_64" ]; then
    exec "${IMG_DIR}/avid_daemon_x86_64" "$@"
else
    echo "[QEMU Run Error] avid_daemon binary not built. Run ./scripts/build_all.sh first." >&2
    exit 1
fi
