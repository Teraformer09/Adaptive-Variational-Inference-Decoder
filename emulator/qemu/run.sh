#!/usr/bin/env bash
# =============================================================================
# emulator/qemu/run.sh — Execute AVID Guest Client in QEMU ARM Cortex-A9 / ARM64
# Remediates DEF-L4-09:
#   - Locates user-space or system ARMhf / ARM64 sysroots
#   - Pins -cpu cortex-a9 (ARMv7) and -cpu cortex-a53 (ARM64)
#   - NEVER silently executes avid_daemon_x86_64 when an ARM guest architecture
#     was requested unless AVID_ALLOW_X86_FALLBACK=1 is explicitly set.
# =============================================================================
set -euo pipefail

if [[ -d "${HOME}/.local/bin" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG_DIR="${SCRIPT_DIR}/images"
PREF_ARCH="${AVID_QEMU_ARCH:-cortex-a9}"
ALLOW_FALLBACK="${AVID_ALLOW_X86_FALLBACK:-0}"

ARMHF_SYSROOT=""
if [[ -e "/usr/arm-linux-gnueabihf/lib/ld-linux-armhf.so.3" ]]; then
    ARMHF_SYSROOT="/usr/arm-linux-gnueabihf"
elif [[ -e "${HOME}/.local/avid-toolchains/usr/arm-linux-gnueabihf/lib/ld-linux-armhf.so.3" ]]; then
    ARMHF_SYSROOT="${HOME}/.local/avid-toolchains/usr/arm-linux-gnueabihf"
fi

AARCH64_SYSROOT=""
if [[ -e "/usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1" ]]; then
    AARCH64_SYSROOT="/usr/aarch64-linux-gnu"
elif [[ -e "${HOME}/.local/avid-toolchains/usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1" ]]; then
    AARCH64_SYSROOT="${HOME}/.local/avid-toolchains/usr/aarch64-linux-gnu"
fi

if [[ "${PREF_ARCH}" == "cortex-a9" || "${PREF_ARCH}" == "arm" || "${PREF_ARCH}" == "armv7" ]]; then
    if [[ -x "${IMG_DIR}/avid_daemon_cortex_a9" && -n "${ARMHF_SYSROOT}" ]]; then
        if command -v qemu-arm >/dev/null 2>&1; then
            exec qemu-arm -cpu cortex-a9 -L "${ARMHF_SYSROOT}" "${IMG_DIR}/avid_daemon_cortex_a9" "$@"
        elif command -v qemu-arm-static >/dev/null 2>&1; then
            exec qemu-arm-static -cpu cortex-a9 -L "${ARMHF_SYSROOT}" "${IMG_DIR}/avid_daemon_cortex_a9" "$@"
        fi
    fi
elif [[ "${PREF_ARCH}" == "aarch64" || "${PREF_ARCH}" == "arm64" || "${PREF_ARCH}" == "cortex-a53" ]]; then
    if [[ -x "${IMG_DIR}/avid_daemon_aarch64" && -n "${AARCH64_SYSROOT}" ]]; then
        if command -v qemu-aarch64 >/dev/null 2>&1; then
            exec qemu-aarch64 -cpu cortex-a53 -L "${AARCH64_SYSROOT}" "${IMG_DIR}/avid_daemon_aarch64" "$@"
        elif command -v qemu-aarch64-static >/dev/null 2>&1; then
            exec qemu-aarch64-static -cpu cortex-a53 -L "${AARCH64_SYSROOT}" "${IMG_DIR}/avid_daemon_aarch64" "$@"
        fi
    fi
elif [[ "${PREF_ARCH}" == "x86_64" || "${PREF_ARCH}" == "native" ]]; then
    if [[ -x "${IMG_DIR}/avid_daemon_x86_64" ]]; then
        exec "${IMG_DIR}/avid_daemon_x86_64" "$@"
    fi
fi

if [[ "${ALLOW_FALLBACK}" == "1" && -x "${IMG_DIR}/avid_daemon_x86_64" ]]; then
    echo "[QEMU Run Warning] AVID_ALLOW_X86_FALLBACK=1 active; falling back to x86_64 binary for requested arch '${PREF_ARCH}'." >&2
    exec "${IMG_DIR}/avid_daemon_x86_64" "$@"
fi

echo "[QEMU Run Error] Requested guest architecture '${PREF_ARCH}' is unavailable (missing QEMU emulator, ARM sysroot, or guest ELF in ${IMG_DIR}). Silent fallback to x86_64 is disabled." >&2
exit 2
