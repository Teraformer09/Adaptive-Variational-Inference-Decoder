#!/usr/bin/env bash
# =============================================================================
# emulator/verilator/build.sh — Compile AVID RTL Model & Co-Simulation Bridge
# Supports:
#   - Native Verilator compilation (--cc --exe --build --trace --assert)
#   - Mandatory Native Verilator enforcement when AVID_REQUIRE_NATIVE_VERILATOR=1
#   - Full RTL source & generated header SHA-256 build provenance manifest
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"
BRIDGE_DIR="${REPO_ROOT}/emulator/bridge"
mkdir -p "${MODEL_DIR}" "${SCRIPT_DIR}/scripts"

RTL_FILES=(
    "${REPO_ROOT}/firmware/rtl/decoder/check_node_minsum.sv"
    "${REPO_ROOT}/firmware/rtl/ingress/axis_syndrome_rx.sv"
    "${REPO_ROOT}/firmware/rtl/decoder/axis_correction_tx.sv"
    "${REPO_ROOT}/firmware/rtl/config/weight_map_reg.sv"
    "${REPO_ROOT}/firmware/rtl/decoder/bp_engine.sv"
    "${REPO_ROOT}/firmware/rtl/top/aa_qec_top.sv"
)

REQUIRE_NATIVE="${AVID_REQUIRE_NATIVE_VERILATOR:-0}"

if command -v verilator >/dev/null 2>&1; then
    echo "[Verilator Build] Compiling SystemVerilog RTL (aa_qec_top) with Verilator (--assert enabled)..."
    VERILATOR_FLAGS="--cc --exe --build --trace --assert -Wno-fatal --top-module aa_qec_top"
    CFLAGS_BASE="-O2 -std=c++17 -DUSE_VERILATOR=1 -I${SCRIPT_DIR}/tb -I${BRIDGE_DIR}/include -I${REPO_ROOT}/software/src"

    verilator ${VERILATOR_FLAGS} \
        -Mdir "${MODEL_DIR}/obj_standalone" \
        -CFLAGS "${CFLAGS_BASE}" \
        "${RTL_FILES[@]}" \
        "${SCRIPT_DIR}/tb/standalone_rtl_tb.cpp" \
        -o standalone_rtl_tb >"${MODEL_DIR}/verilator_standalone_build.log" 2>&1
    cp -f "${MODEL_DIR}/obj_standalone/standalone_rtl_tb" "${MODEL_DIR}/standalone_rtl_tb"

    verilator ${VERILATOR_FLAGS} \
        -Mdir "${MODEL_DIR}/obj_bridge" \
        -CFLAGS "${CFLAGS_BASE}" \
        "${RTL_FILES[@]}" \
        "${BRIDGE_DIR}/src/cosim_bridge_main.cpp" \
        -o avid_cosim_bridge >"${MODEL_DIR}/verilator_bridge_build.log" 2>&1
    cp -f "${MODEL_DIR}/obj_bridge/avid_cosim_bridge" "${MODEL_DIR}/avid_cosim_bridge"

    mkdir -p "${BRIDGE_DIR}/build"
    cmake -S "${BRIDGE_DIR}" -B "${BRIDGE_DIR}/build" -DCMAKE_BUILD_TYPE=Release >"${MODEL_DIR}/verilator_m3_build.log" 2>&1
    cmake --build "${BRIDGE_DIR}/build" --target test_bridge_transactions -j2 >>"${MODEL_DIR}/verilator_m3_build.log" 2>&1
    cp -f "${BRIDGE_DIR}/build/test_bridge_transactions" "${MODEL_DIR}/test_bridge_transactions"

    BUILD_MODE="verilator_c++_rtl"
    VERILATOR_VER="$(verilator --version | head -n1)"
    GEN_HDR_HASH="$(sha256sum "${MODEL_DIR}/obj_standalone/Vaa_qec_top.h" | awk '{print $1}')"
    echo "[Verilator Build] Built Native Verilated binaries in ${MODEL_DIR} (${VERILATOR_VER})"
else
    if [[ "${REQUIRE_NATIVE}" == "1" || "${REQUIRE_NATIVE}" == "true" ]]; then
        echo "[Verilator Build] ERROR: AVID_REQUIRE_NATIVE_VERILATOR=1 is set, but 'verilator' is not installed!" >&2
        exit 2
    fi
    echo "[Verilator Build] Building C++ cycle-exact reference fallback model & bridge via CMake..."
    mkdir -p "${BRIDGE_DIR}/build"
    cmake -S "${BRIDGE_DIR}" -B "${BRIDGE_DIR}/build" -DCMAKE_BUILD_TYPE=Release >"${MODEL_DIR}/cmake_fallback_build.log" 2>&1
    cmake --build "${BRIDGE_DIR}/build" -j2 >>"${MODEL_DIR}/cmake_fallback_build.log" 2>&1
    cp -f "${BRIDGE_DIR}/build/standalone_rtl_tb" "${MODEL_DIR}/standalone_rtl_tb"
    cp -f "${BRIDGE_DIR}/build/avid_cosim_bridge" "${MODEL_DIR}/avid_cosim_bridge"
    cp -f "${BRIDGE_DIR}/build/test_bridge_transactions" "${MODEL_DIR}/test_bridge_transactions"
    BUILD_MODE="cpp_cycle_exact_reference_fallback"
    VERILATOR_VER="not_installed_fallback_active"
    GEN_HDR_HASH="none_fallback_mode"
fi

# Generate RTL Build Provenance Manifest (Section 2.1)
python3 - <<PYEOF
import hashlib, json, subprocess, os

repo_root = "${REPO_ROOT}"
model_dir = "${MODEL_DIR}"
rtl_files = [
    "firmware/rtl/decoder/check_node_minsum.sv",
    "firmware/rtl/ingress/axis_syndrome_rx.sv",
    "firmware/rtl/decoder/axis_correction_tx.sv",
    "firmware/rtl/config/weight_map_reg.sv",
    "firmware/rtl/decoder/bp_engine.sv",
    "firmware/rtl/top/aa_qec_top.sv",
]
rtl_hashes = {}
for rel in rtl_files:
    p = os.path.join(repo_root, rel)
    with open(p, "rb") as f:
        rtl_hashes[rel] = hashlib.sha256(f.read()).hexdigest()

cxx_ver = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
git_commit = subprocess.check_output(["git", "-C", repo_root, "rev-parse", "HEAD"], text=True).strip()

prov = {
    "simulator_mode": "${BUILD_MODE}",
    "native_verilator_compiled": ("${BUILD_MODE}" == "verilator_c++_rtl"),
    "verilator_version": "${VERILATOR_VER}",
    "host_cxx_version": cxx_ver,
    "verilator_flags": "--cc --exe --build --trace --assert -Wno-fatal --top-module aa_qec_top",
    "cflags": "-O2 -std=c++17 -DUSE_VERILATOR=1",
    "git_commit": git_commit,
    "rtl_source_sha256": rtl_hashes,
    "generated_verilator_header_sha256": "${GEN_HDR_HASH}"
}
with open(os.path.join(model_dir, "rtl_build_provenance.json"), "w") as f:
    json.dump(prov, f, indent=2)
PYEOF
