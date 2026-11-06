#!/usr/bin/env bash
# =============================================================================
# emulator/verilator/build.sh — Compile AVID RTL Model & Co-Simulation Bridge
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

if command -v verilator >/dev/null 2>&1; then
    echo "[Verilator Build] Compiling SystemVerilog RTL (aa_qec_top) with Verilator..."
    verilator --cc --exe --build --trace -Wno-fatal \
        --top-module aa_qec_top \
        -Mdir "${MODEL_DIR}/obj_standalone" \
        -CFLAGS "-O2 -std=c++17 -DUSE_VERILATOR=1 -I${SCRIPT_DIR}/tb" \
        "${RTL_FILES[@]}" \
        "${SCRIPT_DIR}/tb/standalone_rtl_tb.cpp" \
        -o "${MODEL_DIR}/standalone_rtl_tb" >/dev/null

    verilator --cc --exe --build --trace -Wno-fatal \
        --top-module aa_qec_top \
        -Mdir "${MODEL_DIR}/obj_bridge" \
        -CFLAGS "-O2 -std=c++17 -DUSE_VERILATOR=1 -I${SCRIPT_DIR}/tb -I${BRIDGE_DIR}/include" \
        "${RTL_FILES[@]}" \
        "${BRIDGE_DIR}/src/cosim_bridge_main.cpp" \
        -o "${MODEL_DIR}/avid_cosim_bridge" >/dev/null

    echo "[Verilator Build] Built Verilated binaries in ${MODEL_DIR}"
else
    echo "[Verilator Build] Building cycle-exact C++ RTL model & bridge via CMake..."
    mkdir -p "${BRIDGE_DIR}/build"
    cmake -S "${BRIDGE_DIR}" -B "${BRIDGE_DIR}/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "${BRIDGE_DIR}/build" -j2 >/dev/null
    cp -f "${BRIDGE_DIR}/build/standalone_rtl_tb" "${MODEL_DIR}/standalone_rtl_tb"
    cp -f "${BRIDGE_DIR}/build/avid_cosim_bridge" "${MODEL_DIR}/avid_cosim_bridge"
    cp -f "${BRIDGE_DIR}/build/test_bridge_transactions" "${MODEL_DIR}/test_bridge_transactions"
fi

# Always ensure test_bridge_transactions is compiled
mkdir -p "${BRIDGE_DIR}/build"
cmake -S "${BRIDGE_DIR}" -B "${BRIDGE_DIR}/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${BRIDGE_DIR}/build" --target test_bridge_transactions -j2 >/dev/null
cp -f "${BRIDGE_DIR}/build/test_bridge_transactions" "${MODEL_DIR}/test_bridge_transactions"
