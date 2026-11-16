#!/usr/bin/env bash
set -euo pipefail
if [[ -d "${HOME}/.local/bin" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if [[ -d "${HOME}/.local/verilator/bin" ]]; then
  export PATH="${HOME}/.local/verilator/bin:${PATH}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs firmware/build_layer5 results/plots
LOG_FILE="logs/layer5_validation.log"

{
  echo "=== Layer 5 Validation: 250 MHz SystemVerilog RTL Decoder ==="
  echo "Branch: $(git rev-parse --abbrev-ref HEAD) | Commit: $(git rev-parse --short HEAD)"
  if command -v verilator >/dev/null 2>&1; then
    echo "[Layer 5] Verilator detected — compiling native SystemVerilog aa_qec_top (--assert)..."
    verilator --cc --exe --build -j 4 --trace --assert -Wno-fatal --top-module aa_qec_top \
      -Mdir firmware/build_layer5/obj_standalone \
      -CFLAGS "-O2 -std=c++17 -DUSE_VERILATOR=1 -I${REPO_ROOT}/emulator/verilator/tb" \
      firmware/rtl/decoder/check_node_minsum.sv \
      firmware/rtl/ingress/axis_syndrome_rx.sv \
      firmware/rtl/decoder/axis_correction_tx.sv \
      firmware/rtl/config/weight_map_reg.sv \
      firmware/rtl/decoder/bp_engine.sv \
      firmware/rtl/top/aa_qec_top.sv \
      emulator/verilator/tb/standalone_rtl_tb.cpp \
      -o standalone_rtl_tb
    cp -f firmware/build_layer5/obj_standalone/standalone_rtl_tb firmware/build_layer5/standalone_rtl_tb
  else
    echo "[Layer 5] Compiling cycle-exact RTL C++ testbench (install Verilator for native RTL --assert)..."
    g++ -O2 -std=c++17 -Wall \
      -Iemulator/verilator/tb \
      emulator/verilator/tb/standalone_rtl_tb.cpp \
      -o firmware/build_layer5/standalone_rtl_tb
  fi
  ./firmware/build_layer5/standalone_rtl_tb \
    --csv results/rtl_cycle_matrix.csv \
    --json results/m1_standalone.json
  PYTHONPATH=. python3 -m pytest -v tests/test_layer5_ip_op.py
  PYTHONPATH=. python3 scripts/plot_layer5_results.py
  rm -rf firmware/build_layer5
  echo "[PASS] Layer 5 IP, OP & plot generation complete."
} 2>&1 | tee "${LOG_FILE}"
