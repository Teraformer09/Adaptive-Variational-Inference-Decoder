#!/usr/bin/env bash
set -euo pipefail

if [[ -d "${HOME}/.local/bin" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if [[ -d "${HOME}/.local/verilator/bin" ]]; then
  export PATH="${HOME}/.local/verilator/bin:${PATH}"
fi

ALLOW_DIRTY_FLAG=""
for arg in "$@"; do
  if [[ "${arg}" == "--allow-dirty" ]]; then
    ALLOW_DIRTY_FLAG="--allow-dirty"
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="remediated_full_layer5_v1"
RUN_DIR="${REPO_ROOT}/validation/layer5/runs/${RUN_ID}"

rm -rf "${RUN_DIR}" "${REPO_ROOT}/build" "${REPO_ROOT}/firmware/build_layer5"
mkdir -p \
  logs \
  firmware/build_layer5 \
  results/plots \
  "${RUN_DIR}/sets" \
  "${RUN_DIR}/standalone" \
  "${RUN_DIR}/waveforms" \
  "${RUN_DIR}/manifests" \
  "${RUN_DIR}/acceptance" \
  validation/layer5/reports

LOG_FILE="logs/layer5_validation.log"

{
  echo "=== AVID Layer 5 Full 5-Set Audit-Grade FPGA RTL Validation ==="
  echo "Branch: $(git rev-parse --abbrev-ref HEAD) | Commit: $(git rev-parse --short HEAD)"

  if ! command -v verilator >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
      echo "[Layer 5] 'verilator' not found on PATH; auto-installing via apt-get..."
      sudo apt-get update -qq && sudo apt-get install -y -qq verilator
    fi
  fi
  if ! command -v verilator >/dev/null 2>&1; then
    echo "[FATAL] Native Verilator (>= 5.0) is required for Layer 5 audit-grade RTL validation." >&2
    exit 2
  fi

  VLT_VER="$(verilator --version)"
  VLT_FLAGS="--cc --exe --build -j 4 --trace --assert -Wno-fatal --top-module aa_qec_top"
  echo "[Layer 5] Compiling native SystemVerilog aa_qec_top + bound SVA with ${VLT_VER}..."

  verilator --cc --exe --build -j 4 --trace --assert -Wno-fatal --top-module aa_qec_top \
    -Mdir "${REPO_ROOT}/firmware/build_layer5/obj_standalone" \
    -CFLAGS "-O2 -std=c++17 -DUSE_VERILATOR=1 -I${REPO_ROOT}/emulator/verilator/tb" \
    "${REPO_ROOT}/firmware/rtl/ingress/axis_ingress_tap.sv" \
    "${REPO_ROOT}/firmware/rtl/ingress/axis_syndrome_rx.sv" \
    "${REPO_ROOT}/firmware/rtl/decoder/check_node_minsum.sv" \
    "${REPO_ROOT}/firmware/rtl/decoder/axis_correction_tx.sv" \
    "${REPO_ROOT}/firmware/rtl/config/weight_map_reg.sv" \
    "${REPO_ROOT}/firmware/rtl/decoder/bp_engine.sv" \
    "${REPO_ROOT}/firmware/rtl/top/aa_qec_top.sv" \
    "${REPO_ROOT}/firmware/formal/assertions/sva_weight_commit_assertions.sv" \
    "${REPO_ROOT}/emulator/verilator/tb/standalone_rtl_tb.cpp" \
    -o standalone_rtl_tb

  cp -f firmware/build_layer5/obj_standalone/standalone_rtl_tb firmware/build_layer5/standalone_rtl_tb

  # Write SHA-256 RTL & SVA build provenance manifest
  python3 - "${REPO_ROOT}" "${RUN_DIR}/manifests/rtl_build_provenance.json" "${VLT_VER}" "${VLT_FLAGS}" <<'PY'
import hashlib, json, sys
from pathlib import Path

repo = Path(sys.argv[1])
out_path = Path(sys.argv[2])
vlt_ver = sys.argv[3]
vlt_flags = sys.argv[4]

sources = [
    "firmware/rtl/top/aa_qec_top.sv",
    "firmware/rtl/ingress/axis_ingress_tap.sv",
    "firmware/rtl/ingress/axis_syndrome_rx.sv",
    "firmware/rtl/config/weight_map_reg.sv",
    "firmware/rtl/decoder/bp_engine.sv",
    "firmware/rtl/decoder/check_node_minsum.sv",
    "firmware/rtl/decoder/axis_correction_tx.sv",
    "firmware/formal/assertions/sva_weight_commit_assertions.sv",
    "firmware/constraints/aa_qec.xdc",
    "emulator/verilator/tb/verilator_rtl_driver.h",
    "emulator/verilator/tb/standalone_rtl_tb.cpp",
]
digests = {}
for rel in sources:
    digests[rel] = hashlib.sha256((repo / rel).read_bytes()).hexdigest()

prov = {
    "layer": 5,
    "top_module": "aa_qec_top",
    "verilator_version": vlt_ver,
    "verilator_flags": vlt_flags,
    "sva_bound_module": "sva_weight_commit_assertions",
    "source_sha256": digests,
}
out_path.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
PY

  echo "[Layer 5] Running standalone M1 native Verilator TB + VCD waveform capture..."
  ./firmware/build_layer5/standalone_rtl_tb \
    --require-native-rtl \
    --csv results/rtl_cycle_matrix.csv \
    --json results/m1_standalone.json \
    --vcd "${RUN_DIR}/waveforms/standalone_rtl_tb.vcd"

  cp -f results/rtl_cycle_matrix.csv "${RUN_DIR}/standalone/rtl_cycle_matrix.csv"
  cp -f results/m1_standalone.json "${RUN_DIR}/standalone/m1_standalone.json"

  echo "[Layer 5] Running pytest IP/OP suite..."
  PYTHONPATH=. python3 -m pytest -v tests/test_layer5_ip_op.py

  echo "[Layer 5] Building CMake 5-Set Verilator C++ test suite..."
  cmake -S tests/layer5 -B build/layer5_tests -DCMAKE_BUILD_TYPE=Release
  cmake --build build/layer5_tests -j 4

  export AVID_L5_REPO_ROOT="${REPO_ROOT}"

  echo "[Layer 5] Executing Set 1 — Input, AXI4-Lite & AXI4-Stream Ingress Contracts (44 tests)..."
  ./build/layer5_tests/test_axis_ingress_contract  --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set1.jsonl"
  ./build/layer5_tests/test_axil_register_contract --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set1.jsonl"

  echo "[Layer 5] Executing Set 2 — Output, Deterministic Cycle Law & Correction Stream (38 tests)..."
  ./build/layer5_tests/test_cycle_latency_law          --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set2.jsonl"
  ./build/layer5_tests/test_correction_output_contract --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set2.jsonl"

  echo "[Layer 5] Executing Set 3 — Pipeline, Ingress Tap & Atomic Weight-Bank Flow (36 tests)..."
  ./build/layer5_tests/test_ingress_tap_pipeline --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set3.jsonl"
  ./build/layer5_tests/test_weight_bank_flow     --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set3.jsonl"

  echo "[Layer 5] Executing Set 5 — Q7.8 Min-Sum, OSD-0, Formal SVA & Constraints (42 tests)..."
  ./build/layer5_tests/test_q78_minsum_osd_math --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set5.jsonl"
  PYTHONPATH=. python3 tests/layer5/set5_hardware_verification/test_sva_and_constraints.py \
    --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set5.jsonl"

  echo "[Layer 5] Executing Set 4 — Reset, Backpressure, 10k Streaming Stress & Persistence (40 tests)..."
  ./build/layer5_tests/test_reset_and_backpressure --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set4.jsonl"
  ./build/layer5_tests/test_streaming_stress       --repo-root "${REPO_ROOT}" --out-jsonl "${RUN_DIR}/sets/set4.jsonl"
  PYTHONPATH=. python3 tests/layer5/set4_execution_stress/test_artifact_persistence.py \
    --repo-root "${REPO_ROOT}" --run-dir "${RUN_DIR}" --out-jsonl "${RUN_DIR}/sets/set4.jsonl"

  echo "[Layer 5] Cleaning temporary CMake & Verilator build directories prior to clean-tree gate check..."
  rm -rf build firmware/build_layer5

  echo "[Layer 5] Evaluating Acceptance Gates G1–G8..."
  if [[ -n "${ALLOW_DIRTY_FLAG}" ]]; then
    PYTHONPATH=. python3 tests/layer5/acceptance/evaluate_layer5_gates.py \
      --repo-root "${REPO_ROOT}" --run-id "${RUN_ID}" --allow-dirty
  else
    PYTHONPATH=. python3 tests/layer5/acceptance/evaluate_layer5_gates.py \
      --repo-root "${REPO_ROOT}" --run-id "${RUN_ID}"
  fi

  echo "[Layer 5] Generating validation summary JSON & diagnostic plots..."
  PYTHONPATH=. python3 scripts/plot_layer5_results.py

  echo "[PASS] Layer 5 5-Set (200/200) & G1-G8 validation complete."
} 2>&1 | tee "${LOG_FILE}"
