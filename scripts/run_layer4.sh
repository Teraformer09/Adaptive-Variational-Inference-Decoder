#!/usr/bin/env bash
# =============================================================================
# scripts/run_layer4.sh — Full Audit-Grade Layer 4 Validation & Gate Pipeline
# Executes:
#   1. Native Verilator RTL + Bridge compilation (AVID_REQUIRE_NATIVE_VERILATOR=1)
#   2. Cross-compiled ARM Cortex-A9 & ARM64 Cortex-A53 QEMU guest builds
#   3. Standalone Verilator RTL testbench (VCD + CSV + JSON) & IP/OP pytest
#   4. Five-Set Layer 4 Suite (223 canonical IDs: Set 1 -> Set 2 -> Set 5 -> Set 3 -> Set 4)
#   5. Acceptance Gate Evaluation (G1..G8) & Provenance Manifest Generation
# =============================================================================
set -euo pipefail

if [[ -d "${HOME}/.local/bin" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
fi
if [[ -d "${HOME}/.local/verilator/bin" ]]; then
    export PATH="${HOME}/.local/verilator/bin:${PATH}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="remediated_full_layer4_v1"
SKIP_EMU_REBUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --skip-emulator-rebuild)
            SKIP_EMU_REBUILD=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

RUN_DIR="${REPO_ROOT}/validation/layer4/runs/${RUN_ID}"
mkdir -p \
    "${REPO_ROOT}/logs" \
    "${RUN_DIR}/raw_results" \
    "${RUN_DIR}/results" \
    "${RUN_DIR}/logs" \
    "${RUN_DIR}/manifests" \
    "${RUN_DIR}/acceptance" \
    "${RUN_DIR}/artifacts"

LOG_FILE="${RUN_DIR}/logs/layer4_execution.log"
RESULTS_JSONL="${RUN_DIR}/raw_results/layer4_results.jsonl"
REPORT_MD="${REPO_ROOT}/validation/layer4/reports/layer4_validation_report.md"
rm -f "${RESULTS_JSONL}"

{
    echo "=== Layer 4 Validation: Co-Simulation Bridge & Emulators (Run: ${RUN_ID}) ==="
    echo "Branch: $(git rev-parse --abbrev-ref HEAD) | Commit: $(git rev-parse HEAD)"

    if [[ "${SKIP_EMU_REBUILD}" -eq 0 ]]; then
        AVID_REQUIRE_NATIVE_VERILATOR=1 bash emulator/verilator/build.sh
        AVID_REQUIRE_ARM_GUEST=1 bash emulator/qemu/build.sh
    fi

    echo "--- [Stage 1] Standalone Native Verilator RTL Testbench & IP/OP Pytest ---"
    ./emulator/verilator/models/standalone_rtl_tb \
        --require-native-rtl \
        --vcd "${RUN_DIR}/artifacts/standalone_rtl_tb.vcd" \
        --csv "${RUN_DIR}/results/standalone_rtl_tb_cycles.csv" \
        --json "${RUN_DIR}/results/standalone_rtl_tb.json"
    cp -f emulator/verilator/models/rtl_build_provenance.json "${RUN_DIR}/manifests/rtl_build_provenance.json"

    PYTHONPATH=. python3 -m pytest -v tests/test_layer4_ip_op.py

    echo "--- [Stage 2] Building 19 Layer 4 C++ Validation Binaries + Fallback Reference Bridge ---"
    cmake -S tests/layer4 -B build/layer4_tests -DCMAKE_BUILD_TYPE=Release
    cmake --build build/layer4_tests -j4

    export AVID_L4_REPO_ROOT="${REPO_ROOT}"
    export AVID_L4_BRIDGE_BIN="${REPO_ROOT}/emulator/verilator/models/avid_cosim_bridge"
    export AVID_L4_FALLBACK_BRIDGE_BIN="${REPO_ROOT}/build/layer4_tests/avid_cosim_bridge_fallback"

    BIN_DIR="${REPO_ROOT}/build/layer4_tests"
    WORK_TMP="/tmp/avid_l4_run_${RUN_ID}_$$"
    mkdir -p "${WORK_TMP}"

    echo "--- [Set 1] Input Contract Tests (47 IDs) ---"
    "${BIN_DIR}/test_rpc_schema"            --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_mmio_contract"         --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_syndrome_transfer"     --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_transaction_ordering"  --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_socket_recovery"       --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"

    echo "--- [Set 2] Output & Transaction Correctness Tests (28 IDs) ---"
    "${BIN_DIR}/test_acknowledgements"         --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_cycle_accounting"         --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_rtl_response_consistency" --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_deterministic_replay"     --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"

    echo "--- [Set 5] Hardware-Interface & Co-Simulation-Specific Tests (53 IDs) ---"
    "${BIN_DIR}/test_axil_protocol"          --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_axis_syndrome_protocol" --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_cycle_fidelity"         --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_decoder_equivalence"    --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_weight_bank_semantics"  --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_cosim_fidelity_limits"  --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"

    echo "--- [Set 3] Pipeline & System-Flow Tests (39 IDs) ---"
    python3 tests/layer4/set3_pipeline_flow/test_end_to_end_transactions.py --out-jsonl "${RESULTS_JSONL}" --work-dir "${WORK_TMP}/set3_e2e"
    python3 tests/layer4/set3_pipeline_flow/test_qemu_verilator_lockstep.py --out-jsonl "${RESULTS_JSONL}" --work-dir "${WORK_TMP}/set3_lock"
    python3 tests/layer4/set3_pipeline_flow/test_epoch_reconciliation.py    --out-jsonl "${RESULTS_JSONL}" --work-dir "${WORK_TMP}/set3_epoch"
    python3 tests/layer4/set3_pipeline_flow/test_m3_trace_integrity.py      --out-jsonl "${RESULTS_JSONL}" --work-dir "${WORK_TMP}/set3_m3"

    echo "--- [Set 4] Execution, Runtime & Recovery Tests (56 IDs) ---"
    "${BIN_DIR}/test_bridge_startup"   --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_runtime_stress"   --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_fault_injection"  --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    "${BIN_DIR}/test_restart_recovery" --repo-root "${REPO_ROOT}" --out-jsonl "${RESULTS_JSONL}"
    python3 tests/layer4/set4_execution_recovery/test_artifact_persistence.py --out-jsonl "${RESULTS_JSONL}" --work-dir "${WORK_TMP}/set4_pers"

    rm -rf "${WORK_TMP}"

    echo "--- [Stage 3] Evaluating Layer 4 Acceptance Gates (G1..G8) ---"
    python3 tests/layer4/acceptance/evaluate_layer4_gates.py \
        --repo-root "${REPO_ROOT}" \
        --run-dir "${RUN_DIR}" \
        --results-jsonl "${RESULTS_JSONL}" \
        --report-out "${REPORT_MD}" \
        --run-id "${RUN_ID}"

    echo "[PASS] Layer 4 complete 5-set validation & gate evaluation succeeded."
} 2>&1 | tee "${LOG_FILE}" "${REPO_ROOT}/logs/layer4_validation.log"
