#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="remediated_set1_v1"
RUN_MODE="remediated"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --mode)
            RUN_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "${RUN_ID}" == "88df97e_set1_baseline" || "${RUN_ID}" == "remediated_set1_v1" ]]; then
    echo "ERROR: Refusing to overwrite preserved audit run '${RUN_ID}'!" >&2
    exit 1
fi

VAL_DIR="${REPO_ROOT}/validation/layer3/runs/${RUN_ID}"
LOG_DIR="${VAL_DIR}/logs/set1"
mkdir -p "${LOG_DIR}" "${VAL_DIR}/raw_results/set1"

echo "========================================================================"
echo " AVID Layer 3 — Test Set 1 (${RUN_ID} | mode=${RUN_MODE})"
echo " Output Validation Dir: ${VAL_DIR}"
echo "========================================================================"

bash "${SCRIPT_DIR}/scripts/verify_branch.sh" "${REPO_ROOT}" "${VAL_DIR}" "${RUN_MODE}"
bash "${SCRIPT_DIR}/scripts/verify_environment.sh" "${REPO_ROOT}" "${VAL_DIR}"
python3 "${SCRIPT_DIR}/scripts/generate_traceability_report.py" --repo-root "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/scripts/generate_fixtures.py" --repo-root "${REPO_ROOT}" --validation-dir "${VAL_DIR}"
bash "${SCRIPT_DIR}/scripts/build_layer3_tests.sh" "${REPO_ROOT}" "${VAL_DIR}"

echo "[run_layer3_set1] Running ASan+UBSan CTest verification pass..."
SAN_RC=0
ASAN_OPTIONS="detect_leaks=0:halt_on_error=1" UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
    ctest --test-dir "${REPO_ROOT}/build/layer3_sanitizers" --output-on-failure -V \
    > "${LOG_DIR}/ctest_sanitizers.log" 2>&1 || SAN_RC=$?
echo "[run_layer3_set1] ASan+UBSan CTest exit code: ${SAN_RC}"

echo "[run_layer3_set1] Running Standard C11/C++17 CTest pass and writing canonical raw_results..."
ctest --test-dir "${REPO_ROOT}/build/layer3_standard" --output-on-failure -V \
    > "${LOG_DIR}/ctest_standard.log" 2>&1

python3 "${SCRIPT_DIR}/scripts/collect_test_results.py" \
    --repo-root "${REPO_ROOT}" \
    --validation-dir "${VAL_DIR}" \
    --sanitizer-ctest-rc "${SAN_RC}" \
    --run-id "${RUN_ID}"

python3 "${SCRIPT_DIR}/acceptance/evaluate_acceptance.py" \
    --repo-root "${REPO_ROOT}" \
    --validation-dir "${VAL_DIR}"

echo "========================================================================"
echo " Layer 3 Set 1 execution complete -> ${VAL_DIR}/"
echo "========================================================================"
