#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/validation/layer3/logs/set1"
mkdir -p "${LOG_DIR}" "${REPO_ROOT}/validation/layer3/raw_results/set1"

echo "========================================================================"
echo " AVID Layer 3 — Test Set 1 (Input Contracts, C11 ABI, Daemon Integration)"
echo "========================================================================"

bash "${SCRIPT_DIR}/scripts/verify_branch.sh" "${REPO_ROOT}"
bash "${SCRIPT_DIR}/scripts/verify_environment.sh" "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/scripts/generate_traceability_report.py" --repo-root "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/scripts/generate_fixtures.py" --repo-root "${REPO_ROOT}"
bash "${SCRIPT_DIR}/scripts/build_layer3_tests.sh" "${REPO_ROOT}"

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
    --sanitizer-ctest-rc "${SAN_RC}" \
    --run-id "88df97e_set1_baseline"

python3 "${SCRIPT_DIR}/acceptance/evaluate_acceptance.py" --repo-root "${REPO_ROOT}"

echo "========================================================================"
echo " Layer 3 Set 1 execution complete -> ${REPO_ROOT}/validation/layer3/"
echo "========================================================================"
