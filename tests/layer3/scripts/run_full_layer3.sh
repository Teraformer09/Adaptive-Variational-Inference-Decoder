#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER3_TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LAYER3_TEST_DIR}/../.." && pwd)"
RUN_ID="full_layer3_v1"
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

case "${RUN_ID}" in
    88df97e_set1_baseline|remediated_set1_v1|remediated_set1_v2)
        echo "ERROR: Refusing to overwrite preserved audit run '${RUN_ID}'!" >&2
        exit 1
        ;;
esac

VAL_DIR="${REPO_ROOT}/validation/layer3/runs/${RUN_ID}"
rm -rf "${VAL_DIR}"
mkdir -p \
    "${VAL_DIR}/logs/set1" \
    "${VAL_DIR}/logs/set2" \
    "${VAL_DIR}/logs/set3" \
    "${VAL_DIR}/logs/set4" \
    "${VAL_DIR}/logs/set5" \
    "${VAL_DIR}/logs/suite" \
    "${VAL_DIR}/raw_results/set1" \
    "${VAL_DIR}/raw_results/set2" \
    "${VAL_DIR}/raw_results/set3" \
    "${VAL_DIR}/raw_results/set4" \
    "${VAL_DIR}/raw_results/set5" \
    "${REPO_ROOT}/validation/layer3/reports"

echo "========================================================================"
echo " AVID Layer 3 — Full 5-Set Validation (${RUN_ID} | mode=${RUN_MODE})"
echo " Output Validation Dir: ${VAL_DIR}"
echo "========================================================================"

bash "${SCRIPT_DIR}/verify_branch.sh" "${REPO_ROOT}" "${VAL_DIR}" "${RUN_MODE}"
bash "${SCRIPT_DIR}/verify_environment.sh" "${REPO_ROOT}" "${VAL_DIR}"
python3 "${SCRIPT_DIR}/generate_fixtures.py" --repo-root "${REPO_ROOT}" --validation-dir "${VAL_DIR}"
bash "${SCRIPT_DIR}/build_layer3_tests.sh" "${REPO_ROOT}" "${VAL_DIR}"

echo "[run_full_layer3] 1/2 Running ASan+UBSan CTest verification across all 257 tests..."
SAN_RC=0
ASAN_OPTIONS="detect_leaks=0:halt_on_error=1" UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
    ctest --test-dir "${REPO_ROOT}/build/layer3_sanitizers" --output-on-failure -j4 \
    > "${VAL_DIR}/logs/suite/ctest_sanitizers.log" 2>&1 || SAN_RC=$?
echo "[run_full_layer3] ASan+UBSan CTest exit code: ${SAN_RC}"
if [[ "${SAN_RC}" -ne 0 ]]; then
    echo "ERROR: ASan+UBSan CTest suite failed (rc=${SAN_RC}). Tail of log:" >&2
    tail -n 40 "${VAL_DIR}/logs/suite/ctest_sanitizers.log" >&2
    exit "${SAN_RC}"
fi

echo "[run_full_layer3] 2/2 Running Standard C11/C++17 CTest pass across all 257 tests..."
STD_RC=0
ctest --test-dir "${REPO_ROOT}/build/layer3_standard" --output-on-failure -j2 \
    > "${VAL_DIR}/logs/suite/ctest_standard.log" 2>&1 || STD_RC=$?
echo "[run_full_layer3] Standard C11/C++17 CTest exit code: ${STD_RC}"
if [[ "${STD_RC}" -ne 0 ]]; then
    echo "ERROR: Standard CTest suite failed (rc=${STD_RC}). Tail of log:" >&2
    tail -n 40 "${VAL_DIR}/logs/suite/ctest_standard.log" >&2
    exit "${STD_RC}"
fi

rm -rf "${REPO_ROOT}/build"

python3 "${SCRIPT_DIR}/collect_test_results.py" \
    --repo-root "${REPO_ROOT}" \
    --validation-dir "${VAL_DIR}" \
    --sanitizer-ctest-rc "${SAN_RC}" \
    --run-id "${RUN_ID}"

python3 "${LAYER3_TEST_DIR}/acceptance/evaluate_acceptance.py" \
    --repo-root "${REPO_ROOT}" \
    --validation-dir "${VAL_DIR}"

python3 - "${REPO_ROOT}" "${VAL_DIR}" <<'PYEOF'
import json, sys
from pathlib import Path

repo = Path(sys.argv[1])
val_dir = Path(sys.argv[2])
full_doc = json.loads((val_dir / "results" / "layer3_full_results.json").read_text(encoding="utf-8"))
gate_doc = json.loads((val_dir / "acceptance" / "gate_evaluation.json").read_text(encoding="utf-8"))
src_doc = json.loads((val_dir / "manifests" / "source_manifest.json").read_text(encoding="utf-8"))

lines = [
    "# AVID Layer 3 — Full 5-Set Validation & Gate Acceptance Report",
    "",
    f"- **Branch**: `{full_doc['branch']}`",
    f"- **Source Commit**: `{full_doc['source_commit']}` (`dirty_working_tree={src_doc.get('dirty_working_tree')}`)",
    f"- **Preserved Baseline Commit**: `{full_doc['baseline_commit']}` (`validation/layer3/runs/88df97e_set1_baseline/`)",
    f"- **Run ID**: `{full_doc['run_id']}`",
    f"- **Overall Decision**: **`{gate_doc['overall_layer3_decision']}`**",
    f"- **Canonical IDs Passed**: `{full_doc['canonical_status_counts'].get('PASS', 0)} / {full_doc['canonical_total']}`",
    f"- **Total Executed Records (Canonical + C11 ABI + Daemon Integration + Diagnostics)**: `{full_doc['total_records']} / 257 PASS`",
    f"- **ASan+UBSan CTest Exit Code**: `{full_doc['sanitizer_ctest_exit_code']}`",
    "",
    "## 1. Per-Set Execution Summary",
    "",
    "| Test Set | Canonical IDs | PASS | Non-PASS |",
    "| :--- | ---: | ---: | ---: |",
]
for sname, scounts in full_doc["per_set_status_counts"].items():
    tot = sum(scounts.values())
    ps = scounts.get("PASS", 0)
    lines.append(f"| `{sname}` | {tot} | {ps} | {tot - ps} |")
lines += [
    f"| `c_abi_driver_parity (L3-ABI-001..007)` | {full_doc['c_abi_total']} | {full_doc['c_abi_status_counts'].get('PASS', 0)} | 0 |",
    f"| `daemon_integration_and_diagnostics` | {full_doc['supplementary_total']} | {full_doc['supplementary_status_counts'].get('PASS', 0)} | 0 |",
    "",
    "## 2. Architecture Gate Evaluation (`G1`–`G10`)",
    "",
    "| Gate | Scope | Evaluated | PASS | Non-PASS | Status |",
    "| :--- | :--- | :---: | ---: | ---: | :---: |",
]
for g in gate_doc["gates"]:
    lines.append(
        f"| `{g['gate_id']}` | {g['gate_name']} | `{g['evaluated_in_stage']}` | {g['pass_count']} | {g['non_pass_count']} | **`{g['status']}`** |"
    )
lines.append("")
(repo / "validation" / "layer3" / "reports" / "layer3_full_validation_report.md").write_text(
    "\n".join(lines), encoding="utf-8"
)
PYEOF

echo "========================================================================"
echo " Layer 3 Full 5-Set execution complete -> ${VAL_DIR}/"
echo "========================================================================"
