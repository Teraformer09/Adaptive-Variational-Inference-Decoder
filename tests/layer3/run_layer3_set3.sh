#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VAL_DIR="${1:-${REPO_ROOT}/validation/layer3/runs/set3_adhoc}"
mkdir -p "${VAL_DIR}/logs/set3" "${VAL_DIR}/raw_results/set3"
python3 "${SCRIPT_DIR}/scripts/generate_fixtures.py" --repo-root "${REPO_ROOT}" --validation-dir "${VAL_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${REPO_ROOT}/build/layer3_set3" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAVID_VALIDATION_DIR="${VAL_DIR}"
cmake --build "${REPO_ROOT}/build/layer3_set3" -j4
ctest --test-dir "${REPO_ROOT}/build/layer3_set3" -R "^L3-FLOW-" --output-on-failure -j4
rm -rf "${REPO_ROOT}/build/layer3_set3"
