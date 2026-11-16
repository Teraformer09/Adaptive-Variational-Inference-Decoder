#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VAL_DIR="${1:-${REPO_ROOT}/validation/layer3/runs/set2_adhoc}"
mkdir -p "${VAL_DIR}/logs/set2" "${VAL_DIR}/raw_results/set2"
python3 "${SCRIPT_DIR}/scripts/generate_fixtures.py" --repo-root "${REPO_ROOT}" --validation-dir "${VAL_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${REPO_ROOT}/build/layer3_set2" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAVID_VALIDATION_DIR="${VAL_DIR}"
cmake --build "${REPO_ROOT}/build/layer3_set2" -j4
ctest --test-dir "${REPO_ROOT}/build/layer3_set2" -R "^(L3-OP-|L3-STAT-)" --output-on-failure -j4
rm -rf "${REPO_ROOT}/build/layer3_set2"
