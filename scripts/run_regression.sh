#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "====================================================================="
echo " AVID V1 Full Regression Suite (Smoke + RTL Regression + Integration)"
echo "====================================================================="
bash "${REPO_ROOT}/scripts/run_all.sh" "${REPO_ROOT}/configs/system/v1_smoke.yaml"
bash "${REPO_ROOT}/scripts/run_all.sh" "${REPO_ROOT}/configs/rtl/rtl_regression.yaml"
bash "${REPO_ROOT}/scripts/run_all.sh" "${REPO_ROOT}/configs/system/v1_integration.yaml"
python3 "${REPO_ROOT}/tools/verify_v0_closure_40.py"
echo "====================================================================="
echo " [REGRESSION PASS] Smoke, RTL Regression, Integration & V0 40/40"
echo "====================================================================="
