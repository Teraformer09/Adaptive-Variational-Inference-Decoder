#!/usr/bin/env bash
# =============================================================================
# scripts/run_cloudshell_end_to_end.sh
# Runs the full AVID Seven-Stage & V1 Dual-Emulator end-to-end pipeline in
# Google Cloud Shell, saves all execution logs to repository log directories,
# and commits + pushes all updated logs and artifacts to origin/AVID_EPJ_Patches.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/Benchmark-Eval/V1_Seven_Stage_Noise_Validation/logs"
STAGE5_LOG_DIR="${REPO_ROOT}/stage5_measurement_validation/logs"
mkdir -p "${LOG_DIR}" "${STAGE5_LOG_DIR}"

TS="$(date -u +%Y%m%d_%H%M%S)"

echo "=== [1/5] Environment Setup (${TS}) ===" | tee "${LOG_DIR}/01_env_setup_${TS}.log"
bash "${REPO_ROOT}/scripts/setup_cloudshell.sh" 2>&1 | tee -a "${LOG_DIR}/01_env_setup_${TS}.log"
python3 -m pip install --user -q numpy scipy pyyaml pytest 2>&1 | tee -a "${LOG_DIR}/01_env_setup_${TS}.log"

echo "=== [2/5] Stage 5 Dynamic Measurement-Noise Pipeline ===" | tee "${STAGE5_LOG_DIR}/stage5_execution.log"
PYTHONPATH="${REPO_ROOT}" python3 -m simulation.stimulus.seven_stage.stage5_measurement_noise \
    --duration 250 \
    --seed 42 \
    --config configs/noise_framework/framework_v1.yaml 2>&1 | tee -a "${STAGE5_LOG_DIR}/stage5_execution.log" "${LOG_DIR}/02_stage5_execution_${TS}.log"

echo "=== [3/5] Seven-Stage Noise & 6-Decoder Validation Suite (Stages 1-7) ===" | tee "${LOG_DIR}/03_seven_stage_eval_${TS}.log"
PYTHONPATH="${REPO_ROOT}" python3 "${REPO_ROOT}/Benchmark-Eval/V1_Seven_Stage_Noise_Validation/run_seven_stage_eval.py" 2>&1 | tee -a "${LOG_DIR}/03_seven_stage_eval_${TS}.log"
cp "${LOG_DIR}/03_seven_stage_eval_${TS}.log" "${LOG_DIR}/seven_stage_eval_latest.log"

echo "=== [4/5] V1 Dual-Emulator Co-Simulation & Acceptance Suite ===" | tee "${LOG_DIR}/04_v1_acceptance_${TS}.log"
bash "${REPO_ROOT}/scripts/run_v1_acceptance.sh" 2>&1 | tee -a "${LOG_DIR}/04_v1_acceptance_${TS}.log"
cp "${LOG_DIR}/04_v1_acceptance_${TS}.log" "${LOG_DIR}/v1_acceptance_latest.log"

echo "=== [5/5] Full 204-Test Pytest Regression Suite ===" | tee "${STAGE5_LOG_DIR}/pytest_regression_204_passed.log"
PYTHONPATH="${REPO_ROOT}" python3 -m pytest "${REPO_ROOT}/simulation/tests/" "${REPO_ROOT}/tests/" -v 2>&1 | tee -a "${STAGE5_LOG_DIR}/pytest_regression_204_passed.log" "${LOG_DIR}/05_pytest_regression_${TS}.log"
cp "${STAGE5_LOG_DIR}/pytest_regression_204_passed.log" "${LOG_DIR}/pytest_regression_latest.log"

echo "=== Committing and Pushing Logs & Artifacts to GitHub (AVID_EPJ_Patches) ==="
git config user.name "Teraformer09"
git config user.email "tirthahaque@users.noreply.github.com"
git add -A "${REPO_ROOT}/stage5_measurement_validation" "${REPO_ROOT}/Benchmark-Eval" "${REPO_ROOT}/datasets/seven_stage_v1"
if ! git diff --cached --quiet; then
    git commit -m "Update Cloud Shell end-to-end execution logs and validation artifacts (${TS})"
    git push origin AVID_EPJ_Patches
    echo "[PASS] Logs and artifacts committed and pushed to origin/AVID_EPJ_Patches."
else
    echo "[INFO] No artifact changes to commit; pushing any pending commits."
    git push origin AVID_EPJ_Patches
fi
