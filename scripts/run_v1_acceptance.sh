#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "${REPO_ROOT}/scripts/run_all.sh" "${REPO_ROOT}/configs/system/v1_integration.yaml"
python3 "${REPO_ROOT}/tools/verify_provenance.py"
python3 "${REPO_ROOT}/tools/verify_requirements.py"
