#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${REPO_ROOT}/tools/verify_v0_closure_40.py"
pytest -q "${REPO_ROOT}/simulation/tests/"
