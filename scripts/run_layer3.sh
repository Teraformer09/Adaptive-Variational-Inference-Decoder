#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs software/build_layer3
LOG_FILE="logs/layer3_validation.log"

{
  echo "=== Layer 3 Validation: Slow-Loop Embedded Software & Control ==="
  echo "Branch: $(git rev-parse --abbrev-ref HEAD) | Commit: $(git rev-parse --short HEAD)"
  cmake -S software -B software/build_layer3 -DCMAKE_BUILD_TYPE=Release
  cmake --build software/build_layer3 -j2
  ./software/build_layer3/test_baseline_suite
  ./software/build_layer3/avid_daemon --self-test
  PYTHONPATH=. python3 -m pytest -v tests/test_layer3_ip_op.py
  rm -rf software/build_layer3
  echo "[PASS] Layer 3 IP & OP verification complete."
} 2>&1 | tee "${LOG_FILE}"
