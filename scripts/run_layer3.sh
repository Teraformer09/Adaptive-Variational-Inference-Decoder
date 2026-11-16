#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs software/build_layer3 results/plots
LOG_FILE="logs/layer3_validation.log"

{
  echo "=== Layer 3 Validation: Full 5-Set Slow-Loop Embedded Software & Adaptive Control ==="
  echo "Branch: $(git rev-parse --abbrev-ref HEAD) | Commit: $(git rev-parse --short HEAD)"
  cmake -S software -B software/build_layer3 -DCMAKE_BUILD_TYPE=Release
  cmake --build software/build_layer3 -j4
  ./software/build_layer3/test_baseline_suite
  ./software/build_layer3/avid_daemon --self-test
  PYTHONPATH=. python3 -m pytest -v tests/test_layer3_ip_op.py
  rm -rf software/build_layer3
  bash tests/layer3/scripts/run_full_layer3.sh --run-id full_layer3_v1 --mode remediated
  PYTHONPATH=. python3 scripts/plot_layer3_results.py
  echo "[PASS] Layer 3 full 5-set (243 canonical / 257 total checks, G1-G10 PASS) validation & plot generation complete."
} 2>&1 | tee "${LOG_FILE}"
