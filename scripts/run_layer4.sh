#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs emulator/verilator/models
LOG_FILE="logs/layer4_validation.log"

{
  echo "=== Layer 4 Validation: Co-Simulation Bridge & Emulators ==="
  echo "Branch: $(git rev-parse --abbrev-ref HEAD) | Commit: $(git rev-parse --short HEAD)"
  cmake -S emulator/bridge -B emulator/bridge/build_layer4 -DCMAKE_BUILD_TYPE=Release
  cmake --build emulator/bridge/build_layer4 --target avid_cosim_bridge test_bridge_transactions -j2
  cp -f emulator/bridge/build_layer4/avid_cosim_bridge emulator/verilator/models/avid_cosim_bridge
  cp -f emulator/bridge/build_layer4/test_bridge_transactions emulator/verilator/models/test_bridge_transactions
  PYTHONPATH=. python3 -m pytest -v tests/test_layer4_ip_op.py
  rm -rf emulator/bridge/build_layer4 emulator/verilator/models
  echo "[PASS] Layer 4 IP & OP verification complete."
} 2>&1 | tee "${LOG_FILE}"
