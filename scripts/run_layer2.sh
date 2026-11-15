#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs tests/build_layer2
LOG_FILE="logs/layer2_validation.log"

{
  echo "=== Layer 2 Validation: R2D2 ML Training, Export & C++ Parity ==="
  echo "Branch: $(git rev-parse --abbrev-ref HEAD) | Commit: $(git rev-parse --short HEAD)"
  g++ -O2 -std=c++17 -Wall -I. \
    tests/cpp_layer2_parity_runner.cpp \
    deployment/cpp/model_loader.cpp \
    deployment/cpp/r2d2_runtime.cpp \
    -o tests/build_layer2/cpp_layer2_parity_runner -lm
  PYTHONPATH=. python3 -m pytest -v \
    tests/test_training.py \
    tests/test_checkpoint_export.py \
    tests/test_layer2_ip_op.py
  rm -rf tests/build_layer2
  echo "[PASS] Layer 2 IP & OP verification complete."
} 2>&1 | tee "${LOG_FILE}"
