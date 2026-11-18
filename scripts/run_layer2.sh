#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/v2 tests/build_layer2 results/v2/plots results/plots
LOG_FILE="logs/v2/layer2_v2_validation.log"

{
  echo "=== Layer 2 V2 Validation: 82-D Syndrome-Conditioned R2D2-LSTM, Rewards, Ablations & C++17 Parity ==="
  echo "Branch: $(git rev-parse --abbrev-ref HEAD) | Commit: $(git rev-parse --short HEAD)"
  g++ -O2 -std=c++17 -Wall -I. \
    tests/parity/cpp_layer2_parity_runner.cpp \
    deployment/cpp/model_loader.cpp \
    deployment/cpp/r2d2_runtime.cpp \
    deployment/cpp/observation_adapter.cpp \
    -o tests/build_layer2/cpp_layer2_parity_runner -lm
  PYTHONPATH=. python3 -m pytest -v \
    tests/unit/test_spatial_mapping.py \
    tests/unit/test_posterior_contract.py \
    tests/unit/test_observation_builder.py \
    tests/unit/test_reward_components.py \
    tests/integration/test_layer3_layer2_interface.py \
    tests/integration/test_training_episode.py \
    tests/integration/test_decoder_control_update.py \
    tests/parity/test_layer2_cpp_python_parity.py \
    tests/test_training.py \
    tests/test_checkpoint_export.py \
    tests/test_layer2_ip_op.py
  PYTHONPATH=. python3 scripts/plot_layer2_results.py
  rm -rf tests/build_layer2
  echo "[PASS] Layer 2 V2 (and preserved V1) full validation & plot generation complete."
} 2>&1 | tee "${LOG_FILE}" | tee "logs/layer2_validation.log"
