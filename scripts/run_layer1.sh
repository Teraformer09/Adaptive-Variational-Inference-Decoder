#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs results/plots
LOG_FILE="logs/layer1_validation.log"

{
  echo "=== Layer 1 Validation: Code Model, 7-Stage Noise, Measurement Noise & Packet Codec ==="
  echo "Branch: $(git rev-parse --abbrev-ref HEAD) | Commit: $(git rev-parse --short HEAD)"
  PYTHONPATH=. python3 -m pytest -v -k "not test_artifacts_and_rtl_consistency" \
    tests/test_layer1_ip_op.py \
    simulation/tests/test_code_geometry.py \
    simulation/tests/test_pd_profiles.py \
    simulation/tests/test_generator_assignment.py \
    simulation/tests/test_generator_lifecycle.py \
    simulation/tests/test_noise_composition.py \
    simulation/tests/test_syndrome_generation.py \
    simulation/tests/test_packet_roundtrip.py \
    simulation/tests/test_measurement_config.py \
    simulation/tests/test_measurement_parameters.py \
    simulation/tests/test_measurement_mapping.py \
    simulation/tests/test_measurement_generator.py \
    simulation/tests/test_measurement_registry.py \
    simulation/tests/test_measurement_assignment.py \
    simulation/tests/test_measurement_sampling.py \
    simulation/tests/test_measurement_model.py \
    simulation/tests/test_measurement_composition.py \
    simulation/tests/test_measurement_syndrome_integration.py \
    simulation/tests/test_measurement_avid_integration.py \
    simulation/tests/test_measurement_baselines.py \
    simulation/tests/test_pd_contract_and_pca.py \
    tests/test_seven_stage_noise_framework.py
  echo "=== Running Stage 5 Measurement Pipeline CLI (250s, seed=42) ==="
  PYTHONPATH=. python3 -m simulation.stimulus.seven_stage.stage5_measurement_noise \
    --duration 250 --seed 42 \
    --config configs/noise_framework/framework_v1.yaml \
    --output /tmp/layer1_stage5_out
  rm -rf /tmp/layer1_stage5_out
  echo "[PASS] Layer 1 IP & OP verification complete."
} 2>&1 | tee "${LOG_FILE}"

PYTHONPATH=. python3 scripts/plot_layer1_results.py
