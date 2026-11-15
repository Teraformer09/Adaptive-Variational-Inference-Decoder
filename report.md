# Layer 2 Report — R2D2 LSTM Training, Checkpoint Export & C++ Runtime Parity

## 1. Layer Scope & Isolation
This branch contains **only Layer 2** of AVID (frozen baseline split from commit `54da68d`):
- PyTorch 64-unit Dueling LSTM R2D2 controller (`training/`)
- Binary (`int8`/`fp32`), JSON, and ONNX checkpoint export (`checkpoints/r2d2_v1_trained.*`)
- Standalone C++ R2D2 runtime & model loader (`deployment/cpp/`)
- Direct standalone C++ parity harness (`tests/cpp_layer2_parity_runner.cpp`)

## 2. Inputs (IP) & Outputs (OP) Validated
- **IP**: 8-feature QEC window observation vector `obs_t` + previous LSTM hidden/cell state `(h_t, c_t)` + binary checkpoint `checkpoints/r2d2_v1_trained.bin`
- **OP**: `action_id in {0..5}`, `bp_iterations K in {3,4,5}`, `pair_weight_scale gamma`, `state_value`, `q_values[6]`, bounded per-qubit `gain_residuals[9] in [-0.30, +0.30]`, verified to `< 1e-12` Python-to-C++ numerical parity

## 3. Execution & Log
- Runner script: `scripts/run_layer2.sh`
- Test suite: `tests/test_training.py`, `tests/test_checkpoint_export.py`, `tests/test_layer2_ip_op.py`
- Saved run log: [`logs/layer2_validation.log`](logs/layer2_validation.log)

```bash
bash scripts/run_layer2.sh
```
