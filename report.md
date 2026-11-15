# Layer 1 Report — Code Model, 7-Stage Noise Framework, Measurement Noise & Packet Codec

## 1. Layer Scope & Isolation
This branch contains **only Layer 1** of AVID (frozen baseline split from commit `54da68d`):
- Canonical rotated `d=3` CSS code model (`Hx.json`, `Hz.json`, Tanner tables, coordinates)
- 16-PD truncated-Gaussian library, 100s persistent assignment, 10 Hz synchronized multi-generator sampling, Pauli XOR / Bernoulli union composition
- Stage 5 dynamic measurement-noise generator & Markov sticky / dropout validity / observation delay model
- Stage 6 stratified dataset generator & Stage 7 exact 32-state Joint-DP distribution oracle
- `AVID-SYNDROME-PACKET-V1` 64-bit packet codec

## 2. Inputs (IP) & Outputs (OP) Validated
- **IP**: `configs/noise_framework/framework_v1.yaml`, `pd_profiles_16.yaml`, `stage5_measurement_imperfections.yaml`, `architecture/code_model/*.json`, `master_seed`, `duration`
- **OP**: `CompositeNoiseSample` trajectories, physical fault masks `(e_x, e_z)`, measurement-noisy `(s_x, s_z)` + binary validity masks + event/delivery timestamps, packed 64-bit `AVID-SYNDROME-PACKET-V1` words

## 3. Execution & Log
- Runner script: `scripts/run_layer1.sh`
- Dedicated IP/OP test: `tests/test_layer1_ip_op.py` + `simulation/tests/` + Stage 5/7 suites
- Saved run log: [`logs/layer1_validation.log`](logs/layer1_validation.log)

```bash
bash scripts/run_layer1.sh
```
