# Layer 3 Report — Slow-Loop Embedded Software, Joint-DP MAP Solver & Control FSM

## 1. Layer Scope & Isolation
This branch contains **only Layer 3** of AVID (frozen baseline split from commit `54da68d`):
- Ingress 64-bit packet validator & 4-state DMA ring buffer (`software/src/ingress/`, `software/src/driver/`)
- Exact 16-state Joint-DP likelihood + L-BFGS-B MAP solver + Q7.8 LLR quantizer (`software/src/estimator/`, `software/src/inference/`)
- 16-round sliding-subwindow burst detector + 4-state fault-containment fallback FSM (`software/src/control/`)
- Dual-loop `avid_daemon` executable & C++ `test_baseline_suite`

## 2. Inputs (IP) & Outputs (OP) Validated
- **IP**: `AVID-SYNDROME-PACKET-V1` 64-bit packets, windowed 16-bin histograms `(hist_x[16], hist_z[16])`, telemetry metrics
- **OP**: 21-parameter MAP estimate `theta_hat`, analytic gradient parity, quantized signed Q7.8 LLRs, burst-trigger flag (`<= 20` rounds), fallback FSM decisions, HAL shadow-bank register updates

## 3. Execution & Log
- Runner script: `scripts/run_layer3.sh`
- Test suite: `software/tests/test_baseline_suite.cpp`, `avid_daemon --self-test`, `tests/test_layer3_ip_op.py`
- Saved run log: [`logs/layer3_validation.log`](logs/layer3_validation.log)

```bash
bash scripts/run_layer3.sh
```
