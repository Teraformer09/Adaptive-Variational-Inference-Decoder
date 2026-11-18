# AVID Layer 3 — Slow-Loop Embedded Software & Adaptive Control

## 1. Layer Work Description

Branch `layer3-slow-loop-software` isolates **Layer 3** of the Adaptive Variational Inference Decoder (AVID): the embedded C11/C++17 slow-loop software control plane linking syndrome ingress and R2D2 LSTM control to cycle-accurate FPGA decoder weight updates (`avid_daemon`).

Core implemented, remediated, and 100% verified modules across all 5 canonical test sets:
- **Ingress Validation & 4-State DMA Ring (`software/src/ingress/`, `software/src/driver/`)**: Validates 64-bit `AVID-SYNDROME-PACKET-V1` packets, Layer 1 Stage-6 `0xAAEC0001` 16-byte stream headers, headless raw 8-byte streams, sequence gaps/duplicates, epoch tags, and readout-dropout (`VALID = 0`) filtering into 16-bin histograms $(H_X[16], H_Z[16])$ with zero side effects on malformed frames.
- **16-State Joint-DP MAP Estimator & Signed Q7.8 Quantizer (`software/src/estimator/`, `software/src/inference/`)**: Exact 16-state dynamic-programming likelihood, analytical gradient (`< 1e-8` vs central finite differences), pre-allocated two-loop L-BFGS-B 21-parameter MAP solver enforcing authoritative `W_min = 64` (`AVID_SOLVER_MIN_WINDOW_SAMPLES = 64u`), 9-class observable orbit identifiability, and clamped signed Q7.8 LLR quantization (`[64, 12288]`).
- **Causal Scheduler, Fast Burst Detector, 4-State Safety FSM & HAL (`software/src/control/`, `software/src/hal/`)**: 16-round sliding-subwindow burst detector ($\le 20$-round trigger latency vs $10\text{ ms}$ slow loop), R2D2 LSTM causal controller, 4-state containment FSM (`NORMAL_ADAPTIVE`, `HOLD_LAST_GOOD`, `SAFE_STATIC_PRIOR`, `DEGRADED_BYPASS`) with post-fault model re-initialization recovery, busy-decoder shadow-swap protection (`weight_reg_driver.c`), warm-resume state persistence (`AvidPersistedStateV1`), and C11/C++17 ABI-identical HAL backends (`physical` MMIO & `sim` Unix-socket RPC).
- **Sequential 21-Parameter Bayesian Posterior & Identifiability Interface (`layer3/`, `tests/test_layer3_posterior_interface.py`)**: Exact 16-state syndrome likelihood (`forward_model/`), analytical sensitivity Jacobian & SVD nullspace decomposition ($\operatorname{rank}(J) = 9, \dim\ker(J) = 12$ in `identifiability/`), causal temporal encoder + logit-space Laplace posterior head with hierarchical coupler-endpoint prior refinement and Gauss-Hermite uncertainty calibration (`inference/`), and strictly non-leaky `Layer3PosteriorPacket` export (`interfaces/`) consumed by Layer 2 V2's 82-dimensional observation builder.

## 2. Run Script to Validate & Generate Results + Plots

Build and execute the C++ baseline suite, `avid_daemon --self-test`, `pytest` IP/OP check, the complete 257-test C11/C++17 + Python 32-seed Monte Carlo suite (under both Standard `RelWithDebInfo` and `ASan+UBSan`), evaluate all 10 acceptance gates (`G1`–`G10`), and generate plots from the repository root:

```bash
python3 -m pip install --user -q numpy pyyaml pytest matplotlib
bash scripts/run_layer3.sh
```

To regenerate `results/layer3_summary.json` and `results/plots/layer3_validation_summary.png` directly from the committed clean audit run (`full_layer3_v1`, with `88df97e_set1_baseline` and `remediated_set1_v2` preserved intact):

```bash
PYTHONPATH=. python3 scripts/plot_layer3_results.py
```

Generated outputs:
- Execution log & report: `logs/layer3_validation.log`, `validation/layer3/reports/layer3_full_validation_report.md`
- Gate manifest & summary: `validation/layer3/runs/full_layer3_v1/acceptance/gate_evaluation.json`, `results/layer3_summary.json`
- Diagnostic plot: `results/plots/layer3_validation_summary.png`

![Layer 3 Validation Summary](results/plots/layer3_validation_summary.png)

## 3. Validation Results

| Validation Set / Acceptance Gate | Target Scope | Baseline (`88df97e`) | Full Suite (`full_layer3_v1`) | Key Verified Evidence | Status |
| :--- | :--- | ---: | ---: | :--- | :--- |
| **Set 1** — Input, C11 ABI & Stage-6 (`G1`) | `L3-IP-001..040`, `7` `L3-ABI`, `5` `L3-INT` (`+2` diag) | `20 / 51` PASS | `52 / 52` PASS (`+2/2` diag) | `W_min=64`, C11/C++17 ABI parity, Stage-6 `.bin` & dropouts | **PASS** |
| **Set 2** — Outputs & 32-Seed Statistics (`G2`) | `L3-OP-001..036` + `L3-STAT-001..015` (`32` seeds/case) | Not evaluated | `51 / 51` PASS | Orbit RMSE $O(1/\sqrt{N})$, Q7.8 `[64,12288]`, KL mismatch $>5\times$ | **PASS** |
| **Set 3** — Pipeline Flow & Causality (`G3`, `G4`) | `L3-FLOW-001..045` (`35` dataflow + `10` causal timing) | Not evaluated | `45 / 45` PASS | Prefix causality, $\le 20$-round burst lag, busy-commit deferral | **PASS** |
| **Set 4** — Execution, Persistence & Faults (`G5`) | `L3-EXEC-001..045` (50k/100k RSS, warm resume, 5s 3-thread) | Not evaluated | `45 / 45` PASS | RSS growth $<16\text{ MB}$, bit-exact replay, SIGINT/SIGKILL recovery | **PASS** |
| **Set 5** — DP, L-BFGS, Rank, Q7.8 & FSM (`G6`–`G9`) | `L3-MATH-001..062` ($2^{21}$ brute-force DP, SVD rank `9`) | Not evaluated | `62 / 62` PASS | DP $<10^{-14}$ vs $2^{21}$ exact, grad $<10^{-8}$, 4-state FSM exhaustive | **PASS** |
| **Sanitizers & Baseline Self-Test** | Full 257-test `ASan+UBSan` CTest + `test_baseline_suite` | `rc != 0` | `257 / 257` PASS (`rc=0`) | Zero memory errors/UB; `avid_daemon --self-test` `< 1 ms` | **PASS** |
| **Sequential Posterior & SVD Interface** (`test_layer3_posterior_interface.py`) | Analytical Jacobian vs FD, $\operatorname{rank}(J)=9, \dim\ker(J)=12$, PPC TVD & Layer 2 V2 packet export | N/A (V2 new) | `2 / 2` PASS | Jacobian err `< 1e-8`, PPC TVD `< 0.05`, nullspace CV calibrated | **PASS** |
| **All 10 Gates `G1`–`G10`** (`evaluate_acceptance.py`) | `243` canonical + `7` ABI + `5` E2E + `2` diag + clean provenance | `G1`: FAIL | **`10 / 10` Gates PASS** | `ACCEPTED_ALL_GATES_G1_TO_G10_PASS` (`Dirty=False`) | **PASS** |
