# AVID: Adaptive Variational Inference Decoder for Real-Time Quantum Error Correction

**AVID** is a hardware/software co-designed real-time quantum error correction (QEC) decoding architecture for the rotated $d=3$ CSS surface code (`[[9, 1, 3]]`). It pairs a deterministic `250 MHz` SystemVerilog Min-Sum Belief Propagation + OSD-0 fast path (`firmware/rtl/`) with a causal two-pass 16-state Joint-DP L-BFGS estimator (`software/src/`) and a Python-trained 64-unit Dueling LSTM R2D2 controller (`training/`, `deployment/cpp/`, `checkpoints/r2d2_v1_trained.bin`, `19,744` parameters = `19.28 KB` `int8` / `77.12 KB` `fp32`).

> **Scope Note (Code-Capacity & Phenomenological Readout vs. Multi-Round Circuit-Level Fault Tolerance)**: In V1, the primary 10-scenario hardware-in-the-loop suite evaluates single-round code-capacity syndrome extraction with nearest-neighbour correlated crosstalk ($Z_i Z_j / X_i X_j$), non-stationary drift, and transient bursts (`1,536` rounds per scenario, resampling physical faults each round), while `heldout_phenomenological_readout` adds $1.5\%$ syndrome measurement bit-flip errors (`Benchmark-Eval/V1_Scientific_Evaluation/heldout_noise/`). Full $d$-round spacetime decoding over circuit-level detector graphs is scoped for V2.

---

## 1. Declared Hardware Targets (V1 Emulation vs. V2 Physical Scope)

To avoid conflating pre-silicon emulation targets with future physical board deployments, AVID declares explicit platform identities (see [`architecture/hardware/target_platforms.yaml`](architecture/hardware/target_platforms.yaml)):

| Scope | Target Platform | Processing System (PS) | Programmable Logic (PL) | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **V1 Primary Emulation Target** | **Xilinx Zynq-7020 SoC (`xc7z020clg484-1`)** | Dual-core **ARM Cortex-A9** (`ARMv7-A`, `arm-linux-gnueabihf-g++ -mcpu=cortex-a9` + `qemu-arm -cpu cortex-a9`) | Artix-7 FPGA fabric @ `250 MHz` (`4.0 ns`) simulated via native **`Verilator 5.020`** (`verilator_c++_rtl`) | **Implemented & Verified** (`avid_v1_integration_20261110_204636_s2026`) |
| **V1 Secondary Compatibility Build** | **ARM64 Linux Guest (`aarch64`)** | `aarch64-linux-gnu-g++` + `qemu-aarch64-static` | Same Verilator RTL bridge | **Implemented & Verified** (`avid_v1_full_benchmark_20261105_223251_s777`) |
| **V2 Intended Physical Target** | **Xilinx Zynq UltraScale+ MPSoC (`ZCU104` `xczu7ev` / `Kria K26` `xck26`)** | Quad-core **ARM Cortex-A53** (`ARMv8-A` @ `1.2 GHz`) | UltraScale+ FPGA fabric @ `250 MHz` | **Planned V2 Physical Board Scope** |

---

## 2. Three-Tier Causal Timing & Latency Contract

All timing claims are governed by [`architecture/contracts/system_timing.yaml`](architecture/contracts/system_timing.yaml) and [`architecture/control/fast_slow_partition.md`](architecture/control/fast_slow_partition.md):

### 2.1 Tier 1 — Fast-Path Hardware Decoder Latency (`250 MHz`, `4.0 ns` clock)
The RTL decoder supports three configurable OSD modes via MMIO register `0x058` (`cfg_osd_mode`) and runtime BP iteration budgets $K \in \{1 \dots 7\}$ via MMIO register `0x054` (`cfg_max_iter`, with status readback at `0x04C`), with `3` AXI4-Stream ingress cycles and `2` egress cycles (`5` cycles stream overhead):

| Iteration Count $K$ | OSD Mode (`0x058`) | BP Core Cycles $L_{\mathrm{core}}(K)$ | BP Core Latency (`ns`) | Total AXI4-Stream Cycles $L_{\mathrm{AXI}}(K)$ | Total AXI4-Stream Latency (`ns`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| $K = 1$ | Mode `0` / `1` | `7` (`2 + 5K`) | `28.0 ns` | `12` (`7 + 5K`) | `48.0 ns` |
| $K = 2$ | Mode `0` / `1` | `12` (`2 + 5K`) | `48.0 ns` | `17` (`7 + 5K`) | `68.0 ns` |
| **$K = 3$ (Default)** | **Mode `0` (Pure 9-LLR BP) / Mode `1` (Inline OSD-0)** | **`17` (`2 + 5K`)** | **`68.0 ns`** | **`22` (`7 + 5K`)** | **`88.0 ns`** |
| **$K = 3$** | **Mode `2` (Post-BP OSD-0 Stage)** | **`19` (`4 + 5K`)** | **`76.0 ns`** | **`24` (`9 + 5K`)** | **`96.0 ns`** |
| $K = 4$ | Mode `0` / `1` | `22` (`2 + 5K`) | `88.0 ns` | `27` (`7 + 5K`) | `108.0 ns` |
| $K = 5$ | Mode `0` / `1` | `27` (`2 + 5K`) | `108.0 ns` | `32` (`7 + 5K`) | `128.0 ns` |
| $K = 7$ | Mode `0` / `1` | `37` (`2 + 5K`) | `148.0 ns` | `42` (`7 + 5K`) | `168.0 ns` |

### 2.2 Tier 2 — Fast 16-Round Sliding-Subwindow Burst Detector
Implemented in [`software/src/control/causal_scheduler.cpp`](software/src/control/causal_scheduler.cpp) (`avid_fast_detector_step`) and executed every round in [`software/src/inference/avid_daemon.cpp`](software/src/inference/avid_daemon.cpp):
- Monitors a **16-round sliding subwindow** (`16 µs` @ `1 MHz`) with a syndrome-density threshold of **`0.35`** and a **`4`-round debounce**, achieving a worst-case detection latency of **`20` rounds (`20 µs`)** inside a `60 µs` burst and triggering immediate sub-epoch adaptation when a burst precursor is detected.

### 2.3 Tier 3 — Slow-Loop MAP Estimation & Trained R2D2 Control
- **V1 Emulation Window**: `window_rounds = 256` (`256 µs` syndrome window) for accelerated multi-epoch co-simulation (`configs/system/v1_integration.yaml`).
- **V2 Production Period**: `10,000` rounds (`10.0 ms` period @ `1 MHz`). Under `qemu-arm -cpu cortex-a9`, the two-pass 16-state Joint-DP L-BFGS + 64-unit LSTM R2D2 update completes in **`6.02 ms` mean** (`60.17%` of the `10 ms` budget).

---

## 3. Closed-Loop V1 Pipeline & Trained R2D2 Controller (`training/` $\to$ `deployment/cpp/`)

The closed-loop V1 control pipeline connects six explicit interfaces:
1. **Observation Interface**: Dual-basis syndrome histograms `(hist_x[16], hist_z[16])`, 64-round DMA-slot empirical syndrome density, and fast burst detector state.
2. **Estimation Interface**: 16-state Joint-DP MAP solver producing 21 physical parameters $\hat{\theta}_Z, \hat{\theta}_X$ and general 12-pair syndrome-alias projection across all nearest-neighbour edges in `NM_CROSSTALK_PAIRS` (zero hardcoded pair indices).
3. **Control Interface**: Trained 64-unit Dueling LSTM R2D2 network (`training/models/r2d2_network.py`, `19,744` parameters = `19.28 KB` `int8` / `77.12 KB` `fp32`, exported to [`checkpoints/r2d2_v1_trained.bin`](checkpoints/r2d2_v1_trained.bin) and [`checkpoints/r2d2_v1_trained.onnx`](checkpoints/r2d2_v1_trained.onnx)) loaded by [`deployment/cpp/model_loader.cpp`](deployment/cpp/model_loader.cpp) and [`deployment/cpp/r2d2_runtime.cpp`](deployment/cpp/r2d2_runtime.cpp) with verified `< 1e-12` Python-to-C++ numerical parity and executable-relative path resolution (`--checkpoint` / `/proc/self/exe`).
4. **Action-to-Decoder Interface**: Selects among 6 discrete actions $a_0 \dots a_5$ (`STATIC_PRIOR_K3`, `MODERATE_PROJ_K4`, `AGGRESSIVE_PROJ_K5`, `DAMPED_PRIOR_K4`, `FAST_RECALIBRATE_K5`, `SAFE_FALLBACK_K3`) plus 9 bounded per-qubit gain residuals $\Delta g \in [-0.30, +0.30]^9$, writing $K \in \{3, 4, 5\}$ to hardware register `0x054` and shadow Q7.8 weights to `0x000..0x048`.
5. **Safety & Fallback Interface**: 4-state fallback FSM (`software/src/control/fallback_fsm.cpp`) reverting to safe static priors upon unphysical parameter bounds, deadline overruns, or `SAFE_FALLBACK_K3`.
6. **Evaluation & Provenance Interface**: Cryptographic SHA-256 verification of checkpoints and RTL build provenance (`tools/verify_provenance.py`), with pre-remediation historical reports preserved in [`reports/archive/pre_remediation_b001b8d/SUPERSEDED_PROVENANCE.md`](reports/archive/pre_remediation_b001b8d/SUPERSEDED_PROVENANCE.md).

---

## 4. Structural Identifiability vs. Prior-Regularized MAP Estimation

As proven in [`mathematical_model/proofs/identifiability_proof.md`](mathematical_model/proofs/identifiability_proof.md) and [`architecture/noise_model/identifiability_report.md`](architecture/noise_model/identifiability_report.md):
1. **Data-Identifiable Syndrome Orbits (`rank(J) = 9`)**: Out of the 21 physical fault parameters per basis (9 single-qubit + 12 adjacent correlated pairs), single-window syndrome statistics independently identify **9 non-zero syndrome signature orbits** (`{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x8, 0xA, 0xC}`). The remaining **12 directions** lie in the gauge/collision nullspace ($\dim\ker(J) = 12$).
2. **Prior-Regularized MAP Hessian (`rank(H_MAP) = 21/21`)**: The Gaussian prior $\frac{\lambda}{2}\|\theta - \theta_{\mathrm{prior}}\|_2^2$ makes the MAP Hessian strictly positive-definite (`rank = 21/21`), selecting a unique canonical point within each gauge coset.
3. **Metric Interpretation**: `orbit_9class_rmse` measures data-driven syndrome-orbit recovery; `param_21_rmse` reflects prior-anchored selection along the 12 null directions and does **not** imply prior-free independent observability of all 21 physical rates.

---

## 5. Verified Benchmark & Ablation Summary

### 5.1 10-Seed (`2026..2035`) Multi-Scenario Evaluation (Static `B0` vs. Memoryless `Joint-DP Proj` vs. `AVID V1`)
Full artifact: [`Benchmark-Eval/V1_Scientific_Evaluation/multi_seed/README.md`](Benchmark-Eval/V1_Scientific_Evaluation/multi_seed/README.md) (`1,536` rounds/seed $\times$ `10` seeds = `15,360` rounds/scenario):

| Scenario | Static `B0` LER (Mean ± SD) | Joint-DP Proj LER (Mean ± SD) | Joint-DP Proj Rel. Reduction | AVID V1 (`avid_r2d2`) LER (Mean ± SD) | AVID V1 Rel. Reduction | Burst-Window Reduction | Paired $t$-test $p$ | Wilcoxon $p$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correlated_noise` | `0.08867 ± 0.00835` | `0.04909 ± 0.00638` | `-44.64%` | **`0.04798 ± 0.00692`** | **`-45.89%`** | `-45.89%` | `8.93e-07` | `1.95e-03` | `3.73` |
| `slowly_drifting_noise` | `0.06322 ± 0.00618` | `0.04453 ± 0.00490` | `-29.56%` | **`0.04245 ± 0.00535`** | **`-32.85%`** | `-47.27%` | `4.91e-06` | `1.95e-03` | `3.04` |
| `burst_disturbance` | `0.06048 ± 0.00550` | `0.04407 ± 0.00621` | `-27.13%` | **`0.04160 ± 0.00361`** | **`-31.21%`** | `-45.86%` | `1.02e-06` | `1.95e-03` | `3.67` |
| `short_burst_60us_vs_10ms` | `0.07676 ± 0.00654` | `0.04655 ± 0.00541` | `-39.35%` | **`0.04277 ± 0.00626`** | **`-44.27%`** | `-51.55%` | `1.19e-09` | `1.95e-03` | `7.95` |

### 5.2 5-Seed (`2026..2030`) $\times$ 7-Scenario Held-Out & Primary Generalization Summary
Across `5` seeds $\times$ `7` scenarios (`1,536` rounds/run = `53,760` rounds per controller), `avid_r2d2` outperforms both `static` and memoryless `joint_dp_proj` on **7 of 7 scenarios (`100%`)** while maintaining mean BP latency at `22.1–22.5` core cycles:

| Scenario (`5` Seeds Summed, `7,680` Rounds/Scenario) | `static` (`B0`) Failures | `joint_dp_proj` Failures | `avid_r2d2` Failures | `avid_r2d2` vs. `static` | `avid_r2d2` vs. `joint_dp_proj` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `stationary_noise` | `226` | `235` | **`225`** | `-0.44%` | `-4.26%` |
| `correlated_noise` | `703` | `395` | **`382`** | `-45.66%` | `-3.29%` |
| `slowly_drifting_noise` | `503` | `315` | **`313`** | `-37.77%` | `-0.63%` |
| `burst_disturbance` | `480` | `320` | **`309`** | `-35.63%` | `-3.44%` |
| `short_burst_60us_vs_10ms` | `602` | `354` | **`346`** | `-42.52%` | `-2.26%` |
| `heldout_multi_pair_12_67` (Unseen Pairs `(1,2)` & `(6,7)`) | `1213` | `655` | **`620`** | `-48.89%` | `-5.34%` |
| `heldout_spatial_3qubit_streak` (Unseen 3-Qubit Streak) | `986` | `840` | **`814`** | `-17.44%` | `-3.10%` |
| **Total Across All 35 Runs (`53,760` Rounds)** | **`4713`** | **`3114` (`-33.93%`)** | **`3009` (`-36.16%`)** | **`-36.16%`** | **`-3.37%` (`7/7` wins)** |

### 5.3 6-Way Architectural & Controller Ablation (`correlated_noise`, `seed = 2026`, `1,536` Rounds)
Full artifact: [`Benchmark-Eval/V1_Scientific_Evaluation/ablations/README.md`](Benchmark-Eval/V1_Scientific_Evaluation/ablations/README.md):

| Variant | Controller Mode | RTL OSD Mode | BP Core / Stream Cycles | Logical Failures | Logical Error Rate | Rel. Improvement vs. Static |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `1_Static_B0` | `static` | `1` | `17 / 22` | `147` | `0.09570` | `+0.00%` |
| `2_Marginal_LBFGS_B1` | `marginal_lbfgs` | `1` | `17 / 22` | `155` | `0.10091` | `-5.44%` |
| `3_Joint_DP_Pure_9Marginal_BP_17cyc` | `joint_dp_map` | `0` | `17 / 22` | `153` | `0.09961` | `-4.09%` |
| `4_Joint_DP_Pair_Projected_17cyc` | `joint_dp_proj` | `1` | `17 / 22` | `79` | `0.05143` | `+46.26%` |
| `5_Full_AVID_R2D2_WB_Inline_OSD_17cyc` | `avid_r2d2` | `1` | `24 / 29` | `79` | `0.05143` | `+46.26%` |
| `6_Full_AVID_R2D2_WB_PostBP_OSD_19cyc` | `avid_r2d2` | `2` | `26 / 31` | `79` | `0.05143` | `+46.26%` |

---

## 6. Quick Start, Verification Suites & Reproducibility

```bash
# 1. Install toolchains (Ubuntu 24.04 / Google Cloud Shell)
bash scripts/setup_environment.sh

# 2. Train R2D2 controller & export C++/ONNX checkpoints (optional; pre-exported in checkpoints/)
python3 -m training.evaluation.evaluate_policy

# 3. Build Verilator RTL, C++ co-simulation bridge, and ARM Cortex-A9 / ARM64 / native guests
bash scripts/build_all.sh

# 4. Run all 36 executable verification & simulation tests (auto-builds binaries if needed)
pytest -v tests/test_training.py tests/test_checkpoint_export.py tests/test_cpp_python_parity.py tests/test_decoder_action_effect.py
pytest -q simulation/tests/ tests/

# 5. Validate architecture contracts, repository structure & checkpoint provenance
python3 tools/validate_architecture.py
python3 tools/validate_repository.py
python3 tools/verify_provenance.py
python3 tools/verify_requirements.py
```
