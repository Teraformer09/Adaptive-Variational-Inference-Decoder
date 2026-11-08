# AVID: Adaptive Variational Inference Decoder for Fault-Tolerant Quantum Error Correction

**AVID** is a hardware/software co-designed real-time quantum error correction (QEC) decoding architecture for the rotated $d=3$ CSS surface code (`[[9, 1, 3]]`). It pairs a deterministic `250 MHz` SystemVerilog Min-Sum Belief Propagation + OSD-0 fast path (`firmware/rtl/`) with a causal two-pass 16-state Joint-DP L-BFGS estimator (`software/src/`) and a trained 64-unit Dueling LSTM R2D2 controller (`training/`, `deployment/cpp/`, `checkpoints/r2d2_v1_trained.bin`).

---

## 1. Declared Hardware Targets (V1 Emulation vs. V2 Physical Scope)

To avoid conflating pre-silicon emulation targets with future physical board deployments, AVID declares explicit platform identities (see [`architecture/hardware/target_platforms.yaml`](architecture/hardware/target_platforms.yaml)):

| Scope | Target Platform | Processing System (PS) | Programmable Logic (PL) | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **V1 Primary Emulation Target** | **Xilinx Zynq-7020 SoC (`xc7z020clg484-1`)** | Dual-core **ARM Cortex-A9** (`ARMv7-A`, `arm-linux-gnueabihf-g++ -mcpu=cortex-a9` + `qemu-arm -cpu cortex-a9`) | Artix-7 FPGA fabric @ `250 MHz` (`4.0 ns`) simulated via native **`Verilator 5.020`** (`verilator_c++_rtl`) | **Implemented & Verified** (`avid_v1_integration_20261106_172251_s2026`) |
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
1. **Observation Interface**: Dual-basis syndrome histograms `(hist_x[16], hist_z[16])` and fast burst detector state.
2. **Estimation Interface**: 16-state Joint-DP MAP solver producing 21 physical parameters $\hat{\theta}_Z, \hat{\theta}_X$ and general 12-pair syndrome-alias projection across all nearest-neighbour edges in `NM_CROSSTALK_PAIRS` (zero hardcoded pair indices).
3. **Control Interface**: Trained 64-unit Dueling LSTM R2D2 network (`training/models/r2d2_network.py`, exported to [`checkpoints/r2d2_v1_trained.bin`](checkpoints/r2d2_v1_trained.bin) and [`checkpoints/r2d2_v1_trained.onnx`](checkpoints/r2d2_v1_trained.onnx)) loaded by [`deployment/cpp/model_loader.cpp`](deployment/cpp/model_loader.cpp) and [`deployment/cpp/r2d2_runtime.cpp`](deployment/cpp/r2d2_runtime.cpp) with verified `< 1e-12` Python-to-C++ numerical parity.
4. **Action-to-Decoder Interface**: Selects among 6 discrete actions $a_0 \dots a_5$ (`STATIC_PRIOR_K3`, `ADAPTIVE_PAIR_K5`, `AGGRESSIVE_PAIR_K7`, `DAMPED_PRIOR_K4`, `FAST_RECALIBRATE_K7`, `CONSERVATIVE_FALLBACK_K3`) plus 9 bounded per-qubit gain residuals $\Delta g \in [-0.30, +0.30]^9$, writing $K \in \{3, 4, 5, 7\}$ to hardware register `0x054` and shadow Q7.8 weights to `0x000..0x048`.
5. **Safety & Fallback Interface**: 4-state fallback FSM (`software/src/control/fallback_fsm.cpp`) reverting to safe static priors upon unphysical parameter bounds, deadline overruns, or `CONSERVATIVE_FALLBACK_K3`.
6. **Evaluation & Provenance Interface**: Cryptographic SHA-256 verification of checkpoints and RTL build provenance (`tools/verify_provenance.py`), with pre-remediation historical reports preserved in [`reports/archive/pre_remediation_b001b8d/SUPERSEDED_PROVENANCE.md`](reports/archive/pre_remediation_b001b8d/SUPERSEDED_PROVENANCE.md).

---

## 4. Structural Identifiability vs. Prior-Regularized MAP Estimation

As proven in [`mathematical_model/proofs/identifiability_proof.md`](mathematical_model/proofs/identifiability_proof.md) and [`architecture/noise_model/identifiability_report.md`](architecture/noise_model/identifiability_report.md):
1. **Data-Identifiable Syndrome Orbits (`rank(J) = 9`)**: Out of the 21 physical fault parameters per basis (9 single-qubit + 12 adjacent correlated pairs), single-window syndrome statistics independently identify **9 non-zero syndrome signature orbits** (`{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x8, 0xA, 0xC}`). The remaining **12 directions** lie in the gauge/collision nullspace ($\dim\ker(J) = 12$).
2. **Prior-Regularized MAP Hessian (`rank(H_MAP) = 21/21`)**: The Gaussian prior $\frac{\lambda}{2}\|\theta - \theta_{\mathrm{prior}}\|_2^2$ makes the MAP Hessian strictly positive-definite (`rank = 21/21`), selecting a unique canonical point within each gauge coset.
3. **Metric Interpretation**: `orbit_9class_rmse` measures data-driven syndrome-orbit recovery; `param_21_rmse` reflects prior-anchored selection along the 12 null directions and does **not** imply prior-free independent observability of all 21 physical rates.

---

## 5. Quick Start, Verification Suites & Reproducibility

```bash
# 1. Install toolchains (Ubuntu 24.04 / Google Cloud Shell)
bash scripts/setup_environment.sh

# 2. Train R2D2 controller & export C++/ONNX checkpoints (optional; pre-exported in checkpoints/)
python3 -m training.evaluation.evaluate_policy

# 3. Build Verilator RTL, C++ co-simulation bridge, and ARM Cortex-A9 / ARM64 / native guests
bash scripts/build_all.sh

# 4. Run the 4 mandatory verification suites + simulation & acceptance tests
pytest -v tests/test_training.py tests/test_checkpoint_export.py tests/test_cpp_python_parity.py tests/test_decoder_action_effect.py
pytest -q simulation/tests/ tests/

# 5. Validate architecture contracts, repository structure & checkpoint provenance
python3 tools/validate_architecture.py
python3 tools/validate_repository.py
python3 tools/verify_provenance.py
python3 tools/verify_requirements.py
```
