# AVID: Adaptive Variational Inference Decoder for Fault-Tolerant Quantum Error Correction

**AVID** is a hardware/software co-designed real-time quantum error correction (QEC) decoding architecture for the rotated $d=3$ CSS surface code (`[[9, 1, 3]]`). It pairs a deterministic `250 MHz` SystemVerilog Min-Sum Belief Propagation + OSD-0 fast path (`firmware/rtl/`) with a causal two-pass 16-state Joint-DP L-BFGS estimator and 64-unit `int8` LSTM Weighted-Bellman R2D2 controller (`software/src/`).

---

## 1. Declared Hardware Targets (V1 Emulation vs. V2 Physical Scope)

To avoid conflating emulation targets with future board deployments, AVID declares explicit platform identities (see [`architecture/hardware/target_platforms.yaml`](architecture/hardware/target_platforms.yaml)):

| Scope | Target Platform | Processing System (PS) | Programmable Logic (PL) | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **V1 Primary Emulation Target** | **Xilinx Zynq-7020 SoC (`xc7z020clg484-1`)** | Dual-core **ARM Cortex-A9** (`ARMv7-A`, `arm-linux-gnueabihf-g++ -mcpu=cortex-a9` + `qemu-arm -cpu cortex-a9`) | Artix-7 FPGA fabric @ `250 MHz` (`4.0 ns`) simulated via native **`Verilator 5.020`** (`verilator_c++_rtl`) | **Implemented & Verified** (`avid_v1_integration_20261106_172251_s2026`) |
| **V1 Secondary Compatibility Build** | **ARM64 Linux Guest (`aarch64`)** | `aarch64-linux-gnu-g++` + `qemu-aarch64-static` | Same Verilator RTL bridge | **Implemented & Verified** (`avid_v1_full_benchmark_20261105_223251_s777`) |
| **V2 Intended Physical Target** | **Xilinx Zynq UltraScale+ MPSoC (`ZCU104` `xczu7ev` / `Kria K26` `xck26`)** | Quad-core **ARM Cortex-A53** (`ARMv8-A` @ `1.2 GHz`) | UltraScale+ FPGA fabric @ `250 MHz` | **Planned V2 Physical Board Scope** |

---

## 2. Three-Tier Causal Timing & Latency Contract

All timing claims are governed by [`architecture/contracts/system_timing.yaml`](architecture/contracts/system_timing.yaml) and [`architecture/control/fast_slow_partition.md`](architecture/control/fast_slow_partition.md):

### 2.1 Tier 1 — Fast-Path Hardware Decoder Latency (`250 MHz`, `4.0 ns` clock)
The RTL decoder supports three configurable OSD modes via MMIO register `0x058` (`cfg_osd_mode`) across $K \in \{1 \dots 5\}$ BP iterations (`0x04C`), with `3` AXI4-Stream ingress cycles and `2` egress cycles (`5` cycles stream overhead):

| Iteration Count $K$ | OSD Mode (`0x058`) | BP Core Cycles $L_{\mathrm{core}}(K)$ | BP Core Latency (`ns`) | Total AXI4-Stream Cycles $L_{\mathrm{AXI}}(K)$ | Total AXI4-Stream Latency (`ns`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| $K = 1$ | Mode `0` / `1` | `7` (`2 + 5K`) | `28.0 ns` | `12` (`7 + 5K`) | `48.0 ns` |
| $K = 2$ | Mode `0` / `1` | `12` (`2 + 5K`) | `48.0 ns` | `17` (`7 + 5K`) | `68.0 ns` |
| **$K = 3$ (Default)** | **Mode `0` (Pure 9-LLR BP) / Mode `1` (Inline OSD-0)** | **`17` (`2 + 5K`)** | **`68.0 ns`** | **`22` (`7 + 5K`)** | **`88.0 ns`** |
| **$K = 3$** | **Mode `2` (Post-BP OSD-0 Stage)** | **`19` (`4 + 5K`)** | **`76.0 ns`** | **`24` (`9 + 5K`)** | **`96.0 ns`** |
| $K = 4$ | Mode `0` / `1` | `22` (`2 + 5K`) | `88.0 ns` | `27` (`7 + 5K`) | `108.0 ns` |
| $K = 5$ | Mode `0` / `1` | `27` (`2 + 5K`) | `108.0 ns` | `32` (`7 + 5K`) | `128.0 ns` |

### 2.2 Tier 2 — Fast 16-Round Sliding-Subwindow Burst Detector
Implemented in [`software/src/control/causal_scheduler.cpp`](software/src/control/causal_scheduler.cpp) (`avid_fast_detector_step`) and executed every round in [`software/src/inference/avid_daemon.cpp`](software/src/inference/avid_daemon.cpp):
- Monitors a **16-round sliding subwindow** (`16 µs` @ `1 MHz`) with a syndrome-density threshold of **`0.35`** and a **`4`-round debounce**, achieving a worst-case detection latency of **`20` rounds (`20 µs`)** inside a `60 µs` burst.

### 2.3 Tier 3 — Slow-Loop MAP Estimation & R2D2-WB Weight Adaptation
- **V1 Emulation Window**: `window_rounds = 256` (`256 µs` syndrome window) for accelerated multi-epoch co-simulation (`configs/system/v1_integration.yaml`).
- **V2 Production Period**: `10,000` rounds (`10.0 ms` period @ `1 MHz`). Under `qemu-arm -cpu cortex-a9`, the two-pass 16-state Joint-DP L-BFGS + 64-unit LSTM R2D2-WB update completes in **`6.02 ms` mean** (`60.17%` of the `10 ms` budget).

---

## 3. Structural Identifiability vs. Prior-Regularized MAP Estimation

As proven in [`mathematical_model/proofs/identifiability_proof.md`](mathematical_model/proofs/identifiability_proof.md) and [`architecture/noise_model/identifiability_report.md`](architecture/noise_model/identifiability_report.md):
1. **Data-Identifiable Syndrome Orbits (`rank(J) = 9`)**: Out of the 21 physical fault parameters per basis (9 single-qubit + 12 adjacent correlated pairs), single-window syndrome statistics independently identify **9 non-zero syndrome signature orbits** (`{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x8, 0xA, 0xC}`). The remaining **12 directions** lie in the gauge/collision nullspace ($\dim\ker(J) = 12$).
2. **Prior-Regularized MAP Hessian (`rank(H_MAP) = 21/21`)**: The Gaussian prior $rac{\lambda}{2}\|	heta - 	heta_{\mathrm{prior}}\|_2^2$ makes the MAP Hessian strictly positive-definite (`rank = 21/21`), selecting a unique canonical point within each gauge coset.
3. **Metric Interpretation**: `orbit_9class_rmse` (`0.0081–0.0148`) measures data-driven syndrome-orbit recovery; `param_21_rmse` (`0.0046–0.0086`) reflects prior-anchored selection along the 12 null directions and does **not** imply prior-free independent observability of all 21 physical rates.

---

## 4. Verified Experimental Evidence (Separated by Experiment Scope)

### Experiment Set A — V1 10-Scenario Dual-Emulator Integration Suite
- **Run ID**: `avid_v1_integration_20261106_172251_s2026` ([`reports/v1/integration/cloudshell_v1_integration_10scenarios_report.json`](reports/v1/integration/cloudshell_v1_integration_10scenarios_report.json))
- **Environment**: Native `Verilator 5.020` (`verilator_c++_rtl`) + `qemu-arm -cpu cortex-a9`, `seed = 2026`, `1,536` rounds/scenario (`15,360` total rounds), `window_rounds = 256`.

| Scenario (`1,536` rounds) | Static B0 LER | AVID V1 RTL LER | 95% Wilson CI | Whole-Run LER Reduction | Burst-Window LER Reduction | Marginal NLL $	o$ Joint-DP NLL | 9-Orbit RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stationary_noise` | `0.00456` | `0.00456` | `[0.00221, 0.00938]` | `0.00%` | `0.00%` | `0.5219 -> 0.4678` | `0.0081` |
| `slowly_drifting_noise` | `0.05208` | `0.02604` | `[0.01918, 0.03527]` | **`-50.00%`** | **`-50.00%`** | `1.0860 -> 0.9872` | `0.0103` |
| `correlated_noise` | `0.05339` | `0.03646` | `[0.02819, 0.04704]` | **`-31.71%`** | **`-31.71%`** | `0.8458 -> 0.7511` | `0.0118` |
| `burst_disturbance` | `0.04688` | `0.02669` | `[0.01973, 0.03601]` | **`-43.06%`** | **`-56.25%`** | `0.4422 -> 0.4135` | `0.0142` |
| `delayed_weight_commit` | `0.07357` | `0.03646` | `[0.02819, 0.04704]` | **`-50.44%`** | **`-50.44%`** | `0.8663 -> 0.8230` | `0.0103` |
| `reset_recovery` | `0.07031` | `0.04036` | `[0.03162, 0.05140]` | **`-42.59%`** | **`-42.59%`** | `1.1322 -> 0.9963` | `0.0148` |
| `busy_decoder` | `0.07292` | `0.04167` | `[0.03276, 0.05286]` | **`-42.86%`** | **`-42.86%`** | `0.9564 -> 0.8760` | `0.0087` |
| `variable_k_adaptation` | `0.06771` | `0.03906` | `[0.03047, 0.04995]` | **`-42.31%`** | **`-42.31%`** | `0.8753 -> 0.8081` | `0.0100` |
| `fallback_safety_trigger` | `0.07031` | `0.05143` | `[0.04147, 0.06363]` | **`-26.85%`** | **`-26.85%`** | `0.9854 -> 0.8966` | `0.0091` |
| `short_burst_60us_vs_10ms` | `0.05404` | `0.03190` | `[0.02422, 0.04191]` | **`-40.96%`** | **`-45.45%`** | `0.4678 -> 0.4137` | `0.0102` |

### Experiment Set B — V1 6-Way C++ Runtime & RTL Decoder Ablation
- **Report**: [`reports/v0/closure/v0_closure_40_40_report.json`](reports/v0/closure/v0_closure_40_40_report.json) (`correlated_noise`, `1,536` rounds, `seed = 2026`):
  1. `1_Static_B0` (`controller_mode=static`, `osd_mode=1`, `17` core cyc): `LER = 0.05339` (`0.0%`)
  2. `2_Marginal_LBFGS_B1` (`controller_mode=marginal_lbfgs`, `osd_mode=1`, `17` core cyc): `LER = 0.05339` (`0.0%`)
  3. `3_Joint_DP_Pure_9Marginal_BP_17cyc` (`controller_mode=joint_dp_map`, `osd_mode=0`, `17` core cyc): `LER = 0.12695` (`+137.8%` degradation when correlated pairs are mapped naively to 9 single-qubit marginals without OSD-0)
  4. `4_Joint_DP_Pair_Projected_17cyc` (`controller_mode=joint_dp_proj`, `osd_mode=1`, `17` core cyc): `LER = 0.04492` (`-15.85%`)
  5. `5_Full_AVID_R2D2_WB_Inline_OSD_17cyc` (`controller_mode=avid_r2d2`, `osd_mode=1`, `17` core cyc): `LER = 0.03581` (**`-32.93%`**)
  6. `6_Full_AVID_R2D2_WB_PostBP_OSD_19cyc` (`controller_mode=avid_r2d2`, `osd_mode=2`, `19` core cyc): `LER = 0.03581` (**`-32.93%`**)

### Experiment Set C — V1 7-Scenario Long-Run Native Verilator Benchmark
- **Run ID**: `avid_v1_full_benchmark_20261105_223251_s777` ([`reports/v1/performance/cloudshell_full_benchmark_57344_rounds_report.json`](reports/v1/performance/cloudshell_full_benchmark_57344_rounds_report.json))
- **Scope**: 7 scenarios $	imes$ `8,192` rounds = `57,344` rounds under native `Verilator 5.020` + `qemu-aarch64-static`.

### Experiment Set D — V0 Offline Reference-Model Benchmarks (`Benchmark-Eval/Baseline_V0/`)
- **Report**: [`reports/v0/benchmark_summary.json`](reports/v0/benchmark_summary.json) & [`checkpoints/baseline_vs_avid_benchmark.json`](checkpoints/baseline_vs_avid_benchmark.json)
- **Scope**: Python/PyTorch reference model evaluations (`BM1` 15-seed significance, `BM2` 8-variant `A0..A7` offline World-Model/LoRA ablation, `BM3` $d=3 	o d=5$ simulation scaling, `BM4` window complexity, `BM5` analytical/emulated timing models, and the 250,000-round `R1–R5` model benchmark).

---

## 5. Quick Start & Reproducibility

```bash
# 1. Install toolchains (Ubuntu 24.04 / Google Cloud Shell)
bash scripts/setup_environment.sh

# 2. Validate architecture contracts, repository structure & provenance
python3 tools/validate_architecture.py
python3 tools/validate_repository.py
python3 tools/verify_provenance.py
python3 tools/verify_requirements.py

# 3. Build Verilator RTL, C++ co-simulation bridge, and ARM Cortex-A9 / ARM64 guests
bash scripts/build_all.sh

# 4. Run the V1 10-scenario dual-emulator integration suite
bash scripts/run_all.sh configs/system/v1_integration.yaml

# 5. Run the 40/40 V0 closure & 6-way runtime ablation verifier
python3 tools/verify_v0_closure_40.py
```
