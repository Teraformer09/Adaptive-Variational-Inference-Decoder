# AVID: Adaptive Variational Inference Decoder (`AVID_EPJ_Patches`)
### Real-Time Variational Inference & Weighted-Bellman R2D2 World Modeling of Correlated Quantum Noise for Hardware-Accelerated Surface Codes

**Target Hardware:** Xilinx Zynq-7020 / Kria K26 SOM (`250 MHz` PL Fast Loop + `ARM Cortex-A53` Slow Loop)  
**Hugging Face Models, Checkpoints & 250k-Round Datasets (`73 files`):** [**`Terrano09/AVID-EPJ-Baseline-Models-and-Data`**](https://huggingface.co/Terrano09/AVID-EPJ-Baseline-Models-and-Data)  
**Hugging Face Peer-Reviewed 5-Benchmark Suite (`30 files`):** [**`Terrano09/AVID-EPJ-Baseline-Models-and-Data/tree/main/Benchmark-Eval/Baseline_V0`**](https://huggingface.co/Terrano09/AVID-EPJ-Baseline-Models-and-Data/tree/main/Benchmark-Eval/Baseline_V0)  
**AVID V1 Two-Emulator Architecture & System Flow:** [**`docs/AVID_V1_Two_Emulator_Architecture_and_System_Flow.md`**](docs/AVID_V1_Two_Emulator_Architecture_and_System_Flow.md)  
**Baseline_V0 Establishment, Q01–Q21 Closure Audit & V1/V2 Roadmap:** [**`docs/AVID_Baseline_V0_Establishment_and_V1_V2_Roadmap.md`**](docs/AVID_Baseline_V0_Establishment_and_V1_V2_Roadmap.md)  
**Mathematical Modeling Suite (`00`–`09`):** [**`docs/`**](docs/00_Mathematical_Modeling_Index.md) & [**`mathematical_model/`**](mathematical_model/00_Mathematical_Modeling_Index.md)

---

## 1. Short Description of the Work

Standard quantum error correction (QEC) surface-code decoders (MWPM, static Min-Sum Belief Propagation) assume a **stationary, uncorrelated noise channel** calibrated offline before execution. Real superconducting transmon processors experience non-stationary flux drift, Two-Level System (TLS) telegraph defects, and dispersive $ZZ$ crosstalk during stabilizer extraction.

**AVID (Adaptive Variational Inference Decoder)** decouples real-time surface-code decoding into a **Dual-Loop Hardware-Software Architecture**:
1. **Fast Loop (`250 MHz` FPGA Programmable Logic / Verilator RTL)**: Executes a deterministic **17-cycle (`68.0 ns` at $K=3$)** Normalized Q7.8 Min-Sum Belief Propagation + 1-cycle combinational Pair-Aware OSD-0 syndrome decoder using an active register bank of Q7.8 log-likelihood ratio (LLR) priors.
2. **Slow Loop (`ARM Cortex-A53` / `qemu-aarch64` Linux Daemon)**: Aggregates windows of $W$ syndrome packets, tracks non-stationary single-qubit and correlated $Z_u Z_v$ crosstalk probabilities via **Exact $2^m$-State Dynamic Programming MAP Estimation** and **Weighted-Bellman R2D2 (`R2D2-Graph-RSSM`) World Modeling**, and atomically commits updated Q7.8 LLR weights into a shadow register bank via AXI4-Lite without stalling the fast path.

---

## 2. Core Mathematical Formulation

### 2.1 Rotated CSS Surface Code Commutation ($d=3$, `[[9, 1, 3]]`)
The parity-check matrices $H_X, H_Z \in \{0,1\}^{4 \times 9}$ satisfy exact binary orthogonality and rank conditions:

$$
H_X H_Z^T \equiv \mathbf{0} \pmod 2, \qquad \text{rank}_{\mathbb{F}_2}(H_X) = \text{rank}_{\mathbb{F}_2}(H_Z) = 4, \qquad k = n - \text{rank}(H_X) - \text{rank}(H_Z) = 1
$$

### 2.2 Exact $2^m$-State Dynamic Programming Syndrome Likelihood & Coset Gradient
Given $M = 21$ independent physical fault mechanisms $e_k \in \mathcal{E}$ (9 single-qubit faults + 12 adjacent $Z_u Z_v$ crosstalk edges) with binary detector signatures $g_{e_k} \in \{0,1\}^4$ and Bernoulli probabilities $p_{e_k} = \sigma(\theta_{e_k})$, the exact joint distribution over all $2^4 = 16$ syndromes $s \in \{0,1\}^4$ is computed via recursive XOR convolution:

$$
P^{(k)}(s) = (1 - p_{e_k}) P^{(k-1)}(s) + p_{e_k} P^{(k-1)}(s \oplus g_{e_k}), \qquad k = 1, \dots, M
$$

with exact single-pass coset-inversion gradient:

$$
\frac{\partial P_{\theta}(s)}{\partial p_e} = \frac{P_{\theta}(s \oplus g_e) - P_{\theta}(s)}{1 - 2p_e}
$$

### 2.3 Pair-Aware Logical-Triangle LLR Projection & Q7.8 Quantization
Single-qubit marginal probabilities $\tilde{p}_i$ are folded via exact odd-parity XOR marginalization and projected across correlated logical-triangle pairs $(u, v)$ before signed 16-bit Q7.8 hardware quantization:

$$
\tilde{p}_i = \frac{1 - \prod_{e \in \mathcal{E}_i} (1 - 2p_e)}{2}, \qquad \gamma_i = \ln\left(\frac{1 - \tilde{p}_i}{\tilde{p}_i}\right), \qquad W_i^{\text{Q7.8}} = \text{clip}\left(\text{round}(256 \cdot \gamma_i), -32768, 32767\right)
$$

### 2.4 Hardware Normalized Min-Sum BP & Latency Contract
At each BP iteration $t \in \{1, \dots, K\}$ with hardware normalization factor $\alpha = \frac{3}{4} = 0.5 + 0.25$ (implemented with right-shifts and adders, zero DSP multipliers):

$$
\mu_{c \to v}^{(t)} = \left(\prod_{v' \in \mathcal{N}(c) \setminus \{v\}} \text{sgn}(m_{v' \to c}^{(t-1)})\right) \cdot \left\lfloor \frac{3}{4} \min_{v' \in \mathcal{N}(c) \setminus \{v\}} \left|m_{v' \to c}^{(t-1)}\right| \right\rfloor, \qquad \Lambda_v^{(K)} = W_v^{\text{Q7.8}} + \sum_{c \in \mathcal{N}(v)} \mu_{c \to v}^{(K)}
$$

The deterministic hardware cycle count for $K$ iterations at $f_{\text{clk}} = 250\text{ MHz}$ ($T_{\text{clk}} = 4.0\text{ ns}$) is:

$$
L_{\text{BP}}(K) = 2 + 5K \text{ cycles} \implies L_{\text{BP}}(3) = 17 \text{ cycles} = 68.0\text{ ns}
$$

---

## 3. Model Performance Details (Shared-Input 250,000-Round Benchmark)

Evaluated across `250,000` fast-loop syndrome rounds (`50,000` rounds/regime) and `600` non-stationary R2D2 replay trajectories (`2,457,600` fast-loop rounds) using the bit-exact 17-cycle Q7.8 hardware BP execution engine ([`checkpoints/baseline_vs_avid_benchmark.json`](checkpoints/baseline_vs_avid_benchmark.json)):

| Noise Regime | `B0` Static BP | `B1` Marginal L-BFGS | `B2` Observable PID | `B3` Covariance | **`AVID_Joint_DP_MAP`** | **`AVID_Weighted_Bellman_R2D2`** | `B4` Oracle BP | Key Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `R1_stationary_baseline` | `0.0486` | `0.0486` | `0.0486` | `0.0486` | **`0.0486`** | **`0.0486`** | `0.0486` | Matches exact Oracle floor |
| `R2_tls_hotspot_q4_q5` | `0.0564` | `0.0564` | `0.0564` | `0.0564` | `0.0588` | **`0.0564`** | `0.0564` | Matches exact Oracle floor |
| `R3_zz_crosstalk_burst` | `0.1984` | `0.1984` | `0.1984` | `0.1984` | **`0.1143`** | **`0.1143`** | `0.1143` | **42.4% LER reduction (matches Oracle)** |
| `R4_boundary_drift` | `0.0534` | `0.0534` | `0.0534` | `0.0534` | **`0.0534`** | **`0.0534`** | `0.0534` | Matches exact Oracle floor |
| `R5_combined_stress` | `0.0970` | `0.0970` | `0.0970` | `0.0970` | `0.0970` | **`0.0544`** | `0.0517` | **43.9% LER reduction (`0.0970 -> 0.0544`)** |

---

## 4. Emulator Performance Details (57,344-Round Long Run in Google Cloud Shell)

Measured in Google Cloud Shell using the **AVID V1 Two-Emulator Architecture** (`configs/v1_full_benchmark.yaml`, archived in [`reports/cloudshell_full_benchmark_57344_rounds_report.json`](reports/cloudshell_full_benchmark_57344_rounds_report.json)):

| Metric | Measured Result in Cloud Shell |
| :--- | :--- |
| **Emulator 1 (Software)** | Cross-compiled `avid_daemon_aarch64` (`aarch64-linux-gnu-g++ 13.3.0`) running on `qemu-aarch64 8.2.2` |
| **Emulator 2 (Hardware RTL)** | Native `verilator_c++_rtl` (`Verilator 5.020`) compiling [`aa_qec_top.sv`](firmware/rtl/top/aa_qec_top.sv), [`bp_engine.sv`](firmware/rtl/decoder/bp_engine.sv), [`check_node_minsum.sv`](firmware/rtl/decoder/check_node_minsum.sv), [`weight_map_reg.sv`](firmware/rtl/config/weight_map_reg.sv) |
| **Fast-Path BP Core Latency ($K = 3$)** | **`17.0 cycles` (`min=17, max=17`) = `68.0 ns` (`0.068 µs`) @ `250 MHz`** (`22 cycles` end-to-end with AXI4-Stream RX FIFO + TX) |
| **Measured $K = 1..5$ Cycle Matrix** | $K = 1$: `7 cyc` (`28 ns`), $K = 2$: `12 cyc` (`48 ns`), $K = 3$: `17 cyc` (`68 ns`), $K = 4$: `22 cyc` (`88 ns`), $K = 5$: `27 cyc` (`108 ns`) |
| **Total Syndrome Rounds Decoded in RTL** | **`57,344 rounds`** (`8,192` rounds $\times$ `7` scenarios, 100% streamed through Bridge $\leftrightarrow$ Verilator RTL) |
| **Total Bridge $\leftrightarrow$ RTL Transactions** | **`61,685 transactions`** (`57,344` syndrome transfers, `4,123` AXI-Lite MMIO writes, `217` MMIO reads, `1` reset, `0` rejected) |
| **Total Simulated FPGA Clock Cycles** | **`1,270,428 clock cycles`** (`181,488` cycles / `725.95 µs` simulated hardware time per scenario) |
| **Host Wall-Clock Runtime** | **`33.548 seconds`** (`overall_status: "PASS"`) |

### Scenario-by-Scenario Dual-Emulator Verification Summary (`avid_v1_full_benchmark_20261106_051922_s777`)

| Scenario | Syndrome Rounds | Active Bank Commits | Static `B0` LER | **AVID V1 RTL LER** | **LER Reduction** | Fast-Path BP Core Latency | Simulated FPGA Cycles | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stationary_noise` | `8,192` | `33` | `0.00745` | **`0.00745`** | `0.00%` (at floor) | `17.0 cyc` (`68.0 ns`) | `181,568` | **PASS** |
| `slowly_drifting_noise` | `8,192` | `33` | `0.04541` | **`0.02832`** | **`-37.63%`** | `17.0 cyc` (`68.0 ns`) | `181,568` | **PASS** |
| `correlated_noise` | `8,192` | `33` | `0.07654` | **`0.03210`** | **`-58.05%`** | `17.0 cyc` (`68.0 ns`) | `181,568` | **PASS** |
| `burst_disturbance` | `8,192` | `33` | `0.04492` | **`0.01782`** | **`-60.33%`** | `17.0 cyc` (`68.0 ns`) | `181,568` | **PASS** |
| `delayed_weight_commit` | `8,192` | `33` | `0.07300` | **`0.03320`** | **`-54.52%`** | `17.0 cyc` (`68.0 ns`) | `181,568` | **PASS** |
| `reset_recovery` | `8,192` | `31` (2 reset drops) | `0.07019` | **`0.03455`** | **`-50.78%`** | `17.0 cyc` (`68.0 ns`) | `181,580` | **PASS** |
| `busy_decoder` | `8,192` | `33` (0 busy overwrites) | `0.06958` | **`0.03198`** | **`-54.04%`** | `17.0 cyc` (`68.0 ns`) | `181,568` | **PASS** |

---

## 5. End-to-End Repository Structure

```text
AVID_EPJ_Patches/
├── emulator/                              # AVID V1 Dual-Emulator Infrastructure (QEMU + Verilator + Bridge)
│   ├── qemu/                              # Emulator 1: ARM64 Linux QEMU Cross-Build & Runner
│   ├── verilator/                         # Emulator 2: Verilator SystemVerilog RTL Harness & Testbenches
│   └── bridge/                            # C++ Co-Simulation Bridge (UNIX Socket IPC + Transaction Logger)
├── configs/                               # Experiment & Emulator Configurations
│   ├── qemu.yaml                          # Emulator 1 (QEMU ARM64) configuration
│   ├── verilator.yaml                     # Emulator 2 (Verilator 250 MHz RTL) configuration
│   ├── v1_smoke.yaml                      # Fast 4-scenario smoke test (4,096 rounds)
│   ├── v1_integration.yaml                # 7-scenario integration benchmark (10,752 rounds)
│   └── v1_full_benchmark.yaml             # 7-scenario full benchmark (57,344 rounds)
├── scripts/                               # Cloud Shell Setup, Build, Execution & Artifact Aggregation
│   ├── setup_cloudshell.sh                # Installs qemu-user, g++-aarch64-linux-gnu, verilator
│   ├── build_all.sh                       # Builds ARM64 daemon, Verilator RTL binaries, and C++ Bridge
│   ├── run_all.sh                         # Runs M1..M5 end-to-end dual-emulator benchmark
│   ├── run_regression.sh                  # Multi-seed regression runner
│   └── collect_artifacts.py               # Generates reports/latest_report.json & summary tables
├── firmware/rtl/                          # Synthesizable SystemVerilog RTL (250 MHz PL Fast Path)
│   ├── top/aa_qec_top.sv                  # Top-level QEC decoder wrapper
│   ├── decoder/{bp_engine.sv, check_node_minsum.sv, axis_correction_tx.sv}
│   ├── config/weight_map_reg.sv           # Atomic double-buffered Q7.8 weight register bank
│   └── ingress/{axis_syndrome_rx.sv, axis_ingress_tap.sv}
├── software/                              # C/C++ ARM PS Slow-Loop Inference Daemon & Drivers
├── simulation/                            # Python Reference Models, Baselines (B0–B4) & 15 Unit Tests
├── architecture/                          # Frozen Contracts & Layer 1–7 Specifications
├── Benchmark-Eval/Baseline_V0/            # 5 Peer-Reviewed Benchmarks (BM1..BM5) & Master Summary
├── checkpoints/                           # Trained Weighted-Bellman R2D2, Transformer, LoRA (.pt) & Q7.8 Banks
├── docs/ & mathematical_model/            # Complete Numbered Mathematical Modeling Suite (00..09)
└── reports/                               # Archived Cloud Shell Dual-Emulator Benchmark Reports
```

---

## 6. Reproducibility Commands (Google Cloud Shell)

### Build & Run Full 57,344-Round Dual-Emulator Benchmark (`QEMU ARM64` + `Verilator RTL`)
```bash
bash scripts/setup_cloudshell.sh
bash scripts/build_all.sh
bash scripts/run_all.sh configs/v1_full_benchmark.yaml
cat reports/latest_report.json
```

### Run Smoke Test & Multi-Seed Regression Suite
```bash
bash scripts/run_all.sh configs/v1_smoke.yaml
bash scripts/run_regression.sh
```

### Run Architecture Gate Validator (Python + C++ Suites)
```bash
PYTHONPATH=. python3 tools/validate_architecture.py
```
