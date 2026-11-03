# AVID: Adaptive Variational Inference Decoder (`AVID_EPJ_Patches`)
### Real-Time Variational Inference & Weighted-Bellman R2D2 World Modeling of Correlated Quantum Noise for Hardware-Accelerated Surface Codes

**Target Hardware:** Xilinx Zynq-7020 / Kria K26 SOM (`250 MHz` PL Fast Loop + `ARM Cortex-A53` Slow Loop)  
**Hugging Face Models, Checkpoints & 250k-Round Datasets:** [**`Terrano09/AVID-EPJ-Baseline-Models-and-Data`**](https://huggingface.co/Terrano09/AVID-EPJ-Baseline-Models-and-Data)  
**Mathematical Modeling Suite (`00`–`08`):** [**`docs/`**](docs/00_Mathematical_Modeling_Index.md) & [**`mathematical_model/`**](mathematical_model/00_Mathematical_Modeling_Index.md)

---

## 1. Overview & Architectural Repair Scope (`A01–A09` / `Q01–Q21`)

Standard surface code decoders (MWPM, static Min-Sum BP) assume a **stationary, uncorrelated noise model** calibrated at device boot. Real superconducting transmon processors experience non-stationary flux drift, Two-Level System (TLS) telegraph defects, and dispersive $ZZ$ crosstalk during CNOT/CZ stabilizer extraction.

This branch (`AVID_EPJ_Patches`) implements the complete end-to-end **AVID Layered Architecture and Repair Plan (`A01–A09` / `Q01–Q21`)**:
1. **Canonical Rotated CSS Surface Code Geometry (`A01 / Q02`, `A08 / Q17`)**:
   - Canonical rotated $d=3$ (`[[9, 1, 3]]`) and $d=5$ (`[[25, 1, 5]]`) CSS parity-check matrices ($H_X H_Z^\top \equiv \mathbf{0} \pmod 2$, $\operatorname{rank}_{\mathbb{F}_2}(H_X)=\operatorname{rank}_{\mathbb{F}_2}(H_Z)=4$, $k=1$, $d_X=d_Z=3$) and degree-padded hardware Tanner tables (`CHECK_ACTIVE_MASK = {0x3, 0xF, 0xF, 0x3}`).
2. **Exact $2^m$-State Dynamic Programming Joint Syndrome Likelihood & Analytic Coset Gradients (`A02 / Q03`)**:
   - Replaces the rank-deficient 4-check marginal Bernoulli objective with the exact $2^m=16$-state XOR-convolution dynamic program $P^{(k)}(s) = (1-p_{e_k})P^{(k-1)}(s) + p_{e_k}P^{(k-1)}(s \oplus g_{e_k})$ ($336$ multiply-adds) and exact single-pass coset-inversion gradient $\frac{\partial P_\theta(s)}{\partial p_e} = \frac{P_\theta(s \oplus g_e) - P_\theta(s)}{1 - 2p_e}$.
3. **Structural Identifiability & Pair-Aware Logical-Triangle LLR Projection (`A03 / Q04`)**:
   - Full Jacobian SVD and Fisher Information rank analysis ($\operatorname{rank}(\mathcal{I}) = 9$ unregularized vs. rank $4$ in the legacy marginal model; full rank $21$, $\kappa = 234.05$ under prior-tethered MAP), plus **Pair-Aware Logical-Triangle LLR Projection** that splits joint $Z_u Z_v$ crosstalk log-likelihoods across endpoints $(u, v)$ to prevent weight-1 logical aliasing in weight-3 logical triangles $\{u, v, k\}$.
4. **Weighted-Bellman R2D2 (`R2D2-Graph-RSSM`) + Spatio-Temporal Transformer + 5 Rank-4 LoRA Adapters**:
   - Non-vision bipartite stabilizer-data graph attention encoder (`13` nodes: `9` data qubits + `4` checks) coupled to a Recurrent State-Space Model trained with **burn-in warmup ($T_{\text{burn}}=6$)**, **Kapturowski invertible value rescaling $h(u)$**, **3-step bootstrapped Bellman targets**, and **Uncertainty-Weighted Bellman TD Loss ($\mathcal{L}_{\text{WB-R2D2}}$)**.
5. **17-Cycle (`68.0 ns` @ `250 MHz`) Hardware Q7.8 Min-Sum BP Engine & Atomic Double-Buffered Weight Registry (`A04 / Q01`, `A05 / Q11`, `A06 / Q06`, `A09 / Q18`)**:
   - Fused 17-cycle $L(K) = 2 + 5K$ state machine in [`bp_engine.sv`](firmware/rtl/bp_engine.sv), `bp_ready_in`-gated atomic shadow-to-active swap in [`weight_map_reg.sv`](firmware/rtl/weight_map_reg.sv), 64-bit little-endian AXI4-Stream syndrome packet contract (`syndrome_packet_v1.yaml`), 4-state DMA ring ownership FSM, and 4-state safety fallback FSM.

---

## 2. Shared-Input Logical Error Rate (LER) Benchmark (50,000 Rounds/Regime)

Evaluated across `250,000` fast-loop syndrome rounds (`50,000` rounds/regime) and `600` non-stationary R2D2 replay trajectories (`2,457,600` fast-loop rounds) using the bit-exact 17-cycle Q7.8 hardware BP execution engine ([`checkpoints/baseline_vs_avid_benchmark.json`](checkpoints/baseline_vs_avid_benchmark.json)):

| Noise Regime | `B0` Static BP | `B1` Marginal L-BFGS | `B2` Observable PID | `B3` Covariance | **`AVID_Joint_DP_MAP`** | **`AVID_Weighted_Bellman_R2D2`** | `B4` Oracle BP | Key Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `R1_stationary_baseline` | `0.0486` | `0.0486` | `0.0486` | `0.0486` | **`0.0486`** | **`0.0486`** | `0.0486` | Matches exact Oracle floor |
| `R2_tls_hotspot_q4_q5` | `0.0564` | `0.0564` | `0.0564` | `0.0564` | `0.0588` | **`0.0564`** | `0.0564` | Matches exact Oracle floor |
| `R3_zz_crosstalk_burst` | `0.1984` | `0.1984` | `0.1984` | `0.1984` | **`0.1143`** | **`0.1143`** | `0.1143` | **42.4% LER reduction (matches Oracle)** |
| `R4_boundary_drift` | `0.0534` | `0.0534` | `0.0534` | `0.0534` | **`0.0534`** | **`0.0534`** | `0.0534` | Matches exact Oracle floor |
| `R5_combined_stress` | `0.0970` | `0.0970` | `0.0970` | `0.0970` | `0.0970` | **`0.0544`** | `0.0517` | **43.9% LER reduction (`0.0970 -> 0.0544`)** |

---

## 3. End-to-End Repository Structure

```text
AVID_EPJ_Patches/
├── architecture/                          # Frozen Contracts & Layer 1–7 Specifications
│   ├── contracts/                         # system_timing, syndrome_packet_v1, code_layout_v1,
│   │                                      # noise_channel_v1, llr_numeric_contract, fault_policy
│   ├── 01_code_model/                     # Hx.json, Hz.json, coordinates.csv, logical_operators.json, tanner_tables.json
│   ├── 02_noise_and_identifiability/      # event_signatures.json, observation_model.md, identifiability_report.md
│   ├── 03_syndrome_transport/             # dma_topology.md, packet_layout.md, buffer_ownership.md
│   ├── 04_variational_estimator/          # likelihood_spec.md, parameter_schema.json, estimator_io.md
│   ├── 05_realtime_control/               # fast_slow_partition.md, deadline_contract.md, epoch_protocol.md, timing_profile_report.json
│   ├── 06_fpga_decoder/                   # cycle_contract.md, stream_interface.md, resource_envelope.md
│   └── 07_fault_management/               # fallback_fsm.md, fault_matrix.csv
├── docs/ & mathematical_model/            # Complete Numbered End-to-End Mathematical Modeling Suite (00..08)
│   ├── 00_Mathematical_Modeling_Index.md
│   ├── 01_CSS_Surface_Code_Algebra_and_Tanner_Geometry.md
│   ├── 02_Correlated_Fault_Channel_and_Observation_Algebra.md
│   ├── 03_Exact_Joint_DP_Likelihood_and_Coset_Gradients.md
│   ├── 04_Structural_Identifiability_Gauge_Orbits_and_Fisher_Rank.md
│   ├── 05_LBFGS_Optimization_XOR_Marginalization_and_Q78_Numerics.md
│   ├── 06_Hardware_MinSum_BP_Decoder_and_Cycle_Latency_Contract.md
│   ├── 07_Dual_Timescale_Causal_Control_DMA_and_Baselines_B0_B4.md
│   ├── 08_Weighted_Bellman_R2D2_and_World_Model_Math.md
│   ├── mathematical_modeling_complete.md
│   └── identifiability_proof.md
├── simulation/                            # Python Reference Models, Baselines (B0–B4) & 15 Unit Tests
│   ├── reference/
│   │   ├── code_model/css_code_builder.py
│   │   ├── noise_model/{joint_dp_likelihood.py, identifiability_svd.py}
│   │   ├── syndrome_stream/causal_stream_runner.py
│   │   └── baselines/{b0_static_bp.py .. b4_oracle_bp.py}
│   ├── stimulus/{packet_codec.py, noise_generator.py, stim_to_bin.py}
│   └── tests/                             # 6 test suites (15 tests) verifying Gate 0–Gate 8
├── software/                              # C/C++ ARM PS Slow-Loop Inference & Cycle-Accurate BP Twin
│   ├── src/
│   │   ├── ingress/{packet_validator.h/.cpp, dma_ingress_tap.h/.cpp}
│   │   ├── estimator/{joint_dp_estimator.h/.cpp, lbfgs_solver.h/.cpp, llr_quantizer.h/.cpp}
│   │   ├── control/{causal_scheduler.h/.cpp, stage_profiler.h/.cpp, fallback_fsm.h/.cpp}
│   │   ├── custom_engine/cycle_accurate_bp_engine.h/.cpp
│   │   ├── driver/{weight_reg_driver.h/.c, weight_bank_driver.h/.c, dma_manager.h/.c}
│   │   └── inference/{noise_model.h/.cpp, lbfgs_optimizer.h/.cpp, weight_injector.h/.cpp, avid_daemon.cpp}
│   └── tests/test_baseline_suite.cpp
├── firmware/                              # SystemVerilog RTL & Verification Testbenches (250 MHz PL)
│   ├── rtl/{aa_qec_top.sv, bp_engine.sv, check_node_minsum.sv, weight_map_reg.sv, axis_syndrome_rx.sv, axis_correction_tx.sv}
│   └── tb/tb_aa_qec_top.sv
├── checkpoints/                           # Trained Weighted-Bellman R2D2, Transformer, LoRA (.pt) & Q7.8 Weight Banks
├── datasets/                              # Dataset Manifest & 64-Bit AXI4-Stream Syndrome Packet Binary
├── configs/                               # d3_baseline.yaml, d5_simulation.yaml, target_board.yaml
├── tools/                                 # validate_architecture.py, cloudshell_hf_pipeline.py, generate_code_artifacts.py
└── run_cloudshell.sh                      # One-Command Gate Validation + Long-Horizon Training + HF Upload
```

---

## 4. Verification & Reproducibility

### Run Full Gate Validator Locally (Python + C++ Suites)
```bash
PYTHONPATH=. python3 tools/validate_architecture.py
```

### Run Full Long-Horizon Data Collection, Weighted-Bellman R2D2 Training & Hugging Face Upload
```bash
bash run_cloudshell.sh
```
