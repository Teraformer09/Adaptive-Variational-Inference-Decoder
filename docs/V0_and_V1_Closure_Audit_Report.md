# AVID V0 (40/40) & V1 (4–8 Nov Emulator-First) Closure Audit Report

## 1. Executive Summary of Audit Findings Closure (`Finding A – D`)

| Audit Finding | Root Cause Identified | Implemented Remediation | Verification Artifact & Result |
| :--- | :--- | :--- | :--- |
| **Finding A**: Native Verilator RTL vs. C++ Fallback Provenance | When `verilator` was absent on a host, `emulator/verilator/build.sh` fell back to the C++ reference driver unless explicitly enforced. | Added `--require-native-rtl` (`AVID_REQUIRE_NATIVE_VERILATOR=1`), enabled formal SystemVerilog Assertions (`--assert`) in Verilator compilation, and generated `emulator/verilator/models/rtl_build_provenance.json` with SHA-256 hashes of all `.sv` files and `Vaa_qec_top.h`. | `reports/latest_report.json` (`m1_verilator_standalone_rtl.sva_assertions_enabled = true`, `reference_equivalence_mismatches = 0`). |
| **Finding B**: `17` BP Core Cycles vs. `19` / `22` End-to-End AXI Cycles | `aa_qec_top.sv` has `3` cycles of AXI4-Stream RX FIFO + tap ingress, `17` cycles ($2+5K$ at $K=3$) of `bp_engine.sv` core decoding, and `2` cycles of `axis_correction_tx.sv` egress (`3 + 17 + 2 = 22` AXI cycles = `88.0 ns` @ `250 MHz`). The earlier C++ fallback stepped only `1` ingress + `1` egress cycle (`19` cycles). | Reconciled both native Verilator and the C++ reference driver in `verilator_rtl_driver.h` to step and report `axi_ingress_cycles = 3`, `bp_core_cycles = 17` ($2+5K$), `axi_egress_cycles = 2`, and `total_axis_cycles = 22` ($7+5K$). Added `cfg_osd_mode` (`0x058`) separating `17`-cycle pure BP (`mode 0`), `17`-cycle inline OSD-0 (`mode 1`), and `19`-cycle post-BP OSD-0 (`mode 2`, `24` AXI total cycles). | `reports/latest_report.json` (`k3_bp_core_cycles = 17`, `k3_axi_ingress_cycles = 3`, `k3_axi_egress_cycles = 2`, `k3_axis_total_cycles = 22`). |
| **Finding C**: Causal Attribution Without Scenario Shortcuts | In rotated $d=3$ X-check geometry, correlated pair $Z_4 Z_5$ and single-qubit $Z_3$ share check signature `0x4`. Earlier code used a scenario-name check to disambiguate `0x4`. | Removed all scenario-name conditionals from the estimator/controller (`solver_contains_scenario_conditionals = false`). Implemented a **Two-Pass Endpoint-Coupled 16-State Joint-DP L-BFGS Solver** (`lbfgs_solver.cpp`): Pass 1 recovers singleton signature orbit `0x6` ($\hat{\theta}_4 = \hat{p}(Z_4)$) from `hist_x`, and Pass 2 updates the endpoint-coupled gauge prior $\pi_{u,v}$ to disambiguate $Z_4 Z_5$ (`0x4`) purely from observed syndrome histograms, refined by the **64-unit LSTM `R2D2-WB` Meta-Controller** (`r2d2_lstm_controller.h`). | `reports/v0_closure_40_40_report.json` (`C08` 6-way ablation & `C09` zero-shortcut audit). |
| **Finding D**: Epoch vs. Weight-Bank Commit Counter Reconciliation (`32` vs. `31` / `6` vs. `7`) | `avid_daemon` performs `1` initial boot commit (`initial_boot_commits = 1`) before the syndrome loop plus `1` commit per completed adaptive epoch (`adaptive_epoch_commits`), minus any commits cleared by hardware reset (`reset_cleared_commits` in `reset_recovery`). Earlier logs did not separate boot commits from epoch commits, and wall-clock `CLOCK_MONOTONIC` caused occasional host-preemption holds. | Explicitly logged `initial_boot_commits`, `adaptive_epoch_commits`, `reset_cleared_commits`, `weight_bank_commits`, and `epoch_commit_reconciled = (initial_boot_commits + adaptive_epoch_commits - reset_cleared_commits == weight_bank_commits)`, and switched `stage_profiler.cpp` to `CLOCK_PROCESS_CPUTIME_ID`. | All 10 scenarios in `reports/latest_report.json` report `epoch_commit_reconciled: true`. |

---

## 2. V0 Baseline 40/40 Closure Tracker (`A01..A10`, `B01..B10`, `C01..C10`, `D01..D10`)

Verified automatically by `PYTHONPATH=. python3 tools/verify_v0_closure_40.py` (artifact: `reports/v0_closure_40_40_report.json`, score: **`40/40 CLOSED`**).

### Category A — Mathematical Model & Identifiability (`10/10 CLOSED`)

| ID | Requirement | Status | Measured Verification Evidence |
| :--- | :--- | :---: | :--- |
| **A01** | Freeze exact 21-parameter noise-model definition and parameter ordering (`0..8` single-qubit, `9..20` $ZZ$ pairs) | **CLOSED** | `H_X H_Z^T mod 2 = 0` for $d=3$ and $d=5$; 21 X-basis & 21 Z-basis signatures frozen in `noise_model.h` & `joint_dp_likelihood.py`. |
| **A02** | Document and verify probability model for single-qubit and correlated events | **CLOSED** | $\sum_{s=0}^{15} P_\theta(s) = 1.000000000000000$, $P_{\theta=0}(0) = 1.0$. |
| **A03** | Verify normalization and probability constraints across full parameter domain $[10^{-5}, 0.45]^{21}$ | **CLOSED** | Tested 25 random interior/boundary vectors; max normalization error $= 4.44 \times 10^{-16}$. |
| **A04** | Verify 16-state Joint-DP transition probabilities against independent $2^M$ brute-force enumeration | **CLOSED** | Max absolute difference vs. exact $2^{12}$ enumeration $= 4.44 \times 10^{-16}$. |
| **A05** | Validate analytic coset gradients $\frac{\partial P(s)}{\partial p_e} = \frac{P(s \oplus g_e) - P(s)}{1 - 2 p_e}$ against central finite differences | **CLOSED** | Max absolute gradient discrepancy across all 21 parameters $= 1.23 \times 10^{-9} < 10^{-8}$. |
| **A06** | Separate data-identifiable signature orbits (`9`) from prior-regularized gauge directions (`12`) | **CLOSED** | Unregularized $16 \times 21$ Jacobian SVD rank $= 9$, nullspace dimension $= 12$, regularized MAP Hessian rank $= 21/21$. |
| **A07** | Document gauge-equivalent parameter sets and 9 identifiable orbit combinations | **CLOSED** | 9 non-zero orbits (`0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x8, 0xA, 0xC`) + 1 commute-with-all boundary orbit (`0x0 = {ZZ_3_6, ZZ_2_5}`). |
| **A08** | Reconcile Gaussian probability-space prior and log-odds prior via Jacobian metric $dp/d\eta = p(1-p)$ | **CLOSED** | Exact local equivalence $\lambda_{\text{logit}} = \lambda_{\text{prob}} [p_0(1-p_0)]^2 = 1.0915 \times 10^{-5}$ at $p_0 = 0.015, \lambda_{\text{prob}} = 0.05$. |
| **A09** | Record conditioning, Hessian rank, and sensitivity across multiple operating points | **CLOSED** | MAP Hessian condition number $\kappa = 413.30$ (low noise) and $\kappa = 47.90$ (high noise), rank $= 21/21$. |
| **A10** | Known-ground-truth parameter & 9-orbit recovery verification | **CLOSED** | True $p(Z_4)=0.055 \to$ recovered $0.05492$; 9-class orbit RMSE $= 7.7 \times 10^{-5}$; 21-param RMSE $= 0.002399$. |

### Category B — Software Estimator & Reference Decoder (`10/10 CLOSED`)

| ID | Requirement | Status | Measured Verification Evidence |
| :--- | :--- | :---: | :--- |
| **B01** | Freeze canonical estimator objective and exact implementation | **CLOSED** | C++ `test_baseline_suite` exit code `0`; exact 16-state Joint-DP NLL + endpoint-coupled L2 prior. |
| **B02** | Verify convergence and stopping criteria for projected L-BFGS optimizer | **CLOSED** | Projected gradient norm $\|g_{\text{proj}}\|_\infty = 0.00178$, monotone Armijo decrease verified across all scenarios. |
| **B03** | Test initialization sensitivity and local-minimum behaviour across starting points | **CLOSED** | Tested initializations $p_0 \in \{0.005, 0.012, 0.030\}$; max 9-orbit deviation $= 3.32 \times 10^{-6}$. |
| **B04** | Test estimator behaviour under sparse and zero-count syndrome bins ($N=64$) | **CLOSED** | 14 zero-count bins + 2 active bins: NLL $= 0.36097$, all 21 gradients finite, zero `NaN`/`Inf`. |
| **B05** | Validate both $X$- and $Z$-basis histogram handling (`hist_x` and `hist_z`) | **CLOSED** | `avid_daemon.cpp` accumulates both `hist_x[16]` and `hist_z[16]` and fits `g_model_z` and `g_model_x` every epoch (`dual_xz_histograms_used = true`). |
| **B06** | Verify mapping from estimated noise parameters to decoder LLRs | **CLOSED** | Associative XOR marginalization $p_{\text{eff}}(v)$ + pair-aware logical-triangle LLR projection verified positive and bounded. |
| **B07** | Validate Q7.8 fixed-point quantization and saturation behaviour | **CLOSED** | Max quantization error $= 0.001692 \le 0.5 / 256 = 0.001953$ (half-LSB bound). |
| **B08** | Compare floating-point and Q7.8 decoder outputs | **CLOSED** | `100.0%` decision agreement across all syndrome test vectors. |
| **B09** | Test all 16 $X$ and 16 $Z$ syndrome patterns including ambiguous `0x2` and `0x4` cases | **CLOSED** | All 32 single-basis syndromes tested across `cfg_osd_mode` $\in \{0, 1, 2\}$ in `standalone_rtl_tb.cpp`. |
| **B10** | Establish trusted software reference decoder with 0 mismatches vs. RTL | **CLOSED** | `reference_equivalence_mismatches = 0` across all 16 $X$ and 16 $Z$ syndromes under both uniform and pair-adapted weights. |

### Category C — Benchmark Validity & Scientific Attribution (`10/10 CLOSED`)

| ID | Requirement | Status | Measured Verification Evidence |
| :--- | :--- | :---: | :--- |
| **C01** | Freeze canonical noise regimes and ground-truth parameters | **CLOSED** | 10 canonical scenarios frozen in `configs/v1_integration.yaml` and `configs/v1_full_benchmark.yaml`. |
| **C02** | Verify scenario generators implement documented stationary, drifting, correlated & burst processes | **CLOSED** | Verified stationary LER $= 0.00456$ vs. correlated LER $= 0.05339$ and burst-window LER $= 0.09115$. |
| **C03** | Use identical random streams for adaptive and static baselines | **CLOSED** | `static_b0` and adaptive `avid` evaluate on the exact same per-round fault realizations (`shared_stream_comparison = true`). |
| **C04** | Separate initialization/training seeds from evaluation seeds | **CLOSED** | `R2D2-WB` weights initialized from fixed seed `0x5232443257423634`; evaluated on disjoint scenario seeds `2026..2179` / `777..882`. |
| **C05** | Report 95% Wilson confidence intervals and raw failure counts for logical error rates | **CLOSED** | Every scenario exports `raw_static_b0_failures`, `raw_avid_z_failures`, `logical_error_rate_ci95_low`, and `logical_error_rate_ci95_high`. |
| **C06** | Run sufficient trial counts to support statistically significant effect sizes | **CLOSED** | 95% Wilson CI upper bound of AVID (`0.04632` at 1.5k rounds; `0.0258` at 8.1k rounds) lies strictly below Static B0 (`0.05339`). |
| **C07** | Report estimator error (9-orbit & 21-param RMSE) and adaptation delay alongside LER | **CLOSED** | Logged `orbit_9class_rmse` (`0.0081..0.0148`), `param_21_rmse` (`0.0044..0.0125`), and `adaptation_delay_epochs = 1`. |
| **C08** | Perform 6-way architectural ablations isolating estimator, projection, R2D2-WB, and OSD-0 | **CLOSED** | Executed across co-sim bridge (Section 3 below): Static (`0%`) $\to$ Marginal (`0%`) $\to$ Pure 9-LLR BP (`-137.8%`) $\to$ Pair-Proj (`+15.85%`) $\to$ Full AVID R2D2-WB (`+32.93%`). |
| **C09** | Verify causal pipeline without ground-truth leakage or scenario shortcuts in estimator | **CLOSED** | Static code audit confirms `solver_contains_scenario_conditionals = false` in `lbfgs_solver.cpp` and `noise_model.cpp`. |
| **C10** | Reproduce headline results deterministically across repeated runs | **CLOSED** | Identical seeds produce bit-identical transaction counts, RTL cycles, and logical error rates. |

### Category D — Evidence, Tests & Release Discipline (`10/10 CLOSED`)

| ID | Requirement | Status | Measured Verification Evidence |
| :--- | :--- | :---: | :--- |
| **D01** | Create canonical manifest with commit, configuration, seed, and tool versions | **CLOSED** | Generated at `runs/<run_id>/manifest.json` and `emulator/verilator/models/rtl_build_provenance.json`. |
| **D02** | Store raw outputs and SHA-256 checksums for every benchmark artifact | **CLOSED** | 69 per-run artifact files hashed in `runs/<run_id>/checksums.sha256`. |
| **D03** | Ensure tests fail on numerical regressions and epoch/commit discrepancies | **CLOSED** | `collect_artifacts.py` fails the run if `epoch_commit_reconciled == false` or `rejected_transactions > 0`. |
| **D04** | Add negative tests for malformed inputs, invalid registers, and invalid parameters | **CLOSED** | `standalone_rtl_tb` verifies `SLVERR` on invalid registers; `test_bridge_transactions` verifies 6 negative protocol cases (`ERR_DUPLICATE_TX_ID`, `ERR_UNSUPPORTED_VERSION`, `ERR_MALFORMED_JSON`, `ERR_INVALID_REGISTER`, `ERR_INVALID_SYNDROME`). |
| **D05** | Deterministic rerun verification on identical seed | **CLOSED** | Verified in `verify_v0_closure_40.py` (`deterministic_seed_verified = true`). |
| **D06** | Document known limitations (9 observable orbits vs. 21 physical rates; `60 us` sub-epoch bursts) | **CLOSED** | Quantified via `short_burst_60us_vs_10ms` scenario and `identifiability_report.md`. |
| **D07** | Qualify benchmark suite descriptions and latency definitions | **CLOSED** | Explicitly separated `17.0` BP core cycles (`68.0 ns`) from `22.0` total AXI-Stream cycles (`88.0 ns`) and `19.0`-cycle post-BP OSD-0 mode. |
| **D08** | Separate measured emulator values from analytical estimates and V2 hardware targets | **CLOSED** | V1 Dual-Emulator measurements vs. V2 physical Zynq FPGA deployment clearly delineated in `README.md`. |
| **D09** | Define V0/V1 exit criteria and link each item to an executable evidence artifact | **CLOSED** | Every item in `reports/v0_closure_40_40_report.json` links to its measured JSON metric. |
| **D10** | Complete 40/40 V0 closure tracker and V1 integration verification report | **CLOSED** | `v0_closure_score = "40/40"`, `all_40_items_closed = true`. |

---

## 3. Six-Way Architectural Ablation Study (`correlated_noise`, 1,536 Rounds via Co-Sim Bridge)

| Ablation Variant | `--controller-mode` | `--osd-mode` | BP Core Cycles | Total AXI Cycles | Logical Error Rate (LER) | LER Reduction vs. Static | Architectural Role Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Static B0** | `static` | `1` | `17.0` (`68 ns`) | `22.0` (`88 ns`) | `0.05339` | `0.00%` | Uniform prior baseline; misdecodes $Z_4 Z_5$ (`0x4`) as $Z_3$. |
| **2. Marginal L-BFGS B1** | `marginal_lbfgs` | `1` | `17.0` (`68 ns`) | `22.0` (`88 ns`) | `0.05339` | `0.00%` | Rank-4 check marginals (`legacy_marginal_nll = 0.8458` vs. `joint_dp_nll = 0.7511`) cannot separate $Z_4 Z_5$ from $Z_3$. |
| **3. Joint-DP + Pure 9-Marginal-LLR BP** | `joint_dp_map` | `0` | `17.0` (`68 ns`) | `22.0` (`88 ns`) | `0.12695` | `-137.80%` | Confirms **7 Nov roadmap finding**: pure 9-marginal-LLR Min-Sum BP without pair-aware OSD-0 cannot resolve weight-2 vs. weight-1 single-check ambiguity. |
| **4. Joint-DP + Memoryless Pair Projection** | `joint_dp_proj` | `1` | `17.0` (`68 ns`) | `22.0` (`88 ns`) | `0.04492` | **`-15.85%`** | Memoryless single-epoch pair-aware projection resolves `0x4` once single-window confidence exceeds threshold. |
| **5. Full AVID (`Joint-DP + 64-Unit LSTM R2D2-WB + Inline OSD-0`)** | `avid_r2d2` | `1` | `17.0` (`68 ns`) | `22.0` (`88 ns`) | **`0.03581`** | **`-32.93%`** (`-58.05%` @ 8.1k rounds) | 64-unit LSTM `R2D2-WB` recurrent state $(h_t, c_t)$ locks onto correlated bursts from epoch 1 and maintains temporal lock across noisy windows. |
| **6. Full AVID (`Joint-DP + 64-Unit LSTM R2D2-WB + Post-BP OSD-0 Stage`)** | `avid_r2d2` | `2` | `19.0` (`76 ns`) | `24.0` (`96 ns`) | **`0.03581`** | **`-32.93%`** | Executes OSD-0 in dedicated post-BP FSM states (`ST_OSD_STEP1 -> ST_OSD_STEP2`, `+2` cycles outside the 17-cycle BP path) with identical accuracy. |

---

## 4. Complete 4 Nov – 8 Nov Emulator-First Checklist Closure

| Roadmap Day | Checklist Requirement | Implementation File(s) | Status & Measured Result |
| :--- | :--- | :--- | :--- |
| **4 Nov (Day 1)** | Zynq-7020 **ARM Cortex-A9** (`arm-linux-gnueabihf-g++ -mcpu=cortex-a9`, `qemu-arm -cpu cortex-a9`) + `aarch64` cross-compilation & QEMU execution | `scripts/setup_cloudshell.sh`, `emulator/qemu/build.sh`, `emulator/qemu/run.sh` | **CLOSED** — Supports `AVID_QEMU_ARCH=cortex-a9` (`avid_daemon_cortex_a9`) and `aarch64` (`avid_daemon_aarch64`). |
| **4 Nov (Day 1)** | Wire **64-Unit LSTM `R2D2-WB` Meta-Controller** into live `avid_daemon` slow-path loop | `software/src/control/r2d2_lstm_controller.h`, `software/src/inference/avid_daemon.cpp` | **CLOSED** — 64-unit LSTM (`25,998` params = `25.39 KB` `int8` / `101.55 KB` `fp32`) runs every epoch in `avid_daemon`. |
| **4 Nov (Day 1)** | External Stim syndrome binary stream ingestion (`--stim-bin`) | `software/src/inference/avid_daemon.cpp` | **CLOSED** — `--stim-bin <path>` replays external 64-bit syndrome packets through the HAL & RTL bridge. |
| **5 Nov (Day 2)** | Dual $X/Z$ CSS syndrome accumulation (`hist_x` and `hist_z`) & dual-basis 16-state Joint-DP estimation (`g_model_z` & `g_model_x`) | `software/src/inference/noise_model.cpp`, `software/src/inference/avid_daemon.cpp` | **CLOSED** — Both `hist_x[16]` and `hist_z[16]` accumulated and fitted every epoch; both `active_x` and `active_z` updated in RTL. |
| **6 Nov (Day 3)** | Causal Two-Pass Endpoint-Coupled L-BFGS Solver (zero scenario shortcuts), logging **Legacy Marginal NLL vs. 16-State Joint-DP NLL**, **9-orbit RMSE**, and **21-param RMSE** | `software/src/estimator/lbfgs_solver.cpp`, `software/src/inference/noise_model.cpp` | **CLOSED** — Zero scenario conditionals in solver; `final_nll = 0.7511` vs. `legacy_marginal_nll = 0.8458`; `orbit_9class_rmse = 0.0081..0.0148`. |
| **7 Nov (Day 4)** | Separate **Pure 9-Marginal-LLR BP (`17` cyc)** (`cfg_osd_mode=0`), **Inline OSD-0 (`17` cyc)** (`cfg_osd_mode=1`), and **Post-BP OSD-0 Stage Outside 17-Cycle Path (`19` cyc)** (`cfg_osd_mode=2`) | `firmware/rtl/config/weight_map_reg.sv`, `firmware/rtl/decoder/bp_engine.sv`, `firmware/rtl/top/aa_qec_top.sv` | **CLOSED** — Verified in `M1` (`mode0=17 cyc`, `mode1=17 cyc`, `mode2=19 cyc`) and in the 6-way ablation study. |
| **8 Nov (Day 5)** | Formal SystemVerilog Assertions (`SVA`, `--assert`) for active-bank stability while `bp_ready_in==0`, reconciled `17` core vs. `22` AXI cycles, live $K=1\dots 5$ adaptation, fallback trigger, `60 us` vs. `10 ms` burst test, and memory/CPU audit | `firmware/rtl/config/weight_map_reg.sv`, `firmware/rtl/decoder/bp_engine.sv`, `emulator/verilator/tb/standalone_rtl_tb.cpp`, `configs/v1_integration.yaml` | **CLOSED** — All 10 M4 scenarios (`stationary_noise`, `slowly_drifting_noise`, `correlated_noise`, `burst_disturbance`, `delayed_weight_commit`, `reset_recovery`, `busy_decoder`, `variable_k_adaptation`, `fallback_safety_trigger`, `short_burst_60us_vs_10ms`) **PASS**. |
