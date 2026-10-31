# AVID Benchmark-Evals — Noise-Generation & Latent-Estimation Suite Specification

This directory defines the **five-benchmark evaluation suite** across the three target superconducting transmon hardware noise families, the **six-stage noise-generation and baseline-estimation pipeline**, the **QuTiP-only classical baseline harness** under [`Qutip-Baseline/`](./Qutip-Baseline/), and all generated empirical evidence inside [`Qutip-Baseline/evidance/`](./Qutip-Baseline/evidance/).

> **Phase Scope:** This benchmark suite is scoped strictly to **physical noise modeling, multi-round noise trajectory generation, and latent noise-state estimation/forecasting (`B0–B5`)**. Downstream QEC decoding is intentionally excluded so that noise generators and noise-state estimators are validated directly on physical observables, trajectory fidelity, and probabilistic noise forecasts before any decoder coupling is introduced.

---

## Directory Layout

```text
Benchmark-Evals/
├── README.md                                  # Complete 5-benchmark & 6-stage noise-estimation specification
└── Qutip-Baseline/
    ├── configs/
    │   └── benchmark_configs.json             # Stage 0 frozen benchmark definitions & parameter provenance
    ├── qutip_baseline_pipeline.py             # Executable QuTiP physical-reference & B0..B5 noise-estimation harness
    ├── generate_benchmark_plots.py            # 300-DPI multi-panel plot generator for all 5 noise benchmarks
    └── evidance/                              # All generated noise-estimation results, logs, samples & figures
        ├── BASELINE_EVIDENCE_REPORT.md        # Quantitative baseline noise-generation & estimation report
        ├── baseline_evaluation_results.json   # Full structured metrics JSON (B1, A1, C1, C2, A2, Mixed-Noise)
        ├── baseline_pipeline.log              # Console execution log from QuTiP baseline harness
        ├── B1_TRANSMON_LEAKAGE_sample.json    # Validated trajectory sample (Varbanov et al. 2020 protocol)
        ├── A1_GATE_CROSSTALK_sample.json     # Validated trajectory sample (Ketterer & Wellens 2023 protocol)
        ├── MIXED_NOISE_sample.json            # Validated multi-mechanism trajectory sample (All 3 families)
        ├── 01_B1_surface17_leakage_hmm_vs_baselines.png
        ├── 02_A1_A2_spatial_crosstalk_and_chipwide_events.png
        ├── 03_C1_C2_nonstationary_tls_and_idle_memory.png
        └── 04_mixed_noise_generation_and_latent_estimation.png
```

---

## 1. Why We Selected These Five Noise-Generation Benchmarks

We selected **three noise families** represented by **five benchmark cases**:

1. **Spatially correlated noise** (`2 cases`): schedule-conditioned local gate crosstalk (`A1`) and chip-wide correlated relaxation events (`A2`).
2. **Leakage** (`1 case`): transitions outside the computational subspace ($|2\rangle$) and their persistence across QEC rounds (`B1`).
3. **Temporally correlated and non-stationary noise** (`2 cases`): driven TLS interactions (`C1`) and memory effects during idle evolution (`C2`).

> **Central Hypothesis:** An estimator that assumes independent, stationary, memoryless faults cannot adequately track or forecast a quantum processor whose noise has spatial structure, hidden persistent states, and temporal memory. A graph-aware recurrent state-space world model should improve estimation and multi-horizon prediction of these noise processes from the classical observations available to the control stack.

| Benchmark | Physical Reference | Hidden Noise Phenomenon | Main Property Tested in the Noise Estimator | Primary Baselines |
| :--- | :--- | :--- | :--- | :--- |
| **`A1` — Gate Crosstalk** | Ketterer & Wellens, *Phys. Rev. Applied* 20, 034065 (2023) | Neighbouring operations influence each other when executed concurrently | Graph structure & schedule conditioning $P(E_i,E_j \mid U_t, G)$ | `B0, B1, B3, B5` |
| **`A2` — Chip-Wide Events** | Wilen et al., *Nature* 594, 369–373 (2021) | Common environmental disturbance ($>600\ \mu\text{m}$) affects separated qubits | Spatial event inference & transient relaxation decay | `B0, B1, B3, B5` |
| **`B1` — Transmon Leakage** | Varbanov et al., *npj Quantum Information* 6, 102 (2020) | Transmon enters persistent non-computational $|2\rangle$ state ($\sim 16$ cycles) | Hidden discrete state $\{C, L\}$ & persistence tracking | `B0, B2, B3, B5` |
| **`C1` — Driven TLS** | Agarwal et al., *arXiv:2306.13021* (2023) | Environmental TLS coupling changes with time and drive $U(\theta)=X_{-\theta}X_\theta$ | Temporal memory & drive-conditioned forecasting | `B0, B3, B4, B5` |
| **`C2` — Idle Memory** | Gulácsi & Burkard, *Phys. Rev. B* 107, 174511 (2023) | Environmental memory causes coherence revivals ($\gamma_{\mathrm{TCL}}(t) < 0$) | Long-range temporal dependence across idle windows | `B0, B2, B4, B5` |

---

## 2. Six-Stage Noise-Generation & Baseline-Estimation Pipeline

1. **Stage 0 — Freeze the benchmark definitions:** Versioned configuration in [`Qutip-Baseline/configs/benchmark_configs.json`](./Qutip-Baseline/configs/benchmark_configs.json) with explicit parameter provenance (`published`, `synthetic_controlled`).
2. **Stage 1 — Establish the physical reference models:** Implemented in [`Qutip-Baseline/qutip_baseline_pipeline.py`](./Qutip-Baseline/qutip_baseline_pipeline.py) using `QuTiP` (`mesolve`, Pauli-twirled channel reductions, Liouvillian superoperators, and composite Hamiltonians).
3. **Stage 2 — Build the multi-round noise trajectory generators:** Translates validated physical channels into repeated syndrome-extraction noise trajectories (`Isolated` benchmarks `A1, A2, B1, C1, C2` and `Mixed-Noise` trajectories).
4. **Stage 3 — Generate the dataset with strict split & observation integrity:** Separates observable $o_t = (d_t, s_t, c_t, a_t, u_t, G)$ from simulator-only `latent_truth` $x_t$, and splits strictly by **complete trajectories and random seeds** (zero round-level leakage).
5. **Stage 4 — Establish the classical noise-estimation baselines (`B0–B5`):**
   - `B0`: Static independent Pauli noise model
   - `B1`: Static correlated noise model (schedule-blind)
   - `B2`: First-order Markov / Forward-Algorithm HMM (including Varbanov et al. 2-state $\{C,L\}$ HMM for syndrome-only and syndrome + analog readout)
   - `B3`: Multi-channel schedule/spatial/analog variational noise tracker
   - `B4`: Classical temporal estimator (`AR(1)` / Ornstein–Uhlenbeck filter)
   - `B5`: Oracle ground-truth noise reference (diagnostic upper bound)
6. **Stage 5 — Evaluate latent noise estimation, multi-horizon forecasting ($H \in \{1, 2, 4, 8\}$), mixed-noise disentanglement & channel ablations:**
   - Detection-event predictive NLL, Brier score, and Expected Calibration Error (ECE)
   - Pairwise joint-fault NLL and correlation RMSE (in-distribution and unseen OOD gate concurrency)
   - Leakage PR-AUC, F1, precision, recall, detection delay (rounds), and optimality ratio
   - Multi-horizon latent fault-rate RMSE ($H \in \{1, 2, 4, 8\}$)
   - Mixed-noise mechanism disentanglement Macro PR-AUC and observation-channel ablations (`Syndrome Only` vs. `+Schedule u_t` vs. `Full Contract (d_t, u_t, c_t, a_t)`).

---

## 3. Implementation Priority Order (Noise Generation & Estimation)

| Priority | Deliverable | Purpose |
| :---: | :--- | :--- |
| **1** | **Surface-17 Leakage Trajectory Generator (`B1`)** | Strongest published QEC-level hidden-state reference and 2-state HMM baseline |
| **2** | **Local Schedule-Conditioned Crosstalk Generator (`A1`)** | Establishes graph topology and parallel-schedule dependence |
| **3** | **Non-Stationary Driven-TLS (`C1`) & Idle-Memory (`C2`) Generators** | Establishes continuous environmental memory and drive-conditioned forecasting |
| **4** | **Chip-Wide Common-Event Generator (`A2`)** | Adds millimetre-scale spatio-temporal relaxation bursts |
| **5** | **Mixed-Noise Trajectory Generator & Channel Ablations (`MIXED`)** | Tests whether estimators can disentangle coexisting noise mechanisms without privileged truth |
| **6** | **Graph-Aware RSSM World Model Training** | Trained only after Priorities 1–5 and classical baselines (`B0–B5`) are frozen |

---

## Reproducing the QuTiP Noise-Generation & Baseline Suite

```bash
/home/tirtha/.gemini/antigravity-cli/brain/486bab03-c9f3-427b-b003-300d8e518af4/scratch/qutip_venv/bin/python \
  /home/tirtha/Benchmark-Evals/Qutip-Baseline/qutip_baseline_pipeline.py

/home/tirtha/.gemini/antigravity-cli/brain/486bab03-c9f3-427b-b003-300d8e518af4/scratch/qutip_venv/bin/python \
  /home/tirtha/Benchmark-Evals/Qutip-Baseline/generate_benchmark_plots.py
```
