# AVID Benchmark-Evals — Suite Specification & QuTiP Baseline Harness

This directory contains the **5-benchmark evaluation specification** across the three target quantum hardware noise families, the **7-stage baseline evaluation pipeline**, the **QuTiP-only classical baseline harness** under [`Qutip-Baseline/`](./Qutip-Baseline/), and all generated empirical evidence inside [`Qutip-Baseline/evidance/`](./Qutip-Baseline/evidance/).

## Directory Layout

```text
Benchmark-Evals/
├── README.md                                  # Complete 5-benchmark & 7-stage baseline specification
└── Qutip-Baseline/
    ├── configs/
    │   └── benchmark_configs.json             # Stage 0 frozen benchmark definitions & parameter provenance
    ├── qutip_baseline_pipeline.py             # Executable QuTiP physical-reference & B0..B5 / D0..D4 harness
    ├── generate_benchmark_plots.py            # 300-DPI multi-panel plot generator for all 5 benchmarks
    └── evidance/                              # All generated benchmark results, logs, samples & figures
        ├── BASELINE_EVIDENCE_REPORT.md        # Stage 0–6 quantitative baseline evaluation report
        ├── baseline_evaluation_results.json   # Full structured metrics JSON (B1, A1, C1, C2, A2, Stage 6)
        ├── baseline_pipeline.log              # Console execution log from QuTiP baseline harness
        ├── B1_TRANSMON_LEAKAGE_sample.json    # Validated trajectory sample (Varbanov et al. 2020 protocol)
        ├── A1_GATE_CROSSTALK_sample.json     # Validated trajectory sample (Ketterer & Wellens 2023 protocol)
        ├── 01_B1_surface17_leakage_hmm_vs_baselines.png
        ├── 02_A1_A2_spatial_crosstalk_and_chipwide_events.png
        ├── 03_C1_C2_nonstationary_tls_and_idle_memory.png
        └── 04_stage6_downstream_surface17_decoder_comparison.png
```

---

## 1. Why We Selected These Five Benchmarks

We selected **three noise families** represented by **five benchmark cases**:

1. **Spatially correlated noise** (`2 cases`): local gate crosstalk (`A1`) and chip-wide correlated events (`A2`).
2. **Leakage** (`1 case`): transitions outside the computational subspace and their persistence across QEC rounds (`B1`).
3. **Temporally correlated and non-stationary noise** (`2 cases`): driven TLS interactions (`C1`) and memory effects during idle evolution (`C2`).

> **Central Hypothesis:** A decoder that assumes independent, stationary, memoryless faults cannot adequately estimate a quantum processor whose noise has spatial structure, hidden persistent states, and temporal memory. A graph-aware recurrent state-space model (RSSM) should improve prediction of these effects from the observations available to the control stack.

| Benchmark | Physical Reference | Hidden Phenomenon | Main Property Tested in World Model | Primary Baselines |
| :--- | :--- | :--- | :--- | :--- |
| **`A1` — Gate Crosstalk** | Ketterer & Wellens, *Phys. Rev. Applied* 20, 034065 (2023) | Neighbouring operations influence each other | Graph structure & schedule conditioning $P(E_i,E_j \mid U_t, G)$ | `B0, B1, B3, B5` |
| **`A2` — Chip-Wide Events** | Wilen et al., *Nature* 594, 369–373 (2021) | Common environmental disturbance ($>600\ \mu\text{m}$) affects separated qubits | Spatial event inference & transient decay | `B0, B1, B3, B5` |
| **`B1` — Transmon Leakage** | Varbanov et al., *npj Quantum Information* 6, 102 (2020) | Transmon enters persistent non-computational $|2\rangle$ state ($\sim 16$ cycles) | Hidden discrete state $\{C, L\}$ & persistence | `B0, B2, B3, B5` |
| **`C1` — Driven TLS** | Agarwal et al., *arXiv:2306.13021* (2023) | Environmental TLS coupling changes with time and drive $U(\theta)=X_{-\theta}X_\theta$ | Temporal memory & control-conditioned prediction | `B0, B3, B4, B5` |
| **`C2` — Idle Memory** | Gulácsi & Burkard, *Phys. Rev. B* 107, 174511 (2023) | Environmental memory causes coherence revivals ($\gamma_{\mathrm{TCL}}(t) < 0$) | Long-range temporal dependence across idle windows | `B0, B2, B4, B5` |

---

## 2. Seven-Stage Baseline Evaluation Pipeline

1. **Stage 0 — Freeze the benchmark definitions:** Versioned configuration in [`Qutip-Baseline/configs/benchmark_configs.json`](./Qutip-Baseline/configs/benchmark_configs.json) with explicit parameter provenance (`published`, `synthetic_controlled`).
2. **Stage 1 — Establish the physical reference models:** Implemented in [`Qutip-Baseline/qutip_baseline_pipeline.py`](./Qutip-Baseline/qutip_baseline_pipeline.py) using `QuTiP` (`mesolve`, Pauli-twirled channel reductions, Liouvillian superoperators, and composite Hamiltonians).
3. **Stage 2 — Build the scalable repeated-QEC environment:** Connects physical channel trajectories to Surface-17 ($d=3$) stabilizer extraction rounds.
4. **Stage 3 — Generate the dataset correctly:** Enforces strict separation between observable $o_t = (d_t, s_t, c_t, a_t, u_t, G)$ and simulator-only `latent_truth` $x_t$, and splits strictly by **complete trajectories and seeds** (zero round-level leakage).
5. **Stage 4 — Establish the classical baseline models:**
   - `B0`: Static independent Pauli model
   - `B1`: Static correlated noise model (schedule-blind)
   - `B2`: First-order Markov / Forward-Algorithm HMM (including Varbanov et al. 2-state $\{C,L\}$ HMM for syndrome-only and syndrome + analog readout)
   - `B3`: Existing AVID variational / schedule-conditioned estimator
   - `B4`: Simple temporal estimator (`AR(1)` / Ornstein–Uhlenbeck filter)
   - `B5`: Oracle-state diagnostic reference
6. **Stage 5 — Evaluate latent-state & multi-horizon forecasting ($H \in \{1, 2, 4, 8\}$):** Reports Detection NLL, Brier score, ECE, Joint-Event RMSE, Leakage PR-AUC/F1/Detection Delay, and Multi-Horizon RMSE.
7. **Stage 6 — Evaluate downstream QEC decoding (`D0` Static MWPM, `D1` Correlated MWPM, `D2` Existing AVID, `D4` Oracle-Weight Reference):** Evaluates 50-round logical failure rate and per-round LER with 95% bootstrap confidence intervals across isolated and mixed-noise trajectories prior to introducing `D3` (World-Model RSSM).

---

## 3. Summary of QuTiP Baseline Results (Full Details in [`Qutip-Baseline/evidance/BASELINE_EVIDENCE_REPORT.md`](./Qutip-Baseline/evidance/BASELINE_EVIDENCE_REPORT.md))

- **Priority 1 (`B1` Surface-17 Leakage):** The 2-state $\{C, L\}$ HMM with analog readout (`B2_HMM_ANALOG`) achieves **`0.9998` PR-AUC** (`99.98%` optimality vs. `B5` Oracle) and **`0.07` rounds** mean detection delay (`0.9946` PR-AUC and `0.18` rounds delay with syndrome-only `B2_HMM_SYNDROME`), compared to `0.1261` PR-AUC for `B0` static Pauli and `1.28` rounds delay for `B3` single-timescale EWMA.
- **Priority 2 (`A1` Gate Crosstalk):** QuTiP Pauli-twirled concurrent operations increase pairwise joint faults from `0.00074` (isolated) to `0.03523` (`13.13x` correlation ratio). Schedule-conditioned estimation (`B3`) reduces pairwise joint RMSE from `2.44%` (`B0`) and `1.73%` (`B1` static correlated) down to **`0.034%`** (`0.49%` on unseen 3-gate OOD concurrency).
- **Priority 3 (`C1` Driven TLS & `C2` Idle Memory):** Composite $\text{Transmon} \otimes \text{TLS}$ evolution under $U(\theta)=X_{-\theta}X_{+\theta}$ demonstrates quadratic drive-induced error growth (`3.63%` at $50\text{ MHz}$ vs. `0.15%` Markovian), and idle memory evolution exhibits negative canonical TCL decay rates ($\min \gamma_{\mathrm{TCL}} = -20.17\ \mu\text{s}^{-1}$, `< 0` for `44.0%` of idle time) and split precession frequencies (`0.80 MHz`, `3.19 MHz`).
- **Priority 4 (`A2` Chip-Wide Events):** Spatial phonon-kernel filtering (`B3`) achieves **`0.8124` Event PR-AUC** and **`0.08` rounds** detection delay across the $1100\ \mu\text{m}$ array, whereas local pairwise models (`B1`) fail to capture chip-wide bursts (`F1 = 0.0087`).
- **Priority 5 & Stage 6 (Mixed-Noise Surface-17 Decoding):** Under simultaneous crosstalk + leakage + temporal memory (`MIXED_ALL_THREE`), 50-round logical failure rates drop from **`58.33%`** (`D0` Static MWPM) and **`51.67%`** (`D1` Correlated MWPM) to **`41.67%`** (`D2` Existing AVID), while the `D4` Oracle achieves **`25.00%`**—establishing a clear **`16.67%` absolute logical error reduction headroom** for the Phase-2 Graph-Aware RSSM (`D3`).

---

## Reproducing the Baseline Suite

```bash
/home/tirtha/.gemini/antigravity-cli/brain/486bab03-c9f3-427b-b003-300d8e518af4/scratch/qutip_venv/bin/python \
  /home/tirtha/Benchmark-Evals/Qutip-Baseline/qutip_baseline_pipeline.py

/home/tirtha/.gemini/antigravity-cli/brain/486bab03-c9f3-427b-b003-300d8e518af4/scratch/qutip_venv/bin/python \
  /home/tirtha/Benchmark-Evals/Qutip-Baseline/generate_benchmark_plots.py
```
