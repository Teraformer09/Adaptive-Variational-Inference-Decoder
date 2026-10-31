# QuTiP 3-Noise Physical Reference & Simulator Limitation Analysis (`Limiation_sim`)

This directory (`Limiation_sim/`) contains the small-system **QuTiP physical-reference explorations**, simulator scaling benchmarks, and illustrative hardware-gap visualizations for the **three target quantum hardware noise families** in the `Noise-Generation_Model` branch.

> **Scope & Provenance Note:** Please read [`INTERPRETATION_NOTES.md`](./INTERPRETATION_NOTES.md) alongside this file. These experiments focus strictly on **physical noise mechanisms, QuTiP simulation limits, and hybrid noise-generation design** (no downstream QEC decoding is evaluated here).

---

## 1. Directory Contents

```text
Limiation_sim/
├── INTERPRETATION_NOTES.md                            — Methodological scope, channel-reduction & provenance rules
├── README.md                                          — Documentation of the 3 noise families & QuTiP limitations
├── verify_qutip_noise_limitations.py                  — Executable 3-test QuTiP verification script
├── verification_results.log                           — Recorded output & runtime/memory benchmarks from QuTiP
├── generate_limitation_plots.py                       — Script generating the 6 exploratory/illustrative PNG figures
├── 01_correlated_zz_crosstalk_comparison.png          — Test 1: Coherent ZZ + Drive Crosstalk vs. Lindblad Dephasing
├── 02_multilevel_transmon_leakage_dynamics.png        — Test 2 (Parts A & B): |2> Gate Leakage & Spectator Scrambling
├── 03_qutip_exponential_scaling_wall.png              — Test 2 (Part C): Measured N=2..5 Runtime/RAM & Dense 3^(2N) Wall
├── 04_nonstationary_nonmarkovian_drift_and_tls.png    — Test 3: OU Drift + TLS Telegraph Switching & Quasi-Static Ramsey
├── 05_qutip_vs_actual_quantum_device_behavior.png     — Illustrative 4-Panel Comparison of Hardware-Level Noise Effects
└── 06_summary_dashboard_all_3_noises.png              — Summary Dashboard of the 3 Noise Families & Simulator Throughput
```

---

## 2. Finalized Three Noise Families for the Noise-Generation Model

To build a realistic noise-generation environment for repeated surface-code cycles on superconducting transmons, we target three physical noise families that violate standard independent, stationary, memoryless Pauli noise assumptions:

| # | Noise Family | Physical Mechanism (Small-System QuTiP Reference) | Scalable Noise-Generation Representation (Custom Trajectory Wrapper + Stim) |
| :--- | :--- | :--- | :--- |
| **1** | **Spatially Correlated Noise & Crosstalk** | Residual dispersive coupling $H_{ZZ} = \sum_{\langle i,j\rangle} \frac{\zeta_{ij}(t)}{2} Z_i Z_j$, simultaneous microwave drive crosstalk $\sum_{i \neq j} \Omega_{ij}(t) X_i$, and substrate-mediated spatial relaxation bursts $\Gamma_{1,i}(t)$. | Schedule-conditioned edge fault channels $P(E_i, E_j \mid u_t, G)$ (derived via Pauli twirling / PTM reduction) and spatio-temporal common-event fields $g(\mathbf{r}_i, t)$. |
| **2** | **Non-Computational State Leakage & Seepage** | Weak transmon anharmonicity ($\alpha_i / 2\pi \approx -200\text{ MHz}$) allows fast microwave drives and Net-Zero CZ flux pulses to excite population outside $\mathcal{H}_{\text{comp}} = \operatorname{span}\{|0\rangle, |1\rangle\}$ into $|2\rangle$, persisting over $T_1^{(2\to 1)}$ and scrambling coupled checks via dispersive shifts $\chi_2 |2\rangle\langle 2| \otimes Z$. | Hidden discrete per-transmon state $S_i(t) \in \{C, L\}$ with gate-conditioned leakage $\Gamma_{C\to L}$, seepage $\Gamma_{L\to C}$, conditional check phase scrambling, and soft analog readout shifts. |
| **3** | **Non-Stationary & Temporally Correlated (Non-Markovian) Noise** | Time-varying environmental couplings governed by slow Ornstein–Uhlenbeck (OU) drift, Two-Level System (TLS) defects coherently exchanging excitations ($g_{\text{TLS}}(\sigma_+\tau_- + \sigma_-\tau_+)$) or switching telegraphically, and non-Markovian idle bath memory ($\gamma_{\text{TCL}}(t) < 0$). | Continuous & discrete latent environmental state trajectories $x_t = (x_t^{\text{drift}}, x_t^{\text{TLS}}, x_t^{\text{memory}})$ evolving across rounds without artificial inter-round bath reset. |

---

## 3. QuTiP's Limitations as a Standalone Large-Scale QEC Noise Generator

While `qutip` and `qutip-qip` accurately integrate microscopic master equations for **2–5 transmons/qutrits**, four structural limitations prevent using QuTiP alone to generate large-scale, multi-round surface-code noise datasets:

1. **Exponential Hilbert & Liouville Space Scaling ($3^{2N}$ Dense Density Matrix / $3^{4N}$ Liouvillian)**:
   - Modeling transmon leakage requires at least 3 levels per site ($d=3$ qutrit). An explicitly dense `complex128` $N$-qutrit density matrix requires $3^{2N} \times 16\text{ bytes}$, and its Liouvillian superoperator has $3^{4N}$ entries.
   - For a 17-transmon distance-3 surface code (`Surface-17`), $3^{17} = 129{,}140{,}163$ and a dense `complex128` density matrix would require $3^{34} \times 16\text{ B} \approx 2.49 \times 10^8\text{ GB}$ ($\sim 249\text{ PB}$). Empirically, `qutip.mesolve` runtime grows from **$0.0592\text{ s}$ ($N=2$)** to **$65.9167\text{ s}$ ($N=5$ qutrits)**.
2. **Time-Local Lindblad Assumption vs. Multi-Round Non-Markovian Bath Memory**:
   - Standard constant-operator `qutip.mesolve` assumes a memoryless Born–Markov bath. Retaining environmental memory across rounds requires either tensoring explicit bath/TLS degrees of freedom into the Hilbert space (further compounding the exponential scaling wall) or re-compiling time-dependent `QobjEvo` trajectories across thousands of Monte Carlo realizations.
3. **Lack of Fast Repeated Mid-Circuit Measurement, Reset, and Stabilizer Sampling**:
   - Generating $10^4\text{–}10^6$ repeated syndrome-extraction rounds with mid-circuit ancilla projection, conditional reset, and trajectory logging is orders of magnitude slower in a density-matrix ODE solver than in a stabilizer frame simulator (`Stim`).
4. **Manual Physical-to-Effective Channel Reduction Required**:
   - QuTiP evolves continuous quantum states $\rho(t)$; turning those evolutions into valid effective circuit fault models requires explicit channel tomography or Pauli-twirl reductions (as implemented in [`Benchmark-Evals/Qutip-Baseline/`](../Benchmark-Evals/Qutip-Baseline/)).

---

## 4. Summary of the Three Exploratory QuTiP Tests (`verify_qutip_noise_limitations.py`)

### Test 1: Coherent $ZZ$ + Spectator Drive Crosstalk vs. Lindblad Dephasing (`01_correlated_zz_crosstalk_comparison.png`)
- **Exploratory Setup:** Two coupled transmons initialized in $|++\rangle$ with exploratory coupling $\zeta_{ZZ}/2\pi = 92\text{ kHz}$ and spectator drive $\Omega_{\text{xtalk}}/2\pi = 25\text{ kHz}$ over $t = 1.0\ \mu\text{s}$.
- **Observation:** The joint state-projection probability onto $|--\rangle$ (`0.0794`) exceeds the product of single-qubit $|-\rangle$ projections (`0.0063`) by **$12.59\times$**, and multi-cycle evolution exhibits coherent revivals ($\langle X_1 \rangle: +0.8412 \to -0.4457 \to +0.9486$) absent from monotonic Lindblad dephasing. *(Note per [`INTERPRETATION_NOTES.md`](./INTERPRETATION_NOTES.md): raw state projections onto $|--\rangle$ are state-projection statistics; see [`Benchmark-Evals/Qutip-Baseline/qutip_baseline_pipeline.py`](../Benchmark-Evals/Qutip-Baseline/qutip_baseline_pipeline.py) for the Pauli-twirled effective channel reduction).*

### Test 2: Multilevel Qutrit Leakage & $3^N$ Scaling (`02_multilevel_transmon_leakage_dynamics.png`, `03_qutip_exponential_scaling_wall.png`)
- **Exploratory Setup:** An illustrative $8\text{ ns}$ Gaussian $X_\pi$ pulse on a 3-level transmon ($\alpha/2\pi = -200\text{ MHz}$) followed by 5 rounds of dispersive coupling ($\chi_2/2\pi = 0.45\text{ MHz}$, $T_1^{(2\to 1)} = 15\ \mu\text{s}$) to a spectator qubit, plus runtime scaling for $N=2\dots 5$ qutrits.
- **Observation:** Excites $12.19\%$ peak mid-pulse $|2\rangle$ population ($0.79\%$ post-pulse residual), retains $71.65\%$ of leaked population after $5\ \mu\text{s}$, and scrambles spectator $\langle X \rangle$ phase. *(Note: this is an illustrative single-pulse qutrit calculation; the Varbanov et al. Surface-17 Net-Zero CZ leakage protocol is benchmarked separately in [`Benchmark-Evals/Qutip-Baseline/`](../Benchmark-Evals/Qutip-Baseline/)).*

### Test 3: Non-Stationary OU + TLS Trajectory & Call-Pattern Overhead (`04_nonstationary_nonmarkovian_drift_and_tls.png`)
- **Exploratory Setup:** 150-round classical OU + telegraph detuning trajectory (lag-1 autocorrelation $+0.9125$, Fano factor $3.004$), quasi-static ensemble Ramsey dephasing, and comparison of 150 separate `mesolve` calls ($741.39\text{ ms}$) vs. a single static call ($4.47\text{ ms}$).

### Illustrative Hardware Gap Figures (`05_qutip_vs_actual_quantum_device_behavior.png`, `06_summary_dashboard_all_3_noises.png`)
- **Provenance Note:** The 4-panel hardware behavioral comparison (heterodyne IQ readout blobs, bias-tee flux-settling tails, phonon burst spatial profile, and real-time latency deadlines) and throughput dashboard are **synthetic/illustrative diagrams** motivating the hybrid noise-generation architecture (`QuTiP / quantumsim` physical reference $\to$ custom stochastic trajectory engine $\to$ `Stim` wrapper).
