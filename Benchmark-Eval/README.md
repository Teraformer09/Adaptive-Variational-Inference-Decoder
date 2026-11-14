# AVID Peer-Reviewed 5-Benchmark Evaluation Suite (`Benchmark-Eval/Baseline_V0`)

This directory contains the 5 finalized, peer-reviewed evaluation benchmarks closing `Q01–Q21` and `Gate 7–Gate 8` of the AVID Architecture Implementation Policy:

1. **[`BM1_MultiSeed_Statistical_Significance/`](Baseline_V0/BM1_MultiSeed_Statistical_Significance/README.md)** (`Q07`, `Q08`, `Q16`, `Gate 7`)
   - Protocol: **Bausch et al. (AlphaQubit), *Nature* 635, 834–840 (2024)** & **Spitz et al., *Adv. Quantum Technol.* 1, 1800012 (2018)**.
   - 15-seed shared-input evaluation separating **Full-Horizon Mean LER** from **Peak-Burst Window LER**, with paired $t$-tests, Cohen's $d$, and exact two-sided Wilcoxon signed-rank $p$-values ($p < 0.001$).
2. **[`BM2_Architectural_and_Loss_Ablation/`](Baseline_V0/BM2_Architectural_and_Loss_Ablation/README.md)** (`Q03`, `Q04`, `Q05`, `Q14`, `Q15`)
   - Protocol: **van den Berg et al. (IBM Quantum), *Nature Physics* 19, 1116–1121 (2023)** & **Wagner et al., *Quantum* 6, 809 (2022)**.
   - 7-stage architectural and loss ablation from `A0` (Static BP) and `A1` (4-Check Marginal L-BFGS) through `A3` (16-State Joint-DP + Pair-Aware Triangle LLR Projection) to `A6` (Full Weighted-Bellman R2D2 + Rank-4 LoRA).
3. **[`BM3_Distance_Scaling_and_Noisy_Readout/`](Baseline_V0/BM3_Distance_Scaling_and_Noisy_Readout/README.md)** (`Q02`, `Q17`, `Q19`)
   - Protocol: **Google Quantum AI (Willow Processor), *Nature* 638, 920–926 (2025)** & **Roffe et al., *Phys. Rev. Research* 2, 043423 (2020)**.
   - Code-distance error suppression factor $\Lambda_{3 \to 5}$ (`d=3 [[9, 1, 3]]` vs. `d=5 [[25, 1, 5]]`) and 25-parameter noisy syndrome measurement ($d_t = s_t \oplus s_{t-1}$) evaluation.
4. **[`BM4_Window_Complexity_and_Causal_Regret/`](Baseline_V0/BM4_Window_Complexity_and_Causal_Regret/README.md)** (`Q01`, `Q09`, `Q11`, `SF4`, `SF6`)
   - Protocol: **Harper, Flammia, & Wallman, *Nature Physics* 16, 1184–1188 (2020)**.
   - Window sample complexity $W \in \{64, \dots, 10000\}$ rounds ($O(W^{-1/2})$ Cramér–Rao convergence), causal commit round $t_{\text{commit}}$, and dynamic regret $\mathcal{R}_T / T$ under bounded-variation drift $V_T$.
5. **[`BM5_Hardware_Latency_WCET_and_Fallback/`](Baseline_V0/BM5_Hardware_Latency_WCET_and_Fallback/README.md)** (`Q06`, `Q10`, `Q12`, `Q13`, `Q18`, `Q19`, `SF2`, `SF10`)
   - Protocol: **Barber et al. (Riverlane), *Nature Electronics* 8, 84–91 (2025)** & **Liyanage et al., *PRX Quantum* (2023–2024)**.
   - Deterministic FPGA cycle matrix $L(K) = 2 + 5K$ ($K \in \{1..5\}$, `17 cycles = 68.0 ns` @ $K=3$), ARM slow-loop p50/p95/p99/max WCET, controller `float32`/`int8` memory & SHA-256 manifest, and fault-injection fallback verification.

## Run All 5 Benchmarks
```bash
bash run_benchmarks_cloudshell.sh
```

---

## Seven-Stage Persistent-PD Multi-Generator Noise & Statistical Validation (`Benchmark-Eval/V1_Seven_Stage_Noise_Validation`)

In addition to `Baseline_V0`, [`V1_Seven_Stage_Noise_Validation/`](V1_Seven_Stage_Noise_Validation/README.md) validates the **Seven-Stage Persistent-PD Multi-Generator Noise Generation and Statistical Validation Framework**:
- **Stage 1 (`stage1_pd_validation/`)**: 16 reusable truncated-Gaussian Probability Distribution (PD) profiles (`configs/noise_framework/pd_profiles_16.yaml`).
- **Stage 2 (`stage2_assignment_validation/` & `generator_assignment_validation/`)**: Persistent 100-second window PD assignment ($P_j^{(w)} \sim \operatorname{Categorical}(\pi_1 \dots \pi_{16})$) and isolated per-generator RNG streams.
- **Stage 3 (`stage3_generator_validation/` & `sampling_synchronization/`)**: Synchronized 10 Hz (`100 ms`) multi-generator stepping (`1,000` samples per 100s window) and spatially correlated 21-mechanism fault generation.
- **Stage 4 (`stage4_composition_validation/` & `composition_validation/`)**: Physical Pauli modulo-2 XOR mask composition ($\mathbf{e}_n^{\mathrm{composite}} = \bigoplus_j \mathbf{e}_{j,n}$) and independent Bernoulli probability composition ($p_{\mathrm{combined}} = 1 - \prod_j (1 - p_j)$).
- **Stage 5 (`stage5_syndrome_validation/`)**: Dynamic `MeasurementNoiseGenerator` (`measurement_noise_01`) and `MeasurementModel` with independent readout errors, Markov sticky temporal correlation, detector dropout validity masks ($v_{c,t} \in \{0,1\}$), and observation delay (`event_timestamps` vs. `delivery_timestamps`).
- **Stage 6 (`stage6_dataset_validation/`)**: Stratified trajectory generation (`datasets/seven_stage_v1/`) with disjoint train/val/test configurations.
- **Stage 7 (`stage7_statistical_validation/`, `distribution_validation/`, `identifiability_analysis/`, `decoder_evaluation/`)**: Exact 32-state joint syndrome-logical distribution $P(s, \ell \mid \theta)$, conditional posterior $P(\ell \mid s, \theta)$, Bayes-optimal Oracle LER $\mathrm{LER}_{\mathrm{oracle}}(\theta)$, syndrome-aliasing analysis, and 6-decoder shared-trajectory evaluation.

```bash
PYTHONPATH=. python3 Benchmark-Eval/V1_Seven_Stage_Noise_Validation/run_seven_stage_eval.py
```
