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
