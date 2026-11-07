# AVID Benchmark & Scientific Evaluation Hierarchy (`Benchmark-Eval/`)

To prevent mixing results from Python reference models with dual-emulator hardware/software co-simulation, `Benchmark-Eval/` is partitioned into four clean scopes:

1. **[`Baseline_V0/`](Baseline_V0/)**: **V0 Algorithmic & Mathematical Reference Suite (`BM1–BM5`)** — Python/NumPy/PyTorch evaluations of 15-seed significance (`BM1`), 8-variant offline world-model/LoRA ablations (`BM2`), $d=3 \to d=5$ scaling & noisy readout (`BM3`), window complexity (`BM4`), and timing/memory models (`BM5`).
2. **[`V1_Integration/`](V1_Integration/README.md)**: **V1 Dual-Emulator Hardware/Software Integration (`M1–M6`)** — Verilator 5.020 RTL (`aa_qec_top.sv`) + QEMU ARM Cortex-A9 (`avid_daemon`) co-simulation across:
   - [`M1_RTL_Correctness/`](V1_Integration/M1_RTL_Correctness/README.md): Exhaustive $16\,X + 16\,Z$ syndromes $\times$ 3 OSD modes (`96` vectors, `0` mismatches), SVA assertions, `SLVERR` invalid-offset rejection.
   - [`M2_Emulator_Validation/`](V1_Integration/M2_Emulator_Validation/README.md): Zynq-7020 ARM Cortex-A9 primary target & ARM64 secondary target, 64-unit `int8` LSTM memory audit (`25.39 KB` vs `256 KB` OCM).
   - [`M3_Bridge_Validation/`](V1_Integration/M3_Bridge_Validation/README.md): Deterministic lockstep Unix-socket transaction audit (`16,560` scenario transactions, `0` unexpected rejections).
   - [`M4_End_to_End_Scenarios/`](V1_Integration/M4_End_to_End_Scenarios/README.md): 10-scenario suite (`15,360` total rounds) with 95% Wilson CIs and whole-run vs. peak-burst-window LER reductions.
   - [`M5_Adaptive_Loop_Validation/`](V1_Integration/M5_Adaptive_Loop_Validation/README.md): 16-state Joint-DP NLL vs. legacy 4-marginal NLL, 9-orbit observable RMSE (`0.0081–0.0148`), and commit reconciliation (`1 + 6 - 0 = 7`).
   - [`M6_Performance_and_Stress/`](V1_Integration/M6_Performance_and_Stress/README.md): $K=1..5$ cycle latency scaling (`68 ns` core / `88 ns` AXI at $K=3$), 16-round fast burst detector (`9 <= 20` rounds), and 57,344-round stress run.
3. **[`V1_Scientific_Evaluation/`](V1_Scientific_Evaluation/README.md)**: **V1 C++ Runtime Ablations, 10-Seed Statistical Evaluation & Multiplicity Audit**:
   - [`baselines/`](V1_Scientific_Evaluation/baselines/README.md): Shared-stream evaluation of `B0 Static BP`, `B1 Marginal L-BFGS`, `Joint-DP Memoryless Pair-Projection`, and `Full AVID V1 (LSTM-64 R2D2-WB)`.
   - [`ablations/`](V1_Scientific_Evaluation/ablations/README.md): 6-Way C++ runtime and hardware OSD mode (`cfg_osd_mode = 0, 1, 2`) ablation study.
   - [`multi_seed/`](V1_Scientific_Evaluation/multi_seed/README.md): 10-seed (`2026..2035`, `15,360` rounds per scenario) paired C++ runtime evaluation (`p = 1.98e-07` to `1.57e-05`, Wilcoxon `p = 1.95e-03`).
   - [`statistical_analysis/`](V1_Scientific_Evaluation/statistical_analysis/README.md): Two-proportion $z$-tests, 95% Wilson CIs, and Holm–Bonferroni step-down adjusted $p$-values across all 10 scenarios.
   - [`results/`](V1_Scientific_Evaluation/results/README.md): Consolidated V1 scientific master summary (`v1_scientific_master_summary.json` & `v1_benchmark_comparison_table.md`).
4. **[`common/`](common/)**: Shared QEC metric calculators (`qec_metrics.py`), hypothesis testing & Holm–Bonferroni utilities (`paired_significance.py`), and V1 causal blind-estimation protocol checks (`v1_acceptance_gates.py`).

## Reproducing the V1 Evaluation Suites

```bash
python3 Benchmark-Eval/V1_Integration/run_v1_integration.py
python3 Benchmark-Eval/V1_Scientific_Evaluation/run_v1_scientific_eval.py
```
