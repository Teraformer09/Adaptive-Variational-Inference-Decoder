# AVID V1 Scientific Evaluation Suite (`Benchmark-Eval/V1_Scientific_Evaluation/`)

- **[`baselines/`](baselines/README.md)**: Shared-stream evaluation of `B0 Static BP`, `B1 Marginal L-BFGS`, `Joint-DP Memoryless Pair-Projection`, and `Full AVID V1 (LSTM-64 R2D2-WB)`.
- **[`ablations/`](ablations/README.md)**: 6-Way C++ runtime and hardware OSD mode (`cfg_osd_mode = 0, 1, 2`) ablation study.
- **[`multi_seed/`](multi_seed/README.md)**: 10-seed (`2026..2035`) shared-stream evaluation across non-stationary and correlated noise scenarios.
- **[`statistical_analysis/`](statistical_analysis/README.md)**: Two-proportion $z$-tests, 95% Wilson CIs, paired $t$-tests, Wilcoxon signed-rank tests, and Holm–Bonferroni multiplicity correction.
- **[`results/`](results/README.md)**: Consolidated V1 scientific master summary (`v1_scientific_master_summary.json` & `v1_benchmark_comparison_table.md`).

# AVID V1 Scientific Evaluation Results Summary

## 1. 10-Seed Paired Statistical Significance (`seeds = 2026..2035`, `1,536` Rounds/Seed)

| Scenario | Static B0 Mean LER | Joint-DP Projection Mean LER | **AVID V1 Mean LER** | Whole-Run Mean Reduction | Burst-Window Mean Reduction | Paired $t$-Test $p$ | Wilcoxon $p$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correlated_noise` | `0.08867` | `0.04909` (`-44.64%`) | **`0.04798`** | **`-45.89%`** | **`-45.89%`** | `8.93e-07` | `1.95e-03` | `3.73` |
| `slowly_drifting_noise` | `0.06322` | `0.04453` (`-29.56%`) | **`0.04245`** | **`-32.85%`** | **`-47.27%`** | `4.91e-06` | `1.95e-03` | `3.04` |
| `burst_disturbance` | `0.06048` | `0.04407` (`-27.13%`) | **`0.04160`** | **`-31.21%`** | **`-45.86%`** | `1.02e-06` | `1.95e-03` | `3.67` |
| `short_burst_60us_vs_10ms` | `0.07676` | `0.04655` (`-39.35%`) | **`0.04277`** | **`-44.27%`** | **`-51.55%`** | `1.19e-09` | `1.95e-03` | `7.95` |

## 2. 6-Way V1 C++ Runtime & Hardware OSD Mode Ablation (`correlated_noise`)

| Variant | Controller Mode | OSD Mode | Core / AXI Cycles | Decoded LER | Relative Change vs. `B0` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1_Static_B0` | `static` | `1` | `17` / `22` | **`0.09570`** | **`+0.00%`** |
| `2_Marginal_LBFGS_B1` | `marginal_lbfgs` | `1` | `17` / `22` | **`0.10091`** | **`-5.44%`** |
| `3_Joint_DP_Pure_9Marginal_BP_17cyc` | `joint_dp_map` | `0` | `17` / `22` | **`0.09961`** | **`-4.09%`** |
| `4_Joint_DP_Pair_Projected_17cyc` | `joint_dp_proj` | `1` | `17` / `22` | **`0.05143`** | **`+46.26%`** |
| `5_Full_AVID_R2D2_WB_Inline_OSD_17cyc` | `avid_r2d2` | `1` | `24` / `30` | **`0.05143`** | **`+46.26%`** |
| `6_Full_AVID_R2D2_WB_PostBP_OSD_19cyc` | `avid_r2d2` | `2` | `26` / `32` | **`0.05143`** | **`+46.26%`** |
