# AVID V1 Scientific Evaluation Suite (`Benchmark-Eval/V1_Scientific_Evaluation/`)

- **[`baselines/`](baselines/README.md)**: Shared-stream evaluation of `B0 Static BP`, `B1 Marginal L-BFGS`, `Joint-DP Memoryless Pair-Projection`, and `Full AVID V1 (LSTM-64 R2D2-WB)`.
- **[`ablations/`](ablations/README.md)**: 6-Way C++ runtime and hardware OSD mode (`cfg_osd_mode = 0, 1, 2`) ablation study.
- **[`multi_seed/`](multi_seed/README.md)**: 10-seed (`2026..2035`) shared-stream evaluation across non-stationary and correlated noise scenarios.
- **[`statistical_analysis/`](statistical_analysis/README.md)**: Two-proportion $z$-tests, 95% Wilson CIs, paired $t$-tests, Wilcoxon signed-rank tests, and Holm–Bonferroni multiplicity correction.
- **[`results/`](results/README.md)**: Consolidated V1 scientific master summary (`v1_scientific_master_summary.json` & `v1_benchmark_comparison_table.md`).

# AVID V1 Scientific Evaluation Results Summary

## 1. 10-Seed Paired Statistical Significance (`seeds = 2026..2035`, `1,536` Rounds/Seed)

| Scenario | Static B0 Mean LER | **AVID V1 Mean LER** | Whole-Run Mean Reduction | Burst-Window Mean Reduction | Paired $t$-Test $p$ | Wilcoxon $p$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correlated_noise` | `0.08867` | **`0.05072`** | **`-42.80%`** | **`-42.80%`** | `1.17e-07` | `1.95e-03` | `4.72` |
| `slowly_drifting_noise` | `0.06322` | **`0.04395`** | **`-30.48%`** | **`-47.43%`** | `2.99e-05` | `1.95e-03` | `2.44` |
| `burst_disturbance` | `0.06048` | **`0.04850`** | **`-19.80%`** | **`-33.66%`** | `3.37e-04` | `1.95e-03` | `1.77` |
| `short_burst_60us_vs_10ms` | `0.07485` | **`0.04647`** | **`-37.92%`** | **`-45.54%`** | `9.82e-08` | `1.95e-03` | `4.82` |

## 2. 6-Way V1 C++ Runtime & Hardware OSD Mode Ablation (`correlated_noise`)

| Variant | Controller Mode | OSD Mode | Core / AXI Cycles | Decoded LER | Relative Change vs. `B0` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1_Static_B0` | `static` | `1` | `17` / `22` | **`0.09570`** | **`+0.00%`** |
| `2_Marginal_LBFGS_B1` | `marginal_lbfgs` | `1` | `17` / `22` | **`0.09570`** | **`+0.00%`** |
| `3_Joint_DP_Pure_9Marginal_BP_17cyc` | `joint_dp_map` | `0` | `17` / `22` | **`0.19531`** | **`-104.09%`** |
| `4_Joint_DP_Pair_Projected_17cyc` | `joint_dp_proj` | `1` | `17` / `22` | **`0.05273`** | **`+44.90%`** |
| `5_Full_AVID_R2D2_WB_Inline_OSD_17cyc` | `avid_r2d2` | `1` | `34` / `39` | **`0.05273`** | **`+44.90%`** |
| `6_Full_AVID_R2D2_WB_PostBP_OSD_19cyc` | `avid_r2d2` | `2` | `36` / `41` | **`0.05273`** | **`+44.90%`** |
