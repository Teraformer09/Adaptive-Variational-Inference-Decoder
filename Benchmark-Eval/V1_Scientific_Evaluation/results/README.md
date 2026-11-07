# AVID V1 Scientific Evaluation Results Summary

## 1. 10-Seed Paired Statistical Significance (`seeds = 2026..2035`, `1,536` Rounds/Seed)

| Scenario | Static B0 Mean LER | **AVID V1 Mean LER** | Whole-Run Mean Reduction | Burst-Window Mean Reduction | Paired $t$-Test $p$ | Wilcoxon $p$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correlated_noise` | `0.07285` | **`0.03411`** | **`-53.17%`** | **`-53.17%`** | `6.93e-07` | `1.95e-03` | `3.84` |
| `slowly_drifting_noise` | `0.04837` | **`0.02539`** | **`-47.51%`** | **`-59.74%`** | `1.05e-06` | `1.95e-03` | `3.66` |
| `burst_disturbance` | `0.04303` | **`0.02513`** | **`-41.61%`** | **`-57.94%`** | `1.57e-05` | `1.95e-03` | `2.64` |
| `short_burst_60us_vs_10ms` | `0.05918` | **`0.02904`** | **`-50.94%`** | **`-57.18%`** | `1.98e-07` | `1.95e-03` | `4.45` |

## 2. 6-Way V1 C++ Runtime & Hardware OSD Mode Ablation (`correlated_noise`)

| Variant | Controller Mode | OSD Mode | Core / AXI Cycles | Decoded LER | Relative Change vs. `B0` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1_Static_B0` | `static` | `1` | `17` / `22` | **`0.06966`** | **`+0.00%`** |
| `2_Marginal_LBFGS_B1` | `marginal_lbfgs` | `1` | `17` / `22` | **`0.06966`** | **`+0.00%`** |
| `3_Joint_DP_Pure_9Marginal_BP_17cyc` | `joint_dp_map` | `0` | `17` / `22` | **`0.14388`** | **`-106.55%`** |
| `4_Joint_DP_Pair_Projected_17cyc` | `joint_dp_proj` | `1` | `17` / `22` | **`0.04622`** | **`+33.65%`** |
| `5_Full_AVID_R2D2_WB_Inline_OSD_17cyc` | `avid_r2d2` | `1` | `17` / `22` | **`0.02865`** | **`+58.87%`** |
| `6_Full_AVID_R2D2_WB_PostBP_OSD_19cyc` | `avid_r2d2` | `2` | `19` / `24` | **`0.02865`** | **`+58.87%`** |
