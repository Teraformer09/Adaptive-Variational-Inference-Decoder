# V1 10-Seed Shared-Stream C++ Runtime Statistical Evaluation (`seeds = 2026..2035`)

| Scenario ($N=10$ Seeds, `1,536` Rounds/Seed) | Static B0 LER ($\mu \pm \sigma$) | **AVID V1 LER ($\mu \pm \sigma$)** | Whole-Run Mean Reduction | Burst-Window Mean Reduction | Paired $t$-Test $p$ | Wilcoxon $p$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correlated_noise` | `0.08867 ± 0.00835` | **`0.05397 ± 0.00958`** | **`-39.14%`** | **`-39.14%`** | `3.74e-06` | `1.95e-03` | `3.15` |
| `slowly_drifting_noise` | `0.06322 ± 0.00618` | **`0.04954 ± 0.00667`** | **`-21.63%`** | **`-33.76%`** | `3.29e-04` | `1.95e-03` | `1.77` |
| `burst_disturbance` | `0.06048 ± 0.00550` | **`0.05091 ± 0.00666`** | **`-15.82%`** | **`-28.47%`** | `6.39e-04` | `1.95e-03` | `1.62` |
| `short_burst_60us_vs_10ms` | `0.07485 ± 0.00769` | **`0.04730 ± 0.00509`** | **`-36.81%`** | **`-43.80%`** | `8.06e-09` | `1.95e-03` | `6.41` |

## Artifacts
- [`v1_multiseed_10seeds_evaluation.json`](v1_multiseed_10seeds_evaluation.json)
- [`v1_multiseed_10seeds_trajectories.csv`](v1_multiseed_10seeds_trajectories.csv)
