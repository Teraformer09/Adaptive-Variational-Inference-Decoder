# V1 10-Seed Shared-Stream C++ Runtime Statistical Evaluation (`seeds = 2026..2035`)

| Scenario ($N=10$ Seeds, `1,536` Rounds/Seed) | Static B0 LER ($\mu \pm \sigma$) | **AVID V1 LER ($\mu \pm \sigma$)** | Whole-Run Mean Reduction | Burst-Window Mean Reduction | Paired $t$-Test $p$ | Wilcoxon $p$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correlated_noise` | `0.08867 ± 0.00835` | **`0.05072 ± 0.00863`** | **`-42.80%`** | **`-42.80%`** | `1.17e-07` | `1.95e-03` | `4.72` |
| `slowly_drifting_noise` | `0.06322 ± 0.00618` | **`0.04395 ± 0.00787`** | **`-30.48%`** | **`-47.43%`** | `2.99e-05` | `1.95e-03` | `2.44` |
| `burst_disturbance` | `0.06048 ± 0.00550` | **`0.04850 ± 0.00752`** | **`-19.80%`** | **`-33.66%`** | `3.37e-04` | `1.95e-03` | `1.77` |
| `short_burst_60us_vs_10ms` | `0.07485 ± 0.00769` | **`0.04647 ± 0.00364`** | **`-37.92%`** | **`-45.54%`** | `9.82e-08` | `1.95e-03` | `4.82` |

## Artifacts
- [`v1_multiseed_10seeds_evaluation.json`](v1_multiseed_10seeds_evaluation.json)
- [`v1_multiseed_10seeds_trajectories.csv`](v1_multiseed_10seeds_trajectories.csv)
