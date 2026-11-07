# V1 10-Seed Shared-Stream C++ Runtime Statistical Evaluation (`seeds = 2026..2035`)

| Scenario ($N=10$ Seeds, `1,536` Rounds/Seed) | Static B0 LER ($\mu \pm \sigma$) | **AVID V1 LER ($\mu \pm \sigma$)** | Whole-Run Mean Reduction | Burst-Window Mean Reduction | Paired $t$-Test $p$ | Wilcoxon $p$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correlated_noise` | `0.07285 ± 0.00820` | **`0.03411 ± 0.00639`** | **`-53.17%`** | **`-53.17%`** | `6.93e-07` | `1.95e-03` | `3.84` |
| `slowly_drifting_noise` | `0.04837 ± 0.00452` | **`0.02539 ± 0.00385`** | **`-47.51%`** | **`-59.74%`** | `1.05e-06` | `1.95e-03` | `3.66` |
| `burst_disturbance` | `0.04303 ± 0.00592` | **`0.02513 ± 0.00397`** | **`-41.61%`** | **`-57.94%`** | `1.57e-05` | `1.95e-03` | `2.64` |
| `short_burst_60us_vs_10ms` | `0.05918 ± 0.00791` | **`0.02904 ± 0.00596`** | **`-50.94%`** | **`-57.18%`** | `1.98e-07` | `1.95e-03` | `4.45` |

## Artifacts
- [`v1_multiseed_10seeds_evaluation.json`](v1_multiseed_10seeds_evaluation.json)
- [`v1_multiseed_10seeds_trajectories.csv`](v1_multiseed_10seeds_trajectories.csv)
