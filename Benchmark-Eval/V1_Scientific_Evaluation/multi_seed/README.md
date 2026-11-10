# V1 10-Seed Shared-Stream C++ Runtime Statistical Evaluation (`seeds = 2026..2035`)

| Scenario ($N=10$ Seeds, `1,536` Rounds/Seed) | Static B0 LER ($\mu \pm \sigma$) | Joint-DP Projection LER ($\mu \pm \sigma$) | **AVID V1 LER ($\mu \pm \sigma$)** | Whole-Run Mean Reduction | Burst-Window Mean Reduction | Paired $t$-Test $p$ | Wilcoxon $p$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correlated_noise` | `0.08867 ± 0.00835` | `0.04909 ± 0.00638` (`-44.64%`) | **`0.04798 ± 0.00692`** | **`-45.89%`** | **`-45.89%`** | `8.93e-07` | `1.95e-03` | `3.73` |
| `slowly_drifting_noise` | `0.06322 ± 0.00618` | `0.04453 ± 0.00490` (`-29.56%`) | **`0.04245 ± 0.00535`** | **`-32.85%`** | **`-47.27%`** | `4.91e-06` | `1.95e-03` | `3.04` |
| `burst_disturbance` | `0.06048 ± 0.00550` | `0.04407 ± 0.00621` (`-27.13%`) | **`0.04160 ± 0.00361`** | **`-31.21%`** | **`-45.86%`** | `1.02e-06` | `1.95e-03` | `3.67` |
| `short_burst_60us_vs_10ms` | `0.07676 ± 0.00654` | `0.04655 ± 0.00541` (`-39.35%`) | **`0.04277 ± 0.00626`** | **`-44.27%`** | **`-51.55%`** | `1.19e-09` | `1.95e-03` | `7.95` |

## Artifacts
- [`v1_multiseed_10seeds_evaluation.json`](v1_multiseed_10seeds_evaluation.json)
- [`v1_multiseed_10seeds_trajectories.csv`](v1_multiseed_10seeds_trajectories.csv)
