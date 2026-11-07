# V1 Fair Shared-Stream Baseline Comparison (`B0`, `B1`, `Memoryless Joint-DP`, `Full AVID R2D2-WB`)

| Scenario (`1,536` Rounds, `seed=2026`) | `B0` Static BP LER | `B1` 4-Marginal L-BFGS LER | `Joint-DP` Memoryless Pair-Proj LER | **Full AVID V1 (`LSTM-64 R2D2-WB`) LER** | Whole-Run Reduction | Burst-Window Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `stationary_noise` | `0.00456` | `0.00456` | `0.00456` | **`0.00456`** | **`-0.00%`** | **`-0.00%`** |
| `slowly_drifting_noise` | `0.05013` | `0.05013` | `0.03646` | **`0.02148`** | **`-57.14%`** | **`-70.91%`** |
| `correlated_noise` | `0.06966` | `0.06966` | `0.04622` | **`0.02865`** | **`-58.88%`** | **`-58.88%`** |
| `burst_disturbance` | `0.04102` | `0.04102` | `0.02669` | **`0.02018`** | **`-50.79%`** | **`-59.32%`** |
| `short_burst_60us_vs_10ms` | `0.05859` | `0.05859` | `0.03190` | **`0.03190`** | **`-45.56%`** | **`-47.67%`** |

## Artifacts
- [`v1_shared_stream_baselines.json`](v1_shared_stream_baselines.json)
- [`v1_shared_stream_baselines.csv`](v1_shared_stream_baselines.csv)
