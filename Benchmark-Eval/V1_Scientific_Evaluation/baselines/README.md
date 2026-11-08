# V1 Fair Shared-Stream Baseline Comparison (`B0`, `Tuned Static`, `B1`, `B2 PID`, `Joint-DP`, `Full AVID R2D2-WB`, `B4 Oracle`)

| Scenario (`1,536` Rounds, `seed=2026`) | `B0` Static | `Tuned Static` | `B1` Marginal | `B2` PID | `Joint-DP` Pair-Proj | **Full AVID V1 (`R2D2`)** | `B4` Oracle | Whole-Run Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stationary_noise` | `0.02995` | `0.02995` | `0.02995` | `0.02995` | `0.03190` | **`0.02995`** | `0.02995` | **`-0.00%`** |
| `slowly_drifting_noise` | `0.06641` | `0.06641` | `0.06641` | `0.03971` | `0.03971` | **`0.04948`** | `0.04102` | **`-25.49%`** |
| `correlated_noise` | `0.09570` | `0.09570` | `0.09570` | `0.05273` | `0.05273` | **`0.05273`** | `0.05273` | **`-44.90%`** |
| `burst_disturbance` | `0.07096` | `0.07096` | `0.07096` | `0.04883` | `0.05078` | **`0.04883`** | `0.04297` | **`-31.19%`** |
| `short_burst_60us_vs_10ms` | `0.08724` | `0.08724` | `0.08724` | `0.04557` | `0.06055` | **`0.05013`** | `0.04427` | **`-42.54%`** |
| `heldout_multi_pair_12_67` | `0.16211` | `0.16211` | `0.16211` | `0.13997` | `0.13997` | **`0.13997`** | `0.13997` | **`-13.65%`** |
| `heldout_spatial_3qubit_streak` | `0.12044` | `0.12044` | `0.12044` | `0.10677` | `0.10677` | **`0.11003`** | `0.10677` | **`-8.65%`** |
| `heldout_phenomenological_readout` | `0.12435` | `0.12435` | `0.12435` | `0.09635` | `0.09701` | **`0.11003`** | `0.08984` | **`-11.52%`** |

## Artifacts
- [`v1_shared_stream_baselines.json`](v1_shared_stream_baselines.json)
- [`v1_shared_stream_baselines.csv`](v1_shared_stream_baselines.csv)
