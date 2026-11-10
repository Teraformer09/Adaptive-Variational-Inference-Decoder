# V1 Fair Shared-Stream Baseline Comparison (`B0`, `Tuned Static`, `B1`, `B2 PID`, `Joint-DP`, `Full AVID R2D2-WB`, `B4 Oracle`)

| Scenario (`1,536` Rounds, `seed=2026`) | `B0` Static | `Tuned Static` | `B1` Marginal | `B2` PID | `Joint-DP` Pair-Proj | **Full AVID V1 (`R2D2`)** | `B4` Oracle | Whole-Run Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stationary_noise` | `0.02995` | `0.02995` | `0.02995` | `0.02995` | `0.02995` | **`0.02995`** | `0.02995` | **`-0.00%`** |
| `slowly_drifting_noise` | `0.06641` | `0.06641` | `0.06836` | `0.03971` | `0.03971` | **`0.04883`** | `0.03971` | **`-26.47%`** |
| `correlated_noise` | `0.09570` | `0.09570` | `0.10091` | `0.05143` | `0.05143` | **`0.05143`** | `0.05143` | **`-46.26%`** |
| `burst_disturbance` | `0.07096` | `0.07096` | `0.07292` | `0.04883` | `0.04297` | **`0.04102`** | `0.04167` | **`-42.20%`** |
| `short_burst_60us_vs_10ms` | `0.08724` | `0.08724` | `0.08138` | `0.05339` | `0.04297` | **`0.04557`** | `0.04297` | **`-47.76%`** |
| `heldout_multi_pair_12_67` | `0.16211` | `0.16211` | `0.14062` | `0.08203` | `0.09896` | **`0.09701`** | `0.06706` | **`-40.16%`** |
| `heldout_spatial_3qubit_streak` | `0.12044` | `0.12044` | `0.12760` | `0.10677` | `0.10677` | **`0.10742`** | `0.10677` | **`-10.81%`** |
| `heldout_phenomenological_readout` | `0.12435` | `0.12435` | `0.12891` | `0.09635` | `0.08984` | **`0.08659`** | `0.08919` | **`-30.37%`** |

## Artifacts
- [`v1_shared_stream_baselines.json`](v1_shared_stream_baselines.json)
- [`v1_shared_stream_baselines.csv`](v1_shared_stream_baselines.csv)
