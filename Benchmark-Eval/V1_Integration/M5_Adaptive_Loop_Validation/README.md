# M5: Causal Adaptive Loop, 16-State Joint-DP NLL & 9-Orbit Identifiability Audit

- **Blind Estimation Audit**: `solver_contains_scenario_conditionals = False`, `r2d2_controller_contains_scenario_conditionals = False`
- **Identifiability Separation**: `orbit_9class_rmse` measures data-driven syndrome orbit recovery (`rank(J) = 9`); `param_21_prior_regularized_rmse` reflects prior-anchored MAP point selection (`rank(H_MAP) = 21/21`).

| Scenario | Legacy 4-Marginal NLL | **16-State Joint-DP NLL** | **9-Orbit Observable RMSE** | 21-Param MAP RMSE | Commit Reconciliation (`init + adapt - rst = hw`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `stationary_noise` | `0.5219` | **`0.4678`** | **`0.0081`** | `0.0046` | `1 + 5 - 0 = 6` (`True`) |
| `slowly_drifting_noise` | `0.9123` | **`0.8163`** | **`0.0103`** | `0.0076` | `1 + 5 - 0 = 6` (`True`) |
| `correlated_noise` | `0.8458` | **`0.7510`** | **`0.0118`** | `0.0093` | `1 + 5 - 0 = 6` (`True`) |
| `burst_disturbance` | `0.4379` | **`0.4080`** | **`0.0142`** | `0.0124` | `1 + 5 - 0 = 6` (`True`) |
| `delayed_weight_commit` | `0.8497` | **`0.7960`** | **`0.0103`** | `0.0081` | `1 + 6 - 0 = 7` (`True`) |
| `reset_recovery` | `0.7955` | **`0.7322`** | **`0.0115`** | `0.0097` | `1 + 5 - 1 = 5` (`True`) |
| `busy_decoder` | `0.9563` | **`0.8760`** | **`0.0087`** | `0.0113` | `1 + 5 - 0 = 6` (`True`) |
| `variable_k_adaptation` | `1.0694` | **`0.9199`** | **`0.0107`** | `0.0100` | `1 + 5 - 0 = 6` (`True`) |
| `fallback_safety_trigger` | `0.9641` | **`0.8808`** | **`0.0116`** | `0.0092` | `1 + 5 - 0 = 6` (`True`) |
| `short_burst_60us_vs_10ms` | `0.5853` | **`0.5223`** | **`0.0148`** | `0.0103` | `1 + 5 - 0 = 6` (`True`) |

## Artifacts
- [`m5_adaptive_loop_and_identifiability.json`](m5_adaptive_loop_and_identifiability.json)
- [`m5_nll_and_orbit_rmse.csv`](m5_nll_and_orbit_rmse.csv)
