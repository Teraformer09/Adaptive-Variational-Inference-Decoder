# M5: Causal Adaptive Loop, 16-State Joint-DP NLL & 9-Orbit Identifiability Audit

- **Blind Estimation Audit**: `solver_contains_scenario_conditionals = False`, `r2d2_controller_contains_scenario_conditionals = False`
- **Identifiability Separation**: `orbit_9class_rmse` measures data-driven syndrome orbit recovery (`rank(J) = 9`); `param_21_prior_regularized_rmse` reflects prior-anchored MAP point selection (`rank(H_MAP) = 21/21`).

| Scenario | Legacy 4-Marginal NLL | **16-State Joint-DP NLL** | **9-Orbit Observable RMSE** | 21-Param MAP RMSE | Commit Reconciliation (`init + adapt - rst = hw`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `stationary_noise` | `0.8137` | **`0.7722`** | **`0.0090`** | `0.0047` | `1 + 5 - 0 = 6` (`True`) |
| `slowly_drifting_noise` | `1.1142` | **`1.0330`** | **`0.0106`** | `0.0096` | `1 + 5 - 0 = 6` (`True`) |
| `correlated_noise` | `1.0672` | **`0.9747`** | **`0.0092`** | `0.0124` | `1 + 5 - 0 = 6` (`True`) |
| `burst_disturbance` | `0.6456` | **`0.6098`** | **`0.0150`** | `0.0113` | `1 + 4 - 0 = 5` (`True`) |
| `delayed_weight_commit` | `1.1677` | **`1.0706`** | **`0.0096`** | `0.0124` | `1 + 4 - 0 = 5` (`True`) |
| `reset_recovery` | `1.0936` | **`1.0041`** | **`0.0089`** | `0.0111` | `1 + 5 - 1 = 5` (`True`) |
| `busy_decoder` | `0.9369` | **`0.8849`** | **`0.0122`** | `0.0125` | `1 + 5 - 0 = 6` (`True`) |
| `variable_k_adaptation` | `1.1314` | **`1.0503`** | **`0.0102`** | `0.0115` | `1 + 5 - 0 = 6` (`True`) |
| `fallback_safety_trigger` | `0.9556` | **`0.8434`** | **`0.0788`** | `0.0791` | `1 + 5 - 0 = 6` (`True`) |
| `short_burst_60us_vs_10ms` | `0.7981` | **`0.7473`** | **`0.0211`** | `0.0148` | `1 + 17 - 0 = 18` (`True`) |

## Artifacts
- [`m5_adaptive_loop_and_identifiability.json`](m5_adaptive_loop_and_identifiability.json)
- [`m5_nll_and_orbit_rmse.csv`](m5_nll_and_orbit_rmse.csv)
