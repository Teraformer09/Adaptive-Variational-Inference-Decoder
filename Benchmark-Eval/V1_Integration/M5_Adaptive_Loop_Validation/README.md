# M5: Causal Adaptive Loop, 16-State Joint-DP NLL & 9-Orbit Identifiability Audit

- **Blind Estimation Audit**: `solver_contains_scenario_conditionals = False`, `r2d2_controller_contains_scenario_conditionals = False`
- **Identifiability Separation**: `orbit_9class_rmse` measures data-driven syndrome orbit recovery (`rank(J) = 9`); `param_21_prior_regularized_rmse` reflects prior-anchored MAP point selection (`rank(H_MAP) = 21/21`).

| Scenario | Legacy 4-Marginal NLL | **16-State Joint-DP NLL** | **9-Orbit Observable RMSE** | 21-Param MAP RMSE | Commit Reconciliation (`init + adapt - rst = hw`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `stationary_noise` | `0.8868` | **`0.8061`** | **`0.0081`** | `0.0044` | `1 + 6 - 0 = 7` (`True`) |
| `slowly_drifting_noise` | `1.1525` | **`1.0827`** | **`0.0143`** | `0.0084` | `1 + 9 - 0 = 10` (`True`) |
| `correlated_noise` | `1.0605` | **`0.9911`** | **`0.0154`** | `0.0090` | `1 + 7 - 0 = 8` (`True`) |
| `burst_disturbance` | `0.6456` | **`0.6098`** | **`0.0152`** | `0.0096` | `1 + 8 - 0 = 9` (`True`) |
| `delayed_weight_commit` | `1.1349` | **`1.0578`** | **`0.0088`** | `0.0053` | `1 + 7 - 0 = 8` (`True`) |
| `reset_recovery` | `1.0593` | **`0.9780`** | **`0.0162`** | `0.0120` | `1 + 10 - 2 = 9` (`True`) |
| `busy_decoder` | `0.9967` | **`0.9147`** | **`0.0123`** | `0.0075` | `1 + 6 - 0 = 7` (`True`) |
| `variable_k_adaptation` | `1.1231` | **`1.0385`** | **`0.0107`** | `0.0061` | `1 + 7 - 0 = 8` (`True`) |
| `fallback_safety_trigger` | `1.0792` | **`0.9589`** | **`0.0627`** | `0.0571` | `1 + 9 - 0 = 10` (`True`) |
| `short_burst_60us_vs_10ms` | `0.7839` | **`0.7545`** | **`0.0153`** | `0.0109` | `1 + 7 - 0 = 8` (`True`) |

## Artifacts
- [`m5_adaptive_loop_and_identifiability.json`](m5_adaptive_loop_and_identifiability.json)
- [`m5_nll_and_orbit_rmse.csv`](m5_nll_and_orbit_rmse.csv)
