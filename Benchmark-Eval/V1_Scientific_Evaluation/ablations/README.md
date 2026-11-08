# V1 6-Way C++ Runtime & Hardware OSD Mode Ablation (`correlated_noise`)

| Ablation Variant | Controller Mode | OSD Mode (`0x058`) | Core / AXI Cycles | Z Failures (Decoded LER) | Relative Change vs. `B0` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1_Static_B0` | `static` | `1` | `17` / `22` | `147` (**`0.09570`**) | **`+0.00%`** |
| `2_Marginal_LBFGS_B1` | `marginal_lbfgs` | `1` | `17` / `22` | `147` (**`0.09570`**) | **`+0.00%`** |
| `3_Joint_DP_Pure_9Marginal_BP_17cyc` | `joint_dp_map` | `0` | `17` / `22` | `300` (**`0.19531`**) | **`-104.09%`** |
| `4_Joint_DP_Pair_Projected_17cyc` | `joint_dp_proj` | `1` | `17` / `22` | `81` (**`0.05273`**) | **`+44.90%`** |
| `5_Full_AVID_R2D2_WB_Inline_OSD_17cyc` | `avid_r2d2` | `1` | `34` / `39` | `81` (**`0.05273`**) | **`+44.90%`** |
| `6_Full_AVID_R2D2_WB_PostBP_OSD_19cyc` | `avid_r2d2` | `2` | `36` / `41` | `81` (**`0.05273`**) | **`+44.90%`** |

## Artifacts
- [`v1_6way_runtime_ablation.json`](v1_6way_runtime_ablation.json)
- [`v1_6way_runtime_ablation.csv`](v1_6way_runtime_ablation.csv)
