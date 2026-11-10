# V1 6-Way C++ Runtime & Hardware OSD Mode Ablation (`correlated_noise`)

| Ablation Variant | Controller Mode | OSD Mode (`0x058`) | Core / AXI Cycles | Z Failures (Decoded LER) | Relative Change vs. `B0` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1_Static_B0` | `static` | `1` | `17` / `22` | `147` (**`0.09570`**) | **`+0.00%`** |
| `2_Marginal_LBFGS_B1` | `marginal_lbfgs` | `1` | `17` / `22` | `155` (**`0.10091`**) | **`-5.44%`** |
| `3_Joint_DP_Pure_9Marginal_BP_17cyc` | `joint_dp_map` | `0` | `17` / `22` | `153` (**`0.09961`**) | **`-4.09%`** |
| `4_Joint_DP_Pair_Projected_17cyc` | `joint_dp_proj` | `1` | `17` / `22` | `79` (**`0.05143`**) | **`+46.26%`** |
| `5_Full_AVID_R2D2_WB_Inline_OSD_17cyc` | `avid_r2d2` | `1` | `24` / `30` | `79` (**`0.05143`**) | **`+46.26%`** |
| `6_Full_AVID_R2D2_WB_PostBP_OSD_19cyc` | `avid_r2d2` | `2` | `26` / `32` | `79` (**`0.05143`**) | **`+46.26%`** |

## Artifacts
- [`v1_6way_runtime_ablation.json`](v1_6way_runtime_ablation.json)
- [`v1_6way_runtime_ablation.csv`](v1_6way_runtime_ablation.csv)
