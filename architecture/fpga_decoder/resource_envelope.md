# Zynq UltraScale+ ZCU104 (`xczu7ev`) Resource Envelope (`A06 / Q06`)

## 1. Post-Synthesis Resource Budget ($d=3$ Rotated Surface Code)
| Module | LUTs | Flip-Flops | DSP48E2 | BRAM (36Kb) |
| :--- | :---: | :---: | :---: | :---: |
| `axis_ingress_tap.sv` | 64 | 72 | 0 | 0 |
| `weight_map_reg.sv` | 185 | 340 | 0 | 0 |
| `check_node_minsum.sv` ($\times 4$) | 520 | 384 | 0 | 0 |
| `bp_engine.sv` (9 vars, 4 checks) | 890 | 760 | 0 | 0 |
| **Total `aa_qec_top` ($d=3$)** | **1,659 (< 0.8%)** | **1,556 (< 0.4%)** | **0 (0%)** | **0 (0%)** |

All arithmetic uses shift-add $\alpha = 0.75$ (`x - (x >> 2)`) in signed 16-bit Q7.8 fixed point, requiring **zero DSP slices**.

## References
- **[R2]** N. Liyanage et al., *arXiv:2301.08419*, 2023.
- **[R9]** Xilinx Inc., *UG1085*, v2.4, 2023.
