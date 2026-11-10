# FPGA Decoder Resource Envelope: V1 Zynq-7020 (`xc7z020`) & V2 ZCU104 (`xczu7ev`)

## 1. Module-Level Post-Synthesis Resource Budget ($d=3$ Rotated Surface Code)

| Module | LUTs | Flip-Flops | DSP Slices | BRAM (36Kb) |
| :--- | :---: | :---: | :---: | :---: |
| `axis_ingress_tap.sv` & `axis_syndrome_rx.sv` | 64 | 72 | 0 | 0 |
| `weight_map_reg.sv` (Dual Q7.8 Banks + `0x058` OSD Config) | 185 | 340 | 0 | 0 |
| `check_node_minsum.sv` ($\times 4$) | 520 | 384 | 0 | 0 |
| `bp_engine.sv` (9 vars, 4 checks, OSD-0 stage) | 890 | 760 | 0 | 0 |
| **Total `aa_qec_top` ($d=3$)** | **1,659** | **1,556** | **0** | **0** |

## 2. Utilization by Declared Platform Target

| Target Scope | Device / Part | Available LUTs | LUT Utilization | Available FFs | FF Utilization | DSP Utilization |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **V1 Emulation Target** | **Xilinx Zynq-7020 (`xc7z020clg484-1`)** | `53,200` | **`3.12%`** | `106,400` | **`1.46%`** | `0 / 220 (0%)` |
| **V2 Physical Target** | **Xilinx ZCU104 (`xczu7ev-ffvc1156-2-e`)** | `230,400` | **`0.72%`** | `460,800` | **`0.34%`** | `0 / 1,728 (0%)` |

All arithmetic uses shift-add scaling $\alpha = 0.75$ (`x - (x >> 2)`) in signed 16-bit Q7.8 fixed point, requiring **zero DSP slices**.
