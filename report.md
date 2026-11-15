# Layer 5 Report — 250 MHz SystemVerilog FPGA RTL Decoder

## 1. Layer Scope & Isolation
This branch contains **only Layer 5** of AVID (frozen baseline split from commit `54da68d`):
- SystemVerilog RTL modules (`firmware/rtl/`: `aa_qec_top.sv`, `bp_engine.sv`, `check_node_minsum.sv`, `weight_map_reg.sv`, `axis_syndrome_rx.sv`, `axis_correction_tx.sv`)
- FPGA timing constraints (`firmware/constraints/aa_qec.xdc`) & SystemVerilog testbench (`firmware/tb/tb_aa_qec_top.sv`)
- Standalone M1 RTL driver & testbench (`emulator/verilator/tb/verilator_rtl_driver.h`, `standalone_rtl_tb.cpp`)

## 2. Inputs (IP) & Outputs (OP) Validated
- **IP**: `s_axis_syndrome` (`s_x[3:0]`, `s_z[3:0]`, `tvalid`, `tlast`), AXI4-Lite shadow Q7.8 registers (`0x000..0x048`), `CFG_MAX_ITER` (`0x054`), `cfg_osd_mode` (`0x058`)
- **OP**: `m_axis_corr` (`corr_x[8:0]`, `corr_z[8:0]`), atomic `bp_ready_in`-gated shadow-to-active swap (`epoch_count`), deterministic core cycle formula `L_core(K) = 2 + 5K = {7, 12, 17, 22, 27}` (`68.0 ns` @ `K=3`, `250 MHz`), SLVERR rejection, backpressure & mid-decode reset recovery

## 3. Execution & Log
- Runner script: `scripts/run_layer5.sh`
- Test suite: `emulator/verilator/tb/standalone_rtl_tb.cpp` + `tests/test_layer5_ip_op.py`
- Saved run log: [`logs/layer5_validation.log`](logs/layer5_validation.log)

```bash
bash scripts/run_layer5.sh
```
