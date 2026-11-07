# AXI4-Stream & AXI4-Lite Hardware Interface Specification (`SF1`, `SF5`)

## 1. Ports on `aa_qec_top.sv`
- **`s_axis_syndrome_*` (AXI4-Stream Slave, 64-bit)**: Accepts canonical 64-bit syndrome packets (`s_x[3:0]`, `s_z[3:0]`, `flags[7:0]`, `epoch_id[15:0]`, `seq_id[31:0]`).
- **`m_axis_dma_*` (AXI4-Stream Master, 64-bit)**: Tapped syndrome stream routed to the PS DMA ring buffer.
- **`m_axis_corr_*` (AXI4-Stream Master, 16-bit)**: Decoded 9-bit Pauli correction vector (`tdata[8:0] = correction_out`, `tvalid = correction_valid`).
- **`s_axi_ctrl_*` (AXI4-Lite Slave, 7-bit addr, 32-bit data)**: Maps `weight_map_reg.sv` (`0x000..0x04C`).

## References
- **[R9]** Xilinx Inc., *Zynq UltraScale+ Device Technical Reference Manual (UG1085)*, v2.4, 2023.
