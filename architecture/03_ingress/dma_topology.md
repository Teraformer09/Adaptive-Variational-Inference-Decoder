# Syndrome Ingress & AXI-Stream Tap Topology (`A05 / Q11`, `SF1`)

## 1. Dual-Consumer Ingress Architecture
To prevent the ARM Cortex-A53 software daemon from polling the FPGA correction output buffer (`g_correction_buf`) in place of raw syndrome observations, the PL fabric splits the incoming 64-bit AXI4-Stream syndrome bus into two deterministic paths via `axis_ingress_tap.sv`:

1. **Fast Real-Time Decoding Path (`m_axis_bp`)**:
   Forwards every validated syndrome packet directly into the pipelined Normalized Min-Sum BP Engine (`bp_engine.sv`) with zero-cycle combinatorial pass-through (`L_tap = 0` cycles).
2. **Slow Estimation Observation Tap (`m_axis_dma`)**:
   Broadcasts the identical validated 64-bit syndrome packet into the AXI-DMA ring buffer (`4 slots x 64 rounds x 8 bytes = 2048 bytes`) so the ARM Cortex-A53 estimator observes genuine `(s_X, s_Z, seq_id)` syndrome packets without stalling the fast decoder.

## References
- **[R2]** N. Liyanage et al., "Scalable quantum error correction for surface codes using FPGA," *arXiv:2301.08419*, 2023.
- **[R9]** Xilinx Inc., *Zynq UltraScale+ Device Technical Reference Manual (UG1085)*, v2.4, 2023.
