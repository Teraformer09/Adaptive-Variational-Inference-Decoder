# AVID Baseline Architecture Specification (`AVID_EPJ_Patches`)

This directory contains the frozen contracts (`architecture/contracts/*.yaml`) and layer-by-layer mathematical and hardware specifications (`01_code_model` through `07_fault_management`) implementing `AVID_Architecture_Implementation_Policy_v1_0.pdf` (`A01–A09`) and `AVID_Implementation_Architecture_and_Repair_Plan_v0_1.docx` (`Q01–Q21`).

## Directory Index
- [`contracts/`](file:///home/tirtha/AVID_EPJ_Patches/architecture/contracts): Frozen YAML contracts for timing, packet layout, CSS code geometry, noise channel, Q7.8 LLR numerics, and fault handling.
- [`01_code_model/`](file:///home/tirtha/AVID_EPJ_Patches/architecture/01_code_model): Auto-generated CSS matrices (`Hx.json`, `Hz.json`), coordinates, logical operators, and shared Tanner tables.
- [`02_noise_model/`](file:///home/tirtha/AVID_EPJ_Patches/architecture/02_noise_model): Exact $2^m$-state correlated event signatures, observation model, and Jacobian SVD identifiability report.
- [`03_ingress/`](file:///home/tirtha/AVID_EPJ_Patches/architecture/03_ingress): Dual-consumer AXI4-Stream tap topology, 64-bit packet specification, and 4-state DMA ring ownership protocol.
- [`04_estimator/`](file:///home/tirtha/AVID_EPJ_Patches/architecture/04_estimator): Exact Joint-DP MAP cross-entropy likelihood, parameter schema, and Q7.8 quantization interface.
- [`05_realtime_control/`](file:///home/tirtha/AVID_EPJ_Patches/architecture/05_realtime_control): Fast/slow loop partition, causal timing budgets, and shadow/active epoch commit protocol.
- [`06_fpga_decoder/`](file:///home/tirtha/AVID_EPJ_Patches/architecture/06_fpga_decoder): Cycle-accurate latency contract $L(K) = 2 + 5K$, stream interfaces, and ZCU104 resource envelope.
- [`07_fault_management/`](file:///home/tirtha/AVID_EPJ_Patches/architecture/07_fault_management): 4-state deterministic fallback FSM and fault containment matrix.

## References
- **[R1]** A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," *Physical Review A*, vol. 86, no. 3, p. 032324, 2012.
- **[R2]** N. Liyanage et al., "Scalable quantum error correction for surface codes using FPGA," *arXiv:2301.08419*, 2023.
- **[R3]** J. P. Bonilla Ataides, D. K. Tuckett, S. D. Bartlett, S. T. Flammia, and B. J. Brown, "The XZZX surface code," *Nature Communications*, vol. 12, p. 2172, 2021.
- **[R4]** E. van den Berg, Z. K. Minev, A. Kandala, and K. Temme, "Probabilistic error cancellation with sparse Pauli-Lindblad models on noisy quantum processors," *Nature Physics*, vol. 19, pp. 1116–1121, 2023.
- **[R5]** S. T. Flammia and J. J. Wallman, "Efficient estimation of Pauli channels," *ACM Transactions on Quantum Computing*, vol. 1, no. 1, pp. 1–32, 2020.
- **[R6]** J. Chen and M. P. C. Fossorier, "Near optimum universal belief propagation based decoding of low-density parity check codes," *IEEE Transactions on Communications*, vol. 50, no. 3, pp. 406–414, 2002.
- **[R7]** D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization," *Mathematical Programming*, vol. 45, pp. 503–528, 1989.
- **[R8]** Y. Tomita and K. M. Svore, "Low-distance surface codes under realistic quantum noise," *Physical Review A*, vol. 90, no. 6, p. 062320, 2014.
- **[R9]** Xilinx Inc., *Zynq UltraScale+ Device Technical Reference Manual (UG1085)*, v2.4, 2023.
