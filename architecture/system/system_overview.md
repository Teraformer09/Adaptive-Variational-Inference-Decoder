# AVID System Overview

**AVID (Adaptive Variational Inference Decoder)** is a dual-timescale hardware/software co-designed quantum error correction (QEC) decoding architecture for the rotated $d=3$ CSS surface code (`[[9, 1, 3]]`).

## Core Architectural Partition

1. **Fast Hardware Decode Path (FPGA / Verilator RTL, `250 MHz`, `4.0 ns` clock)**:
   - Deterministic Min-Sum Belief Propagation (`bp_engine.sv`) + configurable Ordered Statistics Decoding stage (`cfg_osd_mode` at `0x058`).
   - Core latency: $L_{\mathrm{core}}(K) = 2 + 5K$ cycles (`17` cycles = `68.0 ns` at $K=3$ for `cfg_osd_mode in {0, 1}`; `19` cycles = `76.0 ns` for `cfg_osd_mode = 2`).
   - End-to-end AXI4-Stream latency: $L_{\mathrm{AXI}}(K) = 7 + 5K$ cycles (`22` cycles = `88.0 ns` at $K=3$, accounting for `3` ingress + `17` core + `2` egress cycles).
2. **Fast Sliding-Subwindow Burst Detector (`causal_scheduler.cpp` / `avid_daemon.cpp`)**:
   - 16-round sliding subwindow (`subwindow_rounds = 16`), syndrome-density threshold `0.35`, 4-round debounce (`worst_case_trigger_latency_rounds = 20` rounds = `20 us` at `1 MHz`).
3. **Slow Adaptive Estimation & Control Path (ARM PS / QEMU Guest)**:
   - Dual-basis ($X$ and $Z$) 16-state exact XOR-convolution Dynamic Programming (DP) + analytical coset-inversion gradient + two-pass L-BFGS MAP estimator (`noise_model.cpp`, `lbfgs_solver.cpp`).
   - 64-unit Dueling LSTM R2D2 controller (`r2d2_lstm_controller.h` / `r2d2_runtime.cpp`, `19,744` parameters = `19.28 KB` `int8` / `77.12 KB` `fp32`) producing 6 discrete control actions ($K \in \{3, 4, 5\}$, $\gamma_{\mathrm{pair}}$, temporal smoothing $\beta$, recalibrate, fallback) and 9 per-qubit gain residuals committed atomically across `weight_map_reg.sv`.
