# Cross-Implementation Equivalence (Python FP64 vs. C++ Daemon vs. SystemVerilog RTL)

- **Python vs. C++ Joint-DP & L-BFGS**: Verified in `software/tests/test_baseline_suite.cpp` (`dp_vs_bruteforce_diff = 1.39e-17`, `grad_vs_fd_diff = 8.27e-10`).
- **C++ Reference vs. SystemVerilog RTL (`aa_qec_top.sv`)**: Verified across all 32 $X/Z$ syndromes and OSD modes `{0, 1, 2}` with `0` mismatches in `emulator/verilator/tb/standalone_rtl_tb.cpp`.
