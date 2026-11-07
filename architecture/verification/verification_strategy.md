# AVID Verification Strategy

1. **Mathematical & Reference Verification (`simulation/tests/`)**:
   - CSS commutation $H_X H_Z^	op \equiv 0 \pmod 2$.
   - 16-state Joint-DP vs. $2^M$ brute-force enumeration (`max_diff = 4.44e-16`).
   - Analytic coset-inversion gradient vs. central finite differences (`max_diff = 1.23e-9`).
2. **Cycle-Accurate RTL & SVA Verification (`emulator/verilator/tb/standalone_rtl_tb.cpp`)**:
   - Exhaustive check across all 16 $X$-syndromes and 16 $Z$-syndromes across OSD modes `{0, 1, 2}` (`0` mismatches).
   - Formal SystemVerilog Assertions (`--assert`) verifying shadow bank isolation while `bp_ready_in == 0`.
3. **Dual-Emulator Co-Simulation (`scripts/run_all.sh` & `configs/system/v1_integration.yaml`)**:
   - 10-scenario suite executing `avid_daemon` under `qemu-arm -cpu cortex-a9` against `aa_qec_top.sv` under `Verilator 5.020`.
