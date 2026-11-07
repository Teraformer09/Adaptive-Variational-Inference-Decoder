# BM5: Latency, Slow-Loop Execution Time & Fallback Evaluation

> **Provenance Notice (Finding 11 Resolution)**: This directory separates four distinct timing provenance categories:
> 1. **Cycle-Accurate RTL Simulation (`Verilator 5.020`)**: `17` BP core cycles (`68.0 ns`) and `22` total AXI4-Stream cycles (`88.0 ns`) at $K=3$, `250 MHz`.
> 2. **ARM Cortex-A9 QEMU Emulation (`qemu-arm -cpu cortex-a9`)**: Two-pass 16-state Joint-DP L-BFGS + 64-unit LSTM R2D2-WB executes in `~6.02 ms` mean (`60.17%` of the `10 ms` slow-loop budget).
> 3. **Analytical Warm-Started Single-Pass Estimate**: `~18.0 us` for warm-started 5-iteration single-pass L-BFGS on Cortex-A53.
> 4. **Physical Board WCET**: Reserved for V2 physical board deployment (`ZCU104` / `Kria K26`).
