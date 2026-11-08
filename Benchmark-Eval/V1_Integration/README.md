# AVID V1 Dual-Emulator Integration Evaluation Suite (`M1–M6`)

- **Canonical Run ID**: `avid_v1_integration_20261108_083417_s2026` (`Verilator 5.020` `verilator_c++_rtl` + `qemu-arm -cpu cortex-a9`)
- **[`M1_RTL_Correctness/`](M1_RTL_Correctness/README.md)**: 96 syndrome $\times$ OSD-mode vectors (`0` mismatches), SVA assertions, `SLVERR` invalid-offset rejection.
- **[`M2_Emulator_Validation/`](M2_Emulator_Validation/README.md)**: Zynq-7020 ARM Cortex-A9 primary target & ARM64 secondary target, 64-unit `int8` LSTM memory audit (`25.39 KB`).
- **[`M3_Bridge_Validation/`](M3_Bridge_Validation/README.md)**: Lockstep transaction audit (`0` unexpected rejections across `16,560` scenario transactions).
- **[`M4_End_to_End_Scenarios/`](M4_End_to_End_Scenarios/README.md)**: 10-scenario suite (`15,360` total rounds) with 95% Wilson CIs and burst-window LER reductions.
- **[`M5_Adaptive_Loop_Validation/`](M5_Adaptive_Loop_Validation/README.md)**: 16-state Joint-DP NLL vs. legacy marginal NLL, 9-orbit observable RMSE (`0.0081–0.0148`), and commit reconciliation.
- **[`M6_Performance_and_Stress/`](M6_Performance_and_Stress/README.md)**: $K=1..5$ cycle latency scaling, 16-round fast burst detector (`9 <= 20` rounds), and 57,344-round stress run.
