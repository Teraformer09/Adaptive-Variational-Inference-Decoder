# AVID Benchmark & Scientific Evaluation Hierarchy (`Benchmark-Eval/`)

To prevent mixing results from Python reference models with dual-emulator hardware/software co-simulation, `Benchmark-Eval/` is partitioned into three strict scopes:

1. [`Baseline_V0/`](Baseline_V0/): **V0 Algorithmic & Mathematical Reference Suite (`BM1–BM5`)** — Python/NumPy/PyTorch evaluations of 15-seed significance (`BM1`), 8-variant offline world-model/LoRA ablations (`BM2`), $d=3 	o d=5$ scaling & noisy readout (`BM3`), window complexity (`BM4`), and timing/memory models (`BM5`).
2. [`V1_Integration/`](V1_Integration/): **V1 Dual-Emulator Hardware/Software Integration (`M1–M6`)** — Verilator RTL (`aa_qec_top.sv`) + QEMU ARM Cortex-A9 (`avid_daemon`) co-simulation across `M1` standalone RTL, `M2` emulator builds, `M3` bridge transactions, `M4` 10-scenario suite, `M5` adaptive loop, and `M6` stress tests.
3. [`V1_Scientific_Evaluation/`](V1_Scientific_Evaluation/): **V1 Runtime Ablations & Statistical Analysis** — 6-way C++ runtime + RTL OSD mode ablation (`cfg_osd_mode = 0, 1, 2`), Wilson 95% confidence intervals, and whole-run vs. burst-window LER analysis.
4. [`common/`](common/): Shared metric definitions, Wilson CI / paired t-test utilities, plotting scripts, and evaluation protocols.
