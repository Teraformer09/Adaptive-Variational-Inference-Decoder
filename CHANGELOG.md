# Changelog

## [1.0.0-v1-emulation] — 2026-11-07

### Added
- Reorganized repository into canonical directories (`architecture/`, `mathematical_model/`, `firmware/`, `software/`, `simulation/`, `emulator/`, `configs/`, `tests/`, `Benchmark-Eval/`, `reports/`, `runs/`, `logs/`, `checkpoints/`, `datasets/`, `artifacts/`, `tools/`, `scripts/`, `dependencies/`, `figures/`).
- Implemented 16-round sliding-subwindow fast burst detector (`avid_fast_burst_detector_t` in `software/src/control/causal_scheduler.h` and `software/src/inference/avid_daemon.cpp`) with 0.35 syndrome-density threshold, 4-round debounce, and 20-round worst-case trigger latency.
- Added native `Verilator 5.020` (`verilator_c++_rtl`) + `qemu-arm` (`cortex-a9`) 10-scenario integration evidence (`avid_v1_integration_20261106_172251_s2026`).
- Added configurable hardware OSD modes (`cfg_osd_mode = 0, 1, 2` at register `0x058`) separating pure 9-marginal-LLR BP (`17` core cycles), inline pair-aware OSD-0 (`17` core cycles), and post-BP OSD-0 stage (`19` core cycles).

### Fixed
- Reconciled V1 emulation target (Zynq-7020 / ARM Cortex-A9) vs. V2 physical target (ZCU104 / Kria K26 / ARM Cortex-A53) across all documentation and configs.
- Distinguished 9 data-identifiable syndrome signature orbits (`rank(J) = 9`) from 12 gauge null directions (`dim ker(J) = 12`) and 21 prior-regularized MAP parameters (`rank(H_MAP) = 21`).
- Consolidated duplicate mathematical and architecture documentation and removed machine-specific `file://` links.
