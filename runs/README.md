# Immutable Execution Runs (`runs/`)

Each subdirectory `runs/<run-id>/` is an immutable record of a single execution containing its `config.yaml`, `manifest.json`, `report.json`, and per-component evidence (`verilator/`, `qemu/`, `metrics/`, `bridge/` / `transactions/`, `waveforms/`).

- `avid_v1_integration_20261106_172251_s2026/`: Canonical 10-scenario Google Cloud Shell run with native `Verilator 5.020` (`verilator_c++_rtl`) and `qemu-arm -cpu cortex-a9`.
- `avid_v1_integration_20261105_203308_s2026/`, `avid_v1_rtl_regression_20261105_203306_s108/`, `avid_v1_smoke_20261105_203301_s42/`: Preserved historical V1 execution records.
