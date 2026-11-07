# AVID Canonical Configuration Directory

- [`schema.yaml`](schema.yaml): Strict JSON/YAML validation schema for experiment runs.
- [`system/`](system/): Top-level V1 system run configurations (`v1_integration.yaml`, `v1_full_benchmark.yaml`, `v1_smoke.yaml`).
- [`rtl/`](rtl/): Verilator simulation and RTL regression configurations (`verilator.yaml`, `rtl_regression.yaml`).
- [`emulator/`](emulator/): Canonical ARMv7 Cortex-A9 (`qemu_armv7.yaml`) and secondary ARM64 Cortex-A53 (`qemu_aarch64.yaml`) guest configurations.
- [`hardware/`](hardware/): V1 Zynq-7020 emulation target (`v1_emulation.yaml`) and V2 UltraScale+ ZCU104 target (`v2_target_board.yaml`).
- [`code/`](code/): Surface code distance definitions (`d3_reference.yaml`, `d5_simulation.yaml`).
- [`scenarios/`](scenarios/): Individual noise and fault-injection scenario specifications.
