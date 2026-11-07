# Component Boundaries

| Subsystem | Source Directory | Clock / Execution Domain | Primary Interface |
| :--- | :--- | :--- | :--- |
| Fast Decoder RTL | `firmware/rtl/` | `250 MHz` PL Fabric (`4.0 ns`) | AXI4-Stream (`s_axis_syndrome`, `m_axis_corr`) + AXI4-Lite (`s_axil_*`) |
| Verilator Bridge Server | `emulator/bridge/` & `emulator/verilator/` | Deterministic Cycle-Accurate Server | UNIX Domain Socket (`bridge_protocol.h`) |
| Slow-Loop Inference Daemon | `software/src/` | ARM Cortex-A9 (`qemu-arm`) / Cortex-A53 (`qemu-aarch64`) | `avid_hal.h` MMIO & Syndrome Stream |
| Reference Mathematical Oracles | `simulation/reference/` | Host Python (`FP64`) | JSON/NPZ Artifacts & Pytest Verification |
