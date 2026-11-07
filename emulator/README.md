# AVID Dual-Emulator Co-Simulation Environment (`emulator/`)

- [`verilator/`](verilator/): Cycle-accurate SystemVerilog RTL simulator (`Verilator 5.020`) compiling `aa_qec_top.sv`.
- [`qemu/`](qemu/): Cross-compiled ARM guest environment (`qemu-arm -cpu cortex-a9` primary V1 target; `qemu-aarch64-static` secondary V2 compatibility target).
- [`bridge/`](bridge/): Deterministic lockstep UNIX-socket AXI4-Lite + AXI4-Stream co-simulation bridge.
