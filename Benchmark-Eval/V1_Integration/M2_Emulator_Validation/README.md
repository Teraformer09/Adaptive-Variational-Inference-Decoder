# M2: ARM Cortex-A9 / ARM64 QEMU Emulator & Memory Footprint Validation

## Declared Target Matrix & Memory Footprint

| Target Role | SoC / FPGA Part | Guest CPU / Compiler | Emulator | 64-Unit LSTM `int8` | 64-Unit LSTM `fp32` | L-BFGS Workspace | Fits `256 KB` OCM? |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **V1 Primary Target** | Zynq-7020 (`xc7z020clg484-1`) | ARM Cortex-A9 (`arm-linux-gnueabihf-g++ -mcpu=cortex-a9`) | `qemu-arm -cpu cortex-a9` | `19,744 B` (`19.28 KB`) | `78,976 B` (`77.12 KB`) | `2,688 B` (`2.63 KB`) | **`YES (8.6%)`** |
| **V1 Secondary Target** | UltraScale+ (`xczu7ev` / `xck26`) | ARM Cortex-A53 (`aarch64-linux-gnu-g++`) | `qemu-aarch64-static` | `19,744 B` (`19.28 KB`) | `78,976 B` (`77.12 KB`) | `2,688 B` (`2.63 KB`) | **`YES (8.6%)`** |

## Artifacts
- [`m2_emulator_build_and_elf_audit.json`](m2_emulator_build_and_elf_audit.json)
- [`m2_memory_and_toolchain_matrix.csv`](m2_memory_and_toolchain_matrix.csv)
