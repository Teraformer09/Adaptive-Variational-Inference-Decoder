# V1 Canonical Emulation Target: Xilinx Zynq-7020 SoC (`xc7z020clg484-1`)

## 1. Declared V1 Target Identity (Finding 1 Resolution)
The canonical V1 hardware/software co-simulation target is the **Xilinx Zynq-7020 SoC (`xc7z020clg484-1`)**:
- **Processing System (PS)**: Dual-core **ARM Cortex-A9** (`ARMv7-A`, `667 MHz`, `256 KB` OCM, `512 KB` L2 cache), cross-compiled with `arm-linux-gnueabihf-g++ -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard` and executed under `qemu-arm -cpu cortex-a9 -L /usr/arm-linux-gnueabihf`.
- **Programmable Logic (PL)**: Artix-7 FPGA fabric (`53,200` LUTs, `106,400` FFs, `220` DSP48E1 slices) simulated cycle-accurately at `250 MHz` (`4.0 ns` period) via native `Verilator 5.020` (`verilator_c++_rtl`).
- **Secondary Compatibility Build**: `emulator/qemu/build.sh` also compiles `avid_daemon_aarch64` (`aarch64-linux-gnu-g++`) for forward compatibility with V2 UltraScale+ targets. All V1 benchmark reports explicitly record which binary and simulator were executed.
