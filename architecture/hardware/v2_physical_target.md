# V2 Intended Physical Target: Zynq UltraScale+ MPSoC (`ZCU104` / `Kria K26`)

## 1. Declared V2 Target Scope
Physical board deployment (V2 scope) targets the **Xilinx Zynq UltraScale+ MPSoC** family (`ZCU104` `xczu7ev-ffvc1156-2-e` and `Kria K26` `xck26-sfvc784-2lv-c`) featuring a quad-core **ARM Cortex-A53** (`aarch64`) PS and UltraScale+ PL fabric.

## 2. Boundary Between V1 Evidence and V2 Scope
- V1 verifies RTL cycle accuracy (`Verilator 5.020`), AXI4-Lite/AXI4-Stream protocol compliance, SVA assertions, and ARM guest execution (`qemu-arm -cpu cortex-a9`).
- Physical Vivado place-and-route timing closure (`WNS >= 0`), physical AXI DMA loopback, and hardware power/WCET measurements belong strictly to V2 physical validation.
