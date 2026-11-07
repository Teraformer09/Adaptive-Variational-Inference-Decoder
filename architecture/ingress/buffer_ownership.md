# DMA Ring Buffer Ownership & Cache Coherency Protocol (`A05 / Q11`, `SF1`)

## 1. 4-State Slot Ownership Machine
The 4-slot DMA ring buffer (`slot_id in {0, 1, 2, 3}`, 64 packets/slot) enforces a strict 4-state ownership transition table:
```
FREE  --[PL DMA acquires slot]-->  DMA_OWNED
DMA_OWNED  --[64 packets written + TLAST + cache invalidate]-->  CPU_READY
CPU_READY  --[ARM estimator claims slot]-->  CPU_OWNED
CPU_OWNED  --[Histogram accumulated + slot released]-->  FREE
```

## 2. Invariants
- The ARM CPU **never** reads a slot in `FREE` or `DMA_OWNED` state.
- The PL DMA engine **never** overwrites a slot in `CPU_READY` or `CPU_OWNED` state; if all 4 slots are busy, the tap asserts `overflow_warn` in `flags[1]` and increments the drop counter while `m_axis_bp` continues uninterrupted.

## References
- **[R9]** Xilinx Inc., *Zynq UltraScale+ Device Technical Reference Manual (UG1085)*, v2.4, 2023.
