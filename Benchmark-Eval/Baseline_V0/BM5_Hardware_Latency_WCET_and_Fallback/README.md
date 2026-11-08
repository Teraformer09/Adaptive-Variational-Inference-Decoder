# BM-5: Hardware Iteration Latency ($K=1..5$), ARM WCET, Memory Manifest & Fault-Fallback Benchmark (`Q06`, `Q10`, `Q12`, `Q13`, `Q18`, `Q19`)

## Peer-Reviewed Protocol Provenance
- **Barber et al. (Riverlane)**, *"A real-time, scalable, fast and resource-efficient decoder for a quantum computer"*, ***Nature Electronics***, vol. 8, pp. 84–91 (2025).
- **Liyanage et al.**, *"Scalable quantum error correction for surface codes using FPGA"*, ***PRX Quantum*** / ***USENIX*** (2023–2024).

## FPGA Fast-Path Cycle Latency Matrix (@ $250\,\text{MHz}$ PL Clock)

| BP Iterations $K$ | Fused Cycles $L(K)=2+5K$ | Fused Latency (ns) | Legacy Cycles $2+6K$ | Syndrome Satisfaction | Decoded LER |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $K=1$ | **7 cycles** | **28.0 ns** | 8 cycles | 1.0000 | **0.0570** |
| $K=2$ | **12 cycles** | **48.0 ns** | 14 cycles | 1.0000 | **0.0570** |
| $K=3$ | **17 cycles** | **68.0 ns** | 20 cycles | 1.0000 | **0.0570** |
| $K=4$ | **22 cycles** | **88.0 ns** | 26 cycles | 1.0000 | **0.0570** |
| $K=5$ | **27 cycles** | **108.0 ns** | 32 cycles | 1.0000 | **0.0570** |

## Artifacts
- [`bm5_rtl_cycle_matrix.csv`](bm5_rtl_cycle_matrix.csv)
- [`bm5_controller_memory_manifest.json`](bm5_controller_memory_manifest.json)
- [`bm5_hardware_wcet_and_fallback.json`](bm5_hardware_wcet_and_fallback.json)
