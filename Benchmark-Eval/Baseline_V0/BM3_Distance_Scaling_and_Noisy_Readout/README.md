# BM-3: Code Distance Scaling ($d=3 \to d=5$, $\Lambda_{3 \to 5}$) & 25-Parameter Noisy-Readout Benchmark (`Q02`, `Q17`, `Q19`)

## Peer-Reviewed Protocol Provenance
- **Google Quantum AI and Collaborators**, *"Quantum error correction below the surface code threshold"* (**Willow Processor**), ***Nature***, vol. 638, pp. 920–926 (2025).
- **Roffe, White, Burton, & Campbell**, *"Decoding across the quantum low-density parity-check code landscape"* (**BP+OSD**), ***Phys. Rev. Research***, vol. 2, 043423 (2020).

## Distance Scaling ($d=3$ `[[9, 1, 3]]` vs. $d=5$ `[[25, 1, 5]]`)

| Base Physical Error $p$ | $d=3$ Static LER | **$d=3$ AVID LER** | $d=5$ Static LER | **$d=5$ AVID LER** | **Suppression $\Lambda_{3 \to 5}$ (AVID)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.005 | 0.0107 | **0.0083** | 0.0017 | **0.0017** | **5.00x** |
| 0.010 | 0.0197 | **0.0150** | 0.0070 | **0.0070** | **2.14x** |
| 0.015 | 0.0327 | **0.0353** | 0.0097 | **0.0097** | **3.65x** |
| 0.020 | 0.0453 | **0.0387** | 0.0097 | **0.0097** | **4.00x** |
| 0.025 | 0.0580 | **0.0490** | 0.0127 | **0.0127** | **3.87x** |
| 0.030 | 0.0667 | **0.0703** | 0.0133 | **0.0133** | **5.28x** |

## Artifacts
- [`bm3_d3_vs_d5_scaling.csv`](bm3_d3_vs_d5_scaling.csv)
- [`bm3_noisy_readout_25param.json`](bm3_noisy_readout_25param.json)
