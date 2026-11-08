# BM-1: 15-Seed Shared-Input LER & Peak-vs-Full Window Statistical Benchmark (`Q07`, `Q08`, `Q16`)

## Peer-Reviewed Protocol Provenance
- **Bausch et al. (Google DeepMind & Google Quantum AI)**, *"Learning high-accuracy error decoding for quantum processors"* (**AlphaQubit**), ***Nature***, vol. 635, pp. 834–840 (2024).
- **Spitz et al.**, *"Adaptive weight estimator for quantum error correction using time-dependent syndrome correlations"*, ***Adv. Quantum Technol.***, vol. 1, 1800012 (2018).

## Results Across $N = 15$ Independent Seeds per Regime (`10,000` Eval Rounds/Seed)

| Regime | `B0` Full LER | `AVID_DP` Full LER | **`WB_R2D2` Full LER** | `B0` Peak-Burst LER | **`WB_R2D2` Peak-Burst LER** | Paired $t$-Test $p$ | Wilcoxon $p$ ($N=15$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R1_stationary_baseline` | 0.0522 | 0.0522 | **0.0522** | 0.0575 | **0.0575** | 1.00e+00 | 1.00e+00 |
| `R2_tls_hotspot_q4_q5` | 0.0591 | 0.0591 | **0.0591** | 0.0654 | **0.0654** | 1.00e+00 | 1.00e+00 |
| `R3_zz_crosstalk_burst` | 0.2096 | 0.2096 | **0.2096** | 0.2261 | **0.2261** | 1.00e+00 | 1.00e+00 |
| `R4_boundary_drift` | 0.0589 | 0.1277 | **0.1253** | 0.0649 | **0.1356** | 2.11e-20 | 6.53e-04 |
| `R5_combined_stress` | 0.1074 | 0.0663 | **0.0632** | 0.1157 | **0.0696** | 3.53e-19 | 6.52e-04 |

## Artifacts
- [`bm1_multiseed_ler_trajectories.csv`](bm1_multiseed_ler_trajectories.csv)
- [`bm1_statistical_significance.json`](bm1_statistical_significance.json)
