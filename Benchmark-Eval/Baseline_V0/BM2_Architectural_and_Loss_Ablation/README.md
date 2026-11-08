# BM-2: 7-Stage Architectural & Loss Component Ablation Benchmark (`Q03`, `Q04`, `Q05`, `Q14`, `Q15`)

## Peer-Reviewed Protocol Provenance
- **van den Berg, Minev, Kandala, & Temme (IBM Quantum)**, *"Probabilistic error cancellation with sparse Pauli–Lindblad models on noisy quantum processors"*, ***Nature Physics***, vol. 19, pp. 1116–1121 (2023).
- **Wagner, Kampermann, Bruß, & Kliesch**, *"Pauli channels can be estimated from syndrome measurements in quantum error correction"*, ***Quantum***, vol. 6, p. 809 (2022).

## Ablation Table

| Regime | Ablation Stage | Parameter RMSE | Syndrome $D_{\text{KL}}$ | Decoded LER |
| :--- | :--- | :---: | :---: | :---: |
| `R3_zz_crosstalk_burst` | `A0_Static_BP_B0` | 0.02303 | 0.09654 | **0.1934** |
| `R3_zz_crosstalk_burst` | `A1_Legacy_Marginal_4Check_B1` | 0.02262 | 0.07890 | **0.1934** |
| `R3_zz_crosstalk_burst` | `A2_Joint_DP_No_Pair_Projection` | 0.00991 | 0.00017 | **0.1934** |
| `R3_zz_crosstalk_burst` | `A3_AVID_Joint_DP_Pair_Aware` | 0.00991 | 0.00017 | **0.1934** |
| `R3_zz_crosstalk_burst` | `A4_R2D2_Unweighted_No_BurnIn` | 0.01296 | 0.00681 | **0.1934** |
| `R3_zz_crosstalk_burst` | `A5_Weighted_Bellman_R2D2_No_LoRA` | 0.00991 | 0.00017 | **0.1934** |
| `R3_zz_crosstalk_burst` | `A6_Full_WB_R2D2_plus_Rank4_LoRA` | 0.00991 | 0.00017 | **0.1934** |
| `R3_zz_crosstalk_burst` | `A7_Oracle_BP_B4` | 0.00000 | 0.00035 | **0.1069** |
| `R5_combined_stress` | `A0_Static_BP_B0` | 0.02930 | 0.20829 | **0.1023** |
| `R5_combined_stress` | `A1_Legacy_Marginal_4Check_B1` | 0.02868 | 0.17762 | **0.1023** |
| `R5_combined_stress` | `A2_Joint_DP_No_Pair_Projection` | 0.01428 | 0.00017 | **0.1023** |
| `R5_combined_stress` | `A3_AVID_Joint_DP_Pair_Aware` | 0.01428 | 0.00017 | **0.0608** |
| `R5_combined_stress` | `A4_R2D2_Unweighted_No_BurnIn` | 0.01695 | 0.01250 | **0.1023** |
| `R5_combined_stress` | `A5_Weighted_Bellman_R2D2_No_LoRA` | 0.01428 | 0.00017 | **0.0608** |
| `R5_combined_stress` | `A6_Full_WB_R2D2_plus_Rank4_LoRA` | 0.01428 | 0.00017 | **0.0608** |
| `R5_combined_stress` | `A7_Oracle_BP_B4` | 0.00000 | 0.00021 | **0.0593** |

## Artifacts
- [`bm2_ablation_study.csv`](bm2_ablation_study.csv)
- [`bm2_ablation_report.json`](bm2_ablation_report.json)
