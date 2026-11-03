# BM-4: Window-Size Sample Complexity ($W$) & Causal Dynamic Regret Benchmark (`Q01`, `Q09`, `Q11`)

## Peer-Reviewed Protocol Provenance
- **Harper, Flammia, & Wallman**, *"Efficient learning of quantum noise"*, ***Nature Physics***, vol. 16, pp. 1184–1188 (2020).
- **Wagner et al.**, *"Pauli channels can be estimated from syndrome measurements in quantum error correction"*, ***Quantum***, vol. 6, p. 809 (2022).

## Window Scaling & Causal Commit Table

| Window $W$ (Rounds) | Duration @ $1\,\text{MHz}$ | Syndrome Dist. RMSE | Parameter RMSE | Causal Commit Round $t_{\text{commit}}$ | Instantaneous Regret |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64 | 64 $\mu\text{s}$ | 0.01791 | 0.01585 | 24441 | 0.00000 |
| 128 | 128 $\mu\text{s}$ | 0.01885 | 0.01038 | 23709 | 0.00000 |
| 256 | 256 $\mu\text{s}$ | 0.01217 | 0.01168 | 25000 | 0.00000 |
| 512 | 512 $\mu\text{s}$ | 0.00675 | 0.00607 | 24529 | 0.00000 |
| 1024 | 1024 $\mu\text{s}$ | 0.00422 | 0.00676 | 24642 | 0.00000 |
| 2048 | 2048 $\mu\text{s}$ | 0.00402 | 0.00627 | 25287 | 0.00000 |
| 4096 | 4096 $\mu\text{s}$ | 0.00186 | 0.00612 | 27221 | 0.00000 |
| 10000 | 10000 $\mu\text{s}$ | 0.00223 | 0.00938 | 33728 | 0.00000 |

## Artifacts
- [`bm4_window_causal_scaling.csv`](bm4_window_causal_scaling.csv)
- [`bm4_dynamic_regret.json`](bm4_dynamic_regret.json)
