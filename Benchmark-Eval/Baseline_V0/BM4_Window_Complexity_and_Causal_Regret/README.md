# BM-4: Window-Size Sample Complexity ($W$) & Causal Dynamic Regret Benchmark (`Q01`, `Q09`, `Q11`)

## Peer-Reviewed Protocol Provenance
- **Harper, Flammia, & Wallman**, *"Efficient learning of quantum noise"*, ***Nature Physics***, vol. 16, pp. 1184–1188 (2020).
- **Wagner et al.**, *"Pauli channels can be estimated from syndrome measurements in quantum error correction"*, ***Quantum***, vol. 6, p. 809 (2022).

## Window Scaling & Causal Commit Table

| Window $W$ (Rounds) | Duration @ $1\,\text{MHz}$ | Syndrome Dist. RMSE | Parameter RMSE | Causal Commit Round $t_{\text{commit}}$ | Instantaneous Regret |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64 | 64 $\mu\text{s}$ | 0.01778 | 0.01432 | 26337 | 0.00000 |
| 128 | 128 $\mu\text{s}$ | 0.01888 | 0.01040 | 24017 | 0.00000 |
| 256 | 256 $\mu\text{s}$ | 0.01217 | 0.01170 | 25420 | 0.00000 |
| 512 | 512 $\mu\text{s}$ | 0.00670 | 0.00602 | 24230 | 0.00000 |
| 1024 | 1024 $\mu\text{s}$ | 0.00422 | 0.00510 | 24306 | 0.00000 |
| 2048 | 2048 $\mu\text{s}$ | 0.00408 | 0.00416 | 25291 | 0.00000 |
| 4096 | 4096 $\mu\text{s}$ | 0.00178 | 0.00419 | 27236 | 0.00000 |
| 10000 | 10000 $\mu\text{s}$ | 0.00218 | 0.00531 | 33545 | 0.00000 |

## Artifacts
- [`bm4_window_causal_scaling.csv`](bm4_window_causal_scaling.csv)
- [`bm4_dynamic_regret.json`](bm4_dynamic_regret.json)
