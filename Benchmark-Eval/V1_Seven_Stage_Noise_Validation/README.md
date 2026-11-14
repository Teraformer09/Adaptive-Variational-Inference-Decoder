# AVID Seven-Stage Persistent-PD Multi-Generator Noise & Validation Suite

## Stage Verification Summary

| Stage | Subsystem | Key Verified Metric | Status |
| :--- | :--- | :--- | :---: |
| **Stage 1** | 16-Profile PD Library (`pd_profiles_16.yaml`) | `16/16` truncated-Gaussian profiles validated (`25,000` samples/PD) | **`PASS`** |
| **Stage 2** | 100s Persistent PD Assignment & Lifecycle | `2,500` samples over `250s` (`3` windows); RNG stream isolation verified | **`PASS`** |
| **Stage 3** | Spatially Correlated 21-Mechanism Generator | Dominant pair correlation `r = 0.960` | **`PASS`** |
| **Stage 4** | Physical Pauli XOR & Probability Composition | `Z XOR Z = I` cancellation & Bernoulli union error `0.00208` | **`PASS`** |
| **Stage 5** | Dynamic Measurement-Noise Subsystem | `2,500` synchronized 10 Hz samples; sticky Markov, dropout masks & delay verified | **`PASS`** |
| **Stage 6** | Stratified Dataset Generation (`seven_stage_v1`) | `13` trajectories (`26000` samples); `Theta_train ∩ Theta_test = ∅` | **`PASS`** |
| **Stage 7** | Statistical Distribution & 6-Decoder Evaluation | 16-state Joint-DP TVD `0.0318`, Oracle LER `0.1712` | **`PASS`** |

## Held-Out Shared-Trajectory 6-Decoder Benchmark (`8000` Rounds Across `4` Held-Out Seeds)

| Decoder / Controller | Total Logical Failures | Logical Error Rate (LER) |
| :--- | :---: | :---: |
| `1_static_bp` (`B0`) | `2008` | `0.25100` |
| `2_marginal_lbfgs` (`B1`) | `2008` | `0.25100` |
| `3_deterministic_pid` (`B2`) | `2008` | `0.25100` |
| `4_memoryless_joint_dp_proj` | `2008` | `0.25100` |
| **`5_recurrent_avid_r2d2`** | **`1983`** | **`0.24788`** |
| `6_oracle_bp_and_coset` (`B4` Oracle Floor) | `1533` | `0.19162` |
