# AVID Canonical Mathematical Model

This directory is the **single canonical location** for all mathematical derivations, proofs, and numerical validation records in AVID.

## Contents
- [`00_mathematical_modeling_index.md`](00_mathematical_modeling_index.md): Master notation and equation index.
- [`01_surface_code_and_syndrome_model.md`](01_surface_code_and_syndrome_model.md): Rotated $d=3$ CSS surface code algebra and Tanner graph geometry.
- [`02_correlated_pauli_noise_model.md`](02_correlated_pauli_noise_model.md): 21-parameter single-qubit and adjacent correlated Pauli fault channel.
- [`03_joint_dp_likelihood_and_gradient.md`](03_joint_dp_likelihood_and_gradient.md): Exact 16-state XOR-convolution DP and analytic coset-inversion gradient.
- [`04_parameter_estimation_and_lbfgs.md`](04_parameter_estimation_and_lbfgs.md): Two-pass projected L-BFGS MAP estimation.
- [`05_identifiability_and_gauge_structure.md`](05_identifiability_and_gauge_structure.md): 9 observable syndrome orbits, 12 gauge null directions, and regularized MAP Hessian rank.
- [`06_llr_quantization_and_pair_projection.md`](06_llr_quantization_and_pair_projection.md): Pair-aware logical-triangle LLR projection, signed Q7.8 quantization, and BP/OSD-0 decoding.
- [`07_causal_control_dma_and_baselines.md`](07_causal_control_dma_and_baselines.md): Three-tier causal control, DMA ring buffering, and fair baselines `B0–B4`.
- [`08_weighted_bellman_r2d2_and_world_model.md`](08_weighted_bellman_r2d2_and_world_model.md): Weighted-Bellman R2D2 recurrent controller and V0 world-model ablations.
- [`09_baseline_v0_and_v1_v2_roadmap.md`](09_baseline_v0_and_v1_v2_roadmap.md): V0 baseline establishment and V1/V2 roadmap alignment.
- [`proofs/`](proofs/): Formal identifiability proof and mathematical model validation.
- [`validation/`](validation/): Numerical, gradient, and cross-implementation validation summaries.
