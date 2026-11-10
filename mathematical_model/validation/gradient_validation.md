# Analytic Coset-Inversion Gradient Validation

- **Analytic vs. Central Finite-Difference Gradient**: Using step size $\epsilon = 10^{-7}$, the maximum absolute discrepancy across all 21 physical fault parameters is $1.23 \times 10^{-9}$ (`simulation/tests/test_joint_likelihood.py`).
- **Singularity Guard**: Near $p_e \to 0.5$, the coset inversion falls back to a leave-one-out prefix/suffix DP table (`max_diff = 0.0`).
