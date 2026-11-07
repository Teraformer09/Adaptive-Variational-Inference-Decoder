# Numerical Validation of the 16-State Joint-DP Observation Model

- **Normalization across parameter domain**: Tested across 25 random parameter vectors $	heta \in [0.001, 0.20]^{21}$; $\left|\sum_{s=0}^{15} P_	heta(s) - 1ight| \le 4.44 	imes 10^{-16}$.
- **Exactness vs. $2^M$ brute-force enumeration**: Maximum absolute discrepancy across all 16 syndrome bins is $4.44 	imes 10^{-16}$ (`simulation/tests/test_joint_likelihood.py`).
