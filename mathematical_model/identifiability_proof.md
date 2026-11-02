# Structural Identifiability & Jacobian SVD Rank Report (`A03 / Q04`)

## 1. Executive Summary of Rank Deficiency in the Legacy Marginal Model
In the legacy estimator (`noise_model.cpp` prior to `A02`/`A03` repair), the 12 nearest-neighbor $Z_i Z_j$ pair probabilities were folded into 9 effective qubit error probabilities $p_{\text{eff}}(v)$, and the likelihood summed 4 independent Bernoulli check marginals $q_0, q_1, q_2, q_3$. Because the observation map factored through 4 scalar check marginals, its $4 \times 21$ Jacobian $J_{\text{marginal}}$ had structural rank:
$$\operatorname{rank}(J_{\text{marginal}}) = 4 \le 4$$
leaving a $21 - 4 = 17$-dimensional unobservable nullspace.

## 2. Exact Joint Syndrome Distribution & Signature Orbit Analysis
Under the exact $2^m = 16$-state joint syndrome distribution $P_\theta(s)$, the column of the $16 \times 21$ Jacobian $J_{s, e} = \partial P_\theta(s) / \partial p_e$ for mechanism $e$ is:
$$J_{:, e} = \frac{P_\theta(s \oplus g_e) - P_\theta(s)}{1 - 2 p_e}$$
Consequently, two physical fault mechanisms $e_a, e_b$ have linearly dependent (collinear) Jacobian columns if and only if they share the same syndrome signature $g_{e_a} = g_{e_b} \in \{0, \dots, 15\}$, and any mechanism with $g_e = 0$ (stabilizer-equivalent or boundary-commuting fault) has an identically zero Jacobian column.

| Metric | Legacy 4-Marginal Model | Repaired 16-State Joint DP (21-param) | Repaired 16-State Joint DP (25-param) | Prior-Tethered MAP ($\lambda=0.5$) |
| :--- | :---: | :---: | :---: | :---: |
| Parameter Dimension ($E$) | 21 | 21 | 25 | 21 |
| Observation Dimension | 4 | 16 | 16 | 16 + 21 prior |
| Distinct Non-Zero Signatures | N/A | 9 | 9 | 9 |
| Numerical Jacobian / Fisher Rank | **4** | **9** | **9** | **21** |
| MAP Hessian Condition Number | $\infty$ (singular) | $\infty$ on gauge nullspace | $\infty$ on gauge nullspace | **234.05** |

## 3. Gauge Orbits (Syndrome Signature Equivalence Classes)
Across the 21 fault events (9 single-qubit $Z_i$ + 12 nearest-neighbor $Z_i Z_j$ pairs) on the canonical rotated $d=3$ X-check basis:
- **Zero-signature events ($g_e = 0$)**: `ZZ_3_6, ZZ_2_5` (events that commute with all 4 X-checks on the boundary).
- **Distinct non-zero syndrome signatures**: **9** distinct non-zero bitmasks in $\{1, \dots, 15\}$, yielding an exact unregularized Fisher rank of **9** (more than double the legacy rank of 4).
- **Quotient & MAP Resolution**: Within each signature equivalence class $\mathcal{O}_g = \{e : g_e = g\}$, the aggregate class probability $P_g = \bigoplus_{e \in \mathcal{O}_g} p_e$ is **strictly identifiable** from the joint syndrome histogram, while individual members within a multi-event orbit are uniquely resolved by the strictly convex Gaussian prior tether $\frac{\lambda}{2} \|\theta - \theta_{\text{prior}}\|_2^2$ (Hessian rank = **21 / 21**, condition number $\kappa = 234.05$).

## References
- **[R4]** E. van den Berg, Z. K. Minev, A. Kandala, and K. Temme, "Probabilistic error cancellation with sparse Pauli-Lindblad models on noisy quantum processors," *Nature Physics*, vol. 19, pp. 1116–1121, 2023.
- **[R5]** S. T. Flammia and J. J. Wallman, "Efficient estimation of Pauli channels," *ACM Transactions on Quantum Computing*, vol. 1, no. 1, pp. 1–32, 2020.
