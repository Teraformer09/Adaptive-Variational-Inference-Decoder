# 04 — Structural Identifiability, Gauge Orbits, and Fisher Information Rank (`A03 / Q04`)

## 1. Structural Jacobian and Fisher Rank Theorem
Let $J(\theta) \in \mathbb{R}^{2^m \times E}$ denote the sensitivity Jacobian with entries $J_{s, e}(\theta) = \frac{\partial P_\theta(s)}{\partial p_e}$, and let $\mathcal{I}(\theta) \in \mathbb{R}^{E \times E}$ denote the Fisher Information Matrix of the multinomial syndrome observation model:
$$\mathcal{I}(\theta) = \sum_{s=0}^{2^m - 1} \frac{1}{P_\theta(s)} \nabla_\theta P_\theta(s) \nabla_\theta P_\theta(s)^\top = J(\theta)^\top \operatorname{diag}\!\left(P_\theta(s)^{-1}\right) J(\theta)$$

By Theorem 1 in [`03_Exact_Joint_DP_Likelihood_and_Coset_Gradients.md`](docs/03_Exact_Joint_DP_Likelihood_and_Coset_Gradients.md), the $e$-th column of $J(\theta)$ takes the form:
$$J_{:, e}(\theta) = \frac{1}{1 - 2 p_e} \left( \Pi_{g_e} P_\theta - P_\theta \right)$$
where $\Pi_{g_e}$ is the involution permutation matrix $(\Pi_{g_e} v)(s) = v(s \oplus g_e)$. Three structural properties follow immediately:
1. **Unobservable Zero-Signature Nullspace**: Every fault mechanism $e$ with $g_e = \mathbf{0}$ satisfies $\Pi_{\mathbf{0}} = I$ and therefore $J_{:, e}(\theta) \equiv \mathbf{0}$.
2. **Collinearity within Signature Equivalence Classes (Gauge Orbits)**: Any two distinct fault mechanisms $e_a, e_b \in \mathcal{E}$ that produce the same syndrome signature $g_{e_a} = g_{e_b} = g \neq \mathbf{0}$ satisfy $(1 - 2 p_{e_a}) J_{:, e_a}(\theta) = (1 - 2 p_{e_b}) J_{:, e_b}(\theta)$, so their Jacobian columns are strictly collinear at every point $\theta \in (0, 0.5)^E$.
3. **Exact Rank Equals Distinct Non-Zero Signatures**: For generic interior parameters $\theta \in (0, 0.5)^E$, the difference vectors $\{\Pi_g P_\theta - P_\theta : g \in \mathcal{G}_{\neq 0}\}$ over the set of distinct non-zero signatures $\mathcal{G}_{\neq 0} = \{g_e \neq \mathbf{0} : e \in \mathcal{E}\}$ are linearly independent. Consequently:
   $$\operatorname{rank}(J(\theta)) = \operatorname{rank}(\mathcal{I}(\theta)) = |\mathcal{G}_{\neq 0}|$$

---

## 2. Gauge Orbit Decomposition on the Canonical $d=3$ Surface Code
In [`identifiability_svd.py`](simulation/reference/noise_model/identifiability_svd.py) and [`event_signatures.json`](architecture/02_noise_model/event_signatures.json), the $E = 21$ physical mechanisms (9 single-qubit $Z_i$ faults and 12 nearest-neighbor $Z_i Z_j$ pair faults) on the canonical rotated $d=3$ $X$-stabilizer code partition into 10 gauge orbits $\mathcal{O}_g = \{e \in \mathcal{E} : g_e = g\}$:

- **Zero-Signature Orbit ($\mathcal{O}_{\texttt{0x0}}$, 2 mechanisms)**: $\{ZZ_{3,6}, ZZ_{2,5}\}$. Both vertical boundary pairs share a single weight-4 bulk plaquette ($c_2 = \{3,4,6,7\}$ for $(3,6)$, $c_1 = \{1,2,4,5\}$ for $(2,5)$) and no boundary $X$-check, so their XOR signatures vanish ($g_3 \oplus g_6 = \texttt{0x4} \oplus \texttt{0x4} = \texttt{0x0}$, $g_2 \oplus g_5 = \texttt{0x2} \oplus \texttt{0x2} = \texttt{0x0}$).
- **9 Distinct Non-Zero Signature Orbits ($\mathcal{G}_{\neq 0} = \{\texttt{0x1}, \texttt{0x2}, \texttt{0x3}, \texttt{0x4}, \texttt{0x5}, \texttt{0x6}, \texttt{0x8}, \texttt{0xA}, \texttt{0xC}\}$)**:
  1. $\mathcal{O}_{\texttt{0x1}} = \{Z_0, ZZ_{1,2}\}$ (flips $c_0$)
  2. $\mathcal{O}_{\texttt{0x2}} = \{Z_2, Z_5, ZZ_{0,1}, ZZ_{3,4}\}$ (flips $c_1$)
  3. $\mathcal{O}_{\texttt{0x3}} = \{Z_1\}$ (uniquely identifiable single-qubit fault flipping $\{c_0, c_1\}$)
  4. $\mathcal{O}_{\texttt{0x4}} = \{Z_3, Z_6, ZZ_{4,5}, ZZ_{7,8}\}$ (flips $c_2$)
  5. $\mathcal{O}_{\texttt{0x5}} = \{ZZ_{0,3}, ZZ_{1,4}\}$ (uniquely correlated pair signature flipping $\{c_0, c_2\}$)
  6. $\mathcal{O}_{\texttt{0x6}} = \{Z_4\}$ (uniquely identifiable center-qubit fault flipping $\{c_1, c_2\}$)
  7. $\mathcal{O}_{\texttt{0x8}} = \{Z_8, ZZ_{6,7}\}$ (flips $c_3$)
  8. $\mathcal{O}_{\texttt{0xA}} = \{ZZ_{4,7}, ZZ_{5,8}\}$ (uniquely correlated pair signature flipping $\{c_1, c_3\}$)
  9. $\mathcal{O}_{\texttt{0xC}} = \{Z_7\}$ (uniquely identifiable single-qubit fault flipping $\{c_2, c_3\}$)

---

## 3. Resolution via Quotient Orbits and Prior-Tethered MAP Regularization
For each non-zero signature orbit $\mathcal{O}_g$ ($g \in \mathcal{G}_{\neq 0}$), the aggregate orbit flip probability:
$$P_g(\theta) = \sideset{}{^\boxplus}\bigoplus_{e \in \mathcal{O}_g} p_e = \frac{1}{2}\left[1 - \prod_{e \in \mathcal{O}_g} (1 - 2 p_e)\right]$$
is **strictly identifiable** from the 16-state joint syndrome distribution alone. Notice that single-qubit bulk/boundary vertices $Z_1, Z_4, Z_7$ occupy singleton orbits ($\mathcal{O}_{\texttt{0x3}}, \mathcal{O}_{\texttt{0x6}}, \mathcal{O}_{\texttt{0xC}}$) and vertical correlated pairs occupy dedicated pair-only orbits ($\mathcal{O}_{\texttt{0x5}}, \mathcal{O}_{\texttt{0xA}}$), enabling the Joint-DP estimator to distinguish correlated $ZZ$ crosstalk from single-qubit drift—something mathematically impossible in the legacy rank-4 marginal model.

To resolve individual parameters within multi-event orbits (such as $\mathcal{O}_{\texttt{0x2}}$ and $\mathcal{O}_{\texttt{0x4}}$), the Gaussian prior tether $\frac{\lambda}{2}\|\theta - \theta_{\text{prior}}\|_2^2$ ($\lambda = 0.50$) lifts the MAP Hessian to strict positive definiteness across the full 21-dimensional parameter space:
$$H_{\text{MAP}}(\theta) = \mathcal{I}(\theta) + \lambda I_{21} \succ 0, \qquad \operatorname{rank}\!\left(H_{\text{MAP}}(\theta)\right) = 21, \qquad \kappa\!\left(H_{\text{MAP}}(\theta)\right) = \frac{\lambda_{\max}(H_{\text{MAP}})}{\lambda_{\min}(H_{\text{MAP}})} = 234.05$$

---

## References
- **[R4]** E. van den Berg, Z. K. Minev, A. Kandala, and K. Temme, "Probabilistic error cancellation with sparse Pauli-Lindblad models on noisy quantum processors," *Nature Physics*, vol. 19, pp. 1116–1121, 2023.
- **[R5]** S. T. Flammia and J. J. Wallman, "Efficient estimation of Pauli channels," *ACM Transactions on Quantum Computing*, vol. 1, no. 1, pp. 1–32, 2020.
