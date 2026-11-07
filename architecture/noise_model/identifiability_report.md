# Structural Identifiability, Gauge Nullspace & MAP Regularization Report

## 1. Three Distinct Identifiability Quantities (Finding 9 Resolution)

To prevent any conflation between syndrome-only observability and prior-assisted regularization, AVID strictly separates three quantities across all reports and code:

1. **Data-Identifiable Syndrome Orbits (`rank(J) = 9`)**:
   - The rotated $d=3$ surface code has 4 checks per basis ($2^4 = 16$ syndrome states, 15 degrees of freedom after normalization $\sum_s P_	heta(s) = 1$).
   - The 21 physical fault mechanisms per basis (9 single-qubit faults + 12 adjacent correlated pairs) map onto **9 distinct non-zero syndrome signatures**:
     `{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x8, 0xA, 0xC}`.
   - Because XOR convolution is associative and commutative over $\mathbb{F}_2^4$, faults sharing the same syndrome signature $g_e \in \mathbb{F}_2^4$ are strictly indistinguishable from single-window syndrome statistics alone.
   - Consequently, the unregularized $16 	imes 21$ Jacobian $J(	heta) = \partial P_	heta(s) / \partial 	heta$ has exact structural rank **9** (`sv_9 = 0.7289`, `sv_10 = 2.08e-17`), and the gauge/collision nullspace has dimension **$21 - 9 = 12$**.

2. **Prior-Regularized MAP Point Selection (`rank(H_MAP) = 21/21`)**:
   - Adding the quadratic Gaussian prior $rac{\lambda}{2}\|	heta - 	heta_{\mathrm{prior}}\|_2^2$ ($\lambda > 0$) yields the regularized MAP Hessian:
     $$H_{\mathrm{MAP}}(	heta) = J(	heta)^	op W(	heta) J(	heta) + \lambda I_{21}$$
   - $H_{\mathrm{MAP}}$ is strictly positive-definite (`rank = 21/21`, minimum eigenvalue $\ge \lambda$).
   - **Important Mathematical Distinction**: Full rank of $H_{\mathrm{MAP}}$ guarantees a unique, well-conditioned L-BFGS optimum, **not** independent observability of all 21 physical parameters from syndrome data alone. Within each multi-event signature orbit, the split between colliding mechanisms is anchored by $	heta_{\mathrm{prior}}$ (and in Pass 2 of the two-pass solver, by endpoint single-qubit activity).

3. **Ground-Truth Recovery Metrics**:
   - **`orbit_9class_rmse` (`0.0081–0.0148`)**: Measures data-driven recovery of the 9 observable syndrome signature class rates $	ilde{P}_c = rac{1}{2}(1 - \prod_{e \in \mathcal{C}_c}(1 - 2	heta_e))$. This is the primary identifiability metric.
   - **`param_21_rmse` (`0.0046–0.0086`)**: Measures proximity of the regularized 21-parameter MAP estimate $\hat{	heta}_{\mathrm{MAP}}$ to the simulated parameter vector under the calibrated baseline prior. It must never be cited as prior-free 21-parameter observability.
