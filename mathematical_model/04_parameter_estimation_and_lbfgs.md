# 05 — L-BFGS Optimization, XOR Marginalization, and Q7.8 Numerics (`SF4`, `SF8`)

## 1. Zero-Allocation Two-Loop L-BFGS Quasi-Newton Solver
To minimize the regularized Joint-DP negative log-likelihood $\mathcal{L}(\theta)$ within the slow-path real-time budget ($< 250\,\mu\text{s}$) on the ARM Cortex-A53 without dynamic heap allocation, [`lbfgs_optimizer.cpp`](software/src/inference/lbfgs_optimizer.cpp) and [`lbfgs_solver.cpp`](software/src/estimator/lbfgs_solver.cpp) maintain a circular history buffer of depth $M = 5$ storing parameter displacements $s_k = \theta_{k+1} - \theta_k$, gradient displacements $y_k = \nabla \mathcal{L}(\theta_{k+1}) - \nabla \mathcal{L}(\theta_k)$, and inverse inner products $\rho_k = (y_k^\top s_k)^{-1}$ whenever the secant curvature condition $y_k^\top s_k > 10^{-12}$ holds.

Given current gradient $g_k = \nabla \mathcal{L}(\theta_k)$, the two-loop recursion computes the quasi-Newton search direction $d_k = -H_k g_k$ in $O(M \cdot E)$ scalar operations:
1. **Forward Loop ($i = k-1, \dots, k-m$)**:
   $$\alpha_i = \rho_i s_i^\top q, \qquad q \leftarrow q - \alpha_i y_i \quad (\text{initialized at } q = g_k)$$
2. **Initial Hessian Scaling**:
   $$\gamma_k = \frac{s_{k-1}^\top y_{k-1}}{y_{k-1}^\top y_{k-1}}, \qquad r \leftarrow \gamma_k q$$
3. **Backward Loop ($i = k-m, \dots, k-1$)**:
   $$\beta_i = \rho_i y_i^\top r, \qquad r \leftarrow r + s_i (\alpha_i - \beta_i), \qquad d_k = -r$$

The step size $\eta_k = \eta_0 2^{-j}$ ($j \in \{0, \dots, 9\}$) is chosen by projected Armijo backtracking line search onto the interior hypercube $\Theta = [10^{-5}, 0.45]^{21}$:
$$\theta_{k+1} = \Pi_\Theta(\theta_k + \eta_k d_k), \qquad \mathcal{L}(\theta_{k+1}) \le \mathcal{L}(\theta_k) + c_1 \eta_k g_k^\top d_k \quad (c_1 = 10^{-4})$$

---

## 2. Exact Associative XOR Marginalization to Variable-Node Priors
Once the 21 physical fault probabilities $\hat{\theta} = (\hat{p}_0, \dots, \hat{p}_8, \hat{p}_{e_1}, \dots, \hat{p}_{e_{12}})$ are estimated by the Joint-DP MAP solver, they are projected onto the 9 variable nodes of the Belief Propagation Tanner graph. A data qubit $v \in \{0, \dots, 8\}$ suffers a net phase flip if and only if an odd number of its incident fault mechanisms $\{Z_v\} \cup \{Z_u Z_v : (u, v) \in \mathcal{E}_{ZZ}\}$ occur.

Define the associative binary XOR-Bernoulli operator $a \boxplus b := a(1 - b) + (1 - a)b = \frac{1}{2}[1 - (1 - 2a)(1 - 2b)]$. The exact marginal error probability $p_{\text{eff}}(v)$ of data qubit $v$ is:
$$p_{\text{eff}}(v) = \hat{p}_v \boxplus \sideset{}{^\boxplus}\bigoplus_{e \in \mathcal{E}_{ZZ} : v \in e} \hat{p}_e = \frac{1}{2}\left[1 - (1 - 2 \hat{p}_v) \prod_{e \in \mathcal{E}_{ZZ} : v \in e} (1 - 2 \hat{p}_e)\right]$$

---

## 3. Signed 16-Bit Q7.8 Fixed-Point Quantization Contract
In accordance with [`llr_numeric_contract.yaml`](architecture/contracts/llr_numeric_contract.yaml) and [`llr_quantizer.cpp`](software/src/estimator/llr_quantizer.cpp), each qubit's effective error probability $p_{\text{eff}}(v) \in [10^{-5}, 0.49]$ is transformed into its continuous log-likelihood ratio $\gamma_v$ and quantized to a signed 16-bit two's-complement Q7.8 fixed-point integer (1 sign bit, 7 integer bits, 8 fractional bits, step size $2^{-8} = 0.00390625$, dynamic range $[-128.0, +127.99609375]$):
$$\gamma_v = \ln\!\left(\frac{1 - p_{\text{eff}}(v)}{p_{\text{eff}}(v)}\right), \qquad \Gamma_v^{\text{Q7.8}} = \operatorname{clip}\!\left(\operatorname{round}(256.0 \cdot \gamma_v), \; -32768, \; +32767\right) \in \mathbb{Z}_{16}$$

Before committing a candidate weight bank $\Gamma^{\text{Q7.8}} \in \mathbb{Z}_{16}^9$ to the FPGA shadow registers, [`llr_quantizer.cpp`](software/src/estimator/llr_quantizer.cpp) verifies that every entry lies within the safe operating envelope $[\Gamma_{\min}, \Gamma_{\max}] = [\texttt{0x0040}, \texttt{0x3000}]$ ($[+0.25, +48.0]$ in floating-point LLR units).

---

## References
- **[R6]** J. Chen and M. P. C. Fossorier, "Near optimum universal belief propagation based decoding of low-density parity check codes," *IEEE Transactions on Communications*, vol. 50, no. 3, pp. 406–414, 2002.
- **[R7]** D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization," *Mathematical Programming*, vol. 45, pp. 503–528, 1989.
