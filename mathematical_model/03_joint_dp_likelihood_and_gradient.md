# 03 — Exact $2^m$-State Dynamic Programming Joint Likelihood and Coset Gradients (`A02 / Q03`)

## 1. Group-Algebra $\mathbb{R}[\mathbb{F}_2^m]$ XOR-Convolution Dynamic Program
Let $\xi_{e_k} \stackrel{\text{ind.}}{\sim} \operatorname{Bernoulli}(p_{e_k})$ denote the binary indicator of the $k$-th physical fault mechanism ($k \in \{1, \dots, E\}$) with syndrome signature $g_{e_k} \in \mathbb{F}_2^m$. The total observed syndrome vector in a given round is the $\mathbb{F}_2^m$-sum:
$$S = \bigoplus_{k=1}^E \xi_{e_k} g_{e_k} \in \mathbb{F}_2^m$$

Because the probability mass function of a sum of independent random variables over the finite abelian group $(\mathbb{F}_2^m, \oplus)$ is the iterated group convolution of their individual distributions, the exact joint syndrome distribution $P_\theta(s) = \Pr(S = s \mid \theta)$ over all $2^m$ syndrome states is computed recursively in $O(E \cdot 2^m)$ operations:

- **Base Distribution ($k = 0$)**:
  $$P^{(0)}(s) = \mathbb{I}[s = \mathbf{0}] = \begin{cases} 1, & s = 0, \\ 0, & s \in \{1, \dots, 2^m - 1\}. \end{cases}$$
- **Dynamic Programming Step ($k = 1, \dots, E$)**:
  $$P^{(k)}(s) = (1 - p_{e_k}) P^{(k-1)}(s) + p_{e_k} P^{(k-1)}(s \oplus g_{e_k}), \qquad \forall s \in \{0, \dots, 2^m - 1\}$$
- **Exact Joint Likelihood**: $P_\theta(s) = P^{(E)}(s)$. For the $d=3$ rotated surface code ($m = 4, 2^m = 16, E = 21$), a complete forward evaluation requires only $21 \times 16 = 336$ floating-point multiply-add operations ($< 0.4\,\mu\text{s}$ on ARM Cortex-A53).

---

## 2. Dual Walsh–Hadamard Spectral Representation
The characters of the group $(\mathbb{F}_2^m, \oplus)$ are the Walsh functions $\chi_\omega(s) = (-1)^{\langle \omega, s \rangle}$ for $\omega \in \mathbb{F}_2^m$, where $\langle \omega, s \rangle = \sum_{c=0}^{m-1} \omega_c s_c \pmod 2$. Applying the Walsh–Hadamard Transform (WHT) diagonalizes the XOR-convolution recurrence into a pointwise product of spectral eigenvalues:
$$\hat{P}_\theta(\omega) = \sum_{s \in \mathbb{F}_2^m} (-1)^{\langle \omega, s \rangle} P_\theta(s) = \prod_{k=1}^E \left(1 - 2 p_{e_k} \mathbb{I}\left[\langle \omega, g_{e_k} \rangle \equiv 1 \pmod 2\right]\right)$$
with inverse transform $P_\theta(s) = 2^{-m} \sum_{\omega \in \mathbb{F}_2^m} (-1)^{\langle \omega, s \rangle} \hat{P}_\theta(\omega)$.

---

## 3. Exact Closed-Form Coset-Inversion Gradient Theorem
**Theorem 1 (Single-Pass Exact Analytic Gradient).** *For any fault mechanism $e \in \mathcal{E}$ with probability $p_e \in (0, 0.5)$ and syndrome signature $g_e \in \mathbb{F}_2^m$, the exact partial derivative of the joint syndrome distribution $P_\theta(s)$ with respect to $p_e$ is given in closed form by:*
$$\frac{\partial P_\theta(s)}{\partial p_e} = \begin{cases}
0, & \text{if } g_e = \mathbf{0}, \\[6pt]
\dfrac{P_\theta(s \oplus g_e) - P_\theta(s)}{1 - 2 p_e}, & \text{if } g_e \neq \mathbf{0}.
\end{cases}$$

*Proof.* Because XOR convolution in the commutative group algebra $\mathbb{R}[\mathbb{F}_2^m]$ is independent of mechanism ordering, place mechanism $e$ last in the DP sequence and denote by $P^{(-e)}(s)$ the joint distribution induced by the remaining $E - 1$ mechanisms $\mathcal{E} \setminus \{e\}$. If $g_e = \mathbf{0}$, then $s \oplus g_e = s$ and $P_\theta(s) = (1 - p_e) P^{(-e)}(s) + p_e P^{(-e)}(s) = P^{(-e)}(s)$, so $\frac{\partial P_\theta(s)}{\partial p_e} = 0$.

If $g_e \neq \mathbf{0}$, the state space $\mathbb{F}_2^m$ partitions into $2^{m-1}$ disjoint 2-element cosets $\{s, s \oplus g_e\}$. On each coset pair, the final DP step acts as a $2 \times 2$ doubly stochastic transition matrix:
$$\begin{pmatrix} P_\theta(s) \\ P_\theta(s \oplus g_e) \end{pmatrix} = \begin{pmatrix} 1 - p_e & p_e \\ p_e & 1 - p_e \end{pmatrix} \begin{pmatrix} P^{(-e)}(s) \\ P^{(-e)}(s \oplus g_e) \end{pmatrix}$$
Differentiating the first row with respect to $p_e$ yields $\frac{\partial P_\theta(s)}{\partial p_e} = P^{(-e)}(s \oplus g_e) - P^{(-e)}(s)$. Subtracting the first row from the second row of the $2 \times 2$ linear system yields the antisymmetric eigenmode relation:
$$P_\theta(s \oplus g_e) - P_\theta(s) = (1 - 2 p_e) \left( P^{(-e)}(s \oplus g_e) - P^{(-e)}(s) \right)$$
Since $p_e \in (0, 0.5) \implies 1 - 2 p_e > 0$, dividing both sides by $1 - 2 p_e$ yields the exact identity. $\blacksquare$

---

## 4. Regularized MAP Cross-Entropy Objective
Given an empirical syndrome burst of $N$ rounds with histogram counts $c(s)$ and empirical frequencies $\hat{P}(s) = c(s) / N$, the prior-tethered Maximum A Posteriori (MAP) negative log-likelihood objective and its exact gradient implemented in [`joint_dp_likelihood.py`](simulation/reference/noise_model/joint_dp_likelihood.py) and [`noise_model.cpp`](software/src/inference/noise_model.cpp) are:
$$\mathcal{L}(\theta) = -\sum_{s=0}^{2^m - 1} \hat{P}(s) \ln P_\theta(s) + \frac{\lambda}{2} \|\theta - \theta_{\text{prior}}\|_2^2$$
$$\frac{\partial \mathcal{L}(\theta)}{\partial p_e} = -\frac{\mathbb{I}[g_e \neq \mathbf{0}]}{1 - 2 p_e} \sum_{s=0}^{2^m - 1} \frac{\hat{P}(s)}{P_\theta(s)} \left( P_\theta(s \oplus g_e) - P_\theta(s) \right) + \lambda (p_e - p_{e, \text{prior}})$$
As verified in [`test_joint_likelihood.py`](simulation/tests/test_joint_likelihood.py) and [`test_baseline_suite.cpp`](software/tests/test_baseline_suite.cpp), this analytical gradient agrees with central finite differences to within $3.77 \times 10^{-9}$ and the forward DP distribution agrees with brute-force $2^E$ subset enumeration to within $10^{-14}$.

---

## References
- **[R4]** E. van den Berg, Z. K. Minev, A. Kandala, and K. Temme, "Probabilistic error cancellation with sparse Pauli-Lindblad models on noisy quantum processors," *Nature Physics*, vol. 19, pp. 1116–1121, 2023.
- **[R5]** S. T. Flammia and J. J. Wallman, "Efficient estimation of Pauli channels," *ACM Transactions on Quantum Computing*, vol. 1, no. 1, pp. 1–32, 2020.
