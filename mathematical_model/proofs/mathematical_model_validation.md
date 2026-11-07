# Complete Mathematical Modeling of the AVID Architecture (`AVID_EPJ_Patches` & `Noise-Generation_Model`)

This document provides the complete, self-contained mathematical formulation of all theoretical, statistical, algorithmic, and hardware-execution components implemented across the **AVID Baseline Architecture (`AVID_EPJ_Patches`, `A01–A09` / `Q01–Q21`)** and the **Spatio-Temporal Quantum Noise World Model (`Noise-Generation_Model`)**.

---

## 1. CSS Stabilizer Algebra & Rotated Planar Surface Code Geometry (`A01 / Q02`)

### 1.1 Symplectic Commutativity & Code Dimension
A Calderbank–Shor–Steane (CSS) quantum error-correcting code on $n$ physical data qubits is defined by two binary parity-check matrices $H_X \in \mathbb{F}_2^{m_X \times n}$ (whose rows specify $X$-type stabilizer generators $S_c^X = \bigotimes_{v=0}^{n-1} X_v^{(H_X)_{c,v}}$ detecting $Z$ data faults) and $H_Z \in \mathbb{F}_2^{m_Z \times n}$ (whose rows specify $Z$-type stabilizer generators $S_c^Z = \bigotimes_{v=0}^{n-1} Z_v^{(H_Z)_{c,v}}$ detecting $X$ data faults).

For the stabilizer group $\mathcal{S} = \langle S_1^X, \dots, S_{m_X}^X, S_1^Z, \dots, S_{m_Z}^Z \rangle$ to be abelian, every $X$-check must share an even number of data qubits with every $Z$-check:
$$H_X H_Z^\top \equiv \mathbf{0} \pmod 2 \iff \forall c \in \{0, \dots, m_X-1\}, \; c' \in \{0, \dots, m_Z-1\}, \quad |\operatorname{supp}(H_{X,c}) \cap \operatorname{supp}(H_{Z,c'})| \equiv 0 \pmod 2$$

The number of encoded logical qubits is given by the $\mathbb{F}_2$-rank deficiency:
$$k = n - \operatorname{rank}_{\mathbb{F}_2}(H_X) - \operatorname{rank}_{\mathbb{F}_2}(H_Z)$$

### 1.2 Canonical Rotated $d=3$ Planar Surface Code (`[[9, 1, 3]]`)
Data qubits are indexed row-major $v = 3r + c \in \{0, \dots, 8\}$ on a $3 \times 3$ lattice $(r, c) \in \{0, 1, 2\}^2$. Each basis has $m_X = m_Z = (d^2 - 1)/2 = 4$ checks consisting of **two weight-2 boundary checks** and **two weight-4 bulk plaquettes**:

$$H_X = \begin{pmatrix}
1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 0 & 1 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1
\end{pmatrix}, \qquad
H_Z = \begin{pmatrix}
1 & 1 & 0 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 1 & 0 & 1 & 1
\end{pmatrix}$$

Direct verification over $\mathbb{F}_2$ yields:
$$H_X H_Z^\top \equiv \mathbf{0}_{4 \times 4} \pmod 2, \qquad \operatorname{rank}_{\mathbb{F}_2}(H_X) = \operatorname{rank}_{\mathbb{F}_2}(H_Z) = 4, \qquad k = 9 - 4 - 4 = 1$$

### 1.3 Logical Operators & Code Distance
The non-trivial logical operators $\bar{X}, \bar{Z} \in \mathbb{F}_2^n$ must commute with all opposite-basis stabilizers while anticommuting with each other:
$$H_Z \bar{X}^\top \equiv \mathbf{0} \pmod 2, \qquad H_X \bar{Z}^\top \equiv \mathbf{0} \pmod 2, \qquad \bar{X} \cdot \bar{Z}^\top \equiv 1 \pmod 2$$
With $\operatorname{supp}(\bar{X}) = \{0, 3, 6\}$ (left boundary column) and $\operatorname{supp}(\bar{Z}) = \{0, 1, 2\}$ (top boundary row), the exact code distances are:
$$d_X = \min_{x \in \ker(H_Z) \setminus \operatorname{rowspace}(H_X)} \operatorname{wt}(x) = 3, \qquad d_Z = \min_{z \in \ker(H_X) \setminus \operatorname{rowspace}(H_Z)} \operatorname{wt}(z) = 3, \qquad d = \min(d_X, d_Z) = 3$$

### 1.4 Degree-Padded Hardware Tanner Tables
To map heterogeneous weight-2 boundary checks and weight-4 bulk checks onto uniform 4-input hardware comparator trees (`MAX_DEG = 4`), each check $c \in \{0, \dots, m-1\}$ is represented by a padded index tuple $C_c = (v_{c,0}, v_{c,1}, v_{c,2}, v_{c,3}) \in \{0, \dots, n-1\}^4$ and a 4-bit active-edge mask $M_c \in \{0, 1\}^4$:
$$M_{c, d} = \begin{cases} 1, & \text{if } d < |\operatorname{supp}(H_c)|, \\ 0, & \text{otherwise.} \end{cases}$$
For $H_X$ on $d=3$, $(M_0, M_1, M_2, M_3) = (\texttt{0b0011}, \texttt{0b1111}, \texttt{0b1111}, \texttt{0b0011})$.

---

## 2. Correlated Physical Fault Channel & Syndrome Observation Algebra (`A02 / Q03`, `A08 / Q17`)

### 2.1 Elementary Fault Mechanisms & Binary Signatures
Consider the $X$-stabilizer syndrome vector $s \in \mathbb{F}_2^m$ (represented as an integer bitmask $s = \sum_{c=0}^{m-1} s_c 2^c \in \{0, \dots, 2^m - 1\}$). The physical noise channel consists of $E$ independent Bernoulli fault mechanisms $\mathcal{E} = \{e_1, \dots, e_E\}$ with occurrence indicator variables $\xi_e \sim \operatorname{Bernoulli}(p_e)$, $p_e \in (0, 0.5)$, partitioned into three classes:

1. **Single-Qubit Data Faults ($Z_i$, $i \in \{0, \dots, n-1\}$)**:
   Each data qubit $i$ experiences a phase-flip fault with probability $p_i$, imparting the column syndrome signature:
   $$g_i = H_X(:, i) \in \mathbb{F}_2^m$$
2. **Correlated Nearest-Neighbor Crosstalk Faults ($Z_i Z_j$, $(i, j) \in \mathcal{E}_{ZZ}$)**:
   Parasitic dispersive $ZZ$ coupling during syndrome extraction causes simultaneous phase flips on adjacent qubits $(i, j)$ with probability $p_{ij}$. By $\mathbb{F}_2$-linearity of the parity-check map, the syndrome signature of a simultaneous pair fault is the bitwise XOR (symmetric difference) of the individual qubit signatures:
   $$g_{ij} = H_X(:, i) \oplus H_X(:, j) = g_i \oplus g_j \in \mathbb{F}_2^m$$
3. **Syndrome Measurement Bit-Flip Faults ($M_c$, $c \in \{0, \dots, m-1\}$)**:
   Readout errors flip check $c$ directly with probability $p_{\text{meas}, c}$ and unit weight-1 signature:
   $$g_{\text{meas}, c} = \mathbf{u}_c = 2^c \in \mathbb{F}_2^m$$

### 2.2 Mathematical Proof of Failure of Single-Qubit Crosstalk Folding
Suppose two adjacent qubits $i$ and $j$ both belong to check $c$ (such as qubits $4$ and $5$ in check $c_1 = \{1, 2, 4, 5\}$). Under a correlated $Z_4 Z_5$ fault:
$$(g_{4,5})_c = (g_4)_c \oplus (g_5)_c = 1 \oplus 1 = 0$$
Check $c_1$ is flipped **twice** simultaneously and therefore **never fires** due to $Z_4 Z_5$. Folding $p_{4,5}$ into effective independent single-qubit rates $p_{\text{eff}}(4)$ and $p_{\text{eff}}(5)$ treats $Z_4$ and $Z_5$ as independent flips, falsely predicting a non-zero firing probability $2 p_{4,5}(1 - p_{4,5}) > 0$ on check $c_1$ and destroying inter-check parity correlations.

---

## 3. Exact $2^m$-State Dynamic Programming Joint Likelihood & Analytic Gradients (`A02 / Q03`)

### 3.1 Group-Algebra XOR Convolution Dynamic Program
The observed syndrome in any round is the $\mathbb{F}_2^m$-sum of all active fault mechanisms:
$$S = \bigoplus_{k=1}^E \xi_{e_k} g_{e_k} \in \mathbb{F}_2^m, \qquad \xi_{e_k} \stackrel{\text{ind.}}{\sim} \operatorname{Bernoulli}(p_{e_k})$$
Because the sum of independent random variables on the abelian group $(\mathbb{F}_2^m, \oplus)$ is governed by XOR convolution, the exact joint probability distribution $P_\theta(s) = \Pr(S = s \mid \theta)$ over all $2^m$ syndrome states is computed in $O(E \cdot 2^m)$ operations via dynamic programming:

- **Initialization**: $P^{(0)}(s) = \mathbb{I}[s = 0]$ for $s \in \{0, \dots, 2^m - 1\}$.
- **Recurrence ($k = 1, \dots, E$)**:
  $$P^{(k)}(s) = (1 - p_{e_k}) P^{(k-1)}(s) + p_{e_k} P^{(k-1)}(s \oplus g_{e_k}), \qquad \forall s \in \{0, \dots, 2^m - 1\}$$
- **Termination**: $P_\theta(s) = P^{(E)}(s)$. For $d=3$ ($m=4, 2^m=16, E=21$), this requires exactly $21 \times 16 = 336$ multiply-add operations.

### 3.2 Exact Closed-Form Coset-Inversion Gradient Theorem
**Theorem 1 (Exact Single-Pass Coset Gradient).** *For any fault mechanism $e \in \mathcal{E}$ with probability $p_e \in (0, 0.5)$ and signature $g_e \in \mathbb{F}_2^m$, the exact partial derivative of the joint syndrome distribution $P_\theta(s)$ with respect to $p_e$ is given by:*
$$\frac{\partial P_\theta(s)}{\partial p_e} = \begin{cases}
0, & \text{if } g_e = \mathbf{0}, \\[6pt]
\dfrac{P_\theta(s \oplus g_e) - P_\theta(s)}{1 - 2 p_e}, & \text{if } g_e \neq \mathbf{0}.
\end{cases}$$

*Proof.* Because convolution in $\mathbb{R}[\mathbb{F}_2^m]$ is commutative and associative, place mechanism $e$ last in the DP ordering and let $P^{(-e)}(s)$ denote the distribution of $\mathcal{E} \setminus \{e\}$. On each coset pair $\{s, s \oplus g_e\}$ ($g_e \neq \mathbf{0}$):
$$\begin{pmatrix} P_\theta(s) \\ P_\theta(s \oplus g_e) \end{pmatrix} = \begin{pmatrix} 1 - p_e & p_e \\ p_e & 1 - p_e \end{pmatrix} \begin{pmatrix} P^{(-e)}(s) \\ P^{(-e)}(s \oplus g_e) \end{pmatrix}$$
Subtracting the two rows gives $P_\theta(s \oplus g_e) - P_\theta(s) = (1 - 2 p_e) \left( P^{(-e)}(s \oplus g_e) - P^{(-e)}(s) \right)$. Since $\frac{\partial P_\theta(s)}{\partial p_e} = P^{(-e)}(s \oplus g_e) - P^{(-e)}(s)$ and $1 - 2 p_e > 0$ for $p_e < 0.5$, dividing by $1 - 2 p_e$ yields the exact closed-form gradient. $\blacksquare$

### 3.3 Regularized MAP Cross-Entropy Objective
Given empirical syndrome frequencies $\hat{P}(s) = c(s) / N$, the MAP negative log-likelihood and its exact gradient are:
$$\mathcal{L}(\theta) = -\sum_{s=0}^{2^m - 1} \hat{P}(s) \ln P_\theta(s) + \frac{\lambda}{2} \|\theta - \theta_{\text{prior}}\|_2^2$$
$$\frac{\partial \mathcal{L}(\theta)}{\partial p_e} = -\frac{\mathbb{I}[g_e \neq \mathbf{0}]}{1 - 2 p_e} \sum_{s=0}^{2^m - 1} \frac{\hat{P}(s)}{P_\theta(s)} \left( P_\theta(s \oplus g_e) - P_\theta(s) \right) + \lambda (p_e - p_{e, \text{prior}})$$

---

## 4. Structural Identifiability, Gauge Orbits & Fisher Information (`A03 / Q04`)

By Theorem 1, the $e$-th column of the $2^m \times E$ Jacobian $J(\theta)$ is $J_{:, e}(\theta) = \frac{1}{1 - 2 p_e} (\Pi_{g_e} P_\theta - P_\theta)$. Consequently, two mechanisms $e_a, e_b$ have collinear Jacobian columns if and only if $g_{e_a} = g_{e_b}$, and any mechanism with $g_e = \mathbf{0}$ has $J_{:, e} \equiv \mathbf{0}$. Thus, the unregularized Jacobian and Fisher Information Matrix $\mathcal{I}(\theta) = J(\theta)^\top \operatorname{diag}(P_\theta^{-1}) J(\theta)$ have exact structural rank equal to the number of distinct non-zero syndrome signatures:
$$\operatorname{rank}(J(\theta)) = \operatorname{rank}(\mathcal{I}(\theta)) = \left|\{g_e \neq \mathbf{0} : e \in \mathcal{E}\}\right| = 9 \quad (\text{vs. rank } 4 \text{ in the legacy 4-marginal model})$$
With the strictly convex Gaussian prior tether $\lambda I_E \succ 0$ ($\lambda = 0.5$), the MAP Hessian $H_{\text{MAP}}(\theta) = \mathcal{I}(\theta) + \lambda I_{21}$ has full rank **21 / 21** and condition number $\kappa = 234.05$.

---

## 5. Exact XOR Marginalization & Q7.8 Fixed-Point Quantization (`SF4`, `SF8`)

Each data qubit's effective marginal error rate $p_{\text{eff}}(v)$ under its single-qubit fault $p_v$ and incident $ZZ$ pair faults $\{e = (u, v) \in \mathcal{E}_{ZZ}\}$ is:
$$p_{\text{eff}}(v) = p_v \boxplus \sideset{}{^\boxplus}\bigoplus_{e \in \mathcal{E}_{ZZ} : v \in e} p_e = \frac{1}{2} \left[ 1 - (1 - 2 p_v) \prod_{e \in \mathcal{E}_{ZZ} : v \in e} (1 - 2 p_e) \right]$$
which is converted to a signed 16-bit Q7.8 LLR register word:
$$\gamma_v = \ln\!\left(\frac{1 - p_{\text{eff}}(v)}{p_{\text{eff}}(v)}\right), \qquad \Gamma_v^{\text{Q7.8}} = \operatorname{clip}\!\left(\operatorname{round}(256.0 \cdot \gamma_v), \; -32768, \; +32767\right)$$

---

## 6. Hardware-Exact Normalized Min-Sum BP & Cycle Contract (`A06 / Q06`)

Over $K$ iterations, variable-to-check (V2C) and masked check-to-variable (C2V) messages in signed Q7.8 fixed point obey:
$$\mu_{v \to c}^{(t)} = \operatorname{sat}_{\text{Q7.8}}\!\left(L_v^{(t-1)} - \nu_{c \to v}^{(t-1)}\right), \qquad \nu_{c \to v}^{(t)} = (-1)^{s_c \oplus \bigoplus_{v' \in N(c) \setminus \{v\}} \sigma_{v' \to c}^{(t)}} \mathcal{S}_{0.75}\!\left(\min_{v' \in N(c) \setminus \{v\}} |\mu_{v' \to c}^{(t)}|\right)$$
where $\mathcal{S}_{0.75}(x) = x - (x \gg 2)$ implements $\alpha = 0.75$ normalization with zero DSP slices, and inactive padded check slots ($M_{c, d} = 0$) are masked to $+\infty$ ($\texttt{0x7FFF}$). Posterior beliefs accumulate via $L_v^{(t)} = \operatorname{sat}_{\text{Q7.8}}(\Gamma_v^{\text{Q7.8}} + \sum_{c \in N(v)} \nu_{c \to v}^{(t)})$ with hard decision $\hat{e}_v = \mathbb{I}[L_v^{(K)} < 0]$.

The exact deterministic hardware cycle count is:
$$L(K) = 1_{\text{INIT}} + K \cdot \left(1_{\text{V2C}} + 3_{\text{WAIT\_C2V}} + 1_{\text{BELIEF\_UPD}}\right) + 1_{\text{HARD\_DECIDE}} = 2 + 5K \text{ cycles}$$
giving $L(3) = 17 \text{ cycles} = 68.0\,\text{ns}$ at $250\,\text{MHz}$.

---

## References
- **[R1]** A. G. Fowler et al., *Physical Review A*, vol. 86, no. 3, p. 032324, 2012.
- **[R2]** N. Liyanage et al., *arXiv:2301.08419*, 2023.
- **[R3]** J. P. Bonilla Ataides et al., *Nature Communications*, vol. 12, p. 2172, 2021.
- **[R4]** E. van den Berg et al., *Nature Physics*, vol. 19, pp. 1116–1121, 2023.
- **[R5]** S. T. Flammia and J. J. Wallman, *ACM Transactions on Quantum Computing*, vol. 1, no. 1, pp. 1–32, 2020.
- **[R6]** J. Chen and M. P. C. Fossorier, *IEEE Transactions on Communications*, vol. 50, no. 3, pp. 406–414, 2002.
- **[R7]** D. C. Liu and J. Nocedal, *Mathematical Programming*, vol. 45, pp. 503–528, 1989.
- **[R8]** Y. Tomita and K. M. Svore, *Physical Review A*, vol. 90, no. 6, p. 062320, 2014.
- **[R9]** Xilinx Inc., *Zynq UltraScale+ Device Technical Reference Manual (UG1085)*, v2.4, 2023.
