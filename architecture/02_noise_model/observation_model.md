# Correlated Quantum Syndrome Observation Model (`A02 / Q03`, `A08 / Q17`)

## 1. Physical Fault Mechanisms on the Canonical $d=3$ Rotated Surface Code
On the canonical rotated $d=3$ planar surface code ($n=9$ data qubits, $m=4$ X-stabilizer checks $c_0=\{0,1\}$, $c_1=\{1,2,4,5\}$, $c_2=\{3,4,6,7\}$, $c_3=\{7,8\}$), the physical noise channel comprises three distinct fault classes:

1. **Single-Qubit Data Phase-Flip Events ($Z_i$, $i \in \{0, \dots, 8\}$)**:
   Each data qubit $i$ suffers an independent $Z_i$ fault with probability $p_i$, producing the column syndrome signature $g_i = H_X(:, i) \in \{0, 1\}^4$.
2. **Correlated Nearest-Neighbor $Z_i Z_j$ Crosstalk Events ($(i, j) \in \mathcal{E}_{ZZ}$, 12 edges)**:
   Parasitic dispersive $ZZ$ coupling during syndrome extraction induces simultaneous two-qubit $Z_i Z_j$ faults with probability $p_{ij}$. Because stabilizer measurements are $\mathbb{F}_2$-linear, a simultaneous $Z_i Z_j$ fault flips the symmetric difference (bitwise XOR) of the two single-qubit signatures:
   $$g_{ij} = g_i \oplus g_j \in \{0, 1\}^4$$
   Any check $c$ supported on **both** $i$ and $j$ is flipped twice and therefore remains unflipped ($0$) under $Z_i Z_j$. Consequently, folding $p_{ij}$ into single-qubit rates $p_{\text{eff}}(i), p_{\text{eff}}(j)$ before computing check marginals violates the algebraic structure of correlated $ZZ$ errors.
3. **Measurement Bit-Flip Faults ($M_c$, $c \in \{0, \dots, 3\}$)**:
   Readout errors flip check $c$ directly with unit weight-1 signature $g_{\text{meas}, c} = e_c$ ($2^c$) and produce temporal difference syndromes $d_t = s_t \oplus s_{t-1}$ across consecutive rounds.

## 2. Exact $2^m$-State Dynamic Programming Joint Syndrome Distribution
Let $\mathcal{E} = \{e_1, \dots, e_E\}$ denote the $E=21$ (or $E=25$ with readout faults) independent Bernoulli fault mechanisms with probabilities $\theta = (p_{e_1}, \dots, p_{e_E})$ and bitmask signatures $g_{e_k} \in \{0, \dots, 15\}$. Starting from $P^{(0)}(s) = \mathbb{I}[s = 0]$, the exact joint distribution over all $2^m = 16$ syndrome vectors is computed in $O(E \cdot 2^m) = 336$ multiply-add operations via XOR-convolution dynamic programming:
$$P^{(k)}(s) = (1 - p_{e_k}) P^{(k-1)}(s) + p_{e_k} P^{(k-1)}(s \oplus g_{e_k}), \quad s \in \{0, \dots, 15\}$$

## 3. Exact Closed-Form Analytic Gradients via Coset Inversion
Because the $\mathbb{F}_2^m$ XOR convolution group algebra is commutative and each binary transition matrix $(1 - p_e) I + p_e X_{g_e}$ has eigenvalue $1 - 2 p_e > 0$ on the antisymmetric subspace $\{P(s) - P(s \oplus g_e)\}$, the exact partial derivative with respect to every fault parameter $p_e$ ($g_e \neq 0$) is obtained directly from the final distribution $P(s) = P^{(E)}(s)$ without extra DP passes:
$$\frac{\partial P(s)}{\partial p_e} = \frac{P(s \oplus g_e) - P(s)}{1 - 2 p_e}$$

## References
- **[R1]** A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," *Physical Review A*, vol. 86, no. 3, p. 032324, 2012.
- **[R3]** J. P. Bonilla Ataides, D. K. Tuckett, S. D. Bartlett, S. T. Flammia, and B. J. Brown, "The XZZX surface code," *Nature Communications*, vol. 12, p. 2172, 2021.
- **[R8]** Y. Tomita and K. M. Svore, "Low-distance surface codes under realistic quantum noise," *Physical Review A*, vol. 90, no. 6, p. 062320, 2014.
