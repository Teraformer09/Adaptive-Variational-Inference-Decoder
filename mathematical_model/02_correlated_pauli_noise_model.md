# 02 — Correlated Fault Channel and Syndrome Observation Algebra (`A02 / Q03`, `A08 / Q17`)

## 1. Elementary Physical Fault Classes and Syndrome Signatures
In superconducting transmon surface code processors, physical errors arise from single-qubit decoherence ($T_1/T_2$ relaxation and flux drift), two-qubit dispersive $ZZ$ crosstalk during CNOT/CZ stabilizer extraction schedules, and readout resonator discrimination errors. Under Pauli twirling across syndrome extraction rounds, the physical channel for a given basis (such as $Z$-type faults detected by the $m = 4$ $X$-stabilizer checks) is modeled by $E$ independent Bernoulli fault mechanisms $\mathcal{E} = \{e_1, \dots, e_E\}$ with occurrence probabilities $\theta = (p_{e_1}, \dots, p_{e_E}) \in (0, 0.5)^E$ and binary syndrome signatures $g_e \in \mathbb{F}_2^m$ (represented as integer bitmasks $g_e = \sum_{c=0}^{m-1} (g_e)_c 2^c \in \{0, \dots, 2^m - 1\}$):

1. **Single-Qubit Data Phase-Flip Events ($Z_i$, $i \in \{0, \dots, n-1\}$)**:
   An isolated phase flip on data qubit $i$ occurs with probability $p_i$ and flips all $X$-checks incident on qubit $i$:
   $$g_i = H_X(:, i) \in \mathbb{F}_2^m$$
   On the canonical rotated $d=3$ code ($n=9, m=4$), the 9 single-qubit signatures are:
   $$(g_0, \dots, g_8) = (\texttt{0x1}, \texttt{0x3}, \texttt{0x2}, \texttt{0x4}, \texttt{0x6}, \texttt{0x2}, \texttt{0x4}, \texttt{0xC}, \texttt{0x8})$$
2. **Correlated Nearest-Neighbor $Z_i Z_j$ Crosstalk Events ($(i, j) \in \mathcal{E}_{ZZ}$)**:
   Parasitic $ZZ$ coupling between nearest-neighbor data qubits $(i, j)$ on the 2D lattice (12 horizontal and vertical edges on the $3 \times 3$ grid) induces a simultaneous two-qubit $Z_i Z_j$ fault with probability $p_{ij}$. Because syndrome extraction is $\mathbb{F}_2$-linear, a simultaneous fault on qubits $i$ and $j$ produces the bitwise XOR of their individual syndrome signatures:
   $$g_{ij} = H_X(:, i) \oplus H_X(:, j) = g_i \oplus g_j \in \mathbb{F}_2^m$$
3. **Syndrome Readout Bit-Flip Events ($M_c$, $c \in \{0, \dots, m-1\}$)**:
   A measurement error on check ancilla $c$ flips check bit $s_c$ directly with probability $p_{\text{meas}, c}$ and unit weight-1 signature:
   $$g_{\text{meas}, c} = \mathbf{u}_c = 2^c \in \mathbb{F}_2^m$$
   Across consecutive extraction rounds $t-1$ and $t$, a persistent data-qubit fault at round $t$ flips the temporal difference syndrome $d_t = s_t \oplus s_{t-1}$ once at round $t$, whereas a transient readout error at round $t$ flips both $d_t$ and $d_{t+1}$.

---

## 2. Algebraic Proof of the Failure of Single-Qubit Crosstalk Folding
In the legacy noise model prior to the `A02 / Q03` repair, correlated pair rates $p_{ij}$ were folded into effective single-qubit probabilities $p_{\text{eff}}(i) = p_i + \sum_{j \in \mathcal{N}(i)} p_{ij}(1 - 2 p_i)$ before evaluating independent check marginals $q_c = \frac{1}{2}\left[1 - \prod_{v \in \operatorname{supp}(H_c)} (1 - 2 p_{\text{eff}}(v))\right]$.

To see why this folding is algebraically invalid, consider any nearest-neighbor pair $(i, j)$ that shares a stabilizer check $c \in N(i) \cap N(j)$—for example, center qubit $4$ and right-middle qubit $5$, which both lie in the support of check $c_1 = \{1, 2, 4, 5\}$. Under a correlated $Z_4 Z_5$ fault event:
$$(g_{4,5})_{c_1} = (g_4)_{c_1} \oplus (g_5)_{c_1} = 1 \oplus 1 = 0$$
Because qubits $4$ and $5$ flip **simultaneously**, check $c_1$ experiences two simultaneous parity flips that cancel modulo 2, so check $c_1$ **never fires** from a $Z_4 Z_5$ event (only check $c_2 = \{3, 4, 6, 7\}$, which contains qubit $4$ but not qubit $5$, fires with signature $g_{4,5} = \texttt{0x6} \oplus \texttt{0x2} = \texttt{0x4}$).

By contrast, if $p_{4,5}$ is folded into independent single-qubit rates $p_{\text{eff}}(4)$ and $p_{\text{eff}}(5)$, the marginal check model assumes qubits $4$ and $5$ flip independently with probability $p_{4,5}$, predicting a spurious firing probability $2 p_{4,5}(1 - p_{4,5}) > 0$ on check $c_1$ and failing to capture the joint parity distribution across checks. This algebraic defect is completely eliminated in [`joint_dp_likelihood.py`](simulation/reference/noise_model/joint_dp_likelihood.py) and [`noise_model.cpp`](software/src/inference/noise_model.cpp) by evolving the exact $2^m$-state joint distribution using the true pair signatures $g_{ij} = g_i \oplus g_j$.

---

## References
- **[R4]** E. van den Berg, Z. K. Minev, A. Kandala, and K. Temme, "Probabilistic error cancellation with sparse Pauli-Lindblad models on noisy quantum processors," *Nature Physics*, vol. 19, pp. 1116–1121, 2023.
- **[R5]** S. T. Flammia and J. J. Wallman, "Efficient estimation of Pauli channels," *ACM Transactions on Quantum Computing*, vol. 1, no. 1, pp. 1–32, 2020.
- **[R8]** Y. Tomita and K. M. Svore, "Low-distance surface codes under realistic quantum noise," *Physical Review A*, vol. 90, no. 6, p. 062320, 2014.
