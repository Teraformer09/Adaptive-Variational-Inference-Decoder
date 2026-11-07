# 01 — CSS Surface Code Algebra and Tanner Geometry (`A01 / Q02`)

## 1. Symplectic Stabilizer Commutativity and Code Dimension
A Calderbank–Shor–Steane (CSS) stabilizer code on $n$ physical data qubits is specified by two binary parity-check matrices $H_X \in \mathbb{F}_2^{m_X \times n}$ and $H_Z \in \mathbb{F}_2^{m_Z \times n}$. Each row $c \in \{0, \dots, m_X - 1\}$ of $H_X$ defines an $X$-type stabilizer generator $S_c^X = \bigotimes_{v=0}^{n-1} X_v^{(H_X)_{c,v}}$ that detects phase-flip ($Z$) errors, while each row $c' \in \{0, \dots, m_Z - 1\}$ of $H_Z$ defines a $Z$-type stabilizer generator $S_{c'}^Z = \bigotimes_{v=0}^{n-1} Z_v^{(H_Z)_{c',v}}$ that detects bit-flip ($X$) errors.

For the stabilizer group $\mathcal{S} = \langle S_0^X, \dots, S_{m_X-1}^X, S_0^Z, \dots, S_{m_Z-1}^Z \rangle$ to form a valid commuting measurement group, every $X$-generator and every $Z$-generator must overlap on an even number of data qubits. In binary matrix algebra over $\mathbb{F}_2$, this symplectic orthogonality condition takes the form:
$$H_X H_Z^\top \equiv \mathbf{0}_{m_X \times m_Z} \pmod 2 \iff \forall c, c', \quad \left|\operatorname{supp}(H_{X,c}) \cap \operatorname{supp}(H_{Z,c'})\right| \equiv 0 \pmod 2$$

When the stabilizer generators are algebraically independent over $\mathbb{F}_2$, the dimension $k$ of the encoded logical Hilbert space $(\mathbb{C}^2)^{\otimes k}$ is determined by the $\mathbb{F}_2$-rank deficiency:
$$k = n - \operatorname{rank}_{\mathbb{F}_2}(H_X) - \operatorname{rank}_{\mathbb{F}_2}(H_Z)$$

---

## 2. Canonical Rotated $d=3$ and $d=5$ Planar Surface Codes
In the legacy codebase prior to the `A01 / Q02` repair, all four $X$-checks and $Z$-checks on the $3 \times 3$ lattice were defined as weight-4 plaquettes $\{\{0,1,3,4\}, \{1,2,4,5\}, \{3,4,6,7\}, \{4,5,7,8\}\}$, which fails CSS commutativity on adjacent faces sharing a single interior vertex. On a canonical rotated planar surface code of odd distance $d$ with $n = d^2$ data qubits indexed row-major $v = d \cdot r + c \in \{0, \dots, d^2 - 1\}$, the $(d^2 - 1)/2$ checks in each basis consist of $(d-1)^2/2$ weight-4 bulk plaquettes and $(d-1)$ weight-2 boundary digons.

For $d = 3$ (`[[9, 1, 3]]`, $n = 9, m_X = m_Z = 4$), the canonical parity-check matrices implemented in [`css_code_builder.py`](simulation/reference/code_model/css_code_builder.py), [`Hx.json`](architecture/01_code_model/Hx.json), and [`Hz.json`](architecture/01_code_model/Hz.json) are:
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

Here $c_0 = \{0, 1\}$ and $c_3 = \{7, 8\}$ are weight-2 top/bottom $X$-boundaries, $c_1 = \{1, 2, 4, 5\}$ and $c_2 = \{3, 4, 6, 7\}$ are weight-4 $X$-plaquettes, $z_1 = \{2, 5\}$ and $z_2 = \{3, 6\}$ are weight-2 right/left $Z$-boundaries, and $z_0 = \{0, 1, 3, 4\}$ and $z_3 = \{4, 5, 7, 8\}$ are weight-4 $Z$-plaquettes. Direct evaluation confirms $H_X H_Z^\top \equiv \mathbf{0} \pmod 2$ and $\operatorname{rank}_{\mathbb{F}_2}(H_X) = \operatorname{rank}_{\mathbb{F}_2}(H_Z) = 4$, giving $k = 9 - 4 - 4 = 1$ (and for $d = 5$, $n = 25, \operatorname{rank}_{\mathbb{F}_2}(H_X) = \operatorname{rank}_{\mathbb{F}_2}(H_Z) = 12, k = 1$).

---

## 3. Logical Operators, Homological Equivalence, and Code Distance
A logical Pauli operator $\bar{X} \in \mathbb{F}_2^n$ (respectively $\bar{Z} \in \mathbb{F}_2^n$) must commute with all $Z$-stabilizers (respectively $X$-stabilizers), must not belong to the stabilizer rowspace, and must anticommute with its logical partner:
$$H_Z \bar{X}^\top \equiv \mathbf{0} \pmod 2, \qquad H_X \bar{Z}^\top \equiv \mathbf{0} \pmod 2, \qquad \bar{X} \cdot \bar{Z}^\top \equiv 1 \pmod 2$$

On the canonical $d=3$ lattice, the left column $\operatorname{supp}(\bar{X}) = \{0, 3, 6\}$ and top row $\operatorname{supp}(\bar{Z}) = \{0, 1, 2\}$ intersect solely at corner qubit $0$ ($\bar{X} \cdot \bar{Z}^\top = 1$). The exact minimum distances over the homological cosets are:
$$d_X = \min_{x \in \ker(H_Z) \setminus \operatorname{rowspace}(H_X)} \operatorname{wt}(x) = 3, \qquad d_Z = \min_{z \in \ker(H_X) \setminus \operatorname{rowspace}(H_Z)} \operatorname{wt}(z) = 3, \qquad d = \min(d_X, d_Z) = 3$$

---

## 4. Degree-Padded Hardware Tanner Graph Representation
In the FPGA implementation ([`bp_engine.sv`](firmware/rtl/bp_engine.sv) and [`check_node_minsum.sv`](firmware/rtl/check_node_minsum.sv)), every check node instantiates a uniform 4-input comparator tree (`MAX_DEG = 4`). To accommodate both weight-2 boundary checks and weight-4 bulk checks without synthesized multiplexers, each check $c$ is described by a 4-element padded connectivity tuple $C_c = (v_{c,0}, v_{c,1}, v_{c,2}, v_{c,3})$ and a 4-bit active-edge mask $M_c \in \{0, 1\}^4$:
$$M_{c, d} = \begin{cases} 1, & \text{if } d < |\operatorname{supp}(H_c)|, \\ 0, & \text{otherwise.} \end{cases}$$
For the $d=3$ $X$-stabilizer graph exported in [`tanner_tables.json`](architecture/01_code_model/tanner_tables.json), $(M_0, M_1, M_2, M_3)_X = (\texttt{0b0011}, \texttt{0b1111}, \texttt{0b1111}, \texttt{0b0011})$, ensuring inactive padded edges are masked to $+\infty$ ($\texttt{0x7FFF}$) during check-node minimization.

---

## References
- **[R1]** A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," *Physical Review A*, vol. 86, no. 3, p. 032324, 2012.
- **[R3]** J. P. Bonilla Ataides, D. K. Tuckett, S. D. Bartlett, S. T. Flammia, and B. J. Brown, "The XZZX surface code," *Nature Communications*, vol. 12, p. 2172, 2021.
- **[R8]** Y. Tomita and K. M. Svore, "Low-distance surface codes under realistic quantum noise," *Physical Review A*, vol. 90, no. 6, p. 062320, 2014.
