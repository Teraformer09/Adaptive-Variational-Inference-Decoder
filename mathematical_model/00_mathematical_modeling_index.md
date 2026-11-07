# 00 — End-to-End Mathematical Modeling Index (`AVID_EPJ_Patches`)

## Overview
This documentation suite presents the complete, end-to-end mathematical modeling of the **Adaptive Variational Inference Decoder (AVID)** baseline architecture (`AVID_EPJ_Patches`, implementing Policy Items `A01–A09` and Checklist `Q01–Q21`). Each chapter formulates one mathematical layer of the closed-loop quantum error correction pipeline—from the algebraic topology of the rotated CSS planar surface code through the exact $2^m$-state correlated dynamic programming likelihood, structural identifiability theory, quasi-Newton optimization, signed Q7.8 fixed-point quantization, cycle-exact Normalized Min-Sum Belief Propagation (BP), and dual-timescale causal hardware control.

---

## Document Index

1. **[`01_CSS_Surface_Code_Algebra_and_Tanner_Geometry.md`](docs/01_CSS_Surface_Code_Algebra_and_Tanner_Geometry.md)**
   Formulates the symplectic commutativity invariant $H_X H_Z^\top \equiv \mathbf{0} \pmod 2$, $\mathbb{F}_2$-rank code dimension $k = n - \operatorname{rank}_{\mathbb{F}_2}(H_X) - \operatorname{rank}_{\mathbb{F}_2}(H_Z)$, canonical rotated $d=3$ (`[[9, 1, 3]]`) and $d=5$ (`[[25, 1, 5]]`) stabilizer matrices, logical operator coset distance $d_X = d_Z = d$, and degree-padded hardware Tanner graph masks.
2. **[`02_Correlated_Fault_Channel_and_Observation_Algebra.md`](docs/02_Correlated_Fault_Channel_and_Observation_Algebra.md)**
   Defines the physical fault mechanisms—single-qubit phase/bit flips ($g_i = H(:, i)$), nearest-neighbor dispersive $Z_i Z_j$ crosstalk pairs ($g_{ij} = g_i \oplus g_j$), and syndrome readout bit-flips ($d_t = s_t \oplus s_{t-1}$)—and proves algebraically why folding correlated pair probabilities into independent single-qubit rates fails on shared checks.
3. **[`03_Exact_Joint_DP_Likelihood_and_Coset_Gradients.md`](docs/03_Exact_Joint_DP_Likelihood_and_Coset_Gradients.md)**
   Derives the exact $2^m$-state group-algebra $\mathbb{R}[\mathbb{F}_2^m]$ XOR-convolution dynamic programming recurrence $P^{(k)}(s)$, its dual Walsh–Hadamard spectral representation, the proof of the Exact Single-Pass Coset-Inversion Gradient Theorem, and the prior-tethered MAP cross-entropy objective.
4. **[`04_Structural_Identifiability_Gauge_Orbits_and_Fisher_Rank.md`](docs/04_Structural_Identifiability_Gauge_Orbits_and_Fisher_Rank.md)**
   Proves the structural rank theorem for the $2^m \times E$ Jacobian $J(\theta)$ and Fisher Information Matrix $\mathcal{I}(\theta)$, partitions the 21 physical fault mechanisms into 9 identifiable non-zero syndrome signature orbits and 1 zero-signature boundary orbit, and establishes full-rank ($\operatorname{rank} = 21$, $\kappa = 234.05$) convergence under Gaussian prior regularization.
5. **[`05_LBFGS_Optimization_XOR_Marginalization_and_Q78_Numerics.md`](docs/05_LBFGS_Optimization_XOR_Marginalization_and_Q78_Numerics.md)**
   Details the zero-allocation two-loop L-BFGS quasi-Newton recursion with projected Armijo backtracking line search, exact associative XOR marginalization $p_{\text{eff}}(v) = p_v \boxplus \bigoplus_{e \ni v} p_e$ to variable-node priors, and signed 16-bit Q7.8 fixed-point quantization.
6. **[`06_Hardware_MinSum_BP_Decoder_and_Cycle_Latency_Contract.md`](docs/06_Hardware_MinSum_BP_Decoder_and_Cycle_Latency_Contract.md)**
   Formulates the bit-exact Q7.8 Normalized Min-Sum BP equations, active-edge masking for mixed weight-2/weight-4 checks, DSP-free shift-subtract scaling $\mathcal{S}_{0.75}(x) = x - (x \gg 2)$, weighted OSD-0 residual syndrome tie-breaking, and the exact $L(K) = 2 + 5K = 17$-cycle ($68.0\,\text{ns}$ @ $250\,\text{MHz}$) hardware latency contract.
7. **[`07_Dual_Timescale_Causal_Control_DMA_and_Baselines_B0_B4.md`](docs/07_Dual_Timescale_Causal_Control_DMA_and_Baselines_B0_B4.md)**
   Models the dual-timescale causal epoch commit law $t_{\text{commit}}$, `bp_ready_in`-gated atomic shadow-to-active register swap, 4-state DMA ring buffer ownership FSM, 4-state safety fallback FSM, and the mathematical equations of the 5 shared-input baselines (`B0–B4`).
8. **[`08_Weighted_Bellman_R2D2_and_World_Model_Math.md`](docs/08_Weighted_Bellman_R2D2_and_World_Model_Math.md)**
   Formulates the Non-Vision Bipartite Graph Perception layer, `R2D2-Graph-RSSM` with burn-in warmup, Kapturowski invertible value rescaling $h(u)$, Uncertainty-Weighted $n$-Step Bellman TD Loss ($\mathcal{L}_{\text{WB-R2D2}}$), Pair-Aware Logical-Triangle LLR Projection, and Rank-4 Regime LoRA Adaptation.
9. **[`mathematical_modeling_complete.md`](docs/mathematical_modeling_complete.md)** & **[`identifiability_proof.md`](docs/identifiability_proof.md)**
   Consolidated single-file mathematical reference and structural identifiability proof.

---

## References
- **[R1]** A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," *Physical Review A*, vol. 86, no. 3, p. 032324, 2012.
- **[R2]** N. Liyanage et al., "Scalable quantum error correction for surface codes using FPGA," *arXiv:2301.08419*, 2023.
- **[R3]** J. P. Bonilla Ataides, D. K. Tuckett, S. D. Bartlett, S. T. Flammia, and B. J. Brown, "The XZZX surface code," *Nature Communications*, vol. 12, p. 2172, 2021.
- **[R4]** E. van den Berg, Z. K. Minev, A. Kandala, and K. Temme, "Probabilistic error cancellation with sparse Pauli-Lindblad models on noisy quantum processors," *Nature Physics*, vol. 19, pp. 1116–1121, 2023.
- **[R5]** S. T. Flammia and J. J. Wallman, "Efficient estimation of Pauli channels," *ACM Transactions on Quantum Computing*, vol. 1, no. 1, pp. 1–32, 2020.
- **[R6]** J. Chen and M. P. C. Fossorier, "Near optimum universal belief propagation based decoding of low-density parity check codes," *IEEE Transactions on Communications*, vol. 50, no. 3, pp. 406–414, 2002.
- **[R7]** D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization," *Mathematical Programming*, vol. 45, pp. 503–528, 1989.
- **[R8]** Y. Tomita and K. M. Svore, "Low-distance surface codes under realistic quantum noise," *Physical Review A*, vol. 90, no. 6, p. 062320, 2014.
- **[R9]** Xilinx Inc., *Zynq UltraScale+ Device Technical Reference Manual (UG1085)*, v2.4, 2023.
