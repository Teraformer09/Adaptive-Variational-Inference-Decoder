# 06 — Hardware Normalized Min-Sum BP Decoder and Cycle Latency Contract (`A06 / Q06`)

## 1. Bit-Exact Q7.8 Message-Passing Equations
Given an incoming syndrome vector $s \in \{0, 1\}^m$ and the active Q7.8 prior weight bank $\Gamma = (\Gamma_0, \dots, \Gamma_{n-1}) \in \mathbb{Z}_{16}^n$ latched from [`weight_map_reg.sv`](file:///home/tirtha/AVID_EPJ_Patches/firmware/rtl/weight_map_reg.sv), the hardware decoder ([`bp_engine.sv`](file:///home/tirtha/AVID_EPJ_Patches/firmware/rtl/bp_engine.sv) and its bit-exact C++/Python execution twins in [`cycle_accurate_bp_engine.cpp`](file:///home/tirtha/AVID_EPJ_Patches/software/src/custom_engine/cycle_accurate_bp_engine.cpp) and [`causal_stream_runner.py`](file:///home/tirtha/AVID_EPJ_Patches/simulation/reference/syndrome_stream/causal_stream_runner.py)) executes $K$ iterations ($t = 1, \dots, K$):

1. **Initialization (`ST_INIT`, 1 clock cycle)**:
   $$L_v^{(0)} = \Gamma_v, \qquad \mu_{v \to c}^{(0)} = \Gamma_v, \qquad \nu_{c \to v}^{(0)} = 0 \qquad \forall v \in \{0, \dots, n-1\}, \; c \in \{0, \dots, m-1\}$$
2. **Variable-to-Check Extrinsic Subtraction (`ST_V2C`, 1 clock cycle)**:
   $$\mu_{v \to c}^{(t)} = \operatorname{sat}_{\text{Q7.8}}\!\left(L_v^{(t-1)} - \nu_{c \to v}^{(t-1)}\right)$$
   Each signed 16-bit message $\mu_{v \to c}^{(t)}$ is converted to sign-magnitude representation $(\sigma_{v \to c}^{(t)}, m_{v \to c}^{(t)})$, where $\sigma_{v \to c}^{(t)} = \mathbb{I}[\mu_{v \to c}^{(t)} < 0] \in \{0, 1\}$ and $m_{v \to c}^{(t)} = \min(|\mu_{v \to c}^{(t)}|, \texttt{0x7FFF}) \in \{0, \dots, 32767\}$.
3. **Masked Normalized Min-Sum Check-to-Variable Update (`ST_WAIT_C2V`, 3 pipeline cycles)**:
   In [`check_node_minsum.sv`](file:///home/tirtha/AVID_EPJ_Patches/firmware/rtl/check_node_minsum.sv), each check $c$ applies its 4-bit active-edge mask $M_{c, d} \in \{0, 1\}$ ($d \in \{0, 1, 2, 3\}$) to mask inactive padded slots ($M_{c, d} = 0$) to neutral values:
   $$\tilde{\sigma}_{c, d} = M_{c, d} \cdot \sigma_{C_{c,d} \to c}^{(t)}, \qquad \tilde{m}_{c, d} = \begin{cases} m_{C_{c,d} \to c}^{(t)}, & \text{if } M_{c, d} = 1, \\ \texttt{0x7FFF}, & \text{if } M_{c, d} = 0. \end{cases}$$
   A 2-stage pairwise comparator tree extracts the first minimum $\min_1$, winning index $d^*$, and second minimum $\min_2$:
   $$d^* = \arg\min_{d \in \{0..3\}} \tilde{m}_{c, d}, \qquad \min_1 = \tilde{m}_{c, d^*}, \qquad \min_2 = \min_{d \neq d^*} \tilde{m}_{c, d}$$
   Normalized Min-Sum scaling by $\alpha = 0.75$ is computed using DSP-free shift-subtract arithmetic $\mathcal{S}_{0.75}(x) = x - (x \gg 2) = \lfloor \frac{3}{4} x \rfloor$:
   $$\sigma_{c \to C_{c,d}}^{(t)} = s_c \oplus \tilde{\sigma}_{c, d} \oplus \bigoplus_{d'=0}^3 \tilde{\sigma}_{c, d'}, \qquad m_{c \to C_{c,d}}^{(t)} = \mathcal{S}_{0.75}\!\left(\begin{cases} \min_2, & d = d^* \\ \min_1, & d \neq d^* \end{cases}\right)$$
   $$\nu_{c \to C_{c,d}}^{(t)} = \begin{cases} (-1)^{\sigma_{c \to C_{c,d}}^{(t)}} m_{c \to C_{c,d}}^{(t)}, & \text{if } M_{c, d} = 1, \\ 0, & \text{if } M_{c, d} = 0. \end{cases}$$
4. **Posterior Belief Accumulation (`ST_BELIEF_UPD`, 1 clock cycle)**:
   $$L_v^{(t)} = \operatorname{sat}_{\text{Q7.8}}\!\left(\Gamma_v + \sum_{c \in N(v)} \nu_{c \to v}^{(t)}\right)$$
5. **Hard Decision & Weighted Syndrome Tie-Break (`ST_HARD_DECIDE`, 1 clock cycle)**:
   $$\hat{e}_v = \mathbb{I}\!\left[L_v^{(K)} < 0\right] \in \{0, 1\}^n$$
   Whenever $H \hat{e} \not\equiv s \pmod 2$, the custom execution engine resolves symmetric trapping sets via weighted OSD-0 minimum-prior-cost syndrome selection:
   $$\hat{e}^* = \arg\min_{e \in \mathbb{F}_2^n : H e \equiv s \pmod 2} \sum_{v=0}^{n-1} e_v \max(1, \Gamma_v)$$

---

## 2. Deterministic Hardware Cycle Latency Contract
With the iteration counter check fused directly into `ST_BELIEF_UPD` in [`bp_engine.sv`](file:///home/tirtha/AVID_EPJ_Patches/firmware/rtl/bp_engine.sv), every decoding invocation executes a deterministic number of clock cycles $L(K)$ independent of syndrome weight or noise regime:
$$L(K) = 1_{\text{ST\_INIT}} + K \cdot \left(1_{\text{ST\_V2C}} + 3_{\text{ST\_WAIT\_C2V}} + 1_{\text{ST\_BELIEF\_UPD}}\right) + 1_{\text{ST\_HARD\_DECIDE}} = 2 + 5K \text{ cycles}$$

| Iteration Count $K$ | Fused Latency $L(K) = 2 + 5K$ | Legacy Unfused $L_{\text{leg}}(K) = 2 + 6K$ | Wall-Clock Time @ $250\,\text{MHz}$ ($4.0\,\text{ns/cycle}$) | Fraction of $1.0\,\mu\text{s}$ Round Budget |
| :---: | :---: | :---: | :---: | :---: |
| $K = 1$ | 7 cycles | 8 cycles | $28.0\,\text{ns}$ | $2.8\%$ |
| $K = 2$ | 12 cycles | 14 cycles | $48.0\,\text{ns}$ | $4.8\%$ |
| **$K = 3$ (Reference)** | **17 cycles** | **20 cycles** | **$68.0\,\text{ns}$** | **$6.8\%$** |
| $K = 4$ | 22 cycles | 26 cycles | $88.0\,\text{ns}$ | $8.8\%$ |
| $K = 5$ | 27 cycles | 32 cycles | $108.0\,\text{ns}$ | $10.8\%$ |

---

## References
- **[R2]** N. Liyanage et al., "Scalable quantum error correction for surface codes using FPGA," *arXiv:2301.08419*, 2023.
- **[R6]** J. Chen and M. P. C. Fossorier, "Near optimum universal belief propagation based decoding of low-density parity check codes," *IEEE Transactions on Communications*, vol. 50, no. 3, pp. 406–414, 2002.
- **[R9]** Xilinx Inc., *Zynq UltraScale+ Device Technical Reference Manual (UG1085)*, v2.4, 2023.
