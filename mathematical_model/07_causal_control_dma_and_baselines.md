# 07 — Dual-Timescale Causal Control, DMA Ownership, and Baselines B0–B4 (`A04 / Q01`, `A05 / Q11`, `A07 / Q13`, `A09 / Q18`)

## 1. Dual-Timescale Causal Commit Law
In the AVID dual-timescale architecture, the fast decoding loop in Programmable Logic (PL @ $250\,\text{MHz}$) processes every syndrome round $t \in \mathbb{N}$ within $T_{\text{decode}} = 68.0\,\text{ns} \ll T_{\text{round}} = 1000.0\,\text{ns}$ using the currently active weight bank $\Gamma_{\text{active}}^{(t)}$. Concurrently, the slow estimation loop in the Processing System (PS ARM Cortex-A53 @ $1.2\,\text{GHz}$) observes syndrome bursts over window $[t_0, t_1]$.

Let $\Delta t_{\text{dma}}$ denote the DMA ring slot transfer latency, $\Delta t_{\text{est}}$ the Joint-DP L-BFGS optimization latency, and $\Delta t_{\text{axi}}$ the AXI-Lite shadow register write latency. By physical causality, a weight update computed from observations in $[t_0, t_1]$ cannot take effect until round:
$$t_{\text{commit}} = t_1 + \left\lceil \frac{\Delta t_{\text{dma}} + \Delta t_{\text{est}} + \Delta t_{\text{axi}}}{T_{\text{round}}} \right\rceil$$
During all intermediate rounds $t \in (t_1, t_{\text{commit}})$, the fast decoder continues uninterrupted using the previous epoch's active weights $\Gamma_{\text{active}}^{(t_1)}$.

---

## 2. Atomic Shadow-to-Active Register Swap and Safety Fallback FSM
In [`weight_map_reg.sv`](firmware/rtl/weight_map_reg.sv), the PS writes candidate weights to `shadow_mem[0..8]` (`0x000..0x020`) and writes `1` to the `COMMIT` strobe (`0x048`), latching `commit_pending = 1`. The hardware state update obeys the atomic, single-cycle idle-gated transition:
$$\Gamma_{\text{active}}^{(c+1)} = \begin{cases}
\Gamma_{\text{shadow}}^{(c)}, & \text{if } \texttt{commit\_pending}^{(c)} = 1 \land \texttt{bp\_ready\_in}^{(c)} = 1, \\
\Gamma_{\text{active}}^{(c)}, & \text{otherwise.}
\end{cases}$$

To guarantee fault containment under DMA overflows, non-convergence, or deadline misses, [`fallback_fsm.cpp`](software/src/control/fallback_fsm.cpp) enforces a deterministic 4-state safety automaton over state space $\{\texttt{NORMAL\_ADAPTIVE}, \texttt{HOLD\_LAST\_GOOD}, \texttt{SAFE\_STATIC\_PRIOR}, \texttt{DEGRADED\_BYPASS}\}$:
$$\Gamma_{\text{selected}}^{(k)} = \begin{cases}
\Gamma_{\text{candidate}}^{(k)}, & \text{if } \text{estimator\_ok} \land \text{deadline\_ok} \land \text{bounds\_ok} \quad (\texttt{NORMAL\_ADAPTIVE}), \\
\Gamma_{\text{last\_good}}, & \text{if transient fault and } n_{\text{holds}} < 3 \quad (\texttt{HOLD\_LAST\_GOOD}), \\
\Gamma_{\text{static}}, & \text{if } \neg\text{bounds\_ok} \lor n_{\text{holds}} \ge 3 \quad (\texttt{SAFE\_STATIC\_PRIOR}).
\end{cases}$$

---

## 3. Mathematical Formulations of the 5 Shared-Input Baselines (`B0–B4`)
Implemented in [`simulation/reference/baselines/`](simulation/reference/baselines), all five baseline controllers consume the identical 16-bin empirical syndrome histogram $\hat{P}(s)$ and feed the identical 17-cycle Q7.8 Normalized Min-Sum BP decoder:

1. **`B0` — Static Uniform-Prior BP ([`b0_static_bp.py`](simulation/reference/baselines/b0_static_bp.py))**:
   Fixes all variable-node priors at the calibrated $t = 0$ prior $p_0 = 0.012$:
   $$\Gamma_v^{\text{B0}} = \operatorname{round}\!\left(256.0 \cdot \ln\frac{1 - p_0}{p_0}\right) = \texttt{0x0468}$$
2. **`B1` — Legacy 4-Check Marginal L-BFGS ([`b1_marginal_lbfgs.py`](simulation/reference/baselines/b1_marginal_lbfgs.py))**:
   Minimizes the rank-4 marginal Bernoulli check cross-entropy over empirical check firing rates $\hat{q}_c = \sum_{s : s_c = 1} \hat{P}(s)$:
   $$\mathcal{L}_{\text{B1}}(\theta) = -\sum_{c=0}^3 \left[\hat{q}_c \ln q_c(\theta) + (1 - \hat{q}_c) \ln(1 - q_c(\theta))\right] + \frac{\lambda}{2}\|\theta - \theta_0\|_2^2$$
3. **`B2` — Check-Marginal Proportional-Integral (PI) Controller ([`b2_observable_pid.py`](simulation/reference/baselines/b2_observable_pid.py))**:
   Updates each qubit's estimated error probability $\hat{p}_v^{(k)}$ from the average firing rate of its incident checks $N(v)$ via a discrete PI control law:
   $$e_v^{(k)} = \frac{1}{2.5 |N(v)|} \sum_{c \in N(v)} \hat{q}_c^{(k)} - \hat{p}_v^{(k-1)}, \qquad \hat{p}_v^{(k)} = \operatorname{clip}\!\left(\hat{p}_v^{(k-1)} + K_p e_v^{(k)} + K_i \sum_{\tau \le k} e_v^{(\tau)}, \; 10^{-4}, \; 0.40\right)$$
4. **`B3` — Empirical Syndrome Covariance Tracker ([`b3_covariance_tracker.py`](simulation/reference/baselines/b3_covariance_tracker.py))**:
   Computes the empirical syndrome mean $\mu_c = \mathbb{E}[s_c]$ and pairwise correlation matrix $R_{c_1, c_2} = \mathbb{E}[s_{c_1} s_{c_2}]$ with covariance $C_{c_1, c_2} = R_{c_1, c_2} - \mu_{c_1}\mu_{c_2}$, inverting the parity correlation formula for degree-2 bulk variables $N(v) = \{c_1, c_2\}$:
   $$\hat{p}_v^{\text{B3}} = \frac{1}{2} - \frac{1}{2}\sqrt{\max\!\left(0.01, \; 1 - \frac{4 \max(0, C_{c_1, c_2})}{1 - 2(\mu_{c_1} + \mu_{c_2}) + 4 R_{c_1, c_2}}\right)}$$
5. **`B4` — Instantaneous Ground-Truth Oracle BP ([`b4_oracle_bp.py`](simulation/reference/baselines/b4_oracle_bp.py))**:
   Receives the exact instantaneous ground-truth parameter vector $\theta_{\text{true}}(t)$ with zero estimation latency and computes the ideal Q7.8 LLR bank $\Gamma^{\text{B4}}(\theta_{\text{true}}(t))$.

---

## References
- **[R1]** A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," *Physical Review A*, vol. 86, no. 3, p. 032324, 2012.
- **[R2]** N. Liyanage et al., "Scalable quantum error correction for surface codes using FPGA," *arXiv:2301.08419*, 2023.
- **[R9]** Xilinx Inc., *Zynq UltraScale+ Device Technical Reference Manual (UG1085)*, v2.4, 2023.
