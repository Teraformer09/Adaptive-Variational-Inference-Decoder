# Dual-Timescale Fast/Slow Control Partition (`A04 / Q01`, `SF2`, `SF5`)

## 1. Architectural Partitioning
The AVID baseline architecture strictly decouples the hard real-time per-round decoding path (Programmable Logic) from the asynchronous statistical noise-tracking loop (Processing System):

1. **Fast Loop (PL Fabric @ 250 MHz, Round Period $T_{\text{round}} = 1000\,\text{ns}$)**:
   - Consumes each 64-bit syndrome packet `s_t` immediately upon arrival.
   - Runs $K=3$ iterations of Normalized Min-Sum Belief Propagation using the **currently active** Q7.8 LLR weight bank (`active_weights[0..8]`).
   - Emits the 9-bit data-qubit correction vector $\hat{e}_t$ in **68 ns** (17 cycles @ 250 MHz for fused $L(K)=2+5K$, or 80 ns / 20 cycles for legacy $L_{\text{legacy}}(K)=2+6K$).
   - **Never waits** for the ARM Cortex-A53 estimator.
2. **Slow Loop (ARM Cortex-A53 @ 1.2 GHz)**:
   - **Fast Regime-Change Trigger ($N_{\text{fast}} = 64$ rounds = $64\,\mu\text{s}$)**: Computes the log-likelihood ratio / total variation distance between the 64-round empirical syndrome histogram and the active model distribution $P_{\theta_{\text{active}}}(s)$.
   - **Full MAP L-BFGS Re-Estimation ($N_{\text{slow}} = 2048$ rounds = $2.048\,\text{ms}$, or immediately upon fast trigger)**: Solves the 16-state Joint-DP MAP objective, quantizes the updated 9 LLRs to Q7.8, writes them to the **shadow** register bank (`0x000..0x020`), and pulses the atomic `COMMIT` strobe (`0x048`).

## References
- **[R2]** N. Liyanage et al., *arXiv:2301.08419*, 2023.
- **[R6]** J. Chen and M. P. C. Fossorier, *IEEE Trans. Commun.*, vol. 50, no. 3, pp. 406–414, 2002.
- **[R7]** D. C. Liu and J. Nocedal, *Mathematical Programming*, vol. 45, pp. 503–528, 1989.
