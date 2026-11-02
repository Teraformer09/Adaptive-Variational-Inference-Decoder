# Fault Containment & Deterministic Fallback FSM (`A09 / Q18–Q19`, `SF9`)

## 1. 4-State Safety Fallback Machine (`fallback_fsm.cpp`)
To guarantee uninterrupted quantum error correction even when the slow estimation loop experiences DMA drops, non-convergence, or numerical anomalies, the runtime controller implements a deterministic 4-state safety FSM:

1. **`NORMAL_ADAPTIVE`**: Estimator converges within deadline; validated Q7.8 LLRs are committed to shadow bank and swapped atomically.
2. **`HOLD_LAST_GOOD`**: Entered on transient estimator timeout, single-window L-BFGS line-search failure, or DMA ring overflow. Retains current `active_mem[0..8]` weights without issuing `COMMIT`.
3. **`SAFE_STATIC_PRIOR`**: Entered if `HOLD_LAST_GOOD` persists for $> 3$ consecutive slow epochs or if any proposed LLR falls outside `[0x0080, 0x2000]`. Commits calibrated static baseline priors (`llr_static_q7_8 = 0x0468`, corresponding to $p = 0.012$).
4. **`DEGRADED_BYPASS`**: Entered on unrecoverable AXI-Lite bus fault or consecutive sequence gap $> 16$. Locks hardware to reset priors (`0x0400`).

## References
- **[R2]** N. Liyanage et al., *arXiv:2301.08419*, 2023.
- **[R9]** Xilinx Inc., *UG1085*, v2.4, 2023.
