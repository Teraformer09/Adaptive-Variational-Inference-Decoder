# FPGA Normalized Min-Sum BP Cycle-Exact Latency Contract (`A06 / Q06`)

## 1. Finite State Machine Cycle Accounting (`bp_engine.sv`)
At fabric clock $f_{\text{clk}} = 250\,\text{MHz}$ ($T_{\text{clk}} = 4.0\,\text{ns}$), each syndrome packet triggers the deterministic FSM in `bp_engine.sv`:

1. **`ST_INIT` (1 cycle)**: Latches `syndrome_in[3:0]` and copies `active_weights[0..8]` into `posterior[0..8]` and initial variable-to-check messages `v2c_msg`.
2. **Per-Iteration Loop ($k = 0 \dots K-1$)**:
   - **`ST_V2C` (1 cycle)**: Assembles signed magnitude and sign vectors for each check node and asserts `cn_valid_in`.
   - **`ST_WAIT_C2V` (3 cycles)**: Pipelined `check_node_minsum.sv` stages (`s1_valid -> s2_valid -> valid_out`) compute $\min_1, \min_2$, apply normalization $\alpha = 0.75$ (`val - (val >> 2)`), and mask inactive edges via `ACTIVE_EDGES` (`4'b0011` for weight-2 boundary checks, `4'b1111` for weight-4 bulk checks).
   - **`ST_BELIEF_UPD` (1 cycle)**: Accumulates incoming `c2v_msg` into variable-node posteriors with Q7.8 saturation (`[-32768, +32767]`), subtracts extrinsic messages for the next iteration, and advances `iter_cnt` (or transitions to `ST_HARD_DECIDE` on final iteration $k = K - 1$).
3. **`ST_HARD_DECIDE` (1 cycle)**: Slices sign bits `posterior[v] < 0` into `correction_out[8:0]` and asserts `correction_valid`.

## 2. Exact Closed-Form Cycle Formula $L(K)$
- **Fused FSM (`L(K) = 2 + 5K` cycles)**:
  - $K = 1$: $7\text{ cycles} = 28.0\,\text{ns}$
  - $K = 2$: $12\text{ cycles} = 48.0\,\text{ns}$
  - $K = 3$: $17\text{ cycles} = 68.0\,\text{ns}$ (canonical reference operating point)
  - $K = 4$: $22\text{ cycles} = 88.0\,\text{ns}$
  - $K = 5$: $27\text{ cycles} = 108.0\,\text{ns}$
- **Legacy Unfused FSM (`L_legacy(K) = 2 + 6K` cycles, separate `ST_ITER_CHECK`)**:
  - $K = 3$: $20\text{ cycles} = 80.0\,\text{ns}$

Both modes are supported and verified bit-for-bit in `cycle_accurate_bp_engine.cpp` and `bp_engine.sv`.

## References
- **[R2]** N. Liyanage et al., *arXiv:2301.08419*, 2023.
- **[R6]** J. Chen and M. P. C. Fossorier, *IEEE Trans. Commun.*, vol. 50, no. 3, pp. 406–414, 2002.
