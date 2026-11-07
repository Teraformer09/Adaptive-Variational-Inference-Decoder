# End-to-End Execution Flow

## 1. Round-Level Fast Path (`1 us` per QEC Round)
1. **Syndrome Arrival**: 64-bit packet arrives on `s_axis_syndrome` (`axis_syndrome_rx.sv`) carrying 4-bit $s_X$ and 4-bit $s_Z$.
2. **DMA Ingress Tap**: `axis_ingress_tap.sv` forwards a non-blocking copy to the lock-free PS DMA ring while the syndrome enters `bp_engine.sv`.
3. **Min-Sum BP + OSD-0**: `bp_engine.sv` executes $K$ message-passing iterations using active Q7.8 bank weights (`active_x[0..8]`, `active_z[0..8]`) and emits 9-bit corrections $(\hat{e}_X, \hat{e}_Z)$ via `axis_correction_tx.sv`.

## 2. Subwindow Fast Burst Detection (`<= 20` Rounds)
1. `avid_fast_detector_step()` updates a 16-round circular buffer of non-zero syndrome events on every round.
2. If the 16-round syndrome density exceeds `0.35` for `4` consecutive rounds, the detector triggers within $\le 20$ rounds (`20 us`).

## 3. Epoch-Level Slow Adaptation Loop (`256` Rounds in V1 Emulation / `10,000` Rounds in V2 Production)
1. `avid_dma_ring_consume_slot()` accumulates dual empirical histograms `hist_x[0..15]` and `hist_z[0..15]`.
2. `HistogramLBFGSSolver::solve()` runs two-pass projected L-BFGS over the 16-state Joint-DP likelihood.
3. `AvidR2D2LSTMController::step_epoch()` updates its 64-unit recurrent belief state and blends pair-projected LLRs.
4. `avid_hal_commit_weights_q78()` writes the 18 Q7.8 registers (`0x000..0x044`) into the inactive shadow bank and pulses `0x048` (`commit_Toggle`), which swaps banks exclusively when `bp_ready_in == 1`.
