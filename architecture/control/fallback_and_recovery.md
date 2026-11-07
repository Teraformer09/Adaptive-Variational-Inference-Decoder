# Fault Containment, Fallback FSM & Reset Recovery

Implemented in `software/src/control/fallback_fsm.h` and `software/src/control/fallback_fsm.cpp`:
1. **Non-Finite / Non-Convergent Estimator Guard**: If L-BFGS produces `NaN`/`Inf`, negative NLL, or exceeds gradient bounds (tested in `fallback_safety_trigger` via fault injection at Epoch 1), the daemon increments `fallback_events` and reverts to the safe prior Q7.8 weight bank (`+4.40` LLR = `0x0466`).
2. **Busy-Decoder CDC Lock**: `weight_map_reg.sv` latches `commit_pending` if `commit_reg` (`0x048`) is pulsed while `bp_ready_in == 0`, deferring `bank_sel <= ~bank_sel` until the active decode completes.
3. **Mid-Flight Reset Recovery**: Asserting `aresetn == 0` clears `epoch_count` (`0x050`), resets `bank_sel = 0`, and returns the decoder FSM to `ST_IDLE` without deadlock (`reset_recovery` scenario).
