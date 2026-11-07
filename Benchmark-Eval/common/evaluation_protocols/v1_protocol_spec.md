# AVID V1 Dual-Emulator & Runtime Evaluation Protocol

1. **Shared Syndrome Stream Invariant**: Every controller variant (`Static B0`, `Marginal B1`, `Observable PID B2`, `Covariance B3`, `AVID V1`, `Oracle B4`) is evaluated on the exact same pseudo-random fault and syndrome realization for a given `(scenario, seed)` pair.
2. **Strict Causal Lag**: Weights estimated from epoch window $\tau$ (`N_win = 256` rounds) are committed at round $(\tau + 1) N_{\mathrm{win}}$ (or $(\tau + 2) N_{\mathrm{win}}$ under `delayed_weight_commit`) and never decode past or in-window rounds.
3. **Zero Scenario-Name Shortcuts**: Neither `lbfgs_solver.cpp` nor `r2d2_lstm_controller.h` inspects the scenario name string; all adaptation is driven strictly by the 16-bin empirical syndrome histograms `hist_x[0..15]` and `hist_z[0..15]`.
