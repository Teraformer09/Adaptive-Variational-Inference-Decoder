# Causal Epoch Scheduler & Fast Burst Detector

## 1. Strict Causality Invariant
Weights $\mathbf{w}^{(\tau)}$ estimated from syndrome histogram window $\mathcal{W}_\tau = [( \tau - 1 ) N_{\mathrm{win}}, \tau N_{\mathrm{win}} - 1]$ are committed at round $\tau N_{\mathrm{win}} + d_{\mathrm{commit}}$ and decode only rounds $r \ge \tau N_{\mathrm{win}} + d_{\mathrm{commit}}$. Zero future or in-window ground-truth parameters are ever accessed by the estimator or controller.

## 2. 16-Round Sliding-Subwindow Fast Burst Detector
Implemented in `software/src/control/causal_scheduler.h` and `software/src/control/causal_scheduler.cpp` (`avid_fast_burst_detector_t`):
- Maintains a 16-entry circular buffer `ring_nonzero[16]` recording whether each round produced a non-trivial syndrome ($s_X \ne 0 \lor s_Z \ne 0$).
- Computes sliding syndrome density $\rho_{16}(r) = \frac{1}{16}\sum_{j=0}^{15} \mathbb{I}[s^{(r-j)} \ne 0]$.
- When $\rho_{16}(r) \ge 0.35$ for `4` consecutive rounds, `avid_fast_detector_step()` increments `trigger_count` and asserts `burst_active = true` within at most $16 + 4 = 20$ rounds (`20 us`).
