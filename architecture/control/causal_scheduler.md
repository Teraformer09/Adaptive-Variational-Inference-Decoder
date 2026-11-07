# Causal Epoch Scheduler & Fast Burst Detector

## 1. Strict Causality Invariant
Weights $\mathbf{w}^{(	au)}$ estimated from syndrome histogram window $\mathcal{W}_	au = [( 	au - 1 ) N_{\mathrm{win}}, 	au N_{\mathrm{win}} - 1]$ are committed at round $	au N_{\mathrm{win}} + d_{\mathrm{commit}}$ and decode only rounds $r \ge 	au N_{\mathrm{win}} + d_{\mathrm{commit}}$. Zero future or in-window ground-truth parameters are ever accessed by the estimator or controller.

## 2. 16-Round Sliding-Subwindow Fast Burst Detector
Implemented in `software/src/control/causal_scheduler.h` and `software/src/control/causal_scheduler.cpp` (`avid_fast_burst_detector_t`):
- Maintains a 16-entry circular buffer `ring_nonzero[16]` recording whether each round produced a non-trivial syndrome ($s_X 
e 0 \lor s_Z 
e 0$).
- Computes sliding syndrome density $ho_{16}(r) = rac{1}{16}\sum_{j=0}^{15} \mathbb{I}[s^{(r-j)} 
e 0]$.
- When $ho_{16}(r) \ge 0.35$ for `4` consecutive rounds, `avid_fast_detector_step()` increments `trigger_count` and asserts `burst_active = true` within at most $16 + 4 = 20$ rounds (`20 us`).
