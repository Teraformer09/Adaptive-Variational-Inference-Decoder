# Superseded Pre-Remediation Provenance Record (`b001b8d`)

- **Frozen Git Commit**: `b001b8d68ec20b113af0b0c3f595c46305a4e2d5`
- **Git Tag**: `pre-remediation-audit-2026-11-08`
- **Audit Date**: 8 November 2026
- **Status**: **SUPERSEDED — Retained strictly for historical auditability.**

## Why These Pre-Remediation Reports Were Superseded

1. **Controller & Pair-Projection Coupling (`M1–M6`, `M18`)**:
   - `software/src/control/r2d2_lstm_controller.h` initialized weights from a fixed PRNG (`0x5232443257423634ULL`) rather than loading trained checkpoints, left `last_q_value`, `last_td_weight`, and `rec_k` unused at runtime, and bounded gains to `[0.92, 1.08]`, making the `r2d2_gains[4] < 0.90` branch in `noise_model.cpp` unreachable.
   - `NoiseModel::to_pair_projected_q7_8_llr()` changed its `pair_dominant` threshold rule based on `r2d2_gains != nullptr` and hardcoded qubit 4 / pair `(4,5)`.
2. **Estimator Ground-Truth Vector Mismatch (`M8`)**:
   - `avid_daemon.cpp` set `true_theta_z(9+k) = 0.004` on the 11 non-`(4,5)` pairs that were not injected by the built-in generator (`p = 0.0`).
3. **Seed & Metric Provenance Drift (`M10`, `N2`, `N3`)**:
   - The 10-scenario integration runner used `seed = 2026 + idx * 17` (`2060` for `correlated_noise`), while the V1 ablation runner used `seed = 2026`, causing conflicting numbers when both were labeled `seed = 2026`.
   - `Benchmark-Eval/Baseline_V0/BM1_MultiSeed_Statistical_Significance/run_bm1.py` cached seed-0 L-BFGS weights across seeds `1..14`, and `run_bm2.py` used linear interpolations for stages `A4–A6`.
4. **Hardware RTL & Fast-Detector Fixes (`N1`, `N4`, AXI `W`-before-`AW`)**:
   - `weight_map_reg.sv` required `W`-before-`AW` buffering and `BVALID`/`RVALID` stall gating.
   - `bp_engine.sv` `osd_corr_c` used input `channel_llr` instead of post-BP soft beliefs `belief[v]`.
   - `avid_daemon.cpp` discarded the trigger return value of `avid_fast_detector_step()`.

All active reports under `reports/` and `Benchmark-Eval/` have been regenerated from the remediated `Baseline V0` and `Milestone V1` codebase.
