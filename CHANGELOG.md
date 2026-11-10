# Changelog

## [1.2.0-v1-audit-verified] — 2026-11-11

### Added
- **Decoder-in-the-Loop R2D2 Retraining & 19,744-Parameter Checkpoint (`training/`, `checkpoints/`)**:
  - Updated `training/algorithms/r2d2_trainer.py` to evaluate all 6 candidate actions ($a_0 \dots a_5$: `STATIC_PRIOR_K3`, `MODERATE_PROJ_K4`, `AGGRESSIVE_PROJ_K5`, `DAMPED_PRIOR_K4`, `FAST_RECALIBRATE_K5`, `SAFE_FALLBACK_K3` with $K \in \{3, 4, 5\}$) directly through the hardware-equivalent $d=3$ surface-code decoder (`decode_round_with_action_and_gains`), measuring decoded logical failures, syndrome satisfaction, and cycle penalty.
  - Exported verified `19,744`-parameter checkpoint (`19.28 KB` `int8` / `77.12 KB` `fp32`, `157,984 B` binary) to `checkpoints/r2d2_v1_trained.{bin,json,onnx}` with `83.4%` agreement and all 6 actions actively selected.
- **10-Seed (`2026..2035`) & 5-Seed × 7-Scenario Comparative Benchmarks (`Benchmark-Eval/V1_Scientific_Evaluation/`)**:
  - Added memoryless `Joint-DP Projection` (`joint_dp_proj`) alongside `Static B0` and `AVID V1` (`avid_r2d2`) in the 10-seed headline benchmark table (`Benchmark-Eval/V1_Scientific_Evaluation/multi_seed/README.md`) and verified that `avid_r2d2` beats both `joint_dp_proj` and `static` across all 4 10-seed scenarios (`-31.21%` to `-45.89%` LER reduction) and across 7 of 7 scenarios (`3009` vs. `3114` vs. `4713` total failures) in the 5-seed × 7-scenario generalization suite.
- **Executable Gain-Modulation & Auto-Build Pytest Fixtures (`tests/`)**:
  - Added `test_per_qubit_gains_materially_change_decoded_failures` in `tests/test_decoder_action_effect.py` (`36/36` pytest tests passing) and added `tests/conftest.py` so running `pytest` on a clean checkout automatically builds any missing C++/Verilator binaries.

### Fixed
- **SystemVerilog RTL Fast-Path Fixes (`firmware/rtl/`)**:
  - `axis_correction_tx.sv`: Added a 2-entry skid buffer (`buf0`/`buf1`) and `accept_ready` handshake so 0 packets are dropped under arbitrary `m_axis_corr_tready` backpressure (verified 8/8 preserved under a 5-cycle stall), and gated `m_axis_corr_tlast` / `epoch_irq` to fire strictly on the 256-round epoch boundary (`round_cnt_q == 8'hFF`).
  - `weight_map_reg.sv`: Locked shadow register writes (`shadow_write_locked = swap_pending_q && !commit_pulse`) while a bank swap is pending so `commit_pulse` never commits torn shadow weights.
  - `bp_engine.sv`: Added single-weight fallback in Mode 0 (`cfg_osd_mode == 0`) so pure 9-LLR BP satisfies all `15/15` non-zero single-basis syndromes while preserving distinct pair behavior vs. Mode 1 (`17` cycles) and Mode 2 (`19` cycles).
- **Fail-Closed Executable-Relative Checkpoint Loading (`deployment/cpp/model_loader.cpp`, `software/src/inference/avid_daemon.cpp`)**:
  - Removed hardcoded `/home/tirtha/...` paths, resolved `checkpoints/r2d2_v1_trained.bin` relative to `/proc/self/exe`, passed explicit `--checkpoint` in all evaluation scripts, and made `avid_daemon` exit with code `2` if `controller_mode == "avid_r2d2"` and the checkpoint fails to load.
- **Pure Pass-Through C++ R2D2 Wrapper & Active Per-Qubit Gains (`software/src/`)**:
  - Kept `software/src/control/r2d2_lstm_controller.h` as a pure pass-through wrapper around `runtime_.step(features)` (zero C++ `if/else` action overrides) and refined `to_pair_projected_q7_8_llr` (`software/src/inference/noise_model.cpp`) so `qubit_gains[9]` directly modulate single-qubit and adjacent-pair LLRs without cold-neighbor pair overwrite on multi-pair crosstalk (`heldout_multi_pair_12_67`).
- **Measured RTL Parity & Expanded Causal-Blind Audit (`Benchmark-Eval/`)**:
  - Replaced hardcoded `rtl_vs_ref_mismatch: 0` in `Benchmark-Eval/V1_Integration/run_v1_integration.py` with measured hardware metrics (`avid_v1_integration_20261110_204636_s2026`) and expanded `verify_zero_scenario_shortcuts()` in `v1_acceptance_gates.py` to scan all estimator, noise-model, controller, and daemon files across all 13 scenario names.

---

## [1.1.0-v1-remediated] — 2026-11-08

### Added
- **Python-to-C++ Recurrent R2D2 Training & Deployment Pipeline**:
  - Added `training/` (`models/recurrent_encoder.py`, `models/q_value_head.py`, `models/r2d2_network.py`, `replay/sequence_sampler.py`, `replay/prioritized_replay.py`, `algorithms/loss.py`, `algorithms/target_network.py`, `algorithms/r2d2_trainer.py`, `export/export_checkpoint.py`, `evaluation/evaluate_policy.py`) implementing the Kapturowski et al. Dueling Recurrent R2D2 Q-Network over 6 discrete control actions ($a_0 \dots a_5$) and 9 per-qubit logit gain residuals.
  - Added `deployment/cpp/` (`model_loader.h/.cpp`, `r2d2_runtime.h/.cpp`) and `deployment/model_schema/r2d2_schema.json` with cryptographic SHA-256 checkpoint verification (`checkpoints/r2d2_v1_trained.bin`, `.json`, `.onnx`) and `< 1e-12` Python-to-C++ numerical trajectory parity.
- **Held-Out Generalization & Scientific Evaluation (`Benchmark-Eval/V1_Scientific_Evaluation/`)**:
  - Added held-out noise families (`heldout_multi_pair_12_67`, `heldout_spatial_3qubit_streak`, `heldout_phenomenological_readout`), exact Minimum-Weight Perfect Matching (`MWPM`) baseline, unseen evaluation seeds (`2026..2035`), and exact two-sided McNemar + Wilcoxon statistical hypothesis tests.
- **Executable Verification Suites (`tests/`)**:
  - Added `tests/test_training.py`, `tests/test_checkpoint_export.py`, `tests/test_cpp_python_parity.py`, and `tests/test_decoder_action_effect.py`.
- **Google Cloud Shell Native Verilator 5.020 + QEMU ARM Cortex-A9 Integration Bundle**:
  - Added full immutable run evidence bundle and archived pre-remediation reports under `reports/archive/pre_remediation_b001b8d/`.

### Fixed
- **SystemVerilog RTL Hardware Fixes (`firmware/rtl/`)**:
  - Repaired `weight_map_reg.sv` to latch `AW` and `W` channels independently (`W`-before-`AW` safe).
  - Repaired `bp_engine.sv` and `axis_correction_tx.sv` to perform dynamic lowest-LLR-sum OSD-0 pair selection across all 12 Tanner-graph edges.
- **General 12-Pair Syndrome-Alias MAP Projection (`software/src/inference/noise_model.cpp`)**:
  - Replaced hardcoded `(3,4,5)` / `(4,5)` pair checks with general 12-pair syndrome-alias log-odds projection (`nm_project_to_9_marginal_llrs`) controlled by continuous controller pair-weighting $\gamma_{\mathrm{pair}}$ and per-qubit gain residuals.
- **Closed-Loop Daemon Coupling (`software/src/inference/avid_daemon.cpp`)**:
  - Wired R2D2 actions $a_0 \dots a_5$ to dynamically update hardware register `0x054` and $\gamma_{\mathrm{pair}}$, wired `avid_fast_detector_step()` to trigger immediate sub-epoch adaptation on bursts, and wired independent estimator fault detection (`NaN`/`Inf`, bounds, KL divergence) into `fallback_fsm.cpp`.

---

## [1.0.0-v1-emulation] — 2026-11-07

### Added
- Reorganized repository into canonical directories (`architecture/`, `mathematical_model/`, `firmware/`, `software/`, `simulation/`, `emulator/`, `configs/`, `tests/`, `Benchmark-Eval/`, `reports/`, `runs/`, `logs/`, `checkpoints/`, `datasets/`, `artifacts/`, `tools/`, `scripts/`, `dependencies/`, `figures/`).
- Implemented 16-round sliding-subwindow fast burst detector (`avid_fast_burst_detector_t` in `software/src/control/causal_scheduler.h` and `software/src/inference/avid_daemon.cpp`) with 0.35 syndrome-density threshold, 4-round debounce, and 20-round worst-case trigger latency.
- Added configurable hardware OSD modes (`cfg_osd_mode = 0, 1, 2` at register `0x058`) separating pure 9-marginal-LLR BP (`17` core cycles), inline pair-aware OSD-0 (`17` core cycles), and post-BP OSD-0 stage (`19` core cycles).

### Fixed
- Reconciled V1 emulation target (Zynq-7020 / ARM Cortex-A9) vs. V2 physical target (ZCU104 / Kria K26 / ARM Cortex-A53) across all documentation and configs.
- Distinguished 9 data-identifiable syndrome signature orbits (`rank(J) = 9`) from 12 gauge null directions (`dim ker(J) = 12`) and 21 prior-regularized MAP parameters (`rank(H_MAP) = 21`).
- Consolidated duplicate mathematical and architecture documentation and removed machine-specific `file://` links.
