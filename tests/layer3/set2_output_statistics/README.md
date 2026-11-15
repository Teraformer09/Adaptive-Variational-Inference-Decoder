# Layer 3 Test Set 2 — Output & Statistical Validation (`L3-OP-001..036`, `L3-STAT-001..015`)

**Authorization Status:** Deferred per Layer 3 Execution Directive (`Stop after Set 1`).  
**Canonical Registry:** `tests/layer3/common/test_ids.yaml` (`36` `L3-OP` + `15` `L3-STAT` = `51` IDs).

## Planned Executable Inventory (Post-Set-1 & Post-Set-5 Review)

- `estimator_output/`: `test_parameter_output_shape.cpp`, `test_parameter_output_bounds.cpp`, `test_parameter_output_finiteness.cpp`, `test_objective_improvement.cpp`, `test_optimizer_status.cpp` (`L3-OP-001..010`).
- `llr_quantization/`: `test_llr_conversion.cpp`, `test_q78_output.cpp`, `test_q78_saturation.cpp`, `test_q78_rounding.cpp`, `test_decoder_weight_mapping.cpp` (`L3-OP-011..020`).
- `scheduler_output/`: `test_scheduler_decision.cpp`, `test_burst_trigger_output.cpp`, `test_commit_request.cpp`, `test_epoch_output.cpp` (`L3-OP-021..028`).
- `fsm_output/`: `test_fsm_action_output.cpp`, `test_active_configuration.cpp`, `test_fallback_output.cpp` (`L3-OP-029..036`).
- `statistical/` (`>= 30` independent seeds per experiment, reporting Bias, Variance, RMSE, 95% CI, quotient-orbit recovery, and aggregate failure-ratio LER): `test_stationary_recovery.py`, `test_drift_recovery.py`, `test_correlated_noise.py`, `test_burst_contamination.py`, `test_sample_size_sensitivity.py`, `test_seed_variance.py`, `test_prior_sensitivity.py`, `test_initialization_sensitivity.py`, `test_model_mismatch.py`, `test_false_trigger_rate.py`, `test_missed_trigger_rate.py`, `test_confidence_calibration.py`, `test_identifiable_parameter_recovery.py`, `test_logical_error_rate.py`, `test_reproducibility.py` (`L3-STAT-001..015`).
