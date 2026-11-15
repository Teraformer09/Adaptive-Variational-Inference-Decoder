# Layer 3 Test Set 3 — Pipeline & System-Flow Validation (`L3-FLOW-001..045`)

**Authorization Status:** Deferred per Layer 3 Execution Directive (`Stop after Set 1`).  
**Canonical Registry:** `tests/layer3/common/test_ids.yaml` (`45` `L3-FLOW` IDs; priority group `L3-FLOW-001..006`).

## Planned Executable Inventory

- `instrumentation/`: `call_graph_recorder.{h,cpp}`, `data_flow_recorder.{h,cpp}`, `trace_schema.json` (capturing producer/consumer invocation counts, input/output hashes, `epoch_id`, `round_index`, monotonic timestamps).
- `ingress_to_histogram/`: `test_packet_to_histogram.cpp`, `test_invalid_packet_exclusion.cpp`, `test_histogram_propagation.cpp` (`L3-FLOW-001..002`).
- `histogram_to_estimator/`: `test_histogram_snapshot_delivery.cpp`, `test_estimator_input_identity.cpp`, `test_ring_buffer_path.cpp` (`L3-FLOW-003`).
- `optimizer_execution/`: `test_solver_call_path.cpp`, `test_optimizer_type.cpp`, `test_optimizer_result_propagation.cpp`, `test_optimizer_diagnostics.cpp` (`L3-FLOW-004..005`).
- `estimator_to_llr/`: `test_parameter_to_llr.cpp`, `test_llr_to_weight_mapping.cpp`, `test_invalid_estimate_blocking.cpp` (`L3-FLOW-006`).
- `scheduler_integration/`: `test_burst_detector_invocation.cpp`, `test_causal_scheduler_invocation.cpp`, `test_fast_slow_loop_separation.cpp`, `test_commit_scheduling.cpp` (`L3-FLOW-007..008`, `L3-FLOW-026..035`).
- `fsm_to_hal/`: `test_fsm_to_hal_action.cpp`, `test_hal_configuration.cpp`, `test_active_epoch_propagation.cpp` (`L3-FLOW-009..012`).
- `end_to_end/`: `test_stationary_flow.cpp`, `test_drift_flow.cpp`, `test_correlated_noise_flow.cpp`, `test_burst_flow.cpp`, `test_delayed_commit_flow.cpp`, `test_reset_recovery_flow.cpp`, `test_fallback_flow.cpp` (`L3-FLOW-013..025`, `L3-FLOW-036..045`).
