# Layer 3 Test Set 4 — Execution, Persistence & Fault-Recovery Validation (`L3-EXEC-001..045`)

**Authorization Status:** Deferred per Layer 3 Execution Directive (`Stop after Set 1`).  
**Canonical Registry:** `tests/layer3/common/test_ids.yaml` (`45` `L3-EXEC` IDs).

## Planned Executable Inventory

- `startup_shutdown/`: `test_daemon_startup.cpp`, `test_daemon_shutdown.cpp`, `test_invalid_startup_config.cpp`, `test_repeated_start_stop.cpp`, `test_signal_handling.cpp`.
- `runtime_stress/`: `test_long_running_daemon.cpp`, `test_ingress_throughput.cpp`, `test_histogram_load.cpp`, `test_optimizer_load.cpp`, `test_memory_usage.cpp`.
- `persistence/` (must explicitly classify unimplemented disk checkpoint warm-resume as `NOT_IMPLEMENTED`): `test_histogram_persistence.cpp`, `test_estimator_state_persistence.cpp`, `test_optimizer_state_persistence.cpp`, `test_epoch_persistence.cpp`, `test_fsm_persistence.cpp`, `test_warm_resume.cpp`.
- `fault_injection/`: `test_malformed_packet_recovery.cpp`, `test_nan_estimator_recovery.cpp`, `test_optimizer_failure_recovery.cpp`, `test_commit_timeout_recovery.cpp`, `test_epoch_mismatch_recovery.cpp`, `test_decoder_busy_recovery.cpp`, `test_process_crash_recovery.cpp`.
- `concurrency/`: `test_concurrent_ingress.cpp`, `test_concurrent_estimation.cpp`, `test_concurrent_snapshot.cpp`, `test_concurrent_commit.cpp`, `test_race_conditions.cpp`.
