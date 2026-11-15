# Layer 3 — Test Set 1 Execution Report (`88df97e` Baseline)

## 1. Execution Provenance
- **Branch**: `layer3-slow-loop-software`
- **Production Source Commit Under Test**: `88df97e86b579b6e2865cc70dc4d0b08225d677b` (**zero edits** to `software/`, `architecture/`, `deployment/`, `configs/`, `checkpoints/`)
- **Frozen Reference Baseline**: `/home/tirtha/AVID_EPJ_Patches` @ `54da68d92778e4e1ffde9514aaaae585638761ff` (**verified untouched**)
- **Compiler & Toolchain**: `cc` / `c++` `(Debian 15.3.0-1) 15.3.0` (`C11` `-std=c11` extensions OFF, `C++17` `-std=c++17` extensions OFF), CMake `3.31.6`, Python `3.14.6`
- **Build Configurations Executed**:
  1. **Standard (`RelWithDebInfo`)**: **66/66** unique translation units compiled (`19` production, `8` C ABI, `39` C++ Set 1 & common), **53/53** CTest targets executed.
  2. **Sanitized (`Debug` + `-fsanitize=address,undefined`)**: **66/66** translation units compiled, **53/53** CTest targets executed (`ctest` exit code `0`; zero sanitizer violations outside intentional `fork()`-isolated null-dereference child probes).

---

## 2. Canonical Test Set 1 Results (`L3-IP-001` .. `L3-IP-040`)

**Summary (`40/40` Executed)**: `15 PASS` | `20 FAIL` | `3 CONTRACT_MISMATCH` | `2 NOT_IMPLEMENTED` | `0 BLOCKED` | `0 INCONCLUSIVE`

| Test ID | Status | Test Name | Production Target | Classification / Evidence Summary |
| :--- | :--- | :--- | :--- | :--- |
| `L3-IP-001` | `PASS` | `valid_packet_ingress` | `packet_validator.cpp` | Single word & 64-packet sequential stream accepted and decoded bit-exact |
| `L3-IP-002` | `CONTRACT_MISMATCH` | `packet_length_and_wire_contract` | `packet_validator.h` vs `syndrome_packet_v1.yaml` | 8B `uint64_t` by-value C API vs YAML 16B header + 4B word; no byte-length check |
| `L3-IP-003` | `CONTRACT_MISMATCH` | `packet_version_and_endianness` | `packet_validator.cpp` | Little-endian bitshifts verified, but YAML `0xAAEC0002` magic/version parser absent |
| `L3-IP-004` | `FAIL` | `packet_field_encoding_and_flags` | `packet_validator.cpp` | `VALID=0` & `PARITY_ERR=1` rejected, but reserved upper flag bits `0xF1` accepted |
| `L3-IP-005` | `FAIL` | `syndrome_bit_range_and_field_decode` | `packet_validator.cpp` | `0..15` round-trip exact, but out-of-range `s_x=16` silently masked (`& 0x0F`) to `0` |
| `L3-IP-006` | `NOT_IMPLEMENTED` | `timestamp_and_epoch_monotonicity` | `packet_validator.h` | Packet timestamp field absent; `epoch_id` regression (`5 -> 2`) accepted |
| `L3-IP-007` | `FAIL` | `packet_sequence_continuity` | `packet_validator.cpp` | Gap/dup/backward rejected, **but** rejected `seq=999` overwrites `last_seq_id`, breaking next valid `seq=12` |
| `L3-IP-008` | `FAIL` | `packet_corruption_and_no_side_effect` | `packet_validator.cpp`, `avid_daemon.cpp` | `*out` mutated on reject; no CRC; `avid_daemon --stim-bin` bypasses `PARITY_ERR` |
| `L3-IP-009` | `FAIL` | `packet_truncation_and_boundaries` | `packet_validator.cpp`, `avid_daemon.cpp` | `UINT32_MAX->0` rollover works, **but** `avid_daemon --stim-bin` accepts 3-byte truncated file (`rc=0`) |
| `L3-IP-010` | `FAIL` | `dma_ring_packet_burst_and_overflow` | `dma_ingress_tap.cpp` | 4-slot FSM works, **but** 100-packet burst silently truncated to 64 with `overflow_drops=0` |
| `L3-IP-011` | `PASS` | `histogram_dimensions_and_accumulation` | `dma_ingress_tap.cpp` | 16-bin $H_X, H_Z$ counts match independent histogram oracle across 5- and 64-packet fixtures |
| `L3-IP-012` | `FAIL` | `histogram_dimensionality_and_pointer_validation` | `noise_model.cpp`, `lbfgs_solver.cpp` | `const uint32_t hist[16]` decays without length check; `nullptr` crashes (`SIGSEGV`) |
| `L3-IP-013` | `PASS` | `empty_zero_count_histogram_handling` | `noise_model.cpp`, `lbfgs_solver.cpp` | `total==0` returns finite `NLL=0`, `grad=0`, `converged=false`, `iters=0`, preserving $\theta$ |
| `L3-IP-014` | `PASS` | `single_state_histogram` | `dma_ingress_tap.cpp`, `noise_model.cpp` | Single-bin `hist[6]=64` leaves remaining 15 bins zero and produces finite NLL/fit |
| `L3-IP-015` | `PASS` | `histogram_sum_invariance_1000_vs_10000` | `noise_model.cpp`, `lbfgs_solver.cpp` | Per-round normalized `NLL(10*H) == NLL(H)` (`< 1e-12`); orbit RMSE `< 0.005` |
| `L3-IP-016` | `NOT_IMPLEMENTED` | `normalized_empirical_distribution_input_check` | `noise_model.h` | Layer 3 `software/src/` accepts `uint32_t` count histograms only; no float PMF validator |
| `L3-IP-017` | `FAIL` | `histogram_counter_overflow` | `dma_ingress_tap.cpp` | `hist_x[s]++` wraps `UINT32_MAX -> 0` without saturation (`noise_model.cpp` uses `double` safely) |
| `L3-IP-018` | `PASS` | `window_rollover_and_histogram_reset` | `dma_ingress_tap.cpp` | Explicit window buffer zeroing prevents cross-window carryover across consecutive epochs |
| `L3-IP-019` | `FAIL` | `concurrent_histogram_snapshot_and_synchronization` | `dma_ingress_tap.h` | Ring indices/states use plain non-atomic `int`/enum with zero synchronization primitives or snapshot API |
| `L3-IP-020` | `PASS` | `histogram_xz_channel_isolation` | `dma_ingress_tap.cpp`, `noise_model.cpp` | $s_X \to H_X$ and $s_Z \to H_Z$ strictly isolated with distinct X/Z stabilizer signatures |
| `L3-IP-021` | `PASS` | `parameter_vector_dimension_21` | `noise_model.h` | `9 + 12 = 21` (`sizeof(ParamVec)==168`) matches `parameter_schema.json` |
| `L3-IP-022` | `PASS` | `parameter_ordering_and_signatures` | `noise_model.cpp` | All 21 indices, 12 pairs, and X/Z signatures match stabilizer incidence oracle; permutation-sensitive |
| `L3-IP-023` | `PASS` | `valid_parameter_bounds_and_clamping` | `noise_model.cpp` | `validate_parameter_bounds` enforces `[0, 0.5)` and `clamp_params` maps to `[1e-5, 0.45]` |
| `L3-IP-024` | `FAIL` | `invalid_parameter_values_nan_inf_oob` | `noise_model.cpp` | `compute_joint_distribution` skips `validate_parameter_bounds`; `std::clamp(NaN)` propagates `NaN` to `p_joint_` & `NLL` |
| `L3-IP-025` | `FAIL` | `prior_vector_validation_and_schema_consistency` | `noise_model.cpp` | Pair default prior drifts (`0.003` ctor vs `0.004` `init()` vs `0.008` schema); `NaN` `prior_theta_` propagates `NaN` |
| `L3-IP-026` | `FAIL` | `regularization_lambda_validation_and_override` | `noise_model.cpp`, `lbfgs_solver.cpp` | Negative/`NaN` `prior_lambda_` unrejected; `fit_histogram` overwrites caller `lambda` to hardcoded `0.04` |
| `L3-IP-027` | `PASS` | `valid_initial_guess_and_optimizer_reset` | `lbfgs_solver.cpp`, `lbfgs_optimizer.cpp` | Interior $\theta^{(0)}$ converges monotonically and `opt_.reset()` guarantees bit-identical repeat runs |
| `L3-IP-028` | `FAIL` | `out_of_range_and_nan_initial_guess` | `lbfgs_solver.cpp` | `-0.10`/`0.85` silently clamped without warning; `NaN` initial guess corrupts `NLL` & $\theta$ with `NaN` |
| `L3-IP-029` | `FAIL` | `sparse_sample_window_w_min_check` | `lbfgs_solver.cpp` | No `W_min` guard: `N=3` rounds (`hist[6]=2`) reports `converged=true` and swings $\theta_4$ from `0.012` to `>0.30` |
| `L3-IP-030` | `FAIL` | `stale_observations_and_epoch_isolation` | `dma_ingress_tap.cpp`, `causal_scheduler.cpp` | Stale `epoch_id` packets accumulated; older `window_end_round` overwrites newer pending commit |
| `L3-IP-031` | `PASS` | `window_size_configuration_consumption` | `avid_daemon.cpp` | `--window 0/-16` rejected (`rc=2`); `--window 256 -> 4` epochs vs `--window 512 -> 2` epochs |
| `L3-IP-032` | `CONTRACT_MISMATCH` | `commit_delay_configuration_and_daemon_wiring` | `causal_scheduler.cpp`, `avid_daemon.cpp` | `avid_causal_scheduler_t` works standalone, **but** is disconnected from `avid_daemon.cpp` |
| `L3-IP-033` | `FAIL` | `optimizer_max_iters_cap_validation` | `lbfgs_solver.cpp` | `std::max(6, ...)` causes `max_iters = -5, 0, 1` to execute `12` iterations |
| `L3-IP-034` | `FAIL` | `burst_threshold_and_debounce_bounds` | `causal_scheduler.cpp` | `threshold = 1.5` and `+Inf` accepted, permanently disabling fast burst detection |
| `L3-IP-035` | `FAIL` | `bp_iteration_k_bounds_validation` | `cycle_accurate_bp_engine.cpp`, `physical_backend.cpp` | `K = 99` accepted without upper-bound check in both `CycleAccurateBPEngine` and HAL |
| `L3-IP-036` | `FAIL` | `unknown_config_fields_and_schema_enforcement` | `avid_daemon.cpp`, `configs/schema.yaml` | Unknown CLI flags (`--unknown-key`), `--config`, and unknown `controller_mode` silently accepted (`rc=0`) |
| `L3-IP-037` | `PASS` | `runtime_telemetry_and_epoch_reconciliation` | `avid_daemon.cpp` | `total_rounds=256`, `epochs_completed=4`, and `epoch_commit_reconciled=true` verified |
| `L3-IP-038` | `FAIL` | `safety_fsm_state_and_event_inputs` | `fallback_fsm.cpp` | 16/16 boolean tuples deterministic, **but** `candidate_q78 == nullptr` crashes (`SIGSEGV`) |
| `L3-IP-039` | `PASS` | `active_decoder_weight_bank_readback` | `weight_reg_driver.c`, `weight_bank_driver.c` | Shadow isolated prior to commit; exact 9-weight active readback after commit; OOB index returns `0` |
| `L3-IP-040` | `PASS` | `corrupted_candidate_weight_bank_rejection` | `llr_quantizer.cpp`, `fallback_fsm.cpp` | Corrupted probabilities/weights rejected (`bounds_ok=false`), reverting to `SAFE_STATIC_PRIOR` (`1128`) |

---

## 3. Pure C11 ABI (`c_abi/`) & Production Integration (`integration/`) Results

| Test ID | Suite | Status | Key Evidence |
| :--- | :--- | :--- | :--- |
| `L3-ABI-001` | C11 ABI | `PASS` | `packet_validator.h` C11/C++17 layout match (`12B` struct, `8B` wire word) + clean `extern "C"` call |
| `L3-ABI-002` | C11 ABI | `FAIL` | Struct layouts match, **but** `dma_ingress_tap.h` lacks `extern "C"` (`dlsym("avid_dma_ring_init") == NULL`) |
| `L3-ABI-003` | C11 ABI | `PASS` | `llr_quantizer.h` C11 calls & bounds rejection pass (`0x0469`); documents C++-only 21-param `NoiseModel` |
| `L3-ABI-004` | C11 ABI | `FAIL` | `causal_scheduler.h` & `fallback_fsm.h` lack `extern "C"`; `hal/avid_hal.h` fails `cc -std=c11` (`#include <cstdint>`) |
| `L3-ABI-005` | C11 ABI | `PASS` | `77/77` `sizeof`, `_Alignof`, and `offsetof` checks match bit-for-bit between C11 and C++17 across all 11 shared structs |
| `L3-ABI-006` | C11 ABI | `FAIL` | `6/6` `packet_validator` & `llr_quantizer` C entrypoints crash with signal 11 on `NULL` (`weight_bank_driver` safely returns `-1`) |
| `L3-ABI-007` | C11 ABI | `PASS` | Caller-owned heap/stack buffers, 64-byte guard canaries, `ring_buffer_t` wraparound, and `wrd` lifecycle verified |
| `L3-DMA-INVALID-PKT` | DMA | `PASS` | Standalone `avid_dma_ring_consume_slot` excludes `VALID=0` and `PARITY_ERR=1` packets from histograms |
| `L3-CFG-MISSING` | Config | `FAIL` | Missing checkpoint/stim files return `rc=2`, **but** dangling `--window` flag without value exits `rc=0` |
| `L3-INT-001` | Integration | `FAIL` | `avid_daemon --stim-bin` accepts 3-byte truncated file (`rc=0`) and unpacks without `avid_validate_packet` |
| `L3-INT-002` | Integration | `PASS` | `avid_dma_ring_consume_slot -> hist_x/hist_z -> fit_histogram` end-to-end daemon dataflow verified |
| `L3-INT-003` | Integration | `CONTRACT_MISMATCH` | `ring_buffer.h` works standalone (`|nll_rb - nll_hist| < 1e-12`) **but** is `IMPLEMENTED_BUT_UNUSED` in `avid_daemon` |
| `L3-INT-004` | Integration | `FAIL` | `avid_daemon --stim-bin` re-stamps `AVID_PKT_FLAG_VALID` onto 64 `PARITY_ERR=1` packets, contaminating estimator NLL |

---

## 4. Gate `G1` Decision & Mandatory Stage Stop
- **Gate `G1` (`Input Integrity & Contract Conformance`)**: **`FAIL`** (`20` PASS / `31` non-PASS across the 51 gated `L3-IP`, `L3-ABI`, and `L3-INT` items).
- **Gate `G10` (`Reproducibility, Traceability, & Branch Isolation`)**: **`PASS`**.
- **Stage Gate Action**: Per instructions, execution stops cleanly after Set 1 so contract discrepancies and remediation scope can be reviewed before implementing executable tests for Sets 2–5.
