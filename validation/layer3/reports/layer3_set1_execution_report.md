# Layer 3 — Test Set 1 Execution Report (`88df97e` Baseline)

## 1. Execution Provenance & Cross-Branch Scope Note
- **Layer 3 Stage Status**: **Test Set 1 (`set1_input_contracts`, `c_abi`, `integration`) is implemented and executed** against unmodified production baseline `88df97e86b579b6e2865cc70dc4d0b08225d677b` on branch `layer3-slow-loop-software` (initial execution commit `be50761cc6f4cc0c31f79555aa3f69ccd911563d`). Sets 2–5 remain halted at the mandatory Set 1 review gate.
- **Layer 4 Stage Status Clarification**: On separate branch `layer4-cosim-bridge-emu` (`dd5ec1f2dcb531764a6d3f5940f184f426e93f10`), **only pre-implementation traceability (`LAYER4_IMPLEMENTATION_TO_TEST_TRACEABILITY.md`) has been completed and pushed** — Layer 4 test implementation and execution have **not** started.
- **Frozen Reference Baseline**: `/home/tirtha/AVID_EPJ_Patches` @ `54da68d92778e4e1ffde9514aaaae585638761ff` (**verified untouched**).
- **Compiler & Toolchain**: `cc` / `c++` `(Debian 15.3.0-1) 15.3.0` (`C11` `-std=c11` extensions OFF, `C++17` `-std=c++17` extensions OFF), CMake `3.31.6`, Python `3.14.6`.
- **Exact Scope of the `66` Translation Units (`132` Total Compilation Invocations Across Both Builds)**:
  - Each build configuration compiles the exact same **`66` unique source files**:
    - **`19` production source files**: `18` non-`main` `.c`/`.cpp` files compiled into `libavid_layer3_prod.a` + `1` daemon entrypoint (`software/src/inference/avid_daemon.cpp`) compiled into `avid_daemon_under_test`.
    - **`8` C ABI source files**: `1` C++17 layout oracle (`c_abi_cxx_layout_oracle.cpp`) + `7` strict C11 (`-std=c11`) `.c` test programs.
    - **`39` C++17 harness & Set 1 source files**: `6` `tests/layer3/common/` files + `33` `tests/layer3/set1_input_contracts/{packet,dma_histogram,estimator_inputs,runtime_config,integration}/*.cpp` files.
  - Both configurations were built from scratch and executed in full (`66` units in `build/layer3_standard` [`RelWithDebInfo`] + `66` units in `build/layer3_sanitizers` [`Debug` + `-fsanitize=address,undefined`] = **`132` total compilation units built** across the validation run; **53/53** CTest targets executed in each build).

---

## 2. Canonical Test Set 1 Results (`L3-IP-001` .. `L3-IP-040`)

**Summary (`40/40` Canonical IDs Executed)**: `15 PASS` | `20 FAIL` | `3 CONTRACT_MISMATCH` | `2 NOT_IMPLEMENTED` | `0 BLOCKED` | `0 INCONCLUSIVE`

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
| `L3-IP-027` | `PASS` | `valid_initial_guess_and_optimizer_reset` | `lbfgs_solver.cpp`, `lbfgs_optimizer.cpp` | Interior $\theta^{(0)}$ decreases NLL deterministically *(note `DISC-L3-08`: solver runs 2-pass projected GD rather than `opt_.step()`)* |
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

## 3. Non-243 Harness Extensions: Provenance, Sub-Check Disambiguation, & Gate `G1` Arithmetic

The **243 canonical traceability IDs** (`test_ids.yaml`) remain strictly unchanged. Set 1 adds **13 explicitly specified harness targets** registered in `tests/layer3/common/supplementary_set1_specs.yaml`:
- **`7` C11 ABI Targets (`L3-ABI-001..007`)** — **Counted in `G1`** (`4 PASS`, `3 FAIL`).
  - *Explicit Sub-Check Decomposition of `L3-ABI-004` (`test_runtime_config_abi.c`, 7 checks total, 4 failed)*:
    - `L3-ABI-004.a` (`PASS`, 3/3 checks): `weight_reg_driver.h` & `weight_bank_driver.h` pure C11 compilation and calls (`wrd_open`, `avid_weight_bank_write_and_commit`, `wrd_read_active_weight`).
    - `L3-ABI-004.b` (`FAIL`, 0/3 checks -> `ABI-L3-02`): `dlsym` C-symbol checks for `avid_scheduler_init`, `avid_fast_detector_init` (`causal_scheduler.h`), and `avid_fallback_init` (`fallback_fsm.h`) — Category II headers lacking `extern "C"`.
    - `L3-ABI-004.c` (`FAIL`, 0/1 check -> `ABI-L3-01`): `cc -std=c11 -x c -fsyntax-only` check on `software/src/hal/avid_hal.h` — Category I header declaring `extern "C"` at line 14 while including `<cstdint>`/`<cstddef>` at lines 9–10.
- **`4` Daemon Integration Targets (`L3-INT-001..004`)** — **Counted in `G1`** (`1 PASS`, `2 FAIL`, `1 CONTRACT_MISMATCH`).
- **`2` Diagnostic Companions (`L3-DMA-INVALID-PKT`, `L3-CFG-MISSING`)** — **Reported separately (excluded from `G1` 51-item denominator)** so they do not double-count canonical tests `L3-IP-008/010` and `L3-IP-031/036`.

| Test ID | Category | Counted in `G1`? | Parent Canonical IDs | Source File | Status | Summary |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| `L3-ABI-001` | C11 ABI | Yes | `L3-IP-001, 002` | `c_abi/test_packet_abi.c` | `PASS` | `packet_validator.h` C11/C++17 layout match (`12B` struct, `8B` wire word) + `extern "C"` call |
| `L3-ABI-002` | C11 ABI | Yes | `L3-IP-010, 011` | `c_abi/test_histogram_abi.c` | `FAIL` | Struct layouts match; Category II `dma_ingress_tap.h` omits `extern "C"` (`dlsym == NULL`) |
| `L3-ABI-003` | C11 ABI | Yes | `L3-IP-021, 023, 040` | `c_abi/test_parameter_abi.c` | `PASS` | `llr_quantizer.h` C11 calls pass (`0x0469`); Category III `NoiseModel` exempt (C++-only) |
| `L3-ABI-004` | C11 ABI | Yes | `L3-IP-032, 038, 039` | `c_abi/test_runtime_config_abi.c` | `FAIL` | Composite check: `004.a` drivers `PASS`; `004.b` scheduler/FSM `dlsym` `FAIL`; `004.c` `avid_hal.h` C11 `FAIL` |
| `L3-ABI-005` | C11 ABI | Yes | `L3-IP-001..039` | `c_abi/test_struct_layout.c` | `PASS` | `77/77` `sizeof`, `_Alignof`, and `offsetof` checks match between C11 and C++17 across 11 structs |
| `L3-ABI-006` | C11 ABI | Yes | `L3-IP-008, 040` | `c_abi/test_null_pointer.c` | `FAIL` | `6/6` Category I `packet_validator` & `llr_quantizer` C entrypoints crash (`SIGSEGV`) on `NULL` |
| `L3-ABI-007` | C11 ABI | Yes | `L3-IP-011, 039` | `c_abi/test_buffer_ownership.c` | `PASS` | Caller-owned buffers, 64B guard canaries, `ring_buffer_t` wraparound, and `wrd` lifecycle verified |
| `L3-INT-001` | Integration | Yes | `L3-IP-002, 009` | `integration/test_production_ingress_path.cpp` | `FAIL` | `avid_daemon --stim-bin` accepts 3-byte truncated file (`rc=0`) and skips `avid_validate_packet` |
| `L3-INT-002` | Integration | Yes | `L3-IP-011, 020` | `integration/test_daemon_histogram_path.cpp` | `PASS` | `avid_dma_ring_consume_slot -> hist_x/hist_z -> fit_histogram` daemon dataflow verified |
| `L3-INT-003` | Integration | Yes | `L3-IP-011, 018` | `integration/test_ring_buffer_connectivity.cpp` | `CONTRACT_MISMATCH` | `ring_buffer.h` works standalone (`|nll_rb - nll_hist| < 1e-12`) **but** is `IMPLEMENTED_BUT_UNUSED` in `avid_daemon` |
| `L3-INT-004` | Integration | Yes | `L3-IP-004, 008` | `integration/test_invalid_input_propagation.cpp` | `FAIL` | `avid_daemon --stim-bin` re-stamps `AVID_PKT_FLAG_VALID` onto 64 `PARITY_ERR=1` packets, contaminating estimator |
| `L3-DMA-INVALID-PKT` | Companion | No | `L3-IP-008, 010` | `dma_histogram/test_histogram_invalid_packet.cpp` | `PASS` | Standalone `avid_dma_ring_consume_slot` excludes `VALID=0` and `PARITY_ERR=1` packets from histograms |
| `L3-CFG-MISSING` | Companion | No | `L3-IP-031, 036` | `runtime_config/test_config_missing.cpp` | `FAIL` | Missing `--checkpoint`/`--stim-bin` return `rc=2`, **but** dangling `--window` flag without value exits `rc=0` |

---

## 4. Gate `G1` Decision & Mandatory Stage Stop
- **Gate `G1` (`Input Integrity & Contract Conformance`)**: **`FAIL`** (`40 L3-IP + 7 L3-ABI + 4 L3-INT = 51` gated items: **`20` PASS (`15` `L3-IP` + `4` `L3-ABI` + `1` `L3-INT`)**, **`31` non-PASS (`25` `L3-IP` + `3` `L3-ABI` + `3` `L3-INT`)**).
- **Gate `G10` (`Reproducibility, Traceability, & Branch Isolation`)**: **`PASS`**.
- **Stage Gate Action**: Do not proceed to Set 2. Preserve snapshot `validation/layer3/runs/88df97e_set1_baseline/` and commit `be50761cc6f4cc0c31f79555aa3f69ccd911563d` intact; resolve the authoritative packet, C ABI, and slow-loop integration contracts first, then execute a supplemental post-remediation Set 1 run.
