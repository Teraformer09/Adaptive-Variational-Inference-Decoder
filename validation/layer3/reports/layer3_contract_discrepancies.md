# Layer 3 Set 1 Contract Discrepancies & Implementation Defects (`88df97e`)

All findings below were reproduced directly against unmodified commit `88df97e86b579b6e2865cc70dc4d0b08225d677b` on branch `layer3-slow-loop-software`.

## 1. Architectural & Contract Discrepancies (`CONTRACT_MISMATCH` / `NOT_IMPLEMENTED`)

| ID | Subsystem | Finding | Evidence Test IDs |
| :--- | :--- | :--- | :--- |
| **DISC-L3-01** | Packet Wire Format | `syndrome_packet_v1.yaml` defines a `16B` magic/version header (`0xAAEC0002`) + `4B` (`32`-bit) per-round word (`meas_flip` at bit 9, `seq_id_low16` at `[31:16]`), whereas `packet_validator.{h,cpp}` implements a headerless `8B` (`64`-bit) `uint64_t` word (`epoch_id` at `[31:16]`, `seq_id` at `[63:32]`). | `L3-IP-002`, `L3-IP-003` |
| **DISC-L3-02** | Packet Timestamp | Neither `syndrome_packet_v1.yaml` nor `avid_syndrome_packet_t` defines a hardware timestamp field, and `epoch_id` monotonicity is not checked by `avid_validate_packet` or `avid_dma_ring_consume_slot`. | `L3-IP-006`, `L3-IP-030` |
| **DISC-L3-03** | Ring Buffer Production Connectivity | `software/src/inference/ring_buffer.h` (`RING_WINDOW=10000`) and `NoiseModel::compute_nll(const ring_buffer_t&)` work standalone (`|nll_rb - nll_hist| < 1e-12`), **but are `IMPLEMENTED_BUT_UNUSED` in `avid_daemon.cpp`**, which accumulates `uint32_t hist_x[16], hist_z[16]` directly from the 4x64 DMA ring. | `L3-INT-003` |
| **DISC-L3-04** | Causal Scheduler Production Connectivity | `avid_causal_scheduler_t` (`commit_delay_rounds`) works standalone, **but `avid_daemon.cpp` never instantiates or polls `avid_causal_scheduler_t`** and exposes no `--commit-delay` configuration option. | `L3-IP-032` |
| **DISC-L3-05** | Normalized Float PMF Input API | Layer 3 estimator classes in `software/src/` accept raw integer count arrays (`const uint32_t hist[16]`) or `ring_buffer_t` only; no floating-point normalized empirical distribution validator exists in `software/src/`. | `L3-IP-016` |
| **DISC-L3-06** | Prior Default & Tether Override | Pair default prior is `0.003` in `NoiseModel()`, `0.004` in `NoiseModel::init()`, and `0.008` in `architecture/estimator/parameter_schema.json`. Furthermore, `HistogramLBFGSSolver::fit_histogram` (`lbfgs_solver.cpp:107`) unconditionally overwrites `model.prior_lambda_ = 0.04`, discarding caller-configured `prior_lambda_` and schema `0.5`. | `L3-IP-025`, `L3-IP-026` |
| **DISC-L3-07** | Native YAML Schema Loader | `configs/schema.yaml` mandates immediate validation failure on unknown top-level/nested keys, but `avid_daemon` has no YAML parser and silently ignores `--config` and unknown CLI flags with `rc=0`. | `L3-IP-036` |

---

## 2. Functional, Numerical, & Safety Input Defects (`FAIL`)

| ID | File & Line | Defect Description | Evidence Test IDs |
| :--- | :--- | :--- | :--- |
| **BUG-L3-01** | `packet_validator.cpp:36-49` | Reserved flag bits `[7:4]` (`0xF0`) are ignored (`flags=0xF1` accepted); `avid_pack_packet_u64` silently masks `s_x=16 -> 0`; rejected out-of-sequence packet mutates `*out` and overwrites `v->last_seq_id`, causing the next valid sequential packet to fail. | `L3-IP-004`, `L3-IP-005`, `L3-IP-007` |
| **BUG-L3-02** | `avid_daemon.cpp:398-408,491` | `--stim-bin` ignores 3-byte truncated `fread != 1` EOF by seeking to `0` (`rc=0`), bypasses `avid_validate_packet`, and re-packs `PARITY_ERR=1, VALID=0` packets as `AVID_PKT_FLAG_VALID`, contaminating `hist_x`/`hist_z` and estimator updates. | `L3-IP-008`, `L3-IP-009`, `L3-INT-001`, `L3-INT-004` |
| **BUG-L3-03** | `dma_ingress_tap.cpp:25,44-47` | `n_pkts > 64` (`e.g., 100`) is silently clamped to `64` without incrementing `overflow_drops`; `hist_x[s]++` / `hist_z[s]++` wrap 32-bit `uint32_t` counters modulo $2^{32}$ without saturation; ring slot state uses unsynchronized `int`/enum fields without atomics or mutexes; stale `epoch_id` packets are accumulated into current window. | `L3-IP-010`, `L3-IP-017`, `L3-IP-019`, `L3-IP-030` |
| **BUG-L3-04** | `noise_model.cpp:74-86,120` | `compute_joint_distribution` and `compute_nll_from_histogram` do not call `validate_parameter_bounds`; `std::clamp(NaN, 1e-6, 0.49)` returns `NaN`, propagating `NaN` across all 16 `p_joint_` states, `grad_`, and `NLL`. Same `NaN` propagation occurs in `fit_histogram`, `prior_theta_`, and `prior_lambda_`. | `L3-IP-024`, `L3-IP-025`, `L3-IP-026`, `L3-IP-028` |
| **BUG-L3-05** | `lbfgs_solver.cpp:102,109-110` | Solver checks `total == 0` only (no `W_min` guard: 3 sparse rounds drive $\theta_4$ from `0.012` to `>0.30` with `converged=true`); `std::max(6, ...)` causes `max_iters = 0, -5, 1` to execute `12` L-BFGS iterations, violating the configured iteration cap. | `L3-IP-029`, `L3-IP-033` |
| **BUG-L3-06** | `causal_scheduler.cpp:21,43` | `avid_fast_detector_init` checks `density_threshold > 0.0` with no `<= 1.0` or `isfinite` upper bound (`1.5` and `+Inf` accepted, disabling burst detection forever); `avid_scheduler_schedule_commit` allows an older `window_end_round` to overwrite a newer pending commit. | `L3-IP-030`, `L3-IP-034` |
| **BUG-L3-07** | `cycle_accurate_bp_engine.cpp:54`, `physical_backend.cpp:95` | Both `CycleAccurateBPEngine::decode` and `avid_hal_set_bp_iterations` accept `K = 99` without upper-bound rejection; physical HAL writes `0x054` while `avid_hal_send_syndrome` hardcodes `K = 3`. | `L3-IP-035` |
| **BUG-L3-08** | `avid_daemon.cpp:233-263,617` | Unknown CLI flags, dangling value flags (`--window` at end of `argv`), and unknown `--controller-mode` values are silently accepted (`rc=0`). Also `avid_daemon.cpp:617` accesses `theta_.data_` directly, which breaks compilation if system `Eigen3` is linked instead of `software/include/Eigen/Dense`. | `L3-IP-036`, `L3-CFG-MISSING` |

---

## 3. Pure C11 ABI & Null-Safety Defects (`c_abi/`)

| ID | Header / Translation Unit | C11 ABI Finding | Evidence Test IDs |
| :--- | :--- | :--- | :--- |
| **ABI-L3-01** | `software/src/hal/avid_hal.h:9-10` | Includes C++-only `<cstdint>` and `<cstddef>` outside `#ifdef __cplusplus` despite declaring `extern "C"` at line 14 — **fails `cc -std=c11 -fsyntax-only` outright**. | `L3-ABI-004` |
| **ABI-L3-02** | `dma_ingress_tap.h`, `causal_scheduler.h`, `fallback_fsm.h`, `stage_profiler.h` | C-compatible struct/function headers **omit `extern "C"`**, so their `.cpp` implementations export C++-mangled symbols (`_Z18avid_dma_ring_initP15avid_dma_ring_t`, etc.) and fail C11 linkage (`dlsym` for unmangled C symbol returns `NULL`). | `L3-ABI-002`, `L3-ABI-004` |
| **ABI-L3-03** | `packet_validator.cpp`, `llr_quantizer.cpp`, `noise_model.cpp`, `fallback_fsm.cpp` | Unguarded `NULL` pointer dereferences (`SIGSEGV` / signal 11) in `avid_validator_init(NULL)`, `avid_unpack_packet_u64(..., NULL)`, `avid_validate_packet(NULL, ...)`, `avid_quantize_and_validate_bank(NULL, ...)`, `compute_nll_from_histogram(nullptr)`, and `avid_fallback_step(..., nullptr, ...)`. | `L3-ABI-006`, `L3-IP-012`, `L3-IP-038` |
| **ABI-L3-04** | `software/src/driver/dma_manager.c:58,93` | Uses POSIX `usleep` and `clock_gettime(CLOCK_MONOTONIC, ...)` without defining `_POSIX_C_SOURCE` or `_DEFAULT_SOURCE`, failing strict ISO `-std=c11` (`CMAKE_C_EXTENSIONS=OFF`) unless POSIX feature macros are injected by the build system. Also `dma_manager.h` omits `extern "C"` for C++ callers. | Build audit |
