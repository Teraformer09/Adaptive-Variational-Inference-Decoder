# Layer 3 — Test Set 1 Remediation & Supplemental Gate `G1` Re-Evaluation Report (`remediated_set1_v1`)

## 1. Executive Summary & Cross-Branch Scope Statement

- **Layer 3 Status (`layer3-slow-loop-software`)**:
  - **Test Set 1 (`set1_input_contracts`, `c_abi`, `integration`) has been remediated (`698771d2148cc5436f6b1a83eb98b4592ade0deb`) and re-executed in isolated run directory `validation/layer3/runs/remediated_set1_v1/`.**
  - **Stage 1 Gate Outcome**: **`G1 = PASS`** (`51/51` gated items `PASS` + `2/2` diagnostic companions `PASS`; `ASan+UBSan CTest rc = 0`) and **`G10 = PASS`** (`stage1_gate_decision = STAGE1_ACCEPTED_G1_G10_PASS`).
  - **Sets 2–5 Status**: **Not yet started (`G2..G9 = NOT_EVALUATED_YET`, `overall_layer3_decision = PENDING_SETS_2_TO_5`)** — execution halts here per the mandatory Stage 1 review gate.
- **Layer 4 Status (`layer4-cosim-bridge-emu` @ `dd5ec1f2dcb531764a6d3f5940f184f426e93f10`)**:
  - **Only pre-implementation traceability (`LAYER4_IMPLEMENTATION_TO_TEST_TRACEABILITY.md`) is complete and pushed.** Layer 4 test implementation and execution have **not** started.
- **Frozen Reference Baseline (`AVID_EPJ_Patches`)**:
  - Remains untouched and clean at `54da68d92778e4e1ffde9514aaaae585638761ff`.
- **Immutable Preservation of Initial `88df97e` / `be50761` Failing Baseline Evidence**:
  - Top-level `validation/layer3/{raw_results/set1,logs/set1,results/set1_input_contracts.json,manifests,acceptance}` and `validation/layer3/runs/88df97e_set1_baseline/` (`G1 = FAIL`, `20/51 PASS`, `results_sha256 = 7c08a0fd1f6cfbab97041298d815806b4493cb61580cf68fee5b4a813ee43741`) were **not overwritten** (`git diff be50761` across those directories is empty).

---

## 2. Commit Provenance & Build Verification

| Milestone | Commit (`layer3-slow-loop-software`) | Description |
| :--- | :--- | :--- |
| Unmodified Layer 3 Baseline | `88df97e86b579b6e2865cc70dc4d0b08225d677b` | Original Layer 3 slow-loop codebase + 243-ID frozen traceability matrix |
| Initial Set 1 Failing Audit | `be50761cc6f4cc0c31f79555aa3f69ccd911563d` | Set 1 execution against unmodified `88df97e` (`G1 = FAIL`: `20 PASS / 31 non-PASS`, `G10 = PASS`) |
| Audit Trail Clarifications | `1c8f729098b1c84a92f67b4392b176b645475233` | Explicit Layer 3 vs Layer 4 scope disambiguation, `66`-unit / `132`-invocation build breakdown, `L3-ABI-004` sub-checks |
| Set 1 Code & Contract Remediation | `698771d2148cc5436f6b1a83eb98b4592ade0deb` | Implementation of decisions `A1`, `A2`, `A3` across `architecture/contracts/`, `software/`, and `tests/layer3/` |

- **Dual-Build Compilation Verification (`remediated_set1_v1`)**:
  - **66 unique translation units** (`19` production + `8` C11 ABI + `39` C++17 Set 1/common) compiled cleanly from scratch in **both** `RelWithDebInfo` (`build/layer3_standard`) and `Debug` `-fsanitize=address,undefined` (`build/layer3_sanitizers`) = **132 total compilation invocations**, **0 errors**.
  - Standalone production suite (`software/CMakeLists.txt`: `test_baseline_suite` + `avid_daemon --self-test`) also rebuilt and passed `5/5` baseline checks (`rc = 0`).

---

## 3. Architectural Decisions Enforced (`A1`, `A2`, `A3`)

1. **`A1` — Authoritative 64-Bit (`uint64_t`) `A05/Q11` Syndrome Packet Contract (`packet_contract_decision.md`)**:
   - Reconciled `architecture/contracts/syndrome_packet_v1.yaml` (`v1.1.0`) and `fault_policy.yaml` (`v1.1.0`) with the 8-byte little-endian `s_x[3:0] | s_z[7:4] | flags[15:8] | epoch_id[31:16] | seq_id[63:32]` wire contract shared with Layer 1 (`simulation/stimulus/packet_codec.py`), while explicitly documenting the 16-bit fast-path AXI-Stream slice (`axis_fpga_word_d3`).
   - Added `avid_pack_packet_u64_checked`, `avid_validate_packet_bytes` (strict 8-byte length), and `avid_validate_stream_framing` (`len > 0 && len % 8 == 0`), reserved flag rejection (`(flags & ~0x0Fu) == 0`), `epoch_id` non-regression, and side-effect-free rejection (preserving `*out`, `last_seq_id`, and `last_epoch_id` when a packet fails validation).
2. **`A2` — Full C11 Header & `extern "C"` ABI Parity + `NULL` Safety**:
   - Fixed `software/src/hal/avid_hal.h` to use `<stdint.h>`/`<stddef.h>`/`<stdbool.h>` in C11 mode so `cc -std=c11 -x c -fsyntax-only` and direct C11 `#include "hal/avid_hal.h"` succeed cleanly.
   - Added `#ifdef __cplusplus extern "C" { #endif` guards and C11-compatible headers (`<stdint.h>`, `<stdbool.h>`, `<stddef.h>`) to all Category II boundary headers (`dma_ingress_tap.h`, `causal_scheduler.h`, `fallback_fsm.h`, `stage_profiler.h`, `ring_buffer.h`, `dma_manager.h`), preserving `77/77` `sizeof`/`_Alignof`/`offsetof` checks between C11 and C++17.
   - Added explicit `NULL`-pointer guards across all public C/C++ entrypoints (`packet_validator.cpp`, `dma_ingress_tap.cpp`, `ring_buffer.h`, `llr_quantizer.cpp`, `causal_scheduler.cpp`, `fallback_fsm.cpp`, `weight_reg_driver.c`, `weight_bank_driver.c`, `dma_manager.c`) and UBSan-clean `memcpy` enum-state sanitization in `fallback_fsm.cpp`.
3. **`A3` — Slow-Loop Input Hardening, Daemon Integration, & Genuine Two-Loop L-BFGS**:
   - **DMA Ring & Histograms (`dma_ingress_tap.{h,cpp}`)**: Burst overflow accounting (`overflow_drops += count - cap` + `AVID_PKT_FLAG_OVERFLOW`), saturating `UINT32_MAX` bin accumulation, stale-epoch packet filtering (`min_epoch_id`), and sequence-lock atomic `avid_dma_histogram_snapshot_t` (`avid_dma_snapshot_write` / `avid_dma_snapshot_read`).
   - **Noise Model & Two-Loop L-BFGS (`noise_model.{h,cpp}`, `lbfgs_solver.{h,cpp}`, `lbfgs_optimizer.{h,cpp}`, `Eigen/Dense`)**: Unified pair prior default (`0.008` matching `parameter_schema.json`), `NaN`/`Inf`/out-of-bounds rejection without `NaN` propagation, length-checked `compute_nll_from_histogram_checked` and simplex PMF validator `compute_nll_from_pmf`, caller `lambda` preservation (with `reset_lambda_to_default()`), `W_min = 64` sample-count guard, strict `max_iters` cap (`<= 0` rejected, `max_iters = 1 -> <= 1` step), and genuine two-loop `LBFGSOptimizer` curvature (`(s_k, y_k)`) step integration with projected Armijo backtracking in `HistogramLBFGSSolver::fit_histogram` (resolving `DISC-L3-08`).
   - **Scheduler, BP/HAL Bounds, & `avid_daemon` Integration (`causal_scheduler.{h,cpp}`, `cycle_accurate_bp_engine.{h,cpp}`, `simulation_backend.cpp`, `avid_daemon.cpp`)**: Monotonic `window_end_round` enforcement in `avid_scheduler_schedule_commit`, finite `(0, 1]` burst threshold check (`avid_fast_detector_init_checked`), strict $K \in \{1..5\}$ bounds in both `CycleAccurateBPEngine` and `avid_hal_set_bp_iterations`, schema-enforced `--config` / CLI flag parser in `avid_daemon.cpp`, raw `--stim-bin` framing & flag preservation into `avid_dma_ring_push_burst`, `ring_buffer_t` (`ring_push`) wiring, and `avid_causal_scheduler_t` (`--commit-delay`) causal commit scheduling in `avid_daemon.cpp`.

---

## 4. Before (`88df97e_set1_baseline`) vs After (`remediated_set1_v1`) Resolution Matrix

### 4.1 Summary Counts

| Category | Counted in `G1`? | `88df97e_set1_baseline` (`be50761`) | `remediated_set1_v1` (`698771d`) |
| :--- | :---: | :--- | :--- |
| Canonical Set 1 (`L3-IP-001..040`) | Yes (`40`) | `15 PASS`, `20 FAIL`, `3 CONTRACT_MISMATCH`, `2 NOT_IMPLEMENTED` | **`40 / 40 PASS`** |
| C11 ABI (`L3-ABI-001..007`) | Yes (`7`) | `4 PASS`, `3 FAIL` | **`7 / 7 PASS`** |
| Daemon Integration (`L3-INT-001..004`) | Yes (`4`) | `1 PASS`, `2 FAIL`, `1 CONTRACT_MISMATCH` | **`4 / 4 PASS`** |
| **Total Gated `G1` Items** | **Yes (`51`)** | **`20 PASS` / `31 non-PASS` (`G1 = FAIL`)** | **`51 / 51 PASS` (`G1 = PASS`)** |
| Diagnostic Companions (`L3-DMA-INVALID-PKT`, `L3-CFG-MISSING`) | Separate (`2`) | `1 PASS`, `1 FAIL` | **`2 / 2 PASS`** |
| Sanitizer Suite (`ASan + UBSan`) | Required (`rc=0`) | Executed (baseline crashes isolated in `fork()` probes) | **`53 / 53 PASS` (`rc = 0`)** |

### 4.2 Itemized Resolution of All 31 Previously Non-PASS `G1` IDs (+ Companion `L3-CFG-MISSING`)

| Test ID | Baseline (`88df97e`) | Remediated (`698771d`) | Root-Cause Fix Verified |
| :--- | :--- | :--- | :--- |
| `L3-IP-002` | `CONTRACT_MISMATCH` | `PASS` | Reconciled `syndrome_packet_v1.yaml` (`8B` LE `A05/Q11`) + added `avid_validate_packet_bytes` & `avid_validate_stream_framing` |
| `L3-IP-003` | `CONTRACT_MISMATCH` | `PASS` | Verified explicit little-endian byte decoding (`avid_unpack_packet_bytes`) independent of host endianness |
| `L3-IP-004` | `FAIL` | `PASS` | `avid_validate_packet` rejects non-zero reserved flag bits `(flags & ~0x0Fu) != 0` (`reserved_flag_count`) |
| `L3-IP-005` | `FAIL` | `PASS` | `avid_pack_packet_u64_checked` rejects `s_x > 15`, `s_z > 15`, and invalid flags without silent masking |
| `L3-IP-006` | `NOT_IMPLEMENTED` | `PASS` | Enforced monotonic `seq_id` (1 MHz round index) and non-decreasing `epoch_id >= last_epoch_id` (`epoch_regress_count`) |
| `L3-IP-007` | `FAIL` | `PASS` | Rejected gap/duplicate/out-of-order packets no longer overwrite `last_seq_id` |
| `L3-IP-008` | `FAIL` | `PASS` | Rejected packets leave `*out` untouched; `avid_daemon --stim-bin` preserves and rejects `PARITY_ERR` packets |
| `L3-IP-009` | `FAIL` | `PASS` | `avid_daemon --stim-bin` rejects empty (`0B`) and non-8B-multiple (`3B`, `7B`, `9B`) files with `rc = 2` |
| `L3-IP-010` | `FAIL` | `PASS` | `avid_dma_ring_push_burst` counts excess packets beyond 64-entry slot capacity in `overflow_drops` |
| `L3-IP-012` | `FAIL` | `PASS` | Null `hist` pointers guarded across `NoiseModel`/`HistogramLBFGSSolver`; `compute_nll_from_histogram_checked` enforces `n_bins == 16` |
| `L3-IP-016` | `NOT_IMPLEMENTED` | `PASS` | Added `NoiseModel::validate_empirical_pmf` and `NoiseModel::compute_nll_from_pmf` (`+inf` on invalid simplex inputs) |
| `L3-IP-017` | `FAIL` | `PASS` | `avid_dma_ring_consume_slot` saturates `hist_x[s]` / `hist_z[s]` at `UINT32_MAX` instead of wrapping to `0` |
| `L3-IP-019` | `FAIL` | `PASS` | Added lock-free sequence-lock `avid_dma_histogram_snapshot_t` (`avid_dma_snapshot_write` / `avid_dma_snapshot_read`) |
| `L3-IP-024` | `FAIL` | `PASS` | `NoiseModel` validates finiteness & `[0, 0.5)` bounds before DP; `NaN`/`Inf`/OOB yield `+inf` NLL without corrupting `p_joint_` |
| `L3-IP-025` | `FAIL` | `PASS` | Aligned pair prior default to `0.008` (`parameter_schema.json`) and added `set_prior_checked` rejecting `NaN`/OOB priors |
| `L3-IP-026` | `FAIL` | `PASS` | Added `set_prior_lambda_checked` (`lambda >= 0`, finite); `fit_histogram` preserves caller `lambda` instead of overwriting to `0.04` |
| `L3-IP-028` | `FAIL` | `PASS` | `fit_histogram` rejects `NaN`/`Inf` initial guesses cleanly and flags clamped OOB initial guesses via `initial_guess_clamped` |
| `L3-IP-029` | `FAIL` | `PASS` | Enforced `min_window_samples` (`W_min = 64`): sparse windows (`N = 3 < 64`) set `insufficient_samples = true`, preserving prior $\theta$ |
| `L3-IP-030` | `FAIL` | `PASS` | `avid_dma_ring_set_min_epoch` isolates stale-epoch packets; `avid_scheduler_schedule_commit` drops regressive `window_end_round` |
| `L3-IP-032` | `CONTRACT_MISMATCH` | `PASS` | Wired `avid_causal_scheduler_t` and `--commit-delay <rounds>` directly into `avid_daemon.cpp` |
| `L3-IP-033` | `FAIL` | `PASS` | Removed `std::max(6, ...)` floor in `fit_histogram`; `max_iters <= 0` rejected (`0` steps) and `max_iters = 1` runs `<= 1` step |
| `L3-IP-034` | `FAIL` | `PASS` | `avid_fast_detector_init_checked` rejects `threshold <= 0`, `threshold > 1`, `NaN`/`Inf`, and `debounce == 0` |
| `L3-IP-035` | `FAIL` | `PASS` | Enforced supported $K \in \{1,2,3,4,5\}$ in both `CycleAccurateBPEngine::decode` and `avid_hal_set_bp_iterations` |
| `L3-IP-036` | `FAIL` | `PASS` | `avid_daemon` rejects unknown CLI flags, invalid `controller_mode`, `osd_mode > 1`, and unknown/out-of-bounds `--config` YAML keys (`rc = 2`) |
| `L3-IP-038` | `FAIL` | `PASS` | `avid_fallback_step` safely guards `NULL` pointers and UBSan-cleanly sanitizes corrupt `fsm->state` values to `SAFE_STATIC_PRIOR` |
| `L3-ABI-002` | `FAIL` | `PASS` | `dma_ingress_tap.h` exports unmangled `extern "C"` symbols (`avid_dma_ring_init` verified via direct C11 call + `dlsym`) |
| `L3-ABI-004` | `FAIL` | `PASS` | `causal_scheduler.h` & `fallback_fsm.h` export `extern "C"` symbols (`004.b PASS`) and `hal/avid_hal.h` compiles cleanly under `-std=c11` (`004.c PASS`) |
| `L3-ABI-006` | `FAIL` | `PASS` | `9/9` `fork()`-isolated C11 `NULL`-pointer probes across `packet_validator`, `llr_quantizer`, `dma_ingress_tap`, `causal_scheduler`, and `fallback_fsm` exit cleanly (`0` crashes) |
| `L3-INT-001` | `FAIL` | `PASS` | `avid_daemon --stim-bin` enforces 8-byte framing (`3B -> rc=2`) and runs every ingress word through `avid_dma_ring_push_burst` / `avid_validate_packet` |
| `L3-INT-003` | `CONTRACT_MISMATCH` | `PASS` | Wired `ring_buffer_t` (`ring_init` + `ring_push` on validated rounds) into `avid_daemon.cpp` (`ring_buffer_count` emitted in telemetry JSON) |
| `L3-INT-004` | `FAIL` | `PASS` | `avid_daemon --stim-bin` preserves raw packet flags; 64 `PARITY_ERR` packets yield `validator_parity_err_count = 64`, `valid_count = 0`, `bank_swaps = 0` |
| `L3-CFG-MISSING` | `FAIL` (Companion) | `PASS` | Dangling value flags (`--window` without value), missing `--checkpoint` in `avid_r2d2`, and missing `--stim-bin`/`--config` files all exit `rc = 2` |

---

## 5. Stage Gate Status & Mandatory Pause Before Set 2

| Gate | Name | Status | Summary |
| :--- | :--- | :---: | :--- |
| `G1` | Input Integrity & Contract Conformance | **`PASS`** | `51/51` gated items `PASS` (`40 L3-IP` + `7 L3-ABI` + `4 L3-INT`), `2/2` companions `PASS`, `ASan+UBSan rc = 0` |
| `G2`–`G9` | Mathematical, Output/Statistical, Flow, Execution, & Efficacy Gates | `NOT_EVALUATED_YET` | Deferred to Sets 2–5 |
| `G10` | Reproducibility, Traceability, & Branch Isolation Integrity | **`PASS`** | `243/243` canonical IDs intact, `be50761` baseline artifacts untouched, `AVID_EPJ_Patches` (`54da68d`) untouched |

**Next Action**: Pause at Stage 1 completion (`STAGE1_ACCEPTED_G1_G10_PASS`) awaiting user review before starting Layer 3 Test Set 2 (`set2_output_and_stats`).
