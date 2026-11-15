# Layer 3 — Slow-Loop Embedded Software & Adaptive Control: Implementation-to-Test Traceability Specification (`LAYER3_IMPLEMENTATION_TO_TEST_TRACEABILITY.md`)

**Branch Scope:** `layer3-slow-loop-software` (isolated Layer 3 branch split from frozen baseline `AVID_EPJ_Patches` @ `54da68d`)  
**Frozen Baseline Constraint:** `AVID_EPJ_Patches` (`54da68d`) MUST remain strictly unmodified throughout Layer 3 validation and remediation. All Layer 3 traceability docs, test harnesses (`tests/layer3/`), Python/C++ reference oracles, contract guards, and acceptance runners live exclusively on `layer3-slow-loop-software`.

---

## 1. Empirical Source-Code Inspection (Verified Directly from Repo)

Direct line-by-line inspection of `software/src/`, `architecture/`, `configs/`, `deployment/cpp/`, and `tests/` established the following ground-truth architecture before any tests are authored:

| Subsystem | Actual Production Implementation | Verified Exact Interfaces & Behavior |
|---|---|---|
| **Packet Ingress** | [`packet_validator.h`](software/src/ingress/packet_validator.h), [`packet_validator.cpp`](software/src/ingress/packet_validator.cpp) | **64-bit packet (`uint64_t`)**: bits `[3:0]` `s_x`, `[7:4]` `s_z`, `[15:8]` flags (`VALID=0x01`, `OVERFLOW=0x02`, `PARITY_ERR=0x04`, `EPOCH_MARK=0x08`), `[31:16]` `epoch_id`, `[63:32]` `seq_id`. Rejects missing `VALID` or set `PARITY_ERR` or `seq_id != last_seq_id + 1`. **Note (`DEF-L3-01`):** differs from `syndrome_packet_v1.yaml`'s 16-B header + 4-B word framing; has no timestamp or CRC byte. |
| **DMA Ingress & Histograms** | [`dma_ingress_tap.h`](software/src/ingress/dma_ingress_tap.h), [`dma_ingress_tap.cpp`](software/src/ingress/dma_ingress_tap.cpp), [`ring_buffer.h`](software/src/inference/ring_buffer.h) | 4-slot DMA ring (`AVID_DMA_NUM_SLOTS=4`, `AVID_DMA_SLOT_PACKETS=64`) with explicit slot states `FREE(0) -> DMA_OWNED(1) -> CPU_READY(2) -> CPU_OWNED(3) -> FREE(0)`. Consumer calls `avid_validate_packet` per packet and increments separate 16-bin `uint32_t hist_x[16]`, `hist_z[16]`. **Note (`DEF-L3-02`):** `avid_daemon` uses `hist_x/hist_z[16]` directly rather than `ring_buffer_t` (`RING_WINDOW=10000`). |
| **Joint-DP Noise Model (21 params)** | [`noise_model.h`](software/src/inference/noise_model.h), [`noise_model.cpp`](software/src/inference/noise_model.cpp), [`joint_dp_estimator.cpp`](software/src/estimator/joint_dp_estimator.cpp), [`parameter_schema.json`](architecture/estimator/parameter_schema.json) | Frozen `theta_[0..20]`: `[0..8]` = single-qubit `p_0..p_8`; `[9..20]` = 12 NN pairs `(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)`. Exact 16-state XOR-convolution DP over `NM_EVENT_SIGNATURES` (X-checks, detects Z errors) and `NM_EVENT_SIGNATURES_Z_BASIS` (Z-checks, detects X errors). Computes normalized empirical cross-entropy `-\sum_s (hist[s]/N) log P(s|theta) + 0.5*lambda*||theta - prior||^2` and exact analytical coset-inversion gradient. Notice `(3,6)` and `(2,5)` have `0x0` signature in X-checks (`0x0` for `(0,1),(7,8)` in Z-checks) -> unidentifiable zero-likelihood orbits! |
| **Optimizer / L-BFGS Solver** | [`lbfgs_solver.cpp`](software/src/estimator/lbfgs_solver.cpp), [`lbfgs_optimizer.cpp`](software/src/inference/lbfgs_optimizer.cpp) | **Critical finding (`DEF-L3-05`):** `HistogramLBFGSSolver::fit_histogram` declares `LBFGSOptimizer opt_`, **but does NOT call it**! Instead it runs 2 passes of `run_projected_lbfgs_pass` (normalized steepest projected gradient descent with Armijo backtracking, `prior_lambda_=0.04`) + heuristic endpoint-coupled pair prior updates between Pass 1 and Pass 2. Meanwhile `LBFGSOptimizer` implements true Nocedal-Wright M=5 L-BFGS on `ring_buffer_t`. |
| **LLR Quantization (Q7.8)** | [`llr_quantizer.h`](software/src/estimator/llr_quantizer.h), [`llr_quantizer.cpp`](software/src/estimator/llr_quantizer.cpp), `noise_model.cpp:292-427`, [`llr_numeric_contract.yaml`](architecture/contracts/llr_numeric_contract.yaml) | Sign convention: `LLR(p) = log((1-p)/p) > 0` for `p < 0.5`. `avid_prob_to_q78_llr` clamps `p in [1e-5, 0.49]` and computes `lround(llr * 256.0)`; `avid_quantize_and_validate_bank` rejects non-finite or `p not in (0, 0.5)` to `0x0468` (`p=0.012`) and clamps safe range `[0x0040 (+0.25), 0x3000 (+48.0)]`. Separately, `to_pair_projected_q7_8_llr` projects pair excess across aliased edges in `[16, 0x3FFF]`. |
| **Causal Scheduler & Burst Detector** | [`causal_scheduler.h`](software/src/control/causal_scheduler.h), [`causal_scheduler.cpp`](software/src/control/causal_scheduler.cpp), [`system_timing.yaml`](architecture/contracts/system_timing.yaml) | Separate fast & slow components: `avid_causal_scheduler_t` enforces `round_now >= window_end_round + commit_delay_rounds` (single-entry pending swap + `active_epoch_id++`), **though `avid_daemon.cpp` does not instantiate it directly (`DEF-L3-08`)**. `avid_fast_burst_detector_t` implements the **16-round sliding subwindow**, `density_threshold=0.35`, `debounce_rounds=4`, worst-case `20` rounds (`20 us` @ 1 MHz) fast burst detector! Meanwhile slow MAP runs at `window_rounds` (`256` emu / `10,000` prod = `10 ms`). |
| **4-State Safety FSM** | [`fallback_fsm.h`](software/src/control/fallback_fsm.h), [`fallback_fsm.cpp`](software/src/control/fallback_fsm.cpp), [`fault_policy.yaml`](architecture/contracts/fault_policy.yaml) | Four states: `AVID_FSM_NORMAL_ADAPTIVE(0)`, `AVID_FSM_HOLD_LAST_GOOD(1)`, `AVID_FSM_SAFE_STATIC_PRIOR(2)`, `AVID_FSM_DEGRADED_BYPASS(3)`. `!bus_ok -> DEGRADED_BYPASS` (no commit); `estimator && deadline && bounds -> NORMAL_ADAPTIVE` (1-step recovery); `!bounds_ok` or `consecutive_holds >= 3 -> SAFE_STATIC_PRIOR` (commits `static_prior_q78`); else `HOLD_LAST_GOOD` (holds `last_good_q78`, no commit). |
| **HAL, Drivers & `avid_daemon`** | [`avid_hal.h`](software/src/hal/avid_hal.h), [`physical_backend.cpp`](software/src/hal/physical_backend.cpp), [`simulation_backend.cpp`](software/src/hal/simulation_backend.cpp), [`weight_reg_driver.c`](software/src/driver/weight_reg_driver.c), [`avid_daemon.cpp`](software/src/inference/avid_daemon.cpp) | Dual backend HAL (`PHYSICAL` emulated `/dev/mem` fallback via `wrd_handle_t` + local `CycleAccurateBPEngine`; `SIMULATION` over hardened Unix socket JSON tx protocol with `timeout_ms`, `protocol_version==1`, `transaction_id` check). `avid_daemon` drives scenario stream + X/Z Joint-DP + `AvidR2D2LSTMController` (`deployment/cpp/`) + Q7.8 projection + `fallback_fsm` + HAL commit + `--metrics-out` JSON. |

---

## 2. Priority Section 9 Trace & Verified Discrepancies (`DEF-L3-01..10`)

> **Section 9 Priority Trace (`L3-FLOW-001`..`006`):**  
> Before trusting optimizer/statistical behavior, a known 64-bit packet burst (`s_x, s_z, seq_id, epoch_id, VALID`) is traced empirically through:  
> `avid_pack_packet_u64 -> avid_dma_ring_push_burst -> avid_dma_ring_consume_slot (avid_validate_packet -> hist_x[s_x]++, hist_z[s_z]++) -> avid_daemon slot merge -> HistogramLBFGSSolver::fit_histogram (NoiseModel::compute_nll_from_histogram & compute_gradient_from_histogram) -> avid_quantize_and_validate_bank -> avid_fallback_step`.

| Defect ID | Location | Verified Gap / Discrepancy | Planned Remediation / Honest Status Handling |
|---|---|---|---|
| **DEF-L3-01** | `packet_validator.h/cpp` vs `syndrome_packet_v1.yaml` | Code uses 64-bit packet (`s_x, s_z, flags, epoch_id, seq_id`) while YAML specifies 16-B header (`0xAAEC0002`) + 4-B word; lacks explicit timestamp field, duplicate-vs-gap separation, CRC bytes, and null-pointer guards. | Document both exact 64-bit packet layout and YAML stream format; add null/flag/duplicate diagnostics cleanly; treat CRC/hw-timestamp unsupported unless explicitly present (`INCONCLUSIVE`). |
| **DEF-L3-02** | `dma_ingress_tap.cpp` & `avid_daemon.cpp` | `push_burst` silently clamps `n_pkts > 64`; `hist[s]++` lacks `UINT32_MAX` saturation guard; no mutex/atomic snapshot synchronization for concurrent thread ingress/estimator (`L3-IP-019`); `ring_buffer_t` exists separately from `hist[16]` path in daemon. | Validate single-thread slot FSM + `ring_buffer_t -> extract_histogram`; test overflow/truncation contracts & concurrent mutex wrapper in Set 1/3/4. |
| **DEF-L3-03** | `noise_model.cpp:138-159` vs Set 5 formula | `compute_nll_from_histogram` divides by `total` (empirical frequency NLL `-\sum (n_s/N)\log P(s)`) plus unscaled `0.5*\lambda*||\theta-\pi||^2`, so prior weight does not decay with `1/N`. Also `(3,6)` & `(2,5)` in X-checks (and `(0,1),(7,8)` in Z-checks) have `g_e = 0x0` (zero data Fisher info!). | Encode exact empirical cross-entropy oracle in `reference/layer3_reference_oracle.py`; expose rank deficiency (`0x0` & aliased orbits) via Jacobian/Fisher SVD in Set 5C! |
| **DEF-L3-04** | `parameter_schema.json` vs `lbfgs_solver.cpp` vs `llr_numeric_contract.yaml` vs `fault_policy.yaml` | Inconsistent defaults: `lambda` (`0.5` schema vs `0.05` `NoiseModel` vs `0.04` solver); static prior (`0.012` `llr_quantizer.h` vs `0.10` `llr_numeric_contract.yaml` vs `0.01` `fault_policy.yaml`). | Assert actual C++ implementation defaults explicitly against config/reference tables and record parameter-source provenance. |
| **DEF-L3-05** | `estimator/lbfgs_solver.cpp` vs `inference/lbfgs_optimizer.cpp` | `HistogramLBFGSSolver` owns `LBFGSOptimizer opt_` but runs Pass-1/Pass-2 projected gradient descent (`run_projected_lbfgs_pass`) rather than calling `opt_`; convergence tolerance `g_inf < 0.25` is permissive and ignored by `avid_daemon` before quantization! | Validate BOTH `HistogramLBFGSSolver` AND `LBFGSOptimizer` directly against Set 5B finite-diff & two-loop L-BFGS reference oracles; flag ignored non-convergence in `avid_daemon` (`L3-EXEC-025`). |
| **DEF-L3-06** | `llr_quantizer.cpp` vs `weight_reg_driver.h` vs `llr_numeric_contract.yaml` | `llr_quantizer` uses `lround` + `[0x0040, 0x3000]`; `to_q7_8_llr` uses `[-32768, 32767]`; `to_pair_projected` uses `[16, 0x3FFF]`; `wrd_float_to_q78` truncates toward zero. | Test exact contract per API in Set 2B & Set 5D; quantify exact rounding vs truncation differences. |
| **DEF-L3-07** | `fallback_fsm.cpp` vs `fault_policy.yaml` vs `avid_daemon.cpp` | Code implements 4 states (`NORMAL_ADAPTIVE, HOLD_LAST_GOOD, SAFE_STATIC_PRIOR, DEGRADED_BYPASS`), recovers in 1 good window instead of 3, and `avid_daemon.cpp:722` hardcodes `bus_ok = true`. | Test actual 4-state transition table + fault injection of `bus_ok=false`, `!bounds_ok`, `deadline_ok=false`, `estimator_ok=false` in Set 3/4/5F. |
| **DEF-L3-08** | `causal_scheduler.cpp` vs `avid_daemon.cpp` | `avid_causal_scheduler_t` implements delayed commits + `active_epoch_id`, while `avid_daemon` commits synchronously at window boundaries and only calls `avid_fast_detector_step` for 16-round burst detection. Also `avid_fast_detector_step` relies on sequential `round_idx` (`== debounce_rounds`). | Test `avid_causal_scheduler_t` directly in Set 2C/5E AND trace dual timescale (`20 us` fast detector vs `256 us`/`10 ms` slow estimator) honestly in `L3-FLOW-026..035`. |
| **DEF-L3-09** | `avid_daemon.cpp` & `weight_reg_driver.c` | `avid_daemon` has no YAML config loader (uses CLI flags), no restart state checkpoint persistence (`L3-EXEC-018/032`), and `wrd_write_z_llr` writes to shadow offset `0x000..0x020` (same array as X in `wrd_handle_t`). | Enforce CLI `configs/schema.yaml` validation wrapper + honest `INCONCLUSIVE`/`NOT_IMPLEMENTED` flags for absent disk restart state recovery unless persistence is added. |
| **DEF-L3-10** | Across C/C++ boundary functions | Missing null/NaN guards (`compute_nll_from_histogram` clamps `NaN` silently via `std::clamp`; no `hist` overflow check; `--stim-bin` path bypasses `avid_validate_packet`). | Remediate critical input boundary guards cleanly in Set 1 (`L3-IP-*`) without breaking baseline tests. |

---

## 3. Acceptance Gates (`G1`–`G10`) & Status Rules

- **Allowed Gate Statuses:** `PASS` | `FAIL` | `INCONCLUSIVE` | `NOT_IMPLEMENTED`. Unimplemented hardware/daemon features (e.g., CRC checksums if absent, disk checkpoint persistence across `SIGKILL`) MUST be reported explicitly as `INCONCLUSIVE` / `NOT_IMPLEMENTED`, **never** coerced to `PASS`.

| Gate | Scope | Covers Test IDs |
|---|---|---|
| **G1** | Input contracts (packet, histogram, estimator params, config/telemetry) | `L3-IP-001..040` |
| **G2** | Output contracts (MAP, Q7.8 LLR, scheduler/burst, safety FSM, daemon) + statistical recovery | `L3-OP-001..036`, `L3-STAT-001..015` |
| **G3** | Complete ingress-to-commit pipeline dataflow | `L3-FLOW-001..025`, `L3-FLOW-036..045` |
| **G4** | Causal timing, `20 us` fast detector vs `10 ms` slow-loop separation & fast-path isolation | `L3-FLOW-026..035` |
| **G5** | Clean CMake build, `avid_daemon` lifecycle, persistence, failure injection, concurrency | `L3-EXEC-001..045` |
| **G6** | Independent Joint-DP & L-BFGS mathematical oracles | `L3-MATH-001..024` |
| **G7** | Parameter identifiability, Jacobian rank, Fisher information, confidence calibration | `L3-MATH-025..032` |
| **G8** | Q7.8 LLR math reference & causal scheduler equations | `L3-MATH-033..052` |
| **G9** | Four-state safety FSM transition coverage & safety invariants | `L3-MATH-053..062` |
| **G10** | Machine-readable provenance, SHA-256 hashes & 243-ID traceability | All 243 IDs |

---

## 4. Master Implementation-to-Test Traceability Matrix (All 243 Test IDs)

> Tests drive real compiled C/C++ libraries via a lightweight `pybind11`/`ctypes` C-ABI harness + `avid_daemon` + `test_baseline_suite` under `tests/layer3/`.

### Set 1 — Input Contract Validation (`tests/layer3/set1_input_contracts/`) — Gate **G1**

#### 1A. Packet Ingress Contracts (`test_packet_ingress.py`)

| ID | Test | Preconditions | Input Generation & Procedure | Oracle / Expected Result | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L3-IP-001** | Valid packet ingress | Initialized `avid_packet_validator_t` | Pack `(s_x=5, s_z=11, seq=0, epoch=3, flags=AVID_PKT_FLAG_VALID)` -> unpack & validate | Returns `true`; decoded fields exact; `valid_count==1` | G1 | `packet_validator.cpp:avid_pack_packet_u64,avid_validate_packet` |
| **L3-IP-002** | Packet length | Byte-buffer decoder for 64-bit word & `syndrome_packet_v1.yaml` 16B+4B frames | Feed buffers of size `0, 1, 7, 8, 9` B (and `15` B truncated YAML header) | Exact `8` B accepted; mismatched lengths rejected cleanly *(DEF-L3-01)* | G1 | `packet_validator.h`, `syndrome_packet_v1.yaml` |
| **L3-IP-003** | Packet version | Bridge HAL & YAML header validator | Feed `protocol_version=1` vs `0, 2, 99` + stream magic `0xAAEC0002/0xAAEC0001` vs invalid `0xDEADBEEF` | Version `1` / valid magics accepted; others rejected (`rejected_transactions++`) | G1 | `simulation_backend.cpp:133-138`, `syndrome_packet_v1.yaml` |
| **L3-IP-004** | Field encoding | Initialized validator | Test valid flag combos `(VALID, OVERFLOW, EPOCH_MARK)` vs `flags=0` or reserved upper flag bits `[7:4]` set | Missing `VALID` or malformed reserved flags rejected (`invalid_flag_count++`) *(DEF-L3-10)* | G1 | `packet_validator.h:14-18`, `packet_validator.cpp:34-38` |
| **L3-IP-005** | Syndrome bit range | Pack helper & HAL syndrome sender | Test all 256 valid `(s_x in 0..15, s_z in 0..15)` + check masking when passing `16, 255` | Bitwise `(raw & 0xF)`, `((raw>>4) & 0xF)` preserved; validator wrapper rejects `s_x/s_z >= 16` | G1 | `packet_validator.cpp:23-24`, `avid_hal.h:341` |
| **L3-IP-006** | Timestamp validity | `avid_fast_burst_detector_t` / causal scheduler + packet metadata wrapper | Feed monotonic rounds `t=0,1,2...` vs duplicate `t=5,5` vs backward `t=5,4` | Monotonic accepted; duplicate/backward rejected/flagged (`causal` check) *(DEF-L3-01/08)* | G1 | `causal_scheduler.cpp`, `packet_validator.h` (capability check) |
| **L3-IP-007** | Sequence continuity | Validator after packet `seq=10` | Feed `seq=11` (valid), then skip to `seq=13` (gap), then repeat `seq=13` (duplicate) | `seq=11 -> true`; `seq=13` after `11 -> false` & `seq_gap_count==1`; duplicate `seq=13 -> false` & gap/dup count increments | G1 | `packet_validator.cpp:44-51` |
| **L3-IP-008** | Packet corruption | Packet validator & stream parser | Flip `AVID_PKT_FLAG_PARITY_ERR` (`0x04`); separately check CRC/checksum field support | Parity-error packet rejected (`false`, `invalid_flag_count++`); absent hardware CRC field flagged explicitly as `INCONCLUSIVE`/`PARITY_BIT_ONLY` *(DEF-L3-01)* | G1 | `packet_validator.cpp:34-38` |
| **L3-IP-009** | Packet truncation | Unix socket JSON bridge & binary `--stim-bin` reader | Write partial JSON line without `\n` or EOF mid-`uint64_t` (`3` bytes) | Socket returns `-1` (`rejected_transactions++`); short binary frame flagged incomplete | G1 | `simulation_backend.cpp:101-124`, `avid_daemon.cpp:400` |
| **L3-IP-010** | Packet burst | `avid_dma_ring_t` (`4 slots x 64 pkts`) | Push 4 full bursts (`256` pkts) consuming sequentially; then push a 5th burst while all 4 slots are `CPU_READY` | Consumed cadence sustains `0` drops (`overflow_drops==0`); unconsumed 5th burst cleanly returns `false` and increments `overflow_drops==64` | G1 | `dma_ingress_tap.cpp:16-34` |

#### 1B. Syndrome Histogram Contracts (`test_histogram_contracts.py`)

| ID | Test | Preconditions | Input Generation & Procedure | Oracle / Expected Result | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L3-IP-011** | Histogram dimensions | `dma_ingress_tap` & `NoiseModel` | Verify `hist_x[16]`, `hist_z[16]` (`NM_N_STATES == 16`) across all 16 codes `0x0..0xF` | All 16 bins addressable, zero-indexed, out-of-bounds prevented by `& 0x0F` | G1 | `dma_ingress_tap.cpp:52-53`, `noise_model.h:27` |
| **L3-IP-012** | Histogram domain | `NoiseModel::compute_nll_from_histogram` | Pass valid nonnegative counts `[0, 100, ...]` vs test signed negative-cast guard in Python/C++ boundary | Valid nonnegative integers accepted cleanly; invalid negative counts rejected at validator | G1 | `noise_model.cpp:138`, boundary wrapper |
| **L3-IP-013** | Empty histogram | `NoiseModel` & `HistogramLBFGSSolver` | Pass `hist = {0} * 16` (`total == 0`) to `fit_histogram` & `evaluate_nll_and_grad` | Returns `0.0`, `grad == 0`, `diag.converged == false`, leaves `theta_` uncorrupted | G1 | `lbfgs_solver.cpp:82-94`, `noise_model.cpp:144-146` |
| **L3-IP-014** | Single-state histogram | Initialized `avid_dma_ring_t` | Feed 64 valid packets all with `s_x=6, s_z=9` | `hist_x[6]==64`, all other X bins `0`; `hist_z[9]==64`, all other Z bins `0` | G1 | `dma_ingress_tap.cpp:51-54` |
| **L3-IP-015** | Normalized histogram | Known histogram `[800, 50, ..., 0]` (`total=1000`) | Compare `NoiseModel::compute_nll_from_histogram` across scaled `hist` vs `10 * hist` (`10,000`) | Empirical bin proportions identical; NLL invariant under uniform scaling of counts (`DEF-L3-03`) | G1 | `noise_model.cpp:147-152` |
| **L3-IP-016** | Invalid normalization | Normalized PMF input checker in Set 1 wrapper | Pass probability histograms summing to `0.50`, `1.20`, or containing `-0.1` | Normalized validator rejects `|sum - 1.0| > 1e-6` or negative entries | G1 | Validator contract (`tests/layer3/`) |
| **L3-IP-017** | Counter overflow | `dma_ingress_tap` & `NoiseModel` (`uint64_t total`) | Pass bins near `UINT32_MAX / 2` across all 16 bins (`sum > 2^32`) into `compute_nll_from_histogram`, plus saturation test on `consume_slot` | `uint64_t total` in `noise_model.cpp` does NOT wrap around at `2^32`; slot saturation/guard checked *(DEF-L3-02)* | G1 | `noise_model.cpp:140`, `dma_ingress_tap.cpp` |
| **L3-IP-018** | Window rollover | `ring_buffer_t` (`RING_WINDOW=10000`) & daemon epoch reset | Fill `ring_push` with `10,500` items (pattern A for first `500`, pattern B for `10,000`); inspect `ring_at` and daemon epoch clear | Oldest `500` overwritten (`count==10000`); daemon zeros `hist_x, hist_z` after slow epoch | G1 | `ring_buffer.h:35-46`, `avid_daemon.cpp:767-771` |
| **L3-IP-019** | Concurrent histogram update | Multi-thread ingress/consumer harness | Writer pushes slots while estimator takes snapshot on another thread using explicit lock/snapshot protocol (and probe raw lock-free slot state race behavior) | Synchronized snapshot stays internally consistent (`sum(hist) == expected valid pkts`); unsynchronized race hazard documented *(DEF-L3-02)* | G1 | `dma_ingress_tap.cpp` |
| **L3-IP-020** | X/Z independence | Dual ingress streams | Feed bursts with random X patterns & constant `s_z=0`, then constant `s_x=0` & random Z patterns | `hist_x` invariant to Z stream changes and `hist_z` invariant to X stream changes (`syndrome_packet_v1.yaml` invariant #1) | G1 | `dma_ingress_tap.cpp:52-53`, `syndrome_packet_v1.yaml:59` |

#### 1C. Noise-Estimator Parameter Contracts (`test_estimator_inputs.py`)

| ID | Test | Preconditions | Input Generation & Procedure | Oracle / Expected Result | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L3-IP-021** | Parameter vector dimension | `NoiseModel` & `parameter_schema.json` | Assert `NM_N_PARAMS == 21`, `NM_N_QUBIT == 9`, `NM_N_CROSSTALK == 12`, `ParamVec::SizeAtCompileTime == 21` | Exact 21-D schema verified; wrong-length inputs rejected in Python/C++ bridge | G1 | `noise_model.h:23-25`, `parameter_schema.json:4` |
| **L3-IP-022** | Parameter ordering | `NoiseModel`, `parameter_schema.json`, `NM_CROSSTALK_PAIRS` | Compare schema indices `0..20` (`Z_0..Z_8`, pairs `9..20`), XOR signatures `g_e`, and verify permuting `theta` changes signature-matched NLL | Exact 1:1 ordering match with `parameter_schema.json`; permuted vector yields different/incorrect orbit model as detected | G1 | `noise_model.cpp:22-47`, `parameter_schema.json:11-233` |
| **L3-IP-023** | Probability boundaries | `NoiseModel::validate_parameter_bounds` & `clamp_params` | Test `theta = 0.0`, `1e-5`, `0.25`, `0.45`, `0.499`, `0.50`, `1.0` | `validate_parameter_bounds` accepts `[0.0, 0.5)` and rejects `>= 0.5`; `clamp_params` maps to `[1e-5, 0.45]` | G1 | `noise_model.cpp:76-89` |
| **L3-IP-024** | Invalid probabilities | `NoiseModel` & C++ boundary guard | Test `-1e-6`, `0.51`, `1.2`, `NaN`, `+Inf`, `-Inf` | `validate_parameter_bounds` returns `false`; patched NLL/grad boundary rejects `NaN`/`Inf` instead of silent clamp propagate *(DEF-L3-10)* | G1 | `noise_model.cpp:82-89`, guard patch |
| **L3-IP-025** | Prior dimensions | `NoiseModel.prior_theta_` | Construct valid 21-D prior vector vs 9-D / 20-D / 22-D vectors in schema/wrapper | Only 21-D finite prior accepted; wrong dimension rejected with `ValueError` | G1 | `noise_model.h:148`, validator wrapper |
| **L3-IP-026** | Prior strength | `NoiseModel.prior_lambda_` | Test `lambda = 0.0`, `0.04`, `0.5`, boundary `1e6`, and negative `-0.1` | `lambda=0.0` eliminates quadratic penalty; larger `lambda` pulls `theta` toward `prior_theta`; `lambda < 0` rejected | G1 | `noise_model.cpp:154-157,237-239` |
| **L3-IP-027** | Optimizer initialization | `HistogramLBFGSSolver` & `LBFGSOptimizer` | Initialize `NoiseModel` at valid points (`p=0.005`, `0.012`, `0.10`, `[1e-5..0.45]`) and reset optimizer | Both solvers initialize with finite NLL and `hist_count_ == 0` | G1 | `lbfgs_solver.cpp:16`, `lbfgs_optimizer.cpp:9-25` |
| **L3-IP-028** | Invalid optimizer state | Initialized `NoiseModel` + solver wrapper | Pass `NaN`, `+Inf`, or `-0.05` `theta_` into guarded solver entry | Rejected cleanly (`NaN` / negative start cannot execute silent update step) *(DEF-L3-10)* | G1 | `lbfgs_solver.cpp`, `noise_model.cpp` |
| **L3-IP-029** | Missing observations | `HistogramLBFGSSolver` & `avid_daemon` | Empty `hist={0}*16` or sparse `total=3` rounds vs nominal `total=256` | `total==0` returns early with `converged=false`; `total<W_min` keeps conservative fallback/uncertainty behavior | G1 | `lbfgs_solver.cpp:84-94` |
| **L3-IP-030** | Stale observations | `avid_dma_ring_t` + window epoch tracker | Fill all 4 ring slots causing overwrite (`overflow_drops > 0`) or pass epoch-tagged older window after epoch advance | Overrun increments `overflow_drops`; `epoch < current_epoch` window flagged stale and rejected before commit | G1 | `dma_ingress_tap.cpp:21-24`, `fault_policy.yaml:F03` |

#### 1D. Runtime Configuration and Telemetry Contracts (`test_runtime_configuration.py`)

| ID | Test | Preconditions | Input Generation & Procedure | Oracle / Expected Result | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L3-IP-031** | Window configuration | `avid_daemon` & `avid_scheduler_init` | Test `--window` = `64, 256, 1024, 10000` vs `0, -16` | Positive valid lengths accepted; `<= 0` cleanly exits with rc `2` | G1 | `avid_daemon.cpp:250-269` |
| **L3-IP-032** | Commit delay | `avid_causal_scheduler_t` | Init with `commit_delay = 0, 1, 64, 256` vs guard on negative/excessive invalid cast | Valid delays initialize cleanly and enforce `round_now >= end + delay` | G1 | `causal_scheduler.cpp:7-15` |
| **L3-IP-033** | Optimizer limits | `HistogramLBFGSSolver::fit_histogram` & `LBFGSOptimizer` | Pass `max_iters = 1, 10, 20, 50` vs `<= 0`; check `LBFGS_MAX_ITER=50`, `LBFGS_TOL=1e-6` | Executed passes respect iteration bounds; `max_iters <= 0` does zero steps safely / rejected | G1 | `lbfgs_solver.cpp:44,109`, `lbfgs_optimizer.h:20-21` |
| **L3-IP-034** | Burst threshold | `avid_fast_detector_init` | Pass valid `threshold=0.35, 0.50`, `debounce=4` vs `threshold=0.0, -0.5, 1.5, NaN` | Out-of-`(0,1]` threshold either defaults/rejects explicitly without allowing `>=0` always-trigger bug | G1 | `causal_scheduler.cpp:38-46` |
| **L3-IP-035** | Decoder iteration limit | `CycleAccurateBPEngine` & `avid_hal_set_bp_iterations` | Test `K in {1, 2, 3, 4, 5}` (from `system_timing.yaml`) vs `K = 0, 99` in contract validator | Supported `K in 1..5` matches `cycles_fused = 2 + 5*K`; out-of-range flagged invalid | G1 | `system_timing.yaml:8`, `cycle_accurate_bp_engine.cpp:143` |
| **L3-IP-036** | Runtime telemetry | `avid_daemon --metrics-out` & `configs/schema.yaml` validator | Validate valid JSON/YAML config vs malformed unknown keys (`schema.yaml`), missing socket, invalid `--osd-mode 9` | `schema.yaml` & `avid_daemon` reject invalid configs (`rc=2`) and populate complete valid telemetry | G1 | `avid_daemon.cpp:265-289`, `configs/schema.yaml` |
| **L3-IP-037** | Epoch consistency | `weight_reg_driver` + `simulation_backend` + daemon reconciliation | Advance commits & compare packet `pkt.epoch_id` vs `wrd_read_epoch`, plus `init + adapt - reset == hw_epoch` | Exact epoch equality verified; mismatch flagged by bridge (`id_mismatch_errors`) & reconciler | G1 | `weight_commit_protocol.yaml:6`, `avid_daemon.cpp:808-809` |
| **L3-IP-038** | Safety FSM input | `avid_fallback_step` | Drive valid combinations of 4 booleans `(est_ok, ddl_ok, bnd_ok, bus_ok)` & check unknown state enum guard | All 16 boolean tuples map cleanly; corrupted state enum rejected/forced safe *(DEF-L3-10)* | G1 | `fallback_fsm.cpp:17-56` |
| **L3-IP-039** | Active weight bank | `wrd_handle_t` + `CycleAccurateBPEngine` | Write 9 Q7.8 weights (`0x0040..0x3000`) -> commit -> `wrd_read_active_weight(0..8)` | All 9 active bank slots reflect committed vector; out-of-range index returns safe `0` | G1 | `weight_reg_driver.c:108-128` |
| **L3-IP-040** | Shadow weight bank | `avid_quantize_and_validate_bank` -> `wrd_write_all_weights` | Supply valid `[0x0040..0x3000]` bank vs bank containing `0`, negative Q7.8, or `> 0x3000` | Candidate only reaches active bank when `bounds_ok && should_commit`; invalid bank fails validation | G1 | `llr_quantizer.cpp:17-34`, `fallback_fsm.cpp:45` |

---

### Set 2 — Output Contract and Statistical Validation (`tests/layer3/set2_output_statistics/`) — Gate **G2**

#### 2A. MAP Estimator Outputs (`test_map_estimate_outputs.py`)

| ID | Test | Procedure & Acceptance Oracle | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-OP-001** | Estimate dimensions | Run `HistogramLBFGSSolver::fit_histogram` on valid 16-bin histogram -> check output vector length == `21` (`[0..8]` qubit, `[9..20]` pair) | G2 | `noise_model.h:25` |
| **L3-OP-002** | Estimate finiteness | Assert all 21 `theta_(i)` and 21 `grad_(i)` are `std::isfinite` across 20 diverse histograms (sparse, uniform, extreme hotspot) | G2 | `noise_model.cpp`, `lbfgs_solver.cpp` |
| **L3-OP-003** | Parameter bounds | Assert every fitted `theta_(i)` satisfies exact solver box `[1e-5, 0.45]` (and `< 0.5`) | G2 | `noise_model.cpp:76-89` |
| **L3-OP-004** | Objective output | Verify returned NLL `>= 0.0`, finite, and equals `model.compute_nll_from_histogram(hist)` at final `theta_` | G2 | `noise_model.cpp:138-159`, `lbfgs_solver.cpp:173` |
| **L3-OP-005** | Gradient output | Inspect `model.grad()` after `fit_histogram` -> dimension `21`, finite, projected gradient norm matches `diag.final_grad_inf_norm` | G2 | `lbfgs_solver.cpp:20-33,175` |
| **L3-OP-006** | Optimizer status | Verify `LBFGSFitDiagnostics.converged` matches exact condition (`isfinite && nll > 0 && g_inf < 0.25`) vs forced `max_iters=0` failure & `hist=0` | G2 | `lbfgs_solver.cpp:85-93,171-178` |
| **L3-OP-007** | Iteration count | Run with `max_iters=1, 6, 20` -> check `0 <= iterations_executed <= max_iters` matches actual backtracking passes (`it1 + it2`) | G2 | `lbfgs_solver.cpp:51,109-114,169,176` |
| **L3-OP-008** | Initial/final objective | Assert `diag.initial_nll >= diag.final_joint_dp_nll` after monotonic Armijo descent (except where Pass 2 warm-start jump occurs — verify exact Pass 1/Pass 2 accounting!) | G2 | `lbfgs_solver.cpp:105,172-173` |
| **L3-OP-009** | Prior contribution | Evaluate NLL with `prior_lambda_=0` vs `prior_lambda_=0.04` at same final `theta_` -> confirm difference equals exact `0.5 * 0.04 * ||theta - prior||^2` | G2 | `noise_model.cpp:154-157` |
| **L3-OP-010** | Insufficient evidence | Feed `total=0` empty histogram & `total=5` sparse rounds -> `total=0` sets `converged=false, nll=0.0`; `total=5` yields prior-dominated result (`sigma` wide) | G2 | `lbfgs_solver.cpp:84-94` |

#### 2B. Quantized LLR Outputs (`test_q78_outputs.py`)

| ID | Test | Procedure & Acceptance Oracle | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-OP-011** | LLR dimension | Both `avid_quantize_and_validate_bank` & `NoiseModel::to_q7_8_llr` produce exact `int16_t[9]` outputs (`NM_N_QUBIT==9`) | G2 | `llr_quantizer.cpp:17`, `noise_model.cpp:302` |
| **L3-OP-012** | Q7.8 encoding | Test golden reference probabilities `{0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30}` against `llr_numeric_contract.yaml` & `round(log((1-p)/p)*256)` (`<= 1` LSB) | G2 | `llr_numeric_contract.yaml:33-40`, `llr_quantizer.cpp:11` |
| **L3-OP-013** | Quantization rounding | Check half-way and dense grid `llr * 256.0` under `std::lround` (`llr_quantizer.cpp`) vs truncating `wrd_float_to_q78` (`DEF-L3-06`) | G2 | `llr_quantizer.cpp:11`, `weight_reg_driver.h:30` |
| **L3-OP-014** | Saturation | Feed `p=1e-7` (`llr ~ 16.1 -> 4126 < 0x3000`, note to trigger `> 0x3000` need `llr > 48` impossible at `p>=1e-5`, so test `p=1e-5` -> `0x0B83`, `p=0.45` -> `0x0033 < 0x0040` triggers `AVID_Q78_MIN_SAFE` clamp + `all_safe=false`!) | G2 | `llr_quantizer.cpp:27-30` |
| **L3-OP-015** | Zero-probability handling | Pass `p=0.0` into `avid_prob_to_q78_llr` & `avid_quantize_and_validate_bank` -> no `Inf`/`NaN`; bank returns `false` & safe `AVID_Q78_DEFAULT` | G2 | `llr_quantizer.cpp:9,21-25` |
| **L3-OP-016** | Unit-probability handling | Pass `p=1.0` & `p=0.5` -> `avid_prob_to_q78_llr` clamps `0.49` (`>0`); `avid_quantize_and_validate_bank` rejects `>= 0.5` returning `false` & `AVID_Q78_DEFAULT` | G2 | `llr_quantizer.cpp:9,21-25` |
| **L3-OP-017** | Sign convention | Verify monotonically decreasing `q78(p)` on `(0, 0.5)` and `p < 0.5 <=> q78 > 0` matching `llr_numeric_contract.yaml` ("bit 0 preferred for `LLR > 0`") | G2 | `llr_numeric_contract.yaml:8`, `llr_quantizer.cpp:10` |
| **L3-OP-018** | Round-trip conversion | Compute `p -> LLR -> q78 -> q78 / 256.0 -> 1/(1+exp(LLR_q))` over `p in [1e-4, 0.45]` -> check `|LLR_q - LLR| <= 1/512 + eps` where unsaturated | G2 | `llr_quantizer.cpp:11` |
| **L3-OP-019** | Parameter mapping | Set single qubit `q=4` hot (`theta[4]=0.12`), others baseline -> verify ONLY `q=4` (plus any coupled pairs on `4`) drops LLR relative to non-adjacent qubits | G2 | `noise_model.cpp:281-311` |
| **L3-OP-020** | Invalid estimate propagation | Inject `NaN`, `-0.1`, `0.6` into `p_eff` -> assert `avid_quantize_and_validate_bank` returns `false`, `bounds_ok=false` blocks adaptive commit in FSM | G2 | `llr_quantizer.cpp:21-24`, `fallback_fsm.cpp:45` |

#### 2C. Scheduler and Burst Outputs (`test_scheduler_outputs.py`)

| ID | Test | Procedure & Acceptance Oracle | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-OP-021** | Burst indication | Drive `avid_fast_burst_detector_t` with calm `sx=0` then dense `sx=0x2` -> verify `burst_active` and return value boolean state | G2 | `causal_scheduler.cpp:48-76` |
| **L3-OP-022** | Trigger timestamp | Burst starts at `round=100` (`density >= 0.35`, `debounce=4`) -> check exact `round_idx` when `step()` returns `true` (`100 <= r_trig <= 119`) | G2 | `causal_scheduler.cpp:64-70` |
| **L3-OP-023** | Trigger latency | Measure `r_trig - burst_start + 1` across `density in {0.4, 0.5, 1.0}` bursts -> stays in `[debounce, 20]` (never `<= 0` / retrospective!) | G2 | `causal_scheduler.cpp:45,64-70` |
| **L3-OP-024** | Commit scheduling | `avid_scheduler_schedule_commit(window_end=256, w)` with `commit_delay=32` -> `poll_commit_ready(287)==false`, `poll_commit_ready(288)==true` | G2 | `causal_scheduler.cpp:17-36` |
| **L3-OP-025** | Epoch assignment | Poll `0, 1, 2` commits -> `active_epoch_id` increments strictly `0 -> 1 -> 2` matching committed weights | G2 | `causal_scheduler.cpp:32` |
| **L3-OP-026** | Delayed evidence | Schedule commit for `t_target=500`, test re-scheduling with older `window_end` or late HAL response -> guard ensures older timestamp cannot overwrite newer pending commit | G2 | `causal_scheduler.cpp` + scheduler guard |
| **L3-OP-027** | Duplicate trigger | Feed sustained burst for `50` consecutive rounds -> assert `debounce_counter == debounce_rounds` triggers once per burst window (`trigger_count==1`), NOT 40 duplicate commits | G2 | `causal_scheduler.cpp:65-69` |
| **L3-OP-028** | No-evidence condition | Feed `10,000` stationary low-rate rounds (`p=0.012`, subwindow density well below `0.35` almost always; inject constant `p < 0.01` & sub-threshold `5/16=0.3125 < 0.35`) -> exact `0` triggers below threshold | G2 | `causal_scheduler.cpp:64` |

#### 2D. Safety and Daemon Outputs (`test_safety_outputs.py`)

| ID | Test | Procedure & Acceptance Oracle | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-OP-029** | FSM state output | Check `fsm.state in {0, 1, 2, 3}` (`NORMAL_ADAPTIVE, HOLD_LAST_GOOD, SAFE_STATIC_PRIOR, DEGRADED_BYPASS`) after every step | G2 | `fallback_fsm.h:9-14` |
| **L3-OP-030** | Safety transition | Drive exact state sequence (`NORMAL -> HOLD -> HOLD -> SAFE_STATIC -> NORMAL -> DEGRADED -> NORMAL`) verifying exact transition table | G2 | `fallback_fsm.cpp:17-56` |
| **L3-OP-031** | Fallback activation | Independently violate `estimator_ok=false`, `deadline_ok=false`, `bounds_ok=false`, `bus_ok=false` -> verify fallback triggers (`fallback_triggers++`) and exact state | G2 | `fallback_fsm.cpp:25-55` |
| **L3-OP-032** | Recovery eligibility | From `HOLD_LAST_GOOD` & `SAFE_STATIC_PRIOR`: assert faulty steps keep non-`NORMAL` state; healthy step (`all 4 true`) restores `NORMAL_ADAPTIVE` (and document `1`-step vs `yaml`'s `3`-step `DEF-L3-07`) | G2 | `fallback_fsm.cpp:33-40` |
| **L3-OP-033** | Daemon health | Run `avid_daemon --self-test` & scenario runs -> `rc==0` when HAL has `0` rejected tx, non-zero exit when HAL errors or bad args | G2 | `avid_daemon.cpp:774-778,971` |
| **L3-OP-034** | Error reporting | Corrupt packet (`PARITY_ERR`), malformed bridge msg, bad scenario (`--scenario invalid_configuration`) -> counters increment (`invalid_flag_count`, `rejected_transactions`) & stderr reports explicitly | G2 | `packet_validator.cpp:36`, `avid_daemon.cpp:265-269` |
| **L3-OP-035** | Commit result | Inject HAL busy (`bp_ready=0`) vs normal commit -> verify `status_reg`, `hw_epoch`, and `epoch_commit_reconciled` match actual hardware/sim state | G2 | `weight_reg_driver.c:123-132`, `avid_daemon.cpp:808-836` |
| **L3-OP-036** | Runtime counters | Compare `validator.valid_count`, `hal_total_transactions`, `hal_syndrome_packets`, `epochs_completed`, `action_histogram` sum against exact ground-truth loop iteration counts | G2 | `avid_daemon.cpp:826-965` |

#### 2E. Statistical Estimator Validation (`test_estimator_recovery.py` — >= 30 seeds)

| ID | Test | Experimental Design (>= 30 seeds) | Statistical Acceptance Metrics (reported in JSON, not a single metric!) | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L3-STAT-001** | Single-parameter recovery | Vary `theta[4]` (clean singleton orbit `0x6`) across `{0.015, 0.04, 0.08}` with `N=2000`, `30` seeds | Report empirical `Bias`, `Variance`, `MSE`, `95% CI`; `|Bias| <= 0.015`, predictive JS/TV error small | G2 | `lbfgs_solver.cpp`, `parameter_schema.json:482-498` |
| **L3-STAT-002** | Multi-parameter recovery | Vary `theta[1], theta[4], theta[7]` jointly across `30` seeds | Track parameter vector RMSE, 9-class `compute_orbit_rmse`, and KL/TV predictive syndrome error | G2 | `noise_model.cpp:264-279` |
| **L3-STAT-003** | Independent-noise recovery | Pure independent `theta[0..8] ~ U[0.008, 0.03]` (`theta[9..20]=0`) over `30` seeds | Compare orbit RMSE + non-aliased single-qubit recovery (`q1, q4, q7`) vs aliased pairs | G2 | `lbfgs_solver.cpp` |
| **L3-STAT-004** | Correlated-noise recovery | Turn on pairs `(1,2)`, `(4,5)`, `(6,7)` at `{0.03, 0.08}` (`30` seeds) | Measure both `theta[9+k]` error and endpoint-disambiguated pair error + orbit RMSE separately from singles! | G2 | `lbfgs_solver.cpp:131-166` |
| **L3-STAT-005** | Measurement-noise recovery | Generate synthetic syndromes with phenomenological readout bit-flip `p_meas = 0.012` (`30` seeds) | Measure parameter bias under readout misspecification vs readout-free case (`M9`) | G2 | `avid_daemon.cpp:443-448` |
| **L3-STAT-006** | Sample-size sensitivity | Sweep window `W in {256, 1000, 5000, 20000}` (`30` seeds each) on `Z_4` | Empirical estimator variance & orbit RMSE strictly decrease monotonically with `W` | G2 | `lbfgs_solver.cpp`, `noise_model.cpp` |
| **L3-STAT-007** | Seed variation | `30` independent seeds across baseline & burst profiles | Report mean, std, IQR, 5th/95th percentiles for NLL, orbit RMSE, 21-param RMSE | G2 | Set 2 statistical harness |
| **L3-STAT-008** | Prior sensitivity | Sweep `prior_lambda in {0.0, 0.04, 0.5, 5.0}` (`W=500`, `30` seeds) | Quantify shrinkage bias toward `prior_theta` vs variance reduction curve | G2 | `noise_model.cpp:154-157` |
| **L3-STAT-009** | Initial-point sensitivity | Initialize `theta` at `p0 in {0.002, 0.012, 0.05, 0.15}` before `fit_histogram` on identical histograms (`30` cases) | Measure spread across identifiable 9-class orbits vs unidentifiable null/aliased coordinates | G2 | `lbfgs_solver.cpp:78` |
| **L3-STAT-010** | Model mismatch | Inject out-of-model 3-qubit streak (`{3,4,5}`) & asymmetric bias burst (`30` seeds) | Measure KL divergence `D_KL(P_emp || P_hat)` growth & flag misspecification residual cleanly | G2 | `avid_daemon.cpp:197-205` |
| **L3-STAT-011** | Parameter identifiability | Jointly perturb aliased pairs `(Z_0, ZZ_1_2)` keeping XOR-orbit constant (`P(0x1)` fixed) + null edges `ZZ_3_6, ZZ_2_5` | Prove individual coordinates shift along null-space/prior while orbit probability recovers tightly (`DEF-L3-03`) | G2 | `parameter_schema.json`, `noise_model.cpp` |
| **L3-STAT-012** | Drift tracking | Slowly ramp `theta[4]` & `theta[9+3]` across `20` sequential windows (`30` seeds) | Measure tracking lag (windows behind true ramp) and dynamic tracking RMSE | G2 | `avid_daemon.cpp:170-175` |
| **L3-STAT-013** | Burst contamination | Inject `60`-round burst inside `10,000`-round stationary window (`30` seeds) | Measure tiny dilution effect (`0.6%` of window) on slow MAP vs immediate `20 us` fast detector trigger | G2 | `system_timing.yaml` |
| **L3-STAT-014** | Stationary false alarms | Run `50,000` stationary rounds (`p=0.012`, `30` seeds) through `avid_fast_burst_detector_t` | Record empirical false-alarm trigger rate vs `density >= 0.35` binomial tail probability | G2 | `causal_scheduler.cpp:48-76` |
| **L3-STAT-015** | Statistical calibration | Compute inverse observed Fisher std errors on identifiable singleton `Z_1, Z_4, Z_7` across `30` Monte Carlo trials | Compare empirical 95% coverage (`|theta_hat - theta| <= 1.96 * SE`) vs nominal `~95%` | G2 | `noise_model.cpp`, Set 5C |


---

### Set 3 — Pipeline and System-Flow Validation (`tests/layer3/set3_pipeline_flow/`) — Gates **G3, G4**

#### 3A. Component-to-Component Boundary Tests (`test_ingress_to_histogram.py`, `test_histogram_to_estimator.py`, `test_estimator_to_llr.py`, `test_scheduler_commit_flow.py`) — Gate **G3**

> Every boundary captures producer output AND consumer input independently (never merely checking return code `0`). `L3-FLOW-001..006` form the Section 9 priority ingress-to-estimator continuity proof.

| ID | Boundary | Producer Output -> Consumer Input Verification Procedure | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-FLOW-001** | Packet ingress -> histogram | Pack a known 64-bit sequence (`s_x in {0, 6}`, `s_z in {0, 9}`), push into `avid_dma_ring_t`, consume slot -> verify exact bin delta `hist_x[6], hist_z[9]` matches unpacked packet count | G3 | `packet_validator.cpp` -> `dma_ingress_tap.cpp:50-55` |
| **L3-FLOW-002** | Histogram -> observation window | Accumulate `256` rounds across 4 DMA slots (`4 x 64`) in daemon loop (plus `ring_push` rollover) -> verify window `hist_x` sum equals `256` before epoch fit & resets cleanly to `0` after epoch (`subepoch_fast_adapt=false`) | G3 | `dma_ingress_tap.cpp` -> `avid_daemon.cpp:499-505,767-771` |
| **L3-FLOW-003** | Observation window -> estimator | Capture exact `hist_x` snapshot immediately prior to `fit_histogram` AND verify `compute_nll_from_histogram` reads that exact snapshot (perturb 1 count in `hist_x[6]` -> NLL/grad changes proportionally) | G3 | `avid_daemon.cpp:584` -> `lbfgs_solver.cpp:78` |
| **L3-FLOW-004** | Estimator -> optimizer | Verify `fit_histogram` passes `hist`, basis flag (`basis_z=false` X-checks vs `true` Z-checks), priors `(p0, p_pair0)`, bounds `[1e-5, 0.45]`, and Pass-2 coupled prior into `run_projected_lbfgs_pass` *(note `DEF-L3-05`: tests both `HistogramLBFGSSolver` & `LBFGSOptimizer`)* | G3 | `noise_model.cpp` -> `lbfgs_solver.cpp:96-169` |
| **L3-FLOW-005** | Optimizer -> noise estimate | Compare `g_model_z.theta_` after `fit_histogram` against inputs fed into `p_eff` and `AvidR2D2LSTMController::step_causal` — exact bit match | G3 | `lbfgs_solver.cpp` -> `avid_daemon.cpp:617,691` |
| **L3-FLOW-006** | Noise estimate -> LLR converter | Pass live `theta_` through `effective_p(0..8)` and `avid_quantize_and_validate_bank` / `to_pair_projected_q7_8_llr` -> verify Q7.8 candidate outputs match independent `round(log((1-p_eff)/p_eff)*256)` on unprojected bank | G3 | `noise_model.cpp:281-324`, `llr_quantizer.cpp:17-34` |
| **L3-FLOW-007** | LLR converter -> scheduler | Pipe candidate `int16_t cand_q78[9]` from quantizer directly into `avid_scheduler_schedule_commit` (`sched.pending_weights_q78`) -> compare exact 9 words at scheduler ingress & poll egress | G3 | `llr_quantizer.cpp` -> `causal_scheduler.cpp:17-35` |
| **L3-FLOW-008** | Burst detector -> scheduler | Drive `avid_fast_detector_step` over `60`-round burst -> capture trigger round & `trigger_count` -> verify sub-epoch trigger & `burst_score` feeding `step_causal` in `avid_daemon` | G3 | `causal_scheduler.cpp:48` -> `avid_daemon.cpp:461-626` |
| **L3-FLOW-009** | Scheduler -> safety FSM | Pass `(est_ok, ddl_ok, bounds_ok, bus_ok, cand_q78)` into `avid_fallback_step` -> verify output `sel_q78` & `should_commit` match FSM state decision | G3 | `avid_daemon.cpp:718-725` -> `fallback_fsm.cpp:17` |
| **L3-FLOW-010** | Safety FSM -> shadow bank | Fault `bounds_ok=false` vs `all_ok=true`: verify `NORMAL_ADAPTIVE` writes `cand_q78` to shadow reg `WRD_OFF_SHADOW(0..8)`, while `HOLD_LAST_GOOD` blocks commit (`should_commit==false`) leaving active shadow unchanged! | G3 | `fallback_fsm.cpp` -> `weight_reg_driver.c:108-112` |
| **L3-FLOW-011** | Shadow bank -> commit interface | Write `sel_q78` -> `avid_hal_write_weights_and_commit` (`0x048=1`) -> read `0x050` (`hw_epoch`) & active reg `0x024..0x044` -> verify atomic swap & epoch bump | G3 | `weight_bank_driver.c`, `avid_hal.h:69-73` |
| **L3-FLOW-012** | Commit response -> daemon | Inspect HAL `commit_operations`, `rejected_transactions`, `hw_epoch`, and `--metrics-out` JSON `epoch_commit_reconciled`, checking fault response updates daemon stats & exit code | G3 | `physical_backend.cpp`/`simulation_backend.cpp` -> `avid_daemon.cpp:761-971` |

#### 3B. End-to-End Control-Flow Tests (`test_end_to_end_control.py`) — Gate **G3**

| ID | Test | Execution via Real Stack (`avid_daemon` + HAL bridge / sim) | Required Outcome | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L3-FLOW-013** | Nominal stationary operation | `avid_daemon --scenario stationary_noise --rounds 1536 --window 256` | Finite NLL, `fallback_events==0`, `epoch_commit_reconciled==true`, low variance estimates | G3 | `avid_daemon.cpp` |
| **L3-FLOW-014** | Slowly drifting noise | `--scenario slowly_drifting_noise --rounds 2048 --window 256` | Estimated `theta[4], theta[12]` and active Q7.8 LLR track upward drift across epochs | G3 | `avid_daemon.cpp:170-175` |
| **L3-FLOW-015** | Correlated-noise transition | Compare `stationary_noise` vs `correlated_noise` vs `heldout_multi_pair_12_67` | Hotspot pair LLRs adapt downward & outperform `static` B0 LER (`ler < static_b0_logical_error_rate` when `backend=sim`) | G3 | `noise_model.cpp:336-426`, `avid_daemon.cpp` |
| **L3-FLOW-016** | Burst disturbance | `--scenario burst_disturbance --rounds 2048 --window 256` | `decoded_rtl_transactions == 2048` (0 halted rounds); fast detector triggers cleanly while slow loop adapts across windows | G3 | `avid_daemon.cpp:175-180,460-531` |
| **L3-FLOW-017** | Delayed weight commit | Mock Unix bridge HAL holding `bp_ready=0` / delayed commit ack vs `--scenario delayed_weight_commit` | Pending commit deferred until ready (`wrd_commit_and_wait`), epoch counter stays exact (`reconciled==true`) | G3 | `weight_reg_driver.c:123-132`, `causal_scheduler.cpp` |
| **L3-FLOW-018** | Reset recovery | `--scenario reset_recovery --rounds 1024 --window 256` | `avid_hal_reset_rtl` resets controller hidden state (`reset_hidden`) & active bank; post-reset commits resume & `epoch_commit_reconciled==true` | G3 | `avid_daemon.cpp:736-741,808-809` |
| **L3-FLOW-019** | Busy decoder | Drive `status_reg` with `bp_ready=0, commit_pending=1` in emulated register file / mock bridge (`--scenario busy_decoder`) | `wrd_commit_and_wait` times out safely (`-1`) instead of corrupting active weights midway | G3 | `weight_reg_driver.c:123-132` |
| **L3-FLOW-020** | Variable iteration control | `--scenario variable_k_adaptation` (`K=1..5`) + live R2D2 `bp_iterations` MMIO `0x054` writes | Hardware reg `0x054` receives exact `K in {1..5}`; `bp_core_cycles` reflects `2 + 5*K` | G3 | `avid_daemon.cpp:744-758`, `system_timing.yaml:9` |
| **L3-FLOW-021** | Fallback activation | `--scenario fallback_safety_trigger` (corrupt `hist_x=0, theta=0.65`) + injected `!deadline_ok`, `!bus_ok` | `fallback_events >= 1`; `SAFE_STATIC_PRIOR` / `HOLD` / `DEGRADED_BYPASS` activate accurately (`DEF-L3-07`) | G3 | `avid_daemon.cpp:545-551,718-733` |
| **L3-FLOW-022** | Recovery after fallback | Epoch 1 corrupted (`fallback_safety_trigger`) -> Epochs 2..4 healthy contiguous packets | Daemon returns to `AVID_FSM_NORMAL_ADAPTIVE`, resumes adaptive commits without permanent lock | G3 | `avid_daemon.cpp:718-765`, `fallback_fsm.cpp:33` |
| **L3-FLOW-023** | Multiple update cycles | Run 12 consecutive epochs (`rounds=3072, window=256`) | Monotonic epochs `1..13`, zero skipped/reordered weight banks | G3 | `avid_daemon.cpp` |
| **L3-FLOW-024** | Concurrent ingress and optimization | Multi-thread harness running continuous `avid_dma_ring_push_burst` at high cadence while background thread fits histograms under mutex | Zero dropped/corrupted snapshots under valid buffer slack; explicit count accounting | G3 | `dma_ingress_tap.cpp`, `lbfgs_solver.cpp` |
| **L3-FLOW-025** | Concurrent ingress and commit | Inject syndrome decode stream (`avid_hal_send_syndrome`) concurrently while shadow bank write + `0x048` commit executes on HAL/driver | Every syndrome decode completes atomically with either old valid epoch or new valid epoch (never mixed half-written X/Z bank!) | G3 | `avid_hal.h`, `weight_commit_protocol.yaml` |

#### 3C. Causal Timing and Timescale Validation (`test_causal_timing.py`) — Gate **G4**

> **Mandatory Spec Check (`1 us/round`, `10 ms = 10,000 rounds` slow window, `60 us = 60 rounds` burst):** Validates real slow vs fast boundary cleanly with **zero fake instant-burst claims on slow MAP**.

| ID | Test | Procedure & Measurement | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-FLOW-026** | Observation-window timing | Configure `window_rounds in {256, 1000, 10000}` -> assert `acc_rounds == target_win` at every slow MAP execution boundary (when no subepoch fast trigger fired) | G4 | `avid_daemon.cpp:509-511` |
| **L3-FLOW-027** | Evidence availability | Inject burst starting at `round=1000`; freeze stream at `round=999` -> verify slow MAP at `round=256, 512, 768` & fast detector `< 1000` show zero advance knowledge of future burst | G4 | Causal step-by-step execution |
| **L3-FLOW-028** | Commit delay | Set `commit_delay=100` in `avid_causal_scheduler_t` after `window_end=256` -> prove `poll_commit_ready` stays `false` for rounds `256..355` and fires strictly at `round >= 356` (`t_c >= t_e + d_c`) | G4 | `causal_scheduler.cpp:21,29` |
| **L3-FLOW-029** | Burst onset | Start burst at `round=100` on `avid_fast_burst_detector_t` -> record exact `first_burst_evidence_round=100` and trigger round `r_trig >= 103` (`16`-subwindow density >= `0.35`, debounce `4`) | G4 | `causal_scheduler.cpp:48-76` |
| **L3-FLOW-030** | Burst end | End 60-round burst at `round=160` (`return to sx=0`) -> record `burst_end=160` vs subsequent detector recovery (`burst_active=false`) after window drains (`<= 16` rounds later) | G4 | `causal_scheduler.cpp:71-74` |
| **L3-FLOW-031** | Slow-loop latency | Measure real wall/CPU microseconds of `fit_histogram` + R2D2 + Q7.8 (`avid_stage_profiler_t` over 100 runs) -> report `p50, p95, p99, max` us vs `10,000 us` budget | G4 | `stage_profiler.cpp:32-45`, `avid_daemon.cpp:679-681` |
| **L3-FLOW-032** | Fast-loop isolation | Decoupled timing harness: run slow `HistogramLBFGSSolver::fit_histogram` on Thread A while Thread B executes `CycleAccurateBPEngine::decode` -> verify zero shared blocking lock / fast-path decode cycle latency remains strictly `17` cycles (`68 ns @ 250 MHz`) | G4 | `cycle_accurate_bp_engine.cpp`, `system_timing.yaml` |
| **L3-FLOW-033** | Stale estimate protection | Attempt to schedule/commit an estimate tagged `epoch_src < active_epoch_id` (or earlier `window_end`) after newer epoch committed -> guard rejects stale overwrite | G4 | `causal_scheduler.cpp`, daemon epoch guard |
| **L3-FLOW-034** | Delayed detection reporting | Feed `short_burst_60us_vs_10ms` (`window=10000`, `60`-round burst at `r=9940..9999`) -> verify `avid_fast_burst_detector_t` detects within `<= 20` rounds (`20 us`) ONLY via the 16-round fast path, whereas slow `10,000`-round MAP (`0.6%` dilution) does NOT detect cleanly before `r=10,000` | G4 | `avid_daemon.cpp:181-187,461`, `system_timing.yaml` |
| **L3-FLOW-035** | Timescale claim validation | Cross-check reported metrics against measured execution: fast detector trigger rate (`<= 20 rounds`), V1 sim epoch (`256 us`), V2 prod cadence (`10 ms`, `10,000` rounds), and BP cycle latency (`17` cycles) | G4 | `system_timing.yaml`, `avid_daemon.cpp` |

#### 3D. Runtime Parameter Propagation & Bypass Detection (`test_parameter_propagation.py`) — Gate **G3**

| ID | Test | Sensitivity Perturbation Method | Required Detection | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L3-FLOW-036** | Probability propagation | Change generator `p_4` from `0.012` -> `0.090` keeping RNG stream identical -> check fitted `theta[4]` | Strictly increases `theta[4]` (`> 2x`) | G3 | `lbfgs_solver.cpp`, `avid_daemon.cpp` |
| **L3-FLOW-037** | Histogram propagation | Add `+200` counts directly to `hist[6]` (`Z_4` signature) keeping other bins identical | `evaluate_nll_and_grad` & fitted `theta[4]` increase monotonically | G3 | `noise_model.cpp:138`, `lbfgs_solver.cpp` |
| **L3-FLOW-038** | Prior propagation | Change `prior_theta_(4)` from `0.01` -> `0.15` (`prior_lambda_=2.0`) on small sample `N=50` | Objective value & fitted `theta[4]` shift upward toward `0.15` | G3 | `noise_model.cpp:154-157` |
| **L3-FLOW-039** | Window propagation | Run daemon with `--window 128` vs `--window 512` at `--rounds 1024` | `epochs_completed` changes strictly (`8` vs `2`) & variance shrinks | G3 | `avid_daemon.cpp:250,509` |
| **L3-FLOW-040** | Threshold propagation | Init `avid_fast_detector_init` with `threshold=0.20` vs `0.80` on fixed `50%`-density stream | `0.20` triggers (`trigger_count >= 1`); `0.80` triggers `0` times | G3 | `causal_scheduler.cpp:38-46` |
| **L3-FLOW-041** | LLR propagation | Raise fitted `theta[4]` -> run quantizer -> inspect `q78[4]` | `q78[4]` decreases strictly (smaller LLR = higher error probability) | G3 | `llr_quantizer.cpp:8-15` |
| **L3-FLOW-042** | Commit propagation | Write two distinct candidate banks `bankA != bankB` through `wrd_write_all_weights` -> commit -> `wrd_read_active_weight` | Active register bank equals exact `bankB` bitwise | G3 | `weight_reg_driver.c:108-128` |
| **L3-FLOW-043** | Static-output detection | Run two distinct seeds/scenarios (`stationary_noise` vs `correlated_noise`) to fresh `--metrics-out` files (after wiping prior artifacts) | Final NLL, action histograms, & `orbit_9class_rmse` differ cleanly (`no canned fixture output`) | G3 | `avid_daemon.cpp` |
| **L3-FLOW-044** | Component invocation trace | Instrument test driver counters / HAL stats across `avid_dma_ring`, `HistogramLBFGSSolver`, `AvidR2D2LSTMController`, `avid_quantize`, `avid_fallback_step`, `avid_hal` | Every stage invocation count `> 0` (`hal_mmio_writes, hal_mmio_reads, commits, lbfgs_iters, action_histogram` sum `> 0`) | G3 | `avid_daemon.cpp:826-965` |
| **L3-FLOW-045** | Consumer-use trace | Corrupt candidate `q78[0]` right before `avid_fallback_step` / tamper shadow weight before `wrd_commit` | Downstream `CycleAccurateBPEngine::decode` decision & `wrd_read_active_weight` visibly react, proving zero dead intermediate outputs | G3 | `avid_daemon.cpp:695-765`, `cycle_accurate_bp_engine.cpp` |

---

### Set 4 — Execution, Persistence and Fault-Recovery Validation (`tests/layer3/set4_execution_recovery/`) — Gate **G5**

#### 4A. Build and Execution (`test_clean_build.py`, `test_daemon_execution.py`)

| ID | Test | Execution Procedure | Required Result | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L3-EXEC-001** | Clean C++ build | Remove build dir (`tmp_path`), run `cmake -S software -B <build> -DCMAKE_BUILD_TYPE=Release && cmake --build <build> -j2` | Exits `0` with zero missing dependencies, emits `avid_daemon` and `test_baseline_suite` | G5 | `software/CMakeLists.txt` |
| **L3-EXEC-002** | Daemon initialization | Launch `./avid_daemon --self-test --metrics-out out.json` checking startup banner, R2D2 checkpoint load, HAL open, initial boot commit | Exits `0`, `r2d2_checkpoint_loaded == true`, `initial_boot_commits == 1` | G5 | `avid_daemon.cpp:274-331` |
| **L3-EXEC-003** | Self-test | Run `./avid_daemon --self-test` & `./test_baseline_suite` | Both complete genuine math/DMA/FSM/BP/R2D2 checks (`[ALL C++ BASELINE TESTS PASSED]`) | G5 | `avid_daemon.cpp:774`, `test_baseline_suite.cpp` |
| **L3-EXEC-004** | Runtime configuration | Execute `avid_daemon` across distinct `--rounds`, `--window`, `--seed`, `--controller-mode static|avid_r2d2|b2_pid` | Metrics JSON reflects exact supplied CLI parameters & distinct controller trajectories | G5 | `avid_daemon.cpp:233-263,816-830` |
| **L3-EXEC-005** | Invalid configuration | Launch `./avid_daemon --scenario invalid_configuration`, `--window 0`, `--rounds -5`, `--osd-mode 9` | Rejects cleanly with exit code `2` & explicit diagnostic message on stderr | G5 | `avid_daemon.cpp:265-269` |
| **L3-EXEC-006** | Missing input | Launch `./avid_daemon --checkpoint /no/such.bin` & `--stim-bin /no/such.bin` & `--backend sim --socket /no/such.sock` | Safe startup failure (non-zero exit `1`/`2`, zero segfault) | G5 | `avid_daemon.cpp:280-305` |
| **L3-EXEC-007** | Repeated start/stop | Spawn `avid_daemon` cleanly 10 times sequentially in `tmp_path` | Every run produces deterministic identical seed output (`zero cross-run state leakage`) | G5 | `avid_daemon.cpp` |
| **L3-EXEC-008** | Long-running daemon | Run `avid_daemon --rounds 50000 --window 256` (~200 epochs) sampling RSS & runtime | Finishes cleanly (`rc=0`), flat RSS growth (`no unbounded memory leaks`), zero OOM | G5 | `avid_daemon.cpp` |
| **L3-EXEC-009** | Multiple runtime cycles | Run 20 slow adaptation epochs checking continuous `hw_epoch`, `adaptive_epoch_commits`, finite `param_21_rmse` | Continuous monotonic epoch state (`hw_epoch == 1 + adaptive_epoch_commits`) | G5 | `avid_daemon.cpp:808-836` |
| **L3-EXEC-010** | Shutdown | Send `SIGTERM` / `SIGINT` mid-run to `avid_daemon` (`--rounds 5000000`) | Signal handler breaks `while (g_running)` loop cleanly, flushes `--metrics-out` JSON & closes HAL (`rc == 0`)! | G5 | `avid_daemon.cpp:47-48,271-272,787-971` |

#### 4B. Runtime Data and Persistence (`test_persistence.py`)

| ID | Test | Procedure & Audit Rule | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-EXEC-011** | Histogram persistence | Check JSON telemetry (`decoded_rtl_transactions`, `r2d2_action_histogram`, window rounds) AND verify whether raw 16-bin histogram snapshots persist to disk (`INCONCLUSIVE` / gap documented for `--metrics-out` summary-only schema) | G5 | `avid_daemon.cpp:814-966` *(DEF-L3-09)* |
| **L3-EXEC-012** | Estimate persistence | Verify `orbit_9class_rmse`, `param_21_rmse`, `final_nll` persist in `--metrics-out` + document absence of raw `theta_[0..20]` dump in daemon JSON (`DEF-L3-09`) unless diagnostic dump enabled | G5 | `avid_daemon.cpp:838-843` |
| **L3-EXEC-013** | Optimizer diagnostics | Assert `--metrics-out` saves `final_nll`, `legacy_marginal_nll`, `lbfgs_iterations`, `lbfgs_grad_inf_norm` matching live `LBFGSFitDiagnostics` | G5 | `avid_daemon.cpp:838-841` |
| **L3-EXEC-014** | Weight persistence | Probe committed active weights via HAL MMIO read `0x100..0x148` & verify shadow/active equivalence after swap | G5 | `physical_backend.cpp:54-62`, `weight_reg_driver.c` |
| **L3-EXEC-015** | Epoch persistence | Assert `initial_boot_commits`, `adaptive_epoch_commits`, `reset_cleared_commits`, `weight_bank_commits`, `epoch_commit_reconciled` match cycle-by-cycle ledger | G5 | `avid_daemon.cpp:808-836` |
| **L3-EXEC-016** | Safety-state persistence | Assert `fallback_events` count matches `fallback_safety_trigger` activations & report explicit gap (`NOT_IMPLEMENTED` / `INCONCLUSIVE`) where per-transition cause log file isn't persisted by daemon *(DEF-L3-09)* | G5 | `avid_daemon.cpp:731-733,837` |
| **L3-EXEC-017** | Runtime provenance | Validate telemetry retains `r2d2_checkpoint_path`, `seed`, `scenario`, `backend`, `controller_mode` + compute SHA-256 of `avid_daemon` binary & checkpoint in Python harness | G5 | `avid_daemon.cpp:815-828` |
| **L3-EXEC-018** | Restart state | Restart `avid_daemon` cleanly -> verify documented fresh deterministic initialization from checkpoint (`r2d2_v1_trained.bin`) & explicit note that warm disk-checkpoint restoration is not implemented (`INCONCLUSIVE` for warm resume, `PASS` for deterministic clean reset!) *(DEF-L3-09)* | G5 | `avid_daemon.cpp:274-331` |
| **L3-EXEC-019** | Log integrity | Validate valid JSON parsing, non-negative counters, `sum(r2d2_action_histogram) == epochs_completed` (in `avid_r2d2` mode), `initial + adaptive - reset == hw_epoch` | G5 | `avid_daemon.cpp:808-965` |
| **L3-EXEC-020** | Artifact reconstruction | Replay identical CLI flags + seed via reference script reading saved metrics JSON provenance and verify exact bit-for-bit reproducible metric reconstruction! | G5 | `avid_daemon.cpp:50-62,814-966` |

#### 4C. Failure Injection and Recovery (`test_failure_injection.py`)

| ID | Test | Controlled Failure Injection at Component Boundary | Required Safe Outcome | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L3-EXEC-021** | Malformed packet | Inject packet with `flags=0` and packet with `PARITY_ERR` inside a valid DMA slot | Valid packets accumulate normally; corrupted packets rejected (`hist` unpolluted, `invalid_flag_count` increments) | G5 | `dma_ingress_tap.cpp:50-55`, `packet_validator.cpp` |
| **L3-EXEC-022** | Packet loss | Drop `seq=5` cleanly (`4 -> 6`) in ingress stream | `seq_gap_count` increments by `1`; loss explicitly counted | G5 | `packet_validator.cpp:44-50` |
| **L3-EXEC-023** | Packet duplication | Repeat `seq=7` back-to-back (`6, 7, 7, 8`) | Duplicate `7` fails sequence continuity check (`false`) avoiding double-count | G5 | `packet_validator.cpp:44-50` |
| **L3-EXEC-024** | Empty observation window | Force `hist = {0}*16` in solver & `fallback_safety_trigger` epoch `1` in daemon | `fit_histogram` returns `0.0, converged=false`; `last_nll == 0.0 -> estimator_ok=false` triggers `HOLD_LAST_GOOD` | G5 | `lbfgs_solver.cpp:84-94`, `avid_daemon.cpp:547,719` |
| **L3-EXEC-025** | Optimizer non-convergence | Set `max_iters=0` or inject bad start checking `diag.converged == false` | Guarded pipeline checks `diag.converged` *(or rejects unphysical params)* preventing blind commit (`DEF-L3-05`) | G5 | `lbfgs_solver.cpp:177`, `avid_daemon.cpp:719` |
| **L3-EXEC-026** | Optimizer NaN | Inject `NaN` probability into `NoiseModel::theta_` / solver input & `p_eff` into `avid_quantize_and_validate_bank` | Quantizer returns `false` (`bounds_ok=false`) -> `SAFE_STATIC_PRIOR`; guarded NLL rejects `NaN` cleanly *(DEF-L3-10)* | G5 | `llr_quantizer.cpp:21-24`, `fallback_fsm.cpp:45` |
| **L3-EXEC-027** | Out-of-bounds estimate | Set `theta[0..8] = 0.65` (`fallback_safety_trigger`) -> `p_eff = 0.65 >= 0.5` | `avid_quantize_and_validate_bank` returns `false`, replaces candidates with `0x0468`, triggers `SAFE_STATIC_PRIOR` | G5 | `avid_daemon.cpp:549,695-697`, `llr_quantizer.cpp:21` |
| **L3-EXEC-028** | Quantization overflow | Force extreme `p=1e-12` (`llr*256 ~ 7073`) & `p=0.489` (`q78 ~ 11 < 0x0040`) | Saturates deterministically to `0x3000`/`0x0040` and flags `all_safe=false` on out-of-safe-range inputs | G5 | `llr_quantizer.cpp:11-33` |
| **L3-EXEC-029** | Commit timeout | Connect `simulation_backend` to Python Unix-socket mock server that stalls `SYNDROME_TRANSFER`/`COMMIT` past `timeout_ms` (`SO_RCVTIMEO`) | `send_bridge_request` returns `-1`, `timeout_transactions++`, `rejected_transactions++` (never reports fake `OK`!) | G5 | `simulation_backend.cpp:91-124` |
| **L3-EXEC-030** | Epoch mismatch | Mock bridge responds with wrong `transaction_id` (`tx_id + 99`) or unexpected `protocol_version` | `send_bridge_request` rejects (`id_mismatch_errors++`) leaving active state untouched | G5 | `simulation_backend.cpp:133-143` |
| **L3-EXEC-031** | Decoder busy | Lock emulated `status_reg` bit 0 (`commit_pending=1`) in `wrd_handle_t` | `wrd_commit_and_wait(..., 100)` returns `-1` cleanly | G5 | `weight_reg_driver.c:123-132` |
| **L3-EXEC-032** | Daemon restart | Terminate (`SIGKILL` / `SIGTERM`) mid-run -> relaunch clean `avid_daemon` on clean socket/files | Relaunch succeeds cleanly with zero stale socket locks or corrupted binary load | G5 | `avid_daemon.cpp` |
| **L3-EXEC-033** | Memory pressure | Run `avid_daemon` under `resource.setrlimit(RLIMIT_AS, ...)` / tight heap cap | Succeeds cleanly within low static footprint (~few MB) or exits gracefully without silent heap corruption | G5 | `ring_buffer.h`, `lbfgs_optimizer.h` |
| **L3-EXEC-034** | Repeated safety trigger | Trigger `avid_fallback_step` with failures for `100` consecutive calls | FSM stabilizes cleanly in `AVID_FSM_SAFE_STATIC_PRIOR` (`state==2`) with bounded constant static weights (`0x0468`) | G5 | `fallback_fsm.cpp:45-50` |
| **L3-EXEC-035** | Recovery failure | Keep `bounds_ok=false` or `bus_ok=false` while trying to recover | FSM remains strictly locked in `SAFE_STATIC_PRIOR` or `DEGRADED_BYPASS` (never returns `NORMAL_ADAPTIVE`) | G5 | `fallback_fsm.cpp:25-50` |

#### 4D. Concurrency and Performance (`test_concurrency.py`)

| ID | Test | Benchmark & Statistical Distribution Requirements (`p50, p95, p99, max`) | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-EXEC-036** | Ingress throughput | Stream `1,000,000` 64-bit packets through `avid_dma_ring_push_burst` + `consume_slot` in C++ harness; report Mpkts/sec vs `1 MHz` requirement | G5 | `dma_ingress_tap.cpp` |
| **L3-EXEC-037** | Ingress latency | Sample per-64-packet-burst consume time (`std::chrono::steady_clock` across `5,000` bursts) -> report `p50, p95, p99, max` us | G5 | `dma_ingress_tap.cpp` |
| **L3-EXEC-038** | Estimator latency | Benchmark `200` dual-basis `fit_histogram` + R2D2 calls in C++ (`clock_gettime` + `steady_clock`) -> report `p50, p95, p99, max` us (x86_64 host + comparison vs `system_timing.yaml` 10 ms budget) | G5 | `lbfgs_solver.cpp`, `stage_profiler.cpp` |
| **L3-EXEC-039** | Scheduler latency | Benchmark `10,000` calls to `avid_scheduler_poll_commit_ready` & `avid_fast_detector_step` -> report `p50, p95, p99, max` ns/us | G5 | `causal_scheduler.cpp` |
| **L3-EXEC-040** | Commit latency | Measure round-trip latency for `avid_hal_write_weights_and_commit` over `physical` (emulated MMIO) AND `simulation` Unix socket bridge -> report `p50, p95, p99, max` us | G5 | `physical_backend.cpp`, `simulation_backend.cpp` |
| **L3-EXEC-041** | Concurrent workload | Spawn 3 C++ threads: ingress producer, mutex-guarded estimator solver, and telemetry/MMIO reader running `5` seconds -> verify zero data races (`-fsanitize=thread` or deterministic invariants) | G5 | C++ harness + `dma_ingress_tap`, `lbfgs_solver`, `wrd` |
| **L3-EXEC-042** | CPU utilization | Extract `slow_loop_cpu_utilization_10ms_pct` from `avid_daemon` JSON + measure OS `/usr/bin/time` user+sys CPU across `50,000` rounds | G5 | `avid_daemon.cpp:812,849` |
| **L3-EXEC-043** | Memory utilization | Measure `ru_maxrss` across short (`1,024`) vs long (`100,000`) round runs + verify `AvidR2D2LSTMController::get_memory_audit()` static sizes (`total_params=19744, lbfgs_workspace=2688 B`) | G5 | `r2d2_lstm_controller.h:299-311` |
| **L3-EXEC-044** | Priority inversion | Lock/contention test: run heavy L-BFGS estimation loop on low-priority thread while timing high-rate `CycleAccurateBPEngine::decode` calls on fast thread -> verify `p99` decode latency unimpacted | G5 | Fast vs slow isolation harness |
| **L3-EXEC-045** | Timing overrun | Configure `avid_profiler_init(&p, "est", 1.0)` (`1 us` artificially tight deadline!) -> run real `fit_histogram` -> prove `avid_profiler_end` returns `false`, `deadline_misses > 0`, feeding `deadline_ok=false -> HOLD_LAST_GOOD`! | G5 | `stage_profiler.cpp:40-43`, `avid_daemon.cpp:679` |


---

### Set 5 — Layer-Specific Mathematical and Control Validation (`tests/layer3/set5_control_mathematics/`) — Gates **G6, G7, G8, G9**

> **Independent Reference Rule (`reference/layer3_reference_oracle.py`):** Implemented strictly from first-principles binary syndrome algebra (`2^21` tractable brute-force subsets / FWHT / exact enumeration on `k <= 12` active mechanisms), high-precision log cross-entropy + L2 prior, central finite-difference Jacobians/Hessians, IEEE-754 `log((1-p)/p)`, and symbolic FSM/scheduler transition tables — **zero shared C++ estimator logic!**

#### 5A. Joint-DP Likelihood Mathematics (`test_joint_dp_reference.py`) — Gate **G6**

| ID | Property | Test Procedure & Independent Reference Comparison | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-MATH-001** | Probability normalization | Across 20 random valid `theta_[0..20]` vectors (X-checks & Z-checks): assert `\sum_{s=0}^{15} P_joint(s | theta) == 1.0` to `< 1e-14` | G6 | `noise_model.cpp:91-109` |
| **L3-MATH-002** | Nonnegative probabilities | Assert every `P_joint(s | theta) >= 0.0` even at boundary parameters `1e-6` and `0.49` | G6 | `noise_model.cpp:91-109` |
| **L3-MATH-003** | Zero-event case | Set `theta = [1e-6]*21` (and test exact `0.0` in pure reference) -> `P(0)` ~ `1.0`, sum of `s > 0` matches exact `21 * 1e-6` first-order expansion | G6 | `noise_model.cpp:98-108` |
| **L3-MATH-004** | Certain-event case | Set single edge `e` (with `g_e != 0`) to upper clamped bound `0.49` (vs `p_e -> 1.0` XOR swap property `P'(s) = P(s ^ g_e)` in convolution algebra) | G6 | `noise_model.cpp:102-104` |
| **L3-MATH-005** | Single-mechanism likelihood | Turn on ONLY one mechanism `e` at `p_e = 0.08`, others `~0` -> verify `P(0) = 1 - p_e` and `P(g_e) = p_e` (`< 1e-12` modulo `1e-6` clamp) | G6 | `noise_model.cpp:96-108` |
| **L3-MATH-006** | Multi-mechanism likelihood | Activate `k=8` distinct non-zero mechanisms (remaining `13` at `0` / `1e-6`): brute-force enumerate all `2^8 = 256` fault subsets in Python -> assert max error `< 1e-12` against `NoiseModel::compute_joint_distribution()` across both X and Z bases! *(Note `1e-6` floor shifts inactive channels by `~13e-6`, accounted for explicitly or tested with floor matched!)* | G6 | `noise_model.cpp:91-109` |
| **L3-MATH-007** | Syndrome XOR convolution | Check exact Walsh-Hadamard identity `E[(-1)^{<u, s>}] = \prod_e (1 - 2 p_e)^{\mathbb{I}[<u, g_e> = 1]}` across all 16 masks `u in 0..15` (`< 1e-13`) | G6 | `noise_model.cpp:103,248-262` |
| **L3-MATH-008** | Joint parameter model | Verify each `theta[e]` (`0..20`) strictly matches `parameter_schema.json` stabilizer XOR check signature `H_X * e_Z` & `H_Z * e_X` (`g_{i,j} == g_i ^ g_j` on all 12 NN pairs!) | G6 | `noise_model.cpp:31-47`, `parameter_schema.json` |
| **L3-MATH-009** | Negative log-likelihood | Compare `compute_nll_from_histogram` (with `prior_lambda_=0`) against reference empirical cross-entropy `-\sum_s (hist[s]/N) \log P_ref(s|theta)` (`< 1e-12`) | G6 | `noise_model.cpp:138-153` |
| **L3-MATH-010** | MAP regularization | Compare `nll(lambda=0.5) - nll(lambda=0.0)` against `0.5 * 0.5 * ||theta - prior_theta||^2` (`< 1e-13`), checking both `lambda=0.04` solver default and `0.5` schema value (`DEF-L3-03/04`) | G6 | `noise_model.cpp:154-157` |
| **L3-MATH-011** | Gradient calculation | Compare `compute_gradient_from_histogram` coset-inversion derivative against independent Python autodiff / central finite differences (`eps=1e-7`) across all 21 coordinates (`< 1e-7`) | G6 | `noise_model.cpp:201-240` |
| **L3-MATH-012** | Hessian diagnostic | Compute exact `21 x 21` finite-difference Hessian of NLL around MAP optimum & compare eigenvalues against Fisher Information `I(theta) + lambda * I_{21}` | G6 | `noise_model.cpp` + Set 5 oracle |

#### 5B. L-BFGS Optimization Mathematics (`test_lbfgs_reference.py`) — Gate **G6**

| ID | Property | Test Procedure (Tests BOTH `LBFGSOptimizer` two-loop & `HistogramLBFGSSolver` projected GD) | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L3-MATH-013** | Gradient direction | Verify two-loop recursion `d_k` satisfies `g^T d_k < 0` & projected steepest descent `-g_proj^T g < 0` on non-stationary interior states | G6 | `lbfgs_optimizer.cpp:32-60`, `lbfgs_solver.cpp:20-58` |
| **L3-MATH-014** | Line search | Verify Armijo sufficient decrease condition `f(x_new) <= f(x_0) + 1e-4 * <g, x_new - x_0>` holds at every accepted step | G6 | `lbfgs_optimizer.cpp:65-83`, `lbfgs_solver.cpp:56-70` |
| **L3-MATH-015** | Bound constraints | Start near boundaries `0.00002` & `0.449` with adversarial histogram -> assert every intermediate and final iterate remains strictly inside `[1e-5, 0.45]` | G6 | `noise_model.cpp:76-80` |
| **L3-MATH-016** | Convergence criterion | Check early stop at `||g_proj||_inf < 1e-5` (or `||g||_2 < 1e-6`) vs non-convergent stop when `max_iters` exhausted; audit `0.25` threshold in `diag.converged` *(DEF-L3-05)* | G6 | `lbfgs_solver.cpp:48,177`, `lbfgs_optimizer.cpp:99` |
| **L3-MATH-017** | Maximum iterations | Force tiny `max_iters in {1, 4, 8}` on far init -> assert executed iterations bounded exactly `<= max_iters` | G6 | `lbfgs_solver.cpp:109-176`, `lbfgs_optimizer.cpp:92` |
| **L3-MATH-018** | Initial-point sensitivity | Launch optimization from 5 corners `[1e-4, 0.005, 0.012, 0.10, 0.30]` on exact DP multinomial expectation histogram -> verify identical stationary value on identifiable orbits | G6 | `lbfgs_solver.cpp` |
| **L3-MATH-019** | Ill-conditioned model | Construct flat/high-condition problem (`prior_lambda_=1e-8`, near-collinear aliased bins) -> verify solver remains finite, tracks conditioning & doesn't claim false accuracy when unconstrained | G6 | `lbfgs_solver.cpp`, `lbfgs_optimizer.cpp` |
| **L3-MATH-020** | Flat likelihood | Drive `Z_3 Z_6` (`e=16, g=0`) & `Z_2 Z_5` (`e=19, g=0`) -> verify gradient wrt `NLL_data` is exact `0.0`, estimate strictly stays at `prior_theta`, variance stays prior-dominated | G6 | `noise_model.cpp:223-226` |
| **L3-MATH-021** | Synthetic recovery | Feed exact noiseless infinite-sample expectation histogram `round(100000 * P(s | theta_true))` on non-aliased hotspots (`q1, q4, q7`) -> recover `< 0.002` error on identifiable params & `< 1e-3` on `orbit_9class_rmse` | G6 | `lbfgs_solver.cpp`, `noise_model.cpp` |
| **L3-MATH-022** | Model mismatch | Feed uniform empirical histogram `[100]*16` (maximally entropic, violates low-p assumption) -> expose large KS/KL divergence & hitting box cap rather than low confidence | G6 | `lbfgs_solver.cpp` |
| **L3-MATH-023** | Prior sensitivity | Sweep `prior_lambda_` across `{0.0, 0.04, 0.5, 5.0}` keeping histogram constant -> assert monotone decrease in `||theta_hat - prior_theta||_2` | G6 | `noise_model.cpp:154-157,237` |
| **L3-MATH-024** | DP/reference consistency | Compare objective & solution found by C++ `HistogramLBFGSSolver` / `LBFGSOptimizer` against `scipy.optimize.minimize(..., method="L-BFGS-B")` on independent Python DP oracle | G6 | Cross-validation against SciPy L-BFGS-B |

#### 5C. Identifiability and Uncertainty (`test_identifiability.py`) — Gate **G7**

> Emits detailed JSON diagnostics (effective rank, singular values `s_1..s_21`, condition number, Fisher eigenvalues, alias pairs, weak parameter indices).

| ID | Property | Exact Algebraic & SVD Oracle Analysis | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-MATH-025** | Jacobian rank | Compute `16 x 21` sensitivity Jacobian `J_{s,e} = d P(s | theta) / d theta_e` at baseline `theta`: verify **exact theoretical rank <= 9** (at most the 9 non-zero quotient orbits in `parameter_schema.json`!) | G7 | `parameter_schema.json:235-583`, `noise_model.cpp` |
| **L3-MATH-026** | Fisher information | Compute multinomial Fisher matrix `I(theta)_{a,b} = \sum_s (1/P(s)) (dP(s)/d\theta_a)(dP(s)/d\theta_b)` (per shot & scaled by `W`) -> inspect full eigenspectrum (`lambda_1..lambda_21`) | G7 | `noise_model.cpp` |
| **L3-MATH-027** | Parameter correlation | Extract regularized posterior covariance `(N * I(theta) + lambda * I_21)^{-1}` -> identify strongly anti-correlated parameter pairs within shared orbits `(Z_0, ZZ_1_2)`, `(Z_2, Z_5, ZZ_0_1, ZZ_3_4)`, `(Z_3, Z_6, ZZ_4_5, ZZ_7_8)` | G7 | `parameter_schema.json:270-583` |
| **L3-MATH-028** | Non-identifiable parameters | Confirm strictly `12` zero-eigenvalue / null-space directions of unregularized `I(theta)`, specifically including exact zero sensitivity on `ZZ_3_6` (`e=16`) and `ZZ_2_5` (`e=19`) in X-checks (`ZZ_0_1, ZZ_7_8` in Z-checks) | G7 | `noise_model.cpp:32-47` |
| **L3-MATH-029** | Sample-size dependence | Evaluate Cramer-Rao lower bound `sqrt((N*I + lambda*I)^{-1}_{ii})` across `N in {256, 1000, 10000}` -> show identifiable `Z_1, Z_4, Z_7` shrink as `1/sqrt(N)` while null params stay capped by `1/sqrt(lambda)` | G7 | Prior vs sample scaling (`DEF-L3-03`) |
| **L3-MATH-030** | Prior-dominated estimates | Measure ratio of Fisher sample information to prior stiffness `lambda` for `(3,6)` & `(2,5)` -> confirm `~100%` prior-dominated (`ratio == 0.0`) | G7 | `noise_model.cpp:154-157` |
| **L3-MATH-031** | Confidence calibration | Verify empirical Monte Carlo SD across 30 trials matches asymptotic inverse-Hessian SE within statistical factor on singleton orbits `{Z_1, Z_4, Z_7}` | G7 | Estimator calibration report |
| **L3-MATH-032** | Alias construction | Construct two distinct parameter vectors `theta^(A)` (high `Z_0`, low `ZZ_1_2`) vs `theta^(B)` (low `Z_0`, high `ZZ_1_2`) satisfying exact equal `(1-2p_a)(1-2p_b)` on orbit `0x1`: prove exact identical `P_joint(0..15)` (`< 1e-14`)! | G7 | `parameter_schema.json:270-301` |

#### 5D. LLR Conversion and Fixed-Point Mathematics (`test_q78_reference.py`) — Gate **G8**

| ID | Property | Independent Q7.8 Math Verification | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-MATH-033** | LLR reference | Sweep 100 `p in [1e-5, 0.49]` -> compare `avid_prob_to_q78_llr` vs Python `round(math.log((1-p)/p) * 256)` (`diff == 0`) | G8 | `llr_quantizer.cpp:8-15` |
| **L3-MATH-034** | Sign | Confirm `p < 0.5 => LLR > 0` (`q78 > 0`, `0`-bit preferred), `p > 0.5 => LLR < 0` in direct log equation (`clamped to 0.49` inside `avid_prob_to_q78_llr`) | G8 | `llr_numeric_contract.yaml:8` |
| **L3-MATH-035** | Scaling | Verify exact `2^8 = 256.0` scale factor (`LLR = 1.0 => 256`, `2.5 => 640`, `4.0 => 1024`) | G8 | `llr_numeric_contract.yaml:25`, `llr_quantizer.cpp:11` |
| **L3-MATH-036** | Rounding | Test exact `.5` midpoints (`round(x * 256)`) in `llr_quantizer.cpp` (`std::lround`) vs `wrd_float_to_q78` truncation (`DEF-L3-06`) | G8 | `llr_quantizer.cpp:11`, `weight_reg_driver.h:30` |
| **L3-MATH-037** | Saturation | Assert signed 16-bit bounds `[-32768, 32767]` in `avid_prob_to_q78_llr` & `[0x0040, 0x3000]` safe bounds in `avid_quantize_and_validate_bank` | G8 | `llr_quantizer.h:13-15`, `llr_quantizer.cpp:12-30` |
| **L3-MATH-038** | Dequantization | Reconstruct `llr_deq = q78 / 256.0`, `p_deq = 1 / (1 + exp(llr_deq))` | G8 | `llr_quantizer.cpp` |
| **L3-MATH-039** | Error bound | Over unsaturated range `0x0040..0x3000`, assert `|q78/256.0 - log((1-p)/p)| <= 1/512 + 1e-12` everywhere | G8 | `llr_quantizer.cpp:11` |
| **L3-MATH-040** | Extreme probabilities | Evaluate `p in {0, -1, 1e-12, 1e-5, 0.49, 0.5, 1.0, NaN, Inf}` cleanly across `avid_prob_to_q78_llr` & `avid_quantize_and_validate_bank` | G8 | `llr_quantizer.cpp:9,21-25` |
| **L3-MATH-041** | Weight ordering | Set distinct probabilities per qubit `q=0..8` -> verify `q78[0..8]` order matches `effective_p(0..8)` & `NM_CHECK_CONN` qubit indices `0..8` | G8 | `noise_model.cpp:281-311`, `weight_reg_driver.h` |
| **L3-MATH-042** | Decoder compatibility | Write quantized `weights_q78[0..8]` through `wrd_write_all_weights -> wrd_commit` -> decode single-qubit syndromes with `CycleAccurateBPEngine` verifying Q7.8 sign-magnitude/two's-complement bit interpretation matches RTL `weight_map_reg.sv` contract | G8 | `cycle_accurate_bp_engine.cpp:58-72,148-154`, `weight_reg_driver.c` |

#### 5E. Causal Scheduler Mathematics (`test_causal_scheduler.py`) — Gate **G8**

| ID | Property | Mathematical State-Machine Verification | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-MATH-043** | Evidence timestamp | Record exact `window_end_round` passed to `avid_scheduler_schedule_commit` & burst trigger `round_idx` -> match actual window end | G8 | `causal_scheduler.cpp:17-23` |
| **L3-MATH-044** | Commit eligibility | Across 50 random `(t_e, d_c)` pairs: verify `poll_commit_ready(sched, round_now)` is `false` for EVERY `round_now < t_e + d_c` and `true` at `t_c == t_e + d_c` | G8 | `causal_scheduler.cpp:25-36` |
| **L3-MATH-045** | Epoch ordering | Execute 20 scheduled commits -> verify `active_epoch_id` forms strict sequence `0, 1, 2, ..., 20` (and `hw_epoch` strictly monotonic except explicit RTL reset) | G8 | `causal_scheduler.cpp:32` |
| **L3-MATH-046** | Stale update rejection | Attempt to schedule commit with `window_end < current_round` / old epoch when newer commit is already active or pending -> guarded scheduler rejects stale overwrite *(DEF-L3-08)* | G8 | `causal_scheduler.cpp` + guard |
| **L3-MATH-047** | Duplicate commit prevention | Poll repeatedly at `t_c, t_c+1, ...` after single `schedule_commit` -> second poll returns `false`, firing commit strictly once; sustained burst does not re-trigger `trigger_count` | G8 | `causal_scheduler.cpp:31,66-69` |
| **L3-MATH-048** | Busy-state handling | Emulate `bp_ready_in == 0` -> confirm `wrd_commit_and_wait` defers / times out cleanly instead of dropping active bank integrity | G8 | `weight_reg_driver.c:123-132`, `weight_commit_protocol.yaml` |
| **L3-MATH-049** | Reset handling | Call `avid_scheduler_init` (and `avid_fast_detector_init`) mid-pending commit -> verify `has_pending_commit=false`, `active_epoch_id=0`, ring/debounce cleared | G8 | `causal_scheduler.cpp:7-15,38-46` |
| **L3-MATH-050** | Timestamp causality | Prefix-equivalence oracle: feed common prefix `r=0..499` followed by divergent suffixes `A` vs `B` (`r=500..999`) -> prove ALL decisions for `r <= 499` are bit-identical (`zero lookahead leakage`) | G8 | `causal_scheduler.cpp`, `avid_daemon.cpp` |
| **L3-MATH-051** | Burst detection delay | Inject burst at `r=B_0` with densities `0.38..1.0`: compute analytical trigger step from `16`-round sliding sum `>= ceil(0.35 * 16) = 6` plus `4`-round debounce -> match C++ `avid_fast_detector_step` exact round! | G8 | `causal_scheduler.cpp:48-76` |
| **L3-MATH-052** | Slow-loop limitation | Formally contrast 10 ms window (`N_a = 10,000`, dilution `60/10000 = 0.6%`) vs fast 16-round detector (`N_b = 60`, `20 us` max trigger): assert `L3-FLOW-034` only credits fast detector for `< 60 us` detection! | G8 | `system_timing.yaml:22-37` |

#### 5F. Four-State Safety FSM Mathematics (`test_fallback_fsm.py`) — Gate **G9**

| ID | Property | Exhaustive Formal Verification over `4 states x 16 input events` | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L3-MATH-053** | State completeness | Evaluate every `(state in {0..3}) x (est_ok, deadline_ok, bounds_ok, bus_ok in {0,1}^4)` (`64` combinations) — every pair returns defined valid state `0..3` | G9 | `fallback_fsm.h:9-14`, `fallback_fsm.cpp` |
| **L3-MATH-054** | Transition validity | Compare all `64` observed C++ transitions against exact mathematical transition relation extracted from `fallback_fsm.cpp` | G9 | `fallback_fsm.cpp:25-56` |
| **L3-MATH-055** | Safety invariant | Whenever `bounds_ok == false` OR `estimator_ok == false` OR `deadline_ok == false` OR `bus_ok == false`, verify candidate unsafe `cand_q78` NEVER appears in `out_selected_q78` with `should_commit == true`! | G9 | `fallback_fsm.cpp:25-55` |
| **L3-MATH-056** | Fallback invariant | In `SAFE_STATIC_PRIOR(2)` & `DEGRADED_BYPASS(3)`, `out_selected_q78` ALWAYS equals `static_prior_q78`; in `HOLD_LAST_GOOD(1)`, `out_selected_q78` equals `last_good_q78` & `out_should_commit == false` | G9 | `fallback_fsm.cpp:28,47,53` |
| **L3-MATH-057** | Recovery condition | Prove transition back to `NORMAL_ADAPTIVE` happens ONLY when `estimator_ok && deadline_ok && bounds_ok && bus_ok == true` *(explicitly documenting `1`-step code behavior vs `fault_policy.yaml` 3-step spec, `DEF-L3-07`)* | G9 | `fallback_fsm.cpp:33-40` |
| **L3-MATH-058** | Invalid event | Pass uninitialized/out-of-range integer state (`fsm.state = 99`) or null candidate pointer under faulted input -> safe guarded behavior without committing unvalidated weights *(DEF-L3-10)* | G9 | `fallback_fsm.cpp` |
| **L3-MATH-059** | Repeated fault | Drive `HOLD_LAST_GOOD` -> `3x` faults -> `SAFE_STATIC_PRIOR` -> `50x` more faults -> verify `consecutive_holds` never overflows to trigger wrap bug & stays safe | G9 | `fallback_fsm.cpp:42-50` |
| **L3-MATH-060** | Reset invariant | Call `avid_fallback_init(&fsm, AVID_Q78_DEFAULT)` from any degraded state -> verifies `state == NORMAL_ADAPTIVE`, `consecutive_holds == 0`, `fallback_triggers == 0`, banks equal `default_q78` | G9 | `fallback_fsm.cpp:7-15` |
| **L3-MATH-061** | Transition coverage | Construct explicit witness sequences visiting EVERY legal edge in the `4`-state graph (`NORMAL->HOLD`, `HOLD->HOLD`, `HOLD->SAFE_STATIC`, `NORMAL->SAFE_STATIC(!bounds)`, `ANY->DEGRADED(!bus)`, `ANY->NORMAL`) (`100%` edge coverage) | G9 | `fallback_fsm.cpp` |
| **L3-MATH-062** | Forbidden transition coverage | Exhaustively prove over all `(state, inputs)` that forbidden transitions never occur (e.g., `!bus_ok` cannot enter `NORMAL`/`HOLD`/`SAFE_STATIC`; `HOLD` cannot set `should_commit=true`; `!bounds_ok` cannot enter `HOLD`) | G9 | `fallback_fsm.cpp` |

---

## 5. Planned Layer 3 Execution & Implementation Sequence

Following Specification Section 7 & Section 9 strictly on branch `layer3-slow-loop-software`:
1. **[Done]** Produce [`LAYER3_IMPLEMENTATION_TO_TEST_TRACEABILITY.md`](LAYER3_IMPLEMENTATION_TO_TEST_TRACEABILITY.md) covering all **243 IDs**.
2. **Next — Set 1 (`tests/layer3/set1_input_contracts/` + C ABI shim + input guards + `scripts/run_layer3_set1.sh`)**: implement `L3-IP-001..040`, verifying packet ingress (`packet_validator`, `dma_ingress_tap`), histograms, `NoiseModel` 21-param boundaries & NaN guards (`DEF-L3-10`), and config contracts.
3. **Set 5 (`tests/layer3/set5_control_mathematics/` + `reference/layer3_reference_oracle.py` + `scripts/run_layer3_set5.sh`)**: implement independent mathematical oracles for `L3-MATH-001..062` (brute-force Joint-DP, finite-diff grads, SVD/Fisher identifiability report, Q7.8 reference, causal scheduler, exhaustive 4-state FSM).
4. **Set 2 (`tests/layer3/set2_output_statistics/` + `scripts/run_layer3_set2.sh`)**: `L3-OP-001..036` + 30-seed `L3-STAT-001..015`.
5. **Set 3 (`tests/layer3/set3_pipeline_flow/` + `scripts/run_layer3_set3.sh`)**: `L3-FLOW-001..045` starting with Section 9 priority trace `L3-FLOW-001..006`, end-to-end daemon flows, and `20 us` vs `10 ms` causal timing (`L3-FLOW-026..035`).
6. **Set 4 (`tests/layer3/set4_execution_recovery/` + `scripts/run_layer3_set4.sh`)**: `L3-EXEC-001..045` (clean build, daemon execution, persistence audit, mock Unix bridge timeout/mismatch/busy fault injection, throughput/latency distributions).
7. **Acceptance (`tests/layer3/acceptance/` + `scripts/run_layer3_validation.sh`)**: Gate `G1`–`G10` integrated acceptance harness + `requirements_traceability.yaml`.
