# Layer 2 — R2D2 ML Training & C++ Runtime Deployment: Implementation-to-Test Traceability Specification (`LAYER2_IMPLEMENTATION_TO_TEST_TRACEABILITY.md`)

**Branch Scope:** `layer2-r2d2-training-deploy` (isolated Layer 2 branch split from frozen baseline `AVID_EPJ_Patches` @ `54da68d`)  
**Frozen Baseline Constraint:** `AVID_EPJ_Patches` (`54da68d`) MUST remain completely unmodified throughout Layer 2 validation and remediation. All Layer 2 test suites, reference oracles, contract fixes, and acceptance scripts execute strictly on `layer2-r2d2-training-deploy`.

---

## 1. Empirical Source-Code Inspection (Section 7 Answers — Verified Directly from Repo)

Every question posed in Section 7 of the Layer 2 Validation Specification was resolved by direct inspection of `training/`, `deployment/`, `checkpoints/`, and `tests/` on `layer2-r2d2-training-deploy`:

| # | Inspection Question | Verified Reality in `AVID_EPJ_Patches` Layer 2 Source |
|---|---|---|
| **Q1** | Recurrent sequence replay vs plain transition replay? | **True overlapping sequence replay** ([`SequenceSegment`](training/replay/sequence_sampler.py), `slice_episode_into_segments` with `burn_in_steps=4`, `unroll_steps=4`, `n_step_td=3`, `T_seq=11`, `stride=2`), storing per-segment initial hidden/cell vectors `h_init[64], c_init[64]`. |
| **Q2** | Prioritized replay, burn-in, and multi-step returns? | **All three are implemented in pure NumPy**: [`PrioritizedSequenceReplayBuffer`](training/replay/prioritized_replay.py) uses Kapturowski Eq. 2 `p_i = 0.9 max|delta| + 0.1 mean|delta|`, `alpha=0.6`, IS weights `w_i`; `RecurrentLSTMEncoder` executes stop-gradient burn-in over `[0:4]` and analytical BPTT only over `[4:11]`; [`compute_n_step_transformed_td_loss`](training/algorithms/loss.py) computes `n=3` Double-DQN returns with `gamma=0.96` + Kapturowski invertible `value_rescale_h` (`eps=0.001`), **plus** DQfD supervised large-margin loss (`0.35`) and MSE gain supervision (`0.50`). |
| **Q3** | Is the 64-unit LSTM shared across value, advantage, and gain heads? | **Yes.** [`R2D2Network`](training/models/r2d2_network.py) unrolls a single-layer 64-unit [`RecurrentLSTMEncoder`](training/models/recurrent_encoder.py) (`input_dim=8`, `hidden_dim=64`) and feeds `h_next` jointly into [`DuelingQValueHead`](training/models/q_value_head.py) (`W_val[1,64]`, `W_adv[6,64]`, `W_gain[9,64]`). |
| **Q4** | Are the 9 gain residuals a separate head or action-selection? | **Shared hidden state, dedicated linear+tanh projection** inside `DuelingQValueHead`: `delta_g = 0.30 * tanh(clip(W_gain @ h_t + b_gain, -15, 15))` in `[-0.30, +0.30]^9`, independent of discrete argmax `a*`. |
| **Q5** | Is the ONNX exporter executable in deployment? | **No (Serializer only, `DEF-L2-02`).** `export_minimal_onnx_proto` emits a custom protobuf bytes file (`r2d2_v1_trained.onnx`, `DOUBLE` weights) lacking `ValueInfoProto` inputs/outputs, `transB=1` on `Gemm`, `Clip`, and `0.30` scaling on `GainTanh`. `deployment/cpp/` ONLY loads `.bin`, and `onnxruntime` is absent from `dependencies/`. |
| **Q6** | Does INT8 quantize recurrent state and dense layers? | **No INT8 or FP32 implementation exists (`DEF-L2-03`).** `export_checkpoint.py`, `model_loader.cpp`, and `r2d2_runtime.cpp` operate strictly in **IEEE-754 `float64` (FP64)** (`157,984` byte `.bin` checkpoint matching `r2d2_schema.json`). `checkpoints/README.md` mentions INT8/FP32 sizes descriptively only. |
| **Q7** | Do Python and C++ share LSTM gate ordering & reset semantics? | **Yes.** Both [`RecurrentLSTMEncoder`](training/models/recurrent_encoder.py) and [`R2D2Runtime`](deployment/cpp/r2d2_runtime.cpp) use IEEE-754 `float64`, identical `[i, f, g, o]` (`4x64`) gate ordering, identical clamps (`[-4,4]`, `[-30,30]`, `[-15,15]`), and zero-filled `reset_state()`, achieving `< 1e-12` parity. |
| **Q8** | Is the deployed action applied to an actual decoder control or only returned? | **Inside Python training (`r2d2_trainer.py`), YES:** actions (`pair_weight_scale, bp_iterations, fallback`) + `qubit_gains` drive `project_q78_weights_d3 -> build_hardware_osd_lut_z -> count_logical_z_failures` on real `d=3` `err_z_win, sx_win` windows. **Inside C++ Layer 2 (`r2d2_runtime.cpp`), NO standalone decoder exists yet (`DEF-L2-07`):** `R2D2Runtime::step()` returns `R2D2InferenceOutput` to external software (Layer 3 `avid_daemon`); `evaluate_policy.py` also evaluates rewards via lookup heuristic rather than actual decoded `LER`. |

---

## 2. Verified Implementation Discrepancies & Gaps (`DEF-L2-01`..`DEF-L2-10`)

| Defect ID | Location | Verified Gap / Discrepancy | Planned Resolution / Gate Handling |
|---|---|---|---|
| **DEF-L2-01** | `q_value_head.py:12-18` vs `r2d2_network.py:40-47`, `r2d2_v1_config.json`, `r2d2_runtime.cpp:30-37` | Docstring in `q_value_head.py` cites `K4/K6/K8`, while runtime code & config enforce `K in {3,4,5}` (`STATIC_PRIOR_K3`..`CONSERVATIVE_FALLBACK_K3`). | Align docstring; assert canonical `K in {3,4,5}` & params against config in Set 1/4. |
| **DEF-L2-02** | `training/export/export_checkpoint.py:86-135` | `export_minimal_onnx_proto` emits an incomplete ONNX protobuf (`DOUBLE` tensors, missing `ValueInfoProto` I/O, missing `transB=1` & `0.30` scale), and C++ never executes ONNX. | Explicitly mark executable ONNX deployment `INCONCLUSIVE` / unsupported until valid ORT graph + C++ consumer exist; validate `.onnx` protobuf integrity separately. |
| **DEF-L2-03** | `checkpoints/README.md` vs `export_checkpoint.py`, `r2d2_runtime.cpp` | Spec & README mention FP32 & INT8; actual implementation is strictly **FP64 (`float64`)** only. | Per Spec Section 4B & Section 6, FP64 is primary validated precision; INT8/FP32 report explicit `INCONCLUSIVE` ("unsupported precision mode", never coerced to `PASS`). |
| **DEF-L2-04** | `r2d2_network.py` & `r2d2_runtime.h` vs Section 1 IP table | Network input is an **`8`-D feature vector** `[mean_p, max_p, mean_alpha, max_alpha, syn_dens, adj_rate, innov, burst_score]` matching `r2d2_v1_config.json`, NOT raw `hist_x[16], hist_z[16]`. | Validate actual 8-D feature contract (`L2-IP-001..010`); build explicit feature-extractor adapter tests where raw syndrome stats enter `generate_qec_control_episode`. |
| **DEF-L2-05** | `training/algorithms/loss.py:113-134` | R2D2 update combines Huber TD loss with DQfD large-margin classification (`0.35`, using `optimal_actions`) + MSE gain supervision (`0.50`, using `target_gains`), plus reward shaping including `opt_action`. | Test exact implemented loss & pure-TD vs auxiliary gradient attribution in Set 5 (`L2-MATH-018,019,031..035`). |
| **DEF-L2-06** | `recurrent_encoder.py`, `r2d2_runtime.cpp:72-86`, `prioritized_replay.py`, `sequence_sampler.py` | Zero input-boundary validation: Python allows NaN/Inf, wrong feature dims, `action not in 0..5`, `lr <= 0`, `capacity <= 0`, bad `gamma`; C++ `step()` silently returns default `action_id=0` when unloaded (`!loaded_from_checkpoint`) and clamps NaN instead of failing! | **Remediate in Set 1 (`L2-IP-*`) & Set 4 (`L2-DEP-039,045`)**: add explicit boundary validators / error checks on `layer2-r2d2-training-deploy`. |
| **DEF-L2-07** | `r2d2_runtime.cpp` & `evaluate_policy.py` | (a) C++ runtime returns `R2D2InferenceOutput` without a C++ decoder control harness inside Layer 2 (`Set 4E`); (b) `evaluate_policy_on_split()` scores heuristic step reward without running `project_q78_weights_d3` -> `count_logical_z_failures` (`L2-STAT-014` paired `ΔLER`). | Add C++ LUT/Q7.8 decoder harness in `tests/layer2/set4_export_cpp_deployment/cpp/` for Set 4E + paired `ΔLER` evaluation in Set 2C. |
| **DEF-L2-08** | `sequence_sampler.py:53-68` & `loss.py:66-134` | Short-episode padding repeats last transition with `done=1.0`, yet `loss.py` lacks a `valid_mask` ignoring padded timesteps (`[T_ep : T_seq]`). | Add `padding_mask` support & verify zero gradient/loss contribution in Set 1 (`L2-IP-025`) and Set 5 (`L2-MATH-025`). |
| **DEF-L2-09** | `evaluate_policy.py:95-144` vs `training/configs/r2d2_v1_config.json` | `run_full_training_and_export()` hardcodes `lr=0.0028` & hyperparameters rather than loading `r2d2_v1_config.json` (`lr=0.0025`), and lacks CLI args (`--config`, `--seed`, `--training-steps`, `--checkpoint-dir`, `--output-dir`). | Add config-driven CLI training/validation entry point (`training/run_layer2_pipeline.py`) enforcing JSON schema in Set 3/4. |
| **DEF-L2-10** | `r2d2_trainer.py:535` & `loss.py:82` | Fixed 16-epoch episodes set `done=1.0` at `ep=15` without separating time-limit `truncated` vs genuine `terminated` state (`L2-IP-019`, `L2-MATH-015`). | Document and add explicit `truncated` vs `terminated` distinction support in Set 1/5 tests. |

---

## 3. Layer 2 Acceptance Gates (`G1`–`G9`) & Evaluation Design

- **Statuses:** `PASS` | `FAIL` | `INCONCLUSIVE`. `NOT_RUN`, `SKIPPED`, or unsupported precision modes (`INT8`, executable `ONNX`, `FP32` where unimplemented) MUST emit `INCONCLUSIVE` or `FAIL`, **never** silent `PASS`.
- **Precision Parity Targets:** IEEE-754 `float64` Python-to-C++ runtime parity `< 1e-12` for activations, `(h, c)`, `V`, `A`, `Q`, and `qubit_gains`; exact discrete action equality whenever `Q_best - Q_second > 1e-9`.
- **Statistical & Paired Policy Design:** `>= 5` independent training seeds, `>= 30` held-out evaluation seeds across `stationary`, `pair_correlated`, `drifting`, `abrupt_switch`, `burst_intermittency`, and held-out unseen families (`heldout_*`), computing **paired `d_j = LER_baseline,j - LER_R2D2,j`** and bootstrap/t confidence intervals on identical noise shots (`err_z_win, sx_win`).

| Gate | Scope Covered | Functional vs Scientific/Deployment Pillar |
|---|---|---|
| **G1** | Set 1 (`L2-IP-001..039`) | Input, action/reward, sequence & config contracts valid |
| **G2** | Set 2A (`L2-OP-001..012`) & 2B (`L2-STAT-001..010`) | Network architecture, dueling, action & sensitivity output contracts valid |
| **G3** | Set 3A–3C (`L2-FLOW-001..034`) | Live env interaction, replay & BPTT/Adam optimizer updates verified |
| **G4** | Set 3D–3E (`L2-FLOW-035..050`) | End-to-end training & checkpoint provenance/reload verified |
| **G5** | Set 4A–4D (`L2-DEP-001..050`) | Export integrity, FP64 Python-C++ parity & runtime lifecycle pass |
| **G6** | Set 4E (`L2-DEP-051..060`) | C++ actions/gains verified on decoder Q7.8/LUT control interface |
| **G7** | Set 5 (`L2-MATH-001..050`) | Independent reference checks pass for LSTM, Dueling, TD, Replay & Gains |
| **G8** | Set 2C (`L2-STAT-011..020`) | Multi-seed, regime-stratified, paired held-out `ΔLER` evaluation complete |
| **G9** | Cross-cutting | Machine-readable provenance, artifact hashes & 231-ID traceability complete |

---

## 4. Master Implementation-to-Test Traceability Matrix (All 231 Test IDs)

### Set 1 — Input and Training Contract Validation (`tests/layer2/set1_input_contracts/`) — Gate **G1**

#### 1A. Observation Contracts (`test_observation_contracts.py`)

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Oracle / Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|
| **L2-IP-001** | Observation schema | Initialized `R2D2Network` & `R2D2Runtime` | Valid 8D vector `[mean_p, max_p, mean_alpha, max_alpha, syn_dens, adj_rate, innov, burst]` from `r2d2_v1_config.json` (`DEF-L2-04`) | Call `validate_observation` / `net.step(x, h, c)` & `net.forward_sequence(seq)` | Accepted with finite outputs; dict keys match `feature_spec` | G1 | `r2d2_v1_config.json`, `recurrent_encoder.py`, `r2d2_network.py` |
| **L2-IP-002** | X-history / rate feature dim | Valid & invalid X-channel feature/history vectors | Pass correct `len=8` feature vector & `hist_x[16]` adapter vs malformed `len in {0, 4, 7, 9, 15, 17}` | Pass to observation validator & `step()` / `forward_sequence()` | `len=8` accepted; mismatched X/feature lengths raise `ValueError` *(DEF-L2-06)* | G1 | `recurrent_encoder.py`, `r2d2_network.py` |
| **L2-IP-003** | Z-history / feature dim | Valid & invalid observation matrix cols | Pass `(T, 8)` vs `(T, 7)`, `(T, 9)`, 1D `(8,)` to `forward_sequence` | Call `forward_sequence` & observation validator | Strict `ndim==2, shape[1]==8` for sequence and `shape==(8,)` for single step | G1 | `recurrent_encoder.py:forward_sequence,step` |
| **L2-IP-004** | Density dimensions | Synthetic observation with `syn_dens`, `adj_rate` (`[4],[5]`) | Omit feature elements (`None`, empty), or corrupt shape in adapter/obs | Run obs validator & `sample_d3_z_window` feature builder | Missing/malformed density fields raise explicit `ValueError` | G1 | `r2d2_v1_config.json:feature_spec`, `r2d2_trainer.py:488-499` |
| **L2-IP-005** | Burst indicator | Valid burst feature `[7]` | Test `burst=0.0`, `0.5`, `1.0`, valid CUSUM `[0, 1]`, and invalid `-0.5`, `1.5`, `"on"` | Pass to observation validator & `_oracle_action_and_gains_for_state` | Valid range `[0.0, 1.0]` accepted cleanly; out-of-domain / string rejected *(DEF-L2-06)* | G1 | `r2d2_trainer.py`, `recurrent_encoder.py` |
| **L2-IP-006** | Numeric range | `feature_scale`, `feature_offset` clip `[-4, 4]` | Test min `[0]*8`, nominal `[0.014..0.05]`, physical max `[1.0]*8`, and out-of-range `-0.1`, `1e6` | Test raw-feature validator + `normalize_features()` | Valid physical features accepted & normalized within `[-4.0, 4.0]`; negative probabilities rejected at boundary *(DEF-L2-06)* | G1 | `recurrent_encoder.py:normalize_features` |
| **L2-IP-007** | NaN/Inf handling | Initialized `R2D2Network` & `R2D2Runtime` | Insert `np.nan`, `np.inf`, `-np.inf` at each of indices `0..7` in single & sequence mode | Pass into Python validator, `net.step()`, and C++ `R2D2Runtime::step()` | Rejects non-finite inputs explicitly (raises `ValueError` / flagged C++ error instead of silent clamp!) *(DEF-L2-06)* | G1 | `recurrent_encoder.py`, `r2d2_runtime.cpp` |
| **L2-IP-008** | Dtype validation | Numeric & non-numeric arrays | Test `float64`, `float32`, `int64`, `bool`, `object`/`str` arrays | Pass to `normalize_features`, `step`, `forward_sequence` | `float32`/`float64`/`int` promote cleanly to `float64`; `bool`/`str`/`object` rejected cleanly | G1 | `recurrent_encoder.py` |
| **L2-IP-009** | Batch dimensions | Trained/init `R2D2Network` | Test single-step `(8,)`, seq length `1` `(1,8)`, nominal `(11,8)`, batch of `B in {1, 4, 16}` segments | Evaluate `step()`, `forward_sequence()`, and trainer batch loop | Output shapes match `(8,)->scalar/vectors`, `(T,8)->(T,6)/(T,9)` and `len(batch)==B` | G1 | `r2d2_network.py`, `r2d2_trainer.py:train_step` |
| **L2-IP-010** | Observation ordering | Non-stationary drift vs abrupt sequence `(T=8, 8)` | Forward ordered sequence vs time-reversed / permuted sequence | Compare final `(h_final, q_seq[-1])` between original and permuted histories | Ordered differs strictly from reversed/permuted (`h_next` history-dependent); step-by-step causal loop matches `forward_sequence` | G1 | `recurrent_encoder.py:forward_sequence` |

#### 1B. Action and Reward Contracts (`test_action_reward_contracts.py`)

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Oracle / Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|
| **L2-IP-011** | Discrete action domain | `ACTION_DEFINITIONS` (`0..5`) | Test every integer `a in {0,1,2,3,4,5}` | Call `compute_qec_action_reward(feat, a)` & `SequenceSegment` validation | All 6 actions resolve exact metadata (`pair_weight_scale, bp_iterations in {3,4,5}, temporal_beta, recalibrate, fallback`) | G1 | `r2d2_network.py:40-47`, `r2d2_trainer.py:293` |
| **L2-IP-012** | Invalid action | `ACTION_DEFINITIONS` | Inject `a=-1`, `a=6`, `a=100`, `a=True`, `a=1.5` | Pass into `compute_qec_action_reward`, `SequenceSegment` / loss validator | Explicit `IndexError`/`ValueError` for negative, `>= 6`, bool, or non-integer actions | G1 | `r2d2_trainer.py`, `loss.py` |
| **L2-IP-013** | Gain residual dimensions | `R2D2Network`, `SequenceSegment` | Valid `shape=(9,)` & `(T_seq, 9)` vs wrong `(8,)`, `(10,)`, `(T_seq, 5)` | Validate `target_gains` in segment & `qubit_gains` in `project_q78_weights_d3` | Only 9-element gains (`num_qubits=9`) accepted | G1 | `q_value_head.py`, `sequence_sampler.py`, `r2d2_trainer.py` |
| **L2-IP-014** | Gain residual boundaries | Valid nominal `feat`, `gain_clip=0.30` | Test `target_gains = -0.30`, `+0.30`, `-0.3001`, `+0.35`, `nan` | Run `DuelingQValueHead` + target-gain validator | Exact closed `[-0.30, +0.30]` accepted; out-of-range / NaN rejected explicitly unless explicitly clipped per contract | G1 | `q_value_head.py:64-66`, `r2d2_trainer.py:238` |
| **L2-IP-015** | Reward finiteness | Valid `SequenceSegment` | Inject `np.nan`, `np.inf`, `-np.inf` in `segment.rewards` & check finite `[−5.0, +5.0]` | Pass to `compute_n_step_transformed_td_loss` / replay `add` validator | Non-finite rewards rejected explicitly before poisoning TD loss/priorities | G1 | `loss.py`, `prioritized_replay.py` |
| **L2-IP-016** | Reward sign | Valid synthetic segments | Fixed segments with purely `+2.0`, `0.0`, `-2.0` rewards & terminal done | Run `compute_n_step_transformed_td_loss` | Positive return produces positive `y_target`; negative produces negative `y_target`; zero preserves monotonicity | G1 | `loss.py:71-98` |
| **L2-IP-017** | Reward/action alignment | `T_seq=11` (`burn_in=4, unroll=4, n=3`) | Set reward impulse `+5.0` at exact step `t_imp = burn_in + 2` (`6`) and distinct `actions` | Inspect `td_errors[0..3]` & `dq_seq` signs/indices | Impulse shifts ONLY returns covering `[t, t+1, t+2]`, attributing gradient ONLY to `dq_seq[t, actions[t]]` | G1 | `loss.py:75-110` |
| **L2-IP-018** | Terminal transition | Synthetic segment with non-zero `target_net` Q values | Set `dones[t] = 1.0` vs `dones[t] = 0.0` at step `t` inside n-step window | Compare `td_errors` vs manual formula without bootstrap | When `done=1.0`, `v_boot` is strictly excluded (`0` bootstrap); exact equality with manual discounted reward sum | G1 | `loss.py:82-92` |
| **L2-IP-019** | Truncated transition | Time-limit end vs true failure end | Create `done=1.0, truncated=True` vs `terminated=True` (`DEF-L2-10`) | Evaluate TD target under both flags | Truncated boundary allows target bootstrap `gamma^n V_boot` whereas true terminal masks `V_boot=0` *(remediated in Set 1)* | G1 | `loss.py`, `r2d2_trainer.py:535` |
| **L2-IP-020** | Decoder feedback | `generate_qec_control_episode` | Pass valid `err_z_win`, `sx_win` vs shape-mismatched / non-uint16 / out-of-range syndrome `>= 16` | Call `count_logical_z_failures`, `_oracle_action_and_gains_for_state`, `compute_qec_action_reward` | Valid syndromes map cleanly via `STATIC_LUT_Z`/`build_hardware_osd_lut_z`; malformed syndrome/lengths rejected | G1 | `r2d2_trainer.py:216-289,521-534` |

#### 1C. Sequence and Recurrent-State Contracts (`test_sequence_contracts.py`)

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Oracle / Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|
| **L2-IP-021** | Sequence length | Canonical `burn_in=4, unroll=4, n_step=3` (`T_seq=11`) | Slice episodes of length `1`, `5`, `11`, `16`, `64` via `slice_episode_into_segments` | Verify returned segment tensor shapes & `padding_mask` (`DEF-L2-08`) | Every segment has exact `T_seq=11`; invalid negative/zero lengths rejected | G1 | `sequence_sampler.py:slice_episode_into_segments` |
| **L2-IP-022** | Sequence continuity | Episode of length `16` (`obs[t] = t * 0.01`) | Slice with `stride=2` into overlapping segments | Check `seg.obs[k] == ep["obs"][start + k]` across consecutive indices | Exact contiguous indexing preserved across `start = 0, 2, 4` | G1 | `sequence_sampler.py:70-85` |
| **L2-IP-023** | Episode boundary | Two distinct concatenated episodes | Concatenate Ep1 (high burst) and Ep2 (quiet reset); slice each independently vs naive combined | Compare initial `(h, c)` and check segments never straddle two episodes without reset | Segments originate from single-episode streams; episode start resets `h=0, c=0` | G1 | `sequence_sampler.py`, `r2d2_trainer.py:385-390` |
| **L2-IP-024** | Burn-in segment | Same segment with/without burn-in `[0:4]` | Run `forward_sequence(..., burn_in_steps=4)` vs `burn_in_steps=0` | Verify stop-gradient decoupling at `t = burn_in - 1` (`dW` zero wrt burn-in steps) | Modifying loss on `[4:11]` produces zero BPTT sensitivity through weights at steps `[0:4)` | G1 | `recurrent_encoder.py:134-159`, `loss.py` |
| **L2-IP-025** | Padding mask | Short episode `T_ep = 7 < T_seq = 11` | Slice into padded segment (`DEF-L2-08`) with explicit `valid_mask` (`1` for `0..6`, `0` for `7..10`) | Run loss & backward pass | Padded indices `7..10` contribute `0.0` loss and `0.0` gradient | G1 | `sequence_sampler.py:53-68`, `loss.py` |
| **L2-IP-026** | Hidden-state shape | `R2D2Network(hidden_dim=64)` | Valid `(64,)` vs wrong `(32,)`, `(65,)`, `(1, 64)` `h_init`/`c_init` | Call `step`, `forward_sequence`, `SequenceSegment` | Only `shape==(64,)` accepted; mismatched shapes raise `ValueError` | G1 | `recurrent_encoder.py:81-129` |
| **L2-IP-027** | Hidden-state init | `RecurrentLSTMEncoder` | Pass `h_init=None, c_init=None` (defaults to zero) vs explicit non-zero `(h0, c0)` | Compare `forward_sequence` first step against manual `step(x0, h0, c0)` | Exact `1e-15` match with both zero and explicit `(h0, c0)` | G1 | `recurrent_encoder.py:74-129` |
| **L2-IP-028** | Reset behaviour | Sequential multi-episode runner (`eval` / `R2D2Runtime`) | Run noisy episode leaving large non-zero `(h, c)`, call `zero_state()` / `reset_state()`, feed fixed probe `x` | Compare probe output against fresh untouched instance | Exact `0.0` bitwise state leakage (`max_abs_diff == 0.0`) after reset | G1 | `r2d2_network.py:zero_state`, `r2d2_runtime.cpp:reset_state` |

#### 1D. Training Configuration (`test_training_config.py`)

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Oracle / Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|
| **L2-IP-029** | Network config | `training/configs/r2d2_v1_config.json` | Load canonical config + perturbed invalid dims (`hidden_dim<=0`, `input_dim!=8`) | Instantiate `R2D2Network` from config | Exact `(8, 64, 6, 9, gain_clip=0.30)` built; non-positive dims rejected | G1 | `r2d2_v1_config.json`, `r2d2_network.py` |
| **L2-IP-030** | Learning rate | `R2D2Trainer` | Test boundary `1e-6`, nominal `0.0025`/`0.0028`, invalid `0.0`, `-1e-3`, `nan` | Instantiate `R2D2Trainer(..., learning_rate=lr)` | `lr > 0` finite accepted; `<= 0` or NaN raises `ValueError` *(remediated in Set 1)* | G1 | `r2d2_trainer.py:R2D2Trainer.__init__` |
| **L2-IP-031** | Discount factor | `R2D2Trainer` & `loss.py` | Test `gamma=0.0` (myopic), `0.96`, `1.0`, invalid `-0.01`, `1.05`, `nan` | Init `R2D2Trainer` + compute `y_target` at `gamma=0.0` | `0 <= gamma <= 1` accepted (`gamma=0` uses `r_t` only); outside `[0,1]` rejected | G1 | `r2d2_trainer.py`, `loss.py` |
| **L2-IP-032** | Replay capacity | `PrioritizedSequenceReplayBuffer` | Test small `capacity=1`, nominal `4096`, invalid `0`, `-10` | Init buffer & fill past capacity | Valid capacities allocate circular buffer; `<= 0` raises `ValueError` | G1 | `prioritized_replay.py` |
| **L2-IP-033** | Batch size | Non-empty replay buffer (`n=10`) | Test `batch_size in {1, 4, 16, 32}`, invalid `0`, `-1` | Call `replay.sample(batch_size, beta)` | Valid returns exact list length `B` (`replace=True` when `n < B`); `<= 0` rejected | G1 | `prioritized_replay.py:76-99` |
| **L2-IP-034** | Sequence sampling | `slice_episode_into_segments` & config | Test `(burn_in=4, unroll=4, n_step=3)` vs invalid negative steps / `stride<=0` / `unroll<=0` | Slice & check segment bounds | Only positive `unroll>=1, n_step>=1, burn_in>=0, stride>=1` accepted | G1 | `sequence_sampler.py`, `r2d2_v1_config.json` |
| **L2-IP-035** | Target update | `TargetNetworkManager` | Test `sync_interval=1, 25`, `tau=1.0`, soft `tau=0.05`, invalid `sync<=0`, `tau<=0`, `tau>1` | Init & step `TargetNetworkManager` | Exact hard/Polyak rule for valid inputs; out-of-range rejected | G1 | `target_network.py` |
| **L2-IP-036** | Exploration config | `generate_qec_control_episode` / config | Test `eps=0.0` (greedy), `0.35`, `1.0` (uniform random), invalid `-0.1`, `1.1` | Run episode with fixed network & epsilon validation | `[0, 1]` enforced; `eps=0` selects greedy actions strictly; invalid rejected | G1 | `r2d2_trainer.py:511-515` |
| **L2-IP-037** | Optimizer config | `R2D2Trainer` Adam state & `grad_clip_norm` | Verify `beta1=0.9, beta2=0.999, eps=1e-8`, valid/invalid `grad_clip_norm<=0` | Check Adam buffers initialized zeroed with exact parameter keys; reject bad clip | Exact moment tensor shapes matching `net.parameters()`, positive clip enforced | G1 | `r2d2_trainer.py:595-606` |
| **L2-IP-038** | Checkpoint config | Temp output paths | Test nested valid path (`tmp_path/sub/ckpt.bin`), missing parent creation, unwritable `chmod 0555` dir | Call `export_r2d2_checkpoint` | Creates valid nested dir and writes `.bin/.json/.onnx`; fails cleanly on permission error | G1 | `export_checkpoint.py:137-215` |
| **L2-IP-039** | Seed configuration | `R2D2Network`, `PrioritizedSequenceReplayBuffer`, `R2D2Trainer` | Run identical seeds `(42, 42)` vs distinct `(42, 43)` | Compare initial weights, replay choices, and generated episodes | Same seed produces bitwise identical state/trajectories; distinct seeds diverge | G1 | `recurrent_encoder.py`, `q_value_head.py`, `prioritized_replay.py` |

---

### Set 2 — Model Output & Statistical Validation (`tests/layer2/set2_output_statistics/`) — Gates **G2, G8**

#### 2A. Network Architecture & Output Contracts (`test_network_outputs.py`, `test_gain_outputs.py`) — Gate **G2**

| ID | Test | Preconditions | Execution Procedure | Oracle / Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L2-OP-001** | Architecture construction | Init `R2D2Network` | Inspect `W_ih, W_hh, b_ih` dimensions & `encoder.hidden_dim` | `hidden_dim==64`, `W_ih=(256,8)`, `W_hh=(256,64)`, `b_ih=(256,)` | G2 | `recurrent_encoder.py`, `r2d2_network.py` |
| **L2-OP-002** | Dueling value head | `DuelingQValueHead` & `step`/`forward_sequence` | Feed `(64,)` & `(T, 64)` hidden states | `W_val=(1,64)`, scalar `state_value` (`float`) & `v_seq=(T,)` finite | G2 | `q_value_head.py`, `r2d2_network.py` |
| **L2-OP-003** | Dueling advantage head | Same | Feed hidden state through `head.forward_step` | `W_adv=(6,64)`, `advantages.shape == (6,)`, `adv_seq.shape == (T, 6)` | G2 | `q_value_head.py` |
| **L2-OP-004** | Dueling aggregation | Same | Compute `V + (A - mean(A))` independently and check mean advantage centering | `max |q - (V + A - mean(A))| < 1e-15` and `|mean(q) - V| < 1e-15` | G2 | `q_value_head.py:60-63` |
| **L2-OP-005** | Discrete action output | 100 random & regime inputs | Run `net.step` over mixed inputs including equal Q ties | `action_id in {0,1,2,3,4,5}` integer & matches `ACTION_DEFINITIONS[action_id]` | G2 | `r2d2_network.py:93-106` |
| **L2-OP-006** | Gain residual output | Same | Check `qubit_gains` & `gain_seq` | `qubit_gains.shape == (9,)` (`gain_seq=(T, 9)`) of finite `float64` | G2 | `q_value_head.py:64-66` |
| **L2-OP-007** | Gain bounds | Large weights (`W_gain *= 100`) & extreme `h` | Stress `W_gain @ h + b_gain` with magnitude `+/- 100` | Every element obeys `-0.30 <= delta_g[q] <= +0.30` strictly | G2 | `q_value_head.py:64-66` |
| **L2-OP-008** | Batch inference | Multiple batch trajectories `B=1, 8, 16` | Run `forward_sequence` over `B` segments | Outputs preserve `(T, ...)` for every element `b in 0..B-1` | G2 | `r2d2_network.py:forward_sequence` |
| **L2-OP-009** | Recurrent inference | Same input `x_t` with 2 distinct `(h_A, c_A) != (h_B, c_B)` | Call `net.step(x_t, h_A, c_A)` vs `net.step(x_t, h_B, c_B)` | `(h_next, c_next, q_values, qubit_gains)` differ when `W_hh != 0` and match unrolled `forward_sequence` | G2 | `r2d2_network.py:step` |
| **L2-OP-010** | Non-finite output detection | Poisoned weights (`W_val[0,0] = np.nan`) or NaN input | Run validation wrapper / `model_loader` | Explicitly flagged/rejected as non-finite failure rather than silently emitted | G2 | `model_loader.cpp:15-22`, Python validator |
| **L2-OP-011** | Model initialization | Two `R2D2Network(seed=S)` copies in eval (`eps=0`) | Run identical 24-step sequence on both copies | Bitwise identical `q_seq`, `gain_seq`, `h_seq` (`diff == 0.0`) | G2 | `r2d2_network.py` |
| **L2-OP-012** | Evaluation mode | `evaluate_policy_on_split` / `R2D2Network` | Verify zero dropout / stochastic noise during eval; repeat eval pass twice | Identical deterministic action sequence & reward to `0.0` difference | G2 | `evaluate_policy.py` *(explicitly documents no dropout layer)* |

#### 2B. Action-Selection Behaviour (`test_action_selection.py`) — Gate **G2**

| ID | Test | Input / Variation | Execution & Verification | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-STAT-001** | Greedy action selection | Construct weights making `A = [0, 0.2, 0.9, -0.5, 0.1, 0]` | Verify `step()` picks `action_id == 2` cleanly for all argmax indices `0..5` | G2 | `r2d2_network.py:93` |
| **L2-STAT-002** | Exploration behaviour | `generate_qec_control_episode(..., epsilon=0.5, policy_net=net)` | Over `N=1000` steps with fixed greedy preference, check all non-greedy actions sampled within `Binomial(1000, 0.5/6)` CI | G2 | `r2d2_trainer.py:511-515` |
| **L2-STAT-003** | Exploration decay | Multi-phase `populate_replay_from_seeds` (`eps=0.35 -> 0.15 -> 0.05`) & priority `beta: 0.4 -> 1.0` | Verify exploration & PER `beta` schedules progress monotonically per configured schedule | G2 | `evaluate_policy.py:146-172` |
| **L2-STAT-004** | Exploration floor | Extended run (`eps_min = 0.05`) & schedule helper | Assert runtime `epsilon >= eps_min` across all training phases (and `0.0` only during deterministic eval) | G2 | `r2d2_trainer.py:619` |
| **L2-STAT-005** | Tie-breaking | Zeroed advantage head (`W_adv=0, b_adv=0`) in Python & C++ | Both Python (`np.argmax`) and C++ (`>` strict comparison) select lowest tied index `0` identically | G2 | `r2d2_network.py:93`, `r2d2_runtime.cpp:134` |
| **L2-STAT-006** | Action distribution | `5`+ independent seeds of evaluation over all 5 families | Measure action entropy & verify trained checkpoint uses `>= 5` distinct actions (not collapsed to constant 1-action policy) | G2 | `evaluate_policy.py` |
| **L2-STAT-007** | Input sensitivity | Trained checkpoint `r2d2_v1_trained.bin` vs single-feature sweeps (burst spike `obs[7]: 0->0.9`, pair hotspot `obs[3]: 0.005->0.10`) | Action switches cleanly from `0` (stationary) toward `4` (burst recalibrate) / `2` (aggressive pair); gain residuals shift > `1e-3` | G2 | `r2d2_network.py`, `r2d2_trainer.py` |
| **L2-STAT-008** | Constant-input check | Feed identical stationary vector for `30` steps after warmup | Once LSTM stabilizes, `action_id` and `qubit_gains` stay constant without spurious oscillation | G2 | `r2d2_network.py:step` |
| **L2-STAT-009** | Noise-regime sensitivity | Probe sequences across all 6 regimes (`stationary`, `mod_corr`, `strong_corr`, `damped`, `burst`, `corrupted`) | Stratify action & gain distributions per regime; verify distinct regime-appropriate actions selected | G2 | `r2d2_trainer.py:240-265`, `test_layer2_ip_op.py` |
| **L2-STAT-010** | Gain sensitivity | Pass neutral `gains=None` vs trained `qubit_gains` into `project_q78_weights_d3` | Resulting Q7.8 LLR weights & `build_hardware_osd_lut_z` change on hotspot qubits matching attenuation formula | G2 | `r2d2_trainer.py:64-136` |

#### 2C. Statistical Held-Out Policy Evaluation (`test_policy_statistics.py`, `test_heldout_evaluation.py`) — Gate **G8**

| ID | Test | Experimental Setup | Paired Metric / Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-STAT-011** | Held-out evaluation | Evaluate on `heldout_test_seeds` (`2026..2099`, `>=30` seeds) + 3 unseen `heldout_families` (disjoint from train `1000..1039`) | Verify seed/family disjointness from training; report separate held-out LER & reward | G8 | `r2d2_v1_config.json`, `evaluate_policy.py` |
| **L2-STAT-012** | Multi-seed evaluation | Train `>= 5` independent seeds & evaluate across `>= 30` held-out seeds per trained model | Report both across-training-seed variance and across-eval-trajectory variance | G8 | `evaluate_policy.py`, new paired harness |
| **L2-STAT-013** | Regime-stratified evaluation | Stratify metrics across all 5 train-distribution families + 3 unseen held-out families | Per-regime table (`LER`, `reward`, `action_agreement`) emitted in JSON audit | G8 | `r2d2_v1_config.json`, `evaluate_policy.py` |
| **L2-STAT-014** | LER comparison (Paired!) | On identical `(err_z_win, sx_win)` shots per seed, run: (1) Static LUT (`STATIC_LUT_Z`), (2) Fixed/no-adaptation (`action=0`), (3) Rule/oracle baseline, (4) Uniform random, (5) Trained R2D2 | Paired `d_j = LER_base,j - LER_R2D2,j` (`DEF-L2-07` fix: execute `project_q78_weights_d3` + `count_logical_z_failures`, NOT reward-only!) + report absolute LERs | G8 | `r2d2_trainer.py:521-534`, `evaluate_policy.py` |
| **L2-STAT-015** | Reward comparison | Evaluate exact `compute_qec_action_reward` on identical held-out `(obs, opt_act, ler_gain)` streams for all 5 policies | R2D2 mean reward exceeds untrained, random, and static baselines with paired 95% CI | G8 | `evaluate_policy.py`, `r2d2_trainer.py:293` |
| **L2-STAT-016** | Action stability | Compute epoch-to-epoch switching rate `P(a_t != a_{t-1})` inside stationary windows vs switching points | Low chattering inside stationary segments; transitions concentrated at regime boundaries | G8 | `evaluate_policy.py` |
| **L2-STAT-017** | Adaptation delay | Inject step transition (`abrupt_regime_switch` at `ep=8` & `burst_intermittency_r2d2` at `ep=4`) | Measure epochs elapsed between regime transition and first target-regime action (`<= 1–2` epochs) | G8 | `r2d2_trainer.py:443-474` |
| **L2-STAT-018** | Recovery behaviour | Measure epochs post-burst (`ep=6..8` in `burst_intermittency_r2d2` after shock ends) | Policy returns cleanly from `CONSERVATIVE_FALLBACK_K3` / `FAST_RECALIBRATE_K5` to baseline | G8 | `r2d2_trainer.py:456-474` |
| **L2-STAT-019** | Worst-case behaviour | Compute min-reward & max-LER quantile (`95th` percentile & worst single regime) across held-out families | Explicitly logged in acceptance JSON; fallback regime (`action=5`) protects worst-case corrupted epochs | G8 | `r2d2_trainer.py:353-356` |
| **L2-STAT-020** | Confidence intervals | Paired differences `d_j` across `M >= 30` trajectories (`t`-interval / `10,000`-resamples bootstrap CI at 95%/99%) | Every reported comparison includes `(mean, std, ci_low, ci_high, p_value)` — zero point-estimate-only claims | G8 | Set 2 statistical audit |

---

### Set 3 — Training Pipeline and System-Flow Validation (`tests/layer2/set3_training_pipeline/`) — Gates **G3, G4**

#### 3A. Environment-to-Training Dataflow (`test_environment_dataflow.py`) — Gate **G3**

| ID | Test | Preconditions | Execution & Spy / Probe Method | Oracle / Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L2-FLOW-001** | Environment init | Valid config | Invoke `generate_qec_control_episode` across all 5 train + 3 heldout families | Initializes fresh `rng`, `prev_p`, `prev_alpha`, `h=0, c=0`; draws actual shots via `sample_d3_z_window` | G3 | `r2d2_trainer.py:364-482` |
| **L2-FLOW-002** | Observation acquisition | Live env episode | Compare generated `obs` across two seeds & spy `sample_d3_z_window` call count | `call_count == num_epochs`; observations differ across seeds/families cleanly | G3 | `r2d2_trainer.py:481-499` |
| **L2-FLOW-003** | Action dispatch | `policy_net` active (`epsilon=0.0`) | Spy `project_q78_weights_d3` inputs during episode generation | Passed `(pair_weight_scale, fallback)` matches exact chosen `ACTION_DEFINITIONS[chosen_act]` | G3 | `r2d2_trainer.py:507-528` |
| **L2-FLOW-004** | Environment transition | Active `policy_net` | Compare decoded failures when swapping `policy_net` choice (`act=0` vs `act=2`) on identical `err_z_win, sx_win` | `w_chosen` & `lut_chosen` reflect the dispatched action parameters | G3 | `r2d2_trainer.py:521-532` |
| **L2-FLOW-005** | Next observation | Multi-epoch trajectory (`num_epochs=16`) | Verify `innov` at step `t` equals `sqrt((||p_t - p_{t-1}||^2 + ||a_t - a_{t-1}||^2)/21)` | `obs[t][6]` mathematically matches step-to-step state transition `prev_p -> p_t` | G3 | `r2d2_trainer.py:490-495` |
| **L2-FLOW-006** | Reward acquisition | Fixed `err_z_win`, varied `chosen_act` | Recompute `ler_gain` & `compute_qec_action_reward` from `decoded_failures` & `static_failures` | Returned `ep["rewards"][t]` matches independent evaluation of that exact transition | G3 | `r2d2_trainer.py:529-544` |
| **L2-FLOW-007** | Terminal handling | `num_epochs=16` | Inspect `ep["dones"]` & replay bootstrap skip | `dones[:-1] == 0.0`, `dones[-1] == 1.0`, propagated unchanged into `SequenceSegment.dones` | G3 | `r2d2_trainer.py:535`, `sequence_sampler.py:77` |
| **L2-FLOW-008** | Transition integrity | Complete episode | Compare every index of sliced `SequenceSegment` against parent episode arrays `[start : start+11]` | Bitwise equality across all 8 fields (`obs, actions, rewards, dones, target_gains, optimal_actions, h_init, c_init`) | G3 | `sequence_sampler.py:70-85`, `r2d2_trainer.py:627-645` |
| **L2-FLOW-009** | Episode reset | Sequentially generate Ep1 then Ep2 in `populate_replay_from_seeds` | Inspect `h_states[0], c_states[0]` of Ep2 | Strictly `np.zeros(64)` at epoch `0` of every new episode | G3 | `r2d2_trainer.py:385-390` |
| **L2-FLOW-010** | Environment variation | Switch family `"stationary_independent"` -> `"burst_intermittency_r2d2"` | Compare empirical distribution of generated `obs[:, 7]` (burst) & `obs[:, 3]` (`max_alpha`) | Burst intensity shifts upward significantly in epochs `4,5,9,10` matching env physics | G3 | `r2d2_trainer.py:400-474` |

#### 3B. Replay and Sequence Pipeline (`test_replay_pipeline.py`) — Gate **G3**

| ID | Test | Preconditions | Execution Procedure | Oracle / Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L2-FLOW-011** | Replay insertion | `R2D2Trainer` + empty buffer | Populate replay via `populate_replay_from_seeds` (e.g., `2` seeds) | `len(replay) == num_added` (until capacity), priorities finite positive | G3 | `prioritized_replay.py:add`, `r2d2_trainer.py:607` |
| **L2-FLOW-012** | Replay retrieval | Populated buffer | Sample batch -> assert object identity / field equality with `replay.storage[indices[k]]` | Exact segment equality on every retrieved item | G3 | `prioritized_replay.py:sample` |
| **L2-FLOW-013** | Sequence continuity | Sliced episodes (`T_ep=16, T_seq=11, stride=2`) | Verify each segment holds 11 consecutive episode epochs in monotonic time | Zero scrambled, repeated, or skipped interior steps | G3 | `sequence_sampler.py:70-85` |
| **L2-FLOW-014** | Episode isolation | Replay populated from 5 distinct episodes (`ep_0..ep_4`) | Sample `100` segments -> inspect `seg.seed` / `seg.family` & monotonic dones inside each segment | `np.sum(seg.dones[:-1]) == 0` unless explicitly padded at tail; zero cross-episode stitching | G3 | `sequence_sampler.py` |
| **L2-FLOW-015** | Replay capacity | `PrioritizedSequenceReplayBuffer(capacity=10)` | Insert `25` distinct segments | `len(buf) == 10` strictly bounded at capacity | G3 | `prioritized_replay.py:68-74` |
| **L2-FLOW-016** | Replay overwrite | `capacity=4` inserting tagged segments `0..5` | Inspect `buf.storage` IDs after 6 inserts (`pos` FIFO ring wraparound) | Ring slots `0,1` hold segments `4,5` and `2,3` retain `2,3` cleanly | G3 | `prioritized_replay.py:71-74` |
| **L2-FLOW-017** | Prioritized sampling | Buffer with 2 segments (`prio=10.0` vs `prio=0.01`, `alpha=0.6`) | Sample `N=5,000` draws (`batch=1`), measure empirical draw fraction | Matches exact theoretical `((p_i+eps)^0.6) / sum` within Clopper-Pearson 99% CI | G3 | `prioritized_replay.py:86-91` |
| **L2-FLOW-018** | Importance weights | Non-uniform `probs`, `beta=0.4` vs `0.8` | Recompute `((N*P(i))^{-beta}) / max_j ((N*P(j))^{-beta})` manually | Max abs error `< 1e-13`; higher-priority items strictly get lower IS weights (`max <= 1.0`) | G3 | `prioritized_replay.py:93-96` |
| **L2-FLOW-019** | Priority updates | `train_step` on batch of indices `I` | Verify `replay.priorities[i]` updates exclusively for sampled indices `I` matching `0.9*max|td| + 0.1*mean|td| + eps` | Exact match `< 1e-13`; unsampled indices unchanged | G3 | `prioritized_replay.py:101-110`, `r2d2_trainer.py:696` |
| **L2-FLOW-020** | Burn-in handling | Single segment `[0:11]`, `burn_in=4` | Inject NaN/huge sentinel gradient sensitivity on `x[0:4]` vs perturb unroll targets on `[4:8]` | Wait: `x[0:4]` alters `(h_3, c_3)` forward state cleanly, while **no TD/margin/gain loss is accumulated for `t < 4`** and BPTT stops at `t=4`! Verify gradient match vs finite diff over `[4:8]` only! | G3 | `recurrent_encoder.py:134-183`, `loss.py:66-141` |
| **L2-FLOW-021** | Padding handling | Padded segment (`T_ep < T_seq`, `valid_mask`) | Mask out padded timesteps (`DEF-L2-08`) and verify zero contribution to scalar loss and gradients | Exact equality with unpadded sub-loss | G3 | `sequence_sampler.py`, `loss.py` |
| **L2-FLOW-022** | Replay reproducibility | Two buffers with identical `seed=302` & insertion order | Call `sample(batch_size=16, beta=0.6)` 10 times on both | Bitwise identical `(indices, is_weights)` across all 10 draws | G3 | `prioritized_replay.py:37,91` |

#### 3C. Recurrent Training and Optimizer Execution (`test_recurrent_training.py`, `test_optimizer_execution.py`) — Gate **G3**

| ID | Test | Preconditions | Execution & Check | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-FLOW-023** | Forward training pass | Valid replay batch (`B=16`) | Run forward pass over batch -> assert finite `loss`, `q_seq`, `gain_seq`, `h_seq` | G3 | `loss.py`, `r2d2_network.py` |
| **L2-FLOW-024** | Backpropagation | Non-zero TD & gain error batch | Inspect all 9 keys returned by `backward_sequence` (`W_ih, W_hh, b_ih, W_val, b_val, W_adv, b_adv, W_gain, b_gain`) | Every parameter key has non-zero L2 norm gradient | G3 | `r2d2_network.py:147-184` |
| **L2-FLOW-025** | Gradient finiteness | Diverse stress segments | Assert `np.all(np.isfinite(g))` across 20 consecutive update steps | G3 | `r2d2_trainer.py:648-697` |
| **L2-FLOW-026** | Gradient clipping | Force giant grad norm `> grad_clip_norm = 5.0` | Capture clipped `g_scaled` & moments inside `_apply_adam_step`: `||g_scaled||_2 <= 5.0 + 1e-12` cleanly, while reported `grad_norm` records raw unclipped norm | G3 | `r2d2_trainer.py:650-667` |
| **L2-FLOW-027** | Optimizer update | Clone parameters before `train_step()` | Run 1 `train_step()` -> check `all(param_after != param_before)` & `adam_t == 1`, `target_net` unchanged (`step_count=1 < 25`) | G3 | `r2d2_trainer.py:656-700` |
| **L2-FLOW-028** | Target network | `sync_interval=5` (hard `tau=1.0`) & `tau=0.1` (soft) | Run `25` train steps; assert target weights freeze on steps `1..24`, match online `==` on step `25` (`tau=1.0`) or equal `0.9*t + 0.1*o` (`tau<1`) | G3 | `target_network.py`, `r2d2_trainer.py:698` |
| **L2-FLOW-029** | Recurrent BPTT | Segment where `obs[t=4]` affects loss at `t=7` | Zero out losses at `t=4,5,6`, apply loss only at `t=7`; verify non-zero gradient on gate weights driven by step `4->7` chain, matching FD `< 1e-6` | G3 | `recurrent_encoder.py:142-184` |
| **L2-FLOW-030** | Loss masking | Segment with burn-in `=4` and explicit `padding_mask` | Verify TD/margin/gain loss evaluated strictly over valid unroll steps `[burn_in, burn_in+unroll)` with `padding_mask == 1` | G3 | `loss.py:66-134` |
| **L2-FLOW-031** | Training-step counter | `trainer` over 30 calls to `train_step()` | Verify `trainer.adam_t == 30` and `trainer.target_mgr.step_count == 30` | G3 | `r2d2_trainer.py:656,698` |
| **L2-FLOW-032** | Learning-rate schedule | `trainer.lr` + optional config scheduler | Verify `trainer.lr` matches config JSON (`DEF-L2-09`) and any configured decay formula across steps | G3 | `r2d2_trainer.py:588,667` |
| **L2-FLOW-033** | Loss reduction | Controlled synthetic 2-regime learning dataset (fixed small replay, `60` Adam steps) | Verify scalar loss drops monotonically / > 60% from initial mean first-10 to final-10 steps | G3 | `r2d2_trainer.py:train_step` |
| **L2-FLOW-034** | Checkpoint trigger | `run_full_training_and_export` (`eval interval = 40`, best-val trigger) | Spy validation save trigger: verifies best checkpoint updates iff `val_reward >= best_val_reward` | G3 | `evaluate_policy.py:174-186` |

#### 3D. Training-to-Checkpoint Dataflow (`test_checkpoint_provenance.py`) — Gate **G4**

| ID | Test | Preconditions | Execution & Verification | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-FLOW-035** | Checkpoint provenance | Completed training run in `tmp_path` | Check exported `.json` contains run seed, config hash, git commit, timestamps, exact `binary_checkpoint.sha256` *(enhance DEF-L2-09)* | G4 | `export_checkpoint.py:190-214`, runner |
| **L2-FLOW-036** | Parameter persistence | Trained `net` pre-export vs loaded `load_r2d2_checkpoint_bin` | Assert `max |W_saved - W_trained| == 0.0` across all 11 tensors | G4 | `export_checkpoint.py:167-252` |
| **L2-FLOW-037** | Optimizer persistence | Save/resume bundle (`adam_t, m, v, lr`) for resume training (`DEF-L2-09`) | Assert optimizer state `(adam_t, m, v)` serializes and restores identically if resume supported (else mark explicit limitation) | G4 | `r2d2_trainer.py:596-605` |
| **L2-FLOW-038** | Recurrent configuration | Inspect `.bin` 32-byte header & `.json["architecture"]` | Matches `(input_dim=8, hidden_dim=64, num_actions=6, num_qubits=9, gain_clip=0.30)` | G4 | `export_checkpoint.py:151-200` |
| **L2-FLOW-039** | Training metadata | Inspect `.json["training_metrics"]` & `datasets/..._manifest.json` | Contains `num_updates`, `batch_size`, initial/final loss & TD error, seed ranges, families | G4 | `evaluate_policy.py:203-244` |
| **L2-FLOW-040** | Checkpoint reload | Reload `.bin` from fresh `R2D2Network` | Run 24-step sequence on trained net & reloaded net -> bitwise `0.0` output difference | G4 | `export_checkpoint.py:218-252` |
| **L2-FLOW-041** | Training resume | Train 30 steps -> snapshot `(net, target_net, adam_state)` -> resume 30 more steps vs unbroken 60 steps | Resumed weights & losses match unbroken 60-step run to `< 1e-13` | G4 | `r2d2_trainer.py`, resume checkpoint helper |
| **L2-FLOW-042** | Best-vs-latest checkpoint | Trigger artificial val drop at step `N` | Verify `best_state` retains higher-validation weights when final step regresses (`evaluate_policy.py:182-186`) | G4 | `evaluate_policy.py:155-186` |

#### 3E. End-to-End Training Execution (`test_recurrent_training.py` / integration) — Gate **G4**

| ID | Test | Variation / Setup | Expected Result | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-FLOW-043** | Minimal training run | Config-driven run with `num_updates=20, batch_size=8` (`seed=11`) | Completes replay + 20 updates + export in `< 3s` with valid `157,984`-B `.bin` | G4 | `evaluate_policy.py` / `run_layer2_pipeline.py` |
| **L2-FLOW-044** | Nominal training run | Standard `num_updates=320, batch_size=16` (`seed=20260405`) | Final loss `< 0.25 * init_loss`, val agreement `>= 0.80`, distinct actions `>= 5` | G4 | `evaluate_policy.py:95-247` |
| **L2-FLOW-045** | Extended training run | Larger budget (`num_updates=640`) tracking stability & peak memory | Stable loss plateau, zero NaN weights, bounded RSS | G4 | `r2d2_trainer.py`, `evaluate_policy.py` |
| **L2-FLOW-046** | Multiple seeds | `>= 3` short functional seeds + `>= 5` statistical training seeds | Distinct trained weight hashes & non-trivial agreement across every seed | G4 | `evaluate_policy.py` |
| **L2-FLOW-047** | Environment change | Train subset `families=["stationary_independent"]` vs full 5 families | Stationary-only trained replay reflects stationary observations; multi-family generalizes broader action distribution | G4 | `r2d2_trainer.py:607` |
| **L2-FLOW-048** | Reward change | Modify `compute_qec_action_reward` / flip action margin sign on toy task | Learned Q-preferences shift monotonically toward highest-rewarded action | G4 | `loss.py`, `r2d2_trainer.py` |
| **L2-FLOW-049** | Resume training | Interrupt mid-run (after step 20) -> reload state -> continue 20 updates cleanly | Final checkpoint reaches same loss band without re-initializing from scratch | G4 | Training runner |
| **L2-FLOW-050** | Training failure | Inject `NaN` observation into sampled replay batch | Update aborts with explicit exception and does NOT overwrite last valid saved checkpoint on disk | G4 | `r2d2_trainer.py`, checkpoint saver |

---

### Set 4 — Export, C++ Runtime & Deployment Validation (`tests/layer2/set4_export_cpp_deployment/`) — Gates **G5, G6**

#### 4A. Exporter Tests (`test_checkpoint_export.py`, `test_tensor_mapping.py`) — Gate **G5**

| ID | Test | Setup & Verification Procedure | Oracle / Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-DEP-001** | Checkpoint loading | Run fresh training in `tmp_path`, pass trained `net` (NOT repo canned file!) to `export_r2d2_checkpoint` | Exports live trained weights matching `net.get_state_dict()` SHA-256 | G5 | `export_checkpoint.py:137` |
| **L2-DEP-002** | Architecture mapping | Compare `TENSOR_ORDER` in `export_checkpoint.py`, `r2d2_schema.json["tensor_order"]`, and C++ `load_r2d2_binary_checkpoint` struct read order | Exact 1:1 bijection of all 11 tensor names, offsets, and element counts (`sum = 19,744`) | G5 | `export_checkpoint.py:14-26`, `r2d2_schema.json`, `model_loader.cpp:109-119` |
| **L2-DEP-003** | Parameter count | Sum `arr.size` across exported tensors & compare `32 + 19744 * 8` | Exact `19,744` float64 scalars = `157,984` bytes payload+header | G5 | `export_checkpoint.py:165-172`, `model_loader.cpp:55-68` |
| **L2-DEP-004** | Tensor shape mapping | Inject deliberately wrong shape into `state_dict` and verify valid tensor dimensions `(256,8), (256,64), (256,), (1,64), (1,), (6,64), (6,), (9,64), (9,)` | Valid exports cleanly; mismatched tensor shape raises explicit `ValueError` | G5 | `export_checkpoint.py:169-170` |
| **L2-DEP-005** | Tensor ordering | Construct asymmetric sentinel weights `W_ih[r, c] = 1000*r + c` -> export `.bin` -> read inside C++ `test_parity` | Row-major C-contiguous memory ordering in C++ matches Python `(r, c)` element-for-element | G5 | `export_checkpoint.py:168`, `model_loader.cpp:111-118` |
| **L2-DEP-006** | Bias mapping | Zero all `W_*`, set distinct prime biases `b_ih, b_val, b_adv, b_gain` | Python & C++ match identically with weights zeroed (isolates pure bias offset paths) | G5 | `export_checkpoint.py`, `r2d2_runtime.cpp` |
| **L2-DEP-007** | LSTM gate mapping | Isolate one gate at a time (`b_ih[0:64]=1` vs `[64:128]` vs `[128:192]` vs `[192:256]`) | Only intended `[i, f, g, o]` gate activates identically in Python and C++ | G5 | `recurrent_encoder.py:92-95`, `r2d2_runtime.cpp:101-106` |
| **L2-DEP-008** | Gate ordering | Verify `[i, f, g, o]` slice `[0:H, H:2H, 2H:3H, 3H:4H]` in Python, C++, `.onnx`, and schema JSON | Consistent `[i, f, g, o]` ordering across all 4 representations | G5 | `recurrent_encoder.py:34-37`, `r2d2_schema.json:20-22` |
| **L2-DEP-009** | Gate bias handling | Verify single fused `b_ih[256]` (no duplicate `b_hh` offset bias missed or double-counted) | Python `b_ih` matches C++ `weights_.b_ih` to `0.0` | G5 | `recurrent_encoder.py:90`, `r2d2_runtime.cpp:90` |
| **L2-DEP-010** | Dueling head export | Zero LSTM/other bias; set unique `W_val`, `W_adv` & check C++ `out.state_value` & `out.advantages` | Exact `V` and `A[0..5]` parity (`< 1e-14`) | G5 | `r2d2_runtime.cpp:113-128`, `q_value_head.py:60-62` |
| **L2-DEP-011** | Gain head export | Isolate `W_gain[9,64]`, `b_gain[9]` | All 9 residuals `out.qubit_gains[0..8]` match Python `< 1e-14` | G5 | `r2d2_runtime.cpp:141-147`, `q_value_head.py:64-66` |
| **L2-DEP-012** | Metadata export | Parse `.json` sidecar | Contains `model_name, schema_version, magic, architecture, binary_checkpoint, onnx_checkpoint, tensor_stats` | G5 | `export_checkpoint.py:190-214` |
| **L2-DEP-013** | Export checksum | Recompute `hashlib.sha256` on `.bin` and `.onnx` bytes on disk | Exact string match with `meta["binary_checkpoint"]["sha256"]` & `meta["onnx_checkpoint"]["sha256"]` | G5 | `export_checkpoint.py:184,205,209` |
| **L2-DEP-014** | Export corruption | Bad magic, wrong version `2`, wrong `hidden_dim=32`, truncated file (`-8` B), appended bytes (`+8` B), NaN weight inside `.bin` | Both Python `load_r2d2_checkpoint_bin` and C++ `load_r2d2_binary_checkpoint` return error (`False`) | G5 | `export_checkpoint.py:220-250`, `model_loader.cpp:70-122` |
| **L2-DEP-015** | Export round trip | Export `net -> .bin -> load_r2d2_checkpoint_bin -> R2D2Runtime` | Loaded Python & C++ match original `net` `< 1e-12` over 24-step sequence | G5 | `export_checkpoint.py`, `r2d2_runtime.cpp` |

#### 4B. Precision & Quantization Tests (`test_precision_conversion.py`) — Gate **G5**

> Explicit Spec Rule (Section 4B & `DEF-L2-03`): Only IEEE-754 **FP64** is actually implemented in `deployment/cpp/`. `FP32` and `INT8` are explicitly checked for unsupported status (`INCONCLUSIVE` / `NOT_IMPLEMENTED`, NEVER forged as `PASS`).

| ID | Test | Status in Verified Repo | Test Behavior & Reporting Rule | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-DEP-016** | FP32 export | **Unimplemented (`DEF-L2-03`: `.bin` is FP64 `<f8`)** | Explicitly test FP64 native parity (`< 1e-12`) AND emit explicit `INCONCLUSIVE` ("FP64-only backend; no FP32 exporter implemented") unless FP32 path is added | G5 | `export_checkpoint.py`, `r2d2_schema.json` |
| **L2-DEP-017** | INT8 calibration | **Unimplemented (`DEF-L2-03`: no INT8 path in Layer 2)** | Check capability discovery (`has_int8_export == False`); emit explicit `INCONCLUSIVE` (`INT8_UNSUPPORTED`) — never coerce to PASS | G5 | `checkpoints/README.md`, `deployment/cpp/` |
| **L2-DEP-018** | Scale and zero-point | **Unimplemented (`DEF-L2-03`: no INT8 path in Layer 2)** | Verify no bogus scale/zero-point fields exist in schema; emit explicit `INCONCLUSIVE` (`INT8_UNSUPPORTED`) | G5 | `r2d2_schema.json`, `export_checkpoint.py` |
| **L2-DEP-019** | Saturation handling | **Unimplemented (`DEF-L2-03`: no INT8 path in Layer 2)** | Validate FP64 activation/gate/gain saturation clamps (`[-4,4]`, `[-30,30]`, `[-15,15]`, `[-0.30,0.30]`) AND emit explicit `INCONCLUSIVE` for INT8 clamping | G5 | `r2d2_runtime.cpp`, `recurrent_encoder.py` |
| **L2-DEP-020** | Quantized inference | **Unimplemented (`DEF-L2-03`: no INT8 path in Layer 2)** | Verify `model_loader.cpp` / `r2d2_runtime.cpp` lack INT8 kernels; emit explicit `INCONCLUSIVE` (`INT8_UNSUPPORTED`) | G5 | `r2d2_runtime.cpp`, `model_loader.cpp` |
| **L2-DEP-021** | Action agreement | **Unimplemented (`DEF-L2-03`: no INT8 path in Layer 2)** | Assert no INT8 backend exists to compare against FP64 argmax; emit explicit `INCONCLUSIVE` (`INT8_UNSUPPORTED`) | G5 | `r2d2_runtime.cpp` |
| **L2-DEP-022** | Gain deviation | **Unimplemented (`DEF-L2-03`: no INT8 path in Layer 2)** | Assert no INT8 gain output exists to compare against FP64 `[-0.30,+0.30]`; emit explicit `INCONCLUSIVE` (`INT8_UNSUPPORTED`) | G5 | `r2d2_runtime.cpp` |
| **L2-DEP-023** | State quantization | **Unimplemented (`DEF-L2-03`: no INT8 path in Layer 2)** | Confirm runtime state remains IEEE-754 `double[64]`; emit explicit `INCONCLUSIVE` (`INT8_UNSUPPORTED`) | G5 | `r2d2_runtime.h` |
| **L2-DEP-024** | Long-sequence quantization | **Unimplemented (`DEF-L2-03`: no INT8 path in Layer 2)** | Confirm no quantized drift test executable; emit explicit `INCONCLUSIVE` (`INT8_UNSUPPORTED`) | G5 | `r2d2_runtime.cpp` |
| **L2-DEP-025** | Edge-case quantization | **Unimplemented (`DEF-L2-03`: no INT8 path in Layer 2)** | Confirm no quantized near-zero/saturated kernel exists; emit explicit `INCONCLUSIVE` (`INT8_UNSUPPORTED`) | G5 | `r2d2_runtime.cpp` |

*(Note also `.onnx`: tested structurally in `L2-DEP-008/013` as protobuf serializer artifact, and explicitly reported as `INCONCLUSIVE` for ORT execution due to `DEF-L2-02`.)*

#### 4C. Python-to-C++ Parity (`test_python_cpp_parity.py`, `cpp/test_parity.cpp`) — Gate **G5**

| ID | Test | Inputs & Procedure | Precision-Specific Parity Criteria | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-DEP-026** | Initial-state parity | Fresh `R2D2Network.zero_state()` vs `R2D2Runtime::hidden_state() / cell_state()` | Exact `0.0` difference across all 64 `h` and 64 `c` floats | G5 | `recurrent_encoder.py:zero_state`, `r2d2_runtime.cpp:reset_state` |
| **L2-DEP-027** | Single-step parity | 20 diverse observation vectors checked independently from zero state | Max abs error `< 1e-12` on `state_value, advantages, q_values, qubit_gains, h_, c_` | G5 | `r2d2_network.py:step`, `r2d2_runtime.cpp:step` |
| **L2-DEP-028** | Multi-step parity | Same `r2d2_v1_trained.bin` + random untrained nets across 50-step trajectories | Step-by-step `max_abs_err(q, gain, h, c) < 1e-12` at every `t=0..49` | G5 | `r2d2_network.py`, `r2d2_runtime.cpp` |
| **L2-DEP-029** | Reset parity | Run 20 noisy steps -> call `reset_state()` (C++) & `zero_state()` (Py) -> run 15 steps | Post-reset sequence parity `< 1e-12` and identical to un-warmed run (`diff == 0.0`) | G5 | `r2d2_runtime.cpp:67-70`, `r2d2_network.py:81` |
| **L2-DEP-030** | Batch parity | Run 16 trajectories sequentially through C++ loop (`reset_state` between episodes) vs Python `forward_sequence` loop | All 16 sequences match `< 1e-12` (documenting C++ API processes single sequence steps per instance) | G5 | `r2d2_runtime.h`, `r2d2_network.py:forward_sequence` |
| **L2-DEP-031** | Dueling parity | Inspect `state_value`, `advantages[6]`, `q_values[6]` in C++ vs Python | Individual head outputs match `< 1e-12` (`sum(advantages - V)` structure preserved) | G5 | `r2d2_runtime.cpp:113-138`, `q_value_head.py:60-62` |
| **L2-DEP-032** | Action parity | Compare `out.action_id`, `action_name`, `pair_weight_scale`, `bp_iterations`, `temporal_beta`, `recalibrate`, `fallback` | `100%` exact action & field parity whenever margin `Q_best - Q_2nd > 1e-9` | G5 | `r2d2_runtime.cpp:30-37,149-155`, `r2d2_network.py:40-47,94-106` |
| **L2-DEP-033** | Gain parity | Compare all 9 `qubit_gains` floats across 50 steps in 6 regimes | Max abs diff `< 1e-12` on all 9 channels & clipped `[-0.30, +0.30]` | G5 | `r2d2_runtime.cpp:141-147` |
| **L2-DEP-034** | Long-run drift | Unroll `2,000` continuous steps (reset-free!) checking error growth `t=0..1999` | Track max error over 2,000 steps; FP64 cumulative drift stays `< 1e-10` | G5 | `r2d2_runtime.cpp`, `recurrent_encoder.py` |
| **L2-DEP-035** | Multiple regimes | Test all 5 training families + 3 heldout families through both engines | `< 1e-12` parity across every regime | G5 | `r2d2_trainer.py`, `r2d2_runtime.cpp` |
| **L2-DEP-036** | Boundary inputs | All zeros `[0]*8`, negative `[-1]*8`, saturated clip `[10]*8`, `[0.30, 0.30, ...]` | `< 1e-12` parity (`[-4,4]` clamp behaves identically!) | G5 | `r2d2_runtime.cpp:80-85`, `recurrent_encoder.py:72` |
| **L2-DEP-037** | Reset-after-burst | Burst inputs driving saturation -> `reset_state()` -> quiet inputs | C++ & Python recover to identical steady state `< 1e-12` | G5 | `r2d2_runtime.cpp`, `r2d2_network.py` |

#### 4D. Runtime Lifecycle & Safety (`test_runtime_lifecycle.py`, `cpp/test_runtime.cpp`) — Gate **G5**

| ID | Test | Procedure & Safety Oracle | Gate | Codebase Mapping |
|---|---|---|---|---|
| **L2-DEP-038** | Runtime init | `R2D2Runtime::load_checkpoint()` & `load_default_checkpoint()` via `AVID_CHECKPOINT_PATH` | G5 | `model_loader.cpp`, `r2d2_runtime.cpp` |
| **L2-DEP-039** | Missing model | Call `load_checkpoint("/no/such/file.bin")` -> assert `false` + empty env default fallback rejected, AND verify `step()` without loaded model raises/flags unloaded state *(fix DEF-L2-06)* | G5 | `model_loader.cpp:48-52`, `r2d2_runtime.cpp:74-76` |
| **L2-DEP-040** | Invalid model version | Write `.bin` with `version=2` (`version != 1`) -> assert `false` with diagnostic error | G5 | `model_loader.cpp:90-93` |
| **L2-DEP-041** | Invalid tensor dims | Write `.bin` with `hidden_dim=32` in header -> assert `false` with `"Architecture dimension mismatch"` or size error | G5 | `model_loader.cpp:94-100` |
| **L2-DEP-042** | Repeated inference | Execute `100,000` C++ `step()` calls in loop | Finite outputs, stable latency & memory | G5 | `r2d2_runtime.cpp` |
| **L2-DEP-043** | State isolation | Instantiate `R2D2Runtime rtA, rtB`; drive `rtA` with extreme burst, keep `rtB` idle -> query probe on `rtB` | `rtB` equals zero-state baseline bit-for-bit (zero static globals) | G5 | `r2d2_runtime.h:45-49` |
| **L2-DEP-044** | Reset API | Step `rt` 50 times -> check `hidden_state()` non-zero -> `reset_state()` -> verify all 64 `h` and 64 `c` `== 0.0` | Exact zero reset | G5 | `r2d2_runtime.cpp:67-70` |
| **L2-DEP-045** | Invalid observation | Pass `NaN`, `+Inf`, `-Inf`, nullptr in C++ `step()` -> ensure explicit safe error/invalid output flag rather than silent clamp *(fix DEF-L2-06)* | Explicit safety handling verified | G5 | `r2d2_runtime.cpp` |
| **L2-DEP-046** | Runtime memory | Repeated load/step/destroy across 500 `R2D2Runtime` instances tracking RSS & valgrind/ASan cleanly | Zero leaks / unbounded growth | G5 | `r2d2_runtime.cpp`, `model_loader.cpp` |
| **L2-DEP-047** | Inference latency | Time `10,000` individual `rt.step()` calls in C++ (`std::chrono::steady_clock`), report `p50, p95, p99, max` microseconds | Latency distribution recorded in acceptance JSON | G5 | `cpp/test_runtime.cpp` |
| **L2-DEP-048** | Throughput | Sustained batch-free streaming throughput (`steps / sec`) over `50,000` steps | Reported in steps/sec (expected well above 10 Hz / daemon rate) | G5 | `cpp/test_runtime.cpp` |
| **L2-DEP-049** | Concurrent requests | Spawn 4 C++ threads each owning its own `R2D2Runtime` instance (`shared read-only` weights pattern) running `5,000` steps | Zero data races (clean under `-fsanitize=thread` or deterministic hash comparison vs single-thread) | G5 | `cpp/test_runtime.cpp` |
| **L2-DEP-050** | Runtime failure recovery | Fault `step()` with `NaN` or unloaded state -> inspect recovery (`reset_state()` + valid checkpoint load) | Faulted call flags error without returning unvalidated action; clean reload recovers valid inference | G5 | `r2d2_runtime.cpp` |

#### 4E. Actual Decoder Action-Effect Validation (`test_decoder_action_effect.py`, `cpp/` harness) — Gate **G6**

> Remediation note (`DEF-L2-07`): Builds a standalone C++ Q7.8 projection + 16-entry `build_hardware_osd_lut_z` decoder control harness inside `tests/layer2/set4_export_cpp_deployment/cpp/` matching `r2d2_trainer.py:64-220` so C++ `R2D2InferenceOutput` actually drives hardware-equivalent decoder weight banks & logical failure counts!

| ID | Test | Verification Method | Expected Result | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-DEP-051** | Action dispatch | Feed C++ `R2D2Runtime` output `(action_id, bp_iterations, pair_weight_scale, qubit_gains, fallback)` directly into decoder controller struct | Every field propagates without truncation/defaulting | G6 | `r2d2_runtime.h`, `r2d2_trainer.py:64,139` |
| **L2-DEP-052** | Action mapping | Probe all 6 regimes driving C++ `R2D2Runtime` to emit actions `0..5` -> inspect decoder control config | Matches exact `ACTION_DEFINITIONS` table (`K in {3,4,5}`, `pair_weight_scale`, `fallback`) | G6 | `r2d2_runtime.cpp:30-37`, `r2d2_v1_config.json` |
| **L2-DEP-053** | Gain application | Apply uniform `gain=+0.20` on single qubit `q=4` via C++ gain head -> `project_q78_weights_d3` | `w_q78[4]` scales up cleanly relative to `gain=0.0` (while clipped `1 + delta_g in [0.70, 1.30]`) | G6 | `r2d2_trainer.py:82-135` |
| **L2-DEP-054** | Invalid action safety | Force invalid `action_id = -1` / `99` / `NaN` gain into decoder adapter | Adapter clamps/rejects cleanly to `CONSERVATIVE_FALLBACK_K3` (`0x0468`) — never indexes out-of-bounds | G6 | Decoder safety wrapper |
| **L2-DEP-055** | Parameter bounds | Sweep extreme outputs through Q7.8 projection | Every hardware LLR `w_q78[0..8]` obeys `[16, 0x3FFF]` (`16..16383`) & `lut[0..15]` uses 9-bit masks | G6 | `r2d2_trainer.py:85,126` |
| **L2-DEP-056** | Actual decoder response | Decode identical crosstalk syndrome window `sx_win` under C++ `action=0` (`pair_scale=0`) vs C++ `action=2` (`pair_scale=1.35`) | `lut_a2 != lut_a0` AND decoded `LER(a2) < LER(a0)` on strongly correlated pair `(1,2)`! | G6 | `r2d2_trainer.py:139-220` |
| **L2-DEP-057** | No-op action | Dispatch `action=0` (`STATIC_PRIOR_K3`, `pair_weight_scale=0`) and `action=5` (`CONSERVATIVE_FALLBACK_K3`) | `action=0` skips pair-projection cleanly; `action=5` sets uniform fallback `0x0468` LUT matching `STATIC_LUT_Z` | G6 | `r2d2_trainer.py:72-88` |
| **L2-DEP-058** | Repeated action | Apply identical C++ control step 10 times on frozen state | Produces bitwise identical `w_q78`, `lut`, and failure counts | G6 | Decoder adapter + `r2d2_runtime.cpp` |
| **L2-DEP-059** | Action recovery | Drive `CONSERVATIVE_FALLBACK_K3` -> reset / stationary step `STATIC_PRIOR_K3` | Decoder restores baseline adaptive `w_q78` cleanly without stale fallback lock | G6 | `r2d2_runtime.cpp`, decoder adapter |
| **L2-DEP-060** | Closed-loop evaluation | Run deployed C++ `.bin` model in closed-loop over `>= 30` held-out multi-regime trajectories with `sample_d3_z_window -> feat -> C++ R2D2Runtime -> project_q78 -> LUT decode` | Deployed C++ loop reproduces Python held-out `LER` to parity `< 1e-12` (exact integer failure match!) and outperforms static `STATIC_LUT_Z` | G6 | `r2d2_runtime.cpp`, `evaluate_policy.py`, `r2d2_trainer.py` |

---

### Set 5 — R2D2-Specific Mathematical Validation (`tests/layer2/set5_r2d2_mathematics/`) — Gate **G7**

> Independent Reference Rule (`reference/r2d2_reference_oracle.py`): Every mathematical oracle in Set 5 is implemented independently from textbook equations (`sigma(x)`, `tanh(x)`, explicit 4-gate `[i, f, g, o]` LSTM step, dueling centering, Kapturowski `h(z)` / `h^{-1}(y)`, explicit multi-step Double-DQN target, finite-difference Jacobian) **without importing `training/models/` or `training/algorithms/`**.

#### 5A. Recurrent Network Mathematics (`test_lstm_equations.py`, `test_dueling_equations.py`)

| ID | Property | Input / Variation | Independent Reference Calculation (`reference/`) | Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L2-MATH-001** | LSTM gate equations | Known weights `W_ih, W_hh, b_ih` (`H=64` & `H=4`) + arbitrary `x, h_prev` | Explicit gate slices: $i_t=\sigma(g_0), f_t=\sigma(g_1), g_t=\tanh(g_2), o_t=\sigma(g_3)$ with documented clamps `[-30,30]`, `[-15,15]` | Max abs error `< 1e-14` on `i_gate, f_gate, g_gate, o_gate` | G7 | `recurrent_encoder.py:81-113` |
| **L2-MATH-002** | Hidden-state update | Same + non-zero `c_prev` | $h_t = o_t \odot \tanh(\text{clip}(c_t, -15, 15))$ | `< 1e-14` exact agreement | G7 | `recurrent_encoder.py:98-99` |
| **L2-MATH-003** | Cell-state update | Same | $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ | `< 1e-14` exact agreement | G7 | `recurrent_encoder.py:97` |
| **L2-MATH-004** | Gate ordering | 4 sparse tests setting one gate bias block `[0:64],[64:128],[128:192],[192:256]` high/low | Verify block 1 controls forget decay `c_t -> f*c_{t-1}`, block 0 & 2 write input, block 3 gates output | Only expected gate channel changes | G7 | `recurrent_encoder.py:34-37` |
| **L2-MATH-005** | Initial state | Zero default vs explicit `(h_0, c_0)` | Reference step starting from `0` vs explicit `(h_0, c_0)` | Exact `1e-15` match | G7 | `recurrent_encoder.py:74-129` |
| **L2-MATH-006** | Recurrent sequence | Unroll `T=15` sequence | Naive step-by-step Python loop vs `forward_sequence` | `< 1e-14` across entire `(15, 64)` trace | G7 | `recurrent_encoder.py:115-140` |
| **L2-MATH-007** | Reset semantics | Unroll `T=10` -> reset -> unroll `T=5` | Independent unroll of `T=5` from zero state | Exact `0.0` bit difference | G7 | `r2d2_network.py:81`, `r2d2_runtime.cpp:67` |
| **L2-MATH-008** | Gradient propagation | Random small network `(D=8, H=8)`, `T=6`, `burn_in=2` | Central finite differences $(L(\theta+\epsilon) - L(\theta-\epsilon))/(2\epsilon)$ ($\epsilon=10^{-6}$) on EVERY weight tensor `W_ih, W_hh, b_ih, W_val, b_val, W_adv, b_adv, W_gain, b_gain` | Max abs gradient error `< 1e-6` (and `< 1e-8` away from clip corners) | G7 | `recurrent_encoder.py:142-184`, `q_value_head.py:68-107` |
| **L2-MATH-009** | Dueling aggregation | Known `h` & weights | Independent $Q(h,a) = V(h) + A(h,a) - \frac{1}{6}\sum_{a'} A(h,a')$ | `< 1e-14` across all 6 actions | G7 | `q_value_head.py:60-63` |
| **L2-MATH-010** | Action-value invariance | Shift all advantages uniformly $A \leftarrow A + c$ (`c=4.2`); also add constant $c$ to `b_adv` | $Q$ values invariant under constant shift of $A$: $\Delta Q = 0.0$ and $\sum_{a} (Q(a) - V) = 0$ identically! | `< 1e-14` invariance verified | G7 | `q_value_head.py:62` |

#### 5B. R2D2 Target and Learning-Update Mathematics (`test_td_targets.py`, `test_gradient_reference.py`)

| ID | Property | Input / Setup | Independent Reference Oracle (`reference/`) | Acceptance Criterion | Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L2-MATH-011** | Discounted return | Segment rewards `r[0..10]`, `dones=[0..0]`, `n_step=3`, `gamma=0.96` | Explicit sum $G_t^{(3)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V_{\text{boot}}$ | `< 1e-13` match on exact `td_errors` & target | G7 | `loss.py:72-93` |
| **L2-MATH-012** | Double-DQN selection | Craft `online_net` picking `a*=2` while `target_net` has higher scalar on `a=4` at `t+3` | Verify `boot` selects `target_net`'s Q value at **online argmax `a*=2`** (NOT `max_a Q_target`!) | Matches Double-DQN target `< 1e-13`, differs strictly from single-DQN `max_target` | G7 | `loss.py:88-90` |
| **L2-MATH-013** | Target evaluation | Same + check Kapturowski $h^{-1}(Q_{\text{target}})$ before discount & $h(G_t^{(n)})$ after | Exact check of `h_inv` on bootstrap value & `h` on return (plus `h(h^{-1}(y)) == y` to `< 1e-10`) | `< 1e-12` agreement with independent $h, h^{-1}$ functions | G7 | `r2d2_network.py:13-37`, `loss.py:90-93` |
| **L2-MATH-014** | Terminal masking | Place `done[t+1] = 1.0` with huge non-zero `target_net` (`Q_target = 100.0`) | Return stops at `t+1`: $G_t = r_t + \gamma r_{t+1}$, ZERO bootstrap term | `< 1e-13` invariant to `target_net` perturbation | G7 | `loss.py:82-92` |
| **L2-MATH-015** | Truncation handling | Synthetic time-limit segment (`truncated=True` vs `terminated=True`) (`DEF-L2-10`) | True termination zeroes bootstrap; time-limit truncation bootstraps via target network | Both modes verified against exact independent formulas | G7 | `loss.py`, `r2d2_trainer.py` |
| **L2-MATH-016** | Target-network isolation | Run 10 `train_step()` updates (`sync_interval=25`) | Verify `target_mgr.target_net` parameters unchanged (`max diff == 0.0`) while `online_net` changes | Exact zero leak into target weights before sync | G7 | `target_network.py`, `r2d2_trainer.py:680-698` |
| **L2-MATH-017** | Target synchronization | Verify both `tau=1.0` at `step == sync_interval` and `tau=0.2` Polyak update | Check exact equality / convex combination `(1-tau)*T + tau*O` (`< 1e-14`) | Exact periodic & Polyak equations verified | G7 | `target_network.py:23-45` |
| **L2-MATH-018** | TD error | Pure-TD test (`gain_loss_weight=0.0, action_margin_weight=0.0`) | Verify `td_errors[u] == h(G_t^{(n)}) - Q_online(t, a_t)` & check full multi-term loss (`DEF-L2-05`) separately | Pure TD & composite TD+margin+gain match reference `< 1e-13` | G7 | `loss.py:93-134` |
| **L2-MATH-019** | Loss function | Synthetic segment covering both Huber regions (`|delta| <= 1` quadratic vs `|delta| > 1` linear) + IS weight `w` + DQfD margin + gain MSE | Recompute scalar total loss and `dq_seq`, `dgain_seq` by hand formula -> check Huber, margin hinge, and L2 gain terms | Scalar loss & all weight gradients match independent finite differences `< 1e-6` | G7 | `loss.py:101-142` |
| **L2-MATH-020** | Gradient clipping | Scale grads above `grad_clip_norm=5.0` vs below | Independent check `||grads_clipped||_2 == min(||grads||_2, clip_norm)` | Exact global L2 norm bound `< 1e-12` | G7 | `r2d2_trainer.py:650-668` |

#### 5C. Recurrent Replay & Sequence Mathematics (`test_recurrent_replay.py`)

| ID | Property | Input / Setup | Verification Oracle | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-MATH-021** | Sequence reconstruction | Slice unrolled episode at `start=4` storing `h_states[4], c_states[4]` | Unroll segment `[4..14]` from `(h_init, c_init)` vs continuing unbroken episode from step `0..14`: `h_seg[k] == h_ep[4+k]` (`< 1e-14`) | G7 | `sequence_sampler.py`, `recurrent_encoder.py` |
| **L2-MATH-022** | Burn-in correctness | Segment length `11`, `burn_in=4`, `unroll=4` | Perturb `W_ih` ONLY for `[0:4]` manually vs full backward: prove exact analytical gradients equal finite diffs of loss holding post-burn-in `(h_3, c_3)` constant (`dL_unroll / d(burn-in) == 0` wrt carry state!) | G7 | `recurrent_encoder.py:134-183` |
| **L2-MATH-023** | Sequence target alignment | Segment with unique probe rewards & actions per timestep | Independently map unroll index `u in 0..3` -> timestep `t = 4+u` -> lookahead `[t, t+1, t+2]` -> boot `t+3` | `< 1e-13` alignment across all 4 unroll indices | G7 | `loss.py:66-98` |
| **L2-MATH-024** | N-step boundary | Set `done[5] = 1.0` inside `t=4..7` unroll (`burn_in=4`) | Verify `u=0 (t=4)` stops discounting at `step=5` (`r_4 + gamma * r_5`), NOT bleeding into `r_6, V_boot(7)`! | `< 1e-14` exact boundary cut | G7 | `loss.py:82-84` |
| **L2-MATH-025** | Padding mask | Short padded episode `T_ep=8 < 11` (`DEF-L2-08`) | Padded timesteps `[8:11]` carry mask `0.0`, producing identically zero loss/grad contribution | G7 | `sequence_sampler.py:53-68`, `loss.py` |
| **L2-MATH-026** | Priority calculation | Sequence TD error arrays (all-zero, sparse peak `[0,0,0,4.0]`, uniform `[1,1,1,1]`) | Independent $p_i = 0.9 \max | \delta | + 0.1 \operatorname{mean} |\delta| + 10^{-4}$ (`< 1e-15`) | G7 | `prioritized_replay.py:47-53` |
| **L2-MATH-027** | Sampling probability | 4 items with priorities `[p_0..p_3]` | Chi-square GoF (`N=20,000` draws) matching exact theoretical $P(i) = (p_i + \epsilon)^\alpha / \sum_k (p_k + \epsilon)^\alpha$ | G7 | `prioritized_replay.py:86-91` |
| **L2-MATH-028** | IS weights | Sample batch across `beta in {0.0, 0.4, 1.0}` | `beta=0` -> all `is_weights == 1.0`; `beta > 0` matches $(N P(i))^{-\beta}/\max_j$ to `< 1e-14` | G7 | `prioritized_replay.py:93-96` |
| **L2-MATH-029** | Stored state fidelity | Compare `seg.h_init, seg.c_init` in sampled segment vs incorrect zero state | Perturbing `h_init` changes `q_seq[burn_in]` cleanly, proving stored recurrent state actively conditions burn-in start | G7 | `sequence_sampler.py:80-81`, `loss.py:44-48` |
| **L2-MATH-030** | Sequence duplication / overlap | `T_ep=16, T_seq=11, stride=2` -> generates `start in {0, 2, 4}` (3 overlapping segments) | Verify exact count `floor((T_ep - T_seq)/stride) + 1 = 3` & overlapping windows deep-copied (mutating seg0 never mutates seg1!) | G7 | `sequence_sampler.py:70-85` |

#### 5D. Learning Behaviour & Stability (`test_recurrent_training.py` / `set5`)

| ID | Test | Setup & Experiment | Expected Mathematical Outcome | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-MATH-031** | Synthetic known-policy task | Deterministic 2-state / 2-action toy environment where State A -> Action 2 yields `+2`, State B -> Action 4 yields `+2`, wrong action `-1` (`margin_weight=0` pure TD!) | Pure-TD R2D2 converges to `100%` optimal discrete action selection (`Q(A,2) > Q(A,other)` cleanly) | G7 | `R2D2Trainer`, `loss.py` |
| **L2-MATH-032** | Reward inversion | Flip reward sign (`+2 <-> -2`) in pure-TD toy task (`margin_weight=0`) | Learned argmax policy inverts symmetrically (`2 <-> other`) | G7 | `R2D2Trainer`, `loss.py` |
| **L2-MATH-033** | Action-reward causality | Irrelevant noise dummy observation feature + decoupled random reward | Zero systematic preference (`max |Q_diff| < epsilon` across seeds; random feature cannot create false causality beyond null noise) | G7 | `R2D2Trainer` |
| **L2-MATH-034** | Overfit diagnostic | Single deterministic 16-step sequence overfit for 100 Adam steps | Loss approaches `< 1e-4` & discrete action agreement hits `100%` (`16/16`) | G7 | `R2D2Trainer`, `loss.py` |
| **L2-MATH-035** | Gradient sanity | Test zero error (`delta=0, gain_err=0, margin=0`) vs active error | Zero loss -> identically `0.0` gradients on ALL parameters; non-zero error -> non-zero gradients on active heads | G7 | `loss.py`, `r2d2_network.py` |
| **L2-MATH-036** | Loss numerical stability | Extreme valid feature clips `[-4, 4]` & extreme rewards `+/- 100` | Zero NaN/Inf across 50 steps thanks to `value_rescale_h`, Huber `|delta|>1` linear tail & gate clips | G7 | `loss.py`, `r2d2_network.py` |
| **L2-MATH-037** | Seed sensitivity | 3 independent training seeds (`101, 202, 303`) | Initial weights, sample orders, and trained losses differ strictly (never frozen/identical) | G7 | `R2D2Network`, `R2D2Trainer` |
| **L2-MATH-038** | Training stability | Full 320-step trace tracking `(loss, grad_norm, mean_abs_td, reward)` | Finite everywhere, gradient norms bounded post-clip, TD error decreases monotonically on moving window | G7 | `r2d2_trainer.py:701-706` |
| **L2-MATH-039** | Policy generalization | Enforce disjoint train (`1000..1039`), val (`1500..1514`), heldout (`2026..2099`) & evaluate generalization gap | Held-out performance strictly exceeds static/untrained on unseen seeds & `heldout_*` families | G7 | `r2d2_v1_config.json`, `evaluate_policy.py` |
| **L2-MATH-040** | Action effect | Compare `a2 (pair_scale=1.35)` vs `a0 (pair_scale=0.0)` vs `a5 (fallback=True)` via `project_q78_weights_d3` | `a2` lowers crosstalk pair endpoint LLRs below aliased single weight; `a5` forces uniform `0x0468`; matches mathematical intent | G7 | `r2d2_trainer.py:64-136` |

#### 5E. Gain-Residual Mathematical Validation (`test_gain_residual_math.py`)

| ID | Property | Input / Check | Reference Calculation | Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L2-MATH-041** | Output dimension | `DuelingQValueHead.forward_step` & `R2D2Runtime::step` | Exact `len == 9` (`R2D2_NUM_QUBITS = 9`) | G7 | `q_value_head.py:51`, `r2d2_runtime.cpp:141` |
| **L2-MATH-042** | Output activation | Arbitrary `W_gain @ h + b_gain` | Independent `0.30 * tanh(clip(z, -15.0, 15.0))` (`< 1e-15`) | G7 | `q_value_head.py:64-65` |
| **L2-MATH-043** | Boundary mapping | Drive pre-activation to `-20.0`, `0.0`, `+20.0` (clipped `+/-15`) | Maps to `-0.30 * tanh(15) (~ -0.30)`, `0.0`, `+0.30 * tanh(15) (~ +0.30)` strictly inside `[-0.30, +0.30]` | G7 | `q_value_head.py:64` |
| **L2-MATH-044** | Residual scaling | Feature normalization `x_norm` & inverse mapping vs `gains_mult = 1.0 + delta_g` | Forward normalization & gain multiplier `(1 + delta_g) in [0.70, 1.30]` (clipped `[0.72, 1.28]` in oracle / `[0.70, 1.30]` on raw `llr`) exact | G7 | `recurrent_encoder.py:72`, `r2d2_trainer.py:84,515` |
| **L2-MATH-045** | Residual ordering | Isolate row `q` (`W_gain[q]` nonzero, others 0) for `q=0..8` | Only `qubit_gains[q]` changes and alters only `w_q78[q]` (when `pair_scale=0`) | G7 | `q_value_head.py:64`, `r2d2_trainer.py:83-85` |
| **L2-MATH-046** | Zero residual | Zero `W_gain=0, b_gain=0` -> `delta_g = 0.0^9` | `project_q78_weights_d3(..., qubit_gains=1.0 + delta_g)` produces bitwise identical weights to `qubit_gains=None` (when `pair_scale=0`) | G7 | `r2d2_trainer.py:83-85` |
| **L2-MATH-047** | Parameter composition | Compare `w_q78` with multiplicative `(1 + delta_g)` rule at positive & negative residual values | Verify exact `round(llr * (1 + delta_g) * 256)` multiplicative composition matching `project_q78_weights_d3` | G7 | `r2d2_trainer.py:82-85` |
| **L2-MATH-048** | Saturation behaviour | Huge positive & negative weights + `delta_g in [-0.30, +0.30]` into `project_q78_weights_d3` | `w_q78` strictly clipped to valid `[16, 0x3FFF]` (`[16, 0x0220]` on projected pair halves) — zero Q7.8 int overflow | G7 | `r2d2_trainer.py:85,127-130` |
| **L2-MATH-049** | Gradient flow | Pure gain MSE (`td=0, margin=0`) -> compare analytical `dW_gain, db_gain, dW_ih` vs central FD | `< 1e-6` match proving gain head backpropagates into shared LSTM trunk `Encoder`! | G7 | `q_value_head.py:87-98`, `r2d2_network.py:169-183` |
| **L2-MATH-050** | C++ gain parity | 50-step multi-regime trajectories Python vs C++ `.bin` | `max |delta_g_py - delta_g_cpp| < 1e-12` across all 9 channels & 50 steps | G7 | `r2d2_runtime.cpp:141-147`, `q_value_head.py:64-66` |

---

## 5. Layer 2 Test Folder Layout & Execution Contract

Matches Section 3–4 of the specification on `layer2-r2d2-training-deploy`:
- `tests/layer2/conftest.py` (shared config loaders, JSON audit writer, `PASS`/`FAIL`/`INCONCLUSIVE` collector)
- `tests/layer2/set1_input_contracts/` (`test_observation_contracts.py`, `test_action_reward_contracts.py`, `test_sequence_contracts.py`, `test_training_config.py`)
- `tests/layer2/set2_output_statistics/` (`test_network_outputs.py`, `test_action_selection.py`, `test_gain_outputs.py`, `test_policy_statistics.py`, `test_heldout_evaluation.py`)
- `tests/layer2/set3_training_pipeline/` (`test_environment_dataflow.py`, `test_replay_pipeline.py`, `test_recurrent_training.py`, `test_optimizer_execution.py`, `test_checkpoint_provenance.py`)
- `tests/layer2/set4_export_cpp_deployment/` (`test_checkpoint_export.py`, `test_tensor_mapping.py`, `test_precision_conversion.py`, `test_python_cpp_parity.py`, `test_runtime_lifecycle.py`, `test_decoder_action_effect.py`, `cpp/test_runtime.cpp`, `cpp/test_parity.cpp`)
- `tests/layer2/set5_r2d2_mathematics/` (`reference/r2d2_reference_oracle.py`, `test_lstm_equations.py`, `test_dueling_equations.py`, `test_td_targets.py`, `test_recurrent_replay.py`, `test_gradient_reference.py`, `test_gain_residual_math.py`)
- `tests/layer2/acceptance/` (`test_layer2_acceptance.py`, `requirements_traceability.yaml`)
- Runners: `scripts/run_layer2_set{1..5}.sh` and unified `scripts/run_layer2_validation.sh` (`--config --seed --training-steps --checkpoint-dir --output-dir`, driving real training -> export -> C++ compile/run -> parity -> C++ decoder-in-loop -> paired held-out evaluation -> G1..G9 acceptance JSON).

---

## 6. Section 9 Execution Order (Strictly Enforced)

1. **Step 1 — Inspect & Map Layer 2 Source (`Done`)**: Completed via this document ([`LAYER2_IMPLEMENTATION_TO_TEST_TRACEABILITY.md`](LAYER2_IMPLEMENTATION_TO_TEST_TRACEABILITY.md)) with zero unverified assumptions.
2. **Step 2 — Implement Set 1 (`set1_input_contracts` + `scripts/run_layer2_set1.sh`)**: Validate inputs/actions/sequences/configs and remediate boundary gaps (`DEF-L2-06`, `DEF-L2-08`, `DEF-L2-10`) until **G1 passes 39/39**.
3. **Step 3 — Implement Set 5 Mathematical Oracles (`set5_r2d2_mathematics` + `reference/`)**: Lock in independent math reference before trusting training/eval runs (**G7**).
4. **Step 4 — Implement Set 3 (`set3_training_pipeline`) + CLI (`run_layer2_pipeline.py`)**: Verify live env -> replay -> BPTT/Adam -> checkpoint provenance & resume (**G3, G4**).
5. **Step 5 — Implement Set 4 (`set4_export_cpp_deployment` + C++ test binaries)**: Verify export, FP64 parity, runtime safety, C++ Q7.8/LUT decoder action effect (**G5, G6**), plus explicit `INCONCLUSIVE` reporting for unimplemented INT8/ORT paths (`DEF-L2-02, DEF-L2-03`).
6. **Step 6 — Implement Set 2 (`set2_output_statistics`)**: Run 5+ training seeds & 30+ paired held-out evaluation trajectories with bootstrap CIs (**G2, G8**).
7. **Step 7 — Integrated Layer 2 Acceptance (`scripts/run_layer2_validation.sh`)**: Enforce all Gates **G1–G9** end-to-end and verify zero regressions against frozen baseline `54da68d`.
