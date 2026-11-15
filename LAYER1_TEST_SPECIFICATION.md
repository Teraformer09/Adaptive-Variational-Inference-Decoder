# Layer 1 — Code Model, Noise & Stimulus: Implementation-to-Test Traceability Specification (`LAYER1_TEST_SPECIFICATION.md`)

**Branch Scope:** `layer1-noise-stimulus` (isolated Layer 1 development branch split from frozen integrated baseline `AVID_EPJ_Patches` @ `54da68d`)  
**Frozen Baseline Constraint:** `AVID_EPJ_Patches` (`54da68d`) MUST remain completely unmodified throughout Layer 1 test development and remediation. All specification edits, contract fixes, new test suites, and validation runners execute strictly on `layer1-noise-stimulus`.

---

## 1. Verified Layer 1 Boundary & Codebase Mapping (Section 7 Confirmation)

Before authoring tests, every subsystem in the `AVID_EPJ_Patches` Layer 1 codebase was directly inspected. Rather than assuming an abstract diagram, this specification maps each conceptual stage to actual files, classes, functions, and discovered discrepancies.

### 1.1 Exact Layer 1 Inputs (IP) & Entry Points
1. **Master & Stage YAML Configurations (`configs/noise_framework/*.yaml`, `configs/code/*.yaml`)**
   - Loader: [`load_yaml_validated`](simulation/stimulus/noise_framework/config_schema.py), [`compute_canonical_config_hash`](simulation/stimulus/noise_framework/config_schema.py)
   - Master file: [`configs/noise_framework/framework_v1.yaml`](configs/noise_framework/framework_v1.yaml) (references 12 sub-configs and `contract_freeze`: 16 PDs, 100.0s window, 10.0 Hz, 1000 samples/window, 1 MHz QEC clock, 21 mechanisms/basis).
2. **16 Persistent-Disorder (PD) Profiles (`configs/noise_framework/pd_profiles_16.yaml`)**
   - Registry: [`PDRegistry`](simulation/stimulus/noise_framework/pd_registry.py), dataclasses [`NoiseProfile`](simulation/stimulus/noise_framework/config_schema.py)
   - Sampling/Moments: [`sample_truncated_gaussian`](simulation/stimulus/noise_framework/config_schema.py) (exact inverse-CDF via `_std_normal_cdf` / Acklam `_std_normal_ppf`), [`theoretical_truncated_gaussian_moments`](simulation/stimulus/noise_framework/config_schema.py).
3. **CSS Code Matrices `(H_X, H_Z, L_X, L_Z)` (`architecture/code_model/*.json`, `configs/code/*.yaml`)**
   - Builder & Algebra: [`RotatedSurfaceCode`](simulation/reference/code_model/css_code_builder.py), `build_rotated_d3_code()`, `build_rotated_d5_code()`, `verify_css_code_algebra()`, `gf2_rank()`, `gf2_in_rowspace()`, `build_tanner_tables()`.
   - Canonical `d=3` convention (`[[9,1,3]]`): 9 data qubits (`0..8`), `r_X = 4` X-checks, `r_Z = 4` Z-checks, `X_L = [0,3,6]`, `Z_L = [0,1,2]`, 12 NN `zz_edges` matching `CANONICAL_12_PAIRS` (`config_schema.py`), `21` mechanisms per basis (`p_0..p_8`, `p_01..p_58`).
4. **Master Seed, Simulation Duration & Sampling Clock**
   - Seed derivation: `derive_deterministic_seed(master_seed, *components)` -> SHA-256 first 8 LE bytes masked to 63-bit non-negative `int` (`& 0x7FFFFFFFFFFFFFFF`).
   - Clock & Buffer: [`SamplingClock`](simulation/stimulus/noise_framework/sampling/sampling_clock.py) (`round(t0 + sample_id / rate_hz, 9)`), [`SynchronizedSampler`](simulation/stimulus/noise_framework/sampling/synchronized_sampler.py), [`TrajectoryBuffer`](simulation/stimulus/noise_framework/sampling/trajectory_buffer.py).
5. **Generator & Measurement Parameters (`generator_registry.yaml`, `stage5_measurement_imperfections.yaml`, `measurement_parameter_mapping.yaml`, `generator_assignment.yaml`, `stage6_stratified_splits.yaml`)**
   - Registry: [`GeneratorRegistry`](simulation/stimulus/noise_framework/generator_registry.py) mapping 7 active types (`gaussian`, `ou`, `markov`, `telegraph`, `spatial_correlated`, `burst`, `measurement_noise`).
   - Note: `simulation/stimulus/pd_registry.py`, `generator_registry.py`, `assignment/*.py`, `noise_classes/*.py`, `avid_interface/syndrome_interface.py`, and `seven_stage/config_schema.py` are compatibility re-exports of `simulation/stimulus/noise_framework/...`.

### 1.2 Verified Seven-Stage Execution Architecture & Dataflow
Each stage currently exists as a callable Python module under [`simulation/stimulus/seven_stage/`](simulation/stimulus/seven_stage/) plus Stage 7 under [`simulation/reference/noise_model/`](simulation/reference/noise_model/):

```
Inputs (YAML, 16 PDs, H_X/H_Z, master_seed, duration, policy)
  │
  ├─ Stage 1: stage1_pd_profiles.py :: run_stage1_pd_validation()
  │    PDRegistry.load() -> NoiseProfile[PD_01..PD_16] -> Stage7StatisticalValidator.validate_pd_library()
  │
  ├─ Stage 2: stage2_stochastic_temporal.py :: run_stage2_assignment_and_lifecycle_validation()
  │    GeneratorRegistry.instantiate_all() -> PDAssignment(policy) -> AssignmentScheduler(100.0s)
  │    -> SamplingClock(10Hz) -> SynchronizedSampler (OU/Markov/TLS state persisted across 100s windows)
  │
  ├─ Stage 3: stage3_spatial_correlated.py :: run_stage3_spatial_validation()
  │    SpatialCorrelatedGenerator (logit + common latent u_t factor on 9 singles + CANONICAL_12_PAIRS)
  │
  ├─ Stage 4: stage4_nonstationary_bursts.py :: run_stage4_composition_validation()
  │    BurstNoiseGenerator (overlapping 2-bump burst envelope) + NoiseComposer (21-param & XOR masks)
  │    -> List[CompositeNoiseSample] in TrajectoryBuffer
  │
  ├─ Stage 5: stage5_measurement_noise.py :: run_stage5_measurement_noise_pipeline()
  │    MeasurementNoiseGenerator -> MeasurementNoiseParameters (q_base_x[4], q_base_z[4], sticky, dropout, delay)
  │    -> SampleEncoder -> FaultRealizer (e_X, e_Z, l_X, l_Z) -> SyndromeEncoder (s_X=H_X e_Z, s_Z=H_Z e_X)
  │    -> MeasurementModel (Markov sticky flip, dropout mask -> val_{x,z}, delayed delivery index)
  │    -> AvidInputAdapter (.npz, .meta.json, AVID-SYNDROME-PACKET-V1 .bin via stim_to_bin/packet_codec)
  │
  ├─ Stage 6: stage6_stratified_sampler.py :: run_stage6_stratified_dataset_generation()
  │    Partitions {train, val, test} -> generate_single_stratified_trajectory() -> Provenance + ManifestWriter
  │    (theta_hash disjointness checks across splits)
  │
  └─ Stage 7: stage7_statistical_validator.py :: Stage7StatisticalValidator
       LogicalClassDistributionEngine (32-state XOR-conv DP P(s,l), P(l|s), LER_oracle, aliased pairs)
       + validate_pd_library(), validate_composition_rules(), validate_trajectory_distributions(), evaluate_decoders()
```

### 1.3 Verified Discrepancies, Defects & Undocumented Contracts (Must Remediate / Handle Explicitly)
Inspection of `AVID_EPJ_Patches` uncovered **12 concrete discrepancies / contract gaps** between architecture docs and production code:

| Defect ID | Component | Discrepancy / Defect Observed in Source | Target Resolution Phase |
|---|---|---|---|
| **DEF-L1-01** | `config_schema.py:load_yaml_validated` | Uses `yaml.safe_load`, which **silently overwrites duplicate YAML keys**; performs no `schema_version` validation, no strict unknown-key policy, and no temporal parameter bounds (`rate_hz <= 0`, `tau_sec < 0`, `window_duration <= 0`). | Fix in **Set 1** (custom duplicate-key PyYAML `SafeLoader` + strict schema validators) |
| **DEF-L1-02** | `pd_registry.py:PDRegistry.load` & `NoiseProfile` | Wraps `parameters`/`support`/`metadata` in `MappingProxyType`, then casts immediately back to mutable `dict(...)` (`pd_registry.py:62-64`), allowing caller mutation of registered PD state. Also `NoiseProfile.__post_init__` accepts `bool` for numbers and never validates `temporal_mu_sec`, `temporal_sigma_sec > 0`, `burst_amplitude`, `sticky_scale`, `dropout_scale`, `delay_quantiles`. | Fix in **Set 1** (keep read-only `MappingProxyType` or deep-copy on `.get()`; strict type/range guards) |
| **DEF-L1-03** | `NoiseProfile.sample_values` / `theoretical_moments` | `VALID_DISTRIBUTION_TYPES` permits `"gaussian"`, `"beta"`, `"uniform"`, but `sample_values()` & `theoretical_moments()` ignore `distribution_type` and always run `truncated_gaussian`. | Fix in **Set 1/2** (implement exact Beta/Uniform/Gaussian branches or restrict validated schema cleanly) |
| **DEF-L1-04** | `css_code_builder.py` / code loader | `build_rotated_d3_code()` builds constants programmatically without an entry validator for external JSON/matrix inputs (non-binary entries, wrong rank, `Hx @ Hz.T != 0 mod 2`, dimension mismatch). | Add validated CSS loader/validator in **Set 1** |
| **DEF-L1-05** | Layer 1 Unified CLI Entry Point | Only `stage5_measurement_noise.py` has CLI flags (`--config`, `--duration`, `--seed`, `--output`); no single unified CLI supports `--config`, `--profiles`, `--seed`, `--duration`, `--output-dir` orchestrating Stages 1–7. | Add unified `run_layer1_pipeline.py` CLI in **Set 3/4** |
| **DEF-L1-06** | `composition_rules.yaml` vs `noise_composer.py` | `composition_rules.yaml` specifies `bernoulli_xor_parity_convolution` (`0.5*(1-prod(1-2p))`), whereas `NoiseComposer.compose()` (`noise_composer.py:219,228`) invokes `compose_independent_bernoulli` (`1-prod(1-p)`). Stage 4/7 tests currently assert union rule. | Document canonical formula; align config/code & test exact active rule in **Set 3/5** |
| **DEF-L1-07** | `stage3_spatial_correlations.yaml` vs `CANONICAL_12_PAIRS` | `stage3_spatial_correlations.yaml:11-22` includes diagonal `(5,7)` at `pair_idx=10` and omits `(7,8)`, conflicting with `config_schema.py:CANONICAL_12_PAIRS` & `css_code_builder.py:zz_edges` (`(7,8)`). Generators ignore that YAML. | Reconcile YAML/doc & validate `CANONICAL_12_PAIRS` in **Set 1/5** |
| **DEF-L1-08** | `syndrome_packet_v1.yaml` vs `packet_codec.py` & `stim_to_bin.py` | Spec YAML lists magic `0xAAEC0002` + `start_seq_id` + 32-bit packet, whereas `packet_codec.py` & `stim_to_bin.py` emit `0xAAEC0001` + `is_u64` + 64-bit packet (`FLAG_VALID_PKT=1` hardcoded; `validity_mask_{x,z}` dropped checks not mapped to `.bin` flags; `compute_even_parity_bit()` unused). | Validate active V1 64-bit contract & expose gap in **Set 4/5** |
| **DEF-L1-09** | `measurement_parameter_mapping.yaml` dual schema | Config duplicates rules under `target_mappings` (validated by `MeasurementParameterMapping`) and `mappings` (read by `MeasurementNoiseGenerator`), and generator ignores `use_profile_parameter` for `q_base` + hardcodes quantile logic. | Bind/cross-validate both sections in **Set 1/3** |
| **DEF-L1-10** | `MeasurementModel` pre-start delay clamping | `src_idx = max(0, t - d_obs)` (`measurement_model.py:181`) duplicates sample `0` for `t < d_obs` rather than flagging invalid pre-start packets; also dynamic delay jumps can repeat/skip samples. | Test exact implemented clamping & document boundary in **Set 5** |
| **DEF-L1-11** | Hardcoded `100.0s` / unused Stage 6 YAML controls | `BaseNoiseGenerator._check_ready_and_timestamp` hardcodes `100.0`s window; `stage6_stratified_sampler.py` ignores `ou_volatility_scale` and upper `seed_range[1]` from `stage6_stratified_splits.yaml`; `stage5` hardcodes policy instead of reading `default_policy`. | Validate/resolve in **Set 1/3** |
| **DEF-L1-12** | `Stage7StatisticalValidator` X-only scope | `validate_trajectory_distributions` & decoder benchmark only inspect `syndromes_x` / `theta_z` without checking Z-basis or masking `validity_mask_x == 0`. | Extend/verify both X & Z bases in **Set 3/5** |

---

## 2. Statistical Experiment Tiers, Oracles & Acceptance Gates

### 2.1 Evaluation Profiles
| Profile | Seed Count | Typical Trajectory Length | Purpose |
|---|---|---|---|
| **Profile A (Functional)** | `1–5` seeds | `1–50` s (`10–500` samples) | Fast contract verification, deterministic edge cases, schema rejection, wiring smoke tests. |
| **Profile B (Statistical)** | `>= 30` independent seeds | `250–2000` s (`2,500–20,000` samples/seed or `>= 50,000` Bernoulli trials) | Distribution agreement, pairwise covariances, Markov transitions, OU ACF, block-bootstrap CIs. |
| **Profile C (Extended)** | `>= 100` replications (adaptive `N`) | Rare events down to `p ~ 1e-4` using `N >= ((1-p)/p)*(z_{1-alpha/2}/eps)^2` | Tail/rare dropout/flip checks, sensitivity sweeps, long-horizon memory/leakage stability. |

### 2.2 Machine-Readable Audit Metadata (Required Per Statistical Experiment)
Every statistical test in Sets 2 & 5 must emit JSON audit fields:
`test_id`, `profile_tier`, `master_seed`, `derived_seeds`, `num_trajectories`, `num_samples`, `noise_parameters`, `pd_profile_id`, `empirical_estimate`, `theoretical_estimate`, `ci_low`, `ci_high`, `statistical_test`, `alpha`, `p_value` / `multiple_testing_adjusted_alpha`, `status` (`PASS` | `FAIL` | `INCONCLUSIVE`), `config_hash`, `implementation_git_sha`.

### 2.3 Mandatory Acceptance Gates (`G1`–`G7`) & Three-State Outcomes
- **Statuses:** `PASS` (all checks passed), `FAIL` (violated contract/physics), `INCONCLUSIVE` (underpowered `N`, missing dependency, runtime unavailable).
- **Hard Rule:** `NOT_RUN`, `SKIPPED`, or `INCONCLUSIVE` **MUST NEVER** coerce to `PASS`.

| Gate | Scope Covered | Blocking Failure Consequence |
|---|---|---|
| **G1** | Set 1 (`L1-IP-001..031`) | Block pipeline execution (`run_layer1_validation.sh` aborts) |
| **G2** | Set 2A (`L1-OP-001..010`) | Reject generated dataset structure |
| **G3** | Set 2B (`L1-STAT-001..012`, Level A & B) | Reject statistical validity |
| **G4** | Set 3 (`L1-FLOW-001..028`) | Reject seven-stage integration acceptance |
| **G5** | Set 4 (`L1-EXEC-001..033`) | Reject runtime, persistence & reproducibility |
| **G6** | Set 5 (`L1-MATH-001..040`) | Reject physical & mathematical correctness |
| **G7** | Cross-cutting provenance, traceability matrix & audit JSON completeness | Reject final Layer 1 acceptance |

---

## 3. Master Implementation-to-Test Traceability Matrix

### Set 1 — Input Contract Validation (`tests/layer1/set1_input_contracts/`) — Gate **G1**

#### 1A. YAML Configuration Contracts (`test_yaml_contracts.py`)

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Mathematical / Logical Oracle | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|---|
| **L1-IP-001** | Valid framework config | Canonical YAML files exist under `configs/noise_framework/` | Read canonical `framework_v1.yaml` + programmatic valid generated configs | Call `load_yaml_validated` and sub-config loaders | All schema keys present, exact types (`str`, `float`, `int`, `list`, `dict`) match contract | Loads cleanly without coercion; returns valid dict | Profile A / G1 | `config_schema.py:load_yaml_validated`, `framework_v1.yaml` |
| **L1-IP-002** | Missing required fields | Valid YAML schema baseline | Programmatically drop each required key one at a time across `framework_v1`, `pd_profiles_16`, `generator_registry`, `stage5`, `measurement_mapping`, `stage6` | Call `load_yaml_validated(..., required_keys=...)` & registry `.load()` | Every missing required key $k \in R$ must raise `ValueError`/`KeyError` | Explicit error message naming the exact missing key | Profile A / G1 | `config_schema.py:load_yaml_validated`, `pd_registry.py:PDRegistry.load`, `generator_registry.py:GeneratorRegistry.load` |
| **L1-IP-003** | Invalid parameter types | Valid baseline dicts | Replace `float` with `"0.01"` or `bool`, `dict` with `list`, `int` with `float`/`bool` | Pass mutated dict/YAML into `NoiseProfile`, `GeneratorSpec`, `MeasurementParameterMapping` | Strict isinstance check rejecting `bool` as `int`/`float` and stringified numbers | Rejects invalid type immediately without implicit cast *(exposes DEF-L1-02 until fixed)* | Profile A / G1 | `config_schema.py:NoiseProfile`, `GeneratorSpec`, `measurement_mapping.py` |
| **L1-IP-004** | Boundary values | Schema min/max bounds defined | Generate exact min (`min=0.0`, `delay=0`, `seed=0`), exact max (`max=1.0`, `delay=5`), and epsilon-outside bounds (`-1e-12`, `1+1e-12`) | Instantiate `NoiseProfile`, `GeneratorSpec`, `MeasurementNoiseParameters` | Valid iff $x \in [x_{\min}, x_{\max}]$ and $x_{\min} < x_{\max}$ | Accepts exact closed/half-open valid boundaries; rejects epsilon-outside | Profile A / G1 | `config_schema.py:NoiseProfile`, `measurement_parameters.py:MeasurementNoiseParameters` |
| **L1-IP-005** | Invalid probabilities | Valid `NoiseProfile` & `MeasurementNoiseParameters` | Inject $p=-0.01$, $p=1.01$, `float('nan')`, `float('inf')`, `float('-inf')` | Instantiate `NoiseProfile`, `MeasurementNoiseParameters`, `compose_independent_bernoulli`, `compose_xor_parity_bernoulli` | Probability axiom $p \in [0, 1] \cap \mathbb{R}_{\text{finite}}$ | Explicit `ValueError` for every non-finite or out-of-range value | Profile A / G1 | `config_schema.py`, `measurement_parameters.py`, `probability_composition.py` |
| **L1-IP-006** | Invalid temporal parameters | Temporal configs (`sampling_schedule.yaml`, `generator_assignment.yaml`, `stage5`) | Inject `rate_hz <= 0`, `tau_sec < 0`, `window_duration_sec <= 0`, `t0 < 0`, `temporal_sigma_sec <= 0` | Initialize `SamplingClock`, `AssignmentScheduler`, `OUNoiseGenerator`, `GaussianNoiseGenerator`, config validator | Physical clock requires $f_s > 0, T_w > 0, \tau_c \ge 0, t_0 \ge 0$ | Explicit rejection on invalid temporal config *(remediates DEF-L1-01/11)* | Profile A / G1 | `sampling_clock.py`, `assignment_scheduler.py`, `ou_generator.py`, `config_schema.py` |
| **L1-IP-007** | Unknown configuration keys | Strict config validator enabled | Inject unexpected top-level & nested keys (`rogue_param: 123`) into generated YAML | Run strict schema loader | Keyset $K \subseteq K_{\text{allowed}}$ unless extension block explicitly registered | Explicit rejection or explicit warning/recorded extension per schema policy *(remediates DEF-L1-01)* | Profile A / G1 | `config_schema.py:load_yaml_validated` |
| **L1-IP-008** | Duplicate configuration keys | Synthetic raw YAML string with duplicate key (`seed: 1001\nseed: 9999`) | Write temp YAML containing duplicate YAML mapping keys at top and nested levels | Call `load_yaml_validated(path)` | Unique key invariant $\forall k$, multiplicity $= 1$ | Raises duplicate-key `ValueError` instead of silent last-wins overwrite *(remediates DEF-L1-01)* | Profile A / G1 | `config_schema.py:load_yaml_validated` |
| **L1-IP-009** | Configuration version | Valid YAML baseline | Test `schema_version: "1.0"`, missing `schema_version`, unsupported `"99.0"`, non-string `1.0` | Call validated config loader | Version must match supported set `{"1.0"}` | Accepts `"1.0"`; rejects missing, float, or unsupported schema_version *(remediates DEF-L1-01)* | Profile A / G1 | `config_schema.py:load_yaml_validated` |
| **L1-IP-010** | Configuration round trip | Valid loaded configs | Load -> `yaml.safe_dump` -> reload via `load_yaml_validated` | Compare parsed objects & `compute_canonical_config_hash(a) == compute_canonical_config_hash(b)` | Canonical JSON SHA-256 identity | Exact semantic & hash equality after round trip | Profile A / G1 | `config_schema.py:compute_canonical_config_hash` |

#### 1B. Persistent-Disorder Profile Contracts (`test_pd_profile_contracts.py`)

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Mathematical / Logical Oracle | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|---|
| **L1-IP-011** | Registry initialization | `pd_profiles_16.yaml` | Load canonical registry via `PDRegistry(path).load()` | Inspect `list_profiles()` and `get(pid)` for `PD_01..PD_16` | Set equality $\{\text{PD\_01},\dots,\text{PD\_16}\}$ with cardinality 16 | All 16 IDs resolve cleanly with `config_hash` | Profile A / G1 | `pd_registry.py:PDRegistry.load` |
| **L1-IP-012** | Profile completeness | Valid PD entry template | Programmatically remove `mean`, `std`, `min`, `max`, `variable`, `units`, `temporal_mu_sec`, `dominant_qubits`, etc. | Load mutated YAML in `PDRegistry` / `NoiseProfile` | All required fields defined in schema must exist | Rejects incomplete profile naming omitted attribute *(DEF-L1-02)* | Profile A / G1 | `config_schema.py:NoiseProfile`, `pd_registry.py` |
| **L1-IP-013** | Profile parameter ranges | Valid PD entry template | Perturb `mean < min`, `mean > max`, `std <= 0`, `min >= max`, `max > 1.0`, `dropout_scale < 0`, `delay_quantiles` unsorted | Instantiate `NoiseProfile` / `PDRegistry.load()` | Support $[a,b]\subseteq [0,1]$, $a \le \mu \le b$, $\sigma > 0$, quantiles monotone in $(0,1)$ | Explicit `ValueError` for every violation *(DEF-L1-02)* | Profile A / G1 | `config_schema.py:NoiseProfile.__post_init__`, `pd_registry.py:validate` |
| **L1-IP-014** | Duplicate profile IDs | Valid 16-profile YAML | Duplicate `PD_01` (with 16 total entries replacing `PD_16`, and 17 total entries) | Load in `PDRegistry` | Unique ID bijection to `PD_01..PD_16` | Explicit duplicate-ID `ValueError` | Profile A / G1 | `pd_registry.py:PDRegistry.load` |
| **L1-IP-015** | Profile selection | Loaded `PDRegistry` | Request each `PD_01..PD_16` + unknown `"PD_99"` | Call `registry.get(pid)` | Exact match on `profile_id` and YAML numerical params; unknown raises `KeyError` | Exact match returned; `"PD_99"` raises `KeyError` | Profile A / G1 | `pd_registry.py:PDRegistry.get` |
| **L1-IP-016** | Profile composition | Multiple generators assigned distinct profiles via `PDAssignment` + `AssignmentScheduler` | Assign distinct valid combinations (same PD across generators when allowed, different PDs per window) and invalid unknown policy/profile | Instantiate `PDAssignment(..., allowed_profile_ids=...)` + `NoiseComposer` | Normalized weights $\sum w_i = 1$; unknown `allowed_profile_ids` rejected; composed parameters obey composition rules | Valid subsets normalize cleanly; invalid IDs or all-zero weights raise `ValueError` | Profile A / G1 | `assignment/pd_assignment.py`, `composition/noise_composer.py` |
| **L1-IP-017** | Profile immutability | Loaded `PDRegistry` | Attempt `prof.parameters["mean"] = 0.99`, `prof.metadata["dominant_qubits"].append(8)`, `prof.mean = 0.5` | Mutate returned `NoiseProfile` and re-fetch `registry.get(pid)` | Deep read-only `MappingProxyType` + tuples/copies | Raises `TypeError`/`FrozenInstanceError`; source unchanged *(DEF-L1-02 fix)* | Profile A / G1 | `config_schema.py:NoiseProfile`, `pd_registry.py:PDRegistry.load` |
| **L1-IP-018** | Profile serialization | Loaded `PDRegistry` | Serialize all 16 `NoiseProfile` specs back to YAML dict -> reload into fresh `PDRegistry` | Compare `to_dict()`, theoretical moments, and `config_hash` | Canonical hash equality + $\Delta \mu, \Delta \sigma^2 < 10^{-15}$ | Exact parameter, ID, and moment preservation | Profile A / G1 | `pd_registry.py`, `config_schema.py` |

#### 1C. CSS Code-Model Input Contracts (`test_css_code_contracts.py`)

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Mathematical / Logical Oracle | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|---|
| **L1-IP-019** | Load $H_X, H_Z$ | `architecture/code_model/{Hx,Hz,logical_operators}.json` & `build_rotated_d3_code()` | Load JSON matrices & compare with `d=3` and `d=5` builders | Call validated CSS code loader & `verify_css_code_algebra()` | $H_X \in \{0,1\}^{4\times 9}, H_Z \in \{0,1\}^{4\times 9}$ (d=3); entries $\in \{0,1\}$ | Exact shape, dtype, and JSON-vs-builder equality | Profile A / G1 | `css_code_builder.py`, `architecture/code_model/*.json` |
| **L1-IP-020** | Binary matrix validation | Valid JSON/array copy | Inject `-1`, `2`, `0.5`, `NaN` into $H_X$ or $H_Z$ | Pass to CSS code validator *(DEF-L1-04)* | $\forall i,j,\; H_{ij} \in \{0, 1\}$ strictly | Rejects non-binary matrix explicitly | Profile A / G1 | `css_code_builder.py` (new input validator) |
| **L1-IP-021** | CSS commutation | Valid `d=3` & `d=5` codes + perturbed non-commuting pair | Test valid `(Hx, Hz)` and toggle 1 entry of `Hx` breaking orthogonality | Evaluate `(Hx @ Hz.T) % 2` via `verify_css_code_algebra()` and validator | $H_X H_Z^T \equiv 0 \pmod 2$ | Passes for `d=3, 5`; rejects non-commuting pair | Profile A / G1 | `css_code_builder.py:verify_css_code_algebra` |
| **L1-IP-022** | Dimension consistency | Valid code matrices | Alter `Hz` columns from `9` to `8`, or `logical_X` columns to `7`, or swap `n_x_checks` | Pass to CSS validator / `SyndromeEncoder` | $\text{cols}(H_X) = \text{cols}(H_Z) = \text{cols}(L_X) = \text{cols}(L_Z) = n$ | Explicit shape-mismatch `ValueError` | Profile A / G1 | `css_code_builder.py`, `syndrome_encoder.py`, `fault_realizer.py` |
| **L1-IP-023** | Rank consistency | Valid `d=3` & `d=5` matrices | Test canonical codes and linear-dependent row replacement ($H_X[3] \leftarrow H_X[0] \oplus H_X[1]$) | Compute `gf2_rank(Hx)`, `gf2_rank(Hz)`, `k = n - rank_X - rank_Z` | $k = n - \operatorname{rank}_{\mathbb{F}_2}(H_X) - \operatorname{rank}_{\mathbb{F}_2}(H_Z)$ must equal expected $k=1$ | Canonical passes ($k=1, r_X=r_Z=4$); rank-deficient detected | Profile A / G1 | `css_code_builder.py:gf2_rank,compute_exact_distance` |
| **L1-IP-024** | Matrix corruption | Valid `architecture/code_model/*.json` | Flip single bit in `Hx` or `Hz` or truncate `logical_operators.json` | Run `verify_css_code_algebra()` + `compute_exact_distance()` | Detects broken CSS commutation, anticommutation $L_X L_Z^T \neq 1$, or distance drop $< d$ | Explicit validation failure with diagnostic field flags | Profile A / G1 | `css_code_builder.py:verify_css_code_algebra` |
| **L1-IP-025** | Matrix serialization | Valid `RotatedSurfaceCode` | Serialize `Hx, Hz, logical_X, logical_Z` to JSON/NPZ -> reload | Compare arrays bit-for-bit | $\max |A - A'| = 0$ and identical algebra check | Exact binary matrix equality | Profile A / G1 | `css_code_builder.py`, `architecture/code_model/*.json` |

#### 1D. Runtime Initialization Contracts (`test_runtime_initialization.py`)

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Mathematical / Logical Oracle | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|---|
| **L1-IP-026** | Master seed | Canonical configs | Supply valid positive integer seeds (`1`, `42`, `2026`) | Call `derive_deterministic_seed`, `GeneratorRegistry.instantiate_all`, `PDAssignment`, `SampleEncoder` | Every derived seed deterministic 63-bit integer $\ge 0$; unique per `(component, id, window)` | All RNG components initialize deterministically | Profile A / G1 | `config_schema.py:derive_deterministic_seed`, `generator_registry.py` |
| **L1-IP-027** | Seed boundaries | Valid generator/assignment factories | Test `seed=0`, `seed=-1`, `seed=2**31-1`, `seed=2**64`, `seed="42"`, `seed=None` | Pass to `GeneratorSpec`, `derive_deterministic_seed`, `GeneratorRegistry`, CLI | Signed/unsigned valid range $0 \le s < 2^{63}$; non-int or negative rejected | `0` & valid max accepted cleanly; `-1`, strings, `None` rejected cleanly | Profile A / G1 | `config_schema.py:GeneratorSpec`, `generator_registry.py` |
| **L1-IP-028** | Duration | Valid initialized pipeline (`rate_hz=10.0`) | Test short (`0.1`s), fractional (`0.3`s), nominal (`250.0`s), zero (`0.0`s), negative (`-5.0`s) | Execute `SamplingClock` / `SynchronizedSampler.generate_trajectory(duration)` | $N = \operatorname{round}(T \cdot f_s)$ for valid $T > 0$; reject $T \le 0$ | Exact $N \in \{1, 3, 2500\}$; rejects $T \le 0$ with `ValueError` | Profile A / G1 | `synchronized_sampler.py:generate_trajectory`, `sampling_clock.py` |
| **L1-IP-029** | Sampling frequency | Valid configs | Test `10.0` Hz, `20.0` Hz, `1.0` Hz, invalid `0.0` Hz, `-10.0` Hz | Initialize `SamplingClock(0.0, rate_hz, window_duration)` and verify timestamps $t_k = t_0 + k/f_s$ | $t_k - t_{k-1} = 1/f_s$ within $10^{-9}$ s; $f_s \le 0$ raises `ValueError` | Valid clocks exact; non-positive rates rejected | Profile A / G1 | `sampling_clock.py:SamplingClock` |
| **L1-IP-030** | Component initialization | Registry with 7 active generators | Remove `measurement_noise_01` or register unknown `generator_type: "bogus"` or `seed: null` | Call `GeneratorRegistry.load()` / `stage5` / `SynchronizedSampler` | Registered generator type must exist in `GENERATOR_TYPE_MAP`; Stage 5 requires `measurement_noise_01` | Raises explicit `KeyError`/`ValueError` naming missing/unknown item | Profile A / G1 | `generator_registry.py`, `stage5_measurement_noise.py` |
| **L1-IP-031** | Initialization order | Uninitialized `BaseNoiseGenerator` / `AssignmentScheduler` | Call `gen.sample(0.0, 0)` before `assign_profile()`; call `scheduler.advance(0.0)` before `initialize()` | Invoke stage/component methods out of dependency order | Precondition `assigned_profile is not None` and `initialized == True` | Raises explicit `RuntimeError` rejecting premature execution | Profile A / G1 | `base_generator.py`, `assignment_scheduler.py` |

---

### Set 2 — Output Contract and Statistical Validation (`tests/layer1/set2_output_statistics/`) — Gates **G2, G3**

#### 2A. `CompositeNoiseSample` Output Contract (`test_composite_sample_schema.py`) — Gate **G2**

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Mathematical / Logical Oracle | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|---|
| **L1-OP-001** | Sample schema | Valid 7-gen pipeline | Generate 500 samples via `SynchronizedSampler(..., composer=NoiseComposer())` & `SampleEncoder` | Assert attributes on every `CompositeNoiseSample` & encoded dict keys | Required attributes (`sample_id, timestamp, window_id, contributions, composite_parameters, metadata`) & all 22 trajectory dict keys present | Every sample/array conforms strictly | Profile A / G2 | `config_schema.py:CompositeNoiseSample`, `sample_encoder.py` |
| **L1-OP-002** | Physical fault dimensions | `d=3` code (`n=9`) | Run `SampleEncoder` on 2500 composite samples (`rounds_per_composite_sample=1` and `2`) | Check `fault_masks_x.shape`, `fault_masks_z.shape` | Exact shape `(N_rounds, 9)` matching `code.n_data` | Exact shape & `uint8` dtype | Profile A / G2 | `fault_realizer.py`, `sample_encoder.py` |
| **L1-OP-003** | Fault-mask domain | Same as above | Inspect all entries of `fault_masks_x`, `fault_masks_z`, `measurement_faults_{x,z}` | Assert `np.all((arr == 0) | (arr == 1))` | Elementwise membership in $\mathbb{F}_2 = \{0,1\}$ | Zero non-binary values | Profile A / G2 | `fault_realizer.py`, `measurement_model.py` |
| **L1-OP-004** | Syndrome dimensions | `d=3` code (`r_X=r_Z=4`) | Run `SampleEncoder` and `AvidInputAdapter` | Check `ideal_syndromes_{x,z}`, `syndromes_{x,z}`, `detector_events_{x,z}` | Exact shape `(N_rounds, 4)` and $\in \{0,1\}$ | Exact dimension and binary domain | Profile A / G2 | `syndrome_encoder.py`, `sample_encoder.py` |
| **L1-OP-005** | Validity-mask dimensions | Stage 5 measurement active | Inspect `validity_mask_x`, `validity_mask_z` alongside `syndromes_{x,z}` | Compare shapes & domains | `val.shape == syn.shape == (N_rounds, 4)` and $\in \{0,1\}$; dropped entries have `syn == 0` where `val == 0` | Exact match on all checks | Profile A / G2 | `measurement_model.py`, `sample_encoder.py` |
| **L1-OP-006** | Timestamp monotonicity | Multi-window run (`250`s) | Inspect `timestamps`, `event_timestamps`, `delivery_timestamps` across window boundaries | Assert $\Delta t \ge 0$ strictly increasing for `timestamps` and `event_timestamps`, and $t_{\text{del}} \ge t_{\text{ev}}$ | $t_0 < t_1 < \dots < t_{N-1}$ and $t_{\text{del}, i} = t_{\text{ev}, i} + d_i \Delta t$ | Strictly monotonic event grid, non-negative causal delivery offset | Profile A / G2 | `sampling_clock.py`, `measurement_model.py` |
| **L1-OP-007** | Sampling interval | `rate_hz=10.0` (and custom `20.0`) | Compute diffs `np.diff(timestamps)` over `2500` samples | Assert max abs deviation from `1/rate_hz` | $\max_k |(t_k - t_{k-1}) - 1/f_s| \le 10^{-9}\text{ s}$ | Interval invariant holds across window boundaries | Profile A / G2 | `sampling_clock.py`, `synchronized_sampler.py` |
| **L1-OP-008** | Missing-sample handling | Trajectory with `detector_dropout_prob > 0` | Force dropout regime (`PD_16` / high dropout config) | Check `validity_mask == 0` explicitly marks dropped checks and packet `AVID` checks handle invalid flags | Dropped observation represented explicitly by `validity_mask == 0`, never conflated with valid `0`-syndrome | Valid vs dropped 0-bits distinguishable in trajectory metadata/NPZ *(packet flag gap documented in DEF-L1-08)* | Profile A / G2 | `measurement_model.py:apply`, `avid_input_adapter.py` |
| **L1-OP-009** | Metadata integrity | Multi-generator run | Inspect every `NoiseContribution.metadata`, `CompositeNoiseSample.metadata`, and `.meta.json` | Verify `profile_id`, `generator_id`, `seed`, `timestamp`, `theta_hash`, `config_hashes` | Non-empty attribution covering all active generators & provenance hashes | Full metadata chain retained end-to-end | Profile A / G2 | `noise_composer.py`, `sample_encoder.py`, `provenance.py` |
| **L1-OP-010** | Sample serialization | Generated `CompositeNoiseSample` list + encoded trajectory | Save via `AvidInputAdapter.write_npz` + JSON provenance -> reload | Compare arrays (`np.array_equal` / `allclose(..., 0)`), dtypes, shapes, and JSON metadata | Zero semantic loss under round trip | Bitwise array & metadata identity preserved | Profile A / G2 | `avid_input_adapter.py`, `manifest_writer.py` |

#### 2B. Statistical Distribution Validation (`test_fault_distributions.py`, `test_measurement_statistics.py`, `test_temporal_statistics.py`, `test_cross_qubit_correlations.py`) — Gate **G3**

> Statistical Rigor Contract: All tests run across **Profile B (`>= 30` seeds)** or **Profile C (`>= 100` seeds)** with exact Clopper-Pearson binomial CIs / chi-square GoF / block-bootstrap ACF CIs + Holm-Bonferroni correction, saving full JSON audit records. Correlated runs are never tested using naive i.i.d. binomial SEs.

| ID | Test | Preconditions | Input-Generation Method | Execution Procedure | Mathematical / Logical Oracle | Acceptance Criterion | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|---|
| **L1-STAT-001** | Single-qubit X distribution | `FaultRealizer` isolated X-basis | Sweep $\theta_X[0..8]$ across nominal (`0.005`), moderate (`0.03`), boundary (`0.0`, `0.25`) over `30+` seeds (`N` sized by $\epsilon=0.002$) | Realize faults & compare empirical $\hat{p}_{X,q}$ vs exact marginal $p_{\text{eff},X,q}$ (accounting for incident edges!) | Under NN edge $e=(q,r)$, effective marginal fault on $q$ is XOR parity of single + incident edges; when pair=0, $\hat{p}_{X,q} \sim \text{Binomial}(N, p_q)/N$ | Exact Clopper-Pearson 99% CI contains theoretical rate (Holm-Bonferroni adjusted across qubits/regimes) | Profile B/C / G3 | `fault_realizer.py:realize_single_basis` |
| **L1-STAT-002** | Single-qubit Z distribution | `FaultRealizer` isolated Z-basis | Same sweep over $\theta_Z[0..8]$ across `30+` seeds | Realize Z faults & check empirical single-qubit marginal rates | Same exact marginal parity oracle | Exact Clopper-Pearson 99% CI coverage across `>= 30` seeds | Profile B/C / G3 | `fault_realizer.py:realize_single_basis` |
| **L1-STAT-003** | Pairwise ZZ distribution | `FaultRealizer` with active 12 edge rates $\theta_{Z}[9:21] \neq 0$ | Vary all 12 pair parameters in `CANONICAL_12_PAIRS` across `30+` seeds | Estimate joint `P(e_i=1, e_j=1)` and centered covariance $\operatorname{Cov}(e_i, e_j)$ | Exact closed-form two-qubit marginal induced by common pair flips + incident singles/edges | Empirical joint $2\times 2$ distribution passes Chi-square GoF ($\alpha_{\text{adj}}$) & 99% CI contains covariance | Profile B/C / G3 | `fault_realizer.py`, `spatial_correlated_generator.py` |
| **L1-STAT-004** | Measurement-flip distribution | `MeasurementModel` with `sticky=0`, varying `q_base` | Sweep `q_base` in `{1e-4 (Level B), 0.005, 0.05, 0.2}` across `30–100` seeds ($N$ sized via binomial planning formula) | Count `measurement_faults_{x,z}` events against `q_base` | Independent $\text{Binomial}(N, q_{\text{base}})$ when `sticky=0`; stationary Markov $\pi_1 = \frac{q}{1 - s(1-q)}$ when `sticky>0` | Exact Clopper-Pearson CI contains theoretical flip rate; Level B rare $10^{-4}$ uses $N \ge 3\times 10^5$ | Profile B/C / G3 | `measurement_model.py:apply` |
| **L1-STAT-005** | Dropout distribution | `MeasurementModel` with `dropout` sweep | Sweep `detector_dropout_prob` in `{0.0, 2e-4 (Level B), 0.01, 0.1}` across `30–100` seeds | Count `validity_mask == 0` over rounds & checks | $\text{Binomial}(N \cdot 8, p_{\text{drop}})$ independent across checks/rounds | Clopper-Pearson CI contains `p_drop`; zero dropouts verify exact count `0` | Profile B/C / G3 | `measurement_model.py:apply` |
| **L1-STAT-006** | Burst duration | `BurstNoiseGenerator` isolated | Sweep `temporal_sigma_sec` (`3.5`, `10.0`) & kernels (`gaussian_bell`, `fast_attack_exp_decay`, `rectangular_window`) over `30` seeds | Compute FWHM / above-threshold interval (`in_burst_window == True` & $p_k$ elevation width) within each 100s window | Kernel-predicted support interval where $w_1 g_1(t) + 0.65 w_2 g_2(t)$ exceeds threshold | Observed active duration matches kernel oracle within sampling tolerance; multi-kernel profiles match formula | Profile B / G3 | `generators/burst_generator.py` |
| **L1-STAT-007** | Burst arrival / phase rate | `PDAssignment` + `BurstNoiseGenerator` | Run long multi-window streams (`>= 800` windows across `30` seeds) under `uniform_16` vs `curriculum_burst_enriched` | Compute empirical category/window arrival frequency of burst PDs vs policy weights | Categorical($\pi$) Chi-square test comparing window-level PD frequencies with normalized policy $\pi$ | Chi-square GoF $p > \alpha_{\text{adj}}$; window duration invariants verified | Profile B/C / G3 | `stage2_stochastic_temporal.py`, `pd_assignment.py`, `burst_generator.py` |
| **L1-STAT-008** | Temporal autocorrelation | `OUNoiseGenerator` (`tau_sec` $\in \{2, 8, 25\}$), `TelegraphNoiseGenerator`, `MarkovNoiseGenerator`, sticky measurement | Generate multi-window trajectories (`30+` seeds, long $T \gg \tau_c$) at fixed PD | Estimate empirical ACF $\hat{\rho}(k)$ and move-block bootstrap 95%/99% CIs (never i.i.d. SE!) | OU: $\rho(\Delta t) = e^{-\Delta t / \tau_c}$; TLS: $e^{-(r_\uparrow+r_\downarrow)\Delta t}$; Sticky meas: $(1-q)\cdot \text{sticky}$ | Theoretical decay curve lies within block-bootstrap CI across lags $k=1..20$ | Profile B / G3 | `ou_generator.py`, `telegraph_generator.py`, `measurement_model.py` |
| **L1-STAT-009** | Cross-qubit correlation | `SpatialCorrelatedGenerator` vs `GaussianNoiseGenerator` (independent) | Run 30 seeds on `PD_12` (`dominant_pairs=[1,4]`) with `SpatialCorrelatedGenerator` vs single-qubit Gaussian | Compute pairwise Pearson correlation matrix across 21 params and realized qubits `(1,2),(6,7)` vs non-adjacent qubits `(0,8)` | Independent: $|\rho_{ij}|$ bounded by $O(1/\sqrt{N})$; Spatial factor model: strong positive correlation $\rho > 0.70$ on dominant pair latent logits | Spatial correlation significantly exceeds independent null ($p < 10^{-6}$); independent cross-pairs un-correlated | Profile B / G3 | `spatial_correlated_generator.py`, `stage3_spatial_correlated.py` |
| **L1-STAT-010** | Cross-axis correlation | Full pipeline with distinct `X` and `Z` generators | Generate 30 trajectories tracking `theta_x`, `theta_z`, `fault_masks_x`, `fault_masks_z` (and conditionally within fixed window) | Measure empirical $\operatorname{Cov}(e_X, e_Z)$ conditioned on fixed `(PD_X, PD_Z)` window vs unconditional | Within fixed window, X and Z fault generators have independent RNG streams & parameters $\implies \operatorname{Cov}(e_{X,i}, e_{Z,j} \mid W) \approx 0$ | Conditional X-Z zero-correlation hypothesis not rejected across 30 seeds | Profile B / G3 | `noise_composer.py`, `fault_realizer.py` |
| **L1-STAT-011** | PD-conditioned rates | `PD_01..PD_16` via inverse-CDF truncated Gaussian | Sample all 16 PDs over `30` seeds x `25,000` draws + condition full-pipeline rates on active window PD | Compare conditional empirical mean, variance, KS-statistic, and quantiles vs `theoretical_truncated_gaussian_moments` & CDF | Exact truncated-normal PDF/CDF on $[a,b]$; conditional `PD_15` rate $<$ `PD_16` rate | All 16 PDs pass Kolmogorov-Smirnov test + moment CI checks | Profile B / G3 | `config_schema.py`, `stage1_pd_profiles.py`, `stage7_statistical_validator.py` |
| **L1-STAT-012** | Mixed-noise distribution | Simultaneous active Z/X/meas generators | Compose 2, 3, and all 4 data generators via `NoiseComposer` + realize syndromes over `30` seeds | Compare realized empirical marginal & joint mechanism rates against exact active composition law | Active rule: `1 - prod(1 - p_j)` (vs `composition_rules.yaml` XOR-parity discrepancy DEF-L1-06 explicitly tested against actual function!) | Empirical composite parameter & fault rates fall within CI of theoretical composed vector | Profile B / G3 | `noise_composer.py`, `probability_composition.py`, `fault_realizer.py` |

---

### Set 3 — Pipeline and System-Flow Validation (`tests/layer1/set3_pipeline_flow/`) — Gate **G4**

#### 3A. Stage-to-Stage IP/OP Boundary Contracts (`test_stage_boundaries.py`)

| ID | Boundary | Preconditions | Input-Generation Method | Execution Procedure | Verification / Bypass Detection | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|---|
| **L1-FLOW-001** | Configuration $\rightarrow$ Stage 1 | Modified temp YAML configs (`pd_profiles`, `generator_registry`) | Perturb one PD mean/std and one generator param in temp YAML | Run framework init / `run_stage1_pd_validation(custom_yaml)` | Verify `PDRegistry.config_hash` & initialized `NoiseProfile.mean` reflect modified YAML, NOT default file | Mutated YAML parameter propagates directly into Stage 1 objects | Profile A / G4 | `config_schema.py`, `pd_registry.py`, `stage1_pd_profiles.py` |
| **L1-FLOW-002** | Stage 1 $\rightarrow$ Stage 2 | Loaded custom `PDRegistry` | Register perturbed PD profile into Stage 2 scheduler | Run `PDAssignment` + `AssignmentScheduler` + `SynchronizedSampler` | Verify Stage 2 generator contributions cite exact `PDRegistry` profile IDs & perturbed mean | Stage 2 consumes live Stage 1 objects | Profile A / G4 | `stage2_stochastic_temporal.py`, `assignment_scheduler.py` |
| **L1-FLOW-003** | Stage 2 $\rightarrow$ Stage 3 | Initialized `AssignmentScheduler` with 2+ windows (`200s`) | Run trajectory crossing $t = 100.0\text{ s}$ | Trace active `profile_id` and persistent `x_state`/`tls_states`/`current_state` into `SpatialCorrelatedGenerator` & temporal gens | Verify PD window assignment flips at $t=100.0\text{ s}$ (if drawn) AND generator continuous internal state persists without unexpected reset | PD state drives fault parameters across boundary | Profile A / G4 | `stage2_stochastic_temporal.py`, `stage3_spatial_correlated.py` |
| **L1-FLOW-004** | Stage 3 $\rightarrow$ Stage 4 | Active generators emitting `NoiseContribution` | Collect uncomposed contributions vs `NoiseComposer.compose()` | Compare individual `(generator_id, profile_id, values)` inside `CompositeNoiseSample.contributions` with composed `theta_z/x` | Recompute composed `theta_z` independently from `sample.contributions` and verify equality to `sample.composite_parameters["theta_z"]` | Contributions preserve provenance IDs and compose without drop/overwrite | Profile A / G4 | `synchronized_sampler.py`, `noise_composer.py`, `stage4_nonstationary_bursts.py` |
| **L1-FLOW-005** | Stage 4 $\rightarrow$ Stage 5 | Composed `CompositeNoiseSample` stream + `MeasurementNoiseParameters` | Encode via `SampleEncoder` using both ideal and Stage 5 measurement params | Verify `ideal_syndromes_{x,z}` equals `(H @ e) % 2` from Stage 4 fault realization, then `MeasurementModel` operates ONLY on `ideal_syndromes` | Perturb `measurement_noise_01` and assert `fault_masks_{x,z}` & `ideal_syndromes_{x,z}` stay bitwise identical while `syndromes_{x,z}` change! | Exact causal isolation of measurement noise from physical faults | Profile A / G4 | `sample_encoder.py`, `fault_realizer.py`, `syndrome_encoder.py`, `measurement_model.py` |
| **L1-FLOW-006** | Stage 5 $\rightarrow$ Stage 6 | Stage 5 noisy trajectory (`syndromes`, `validity_mask`, `delays`) | Run Stage 6 (`generate_single_stratified_trajectory`) & serialize `.npz` / `.bin` | Verify `.npz` & `.bin` contain Stage 5 corrupted `syndromes`, `validity_mask`, and Stage 6 does NOT bypass back to `ideal_syndromes` | Inject known measurement flip or dropout on Stage 5; verify Stage 6 `.npz` & `.bin` reflect the corrupted syndrome (never noiseless bypass) | Stage 6 consumes exact Stage 5 outputs | Profile A / G4 | `stage5_measurement_noise.py`, `stage6_stratified_sampler.py`, `avid_input_adapter.py` |
| **L1-FLOW-007** | Stage 6 $\rightarrow$ Stage 7 | Saved Stage 6 trajectory artifacts | Pass generated Stage 6 trajectory into `Stage7StatisticalValidator` | Verify Stage 7 uses runtime `theta_z`, `syndromes`, `fault_masks` to compute empirical stats vs 32-state oracle | Compare Stage 7 metrics against independent calculation over saved Stage 6 arrays | Stage 7 evaluates actual run trajectory, not canned fixtures | Profile A / G4 | `stage6_stratified_sampler.py`, `stage7_statistical_validator.py` |
| **L1-FLOW-008** | Stage 7 $\rightarrow$ Output | Completed Stage 5–7 run | Inspect JSON reports (`validation_report.json`, `statistical_metrics.json`, `trajectory_manifest.json`) | Cross-check report `config_hashes`, `seed`, `theta_hash`, `summary_metrics` by recomputing from `.npz`/`.bin` | Every numeric summary & SHA-256 hash matches reloaded artifacts to float precision `1e-12` | Reports authentically reflect the current run | Profile A / G4 | `stage5_measurement_noise.py`, `manifest_writer.py`, `stage7_statistical_validator.py` |

#### 3B. Full Pipeline Execution Cases (`test_seven_stage_execution.py`)

| ID | Test | Preconditions | Execution Method | Oracle / Check | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|---|
| **L1-FLOW-009** | Nominal seven-stage run | Canonical configs | Execute unified Stages 1–7 pipeline on nominal `duration=250.0`s, `seed=2026` | Stage order telemetry `[Stage1, Stage2, Stage3, Stage4, Stage5, Stage6, Stage7]`, zero errors, all gates green | Full connected execution & artifacts produced | Profile A / G4 | All `stage1..6` + `stage7_statistical_validator.py` |
| **L1-FLOW-010** | Minimal-duration run | Canonical configs | Run shortest duration `0.1`s (`1` sample) and `1.0`s (`10` samples) | Exact window count `1`, zero empty-slice exceptions, finite summary stats | Clean execution with exact `1`/`10` samples | Profile A / G4 | `synchronized_sampler.py`, `sample_encoder.py`, `stage5` |
| **L1-FLOW-011** | Extended-duration run | Canonical configs | Run extended `1200.0`s (`12,000` samples across `12` windows) | Monotonic sample IDs `0..11999`, window IDs `0..11`, stable RSS memory & consistent statistics | No corruption, overflow, or resource leak | Profile B / G4 | `synchronized_sampler.py`, `trajectory_buffer.py`, `sample_encoder.py` |
| **L1-FLOW-012** | Multiple PD profiles | Specific single-PD allowed policies | Execute sequential runs restricting `allowed_profile_ids` to `[PD_01]`, `[PD_07]`, `[PD_15]`, `[PD_16]` | Each trajectory assigns exclusively its restricted profile and matches that PD's mean | Exact PD isolation per trajectory | Profile A / G4 | `pd_assignment.py`, `stage6_stratified_sampler.py` |
| **L1-FLOW-013** | Mixed-profile run | Train/Val/Test stratified policies | Execute `run_stage6_stratified_dataset_generation` generating multi-profile windows | Verify `ManifestWriter` provenance confirms distinct `theta_hash` across `train`, `val`, `test` | Strict split disjointness + multi-PD transitions | Profile A / G4 | `stage6_stratified_sampler.py`, `manifest_writer.py` |
| **L1-FLOW-014** | Measurement imperfections enabled | `stage5_measurement_imperfections.yaml` enabled | Execute full pipeline with active `measurement_noise_01` | `realized_measurement_flip_rate > 0`, `observed != ideal` on flipped indices, dropout masks present | Stage 6 & 7 consume corrupted stream | Profile A / G4 | `stage5_measurement_noise.py`, `measurement_model.py` |
| **L1-FLOW-015** | Measurement imperfections disabled | Custom config disabling Stage 5 (`ideal_noiseless` baseline / `q_base=0, sticky=0, dropout=0, delay=0`) | Run pipeline with explicitly disabled/zeroed measurement imperfections | Assert `syndromes == ideal_syndromes` bitwise (`val == 1` everywhere, `delays == 0`) | Zero corruption emitted when disabled | Profile A / G4 | `measurement_parameters.py`, `measurement_model.py` |
| **L1-FLOW-016** | Correlated-noise run | `zz_crosstalk_01` active (`PD_12`) | Run Stage 3 + composer + encoder pipeline | Downstream `fault_masks_z` shows significant pair co-occurrence on `(1,2)` & `(6,7)` edges | Pair correlation preserved into realized syndromes | Profile B / G4 | `spatial_correlated_generator.py`, `fault_realizer.py` |
| **L1-FLOW-017** | Burst-noise run | `burst_noise_01` active | Run Stage 4 + encoder with single window (`100`s) | Sample fault counts inside `in_burst_window == True` significantly exceed out-of-window baseline | Burst timing preserved in realized faults & syndromes | Profile B / G4 | `burst_generator.py`, `noise_composer.py` |
| **L1-FLOW-018** | Pipeline interruption | Run harness with simulated termination after Stage 4 | Interrupt subprocess / raise termination hook before Stage 6/7 | Check exit code != 0 and validation/acceptance report does NOT report `PASS` | Incomplete run fails loudly; never marked successful | Profile A / G4 | Layer 1 acceptance runner |
| **L1-FLOW-019** | Stage failure propagation | Hooked fault injector per stage | Inject deterministic exception in Stage 1, 2, 3, 4, 5, 6, 7 sequentially | Verify exception propagates cleanly with non-zero exit & stage attribution | Every single stage failure aborts pipeline cleanly | Profile A / G4 | Unified runner & `stage1..7` |
| **L1-FLOW-020** | Pipeline restart | Existing stale output directory | Create stale fake output JSON/NPZ in `--output-dir` -> rerun pipeline cleanly with new seed | Compare all output timestamps, hashes, & sample arrays against new seed run | Fresh artifacts overwrite/isolate cleanly; zero stale reuse | Profile A / G4 | `stage5`, `stage6`, `avid_input_adapter.py` |

#### 3C. Data Lineage, Propagation & Bypass Detection (`test_data_lineage.py`, `test_runtime_parameter_propagation.py`)

| ID | Test | Method | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L1-FLOW-021** | Runtime provenance | Inject unique run ID / temp timestamp into pipeline invocation and trace across manifest, NPZ meta, trajectory manifest, Stage 7 report | One consistent `run_id` & `config_hashes` end-to-end | Profile A / G4 | `provenance.py`, `manifest_writer.py`, `stage5` |
| **L1-FLOW-022** | Input configuration propagation | Alter a supported parameter (e.g., scale readout `scale_factor_x` from `0.85` to `0.10` or `tau_sec` from `8.0` to `0.5`) in temp YAML | Measurable directional change in downstream generated output matching model prediction | Profile A/B / G4 | `measurement_noise_generator.py`, `ou_generator.py` |
| **L1-FLOW-023** | PD propagation | Switch PD policy/allowed pool from `PD_15` (`mean=0.0018`) to `PD_16` (`mean=0.0125`) | Downstream fault rate & syndrome weight jump strictly upward matching PD ratio | Profile A / G4 | `pd_assignment.py`, `stage6`, `stage7` |
| **L1-FLOW-024** | Measurement propagation | Modify `stage5` / `measurement_parameter_mapping.yaml` readout rate with identical master seed | `syndromes_{x,z}` & `measurement_faults` change in predicted direction while `fault_masks_{x,z}` & `ideal_syndromes` remain **bitwise identical** | Profile A / G4 | `sample_encoder.py`, `measurement_model.py` |
| **L1-FLOW-025** | Seed propagation | Run with `seed=2026` vs `seed=2027` on same YAML | Bitwise distinct fault/syndrome streams; repeat `seed=2026` matches first run identically | Profile A / G4 | `derive_deterministic_seed`, all generators |
| **L1-FLOW-026** | Static-output detection | Pre-seed fake static fixture data matching known dummy patterns -> run pipeline into temp dir & also check no file under `configs/.../baselines/` is ever loaded by default Stage 5 | Runtime outputs differ from static baselines & match dynamic config hash; `baselines/` excluded unless explicitly requested | Profile A / G4 | `stage5_measurement_noise.py`, `measurement_parameters.py` |
| **L1-FLOW-027** | Stage invocation trace | Instrument/spy real registered stage methods (`PDRegistry.load`, `GeneratorRegistry.instantiate_all`, `AssignmentScheduler.advance`, `NoiseComposer.compose`, `MeasurementModel.apply`, `Stage7StatisticalValidator.*`) during full run | Every required stage records `call_count >= 1` in strict causal order | Profile A / G4 | Integrated Layer 1 runner |
| **L1-FLOW-028** | Output-consumption trace | Spy consumer inputs (`NoiseComposer.compose` args, `SampleEncoder.encode_composite_trajectory` args, `AvidInputAdapter.write_npz`, `Stage7StatisticalValidator`) | Verify `object`/hash identity matches exact prior stage return values (no shadow bypass objects) | Profile A / G4 | Integrated Layer 1 runner |

---

### Set 4 — Execution, Persistence and Reproducibility (`tests/layer1/set4_execution_reproducibility/`) — Gate **G5**

#### 4A. Execution Tests (`test_clean_execution.py`)

| ID | Test | Execution Method | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L1-EXEC-001** | Clean environment | Run subprocess with clean minimal `PYTHONPATH=.` & verify imports from `dependencies/requirements.txt` (`numpy`, `scipy`, `yaml`, `pytest`) without repo cross-layer code (`training/`, `firmware/`, `software/`) | Clean import & Stage 1–7 initialization using zero Layer 2–5 modules | Profile A / G5 | Layer 1 Python packages |
| **L1-EXEC-002** | Independent entry point | Execute official Layer 1 CLI runner (`scripts/run_layer1_pipeline.py` / `stage5_measurement_noise.py` subprocess) with zero interactive input | Exits `0`; writes all expected reports, NPZ, BIN, and manifests *(DEF-L1-05)* | Profile A / G5 | `run_layer1_pipeline.py` (to add), `stage5_measurement_noise.py` |
| **L1-EXEC-003** | CLI configuration | Pass custom temp `--config` & `--profiles` pointing to non-default YAML | Output provenance `framework_sha256` & `pd_registry_sha256` match custom YAML, not repo default | Profile A / G5 | CLI `--config`, `--profiles` |
| **L1-EXEC-004** | CLI seed override | Run CLI with `--seed 777` vs `--seed 888` | Manifests record exact CLI seed and derived generator/encoder streams differ cleanly | Profile A / G5 | CLI `--seed` |
| **L1-EXEC-005** | CLI duration override | Run CLI with `--duration 10`, `--duration 50`, `--duration 250` | Exact sample counts `100`, `500`, `2500` in NPZ, BIN header `n_rounds`, and JSON report | Profile A / G5 | CLI `--duration` |
| **L1-EXEC-006** | Invalid CLI arguments | Pass `--duration -10`, `--seed abc`, `--unknown-flag`, missing value | Clean non-zero exit code (`2` or `1`) + descriptive stderr | Profile A / G5 | `argparse` CLI |
| **L1-EXEC-007** | Missing input file | Pass nonexistent `--config /tmp/does_not_exist.yaml` or `--profiles /tmp/missing.yaml` | Non-zero exit code; zero fallback to default repo configs | Profile A / G5 | `load_yaml_validated`, CLI |
| **L1-EXEC-008** | Missing output directory | Pass nested non-existent `--output-dir /tmp/new/nested/dir` AND read-only parent `/proc/invalid_dir` | Creates writable nested dir cleanly; fails non-zero when uncreatable | Profile A / G5 | CLI / `mkdir(parents=True)` |
| **L1-EXEC-009** | Long execution | Run extended duration (`2000.0`s = `20,000` samples) tracking peak RSS (`tracemalloc` / `resource.getrusage`) & wall time | Bounded $O(N)$ memory matching array size; zero quadratic growth or leak | Profile B / G5 | `synchronized_sampler.py`, `sample_encoder.py` |
| **L1-EXEC-010** | Repeated execution | Run pipeline 5 times sequentially inside same Python interpreter and via subprocess with different configs/seeds | Zero global state bleed (`PDRegistry`, caches, numpy legacy global RNG never relied upon) | Profile A / G5 | All Layer 1 stateful classes |

#### 4B. Artifact Integrity (`test_artifact_integrity.py`)

| ID | Artifact | Integrity Oracle / Check | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L1-EXEC-011** | NPZ trajectory | Reload `observed_syndromes.npz`, `measurement_parameters.npz`, `measurement_ground_truth.npz`, `stage5_trajectory_*.npz` | Verify every required key, shape `(N, ...)`, `uint8`/`int32`/`float64` dtype, finite values, `{0,1}` domains | Profile A / G5 | `stage5`, `avid_input_adapter.py` |
| **L1-EXEC-012** | Binary packet stream | Parse raw `.bin` bytes with independent `struct` unpacker (NOT `packet_codec.py`) | Verify 16B LE header (`magic=0xAAEC0001`, `n_rounds=N`, `is_u64=1`, `n_check_x=4`, `n_check_z=4`, `d=3`), exact `16 + 8*N` file size, `[3:0]=s_x`, `[7:4]=s_z`, `[63:32]=0..N-1` | Profile A / G5 | `packet_codec.py`, `stim_to_bin.py`, `avid_input_adapter.py` |
| **L1-EXEC-013** | Metadata | Inspect `trajectory_manifest.json`, `.meta.json`, `assignment_manifest.json` | Contains `seed`, active PDs, SHA-256 config hashes, code geometry hash, `theta_hash`, `run_id` matching live inputs | Profile A / G5 | `provenance.py`, `manifest_writer.py` |
| **L1-EXEC-014** | Stage reports | Inspect `stage1..7` / `validation_report.json` | Stage timestamps, executed flags, check booleans reflect actual runtime execution | Profile A / G5 | `stage1..5`, `stage7_statistical_validator.py` |
| **L1-EXEC-015** | Statistical report | Read `statistical_metrics.json`, then recompute every mean, std, flip rate, dropout count, max delay, per-window mean directly from reloaded `.npz` arrays | Independent numpy recomputation matches saved JSON to `< 1e-12` (`0` integer diff for counts) | Profile A / G5 | `stage5_measurement_noise.py:276-312` |
| **L1-EXEC-016** | Oracle report | Read Stage 7 report (`marginal_s`, `ler_oracle`, `empirical_ler_oracle`) and verify using independent Set 5 brute-force reference on `theta` | Matches reference probability table and `LER_opt` to `< 1e-12` | Profile A / G5 | `stage7_statistical_validator.py`, `logical_class_distribution.py` |
| **L1-EXEC-017** | Manifest | Read `seven_stage_v1_manifest.json` / layer manifest and recompute SHA-256 on every `.npz` and `.bin` file on disk | Every listed file exists and SHA-256 matches `hashlib.sha256` byte-for-byte | Profile A / G5 | `manifest_writer.py` |
| **L1-EXEC-018** | Artifact reload | Reload saved `.npz` + `.bin` + JSON and feed back through `AvidInputAdapter.validate_stream` and `Stage7StatisticalValidator` | Full semantic parity with in-memory `encoded_traj` | Profile A / G5 | `avid_input_adapter.py`, `stage7_statistical_validator.py` |

#### 4C. Reproducibility and RNG Isolation (`test_seed_reproducibility.py`, `test_rng_isolation.py`)

| ID | Test | Method | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L1-EXEC-019** | Same-seed reproducibility | Execute entire pipeline twice with `seed=2026` | Bitwise identical `.npz` arrays, `.bin` packet bytes, and JSON numerical outputs | Profile A / G5 | `derive_deterministic_seed`, `SynchronizedSampler`, `SampleEncoder` |
| **L1-EXEC-020** | Different-seed variation | Execute with 5 distinct seeds (`101, 202, 303, 404, 505`) | All trajectories differ in PD sequence, fault masks, and syndromes while remaining valid | Profile A / G5 | All seeded components |
| **L1-EXEC-021** | Generator RNG isolation | Run with `7` canonical generators, then add an extra 8th independent generator `"extra_test_gen_01"` (`seed=9999`) keeping master seed fixed | Original 7 generators emit **bitwise identical** `NoiseContribution.values` and `PDAssignment` sequences! (Check/fix `PDAssignment` per-generator keyed derivation!) | Profile A / G5 | `generator_registry.py`, `pd_assignment.py`, `base_generator.py` |
| **L1-EXEC-022** | Stage-order isolation | Reorder generator registration order in YAML list (preserving IDs & seeds) and swap X-check vs Z-check sampling order where independent | Generator `values` and X vs Z `FaultRealizer` streams stay unaffected by YAML list permutation or extra unlinked generators | Profile A / G5 | `generator_registry.py`, `pd_assignment.py`, `fault_realizer.py` |
| **L1-EXEC-023** | Process reproducibility | Spawn two separate OS subprocesses (`python3 -m ...`) executing full pipeline with same seed & config | Byte-identical `.bin` & array-identical `.npz` across separate OS processes | Profile A / G5 | CLI & subprocess runner |
| **L1-EXEC-024** | Environment provenance | Record `python_version`, `numpy_version`, `scipy_version`, `pyyaml_version`, `git_sha` in acceptance/run metadata | Environment manifest present & matches actual runtime `sys.version` / `numpy.__version__` | Profile A / G5 | Runner & `provenance.py` |
| **L1-EXEC-025** | Checkpoint/restart | Export `state_dict()` at sample $t=500$ (`OU`, `Markov`, `TLS`, `Measurement`) -> instantiate fresh generators with `load_state_dict()` + exact RNG state -> resume to $t=1000$ vs unbroken 1000-step run | Bitwise / exact numerical continuity matching unbroken trajectory (test documented RNG snapshot contract) | Profile A / G5 | `base_generator.py:state_dict,load_state_dict`, generators |

#### 4D. Failure and Recovery (`test_failure_recovery.py`)

| ID | Test | Failure Injection | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|
| **L1-EXEC-026** | Corrupt YAML | Invalid YAML syntax (`:\n  [unclosed`) | Non-zero exit / clean explicit YAML error; no partial `PASS` | Profile A / G5 | `config_schema.py:load_yaml_validated` |
| **L1-EXEC-027** | Corrupt profile file | Malformed PD profile YAML (wrong count, invalid types, NaN) | Clean explicit `ValueError`; runner fails at G1 | Profile A / G5 | `pd_registry.py` |
| **L1-EXEC-028** | Corrupt CSS matrix | Malformed JSON (`Hx.json` broken syntax or wrong row count) | Clean explicit validation failure before simulation | Profile A / G5 | CSS code validator |
| **L1-EXEC-029** | Disk-write failure | Pass un-writable read-only `--output-dir` (`chmod 0555`) | Raises `PermissionError` / exits non-zero; never reports success | Profile A / G5 | `stage5`, `avid_input_adapter.py`, runner |
| **L1-EXEC-030** | Interrupted run | Simulate mid-run crash leaving incomplete staging/partial files | Acceptance validator marks run `FAIL`/`INCOMPLETE` (never `PASS`) | Profile A / G5 | Acceptance runner |
| **L1-EXEC-031** | Partial packet file | Truncate valid `.bin` stream mid-packet (`size - 3` bytes) and corrupt header `n_rounds` | `AvidInputAdapter.validate_stream` & independent decoder raise error / report `valid=False` | Profile A / G5 | `avid_input_adapter.py:validate_stream`, `packet_codec.py` |
| **L1-EXEC-032** | Partial NPZ file | Corrupt `.npz` archive (truncate bytes / remove `"syndromes_x"` key) | Reload raises `EOFError`/`BadZipFile`/`KeyError` cleanly | Profile A / G5 | `AvidInputAdapter`, integrity check |
| **L1-EXEC-033** | Failed Stage 7 | Inject statistical / oracle check failure in Stage 7 | Overall runner (`run_layer1_validation.sh` / acceptance) returns non-zero and blocks G6/G7 | Profile A / G5 | `stage7_statistical_validator.py`, acceptance runner |

---

### Set 5 — Layer-Specific Mathematical & Physical Validation (`tests/layer1/set5_noise_mathematics/`) — Gate **G6**

> Oracle Independence Rule: Reference implementations live in `tests/layer1/set5_noise_mathematics/reference/` and compute GF(2) syndromes, truncated-normal moments, composition laws, and exhaustive $2^{21}$ state enumerations from first principles **without importing `simulation/reference/noise_model/` or `simulation/stimulus/.../syndrome_encoder.py`**.

#### 5A. Physical Fault and Syndrome Mathematics (`test_syndrome_algebra.py`)

| ID | Property | Input / Variation | Independent Reference Oracle (`reference/`) | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L1-MATH-001** | Syndrome parity | 1,000 random binary $(e_X, e_Z) \in \mathbb{F}_2^9 \times \mathbb{F}_2^9$ (covering all $2^9=512$ vectors!) | Independent bitwise GF(2) dot product: $s_{X,c} = \bigoplus_{q \in \text{supp}(H_{X,c})} e_{Z,q}$, $s_{Z,c} = \bigoplus_{q \in \text{supp}(H_{Z,c})} e_{X,q}$ | $100\%$ exact agreement with `SyndromeEncoder.encode()` across all 512 single-basis patterns | Profile A / G6 | `syndrome_encoder.py:SyndromeEncoder` |
| **L1-MATH-002** | Zero-error identity | $e_X = \mathbf{0}_9, e_Z = \mathbf{0}_9$ | $H \cdot \mathbf{0} \bmod 2 = \mathbf{0}_4$ | Exact $s_X = \mathbf{0}_4, s_Z = \mathbf{0}_4$, $l_X=0, l_Z=0$ | Profile A / G6 | `syndrome_encoder.py`, `fault_realizer.py` |
| **L1-MATH-003** | Single-qubit error | All 9 single-qubit vectors $\mathbf{u}_q$ ($q=0..8$) in X and Z | Column $q$ of $H_X$ (for $Z_q$) and column $q$ of $H_Z$ (for $X_q$) | $s_X(\mathbf{u}_q) = H_X[:, q]$, $s_Z(\mathbf{u}_q) = H_Z[:, q]$ for all $q \in \{0..8\}$ | Profile A / G6 | `syndrome_encoder.py`, `css_code_builder.py` |
| **L1-MATH-004** | Multi-qubit error | Random pairs/triples $e_1, e_2 \in \mathbb{F}_2^9$ | $s(e_1 \oplus e_2) = s(e_1) \oplus s(e_2) \pmod 2$ | GF(2) linearity holds exhaustively | Profile A / G6 | `syndrome_encoder.py`, `pauli_composition.py` |
| **L1-MATH-005** | Error cancellation | $m_1 = m_2 = e$ passed to `compose_pauli_masks_xor([m_1, m_2])` | $e \oplus e = \mathbf{0}_9 \implies s(e \oplus e) = \mathbf{0}_4$ | Zero residual error mask and zero syndrome | Profile A / G6 | `pauli_composition.py`, `stage7:validate_composition_rules` |
| **L1-MATH-006** | CSS commutation | All $d=3$ & $d=5$ code stabilizers & generators | Verify $\langle r_i, c_j \rangle_{\mathbb{F}_2} = 0$ for all $(X\text{-check}_i, Z\text{-check}_j)$ and $(L_X, H_Z)$, $(L_Z, H_X)$; and $\langle L_X, L_Z \rangle = 1$ | Zero commutator everywhere between checks; anticommutation on logical operators | Profile A / G6 | `css_code_builder.py:verify_css_code_algebra` |
| **L1-MATH-007** | Stabilizer-equivalent errors | Arbitrary $e$ + all $2^4=16$ stabilizer coset offsets $g \in \operatorname{rowspace}(H_X)$ (or $H_Z$) | Verify $s(e \oplus g) = s(e)$ and $\ell(e \oplus g) = \ell(e)$ (since $L$ commutes with stabilizers!) | Every stabilizer-equivalent fault has identical syndrome & identical logical class | Profile A / G6 | `css_code_builder.py`, `fault_realizer.py` |
| **L1-MATH-008** | Logical equivalence | $e$ vs $e \oplus L_Z$ (or $L_X$) | Verify $s(e \oplus L) = s(e)$ while $\ell(e \oplus L) = 1 - \ell(e)$ | Identical syndrome but flipped logical class (verifies coset separation) | Profile A / G6 | `fault_realizer.py`, `logical_class_distribution.py` |

#### 5B. Persistent-Disorder Mathematical Properties (`test_persistent_disorder.py`)

| ID | Property | Input / Setup | Reference Calculation | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L1-MATH-009** | State persistence | Multi-window 10 Hz run (`500`s = `5` x `100`s windows) | Check `profile_id` constancy for `[0, 100), [100, 200), ...` AND continuous `x_state`/`tls_states` equality at $t=99.9\text{ s} \to 100.0\text{ s}$ | $100\%$ PD invariance inside each 100s window + state continuity confirmed | Profile A / G6 | `assignment_scheduler.py`, `stage2_stochastic_temporal.py` |
| **L1-MATH-010** | Profile separation | All 16 PDs tested pairwise | Compute KL divergence / Wasserstein distance / Welch $t$-statistic across `mean`, `std`, `temporal_mu`, `dominant_pairs` | Distinct PDs (`PD_15` vs `PD_16`, `PD_01` vs `PD_07`) show theoretically predicted mean separation | Profile B / G6 | `pd_profiles_16.yaml`, `config_schema.py` |
| **L1-MATH-011** | Temporal correlation | Isolated OU (`tau_sec`), TLS (`rate_up, rate_down`), Markov (`T`) over $10^5$ steps | Exact OU: $C(\tau) = \sigma^2_{\text{eff}} e^{-\tau/\tau_c}$; TLS: $e^{-(r_\uparrow + r_\downarrow)\tau}$; Markov: spectral eigenvalues $\lambda_2^k, \lambda_3^k$ | Empirical normalized ACF matches exact exponential / matrix-power law within block-bootstrap CIs | Profile B / G6 | `ou_generator.py`, `telegraph_generator.py`, `markov_generator.py` |
| **L1-MATH-012** | Stationary behaviour | Fixed `PD_08` OU/Markov trajectory split into first vs second half (`N=50,000` each) | Two-sample KS test + lag-window mean/variance comparison across halves | Half-1 vs Half-2 distributions statistically indistinguishable ($p > 0.01$) | Profile B / G6 | `ou_generator.py`, `markov_generator.py` |
| **L1-MATH-013** | Non-stationary behaviour | Fixed window `BurstNoiseGenerator` + `GaussianNoiseGenerator` (time-varying envelope at `temporal_mu_sec`) | Compare empirical mean rate at $t \approx \mu_t \pm \sigma_t/2$ (peak) vs $t$ far from $\mu_t$ (baseline) against exact analytic burst sigmoid formula | Peak rate exceeds tail baseline by predicted ratio and rejects constant-rate null ($p < 10^{-6}$) | Profile B / G6 | `burst_generator.py`, `gaussian_generator.py` |
| **L1-MATH-014** | Transition behaviour | `MarkovNoiseGenerator` ($3\times 3$ matrix $T$) & `TelegraphNoiseGenerator` ($r_\uparrow, r_\downarrow$) | Estimate empirical $\hat{P}_{ij} = N_{ij}/\sum_k N_{ik}$ over `200,000` transitions; check $99\%$ multinomial/binomial CIs vs $T_{ij}$ & $1-e^{-r\Delta t}$ | All 9 Markov entries & 2 TLS transition probabilities fall within $99\%$ theoretical CIs | Profile B/C / G6 | `markov_generator.py`, `telegraph_generator.py` |
| **L1-MATH-015** | Profile switching | `AssignmentScheduler(..., preserve_generator_state=True)` vs `preserve_generator_state=False` across `100.0s` boundary | Compare `x_state(t=100.0)` immediately after window flip: `True` preserves continuous prior `x_state` pulled smoothly toward new $\mu_{\text{new}}$; `False` calls `gen.reset()` | Exact adherence to documented continuity vs reset semantics | Profile A / G6 | `assignment_scheduler.py:advance`, `ou_generator.py` |
| **L1-MATH-016** | Long-run behaviour | Run `MarkovNoiseGenerator` & `TelegraphNoiseGenerator` (`500,000` steps) | Compute left eigenvector $\pi T = \pi$ (`np.linalg.eig(T.T)`) and TLS stationary $\pi_1 = p_{01}/(p_{01}+p_{10})$ | Empirical state occupancies match stationary vector $\pi$ within $99\%$ Markov-chain CLT bounds | Profile C / G6 | `markov_generator.py`, `telegraph_generator.py` |

#### 5C. Correlated Noise and Joint-Distribution Correctness (`test_correlated_noise.py`)

| ID | Property | Input / Setup | Independent Reference Formula | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L1-MATH-017** | Independent mechanisms | Pure single-qubit `theta = [p_0, p_1, ..., 0, ..., 0]` (`theta[9:]=0`) in `FaultRealizer` | $P(e_i=1, e_j=1) = p_i p_j$, $P(X,Y) = P(X)P(Y)$ | Empirical $2\times 2$ contingency passes Fisher/Chi-square independence; $|\hat{P}(1,1) - p_i p_j|$ in CI | Profile B / G6 | `fault_realizer.py` |
| **L1-MATH-018** | Correlated mechanisms | Active single + pair `theta` on qubits `(0,1)` with pair `p_{01} > 0` | Exact closed form: $P(e_0, e_1)$ induced by independent XOR events $(A_0, A_1, B_{01})$ summed over hidden combinations | Empirical 4-cell distribution `(00,01,10,11)` passes Chi-square GoF vs analytical $(P_{00}, P_{01}, P_{10}, P_{11})$ | Profile B / G6 | `fault_realizer.py`, `logical_class_distribution.py` |
| **L1-MATH-019** | ZZ pair generation | `SpatialCorrelatedGenerator` + `FaultRealizer` over `12` edges `CANONICAL_12_PAIRS` | Compare realized simultaneous flips on each canonical NN edge vs non-edge diagonal pairs | Canonical NN edges match configured pair rates; non-canonical pairs have only accidental $p_i p_j$ coincidence | Profile B / G6 | `spatial_correlated_generator.py`, `fault_realizer.py` |
| **L1-MATH-020** | Pairwise covariance | Analytical two-qubit model $(A_0 \oplus B_{01}, A_1 \oplus B_{01})$ | Exact $\operatorname{Cov}(e_0, e_1) = p_{01}(1-p_{01})(1-2p_0)(1-2p_1)$ | Empirical sample covariance matches exact formula within bootstrap 99% CI | Profile B / G6 | `fault_realizer.py` |
| **L1-MATH-021** | Multi-mechanism composition | Arbitrary generator vectors $p^{(1)}, \dots, p^{(G)}$ | Compare `compose_independent_bernoulli` against $1-\prod_j(1-p_j)$ AND `compose_xor_parity_bernoulli` against $0.5(1-\prod_j(1-2p_j))$ (`< 1e-14`), and explicitly assert documented `NoiseComposer` behavior (`DEF-L1-06`) | Exact machine-precision algebraic match (`< 1e-14`) on both implemented composition functions | Profile A / G6 | `probability_composition.py`, `noise_composer.py` |
| **L1-MATH-022** | X/Z mechanism interaction | Full dual-basis `FaultRealizer` with fixed `(theta_x, theta_z)` | Joint $(e_X, e_Z)$ independence given `(theta_x, theta_z)`: $P(e_X, e_Z \mid \theta) = P(e_X \mid \theta_x) P(e_Z \mid \theta_z)$ | Conditional X vs Z contingency test passes independence across `30` seeds | Profile B / G6 | `fault_realizer.py:realize_faults` |
| **L1-MATH-023** | Boundary probabilities | $p_k = 0.0$ everywhere; $p_k = 1.0$ on subset; $p_{01} = 0.5$ | Exact deterministic outcomes (`0` flips at $p=0$; certainty at $p=1$; exact $50/50$ marginal at $p=0.5$) | Zero exceptions; exact degenerate distributions match oracle | Profile A / G6 | `fault_realizer.py`, `probability_composition.py`, `logical_class_distribution.py` |
| **L1-MATH-024** | Probability normalization | Joint tables $P(e_0,e_1)$, $P(s,l)$, $P(s)$ across 10 random valid `theta` vectors | $\sum P = 1.0$, $\min P \ge 0.0$ within `< 1e-12` | Exact unit normalization & non-negativity | Profile A / G6 | `logical_class_distribution.py`, `joint_dp_likelihood.py` |

#### 5D. Measurement-Noise Mathematical Validation (`test_measurement_noise.py`)

| ID | Property | Input / Setup | Reference Calculation | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L1-MATH-025** | Measurement XOR | Ideal `s_ideal` with known injected `m`, `val=1`, `delay=0` | `s_obs == s_ideal ^ m` bitwise on both X & Z; AND `fault_masks_{x,z}` unchanged! | $100\%$ bitwise XOR accuracy; physical fault masks strictly invariant | Profile A / G6 | `measurement_model.py` |
| **L1-MATH-026** | Flip probability | `sticky=0` sweep & `sticky>0` Markov flip chain (`N=100,000` rounds/check) | Independent $P(m_t=1)=q$; Markov stationary $P(m_t=1) = \frac{q}{q + (1-s)(1-q)}$ & transition $P(1|1) = (1-s)q+s$ | Empirical marginal & conditional transition rates match formulas within 99% CI | Profile B/C / G6 | `measurement_model.py:124-142` |
| **L1-MATH-027** | Dropout semantics | Set `dropout = 0.25, q_base = 0.0, delay = 0` on non-zero `s_ideal` | Where `val == 0`: observed check forced to `0` (`(s ^ m) & val`) AND `val == 0` flag is preserved; where `val == 1`: `s_obs == s_ideal` | Missing data strictly flagged by `val == 0`, never presented as valid zero measurement | Profile A / G6 | `measurement_model.py:160-175` |
| **L1-MATH-028** | Validity propagation | Encode into Stage 5 `.npz` (`observed_syndromes.npz`) & inspect `validity_mask_{x,z}` & `detector_events_{x,z}` | Reload saved Stage 5 artifacts and confirm invalid `(t, c)` indices persist identically | 100% parity between `MeasurementModel` output and persisted `observed_syndromes.npz` masks *(noting DEF-L1-08 `.bin` flag gap)* | Profile A / G6 | `stage5_measurement_noise.py`, `observed_syndromes.npz` |
| **L1-MATH-029** | Measurement independence | `sticky=0, delay=0, dropout=0`, uniform `q_base=0.05` across `100,000` rounds | Lag-1 ACF $\rho(1) \approx 0$ and cross-check correlation $\approx 0$ across all 8 checks | Zero serial or spatial correlation beyond $99\%$ null CI | Profile B / G6 | `measurement_model.py` |
| **L1-MATH-030** | Measurement temporal correlation & delay | Fixed `sticky` $\in \{0.15, 0.50\}$ and step delay `d_obs` $\in \{0, 1, 2\}$ | Empirical lag-1 ACF of `m_t` matches theoretical $P(1|1)-q = s(1-q)$; delayed array satisfies `s_obs[t] == s_undelayed[max(0, t-d)]` & `t_del = t_ev + d*dt` | Exact delay indexing verified round-by-round; sticky ACF matches theory *(explicitly asserting DEF-L1-10 clamping at `t < d_obs`)* | Profile B / G6 | `measurement_model.py:124-210` |

#### 5E. Stage 7 Exact Logical-Class & Brute-Force Oracle (`test_joint_logical_distribution.py`, `test_exact_oracle.py`, `reference/exact_bruteforce_oracle.py`)

| ID | Test | Input / Setup | Independent Brute-Force Oracle (`reference/`) | Expected Result | Tier / Gate | Codebase Mapping |
|---|---|---|---|---|---|---|
| **L1-MATH-031** | Logical-class assignment | Exhaustive $2^9 = 512$ qubit vectors + random `21`-mechanism vectors | Directly evaluate $(L_X \cdot e_Z) \bmod 2$ / $(L_Z \cdot e_X) \bmod 2$ and compare against `LogicalClassDistributionEngine` mechanism table | $100\%$ of 512 vectors & 21 mechanisms assign identical `(s_int, l_bit)` | Profile A / G6 | `logical_class_distribution.py`, `fault_realizer.py` |
| **L1-MATH-032** | Syndrome-class partition | All $2^9 = 512$ patterns per basis | Partition all 512 patterns into $(s, l) \in \{0..15\} \times \{0,1\}$ buckets; verify each bucket has exactly $2^{9-4-1} = 16$ stabilizer-coset elements | Every one of the 32 cells has exact size $16$ ($32 \times 16 = 512$) | Profile A / G6 | `logical_class_distribution.py` |
| **L1-MATH-033** | Joint distribution | Symmetric uniform $\theta$, asymmetric biased $\theta$, pair-dominated $\theta$ | Exact independent $2^{21}$ mechanism enumeration (or meet-in-the-middle $2^{11} \times 2^{10}$ exact convolution in `reference/` NOT calling DP) | $\max |P_{\text{DP}}(s,l) - P_{\text{ref}}(s,l)| < 10^{-13}$ over all 32 entries | Profile A / G6 | `logical_class_distribution.py:compute_joint_distribution` |
| **L1-MATH-034** | Marginal syndrome distribution | Same random + structured $\theta$ vectors | Compare $\sum_{l \in \{0,1\}} P_{\text{DP}}(s, l)$ vs independent 16-state exact reference AND `CorrelatedNoiseModel.marginal_syndrome_distribution()` | Both agree within `< 10^{-13}` across all 16 syndromes | Profile A / G6 | `logical_class_distribution.py`, `joint_dp_likelihood.py` |
| **L1-MATH-035** | Conditional logical distribution | Valid positive-support $\theta$ vectors | Compare $P_{\text{DP}}(l \mid s)$ vs $P_{\text{ref}}(s, l) / \sum_{l'} P_{\text{ref}}(s, l')$ | Max abs error `< 10^{-12}` and $\sum_l P(l \mid s) = 1$ for all $s$ | Profile A / G6 | `logical_class_distribution.py:compute_posterior_and_oracle_ler` |
| **L1-MATH-036** | Normalization | 20 randomized + boundary $\theta \in [0, 0.48]^{21}$ | Check $\sum_{s=0}^{15} \sum_{l=0}^1 P(s, l) = 1.0$, $\min \ge 0.0$ | Deviation $< 10^{-13}$ across all test vectors | Profile A / G6 | `logical_class_distribution.py`, `joint_dp_likelihood.py` |
| **L1-MATH-037** | Oracle LER | 10 varied $\theta$ vectors + sampled Monte Carlo verification (`500,000` FaultRealizer rounds) | Compare DP `ler_oracle` vs $\sum_s \min\{P_{\text{ref}}(s,0), P_{\text{ref}}(s,1)\}$ AND verify Monte Carlo optimal-coset decoder empirical LER converges within CI of $LER_{\text{opt}}$ | Analytic match `< 10^{-13}`; MC empirical oracle LER within $99\%$ binomial CI | Profile A/B / G6 | `logical_class_distribution.py`, `stage7_statistical_validator.py` |
| **L1-MATH-038** | Brute-force comparison | Subset models with $m \le 12$ active mechanisms ($2^{12}=4096$ full enumeration in `< 5` ms!) AND full $2^{21}$ exact split enumeration | Naive loop over every binary vector $u \in \{0,1\}^m$ accumulating $\prod p_k^{u_k}(1-p_k)^{1-u_k}$ into `ref[s(u), l(u)]` (zero shared DP code!) | Exact agreement `< 10^{-13}` for every cell `(s, l)` (`16x2`) | Profile A / G6 | `logical_class_distribution.py`, `joint_dp_likelihood.py:joint_distribution_bruteforce` |
| **L1-MATH-039** | Degenerate noise model | $\theta = \mathbf{0}_{21}$; pure X stabilizer fault (`H_X` row-equivalent prob); single-qubit certainty $p_4 = 1.0$ | Direct discrete deterministic evaluation | $\theta=\mathbf{0} \implies P(0,0)=1.0, LER_{\text{opt}}=0.0$; deterministic fault concentrates mass $1.0$ in exact single cell $(s(e), \ell(e))$ | Profile A / G6 | `logical_class_distribution.py`, `joint_dp_likelihood.py` |
| **L1-MATH-040** | Logical ambiguity (Section 5F Identifiability Limit!) | Construct syndrome-aliased pair `(theta_A, theta_B)` via `construct_syndrome_aliased_pair()` + exact stabilizer-equivalent gauge shift on `theta` where $P(s|\theta_A) \equiv P(s|\theta_B)$! | Run **both**: (1) Stage 7 Joint Oracle `P(s,l)` on `(theta_A, theta_B)` & (2) Syndrome-only L-BFGS (`CorrelatedNoiseModel.fit_lbfgs` given ONLY histogram `s` without `l` or `e`) | **Proves mathematical boundary:** (1) Stage 7 joint oracle separates $P(l|s, \theta_A) \neq P(l|s, \theta_B)$ & distinguishes $LER_{\text{opt}}$; (2) Syndrome-only likelihood `NLL(s | \theta_A) \approx NLL(s | \theta_B)` cannot break the gauge without hidden `(e, l)` access! | Profile A / G6 | `logical_class_distribution.py`, `joint_dp_likelihood.py`, `mathematical_model/proofs/identifiability_proof.md` |

---

## 4. Target Test Directory & Execution Runner Architecture

```
layer1-noise-stimulus/
├── LAYER1_TEST_SPECIFICATION.md          # This traceability specification (Deliverable 1)
├── simulation/stimulus/seven_stage/
│   └── run_layer1_pipeline.py            # Unified Stages 1–7 CLI (--config --profiles --seed --duration --output-dir) [Set 3/4]
├── tests/layer1/
│   ├── conftest.py                       # Shared audit logger, Profile A/B/C tier helpers, temp YAML generators
│   ├── set1_input_contracts/
│   │   ├── test_yaml_contracts.py        # L1-IP-001 .. L1-IP-010
│   │   ├── test_pd_profile_contracts.py  # L1-IP-011 .. L1-IP-018
│   │   ├── test_css_code_contracts.py    # L1-IP-019 .. L1-IP-025
│   │   ├── test_runtime_initialization.py# L1-IP-026 .. L1-IP-031
│   │   └── fixtures/
│   ├── set2_output_statistics/
│   │   ├── test_composite_sample_schema.py   # L1-OP-001 .. L1-OP-010
│   │   ├── test_fault_distributions.py       # L1-STAT-001..003, 011..012
│   │   ├── test_measurement_statistics.py    # L1-STAT-004..005
│   │   ├── test_temporal_statistics.py       # L1-STAT-006..008
│   │   ├── test_cross_qubit_correlations.py  # L1-STAT-009..010
│   │   └── fixtures/
│   ├── set3_pipeline_flow/
│   │   ├── test_stage_boundaries.py              # L1-FLOW-001 .. L1-FLOW-008
│   │   ├── test_seven_stage_execution.py         # L1-FLOW-009 .. L1-FLOW-020
│   │   ├── test_data_lineage.py                  # L1-FLOW-021, 025..028
│   │   ├── test_runtime_parameter_propagation.py # L1-FLOW-022 .. L1-FLOW-024
│   │   └── fixtures/
│   ├── set4_execution_reproducibility/
│   │   ├── test_clean_execution.py       # L1-EXEC-001 .. L1-EXEC-010
│   │   ├── test_artifact_integrity.py    # L1-EXEC-011 .. L1-EXEC-018
│   │   ├── test_seed_reproducibility.py  # L1-EXEC-019, 020, 023..025
│   │   ├── test_rng_isolation.py         # L1-EXEC-021 .. L1-EXEC-022
│   │   ├── test_failure_recovery.py      # L1-EXEC-026 .. L1-EXEC-033
│   │   └── fixtures/
│   ├── set5_noise_mathematics/
│   │   ├── reference/
│   │   │   └── exact_bruteforce_oracle.py# Independent GF(2), truncated-Gaussian, DP/brute-force & Markov reference
│   │   ├── test_syndrome_algebra.py          # L1-MATH-001 .. L1-MATH-008
│   │   ├── test_persistent_disorder.py       # L1-MATH-009 .. L1-MATH-016
│   │   ├── test_correlated_noise.py          # L1-MATH-017 .. L1-MATH-024
│   │   ├── test_measurement_noise.py         # L1-MATH-025 .. L1-MATH-030
│   │   ├── test_joint_logical_distribution.py# L1-MATH-031 .. L1-MATH-036
│   │   └── test_exact_oracle.py              # L1-MATH-037 .. L1-MATH-040
│   └── acceptance/
│       ├── test_layer1_acceptance.py     # Gate G1..G7 evaluator & JSON report generator
│       └── requirements_traceability.yaml# Machine-readable ID -> test function -> gate mapping
└── scripts/
    ├── run_layer1_set1.sh
    ├── run_layer1_set2.sh
    ├── run_layer1_set3.sh
    ├── run_layer1_set4.sh
    ├── run_layer1_set5.sh
    └── run_layer1_validation.sh          # Unified acceptance runner (G1..G7 + integrated CLI pipeline run)
```

---

## 5. Strict Phased Execution Plan

1. **Phase 0 (Completed by this document):** Inspect `AVID_EPJ_Patches` source, freeze baseline, produce `LAYER1_TEST_SPECIFICATION.md` with all 154 test IDs & discovered discrepancies (`DEF-L1-01..12`).
2. **Phase 1 — Set 1 Implementation & Input Contract Remediation:** Implement `tests/layer1/set1_input_contracts/` + `scripts/run_layer1_set1.sh`, run against current Layer 1 code, fix genuine input-contract defects (`DEF-L1-01`, `DEF-L1-02`, `DEF-L1-03`, `DEF-L1-04`, `DEF-L1-09` boundary checks) on `layer1-noise-stimulus` until **Gate G1 passes 31/31**.
3. **Phase 2 — Set 2 Implementation:** Implement `tests/layer1/set2_output_statistics/` + `scripts/run_layer1_set2.sh` with Profile A/B/C statistical audit logging, binomial/chi-square/block-bootstrap checks (**Gates G2 & G3**).
4. **Phase 3 — Set 3 Implementation & Unified Pipeline Wiring:** Build unified 7-stage CLI entry point (`DEF-L1-05`), resolve config/composition wiring findings (`DEF-L1-06`, `DEF-L1-11`), implement `set3_pipeline_flow/` (**Gate G4**).
5. **Phase 4 — Set 4 Implementation:** Implement `set4_execution_reproducibility/` + `scripts/run_layer1_set4.sh`, verify CLI overrides, artifact checksums/recomputations, per-generator RNG stream isolation, and failure injection (**Gate G5**).
6. **Phase 5 — Set 5 Independent Mathematical Oracle & Physics Suite:** Implement `reference/exact_bruteforce_oracle.py` and `set5_noise_mathematics/`, verify 32-state `P(s,l)` vs exhaustive enumeration and demonstrate syndrome-only identifiability aliasing (**Gate G6**).
7. **Phase 6 — Integrated Acceptance Runner (`scripts/run_layer1_validation.sh`):** Enforce **G1..G7** strictly with `PASS`/`FAIL`/`INCONCLUSIVE` reporting (zero silent `SKIPPED -> PASS`), verify clean run end-to-end, and commit + push to `layer1-noise-stimulus`.
