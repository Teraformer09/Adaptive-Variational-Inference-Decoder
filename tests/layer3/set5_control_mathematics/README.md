# Layer 3 Test Set 5 — Mathematical & Control Validation (`L3-MATH-001..062`)

**Authorization Status:** Deferred per Layer 3 Execution Directive (`Stop after Set 1`).  
**Canonical Registry:** `tests/layer3/common/test_ids.yaml` (`62` `L3-MATH` IDs; scheduled immediately after Set 1 review).

## Planned Executable Inventory & Independent Oracles

- `oracles/` (strictly independent of production `noise_model.cpp`, `lbfgs_solver.cpp`, `llr_quantizer.cpp`): `brute_force_likelihood.py` ($2^{21}$ / exact subset enumeration), `finite_difference_gradient.py` ($\epsilon$-sweep central differences), `identifiability_reference.py` (Jacobian SVD, rank, null-space & quotient orbits), `q78_reference.py`.
- `likelihood/`: `test_probability_normalization.cpp`, `test_probability_nonnegative.cpp`, `test_joint_distribution.cpp`, `test_empirical_nll.cpp`, `test_prior_regularization.cpp`, `test_dp_vs_bruteforce.cpp`, `test_xor_convolution.cpp`.
- `gradient/`: `test_analytical_gradient.cpp`, `test_finite_difference_gradient.cpp`, `test_gradient_boundaries.cpp`, `test_gradient_numerical_stability.cpp`.
- `optimizer/`: `test_lbfgs_standalone.cpp`, `test_lbfgs_line_search.cpp`, `test_lbfgs_convergence.cpp`, `test_lbfgs_bounds.cpp`, `test_histogram_solver_optimizer_path.cpp`, `test_production_optimizer_invocation.cpp`, `test_optimizer_failure.cpp`.
- `identifiability/`: `test_parameter_sensitivity.cpp`, `test_jacobian_rank.cpp`, `test_null_space.cpp`, `test_zero_signature_edges.cpp`, `test_quotient_orbit.cpp`, `test_parameter_recovery.cpp`.
- `scheduler/`: `test_window_length.cpp`, `test_burst_threshold.cpp`, `test_debounce.cpp`, `test_trigger_latency.cpp`, `test_causal_ordering.cpp`, `test_production_scheduler_path.cpp`.
- `quantization/`: `test_llr_formula.cpp`, `test_q78_scaling.cpp`, `test_q78_rounding.cpp`, `test_q78_saturation.cpp`, `test_llr_sign_convention.cpp`.
- `safety_fsm/`: `test_fsm_states.cpp`, `test_fsm_transitions.cpp`, `test_fsm_invalid_events.cpp`, `test_fsm_recovery.cpp`, `test_fsm_safety_invariants.cpp`.
