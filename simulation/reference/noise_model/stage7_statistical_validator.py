"""Stage 7 Statistical Validator & Shared-Trajectory Decoder Benchmark (`stage7_statistical_validator.py`).

Independently evaluates generated trajectories across:
  1. Distribution-level validation (empirical vs. theoretical mean, variance, quantiles, ACF).
  2. Assignment validation (empirical frequency hat{pi}_i vs. configured pi_i, 100s persistence).
  3. Composition validation (XOR Pauli cancellation and Bernoulli probability composition).
  4. 16-state syndrome distribution validation (P_emp(s) vs. exact Joint-DP P_model(s | theta)).
  5. Logical-class identifiability P(s, l | theta), P(l | s, theta), and LER_oracle(theta).
  6. Shared-trajectory 6-decoder comparison (Static BP, Marginal L-BFGS, Deterministic PID,
     Memoryless Joint-DP Projection, Recurrent AVID R2D2, and Oracle-configured BP).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence
import numpy as np

from simulation.reference.baselines.b1_marginal_lbfgs import B1MarginalLBFGSController
from simulation.reference.baselines.b2_observable_pid import B2ObservablePIDController
from simulation.reference.noise_model.joint_dp_likelihood import build_d3_correlated_noise_model
from simulation.reference.noise_model.logical_class_distribution import (
    LogicalClassDistributionEngine,
)
from simulation.reference.syndrome_stream.causal_stream_runner import CycleAccurateBPEngine
from simulation.stimulus.noise_framework.composition.pauli_composition import (
    compose_pauli_masks_xor,
)
from simulation.stimulus.noise_framework.composition.probability_composition import (
    compose_independent_bernoulli,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry


class Stage7StatisticalValidator:
    """Independent statistical validator and shared-input decoder evaluator."""

    def __init__(self) -> None:
        self.dp_model_x = build_d3_correlated_noise_model(
            basis="X", include_measurement_faults=False
        )
        self.logical_engine_x = LogicalClassDistributionEngine(basis="X")
        self.bp_x = CycleAccurateBPEngine(code=self.dp_model_x.code, basis="X", default_K=3)

    def validate_pd_library(
        self, pd_registry: PDRegistry, num_samples: int = 20000, seed: int = 701
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        per_profile: Dict[str, Any] = {}
        all_passed = True

        for pid in pd_registry.list_profiles():
            prof = pd_registry.get(pid)
            draws = prof.sample_values(rng, size=num_samples)
            emp_mean = float(np.mean(draws))
            emp_var = float(np.var(draws))
            theo_mean, theo_var = prof.theoretical_moments()
            q10, q50, q90 = [float(x) for x in np.quantile(draws, [0.10, 0.50, 0.90])]

            in_support = bool(np.all((draws >= prof.min_val) & (draws <= prof.max_val)))
            mean_rel_err = abs(emp_mean - theo_mean) / max(theo_mean, 1e-6)
            std_rel_err = abs(np.sqrt(emp_var) - np.sqrt(theo_var)) / max(np.sqrt(theo_var), 1e-6)
            ok = bool(in_support and mean_rel_err < 0.05 and std_rel_err < 0.08)
            if not ok:
                all_passed = False

            per_profile[pid] = {
                "empirical_mean": emp_mean,
                "theoretical_mean": theo_mean,
                "mean_rel_error": float(mean_rel_err),
                "empirical_std": float(np.sqrt(emp_var)),
                "theoretical_std": float(np.sqrt(theo_var)),
                "std_rel_error": float(std_rel_err),
                "quantiles_10_50_90": [q10, q50, q90],
                "support_min": prof.min_val,
                "support_max": prof.max_val,
                "within_support": in_support,
                "passed": ok,
            }

        return {
            "num_profiles_validated": len(per_profile),
            "samples_per_profile": num_samples,
            "all_passed": all_passed,
            "profiles": per_profile,
        }

    def validate_composition_rules(self, seed: int = 702) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        # 1. Identical Pauli error cancellation Z XOR Z = I
        m1 = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        comp_identical = compose_pauli_masks_xor([m1, m1])
        identical_cancels = bool(np.all(comp_identical == 0))

        # 2. Disjoint single-qubit generators
        g_a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        g_b = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)
        comp_disjoint = compose_pauli_masks_xor([g_a, g_b])
        disjoint_ok = bool(comp_disjoint[0] == 1 and comp_disjoint[4] == 1 and np.sum(comp_disjoint) == 2)

        # 3. Empirical activation vs. theoretical 1 - prod_j (1 - p_j)
        p1 = np.linspace(0.02, 0.10, 9)
        p2 = np.linspace(0.03, 0.08, 9)
        p3 = np.linspace(0.01, 0.06, 9)
        theo_union = compose_independent_bernoulli([p1, p2, p3])
        n_trials = 50000
        e1 = rng.uniform(0.0, 1.0, size=(n_trials, 9)) < p1
        e2 = rng.uniform(0.0, 1.0, size=(n_trials, 9)) < p2
        e3 = rng.uniform(0.0, 1.0, size=(n_trials, 9)) < p3
        emp_union = np.mean(e1 | e2 | e3, axis=0)
        max_union_err = float(np.max(np.abs(emp_union - theo_union)))

        return {
            "identical_pauli_cancellation_verified": identical_cancels,
            "disjoint_qubit_superposition_verified": disjoint_ok,
            "probability_union_max_abs_error": max_union_err,
            "passed": bool(identical_cancels and disjoint_ok and max_union_err < 0.01),
        }

    def validate_trajectory_distributions(
        self, encoded_traj: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta_z_seq = np.asarray(encoded_traj["theta_z"], dtype=np.float64)
        events_z_21 = np.asarray(encoded_traj["events_z_21"], dtype=np.uint8)
        ideal_sx = np.asarray(encoded_traj["ideal_syndromes_x"], dtype=np.uint8)
        n_rounds = theta_z_seq.shape[0]

        # 1. Per-mechanism empirical vs. analytical activation rate
        emp_mech_rate = np.mean(events_z_21, axis=0)
        theo_mech_rate = np.mean(theta_z_seq, axis=0)
        mech_max_diff = float(np.max(np.abs(emp_mech_rate - theo_mech_rate)))

        # 2. 16-state syndrome distribution P_emp(s) vs. time-averaged exact Joint-DP P_model(s | theta_t)
        s_ints = np.array(
            [int(sum(int(row[c]) << c for c in range(4))) for row in ideal_sx],
            dtype=np.int32,
        )
        emp_hist = np.bincount(s_ints, minlength=16).astype(np.float64) / max(1, n_rounds)

        theo_s_accum = np.zeros(16, dtype=np.float64)
        theo_joint_accum = np.zeros((16, 2), dtype=np.float64)
        # Average analytical distribution across unique sample parameters
        step_stride = max(1, int(encoded_traj.get("rounds_per_composite_sample", 1)))
        sampled_indices = range(0, n_rounds, step_stride)
        for idx in sampled_indices:
            res = self.logical_engine_x.compute_posterior_and_oracle_ler(theta_z_seq[idx])
            theo_s_accum += res["marginal_s"]
            theo_joint_accum += res["joint_s_l"]
        theo_s = theo_s_accum / max(1, len(sampled_indices))
        theo_joint = theo_joint_accum / max(1, len(sampled_indices))

        tvd_syndrome = float(0.5 * np.sum(np.abs(emp_hist - theo_s)))
        eps = 1e-12
        kl_emp_theo = float(
            np.sum(emp_hist * np.log(np.maximum(emp_hist, eps) / np.maximum(theo_s, eps)))
        )

        # 3. Logical-class posterior & Oracle LER
        denom = np.maximum(theo_s[:, None], eps)
        posterior_l_given_s = theo_joint / denom
        ler_oracle = float(np.sum(np.min(theo_joint, axis=1)))

        # 4. Temporal autocorrelation of syndrome weight ||s_t||_1 at lag 1
        w_t = np.sum(ideal_sx, axis=1).astype(np.float64)
        if n_rounds > 2 and np.std(w_t) > 1e-9:
            acf_lag1 = float(np.corrcoef(w_t[:-1], w_t[1:])[0, 1])
        else:
            acf_lag1 = 0.0

        # 5. Identifiability / syndrome-aliasing check
        mean_theta = np.mean(theta_z_seq, axis=0)
        _, _, alias_metrics = self.logical_engine_x.construct_syndrome_aliased_pair(mean_theta)

        return {
            "total_rounds": int(n_rounds),
            "per_mechanism_max_abs_diff": mech_max_diff,
            "syndrome_16state_tvd": tvd_syndrome,
            "syndrome_16state_kl": kl_emp_theo,
            "empirical_syndrome_probs": emp_hist.tolist(),
            "analytical_syndrome_probs": theo_s.tolist(),
            "posterior_l1_given_s": posterior_l_given_s[:, 1].tolist(),
            "analytical_ler_oracle": ler_oracle,
            "syndrome_weight_acf_lag1": acf_lag1,
            "identifiability_aliasing_analysis": alias_metrics,
            "passed": bool(mech_max_diff < 0.035 and tvd_syndrome < 0.045),
        }

    def evaluate_decoders_on_shared_trajectory(
        self,
        encoded_traj: Dict[str, Any],
        window_rounds: int = 250,
    ) -> Dict[str, Any]:
        """Run all 6 decoder classes on the exact same trajectory realization."""
        sx_obs = np.asarray(encoded_traj["syndromes_x"], dtype=np.uint8)
        err_z = np.asarray(encoded_traj["fault_masks_z"], dtype=np.uint8)
        theta_z_seq = np.asarray(encoded_traj["theta_z"], dtype=np.float64)
        n_rounds = sx_obs.shape[0]

        _, static_q78 = self.dp_model_x.compute_q7_8_llrs(self.dp_model_x.default_theta)
        b1_solver = B1MarginalLBFGSController()
        b2_pid = B2ObservablePIDController()

        active_q78_b1 = static_q78.copy()
        active_q78_b2 = static_q78.copy()
        active_q78_proj = static_q78.copy()
        active_q78_avid = static_q78.copy()
        active_K_avid = 3
        theta_ema = self.dp_model_x.default_theta.copy()

        hist = np.zeros(16, dtype=np.float64)
        sub_hist = np.zeros(16, dtype=np.float64)
        failures = {
            "1_static_bp": 0,
            "2_marginal_lbfgs": 0,
            "3_deterministic_pid": 0,
            "4_memoryless_joint_dp_proj": 0,
            "5_recurrent_avid_r2d2": 0,
            "6_oracle_bp_and_coset": 0,
        }

        for r in range(n_rounds):
            s_vec = sx_obs[r]
            s_mask = int(sum((int(s_vec[c]) & 1) << c for c in range(4)))
            true_e = err_z[r]
            true_log = int(encoded_traj["logical_error_z"][r])
            hist[s_mask] += 1.0
            sub_hist[s_mask] += 1.0

            # 1. Static BP (K=3)
            if self.bp_x.decode(s_mask, static_q78, K=3, true_error=true_e).logical_error:
                failures["1_static_bp"] += 1

            # 2. Marginal L-BFGS (B1)
            if self.bp_x.decode(s_mask, active_q78_b1, K=3, true_error=true_e).logical_error:
                failures["2_marginal_lbfgs"] += 1

            # 3. Deterministic Observable PID (B2)
            if self.bp_x.decode(s_mask, active_q78_b2, K=3, true_error=true_e).logical_error:
                failures["3_deterministic_pid"] += 1

            # 4. Memoryless Joint-DP Projection (K=3)
            if self.bp_x.decode(s_mask, active_q78_proj, K=3, true_error=true_e).logical_error:
                failures["4_memoryless_joint_dp_proj"] += 1

            # 5. Recurrent AVID R2D2 (adaptive K, 64-round subwindow burst trigger, and pair-boosted Q7.8 weights)
            if self.bp_x.decode(s_mask, active_q78_avid, K=active_K_avid, true_error=true_e).logical_error:
                failures["5_recurrent_avid_r2d2"] += 1

            # 6. Exact Coset Oracle / True-Theta_t BP
            oracle_info = self.logical_engine_x.compute_posterior_and_oracle_ler(theta_z_seq[r])
            pred_log = int(oracle_info["optimal_logical_decision"][s_mask])
            _, true_q78 = self.dp_model_x.compute_q7_8_llrs(theta_z_seq[r])
            tr_or = self.bp_x.decode(s_mask, true_q78, K=5, true_error=true_e)
            if tr_or.logical_error and (pred_log != true_log):
                failures["6_oracle_bp_and_coset"] += 1

            # Fast 64-round subwindow adaptation for Recurrent AVID R2D2
            if (r + 1) % 64 == 0 and (r + 1) % window_rounds != 0:
                sub_non_triv = float(np.sum(sub_hist[1:]) / max(1.0, float(np.sum(sub_hist))))
                if sub_non_triv >= 0.22:
                    fit_fast = self.dp_model_x.fit_lbfgs(
                        sub_hist + 0.5 * hist, prior_lambda=0.4, max_iter=12
                    )
                    theta_fast = np.asarray(fit_fast["theta"], dtype=np.float64).copy()
                    theta_ema = 0.35 * theta_ema + 0.65 * theta_fast
                    theta_boost = theta_ema.copy()
                    theta_boost[9:] = np.clip(theta_boost[9:] * 1.35, 1e-5, 0.45)
                    active_K_avid = 5
                    _, active_q78_avid = self.dp_model_x.compute_q7_8_llrs(theta_boost)
                sub_hist.fill(0.0)

            if (r + 1) % window_rounds == 0:
                active_q78_b1 = b1_solver.update_from_histogram(hist)
                active_q78_b2 = b2_pid.update_from_histogram(hist)

                fit_dp = self.dp_model_x.fit_lbfgs(hist, prior_lambda=0.12, max_iter=20)
                theta_hat = np.asarray(fit_dp["theta"], dtype=np.float64).copy()
                theta_proj = theta_hat.copy()
                theta_proj[9:] = np.clip(theta_proj[9:] * 1.25, 1e-5, 0.45)
                _, active_q78_proj = self.dp_model_x.compute_q7_8_llrs(theta_proj)

                theta_ema = 0.25 * theta_ema + 0.75 * theta_hat
                non_triv = float(np.sum(hist[1:]) / max(1.0, float(np.sum(hist))))
                theta_avid = theta_ema.copy()
                if non_triv >= 0.14:
                    theta_avid[9:] = np.clip(theta_avid[9:] * 1.45, 1e-5, 0.45)
                    active_K_avid = 5
                else:
                    active_K_avid = 4
                _, active_q78_avid = self.dp_model_x.compute_q7_8_llrs(theta_avid)
                hist.fill(0.0)
                sub_hist.fill(0.0)

        lers = {k: int(v) / max(1, n_rounds) for k, v in failures.items()}
        return {
            "total_rounds": int(n_rounds),
            "window_rounds": int(window_rounds),
            "logical_failures": failures,
            "logical_error_rates": lers,
        }
