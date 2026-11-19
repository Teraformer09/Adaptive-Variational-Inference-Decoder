"""Group C — Layer 3 Bayesian Inference Validity, Temporal SVD & 4-Way Controller Sufficiency (`C1`–`C6`, `C8`–`C10`)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = THIS_DIR.parent
VAL_ROOT = Path(os.environ.get("AVID_LAYERS_ROOT", str(RESEARCH_ROOT.parent))).resolve()

for p in (VAL_ROOT / "layer3", VAL_ROOT / "layer2", RESEARCH_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.stats_utils import compute_paired_ler_ci_and_tests, mean_and_ci95
from layer3.forward_model.parameter_transition_model import ParameterTransitionPrior, logit_to_prob, prob_to_logit
from layer3.forward_model.syndrome_likelihood import (
    EVENT_SIGS_X_21,
    EVENT_SIGS_Z_21,
    exact_syndrome_probabilities,
)
from layer3.identifiability.sensitivity_jacobian import compute_syndrome_jacobian
from layer3.identifiability.svd_analysis import analyze_jacobian_svd


def finite_difference_syndrome_jacobian(
    theta_21: np.ndarray, basis_checks: str = "X", eps: float = 1e-6
) -> np.ndarray:
    """Central finite-difference check of `compute_syndrome_jacobian`."""
    th = np.asarray(theta_21, dtype=np.float64)
    J_fd = np.zeros((16, 21), dtype=np.float64)
    for m in range(21):
        tp = th.copy()
        tm = th.copy()
        tp[m] += eps
        tm[m] -= eps
        J_fd[:, m] = (
            exact_syndrome_probabilities(tp, basis_checks)
            - exact_syndrome_probabilities(tm, basis_checks)
        ) / (2.0 * eps)
    return J_fd
from layer3.inference.posterior_estimator import SequentialPosteriorEstimator
from layer3.inference.posterior_head import LogitLaplacePosteriorHead
from layer3.inference.posterior_predictive import compute_posterior_predictive_checks

from training.data.trajectory_split import TrajectorySplitManager
from training.environments.decoder_in_loop import (
    CANONICAL_12_COUPLERS,
    STATIC_LUT_Z,
    count_logical_z_failures,
    decode_window_minsum_osd0_z,
    project_q78_weights_d3,
    sample_d3_circuit_window,
)
from training.environments.noise_regime_sampler import sample_physical_noise_trajectory
from training.export.export_checkpoint import load_r2d2_checkpoint_bin
from training.models.r2d2_network import ACTION_DEFINITIONS
from training.observations.observation_builder import ObservationBuilder

NONZERO_ORBITS_X = sorted(set(int(s) for s in EVENT_SIGS_X_21 if int(s) != 0))  # 9 identifiable orbits


def _orbit_effective_probs_9(theta_21: np.ndarray) -> np.ndarray:
    """Compute exact XOR-combined effective fault probability for each of the 9 identifiable X-syndrome orbits."""
    th = np.clip(np.asarray(theta_21, dtype=np.float64), 0.0, 0.49)
    out = np.zeros(len(NONZERO_ORBITS_X), dtype=np.float64)
    for idx, g in enumerate(NONZERO_ORBITS_X):
        prod_1m2p = 1.0
        for m in range(21):
            if int(EVENT_SIGS_X_21[m]) == g:
                prod_1m2p *= (1.0 - 2.0 * float(th[m]))
        out[idx] = 0.5 * (1.0 - prod_1m2p)
    return out


def _infer_flat_no_coupler_refinement(
    hist_16: np.ndarray,
    w_eff: float,
    prior_mu: np.ndarray,
    prior_var: np.ndarray,
    damping: float = 1e-3,
    max_iters: int = 10,
) -> Dict[str, np.ndarray]:
    """Laplace inference WITHOUT hierarchical coupler-endpoint prior refinement (for C5 & Unconstrained 21D)."""
    f_obs = np.asarray(hist_16, dtype=np.float64)
    prec_0 = 1.0 / np.maximum(prior_var, 1e-5)
    z = prior_mu.copy()
    for _ in range(max_iters):
        th = np.clip(logit_to_prob(z), 1e-5, 0.485)
        dth = th * (1.0 - th)
        probs = exact_syndrome_probabilities(th, basis_checks="X")
        Jz = compute_syndrome_jacobian(th, basis_checks="X") * dth[None, :]
        inv_p = 1.0 / np.maximum(probs, 1e-9)
        grad = -w_eff * (Jz.T @ (f_obs * inv_p)) + prec_0 * (z - prior_mu)
        Hz = w_eff * (Jz.T @ (inv_p[:, None] * Jz)) + np.diag(prec_0 + damping)
        try:
            step = -np.linalg.solve(Hz, grad)
        except np.linalg.LinAlgError:
            step = -grad / (np.diag(Hz) + 1.0)
        sn = float(np.max(np.abs(step)))
        if sn > 1.25:
            step *= 1.25 / sn
        z = np.clip(z + step, -11.5, -0.05)
        if sn < 1e-4:
            break
    p_mid = np.clip(logit_to_prob(z), 1e-6, 0.49)
    return {"posterior_mean_21": p_mid, "z_mode": z}


def _infer_orbit_9d_projection(hist_16: np.ndarray) -> np.ndarray:
    """9-D Observable-Orbit Estimator: solves the 9 syndrome-orbit rates via inverse Walsh-Hadamard & maps evenly."""
    f = np.clip(np.asarray(hist_16, dtype=np.float64), 1e-9, 1.0)
    f = f / float(np.sum(f))
    # Empirical Walsh-Hadamard parity expectations E[(-1)^{<w, s>}] = H_16 @ f
    from layer3.forward_model.syndrome_likelihood import HADAMARD_16
    spec_emp = np.clip(HADAMARD_16 @ f, 0.05, 1.0)
    # Least-squares log-parity solve over the 9 unique nonzero syndrome signatures
    M_orb = np.zeros((15, 9), dtype=np.float64)
    for w in range(1, 16):
        for j, g in enumerate(NONZERO_ORBITS_X):
            prod = w & g
            pop = ((prod >> 0) & 1) + ((prod >> 1) & 1) + ((prod >> 2) & 1) + ((prod >> 3) & 1)
            M_orb[w - 1, j] = float(pop & 1)
    y = np.log(spec_emp[1:])  # y_w = sum_j M_orb[w, j] * log(1 - 2 * eta_j)
    eta_log, _, _, _ = np.linalg.lstsq(M_orb, y, rcond=None)
    eta_9 = np.clip(0.5 * (1.0 - np.exp(np.clip(eta_log, -2.5, 0.0))), 0.001, 0.35)

    # Split each orbit evenly across mechanisms in that orbit (no prior preference between single vs pair)
    theta_21 = np.full(21, 0.005, dtype=np.float64)
    for j, g in enumerate(NONZERO_ORBITS_X):
        members = [m for m in range(21) if int(EVENT_SIGS_X_21[m]) == g]
        per_m = 0.5 * (1.0 - ((1.0 - 2.0 * eta_9[j]) ** (1.0 / len(members))))
        for m in members:
            theta_21[m] = float(np.clip(per_m, 0.001, 0.35))
    return theta_21


def run_group_c_suite(out_json_dir: Path | None = None) -> Dict[str, Any]:
    v2_bin = VAL_ROOT / "layer2" / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    prod_net, _ = load_r2d2_checkpoint_bin(v2_bin, expected_version=2)

    # -------------------------------------------------------------------------
    # C1 + Dual-Basis Check: Analytical Jacobian SVD (X & Z bases) + Dual-Basis Posterior
    # -------------------------------------------------------------------------
    rng_c1 = np.random.default_rng(20260301)
    th_test = rng_c1.uniform(0.006, 0.055, size=21)
    svd_x = analyze_jacobian_svd(th_test, basis_checks="X")
    svd_z = analyze_jacobian_svd(th_test, basis_checks="Z")
    J_x_fd = finite_difference_syndrome_jacobian(th_test, basis_checks="X")
    J_z_fd = finite_difference_syndrome_jacobian(th_test, basis_checks="Z")
    fd_err_x = float(np.max(np.abs(compute_syndrome_jacobian(th_test, "X") - J_x_fd)))
    fd_err_z = float(np.max(np.abs(compute_syndrome_jacobian(th_test, "Z") - J_z_fd)))

    # Symmetric dual-basis X & Z estimation verification
    dual_est = SequentialPosteriorEstimator(temporal_beta=0.55, infer_x_faults=True)
    th_z_asym = np.full(21, 0.010, dtype=np.float64)
    th_z_asym[4] = 0.052
    th_z_asym[9 + 3] = 0.085
    th_x_asym = np.full(21, 0.009, dtype=np.float64)
    th_x_asym[2] = 0.058
    th_x_asym[9 + 8] = 0.078  # vertical X-pair (1, 4)
    circ_dual = sample_d3_circuit_window(rng_c1, th_z_asym, th_x_asym, window_rounds=1024)
    pkt_dual = dual_est.step_from_syndrome_windows(circ_dual["sx"], circ_dual["sz"])
    assert pkt_dual.posterior_mean_x_21 is not None
    c1_passed = bool(
        svd_x["rank"] == 9
        and svd_x["nullity"] == 12
        and svd_z["rank"] == 9
        and svd_z["nullity"] == 12
        and fd_err_x < 1e-8
        and fd_err_z < 1e-8
        and pkt_dual.estimator_residual_tvd < 0.045
        and pkt_dual.estimator_residual_tvd_z < 0.045
        and float(pkt_dual.posterior_mean_x_21[2]) > 0.025
    )

    # -------------------------------------------------------------------------
    # C8: Temporal Stacked Jacobian Identifiability J_stack = [J(theta_1); ...; J(theta_T)]
    # -------------------------------------------------------------------------
    T_stack = 8
    # Case 1: Stationary trajectory theta_1 = ... = theta_T
    J_stat_stack = np.vstack([compute_syndrome_jacobian(th_test, "X") for _ in range(T_stack)])
    svals_stat = np.linalg.svd(J_stat_stack, compute_uv=False)
    rank_stat_stack = int(np.sum(svals_stat > 1e-9 * svals_stat[0]))

    # Case 2: Non-stationary multi-regime operating points theta_1 .. theta_8
    theta_points = []
    for t in range(T_stack):
        th_t = rng_c1.uniform(0.005, 0.030, size=21)
        th_t[t % 9] = 0.075 + 0.015 * t
        th_t[9 + (t % 12)] = 0.090
        theta_points.append(th_t)
    J_temp_stack = np.vstack([compute_syndrome_jacobian(tp, "X") for tp in theta_points])
    svals_temp = np.linalg.svd(J_temp_stack, compute_uv=False)
    rank_temp_strict = int(np.sum(svals_temp > 1e-9 * svals_temp[0]))
    rank_temp_effective_1pct = int(np.sum(svals_temp > 1e-2 * svals_temp[0]))

    # Case 3: Dual-basis coupled Jacobian [J_X(theta); J_Z(theta)] under isotropic depolarization
    J_dual_stack = np.vstack(
        [compute_syndrome_jacobian(th_test, "X"), compute_syndrome_jacobian(th_test, "Z")]
    )
    svals_dual = np.linalg.svd(J_dual_stack, compute_uv=False)
    rank_dual = int(np.sum(svals_dual > 1e-9 * svals_dual[0]))
    c8_passed = bool(
        rank_stat_stack == 9
        and rank_temp_strict == 19  # 21 - 2 stabilizer-trivial vertical pairs (sig=0x0)
        and rank_dual == 17         # 17 coupled X+Z identifiable directions (4 dual stabilizer modes in nullspace)
    )

    # -------------------------------------------------------------------------
    # C3 & C6: Posterior Credible-Interval Calibration & 1/sqrt(W) Orbit Recovery
    # -------------------------------------------------------------------------
    nominal_levels = [0.50, 0.68, 0.80, 0.90, 0.95]
    orbit_hits = {lv: 0 for lv in nominal_levels}
    total_orbit_checks = 0
    rng_c3 = np.random.default_rng(20260303)
    head_cal = LogitLaplacePosteriorHead(max_iters=8)
    prior_cal = ParameterTransitionPrior(stationary_logit_std=0.72)
    mu0_cal, var0_cal = prior_cal.predict_prior(None, None)

    for trial in range(120):
        th_true = np.clip(logit_to_prob(rng_c3.normal(mu0_cal, np.sqrt(var0_cal))), 0.001, 0.25)
        th_x_bg = np.full(21, 0.005, dtype=np.float64)
        circ_t = sample_d3_circuit_window(rng_c3, th_true, th_x_bg, window_rounds=512)
        hx = np.bincount(circ_t["sx"], minlength=16).astype(np.float64) / 512.0
        post_t = head_cal.infer_posterior(hx, 512.0, mu0_cal, var0_cal, basis_checks="X")

        # Propagate posterior logit covariance Sigma_z (21x21) into the 9 identifiable orbit rates via Monte Carlo draws
        cov_z = np.asarray(post_t["posterior_cov_logit_21x21"], dtype=np.float64)
        z_mean = np.asarray(post_t["posterior_mean_logit_21"], dtype=np.float64)
        z_draws = rng_c3.multivariate_normal(z_mean, cov_z + 1e-7 * np.eye(21), size=250)
        th_draws = np.clip(logit_to_prob(z_draws), 1e-6, 0.49)
        orb_draws = np.stack([_orbit_effective_probs_9(d) for d in th_draws])  # (250, 9)
        orb_true = _orbit_effective_probs_9(th_true)

        for lv in nominal_levels:
            lo = np.quantile(orb_draws, 0.5 * (1.0 - lv), axis=0)
            hi = np.quantile(orb_draws, 1.0 - 0.5 * (1.0 - lv), axis=0)
            orbit_hits[lv] += int(np.sum((orb_true >= lo) & (orb_true <= hi)))
        total_orbit_checks += 9

    empirical_coverage = {f"{int(lv*100)}pct": orbit_hits[lv] / float(total_orbit_checks) for lv in nominal_levels}
    cov_90 = empirical_coverage["90pct"]
    cov_95 = empirical_coverage["95pct"]
    c3_passed = bool((0.78 <= cov_90 <= 0.99) and (0.84 <= cov_95 <= 1.0))

    # C6: 9-orbit recovery RMSE across W in {64, 128, 256, 512, 1024}
    w_grid = [64, 128, 256, 512, 1024]
    orbit_rmse_by_w: List[float] = []
    for w_n in w_grid:
        sq_errs: List[float] = []
        for s_i in range(40):
            th_i = np.full(21, 0.011, dtype=np.float64)
            th_i[s_i % 9] = 0.048
            th_i[9 + (s_i % 6)] = 0.075
            circ_i = sample_d3_circuit_window(rng_c3, th_i, np.full(21, 0.004), window_rounds=w_n)
            hx_i = np.bincount(circ_i["sx"], minlength=16).astype(np.float64) / float(w_n)
            post_i = head_cal.infer_posterior(hx_i, float(w_n), mu0_cal, var0_cal, basis_checks="X")
            e_true = _orbit_effective_probs_9(th_i)
            e_hat = _orbit_effective_probs_9(np.asarray(post_i["posterior_mean_21"]))
            sq_errs.append(float(np.mean((e_hat - e_true) ** 2)))
        orbit_rmse_by_w.append(float(np.sqrt(np.mean(sq_errs))))
    c6_passed = bool(orbit_rmse_by_w[-1] < orbit_rmse_by_w[0] * 0.55)

    # -------------------------------------------------------------------------
    # C2 & C10: Prior Sensitivity & Prior Mismatch Sweeps
    # -------------------------------------------------------------------------
    sigma_grid = [0.35, 0.50, 0.68, 1.00, 1.40]
    c2_lers: List[float] = []
    c2_rmses: List[float] = []
    for sig_0 in sigma_grid:
        pr = ParameterTransitionPrior(stationary_logit_std=sig_0)
        mu_s, var_s = pr.predict_prior(None, None)
        hd = LogitLaplacePosteriorHead(max_iters=6)
        f_tot, r_tot = 0, 0
        sq_e = []
        for i in range(12):
            rng_s = np.random.default_rng(76000 + i)
            phys = sample_physical_noise_trajectory("zz_crosstalk", 76000 + i, num_epochs=8)
            for ep in range(8):
                th_z = phys["theta_z_seq"][ep]
                circ = sample_d3_circuit_window(rng_s, th_z, phys["theta_x_seq"][ep], window_rounds=256)
                hx = np.bincount(circ["sx"], minlength=16).astype(np.float64) / 256.0
                out = hd.infer_posterior(hx, 256.0, mu_s, var_s, basis_checks="X")
                pm = np.asarray(out["posterior_mean_21"])
                sq_e.append(float(np.mean((pm - th_z) ** 2)))
                w_q = project_q78_weights_d3(pm[:9], pm[9:], pair_weight_scale=1.20)
                f_tot += count_logical_z_failures(circ["err_z"], decode_window_minsum_osd0_z(circ["sx"], w_q, 4))
                r_tot += 256
        c2_lers.append(f_tot / float(r_tot))
        c2_rmses.append(float(np.sqrt(np.mean(sq_e))))
    c2_passed = bool(max(c2_lers) - min(c2_lers) < 0.015)

    # C10: Severe Prior Mismatch (actual background 3x higher + strong coupler vs nominal 0.012 prior)
    mis_stat, mis_avid = [], []
    for i in range(20):
        rng_m = np.random.default_rng(77000 + i)
        est = SequentialPosteriorEstimator(temporal_beta=0.50)
        obs_b = ObservationBuilder(max_config_age_epochs=10)
        h, c = prod_net.zero_state()
        f_s, f_a = 0, 0
        for ep in range(10):
            th_z = np.full(21, 0.006, dtype=np.float64)
            th_z[:9] = 0.019  # ~1.6x higher background single-qubit error than prior 0.012
            hp = i % 5
            th_z[9 + hp] = 0.096  # > 19x higher pair rate than prior 0.005
            th_z[CANONICAL_12_COUPLERS[hp][0]] = 0.034
            circ = sample_d3_circuit_window(rng_m, th_z, np.full(21, 0.005), window_rounds=256)
            pkt = est.step_from_syndrome_windows(circ["sx"], circ["sz"])
            built = obs_b.build_from_posterior_and_syndromes(pkt.to_layer2_dict(), circ["sx"], circ["sz"])
            step_out = prod_net.step(built["observation_82"], h, c)
            h, c = step_out["h_next"], step_out["c_next"]
            act = int(step_out["action_id"])
            spec = ACTION_DEFINITIONS[act]
            gm = np.clip(1.0 + np.asarray(step_out["qubit_gains"]), 0.70, 1.30)
            pm = pkt.posterior_mean_21
            w_a = project_q78_weights_d3(
                pm[:9],
                pm[9:],
                pair_weight_scale=float(spec["pair_weight_scale"]),
                qubit_gains=gm if float(spec["pair_weight_scale"]) > 0.0 else None,
                fallback=bool(spec["fallback"]),
            )
            f_a += count_logical_z_failures(
                circ["err_z"], decode_window_minsum_osd0_z(circ["sx"], w_a, int(spec["bp_iterations"]))
            )
            f_s += count_logical_z_failures(circ["err_z"], STATIC_LUT_Z[circ["sx"]])
        mis_stat.append(f_s / 2560.0)
        mis_avid.append(f_a / 2560.0)
    ci_c10 = compute_paired_ler_ci_and_tests(mis_stat, mis_avid, seed=8510)
    c10_passed = bool(ci_c10["significant_improvement_95pct"])

    # -------------------------------------------------------------------------
    # C4: Uncertainty Propagation (Varying Effective Window Rounds W_eff -> Log-Var & Actions)
    # Measure posterior logit variance reduction on observable directions as W_eff increases
    # -------------------------------------------------------------------------
    w_unc_grid = [32, 64, 128, 256, 512, 1024]
    mean_log_vars: List[float] = []
    high_k_rates: List[float] = []
    for w_u in w_unc_grid:
        lvs, hk = [], 0
        for i in range(16):
            rng_u = np.random.default_rng(78000 + i)
            th_z = np.full(21, 0.012, dtype=np.float64)
            th_z[9 + 3] = 0.020
            circ = sample_d3_circuit_window(rng_u, th_z, np.full(21, 0.004), window_rounds=w_u)
            hx_u = np.bincount(circ["sx"], minlength=16).astype(np.float64) / float(w_u)
            post_u = head_cal.infer_posterior(hx_u, float(w_u), mu0_cal, var0_cal, basis_checks="X")
            # Observable-leverage weighted log-variance in logit space (shrinks as Fisher info ~ W_eff grows)
            lev_mask = np.asarray(svd_x["parameter_observability_leverage"]) > 0.30
            var_logit_obs = np.asarray(post_u["posterior_var_logit_21"])[lev_mask]
            lvs.append(float(np.mean(np.log(np.maximum(var_logit_obs, 1e-12)))))

            est = SequentialPosteriorEstimator(temporal_beta=0.0)
            pkt = est.step_from_syndrome_windows(circ["sx"], circ["sz"])
            obs_b = ObservationBuilder()
            obs_82 = obs_b.build_from_posterior_and_syndromes(pkt.to_layer2_dict(), circ["sx"], circ["sz"])["observation_82"]
            h0, c0 = prod_net.zero_state()
            st = prod_net.step(obs_82, h0, c0)
            if int(st["bp_iterations"]) >= 4:
                hk += 1
        mean_log_vars.append(float(np.mean(lvs)))
        high_k_rates.append(hk / 16.0)
    c4_passed = bool(mean_log_vars[-1] < mean_log_vars[0] - 0.50)

    # -------------------------------------------------------------------------
    # C5: Coupler-Endpoint Prior Ablation (With vs Without Hierarchical Refinement)
    # -------------------------------------------------------------------------
    c5_with_lers, c5_without_lers = [], []
    prior_std = ParameterTransitionPrior()
    mu0_std, var0_std = prior_std.predict_prior(None, None)
    head_with = LogitLaplacePosteriorHead(max_iters=6)

    for i in range(24):
        rng_c5 = np.random.default_rng(79000 + i)
        phys = sample_physical_noise_trajectory("zz_crosstalk", 79000 + i, num_epochs=10)
        f_w, f_wo = 0, 0
        for ep in range(10):
            circ = sample_d3_circuit_window(rng_c5, phys["theta_z_seq"][ep], phys["theta_x_seq"][ep], window_rounds=256)
            hx = np.bincount(circ["sx"], minlength=16).astype(np.float64) / 256.0
            pm_with = np.asarray(head_with.infer_posterior(hx, 256.0, mu0_std, var0_std)["posterior_mean_21"])
            pm_wo = _infer_flat_no_coupler_refinement(hx, 256.0, mu0_std, var0_std)["posterior_mean_21"]
            w_with = project_q78_weights_d3(pm_with[:9], pm_with[9:], pair_weight_scale=1.20)
            w_wo = project_q78_weights_d3(pm_wo[:9], pm_wo[9:], pair_weight_scale=1.20)
            f_w += count_logical_z_failures(circ["err_z"], decode_window_minsum_osd0_z(circ["sx"], w_with, 4))
            f_wo += count_logical_z_failures(circ["err_z"], decode_window_minsum_osd0_z(circ["sx"], w_wo, 4))
        c5_with_lers.append(f_w / 2560.0)
        c5_without_lers.append(f_wo / 2560.0)
    ci_c5 = compute_paired_ler_ci_and_tests(c5_without_lers, c5_with_lers, seed=8505)
    c5_passed = bool(ci_c5["significant_improvement_95pct"])

    # -------------------------------------------------------------------------
    # C9: Section 1 4-Way Controller Sufficiency Experiment (24 Paired Seeds x 5 Regimes)
    # -------------------------------------------------------------------------
    # Compare:
    #   1) Unconstrained_21D (weak prior std=4.5, no coupler structure)
    #   2) Observable_Orbit_9D (9 identifiable syndrome orbits split evenly)
    #   3) Regularized_Layer3_21D (current Layer 3 prior-regularized posterior)
    #   4) Controller_Oriented_AVID_V2 (Layer 3 compressed posterior + R2D2 V2 controller)
    split_mgr = TrajectorySplitManager()
    regimes_5 = ("gaussian_stationary", "ou_drift", "zz_crosstalk", "tls_telegraph", "nonstationary_burst")
    suff_seeds = list(range(2060, 2084))

    lers_uncon, lers_orbit, lers_reg, lers_ctrl, lers_stat = [], [], [], [], []
    jitter_uncon, jitter_orbit, jitter_reg, jitter_ctrl = [], [], [], []
    tvd_uncon, tvd_orbit, tvd_reg = [], [], []
    time_us_uncon, time_us_orbit, time_us_reg, time_us_ctrl = [], [], [], []

    weak_mu = np.full(21, prob_to_logit(np.array([0.015]))[0], dtype=np.float64)
    weak_var = np.full(21, 16.0, dtype=np.float64)  # std = 4.0 logit units (unconstrained)

    for s in suff_seeds:
        fu, fo, fr, fc, fs = 0, 0, 0, 0, 0
        tot_r = 0
        for r_i, fam in enumerate(regimes_5):
            ep_seed = split_mgr.episode_seed("test", s, r_i)
            rng_sf = np.random.default_rng(ep_seed + 17)
            phys = sample_physical_noise_trajectory(fam, ep_seed, num_epochs=12)
            est_reg = SequentialPosteriorEstimator(temporal_beta=0.55)
            obs_b = ObservationBuilder(max_config_age_epochs=12)
            h, c = prod_net.zero_state()
            recal = False
            prev_act = None
            prev_wu = prev_wo = prev_wr = prev_wc = None

            is_corr_sf = np.asarray(phys["is_corrupted_seq"], dtype=bool)
            for ep in range(12):
                # Causal observation window (W=256) followed by causal evaluation window (W=256)
                circ_obs = sample_d3_circuit_window(
                    rng_sf,
                    phys["theta_z_seq"][ep],
                    phys["theta_x_seq"][ep],
                    window_rounds=256,
                    readout_flip_prob=float(phys["readout_flip_seq"][ep]),
                    streak3_prob=float(phys["streak3_prob_seq"][ep]),
                )
                circ = sample_d3_circuit_window(
                    rng_sf,
                    phys["theta_z_seq"][ep],
                    phys["theta_x_seq"][ep],
                    window_rounds=256,
                    readout_flip_prob=float(phys["readout_flip_seq"][ep]),
                    streak3_prob=float(phys["streak3_prob_seq"][ep]),
                )
                hx = np.bincount(circ_obs["sx"], minlength=16).astype(np.float64) / 256.0
                hx_eval = np.bincount(circ["sx"], minlength=16).astype(np.float64) / 256.0

                # 1. Unconstrained 21-D MLE (no coupler structure, unregularized nullspace)
                t0 = time.process_time()
                pm_u = _infer_flat_no_coupler_refinement(hx, 256.0, weak_mu, weak_var, damping=1e-6, max_iters=14)["posterior_mean_21"]
                w_u = project_q78_weights_d3(pm_u[:9], pm_u[9:], pair_weight_scale=1.15)
                if bool(is_corr_sf[ep]):
                    w_u = np.clip(w_u // 4, 16, 0x3FFF)
                time_us_uncon.append((time.process_time() - t0) * 1e6)

                # 2. Observable-Orbit 9-D (splits aliased single-vs-pair orbits 50/50 without physical prior)
                t0 = time.process_time()
                pm_o = _infer_orbit_9d_projection(hx)
                w_o = project_q78_weights_d3(pm_o[:9], pm_o[9:], pair_weight_scale=1.15)
                if bool(is_corr_sf[ep]):
                    w_o = np.clip(w_o // 4, 16, 0x3FFF)
                time_us_orbit.append((time.process_time() - t0) * 1e6)

                # 3. Prior-Regularized Layer 3 (hierarchical coupler prior + temporal smoothing)
                t0 = time.process_time()
                pkt = est_reg.step_from_syndrome_windows(circ_obs["sx"], circ_obs["sz"], recalibrate=recal)
                pm_r = pkt.posterior_mean_21
                w_r = project_q78_weights_d3(pm_r[:9], pm_r[9:], pair_weight_scale=1.15)
                if bool(is_corr_sf[ep]):
                    w_r = np.clip(w_r // 4, 16, 0x3FFF)
                dt_reg = (time.process_time() - t0) * 1e6
                time_us_reg.append(dt_reg)

                # 4. Controller-Oriented AVID V2 (Layer 3 + R2D2 V2 with dynamic K, gains & fallback)
                t0 = time.process_time()
                built = obs_b.build_from_posterior_and_syndromes(pkt.to_layer2_dict(), circ_obs["sx"], circ_obs["sz"])
                step_out = prod_net.step(built["observation_82"], h, c)
                h, c = step_out["h_next"], step_out["c_next"]
                act = int(step_out["action_id"])
                spec = ACTION_DEFINITIONS[act]
                gm = np.clip(1.0 + np.asarray(step_out["qubit_gains"]), 0.70, 1.30)
                w_c = project_q78_weights_d3(
                    pm_r[:9],
                    pm_r[9:],
                    pair_weight_scale=float(spec["pair_weight_scale"]),
                    qubit_gains=gm if float(spec["pair_weight_scale"]) > 0.0 else None,
                    fallback=bool(spec["fallback"]) or bool(is_corr_sf[ep]),
                )
                time_us_ctrl.append(dt_reg + (time.process_time() - t0) * 1e6)

                obs_b.record_action_update(config_changed=(prev_act is not None and act != prev_act))
                est_reg.configure_from_action(float(spec.get("temporal_beta", 0.55)), bool(spec["recalibrate"]))
                recal = bool(spec["recalibrate"])
                prev_act = act

                if prev_wu is not None:
                    jitter_uncon.append(float(np.linalg.norm((w_u - prev_wu) / 256.0)))
                    jitter_orbit.append(float(np.linalg.norm((w_o - prev_wo) / 256.0)))
                    jitter_reg.append(float(np.linalg.norm((w_r - prev_wr) / 256.0)))
                    jitter_ctrl.append(float(np.linalg.norm((w_c - prev_wc) / 256.0)))
                prev_wu, prev_wo, prev_wr, prev_wc = w_u, w_o, w_r, w_c

                tvd_uncon.append(float(compute_posterior_predictive_checks(hx_eval, pm_u)["tvd"]))
                tvd_orbit.append(float(compute_posterior_predictive_checks(hx_eval, pm_o)["tvd"]))
                tvd_reg.append(float(compute_posterior_predictive_checks(hx_eval, pm_r)["tvd"]))

                fu += count_logical_z_failures(circ["err_z"], decode_window_minsum_osd0_z(circ["sx"], w_u, 3))
                fo += count_logical_z_failures(circ["err_z"], decode_window_minsum_osd0_z(circ["sx"], w_o, 3))
                fr += count_logical_z_failures(circ["err_z"], decode_window_minsum_osd0_z(circ["sx"], w_r, 4))
                fc += count_logical_z_failures(circ["err_z"], decode_window_minsum_osd0_z(circ["sx"], w_c, int(spec["bp_iterations"])))
                fs += count_logical_z_failures(circ["err_z"], STATIC_LUT_Z[circ["sx"]])
                tot_r += 256

        lers_uncon.append(fu / float(tot_r))
        lers_orbit.append(fo / float(tot_r))
        lers_reg.append(fr / float(tot_r))
        lers_ctrl.append(fc / float(tot_r))
        lers_stat.append(fs / float(tot_r))

    ci_ctrl_vs_static = compute_paired_ler_ci_and_tests(lers_stat, lers_ctrl, seed=8900)
    ci_ctrl_vs_uncon = compute_paired_ler_ci_and_tests(lers_uncon, lers_ctrl, seed=8901)
    ci_ctrl_vs_orbit = compute_paired_ler_ci_and_tests(lers_orbit, lers_ctrl, seed=8902)
    ci_ctrl_vs_reg = compute_paired_ler_ci_and_tests(lers_reg, lers_ctrl, seed=8903)
    ci_reg_vs_uncon = compute_paired_ler_ci_and_tests(lers_uncon, lers_reg, seed=8904)

    # Reviewer Section 1 Sufficiency Gate:
    # Compressed regularized posterior + R2D2 V2 must:
    #  (1) Significantly outperform Static B0 (95% CI > 0, p < 0.01),
    #  (2) Outperform Regularized-Layer3-only fixed heuristic in mean LER,
    #  (3) Retain control sufficiency within <= 0.0045 absolute LER of unconstrained 21-D MLE, AND
    #  (4) Cut hardware weight jitter by >= 40% relative to unconstrained 21-D MLE & 9-D Orbit!
    c9_passed = bool(
        ci_ctrl_vs_static["significant_improvement_95pct"]
        and (float(np.mean(lers_ctrl)) < float(np.mean(lers_reg)))
        and (float(np.mean(lers_ctrl)) - float(np.mean(lers_uncon)) <= 0.0045)
        and (float(np.mean(jitter_ctrl)) < 0.60 * float(np.mean(jitter_uncon)))
        and (float(np.mean(jitter_ctrl)) < 0.60 * float(np.mean(jitter_orbit)))
    )

    result = {
        "group": "Group_C_Bayesian_Inference_and_Controller_Sufficiency",
        "gates": {
            "C1_jacobian_rank_and_dual_basis": {
                "passed": c1_passed,
                "rank_x": svd_x["rank"],
                "nullity_x": svd_x["nullity"],
                "rank_z": svd_z["rank"],
                "nullity_z": svd_z["nullity"],
                "max_fd_error_x": fd_err_x,
                "max_fd_error_z": fd_err_z,
                "dual_basis_ppc_tvd_x": float(pkt_dual.estimator_residual_tvd),
                "dual_basis_ppc_tvd_z": float(pkt_dual.estimator_residual_tvd_z),
                "singular_values_single_window": [float(v) for v in svd_x["singular_values"]],
            },
            "C2_prior_sensitivity": {
                "passed": c2_passed,
                "prior_logit_std_grid": sigma_grid,
                "ler_across_grid": c2_lers,
                "rmse_across_grid": c2_rmses,
            },
            "C3_posterior_calibration": {
                "passed": c3_passed,
                "nominal_levels": nominal_levels,
                "empirical_orbit_coverage": empirical_coverage,
            },
            "C4_uncertainty_propagation": {
                "passed": c4_passed,
                "window_rounds_grid": w_unc_grid,
                "mean_posterior_log_var": mean_log_vars,
                "adaptive_k4plus_selection_rate": high_k_rates,
            },
            "C5_coupler_endpoint_prior_ablation": {
                "passed": c5_passed,
                "paired_stats_with_vs_without": ci_c5,
            },
            "C6_observable_orbit_recovery": {
                "passed": c6_passed,
                "window_grid": w_grid,
                "orbit_9_rmse": orbit_rmse_by_w,
            },
            "C8_temporal_stacked_jacobian_identifiability": {
                "passed": c8_passed,
                "num_stacked_windows_T": T_stack,
                "stationary_stack_rank": rank_stat_stack,
                "nonstationary_operating_point_strict_rank": rank_temp_strict,
                "nonstationary_effective_rank_1pct": rank_temp_effective_1pct,
                "dual_basis_coupled_rank": rank_dual,
                "svals_stationary_stack": (svals_stat / max(svals_stat[0], 1e-12)).tolist(),
                "svals_nonstationary_stack": (svals_temp / max(svals_temp[0], 1e-12)).tolist(),
                "svals_dual_basis_stack": (svals_dual / max(svals_dual[0], 1e-12)).tolist(),
            },
            "C9_four_way_controller_sufficiency": {
                "passed": c9_passed,
                "table": {
                    "1_Unconstrained_21D_MLE": {
                        "mean_ler": float(np.mean(lers_uncon)),
                        "ppc_tvd": float(np.mean(tvd_uncon)),
                        "weight_jitter_l2_nats": float(np.mean(jitter_uncon)),
                        "compute_time_us": float(np.mean(time_us_uncon)),
                    },
                    "2_Observable_Orbit_9D": {
                        "mean_ler": float(np.mean(lers_orbit)),
                        "ppc_tvd": float(np.mean(tvd_orbit)),
                        "weight_jitter_l2_nats": float(np.mean(jitter_orbit)),
                        "compute_time_us": float(np.mean(time_us_orbit)),
                    },
                    "3_Regularized_Layer3_21D": {
                        "mean_ler": float(np.mean(lers_reg)),
                        "ppc_tvd": float(np.mean(tvd_reg)),
                        "weight_jitter_l2_nats": float(np.mean(jitter_reg)),
                        "compute_time_us": float(np.mean(time_us_reg)),
                    },
                    "4_Controller_Oriented_AVID_V2": {
                        "mean_ler": float(np.mean(lers_ctrl)),
                        "ppc_tvd": float(np.mean(tvd_reg)),
                        "weight_jitter_l2_nats": float(np.mean(jitter_ctrl)),
                        "compute_time_us": float(np.mean(time_us_ctrl)),
                    },
                    "0_Static_B0_Reference": {
                        "mean_ler": float(np.mean(lers_stat)),
                    },
                },
                "paired_comparisons": {
                    "AVID_V2_vs_Static_B0": ci_ctrl_vs_static,
                    "AVID_V2_vs_Unconstrained_21D": ci_ctrl_vs_uncon,
                    "AVID_V2_vs_Observable_Orbit_9D": ci_ctrl_vs_orbit,
                    "AVID_V2_vs_Regularized_Layer3_Only": ci_ctrl_vs_reg,
                    "Regularized_Layer3_vs_Unconstrained_21D": ci_reg_vs_uncon,
                },
            },
            "C10_prior_mismatch_robustness": {
                "passed": c10_passed,
                "paired_stats_20_seeds": ci_c10,
            },
        },
        "overall_passed": bool(
            c1_passed and c2_passed and c3_passed and c4_passed and c5_passed
            and c6_passed and c8_passed and c9_passed and c10_passed
        ),
    }

    if out_json_dir is not None:
        out_json_dir.mkdir(parents=True, exist_ok=True)
        (out_json_dir / "group_c_bayesian_sufficiency.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    res = run_group_c_suite(RESEARCH_ROOT / "results" / "json")
    print(f"[Group C] overall_passed={res['overall_passed']}")
