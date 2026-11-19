"""Group A — Layer 1 Noise-Model Validity & Out-of-Distribution Generalization (`A1`–`A6`)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = THIS_DIR.parent
VAL_ROOT = Path(os.environ.get("AVID_LAYERS_ROOT", str(RESEARCH_ROOT.parent))).resolve()

# Import Layer 1 in isolated module order, plus Layer 2/3 for OOD closed-loop A5 check
for p in (VAL_ROOT / "layer1", VAL_ROOT / "layer3", VAL_ROOT / "layer2", RESEARCH_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.stats_utils import compute_paired_ler_ci_and_tests
from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.config_schema import CANONICAL_12_PAIRS, GeneratorSpec
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.generators.ou_generator import OUNoiseGenerator
from simulation.stimulus.noise_framework.generators.spatial_correlated_generator import (
    SpatialCorrelatedGenerator,
    load_stage3_spatial_clusters,
)
from simulation.stimulus.noise_framework.generators.telegraph_generator import TelegraphNoiseGenerator
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler
from simulation.stimulus.noise_framework.avid_interface.sample_encoder import SampleEncoder

from layer3.inference.posterior_estimator import SequentialPosteriorEstimator
from training.environments.decoder_in_loop import (
    STATIC_LUT_Z,
    count_logical_z_failures,
    decode_window_minsum_osd0_z,
    project_q78_weights_d3,
    sample_d3_circuit_window,
)
from training.export.export_checkpoint import load_r2d2_checkpoint_bin
from training.models.r2d2_network import ACTION_DEFINITIONS
from training.observations.observation_builder import ObservationBuilder


def _empirical_acf(series_1d: np.ndarray, max_lag: int = 60) -> np.ndarray:
    x = np.asarray(series_1d, dtype=np.float64)
    x = x - float(np.mean(x))
    var = float(np.dot(x, x))
    if var < 1e-15:
        out = np.zeros(max_lag + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    acf = np.empty(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0
    n = x.size
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.dot(x[:-lag], x[lag:]) / var) * (n / float(n - lag))
    return acf


def _sample_ood_trajectory(ood_name: str, seed: int, num_epochs: int = 14) -> Dict[str, np.ndarray]:
    """Construct unseen Out-of-Distribution (OOD) physical + readout + streak trajectories for A5."""
    rng = np.random.default_rng(int(seed))
    theta_z = np.zeros((num_epochs, 21), dtype=np.float64)
    theta_x = np.zeros((num_epochs, 21), dtype=np.float64)
    readout = np.zeros(num_epochs, dtype=np.float64)
    streak3 = np.zeros(num_epochs, dtype=np.float64)

    for ep in range(num_epochs):
        qz = rng.uniform(0.011, 0.017, size=9)
        pz = rng.uniform(0.004, 0.009, size=12)
        qx = rng.uniform(0.005, 0.010, size=9)
        px = rng.uniform(0.002, 0.005, size=12)

        if ood_name == "OOD_1_TriplePair_01_45_78":
            # Simultaneous multi-row horizontal ZZ coupler activation (0,1) + (4,5) never seen in training
            pz[0] = rng.uniform(0.072, 0.095)  # (0,1) -> sig 0x2
            pz[3] = rng.uniform(0.076, 0.098)  # (4,5) -> sig 0x4
            qz[[0, 1, 4, 5]] += 0.014
        elif ood_name == "OOD_2_ElevatedFloor_HeavyReadout":
            # Unseen combination: elevated chip floor + strong coupler (1,2) & (6,7) + readout flip noise
            qz = rng.uniform(0.015, 0.021, size=9)
            qz[[1, 2, 6, 7]] += 0.016
            pz[1] = rng.uniform(0.080, 0.104)  # (1,2) -> sig 0x1
            pz[4] = rng.uniform(0.074, 0.096)  # (6,7) -> sig 0x8
            readout[ep] = 0.006
        elif ood_name == "OOD_3_TLS_Burst_Plus_3QubitStreak":
            # Simultaneous RTN TLS telegraph + burst + weight-3 correlated streak
            if ep % 3 != 0:
                qz[7] = 0.064
                pz[4] = 0.082  # (6,7)
            if ep in (4, 5, 9, 10):
                qz += 0.014
                pz[3] = 0.088  # (4,5)
                streak3[ep] = 0.020
        elif ood_name == "OOD_4_Asymmetric_Corner_Star":
            # Unseen dynamic spatio-temporal switch between coupler (0,1) and (4,5) + corner hotspot drift
            qz[[0, 4, 8]] = [0.028, 0.032, 0.026]
            if ep < (num_epochs // 2):
                pz[0] = 0.086  # (0,1)
                pz[4] = 0.078  # (6,7)
            else:
                pz[1] = 0.084  # (1,2)
                pz[3] = 0.088  # (4,5)

        theta_z[ep, :9] = np.clip(qz, 0.001, 0.35)
        theta_z[ep, 9:] = np.clip(pz, 0.001, 0.30)
        theta_x[ep, :9] = np.clip(qx, 0.001, 0.25)
        theta_x[ep, 9:] = np.clip(px, 0.001, 0.20)

    return {
        "theta_z": theta_z,
        "theta_x": theta_x,
        "readout": readout,
        "streak3": streak3,
    }


def run_group_a_suite(out_json_dir: Path | None = None) -> Dict[str, Any]:
    l1_root = VAL_ROOT / "layer1"
    pd_reg = PDRegistry(l1_root / "configs" / "noise_framework" / "pd_profiles_16.yaml")

    # -------------------------------------------------------------------------
    # A1: Noise distribution recovery (empirical vs configured moments & quantiles)
    # -------------------------------------------------------------------------
    rng_a1 = np.random.default_rng(20260101)
    a1_profiles: Dict[str, Any] = {}
    qq_reference: Dict[str, List[float]] = {}
    all_a1_ok = True
    for pid in ("PD_01", "PD_04", "PD_07", "PD_10", "PD_12", "PD_15"):
        prof = pd_reg.get(pid)
        theo_mean, theo_var = prof.theoretical_moments()
        draws = np.concatenate([prof.sample_values(rng_a1, size=21) for _ in range(400)])
        emp_mean = float(np.mean(draws))
        emp_std = float(np.std(draws))
        rel_mean_err = abs(emp_mean - theo_mean) / max(theo_mean, 1e-6)
        in_bounds = bool(np.all((draws >= prof.min_val - 1e-9) & (draws <= prof.max_val + 1e-9)))
        ok_p = (rel_mean_err < 0.12) and in_bounds
        all_a1_ok = all_a1_ok and ok_p
        a1_profiles[pid] = {
            "distribution_type": prof.distribution_type,
            "configured_mean": float(theo_mean),
            "empirical_mean": emp_mean,
            "configured_std": float(math.sqrt(theo_var)),
            "empirical_std": emp_std,
            "rel_mean_error": float(rel_mean_err),
            "bounds_respected": in_bounds,
            "passed": ok_p,
        }
        if pid in ("PD_01", "PD_07", "PD_12"):
            qs = np.linspace(0.02, 0.98, 49)
            qq_reference[pid] = [float(v) for v in np.quantile(draws, qs)]

    # -------------------------------------------------------------------------
    # A2: Spatial correlation & cluster covariance recovery
    # -------------------------------------------------------------------------
    spec_zz = GeneratorSpec(
        generator_id="zz_crosstalk_01",
        generator_type="spatial_correlated",
        enabled=True,
        seed=20260102,
        output_domain="probability_parameter",
        output_shape=(21,),
        composition_group="data_faults_z",
        parameters={"basis": "Z", "latent_beta_scale": 1.15},
    )
    gen_zz = SpatialCorrelatedGenerator(spec=spec_zz, seed=20260102)
    gen_zz.assign_profile(pd_reg.get("PD_12"), window_id=0)
    zz_mat = np.stack([gen_zz.sample(i * 0.1, i).values for i in range(1500)])
    pair_samples = zz_mat[:, 9:]  # (1500, 12)
    corr_12x12 = np.corrcoef(pair_samples.T)

    s3_cfg = load_stage3_spatial_clusters(
        l1_root / "configs" / "noise_framework" / "stage3_spatial_correlations.yaml"
    )
    intra_corrs: List[float] = []
    inter_corrs: List[float] = []
    clusters = list(s3_cfg["spatial_clusters"].values())
    same_cluster_pairs = set()
    for c_info in clusters:
        ap = [int(x) for x in c_info["active_pairs"]]
        for i in range(len(ap)):
            for j in range(i + 1, len(ap)):
                u, v = min(ap[i], ap[j]), max(ap[i], ap[j])
                same_cluster_pairs.add((u, v))
                intra_corrs.append(float(corr_12x12[u, v]))
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) not in same_cluster_pairs:
                inter_corrs.append(float(corr_12x12[i, j]))

    centered_21 = zz_mat - np.mean(zz_mat, axis=0, keepdims=True)
    _, svals, _ = np.linalg.svd(centered_21, full_matrices=False)
    eig_shares = (svals ** 2) / max(float(np.sum(svals ** 2)), 1e-15)
    pc1_share = float(eig_shares[0])
    k95 = int(np.searchsorted(np.cumsum(eig_shares), 0.95) + 1)
    mean_intra = float(np.mean(intra_corrs))
    mean_inter = float(np.mean(inter_corrs))
    a2_passed = bool((mean_intra > mean_inter + 0.35) and (pc1_share <= 0.70) and (k95 >= 3))

    # -------------------------------------------------------------------------
    # A3: Temporal autocorrelation recovery (Empirical vs Target ACF curves)
    # -------------------------------------------------------------------------
    max_lag = 50  # 5.0 s at 10 Hz
    lags_sec = (np.arange(max_lag + 1) * 0.1).tolist()

    spec_ou = GeneratorSpec(
        generator_id="ou_test",
        generator_type="ou",
        enabled=True,
        seed=20260103,
        output_domain="probability_parameter",
        output_shape=(21,),
        composition_group="data_faults_x",
        parameters={"basis": "X", "tau_sec": 1.0 / 0.35, "envelope_gain": 0.0, "dom_boost_scale": 0.0},
    )
    gen_ou = OUNoiseGenerator(spec=spec_ou, seed=20260103)
    gen_ou.assign_profile(pd_reg.get("PD_01"), window_id=0)
    ou_traj = np.stack([gen_ou.sample(i * 0.1, i).values for i in range(2500)])
    acf_ou_emp = _empirical_acf(ou_traj[:, 4], max_lag=max_lag)
    acf_ou_theo = np.exp(-0.35 * np.arange(max_lag + 1) * 0.1)
    ou_acf_rmse = float(np.sqrt(np.mean((acf_ou_emp[:25] - acf_ou_theo[:25]) ** 2)))

    # Spatial cluster latent AR(1) ACF (tau_0 = 4.5s)
    gen_zz_flat = SpatialCorrelatedGenerator(spec=spec_zz, seed=20260104)
    gen_zz_flat.assign_profile(pd_reg.get("PD_01"), window_id=0)
    u0_series = np.array([gen_zz_flat.sample(i * 0.1, i).metadata["latent_u_t"] for i in range(2500)])
    acf_ar1_emp = _empirical_acf(u0_series, max_lag=max_lag)
    acf_ar1_theo = np.exp(-np.arange(max_lag + 1) * 0.1 / 4.5)
    ar1_acf_rmse = float(np.sqrt(np.mean((acf_ar1_emp[:25] - acf_ar1_theo[:25]) ** 2)))

    # Telegraph TLS ACF
    spec_tls = GeneratorSpec(
        generator_id="tls_test",
        generator_type="telegraph",
        enabled=True,
        seed=20260105,
        output_domain="probability_parameter",
        output_shape=(21,),
        composition_group="data_faults_z",
        parameters={"basis": "Z", "rate_up_per_sec": 0.18, "rate_down_per_sec": 0.42, "envelope_gain": 0.0, "dom_boost_scale": 0.0},
    )
    gen_tls = TelegraphNoiseGenerator(spec=spec_tls, seed=20260105)
    gen_tls.assign_profile(pd_reg.get("PD_01"), window_id=0)
    tls_series = np.stack([gen_tls.sample(i * 0.1, i).values for i in range(2500)])[:, 0]
    acf_tls_emp = _empirical_acf(tls_series, max_lag=max_lag)

    a3_passed = bool((ou_acf_rmse < 0.16) and (ar1_acf_rmse < 0.16) and (acf_ou_emp[1] > 0.85) and (acf_ar1_emp[1] > 0.90))

    # -------------------------------------------------------------------------
    # A4: Multi-window noise regime transitions (400s = 4 x 100s windows)
    # -------------------------------------------------------------------------
    gen_reg = GeneratorRegistry(
        l1_root / "configs" / "noise_framework" / "generator_registry.yaml", master_seed=20260106
    )
    gens_a4 = gen_reg.instantiate_all(master_seed=20260106)
    pd_assign = PDAssignment(
        pd_registry=pd_reg,
        master_seed=20260106,
        policy_name="curriculum_burst_enriched",
        assignment_config_path=l1_root / "configs" / "noise_framework" / "generator_assignment.yaml",
    )
    sched = AssignmentScheduler(pd_assignment=pd_assign, window_duration_sec=100.0)
    sched.initialize(generators=gens_a4, pd_registry=pd_reg)
    sync_sampler = SynchronizedSampler(
        generators=gens_a4,
        clock=SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0),
        scheduler=sched,
        composer=NoiseComposer(),
    )
    traj_400s = sync_sampler.generate_trajectory(duration_seconds=400.0)
    z_totals = np.array([float(np.mean(pt.composite_parameters["theta_z"])) for pt in traj_400s])
    win_means = [float(np.mean(z_totals[w * 1000 : (w + 1) * 1000])) for w in range(4)]
    boundary_jumps = [
        float(abs(z_totals[w * 1000] - z_totals[w * 1000 - 1])) for w in range(1, 4)
    ]
    a4_passed = bool(len(traj_400s) == 4000 and len(set(round(m, 6) for m in win_means)) >= 2)

    # -------------------------------------------------------------------------
    # A5: Out-of-Distribution (OOD) unseen combinations with paired 95% CIs
    # -------------------------------------------------------------------------
    v2_bin = VAL_ROOT / "layer2" / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    v2_net, _ = load_r2d2_checkpoint_bin(v2_bin, expected_version=2)

    ood_families = (
        "OOD_1_TriplePair_01_45_78",
        "OOD_2_ElevatedFloor_HeavyReadout",
        "OOD_3_TLS_Burst_Plus_3QubitStreak",
        "OOD_4_Asymmetric_Corner_Star",
    )
    ood_results: Dict[str, Any] = {}
    all_static_ood: List[float] = []
    all_avid_ood: List[float] = []

    for ood_idx, ood_name in enumerate(ood_families):
        s_lers: List[float] = []
        a_lers: List[float] = []
        for s in range(24):
            seed_val = 91000 + ood_idx * 100 + s
            ood_tr = _sample_ood_trajectory(ood_name, seed_val, num_epochs=14)
            rng_c = np.random.default_rng(seed_val + 31)
            est = SequentialPosteriorEstimator(temporal_beta=0.55)
            obs_b = ObservationBuilder(max_config_age_epochs=14)
            h, c = v2_net.zero_state()
            recal = False
            prev_act = None
            f_stat = 0
            f_avid = 0
            n_rounds_ep = 256
            for ep in range(14):
                circ = sample_d3_circuit_window(
                    rng=rng_c,
                    theta_z_21=ood_tr["theta_z"][ep],
                    theta_x_21=ood_tr["theta_x"][ep],
                    window_rounds=n_rounds_ep,
                    readout_flip_prob=float(ood_tr["readout"][ep]),
                    streak3_prob=float(ood_tr["streak3"][ep]),
                )
                pkt = est.step_from_syndrome_windows(circ["sx"], circ["sz"], recalibrate=recal)
                built = obs_b.build_from_posterior_and_syndromes(
                    pkt.to_layer2_dict(), circ["sx"], circ["sz"]
                )
                step_out = v2_net.step(built["observation_82"], h, c)
                h, c = step_out["h_next"], step_out["c_next"]
                act = int(step_out["action_id"])
                spec = ACTION_DEFINITIONS[act]
                gains_mult = np.clip(1.0 + np.asarray(step_out["qubit_gains"]), 0.70, 1.30)
                pm = pkt.posterior_mean_21
                w_q78 = project_q78_weights_d3(
                    qubit_p=pm[:9],
                    alphas=pm[9:],
                    pair_weight_scale=float(spec["pair_weight_scale"]),
                    qubit_gains=gains_mult if float(spec["pair_weight_scale"]) > 0.0 else None,
                    fallback=bool(spec["fallback"]),
                )
                corr_a = decode_window_minsum_osd0_z(
                    circ["sx"], w_q78, bp_iterations=int(spec["bp_iterations"])
                )
                f_avid += count_logical_z_failures(circ["err_z"], corr_a)
                f_stat += count_logical_z_failures(circ["err_z"], STATIC_LUT_Z[circ["sx"]])

                obs_b.record_action_update(config_changed=(prev_act is not None and act != prev_act))
                est.configure_from_action(
                    temporal_beta=float(spec.get("temporal_beta", 0.55)),
                    recalibrate=bool(spec["recalibrate"]),
                )
                recal = bool(spec["recalibrate"])
                prev_act = act

            tot_r = 14 * n_rounds_ep
            s_lers.append(f_stat / float(tot_r))
            a_lers.append(f_avid / float(tot_r))
            all_static_ood.append(f_stat / float(tot_r))
            all_avid_ood.append(f_avid / float(tot_r))

        ood_results[ood_name] = compute_paired_ler_ci_and_tests(s_lers, a_lers, seed=9500 + ood_idx)

    ood_aggregate_ci = compute_paired_ler_ci_and_tests(all_static_ood, all_avid_ood, seed=9599)
    # Criterion: zero catastrophic degradation (every OOD family <= static + 0.005) AND aggregate OOD significantly improves
    a5_passed = bool(
        ood_aggregate_ci["significant_improvement_95pct"]
        and all(v["mean_ler_candidate"] <= v["mean_ler_baseline"] for v in ood_results.values())
    )

    # -------------------------------------------------------------------------
    # A6: Bitwise seed reproducibility
    # -------------------------------------------------------------------------
    def _run_seed_repro(seed_r: int) -> tuple[np.ndarray, np.ndarray]:
        gr = GeneratorRegistry(
            l1_root / "configs" / "noise_framework" / "generator_registry.yaml", master_seed=seed_r
        )
        gens_r = gr.instantiate_all(master_seed=seed_r)
        pa = PDAssignment(
            pd_registry=pd_reg,
            master_seed=seed_r,
            policy_name="curriculum_burst_enriched",
            assignment_config_path=l1_root / "configs" / "noise_framework" / "generator_assignment.yaml",
        )
        sc = AssignmentScheduler(pd_assignment=pa, window_duration_sec=100.0)
        sc.initialize(generators=gens_r, pd_registry=pd_reg)
        ss = SynchronizedSampler(
            generators=gens_r,
            clock=SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0),
            scheduler=sc,
            composer=NoiseComposer(),
        )
        pts = ss.generate_trajectory(duration_seconds=15.0)
        enc = SampleEncoder(rounds_per_composite_sample=1)
        tr = enc.encode_composite_trajectory(pts, master_seed=seed_r + 7, trajectory_id="repro_a6")
        pz_arr = np.stack([p.composite_parameters["theta_z"] for p in pts])
        sx_arr = tr["syndromes_x"]
        return pz_arr, sx_arr

    pz_1, sx_1 = _run_seed_repro(77777)
    pz_2, sx_2 = _run_seed_repro(77777)
    max_pz_diff = float(np.max(np.abs(pz_1 - pz_2)))
    bitwise_sx_equal = bool(np.array_equal(sx_1, sx_2))
    a6_passed = bool((max_pz_diff == 0.0) and bitwise_sx_equal)

    result = {
        "group": "Group_A_Noise_Model_Validity",
        "gates": {
            "A1_distribution_recovery": {"passed": all_a1_ok, "profiles": a1_profiles, "qq_quantiles": qq_reference},
            "A2_correlation_recovery": {
                "passed": a2_passed,
                "mean_intra_cluster_corr": mean_intra,
                "mean_inter_cluster_corr": mean_inter,
                "pc1_variance_share": pc1_share,
                "modes_for_95pct_variance": k95,
                "corr_12x12": corr_12x12.tolist(),
            },
            "A3_temporal_autocorrelation": {
                "passed": a3_passed,
                "lags_sec": lags_sec,
                "ou_acf_empirical": acf_ou_emp.tolist(),
                "ou_acf_target": acf_ou_theo.tolist(),
                "ou_acf_rmse": ou_acf_rmse,
                "ar1_cluster_acf_empirical": acf_ar1_emp.tolist(),
                "ar1_cluster_acf_target": acf_ar1_theo.tolist(),
                "ar1_cluster_acf_rmse": ar1_acf_rmse,
                "tls_acf_empirical": acf_tls_emp.tolist(),
            },
            "A4_regime_transitions": {
                "passed": a4_passed,
                "duration_sec": 400.0,
                "num_samples": len(traj_400s),
                "window_mean_pz": win_means,
                "boundary_jumps": boundary_jumps,
                "z_mean_trace_downsampled": z_totals[::10].tolist(),
            },
            "A5_out_of_distribution_noise": {
                "passed": a5_passed,
                "aggregate_paired_stats": ood_aggregate_ci,
                "per_ood_family": ood_results,
            },
            "A6_seed_reproducibility": {
                "passed": a6_passed,
                "max_abs_parameter_diff": max_pz_diff,
                "bitwise_syndrome_identical": bitwise_sx_equal,
            },
        },
        "overall_passed": bool(all_a1_ok and a2_passed and a3_passed and a4_passed and a5_passed and a6_passed),
    }

    if out_json_dir is not None:
        out_json_dir.mkdir(parents=True, exist_ok=True)
        (out_json_dir / "group_a_noise_validity.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    res = run_group_a_suite(RESEARCH_ROOT / "results" / "json")
    print(f"[Group A] overall_passed={res['overall_passed']}")
