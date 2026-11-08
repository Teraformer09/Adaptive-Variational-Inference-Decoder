#!/usr/bin/env python3
"""
BM-1: 15-Seed Shared-Input LER & Peak-vs-Full Window Statistical Benchmark
Peer-Reviewed Protocols:
  - Bausch et al. (Google DeepMind & Google Quantum AI), "Learning high-accuracy error
    decoding for quantum processors" (AlphaQubit), Nature 635, 834–840 (2024).
  - Spitz et al., "Adaptive weight estimator for quantum error correction using
    time-dependent syndrome correlations", Adv. Quantum Technol. 1, 1800012 (2018).
Closes EPJ Checklist Items: Q07, Q08, Q16, Gate 7.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import time
from typing import Dict, List
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.reference.noise_model.joint_dp_likelihood import build_d3_correlated_noise_model
from simulation.reference.syndrome_stream.causal_stream_runner import CycleAccurateBPEngine
from simulation.reference.baselines import (
    B0StaticBPController,
    B1MarginalLBFGSController,
    B2ObservablePIDController,
    B3CovarianceTrackerController,
    B4OracleBPController,
)
from training.export.export_checkpoint import load_r2d2_checkpoint_bin

OUT_DIR = Path(__file__).resolve().parent


def build_five_regimes(default_theta: np.ndarray) -> Dict[str, np.ndarray]:
    regimes: Dict[str, np.ndarray] = {}
    r1 = default_theta.copy()
    regimes["R1_stationary_baseline"] = r1

    r2 = default_theta.copy()
    r2[4] = 0.085
    r2[5] = 0.055
    regimes["R2_tls_hotspot_q4_q5"] = r2

    r3 = default_theta.copy()
    r3[9 + 0] = 0.065
    r3[9 + 3] = 0.080
    r3[9 + 5] = 0.060
    regimes["R3_zz_crosstalk_burst"] = r3

    r4 = default_theta.copy()
    r4[0] = 0.050
    r4[1] = 0.045
    r4[3] = 0.055
    r4[6] = 0.040
    regimes["R4_boundary_drift"] = r4

    r5 = default_theta.copy()
    r5[4] = 0.095
    r5[7] = 0.065
    r5[9 + 3] = 0.075
    r5[9 + 9] = 0.070
    regimes["R5_combined_stress"] = r5
    return regimes


def apply_trained_r2d2_policy(
    r2d2_net,
    model_x,
    hist_16: np.ndarray,
    theta_est: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Run trained R2D2 controller on causal features and return (q78_weights, K)."""
    total_counts = max(1.0, float(np.sum(hist_16)))
    probs = hist_16 / total_counts
    mean_p = float(np.mean(theta_est[:9]))
    max_p = float(np.max(theta_est[:9]))
    mean_alpha = float(np.mean(theta_est[9:21]))
    max_alpha = float(np.max(theta_est[9:21]))
    syn_density = float(1.0 - probs[0])
    adj_rate = float(sum(probs[s] for s in range(1, 16) if bin(s).count("1") >= 2))
    feat = np.array(
        [mean_p, max_p, mean_alpha, max_alpha, syn_density, adj_rate, 0.008, 0.10],
        dtype=np.float64,
    )
    h, c = r2d2_net.zero_state()
    step_out = r2d2_net.step(feat, h, c)
    gains = 1.0 + step_out["qubit_gains"]
    _, q78_base = model_x.compute_q7_8_llrs(theta_est, pair_aware_projection=True)
    q78_mod = np.clip(np.round(q78_base.astype(np.float64) * gains), 64, 32767).astype(np.int16)
    return q78_mod, int(step_out["bp_iterations"])


_LUT_CACHE: Dict[tuple, np.ndarray] = {}


def decode_ler_exact_lut(
    bp: CycleAccurateBPEngine,
    sx_masks: np.ndarray,
    errs_z: np.ndarray,
    q78_weights: np.ndarray,
    K: int = 3,
) -> float:
    """
    Bit-exact evaluation of CycleAccurateBPEngine over arbitrary stream lengths
    by pre-evaluating all 2^4 = 16 syndrome states for the given Q7.8 weight bank.
    """
    key = (K, tuple(int(x) for x in q78_weights))
    if key not in _LUT_CACHE:
        corr_lut = np.zeros((16, bp.n_vars), dtype=np.uint8)
        for s_mask in range(16):
            corr_lut[s_mask] = bp.decode(s_mask, q78_weights, K=K).correction_bits
        _LUT_CACHE[key] = corr_lut
    corr_stream = _LUT_CACHE[key][sx_masks]
    residual = (errs_z ^ corr_stream) & 1
    logical_flips = (residual @ bp.logical_op) & 1
    return float(np.mean(logical_flips))


def run_bm1(n_seeds: int = 15, cal_rounds: int = 10000, eval_rounds: int = 10000) -> Dict[str, object]:
    t0 = time.perf_counter()
    model_x = build_d3_correlated_noise_model(basis="X", include_measurement_faults=False)
    bp = CycleAccurateBPEngine(code=model_x.code, basis="X", default_K=3, clock_mhz=250.0)
    regimes = build_five_regimes(model_x.default_theta)

    b0 = B0StaticBPController()
    b1 = B1MarginalLBFGSController()
    b2 = B2ObservablePIDController()
    b3 = B3CovarianceTrackerController()
    b4 = B4OracleBPController()

    r2d2_net, _ = load_r2d2_checkpoint_bin(REPO_ROOT / "checkpoints" / "r2d2_v1_trained.bin")
    csv_rows: List[Dict[str, object]] = []
    per_regime_stats: Dict[str, object] = {}

    for r_name, true_theta in regimes.items():
        seed_records: Dict[str, List[float]] = {
            "B0_Static_BP_full": [],
            "B1_Marginal_LBFGS_full": [],
            "B2_Observable_PID_full": [],
            "B3_Covariance_Tracker_full": [],
            "AVID_Joint_DP_MAP_full": [],
            "AVID_Weighted_Bellman_R2D2_full": [],
            "B4_Oracle_BP_full": [],
            "B0_Static_BP_peak": [],
            "AVID_Joint_DP_MAP_peak": [],
            "AVID_Weighted_Bellman_R2D2_peak": [],
            "B4_Oracle_BP_peak": [],
        }

        for s_idx in range(n_seeds):
            seed = 1001 + s_idx * 37
            rng = np.random.default_rng(seed)
            total_n = cal_rounds + eval_rounds

            p_single = true_theta[:9]
            p_pairs = true_theta[9:21]

            errs_z = rng.binomial(1, p_single, size=(total_n, 9)).astype(np.uint8)
            for e_idx, (u, v) in enumerate(model_x.code.zz_edges):
                pe = float(p_pairs[e_idx])
                if pe > 0.0:
                    c_flips = rng.binomial(1, pe, size=total_n).astype(np.uint8)
                    errs_z[:, u] ^= c_flips
                    errs_z[:, v] ^= c_flips

            # Peak burst window (second half of evaluation stream) amplifies active crosstalk/hotspot by 1.20x
            peak_start = cal_rounds + (eval_rounds // 2)
            peak_theta = np.clip(true_theta * 1.20, 1e-4, 0.42)
            errs_peak = rng.binomial(1, peak_theta[:9], size=(eval_rounds // 2, 9)).astype(np.uint8)
            for e_idx, (u, v) in enumerate(model_x.code.zz_edges):
                pe = float(peak_theta[9 + e_idx])
                if pe > 0.0:
                    c_flips = rng.binomial(1, pe, size=eval_rounds // 2).astype(np.uint8)
                    errs_peak[:, u] ^= c_flips
                    errs_peak[:, v] ^= c_flips
            errs_z[peak_start:] = errs_peak

            sx_bits = (errs_z @ model_x.H.T) % 2
            sx_masks = np.sum(sx_bits * (1 << np.arange(4)), axis=1).astype(np.uint8)

            cal_masks = sx_masks[:cal_rounds]
            eval_masks_full = sx_masks[cal_rounds:]
            eval_errs_full = errs_z[cal_rounds:]
            eval_masks_peak = sx_masks[peak_start:]
            eval_errs_peak = errs_z[peak_start:]

            hist_16 = np.bincount(cal_masks, minlength=16).astype(np.float64)

            # Fit every seed independently on its own empirical syndrome histogram (Fixes N2)
            fit_avid = model_x.fit_lbfgs(hist_16, prior_lambda=0.04, max_iter=25)
            warm_theta = np.asarray(fit_avid["theta"], dtype=np.float64)
            w_avid = fit_avid["llr_q7_8"]
            w_r2d2, k_r2d2 = apply_trained_r2d2_policy(r2d2_net, model_x, hist_16, warm_theta)

            w_b0 = b0.update_from_histogram(hist_16)
            w_b1 = b1.update_from_histogram(hist_16)
            w_b2 = b2.update_from_histogram(hist_16)
            w_b3 = b3.update_from_histogram(hist_16)
            w_b4 = b4.weights_from_true_theta(true_theta)

            ler_b0_full = decode_ler_exact_lut(bp, eval_masks_full, eval_errs_full, w_b0)
            ler_b1_full = decode_ler_exact_lut(bp, eval_masks_full, eval_errs_full, w_b1)
            ler_b2_full = decode_ler_exact_lut(bp, eval_masks_full, eval_errs_full, w_b2)
            ler_b3_full = decode_ler_exact_lut(bp, eval_masks_full, eval_errs_full, w_b3)
            ler_avid_full = decode_ler_exact_lut(bp, eval_masks_full, eval_errs_full, w_avid)
            ler_r2d2_full = decode_ler_exact_lut(bp, eval_masks_full, eval_errs_full, w_r2d2, K=k_r2d2)
            ler_b4_full = decode_ler_exact_lut(bp, eval_masks_full, eval_errs_full, w_b4)

            ler_b0_peak = decode_ler_exact_lut(bp, eval_masks_peak, eval_errs_peak, w_b0)
            ler_avid_peak = decode_ler_exact_lut(bp, eval_masks_peak, eval_errs_peak, w_avid)
            ler_r2d2_peak = decode_ler_exact_lut(bp, eval_masks_peak, eval_errs_peak, w_r2d2, K=k_r2d2)
            ler_b4_peak = decode_ler_exact_lut(bp, eval_masks_peak, eval_errs_peak, w_b4)

            seed_records["B0_Static_BP_full"].append(ler_b0_full)
            seed_records["B1_Marginal_LBFGS_full"].append(ler_b1_full)
            seed_records["B2_Observable_PID_full"].append(ler_b2_full)
            seed_records["B3_Covariance_Tracker_full"].append(ler_b3_full)
            seed_records["AVID_Joint_DP_MAP_full"].append(ler_avid_full)
            seed_records["AVID_Weighted_Bellman_R2D2_full"].append(ler_r2d2_full)
            seed_records["B4_Oracle_BP_full"].append(ler_b4_full)

            seed_records["B0_Static_BP_peak"].append(ler_b0_peak)
            seed_records["AVID_Joint_DP_MAP_peak"].append(ler_avid_peak)
            seed_records["AVID_Weighted_Bellman_R2D2_peak"].append(ler_r2d2_peak)
            seed_records["B4_Oracle_BP_peak"].append(ler_b4_peak)

            csv_rows.append({
                "regime": r_name,
                "seed": seed,
                "B0_full": round(ler_b0_full, 5),
                "B1_full": round(ler_b1_full, 5),
                "B2_full": round(ler_b2_full, 5),
                "B3_full": round(ler_b3_full, 5),
                "AVID_Joint_DP_full": round(ler_avid_full, 5),
                "WB_R2D2_full": round(ler_r2d2_full, 5),
                "B4_Oracle_full": round(ler_b4_full, 5),
                "B0_peak": round(ler_b0_peak, 5),
                "AVID_Joint_DP_peak": round(ler_avid_peak, 5),
                "WB_R2D2_peak": round(ler_r2d2_peak, 5),
                "B4_Oracle_peak": round(ler_b4_peak, 5),
            })

        arr_b0 = np.array(seed_records["B0_Static_BP_full"])
        arr_r2d2 = np.array(seed_records["AVID_Weighted_Bellman_R2D2_full"])
        arr_avid = np.array(seed_records["AVID_Joint_DP_MAP_full"])
        arr_b0_pk = np.array(seed_records["B0_Static_BP_peak"])
        arr_r2d2_pk = np.array(seed_records["AVID_Weighted_Bellman_R2D2_peak"])

        diff_full = arr_b0 - arr_r2d2
        if np.std(diff_full) > 1e-12:
            t_res = stats.ttest_rel(arr_b0, arr_r2d2)
            t_stat, p_val_t = float(t_res.statistic), float(t_res.pvalue)
            cohen_d = float(np.mean(diff_full) / np.std(diff_full, ddof=1))
            try:
                w_res = stats.wilcoxon(arr_b0, arr_r2d2, zero_method="wilcox", alternative="two-sided")
                w_stat, p_val_w = float(w_res.statistic), float(w_res.pvalue)
            except ValueError:
                w_stat, p_val_w = 0.0, 1.0
        else:
            t_stat, p_val_t, cohen_d, w_stat, p_val_w = 0.0, 1.0, 0.0, 0.0, 1.0

        per_regime_stats[r_name] = {
            "n_seeds": n_seeds,
            "full_horizon_mean_ler": {
                "B0_Static_BP": round(float(np.mean(arr_b0)), 5),
                "B1_Marginal_LBFGS": round(float(np.mean(seed_records["B1_Marginal_LBFGS_full"])), 5),
                "B2_Observable_PID": round(float(np.mean(seed_records["B2_Observable_PID_full"])), 5),
                "B3_Covariance_Tracker": round(float(np.mean(seed_records["B3_Covariance_Tracker_full"])), 5),
                "AVID_Joint_DP_MAP": round(float(np.mean(arr_avid)), 5),
                "AVID_Weighted_Bellman_R2D2": round(float(np.mean(arr_r2d2)), 5),
                "B4_Oracle_BP": round(float(np.mean(seed_records["B4_Oracle_BP_full"])), 5),
            },
            "full_horizon_std_ler": {
                "B0_Static_BP": round(float(np.std(arr_b0, ddof=1)), 5),
                "AVID_Joint_DP_MAP": round(float(np.std(arr_avid, ddof=1)), 5),
                "AVID_Weighted_Bellman_R2D2": round(float(np.std(arr_r2d2, ddof=1)), 5),
            },
            "peak_burst_window_mean_ler": {
                "B0_Static_BP": round(float(np.mean(arr_b0_pk)), 5),
                "AVID_Joint_DP_MAP": round(float(np.mean(seed_records["AVID_Joint_DP_MAP_peak"])), 5),
                "AVID_Weighted_Bellman_R2D2": round(float(np.mean(arr_r2d2_pk)), 5),
                "B4_Oracle_BP": round(float(np.mean(seed_records["B4_Oracle_BP_peak"])), 5),
            },
            "paired_significance_B0_vs_WB_R2D2": {
                "mean_ler_reduction": round(float(np.mean(diff_full)), 5),
                "relative_reduction_pct": round(float(100.0 * np.mean(diff_full) / max(1e-9, np.mean(arr_b0))), 2),
                "paired_t_stat": round(t_stat, 4),
                "paired_t_pvalue": float(f"{p_val_t:.4e}"),
                "cohens_d": round(cohen_d, 4),
                "wilcoxon_signed_rank_stat": round(w_stat, 2),
                "wilcoxon_two_sided_pvalue": float(f"{p_val_w:.4e}"),
            },
        }

    csv_path = OUT_DIR / "bm1_multiseed_ler_trajectories.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    report = {
        "benchmark_id": "BM-1",
        "title": "15-Seed Shared-Input LER & Peak-vs-Full Window Statistical Significance Benchmark",
        "closes_issues": ["Q07", "Q08", "Q16", "Gate_7"],
        "peer_reviewed_references": [
            "Bausch et al. (Google DeepMind & Google Quantum AI), Nature 635, 834-840 (2024)",
            "Spitz et al., Adv. Quantum Technol. 1, 1800012 (2018)",
        ],
        "n_seeds": n_seeds,
        "calibration_rounds_per_seed": cal_rounds,
        "evaluation_rounds_per_seed": eval_rounds,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "regimes": per_regime_stats,
    }
    (OUT_DIR / "bm1_statistical_significance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[BM-1 Complete] Saved {csv_path.name} and bm1_statistical_significance.json ({report['elapsed_sec']}s)")
    return report


if __name__ == "__main__":
    run_bm1()
