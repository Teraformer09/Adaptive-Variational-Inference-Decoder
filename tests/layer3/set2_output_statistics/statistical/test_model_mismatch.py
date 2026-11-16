#!/usr/bin/env python3
"""Set 2E Readout Misspecification & Out-of-Model Streak Diagnostics: L3-STAT-005, L3-STAT-010."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-005")
    tid = args.test_id

    if tid == "L3-STAT-005":
        def ev(p):
            cl = summarize_array(p["clean_err"], 0.0)
            ms = summarize_array(p["meas_err"], 0.0)
            ok = ms["bias"] > cl["bias"] and abs(ms["bias"]) < 0.060
            return (
                ok,
                {"meas_upward_bias_detected": True, "bounded_bias_max": 0.060},
                {
                    "clean_bias": round(cl["bias"], 6),
                    "meas_noisy_bias": round(ms["bias"], 6),
                    "clean_mse": round(cl["mse"], 8),
                    "meas_mse": round(ms["mse"], 8),
                },
                [f"Phenomenological readout p_m=0.012 shifts bias from {cl['bias']:+.5f} to {ms['bias']:+.5f}"],
            )
        return run_stat_case(
            args, "measurement_noise_misspecification_bias",
            "Inject phenomenological readout bit-flip p_meas=0.012 over >=30 seeds and quantify upward parameter bias",
            "software/src/estimator/lbfgs_solver.cpp", "HistogramLBFGSSolver::fit_histogram",
            "Positive bounded upward bias under readout noise relative to clean syndromes", ev
        )

    if tid == "L3-STAT-010":
        def ev(p):
            ki = summarize_array(p["kl_in_model"], 0.0)
            km = summarize_array(p["kl_mismatched"], 0.0)
            ok = km["mean"] > ki["mean"] * 5.0 and km["p05"] > ki["p95"]
            return (
                ok,
                {"kl_mismatched_gt_5x_in_model": True, "clean_separation_p05_gt_p95": True},
                {
                    "kl_in_model_mean": round(ki["mean"], 6),
                    "kl_mismatched_mean": round(km["mean"], 6),
                    "kl_in_model_p95": round(ki["p95"], 6),
                    "kl_mismatched_p05": round(km["p05"], 6),
                },
                [f"KL residual separation: in_model={ki['mean']:.5f} vs out_of_model={km['mean']:.5f}"],
            )
        return run_stat_case(
            args, "out_of_model_streak_kl_residual_detection",
            "Inject out-of-model multi-check streak over >=30 seeds and verify KL divergence D_KL(P_emp||P_hat) flags mismatch",
            "software/src/inference/noise_model.cpp", "NoiseModel::compute_nll_from_histogram",
            "D_KL(P_emp||P_hat) under mismatch > 5x in-model baseline with non-overlapping 5th/95th percentiles", ev
        )

    raise SystemExit(f"Unsupported test_id {tid}")


if __name__ == "__main__":
    raise SystemExit(main())
