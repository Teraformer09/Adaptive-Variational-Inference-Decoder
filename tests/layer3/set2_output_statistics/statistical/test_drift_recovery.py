#!/usr/bin/env python3
"""Set 2E Sequential Non-Stationary Drift Tracking: L3-STAT-012."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-012")

    def ev(p):
        dyn = summarize_array(p["dynamic_rmses"], 0.0)
        sta = summarize_array(p["static_rmses"], 0.0)
        cor = summarize_array(p["tracking_corrs"], 1.0)
        ok = dyn["mean"] < sta["mean"] * 0.45 and cor["mean"] >= 0.85
        return (
            ok,
            {"dynamic_rmse_lt_045_static": True, "tracking_corr_min": 0.85},
            {
                "dynamic_rmse_mean": round(dyn["mean"], 6),
                "static_rmse_mean": round(sta["mean"], 6),
                "tracking_correlation_mean": round(cor["mean"], 5),
            },
            [f"20-window ramp tracking: dynamic_rmse={dyn['mean']:.5f} vs static={sta['mean']:.5f}, r={cor['mean']:.4f}"],
        )

    return run_stat_case(
        args, "nonstationary_drift_ramp_tracking",
        "Ramp theta[4] across 20 sequential windows over >=30 seeds measuring dynamic tracking RMSE and correlation",
        "software/src/estimator/lbfgs_solver.cpp", "HistogramLBFGSSolver::fit_histogram",
        "Dynamic tracking RMSE < 0.45 * static RMSE and Pearson correlation >= 0.85", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
