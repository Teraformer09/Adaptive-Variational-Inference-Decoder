#!/usr/bin/env python3
"""Set 2E Correlated Crosstalk Recovery: L3-STAT-004."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-004")

    def ev(p):
        orb = summarize_array(p["orbit_rmses"], 0.0)
        p45 = summarize_array(p["pair_45_hat"], 0.080)
        pin = summarize_array(p["inactive_pair_hat"], 0.004)
        ok = orb["mean"] <= 0.025 and p45["mean"] > pin["mean"] * 2.0
        return (
            ok,
            {"orbit_rmse_max": 0.025, "pair_45_exceeds_2x_inactive": True},
            {
                "orbit_rmse_mean": round(orb["mean"], 6),
                "pair_45_mean": round(p45["mean"], 6),
                "pair_45_bias": round(p45["bias"], 6),
                "inactive_pair_mean": round(pin["mean"], 6),
            },
            [f"Correlated pair (4,5) hat={p45['mean']:.5f} vs inactive={pin['mean']:.5f}, orbit_rmse={orb['mean']:.5f}"],
        )

    return run_stat_case(
        args, "correlated_pair_crosstalk_recovery",
        "Activate correlated pair (4,5) across >=30 seeds and verify Pass-2 endpoint-coupled pair separation & orbit RMSE",
        "software/src/estimator/lbfgs_solver.cpp", "HistogramLBFGSSolver::fit_histogram",
        "orbit_rmse <= 0.025 and active pair estimate > 2x inactive pair estimate", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
