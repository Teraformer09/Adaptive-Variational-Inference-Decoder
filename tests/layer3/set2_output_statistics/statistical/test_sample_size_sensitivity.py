#!/usr/bin/env python3
"""Set 2E Sample-Size Sensitivity: L3-STAT-006."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case


def main() -> int:
    args = parse_stat_args("L3-STAT-006")

    def ev(p):
        orbs = p["mean_orbit_rmse"]
        vars_ = p["var_z4"]
        mono_orb = orbs[0] > orbs[1] > orbs[2] > orbs[3]
        mono_var = vars_[0] > vars_[1] > vars_[2] > vars_[3]
        ok = mono_orb and mono_var
        return (
            ok,
            {"monotone_decreasing_orbit_rmse": True, "monotone_decreasing_variance": True},
            {
                "orbit_rmse_W256_1k_5k_20k": [round(x, 6) for x in orbs],
                "var_z4_W256_1k_5k_20k": [round(x, 8) for x in vars_],
            },
            [f"Monotone shrinkage across W=[256,1k,5k,20k]: orbit_rmse={orbs}, var_z4={vars_}"],
        )

    return run_stat_case(
        args, "sample_size_monotone_variance_shrinkage",
        "Sweep W in {256, 1000, 5000, 20000} across >=30 seeds each and verify monotone variance & orbit RMSE decrease",
        "software/src/estimator/lbfgs_solver.cpp", "HistogramLBFGSSolver::fit_histogram",
        "Strict monotone decrease of estimator variance and orbit RMSE with W", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
