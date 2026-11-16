#!/usr/bin/env python3
"""Set 2E Prior Regularization Shrinkage vs Variance Trade-off: L3-STAT-008."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case


def main() -> int:
    args = parse_stat_args("L3-STAT-008")

    def ev(p):
        dists = p["mean_dist_to_prior"]
        vars_ = p["var_z4"]
        mono_shrink = dists[0] > dists[1] > dists[2] > dists[3]
        var_drop = vars_[3] < vars_[0]
        ok = mono_shrink and var_drop
        return (
            ok,
            {"monotone_shrinkage_to_prior": True, "variance_reduced_at_high_lambda": True},
            {
                "dist_to_prior_lam_0_004_05_5": [round(x, 6) for x in dists],
                "var_z4_lam_0_004_05_5": [round(x, 8) for x in vars_],
            },
            [f"Prior sweep lambda=[0,0.04,0.5,5.0]: ||theta-theta0||={dists}, var_z4={vars_}"],
        )

    return run_stat_case(
        args, "prior_lambda_bias_variance_shrinkage",
        "Sweep prior_lambda in {0.0, 0.04, 0.5, 5.0} over >=30 shared-histogram seeds to verify monotone shrinkage & variance drop",
        "software/src/inference/noise_model.cpp", "NoiseModel::compute_nll_from_histogram",
        "Strict monotone decrease in ||theta_hat - prior_theta||_2 and lower variance at high lambda", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
