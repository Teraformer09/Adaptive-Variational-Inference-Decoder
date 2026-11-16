#!/usr/bin/env python3
"""Set 2E Asymptotic Fisher Confidence Interval Calibration: L3-STAT-015."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-015")

    def ev(p):
        cov = summarize_array(p["covered"], 0.95)
        zs = summarize_array(p["z_scores"], 0.0)
        emp_cov = cov["mean"]
        ok = 0.88 <= emp_cov <= 0.995 and 0.75 <= zs["std"] <= 1.25
        return (
            ok,
            {"empirical_95ci_coverage_band": "[0.88, 0.995]", "z_score_std_band": "[0.75, 1.25]"},
            {
                "empirical_95ci_coverage": round(emp_cov, 4),
                "z_score_mean": round(zs["mean"], 4),
                "z_score_std": round(zs["std"], 4),
                "fisher_se_z1_z4_z7": [round(x, 6) for x in p["fisher_se"]],
            },
            [f"180 singleton CI checks (60 trials x 3 singletons): coverage={emp_cov*100:.1f}%, z_std={zs['std']:.3f}"],
        )

    return run_stat_case(
        args, "fisher_information_confidence_interval_calibration",
        "Compare empirical 95% CI coverage |theta_hat - theta| <= 1.96*SE_Fisher on identifiable singletons {Z_1, Z_4, Z_7} over 60 MC trials",
        "software/src/inference/noise_model.cpp", "NoiseModel::compute_joint_distribution",
        "Empirical 95% Fisher CI coverage in [0.88, 0.995] and standardized z-score std in [0.75, 1.25]", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
