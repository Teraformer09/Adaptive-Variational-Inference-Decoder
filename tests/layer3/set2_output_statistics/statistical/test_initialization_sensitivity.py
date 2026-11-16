#!/usr/bin/env python3
"""Set 2E Initialization Sensitivity Across Identifiable vs Null Coordinates: L3-STAT-009."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-009")

    def ev(p):
        s_sp = summarize_array(p["singleton_spreads"], 0.0)
        n_sp = summarize_array(p["null_spreads"], 0.0)
        ok = s_sp["mean"] <= 0.012 and n_sp["mean"] > 0.14
        return (
            ok,
            {"singleton_init_spread_max": 0.012, "null_space_init_retention_min": 0.14},
            {
                "singleton_z4_spread_mean": round(s_sp["mean"], 6),
                "null_zz36_spread_mean": round(n_sp["mean"], 6),
            },
            [f"Init sweep p0=[0.002..0.15]: singleton spread={s_sp['mean']:.6f} vs null-space spread={n_sp['mean']:.6f}"],
        )

    return run_stat_case(
        args, "initial_point_sensitivity_identifiable_vs_null",
        "Initialize p0 in {0.002, 0.012, 0.05, 0.15} across >=30 histograms contrasting singleton convergence vs null-space invariance",
        "software/src/inference/lbfgs_optimizer.cpp", "LBFGSOptimizer::step_histogram",
        "Identifiable singleton spread <= 0.012 while unregularized g=0 null-space retains exact initial spread (0.148)", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
