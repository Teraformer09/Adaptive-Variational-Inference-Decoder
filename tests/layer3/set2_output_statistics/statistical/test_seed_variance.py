#!/usr/bin/env python3
"""Set 2E Multi-Seed Dispersion & Quantile Characterization: L3-STAT-007."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-007")

    def ev(p):
        nll = summarize_array(p["nlls"], 0.0)
        orb = summarize_array(p["orbit_rmses"], 0.0)
        prm = summarize_array(p["param_rmses"], 0.0)
        ok = nll["std"] > 0.0 and orb["p95"] <= 0.025 and prm["p95"] <= 0.045
        return (
            ok,
            {"non_degenerate_seed_variance": True, "orbit_rmse_p95_max": 0.025, "param_rmse_p95_max": 0.045},
            {
                "nll_mean_std_iqr": f"{nll['mean']:.5f} / {nll['std']:.5f} / {nll['iqr']:.5f}",
                "orbit_rmse_p05_p95": f"[{orb['p05']:.5f}, {orb['p95']:.5f}]",
                "param_rmse_p05_p95": f"[{prm['p05']:.5f}, {prm['p95']:.5f}]",
            },
            [f"32-seed dispersion: NLL mean={nll['mean']:.5f} (IQR={nll['iqr']:.5f}), orbit p95={orb['p95']:.5f}"],
        )

    return run_stat_case(
        args, "multi_seed_dispersion_and_percentiles",
        "Characterize mean, std, IQR, and 5th/95th percentiles across >=30 independent seeds for NLL, orbit RMSE, and 21-param RMSE",
        "software/src/estimator/lbfgs_solver.cpp", "HistogramLBFGSSolver::fit_histogram",
        "Bounded 95th-percentile orbit/parameter RMSE with non-degenerate multi-seed dispersion", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
