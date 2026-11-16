#!/usr/bin/env python3
"""Set 2E Orbit vs Aliased/Null Coordinate Identifiability: L3-STAT-011."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-011")

    def ev(p):
        orb = summarize_array(p["orbit_rmses"], 0.0)
        nul = summarize_array(p["null_coord_abs_err"], 0.0)
        ok = orb["mean"] <= 0.015 and nul["mean"] > 0.070
        return (
            ok,
            {"orbit_rmse_mean_max": 0.015, "null_coord_prior_pinned_err_min": 0.070},
            {
                "orbit_rmse_mean": round(orb["mean"], 6),
                "orbit_rmse_ci95": f"[{orb['ci95_low']:.5f}, {orb['ci95_high']:.5f}]",
                "null_zz36_abs_err_mean": round(nul["mean"], 6),
            },
            [f"Aliased/null recovery: 9-class orbit_rmse={orb['mean']:.6f} while g=0 ZZ_3_6 stays prior-pinned (err={nul['mean']:.5f})"],
        )

    return run_stat_case(
        args, "quotient_orbit_vs_aliased_coordinate_identifiability",
        "Perturb aliased orbit 0x1 (Z_0, ZZ_1_2) and invisible g=0 edge ZZ_3_6 across >=30 seeds proving tight 9-orbit recovery",
        "software/src/inference/noise_model.cpp", "NoiseModel::compute_orbit_rmse",
        "Tight 9-class observable orbit RMSE <= 0.015 while unobservable g=0 coordinate stays pinned to prior", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
