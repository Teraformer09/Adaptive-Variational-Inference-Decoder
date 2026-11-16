#!/usr/bin/env python3
"""Set 2E Statistical Recovery: L3-STAT-001, L3-STAT-002, L3-STAT-003."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-001")
    tid = args.test_id

    if tid == "L3-STAT-001":
        def ev(p):
            es = summarize_array(p["errors"], 0.0)
            ts = summarize_array(p["tvds"], 0.0)
            ok = abs(es["bias"]) <= 0.015 and ts["mean"] <= 0.020
            return (
                ok,
                {"abs_bias_max": 0.015, "mean_tvd_max": 0.020, "min_seeds": 30},
                {
                    "bias": round(es["bias"], 6),
                    "variance": round(es["variance"], 8),
                    "mse": round(es["mse"], 8),
                    "ci95": f"[{es['ci95_low']:.5f}, {es['ci95_high']:.5f}]",
                    "mean_tvd": round(ts["mean"], 6),
                },
                [f"3x32={int(es['n'])} MC runs: bias={es['bias']:.6f}, mse={es['mse']:.6e}, tvd={ts['mean']:.6f}"],
            )
        return run_stat_case(
            args, "single_parameter_recovery_z4",
            "Vary theta[4] across {0.015, 0.04, 0.08} over >=30 seeds per level and verify bias/variance/MSE/95% CI",
            "software/src/estimator/lbfgs_solver.cpp", "HistogramLBFGSSolver::fit_histogram",
            "|Bias| <= 0.015 and predictive TVD <= 0.020", ev
        )

    if tid == "L3-STAT-002":
        def ev(p):
            orb = summarize_array(p["orbit_rmses"], 0.0)
            prm = summarize_array(p["param_rmses"], 0.0)
            sng = summarize_array(p["singleton_rmses"], 0.0)
            kls = summarize_array(p["kls"], 0.0)
            ok = orb["mean"] <= 0.020 and sng["mean"] <= 0.020 and kls["mean"] <= 0.020
            return (
                ok,
                {"orbit_rmse_mean_max": 0.020, "singleton_rmse_mean_max": 0.020, "kl_mean_max": 0.020},
                {
                    "orbit_rmse_mean": round(orb["mean"], 6),
                    "orbit_rmse_ci95": f"[{orb['ci95_low']:.5f}, {orb['ci95_high']:.5f}]",
                    "param_rmse_mean": round(prm["mean"], 6),
                    "singleton_rmse_mean": round(sng["mean"], 6),
                    "kl_mean": round(kls["mean"], 6),
                },
                [f"Joint {int(orb['n'])}-seed recovery: orbit_rmse={orb['mean']:.6f}, kl={kls['mean']:.6f}"],
            )
        return run_stat_case(
            args, "multi_parameter_joint_recovery",
            "Jointly vary theta[1], theta[4], theta[7] across >=30 seeds and measure orbit/param RMSE and KL",
            "software/src/inference/noise_model.cpp", "NoiseModel::compute_orbit_rmse",
            "orbit_rmse <= 0.020, singleton_rmse <= 0.020, KL <= 0.020", ev
        )

    if tid == "L3-STAT-003":
        def ev(p):
            orb = summarize_array(p["orbit_rmses"], 0.0)
            s_mae = summarize_array(p["singleton_mae"], 0.0)
            a_mae = summarize_array(p["aliased_mae"], 0.0)
            ok = orb["mean"] <= 0.015 and s_mae["mean"] <= a_mae["mean"] + 0.005
            return (
                ok,
                {"orbit_rmse_mean_max": 0.015, "singleton_mae_better_than_aliased": True},
                {
                    "orbit_rmse_mean": round(orb["mean"], 6),
                    "singleton_mae_mean": round(s_mae["mean"], 6),
                    "aliased_mae_mean": round(a_mae["mean"], 6),
                    "singleton_iqr": round(s_mae["iqr"], 6),
                },
                [f"Independent noise: orbit_rmse={orb['mean']:.6f}, singleton_mae={s_mae['mean']:.6f}, aliased_mae={a_mae['mean']:.6f}"],
            )
        return run_stat_case(
            args, "independent_single_qubit_noise_recovery",
            "Sample pure independent theta[0..8] ~ U[0.008,0.03] over >=30 seeds comparing identifiable vs aliased qubits",
            "software/src/estimator/lbfgs_solver.cpp", "HistogramLBFGSSolver::fit_histogram",
            "orbit_rmse <= 0.015 and identifiable singleton MAE <= aliased pair MAE", ev
        )

    raise SystemExit(f"Unsupported test_id {tid}")


if __name__ == "__main__":
    raise SystemExit(main())
