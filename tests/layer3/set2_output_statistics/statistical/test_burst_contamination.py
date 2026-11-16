#!/usr/bin/env python3
"""Set 2E 60-Round Burst Dilution vs 16-Round Fast Subwindow Detection: L3-STAT-013."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-013")

    def ev(p):
        sh = summarize_array(p["slow_z4_shifts"], 0.0)
        lg = summarize_array(p["fast_trigger_lags"], 0.0)
        all_fast_ok = all(4.0 <= x <= 20.0 for x in p["fast_trigger_lags"])
        ok = sh["mean"] <= 0.010 and all_fast_ok
        return (
            ok,
            {"slow_10k_dilution_shift_max": 0.010, "fast_detector_lag_in_4_20_rounds": True},
            {
                "slow_z4_shift_mean": round(sh["mean"], 6),
                "fast_trigger_lag_mean": round(lg["mean"], 3),
                "fast_trigger_lag_max": round(max(p["fast_trigger_lags"]), 1),
            },
            [f"60/10000 burst dilution shift={sh['mean']:.5f}; fast 16-round detector lag={lg['mean']:.2f} rounds (<=20)"],
        )

    return run_stat_case(
        args, "short_burst_dilution_vs_fast_subwindow_trigger",
        "Inject 60-round burst inside 10,000-round stationary window across >=30 seeds contrasting 0.6% slow MAP dilution vs <=20-round fast detector",
        "software/src/control/causal_scheduler.cpp", "avid_fast_detector_step",
        "Slow 10k-window MAP shift <= 0.010 while fast 16-round detector triggers within [4, 20] rounds across all seeds", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
