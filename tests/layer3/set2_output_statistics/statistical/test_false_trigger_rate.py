#!/usr/bin/env python3
"""Set 2E Stationary False-Trigger Rate Over 50,000 Rounds x >=30 Seeds: L3-STAT-014."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_harness import parse_stat_args, run_stat_case, summarize_array


def main() -> int:
    args = parse_stat_args("L3-STAT-014")

    def ev(p):
        rt = summarize_array(p["trigger_rates"], 0.0)
        ct = summarize_array(p["trigger_counts"], 0.0)
        ok = rt["mean"] <= 1e-4 and rt["p95"] <= 2e-4
        return (
            ok,
            {"mean_false_alarm_rate_max": 1e-4, "p95_false_alarm_rate_max": 2e-4},
            {
                "p_nonzero": round(p["p_nonzero"], 6),
                "mean_triggers_per_50k": round(ct["mean"], 4),
                "mean_false_alarm_rate": round(rt["mean"], 8),
                "p95_false_alarm_rate": round(rt["p95"], 8),
            },
            [f"32x50,000=1.6M stationary rounds: mean false triggers={ct['mean']:.3f}, rate={rt['mean']:.2e}"],
        )

    return run_stat_case(
        args, "stationary_false_trigger_rate_50k_rounds",
        "Stream 50,000 stationary low-noise rounds across >=30 seeds through avid_fast_burst_detector_t and measure false-alarm rate",
        "software/src/control/causal_scheduler.cpp", "avid_fast_detector_step",
        "Empirical false-alarm rate <= 1e-4 per round across 1.6M total stationary rounds", ev
    )


if __name__ == "__main__":
    raise SystemExit(main())
