#!/usr/bin/env python3
"""
Per-Stage Timing Profiler & Budget Validator (Policy Item A04 / Checklist Q01).
Measures execution latency of:
  1. 64-bit Syndrome Packet Codec (pack + unpack)
  2. Exact 16-State Dynamic Programming Forward Likelihood Pass
  3. Exact Closed-Form Coset-Inversion Analytic Gradient Pass
  4. Bounded L-BFGS MAP Re-Estimation Step (10 iterations)
  5. Cycle-Accurate Q7.8 Min-Sum BP Hardware Cycle Accounting (17 cycles = 68.0 ns @ 250 MHz)
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.reference.noise_model.joint_dp_likelihood import build_d3_correlated_noise_model
from simulation.reference.syndrome_stream.causal_stream_runner import CycleAccurateBPEngine
from simulation.stimulus.packet_codec import pack_packet_u64, unpack_packet_u64


def main() -> int:
    model = build_d3_correlated_noise_model(basis="X", include_measurement_faults=False)
    bp = CycleAccurateBPEngine(code=model.code, basis="X", default_K=3, clock_mhz=250.0)
    theta = model.default_theta.copy()
    counts = np.full(16, 64.0, dtype=np.float64)

    # 1. Packet codec timing
    n_trials = 500
    t0 = time.perf_counter()
    for i in range(n_trials):
        w = pack_packet_u64(i & 0xF, (i >> 4) & 0xF, i, 1)
        _ = unpack_packet_u64(w)
    pkt_us = ((time.perf_counter() - t0) / n_trials) * 1e6

    # 2. DP forward pass timing
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = model.joint_distribution_dp(theta)
    dp_us = ((time.perf_counter() - t0) / n_trials) * 1e6

    # 3. Analytic gradient pass timing
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = model.negative_log_likelihood_and_grad(theta, counts, prior_theta=theta, prior_lambda=0.5)
    grad_us = ((time.perf_counter() - t0) / n_trials) * 1e6

    # 4. L-BFGS 10-iteration update timing
    t0 = time.perf_counter()
    for _ in range(20):
        _ = model.fit_lbfgs(counts, prior_lambda=0.5, max_iter=10)
    lbfgs_us = ((time.perf_counter() - t0) / 20) * 1e6

    report = {
        "contract_id": "AVID-SYSTEM-TIMING-V1",
        "fpga_fast_path": {
            "clock_mhz": 250.0,
            "K_default": 3,
            "cycles_by_K_fused": {str(k): bp.cycle_count_fused(k) for k in range(1, 6)},
            "cycles_by_K_legacy": {str(k): bp.cycle_count_legacy(k) for k in range(1, 6)},
            "latency_ns_K3_fused": bp.cycle_count_fused(3) * 4.0,
            "latency_ns_K3_legacy": bp.cycle_count_legacy(3) * 4.0,
            "round_budget_ns": 1000.0,
            "meets_round_budget": (bp.cycle_count_fused(3) * 4.0) < 1000.0,
        },
        "ps_slow_path_measured_us": {
            "packet_codec_roundtrip_us": round(pkt_us, 3),
            "joint_dp_forward_pass_us": round(dp_us, 3),
            "analytic_gradient_pass_us": round(grad_us, 3),
            "lbfgs_10_iter_fit_us": round(lbfgs_us, 3),
        },
    }

    out_path = REPO_ROOT / "architecture" / "05_realtime_control" / "timing_profile_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[OK] Saved timing profile report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
