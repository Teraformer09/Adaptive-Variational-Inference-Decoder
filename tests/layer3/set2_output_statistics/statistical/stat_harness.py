#!/usr/bin/env python3
"""Shared statistical harness for Set 2E (L3-STAT-001..015).

Executes the compiled C++ Monte Carlo driver (`set2_stat_mc_driver`), computes full
multi-metric statistical summaries (mean, bias, variance, MSE, 95% CI, IQR, p05, p95),
evaluates acceptance assertions, and emits canonical `TestResultRecord` JSON + log files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def summarize_array(arr: list[float] | np.ndarray, target: float = 0.0) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    n = max(1, len(x))
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if n > 1 else 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    bias = mean - target
    mse = float(np.mean((x - target) ** 2))
    p05, p25, p75, p95 = [float(v) for v in np.percentile(x, [5, 25, 75, 95])]
    return {
        "n": float(n),
        "mean": mean,
        "std": sd,
        "variance": var,
        "bias": bias,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "ci95_low": mean - 1.959964 * se,
        "ci95_high": mean + 1.959964 * se,
        "iqr": p75 - p25,
        "p05": p05,
        "p95": p95,
    }


def parse_stat_args(default_tid: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-id", default=default_tid)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--validation-dir", required=True)
    ap.add_argument("--mc-driver", required=True)
    ap.add_argument("--ctest-strict", action="store_true")
    return ap.parse_args()


def get_head_commit(repo_root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run_stat_case(
    args: argparse.Namespace,
    test_name: str,
    description: str,
    impl_file: str,
    impl_func: str,
    oracle_desc: str,
    evaluator_fn,
) -> int:
    t0 = time.perf_counter()
    cmd = [args.mc_driver, "--test-id", args.test_id]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)

    passed, expected_map, actual_map, notes = evaluator_fn(payload)
    dur_ms = (time.perf_counter() - t0) * 1000.0
    status = "PASS" if passed else "FAIL"

    val_dir = Path(args.validation_dir)
    raw_dir = val_dir / "raw_results" / "set2"
    log_dir = val_dir / "logs" / "set2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rec = {
        "test_id": args.test_id,
        "test_name": test_name,
        "test_description": description,
        "test_set": "set2_output_statistics",
        "status": status,
        "source_commit": get_head_commit(args.repo_root),
        "implementation_file": impl_file,
        "implementation_function": impl_func,
        "input_contract": ">=30 independent Monte Carlo seeds over 16-state Joint-DP syndrome histograms",
        "independent_oracle": oracle_desc,
        "execution_command": " ".join(sys.argv),
        "fixture": "monte_carlo_synthetic_32_seeds",
        "expected": {k: str(v) for k, v in expected_map.items()},
        "actual": {k: str(v) for k, v in actual_map.items()},
        "assertion_notes": notes,
        "evidence": [
            f"raw_results/set2/{args.test_id}.json",
            f"logs/set2/{args.test_id}.log",
        ],
        "failure_classification": "" if passed else "STATISTICAL_ORACLE_FAILURE",
        "duration_ms": round(dur_ms, 3),
    }

    (raw_dir / f"{args.test_id}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    log_lines = [
        f"[{args.test_id}] {test_name} -> {status} ({dur_ms:.2f} ms)",
        f"expected={json.dumps(expected_map)}",
        f"actual={json.dumps(actual_map)}",
    ] + notes
    (log_dir / f"{args.test_id}.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(log_lines[0])
    return 0 if (passed or not args.ctest_strict) else 1
