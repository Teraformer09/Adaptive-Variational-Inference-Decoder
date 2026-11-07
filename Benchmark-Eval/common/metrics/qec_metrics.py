#!/usr/bin/env python3
'''Canonical QEC & Statistical Metrics for AVID V1 Evaluation.'''
from __future__ import annotations
import math
from typing import Tuple


def wilson_ci95(failures: int, total_rounds: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total_rounds <= 0:
        return (0.0, 0.0)
    p = failures / float(total_rounds)
    n = float(total_rounds)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))
    return (round(max(0.0, center - half), 5), round(min(1.0, center + half), 5))


def ler_reduction_pct(static_ler: float, adaptive_ler: float) -> float:
    if static_ler <= 1e-12:
        return 0.0
    return round(100.0 * (static_ler - adaptive_ler) / static_ler, 2)


def verify_epoch_commit_reconciliation(
    initial_boot_commits: int,
    adaptive_epoch_commits: int,
    reset_cleared_commits: int,
    weight_bank_commits: int,
) -> bool:
    return (initial_boot_commits + adaptive_epoch_commits - reset_cleared_commits) == weight_bank_commits
