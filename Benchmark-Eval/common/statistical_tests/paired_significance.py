#!/usr/bin/env python3
'''Hypothesis Testing & Multiple-Comparison Correction for AVID V1 Evaluation.'''
from __future__ import annotations
import math
from typing import Dict, List
import numpy as np
from scipy import stats


def two_proportion_z_test(k1: int, n1: int, k2: int, n2: int) -> Dict[str, float]:
    '''Two-proportion z-test comparing static failures (k1/n1) vs adaptive failures (k2/n2).'''
    p1 = k1 / float(max(1, n1))
    p2 = k2 / float(max(1, n2))
    p_pool = (k1 + k2) / float(max(1, n1 + n2))
    se = math.sqrt(max(1e-18, p_pool * (1.0 - p_pool) * (1.0 / max(1, n1) + 1.0 / max(1, n2))))
    z_stat = (p1 - p2) / se
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    return {"z_statistic": round(z_stat, 4), "two_sided_p_value": p_val}


def paired_multiseed_tests(static_lers: List[float], adaptive_lers: List[float]) -> Dict[str, float]:
    '''Paired t-test, Wilcoxon signed-rank test, and Cohen's d across N independent seeds.'''
    a = np.asarray(static_lers, dtype=np.float64)
    b = np.asarray(adaptive_lers, dtype=np.float64)
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1e-12
    cohens_d = mean_diff / max(std_diff, 1e-12)
    t_res = stats.ttest_rel(a, b)
    try:
        w_res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        w_pval = float(w_res.pvalue)
    except ValueError:
        w_pval = 1.0
    return {
        "n_seeds": len(a),
        "static_mean_ler": round(float(np.mean(a)), 5),
        "static_std_ler": round(float(np.std(a, ddof=1)), 5),
        "adaptive_mean_ler": round(float(np.mean(b)), 5),
        "adaptive_std_ler": round(float(np.std(b, ddof=1)), 5),
        "mean_ler_reduction_pct": round(float(100.0 * (np.mean(a) - np.mean(b)) / max(np.mean(a), 1e-12)), 2),
        "paired_t_statistic": round(float(t_res.statistic), 4),
        "paired_t_pvalue": float(t_res.pvalue),
        "wilcoxon_two_sided_pvalue": w_pval,
        "cohens_d": round(float(cohens_d), 4),
    }


def holm_bonferroni_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    '''Holm-Bonferroni step-down multiple-testing correction.'''
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for idx, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - idx))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted
