"""Paired Statistical Analysis & Confidence Interval Utilities (`stats_utils.py`).

Implements the publication-gate statistical requirement:
  - Paired trajectory LER differences Delta_i = LER_baseline(seed_i) - LER_method(seed_i)
  - Parametric 95% Student-t confidence intervals
  - Non-parametric BCa / percentile paired bootstrap 95% confidence intervals (B = 5,000 resamples)
  - Paired two-sided t-test and Wilcoxon signed-rank p-values
  - Holm-Bonferroni step-down family-wise error control
"""

from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
from scipy import stats


def compute_paired_ler_ci_and_tests(
    ler_baseline: Sequence[float],
    ler_candidate: Sequence[float],
    num_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 20261119,
) -> Dict[str, float | bool]:
    """Compute paired LER reduction statistics, 95% CIs (bootstrap + Student-t), and p-values.

    Positive `mean_delta_ler` means `ler_candidate < ler_baseline` (improvement).
    """
    b = np.asarray(ler_baseline, dtype=np.float64)
    c = np.asarray(ler_candidate, dtype=np.float64)
    if b.shape != c.shape or b.ndim != 1 or b.size < 2:
        raise ValueError(f"Expected 1D paired arrays of size >= 2, got {b.shape} vs {c.shape}")

    n = int(b.size)
    deltas = b - c  # positive => candidate reduces LER
    mean_b = float(np.mean(b))
    mean_c = float(np.mean(c))
    mean_d = float(np.mean(deltas))
    sd_d = float(np.std(deltas, ddof=1))
    se_d = sd_d / np.sqrt(n)

    alpha = 1.0 - float(ci_level)
    t_crit = float(stats.t.ppf(1.0 - 0.5 * alpha, df=n - 1))
    t_ci_low = mean_d - t_crit * se_d
    t_ci_high = mean_d + t_crit * se_d

    rng = np.random.default_rng(int(seed))
    boot_idx = rng.integers(0, n, size=(int(num_bootstrap), n))
    boot_means = np.mean(deltas[boot_idx], axis=1)
    boot_ci_low = float(np.percentile(boot_means, 100.0 * (0.5 * alpha)))
    boot_ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - 0.5 * alpha)))

    # Relative reduction (%) and bootstrap CI on relative reduction (%)
    boot_b = np.mean(b[boot_idx], axis=1)
    boot_c = np.mean(c[boot_idx], axis=1)
    boot_rel_pct = 100.0 * (boot_b - boot_c) / np.maximum(boot_b, 1e-12)
    rel_pct = 100.0 * (mean_b - mean_c) / max(mean_b, 1e-12)
    rel_ci_low = float(np.percentile(boot_rel_pct, 100.0 * (0.5 * alpha)))
    rel_ci_high = float(np.percentile(boot_rel_pct, 100.0 * (1.0 - 0.5 * alpha)))

    if sd_d < 1e-15:
        t_stat = 0.0
        p_ttest = 1.0 if abs(mean_d) < 1e-15 else 0.0
    else:
        t_res = stats.ttest_rel(b, c)
        t_stat = float(t_res.statistic)
        p_ttest = float(t_res.pvalue)

    nonzero = deltas[np.abs(deltas) > 1e-15]
    if nonzero.size < 3:
        p_wilcoxon = p_ttest
    else:
        try:
            w_res = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
            p_wilcoxon = float(w_res.pvalue)
        except ValueError:
            p_wilcoxon = p_ttest

    cohen_dz = float(mean_d / max(sd_d, 1e-12)) if sd_d > 1e-15 else 0.0
    statistically_significant = bool((boot_ci_low > 0.0) and (p_ttest < 0.05))

    return {
        "n_paired_seeds": n,
        "mean_ler_baseline": mean_b,
        "mean_ler_candidate": mean_c,
        "mean_delta_ler": mean_d,
        "std_delta_ler": sd_d,
        "se_delta_ler": float(se_d),
        "t_ci95_low": float(t_ci_low),
        "t_ci95_high": float(t_ci_high),
        "boot_ci95_low": boot_ci_low,
        "boot_ci95_high": boot_ci_high,
        "relative_reduction_pct": float(rel_pct),
        "relative_reduction_ci95_low_pct": rel_ci_low,
        "relative_reduction_ci95_high_pct": rel_ci_high,
        "paired_t_stat": t_stat,
        "p_value_paired_t": p_ttest,
        "p_value_wilcoxon": p_wilcoxon,
        "cohen_dz": cohen_dz,
        "significant_improvement_95pct": statistically_significant,
    }


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Compute Holm-Bonferroni step-down adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda kv: kv[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        scaled = min(1.0, p_val * float(m - rank))
        running_max = max(running_max, scaled)
        adj[orig_idx] = running_max
    return adj


def mean_and_ci95(values: Sequence[float]) -> Dict[str, float]:
    """Return sample mean, SD, and 95% t-interval `[ci_low, ci_high]`."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    m = float(np.mean(arr))
    if n < 2:
        return {"mean": m, "std": 0.0, "ci95_low": m, "ci95_high": m, "half_width": 0.0}
    sd = float(np.std(arr, ddof=1))
    hw = float(stats.t.ppf(0.975, df=n - 1)) * sd / np.sqrt(n)
    return {
        "mean": m,
        "std": sd,
        "ci95_low": m - hw,
        "ci95_high": m + hw,
        "half_width": hw,
    }
