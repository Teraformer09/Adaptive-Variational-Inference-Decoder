"""Reward Decomposition & Worst-Regime Diagnostics Tracker (`reward_diagnostics.py`)."""

from __future__ import annotations

from typing import Dict, List
import numpy as np

REWARD_TERM_KEYS = (
    "reward/qec_performance",
    "reward/gaussian_stability",
    "reward/ou_tracking",
    "reward/zz_correlation",
    "reward/tls_switching",
    "reward/burst_recovery",
    "reward/decoder_stability",
    "reward/safety",
    "reward/total_reward",
)


class RewardDiagnosticsTracker:
    """Accumulates decomposed reward components and per-regime / worst-regime summaries."""

    def __init__(self) -> None:
        self._records: List[Dict[str, float]] = []
        self._by_regime: Dict[str, List[float]] = {}

    def reset(self) -> None:
        self._records.clear()
        self._by_regime.clear()

    def log_step(self, reward_dict: Dict[str, float], regime: str = "gaussian_stationary") -> None:
        for k in REWARD_TERM_KEYS:
            if k not in reward_dict:
                raise KeyError(f"Missing required reward decomposition key '{k}'")
        self._records.append({k: float(reward_dict[k]) for k in REWARD_TERM_KEYS})
        self._by_regime.setdefault(regime, []).append(float(reward_dict["reward/total_reward"]))

    def summary(self) -> Dict[str, object]:
        if not self._records:
            return {
                "num_steps": 0,
                "mean_terms": {k: 0.0 for k in REWARD_TERM_KEYS},
                "per_regime_total_reward": {},
                "worst_regime": None,
                "worst_regime_reward": 0.0,
            }

        mean_terms = {
            k: float(np.mean([r[k] for r in self._records])) for k in REWARD_TERM_KEYS
        }
        per_regime = {
            reg: float(np.mean(vals)) for reg, vals in self._by_regime.items() if vals
        }
        worst_reg = min(per_regime, key=per_regime.get) if per_regime else None
        worst_val = float(per_regime[worst_reg]) if worst_reg is not None else 0.0

        return {
            "num_steps": len(self._records),
            "mean_terms": mean_terms,
            "per_regime_total_reward": per_regime,
            "worst_regime": worst_reg,
            "worst_regime_reward": worst_val,
        }
