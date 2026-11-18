"""Noise-Aware, Bounded, Decomposed QEC Reward Package."""

from training.rewards.noise_aware_reward import (
    CANONICAL_REGIMES,
    LEGACY_FAMILY_TO_REGIME,
    compute_noise_aware_reward,
    resolve_regime_responsibilities,
)
from training.rewards.reward_diagnostics import REWARD_TERM_KEYS, RewardDiagnosticsTracker
from training.rewards.reward_normalization import (
    COMPONENT_MAX,
    COMPONENT_MIN,
    TOTAL_REWARD_MAX,
    TOTAL_REWARD_MIN,
    calibrated_ler_improvement_score,
    clip_component,
    clip_total_reward,
)

__all__ = [
    "CANONICAL_REGIMES",
    "COMPONENT_MAX",
    "COMPONENT_MIN",
    "LEGACY_FAMILY_TO_REGIME",
    "REWARD_TERM_KEYS",
    "RewardDiagnosticsTracker",
    "TOTAL_REWARD_MAX",
    "TOTAL_REWARD_MIN",
    "calibrated_ler_improvement_score",
    "clip_component",
    "clip_total_reward",
    "compute_noise_aware_reward",
    "resolve_regime_responsibilities",
]
