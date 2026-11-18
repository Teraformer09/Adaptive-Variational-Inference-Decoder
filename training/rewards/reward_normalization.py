"""Bounded Normalization Utilities for Layer 2 V2 Reward Components (`reward_normalization.py`)."""

from __future__ import annotations

import numpy as np

COMPONENT_MIN: float = -1.0
COMPONENT_MAX: float = 1.0
TOTAL_REWARD_MIN: float = -2.0
TOTAL_REWARD_MAX: float = 2.0


def clip_component(val: float) -> float:
    """Clip an individual reward component strictly into [-1.0, +1.0]."""
    if not np.isfinite(val):
        return -1.0
    return float(np.clip(val, COMPONENT_MIN, COMPONENT_MAX))


def clip_total_reward(val: float) -> float:
    """Clip total composite step reward strictly into [-2.0, +2.0]."""
    if not np.isfinite(val):
        return TOTAL_REWARD_MIN
    return float(np.clip(val, TOTAL_REWARD_MIN, TOTAL_REWARD_MAX))


def calibrated_ler_improvement_score(
    ler_action: float,
    ler_reference: float,
    floor_ler: float = 0.015,
) -> float:
    """Compute bounded QEC decoder-outcome score in [-1.0, +1.0], safe when `ler_reference` ~ 0."""
    la = float(np.clip(ler_action, 0.0, 1.0))
    lr = float(np.clip(ler_reference, 0.0, 1.0))
    denom = max(lr, float(floor_ler))
    rel_gain = (lr - la) / denom

    # When both reference and action achieve very low LER in quiet regimes, add small positive anchor
    quiet_anchor = 0.15 * max(0.0, 1.0 - la / max(float(floor_ler) * 1.5, 1e-6)) if la <= lr + 1e-6 else 0.0
    return clip_component(1.25 * rel_gain + quiet_anchor)
