"""Observation Window & Readout-Noise Robustness Evaluation (`evaluate_observation_robustness.py`)."""

from __future__ import annotations

from typing import Dict
from training.evaluation.evaluate_policy import evaluate_policy_on_split
from training.models.r2d2_network import R2D2Network


def evaluate_window_and_noise_robustness(
    net: R2D2Network,
    seed_start: int = 2040,
    seed_end: int = 2045,
) -> Dict[str, object]:
    """Evaluate how V2 R2D2 degrades gracefully across short vs long syndrome windows W."""
    regimes = ["zz_crosstalk", "ou_drift", "nonstationary_burst"]
    by_window: Dict[str, Dict[str, float]] = {}

    for w in (64, 128, 256, 512):
        res = evaluate_policy_on_split(
            net=net,
            regimes=regimes,
            seed_start=seed_start,
            seed_end=seed_end,
            split_name="test",
            num_epochs=12,
            window_rounds=w,
        )
        by_window[f"W_{w}"] = {
            "window_rounds": w,
            "mean_ler": float(res["mean_ler"]),
            "static_ler": float(res["static_ler"]),
            "ler_reduction_pct": float(res["ler_reduction_pct"]),
            "posterior_21_rmse": float(res["posterior_21_rmse"]),
        }

    return {"window_sweep": by_window}
