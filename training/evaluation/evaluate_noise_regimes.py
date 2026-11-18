"""Per-Regime & Unseen Held-Out Mixture Evaluation (`evaluate_noise_regimes.py`)."""

from __future__ import annotations

from typing import Dict
from training.data.trajectory_split import TrajectorySplitManager
from training.evaluation.evaluate_policy import evaluate_policy_on_split
from training.models.r2d2_network import R2D2Network


def evaluate_across_regimes_and_heldout_mixtures(
    net: R2D2Network,
    test_seed_start: int = 2026,
    test_seed_end: int = 2033,
    heldout_seed_start: int = 3000,
    heldout_seed_end: int = 3007,
) -> Dict[str, object]:
    """Evaluate trained V2 policy on disjoint 5-regime test split and 3 unseen held-out mixtures (EVAL-01)."""
    core_regimes = [
        "gaussian_stationary",
        "ou_drift",
        "zz_crosstalk",
        "tls_telegraph",
        "nonstationary_burst",
    ]
    heldout_mixtures = [
        "heldout_multi_pair_12_67",
        "heldout_spatial_3qubit_streak",
        "heldout_phenomenological_readout",
    ]

    test_res = evaluate_policy_on_split(
        net=net,
        regimes=core_regimes,
        seed_start=test_seed_start,
        seed_end=test_seed_end,
        split_name="test",
        num_epochs=16,
    )
    heldout_res = evaluate_policy_on_split(
        net=net,
        regimes=heldout_mixtures,
        seed_start=heldout_seed_start,
        seed_end=heldout_seed_end,
        split_name="heldout_mixtures",
        num_epochs=16,
    )

    TrajectorySplitManager.assert_no_trajectory_leakage(
        set(test_res["trajectory_ids"]), set(heldout_res["trajectory_ids"])
    )

    return {
        "core_5_regimes_test": {k: v for k, v in test_res.items() if k != "trajectory_ids"},
        "heldout_3_mixtures": {k: v for k, v in heldout_res.items() if k != "trajectory_ids"},
        "disjoint_trajectories_verified": True,
    }
