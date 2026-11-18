"""SVD Rank & Gauge Nullspace Analysis of the 21-Parameter Forward Model (`svd_analysis.py`)."""

from __future__ import annotations

from typing import Dict
import numpy as np

from layer3.identifiability.sensitivity_jacobian import compute_syndrome_jacobian


def analyze_jacobian_svd(
    theta_21: np.ndarray,
    basis_checks: str = "X",
    readout_flip_prob: float = 0.0,
    rank_tol: float = 1.0e-9,
) -> Dict[str, object]:
    """Perform SVD on J(theta) = U Sigma V^T and partition R^21 into observable and null subspaces."""
    J = compute_syndrome_jacobian(
        theta_21=theta_21, basis_checks=basis_checks, readout_flip_prob=readout_flip_prob
    )
    U, s_vals, Vt = np.linalg.svd(J, full_matrices=True)
    rank = int(np.sum(s_vals > rank_tol))
    nullity = int(21 - rank)

    V_obs = Vt[:rank, :].T       # (21, rank)
    V_null = Vt[rank:, :].T      # (21, 21 - rank)

    # Parameter observability score: squared norm of projection onto observable rowspace V_obs
    obs_leverage = np.sum(V_obs ** 2, axis=1)  # in [0, 1]^21

    return {
        "jacobian": J,
        "singular_values": s_vals,
        "rank": rank,
        "nullity": nullity,
        "V_observable": V_obs,
        "V_nullspace": V_null,
        "parameter_observability_leverage": obs_leverage,
    }
