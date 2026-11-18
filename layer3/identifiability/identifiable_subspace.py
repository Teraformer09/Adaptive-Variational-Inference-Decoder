"""Identifiable Syndrome-Orbit Subspace Projections (`identifiable_subspace.py`)."""

from __future__ import annotations

from typing import Dict
import numpy as np

from layer3.identifiability.svd_analysis import analyze_jacobian_svd


def project_identifiable_and_null_coordinates(
    theta_21: np.ndarray,
    reference_theta_21: np.ndarray | None = None,
    basis_checks: str = "X",
) -> Dict[str, np.ndarray]:
    """Decompose `theta_21` into identifiable orbit coordinates psi_obs and null coordinates psi_null."""
    ref = (
        np.asarray(reference_theta_21, dtype=np.float64)
        if reference_theta_21 is not None
        else np.full(21, 0.012, dtype=np.float64)
    )
    svd_info = analyze_jacobian_svd(ref, basis_checks=basis_checks)
    V_obs = np.asarray(svd_info["V_observable"], dtype=np.float64)
    V_null = np.asarray(svd_info["V_nullspace"], dtype=np.float64)

    th = np.asarray(theta_21, dtype=np.float64)
    psi_obs = V_obs.T @ th
    psi_null = V_null.T @ th
    return {
        "psi_observable": psi_obs,
        "psi_nullspace": psi_null,
        "reconstructed_observable_part": V_obs @ psi_obs,
        "observability_leverage": np.asarray(
            svd_info["parameter_observability_leverage"], dtype=np.float64
        ),
    }


def sample_nullspace_perturbation(
    theta_21: np.ndarray,
    rng: np.random.Generator,
    scale: float = 0.015,
    basis_checks: str = "X",
) -> np.ndarray:
    """Construct a perturbed 21-parameter vector `theta_alt` differing along ker(J(theta_21))."""
    th = np.asarray(theta_21, dtype=np.float64)
    svd_info = analyze_jacobian_svd(th, basis_checks=basis_checks)
    V_null = np.asarray(svd_info["V_nullspace"], dtype=np.float64)
    raw_dir = V_null @ rng.standard_normal(V_null.shape[1])
    norm = float(np.linalg.norm(raw_dir))
    if norm > 1e-12:
        raw_dir = raw_dir / norm
    theta_alt = np.clip(th + float(scale) * raw_dir, 1.0e-4, 0.45)
    return theta_alt
