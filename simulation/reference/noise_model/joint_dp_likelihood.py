"""
Exact 2^m-State Dynamic Programming (DP) Joint Syndrome Likelihood & L-BFGS Estimator.
Addresses Policy Item A02 / Checklist Q03 & A08 / Checklist Q17:
  - Separates single-qubit data faults (Z_i), correlated pair faults (Z_i Z_j),
    and measurement bit-flip faults.
  - Computes exact joint distribution P(s) over all 2^m syndrome states via XOR-convolution DP:
      P_new(s) = (1 - p_e) * P_old(s) + p_e * P_old(s ^ g_e)
  - Computes exact closed-form analytic gradients:
      dP(s)/dp_e = (P(s ^ g_e) - P(s)) / (1 - 2*p_e)   for g_e != 0
  - Provides brute-force 2^E subset enumeration for bit-exact verification.
  - Computes effective per-qubit channel LLRs for Normalized Min-Sum BP.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from simulation.reference.code_model.css_code_builder import (
    RotatedSurfaceCode,
    build_rotated_d3_code,
)


@dataclass(frozen=True)
class FaultEvent:
    event_id: int
    name: str
    category: str  # "single_qubit", "zz_pair", "measurement"
    qubits: Tuple[int, ...]
    signature_bits: Tuple[int, ...]
    signature_mask: int
    default_prob: float


class CorrelatedNoiseModel:
    """
    Exact 2^m-state Dynamic Programming correlated syndrome likelihood model.
    Supports both canonical 21-parameter (9 single + 12 ZZ pair) and 25-parameter
    (+ 4 measurement bit-flip) parameterizations for d=3 rotated surface code.
    """

    def __init__(
        self,
        code: RotatedSurfaceCode,
        basis: str = "X",
        include_measurement_faults: bool = False,
        base_single_p: float = 0.012,
        base_pair_p: float = 0.008,
        base_meas_p: float = 0.003,
    ) -> None:
        self.code = code
        self.basis = basis.upper()
        self.H = code.Hx if self.basis == "X" else code.Hz
        self.n_checks, self.n_vars = self.H.shape
        self.n_states = 1 << self.n_checks
        self.include_measurement_faults = include_measurement_faults

        self.events: List[FaultEvent] = []
        eid = 0

        # 1. Single-qubit data faults (9 for d=3)
        for q in range(self.n_vars):
            col = self.H[:, q].astype(int)
            mask = int(sum((int(col[c]) & 1) << c for c in range(self.n_checks)))
            self.events.append(
                FaultEvent(
                    event_id=eid,
                    name=f"Z_{q}" if self.basis == "X" else f"X_{q}",
                    category="single_qubit",
                    qubits=(q,),
                    signature_bits=tuple(col.tolist()),
                    signature_mask=mask,
                    default_prob=base_single_p,
                )
            )
            eid += 1

        # 2. Correlated nearest-neighbor pair faults (12 for d=3)
        for u, v in code.zz_edges:
            col_u = self.H[:, u].astype(int)
            col_v = self.H[:, v].astype(int)
            col_uv = (col_u ^ col_v) & 1
            mask = int(sum((int(col_uv[c]) & 1) << c for c in range(self.n_checks)))
            self.events.append(
                FaultEvent(
                    event_id=eid,
                    name=f"ZZ_{u}_{v}" if self.basis == "X" else f"XX_{u}_{v}",
                    category="zz_pair",
                    qubits=(u, v),
                    signature_bits=tuple(col_uv.tolist()),
                    signature_mask=mask,
                    default_prob=base_pair_p,
                )
            )
            eid += 1

        # 3. Optional measurement bit-flip faults (4 for d=3)
        if include_measurement_faults:
            for c in range(self.n_checks):
                bits = [1 if i == c else 0 for i in range(self.n_checks)]
                mask = 1 << c
                self.events.append(
                    FaultEvent(
                        event_id=eid,
                        name=f"MEAS_c{c}",
                        category="measurement",
                        qubits=(),
                        signature_bits=tuple(bits),
                        signature_mask=mask,
                        default_prob=base_meas_p,
                    )
                )
                eid += 1

        self.n_params = len(self.events)
        self.signatures = np.array([e.signature_mask for e in self.events], dtype=np.int64)
        self.default_theta = np.array([e.default_prob for e in self.events], dtype=np.float64)

        # Precompute XOR permutation table: xor_table[e, s] = s ^ g_e
        states = np.arange(self.n_states, dtype=np.int64)
        self.xor_table = np.bitwise_xor(self.signatures[:, None], states[None, :])

    def joint_distribution_dp(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute exact joint syndrome distribution P(s) for s in 0..2^m-1
        using O(E * 2^m) dynamic programming:
          P_new(s) = (1 - p_e) * P_old(s) + p_e * P_old(s ^ g_e)
        """
        p = np.asarray(theta, dtype=np.float64)
        if p.shape != (self.n_params,):
            raise ValueError(f"Expected theta of shape ({self.n_params},), got {p.shape}")

        dist = np.zeros(self.n_states, dtype=np.float64)
        dist[0] = 1.0

        for e_idx in range(self.n_params):
            pe = float(p[e_idx])
            g = int(self.signatures[e_idx])
            if g != 0 and pe != 0.0:
                dist = (1.0 - pe) * dist + pe * dist[self.xor_table[e_idx]]
        return dist

    def joint_distribution_bruteforce(self, theta: np.ndarray, max_events: int = 16) -> np.ndarray:
        """
        Brute-force 2^E subset enumeration of P(s) for exact mathematical verification.
        Used in unit tests to verify joint_distribution_dp to machine precision (<1e-14).
        """
        p = np.asarray(theta, dtype=np.float64)
        E = len(p)
        if E > max_events:
            raise ValueError(f"Brute-force enumeration capped at {max_events} events, got {E}")
        dist = np.zeros(self.n_states, dtype=np.float64)
        for mask in range(1 << E):
            prob = 1.0
            syn = 0
            for e_idx in range(E):
                if (mask >> e_idx) & 1:
                    prob *= p[e_idx]
                    syn ^= int(self.signatures[e_idx])
                else:
                    prob *= 1.0 - p[e_idx]
            dist[syn] += prob
        return dist

    def distribution_and_jacobian(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute exact joint distribution P(s) (shape [2^m]) and exact Jacobian
        J[s, e] = dP(s)/dp_e (shape [2^m, E]) in a single pass.
        Uses the exact coset inversion identity:
          dP(s)/dp_e = (P(s ^ g_e) - P(s)) / (1 - 2 * p_e)  for g_e != 0,
          dP(s)/dp_e = 0                                    for g_e == 0.
        """
        p = np.asarray(theta, dtype=np.float64)
        dist = self.joint_distribution_dp(p)
        jac = np.zeros((self.n_states, self.n_params), dtype=np.float64)

        for e_idx in range(self.n_params):
            g = int(self.signatures[e_idx])
            if g == 0:
                continue
            denom = 1.0 - 2.0 * float(p[e_idx])
            if abs(denom) < 1e-12:
                denom = 1e-12 if denom >= 0 else -1e-12
            jac[:, e_idx] = (dist[self.xor_table[e_idx]] - dist) / denom
        return dist, jac

    def negative_log_likelihood_and_grad(
        self,
        theta: np.ndarray,
        empirical_probs: np.ndarray,
        prior_theta: Optional[np.ndarray] = None,
        prior_lambda: float = 0.0,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute exact regularized negative log-likelihood (cross-entropy) and gradient:
          L(theta) = -sum_s P_emp(s) * log(P_theta(s)) + 0.5 * lambda * ||theta - theta_prior||^2
        """
        p = np.clip(np.asarray(theta, dtype=np.float64), 1e-6, 0.49)
        emp = np.asarray(empirical_probs, dtype=np.float64)
        emp_sum = float(np.sum(emp))
        if emp_sum > 0:
            emp = emp / emp_sum

        dist, jac = self.distribution_and_jacobian(p)
        safe_dist = np.maximum(dist, 1e-15)

        nll = float(-np.sum(emp * np.log(safe_dist)))
        weights = -(emp / safe_dist)  # shape (2^m,)
        grad = weights @ jac  # shape (E,)

        if prior_theta is not None and prior_lambda > 0.0:
            diff = p - np.asarray(prior_theta, dtype=np.float64)
            nll += 0.5 * prior_lambda * float(np.dot(diff, diff))
            grad = grad + prior_lambda * diff

        return nll, grad

    def compute_effective_qubit_probs(self, theta: np.ndarray) -> np.ndarray:
        """
        Map the 21 (or 25) physical fault parameters to effective per-qubit error
        probabilities p_eff[v] for initializing the 9 variable-node priors in BP:
          p_eff[v] = p_single[v] XOR_{e=(u,v)} p_pair[e]
        where p1 XOR p2 = p1*(1 - p2) + (1 - p1)*p2.
        """
        p = np.clip(np.asarray(theta, dtype=np.float64), 1e-6, 0.49)
        p_eff = p[: self.n_vars].copy()
        for idx, (u, v) in enumerate(self.code.zz_edges):
            pe = float(p[self.n_vars + idx])
            p_eff[u] = p_eff[u] * (1.0 - pe) + (1.0 - p_eff[u]) * pe
            p_eff[v] = p_eff[v] * (1.0 - pe) + (1.0 - p_eff[v]) * pe
        return np.clip(p_eff, 1e-6, 0.49)

    def compute_q7_8_llrs(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute floating-point LLRs and quantized signed 16-bit Q7.8 integers
        matching `llr_numeric_contract.yaml` and `weight_map_reg.sv`.
        """
        p_eff = self.compute_effective_qubit_probs(theta)
        llr_float = np.log((1.0 - p_eff) / p_eff)
        llr_q7_8 = np.clip(np.round(llr_float * 256.0), -32768, 32767).astype(np.int16)
        return llr_float, llr_q7_8

    def fit_lbfgs(
        self,
        empirical_counts: np.ndarray,
        init_theta: Optional[np.ndarray] = None,
        prior_theta: Optional[np.ndarray] = None,
        prior_lambda: float = 0.5,
        max_iter: int = 25,
    ) -> Dict[str, object]:
        """
        Estimate noise parameters theta from a 16-bin syndrome histogram using L-BFGS-B
        with exact analytic gradients from the 2^m DP likelihood.
        """
        x0 = (init_theta if init_theta is not None else self.default_theta).copy()
        prior = (prior_theta if prior_theta is not None else x0).copy()
        bounds = [(1e-5, 0.45) for _ in range(self.n_params)]

        res = minimize(
            fun=lambda x: self.negative_log_likelihood_and_grad(
                x, empirical_counts, prior_theta=prior, prior_lambda=prior_lambda
            ),
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-9},
        )
        est_theta = np.clip(res.x, 1e-5, 0.45)
        llr_float, llr_q7_8 = self.compute_q7_8_llrs(est_theta)
        return {
            "theta": est_theta,
            "nll": float(res.fun),
            "nit": int(res.nit),
            "success": bool(res.success),
            "llr_float": llr_float,
            "llr_q7_8": llr_q7_8,
        }


def build_d3_correlated_noise_model(
    basis: str = "X",
    include_measurement_faults: bool = False,
) -> CorrelatedNoiseModel:
    code = build_rotated_d3_code()
    return CorrelatedNoiseModel(
        code=code,
        basis=basis,
        include_measurement_faults=include_measurement_faults,
    )
