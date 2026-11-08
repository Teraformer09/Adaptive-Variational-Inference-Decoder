"""Dueling Q-Network Head and Bounded Per-Qubit Gain Residual Head."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


class DuelingQValueHead:
    """Dueling action-value head + bounded per-qubit gain residual head.

    Action space (6 discrete actions):
      - a0: STATIC_PRIOR_K4        (pair_weight_scale = 0.00, K = 4)
      - a1: ADAPTIVE_PAIR_K6       (pair_weight_scale = 1.00, K = 6)
      - a2: AGGRESSIVE_PAIR_K8     (pair_weight_scale = 1.35, K = 8)
      - a3: DAMPED_PRIOR_K4        (pair_weight_scale = 0.85, K = 4)
      - a4: FAST_RECALIBRATE_K8    (pair_weight_scale = 1.15, K = 8, recalibrate = True)
      - a5: CONSERVATIVE_FALLBACK  (pair_weight_scale = 0.00, K = 4, fallback = True)

    Dueling decomposition:
      Q(h, a) = V(h) + A(h, a) - (1 / |A|) * sum_{a'} A(h, a')

    Per-qubit LLR gain residual:
      delta_g(h) = gain_clip * tanh(W_gain @ h + b_gain) in [-0.30, +0.30]^9
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_actions: int = 6,
        num_qubits: int = 9,
        gain_clip: float = 0.30,
        seed: int = 20260406,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.num_qubits = int(num_qubits)
        self.gain_clip = float(gain_clip)

        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / self.hidden_dim)

        self.W_val = rng.normal(0.0, scale, size=(1, self.hidden_dim)).astype(np.float64)
        self.b_val = np.zeros(1, dtype=np.float64)

        self.W_adv = rng.normal(0.0, scale, size=(self.num_actions, self.hidden_dim)).astype(
            np.float64
        )
        self.b_adv = np.zeros(self.num_actions, dtype=np.float64)

        self.W_gain = rng.normal(0.0, scale, size=(self.num_qubits, self.hidden_dim)).astype(
            np.float64
        )
        self.b_gain = np.zeros(self.num_qubits, dtype=np.float64)

    def forward_step(
        self, h: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Compute (q_values[6], state_value, advantages[6], qubit_gains[9]) for one hidden state."""
        val = float((self.W_val @ h + self.b_val)[0])
        adv = self.W_adv @ h + self.b_adv
        q_values = val + (adv - np.mean(adv))

        gain_pre = np.clip(self.W_gain @ h + self.b_gain, -15.0, 15.0)
        qubit_gains = self.gain_clip * np.tanh(gain_pre)
        return q_values, val, adv, qubit_gains

    def backward_step(
        self,
        h: np.ndarray,
        dq_values: np.ndarray,
        dqubit_gains: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Backpropagate gradients from dL/dQ and dL/d(delta_g) to hidden state h and weights."""
        # Dueling Q(h, a) = V(h) + A(h, a) - mean(A(h, .))
        # dL/dV = sum_a dL/dQ_a
        dval = np.array([np.sum(dq_values)], dtype=np.float64)
        # dL/dA_a = dL/dQ_a - (1 / |A|) * sum_{a'} dL/dQ_{a'}
        dadv = dq_values - np.mean(dq_values)

        dW_val = np.outer(dval, h)
        db_val = dval.copy()

        dW_adv = np.outer(dadv, h)
        db_adv = dadv.copy()

        gain_pre = np.clip(self.W_gain @ h + self.b_gain, -15.0, 15.0)
        tanh_g = np.tanh(gain_pre)
        dgain_pre = dqubit_gains * self.gain_clip * (1.0 - tanh_g ** 2)

        dW_gain = np.outer(dgain_pre, h)
        db_gain = dgain_pre.copy()

        dh = (
            (self.W_val.T @ dval)
            + (self.W_adv.T @ dadv)
            + (self.W_gain.T @ dgain_pre)
        )
        grads = {
            "W_val": dW_val,
            "b_val": db_val,
            "W_adv": dW_adv,
            "b_adv": db_adv,
            "W_gain": dW_gain,
            "b_gain": db_gain,
        }
        return dh, grads
