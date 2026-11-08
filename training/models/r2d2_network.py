"""Full R2D2 Network with Kapturowski et al. invertible value rescaling and Adam optimizer."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple
import numpy as np

from training.models.recurrent_encoder import LSTMStepCache, RecurrentLSTMEncoder
from training.models.q_value_head import DuelingQValueHead


def value_rescale_h(z: np.ndarray | float, eps: float = 1e-3) -> np.ndarray | float:
    """Kapturowski et al. (ICLR 2019) invertible value rescaling function h(z):
    h(z) = sign(z) * (sqrt(|z| + 1) - 1) + eps * z
    """
    z_arr = np.asarray(z, dtype=np.float64)
    res = np.sign(z_arr) * (np.sqrt(np.abs(z_arr) + 1.0) - 1.0) + eps * z_arr
    return float(res) if np.ndim(z) == 0 else res


def value_rescale_h_inv(y: np.ndarray | float, eps: float = 1e-3) -> np.ndarray | float:
    """Exact analytical inverse h^{-1}(y) of Kapturowski et al. value rescaling:
    h^{-1}(y) = sign(y) * (((sqrt(1 + 4 * eps * (|y| + 1 + eps)) - 1) / (2 * eps))^2 - 1)
    """
    y_arr = np.asarray(y, dtype=np.float64)
    abs_y = np.abs(y_arr)
    inner = (np.sqrt(1.0 + 4.0 * eps * (abs_y + 1.0 + eps)) - 1.0) / (2.0 * eps)
    res = np.sign(y_arr) * (inner * inner - 1.0)
    return float(res) if np.ndim(y) == 0 else res


def value_rescale_h_prime(z: np.ndarray | float, eps: float = 1e-3) -> np.ndarray | float:
    """Derivative h'(z) = 1 / (2 * sqrt(|z| + 1)) + eps."""
    z_arr = np.asarray(z, dtype=np.float64)
    res = 0.5 / np.sqrt(np.abs(z_arr) + 1.0) + eps
    return float(res) if np.ndim(z) == 0 else res


ACTION_DEFINITIONS: List[Dict[str, object]] = [
    {"action_id": 0, "name": "STATIC_PRIOR_K3", "pair_weight_scale": 0.00, "bp_iterations": 3, "recalibrate": False, "fallback": False},
    {"action_id": 1, "name": "ADAPTIVE_PAIR_K5", "pair_weight_scale": 1.00, "bp_iterations": 5, "recalibrate": False, "fallback": False},
    {"action_id": 2, "name": "AGGRESSIVE_PAIR_K7", "pair_weight_scale": 1.35, "bp_iterations": 7, "recalibrate": False, "fallback": False},
    {"action_id": 3, "name": "DAMPED_PRIOR_K4", "pair_weight_scale": 0.85, "bp_iterations": 4, "recalibrate": False, "fallback": False},
    {"action_id": 4, "name": "FAST_RECALIBRATE_K7", "pair_weight_scale": 1.15, "bp_iterations": 7, "recalibrate": True, "fallback": False},
    {"action_id": 5, "name": "CONSERVATIVE_FALLBACK_K3", "pair_weight_scale": 0.00, "bp_iterations": 3, "recalibrate": False, "fallback": True},
]


class R2D2Network:
    """Recurrent Replay Distributed DQN (R2D2) model for adaptive QEC decoder control."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_actions: int = 6,
        num_qubits: int = 9,
        gain_clip: float = 0.30,
        seed: int = 20260405,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.num_qubits = int(num_qubits)
        self.gain_clip = float(gain_clip)

        self.encoder = RecurrentLSTMEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            seed=seed,
        )
        self.head = DuelingQValueHead(
            hidden_dim=self.hidden_dim,
            num_actions=self.num_actions,
            num_qubits=self.num_qubits,
            gain_clip=self.gain_clip,
            seed=seed + 1,
        )

    def zero_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.encoder.zero_state()

    def step(
        self,
        x_raw: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Dict[str, object]:
        """Single online recurrent inference step."""
        h_next, c_next, cache = self.encoder.step(x_raw, h_prev, c_prev)
        q_values, state_val, advantages, qubit_gains = self.head.forward_step(h_next)
        best_action = int(np.argmax(q_values))
        action_spec = ACTION_DEFINITIONS[best_action]
        return {
            "q_values": q_values,
            "state_value": state_val,
            "advantages": advantages,
            "qubit_gains": qubit_gains,
            "action_id": best_action,
            "action_name": action_spec["name"],
            "pair_weight_scale": float(action_spec["pair_weight_scale"]),
            "bp_iterations": int(action_spec["bp_iterations"]),
            "recalibrate": bool(action_spec["recalibrate"]),
            "fallback": bool(action_spec["fallback"]),
            "h_next": h_next,
            "c_next": c_next,
            "cache": cache,
        }

    def forward_sequence(
        self,
        obs_seq: np.ndarray,
        h_init: Optional[np.ndarray] = None,
        c_init: Optional[np.ndarray] = None,
        burn_in_steps: int = 0,
    ) -> Dict[str, object]:
        """Unroll R2D2 network across sequence (T, D) with burn-in."""
        T = obs_seq.shape[0]
        h_seq, (h_final, c_final), caches = self.encoder.forward_sequence(
            obs_seq, h_init=h_init, c_init=c_init, burn_in_steps=burn_in_steps
        )
        q_seq = np.zeros((T, self.num_actions), dtype=np.float64)
        v_seq = np.zeros(T, dtype=np.float64)
        adv_seq = np.zeros((T, self.num_actions), dtype=np.float64)
        gain_seq = np.zeros((T, self.num_qubits), dtype=np.float64)

        for t in range(T):
            q_t, v_t, adv_t, g_t = self.head.forward_step(h_seq[t])
            q_seq[t] = q_t
            v_seq[t] = v_t
            adv_seq[t] = adv_t
            gain_seq[t] = g_t

        return {
            "h_seq": h_seq,
            "q_seq": q_seq,
            "v_seq": v_seq,
            "adv_seq": adv_seq,
            "gain_seq": gain_seq,
            "h_final": h_final,
            "c_final": c_final,
            "caches": caches,
        }

    def backward_sequence(
        self,
        h_seq: np.ndarray,
        caches: List[LSTMStepCache],
        dq_seq: np.ndarray,
        dgain_seq: np.ndarray,
        burn_in_steps: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Backpropagate sequence gradients through Dueling Q-head and LSTM (respecting burn-in)."""
        T = h_seq.shape[0]
        dh_seq = np.zeros_like(h_seq)

        head_grads = {
            "W_val": np.zeros_like(self.head.W_val),
            "b_val": np.zeros_like(self.head.b_val),
            "W_adv": np.zeros_like(self.head.W_adv),
            "b_adv": np.zeros_like(self.head.b_adv),
            "W_gain": np.zeros_like(self.head.W_gain),
            "b_gain": np.zeros_like(self.head.b_gain),
        }

        start_t = max(0, int(burn_in_steps))
        for t in range(start_t, T):
            dh_t, step_grads = self.head.backward_step(
                h_seq[t], dq_seq[t], dgain_seq[t]
            )
            dh_seq[t] = dh_t
            for k in head_grads:
                head_grads[k] += step_grads[k]

        lstm_grads = self.encoder.backward_sequence(
            dh_seq, caches, burn_in_steps=burn_in_steps
        )
        grads = {}
        grads.update(lstm_grads)
        grads.update(head_grads)
        return grads

    def parameters(self) -> Dict[str, np.ndarray]:
        """Return dictionary of mutable parameter arrays."""
        return {
            "W_ih": self.encoder.W_ih,
            "W_hh": self.encoder.W_hh,
            "b_ih": self.encoder.b_ih,
            "W_val": self.head.W_val,
            "b_val": self.head.b_val,
            "W_adv": self.head.W_adv,
            "b_adv": self.head.b_adv,
            "W_gain": self.head.W_gain,
            "b_gain": self.head.b_gain,
        }

    def get_state_dict(self) -> Dict[str, np.ndarray]:
        """Deep copy all weights and normalization vectors."""
        d = {
            "feature_scale": self.encoder.feature_scale.copy(),
            "feature_offset": self.encoder.feature_offset.copy(),
        }
        for k, v in self.parameters().items():
            d[k] = v.copy()
        return d

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """Load weights and normalization vectors into network."""
        if "feature_scale" in state_dict:
            self.encoder.feature_scale[:] = state_dict["feature_scale"]
        if "feature_offset" in state_dict:
            self.encoder.feature_offset[:] = state_dict["feature_offset"]
        params = self.parameters()
        for k, arr in params.items():
            if k not in state_dict:
                raise KeyError(f"Missing parameter '{k}' in state_dict")
            if arr.shape != state_dict[k].shape:
                raise ValueError(
                    f"Shape mismatch for '{k}': expected {arr.shape}, got {state_dict[k].shape}"
                )
            arr[:] = state_dict[k]

    def clone(self) -> "R2D2Network":
        net = copy.deepcopy(self)
        return net
