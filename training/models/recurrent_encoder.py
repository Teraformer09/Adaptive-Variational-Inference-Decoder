"""64-unit Recurrent LSTM Encoder with analytical Backpropagation Through Time (BPTT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


@dataclass
class LSTMStepCache:
    """Cached activations for a single LSTM timestep needed by BPTT."""
    x_norm: np.ndarray
    h_prev: np.ndarray
    c_prev: np.ndarray
    i_gate: np.ndarray
    f_gate: np.ndarray
    g_gate: np.ndarray
    o_gate: np.ndarray
    c_next: np.ndarray
    tanh_c_next: np.ndarray
    h_next: np.ndarray


class RecurrentLSTMEncoder:
    """Single-layer 64-unit LSTM encoder matching C++ deployment bit-for-bit.

    Gate layout in `W_ih` (4H x D), `W_hh` (4H x H), and `b_ih` (4H):
      - [0:H]     -> input gate i_t
      - [H:2H]    -> forget gate f_t
      - [2H:3H]   -> cell candidate g_t
      - [3H:4H]   -> output gate o_t
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        seed: int = 20260405,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        rng = np.random.default_rng(seed)

        # Feature normalization constants matching deployment schema
        self.feature_scale = np.array(
            [20.0, 15.0, 20.0, 15.0, 4.0, 6.0, 15.0, 1.0], dtype=np.float64
        )
        self.feature_offset = np.array(
            [-0.35, -0.45, -0.30, -0.60, -0.70, -0.35, -0.25, -0.50], dtype=np.float64
        )

        H = self.hidden_dim
        D = self.input_dim
        scale_ih = np.sqrt(2.0 / (D + H))
        scale_hh = np.sqrt(1.0 / H)

        self.W_ih = rng.normal(0.0, scale_ih, size=(4 * H, D)).astype(np.float64)
        self.W_hh = rng.normal(0.0, scale_hh, size=(4 * H, H)).astype(np.float64)
        self.b_ih = np.zeros(4 * H, dtype=np.float64)
        # Positive forget-gate bias initialization (+1.0) per Jozefowicz et al. / Kapturowski et al.
        self.b_ih[H : 2 * H] = 1.0

    def normalize_features(self, x_raw: np.ndarray) -> np.ndarray:
        """Normalize raw 8-D feature vector into [-4.0, 4.0]."""
        x = np.asarray(x_raw, dtype=np.float64)
        return np.clip(x * self.feature_scale + self.feature_offset, -4.0, 4.0)

    def zero_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return zero (h_0, c_0) state vectors."""
        return (
            np.zeros(self.hidden_dim, dtype=np.float64),
            np.zeros(self.hidden_dim, dtype=np.float64),
        )

    def step(
        self,
        x_raw: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, LSTMStepCache]:
        """Execute a single recurrent LSTM step."""
        H = self.hidden_dim
        x_norm = self.normalize_features(x_raw)
        gates = self.W_ih @ x_norm + self.W_hh @ h_prev + self.b_ih

        i_gate = _sigmoid(gates[0:H])
        f_gate = _sigmoid(gates[H : 2 * H])
        g_gate = np.tanh(np.clip(gates[2 * H : 3 * H], -15.0, 15.0))
        o_gate = _sigmoid(gates[3 * H : 4 * H])

        c_next = f_gate * c_prev + i_gate * g_gate
        tanh_c_next = np.tanh(np.clip(c_next, -15.0, 15.0))
        h_next = o_gate * tanh_c_next

        cache = LSTMStepCache(
            x_norm=x_norm,
            h_prev=h_prev.copy(),
            c_prev=c_prev.copy(),
            i_gate=i_gate,
            f_gate=f_gate,
            g_gate=g_gate,
            o_gate=o_gate,
            c_next=c_next,
            tanh_c_next=tanh_c_next,
            h_next=h_next,
        )
        return h_next, c_next, cache

    def forward_sequence(
        self,
        obs_seq: np.ndarray,
        h_init: Optional[np.ndarray] = None,
        c_init: Optional[np.ndarray] = None,
        burn_in_steps: int = 0,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], List[LSTMStepCache]]:
        """Unroll LSTM over a sequence of shape (T, D) with optional burn-in."""
        T = obs_seq.shape[0]
        if h_init is None or c_init is None:
            h, c = self.zero_state()
        else:
            h = h_init.astype(np.float64).copy()
            c = c_init.astype(np.float64).copy()

        h_seq = np.zeros((T, self.hidden_dim), dtype=np.float64)
        caches: List[LSTMStepCache] = []
        for t in range(T):
            h, c, cache = self.step(obs_seq[t], h, c)
            # If t == burn_in_steps - 1, stop-gradient decouples burn-in state history
            if burn_in_steps > 0 and t == burn_in_steps - 1:
                h = h.copy()
                c = c.copy()
            h_seq[t] = h
            caches.append(cache)
        return h_seq, (h, c), caches

    def backward_sequence(
        self,
        dh_seq: np.ndarray,
        caches: List[LSTMStepCache],
        burn_in_steps: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Analytical BPTT backward pass over unroll steps [burn_in_steps : T]."""
        H = self.hidden_dim
        T = len(caches)
        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        db_ih = np.zeros_like(self.b_ih)

        dh_next = np.zeros(H, dtype=np.float64)
        dc_next = np.zeros(H, dtype=np.float64)

        start_t = max(0, int(burn_in_steps))
        for t in range(T - 1, start_t - 1, -1):
            cache = caches[t]
            dh = dh_seq[t] + dh_next

            do = dh * cache.tanh_c_next
            dc = dh * cache.o_gate * (1.0 - cache.tanh_c_next ** 2) + dc_next

            df = dc * cache.c_prev
            di = dc * cache.g_gate
            dg = dc * cache.i_gate

            di_raw = di * cache.i_gate * (1.0 - cache.i_gate)
            df_raw = df * cache.f_gate * (1.0 - cache.f_gate)
            dg_raw = dg * (1.0 - cache.g_gate ** 2)
            do_raw = do * cache.o_gate * (1.0 - cache.o_gate)

            dgates = np.concatenate([di_raw, df_raw, dg_raw, do_raw], axis=0)

            dW_ih += np.outer(dgates, cache.x_norm)
            dW_hh += np.outer(dgates, cache.h_prev)
            db_ih += dgates

            dh_next = self.W_hh.T @ dgates
            dc_next = dc * cache.f_gate

        return {"W_ih": dW_ih, "W_hh": dW_hh, "b_ih": db_ih}
