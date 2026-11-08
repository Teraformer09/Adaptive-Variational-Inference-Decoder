"""Overlapping sequence slicer with stored initial recurrent hidden state (h_0, c_0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class SequenceSegment:
    """Fixed-length sequence segment for R2D2 replay with stored initial hidden state.

    Total sequence length T_seq = burn_in_steps + unroll_steps + n_step_td.
    - [0 : burn_in_steps] -> burn-in window (no gradient propagation)
    - [burn_in_steps : burn_in_steps + unroll_steps] -> training unroll window
    - [burn_in_steps + unroll_steps : T_seq] -> bootstrap lookahead for n-step TD targets
    """

    obs: np.ndarray          # shape (T_seq, 8)
    actions: np.ndarray      # shape (T_seq,), int64
    rewards: np.ndarray      # shape (T_seq,), float64
    dones: np.ndarray        # shape (T_seq,), float64 (0.0 or 1.0)
    target_gains: np.ndarray # shape (T_seq, 9), float64
    optimal_actions: np.ndarray # shape (T_seq,), int64 (oracle action reference for diagnostics)
    h_init: np.ndarray       # shape (64,), float64
    c_init: np.ndarray       # shape (64,), float64
    family: str = "stationary_independent"
    seed: int = 0


def slice_episode_into_segments(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    target_gains: np.ndarray,
    optimal_actions: np.ndarray,
    h_states: np.ndarray,
    c_states: np.ndarray,
    burn_in_steps: int = 4,
    unroll_steps: int = 4,
    n_step_td: int = 3,
    stride: int = 2,
    family: str = "stationary_independent",
    seed: int = 0,
) -> List[SequenceSegment]:
    """Slice an episode of length T_ep into overlapping R2D2 replay segments."""
    T_ep = obs.shape[0]
    T_seq = int(burn_in_steps + unroll_steps + n_step_td)
    segments: List[SequenceSegment] = []

    if T_ep < T_seq:
        # Pad short episode by repeating last transition
        pad_len = T_seq - T_ep
        obs = np.concatenate([obs, np.repeat(obs[-1:], pad_len, axis=0)], axis=0)
        actions = np.concatenate([actions, np.repeat(actions[-1:], pad_len, axis=0)], axis=0)
        rewards = np.concatenate([rewards, np.zeros(pad_len, dtype=np.float64)], axis=0)
        dones = np.concatenate([dones, np.ones(pad_len, dtype=np.float64)], axis=0)
        target_gains = np.concatenate(
            [target_gains, np.repeat(target_gains[-1:], pad_len, axis=0)], axis=0
        )
        optimal_actions = np.concatenate(
            [optimal_actions, np.repeat(optimal_actions[-1:], pad_len, axis=0)], axis=0
        )
        h_states = np.concatenate([h_states, np.repeat(h_states[-1:], pad_len, axis=0)], axis=0)
        c_states = np.concatenate([c_states, np.repeat(c_states[-1:], pad_len, axis=0)], axis=0)
        T_ep = T_seq

    for start in range(0, T_ep - T_seq + 1, max(1, stride)):
        end = start + T_seq
        segments.append(
            SequenceSegment(
                obs=obs[start:end].astype(np.float64).copy(),
                actions=actions[start:end].astype(np.int64).copy(),
                rewards=rewards[start:end].astype(np.float64).copy(),
                dones=dones[start:end].astype(np.float64).copy(),
                target_gains=target_gains[start:end].astype(np.float64).copy(),
                optimal_actions=optimal_actions[start:end].astype(np.int64).copy(),
                h_init=h_states[start].astype(np.float64).copy(),
                c_init=c_states[start].astype(np.float64).copy(),
                family=family,
                seed=seed,
            )
        )
    return segments
