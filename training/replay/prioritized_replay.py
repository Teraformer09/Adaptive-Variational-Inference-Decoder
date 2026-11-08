"""Prioritized Sequence Replay Buffer with Kapturowski max/mean TD-error mixture."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from training.replay.sequence_sampler import SequenceSegment


class PrioritizedSequenceReplayBuffer:
    """R2D2 Prioritized Sequence Replay Buffer.

    Priority aggregation across unroll steps t = 1..T_unroll (Kapturowski et al. Eq. 2):
      p_i = eta * max_t |delta_{i,t}| + (1 - eta) * mean_t |delta_{i,t}|
    with eta = 0.9.

    Sampling probability:
      P(i) = (p_i + eps)^alpha / sum_k (p_k + eps)^alpha

    Importance-sampling weight:
      w_i = (N * P(i))^{-beta} / max_j ((N * P(j))^{-beta})
    """

    def __init__(
        self,
        capacity: int = 4096,
        alpha: float = 0.6,
        eta: float = 0.9,
        eps: float = 1e-4,
        seed: int = 20260407,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        self.storage: List[SequenceSegment] = []
        self.priorities: List[float] = []
        self.pos: int = 0
        self.max_priority: float = 1.0

    def __len__(self) -> int:
        return len(self.storage)

    def compute_sequence_priority(self, td_errors: np.ndarray) -> float:
        """Compute Kapturowski mixture priority from sequence TD errors."""
        abs_td = np.abs(np.asarray(td_errors, dtype=np.float64))
        if abs_td.size == 0:
            return self.max_priority
        p = self.eta * float(np.max(abs_td)) + (1.0 - self.eta) * float(np.mean(abs_td))
        return max(float(p + self.eps), self.eps)

    def add(
        self,
        segment: SequenceSegment,
        td_errors: Optional[np.ndarray] = None,
    ) -> None:
        """Insert a sequence segment with initial priority."""
        p = (
            self.compute_sequence_priority(td_errors)
            if td_errors is not None
            else self.max_priority
        )
        self.max_priority = max(self.max_priority, p)

        if len(self.storage) < self.capacity:
            self.storage.append(segment)
            self.priorities.append(p)
        else:
            self.storage[self.pos] = segment
            self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> Tuple[List[SequenceSegment], np.ndarray, np.ndarray]:
        """Sample a prioritized batch of sequence segments, returning (batch, indices, is_weights)."""
        n = len(self.storage)
        if n == 0:
            raise ValueError("Cannot sample from an empty PrioritizedSequenceReplayBuffer")

        prios = np.asarray(self.priorities, dtype=np.float64)
        scaled = np.power(prios + self.eps, self.alpha)
        probs = scaled / np.sum(scaled)

        replace = n < batch_size
        indices = self.rng.choice(n, size=batch_size, replace=replace, p=probs)

        # Importance-sampling weights
        raw_weights = np.power(n * probs[indices], -float(beta))
        max_weight = np.max(np.power(n * probs, -float(beta)))
        is_weights = (raw_weights / max(max_weight, 1e-12)).astype(np.float64)

        batch = [self.storage[int(i)] for i in indices]
        return batch, indices.astype(np.int64), is_weights

    def update_priorities(
        self,
        indices: np.ndarray,
        sequence_td_errors: List[np.ndarray],
    ) -> None:
        """Update priorities for sampled sequences from updated unroll TD errors."""
        for idx, td_err in zip(indices, sequence_td_errors):
            p = self.compute_sequence_priority(td_err)
            self.priorities[int(idx)] = p
            self.max_priority = max(self.max_priority, p)
