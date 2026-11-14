"""Independent per-generator Categorical PD Assignment (`pd_assignment.py`).

For generator j and 100-second window w:
  P_j^{(w)} ~ Categorical(pi_1, ..., pi_16)
using an isolated RNG stream derived from H(master_seed, "pd_assign", generator_id, window_id)
so that adding or modifying one generator never alters the assignment sequence of any other generator.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    derive_deterministic_seed,
    load_yaml_validated,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry


class PDAssignment:
    """Independently assigns a Probability Distribution (PD) from `PDRegistry` to each generator."""

    def __init__(
        self,
        pd_registry: PDRegistry,
        master_seed: int = 42,
        policy_name: str = "uniform_16",
        assignment_config_path: Optional[str | Path] = None,
        allowed_profile_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.pd_registry = pd_registry
        self.master_seed = int(master_seed)
        self.policy_name = str(policy_name)
        all_ids = self.pd_registry.list_profiles()

        raw_weights = np.ones(len(all_ids), dtype=np.float64) / float(len(all_ids))
        if assignment_config_path is not None:
            cfg = load_yaml_validated(assignment_config_path, required_keys=("policies",))
            policies = cfg["policies"]
            if self.policy_name not in policies:
                raise KeyError(
                    f"Unknown assignment policy '{self.policy_name}' in {assignment_config_path}"
                )
            w_list = policies[self.policy_name]["weights"]
            if len(w_list) != len(all_ids):
                raise ValueError(
                    f"Policy '{self.policy_name}' weight length {len(w_list)} != {len(all_ids)}"
                )
            raw_weights = np.asarray(w_list, dtype=np.float64)

        if np.any(raw_weights < 0.0) or not np.all(np.isfinite(raw_weights)):
            raise ValueError("Categorical assignment weights must be finite and non-negative")

        if allowed_profile_ids is not None:
            allowed_set = set(allowed_profile_ids)
            unknown = allowed_set - set(all_ids)
            if unknown:
                raise KeyError(f"Unknown allowed_profile_ids: {sorted(unknown)}")
            mask = np.array([1.0 if pid in allowed_set else 0.0 for pid in all_ids], dtype=np.float64)
            raw_weights = raw_weights * mask

        total_w = float(np.sum(raw_weights))
        if total_w <= 0.0:
            raise ValueError("Sum of categorical assignment weights must be strictly positive")
        self.probabilities = raw_weights / total_w
        self.profile_ids: List[str] = list(all_ids)

        # (generator_id, window_id) -> profile_id
        self._records: Dict[Tuple[str, int], str] = {}
        self._history: List[Dict[str, Any]] = []

    def assign(
        self,
        generator_id: str,
        window_id: int,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """Assign a PD to `generator_id` for 100-second window `window_id`."""
        if not generator_id:
            raise ValueError("generator_id must be non-empty")
        if window_id < 0:
            raise ValueError(f"window_id must be non-negative, got {window_id}")

        key = (generator_id, int(window_id))
        if key in self._records:
            return self._records[key]

        if rng is None:
            derived = derive_deterministic_seed(
                self.master_seed, "pd_assignment", generator_id, int(window_id)
            )
            local_rng = np.random.default_rng(derived)
        else:
            local_rng = rng

        idx = int(local_rng.choice(len(self.profile_ids), p=self.probabilities))
        chosen_pid = self.profile_ids[idx]
        self._records[key] = chosen_pid
        self._history.append(
            {
                "generator_id": generator_id,
                "window_id": int(window_id),
                "profile_id": chosen_pid,
                "policy_name": self.policy_name,
                "assignment_probability": float(self.probabilities[idx]),
            }
        )
        return chosen_pid

    def get_assignment(self, generator_id: str, window_id: int) -> str:
        key = (generator_id, int(window_id))
        if key not in self._records:
            raise KeyError(f"No PD assignment recorded for generator='{generator_id}', window={window_id}")
        return self._records[key]

    def assignment_manifest(self) -> Dict[str, Any]:
        counts = {pid: 0 for pid in self.profile_ids}
        for rec in self._history:
            counts[rec["profile_id"]] += 1
        total = max(len(self._history), 1)
        emp_freq = {pid: counts[pid] / float(total) for pid in self.profile_ids}
        return {
            "master_seed": self.master_seed,
            "policy_name": self.policy_name,
            "configured_probabilities": {
                pid: float(p) for pid, p in zip(self.profile_ids, self.probabilities)
            },
            "assignment_counts": counts,
            "empirical_frequencies": emp_freq,
            "total_assignments": len(self._history),
            "history": list(self._history),
        }
