"""Persistent 100-Second Window Assignment Scheduler (`assignment_scheduler.py`).

Enforces:
  - Window index w(t) = floor(t / 100.0).
  - Each active generator receives exactly one PD assignment per 100-second window.
  - Assignment remains invariant throughout [100*w, 100*(w+1)).
  - At each 100-second boundary, each generator receives a new PD assignment via
    `generator.assign_profile(profile)` WITHOUT resetting the generator's continuous
    internal stochastic state (unless `preserve_state=False`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.pd_registry import PDRegistry


class AssignmentScheduler:
    """Coordinates 100-second persistent PD assignments across all active generators."""

    def __init__(
        self,
        pd_assignment: PDAssignment,
        window_duration_sec: float = 100.0,
        preserve_generator_state: bool = True,
    ) -> None:
        if window_duration_sec <= 0.0 or not math.isfinite(window_duration_sec):
            raise ValueError(f"Invalid window_duration_sec {window_duration_sec}")
        self.pd_assignment = pd_assignment
        self.window_duration_sec = float(window_duration_sec)
        self.preserve_generator_state = bool(preserve_generator_state)

        self._generators: List[Any] = []
        self._pd_registry: Optional[PDRegistry] = None
        self._current_window: Optional[int] = None
        self._active_assignments: Dict[str, str] = {}

    def initialize(self, generators: Sequence[Any], pd_registry: PDRegistry) -> None:
        self._generators = list(generators)
        self._pd_registry = pd_registry
        self._current_window = None
        self._active_assignments.clear()
        # Initialize window 0 at t = 0.0
        self.advance(0.0)

    def window_for_timestamp(self, timestamp: float) -> int:
        if timestamp < 0.0 or not math.isfinite(timestamp):
            raise ValueError(f"Timestamp must be finite and non-negative, got {timestamp}")
        # Add tiny epsilon (1e-12) to prevent float representation jitter at exact multiples of 0.1s
        return int(math.floor((timestamp + 1e-12) / self.window_duration_sec))

    def advance(self, timestamp: float) -> bool:
        """Advance scheduler to `timestamp`. Returns True if a new 100s window assignment occurred."""
        if self._pd_registry is None:
            raise RuntimeError("AssignmentScheduler.initialize() must be called before advance()")
        w = self.window_for_timestamp(timestamp)
        if self._current_window is not None and w == self._current_window:
            return False

        self._current_window = w
        for gen in self._generators:
            gid = gen.generator_id
            pid = self.pd_assignment.assign(gid, w)
            prof = self._pd_registry.get(pid)
            if not self.preserve_generator_state and w > 0:
                gen.reset()
            gen.assign_profile(prof, window_id=w)
            self._active_assignments[gid] = pid
        return True

    def assignment_for(self, generator_id: str) -> str:
        if generator_id not in self._active_assignments:
            raise KeyError(f"Generator '{generator_id}' has no active PD assignment")
        return self._active_assignments[generator_id]

    def current_window(self) -> int:
        if self._current_window is None:
            raise RuntimeError("AssignmentScheduler has not been initialized")
        return self._current_window
