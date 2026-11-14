"""Deterministic 10 Hz Simulation Sampling Clock (`sampling_clock.py`).

Enforces:
  - t_n = t_0 + n / 10.0 (100 ms sampling interval, 10 samples/second).
  - Current 100-second assignment window w(t_n) = floor(t_n / 100.0).
  - Detection of 100-second window boundary crossings.
  - Explicit separation from the 1 MHz QEC syndrome round clock.
"""

from __future__ import annotations

import math


class SamplingClock:
    """Deterministic simulation clock for 10 Hz synchronized multi-generator sampling."""

    def __init__(
        self,
        t0: float = 0.0,
        rate_hz: float = 10.0,
        window_duration_sec: float = 100.0,
    ) -> None:
        if rate_hz <= 0.0 or not math.isfinite(rate_hz):
            raise ValueError(f"rate_hz must be positive and finite, got {rate_hz}")
        if window_duration_sec <= 0.0 or not math.isfinite(window_duration_sec):
            raise ValueError(f"window_duration_sec must be positive, got {window_duration_sec}")
        self.t0 = float(t0)
        self.rate_hz = float(rate_hz)
        self.sampling_interval = 1.0 / self.rate_hz
        self.window_duration_sec = float(window_duration_sec)
        self.samples_per_window = int(round(self.window_duration_sec * self.rate_hz))

        self._sample_id: int = 0
        self._prev_window_id: int | None = None
        self._boundary_crossed: bool = True

    def reset(self) -> None:
        self._sample_id = 0
        self._prev_window_id = None
        self._boundary_crossed = True

    @property
    def current_sample_id(self) -> int:
        return self._sample_id

    @property
    def timestamp(self) -> float:
        return round(self.t0 + self._sample_id * self.sampling_interval, 9)

    @property
    def current_window(self) -> int:
        return int(math.floor((self.timestamp + 1e-12) / self.window_duration_sec))

    @property
    def window_boundary_crossed(self) -> bool:
        return self._boundary_crossed

    def step(self) -> tuple[int, float, int, bool]:
        """Return `(sample_id, timestamp, window_id, boundary_crossed)` and advance counter."""
        sid = self._sample_id
        ts = self.timestamp
        wid = self.current_window
        crossed = self._prev_window_id is None or wid != self._prev_window_id
        self._boundary_crossed = crossed
        self._prev_window_id = wid
        self._sample_id += 1
        return sid, ts, wid, crossed

    def iter_duration(self, duration_seconds: float):
        """Yield `(sample_id, timestamp, window_id)` for `int(round(duration_seconds * rate_hz))` steps."""
        n_steps = int(round(float(duration_seconds) * self.rate_hz))
        for _ in range(n_steps):
            sid, ts, wid, _ = self.step()
            yield sid, ts, wid
