"""Measurement-Noise & Imperfect Observation Model (`measurement_model.py`).

Implements Stage 5 Step 6 and Step 8 (`inst_0` .. `inst_10`):
  - Consumes ideal X/Z syndromes and `MeasurementNoiseParameters` (never loads YAML or assigns PDs).
  - Applies independent Bernoulli readout bit-flips `q_base_x`, `q_base_z`.
  - Applies a Markov temporal-correlation transition model governed by `temporal_sticky_prob`:
      P(m_{c,t} = 1 | m_{c,t-1} = 0) = q_{c,t}
      P(m_{c,t} = 1 | m_{c,t-1} = 1) = (1 - p_sticky) * q_{c,t} + p_sticky
  - Applies detector dropout with an explicit validity mask `v_{c,t} in {0, 1}` so dropped
    detector readings are never silently mistaken for valid zero syndrome bits.
  - Applies non-negative integer observation delay `d_obs`, preserving both `event_timestamps`
    and `delivery_timestamps`.
  - Keeps measurement faults strictly separate from physical data-qubit fault masks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np

from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


class MeasurementModel:
    """Applies readout bit-flips, temporal correlation, dropout, and observation delay."""

    def __init__(self, sampling_interval_sec: float = 0.1) -> None:
        self.sampling_interval_sec = float(sampling_interval_sec)
        self._prev_meas_x = np.zeros(4, dtype=np.uint8)
        self._prev_meas_z = np.zeros(4, dtype=np.uint8)

    def reset(self) -> None:
        self._prev_meas_x = np.zeros(4, dtype=np.uint8)
        self._prev_meas_z = np.zeros(4, dtype=np.uint8)

    def _sample_correlated_readout_step(
        self,
        q_base: np.ndarray,
        sticky_prob: float,
        prev_fault: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # P(m_t = 1 | m_{t-1} = 0) = q_base
        # P(m_t = 1 | m_{t-1} = 1) = (1 - sticky) * q_base + sticky
        p_flip = np.where(
            prev_fault == 1,
            (1.0 - sticky_prob) * q_base + sticky_prob,
            q_base,
        )
        p_flip = np.clip(p_flip, 0.0, 1.0)
        m_t = (rng.uniform(0.0, 1.0, size=4) < p_flip).astype(np.uint8)
        return m_t

    def apply_trajectory(
        self,
        ideal_syndromes_x: np.ndarray,
        ideal_syndromes_z: np.ndarray,
        parameters_seq: Sequence[MeasurementNoiseParameters] | MeasurementNoiseParameters,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Apply measurement noise across a trajectory of N samples (or a single sample)."""
        sx_in = np.asarray(ideal_syndromes_x, dtype=np.uint8) & 1
        sz_in = np.asarray(ideal_syndromes_z, dtype=np.uint8) & 1
        single_step = sx_in.ndim == 1
        if single_step:
            sx_in = sx_in[None, :]
            sz_in = sz_in[None, :]

        if sx_in.shape != sz_in.shape or sx_in.shape[1] != 4:
            raise ValueError(
                f"Ideal syndromes must have shape (N, 4), got {sx_in.shape} and {sz_in.shape}"
            )
        n_steps = sx_in.shape[0]

        if isinstance(parameters_seq, MeasurementNoiseParameters):
            param_list: List[MeasurementNoiseParameters] = [parameters_seq] * n_steps
        else:
            param_list = list(parameters_seq)
            if len(param_list) != n_steps:
                raise ValueError(
                    f"Length of parameters_seq ({len(param_list)}) != n_steps ({n_steps})"
                )

        meas_faults_x = np.zeros((n_steps, 4), dtype=np.uint8)
        meas_faults_z = np.zeros((n_steps, 4), dtype=np.uint8)
        noisy_undelayed_x = np.zeros((n_steps, 4), dtype=np.uint8)
        noisy_undelayed_z = np.zeros((n_steps, 4), dtype=np.uint8)
        validity_undelayed_x = np.ones((n_steps, 4), dtype=np.uint8)
        validity_undelayed_z = np.ones((n_steps, 4), dtype=np.uint8)

        event_timestamps = np.zeros(n_steps, dtype=np.float64)
        delivery_timestamps = np.zeros(n_steps, dtype=np.float64)
        delays = np.zeros(n_steps, dtype=np.int32)

        for t in range(n_steps):
            mp = param_list[t]
            if not isinstance(mp, MeasurementNoiseParameters):
                raise TypeError(f"Expected MeasurementNoiseParameters at step {t}, got {type(mp)}")

            m_x = self._sample_correlated_readout_step(
                mp.q_base_x, mp.temporal_sticky_prob, self._prev_meas_x, rng
            )
            m_z = self._sample_correlated_readout_step(
                mp.q_base_z, mp.temporal_sticky_prob, self._prev_meas_z, rng
            )
            self._prev_meas_x = m_x
            self._prev_meas_z = m_z

            meas_faults_x[t] = m_x
            meas_faults_z[t] = m_z

            # Dropout validity mask: v_{c,t} = 1 if valid, 0 if dropped
            if mp.detector_dropout_prob > 0.0:
                drop_x = (rng.uniform(0.0, 1.0, size=4) < mp.detector_dropout_prob).astype(np.uint8)
                drop_z = (rng.uniform(0.0, 1.0, size=4) < mp.detector_dropout_prob).astype(np.uint8)
                val_x = (1 - drop_x).astype(np.uint8)
                val_z = (1 - drop_z).astype(np.uint8)
            else:
                val_x = np.ones(4, dtype=np.uint8)
                val_z = np.ones(4, dtype=np.uint8)

            validity_undelayed_x[t] = val_x
            validity_undelayed_z[t] = val_z

            # Raw noisy syndrome gated by validity mask (invalid bits retained in validity_mask)
            noisy_undelayed_x[t] = (sx_in[t] ^ m_x) & val_x
            noisy_undelayed_z[t] = (sz_in[t] ^ m_z) & val_z

            d_obs = int(mp.observation_delay_samples)
            delays[t] = d_obs
            event_timestamps[t] = float(mp.timestamp)
            delivery_timestamps[t] = round(
                float(mp.timestamp) + d_obs * self.sampling_interval_sec, 9
            )

        observed_x = np.zeros((n_steps, 4), dtype=np.uint8)
        observed_z = np.zeros((n_steps, 4), dtype=np.uint8)
        validity_x = np.ones((n_steps, 4), dtype=np.uint8)
        validity_z = np.ones((n_steps, 4), dtype=np.uint8)
        source_event_indices = np.zeros(n_steps, dtype=np.int32)

        for t in range(n_steps):
            src_idx = max(0, t - int(delays[t]))
            source_event_indices[t] = src_idx
            observed_x[t] = noisy_undelayed_x[src_idx]
            observed_z[t] = noisy_undelayed_z[src_idx]
            validity_x[t] = validity_undelayed_x[src_idx]
            validity_z[t] = validity_undelayed_z[src_idx]

        # Detector difference events d_t = s_tilde_t XOR s_tilde_{t-1}
        detector_x = np.zeros_like(observed_x)
        detector_z = np.zeros_like(observed_z)
        detector_x[0] = observed_x[0]
        detector_z[0] = observed_z[0]
        if n_steps > 1:
            detector_x[1:] = observed_x[1:] ^ observed_x[:-1]
            detector_z[1:] = observed_z[1:] ^ observed_z[:-1]

        if single_step:
            return {
                "observed_syndromes_x": observed_x[0],
                "observed_syndromes_z": observed_z[0],
                "validity_mask_x": validity_x[0],
                "validity_mask_z": validity_z[0],
                "measurement_faults_x": meas_faults_x[0],
                "measurement_faults_z": meas_faults_z[0],
                "event_timestamps": float(event_timestamps[0]),
                "delivery_timestamps": float(delivery_timestamps[0]),
                "observation_delays": int(delays[0]),
                "source_event_indices": int(source_event_indices[0]),
                "detector_events_x": detector_x[0],
                "detector_events_z": detector_z[0],
            }

        return {
            "observed_syndromes_x": observed_x,
            "observed_syndromes_z": observed_z,
            "validity_mask_x": validity_x,
            "validity_mask_z": validity_z,
            "measurement_faults_x": meas_faults_x,
            "measurement_faults_z": meas_faults_z,
            "event_timestamps": event_timestamps,
            "delivery_timestamps": delivery_timestamps,
            "observation_delays": delays,
            "source_event_indices": source_event_indices,
            "detector_events_x": detector_x,
            "detector_events_z": detector_z,
        }
