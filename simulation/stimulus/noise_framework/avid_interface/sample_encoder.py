"""Sample Encoder bridging 10 Hz Composite Noise Samples and QEC Syndrome Rounds (`sample_encoder.py`).

Implements Stage 5 Step 10/12 and Stage 6:
  - Converts a sequence of `CompositeNoiseSample` objects into physical X/Z data faults,
    ideal syndromes, and measurement-noisy observed syndromes.
  - Explicitly bridges the 10 Hz composite noise-condition clock (100 ms per sample)
    and the QEC syndrome round clock via `rounds_per_composite_sample` (default 1 in
    trajectory-sample mode, or configurable block hold for multi-round segments).
  - Preserves X/Z syndrome separation, round sequence IDs, window IDs, and observation
    validity masks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    CompositeNoiseSample,
    derive_deterministic_seed,
)
from simulation.stimulus.noise_framework.syndrome.fault_realizer import FaultRealizer
from simulation.stimulus.noise_framework.syndrome.measurement_model import MeasurementModel
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)
from simulation.stimulus.noise_framework.syndrome.syndrome_encoder import SyndromeEncoder


class SampleEncoder:
    """Encodes composite noise samples into physical fault and syndrome trajectories."""

    def __init__(
        self,
        rounds_per_composite_sample: int = 1,
        sampling_interval_sec: float = 0.1,
        qec_round_period_us: float = 1.0,
    ) -> None:
        if int(rounds_per_composite_sample) < 1:
            raise ValueError("rounds_per_composite_sample must be >= 1")
        self.rounds_per_composite_sample = int(rounds_per_composite_sample)
        self.sampling_interval_sec = float(sampling_interval_sec)
        self.qec_round_period_us = float(qec_round_period_us)
        self.fault_realizer = FaultRealizer()
        self.syndrome_encoder = SyndromeEncoder()
        self.measurement_model = MeasurementModel(sampling_interval_sec=self.sampling_interval_sec)

    def encode_composite_trajectory(
        self,
        composite_samples: Sequence[CompositeNoiseSample],
        master_seed: int = 42,
        trajectory_id: str = "traj_000",
        split_id: str = "train",
    ) -> Dict[str, Any]:
        """Realize faults, ideal syndromes, and measurement-noisy syndromes for a trajectory."""
        if not composite_samples:
            raise ValueError("composite_samples must be non-empty")

        n_samples = len(composite_samples)
        r_per_s = self.rounds_per_composite_sample
        total_rounds = n_samples * r_per_s

        fault_rng = np.random.default_rng(
            derive_deterministic_seed(master_seed, "fault_realizer", trajectory_id)
        )
        meas_rng = np.random.default_rng(
            derive_deterministic_seed(master_seed, "measurement_model", trajectory_id)
        )
        self.measurement_model.reset()

        theta_z_arr = np.zeros((total_rounds, 21), dtype=np.float64)
        theta_x_arr = np.zeros((total_rounds, 21), dtype=np.float64)
        fault_masks_z = np.zeros((total_rounds, 9), dtype=np.uint8)
        fault_masks_x = np.zeros((total_rounds, 9), dtype=np.uint8)
        events_z_21 = np.zeros((total_rounds, 21), dtype=np.uint8)
        events_x_21 = np.zeros((total_rounds, 21), dtype=np.uint8)
        logical_z = np.zeros(total_rounds, dtype=np.uint8)
        logical_x = np.zeros(total_rounds, dtype=np.uint8)
        sample_ids = np.zeros(total_rounds, dtype=np.int32)
        window_ids = np.zeros(total_rounds, dtype=np.int32)
        sample_timestamps = np.zeros(total_rounds, dtype=np.float64)
        qec_round_timestamps_us = np.zeros(total_rounds, dtype=np.float64)
        burst_flags = np.zeros(total_rounds, dtype=np.uint8)

        meas_params_expanded: List[MeasurementNoiseParameters] = []
        assignment_by_window: Dict[int, Dict[str, str]] = {}

        for idx, comp in enumerate(composite_samples):
            r_start = idx * r_per_s
            r_end = r_start + r_per_s

            tz = np.asarray(comp.composite_parameters["theta_z"], dtype=np.float64)
            tx = np.asarray(comp.composite_parameters["theta_x"], dtype=np.float64)
            theta_z_arr[r_start:r_end] = tz[None, :]
            theta_x_arr[r_start:r_end] = tx[None, :]

            realized = self.fault_realizer.realize(comp, rng=fault_rng, n_rounds=r_per_s)
            ez = np.asarray(realized["err_z"], dtype=np.uint8)
            ex = np.asarray(realized["err_x"], dtype=np.uint8)
            ev_z = np.asarray(realized["events_z_21"], dtype=np.uint8)
            ev_x = np.asarray(realized["events_x_21"], dtype=np.uint8)

            if r_per_s == 1 and ez.ndim == 1:
                fault_masks_z[r_start] = ez
                fault_masks_x[r_start] = ex
                events_z_21[r_start] = ev_z
                events_x_21[r_start] = ev_x
                logical_z[r_start] = int(realized["logical_z_flip"])
                logical_x[r_start] = int(realized["logical_x_flip"])
            else:
                fault_masks_z[r_start:r_end] = ez
                fault_masks_x[r_start:r_end] = ex
                events_z_21[r_start:r_end] = ev_z
                events_x_21[r_start:r_end] = ev_x
                logical_z[r_start:r_end] = np.asarray(realized["logical_z_flip"], dtype=np.uint8)
                logical_x[r_start:r_end] = np.asarray(realized["logical_x_flip"], dtype=np.uint8)

            sample_ids[r_start:r_end] = int(comp.sample_id)
            window_ids[r_start:r_end] = int(comp.window_id)
            sample_timestamps[r_start:r_end] = float(comp.timestamp)
            for k in range(r_per_s):
                qec_round_timestamps_us[r_start + k] = (
                    float(comp.timestamp) * 1e6 + k * self.qec_round_period_us
                )
            burst_flags[r_start:r_end] = 1 if comp.metadata.get("in_burst_window", False) else 0

            if int(comp.window_id) not in assignment_by_window:
                assignment_by_window[int(comp.window_id)] = dict(
                    comp.metadata.get("profile_assignments", {})
                )

            mp: Optional[MeasurementNoiseParameters] = comp.composite_parameters.get("measurement")
            if mp is None:
                mp = MeasurementNoiseParameters(
                    q_base_x=np.zeros(4, dtype=np.float64),
                    q_base_z=np.zeros(4, dtype=np.float64),
                    temporal_sticky_prob=0.0,
                    detector_dropout_prob=0.0,
                    observation_delay_samples=0,
                    generator_id="ideal_default",
                    profile_id="PD_15",
                    sample_id=int(comp.sample_id),
                    timestamp=float(comp.timestamp),
                    window_id=int(comp.window_id),
                )
            for _ in range(r_per_s):
                meas_params_expanded.append(mp)

        ideal_sx, ideal_sz = self.syndrome_encoder.encode(
            x_errors=fault_masks_x, z_errors=fault_masks_z
        )
        meas_out = self.measurement_model.apply_trajectory(
            ideal_syndromes_x=ideal_sx,
            ideal_syndromes_z=ideal_sz,
            parameters_seq=meas_params_expanded,
            rng=meas_rng,
        )

        return {
            "trajectory_id": trajectory_id,
            "split_id": split_id,
            "master_seed": int(master_seed),
            "num_composite_samples": n_samples,
            "rounds_per_composite_sample": r_per_s,
            "total_rounds": total_rounds,
            "theta_z": theta_z_arr,
            "theta_x": theta_x_arr,
            "fault_masks_z": fault_masks_z,
            "fault_masks_x": fault_masks_x,
            "events_z_21": events_z_21,
            "events_x_21": events_x_21,
            "ideal_syndromes_x": ideal_sx,
            "ideal_syndromes_z": ideal_sz,
            "syndromes_x": meas_out["observed_syndromes_x"],
            "syndromes_z": meas_out["observed_syndromes_z"],
            "validity_mask_x": meas_out["validity_mask_x"],
            "validity_mask_z": meas_out["validity_mask_z"],
            "measurement_faults_x": meas_out["measurement_faults_x"],
            "measurement_faults_z": meas_out["measurement_faults_z"],
            "detector_events_x": meas_out["detector_events_x"],
            "detector_events_z": meas_out["detector_events_z"],
            "observation_delays": meas_out["observation_delays"],
            "logical_error_z": logical_z,
            "logical_error_x": logical_x,
            "sample_ids": sample_ids,
            "window_ids": window_ids,
            "timestamps": sample_timestamps,
            "event_timestamps": meas_out["event_timestamps"],
            "delivery_timestamps": meas_out["delivery_timestamps"],
            "qec_round_timestamps_us": qec_round_timestamps_us,
            "in_burst_window": burst_flags,
            "assignment_history": assignment_by_window,
            "measurement_parameters_vec": np.stack([m.to_vector() for m in meas_params_expanded]),
        }
