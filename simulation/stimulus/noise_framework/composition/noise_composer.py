"""Physical Multi-Generator Noise Composer (`noise_composer.py`).

Implements Stage 4 and Stage 5 Step 8 / Step 10:
  - Groups simultaneous generator contributions by physical output domain:
      * `probability_parameter` (21-mechanism Z and X fault rate vectors)
      * `pauli_fault_mask` (9-qubit binary X and Z Pauli masks composed via modulo-2 XOR)
      * `measurement_fault_parameters` (`MeasurementNoiseParameters` composed via
         q_combined = 1 - prod_j (1 - q_j))
  - Rejects incompatible domains, timestamp mismatches, or invalid shapes.
  - Preserves every individual generator's contribution and provenance inside `CompositeNoiseSample`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence
import numpy as np

from simulation.stimulus.noise_framework.composition.pauli_composition import (
    compose_pauli_masks_xor,
)
from simulation.stimulus.noise_framework.composition.probability_composition import (
    compose_independent_bernoulli,
    place_into_21_mechanism_vector,
)
from simulation.stimulus.noise_framework.config_schema import (
    VALID_OUTPUT_DOMAINS,
    CompositeNoiseSample,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


class NoiseComposer:
    """Composes simultaneous multi-generator contributions into a `CompositeNoiseSample`."""

    def __init__(self, default_base_rate: float = 0.002) -> None:
        self.default_base_rate = float(default_base_rate)

    def _normalize_contribution(self, item: Any) -> NoiseContribution:
        if isinstance(item, NoiseContribution):
            return item
        if isinstance(item, MeasurementNoiseParameters):
            return item.to_contribution()
        if hasattr(item, "to_contribution"):
            return item.to_contribution()
        raise TypeError(f"Unsupported contribution object type: {type(item)}")

    def validate_compatibility(
        self,
        contributions: Sequence[Any],
        timestamp: float | None = None,
        sample_id: int | None = None,
    ) -> List[NoiseContribution]:
        if not contributions:
            raise ValueError("NoiseComposer requires a non-empty list of contributions")

        normed: List[NoiseContribution] = []
        seen_generator_ids = set()
        ref_ts = float(timestamp) if timestamp is not None else float(contributions[0].timestamp)
        ref_sid = int(sample_id) if sample_id is not None else int(contributions[0].sample_id)
        ref_wid = int(contributions[0].window_id)

        for raw in contributions:
            c = self._normalize_contribution(raw)
            if c.generator_id in seen_generator_ids:
                raise ValueError(
                    f"Duplicate contribution from generator '{c.generator_id}' at sample {ref_sid}"
                )
            seen_generator_ids.add(c.generator_id)
            if c.output_domain not in VALID_OUTPUT_DOMAINS:
                raise ValueError(f"Unsupported output_domain '{c.output_domain}'")
            if int(c.sample_id) != ref_sid:
                raise ValueError(
                    f"Sample ID mismatch in NoiseComposer: {c.sample_id} != {ref_sid}"
                )
            if abs(float(c.timestamp) - ref_ts) > 1e-9:
                raise ValueError(
                    f"Timestamp mismatch in NoiseComposer: {c.timestamp} != {ref_ts}"
                )
            if int(c.window_id) != ref_wid:
                raise ValueError(
                    f"Window ID mismatch in NoiseComposer: {c.window_id} != {ref_wid}"
                )
            normed.append(c)

        return normed

    def preserve_provenance(self, contributions: Sequence[NoiseContribution]) -> Dict[str, Any]:
        return {
            "generator_ids": [c.generator_id for c in contributions],
            "profile_assignments": {c.generator_id: c.profile_id for c in contributions},
            "domains": {c.generator_id: c.output_domain for c in contributions},
            "num_contributions": len(contributions),
        }

    def _extract_21_vector(self, c: NoiseContribution) -> np.ndarray:
        arr = np.asarray(c.values, dtype=np.float64)
        if arr.shape == (21,):
            return arr
        if arr.shape == (9,):
            return place_into_21_mechanism_vector(single_probs=arr, pair_probs=None)
        if arr.shape == (12,):
            return place_into_21_mechanism_vector(single_probs=None, pair_probs=arr)
        raise ValueError(
            f"Probability contribution from '{c.generator_id}' has unsupported shape {arr.shape}; "
            f"expected (21,), (9,), or (12,)"
        )

    def _compose_measurement_contributions(
        self,
        meas_contribs: List[NoiseContribution],
        timestamp: float,
        sample_id: int,
        window_id: int,
    ) -> MeasurementNoiseParameters:
        q_x_list: List[np.ndarray] = []
        q_z_list: List[np.ndarray] = []
        sticky_list: List[np.ndarray] = []
        dropout_list: List[np.ndarray] = []
        delays: List[int] = []
        gen_ids: List[str] = []
        prof_ids: List[str] = []

        for c in meas_contribs:
            mp = c.metadata.get("measurement_parameters")
            if isinstance(mp, MeasurementNoiseParameters):
                q_x_list.append(mp.q_base_x)
                q_z_list.append(mp.q_base_z)
                sticky_list.append(np.array([mp.temporal_sticky_prob], dtype=np.float64))
                dropout_list.append(np.array([mp.detector_dropout_prob], dtype=np.float64))
                delays.append(int(mp.observation_delay_samples))
            else:
                vec = np.asarray(c.values, dtype=np.float64)
                if vec.shape != (11,):
                    raise ValueError(
                        f"Measurement contribution from '{c.generator_id}' must have shape (11,), got {vec.shape}"
                    )
                delay_raw = vec[10]
                if not math.isclose(delay_raw, round(delay_raw), abs_tol=1e-9) or delay_raw < 0:
                    raise ValueError(f"Invalid delay value {delay_raw} in measurement contribution")
                q_x_list.append(vec[0:4])
                q_z_list.append(vec[4:8])
                sticky_list.append(np.array([vec[8]], dtype=np.float64))
                dropout_list.append(np.array([vec[9]], dtype=np.float64))
                delays.append(int(round(delay_raw)))
            gen_ids.append(c.generator_id)
            prof_ids.append(c.profile_id)

        comp_qx = compose_independent_bernoulli(q_x_list)
        comp_qz = compose_independent_bernoulli(q_z_list)
        comp_sticky = float(compose_independent_bernoulli(sticky_list)[0])
        comp_dropout = float(compose_independent_bernoulli(dropout_list)[0])
        comp_delay = int(max(delays))

        return MeasurementNoiseParameters(
            q_base_x=comp_qx,
            q_base_z=comp_qz,
            temporal_sticky_prob=comp_sticky,
            detector_dropout_prob=comp_dropout,
            observation_delay_samples=comp_delay,
            generator_id="+".join(gen_ids),
            profile_id="+".join(prof_ids),
            sample_id=sample_id,
            timestamp=timestamp,
            window_id=window_id,
            metadata={"source_generators": gen_ids, "source_profiles": prof_ids},
        )

    def compose(
        self,
        contributions: Sequence[Any],
        timestamp: float,
        sample_id: int,
    ) -> CompositeNoiseSample:
        normed = self.validate_compatibility(
            contributions=contributions, timestamp=timestamp, sample_id=sample_id
        )
        window_id = int(normed[0].window_id)

        prob_z_list: List[np.ndarray] = []
        prob_x_list: List[np.ndarray] = []
        mask_z_list: List[np.ndarray] = []
        mask_x_list: List[np.ndarray] = []
        meas_contribs: List[NoiseContribution] = []
        in_burst = False

        for c in normed:
            if bool(c.metadata.get("in_burst_window", False)):
                in_burst = True
            if c.output_domain == "probability_parameter":
                vec21 = self._extract_21_vector(c)
                basis = str(c.metadata.get("basis", "Z")).upper()
                if basis == "X":
                    prob_x_list.append(vec21)
                elif basis == "Z":
                    prob_z_list.append(vec21)
                elif basis == "BOTH":
                    prob_z_list.append(vec21)
                    prob_x_list.append(vec21)
                else:
                    raise ValueError(f"Unknown basis '{basis}' in contribution from {c.generator_id}")
            elif c.output_domain == "pauli_fault_mask":
                basis = str(c.metadata.get("basis", "Z")).upper()
                if basis == "X":
                    mask_x_list.append(np.asarray(c.values, dtype=np.uint8))
                elif basis == "Z":
                    mask_z_list.append(np.asarray(c.values, dtype=np.uint8))
                else:
                    raise ValueError(f"Unknown basis '{basis}' in mask from {c.generator_id}")
            elif c.output_domain == "measurement_fault_parameters":
                meas_contribs.append(c)

        composite_params: Dict[str, Any] = {}
        if prob_z_list:
            composite_params["theta_z"] = np.clip(
                compose_independent_bernoulli(prob_z_list), 1e-6, 0.48
            )
        else:
            default_z = np.full(21, self.default_base_rate, dtype=np.float64)
            default_z[9:] *= 0.25
            composite_params["theta_z"] = default_z

        if prob_x_list:
            composite_params["theta_x"] = np.clip(
                compose_independent_bernoulli(prob_x_list), 1e-6, 0.48
            )
        else:
            default_x = np.full(21, self.default_base_rate * 0.8, dtype=np.float64)
            default_x[9:] *= 0.25
            composite_params["theta_x"] = default_x

        if meas_contribs:
            composite_params["measurement"] = self._compose_measurement_contributions(
                meas_contribs=meas_contribs,
                timestamp=float(timestamp),
                sample_id=int(sample_id),
                window_id=window_id,
            )

        composite_faults: Dict[str, np.ndarray] | None = None
        if mask_z_list or mask_x_list:
            composite_faults = {}
            if mask_z_list:
                composite_faults["err_z"] = compose_pauli_masks_xor(mask_z_list)
            if mask_x_list:
                composite_faults["err_x"] = compose_pauli_masks_xor(mask_x_list)

        prov = self.preserve_provenance(normed)
        prov["in_burst_window"] = in_burst

        return CompositeNoiseSample(
            sample_id=int(sample_id),
            timestamp=float(timestamp),
            window_id=window_id,
            contributions=normed,
            composite_parameters=composite_params,
            composite_faults=composite_faults,
            metadata=prov,
        )
