"""Dynamic Measurement-Noise Generator (`measurement_noise_generator.py`).

Implements Stage 5 Steps 3, 4, 5, and 7:
  - Extends `BaseNoiseGenerator`.
  - Receives its assigned `NoiseProfile` from the shared 16-PD registry via `assign_profile()`.
  - Reads the explicit parameter mapping from `configs/noise_framework/measurement_parameter_mapping.yaml`.
  - Rejects incompatible PD profiles rather than silently reinterpreting them.
  - Samples `q_base_x` (4,), `q_base_z` (4,), `temporal_sticky_prob`, `detector_dropout_prob`,
    and `observation_delay_samples` using its independent RNG stream.
  - Returns a validated `MeasurementNoiseParameters` object at every 10 Hz sampling instant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    GeneratorSpec,
    NoiseProfile,
    load_yaml_validated,
)
from simulation.stimulus.noise_framework.generators.base_generator import BaseNoiseGenerator
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


class MeasurementNoiseGenerator(BaseNoiseGenerator):
    """Generates dynamic measurement-fault parameter realizations from its assigned PD profile."""

    REQUIRED_MECHANISMS = (
        "independent_readout",
        "temporally_correlated_readout",
        "detector_dropout",
        "observation_delay",
    )

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        super().__init__(spec=spec, seed=seed)
        if self.output_domain != "measurement_fault_parameters":
            raise ValueError(
                f"MeasurementNoiseGenerator requires output_domain='measurement_fault_parameters', "
                f"got '{self.output_domain}'"
            )

        cfg_rel = spec.parameters.get(
            "config_path", "configs/noise_framework/stage5_measurement_imperfections.yaml"
        )
        map_rel = spec.parameters.get(
            "mapping_path", "configs/noise_framework/measurement_parameter_mapping.yaml"
        )
        cfg_path = Path(cfg_rel) if Path(cfg_rel).is_absolute() else (REPO_ROOT / cfg_rel)
        map_path = Path(map_rel) if Path(map_rel).is_absolute() else (REPO_ROOT / map_rel)

        self.stage5_config = load_yaml_validated(
            cfg_path, required_keys=("schema_version", "stage", "enabled", "generator")
        )
        if "regimes" in self.stage5_config:
            raise ValueError(
                "stage5_measurement_imperfections.yaml must not contain hard-coded default 'regimes'"
            )

        gen_cfg = self.stage5_config["generator"]
        supported = set(gen_cfg.get("supported_mechanisms", []))
        for req_m in self.REQUIRED_MECHANISMS:
            if req_m not in supported:
                raise ValueError(
                    f"Missing required mechanism '{req_m}' in {cfg_path}"
                )

        self.mapping_config = load_yaml_validated(
            map_path, required_keys=("schema_version", "generator_type", "mappings", "validation")
        )
        if self.mapping_config.get("generator_type") != "measurement_noise":
            raise ValueError(
                f"Mapping generator_type must be 'measurement_noise', got {self.mapping_config.get('generator_type')}"
            )
        self._validate_mapping_schema()

    def _validate_mapping_schema(self) -> None:
        mappings = self.mapping_config["mappings"]
        for req_m in self.REQUIRED_MECHANISMS:
            if req_m not in mappings:
                raise ValueError(f"Missing required mapping for '{req_m}'")
            entry = mappings[req_m]
            if "check_sampling_mode" not in entry or "basis_sampling_mode" not in entry:
                raise ValueError(
                    f"Mapping '{req_m}' must explicitly specify 'check_sampling_mode' and 'basis_sampling_mode'"
                )
            if entry["check_sampling_mode"] not in ("independent_per_check", "shared_across_checks"):
                raise ValueError(f"Invalid check_sampling_mode in '{req_m}'")
            if entry["basis_sampling_mode"] not in ("independent_x_and_z", "shared_x_and_z"):
                raise ValueError(f"Invalid basis_sampling_mode in '{req_m}'")

    def _validate_profile_compatibility(self, profile: NoiseProfile) -> None:
        super()._validate_profile_compatibility(profile)
        mappings = self.mapping_config["mappings"]
        prof_var = profile.support.get("variable")
        for m_name, entry in mappings.items():
            req_var = entry.get("source_variable", "fault_probability")
            if prof_var != req_var:
                raise ValueError(
                    f"Profile '{profile.profile_id}' variable '{prof_var}' is incompatible "
                    f"with mapping '{m_name}' requiring '{req_var}'"
                )

    def _sample_independent_readout(self, prof: NoiseProfile) -> tuple[np.ndarray, np.ndarray]:
        m_cfg = self.mapping_config["mappings"]["independent_readout"]
        dim_x = int(m_cfg["dimensions"]["x_checks"])
        dim_z = int(m_cfg["dimensions"]["z_checks"])
        scale_x = float(m_cfg.get("scale_factor_x", 1.0))
        scale_z = float(m_cfg.get("scale_factor_z", 1.0))
        check_mode = m_cfg["check_sampling_mode"]
        basis_mode = m_cfg["basis_sampling_mode"]

        if check_mode == "shared_across_checks":
            base_x = np.full(dim_x, float(prof.sample_values(self.rng, size=1)[0]), dtype=np.float64)
            if basis_mode == "shared_x_and_z":
                base_z = base_x.copy()
            else:
                base_z = np.full(
                    dim_z, float(prof.sample_values(self.rng, size=1)[0]), dtype=np.float64
                )
        else:
            base_x = prof.sample_values(self.rng, size=dim_x).astype(np.float64)
            if basis_mode == "shared_x_and_z":
                base_z = base_x.copy()
            else:
                base_z = prof.sample_values(self.rng, size=dim_z).astype(np.float64)

        val_cfg = self.mapping_config["validation"]
        p_min = float(val_cfg.get("probability_min", 0.0))
        p_max = float(val_cfg.get("probability_max", 1.0))
        q_x = np.clip(base_x * scale_x, p_min, p_max)
        q_z = np.clip(base_z * scale_z, p_min, p_max)
        return q_x, q_z

    def _sample_temporal_sticky(self, prof: NoiseProfile) -> float:
        m_cfg = self.mapping_config["mappings"]["temporally_correlated_readout"]
        draw = float(prof.sample_values(self.rng, size=1)[0])
        norm_pos = (draw - prof.min_val) / max(prof.max_val - prof.min_val, 1e-12)
        sticky_anchor = float(prof.parameters.get(m_cfg.get("use_profile_parameter", "sticky_scale"), 0.08))
        scale = float(m_cfg.get("scale_factor", 1.0))
        sticky = sticky_anchor * (0.5 + 0.5 * norm_pos) + scale * draw
        val_cfg = self.mapping_config["validation"]
        return float(
            np.clip(
                sticky,
                float(val_cfg.get("probability_min", 0.0)),
                float(val_cfg.get("probability_max", 1.0)),
            )
        )

    def _sample_detector_dropout(self, prof: NoiseProfile) -> float:
        m_cfg = self.mapping_config["mappings"]["detector_dropout"]
        draw = float(prof.sample_values(self.rng, size=1)[0])
        dropout_anchor = float(
            prof.parameters.get(m_cfg.get("use_profile_parameter", "dropout_scale"), 0.001)
        )
        scale = float(m_cfg.get("scale_factor", 0.25))
        dropout = dropout_anchor + scale * draw
        val_cfg = self.mapping_config["validation"]
        return float(
            np.clip(
                dropout,
                float(val_cfg.get("probability_min", 0.0)),
                float(val_cfg.get("probability_max", 1.0)),
            )
        )

    def _sample_observation_delay(self, prof: NoiseProfile) -> int:
        m_cfg = self.mapping_config["mappings"]["observation_delay"]
        draw = float(prof.sample_values(self.rng, size=1)[0])
        rule = str(m_cfg.get("quantization_rule", "threshold_quantiles"))
        if rule != "threshold_quantiles":
            raise ValueError(f"Unsupported delay quantization_rule '{rule}'")
        thresholds = [float(t) for t in m_cfg.get("thresholds", [0.0085, 0.0160])]
        int_values = [int(v) for v in m_cfg.get("integer_values", [0, 1, 2])]
        if len(int_values) != len(thresholds) + 1:
            raise ValueError("integer_values must have length len(thresholds) + 1")

        chosen = int_values[0]
        for thr, val in zip(thresholds, int_values[1:]):
            if draw >= thr:
                chosen = int(val)

        val_cfg = self.mapping_config["validation"]
        d_min = int(val_cfg.get("delay_min", 0))
        d_max = int(val_cfg.get("delay_max", 5))
        return int(max(d_min, min(d_max, chosen)))

    def sample(self, timestamp: float, sample_id: int) -> MeasurementNoiseParameters:
        window_id = self._check_ready_and_timestamp(timestamp, sample_id)
        prof = self.current_profile
        assert prof is not None

        q_x, q_z = self._sample_independent_readout(prof)
        sticky = self._sample_temporal_sticky(prof)
        dropout = self._sample_detector_dropout(prof)
        delay = self._sample_observation_delay(prof)

        self.samples_generated += 1
        self.last_timestamp = float(timestamp)

        return MeasurementNoiseParameters(
            q_base_x=q_x,
            q_base_z=q_z,
            temporal_sticky_prob=sticky,
            detector_dropout_prob=dropout,
            observation_delay_samples=delay,
            generator_id=self.generator_id,
            profile_id=prof.profile_id,
            sample_id=int(sample_id),
            timestamp=float(timestamp),
            window_id=window_id,
        )

    def sample_parameters(
        self,
        sample_id: int = 0,
        timestamp: float = 0.0,
        window_id: Optional[int] = None,
    ) -> MeasurementNoiseParameters:
        """Convenience alias returning `MeasurementNoiseParameters` at `(timestamp, sample_id)`."""
        return self.sample(timestamp=timestamp, sample_id=sample_id)

