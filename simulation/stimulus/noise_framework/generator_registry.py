"""Active Noise Generator Instance Registry (`generator_registry.py`).

Responsibilities:
  - Register `GeneratorSpec` instances.
  - Reject duplicate generator IDs, unknown generator types, unsupported output domains,
    invalid composition groups, or missing seeds.
  - Instantiate active generator instances with isolated deterministic RNG streams
    `H(master_seed, generator_id, spec.seed)`.
  - Never assign PDs directly (that responsibility belongs to `PDAssignment` and `AssignmentScheduler`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

from simulation.stimulus.noise_framework.config_schema import (
    VALID_OUTPUT_DOMAINS,
    GeneratorSpec,
    compute_canonical_config_hash,
    derive_deterministic_seed,
    load_yaml_validated,
)
from simulation.stimulus.noise_framework.generators.base_generator import BaseNoiseGenerator
from simulation.stimulus.noise_framework.generators.burst_generator import BurstNoiseGenerator
from simulation.stimulus.noise_framework.generators.gaussian_generator import GaussianNoiseGenerator
from simulation.stimulus.noise_framework.generators.markov_generator import MarkovNoiseGenerator
from simulation.stimulus.noise_framework.generators.measurement_noise_generator import (
    MeasurementNoiseGenerator,
)
from simulation.stimulus.noise_framework.generators.ou_generator import OUNoiseGenerator
from simulation.stimulus.noise_framework.generators.spatial_correlated_generator import (
    SpatialCorrelatedGenerator,
)
from simulation.stimulus.noise_framework.generators.telegraph_generator import (
    TelegraphNoiseGenerator,
)


GENERATOR_TYPE_MAP: Dict[str, Type[BaseNoiseGenerator]] = {
    "gaussian": GaussianNoiseGenerator,
    "markov": MarkovNoiseGenerator,
    "ou": OUNoiseGenerator,
    "telegraph": TelegraphNoiseGenerator,
    "spatial_correlated": SpatialCorrelatedGenerator,
    "burst": BurstNoiseGenerator,
    "measurement_noise": MeasurementNoiseGenerator,
}

VALID_COMPOSITION_GROUPS = {
    "data_faults",
    "data_faults_z",
    "data_faults_x",
    "measurement_faults",
}


class GeneratorRegistry:
    """Registry managing specifications and instantiated active noise generators."""

    def __init__(
        self,
        path: Optional[str | Path] = None,
        master_seed: int = 42,
    ) -> None:
        self.master_seed = int(master_seed)
        self._specs: Dict[str, GeneratorSpec] = {}
        self._instances: Dict[str, BaseNoiseGenerator] = {}
        self._config_hash: str = ""
        if path is not None:
            self.load(path)

    def register(self, spec: GeneratorSpec) -> None:
        if not isinstance(spec, GeneratorSpec):
            raise TypeError(f"Expected GeneratorSpec, got {type(spec)}")
        if spec.generator_id in self._specs:
            raise ValueError(f"Duplicate generator_id '{spec.generator_id}'")
        if spec.generator_type not in GENERATOR_TYPE_MAP:
            raise ValueError(
                f"Unknown generator_type '{spec.generator_type}' for '{spec.generator_id}'"
            )
        if spec.output_domain not in VALID_OUTPUT_DOMAINS:
            raise ValueError(
                f"Unsupported output_domain '{spec.output_domain}' for '{spec.generator_id}'"
            )
        if spec.composition_group not in VALID_COMPOSITION_GROUPS:
            raise ValueError(
                f"Invalid composition_group '{spec.composition_group}' for '{spec.generator_id}'"
            )
        self._specs[spec.generator_id] = spec

    def load(self, path: str | Path) -> "GeneratorRegistry":
        raw = load_yaml_validated(path, required_keys=("schema_version", "generators"))
        gen_list = raw["generators"]
        if not isinstance(gen_list, list) or len(gen_list) == 0:
            raise ValueError(f"Generator registry {path} must define a non-empty 'generators' list")
        for entry in gen_list:
            if not isinstance(entry, dict):
                raise ValueError("Generator entry must be a mapping")
            if "seed" not in entry or entry["seed"] is None:
                raise ValueError(
                    f"Generator '{entry.get('generator_id')}' is missing required 'seed'"
                )
            shape_raw = entry.get("output_shape", [21])
            spec = GeneratorSpec(
                generator_id=str(entry.get("generator_id", "")),
                generator_type=str(entry.get("generator_type", "")),
                enabled=bool(entry.get("enabled", True)),
                seed=int(entry["seed"]),
                output_domain=str(entry.get("output_domain", "")),
                output_shape=tuple(int(d) for d in shape_raw),
                composition_group=str(entry.get("composition_group", "")),
                parameters=dict(entry.get("parameters", {})),
            )
            self.register(spec)
        self._config_hash = compute_canonical_config_hash(raw)
        return self

    def instantiate_all(self, master_seed: Optional[int] = None) -> List[BaseNoiseGenerator]:
        if master_seed is not None:
            self.master_seed = int(master_seed)
        self._instances.clear()
        for gid, spec in self._specs.items():
            if not spec.enabled:
                continue
            cls = GENERATOR_TYPE_MAP[spec.generator_type]
            inst_seed = derive_deterministic_seed(self.master_seed, "instance", gid, spec.seed)
            self._instances[gid] = cls(spec=spec, seed=inst_seed)
        return list(self._instances.values())

    def get(self, generator_id: str) -> BaseNoiseGenerator:
        if generator_id not in self._instances:
            raise KeyError(f"Generator '{generator_id}' is not instantiated or not active")
        return self._instances[generator_id]

    def active_generators(self) -> List[BaseNoiseGenerator]:
        return list(self._instances.values())

    @property
    def config_hash(self) -> str:
        if not self._config_hash:
            self._config_hash = compute_canonical_config_hash(
                {gid: s.__dict__ for gid, s in self._specs.items()}
            )
        return self._config_hash
