"""Immutable 16-Profile Probability Distribution Registry (`pd_registry.py`).

Validates:
  - Exactly 16 unique profile IDs (`PD_01` .. `PD_16`).
  - Distribution parameters and probability bounds `[min, max] subset [0, 1]`.
  - Explicit variable domain and units on every profile.
  - Immutability after loading and deterministic SHA-256 configuration hash.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Optional

from simulation.stimulus.noise_framework.config_schema import (
    NoiseProfile,
    compute_canonical_config_hash,
    load_yaml_validated,
)


class PDRegistry:
    """Immutable registry of the 16 canonical Probability Distribution (PD) profiles."""

    EXPECTED_PROFILE_COUNT = 16
    DEFAULT_PATH = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "noise_framework"
        / "pd_profiles_16.yaml"
    )

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self._profiles: Dict[str, NoiseProfile] = {}
        self._config_hash: str = ""
        self._loaded: bool = False
        if path is not None:
            self.load(path)

    def load(self, path: Optional[str | Path] = None) -> "PDRegistry":
        if self._loaded:
            raise RuntimeError("PDRegistry is immutable once loaded; instantiate a new PDRegistry instead.")
        target_path = Path(path) if path is not None else self.DEFAULT_PATH
        raw = load_yaml_validated(target_path, required_keys=("schema_version", "profiles"))
        profiles_list = raw["profiles"]
        if not isinstance(profiles_list, list) or len(profiles_list) != self.EXPECTED_PROFILE_COUNT:
            raise ValueError(
                f"Expected exactly {self.EXPECTED_PROFILE_COUNT} profiles in {path}, "
                f"found {len(profiles_list) if isinstance(profiles_list, list) else 'non-list'}"
            )
        parsed: Dict[str, NoiseProfile] = {}
        for entry in profiles_list:
            if not isinstance(entry, dict):
                raise ValueError("Each profile entry must be a mapping")
            pid = str(entry.get("profile_id", ""))
            if pid in parsed:
                raise ValueError(f"Duplicate profile_id '{pid}' in {path}")
            prof = NoiseProfile(
                profile_id=pid,
                distribution_type=str(entry.get("distribution_type", "")),
                parameters=dict(MappingProxyType(dict(entry.get("parameters", {})))),
                support=dict(MappingProxyType(dict(entry.get("support", {})))),
                metadata=dict(MappingProxyType(dict(entry.get("metadata", {})))),
            )
            parsed[pid] = prof

        self._profiles = parsed
        self._config_hash = compute_canonical_config_hash(raw)
        self._loaded = True
        self.validate()
        return self

    def validate(self) -> bool:
        if not self._loaded or len(self._profiles) != self.EXPECTED_PROFILE_COUNT:
            raise ValueError("PDRegistry must contain 16 loaded profiles")
        expected_ids = [f"PD_{i:02d}" for i in range(1, self.EXPECTED_PROFILE_COUNT + 1)]
        for pid in expected_ids:
            if pid not in self._profiles:
                raise ValueError(f"Missing required canonical profile_id '{pid}'")
            prof = self._profiles[pid]
            # Validate moments are finite and within support
            m, v = prof.theoretical_moments()
            if not (prof.min_val <= m <= prof.max_val and v > 0.0):
                raise ValueError(f"Invalid theoretical moments ({m}, {v}) for {pid}")
        return True

    def get(self, profile_id: str) -> NoiseProfile:
        if not self._loaded:
            self.load()
        if profile_id not in self._profiles:
            raise KeyError(f"Unknown profile_id '{profile_id}'")
        return self._profiles[profile_id]

    def list_profiles(self) -> List[str]:
        if not self._loaded:
            self.load()
        return list(self._profiles.keys())

    @property
    def config_hash(self) -> str:
        return self._config_hash
