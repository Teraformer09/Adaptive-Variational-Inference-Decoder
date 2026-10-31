"""Data generation, schema validation, and R2D2 replay slicing for the AVID World Model."""

from world_model.data.generator import (
    DeviceTopology,
    QuantumNoiseTrajectoryGenerator,
    build_surface17_topology,
)
from world_model.data.replay_dataset import R2D2TrajectoryReplayDataset, trajectory_to_tensors
from world_model.data.schema_validator import (
    TrajectorySchemaValidator,
    strip_latent_truth,
    verify_zero_leakage_splits,
)

__all__ = [
    "DeviceTopology",
    "QuantumNoiseTrajectoryGenerator",
    "build_surface17_topology",
    "R2D2TrajectoryReplayDataset",
    "trajectory_to_tensors",
    "TrajectorySchemaValidator",
    "strip_latent_truth",
    "verify_zero_leakage_splits",
]
