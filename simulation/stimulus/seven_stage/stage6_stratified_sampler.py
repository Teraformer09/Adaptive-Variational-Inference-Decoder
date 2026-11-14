"""Stage 6: Stratified Trajectory Generation & Disjoint Dataset Splits (`stage6_stratified_sampler.py`).

Constructs train, val, and test partitions at the configuration/trajectory level so that:
  Theta_train ∩ Theta_val = ∅, Theta_train ∩ Theta_test = ∅, Theta_val ∩ Theta_test = ∅
and writes full provenance manifests to `datasets/seven_stage_v1/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.avid_interface.avid_input_adapter import AvidInputAdapter
from simulation.stimulus.noise_framework.avid_interface.sample_encoder import SampleEncoder
from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.config_schema import load_yaml_validated
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.metadata.manifest_writer import ManifestWriter
from simulation.stimulus.noise_framework.metadata.provenance import build_provenance_record
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler
from simulation.stimulus.noise_framework.sampling.trajectory_buffer import TrajectoryBuffer


def generate_single_stratified_trajectory(
    trajectory_id: str,
    split_id: str,
    master_seed: int,
    duration_seconds: float,
    assignment_policy: str,
    allowed_profile_ids: List[str] | None = None,
    rounds_per_composite_sample: int = 1,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    pd_reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
    gen_reg = GeneratorRegistry(
        REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml",
        master_seed=master_seed,
    )
    generators = gen_reg.instantiate_all(master_seed=master_seed)

    pd_assign = PDAssignment(
        pd_registry=pd_reg,
        master_seed=master_seed,
        policy_name=assignment_policy,
        assignment_config_path=REPO_ROOT / "configs" / "noise_framework" / "generator_assignment.yaml",
        allowed_profile_ids=allowed_profile_ids,
    )
    scheduler = AssignmentScheduler(pd_assignment=pd_assign, window_duration_sec=100.0)
    scheduler.initialize(generators=generators, pd_registry=pd_reg)

    clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)
    composer = NoiseComposer()
    sampler = SynchronizedSampler(
        generators=generators, clock=clock, scheduler=scheduler, composer=composer
    )
    buf = TrajectoryBuffer()
    composite_samples = sampler.generate_trajectory(duration_seconds=duration_seconds, buffer=buf)

    encoder = SampleEncoder(rounds_per_composite_sample=rounds_per_composite_sample)
    encoded_traj = encoder.encode_composite_trajectory(
        composite_samples=composite_samples,
        master_seed=master_seed,
        trajectory_id=trajectory_id,
        split_id=split_id,
    )

    assign_manifest = pd_assign.assignment_manifest()
    prov = build_provenance_record(
        master_seed=master_seed,
        assignment_policy=assignment_policy,
        duration_seconds=duration_seconds,
        split_id=split_id,
        extra_theta={
            "allowed_profile_ids": sorted(allowed_profile_ids) if allowed_profile_ids else "ALL_16",
            "assignment_history": encoded_traj["assignment_history"],
        },
    )
    return encoded_traj, prov, assign_manifest


def run_stage6_stratified_dataset_generation(
    splits_yaml_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    validation_out_dir: str | Path | None = None,
) -> Dict[str, Any]:
    s_path = (
        Path(splits_yaml_path)
        if splits_yaml_path is not None
        else (REPO_ROOT / "configs" / "noise_framework" / "stage6_stratified_splits.yaml")
    )
    cfg = load_yaml_validated(s_path, required_keys=("schema_version", "partitions"))

    ds_root = (
        Path(dataset_root)
        if dataset_root is not None
        else (REPO_ROOT / "datasets" / "seven_stage_v1")
    )
    for sub in ("train", "val", "test", "manifests", "metadata"):
        (ds_root / sub).mkdir(parents=True, exist_ok=True)

    adapter = AvidInputAdapter()
    manifest_writer = ManifestWriter()

    for split_name, split_cfg in cfg["partitions"].items():
        n_traj = int(split_cfg["num_trajectories"])
        dur_sec = float(split_cfg["duration_seconds"])
        seed_start = int(split_cfg["seed_range"][0])
        policy = str(split_cfg["assignment_policy"])
        allowed_pids = list(split_cfg.get("allowed_profile_ids", []))

        for idx in range(n_traj):
            t_seed = seed_start + idx * 17
            traj_id = f"{split_name}_traj_{idx:02d}_s{t_seed}"
            encoded_traj, prov, assign_manifest = generate_single_stratified_trajectory(
                trajectory_id=traj_id,
                split_id=split_name,
                master_seed=t_seed,
                duration_seconds=dur_sec,
                assignment_policy=policy,
                allowed_profile_ids=allowed_pids if allowed_pids else None,
            )

            split_dir = ds_root / split_name
            npz_path = adapter.write_npz(
                encoded_traj, split_dir / f"{traj_id}.npz", extra_metadata={"provenance": prov}
            )
            bin_path = adapter.write_packet_stream(
                encoded_traj, split_dir / f"{traj_id}.bin"
            )
            manifest_writer.add_trajectory_record(
                encoded_traj=encoded_traj,
                npz_path=npz_path,
                provenance=prov,
                assignment_manifest=assign_manifest,
                bin_path=bin_path,
            )

    manifest_path = manifest_writer.write_manifest(
        ds_root / "manifests" / "seven_stage_v1_manifest.json"
    )
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Also write metadata summary
    meta_summary_path = ds_root / "metadata" / "partition_disjointness_report.json"
    meta_summary_path.write_text(
        json.dumps(manifest_data["partition_disjointness"], indent=2) + "\n",
        encoding="utf-8",
    )

    if validation_out_dir is not None:
        v_dir = Path(validation_out_dir)
        v_dir.mkdir(parents=True, exist_ok=True)
        (v_dir / "stage6_dataset_validation_report.json").write_text(
            json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8"
        )

    return manifest_data


if __name__ == "__main__":
    res = run_stage6_stratified_dataset_generation(
        validation_out_dir=REPO_ROOT
        / "Benchmark-Eval"
        / "V1_Seven_Stage_Noise_Validation"
        / "stage6_dataset_validation"
    )
    if not res["partition_disjointness"]["all_disjoint"]:
        raise SystemExit(1)
    print(
        f"[PASS] Stage 6 generated {res['total_trajectories']} stratified trajectories "
        f"({res['total_composite_samples']} composite samples) with disjoint splits."
    )
