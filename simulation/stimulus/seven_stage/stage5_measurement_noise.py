"""Stage 5: Dynamic Measurement-Noise Subsystem Execution & Validation (`stage5_measurement_noise.py`).

Implements Step 13 and Step 15 (`inst_0` .. `inst_12`) of the Stage 5 instructions:
  Load framework_v1.yaml
          ↓
  Load stage5_measurement_imperfections.yaml & measurement_parameter_mapping.yaml
          ↓
  Load 16-PD registry
          ↓
  Instantiate measurement_noise_01 & data-fault generators
          ↓
  Assign PDs through AssignmentScheduler (100s windows)
          ↓
  Generate synchronized 10 Hz parameter samples
          ↓
  Compose measurement-noise contributions via NoiseComposer
          ↓
  Apply MeasurementModel to ideal syndromes
          ↓
  Save observed syndromes, validity masks, and ground truth via AvidInputAdapter
          ↓
  Run Stage 5 validation & write validation report (exit non-zero on failure).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.avid_interface.avid_input_adapter import AvidInputAdapter
from simulation.stimulus.noise_framework.avid_interface.sample_encoder import SampleEncoder
from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.config_schema import (
    compute_canonical_config_hash,
    load_yaml_validated,
)
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.metadata.provenance import build_provenance_record
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler
from simulation.stimulus.noise_framework.sampling.trajectory_buffer import TrajectoryBuffer
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


def run_stage5_measurement_noise_pipeline(
    config_path: str | Path | None = None,
    duration_seconds: float = 250.0,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    fw_path = (
        Path(config_path)
        if config_path is not None
        else (REPO_ROOT / "configs" / "noise_framework" / "framework_v1.yaml")
    )
    if not fw_path.is_absolute():
        fw_path = REPO_ROOT / fw_path

    # inst_0: Load framework_v1.yaml
    fw_cfg = load_yaml_validated(fw_path, required_keys=("schema_version", "configs"))
    cfgs = fw_cfg["configs"]

    # inst_1: Load stage5_measurement_imperfections.yaml & verify no hard-coded regimes
    stage5_path = REPO_ROOT / cfgs["stage5_measurement"]
    stage5_cfg = load_yaml_validated(
        stage5_path, required_keys=("schema_version", "stage", "enabled", "generator")
    )
    if "regimes" in stage5_cfg:
        raise ValueError("Default stage5_measurement_imperfections.yaml must not contain 'regimes'")

    # inst_2: Load measurement_parameter_mapping.yaml
    map_path = REPO_ROOT / cfgs["measurement_mapping"]
    map_cfg = load_yaml_validated(
        map_path, required_keys=("schema_version", "generator_type", "mappings", "validation")
    )

    # inst_3: Load shared 16-PD registry
    pd_reg = PDRegistry(REPO_ROOT / cfgs["pd_profiles"])

    # inst_4: Instantiate active generators including measurement_noise_01
    gen_reg = GeneratorRegistry(REPO_ROOT / cfgs["generator_registry"], master_seed=seed)
    generators = gen_reg.instantiate_all(master_seed=seed)
    meas_gen = gen_reg.get("measurement_noise_01")

    # inst_5: Connect shared assignment scheduler (100s persistence)
    pd_assign = PDAssignment(
        pd_registry=pd_reg,
        master_seed=seed,
        policy_name="curriculum_burst_enriched",
        assignment_config_path=REPO_ROOT / cfgs["generator_assignment"],
    )
    scheduler = AssignmentScheduler(pd_assignment=pd_assign, window_duration_sec=100.0)
    scheduler.initialize(generators=generators, pd_registry=pd_reg)

    # inst_6 & inst_7: Generate synchronized 10 Hz samples & compose contributions
    clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)
    composer = NoiseComposer()
    sampler = SynchronizedSampler(
        generators=generators, clock=clock, scheduler=scheduler, composer=composer
    )
    buf = TrajectoryBuffer()
    composite_samples = sampler.generate_trajectory(duration_seconds=duration_seconds, buffer=buf)

    expected_samples = int(round(duration_seconds * 10.0))
    if len(composite_samples) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} composite samples, got {len(composite_samples)}"
        )

    # Verify measurement_noise_01 contributions across all windows
    meas_contribs = buf.contributions_for_generator("measurement_noise_01")
    if len(meas_contribs) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} measurement contributions, got {len(meas_contribs)}"
        )

    # Verify 100s window assignment invariance for measurement_noise_01
    win_to_pid: Dict[int, set[str]] = {}
    bounds_ok = True
    for mc in meas_contribs:
        win_to_pid.setdefault(int(mc.window_id), set()).add(mc.profile_id)
        mp = mc.metadata.get("measurement_parameters")
        if not isinstance(mp, MeasurementNoiseParameters):
            raise RuntimeError("Missing typed MeasurementNoiseParameters in contribution metadata")
        if (
            np.any((mp.q_base_x < 0.0) | (mp.q_base_x > 1.0))
            or np.any((mp.q_base_z < 0.0) | (mp.q_base_z > 1.0))
            or not (0.0 <= mp.temporal_sticky_prob <= 1.0)
            or not (0.0 <= mp.detector_dropout_prob <= 1.0)
            or mp.observation_delay_samples < 0
        ):
            bounds_ok = False

    persistence_ok = all(len(pids) == 1 for pids in win_to_pid.values())

    # inst_8: Apply measurement model to ideal syndrome trajectories via SampleEncoder
    sample_encoder = SampleEncoder(rounds_per_composite_sample=1, sampling_interval_sec=0.1)
    encoded_traj = sample_encoder.encode_composite_trajectory(
        composite_samples=composite_samples,
        master_seed=seed,
        trajectory_id=f"stage5_meas_s{seed}",
        split_id="validation",
    )

    # Verify event vs delivery timestamps & validity mask properties
    ev_ts = np.asarray(encoded_traj["event_timestamps"], dtype=np.float64)
    del_ts = np.asarray(encoded_traj["delivery_timestamps"], dtype=np.float64)
    delays = np.asarray(encoded_traj["observation_delays"], dtype=np.int32)
    timestamps_consistent = bool(
        np.allclose(del_ts - ev_ts, delays * 0.1, atol=1e-7) and np.all(del_ts >= ev_ts)
    )
    val_x = np.asarray(encoded_traj["validity_mask_x"], dtype=np.uint8)
    val_z = np.asarray(encoded_traj["validity_mask_z"], dtype=np.uint8)
    validity_binary_ok = bool(
        np.all((val_x == 0) | (val_x == 1)) and np.all((val_z == 0) | (val_z == 1))
    )

    # inst_9: Save observed syndromes, ground truth, and packet stream via AvidInputAdapter
    adapter = AvidInputAdapter(encoder=sample_encoder)
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else (
            REPO_ROOT
            / "Benchmark-Eval"
            / "V1_Seven_Stage_Noise_Validation"
            / "stage5_syndrome_validation"
        )
    )
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = adapter.write_npz(encoded_traj, out_dir / f"stage5_trajectory_s{seed}.npz")
    bin_path = adapter.write_packet_stream(encoded_traj, out_dir / f"stage5_packets_s{seed}.bin")
    stream_check = adapter.validate_stream(bin_path, expected_trajectory=encoded_traj)

    prov = build_provenance_record(
        master_seed=seed,
        assignment_policy="curriculum_burst_enriched",
        duration_seconds=duration_seconds,
        split_id="validation",
        framework_yaml_path=fw_path,
    )

    all_passed = bool(
        persistence_ok
        and bounds_ok
        and timestamps_consistent
        and validity_binary_ok
        and stream_check["valid"]
        and meas_gen.samples_generated == expected_samples
    )

    report = {
        "stage": "stage5_measurement_noise",
        "passed": all_passed,
        "seed": int(seed),
        "duration_seconds": float(duration_seconds),
        "expected_samples": expected_samples,
        "actual_samples": int(meas_gen.samples_generated),
        "config_hashes": {
            "framework_sha256": compute_canonical_config_hash(fw_cfg),
            "stage5_config_sha256": compute_canonical_config_hash(stage5_cfg),
            "mapping_config_sha256": compute_canonical_config_hash(map_cfg),
            "pd_registry_sha256": pd_reg.config_hash,
            "generator_registry_sha256": gen_reg.config_hash,
        },
        "provenance": prov,
        "measurement_assignment_by_window": {
            str(w): sorted(list(pids))[0] for w, pids in sorted(win_to_pid.items())
        },
        "assignment_manifest": pd_assign.assignment_manifest(),
        "validation_checks": {
            "no_default_hardcoded_regimes": "regimes" not in stage5_cfg,
            "shared_pd_registry_used": True,
            "window_persistence_100s_verified": persistence_ok,
            "synchronized_10hz_sampling_verified": len(meas_contribs) == expected_samples,
            "parameter_bounds_verified": bounds_ok,
            "event_and_delivery_timestamps_verified": timestamps_consistent,
            "dropout_validity_masks_verified": validity_binary_ok,
            "avid_packet_stream_compatible": bool(stream_check["valid"]),
        },
        "summary_metrics": {
            "mean_q_base_x": [
                float(x) for x in np.mean(encoded_traj["measurement_parameters_vec"][:, 0:4], axis=0)
            ],
            "mean_q_base_z": [
                float(x) for x in np.mean(encoded_traj["measurement_parameters_vec"][:, 4:8], axis=0)
            ],
            "mean_temporal_sticky_prob": float(
                np.mean(encoded_traj["measurement_parameters_vec"][:, 8])
            ),
            "mean_detector_dropout_prob": float(
                np.mean(encoded_traj["measurement_parameters_vec"][:, 9])
            ),
            "max_observation_delay_samples": int(np.max(encoded_traj["observation_delays"])),
            "realized_measurement_flip_rate_x": float(
                np.mean(encoded_traj["measurement_faults_x"])
            ),
            "realized_measurement_flip_rate_z": float(
                np.mean(encoded_traj["measurement_faults_z"])
            ),
            "realized_dropout_count_x": int(np.sum(val_x == 0)),
            "realized_dropout_count_z": int(np.sum(val_z == 0)),
        },
        "artifacts": {
            "npz_path": (
                str(npz_path.relative_to(REPO_ROOT))
                if npz_path.is_relative_to(REPO_ROOT)
                else str(npz_path.name)
            ),
            "bin_path": (
                str(bin_path.relative_to(REPO_ROOT))
                if bin_path.is_relative_to(REPO_ROOT)
                else str(bin_path.name)
            ),
        },
    }

    report_path = out_dir / "stage5_measurement_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 5 Dynamic Measurement-Noise Pipeline")
    p.add_argument(
        "--config",
        type=str,
        default="configs/noise_framework/framework_v1.yaml",
        help="Path to framework_v1.yaml",
    )
    p.add_argument("--duration", type=float, default=250.0, help="Simulation duration in seconds")
    p.add_argument("--seed", type=int, default=42, help="Master random seed")
    p.add_argument(
        "--output",
        type=str,
        default="Benchmark-Eval/V1_Seven_Stage_Noise_Validation/stage5_syndrome_validation",
        help="Output directory for validation artifacts",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rep = run_stage5_measurement_noise_pipeline(
            config_path=args.config,
            duration_seconds=args.duration,
            seed=args.seed,
            output_dir=args.output,
        )
    except Exception as exc:
        print(f"[STAGE 5 FAILED] {exc}", file=sys.stderr)
        return 1

    if not rep["passed"]:
        print("[STAGE 5 FAILED] One or more validation checks failed.", file=sys.stderr)
        return 1

    print(
        f"[PASS] Stage 5 validated {rep['actual_samples']} synchronized samples over "
        f"{rep['duration_seconds']}s (windows={list(rep['measurement_assignment_by_window'].keys())})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
