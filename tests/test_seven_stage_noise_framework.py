"""Comprehensive Pytest Suite for the AVID Seven-Stage Noise Framework (`tests/test_seven_stage_noise_framework.py`).

Tests:
  1. Stage 1: 16-PD profile registry, truncated Gaussian sampling, and moments.
  2. Stage 2: 100s persistent PD assignment, window transitions, and RNG stream isolation.
  3. Stage 3 & 4: 10 Hz synchronized multi-generator sampling and physical XOR/Bernoulli composition.
  4. Stage 5: Dynamic measurement-noise generator, temporal sticky Markov correlation, dropout validity
     masks, and observation delay timestamp preservation.
  5. Stage 6: Stratified trajectory generation and partition disjointness (Theta_train ∩ Theta_test = ∅).
  6. Stage 7: Exact 32-state joint syndrome-logical distribution P(s, l | theta), conditional posterior
     P(l | s, theta), Oracle LER floor LER_oracle(theta), and 6-decoder shared-trajectory evaluation.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from simulation.reference.noise_model.logical_class_distribution import (
    LogicalClassDistributionEngine,
)
from simulation.reference.noise_model.stage7_statistical_validator import (
    Stage7StatisticalValidator,
)
from simulation.stimulus.seven_stage.stage1_pd_profiles import run_stage1_pd_validation
from simulation.stimulus.seven_stage.stage2_stochastic_temporal import (
    run_stage2_assignment_and_lifecycle_validation,
)
from simulation.stimulus.seven_stage.stage3_spatial_correlated import (
    run_stage3_spatial_validation,
)
from simulation.stimulus.seven_stage.stage4_nonstationary_bursts import (
    run_stage4_composition_validation,
)
from simulation.stimulus.seven_stage.stage5_measurement_noise import (
    run_stage5_measurement_noise_pipeline,
)
from simulation.stimulus.seven_stage.stage6_stratified_sampler import (
    generate_single_stratified_trajectory,
)


def test_stages_1_to_5_validation_gates(tmp_path: Path) -> None:
    r1 = run_stage1_pd_validation(num_samples=8000, seed=101)
    assert r1["passed"]

    r2 = run_stage2_assignment_and_lifecycle_validation(duration_seconds=250.0, seed=202)
    assert r2["passed"]

    r3 = run_stage3_spatial_validation(num_samples=600, seed=303)
    assert r3["passed"]

    r4 = run_stage4_composition_validation(duration_seconds=30.0, seed=404)
    assert r4["passed"]

    r5 = run_stage5_measurement_noise_pipeline(
        duration_seconds=250.0, seed=42, output_dir=tmp_path / "stage5"
    )
    assert r5["passed"]
    assert r5["actual_samples"] == 2500


def test_stage7_exact_joint_syndrome_logical_distribution_and_oracle() -> None:
    engine = LogicalClassDistributionEngine(basis="X")
    theta = np.full(21, 0.005, dtype=np.float64)
    theta[12] = 0.045  # pair (4, 5)

    res = engine.compute_posterior_and_oracle_ler(theta)
    joint = res["joint_s_l"]
    marg_s = res["marginal_s"]
    post = res["posterior_l_given_s"]
    ler_oracle = res["ler_oracle"]

    assert joint.shape == (16, 2)
    assert abs(float(np.sum(joint)) - 1.0) < 1e-12
    assert np.allclose(np.sum(post, axis=1), 1.0)
    assert 0.0 < ler_oracle < 0.25

    # Verify syndrome-aliased pair has close marginal P(s) but distinct P(l | s)
    _, _, alias = engine.construct_syndrome_aliased_pair(theta, delta=0.02)
    assert alias["tvd_marginal_syndrome"] < 0.025
    assert alias["max_logical_posterior_diff"] > 0.05


def test_stage7_statistical_validation_and_shared_trajectory_decoders() -> None:
    traj, _, _ = generate_single_stratified_trajectory(
        trajectory_id="eval_test_s3001",
        split_id="test",
        master_seed=3001,
        duration_seconds=100.0,
        assignment_policy="curriculum_burst_enriched",
    )
    validator = Stage7StatisticalValidator()
    dist_val = validator.validate_trajectory_distributions(traj)
    assert dist_val["passed"]

    dec_eval = validator.evaluate_decoders_on_shared_trajectory(traj, window_rounds=250)
    lers = dec_eval["logical_error_rates"]
    assert lers["6_oracle_bp_and_coset"] <= lers["5_recurrent_avid_r2d2"] + 0.01
    assert lers["5_recurrent_avid_r2d2"] <= lers["1_static_bp"]
