"""Step 13 — Controlled Measurement Baseline Regimes Tests."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.stimulus.noise_framework.config_schema import GeneratorSpec
from simulation.stimulus.noise_framework.generators.measurement_noise_generator import (
    MeasurementNoiseGenerator,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.syndrome.measurement_model import MeasurementModel
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    load_all_measurement_baselines,
    load_measurement_baseline,
)


def test_all_three_baselines_load():
    baselines = load_all_measurement_baselines()
    assert set(baselines.keys()) == {
        "ideal_noiseless",
        "moderate_readout_noise",
        "correlated_readout_and_dropout",
    }


def test_exact_numerical_values_are_preserved():
    b_ideal = load_measurement_baseline("ideal_noiseless")
    np.testing.assert_allclose(b_ideal.q_base_x, [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(b_ideal.q_base_z, [0.0, 0.0, 0.0, 0.0])
    assert b_ideal.temporal_sticky_prob == 0.0
    assert b_ideal.detector_dropout_prob == 0.0
    assert b_ideal.observation_delay_samples == 0

    b_mod = load_measurement_baseline("moderate_readout_noise")
    np.testing.assert_allclose(b_mod.q_base_x, [0.004, 0.005, 0.005, 0.004])
    np.testing.assert_allclose(b_mod.q_base_z, [0.004, 0.004, 0.004, 0.005])
    assert b_mod.temporal_sticky_prob == pytest.approx(0.08)
    assert b_mod.detector_dropout_prob == pytest.approx(0.001)
    assert b_mod.observation_delay_samples == 0

    b_corr = load_measurement_baseline("correlated_readout_and_dropout")
    np.testing.assert_allclose(b_corr.q_base_x, [0.010, 0.014, 0.014, 0.010])
    np.testing.assert_allclose(b_corr.q_base_z, [0.010, 0.012, 0.012, 0.010])
    assert b_corr.temporal_sticky_prob == pytest.approx(0.22)
    assert b_corr.detector_dropout_prob == pytest.approx(0.005)
    assert b_corr.observation_delay_samples == 1


def test_explicit_baseline_selection_works():
    selected = load_measurement_baseline(
        "correlated_readout_and_dropout", sample_id=7, timestamp=0.7, window_id=0
    )
    assert selected.metadata["baseline_regime"] == "correlated_readout_and_dropout"
    assert selected.metadata["explicit_baseline_mode"] is True
    assert selected.sample_id == 7
    assert selected.timestamp == pytest.approx(0.7)


def test_unknown_baseline_names_are_rejected():
    with pytest.raises(KeyError, match="Unknown baseline regime"):
        load_measurement_baseline("nonexistent_regime")
    with pytest.raises(ValueError, match="Explicit non-empty baseline"):
        load_measurement_baseline("")


def test_dynamic_mode_does_not_load_baseline_regimes():
    registry = PDRegistry().load()
    spec = GeneratorSpec(
        generator_id="measurement_noise_01",
        generator_type="measurement_noise",
        output_domain="measurement_fault_parameters",
        seed=501,
    )
    gen = MeasurementNoiseGenerator(spec)
    gen.assign_profile("PD_08", window_id=0, registry=registry)
    params = gen.sample_parameters(sample_id=0, timestamp=0.0, window_id=0)

    assert "baseline_regime" not in params.metadata
    assert not params.profile_id.startswith("BASELINE_")
    assert params.profile_id == "PD_08"


def test_baseline_output_is_reproducible():
    params_seq = [
        load_measurement_baseline(
            "correlated_readout_and_dropout", sample_id=k, timestamp=k * 0.1, window_id=0
        )
        for k in range(50)
    ]
    ideal_sx = np.zeros((50, 4), dtype=np.uint8)
    ideal_sz = np.ones((50, 4), dtype=np.uint8)

    model1 = MeasurementModel()
    out1 = model1.apply_trajectory(
        ideal_syndromes_x=ideal_sx,
        ideal_syndromes_z=ideal_sz,
        parameters_seq=params_seq,
        rng=np.random.default_rng(12345),
    )

    model2 = MeasurementModel()
    out2 = model2.apply_trajectory(
        ideal_syndromes_x=ideal_sx,
        ideal_syndromes_z=ideal_sz,
        parameters_seq=params_seq,
        rng=np.random.default_rng(12345),
    )

    np.testing.assert_array_equal(out1["observed_syndromes_x"], out2["observed_syndromes_x"])
    np.testing.assert_array_equal(out1["observed_syndromes_z"], out2["observed_syndromes_z"])
    np.testing.assert_array_equal(out1["validity_mask_x"], out2["validity_mask_x"])
    np.testing.assert_array_equal(out1["validity_mask_z"], out2["validity_mask_z"])
