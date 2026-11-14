"""Framework Test 3: Generator Lifecycle & Continuous State Across Windows (`test_generator_lifecycle.py`)."""

from __future__ import annotations

from pathlib import Path
import unittest
import numpy as np

from simulation.stimulus.noise_framework.config_schema import GeneratorSpec
from simulation.stimulus.noise_framework.generators.ou_generator import OUNoiseGenerator
from simulation.stimulus.noise_framework.pd_registry import PDRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestGeneratorLifecycle(unittest.TestCase):
    def test_ou_state_persists_across_100s_profile_reassignment(self) -> None:
        pd_reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
        spec = GeneratorSpec(
            generator_id="x_single_noise_01",
            generator_type="ou",
            enabled=True,
            seed=1002,
            output_domain="probability_parameter",
            output_shape=(21,),
            composition_group="data_faults_x",
            parameters={"basis": "X", "tau_sec": 10.0, "volatility_scale": 0.3},
        )
        gen = OUNoiseGenerator(spec=spec, seed=1002)
        gen.assign_profile(pd_reg.get("PD_01"), window_id=0)
        for sid in range(1000):
            gen.sample(timestamp=sid * 0.1, sample_id=sid)

        state_before_reassignment = gen.x_state.copy()
        self.assertIsNotNone(state_before_reassignment)

        # Reassign profile at 100s boundary: x_state must NOT be wiped to None
        gen.assign_profile(pd_reg.get("PD_14"), window_id=1)
        self.assertIsNotNone(gen.x_state)
        self.assertTrue(np.array_equal(gen.x_state, state_before_reassignment))

        out_next = gen.sample(timestamp=100.0, sample_id=1000)
        self.assertEqual(out_next.window_id, 1)
        self.assertEqual(out_next.profile_id, "PD_14")


if __name__ == "__main__":
    unittest.main()
