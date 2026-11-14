"""Framework Test 5: Pauli XOR Cancellation & Probability Composition (`test_noise_composition.py`)."""

from __future__ import annotations

import unittest
import numpy as np

from simulation.stimulus.noise_framework.composition.pauli_composition import (
    compose_pauli_masks_xor,
)
from simulation.stimulus.noise_framework.composition.probability_composition import (
    compose_independent_bernoulli,
)
from simulation.stimulus.seven_stage.stage4_nonstationary_bursts import (
    run_stage4_composition_validation,
)


class TestNoiseComposition(unittest.TestCase):
    def test_pauli_xor_cancellation_and_probability_union(self) -> None:
        # Test D: Pauli cancellation Z XOR Z = I
        z_mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        composed = compose_pauli_masks_xor([z_mask, z_mask])
        self.assertTrue(np.all(composed == 0))

        # Probability union 1 - (1 - p1)(1 - p2)
        p1 = np.array([0.1, 0.2, 0.0], dtype=np.float64)
        p2 = np.array([0.2, 0.2, 0.5], dtype=np.float64)
        p_comb = compose_independent_bernoulli([p1, p2])
        self.assertTrue(np.allclose(p_comb, np.array([0.28, 0.36, 0.50])))

        rep = run_stage4_composition_validation(duration_seconds=20.0, seed=404)
        self.assertTrue(rep["passed"])


if __name__ == "__main__":
    unittest.main()
