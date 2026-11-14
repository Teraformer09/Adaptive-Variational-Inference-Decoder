"""Framework Test 6: CSS Syndrome Generation (`test_syndrome_generation.py`)."""

from __future__ import annotations

import unittest
import numpy as np

from simulation.reference.code_model.css_code_builder import build_rotated_d3_code
from simulation.stimulus.noise_framework.syndrome.fault_realizer import FaultRealizer
from simulation.stimulus.noise_framework.syndrome.syndrome_encoder import SyndromeEncoder


class TestSyndromeGeneration(unittest.TestCase):
    def test_encoded_syndromes_equal_hx_ez_and_hz_ex_mod2(self) -> None:
        code = build_rotated_d3_code()
        realizer = FaultRealizer()
        encoder = SyndromeEncoder()
        rng = np.random.default_rng(606)

        theta_z = np.full(21, 0.02, dtype=np.float64)
        theta_x = np.full(21, 0.015, dtype=np.float64)
        err_z, _ = realizer.realize_z_faults(theta_z, rng=rng, n_rounds=256)
        err_x, _ = realizer.realize_x_faults(theta_x, rng=rng, n_rounds=256)

        sx, sz = encoder.encode(x_errors=err_x, z_errors=err_z)
        ref_sx = (err_z @ code.hx.T) % 2
        ref_sz = (err_x @ code.hz.T) % 2

        self.assertTrue(np.array_equal(sx, ref_sx))
        self.assertTrue(np.array_equal(sz, ref_sz))


if __name__ == "__main__":
    unittest.main()
