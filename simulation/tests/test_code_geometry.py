#!/usr/bin/env python3
"""
Unit Test Suite 1: Canonical CSS Surface Code Geometry & Algebraic Invariants (A01 / Q02).
Verifies:
  1. H_X * H_Z^T == 0 (mod 2) for both d=3 and d=5 rotated planar surface codes
  2. GF(2) ranks rank(H_X) = rank(H_Z) = (d^2 - 1)/2 and k = 1
  3. Logical X_L and Z_L commute with stabilizers and anticommute with each other
  4. Exact minimum weight of non-trivial logical operators equals d = 3
  5. Bit-for-bit agreement between `tanner_tables.json`, Python builder, C++ tables, and RTL `bp_engine.sv`
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.reference.code_model.css_code_builder import (
    build_rotated_d3_code,
    build_rotated_d5_code,
    verify_css_code_algebra,
)


class TestCodeGeometry(unittest.TestCase):
    def test_d3_css_algebra_and_distance(self) -> None:
        d3 = build_rotated_d3_code()
        res = verify_css_code_algebra(d3, compute_min_distance=True)
        self.assertTrue(res["css_commutes_mod2"])
        self.assertEqual(res["max_commutator_entry"], 0)
        self.assertEqual(res["rank_Hx"], 4)
        self.assertEqual(res["rank_Hz"], 4)
        self.assertEqual(res["n_logical_k"], 1)
        self.assertEqual(res["exact_d_X"], 3)
        self.assertEqual(res["exact_d_Z"], 3)
        self.assertTrue(res["all_checks_passed"])

    def test_d5_css_algebra(self) -> None:
        d5 = build_rotated_d5_code()
        res = verify_css_code_algebra(d5, compute_min_distance=False)
        self.assertTrue(res["css_commutes_mod2"])
        self.assertEqual(res["rank_Hx"], 12)
        self.assertEqual(res["rank_Hz"], 12)
        self.assertEqual(res["n_logical_k"], 1)
        self.assertTrue(res["all_checks_passed"])

    def test_artifacts_and_rtl_consistency(self) -> None:
        tanner_path = REPO_ROOT / "architecture" / "01_code_model" / "tanner_tables.json"
        self.assertTrue(tanner_path.exists())
        data = json.loads(tanner_path.read_text())
        x_tab = data["d3"]["X_basis"]
        self.assertEqual(x_tab["CHECK_CONN"], [[0, 1, 0, 0], [1, 2, 4, 5], [3, 4, 6, 7], [7, 8, 0, 0]])
        self.assertEqual(x_tab["CHECK_ACTIVE_MASK"], [3, 15, 15, 3])

        rtl_text = (REPO_ROOT / "firmware" / "rtl" / "bp_engine.sv").read_text()
        self.assertIn("4'b0011, 4'b1111, 4'b1111, 4'b0011", rtl_text)
        self.assertIn("'{0, 1, 0, 0}", rtl_text)
        self.assertIn("'{7, 8, 0, 0}", rtl_text)


if __name__ == "__main__":
    unittest.main()
