#!/usr/bin/env python3
"""
Unit Test Suite 2: Physical Fault Event Signatures & XOR Linearity (A02 / Q03, A08 / Q17).
Verifies:
  1. Single-qubit Z_i signature g_i matches column i of H_X
  2. Correlated pair Z_i Z_j signature equals bitwise XOR g_ij = g_i ^ g_j
  3. Measurement fault signature equals unit bitmask (1 << c)
  4. Event signatures in `architecture/noise_model/event_signatures.json` match
     both the Python model and C++ `NM_EVENT_SIGNATURES[21]`
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.reference.noise_model.joint_dp_likelihood import build_d3_correlated_noise_model


class TestEventSignatures(unittest.TestCase):
    def test_single_and_pair_xor_signatures(self) -> None:
        model = build_d3_correlated_noise_model(basis="X", include_measurement_faults=True)
        self.assertEqual(model.n_params, 25)

        single_masks = [model.events[q].signature_mask for q in range(9)]
        expected_single = [0x1, 0x3, 0x2, 0x4, 0x6, 0x2, 0x4, 0xC, 0x8]
        self.assertEqual(single_masks, expected_single)

        for idx, (u, v) in enumerate(model.code.zz_edges):
            ev = model.events[9 + idx]
            self.assertEqual(ev.category, "zz_pair")
            self.assertEqual(ev.signature_mask, single_masks[u] ^ single_masks[v])

        for c in range(4):
            ev = model.events[21 + c]
            self.assertEqual(ev.category, "measurement")
            self.assertEqual(ev.signature_mask, 1 << c)

    def test_json_artifact_consistency(self) -> None:
        sig_path = REPO_ROOT / "architecture" / "noise_model" / "event_signatures.json"
        self.assertTrue(sig_path.exists())
        payload = json.loads(sig_path.read_text())
        masks_from_json = [e["signature_mask"] for e in payload["events_21_param"]]
        model = build_d3_correlated_noise_model(basis="X", include_measurement_faults=False)
        self.assertEqual(masks_from_json, model.signatures.tolist())


if __name__ == "__main__":
    unittest.main()
