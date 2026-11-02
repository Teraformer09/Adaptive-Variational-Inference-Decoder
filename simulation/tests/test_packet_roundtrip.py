#!/usr/bin/env python3
"""
Unit Test Suite 5: Canonical 64-Bit Syndrome Packet Codec & Stream Validation (A05 / Q11).
Verifies:
  1. Bit-exact packing and unpacking of 64-bit syndrome words (`s_x`, `s_z`, `flags`, `epoch_id`, `seq_id`)
  2. Independent `s_x` and `s_z` preservation (no X->Z duplication)
  3. Detection of sequence gaps (`seq_id != prev_seq_id + 1`) and invalid packet flags
"""

from __future__ import annotations
from pathlib import Path
import struct
import sys
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.stimulus.packet_codec import (
    FLAG_PARITY_ERR,
    FLAG_VALID_PKT,
    pack_packet_stream,
    pack_packet_u64,
    unpack_packet_stream,
    unpack_packet_u64,
)
from simulation.stimulus.stim_to_bin import pack_syndrome_word


class TestPacketRoundtrip(unittest.TestCase):
    def test_single_packet_roundtrip(self) -> None:
        w = pack_packet_u64(s_x=0xA, s_z=0x5, seq_id=123456, epoch_id=42, flags=FLAG_VALID_PKT)
        pkt = unpack_packet_u64(w)
        self.assertEqual(pkt.s_x, 0xA)
        self.assertEqual(pkt.s_z, 0x5)
        self.assertNotEqual(pkt.s_x, pkt.s_z)
        self.assertEqual(pkt.seq_id, 123456)
        self.assertEqual(pkt.epoch_id, 42)
        self.assertTrue(pkt.is_valid)

    def test_stream_roundtrip_and_gap_detection(self) -> None:
        rng = np.random.default_rng(7)
        sx = rng.integers(0, 16, size=128)
        sz = rng.integers(0, 16, size=128)
        raw = pack_packet_stream(sx, sz, start_seq_id=1000, epoch_id=3)
        pkts, stats = unpack_packet_stream(raw)
        self.assertEqual(stats["total_packets"], 128)
        self.assertEqual(stats["seq_gaps"], 0)
        self.assertEqual(stats["invalid_packets"], 0)
        for i in range(128):
            self.assertEqual(pkts[i].s_x, int(sx[i]))
            self.assertEqual(pkts[i].s_z, int(sz[i]))

        # Inject a sequence gap and an invalid flag
        bad_words = list(struct.unpack("<128Q", raw))
        bad_words[50] = pack_packet_u64(s_x=1, s_z=2, seq_id=999999, epoch_id=3, flags=FLAG_VALID_PKT)
        bad_words[70] = pack_packet_u64(s_x=1, s_z=2, seq_id=1070, epoch_id=3, flags=FLAG_PARITY_ERR)
        bad_raw = struct.pack("<128Q", *bad_words)
        _, bad_stats = unpack_packet_stream(bad_raw)
        self.assertGreaterEqual(bad_stats["seq_gaps"], 1)
        self.assertEqual(bad_stats["invalid_packets"], 1)

    def test_stim_to_bin_no_xz_duplication(self) -> None:
        w = pack_syndrome_word([1, 0, 1, 0], [0, 1, 1, 1])
        self.assertEqual(w & 0xF, 0b0101)
        self.assertEqual((w >> 4) & 0xF, 0b1110)


if __name__ == "__main__":
    unittest.main()
