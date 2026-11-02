"""
Canonical 64-Bit Syndrome Packet Encoder/Decoder (Policy Item A05 / Checklist Q11).
Enforces `architecture/contracts/syndrome_packet_v1.yaml`:
  - bits [3:0]   : s_x (4-bit X-stabilizer syndrome, detects Z faults)
  - bits [7:4]   : s_z (4-bit Z-stabilizer syndrome, detects X faults)
  - bits [15:8]  : flags (bit 0 = valid_pkt, bit 1 = overflow_warn, bit 2 = parity_err, bit 3 = epoch_marker)
  - bits [31:16] : epoch_id (16-bit weight bank epoch ID)
  - bits [63:32] : seq_id (32-bit strictly monotonic round sequence number)
"""

from __future__ import annotations
from dataclasses import dataclass
import struct
from typing import List, Sequence, Tuple
import numpy as np


FLAG_VALID_PKT = 1 << 0
FLAG_OVERFLOW_WARN = 1 << 1
FLAG_PARITY_ERR = 1 << 2
FLAG_EPOCH_MARKER = 1 << 3


@dataclass(frozen=True)
class SyndromePacket:
    seq_id: int
    s_x: int
    s_z: int
    epoch_id: int = 0
    flags: int = FLAG_VALID_PKT

    @property
    def is_valid(self) -> bool:
        return bool(self.flags & FLAG_VALID_PKT) and not bool(self.flags & FLAG_PARITY_ERR)


def compute_even_parity_bit(s_x: int, s_z: int, epoch_id: int, seq_id: int) -> int:
    """Compute 1-bit XOR parity across (s_x, s_z, epoch_id, seq_id)."""
    val = (s_x & 0xF) ^ ((s_z & 0xF) << 4) ^ (epoch_id & 0xFFFF) ^ (seq_id & 0xFFFFFFFF)
    return bin(val).count("1") & 1


def pack_packet_u64(
    s_x: int,
    s_z: int,
    seq_id: int,
    epoch_id: int = 0,
    flags: int = FLAG_VALID_PKT,
) -> int:
    """Pack a single canonical 64-bit syndrome packet word."""
    sx = int(s_x) & 0xF
    sz = int(s_z) & 0xF
    fl = int(flags) & 0xFF
    ep = int(epoch_id) & 0xFFFF
    sq = int(seq_id) & 0xFFFFFFFF
    word = sx | (sz << 4) | (fl << 8) | (ep << 16) | (sq << 32)
    return int(word)


def unpack_packet_u64(word: int) -> SyndromePacket:
    """Unpack a 64-bit integer into a SyndromePacket."""
    w = int(word) & 0xFFFFFFFFFFFFFFFF
    s_x = w & 0xF
    s_z = (w >> 4) & 0xF
    flags = (w >> 8) & 0xFF
    epoch_id = (w >> 16) & 0xFFFF
    seq_id = (w >> 32) & 0xFFFFFFFF
    return SyndromePacket(
        seq_id=seq_id,
        s_x=s_x,
        s_z=s_z,
        epoch_id=epoch_id,
        flags=flags,
    )


def pack_packet_stream(
    s_x_array: Sequence[int],
    s_z_array: Sequence[int],
    start_seq_id: int = 0,
    epoch_id: int = 0,
) -> bytes:
    """Pack arrays of (s_x, s_z) into a little-endian uint64 binary stream."""
    if len(s_x_array) != len(s_z_array):
        raise ValueError("s_x_array and s_z_array must have identical length")
    n = len(s_x_array)
    words = np.zeros(n, dtype="<u8")
    for i in range(n):
        words[i] = pack_packet_u64(
            s_x=int(s_x_array[i]),
            s_z=int(s_z_array[i]),
            seq_id=start_seq_id + i,
            epoch_id=epoch_id,
            flags=FLAG_VALID_PKT,
        )
    return words.tobytes()


def unpack_packet_stream(raw_bytes: bytes) -> Tuple[List[SyndromePacket], Dict[str, int]]:
    """
    Unpack a binary stream of 64-bit syndrome packets and verify monotonic seq_id
    and packet validity flags.
    """
    if len(raw_bytes) % 8 != 0:
        raise ValueError(f"Stream byte length {len(raw_bytes)} is not a multiple of 8")
    n = len(raw_bytes) // 8
    packets: List[SyndromePacket] = []
    seq_gaps = 0
    invalid_pkts = 0

    for i in range(n):
        (word,) = struct.unpack_from("<Q", raw_bytes, i * 8)
        pkt = unpack_packet_u64(word)
        if not pkt.is_valid:
            invalid_pkts += 1
        if i > 0 and pkt.seq_id != (packets[-1].seq_id + 1) & 0xFFFFFFFF:
            seq_gaps += 1
        packets.append(pkt)

    stats = {
        "total_packets": n,
        "seq_gaps": seq_gaps,
        "invalid_packets": invalid_pkts,
    }
    return packets, stats
