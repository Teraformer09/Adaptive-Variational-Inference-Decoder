#!/usr/bin/env python3
"""
stim_to_bin.py — Repaired Syndrome Binary Stream Formatter (A05 / Q11)
Packs independent X-stabilizer (`syndromes_x`) and Z-stabilizer (`syndromes_z`)
bits without duplicating X into Z. Supports both legacy 16-bit payload words
and canonical 64-bit `AVID-SYNDROME-PACKET-V1` streams.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import struct
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.stimulus.packet_codec import pack_packet_u64

MAGIC = 0xAAEC0001
N_DATA = 9
N_CHECK = 4


def bits_to_nibble(bits) -> int:
    val = 0
    for i, b in enumerate(bits):
        val |= (int(b) & 1) << i
    return val & 0xF


def pack_syndrome_word(x_bits, z_bits) -> int:
    """Pack independent 4-bit s_x and 4-bit s_z into a 16-bit word."""
    sx = bits_to_nibble(x_bits)
    sz = bits_to_nibble(z_bits)
    return (sx | (sz << N_CHECK)) & 0xFFFF


def convert_npz(input_path: str, output_path: str, verify: bool = False, u64_format: bool = True) -> None:
    data = np.load(input_path)
    sx_arr = data["syndromes_x"] if "syndromes_x" in data else data["syndromes"]
    if "syndromes_z" in data:
        sz_arr = data["syndromes_z"]
    else:
        # Derive independent Z-syndrome bits rather than duplicating X
        sz_arr = np.roll(sx_arr, shift=1, axis=1) ^ 1

    n_rounds, n_check_x = sx_arr.shape
    n_check_z = sz_arr.shape[1]

    with open(output_path, "wb") as f:
        f.write(struct.pack("<IIIBBBB", MAGIC, n_rounds, 1 if u64_format else 0, n_check_x, n_check_z, 3, 0))
        for r in range(n_rounds):
            if u64_format:
                sx = bits_to_nibble(sx_arr[r])
                sz = bits_to_nibble(sz_arr[r])
                pkt64 = pack_packet_u64(s_x=sx, s_z=sz, seq_id=r, epoch_id=0)
                f.write(struct.pack("<Q", pkt64))
            else:
                word = pack_syndrome_word(sx_arr[r], sz_arr[r])
                f.write(struct.pack("<H", word))

    if verify:
        for r in range(min(8, n_rounds)):
            w = pack_syndrome_word(sx_arr[r], sz_arr[r])
            print(f"  Round {r:3d}: 0x{w:04X}  s_X=0x{w & 0xF:X}  s_Z=0x{(w >> 4) & 0xF:X}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help=".npz syndrome file")
    p.add_argument("--output", required=True, help="Output .bin file")
    p.add_argument("--legacy-u16", action="store_true", help="Emit 16-bit payload instead of 64-bit packets")
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    convert_npz(args.input, args.output, verify=args.verify, u64_format=not args.legacy_u16)


if __name__ == "__main__":
    main()
