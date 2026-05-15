#!/usr/bin/env python3
"""
stim_to_bin.py — AVID Phase 4.2
Syndrome Binary Stream Formatter
=================================
Converts syndromes from noise_generator.py (.npz) or Stim (.det) format
into the AXI-Stream binary blob consumed by the ARM DMA descriptor chain.

Binary format (little-endian):
  Header (16 bytes):
    uint32 magic       = 0xAAEC0001
    uint32 n_rounds
    uint8  n_check_x
    uint8  n_check_z
    uint8  d           (surface code distance)
    uint8  reserved    = 0x00
    uint32 reserved2   = 0x00000000

  Payload (2 bytes per round):
    uint16 syndrome_word:
      bits [n_check_x-1:0]               = X-stabilizer syndromes
      bits [n_check_x+n_check_z-1:n_check_x] = Z-stabilizer syndromes
      remaining bits = 0

ARM usage (C):
  uint8_t *buf; size_t n_bytes;
  load_syndrome_file("data.bin", &buf, &n_bytes);
  // Skip 16-byte header
  uint16_t *syndrome_words = (uint16_t*)(buf + 16);
  dma_load_syndromes(&dma, syndrome_words, (n_rounds * 2));
  dma_start_stream(&dma, n_rounds * 2, n_rounds * 4);

Usage:
  # From noise_generator.py output:
  python stim_to_bin.py --input syndromes.npz --output data.bin

  # Verify first 8 words:
  python stim_to_bin.py --input syndromes.npz --output data.bin --verify
"""

import argparse
import struct
import sys
import numpy as np
from pathlib import Path

MAGIC    = 0xAAEC0001
N_DATA   = 9
N_CHECK  = 4  # d=3

def pack_syndrome_word(x_bits, z_bits):
    word = 0
    for i, b in enumerate(x_bits): word |= (int(b) & 1) << i
    for i, b in enumerate(z_bits):  word |= (int(b) & 1) << (N_CHECK + i)
    return word & 0xFFFF


def convert_npz(input_path: str, output_path: str, verify: bool):
    data = np.load(input_path)
    syndromes = data["syndromes"]   # (n_rounds, n_check)
    n_rounds, n_check_x = syndromes.shape
    n_check_z = n_check_x  # Symmetric for d=3

    print(f"Input  : {input_path}  ({n_rounds:,} rounds, {n_check_x} checks)")

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<IIIBBBB",
                            MAGIC, n_rounds, 0,       # padding for alignment
                            n_check_x, n_check_z, 3, 0))

        # Payload
        for r in range(n_rounds):
            word = pack_syndrome_word(syndromes[r], syndromes[r])
            f.write(struct.pack("<H", word))

    total = 16 + 2 * n_rounds
    print(f"Output : {output_path}  ({total:,} bytes)")

    if verify:
        print("\nFirst 8 syndrome words:")
        for r in range(min(8, n_rounds)):
            word = pack_syndrome_word(syndromes[r], syndromes[r])
            print(f"  Round {r:3d}: 0x{word:04X}  X={syndromes[r]:08b}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True,  help=".npz syndrome file")
    p.add_argument("--output", required=True,  help="Output .bin file")
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    convert_npz(args.input, args.output, args.verify)
    print("Done.  Use dma_load_syndromes() on the ARM to stream to FPGA.")


if __name__ == "__main__":
    main()
