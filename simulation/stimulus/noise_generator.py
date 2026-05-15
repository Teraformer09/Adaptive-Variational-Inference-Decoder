#!/usr/bin/env python3
"""
noise_generator.py — AVID Phase 4.1
Non-Markovian Noise Scenario Generator
=======================================
Generates a 1,000,000-round surface code syndrome dataset with deliberate
time-varying noise to validate the adaptive decoder.

Noise scenario (designed to stress the static decoder):

  Phase 0  t = 0..499,999   : Stationary regime
                               p_Z_uniform  = 0.01 (all qubits)
                               p_ZZ_crosstalk = 0.000

  Phase 1  t = 0..999,999   : Drifting flux-bias on qubits 4 and 5
                               p_Z[4] linearly 0.01 → 0.05 (5x increase)
                               p_Z[5] linearly 0.01 → 0.03
                               Models: slow charge noise / flux creep

  Burst    t = 500,000..510,000 : Sudden ZZ crosstalk on pair (4,5)
                               p_ZZ[4,5] = 0.08 for 10,000 rounds
                               Models: transient microwave crosstalk / TLS

  The static BP decoder (optimised at t=0) will degrade in Phase 1 and
  catastrophically fail during the Burst.  AVID tracks the drift and
  recovers within a few update epochs (~100ms).

Usage
-----
  python simulation/stimulus/noise_generator.py [--rounds 1000000] [--output syndromes.npy]
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import stim
    STIM_AVAILABLE = True
except ImportError:
    STIM_AVAILABLE = False
    print("[WARNING] stim not installed — using Bernoulli approximation")

# ============================================================
# d=3 surface code Tanner graph (X-type stabilizers)
# ============================================================
D = 3
N_DATA   = D * D          # 9 qubits
N_CHECK  = (N_DATA - 1) // 2  # 4 X-stabilizers

CHECK_CONN = [
    [0, 1, 3, 4],
    [1, 2, 4, 5],
    [3, 4, 6, 7],
    [4, 5, 7, 8],
]

# ============================================================
# Syndrome computation from error pattern (exact formula)
# ============================================================

def compute_syndromes(errors: np.ndarray) -> np.ndarray:
    """
    errors: (n_rounds, n_data) binary array
    returns: (n_rounds, n_check) syndrome bits
    """
    n_rounds = errors.shape[0]
    synd = np.zeros((n_rounds, N_CHECK), dtype=np.uint8)
    for c, conn in enumerate(CHECK_CONN):
        synd[:, c] = errors[:, conn].sum(axis=1) % 2
    return synd


def sample_errors(n_rounds: int, p_single: np.ndarray,
                  p_cross_pairs: list, rng: np.random.Generator) -> np.ndarray:
    """
    Sample Z-error patterns under correlated Pauli noise model.

    p_single: (N_DATA,) array of single-qubit Z-error rates
    p_cross_pairs: list of (i, j, p_ij) tuples for ZZ crosstalk
    """
    errors = rng.binomial(1, p_single, size=(n_rounds, N_DATA)).astype(np.uint8)

    for (i, j, p_ij) in p_cross_pairs:
        if p_ij > 0:
            # ZZ error: both qubits i and j flip together
            corr = rng.binomial(1, p_ij, size=n_rounds).astype(np.uint8)
            errors[:, i] ^= corr
            errors[:, j] ^= corr

    return errors % 2


# ============================================================
# Non-Markovian scenario builder
# ============================================================

def generate_scenario(total_rounds: int, rng: np.random.Generator):
    """
    Generate syndrome dataset with drifting + burst noise.
    Returns:
      syndromes: (total_rounds, N_CHECK) uint8 — X-stabilizer syndromes
      true_p:    (total_rounds, N_DATA)  float  — ground-truth p_i(t) for tracking
      true_pzz:  (total_rounds,)         float  — p_ZZ[4,5](t)
    """
    print(f"Generating {total_rounds:,} rounds of non-Markovian noise...")

    syndromes = np.zeros((total_rounds, N_CHECK), dtype=np.uint8)
    true_p    = np.zeros((total_rounds, N_DATA),  dtype=np.float32)
    true_pzz  = np.zeros(total_rounds,            dtype=np.float32)

    CHUNK = 10000  # Process in chunks to avoid RAM overflow

    for t_start in range(0, total_rounds, CHUNK):
        t_end = min(t_start + CHUNK, total_rounds)
        n     = t_end - t_start

        # Mid-chunk time index for parameter evaluation
        t_mid = (t_start + t_end) // 2

        # ---- Per-qubit error rates (time-varying) ----
        p = np.full(N_DATA, 0.01, dtype=np.float64)

        # Phase 1: linear drift on q4 and q5
        if t_mid < total_rounds:
            frac = t_mid / total_rounds
            p[4] = 0.01 + 0.04 * frac        # 0.01 → 0.05
            p[5] = 0.01 + 0.02 * frac        # 0.01 → 0.03

        # ---- Crosstalk burst ----
        p_zz_45 = 0.0
        if 500_000 <= t_mid < 510_000:
            p_zz_45 = 0.08
            print(f"  [burst] ZZ crosstalk p(4,5)=0.08 at t={t_mid:,}")

        cross_pairs = [(4, 5, p_zz_45)] if p_zz_45 > 0 else []

        # ---- Sample errors and compute syndromes ----
        errors = sample_errors(n, p, cross_pairs, rng)
        synd   = compute_syndromes(errors)

        syndromes[t_start:t_end] = synd
        true_p[t_start:t_end]    = p.astype(np.float32)
        true_pzz[t_start:t_end]  = p_zz_45

    return syndromes, true_p, true_pzz


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds",  type=int, default=1_000_000)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--output",  type=str, default="simulation/stimulus/syndromes.npz")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    syndromes, true_p, true_pzz = generate_scenario(args.rounds, rng)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path),
                        syndromes=syndromes,
                        true_p=true_p,
                        true_pzz=true_pzz,
                        n_check=N_CHECK,
                        n_data=N_DATA,
                        total_rounds=args.rounds)

    print(f"Saved {args.rounds:,} rounds → {out_path}")
    print(f"  Array shapes: syndromes={syndromes.shape}  true_p={true_p.shape}")
    syndrome_rate = syndromes.mean()
    print(f"  Mean syndrome fire rate: {syndrome_rate:.4f}")


if __name__ == "__main__":
    main()
