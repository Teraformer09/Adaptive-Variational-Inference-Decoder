#!/usr/bin/env python3
"""
noise_generator.py — Canonical CSS Non-Markovian & Seven-Stage Noise Scenario Generator.
Addresses Policy Items A01 / Q02, A05 / Q11, and A08 / Q17 while also exposing
`generate_seven_stage_trajectory()` backed by the persistent-PD multi-generator framework.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.reference.code_model.css_code_builder import build_rotated_d3_code

D = 3
N_DATA = 9
N_CHECK = 4

_D3_CODE = build_rotated_d3_code()
CHECK_CONN = _D3_CODE.x_check_supports  # [[0, 1], [1, 2, 4, 5], [3, 4, 6, 7], [7, 8]]
Z_CHECK_CONN = _D3_CODE.z_check_supports  # [[0, 1, 3, 4], [2, 5], [3, 6], [4, 5, 7, 8]]


def compute_syndromes(errors: np.ndarray, check_supports=None) -> np.ndarray:
    """
    errors: (n_rounds, n_data) binary array
    returns: (n_rounds, n_check) syndrome bits
    """
    supports = CHECK_CONN if check_supports is None else check_supports
    n_rounds = errors.shape[0]
    synd = np.zeros((n_rounds, len(supports)), dtype=np.uint8)
    for c, conn in enumerate(supports):
        synd[:, c] = errors[:, conn].sum(axis=1) % 2
    return synd


def sample_errors(
    n_rounds: int,
    p_single: np.ndarray,
    p_cross_pairs: list,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample Pauli fault patterns under single-qubit + correlated pair noise model.
    """
    errors = rng.binomial(1, p_single, size=(n_rounds, N_DATA)).astype(np.uint8)
    for i, j, p_ij in p_cross_pairs:
        if p_ij > 0.0:
            corr = rng.binomial(1, p_ij, size=n_rounds).astype(np.uint8)
            errors[:, i] ^= corr
            errors[:, j] ^= corr
    return errors & 1


def generate_scenario(
    total_rounds: int,
    rng: np.random.Generator,
    p_meas: float = 0.002,
):
    """
    Generate syndrome dataset with drifting + burst ZZ crosstalk + independent X/Z errors.
    Returns:
      syndromes_x: (total_rounds, N_CHECK) uint8 — X-stabilizer syndromes (detect Z errors)
      syndromes_z: (total_rounds, N_CHECK) uint8 — Z-stabilizer syndromes (detect X errors)
      true_p:      (total_rounds, N_DATA)  float32 — ground-truth p_Z(t)
      true_pzz:    (total_rounds,)         float32 — ground-truth p_ZZ[4,5](t)
    """
    syndromes_x = np.zeros((total_rounds, N_CHECK), dtype=np.uint8)
    syndromes_z = np.zeros((total_rounds, N_CHECK), dtype=np.uint8)
    true_p = np.zeros((total_rounds, N_DATA), dtype=np.float32)
    true_pzz = np.zeros(total_rounds, dtype=np.float32)

    chunk = min(10_000, max(1, total_rounds))

    for t_start in range(0, total_rounds, chunk):
        t_end = min(t_start + chunk, total_rounds)
        n = t_end - t_start
        t_mid = (t_start + t_end) // 2

        p_z = np.full(N_DATA, 0.01, dtype=np.float64)
        p_x = np.full(N_DATA, 0.008, dtype=np.float64)

        frac = t_mid / max(1, total_rounds)
        p_z[4] = 0.01 + 0.04 * frac
        p_z[5] = 0.01 + 0.02 * frac
        p_x[3] = 0.008 + 0.02 * frac

        p_zz_45 = 0.08 if (0.50 * total_rounds <= t_mid < 0.55 * total_rounds) else 0.004
        cross_z = [(4, 5, p_zz_45)]
        cross_x = [(1, 4, 0.003)]

        err_z = sample_errors(n, p_z, cross_z, rng)
        err_x = sample_errors(n, p_x, cross_x, rng)

        sx = compute_syndromes(err_z, CHECK_CONN)
        sz = compute_syndromes(err_x, Z_CHECK_CONN)

        if p_meas > 0.0:
            meas_flip_x = rng.binomial(1, p_meas, size=sx.shape).astype(np.uint8)
            meas_flip_z = rng.binomial(1, p_meas, size=sz.shape).astype(np.uint8)
            sx ^= meas_flip_x
            sz ^= meas_flip_z

        syndromes_x[t_start:t_end] = sx
        syndromes_z[t_start:t_end] = sz
        true_p[t_start:t_end] = p_z.astype(np.float32)
        true_pzz[t_start:t_end] = p_zz_45

    return syndromes_x, syndromes_z, true_p, true_pzz


def generate_seven_stage_trajectory(
    duration_seconds: float = 100.0,
    seed: int = 42,
    assignment_policy: str = "curriculum_burst_enriched",
):
    """Generate a trajectory using the Seven-Stage persistent-PD multi-generator framework."""
    from simulation.stimulus.seven_stage.stage6_stratified_sampler import (
        generate_single_stratified_trajectory,
    )

    encoded, prov, assign_manifest = generate_single_stratified_trajectory(
        trajectory_id=f"seven_stage_s{seed}",
        split_id="eval",
        master_seed=seed,
        duration_seconds=duration_seconds,
        assignment_policy=assignment_policy,
    )
    return encoded, prov, assign_manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["legacy", "seven_stage"], default="legacy")
    p.add_argument("--duration-sec", type=float, default=100.0)
    p.add_argument("--output", type=str, default="simulation/stimulus/syndromes.npz")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "seven_stage":
        from simulation.stimulus.noise_framework.avid_interface.avid_input_adapter import (
            AvidInputAdapter,
        )

        encoded, prov, _ = generate_seven_stage_trajectory(
            duration_seconds=args.duration_sec, seed=args.seed
        )
        AvidInputAdapter().write_npz(encoded, out_path, extra_metadata={"provenance": prov})
        print(f"Saved {encoded['total_rounds']:,} seven-stage CSS rounds -> {out_path}")
        return

    rng = np.random.default_rng(args.seed)
    sx, sz, true_p, true_pzz = generate_scenario(args.rounds, rng)

    np.savez_compressed(
        str(out_path),
        syndromes=sx,
        syndromes_x=sx,
        syndromes_z=sz,
        true_p=true_p,
        true_pzz=true_pzz,
        n_check=N_CHECK,
        n_data=N_DATA,
        total_rounds=args.rounds,
    )
    print(f"Saved {args.rounds:,} canonical CSS rounds -> {out_path}")


if __name__ == "__main__":
    main()
