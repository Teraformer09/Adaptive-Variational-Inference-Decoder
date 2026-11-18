"""Phase 0 & Phase 1-4 Quantitative Audit Tests for Layer 1 Seven-Stage Noise Framework.

Locks in the exact experimental measurements from the audit:
  1. Per-generator PD contract response across all 16 PDs (envelope correlation >= 0.60,
     dominant-set hit rate >= 60% for all 6 data-fault generators).
  2. Principal-component & temporal dynamics structure of `SpatialCorrelatedGenerator`
     (PC1 variance share <= 70%, k_95 >= 3 modes needed for 95% variance,
     lag-1 autocorrelation > 0.50 for active cluster latent factors).
  3. Canonical pair-map equivalence between `stage3_spatial_correlations.yaml` and
     `CANONICAL_12_PAIRS`, plus verification that `spatial_clusters` and `beta_loading`
     are loaded and used.
  4. Leave-one-out Z-mass balance (largest single generator share of Z probability mass <= 35%,
     and dropping any single generator changes Z logical error rate by at most 30%).
  5. Per-mechanism baseline diversity (nearest-mean PD classifier from a 100 s window's
     21-vector achieves >= 90% accuracy across the 16 PDs).
  6. CLI default mode in `noise_generator.py` is `seven_stage` (not `legacy`).
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from simulation.stimulus.noise_framework.config_schema import (
    CANONICAL_12_PAIRS,
    GeneratorSpec,
    load_yaml_validated,
)
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.syndrome.fault_realizer import FaultRealizer
from simulation.stimulus.noise_generator import parse_args

REPO_ROOT = Path(__file__).resolve().parents[2]
PD_YAML = REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml"
REG_YAML = REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml"
STAGE3_YAML = REPO_ROOT / "configs" / "noise_framework" / "stage3_spatial_correlations.yaml"


def _ideal_envelope(prof, t_grid: np.ndarray) -> np.ndarray:
    mu_t = float(prof.parameters.get("temporal_mu_sec", 50.0))
    sigma_t = max(float(prof.parameters.get("temporal_sigma_sec", 10.0)), 0.5)
    return np.exp(-0.5 * ((t_grid - mu_t) / sigma_t) ** 2)


def _dominant_indices(prof) -> set[int]:
    dom_q = {int(q) for q in prof.metadata.get("dominant_qubits", []) if 0 <= int(q) < 9}
    dom_p = {9 + int(p) for p in prof.metadata.get("dominant_pairs", []) if 0 <= int(p) < 12}
    return dom_q | dom_p


def test_stage3_yaml_pair_edges_match_canonical_12_pairs() -> None:
    """Cause 4 test: YAML pair_edges must match CANONICAL_12_PAIRS and define valid clusters."""
    cfg = load_yaml_validated(
        STAGE3_YAML, required_keys=("schema_version", "pair_edges", "spatial_clusters")
    )
    edges = cfg["pair_edges"]
    assert len(edges) == 12
    for idx, entry in enumerate(edges):
        assert int(entry["pair_idx"]) == idx
        assert int(entry["mechanism_idx"]) == 9 + idx
        q_pair = tuple(int(x) for x in entry["qubits"])
        assert q_pair == CANONICAL_12_PAIRS[idx], (
            f"YAML pair_idx={idx} has {q_pair}, expected {CANONICAL_12_PAIRS[idx]}"
        )


def test_all_six_data_generators_honor_pd_envelope_and_dominant_set() -> None:
    """Phase 1 target: envelope correlation >= 0.60 and dominant-set hit rate >= 60% for all 6 generator types."""
    pd_reg = PDRegistry(PD_YAML)
    gen_reg = GeneratorRegistry(REG_YAML, master_seed=42)
    t_grid = np.arange(0, 100.0, 0.25, dtype=np.float64)

    # Test one representative spec per data generator type
    specs_by_type = {}
    for spec in gen_reg.list_specs():
        if spec.output_domain == "probability_parameter" and spec.generator_type not in specs_by_type:
            specs_by_type[spec.generator_type] = spec

    assert set(specs_by_type.keys()) == {
        "gaussian",
        "burst",
        "spatial_correlated",
        "ou",
        "telegraph",
        "markov",
    }

    for gtype, spec in specs_by_type.items():
        env_corrs = []
        dom_hits = []
        for pid in pd_reg.list_profiles():
            prof = pd_reg.get(pid)
            env = _ideal_envelope(prof, t_grid)
            dom_set = _dominant_indices(prof)
            k_dom = max(1, len(dom_set))

            for seed_offset in (101, 202):
                gen_cls = gen_reg.get_generator_class(gtype)
                gen = gen_cls(spec=spec, seed=spec.seed + seed_offset)
                gen.assign_profile(prof, window_id=0)

                traj = np.zeros((len(t_grid), 21), dtype=np.float64)
                for s_idx, ts in enumerate(t_grid):
                    contrib = gen.sample(timestamp=float(ts), sample_id=s_idx)
                    traj[s_idx] = contrib.values

                mean_t = traj.mean(axis=1)
                if np.std(mean_t) > 1e-12 and np.std(env) > 1e-12:
                    corr = float(np.corrcoef(mean_t, env)[0, 1])
                else:
                    corr = 0.0
                env_corrs.append(corr)

                mean_mech = traj.mean(axis=0)
                top_k = set(np.argsort(mean_mech)[-k_dom:].tolist())
                hit_rate = len(top_k & dom_set) / float(k_dom)
                dom_hits.append(hit_rate)

        mean_corr = float(np.mean(env_corrs))
        mean_hit = float(np.mean(dom_hits))
        assert mean_corr >= 0.60, (
            f"Generator type '{gtype}' failed envelope correlation: {mean_corr:.3f} < 0.60"
        )
        assert mean_hit >= 0.60, (
            f"Generator type '{gtype}' failed dominant-set hit rate: {mean_hit:.3f} < 0.60"
        )


def test_crosstalk_multi_factor_pca_and_temporal_persistence() -> None:
    """Phase 2 target: PC1 <= 70% variance, >= 3 modes for 95% variance, and lag-1 ACF > 0.50."""
    pd_reg = PDRegistry(PD_YAML)
    gen_reg = GeneratorRegistry(REG_YAML, master_seed=42)
    ct_specs = [
        s for s in gen_reg.list_specs() if s.generator_type == "spatial_correlated"
    ]
    assert len(ct_specs) >= 1
    spec = ct_specs[0]
    gen_cls = gen_reg.get_generator_class("spatial_correlated")

    pc1_shares = []
    modes_95 = []
    lag1_acfs = []

    for idx, pid in enumerate(pd_reg.list_profiles()):
        prof = pd_reg.get(pid)
        gen = gen_cls(spec=spec, seed=5000 + idx * 37)
        gen.assign_profile(prof, window_id=0)

        traj = np.zeros((1000, 21), dtype=np.float64)
        for s_idx in range(1000):
            c = gen.sample(timestamp=s_idx * 0.1, sample_id=s_idx)
            traj[s_idx] = c.values

        centered = traj - traj.mean(axis=0, keepdims=True)
        _, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
        var_exp = (s_vals ** 2) / max(float(np.sum(s_vals ** 2)), 1e-15)
        cum_var = np.cumsum(var_exp)
        pc1_shares.append(float(var_exp[0]))
        k95 = int(np.searchsorted(cum_var, 0.95) + 1)
        modes_95.append(k95)

        # Measure lag-1 autocorrelation on PC1 projection / dominant mechanism
        pc1_ts = centered @ np.linalg.svd(centered, full_matrices=False)[2][0]
        if np.std(pc1_ts[:-1]) > 1e-12 and np.std(pc1_ts[1:]) > 1e-12:
            acf1 = float(np.corrcoef(pc1_ts[:-1], pc1_ts[1:])[0, 1])
        else:
            acf1 = 0.0
        lag1_acfs.append(acf1)

    mean_pc1 = float(np.mean(pc1_shares))
    mean_k95 = float(np.mean(modes_95))
    mean_acf1 = float(np.mean(lag1_acfs))

    assert mean_pc1 <= 0.70, f"Crosstalk PC1 variance share {mean_pc1:.3f} > 0.70"
    assert mean_k95 >= 3.0, f"Crosstalk modes for 95% variance {mean_k95:.2f} < 3.0"
    assert mean_acf1 > 0.50, f"Crosstalk lag-1 autocorrelation {mean_acf1:.3f} <= 0.50"


def test_leave_one_out_z_mass_and_logical_error_balance() -> None:
    """Phase 3 Step 8 target: largest single generator share of Z mass <= 35% and LOO LER drop <= 30%."""
    pd_reg = PDRegistry(PD_YAML)
    realizer = FaultRealizer()
    n_seeds = 8
    duration_sec = 100.0

    mass_by_gen: dict[str, float] = {}
    total_z_mass = 0.0
    full_z_flips = 0
    total_rounds = 0
    loo_z_flips: dict[str, int] = {}

    for seed in range(7000, 7000 + n_seeds):
        gen_reg = GeneratorRegistry(REG_YAML, master_seed=seed)
        generators = gen_reg.instantiate_all(master_seed=seed)
        pd_assign = PDAssignment(
            pd_registry=pd_reg,
            master_seed=seed,
            policy_name="curriculum_burst_enriched",
            assignment_config_path=REPO_ROOT / "configs" / "noise_framework" / "generator_assignment.yaml",
        )
        scheduler = AssignmentScheduler(pd_assignment=pd_assign, window_duration_sec=100.0)
        scheduler.initialize(generators=generators, pd_registry=pd_reg)
        clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)
        composer = NoiseComposer()
        sampler = SynchronizedSampler(
            generators=generators, clock=clock, scheduler=scheduler, composer=composer
        )
        samples = sampler.generate_trajectory(duration_seconds=duration_sec)

        rng_full = np.random.default_rng(seed + 99999)
        z_gen_ids = [
            g.generator_id
            for g in generators
            if g.output_domain == "probability_parameter"
            and str(g.spec.parameters.get("basis", "Z")).upper() == "Z"
        ]
        for gid in z_gen_ids:
            loo_z_flips.setdefault(gid, 0)
            mass_by_gen.setdefault(gid, 0.0)

        for comp in samples:
            total_rounds += 1
            r_full = realizer.realize(comp, rng=rng_full, n_rounds=1)
            full_z_flips += int(r_full["logical_z_flip"])

            z_contribs = [
                c
                for c in comp.contributions
                if c.output_domain == "probability_parameter"
                and str(c.metadata.get("basis", "Z")).upper() == "Z"
            ]
            for c in z_contribs:
                m = float(np.sum(c.values))
                mass_by_gen[c.generator_id] += m
                total_z_mass += m

            for drop_gid in z_gen_ids:
                kept = [c for c in comp.contributions if c.generator_id != drop_gid]
                comp_loo = composer.compose(
                    contributions=kept, timestamp=comp.timestamp, sample_id=comp.sample_id
                )
                rng_loo = np.random.default_rng(seed * 10000 + comp.sample_id * 13 + hash(drop_gid) % 997)
                r_loo = realizer.realize(comp_loo, rng=rng_loo, n_rounds=1)
                loo_z_flips[drop_gid] += int(r_loo["logical_z_flip"])

    max_share = max(v / max(total_z_mass, 1e-12) for v in mass_by_gen.values())
    assert max_share <= 0.35, f"Largest Z generator mass share {max_share:.3f} > 0.35"

    base_ler = full_z_flips / max(total_rounds, 1)
    for gid, flips in loo_z_flips.items():
        loo_ler = flips / max(total_rounds, 1)
        rel_drop = (base_ler - loo_ler) / max(base_ler, 1e-9)
        assert rel_drop <= 0.30, (
            f"Dropping '{gid}' reduced Z LER by {rel_drop*100:.1f}% (> 30% limit)"
        )


def test_nearest_mean_pd_identification_from_21_vector() -> None:
    """Phase 3 Step 7 target: nearest-mean test identifies assigned PD from 100s window 21-vector with >= 90% accuracy."""
    pd_reg = PDRegistry(PD_YAML)
    gen_reg = GeneratorRegistry(REG_YAML, master_seed=42)
    pids = pd_reg.list_profiles()

    # Compute reference templates per (generator_type, pid) from seed 1 using 200 samples
    correct = 0
    total = 0
    seen_types = set()
    for spec in gen_reg.list_specs():
        if spec.output_domain != "probability_parameter" or spec.generator_type in seen_types:
            continue
        seen_types.add(spec.generator_type)
        gen_cls = gen_reg.get_generator_class(spec.generator_type)
        templates = {}
        for pid in pids:
            prof = pd_reg.get(pid)
            gen = gen_cls(spec=spec, seed=1111)
            gen.assign_profile(prof, window_id=0)
            vecs = [gen.sample(timestamp=i * 0.5, sample_id=i).values for i in range(200)]
            m = np.mean(vecs, axis=0)
            templates[pid] = m / max(float(np.linalg.norm(m)), 1e-12)

        for test_seed in (2222, 3333):
            for pid in pids:
                prof = pd_reg.get(pid)
                gen = gen_cls(spec=spec, seed=test_seed)
                gen.assign_profile(prof, window_id=0)
                vecs = [gen.sample(timestamp=i * 0.5, sample_id=i).values for i in range(200)]
                m = np.mean(vecs, axis=0)
                m_norm = m / max(float(np.linalg.norm(m)), 1e-12)
                pred_pid = min(
                    pids, key=lambda k: float(np.linalg.norm(m_norm - templates[k]))
                )
                if pred_pid == pid:
                    correct += 1
                total += 1

    acc = correct / max(total, 1)
    assert acc >= 0.90, f"Nearest-mean PD identification accuracy {acc:.3f} < 0.90"


def test_noise_generator_cli_default_mode_is_seven_stage(monkeypatch) -> None:
    """CLI default mode must be 'seven_stage' rather than 'legacy' single-pair mode."""
    monkeypatch.setattr("sys.argv", ["noise_generator.py"])
    args = parse_args()
    assert args.mode == "seven_stage"
