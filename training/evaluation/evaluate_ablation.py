"""Full Representation & Controller Ablation Suite (`evaluate_ablation.py`).

Evaluates (EVAL-02):
  1. `1_Static_B0` (Fixed uniform hardware LUT, K=3)
  2. `2_Memoryless_JointDP_Proj` (Layer 3 posterior mean + fixed pair projection K=4, no R2D2)
  3. `3_R2D2_V2_MeanOnly_NoUncertainty` (Ablating the 21 uncertainty features)
  4. `4_R2D2_V2_Full_82D` (Deployable syndrome-conditioned 82-D R2D2 V2)
  5. `5_R2D2_V2_Oracle_7x3` (Privileged ground-truth 7x3 upper-bound reference)
"""

from __future__ import annotations

from typing import Dict, Sequence
import numpy as np

from training.data.trajectory_split import TrajectorySplitManager
from training.environments.decoder_in_loop import (
    STATIC_LUT_Z,
    count_logical_z_failures,
    decode_window_minsum_osd0_z,
    project_q78_weights_d3,
)
from training.environments.qec_episode_generator import generate_v2_qec_episode
from training.models.r2d2_network import R2D2Network


def run_representation_and_controller_ablation(
    trained_v2_net: R2D2Network,
    regimes: Sequence[str] = (
        "gaussian_stationary",
        "ou_drift",
        "zz_crosstalk",
        "tls_telegraph",
        "nonstationary_burst",
    ),
    seed_start: int = 2026,
    seed_end: int = 2033,
    num_epochs: int = 16,
    window_rounds: int = 256,
) -> Dict[str, Dict[str, float]]:
    """Run controlled shared-seed ablations isolating Layer 3 projection, uncertainty, and R2D2 V2."""
    split_mgr = TrajectorySplitManager()

    # Evaluate 4_R2D2_V2_Full_82D, 3_MeanOnly_NoUncertainty, and 5_Oracle_7x3
    static_fails = 0
    mem_proj_fails = 0
    no_unc_fails = 0
    full_v2_fails = 0
    oracle_fails = 0
    total_rounds = 0

    full_rew_sum = 0.0
    no_unc_rew_sum = 0.0
    total_steps = 0

    for s in range(seed_start, seed_end + 1):
        for r_idx, fam in enumerate(regimes):
            ep_seed = split_mgr.episode_seed("test", s, r_idx)

            ep_full = generate_v2_qec_episode(
                family=fam,
                seed=ep_seed,
                split_name="test",
                num_epochs=num_epochs,
                policy_net=trained_v2_net,
                epsilon=0.0,
                window_rounds=window_rounds,
                observation_mode="full_82d",
            )
            ep_no_unc = generate_v2_qec_episode(
                family=fam,
                seed=ep_seed,
                split_name="test",
                num_epochs=num_epochs,
                policy_net=trained_v2_net,
                epsilon=0.0,
                window_rounds=window_rounds,
                observation_mode="mean_only_no_uncertainty",
            )
            ep_oracle = generate_v2_qec_episode(
                family=fam,
                seed=ep_seed,
                split_name="test",
                num_epochs=num_epochs,
                policy_net=trained_v2_net,
                epsilon=0.0,
                window_rounds=window_rounds,
                observation_mode="oracle_7x3",
            )

            # Memoryless Joint-DP pair projection baseline (fixed action 1: pair_scale=1.15, K=4, zero gains)
            rng_mem = np.random.default_rng(ep_seed + 17)
            from training.environments.decoder_in_loop import sample_d3_circuit_window
            from training.environments.noise_regime_sampler import sample_physical_noise_trajectory
            phys = sample_physical_noise_trajectory(fam, ep_seed, num_epochs=num_epochs)
            for t in range(num_epochs):
                circ = sample_d3_circuit_window(
                    rng=rng_mem,
                    theta_z_21=phys["theta_z_seq"][t],
                    theta_x_21=phys["theta_x_seq"][t],
                    window_rounds=window_rounds,
                    readout_flip_prob=float(phys["readout_flip_seq"][t]),
                    streak3_prob=float(phys["streak3_prob_seq"][t]),
                )
                pm = ep_full.posterior_means[t]
                w_mem = project_q78_weights_d3(
                    qubit_p=pm[:9], alphas=pm[9:], pair_weight_scale=1.15, qubit_gains=None
                )
                corr_mem = decode_window_minsum_osd0_z(circ["sx"], w_mem, bp_iterations=4)
                mem_proj_fails += count_logical_z_failures(circ["err_z"], corr_mem)

            static_fails += int(np.sum(ep_full.static_failures))
            full_v2_fails += int(np.sum(ep_full.decoded_failures))
            no_unc_fails += int(np.sum(ep_no_unc.decoded_failures))
            oracle_fails += int(np.sum(ep_oracle.decoded_failures))
            total_rounds += num_epochs * window_rounds

            full_rew_sum += float(np.sum(ep_full.rewards))
            no_unc_rew_sum += float(np.sum(ep_no_unc.rewards))
            total_steps += num_epochs

    s_ler = static_fails / float(max(1, total_rounds))
    m_ler = mem_proj_fails / float(max(1, total_rounds))
    nu_ler = no_unc_fails / float(max(1, total_rounds))
    f_ler = full_v2_fails / float(max(1, total_rounds))
    o_ler = oracle_fails / float(max(1, total_rounds))

    def _rel(x: float) -> float:
        return float(100.0 * (s_ler - x) / max(s_ler, 1e-9))

    return {
        "1_Static_B0": {"ler": s_ler, "ler_reduction_vs_static_pct": 0.0},
        "2_Memoryless_JointDP_Proj": {"ler": m_ler, "ler_reduction_vs_static_pct": _rel(m_ler)},
        "3_R2D2_V2_NoUncertainty": {
            "ler": nu_ler,
            "ler_reduction_vs_static_pct": _rel(nu_ler),
            "mean_reward": full_rew_sum / max(1, total_steps) if total_steps == 0 else no_unc_rew_sum / total_steps,
        },
        "4_R2D2_V2_Full_82D": {
            "ler": f_ler,
            "ler_reduction_vs_static_pct": _rel(f_ler),
            "mean_reward": full_rew_sum / max(1, total_steps),
        },
        "5_R2D2_V2_Oracle_7x3": {"ler": o_ler, "ler_reduction_vs_static_pct": _rel(o_ler)},
    }
