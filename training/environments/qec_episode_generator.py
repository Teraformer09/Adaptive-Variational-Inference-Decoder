"""Syndrome-Conditioned QEC Control Episode Generator (`qec_episode_generator.py`).

Enforces the strict non-leaky causal pipeline:
  Layer 1 physical trajectory -> Circuit syndrome history -> Layer 3 posterior estimator ->
  82-D observation -> R2D2 policy -> Decoder-in-the-loop -> Noise-aware reward.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

# Discover sibling Layer 3 checkout (`../layer3` or `../avid-layer3`) when available
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sib in ("layer3", "avid-layer3"):
    _cand = _REPO_ROOT.parent / _sib
    if (_cand / "layer3" / "__init__.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

try:
    from layer3.inference.posterior_estimator import SequentialPosteriorEstimator
except ImportError:  # Standalone fallback if Layer 2 is cloned in isolation without sibling Layer 3
    SequentialPosteriorEstimator = None  # type: ignore

from training.data.episode_dataset import EpisodeRecord
from training.environments.decoder_in_loop import (
    CROSSTALK_PAIRS,
    EVENT_SIGS_X,
    STATIC_LUT_Z,
    count_logical_z_failures,
    decode_window_minsum_osd0_z,
    project_q78_weights_d3,
    sample_d3_circuit_window,
)
from training.environments.noise_regime_sampler import sample_physical_noise_trajectory
from training.models.r2d2_network import ACTION_DEFINITIONS, R2D2Network
from training.observations.observation_builder import ObservationBuilder
from training.observations.observation_schema import (
    OBS_V2_DIM,
    SLICE_POSTERIOR_LOG_VAR,
    SLICE_POSTERIOR_MEAN,
)
from training.observations.syndrome_histogram import compute_basis_syndrome_stats
from training.rewards.noise_aware_reward import compute_noise_aware_reward


class _StandaloneSyndromeLaplaceAdapter:
    """Minimal syndrome-only estimator used ONLY if sibling `layer3` repo is absent."""

    def __init__(self, temporal_beta: float = 0.55) -> None:
        self.temporal_beta = float(temporal_beta)
        self._ema_hx: Optional[np.ndarray] = None
        self._prev_mean: Optional[np.ndarray] = None
        self._ep = 0

    def configure_from_action(self, temporal_beta: float, recalibrate: bool = False) -> None:
        self.temporal_beta = float(np.clip(temporal_beta, 0.0, 0.95))
        if recalibrate:
            self._ema_hx = None

    def step_from_syndrome_windows(
        self, sx: np.ndarray, sz: np.ndarray, recalibrate: bool = False
    ) -> Dict[str, object]:
        st_x = compute_basis_syndrome_stats(sx)
        st_z = compute_basis_syndrome_stats(sz)
        hx = np.asarray(st_x["histogram"], dtype=np.float64)
        hz = np.asarray(st_z["histogram"], dtype=np.float64)
        if recalibrate or self._ema_hx is None or self.temporal_beta <= 0.0:
            self._ema_hx = hx.copy()
        else:
            self._ema_hx = self.temporal_beta * self._ema_hx + (1.0 - self.temporal_beta) * hx

        # Syndrome-signature moment back-projection (purely from observed syndromes!)
        mean_21 = np.full(21, 0.012, dtype=np.float64)
        var_21 = np.full(21, 1.5e-4, dtype=np.float64)
        for m in range(21):
            sig = int(EVENT_SIGS_X[m])
            if sig == 0:
                mean_21[m] = 0.005
                var_21[m] = 8.5e-4
            else:
                p_sig = float(self._ema_hx[sig])
                mean_21[m] = float(np.clip(0.55 * p_sig + 0.006, 0.002, 0.35))
                var_21[m] = float(max(2.0e-5, p_sig * (1.0 - p_sig) / max(len(sx), 1)))
        innov = 0.0 if self._prev_mean is None else float(np.linalg.norm(mean_21 - self._prev_mean))
        self._prev_mean = mean_21.copy()
        self._ep += 1
        return {
            "epoch_id": self._ep - 1,
            "posterior_mean_21": mean_21,
            "posterior_var_21": var_21,
            "hist_x_16": hx,
            "hist_z_16": hz,
            "valid_rounds_x": int(st_x["valid_rounds"]),
            "valid_rounds_z": int(st_z["valid_rounds"]),
            "estimator_residual_tvd": 0.02,
            "posterior_innovation_l2": innov,
            "burst_score": -1.0,
        }


def _compute_best_reference_action_and_gains(
    post_mean_21: np.ndarray,
    true_theta_z_21: np.ndarray,
    err_z_win: np.ndarray,
    sx_win: np.ndarray,
    regime: str,
    prev_action_id: Optional[int],
    prev_true_theta_z_21: Optional[np.ndarray],
    in_burst: bool,
    post_burst: bool,
    is_corrupted: bool,
) -> Tuple[int, np.ndarray]:
    """Compute target per-qubit gain residuals and reference best action via simulation decoder sweep."""
    qp_true = true_theta_z_21[:9]
    pair_true = true_theta_z_21[9:]
    p_centered = qp_true - float(np.mean(qp_true))

    hotspot_mask = np.zeros(9, dtype=np.float64)
    for k, (u, v) in enumerate(CROSSTALK_PAIRS):
        if pair_true[k] >= 0.040:
            hotspot_mask[u] = 1.0
            hotspot_mask[v] = 1.0

    max_pair = float(np.max(pair_true))
    raw_gain = -7.5 * p_centered - 0.13 * hotspot_mask * max(0.35, min(1.5, max_pair / 0.08))
    target_gains = np.clip(raw_gain, -0.24, 0.24).astype(np.float64)
    gains_mult = np.clip(1.0 + target_gains, 0.72, 1.28)

    n_rounds = max(1, len(sx_win))
    f_static = count_logical_z_failures(err_z_win, STATIC_LUT_Z[sx_win])
    ler_ref = f_static / float(n_rounds)

    best_act = 0
    best_rew = -1e9
    for a in range(len(ACTION_DEFINITIONS)):
        spec = ACTION_DEFINITIONS[a]
        w_a = project_q78_weights_d3(
            qubit_p=post_mean_21[:9],
            alphas=post_mean_21[9:],
            pair_weight_scale=float(spec["pair_weight_scale"]),
            qubit_gains=gains_mult if float(spec["pair_weight_scale"]) > 0.0 else None,
            fallback=bool(spec["fallback"]),
        )
        corr_a = decode_window_minsum_osd0_z(
            sx_win, w_a, bp_iterations=int(spec["bp_iterations"])
        )
        ler_a = count_logical_z_failures(err_z_win, corr_a) / float(n_rounds)
        r_dict = compute_noise_aware_reward(
            action_id=a,
            qubit_gains=target_gains,
            ler_action=ler_a,
            ler_reference=ler_ref,
            regime=regime,
            prev_action_id=prev_action_id,
            prev_qubit_gains=target_gains,
            true_theta_z_21=true_theta_z_21,
            prev_true_theta_z_21=prev_true_theta_z_21,
            in_burst=in_burst,
            post_burst_recovery=post_burst,
            is_corrupted=is_corrupted,
        )
        if r_dict["reward/total_reward"] > best_rew:
            best_rew = r_dict["reward/total_reward"]
            best_act = a

    return best_act, target_gains


def generate_v2_qec_episode(
    family: str,
    seed: int,
    split_name: str = "train",
    num_epochs: int = 16,
    policy_net: Optional[R2D2Network] = None,
    epsilon: float = 0.25,
    window_rounds: int = 256,
    observation_mode: str = "full_82d",
) -> EpisodeRecord:
    """Run a multi-epoch syndrome-conditioned QEC control episode.

    `observation_mode` supports controlled ablations (EVAL-02):
      - `"full_82d"`: Deployable Layer 3 posterior mean + log-variance + X/Z histograms + telemetry
      - `"mean_only_no_uncertainty"`: Zeros out the 21 posterior log-variance features
      - `"oracle_7x3"`: Replaces posterior mean with ground-truth 7x3 rates (privileged upper-bound ablation ONLY)
    """
    rng_circ = np.random.default_rng(int(seed) + 17)
    rng_pol = np.random.default_rng(int(seed) + 991)
    phys = sample_physical_noise_trajectory(family=family, seed=seed, num_epochs=num_epochs)
    regime = str(phys["canonical_regime"])
    theta_z_seq = np.asarray(phys["theta_z_seq"], dtype=np.float64)
    theta_x_seq = np.asarray(phys["theta_x_seq"], dtype=np.float64)
    in_burst_seq = np.asarray(phys["in_burst_seq"], dtype=bool)
    post_burst_seq = np.asarray(phys["post_burst_seq"], dtype=bool)
    is_corrupted_seq = np.asarray(phys["is_corrupted_seq"], dtype=bool)
    readout_flip_seq = np.asarray(phys["readout_flip_seq"], dtype=np.float64)
    streak3_prob_seq = np.asarray(phys["streak3_prob_seq"], dtype=np.float64)

    estimator = (
        SequentialPosteriorEstimator(temporal_beta=0.55)
        if SequentialPosteriorEstimator is not None
        else _StandaloneSyndromeLaplaceAdapter(temporal_beta=0.55)
    )
    obs_builder = ObservationBuilder(max_config_age_epochs=num_epochs)

    if policy_net is not None:
        h_curr, c_curr = policy_net.zero_state()
    else:
        h_curr = np.zeros(64, dtype=np.float64)
        c_curr = np.zeros(64, dtype=np.float64)

    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    rew_list: List[float] = []
    done_list: List[float] = []
    gain_list: List[np.ndarray] = []
    opt_act_list: List[int] = []
    h_list: List[np.ndarray] = []
    c_list: List[np.ndarray] = []
    post_mean_list: List[np.ndarray] = []
    post_var_list: List[np.ndarray] = []
    dec_fail_list: List[int] = []
    stat_fail_list: List[int] = []
    rew_terms_list: List[Dict[str, float]] = []

    prev_act: Optional[int] = None
    prev_gains: Optional[np.ndarray] = None
    recal_next = False

    for ep in range(num_epochs):
        th_z = theta_z_seq[ep]
        th_x = theta_x_seq[ep]

        circ = sample_d3_circuit_window(
            rng=rng_circ,
            theta_z_21=th_z,
            theta_x_21=th_x,
            window_rounds=window_rounds,
            readout_flip_prob=float(readout_flip_seq[ep]),
            streak3_prob=float(streak3_prob_seq[ep]),
        )
        err_z_win = circ["err_z"]
        sx_win = circ["sx"]
        sz_win = circ["sz"]

        pkt_or_dict = estimator.step_from_syndrome_windows(
            sx_win, sz_win, recalibrate=recal_next
        )
        pkt_dict = (
            pkt_or_dict.to_layer2_dict()
            if hasattr(pkt_or_dict, "to_layer2_dict")
            else dict(pkt_or_dict)
        )
        post_mean_21 = np.asarray(pkt_dict["posterior_mean_21"], dtype=np.float64)
        post_var_21 = np.asarray(pkt_dict["posterior_var_21"], dtype=np.float64)

        built = obs_builder.build_from_posterior_and_syndromes(
            posterior_packet=pkt_dict,
            syndromes_x=sx_win,
            syndromes_z=sz_win,
        )
        obs_82 = built["observation_82"].copy()

        if observation_mode == "mean_only_no_uncertainty":
            obs_82[SLICE_POSTERIOR_LOG_VAR] = -9.21034  # constant neutral log-variance
        elif observation_mode == "oracle_7x3":
            obs_82[SLICE_POSTERIOR_MEAN] = np.clip(th_z, 0.0, 0.49)
            obs_82[SLICE_POSTERIOR_LOG_VAR] = -16.0

        prev_th_z = theta_z_seq[ep - 1] if ep > 0 else None
        opt_act, targ_gains = _compute_best_reference_action_and_gains(
            post_mean_21=post_mean_21 if observation_mode != "oracle_7x3" else th_z,
            true_theta_z_21=th_z,
            err_z_win=err_z_win,
            sx_win=sx_win,
            regime=regime,
            prev_action_id=prev_act,
            prev_true_theta_z_21=prev_th_z,
            in_burst=bool(in_burst_seq[ep]),
            post_burst=bool(post_burst_seq[ep]),
            is_corrupted=bool(is_corrupted_seq[ep]),
        )

        h_list.append(h_curr.copy())
        c_list.append(c_curr.copy())

        if policy_net is not None and policy_net.input_dim == OBS_V2_DIM:
            step_out = policy_net.step(obs_82, h_curr, c_curr)
            h_curr = step_out["h_next"]
            c_curr = step_out["c_next"]
            if rng_pol.random() < epsilon:
                chosen_act = int(rng_pol.integers(0, policy_net.num_actions))
            else:
                chosen_act = int(step_out["action_id"])
            raw_policy_gains = np.asarray(step_out["qubit_gains"], dtype=np.float64)
        else:
            chosen_act = opt_act if rng_pol.random() > epsilon else int(rng_pol.integers(0, 6))
            raw_policy_gains = targ_gains.copy()

        gains_mult = np.clip(1.0 + raw_policy_gains, 0.70, 1.30)
        act_spec = ACTION_DEFINITIONS[chosen_act]

        # Project weights strictly from Layer 3's posterior_mean_21 (or oracle_7x3 only in oracle ablation)
        infer_source = th_z if observation_mode == "oracle_7x3" else post_mean_21
        is_corrupted_obs = float(obs_82[74]) > 0.44 or float(obs_82[80]) > 0.22
        w_chosen = project_q78_weights_d3(
            qubit_p=infer_source[:9],
            alphas=infer_source[9:],
            pair_weight_scale=float(act_spec["pair_weight_scale"]),
            qubit_gains=gains_mult if float(act_spec["pair_weight_scale"]) > 0.0 else None,
            fallback=bool(act_spec["fallback"]),
        )
        if bool(is_corrupted_seq[ep]) and not bool(act_spec["fallback"]) and is_corrupted_obs:
            # Corrupted unphysical estimator epoch without safety fallback degrades weight bank
            w_chosen = np.clip(w_chosen // 4, 16, 0x3FFF)

        corr_chosen = decode_window_minsum_osd0_z(
            sx_win, w_chosen, bp_iterations=int(act_spec["bp_iterations"])
        )
        f_chosen = count_logical_z_failures(err_z_win, corr_chosen)
        f_static = count_logical_z_failures(err_z_win, STATIC_LUT_Z[sx_win])

        ler_chosen = f_chosen / float(window_rounds)
        ler_static = f_static / float(window_rounds)

        r_terms = compute_noise_aware_reward(
            action_id=chosen_act,
            qubit_gains=raw_policy_gains,
            ler_action=ler_chosen,
            ler_reference=ler_static,
            regime=regime,
            prev_action_id=prev_act,
            prev_qubit_gains=prev_gains,
            true_theta_z_21=th_z,
            prev_true_theta_z_21=prev_th_z,
            in_burst=bool(in_burst_seq[ep]),
            post_burst_recovery=bool(post_burst_seq[ep]),
            is_corrupted=bool(is_corrupted_seq[ep]),
        )

        config_changed = (prev_act is not None) and (chosen_act != prev_act)
        obs_builder.record_action_update(config_changed=config_changed)
        estimator.configure_from_action(
            temporal_beta=float(act_spec.get("temporal_beta", 0.55)),
            recalibrate=bool(act_spec["recalibrate"]),
        )
        recal_next = bool(act_spec["recalibrate"])
        prev_act = chosen_act
        prev_gains = raw_policy_gains.copy()

        obs_list.append(obs_82)
        act_list.append(chosen_act)
        rew_list.append(float(r_terms["reward/total_reward"]))
        done_list.append(1.0 if ep == num_epochs - 1 else 0.0)
        gain_list.append(targ_gains)
        opt_act_list.append(opt_act)
        post_mean_list.append(post_mean_21)
        post_var_list.append(post_var_21)
        dec_fail_list.append(f_chosen)
        stat_fail_list.append(f_static)
        rew_terms_list.append(r_terms)

    traj_id = f"{split_name}_{family}_s{seed}_{observation_mode}"
    return EpisodeRecord(
        trajectory_id=traj_id,
        split_name=split_name,
        regime=regime,
        seed=int(seed),
        obs=np.asarray(obs_list, dtype=np.float64),
        actions=np.asarray(act_list, dtype=np.int64),
        rewards=np.asarray(rew_list, dtype=np.float64),
        dones=np.asarray(done_list, dtype=np.float64),
        target_gains=np.asarray(gain_list, dtype=np.float64),
        optimal_actions=np.asarray(opt_act_list, dtype=np.int64),
        h_states=np.asarray(h_list, dtype=np.float64),
        c_states=np.asarray(c_list, dtype=np.float64),
        posterior_means=np.asarray(post_mean_list, dtype=np.float64),
        posterior_vars=np.asarray(post_var_list, dtype=np.float64),
        true_theta_z=theta_z_seq,
        decoded_failures=np.asarray(dec_fail_list, dtype=np.int64),
        static_failures=np.asarray(stat_fail_list, dtype=np.int64),
        window_rounds=window_rounds,
        reward_terms=rew_terms_list,
        metadata={"family": family, "observation_mode": observation_mode},
    )
