"""R2D2 Trainer refactored to consume 82-D syndrome-conditioned episodes and `training/rewards/`."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from training.algorithms.loss import compute_n_step_transformed_td_loss
from training.algorithms.target_network import TargetNetworkManager
from training.data.trajectory_split import TrajectorySplitManager
from training.environments.decoder_in_loop import (
    CROSSTALK_PAIRS,
    EVENT_SIGS_X,
    STATIC_LUT_Z,
    build_hardware_osd_lut_z,
    count_logical_z_failures,
    project_q78_weights_d3,
)
from training.environments.qec_episode_generator import generate_v2_qec_episode
from training.models.r2d2_network import ACTION_DEFINITIONS, R2D2Network
from training.replay.prioritized_replay import PrioritizedSequenceReplayBuffer
from training.replay.sequence_sampler import SequenceSegment, slice_episode_into_segments
from training.rewards.noise_aware_reward import compute_noise_aware_reward
from training.rewards.reward_diagnostics import RewardDiagnosticsTracker


def compute_qec_action_reward(
    action_id: int,
    qubit_gains: Optional[np.ndarray] = None,
    ler_action: float = 0.03,
    ler_reference: float = 0.05,
    regime: str = "gaussian_stationary",
    prev_action_id: Optional[int] = None,
    in_burst: bool = False,
    post_burst_recovery: bool = False,
    is_corrupted: bool = False,
) -> float:
    """Delegate reward calculation strictly to `training.rewards.noise_aware_reward` (no oracle bonus)."""
    gains = np.zeros(9, dtype=np.float64) if qubit_gains is None else np.asarray(qubit_gains, dtype=np.float64)
    res = compute_noise_aware_reward(
        action_id=action_id,
        qubit_gains=gains,
        ler_action=ler_action,
        ler_reference=ler_reference,
        regime=regime,
        prev_action_id=prev_action_id,
        in_burst=in_burst,
        post_burst_recovery=post_burst_recovery,
        is_corrupted=is_corrupted,
    )
    return float(res["reward/total_reward"])


def generate_qec_control_episode(
    family: str,
    seed: int,
    num_epochs: int = 16,
    policy_net: Optional[R2D2Network] = None,
    epsilon: float = 0.25,
    window_rounds: int = 256,
    split_name: str = "train",
    observation_mode: str = "full_82d",
) -> Dict[str, np.ndarray]:
    """Generate a syndrome-conditioned 82-D QEC control episode with zero ground-truth leakage."""
    record = generate_v2_qec_episode(
        family=family,
        seed=seed,
        split_name=split_name,
        num_epochs=num_epochs,
        policy_net=policy_net,
        epsilon=epsilon,
        window_rounds=window_rounds,
        observation_mode=observation_mode,
    )
    return {
        "trajectory_id": record.trajectory_id,
        "obs": record.obs,
        "actions": record.actions,
        "rewards": record.rewards,
        "dones": record.dones,
        "target_gains": record.target_gains,
        "optimal_actions": record.optimal_actions,
        "h_states": record.h_states,
        "c_states": record.c_states,
        "posterior_means": record.posterior_means,
        "posterior_vars": record.posterior_vars,
        "true_theta_z": record.true_theta_z,
        "decoded_failures": record.decoded_failures,
        "static_failures": record.static_failures,
        "reward_terms": record.reward_terms,
    }


class R2D2Trainer:
    """Recurrent R2D2 Trainer with Adam optimizer, target network, and prioritized sequence replay."""

    def __init__(
        self,
        online_net: R2D2Network,
        learning_rate: float = 0.0022,
        gamma: float = 0.96,
        burn_in_steps: int = 4,
        unroll_steps: int = 4,
        n_step_td: int = 3,
        target_sync_interval: int = 25,
        grad_clip_norm: float = 5.0,
        replay_capacity: int = 4096,
        observation_mode: str = "full_82d",
        seed: int = 20260405,
    ) -> None:
        self.online_net = online_net
        self.observation_mode = str(observation_mode)
        self.target_mgr = TargetNetworkManager(
            online_net=online_net,
            sync_interval=target_sync_interval,
            tau=1.0,
        )
        self.replay = PrioritizedSequenceReplayBuffer(
            capacity=replay_capacity,
            alpha=0.6,
            eta=0.9,
            seed=seed,
        )
        self.split_mgr = TrajectorySplitManager()
        self.reward_tracker = RewardDiagnosticsTracker()

        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.burn_in_steps = int(burn_in_steps)
        self.unroll_steps = int(unroll_steps)
        self.n_step_td = int(n_step_td)
        self.grad_clip_norm = float(grad_clip_norm)

        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8
        self.m: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.online_net.parameters().items()
        }
        self.v: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.online_net.parameters().items()
        }

    def populate_replay_from_seeds(
        self,
        families: List[str],
        seed_start: int = 1000,
        seed_end: int = 1024,
        epsilon: float = 0.35,
        window_rounds: int = 256,
    ) -> int:
        """Collect syndrome-conditioned episodes across training seeds and slice into replay."""
        added = 0
        for s in range(seed_start, seed_end + 1):
            for fam_idx, fam in enumerate(families):
                ep_seed = self.split_mgr.episode_seed("train", s, fam_idx)
                for eps_idx, eps_val in enumerate((epsilon, 0.05)):
                    rec = generate_v2_qec_episode(
                        family=fam,
                        seed=ep_seed + eps_idx * 1000,
                        split_name="train",
                        num_epochs=14,
                        policy_net=self.online_net if eps_idx == 0 else None,
                        epsilon=eps_val,
                        window_rounds=window_rounds,
                        observation_mode=self.observation_mode,
                    )
                    for rt in rec.reward_terms:
                        self.reward_tracker.log_step(rt, regime=rec.regime)
                    segs = slice_episode_into_segments(
                        obs=rec.obs,
                        actions=rec.actions,
                        rewards=rec.rewards,
                        dones=rec.dones,
                        target_gains=rec.target_gains,
                        optimal_actions=rec.optimal_actions,
                        h_states=rec.h_states,
                        c_states=rec.c_states,
                        burn_in_steps=self.burn_in_steps,
                        unroll_steps=self.unroll_steps,
                        n_step_td=self.n_step_td,
                        stride=2,
                        family=rec.regime,
                        seed=s,
                        trajectory_id=rec.trajectory_id,
                    )
                    for seg in segs:
                        self.replay.add(seg)
                        added += 1
        return added

    def _apply_adam_step(self, grads: Dict[str, np.ndarray]) -> float:
        """Clip global gradient norm and apply one Adam parameter update."""
        total_sq = sum(float(np.sum(g * g)) for g in grads.values())
        grad_norm = float(np.sqrt(max(total_sq, 0.0)))
        clip_scale = 1.0
        if grad_norm > self.grad_clip_norm:
            clip_scale = self.grad_clip_norm / (grad_norm + 1e-12)

        self.adam_t += 1
        b1_corr = 1.0 - (self.beta1 ** self.adam_t)
        b2_corr = 1.0 - (self.beta2 ** self.adam_t)

        params = self.online_net.parameters()
        for k, param in params.items():
            g = grads[k] * clip_scale
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[k] / b1_corr
            v_hat = self.v[k] / b2_corr
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)

        return grad_norm

    def train_step(self, batch_size: int = 16, beta: float = 0.6) -> Dict[str, float]:
        """Sample a prioritized batch of sequences, compute BPTT gradients, and update weights."""
        batch, indices, is_weights = self.replay.sample(batch_size=batch_size, beta=beta)
        accum_grads = {
            k: np.zeros_like(v) for k, v in self.online_net.parameters().items()
        }
        batch_loss = 0.0
        seq_td_errors: List[np.ndarray] = []

        for seg, w_i in zip(batch, is_weights):
            loss_i, td_err_i, grads_i = compute_n_step_transformed_td_loss(
                online_net=self.online_net,
                target_net=self.target_mgr.target_net,
                segment=seg,
                burn_in_steps=self.burn_in_steps,
                unroll_steps=self.unroll_steps,
                n_step_td=self.n_step_td,
                gamma=self.gamma,
                is_weight=float(w_i),
            )
            batch_loss += loss_i / len(batch)
            seq_td_errors.append(td_err_i)
            for k in accum_grads:
                accum_grads[k] += grads_i[k] / len(batch)

        self.replay.update_priorities(indices, seq_td_errors)
        grad_norm = self._apply_adam_step(accum_grads)
        synced = self.target_mgr.step()
        mean_abs_td = float(np.mean([np.mean(np.abs(e)) for e in seq_td_errors]))

        return {
            "loss": float(batch_loss),
            "mean_abs_td": mean_abs_td,
            "grad_norm": grad_norm,
            "target_synced": 1.0 if synced else 0.0,
        }
