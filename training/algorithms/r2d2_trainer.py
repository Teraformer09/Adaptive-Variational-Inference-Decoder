"""R2D2 Trainer with Adam optimizer, gradient clipping, and QEC trajectory generation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from training.models.r2d2_network import ACTION_DEFINITIONS, R2D2Network
from training.replay.sequence_sampler import SequenceSegment, slice_episode_into_segments
from training.replay.prioritized_replay import PrioritizedSequenceReplayBuffer
from training.algorithms.loss import compute_n_step_transformed_td_loss
from training.algorithms.target_network import TargetNetworkManager


def _oracle_action_and_gains_for_state(
    feat: np.ndarray,
    qubit_p: np.ndarray,
    pair_hotspot_mask: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Determine physics-grounded optimal control action and target per-qubit gain residual.

    Features (8-D):
      0: mean_p
      1: max_p
      2: mean_alpha
      3: max_alpha
      4: syndrome_density
      5: adjacent_pair_rate
      6: innovation_norm
      7: burst_score
    """
    mean_p, max_p, mean_alpha, max_alpha, syn_dens, adj_rate, innov, burst = feat

    # Safety fallback when noise estimate is unphysically high / corrupted
    if mean_p > 0.135 or max_p > 0.24 or syn_dens > 0.46:
        opt_action = 5  # CONSERVATIVE_FALLBACK_K4
    # Fast burst / abrupt regime switch requires immediate covariance recalibration + K=8
    elif burst > 0.45 or innov > 0.032:
        opt_action = 4  # FAST_RECALIBRATE_K8
    # Strong pair crosstalk requires aggressive pair projection + K=8
    elif max_alpha >= 0.062 or (mean_alpha >= 0.025 and adj_rate >= 0.085):
        opt_action = 2  # AGGRESSIVE_PAIR_K8
    # Moderate correlated crosstalk / drift benefits from adaptive pair projection + K=6
    elif max_alpha >= 0.030 or mean_alpha >= 0.015:
        opt_action = 1  # ADAPTIVE_PAIR_K6
    # Elevated independent noise with weak pair correlation benefits from damped prior
    elif mean_p >= 0.024 and max_alpha < 0.030:
        opt_action = 3  # DAMPED_PRIOR_K4
    # Calm stationary independent regime: static prior K=4 avoids over-projection and saves cycles
    else:
        opt_action = 0  # STATIC_PRIOR_K4

    # Per-qubit gain residual target in [-0.25, +0.25]^9:
    # Quiet qubits get positive LLR boost (+gain -> stronger prior confidence);
    # Noisy/crosstalk hotspot qubits get negative LLR attenuation (-gain -> lower confidence so BP flips them more easily)
    p_centered = qubit_p - np.mean(qubit_p)
    raw_gain = -8.0 * p_centered - 0.12 * pair_hotspot_mask * (max_alpha / 0.10)
    target_gains = np.clip(raw_gain, -0.24, 0.24).astype(np.float64)
    return opt_action, target_gains


def compute_qec_action_reward(
    feat: np.ndarray,
    action_id: int,
    opt_action: int,
) -> float:
    """Compute QEC control reward combining logical fidelity gain and hardware cycle efficiency."""
    mean_p, max_p, mean_alpha, max_alpha, syn_dens, adj_rate, innov, burst = feat
    act_spec = ACTION_DEFINITIONS[action_id]
    gamma_pair = float(act_spec["pair_weight_scale"])
    k_iters = int(act_spec["bp_iterations"])
    recal = bool(act_spec["recalibrate"])
    fallback = bool(act_spec["fallback"])

    # Ideal pair projection scale for current correlation strength
    if mean_p > 0.135 or max_p > 0.24 or syn_dens > 0.46:
        ideal_gamma = 0.0
        ideal_k = 3
    elif burst > 0.45 or innov > 0.032:
        ideal_gamma = 1.15
        ideal_k = 7
    elif max_alpha >= 0.062 or (mean_alpha >= 0.025 and adj_rate >= 0.085):
        ideal_gamma = 1.35
        ideal_k = 7
    elif max_alpha >= 0.030 or mean_alpha >= 0.015:
        ideal_gamma = 1.00
        ideal_k = 5
    elif mean_p >= 0.024:
        ideal_gamma = 0.85
        ideal_k = 4
    else:
        ideal_gamma = 0.00
        ideal_k = 3

    # 1. Pair projection mismatch penalty
    gamma_err = abs(gamma_pair - ideal_gamma)
    fidelity_term = 1.6 - 2.2 * gamma_err

    # 2. Iteration budget sufficiency vs latency overhead
    if k_iters < ideal_k:
        fidelity_term -= 0.45 * (ideal_k - k_iters)
    else:
        fidelity_term -= 0.08 * (k_iters - ideal_k)

    # 3. Recalibration bonus during bursts / penalty during stationary regimes
    if burst > 0.45 or innov > 0.032:
        fidelity_term += 0.85 if recal else -0.65
    else:
        if recal:
            fidelity_term -= 0.45

    # 4. Fallback safety bonus during unphysical noise / penalty during normal operation
    if mean_p > 0.135 or max_p > 0.24 or syn_dens > 0.46:
        fidelity_term += 1.25 if fallback else -1.20
    else:
        if fallback:
            fidelity_term -= 0.90

    if action_id == opt_action:
        fidelity_term += 0.50

    return float(fidelity_term)


def generate_qec_control_episode(
    family: str,
    seed: int,
    num_epochs: int = 16,
    policy_net: Optional[R2D2Network] = None,
    epsilon: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Generate a multi-epoch QEC syndrome & parameter estimation trajectory."""
    rng = np.random.default_rng(seed)
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    rew_list: List[float] = []
    done_list: List[float] = []
    gain_list: List[np.ndarray] = []
    opt_act_list: List[int] = []
    h_list: List[np.ndarray] = []
    c_list: List[np.ndarray] = []

    if policy_net is not None:
        h_curr, c_curr = policy_net.zero_state()
    else:
        h_curr = np.zeros(64, dtype=np.float64)
        c_curr = np.zeros(64, dtype=np.float64)

    prev_p = np.full(9, 0.015, dtype=np.float64)
    prev_alpha = np.full(12, 0.008, dtype=np.float64)

    for ep in range(num_epochs):
        qubit_p = np.full(9, 0.015, dtype=np.float64)
        alphas = np.full(12, 0.006, dtype=np.float64)
        hotspot_mask = np.zeros(9, dtype=np.float64)
        burst_score = 0.05 + 0.04 * rng.random()

        if family == "stationary_independent":
            base_p = rng.uniform(0.010, 0.019)
            qubit_p = base_p + rng.normal(0.0, 0.0012, size=9)
            alphas = rng.uniform(0.003, 0.012, size=12)
            if ep in (3, 7, 11, 15):
                # Elevated independent depolarization with low pair crosstalk -> Action 3 (DAMPED_PRIOR_K4)
                qubit_p = rng.uniform(0.028, 0.036, size=9)
                alphas = rng.uniform(0.004, 0.011, size=12)
        elif family == "pair_correlated_crosstalk":
            base_p = rng.uniform(0.014, 0.022)
            qubit_p = base_p + rng.normal(0.0, 0.002, size=9)
            hot_pair = int(rng.integers(0, 12))
            hot_val = rng.uniform(0.072, 0.105) if ep % 3 != 0 else rng.uniform(0.038, 0.054)
            alphas = rng.uniform(0.006, 0.016, size=12)
            alphas[hot_pair] = hot_val
            # Mark qubits in hot pair
            q_a = hot_pair % 9
            q_b = (hot_pair + 1) % 9
            hotspot_mask[q_a] = 1.0
            hotspot_mask[q_b] = 1.0
            qubit_p[q_a] += 0.008
            qubit_p[q_b] += 0.008
        elif family == "smooth_sinusoidal_drift":
            phase = 2.0 * np.pi * ep / max(1, num_epochs)
            base_p = 0.017 + 0.006 * np.sin(phase)
            qubit_p = base_p + np.linspace(-0.003, 0.004, 9) + rng.normal(0.0, 0.001, size=9)
            hot_alpha = 0.038 + 0.016 * (0.5 * (1.0 + np.sin(phase)))
            alphas = rng.uniform(0.008, 0.016, size=12)
            alphas[3] = hot_alpha
            hotspot_mask[4] = 1.0
            hotspot_mask[5] = 1.0
        elif family == "abrupt_regime_switch":
            if ep < num_epochs // 2:
                qubit_p = rng.uniform(0.011, 0.017, size=9)
                alphas = rng.uniform(0.004, 0.011, size=12)
            else:
                qubit_p = rng.uniform(0.026, 0.038, size=9)
                alphas = rng.uniform(0.012, 0.025, size=12)
                alphas[3] = rng.uniform(0.075, 0.105)
                hotspot_mask[4] = 1.0
                hotspot_mask[5] = 1.0
                if ep == num_epochs // 2 or ep == num_epochs // 2 + 1:
                    burst_score = rng.uniform(0.62, 0.88)
        elif family == "burst_intermittency_r2d2":
            if ep in (4, 5, 9, 10):
                qubit_p = rng.uniform(0.030, 0.045, size=9)
                alphas = rng.uniform(0.018, 0.032, size=12)
                alphas[int(seed % 12)] = rng.uniform(0.080, 0.115)
                burst_score = rng.uniform(0.65, 0.94)
                hotspot_mask[int(seed % 9)] = 1.0
                hotspot_mask[int((seed + 3) % 9)] = 1.0
            elif ep in (6, 7, 11, 12):
                # Corrupted / safety fallback stress epochs -> Action 5 (CONSERVATIVE_FALLBACK_K4)
                qubit_p = rng.uniform(0.165, 0.250, size=9)
                alphas = rng.uniform(0.095, 0.165, size=12)
                burst_score = 0.15
            else:
                qubit_p = rng.uniform(0.012, 0.018, size=9)
                alphas = rng.uniform(0.005, 0.012, size=12)
        else:
            qubit_p = rng.uniform(0.014, 0.024, size=9)
            alphas = rng.uniform(0.008, 0.045, size=12)

        qubit_p = np.clip(qubit_p, 0.001, 0.30)
        alphas = np.clip(alphas, 0.0, 0.25)

        mean_p = float(np.mean(qubit_p))
        max_p = float(np.max(qubit_p))
        mean_alpha = float(np.mean(alphas))
        max_alpha = float(np.max(alphas))
        syn_dens = float(np.clip(2.4 * mean_p + 1.6 * mean_alpha + rng.normal(0.0, 0.005), 0.01, 0.65))
        adj_rate = float(np.clip(0.9 * max_alpha + 0.8 * mean_alpha + rng.normal(0.0, 0.003), 0.005, 0.40))
        innov = float(
            np.sqrt(np.mean((qubit_p - prev_p) ** 2) + np.mean((alphas - prev_alpha) ** 2))
        )
        prev_p = qubit_p.copy()
        prev_alpha = alphas.copy()

        feat = np.array(
            [mean_p, max_p, mean_alpha, max_alpha, syn_dens, adj_rate, innov, burst_score],
            dtype=np.float64,
        )
        opt_act, targ_gains = _oracle_action_and_gains_for_state(feat, qubit_p, hotspot_mask)

        h_list.append(h_curr.copy())
        c_list.append(c_curr.copy())

        if policy_net is not None:
            step_out = policy_net.step(feat, h_curr, c_curr)
            h_curr = step_out["h_next"]
            c_curr = step_out["c_next"]
            if rng.random() < epsilon:
                chosen_act = int(rng.integers(0, policy_net.num_actions))
            else:
                chosen_act = int(step_out["action_id"])
        else:
            chosen_act = opt_act if rng.random() > epsilon else int(rng.integers(0, 6))

        rew = compute_qec_action_reward(feat, chosen_act, opt_act)
        done = 1.0 if (ep == num_epochs - 1) else 0.0

        obs_list.append(feat)
        act_list.append(chosen_act)
        rew_list.append(rew)
        done_list.append(done)
        gain_list.append(targ_gains)
        opt_act_list.append(opt_act)

    return {
        "obs": np.asarray(obs_list, dtype=np.float64),
        "actions": np.asarray(act_list, dtype=np.int64),
        "rewards": np.asarray(rew_list, dtype=np.float64),
        "dones": np.asarray(done_list, dtype=np.float64),
        "target_gains": np.asarray(gain_list, dtype=np.float64),
        "optimal_actions": np.asarray(opt_act_list, dtype=np.int64),
        "h_states": np.asarray(h_list, dtype=np.float64),
        "c_states": np.asarray(c_list, dtype=np.float64),
    }


class R2D2Trainer:
    """Recurrent R2D2 Trainer with Adam optimizer, target network, and prioritized sequence replay."""

    def __init__(
        self,
        online_net: R2D2Network,
        learning_rate: float = 0.0025,
        gamma: float = 0.96,
        burn_in_steps: int = 4,
        unroll_steps: int = 4,
        n_step_td: int = 3,
        target_sync_interval: int = 25,
        grad_clip_norm: float = 5.0,
        replay_capacity: int = 4096,
        seed: int = 20260405,
    ) -> None:
        self.online_net = online_net
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
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.burn_in_steps = int(burn_in_steps)
        self.unroll_steps = int(unroll_steps)
        self.n_step_td = int(n_step_td)
        self.grad_clip_norm = float(grad_clip_norm)

        # Adam optimizer moments
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
        seed_end: int = 1039,
        epsilon: float = 0.35,
    ) -> int:
        """Collect episodes across training seeds and slice into replay buffer."""
        added = 0
        for s in range(seed_start, seed_end + 1):
            for fam_idx, fam in enumerate(families):
                ep_seed = s * 31 + fam_idx * 7
                # Collect one exploratory trajectory and one behavior-guided trajectory
                for eps_val in (epsilon, 0.05):
                    ep = generate_qec_control_episode(
                        family=fam,
                        seed=ep_seed + int(eps_val * 100),
                        num_epochs=16,
                        policy_net=self.online_net,
                        epsilon=eps_val,
                    )
                    segs = slice_episode_into_segments(
                        obs=ep["obs"],
                        actions=ep["actions"],
                        rewards=ep["rewards"],
                        dones=ep["dones"],
                        target_gains=ep["target_gains"],
                        optimal_actions=ep["optimal_actions"],
                        h_states=ep["h_states"],
                        c_states=ep["c_states"],
                        burn_in_steps=self.burn_in_steps,
                        unroll_steps=self.unroll_steps,
                        n_step_td=self.n_step_td,
                        stride=2,
                        family=fam,
                        seed=s,
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
