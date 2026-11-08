"""Transformed Multi-Step Double-DQN TD Loss and Per-Qubit Gain Auxiliary Loss."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from training.models.r2d2_network import (
    R2D2Network,
    value_rescale_h,
    value_rescale_h_inv,
)
from training.replay.sequence_sampler import SequenceSegment


def compute_n_step_transformed_td_loss(
    online_net: R2D2Network,
    target_net: R2D2Network,
    segment: SequenceSegment,
    burn_in_steps: int = 4,
    unroll_steps: int = 4,
    n_step_td: int = 3,
    gamma: float = 0.96,
    is_weight: float = 1.0,
    gain_loss_weight: float = 0.50,
    action_margin_weight: float = 0.35,
    rescaling_eps: float = 1e-3,
) -> Tuple[float, np.ndarray, Dict[str, np.ndarray]]:
    """Compute R2D2 n-step transformed Double-DQN loss and analytical BPTT gradients.

    For each unroll step t in [burn_in_steps, burn_in_steps + unroll_steps):
      1. Double-DQN bootstrap action at t + n:
         a*_{t+n} = argmax_a Q_online(s_{t+n}, a)
      2. Invert value rescaling on target network prediction:
         V_boot = h^{-1}(Q_target(s_{t+n}, a*_{t+n}))
      3. Multi-step return:
         G_t^{(n)} = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n * (1 - done) * V_boot
      4. Rescaled target:
         y_t = h(G_t^{(n)})
      5. TD error:
         delta_t = y_t - Q_online(s_t, a_t)
    """
    T_seq = segment.obs.shape[0]
    online_out = online_net.forward_sequence(
        segment.obs,
        h_init=segment.h_init,
        c_init=segment.c_init,
        burn_in_steps=burn_in_steps,
    )
    target_out = target_net.forward_sequence(
        segment.obs,
        h_init=segment.h_init,
        c_init=segment.c_init,
        burn_in_steps=burn_in_steps,
    )

    q_online = online_out["q_seq"]
    q_target = target_out["q_seq"]
    gain_pred = online_out["gain_seq"]

    dq_seq = np.zeros_like(q_online)
    dgain_seq = np.zeros_like(gain_pred)
    td_errors = np.zeros(unroll_steps, dtype=np.float64)
    total_loss = 0.0

    for u_idx in range(unroll_steps):
        t = burn_in_steps + u_idx
        if t >= T_seq:
            break

        # Compute n-step discounted return
        g_return = 0.0
        discount = 1.0
        terminated = False
        for k in range(n_step_td):
            step_idx = t + k
            if step_idx >= T_seq:
                terminated = True
                break
            g_return += discount * float(segment.rewards[step_idx])
            discount *= gamma
            if segment.dones[step_idx] > 0.5:
                terminated = True
                break

        boot_idx = t + n_step_td
        if not terminated and boot_idx < T_seq:
            best_next_a = int(np.argmax(q_online[boot_idx]))
            q_targ_rescaled = float(q_target[boot_idx, best_next_a])
            v_boot = float(value_rescale_h_inv(q_targ_rescaled, eps=rescaling_eps))
            g_return += discount * v_boot

        y_target = float(value_rescale_h(g_return, eps=rescaling_eps))
        act_t = int(segment.actions[t])
        q_pred = float(q_online[t, act_t])

        # TD error delta_t = y_target - q_pred
        delta_t = y_target - q_pred
        td_errors[u_idx] = delta_t

        # Huber / smooth L1 loss on TD error weighted by importance sampling weight
        abs_delta = abs(delta_t)
        if abs_delta <= 1.0:
            step_td_loss = 0.5 * delta_t * delta_t
            grad_q_act = -delta_t
        else:
            step_td_loss = abs_delta - 0.5
            grad_q_act = -np.sign(delta_t)

        dq_seq[t, act_t] += (is_weight / unroll_steps) * grad_q_act
        total_loss += (is_weight / unroll_steps) * step_td_loss

        # Large-margin classification regularizer (Piot et al. / Hester et al. DQfD margin)
        # Ensures off-policy trajectories respect regime-optimal control boundaries
        if action_margin_weight > 0.0:
            opt_a = int(segment.optimal_actions[t])
            class_balance = 3.2 if opt_a in (3, 5) else 1.2
            margin = 0.35
            for a_other in range(online_net.num_actions):
                if a_other == opt_a:
                    continue
                violation = float(q_online[t, a_other] + margin - q_online[t, opt_a])
                if violation > 0.0:
                    w_m = (is_weight * action_margin_weight * class_balance) / unroll_steps
                    total_loss += 0.5 * w_m * violation * violation
                    dq_seq[t, a_other] += w_m * violation
                    dq_seq[t, opt_a] -= w_m * violation

        # Per-qubit gain residual auxiliary loss
        gain_diff = gain_pred[t] - segment.target_gains[t]
        w_g = (is_weight * gain_loss_weight) / unroll_steps
        total_loss += 0.5 * w_g * float(np.sum(gain_diff * gain_diff))
        dgain_seq[t] += w_g * gain_diff

    grads = online_net.backward_sequence(
        h_seq=online_out["h_seq"],
        caches=online_out["caches"],
        dq_seq=dq_seq,
        dgain_seq=dgain_seq,
        burn_in_steps=burn_in_steps,
    )
    return float(total_loss), td_errors, grads
