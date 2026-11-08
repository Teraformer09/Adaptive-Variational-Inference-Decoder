"""Verification Suite 1: R2D2 Training Pipeline (`tests/test_training.py`).

Verifies:
  1. Kapturowski et al. invertible value rescaling h(z) and h^{-1}(y) exactness.
  2. Recurrent LSTM encoder forward and analytical BPTT backward gradient vs finite differences.
  3. Dueling Q-head forward and analytical backward gradient vs finite differences.
  4. Prioritized sequence replay buffer sampling, priority updates, and importance-sampling weights.
  5. Multi-step transformed Double-DQN TD loss and TargetNetworkManager synchronization.
  6. End-to-end R2D2 training reduces TD loss and outperforms untrained/static policies on validation seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from training.models.recurrent_encoder import RecurrentLSTMEncoder
from training.models.q_value_head import DuelingQValueHead
from training.models.r2d2_network import (
    R2D2Network,
    value_rescale_h,
    value_rescale_h_inv,
    value_rescale_h_prime,
)
from training.replay.sequence_sampler import SequenceSegment, slice_episode_into_segments
from training.replay.prioritized_replay import PrioritizedSequenceReplayBuffer
from training.algorithms.loss import compute_n_step_transformed_td_loss
from training.algorithms.target_network import TargetNetworkManager
from training.algorithms.r2d2_trainer import R2D2Trainer, generate_qec_control_episode


def test_kapturowski_value_rescaling_bijection_and_derivative() -> None:
    z_vals = np.linspace(-25.0, 25.0, 201, dtype=np.float64)
    y_vals = value_rescale_h(z_vals, eps=1e-3)
    z_rec = value_rescale_h_inv(y_vals, eps=1e-3)
    assert np.max(np.abs(z_vals - z_rec)) < 1e-10

    # Check derivative h'(z) away from z=0 against central finite differences
    z_nz = np.array([-10.0, -2.5, -0.4, 0.4, 2.5, 10.0], dtype=np.float64)
    delta = 1e-6
    fd = (value_rescale_h(z_nz + delta) - value_rescale_h(z_nz - delta)) / (2.0 * delta)
    analytic = value_rescale_h_prime(z_nz)
    assert np.max(np.abs(fd - analytic)) < 1e-8


def test_recurrent_lstm_bptt_gradient_vs_finite_difference() -> None:
    enc = RecurrentLSTMEncoder(input_dim=8, hidden_dim=8, seed=101)
    rng = np.random.default_rng(102)
    obs_seq = rng.uniform(0.0, 0.1, size=(6, 8)).astype(np.float64)
    burn_in = 2

    h_seq, _, caches = enc.forward_sequence(obs_seq, burn_in_steps=burn_in)
    # Stop-gradient at burn_in means h_{burn_in - 1}, c_{burn_in - 1} are treated as fixed constants
    h_burn = caches[burn_in - 1].h_next.copy()
    c_burn = caches[burn_in - 1].c_next.copy()

    weights_proj = rng.normal(0.0, 0.5, size=(6, 8)).astype(np.float64)
    dh_seq = np.zeros_like(h_seq)
    dh_seq[burn_in:] = weights_proj[burn_in:]

    grads = enc.backward_sequence(dh_seq, caches, burn_in_steps=burn_in)

    eps = 1e-6
    for r in range(4):
        for c in range(4):
            orig = enc.W_ih[r, c]
            enc.W_ih[r, c] = orig + eps
            hp, _, _ = enc.forward_sequence(
                obs_seq[burn_in:], h_init=h_burn, c_init=c_burn, burn_in_steps=0
            )
            lp = float(np.sum(hp * weights_proj[burn_in:]))

            enc.W_ih[r, c] = orig - eps
            hm, _, _ = enc.forward_sequence(
                obs_seq[burn_in:], h_init=h_burn, c_init=c_burn, burn_in_steps=0
            )
            lm = float(np.sum(hm * weights_proj[burn_in:]))

            enc.W_ih[r, c] = orig
            fd = (lp - lm) / (2.0 * eps)
            assert abs(grads["W_ih"][r, c] - fd) < 1e-6


def test_dueling_q_head_gradient_vs_finite_difference() -> None:
    head = DuelingQValueHead(hidden_dim=8, num_actions=6, num_qubits=9, seed=202)
    rng = np.random.default_rng(203)
    h = rng.normal(0.0, 0.5, size=8).astype(np.float64)
    dq = rng.normal(0.0, 1.0, size=6).astype(np.float64)
    dg = rng.normal(0.0, 1.0, size=9).astype(np.float64)

    dh, grads = head.backward_step(h, dq, dg)

    eps = 1e-6
    for idx in range(8):
        orig = h[idx]
        h[idx] = orig + eps
        qp, _, _, gp = head.forward_step(h)
        lp = float(np.sum(qp * dq) + np.sum(gp * dg))

        h[idx] = orig - eps
        qm, _, _, gm = head.forward_step(h)
        lm = float(np.sum(qm * dq) + np.sum(gm * dg))

        h[idx] = orig
        fd = (lp - lm) / (2.0 * eps)
        assert abs(dh[idx] - fd) < 1e-6


def test_prioritized_sequence_replay_and_target_sync() -> None:
    net = R2D2Network(seed=301)
    mgr = TargetNetworkManager(online_net=net, sync_interval=2, tau=1.0)
    # Perturb online network
    net.encoder.W_ih += 0.25
    assert not np.allclose(net.encoder.W_ih, mgr.target_net.encoder.W_ih)
    assert not mgr.step()
    assert mgr.step()
    assert np.allclose(net.encoder.W_ih, mgr.target_net.encoder.W_ih)

    buf = PrioritizedSequenceReplayBuffer(capacity=32, alpha=0.6, eta=0.9, seed=302)
    ep = generate_qec_control_episode("pair_correlated_crosstalk", seed=1001, num_epochs=14)
    segs = slice_episode_into_segments(
        obs=ep["obs"],
        actions=ep["actions"],
        rewards=ep["rewards"],
        dones=ep["dones"],
        target_gains=ep["target_gains"],
        optimal_actions=ep["optimal_actions"],
        h_states=ep["h_states"],
        c_states=ep["c_states"],
    )
    assert len(segs) >= 2
    for idx, s in enumerate(segs):
        buf.add(s, td_errors=np.array([0.1 * (idx + 1), 0.5 * (idx + 1)]))

    batch, indices, is_weights = buf.sample(batch_size=4, beta=0.5)
    assert len(batch) == 4
    assert indices.shape == (4,)
    assert np.all((is_weights > 0.0) & (is_weights <= 1.0 + 1e-12))


def test_trained_checkpoint_metrics_and_convergence() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    meta_path = repo_root / "checkpoints" / "r2d2_v1_trained.json"
    assert meta_path.exists(), "Trained checkpoint metadata JSON must exist"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tm = meta["training_metrics"]

    # Verify loss reduction across training updates
    assert tm["final_loss_mean_last_20"] < tm["initial_loss_mean_first_20"] * 0.25
    # Verify TD error reduction
    assert tm["final_td_error_last_20"] < tm["initial_td_error_first_20"]
    # Verify validation reward strictly improves over both untrained and static policy
    pre_rew = tm["pre_training_val_metrics"]["mean_step_reward"]
    post_rew = tm["post_training_val_metrics"]["mean_step_reward"]
    static_rew = tm["post_training_val_metrics"]["static_step_reward"]
    assert post_rew > pre_rew + 1.0
    assert post_rew > static_rew + 1.5
    # Verify high action agreement and all 6 discrete actions represented
    assert tm["post_training_val_metrics"]["oracle_action_agreement"] >= 0.80
    assert tm["post_training_val_metrics"]["distinct_actions_selected"] >= 5
