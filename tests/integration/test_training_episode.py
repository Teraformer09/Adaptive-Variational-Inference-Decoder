"""NET-01, NET-02, EVAL-01, & EVAL-02 Integration Tests (`test_training_episode.py`)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from training.algorithms.r2d2_trainer import R2D2Trainer
from training.data.dataset_validation import validate_episode_datasets
from training.data.episode_dataset import EpisodeDataset
from training.data.trajectory_split import TrajectorySplitManager
from training.environments.qec_episode_generator import generate_v2_qec_episode
from training.evaluation.evaluate_ablation import run_representation_and_controller_ablation
from training.evaluation.evaluate_noise_regimes import evaluate_across_regimes_and_heldout_mixtures
from training.export.export_checkpoint import load_r2d2_checkpoint_bin
from training.models.r2d2_network import R2D2Network
from training.replay.sequence_sampler import slice_episode_into_segments


def test_net01_python_82_input_architecture_and_parameter_count() -> None:
    """NET-01: Verify exact 82-input V2 shapes, parameter counts (38,672 trainable / 38,836 total), and input validation."""
    net = R2D2Network()
    assert net.input_dim == 82
    assert net.hidden_dim == 64
    assert net.num_actions == 6
    assert net.num_qubits == 9

    assert net.encoder.W_ih.shape == (256, 82)
    assert net.encoder.W_hh.shape == (256, 64)
    assert net.encoder.b_ih.shape == (256,)
    assert net.head.W_val.shape == (1, 64)
    assert net.head.W_adv.shape == (6, 64)
    assert net.head.W_gain.shape == (9, 64)

    assert net.num_trainable_parameters == 38672
    assert net.num_serialized_floats == 38836

    h0, c0 = net.zero_state()
    with pytest.raises(ValueError, match="expected observation shape"):
        net.step(np.zeros(8), h0, c0)

    bad_obs = np.zeros(82)
    bad_obs[10] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        net.step(bad_obs, h0, c0)


def test_net02_sequence_continuity_burnin_and_bptt_82d() -> None:
    """NET-02: Verify sequence continuity, stop-gradient burn-in semantics, and finite-difference BPTT on 82-D inputs."""
    net = R2D2Network(input_dim=82, hidden_dim=64, seed=909)
    rec = generate_v2_qec_episode("zz_crosstalk", seed=10010, split_name="train", num_epochs=12)
    segs = slice_episode_into_segments(
        obs=rec.obs,
        actions=rec.actions,
        rewards=rec.rewards,
        dones=rec.dones,
        target_gains=rec.target_gains,
        optimal_actions=rec.optimal_actions,
        h_states=rec.h_states,
        c_states=rec.c_states,
        burn_in_steps=4,
        unroll_steps=4,
        n_step_td=3,
        stride=2,
    )
    assert len(segs) >= 1
    seg = segs[0]
    assert seg.obs.shape == (11, 82)

    # Verify burn-in stop-gradient: gradients do not depend on W_ih perturbations during burn-in if h_burn, c_burn are fixed
    burn_in = 4
    out = net.forward_sequence(seg.obs, h_init=seg.h_init, c_init=seg.c_init, burn_in_steps=burn_in)
    caches = out["caches"]
    h_burn = caches[burn_in - 1].h_next.copy()
    c_burn = caches[burn_in - 1].c_next.copy()

    rng = np.random.default_rng(910)
    dq_seq = np.zeros((11, 6), dtype=np.float64)
    dg_seq = np.zeros((11, 9), dtype=np.float64)
    dq_seq[burn_in:] = rng.normal(0.0, 0.5, size=(7, 6))
    dg_seq[burn_in:] = rng.normal(0.0, 0.5, size=(7, 9))

    grads = net.backward_sequence(out["h_seq"], caches, dq_seq, dg_seq, burn_in_steps=burn_in)

    eps = 1e-6
    for r, c in ((0, 0), (5, 25), (30, 74)):
        orig = net.encoder.W_ih[r, c]
        net.encoder.W_ih[r, c] = orig + eps
        op = net.forward_sequence(seg.obs[burn_in:], h_init=h_burn, c_init=c_burn, burn_in_steps=0)
        lp = float(np.sum(op["q_seq"] * dq_seq[burn_in:]) + np.sum(op["gain_seq"] * dg_seq[burn_in:]))

        net.encoder.W_ih[r, c] = orig - eps
        om = net.forward_sequence(seg.obs[burn_in:], h_init=h_burn, c_init=c_burn, burn_in_steps=0)
        lm = float(np.sum(om["q_seq"] * dq_seq[burn_in:]) + np.sum(om["gain_seq"] * dg_seq[burn_in:]))

        net.encoder.W_ih[r, c] = orig
        fd = (lp - lm) / (2.0 * eps)
        assert abs(grads["W_ih"][r, c] - fd) < 1e-5


def test_eval01_and_eval02_disjoint_heldout_mixtures_and_ablation() -> None:
    """EVAL-01 & EVAL-02: Verify strict split disjointness, held-out generalization, and full 5-way ablation."""
    repo_root = Path(__file__).resolve().parents[2]
    v2_bin = repo_root / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    assert v2_bin.exists(), "Missing trained V2 checkpoint checkpoints/v2/r2d2_v2_trained.bin"

    net, _ = load_r2d2_checkpoint_bin(v2_bin, expected_version=2)

    # Check EpisodeDataset validation and anti-leakage gate
    split_mgr = TrajectorySplitManager()
    ds_train = EpisodeDataset("train")
    ds_test = EpisodeDataset("test")
    ds_train.add(
        generate_v2_qec_episode(
            "zz_crosstalk", split_mgr.episode_seed("train", 1000, 2), split_name="train", num_epochs=8
        )
    )
    ds_test.add(
        generate_v2_qec_episode(
            "zz_crosstalk", split_mgr.episode_seed("test", 2026, 2), split_name="test", num_epochs=8
        )
    )
    validate_episode_datasets([ds_train, ds_test])

    # EVAL-01: Held-out mixtures & disjoint trajectories
    reg_res = evaluate_across_regimes_and_heldout_mixtures(
        net,
        test_seed_start=2026,
        test_seed_end=2028,
        heldout_seed_start=3000,
        heldout_seed_end=3002,
    )
    assert reg_res["disjoint_trajectories_verified"] is True
    assert reg_res["core_5_regimes_test"]["mean_ler"] < reg_res["core_5_regimes_test"]["static_ler"]
    assert reg_res["heldout_3_mixtures"]["mean_ler"] < reg_res["heldout_3_mixtures"]["static_ler"]

    # EVAL-02: 5-way ablation
    abl = run_representation_and_controller_ablation(
        net, seed_start=2026, seed_end=2028, num_epochs=12
    )
    assert set(abl.keys()) == {
        "1_Static_B0",
        "2_Memoryless_JointDP_Proj",
        "3_R2D2_V2_NoUncertainty",
        "4_R2D2_V2_Full_82D",
        "5_R2D2_V2_Oracle_7x3",
    }
    assert abl["4_R2D2_V2_Full_82D"]["ler"] < abl["1_Static_B0"]["ler"]
