"""Evaluate R2D2 policy against untrained and static baselines, and run training + export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from training.models.r2d2_network import R2D2Network
from training.algorithms.r2d2_trainer import (
    R2D2Trainer,
    compute_qec_action_reward,
    generate_qec_control_episode,
)
from training.export.export_checkpoint import export_r2d2_checkpoint


def evaluate_policy_on_split(
    net: R2D2Network,
    families: List[str],
    seed_start: int,
    seed_end: int,
    num_epochs: int = 16,
) -> Dict[str, object]:
    """Evaluate greedy policy on a seed split and compare with static and untrained baselines."""
    total_reward = 0.0
    static_reward = 0.0
    oracle_reward = 0.0
    total_steps = 0
    action_matches = 0
    action_histogram = [0] * net.num_actions
    gain_mse_accum = 0.0
    per_family_acc: Dict[str, List[float]] = {f: [] for f in families}

    for s in range(seed_start, seed_end + 1):
        for fam_idx, fam in enumerate(families):
            ep_seed = s * 31 + fam_idx * 7 + 999
            ep = generate_qec_control_episode(
                family=fam,
                seed=ep_seed,
                num_epochs=num_epochs,
                policy_net=None,
                epsilon=0.0,
            )
            obs = ep["obs"]
            opt_actions = ep["optimal_actions"]
            target_gains = ep["target_gains"]

            h, c = net.zero_state()
            fam_hits = 0
            for t in range(obs.shape[0]):
                out = net.step(obs[t], h, c)
                h, c = out["h_next"], out["c_next"]
                act = int(out["action_id"])
                opt_a = int(opt_actions[t])

                rew_policy = compute_qec_action_reward(obs[t], act, opt_a)
                rew_static = compute_qec_action_reward(obs[t], 0, opt_a)
                rew_oracle = compute_qec_action_reward(obs[t], opt_a, opt_a)

                total_reward += rew_policy
                static_reward += rew_static
                oracle_reward += rew_oracle
                total_steps += 1
                action_histogram[act] += 1

                if act == opt_a:
                    action_matches += 1
                    fam_hits += 1

                gain_diff = out["qubit_gains"] - target_gains[t]
                gain_mse_accum += float(np.mean(gain_diff * gain_diff))

            per_family_acc[fam].append(fam_hits / max(1, obs.shape[0]))

    return {
        "mean_step_reward": float(total_reward / max(1, total_steps)),
        "static_step_reward": float(static_reward / max(1, total_steps)),
        "oracle_step_reward": float(oracle_reward / max(1, total_steps)),
        "reward_advantage_over_static": float(
            (total_reward - static_reward) / max(1, total_steps)
        ),
        "oracle_action_agreement": float(action_matches / max(1, total_steps)),
        "gain_mse": float(gain_mse_accum / max(1, total_steps)),
        "action_histogram": action_histogram,
        "distinct_actions_selected": int(sum(1 for c in action_histogram if c > 0)),
        "per_family_agreement": {
            f: float(np.mean(vals)) for f, vals in per_family_acc.items()
        },
        "total_steps": total_steps,
    }


def run_full_training_and_export(
    repo_root: Optional[Path] = None,
    num_updates: int = 320,
    batch_size: int = 16,
    seed: int = 20260405,
) -> Dict[str, object]:
    """Train the recurrent R2D2 controller, validate on held-out seeds, and export checkpoints."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    else:
        repo_root = Path(repo_root)

    families = [
        "stationary_independent",
        "pair_correlated_crosstalk",
        "smooth_sinusoidal_drift",
        "abrupt_regime_switch",
        "burst_intermittency_r2d2",
    ]

    net = R2D2Network(
        input_dim=8,
        hidden_dim=64,
        num_actions=6,
        num_qubits=9,
        gain_clip=0.30,
        seed=seed,
    )

    # Evaluate untrained baseline before any updates
    pre_val = evaluate_policy_on_split(
        net=net,
        families=families,
        seed_start=1500,
        seed_end=1514,
        num_epochs=16,
    )

    trainer = R2D2Trainer(
        online_net=net,
        learning_rate=0.0028,
        gamma=0.96,
        burn_in_steps=4,
        unroll_steps=4,
        n_step_td=3,
        target_sync_interval=25,
        grad_clip_norm=5.0,
        replay_capacity=4096,
        seed=seed,
    )

    num_segments = trainer.populate_replay_from_seeds(
        families=families,
        seed_start=1000,
        seed_end=1039,
        epsilon=0.35,
    )

    loss_history: List[float] = []
    td_history: List[float] = []
    best_val_reward = -1e9
    best_state = net.get_state_dict()

    for step in range(1, num_updates + 1):
        frac = step / max(1, num_updates)
        beta = 0.4 + 0.6 * frac
        stats = trainer.train_step(batch_size=batch_size, beta=beta)
        loss_history.append(stats["loss"])
        td_history.append(stats["mean_abs_td"])

        if step == num_updates // 2:
            # Refresh replay with on-policy trajectories halfway through training
            trainer.populate_replay_from_seeds(
                families=families,
                seed_start=1000,
                seed_end=1019,
                epsilon=0.15,
            )

        if step % 40 == 0 or step == num_updates:
            val_eval = evaluate_policy_on_split(
                net=net,
                families=families,
                seed_start=1500,
                seed_end=1514,
                num_epochs=16,
            )
            if val_eval["mean_step_reward"] >= best_val_reward:
                best_val_reward = float(val_eval["mean_step_reward"])
                best_state = net.get_state_dict()

    net.load_state_dict(best_state)
    post_train = evaluate_policy_on_split(
        net=net,
        families=families,
        seed_start=1000,
        seed_end=1014,
        num_epochs=16,
    )
    post_val = evaluate_policy_on_split(
        net=net,
        families=families,
        seed_start=1500,
        seed_end=1514,
        num_epochs=16,
    )

    # Save training dataset summary under datasets/
    datasets_dir = repo_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = {
        "dataset_name": "AVID_V1_R2D2_Sequence_Replay_Dataset",
        "num_replay_segments": num_segments,
        "sequence_length": 11,
        "burn_in_steps": 4,
        "unroll_steps": 4,
        "n_step_td": 3,
        "train_seed_range": [1000, 1039],
        "val_seed_range": [1500, 1514],
        "heldout_test_seed_range": [2026, 2099],
        "families": families,
    }
    (datasets_dir / "r2d2_v1_dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2), encoding="utf-8"
    )

    training_summary = {
        "num_updates": num_updates,
        "batch_size": batch_size,
        "initial_loss_mean_first_20": float(np.mean(loss_history[:20])),
        "final_loss_mean_last_20": float(np.mean(loss_history[-20:])),
        "initial_td_error_first_20": float(np.mean(td_history[:20])),
        "final_td_error_last_20": float(np.mean(td_history[-20:])),
        "pre_training_val_metrics": pre_val,
        "post_training_train_metrics": post_train,
        "post_training_val_metrics": post_val,
    }

    bin_path = repo_root / "checkpoints" / "r2d2_v1_trained.bin"
    json_path = repo_root / "checkpoints" / "r2d2_v1_trained.json"
    onnx_path = repo_root / "checkpoints" / "r2d2_v1_trained.onnx"

    export_meta = export_r2d2_checkpoint(
        net=net,
        bin_path=bin_path,
        json_path=json_path,
        onnx_path=onnx_path,
        training_metrics=training_summary,
    )
    return {
        "export_meta": export_meta,
        "training_summary": training_summary,
    }


if __name__ == "__main__":
    res = run_full_training_and_export()
    ts = res["training_summary"]
    print(
        f"[R2D2 Training Complete] "
        f"loss: {ts['initial_loss_mean_first_20']:.4f} -> {ts['final_loss_mean_last_20']:.4f} | "
        f"val reward: {ts['pre_training_val_metrics']['mean_step_reward']:.4f} -> "
        f"{ts['post_training_val_metrics']['mean_step_reward']:.4f} | "
        f"agreement: {ts['post_training_val_metrics']['oracle_action_agreement']*100:.1f}% | "
        f"distinct actions: {ts['post_training_val_metrics']['distinct_actions_selected']}"
    )
