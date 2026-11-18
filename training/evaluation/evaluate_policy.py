"""Evaluate R2D2 V2 Policy: Per-Regime LER, Worst-Regime LER, Recovery Time, Stability & Inference Error."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np

from training.algorithms.r2d2_trainer import R2D2Trainer
from training.data.episode_dataset import EpisodeDataset
from training.data.trajectory_split import TrajectorySplitManager
from training.environments.qec_episode_generator import generate_v2_qec_episode
from training.export.export_checkpoint import export_r2d2_checkpoint
from training.models.r2d2_network import R2D2Network
from training.rewards.reward_diagnostics import RewardDiagnosticsTracker


def evaluate_policy_on_split(
    net: Optional[R2D2Network],
    regimes: Sequence[str],
    seed_start: int,
    seed_end: int,
    split_name: str = "val",
    num_epochs: int = 16,
    window_rounds: int = 256,
    observation_mode: str = "full_82d",
) -> Dict[str, object]:
    """Run closed-loop greedy rollouts on `[seed_start..seed_end]` and record full QEC diagnostics."""
    split_mgr = TrajectorySplitManager()
    tracker = RewardDiagnosticsTracker()
    ds = EpisodeDataset(split_name=split_name)

    total_dec_failures = 0
    total_static_failures = 0
    total_rounds = 0
    total_steps = 0
    action_matches = 0
    action_switches = 0
    total_transitions = 0
    gain_mse_accum = 0.0
    post_sq_err_accum = 0.0
    recovery_times: List[int] = []

    num_actions = net.num_actions if net is not None else 6
    action_histogram = [0] * num_actions
    per_regime_dec_fail: Dict[str, int] = {}
    per_regime_stat_fail: Dict[str, int] = {}
    per_regime_rounds: Dict[str, int] = {}

    for s in range(seed_start, seed_end + 1):
        for r_idx, fam in enumerate(regimes):
            ep_seed = split_mgr.episode_seed(split_name, s, r_idx)
            rec = generate_v2_qec_episode(
                family=fam,
                seed=ep_seed,
                split_name=split_name,
                num_epochs=num_epochs,
                policy_net=net,
                epsilon=0.0,
                window_rounds=window_rounds,
                observation_mode=observation_mode,
            )
            ds.add(rec)

            reg = rec.regime
            f_dec = int(np.sum(rec.decoded_failures))
            f_stat = int(np.sum(rec.static_failures))
            r_tot = num_epochs * window_rounds

            total_dec_failures += f_dec
            total_static_failures += f_stat
            total_rounds += r_tot

            per_regime_dec_fail[reg] = per_regime_dec_fail.get(reg, 0) + f_dec
            per_regime_stat_fail[reg] = per_regime_stat_fail.get(reg, 0) + f_stat
            per_regime_rounds[reg] = per_regime_rounds.get(reg, 0) + r_tot

            for t in range(num_epochs):
                act = int(rec.actions[t])
                opt_a = int(rec.optimal_actions[t])
                action_histogram[act] += 1
                total_steps += 1
                if act == opt_a:
                    action_matches += 1
                if t > 0:
                    total_transitions += 1
                    if act != int(rec.actions[t - 1]):
                        action_switches += 1
                tracker.log_step(rec.reward_terms[t], regime=reg)

                # Posterior inference RMSE vs ground-truth 21 parameters
                post_sq_err_accum += float(
                    np.mean((rec.posterior_means[t] - rec.true_theta_z[t]) ** 2)
                )

            # Evaluate gain prediction MSE when policy network is active
            if net is not None:
                out_seq = net.forward_sequence(rec.obs)
                gain_diff = out_seq["gain_seq"] - rec.target_gains
                gain_mse_accum += float(np.sum(np.mean(gain_diff ** 2, axis=1)))

            # Measure post-burst recovery epochs on burst trajectories
            if "burst" in fam:
                # Burst ends at epoch 5; measure epochs until LER <= static LER
                recovered_in = 3
                for offset in range(1, 4):
                    if 5 + offset < num_epochs:
                        if rec.decoded_failures[5 + offset] <= rec.static_failures[5 + offset]:
                            recovered_in = offset
                            break
                recovery_times.append(recovered_in)

    mean_ler = total_dec_failures / float(max(1, total_rounds))
    static_ler = total_static_failures / float(max(1, total_rounds))
    ler_red_pct = 100.0 * (static_ler - mean_ler) / max(static_ler, 1e-9)

    per_regime_ler = {
        reg: per_regime_dec_fail[reg] / float(max(1, per_regime_rounds[reg]))
        for reg in per_regime_rounds
    }
    per_regime_static_ler = {
        reg: per_regime_stat_fail[reg] / float(max(1, per_regime_rounds[reg]))
        for reg in per_regime_rounds
    }
    worst_regime = max(per_regime_ler, key=per_regime_ler.get) if per_regime_ler else None
    worst_regime_ler = float(per_regime_ler[worst_regime]) if worst_regime else 0.0

    rew_summary = tracker.summary()

    return {
        "split_name": split_name,
        "observation_mode": observation_mode,
        "mean_ler": float(mean_ler),
        "static_ler": float(static_ler),
        "ler_reduction_pct": float(ler_red_pct),
        "per_regime_ler": per_regime_ler,
        "per_regime_static_ler": per_regime_static_ler,
        "worst_regime": worst_regime,
        "worst_regime_ler": worst_regime_ler,
        "mean_step_reward": float(rew_summary["mean_terms"]["reward/total_reward"]),
        "reward_decomposition": rew_summary["mean_terms"],
        "worst_regime_reward": float(rew_summary["worst_regime_reward"]),
        "oracle_action_agreement": float(action_matches / max(1, total_steps)),
        "action_switch_rate": float(action_switches / max(1, total_transitions)),
        "mean_post_burst_recovery_epochs": (
            float(np.mean(recovery_times)) if recovery_times else 1.0
        ),
        "posterior_21_rmse": float(np.sqrt(post_sq_err_accum / max(1, total_steps))),
        "gain_mse": float(gain_mse_accum / max(1, total_steps)),
        "action_histogram": action_histogram,
        "distinct_actions_selected": int(sum(1 for c in action_histogram if c > 0)),
        "total_steps": total_steps,
        "total_rounds": total_rounds,
        "trajectory_ids": sorted(ds.trajectory_ids),
    }


def run_v2_training_and_export(
    repo_root: Optional[Path] = None,
    num_updates: int = 240,
    batch_size: int = 16,
    seed: int = 20260405,
    observation_mode: str = "full_82d",
    export_files: bool = True,
) -> Dict[str, object]:
    """Train the 82-input R2D2 V2 controller, validate on disjoint splits, and export V2 checkpoints."""
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)
    regimes = [
        "gaussian_stationary",
        "ou_drift",
        "zz_crosstalk",
        "tls_telegraph",
        "nonstationary_burst",
    ]

    net = R2D2Network(
        input_dim=82,
        hidden_dim=64,
        num_actions=6,
        num_qubits=9,
        gain_clip=0.30,
        seed=seed,
    )

    pre_val = evaluate_policy_on_split(
        net=net,
        regimes=regimes,
        seed_start=1500,
        seed_end=1507,
        split_name="val",
        num_epochs=14,
        observation_mode=observation_mode,
    )

    trainer = R2D2Trainer(
        online_net=net,
        learning_rate=0.0022,
        gamma=0.96,
        burn_in_steps=4,
        unroll_steps=4,
        n_step_td=3,
        target_sync_interval=25,
        grad_clip_norm=5.0,
        replay_capacity=4096,
        observation_mode=observation_mode,
        seed=seed,
    )

    num_segments = trainer.populate_replay_from_seeds(
        families=regimes,
        seed_start=1000,
        seed_end=1011,
        epsilon=0.35,
    )

    loss_history: List[float] = []
    td_history: List[float] = []
    best_score = -1e9
    best_state = net.get_state_dict()

    for step in range(1, num_updates + 1):
        frac = step / max(1, num_updates)
        beta = 0.4 + 0.6 * frac
        stats = trainer.train_step(batch_size=batch_size, beta=beta)
        loss_history.append(stats["loss"])
        td_history.append(stats["mean_abs_td"])

        if step == num_updates // 2:
            trainer.populate_replay_from_seeds(
                families=regimes,
                seed_start=1000,
                seed_end=1005,
                epsilon=0.15,
            )

        if step % 60 == 0 or step == num_updates:
            val_eval = evaluate_policy_on_split(
                net=net,
                regimes=regimes,
                seed_start=1500,
                seed_end=1507,
                split_name="val",
                num_epochs=14,
                observation_mode=observation_mode,
            )
            score = float(val_eval["mean_step_reward"]) + 25.0 * (
                float(val_eval["static_ler"]) - float(val_eval["mean_ler"])
            )
            if score >= best_score:
                best_score = score
                best_state = net.get_state_dict()

    net.load_state_dict(best_state)
    post_val = evaluate_policy_on_split(
        net=net,
        regimes=regimes,
        seed_start=1500,
        seed_end=1509,
        split_name="val",
        num_epochs=16,
        observation_mode=observation_mode,
    )
    post_test = evaluate_policy_on_split(
        net=net,
        regimes=regimes,
        seed_start=2026,
        seed_end=2035,
        split_name="test",
        num_epochs=16,
        observation_mode=observation_mode,
    )

    # Strip verbose trajectory ID lists from checkpoint JSON payload
    pre_val_clean = {k: v for k, v in pre_val.items() if k != "trajectory_ids"}
    post_val_clean = {k: v for k, v in post_val.items() if k != "trajectory_ids"}
    post_test_clean = {k: v for k, v in post_test.items() if k != "trajectory_ids"}

    training_metrics = {
        "num_updates": num_updates,
        "batch_size": batch_size,
        "replay_segments_populated": num_segments,
        "replay_regime_distribution": trainer.replay.regime_distribution(),
        "initial_loss_mean_first_20": float(np.mean(loss_history[:20])),
        "final_loss_mean_last_20": float(np.mean(loss_history[-20:])),
        "initial_td_error_first_20": float(np.mean(td_history[:20])),
        "final_td_error_last_20": float(np.mean(td_history[-20:])),
        "loss_curve_every_20": [
            float(np.mean(loss_history[max(0, i - 20) : i]))
            for i in range(20, len(loss_history) + 1, 20)
        ],
        "pre_training_val_metrics": pre_val_clean,
        "post_training_val_metrics": post_val_clean,
        "heldout_test_metrics": post_test_clean,
    }

    meta = {}
    if export_files:
        v2_dir = root / "checkpoints" / "v2"
        meta = export_r2d2_checkpoint(
            net=net,
            bin_path=v2_dir / "r2d2_v2_trained.bin",
            json_path=v2_dir / "r2d2_v2_trained.json",
            onnx_path=v2_dir / "r2d2_v2_trained.onnx",
            training_metrics=training_metrics,
            schema_version=2,
        )

    return {"network": net, "training_metrics": training_metrics, "checkpoint_meta": meta}


run_full_training_and_export = run_v2_training_and_export
