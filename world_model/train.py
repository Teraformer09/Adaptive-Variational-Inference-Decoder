"""Phase 5: 3-Stage Curriculum Trainer and Mandatory Ablation Runner for the AVID World Model."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from world_model.data import (
    QuantumNoiseTrajectoryGenerator,
    R2D2TrajectoryReplayDataset,
    TrajectorySchemaValidator,
    strip_latent_truth,
    verify_zero_leakage_splits,
)
from world_model.heads import WorldModelELBOLoss
from world_model.models import AVIDWorldModel, WorldModelConfig

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def set_all_seeds(seed: int = 42) -> None:
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)


def verify_inference_contract(model: AVIDWorldModel, sample_batch: Dict[str, Any]) -> bool:
    """Verify that `model.predict()` rejects `latent_truth` and works on sanitized observations."""
    obs_win = {k: v[:2, :48] for k, v in sample_batch["observations"].items()}
    fut_sched = {
        k: v[:2, 48:56]
        for k, v in sample_batch["observations"].items()
        if k in ("schedule_adj", "concurrency_group", "drive_norm", "detection_vec", "syndrome_vec", "analog_llr", "analog_mask", "telemetry_val", "telemetry_mask")
    }

    # 1. Must succeed without latent_truth
    belief, forecast = model.predict(
        observation_window=obs_win,
        future_schedule=fut_sched,
        horizon=8,
    )
    assert "posterior_estimates" in belief and 8 in forecast

    # 2. Must raise ValueError if latent_truth is passed
    rejected = False
    try:
        model.predict(
            observation_window=obs_win,
            future_schedule=fut_sched,
            horizon=8,
            latent_truth=sample_batch["latent_truth"],
        )
    except ValueError:
        rejected = True
    assert rejected, "Inference contract failed to reject `latent_truth`!"
    return True


def train_single_model(
    name: str,
    config: WorldModelConfig,
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    epochs_stage1: int = 8,
    epochs_stage2: int = 10,
    lr: float = 2.5e-3,
    batch_size: int = 16,
) -> Tuple[AVIDWorldModel, Dict[str, Any]]:
    """Train a world model variant through the 3-stage curriculum."""
    set_all_seeds(101)
    model = AVIDWorldModel(config=config)
    criterion = WorldModelELBOLoss(
        lambda_obs=1.0,
        beta_kl=0.04,
        lambda_aux=2.2,
        lambda_forecast=1.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Separate single-family records (Stage I) vs. all records including MIXED_ALL_THREE (Stage II)
    single_family_records = [
        r for r in train_records if r["latent_truth"]["family"] != "MIXED_ALL_THREE"
    ]
    ds_stage1 = R2D2TrajectoryReplayDataset(
        single_family_records,
        burn_in_steps=config.burn_in_steps,
        train_steps=config.train_steps,
        horizon_max=8,
        stride=16,
    )
    ds_stage2 = R2D2TrajectoryReplayDataset(
        train_records,
        burn_in_steps=config.burn_in_steps,
        train_steps=config.train_steps,
        horizon_max=8,
        stride=16,
    )
    ds_val = R2D2TrajectoryReplayDataset(
        val_records,
        burn_in_steps=config.burn_in_steps,
        train_steps=config.train_steps,
        horizon_max=8,
        stride=24,
    )

    loader_s1 = DataLoader(ds_stage1, batch_size=batch_size, shuffle=True)
    loader_s2 = DataLoader(ds_stage2, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    history: List[Dict[str, float]] = []
    t0 = time.time()

    total_epochs = epochs_stage1 + epochs_stage2
    for epoch in range(1, total_epochs + 1):
        is_stage1 = epoch <= epochs_stage1
        loader = loader_s1 if is_stage1 else loader_s2
        horizons = (1, 2) if is_stage1 else (1, 2, 4, 8)
        beta_scale = min(1.0, epoch / float(max(1, epochs_stage1)))

        model.train()
        epoch_losses: List[float] = []
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                obs=batch["observations"],
                burn_in_steps=config.burn_in_steps,
                train_steps=config.train_steps,
                horizons=horizons,
            )
            loss_dict = criterion(
                model_out=out,
                obs=batch["observations"],
                latent_truth=batch["latent_truth"],
                burn_in_steps=config.burn_in_steps,
                train_steps=config.train_steps,
                beta_scale=beta_scale,
            )
            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        # Validation pass
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for vbatch in loader_val:
                vout = model(
                    obs=vbatch["observations"],
                    burn_in_steps=config.burn_in_steps,
                    train_steps=config.train_steps,
                    horizons=(1, 2, 4, 8),
                )
                vloss = criterion(
                    model_out=vout,
                    obs=vbatch["observations"],
                    latent_truth=vbatch["latent_truth"],
                    burn_in_steps=config.burn_in_steps,
                    train_steps=config.train_steps,
                    beta_scale=1.0,
                )
                val_losses.append(float(vloss["total_loss"].item()))

        history.append({
            "epoch": epoch,
            "stage": 1 if is_stage1 else 2,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": float(np.mean(val_losses)),
        })

    # Stage III: Verify strict inference contract
    sample_batch = next(iter(loader_val))
    contract_ok = verify_inference_contract(model, sample_batch)
    elapsed_s = round(time.time() - t0, 2)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {
        "name": name,
        "config": config.__dict__,
        "num_parameters": num_params,
        "training_time_sec": elapsed_s,
        "initial_train_loss": round(history[0]["train_loss"], 4),
        "final_train_loss": round(history[-1]["train_loss"], 4),
        "final_val_loss": round(history[-1]["val_loss"], 4),
        "inference_contract_verified": contract_ok,
        "history": history,
    }
    return model, summary


def build_ablation_suite() -> Dict[str, WorldModelConfig]:
    """Define the dual backbones and mandatory ablation configurations."""
    return {
        "WM_R2D2_Graph_RSSM_Full": WorldModelConfig(
            backbone="r2d2_graph_rssm",
            spatial_mode="gnn",
            temporal_mode="r2d2_lstm_rssm",
            burn_in_steps=16,
            train_steps=32,
            use_schedule=True,
            use_analog=True,
        ),
        "WM_Small_ST_Transformer_Full": WorldModelConfig(
            backbone="spatio_temporal_transformer",
            spatial_mode="graph_transformer",
            temporal_mode="causal_transformer",
            burn_in_steps=16,
            train_steps=32,
            use_schedule=True,
            use_analog=True,
        ),
        "Ablation_No_R2D2_BurnIn": WorldModelConfig(
            backbone="r2d2_graph_rssm",
            spatial_mode="gnn",
            temporal_mode="r2d2_lstm_rssm",
            burn_in_steps=0,
            train_steps=32,
            use_schedule=True,
            use_analog=True,
        ),
        "Ablation_No_Spatial_Graph": WorldModelConfig(
            backbone="r2d2_graph_rssm",
            spatial_mode="none",
            temporal_mode="r2d2_lstm_rssm",
            burn_in_steps=16,
            train_steps=32,
            use_schedule=True,
            use_analog=True,
        ),
        "Ablation_No_Temporal_Memory": WorldModelConfig(
            backbone="r2d2_graph_rssm",
            spatial_mode="gnn",
            temporal_mode="none",
            burn_in_steps=0,
            train_steps=32,
            use_schedule=True,
            use_analog=True,
        ),
        "Ablation_No_Schedule_Cond": WorldModelConfig(
            backbone="r2d2_graph_rssm",
            spatial_mode="gnn",
            temporal_mode="r2d2_lstm_rssm",
            burn_in_steps=16,
            train_steps=32,
            use_schedule=False,
            use_analog=True,
        ),
        "Ablation_Syndrome_Only": WorldModelConfig(
            backbone="r2d2_graph_rssm",
            spatial_mode="gnn",
            temporal_mode="r2d2_lstm_rssm",
            burn_in_steps=16,
            train_steps=32,
            use_schedule=True,
            use_analog=False,
        ),
    }


def run_training_pipeline() -> Tuple[Dict[str, AVIDWorldModel], Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """Generate schema-validated splits and train both backbones + all ablations."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    gen = QuantumNoiseTrajectoryGenerator()
    splits = gen.generate_benchmark_splits(
        num_train_per_family=8,
        num_val_per_family=3,
        num_test_per_family=4,
        num_rounds=72,
    )

    validator = TrajectorySchemaValidator()
    total_validated = sum(validator.validate_batch(recs) for recs in splits.values())
    split_ok, split_report = verify_zero_leakage_splits(splits)
    if not split_ok:
        raise RuntimeError(f"Dataset split leakage detected: {split_report}")

    # Verify strip_latent_truth helper
    sample_sanitized = strip_latent_truth(splits["test_id"][0])
    assert "latent_truth" not in sample_sanitized

    suite = build_ablation_suite()
    trained_models: Dict[str, AVIDWorldModel] = {}
    training_summaries: Dict[str, Any] = {
        "dataset_validation": {
            "total_validated_trajectories": total_validated,
            "split_integrity": split_report,
        },
        "models": {},
    }

    for name, cfg in suite.items():
        print(f"[Train] Starting 3-stage curriculum for {name} ...", flush=True)
        model, summary = train_single_model(
            name=name,
            config=cfg,
            train_records=splits["train"],
            val_records=splits["val"],
            epochs_stage1=4,
            epochs_stage2=6,
            batch_size=24,
        )
        trained_models[name] = model
        training_summaries["models"][name] = summary
        ckpt_path = CHECKPOINT_DIR / f"{name}.pt"
        torch.save(
            {
                "name": name,
                "config": cfg.__dict__,
                "state_dict": model.state_dict(),
                "summary": summary,
            },
            ckpt_path,
        )
        print(
            f"  -> {name}: params={summary['num_parameters']}, "
            f"train_loss={summary['initial_train_loss']} -> {summary['final_train_loss']}, "
            f"val_loss={summary['final_val_loss']}, time={summary['training_time_sec']}s",
            flush=True,
        )

    with open(CHECKPOINT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(training_summaries, f, indent=2)

    return trained_models, training_summaries, splits


if __name__ == "__main__":
    run_training_pipeline()
