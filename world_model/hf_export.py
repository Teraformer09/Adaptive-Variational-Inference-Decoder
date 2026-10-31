"""Hugging Face Hub Exporter & LoRA Regime Adapter Trainer for the AVID World Model.

Packages:
1. Main Pretrained World Models (`main_models/WM_R2D2_Graph_RSSM_Full.pt`, `main_models/WM_Small_ST_Transformer_Full.pt`)
2. 5 Lightweight LoRA Noise-Regime Adapters (`adapters/adapter_*.pt`)
3. Full Ablation Checkpoints (`checkpoints/*.pt`, `checkpoints/training_summary.json`)
4. Evaluation Metrics & Plots (`evaluation/world_model_evaluation_results.json`, `evaluation/*.png`)
5. Hugging Face `config.json` and Model Card (`README.md`)

And uploads the complete bundle directly to Hugging Face Hub via `huggingface_hub`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from world_model.data import R2D2TrajectoryReplayDataset
from world_model.heads import WorldModelELBOLoss
from world_model.models import AVIDWorldModel, NoiseRegimeAdapter

HF_BUNDLE_DIR = Path(__file__).resolve().parent / "hf_hub_bundle"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
EVIDENCE_DIR = Path(__file__).resolve().parent / "evidance"


def load_env_file() -> None:
    """Load key=value pairs from `.env` in repo root, `world_model/.env`, or current working directory."""
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    for env_path in candidate_paths:
        if env_path.is_file():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    k, v = stripped.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))



def train_regime_adapters(
    base_model: AVIDWorldModel,
    splits: Dict[str, List[Dict[str, Any]]],
    epochs: int = 5,
    lr: float = 3e-3,
) -> Dict[str, Dict[str, Any]]:
    """Freeze the main `AVIDWorldModel` and train 5 lightweight LoRA regime adapters."""
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    criterion = WorldModelELBOLoss(lambda_obs=1.0, beta_kl=0.02, lambda_aux=2.5, lambda_forecast=1.0)
    b_steps = base_model.config.burn_in_steps
    t_steps = base_model.config.train_steps

    regime_specs = {
        "adapter_B1_transmon_leakage": [
            r for r in splits["train"] if r["latent_truth"]["family"] == "B1_TRANSMON_LEAKAGE"
        ],
        "adapter_A1_gate_crosstalk": [
            r for r in splits["train"] if r["latent_truth"]["family"] == "A1_GATE_CROSSTALK"
        ],
        "adapter_C1_C2_tls_idle_memory": [
            r
            for r in splits["train"]
            if r["latent_truth"]["family"] in ("C1_DRIVEN_TLS", "C2_IDLE_MEMORY")
        ],
        "adapter_A2_chipwide_events": [
            r for r in splits["train"] if r["latent_truth"]["family"] == "A2_CHIPWIDE_EVENTS"
        ],
        "adapter_OOD_schedule_calibration": splits["test_ood_schedule"][:6],
    }

    adapter_dir = HF_BUNDLE_DIR / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_summaries: Dict[str, Dict[str, Any]] = {}

    for adapter_name, records in regime_specs.items():
        adapter = NoiseRegimeAdapter(
            regime_name=adapter_name,
            d_model=base_model.config.d_model,
            d_subspace=base_model.config.d_subspace,
            rank=8,
        )
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
        ds = R2D2TrajectoryReplayDataset(
            records, burn_in_steps=b_steps, train_steps=t_steps, horizon_max=8, stride=16
        )
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        losses: List[float] = []
        for _ in range(epochs):
            batch_losses: List[float] = []
            for batch in loader:
                optimizer.zero_grad()
                out = adapter.forward_adapted(
                    base_model=base_model,
                    obs=batch["observations"],
                    burn_in_steps=b_steps,
                    train_steps=t_steps,
                    horizons=(1, 2, 4, 8),
                )
                loss_dict = criterion(
                    model_out=out,
                    obs=batch["observations"],
                    latent_truth=batch["latent_truth"],
                    burn_in_steps=b_steps,
                    train_steps=t_steps,
                )
                loss = loss_dict["total_loss"]
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))
            losses.append(float(np.mean(batch_losses)))

        save_path = adapter_dir / f"{adapter_name}.pt"
        adapter.save_adapter(save_path)
        num_params = sum(p.numel() for p in adapter.parameters())
        adapter_summaries[adapter_name] = {
            "file": f"adapters/{adapter_name}.pt",
            "num_parameters": num_params,
            "initial_loss": round(losses[0], 4),
            "final_loss": round(losses[-1], 4),
        }
        print(
            f"  [Adapter] {adapter_name}: params={num_params}, loss={losses[0]:.4f} -> {losses[-1]:.4f}",
            flush=True,
        )

    return adapter_summaries


def build_hf_model_card(
    eval_results: Dict[str, Any],
    adapter_summaries: Dict[str, Dict[str, Any]],
) -> str:
    """Create the Hugging Face Hub `README.md` Model Card with YAML metadata and benchmark metrics."""
    wm = eval_results["world_model_variants"]
    r2d2 = wm["WM_R2D2_Graph_RSSM_Full"]
    st = wm["WM_Small_ST_Transformer_Full"]

    return f"""---
library_name: pytorch
license: mit
tags:
  - quantum-computing
  - quantum-error-correction
  - superconducting-qubits
  - world-model
  - rssm
  - r2d2
  - lora-adapters
  - noise-generation
---

# AVID Quantum Hardware Noise World Model (Main Models, Checkpoints & LoRA Adapters)

This repository hosts the pretrained **AVID Quantum Hardware Noise World Model** (`WM_R2D2_Graph_RSSM_Full` and `WM_Small_ST_Transformer_Full`), all **5 LoRA Noise-Regime Adapters**, and the **7 Ablation Checkpoints** for spatio-temporal quantum noise generation, latent noise-state estimation, and multi-horizon fault forecasting on superconducting tunable-coupler transmon processors (Surface-17 topology).

## 1. Included Artifacts

- **Main Pretrained Models (`main_models/`)**:
  - `main_models/WM_R2D2_Graph_RSSM_Full.pt`: Version A (`R2D2-Graph-RSSM`) combining non-vision `GATv2` graph perception, Chrono-Initialized dual-timescale LSTM-RSSM memory, and R2D2 `burn_in_steps=16` partial-observability warmup.
  - `main_models/WM_Small_ST_Transformer_Full.pt`: Version B (`Small Spatio-Temporal Causal Transformer`) combining edge-biased spatial self-attention with causal temporal self-attention.
- **Modular LoRA Noise-Regime Adapters (`adapters/`)**:
  - `adapters/adapter_B1_transmon_leakage.pt` ({adapter_summaries['adapter_B1_transmon_leakage']['num_parameters']} params): Specializes the frozen base model for multi-level $|2\\rangle$ qutrit leakage and seepage dynamics (Varbanov et al., 2020).
  - `adapters/adapter_A1_gate_crosstalk.pt` ({adapter_summaries['adapter_A1_gate_crosstalk']['num_parameters']} params): Specializes for schedule-conditioned parallel CZ crosstalk (Ketterer & Wellens, 2023).
  - `adapters/adapter_C1_C2_tls_idle_memory.pt` ({adapter_summaries['adapter_C1_C2_tls_idle_memory']['num_parameters']} params): Specializes for driven TLS defects and non-Markovian idle bath memory revivals (Agarwal et al., 2023; Gulácsi & Burkard, 2023).
  - `adapters/adapter_A2_chipwide_events.pt` ({adapter_summaries['adapter_A2_chipwide_events']['num_parameters']} params): Specializes for cosmic-ray substrate phonon bursts (Wilen et al., 2021).
  - `adapters/adapter_OOD_schedule_calibration.pt` ({adapter_summaries['adapter_OOD_schedule_calibration']['num_parameters']} params): Fast calibration adapter for unseen out-of-distribution 3-gate concurrency schedules.
- **Ablation Checkpoints (`checkpoints/`)**: Complete checkpoints for all 7 architectural variants (`No_R2D2_BurnIn`, `No_Spatial_Graph`, `No_Temporal_Memory`, `No_Schedule_Cond`, `Syndrome_Only`).
- **Evaluation Metrics & Diagnostic Figures (`evaluation/`)**: Full JSON benchmark results (`world_model_evaluation_results.json`) and 300-DPI comparison plots against QuTiP classical baselines (`B0–B5`).

## 2. Benchmark Performance Summary

| Model Variant | B1 Leakage PR-AUC | A1 OOD Crosstalk RMSE | C1/C2 H=8 Rate RMSE | A2 Burst PR-AUC | Mixed Disentanglement Macro PR-AUC | Mixed H=1 Forecast NLL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`WM_R2D2_Graph_RSSM_Full`** | `{r2d2['B1_TRANSMON_LEAKAGE']['leakage_pr_auc']}` | `{r2d2['A1_GATE_CROSSTALK']['pairwise_corr_rmse_ood']}` | `{r2d2['C1_C2_TLS_AND_IDLE_MEMORY']['rate_rmse_H8']}` | `{r2d2['A2_CHIPWIDE_EVENTS']['event_pr_auc']}` | `{r2d2['MIXED_ALL_THREE']['disentanglement_macro_pr_auc']}` | `{r2d2['MIXED_ALL_THREE']['forecast_nll']['H1']}` |
| **`WM_Small_ST_Transformer_Full`** | `{st['B1_TRANSMON_LEAKAGE']['leakage_pr_auc']}` | `{st['A1_GATE_CROSSTALK']['pairwise_corr_rmse_ood']}` | `{st['C1_C2_TLS_AND_IDLE_MEMORY']['rate_rmse_H8']}` | `{st['A2_CHIPWIDE_EVENTS']['event_pr_auc']}` | `{st['MIXED_ALL_THREE']['disentanglement_macro_pr_auc']}` | `{st['MIXED_ALL_THREE']['forecast_nll']['H1']}` |

## 3. Quickstart Usage (Loading Main Model + LoRA Adapter)

```python
import torch
from world_model.models import AVIDWorldModel, WorldModelConfig, NoiseRegimeAdapter

# 1. Load Main R2D2-Graph-RSSM World Model
ckpt = torch.load("main_models/WM_R2D2_Graph_RSSM_Full.pt", map_location="cpu")
model = AVIDWorldModel(config=WorldModelConfig(**ckpt["config"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 2. (Optional) Attach a LoRA Regime Adapter
adapter_ckpt = torch.load("adapters/adapter_B1_transmon_leakage.pt", map_location="cpu")
adapter = NoiseRegimeAdapter(regime_name=adapter_ckpt["regime_name"])
adapter.load_state_dict(adapter_ckpt["state_dict"])
adapter.eval()
```
"""


def build_and_optionally_upload_hf_bundle(
    repo_id: str | None = None,
    hf_token: str | None = None,
) -> Path:
    """Run training, evaluation, LoRA adapter fitting, and package `world_model/hf_hub_bundle/`."""
    from world_model.evaluate_vs_baselines import (
        BASELINE_JSON_PATH,
        evaluate_model_on_splits,
        generate_evaluation_plots,
    )
    from world_model.train import run_training_pipeline

    HF_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    trained_models, training_summaries, splits = run_training_pipeline()

    with open(BASELINE_JSON_PATH, "r", encoding="utf-8") as f:
        baseline_ref = json.load(f)

    wm_eval: Dict[str, Any] = {}
    for name, model in trained_models.items():
        print(f"[Eval] Evaluating {name} on held-out test_id and test_ood_schedule splits...", flush=True)
        wm_eval[name] = evaluate_model_on_splits(
            model=model,
            test_id_records=splits["test_id"],
            test_ood_records=splits["test_ood_schedule"],
        )

    classical_summary = {
        "B1_TRANSMON_LEAKAGE": baseline_ref["benchmarks"]["B1_TRANSMON_LEAKAGE"]["baselines"],
        "A1_GATE_CROSSTALK": baseline_ref["benchmarks"]["A1_GATE_CROSSTALK"]["baselines"],
        "A2_CHIPWIDE_EVENTS": baseline_ref["benchmarks"]["A2_CHIPWIDE_EVENTS"]["baselines"],
        "MIXED_NOISE": baseline_ref["benchmarks"]["MIXED_NOISE"]["baselines"],
    }

    # Train the 5 LoRA Regime Adapters on top of the frozen main R2D2-Graph-RSSM model
    print("[Adapters] Training 5 lightweight LoRA Noise-Regime Adapters...", flush=True)
    adapter_summaries = train_regime_adapters(
        base_model=trained_models["WM_R2D2_Graph_RSSM_Full"],
        splits=splits,
    )

    full_results = {
        "scope": "Noise Generation, Latent Noise-State Estimation, and Short-Horizon Forecasting Only (No QEC Decoding)",
        "training_summary": training_summaries["dataset_validation"],
        "model_metadata": {
            k: {
                "num_parameters": v["num_parameters"],
                "training_time_sec": v["training_time_sec"],
                "final_val_loss": v["final_val_loss"],
                "inference_contract_verified": v["inference_contract_verified"],
            }
            for k, v in training_summaries["models"].items()
        },
        "lora_adapters": adapter_summaries,
        "world_model_variants": wm_eval,
        "classical_baselines_reference": classical_summary,
    }

    out_json = EVIDENCE_DIR / "world_model_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    generate_evaluation_plots(full_results)

    # Assemble Hugging Face Hub bundle directory
    main_dir = HF_BUNDLE_DIR / "main_models"
    ckpt_bundle_dir = HF_BUNDLE_DIR / "checkpoints"
    eval_bundle_dir = HF_BUNDLE_DIR / "evaluation"
    main_dir.mkdir(parents=True, exist_ok=True)
    ckpt_bundle_dir.mkdir(parents=True, exist_ok=True)
    eval_bundle_dir.mkdir(parents=True, exist_ok=True)

    # Copy main models
    for main_name in ("WM_R2D2_Graph_RSSM_Full", "WM_Small_ST_Transformer_Full"):
        shutil.copy2(CHECKPOINT_DIR / f"{main_name}.pt", main_dir / f"{main_name}.pt")

    # Copy all checkpoints & summary
    for p in CHECKPOINT_DIR.glob("*"):
        shutil.copy2(p, ckpt_bundle_dir / p.name)

    # Copy evaluation results & plots
    for p in EVIDENCE_DIR.glob("*"):
        shutil.copy2(p, eval_bundle_dir / p.name)

    # Write HF config.json
    hf_config = {
        "architectures": ["AVIDWorldModel"],
        "model_type": "quantum_graph_rssm_world_model",
        "main_models": {
            "r2d2_graph_rssm": "main_models/WM_R2D2_Graph_RSSM_Full.pt",
            "small_st_transformer": "main_models/WM_Small_ST_Transformer_Full.pt",
        },
        "adapters": adapter_summaries,
        "default_config": trained_models["WM_R2D2_Graph_RSSM_Full"].config.__dict__,
    }
    with open(HF_BUNDLE_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    # Write HF Model Card README.md
    model_card = build_hf_model_card(full_results, adapter_summaries)
    with open(HF_BUNDLE_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    print(f"[HF Bundle] Ready at: {HF_BUNDLE_DIR}", flush=True)

    load_env_file()
    token = hf_token or os.environ.get("HF_TOKEN")
    resolved_repo_id = repo_id or os.environ.get("HF_REPO_ID", "Terrano09/AVID-Quantum-Noise-World-Model")
    if resolved_repo_id and token:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        print(f"[HF Upload] Creating/verifying Hugging Face repo: {resolved_repo_id} ...", flush=True)
        api.create_repo(repo_id=resolved_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(HF_BUNDLE_DIR),
            repo_id=resolved_repo_id,
            repo_type="model",
            commit_message="Upload AVID Quantum Noise World Model, Checkpoints, and 5 LoRA Regime Adapters",
        )
        print(f"[HF Upload] Successfully uploaded to https://huggingface.co/{resolved_repo_id}", flush=True)

    return HF_BUNDLE_DIR


if __name__ == "__main__":
    load_env_file()
    parser = argparse.ArgumentParser(description="Train, package, and upload AVID World Model + Adapters to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face repo ID (defaults to HF_REPO_ID in .env)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token (defaults to HF_TOKEN in .env)")
    args = parser.parse_args()
    build_and_optionally_upload_hf_bundle(repo_id=args.repo_id, hf_token=args.token)
