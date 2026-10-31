"""Phase 6: Head-to-Head Evaluation of Dual-Backbone World Models vs. QuTiP Classical Baselines (B0-B5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from world_model.data import strip_latent_truth, trajectory_to_tensors
from world_model.models import AVIDWorldModel
from world_model.train import run_training_pipeline

EVIDENCE_DIR = Path(__file__).resolve().parent / "evidance"
BASELINE_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "Benchmark-Evals"
    / "Qutip-Baseline"
    / "evidance"
    / "baseline_evaluation_results.json"
)


def binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute exact Average Precision (PR-AUC) in pure NumPy."""
    y_true = y_true.astype(np.float64).ravel()
    y_score = y_score.astype(np.float64).ravel()
    total_pos = np.sum(y_true > 0.5)
    if total_pos == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_sorted = (y_true[order] > 0.5).astype(np.float64)
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    precisions = tp / ranks
    return float(np.sum(precisions * y_sorted) / total_pos)


def binary_f1(y_true: np.ndarray, y_score: np.ndarray, thresh: float = 0.45) -> float:
    y_t = (y_true.ravel() > 0.5).astype(np.int64)
    y_p = (y_score.ravel() >= thresh).astype(np.int64)
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    if 2 * tp + fp + fn == 0:
        return 0.0
    return float(2 * tp / (2 * tp + fp + fn))


def compute_detection_delay(y_true_seq: np.ndarray, y_score_seq: np.ndarray, thresh: float = 0.42) -> float:
    """Compute mean detection delay (in rounds) from onset of true episodes."""
    # y_true_seq: (N_traj, T, C) or (N_traj, T)
    if y_true_seq.ndim == 2:
        y_true_seq = y_true_seq[:, :, None]
        y_score_seq = y_score_seq[:, :, None]
    N, T, C = y_true_seq.shape
    delays: List[float] = []
    for n in range(N):
        for c in range(C):
            in_ep = False
            onset = 0
            detected = False
            for t in range(T):
                if y_true_seq[n, t, c] > 0.5 and not in_ep:
                    in_ep = True
                    onset = t
                    detected = False
                if in_ep and not detected and y_score_seq[n, t, c] >= thresh:
                    delays.append(float(t - onset))
                    detected = True
                if y_true_seq[n, t, c] <= 0.5 and in_ep:
                    if not detected:
                        delays.append(3.0)
                    in_ep = False
    return float(np.mean(delays)) if len(delays) > 0 else 0.25


def binary_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    y = y_true.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y = y_true.ravel().astype(np.float64)
    p = y_prob.ravel().astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if np.any(mask):
            ece += (np.sum(mask) / len(p)) * abs(np.mean(p[mask]) - np.mean(y[mask]))
    return float(ece)


def evaluate_model_on_splits(
    model: AVIDWorldModel,
    test_id_records: List[Dict[str, Any]],
    test_ood_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run strict inference-only `.predict()` across held-out trajectories and compute all benchmark metrics."""
    b_steps = model.config.burn_in_steps
    horizon = 8

    def run_records(records: List[Dict[str, Any]]) -> Dict[str, List[np.ndarray]]:
        outputs: Dict[str, List[np.ndarray]] = {
            "gt_det": [], "pred_det": [],
            "gt_leak": [], "pred_leak": [],
            "gt_corr_pairs": [], "pred_corr_pairs": [],
            "gt_burst_active": [], "pred_burst_prob": [],
            "gt_footprint": [], "pred_footprint": [],
            "gt_drift_mem": [], "pred_drift_mem": [],
            "gt_det_h1": [], "pred_det_h1": [],
            "gt_det_h2": [], "pred_det_h2": [],
            "gt_det_h4": [], "pred_det_h4": [],
            "gt_det_h8": [], "pred_det_h8": [],
            "gt_rate_h1": [], "pred_rate_h1": [],
            "gt_rate_h4": [], "pred_rate_h4": [],
            "gt_rate_h8": [], "pred_rate_h8": [],
        }

        for rec in records:
            # 1. Strip latent_truth BEFORE calling model.predict()
            sanitized_rec = strip_latent_truth(rec)
            assert "latent_truth" not in sanitized_rec
            inf_tensors = trajectory_to_tensors(sanitized_rec, include_latent_truth=False)
            full_tensors = trajectory_to_tensors(rec, include_latent_truth=True)

            T_total = inf_tensors["observations"]["detection_vec"].shape[0]
            obs_win = {
                k: v[: T_total - horizon].unsqueeze(0)
                for k, v in inf_tensors["observations"].items()
            }
            fut_sched = {
                k: v[T_total - horizon : T_total].unsqueeze(0)
                for k, v in inf_tensors["observations"].items()
            }

            belief, forecast = model.predict(
                observation_window=obs_win,
                future_schedule=fut_sched,
                horizon=horizon,
                burn_in_steps=b_steps,
            )

            post = belief["posterior_estimates"]
            active_end = T_total - horizon
            s_idx = b_steps

            lt = full_tensors["latent_truth"]
            obs_gt = full_tensors["observations"]

            outputs["gt_det"].append(obs_gt["detection_vec"][s_idx:active_end].numpy())
            outputs["pred_det"].append(post["det_prob"][0].cpu().numpy())

            outputs["gt_leak"].append(lt["z_leak"][s_idx:active_end].numpy())
            outputs["pred_leak"].append(post["leak_prob"][0].cpu().numpy())

            outputs["gt_corr_pairs"].append(lt["z_corr_pairs"][s_idx:active_end].numpy())
            outputs["pred_corr_pairs"].append(post["pair_corr_prob"][0].cpu().numpy())

            outputs["gt_burst_active"].append(lt["z_burst_active"][s_idx:active_end].numpy())
            outputs["pred_burst_prob"].append(post["burst_prob"][0].cpu().numpy())

            outputs["gt_footprint"].append(lt["z_burst_footprint"][s_idx:active_end].numpy())
            outputs["pred_footprint"].append(post["burst_footprint_pred"][0].cpu().numpy())

            gt_dm = (lt["z_drift"][s_idx:active_end] + lt["z_memory"][s_idx:active_end]).numpy()
            pred_dm = (post["drift_pred"][0] + post["memory_pred"][0]).cpu().numpy()
            outputs["gt_drift_mem"].append(gt_dm)
            outputs["pred_drift_mem"].append(pred_dm)

            for h_step in (1, 2, 4, 8):
                outputs[f"gt_det_h{h_step}"].append(
                    obs_gt["detection_vec"][s_idx + h_step : active_end + h_step].numpy()
                )
                outputs[f"pred_det_h{h_step}"].append(
                    forecast[h_step]["det_prob"][0].cpu().numpy()
                )
            for h_step in (1, 4, 8):
                outputs[f"gt_rate_h{h_step}"].append(
                    lt["true_detection_prob"][s_idx + h_step : active_end + h_step].numpy()
                )
                outputs[f"pred_rate_h{h_step}"].append(
                    forecast[h_step]["det_prob"][0].cpu().numpy()
                )

        return {k: np.stack(v, axis=0) for k, v in outputs.items()}

    # Partition test records by benchmark family
    by_fam_id: Dict[str, List[Dict[str, Any]]] = {}
    by_fam_ood: Dict[str, List[Dict[str, Any]]] = {}
    for r in test_id_records:
        fam = r["latent_truth"]["family"]
        by_fam_id.setdefault(fam, []).append(r)
    for r in test_ood_records:
        fam = r["latent_truth"]["family"]
        by_fam_ood.setdefault(fam, []).append(r)

    # 1. B1_TRANSMON_LEAKAGE
    b1_res = run_records(by_fam_id["B1_TRANSMON_LEAKAGE"] + by_fam_id["MIXED_ALL_THREE"])
    b1_pr_auc = binary_pr_auc(b1_res["gt_leak"], b1_res["pred_leak"])
    b1_f1 = binary_f1(b1_res["gt_leak"], b1_res["pred_leak"], thresh=0.45)
    b1_delay = compute_detection_delay(b1_res["gt_leak"], b1_res["pred_leak"], thresh=0.42)

    # 2. A1_GATE_CROSSTALK (ID & Unseen 3-Gate OOD Concurrency)
    a1_id = run_records(by_fam_id["A1_GATE_CROSSTALK"])
    a1_ood = run_records(by_fam_ood["A1_GATE_CROSSTALK"])
    a1_rmse_id = float(np.sqrt(np.mean((a1_id["gt_corr_pairs"] - a1_id["pred_corr_pairs"]) ** 2)))
    a1_rmse_ood = float(np.sqrt(np.mean((a1_ood["gt_corr_pairs"] - a1_ood["pred_corr_pairs"]) ** 2)))

    # 3. C1_DRIVEN_TLS & C2_IDLE_MEMORY
    c_res = run_records(by_fam_id["C1_DRIVEN_TLS"] + by_fam_id["C2_IDLE_MEMORY"])
    c_rmse_h1 = float(np.sqrt(np.mean((c_res["gt_rate_h1"] - c_res["pred_rate_h1"]) ** 2)))
    c_rmse_h4 = float(np.sqrt(np.mean((c_res["gt_rate_h4"] - c_res["pred_rate_h4"]) ** 2)))
    c_rmse_h8 = float(np.sqrt(np.mean((c_res["gt_rate_h8"] - c_res["pred_rate_h8"]) ** 2)))

    # 4. A2_CHIPWIDE_EVENTS
    a2_res = run_records(by_fam_id["A2_CHIPWIDE_EVENTS"] + by_fam_id["MIXED_ALL_THREE"])
    a2_pr_auc = binary_pr_auc(a2_res["gt_burst_active"], a2_res["pred_burst_prob"])
    a2_f1 = binary_f1(a2_res["gt_burst_active"], a2_res["pred_burst_prob"], thresh=0.42)
    a2_footprint_rmse = float(np.sqrt(np.mean((a2_res["gt_footprint"] - a2_res["pred_footprint"]) ** 2)))

    # 5. MIXED_ALL_THREE Disentanglement & Multi-Horizon Forecasting
    mix_res = run_records(by_fam_id["MIXED_ALL_THREE"])
    pr_leak = binary_pr_auc(mix_res["gt_leak"], mix_res["pred_leak"])
    pr_burst = binary_pr_auc(mix_res["gt_burst_active"], mix_res["pred_burst_prob"])
    gt_corr_active = (mix_res["gt_corr_pairs"].max(axis=(-1, -2)) > 0.01).astype(np.float32)
    pred_corr_active = np.clip(mix_res["pred_corr_pairs"].max(axis=(-1, -2)) / 0.04, 0.0, 1.0)
    pr_corr = binary_pr_auc(gt_corr_active, pred_corr_active)
    gt_dm_high = (mix_res["gt_drift_mem"] > np.median(mix_res["gt_drift_mem"])).astype(np.float32)
    pr_dm = binary_pr_auc(gt_dm_high, mix_res["pred_drift_mem"])
    disentangle_macro_pr_auc = float(np.mean([pr_leak, pr_burst, pr_corr, pr_dm]))

    nll_h1 = binary_nll(mix_res["gt_det_h1"], mix_res["pred_det_h1"])
    nll_h2 = binary_nll(mix_res["gt_det_h2"], mix_res["pred_det_h2"])
    nll_h4 = binary_nll(mix_res["gt_det_h4"], mix_res["pred_det_h4"])
    nll_h8 = binary_nll(mix_res["gt_det_h8"], mix_res["pred_det_h8"])
    brier_h1 = float(np.mean((mix_res["gt_det_h1"] - mix_res["pred_det_h1"]) ** 2))
    ece_h1 = compute_ece(mix_res["gt_det_h1"], mix_res["pred_det_h1"])
    rate_rmse_h1 = float(np.sqrt(np.mean((mix_res["gt_rate_h1"] - mix_res["pred_rate_h1"]) ** 2)))

    return {
        "B1_TRANSMON_LEAKAGE": {
            "leakage_pr_auc": round(b1_pr_auc, 4),
            "leakage_f1": round(b1_f1, 4),
            "detection_delay_rounds": round(b1_delay, 2),
        },
        "A1_GATE_CROSSTALK": {
            "pairwise_corr_rmse_id": round(a1_rmse_id, 5),
            "pairwise_corr_rmse_ood": round(a1_rmse_ood, 5),
        },
        "C1_C2_TLS_AND_IDLE_MEMORY": {
            "rate_rmse_H1": round(c_rmse_h1, 5),
            "rate_rmse_H4": round(c_rmse_h4, 5),
            "rate_rmse_H8": round(c_rmse_h8, 5),
        },
        "A2_CHIPWIDE_EVENTS": {
            "event_pr_auc": round(a2_pr_auc, 4),
            "event_f1": round(a2_f1, 4),
            "spatial_footprint_rmse": round(a2_footprint_rmse, 5),
        },
        "MIXED_ALL_THREE": {
            "disentanglement_macro_pr_auc": round(disentangle_macro_pr_auc, 4),
            "per_mechanism_pr_auc": {
                "leakage": round(pr_leak, 4),
                "crosstalk": round(pr_corr, 4),
                "burst": round(pr_burst, 4),
                "drift_memory": round(pr_dm, 4),
            },
            "forecast_nll": {
                "H1": round(nll_h1, 4),
                "H2": round(nll_h2, 4),
                "H4": round(nll_h4, 4),
                "H8": round(nll_h8, 4),
            },
            "brier_H1": round(brier_h1, 4),
            "ece_H1": round(ece_h1, 4),
            "latent_rate_rmse_H1": round(rate_rmse_h1, 4),
        },
    }


def generate_evaluation_plots(results: Dict[str, Any]) -> None:
    """Generate 300-DPI diagnostic comparison figures in `world_model/evidance/`."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    wm = results["world_model_variants"]
    cb = results["classical_baselines_reference"]

    # Plot 1: World Models (R2D2-Graph-RSSM & Small ST-Transformer) vs. Classical Baselines (B0-B5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)

    # Panel A: Mixed-Noise Disentanglement Macro PR-AUC
    names_p1 = [
        "B0 Static",
        "B2 Single HMM",
        "B4 AR(1)/OU",
        "B3 Classical\nTracker",
        "WM: Small\nST-Transformer",
        "WM: R2D2\nGraph-RSSM",
        "B5 Oracle\n(Upper Bound)",
    ]
    vals_p1 = [
        cb["MIXED_NOISE"]["B0_STATIC"]["disentanglement_macro_pr_auc"],
        cb["MIXED_NOISE"]["B2_SINGLE_HMM"]["disentanglement_macro_pr_auc"],
        cb["MIXED_NOISE"]["B4_AR1_OU_FILTER"]["disentanglement_macro_pr_auc"],
        cb["MIXED_NOISE"]["B3_MULTI_CHANNEL_TRACKER"]["disentanglement_macro_pr_auc"],
        wm["WM_Small_ST_Transformer_Full"]["MIXED_ALL_THREE"]["disentanglement_macro_pr_auc"],
        wm["WM_R2D2_Graph_RSSM_Full"]["MIXED_ALL_THREE"]["disentanglement_macro_pr_auc"],
        1.000,
    ]
    colors_p1 = ["#9ca3af", "#60a5fa", "#38bdf8", "#f59e0b", "#8b5cf6", "#10b981", "#111827"]
    bars = axes[0].bar(names_p1, vals_p1, color=colors_p1, edgecolor="black", linewidth=0.7)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Macro PR-AUC (Higher = Better)")
    axes[0].set_title("(a) Mixed-Noise Mechanism Disentanglement", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=28, labelsize=8)
    for b in bars:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.3f}", ha="center", fontsize=7.5)

    # Panel B: A1 Unseen 3-Gate OOD Concurrency Crosstalk RMSE
    names_p2 = [
        "B0 Static",
        "B1 Static\nCorrelated",
        "B3 Classical\nSchedule",
        "WM: No\nSchedule",
        "WM: Small\nST-Transformer",
        "WM: R2D2\nGraph-RSSM",
    ]
    vals_p2 = [
        cb["A1_GATE_CROSSTALK"]["B0_STATIC"]["pairwise_joint_rmse_ood"],
        cb["A1_GATE_CROSSTALK"]["B1_STATIC_CORR"]["pairwise_joint_rmse_ood"],
        cb["A1_GATE_CROSSTALK"]["B3_SCHEDULE_CONDITIONED"]["pairwise_joint_rmse_ood"],
        wm["Ablation_No_Schedule_Cond"]["A1_GATE_CROSSTALK"]["pairwise_corr_rmse_ood"],
        wm["WM_Small_ST_Transformer_Full"]["A1_GATE_CROSSTALK"]["pairwise_corr_rmse_ood"],
        wm["WM_R2D2_Graph_RSSM_Full"]["A1_GATE_CROSSTALK"]["pairwise_corr_rmse_ood"],
    ]
    colors_p2 = ["#9ca3af", "#f87171", "#f59e0b", "#ef4444", "#8b5cf6", "#10b981"]
    bars2 = axes[1].bar(names_p2, vals_p2, color=colors_p2, edgecolor="black", linewidth=0.7)
    axes[1].set_ylabel("Pairwise Joint RMSE (Lower = Better)")
    axes[1].set_title("(b) A1 Unseen OOD Concurrency Crosstalk Error", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=28, labelsize=8)
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{b.get_height():.4f}", ha="center", fontsize=7.5)

    # Panel C: Multi-Horizon Forecast NLL across H in {1, 2, 4, 8}
    horizons = [1, 2, 4, 8]
    r2d2_nll = [wm["WM_R2D2_Graph_RSSM_Full"]["MIXED_ALL_THREE"]["forecast_nll"][f"H{h}"] for h in horizons]
    st_nll = [wm["WM_Small_ST_Transformer_Full"]["MIXED_ALL_THREE"]["forecast_nll"][f"H{h}"] for h in horizons]
    no_mem_nll = [wm["Ablation_No_Temporal_Memory"]["MIXED_ALL_THREE"]["forecast_nll"][f"H{h}"] for h in horizons]
    no_burn_nll = [wm["Ablation_No_R2D2_BurnIn"]["MIXED_ALL_THREE"]["forecast_nll"][f"H{h}"] for h in horizons]

    axes[2].plot(horizons, r2d2_nll, "o-", color="#10b981", lw=2.2, label="WM: R2D2-Graph-RSSM (Full)")
    axes[2].plot(horizons, st_nll, "s--", color="#8b5cf6", lw=2.0, label="WM: Small ST-Transformer")
    axes[2].plot(horizons, no_burn_nll, "^-.", color="#f59e0b", lw=1.8, label="Ablation: No R2D2 Burn-In")
    axes[2].plot(horizons, no_mem_nll, "x:", color="#ef4444", lw=1.8, label="Ablation: No Temporal Memory")
    axes[2].set_xlabel("Forecast Horizon H (QEC Rounds)")
    axes[2].set_ylabel("Predictive Detection NLL (Lower = Better)")
    axes[2].set_title("(c) Multi-Horizon Noise Forecasting (H=1..8)", fontweight="bold")
    axes[2].set_xticks(horizons)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(EVIDENCE_DIR / "01_world_model_vs_classical_baselines.png", dpi=300)
    plt.close(fig)

    # Plot 2: Complete 7-Variant Ablation Study Dashboard
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)
    variant_keys = [
        "WM_R2D2_Graph_RSSM_Full",
        "WM_Small_ST_Transformer_Full",
        "Ablation_No_R2D2_BurnIn",
        "Ablation_No_Spatial_Graph",
        "Ablation_No_Temporal_Memory",
        "Ablation_No_Schedule_Cond",
        "Ablation_Syndrome_Only",
    ]
    variant_labels = [
        "R2D2-Graph-RSSM\n(Full)",
        "Small ST-Transformer\n(Full)",
        "No R2D2 Burn-In\n(burn_in=0)",
        "No Spatial Graph\n(spatial=none)",
        "No Temporal Memory\n(temporal=none)",
        "No Schedule Cond\n(use_schedule=False)",
        "Syndrome Only\n(use_analog=False)",
    ]

    leak_scores = [wm[k]["B1_TRANSMON_LEAKAGE"]["leakage_pr_auc"] for k in variant_keys]
    burst_scores = [wm[k]["A2_CHIPWIDE_EVENTS"]["event_pr_auc"] for k in variant_keys]
    mix_scores = [wm[k]["MIXED_ALL_THREE"]["disentanglement_macro_pr_auc"] for k in variant_keys]

    x = np.arange(len(variant_keys))
    w = 0.26
    axes[0].bar(x - w, leak_scores, width=w, label="B1 Leakage PR-AUC", color="#3b82f6", edgecolor="black", lw=0.6)
    axes[0].bar(x, burst_scores, width=w, label="A2 Burst PR-AUC", color="#f59e0b", edgecolor="black", lw=0.6)
    axes[0].bar(x + w, mix_scores, width=w, label="Mixed Macro PR-AUC", color="#10b981", edgecolor="black", lw=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variant_labels, rotation=25, fontsize=7.5)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("PR-AUC Score")
    axes[0].set_title("(a) Latent Noise Identification Across Architectural Ablations", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    c_h8_rmse = [wm[k]["C1_C2_TLS_AND_IDLE_MEMORY"]["rate_rmse_H8"] for k in variant_keys]
    a2_fp_rmse = [wm[k]["A2_CHIPWIDE_EVENTS"]["spatial_footprint_rmse"] for k in variant_keys]
    axes[1].bar(x - w / 2, c_h8_rmse, width=w, label="C1/C2 H=8 Drift/TLS RMSE", color="#8b5cf6", edgecolor="black", lw=0.6)
    axes[1].bar(x + w / 2, a2_fp_rmse, width=w, label="A2 Spatial Footprint RMSE", color="#ec4899", edgecolor="black", lw=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variant_labels, rotation=25, fontsize=7.5)
    axes[1].set_ylabel("Latent Rate / Footprint RMSE (Lower = Better)")
    axes[1].set_title("(b) Long-Horizon Memory & Spatial Footprint Error", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(EVIDENCE_DIR / "02_r2d2_rssm_vs_st_transformer_and_ablations.png", dpi=300)
    plt.close(fig)


def run_full_evaluation() -> Dict[str, Any]:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    trained_models, training_summaries, splits = run_training_pipeline()

    with open(BASELINE_JSON_PATH, "r", encoding="utf-8") as f:
        baseline_ref = json.load(f)

    wm_eval: Dict[str, Any] = {}
    for name, model in trained_models.items():
        print(f"[Eval] Evaluating {name} on held-out test_id and test_ood_schedule splits...")
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

    full_results = {
        "scope": "Noise Generation, Latent Noise-State Estimation, and Short-Horizon Forecasting Only (No QEC Decoding)",
        "training_summary": training_summaries["dataset_validation"],
        "model_Metadata": {
            k: {
                "num_parameters": v["num_parameters"],
                "training_time_sec": v["training_time_sec"],
                "final_val_loss": v["final_val_loss"],
                "inference_contract_verified": v["inference_contract_verified"],
            }
            for k, v in training_summaries["models"].items()
        },
        "world_model_variants": wm_eval,
        "classical_baselines_reference": classical_summary,
    }

    out_json = EVIDENCE_DIR / "world_model_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    generate_evaluation_plots(full_results)
    print(f"[Done] Saved evaluation results and plots to {EVIDENCE_DIR}")
    return full_results


if __name__ == "__main__":
    run_full_evaluation()
