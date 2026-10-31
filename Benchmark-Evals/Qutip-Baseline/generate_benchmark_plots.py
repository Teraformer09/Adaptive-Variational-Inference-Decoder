#!/usr/bin/env python3
"""
Generate 300-DPI Publication Figures for AVID Benchmark-Evals (QuTiP Baseline Suite)
====================================================================================
Reads `/home/tirtha/Benchmark-Evals/results/baseline_evaluation_results.json` and
saves 4 multi-panel 300-DPI PNG figures in `/home/tirtha/Benchmark-Evals/plots/`.
All figures explicitly label simulation provenance (`[QuTiP Physical Reference & Baseline Evaluation]`).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = ROOT_DIR / "evidance" / "baseline_evaluation_results.json"
PLOTS_DIR = ROOT_DIR / "evidance"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "figure.dpi": 300,
})


def plot_01_b1_leakage(data: dict) -> Path:
    b1 = data["benchmarks"]["B1_TRANSMON_LEAKAGE"]
    trace = b1["sample_trace"]
    base = b1["baselines"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(
        "Benchmark B1 — Surface-17 Transmon CZ Leakage & HMM Inference (Varbanov et al. 2020)\n"
        "[QuTiP Physical Reference & Classical Baseline Evaluation]",
        fontsize=12, fontweight="bold", y=1.02
    )

    # Panel A: 50-Round Trajectory Posterior Tracking
    ax = axes[0]
    rounds = np.arange(len(trace["true_leaked_flags"]))
    ax.fill_between(rounds, 0, trace["true_leaked_flags"], color="#fca5a5", alpha=0.45, step="mid", label="True Hidden Leakage $L_i(t)$")
    ax.plot(rounds, trace["b2_hmm_analog"], color="#1d4ed8", lw=2.0, label="B2: HMM (Syndrome + Analog IQ)")
    ax.plot(rounds, trace["b2_hmm_syndrome"], color="#059669", lw=1.8, ls="--", label="B2: HMM (Syndrome Only)")
    ax.plot(rounds, trace["b3_avid"], color="#d97706", lw=1.5, ls=":", label="B3: Existing AVID EWMA")
    ax.axhline(0.5, color="#4b5563", lw=1.0, ls="-.")
    ax.set_title("(a) 50-Round Leakage State & Posterior Tracking")
    ax.set_xlabel("QEC Round $t$")
    ax.set_ylabel("Leakage Belief $\\hat{p}_{L,i}(t)$")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="upper right", fontsize=8)

    # Panel B: PR-AUC & F1 Comparison across B0, B3, B2(Synd), B2(+Analog), B5(Oracle)
    ax = axes[1]
    labels = ["B0: Static\nPauli", "B3: Existing\nAVID", "B2: HMM\n(Syndrome)", "B2: HMM\n(+Analog)", "B5: Oracle\nReference"]
    keys = ["B0_STATIC", "B3_AVID_VARIATIONAL", "B2_HMM_SYNDROME", "B2_HMM_ANALOG", "B5_ORACLE"]
    pr_aucs = [base[k]["leakage_pr_auc"] for k in keys]
    f1s = [base[k]["leakage_f1"] for k in keys]
    x = np.arange(len(labels))
    w = 0.36
    b_pr = ax.bar(x - w / 2, pr_aucs, w, color="#2563eb", label="PR-AUC")
    b_f1 = ax.bar(x + w / 2, f1s, w, color="#10b981", label="F1 Score (@ 0.5)")
    for bar in b_pr:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("(b) Leakage Inference PR-AUC & F1 Score")
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel C: Detection Delay (Rounds) vs Detection NLL
    ax = axes[2]
    eval_keys = ["B3_AVID_VARIATIONAL", "B2_HMM_SYNDROME", "B2_HMM_ANALOG"]
    eval_names = ["B3: Existing AVID", "B2: HMM (Syndrome)", "B2: HMM (+Analog)"]
    delays = [base[k]["mean_detection_delay_rounds"] for k in eval_keys]
    nlls = [base[k]["detection_nll"] for k in eval_keys]
    colors = ["#d97706", "#059669", "#1d4ed8"]
    bars = ax.barh(eval_names, delays, color=colors, height=0.48)
    for bar, nll in zip(bars, nlls):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f} rds (NLL={nll:.3f})", va="center", fontsize=8.5)
    ax.axvline(2.0, color="#dc2626", ls="--", lw=1.2, label="Varbanov <= 2 Cycle Target")
    ax.set_xlim(0, max(delays) + 1.8)
    ax.set_xlabel("Mean Leakage Detection Delay (QEC Rounds)")
    ax.set_title("(c) Leakage Detection Delay vs Target")
    ax.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    out = PLOTS_DIR / "01_B1_surface17_leakage_hmm_vs_baselines.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_02_a1_a2_spatial(data: dict) -> Path:
    a1 = data["benchmarks"]["A1_GATE_CROSSTALK"]
    a2 = data["benchmarks"]["A2_CHIPWIDE_EVENTS"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    fig.suptitle(
        "Benchmark Family A — Spatial Crosstalk (A1: Ketterer & Wellens 2023) & Chip-Wide Events (A2: Wilen et al. 2021)\n"
        "[QuTiP Pauli-Twirled Channel & Spatiotemporal Lindblad Reference]",
        fontsize=12, fontweight="bold", y=1.01
    )

    # Panel A: A1 Simultaneous vs Isolated RB Survival Curve
    ax = axes[0, 0]
    m_len = a1["physical_reference"]["rb_sequence_lengths"]
    ax.plot(m_len, a1["physical_reference"]["rb_survival_isolated"], "o-", color="#059669", lw=2, label="Isolated Schedule ($Q_1,Q_2$ only)")
    ax.plot(m_len, a1["physical_reference"]["rb_survival_concurrent"], "s-", color="#dc2626", lw=2, label="Concurrent Schedule ($Q_1,Q_2 + Q_3$ active)")
    ax.set_title("(a) A1: Simultaneous vs Isolated RB Survival Probability")
    ax.set_xlabel("RB Clifford Sequence Length $m$")
    ax.set_ylabel("Survival Probability $P(|00\\rangle)$")
    ax.legend(fontsize=8.5)

    # Panel B: A1 Joint-Event RMSE (In-Distribution vs Unseen Concurrency OOD)
    ax = axes[0, 1]
    b_keys = ["B0", "B1", "B3", "B5"]
    b_labels = ["B0: Static\nIndep.", "B1: Static\nCorrelated", "B3: AVID\nSchedule-Aware", "B5: Oracle\nReference"]
    rmse_id = [a1["baselines_in_distribution"][k]["joint_prob_rmse"] * 100 for k in b_keys]
    rmse_ood = [a1["baselines_unseen_concurrency_ood"][k]["joint_prob_rmse"] * 100 for k in b_keys]
    x = np.arange(len(b_keys))
    w = 0.35
    ax.bar(x - w / 2, rmse_id, w, color="#3b82f6", label="Held-Out Test (ID Schedules)")
    ax.bar(x + w / 2, rmse_ood, w, color="#f59e0b", label="Unseen 3-Gate Concurrency (OOD)")
    ax.set_xticks(x)
    ax.set_xticklabels(b_labels, fontsize=8.5)
    ax.set_ylabel("Pairwise Joint Fault RMSE (%)")
    ax.set_title("(b) A1: Schedule-Conditioned Joint Fault Error (ID vs OOD)")
    ax.legend(fontsize=8.5)

    # Panel C: A2 Spatiotemporal Relaxation Trajectory across 4 Qubits (0..1100 um)
    ax = axes[1, 0]
    probs_t = np.array(a2["sample_trace"]["true_relax_probs"])
    pos_um = a2["sample_trace"]["qubit_positions_um"]
    colors = ["#dc2626", "#ea580c", "#2563eb", "#7c3aed"]
    for q in range(4):
        ax.plot(np.arange(len(probs_t)), probs_t[:, q], lw=1.8, color=colors[q], label=f"$Q_{q+1}$ ($x={int(pos_um[q])}\\,\\mu$m)")
    for ev_t in a2["sample_trace"]["event_times"]:
        ax.axvline(ev_t, color="#111827", ls="--", lw=1.2, alpha=0.7)
    ax.set_title("(c) A2: Millimetre-Scale Phonon/Quasiparticle Relaxation Bursts")
    ax.set_xlabel("QEC Round $t$")
    ax.set_ylabel("Per-Round Relaxation Probability $P(T_1\\text{ fault})$")
    ax.legend(fontsize=8.5, ncol=2)

    # Panel D: A2 Event Detection PR-AUC, F1 & Footprint Error
    ax = axes[1, 1]
    a2_keys = ["B0_STATIC", "B1_LOCAL_CORR", "B3_AVID_SPATIAL_FILTER", "B5_ORACLE"]
    a2_names = ["B0: Static\nIndep.", "B1: Local\nPairwise", "B3: AVID\nSpatial Filter", "B5: Oracle\nReference"]
    pr_vals = [a2["baselines"][k]["event_pr_auc"] for k in a2_keys]
    f1_vals = [a2["baselines"][k]["event_f1"] for k in a2_keys]
    fp_err = [1.0 - a2["baselines"][k]["spatial_footprint_rmse"] for k in a2_keys]
    x = np.arange(len(a2_keys))
    w = 0.26
    ax.bar(x - w, pr_vals, w, color="#2563eb", label="Event PR-AUC")
    ax.bar(x, f1_vals, w, color="#10b981", label="Event F1 Score")
    ax.bar(x + w, fp_err, w, color="#8b5cf6", label="1 - Footprint RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(a2_names, fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("(d) A2: Chip-Wide Event Detection & Spatial Footprint Accuracy")
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    out = PLOTS_DIR / "02_A1_A2_spatial_crosstalk_and_chipwide_events.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_03_c1_c2_temporal(data: dict) -> Path:
    c1 = data["benchmarks"]["C1_DRIVEN_TLS"]
    c2 = data["benchmarks"]["C2_IDLE_MEMORY"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    fig.suptitle(
        "Benchmark Family C — Driven Qubit–TLS (C1: Agarwal et al. 2023) & Idle Memory (C2: Gulácsi & Burkard 2023)\n"
        "[QuTiP Composite Environment Evolution & Multi-Horizon Forecast Evaluation]",
        fontsize=12, fontweight="bold", y=1.01
    )

    # Panel A: C1 Mirrored Pseudoidentity U(theta) = X_{-theta} X_{+theta} vs Drive Amplitude
    ax = axes[0, 0]
    amps = c1["physical_reference"]["drive_amps_mhz"]
    ax.plot(amps, np.array(c1["physical_reference"]["markovian_error"]) * 100, "k--o", lw=2, label="Purely Markovian Lindblad Model")
    palette = ["#2563eb", "#d97706", "#dc2626"]
    for (k, v), col in zip(c1["physical_reference"]["tls_curves"].items(), palette):
        ax.plot(amps, np.array(v) * 100, "s-", color=col, lw=2, label=f"Composite Qubit–TLS ({k.replace('_', '=')})")
    ax.set_title("(a) C1: Mirrored Pseudoidentity $U(\\theta)=X_{-\\theta}X_{\\theta}$ Residual Error")
    ax.set_xlabel("Microwave Drive Amplitude $\\Omega / 2\\pi$ (MHz)")
    ax.set_ylabel("Residual Population $P(|1\\rangle)$ after $U(\\theta)$ (%)")
    ax.legend(fontsize=8)

    # Panel B: C1 Multi-Horizon Forecast RMSE across H in {1, 2, 4, 8}
    ax = axes[0, 1]
    horizons = [1, 2, 4, 8]
    h_keys = [f"H={h}" for h in horizons]
    for mode, label, col, mk in [
        ("B0_STATIC", "B0: Static Independent", "#6b7280", "o--"),
        ("B4_AR1_OU_FILTER", "B4: Classical AR(1)/OU Filter (Drive-Blind)", "#d97706", "s-"),
        ("B3_AVID_DRIVE_AWARE", "B3: AVID Schedule/Drive-Aware Estimator", "#1d4ed8", "^-"),
    ]:
        rmses = [c1["horizon_metrics"][hk][mode]["latent_rate_rmse"] * 100 for hk in h_keys]
        ax.plot(horizons, rmses, mk, color=col, lw=2, label=label)
    ax.set_xticks(horizons)
    ax.set_title("(b) C1: Multi-Horizon Latent Error Forecast RMSE ($H \\in \\{1,2,4,8\\}$)")
    ax.set_xlabel("Forecast Horizon $H$ (QEC Rounds)")
    ax.set_ylabel("Latent Error Rate RMSE (%)")
    ax.legend(fontsize=8)

    # Panel C: C2 Idle Transmon Coherence Revivals & Negative Canonical TCL Decay Rate
    ax = axes[1, 0]
    t_us = np.array(c2["physical_reference"]["t_us"])
    coh_amp = np.array(c2["physical_reference"]["coherence_amp"])
    coh_mk = np.array(c2["physical_reference"]["coherence_markov"])
    gamma_tcl = np.array(c2["physical_reference"]["gamma_tcl"])
    ax.plot(t_us, coh_amp, color="#1d4ed8", lw=2.2, label="Composite Qubit–Bath $|\\langle\\sigma_+(t)\\rangle|$ (Revivals)")
    ax.plot(t_us, coh_mk, color="#4b5563", ls="--", lw=1.6, label="Best-Fit Markovian Exponential")
    ax.fill_between(t_us, 0, coh_amp, where=(gamma_tcl < 0), color="#10b981", alpha=0.28, label="Non-Markovian Backflow ($\\gamma_{\\mathrm{TCL}}(t) < 0$)")
    ax.set_title("(c) C2: Idle Coherence Revivals & Negative TCL Decay Rate Regions")
    ax.set_xlabel("Idle Evolution Time $t$ ($\\mu$s)")
    ax.set_ylabel("Coherence Amplitude $|\\langle X \\rangle + i\\langle Y \\rangle|$")
    ax.legend(fontsize=8)

    # Panel D: C2 Multi-Horizon Idle-Fault Prediction Error
    ax = axes[1, 1]
    for mode, label, col, mk in [
        ("B0_MARKOVIAN_EXP_RMSE", "B0: Markovian Exponential Fit", "#6b7280", "o--"),
        ("B4_AR1_OU_RMSE", "B4: First-Order AR(1)/OU Filter", "#d97706", "s-"),
        ("B2_MEMORY_KERNEL_RMSE", "B2: 2nd-Order Memory Kernel Estimator", "#059669", "^-"),
    ]:
        vals = [c2["horizon_metrics"][hk][mode] * 100 for hk in h_keys]
        ax.plot(horizons, vals, mk, color=col, lw=2, label=label)
    ax.set_xticks(horizons)
    ax.set_title("(d) C2: Idle Revival Forecast RMSE vs Horizon $H$")
    ax.set_xlabel("Forecast Horizon $H$ (Idle Check Rounds)")
    ax.set_ylabel("Idle Fault Probability RMSE (%)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = PLOTS_DIR / "03_C1_C2_nonstationary_tls_and_idle_memory.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_04_mixed_noise_estimation(data: dict) -> Path:
    mx = data["priority5_mixed_noise_estimation"]
    models = [
        ("B0_STATIC_INDEP", "B0: Static\nIndep.", "#6b7280"),
        ("B1_STATIC_CORR", "B1: Static\nCorrelated", "#f59e0b"),
        ("B2_SINGLE_HMM", "B2: Single\nHMM", "#8b5cf6"),
        ("B4_AR1_OU_FILTER", "B4: AR(1)/OU\nFilter", "#d97706"),
        ("B3_MULTI_CHANNEL_TRACKER", "B3: Schedule+Analog\nVariational", "#2563eb"),
        ("B5_ORACLE_REF", "B5: Oracle\nGround Truth", "#10b981"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    fig.suptitle(
        "Priority 5 — Mixed-Noise Trajectory Generation, Latent Mechanism Disentanglement & Channel Ablations\n"
        "[Simultaneous Crosstalk (A1) + Chip-Wide Burst (A2) + Leakage (B1) + Temporal Drift (C1/C2) — Noise Estimation Only]",
        fontsize=11.5, fontweight="bold", y=1.01
    )

    # Panel A: Sample Mixed-Noise Trajectory Fault Rate & Coexisting Causes
    ax = axes[0, 0]
    tr = mx["sample_trace"]
    rates = np.array(tr["mean_effective_fault_rate"])
    causes = np.array(tr["active_causes_xtalk_leak_burst"])
    rds = np.arange(len(rates))
    ax.plot(rds, rates, color="#111827", lw=2.0, label="True Combined Fault Rate $\\bar{p}_t$")
    ax.fill_between(rds, 0, rates, where=(causes[:, 0] == 1), color="#93c5fd", alpha=0.35, step="mid", label="Concurrent Schedule Active (A1)")
    ax.fill_between(rds, 0, rates, where=(causes[:, 1] == 1), color="#fca5a5", alpha=0.55, step="mid", label="Transmon Leaked $|2\\rangle$ (B1)")
    ax.fill_between(rds, 0, rates, where=(causes[:, 2] == 1), color="#fde047", alpha=0.55, step="mid", label="Chip-Wide Phonon Burst (A2)")
    ax.set_title("(a) Generated Mixed-Noise Trajectory & Coexisting Causes")
    ax.set_xlabel("QEC Round $t$")
    ax.set_ylabel("Mean Effective Check Fault Rate")
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: Latent Mechanism Disentanglement Macro PR-AUC across B0..B5
    ax = axes[0, 1]
    names = [m[1] for m in models]
    macro_pr = [mx["models"][m[0]]["disentanglement_macro_pr_auc"] for m in models]
    colors = [m[2] for m in models]
    bars = ax.bar(np.arange(len(models)), macro_pr, color=colors, width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Macro PR-AUC (Crosstalk / Leakage / Burst)")
    ax.set_title("(b) Latent Noise Cause Disentanglement under Mixed Noise")

    # Panel C: Multi-Horizon Latent Noise Rate RMSE across H in {1, 2, 4, 8}
    ax = axes[1, 0]
    horizons = [1, 2, 4, 8]
    h_keys = [f"H={h}" for h in horizons]
    for m_key, m_label, col in models[:-1]:
        vals = [mx["models"][m_key]["horizons"][hk]["latent_rate_rmse"] * 100 for hk in h_keys]
        ax.plot(horizons, vals, "o-", color=col, lw=2, label=m_label.replace("\n", " "))
    ax.set_xticks(horizons)
    ax.set_xlabel("Forecast Horizon $H$ (Rounds)")
    ax.set_ylabel("Latent Fault-Rate Forecast RMSE (%)")
    ax.set_title("(c) Multi-Horizon Mixed-Noise Forecast RMSE ($H \\in \\{1,2,4,8\\}$)")
    ax.legend(fontsize=8)

    # Panel D: Observation-Channel Ablation Study
    ax = axes[1, 1]
    abl = mx["channel_ablations"]
    abl_keys = list(abl.keys())
    abl_labels = ["Syndrome Only\n$(d_t, s_t)$", "Syndrome +\nSchedule $(u_t)$", "Full Contract\n$(d_t, u_t, c_t, a_t)$", "Oracle Truth\n$(x_t)$"]
    nll_vals = [abl[k]["H1_detection_nll"] for k in abl_keys]
    pr_vals = [abl[k]["disentanglement_macro_pr_auc"] for k in abl_keys]
    x = np.arange(len(abl_keys))
    w = 0.35
    ax.bar(x - w / 2, nll_vals, w, color="#dc2626", label="$H=1$ Detection NLL (Lower Better)")
    ax.bar(x + w / 2, pr_vals, w, color="#2563eb", label="Cause Disentangle PR-AUC (Higher Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(abl_labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Metric Value")
    ax.set_title("(d) Observation-Channel Ablation (Why $u_t, c_t, a_t$ Matter)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = PLOTS_DIR / "04_mixed_noise_generation_and_latent_estimation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
    p1 = plot_01_b1_leakage(data)
    p2 = plot_02_a1_a2_spatial(data)
    p3 = plot_03_c1_c2_temporal(data)
    p4 = plot_04_mixed_noise_estimation(data)
    print("Generated 300-DPI benchmark evaluation plots (Noise Generation & Estimation Only):")
    for p in (p1, p2, p3, p4):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
