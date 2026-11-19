"""Pytest Acceptance Verification for AVID Minimum Publication Gate (`test_publication_gate_suite.py`)."""

from __future__ import annotations

import json
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = RESEARCH_ROOT / "results" / "json"
PLOT_DIR = RESEARCH_ROOT / "results" / "plots"


def _load(name: str) -> dict:
    path = JSON_DIR / name
    assert path.is_file(), f"Missing required result JSON: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_group_a_noise_model_validity_gates() -> None:
    ga = _load("group_a_noise_validity.json")
    g = ga["gates"]
    assert g["A1_distribution_recovery"]["passed"] is True
    assert g["A2_correlation_recovery"]["passed"] is True
    assert g["A3_temporal_autocorrelation"]["passed"] is True
    assert g["A4_regime_transitions"]["passed"] is True
    assert g["A5_out_of_distribution_noise"]["passed"] is True
    assert g["A6_seed_reproducibility"]["passed"] is True
    assert g["A5_out_of_distribution_noise"]["aggregate_paired_stats"]["boot_ci95_low"] > 0.0
    assert ga["overall_passed"] is True


def test_group_b_adaptive_controller_and_paired_ci_gates() -> None:
    gb = _load("group_b_adaptive_performance.json")
    g = gb["gates"]
    assert g["B1_multiseed_training_stability"]["passed"] is True
    assert g["B2_unseen_noise_combinations"]["passed"] is True
    assert g["B3_abrupt_regime_switching"]["passed"] is True
    assert g["B4_gradual_parameter_drift"]["passed"] is True
    assert g["B5_and_B10_controller_ablation_with_paired_cis"]["passed"] is True
    assert g["B7_and_B11_posterior_corruption_test"]["passed"] is True
    assert g["B9_static_vs_adaptive_crossover"]["passed"] is True
    h2h = g["B5_and_B10_controller_ablation_with_paired_cis"]["head_to_head_paired_ci"][
        "Full_82D_vs_Memoryless_JointDP"
    ]
    assert h2h["significant_improvement_95pct"] is True
    assert gb["overall_passed"] is True


def test_group_c_bayesian_inference_and_controller_sufficiency_gates() -> None:
    gc = _load("group_c_bayesian_sufficiency.json")
    g = gc["gates"]
    assert g["C1_jacobian_rank_and_dual_basis"]["passed"] is True
    assert g["C2_prior_sensitivity"]["passed"] is True
    assert g["C3_posterior_calibration"]["passed"] is True
    assert g["C4_uncertainty_propagation"]["passed"] is True
    assert g["C5_coupler_endpoint_prior_ablation"]["passed"] is True
    assert g["C6_observable_orbit_recovery"]["passed"] is True
    assert g["C8_temporal_stacked_jacobian_identifiability"]["passed"] is True
    assert g["C9_four_way_controller_sufficiency"]["passed"] is True
    assert g["C10_prior_mismatch_robustness"]["passed"] is True
    assert g["C8_temporal_stacked_jacobian_identifiability"]["stationary_stack_rank"] == 9
    assert g["C8_temporal_stacked_jacobian_identifiability"]["nonstationary_operating_point_strict_rank"] == 19
    assert gc["overall_passed"] is True


def test_group_e_native_verilator_rtl_and_dualloop_isolation_gates() -> None:
    ge = _load("group_e_rtl_correctness.json")
    assert ge["native_verilator"] is True
    assert ge["simulator"] == "verilator_c++_rtl"
    g = ge["gates"]
    for key in (
        "E2_fixed_point_q78_equivalence",
        "E3_bp_mode0_golden_equivalence",
        "E4_osd0_degenerate_postprocessing",
        "E5_atomic_shadow_commit_streaming",
        "E6_mid_decode_commit_and_slverr_lock",
        "E7_elastic_fifo_full_empty_backpressure",
        "E8_synchronous_reset_recovery",
        "E9_10k_randomized_rtl_stress",
        "E10_cycle_budget_latency_law",
        "E11_weight_update_interference",
        "E12_fast_loop_300ns_worst_case_deadline",
    ):
        assert g[key]["passed"] is True, f"Gate failed: {key}"
    assert g["E11_weight_update_interference"]["max_abs_delta_cycles"] == 0
    assert g["E12_fast_loop_300ns_worst_case_deadline"]["max_service_latency_ns"] <= 300.0
    assert g["E12_fast_loop_300ns_worst_case_deadline"]["deadline_misses"] == 0
    assert ge["overall_passed"] is True


def test_group_f_full_system_closedloop_and_honest_timescales() -> None:
    gf = _load("group_f_full_system.json")
    g = gf["gates"]
    assert g["F1_and_F2_end_to_end_closedloop_and_cpp_parity"]["passed"] is True
    assert g["F3_mid_stream_slow_loop_commits_zero_stall"]["passed"] is True
    assert g["F3_mid_stream_slow_loop_commits_zero_stall"]["max_abs_mid_commit_delta_cycles"] == 0
    assert g["F4_and_F5_honest_dual_timescale_60us_vs_10ms"]["passed_F4"] is True
    assert g["F4_and_F5_honest_dual_timescale_60us_vs_10ms"]["passed_F5"] is True
    assert (
        g["F4_and_F5_honest_dual_timescale_60us_vs_10ms"]["sub_10ms_60us_microburst"][
            "slow_loop_commits_during_60us"
        ]
        == 0
    )
    assert g["F6_and_F9_long_60epoch_closedloop_rtl_stability"]["passed"] is True
    assert g["F7_readout_dropout_uncertainty_propagation_rtl"]["passed"] is True
    assert g["F10_full_system_rtl_paired_statistical_comparison"]["passed"] is True
    assert gf["overall_passed"] is True


def test_checklist_bm3_bm4_and_publication_plots_present() -> None:
    gbm = _load("checklist_bm3_bm4_scaling.json")
    assert gbm["gates"]["BM4_window_complexity_and_causal_regret"]["passed"] is True
    assert gbm["gates"]["BM3_distance_scaling_d3_vs_d5"]["passed"] is True
    assert gbm["overall_passed"] is True

    master = _load("publication_gate_master_summary.json")
    assert master["overall_publication_gate_passed"] is True

    stems = (
        "fig01_groupA_noise_validity_and_ood",
        "fig02_groupB_adaptive_performance_and_cis",
        "fig03_groupC_bayesian_sufficiency_and_temporal_svd",
        "fig04_groupE_rtl_correctness_and_dualloop_isolation",
        "fig05_groupF_full_system_closedloop_and_timescales",
        "fig06_checklist_window_regret_and_distance_scaling",
    )
    for st in stems:
        png_p = PLOT_DIR / f"{st}.png"
        pdf_p = PLOT_DIR / f"{st}.pdf"
        assert png_p.is_file() and png_p.stat().st_size > 25_000, f"Missing/tiny PNG: {png_p}"
        assert pdf_p.is_file() and pdf_p.stat().st_size > 10_000, f"Missing/tiny PDF: {pdf_p}"

    report_md = RESEARCH_ROOT / "results" / "AVID_Publication_Gate_Report.md"
    assert report_md.is_file() and report_md.stat().st_size > 2_000
