#!/usr/bin/env python3
'''V1 Acceptance Gate Checks for Dual-Emulator & Runtime Evaluation.'''
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_SCENARIO_TOKENS: List[str] = [
    "stationary_noise",
    "slowly_drifting_noise",
    "correlated_noise",
    "correlated_pauli_events",
    "burst_disturbance",
    "short_burst_60us_vs_10ms",
    "heldout_multi_pair_12_67",
    "heldout_spatial_3qubit_streak",
    "heldout_phenomenological_readout",
    "delayed_weight_commit",
    "busy_decoder",
]


def verify_zero_scenario_shortcuts() -> Dict[str, Any]:
    audited_files = [
        REPO_ROOT / "software/src/estimator/lbfgs_solver.cpp",
        REPO_ROOT / "software/src/estimator/llr_quantizer.h",
        REPO_ROOT / "software/src/inference/noise_model.cpp",
        REPO_ROOT / "software/src/control/r2d2_lstm_controller.h",
        REPO_ROOT / "software/src/control/causal_scheduler.h",
        REPO_ROOT / "software/src/control/fallback_fsm.h",
        REPO_ROOT / "deployment/cpp/r2d2_runtime.cpp",
    ]
    file_violations: Dict[str, List[str]] = {}
    for path in audited_files:
        txt = path.read_text(encoding="utf-8")
        hits = [tok for tok in FORBIDDEN_SCENARIO_TOKENS if tok in txt]
        if hits:
            file_violations[str(path.relative_to(REPO_ROOT))] = hits

    daemon_cpp = (REPO_ROOT / "software/src/inference/avid_daemon.cpp").read_text(encoding="utf-8")
    # Ensure sub-epoch adaptation in avid_daemon.cpp is purely causal and never checks scenario=="short_burst_60us_vs_10ms"
    subepoch_section = daemon_cpp.split("while (g_running)", 1)[-1]
    subepoch_shortcut = "short_burst_60us_vs_10ms" in subepoch_section

    solver_clean = "software/src/estimator/lbfgs_solver.cpp" not in file_violations
    r2d2_clean = "software/src/control/r2d2_lstm_controller.h" not in file_violations
    all_clean = (len(file_violations) == 0) and (not subepoch_shortcut)

    return {
        "audited_source_files": [str(p.relative_to(REPO_ROOT)) for p in audited_files],
        "file_violations": file_violations,
        "subepoch_contains_scenario_shortcut": subepoch_shortcut,
        "solver_contains_scenario_conditionals": not solver_clean,
        "r2d2_controller_contains_scenario_conditionals": not r2d2_clean,
        "causal_blind_estimation_verified": all_clean,
    }
