#!/usr/bin/env python3
'''V1 Acceptance Gate Checks for Dual-Emulator & Runtime Evaluation.'''
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]


def verify_zero_scenario_shortcuts() -> Dict[str, Any]:
    solver_cpp = (REPO_ROOT / "software/src/estimator/lbfgs_solver.cpp").read_text(encoding="utf-8")
    r2d2_h = (REPO_ROOT / "software/src/control/r2d2_lstm_controller.h").read_text(encoding="utf-8")
    forbidden = ["stationary_noise", "slowly_drifting_noise", "correlated_noise", "burst_disturbance"]
    solver_clean = not any(tok in solver_cpp for tok in forbidden)
    r2d2_clean = not any(tok in r2d2_h for tok in forbidden)
    return {
        "solver_contains_scenario_conditionals": not solver_clean,
        "r2d2_controller_contains_scenario_conditionals": not r2d2_clean,
        "causal_blind_estimation_verified": solver_clean and r2d2_clean,
    }
