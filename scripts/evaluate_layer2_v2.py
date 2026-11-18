#!/usr/bin/env python3
"""Evaluate an exported R2D2 V2 binary checkpoint across regimes, robustness sweeps, and ablations."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.evaluation.evaluate_ablation import run_representation_and_controller_ablation
from training.evaluation.evaluate_noise_regimes import evaluate_across_regimes_and_heldout_mixtures
from training.evaluation.evaluate_observation_robustness import evaluate_window_and_noise_robustness
from training.export.export_checkpoint import load_r2d2_checkpoint_bin


def main() -> None:
    bin_path = REPO_ROOT / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    net, _ = load_r2d2_checkpoint_bin(bin_path, expected_version=2)

    regimes = evaluate_across_regimes_and_heldout_mixtures(net)
    robust = evaluate_window_and_noise_robustness(net)
    ablation = run_representation_and_controller_ablation(net)

    out_path = REPO_ROOT / "results" / "v2" / "layer2_v2_standalone_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "checkpoint": str(bin_path.relative_to(REPO_ROOT)),
                "regime_evaluation": regimes,
                "robustness_evaluation": robust,
                "ablation_study": ablation,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] Wrote evaluation report -> {out_path}")


if __name__ == "__main__":
    main()
