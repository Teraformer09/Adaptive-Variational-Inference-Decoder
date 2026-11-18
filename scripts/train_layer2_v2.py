#!/usr/bin/env python3
"""Train R2D2 V2 (82-input, 38,672 trainable params) and export `checkpoints/v2/r2d2_v2_trained.*`."""

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
from training.evaluation.evaluate_policy import run_v2_training_and_export


def main() -> None:
    res = run_v2_training_and_export(repo_root=REPO_ROOT, num_updates=240, batch_size=16)
    net = res["network"]

    regime_eval = evaluate_across_regimes_and_heldout_mixtures(net)
    robust_eval = evaluate_window_and_noise_robustness(net)
    ablation_eval = run_representation_and_controller_ablation(net)

    out_dir = REPO_ROOT / "results" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "AVID_R2D2_Dueling_LSTM_V2",
        "schema_version": 2,
        "input_dim": 82,
        "trainable_parameters": net.num_trainable_parameters,
        "total_serialized_floats": net.num_serialized_floats,
        "training_metrics": res["training_metrics"],
        "regime_and_heldout_evaluation": regime_eval,
        "robustness_evaluation": robust_eval,
        "ablation_study": ablation_eval,
    }
    (out_dir / "layer2_v2_evaluation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[PASS] Trained & exported R2D2 V2 ({net.num_trainable_parameters} params): "
        f"Test LER={res['training_metrics']['heldout_test_metrics']['mean_ler']:.5f} vs "
        f"Static={res['training_metrics']['heldout_test_metrics']['static_ler']:.5f} "
        f"(-{res['training_metrics']['heldout_test_metrics']['ler_reduction_pct']:.2f}%)"
    )


if __name__ == "__main__":
    main()
