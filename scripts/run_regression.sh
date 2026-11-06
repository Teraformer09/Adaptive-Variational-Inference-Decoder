#!/usr/bin/env bash
# =============================================================================
# scripts/run_regression.sh — Run Full AVID V1 Emulator Regression Suite
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${REPO_ROOT}/scripts/run_all.sh" "${REPO_ROOT}/configs/v1_smoke.yaml"
bash "${REPO_ROOT}/scripts/run_all.sh" "${REPO_ROOT}/configs/v1_rtl_regression.yaml"
bash "${REPO_ROOT}/scripts/run_all.sh" "${REPO_ROOT}/configs/v1_integration.yaml"

PYTHONPATH="${REPO_ROOT}" python3 - <<'PY'
import json
from pathlib import Path

repo = Path(".").resolve()
runs_dir = repo / "runs"
reports_dir = repo / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

summaries = []
for rep_file in sorted(runs_dir.glob("*/report.json")):
    data = json.loads(rep_file.read_text(encoding="utf-8"))
    summaries.append({
        "run_id": data.get("run_id"),
        "overall_status": data.get("overall_status"),
        "host_wall_clock_seconds": data.get("host_wall_clock_seconds"),
        "k3_bp_core_cycles": data.get("m1_verilator_standalone_rtl", {}).get("k3_bp_core_cycles"),
        "k3_bp_core_latency_ns": data.get("m1_verilator_standalone_rtl", {}).get("k3_bp_core_latency_ns"),
        "m3_transactions": data.get("m3_bridge_transaction_test", {}).get("total_transactions"),
        "scenario_count": len(data.get("m4_scenarios", [])),
    })

out = {
    "suite": "AVID_V1_Two_Emulator_Regression",
    "total_runs": len(summaries),
    "all_passed": all(s["overall_status"] == "PASS" for s in summaries),
    "runs": summaries,
}
(reports_dir / "regression_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(f"[Regression Summary] Wrote reports/regression_summary.json ({len(summaries)} runs)")
PY
