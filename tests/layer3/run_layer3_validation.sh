#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[run_layer3_validation] Executing authorized Stage 1 (Set 1 Input Contracts)..."
bash "${SCRIPT_DIR}/run_layer3_set1.sh"
echo "[run_layer3_validation] Stopping after Set 1 per mandatory Layer 3 stage gate (Sets 2-5 await review)."
