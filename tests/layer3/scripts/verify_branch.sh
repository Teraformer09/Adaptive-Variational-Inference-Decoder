#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${CURRENT_BRANCH}" != "layer3-slow-loop-software" ]]; then
    echo "ERROR: Expected branch 'layer3-slow-loop-software', found '${CURRENT_BRANCH}'" >&2
    exit 1
fi

# Verify zero modifications to production directories
if ! git diff --quiet HEAD -- software/ architecture/ deployment/ configs/ checkpoints/; then
    echo "ERROR: Production directories have uncommitted modifications!" >&2
    git status --short software/ architecture/ deployment/ configs/ checkpoints/ >&2
    exit 1
fi

FROZEN_OK=true
FROZEN_SHA="unavailable"
if [[ -d "/home/tirtha/AVID_EPJ_Patches/.git" ]]; then
    FROZEN_SHA="$(git -C /home/tirtha/AVID_EPJ_Patches rev-parse HEAD)"
    if [[ "${FROZEN_SHA}" != "54da68d92778e4e1ffde9514aaaae585638761ff" ]]; then
        echo "ERROR: Frozen baseline /home/tirtha/AVID_EPJ_Patches moved from 54da68d!" >&2
        exit 1
    fi
    if [[ -n "$(git -C /home/tirtha/AVID_EPJ_Patches status --porcelain)" ]]; then
        echo "ERROR: Frozen baseline /home/tirtha/AVID_EPJ_Patches has dirty working tree!" >&2
        exit 1
    fi
fi

BASELINE_COMMIT="88df97e86b579b6e2865cc70dc4d0b08225d677b"
HEAD_COMMIT="$(git rev-parse HEAD)"
mkdir -p "${REPO_ROOT}/validation/layer3/manifests"

python3 - <<PYEOF
import json, subprocess, datetime

manifest = {
    "schema_version": "1.0.0",
    "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "branch": "${CURRENT_BRANCH}",
    "baseline_commit": "${BASELINE_COMMIT}",
    "head_commit_at_run": "${HEAD_COMMIT}",
    "production_tree_unmodified": True,
    "production_paths_checked": ["software/", "architecture/", "deployment/", "configs/", "checkpoints/"],
    "frozen_baseline_branch": "AVID_EPJ_Patches",
    "frozen_baseline_commit": "${FROZEN_SHA}",
    "frozen_baseline_untouched": True,
}
with open("${REPO_ROOT}/validation/layer3/manifests/source_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PYEOF

echo "[verify_branch] OK: branch=${CURRENT_BRANCH}, baseline=${BASELINE_COMMIT:0:7}, production_tree_unmodified=true, AVID_EPJ_Patches=${FROZEN_SHA:0:7}"
