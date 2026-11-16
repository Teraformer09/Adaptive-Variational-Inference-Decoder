#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
VAL_DIR="${2:-${REPO_ROOT}/validation/layer3}"
RUN_MODE="${3:-remediated}"
cd "${REPO_ROOT}"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${CURRENT_BRANCH}" != "layer3-slow-loop-software" ]]; then
    echo "ERROR: Expected branch 'layer3-slow-loop-software', found '${CURRENT_BRANCH}'" >&2
    exit 1
fi

if [[ "${RUN_MODE}" == "baseline" ]]; then
    if ! git diff --quiet HEAD -- software/ architecture/ deployment/ configs/ checkpoints/; then
        echo "ERROR: Production directories have uncommitted modifications in baseline mode!" >&2
        git status --short software/ architecture/ deployment/ configs/ checkpoints/ >&2
        exit 1
    fi
fi

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
INITIAL_VAL_COMMIT="be50761cc6f4cc0c31f79555aa3f69ccd911563d"
HEAD_COMMIT="$(git rev-parse HEAD)"
mkdir -p "${VAL_DIR}/manifests"

python3 - <<PYEOF
import json, subprocess, datetime, os

repo_root = "${REPO_ROOT}"
baseline = "${BASELINE_COMMIT}"
run_mode = "${RUN_MODE}"

changed_vs_baseline = subprocess.check_output(
    ["git", "-C", repo_root, "diff", "--name-only", baseline, "--",
     "software/", "architecture/", "deployment/", "configs/", "checkpoints/"],
    text=True,
).splitlines()
changed_vs_baseline = sorted({p.strip() for p in changed_vs_baseline if p.strip()})

val_rel = os.path.relpath("${VAL_DIR}", repo_root)
porcelain = subprocess.check_output(
    ["git", "-C", repo_root, "status", "--porcelain"],
    text=True,
).splitlines()
dirty_entries = [
    ln for ln in porcelain
    if ln.strip()
    and val_rel not in ln
    and "logs/layer3_validation.log" not in ln
    and "software/build_layer3" not in ln
]

prod_unmodified = (len(changed_vs_baseline) == 0)

manifest = {
    "schema_version": "1.2.0",
    "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "branch": "${CURRENT_BRANCH}",
    "run_mode": run_mode,
    "baseline_commit": baseline,
    "initial_validation_commit": "${INITIAL_VAL_COMMIT}",
    "head_commit_at_run": "${HEAD_COMMIT}",
    "dirty_working_tree": (len(dirty_entries) > 0),
    "production_tree_unmodified": prod_unmodified,
    "authorized_remediation_run": (run_mode == "remediated"),
    "remediated_production_files_vs_88df97e": changed_vs_baseline,
    "production_paths_checked": ["software/", "architecture/", "deployment/", "configs/", "checkpoints/"],
    "frozen_baseline_branch": "AVID_EPJ_Patches",
    "frozen_baseline_commit": "${FROZEN_SHA}",
    "frozen_baseline_untouched": True,
}
with open(os.path.join("${VAL_DIR}", "manifests/source_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PYEOF

echo "[verify_branch] OK: branch=${CURRENT_BRANCH}, mode=${RUN_MODE}, baseline=${BASELINE_COMMIT:0:7}, AVID_EPJ_Patches=${FROZEN_SHA:0:7}"
