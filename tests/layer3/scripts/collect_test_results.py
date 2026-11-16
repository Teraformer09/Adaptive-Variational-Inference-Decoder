#!/usr/bin/env python3
"""
collect_test_results.py — Aggregates Set 1 Canonical, C11 ABI, Supplementary/Integration,
and ASan/UBSan results into structured manifests for the active validation directory.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=default_root)
    ap.add_argument("--validation-dir", default="")
    ap.add_argument("--sanitizer-ctest-rc", type=int, default=0)
    ap.add_argument("--run-id", default="remediated_set1_v1")
    args = ap.parse_args()

    r = os.path.abspath(args.repo_root)
    val_dir = os.path.abspath(args.validation_dir) if args.validation_dir else os.path.join(r, "validation/layer3")
    raw_dir = os.path.join(val_dir, "raw_results/set1")

    head_commit = subprocess.check_output(["git", "-C", r, "rev-parse", "HEAD"], text=True).strip()

    canonical_ids = [f"L3-IP-{i:03d}" for i in range(1, 41)]
    c_abi_ids = [f"L3-ABI-{i:03d}" for i in range(1, 8)]
    supp_ids = [
        "L3-DMA-INVALID-PKT",
        "L3-CFG-MISSING",
        "L3-INT-001",
        "L3-INT-002",
        "L3-INT-003",
        "L3-INT-004",
        "L3-INT-005",
    ]

    def load_group(id_list):
        out = []
        for tid in id_list:
            p = os.path.join(raw_dir, f"{tid}.json")
            if not os.path.exists(p):
                print(f"ERROR: Missing raw result JSON for {tid} at {p}", file=sys.stderr)
                sys.exit(1)
            with open(p, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        return out

    canonical_recs = load_group(canonical_ids)
    c_abi_recs = load_group(c_abi_ids)
    supp_recs = load_group(supp_ids)

    status_counts = {}
    for rec in canonical_recs:
        st = rec["status"]
        status_counts[st] = status_counts.get(st, 0) + 1

    c_abi_counts = {}
    for rec in c_abi_recs:
        st = rec["status"]
        c_abi_counts[st] = c_abi_counts.get(st, 0) + 1

    supp_counts = {}
    for rec in supp_recs:
        st = rec["status"]
        supp_counts[st] = supp_counts.get(st, 0) + 1

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    set1_summary = {
        "schema_version": "1.1.0",
        "generated_at_utc": now_utc,
        "run_id": args.run_id,
        "baseline_commit": "88df97e86b579b6e2865cc70dc4d0b08225d677b",
        "initial_validation_commit": "be50761cc6f4cc0c31f79555aa3f69ccd911563d",
        "source_commit": head_commit,
        "branch": "layer3-slow-loop-software",
        "stage": "set1_input_contracts",
        "sanitizer_ctest_exit_code": args.sanitizer_ctest_rc,
        "canonical_total": len(canonical_recs),
        "canonical_status_counts": status_counts,
        "c_abi_total": len(c_abi_recs),
        "c_abi_status_counts": c_abi_counts,
        "supplementary_total": len(supp_recs),
        "supplementary_status_counts": supp_counts,
        "canonical_results": canonical_recs,
        "c_abi_results": c_abi_recs,
        "supplementary_results": supp_recs,
    }

    res_dir = os.path.join(val_dir, "results")
    man_dir = os.path.join(val_dir, "manifests")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(man_dir, exist_ok=True)

    out_set1 = os.path.join(res_dir, "set1_input_contracts.json")
    with open(out_set1, "w", encoding="utf-8") as wf:
        json.dump(set1_summary, wf, indent=2)
        wf.write("\n")

    test_manifest = {
        "schema_version": "1.0.0",
        "generated_at_utc": now_utc,
        "run_id": args.run_id,
        "total_canonical_layer3_ids": 243,
        "executed_in_stage": {
            "canonical_set1_ids": canonical_ids,
            "c_abi_ids": c_abi_ids,
            "supplementary_and_integration_ids": supp_ids,
        },
        "deferred_future_sets": {
            "set2_output_statistics": "L3-OP-001..036, L3-STAT-001..015 (51 IDs)",
            "set3_pipeline_flow": "L3-FLOW-001..045 (45 IDs)",
            "set4_execution_recovery": "L3-EXEC-001..045 (45 IDs)",
            "set5_control_mathematics": "L3-MATH-001..062 (62 IDs)",
        },
    }
    with open(os.path.join(man_dir, "test_manifest.json"), "w", encoding="utf-8") as wf:
        json.dump(test_manifest, wf, indent=2)
        wf.write("\n")

    val_manifest = {
        "schema_version": "1.1.0",
        "generated_at_utc": now_utc,
        "run_id": args.run_id,
        "baseline_commit": "88df97e86b579b6e2865cc70dc4d0b08225d677b",
        "source_commit": head_commit,
        "results_file": os.path.relpath(out_set1, r),
        "results_sha256": sha256_file(out_set1),
        "canonical_status_counts": status_counts,
        "c_abi_status_counts": c_abi_counts,
        "supplementary_status_counts": supp_counts,
        "asan_ubsan_suite_executed": (args.sanitizer_ctest_rc == 0),
    }
    with open(os.path.join(man_dir, "validation_manifest.json"), "w", encoding="utf-8") as wf:
        json.dump(val_manifest, wf, indent=2)
        wf.write("\n")

    # Also copy flat convenience copies at root of run directory when val_dir is a run dir
    if os.path.basename(os.path.dirname(val_dir)) == "runs":
        shutil.copy2(out_set1, os.path.join(val_dir, "set1_input_contracts.json"))
        for mf in ("source_manifest.json", "environment_manifest.json", "build_manifest.json", "test_manifest.json", "validation_manifest.json"):
            src_mf = os.path.join(man_dir, mf)
            if os.path.exists(src_mf):
                shutil.copy2(src_mf, os.path.join(val_dir, mf))

    print(
        f"[collect_test_results] Aggregated {len(canonical_recs)} L3-IP ({status_counts}) + "
        f"{len(c_abi_recs)} C_ABI ({c_abi_counts}) + {len(supp_recs)} Integration/Supp ({supp_counts}) -> {out_set1}"
    )


if __name__ == "__main__":
    main()
