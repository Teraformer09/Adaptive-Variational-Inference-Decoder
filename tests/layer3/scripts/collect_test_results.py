#!/usr/bin/env python3
"""
collect_test_results.py — Aggregates Layer 3 Validation Results.

Always writes `results/set1_input_contracts.json` for Set 1 backward compatibility.
When `raw_results/set2..set5` exist in `--validation-dir`, aggregates all 243 canonical
IDs (`40 L3-IP + 36 L3-OP + 15 L3-STAT + 45 L3-FLOW + 45 L3-EXEC + 62 L3-MATH`) plus
`7 L3-ABI` + `5 L3-INT` + `2 diagnostic companions` (`257` total records) into
`results/layer3_full_results.json` and updates `manifests/test_manifest.json` &
`manifests/validation_manifest.json`.
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


def count_statuses(recs):
    counts = {}
    for rec in recs:
        st = rec["status"]
        counts[st] = counts.get(st, 0) + 1
    return counts


def main():
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=default_root)
    ap.add_argument("--validation-dir", default="")
    ap.add_argument("--sanitizer-ctest-rc", type=int, default=0)
    ap.add_argument("--run-id", default="full_layer3_v1")
    args = ap.parse_args()

    r = os.path.abspath(args.repo_root)
    val_dir = os.path.abspath(args.validation_dir) if args.validation_dir else os.path.join(r, "validation/layer3")
    raw_set1 = os.path.join(val_dir, "raw_results/set1")

    head_commit = subprocess.check_output(["git", "-C", r, "rev-parse", "HEAD"], text=True).strip()

    ip_ids = [f"L3-IP-{i:03d}" for i in range(1, 41)]
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

    def load_group(raw_subdir, id_list):
        out = []
        for tid in id_list:
            p = os.path.join(raw_subdir, f"{tid}.json")
            if not os.path.exists(p):
                print(f"ERROR: Missing raw result JSON for {tid} at {p}", file=sys.stderr)
                sys.exit(1)
            with open(p, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        return out

    ip_recs = load_group(raw_set1, ip_ids)
    c_abi_recs = load_group(raw_set1, c_abi_ids)
    supp_recs = load_group(raw_set1, supp_ids)

    ip_counts = count_statuses(ip_recs)
    c_abi_counts = count_statuses(c_abi_recs)
    supp_counts = count_statuses(supp_recs)

    has_full_sets = all(
        os.path.isdir(os.path.join(val_dir, f"raw_results/set{s}"))
        for s in (2, 3, 4, 5)
    )

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    set1_summary = {
        "schema_version": "1.2.0",
        "generated_at_utc": now_utc,
        "run_id": args.run_id,
        "baseline_commit": "88df97e86b579b6e2865cc70dc4d0b08225d677b",
        "initial_validation_commit": "be50761cc6f4cc0c31f79555aa3f69ccd911563d",
        "source_commit": head_commit,
        "branch": "layer3-slow-loop-software",
        "stage": "full_layer3_sets1_to_5" if has_full_sets else "set1_input_contracts",
        "sanitizer_ctest_exit_code": args.sanitizer_ctest_rc,
        "canonical_total": len(ip_recs),
        "canonical_status_counts": ip_counts,
        "c_abi_total": len(c_abi_recs),
        "c_abi_status_counts": c_abi_counts,
        "supplementary_total": len(supp_recs),
        "supplementary_status_counts": supp_counts,
        "canonical_results": ip_recs,
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

    if not has_full_sets:
        test_manifest = {
            "schema_version": "1.0.0",
            "generated_at_utc": now_utc,
            "run_id": args.run_id,
            "total_canonical_layer3_ids": 243,
            "executed_in_stage": {
                "canonical_set1_ids": ip_ids,
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
        val_manifest = {
            "schema_version": "1.1.0",
            "generated_at_utc": now_utc,
            "run_id": args.run_id,
            "baseline_commit": "88df97e86b579b6e2865cc70dc4d0b08225d677b",
            "source_commit": head_commit,
            "results_file": os.path.relpath(out_set1, r),
            "results_sha256": sha256_file(out_set1),
            "canonical_status_counts": ip_counts,
            "c_abi_status_counts": c_abi_counts,
            "supplementary_status_counts": supp_counts,
            "asan_ubsan_suite_executed": (args.sanitizer_ctest_rc == 0),
        }
        out_primary = out_set1
    else:
        op_ids = [f"L3-OP-{i:03d}" for i in range(1, 37)]
        stat_ids = [f"L3-STAT-{i:03d}" for i in range(1, 16)]
        flow_ids = [f"L3-FLOW-{i:03d}" for i in range(1, 46)]
        exec_ids = [f"L3-EXEC-{i:03d}" for i in range(1, 46)]
        math_ids = [f"L3-MATH-{i:03d}" for i in range(1, 63)]

        set2_recs = load_group(os.path.join(val_dir, "raw_results/set2"), op_ids + stat_ids)
        set3_recs = load_group(os.path.join(val_dir, "raw_results/set3"), flow_ids)
        set4_recs = load_group(os.path.join(val_dir, "raw_results/set4"), exec_ids)
        set5_recs = load_group(os.path.join(val_dir, "raw_results/set5"), math_ids)

        all_canonical = ip_recs + set2_recs + set3_recs + set4_recs + set5_recs
        all_canonical_counts = count_statuses(all_canonical)
        per_set_counts = {
            "set1_input_contracts": ip_counts,
            "set2_output_statistics": count_statuses(set2_recs),
            "set3_pipeline_flow": count_statuses(set3_recs),
            "set4_execution_recovery": count_statuses(set4_recs),
            "set5_control_mathematics": count_statuses(set5_recs),
        }

        full_summary = {
            "schema_version": "2.0.0",
            "generated_at_utc": now_utc,
            "run_id": args.run_id,
            "baseline_commit": "88df97e86b579b6e2865cc70dc4d0b08225d677b",
            "initial_validation_commit": "be50761cc6f4cc0c31f79555aa3f69ccd911563d",
            "source_commit": head_commit,
            "branch": "layer3-slow-loop-software",
            "stage": "full_layer3_sets1_to_5",
            "sanitizer_ctest_exit_code": args.sanitizer_ctest_rc,
            "canonical_total": len(all_canonical),
            "canonical_status_counts": all_canonical_counts,
            "per_set_status_counts": per_set_counts,
            "c_abi_total": len(c_abi_recs),
            "c_abi_status_counts": c_abi_counts,
            "supplementary_total": len(supp_recs),
            "supplementary_status_counts": supp_counts,
            "total_records": len(all_canonical) + len(c_abi_recs) + len(supp_recs),
            "canonical_results": all_canonical,
            "c_abi_results": c_abi_recs,
            "supplementary_results": supp_recs,
        }
        out_full = os.path.join(res_dir, "layer3_full_results.json")
        with open(out_full, "w", encoding="utf-8") as wf:
            json.dump(full_summary, wf, indent=2)
            wf.write("\n")

        test_manifest = {
            "schema_version": "2.0.0",
            "generated_at_utc": now_utc,
            "run_id": args.run_id,
            "total_canonical_layer3_ids": 243,
            "executed_in_stage": {
                "canonical_set1_ids": ip_ids,
                "canonical_set2_ids": op_ids + stat_ids,
                "canonical_set3_ids": flow_ids,
                "canonical_set4_ids": exec_ids,
                "canonical_set5_ids": math_ids,
                "c_abi_ids": c_abi_ids,
                "supplementary_and_integration_ids": supp_ids,
            },
            "deferred_future_sets": {},
        }
        val_manifest = {
            "schema_version": "2.0.0",
            "generated_at_utc": now_utc,
            "run_id": args.run_id,
            "baseline_commit": "88df97e86b579b6e2865cc70dc4d0b08225d677b",
            "source_commit": head_commit,
            "results_file": os.path.relpath(out_full, r),
            "results_sha256": sha256_file(out_full),
            "set1_results_file": os.path.relpath(out_set1, r),
            "set1_results_sha256": sha256_file(out_set1),
            "canonical_status_counts": all_canonical_counts,
            "per_set_status_counts": per_set_counts,
            "c_abi_status_counts": c_abi_counts,
            "supplementary_status_counts": supp_counts,
            "asan_ubsan_suite_executed": (args.sanitizer_ctest_rc == 0),
        }
        out_primary = out_full

    with open(os.path.join(man_dir, "test_manifest.json"), "w", encoding="utf-8") as wf:
        json.dump(test_manifest, wf, indent=2)
        wf.write("\n")
    with open(os.path.join(man_dir, "validation_manifest.json"), "w", encoding="utf-8") as wf:
        json.dump(val_manifest, wf, indent=2)
        wf.write("\n")

    if os.path.basename(os.path.dirname(val_dir)) == "runs":
        shutil.copy2(out_set1, os.path.join(val_dir, "set1_input_contracts.json"))
        if has_full_sets:
            shutil.copy2(out_primary, os.path.join(val_dir, "layer3_full_results.json"))
        for mf in ("source_manifest.json", "environment_manifest.json", "build_manifest.json", "test_manifest.json", "validation_manifest.json"):
            src_mf = os.path.join(man_dir, mf)
            if os.path.exists(src_mf):
                shutil.copy2(src_mf, os.path.join(val_dir, mf))

    if has_full_sets:
        print(
            f"[collect_test_results] Aggregated 243/243 canonical IDs ({val_manifest['canonical_status_counts']}) + "
            f"{len(c_abi_recs)} C_ABI + {len(supp_recs)} Integration/Supp (257 total) -> {out_primary}"
        )
    else:
        print(
            f"[collect_test_results] Aggregated {len(ip_recs)} L3-IP ({ip_counts}) + "
            f"{len(c_abi_recs)} C_ABI ({c_abi_counts}) + {len(supp_recs)} Integration/Supp ({supp_counts}) -> {out_set1}"
        )


if __name__ == "__main__":
    main()
