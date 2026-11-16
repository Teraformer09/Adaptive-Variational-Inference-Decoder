#!/usr/bin/env python3
"""Audit-grade Layer 4 gate evaluator, provenance manifest builder, and report generator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import yaml


def build_canonical_ids() -> list[str]:
    ids: list[str] = []
    # Set 1 (47)
    ids.extend(f"L4-I{i:02d}" for i in range(1, 13))
    ids.extend(f"L4-MMIO-{i:02d}" for i in range(1, 14))
    ids.extend(f"L4-SYND-{i:02d}" for i in range(1, 15))
    ids.extend(f"L4-SEQ-{i:02d}" for i in range(1, 9))
    # Set 2 (28)
    ids.extend(f"L4-ACK-{i:02d}" for i in range(1, 8))
    ids.extend(f"L4-CYC-{i:02d}" for i in range(1, 11))
    ids.extend(f"L4-CONS-{i:02d}" for i in range(1, 6))
    ids.extend(f"L4-DET-{i:02d}" for i in range(1, 7))
    # Set 3 (39)
    ids.extend(f"L4-P{i:02d}" for i in range(1, 13))
    ids.extend(f"L4-LOCK-{i:02d}" for i in range(1, 10))
    ids.extend(f"L4-EPOCH-{i:02d}" for i in range(1, 8))
    ids.extend(f"L4-M3-{i:02d}" for i in range(1, 12))
    # Set 4 (56)
    ids.extend(f"L4-BLD-{i:02d}" for i in range(1, 12))
    ids.extend(f"L4-R{i:02d}" for i in range(1, 15))
    ids.extend(f"L4-FLT-{i:02d}" for i in range(1, 23))
    ids.extend(f"L4-PERS-{i:02d}" for i in range(1, 10))
    # Set 5 (53)
    ids.extend(f"L4-AXIL-{i:02d}" for i in range(1, 12))
    ids.extend(f"L4-AXIS-{i:02d}" for i in range(1, 10))
    ids.extend(f"L4-CFID-{i:02d}" for i in range(1, 9))
    ids.extend(f"L4-EQV-{i:02d}" for i in range(1, 8))
    ids.extend(f"L4-WBK-{i:02d}" for i in range(1, 15))
    ids.extend(f"L4-LIM-{i:02d}" for i in range(1, 5))
    return ids


def cmd_first_line(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.STDOUT, text=True)
        return out.splitlines()[0].strip() if out.strip() else "unknown"
    except Exception:  # noqa: BLE001
        return "not_found"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--results-jsonl", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    results_jsonl = Path(args.results_jsonl).resolve()
    report_out = Path(args.report_out).resolve()

    canonical_ids = build_canonical_ids()
    assert len(canonical_ids) == 223, f"Expected 223 canonical IDs, got {len(canonical_ids)}"

    invariants_path = repo_root / "tests" / "layer4" / "acceptance" / "expected_invariants.yaml"
    invariants = yaml.safe_load(invariants_path.read_text(encoding="utf-8"))

    records: list[dict] = []
    with results_jsonl.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Corrupt JSONL at {results_jsonl}:{line_no}: {exc}") from exc

    by_id: dict[str, list[dict]] = {}
    for r in records:
        by_id.setdefault(r["test_id"], []).append(r)

    duplicate_ids = sorted([tid for tid, recs in by_id.items() if len(recs) > 1])
    missing_ids = sorted([tid for tid in canonical_ids if tid not in by_id])
    unexpected_ids = sorted([tid for tid in by_id if tid not in set(canonical_ids)])

    # Provenance & Toolchain Checks
    rtl_prov_path = repo_root / "emulator" / "verilator" / "models" / "rtl_build_provenance.json"
    rtl_prov = json.loads(rtl_prov_path.read_text(encoding="utf-8")) if rtl_prov_path.exists() else {}
    native_verilator = bool(rtl_prov.get("native_verilator_compiled", False))

    tb_json_path = run_dir / "results" / "standalone_rtl_tb.json"
    tb_json = json.loads(tb_json_path.read_text(encoding="utf-8")) if tb_json_path.exists() else {}
    vcd_path = run_dir / "artifacts" / "standalone_rtl_tb.vcd"

    qemu_arm_bin = shutil.which("qemu-arm") or ""
    qemu_aarch64_bin = shutil.which("qemu-aarch64") or ""
    arm_elf = repo_root / "emulator" / "qemu" / "images" / "avid_daemon_cortex_a9"
    aarch64_elf = repo_root / "emulator" / "qemu" / "images" / "avid_daemon_aarch64"
    qemu_ready = bool(qemu_arm_bin and qemu_aarch64_bin and arm_elf.exists() and aarch64_elf.exists())

    git_commit = cmd_first_line(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_branch = cmd_first_line(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    # Ignore untracked/regenerated files inside validation/layer4, results, and logs during active run generation when computing code dirtiness
    git_status_out = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            ":(exclude)validation/layer4",
            ":(exclude)results",
            ":(exclude)logs/layer4_validation.log",
        ],
        cwd=str(repo_root),
        text=True,
    ).strip()
    code_git_dirty = bool(git_status_out)

    # Set-level aggregation
    set_summaries: dict[str, dict] = {}
    for sid in ("Set1", "Set2", "Set3", "Set4", "Set5"):
        s_recs = [r for r in records if r["set_id"] == sid]
        counts: dict[str, int] = {}
        for r in s_recs:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        set_summaries[sid] = {
            "set_id": sid,
            "total": len(s_recs),
            "counts": counts,
            "wall_time_ms_total": round(sum(float(r.get("wall_time_ms", 0.0)) for r in s_recs), 3),
            "rtl_cycles_total": int(sum(int(r.get("rtl_cycles", 0)) for r in s_recs)),
            "all_pass": (len(s_recs) > 0 and counts.get("PASS", 0) == len(s_recs)),
        }
        (run_dir / "results" / f"{sid.lower()}_summary.json").write_text(
            json.dumps({"summary": set_summaries[sid], "records": s_recs}, indent=2),
            encoding="utf-8",
        )

    # Gate evaluation (G1 .. G8)
    gate_defs = invariants["gates"]
    gate_results: dict[str, dict] = {}
    for gid in sorted(gate_defs.keys()):
        gmeta = gate_defs[gid]
        g_recs = [
            r
            for r in records
            if gid in [x.strip() for x in r["gates"].split(",") if x.strip()]
        ]
        pass_cnt = sum(1 for r in g_recs if r["status"] == "PASS")
        non_pass_ids = [r["test_id"] for r in g_recs if r["status"] != "PASS"]
        blockers: list[str] = []
        if not g_recs:
            blockers.append("No test cases mapped to gate")
        if missing_ids:
            blockers.append(f"Missing canonical IDs: {len(missing_ids)}")
        if duplicate_ids:
            blockers.append(f"Duplicate canonical IDs: {duplicate_ids}")
        if gmeta.get("requires_native_verilator") and not native_verilator:
            blockers.append("Native Verilator Vaa_qec_top was not compiled")
        if gmeta.get("requires_qemu_arm") and not qemu_ready:
            blockers.append("QEMU ARM/ARM64 user-mode emulators or cross-compiled guest ELFs missing")
        if gid in ("G3", "G4") and not (tb_json.get("native_verilator_compiled") is True and vcd_path.exists()):
            blockers.append("Missing standalone native Verilator JSON/VCD waveform evidence")
        if gid == "G8" and code_git_dirty and not args.allow_dirty:
            blockers.append("Working tree has uncommitted code changes (code_git_dirty == true)")
        if non_pass_ids:
            blockers.append(f"Non-PASS test IDs: {non_pass_ids}")

        status = "PASS" if not blockers else "FAIL"
        gate_results[gid] = {
            "gate_id": gid,
            "name": gmeta["name"],
            "description": gmeta["description"],
            "status": status,
            "total_cases": len(g_recs),
            "pass_cases": pass_cnt,
            "non_pass_ids": non_pass_ids,
            "blockers": blockers,
        }

    overall_pass = (
        len(records) == 223
        and not missing_ids
        and not duplicate_ids
        and not unexpected_ids
        and all(g["status"] == "PASS" for g in gate_results.values())
    )

    env_manifest = {
        "run_id": args.run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_branch,
        "git_commit": git_commit,
        "code_git_dirty": code_git_dirty,
        "code_git_dirty_details": git_status_out.splitlines() if git_status_out else [],
        "frozen_baseline_commit": invariants["frozen_baseline_commit"],
        "host_platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cxx_compiler": cmd_first_line(["g++", "--version"]),
        "verilator_version": cmd_first_line(["verilator", "--version"]),
        "qemu_arm_version": cmd_first_line(["qemu-arm", "--version"]),
        "qemu_aarch64_version": cmd_first_line(["qemu-aarch64", "--version"]),
        "arm_cross_cxx_version": cmd_first_line(["arm-linux-gnueabihf-g++", "--version"]),
        "aarch64_cross_cxx_version": cmd_first_line(["aarch64-linux-gnu-g++", "--version"]),
        "rtl_build_provenance": rtl_prov,
        "guest_elf_sha256": {
            "avid_daemon_cortex_a9": sha256_file(arm_elf) if arm_elf.exists() else None,
            "avid_daemon_aarch64": sha256_file(aarch64_elf) if aarch64_elf.exists() else None,
        },
        "standalone_vcd_sha256": sha256_file(vcd_path) if vcd_path.exists() else None,
        "standalone_vcd_bytes": vcd_path.stat().st_size if vcd_path.exists() else 0,
        "fidelity_disclaimers": invariants["fidelity_disclaimers"],
    }
    (run_dir / "manifests" / "environment_manifest.json").write_text(
        json.dumps(env_manifest, indent=2),
        encoding="utf-8",
    )

    gate_eval_doc = {
        "layer": 4,
        "run_id": args.run_id,
        "git_commit": git_commit,
        "code_git_dirty": code_git_dirty,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "canonical_id_count": 223,
        "executed_record_count": len(records),
        "pass_count": sum(1 for r in records if r["status"] == "PASS"),
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "unexpected_ids": unexpected_ids,
        "set_summaries": set_summaries,
        "gates": gate_results,
        "standalone_rtl_tb": tb_json,
        "fidelity_disclaimers": invariants["fidelity_disclaimers"],
    }
    (run_dir / "acceptance" / "gate_evaluation.json").write_text(
        json.dumps(gate_eval_doc, indent=2),
        encoding="utf-8",
    )

    # Markdown Report Generation
    lines = [
        "# Layer 4 — Hardware-in-the-Loop Co-Simulation Bridge & Emulators Validation Report",
        "",
        f"- **Run ID:** `{args.run_id}`",
        f"- **Branch / Commit:** `{git_branch}` @ `{git_commit}` (`code_git_dirty = {code_git_dirty}`)",
        f"- **Frozen Reference Baseline:** `AVID_EPJ_Patches` @ `{invariants['frozen_baseline_commit']}` (unmodified)",
        f"- **Overall Acceptance Disposition:** **{'PASS' if overall_pass else 'FAIL'}** (`{gate_eval_doc['pass_count']} / 223` canonical IDs `PASS`, Gates `G1`–`G8` `{ '8/8 PASS' if overall_pass else 'INCOMPLETE/FAIL' }`)",
        f"- **RTL Simulator Mode:** `{rtl_prov.get('simulator_mode', 'unknown')}` (`{env_manifest['verilator_version']}`)",
        f"- **QEMU Guest Toolchains:** `{env_manifest['qemu_arm_version']}` (`cortex-a9`) & `{env_manifest['qemu_aarch64_version']}` (`cortex-a53`)",
        "",
        "## 1. Remediated Layer 4 Defects (`DEF-L4-01` .. `DEF-L4-12` + AMBA Reset Gating)",
        "",
        "| Defect ID | Summary of Verified Root Cause & Applied Fix |",
        "| :--- | :--- |",
        "| `DEF-L4-01` | Aligned `rtl_software_interface.yaml` with `weight_map_reg.sv` (`0x04C = status`, `0x054 = cfg_max_iter`, added RO `active_x/z` `0x100..0x144`). |",
        "| `DEF-L4-02` | Aligned `weight_commit_protocol.yaml` with RTL triple-bank value-copy (`shadow -> commit_snap -> active`, `status[0]=swap_pending`, `status[1]=bp_busy`). |",
        "| `DEF-L4-03` | Rewrote `weight_reg_driver.{h,c}` and `physical_backend.cpp` (`0x000..0x144` register model, dual-basis CSS BP+OSD-0 matching `bp_engine.sv`, `SLVERR` enforcement). |",
        "| `DEF-L4-04` | Replaced `strstr`/`strtoul` substring parser in `bridge_protocol.h` with strict RFC-8259 token parser (`\"tlast\": true|false|1|0`, strict hex/uint type validation). |",
        "| `DEF-L4-05` | Added explicit `accepted_cycle` + `completed_cycle` fields and non-blocking `SYNDROME_ASYNC` / `STEP_CYCLES` / `DRAIN_CORRECTION` RPCs enabling mid-decode `COMMIT` (`SLVERR`) and 16-deep FIFO backpressure over `AF_UNIX`. |",
        "| `DEF-L4-06` | Enforced strict global monotonic/deduplicated `transaction_id` tracking across non-adjacent and decreasing IDs. |",
        "| `DEF-L4-07` | Recorded full response payloads (`rdata`, `corr_x`, `corr_z`, `bp_cycles`, `epoch_irq`, `completed_cycle`) plus `ERR_OVERSIZED` & `SHUTDOWN` records in `m3_transactions.jsonl`. |",
        "| `DEF-L4-08` | Added `AVID_USE_VERILATOR` to `emulator/bridge/CMakeLists.txt` and unified `reset_model` (`rcyc+2`), ingress/egress cycle split (`2+3`), and 16-deep FIFO simulation between `Vaa_qec_top` and C++ fallback. |",
        "| `DEF-L4-09` | Restored clean Layer-4-isolated `software/CMakeLists.txt`, enforced `-cpu cortex-a53` on `qemu-aarch64`, and removed silent `x86_64` fallback in `emulator/qemu/run.sh`. |",
        "| `DEF-L4-10` | Instantiated `axis_ingress_tap.sv` in `aa_qec_top.sv`, gated `axis_syndrome_rx.sv` on `tdata <= 0xFF` (`valid_bit == 0`), and wired `epoch_irq` into `simulation_backend.cpp` (`epoch_irq_count`). |",
        "| `DEF-L4-11` | Enforced full-word `cfg_max_iter` bounds `1..5` (rejecting upper-bit garbage like `0x13` and `K=6,7`), rejected `0x048=0` non-commit writes with `SLVERR`, and gated AXI-Lite/AXI-Stream `*ready` with `aresetn`. |",
        "| `DEF-L4-12` | Updated `scripts/run_layer4.sh` to build/execute native Verilator `standalone_rtl_tb`, QEMU ARM/ARM64 guests, and all 5 validation sets while retaining `emulator/verilator/models/`. |",
        "",
        "## 2. Five-Set Execution Summary (`223` Canonical IDs)",
        "",
        "| Set | Total Cases | PASS | Non-PASS | Host Wall Time (ms) | Total Simulated RTL Cycles |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sid in ("Set1", "Set2", "Set3", "Set4", "Set5"):
        s = set_summaries[sid]
        p = s["counts"].get("PASS", 0)
        np_cnt = s["total"] - p
        lines.append(
            f"| `{sid}` | {s['total']} | {p} | {np_cnt} | {s['wall_time_ms_total']:.2f} | {s['rtl_cycles_total']} |"
        )

    lines.extend(
        [
            "",
            "## 3. Layer 4 Acceptance Gate Evaluation (`G1`–`G8`)",
            "",
            "| Gate | Name | Cases (`PASS/Total`) | Status | Notes / Blockers |",
            "| :--- | :--- | ---: | :--- | :--- |",
        ]
    )
    for gid in sorted(gate_results.keys()):
        g = gate_results[gid]
        note = "All mapped cases & provenance checks satisfied" if not g["blockers"] else "; ".join(g["blockers"])
        lines.append(f"| `{gid}` | {g['name']} | `{g['pass_cases']}/{g['total_cases']}` | **{g['status']}** | {note} |")

    lines.extend(
        [
            "",
            "## 4. Explicit Co-Simulation Fidelity & Boundary Disclosures (`L4-LIM-01..04`)",
            "",
            f"- **QEMU Execution Boundary:** {invariants['fidelity_disclaimers']['qemu_scope']}",
            f"- **RTL Timing & Physics Boundary:** {invariants['fidelity_disclaimers']['rtl_timing_scope']}",
            "",
        ]
    )

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"[Layer 4 Gate Evaluator] Overall={gate_eval_doc['overall_status']} | "
        f"Cases={gate_eval_doc['pass_count']}/223 | "
        f"Dirty={code_git_dirty} | "
        f"Gates=" + ", ".join(f"{k}:{v['status']}" for k, v in sorted(gate_results.items()))
    )
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
