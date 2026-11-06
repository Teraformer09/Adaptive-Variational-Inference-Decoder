#!/usr/bin/env python3
"""
AVID V1 Experiment Controller, Schema Validator, Co-Simulation Orchestrator & Evidence Collector
Implements Milestones M0–M5:
  1. Validates experiment YAML strictly against `configs/schema.yaml` (rejecting unknown keys).
  2. Records Git commit, branch, and pinned toolchain versions (QEMU, Verilator, GCC, CMake, Python).
  3. Creates isolated run directory `runs/<run_id>/{qemu,verilator,bridge,metrics,waveforms}`.
  4. Executes Milestone M1 standalone Verilator RTL verification (K=1..5 cycle matrix & 68.0 ns @ K=3).
  5. Executes Milestone M3 HAL <-> C++ Bridge <-> Verilator RTL transaction test.
  6. Executes Milestone M4/M5 full dual-loop QEMU ARM64 (`avid_daemon`) + C++ Bridge + Verilator RTL
     across all configured scenarios with deterministic simulation clock and timeout supervision.
  7. Writes `runs/<run_id>/manifest.json` (with SHA-256 artifact checksums) and `runs/<run_id>/report.json`.
"""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def get_cmd_version(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        first_line = out.strip().splitlines()[0] if out.strip() else "unknown"
        return first_line
    except Exception:
        return "not_installed"


def validate_config_strict(cfg: Dict[str, Any], schema_path: Path) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    allowed_sections: Dict[str, Any] = schema["allowed_sections"]

    unknown_top = set(cfg.keys()) - set(allowed_sections.keys())
    if unknown_top:
        raise ValueError(f"Unknown top-level configuration section(s): {sorted(unknown_top)}")

    for sec_name, sec_rules in allowed_sections.items():
        if sec_name not in cfg:
            raise ValueError(f"Missing required top-level section: '{sec_name}'")
        val = cfg[sec_name]
        if sec_name == "scenarios":
            if not isinstance(val, list) or len(val) == 0:
                raise ValueError("'scenarios' must be a non-empty list.")
            allowed_scen = set(sec_rules["allowed_values"])
            bad_scen = [s for s in val if s not in allowed_scen]
            if bad_scen:
                raise ValueError(f"Unknown scenario(s): {bad_scen}. Allowed: {sorted(allowed_scen)}")
        else:
            if not isinstance(val, dict):
                raise ValueError(f"Section '{sec_name}' must be a mapping.")
            req_keys = set(sec_rules.get("required", []))
            opt_keys = set(sec_rules.get("optional", []))
            missing_k = req_keys - set(val.keys())
            if missing_k:
                raise ValueError(f"Section '{sec_name}' missing required key(s): {sorted(missing_k)}")
            unknown_k = set(val.keys()) - (req_keys | opt_keys)
            if unknown_k:
                raise ValueError(f"Section '{sec_name}' contains unknown key(s): {sorted(unknown_k)}")


def run_experiment(config_path: Path) -> int:
    t_start_wall = time.perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()

    schema_path = REPO_ROOT / "configs" / "schema.yaml"
    if not config_path.exists():
        print(f"[ERROR] Configuration file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_config_strict(cfg, schema_path)
    except Exception as exc:
        print(f"[SCHEMA VALIDATION FAILED] {exc}", file=sys.stderr)
        return 2

    exp_name = str(cfg["experiment"]["name"])
    seed = int(cfg["experiment"]["seed"])
    timeout_sec = int(cfg["experiment"]["timeout_seconds"])
    rounds_per_scen = int(cfg["experiment"].get("rounds_per_scenario", 1024))
    window_rounds = int(cfg["experiment"].get("window_rounds", 256))
    clock_period_ns = int(cfg["rtl"]["clock_period_ns"])
    reset_cycles = int(cfg["rtl"]["reset_cycles"])
    save_vcd = bool(cfg["rtl"]["waveform"]) and bool(cfg["artifacts"]["save_waveforms"])
    require_native_rtl = (
        bool(cfg["rtl"].get("require_native_verilator", False))
        or bool(cfg["validation"].get("require_native_verilator", False))
        or os.environ.get("AVID_REQUIRE_NATIVE_VERILATOR", "0") in ("1", "true")
    )
    controller_mode = str(cfg["software"].get("controller_mode", "avid_r2d2"))
    osd_mode = int(cfg["software"].get("osd_mode", 1))
    stim_bin = str(cfg["software"].get("stim_bin", ""))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{exp_name}_{stamp}_s{seed}"
    out_root = REPO_ROOT / str(cfg["artifacts"]["output_dir"])
    run_dir = out_root / run_id

    qemu_dir = run_dir / "qemu"
    verilator_dir = run_dir / "verilator"
    bridge_dir = run_dir / "bridge"
    metrics_dir = run_dir / "metrics"
    waveforms_dir = run_dir / "waveforms"

    for d in (qemu_dir, verilator_dir, bridge_dir, metrics_dir, waveforms_dir):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, run_dir / "config.yaml")

    print("=================================================================")
    print(f"  AVID V1 Dual-Emulator Experiment Runner: {run_id}")
    print("=================================================================")

    # Step 1: Build or verify emulators
    build_log_path = run_dir / "build.log"
    with build_log_path.open("w", encoding="utf-8") as blog:
        rc_build = subprocess.call(
            ["bash", str(REPO_ROOT / "scripts" / "build_all.sh")],
            cwd=str(REPO_ROOT),
            stdout=blog,
            stderr=subprocess.STDOUT,
        )
    if rc_build != 0:
        print(f"[FAIL] Build failed (see {build_log_path})", file=sys.stderr)
        return 1

    # Step 2: Milestone M1 — Verilator Standalone RTL & K=1..5 Cycle Matrix Verification
    m1_vcd = str(waveforms_dir / "m1_standalone_rtl.vcd") if save_vcd else ""
    m1_csv = verilator_dir / "rtl_cycle_matrix.csv"
    m1_json = verilator_dir / "m1_verilator_standalone.json"
    m1_log = verilator_dir / "m1_standalone.log"

    cmd_m1 = [
        "bash",
        str(REPO_ROOT / "emulator" / "verilator" / "run.sh"),
        "--standalone",
        "--csv",
        str(m1_csv),
        "--json",
        str(m1_json),
    ]
    if m1_vcd:
        cmd_m1.extend(["--vcd", m1_vcd])
    if require_native_rtl:
        cmd_m1.append("--require-native-rtl")

    with m1_log.open("w", encoding="utf-8") as flog:
        rc_m1 = subprocess.call(cmd_m1, cwd=str(REPO_ROOT), stdout=flog, stderr=subprocess.STDOUT)
    if rc_m1 != 0:
        print(f"[FAIL] Milestone M1 Verilator standalone verification failed (see {m1_log})", file=sys.stderr)
        return 1
    m1_data = json.loads(m1_json.read_text(encoding="utf-8"))
    print(
        f"[PASS M1] Verilator RTL verified ({m1_data.get('simulator')}): "
        f"K=3 core={m1_data['k3_bp_core_cycles']} cyc ({m1_data['k3_bp_core_latency_ns']} ns), "
        f"AXI total={m1_data.get('k3_axis_total_cycles', 22)} cyc ({m1_data.get('k3_axis_total_latency_ns', 88.0)} ns) @ 250 MHz"
    )

    # Step 3: Milestone M3 — HAL <-> Co-Sim Bridge <-> Verilator RTL Transaction Test
    m3_sock = f"/tmp/avid_m3_{os.getpid()}.sock"
    m3_tx_log = bridge_dir / "m3_transactions.jsonl"
    m3_metrics = bridge_dir / "m3_bridge_metrics.json"
    m3_log = bridge_dir / "m3_test.log"

    bridge_proc_m3 = subprocess.Popen(
        [
            "bash",
            str(REPO_ROOT / "emulator" / "verilator" / "run.sh"),
            "--bridge",
            "--socket",
            m3_sock,
            "--tx-log",
            str(m3_tx_log),
            "--metrics-out",
            str(m3_metrics),
            "--clock-period-ns",
            str(clock_period_ns),
            "--reset-cycles",
            str(reset_cycles),
            "--max-clients",
            "2",
            "--allow-negative-tests",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with m3_log.open("w", encoding="utf-8") as flog:
            rc_m3 = subprocess.call(
                [str(REPO_ROOT / "emulator" / "verilator" / "models" / "test_bridge_transactions"), m3_sock],
                cwd=str(REPO_ROOT),
                stdout=flog,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
            )
        bridge_proc_m3.wait(timeout=10)
    finally:
        if bridge_proc_m3.poll() is None:
            bridge_proc_m3.kill()

    if rc_m3 != 0:
        print(f"[FAIL] Milestone M3 Bridge transaction verification failed (see {m3_log})", file=sys.stderr)
        return 1
    print("[PASS M3] Co-Simulation Bridge <-> Verilator RTL register & syndrome transactions verified")

    # Step 4: Milestone M4 & M5 — Integrated QEMU ARM64 (avid_daemon) + Bridge + Verilator Scenarios
    scenario_results: List[Dict[str, Any]] = []
    overall_pass = True

    for idx, scen in enumerate(cfg["scenarios"]):
        scen_seed = seed + idx * 17
        sock_path = f"/tmp/avid_v1_{os.getpid()}_{idx}.sock"
        tx_log = bridge_dir / f"{scen}_transactions.jsonl"
        br_metrics = bridge_dir / f"{scen}_bridge_metrics.json"
        sw_metrics = metrics_dir / f"{scen}_qemu_metrics.json"
        q_log = qemu_dir / f"{scen}_qemu.log"
        b_log = verilator_dir / f"{scen}_verilator_bridge.log"
        vcd_file = str(waveforms_dir / f"{scen}.vcd") if save_vcd else ""

        if scen == "invalid_configuration":
            # Invalid configuration scenario tests safe daemon rejection (non-zero return expected from daemon)
            with q_log.open("w", encoding="utf-8") as qf:
                rc_inv = subprocess.call(
                    [
                        "bash",
                        str(REPO_ROOT / "emulator" / "qemu" / "run.sh"),
                        "--backend",
                        "sim",
                        "--scenario",
                        "invalid_configuration",
                    ],
                    cwd=str(REPO_ROOT),
                    stdout=qf,
                    stderr=subprocess.STDOUT,
                )
            passed_inv = (rc_inv != 0)
            scenario_results.append({
                "scenario": scen,
                "seed": scen_seed,
                "status": "PASS" if passed_inv else "FAIL",
                "safe_rejection_exit_code": rc_inv,
            })
            if not passed_inv:
                overall_pass = False
            continue

        bridge_cmd = [
            "bash",
            str(REPO_ROOT / "emulator" / "verilator" / "run.sh"),
            "--bridge",
            "--socket",
            sock_path,
            "--tx-log",
            str(tx_log),
            "--metrics-out",
            str(br_metrics),
            "--clock-period-ns",
            str(clock_period_ns),
            "--reset-cycles",
            str(reset_cycles),
            "--max-clients",
            "1",
        ]
        if vcd_file:
            bridge_cmd.extend(["--vcd", vcd_file])
        if require_native_rtl:
            bridge_cmd.append("--require-native-rtl")

        bf = b_log.open("w", encoding="utf-8")
        qf = q_log.open("w", encoding="utf-8")
        bridge_proc = subprocess.Popen(bridge_cmd, cwd=str(REPO_ROOT), stdout=bf, stderr=subprocess.STDOUT)
        qemu_proc = None
        scen_ok = False
        failure_reason = ""

        try:
            qemu_cmd = [
                "bash",
                str(REPO_ROOT / "emulator" / "qemu" / "run.sh"),
                "--backend",
                "sim",
                "--socket",
                sock_path,
                "--scenario",
                str(scen),
                "--seed",
                str(scen_seed),
                "--rounds",
                str(rounds_per_scen),
                "--window",
                str(window_rounds),
                "--controller-mode",
                controller_mode,
                "--osd-mode",
                str(osd_mode),
                "--metrics-out",
                str(sw_metrics),
            ]
            if stim_bin:
                qemu_cmd.extend(["--stim-bin", stim_bin])
            qemu_proc = subprocess.Popen(qemu_cmd, cwd=str(REPO_ROOT), stdout=qf, stderr=subprocess.STDOUT)
            rc_q = qemu_proc.wait(timeout=timeout_sec)
            rc_b = bridge_proc.wait(timeout=15)
            scen_ok = (rc_q == 0 and rc_b == 0 and sw_metrics.exists() and br_metrics.exists())
            if not scen_ok:
                failure_reason = f"rc_qemu={rc_q} rc_bridge={rc_b}"
        except subprocess.TimeoutExpired:
            failure_reason = f"timeout_exceeded_{timeout_sec}s"
            scen_ok = False
        finally:
            if qemu_proc and qemu_proc.poll() is None:
                qemu_proc.kill()
            if bridge_proc.poll() is None:
                bridge_proc.kill()
            bf.close()
            qf.close()
            if Path(sock_path).exists():
                Path(sock_path).unlink(missing_ok=True)

        sw_data = json.loads(sw_metrics.read_text(encoding="utf-8")) if sw_metrics.exists() else {}
        br_data = json.loads(br_metrics.read_text(encoding="utf-8")) if br_metrics.exists() else {}

        rec = {
            "scenario": scen,
            "seed": scen_seed,
            "status": "PASS" if scen_ok else "FAIL",
            "failure_reason": failure_reason,
            "qemu_software_metrics": sw_data,
            "bridge_rtl_metrics": br_data,
        }
        scenario_results.append(rec)
        if scen_ok:
            print(
                f"[PASS M4] Scenario '{scen}': rounds={sw_data.get('total_rounds')} "
                f"epochs={sw_data.get('epochs_completed')} commits={sw_data.get('weight_bank_commits')} "
                f"(reconciled={sw_data.get('epoch_commit_reconciled')}) "
                f"core_cyc={sw_data.get('bp_core_cycles_mean')} axis_cyc={sw_data.get('axis_total_cycles_mean')} "
                f"static_ler={sw_data.get('static_b0_logical_error_rate')} "
                f"avid_ler={sw_data.get('logical_error_rate')} "
                f"(-{sw_data.get('ler_reduction_vs_static_pct')}%) "
                f"orbit_rmse={sw_data.get('orbit_9class_rmse')}"
            )
        else:
            print(f"[FAIL M4] Scenario '{scen}' failed: {failure_reason}", file=sys.stderr)
            overall_pass = False

    end_iso = datetime.now(timezone.utc).isoformat()
    wall_elapsed_sec = round(time.perf_counter() - t_start_wall, 3)

    git_sha = get_cmd_version(["git", "rev-parse", "HEAD"])
    git_branch = get_cmd_version(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    qemu_ver = get_cmd_version(["qemu-aarch64", "--version"])
    verilator_ver = get_cmd_version(["verilator", "--version"])
    gcc_ver = get_cmd_version(["g++", "--version"])
    arm_gcc_ver = get_cmd_version(["aarch64-linux-gnu-g++", "--version"])
    cmake_ver = get_cmd_version(["cmake", "--version"])

    # Collect SHA-256 checksums of all generated run files
    artifact_checksums: Dict[str, str] = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json", "report.json"):
            rel = str(p.relative_to(run_dir))
            artifact_checksums[rel] = sha256_file(p)

    rtl_prov_path = REPO_ROOT / "emulator" / "verilator" / "models" / "rtl_build_provenance.json"
    rtl_prov = json.loads(rtl_prov_path.read_text(encoding="utf-8")) if rtl_prov_path.exists() else {}

    manifest = {
        "run_id": run_id,
        "experiment_name": exp_name,
        "status": "PASS" if overall_pass else "FAIL",
        "git_commit_sha": git_sha,
        "git_branch": git_branch,
        "config_file": str(config_path.relative_to(REPO_ROOT)) if config_path.is_relative_to(REPO_ROOT) else str(config_path),
        "config_sha256": sha256_file(run_dir / "config.yaml"),
        "toolchain_versions": {
            "python": sys.version.split()[0],
            "qemu_arm_cortex_a9": get_cmd_version(["qemu-arm", "--version"]),
            "qemu_aarch64": qemu_ver,
            "verilator": verilator_ver,
            "host_gxx": gcc_ver,
            "arm_cortex_a9_gxx": get_cmd_version(["arm-linux-gnueabihf-g++", "--version"]),
            "aarch64_linux_gnu_gxx": arm_gcc_ver,
            "cmake": cmake_ver,
        },
        "cortex_a9_cross_binary_built": (REPO_ROOT / "emulator" / "qemu" / "images" / "avid_daemon_cortex_a9").exists(),
        "arm64_cross_binary_built": (REPO_ROOT / "emulator" / "qemu" / "images" / "avid_daemon_aarch64").exists(),
        "native_verilator_compiled": bool(m1_data.get("native_verilator_compiled", False)),
        "rtl_build_provenance": rtl_prov,
        "seed": seed,
        "start_timestamp_utc": start_iso,
        "end_timestamp_utc": end_iso,
        "host_wall_clock_seconds": wall_elapsed_sec,
        "artifact_checksums_sha256": artifact_checksums,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = {
        "run_id": run_id,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "host_wall_clock_seconds": wall_elapsed_sec,
        "m1_verilator_standalone_rtl": m1_data,
        "m3_bridge_transaction_test": json.loads(m3_metrics.read_text(encoding="utf-8")),
        "m4_scenarios": scenario_results,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "latest_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (reports_dir / "latest_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("=================================================================")
    print(f"  [RUN COMPLETE] Status={report['overall_status']} | Run Dir: runs/{run_id}")
    print("=================================================================")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: collect_artifacts.py <config.yaml>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run_experiment(Path(sys.argv[1]).resolve()))
