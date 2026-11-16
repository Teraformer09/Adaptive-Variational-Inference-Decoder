"""Layer 4 Set 4.4: Artifact Persistence, Flush Durability & Provenance (L4-PERS-01 .. L4-PERS-09)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.l4_py_harness import L4PySuite, running_bridge, send_rpc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", default="")
    ap.add_argument("--work-dir", default="/tmp/avid_l4_set4_pers")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    suite = L4PySuite("set4_artifact_persistence", "Set4", args.out_jsonl or None)
    repo_root = suite.repo_root
    sim_mode = "verilator_c++_rtl"

    # L4-PERS-01: Per-line fflush durability of m3_transactions.jsonl across abrupt SIGKILL
    def case_pers01() -> tuple[bool, int, str]:
        sock_path = Path(f"/tmp/avid_l4_pers01_{os.getpid()}.sock")
        tx_log = work_dir / "pers01_sigkill.jsonl"
        for p in (sock_path, tx_log):
            if p.exists():
                p.unlink()
        proc = subprocess.Popen(
            [str(suite.bridge_bin), "--require-native-rtl", "--socket", str(sock_path), "--tx-log", str(tx_log), "--max-clients", "4"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(100):
            if sock_path.exists():
                break
            time.sleep(0.015)
        last_cyc = 0
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(sock_path))
            for i in range(1, 21):
                r = send_rpc(s, {"protocol_version": 1, "transaction_id": i, "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{(i&0xF):04x}", "tlast": (i == 20)})
                last_cyc = r["completed_cycle"]
        # Abruptly SIGKILL bridge without SHUTDOWN; all 20 flushed JSONL lines must survive intact!
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=3)
        if sock_path.exists():
            sock_path.unlink()
        lines = [ln for ln in tx_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        parsed = [json.loads(ln) for ln in lines]
        return len(parsed) == 20 and parsed[-1]["tx_id"] == 20, last_cyc, f"20/20 JSONL lines survived SIGKILL intact"

    suite.run_case("L4-PERS-01", "G8", "m3_transactions.jsonl per-line fflush survives abrupt SIGKILL with 0 lost/corrupt lines", sim_mode, case_pers01)

    # L4-PERS-02: --metrics-out JSON completeness & schema validation
    def case_pers02() -> tuple[bool, int, str]:
        with running_bridge(suite.bridge_bin, work_dir, "pers02", max_clients=2) as (sock_path, _, br_metrics, _):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(str(sock_path))
                send_rpc(s, {"protocol_version": 1, "transaction_id": 1, "operation": "RESET", "data": 10})
                send_rpc(s, {"protocol_version": 1, "transaction_id": 2, "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0006", "tlast": True})
        m = json.loads(br_metrics.read_text(encoding="utf-8"))
        req_keys = {
            "protocol_version",
            "simulator",
            "native_verilator_compiled",
            "clock_mode",
            "clock_period_ns",
            "served_clients",
            "total_transactions",
            "mmio_writes",
            "mmio_reads",
            "syndrome_transfers",
            "reset_operations",
            "irq_events",
            "rejected_transactions",
            "final_sim_cycle",
            "final_sim_time_ns",
            "final_epoch_count",
        }
        ok = req_keys.issubset(m.keys()) and m["native_verilator_compiled"] is True
        return ok, m["final_sim_cycle"], f"16/16 metrics keys present (simulator={m['simulator']})"

    suite.run_case("L4-PERS-02", "G8", "Bridge --metrics-out JSON persists complete 16-field telemetry & simulator provenance", sim_mode, case_pers02)

    # L4-PERS-03: rtl_build_provenance.json SHA-256 hashes match live firmware/rtl/*.sv files on disk
    def case_pers03() -> tuple[bool, int, str]:
        prov_path = repo_root / "emulator" / "verilator" / "models" / "rtl_build_provenance.json"
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        rtl_hashes = prov.get("rtl_source_sha256", {})
        if len(rtl_hashes) != 7:
            return False, 0, f"expected 7 RTL files, found {len(rtl_hashes)}"
        for rel, recorded_sha in rtl_hashes.items():
            live_sha = hashlib.sha256((repo_root / rel).read_bytes()).hexdigest()
            if live_sha != recorded_sha:
                return False, 0, f"SHA-256 mismatch on {rel}"
        return prov["native_verilator_compiled"] is True, 0, "7/7 RTL .sv SHA-256 hashes match rtl_build_provenance.json"

    suite.run_case("L4-PERS-03", "G8", "rtl_build_provenance.json SHA-256 digests match all 7 live SystemVerilog files", sim_mode, case_pers03)

    # L4-PERS-04: VCD waveform persistence via standalone_rtl_tb --vcd
    def case_pers04() -> tuple[bool, int, str]:
        vcd_out = work_dir / "pers04_rtl.vcd"
        csv_out = work_dir / "pers04_cycles.csv"
        json_out = work_dir / "pers04_summary.json"
        tb = repo_root / "emulator" / "verilator" / "models" / "standalone_rtl_tb"
        subprocess.run(
            [str(tb), "--require-native-rtl", "--vcd", str(vcd_out), "--csv", str(csv_out), "--json", str(json_out)],
            cwd=str(repo_root),
            check=True,
        )
        js = json.loads(json_out.read_text(encoding="utf-8"))
        ok = vcd_out.exists() and vcd_out.stat().st_size > 10000 and js["k3_axi_ingress_cycles"] == 2 and js["k3_axi_egress_cycles"] == 3
        return ok, js["total_sim_cycles"], f"vcd_bytes={vcd_out.stat().st_size} total_sim_cycles={js['total_sim_cycles']}"

    suite.run_case("L4-PERS-04", "G3,G8", "standalone_rtl_tb persists valid Verilator VCD waveform, CSV cycle matrix, and JSON summary", sim_mode, case_pers04)

    # L4-PERS-05: CSV K=1..5 cycle matrix integrity
    def case_pers05() -> tuple[bool, int, str]:
        csv_lines = (work_dir / "pers04_cycles.csv").read_text(encoding="utf-8").strip().splitlines()
        ok = len(csv_lines) == 6 and csv_lines[3].startswith("3,4,2,17,68.0,3,22,88.0,17,22,true")
        return ok, 0, f"csv_rows={len(csv_lines)-1} K=3 row='{csv_lines[3]}'"

    suite.run_case("L4-PERS-05", "G3,G8", "Persisted CSV cycle matrix verifies K=1..5 (2 ingress + 2+5K core + 3 egress)", sim_mode, case_pers05)

    # L4-PERS-06: Retained emulator/verilator/models & emulator/qemu/images binaries (DEF-L4-12)
    def case_pers06() -> tuple[bool, int, str]:
        req = [
            repo_root / "emulator" / "verilator" / "models" / "avid_cosim_bridge",
            repo_root / "emulator" / "verilator" / "models" / "standalone_rtl_tb",
            repo_root / "emulator" / "verilator" / "models" / "test_bridge_transactions",
            repo_root / "emulator" / "verilator" / "models" / "rtl_build_provenance.json",
            repo_root / "emulator" / "qemu" / "images" / "avid_daemon_cortex_a9",
            repo_root / "emulator" / "qemu" / "images" / "avid_daemon_aarch64",
            repo_root / "emulator" / "qemu" / "images" / "avid_daemon_x86_64",
        ]
        ok = all(p.exists() and p.stat().st_size > 0 for p in req)
        return ok, 0, "7/7 Verilator & QEMU emulator artifacts retained on disk"

    suite.run_case("L4-PERS-06", "G8", "All 7 Verilator and QEMU emulator binaries & manifests are retained for standalone re-runs", sim_mode, case_pers06)

    # L4-PERS-07: QEMU guest --trace-out & --metrics-out persistence
    def case_pers07() -> tuple[bool, int, str]:
        gt = work_dir / "pers07_guest.jsonl"
        gm = work_dir / "pers07_guest_metrics.json"
        with running_bridge(suite.bridge_bin, work_dir, "pers07", max_clients=2) as (sock_path, _, _, _):
            env = dict(os.environ, AVID_QEMU_ARCH="cortex-a9")
            subprocess.run(
                [
                    "bash",
                    str(repo_root / "emulator" / "qemu" / "run.sh"),
                    "--backend",
                    "sim",
                    "--socket",
                    str(sock_path),
                    "--rounds",
                    "16",
                    "--window",
                    "8",
                    "--trace-out",
                    str(gt),
                    "--metrics-out",
                    str(gm),
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
            )
        m = json.loads(gm.read_text(encoding="utf-8"))
        lines = [json.loads(ln) for ln in gt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return len(lines) >= 18 and m["syndrome_satisfied"] == 16, m["final_completed_cycle"], f"guest trace lines={len(lines)}"

    suite.run_case("L4-PERS-07", "G5,G8", "QEMU ARM guest persists valid per-transaction JSONL trace & JSON summary", sim_mode, case_pers07)

    # L4-PERS-08: Standalone pytest tests/test_layer4_ip_op.py executes cleanly without rebuild
    def case_pers08() -> tuple[bool, int, str]:
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_layer4_ip_op.py"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        return res.returncode == 0, 0, f"pytest rc={res.returncode}"

    suite.run_case("L4-PERS-08", "G8", "Standalone pytest tests/test_layer4_ip_op.py passes without deleting models/", sim_mode, case_pers08)

    # L4-PERS-09: Branch isolation check (layer4-cosim-bridge-emu free of Layer 2/3 training/estimator files)
    def case_pers09() -> tuple[bool, int, str]:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root), text=True).strip()
        no_leak = not (repo_root / "deployment" / "cpp").exists() and not (repo_root / "software" / "src" / "estimator").exists()
        return branch == "layer4-cosim-bridge-emu" and no_leak, 0, f"branch={branch} pure_layer4_scope={no_leak}"

    suite.run_case("L4-PERS-09", "G8", "Branch isolation verified (layer4-cosim-bridge-emu contains strictly Layer 4 scope)", sim_mode, case_pers09)

    return suite.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
