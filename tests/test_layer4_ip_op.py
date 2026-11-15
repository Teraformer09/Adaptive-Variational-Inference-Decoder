"""Layer 4 IP/OP Validation Suite: HAL <-> Co-Simulation Bridge <-> RTL Pin Driver."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import time


def test_layer4_bridge_ip_op_protocol(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "emulator" / "verilator" / "models"
    bridge_bin = model_dir / "avid_cosim_bridge"
    m3_bin = model_dir / "test_bridge_transactions"
    assert bridge_bin.exists() and m3_bin.exists()

    sock_path = tmp_path / "layer4_test.sock"
    trace_path = tmp_path / "layer4_trace.jsonl"
    proc = subprocess.Popen(
        [str(bridge_bin), "--socket", str(sock_path), "--tx-log", str(trace_path), "--max-clients", "5"],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(100):
            if sock_path.exists():
                break
            time.sleep(0.02)
        assert sock_path.exists()

        # Run full C++ M3 HAL <-> Bridge transaction suite
        res = subprocess.run([str(m3_bin), str(sock_path)], cwd=str(repo_root), capture_output=True, text=True, check=True)
        assert res.returncode == 0

        # Direct raw JSON protocol IP -> OP check (SYNDROME_TRANSFER for q4 Z error sx=0x6 -> corr_z=0x010)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(sock_path))
            req = {"protocol_version": 1, "transaction_id": 99001, "operation": "SYNDROME_TRANSFER", "address": "0x00000000", "data": "0x0006", "tlast": 1, "byte_enable": 3}
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            resp = json.loads(s.makefile().readline())
            assert resp["status"] == "OK"
            assert resp["transaction_id"] == 99001
            assert int(resp["bp_cycles"]) in (17, 22, 27, 12, 7)
            assert int(resp["corr_z"], 16) == 0x0010
    finally:
        proc.terminate()
        proc.wait(timeout=5)
