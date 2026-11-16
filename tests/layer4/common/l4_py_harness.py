"""Audit-grade Python Layer 4 test harness with JSONL output & bridge process management."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Callable, Iterator


class L4PySuite:
    def __init__(self, suite_name: str, set_id: str, out_jsonl: str | None = None, repo_root: Path | None = None) -> None:
        self.suite_name = suite_name
        self.set_id = set_id
        self.out_jsonl = Path(out_jsonl) if out_jsonl else None
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.records: list[dict] = []

    @property
    def bridge_bin(self) -> Path:
        env_b = os.environ.get("AVID_L4_BRIDGE_BIN")
        if env_b:
            return Path(env_b)
        return self.repo_root / "emulator" / "verilator" / "models" / "avid_cosim_bridge"

    def run_case(
        self,
        test_id: str,
        gates: str,
        title: str,
        simulator_mode: str,
        fn: Callable[[], tuple[bool, int, str]],
    ) -> None:
        t0 = time.perf_counter()
        rtl_cyc = 0
        try:
            ok, rtl_cyc, ev = fn()
            status = "PASS" if ok else "FAIL"
        except Exception as exc:  # noqa: BLE001
            status = "FAIL"
            ev = f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000.0
        rec = {
            "test_id": test_id,
            "suite": self.suite_name,
            "set_id": self.set_id,
            "gates": gates,
            "title": title,
            "status": status,
            "wall_time_ms": round(ms, 4),
            "rtl_cycles": int(rtl_cyc),
            "simulator_mode": simulator_mode,
            "evidence": ev,
        }
        self.records.append(rec)
        print(f"[{self.suite_name}] {test_id:<14} {status:<6} ({ms:.2f} ms, rtl_cyc={rtl_cyc}) {ev}", flush=True)

    def finalize(self) -> int:
        if self.out_jsonl is not None:
            self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.out_jsonl.open("a", encoding="utf-8") as fp:
                for rec in self.records:
                    fp.write(json.dumps(rec) + "\n")
        return 0 if all(r["status"] == "PASS" for r in self.records) else 1


@contextmanager
def running_bridge(
    bridge_bin: Path,
    work_dir: Path,
    tag: str,
    max_clients: int = 16,
    require_native_rtl: bool = True,
) -> Iterator[tuple[Path, Path, Path, subprocess.Popen]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    # Keep AF_UNIX sun_path < 108 bytes by putting socket in /tmp
    sock_path = Path(f"/tmp/avid_l4py_{os.getpid()}_{tag}.sock")
    tx_log = work_dir / f"{tag}_m3_transactions.jsonl"
    metrics_out = work_dir / f"{tag}_bridge_metrics.json"
    for p in (sock_path, tx_log, metrics_out):
        if p.exists():
            p.unlink()

    cmd = [
        str(bridge_bin),
        "--socket",
        str(sock_path),
        "--tx-log",
        str(tx_log),
        "--metrics-out",
        str(metrics_out),
        "--max-clients",
        str(max_clients),
    ]
    if require_native_rtl:
        cmd.insert(1, "--require-native-rtl")

    proc = subprocess.Popen(cmd, cwd=str(bridge_bin.parents[3]), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        ready = False
        for _ in range(150):
            if sock_path.exists():
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.settimeout(0.5)
                        s.connect(str(sock_path))
                        ready = True
                        break
                except OSError:
                    pass
            if proc.poll() is not None:
                break
            time.sleep(0.015)
        if not ready:
            out, err = proc.communicate(timeout=2)
            raise RuntimeError(f"Bridge failed to start: rc={proc.returncode} err={err.decode(errors='replace')}")
        yield sock_path, tx_log, metrics_out, proc
    finally:
        if proc.poll() is None:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect(str(sock_path))
                    s.sendall(b'{"protocol_version":1,"transaction_id":99999999,"operation":"SHUTDOWN"}\n')
                    s.recv(512)
            except OSError:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=2)
        if sock_path.exists():
            sock_path.unlink()


def send_rpc(sock: socket.socket, payload: dict) -> dict:
    raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    sock.sendall(raw)
    buf = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            break
        buf.extend(b)
        if b == b"\n":
            break
    return json.loads(buf.decode("utf-8"))
