"""Audit-grade Python Layer 5 test harness with structured JSONL output."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Callable


class L5PySuite:
    def __init__(self, suite_name: str, set_id: str, out_jsonl: str | None = None, repo_root: Path | None = None) -> None:
        self.suite_name = suite_name
        self.set_id = set_id
        self.out_jsonl = Path(out_jsonl) if out_jsonl else None
        env_root = os.environ.get("AVID_L5_REPO_ROOT")
        self.repo_root = repo_root or (Path(env_root) if env_root else Path(__file__).resolve().parents[3])
        self.records: list[dict] = []

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
            ok, rtl_cyc, det = fn()
            status = "PASS" if ok else "FAIL"
        except Exception as exc:  # noqa: BLE001
            status = "FAIL"
            det = f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000.0
        gate_ids = [g.strip() for g in gates.split(",") if g.strip()]
        rec = {
            "test_id": test_id,
            "suite": self.suite_name,
            "set_id": self.set_id,
            "gate_ids": gate_ids,
            "gates": gates,
            "title": title,
            "status": status,
            "rtl_cycles": int(rtl_cyc),
            "wall_time_ms": round(ms, 4),
            "simulator_mode": simulator_mode,
            "details": det,
            "evidence": det,
        }
        self.records.append(rec)
        print(f"[{self.suite_name}] {test_id:<14} {status:<6} ({ms:.2f} ms, rtl_cyc={rtl_cyc}) {det}", flush=True)

    def finalize(self) -> int:
        if self.out_jsonl is not None:
            self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.out_jsonl.open("a", encoding="utf-8") as fp:
                for rec in self.records:
                    fp.write(json.dumps(rec) + "\n")
        return 0 if all(r["status"] == "PASS" for r in self.records) else 1
