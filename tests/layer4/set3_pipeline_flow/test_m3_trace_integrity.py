"""Layer 4 Set 3.4: m3_transactions.jsonl Audit & Trace Reconstruction (L4-M3-01 .. L4-M3-11)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.l4_py_harness import L4PySuite, running_bridge, send_rpc  # noqa: E402


REQUIRED_KEYS = {
    "tx_id",
    "op",
    "addr",
    "wdata",
    "rdata",
    "byte_enable",
    "tlast",
    "corr_x",
    "corr_z",
    "bp_cycles",
    "axis_total_cycles",
    "epoch_irq",
    "start_cycle",
    "accepted_cycle",
    "completed_cycle",
    "sim_time_ns",
    "status",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", default="")
    ap.add_argument("--work-dir", default="/tmp/avid_l4_set3_m3")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    suite = L4PySuite("set3_m3_trace_integrity", "Set3", args.out_jsonl or None)
    sim_mode = "verilator_c++_rtl"

    client_responses: list[dict] = []

    with running_bridge(suite.bridge_bin, work_dir, "m3_audit", max_clients=4) as (sock_path, tx_log, br_metrics, _):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect(str(sock_path))
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 10, "operation": "RESET", "data": 10}))
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 11, "operation": "MMIO_WRITE", "address": "0x00000000", "data": "0x0177", "byte_enable": 3}))
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 12, "operation": "MMIO_WRITE", "address": "0x00000024", "data": "0x0266", "byte_enable": 3}))
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 13, "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1}))
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 14, "operation": "MMIO_READ", "address": "0x00000100"}))
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 15, "operation": "MMIO_READ", "address": "0x00000050"}))
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 16, "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0096", "tlast": True}))
            # Inject negative / error cases including ERR_OVERSIZED
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 17, "operation": "MMIO_READ", "address": "0x00000990"}))
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 17, "operation": "MMIO_READ", "address": "0x00000050"}))
            s.sendall((("X" * 2500) + "\n").encode("utf-8"))
            over_raw = b""
            while not over_raw.endswith(b"\n"):
                over_raw += s.recv(1)
            client_responses.append(json.loads(over_raw.decode("utf-8")))
            client_responses.append(send_rpc(s, {"protocol_version": 1, "transaction_id": 18, "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0006", "tlast": False}))

    lines = [line for line in tx_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    metrics = json.loads(br_metrics.read_text(encoding="utf-8"))
    final_cyc = metrics["final_sim_cycle"]

    suite.run_case(
        "L4-M3-01",
        "G2,G8",
        "Every line of m3_transactions.jsonl parses as valid JSON with all 17 required fields",
        sim_mode,
        lambda: (all(REQUIRED_KEYS.issubset(r.keys()) for r in records) and len(records) >= 12, final_cyc, f"lines={len(records)} all have 17/17 keys"),
    )

    suite.run_case(
        "L4-M3-02",
        "G2,G8",
        "Bridge total_transactions equals exact m3_transactions.jsonl line count (including ERR_OVERSIZED & SHUTDOWN)",
        sim_mode,
        lambda: (metrics["total_transactions"] == len(records), final_cyc, f"metrics.total={metrics['total_transactions']} jsonl_lines={len(records)}"),
    )

    def case_m3_03() -> tuple[bool, int, str]:
        ok_tids = [r["tx_id"] for r in records if r["status"] == "OK"]
        strictly_inc = all(ok_tids[i] < ok_tids[i + 1] for i in range(len(ok_tids) - 1))
        return strictly_inc and len(ok_tids) == len(set(ok_tids)), final_cyc, f"accepted tx_ids={ok_tids}"

    suite.run_case("L4-M3-03", "G2", "Accepted OK records have unique strictly increasing transaction_ids", sim_mode, case_m3_03)

    suite.run_case(
        "L4-M3-04",
        "G2",
        "1-to-1 pairing between client RPC responses and m3_transactions.jsonl records",
        sim_mode,
        lambda: (
            len(records) == len(client_responses) + 1  # +1 for context-manager SHUTDOWN
            and all(records[i]["status"] == client_responses[i]["status"] for i in range(len(client_responses))),
            final_cyc,
            f"paired {len(client_responses)} client responses + 1 SHUTDOWN",
        ),
    )

    suite.run_case(
        "L4-M3-05",
        "G2,G3",
        "Every record satisfies start_cycle == accepted_cycle <= completed_cycle and monotonic progression",
        sim_mode,
        lambda: (
            all(r["start_cycle"] == r["accepted_cycle"] <= r["completed_cycle"] for r in records)
            and all(records[i]["completed_cycle"] <= records[i + 1]["accepted_cycle"] for i in range(len(records) - 1)),
            final_cyc,
            "monotonic non-overlapping cycle intervals verified",
        ),
    )

    suite.run_case(
        "L4-M3-06",
        "G3,G8",
        "Every record satisfies sim_time_ns == 4 * completed_cycle",
        sim_mode,
        lambda: (all(r["sim_time_ns"] == 4 * r["completed_cycle"] for r in records), final_cyc, "sim_time_ns == 4 * completed_cycle across all records"),
    )

    suite.run_case(
        "L4-M3-07",
        "G2",
        "MMIO_READ records log exact hex rdata matching client response",
        sim_mode,
        lambda: (
            any(r["tx_id"] == 14 and int(r["rdata"], 16) == 0x0177 for r in records)
            and any(r["tx_id"] == 15 and int(r["rdata"], 16) == 1 for r in records),
            final_cyc,
            "tx 14 rdata=0x0177, tx 15 rdata=0x1 verified in JSONL",
        ),
    )

    suite.run_case(
        "L4-M3-08",
        "G2",
        "SYNDROME_TRANSFER records log corr_x, corr_z, bp_cycles, axis_total_cycles, and epoch_irq",
        sim_mode,
        lambda: (
            any(
                r["tx_id"] == 16
                and r["bp_cycles"] == 17
                and r["axis_total_cycles"] == 22
                and r["epoch_irq"] == 1
                and int(r["corr_x"], 16) != 0
                and int(r["corr_z"], 16) != 0
                for r in records
            ),
            final_cyc,
            "tx 16 syndrome response fields verified in JSONL",
        ),
    )

    suite.run_case(
        "L4-M3-09",
        "G2,G7",
        "Rejected SLVERR, ERR_DUPLICATE_TX_ID, and ERR_OVERSIZED transactions are explicitly logged",
        sim_mode,
        lambda: (
            {"ERR_INVALID_REGISTER", "ERR_DUPLICATE_TX_ID", "ERR_OVERSIZED"}.issubset({r["status"] for r in records}),
            final_cyc,
            "all 3 injected error classes present in m3_transactions.jsonl",
        ),
    )

    suite.run_case(
        "L4-M3-10",
        "G2,G8",
        "SHUTDOWN record is explicitly flushed to m3_transactions.jsonl before bridge exit",
        sim_mode,
        lambda: (records[-1]["op"] == "SHUTDOWN" and records[-1]["status"] == "OK", final_cyc, f"last record op={records[-1]['op']}"),
    )

    # L4-M3-11: Reconstruct RTL shadow/active register & epoch state purely from m3_transactions.jsonl
    def case_m3_11() -> tuple[bool, int, str]:
        shadow_x = [0x0232] * 9
        active_x = [0x0232] * 9
        epoch = 0
        for r in records:
            if r["status"] != "OK":
                continue
            if r["op"] == "RESET":
                shadow_x = [0x0232] * 9
                active_x = [0x0232] * 9
                epoch = 0
            elif r["op"] == "MMIO_WRITE":
                addr = int(r["addr"], 16)
                wdata = int(r["wdata"], 16)
                if addr < 0x24:
                    shadow_x[addr >> 2] = wdata & 0xFFFF
                elif addr == 0x48 and (wdata & 1):
                    active_x = list(shadow_x)
                    epoch += 1
            elif r["op"] == "MMIO_READ":
                addr = int(r["addr"], 16)
                rdata = int(r["rdata"], 16)
                if 0x100 <= addr < 0x124 and rdata != active_x[(addr - 0x100) >> 2]:
                    return False, final_cyc, "reconstructed active_x mismatch"
                if addr == 0x50 and rdata != epoch:
                    return False, final_cyc, "reconstructed epoch mismatch"
        return epoch == metrics["final_epoch_count"] and active_x[0] == 0x0177, final_cyc, f"reconstructed epoch={epoch} active_x[0]=0x{active_x[0]:04x}"

    suite.run_case("L4-M3-11", "G2,G8", "Deterministic state-machine replay from m3_transactions.jsonl matches RTL readbacks & final metrics", sim_mode, case_m3_11)

    return suite.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
