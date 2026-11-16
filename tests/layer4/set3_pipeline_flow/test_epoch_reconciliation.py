"""Layer 4 Set 3.3: Live Software <-> RTL Epoch & Weight-Bank Reconciliation (L4-EPOCH-01 .. L4-EPOCH-07)."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.l4_py_harness import L4PySuite, running_bridge, send_rpc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", default="")
    ap.add_argument("--work-dir", default="/tmp/avid_l4_set3_epoch")
    args = ap.parse_args()

    suite = L4PySuite("set3_epoch_reconciliation", "Set3", args.out_jsonl or None)
    sim_mode = "verilator_c++_rtl"
    tid = 1000

    def nt() -> int:
        nonlocal tid
        tid += 1
        return tid

    with running_bridge(suite.bridge_bin, Path(args.work_dir), "epoch_rec", max_clients=4) as (sock_path, _, _, _):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect(str(sock_path))

            # L4-EPOCH-01: Shadow writes without commit leave epoch_count and active bank unchanged
            def case_ep01() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "RESET", "data": 10})
                for q in range(9):
                    send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": f"0x{4*q:08x}", "data": "0x0133", "byte_enable": 3})
                ep = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000050"})
                ax = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000100"})
                ok = int(ep["rdata"], 16) == 0 and int(ax["rdata"], 16) == 0x0232
                return ok, ax["completed_cycle"], f"uncommitted: epoch={ep['rdata']} active_x[0]={ax['rdata']}"

            suite.run_case("L4-EPOCH-01", "G6", "Uncommitted shadow writes leave RTL epoch_count=0 and active_x=0x0232", sim_mode, case_ep01)

            # L4-EPOCH-02: Valid COMMIT (0x048=1) advances epoch_count by +1 and syncs active == shadow
            def case_ep02() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                ep = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000050"})
                ax = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000100"})
                ok = int(ep["rdata"], 16) == 1 and int(ax["rdata"], 16) == 0x0133
                return ok, ax["completed_cycle"], f"committed: epoch={ep['rdata']} active_x[0]={ax['rdata']}"

            suite.run_case("L4-EPOCH-02", "G6", "COMMIT=1 atomically advances epoch_count=1 and updates active bank", sim_mode, case_ep02)

            # L4-EPOCH-03: Writing 0 to COMMIT (0x048=0) does NOT increment epoch_count
            def case_ep03() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": "0x00000000", "data": "0x01AA", "byte_enable": 3})
                send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x0", "byte_enable": 1})
                ep = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000050"})
                ax = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000100"})
                ok = int(ep["rdata"], 16) == 1 and int(ax["rdata"], 16) == 0x0133
                return ok, ax["completed_cycle"], f"after COMMIT=0: epoch={ep['rdata']} active_x[0]={ax['rdata']}"

            suite.run_case("L4-EPOCH-03", "G6", "No-op COMMIT=0 leaves epoch_count=1 and active_x[0]=0x0133 unchanged", sim_mode, case_ep03)

            # L4-EPOCH-04: Mid-decode COMMIT deferral & duplicate COMMIT SLVERR protection
            def case_ep04() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "SYNDROME_START", "data": "0x0006", "tlast": True})
                c1 = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                ep_mid = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000050"})
                c2_dup = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "SYNDROME_WAIT"})
                ep_post = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000050"})
                ax_post = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000100"})
                ok = (
                    c1["status"] == "OK"
                    and int(ep_mid["rdata"], 16) == 1  # NOT yet incremented mid-decode!
                    and c2_dup["status"] == "ERR_INVALID_REGISTER"
                    and int(ep_post["rdata"], 16) == 2  # Incremented exactly once on decode finish!
                    and int(ax_post["rdata"], 16) == 0x01AA
                )
                return ok, ax_post["completed_cycle"], f"ep_mid={ep_mid['rdata']} dup={c2_dup['status']} ep_post={ep_post['rdata']}"

            suite.run_case("L4-EPOCH-04", "G6", "Mid-decode COMMIT defers epoch increment until decode completes and rejects duplicate COMMIT", sim_mode, case_ep04)

            # L4-EPOCH-05: 12-window slow-loop software<->RTL epoch lockstep without off-by-one drift
            def case_ep05() -> tuple[bool, int, str]:
                sw_epoch = 2
                cyc = 0
                for w in range(12):
                    for r in range(8):
                        tl = r == 7
                        syn = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{((r+w)&0xF):04x}", "tlast": tl})
                        if syn["epoch_irq"] != (1 if tl else 0):
                            return False, syn["completed_cycle"], "irq mismatch"
                    val = 0x0140 + w * 3
                    send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": "0x00000000", "data": f"0x{val:04x}", "byte_enable": 3})
                    send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                    sw_epoch += 1
                    ep = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000050"})
                    ax = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000100"})
                    cyc = ax["completed_cycle"]
                    if int(ep["rdata"], 16) != sw_epoch or int(ax["rdata"], 16) != val:
                        return False, cyc, f"drift at window {w}"
                return True, cyc, f"12 windows reconciled with zero off-by-one drift (epoch={sw_epoch})"

            suite.run_case("L4-EPOCH-05", "G6", "12 consecutive slow-loop windows maintain exact software==RTL epoch & active bank agreement", sim_mode, case_ep05)

            # L4-EPOCH-06: Independent rx_epoch_count (TLAST boundary count) vs weight_epoch (COMMIT count)
            def case_ep06() -> tuple[bool, int, str]:
                # Send 3 TLAST=true syndromes without committing weights: epoch_count (0x050) must stay 14!
                for _ in range(3):
                    r = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0001", "tlast": True})
                    if r["epoch_irq"] != 1:
                        return False, r["completed_cycle"], "missing epoch_irq"
                ep = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000050"})
                return int(ep["rdata"], 16) == 14, ep["completed_cycle"], f"3 TLAST pulses left weight epoch_count={int(ep['rdata'], 16)} untouched"

            suite.run_case("L4-EPOCH-06", "G6", "Syndrome TLAST/epoch_irq pulses do not spuriously increment weight epoch_count (0x050)", sim_mode, case_ep06)

            # L4-EPOCH-07: Hardware RESET resets epoch_count -> 0 and reconciles post-reset epoch 1
            def case_ep07() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "RESET", "data": 10})
                ep0 = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000050"})
                send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": "0x00000000", "data": "0x0188", "byte_enable": 3})
                send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                ep1 = send_rpc(s, {"protocol_version": 1, "transaction_id": nt(), "operation": "MMIO_READ", "address": "0x00000050"})
                ok = int(ep0["rdata"], 16) == 0 and int(ep1["rdata"], 16) == 1
                return ok, ep1["completed_cycle"], f"post-reset ep0={ep0['rdata']} -> ep1={ep1['rdata']}"

            suite.run_case("L4-EPOCH-07", "G6,G7", "RESET clears epoch_count=0 and subsequent commit cleanly starts new epoch 1", sim_mode, case_ep07)

    return suite.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
