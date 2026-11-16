"""Layer 4 Set 3.1: End-to-End Pipeline Transaction Scenarios (L4-P01 .. L4-P12)."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.l4_py_harness import L4PySuite, running_bridge, send_rpc  # noqa: E402


def compute_sx(cz: int) -> int:
    b0 = ((cz >> 0) ^ (cz >> 1)) & 1
    b1 = ((cz >> 1) ^ (cz >> 2) ^ (cz >> 4) ^ (cz >> 5)) & 1
    b2 = ((cz >> 3) ^ (cz >> 4) ^ (cz >> 6) ^ (cz >> 7)) & 1
    b3 = ((cz >> 7) ^ (cz >> 8)) & 1
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)


def compute_sz(cx: int) -> int:
    b0 = ((cx >> 0) ^ (cx >> 1) ^ (cx >> 3) ^ (cx >> 4)) & 1
    b1 = ((cx >> 2) ^ (cx >> 5)) & 1
    b2 = ((cx >> 3) ^ (cx >> 6)) & 1
    b3 = ((cx >> 4) ^ (cx >> 5) ^ (cx >> 7) ^ (cx >> 8)) & 1
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", default="")
    ap.add_argument("--work-dir", default="/tmp/avid_l4_set3_e2e")
    args = ap.parse_args()

    suite = L4PySuite("set3_e2e_transactions", "Set3", args.out_jsonl or None)
    sim_mode = "verilator_c++_rtl"
    tx_seq = 100

    def next_tid() -> int:
        nonlocal tx_seq
        tx_seq += 1
        return tx_seq

    with running_bridge(suite.bridge_bin, Path(args.work_dir), "p_e2e", max_clients=8) as (sock_path, _, _, _):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect(str(sock_path))

            # L4-P01: Register configuration followed by syndrome transfer
            def case_p01() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "RESET", "data": 10})
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000054", "data": "0x4", "byte_enable": 1})
                for q in range(9):
                    wx = 0x00A0 if q == 8 else 0x0280
                    wz = 0x00A0 if q == 5 else 0x0280
                    send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": f"0x{4*q:08x}", "data": f"0x{wx:04x}", "byte_enable": 3})
                    send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": f"0x{0x24+4*q:08x}", "data": f"0x{wz:04x}", "byte_enable": 3})
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                # sx=0x2 ({q2,q5} weight-1 Z degeneracy) -> steers cz to q5; sz=0x8 ({q7,q8} weight-1 X degeneracy) -> steers cx to q8
                r = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0082", "tlast": True})
                cx, cz = int(r["corr_x"], 16), int(r["corr_z"], 16)
                return (r["bp_cycles"] == 22 and cx == (1 << 8) and cz == (1 << 5)), r["completed_cycle"], f"K=4 bp=22 cx=0x{cx:x} cz=0x{cz:x}"

            suite.run_case("L4-P01", "G2,G4", "Inhomogeneous wx!=wz + K=4 configuration steers subsequent decode", sim_mode, case_p01)

            # L4-P02: Syndrome transfer across all 9 single-qubit X and Z faults
            def case_p02() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "RESET", "data": 10})
                cal = [0x0190, 0x0240, 0x0190, 0x0190, 0x0140, 0x0260, 0x0260, 0x0240, 0x0190]
                for q, w in enumerate(cal):
                    send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": f"0x{4*q:08x}", "data": f"0x{w:04x}", "byte_enable": 3})
                    send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": f"0x{0x24+4*q:08x}", "data": f"0x{w:04x}", "byte_enable": 3})
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                cyc = 0
                for q in range(9):
                    sx = compute_sx(1 << q)
                    sz = compute_sz(1 << q)
                    r = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{(sz << 4) | sx:04x}", "tlast": False})
                    cyc = r["completed_cycle"]
                    if compute_sx(int(r["corr_z"], 16)) != sx or compute_sz(int(r["corr_x"], 16)) != sz:
                        return False, cyc, f"qubit {q} syndrome unsatisfied"
                return True, cyc, "9/9 single-qubit X & Z syndromes satisfied"

            suite.run_case("L4-P02", "G2,G4", "All 9 single-qubit X/Z error syndromes produce valid parity-satisfying corrections", sim_mode, case_p02)

            # L4-P03: Multiple sequential syndrome transfers with exact +22 cycle spacing
            def case_p03() -> tuple[bool, int, str]:
                prev_c = None
                for i in range(16):
                    r = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{((15-i)<<4)|i:04x}", "tlast": (i == 15)})
                    if prev_c is not None and r["accepted_cycle"] != prev_c:
                        return False, r["completed_cycle"], "gap in cycle timeline"
                    if (r["completed_cycle"] - r["accepted_cycle"]) != 22:
                        return False, r["completed_cycle"], "latency != 22"
                    prev_c = r["completed_cycle"]
                return True, prev_c or 0, "16 sequential syndromes had contiguous +22 cycle timeline"

            suite.run_case("L4-P03", "G2", "16 sequential syndrome transfers maintain contiguous +22 cycle timeline", sim_mode, case_p03)

            # L4-P04: X-only syndrome sweep (sx=1..15, sz=0 -> corr_x==0)
            def case_p04() -> tuple[bool, int, str]:
                cyc = 0
                for sx in range(1, 16):
                    r = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{sx:04x}", "tlast": False})
                    cx, cz = int(r["corr_x"], 16), int(r["corr_z"], 16)
                    cyc = r["completed_cycle"]
                    if cx != 0 or compute_sx(cz) != sx:
                        return False, cyc, f"sx={sx} failed"
                return True, cyc, "15/15 X-only syndromes satisfied with corr_x==0"

            suite.run_case("L4-P04", "G4", "All 15 non-zero X-only syndromes satisfy H_X*corr_z==sx with corr_x==0", sim_mode, case_p04)

            # L4-P05: Z-only syndrome sweep (sx=0, sz=1..15 -> corr_z==0)
            def case_p05() -> tuple[bool, int, str]:
                cyc = 0
                for sz in range(1, 16):
                    r = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{sz<<4:04x}", "tlast": False})
                    cx, cz = int(r["corr_x"], 16), int(r["corr_z"], 16)
                    cyc = r["completed_cycle"]
                    if cz != 0 or compute_sz(cx) != sz:
                        return False, cyc, f"sz={sz} failed"
                return True, cyc, "15/15 Z-only syndromes satisfied with corr_z==0"

            suite.run_case("L4-P05", "G4", "All 15 non-zero Z-only syndromes satisfy H_Z*corr_x==sz with corr_z==0", sim_mode, case_p05)

            # L4-P06: Simultaneous X/Z operation vs separate X and Z decodes
            def case_p06() -> tuple[bool, int, str]:
                cyc = 0
                for sx in (0x3, 0x6, 0x9, 0xF):
                    for sz in (0x5, 0xA, 0xC, 0xF):
                        rx = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{sx:04x}", "tlast": False})
                        rz = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{sz<<4:04x}", "tlast": False})
                        rb = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{(sz<<4)|sx:04x}", "tlast": False})
                        cyc = rb["completed_cycle"]
                        if rb["corr_z"] != rx["corr_z"] or rb["corr_x"] != rz["corr_x"]:
                            return False, cyc, "X/Z channel interference detected"
                return True, cyc, "16 simultaneous X+Z pairs identical to isolated channel runs"

            suite.run_case("L4-P06", "G4", "Simultaneous X+Z decoding matches isolated single-channel results identically", sim_mode, case_p06)

            # L4-P07: Shadow weight-bank isolation prior to COMMIT
            def case_p07() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "RESET", "data": 10})
                r_pre = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0002", "tlast": False})
                for q in range(9):
                    w = 0x0080 if q == 5 else 0x0320
                    send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": f"0x{0x24+4*q:08x}", "data": f"0x{w:04x}", "byte_enable": 3})
                r_uncom = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0002", "tlast": False})
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                r_post = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0002", "tlast": False})
                ok = (r_pre["corr_z"] == r_uncom["corr_z"]) and (int(r_post["corr_z"], 16) == (1 << 5))
                return ok, r_post["completed_cycle"], f"pre={r_pre['corr_z']} uncom={r_uncom['corr_z']} post={r_post['corr_z']}"

            suite.run_case("L4-P07", "G6", "Shadow writes do not affect decoder output until 0x048=1 commits atomically", sim_mode, case_p07)

            # L4-P08: Delayed weight commit across multi-beat window
            def case_p08() -> tuple[bool, int, str]:
                for q in range(9):
                    w = 0x0080 if q == 2 else 0x0320
                    send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": f"0x{0x24+4*q:08x}", "data": f"0x{w:04x}", "byte_enable": 3})
                # Run 4 syndromes BEFORE committing -> all 4 must still use prior epoch (q5=0x0020)
                for _ in range(4):
                    r = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0002", "tlast": False})
                    if int(r["corr_z"], 16) != (1 << 5):
                        return False, r["completed_cycle"], "premature weight swap"
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                r2 = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0002", "tlast": True})
                return int(r2["corr_z"], 16) == (1 << 2), r2["completed_cycle"], "4 delayed rounds held q5, post-commit switched to q2"

            suite.run_case("L4-P08", "G6", "Delayed weight commit holds active weights steady for 4 rounds until commit", sim_mode, case_p08)

            # L4-P09: Epoch interrupt (epoch_irq == 1 iff tlast == True/1)
            def case_p09() -> tuple[bool, int, str]:
                irqs = []
                cyc = 0
                for tl in (False, 0, True, 1, False):
                    r = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0006", "tlast": tl})
                    irqs.append(r["epoch_irq"])
                    cyc = r["completed_cycle"]
                return irqs == [0, 0, 1, 1, 0], cyc, f"irqs={irqs}"

            suite.run_case("L4-P09", "G6", "epoch_irq fires strictly on tlast=True/1 and stays 0 on tlast=False/0", sim_mode, case_p09)

            # L4-P10: Mid-session RESET and clean recovery
            def case_p10() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "RESET", "data": 12})
                ep = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_READ", "address": "0x00000050"})
                ax = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_READ", "address": "0x00000124"})
                syn = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": "0x0096", "tlast": True})
                ok = int(ep["rdata"], 16) == 0 and int(ax["rdata"], 16) == 0x0232 and syn["status"] == "OK"
                return ok, syn["completed_cycle"], f"post-reset epoch={ep['rdata']} active={ax['rdata']}"

            suite.run_case("L4-P10", "G6,G7", "Mid-session RESET clears epoch=0, restores 0x0232, and resumes decoding", sim_mode, case_p10)

            # L4-P11: Decoder busy condition over socket (SYNDROME_START -> STATUS==0x3 -> torn write SLVERR)
            def case_p11() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000000", "data": "0x0155", "byte_enable": 3})
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_START", "data": "0x0006", "tlast": True})
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                st = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_READ", "address": "0x0000004C"})
                torn = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000000", "data": "0x03EE", "byte_enable": 3})
                wt = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_WAIT"})
                act = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_READ", "address": "0x00000100"})
                ok = (int(st["rdata"], 16) == 3 and torn["status"] == "ERR_INVALID_REGISTER" and int(act["rdata"], 16) == 0x0155)
                return ok, act["completed_cycle"], f"STATUS={st['rdata']} torn={torn['status']} act={act['rdata']}"

            suite.run_case("L4-P11", "G1,G6", "Busy decoder sets STATUS=0x3, rejects torn shadow writes, and commits cleanly after finish", sim_mode, case_p11)

            # L4-P12: 1,000+ mixed operation cycles without state leakage
            def case_p12() -> tuple[bool, int, str]:
                send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "RESET", "data": 10})
                cyc = 0
                commits = 0
                for step in range(1000):
                    sx = (step * 3 + 1) & 0xF
                    sz = (step * 5 + 2) & 0xF
                    tl = ((step + 1) % 100) == 0
                    r = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "SYNDROME_TRANSFER", "address": "0x0", "data": f"0x{(sz<<4)|sx:04x}", "tlast": tl})
                    cyc = r["completed_cycle"]
                    if r["status"] != "OK":
                        return False, cyc, f"failed at step {step}"
                    if tl:
                        w = 0x0120 + commits * 4
                        send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000000", "data": f"0x{w:04x}", "byte_enable": 3})
                        send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_WRITE", "address": "0x00000048", "data": "0x1", "byte_enable": 1})
                        commits += 1
                ep = send_rpc(s, {"protocol_version": 1, "transaction_id": next_tid(), "operation": "MMIO_READ", "address": "0x00000050"})
                return int(ep["rdata"], 16) == commits, ep["completed_cycle"], f"1000 syndromes + {commits} commits -> epoch={int(ep['rdata'], 16)}"

            suite.run_case("L4-P12", "G2,G7", "1,000 mixed syndrome/MMIO/commit operations complete with zero state drift", sim_mode, case_p12)

    return suite.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
