"""Layer 4 Set 3.2: Live QEMU ARM Cortex-A9 / ARM64 <-> Bridge <-> Verilator Lockstep (L4-LOCK-01 .. L4-LOCK-09)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.l4_py_harness import L4PySuite, running_bridge  # noqa: E402


def read_elf_machine(elf_path: Path) -> tuple[int, int]:
    """Return (ei_class, e_machine) from ELF header: (1=32b, 2=64b), (40=ARM, 183=AARCH64, 62=X86_64)."""
    raw = elf_path.read_bytes()[:24]
    if raw[:4] != b"\x7fELF":
        return (0, 0)
    ei_class = raw[4]
    e_machine = struct.unpack("<H", raw[18:20])[0]
    return (ei_class, e_machine)


def run_guest_lockstep(
    repo_root: Path,
    bridge_bin: Path,
    work_dir: Path,
    arch: str,
    tag: str,
    rounds: int = 64,
    window: int = 16,
    k_iters: int = 3,
    osd_mode: int = 1,
    reset_mid: bool = False,
) -> tuple[dict, dict, list[dict], list[dict]]:
    trace_out = work_dir / f"{tag}_{arch}_guest.jsonl"
    metrics_out = work_dir / f"{tag}_{arch}_guest_metrics.json"
    with running_bridge(bridge_bin, work_dir, f"{tag}_{arch}", max_clients=2) as (sock_path, tx_log, br_metrics, proc):
        env = dict(os.environ, AVID_QEMU_ARCH=arch)
        cmd = [
            "bash",
            str(repo_root / "emulator" / "qemu" / "run.sh"),
            "--backend",
            "sim",
            "--socket",
            str(sock_path),
            "--rounds",
            str(rounds),
            "--window",
            str(window),
            "--k-iters",
            str(k_iters),
            "--osd-mode",
            str(osd_mode),
            "--trace-out",
            str(trace_out),
            "--metrics-out",
            str(metrics_out),
        ]
        if reset_mid:
            cmd.append("--reset-mid-run")
        res = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True, check=True)
        assert res.returncode == 0
    g_m = json.loads(metrics_out.read_text(encoding="utf-8"))
    b_m = json.loads(br_metrics.read_text(encoding="utf-8"))
    g_t = [json.loads(line) for line in trace_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    b_t = [json.loads(line) for line in tx_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return g_m, b_m, g_t, b_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", default="")
    ap.add_argument("--work-dir", default="/tmp/avid_l4_set3_lockstep")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    suite = L4PySuite("set3_qemu_verilator_lockstep", "Set3", args.out_jsonl or None)
    repo_root = suite.repo_root
    sim_mode = "verilator_c++_rtl"

    # L4-LOCK-01: Genuine ARMv7 Cortex-A9 (EM_ARM=40) & ARM64 Cortex-A53 (EM_AARCH64=183) ELF verification
    def case_lock01() -> tuple[bool, int, str]:
        img_dir = repo_root / "emulator" / "qemu" / "images"
        c_a9, m_a9 = read_elf_machine(img_dir / "avid_daemon_cortex_a9")
        c_a64, m_a64 = read_elf_machine(img_dir / "avid_daemon_aarch64")
        c_x86, m_x86 = read_elf_machine(img_dir / "avid_daemon_x86_64")
        ok = (c_a9, m_a9) == (1, 40) and (c_a64, m_a64) == (2, 183) and (c_x86, m_x86) == (2, 62)
        return ok, 0, f"ELF headers: a9=({c_a9},{m_a9}) aarch64=({c_a64},{m_a64}) x86=({c_x86},{m_x86})"

    suite.run_case("L4-LOCK-01", "G5", "Verify compiled guest ELFs have genuine EM_ARM(40) 32b and EM_AARCH64(183) 64b headers", sim_mode, case_lock01)

    # Execute baseline 64-round lockstep across Cortex-A9, Cortex-A53 (aarch64), and x86_64 reference twin
    runs: dict[str, tuple[dict, dict, list[dict], list[dict]]] = {}
    for arch in ("cortex-a9", "aarch64", "x86_64"):
        runs[arch] = run_guest_lockstep(repo_root, suite.bridge_bin, work_dir, arch, "base64", rounds=64, window=16)

    # L4-LOCK-02: QEMU Zynq-7020 ARM Cortex-A9 (32-bit ILP32) live lockstep execution
    def case_lock02() -> tuple[bool, int, str]:
        gm, bm, _, _ = runs["cortex-a9"]
        ok = (
            gm["guest_isa"] == "arm_cortex_a9"
            and gm["pointer_bits"] == 32
            and gm["syndrome_satisfied"] == 64
            and gm["rejected_transactions"] == 0
            and bm["native_verilator_compiled"] is True
            and gm["final_completed_cycle"] == bm["final_sim_cycle"]
        )
        return ok, bm["final_sim_cycle"], f"isa={gm['guest_isa']}(32b) sat=64/64 cyc={bm['final_sim_cycle']}"

    suite.run_case("L4-LOCK-02", "G5", "Live QEMU ARM Cortex-A9 (32-bit) <-> Bridge <-> Verilator Vaa_qec_top 64-round lockstep", sim_mode, case_lock02)

    # L4-LOCK-03: QEMU Kria K26 ARM64 Cortex-A53 (64-bit LP64) live lockstep execution
    def case_lock03() -> tuple[bool, int, str]:
        gm, bm, _, _ = runs["aarch64"]
        ok = (
            gm["guest_isa"] == "aarch64"
            and gm["pointer_bits"] == 64
            and gm["syndrome_satisfied"] == 64
            and gm["rejected_transactions"] == 0
            and bm["native_verilator_compiled"] is True
            and gm["final_completed_cycle"] == bm["final_sim_cycle"]
        )
        return ok, bm["final_sim_cycle"], f"isa={gm['guest_isa']}(64b) sat=64/64 cyc={bm['final_sim_cycle']}"

    suite.run_case("L4-LOCK-03", "G5", "Live QEMU ARM64 Cortex-A53 (64-bit) <-> Bridge <-> Verilator Vaa_qec_top 64-round lockstep", sim_mode, case_lock03)

    # L4-LOCK-04: Cross-ISA bit-exact transaction equivalence (32b ARMv7 == 64b ARM64 == 64b x86_64)
    def case_lock04() -> tuple[bool, int, str]:
        _, bm_a9, gt_a9, bt_a9 = runs["cortex-a9"]
        _, bm_a64, gt_a64, bt_a64 = runs["aarch64"]
        _, bm_x86, gt_x86, bt_x86 = runs["x86_64"]

        def strip_isa(recs: list[dict]) -> list[dict]:
            return [{k: v for k, v in r.items() if k not in ("guest_isa", "ptr_bits")} for r in recs]

        ok = (
            strip_isa(gt_a9) == strip_isa(gt_a64) == strip_isa(gt_x86)
            and bt_a9 == bt_a64 == bt_x86
            and bm_a9 == bm_a64 == bm_x86
        )
        return ok, bm_a9["final_sim_cycle"], f"{len(bt_a9)} bridge JSONL records bit-identical across ARM32, ARM64, and x86_64"

    suite.run_case("L4-LOCK-04", "G5", "Bit-exact RPC & RTL trace equivalence across QEMU ARM32, QEMU ARM64, and x86_64 twin", sim_mode, case_lock04)

    # L4-LOCK-05: Guest trace <-> Bridge m3_transactions.jsonl 1-to-1 reconciliation
    def case_lock05() -> tuple[bool, int, str]:
        gm, bm, gt, bt = runs["cortex-a9"]
        by_tid = {r["tx_id"]: r for r in bt if r["tx_id"] != 99999999}
        for ev_rec in gt:
            if ev_rec["event"] == "SYNDROME":
                tid = ev_rec["tx_id_first"]
                br = by_tid.get(tid)
                if not br:
                    return False, 0, f"missing tx_id {tid}"
                if (
                    int(br["corr_x"], 16) != int(ev_rec["corr_x"], 16)
                    or int(br["corr_z"], 16) != int(ev_rec["corr_z"], 16)
                    or br["accepted_cycle"] != ev_rec["accepted_cycle"]
                    or br["completed_cycle"] != ev_rec["completed_cycle"]
                    or br["epoch_irq"] != ev_rec["epoch_irq"]
                ):
                    return False, 0, f"mismatch at tx_id {tid}"
        ok_counts = (
            gm["mmio_writes"] == bm["mmio_writes"]
            and gm["mmio_reads"] == bm["mmio_reads"]
            and gm["syndrome_packets"] == bm["syndrome_transfers"]
            and gm["final_hw_epoch"] == bm["final_epoch_count"]
        )
        return ok_counts, bm["final_sim_cycle"], f"64/64 syndrome events + {bm['mmio_writes']} MMIO writes reconciled 1-to-1"

    suite.run_case("L4-LOCK-05", "G2,G5", "Guest JSONL trace reconciles 1-to-1 with Bridge m3_transactions.jsonl", sim_mode, case_lock05)

    # L4-LOCK-06: Variable K=4 & Post-BP OSD Mode 2 lockstep across ARM Cortex-A9 & ARM64
    def case_lock06() -> tuple[bool, int, str]:
        gm_a9, bm_a9, gt_a9, bt_a9 = run_guest_lockstep(
            repo_root, suite.bridge_bin, work_dir, "cortex-a9", "k4m2", rounds=32, window=16, k_iters=4, osd_mode=2
        )
        gm_a64, bm_a64, _, bt_a64 = run_guest_lockstep(
            repo_root, suite.bridge_bin, work_dir, "aarch64", "k4m2", rounds=32, window=16, k_iters=4, osd_mode=2
        )
        # K=4, Mode=2 -> 4 + 5*4 = 24 core cycles, 9 + 5*4 = 29 AXI cycles
        synd_evs = [r for r in gt_a9 if r["event"] == "SYNDROME"]
        ok = (
            bt_a9 == bt_a64
            and all(r["bp_cycles"] == 24 and (r["completed_cycle"] - r["accepted_cycle"]) == 29 for r in synd_evs)
            and gm_a9["syndrome_satisfied"] == 32
            and gm_a64["syndrome_satisfied"] == 32
        )
        return ok, bm_a9["final_sim_cycle"], f"K=4 Mode=2 -> bp=24, axis=29 across ARM32 & ARM64 (cyc={bm_a9['final_sim_cycle']})"

    suite.run_case("L4-LOCK-06", "G3,G5", "QEMU ARM32 & ARM64 lockstep under K=4 and OSD Mode 2 (24 core / 29 AXI cycles)", sim_mode, case_lock06)

    # L4-LOCK-07: Mid-run hardware RESET & re-initialization lockstep inside QEMU guest
    def case_lock07() -> tuple[bool, int, str]:
        gm, bm, gt, _ = run_guest_lockstep(
            repo_root, suite.bridge_bin, work_dir, "cortex-a9", "midrst", rounds=32, window=16, reset_mid=True
        )
        events = [r["event"] for r in gt]
        ok = (
            "MID_RUN_RESET" in events
            and "COMMIT_POST_RESET" in events
            and bm["reset_operations"] == 2
            and gm["syndrome_satisfied"] == 32
            and bm["final_epoch_count"] == gm["final_hw_epoch"]
        )
        return ok, bm["final_sim_cycle"], f"reset_ops={bm['reset_operations']} post_reset_epoch={bm['final_epoch_count']}"

    suite.run_case("L4-LOCK-07", "G5,G7", "QEMU ARM Cortex-A9 mid-run RESET + re-commit reconciles cleanly with Verilator", sim_mode, case_lock07)

    # L4-LOCK-08: Anti-silent-fallback enforcement in emulator/qemu/run.sh (DEF-L4-09)
    def case_lock08() -> tuple[bool, int, str]:
        env = dict(os.environ, AVID_QEMU_ARCH="bogus_arch", AVID_ALLOW_X86_FALLBACK="0")
        res = subprocess.run(
            ["bash", str(repo_root / "emulator" / "qemu" / "run.sh"), "--self-test"],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
        )
        return res.returncode == 2, 0, f"bogus_arch rejected with rc={res.returncode} (no silent x86_64 fallback)"

    suite.run_case("L4-LOCK-08", "G5,G8", "emulator/qemu/run.sh refuses silent fallback to x86_64 when requested guest ISA is unavailable", sim_mode, case_lock08)

    # L4-LOCK-09: Sequential multi-guest handoff on one Verilator bridge session (ARM32 client #1 -> ARM64 client #2)
    def case_lock09() -> tuple[bool, int, str]:
        with running_bridge(suite.bridge_bin, work_dir, "handoff", max_clients=3) as (sock_path, _, br_metrics, _):
            for idx, arch in enumerate(("cortex-a9", "aarch64")):
                env = dict(os.environ, AVID_QEMU_ARCH=arch)
                gm_p = work_dir / f"handoff_{idx}_{arch}.json"
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
                        "16",
                        "--metrics-out",
                        str(gm_p),
                    ],
                    cwd=str(repo_root),
                    env=env,
                    check=True,
                )
        bm = json.loads(br_metrics.read_text(encoding="utf-8"))
        ok = bm["served_clients"] == 3 and bm["syndrome_transfers"] == 32 and bm["rejected_transactions"] == 0
        return ok, bm["final_sim_cycle"], f"served ARM32 + ARM64 sequentially on 1 bridge (syndrome_transfers={bm['syndrome_transfers']})"

    suite.run_case("L4-LOCK-09", "G5,G7", "Sequential QEMU ARM32 -> QEMU ARM64 client handoff on shared Verilator bridge session", sim_mode, case_lock09)

    return suite.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
