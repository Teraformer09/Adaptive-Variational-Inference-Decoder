/* ===========================================================================
 * test_weight_bank_flow.cpp — Layer 5 Set 3.2 Atomic Weight-Bank System Flow
 * Canonical IDs: L5-FLOW-17 .. L5-FLOW-36 (Gate G4)
 * Verifies shadow -> commit_snap -> active triple-buffered semantics, mid-decode
 * deferral across BP states, busy-lock SLVERR, and multi-epoch adaptive commits.
 * =========================================================================*/
#include "../common/l5_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l5::L5Suite suite("set3_weight_bank_flow", "Set3", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) return 2;
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L5-FLOW-17: Uncommitted shadow writes isolated from active bank & decoder */
    suite.run_case("L5-FLOW-17", "G4", "Writing all 18 shadow registers without COMMIT leaves active_x/z and decoded corrections unchanged", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        RtlDecodeTrace pre = rtl.decode_syndrome(0x2u, 0x0u, false);
        for (int i = 0; i < 9; i++) {
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(i), 0x0080u, 0x3u, &cyc);
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), (i == 5) ? 0x0060u : 0x0380u, 0x3u, &cyc);
        }
        RtlDecodeTrace post_sh = rtl.decode_syndrome(0x2u, 0x0u, false);
        uint32_t az5 = 0;
        rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(5), &az5, &cyc);
        det = "pre_cz=" + std::to_string(pre.corr_z) + " uncommitted_cz=" + std::to_string(post_sh.corr_z) + " active_z[5]=0x0232";
        return pre.corr_z == post_sh.corr_z && az5 == 0x0232u;
    });

    /* L5-FLOW-18: Idle COMMIT atomically updates all 18 active weights in 1 cycle & increments epoch_count */
    suite.run_case("L5-FLOW-18", "G4", "Idle COMMIT (0x048=1) updates all 18 active_x/z weights and increments epoch_count 0->1", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 9; i++) {
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(i), 0x0110u + i, 0x3u, &cyc);
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), 0x0210u + i, 0x3u, &cyc);
        }
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        uint32_t ep = 0, st = 99;
        rtl.axil_read(0x050u, &ep, &cyc);
        rtl.axil_read(0x04Cu, &st, &cyc);
        for (int i = 0; i < 9; i++) {
            uint32_t ax = 0, az = 0;
            rtl.axil_read(avid_l5::L5_OFF_ACTIVE_X(i), &ax, &cyc);
            rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(i), &az, &cyc);
            if (ax != (0x0110u + i) || az != (0x0210u + i)) return false;
        }
        det = "18/18 active weights updated; epoch_count=" + std::to_string(ep) + " status=" + std::to_string(st);
        return ep == 1u && st == 0u;
    });

    /* L5-FLOW-19: Single-qubit partial shadow update + COMMIT preserves other 17 weights */
    suite.run_case("L5-FLOW-19", "G4", "Updating only shadow_z[4] + COMMIT updates active_z[4] while keeping other 17 weights at 0x0232", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(4), 0x00D9u, 0x3u, &cyc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        for (int i = 0; i < 9; i++) {
            uint32_t ax = 0, az = 0;
            rtl.axil_read(avid_l5::L5_OFF_ACTIVE_X(i), &ax, &cyc);
            rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(i), &az, &cyc);
            if (ax != 0x0232u || az != ((i == 4) ? 0x00D9u : 0x0232u)) return false;
        }
        det = "active_z[4]=0x00D9, remaining 17 weights=0x0232";
        return true;
    });

    /* L5-FLOW-20: Multi-epoch sequential progression (epoch 0 -> 1 -> 2 -> 3 -> 4) steering decode */
    suite.run_case("L5-FLOW-20", "G4,G6", "Multi-epoch commits (0->1->2->3->4) steer degenerate sx=0x2 between q2, q5, {q0,q1}, and {q3,q4}", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        /* Epoch 1: favor q2 -> 0x004 */
        for (int i = 0; i < 9; i++) rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), (i == 2) ? 0x0080u : 0x0300u, 0x3u, &cyc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        RtlDecodeTrace e1 = rtl.decode_syndrome(0x2u, 0x0u, false);
        /* Epoch 2: favor q5 -> 0x020 */
        for (int i = 0; i < 9; i++) rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), (i == 5) ? 0x0080u : 0x0300u, 0x3u, &cyc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        RtlDecodeTrace e2 = rtl.decode_syndrome(0x2u, 0x0u, false);
        /* Epoch 3: favor pair {q0,q1} -> 0x003 */
        for (int i = 0; i < 9; i++) rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), (i == 0 || i == 1) ? 0x0060u : 0x0300u, 0x3u, &cyc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        RtlDecodeTrace e3 = rtl.decode_syndrome(0x2u, 0x0u, false);
        /* Epoch 4: favor pair {q3,q4} -> 0x018 */
        for (int i = 0; i < 9; i++) rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), (i == 3 || i == 4) ? 0x0060u : 0x0300u, 0x3u, &cyc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        RtlDecodeTrace e4 = rtl.decode_syndrome(0x2u, 0x0u, false);
        uint32_t ep = 0;
        rtl.axil_read(0x050u, &ep, &cyc);
        det = "ep=" + std::to_string(ep) + " cz={0x" + std::to_string(e1.corr_z) + ",0x" +
              std::to_string(e2.corr_z) + ",0x" + std::to_string(e3.corr_z) + ",0x" + std::to_string(e4.corr_z) + "}";
        return ep == 4u && e1.corr_z == 0x004u && e2.corr_z == 0x020u && e3.corr_z == 0x003u && e4.corr_z == 0x018u;
    });

    /* L5-FLOW-21: Strict shadow_x vs shadow_z bank independence */
    suite.run_case("L5-FLOW-21", "G4", "Committing shadow_x never perturbs active_z and committing shadow_z never perturbs active_x", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 9; i++) rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(i), 0x0144u, 0x3u, &cyc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        for (int i = 0; i < 9; i++) {
            uint32_t az = 0;
            rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(i), &az, &cyc);
            if (az != 0x0232u) return false;
        }
        for (int i = 0; i < 9; i++) rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), 0x0288u, 0x3u, &cyc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        for (int i = 0; i < 9; i++) {
            uint32_t ax = 0, az = 0;
            rtl.axil_read(avid_l5::L5_OFF_ACTIVE_X(i), &ax, &cyc);
            rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(i), &az, &cyc);
            if (ax != 0x0144u || az != 0x0288u) return false;
        }
        det = "active_x=0x0144 & active_z=0x0288 strictly orthogonal";
        return true;
    });

    /* L5-FLOW-22: Mid-decode COMMIT during ST_INIT (early core) deferral & uncorrupted output */
    suite.run_case("L5-FLOW-22", "G4", "COMMIT issued on cycle 1 of core (ST_INIT) defers swap until decode finishes with old bank", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        /* Stage weights that would steer sx=0x2 to q5 (0x020) instead of default q2 (0x004) */
        for (int i = 0; i < 9; i++) rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), (i == 5) ? 0x0060u : 0x0320u, 0x3u, &cyc);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x2u, 0x0u, false, &acc); /* Now in ST_INIT */
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        uint32_t st = 0, az5_mid = 0;
        rtl.axil_read(0x04Cu, &st, &cyc);
        rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(5), &az5_mid, &cyc);
        RtlDecodeTrace inflight;
        rtl.finish_async_syndrome(&inflight);
        RtlDecodeTrace next_tr = rtl.decode_syndrome(0x2u, 0x0u, false);
        cyc = rtl.sim_cycle;
        det = "mid_STATUS=" + std::to_string(st) + " inflight_cz=0x" + std::to_string(inflight.corr_z) +
              " post_swap_cz=0x" + std::to_string(next_tr.corr_z);
        return st == 3u && az5_mid == 0x0232u && inflight.corr_z == 0x004u && next_tr.corr_z == 0x020u;
    });

    /* L5-FLOW-23: Mid-decode COMMIT during ST_WAIT_C2V (mid-iteration check-node pipe) */
    suite.run_case("L5-FLOW-23", "G4", "COMMIT issued mid-iteration (ST_WAIT_C2V, core cycle 8) defers cleanly until bp_ready_in==1", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(5), 0x0060u, 0x3u, &cyc);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x2u, 0x0u, false, &acc);
        rtl.step_clock(6); /* Core cycle ~8: inside ST_WAIT_C2V of iter 1 */
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        uint32_t st = 0;
        rtl.axil_read(0x04Cu, &st, &cyc);
        RtlDecodeTrace inflight;
        rtl.finish_async_syndrome(&inflight);
        uint32_t ep = 0;
        rtl.axil_read(0x050u, &ep, &cyc);
        det = "mid_STATUS=" + std::to_string(st) + " inflight_cz=0x" + std::to_string(inflight.corr_z) + " post_ep=" + std::to_string(ep);
        return st == 3u && inflight.corr_z == 0x004u && ep == 1u;
    });

    /* L5-FLOW-24: Mid-decode COMMIT right before ST_HARD_DECIDE (final core cycle) */
    suite.run_case("L5-FLOW-24", "G4", "COMMIT arriving on final core cycle (ST_HARD_DECIDE) does not corrupt combinational OSD-0 decision", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 9; i++) rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), (i == 5) ? 0x0050u : 0x0350u, 0x3u, &cyc);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x2u, 0x0u, false, &acc);
        rtl.step_clock(14); /* Right before ST_HARD_DECIDE (cycle 15..17) */
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        RtlDecodeTrace inflight;
        rtl.finish_async_syndrome(&inflight);
        RtlDecodeTrace post = rtl.decode_syndrome(0x2u, 0x0u, false);
        cyc = rtl.sim_cycle;
        det = "inflight_cz=0x" + std::to_string(inflight.corr_z) + " post_cz=0x" + std::to_string(post.corr_z);
        return inflight.corr_z == 0x004u && post.corr_z == 0x020u;
    });

    /* L5-FLOW-25: Busy-lock SLVERR on shadow_x[0..8] writes while swap_pending==1 */
    suite.run_case("L5-FLOW-25", "G1,G4", "All 9 shadow_x offsets reject writes with SLVERR while swap_pending==1 && !bp_ready_in", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x054u, 5u, 0x1u, &cyc); /* K=5 -> 27 core cycles so plenty of busy window */
        rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(0), 0x0144u, 0x3u, &cyc);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x6u, 0x0u, false, &acc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        rtl.bp_busy = true;
        for (int q = 0; q < 9; q++) {
            if (rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(q), 0x03AAu, 0x3u, &cyc) || rtl.last_bresp != 2u) return false;
        }
        RtlDecodeTrace tr;
        rtl.finish_async_syndrome(&tr);
        uint32_t ax0 = 0;
        rtl.axil_read(0x100u, &ax0, &cyc);
        det = "9/9 shadow_x writes rejected with SLVERR; active_x[0]=0x0144";
        return ax0 == 0x0144u;
    });

    /* L5-FLOW-26: Busy-lock SLVERR on shadow_z[0..8] writes while swap_pending==1 */
    suite.run_case("L5-FLOW-26", "G1,G4", "All 9 shadow_z offsets reject writes with SLVERR while swap_pending==1 && !bp_ready_in", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x054u, 5u, 0x1u, &cyc);
        rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(8), 0x0166u, 0x3u, &cyc);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x6u, 0x0u, false, &acc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        rtl.bp_busy = true;
        for (int q = 0; q < 9; q++) {
            if (rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(q), 0x03BBu, 0x3u, &cyc) || rtl.last_bresp != 2u) return false;
        }
        RtlDecodeTrace tr;
        rtl.finish_async_syndrome(&tr);
        uint32_t az8 = 0;
        rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(8), &az8, &cyc);
        det = "9/9 shadow_z writes rejected with SLVERR; active_z[8]=0x0166";
        return az8 == 0x0166u;
    });

    /* L5-FLOW-27: Busy-lock SLVERR on duplicate COMMIT (0x048=1) while swap_pending==1 */
    suite.run_case("L5-FLOW-27", "G1,G4", "Duplicate COMMIT write while swap_pending==1 returns SLVERR and increments epoch_count by +1 only", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x6u, 0x0u, false, &acc);
        bool c1 = rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        rtl.bp_busy = true;
        bool c2_rej = !rtl.axil_write(0x048u, 1u, 0x1u, &cyc) && (rtl.last_bresp == 2u);
        RtlDecodeTrace tr;
        rtl.finish_async_syndrome(&tr);
        uint32_t ep = 99;
        rtl.axil_read(0x050u, &ep, &cyc);
        det = "first_commit=" + std::to_string(c1) + " dup_rejected=" + std::to_string(c2_rej) + " final_epoch=" + std::to_string(ep);
        return c1 && c2_rej && ep == 1u;
    });

    /* L5-FLOW-28: Staging shadow writes ALLOWED while decoder is busy BEFORE COMMIT (swap_pending==0) */
    suite.run_case("L5-FLOW-28", "G4", "Shadow writes succeed (OKAY) while decoder is busy when swap_pending==0 (concurrent slow-path staging)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x054u, 5u, 0x1u, &cyc);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x6u, 0x0u, false, &acc);
        /* swap_pending == 0 while bp_busy == 1: staging shadow_x[0..3] MUST succeed! */
        for (int q = 0; q < 4; q++) {
            if (!rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(q), 0x0170u + q, 0x3u, &cyc)) return false;
        }
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        RtlDecodeTrace tr;
        rtl.finish_async_syndrome(&tr);
        uint32_t ax3 = 0;
        rtl.axil_read(avid_l5::L5_OFF_ACTIVE_X(3), &ax3, &cyc);
        det = "concurrent shadow staging during active decode committed active_x[3]=0x0173";
        return ax3 == 0x0173u;
    });

    /* L5-FLOW-29: STATUS, EPOCH_COUNT, and active_x/z reads succeed while shadow bank is locked */
    suite.run_case("L5-FLOW-29", "G1,G4", "Read-only STATUS, EPOCH_COUNT, and active_x/z remain readable (OKAY) while swap_pending==1", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x6u, 0x0u, false, &acc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        uint32_t st = 0, ep = 99, ax0 = 0, az0 = 0;
        bool ok = rtl.axil_read(0x04Cu, &st, &cyc) &&
                  rtl.axil_read(0x050u, &ep, &cyc) &&
                  rtl.axil_read(0x100u, &ax0, &cyc) &&
                  rtl.axil_read(0x124u, &az0, &cyc);
        RtlDecodeTrace tr;
        rtl.finish_async_syndrome(&tr);
        det = "STATUS=" + std::to_string(st) + " ep=" + std::to_string(ep) + " ax0=0x0232";
        return ok && st == 3u && ep == 0u && ax0 == 0x0232u && az0 == 0x0232u;
    });

    /* L5-FLOW-30: COMMIT (0x048) readback reflects {31'b0, swap_pending} */
    suite.run_case("L5-FLOW-30", "G1,G4", "Reading COMMIT (0x048) returns 1 while swap_pending is deferred mid-decode and 0 once swapped", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x6u, 0x0u, false, &acc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        uint32_t com_mid = 0, com_post = 9;
        rtl.axil_read(0x048u, &com_mid, &cyc);
        RtlDecodeTrace tr;
        rtl.finish_async_syndrome(&tr);
        rtl.axil_read(0x048u, &com_post, &cyc);
        det = "COMMIT readback mid=" + std::to_string(com_mid) + " post=" + std::to_string(com_post);
        return com_mid == 1u && com_post == 0u;
    });

    /* L5-FLOW-31: STATUS (0x04C) bitfield encoding verification (0x0, 0x2, 0x3) */
    suite.run_case("L5-FLOW-31", "G1,G4", "STATUS (0x04C) encodes [0]=swap_pending, [1]=bp_busy accurately across 0x0 (idle), 0x2 (busy), 0x3 (deferred)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint32_t s_idle = 9, s_busy = 9, s_def = 9, s_done = 9;
        rtl.axil_read(0x04Cu, &s_idle, &cyc);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x6u, 0x0u, false, &acc);
        rtl.axil_read(0x04Cu, &s_busy, &cyc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        rtl.axil_read(0x04Cu, &s_def, &cyc);
        RtlDecodeTrace tr;
        rtl.finish_async_syndrome(&tr);
        rtl.axil_read(0x04Cu, &s_done, &cyc);
        det = "STATUS states: idle=" + std::to_string(s_idle) + " busy=" + std::to_string(s_busy) +
              " deferred=" + std::to_string(s_def) + " done=" + std::to_string(s_done);
        return s_idle == 0u && s_busy == 2u && s_def == 3u && s_done == 0u;
    });

    /* L5-FLOW-32: Queued RX FIFO packet + mid-decode COMMIT swaps atomically on ST_IDLE gap between packets */
    suite.run_case("L5-FLOW-32", "G3,G4", "Deferred COMMIT swaps atomically on the 1-cycle ST_IDLE boundary between two back-to-back queued syndromes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 9; i++) rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), (i == 5) ? 0x0060u : 0x0320u, 0x3u, &cyc);
        /* Push 2 back-to-back sx=0x2 syndromes into RX FIFO */
        for (int p = 0; p < 2; p++) {
            rtl.top->s_axis_syndrome_tdata = 0x02u;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.step_clock(1);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.step_clock(2); /* Packet 0 now active in BP core; Packet 1 waiting in RX FIFO! */
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc); /* Queue COMMIT during Packet 0! */
        std::vector<uint16_t> czs;
        for (int g = 0; g < 60 && czs.size() < 2; g++) {
            if (rtl.top->m_axis_correction_tvalid && rtl.top->m_axis_correction_tready) {
                czs.push_back(static_cast<uint16_t>((rtl.top->m_axis_correction_tdata >> 9) & 0x1FFu));
            }
            rtl.step_clock(1);
        }
        cyc = rtl.sim_cycle;
        det = "pkt0_cz=0x" + std::to_string(czs.size() > 0 ? czs[0] : 0) +
              " pkt1_cz=0x" + std::to_string(czs.size() > 1 ? czs[1] : 0);
        return czs.size() == 2 && czs[0] == 0x004u && czs[1] == 0x020u;
    });

    /* L5-FLOW-33: 32-epoch adaptive drift-tracking progression (epoch 0..32) */
    suite.run_case("L5-FLOW-33", "G4", "32 sequential adaptive weight commits track shifting qubit hotspots cleanly to epoch_count==32", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint32_t ep = 1; ep <= 32; ep++) {
            int hot_q = static_cast<int>(ep % 9);
            for (int i = 0; i < 9; i++) {
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), (i == hot_q) ? 0x0080u : 0x0320u, 0x3u, &cyc);
            }
            rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
            uint8_t sx = avid_l5::gf2_hx_mul_ez(static_cast<uint16_t>(1u << hot_q));
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, 0u, (ep % 8) == 0);
            if (tr.corr_z != (1u << hot_q)) return false;
        }
        uint32_t final_ep = 0;
        rtl.axil_read(0x050u, &final_ep, &cyc);
        det = "32/32 shifting hotspot epochs decoded accurately; epoch_count=" + std::to_string(final_ep);
        return final_ep == 32u;
    });

    /* L5-FLOW-34: Hardware reset while swap_pending==1 clears lock and restores 0x0232 */
    suite.run_case("L5-FLOW-34", "G4,G5", "Asserting aresetn=0 during pending mid-decode swap clears swap_pending=0 and restores 0x0232 defaults", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x000u, 0x0155u, 0x3u, &cyc);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x6u, 0x0u, false, &acc);
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        rtl.reset_model(6);
        uint32_t st = 9, ep = 9, ax0 = 0;
        rtl.axil_read(0x04Cu, &st, &cyc);
        rtl.axil_read(0x050u, &ep, &cyc);
        rtl.axil_read(0x100u, &ax0, &cyc);
        det = "post-reset STATUS=" + std::to_string(st) + " ep=" + std::to_string(ep) + " ax0=0x0232";
        return st == 0u && ep == 0u && ax0 == 0x0232u;
    });

    /* L5-FLOW-35: Post-swap shadow unlock verification */
    suite.run_case("L5-FLOW-35", "G4", "Shadow bank unlocks immediately once deferred swap completes, accepting new epoch writes with OKAY", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint32_t delayed = 0;
        if (!rtl.verify_busy_swap_protection(&delayed)) return false;
        bool unlocked = rtl.axil_write(0x000u, 0x0299u, 0x3u, &cyc) &&
                        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        uint32_t ax0 = 0, ep = 0;
        rtl.axil_read(0x100u, &ax0, &cyc);
        rtl.axil_read(0x050u, &ep, &cyc);
        det = "unlocked post-swap write succeeded; active_x[0]=0x0299, epoch=" + std::to_string(ep);
        return unlocked && ax0 == 0x0299u && ep == 2u;
    });

    /* L5-FLOW-36: Epoch accounting invariant across commits and resets */
    suite.run_case("L5-FLOW-36", "G4,G8", "Epoch accounting invariant holds across boot commits, adaptive commits, rejected commits, and resets", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 5; i++) rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        uint32_t ep_pre_rst = 0;
        rtl.axil_read(0x050u, &ep_pre_rst, &cyc);
        rtl.reset_model(6); /* clears 5 commits */
        for (int i = 0; i < 3; i++) rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        rtl.axil_write(0x048u, 0u, 0x1u, &cyc); /* no-op */
        uint32_t ep_post = 0;
        rtl.axil_read(0x050u, &ep_post, &cyc);
        det = "pre_reset_ep=" + std::to_string(ep_pre_rst) + ", cleared=5, post_commits=3 -> ep=" + std::to_string(ep_post);
        return ep_pre_rst == 5u && ep_post == 3u;
    });

    return suite.finalize();
}
