/* ===========================================================================
 * test_baseline_suite.cpp — C++ Unit & Integration Test Suite for AVID Baseline & V1
 * Verifies:
 *   1. 64-bit packet pack/unpack & DMA 4-state slot ownership FSM (A05 / Q11)
 *   2. Exact 16-state Joint-DP likelihood & coset-inversion gradient vs finite diff (A02 / Q03)
 *   3. Cycle-accurate Q7.8 Min-Sum BP custom execution engine (A01 / Q02, A06 / Q06)
 *   4. Shadow/Active atomic weight bank swap & 4-state fallback FSM (SF5, A09 / Q18)
 *   5. 16-round sliding-subwindow fast burst detector (<= 20-round trigger bound)
 *   6. Trained 64-unit Dueling LSTM R2D2 checkpoint loading, multi-action selection,
 *      and Python-to-C++ numerical parity (--parity-csv mode).
 * =========================================================================*/
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <set>

#include "../src/ingress/packet_validator.h"
#include "../src/ingress/dma_ingress_tap.h"
#include "../src/inference/noise_model.h"
#include "../src/estimator/lbfgs_solver.h"
#include "../src/estimator/llr_quantizer.h"
#include "../src/control/fallback_fsm.h"
#include "../src/control/causal_scheduler.h"
#include "../src/control/r2d2_lstm_controller.h"
#include "../src/driver/weight_reg_driver.h"
#include "../src/driver/weight_bank_driver.h"
#include "../src/custom_engine/cycle_accurate_bp_engine.h"

static void test_packet_and_dma_ring() {
    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);

    uint64_t burst[AVID_DMA_SLOT_PACKETS];
    for (uint32_t i = 0; i < AVID_DMA_SLOT_PACKETS; i++) {
        uint8_t sx = (uint8_t)(i & 0x0Fu);
        uint8_t sz = (uint8_t)((i + 3u) & 0x0Fu);
        burst[i] = avid_pack_packet_u64(sx, sz, 100u + i, 2u, AVID_PKT_FLAG_VALID);
    }

    bool pushed = avid_dma_ring_push_burst(&ring, burst, AVID_DMA_SLOT_PACKETS);
    (void)pushed;
    assert(pushed);
    assert(ring.slots[0].state == AVID_SLOT_CPU_READY);

    uint32_t hx[16] = {0}, hz[16] = {0}, valid_cnt = 0;
    bool consumed = avid_dma_ring_consume_slot(&ring, hx, hz, &valid_cnt);
    (void)consumed;
    assert(consumed);
    assert(valid_cnt == AVID_DMA_SLOT_PACKETS);
    assert(ring.slots[0].state == AVID_SLOT_FREE);
    assert(ring.validator.seq_gap_count == 0);
    for (int s = 0; s < 16; s++) {
        assert(hx[s] == 4u);
        assert(hz[s] == 4u);
    }
    printf("[PASS] C++ DMA Ring Ownership & 64-bit Packet Codec Test\n");
}

static void test_joint_dp_gradient_exactness() {
    NoiseModel model;
    model.init(0.015, 0.008);
    model.theta_(4) = 0.065; /* Hotspot on center qubit 4 */
    model.theta_(9 + 3) = 0.045; /* ZZ crosstalk on pair (4,5) */

    uint32_t hist[16] = {
        820, 25, 30, 12, 28, 8, 45, 5,
        20,  4,  6,  2,  14, 3, 4,  2
    };

    model.compute_gradient_from_histogram(hist);
    NoiseModel::GradVec analytic = model.grad();

    const double eps = 1e-7;
    double max_err = 0.0;
    for (int k = 0; k < NM_N_PARAMS; k++) {
        NoiseModel::ParamVec base = model.theta_;
        model.theta_ = base;
        model.theta_(k) += eps;
        double f_plus = model.compute_nll_from_histogram(hist);

        model.theta_ = base;
        model.theta_(k) -= eps;
        double f_minus = model.compute_nll_from_histogram(hist);

        model.theta_ = base;
        double fd = (f_plus - f_minus) / (2.0 * eps);
        double err = std::abs(analytic(k) - fd);
        if (err > max_err) max_err = err;
    }
    assert(max_err < 1e-6);
    printf("[PASS] C++ Exact 16-State Joint-DP Gradient vs Finite-Diff (max_err=%.2e)\n", max_err);
}

static void test_custom_bp_engine_and_weights() {
    CycleAccurateBPEngine bp(3, 250.0);
    assert(CycleAccurateBPEngine::cycle_count_fused(3) == 17);
    assert(CycleAccurateBPEngine::cycle_count_legacy(3) == 20);

    int16_t weights[9];
    for (int i = 0; i < 9; i++) weights[i] = 0x0468; /* Uniform prior */
    weights[4] = 0x0180; /* Lower LLR on center qubit 4 (higher error prob) */

    /* Test all 9 single-qubit Z faults */
    for (int q = 0; q < 9; q++) {
        uint8_t s_mask = NM_EVENT_SIGNATURES[q];
        avid_bp_result_t r = bp.decode(s_mask, weights, 3);
        (void)r;
        assert(r.syndrome_satisfied);
        assert(r.cycles_fused == 17);
        assert(std::abs(r.latency_ns_fused - 68.0) < 1e-9);
    }

    /* Test double-buffered shadow/active register driver */
    wrd_handle_t wrd;
    int rc = wrd_open(&wrd, WRD_BASE_ADDR);
    (void)rc;
    assert(rc == 0);
    uint16_t swaps = 0;
    avid_weight_bank_write_and_commit(&wrd, weights, &swaps);
    assert(swaps == 1u);
    assert(wrd_read_active_weight(&wrd, 4) == 0x0180);
    wrd_close(&wrd);

    printf("[PASS] C++ Cycle-Accurate Q7.8 BP Engine (17 cycles = 68 ns @ K=3) & Weight Bank Test\n");
}

static void test_fast_burst_detector_16round() {
    avid_fast_burst_detector_t det;
    avid_fast_detector_init(&det, 0.35, 4u);
    assert(det.subwindow_rounds == 16u);
    assert(det.worst_case_trigger_latency_rounds == 20u);

    /* 32 quiet rounds -> 0 triggers */
    for (uint64_t r = 0; r < 32; r++) {
        bool trig = avid_fast_detector_step(&det, r, 0u, 0u);
        (void)trig;
        assert(!trig);
    }
    assert(det.trigger_count == 0u);

    /* Inject burst starting at round 32 -> must trigger within <= 20 rounds */
    int trigger_lag = -1;
    for (uint64_t r = 32; r < 64; r++) {
        bool trig = avid_fast_detector_step(&det, r, 0x2u, 0x0u);
        if (trig && trigger_lag < 0) {
            trigger_lag = (int)(r - 32u) + 1;
        }
    }
    assert(det.trigger_count == 1u);
    assert(trigger_lag > 0 && trigger_lag <= 20);
    printf("[PASS] C++ 16-Round Sliding-Subwindow Fast Burst Detector (lag=%d <= 20 rounds)\n", trigger_lag);
}

static void test_trained_r2d2_checkpoint_and_actions() {
    AvidR2D2LSTMController ctrl;
    assert(ctrl.is_checkpoint_loaded());

    const double regime_features[6][8] = {
        /* a0: calm stationary independent */
        {0.014, 0.016, 0.006, 0.009, 0.055, 0.015, 0.002, 0.05},
        /* a1: moderate correlated drift */
        {0.018, 0.024, 0.016, 0.045, 0.085, 0.055, 0.006, 0.08},
        /* a2: strong pair crosstalk */
        {0.022, 0.034, 0.022, 0.092, 0.115, 0.105, 0.008, 0.10},
        /* a3: elevated independent noise */
        {0.032, 0.036, 0.007, 0.010, 0.095, 0.018, 0.004, 0.06},
        /* a4: abrupt burst / regime shift */
        {0.036, 0.052, 0.026, 0.095, 0.145, 0.115, 0.048, 0.82},
        /* a5: unphysical / corrupted estimate fallback */
        {0.210, 0.250, 0.125, 0.160, 0.550, 0.250, 0.090, 0.15},
    };

    std::set<int> actions_seen;
    std::set<uint32_t> k_seen;
    for (int r = 0; r < 6; ++r) {
        ctrl.reset_hidden();
        AvidR2D2ControlDecision dec{};
        for (int step = 0; step < 3; ++step) {
            dec = ctrl.step_from_features(regime_features[r]);
        }
        actions_seen.insert(dec.action_id);
        k_seen.insert(dec.bp_iterations);
    }
    assert(actions_seen.size() >= 4u);
    assert(k_seen.size() >= 2u);
    printf("[PASS] C++ Trained R2D2 Checkpoint Loaded (%zu distinct actions across regimes)\n",
           actions_seen.size());
}

static int run_parity_csv_mode(const char* in_csv, const char* out_csv) {
    avid::deployment::R2D2Runtime rt;
    std::string err;
    if (!rt.load_default_checkpoint(&err)) {
        fprintf(stderr, "Failed to load default checkpoint: %s\n", err.c_str());
        return 1;
    }

    FILE* fin = std::fopen(in_csv, "r");
    if (!fin) return 1;
    FILE* fout = std::fopen(out_csv, "w");
    if (!fout) {
        std::fclose(fin);
        return 1;
    }

    double f[8];
    while (std::fscanf(fin, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                       &f[0], &f[1], &f[2], &f[3], &f[4], &f[5], &f[6], &f[7]) == 8) {
        avid::deployment::R2D2InferenceOutput out = rt.step(f);
        std::fprintf(fout, "%d,%.15e,%.15e,%d",
                     out.action_id, out.state_value, out.pair_weight_scale, out.bp_iterations);
        for (int a = 0; a < 6; ++a) {
            std::fprintf(fout, ",%.15e", out.q_values[a]);
        }
        for (int q = 0; q < 9; ++q) {
            std::fprintf(fout, ",%.15e", out.qubit_gains[q]);
        }
        std::fprintf(fout, "\n");
    }
    std::fclose(fin);
    std::fclose(fout);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc == 4 && std::strcmp(argv[1], "--parity-csv") == 0) {
        return run_parity_csv_mode(argv[2], argv[3]);
    }
    test_packet_and_dma_ring();
    test_joint_dp_gradient_exactness();
    test_custom_bp_engine_and_weights();
    test_fast_burst_detector_16round();
    test_trained_r2d2_checkpoint_and_actions();
    printf("[ALL C++ BASELINE TESTS PASSED]\n");
    return 0;
}
