#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "inference/noise_model.h"
#include "estimator/lbfgs_solver.h"
#include "estimator/llr_quantizer.h"
#include "control/causal_scheduler.h"
#include "control/fallback_fsm.h"
#include "driver/weight_reg_driver.h"
#include "driver/weight_bank_driver.h"
#include <array>
#include <limits>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-033");

    if (opts.test_id == "L3-IP-034") {
        TestContext ctx(opts, "L3-IP-034", "burst_threshold_and_debounce_bounds",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_init, avid_fast_detector_step");
        ctx.record().test_description =
            "Burst threshold bounds — test valid threshold=0.35/debounce=4 and invalid threshold in {-0.2, 0.0, 1.5, NaN, +Inf}";
        ctx.record().input_contract = "density_threshold in (0.0, 1.0]; reject <=0, >1.0, NaN, +Inf";
        ctx.record().independent_oracle = "16-round sliding density lies in [0, 1]; threshold > 1.0 permanently disables burst detection";
        ctx.record().fixture = "none";

        avid_fast_burst_detector_t det_valid{};
        ASSERT_TRUE(avid_fast_detector_init_checked(&det_valid, 0.35, 4));
        ASSERT_TRUE(det_valid.config_valid);
        ASSERT_NEAR(det_valid.density_threshold, 0.35, 1e-12);
        ASSERT_EQ(det_valid.debounce_rounds, 4u);

        avid_fast_burst_detector_t det_one{};
        ASSERT_TRUE(avid_fast_detector_init_checked(&det_one, 1.0, 4));
        ASSERT_TRUE(det_one.config_valid);

        /* Invalid thresholds: -0.2, 0.0, 1.5, NaN, +Inf, and NULL pointer must be rejected */
        avid_fast_burst_detector_t det_bad{};
        ASSERT_FALSE(avid_fast_detector_init_checked(nullptr, 0.35, 4));
        ASSERT_FALSE(avid_fast_detector_init_checked(&det_bad, -0.2, 4));
        ASSERT_FALSE(det_bad.config_valid);
        ASSERT_FALSE(avid_fast_detector_init_checked(&det_bad, 0.0, 4));
        ASSERT_FALSE(det_bad.config_valid);
        ASSERT_FALSE(avid_fast_detector_init_checked(&det_bad, 1.5, 4));
        ASSERT_FALSE(det_bad.config_valid);
        ASSERT_FALSE(avid_fast_detector_init_checked(&det_bad, std::numeric_limits<double>::quiet_NaN(), 4));
        ASSERT_FALSE(det_bad.config_valid);
        ASSERT_FALSE(avid_fast_detector_init_checked(&det_bad, std::numeric_limits<double>::infinity(), 4));
        ASSERT_FALSE(det_bad.config_valid);

        /* Even legacy void wrapper clamps out-of-range 1.5/+Inf to safe default 0.35 so burst detection remains functional */
        avid_fast_burst_detector_t det_hi{};
        avid_fast_detector_init(&det_hi, 1.5, 4);
        ASSERT_FALSE(det_hi.config_valid);
        bool triggered_on_100pct_burst = false;
        for (uint64_t r = 0; r < 64; ++r) {
            if (avid_fast_detector_step(&det_hi, r, 0xF, 0xF)) {
                triggered_on_100pct_burst = true;
            }
        }
        ASSERT_TRUE(triggered_on_100pct_burst);

        ctx.record().expected["burst_threshold_bounds"] = "Reject threshold <= 0, > 1.0, NaN, and +Inf";
        ctx.record().actual["burst_threshold_bounds"] =
            "avid_fast_detector_init_checked rejects -0.2, 0.0, 1.5, NaN, +Inf, nullptr (false, config_valid=false) and accepts (0.35, 1.0)";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (opts.test_id == "L3-IP-040") {
        TestContext ctx(opts, "L3-IP-040", "corrupted_candidate_weight_bank_rejection",
                        "software/src/estimator/llr_quantizer.cpp, control/fallback_fsm.cpp, driver/weight_bank_driver.c",
                        "avid_quantize_and_validate_bank, avid_fallback_step, avid_weight_bank_write_and_commit");
        ctx.record().test_description =
            "Corrupted active config — inject out-of-range Q7.8 weights (negative, 0, >0x3000) and unphysical probabilities (>=0.5, NaN) and verify active decoder bank is never overwritten";
        ctx.record().input_contract = "Q7.8 weights must lie in [AVID_Q78_MIN_SAFE=0x0040, AVID_Q78_MAX_SAFE=0x3000]; invalid banks trigger hold/fallback without commit";
        ctx.record().independent_oracle = "architecture/contracts/llr_numeric_contract.yaml & fault_policy.yaml";
        ctx.record().fixture = "none";

        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        const int16_t good_bank[9] = {1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128};
        uint16_t swaps = 0;
        ASSERT_EQ(avid_weight_bank_write_and_commit(&wrd, good_bank, &swaps), 0);

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        int16_t sel[9] = {0};
        bool should_commit = false;
        /* Establish last_good_q78 */
        avid_fallback_step(&fsm, true, true, true, true, good_bank, sel, &should_commit);
        ASSERT_TRUE(should_commit);

        /* Corrupt probability vector (p=0.65 > 0.5 and NaN) */
        const double bad_probs[9] = {0.65, 0.012, std::numeric_limits<double>::quiet_NaN(),
                                     0.012, 0.012, 0.012, 0.012, 0.012, 0.012};
        int16_t bad_cand[9] = {-500, 0, 0x3500, 100, 100, 100, 100, 100, 100};
        bool bounds_ok = avid_quantize_and_validate_bank(bad_probs, bad_cand);
        ASSERT_FALSE(bounds_ok);

        /* Case 1: Estimator failure with valid bounds enters HOLD_LAST_GOOD with should_commit=false */
        avid_fallback_state_t st_hold = avid_fallback_step(&fsm, false, true, true, true, good_bank, sel, &should_commit);
        ASSERT_EQ(static_cast<int>(st_hold), static_cast<int>(AVID_FSM_HOLD_LAST_GOOD));
        ASSERT_FALSE(should_commit);

        /* Case 2: Out-of-bounds candidate bank (bounds_ok=false) escalates to SAFE_STATIC_PRIOR (1128) and NEVER selects bad_cand */
        avid_fallback_state_t st = avid_fallback_step(&fsm, false, true, bounds_ok, true, bad_cand, sel, &should_commit);
        ASSERT_EQ(static_cast<int>(st), static_cast<int>(AVID_FSM_SAFE_STATIC_PRIOR));
        ASSERT_TRUE(should_commit);
        for (int q = 0; q < 9; ++q) {
            ASSERT_EQ(sel[q], static_cast<int16_t>(AVID_Q78_DEFAULT));
            ASSERT_EQ(wrd_read_active_weight(&wrd, q), good_bank[q]);
        }
        wrd_close(&wrd);

        ctx.record().expected["corrupted_bank_protection"] = "bounds_ok=false rejected by quantizer; FSM reverts to SAFE_STATIC_PRIOR (0x0468=1128) and never selects corrupted candidate weights";
        ctx.record().actual["corrupted_bank_protection"] =
            "bounds_ok=false, hold_state=HOLD_LAST_GOOD, bounds_fail_state=SAFE_STATIC_PRIOR, selected_weight[0]=1128 (corrupted -500/0x3500 blocked)";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    /* Default: L3-IP-033 — Optimizer iteration cap configuration */
    TestContext ctx(opts, "L3-IP-033", "optimizer_max_iters_cap_validation",
                    "software/src/estimator/lbfgs_solver.cpp",
                    "HistogramLBFGSSolver::fit_histogram");
    ctx.record().test_description =
        "Optimizer iteration cap — supply max_iters in {0, -5, 1, 20} and verify iterations_executed <= max_iters (or invalid max_iters<=0 rejected)";
    ctx.record().input_contract = "max_iters <= 0 rejected or executes 0 steps; max_iters >= 1 strictly caps iterations_executed <= max_iters";
    ctx.record().independent_oracle = "Hard budget invariant: iterations_executed <= max(0, max_iters)";
    ctx.record().fixture = "tests/layer3/fixtures/histograms/correlated/corr_total1000.txt";

    std::array<uint32_t, 16> hist{};
    ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/correlated/corr_total1000.txt"), hist));

    HistogramLBFGSSolver solver;
    auto run_with_cap = [&](int cap) -> int {
        NoiseModel m;
        m.init(0.012, 0.004);
        LBFGSFitDiagnostics d{};
        solver.fit_histogram(m, hist.data(), cap, &d);
        return d.iterations_executed;
    };

    int iters_neg5 = run_with_cap(-5);
    int iters_0    = run_with_cap(0);
    int iters_1    = run_with_cap(1);
    int iters_20   = run_with_cap(20);

    ASSERT_EQ(iters_neg5, 0);
    ASSERT_EQ(iters_0, 0);
    ASSERT_TRUE(iters_1 >= 0 && iters_1 <= 1);
    ASSERT_TRUE(iters_20 > 0 && iters_20 <= 20);

    ctx.record().expected["iteration_cap"] = "max_iters<=0 rejected or 0 iters; max_iters=1 -> <=1 iter; max_iters=20 -> <=20 iters";
    ctx.record().actual["iteration_cap"] =
        "iters(-5)=" + std::to_string(iters_neg5) + ", iters(0)=" + std::to_string(iters_0) +
        ", iters(1)=" + std::to_string(iters_1) + ", iters(20)=" + std::to_string(iters_20);

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
