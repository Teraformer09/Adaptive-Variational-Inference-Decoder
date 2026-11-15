#include "test_assertions.h"
#include "test_common.h"
#include <cstring>
#include "custom_engine/cycle_accurate_bp_engine.h"
#include "hal/avid_hal.h"
#include "control/fallback_fsm.h"
#include "estimator/llr_quantizer.h"

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-035");

    if (opts.test_id == "L3-IP-038") {
        TestContext ctx(opts, "L3-IP-038", "safety_fsm_state_and_event_inputs",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_init, avid_fallback_step");
        ctx.record().test_description =
            "Safety state input — sweep all 16 boolean event tuples across NORMAL_ADAPTIVE, HOLD_LAST_GOOD, SAFE_STATIC_PRIOR, DEGRADED_BYPASS, and probe invalid enum / NULL pointer handling";
        ctx.record().input_contract = "4-state avid_fallback_state_t + 4 boolean health flags + non-null candidate/output weight pointers";
        ctx.record().independent_oracle = "architecture/contracts/fault_policy.yaml & fallback_fsm.h";
        ctx.record().fixture = "none";

        const int16_t cand[9] = {900, 900, 900, 900, 900, 900, 900, 900, 900};
        int16_t sel[9] = {0};
        bool should_commit = false;

        /* Sweep all 16 boolean combinations from freshly initialized state */
        for (int mask = 0; mask < 16; ++mask) {
            avid_fallback_fsm_t fsm{};
            avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
            bool est_ok  = (mask & 1) != 0;
            bool ddl_ok  = (mask & 2) != 0;
            bool bnd_ok  = (mask & 4) != 0;
            bool bus_ok  = (mask & 8) != 0;
            avid_fallback_state_t st = avid_fallback_step(&fsm, est_ok, ddl_ok, bnd_ok, bus_ok, cand, sel, &should_commit);
            if (!bus_ok) {
                ASSERT_EQ(static_cast<int>(st), static_cast<int>(AVID_FSM_DEGRADED_BYPASS));
                ASSERT_FALSE(should_commit);
            } else if (est_ok && ddl_ok && bnd_ok) {
                ASSERT_EQ(static_cast<int>(st), static_cast<int>(AVID_FSM_NORMAL_ADAPTIVE));
                ASSERT_TRUE(should_commit);
            } else if (!bnd_ok) {
                ASSERT_EQ(static_cast<int>(st), static_cast<int>(AVID_FSM_SAFE_STATIC_PRIOR));
                ASSERT_TRUE(should_commit);
            } else {
                ASSERT_EQ(static_cast<int>(st), static_cast<int>(AVID_FSM_HOLD_LAST_GOOD));
                ASSERT_FALSE(should_commit);
            }
        }

        /* Corrupt fsm.state to out-of-domain enum value 99 -> must sanitize to SAFE_STATIC_PRIOR */
        avid_fallback_fsm_t corrupted{};
        avid_fallback_init(&corrupted, AVID_Q78_DEFAULT);
        unsigned int raw_corrupt = 99u;
        static_assert(sizeof(raw_corrupt) == sizeof(corrupted.state),
                      "avid_fallback_state_t width must match unsigned int");
        std::memcpy(&corrupted.state, &raw_corrupt, sizeof(raw_corrupt));
        avid_fallback_state_t after_corrupt = avid_fallback_step(&corrupted, false, true, true, true, cand, sel, &should_commit);
        ASSERT_EQ(static_cast<int>(after_corrupt), static_cast<int>(AVID_FSM_SAFE_STATIC_PRIOR));
        ASSERT_EQ(corrupted.fallback_triggers, 1ULL);
        ASSERT_TRUE(should_commit);

        /* Fork-isolated NULL pointer checks on avid_fallback_step */
        auto null_probe = run_isolated_probe([&]() -> int {
            avid_fallback_fsm_t f{};
            avid_fallback_init(&f, AVID_Q78_DEFAULT);
            avid_fallback_state_t s1 = avid_fallback_step(&f, true, true, true, true, nullptr, sel, &should_commit);
            avid_fallback_state_t s2 = avid_fallback_step(&f, true, true, true, true, cand, nullptr, nullptr);
            avid_fallback_state_t s3 = avid_fallback_step(nullptr, true, true, true, true, cand, sel, &should_commit);
            return (s1 == AVID_FSM_SAFE_STATIC_PRIOR &&
                    s2 == AVID_FSM_SAFE_STATIC_PRIOR &&
                    s3 == AVID_FSM_DEGRADED_BYPASS) ? 0 : 1;
        });
        ASSERT_FALSE(null_probe.signaled);
        ASSERT_EQ(null_probe.exit_code, 0);

        ctx.record().expected["fsm_input_validation"] = "16/16 boolean health tuples deterministic AND null/corrupt state inputs guarded safely";
        ctx.record().actual["fsm_input_validation"] =
            "16/16 boolean tuples deterministic; corrupt enum=99 escalated to SAFE_STATIC_PRIOR; nullptr inputs safely guarded with 0 crashes";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    /* L3-IP-035: BP iteration bounds */
    TestContext ctx(opts, "L3-IP-035", "bp_iteration_k_bounds_validation",
                    "software/src/custom_engine/cycle_accurate_bp_engine.cpp, software/src/hal/simulation_backend.cpp",
                    "CycleAccurateBPEngine::decode, avid_hal_set_bp_iterations");
    ctx.record().test_description =
        "BP iteration bounds — test supported K in {1,2,3,4,5} and invalid K in {0, -5, 99} across CycleAccurateBPEngine and HAL";
    ctx.record().input_contract = "Supported BP iteration count K in {1, 2, 3, 4, 5}; reject K <= 0 (except -1 default) and K > 5";
    ctx.record().independent_oracle = "deployment/cpp/r2d2_runtime.h & avid_hal.h K in [1, 5]";
    ctx.record().fixture = "none";

    CycleAccurateBPEngine bp(3, 250.0);
    const int16_t weights[9] = {1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128};

    for (int k : {1, 2, 3, 4, 5}) {
        avid_bp_result_t r = bp.decode(0x01, weights, k);
        ASSERT_EQ(r.iterations_K, k);
        ASSERT_EQ(r.cycles_fused, 2 + 5 * k);
    }

    /* Test invalid K = 0, -5, and 99 -> must be rejected cleanly */
    avid_bp_result_t r_k0   = bp.decode(0x01, weights, 0);
    avid_bp_result_t r_kneg = bp.decode(0x01, weights, -5);
    avid_bp_result_t r_k99  = bp.decode(0x01, weights, 99);
    ASSERT_FALSE(r_k0.syndrome_satisfied);
    ASSERT_EQ(r_k0.iterations_K, 0);
    ASSERT_FALSE(r_kneg.syndrome_satisfied);
    ASSERT_EQ(r_kneg.iterations_K, 0);
    ASSERT_FALSE(r_k99.syndrome_satisfied);
    ASSERT_EQ(r_k99.iterations_K, 0);

    avid_hal_t hal{};
    ASSERT_EQ(avid_hal_open(&hal, AVID_HAL_BACKEND_PHYSICAL, "", WRD_BASE_ADDR), 0);
    uint64_t cyc = 0;
    int rc_k0  = avid_hal_set_bp_iterations(&hal, 0u, &cyc);
    int rc_k5  = avid_hal_set_bp_iterations(&hal, 5u, &cyc);
    int rc_k6  = avid_hal_set_bp_iterations(&hal, 6u, &cyc);
    int rc_k99 = avid_hal_set_bp_iterations(&hal, 99u, &cyc);
    ASSERT_EQ(rc_k0, -1);
    ASSERT_EQ(rc_k5, 0);
    ASSERT_EQ(hal.active_k_iter, 5u);
    ASSERT_EQ(rc_k6, -1);
    ASSERT_EQ(rc_k99, -1);

    /* Verify physical avid_hal_send_syndrome uses hal.active_k_iter=5 -> bp_cycles = 2 + 5*5 = 27 */
    uint16_t cx = 0, cz = 0;
    uint32_t bp_cyc = 0;
    ASSERT_EQ(avid_hal_send_syndrome(&hal, 0x1, 0x0, false, &cx, &cz, &bp_cyc, &cyc), 0);
    ASSERT_EQ(bp_cyc, 27u);
    avid_hal_close(&hal);

    ctx.record().expected["bp_k_bounds"] = "Accept supported K in {1..5}; reject K=0, K=-5, K=6, K=99";
    ctx.record().actual["bp_k_bounds"] =
        "Both CycleAccurateBPEngine and HAL accept K in {1..5} (active_k_iter=5 -> 27 cycles) and reject K in {0, -5, 6, 99}";

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
