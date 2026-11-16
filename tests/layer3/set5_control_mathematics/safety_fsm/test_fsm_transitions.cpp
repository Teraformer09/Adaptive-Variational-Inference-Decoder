#include "test_assertions.h"
#include "test_common.h"
#include "control/fallback_fsm.h"
#include "estimator/llr_quantizer.h"
#include <cstring>
#include <set>
#include <string>
#include <utility>

using namespace avid::layer3_test;

namespace {

struct RefFsmOutcome {
    avid_fallback_state_t next_state;
    bool should_commit;
    int bank_src; /* 0 = candidate, 1 = last_good, 2 = static_prior */
};

RefFsmOutcome ref_fsm_transition(uint32_t incoming_holds,
                                 uint32_t max_holds,
                                 bool est_ok,
                                 bool ddl_ok,
                                 bool bnd_ok,
                                 bool bus_ok) {
    if (!bus_ok) {
        return {AVID_FSM_DEGRADED_BYPASS, false, 2};
    }
    if (est_ok && ddl_ok && bnd_ok) {
        return {AVID_FSM_NORMAL_ADAPTIVE, true, 0};
    }
    uint32_t holds = incoming_holds + 1u;
    if (!bnd_ok || holds >= max_holds) {
        return {AVID_FSM_SAFE_STATIC_PRIOR, true, 2};
    }
    return {AVID_FSM_HOLD_LAST_GOOD, false, 1};
}

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;
    const int16_t cand_bank[9] = {710, 720, 730, 740, 750, 760, 770, 780, 790};
    const int16_t good_bank[9] = {810, 820, 830, 840, 850, 860, 870, 880, 890};

    if (tid == "L3-MATH-053") {
        TestContext ctx(opts, tid, "fsm_exhaustive_64_state_input_completeness",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Evaluate all 4 states x 16 boolean input tuples (64 combinations) and verify every transition returns valid state 0..3";
        ctx.record().independent_oracle = "64/64 valid defined states in {0, 1, 2, 3}";

        int total_checked = 0;
        for (int st = 0; st < 4; ++st) {
            for (int mask = 0; mask < 16; ++mask) {
                avid_fallback_fsm_t fsm{};
                avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
                fsm.state = static_cast<avid_fallback_state_t>(st);
                std::memcpy(fsm.last_good_q78, good_bank, sizeof(good_bank));
                int16_t sel[9] = {0};
                bool commit = false;
                auto next = avid_fallback_step(&fsm,
                                               (mask & 1) != 0,
                                               (mask & 2) != 0,
                                               (mask & 4) != 0,
                                               (mask & 8) != 0,
                                               cand_bank, sel, &commit);
                ASSERT_GE(static_cast<int>(next), 0);
                ASSERT_LE(static_cast<int>(next), 3);
                ASSERT_EQ(next, fsm.state);
                total_checked++;
            }
        }
        ASSERT_EQ(total_checked, 64);
        ctx.record().expected["combinations_checked"] = 64;
        ctx.record().actual["combinations_checked"] = total_checked;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-054") {
        TestContext ctx(opts, tid, "fsm_transition_relation_exact_oracle_match",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Compare all 64 (state, input) transitions and holds=2 escalation against independent symbolic FSM specification";
        ctx.record().independent_oracle = "128/128 exact matches on next_state, should_commit, and selected bank";

        int matches = 0;
        for (uint32_t holds_in : {0u, 2u}) {
            for (int st = 0; st < 4; ++st) {
                for (int mask = 0; mask < 16; ++mask) {
                    avid_fallback_fsm_t fsm{};
                    avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
                    fsm.state = static_cast<avid_fallback_state_t>(st);
                    fsm.consecutive_holds = holds_in;
                    std::memcpy(fsm.last_good_q78, good_bank, sizeof(good_bank));

                    bool est = (mask & 1) != 0;
                    bool ddl = (mask & 2) != 0;
                    bool bnd = (mask & 4) != 0;
                    bool bus = (mask & 8) != 0;
                    int16_t sel[9] = {0};
                    bool commit = false;
                    auto obs_st = avid_fallback_step(&fsm, est, ddl, bnd, bus, cand_bank, sel, &commit);
                    auto ref = ref_fsm_transition(holds_in, fsm.max_holds_before_static, est, ddl, bnd, bus);

                    ASSERT_EQ(obs_st, ref.next_state);
                    ASSERT_EQ(commit, ref.should_commit);
                    const int16_t *exp_bank = (ref.bank_src == 0) ? cand_bank :
                                              (ref.bank_src == 1) ? good_bank : fsm.static_prior_q78;
                    ASSERT_EQ(std::memcmp(sel, exp_bank, sizeof(sel)), 0);
                    matches++;
                }
            }
        }
        ASSERT_EQ(matches, 128);
        ctx.record().expected["exact_symbolic_matches"] = 128;
        ctx.record().actual["exact_symbolic_matches"] = matches;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-055") {
        TestContext ctx(opts, tid, "fsm_unsafe_candidate_never_committed_invariant",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Prove across all states and fault combinations (!est || !ddl || !bnd || !bus) that unsafe candidate weights NEVER get selected";
        ctx.record().independent_oracle = "0 unsafe candidate leakages across all 60 faulted (state, input) pairs";

        const int16_t poison_cand[9] = {-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999};
        int leakages = 0;
        for (int st = 0; st < 4; ++st) {
            for (int mask = 0; mask < 15; ++mask) { /* mask 15 is the sole all-healthy input */
                avid_fallback_fsm_t fsm{};
                avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
                fsm.state = static_cast<avid_fallback_state_t>(st);
                std::memcpy(fsm.last_good_q78, good_bank, sizeof(good_bank));
                int16_t sel[9] = {0};
                bool commit = false;
                avid_fallback_step(&fsm,
                                   (mask & 1) != 0,
                                   (mask & 2) != 0,
                                   (mask & 4) != 0,
                                   (mask & 8) != 0,
                                   poison_cand, sel, &commit);
                for (int q = 0; q < 9; ++q) {
                    if (sel[q] == poison_cand[q]) leakages++;
                }
            }
        }
        ASSERT_EQ(leakages, 0);
        ctx.record().expected["unsafe_candidate_leakages"] = 0;
        ctx.record().actual["unsafe_candidate_leakages"] = leakages;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-056") {
        TestContext ctx(opts, tid, "fsm_fallback_bank_selection_invariants",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify SAFE_STATIC_PRIOR & DEGRADED_BYPASS output static_prior_q78, while HOLD_LAST_GOOD outputs last_good_q78 with should_commit==false";
        ctx.record().independent_oracle = "Exact bank equality in all 3 fallback states";

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        std::memcpy(fsm.last_good_q78, good_bank, sizeof(good_bank));
        int16_t sel[9] = {0};
        bool commit = true;

        /* HOLD_LAST_GOOD */
        ASSERT_EQ(avid_fallback_step(&fsm, false, true, true, true, cand_bank, sel, &commit), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_FALSE(commit);
        ASSERT_EQ(std::memcmp(sel, good_bank, sizeof(sel)), 0);

        /* SAFE_STATIC_PRIOR */
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, false, true, cand_bank, sel, &commit), AVID_FSM_SAFE_STATIC_PRIOR);
        ASSERT_TRUE(commit);
        ASSERT_EQ(std::memcmp(sel, fsm.static_prior_q78, sizeof(sel)), 0);

        /* DEGRADED_BYPASS */
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, true, false, cand_bank, sel, &commit), AVID_FSM_DEGRADED_BYPASS);
        ASSERT_FALSE(commit);
        ASSERT_EQ(std::memcmp(sel, fsm.static_prior_q78, sizeof(sel)), 0);

        ctx.record().expected["fallback_bank_invariants"] = "HOLD->last_good(no commit), STATIC->static_prior(commit), DEGRADED->static_prior(no commit)";
        ctx.record().actual["fallback_bank_invariants"] = "Verified";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-057") {
        TestContext ctx(opts, tid, "fsm_recovery_necessary_and_sufficient_condition",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Prove transition into NORMAL_ADAPTIVE occurs IFF (estimator_ok && deadline_ok && bounds_ok && bus_ok) == true";
        ctx.record().independent_oracle = "Across all 4 source states x 16 inputs, next_state == NORMAL_ADAPTIVE iff mask == 0xF (matching fault_policy.yaml 1-window recovery)";

        for (int st = 0; st < 4; ++st) {
            for (int mask = 0; mask < 16; ++mask) {
                avid_fallback_fsm_t fsm{};
                avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
                fsm.state = static_cast<avid_fallback_state_t>(st);
                fsm.consecutive_holds = 2u;
                int16_t sel[9] = {0};
                bool commit = false;
                auto next = avid_fallback_step(&fsm,
                                               (mask & 1) != 0,
                                               (mask & 2) != 0,
                                               (mask & 4) != 0,
                                               (mask & 8) != 0,
                                               cand_bank, sel, &commit);
                if (mask == 15) {
                    ASSERT_EQ(next, AVID_FSM_NORMAL_ADAPTIVE);
                    ASSERT_EQ(fsm.consecutive_holds, 0u);
                } else {
                    ASSERT_NE(next, AVID_FSM_NORMAL_ADAPTIVE);
                }
            }
        }
        ctx.record().expected["iff_condition"] = "next == NORMAL_ADAPTIVE <=> all 4 predicates true";
        ctx.record().actual["iff_condition"] = "Verified over 64 transitions";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-058") {
        TestContext ctx(opts, tid, "fsm_corrupt_enum_and_null_pointer_sanitization",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Inject corrupt raw enum value (99) via memcpy and null pointers into avid_fallback_step without UB or unvalidated commit";
        ctx.record().independent_oracle = "Corrupt state 99 -> SAFE_STATIC_PRIOR (or DEGRADED_BYPASS if !bus_ok); nullptr candidate/outputs handled safely";

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        unsigned int bad_raw = 99u;
        std::memcpy(&fsm.state, &bad_raw, sizeof(bad_raw));
        int16_t sel[9] = {0};
        bool commit = false;
        auto next1 = avid_fallback_step(&fsm, true, true, true, true, cand_bank, sel, &commit);
        ASSERT_EQ(next1, AVID_FSM_SAFE_STATIC_PRIOR);
        ASSERT_EQ(std::memcmp(sel, fsm.static_prior_q78, sizeof(sel)), 0);

        std::memcpy(&fsm.state, &bad_raw, sizeof(bad_raw));
        auto next2 = avid_fallback_step(&fsm, true, true, true, false, cand_bank, sel, &commit);
        ASSERT_EQ(next2, AVID_FSM_DEGRADED_BYPASS);
        ASSERT_FALSE(commit);

        /* Null candidate pointer */
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        auto next3 = avid_fallback_step(&fsm, true, true, true, true, nullptr, sel, &commit);
        ASSERT_EQ(next3, AVID_FSM_SAFE_STATIC_PRIOR);
        ASSERT_FALSE(commit);

        /* Null fsm pointer */
        ASSERT_EQ(avid_fallback_step(nullptr, true, true, true, true, cand_bank, sel, &commit), AVID_FSM_DEGRADED_BYPASS);
        ASSERT_FALSE(commit);

        ctx.record().expected["corrupt_and_null_guards"] = "Sanitized cleanly without UBSan trap";
        ctx.record().actual["corrupt_and_null_guards"] = "PASS";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-059") {
        TestContext ctx(opts, tid, "fsm_repeated_fault_escalation_and_stability",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Drive 2 holds -> 3rd fault escalates to SAFE_STATIC_PRIOR -> 50 additional faults stay strictly in SAFE_STATIC_PRIOR";
        ctx.record().independent_oracle = "holds 1,2 == HOLD_LAST_GOOD; faults 3..53 == SAFE_STATIC_PRIOR; fallback_triggers == 53";

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        int16_t sel[9] = {0};
        bool commit = false;

        ASSERT_EQ(avid_fallback_step(&fsm, false, true, true, true, cand_bank, sel, &commit), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_EQ(avid_fallback_step(&fsm, false, true, true, true, cand_bank, sel, &commit), AVID_FSM_HOLD_LAST_GOOD);
        for (int i = 3; i <= 53; ++i) {
            ASSERT_EQ(avid_fallback_step(&fsm, false, true, true, true, cand_bank, sel, &commit), AVID_FSM_SAFE_STATIC_PRIOR);
        }
        ASSERT_EQ(fsm.fallback_triggers, 53ULL);
        ctx.record().expected["fallback_triggers"] = 53;
        ctx.record().actual["fallback_triggers"] = fsm.fallback_triggers;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-060") {
        TestContext ctx(opts, tid, "fsm_reset_invariant_from_all_degraded_states",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_init", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Call avid_fallback_init from HOLD, SAFE_STATIC, and DEGRADED states and verify complete restoration";
        ctx.record().independent_oracle = "state==NORMAL_ADAPTIVE, consecutive_holds==0, fallback_triggers==0, banks==AVID_Q78_DEFAULT";

        for (int st = 1; st <= 3; ++st) {
            avid_fallback_fsm_t fsm{};
            fsm.state = static_cast<avid_fallback_state_t>(st);
            fsm.consecutive_holds = 17u;
            fsm.fallback_triggers = 99ULL;
            avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
            ASSERT_EQ(fsm.state, AVID_FSM_NORMAL_ADAPTIVE);
            ASSERT_EQ(fsm.consecutive_holds, 0u);
            ASSERT_EQ(fsm.fallback_triggers, 0ULL);
            for (int q = 0; q < 9; ++q) {
                ASSERT_EQ(fsm.last_good_q78[q], AVID_Q78_DEFAULT);
                ASSERT_EQ(fsm.static_prior_q78[q], AVID_Q78_DEFAULT);
            }
        }
        ctx.record().expected["reset_invariant"] = "Clean restoration from all 3 degraded states";
        ctx.record().actual["reset_invariant"] = "Verified";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-061") {
        TestContext ctx(opts, tid, "fsm_complete_legal_edge_witness_coverage",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Construct witness executions visiting every reachable directed edge (s -> s') in the 4-state safety FSM";
        ctx.record().independent_oracle = "All 16 reachable (s, s') directed edges witnessed (100% legal edge coverage)";

        std::set<std::pair<int, int>> reachable_spec, observed;
        for (int st = 0; st < 4; ++st) {
            for (uint32_t holds : {0u, 1u, 2u, 3u}) {
                for (int mask = 0; mask < 16; ++mask) {
                    auto r = ref_fsm_transition(holds, 3u, (mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0, (mask & 8) != 0);
                    reachable_spec.insert({st, static_cast<int>(r.next_state)});

                    avid_fallback_fsm_t fsm{};
                    avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
                    fsm.state = static_cast<avid_fallback_state_t>(st);
                    fsm.consecutive_holds = holds;
                    int16_t sel[9] = {0};
                    bool commit = false;
                    auto nx = avid_fallback_step(&fsm, (mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0, (mask & 8) != 0, cand_bank, sel, &commit);
                    observed.insert({st, static_cast<int>(nx)});
                }
            }
        }
        ASSERT_EQ(observed.size(), reachable_spec.size());
        ASSERT_EQ(observed.size(), static_cast<size_t>(16));
        ctx.record().expected["legal_directed_edges"] = 16;
        ctx.record().actual["witnessed_directed_edges"] = observed.size();
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-062") {
        TestContext ctx(opts, tid, "fsm_forbidden_transition_impossibility_proof",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Exhaustively prove forbidden condition-transition pairs never occur (!bus_ok -> non-DEGRADED, !bounds_ok -> HOLD/NORMAL, HOLD -> commit==true)";
        ctx.record().independent_oracle = "0 forbidden transition occurrences across all states, hold counts 0..5, and 16 input tuples";

        int forbidden = 0;
        for (int st = 0; st < 4; ++st) {
            for (uint32_t holds = 0; holds <= 5; ++holds) {
                for (int mask = 0; mask < 16; ++mask) {
                    bool est = (mask & 1) != 0;
                    bool ddl = (mask & 2) != 0;
                    bool bnd = (mask & 4) != 0;
                    bool bus = (mask & 8) != 0;
                    avid_fallback_fsm_t fsm{};
                    avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
                    fsm.state = static_cast<avid_fallback_state_t>(st);
                    fsm.consecutive_holds = holds;
                    int16_t sel[9] = {0};
                    bool commit = false;
                    auto nx = avid_fallback_step(&fsm, est, ddl, bnd, bus, cand_bank, sel, &commit);
                    if (!bus && (nx != AVID_FSM_DEGRADED_BYPASS || commit)) forbidden++;
                    if (bus && !bnd && (nx == AVID_FSM_HOLD_LAST_GOOD || nx == AVID_FSM_NORMAL_ADAPTIVE)) forbidden++;
                    if (nx == AVID_FSM_HOLD_LAST_GOOD && commit) forbidden++;
                    if (holds >= 2u && !(est && ddl && bnd && bus) && nx == AVID_FSM_HOLD_LAST_GOOD) forbidden++;
                }
            }
        }
        ASSERT_EQ(forbidden, 0);
        ctx.record().expected["forbidden_transitions"] = 0;
        ctx.record().actual["forbidden_transitions"] = forbidden;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-MATH-053", "L3-MATH-054", "L3-MATH-055", "L3-MATH-056", "L3-MATH-057",
        "L3-MATH-058", "L3-MATH-059", "L3-MATH-060", "L3-MATH-061", "L3-MATH-062"
    };
    if (opts.test_id == "ALL") {
        int rc = 0;
        for (const char *id : all_ids) {
            if (run_one(opts, id) != 0) rc = 1;
        }
        return rc;
    }
    return run_one(opts, opts.test_id);
}
