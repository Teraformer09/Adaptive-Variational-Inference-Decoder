#include "test_assertions.h"
#include "test_common.h"
#include "inference/noise_model.h"
#include <cmath>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-022");
    TestContext ctx(opts, opts.test_id, "parameter_ordering_and_signatures",
                    "software/src/inference/noise_model.cpp",
                    "NM_CROSSTALK_PAIRS, NM_EVENT_SIGNATURES, NM_EVENT_SIGNATURES_Z_BASIS");

    ctx.record().test_description =
        "Parameter ordering — verify exact 1-to-1 ordering of 9 single-qubit and 12 crosstalk pair parameters, stabilizer signatures, and permutation sensitivity";
    ctx.record().input_contract = "theta_[0..8]=p_0..p_8, theta_[9..20]=12 grid pairs matching parameter_schema.json";
    ctx.record().independent_oracle = "Rotated d=3 stabilizer incidence X={01,1245,3467,78}, Z={0134,25,36,4578} & parameter_schema.json";
    ctx.record().fixture = "architecture/estimator/parameter_schema.json";

    const int expected_pairs[12][2] = {
        {0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7}, {7, 8},
        {0, 3}, {3, 6}, {1, 4}, {4, 7}, {2, 5}, {5, 8}
    };
    for (int k = 0; k < 12; ++k) {
        ASSERT_EQ(NM_CROSSTALK_PAIRS[k].i, expected_pairs[k][0]);
        ASSERT_EQ(NM_CROSSTALK_PAIRS[k].j, expected_pairs[k][1]);
    }

    /* Independent stabilizer incidence oracle for X-checks (detecting Z faults) */
    const uint8_t oracle_x_single[9] = {1, 3, 2, 4, 6, 2, 4, 12, 8};
    for (int q = 0; q < 9; ++q) {
        ASSERT_EQ(static_cast<int>(NM_EVENT_SIGNATURES[q]), static_cast<int>(oracle_x_single[q]));
    }
    for (int k = 0; k < 12; ++k) {
        int u = expected_pairs[k][0];
        int v = expected_pairs[k][1];
        uint8_t expected_pair_sig = static_cast<uint8_t>(oracle_x_single[u] ^ oracle_x_single[v]);
        ASSERT_EQ(static_cast<int>(NM_EVENT_SIGNATURES[9 + k]), static_cast<int>(expected_pair_sig));
    }

    /* Independent stabilizer incidence oracle for Z-checks (detecting X faults):
     * z0={0,1,3,4}->bit0, z1={2,5}->bit1, z2={3,6}->bit2, z3={4,5,7,8}->bit3 */
    const uint8_t oracle_z_single[9] = {1, 1, 2, 5, 9, 10, 4, 8, 8};
    for (int q = 0; q < 9; ++q) {
        ASSERT_EQ(static_cast<int>(NM_EVENT_SIGNATURES_Z_BASIS[q]), static_cast<int>(oracle_z_single[q]));
    }
    for (int k = 0; k < 12; ++k) {
        int u = expected_pairs[k][0];
        int v = expected_pairs[k][1];
        uint8_t expected_pair_sig_z = static_cast<uint8_t>(oracle_z_single[u] ^ oracle_z_single[v]);
        ASSERT_EQ(static_cast<int>(NM_EVENT_SIGNATURES_Z_BASIS[9 + k]), static_cast<int>(expected_pair_sig_z));
    }

    /* Verify permutation sensitivity: swapping theta_(1) [sig=3] with theta_(4) [sig=6] alters p_joint_ */
    NoiseModel m_a, m_b;
    m_a.init(0.01, 0.002);
    m_b.init(0.01, 0.002);
    m_a.theta_(1) = 0.18;
    m_b.theta_(4) = 0.18;
    m_a.compute_joint_distribution();
    m_b.compute_joint_distribution();

    double diff_state3 = std::abs(m_a.joint_probs()(3) - m_b.joint_probs()(3));
    double diff_state6 = std::abs(m_a.joint_probs()(6) - m_b.joint_probs()(6));
    ASSERT_TRUE(diff_state3 > 0.10);
    ASSERT_TRUE(diff_state6 > 0.10);

    ctx.record().expected["ordering_match"] = "21/21 parameters and X/Z signatures match oracle; non-aliased permutations change p_joint_";
    ctx.record().actual["ordering_match"] =
        "21/21 matched; |dp_joint[3]|=" + std::to_string(diff_state3) + ", |dp_joint[6]|=" + std::to_string(diff_state6);

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
