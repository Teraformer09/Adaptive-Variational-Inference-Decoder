#include "test_assertions.h"
#include "test_common.h"
#include "inference/noise_model.h"
#include "estimator/lbfgs_solver.h"
#include "estimator/llr_quantizer.h"
#include <algorithm>
#include <cmath>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-FLOW-006");
    TestContext ctx(opts, "L3-FLOW-006", "noise_estimate_to_q78_llr_quantizer_continuity",
                    "software/src/inference/noise_model.cpp, software/src/estimator/llr_quantizer.cpp",
                    "NoiseModel::effective_p -> avid_quantize_and_validate_bank", "set3");
    ctx.record().test_set = "set3_pipeline_flow";
    ctx.record().test_description = "Pipe live fitted theta_ through effective_p(0..8) and avid_quantize_and_validate_bank, comparing against independent log((1-p)/p)*256";
    ctx.record().independent_oracle = "0 LSB difference on unprojected Q7.8 bank and bounded safe range [0x0040, 0x3000] on pair-projected bank";

    uint32_t hx[16] = {790, 14, 24, 12, 20, 10, 82, 5, 16, 4, 10, 4, 4, 2, 2, 1};
    NoiseModel m;
    m.init(0.012, 0.008);
    HistogramLBFGSSolver solver;
    solver.fit_histogram(m, hx, 25);

    double p_eff[9];
    for (int q = 0; q < 9; ++q) p_eff[q] = m.effective_p(q);

    int16_t cand_q78[9] = {0};
    ASSERT_TRUE(avid_quantize_and_validate_bank(p_eff, cand_q78));

    for (int q = 0; q < 9; ++q) {
        double pc = std::clamp(p_eff[q], 1e-6, 0.49);
        long ref = std::lround(std::log((1.0 - pc) / pc) * 256.0);
        int16_t ref_q78 = static_cast<int16_t>(std::clamp<long>(ref, AVID_Q78_MIN_SAFE, AVID_Q78_MAX_SAFE));
        ASSERT_EQ(cand_q78[q], ref_q78);
    }

    int16_t proj_q78[9] = {0};
    m.to_pair_projected_q7_8_llr(proj_q78, nullptr, 1.0);
    for (int q = 0; q < 9; ++q) {
        ASSERT_TRUE(proj_q78[q] >= AVID_Q78_MIN_SAFE && proj_q78[q] <= AVID_Q78_MAX_SAFE);
    }
    ASSERT_TRUE(proj_q78[4] < proj_q78[0]); /* hotspot Z_4 has lower LLR than quiet Z_0 */

    ctx.record().expected["unprojected_lsb_diff"] = "0";
    ctx.record().actual["hotspot_q4_q78"] = std::to_string(proj_q78[4]);
    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
