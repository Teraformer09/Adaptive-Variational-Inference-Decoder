#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "inference/noise_model.h"
#include <string>
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-021");
    TestContext ctx(opts, opts.test_id, "parameter_vector_dimension_21",
                    "software/src/inference/noise_model.h",
                    "NoiseModel::ParamVec, NM_N_QUBIT, NM_N_CROSSTALK, NM_N_PARAMS");

    ctx.record().test_description =
        "21-parameter vector dimension check — verify NM_N_QUBIT==9, NM_N_CROSSTALK==12, NM_N_PARAMS==21, sizeof(ParamVec)==21*sizeof(double), and agreement with architecture/estimator/parameter_schema.json";
    ctx.record().input_contract = "Fixed 21-element parameter vector (9 single-qubit + 12 nearest-neighbor pairs)";
    ctx.record().independent_oracle = "architecture/estimator/parameter_schema.json (n_params_full=21, n_identifiable_signature_orbits=9)";
    ctx.record().fixture = "tests/layer3/fixtures/parameters/valid/default_21.txt";

    ASSERT_EQ(NM_N_QUBIT, 9);
    ASSERT_EQ(NM_N_CROSSTALK, 12);
    ASSERT_EQ(NM_N_PARAMS, 21);
    ASSERT_EQ(NM_N_STATES, 16);
    ASSERT_EQ(NM_N_ORBITS, 9);
    ASSERT_EQ(sizeof(NoiseModel::ParamVec), static_cast<size_t>(21 * sizeof(double)));
    ASSERT_EQ(sizeof(NoiseModel::GradVec), static_cast<size_t>(21 * sizeof(double)));
    ASSERT_EQ(sizeof(NoiseModel::StateVec), static_cast<size_t>(16 * sizeof(double)));

    std::vector<double> fx_vec;
    ASSERT_TRUE(load_f64_vector_fixture(ctx.fixture_path("parameters/valid/default_21.txt"), fx_vec));
    ASSERT_EQ(fx_vec.size(), static_cast<size_t>(21));

    std::string schema_txt;
    std::string schema_path = opts.repo_root + "/architecture/estimator/parameter_schema.json";
    ASSERT_TRUE(load_text_fixture(schema_path, schema_txt));
    ASSERT_TRUE(schema_txt.find("\"n_params_full\": 21") != std::string::npos);
    ASSERT_TRUE(schema_txt.find("\"n_identifiable_signature_orbits\": 9") != std::string::npos);

    NoiseModel model;
    model.init(0.012, 0.004);
    for (int i = 0; i < 9; ++i) {
        ASSERT_NEAR(model.theta_(i), 0.012, 1e-12);
    }
    for (int i = 9; i < 21; ++i) {
        ASSERT_NEAR(model.theta_(i), 0.004, 1e-12);
    }

    ctx.log("Note: NoiseModel::ParamVec enforces dimension 21 at compile time via Eigen::Matrix<double, 21, 1>; runtime array loader for arbitrary-length double* does not exist separately from checkpoint/CLI initialization.");
    ctx.record().expected["dimensions"] = "NM_N_QUBIT=9, NM_N_CROSSTALK=12, NM_N_PARAMS=21, sizeof(ParamVec)=168";
    ctx.record().actual["dimensions"] =
        "NM_N_QUBIT=" + std::to_string(NM_N_QUBIT) + ", NM_N_CROSSTALK=" + std::to_string(NM_N_CROSSTALK) +
        ", NM_N_PARAMS=" + std::to_string(NM_N_PARAMS) + ", sizeof(ParamVec)=" + std::to_string(sizeof(NoiseModel::ParamVec));

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
