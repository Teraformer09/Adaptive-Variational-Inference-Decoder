#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include <string>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-036");
    TestContext ctx(opts, "L3-IP-036", "unknown_config_fields_and_schema_enforcement",
                    "software/src/inference/avid_daemon.cpp, configs/schema.yaml",
                    "avid_daemon CLI parser & YAML schema contract");

    ctx.record().test_description =
        "Unknown configuration keys — verify configs/schema.yaml requires immediate failure on unknown keys, and test avid_daemon with unknown CLI flags (--unknown-key, --config)";
    ctx.record().input_contract = "configs/schema.yaml: 'Unknown top-level or nested keys trigger an immediate validation failure'";
    ctx.record().independent_oracle = "Non-zero exit code on unknown CLI flags and invalid configuration values";
    ctx.record().fixture = "tests/layer3/fixtures/configs/invalid/unknown_key_experiment.yaml";

    std::string daemon = opts.daemon_bin;
    ASSERT_TRUE(!daemon.empty());

    /* Known invalid values explicitly checked by avid_daemon */
    auto r_inv_scen = run_command_capture("\"" + daemon + "\" --scenario invalid_configuration --controller-mode static --rounds 64 --window 64");
    ASSERT_EQ(r_inv_scen.exit_code, 2);

    auto r_bad_osd = run_command_capture("\"" + daemon + "\" --osd-mode 9 --controller-mode static --rounds 64 --window 64");
    ASSERT_EQ(r_bad_osd.exit_code, 2);

    /* Unknown CLI flag `--unknown-key foo`, invalid YAMLs, and valid YAML (`valid_experiment.yaml`) */
    auto r_unk_flag = run_command_capture("\"" + daemon + "\" --controller-mode static --rounds 64 --window 64 --unknown-key rogue_value");
    std::string bad_yaml = ctx.fixture_path("configs/invalid/unknown_key_experiment.yaml");
    std::string missing_yaml = ctx.fixture_path("configs/invalid/missing_required.yaml");
    std::string valid_yaml = ctx.fixture_path("configs/valid/valid_experiment.yaml");

    auto r_yaml_flag = run_command_capture("\"" + daemon + "\" --controller-mode static --rounds 64 --window 64 --config \"" + bad_yaml + "\"");
    auto r_missing_yaml = run_command_capture("\"" + daemon + "\" --config \"" + missing_yaml + "\"");
    auto r_valid_yaml = run_command_capture("\"" + daemon + "\" --config \"" + valid_yaml + "\" --rounds 64 --window 64");
    auto r_bad_ctrl = run_command_capture("\"" + daemon + "\" --controller-mode nonexistent_controller --rounds 64 --window 64");

    ASSERT_EQ(r_unk_flag.exit_code, 2);
    ASSERT_EQ(r_yaml_flag.exit_code, 2);
    ASSERT_EQ(r_missing_yaml.exit_code, 2);
    ASSERT_EQ(r_valid_yaml.exit_code, 0);
    ASSERT_EQ(r_bad_ctrl.exit_code, 2);

    ctx.record().expected["unknown_field_rejection"] = "Immediate exit code 2 on unknown CLI flags, unknown controller_mode, and unknown/missing YAML keys; exit code 0 on valid_experiment.yaml";
    ctx.record().actual["unknown_field_rejection"] =
        "invalid_configuration rc=" + std::to_string(r_inv_scen.exit_code) +
        ", osd_mode=9 rc=" + std::to_string(r_bad_osd.exit_code) +
        ", --unknown-key rc=" + std::to_string(r_unk_flag.exit_code) +
        ", --config unknown_key.yaml rc=" + std::to_string(r_yaml_flag.exit_code) +
        ", --config missing_required.yaml rc=" + std::to_string(r_missing_yaml.exit_code) +
        ", --config valid_experiment.yaml rc=" + std::to_string(r_valid_yaml.exit_code) +
        ", bad controller_mode rc=" + std::to_string(r_bad_ctrl.exit_code);

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
