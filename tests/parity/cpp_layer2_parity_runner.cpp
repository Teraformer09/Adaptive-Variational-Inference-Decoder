#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "deployment/cpp/observation_adapter.h"
#include "deployment/cpp/r2d2_runtime.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "Usage: %s [--v1 | --expect-fail-v2 | --expect-fail-v1 | --bench-latency] <checkpoint.bin> <in.csv> <out.csv>\n",
            argv[0]);
        return 2;
    }

    int arg_idx = 1;
    bool use_v1 = false;
    bool expect_fail_v2 = false;
    bool expect_fail_v1 = false;
    bool bench_latency = false;

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (std::strcmp(argv[arg_idx], "--v1") == 0) {
            use_v1 = true;
        } else if (std::strcmp(argv[arg_idx], "--expect-fail-v2") == 0) {
            expect_fail_v2 = true;
        } else if (std::strcmp(argv[arg_idx], "--expect-fail-v1") == 0) {
            expect_fail_v1 = true;
        } else if (std::strcmp(argv[arg_idx], "--bench-latency") == 0) {
            bench_latency = true;
        }
        ++arg_idx;
    }

    if (argc - arg_idx < 1) {
        return 2;
    }
    const std::string ckpt_path = argv[arg_idx];

    if (expect_fail_v2) {
        avid::deployment::R2D2Runtime rt;
        std::string err;
        if (rt.load_checkpoint(ckpt_path, &err)) {
            std::fprintf(stderr, "ERROR: V2 runtime unexpectedly accepted incompatible checkpoint!\n");
            return 1;
        }
        std::printf("EXPECTED_REJECT_V2: %s\n", err.c_str());
        return 0;
    }

    if (expect_fail_v1) {
        avid::deployment::R2D2RuntimeV1 rt1;
        std::string err;
        if (rt1.load_checkpoint(ckpt_path, &err)) {
            std::fprintf(stderr, "ERROR: V1 runtime unexpectedly accepted incompatible checkpoint!\n");
            return 1;
        }
        std::printf("EXPECTED_REJECT_V1: %s\n", err.c_str());
        return 0;
    }

    if (argc - arg_idx < 3) {
        return 2;
    }
    const std::string in_path = argv[arg_idx + 1];
    const std::string out_path = argv[arg_idx + 2];

    std::ifstream fin(in_path);
    std::ofstream fout(out_path);
    std::string line;

    if (use_v1) {
        avid::deployment::R2D2RuntimeV1 rt1;
        std::string err;
        if (!rt1.load_checkpoint(ckpt_path, &err)) {
            std::fprintf(stderr, "V1 load error: %s\n", err.c_str());
            return 1;
        }
        rt1.reset_state();
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            std::vector<double> obs;
            std::stringstream ss(line);
            std::string item;
            while (std::getline(ss, item, ',')) {
                obs.push_back(std::strtod(item.c_str(), nullptr));
            }
            if (obs.size() != avid::deployment::R2D2_V1_INPUT_DIM) {
                std::fprintf(stderr, "Expected 8 features per row for V1, got %zu\n", obs.size());
                return 3;
            }
            auto out = rt1.step(obs.data());
            fout << out.action_id << "," << out.state_value << ","
                 << out.pair_weight_scale << "," << out.bp_iterations;
            for (int a = 0; a < 6; ++a) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), ",%.17g", out.q_values[a]);
                fout << buf;
            }
            for (int q = 0; q < 9; ++q) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), ",%.17g", out.qubit_gains[q]);
                fout << buf;
            }
            fout << "\n";
        }
        return 0;
    }

    avid::deployment::R2D2Runtime rt2;
    std::string err;
    if (!rt2.load_checkpoint(ckpt_path, &err)) {
        std::fprintf(stderr, "V2 load error: %s\n", err.c_str());
        return 1;
    }
    rt2.reset_state();

    std::vector<std::vector<double>> rows;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::vector<double> obs;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            obs.push_back(std::strtod(item.c_str(), nullptr));
        }
        if (obs.size() != avid::deployment::R2D2_INPUT_DIM) {
            std::fprintf(stderr, "Expected 82 features per row for V2, got %zu\n", obs.size());
            return 3;
        }
        std::string val_err;
        if (!avid::deployment::validate_observation_v2(obs.data(), &val_err)) {
            std::fprintf(stderr, "Observation validation failed: %s\n", val_err.c_str());
            return 4;
        }
        rows.push_back(obs);
    }

    for (const auto& obs : rows) {
        auto out = rt2.step(obs.data());
        fout << out.action_id << "," << out.state_value << ","
             << out.pair_weight_scale << "," << out.bp_iterations;
        for (int a = 0; a < 6; ++a) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), ",%.17g", out.q_values[a]);
            fout << buf;
        }
        for (int q = 0; q < 9; ++q) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), ",%.17g", out.qubit_gains[q]);
            fout << buf;
        }
        fout << "\n";
    }

    if (bench_latency && !rows.empty()) {
        rt2.reset_state();
        constexpr int kIters = 5000;
        volatile double sink = 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIters; ++i) {
            const auto out = rt2.step(rows[i % rows.size()].data());
            sink += out.state_value;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double total_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
        const double mean_us = total_us / static_cast<double>(kIters);
        std::printf(
            "LATENCY_BENCH_V2: steps=%d mean_step_us=%.4f weights_bytes=%zu sink=%.3f\n",
            kIters,
            mean_us,
            sizeof(avid::deployment::R2D2ModelWeights),
            sink);
    }

    return 0;
}
