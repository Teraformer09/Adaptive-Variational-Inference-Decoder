#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "deployment/cpp/r2d2_runtime.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s <checkpoint.bin> <in.csv> <out.csv>\n", argv[0]);
        return 2;
    }
    avid::deployment::R2D2Runtime rt;
    std::string err;
    if (!rt.load_checkpoint(argv[1], &err)) {
        std::fprintf(stderr, "Load error: %s\n", err.c_str());
        return 1;
    }
    rt.reset_state();

    std::ifstream fin(argv[2]);
    std::ofstream fout(argv[3]);
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::vector<double> obs;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            obs.push_back(std::strtod(item.c_str(), nullptr));
        }
        if (obs.size() != 8) {
            std::fprintf(stderr, "Expected 8 features per row, got %zu\n", obs.size());
            return 3;
        }
        auto out = rt.step(obs.data());
        fout << out.action_id << "," << out.state_value << ","
             << out.pair_weight_scale << "," << out.bp_iterations;
        for (int a = 0; a < 6; a++) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), ",%.17g", out.q_values[a]);
            fout << buf;
        }
        for (int q = 0; q < 9; q++) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), ",%.17g", out.qubit_gains[q]);
            fout << buf;
        }
        fout << "\n";
    }
    return 0;
}
