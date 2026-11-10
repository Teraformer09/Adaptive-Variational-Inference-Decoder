#include "deployment/cpp/model_loader.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <vector>

namespace avid {
namespace deployment {

namespace {

bool check_all_finite(const double* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) {
            return false;
        }
    }
    return true;
}

void append_exe_relative_candidates(std::vector<std::string>& out_paths) {
    char exe_buf[4096];
    const ssize_t len = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    if (len <= 0) {
        return;
    }
    exe_buf[len] = '\0';
    std::string dir(exe_buf);
    for (int depth = 0; depth < 6; ++depth) {
        const size_t slash = dir.find_last_of('/');
        if (slash == std::string::npos || slash == 0) {
            break;
        }
        dir.resize(slash);
        out_paths.push_back(dir + "/checkpoints/r2d2_v1_trained.bin");
    }
}

}  // namespace

bool load_r2d2_binary_checkpoint(
    const std::string& path,
    R2D2ModelWeights& out_weights,
    std::string* err_msg) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        if (err_msg) *err_msg = "Failed to open checkpoint file: " + path;
        return false;
    }

    const std::streamsize file_size = ifs.tellg();
    constexpr size_t kExpectedFloats =
        R2D2_INPUT_DIM +
        R2D2_INPUT_DIM +
        (4 * R2D2_HIDDEN_DIM * R2D2_INPUT_DIM) +
        (4 * R2D2_HIDDEN_DIM * R2D2_HIDDEN_DIM) +
        (4 * R2D2_HIDDEN_DIM) +
        R2D2_HIDDEN_DIM +
        1 +
        (R2D2_NUM_ACTIONS * R2D2_HIDDEN_DIM) +
        R2D2_NUM_ACTIONS +
        (R2D2_NUM_QUBITS * R2D2_HIDDEN_DIM) +
        R2D2_NUM_QUBITS;
    constexpr std::streamsize kExpectedBytes =
        static_cast<std::streamsize>(sizeof(R2D2BinaryHeader) + kExpectedFloats * sizeof(double));

    if (file_size != kExpectedBytes) {
        if (err_msg) {
            *err_msg = "Checkpoint size mismatch in " + path + ": expected " +
                       std::to_string(kExpectedBytes) + " bytes, got " +
                       std::to_string(file_size);
        }
        return false;
    }

    ifs.seekg(0, std::ios::beg);
    R2D2BinaryHeader hdr{};
    if (!ifs.read(reinterpret_cast<char*>(&hdr), sizeof(R2D2BinaryHeader))) {
        if (err_msg) *err_msg = "Failed to read header from " + path;
        return false;
    }

    if (std::memcmp(hdr.magic, "AVIDR2D2", 8) != 0) {
        if (err_msg) *err_msg = "Invalid checkpoint magic in " + path;
        return false;
    }
    if (hdr.version != 1) {
        if (err_msg) *err_msg = "Unsupported checkpoint version: " + std::to_string(hdr.version);
        return false;
    }
    if (hdr.input_dim != R2D2_INPUT_DIM ||
        hdr.hidden_dim != R2D2_HIDDEN_DIM ||
        hdr.num_actions != R2D2_NUM_ACTIONS ||
        hdr.num_qubits != R2D2_NUM_QUBITS) {
        if (err_msg) *err_msg = "Architecture dimension mismatch in " + path;
        return false;
    }

    auto read_floats = [&](double* dst, size_t count) -> bool {
        if (!ifs.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(count * sizeof(double)))) {
            return false;
        }
        return check_all_finite(dst, count);
    };

    if (!read_floats(out_weights.feature_scale, R2D2_INPUT_DIM) ||
        !read_floats(out_weights.feature_offset, R2D2_INPUT_DIM) ||
        !read_floats(&out_weights.W_ih[0][0], 4 * R2D2_HIDDEN_DIM * R2D2_INPUT_DIM) ||
        !read_floats(&out_weights.W_hh[0][0], 4 * R2D2_HIDDEN_DIM * R2D2_HIDDEN_DIM) ||
        !read_floats(out_weights.b_ih, 4 * R2D2_HIDDEN_DIM) ||
        !read_floats(out_weights.W_val, R2D2_HIDDEN_DIM) ||
        !read_floats(&out_weights.b_val, 1) ||
        !read_floats(&out_weights.W_adv[0][0], R2D2_NUM_ACTIONS * R2D2_HIDDEN_DIM) ||
        !read_floats(out_weights.b_adv, R2D2_NUM_ACTIONS) ||
        !read_floats(&out_weights.W_gain[0][0], R2D2_NUM_QUBITS * R2D2_HIDDEN_DIM) ||
        !read_floats(out_weights.b_gain, R2D2_NUM_QUBITS)) {
        if (err_msg) *err_msg = "Corrupted or non-finite weight tensor in " + path;
        return false;
    }

    out_weights.loaded_from_checkpoint = true;
    out_weights.checkpoint_path = path;
    return true;
}

bool load_default_r2d2_checkpoint(
    R2D2ModelWeights& out_weights,
    std::string* err_msg) {
    std::vector<std::string> candidate_paths;
    if (const char* env_path = std::getenv("AVID_CHECKPOINT_PATH")) {
        if (env_path[0] != '\0') {
            candidate_paths.emplace_back(env_path);
        }
    }
    candidate_paths.emplace_back("checkpoints/r2d2_v1_trained.bin");
    candidate_paths.emplace_back("../checkpoints/r2d2_v1_trained.bin");
    candidate_paths.emplace_back("../../checkpoints/r2d2_v1_trained.bin");
    candidate_paths.emplace_back("../../../checkpoints/r2d2_v1_trained.bin");
    append_exe_relative_candidates(candidate_paths);

    std::string last_err;
    for (const auto& candidate : candidate_paths) {
        std::ifstream probe(candidate, std::ios::binary);
        if (probe.good()) {
            probe.close();
            if (load_r2d2_binary_checkpoint(candidate, out_weights, &last_err)) {
                return true;
            }
        }
    }
    if (err_msg) {
        *err_msg = last_err.empty()
                       ? "Could not locate checkpoints/r2d2_v1_trained.bin"
                       : last_err;
    }
    return false;
}

}  // namespace deployment
}  // namespace avid
