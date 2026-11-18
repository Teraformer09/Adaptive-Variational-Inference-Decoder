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

void append_exe_relative_candidates(
    std::vector<std::string>& out_paths, const std::string& rel_suffix) {
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
        out_paths.push_back(dir + "/" + rel_suffix);
    }
}

}  // namespace

bool load_r2d2_binary_checkpoint(
    const std::string& path,
    R2D2ModelWeights& out_weights,
    std::string* err_msg) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        if (err_msg) *err_msg = "Failed to open V2 checkpoint file: " + path;
        return false;
    }

    const std::streamsize file_size = ifs.tellg();
    if (file_size < static_cast<std::streamsize>(sizeof(R2D2BinaryHeader))) {
        if (err_msg) *err_msg = "Checkpoint file smaller than 32-byte header: " + path;
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
    if (hdr.version != R2D2_SCHEMA_VERSION_V2) {
        if (err_msg) {
            *err_msg = "Incompatible checkpoint schema version in " + path +
                       ": V2 runtime requires version 2, got version " +
                       std::to_string(hdr.version);
        }
        return false;
    }
    if (hdr.input_dim != R2D2_INPUT_DIM ||
        hdr.hidden_dim != R2D2_HIDDEN_DIM ||
        hdr.num_actions != R2D2_NUM_ACTIONS ||
        hdr.num_qubits != R2D2_NUM_QUBITS) {
        if (err_msg) {
            *err_msg = "V2 architecture dimension mismatch in " + path +
                       " (expected input_dim=82, hidden_dim=64, actions=6, qubits=9)";
        }
        return false;
    }

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
            *err_msg = "V2 checkpoint size mismatch in " + path + ": expected " +
                       std::to_string(kExpectedBytes) + " bytes, got " +
                       std::to_string(file_size);
        }
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

    out_weights.schema_version = hdr.version;
    out_weights.loaded_from_checkpoint = true;
    out_weights.checkpoint_path = path;
    return true;
}

bool load_r2d2_v1_binary_checkpoint(
    const std::string& path,
    R2D2ModelWeightsV1& out_weights,
    std::string* err_msg) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        if (err_msg) *err_msg = "Failed to open V1 checkpoint file: " + path;
        return false;
    }

    const std::streamsize file_size = ifs.tellg();
    if (file_size < static_cast<std::streamsize>(sizeof(R2D2BinaryHeader))) {
        if (err_msg) *err_msg = "Checkpoint file smaller than 32-byte header: " + path;
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
    if (hdr.version != R2D2_SCHEMA_VERSION_V1) {
        if (err_msg) {
            *err_msg = "Incompatible checkpoint schema version in " + path +
                       ": V1 loader requires version 1, got version " +
                       std::to_string(hdr.version);
        }
        return false;
    }
    if (hdr.input_dim != R2D2_V1_INPUT_DIM ||
        hdr.hidden_dim != R2D2_HIDDEN_DIM ||
        hdr.num_actions != R2D2_NUM_ACTIONS ||
        hdr.num_qubits != R2D2_NUM_QUBITS) {
        if (err_msg) *err_msg = "V1 architecture dimension mismatch in " + path;
        return false;
    }

    constexpr size_t kExpectedFloatsV1 =
        R2D2_V1_INPUT_DIM +
        R2D2_V1_INPUT_DIM +
        (4 * R2D2_HIDDEN_DIM * R2D2_V1_INPUT_DIM) +
        (4 * R2D2_HIDDEN_DIM * R2D2_HIDDEN_DIM) +
        (4 * R2D2_HIDDEN_DIM) +
        R2D2_HIDDEN_DIM +
        1 +
        (R2D2_NUM_ACTIONS * R2D2_HIDDEN_DIM) +
        R2D2_NUM_ACTIONS +
        (R2D2_NUM_QUBITS * R2D2_HIDDEN_DIM) +
        R2D2_NUM_QUBITS;
    constexpr std::streamsize kExpectedBytesV1 =
        static_cast<std::streamsize>(sizeof(R2D2BinaryHeader) + kExpectedFloatsV1 * sizeof(double));

    if (file_size != kExpectedBytesV1) {
        if (err_msg) *err_msg = "V1 checkpoint size mismatch in " + path;
        return false;
    }

    auto read_floats = [&](double* dst, size_t count) -> bool {
        if (!ifs.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(count * sizeof(double)))) {
            return false;
        }
        return check_all_finite(dst, count);
    };

    if (!read_floats(out_weights.feature_scale, R2D2_V1_INPUT_DIM) ||
        !read_floats(out_weights.feature_offset, R2D2_V1_INPUT_DIM) ||
        !read_floats(&out_weights.W_ih[0][0], 4 * R2D2_HIDDEN_DIM * R2D2_V1_INPUT_DIM) ||
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

    out_weights.schema_version = hdr.version;
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
    candidate_paths.emplace_back("checkpoints/v2/r2d2_v2_trained.bin");
    candidate_paths.emplace_back("../checkpoints/v2/r2d2_v2_trained.bin");
    candidate_paths.emplace_back("../../checkpoints/v2/r2d2_v2_trained.bin");
    append_exe_relative_candidates(candidate_paths, "checkpoints/v2/r2d2_v2_trained.bin");

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
                       ? "Could not locate checkpoints/v2/r2d2_v2_trained.bin"
                       : last_err;
    }
    return false;
}

}  // namespace deployment
}  // namespace avid
