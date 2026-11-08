#ifndef AVID_DEPLOYMENT_CPP_MODEL_LOADER_H
#define AVID_DEPLOYMENT_CPP_MODEL_LOADER_H

#include <cstdint>
#include <string>

namespace avid {
namespace deployment {

constexpr int R2D2_INPUT_DIM = 8;
constexpr int R2D2_HIDDEN_DIM = 64;
constexpr int R2D2_NUM_ACTIONS = 6;
constexpr int R2D2_NUM_QUBITS = 9;
constexpr double R2D2_GAIN_CLIP = 0.30;

struct R2D2BinaryHeader {
    char magic[8];
    uint32_t version;
    uint32_t input_dim;
    uint32_t hidden_dim;
    uint32_t num_actions;
    uint32_t num_qubits;
    uint32_t reserved;
};

struct R2D2ModelWeights {
    double feature_scale[R2D2_INPUT_DIM];
    double feature_offset[R2D2_INPUT_DIM];
    double W_ih[4 * R2D2_HIDDEN_DIM][R2D2_INPUT_DIM];
    double W_hh[4 * R2D2_HIDDEN_DIM][R2D2_HIDDEN_DIM];
    double b_ih[4 * R2D2_HIDDEN_DIM];
    double W_val[R2D2_HIDDEN_DIM];
    double b_val;
    double W_adv[R2D2_NUM_ACTIONS][R2D2_HIDDEN_DIM];
    double b_adv[R2D2_NUM_ACTIONS];
    double W_gain[R2D2_NUM_QUBITS][R2D2_HIDDEN_DIM];
    double b_gain[R2D2_NUM_QUBITS];
    bool loaded_from_checkpoint = false;
    std::string checkpoint_path;
};

/**
 * Load trained R2D2 weights from an IEEE-754 binary checkpoint (.bin).
 * Validates magic header ("AVIDR2D2"), schema version (1), tensor dimensions,
 * exact file byte length (157,984 bytes), and checks that all weights are finite.
 */
bool load_r2d2_binary_checkpoint(
    const std::string& path,
    R2D2ModelWeights& out_weights,
    std::string* err_msg = nullptr);

/**
 * Resolve default checkpoint path across working directories and load weights.
 */
bool load_default_r2d2_checkpoint(
    R2D2ModelWeights& out_weights,
    std::string* err_msg = nullptr);

}  // namespace deployment
}  // namespace avid

#endif  // AVID_DEPLOYMENT_CPP_MODEL_LOADER_H
