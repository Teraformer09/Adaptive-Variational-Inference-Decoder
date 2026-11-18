#ifndef AVID_DEPLOYMENT_CPP_MODEL_LOADER_H
#define AVID_DEPLOYMENT_CPP_MODEL_LOADER_H

#include <cstdint>
#include <string>

namespace avid {
namespace deployment {

// Canonical V2 82-input syndrome-conditioned architecture constants
constexpr uint32_t R2D2_SCHEMA_VERSION_V1 = 1;
constexpr uint32_t R2D2_SCHEMA_VERSION_V2 = 2;

constexpr int R2D2_V1_INPUT_DIM = 8;
constexpr int R2D2_INPUT_DIM = 82;
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
    uint32_t schema_version = 0;
    bool loaded_from_checkpoint = false;
    std::string checkpoint_path;
};

// Frozen V1 weight structure preserved strictly for V1 legacy parity verification
struct R2D2ModelWeightsV1 {
    double feature_scale[R2D2_V1_INPUT_DIM];
    double feature_offset[R2D2_V1_INPUT_DIM];
    double W_ih[4 * R2D2_HIDDEN_DIM][R2D2_V1_INPUT_DIM];
    double W_hh[4 * R2D2_HIDDEN_DIM][R2D2_HIDDEN_DIM];
    double b_ih[4 * R2D2_HIDDEN_DIM];
    double W_val[R2D2_HIDDEN_DIM];
    double b_val;
    double W_adv[R2D2_NUM_ACTIONS][R2D2_HIDDEN_DIM];
    double b_adv[R2D2_NUM_ACTIONS];
    double W_gain[R2D2_NUM_QUBITS][R2D2_HIDDEN_DIM];
    double b_gain[R2D2_NUM_QUBITS];
    uint32_t schema_version = 0;
    bool loaded_from_checkpoint = false;
    std::string checkpoint_path;
};

/**
 * Load trained R2D2 V2 weights from an IEEE-754 binary checkpoint (.bin).
 * Validates magic ("AVIDR2D2"), schema version == 2, input_dim == 82,
 * exact file byte length (310,720 bytes), and finite weights.
 * Explicitly rejects V1 checkpoints rather than silently misloading them.
 */
bool load_r2d2_binary_checkpoint(
    const std::string& path,
    R2D2ModelWeights& out_weights,
    std::string* err_msg = nullptr);

/**
 * Load frozen V1 weights (version == 1, input_dim == 8, 157,984 bytes).
 * Explicitly rejects V2 checkpoints.
 */
bool load_r2d2_v1_binary_checkpoint(
    const std::string& path,
    R2D2ModelWeightsV1& out_weights,
    std::string* err_msg = nullptr);

bool load_default_r2d2_checkpoint(
    R2D2ModelWeights& out_weights,
    std::string* err_msg = nullptr);

}  // namespace deployment
}  // namespace avid

#endif  // AVID_DEPLOYMENT_CPP_MODEL_LOADER_H
