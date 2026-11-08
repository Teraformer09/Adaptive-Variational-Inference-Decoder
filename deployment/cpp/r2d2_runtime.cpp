#include "deployment/cpp/r2d2_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace avid {
namespace deployment {

namespace {

inline double clamp_d(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

inline double sigmoid_d(double x) {
    const double xc = clamp_d(x, -30.0, 30.0);
    return 1.0 / (1.0 + std::exp(-xc));
}

struct ActionMeta {
    const char* name;
    double pair_weight_scale;
    int bp_iterations;
    bool recalibrate;
    bool fallback;
};

constexpr ActionMeta kActionTable[R2D2_NUM_ACTIONS] = {
    {"STATIC_PRIOR_K3",          0.00, 3, false, false},
    {"ADAPTIVE_PAIR_K5",         1.00, 5, false, false},
    {"AGGRESSIVE_PAIR_K7",       1.35, 7, false, false},
    {"DAMPED_PRIOR_K4",          0.85, 4, false, false},
    {"FAST_RECALIBRATE_K7",      1.15, 7, true,  false},
    {"CONSERVATIVE_FALLBACK_K3", 0.00, 3, false, true},
};

}  // namespace

double value_rescale_h(double z, double eps) {
    const double s = (z > 0.0) ? 1.0 : ((z < 0.0) ? -1.0 : 0.0);
    return s * (std::sqrt(std::abs(z) + 1.0) - 1.0) + eps * z;
}

double value_rescale_h_inv(double y, double eps) {
    const double s = (y > 0.0) ? 1.0 : ((y < 0.0) ? -1.0 : 0.0);
    const double abs_y = std::abs(y);
    const double inner = (std::sqrt(1.0 + 4.0 * eps * (abs_y + 1.0 + eps)) - 1.0) / (2.0 * eps);
    return s * (inner * inner - 1.0);
}

R2D2Runtime::R2D2Runtime() {
    reset_state();
}

bool R2D2Runtime::load_checkpoint(const std::string& path, std::string* err_msg) {
    reset_state();
    return load_r2d2_binary_checkpoint(path, weights_, err_msg);
}

bool R2D2Runtime::load_default_checkpoint(std::string* err_msg) {
    reset_state();
    return load_default_r2d2_checkpoint(weights_, err_msg);
}

void R2D2Runtime::reset_state() {
    std::memset(h_, 0, sizeof(h_));
    std::memset(c_, 0, sizeof(c_));
}

R2D2InferenceOutput R2D2Runtime::step(const double raw_features[R2D2_INPUT_DIM]) {
    R2D2InferenceOutput out{};
    if (!weights_.loaded_from_checkpoint) {
        return out;
    }

    // 1. Normalize features into [-4.0, 4.0]
    double x_norm[R2D2_INPUT_DIM];
    for (int k = 0; k < R2D2_INPUT_DIM; ++k) {
        x_norm[k] = clamp_d(
            raw_features[k] * weights_.feature_scale[k] + weights_.feature_offset[k],
            -4.0,
            4.0);
    }

    // 2. Compute LSTM gates: gates = W_ih @ x_norm + W_hh @ h_prev + b_ih
    double gates[4 * R2D2_HIDDEN_DIM];
    for (int r = 0; r < 4 * R2D2_HIDDEN_DIM; ++r) {
        double acc = weights_.b_ih[r];
        for (int k = 0; k < R2D2_INPUT_DIM; ++k) {
            acc += weights_.W_ih[r][k] * x_norm[k];
        }
        for (int j = 0; j < R2D2_HIDDEN_DIM; ++j) {
            acc += weights_.W_hh[r][j] * h_[j];
        }
        gates[r] = acc;
    }

    // 3. Apply gate activations and update cell & hidden states
    for (int r = 0; r < R2D2_HIDDEN_DIM; ++r) {
        const double i_gate = sigmoid_d(gates[r]);
        const double f_gate = sigmoid_d(gates[R2D2_HIDDEN_DIM + r]);
        const double g_gate = std::tanh(clamp_d(gates[2 * R2D2_HIDDEN_DIM + r], -15.0, 15.0));
        const double o_gate = sigmoid_d(gates[3 * R2D2_HIDDEN_DIM + r]);

        c_[r] = f_gate * c_[r] + i_gate * g_gate;
        const double tanh_c = std::tanh(clamp_d(c_[r], -15.0, 15.0));
        h_[r] = o_gate * tanh_c;
    }

    // 4. Dueling state-value V(h) and advantage A(h, a)
    double val = weights_.b_val;
    for (int j = 0; j < R2D2_HIDDEN_DIM; ++j) {
        val += weights_.W_val[j] * h_[j];
    }
    out.state_value = val;

    double adv_mean = 0.0;
    for (int a = 0; a < R2D2_NUM_ACTIONS; ++a) {
        double adv_a = weights_.b_adv[a];
        for (int j = 0; j < R2D2_HIDDEN_DIM; ++j) {
            adv_a += weights_.W_adv[a][j] * h_[j];
        }
        out.advantages[a] = adv_a;
        adv_mean += adv_a;
    }
    adv_mean /= static_cast<double>(R2D2_NUM_ACTIONS);

    int best_action = 0;
    double best_q = -1e30;
    for (int a = 0; a < R2D2_NUM_ACTIONS; ++a) {
        out.q_values[a] = out.state_value + (out.advantages[a] - adv_mean);
        if (out.q_values[a] > best_q) {
            best_q = out.q_values[a];
            best_action = a;
        }
    }

    // 5. Per-qubit bounded LLR gain residuals delta_g in [-0.30, +0.30]^9
    for (int q = 0; q < R2D2_NUM_QUBITS; ++q) {
        double acc = weights_.b_gain[q];
        for (int j = 0; j < R2D2_HIDDEN_DIM; ++j) {
            acc += weights_.W_gain[q][j] * h_[j];
        }
        out.qubit_gains[q] = R2D2_GAIN_CLIP * std::tanh(clamp_d(acc, -15.0, 15.0));
    }

    out.action_id = best_action;
    out.action_name = kActionTable[best_action].name;
    out.pair_weight_scale = kActionTable[best_action].pair_weight_scale;
    out.bp_iterations = kActionTable[best_action].bp_iterations;
    out.recalibrate = kActionTable[best_action].recalibrate;
    out.fallback = kActionTable[best_action].fallback;
    return out;
}

}  // namespace deployment
}  // namespace avid
