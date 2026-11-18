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
    double temporal_beta;
    bool recalibrate;
    bool fallback;
};

constexpr ActionMeta kActionTable[R2D2_NUM_ACTIONS] = {
    {"STATIC_PRIOR_K3",          0.00, 3, 0.75, false, false},
    {"ADAPTIVE_PAIR_K4",         1.15, 4, 0.65, false, false},
    {"AGGRESSIVE_PAIR_K5",       1.35, 5, 0.50, false, false},
    {"DAMPED_PRIOR_K4",          1.05, 4, 0.55, false, false},
    {"FAST_RECALIBRATE_K5",      1.25, 5, 0.00, true,  false},
    {"CONSERVATIVE_FALLBACK_K3", 0.00, 3, 0.00, false, true},
};

template <int InputDim, typename WeightsType>
R2D2InferenceOutput run_lstm_dueling_step(
    const double* raw_features,
    const WeightsType& weights,
    double h[R2D2_HIDDEN_DIM],
    double c[R2D2_HIDDEN_DIM]) {
    R2D2InferenceOutput out{};
    if (!weights.loaded_from_checkpoint) {
        return out;
    }

    double x_norm[InputDim];
    for (int k = 0; k < InputDim; ++k) {
        x_norm[k] = clamp_d(
            raw_features[k] * weights.feature_scale[k] + weights.feature_offset[k],
            -4.0,
            4.0);
    }

    double gates[4 * R2D2_HIDDEN_DIM];
    for (int r = 0; r < 4 * R2D2_HIDDEN_DIM; ++r) {
        double acc = weights.b_ih[r];
        for (int k = 0; k < InputDim; ++k) {
            acc += weights.W_ih[r][k] * x_norm[k];
        }
        for (int j = 0; j < R2D2_HIDDEN_DIM; ++j) {
            acc += weights.W_hh[r][j] * h[j];
        }
        gates[r] = acc;
    }

    for (int r = 0; r < R2D2_HIDDEN_DIM; ++r) {
        const double i_gate = sigmoid_d(gates[r]);
        const double f_gate = sigmoid_d(gates[R2D2_HIDDEN_DIM + r]);
        const double g_gate = std::tanh(clamp_d(gates[2 * R2D2_HIDDEN_DIM + r], -15.0, 15.0));
        const double o_gate = sigmoid_d(gates[3 * R2D2_HIDDEN_DIM + r]);

        c[r] = f_gate * c[r] + i_gate * g_gate;
        const double tanh_c = std::tanh(clamp_d(c[r], -15.0, 15.0));
        h[r] = o_gate * tanh_c;
    }

    double val = weights.b_val;
    for (int j = 0; j < R2D2_HIDDEN_DIM; ++j) {
        val += weights.W_val[j] * h[j];
    }
    out.state_value = val;

    double adv_mean = 0.0;
    for (int a = 0; a < R2D2_NUM_ACTIONS; ++a) {
        double adv_a = weights.b_adv[a];
        for (int j = 0; j < R2D2_HIDDEN_DIM; ++j) {
            adv_a += weights.W_adv[a][j] * h[j];
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

    for (int q = 0; q < R2D2_NUM_QUBITS; ++q) {
        double acc = weights.b_gain[q];
        for (int j = 0; j < R2D2_HIDDEN_DIM; ++j) {
            acc += weights.W_gain[q][j] * h[j];
        }
        out.qubit_gains[q] = R2D2_GAIN_CLIP * std::tanh(clamp_d(acc, -15.0, 15.0));
    }

    out.action_id = best_action;
    out.action_name = kActionTable[best_action].name;
    out.pair_weight_scale = kActionTable[best_action].pair_weight_scale;
    out.bp_iterations = kActionTable[best_action].bp_iterations;
    out.temporal_beta = kActionTable[best_action].temporal_beta;
    out.recalibrate = kActionTable[best_action].recalibrate;
    out.fallback = kActionTable[best_action].fallback;
    return out;
}

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
    return run_lstm_dueling_step<R2D2_INPUT_DIM>(raw_features, weights_, h_, c_);
}

R2D2RuntimeV1::R2D2RuntimeV1() {
    reset_state();
}

bool R2D2RuntimeV1::load_checkpoint(const std::string& path, std::string* err_msg) {
    reset_state();
    return load_r2d2_v1_binary_checkpoint(path, weights_, err_msg);
}

void R2D2RuntimeV1::reset_state() {
    std::memset(h_, 0, sizeof(h_));
    std::memset(c_, 0, sizeof(c_));
}

R2D2InferenceOutput R2D2RuntimeV1::step(const double raw_features[R2D2_V1_INPUT_DIM]) {
    return run_lstm_dueling_step<R2D2_V1_INPUT_DIM>(raw_features, weights_, h_, c_);
}

}  // namespace deployment
}  // namespace avid
