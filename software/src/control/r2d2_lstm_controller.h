/* ===========================================================================
 * r2d2_lstm_controller.h — Trained 64-Unit Dueling LSTM R2D2 Meta-Controller
 * Implements:
 *   - Trained checkpoint loading via deployment/cpp/r2d2_runtime.h (fixes M2)
 *   - Dueling Q-network action selection over 6 discrete actions a0..a5 (fixes M1, M3)
 *   - Causal hardware iteration budget K and pair_weight_scale control (fixes M4)
 *   - Strictly causal inference (consumes only syndrome histograms, MAP estimates,
 *     temporal innovation norm, and fast burst detector score; zero ground-truth leakage).
 * =========================================================================*/
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>

#include "deployment/cpp/r2d2_runtime.h"

struct AvidR2D2MemoryAudit {
    uint32_t input_dim;
    uint32_t hidden_dim;
    uint32_t output_dim;
    uint32_t lstm_gate_params;
    uint32_t head_params;
    uint32_t total_params;
    uint32_t fp32_bytes;
    uint32_t int8_uncompressed_bytes;
    uint32_t lbfgs_workspace_bytes;
};

struct AvidR2D2ControlDecision {
    int action_id = 0;
    const char* action_name = "STATIC_PRIOR_K3";
    double pair_weight_scale = 0.0;
    uint32_t bp_iterations = 3;
    double temporal_beta = 0.75;
    bool recalibrate = false;
    bool fallback = false;
    double state_value = 0.0;
    double q_values[6] = {0.0};
    double qubit_gains[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
};

class AvidR2D2LSTMController {
public:
    static constexpr int INPUT_DIM  = avid::deployment::R2D2_INPUT_DIM;
    static constexpr int HIDDEN_DIM = avid::deployment::R2D2_HIDDEN_DIM;
    static constexpr int OUTPUT_DIM = 16; /* 1 V(h) + 6 A(h,a) + 9 per-qubit gain residuals */

    avid::deployment::R2D2Runtime runtime_;
    double prev_theta_[21];
    bool has_prev_theta_;
    int step_count;
    float last_q_value;
    float last_td_weight;
    int last_action_id;
    double last_pair_weight_scale;
    bool last_recalibrate;
    bool last_fallback;
    double last_q_vector[6];
    uint32_t action_histogram[6];

    AvidR2D2LSTMController() {
        reset_All();
    }

    bool load_checkpoint(const std::string& path, std::string* err_msg = nullptr) {
        return runtime_.load_checkpoint(path, err_msg);
    }

    bool is_checkpoint_loaded() const {
        return runtime_.is_loaded();
    }

    const std::string& checkpoint_path() const {
        return runtime_.checkpoint_path();
    }

    void reset_All() {
        step_count = 0;
        last_q_value = 0.0f;
        last_td_weight = 1.0f;
        last_action_id = 0;
        last_pair_weight_scale = 1.0;
        last_recalibrate = false;
        last_fallback = false;
        has_prev_theta_ = false;
        std::memset(prev_theta_, 0, sizeof(prev_theta_));
        std::memset(last_q_vector, 0, sizeof(last_q_vector));
        std::memset(action_histogram, 0, sizeof(action_histogram));
        runtime_.load_default_checkpoint(nullptr);
        runtime_.reset_state();
    }

    void reset_hidden() {
        runtime_.reset_state();
        has_prev_theta_ = false;
        std::memset(prev_theta_, 0, sizeof(prev_theta_));
        step_count = 0;
    }

    /* Kapturowski invertible value rescaling: h(z) = sign(z)*(sqrt(|z|+1)-1) + eps*z */
    static float kapturowski_h(float z) {
        return static_cast<float>(avid::deployment::value_rescale_h(static_cast<double>(z), 1e-3));
    }

    /**
     * Build the 8-D causal feature vector from syndrome histograms, MAP parameter vector,
     * and fast detector burst score:
     *   0: mean_p
     *   1: max_p
     *   2: mean_alpha
     *   3: max_alpha
     *   4: syndrome_density
     *   5: adjacent_pair_rate
     *   6: innovation_norm
     *   7: burst_score
     */
    void extract_causal_features(
        const uint32_t hist_x[16],
        const uint32_t hist_z[16],
        uint32_t window_rounds,
        const double theta_21[21],
        double burst_score,
        double features_out[8]) {
        double sum_p = 0.0;
        double max_p = 0.0;
        for (int q = 0; q < 9; ++q) {
            const double pq = std::max(0.0, theta_21[q]);
            sum_p += pq;
            if (pq > max_p) max_p = pq;
        }
        const double mean_p = sum_p / 9.0;

        double sum_alpha = 0.0;
        double max_alpha = 0.0;
        for (int e = 0; e < 12; ++e) {
            const double ae = std::max(0.0, theta_21[9 + e]);
            sum_alpha += ae;
            if (ae > max_alpha) max_alpha = ae;
        }
        const double mean_alpha = sum_alpha / 12.0;

        const double inv_w = 1.0 / static_cast<double>(std::max(1u, window_rounds));
        double non_trivial_x = 0.0;
        double non_trivial_z = 0.0;
        double adj_co_excite = 0.0;
        for (int s = 1; s < 16; ++s) {
            const double px = static_cast<double>(hist_x[s]) * inv_w;
            const double pz = static_cast<double>(hist_z[s]) * inv_w;
            non_trivial_x += px;
            non_trivial_z += pz;
            const int pop = __builtin_popcount(static_cast<unsigned int>(s));
            if (pop >= 2) {
                adj_co_excite += 0.5 * (px + pz);
            }
        }
        const double syn_density = std::clamp(0.5 * (non_trivial_x + non_trivial_z), 0.0, 1.0);
        const double adj_pair_rate = std::clamp(adj_co_excite, 0.0, 1.0);

        double innov_sq = 0.0;
        if (has_prev_theta_) {
            for (int i = 0; i < 21; ++i) {
                const double diff = theta_21[i] - prev_theta_[i];
                innov_sq += diff * diff;
            }
            innov_sq /= 21.0;
        }
        for (int i = 0; i < 21; ++i) {
            prev_theta_[i] = theta_21[i];
        }
        has_prev_theta_ = true;

        features_out[0] = mean_p;
        features_out[1] = max_p;
        features_out[2] = mean_alpha;
        features_out[3] = max_alpha;
        features_out[4] = syn_density;
        features_out[5] = adj_pair_rate;
        features_out[6] = std::sqrt(innov_sq);
        features_out[7] = std::clamp(burst_score, 0.0, 1.0);
    }

    /**
     * Execute one causal step of the trained 64-unit Dueling LSTM R2D2 controller.
     */
    AvidR2D2ControlDecision step_causal_decision(
        const uint32_t hist_x[16],
        const uint32_t hist_z[16],
        uint32_t window_rounds,
        const double theta_21[21],
        double fit_nll,
        double burst_score = 0.0) {
        (void)fit_nll;
        double features[8] = {0.0};
        extract_causal_features(hist_x, hist_z, window_rounds, theta_21, burst_score, features);
        AvidR2D2ControlDecision dec = step_from_features(features);

        /* Couple the R2D2 gain head residual delta_g[q] with the causal 21-D MAP posterior
         * across all 12 lattice edges (permutation-covariant across any hot pair (u,v)): */
        static constexpr int kEdgeEndpoints[12][2] = {
            {0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7}, {7, 8},
            {0, 3}, {3, 6}, {1, 4}, {4, 7}, {2, 5}, {5, 8}
        };
        double activity[9] = {0.0};
        double sum_act = 0.0;
        double max_act = 0.0;
        int hot_qubits = 0;
        for (int q = 0; q < 9; ++q) {
            double max_inc_pair = 0.0;
            for (int e = 0; e < 12; ++e) {
                if (kEdgeEndpoints[e][0] == q || kEdgeEndpoints[e][1] == q) {
                    max_inc_pair = std::max(max_inc_pair, theta_21[9 + e]);
                }
            }
            activity[q] = std::max(0.0, theta_21[q] - 0.016) +
                          0.90 * std::max(0.0, max_inc_pair - 0.014);
            sum_act += activity[q];
            max_act = std::max(max_act, activity[q]);
            if (activity[q] > 0.028) {
                hot_qubits++;
            }
        }
        const double mean_act = sum_act / 9.0;
        if (dec.pair_weight_scale > 0.0 && max_act > 0.008) {
            for (int q = 0; q < 9; ++q) {
                const double net_mag = std::clamp(std::abs(dec.qubit_gains[q] - 1.0), 0.04, 0.30);
                if (activity[q] > mean_act + 0.004) {
                    const double rel = std::clamp(activity[q] / std::max(0.01, max_act), 0.25, 1.0);
                    dec.qubit_gains[q] = std::clamp(1.0 - (0.14 + 0.45 * net_mag) * rel, 0.72, 0.88);
                } else if (activity[q] < 0.004) {
                    dec.qubit_gains[q] = std::clamp(1.0 + (0.12 + 0.40 * net_mag), 1.12, 1.26);
                }
            }
        }
        return dec;
    }

    AvidR2D2ControlDecision step_from_features(const double features[8]) {
        const avid::deployment::R2D2InferenceOutput inf = runtime_.step(features);
        AvidR2D2ControlDecision dec{};
        dec.action_id = inf.action_id;
        dec.action_name = inf.action_name;
        dec.pair_weight_scale = inf.pair_weight_scale;
        dec.bp_iterations = static_cast<uint32_t>(inf.bp_iterations);
        dec.temporal_beta = inf.temporal_beta;
        dec.recalibrate = inf.recalibrate;
        dec.fallback = inf.fallback;
        dec.state_value = inf.state_value;
        for (int a = 0; a < 6; ++a) {
            dec.q_values[a] = inf.q_values[a];
            last_q_vector[a] = inf.q_values[a];
        }
        for (int q = 0; q < 9; ++q) {
            dec.qubit_gains[q] = std::clamp(1.0 + inf.qubit_gains[q], 0.70, 1.30);
        }

        last_action_id = dec.action_id;
        last_pair_weight_scale = dec.pair_weight_scale;
        last_recalibrate = dec.recalibrate;
        last_fallback = dec.fallback;
        last_q_value = static_cast<float>(dec.q_values[dec.action_id]);
        last_td_weight = 1.0f / static_cast<float>(1.0 + 0.25 * features[4] * features[4]);
        if (dec.action_id >= 0 && dec.action_id < 6) {
            action_histogram[dec.action_id]++;
        }
        step_count++;
        return dec;
    }

    /* Backwards-compatible wrapper that also outputs pair_weight_scale and action flags */
    void step_causal(const uint32_t hist_x[16],
                     const uint32_t hist_z[16],
                     uint32_t window_rounds,
                     const double theta_21[21],
                     double fit_nll,
                     double gain_out[9],
                     uint32_t *recommended_k_out,
                     double burst_score = 0.0,
                     double *pair_weight_scale_out = nullptr,
                     int *action_id_out = nullptr,
                     bool *recalibrate_out = nullptr,
                     bool *fallback_out = nullptr,
                     double *temporal_beta_out = nullptr) {
        const AvidR2D2ControlDecision dec = step_causal_decision(
            hist_x, hist_z, window_rounds, theta_21, fit_nll, burst_score);
        for (int q = 0; q < 9; ++q) {
            gain_out[q] = dec.qubit_gains[q];
        }
        if (recommended_k_out) *recommended_k_out = dec.bp_iterations;
        if (pair_weight_scale_out) *pair_weight_scale_out = dec.pair_weight_scale;
        if (action_id_out) *action_id_out = dec.action_id;
        if (recalibrate_out) *recalibrate_out = dec.recalibrate;
        if (fallback_out) *fallback_out = dec.fallback;
        if (temporal_beta_out) *temporal_beta_out = dec.temporal_beta;
    }

    static AvidR2D2MemoryAudit get_memory_audit() {
        AvidR2D2MemoryAudit m;
        m.input_dim               = INPUT_DIM;
        m.hidden_dim              = HIDDEN_DIM;
        m.output_dim              = OUTPUT_DIM;
        m.lstm_gate_params        = 4u * HIDDEN_DIM * (INPUT_DIM + HIDDEN_DIM + 1u); /* 18,688 */
        m.head_params             = (HIDDEN_DIM + 1u) + 6u * (HIDDEN_DIM + 1u) + 9u * (HIDDEN_DIM + 1u); /* 1,040 */
        m.total_params            = m.lstm_gate_params + m.head_params + 16u;       /* 19,744 */
        m.fp32_bytes              = m.total_params * 4u;                            /* 78,976 B */
        m.int8_uncompressed_bytes = m.total_params * 1u;                            /* 19,744 B */
        m.lbfgs_workspace_bytes   = 21u * 8u * 16u;                                 /* 2,688 B */
        return m;
    }
};
