#include "deployment/cpp/observation_adapter.h"

#include <algorithm>
#include <cmath>

namespace avid {
namespace deployment {

bool validate_observation_v2(
    const double obs82[R2D2_INPUT_DIM],
    std::string* err_msg) {
    if (!obs82) {
        if (err_msg) *err_msg = "Null observation pointer";
        return false;
    }

    for (int i = 0; i < R2D2_INPUT_DIM; ++i) {
        if (!std::isfinite(obs82[i])) {
            if (err_msg) *err_msg = "Non-finite value at observation index " + std::to_string(i);
            return false;
        }
    }

    for (int i = 0; i < 21; ++i) {
        if (obs82[i] < -1e-9 || obs82[i] > 0.50 + 1e-9) {
            if (err_msg) *err_msg = "Posterior mean out of bounds [0, 0.50] at index " + std::to_string(i);
            return false;
        }
    }

    const double min_logvar = std::log(R2D2_POSTERIOR_VAR_EPS) - 1e-6;
    for (int i = 21; i < 42; ++i) {
        if (obs82[i] < min_logvar || obs82[i] > 2.0) {
            if (err_msg) *err_msg = "Posterior log-variance out of bounds at index " + std::to_string(i);
            return false;
        }
    }

    double sum_x = 0.0;
    double sum_z = 0.0;
    for (int s = 0; s < 16; ++s) {
        const double fx = obs82[42 + s];
        const double fz = obs82[58 + s];
        if (fx < -1e-9 || fz < -1e-9) {
            if (err_msg) *err_msg = "Negative syndrome histogram entry";
            return false;
        }
        sum_x += fx;
        sum_z += fz;
    }
    if (std::abs(sum_x - 1.0) > 1e-4) {
        if (err_msg) *err_msg = "X syndrome histogram does not sum to 1.0";
        return false;
    }
    if (std::abs(sum_z - 1.0) > 1e-4) {
        if (err_msg) *err_msg = "Z syndrome histogram does not sum to 1.0";
        return false;
    }

    for (int i = 74; i < 82; ++i) {
        if (obs82[i] < -1e-9) {
            if (err_msg) *err_msg = "Negative telemetry feature at index " + std::to_string(i);
            return false;
        }
    }
    return true;
}

bool serialize_layer3_to_observation_v2(
    const Layer3PosteriorInputV2& input,
    double out_obs82[R2D2_INPUT_DIM],
    std::string* err_msg) {
    if (!out_obs82) {
        if (err_msg) *err_msg = "Null output buffer";
        return false;
    }

    for (int i = 0; i < 21; ++i) {
        if (!std::isfinite(input.posterior_mean_21[i]) ||
            !std::isfinite(input.posterior_var_21[i]) ||
            input.posterior_var_21[i] < 0.0) {
            if (err_msg) *err_msg = "Invalid posterior mean or variance at index " + std::to_string(i);
            return false;
        }
        out_obs82[i] = std::max(0.0, std::min(0.49, input.posterior_mean_21[i]));
        out_obs82[21 + i] = std::log(input.posterior_var_21[i] + R2D2_POSTERIOR_VAR_EPS);
    }

    for (int s = 0; s < 16; ++s) {
        out_obs82[42 + s] = input.hist_x_16[s];
        out_obs82[58 + s] = input.hist_z_16[s];
    }

    out_obs82[74] = std::max(0.0, std::min(1.0, input.syndrome_density_x));
    out_obs82[75] = std::max(0.0, std::min(1.0, input.syndrome_density_z));
    out_obs82[76] = std::max(0.0, std::min(1.0, input.adjacent_coactivation_x));
    out_obs82[77] = std::max(0.0, std::min(1.0, input.adjacent_coactivation_z));
    out_obs82[78] = std::max(0.0, input.posterior_innovation_l2);
    out_obs82[79] = std::max(0.0, std::min(1.0, input.burst_score));
    out_obs82[80] = std::max(0.0, std::min(1.0, input.estimator_residual_tvd));
    out_obs82[81] = std::max(0.0, std::min(1.0, input.decoder_config_age_norm));

    return validate_observation_v2(out_obs82, err_msg);
}

}  // namespace deployment
}  // namespace avid
