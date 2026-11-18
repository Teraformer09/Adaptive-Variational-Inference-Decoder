#ifndef AVID_DEPLOYMENT_CPP_OBSERVATION_ADAPTER_H
#define AVID_DEPLOYMENT_CPP_OBSERVATION_ADAPTER_H

#include "deployment/cpp/model_loader.h"
#include <string>

namespace avid {
namespace deployment {

constexpr double R2D2_POSTERIOR_VAR_EPS = 1.0e-8;

struct Layer3PosteriorInputV2 {
    double posterior_mean_21[21]{};
    double posterior_var_21[21]{};
    double hist_x_16[16]{};
    double hist_z_16[16]{};
    double syndrome_density_x = 0.0;
    double syndrome_density_z = 0.0;
    double adjacent_coactivation_x = 0.0;
    double adjacent_coactivation_z = 0.0;
    double posterior_innovation_l2 = 0.0;
    double burst_score = 0.0;
    double estimator_residual_tvd = 0.0;
    double decoder_config_age_norm = 0.0;
};

/**
 * Validate a serialized 82-element V2 observation contract:
 *  - [0:21]  posterior means in [0.0, 0.50]
 *  - [21:42] log(sigma^2 + 1e-8) in [log(1e-8), 2.0]
 *  - [42:58] X-check 16-state histogram >= 0, sum == 1.0
 *  - [58:74] Z-check 16-state histogram >= 0, sum == 1.0
 *  - [74:82] non-negative finite telemetry
 */
bool validate_observation_v2(
    const double obs82[R2D2_INPUT_DIM],
    std::string* err_msg = nullptr);

/**
 * Serialize a Layer 3 posterior + syndrome summary struct into the 82-element R2D2 V2 observation.
 */
bool serialize_layer3_to_observation_v2(
    const Layer3PosteriorInputV2& input,
    double out_obs82[R2D2_INPUT_DIM],
    std::string* err_msg = nullptr);

}  // namespace deployment
}  // namespace avid

#endif  // AVID_DEPLOYMENT_CPP_OBSERVATION_ADAPTER_H
