#ifndef AVID_DEPLOYMENT_CPP_R2D2_RUNTIME_H
#define AVID_DEPLOYMENT_CPP_R2D2_RUNTIME_H

#include "deployment/cpp/model_loader.h"
#include <string>

namespace avid {
namespace deployment {

struct R2D2InferenceOutput {
    int action_id = 0;
    const char* action_name = "STATIC_PRIOR_K3";
    double pair_weight_scale = 0.0;
    int bp_iterations = 3;
    double temporal_beta = 0.75;
    bool recalibrate = false;
    bool fallback = false;
    double state_value = 0.0;
    double advantages[R2D2_NUM_ACTIONS] = {0.0};
    double q_values[R2D2_NUM_ACTIONS] = {0.0};
    double qubit_gains[R2D2_NUM_QUBITS] = {0.0};
};

class R2D2Runtime {
public:
    R2D2Runtime();

    bool load_checkpoint(const std::string& path, std::string* err_msg = nullptr);
    bool load_default_checkpoint(std::string* err_msg = nullptr);

    void reset_state();

    /**
     * Execute one 82-input recurrent LSTM + Dueling Q-head + per-qubit gain inference step.
     * Matches Python `R2D2Network.step()` (V2, input_dim=82) to within < 1e-12 numerical precision.
     */
    R2D2InferenceOutput step(const double raw_features[R2D2_INPUT_DIM]);

    bool is_loaded() const { return weights_.loaded_from_checkpoint; }
    const std::string& checkpoint_path() const { return weights_.checkpoint_path; }
    const R2D2ModelWeights& weights() const { return weights_; }
    const double* hidden_state() const { return h_; }
    const double* cell_state() const { return c_; }

private:
    R2D2ModelWeights weights_{};
    double h_[R2D2_HIDDEN_DIM]{};
    double c_[R2D2_HIDDEN_DIM]{};
};

/**
 * Frozen V1 8-input runtime retained strictly for reproducing V1 Python/C++ parity evidence.
 */
class R2D2RuntimeV1 {
public:
    R2D2RuntimeV1();
    bool load_checkpoint(const std::string& path, std::string* err_msg = nullptr);
    void reset_state();
    R2D2InferenceOutput step(const double raw_features[R2D2_V1_INPUT_DIM]);
    bool is_loaded() const { return weights_.loaded_from_checkpoint; }

private:
    R2D2ModelWeightsV1 weights_{};
    double h_[R2D2_HIDDEN_DIM]{};
    double c_[R2D2_HIDDEN_DIM]{};
};

double value_rescale_h(double z, double eps = 1e-3);
double value_rescale_h_inv(double y, double eps = 1e-3);

}  // namespace deployment
}  // namespace avid

#endif  // AVID_DEPLOYMENT_CPP_R2D2_RUNTIME_H
