# 64-Unit Dueling LSTM R2D2 Controller Interface

Implemented in `software/src/control/r2d2_lstm_controller.h` (`AvidR2D2LSTMController`) and `deployment/cpp/r2d2_runtime.cpp` (`R2D2Runtime`):
- **Input dimension**: `D_in = 8` (`mean_p`, `max_p`, `mean_alpha`, `max_alpha`, `syndrome_density`, `adjacent_pair_rate`, `innovation_norm`, `burst_score`) + `16` affine normalization parameters (`feature_scale[8]`, `feature_offset[8]`).
- **Recurrent core**: `H = 64` LSTM hidden units (`4 * 64 * (8 + 64 + 1) = 18,688` parameters).
- **Output heads**: Dueling Q-value head (`V(h): 64 * 1 + 1 = 65` parameters; `A(h, a): 64 * 6 + 6 = 390` parameters over 6 discrete actions $a_0 \dots a_5$: `STATIC_PRIOR_K3`, `MODERATE_PROJ_K4`, `AGGRESSIVE_PROJ_K5`, `DAMPED_PRIOR_K4`, `FAST_RECALIBRATE_K5`, `SAFE_FALLBACK_K3`) + 9-qubit per-qubit gain residual head (`64 * 9 + 9 = 585` parameters).
- **Total footprint**: `19,744` parameters (`19,744` bytes = `19.28 KB` in `int8`; `78,976` bytes = `77.12 KB` in `fp32`; `157,984` bytes in `fp64` binary checkpoint `checkpoints/r2d2_v1_trained.bin`), fitting entirely inside the 256 KB Zynq-7020 OCM / L2 cache.
