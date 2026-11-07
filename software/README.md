# AVID C/C++ Slow-Path Runtime & HAL (`software/`)

- `src/inference/`: `avid_daemon.cpp`, `noise_model.cpp` (16-state Joint-DP & analytic coset gradient), `lbfgs_optimizer.cpp`.
- `src/estimator/`: `lbfgs_solver.cpp` (two-pass endpoint-coupled projected L-BFGS), `joint_dp_estimator.cpp`, `llr_quantizer.cpp` (signed Q7.8 quantizer).
- `src/control/`: `causal_scheduler.cpp` (epoch scheduler + 16-round sliding-subwindow fast burst detector), `r2d2_lstm_controller.h` (64-unit `int8` LSTM R2D2-WB controller), `fallback_fsm.cpp`, `stage_profiler.cpp`.
- `src/driver/` & `src/hal/`: MMIO weight-bank driver and HAL (`simulation_backend.cpp` TCP/UNIX bridge client and `physical_backend.cpp`).
- `src/ingress/`: `dma_ingress_tap.cpp` and `packet_validator.cpp`.
