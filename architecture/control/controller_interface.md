# 64-Unit LSTM R2D2-WB Controller Interface

Implemented in `software/src/control/r2d2_lstm_controller.h` (`AvidR2D2LSTMController`):
- **Input dimension**: `D_in = 34` (`16` normalized syndrome histogram bins + `9` single-qubit MAP estimates + `9` previous effective marginal probabilities).
- **Recurrent core**: `H = 64` LSTM hidden units (`4 * 64 * (34 + 64 + 1) = 25,344` parameters).
- **Output head**: `64 -> 9` residual adjustment + `1` trust gate (`64 * 10 + 14 = 654` parameters).
- **Total footprint**: `25,998` parameters (`25,998` bytes = `25.39 KB` in `int8`; `103,992` bytes = `101.55 KB` in `fp32`), fitting entirely inside the 256 KB Zynq-7020 OCM / L2 cache.
