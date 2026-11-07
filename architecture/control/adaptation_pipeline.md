# Slow-Loop Parameter Estimation & Weight Adaptation Pipeline

1. **Dual Histogram Accumulation**: `avid_dma_ring_consume_slot()` increments 16-bin histograms `hist_x[0..15]` ($X$-stabilizer syndromes diagnosing $Z$ faults) and `hist_z[0..15]` ($Z$-stabilizer syndromes diagnosing $X$ faults).
2. **Two-Pass 16-State Joint-DP L-BFGS**:
   - **Pass 1**: Optimizes the 9 single-qubit parameters (`0..8`) against the exact 16-state XOR-convolution likelihood + Gaussian prior.
   - **Pass 2**: Activates correlated pair parameters (`9..20`) coupled to hot single-qubit endpoints (`theta_q > 0.025`) to resolve adjacent $ZZ$/$XX$ crosstalk without diluting single-qubit hot spots across the 12 gauge null directions.
3. **Pair-Aware Logical-Triangle Projection & R2D2-WB Blending**:
   - Projects estimated pair rates onto effective per-qubit marginals and blends with the 64-unit LSTM R2D2-WB recurrent state before Q7.8 quantization.
