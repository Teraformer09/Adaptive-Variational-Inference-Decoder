# Estimator Input/Output & Q7.8 Quantization Interface (`SF4`, `SF8`)

## 1. Estimator Inputs
- **`empirical_counts[16]` (`uint32_t[16]`)**: 16-bin syndrome histogram accumulated from the DMA ring buffer over a burst window ($N = 64$ rounds for fast trigger detection, $N = 2048$ rounds for full L-BFGS re-estimation).
- **`prior_theta[21]` (`double[21]`)**: Prior physical error rates (9 single-qubit + 12 nearest-neighbor $ZZ$ pairs).

## 2. Estimator Outputs
- **`theta_hat[21]` (`double[21]`)**: MAP-estimated physical error rates in $[10^{-5}, 0.45]$.
- **`p_eff[9]` (`double[9]`)**: Marginalized per-qubit error rates $p_{\text{eff}}(v) = p_v \oplus \bigoplus_{e \ni v} p_e$.
- **`llr_q7_8[9]` (`int16_t[9]`)**: Signed 16-bit Q7.8 log-likelihood ratios $\text{LLR}_v = \operatorname{round}(256 \cdot \ln\frac{1 - p_{\text{eff}}(v)}{p_{\text{eff}}(v)})$ written to the shadow register bank (`0x000..0x020`) and swapped atomically via `COMMIT` (`0x048`).

## References
- **[R6]** J. Chen and M. P. C. Fossorier, "Near optimum universal belief propagation based decoding of low-density parity check codes," *IEEE Transactions on Communications*, vol. 50, no. 3, pp. 406–414, 2002.
