# AVID Layer 1 — Code Model, 7-Stage Noise Stimulus & Packet Codec

## 1. Layer Work Description

Branch `layer1-noise-stimulus` isolates **Layer 1** of AVID: the physical/measurement noise generator, surface-code syndrome stimulus pipeline, exact statistical oracle, and 64-bit hardware packet encoder for the rotated $d=3$ CSS surface code ($[[9,1,3]]$, $H_X, H_Z \in \{0,1\}^{4 \times 9}$, $H_X H_Z^T \equiv 0 \pmod 2$).

Core implemented and remediated work:
- **Unified PD Contract (`evaluate_pd_contract` in `config_schema.py`)**: Every generator consumes the assigned profile's per-mechanism level vector, temporal envelope $\text{env}(t)$, and 21-entry dominant mask (`dominant_qubits` + `dominant_pairs`), achieving $\ge 0.887$ envelope correlation and $\ge 80.7\%$ dominant-set hit rate across all 6 data-fault generator types.
- **Multi-Cluster $ZZ$ Crosstalk (`spatial_correlated_generator.py` & `stage3_spatial_correlations.yaml`)**: Replaces single-scalar white jitter with $K=4$ independent AR(1)/OU cluster latent factors (`top_left_plaquette`, `bottom_right_plaquette`, `anti_diagonal_streak`, `central_cross_star`) aligned to `CANONICAL_12_PAIRS`, reducing PC1 variance share from $95.8\%$ to $34.1\%$ ($k_{95\%}=13$ modes, lag-1 $\text{ACF}=0.9951$) and balancing single-generator Z-mass shares to $\le 26.8\%$.
- **Stage 5–7 Measurement Imperfections, `0xAAEC0001` Stream & 32-State Oracle**: Models sticky Markov readout errors, detector dropouts (`VALID` bit 0 parity), and exact 32-state joint syndrome–logical distributions $P(s, \ell \mid \theta)$.

## 2. Run Script to Validate & Generate Results + Plots

Run the Layer 1 validation suite (`95` pytest checks), Stage-5 250 s CLI check, and the **10-minute ($600\text{ s} = 6{,}000\text{ samples}$ at $10\text{ Hz}$ across $6 \times 100\text{ s}$ windows)** multi-noise and $ZZ$ correlation plots:

```bash
python3 -m pip install --user -q numpy scipy pyyaml pytest matplotlib
bash scripts/run_layer1.sh
PYTHONPATH=. python3 scripts/plot_layer1_results.py
```

Generated outputs:
- Logs & metrics: `logs/layer1_validation.log`, `results/layer1_summary.json`, `results/layer1_10min_benchmark.json`
- 10-Minute 5-Noise & $ZZ$ Correlation Plot: `results/plots/layer1_10min_5noise_and_zz_correlations.png`
- Validation & Mass-Share Summary Plot: `results/plots/layer1_validation_summary.png`

![Layer 1 10-Minute 5-Noise & ZZ Correlations](results/plots/layer1_10min_5noise_and_zz_correlations.png)

![Layer 1 Validation Summary](results/plots/layer1_validation_summary.png)

## 3. Validation Results & Performance Analysis vs. Baseline

### Table 3A — 10-Minute ($600\text{ s}$) Multi-Noise & $ZZ$ Crosstalk Performance vs. Baseline

| Noise Family / Generator (`600 s`, `6,000` Samples @ `10 Hz`) | Pre-Fix Env Corr | Post-Fix Env Corr | Pre-Fix Dom Hit | Post-Fix Dom Hit | Post-Fix Lag-1 ACF | Post-Fix PC1 ($k_{95\%}$) | Z-Mass Share |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. Gaussian Envelope (`z_single_noise_01`)** | `0.950` | **`0.9955`** | `80.0%` | **`100.0%`** | `0.9956` | `64.1%` (`3`) | `21.53%` |
| **2. OU Continuous Drift (`x_single_noise_01`)** | `0.020` | **`0.9242`** | `40.0%` | **`100.0%`** | `0.9999` | `84.9%` (`3`) | *(X-basis)* |
| **3. Multi-Cluster $ZZ$ Crosstalk (`zz_01` + `zz_02`)** | `0.000` | **`0.8973`** | `66.0%` | **`80.67%`** | `0.9951` | **`34.1%` (`13`)** | `16.48%` / `13.96%` |
| **4. RTN Telegraph TLS (`telegraph_tls_01`)** | `0.010` | **`0.9823`** | `39.0%` | **`100.0%`** | `0.9935` | `67.8%` (`7`) | `26.75%` |
| **5. Non-Stationary Burst (`burst_noise_01`)** | `0.860` | **`0.8874`** | `97.0%` | **`100.0%`** | `0.9828` | `78.9%` (`7`) | `21.28%` |
| **6. Markov Regime Switching (`markov_regime_01`)** | `0.020` | **`0.9439`** | `38.0%` | **`100.0%`** | `0.9851` | `67.1%` (`5`) | *(X-basis)* |

### Table 3B — Named Verification Test Breakdown (`95/95` Passed)

| Validation Suite / Named Check | Target Scope | Executed | Passed | Key Measured Metric | Status |
| :--- | :--- | ---: | ---: | :--- | :--- |
| Phase 0–3 PD Contract, PCA & LOO Audit (`test_pd_contract_and_pca.py`) | 16 PDs $\times$ 6 generators, `CANONICAL_12_PAIRS`, PCA, LOO Z-mass, PD ID | 6 | 6 | Env corr $\ge 0.887$, PC1 `34.1%`, Max Z-share `26.75%` ($\le 35\%$), PD ID $\ge 90\%$ | **PASS** |
| End-to-End IP/OP & Core Geometry (`test_layer1_ip_op.py` + 7 suites) | $H_X H_Z^T = 0$, 16 PDs, 10 Hz sampler, `u64` packet round-trip | 31 | 31 | Exact GF(2) commutation, RNG isolation, `u64` codec parity | **PASS** |
| Stage 5 Measurement Imperfections (`test_measurement_*.py`) | Sticky Markov readout, detector dropouts, AVID baselines (12 modules) | 55 | 55 | `2,500` Stage-5 samples (`250 s`), `100.0%` (`250/250`) `VALID` parity | **PASS** |
| 7-Stage Integrated Gates & Oracle (`test_seven_stage_noise_framework.py`) | Disjoint splits $\Theta_{\text{train}} \cap \Theta_{\text{test}} = \emptyset$, 32-state Joint-DP | 3 | 3 | $\text{LER}_{\text{oracle}} = 0.0678$, exact vs MC $\text{TVD} = 0.0077$ | **PASS** |
| **Layer 1 Total** | **22 Pytest Modules + 10-Min (600 s) Benchmark + Stage 5/6/7** | **95 tests** | **95 tests** | **`0` failures** | **PASS** |
