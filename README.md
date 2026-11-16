# AVID Layer 1 — Code Model, 7-Stage Noise Stimulus & Packet Codec

## 1. Layer Work Description

Branch `layer1-noise-stimulus` isolates **Layer 1** of the Adaptive Variational Inference Decoder (AVID): the physical/measurement noise generator, surface-code syndrome stimulus pipeline, exact statistical oracle, and 64-bit hardware packet encoder for the rotated $d=3$ CSS surface code ($[[9,1,3]]$, $H_X, H_Z \in \{0,1\}^{4 \times 9}$, $H_X H_Z^T \equiv 0 \pmod 2$).

Core implemented and remediated work in this layer:
- **Stage 1–4 Physical Noise Pipeline**: 16 truncated-Gaussian probability distribution (PD) profiles, 100 s persistent curriculum assignment, 10 Hz synchronized multi-generator sampling, and Pauli XOR / independent-channel Bernoulli composition across single-qubit and 12 adjacent pair-correlated edges ($\theta \in [0,1]^{21}$).
- **Stage 5 Dynamic Measurement Imperfections**: Temporally sticky two-state Markov readout errors, readout detector dropouts (`validity_mask_x`, `validity_mask_z`), and event/delivery observation timestamps.
- **Stage 6 Stratified Dataset & Binary Stream Codec**: Disjoint configuration-level `train` / `val` / `test` trajectory generation (`.npz`, `.meta.json`, and 16-byte `0xAAEC0001`-prefixed `.bin` packet streams) with `AVID-SYNDROME-PACKET-V1` 64-bit little-endian encoding. Readout dropouts (`validity_mask == 0`) deterministically clear bit 0 of `flags` (`VALID = 0`) for downstream Layer 3 ingress filtering.
- **Stage 7 Exact 32-State Joint-DP Oracle**: Exact joint syndrome–logical distribution $P(s, \ell \mid \theta)$, syndrome-aliasing diagnostics, and shared-trajectory Bayes-optimal logical error rate (LER) floor evaluation.

## 2. Run Script to Validate & Generate Results + Plots

Run the full Layer 1 validation suite, Stage-5 250 s CLI check, summary JSON export, and diagnostic plot generation from the repository root:

```bash
python3 -m pip install --user -q numpy scipy pyyaml pytest matplotlib
bash scripts/run_layer1.sh
```

To regenerate `results/layer1_summary.json` and the figure directly from the recorded validation log and live Stage-6/Stage-7 checks:

```bash
PYTHONPATH=. python3 scripts/plot_layer1_results.py
```

Generated outputs:
- Validation log: `logs/layer1_validation.log`
- Metrics JSON: `results/layer1_summary.json`
- Diagnostic plot: `results/plots/layer1_validation_summary.png`

![Layer 1 Validation Summary](results/plots/layer1_validation_summary.png)

## 3. Validation Results

| Validation Suite / Check | Target Scope | Executed | Passed | Key Measured Metric | Status |
| :--- | :--- | ---: | ---: | :--- | :--- |
| End-to-End IP/OP Contract (`test_layer1_ip_op.py`) | $H_X H_Z^T = 0$, 16 PDs, 10 Hz sampler, `u64` codec | 1 | 1 | `50` synchronized frames, exact `u64` round-trip | **PASS** |
| Core Geometry, PD, Lifecycle & Syndrome Suites (`simulation/tests/`) | Stages 1–4 & packet round-trip (7 modules) | 30 | 30 | Exact GF(2) commutation & RNG stream isolation | **PASS** |
| Stage 5 Measurement Imperfection Suites (`simulation/tests/`) | Config, sticky Markov, validity, AVID baselines (12 modules) | 55 | 55 | 3 static baselines + dynamic readout dropout masks | **PASS** |
| 7-Stage Integrated Gates & Oracle (`test_seven_stage_noise_framework.py`) | Stages 1–7, $\Theta_{\text{train}} \cap \Theta_{\text{test}} = \emptyset$, 32-state DP | 3 | 3 | $\text{LER}_{\text{oracle}} = 0.0678$, R2D2 $\le$ Static BP LER | **PASS** |
| Stage 5 Measurement CLI (`stage5_measurement_noise.py`) | `250.0 s` @ `10 Hz` (`seed=42`) | 2,500 samples | 2,500 samples | `3` persistent 100 s windows validated | **PASS** |
| Stage 6 Readout Dropout $\to$ Packet `VALID` Flag | `0xAAEC0001` `.bin` stream (`250` packets) | 250 pkts | 250 pkts | `100.0%` (`1.000`) mask-to-`VALID` bit parity | **PASS** |
| Stage 7 Exact DP vs Empirical Monte Carlo | $P(s_X \mid \theta)$ over 16 syndrome states ($N=4096$) | 16 states | 16 states | $\text{TVD} = 0.0077$ | **PASS** |
| **Layer 1 Total** | **21 Pytest Modules + Stage 5/6/7 CLI & Plot Checks** | **89 tests** | **89 tests** | **`0` failures (`183.52 s`)** | **PASS** |
