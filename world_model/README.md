# AVID World Model — Noise-State Estimation and Forecasting Contract

**Status:** design specification; the RSSM described here is not yet implemented or trained.  
**Target:** repeated-round surface-code noise trajectories on superconducting transmon hardware with tunable couplers.  
**Scope:** **noise generation interface, latent noise-state estimation, and short-horizon noise forecasting only.** Downstream QEC decoding, decoder weight adaptation, RL policies, and autonomous pulse control are excluded from this phase.

## 1. Role in the Noise-Generation and Estimation Pipeline

The world model is a **predictive latent noise-state estimator** for a partially observed, time-varying quantum hardware noise process. It consumes only classical observations generated during repeated syndrome-extraction cycles: detection-event history, raw stabilizer outcomes, known circuit schedule, calibration telemetry, device topology, and optional analog readout features. Simulator-only hidden variables (`latent_truth`) are used strictly as offline training supervision and evaluation ground truth—never as inference inputs.

It estimates a posterior belief over the latent environmental/noise state and forecasts future observable fault/detection statistics over horizons $H \in \{1, 2, 4, 8\}$. Classical noise estimators (`B0` static independent, `B1` static correlated, `B2` Markov/HMM, `B3` empirical/variational tracker, and `B4` AR(1)/OU filter) serve as the mandatory baselines that the world model must outperform on held-out noise trajectories.

## 2. Observation contract

For QEC round `t`, define

\[
o_t = (d_t,s_t,c_t,a_t,u_t,G),
\]

where:

- `d_t`: sparse detection events or a binary detector vector;
- `s_t`: optional raw stabilizer outcomes, kept separate from detection events;
- `c_t`: calibration telemetry with timestamps and validity masks (e.g. measured T1/T2, readout assignment statistics, gate RB estimates, qubit/coupler frequencies);
- `a_t`: optional analog readout features (IQ summaries or calibrated discriminator log-likelihood ratios; raw IQ is not assumed available);
- `u_t`: known schedule features (gate type, targets, couplers, simultaneous-gate concurrency group, drive amplitude, idle/measurement/reset windows, and elapsed physical time);
- `G`: versioned device/check graph.

For consecutive stabilizer measurements, `d_t = s_t XOR s_(t-1)` is a common convention, but it must be adjusted for initialization, measurement basis, boundary checks, and the code's detector definition. Do not apply it blindly to every raw syndrome bit.

Missing calibration/readout channels must have explicit masks. Zero is a valid value and must not double as a missing-value marker.

**Inference API (conceptual):**

```python
belief, forecast = world_model.predict(
    observation_window=window,
    topology=topology,
    future_schedule=known_schedule,
    horizon=H,
)
```

The call must not accept `latent_truth` or hidden simulator state.

## 3. Structured latent noise state

Use typed latent noise components rather than one uninterpretable vector:

- `z_corr`: local schedule-conditioned correlated interaction/crosstalk regime and edge-level latent variables;
- `z_leak`: per-transmon non-computational leakage/seepage belief;
- `z_drift`: slow calibration/relaxation/dephasing drift;
- `z_memory`: temporal-memory variables for driven TLS defects or non-Markovian idle baths;
- `z_event`: sparse chip-wide common-mode disturbance state and spatial footprint.

Simulator-only `latent_truth` contains exact Hamiltonian parameters, qutrit populations, injected event locations, true leakage flags, composite TLS density-matrix state, and injected fault causes. Keep this in a sidecar field and exclude it from inference inputs.

## 4. Model family: graph-aware RSSM

Let `G=(V,E)` denote the hardware/check graph. Encode node, edge, observation, and schedule features with message passing:

\[
g_t = \operatorname{GNN}_{\phi}(G, X_t, E_t).
\]

Maintain deterministic recurrent memory `h_t` and stochastic latent noise state `z_t`:

\[
h_t = f_{\theta}(h_{t-1},z_{t-1},g_t),
\]
\[
q_{\phi}(z_t\mid h_t,g_t) \quad\text{(posterior)}, \qquad
p_{\theta}(z_t\mid h_t) \quad\text{(prior)}.
\]

At each round, the prior predicts before seeing the new observation; the posterior incorporates the observation. Multi-step noise forecast distributions are formed from the posterior belief and the known future gate schedule:

\[
p(o_{t+1:t+H}\mid b_t,u_{t+1:t+H}),
\]

where `b_t` denotes the inferred noise belief and `H ∈ {1,2,4,8}` rounds.

Use sparse local/edge correlation heads conditioned on the simultaneous gate schedule `u_t`. Physical coherent interactions and effective stochastic circuit faults must remain distinct representations.

## 5. Noise-prediction heads

1. **Detection head:** per-check detection-event probabilities and calibrated uncertainty across horizons `1..H`.
2. **Correlation head:** sparse edge/local joint-fault probabilities or pairwise log-odds conditioned on the gate schedule `u_t`.
3. **Leakage head:** per-transmon leakage/seepage probability and persistence forecast.
4. **Drift/memory head:** posterior and multi-step forecast distributions for slow drift, driven TLS state, and idle memory revivals.
5. **Calibration head:** predictive distributions for future calibration telemetry measurements, with missingness masks handled explicitly.
6. **Common-event head:** chip-wide burst probability and spatial-footprint distribution across the processor geometry.

No logical-error or decoder head is included in this phase; evaluation is strictly on observable noise prediction and latent noise-state estimation.

## 6. Training objective

Use a multi-task sequential variational objective:

\[
\mathcal L = \lambda_{obs}\mathcal L_{obs}
+ \beta\,D_{KL}\!\left[q_{\phi}(z_t\mid h_t,g_t)\,\|\,p_{\theta}(z_t\mid h_t)\right]
+ \lambda_{aux}\mathcal L_{aux}
+ \lambda_{forecast}\mathcal L_{forecast}.
\]

- `L_obs`: proper negative log-likelihoods for binary detections, readout features, and continuous calibration telemetry (masked where absent).
- `KL`: posterior-to-prior transition dynamics regularization.
- `L_aux`: simulator-only auxiliary supervision for hidden noise causes (leakage state, injected common events, TLS/drift state, and schedule-conditioned correlated fault rates) used only during training.
- `L_forecast`: multi-step predictive likelihood and calibration across horizons `H ∈ {1,2,4,8}`.

## 7. Noise-simulator responsibilities

| Layer | Tool | Responsibility | Explicit limitation |
|---|---|---|---|
| Physical reference | QuTiP / QuTiP-QIP | Small-system coherent ZZ/crosstalk, qutrit leakage, spatiotemporal relaxation, and composite qubit–TLS/bath dynamics | Exponential scaling prevents large-code multi-round simulation |
| Published leakage reproduction | quantumsim | Reproduce the full Surface-17 density-matrix leakage benchmark of Varbanov et al. (2020) | Specialized to density-matrix stabilizer cycles |
| Stabilizer circuit | Stim | Scalable stabilizer sampling and Pauli-compatible effective fault injection | Cannot natively evolve coherent non-Clifford, qutrit, or non-Markovian bath states |
| Trajectory generator | Custom noise engine + leakage wrapper | Evolve latent crosstalk, event, leakage, and TLS/memory processes and emit `(observations, latent_truth)` trajectories | Every effective channel approximation must be validated against QuTiP/quantumsim physical references |

## 8. Dataset and split integrity

Each trajectory record carries a stable `trajectory_id`, simulator seed, device/topology version, simulator/config hash, and round range (`world_model/dataset/trajectory.schema.json`). Split train/validation/test strictly by **complete trajectories and random seeds**—never by individual rounds within a trajectory—to prevent temporal data leakage across splits.

## 9. Noise-estimation evaluation protocol

Report each noise benchmark (`A1, A2, B1, C1, C2`) separately and under **mixed-noise generation** (`MIXED_ALL_THREE`):

- detection-event predictive NLL, Brier score, and Expected Calibration Error (ECE);
- multi-step forecast NLL and latent rate RMSE by horizon `H ∈ {1,2,4,8}`;
- correlation-head joint-event NLL and pairwise correlation RMSE (including unseen gate-concurrency schedules);
- leakage PR-AUC, F1, precision, recall, detection delay (in rounds), and persistence calibration;
- chip-wide event PR-AUC, F1, detection delay, and spatial-footprint RMSE;
- disentanglement of coexisting noise mechanisms in mixed-noise trajectories.

Compare against all classical noise estimators (`B0` static independent, `B1` static correlated, `B2` Markov/HMM, `B3` schedule/spatial variational tracker, `B4` AR(1)/OU filter) and the `B5` Oracle diagnostic bound on identical trajectories.

## 10. Identifiability limits

Syndrome-only observations may not uniquely distinguish coherent ZZ, stochastic correlated faults, leakage-induced syndrome scrambling, and common-mode relaxation bursts. Conditioning on the gate schedule `u_t` and incorporating calibration/analog readout channels (`c_t, a_t`) improves identifiability, but remaining observational ambiguity must be preserved in the posterior uncertainty.

## 11. Acceptance gates for the noise-generation and estimation phase

1. Freeze benchmark specifications and parameter provenance (`Benchmark-Evals/Qutip-Baseline/configs/benchmark_configs.json`).
2. Validate QuTiP physical-reference models across all five benchmarks (`A1, A2, B1, C1, C2`).
3. Verify schema compliance and zero-leakage trajectory-level dataset splits.
4. Establish reproducible classical baseline (`B0–B5`) metrics across isolated and mixed-noise trajectories (`Benchmark-Evals/Qutip-Baseline/evidance/`).
5. Train and evaluate the graph-aware RSSM against `B0–B4` on held-out and out-of-distribution noise trajectories before any downstream decoder integration is considered.
