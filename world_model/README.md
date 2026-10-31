# AVID World Model — Architecture and Integration Contract

**Status:** design specification; the RSSM described here is not yet implemented or trained.  
**Target:** repeated-round surface-code experiments on superconducting transmon hardware with tunable couplers.  
**Scope:** latent noise-state estimation and short-horizon forecasting. No RL policy or autonomous pulse control is included in this phase.

## 1. Role in AVID

The world model is a predictive state estimator for a partially observed, time-varying hardware-noise process. It receives only classical information available to the control/decoding stack: syndrome or detection-event history, known circuit schedule, calibration telemetry, device topology, and optional analog readout features. Simulator-only hidden variables are training/evaluation labels, never ordinary inference inputs.

It estimates a posterior belief over latent environmental/noise state and predicts future observable statistics. The existing AVID HMM/variational inference path remains the baseline and fallback until the world model demonstrates better held-out predictive calibration.

## 2. Observation contract

For QEC round `t`, define

\[
o_t = (d_t,s_t,c_t,a_t,u_t,G),
\]

where:

- `d_t`: sparse detection events or a binary detector vector;
- `s_t`: optional raw stabilizer outcomes, kept separate from detection events;
- `c_t`: calibration telemetry with timestamps and validity masks (e.g. measured T1/T2, readout assignment statistics, gate RB estimates, qubit/coupler frequencies);
- `a_t`: optional analog readout features (IQ summaries or calibrated discriminator probabilities; raw IQ is not assumed available);
- `u_t`: known schedule features (gate type, targets, couplers, simultaneous-gate group, idle/measurement/reset windows and elapsed physical time);
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

## 3. Structured latent state

Use typed latent components rather than one uninterpretable noise vector:

- `z_corr`: local correlated interaction/crosstalk regime and edge-level latent variables;
- `z_leak`: per-transmon leakage/seepage belief;
- `z_drift`: slow calibration/noise drift;
- `z_memory`: temporal-memory variables for TLS-like or colored-noise processes;
- `z_event`: sparse common-mode event state and spatial footprint.

Simulator-only truth may contain exact Hamiltonian parameters, qutrit populations, injected event locations, leakage flags, TLS state, and injected effective fault causes. Keep this in a sidecar, never in the inference feature tensor.

## 4. Model family: graph-aware RSSM

Let `G=(V,E)` denote the hardware/check graph. Encode node, edge, observation and schedule features with message passing:

\[
g_t = \operatorname{GNN}_{\phi}(G, X_t, E_t).
\]

Maintain deterministic recurrent memory `h_t` and stochastic latent state `z_t`:

\[
h_t = f_{\theta}(h_{t-1},z_{t-1},g_t),
\]
\[
q_{\phi}(z_t\mid h_t,g_t) \quad\text{(posterior)}, \qquad
p_{\theta}(z_t\mid h_t) \quad\text{(prior)}.
\]

At each round, the prior predicts before seeing the new observation; the posterior incorporates the observation. Forecast distributions are then formed from the posterior belief and known future schedule:

\[
p(o_{t+1:t+H}\mid b_t,u_{t+1:t+H}),
\]

where `b_t` denotes the inferred belief. Initial forecast horizons may be `H ∈ {1,2,4,8}` rounds as experimental settings, not physical constants.

Use sparse local/edge correlation heads. A dense all-pairs covariance is unnecessary and scales poorly. Condition crosstalk predictions on the simultaneous gate schedule; physical coherent interactions and effective stochastic QEC faults must remain distinct representations.

## 5. Prediction heads

1. **Detection head:** per-check detection-event probabilities and calibrated uncertainty.
2. **Correlation head:** sparse edge/local joint-event probabilities or pairwise log-odds conditioned on schedule.
3. **Leakage head:** per-transmon leakage/seepage probability and persistence forecast. This is an inference, not a direct observation of the true leakage flag.
4. **Drift/memory head:** posterior and forecast distributions for slow drift and temporally correlated latent components.
5. **Calibration head:** predictive distributions for future measured calibration values, with missingness handled explicitly.
6. **Common-event head:** event probability and spatial-footprint distribution when that process is enabled in the simulator.

A logical-failure head is not required for the first stage. Add it only after observable prediction is validated.

## 6. Training objective

Use a multi-task sequential variational objective:

\[
\mathcal L = \lambda_{obs}\mathcal L_{obs}
+ \beta\,D_{KL}\!\left[q_{\phi}(z_t\mid h_t,g_t)\,\|\,p_{\theta}(z_t\mid h_t)\right]
+ \lambda_{aux}\mathcal L_{aux}
+ \lambda_{forecast}\mathcal L_{forecast}.
\]

- `L_obs`: proper negative log-likelihoods for binary detections, categorical/readout outcomes and continuous calibration features. Apply masks to absent channels.
- `KL`: posterior-to-prior dynamics regularization. Monitor posterior collapse; document any KL annealing/free-bits strategy.
- `L_aux`: simulator-only supervision for hidden causes (leakage, injected common events, known drift/TLS state and effective correlated-fault labels). These targets must not appear in inference inputs.
- `L_forecast`: multi-step predictive likelihood/calibration across the chosen horizons.

Select loss weights and schedules on validation trajectories. Report per-head metrics; a single aggregate loss is not sufficient.

## 7. Simulator responsibilities

| Layer | Tool | Responsibility | Explicit limitation |
|---|---|---|---|
| Physical reference | QuTiP / QuTiP-QIP | Small-system coherent ZZ, qutrit leakage, and explicitly modelled qubit–TLS dynamics | Not the repeated-round large-code simulator |
| Published leakage reproduction | quantumsim | Reproduce the Surface-17 leakage protocol and settings before claiming paper reproduction | Do not assume equivalence with the current QuTiP pulse toy model |
| Stabilizer circuit | Stim | Scalable circuit sampling and Pauli-compatible effective channels | Not a general qutrit/coherent/non-Markovian dynamics engine |
| Trajectory augmentation | Custom engine + leakage-aware wrapper | Evolve latent drift/TLS/event/leakage processes and inject validated effective channels into the stabilizer simulation | Every approximation must be validated against the physical reference cases |

Keep three layers separate in code and data: (1) physical process, (2) effective QEC fault/trajectory model, and (3) detector/readout observation model.

## 8. Dataset and leakage prevention

Each sample is a contiguous window from a complete trajectory and carries a stable `trajectory_id`, simulator seed, device/topology version, simulator/config version, and round range. Split train/validation/test by complete trajectories or seeds—not by individual rounds—to prevent adjacent temporal windows from leaking across splits.

See `dataset/trajectory.schema.json` for the starter schema. Keep `latent_truth` as a separate simulator-sidecar field and explicitly exclude it from inference serialization.

## 9. Evaluation protocol

Report each noise family separately and in mixed-noise conditions:

- detection predictive NLL, Brier score and reliability calibration;
- multi-step NLL/calibration by forecast horizon;
- correlation-head joint-event NLL and sparse correlation-matrix error;
- leakage PR-AUC, precision, recall, F1, detection delay and persistence calibration;
- drift forecast error and uncertainty-interval coverage;
- generalization to unseen seeds, trajectories, drift rates, burst sizes and schedule patterns.

Compare against (a) static independent-Pauli noise, (b) the current AVID HMM/variational baseline, and (c) an RSSM without graph message passing. Use the same generated trajectories and observation budget. Ablate graph structure, recurrence, auxiliary labels and each observation channel independently.

## 10. Integration boundary

The first integration is read-only shadow mode: log world-model beliefs and forecasts beside current AVID estimates. Do not let unvalidated predictions update decoder weights, the existing 21-parameter estimator, or FPGA registers.

After held-out trajectory forecasting, uncertainty calibration and out-of-distribution tests pass, implement a narrow adapter to the existing AVID parameter vector. Map each output explicitly; document parameters that are not identifiable from the available observations. Keep the current HMM/variational estimator available as baseline and fallback.

## 11. Identifiability limits

Syndrome-only observations may not uniquely distinguish coherent ZZ, stochastic correlated faults, leakage-induced readout bias and common-mode bursts. Conditioning on schedule and adding calibration/readout features can improve identifiability, but ambiguity must remain represented in posterior uncertainty. The RSSM does not reconstruct the full quantum state and must not be described as identifying a unique microscopic cause from syndrome data alone.

## 12. Acceptance gates before decoder integration

1. Validate the trajectory schema and deterministic seeded trajectory generation.
2. Validate physical-reference cases and compare effective channels using process/observable metrics—not one projection probability.
3. Demonstrate held-out trajectory forecasts against static and current AVID/HMM baselines.
4. Verify uncertainty calibration and out-of-distribution behaviour.
5. Run read-only shadow evaluation beside AVID.
6. Only then define a decoder adapter. Policy learning and autonomous hardware control are outside this phase.
