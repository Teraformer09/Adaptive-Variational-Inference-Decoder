# AVID — Noise Generation and World-Model Integration

This branch contains physical-reference noise experiments under `Limiation_sim/` and the design contract for a graph-aware recurrent state-space world model under `world_model/`.

## Current scope

The world model is a **predictive latent-state estimator** for repeated-round superconducting transmon QEC. It estimates hidden, time-varying noise conditions from classical observations available to the controller. It is not a quantum-state simulator, an RL policy, or a direct pulse controller.

Target operating context: repeated surface-code syndrome extraction on a tunable-coupler transmon layout. No bosonic storage cavity is assumed.

## Components

- `Limiation_sim/`: small-system QuTiP exploratory models and simulator-limit investigations. These are not all reproductions of the cited experimental papers and should not be described as measured hardware benchmarks.
- `world_model/README.md`: architecture, model equations, observation/latent-state contract, objectives, evaluation and AVID integration boundary.
- `world_model/dataset/trajectory.schema.json`: starter schema for trajectory records, including a strict separation between inference observations and simulator-only ground truth.

## Data and model flow

```text
QuTiP / quantumsim physical references
                 |
                 v
Custom stochastic trajectory/noise engine
                 |
                 v
Stim + leakage-aware repeated-QEC wrapper
                 |
        observation stream + truth sidecar
                 |
                 v
Graph encoder + recurrent state-space model
                 |
      posterior belief + forecast distributions
                 |
         read-only AVID shadow evaluation
                 |
      validated adapter (future work only)
```

QuTiP remains a small-system physical reference, not the large-code QEC execution engine. Stim handles stabilizer-circuit sampling and Pauli-compatible channels; a custom wrapper is required for trajectory-dependent leakage and noise processes. `quantumsim` is the candidate for reproducing the published Surface-17 leakage benchmark before claiming that reproduction.

## Important interpretation rules

1. Keep physical dynamics, effective QEC fault processes, and detector/readout observation models as separate layers.
2. Do not label a state-projection probability as a Pauli fault probability without a defined preparation–circuit–measurement protocol and a justified channel reduction.
3. Treat hand-selected parameters and generated IQ/flux/quasiparticle/latency plots as synthetic or illustrative unless sourced to a device measurement or a published dataset.
4. Do not claim the RSSM uniquely identifies microscopic noise causes from syndrome data. Several causes can be observationally confounded.
5. Preserve the existing AVID HMM/variational estimator as a baseline. The world model first runs read-only in shadow mode; no decoder-weight or FPGA-register updates are enabled by this design document.

## Status

The world-model files in this integration package are specifications and schemas, not a trained or implemented neural model. No claim of predictive performance is made. The initial acceptance gates are documented in `world_model/README.md`.
