# AVID — Noise Generation and World-Model Specification (`Noise-Generation_Model`)

This branch is scoped strictly to **quantum hardware noise modeling, multi-round noise trajectory generation, classical baseline noise estimation, and the predictive world-model contract**.

> **Current Phase Scope:** Noise generation and latent noise-state estimation/forecasting only. **Downstream QEC decoding, decoder weight adaptation, RL control policies, and FPGA decoder hardware are out of scope for this branch.**

---

## Repository Structure

```text
├── README.md                                  # Branch overview, data flow, and scope rules
├── Limiation_sim/                             # Small-system QuTiP exploratory models & simulator-limit tests
│   ├── INTERPRETATION_NOTES.md                # Methodological scope & provenance notes for QuTiP tests
│   ├── README.md                              # Documentation of the 3 noise families & QuTiP scaling limits
│   ├── verify_qutip_noise_limitations.py      # 3-test QuTiP physical & scaling verification script
│   ├── verification_results.log               # Recorded output from the 3 QuTiP tests
│   ├── generate_limitation_plots.py           # Plot generator for the 6 QuTiP limitation figures
│   └── 01..06_*.png                           # 300-DPI exploratory & illustrative figures
├── world_model/                               # Graph-aware RSSM noise-estimator specification
│   ├── README.md                              # Architecture, observation/latent contract & training objective
│   └── dataset/
│       └── trajectory.schema.json             # JSON schema separating observations from latent_truth
└── Benchmark-Evals/                           # 5-benchmark noise generation & baseline estimation suite
    ├── README.md                              # Specification of the 5 benchmarks (A1, A2, B1, C1, C2) & pipeline
    └── Qutip-Baseline/                        # Reproducible QuTiP-only noise generation & B0..B5 baseline harness
        ├── configs/
        │   └── benchmark_configs.json         # Stage 0 frozen benchmark definitions & parameter provenance
        ├── qutip_baseline_pipeline.py         # Isolated & mixed-noise generator + B0..B5 baseline evaluator
        ├── generate_benchmark_plots.py        # 300-DPI plot generator for the 5 noise benchmarks
        └── evidance/                          # Generated trajectory samples, metrics JSON, logs & figures
```

---

## Data and Noise-Generation Flow

```text
QuTiP / quantumsim physical reference models (A1, A2, B1, C1, C2)
                             |
                             v
      Custom stochastic noise trajectory generator
  (spatial crosstalk, chip-wide events, leakage, TLS & idle memory)
                             |
                             v
        Stim + leakage-aware repeated-circuit wrapper
                             |
         observable stream o_t  +  latent_truth sidecar x_t
                             |
                             v
    Classical noise baselines (B0..B5) & Graph-aware RSSM world model
                             |
                             v
     Posterior noise belief b_t + multi-horizon noise forecasts
                  p(o_{t+1:t+H} | b_t, u_{t+1:t+H})
```

QuTiP serves as the small-system physical reference engine (2–5 transmons/qutrits or composite qubit–TLS baths), not the large-code execution engine. `Stim` and a custom trajectory wrapper handle scalable stabilizer-circuit sampling and trajectory-dependent noise injection, while `quantumsim` is the reference target for full Surface-17 density-matrix leakage reproduction.

---

## Core Interpretation and Scope Rules

1. **Noise Generation & Estimation Only:** Do not conflate noise trajectory estimation with downstream QEC decoding. A noise model is evaluated here by whether it generates physically defensible observables and whether estimators accurately infer and forecast the hidden noise process (`o_{t+1:t+H}` and latent components).
2. **Three Separate Layers:** Keep (a) microscopic physical dynamics, (b) effective circuit fault/trajectory processes, and (c) detector/readout observation models strictly separated in code and schemas.
3. **Channel Reduction Before Pauli Labeling:** Do not label a coherent state-projection probability as an effective Pauli fault probability without an explicit Pauli twirl or process-matrix reduction.
4. **Strict Parameter & Plot Provenance:** Distinguish published benchmark settings from synthetic/illustrative controlled parameters, and never pass simulator-only `latent_truth` into inference models (`B0–B4` or the world model).
5. **Observational Confounding:** Multiple microscopic noise mechanisms can produce similar syndrome statistics; estimators must represent this ambiguity via calibrated posterior uncertainty rather than claiming unique microscopic identification from syndromes alone.
