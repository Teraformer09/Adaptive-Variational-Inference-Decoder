# Layer 3 Acceptance Evaluation Summary (full_layer3_v1)

- **Run ID**: `full_layer3_v1` (`mode=remediated`)
- **Source Commit**: `3a1a6d68b444559dfbfa4508781e10dfe20eda70` (remediated from baseline `88df97e86b579b6e2865cc70dc4d0b08225d677b`; original failing run preserved at `validation/layer3/runs/88df97e_set1_baseline/`)
- **Evaluated At (UTC)**: `2026-11-16T12:58:03Z`
- **Overall Layer 3 Decision**: **`ACCEPTED_ALL_GATES_G1_TO_G10_PASS`**
- **Sanitizer Suite (`ASan+UBSan`)**: exit code `0`

## Gate Table (`G1`–`G10`)

| Gate | Name | Stage | Status | PASS | Non-PASS |
| :--- | :--- | :--- | :--- | ---: | ---: |
| `G1` | Input Integrity & Contract Conformance | Sets 1–5 | **`PASS`** | 52 | 0 |
| `G2` | Output Contracts & 32-Seed Statistical Recovery | Sets 1–5 | **`PASS`** | 51 | 0 |
| `G3` | Complete Ingress-to-Commit Pipeline Dataflow | Sets 1–5 | **`PASS`** | 35 | 0 |
| `G4` | Causal Timing, 20 us Fast Detector vs 10 ms Slow Loop | Sets 1–5 | **`PASS`** | 10 | 0 |
| `G5` | Clean Build, Daemon Lifecycle, Persistence, Faults & Concurrency | Sets 1–5 | **`PASS`** | 45 | 0 |
| `G6` | Independent Joint-DP & L-BFGS Mathematical Oracles | Sets 1–5 | **`PASS`** | 24 | 0 |
| `G7` | Parameter Identifiability, Jacobian Rank & Fisher Information | Sets 1–5 | **`PASS`** | 8 | 0 |
| `G8` | Q7.8 LLR Reference & Causal Scheduler Mathematics | Sets 1–5 | **`PASS`** | 20 | 0 |
| `G9` | Four-State Safety FSM Exhaustive Verification | Sets 1–5 | **`PASS`** | 10 | 0 |
| `G10` | Reproducibility, Traceability, & Branch Isolation Integrity | Sets 1–5 | **`PASS`** | 243 | 0 |

## Resolution Status

Zero unresolved items across all evaluated gates and ASan+UBSan suites.
