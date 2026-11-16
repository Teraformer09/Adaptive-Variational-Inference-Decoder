# Layer 3 Acceptance Evaluation Summary (remediated_set1_v2)

- **Run ID**: `remediated_set1_v2` (`mode=remediated`)
- **Source Commit**: `d56747fe57149aba132eeddff499482a23620a14` (remediated from baseline `88df97e86b579b6e2865cc70dc4d0b08225d677b`; original failing run preserved at `validation/layer3/runs/88df97e_set1_baseline/`)
- **Evaluated At (UTC)**: `2026-11-16T04:49:18Z`
- **Gate `G1` (Input Integrity & Contract Conformance)**: **`PASS`** (`52/52` gated PASS + `2/2` diagnostic companions PASS; ASan+UBSan exit code `0`)
- **Gate `G10` (Reproducibility, Traceability, & Branch Isolation)**: **`PASS`**
- **Gates `G2`–`G9`**: **`NOT_EVALUATED_YET`** (halted at mandatory Set 1 review gate)
- **Stage 1 Gate Decision**: **`STAGE1_ACCEPTED_G1_G10_PASS`** (Overall Layer 3: **`PENDING_SETS_2_TO_5`**)

## Gate Table

| Gate | Name | Stage | Status | PASS | Non-PASS |
| :--- | :--- | :--- | :--- | ---: | ---: |
| `G1` | Input Integrity & Contract Conformance | Set 1 | **`PASS`** | 52 | 0 |
| `G2` | Likelihood Exactness & Analytical Gradient Fidelity | Future Sets 2–5 | **`NOT_EVALUATED_YET`** | 0 | 0 |
| `G3` | L-BFGS Optimization & Quotient-Orbit Parameter Recovery | Future Sets 2–5 | **`NOT_EVALUATED_YET`** | 0 | 0 |
| `G4` | Signed Q7.8 LLR Quantization & Bank Bounds Integrity | Future Sets 2–5 | **`NOT_EVALUATED_YET`** | 0 | 0 |
| `G5` | Causal Scheduling & Sub-Epoch Burst Detection | Future Sets 2–5 | **`NOT_EVALUATED_YET`** | 0 | 0 |
| `G6` | Four-State Safety Fallback FSM & Hardware Protection | Future Sets 2–5 | **`NOT_EVALUATED_YET`** | 0 | 0 |
| `G7` | End-to-End Slow-Loop Pipeline Flow & HAL Commit Atomicity | Future Sets 2–5 | **`NOT_EVALUATED_YET`** | 0 | 0 |
| `G8` | Runtime Execution, Memory Zero-Allocation, & Fault Recovery | Future Sets 2–5 | **`NOT_EVALUATED_YET`** | 0 | 0 |
| `G9` | Adaptive Control Efficacy & Non-Regression Over Static Baselines | Future Sets 2–5 | **`NOT_EVALUATED_YET`** | 0 | 0 |
| `G10` | Reproducibility, Traceability, & Branch Isolation Integrity | Set 1 | **`PASS`** | 1 | 0 |

## `G1` Resolution Status

Zero unresolved `G1` items (`52/52` gated items `PASS` + `2/2` diagnostic companions `PASS` + ASan/UBSan clean).
