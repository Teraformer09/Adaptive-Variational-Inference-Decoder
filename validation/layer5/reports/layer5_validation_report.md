# AVID Layer 5 — 250 MHz SystemVerilog FPGA RTL Validation Report

- **Run ID**: `remediated_full_layer5_v1`
- **Branch / Commit**: `layer5-fpga-rtl-decoder` @ `fc5b756` (`code_git_dirty=false`)
- **Simulator**: `Verilator 5.048 2026-04-26 rev vUNKNOWN-built20260429` (`--cc --trace --assert`, native `Vaa_qec_top`)
- **Canonical Test IDs**: `200 / 200 PASS` (`100.0%`)
- **Acceptance Gates (`G1`–`G8`)**: `8 / 8 PASS`
- **Total Simulated RTL Cycles**: `617,521` cycles (`615,945` across Sets 1–5 + `1,576` standalone M1 TB)
- **Overall Verdict**: **`PASS`**

## 1. Five-Set Validation Summary

| Set | Scope | Canonical Tests | Passed | Failed | Verilator RTL Cycles | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Set1` | Input, AXI4-Lite & AXI4-Stream Ingress Contracts (`L5-I*`, `L5-AXIL-*`) | `44` | `44` | `0` | `2,868` | **PASS** |
| `Set2` | Output, Deterministic Cycle Law & Stream Contracts (`L5-CYC-*`, `L5-O*`) | `38` | `38` | `0` | `35,989` | **PASS** |
| `Set3` | Pipeline, Non-Blocking Ingress Tap & Atomic Weight-Bank Flow (`L5-FLOW-*`) | `36` | `36` | `0` | `17,000` | **PASS** |
| `Set4` | Execution, Backpressure, Reset, 10k Stress & Provenance (`L5-EXEC-*`, `L5-FLT-*`) | `40` | `40` | `0` | `384,457` | **PASS** |
| `Set5` | Q7.8 Min-Sum, OSD-0, Formal SVA & Constraints Verification (`L5-MATH-*`, `L5-HW-*`) | `42` | `42` | `0` | `175,631` | **PASS** |
| **Total** | **All 5 Sets + Standalone M1 TB** | **`200`** | **`200`** | **`0`** | **`617,521`** | **PASS** |

## 2. Acceptance Gate Evaluation (`G1` – `G8`)

| Gate | Invariant Name | Cases Checked | Passed | Status |
| :---: | :--- | :---: | :---: | :---: |
| `G1` | AXI4-Lite & AXI4-Stream protocol compliance | `57` | `57` | **PASS** |
| `G2` | Deterministic cycle law & fast-loop budget | `57` | `57` | **PASS** |
| `G3` | Non-blocking ingress tap & pipeline isolation | `22` | `22` | **PASS** |
| `G4` | Atomic triple-buffered weight-bank & epoch safety | `30` | `30` | **PASS** |
| `G5` | Elastic FIFO backpressure, reset recovery & stress | `45` | `45` | **PASS** |
| `G6` | Q7.8 Min-Sum, OSD-0 & GF(2) mathematical equivalence | `44` | `44` | **PASS** |
| `G7` | Formal SVA verification & hardware synthesis constraints | `21` | `21` | **PASS** |
| `G8` | Deterministic replay, clean provenance & STA disclosure | `18` | `18` | **PASS** |

## 3. Deterministic 250 MHz Cycle & Fast-Loop Budget Matrix

| BP Cap $K$ | Core (`Modes 0,1`: $2+5K$) | Core (`Mode 2`: $4+5K$) | AXI Total (`Modes 0,1`: $7+5K$) | AXI Total (`Mode 2`: $9+5K$) | Fast-Loop Budget ($\le 300\text{ ns}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `7 cyc (28.0 ns)` | `9 cyc (36.0 ns)` | `12 cyc (48.0 ns)` | `14 cyc (56.0 ns)` | **PASS** |
| `2` | `12 cyc (48.0 ns)` | `14 cyc (56.0 ns)` | `17 cyc (68.0 ns)` | `19 cyc (76.0 ns)` | **PASS** |
| `3` *(Nominal)* | `17 cyc (68.0 ns)` | `19 cyc (76.0 ns)` | `22 cyc (88.0 ns)` | `24 cyc (96.0 ns)` | **PASS** |
| `4` | `22 cyc (88.0 ns)` | `24 cyc (96.0 ns)` | `27 cyc (108.0 ns)` | `29 cyc (116.0 ns)` | **PASS** |
| `5` | `27 cyc (108.0 ns)` | `29 cyc (116.0 ns)` | `32 cyc (128.0 ns)` | `34 cyc (136.0 ns)` | **PASS** |

## 4. Hardware Simulation vs. Vivado Post-Route STA Disclosure

- **RTL Simulation Scope**: Native Verilator 5.048 (--assert --trace) compiles the actual synthesizable SystemVerilog hierarchy (aa_qec_top, axis_ingress_tap, axis_syndrome_rx, weight_map_reg, bp_engine, check_node_minsum, axis_correction_tx, plus bound sva_weight_commit_assertions) into cycle-accurate 2-state C++ logic, proving exact synchronous FSM cycle laws, AMBA AXI4-Lite/AXI4-Stream handshakes, Q7.8 arithmetic, and formal SVA invariants at a nominal 4.000 ns (250.0 MHz) simulation clock.
- **Vivado Post-Route STA Scope**: Cycle-accurate Verilator RTL simulation does NOT perform gate-level netlist synthesis, FPGA placement and routing, wire-delay back-annotation, or post-route Static Timing Analysis (WNS/TNS). Physical 250.0 MHz timing closure on Xilinx Zynq-7020 (xc7z020clg484-1) or UltraScale+ ZCU104/K26 requires Vivado synthesis and post-route STA against firmware/constraints/aa_qec.xdc.
