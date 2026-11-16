# AVID Layer 5 — 250 MHz SystemVerilog FPGA RTL Decoder

## 1. Layer Work Description

Branch `layer5-fpga-rtl-decoder` isolates **Layer 5** of the Adaptive Variational Inference Decoder (AVID): the synthesizable 250 MHz ($4.0\text{ ns}$ clock) SystemVerilog FPGA fast-loop belief-propagation (BP) and ordered-statistics (OSD-0) surface-code decoder (`aa_qec_top`).

Core implemented and verified hardware modules in this layer:
- **Pipelined Q7.8 Min-Sum BP & OSD-0 Core (`firmware/rtl/decoder/`)**: `bp_engine.sv` and `check_node_minsum.sv` implement deterministic $d=3$ rotated CSS decoding with exact core latency $L_{\text{core}}(K) = 2 + 5K$ cycles ($17\text{ cycles} = 68.0\text{ ns}$ at $K=3$, well inside the $300\text{ ns}$ fast-loop budget) and three configurable `cfg_osd_mode` paths (`Mode 0` pure 9-marginal BP, `Mode 1` inline OSD-0 in 17 cycles, `Mode 2` post-BP OSD-0 in 19 cycles).
- **AXI4-Stream Ingress/Egress & Atomic Weight-Bank Registers (`firmware/rtl/ingress/`, `firmware/rtl/config/`)**: `axis_syndrome_rx.sv` and `axis_correction_tx.sv` stream `(s_x[3:0], s_z[3:0])` to `(corr_x[8:0], corr_z[8:0])` in $L_{\text{axis}}(K) = 7 + 5K$ total cycles with full `tready` backpressure and selective `tlast` `epoch_irq` assertion. `weight_map_reg.sv` provides AXI4-Lite shadow Q7.8 X/Z weight registers (`0x000..0x048`), `bp_ready_in`-gated atomic shadow-to-active bank commits, busy-lock SLVERR protection, and timing constraints (`firmware/constraints/aa_qec.xdc`).

## 2. Run Script to Validate & Generate Results + Plots

Compile `aa_qec_top` with native `Verilator --assert` (or C++ cycle-exact fallback when Verilator is absent), execute the standalone M1 RTL verification matrix + IP/OP pytest suite, and generate plots from the repository root:

```bash
python3 -m pip install --user -q numpy pytest matplotlib
bash scripts/run_layer5.sh
```

To regenerate `results/layer5_summary.json` and `results/plots/layer5_validation_summary.png` from the saved RTL CSV/JSON artifacts:

```bash
PYTHONPATH=. python3 scripts/plot_layer5_results.py
```

Generated outputs:
- Validation log: `logs/layer5_validation.log`
- Cycle & RTL artifacts: `results/rtl_cycle_matrix.csv`, `results/m1_standalone.json`, `results/layer5_summary.json`
- Diagnostic plot: `results/plots/layer5_validation_summary.png`

![Layer 5 Validation Summary](results/plots/layer5_validation_summary.png)

## 3. Validation Results

| RTL Verification Check | Hardware Mechanism Tested | Simulator / Mode | Expected | Measured | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Deterministic Cycle Law ($K=1..5$) | `bp_engine.sv` + AXI4-Stream RX/TX | `Verilator 5.048 (--assert)` | Core `{7,12,17,22,27}`; AXI `{12,17,22,27,32}` | Core `{7,12,17,22,27}`; AXI `{12,17,22,27,32}` | **PASS** |
| Nominal $K=3$ Fast-Loop Budget | $4.0\text{ ns}$ (`250 MHz`) decode latency | `Verilator 5.048 (--assert)` | $\le 300.0\text{ ns}$ (`17` core / `22` AXI cyc) | `68.0 ns` core / `88.0 ns` AXI total | **PASS** |
| Configurable OSD-0 Modes ($K=3$) | `cfg_osd_mode` `0`, `1`, `2` (`0x058`) | `Verilator 5.048 (--assert)` | `{17, 17, 19}` core cycles | `{17, 17, 19}` core cycles (`68, 68, 76 ns`) | **PASS** |
| Basis Syndrome Satisfaction | All non-zero 4-bit $s_X, s_Z \in \{1..15\}$ | `Verilator 5.048 (--assert)` | `15/15` $X$ & `15/15` $Z$ satisfied | `15/15` $X$ & `15/15` $Z$ satisfied | **PASS** |
| Golden Reference Equivalence | RTL `Vaa_qec_top` vs C++ Q7.8 reference | `Verilator 5.048 (--assert)` | `0` mismatches | `0` mismatches (`1,582` sim cycles) | **PASS** |
| AXI4-Lite & Weight-Bank Safety | Split `AW`/`W`, `WSTRB`, SLVERR, atomic swap, busy lock, torn-write guard | `Verilator 5.048 (--assert)` | `6 / 6` protocol invariants | `6 / 6` verified (`SVA` enabled) | **PASS** |
| AXI4-Stream Backpressure & Reset | 8-beat stall burst, `tlast` `epoch_irq`, mid-decode `aresetn` | `Verilator 5.048 (--assert)` | `4 / 4` stream/reset invariants | `4 / 4` verified + `pytest` `1/1` PASS | **PASS** |
