# Layer 5 — Implementation-to-Test Traceability & Hardware STA Fidelity Matrix

## 1. Scope & RTL Hierarchy (`layer5-fpga-rtl-decoder`)

Layer 5 isolates the synthesizable **250.0 MHz ($T_{\text{clk}} = 4.000\text{ ns}$) SystemVerilog FPGA fast-loop belief-propagation (BP) and ordered-statistics (OSD-0) surface-code decoder** (`aa_qec_top`) for the $d = 3$ rotated CSS surface code ($N_{\text{data}} = 9, N_{\text{check},X} = 4, N_{\text{check},Z} = 4$).

| Hardware / Verification Artifact | Path | Functional Responsibility |
| :--- | :--- | :--- |
| Top-Level Integration | `firmware/rtl/top/aa_qec_top.sv` | Integrates `axis_ingress_tap`, `axis_syndrome_rx`, `weight_map_reg`, dual `bp_engine` (`u_bp_x`, `u_bp_z`), `axis_correction_tx`, and 1-cycle `epoch_irq` generator. |
| Non-Blocking Syndrome Tap | `firmware/rtl/ingress/axis_ingress_tap.sv` | Zero-cycle combinational fast-path pass-through + non-blocking slow DMA tap with `dma_drop_count` and sticky bit-9 `overflow_warn`. |
| Elastic Syndrome RX FIFO | `firmware/rtl/ingress/axis_syndrome_rx.sv` | 16-deep synchronous AXI4-Stream slave FIFO (`s_axis_tready = aresetn && !fifo_full`), unpacking `s_x[3:0]` and `s_z[7:4]`. |
| AXI4-Lite Weight Registry | `firmware/rtl/config/weight_map_reg.sv` | Triple-buffered Q7.8 registry (`shadow_x/z -> commit_snap_x/z -> active_x/z`), split `AW`/`W`, `WSTRB`, `bp_ready_in`-gated 1-cycle swap, SLVERR busy-lock, and embedded SVAs. |
| Min-Sum BP + OSD-0 Core | `firmware/rtl/decoder/bp_engine.sv` | Deterministic $L_{\text{core}}(K) = 2 + 5K$ cycles (`Modes 0, 1`) and $4 + 5K$ cycles (`Mode 2`) signed 16-bit Q7.8 decoder with cycle-count SVAs. |
| Pipelined Check Node PE | `firmware/rtl/decoder/check_node_minsum.sv` | 3-stage shift-add Normalized Min-Sum check-node processor ($\alpha = 0.75$ non-min, $\min_2 + 0.5\min_1$ min1, zero DSP48E1 slices). |
| Elastic Correction TX FIFO | `firmware/rtl/decoder/axis_correction_tx.sv` | 16-deep synchronous AXI4-Stream master FIFO packing `{14'b0, corr_z[8:0], corr_x[8:0]}` with `corr_ready` watermark (`FIFO_DEPTH - 2 = 14`). |
| Bound Formal SVA Module | `firmware/formal/assertions/sva_weight_commit_assertions.sv` | Concurrent `p_no_swap_while_busy` property (`(!bp_ready) \|=> $stable(bank_sel)`) bound directly onto `weight_map_reg` under `Verilator 5.048 --assert`. |
| Timing Constraints (XDC) | `firmware/constraints/aa_qec.xdc` | Active primary fabric clock constraint `create_clock -period 4.000 -name pl_clk0 [get_ports aclk]` (`250.0 MHz`). |
| Cycle-Stepped Verilator Driver | `emulator/verilator/tb/verilator_rtl_driver.h` | Native `Vaa_qec_top` + `VerilatedVcdC` cycle-accurate C++ harness driving AXI4-Lite & AXI4-Stream pins. |

---

## 2. Canonical AXI4-Lite Register Map (`0x43C0_1000`)

| Byte Offset | Register Name | Access | Reset Default | Semantics & Error Handling |
| :--- | :--- | :---: | :---: | :--- |
| `0x000 .. 0x020` | `SHADOW_X[0..8]` | `RW` | `0x0232` (`+562`) | Signed Q7.8 X-error channel LLRs; locked with `SLVERR` (`2'b10`) while `swap_pending && !bp_ready_in`. |
| `0x024 .. 0x044` | `SHADOW_Z[0..8]` | `RW` | `0x0232` (`+562`) | Signed Q7.8 Z-error channel LLRs; locked with `SLVERR` (`2'b10`) while `swap_pending && !bp_ready_in`. |
| `0x048` | `COMMIT` | `RW` | `0x0` | `wdata[0]=1 & wstrb[0]=1` snapshots `shadow -> commit_snap` & sets `swap_pending=1`; `wdata=0` is safe no-op; locked (`SLVERR`) while `swap_pending && !bp_ready_in`. |
| `0x04C` | `STATUS` | `RO` | `0x0` | `[0] = swap_pending`, `[1] = !bp_ready_in` (`bp_busy`); writes return `SLVERR`. |
| `0x050` | `EPOCH_COUNT` | `RO` | `0x0` | Monotonic 32-bit committed weight-bank swap counter; writes return `SLVERR`. |
| `0x054` | `CFG_MAX_ITER` | `RW` | `0x3` | Runtime BP iteration cap $K \in \{1, 2, 3, 4, 5\}$; values outside `1..5` or `wstrb[0]==0` return `SLVERR`. |
| `0x058` | `CFG_OSD_MODE` | `RW` | `0x1` | `0` = Pure 9-Marginal BP (`2+5K`), `1` = Inline OSD-0 (`2+5K`), `2` = Post-BP OSD-0 (`4+5K`); values `> 2` return `SLVERR`. |
| `0x05C .. 0x0FC` | *(Unmapped Gap)* | — | — | Reads (`RDATA=0xDEADBEEF`) and writes rejected with `SLVERR` (`2'b10`). |
| `0x100 .. 0x120` | `ACTIVE_X[0..8]` | `RO` | `0x0232` (`+562`) | Read-only mirror of active X-error Q7.8 bank feeding `u_bp_x`; writes return `SLVERR`. |
| `0x124 .. 0x144` | `ACTIVE_Z[0..8]` | `RO` | `0x0232` (`+562`) | Read-only mirror of active Z-error Q7.8 bank feeding `u_bp_z`; writes return `SLVERR`. |
| `>= 0x148` or unaligned | *(Unmapped / Unaligned)* | — | — | All unaligned (`addr[1:0] != 0`) and upper offsets rejected with `SLVERR` (`2'b10`). |

---

## 3. Remediated RTL, SVA & Contract Defects (`DEF-L5-01` .. `DEF-L5-08`)

1. **`DEF-L5-01` (AMBA Reset `*ready` Gating)**: Gated `s_axil_awready`, `s_axil_wready`, and `s_axil_arready` in `weight_map_reg.sv` and `s_axis_tready` in `axis_syndrome_rx.sv` with `aresetn && ...` so no AXI4-Lite or AXI4-Stream slave asserts ready during active reset.
2. **`DEF-L5-02` (Strict Full-Word `CFG_MAX_ITER` Range Guard)**: Tightened `0x054` write validation in `weight_map_reg.sv` from bit-sliced `wr_data_c[3:0]` to full 32-bit `wr_strb_c[0] && wr_data_c >= 32'd1 && wr_data_c <= 32'd5` (preventing upper-bit aliasing such as `0x13 -> 3`) and clamped `active_max_iter` in `bp_engine.sv` to `1..5`.
3. **`DEF-L5-03` (`axis_ingress_tap` Top-Level Integration)**: Instantiated `axis_ingress_tap` (`u_ingress_tap`) inside `aa_qec_top.sv` ahead of `u_syndrome_rx` and wired `dma_tap_stall_in`, `dma_tap_tdata_mon`, `dma_tap_tvalid_mon`, `dma_tap_drop_count_mon`, and `rx_epoch_count_mon`.
4. **`DEF-L5-04` (Concurrent SVA Preponed Implication & Live Verilator Binding)**: Fixed `sva_weight_commit_assertions.sv` from overlapping `|->` (which falsely flagged legal idle swaps coinciding with `ST_IDLE -> ST_INIT` on the same posedge) to non-overlapping `(!bp_ready) |=> $stable(bank_sel)` and bound `u_sva_weight_commit` directly onto `weight_map_reg` under `Verilator 5.048 --assert`.
5. **`DEF-L5-05` (Cycle-Exact Unified Verilator Driver)**: Ported remediated `verilator_rtl_driver.h` and `standalone_rtl_tb.cpp` with aligned reset cycles, split `AW`/`W` helpers, async mid-decode MMIO stepping, and `--require-native-rtl` enforcement.
6. **`DEF-L5-06` (Hardware YAML Contract Alignment)**: Corrected swapped `0x04C`/`0x054` entries and added `0x100..0x144` RO banks in `memory_map.yaml`, updated triple-bank SLVERR semantics in `weight_commit_protocol.yaml`, documented exact `2` ingress + `3` egress cycles in `system_timing.yaml`, and recorded `Verilator 5.048` in `target_platforms.yaml`.
7. **`DEF-L5-07` (Active XDC Clock & Testbench Port Completeness)**: Enabled active `create_clock -period 4.000 -name pl_clk0 [get_ports aclk]` in `aa_qec.xdc` and remediated stale module/port bindings across `tb_aa_qec_top.sv`, `tb_bp_latency.sv`, and `tb_weight_commit.sv`.
8. **`DEF-L5-08` (Audit-Grade 5-Set RTL Validation & Gate Evaluator)**: Authored `200` canonical `L5-*` tests across 5 sets executing against native `Vaa_qec_top` (`--assert --trace`) with automated `G1`–`G8` gate evaluation and clean-tree provenance auditing.

---

## 4. Explicit Hardware Fidelity & Vivado Post-Route STA Disclosure

- **What Native Verilator 5.048 Proves in Layer 5**: Compiling the synthesizable SystemVerilog hierarchy (`aa_qec_top.sv`, `axis_ingress_tap.sv`, `axis_syndrome_rx.sv`, `weight_map_reg.sv`, `bp_engine.sv`, `check_node_minsum.sv`, `axis_correction_tx.sv`, and bound `sva_weight_commit_assertions.sv`) with `verilator --cc --trace --assert` proves **exact synchronous cycle counts** ($L_{\text{core}}(K) = 2 + 5K$, $L_{\text{axis}}(K) = 7 + 5K$), **AMBA AXI4-Lite / AXI4-Stream protocol handshakes**, **16-deep elastic FIFO flow control**, **signed 16-bit Q7.8 Min-Sum + OSD-0 bit-equivalence**, and **formal SVA invariants** at a nominal $4.000\text{ ns}$ ($250.0\text{ MHz}$) clock.
- **What Requires Xilinx Vivado Synthesis & Post-Route STA**: Cycle-accurate Verilator RTL simulation does **not** perform FPGA technology mapping, placement, routing, interconnect RC delay extraction, or post-route Static Timing Analysis (**WNS / TNS**). Physical $250.0\text{ MHz}$ timing closure (verifying $\text{WNS} \ge 0.000\text{ ns}$ and $\text{TNS} = 0.000\text{ ns}$) and exact post-place-and-route LUT/FF/BRAM counts on Xilinx Zynq-7020 (`xc7z020clg484-1`) or Zynq UltraScale+ (`ZCU104` / `Kria K26`) require running Xilinx Vivado synthesis and post-route STA against `firmware/constraints/aa_qec.xdc`.

---

## 5. Canonical 200-Test Traceability Matrix (`Set 1` – `Set 5`, Gates `G1` – `G8`)

### Set 1 — Input, AXI4-Lite & AXI4-Stream Ingress Contracts (`44` Tests)

| Canonical ID | Set | Gates | Executable / Script | Verification Objective |
| :--- | :---: | :---: | :--- | :--- |
| `L5-I01` | `Set1` | `G1` | `test_axis_ingress_contract` | Single-beat `s_axis_syndrome` handshake accepted in 1 cycle with `tready=1` and `tlast=1`. |
| `L5-I02` | `Set1` | `G1` | `test_axis_ingress_contract` | Non-epoch `s_axis_syndrome` beat (`tlast=0`) accepted cleanly without asserting `epoch_irq`. |
| `L5-I03` | `Set1` | `G1` | `test_axis_ingress_contract` | Holding `s_axis_syndrome_tvalid=0` across 64 clock cycles keeps decoder idle with zero spurious outputs. |
| `L5-I04` | `Set1` | `G1,G6` | `test_axis_ingress_contract` | `tdata[3:0]` routes `s_x` exclusively to Z-error BP engine (`corr_x==0` across all 16 `sx` values). |
| `L5-I05` | `Set1` | `G1,G6` | `test_axis_ingress_contract` | `tdata[7:4]` routes `s_z` exclusively to X-error BP engine (`corr_z==0` across all 16 `sz` values). |
| `L5-I06` | `Set1` | `G1` | `test_axis_ingress_contract` | Combined `tdata[7:0]=(sz<<4)\|sx` produces identical `corr_x`/`corr_z` to separate basis decodes. |
| `L5-I07` | `Set1` | `G1` | `test_axis_ingress_contract` | Non-zero upper flag bits `tdata[15:8]` do not alter fast-path BP output. |
| `L5-I08` | `Set1` | `G1,G5` | `test_axis_ingress_contract` | 8 consecutive single-cycle `s_axis_syndrome` beats ingested at 1 beat/cycle into RX FIFO. |
| `L5-I09` | `Set1` | `G1` | `test_axis_ingress_contract` | Variable inter-packet idle gaps (1, 3, 7 cycles) preserve exact per-packet decode determinism. |
| `L5-I10` | `Set1` | `G1,G5` | `test_axis_ingress_contract` | `s_axis_syndrome_tready` gated to `0` throughout `aresetn==0` per AMBA AXI4-Stream specification. |
| `L5-I11` | `Set1` | `G1,G5` | `test_axis_ingress_contract` | Driving `s_axis_syndrome_tvalid=1` while `aresetn==0` is ignored and leaves RX FIFO empty post-reset. |
| `L5-I12` | `Set1` | `G1,G3` | `test_axis_ingress_contract` | `rx_epoch_count_mon` increments strictly on `tlast=1` syndrome pops and holds steady on `tlast=0`. |
| `L5-I13` | `Set1` | `G1,G2` | `test_axis_ingress_contract` | Zero-syndrome `(0, 0)` executes full 17-cycle core / 22-cycle AXI pipeline and outputs `(0, 0)`. |
| `L5-I14` | `Set1` | `G1,G6` | `test_axis_ingress_contract` | All-ones syndrome `(sx=0xF, sz=0xF)` decodes in 17 core cycles and satisfies both `Hx` and `Hz`. |
| `L5-AXIL-01` | `Set1` | `G1` | `test_axil_register_contract` | Simultaneous `AW+W` write completes in 2 clock cycles with `BRESP=2'b00` (`OKAY`). |
| `L5-AXIL-02` | `Set1` | `G1` | `test_axil_register_contract` | Split `AW`-before-`W` latches `aw_addr_lat` cleanly across 1, 4, and 8 idle cycles. |
| `L5-AXIL-03` | `Set1` | `G1` | `test_axil_register_contract` | Split `W`-before-`AW` latches `w_data_lat` & `w_strb_lat` cleanly across 1, 4, and 8 idle cycles. |
| `L5-AXIL-04` | `Set1` | `G1` | `test_axil_register_contract` | Holding `s_axil_bready=0` holds `s_axil_bvalid=1` & `bresp` stable and gates `awready=wready=0`. |
| `L5-AXIL-05` | `Set1` | `G1` | `test_axil_register_contract` | Holding `s_axil_rready=0` holds `s_axil_rvalid=1`, `rdata`, and `rresp` stable and gates `arready=0`. |
| `L5-AXIL-06` | `Set1` | `G1` | `test_axil_register_contract` | Simultaneous AXI4-Lite `AW+W` write (`0x000`) and `AR` read (`0x024`) complete on the same clock cycle. |
| `L5-AXIL-07` | `Set1` | `G1,G5` | `test_axil_register_contract` | `s_axil_awready`, `s_axil_wready`, and `s_axil_arready` held `0` while `aresetn==0`. |
| `L5-AXIL-08` | `Set1` | `G1` | `test_axil_register_contract` | All 9 `shadow_x[0..8]` registers (`0x000..0x020`) accept distinct 16-bit writes and read back cleanly. |
| `L5-AXIL-09` | `Set1` | `G1` | `test_axil_register_contract` | All 9 `shadow_z[0..8]` registers (`0x024..0x044`) accept distinct 16-bit writes and read back cleanly. |
| `L5-AXIL-10` | `Set1` | `G1` | `test_axil_register_contract` | Control & status registers (`0x048`, `0x04C`, `0x050`, `0x054`, `0x058`) return `OKAY`. |
| `L5-AXIL-11` | `Set1` | `G1,G4` | `test_axil_register_contract` | All 18 `active_x/z` read-only registers (`0x100..0x144`) return committed Q7.8 weights with `RRESP=OKAY`. |
| `L5-AXIL-12` | `Set1` | `G1` | `test_axil_register_contract` | `WSTRB=0x3` writes lower 16 bits while `WSTRB=0xC` ignores upper 16 bits `[31:16]`. |
| `L5-AXIL-13` | `Set1` | `G1` | `test_axil_register_contract` | `WSTRB=0x1` updates byte `[7:0]` while preserving byte `[15:8]`. |
| `L5-AXIL-14` | `Set1` | `G1` | `test_axil_register_contract` | `WSTRB=0x2` updates byte `[15:8]` while preserving byte `[7:0]`. |
| `L5-AXIL-15` | `Set1` | `G1` | `test_axil_register_contract` | `WSTRB=0x0` leaves shadow registers unchanged and rejects `0x054`/`0x058` config writes with `SLVERR`. |
| `L5-AXIL-16` | `Set1` | `G1,G4` | `test_axil_register_contract` | Halfword & byte writes (`WSTRB=0x1, 0x2, 0x3`) to shadow banks rejected with `SLVERR` during pending commit. |
| `L5-AXIL-17` | `Set1` | `G1` | `test_axil_register_contract` | Unaligned write addresses (`+1, +2, +3`) return `SLVERR` (`2'b10`) with zero mutation. |
| `L5-AXIL-18` | `Set1` | `G1` | `test_axil_register_contract` | Unaligned read addresses (`+1, +2, +3`) return `RRESP=2'b10` (`SLVERR`) and `RDATA=0xDEADBEEF`. |
| `L5-AXIL-19` | `Set1` | `G1` | `test_axil_register_contract` | Unmapped gap offsets `0x05C..0x0FC` rejected on read and write with `SLVERR`. |
| `L5-AXIL-20` | `Set1` | `G1` | `test_axil_register_contract` | Upper unmapped offsets (`0x148`, `0x200`, `0xFFC`, `0xFFFFFFFC`) rejected with `SLVERR`. |
| `L5-AXIL-21` | `Set1` | `G1,G4` | `test_axil_register_contract` | Writes to read-only `STATUS` (`0x04C`) and `EPOCH_COUNT` (`0x050`) rejected with `SLVERR`. |
| `L5-AXIL-22` | `Set1` | `G1,G4` | `test_axil_register_contract` | Direct AXI4-Lite writes to all 18 `active_x/z` registers (`0x100..0x144`) return `SLVERR`. |
| `L5-AXIL-23` | `Set1` | `G1,G2` | `test_axil_register_contract` | `CFG_MAX_ITER` (`0x054`) accepts all supported $K \in \{1, 2, 3, 4, 5\}$ with exact readback. |
| `L5-AXIL-24` | `Set1` | `G1` | `test_axil_register_contract` | Out-of-range `CFG_MAX_ITER` values `{0, 6, 7, 8, 15, 255}` rejected with `SLVERR`. |
| `L5-AXIL-25` | `Set1` | `G1` | `test_axil_register_contract` | `CFG_MAX_ITER` rejects upper-bit aliasing patterns `{0x13, 0x103, 0xFFFF0003}` with `SLVERR`. |
| `L5-AXIL-26` | `Set1` | `G1,G2` | `test_axil_register_contract` | `OSD_MODE` (`0x058`) accepts `{0, 1, 2}` with exact readback. |
| `L5-AXIL-27` | `Set1` | `G1` | `test_axil_register_contract` | Out-of-range `OSD_MODE` values `{3, 4, 7, 16, 255, 0x101}` rejected with `SLVERR`. |
| `L5-AXIL-28` | `Set1` | `G1,G6` | `test_axil_register_contract` | Q7.8 non-negative boundaries `{0, +1, +562, +5120, +32767}` round-trip cleanly. |
| `L5-AXIL-29` | `Set1` | `G1,G6` | `test_axil_register_contract` | Q7.8 negative boundaries `{-32768, -5120, -1}` preserve two's-complement bits & sign. |
| `L5-AXIL-30` | `Set1` | `G1,G4` | `test_axil_register_contract` | `COMMIT` (`0x048`) treats `wdata=0` or `wstrb=0` as safe `OKAY` no-op and swaps strictly on `wdata[0]&wstrb[0]`. |

### Set 2 — Output, Deterministic Cycle Law & Correction Stream Contracts (`38` Tests)

| Canonical ID | Set | Gates | Executable / Script | Verification Objective |
| :--- | :---: | :---: | :--- | :--- |
| `L5-CYC-01` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=0` (Pure BP) at `K=1` executes in exact `7` core / `12` AXI cycles (`28` / `48 ns`). |
| `L5-CYC-02` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=0` (Pure BP) at `K=2` executes in exact `12` core / `17` AXI cycles (`48` / `68 ns`). |
| `L5-CYC-03` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=0` (Pure BP) at `K=3` executes in exact `17` core / `22` AXI cycles (`68` / `88 ns`). |
| `L5-CYC-04` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=0` (Pure BP) at `K=4` executes in exact `22` core / `27` AXI cycles (`88` / `108 ns`). |
| `L5-CYC-05` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=0` (Pure BP) at `K=5` executes in exact `27` core / `32` AXI cycles (`108` / `128 ns`). |
| `L5-CYC-06` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=1` (Inline OSD-0) at `K=1` executes in exact `7` core / `12` AXI cycles (`28` / `48 ns`). |
| `L5-CYC-07` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=1` (Inline OSD-0) at `K=2` executes in exact `12` core / `17` AXI cycles (`48` / `68 ns`). |
| `L5-CYC-08` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=1` (Inline OSD-0) at `K=3` executes in exact `17` core / `22` AXI cycles (`68` / `88 ns`). |
| `L5-CYC-09` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=1` (Inline OSD-0) at `K=4` executes in exact `22` core / `27` AXI cycles (`88` / `108 ns`). |
| `L5-CYC-10` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=1` (Inline OSD-0) at `K=5` executes in exact `27` core / `32` AXI cycles (`108` / `128 ns`). |
| `L5-CYC-11` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=2` (Post-BP OSD-0) at `K=1` executes in exact `9` core / `14` AXI cycles (`36` / `56 ns`). |
| `L5-CYC-12` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=2` (Post-BP OSD-0) at `K=2` executes in exact `14` core / `19` AXI cycles (`56` / `76 ns`). |
| `L5-CYC-13` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=2` (Post-BP OSD-0) at `K=3` executes in exact `19` core / `24` AXI cycles (`76` / `96 ns`). |
| `L5-CYC-14` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=2` (Post-BP OSD-0) at `K=4` executes in exact `24` core / `29` AXI cycles (`96` / `116 ns`). |
| `L5-CYC-15` | `Set2` | `G2` | `test_cycle_latency_law` | `OSD_MODE=2` (Post-BP OSD-0) at `K=5` executes in exact `29` core / `34` AXI cycles (`116` / `136 ns`). |
| `L5-CYC-16` | `Set2` | `G2` | `test_cycle_latency_law` | `K=3` sub-stage split: `2` ingress (`8 ns`) + `17` core (`68 ns`) + `3` egress (`12 ns`) = `22` cycles (`88 ns`). |
| `L5-CYC-17` | `Set2` | `G2` | `test_cycle_latency_law` | All 15 `(K=1..5 x OSD_MODE=0..2)` states satisfy `<= 300.0 ns` fast-loop budget (`max 136.0 ns`). |
| `L5-CYC-18` | `Set2` | `G2` | `test_cycle_latency_law` | Zero data-dependent cycle variation across all 256 `(sx, sz)` inputs (`17` core / `22` AXI cycles). |
| `L5-CYC-19` | `Set2` | `G2` | `test_cycle_latency_law` | Zero weight-dependent cycle variation across uniform, inhomogeneous, negative, and extreme Q7.8 priors. |
| `L5-CYC-20` | `Set2` | `G2` | `test_cycle_latency_law` | `bp_core_start_mon` asserts for strictly 1 clock cycle when RX FIFO pops syndrome into idle BP core. |
| `L5-CYC-21` | `Set2` | `G2` | `test_cycle_latency_law` | `bp_core_done_mon` pulses for strictly 1 cycle on core completion (`17`th cycle Mode 1, `19`th Mode 2). |
| `L5-CYC-22` | `Set2` | `G2` | `test_cycle_latency_law` | Parallel X-basis and Z-basis BP engines run in strict cycle-by-cycle lockstep across `K=1..5`. |
| `L5-O01` | `Set2` | `G2` | `test_correction_output_contract` | `m_axis_correction_tdata` packs `corr_x` in `[8:0]`, `corr_z` in `[17:9]`, and zeros `[31:18]`. |
| `L5-O02` | `Set2` | `G2,G6` | `test_correction_output_contract` | Zero-syndrome `(sx=0, sz=0)` maps strictly to `(0, 0)` across all 15 `(K, OSD_MODE)` states. |
| `L5-O03` | `Set2` | `G2,G6` | `test_correction_output_contract` | All 15 non-zero X-syndromes with `sz=0` produce `corr_x==0` and valid non-zero `corr_z`. |
| `L5-O04` | `Set2` | `G2,G6` | `test_correction_output_contract` | All 15 non-zero Z-syndromes with `sx=0` produce `corr_z==0` and valid non-zero `corr_x`. |
| `L5-O05` | `Set2` | `G2,G6` | `test_correction_output_contract` | All 9 single-qubit Z-errors `e_Z=(1<<q)` recovered uniquely under qubit-calibrated priors. |
| `L5-O06` | `Set2` | `G2,G6` | `test_correction_output_contract` | All 9 single-qubit X-errors `e_X=(1<<q)` recovered uniquely under qubit-calibrated priors. |
| `L5-O07` | `Set2` | `G2,G6` | `test_correction_output_contract` | Simultaneous Pauli-Y errors (`e_X = e_Z = 1<<q`) decoded in parallel across all 9 qubits. |
| `L5-O08` | `Set2` | `G2` | `test_correction_output_contract` | `m_axis_correction_tlast` matches `s_axis_syndrome_tlast` 1-to-1 across 16 mixed packets. |
| `L5-O09` | `Set2` | `G2` | `test_correction_output_contract` | `epoch_irq` asserts for strictly 1 clock cycle (`4.0 ns`) on correction handshake with `tlast=1`. |
| `L5-O10` | `Set2` | `G2` | `test_correction_output_contract` | `epoch_irq` remains `0` across 100 consecutive syndromes with `s_axis_syndrome_tlast=0`. |
| `L5-O11` | `Set2` | `G2,G5` | `test_correction_output_contract` | `epoch_irq` remains `0` while stalled by `m_axis_correction_tready=0`, firing once upon release. |
| `L5-O12` | `Set2` | `G2` | `test_correction_output_contract` | `m_axis_correction_tdata` and `tlast` drive `0` cleanly whenever `m_axis_correction_tvalid==0`. |
| `L5-O13` | `Set2` | `G2,G8` | `test_correction_output_contract` | 5 independent `Vaa_qec_top` instances produce bit-for-bit and cycle-for-cycle identical traces. |
| `L5-O14` | `Set2` | `G2,G8` | `test_correction_output_contract` | 5 consecutive `reset_model()` cycles inside one instance produce identical outputs & cycle deltas. |
| `L5-O15` | `Set2` | `G2,G8` | `test_correction_output_contract` | 256-round seeded syndrome replay is bit-exact for identical seeds (`42==42`) and distinct for `42!=108`. |
| `L5-O16` | `Set2` | `G2,G6` | `test_correction_output_contract` | `Vaa_qec_top` matches C++ golden reference across `OSD_MODE` `0, 1, 2` for all 16 orthogonal pairs. |

### Set 3 — Pipeline, Ingress Tap & Atomic Weight-Bank System Flow (`36` Tests)

| Canonical ID | Set | Gates | Executable / Script | Verification Objective |
| :--- | :---: | :---: | :--- | :--- |
| `L5-FLOW-01` | `Set3` | `G2,G3` | `test_ingress_tap_pipeline` | `axis_ingress_tap` adds `0` clock cycles to fast BP path (`22` total AXI cycles @ `K=3`). |
| `L5-FLOW-02` | `Set3` | `G3` | `test_ingress_tap_pipeline` | With `dma_tap_stall_in=0`, `dma_tap_tvalid_mon` mirrors accepted beats with `drop_count=0`. |
| `L5-FLOW-03` | `Set3` | `G3` | `test_ingress_tap_pipeline` | Asserting `dma_tap_stall_in=1` increments `dma_tap_drop_count_mon` by `+1` per accepted syndrome. |
| `L5-FLOW-04` | `Set3` | `G2,G3` | `test_ingress_tap_pipeline` | `dma_tap_stall_in=1` causes zero stall (`s_axis_tready=1`) and zero cycle increase (`22` cyc) on fast path. |
| `L5-FLOW-05` | `Set3` | `G3` | `test_ingress_tap_pipeline` | After DMA stall drop, next unstalled DMA beat asserts `overflow_warn` (`tdata[9]=1`) and clears after. |
| `L5-FLOW-06` | `Set3` | `G3` | `test_ingress_tap_pipeline` | Incoming `s_axis_tdata[9]=1` passes transparently through `dma_tap_tdata_mon[9]` when unstalled. |
| `L5-FLOW-07` | `Set3` | `G3,G5` | `test_ingress_tap_pipeline` | 256-syndrome stream under permanent `dma_tap_stall_in=1` delivers `256/256` corrections (`drops=256`). |
| `L5-FLOW-08` | `Set3` | `G3` | `test_ingress_tap_pipeline` | Alternating DMA stall pattern (`3` stalled, `2` unstalled $\times 20$) counts exact `60` drops. |
| `L5-FLOW-09` | `Set3` | `G3,G5` | `test_ingress_tap_pipeline` | `aresetn=0` resets `dma_tap_drop_count_mon=0` and clears `sticky_overflow`. |
| `L5-FLOW-10` | `Set3` | `G3,G5` | `test_ingress_tap_pipeline` | When RX FIFO is full (`s_axis_tready=0`), `dma_tap_stall_in=1` does not over-count unaccepted beats. |
| `L5-FLOW-11` | `Set3` | `G2,G3` | `test_ingress_tap_pipeline` | `4x64`-round streaming window triggers exactly `4` `epoch_irq` pulses and `rx_epoch_count_mon==4`. |
| `L5-FLOW-12` | `Set3` | `G2,G3` | `test_ingress_tap_pipeline` | Runtime `CFG_MAX_ITER` transitions `K=1->2->3->4->5->3` switch core latency cleanly without reset. |
| `L5-FLOW-13` | `Set3` | `G2,G3` | `test_ingress_tap_pipeline` | Runtime `OSD_MODE` transitions `0->1->2->0->1` switch between `17`-cycle and `19`-cycle paths cleanly. |
| `L5-FLOW-14` | `Set3` | `G2,G3` | `test_ingress_tap_pipeline` | Writing `0x054` (`K=5`) and `0x058` (`Mode=2`) mid-decode does not alter in-flight `K=3 Mode=1` decode. |
| `L5-FLOW-15` | `Set3` | `G3,G5` | `test_ingress_tap_pipeline` | 12 back-to-back queued syndromes drain continuously through `bp_engine` without loss or reordering. |
| `L5-FLOW-16` | `Set3` | `G1,G3` | `test_ingress_tap_pipeline` | Continuous AXI4-Lite `STATUS`/`EPOCH` polling during active AXI4-Stream decoding causes zero interference. |
| `L5-FLOW-17` | `Set3` | `G4` | `test_weight_bank_flow` | Writing all 18 shadow registers without `COMMIT` leaves `active_x/z` and decoded corrections unchanged. |
| `L5-FLOW-18` | `Set3` | `G4` | `test_weight_bank_flow` | Idle `COMMIT` (`0x048=1`) updates all 18 `active_x/z` weights and increments `epoch_count` `0->1`. |
| `L5-FLOW-19` | `Set3` | `G4` | `test_weight_bank_flow` | Updating only `shadow_z[4]` + `COMMIT` updates `active_z[4]` while keeping other 17 weights at `0x0232`. |
| `L5-FLOW-20` | `Set3` | `G4,G6` | `test_weight_bank_flow` | Multi-epoch commits (`0->1->2->3->4`) steer degenerate `sx=0x2` across `q2`, `q5`, `{q0,q1}`, `{q3,q4}`. |
| `L5-FLOW-21` | `Set3` | `G4` | `test_weight_bank_flow` | Committing `shadow_x` never perturbs `active_z` and committing `shadow_z` never perturbs `active_x`. |
| `L5-FLOW-22` | `Set3` | `G4` | `test_weight_bank_flow` | `COMMIT` issued on cycle 1 of core (`ST_INIT`) defers swap until decode finishes with old bank. |
| `L5-FLOW-23` | `Set3` | `G4` | `test_weight_bank_flow` | `COMMIT` issued mid-iteration (`ST_WAIT_C2V`, cycle 8) defers cleanly until `bp_ready_in==1`. |
| `L5-FLOW-24` | `Set3` | `G4` | `test_weight_bank_flow` | `COMMIT` arriving on final core cycle (`ST_HARD_DECIDE`) does not corrupt combinational OSD-0 decision. |
| `L5-FLOW-25` | `Set3` | `G1,G4` | `test_weight_bank_flow` | All 9 `shadow_x` offsets reject writes with `SLVERR` while `swap_pending==1 && !bp_ready_in`. |
| `L5-FLOW-26` | `Set3` | `G1,G4` | `test_weight_bank_flow` | All 9 `shadow_z` offsets reject writes with `SLVERR` while `swap_pending==1 && !bp_ready_in`. |
| `L5-FLOW-27` | `Set3` | `G1,G4` | `test_weight_bank_flow` | Duplicate `COMMIT` write while `swap_pending==1` returns `SLVERR` and increments `epoch_count` by `+1` only. |
| `L5-FLOW-28` | `Set3` | `G4` | `test_weight_bank_flow` | Shadow writes succeed (`OKAY`) while decoder is busy when `swap_pending==0`. |
| `L5-FLOW-29` | `Set3` | `G1,G4` | `test_weight_bank_flow` | Read-only `STATUS`, `EPOCH_COUNT`, and `active_x/z` remain readable (`OKAY`) while `swap_pending==1`. |
| `L5-FLOW-30` | `Set3` | `G1,G4` | `test_weight_bank_flow` | Reading `COMMIT` (`0x048`) returns `1` while deferred mid-decode and `0` once swapped. |
| `L5-FLOW-31` | `Set3` | `G1,G4` | `test_weight_bank_flow` | `STATUS` (`0x04C`) encodes `[0]=swap_pending, [1]=bp_busy` across `0x0`, `0x2`, and `0x3`. |
| `L5-FLOW-32` | `Set3` | `G3,G4` | `test_weight_bank_flow` | Deferred `COMMIT` swaps atomically on the 1-cycle `ST_IDLE` gap between two queued syndromes. |
| `L5-FLOW-33` | `Set3` | `G4` | `test_weight_bank_flow` | 32 sequential adaptive weight commits track shifting hotspots cleanly to `epoch_count==32`. |
| `L5-FLOW-34` | `Set3` | `G4,G5` | `test_weight_bank_flow` | Asserting `aresetn=0` during pending mid-decode swap clears `swap_pending=0` and restores `0x0232`. |
| `L5-FLOW-35` | `Set3` | `G4` | `test_weight_bank_flow` | Shadow bank unlocks immediately once deferred swap completes, accepting new writes with `OKAY`. |
| `L5-FLOW-36` | `Set3` | `G4,G8` | `test_weight_bank_flow` | Epoch accounting invariant holds across boot commits, adaptive commits, rejected commits, and resets. |

### Set 4 — Execution, Backpressure, Reset, Stress & Artifact Persistence (`40` Tests)

| Canonical ID | Set | Gates | Executable / Script | Verification Objective |
| :--- | :---: | :---: | :--- | :--- |
| `L5-EXEC-01` | `Set4` | `G5` | `test_reset_and_backpressure` | Power-on reset defaults verified across all 41 AXI4-Lite registers and top-level outputs. |
| `L5-EXEC-02` | `Set4` | `G5` | `test_reset_and_backpressure` | Short (`2`-cycle) and extended (`32`-cycle) synchronous resets restore identical state and latency. |
| `L5-EXEC-03` | `Set4` | `G5` | `test_reset_and_backpressure` | Synchronous mid-decode `aresetn=0` in `ST_INIT` aborts cleanly and recovers `17`-cycle decode. |
| `L5-EXEC-04` | `Set4` | `G5` | `test_reset_and_backpressure` | Synchronous mid-decode `aresetn=0` in `ST_V2C` aborts cleanly with zero ghost outputs. |
| `L5-EXEC-05` | `Set4` | `G5` | `test_reset_and_backpressure` | Synchronous mid-decode `aresetn=0` across all 3 `ST_WAIT_C2V` `check_node_minsum` stages flushes cleanly. |
| `L5-EXEC-06` | `Set4` | `G5` | `test_reset_and_backpressure` | Synchronous mid-decode `aresetn=0` in `ST_BELIEF_UPD` clears beliefs and recovers cleanly. |
| `L5-EXEC-07` | `Set4` | `G5` | `test_reset_and_backpressure` | Synchronous mid-decode `aresetn=0` in `ST_HARD_DECIDE` suppresses `correction_valid` and recovers. |
| `L5-EXEC-08` | `Set4` | `G5` | `test_reset_and_backpressure` | Synchronous mid-decode `aresetn=0` in `ST_OSD_STEP1` and `ST_OSD_STEP2` (`Mode 2`) aborts and restores `Mode 1`. |
| `L5-EXEC-09` | `Set4` | `G1,G5` | `test_reset_and_backpressure` | Asserting `aresetn=0` mid-AXI4-Lite split `AW`/`W` clears latches without lockup. |
| `L5-EXEC-10` | `Set4` | `G2,G5` | `test_reset_and_backpressure` | `m_axis_correction_tready=0` holds `tvalid=1`, `tdata`, and `tlast` stable across 32 stalled cycles. |
| `L5-EXEC-11` | `Set4` | `G5` | `test_reset_and_backpressure` | 8-packet burst stalled at egress drains in exact FIFO order under pulsed `m_axis_correction_tready`. |
| `L5-EXEC-12` | `Set4` | `G3,G5` | `test_reset_and_backpressure` | Filling `axis_correction_tx` to 14-entry watermark under `tready=0` backpressures BP core with `0` drops. |
| `L5-EXEC-13` | `Set4` | `G1,G5` | `test_reset_and_backpressure` | Filling 16-entry `axis_syndrome_rx` FIFO (`1` active in `K=5` BP + `16` queued) drives `s_axis_syndrome_tready=0`. |
| `L5-EXEC-14` | `Set4` | `G1,G5` | `test_reset_and_backpressure` | Full 16-entry RX FIFO blocks unready beats without overwrite and drains all 17 accepted packets losslessly. |
| `L5-EXEC-15` | `Set4` | `G5` | `test_reset_and_backpressure` | 4 consecutive 16-beat RX FIFO bursts wrap `wr_ptr`/`rd_ptr` cleanly across 64 lossless decodes. |
| `L5-EXEC-16` | `Set4` | `G3,G5` | `test_reset_and_backpressure` | 3-way pseudo-random `s_axis_tvalid`, `m_axis_tready`, and `dma_tap_stall_in` stress delivers `256/256` decodes. |
| `L5-EXEC-17` | `Set4` | `G5` | `test_reset_and_backpressure` | `aresetn=0` with non-empty RX and TX FIFOs flushes both `wr_ptr`/`rd_ptr=0` with zero phantom beats. |
| `L5-EXEC-18` | `Set4` | `G1,G5` | `test_reset_and_backpressure` | 64 invalid AXI4-Lite `SLVERR` writes injected mid-decode cause zero perturbation to 64 fast-path decodes. |
| `L5-EXEC-19` | `Set4` | `G5,G8` | `test_reset_and_backpressure` | Native `VerilatedVcdC` waveform dump captures `aa_qec_top` hierarchy, clocks, and signal transitions. |
| `L5-EXEC-20` | `Set4` | `G5,G8` | `test_reset_and_backpressure` | Compiled binary links directly against native Verilator `Vaa_qec_top` (`verilator_c++_rtl`). |
| `L5-EXEC-21` | `Set4` | `G4,G5` | `test_reset_and_backpressure` | Synchronous reset clears simultaneous `swap_pending=1` shadow lock and full RX FIFO cleanly. |
| `L5-EXEC-22` | `Set4` | `G2,G5` | `test_reset_and_backpressure` | Buffered corrections in `axis_correction_tx` stream out at 1 beat/cycle (`250 Mwords/s` peak) upon `tready=1`. |
| `L5-FLT-01` | `Set4` | `G5,G6,G8` | `test_streaming_stress` | 10,000-syndrome continuous streaming stress (`> 220,000` RTL cycles) completes with `0` mismatches & `0` deadlocks. |
| `L5-FLT-02` | `Set4` | `G4,G5,G6` | `test_streaming_stress` | 2,000-syndrome stream with 20 live adaptive commits (including mid-decode deferred swaps) has `0` mismatches. |
| `L5-FLT-03` | `Set4` | `G2,G5,G6` | `test_streaming_stress` | 1,000-syndrome worst-case `K=5 OSD_MODE=2` stream holds exact `29`-cycle core / `34`-cycle AXI latency. |
| `L5-FLT-04` | `Set4` | `G2,G5,G6` | `test_streaming_stress` | 1,000-syndrome minimum-iteration `K=1 OSD_MODE=0` stream holds exact `7`-cycle core / `12`-cycle AXI latency. |
| `L5-FLT-05` | `Set4` | `G3,G5,G6` | `test_streaming_stress` | 1,000-syndrome pipelined stream under periodic egress stalls & DMA tap drops preserves 100% accuracy. |
| `L5-FLT-06` | `Set4` | `G5,G6` | `test_streaming_stress` | 512-syndrome stream with mixed negative/positive Q7.8 priors matches golden reference bit-for-bit. |
| `L5-FLT-07` | `Set4` | `G2,G3,G5` | `test_streaming_stress` | 64 variable-length epochs (`{1, 4, 8, 16}` rounds) assert `epoch_irq` strictly 64 times. |
| `L5-FLT-08` | `Set4` | `G4,G5,G7` | `test_streaming_stress` | Cycle-by-cycle `COMMIT` race sweep across `dt=0..24` cycles proves `0` SVA violations & `0` torn decodes. |
| `L5-FLT-09` | `Set4` | `G5` | `test_streaming_stress` | Cycle-by-cycle `aresetn` race sweep across `dt=0..24` cycles recovers cleanly on every clock phase. |
| `L5-FLT-10` | `Set4` | `G5,G8` | `test_streaming_stress` | 10 consecutive `Vaa_qec_top` construct/stream/destruct lifecycles produce identical cycle & output hashes. |
| `L5-FLT-11` | `Set4` | `G5,G8` | `test_artifact_persistence.py` | Persisted `m1_standalone.json` confirms native Verilator RTL, `17`/`22` cycles, and `11/11` safety flags. |
| `L5-FLT-12` | `Set4` | `G2,G8` | `test_artifact_persistence.py` | Persisted `rtl_cycle_matrix.csv` matches `architecture/fpga_decoder/rtl_cycle_matrix.csv` across `K=1..5`. |
| `L5-FLT-13` | `Set4` | `G5,G8` | `test_artifact_persistence.py` | Persisted `standalone_rtl_tb.vcd` contains IEEE-1364 hierarchy and multi-hundred-edge Verilator transitions. |
| `L5-FLT-14` | `Set4` | `G7,G8` | `test_artifact_persistence.py` | SHA-256 `rtl_build_provenance.json` matches all 10 RTL/SVA/XDC/driver source files & `Verilator 5.048`. |
| `L5-FLT-15` | `Set4` | `G2,G8` | `test_artifact_persistence.py` | `pytest tests/test_layer5_ip_op.py` passes against compiled native Verilator `standalone_rtl_tb`. |
| `L5-FLT-16` | `Set4` | `G7,G8` | `test_artifact_persistence.py` | All 4 SystemVerilog testbenches instantiate current RTL modules with complete ports and zero stale names. |
| `L5-FLT-17` | `Set4` | `G8` | `test_artifact_persistence.py` | RTL regression YAML configs specify `simulator=verilator`, `top=aa_qec_top`, `4ns` clock, `K=1..5`, `assertions=true`. |
| `L5-FLT-18` | `Set4` | `G8` | `test_artifact_persistence.py` | All structured JSONL records conform to required audit schema with unique canonical IDs and `100% PASS`. |

### Set 5 — Hardware-Specific Q7.8 Min-Sum, OSD-0, SVA & Constraints Verification (`42` Tests)

| Canonical ID | Set | Gates | Executable / Script | Verification Objective |
| :--- | :---: | :---: | :--- | :--- |
| `L5-MATH-01` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Exhaustive 256-syndrome `(sx, sz)` sweep under uniform `0x0232` priors has `0` mismatches vs golden Q7.8. |
| `L5-MATH-02` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Exhaustive 256-syndrome sweep under inhomogeneous asymmetric `wx != wz` weights has `0` golden mismatches. |
| `L5-MATH-03` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Exhaustive 256-syndrome sweep in Pure 9-Marginal BP (`OSD_MODE=0, K=3`) satisfies `256/256` syndromes. |
| `L5-MATH-04` | `Set5` | `G6` | `test_q78_minsum_osd_math` | `OSD_MODE=2` (`19`-cyc Post-BP OSD-0) matches `OSD_MODE=1` (`17`-cyc Inline OSD-0) & golden across 256 syndromes. |
| `L5-MATH-05` | `Set5` | `G2,G6` | `test_q78_minsum_osd_math` | Exhaustive 3,840-state (`K=1..5 x OSD_MODE=0..2 x 256`) sweep achieves `0` golden mismatches & `3,584/3,584` converged GF(2) parity. |
| `L5-MATH-06` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Extreme high-SNR Q7.8 weights (`0x06E8..0x3FFF`) match golden across all 256 syndromes without overflow. |
| `L5-MATH-07` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Extreme low-SNR Q7.8 weights (`0x0008..0x00D9`) match golden & satisfy all 256 syndromes. |
| `L5-MATH-08` | `Set5` | `G6` | `test_q78_minsum_osd_math` | All 7 golden Q7.8 reference vectors from `llr_numeric_contract.yaml` verified analytically and in RTL. |
| `L5-MATH-09` | `Set5` | `G6` | `test_q78_minsum_osd_math` | GF(2) parity `Hx*corr_z^T==sx` and `Hz*corr_x^T==sz` holds across 1,024 syndrome-profile combinations. |
| `L5-MATH-10` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Degenerate `sz=0x8` (`{q7,q8}` tie) broken deterministically across `q7`, `q8`, `{q1,q4}`, and `{q2,q5}` in Mode 0 & 1. |
| `L5-MATH-11` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Degenerate `sz=0x1` steered cleanly across all 4 coset solutions (`q0`, `q1`, `{q3,q6}`, `{q4,q7}`). |
| `L5-MATH-12` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Degenerate `sx=0x2` steered cleanly across all 4 coset solutions (`q2`, `q5`, `{q0,q1}`, `{q3,q4}`). |
| `L5-MATH-13` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Degenerate `sx=0x4` steered cleanly across all 4 coset solutions (`q3`, `q6`, `{q4,q5}`, `{q7,q8}`). |
| `L5-MATH-14` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Boundary syndromes `sx=0x1` and `sx=0x8` switch cleanly between weight-1 boundary and weight-2 pair solutions. |
| `L5-MATH-15` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Boundary syndromes `sz=0x2` and `sz=0x4` switch cleanly between weight-1 boundary and weight-2 pair solutions. |
| `L5-MATH-16` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Center bulk qubit `q4` syndromes (`sx=0x6, sz=0x9`) recover `0x010` and switch `sx=0x6` to `{q3,q5}=0x028` under anti-`q4` priors. |
| `L5-MATH-17` | `Set5` | `G6` | `test_q78_minsum_osd_math` | All 6 horizontal Z-error pairs `{(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)}` recovered uniquely under pair hotspot priors. |
| `L5-MATH-18` | `Set5` | `G6` | `test_q78_minsum_osd_math` | All 6 vertical X-error pairs `{(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)}` recovered uniquely under pair hotspot priors. |
| `L5-MATH-19` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Check-node Min-Sum shift-add scaling (`0.75` non-min / `min2+0.5*min1` min1) matches RTL across `K=1..5` in Mode 0. |
| `L5-MATH-20` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Residual error `(e XOR corr)` lies strictly in `kernel(H)` across all 45 weight-1 & weight-2 Pauli errors. |
| `L5-MATH-21` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Full-scale 16-bit Q7.8 extremes (`0x7FFF` and `0x8000`) execute cleanly in Mode 0 & Mode 1 matching golden. |
| `L5-MATH-22` | `Set5` | `G6` | `test_q78_minsum_osd_math` | CSS commutation `H_X * H_Z^T == 0 (mod 2)` verified across all 16 stabilizer pairs & 24 Tanner edges. |
| `L5-MATH-23` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Composite weight-3/weight-4 syndromes `{0x7, 0xB, 0xD, 0xE, 0xF}` satisfy GF(2) & match golden across 5 profiles. |
| `L5-MATH-24` | `Set5` | `G6` | `test_q78_minsum_osd_math` | Mutating `active_x` leaves `corr_z` 100% invariant (and vice versa) across all 256 `(sx, sz)` syndromes. |
| `L5-HW-01` | `Set5` | `G7` | `test_sva_and_constraints.py` | `sva_weight_commit_assertions` is bound to `weight_map_reg` and compiled into `Vaa_qec_top` C++ model. |
| `L5-HW-02` | `Set5` | `G7` | `test_sva_and_constraints.py` | Verilator `--assert` SVA `p_no_swap_while_busy` passes across legal idle & boundary bank swaps. |
| `L5-HW-03` | `Set5` | `G7` | `test_sva_and_constraints.py` | Negative fault injection: mid-decode `bank_sel` toggle while `bp_ready==0` actively fires SVA error. |
| `L5-HW-04` | `Set5` | `G5,G7` | `test_sva_and_constraints.py` | SVA `disable iff (!aresetn)` cleanly masks `bank_sel` reset transitions while `aresetn==0`. |
| `L5-HW-05` | `Set5` | `G4,G7` | `test_sva_and_constraints.py` | `weight_map_reg.sv` embeds 5 formal SVA invariants guarding `active_x/z` and `commit_snap_x/z`. |
| `L5-HW-06` | `Set5` | `G2,G7` | `test_sva_and_constraints.py` | `bp_engine.sv` embeds 3 formal cycle-count SVAs enforcing `2+5K` (`Modes 0/1`) and `4+5K` (`Mode 2`). |
| `L5-HW-07` | `Set5` | `G7` | `test_sva_and_constraints.py` | `verilator --lint-only --assert` passes with zero errors across all 8 RTL + bound SVA files. |
| `L5-HW-08` | `Set5` | `G7` | `test_sva_and_constraints.py` | All 7 RTL modules strictly use synthesizable `always_ff`/`always_comb` with zero latches or `#delays`. |
| `L5-HW-09` | `Set5` | `G7` | `test_sva_and_constraints.py` | Single-clock synchronous 250 MHz architecture verified (zero async `negedge` CDC hazards). |
| `L5-HW-10` | `Set5` | `G7` | `test_sva_and_constraints.py` | `firmware/constraints/aa_qec.xdc` defines active `4.000 ns` (`250.0 MHz`) `create_clock` on port `aclk`. |
| `L5-HW-11` | `Set5` | `G1,G7` | `test_sva_and_constraints.py` | `architecture/hardware/memory_map.yaml` matches `weight_map_reg.sv` across all 41 register offsets. |
| `L5-HW-12` | `Set5` | `G4,G7` | `test_sva_and_constraints.py` | `weight_commit_protocol.yaml` matches triple-bank `snapshot->active` swap & `SLVERR` busy-lock RTL. |
| `L5-HW-13` | `Set5` | `G2,G7` | `test_sva_and_constraints.py` | `system_timing.yaml` & `cycle_contract.md` match measured `2+5K`/`4+5K` core and `7+5K`/`9+5K` AXI cycle laws. |
| `L5-HW-14` | `Set5` | `G1,G6,G7` | `test_sva_and_constraints.py` | `syndrome_packet_v1.yaml` & `llr_numeric_contract.yaml` match 16-bit ingress slicing & signed Q7.8 RTL. |
| `L5-HW-15` | `Set5` | `G7` | `test_sva_and_constraints.py` | `resource_budget.yaml` & `resource_envelope.md` confirm `<5%` LUT, `<2%` FF, and `0` DSP48E1 on Zynq-7020. |
| `L5-HW-16` | `Set5` | `G7` | `test_sva_and_constraints.py` | `check_node_minsum.sv` & `bp_engine.sv` implement scaling purely via bit-shifts (zero DSP multipliers). |
| `L5-HW-17` | `Set5` | `G7,G8` | `test_sva_and_constraints.py` | `target_platforms.yaml` declares Zynq-7020 V1 `Verilator 5.048` target & UltraScale+ ZCU104/K26 V2 target. |
| `L5-HW-18` | `Set5` | `G7,G8` | `test_sva_and_constraints.py` | Explicit disclosure verified: Verilator proves cycle-exact RTL while post-route 250 MHz WNS/TNS requires Vivado STA. |
