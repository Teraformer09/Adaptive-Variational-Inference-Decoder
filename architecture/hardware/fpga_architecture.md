# FPGA Decoder Datapath & Register Architecture

1. **Ingress (`axis_syndrome_rx.sv` & `axis_ingress_tap.sv`)**: Accepts 64-bit syndrome packets (`3` cycles ingress pipeline).
2. **Ping-Pong Weight Bank (`weight_map_reg.sv`)**: Dual 18-register signed Q7.8 banks (`Bank 0` and `Bank 1`) with atomic `commit_pending` lock gated by `bp_ready_in`.
3. **Min-Sum BP + OSD-0 Core (`bp_engine.sv` & `check_node_minsum.sv`)**:
   - Parallel check-node (`ST_CHECK`, 2 cycles/iteration) and variable-node (`ST_VAR`, 3 cycles/iteration) message passing over the corrected rotated $d=3$ Tanner graph (weight-2 boundary checks `c0, c3` and weight-4 bulk checks `c1, c2`).
   - Configurable OSD mode via `0x058` (`cfg_osd_mode`):
     - `0`: Pure 9-marginal-LLR BP hard decision (`2 + 5K = 17` core cycles at $K=3$).
     - `1`: Inline pair-aware OSD-0 disambiguation inside `ST_DECIDE` (`2 + 5K = 17` core cycles at $K=3$).
     - `2`: Dedicated post-BP OSD-0 stage `ST_OSD_POST` (`4 + 5K = 19` core cycles at $K=3$).
4. **Egress (`axis_correction_tx.sv`)**: Formats and transmits 9-bit $X/Z$ corrections (`2` cycles egress pipeline).
