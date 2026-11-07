# AVID Synthesizable SystemVerilog RTL (`firmware/`)

- `rtl/top/aa_qec_top.sv`: Top-level decoder integrating AXI4-Stream syndrome ingress, ping-pong Q7.8 weight bank, Min-Sum BP + OSD-0 core, and AXI4-Stream correction egress.
- `rtl/config/weight_map_reg.sv`: Dual-bank Q7.8 register file (`0x000..0x058`) with `SLVERR` invalid-offset rejection and `bp_ready_in` CDC commit lock.
- `rtl/decoder/bp_engine.sv` & `check_node_minsum.sv`: Deterministic $K=1..5$ Min-Sum BP + OSD-0 engine (`2+5K` core cycles for `cfg_osd_mode in {0,1}`, `4+5K` core cycles for `cfg_osd_mode=2`).
- `rtl/ingress/`: `axis_syndrome_rx.sv` and `axis_ingress_tap.sv`.
- `constraints/aa_qec.xdc`: `250 MHz` (`4.0 ns`) timing constraints.
- `tb/` & `formal/`: SystemVerilog testbenches and formal SVA properties.
